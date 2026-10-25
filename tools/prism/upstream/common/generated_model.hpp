// Production user-model initialization and K-cache APIs. Rust owns metadata,
// generated weights, bias inputs, inference schedules and every assertion.
#pragma once
#include "llama.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "kv-mean-center.h"
#include "llama-kv-cache.h"
#include <nlohmann/json.hpp>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace prism_generated_model {
using json=nlohmann::ordered_json;
inline void require(bool value,const char*message){if(!value)throw std::runtime_error(message);}
inline void fields(const json&v,std::initializer_list<const char*>names){require(v.is_object()&&v.size()==names.size(),"generated input fields mismatch");for(const auto*n:names)require(v.contains(n),"generated input field missing");}
inline std::uint64_t number(const json&v,std::uint64_t bound){require(v.is_number_integer()&&(v.is_number_unsigned()||v.get<std::int64_t>()>=0),"generated unsigned integer required");auto n=v.get<std::uint64_t>();require(n<=bound,"generated integer exceeds bound");return n;}
inline std::string text(const json&v,std::size_t bound){require(v.is_string(),"generated input text required");auto s=v.get<std::string>();require(s.size()<=bound&&s.find('\0')==std::string::npos,"generated input text extent/NUL violation");return s;}
inline std::filesystem::path owned_file(const json&value){auto p=std::filesystem::u8path(text(value,32768));require(p.is_absolute(),"generated input file must be absolute");auto actual=std::filesystem::canonical(p);require(std::filesystem::is_regular_file(actual)&&std::filesystem::equivalent(actual.parent_path(),std::filesystem::current_path()),"generated input file must be directly inside session directory");return actual;}
inline std::string name(const json&v){auto s=text(v,128);require(!s.empty()&&s.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")==std::string::npos&&s.find("..") == std::string::npos,"generated output requires owned plain filename");return s;}
inline std::vector<std::uint8_t> read(const std::filesystem::path&p,std::size_t size){require(size<=64*1024*1024&&std::filesystem::file_size(p)==size,"generated input file size differs from descriptor");std::ifstream input(p,std::ios::binary);require(static_cast<bool>(input),"cannot open generated input");std::vector<std::uint8_t>data(size);input.read(reinterpret_cast<char*>(data.data()),static_cast<std::streamsize>(size));require(input.gcount()==static_cast<std::streamsize>(size)&&input.peek()==std::char_traits<char>::eof(),"generated input truncated/grew");return data;}
inline std::string hex(const void*pointer,std::size_t size){require(size<=8*1024*1024,"generated observation exceeds8MiB");constexpr char digits[]="0123456789abcdef";auto*p=static_cast<const std::uint8_t*>(pointer);std::string value(size*2,'\0');for(std::size_t i=0;i<size;++i){value[2*i]=digits[p[i]>>4];value[2*i+1]=digits[p[i]&15];}return value;}
inline std::vector<std::uint8_t> unhex(const json&v,std::size_t bound){auto s=text(v,bound*2);require(s.size()%2==0,"generated input hex extent invalid");std::vector<std::uint8_t>value(s.size()/2);auto digit=[](char c)->std::uint8_t{if(c>='0'&&c<='9')return static_cast<std::uint8_t>(c-'0');if(c>='a'&&c<='f')return static_cast<std::uint8_t>(c-'a'+10);throw std::runtime_error("invalid generated input hex digit");};for(std::size_t i=0;i<value.size();++i)value[i]=static_cast<std::uint8_t>(16*digit(s[2*i])+digit(s[2*i+1]));return value;}
inline json tensor_spec(const ggml_tensor*tensor){json dimensions=json::array();for(const auto extent:tensor->ne){require(extent>0,"production tensor has invalid extent");dimensions.push_back(extent);}return json{{"Name",tensor->name},{"Type",static_cast<int>(tensor->type)},{"Dimensions",dimensions},{"Bytes",ggml_nbytes(tensor)}};}
inline void fill(ggml_tensor*tensor,const json&reply){
    fields(reply,{"DataFile","Length","Encoding"});auto bytes=number(reply.at("Length"),64*1024*1024);auto encoding=text(reply.at("Encoding"),32);auto file=owned_file(reply.at("DataFile"));const auto elements=ggml_nelements(tensor);require(elements>0&&ggml_nbytes(tensor)<=64*1024*1024,"production generated tensor exceeds bound");
    if(encoding=="tensor-le"){require(bytes==ggml_nbytes(tensor),"generated tensor byte extent differs");auto data=read(file,static_cast<std::size_t>(bytes));ggml_backend_tensor_set(tensor,data.data(),0,data.size());return;}
    require(encoding=="float32-le"&&(tensor->type==GGML_TYPE_F16||tensor->type==GGML_TYPE_F32)&&static_cast<std::uint64_t>(elements)<=16*1024*1024&&bytes==static_cast<std::uint64_t>(elements)*sizeof(float),"unsupported generated float conversion/extent");auto data=read(file,static_cast<std::size_t>(bytes));
    if(tensor->type==GGML_TYPE_F32){ggml_backend_tensor_set(tensor,data.data(),0,data.size());return;}
    std::vector<ggml_fp16_t>half(static_cast<std::size_t>(elements));for(std::size_t i=0;i<half.size();++i){float value;std::memcpy(&value,data.data()+i*sizeof(float),sizeof(float));half[i]=ggml_fp32_to_fp16(value);}ggml_backend_tensor_set(tensor,half.data(),0,half.size()*sizeof(ggml_fp16_t));
}
struct initialized{llama_model*value;std::shared_ptr<gguf_context>metadata;};
inline initialized initialize(const json&metadata_file,llama_model_params parameters,const std::function<json(const json&)>&callback){
    auto path=owned_file(metadata_file);require(std::filesystem::file_size(path)<=1024*1024,"generated metadata exceeds1MiB");gguf_init_params gp{};gp.no_alloc=true;std::shared_ptr<gguf_context>metadata(gguf_init_from_file(path.string().c_str(),gp),gguf_free);require(metadata!=nullptr,"production GGUF reader rejected generated metadata");
    struct state{const std::function<json(const json&)>*callback;std::mutex mutex;std::exception_ptr error;std::atomic<bool>failed{false};}state{&callback};
    parameters.progress_callback=[](float,void*opaque){return !static_cast<struct state*>(opaque)->failed.load();};parameters.progress_callback_user_data=&state;
    auto setter=[](ggml_tensor*tensor,void*opaque){auto&s=*static_cast<struct state*>(opaque);if(s.failed.load())return;try{fill(tensor,(*s.callback)(tensor_spec(tensor)));}catch(...){std::lock_guard<std::mutex>guard(s.mutex);if(!s.error)s.error=std::current_exception();s.failed=true;}};
    std::unique_ptr<llama_model,decltype(&llama_model_free)>model(llama_model_init_from_user(metadata.get(),setter,&state,parameters),llama_model_free);
    if(state.error)std::rethrow_exception(state.error);return initialized{model.release(),std::move(metadata)};
}
inline initialized initialize_metadata(const json&metadata_file,llama_model_params parameters){
    auto file=owned_file(metadata_file);require(std::filesystem::file_size(file)<=64*1024*1024,"generated metadata exceeds64MiB");gguf_init_params gp{};gp.no_alloc=true;std::shared_ptr<gguf_context>metadata(gguf_init_from_file(file.string().c_str(),gp),gguf_free);require(metadata!=nullptr,"production GGUF reader rejected generated metadata");
    parameters.no_alloc=true;return initialized{llama_model_init_from_user(metadata.get(),[](ggml_tensor*,void*){},nullptr,parameters),std::move(metadata)};
}
inline bool handles(const std::string&op){return op=="kv-center-write"||op=="kv-center-load"||op=="context-logits"||op=="context-metadata";}
inline json execute(const json&operation,llama_context*context){
    const auto op=operation.at("Op").get<std::string>();
    if(op=="kv-center-write"){
        fields(operation,{"Op","Name","Rotated","Layers"});auto filename=name(operation.at("Name"));require(operation.at("Rotated").is_boolean(),"KV bias basis must be boolean");const auto&layers=operation.at("Layers");require(layers.is_array()&&layers.size()<=256,"KV bias layers exceed256");std::vector<common_kv_mean_center_layer>values;
        for(const auto&layer:layers){fields(layer,{"Index","BiasHex"});common_kv_mean_center_layer value;value.il=static_cast<std::int32_t>(number(layer.at("Index"),255));auto data=unhex(layer.at("BiasHex"),1024*1024);require(data.size()%sizeof(float)==0,"KV bias float extent differs");value.bias.resize(data.size()/sizeof(float));std::memcpy(value.bias.data(),data.data(),data.size());values.push_back(std::move(value));}return json{{"Written",common_kv_mean_center_write(filename,values,operation.at("Rotated").get<bool>())}};
    }
    require(context!=nullptr,"generated context operation requires production context");
    if(op=="context-metadata"){fields(operation,{"Op","Context"});auto*model=llama_get_model(context);return json{{"ContextSize",llama_n_ctx(context)},{"RecurrentSnapshots",llama_n_rs_seq(context)},{"Recurrent",llama_model_is_recurrent(model)},{"Hybrid",llama_model_is_hybrid(model)},{"Vocabulary",llama_vocab_n_tokens(llama_model_get_vocab(model))}};}
    if(op=="kv-center-load"){fields(operation,{"Op","Context","Name","RequireQ4"});auto filename=name(operation.at("Name"));require(operation.at("RequireQ4").is_boolean(),"KV bias gate flag must be boolean");auto*cache=dynamic_cast<llama_kv_cache*>(llama_get_memory(context));return json{{"CacheExists",cache!=nullptr},{"Loaded",cache?json(cache->load_kv_mean_center(filename.c_str(),operation.at("RequireQ4").get<bool>())):json(nullptr)}};}
    fields(operation,{"Op","Context","Indices"});require(op=="context-logits"&&operation.at("Indices").is_array()&&operation.at("Indices").size()<=2048,"logit request indices exceed bound");auto*vocab=llama_model_get_vocab(llama_get_model(context));auto count=llama_vocab_n_tokens(vocab);require(count>0&&count<=1048576&&operation.at("Indices").size()*static_cast<std::uint64_t>(count)*sizeof(float)<=8*1024*1024,"requested logits exceed8MiB");json values=json::array();for(const auto&index:operation.at("Indices")){require(index.is_number_integer(),"logit row must be integer");auto i=index.get<std::int64_t>();require(i>=-1&&i<=2047,"logit row outside source range");const auto*data=llama_get_logits_ith(context,static_cast<std::int32_t>(i));values.push_back(json{{"Present",data!=nullptr},{"DataHex",data?json(hex(data,static_cast<std::size_t>(count)*sizeof(float))):json(nullptr)}});}return json{{"Vocabulary",count},{"Values",values}};
}
} // namespace prism_generated_model
