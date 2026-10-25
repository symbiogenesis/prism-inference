// Production sampler primitives and caller-owned observer callbacks. Source
// schedules, token policies, counts, comparisons and references belong to Rust.
#pragma once
#include "sampling.hpp"
#include "ggml-backend.h"
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>

namespace prism_backend_sampler {
using json=nlohmann::ordered_json;
using prism_sampling::fields;
using prism_sampling::require;
using prism_sampling::i32;
using prism_sampling::u32;
using prism_sampling::from_bits;
inline std::string text(const json&v,std::size_t maximum=65536){require(v.is_string(),"backend sampler text required");auto s=v.get<std::string>();require(s.size()<=maximum&&s.find('\0')==std::string::npos,"backend sampler text extent differs");return s;}
inline bool handles(const std::string&op){return op.rfind("bs-",0)==0;}
inline std::filesystem::path file(const json&v){auto p=std::filesystem::path(text(v));require(p.is_absolute(),"backend sampler file must be absolute");p=std::filesystem::weakly_canonical(p);auto root=std::filesystem::canonical(std::filesystem::current_path());auto relative=p.lexically_relative(root);require(!relative.empty()&&!relative.is_absolute(),"backend sampler file outside session");for(const auto&part:relative)require(part!="..","backend sampler file outside session");return p;}
inline std::vector<std::uint8_t>read(const json&name,std::size_t bytes){const auto p=file(name);require(bytes<=12*1024*1024&&std::filesystem::is_regular_file(p)&&std::filesystem::file_size(p)==bytes,"backend sampler file extent differs");std::ifstream input(p,std::ios::binary);std::vector<std::uint8_t>data(bytes);if(bytes)input.read(reinterpret_cast<char*>(data.data()),std::streamsize(bytes));require(bool(input),"backend sampler file read failed");return data;}
inline void write(const json&name,const void*data,std::size_t bytes){require(bytes<=12*1024*1024,"backend sampler output exceeds bound");std::ofstream output(file(name),std::ios::binary|std::ios::trunc);if(bytes)output.write(static_cast<const char*>(data),std::streamsize(bytes));require(bool(output),"backend sampler file write failed");}
struct state;
struct observer{state*parent;std::string name;std::uint32_t label;std::exception_ptr error;std::mutex mutex;};
struct sampler_owner{llama_sampler*value=nullptr;std::vector<std::shared_ptr<observer>>observers;~sampler_owner(){if(value)llama_sampler_free(value);}};
struct state{
    std::map<std::uint32_t,std::shared_ptr<sampler_owner>>samplers;
    std::map<std::uint32_t,std::shared_ptr<prism_sampling::candidate_storage>>candidates;
    std::vector<std::weak_ptr<observer>>observers;
    std::function<json(std::uint64_t,const json&)>sink;
    std::uint64_t request=0;
    std::uint32_t next=1;
    void prepare(std::uint64_t id,std::function<json(std::uint64_t,const json&)>callback){request=id;sink=std::move(callback);}
    template<class T>static std::shared_ptr<T>get(const std::map<std::uint32_t,std::shared_ptr<T>>&table,const json&key){const auto it=table.find(u32(key,1000000));require(it!=table.end(),"backend sampler unknown handle");return it->second;}
    template<class T>std::uint32_t add(std::map<std::uint32_t,std::shared_ptr<T>>&table,std::shared_ptr<T>value){require(table.size()<128&&next<1000000,"backend sampler handle bound");const auto id=next++;table.emplace(id,std::move(value));return id;}
    std::shared_ptr<sampler_owner>sampler(const json&key){return get(samplers,key);}
    void check(){for(auto i=observers.begin();i!=observers.end();){if(auto o=i->lock()){std::lock_guard<std::mutex>guard(o->mutex);if(o->error){auto e=o->error;o->error=nullptr;std::rethrow_exception(e);}++i;}else i=observers.erase(i);}}
    static json event(observer&o,const char*kind,const json&extra=json::object()){json value={{"Kind",kind},{"Sampler",o.label}};for(auto i=extra.begin();i!=extra.end();++i)value[i.key()]=i.value();require(bool(o.parent->sink)&&o.parent->request>0,"backend sampler callback outside active request");return o.parent->sink(o.parent->request,value);}
    static void latch(observer&o)noexcept{std::lock_guard<std::mutex>guard(o.mutex);o.error=std::current_exception();}
    static llama_sampler_i*observer_interface(){static auto iface=[](){llama_sampler_i i{};
        i.name=[](const llama_sampler*s){return static_cast<std::shared_ptr<observer>*>(s->ctx)->get()->name.c_str();};
        i.free=[](llama_sampler*s){delete static_cast<std::shared_ptr<observer>*>(s->ctx);};
        i.apply=[](llama_sampler*s,llama_token_data_array*){auto&o=**static_cast<std::shared_ptr<observer>*>(s->ctx);try{auto result=event(o,"apply");require(result.is_object()&&result.empty(),"backend sampler observer void reply differs");}catch(...){latch(o);}};
        i.backend_init=[](llama_sampler*s,ggml_backend_buffer_type_t buft,std::uint32_t outputs){auto&o=**static_cast<std::shared_ptr<observer>*>(s->ctx);try{auto result=event(o,"backend-init",{{"OutputsMaxPerSequence",outputs},{"BufferTypeName",ggml_backend_buft_name(buft)}});fields(result,{"Accept"});require(result.at("Accept").is_boolean(),"backend sampler observer Accept must be boolean");return result.at("Accept").get<bool>();}catch(...){latch(o);return false;}};
        i.backend_apply=[](llama_sampler*s,ggml_context*,ggml_cgraph*,llama_sampler_data*){auto&o=**static_cast<std::shared_ptr<observer>*>(s->ctx);try{auto result=event(o,"backend-apply");require(result.is_object()&&result.empty(),"backend sampler observer void reply differs");}catch(...){latch(o);}};
        return i;}();return &iface;}
    static json view(const prism_sampling::candidate_storage&c){return{{"BufferSize",c.backing.size()},{"Size",c.view.size},{"Offset",c.offset()},{"Selected",c.view.selected},{"Sorted",c.view.sorted}};}
    template<class Context,class Model>json execute(const json&o,std::uint32_t worker,Context resolve_context,Model resolve_model){
        require(worker==0,"backend sampler state belongs to worker zero");const auto op=text(o.at("Op"),64);
        if(op=="bs-init"){
            fields(o,{"Op","Type","Parameters"});const auto type=text(o.at("Type"),64);const auto&p=o.at("Parameters");auto value=std::make_shared<sampler_owner>();
            if(type=="chain"){fields(p,{"NoPerf"});auto params=llama_sampler_chain_default_params();if(!p.at("NoPerf").is_null()){require(p.at("NoPerf").is_boolean(),"backend sampler NoPerf invalid");params.no_perf=p.at("NoPerf").get<bool>();}value->value=llama_sampler_chain_init(params);}
            else if(type=="greedy"){fields(p,{});value->value=llama_sampler_init_greedy();}
            else if(type=="logit-bias"){fields(p,{"VocabularySize","Biases"});auto n=u32(p.at("VocabularySize"),1048576);require(n>0&&p.at("Biases").is_array()&&p.at("Biases").size()<=1048576,"backend sampler bias extent invalid");std::vector<llama_logit_bias>bias;for(const auto&b:p.at("Biases")){fields(b,{"Token","BiasBits"});bias.push_back({i32(b.at("Token")),from_bits(b.at("BiasBits"))});}value->value=llama_sampler_init_logit_bias(std::int32_t(n),std::int32_t(bias.size()),bias.data());}
            else if(type=="observer"){fields(p,{"Name","Label"});auto observed=std::make_shared<observer>();observed->parent=this;observed->name=text(p.at("Name"),128);observed->label=u32(p.at("Label"));auto*owned=new std::shared_ptr<observer>(observed);value->value=llama_sampler_init(observer_interface(),owned);if(!value->value)delete owned;value->observers.push_back(observed);observers.push_back(observed);}
            else if(type=="penalties"){fields(p,{"VocabularySize","PenaltyLastN","RepeatBits","FrequencyBits","PresenceBits"});require(u32(p.at("VocabularySize"),1048576)>0,"backend sampler vocabulary empty");value->value=llama_sampler_init_penalties(i32(p.at("VocabularySize")),i32(p.at("PenaltyLastN")),from_bits(p.at("RepeatBits")),from_bits(p.at("FrequencyBits")),from_bits(p.at("PresenceBits")));}
            else{prism_sampling::parameters(type,p);value->value=prism_sampling::create_sampler({{"Type",type},{"Parameters",p}});}
            return{{"Sampler",value->value?json(add(samplers,value)):json(nullptr)}};
        }
        if(op=="bs-chain-add"){fields(o,{"Op","Chain","Sampler"});require(o.at("Chain")!=o.at("Sampler"),"backend sampler cannot own itself");auto chain=sampler(o.at("Chain")),child=sampler(o.at("Sampler"));llama_sampler_chain_add(chain->value,child->value);chain->observers.insert(chain->observers.end(),child->observers.begin(),child->observers.end());child->value=nullptr;samplers.erase(u32(o.at("Sampler")));return nullptr;}
        if(op=="bs-info"){fields(o,{"Op","Sampler"});auto s=sampler(o.at("Sampler"));return{{"HasBackendApply",s->value->iface->backend_apply!=nullptr}};}
        if(op=="bs-clone"){fields(o,{"Op","Sampler"});auto source=sampler(o.at("Sampler"));auto value=std::make_shared<sampler_owner>();value->value=llama_sampler_clone(source->value);return{{"Sampler",value->value?json(add(samplers,value)):json(nullptr)}};}
        if(op=="bs-copy"){fields(o,{"Op","Destination","Source"});llama_sampler_copy(sampler(o.at("Source"))->value,sampler(o.at("Destination"))->value);return nullptr;}
        if(op=="bs-accept"||op=="bs-free"){if(op=="bs-accept")fields(o,{"Op","Sampler","Token"});else fields(o,{"Op","Sampler"});auto s=sampler(o.at("Sampler"));if(op=="bs-accept")llama_sampler_accept(s->value,i32(o.at("Token")));else samplers.erase(u32(o.at("Sampler")));return nullptr;}
        if(op=="bs-candidates-new"){fields(o,{"Op","Count","File","Selected","Sorted"});const auto n=u32(o.at("Count"),1048576);require(n>0&&o.at("Sorted").is_boolean(),"backend sampler candidate extent differs");const auto bytes=read(o.at("File"),std::size_t(n)*12);auto c=std::make_shared<prism_sampling::candidate_storage>();c->backing.resize(n);for(std::size_t i=0;i<n;++i){std::memcpy(&c->backing[i].id,bytes.data()+12*i,4);std::memcpy(&c->backing[i].logit,bytes.data()+12*i+4,4);std::memcpy(&c->backing[i].p,bytes.data()+12*i+8,4);}c->view={c->backing.data(),n,prism_sampling::integer(o.at("Selected")),o.at("Sorted").get<bool>()};return{{"Candidates",add(candidates,c)}};}
        if(op=="bs-apply"){fields(o,{"Op","Sampler","Candidates"});auto c=get(candidates,o.at("Candidates"));llama_sampler_apply(sampler(o.at("Sampler"))->value,&c->view);return view(*c);}
        if(op=="bs-candidates-export"){fields(o,{"Op","Candidates","File","Selection","Count"});auto c=get(candidates,o.at("Candidates"));const auto selection=text(o.at("Selection"),16);std::size_t offset=0,count=c->backing.size();if(selection=="active"){offset=c->offset();count=c->view.size;}else if(selection=="prefix"){count=u32(o.at("Count"),std::uint32_t(c->backing.size()));}else throw std::runtime_error("backend sampler candidate selection invalid");std::vector<std::uint8_t>bytes(count*12);for(std::size_t i=0;i<count;++i){auto&v=c->backing[offset+i];std::memcpy(bytes.data()+12*i,&v.id,4);std::memcpy(bytes.data()+12*i+4,&v.logit,4);std::memcpy(bytes.data()+12*i+8,&v.p,4);}write(o.at("File"),bytes.data(),bytes.size());auto result=view(*c);result["Rows"]=count;result["Bytes"]=bytes.size();return result;}
        if(op=="bs-candidates-free"){fields(o,{"Op","Candidates"});get(candidates,o.at("Candidates"));candidates.erase(u32(o.at("Candidates")));return nullptr;}
        if(op=="bs-sample"){fields(o,{"Op","Sampler","Context","Index"});auto c=resolve_context(o.at("Context"));return{{"Token",llama_sampler_sample(sampler(o.at("Sampler"))->value,c->value,i32(o.at("Index")))}};}
        if(op=="bs-context-set"){fields(o,{"Op","Context","Sequence","Sampler"});auto c=resolve_context(o.at("Context"));const auto sequence=i32(o.at("Sequence"));std::shared_ptr<sampler_owner>s;if(!o.at("Sampler").is_null())s=sampler(o.at("Sampler"));const auto set=llama_set_sampler(c->value,sequence,s?s->value:nullptr);if(set){if(s)c->backend_samplers[sequence]=s;else c->backend_samplers.erase(sequence);}return{{"Set",set}};}
        if(op=="bs-tokenize"){fields(o,{"Op","Model","InputHex","Capacity","AddSpecial","ParseSpecial"});auto m=resolve_model(o.at("Model"));const auto input=prism_vocab_session::bytes(o.at("InputHex"));const auto capacity=u32(o.at("Capacity"),1048576);require(o.at("AddSpecial").is_boolean()&&o.at("ParseSpecial").is_boolean(),"backend sampler tokenizer flags invalid");std::vector<llama_token>tokens(capacity);const auto ret=llama_tokenize(llama_model_get_vocab(m->value),input.data(),std::int32_t(input.size()),tokens.data(),std::int32_t(tokens.size()),o.at("AddSpecial").get<bool>(),o.at("ParseSpecial").get<bool>());return{{"ReturnCode",ret},{"Backing",tokens}};}
        if(op=="bs-piece"){fields(o,{"Op","Model","Token","Capacity","LStrip","Special"});auto m=resolve_model(o.at("Model"));std::vector<char>bytes(u32(o.at("Capacity"),1048576));require(o.at("Special").is_boolean(),"backend sampler piece flag invalid");const auto ret=llama_token_to_piece(llama_model_get_vocab(m->value),i32(o.at("Token")),bytes.data(),std::int32_t(bytes.size()),i32(o.at("LStrip")),o.at("Special").get<bool>());return{{"ReturnCode",ret},{"BackingHex",prism_vocab_session::hex(std::string(bytes.begin(),bytes.end()))}};}
        if(op=="bs-output"){
            fields(o,{"Op","Context","Index","Prefix","Raw","Logits","Probabilities","Candidates","Token"});auto c=resolve_context(o.at("Context"));const auto index=i32(o.at("Index"));const auto prefix=file(o.at("Prefix")).string();json result=json::object();
            for(const auto*name:{"Raw","Logits","Probabilities","Candidates","Token"})require(o.at(name).is_boolean(),"backend sampler output flag invalid");
            const auto vocab=llama_vocab_n_tokens(llama_model_get_vocab(c->parent->value));require(vocab>0&&vocab<=1048576,"backend sampler vocabulary outside bound");
            const auto save=[&](const char*key,const void*data,std::size_t count,const char*suffix){require(count<=1048576,"backend sampler observation count exceeds bound");json path=nullptr;if(data){path=prefix+suffix;write(path,data,count*4);}result[key]={{"Count",count},{"Present",data!=nullptr},{"File",path}};};
            if(o.at("Raw").get<bool>())save("Raw",llama_get_logits_ith(c->value,index),std::size_t(vocab),"-raw.bin");else result["Raw"]=nullptr;
            if(o.at("Logits").get<bool>())save("Logits",llama_get_sampled_logits_ith(c->value,index),llama_get_sampled_logits_count_ith(c->value,index),"-logits.bin");else result["Logits"]=nullptr;
            if(o.at("Probabilities").get<bool>())save("Probabilities",llama_get_sampled_probs_ith(c->value,index),llama_get_sampled_probs_count_ith(c->value,index),"-probs.bin");else result["Probabilities"]=nullptr;
            if(o.at("Candidates").get<bool>())save("Candidates",llama_get_sampled_candidates_ith(c->value,index),llama_get_sampled_candidates_count_ith(c->value,index),"-candidates.bin");else result["Candidates"]=nullptr;
            result["Token"]=o.at("Token").get<bool>()?json(llama_get_sampled_token_ith(c->value,index)):json(nullptr);return result;
        }
        throw std::runtime_error("unknown backend sampler primitive");
    }
};
} // namespace prism_backend_sampler
