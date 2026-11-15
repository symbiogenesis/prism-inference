// Production sampler primitives and caller-owned observer callbacks. Source
// schedules, token policies, counts, comparisons and references belong to Rust.
#pragma once
#include "llama.h"
#include "llama-sampler.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <nlohmann/json.hpp>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace prism_backend_sampler {
using json=nlohmann::ordered_json;
constexpr std::size_t max_candidates = 131072;
inline void require(bool condition, const char * message) { if (!condition) throw std::runtime_error(message); }
inline void fields(const json & value, std::initializer_list<const char *> names) {
    require(value.is_object() && value.size() == names.size(), "sampler fields mismatch");
    for (const auto * name : names) require(value.contains(name), "missing sampler field");
}
inline std::int64_t integer(const json & value) {
    require(value.is_number_integer(), "sampler value must be integer");
    if (value.is_number_unsigned()) {
        const auto number = value.get<std::uint64_t>();
        require(number <= std::uint64_t(INT64_MAX), "sampler signed integer overflow");
        return static_cast<std::int64_t>(number);
    }
    return value.get<std::int64_t>();
}
inline std::int32_t i32(const json & value) {
    const auto number = integer(value);
    require(number >= INT32_MIN && number <= INT32_MAX, "sampler i32 overflow");
    return static_cast<std::int32_t>(number);
}
inline std::uint32_t u32(const json & value, std::uint32_t maximum = UINT32_MAX) {
    const auto number = integer(value);
    require(number >= 0 && std::uint64_t(number) <= maximum, "sampler unsigned value outside bound");
    return static_cast<std::uint32_t>(number);
}
inline float from_bits(const json & value) {
    const auto bits = u32(value); float result;
    std::memcpy(&result, &bits, sizeof(result)); return result;
}
inline float parameter_float(const json & value) {
    const auto result = from_bits(value);
    require(std::isfinite(result), "sampler parameter must be finite"); return result;
}
inline void min_keep(const json & value) { u32(value, static_cast<std::uint32_t>(max_candidates)); }
inline void parameters(const std::string & type, const json & p) {
    if (type == "temp") {
        fields(p, {"TemperatureBits"}); parameter_float(p.at("TemperatureBits"));
    } else if (type == "temp-ext") {
        fields(p, {"TemperatureBits", "DeltaBits", "ExponentBits"});
        parameter_float(p.at("TemperatureBits")); parameter_float(p.at("DeltaBits")); parameter_float(p.at("ExponentBits"));
    } else if (type == "dist") {
        fields(p, {"Seed"}); u32(p.at("Seed"));
    } else if (type == "top-k") {
        fields(p, {"K"}); i32(p.at("K"));
    } else if (type == "top-p" || type == "min-p" || type == "typical") {
        fields(p, {"ProbabilityBits", "MinKeep"}); parameter_float(p.at("ProbabilityBits")); min_keep(p.at("MinKeep"));
    } else if (type == "xtc") {
        fields(p, {"ProbabilityBits", "ThresholdBits", "MinKeep", "Seed"});
        parameter_float(p.at("ProbabilityBits")); parameter_float(p.at("ThresholdBits")); min_keep(p.at("MinKeep")); u32(p.at("Seed"));
    } else if (type == "penalties") {
        fields(p, {"VocabularySize", "PenaltyLastN", "RepeatBits", "FrequencyBits", "PresenceBits"});
        require(u32(p.at("VocabularySize"), max_candidates) > 0, "sampler vocabulary cannot be empty");
        const auto last = i32(p.at("PenaltyLastN"));
        require(last >= -1 && last <= static_cast<std::int32_t>(max_candidates), "penalty history exceeds bound");
        require(parameter_float(p.at("RepeatBits")) > 0, "repeat penalty must be positive");
        parameter_float(p.at("FrequencyBits")); parameter_float(p.at("PresenceBits"));
    } else if (type == "dry-testing") {
        fields(p, {"MultiplierBits", "BaseBits", "AllowedLength", "PenaltyLastN", "SequenceBreakers"});
        parameter_float(p.at("MultiplierBits")); parameter_float(p.at("BaseBits"));
        const auto allowed = i32(p.at("AllowedLength")), last = i32(p.at("PenaltyLastN"));
        require(allowed >= 0 && allowed <= static_cast<std::int32_t>(max_candidates), "DRY allowed length exceeds bound");
        require(last >= -1 && last <= static_cast<std::int32_t>(max_candidates), "DRY history exceeds bound");
        const auto & breakers = p.at("SequenceBreakers");
        require(breakers.is_array() && breakers.size() <= 128, "too many DRY sequence breakers");
        std::size_t tokens = 0;
        for (const auto & breaker : breakers) {
            require(breaker.is_array() && breaker.size() <= 1024, "DRY breaker exceeds bound");
            tokens += breaker.size(); require(tokens <= 8192, "DRY breaker tokens exceed bound");
            for (const auto & token : breaker) i32(token);
        }
    } else if (type == "top-n-sigma") {
        fields(p, {"NBits"}); parameter_float(p.at("NBits"));
    } else throw std::runtime_error("unknown sampler Type");
}
inline llama_sampler * create_sampler(const std::string & type, const json & p) {
    if (type == "temp") return llama_sampler_init_temp(from_bits(p.at("TemperatureBits")));
    if (type == "temp-ext") return llama_sampler_init_temp_ext(from_bits(p.at("TemperatureBits")), from_bits(p.at("DeltaBits")), from_bits(p.at("ExponentBits")));
    if (type == "dist") return llama_sampler_init_dist(u32(p.at("Seed")));
    if (type == "top-k") return llama_sampler_init_top_k(i32(p.at("K")));
    if (type == "top-p") return llama_sampler_init_top_p(from_bits(p.at("ProbabilityBits")), u32(p.at("MinKeep")));
    if (type == "min-p") return llama_sampler_init_min_p(from_bits(p.at("ProbabilityBits")), u32(p.at("MinKeep")));
    if (type == "typical") return llama_sampler_init_typical(from_bits(p.at("ProbabilityBits")), u32(p.at("MinKeep")));
    if (type == "xtc") return llama_sampler_init_xtc(from_bits(p.at("ProbabilityBits")), from_bits(p.at("ThresholdBits")), u32(p.at("MinKeep")), u32(p.at("Seed")));
    if (type == "penalties") return llama_sampler_init_penalties(i32(p.at("VocabularySize")), i32(p.at("PenaltyLastN")), from_bits(p.at("RepeatBits")), from_bits(p.at("FrequencyBits")), from_bits(p.at("PresenceBits")));
    if (type == "dry-testing") {
        std::vector<std::vector<llama_token>> breakers;
        for (const auto & entry : p.at("SequenceBreakers")) {
            std::vector<llama_token> tokens; for (const auto & token : entry) tokens.push_back(i32(token)); breakers.push_back(std::move(tokens));
        }
        return llama_sampler_init_dry_testing(from_bits(p.at("MultiplierBits")), from_bits(p.at("BaseBits")), i32(p.at("AllowedLength")), i32(p.at("PenaltyLastN")), breakers);
    }
    return llama_sampler_init_top_n_sigma(from_bits(p.at("NBits")));
}
struct candidate_storage {
    std::vector<llama_token_data> backing;
    llama_token_data_array view{};
    std::size_t offset() const {
        const auto begin = reinterpret_cast<std::uintptr_t>(backing.data());
        const auto pointer = reinterpret_cast<std::uintptr_t>(view.data);
        const auto bytes = backing.size() * sizeof(llama_token_data);
        require(pointer >= begin && pointer - begin <= bytes && (pointer - begin) % sizeof(llama_token_data) == 0, "production candidate pointer outside backing storage");
        const auto index = static_cast<std::size_t>((pointer - begin) / sizeof(llama_token_data));
        require(view.size <= backing.size() - index, "production candidate view exceeds backing storage");
        return index;
    }
};
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
    std::map<std::uint32_t,std::shared_ptr<candidate_storage>>candidates;
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
    static json view(const candidate_storage&c){return{{"BufferSize",c.backing.size()},{"Size",c.view.size},{"Offset",c.offset()},{"Selected",c.view.selected},{"Sorted",c.view.sorted}};}
    template<class Context,class Model>json execute(const json&o,std::uint32_t worker,Context resolve_context,Model resolve_model){
        require(worker==0,"backend sampler state belongs to worker zero");const auto op=text(o.at("Op"),64);
        if(op=="bs-init"){
            fields(o,{"Op","Type","Parameters"});const auto type=text(o.at("Type"),64);const auto&p=o.at("Parameters");auto value=std::make_shared<sampler_owner>();
            if(type=="chain"){fields(p,{"NoPerf"});auto params=llama_sampler_chain_default_params();if(!p.at("NoPerf").is_null()){require(p.at("NoPerf").is_boolean(),"backend sampler NoPerf invalid");params.no_perf=p.at("NoPerf").get<bool>();}value->value=llama_sampler_chain_init(params);}
            else if(type=="greedy"){fields(p,{});value->value=llama_sampler_init_greedy();}
            else if(type=="logit-bias"){fields(p,{"VocabularySize","Biases"});auto n=u32(p.at("VocabularySize"),1048576);require(n>0&&p.at("Biases").is_array()&&p.at("Biases").size()<=1048576,"backend sampler bias extent invalid");std::vector<llama_logit_bias>bias;for(const auto&b:p.at("Biases")){fields(b,{"Token","BiasBits"});bias.push_back({i32(b.at("Token")),from_bits(b.at("BiasBits"))});}value->value=llama_sampler_init_logit_bias(std::int32_t(n),std::int32_t(bias.size()),bias.data());}
            else if(type=="observer"){fields(p,{"Name","Label"});auto observed=std::make_shared<observer>();observed->parent=this;observed->name=text(p.at("Name"),128);observed->label=u32(p.at("Label"));auto*owned=new std::shared_ptr<observer>(observed);value->value=llama_sampler_init(observer_interface(),owned);if(!value->value)delete owned;value->observers.push_back(observed);observers.push_back(observed);}
            else if(type=="penalties"){fields(p,{"VocabularySize","PenaltyLastN","RepeatBits","FrequencyBits","PresenceBits"});require(u32(p.at("VocabularySize"),1048576)>0,"backend sampler vocabulary empty");value->value=llama_sampler_init_penalties(i32(p.at("VocabularySize")),i32(p.at("PenaltyLastN")),from_bits(p.at("RepeatBits")),from_bits(p.at("FrequencyBits")),from_bits(p.at("PresenceBits")));}
            else{parameters(type,p);value->value=create_sampler(type,p);}
            return{{"Sampler",value->value?json(add(samplers,value)):json(nullptr)}};
        }
        if(op=="bs-chain-add"){fields(o,{"Op","Chain","Sampler"});require(o.at("Chain")!=o.at("Sampler"),"backend sampler cannot own itself");auto chain=sampler(o.at("Chain")),child=sampler(o.at("Sampler"));llama_sampler_chain_add(chain->value,child->value);chain->observers.insert(chain->observers.end(),child->observers.begin(),child->observers.end());child->value=nullptr;samplers.erase(u32(o.at("Sampler")));return nullptr;}
        if(op=="bs-info"){fields(o,{"Op","Sampler"});auto s=sampler(o.at("Sampler"));return{{"HasBackendApply",s->value->iface->backend_apply!=nullptr}};}
        if(op=="bs-clone"){fields(o,{"Op","Sampler"});auto source=sampler(o.at("Sampler"));auto value=std::make_shared<sampler_owner>();value->value=llama_sampler_clone(source->value);return{{"Sampler",value->value?json(add(samplers,value)):json(nullptr)}};}
        if(op=="bs-copy"){fields(o,{"Op","Destination","Source"});llama_sampler_copy(sampler(o.at("Source"))->value,sampler(o.at("Destination"))->value);return nullptr;}
        if(op=="bs-accept"||op=="bs-free"){if(op=="bs-accept")fields(o,{"Op","Sampler","Token"});else fields(o,{"Op","Sampler"});auto s=sampler(o.at("Sampler"));if(op=="bs-accept")llama_sampler_accept(s->value,i32(o.at("Token")));else samplers.erase(u32(o.at("Sampler")));return nullptr;}
        if(op=="bs-candidates-new"){fields(o,{"Op","Count","File","Selected","Sorted"});const auto n=u32(o.at("Count"),1048576);require(n>0&&o.at("Sorted").is_boolean(),"backend sampler candidate extent differs");const auto bytes=read(o.at("File"),std::size_t(n)*12);auto c=std::make_shared<candidate_storage>();c->backing.resize(n);for(std::size_t i=0;i<n;++i){std::memcpy(&c->backing[i].id,bytes.data()+12*i,4);std::memcpy(&c->backing[i].logit,bytes.data()+12*i+4,4);std::memcpy(&c->backing[i].p,bytes.data()+12*i+8,4);}c->view={c->backing.data(),n,integer(o.at("Selected")),o.at("Sorted").get<bool>()};return{{"Candidates",add(candidates,c)}};}
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
