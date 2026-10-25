// Scoped production samplers and caller-owned candidate bytes; no match policy.
#pragma once
#include "llama.h"
#include "sampling.h"
#include <nlohmann/json.hpp>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <vector>

namespace prism_llguidance {
using json=nlohmann::ordered_json;
inline void require(bool ok,const char * message){if(!ok)throw std::runtime_error(message);}
inline void fields(const json&o,std::initializer_list<const char*>names){require(o.is_object()&&o.size()==names.size(),"LLG fields differ");for(auto*n:names)require(o.contains(n),"LLG field missing");}
inline std::uint32_t number(const json&v,std::uint32_t maximum=UINT32_MAX){require(v.is_number_integer()&&(v.is_number_unsigned()||v.get<std::int64_t>()>=0),"LLG unsigned integer required");const auto n=v.get<std::uint64_t>();require(n<=maximum,"LLG integer exceeds bound");return std::uint32_t(n);}
inline std::int64_t signed_number(const json&v){require(v.is_number_integer(),"LLG signed integer required");if(v.is_number_unsigned())require(v.get<std::uint64_t>()<=std::uint64_t(INT64_MAX),"LLG signed overflow");return v.get<std::int64_t>();}
inline std::string text(const json&v){require(v.is_string(),"LLG text required");auto s=v.get<std::string>();require(s.size()<=65536&&s.find('\0')==std::string::npos,"LLG text outside bound");return s;}
inline float floating(const json&v){const auto bits=number(v);float x;std::memcpy(&x,&bits,4);return x;}
inline std::uint32_t bits(float x){std::uint32_t value;std::memcpy(&value,&x,4);return value;}
inline bool handles(const std::string&op){return op.rfind("llg-",0)==0;}
inline bool enabled(){
#ifdef LLAMA_USE_LLGUIDANCE
return true;
#else
return false;
#endif
}
struct sampler_owner{llama_sampler*value=nullptr;std::vector<std::shared_ptr<void>>models;~sampler_owner(){if(value)llama_sampler_free(value);}};
struct candidates{std::vector<llama_token_data>data;llama_token_data_array view{};};
struct state{
    std::map<std::uint32_t,std::shared_ptr<sampler_owner>>samplers;
    std::map<std::uint32_t,std::shared_ptr<candidates>>arrays;
    std::uint32_t next=1;
    template<class T>static std::shared_ptr<T>get(const std::map<std::uint32_t,std::shared_ptr<T>>&table,const json&handle){const auto it=table.find(number(handle,1000000));require(it!=table.end(),"LLG unknown handle");return it->second;}
    template<class T>std::uint32_t add(std::map<std::uint32_t,std::shared_ptr<T>>&table,std::shared_ptr<T>value){require(table.size()<64&&next<1000000,"LLG handle bound");const auto id=next++;table.emplace(id,std::move(value));return id;}
    static std::vector<std::uint8_t>read(const json&name,std::size_t expected){const auto path=std::filesystem::canonical(text(name));require(path.is_absolute()&&std::filesystem::is_regular_file(path),"LLG candidate file must be absolute and regular");const auto root=std::filesystem::canonical(std::filesystem::current_path());const auto relative=path.lexically_relative(root);require(!relative.empty()&&!relative.is_absolute(),"LLG file outside session directory");for(const auto&part:relative)require(part!="..","LLG file outside session directory");require(expected<=12*1024*1024&&std::filesystem::file_size(path)==expected,"LLG candidate file extent differs");std::ifstream f(path,std::ios::binary);std::vector<std::uint8_t>data(expected);if(expected)f.read(reinterpret_cast<char*>(data.data()),std::streamsize(expected));require(bool(f),"LLG candidate file read failed");return data;}
    static json candidate(const llama_token_data&v){return{{"Id",v.id},{"LogitBits",bits(v.logit)},{"ProbabilityBits",bits(v.p)}};}
    static void import(candidates&c,const json&name,const std::string&encoding){const auto size=encoding=="candidate-le"?12:encoding=="id-logit-le"?8:encoding=="logits-le"?4:0;require(size!=0,"LLG candidate encoding unknown");const auto data=read(name,c.data.size()*size);for(std::size_t i=0;i<c.data.size();++i){auto*p=data.data()+i*size;std::uint32_t word;if(size>=8){std::memcpy(&c.data[i].id,p,4);p+=4;}std::memcpy(&word,p,4);std::memcpy(&c.data[i].logit,&word,4);if(size==12)std::memcpy(&c.data[i].p,p+4,4);}}
    // Copy only the fields supplied by the original encoding. In particular a logits
    // reset preserves candidate probabilities, IDs and the active view metadata.
    static void copy(candidates&destination,const candidates&source,const std::string&encoding){
        require(destination.data.size()==source.data.size(),"LLG candidate copy extent differs");
        if(encoding=="candidate-le"){
            for(std::size_t i=0;i<source.data.size();++i)std::memcpy(&destination.data[i],&source.data[i],sizeof(llama_token_data));
        }else if(encoding=="id-logit-le"){
            for(std::size_t i=0;i<source.data.size();++i){destination.data[i].id=source.data[i].id;std::memcpy(&destination.data[i].logit,&source.data[i].logit,4);}
        }else{
            require(encoding=="logits-le","LLG candidate encoding unknown");
            for(std::size_t i=0;i<source.data.size();++i)std::memcpy(&destination.data[i].logit,&source.data[i].logit,4);
        }
    }
    static std::shared_ptr<candidates>clone(const candidates&source){const auto begin=reinterpret_cast<std::uintptr_t>(source.data.data()),active=reinterpret_cast<std::uintptr_t>(source.view.data);require(active>=begin&&(active-begin)%sizeof(llama_token_data)==0,"LLG candidate active pointer invalid");const auto offset=(active-begin)/sizeof(llama_token_data);require(offset<=source.data.size()&&source.view.size<=source.data.size()-offset,"LLG candidate returned extent invalid");auto c=std::make_shared<candidates>();c->data=source.data;c->view={c->data.data()+offset,source.view.size,source.view.selected,source.view.sorted};return c;}
    static void metadata(candidates&c,const json&m){if(!m.is_null()){fields(m,{"Offset","Size","Selected","Sorted"});const auto offset=number(m.at("Offset"),std::uint32_t(c.data.size())),size=number(m.at("Size"),std::uint32_t(c.data.size()));require(size<=c.data.size()-offset&&m.at("Sorted").is_boolean(),"LLG candidate active extent differs");c.view={c.data.data()+offset,size,signed_number(m.at("Selected")),m.at("Sorted").get<bool>()};}}
    static json inspect(const candidates&c,const json&indices){
        require(indices.is_array()&&indices.size()<=512,"LLG probe count exceeds bound");
        const auto begin=reinterpret_cast<std::uintptr_t>(c.data.data()),active=reinterpret_cast<std::uintptr_t>(c.view.data);
        require(active>=begin&&(active-begin)%sizeof(llama_token_data)==0,"LLG candidate active pointer invalid");
        const auto offset=(active-begin)/sizeof(llama_token_data);
        require(offset<=c.data.size()&&c.view.size<=c.data.size()-offset,"LLG candidate returned extent invalid");
        json probes=json::array();for(const auto&i:indices){const auto n=number(i,std::uint32_t(c.data.size()-1));probes.push_back({{"Index",n},{"Candidate",candidate(c.data[n])}});}
        json selected=nullptr;if(c.view.selected>=0&&std::uint64_t(c.view.selected)<c.data.size())selected=candidate(c.data[std::size_t(c.view.selected)]);
        return{{"BufferSize",c.data.size()},{"Size",c.view.size},{"Offset",offset},{"Selected",c.view.selected},{"Sorted",c.view.sorted},{"SelectedBacking",selected},{"Probes",probes}};
    }
    template<class Model>json execute(const json&o,std::uint32_t worker,Model resolve_model){
        require(worker==0,"LLG state belongs to worker zero");const auto op=text(o.at("Op"));
        if(op=="llg-available"){fields(o,{"Op"});return{{"Enabled",enabled()}};}
        require(enabled()||op.rfind("llg-candidates-",0)==0,"production LLGuidance feature was not built");
        if(op=="llg-vocabulary"){fields(o,{"Op","Model"});auto m=resolve_model(o.at("Model"));const auto*v=llama_model_get_vocab(m->value);return{{"Count",llama_vocab_n_tokens(v)},{"Eot",llama_vocab_eot(v)},{"Eos",llama_vocab_eos(v)}};}
        if(op=="llg-new"){fields(o,{"Op","Model","GrammarKind","GrammarData"});auto m=resolve_model(o.at("Model"));auto owner=std::make_shared<sampler_owner>();owner->models.push_back(m);const auto kind=text(o.at("GrammarKind")),data=text(o.at("GrammarData"));owner->value=llama_sampler_init_llg(llama_model_get_vocab(m->value),kind.c_str(),data.c_str());return{{"Sampler",owner->value?json(add(samplers,owner)):json(nullptr)}};}
        if(op=="llg-chain-new"){fields(o,{"Op","NoPerf"});require(o.at("NoPerf").is_boolean(),"LLG NoPerf boolean required");auto owner=std::make_shared<sampler_owner>();auto params=llama_sampler_chain_default_params();params.no_perf=o.at("NoPerf").get<bool>();owner->value=llama_sampler_chain_init(params);return{{"Sampler",owner->value?json(add(samplers,owner)):json(nullptr)}};}
        if(op=="llg-dist-new"){fields(o,{"Op","Seed"});auto owner=std::make_shared<sampler_owner>();owner->value=llama_sampler_init_dist(number(o.at("Seed")));return{{"Sampler",owner->value?json(add(samplers,owner)):json(nullptr)}};}
        if(op=="llg-chain-add"){fields(o,{"Op","Chain","Sampler"});require(o.at("Chain")!=o.at("Sampler"),"LLG chain cannot own itself");auto chain=get(samplers,o.at("Chain")),child=get(samplers,o.at("Sampler"));llama_sampler_chain_add(chain->value,child->value);chain->models.insert(chain->models.end(),child->models.begin(),child->models.end());child->value=nullptr;samplers.erase(number(o.at("Sampler")));return nullptr;}
        if(op=="llg-candidates-new"){fields(o,{"Op","Count","File","Selected","Sorted"});const auto count=number(o.at("Count"),1048576);require(count>0&&o.at("Sorted").is_boolean(),"LLG candidate extent/Sorted invalid");auto c=std::make_shared<candidates>();c->data.resize(count);import(*c,o.at("File"),"candidate-le");c->view={c->data.data(),c->data.size(),signed_number(o.at("Selected")),o.at("Sorted").get<bool>()};return{{"Candidates",add(arrays,c)}};}
        if(op=="llg-candidates-inspect"){fields(o,{"Op","Candidates","Indices"});return inspect(*get(arrays,o.at("Candidates")),o.at("Indices"));}
        if(op=="llg-candidates-clone"){fields(o,{"Op","Source"});return{{"Candidates",add(arrays,clone(*get(arrays,o.at("Source"))))}};}
        if(op=="llg-candidates-copy"){fields(o,{"Op","Candidates","Source","Encoding","Metadata"});auto c=get(arrays,o.at("Candidates"));copy(*c,*get(arrays,o.at("Source")),text(o.at("Encoding")));metadata(*c,o.at("Metadata"));return nullptr;}
        if(op=="llg-candidates-load"){fields(o,{"Op","Candidates","File","Encoding","Metadata"});auto c=get(arrays,o.at("Candidates"));import(*c,o.at("File"),text(o.at("Encoding")));metadata(*c,o.at("Metadata"));return nullptr;}
        if(op=="llg-candidates-patch"){fields(o,{"Op","Candidates","Patches"});auto c=get(arrays,o.at("Candidates"));require(o.at("Patches").is_array()&&o.at("Patches").size()<=512,"LLG patch count exceeds bound");for(const auto&p:o.at("Patches")){fields(p,{"Index","Fields"});const auto index=number(p.at("Index"),std::uint32_t(c->data.size()-1));fields(p.at("Fields"),{"LogitBits"});c->data[index].logit=floating(p.at("Fields").at("LogitBits"));}return nullptr;}
        if(op=="llg-apply"){fields(o,{"Op","Sampler","Candidates","Indices"});auto s=get(samplers,o.at("Sampler"));auto c=get(arrays,o.at("Candidates"));require(o.at("Indices").is_array()&&o.at("Indices").size()<=512,"LLG probe count exceeds bound");llama_sampler_apply(s->value,&c->view);return inspect(*c,o.at("Indices"));}
        if(op=="llg-reset"||op=="llg-accept"||op=="llg-free"){if(op=="llg-accept")fields(o,{"Op","Sampler","Token"});else fields(o,{"Op","Sampler"});auto s=get(samplers,o.at("Sampler"));if(op=="llg-reset")llama_sampler_reset(s->value);else if(op=="llg-free")samplers.erase(number(o.at("Sampler")));else{const auto token=signed_number(o.at("Token"));require(token>=INT32_MIN&&token<=INT32_MAX,"LLG accepted token outside range");llama_sampler_accept(s->value,llama_token(token));}return nullptr;}
        if(op=="llg-candidates-free"){fields(o,{"Op","Candidates"});get(arrays,o.at("Candidates"));arrays.erase(number(o.at("Candidates")));return nullptr;}
        throw std::runtime_error("unknown LLG operation");
    }
};
} // namespace prism_llguidance
