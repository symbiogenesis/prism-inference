// Typed production argument/parser/download observations. No source expectations.
#pragma once
#include "arg.h"
#include "common.h"
#include "download.h"
#include "speculative.h"
#include <nlohmann/json.hpp>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace prism_arg_parser {
using json=nlohmann::ordered_json;
inline void require(bool value,const char*message){if(!value)throw std::runtime_error(message);}
inline void fields(const json&v,std::initializer_list<const char*>names){require(v.is_object()&&v.size()==names.size(),"argument observation fields mismatch");for(const auto*n:names)require(v.contains(n),"argument observation field missing");}
inline std::int32_t integer(const json&v,std::int32_t low,std::int32_t high){require(v.is_number_integer(),"argument observation integer required");if(v.is_number_unsigned())require(v.get<std::uint64_t>()<=static_cast<std::uint64_t>(high),"argument observation integer exceeds bound");auto n=v.get<std::int64_t>();require(n>=low&&n<=high,"argument observation integer outside bounds");return static_cast<std::int32_t>(n);}
inline std::string text(const json&v,std::size_t maximum){require(v.is_string(),"argument observation string required");auto s=v.get<std::string>();require(s.size()<=maximum&&s.find('\0')==std::string::npos,"argument observation string extent/NUL violation");return s;}
inline std::size_t slot(const json&v,std::size_t count){auto n=integer(v,0,15);require(static_cast<std::size_t>(n)<count,"argument params slot outside inventory");return static_cast<std::size_t>(n);}
inline bool handles(const std::string&kind){return kind=="argument-enums"||kind=="argument-parser-trace"||kind=="remote-get-content";}
inline void validate(const json&op){
    const auto kind=op.at("Kind").get<std::string>();
    if(kind=="argument-enums"){fields(op,{"Kind"});return;}
    if(kind=="remote-get-content"){
        fields(op,{"Kind","Url","Headers","TimeoutSeconds","MaxSize"});auto url=text(op.at("Url"),4096);require(url.rfind("http://",0)==0||url.rfind("https://",0)==0,"remote content URL must be HTTP(S)");integer(op.at("TimeoutSeconds"),0,60);integer(op.at("MaxSize"),0,8*1024*1024);const auto&headers=op.at("Headers");require(headers.is_array()&&headers.size()<=32,"remote header count exceeds32");for(const auto&h:headers){fields(h,{"Name","Value"});for(const auto*n:{"Name","Value"}){auto s=text(h.at(n),4096);require(s.find_first_of("\r\n")==std::string::npos,"remote header newline violation");}}return;
    }
    require(kind=="argument-parser-trace","unknown argument observation kind");fields(op,{"Kind","Initials","Actions"});const auto&initials=op.at("Initials");require(initials.is_array()&&!initials.empty()&&initials.size()<=16,"argument params slots exceed bound");for(const auto&i:initials){require(i.is_object()&&i.size()<=2,"argument initial fields exceed bound");for(const auto&item:i.items()){require(item.key()=="Parallel"||item.key()=="OutputsMaxPerSequence","unknown argument initial field");integer(item.value(),1,2147483647);}}
    const auto&actions=op.at("Actions");require(actions.is_array()&&actions.size()<=256,"argument actions exceed256");
    for(const auto&a:actions){require(a.is_object()&&a.contains("Action"),"argument action required");auto action=text(a.at("Action"),64);
        if(action=="limits"){fields(a,{"Action","Batch","Parallel","Draft"});integer(a.at("Batch"),1,2147483647);integer(a.at("Parallel"),1,2147483647);integer(a.at("Draft"),-1,2147483647);continue;}
        if(action=="snapshot"||action=="speculative"){fields(a,{"Action","Slot"});slot(a.at("Slot"),initials.size());continue;}
        if(action=="inventory")fields(a,{"Action","Slot","Example"});
        else if(action=="parse"){fields(a,{"Action","Slot","Example","Arguments"});const auto&args=a.at("Arguments");require(args.is_array()&&!args.empty()&&args.size()<=64,"argument argv extent outside bounds");std::size_t total=0;for(const auto&arg:args){total+=text(arg,16384).size();require(total<=65536,"argument argv bytes exceed64KiB");}}
        else throw std::runtime_error("unknown argument observation action");
        slot(a.at("Slot"),initials.size());integer(a.at("Example"),0,LLAMA_EXAMPLE_COUNT-1);
    }
}
inline json snapshot(const common_params&p){json lora=json::array();for(const auto&a:p.lora_adapters)lora.push_back(a.path);return json{{"ModelPath",p.model.path},{"Threads",p.cpuparams.n_threads},{"Verbosity",p.verbosity},{"Predict",p.n_predict},{"Batch",p.n_batch},{"DraftMax",p.speculative.draft.n_max},{"LoadMode",static_cast<int>(p.load_mode)},{"PenaltyLast",p.sampling.penalty_last_n},{"DryPenaltyLast",p.sampling.dry_penalty_last_n},{"LoraPaths",lora},{"OutputsMax",p.n_outputs_max},{"OutputsMaxPerSequence",p.n_outputs_max_per_seq}};}
inline json exception(const std::exception&e){return json{{"Kind","standard"},{"Message",std::string(e.what()).substr(0,4096)}};}
inline std::string hex(const std::vector<char>&data){require(data.size()<=8*1024*1024,"remote observation body exceeds8MiB");constexpr char digits[]="0123456789abcdef";std::string out(data.size()*2,'\0');for(std::size_t i=0;i<data.size();++i){auto b=static_cast<unsigned char>(data[i]);out[2*i]=digits[b>>4];out[2*i+1]=digits[b&15];}return out;}
inline json execute(const json&op){
    const auto kind=op.at("Kind").get<std::string>();
    if(kind=="argument-enums")return json{{"Count",LLAMA_EXAMPLE_COUNT},{"Common",LLAMA_EXAMPLE_COMMON},{"Speculative",LLAMA_EXAMPLE_SPECULATIVE},{"Embedding",LLAMA_EXAMPLE_EMBEDDING}};
    if(kind=="remote-get-content"){
        common_remote_params params;params.timeout=integer(op.at("TimeoutSeconds"),0,60);params.max_size=integer(op.at("MaxSize"),0,8*1024*1024);for(const auto&h:op.at("Headers"))params.headers.emplace_back(h.at("Name").get<std::string>(),h.at("Value").get<std::string>());
        try{const auto value=common_remote_get_content(op.at("Url").get<std::string>(),params);return json{{"HttpCode",value.first},{"BodyHex",hex(value.second)},{"Exception",nullptr}};}catch(const std::exception&e){return json{{"HttpCode",nullptr},{"BodyHex",nullptr},{"Exception",exception(e)}};}
    }
    std::vector<std::unique_ptr<common_params>>params;for(const auto&initial:op.at("Initials")){auto value=std::make_unique<common_params>();if(initial.contains("Parallel"))value->n_parallel=initial.at("Parallel").get<int>();if(initial.contains("OutputsMaxPerSequence"))value->n_outputs_max_per_seq=initial.at("OutputsMaxPerSequence").get<int>();params.push_back(std::move(value));}
    json trace=json::array();for(const auto&a:op.at("Actions")){
        try{
            auto action=a.at("Action").get<std::string>();
            if(action=="limits"){auto limits=common_speculative_get_output_limits(a.at("Batch").get<std::int32_t>(),a.at("Parallel").get<std::int32_t>(),a.at("Draft").get<std::int32_t>());trace.push_back(json{{"Total",limits.total},{"PerSequence",limits.per_seq}});continue;}
            auto&p=*params.at(slot(a.at("Slot"),params.size()));
            if(action=="snapshot")trace.push_back(json{{"Params",snapshot(p)}});
            else if(action=="speculative")trace.push_back(json{{"Params",snapshot(common_base_params_to_speculative(p))}});
            else if(action=="inventory"){
                auto context=common_params_parser_init(p,static_cast<llama_example>(a.at("Example").get<int>()));common_params_add_preset_options(context.options);json options=json::array();for(const auto&o:context.options){json positive=json::array(),negative=json::array();for(const auto*arg:o.args)positive.push_back(arg);for(const auto*arg:o.args_neg)negative.push_back(arg);options.push_back(json{{"Args",positive},{"NegatedArgs",negative},{"AllArgs",o.get_args()},{"Environment",o.get_env()},{"Speculative",o.is_spec}});}trace.push_back(json{{"Options",options}});
            }else{
                std::vector<std::string>storage;for(const auto&v:a.at("Arguments"))storage.push_back(v.get<std::string>());std::vector<char*>argv;for(auto&s:storage)argv.push_back(s.data());bool parsed=common_params_parse(static_cast<int>(argv.size()),argv.data(),p,static_cast<llama_example>(a.at("Example").get<int>()));trace.push_back(json{{"Parsed",parsed},{"Params",snapshot(p)}});
            }
        }catch(const std::exception&e){trace.push_back(json{{"Exception",exception(e)}});break;}
    }
    return json{{"Trace",trace}};
}
} // namespace prism_arg_parser
