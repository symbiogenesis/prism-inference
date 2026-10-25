// Generic production template observations; source schedules and verdicts are Rust.
#pragma once
#include "chat.h"
#include "chat-auto-parser.h"
#include "chat-auto-parser-helpers.h"
#include "jinja/runtime.h"
#include <nlohmann/json.hpp>
#include <chrono>
#include <cstdint>
#include <initializer_list>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace prism_chat_analysis {
using json=nlohmann::ordered_json;
inline void require(bool value,const char*message){if(!value)throw std::runtime_error(message);}
inline void fields(const json&value,std::initializer_list<const char*>names){require(value.is_object()&&value.size()==names.size(),"chat observation field mismatch");for(const auto*name:names)require(value.contains(name),"chat observation field missing");}
inline const std::string&text(const json&value,std::size_t bound){require(value.is_string(),"chat observation string required");const auto&s=value.get_ref<const std::string&>();require(s.size()<=bound&&s.find('\0')==std::string::npos,"chat observation string bound/NUL violation");return s;}
inline std::size_t index(const json&value,std::size_t limit){require(value.is_number_integer()&&(value.is_number_unsigned()||value.get<std::int64_t>()>=0),"chat observation index must be nonnegative");auto n=value.get<std::uint64_t>();require(n<limit,"chat observation index outside prior trace");return static_cast<std::size_t>(n);}
inline common_json object(const json&value){const auto&s=text(value,256*1024);auto result=common_json::parse(s);require(result.is_object(),"chat parameters/globals must be object");return result;}
inline std::string hex(const std::string&value){require(value.size()<=1024*1024,"chat observation output exceeds1MiB");constexpr char digits[]="0123456789abcdef";std::string result(value.size()*2,'\0');for(std::size_t i=0;i<value.size();++i){auto c=static_cast<unsigned char>(value[i]);result[2*i]=digits[c>>4];result[2*i+1]=digits[c&15];}return result;}
inline bool handles(const std::string&kind){return kind=="chat-template-trace";}
inline void validate(const json&operation){
    fields(operation,{"Kind","Template","Bos","Eos","StopOnError","Actions"});
    require(operation.at("Kind")=="chat-template-trace","unknown chat observation kind");
    text(operation.at("Template"),1024*1024);text(operation.at("Bos"),65536);text(operation.at("Eos"),65536);
    require(operation.at("StopOnError").is_boolean(),"StopOnError must be boolean");const auto&actions=operation.at("Actions");require(actions.is_array()&&actions.size()<=128,"chat observation actions exceed128");
    for(std::size_t i=0;i<actions.size();++i){const auto&action=actions[i];require(action.is_object()&&action.contains("Action"),"chat observation Action required");auto name=text(action.at("Action"),64);
        if(name=="caps")fields(action,{"Action"});
        else if(name=="apply"){fields(action,{"Action","ParametersJson","NowUnixSeconds"});index(action.at("NowUnixSeconds"),4102444801ULL);auto p=object(action.at("ParametersJson"));for(const auto&item:p.items()){auto key=item.key();require(key=="messages"||key=="tools"||key=="add_generation_prompt"||key=="enable_thinking","unknown direct-apply parameter");if(key=="add_generation_prompt"||key=="enable_thinking")require(item.value().is_boolean(),"direct-apply flag must be boolean");}}
        else if(name=="diff"){fields(action,{"Action","Left","Right"});for(const auto*key:{"Left","Right"}){auto prior=index(action.at(key),i);require(actions.at(prior).at("Action")=="apply","diff reference must name prior apply");}}
        else if(name=="undefined-probe"){fields(action,{"Action","VariablesJson","Names"});object(action.at("VariablesJson"));const auto&names=action.at("Names");require(names.is_array()&&names.size()<=128,"undefined probe names exceed128");std::set<std::string>seen;for(const auto&value:names)require(seen.insert(text(value,256)).second,"duplicate undefined probe name");}
        else throw std::runtime_error("unknown chat observation action");
    }
}
inline json caps(const chat_template_caps&c){return json{{"supports_tools",c.supports_tools},{"supports_tool_calls",c.supports_tool_calls},{"supports_system_role",c.supports_system_role},{"supports_parallel_tool_calls",c.supports_parallel_tool_calls},{"supports_typed_content",c.supports_typed_content},{"supports_string_content",c.supports_string_content}};}
inline json error(const std::exception&e){return json{{"Kind","standard"},{"Message",std::string(e.what()).substr(0,4096)}};}
inline json execute(const json&operation){
    json trace=json::array();
    try{
        common_chat_template tmpl(operation.at("Template").get<std::string>(),operation.at("Bos").get<std::string>(),operation.at("Eos").get<std::string>());
        std::vector<std::string>rendered(operation.at("Actions").size());
        for(std::size_t i=0;i<operation.at("Actions").size();++i){const auto&action=operation.at("Actions").at(i);auto name=action.at("Action").get<std::string>();
            try{
                if(name=="caps")trace.push_back(json{{"Caps",caps(tmpl.original_caps())}});
                else if(name=="apply"){
                    auto values=object(action.at("ParametersJson"));autoparser::generation_params params;
                    params.now=std::chrono::system_clock::time_point(std::chrono::duration_cast<std::chrono::system_clock::duration>(std::chrono::seconds(action.at("NowUnixSeconds").get<std::int64_t>())));
                    if(values.contains("messages"))params.messages=values.at("messages");if(values.contains("tools"))params.tools=values.at("tools");if(values.contains("add_generation_prompt"))params.add_generation_prompt=values.at("add_generation_prompt").get<bool>();if(values.contains("enable_thinking"))params.enable_thinking=values.at("enable_thinking").get<bool>();
                    rendered[i]=common_chat_template_direct_apply(tmpl,params);trace.push_back(json{{"RenderedHex",hex(rendered[i])}});
                }else if(name=="diff"){
                    const auto value=calculate_diff_split(rendered.at(index(action.at("Left"),i)),rendered.at(index(action.at("Right"),i)));
                    trace.push_back(json{{"Diff",{{"PrefixHex",hex(value.prefix)},{"SuffixHex",hex(value.suffix)},{"LeftHex",hex(value.left)},{"RightHex",hex(value.right)}}}});
                }else{
                    jinja::context context;context.is_get_stats=true;jinja::global_from_json(context,object(action.at("VariablesJson")),true);
                    for(const auto&value:action.at("Names")){const auto name=value.get<std::string>();context.set_val(name,jinja::mk_val<jinja::value_undefined_t>(name));}
                    json runtime_error=nullptr;try{jinja::runtime runtime(context);runtime.execute(tmpl.prog);}catch(const std::exception&e){runtime_error=error(e);}
                    json variables=json::array();for(const auto&value:action.at("Names")){const auto name=value.get<std::string>();const auto observed=context.get_val(name);variables.push_back(json{{"Name",name},{"Undefined",observed->is_undefined()},{"Used",observed->stats.used}});}
                    trace.push_back(json{{"Variables",variables},{"RuntimeError",runtime_error}});
                }
            }catch(const std::exception&e){trace.push_back(json{{"ActionError",error(e)}});if(operation.at("StopOnError").get<bool>())break;}
        }
        return json{{"Trace",trace},{"TemplateError",nullptr}};
    }catch(const std::exception&e){return json{{"Trace",trace},{"TemplateError",error(e)}};}
}
} // namespace prism_chat_analysis
