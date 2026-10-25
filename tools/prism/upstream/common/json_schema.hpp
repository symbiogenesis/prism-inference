#pragma once
// Production schema conversion and type-probe observations only.
#include "json-schema-to-grammar.h"
#include <nlohmann/json.hpp>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace prism_json_schema {
using json=nlohmann::ordered_json;
inline void require(bool value,const char * message){if(!value)throw std::runtime_error(message);}
inline void fields(const json & value,std::initializer_list<const char *> keys){require(value.is_object()&&value.size()==keys.size(),"schema fields mismatch");for(const auto * key:keys)require(value.contains(key),"missing or unknown schema field");}
inline void text(const json & value){require(value.is_string()&&value.get_ref<const std::string &>().size()<=262144,"schema text exceeds bound");}
inline bool handles(const std::string & kind){return kind=="json-schema-convert"||kind=="json-schema-info";}
inline void validate(const json & op){
    text(op.at("SchemaJson"));
    if(op.at("Kind")=="json-schema-convert"){fields(op,{"Kind","SchemaJson","ForceGbnf"});require(op.at("ForceGbnf").is_boolean(),"schema ForceGbnf must be boolean");}
    else{fields(op,{"Kind","SchemaJson","Actions"});const auto & actions=op.at("Actions");require(actions.is_array()&&actions.size()<=128,"schema-info actions exceed bound");for(const auto & action:actions){text(action);require(action=="resolve-refs"||action=="resolves-to-string","unknown schema-info action");}}
}
inline json execute(const json & op){
    if(op.at("Kind")=="json-schema-convert"){
        json value{{"Grammar",nullptr},{"ExceptionKind",nullptr},{"Message",""}};
        try{value["Grammar"]=json_schema_to_grammar(common_json::parse(op.at("SchemaJson").get<std::string>()),op.at("ForceGbnf").get<bool>());}
        catch(const std::invalid_argument & error){value["ExceptionKind"]="invalid-argument";value["Message"]=error.what();}
        catch(const std::runtime_error & error){value["ExceptionKind"]="runtime";value["Message"]=error.what();}
        catch(const std::exception & error){value["ExceptionKind"]="standard";value["Message"]=error.what();}
        catch(...){value["ExceptionKind"]="unknown";}
        return value;
    }
    common_schema_info info;auto schema=common_json::parse(op.at("SchemaJson").get<std::string>());json trace=json::array();
    for(const auto & action:op.at("Actions")){if(action=="resolve-refs"){info.resolve_refs(schema);trace.push_back(json{{"SchemaJson",schema.dump()}});}else{trace.push_back(json{{"ResolvesToString",info.resolves_to_string(schema)}});}}
    return json{{"Trace",trace}};
}
} // namespace prism_json_schema
