// Raw production graph reservation observations; Rust owns export policy.
#pragma once
#include "llama-ext.h"
#include "ggml-impl.h"
namespace prism_context_graph {
using json=nlohmann::ordered_json;
inline bool handles(const std::string&op){return op=="context-limits"||op=="context-graph-reserve";}
inline json metadata(const ggml_tensor*t){
    json ne=json::array(),nb=json::array();for(int i=0;i<GGML_MAX_DIMS;++i){ne.push_back(t->ne[i]);nb.push_back(t->nb[i]);}
    constexpr char digits[]="0123456789abcdef";std::string params(sizeof(t->op_params)*2,'\0');const auto*bytes=reinterpret_cast<const unsigned char*>(t->op_params);for(std::size_t i=0;i<sizeof(t->op_params);++i){params[2*i]=digits[bytes[i]>>4];params[2*i+1]=digits[bytes[i]&15];}
    return{{"Op",static_cast<int>(t->op)},{"Type",static_cast<int>(t->type)},{"NE",ne},{"NB",nb},{"OpParamsHex",params},{"Name",ggml_get_name(t)}};
}
inline json execute(const json&o,llama_context*context){
    using prism_generated_model::fields;using prism_generated_model::require;using prism_generated_model::number;
    const auto op=o.at("Op").get<std::string>();
    if(op=="context-limits"){fields(o,{"Op","Context"});return{{"SeqMax",llama_n_seq_max(context)},{"Ctx",llama_n_ctx(context)},{"Ubatch",llama_n_ubatch(context)},{"Ops",{{"NONE",GGML_OP_NONE},{"VIEW",GGML_OP_VIEW},{"RESHAPE",GGML_OP_RESHAPE},{"PERMUTE",GGML_OP_PERMUTE},{"TRANSPOSE",GGML_OP_TRANSPOSE}}}};}
    fields(o,{"Op","Context","Tokens","Sequences","Outputs"});require(op=="context-graph-reserve","unknown context graph operation");const auto tokens=number(o.at("Tokens"),1048576),sequences=number(o.at("Sequences"),65536),outputs=number(o.at("Outputs"),1048576);require(tokens>0&&sequences>0,"context graph reserve counts empty");auto*graph=llama_graph_reserve(context,static_cast<std::uint32_t>(tokens),static_cast<std::uint32_t>(sequences),static_cast<std::uint32_t>(outputs));json nodes=json::array();if(graph){require(graph->n_nodes>=0&&graph->n_nodes<=65536,"reserved graph node count exceeds bound");for(int i=0;i<graph->n_nodes;++i){const auto*t=graph->nodes[i];require(t,"reserved graph contains null tensor");auto value=metadata(t);value["Index"]=i;json sources=json::array();for(const auto*source:t->src)sources.push_back(source?metadata(source):json(nullptr));value["Sources"]=std::move(sources);nodes.push_back(std::move(value));}}
    return{{"Present",graph!=nullptr},{"Nodes",std::move(nodes)}};
}
}