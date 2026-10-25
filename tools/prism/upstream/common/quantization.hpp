#pragma once
// Supplied metadata/tensor descriptors and production quantization APIs only.
// Filtering, sorting, model catalogs, ftype names and expected schemas are Rust.
#include "llama-ext.h"
#include "llama-arch.h"
#include "llama-hparams.h"
#include "ggml.h"
#include <nlohmann/json.hpp>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace prism_quantization {
using json = nlohmann::ordered_json;
inline void require(bool value,const char * message) { if (!value) throw std::runtime_error(message); }
inline void fields(const json & value,std::initializer_list<const char *> names) { require(value.is_object() && value.size()==names.size(),"quantization fields mismatch"); for (const auto * name:names) require(value.contains(name),"missing quantization field"); }
inline std::uint64_t number(const json & value,std::uint64_t maximum=UINT32_MAX) { require(value.is_number_integer() && (value.is_number_unsigned() || value.get<std::int64_t>()>=0),"quantization unsigned integer required"); const auto result=value.get<std::uint64_t>(); require(result<=maximum,"quantization integer exceeds bound"); return result; }
inline std::string text(const json & value,std::size_t maximum) { require(value.is_string(),"quantization text required"); auto result=value.get<std::string>(); require(result.size()<=maximum && result.find('\0')==std::string::npos,"quantization text exceeds bound or contains NUL"); return result; }
inline bool handles(const std::string & kind) { return kind=="quantization-info" || kind=="quantization-model"; }
inline void validate(const json & operation) {
    if (operation.at("Kind")=="quantization-info") { fields(operation,{"Kind","Ftypes"}); const auto & values=operation.at("Ftypes"); require(values.is_array() && values.size()<=LLAMA_FTYPE_GUESSED,"quantization ftype count exceeds bound"); for (const auto & value:values) number(value,LLAMA_FTYPE_GUESSED-1); return; }
    fields(operation,{"Kind","Model","Tensors","Actions"}); const auto & model=operation.at("Model");
    fields(model,{"Architecture","Embedding","FeedForward","Layers","Heads","KvHeads","Experts","HeadKey","HeadValue"});
    const auto architecture=text(model.at("Architecture"),128); require(llm_arch_from_string(architecture)!=LLM_ARCH_UNKNOWN,"unknown production quantization architecture");
    for (const auto * key:{"Embedding","FeedForward","Heads","KvHeads","Experts","HeadKey","HeadValue"}) number(model.at(key),1024*1024);
    const auto layers=number(model.at("Layers"),LLAMA_MAX_LAYERS); require(layers>0,"quantization model needs layers");
    const auto & tensors=operation.at("Tensors"); require(tensors.is_array() && !tensors.empty() && tensors.size()<=32768,"quantization tensor count exceeds bound");
    for (const auto & tensor:tensors) { fields(tensor,{"Name","Dimensions"}); text(tensor.at("Name"),1024); const auto & dims=tensor.at("Dimensions"); require(dims.is_array() && dims.size()==4,"quantization dimensions must have four entries"); std::uint64_t elements=1; for (const auto & dim:dims) { const auto size=number(dim,INT32_MAX); require(size>0 && elements<=std::uint64_t(INT64_MAX)/(4*size),"quantization tensor extent exceeds bound"); elements*=size; } }
    const auto & actions=operation.at("Actions"); require(actions.is_array() && actions.size()<=128,"quantization actions exceed bound"); std::size_t returned=0;
    for (const auto & action:actions) {
        require(action.is_object() && action.contains("Op") && action.at("Op").is_string(),"quantization action Op required");
        if (action.at("Op")=="allows") { fields(action,{"Op"}); returned+=tensors.size(); }
        else { fields(action,{"Op","Ftype","TensorIndices"}); require(action.at("Op")=="types","unknown quantization action"); const auto ftype=static_cast<llama_ftype>(number(action.at("Ftype"),LLAMA_FTYPE_GUESSED-1)); require(llama_ftype_get_default_type(ftype)!=GGML_TYPE_COUNT,"unsupported production quantization ftype"); const auto & indices=action.at("TensorIndices"); require(indices.is_array() && indices.size()<=tensors.size(),"quantization index count exceeds tensor extent"); for (const auto & index:indices) number(index,tensors.size()-1); returned+=indices.size(); }
        require(returned<=2*1024*1024,"quantization response entries exceed bound");
    }
}
struct model_deleter { void operator()(llama_model * model) const { llama_model_free(model); } };
struct quant_deleter { void operator()(quantize_state_impl * value) const { llama_quant_free(value); } };
struct context_deleter { void operator()(ggml_context * value) const { ggml_free(value); } };
inline json execute(const json & operation) {
    if (operation.at("Kind")=="quantization-info") {
        json types=json::array(),defaults=json::array();
        for (int i=0;i<GGML_TYPE_COUNT;++i) { const auto * name=ggml_type_name(static_cast<ggml_type>(i)); types.push_back({{"Type",i},{"Name",name?json(name):json(nullptr)}}); }
        for (const auto & value:operation.at("Ftypes")) defaults.push_back(static_cast<int>(llama_ftype_get_default_type(static_cast<llama_ftype>(number(value,LLAMA_FTYPE_GUESSED-1)))));
        return json{{"TypeCount",GGML_TYPE_COUNT},{"FtypeGuessed",LLAMA_FTYPE_GUESSED},{"Types",types},{"Defaults",defaults}};
    }
    const auto & source=operation.at("Model"); const auto architecture=source.at("Architecture").get<std::string>(); llama_quant_model_desc desc{};
    desc.architecture=architecture.c_str(); desc.n_embd=source.at("Embedding").get<std::uint32_t>(); desc.n_ff=source.at("FeedForward").get<std::uint32_t>(); desc.n_layer=source.at("Layers").get<std::uint32_t>(); desc.n_head=source.at("Heads").get<std::uint32_t>(); desc.n_head_kv=source.at("KvHeads").get<std::uint32_t>(); desc.n_expert=source.at("Experts").get<std::uint32_t>(); desc.n_embd_head_k=source.at("HeadKey").get<std::uint32_t>(); desc.n_embd_head_v=source.at("HeadValue").get<std::uint32_t>();
    std::unique_ptr<llama_model,model_deleter> model(llama_quant_model_from_metadata(&desc)); require(bool(model),"production quantization model construction failed");
    auto params=llama_model_quantize_default_params(); std::unique_ptr<quantize_state_impl,quant_deleter> state(llama_quant_init(model.get(),&params)); require(bool(state),"production quantization state construction failed");
    const auto & supplied=operation.at("Tensors"); std::unique_ptr<ggml_context,context_deleter> context(ggml_init({supplied.size()*ggml_tensor_overhead(),nullptr,true})); require(bool(context),"production quantization tensor context creation failed");
    std::vector<ggml_tensor *> tensors;
    for (const auto & item:supplied) { const auto & dims=item.at("Dimensions"); auto * tensor=ggml_new_tensor_4d(context.get(),GGML_TYPE_F32,dims[0].get<std::int64_t>(),dims[1].get<std::int64_t>(),dims[2].get<std::int64_t>(),dims[3].get<std::int64_t>()); ggml_set_name(tensor,item.at("Name").get<std::string>().c_str()); tensors.push_back(tensor); }
    json results=json::array();
    for (const auto & action:operation.at("Actions")) {
        if (action.at("Op")=="allows") { json allowed=json::array(),names=json::array(); for (const auto * tensor:tensors) { allowed.push_back(llama_quant_tensor_allows_quantization(state.get(),tensor)); names.push_back(ggml_get_name(tensor)); } results.push_back({{"Allowed",allowed},{"Names",names}}); }
        else { const auto ftype=static_cast<llama_ftype>(number(action.at("Ftype"),LLAMA_FTYPE_GUESSED-1)); std::vector<ggml_tensor *> selected; for (const auto & index:action.at("TensorIndices")) { auto * tensor=tensors[std::size_t(number(index,tensors.size()-1))]; require(llama_quant_tensor_allows_quantization(state.get(),tensor),"quantization compute requires quantizable tensors"); selected.push_back(tensor); } std::vector<ggml_type> assigned(selected.size()); llama_quant_compute_types(state.get(),ftype,selected.data(),assigned.data(),selected.size()); json encoded=json::array(); for (const auto type:assigned) encoded.push_back(static_cast<int>(type)); results.push_back({{"DefaultType",static_cast<int>(llama_ftype_get_default_type(ftype))},{"Types",encoded}}); }
    }
    return json{{"Created",true},{"Results",results}};
}
} // namespace prism_quantization
