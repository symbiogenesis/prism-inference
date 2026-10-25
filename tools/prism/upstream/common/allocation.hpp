// Generic graph/buffer input construction and production gallocr observations.
#pragma once
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include <nlohmann/json.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace prism_allocation {
using json = nlohmann::ordered_json;
inline void require(bool value, const char * message) { if (!value) throw std::runtime_error(message); }
inline void keys(const json & value, std::initializer_list<const char *> names) {
    require(value.is_object() && value.size() == names.size(), "allocation object fields mismatch");
    for (const auto * name : names) require(value.contains(name), "missing allocation field");
}
inline std::size_t integer(const json & value, std::size_t maximum) {
    require(value.is_number_integer() && (value.is_number_unsigned() || value.get<std::int64_t>() >= 0), "allocation nonnegative integer required");
    const auto number = value.get<std::uint64_t>(); require(number <= maximum, "allocation integer outside bound");
    return static_cast<std::size_t>(number);
}
inline bool handles(const std::string & kind) { return kind == "allocation-graph"; }
using shape = std::array<std::size_t, 4>;
inline std::size_t elements(const shape & dimensions) {
    std::size_t product = 1;
    for (auto dimension : dimensions) { require(dimension > 0 && dimension <= 1024 && product <= 16384 / dimension, "allocation tensor extent exceeds bound"); product *= dimension; }
    return product;
}
inline void validate(const json & operation) {
    keys(operation, {"Kind", "Backends", "Steps"});
    require(operation.at("Kind").is_string() && handles(operation.at("Kind").get<std::string>()), "unknown allocation Kind");
    const auto & backends = operation.at("Backends");
    require(backends.is_array() && !backends.empty() && backends.size() <= 3, "allocation backend count outside bound");
    for (const auto & backend : backends) {
        keys(backend, {"MaxBufferSize", "Alignment"});
        const auto alignment = integer(backend.at("Alignment"), 64);
        require(alignment > 0 && (alignment & (alignment - 1)) == 0, "allocation alignment must be a power of two");
        if (!backend.at("MaxBufferSize").is_null()) require(integer(backend.at("MaxBufferSize"), 1024 * 1024) > 0, "allocation maximum buffer size must be positive");
    }
    const auto & steps = operation.at("Steps");
    require(steps.is_array() && !steps.empty() && steps.size() <= 4, "allocation step count outside bound");
    for (const auto & step : steps) {
        keys(step, {"Nodes", "Output", "BufferTypes", "ExternalAllocations", "Reserve"});
        const auto & nodes = step.at("Nodes");
        require(nodes.is_array() && !nodes.empty() && nodes.size() <= 40, "allocation graph node count outside bound");
        integer(step.at("Output"), nodes.size() - 1);
        require(step.at("Reserve").is_boolean(), "allocation Reserve must be boolean");
        require(step.at("BufferTypes").is_array() && step.at("BufferTypes").size() == nodes.size(), "allocation buffer type mapping extent mismatch");
        for (const auto & type : step.at("BufferTypes")) integer(type, backends.size() - 1);
        require(step.at("ExternalAllocations").is_array() && step.at("ExternalAllocations").size() <= 2, "allocation external context count outside bound");
        std::array<bool, 3> external{};
        for (const auto & item : step.at("ExternalAllocations")) {
            keys(item, {"Context", "BufferType"});
            const auto context = integer(item.at("Context"), 2);
            require(context > 0 && !external[context], "external allocation context must be unique and nonzero");
            external[context] = true; integer(item.at("BufferType"), backends.size() - 1);
        }
        std::vector<shape> shapes;
        for (std::size_t index = 0; index < nodes.size(); ++index) {
            const auto & node = nodes.at(index);
            require(node.is_object() && node.contains("Op") && node.at("Op").is_string() && node.contains("Context"), "allocation node Op/Context required");
            const auto context = integer(node.at("Context"), 2);
            require(context == 0 || external[context], "allocation node belongs to unallocated external context");
            const auto op = node.at("Op").get<std::string>();
            shape dimensions{1,1,1,1};
            if (op == "input") {
                keys(node, {"Op", "Context", "Elements"}); dimensions[0] = integer(node.at("Elements"), 1024);
            } else {
                require(context == 0 && node.contains("Inputs") && node.at("Inputs").is_array(), "allocation operations require context zero and Inputs");
                const auto binary = op == "add" || op == "mul" || op == "concat";
                require(node.at("Inputs").size() == (binary ? 2 : 1), "allocation node input count mismatch");
                require(index > 0, "allocation node references require an earlier tensor");
                const auto source = integer(node.at("Inputs").at(0), index - 1);
                dimensions = shapes.at(source);
                if (binary) {
                    keys(node, {"Op", "Context", "Inputs"});
                    const auto right = shapes.at(integer(node.at("Inputs").at(1), index - 1));
                    if (op == "concat") { for (std::size_t axis = 1; axis < 4; ++axis) require(dimensions[axis] == right[axis], "allocation concat dimensions mismatch"); dimensions[0] += right[0]; }
                    else for (std::size_t axis = 0; axis < 4; ++axis) require(dimensions[axis] == right[axis], "allocation elementwise dimensions mismatch");
                } else if (op == "mean") { keys(node, {"Op", "Context", "Inputs"}); dimensions[0] = 1; }
                else if (op == "scale") { keys(node, {"Op", "Context", "Inputs", "Factor"}); require(node.at("Factor").is_number() && std::isfinite(node.at("Factor").get<float>()), "allocation scale Factor must be finite"); }
                else if (op == "pad") { keys(node, {"Op", "Context", "Inputs", "Amount"}); dimensions[0] += integer(node.at("Amount"), 1024); }
                else if (op == "reshape") {
                    keys(node, {"Op", "Context", "Inputs", "Dimensions"});
                    const auto & values = node.at("Dimensions"); require(values.is_array() && values.size() == 2, "allocation reshape requires two dimensions");
                    const auto count = elements(dimensions); dimensions = {integer(values.at(0),1024),integer(values.at(1),1024),1,1};
                    require(elements(dimensions) == count, "allocation reshape element count mismatch");
                } else if (op == "permute") {
                    keys(node, {"Op", "Context", "Inputs", "Axes"});
                    const auto & axes = node.at("Axes"); require(axes.is_array() && axes.size() == 4, "allocation permutation requires four axes");
                    std::array<bool,4> seen{}; const auto original = dimensions;
                    for (std::size_t axis=0;axis<4;++axis) { const auto destination=integer(axes.at(axis),3); require(!seen[destination], "allocation permutation axes must be distinct"); seen[destination]=true; dimensions[destination]=original[axis]; }
                } else if (op == "view") {
                    keys(node, {"Op", "Context", "Inputs", "Elements", "OffsetBytes"});
                    const auto count=integer(node.at("Elements"),1024), offset=integer(node.at("OffsetBytes"),65536);
                    require(offset % 4 == 0 && count > 0 && offset + count*4 <= elements(dimensions)*4, "allocation view exceeds source extent");
                    dimensions={count,1,1,1};
                } else throw std::runtime_error("unknown allocation node Op");
            }
            elements(dimensions); shapes.push_back(dimensions);
        }
    }
}

struct backend;
struct payload { backend * owner; std::vector<std::uint8_t> storage; void * base; };
struct record { std::size_t id, size; };
struct backend {
    ggml_backend_buffer_type type{};
    ggml_backend_buffer_i buffer_interface{};
    std::size_t maximum, alignment, next_id=0;
    std::map<ggml_backend_buffer_t, record> buffers;
    explicit backend(const json & input);
};
inline const char * name(ggml_backend_buffer_type_t) { return "supplied_allocation_buffer"; }
inline void free_buffer(ggml_backend_buffer_t buffer) { auto * data=static_cast<payload *>(buffer->context); data->owner->buffers.erase(buffer); delete data; }
inline void * base(ggml_backend_buffer_t buffer) { return static_cast<payload *>(buffer->context)->base; }
inline ggml_status init_tensor(ggml_backend_buffer_t, ggml_tensor *) { return GGML_STATUS_SUCCESS; }
inline void memset_tensor(ggml_backend_buffer_t, ggml_tensor *, std::uint8_t, std::size_t, std::size_t) {}
inline void set_tensor(ggml_backend_buffer_t, ggml_tensor *, const void *, std::size_t, std::size_t) {}
inline void get_tensor(ggml_backend_buffer_t, const ggml_tensor *, void *, std::size_t, std::size_t) {}
inline void clear_buffer(ggml_backend_buffer_t, std::uint8_t) {}
inline std::size_t alignment(ggml_backend_buffer_type_t type) { return static_cast<backend *>(type->context)->alignment; }
inline std::size_t maximum(ggml_backend_buffer_type_t type) { return static_cast<backend *>(type->context)->maximum; }
inline bool is_host(ggml_backend_buffer_type_t) { return true; }
inline ggml_backend_buffer_t allocate(ggml_backend_buffer_type_t type, std::size_t size) {
    require(size <= 4*1024*1024, "production allocation exceeds adapter resource bound");
    auto * owner=static_cast<backend *>(type->context);
    auto data=std::make_unique<payload>(); data->owner=owner; data->storage.resize(size+64);
    const auto pointer=reinterpret_cast<std::uintptr_t>(data->storage.data());
    data->base=reinterpret_cast<void *>((pointer+63)&~std::uintptr_t(63));
    auto * buffer=ggml_backend_buffer_init(type,owner->buffer_interface,data.get(),size);
    require(buffer != nullptr,"production buffer creation failed");
    owner->buffers.emplace(buffer,record{owner->next_id++,size}); data.release(); return buffer;
}
inline backend::backend(const json & input) : maximum(input.at("MaxBufferSize").is_null() ? std::numeric_limits<std::size_t>::max() : input.at("MaxBufferSize").get<std::size_t>()), alignment(input.at("Alignment").get<std::size_t>()) {
    buffer_interface.free_buffer=free_buffer; buffer_interface.get_base=base; buffer_interface.init_tensor=init_tensor;
    buffer_interface.memset_tensor=memset_tensor; buffer_interface.set_tensor=set_tensor; buffer_interface.get_tensor=get_tensor; buffer_interface.clear=clear_buffer;
    type.context=this; type.iface.get_name=name; type.iface.alloc_buffer=allocate;
    type.iface.get_alignment=prism_allocation::alignment; type.iface.get_max_size=prism_allocation::maximum; type.iface.is_host=is_host;
}
struct context_free { void operator()(ggml_context * value) const { ggml_free(value); } };
struct buffer_free { void operator()(ggml_backend_buffer * value) const { ggml_backend_buffer_free(value); } };
struct allocator_free { void operator()(ggml_gallocr * value) const { ggml_gallocr_free(value); } };
using context_ptr=std::unique_ptr<ggml_context,context_free>;
using buffer_ptr=std::unique_ptr<ggml_backend_buffer,buffer_free>;
using allocator_ptr=std::unique_ptr<ggml_gallocr,allocator_free>;
inline std::size_t tensor_id(const std::vector<ggml_tensor *> & tensors, const ggml_tensor * tensor) {
    for(std::size_t i=0;i<tensors.size();++i) if(tensors[i]==tensor) return i;
    throw std::runtime_error("production graph references an unknown tensor");
}
inline json pointer_id(const std::vector<ggml_tensor *> & tensors,const ggml_tensor * tensor) { return tensor ? json(tensor_id(tensors,tensor)) : json(nullptr); }
inline json snapshot(const std::vector<ggml_tensor *> & tensors,const json & descriptions,ggml_cgraph * graph,const std::vector<std::unique_ptr<backend>> & backends,bool allocated,json reserved) {
    json buffers=json::array(); std::map<ggml_backend_buffer_t,std::size_t> ids;
    for(std::size_t type=0;type<backends.size();++type) for(const auto & entry:backends[type]->buffers) {
        const auto id=ids.size(); ids.emplace(entry.first,id);
        buffers.push_back(json{{"Id",id},{"Type",type},{"Size",entry.second.size},
            {"Maximum",backends[type]->maximum==std::numeric_limits<std::size_t>::max()?json(nullptr):json(backends[type]->maximum)}});
    }
    json values=json::array();
    for(std::size_t index=0;index<tensors.size();++index) {
        const auto * tensor=tensors[index]; json buffer=nullptr,offset=nullptr;
        if(tensor->buffer) {
            const auto found=ids.find(tensor->buffer); require(found!=ids.end(),"production tensor uses unknown buffer"); buffer=found->second;
            if(tensor->data) {
                const auto address=reinterpret_cast<std::uintptr_t>(tensor->data), origin=reinterpret_cast<std::uintptr_t>(ggml_backend_buffer_get_base(tensor->buffer));
                // Preserve an out-of-bounds value as an observation. Rust owns
                // the actual bounds assertion, including negative offsets.
                offset=address>=origin?json(address-origin):json(-static_cast<std::int64_t>(origin-address));
            }
        }
        json sources=json::array();for(int s=0;s<GGML_MAX_SRC;++s)sources.push_back(pointer_id(tensors,tensor->src[s]));
        values.push_back(json{{"Index",index},{"Context",descriptions.at(index).at("Context")},{"Buffer",buffer},{"DataPresent",tensor->data!=nullptr},
            {"Offset",offset},{"Bytes",ggml_nbytes(tensor)},{"ViewSource",pointer_id(tensors,tensor->view_src)},
            {"Sources",sources},{"Op",static_cast<std::uint32_t>(tensor->op)},{"Flags",static_cast<std::uint32_t>(tensor->flags)},
            {"CanInplace",ggml_op_can_inplace(tensor->op)},{"Output",(tensor->flags&GGML_TENSOR_FLAG_OUTPUT)!=0}});
    }
    json nodes=json::array(),leaves=json::array();
    for(int i=0;i<ggml_graph_n_nodes(graph);++i)nodes.push_back(tensor_id(tensors,ggml_graph_node(graph,i)));
    for(int i=0;i<graph->n_leafs;++i)leaves.push_back(tensor_id(tensors,graph->leafs[i]));
    return json{{"Allocated",allocated},{"Reserved",reserved},{"Tensors",values},{"Buffers",buffers},{"GraphNodes",nodes},{"GraphLeaves",leaves}};
}
inline json execute(const json & operation) {
    std::vector<std::unique_ptr<backend>> backends; std::vector<ggml_backend_buffer_type_t> types;
    for(const auto & item:operation.at("Backends")){backends.emplace_back(std::make_unique<backend>(item));types.push_back(&backends.back()->type);}
    allocator_ptr allocator(types.size()==1?ggml_gallocr_new(types[0]):ggml_gallocr_new_n(types.data(),types.size()));
    require(bool(allocator),"production gallocr creation failed");json observations=json::array();
    for(const auto & step:operation.at("Steps")) {
        std::vector<context_ptr> contexts;
        for(int i=0;i<3;++i){ggml_init_params params{};params.no_alloc=true;params.mem_size=48*ggml_tensor_overhead()+ggml_graph_overhead();contexts.emplace_back(ggml_init(params));require(bool(contexts.back()),"production graph context creation failed");}
        auto * graph=ggml_new_graph(contexts[0].get());std::vector<ggml_tensor *> tensors;
        for(const auto & node:step.at("Nodes")) {
            auto * context=contexts.at(node.at("Context").get<std::size_t>()).get(); const auto op=node.at("Op").get<std::string>();ggml_tensor * tensor=nullptr;
            if(op=="input"){tensor=ggml_new_tensor_1d(context,GGML_TYPE_F32,node.at("Elements").get<std::int64_t>());ggml_set_input(tensor);}
            else { auto * source=tensors.at(node.at("Inputs").at(0).get<std::size_t>());
                if(op=="add"||op=="mul"||op=="concat"){auto * right=tensors.at(node.at("Inputs").at(1).get<std::size_t>());tensor=op=="add"?ggml_add(context,source,right):op=="mul"?ggml_mul(context,source,right):ggml_concat(context,source,right,0);}
                else if(op=="mean")tensor=ggml_mean(context,source);
                else if(op=="scale")tensor=ggml_scale(context,source,node.at("Factor").get<float>());
                else if(op=="pad")tensor=ggml_pad(context,source,node.at("Amount").get<int>(),0,0,0);
                else if(op=="reshape")tensor=ggml_reshape_2d(context,source,node.at("Dimensions").at(0).get<std::int64_t>(),node.at("Dimensions").at(1).get<std::int64_t>());
                else if(op=="permute")tensor=ggml_permute(context,source,node.at("Axes").at(0).get<int>(),node.at("Axes").at(1).get<int>(),node.at("Axes").at(2).get<int>(),node.at("Axes").at(3).get<int>());
                else tensor=ggml_view_1d(context,source,node.at("Elements").get<std::int64_t>(),node.at("OffsetBytes").get<std::size_t>());
            }
            require(tensor!=nullptr,"production tensor creation failed");ggml_format_name(tensor,"tensor%zu",tensors.size());tensors.push_back(tensor);
        }
        std::vector<buffer_ptr> external;
        for(const auto & item:step.at("ExternalAllocations")){external.emplace_back(ggml_backend_alloc_ctx_tensors_from_buft(contexts.at(item.at("Context").get<std::size_t>()).get(),types.at(item.at("BufferType").get<std::size_t>())));require(bool(external.back()),"production external context allocation failed");}
        auto * output=tensors.at(step.at("Output").get<std::size_t>());ggml_set_output(output);ggml_build_forward_expand(graph,output);
        json reserved=nullptr;
        if(step.at("Reserve").get<bool>()) {
            std::vector<int> node_types,leaf_types;
            for(int i=0;i<ggml_graph_n_nodes(graph);++i)node_types.push_back(step.at("BufferTypes").at(tensor_id(tensors,ggml_graph_node(graph,i))).get<int>());
            for(int i=0;i<graph->n_leafs;++i)leaf_types.push_back(step.at("BufferTypes").at(tensor_id(tensors,graph->leafs[i])).get<int>());
            reserved=ggml_gallocr_reserve_n(allocator.get(),graph,node_types.data(),leaf_types.data());
        }
        const bool allocated=ggml_gallocr_alloc_graph(allocator.get(),graph);
        observations.push_back(snapshot(tensors,step.at("Nodes"),graph,backends,allocated,reserved));
    }
    return json{{"Steps",observations}};
}
} // namespace prism_allocation
