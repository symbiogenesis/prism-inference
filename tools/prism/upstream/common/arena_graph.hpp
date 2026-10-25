// Generic production CPU backend graphs with caller-supplied buffer placement.
// Graphs, byte offsets, input generators and comparisons belong to safe Rust.
#pragma once
#include "tensor_graph.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <memory>

namespace prism_arena {
using namespace prism_tensor_graph;
inline bool handles(const std::string & kind) { return kind == "cpu-buffer-info" || kind == "cpu-arena-graph"; }
inline void validate(const json & operation) {
    if (operation.at("Kind") == "cpu-buffer-info") { keys(operation, {"Kind"}); return; }
    keys(operation, {"Kind", "Graph", "Forward", "Allocation"});
    const auto shapes = plan(operation.at("Graph"));
    const auto & nodes = operation.at("Graph").at("Nodes");
    for (const auto & node : nodes) {
        const auto op = node.at("Op").get<std::string>();
        require(op == "input" || op == "mul" || op == "add" || op == "view_2d", "unsupported arena graph operation");
        if (op == "input") require(node.at("Type") == "F32", "arena inputs must be F32");
    }
    const auto & forward = operation.at("Forward");
    require(forward.is_array() && !forward.empty() && forward.size() <= nodes.size(), "arena forward list outside bound");
    for (const auto & index : forward) integer(index, 0, static_cast<std::int64_t>(nodes.size()-1));
    const auto & allocation = operation.at("Allocation");
    if (!allocation.is_null()) {
        keys(allocation, {"BufferBytes", "Offsets"});
        const auto size = integer(allocation.at("BufferBytes"), 1, max_storage);
        const auto & offsets = allocation.at("Offsets");
        require(offsets.is_array() && offsets.size() == nodes.size(), "arena placement count mismatch");
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            if (nodes[i].at("Op") == "view_2d") require(offsets[i].is_null(), "view storage must derive from its source");
            else require(std::uint64_t(integer(offsets[i], 0, size)) + shapes[i].bytes() <= std::uint64_t(size), "arena placement exceeds buffer");
        }
    }
}
inline json execute(const json & operation) {
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
    require(bool(backend), "CPU backend initialization failed");
    const auto buffer_type = ggml_backend_get_default_buffer_type(backend.get());
    const auto alignment = ggml_backend_buft_get_alignment(buffer_type);
    require(alignment > 0, "CPU buffer alignment is zero");
    if (operation.at("Kind") == "cpu-buffer-info") return json{{"Alignment", alignment}};
    const auto & graph = operation.at("Graph");
    const auto shapes = plan(graph);
    const auto & nodes = graph.at("Nodes");
    const ggml_init_params params = {ggml_tensor_overhead() * (nodes.size() + 8) + ggml_graph_overhead(), nullptr, true};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(params), ggml_free);
    require(bool(context), "arena graph context allocation failed");
    auto * forward = ggml_new_graph(context.get());
    std::vector<ggml_tensor *> tensors;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const auto & node = nodes[i];
        const auto op = node.at("Op").get<std::string>();
        const auto & shape = shapes[i];
        ggml_tensor * tensor = nullptr;
        if (op == "input") tensor = ggml_new_tensor(context.get(), shape.type, static_cast<int>(node.at("Dimensions").size()), shape.ne.data());
        else {
            auto * a = tensors[node.at("Inputs")[0].get<std::size_t>()];
            if (op == "view_2d") tensor = ggml_view_2d(context.get(), a, shape.ne[0], shape.ne[1], node.at("RowStrideBytes").get<std::size_t>(), node.at("OffsetBytes").get<std::size_t>());
            else {
                auto * b = tensors[node.at("Inputs")[1].get<std::size_t>()];
                tensor = op == "mul" ? ggml_mul(context.get(), a, b) : ggml_add(context.get(), a, b);
            }
        }
        require(tensor != nullptr, "arena tensor construction failed");
        tensors.push_back(tensor);
    }
    for (const auto & index : operation.at("Forward")) ggml_build_forward_expand(forward, tensors[index.get<std::size_t>()]);
    ggml_backend_buffer_t allocated = nullptr;
    const auto & allocation = operation.at("Allocation");
    if (allocation.is_null()) allocated = ggml_backend_alloc_ctx_tensors(context.get(), backend.get());
    else allocated = ggml_backend_alloc_buffer(backend.get(), allocation.at("BufferBytes").get<std::size_t>());
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(allocated, ggml_backend_buffer_free);
    require(bool(buffer), "arena backend buffer allocation failed");
    if (!allocation.is_null()) {
        auto * base = static_cast<char *>(ggml_backend_buffer_get_base(buffer.get()));
        require(base != nullptr, "arena CPU buffer base is null");
        const auto & offsets = allocation.at("Offsets");
        for (std::size_t i = 0; i < nodes.size(); ++i) if (!offsets[i].is_null()) {
            const auto offset = offsets[i].get<std::size_t>();
            require(offset % alignment == 0, "arena placement is misaligned");
            require(offset + ggml_nbytes(tensors[i]) <= ggml_backend_buffer_get_size(buffer.get()), "arena placement exceeds production buffer");
            require(ggml_backend_tensor_alloc(buffer.get(), tensors[i], base + offset) == GGML_STATUS_SUCCESS, "arena tensor allocation failed");
        }
        for (std::size_t i = 0; i < nodes.size(); ++i) if (offsets[i].is_null()) require(ggml_backend_view_init(tensors[i]) == GGML_STATUS_SUCCESS, "arena view initialization failed");
    }
    for (std::size_t i = 0; i < nodes.size(); ++i) if (nodes[i].at("Op") == "input") {
        const auto bytes = unhex(nodes[i].at("DataHex").get_ref<const std::string &>());
        ggml_backend_tensor_set(tensors[i], bytes.data(), 0, bytes.size());
    }
    ggml_backend_cpu_set_n_threads(backend.get(), graph.at("Threads").get<int>());
    const auto status = ggml_backend_graph_compute(backend.get(), forward);
    json outputs = json::array();
    if (status == GGML_STATUS_SUCCESS) for (const auto & index : graph.at("Outputs")) {
        auto * tensor = tensors[index.get<std::size_t>()];
        std::vector<std::uint8_t> bytes(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
        outputs.push_back(json{{"DataHex", hex(bytes.data(), bytes.size())}});
    }
    return json{{"Status", static_cast<int>(status)}, {"Outputs", outputs}};
}
} // namespace prism_arena
