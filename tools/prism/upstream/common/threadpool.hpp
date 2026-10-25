// Caller-owned graph/plan/compute schedules; production CPU APIs only.
#pragma once
#include "ggml.h"
#include "ggml-cpu.h"
#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

namespace prism_threadpool {
using json = nlohmann::ordered_json;
inline void require(bool condition, const char * message) {
    if (!condition) throw std::invalid_argument(message);
}
inline void keys(const json & value, std::initializer_list<const char *> names) {
    require(value.is_object() && value.size() == names.size(), "threadpool object fields differ");
    for (const char * name : names) require(value.contains(name), "threadpool object field missing");
}
inline std::size_t number(const json & value, std::size_t low, std::size_t high) {
    require(value.is_number_unsigned() || value.is_number_integer(), "threadpool integer required");
    const auto n = value.get<std::int64_t>();
    require(n >= 0 && std::uint64_t(n) >= low && std::uint64_t(n) <= high, "threadpool integer outside bound");
    return static_cast<std::size_t>(n);
}
inline void validate(const json & operation) {
    keys(operation, {"Kind", "ContextBytes", "PoolThreads", "Graphs", "Actions"});
    const auto memory = number(operation.at("ContextBytes"), 1024, std::size_t(1) << 30);
    const auto threads = number(operation.at("PoolThreads"), 1, 256);
    const auto & descriptions = operation.at("Graphs");
    const auto & actions = operation.at("Actions");
    require(descriptions.is_array() && !descriptions.empty() && descriptions.size() <= 16, "threadpool graph count outside bound");
    require(actions.is_array() && !actions.empty() && actions.size() <= 25000, "threadpool action count outside bound");
    std::size_t total_nodes = 0;
    std::size_t storage_bytes = 0;
    for (const auto & graph : descriptions) {
        keys(graph, {"Nodes", "Output"});
        const auto & nodes = graph.at("Nodes");
        require(nodes.is_array() && !nodes.empty() && nodes.size() <= 8192, "threadpool node count outside bound");
        total_nodes += nodes.size();
        require(total_nodes <= 16384, "threadpool total node count outside bound");
        number(graph.at("Output"), 0, nodes.size()-1);
        std::vector<std::array<std::size_t, 2>> shapes;
        std::vector<ggml_type> types;
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            const auto & node = nodes[i];
            const auto op = node.at("Op").get<std::string>();
            if (op == "input") {
                keys(node, {"Op", "Type", "Dimensions"});
                require(node.at("Type") == "F32" || node.at("Type") == "Q4_0", "threadpool input type unsupported");
                const auto & dimensions = node.at("Dimensions");
                require(dimensions.is_array() && dimensions.size() >= 1 && dimensions.size() <= 2, "threadpool input rank outside bound");
                for (const auto & dimension : dimensions) number(dimension, 1, 1048576);
                const auto type = node.at("Type") == "F32" ? GGML_TYPE_F32 : GGML_TYPE_Q4_0;
                const std::array<std::size_t, 2> shape{number(dimensions[0], 1, 1048576), dimensions.size() == 2 ? number(dimensions[1], 1, 1048576) : 1};
                require(shape[0] % ggml_blck_size(type) == 0, "threadpool quantized row is incomplete");
                shapes.push_back(shape); types.push_back(type);
            } else {
                require(op == "mul_mat" && i > 0, "threadpool operation unsupported");
                keys(node, {"Op", "A", "B"});
                number(node.at("A"), 0, i-1); number(node.at("B"), 0, i-1);
                const auto a = node.at("A").get<std::size_t>(), b = node.at("B").get<std::size_t>();
                require(shapes[a][0] == shapes[b][0] && types[b] == GGML_TYPE_F32, "threadpool matrix dimensions or type differ");
                shapes.push_back({shapes[a][1], shapes[b][1]}); types.push_back(GGML_TYPE_F32);
            }
            const auto & shape = shapes.back();
            const auto row = ggml_row_size(types.back(), static_cast<std::int64_t>(shape[0]));
            require(row <= memory / shape[1], "threadpool tensor exceeds context bound");
            const auto bytes = row * shape[1];
            require(bytes <= memory/2 && storage_bytes <= memory/2-bytes, "threadpool data exceeds context bound");
            storage_bytes += bytes;
        }
    }
    require(total_nodes * ggml_tensor_overhead() + descriptions.size() * ggml_graph_overhead() <= memory/4, "threadpool metadata exceeds context bound");
    for (const auto & action : actions) {
        const auto op = action.at("Op").get<std::string>();
        if (op == "plan") {
            keys(action, {"Op", "Slot", "Graph", "Threads"});
            number(action.at("Slot"), 0, 15); number(action.at("Graph"), 0, descriptions.size()-1);
            number(action.at("Threads"), 1, threads);
        } else if (op == "compute") {
            keys(action, {"Op", "Slot"}); number(action.at("Slot"), 0, 15);
        } else {
            require(op == "timer-start" || op == "timer-stop", "threadpool action unsupported");
            keys(action, {"Op"});
        }
    }
}
inline json execute(const json & operation) {
    validate(operation);
    const auto memory = number(operation.at("ContextBytes"), 1024, std::size_t(1) << 30);
    const auto threads = number(operation.at("PoolThreads"), 1, 256);
    const auto & descriptions = operation.at("Graphs");
    const auto & actions = operation.at("Actions");
    const ggml_init_params parameters = {memory, nullptr, false};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(parameters), ggml_free);
    require(bool(context), "threadpool context allocation failed");
    std::vector<ggml_cgraph *> graphs;
    json node_counts = json::array();
    for (const auto & graph : descriptions) {
        auto * forward = ggml_new_graph(context.get());
        std::vector<ggml_tensor *> tensors;
        for (const auto & node : graph.at("Nodes")) {
            ggml_tensor * tensor;
            if (node.at("Op") == "input") {
                const auto type = node.at("Type") == "F32" ? GGML_TYPE_F32 : GGML_TYPE_Q4_0;
                const auto & dimensions = node.at("Dimensions");
                const std::int64_t ne[] = {dimensions[0].get<std::int64_t>(), dimensions.size() == 2 ? dimensions[1].get<std::int64_t>() : 1};
                tensor = ggml_new_tensor(context.get(), type, static_cast<int>(dimensions.size()), ne);
            } else {
                tensor = ggml_mul_mat(context.get(), tensors[node.at("A").get<std::size_t>()], tensors[node.at("B").get<std::size_t>()]);
            }
            require(tensor != nullptr, "threadpool tensor construction failed");
            tensors.push_back(tensor);
        }
        ggml_build_forward_expand(forward, tensors[graph.at("Output").get<std::size_t>()]);
        node_counts.push_back(ggml_graph_n_nodes(forward));
        graphs.push_back(forward);
    }
    auto pool_parameters = ggml_threadpool_params_default(static_cast<int>(threads));
    std::unique_ptr<ggml_threadpool, decltype(&ggml_threadpool_free)> pool(ggml_threadpool_new(&pool_parameters), ggml_threadpool_free);
    require(bool(pool), "threadpool creation failed");
    struct Plan {
        ggml_cgraph * graph = nullptr;
        ggml_cplan value{};
        std::vector<std::uint8_t> workspace;
    };
    std::array<Plan, 16> plans;
    json statuses = json::array(), times = json::array();
    std::size_t planned = 0;
    bool timer_running = false;
    std::chrono::steady_clock::time_point started;
    for (const auto & action : actions) {
        const auto op = action.at("Op").get<std::string>();
        if (op == "plan") {
            auto & plan = plans[action.at("Slot").get<std::size_t>()];
            plan.graph = graphs[action.at("Graph").get<std::size_t>()];
            plan.value = ggml_graph_plan(plan.graph, action.at("Threads").get<int>(), pool.get());
            require(plan.value.work_size <= (std::size_t(1) << 28), "threadpool workspace outside bound");
            // Each supplied plan action owns a fresh workspace, matching API lifetime.
            plan.workspace = std::vector<std::uint8_t>(plan.value.work_size);
            plan.value.work_data = plan.workspace.data();
            ++planned;
        } else if (op == "compute") {
            auto & plan = plans[action.at("Slot").get<std::size_t>()];
            require(plan.graph != nullptr, "threadpool plan used before construction");
            statuses.push_back(ggml_graph_compute(plan.graph, &plan.value));
        } else if (op == "timer-start") {
            require(!timer_running, "threadpool timer already running");
            timer_running = true; started = std::chrono::steady_clock::now();
        } else {
            require(timer_running, "threadpool timer not running");
            times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-started).count());
            timer_running = false;
        }
    }
    require(!timer_running, "threadpool timer left running");
    return json{{"GraphNodeCounts", node_counts}, {"Plans", planned}, {"ComputeStatuses", statuses}, {"Nanoseconds", times}};
}
}
