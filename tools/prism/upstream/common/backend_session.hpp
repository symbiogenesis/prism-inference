// Persistent transport only. Every geometry owns a fresh production state;
// action generation, graph schedules and numerical verdicts remain in Rust.
#pragma once
#include "backend_ops.hpp"
#include <cstdio>
#include <memory>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace prism_backend_session {
using prism_backend_ops::bounded;
using prism_backend_ops::json;
using prism_backend_ops::keys;
using prism_backend_ops::require;
using prism_backend_ops::state;
using prism_backend_ops::text;
using prism_backend_ops::u64;
constexpr std::size_t max_request = 128 * 1024 * 1024;
constexpr std::size_t max_response = 16 * 1024 * 1024;

inline json read_frame() {
    unsigned char header[4];
    require(std::fread(header, 1, 4, stdin) == 4, "backend session truncated frame header");
    const std::uint32_t length = std::uint32_t(header[0]) | (std::uint32_t(header[1]) << 8)
        | (std::uint32_t(header[2]) << 16) | (std::uint32_t(header[3]) << 24);
    require(length > 0 && length <= max_request, "backend session request frame outside bound");
    std::string bytes(length, '\0');
    require(std::fread(bytes.data(), 1, length, stdin) == length, "backend session truncated frame");
    std::vector<std::set<std::string>> object_keys;
    const auto callback = [&](int depth, json::parse_event_t event, json & parsed) {
        require(depth <= 160, "backend session nesting exceeds limit");
        if (event == json::parse_event_t::object_start) object_keys.emplace_back();
        else if (event == json::parse_event_t::key)
            require(!object_keys.empty() && object_keys.back().insert(parsed.get<std::string>()).second,
                "backend session duplicate key");
        else if (event == json::parse_event_t::object_end) object_keys.pop_back();
        return true;
    };
    return json::parse(bytes, callback);
}

inline void write_frame(const json & value) {
    const auto bytes = value.dump();
    require(!bytes.empty() && bytes.size() <= max_response, "backend session response exceeds16MiB");
    const auto size = std::uint32_t(bytes.size());
    const unsigned char header[4] = {static_cast<unsigned char>(size), static_cast<unsigned char>(size >> 8),
        static_cast<unsigned char>(size >> 16), static_cast<unsigned char>(size >> 24)};
    require(std::fwrite(header, 1, 4, stdout) == 4
        && std::fwrite(bytes.data(), 1, bytes.size(), stdout) == bytes.size()
        && std::fflush(stdout) == 0, "backend session response write failed");
}

inline int serve() {
    std::unique_ptr<state> current;
    std::uint64_t sequence = 0;
    std::size_t cases = 0, total = 0;
    for (;;) {
        const auto request = read_frame();
        keys(request, {"Sequence", "Operation"});
        require(u64(request.at("Sequence")) == ++sequence, "backend session sequence differs");
        const auto & operation = request.at("Operation");
        const auto op = text(operation.at("Op"));
        json result = json::object();
        bool done = false;
        if (op == "open") {
            keys(operation, {"Op", "Directory", "MaxFileBytes"});
            require(!current && cases++ < 22817, "backend session overlapping or excessive geometry scopes");
            bounded(operation.at("MaxFileBytes"), 1, std::size_t(4) * 1024 * 1024 * 1024);
            require(std::filesystem::path(text(operation.at("Directory"))).is_absolute(), "backend Directory must be absolute");
            current = std::make_unique<state>(operation);
            // Preserve the file protocol's initial/per-batch checkpoint slots
            // in its cumulative action bound and diagnostic action indices.
            total = 1;
        } else if (op == "actions") {
            keys(operation, {"Op", "Actions"});
            require(bool(current), "backend session has no geometry scope");
            const auto & actions = operation.at("Actions");
            require(actions.is_array() && !actions.empty() && total < 524288 && actions.size() < 524288 - total,
                "backend session cumulative action count exceeds bound");
            auto trace = json::array();
            for (const auto & action : actions) {
                try {
                    require(action.is_object() && action.contains("Op") && action.at("Op").is_string(), "backend action Op required");
                    trace.push_back(current->run(action));
                } catch (const std::exception & error) {
                    throw std::runtime_error("backend action " + std::to_string(total) + ": " + error.what());
                }
                ++total;
            }
            ++total;
            result = {{"Trace", trace}};
        } else if (op == "close") {
            keys(operation, {"Op"});
            require(bool(current), "backend session has no geometry scope to close");
            current.reset(); // Free every buffer, backend and arena before acknowledging.
        } else if (op == "shutdown") {
            keys(operation, {"Op"});
            require(!current, "backend session shutdown has an open geometry scope");
            done = true;
        } else throw std::invalid_argument("backend session unknown operation");
        write_frame({{"Sequence", sequence}, {"Result", result}});
        if (done) return 0;
    }
}
} // namespace prism_backend_session
