// Bounded concurrent production logger operations; schedules and verdicts live in Rust.
#pragma once
#include "log.h"
#include <nlohmann/json.hpp>
#include <exception>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace prism_concurrency {
using json = nlohmann::ordered_json;
inline void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}
inline void keys(const json & value, std::initializer_list<const char *> names) {
    require(value.is_object() && value.size() == names.size(), "concurrent logger fields mismatch");
    for (const auto * name : names) require(value.contains(name), "missing concurrent logger field");
}
inline void validate(const json & operation) {
    keys(operation, {"Kind", "Workers", "Finalize"});
    require(operation.at("Kind") == "concurrent-log", "unknown concurrent operation");
    const auto & workers = operation.at("Workers");
    require(workers.is_array() && !workers.empty() && workers.size() <= 32, "logger requires 1..32 workers");
    std::size_t total = 0;
    for (const auto & actions : workers) {
        require(actions.is_array() && actions.size() <= 16384, "logger worker action bound exceeded");
        total += actions.size(); require(total <= 65536, "logger total action bound exceeded");
        for (const auto & action : actions) {
            require(action.is_object() && action.contains("Kind") && action.at("Kind").is_string(), "logger action Kind must be text");
            const auto kind = action.at("Kind").get<std::string>();
            if (kind == "emit") {
                keys(action, {"Kind", "Level", "Text"});
                require(action.at("Level").is_string(), "logger Level must be text");
                const auto level = action.at("Level").get<std::string>();
                require(level == "info" || level == "warning" || level == "error" || level == "debug", "unknown logger Level");
                require(action.at("Text").is_string(), "logger Text must be text");
                const auto & text = action.at("Text").get_ref<const std::string &>();
                require(text.size() <= 1024 && text.find('\0') == std::string::npos, "logger text exceeds bounds");
            } else {
                keys(action, {"Kind", "Value"});
                require((kind == "timestamps" || kind == "prefix") && action.at("Value").is_boolean(), "invalid logger setting");
            }
        }
    }
    const auto & final = operation.at("Finalize");
    require(final.is_array() && final.size() <= 8, "logger finalization bound exceeded");
    bool freed = false;
    for (const auto & action : final) {
        require(action.is_string() && !freed, "logger finalization after free");
        const auto kind = action.get<std::string>();
        require(kind == "flush" || kind == "free", "unknown logger finalization");
        freed = kind == "free";
    }
    require(freed, "logger singleton must be explicitly freed");
}
inline void apply(const json & action) {
    const auto kind = action.at("Kind").get<std::string>();
    if (kind == "emit") {
        const auto & text = action.at("Text").get_ref<const std::string &>();
        const auto level = action.at("Level").get<std::string>();
        if (level == "info") { LOG_INF("%s", text.c_str()); }
        else if (level == "warning") { LOG_WRN("%s", text.c_str()); }
        else if (level == "error") { LOG_ERR("%s", text.c_str()); }
        else { LOG_DBG("%s", text.c_str()); }
    } else if (kind == "timestamps") {
        common_log_set_timestamps(common_log_main(), action.at("Value").get<bool>());
    } else {
        common_log_set_prefix(common_log_main(), action.at("Value").get<bool>());
    }
}
inline json execute(const json & operation) {
    const auto & workers = operation.at("Workers");
    std::vector<std::thread> threads;
    threads.reserve(workers.size());
    std::vector<std::exception_ptr> errors(workers.size());
    std::vector<std::size_t> completed(workers.size());
    try {
        for (std::size_t i = 0; i < workers.size(); ++i) {
            threads.emplace_back([&, i]() {
                try {
                    for (const auto & action : workers[i]) { apply(action); ++completed[i]; }
                } catch (...) { errors[i] = std::current_exception(); }
            });
        }
    } catch (...) {
        for (auto & thread : threads) thread.join();
        common_log_flush(common_log_main()); common_log_free(common_log_main());
        throw;
    }
    for (auto & thread : threads) thread.join();
    for (const auto & action : operation.at("Finalize")) {
        if (action == "flush") common_log_flush(common_log_main());
        else common_log_free(common_log_main());
    }
    for (const auto & error : errors) if (error) std::rethrow_exception(error);
    return json{{"Completed", completed}};
}
} // namespace prism_concurrency
