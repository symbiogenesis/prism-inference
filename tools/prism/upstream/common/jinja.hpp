#pragma once

// Generic serialization for production Jinja APIs. Inputs, workloads, expected
// strings and pass/fail decisions belong to the safe Rust caller.
#include "json.h"
#include "jinja/caps.h"
#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "jinja/utils.h"
#include <nlohmann/json.hpp>

#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace prism_jinja {
using json = nlohmann::ordered_json;

inline void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

inline bool handles(const std::string & kind) {
    return kind == "jinja-render" || kind == "jinja-hash" || kind == "jinja-hash-trace" || kind == "jinja-stats" ||
           kind == "jinja-caps" || kind == "jinja-string-parts" || kind == "jinja-builtins";
}

inline std::uint64_t unsigned_integer(const json & value, std::uint64_t maximum) {
    require(value.is_number_integer(), "expected nonnegative integer");
    require(value.is_number_unsigned() || value.get<std::int64_t>() >= 0, "negative integer is not permitted");
    const auto number = value.get<std::uint64_t>();
    require(number <= maximum, "integer exceeds protocol limit");
    return number;
}

inline void fields(const json & operation, std::initializer_list<const char *> required,
                   std::initializer_list<const char *> optional = {}) {
    std::set<std::string> allowed;
    for (const auto * field : required) {
        require(operation.contains(field), "missing Jinja field");
        allowed.insert(field);
    }
    for (const auto * field : optional) {
        allowed.insert(field);
    }
    for (const auto & item : operation.items()) {
        require(allowed.count(item.key()) != 0, "unknown Jinja field");
    }
}

inline void validate_path(const json & path) {
    require(path.is_array() && !path.empty() && path.size() <= 32, "path must contain 1 through 32 segments");
    require(path.front().is_string(), "path must begin with a global name");
    for (const auto & segment : path) {
        if (segment.is_string()) {
            require(segment.get_ref<const std::string &>().size() <= 4096, "path string exceeds size limit");
        } else {
            unsigned_integer(segment, 65535);
        }
    }
}

inline std::string variables_text(const json & operation) {
    require(operation.contains("Variables") != operation.contains("VariablesJson"),
            "exactly one of Variables and VariablesJson is required");
    if (operation.contains("VariablesJson")) {
        require(operation.at("VariablesJson").is_string(), "VariablesJson must be a string");
        return operation.at("VariablesJson").get<std::string>();
    }
    require(operation.at("Variables").is_object(), "Variables must be an object");
    return operation.at("Variables").dump();
}

inline common_json variables(const json & operation) {
    const auto content = variables_text(operation);
    require(content.size() <= 256 * 1024, "variables exceed 256 KiB limit");
    std::vector<std::set<std::string>> object_keys;
    const auto callback = [&](int depth, json::parse_event_t event, json & parsed) {
        require(depth <= 32, "variables nesting exceeds limit");
        if (event == json::parse_event_t::object_start) {
            object_keys.emplace_back();
        } else if (event == json::parse_event_t::key) {
            require(!object_keys.empty() && object_keys.back().insert(parsed.get<std::string>()).second,
                    "duplicate variables object key");
        } else if (event == json::parse_event_t::object_end) {
            object_keys.pop_back();
        }
        return true;
    };
    require(json::parse(content, callback).is_object(), "variables JSON must contain an object");
    auto converted = common_json::parse(content);
    if (operation.contains("ByteStrings")) {
        for (const auto & overlay : operation.at("ByteStrings")) {
            auto * destination = &converted;
            for (const auto & segment : overlay.at("Path")) {
                if (segment.is_string()) {
                    destination = &destination->at(segment.get<std::string>());
                } else {
                    destination = &destination->at(static_cast<std::size_t>(unsigned_integer(segment, 65535)));
                }
            }
            require(destination->is_string(), "ByteStrings must replace an existing string");
            std::string bytes;
            bytes.reserve(overlay.at("Bytes").size());
            for (const auto & byte : overlay.at("Bytes")) {
                bytes.push_back(static_cast<char>(unsigned_integer(byte, 255)));
            }
            *destination = common_json(bytes);
        }
    }
    return converted;
}

inline void validate(const json & operation) {
    const auto kind = operation.at("Kind").get<std::string>();
    if (kind == "jinja-hash-trace") {
        fields(operation, {"Kind", "Actions"});
        const auto & actions = operation.at("Actions");
        require(actions.is_array() && actions.size() <= 2048, "Actions must contain at most 2048 entries");
        std::size_t total_bytes = 0;
        for (const auto & action : actions) {
            require(action.is_object() && action.size() == 1, "hash action must contain exactly one field");
            if (action.contains("Update")) {
                const auto & bytes = action.at("Update");
                require(bytes.is_array(), "hash Update must be a byte array");
                total_bytes += bytes.size();
                require(total_bytes <= 256 * 1024, "hash input exceeds 256 KiB limit");
                for (const auto & byte : bytes) {
                    unsigned_integer(byte, 255);
                }
            } else {
                require(action.contains("Digest") && action.at("Digest").is_boolean() && action.at("Digest").get<bool>(),
                        "hash action must contain Update bytes or Digest true");
            }
        }
        return;
    }
    if (kind == "jinja-hash") {
        fields(operation, {"Kind", "Chunks"});
        require(operation.at("Chunks").is_array() && operation.at("Chunks").size() <= 1024,
                "Chunks must contain at most 1024 byte arrays");
        std::size_t total_bytes = 0;
        for (const auto & chunk : operation.at("Chunks")) {
            require(chunk.is_array(), "hash chunk must be a byte array");
            total_bytes += chunk.size();
            require(total_bytes <= 256 * 1024, "hash input exceeds 256 KiB limit");
            for (const auto & byte : chunk) {
                unsigned_integer(byte, 255);
            }
        }
        return;
    }
    if (kind == "jinja-builtins") {
        fields(operation, {"Kind"});
        return;
    }
    if (kind == "jinja-caps") {
        fields(operation, {"Kind", "Template"});
    } else if (kind == "jinja-render") {
        fields(operation, {"Kind", "Template"}, {"Variables", "VariablesJson", "ByteStrings"});
    } else if (kind == "jinja-stats") {
        fields(operation, {"Kind", "Template", "Paths"}, {"Variables", "VariablesJson"});
        require(operation.at("Paths").is_array() && operation.at("Paths").size() <= 1024,
                "Paths must contain at most 1024 paths");
        for (const auto & path : operation.at("Paths")) {
            validate_path(path);
        }
    } else if (kind == "jinja-string-parts") {
        fields(operation, {"Kind", "Template"}, {"Variables", "VariablesJson"});
    } else {
        throw std::runtime_error("unknown Jinja operation");
    }
    require(operation.at("Template").is_string() &&
            operation.at("Template").get_ref<const std::string &>().size() <= 65536,
            "Template must be a string of at most 65536 UTF-8 bytes");
    if (operation.contains("ByteStrings")) {
        const auto & overlays = operation.at("ByteStrings");
        require(overlays.is_array() && overlays.size() <= 16, "ByteStrings must contain at most 16 overlays");
        for (const auto & overlay : overlays) {
            require(overlay.is_object(), "byte-string overlay must be an object");
            fields(overlay, {"Path", "Bytes"});
            validate_path(overlay.at("Path"));
            const auto & bytes = overlay.at("Bytes");
            require(bytes.is_array() && bytes.size() <= 65536, "byte string must contain at most 65536 bytes");
            for (const auto & byte : bytes) {
                unsigned_integer(byte, 255);
            }
        }
    }
    if (kind != "jinja-caps") {
        variables(operation);
    }
}

inline json execute_unchecked(const json & operation, bool & runtime_phase) {
    const auto kind = operation.at("Kind").get<std::string>();
    if (kind == "jinja-hash-trace") {
        jinja::hasher hash;
        json digests = json::array();
        for (const auto & action : operation.at("Actions")) {
            if (action.contains("Update")) {
                std::vector<std::uint8_t> bytes;
                for (const auto & byte : action.at("Update")) {
                    bytes.push_back(static_cast<std::uint8_t>(byte.get<std::uint64_t>()));
                }
                hash.update(bytes.data(), bytes.size());
            } else {
                digests.push_back(static_cast<std::uint64_t>(hash.digest()));
            }
        }
        return json{{"Digests", digests}};
    }
    if (kind == "jinja-hash") {
        static_assert(sizeof(std::size_t) == 8, "Jinja protocol uses the production 64-bit hasher");
        jinja::hasher hash;
        for (const auto & chunk : operation.at("Chunks")) {
            std::vector<std::uint8_t> bytes;
            bytes.reserve(chunk.size());
            for (const auto & byte : chunk) {
                bytes.push_back(static_cast<std::uint8_t>(byte.get<std::uint64_t>()));
            }
            hash.update(bytes.data(), bytes.size());
        }
        return json{{"Digest", static_cast<std::uint64_t>(hash.digest())}};
    }
    if (kind == "jinja-builtins") {
        json inventory = json::array();
        const auto append = [&](const char * type, const jinja::func_builtins & builtins) {
            for (const auto & builtin : builtins) {
                inventory.push_back(json{{"Type", type}, {"Name", builtin.first}});
            }
        };
        append("global", jinja::global_builtins());
        append("int", jinja::value_int_t(0).get_builtins());
        append("float", jinja::value_float_t(0.0).get_builtins());
        append("string", jinja::value_string_t().get_builtins());
        append("array", jinja::value_array_t().get_builtins());
        append("object", jinja::value_object_t().get_builtins());
        return json{{"Builtins", inventory}};
    }
    const auto text = operation.at("Template").get<std::string>();
    jinja::lexer lexer;
    auto tokens = lexer.tokenize(text);
    auto program = jinja::parse_from_tokens(tokens);
    if (kind == "jinja-caps") {
        const auto capabilities = jinja::caps_get(program);
        return json{{"SupportsStringContent", capabilities.supports_string_content},
                    {"SupportsTypedContent", capabilities.supports_typed_content}};
    }
    jinja::context context(text);
    jinja::global_from_json(context, variables(operation), true);
    context.is_get_stats = kind == "jinja-stats";
    jinja::runtime runtime(context);
    runtime_phase = true;
    const auto result = runtime.execute(program);
    if (kind == "jinja-stats") {
        json used = json::array();
        for (const auto & path : operation.at("Paths")) {
            auto value = context.get_val(path.front().get<std::string>());
            for (std::size_t index = 1; index < path.size(); ++index) {
                const auto & segment = path.at(index);
                value = segment.is_string() ? value->at(segment.get<std::string>())
                        : value->at(static_cast<std::int64_t>(segment.get<std::uint64_t>()));
            }
            used.push_back(value->stats.used);
        }
        return json{{"Used", used}};
    }
    const auto string = runtime.gather_string_parts(result)->as_string();
    if (kind == "jinja-string-parts") {
        json parts = json::array();
        for (const auto & part : string.parts) {
            parts.push_back(json{{"Value", part.val}, {"IsInput", part.is_input}});
        }
        return json{{"Parts", parts}};
    }
    return json{{"Rendered", string.str()}};
}

inline json execute(const json & operation) {
    bool runtime_phase = false;
    try {
        auto result = execute_unchecked(operation, runtime_phase);
        // Validate that production strings can be represented on the JSON wire.
        // Encoding errors are observations too, rather than a crashed process.
        result.dump();
        return result;
    } catch (const jinja::not_implemented_exception & error) {
        return json{{"ExceptionKind", runtime_phase ? "not-implemented" : "standard"}, {"Message", error.what()}};
    } catch (const std::exception & error) {
        return json{{"ExceptionKind", "standard"}, {"Message", error.what()}};
    } catch (...) {
        return json{{"ExceptionKind", "unknown"}, {"Message", "non-standard exception"}};
    }
}
} // namespace prism_jinja
