#pragma once

// Generic sampler lifecycle and byte serialization only. The Rust caller owns
// token sequences, candidate vectors, action ordering and every assertion.
#include "reasoning-budget.h"
#include "common/unicode.h"
#include <nlohmann/json.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace prism_reasoning_budget {
using json = nlohmann::ordered_json;

inline void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

inline void fields(const json & object, std::initializer_list<const char *> keys) {
    require(object.is_object() && object.size() == keys.size(), "invalid reasoning-budget object fields");
    for (const auto * key : keys) {
        require(object.contains(key), "missing or unknown reasoning-budget field");
    }
}

inline std::int64_t signed_integer(const json & value) {
    require(value.is_number_integer(), "reasoning-budget value must be an integer");
    if (value.is_number_unsigned()) {
        const auto number = value.get<std::uint64_t>();
        require(number <= std::uint64_t(std::numeric_limits<std::int64_t>::max()), "integer outside signed range");
        return static_cast<std::int64_t>(number);
    }
    return value.get<std::int64_t>();
}

inline std::int32_t i32(const json & value) {
    const auto number = signed_integer(value);
    require(number >= std::numeric_limits<std::int32_t>::min() && number <= std::numeric_limits<std::int32_t>::max(),
            "integer outside signed 32-bit range");
    return static_cast<std::int32_t>(number);
}

inline std::uint32_t u32(const json & value, std::uint32_t maximum = std::numeric_limits<std::uint32_t>::max()) {
    const auto number = signed_integer(value);
    require(number >= 0 && std::uint64_t(number) <= maximum, "integer outside unsigned protocol range");
    return static_cast<std::uint32_t>(number);
}

inline bool handles(const std::string & kind) {
    return kind == "reasoning-budget-trace" || kind == "utf8-is-complete";
}

inline void tokens(const json & sequence) {
    require(sequence.is_array() && sequence.size() <= 1024, "token sequence exceeds 1024 elements");
    for (const auto & token : sequence) {
        i32(token);
    }
}

inline void sequences(const json & collection) {
    require(collection.is_array() && collection.size() <= 64, "token sequence collection exceeds 64 entries");
    for (const auto & sequence : collection) {
        tokens(sequence);
    }
}

inline void validate(const json & operation) {
    const auto kind = operation.at("Kind").get<std::string>();
    if (kind == "utf8-is-complete") {
        fields(operation, {"Kind", "Bytes"});
        const auto & bytes = operation.at("Bytes");
        require(bytes.is_array() && bytes.size() <= 65536, "UTF-8 input exceeds 65536 bytes");
        for (const auto & byte : bytes) {
            u32(byte, 255);
        }
        return;
    }
    fields(operation, {"Kind", "Actions"});
    const auto & actions = operation.at("Actions");
    require(actions.is_array() && !actions.empty() && actions.size() <= 4096,
            "reasoning-budget trace must contain 1 through 4096 actions");
    std::set<std::uint32_t> live;
    std::size_t total_candidates = 0;
    for (const auto & action : actions) {
        require(action.is_object() && action.contains("Kind") && action.at("Kind").is_string() && action.contains("Sampler"),
                "trace action must name Kind and Sampler");
        const auto action_kind = action.at("Kind").get<std::string>();
        if (action_kind == "init") {
            fields(action, {"Kind", "Sampler", "Vocabulary", "StartSequences", "EndSequences", "ForcedTokens", "Budget", "InitialState"});
            const auto sampler = u32(action.at("Sampler"), 31);
            require(live.count(sampler) == 0, "sampler handle already exists");
            require(action.at("Vocabulary").is_null(), "only a null Vocabulary is supported");
            sequences(action.at("StartSequences"));
            sequences(action.at("EndSequences"));
            tokens(action.at("ForcedTokens"));
            i32(action.at("Budget"));
            u32(action.at("InitialState"), 4);
            live.insert(sampler);
            continue;
        }
        const bool is_null = action.at("Sampler").is_null();
        const auto sampler = is_null ? 0 : u32(action.at("Sampler"), 31);
        require(is_null || live.count(sampler) != 0, "unknown or freed sampler handle");
        if (action_kind == "apply") {
            fields(action, {"Kind", "Sampler", "Candidates", "Selected", "Sorted"});
            const auto & candidates = action.at("Candidates");
            require(candidates.is_array() && candidates.size() <= 4096, "candidate vector exceeds 4096 entries");
            total_candidates += candidates.size();
            require(total_candidates <= 65536, "trace exceeds 65536 total candidate entries");
            const auto selected = signed_integer(action.at("Selected"));
            require(selected >= -1 && (selected == -1 || std::uint64_t(selected) < candidates.size()), "invalid selected candidate index");
            require(action.at("Sorted").is_boolean(), "Sorted must be boolean");
            for (const auto & candidate : candidates) {
                fields(candidate, {"Id", "LogitBits", "ProbabilityBits"});
                i32(candidate.at("Id"));
                u32(candidate.at("LogitBits"));
                u32(candidate.at("ProbabilityBits"));
            }
        } else if (action_kind == "accept") {
            fields(action, {"Kind", "Sampler", "Token"});
            i32(action.at("Token"));
        } else if (action_kind == "clone") {
            fields(action, {"Kind", "Sampler", "Target"});
            const auto target = u32(action.at("Target"), 31);
            require(live.count(target) == 0, "clone target already exists");
            live.insert(target);
        } else if (action_kind == "state" || action_kind == "force" || action_kind == "end-match" ||
                   action_kind == "free") {
            fields(action, {"Kind", "Sampler"});
            if (action_kind == "free" && !is_null) {
                live.erase(sampler);
            }
        } else {
            throw std::runtime_error("unknown reasoning-budget action Kind");
        }
    }
}

inline float from_bits(const json & value) {
    const auto bits = u32(value);
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

inline std::uint32_t to_bits(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

struct sampler_deleter {
    void operator()(llama_sampler * sampler) const { llama_sampler_free(sampler); }
};

inline json execute(const json & operation) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "float protocol requires IEEE binary32");
    if (operation.at("Kind") == "utf8-is-complete") {
        std::string bytes;
        for (const auto & byte : operation.at("Bytes")) {
            bytes.push_back(static_cast<char>(u32(byte, 255)));
        }
        return json{{"Complete", common_utf8_is_complete(bytes)}};
    }
    std::array<std::unique_ptr<llama_sampler, sampler_deleter>, 32> samplers;
    json trace = json::array();
    for (const auto & action : operation.at("Actions")) {
        const auto kind = action.at("Kind").get<std::string>();
        const bool is_null = action.at("Sampler").is_null();
        const auto index = is_null ? 0 : u32(action.at("Sampler"), 31);
        auto * sampler = is_null ? nullptr : samplers[index].get();
        json event = json::object();
        if (kind == "init") {
            samplers[index].reset(common_reasoning_budget_init(nullptr,
                    action.at("StartSequences").get<std::vector<llama_tokens>>(),
                    action.at("EndSequences").get<std::vector<llama_tokens>>(),
                    action.at("ForcedTokens").get<llama_tokens>(), i32(action.at("Budget")),
                    static_cast<common_reasoning_budget_state>(u32(action.at("InitialState"), 4))));
            require(bool(samplers[index]), "reasoning-budget sampler initialization failed");
        } else if (kind == "apply") {
            std::vector<llama_token_data> candidates;
            for (const auto & candidate : action.at("Candidates")) {
                candidates.push_back({i32(candidate.at("Id")), from_bits(candidate.at("LogitBits")),
                                      from_bits(candidate.at("ProbabilityBits"))});
            }
            llama_token_data_array array = {candidates.data(), candidates.size(),
                                           signed_integer(action.at("Selected")), action.at("Sorted").get<bool>()};
            llama_sampler_apply(sampler, &array);
            json values = json::array();
            // Preserve the complete backing vector, just as supplied by the caller.
            for (const auto & candidate : candidates) {
                values.push_back(json{{"Id", candidate.id}, {"LogitBits", to_bits(candidate.logit)},
                                      {"ProbabilityBits", to_bits(candidate.p)}});
            }
            event = json{{"Candidates", values}, {"Selected", array.selected}, {"Sorted", array.sorted}};
        } else if (kind == "accept") {
            llama_sampler_accept(sampler, i32(action.at("Token")));
        } else if (kind == "clone") {
            const auto target = u32(action.at("Target"), 31);
            samplers[target].reset(llama_sampler_clone(sampler));
            require(!sampler || bool(samplers[target]), "sampler clone failed");
        } else if (kind == "state") {
            event = json{{"State", static_cast<std::uint32_t>(common_reasoning_budget_get_state(sampler))}};
        } else if (kind == "force") {
            event = json{{"Forced", common_reasoning_budget_force(sampler)}};
        } else if (kind == "end-match") {
            const auto * match = common_reasoning_budget_get_end_match(sampler);
            event = json{{"EndMatch", match ? json(*match) : json(nullptr)}};
        } else if (kind == "free") {
            if (is_null) {
                llama_sampler_free(nullptr);
            } else {
                samplers[index].reset();
            }
        }
        trace.push_back(std::move(event));
    }
    return json{{"Trace", trace}};
}
} // namespace prism_reasoning_budget
