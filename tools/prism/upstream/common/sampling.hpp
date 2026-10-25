#pragma once

// Production sampler lifetimes and raw candidate serialization. Workloads,
// probabilities, random inputs, assertions and timing intervals belong to Rust.
#include "llama.h"
#include "llama-sampler.h"
#include "ggml.h"
#include <nlohmann/json.hpp>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace prism_sampling {
using json = nlohmann::ordered_json;
constexpr std::size_t max_candidates = 131072;
constexpr std::size_t max_retained_candidates = 262144;
inline void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}
inline void fields(const json & value, std::initializer_list<const char *> names) {
    require(value.is_object() && value.size() == names.size(), "sampler fields mismatch");
    for (const auto * name : names) require(value.contains(name), "missing sampler field");
}
inline std::int64_t integer(const json & value) {
    require(value.is_number_integer(), "sampler value must be integer");
    if (value.is_number_unsigned()) {
        const auto number = value.get<std::uint64_t>();
        require(number <= std::uint64_t(INT64_MAX), "sampler signed integer overflow");
        return static_cast<std::int64_t>(number);
    }
    return value.get<std::int64_t>();
}
inline std::int32_t i32(const json & value) {
    const auto number = integer(value);
    require(number >= INT32_MIN && number <= INT32_MAX, "sampler i32 overflow");
    return static_cast<std::int32_t>(number);
}
inline std::uint32_t u32(const json & value, std::uint32_t maximum = UINT32_MAX) {
    const auto number = integer(value);
    require(number >= 0 && std::uint64_t(number) <= maximum, "sampler unsigned value outside bound");
    return static_cast<std::uint32_t>(number);
}
inline float from_bits(const json & value) {
    const auto bits = u32(value); float result;
    std::memcpy(&result, &bits, sizeof(result)); return result;
}
inline std::uint32_t to_bits(float value) {
    std::uint32_t result; std::memcpy(&result, &value, sizeof(result)); return result;
}
inline float parameter_float(const json & value) {
    const auto result = from_bits(value);
    require(std::isfinite(result), "sampler parameter must be finite"); return result;
}
inline void min_keep(const json & value) { u32(value, static_cast<std::uint32_t>(max_candidates)); }
inline void parameters(const std::string & type, const json & p) {
    if (type == "temp") {
        fields(p, {"TemperatureBits"}); parameter_float(p.at("TemperatureBits"));
    } else if (type == "temp-ext") {
        fields(p, {"TemperatureBits", "DeltaBits", "ExponentBits"});
        parameter_float(p.at("TemperatureBits")); parameter_float(p.at("DeltaBits")); parameter_float(p.at("ExponentBits"));
    } else if (type == "dist") {
        fields(p, {"Seed"}); u32(p.at("Seed"));
    } else if (type == "top-k") {
        fields(p, {"K"}); i32(p.at("K"));
    } else if (type == "top-p" || type == "min-p" || type == "typical") {
        fields(p, {"ProbabilityBits", "MinKeep"}); parameter_float(p.at("ProbabilityBits")); min_keep(p.at("MinKeep"));
    } else if (type == "xtc") {
        fields(p, {"ProbabilityBits", "ThresholdBits", "MinKeep", "Seed"});
        parameter_float(p.at("ProbabilityBits")); parameter_float(p.at("ThresholdBits")); min_keep(p.at("MinKeep")); u32(p.at("Seed"));
    } else if (type == "penalties") {
        fields(p, {"VocabularySize", "PenaltyLastN", "RepeatBits", "FrequencyBits", "PresenceBits"});
        require(u32(p.at("VocabularySize"), max_candidates) > 0, "sampler vocabulary cannot be empty");
        const auto last = i32(p.at("PenaltyLastN"));
        require(last >= -1 && last <= static_cast<std::int32_t>(max_candidates), "penalty history exceeds bound");
        require(parameter_float(p.at("RepeatBits")) > 0, "repeat penalty must be positive");
        parameter_float(p.at("FrequencyBits")); parameter_float(p.at("PresenceBits"));
    } else if (type == "dry-testing") {
        fields(p, {"MultiplierBits", "BaseBits", "AllowedLength", "PenaltyLastN", "SequenceBreakers"});
        parameter_float(p.at("MultiplierBits")); parameter_float(p.at("BaseBits"));
        const auto allowed = i32(p.at("AllowedLength")), last = i32(p.at("PenaltyLastN"));
        require(allowed >= 0 && allowed <= static_cast<std::int32_t>(max_candidates), "DRY allowed length exceeds bound");
        require(last >= -1 && last <= static_cast<std::int32_t>(max_candidates), "DRY history exceeds bound");
        const auto & breakers = p.at("SequenceBreakers");
        require(breakers.is_array() && breakers.size() <= 128, "too many DRY sequence breakers");
        std::size_t tokens = 0;
        for (const auto & breaker : breakers) {
            require(breaker.is_array() && breaker.size() <= 1024, "DRY breaker exceeds bound");
            tokens += breaker.size(); require(tokens <= 8192, "DRY breaker tokens exceed bound");
            for (const auto & token : breaker) i32(token);
        }
    } else if (type == "top-n-sigma") {
        fields(p, {"NBits"}); parameter_float(p.at("NBits"));
    } else throw std::runtime_error("unknown sampler Type");
}
inline bool handles(const std::string & kind) { return kind == "sampler-trace"; }
inline void validate(const json & operation) {
    fields(operation, {"Kind", "Actions"});
    const auto & actions = operation.at("Actions");
    require(actions.is_array() && !actions.empty() && actions.size() <= 4096, "sampler action count outside bound");
    std::set<std::uint32_t> samplers;
    std::map<std::uint32_t, std::size_t> candidates;
    std::size_t retained = 0;
    for (const auto & action : actions) {
        require(action.is_object() && action.contains("Kind") && action.at("Kind").is_string(), "sampler action Kind must be string");
        const auto kind = action.at("Kind").get<std::string>();
        if (kind == "clock") { fields(action, {"Kind"}); continue; }
        if (kind == "init-sampler") {
            fields(action, {"Kind", "Sampler", "Type", "Parameters"}); const auto id = u32(action.at("Sampler"), 31);
            require(!samplers.count(id), "sampler handle already exists"); require(action.at("Type").is_string(), "sampler Type must be string");
            parameters(action.at("Type").get<std::string>(), action.at("Parameters")); samplers.insert(id); continue;
        }
        if (kind == "init-candidates") {
            fields(action, {"Kind", "Candidates", "Data", "Selected", "Sorted"});
            const auto id = u32(action.at("Candidates"), 31); require(!candidates.count(id), "candidate handle already exists");
            const auto & data = action.at("Data");
            require(data.is_array() && !data.empty() && data.size() <= max_candidates, "candidate vector extent outside bound");
            retained += data.size(); require(retained <= max_retained_candidates, "retained candidates exceed bound");
            const auto selected = integer(action.at("Selected"));
            require(selected >= -1 && (selected == -1 || std::uint64_t(selected) < data.size()), "selected candidate outside extent");
            require(action.at("Sorted").is_boolean(), "candidate Sorted must be boolean");
            for (const auto & entry : data) {
                fields(entry, {"Id", "LogitBits", "ProbabilityBits"}); i32(entry.at("Id")); u32(entry.at("LogitBits")); u32(entry.at("ProbabilityBits"));
            }
            candidates[id] = data.size(); continue;
        }
        if (kind == "copy-candidates") {
            fields(action, {"Kind", "Source", "Target"}); const auto source = u32(action.at("Source"), 31), target = u32(action.at("Target"), 31);
            require(candidates.count(source), "unknown candidate source");
            const auto size = candidates.at(source); if (candidates.count(target)) retained -= candidates.at(target);
            retained += size; require(retained <= max_retained_candidates, "retained candidates exceed bound"); candidates[target] = size; continue;
        }
        if (kind == "snapshot-candidates" || kind == "free-candidates") {
            fields(action, {"Kind", "Candidates"}); const auto id = u32(action.at("Candidates"), 31);
            require(candidates.count(id), "unknown candidate handle");
            if (kind == "free-candidates") { retained -= candidates.at(id); candidates.erase(id); }
            continue;
        }
        require(action.contains("Sampler"), "sampler action requires handle");
        const auto id = u32(action.at("Sampler"), 31); require(samplers.count(id), "unknown sampler handle");
        if (kind == "apply") {
            fields(action, {"Kind", "Sampler", "Candidates"}); require(candidates.count(u32(action.at("Candidates"), 31)), "unknown apply candidates");
        } else if (kind == "accept") {
            fields(action, {"Kind", "Sampler", "Token"}); i32(action.at("Token"));
        } else if (kind == "reset" || kind == "free-sampler") {
            fields(action, {"Kind", "Sampler"}); if (kind == "free-sampler") samplers.erase(id);
        } else throw std::runtime_error("unknown sampler action");
    }
}
inline llama_sampler * create_sampler(const json & action) {
    const auto type = action.at("Type").get<std::string>(); const auto & p = action.at("Parameters");
    if (type == "temp") return llama_sampler_init_temp(from_bits(p.at("TemperatureBits")));
    if (type == "temp-ext") return llama_sampler_init_temp_ext(from_bits(p.at("TemperatureBits")), from_bits(p.at("DeltaBits")), from_bits(p.at("ExponentBits")));
    if (type == "dist") return llama_sampler_init_dist(u32(p.at("Seed")));
    if (type == "top-k") return llama_sampler_init_top_k(i32(p.at("K")));
    if (type == "top-p") return llama_sampler_init_top_p(from_bits(p.at("ProbabilityBits")), u32(p.at("MinKeep")));
    if (type == "min-p") return llama_sampler_init_min_p(from_bits(p.at("ProbabilityBits")), u32(p.at("MinKeep")));
    if (type == "typical") return llama_sampler_init_typical(from_bits(p.at("ProbabilityBits")), u32(p.at("MinKeep")));
    if (type == "xtc") return llama_sampler_init_xtc(from_bits(p.at("ProbabilityBits")), from_bits(p.at("ThresholdBits")), u32(p.at("MinKeep")), u32(p.at("Seed")));
    if (type == "penalties") return llama_sampler_init_penalties(i32(p.at("VocabularySize")), i32(p.at("PenaltyLastN")), from_bits(p.at("RepeatBits")), from_bits(p.at("FrequencyBits")), from_bits(p.at("PresenceBits")));
    if (type == "dry-testing") {
        std::vector<std::vector<llama_token>> breakers;
        for (const auto & entry : p.at("SequenceBreakers")) {
            std::vector<llama_token> tokens; for (const auto & token : entry) tokens.push_back(i32(token)); breakers.push_back(std::move(tokens));
        }
        return llama_sampler_init_dry_testing(from_bits(p.at("MultiplierBits")), from_bits(p.at("BaseBits")), i32(p.at("AllowedLength")), i32(p.at("PenaltyLastN")), breakers);
    }
    return llama_sampler_init_top_n_sigma(from_bits(p.at("NBits")));
}
struct sampler_deleter { void operator()(llama_sampler * value) const { llama_sampler_free(value); } };
struct candidate_storage {
    std::vector<llama_token_data> backing;
    llama_token_data_array view{};
    std::size_t offset() const {
        const auto begin = reinterpret_cast<std::uintptr_t>(backing.data());
        const auto pointer = reinterpret_cast<std::uintptr_t>(view.data);
        const auto bytes = backing.size() * sizeof(llama_token_data);
        require(pointer >= begin && pointer - begin <= bytes && (pointer - begin) % sizeof(llama_token_data) == 0, "production candidate pointer outside backing storage");
        const auto index = static_cast<std::size_t>((pointer - begin) / sizeof(llama_token_data));
        require(view.size <= backing.size() - index, "production candidate view exceeds backing storage");
        return index;
    }
};
inline json execute(const json & operation) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "sampler protocol requires IEEE binary32");
    ggml_time_init();
    std::array<std::unique_ptr<llama_sampler, sampler_deleter>, 32> samplers;
    std::array<std::unique_ptr<candidate_storage>, 32> candidates;
    json trace = json::array();
    for (const auto & action : operation.at("Actions")) {
        const auto kind = action.at("Kind").get<std::string>(); json event = json::object();
        if (kind == "clock") event["Microseconds"] = ggml_time_us();
        else if (kind == "init-sampler") {
            auto & sampler = samplers[u32(action.at("Sampler"), 31)]; sampler.reset(create_sampler(action)); require(bool(sampler), "production sampler construction returned null");
        } else if (kind == "init-candidates") {
            auto storage = std::make_unique<candidate_storage>();
            for (const auto & entry : action.at("Data")) storage->backing.push_back({i32(entry.at("Id")), from_bits(entry.at("LogitBits")), from_bits(entry.at("ProbabilityBits"))});
            storage->view = {storage->backing.data(), storage->backing.size(), integer(action.at("Selected")), action.at("Sorted").get<bool>()};
            candidates[u32(action.at("Candidates"), 31)] = std::move(storage);
        } else if (kind == "copy-candidates") {
            const auto source = u32(action.at("Source"), 31), target = u32(action.at("Target"), 31);
            const auto & previous = *candidates[source]; const auto offset = previous.offset();
            if (source != target) {
                auto storage = std::make_unique<candidate_storage>(); storage->backing = previous.backing;
                storage->view = {storage->backing.data() + offset, previous.view.size, previous.view.selected, previous.view.sorted}; candidates[target] = std::move(storage);
            }
        } else if (kind == "free-candidates") candidates[u32(action.at("Candidates"), 31)].reset();
        else if (kind == "snapshot-candidates") {
            const auto & storage = *candidates[u32(action.at("Candidates"), 31)]; storage.offset();
            json data = json::array();
            for (std::size_t i = 0; i < storage.view.size; ++i) {
                const auto & entry = storage.view.data[i]; data.push_back({{"Id", entry.id}, {"LogitBits", to_bits(entry.logit)}, {"ProbabilityBits", to_bits(entry.p)}});
            }
            event = json{{"Data", data}, {"Size", storage.view.size}, {"Selected", storage.view.selected}, {"Sorted", storage.view.sorted}};
        } else {
            const auto id = u32(action.at("Sampler"), 31); auto * sampler = samplers[id].get();
            if (kind == "apply") {
                auto & storage = *candidates[u32(action.at("Candidates"), 31)]; storage.offset();
                require(storage.view.size > 0, "cannot apply sampler to empty candidate view"); llama_sampler_apply(sampler, &storage.view); storage.offset();
            } else if (kind == "accept") llama_sampler_accept(sampler, i32(action.at("Token")));
            else if (kind == "reset") llama_sampler_reset(sampler);
            else samplers[id].reset();
        }
        trace.push_back(std::move(event));
    }
    return json{{"Trace", trace}};
}
} // namespace prism_sampling
