// Supplied batches/memory ranges and production allocator observations only.
#pragma once
#include "llama-batch.h"
#include "llama-memory.h"
#include "llama-vocab.h"
#include <nlohmann/json.hpp>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace prism_batch_allocator {
using json = nlohmann::ordered_json;
inline void require(bool condition, const char * message) { if (!condition) throw std::runtime_error(message); }
inline void keys(const json & value, std::initializer_list<const char *> names) {
    require(value.is_object() && value.size() == names.size(), "batch allocator object fields mismatch");
    for (const auto * name : names) require(value.contains(name), "missing batch allocator field");
}
inline std::int32_t integer(const json & value, std::int32_t minimum, std::int32_t maximum) {
    require(value.is_number_integer(), "batch allocator integer required");
    if (value.is_number_unsigned()) require(value.get<std::uint64_t>() <= static_cast<std::uint64_t>(maximum), "batch allocator integer outside range");
    const auto number = value.get<std::int64_t>();
    require(number >= minimum && number <= maximum, "batch allocator integer outside range");
    return static_cast<std::int32_t>(number);
}
inline void list(const json & value, std::size_t size) { require(value.is_array() && value.size() == size, "batch allocator array extent mismatch"); }
inline bool handles(const std::string & kind) { return kind == "batch-allocator"; }

inline void validate(const json & operation) {
    keys(operation, {"Kind", "PositionsPerEmbedding", "Actions"});
    require(operation.at("Kind").is_string() && handles(operation.at("Kind").get<std::string>()), "unknown batch allocator Kind");
    const auto n_pos = integer(operation.at("PositionsPerEmbedding"), 1, 4);
    require(n_pos == 1 || n_pos == 4, "PositionsPerEmbedding must be 1 or 4");
    require(operation.at("Actions").is_array() && !operation.at("Actions").empty() && operation.at("Actions").size() <= 64, "batch allocator action count outside bound");
    for (const auto & action : operation.at("Actions")) {
        require(action.is_object() && action.contains("Op") && action.at("Op").is_string(), "batch allocator action Op required");
        const auto op = action.at("Op").get<std::string>();
        if (op == "init") {
            keys(action, {"Op", "EmbeddingWidth", "Embeddings", "Tokens", "Positions", "SequenceIds", "Logits", "SequenceLimit", "OutputAll", "MemoryRanges"});
            const auto width = integer(action.at("EmbeddingWidth"), 0, 8);
            integer(action.at("SequenceLimit"), 1, LLAMA_MAX_SEQ + 1);
            require(action.at("OutputAll").is_boolean(), "batch allocator OutputAll must be boolean");
            const bool embeddings = !action.at("Embeddings").is_null();
            require(embeddings != !action.at("Tokens").is_null(), "supply either tokens or embeddings");
            std::size_t count;
            if (embeddings) {
                require(width > 0 && action.at("Embeddings").is_array(), "embedding input requires positive width and array");
                count = action.at("Embeddings").size() / static_cast<std::size_t>(width);
                list(action.at("Embeddings"), count * static_cast<std::size_t>(width));
                for (const auto & value : action.at("Embeddings")) require(value.is_number() && std::isfinite(value.get<float>()), "embedding values must be finite");
            } else {
                require(width == 0 && action.at("Tokens").is_array(), "token input requires zero embedding width");
                count = action.at("Tokens").size();
                for (const auto & value : action.at("Tokens")) integer(value, -32768, 32768);
            }
            require(count > 0 && count <= 64, "batch allocator token count outside bound");
            if (!action.at("Positions").is_null()) {
                list(action.at("Positions"), count * static_cast<std::size_t>(embeddings ? n_pos : 1));
                for (const auto & value : action.at("Positions")) integer(value, -32768, 32768);
            }
            if (!action.at("SequenceIds").is_null()) {
                list(action.at("SequenceIds"), count);
                for (const auto & ids : action.at("SequenceIds")) {
                    require(ids.is_array() && !ids.empty() && ids.size() <= 8, "batch allocator sequence list outside bound");
                    for (const auto & id : ids) integer(id, -1, LLAMA_MAX_SEQ);
                }
            }
            if (!action.at("Logits").is_null()) {
                list(action.at("Logits"), count);
                for (const auto & value : action.at("Logits")) integer(value, 0, 1);
            }
            if (!action.at("MemoryRanges").is_null()) {
                require(action.at("MemoryRanges").is_array() && action.at("MemoryRanges").size() <= 16, "batch allocator memory range count outside bound");
                std::map<std::int32_t, bool> seen;
                for (const auto & range : action.at("MemoryRanges")) {
                    keys(range, {"Sequence", "Min", "Max"});
                    const auto id = integer(range.at("Sequence"), 0, LLAMA_MAX_SEQ - 1);
                    require(seen.emplace(id, true).second, "duplicate memory sequence range");
                    integer(range.at("Min"), -1, 32768); integer(range.at("Max"), -1, 32768);
                }
            }
        } else if (op == "split-equal") {
            keys(action, {"Op", "Capacity", "Sequential", "KeepTail"});
            integer(action.at("Capacity"), 1, 64); integer(action.at("KeepTail"), 0, 64);
            require(action.at("Sequential").is_boolean(), "batch allocator Sequential must be boolean");
        } else if (op == "split-simple" || op == "split-seq") {
            keys(action, {"Op", "Capacity"}); integer(action.at("Capacity"), 1, 64);
        } else if (op == "split-reset") keys(action, {"Op"});
        else if (op == "reserve") {
            keys(action, {"Op", "SequenceTokens", "Sequences"});
            const auto tokens = integer(action.at("SequenceTokens"), 1, 16);
            const auto sequences = integer(action.at("Sequences"), 1, 16);
            require(tokens * sequences <= 64, "reserved batch exceeds token count bound");
        } else throw std::runtime_error("unknown batch allocator action Op");
    }
}

// A typed input fixture, not an implementation of allocator behavior.
struct range_memory final : llama_memory_i {
    std::map<llama_seq_id, std::pair<llama_pos, llama_pos>> ranges;
    [[noreturn]] static void unavailable() { throw std::runtime_error("range-only input memory method is unavailable"); }
    llama_memory_context_ptr init_batch(llama_batch_allocr &, std::uint32_t, bool) override { unavailable(); }
    llama_memory_context_ptr init_full() override { unavailable(); }
    llama_memory_context_ptr init_update(llama_context *, bool) override { unavailable(); }
    bool get_can_shift() const override { unavailable(); }
    void clear(bool) override { unavailable(); }
    bool seq_rm(llama_seq_id, llama_pos, llama_pos) override { unavailable(); }
    void seq_cp(llama_seq_id, llama_seq_id, llama_pos, llama_pos) override { unavailable(); }
    void seq_keep(llama_seq_id) override { unavailable(); }
    void seq_add(llama_seq_id, llama_pos, llama_pos, llama_pos) override { unavailable(); }
    void seq_div(llama_seq_id, llama_pos, llama_pos, int) override { unavailable(); }
    llama_pos seq_pos_min(llama_seq_id id) const override { const auto it = ranges.find(id); return it == ranges.end() ? -1 : it->second.first; }
    llama_pos seq_pos_max(llama_seq_id id) const override { const auto it = ranges.find(id); return it == ranges.end() ? -1 : it->second.second; }
    std::map<ggml_backend_buffer_type_t, std::size_t> memory_breakdown() const override { return {}; }
    void state_write(llama_io_write_i &, llama_seq_id, llama_state_seq_flags) const override { unavailable(); }
    void state_read(llama_io_read_i &, llama_seq_id, llama_state_seq_flags) override { unavailable(); }
};

template <typename T> inline json values(const T * pointer, std::size_t count) {
    if (!pointer) return nullptr;
    json result = json::array();
    for (std::size_t i = 0; i < count; ++i) result.push_back(pointer[i]);
    return result;
}
inline json sequences(llama_seq_id * const * pointer, const std::int32_t * counts, std::size_t count) {
    if (!pointer || !counts) return nullptr;
    json result = json::array();
    for (std::size_t i = 0; i < count; ++i) {
        require(counts[i] >= 0 && counts[i] <= 8, "production batch sequence count exceeds bound");
        if (counts[i] == 0) result.push_back(json::array());
        else { require(pointer[i] != nullptr, "production batch sequence pointer is null"); result.push_back(values(pointer[i], static_cast<std::size_t>(counts[i]))); }
    }
    return result;
}
inline json batch(const llama_batch & value, std::uint32_t width, std::uint32_t n_pos) {
    require(value.n_tokens >= 0 && value.n_tokens <= 64, "production batch token count outside bound");
    const auto count = static_cast<std::size_t>(value.n_tokens);
    return json{{"Embeddings", values(value.embd, count * width)}, {"Tokens", values(value.token, count)},
        {"Positions", values(value.pos, count * (value.embd ? n_pos : 1))},
        {"SequenceIds", sequences(value.seq_id, value.n_seq_id, count)}, {"Logits", values(value.logits, count)}};
}
inline json state(llama_batch_allocr & allocator, std::uint32_t width, std::uint32_t n_pos, std::uint32_t limit) {
    require(limit <= LLAMA_MAX_SEQ, "production allocator accepted an oversized sequence limit");
    json positions = json::array();
    for (std::uint32_t id = 0; id < limit; ++id) positions.push_back(json{{"Min", allocator.seq_pos_min(id)}, {"Max", allocator.seq_pos_max(id)}});
    const auto & out_ids = allocator.get_out_ids();
    require(out_ids.size() <= 64, "production output ID count exceeds bound");
    return json{{"NTokens", allocator.get_n_tokens()}, {"NOutputs", allocator.get_n_outputs()}, {"NUsed", allocator.get_n_used()},
        {"Batch", batch(allocator.get_batch(), width, n_pos)}, {"OutIds", out_ids}, {"SequencePositions", positions}};
}
inline json ubatch(const llama_ubatch & value, std::uint32_t width) {
    require(value.n_tokens <= 64 && value.n_pos <= 4 && value.n_seqs_unq <= LLAMA_MAX_SEQ, "production ubatch counts exceed bound");
    return json{{"NTokens", value.n_tokens}, {"NSequenceTokens", value.n_seq_tokens}, {"NSequences", value.n_seqs},
        {"NUniqueSequences", value.n_seqs_unq}, {"NPositions", value.n_pos}, {"EqualSequences", value.equal_seqs()}, {"IsPosition2D", value.is_pos_2d()},
        {"HasTokens", value.token != nullptr}, {"HasEmbeddings", value.embd != nullptr},
        {"Tokens", values(value.token, value.n_tokens)}, {"Embeddings", values(value.embd, value.n_tokens * width)},
        {"Positions", values(value.pos, value.n_tokens * value.n_pos)},
        {"SequenceIds", sequences(value.seq_id, value.n_seq_id, value.n_tokens)},
        {"UniqueSequenceIds", values(value.seq_id_unq, value.n_seqs_unq)},
        {"SequenceIndices", values(value.seq_idx, value.seq_idx ? LLAMA_MAX_SEQ : 0)}, {"Output", values(value.output, value.n_tokens)}};
}
struct input {
    std::vector<float> embeddings;
    std::vector<llama_token> tokens;
    std::vector<llama_pos> positions;
    std::vector<std::int8_t> logits;
    std::vector<std::vector<llama_seq_id>> ids;
    std::vector<llama_seq_id *> pointers;
    std::vector<std::int32_t> counts;
    range_memory memory;
    llama_batch value{};
    explicit input(const json & action) {
        if (!action.at("Embeddings").is_null()) embeddings = action.at("Embeddings").get<std::vector<float>>();
        if (!action.at("Tokens").is_null()) tokens = action.at("Tokens").get<std::vector<llama_token>>();
        if (!action.at("Positions").is_null()) positions = action.at("Positions").get<std::vector<llama_pos>>();
        if (!action.at("Logits").is_null()) logits = action.at("Logits").get<std::vector<std::int8_t>>();
        if (!action.at("SequenceIds").is_null()) {
            ids = action.at("SequenceIds").get<std::vector<std::vector<llama_seq_id>>>();
            for (auto & sequence : ids) { pointers.push_back(sequence.data()); counts.push_back(static_cast<std::int32_t>(sequence.size())); }
            pointers.push_back(nullptr);
        }
        if (!action.at("MemoryRanges").is_null()) for (const auto & range : action.at("MemoryRanges")) memory.ranges.emplace(range.at("Sequence").get<llama_seq_id>(), std::make_pair(range.at("Min").get<llama_pos>(), range.at("Max").get<llama_pos>()));
        const auto width = action.at("EmbeddingWidth").get<std::uint32_t>();
        value.n_tokens = static_cast<std::int32_t>(width ? embeddings.size() / width : tokens.size());
        value.embd = embeddings.empty() ? nullptr : embeddings.data(); value.token = tokens.empty() ? nullptr : tokens.data();
        value.pos = positions.empty() ? nullptr : positions.data(); value.logits = logits.empty() ? nullptr : logits.data();
        value.seq_id = pointers.empty() ? nullptr : pointers.data(); value.n_seq_id = counts.empty() ? nullptr : counts.data();
    }
};
inline json execute(const json & operation) {
    const auto n_pos = operation.at("PositionsPerEmbedding").get<std::uint32_t>();
    llama_batch_allocr allocator(n_pos);
    llama_vocab vocab;
    std::vector<std::unique_ptr<input>> inputs;
    std::uint32_t width = 0, limit = 0;
    bool initialized = false;
    json observations = json::array();
    for (const auto & action : operation.at("Actions")) {
        const auto op = action.at("Op").get<std::string>();
        if (op == "init") {
            inputs.emplace_back(std::make_unique<input>(action));
            auto & owned = *inputs.back();
            width = action.at("EmbeddingWidth").get<std::uint32_t>(); limit = action.at("SequenceLimit").get<std::uint32_t>();
            initialized = allocator.init(owned.value, vocab, action.at("MemoryRanges").is_null() ? nullptr : &owned.memory,
                    width, limit, action.at("OutputAll").get<bool>());
            observations.push_back(json{{"Initialized", initialized}, {"State", initialized ? state(allocator, width, n_pos, limit) : json(nullptr)}});
        } else if (op == "reserve") {
            const auto value = allocator.ubatch_reserve(action.at("SequenceTokens").get<std::uint32_t>(), action.at("Sequences").get<std::uint32_t>());
            initialized = false;
            observations.push_back(json{{"Ubatch", ubatch(value, 0)}, {"State", nullptr}});
        } else {
            require(initialized, "cannot split or reset a rejected/uninitialized production batch");
            if (op == "split-reset") {
                allocator.split_reset(); observations.push_back(json{{"State", state(allocator, width, n_pos, limit)}});
            } else {
                const auto capacity = action.at("Capacity").get<std::uint32_t>();
                const auto value = op == "split-simple" ? allocator.split_simple(capacity) :
                    op == "split-seq" ? allocator.split_seq(capacity) : allocator.split_equal(capacity, action.at("Sequential").get<bool>(), action.at("KeepTail").get<std::uint32_t>());
                observations.push_back(json{{"Ubatch", ubatch(value, width)}, {"State", state(allocator, width, n_pos, limit)}});
            }
        }
    }
    return json{{"Actions", observations}};
}
} // namespace prism_batch_allocator
