// Generic MTMD public chunk lifetime/serialization transport. Assertions live in Rust.
#pragma once
#include "mtmd.h"
#include "mtmd-helper.h"
#include <nlohmann/json.hpp>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace prism_mtmd_chunks {
using json = nlohmann::ordered_json;
inline void require(bool condition, const char * message) { if (!condition) throw std::runtime_error(message); }
inline void keys(const json & value, std::initializer_list<const char *> names) {
    require(value.is_object() && value.size() == names.size(), "MTMD chunk fields mismatch");
    for (const auto * name : names) require(value.contains(name), "missing MTMD chunk field");
}
inline std::size_t integer(const json & value, std::size_t maximum) {
    require(value.is_number_integer() && (value.is_number_unsigned() || value.get<std::int64_t>() >= 0), "MTMD chunk integer must be nonnegative");
    const auto number = value.get<std::uint64_t>(); require(number <= maximum, "MTMD chunk integer exceeds bound");
    return static_cast<std::size_t>(number);
}
inline void validate(const json & operation) {
    keys(operation, {"Kind", "Actions"}); require(operation.at("Kind") == "mtmd-chunks-trace", "unknown MTMD chunk operation");
    const auto & actions = operation.at("Actions"); require(actions.is_array() && !actions.empty() && actions.size() <= 256, "MTMD chunk action count outside bound");
    std::map<std::size_t, bool> chunks, chunk_handles;
    std::map<std::size_t, std::size_t> buffers;
    std::size_t bytes = 0;
    for (const auto & action : actions) {
        require(action.is_object() && action.contains("Kind") && action.at("Kind").is_string(), "MTMD action Kind must be text");
        const auto kind = action.at("Kind").get<std::string>();
        if (kind == "defaults") { keys(action, {"Kind"}); continue; }
        if (kind == "create-test-inputs") {
            keys(action, {"Kind", "Chunks"}); const auto id = integer(action.at("Chunks"), 31);
            require(!chunks.count(id), "MTMD chunk collection handle reused"); chunks[id] = true; continue;
        }
        if (kind == "collection-info" || kind == "get") {
            if (kind == "get") keys(action, {"Kind", "Chunks", "Index", "Chunk"});
            else keys(action, {"Kind", "Chunks"});
            require(chunks.count(integer(action.at("Chunks"), 31)), "unknown MTMD collection handle");
            if (kind == "get") {
                integer(action.at("Index"), 4095); const auto id = integer(action.at("Chunk"), 63);
                require(!chunk_handles.count(id), "MTMD chunk handle reused"); chunk_handles[id] = false;
            }
            continue;
        }
        if (kind == "load") {
            keys(action, {"Kind", "Buffer", "Length", "Chunk"}); const auto buffer = integer(action.at("Buffer"), 63), id = integer(action.at("Chunk"), 63);
            require(buffers.count(buffer) && integer(action.at("Length"), 1024*1024) <= buffers.at(buffer), "MTMD load exceeds buffer");
            require(!chunk_handles.count(id), "MTMD loaded chunk handle reused"); chunk_handles[id] = true; continue;
        }
        require(action.contains("Chunk"), "MTMD action requires chunk handle"); const auto id = integer(action.at("Chunk"), 63);
        require(chunk_handles.count(id), "unknown MTMD chunk handle");
        if (kind == "snapshot" || kind == "query-save") keys(action, {"Kind", "Chunk"});
        else if (kind == "image-position") { keys(action, {"Kind", "Chunk", "Index", "Position0"}); integer(action.at("Index"), 1048575); integer(action.at("Position0"), 1048576); }
        else if (kind == "save") {
            keys(action, {"Kind", "Chunk", "Buffer", "Length"}); const auto buffer = integer(action.at("Buffer"), 63), length = integer(action.at("Length"), 1024*1024);
            require(!buffers.count(buffer), "MTMD byte buffer handle reused"); require(length > 0, "MTMD save buffer cannot be empty");
            bytes += length; require(bytes <= 4*1024*1024, "MTMD total save buffer bound exceeded"); buffers[buffer] = length;
        } else if (kind == "free") {
            keys(action, {"Kind", "Chunk"}); require(chunk_handles.at(id), "borrowed MTMD chunk cannot be freed"); chunk_handles.erase(id);
        } else throw std::runtime_error("unknown MTMD chunk action");
    }
}
inline json snapshot(const mtmd_input_chunk * chunk) {
    if (!chunk) return json{{"Present", false}};
    const auto type = mtmd_input_chunk_get_type(chunk);
    json value{{"Present", true}, {"Type", static_cast<int>(type)}, {"TokenCount", mtmd_input_chunk_get_n_tokens(chunk)}, {"PositionCount", mtmd_input_chunk_get_n_pos(chunk)}};
    const auto * id = mtmd_input_chunk_get_id(chunk); value["Id"] = id ? json(id) : json(nullptr);
    if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        std::size_t count = 0; const auto * tokens = mtmd_input_chunk_get_tokens_text(chunk, &count);
        require(count <= 1048576, "MTMD text token count exceeds bound");
        value["TokensPresent"] = tokens != nullptr; value["TextCount"] = count;
        value["TextTokens"] = json::array(); if (tokens) for (std::size_t i = 0; i < count; ++i) value["TextTokens"].push_back(tokens[i]);
    } else if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
        const auto * image = mtmd_input_chunk_get_tokens_image(chunk); value["ImagePresent"] = image != nullptr;
        if (image) {
            value["ImageCount"] = mtmd_image_tokens_get_n_tokens(image);
            const auto * image_id = mtmd_image_tokens_get_id(image); value["ImageId"] = image_id ? json(image_id) : json(nullptr);
        }
    }
    return value;
}
inline json execute(const json & operation) {
    using collection = std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)>;
    using owned_chunk = std::unique_ptr<mtmd_input_chunk, decltype(&mtmd_input_chunk_free)>;
    std::map<std::size_t, collection> collections;
    std::map<std::size_t, owned_chunk> owned;
    std::map<std::size_t, const mtmd_input_chunk *> chunks;
    std::map<std::size_t, std::vector<char>> buffers;
    json events = json::array();
    for (const auto & action : operation.at("Actions")) {
        const auto kind = action.at("Kind").get<std::string>(); json event = json::object();
        if (kind == "defaults") {
            const auto params = mtmd_context_params_default(); event["ImageMarker"] = params.image_marker ? json(params.image_marker) : json(nullptr);
        } else if (kind == "create-test-inputs") {
            // This public upstream production helper is itself the original source's SUT factory.
            const auto id = integer(action.at("Chunks"), 31); auto * value = mtmd_test_create_input_chunks();
            collections.emplace(id, collection(value, mtmd_input_chunks_free)); event["Created"] = value != nullptr;
        } else if (kind == "collection-info" || kind == "get") {
            auto * value = collections.at(integer(action.at("Chunks"), 31)).get(); require(value != nullptr, "MTMD collection is null");
            const auto count = mtmd_input_chunks_size(value); require(count <= 4096, "MTMD collection size exceeds bound");
            if (kind == "collection-info") { event["Count"] = count; event["TotalTokens"] = mtmd_helper_get_n_tokens(value); }
            else {
                const auto index = integer(action.at("Index"), 4095); require(index < count, "MTMD collection index outside extent");
                const auto * chunk = mtmd_input_chunks_get(value, index); chunks[integer(action.at("Chunk"), 63)] = chunk; event["Present"] = chunk != nullptr;
            }
        } else if (kind == "load") {
            const auto & data = buffers.at(integer(action.at("Buffer"), 63)); const auto id = integer(action.at("Chunk"), 63);
            auto * value = mtmd_input_chunk_load(data.data(), integer(action.at("Length"), data.size()));
            owned.emplace(id, owned_chunk(value, mtmd_input_chunk_free)); chunks[id] = value; event["Created"] = value != nullptr;
        } else if (kind == "free") {
            const auto id = integer(action.at("Chunk"), 63); owned.erase(id); chunks.erase(id);
        } else {
            const auto * chunk = chunks.at(integer(action.at("Chunk"), 63));
            if (kind == "snapshot") event = snapshot(chunk);
            else {
                require(chunk != nullptr, "MTMD operation requires live chunk");
                if (kind == "image-position") {
                    require(mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE, "MTMD position requires image");
                    const auto * image = mtmd_input_chunk_get_tokens_image(chunk); require(image != nullptr, "MTMD image is null");
                    const auto index = integer(action.at("Index"), 1048575); require(index < mtmd_image_tokens_get_n_tokens(image), "MTMD image position index outside extent");
                    const auto pos = mtmd_image_tokens_get_decoder_pos(image, static_cast<llama_pos>(integer(action.at("Position0"), 1048576)), index); event = json{{"T", pos.t}, {"X", pos.x}, {"Y", pos.y}, {"Z", pos.z}};
                } else if (kind == "query-save") {
                    std::size_t expected = 0; event["ReturnCode"] = mtmd_input_chunk_save(chunk, nullptr, 0, &expected); event["ExpectedLength"] = expected;
                } else {
                    const auto length = integer(action.at("Length"), 1024*1024), id = integer(action.at("Buffer"), 63);
                    auto & buffer = buffers[id]; buffer.resize(length); event["Allocated"] = true;
                    event["ReturnCode"] = mtmd_input_chunk_save(chunk, buffer.data(), buffer.size(), nullptr);
                }
            }
        }
        events.push_back(std::move(event));
    }
    return json{{"Events", events}};
}
} // namespace prism_mtmd_chunks
