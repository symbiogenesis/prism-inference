#pragma once

// Bounded serialization of supplied GGUF bytes/descriptors into production APIs.
// Input generators, crafted files, ordering checks and expected values are Rust.
#include "gguf.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace prism_gguf {
using json = nlohmann::ordered_json;
constexpr std::size_t max_bytes = 8 * 1024 * 1024;
inline void require(bool condition, const char * message) { if (!condition) throw std::runtime_error(message); }
inline void fields(const json & value, std::initializer_list<const char *> names) {
    require(value.is_object() && value.size() == names.size(), "GGUF fields mismatch");
    for (const auto * name : names) require(value.contains(name), "missing GGUF field");
}
inline std::size_t integer(const json & value, std::uint64_t maximum) {
    require(value.is_number_integer() && (value.is_number_unsigned() || value.get<std::int64_t>() >= 0), "GGUF integer must be nonnegative");
    const auto result = value.get<std::uint64_t>(); require(result <= maximum, "GGUF integer exceeds bound"); return static_cast<std::size_t>(result);
}
inline void text(const json & value, std::size_t maximum) {
    require(value.is_string(), "GGUF text must be string"); const auto & data = value.get_ref<const std::string &>();
    require(data.size() <= maximum && data.find('\0') == std::string::npos, "GGUF text exceeds bound or contains NUL");
}
inline std::uint8_t nibble(char value) {
    require((value >= '0' && value <= '9') || (value >= 'a' && value <= 'f'), "GGUF hex must be lowercase");
    return static_cast<std::uint8_t>(value <= '9' ? value - '0' : value - 'a' + 10);
}
inline std::size_t hex_size(const json & value, std::size_t maximum = max_bytes) {
    require(value.is_string(), "GGUF hex must be string"); const auto & data = value.get_ref<const std::string &>();
    require(data.size() % 2 == 0 && data.size() / 2 <= maximum, "GGUF hex extent exceeds bound");
    for (const auto byte : data) nibble(byte); return data.size() / 2;
}
inline std::vector<std::uint8_t> unhex(const json & value) {
    const auto & data = value.get_ref<const std::string &>(); std::vector<std::uint8_t> result(data.size() / 2);
    for (std::size_t i = 0; i < result.size(); ++i) result[i] = static_cast<std::uint8_t>((nibble(data[2*i]) << 4) | nibble(data[2*i+1]));
    return result;
}
inline std::string hex(const void * data, std::size_t size) {
    require(size <= max_bytes, "GGUF output byte extent exceeds bound"); constexpr char digits[] = "0123456789abcdef";
    const auto * bytes = static_cast<const std::uint8_t *>(data); require(size == 0 || bytes, "GGUF nonempty byte pointer is null");
    std::string result(size * 2, '\0');
    for (std::size_t i = 0; i < size; ++i) { result[2*i] = digits[bytes[i] >> 4]; result[2*i+1] = digits[bytes[i] & 15]; } return result;
}
inline std::size_t scalar_width(enum gguf_type type) {
    switch (type) {
        case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL: return 1;
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: return 2;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: return 8;
        default: throw std::runtime_error("GGUF type has no numeric wire width");
    }
}
inline std::string string_bytes(const json & value) {
    const auto bytes = unhex(value); require(std::find(bytes.begin(), bytes.end(), 0) == bytes.end(), "GGUF setter string contains NUL");
    return std::string(bytes.begin(), bytes.end());
}
inline void mode(const json & value) {
    require(value == "file" || value == "buffer" || value == "callback", "unknown GGUF ReadMode");
}
inline void data_indices(const json & value) {
    require(value.is_array() && value.size() <= 256, "GGUF tensor data selection exceeds bound");
    std::set<std::size_t> selected;
    for (const auto & index : value) require(selected.insert(integer(index, 255)).second, "GGUF tensor data index duplicated");
}
inline void descriptor(const json & value, std::size_t & total) {
    fields(value, {"Entries", "Tensors"}); const auto & entries = value.at("Entries"), & tensors = value.at("Tensors");
    require(entries.is_array() && entries.size() <= 256 && tensors.is_array() && tensors.size() <= 256, "GGUF descriptor count exceeds bound");
    for (const auto & entry : entries) {
        fields(entry, {"Key", "Type", "ArrayType", "Count", "DataHex", "StringsHex"}); text(entry.at("Key"), 4096);
        const auto type = static_cast<gguf_type>(integer(entry.at("Type"), GGUF_TYPE_COUNT - 1));
        const auto count = integer(entry.at("Count"), 1048576); const auto bytes = hex_size(entry.at("DataHex"));
        const auto & strings = entry.at("StringsHex"); require(strings.is_array(), "GGUF StringsHex must be array");
        auto element = type;
        if (type == GGUF_TYPE_ARRAY) {
            element = static_cast<gguf_type>(integer(entry.at("ArrayType"), GGUF_TYPE_COUNT - 1)); require(element != GGUF_TYPE_ARRAY, "nested GGUF arrays unsupported");
        } else { require(entry.at("ArrayType").is_null() && count == 1, "GGUF scalar descriptor mismatch"); }
        if (element == GGUF_TYPE_STRING) {
            require(bytes == 0 && strings.size() == count, "GGUF string descriptor extent mismatch");
            for (const auto & string : strings) { total += hex_size(string, 1024*1024); string_bytes(string); }
        } else {
            require(strings.empty() && bytes == count * scalar_width(element), "GGUF numeric descriptor extent mismatch"); total += bytes;
            if (element == GGUF_TYPE_BOOL) {
                const auto raw = unhex(entry.at("DataHex")); for (const auto byte : raw) require(byte <= 1, "GGUF bool setter byte must be 0 or 1");
            }
        }
        require(total <= max_bytes, "GGUF descriptor bytes exceed bound");
        if (entry.at("Key") == GGUF_KEY_GENERAL_ALIGNMENT) {
            require(type == GGUF_TYPE_UINT32, "GGUF alignment setter must be u32");
            const auto raw = unhex(entry.at("DataHex")); std::uint32_t alignment; std::memcpy(&alignment, raw.data(), sizeof(alignment));
            require(alignment > 0 && alignment <= 4096 && (alignment & (alignment - 1)) == 0, "GGUF setter alignment outside bound");
        }
    }
    std::set<std::string> names;
    for (const auto & tensor : tensors) {
        fields(tensor, {"Name", "Type", "Dimensions", "DataHex"}); text(tensor.at("Name"), GGML_MAX_NAME - 1);
        require(names.insert(tensor.at("Name").get<std::string>()).second, "duplicate GGUF descriptor tensor name");
        const auto type = static_cast<ggml_type>(integer(tensor.at("Type"), GGML_TYPE_COUNT - 1));
        const auto block = ggml_blck_size(type); const auto width = ggml_type_size(type); require(block > 0 && width > 0, "GGUF descriptor tensor type is deprecated");
        const auto & dimensions = tensor.at("Dimensions"); require(dimensions.is_array() && dimensions.size() == GGML_MAX_DIMS, "GGUF tensor must have four dimensions");
        const auto first = integer(dimensions[0], 1048576); require(first > 0 && first % block == 0, "GGUF tensor first dimension not block aligned");
        std::size_t bytes = first / static_cast<std::size_t>(block) * width;
        for (std::size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            const auto extent = integer(dimensions[i], 1048576); require(extent > 0 && bytes <= max_bytes / extent, "GGUF tensor descriptor allocation exceeds bound"); bytes *= extent;
        }
        require(hex_size(tensor.at("DataHex")) == bytes, "GGUF tensor descriptor data extent mismatch"); total += bytes;
        require(total <= max_bytes, "GGUF total descriptor data exceeds bound");
    }
}
inline bool handles(const std::string & kind) { return kind == "gguf-inventory" || kind == "gguf-read" || kind == "gguf-context-trace"; }
inline void validate(const json & operation) {
    const auto kind = operation.at("Kind").get<std::string>();
    if (kind == "gguf-inventory") { fields(operation, {"Kind"}); return; }
    if (kind == "gguf-read") {
        fields(operation, {"Kind", "DataHex", "ReadMode", "LoadTensors", "TensorDataIndices"}); hex_size(operation.at("DataHex")); mode(operation.at("ReadMode"));
        require(operation.at("LoadTensors").is_boolean(), "GGUF LoadTensors must be boolean"); data_indices(operation.at("TensorDataIndices")); return;
    }
    fields(operation, {"Kind", "Backend", "Contexts", "Actions"}); integer(operation.at("Backend"), 63);
    const auto & contexts = operation.at("Contexts"), & actions = operation.at("Actions");
    require(contexts.is_array() && !contexts.empty() && contexts.size() <= 8, "GGUF context count outside bound");
    std::array<bool, 8> live{}; std::size_t total = 0;
    for (std::size_t i = 0; i < contexts.size(); ++i) if (!contexts[i].is_null()) { descriptor(contexts[i], total); live[i] = true; }
    require(actions.is_array() && !actions.empty() && actions.size() <= 32, "GGUF action count outside bound");
    for (const auto & action : actions) {
        require(action.is_object() && action.contains("Op") && action.at("Op").is_string(), "GGUF action Op must be string");
        const auto op = action.at("Op").get<std::string>();
        if (op == "snapshot") {
            fields(action, {"Op", "Context", "TensorDataIndices"}); integer(action.at("Context"), contexts.size()-1); data_indices(action.at("TensorDataIndices"));
        } else if (op == "copy-kv") {
            fields(action, {"Op", "Source", "Destination"}); const auto source = integer(action.at("Source"), contexts.size()-1), destination = integer(action.at("Destination"), contexts.size()-1);
            require(live[source] && live[destination] && source != destination, "GGUF copy requires distinct existing contexts");
        } else if (op == "write-read") {
            fields(action, {"Op", "Context", "Destination", "OnlyMetadata", "ReadMode"}); const auto source = integer(action.at("Context"), contexts.size()-1), destination = integer(action.at("Destination"), contexts.size()-1);
            require(live[source] && source != destination, "GGUF write-read requires distinct live source"); require(action.at("OnlyMetadata").is_boolean(), "GGUF OnlyMetadata must be boolean");
            mode(action.at("ReadMode")); live[destination] = true;
        } else throw std::runtime_error("unknown GGUF action Op");
    }
}
struct context {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer{nullptr, ggml_backend_buffer_free};
    std::unique_ptr<gguf_context, decltype(&gguf_free)> gguf{nullptr, gguf_free};
};
// Exclusive owned scratch directory under the caller's absolute process cwd.
// No process/profile environment is changed and cleanup never recurses.
struct scratch_file {
    std::filesystem::path directory, path;
    std::unique_ptr<FILE, decltype(&std::fclose)> file{nullptr, std::fclose};
    scratch_file() {
        static std::atomic<std::uint64_t> counter{0};
        const auto base = std::filesystem::current_path(); require(base.is_absolute(), "GGUF process cwd must be absolute");
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        for (int attempt = 0; attempt < 32; ++attempt) {
            directory = base / ("prism-gguf-" + std::to_string(stamp) + "-" + std::to_string(counter++));
            std::error_code error;
            if (!std::filesystem::create_directory(directory, error)) continue;
            path = directory / "payload.gguf";
#ifdef _WIN32
            file.reset(_wfopen(path.c_str(), L"w+b"));
#else
            file.reset(std::fopen(path.c_str(), "w+b"));
#endif
            if (!file) { std::filesystem::remove(directory, error); throw std::runtime_error("cannot create GGUF scratch file"); }
            return;
        }
        throw std::runtime_error("cannot create exclusive GGUF scratch directory");
    }
    ~scratch_file() { file.reset(); std::error_code error; std::filesystem::remove(path, error); std::filesystem::remove(directory, error); }
    void rewind() { require(std::fflush(file.get()) == 0 && std::fseek(file.get(), 0, SEEK_SET) == 0, "cannot rewind GGUF scratch file"); }
    std::vector<std::uint8_t> bytes() {
        require(std::fflush(file.get()) == 0 && std::fseek(file.get(), 0, SEEK_END) == 0, "cannot seek GGUF scratch file");
        const auto length = std::ftell(file.get()); require(length >= 0 && static_cast<std::size_t>(length) <= max_bytes, "GGUF scratch file exceeds bound");
        rewind(); std::vector<std::uint8_t> result(static_cast<std::size_t>(length));
        require(result.empty() || std::fread(result.data(), 1, result.size(), file.get()) == result.size(), "cannot read GGUF scratch file"); return result;
    }
};
inline std::size_t callback(void * userdata, void * output, std::uint64_t offset, std::size_t length) {
    const auto & data = *static_cast<const std::vector<std::uint8_t> *>(userdata);
    if (offset >= data.size()) return 0;
    const auto count = std::min(length, data.size() - static_cast<std::size_t>(offset));
    if (count) std::memcpy(output, data.data() + static_cast<std::size_t>(offset), count); return count;
}
inline std::unique_ptr<context> read(const std::string & reader, bool load, scratch_file * file, const std::vector<std::uint8_t> & bytes) {
    auto result = std::make_unique<context>(); ggml_context * tensors = nullptr; const gguf_init_params params{false, load ? &tensors : nullptr};
    if (reader == "file") { require(file != nullptr, "GGUF file reader requires file"); file->rewind(); result->gguf.reset(gguf_init_from_file_ptr(file->file.get(), params)); }
    else if (reader == "buffer") result->gguf.reset(gguf_init_from_buffer(bytes.data(), bytes.size(), params));
    else result->gguf.reset(gguf_init_from_callback(callback, const_cast<std::vector<std::uint8_t> *>(&bytes), 4096, 4ull << 30, params));
    result->tensors.reset(tensors); return result;
}
template<typename T> inline T scalar(const std::vector<std::uint8_t> & bytes) {
    require(bytes.size() == sizeof(T), "GGUF scalar payload width mismatch"); T result; std::memcpy(&result, bytes.data(), sizeof(result)); return result;
}
inline void set_entry(gguf_context * ctx, const json & entry) {
    const auto key = entry.at("Key").get<std::string>(); const auto type = static_cast<gguf_type>(integer(entry.at("Type"), GGUF_TYPE_COUNT - 1));
    const auto bytes = unhex(entry.at("DataHex"));
    if (type == GGUF_TYPE_ARRAY) {
        const auto element = static_cast<gguf_type>(integer(entry.at("ArrayType"), GGUF_TYPE_COUNT - 1)); const auto count = integer(entry.at("Count"), 1048576);
        if (element == GGUF_TYPE_STRING) {
            std::vector<std::string> storage; for (const auto & string : entry.at("StringsHex")) storage.push_back(string_bytes(string));
            std::vector<const char *> pointers; for (const auto & string : storage) pointers.push_back(string.c_str()); gguf_set_arr_str(ctx, key.c_str(), pointers.data(), count);
        } else gguf_set_arr_data(ctx, key.c_str(), element, bytes.data(), count);
        return;
    }
    switch (type) {
        case GGUF_TYPE_UINT8: gguf_set_val_u8(ctx, key.c_str(), scalar<std::uint8_t>(bytes)); break;
        case GGUF_TYPE_INT8: gguf_set_val_i8(ctx, key.c_str(), scalar<std::int8_t>(bytes)); break;
        case GGUF_TYPE_UINT16: gguf_set_val_u16(ctx, key.c_str(), scalar<std::uint16_t>(bytes)); break;
        case GGUF_TYPE_INT16: gguf_set_val_i16(ctx, key.c_str(), scalar<std::int16_t>(bytes)); break;
        case GGUF_TYPE_UINT32: gguf_set_val_u32(ctx, key.c_str(), scalar<std::uint32_t>(bytes)); break;
        case GGUF_TYPE_INT32: gguf_set_val_i32(ctx, key.c_str(), scalar<std::int32_t>(bytes)); break;
        case GGUF_TYPE_FLOAT32: gguf_set_val_f32(ctx, key.c_str(), scalar<float>(bytes)); break;
        case GGUF_TYPE_BOOL: gguf_set_val_bool(ctx, key.c_str(), scalar<std::uint8_t>(bytes) != 0); break;
        case GGUF_TYPE_STRING: { const auto value = string_bytes(entry.at("StringsHex")[0]); gguf_set_val_str(ctx, key.c_str(), value.c_str()); break; }
        case GGUF_TYPE_UINT64: gguf_set_val_u64(ctx, key.c_str(), scalar<std::uint64_t>(bytes)); break;
        case GGUF_TYPE_INT64: gguf_set_val_i64(ctx, key.c_str(), scalar<std::int64_t>(bytes)); break;
        case GGUF_TYPE_FLOAT64: gguf_set_val_f64(ctx, key.c_str(), scalar<double>(bytes)); break;
        default: throw std::runtime_error("invalid GGUF scalar setter type");
    }
}
inline std::unique_ptr<context> create(const json & descriptor, ggml_backend_t backend) {
    auto result = std::make_unique<context>(); result->gguf.reset(gguf_init_empty()); require(bool(result->gguf), "GGUF empty context allocation failed");
    for (const auto & entry : descriptor.at("Entries")) set_entry(result->gguf.get(), entry);
    const auto & tensors = descriptor.at("Tensors");
    if (tensors.empty()) return result;
    const ggml_init_params params{ggml_tensor_overhead() * (tensors.size() + 1), nullptr, true};
    result->tensors.reset(ggml_init(params)); require(bool(result->tensors), "GGUF tensor context allocation failed");
    std::vector<ggml_tensor *> native;
    for (const auto & tensor : tensors) {
        const auto dimensions = tensor.at("Dimensions").get<std::array<std::int64_t, GGML_MAX_DIMS>>();
        auto * value = ggml_new_tensor(result->tensors.get(), static_cast<ggml_type>(tensor.at("Type").get<int>()), GGML_MAX_DIMS, dimensions.data());
        require(value != nullptr, "GGUF tensor construction failed"); ggml_set_name(value, tensor.at("Name").get_ref<const std::string &>().c_str()); native.push_back(value);
    }
    if (!tensors.empty()) {
        result->buffer.reset(ggml_backend_alloc_ctx_tensors(result->tensors.get(), backend)); require(bool(result->buffer), "GGUF backend tensor allocation failed");
        for (std::size_t i = 0; i < tensors.size(); ++i) {
            const auto bytes = unhex(tensors[i].at("DataHex")); ggml_backend_tensor_set(native[i], bytes.data(), 0, bytes.size()); gguf_add_tensor(result->gguf.get(), native[i]);
        }
    }
    return result;
}
inline std::string c_string_hex(const char * value) {
    require(value != nullptr, "GGUF string pointer is null"); const auto size = std::strlen(value); return hex(value, size);
}
inline json snapshot(const context * value, const json & data_selection) {
    const auto * ctx = value ? value->gguf.get() : nullptr; const auto * tensors = value ? value->tensors.get() : nullptr;
    json result{{"ContextExists", ctx != nullptr}, {"GgmlContextExists", tensors != nullptr}, {"Header", nullptr}, {"ContextTensors", nullptr}};
    std::set<std::string> selected;
    if (ctx) {
        const auto nkeys = gguf_get_n_kv(ctx), ntensors = gguf_get_n_tensors(ctx); require(nkeys >= 0 && nkeys <= 1024 && ntensors >= 0 && ntensors <= 256, "GGUF snapshot metadata count exceeds bound");
        json keys = json::array(), infos = json::array();
        for (std::int64_t i = 0; i < nkeys; ++i) {
            const auto type = gguf_get_kv_type(ctx, i), element = type == GGUF_TYPE_ARRAY ? gguf_get_arr_type(ctx, i) : type;
            const auto count = type == GGUF_TYPE_ARRAY ? gguf_get_arr_n(ctx, i) : 1; require(count <= max_bytes, "GGUF snapshot array count exceeds bound");
            json strings = json::array(); std::string data;
            if (element == GGUF_TYPE_STRING) for (std::size_t j = 0; j < count; ++j) strings.push_back(c_string_hex(type == GGUF_TYPE_ARRAY ? gguf_get_arr_str(ctx, i, j) : gguf_get_val_str(ctx, i)));
            else {
                const auto width = scalar_width(element); require(count <= max_bytes / width, "GGUF snapshot numeric data exceeds bound");
                data = hex(type == GGUF_TYPE_ARRAY ? gguf_get_arr_data(ctx, i) : gguf_get_val_data(ctx, i), count * width);
            }
            const auto * key = gguf_get_key(ctx, i);
            keys.push_back({{"Index", i}, {"LookupIndex", gguf_find_key(ctx, key)}, {"Key", key}, {"Type", static_cast<int>(type)}, {"ArrayType", type == GGUF_TYPE_ARRAY ? json(static_cast<int>(element)) : json(nullptr)}, {"Count", count}, {"DataHex", data}, {"StringsHex", strings}});
        }
        for (std::int64_t i = 0; i < ntensors; ++i) {
            const auto * name = gguf_get_tensor_name(ctx, i); const auto * dimensions = gguf_get_tensor_ne(ctx, i); json shape = json::array();
            for (int d = 0; d < GGML_MAX_DIMS; ++d) shape.push_back(dimensions[d]);
            infos.push_back({{"Index", i}, {"LookupIndex", gguf_find_tensor(ctx, name)}, {"Name", name}, {"Type", static_cast<int>(gguf_get_tensor_type(ctx, i))}, {"Dimensions", shape}, {"Offset", gguf_get_tensor_offset(ctx, i)}, {"Bytes", gguf_get_tensor_size(ctx, i)}});
        }
        for (const auto & index : data_selection) { const auto i = integer(index, 255); require(i < static_cast<std::size_t>(ntensors), "GGUF selected tensor outside extent"); selected.insert(gguf_get_tensor_name(ctx, i)); }
        result["Header"] = json{{"Version", gguf_get_version(ctx)}, {"Alignment", gguf_get_alignment(ctx)}, {"DataOffset", gguf_get_data_offset(ctx)}, {"NKeys", nkeys}, {"NTensors", ntensors}, {"Keys", keys}, {"Tensors", infos}};
    }
    if (tensors) {
        json values = json::array(); std::size_t total = 0, count = 0;
        for (auto * tensor = ggml_get_first_tensor(tensors); tensor != nullptr; tensor = ggml_get_next_tensor(tensors, tensor)) {
            require(++count <= 257, "GGUF ggml tensor count exceeds bound"); const auto size = ggml_nbytes(tensor); const auto name = std::string(ggml_get_name(tensor)); json data = nullptr;
            if (selected.count(name)) {
                total += size; require(total <= 2*1024*1024, "GGUF snapshot selected tensor bytes exceed 2 MiB"); std::vector<std::uint8_t> bytes(size);
                if (tensor->buffer) ggml_backend_tensor_get(tensor, bytes.data(), 0, size);
                else { require(size == 0 || tensor->data != nullptr, "GGUF loaded tensor data is null"); if (size) std::memcpy(bytes.data(), tensor->data, size); }
                data = hex(bytes.data(), bytes.size());
            }
            values.push_back({{"Name", name}, {"Bytes", size}, {"DataHex", data}});
        }
        result["ContextTensors"] = std::move(values);
    }
    return result;
}
inline json execute(const json & operation) {
    const auto kind = operation.at("Kind").get<std::string>();
    if (kind == "gguf-inventory") {
        json devices = json::array(); const auto count = ggml_backend_dev_count(); require(count <= 64, "GGUF backend device count exceeds bound");
        for (std::size_t i = 0; i < count; ++i) { const auto dev = ggml_backend_dev_get(i); devices.push_back({{"Index", i}, {"Name", ggml_backend_dev_name(dev)}, {"Type", static_cast<int>(ggml_backend_dev_type(dev))}}); }
        return json{{"Devices", devices}};
    }
    if (kind == "gguf-read") {
        const auto bytes = unhex(operation.at("DataHex")); const auto mode = operation.at("ReadMode").get<std::string>(); std::unique_ptr<scratch_file> file;
        if (mode == "file") { file = std::make_unique<scratch_file>(); require(bytes.empty() || std::fwrite(bytes.data(), 1, bytes.size(), file->file.get()) == bytes.size(), "cannot write GGUF raw scratch bytes"); }
        const auto value = read(mode, operation.at("LoadTensors").get<bool>(), file.get(), bytes); return snapshot(value.get(), operation.at("TensorDataIndices"));
    }
    const auto device = integer(operation.at("Backend"), 63); require(device < ggml_backend_dev_count(), "GGUF backend index outside inventory");
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_dev_init(ggml_backend_dev_get(device), nullptr), ggml_backend_free); require(bool(backend), "GGUF backend initialization failed");
    std::array<std::unique_ptr<context>, 8> contexts;
    const auto & descriptors = operation.at("Contexts");
    for (std::size_t i = 0; i < descriptors.size(); ++i) if (!descriptors[i].is_null()) contexts[i] = create(descriptors[i], backend.get());
    json actions = json::array();
    for (const auto & action : operation.at("Actions")) {
        const auto op = action.at("Op").get<std::string>(); json event{{"Snapshot", nullptr}, {"WriteResult", nullptr}};
        if (op == "snapshot") event["Snapshot"] = snapshot(contexts[integer(action.at("Context"), 7)].get(), action.at("TensorDataIndices"));
        else if (op == "copy-kv") {
            auto * source = contexts[integer(action.at("Source"), 7)].get(), * destination = contexts[integer(action.at("Destination"), 7)].get();
            require(source && source->gguf && destination && destination->gguf, "GGUF copy requires existing contexts"); gguf_set_kv(destination->gguf.get(), source->gguf.get());
        } else {
            auto * source = contexts[integer(action.at("Context"), 7)].get(); require(source && source->gguf, "GGUF writer requires live context");
            scratch_file file; const auto only_meta = action.at("OnlyMetadata").get<bool>(); const auto written = gguf_write_to_file_ptr(source->gguf.get(), file.file.get(), only_meta); event["WriteResult"] = written;
            const auto mode = action.at("ReadMode").get<std::string>(); const auto bytes = mode == "file" ? std::vector<std::uint8_t>() : file.bytes();
            contexts[integer(action.at("Destination"), 7)] = read(mode, !only_meta, &file, bytes);
        }
        actions.push_back(std::move(event));
    }
    return json{{"Actions", actions}};
}
} // namespace prism_gguf
