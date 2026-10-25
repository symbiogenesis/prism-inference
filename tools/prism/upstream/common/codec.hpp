// Typed byte transport and production codec calls only. Assertions live in Rust.
#pragma once

#include "ggml.h"
#include "ggml-cpu.h"
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace prism_codec {
using json = nlohmann::ordered_json;

inline void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

inline void keys(const json & value, std::initializer_list<const char *> names) {
    require(value.is_object() && value.size() == names.size(), "codec object fields mismatch");
    for (const auto * name : names) require(value.contains(name), "missing codec field");
}

inline std::size_t integer(const json & value, std::size_t maximum) {
    require(value.is_number_integer() && (value.is_number_unsigned() || value.get<std::int64_t>() >= 0),
            "codec integer must be nonnegative");
    const auto number = value.get<std::uint64_t>();
    require(number <= maximum, "codec integer exceeds bound");
    return static_cast<std::size_t>(number);
}

inline ggml_type type(const json & operation) {
    return static_cast<ggml_type>(integer(operation.at("TypeId"), GGML_TYPE_COUNT - 1));
}

inline unsigned nibble(char character) {
    if (character >= '0' && character <= '9') return static_cast<unsigned>(character - '0');
    if (character >= 'a' && character <= 'f') return static_cast<unsigned>(character - 'a' + 10);
    throw std::runtime_error("codec DataHex must use lowercase hexadecimal");
}

inline std::vector<std::uint8_t> bytes(const json & value, std::size_t expected) {
    require(value.is_string() && expected <= 1024 * 1024, "codec byte extent exceeds bound");
    const auto & text = value.get_ref<const std::string &>();
    require(text.size() == expected * 2, "codec DataHex extent mismatch");
    std::vector<std::uint8_t> result(expected);
    for (std::size_t i = 0; i < expected; ++i) {
        result[i] = static_cast<std::uint8_t>(16 * nibble(text[2 * i]) + nibble(text[2 * i + 1]));
    }
    return result;
}

inline std::string hex(const void * data, std::size_t size) {
    require(size <= 1024 * 1024, "codec output exceeds byte bound");
    constexpr char digits[] = "0123456789abcdef";
    const auto * input = static_cast<const std::uint8_t *>(data);
    std::string result(size * 2, '\0');
    for (std::size_t i = 0; i < size; ++i) {
        result[2 * i] = digits[input[i] >> 4];
        result[2 * i + 1] = digits[input[i] & 15];
    }
    return result;
}

inline std::size_t extent(ggml_type kind, std::size_t count) {
    const auto * traits = ggml_get_type_traits(kind);
    require(traits->blck_size > 0 && traits->blck_size <= 4096 && traits->type_size > 0 &&
            traits->type_size <= 4096, "codec type has no active block layout");
    const auto block = static_cast<std::size_t>(traits->blck_size);
    require(count > 0 && count <= 8192 && count % block == 0, "codec count must contain whole blocks");
    const auto size = count / block * traits->type_size;
    require(size <= 1024 * 1024, "codec packed byte extent exceeds bound");
    return size;
}

// C codec implementations may access their payload as SIMD words or structs.
struct aligned_bytes {
    std::vector<std::max_align_t> storage;
    explicit aligned_bytes(std::size_t size) : storage((size + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t)) {}
    explicit aligned_bytes(const std::vector<std::uint8_t> & source) : aligned_bytes(source.size()) {
        std::memcpy(storage.data(), source.data(), source.size());
    }
    void * data() { return storage.data(); }
};

inline std::vector<float> floats(const json & value, std::size_t count) {
    const auto raw = bytes(value, count * sizeof(float));
    std::vector<float> result(count);
    std::memcpy(result.data(), raw.data(), raw.size());
    for (const float number : result) require(std::isfinite(number), "codec float input must be finite");
    return result;
}

inline bool handles(const std::string & kind) {
    return kind == "codec-inventory" || kind == "codec-quantize" ||
           kind == "codec-dequantize" || kind == "codec-vec-dot";
}

inline void validate(const json & operation) {
    const std::uint32_t endian_probe = 1;
    require(*reinterpret_cast<const unsigned char *>(&endian_probe) == 1,
            "codec transport requires little-endian production host");
    require(operation.is_object() && operation.contains("Kind") && operation.at("Kind").is_string(),
            "codec Kind must be text");
    const auto kind = operation.at("Kind").get<std::string>();
    require(handles(kind), "unknown codec Kind");
    if (kind == "codec-inventory") {
        keys(operation, {"Kind"});
        return;
    }
    if (kind == "codec-quantize") keys(operation, {"Kind", "TypeId", "Mode", "Count", "DataHex"});
    else if (kind == "codec-dequantize") keys(operation, {"Kind", "TypeId", "Count", "DataHex"});
    else keys(operation, {"Kind", "TypeId", "Count", "WeightsHex", "ActivationsHex"});
    const auto id = type(operation);
    const auto count = integer(operation.at("Count"), 8192);
    const auto size = extent(id, count);
    const auto * traits = ggml_get_type_traits(id);
    const auto * cpu = ggml_get_type_traits_cpu(id);
    if (kind == "codec-quantize") {
        require(operation.at("Mode").is_string(), "codec quantizer Mode must be text");
        const auto mode = operation.at("Mode").get<std::string>();
        require((mode == "cpu" && cpu->from_float) || (mode == "reference" && traits->from_float_ref),
                "requested production quantizer is unavailable");
        floats(operation.at("DataHex"), count);
    } else if (kind == "codec-dequantize") {
        require(traits->to_float != nullptr, "production decoder is unavailable");
        bytes(operation.at("DataHex"), size);
    } else {
        require(cpu->vec_dot != nullptr && cpu->vec_dot_type >= 0 && cpu->vec_dot_type < GGML_TYPE_COUNT,
                "production CPU dot is unavailable");
        bytes(operation.at("WeightsHex"), size);
        bytes(operation.at("ActivationsHex"), extent(cpu->vec_dot_type, count));
    }
}

inline json execute(const json & operation) {
    ggml_cpu_init();
    const auto kind = operation.at("Kind").get<std::string>();
    if (kind == "codec-inventory") {
        json types = json::array();
        for (int id = 0; id < GGML_TYPE_COUNT; ++id) {
            const auto type_id = static_cast<ggml_type>(id);
            const auto * traits = ggml_get_type_traits(type_id);
            const auto * cpu = ggml_get_type_traits_cpu(type_id);
            if (traits->blck_size != 0) ggml_quantize_init(type_id);
            types.push_back(json{{"TypeId", id}, {"Name", traits->type_name ? json(traits->type_name) : json(nullptr)},
                {"BlockSize", traits->blck_size}, {"TypeSize", traits->type_size},
                {"HasFromFloatCpu", cpu->from_float != nullptr}, {"HasFromFloatRef", traits->from_float_ref != nullptr},
                {"HasToFloat", traits->to_float != nullptr}, {"HasVecDot", cpu->vec_dot != nullptr},
                {"VecDotType", static_cast<int>(cpu->vec_dot_type)}});
        }
        return json{{"Types", types}};
    }
    const auto id = type(operation);
    const auto count = integer(operation.at("Count"), 8192);
    const auto size = extent(id, count);
    const auto * traits = ggml_get_type_traits(id);
    const auto * cpu = ggml_get_type_traits_cpu(id);
    ggml_quantize_init(id);
    if (kind == "codec-quantize") {
        const auto input = floats(operation.at("DataHex"), count);
        aligned_bytes output(size);
        const auto function = operation.at("Mode") == "cpu" ? cpu->from_float : traits->from_float_ref;
        function(input.data(), output.data(), static_cast<std::int64_t>(count));
        return json{{"DataHex", hex(output.data(), size)}};
    }
    if (kind == "codec-dequantize") {
        aligned_bytes input(bytes(operation.at("DataHex"), size));
        std::vector<float> output(count);
        traits->to_float(input.data(), output.data(), static_cast<std::int64_t>(count));
        return json{{"DataHex", hex(output.data(), count * sizeof(float))}};
    }
    aligned_bytes weights(bytes(operation.at("WeightsHex"), size));
    aligned_bytes activations(bytes(operation.at("ActivationsHex"), extent(cpu->vec_dot_type, count)));
    float output = std::numeric_limits<float>::infinity();
    cpu->vec_dot(static_cast<int>(count), &output, 0, weights.data(), 0, activations.data(), 0, 1);
    return json{{"DataHex", hex(&output, sizeof(output))}};
}
} // namespace prism_codec
