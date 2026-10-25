// Generic typed tensor graph transport. Cases, generators and assertions live in Rust.
#pragma once
#include "ggml.h"
#include "ggml-cpu.h"
#include <nlohmann/json.hpp>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace prism_tensor_graph {
using json = nlohmann::ordered_json;
constexpr std::uint64_t max_elements = 2 * 1024 * 1024;
constexpr std::uint64_t max_storage = 96 * 1024 * 1024;
struct descriptor {
    std::array<std::int64_t, 4> ne = {1, 1, 1, 1};
    ggml_type type = GGML_TYPE_F32;
    bool contiguous = true;
    std::uint64_t count() const { return std::uint64_t(ne[0]) * ne[1] * ne[2] * ne[3]; }
    std::uint64_t bytes() const { return count() * (type == GGML_TYPE_F16 || type == GGML_TYPE_BF16 ? 2 : 4); }
};
inline void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}
inline void keys(const json & value, std::initializer_list<const char *> expected) {
    require(value.is_object() && value.size() == expected.size(), "tensor graph fields mismatch");
    for (const auto * key : expected) require(value.contains(key), "tensor graph missing field");
}
inline std::int64_t integer(const json & value, std::int64_t low, std::int64_t high) {
    require(value.is_number_integer(), "tensor graph requires integer");
    if (value.is_number_unsigned()) {
        const auto number = value.get<std::uint64_t>();
        require(number <= std::uint64_t(high), "tensor graph integer above bound");
        require(low <= 0 || number >= std::uint64_t(low), "tensor graph integer below bound");
        return static_cast<std::int64_t>(number);
    }
    const auto number = value.get<std::int64_t>();
    require(number >= low && number <= high, "tensor graph integer outside bound");
    return number;
}
inline float real(const json & value, float low, float high) {
    require(value.is_number(), "tensor graph requires real number");
    const auto number = value.get<float>();
    require(std::isfinite(number) && number >= low && number <= high, "tensor graph real outside bound");
    return number;
}
inline ggml_type type(const json & value) {
    require(value.is_string(), "tensor type must be text");
    const auto name = value.get<std::string>();
    if (name == "F32") return GGML_TYPE_F32;
    if (name == "I32") return GGML_TYPE_I32;
    if (name == "F16") return GGML_TYPE_F16;
    if (name == "BF16") return GGML_TYPE_BF16;
    throw std::runtime_error("unsupported tensor transport type");
}
inline const char * type_name(ggml_type value) {
    switch (value) {
        case GGML_TYPE_F32: return "F32";
        case GGML_TYPE_I32: return "I32";
        case GGML_TYPE_F16: return "F16";
        case GGML_TYPE_BF16: return "BF16";
        default: throw std::runtime_error("unsupported tensor response type");
    }
}
inline descriptor dimensions(const json & value, ggml_type kind) {
    require(value.is_array() && !value.empty() && value.size() <= 4, "tensor requires 1..4 dimensions");
    descriptor result;
    result.type = kind;
    for (std::size_t i = 0; i < value.size(); ++i) result.ne[i] = integer(value[i], 1, 8192);
    require(result.count() <= max_elements, "tensor element count exceeds bound");
    return result;
}
inline unsigned nibble(char digit) {
    if (digit >= '0' && digit <= '9') return unsigned(digit - '0');
    if (digit >= 'a' && digit <= 'f') return unsigned(digit - 'a' + 10);
    if (digit >= 'A' && digit <= 'F') return unsigned(digit - 'A' + 10);
    throw std::runtime_error("invalid tensor hex byte");
}
inline std::vector<unsigned char> unhex(const std::string & text) {
    require(text.size() % 2 == 0 && text.size() <= 16 * 1024 * 1024, "invalid tensor hex extent");
    std::vector<unsigned char> bytes(text.size() / 2);
    for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<unsigned char>((nibble(text[2*i]) << 4) | nibble(text[2*i+1]));
    return bytes;
}
inline std::string hex(const void * data, std::size_t count) {
    constexpr char digits[] = "0123456789abcdef";
    const auto * bytes = static_cast<const unsigned char *>(data);
    std::string result(count * 2, '0');
    for (std::size_t i = 0; i < count; ++i) { result[2*i] = digits[bytes[i] >> 4]; result[2*i+1] = digits[bytes[i] & 15]; }
    return result;
}
inline std::vector<std::size_t> references(const json & node, std::size_t available, std::size_t count) {
    const auto & inputs = node.at("Inputs");
    require(inputs.is_array() && inputs.size() == count && available > 0, "tensor input reference count mismatch");
    std::vector<std::size_t> result;
    for (const auto & input : inputs) result.push_back(static_cast<std::size_t>(integer(input, 0, static_cast<std::int64_t>(available - 1))));
    return result;
}
inline bool handles(const std::string & kind) { return kind == "tensor-graph"; }
inline std::vector<descriptor> plan(const json & operation) {
    keys(operation, {"Kind", "Threads", "Nodes", "Outputs"});
    require(operation.at("Kind") == "tensor-graph", "unknown tensor graph Kind");
    integer(operation.at("Threads"), 1, 16);
    const auto & nodes = operation.at("Nodes");
    require(nodes.is_array() && !nodes.empty() && nodes.size() <= 96, "tensor graph requires 1..96 nodes");
    std::vector<descriptor> shapes;
    std::uint64_t storage = 0;
    for (const auto & node : nodes) {
        require(node.is_object() && node.contains("Op") && node.at("Op").is_string(), "node Op must be text");
        const auto op = node.at("Op").get<std::string>();
        descriptor output;
        if (op == "input") {
            keys(node, {"Op", "Type", "Dimensions", "DataHex"});
            output = dimensions(node.at("Dimensions"), type(node.at("Type")));
            require(output.type == GGML_TYPE_F32 || output.type == GGML_TYPE_I32, "input transport supports F32/I32");
            require(node.at("DataHex").is_string(), "input DataHex must be text");
            const auto bytes = unhex(node.at("DataHex").get_ref<const std::string &>());
            require(bytes.size() == output.bytes(), "tensor input bytes do not match dimensions/type");
            if (output.type == GGML_TYPE_F32) for (std::size_t i = 0; i < bytes.size(); i += 4) {
                float value; std::memcpy(&value, bytes.data() + i, 4); require(std::isfinite(value), "nonfinite graph input");
            }
        } else {
            const bool binary = op == "mul_mat" || op == "add" || op == "mul" || op == "conv_transpose_1d" || op == "rope" || op == "rope_multi";
            const auto refs = references(node, shapes.size(), binary ? 2 : 1);
            const auto & a = shapes[refs[0]];
            const auto & b = shapes[refs[binary ? 1 : 0]];
            output = a;
            if (op == "cont" || op == "transpose" || op == "soft_max") {
                keys(node, {"Op", "Inputs"});
                if (op == "transpose") { std::swap(output.ne[0], output.ne[1]); output.contiguous = false; }
                else { output.contiguous = true; if (op == "soft_max") require(a.type == GGML_TYPE_F32 && a.contiguous, "softmax requires contiguous F32"); }
            } else if (op == "permute") {
                keys(node, {"Op", "Inputs", "Axes"});
                const auto & axes = node.at("Axes"); require(axes.is_array() && axes.size() == 4, "permute needs four axes");
                std::set<int> unique;
                for (int i = 0; i < 4; ++i) { const int axis = static_cast<int>(integer(axes[i], 0, 3)); unique.insert(axis); output.ne[axis] = a.ne[i]; }
                require(unique.size() == 4, "permute axes must be unique"); output.contiguous = false;
            } else if (op == "reshape") {
                keys(node, {"Op", "Inputs", "Dimensions"}); output = dimensions(node.at("Dimensions"), a.type);
                require(a.contiguous && output.count() == a.count(), "reshape requires matching contiguous extent");
            } else if (op == "cast") {
                keys(node, {"Op", "Inputs", "Type"}); output.type = type(node.at("Type")); output.contiguous = true;
                require(a.type == GGML_TYPE_F32 && output.type != GGML_TYPE_I32, "unsupported graph cast");
            } else if (op == "mul_mat" || op == "add" || op == "mul") {
                keys(node, {"Op", "Inputs"}); require(a.type == GGML_TYPE_F32 && b.type == GGML_TYPE_F32, "arithmetic requires F32");
                if (op == "mul_mat") {
                    require(a.contiguous && a.ne[0] == b.ne[0] && b.ne[2] % a.ne[2] == 0 && b.ne[3] % a.ne[3] == 0, "incompatible matmul shapes");
                    output.ne = {a.ne[1], b.ne[1], b.ne[2], b.ne[3]};
                } else for (int axis = 0; axis < 4; ++axis) require(a.ne[axis] % b.ne[axis] == 0, "incompatible broadcast shapes");
                output.contiguous = true;
            } else if (op == "rms_norm") {
                keys(node, {"Op", "Inputs", "Epsilon"}); require(a.type == GGML_TYPE_F32 && a.contiguous, "RMS norm requires contiguous F32");
                real(node.at("Epsilon"), 0, 1); output.contiguous = true;
            } else if (op == "view_2d") {
                keys(node, {"Op", "Inputs", "Dimensions", "RowStrideBytes", "OffsetBytes"});
                require(a.contiguous && a.type == GGML_TYPE_F32 && node.at("Dimensions").size() == 2, "view requires contiguous F32 and two dimensions");
                output = dimensions(node.at("Dimensions"), a.type);
                const auto stride = integer(node.at("RowStrideBytes"), 1, max_storage), offset = integer(node.at("OffsetBytes"), 0, max_storage);
                require(stride % 4 == 0 && offset % 4 == 0 && stride >= output.ne[0] * 4 && std::uint64_t(offset + (output.ne[1]-1)*stride + output.ne[0]*4) <= a.bytes(), "view exceeds source byte extent");
                output.contiguous = stride == output.ne[0] * 4;
            } else if (op == "conv_transpose_1d") {
                keys(node, {"Op", "Inputs", "Stride", "Padding", "Dilation"});
                const auto stride = integer(node.at("Stride"), 1, 4096);
                require(integer(node.at("Padding"), 0, 8192) == 0 && integer(node.at("Dilation"), 1, 4096) == 1, "unsupported transposed convolution padding/dilation");
                require(a.type == GGML_TYPE_F32 && b.type == GGML_TYPE_F32 && a.ne[2] == b.ne[1] && a.ne[3] == 1 && b.ne[2] == 1 && b.ne[3] == 1 && a.contiguous && b.contiguous, "incompatible transposed convolution tensors");
                output.ne = {(b.ne[0]-1)*stride + a.ne[0], a.ne[1], 1, 1}; output.contiguous = true;
            } else if (op == "col2im_1d") {
                keys(node, {"Op", "Inputs", "Stride", "Channels", "Padding"});
                const auto stride = integer(node.at("Stride"), 1, 4096), channels = integer(node.at("Channels"), 1, 4096), padding = integer(node.at("Padding"), 0, 8192);
                require(a.contiguous && a.type != GGML_TYPE_I32 && a.ne[2] == 1 && a.ne[3] == 1 && a.ne[0] % channels == 0, "incompatible col2im tensor");
                const auto length = (a.ne[1]-1)*stride + a.ne[0]/channels - 2*padding;
                require(length > 0, "col2im crop removes output"); output.ne = {length, channels, 1, 1}; output.contiguous = true;
            } else if (op == "rope" || op == "rope_multi") {
                if (op == "rope") keys(node, {"Op", "Inputs", "Rotations", "Mode"});
                else keys(node, {"Op", "Inputs", "Rotations", "Mode", "Sections", "Context", "FrequencyBase", "FrequencyScale", "ExtensionFactor", "AttentionFactor", "BetaFast", "BetaSlow"});
                const auto rotations = integer(node.at("Rotations"), 2, 8192), mode = integer(node.at("Mode"), 0, 40);
                require(a.type == GGML_TYPE_F32 && a.contiguous && rotations <= a.ne[0] && rotations % 2 == 0 && a.ne[0] % 2 == 0 && b.type == GGML_TYPE_I32 && b.ne[1] == 1 && b.ne[2] == 1 && b.ne[3] == 1, "incompatible RoPE tensors");
                if (op == "rope") require((mode == 0 || mode == 2) && b.ne[0] == a.ne[2], "invalid ordinary RoPE mode/positions");
                else {
                    require((mode == 8 || mode == 24 || mode == 40) && b.ne[0] == 4*a.ne[2], "invalid multidimensional RoPE mode/positions");
                    const auto & sections = node.at("Sections"); require(sections.is_array() && sections.size() == 4, "RoPE requires four sections");
                    std::int64_t sum = 0; for (const auto & section : sections) sum += integer(section, 0, rotations);
                    require(sum == rotations/2 && (mode != 24 || a.ne[0] == 2*rotations), "incompatible RoPE sections");
                    integer(node.at("Context"), 1, 1048576);
                    real(node.at("FrequencyBase"), 1, 1e9f); real(node.at("FrequencyScale"), 1e-9f, 1e9f);
                    real(node.at("ExtensionFactor"), 0, 1e9f); real(node.at("AttentionFactor"), 1e-9f, 1e9f);
                    real(node.at("BetaFast"), 0, 1e9f); real(node.at("BetaSlow"), 0, 1e9f);
                }
                output.contiguous = true;
            } else throw std::runtime_error("unknown tensor graph Op");
        }
        for (const auto extent : output.ne) require(extent > 0 && extent <= 8192, "graph tensor dimension outside bound");
        require(output.count() <= max_elements, "graph tensor exceeds element bound");
        storage += output.bytes(); require(storage <= max_storage, "graph storage estimate exceeds bound");
        shapes.push_back(output);
    }
    const auto & outputs = operation.at("Outputs");
    require(outputs.is_array() && !outputs.empty() && outputs.size() <= 16, "graph needs 1..16 outputs");
    std::uint64_t output_bytes = 0;
    for (const auto & index : outputs) {
        const auto & shape = shapes[static_cast<std::size_t>(integer(index, 0, static_cast<std::int64_t>(shapes.size()-1)))];
        require(shape.contiguous, "requested graph outputs must be contiguous"); output_bytes += shape.bytes();
    }
    require(output_bytes * 2 <= 15 * 1024 * 1024, "graph response hex exceeds bound");
    return shapes;
}
inline void validate(const json & operation) { (void) plan(operation); }
inline json execute(const json & operation) {
    const std::uint16_t endian = 1;
    require(*reinterpret_cast<const unsigned char *>(&endian) == 1, "tensor graph requires little-endian host");
    const auto shapes = plan(operation);
    const ggml_init_params params = {128u * 1024 * 1024, nullptr, false};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(params), ggml_free);
    require(bool(context), "graph context allocation failed");
    std::vector<ggml_tensor *> tensors;
    for (const auto & node : operation.at("Nodes")) {
        const auto op = node.at("Op").get<std::string>();
        ggml_tensor * output = nullptr;
        if (op == "input") {
            const auto & shape = shapes[tensors.size()];
            output = ggml_new_tensor(context.get(), shape.type, static_cast<int>(node.at("Dimensions").size()), shape.ne.data());
            const auto bytes = unhex(node.at("DataHex").get_ref<const std::string &>());
            std::memcpy(output->data, bytes.data(), bytes.size());
        } else {
            const auto & inputs = node.at("Inputs");
            auto * a = tensors[inputs[0].get<std::size_t>()];
            auto * b = inputs.size() == 2 ? tensors[inputs[1].get<std::size_t>()] : nullptr;
            const auto & shape = shapes[tensors.size()];
            if (op == "cont") output = ggml_cont(context.get(), a);
            else if (op == "transpose") output = ggml_transpose(context.get(), a);
            else if (op == "soft_max") output = ggml_soft_max(context.get(), a);
            else if (op == "permute") { const auto axes = node.at("Axes").get<std::array<int,4>>(); output = ggml_permute(context.get(), a, axes[0], axes[1], axes[2], axes[3]); }
            else if (op == "reshape") {
                const auto dims = node.at("Dimensions").size();
                if (dims == 1) output = ggml_reshape_1d(context.get(), a, shape.ne[0]);
                else if (dims == 2) output = ggml_reshape_2d(context.get(), a, shape.ne[0], shape.ne[1]);
                else if (dims == 3) output = ggml_reshape_3d(context.get(), a, shape.ne[0], shape.ne[1], shape.ne[2]);
                else output = ggml_reshape_4d(context.get(), a, shape.ne[0], shape.ne[1], shape.ne[2], shape.ne[3]);
            } else if (op == "cast") output = ggml_cast(context.get(), a, shape.type);
            else if (op == "mul_mat") output = ggml_mul_mat(context.get(), a, b);
            else if (op == "add") output = ggml_add(context.get(), a, b);
            else if (op == "mul") output = ggml_mul(context.get(), a, b);
            else if (op == "rms_norm") output = ggml_rms_norm(context.get(), a, node.at("Epsilon").get<float>());
            else if (op == "view_2d") output = ggml_view_2d(context.get(), a, shape.ne[0], shape.ne[1], node.at("RowStrideBytes").get<std::size_t>(), node.at("OffsetBytes").get<std::size_t>());
            else if (op == "conv_transpose_1d") output = ggml_conv_transpose_1d(context.get(), a, b, node.at("Stride").get<int>(), node.at("Padding").get<int>(), node.at("Dilation").get<int>());
            else if (op == "col2im_1d") output = ggml_col2im_1d(context.get(), a, node.at("Stride").get<int>(), node.at("Channels").get<int>(), node.at("Padding").get<int>());
            else if (op == "rope") output = ggml_rope(context.get(), a, b, node.at("Rotations").get<int>(), node.at("Mode").get<int>());
            else if (op == "rope_multi") {
                auto sections = node.at("Sections").get<std::array<int,4>>();
                output = ggml_rope_multi(context.get(), a, b, nullptr, node.at("Rotations").get<int>(), sections.data(), node.at("Mode").get<int>(), node.at("Context").get<int>(), node.at("FrequencyBase").get<float>(), node.at("FrequencyScale").get<float>(), node.at("ExtensionFactor").get<float>(), node.at("AttentionFactor").get<float>(), node.at("BetaFast").get<float>(), node.at("BetaSlow").get<float>());
            }
        }
        require(output != nullptr, "graph node creation failed");
        const auto & expected = shapes[tensors.size()];
        require(output->type == expected.type && std::equal(expected.ne.begin(), expected.ne.end(), output->ne), "graph node metadata differs from validated descriptor");
        tensors.push_back(output);
    }
    auto * graph = ggml_new_graph_custom(context.get(), 4096, false);
    for (const auto & index : operation.at("Outputs")) ggml_build_forward_expand(graph, tensors[index.get<std::size_t>()]);
    require(ggml_graph_compute_with_ctx(context.get(), graph, operation.at("Threads").get<int>()) == GGML_STATUS_SUCCESS, "production graph compute failed");
    json outputs = json::array();
    for (const auto & index : operation.at("Outputs")) {
        const auto * tensor = tensors[index.get<std::size_t>()];
        require(ggml_is_contiguous(tensor), "graph output is not contiguous");
        outputs.push_back(json{{"Type", type_name(tensor->type)}, {"Dimensions", std::vector<std::int64_t>(tensor->ne, tensor->ne+4)}, {"DataHex", hex(tensor->data, ggml_nbytes(tensor))}});
    }
    return json{{"Outputs", outputs}};
}
} // namespace prism_tensor_graph
