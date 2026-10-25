// A process boundary for the safe Rust suite to call production upstream ggml.
// No test sources, generated inputs, reference calculations or tolerances live here.
#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
constexpr std::uint32_t max_requests = 64;
constexpr std::uint64_t max_file_bytes = 128ull * 1024 * 1024;
constexpr std::uint64_t max_packed_bytes = 16ull * 1024 * 1024;
constexpr std::uint64_t max_float_count = 4ull * 1024 * 1024;
constexpr std::uint64_t max_total_output_count = 16ull * 1024 * 1024;

struct request {
    std::uint32_t operation, k, m, n;
    std::vector<std::uint8_t> packed;
    std::vector<float> activations;
};

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::uint32_t read_u32(std::istream & input) {
    unsigned char bytes[4];
    input.read(reinterpret_cast<char *>(bytes), sizeof(bytes));
    require(bool(input), "truncated request");
    return std::uint32_t(bytes[0]) | (std::uint32_t(bytes[1]) << 8) |
           (std::uint32_t(bytes[2]) << 16) | (std::uint32_t(bytes[3]) << 24);
}

void write_u32(std::ostream & output, std::uint32_t value) {
    const char bytes[] = {char(value), char(value >> 8), char(value >> 16), char(value >> 24)};
    output.write(bytes, sizeof(bytes));
}

std::uint64_t output_count(const request & item) {
    return item.operation == 1 ? std::uint64_t(item.m) * item.n : std::uint64_t(item.k) * item.m;
}

std::vector<request> read_requests(const std::filesystem::path & path) {
    require(std::filesystem::file_size(path) <= max_file_bytes, "request file exceeds size limit");
    std::ifstream input(path, std::ios::binary);
    require(bool(input), "cannot open request file");
    char magic[8];
    input.read(magic, sizeof(magic));
    require(bool(input) && std::memcmp(magic, "BSPREQ1\n", sizeof(magic)) == 0, "invalid request magic");
    const auto count = read_u32(input);
    require(count > 0 && count <= max_requests, "request count outside 1..64");
    std::vector<request> requests;
    requests.reserve(count);
    std::uint64_t total_output_count = 0;
    for (std::uint32_t i = 0; i < count; ++i) {
        request item;
        item.operation = read_u32(input);
        item.k = read_u32(input);
        item.m = read_u32(input);
        item.n = read_u32(input);
        const auto packed_count = read_u32(input);
        const auto activation_count = read_u32(input);
        require(packed_count <= max_packed_bytes && activation_count <= max_float_count,
                "request payload exceeds size limit");
        if (item.operation == 1) {
            require(item.k > 0 && item.k <= 8192 && item.k % 128 == 0 &&
                    item.m > 0 && item.m <= 4096 && item.n > 0 && item.n <= 256,
                    "invalid PQ2 matrix shape");
            require(std::uint64_t(item.k) * item.m <= 8ull * 1024 * 1024,
                    "PQ2 weights exceed element limit");
            require(packed_count == std::uint64_t(item.k / 128) * item.m * 34 &&
                    activation_count == std::uint64_t(item.k) * item.n,
                    "PQ2 payload does not match shape");
        } else if (item.operation == 2) {
            require(item.k == 128 && item.m > 0 && item.m <= 20000 && item.n == 0,
                    "invalid PTQ1 batch shape");
            require(packed_count == std::uint64_t(item.m) * 28 && activation_count == 0,
                    "PTQ1 payload does not match shape");
        } else {
            throw std::runtime_error("unknown operation");
        }
        require(output_count(item) <= max_float_count, "response exceeds element limit");
        total_output_count += output_count(item);
        require(total_output_count <= max_total_output_count, "responses exceed total element limit");
        item.packed.resize(packed_count);
        input.read(reinterpret_cast<char *>(item.packed.data()), packed_count);
        require(bool(input), "truncated packed payload");
        item.activations.resize(activation_count);
        for (auto & value : item.activations) {
            const auto bits = read_u32(input);
            std::memcpy(&value, &bits, sizeof(value));
        }
        requests.push_back(std::move(item));
    }
    require(input.peek() == std::char_traits<char>::eof() && !input.bad(), "trailing request data");
    return requests;
}

std::vector<float> execute(const request & item) {
    const ggml_init_params parameters = {128u * 1024 * 1024, nullptr, false};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(parameters), ggml_free);
    require(bool(context), "ggml_init failed");
    const auto type = item.operation == 1 ? GGML_TYPE_PQ2_0 : GGML_TYPE_PTQ1_0;
    auto * packed = ggml_new_tensor_2d(context.get(), type, item.k, item.m);
    require(ggml_nbytes(packed) == item.packed.size(), "production packed layout mismatch");
    std::memcpy(packed->data, item.packed.data(), item.packed.size());
    std::vector<float> result(static_cast<std::size_t>(output_count(item)));
    if (item.operation == 1) {
        auto * activations = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, item.k, item.n);
        std::memcpy(activations->data, item.activations.data(), item.activations.size() * sizeof(float));
        auto * product = ggml_mul_mat(context.get(), packed, activations);
        auto * graph = ggml_new_graph(context.get());
        ggml_build_forward_expand(graph, product);
        require(ggml_graph_compute_with_ctx(context.get(), graph, 4) == GGML_STATUS_SUCCESS,
                "ggml graph execution failed");
        require(ggml_nbytes(product) == result.size() * sizeof(float), "production output layout mismatch");
        std::memcpy(result.data(), product->data, result.size() * sizeof(float));
    } else {
        const auto * traits = ggml_get_type_traits(GGML_TYPE_PTQ1_0);
        require(traits && traits->to_float, "production PTQ1 decoder unavailable");
        traits->to_float(packed->data, result.data(), static_cast<std::int64_t>(result.size()));
    }
    return result;
}
} // namespace

int main(int argc, char ** argv) {
    try {
        static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                      "protocol requires IEEE-754 binary32");
        const std::uint16_t endian = 1;
        require(*reinterpret_cast<const unsigned char *>(&endian) == 1,
                "packed ggml protocol requires a little-endian host");
        require(argc == 5 && std::strcmp(argv[1], "--input") == 0 && std::strcmp(argv[3], "--output") == 0,
                "usage: prism-upstream-bridge --input ABSOLUTE_PATH --output ABSOLUTE_PATH");
        const std::filesystem::path input_path(argv[2]), output_path(argv[4]);
        require(input_path.is_absolute() && output_path.is_absolute(), "protocol paths must be absolute");
        require(input_path.lexically_normal() != output_path.lexically_normal(), "input and output paths must differ");
        const auto requests = read_requests(input_path);
        std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
        require(bool(output), "cannot open response file");
        output.exceptions(std::ios::badbit | std::ios::failbit);
        output.write("BSPRES1\n", 8);
        write_u32(output, static_cast<std::uint32_t>(requests.size()));
        for (const auto & item : requests) {
            const auto result = execute(item);
            write_u32(output, static_cast<std::uint32_t>(result.size()));
            for (const auto value : result) {
                std::uint32_t bits;
                std::memcpy(&bits, &value, sizeof(bits));
                write_u32(output, bits);
            }
        }
        output.close();
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "prism-upstream-bridge: %s\n", error.what());
        return 1;
    }
}
