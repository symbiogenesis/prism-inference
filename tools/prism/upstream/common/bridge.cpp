// JSON serialization and production API dispatch only. The safe Rust suite supplies
// inputs, expected results and assertions; no upstream test source is compiled.
#include "speculative.h"
#include "unicode.h"
#include "jinja.hpp"
#include "mtmd.hpp"
#include "grammar_parser.hpp"
#include "reasoning_budget.hpp"
#include "tensor_graph.hpp"
#include "grammar.hpp"
#include "codec.hpp"
#include "peg.hpp"
// MTMD publishes its own logging macros; the concurrency adapter uses common's.
#undef LOG_DBG
#undef LOG_TRC
#undef LOG_INF
#undef LOG_WRN
#undef LOG_ERR
#undef LOG_CNT
#include "concurrency.hpp"
#include "arena_graph.hpp"
#include "batch_allocator.hpp"
#include "allocation.hpp"
#include "mtmd_chunks.hpp"
#include "sampling.hpp"
#include "gguf.hpp"
#include "chat_templates.hpp"
#include "chat_parser.hpp"
#include "chat_auto.hpp"
#include "vocab.hpp"
#include "model_runtime.hpp"
#include "json_schema.hpp"
#include "chat_analysis.hpp"
#include "arg_parser.hpp"
#include "chat.hpp"
#include "optimizer.hpp"
#include "quantization.hpp"
#include "dspark.hpp"
#include "threadpool.hpp"
#include "backend_ops.hpp"
#include "backend_session.hpp"
#include "public_c_abi.h"
#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using wire_json = nlohmann::ordered_json;
constexpr std::size_t max_file_bytes = 16 * 1024 * 1024;
constexpr std::size_t max_response_bytes = 16 * 1024 * 1024;
constexpr std::size_t max_operations = 128;
constexpr std::size_t max_text_bytes = 64 * 1024;
constexpr std::size_t max_expression_bytes = 8192;
constexpr std::size_t max_expressions = 16;

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_keys(const wire_json & object, std::initializer_list<const char *> keys) {
    require(object.is_object() && object.size() == keys.size(), "invalid object fields");
    for (const auto * key : keys) {
        require(object.contains(key), "missing or unknown object field");
    }
}

std::int32_t integer(const wire_json & value) {
    require(value.is_number_integer(), "expected signed 32-bit integer");
    if (value.is_number_unsigned()) {
        const auto number = value.get<std::uint64_t>();
        require(number <= std::uint64_t(std::numeric_limits<std::int32_t>::max()),
                "integer outside signed 32-bit range");
        return static_cast<std::int32_t>(number);
    }
    const auto number = value.get<std::int64_t>();
    require(number >= std::numeric_limits<std::int32_t>::min() &&
            number <= std::numeric_limits<std::int32_t>::max(),
            "integer outside signed 32-bit range");
    return static_cast<std::int32_t>(number);
}

wire_json read_request(const std::filesystem::path & path) {
    const auto size = std::filesystem::file_size(path);
    require(size > 0 && size <= max_file_bytes, "request size must be 1 byte through 16 MiB");
    std::ifstream input(path, std::ios::binary);
    require(bool(input), "cannot open request file");
    std::string content(static_cast<std::size_t>(size), '\0');
    input.read(content.data(), static_cast<std::streamsize>(content.size()));
    require(bool(input) && input.peek() == std::char_traits<char>::eof() && !input.bad(),
            "request size changed during read");
    std::vector<std::set<std::string>> object_keys;
    const auto callback = [&](int depth, wire_json::parse_event_t event, wire_json & parsed) {
        require(depth <= 160, "JSON nesting exceeds limit");
        if (event == wire_json::parse_event_t::object_start) {
            object_keys.emplace_back();
        } else if (event == wire_json::parse_event_t::key) {
            require(!object_keys.empty() && object_keys.back().insert(parsed.get<std::string>()).second,
                    "duplicate JSON object key");
        } else if (event == wire_json::parse_event_t::object_end) {
            require(!object_keys.empty(), "invalid JSON object nesting");
            object_keys.pop_back();
        }
        return true;
    };
    return wire_json::parse(content, callback);
}

void validate_operation(const wire_json & operation) {
    require(operation.is_object() && operation.contains("Kind") && operation.at("Kind").is_string(),
            "operation Kind must be a string");
    const auto & kind = operation.at("Kind").get_ref<const std::string &>();
    if (kind == "public-c-abi") {
        require_keys(operation, {"Kind"});
    } else if (kind == "unicode-encode" || kind == "unicode-decode") {
        auto adapted = operation; adapted.erase("Kind"); adapted["Op"] = kind;
        prism_vocab_session::validate_unicode(adapted);
    } else if (kind == "unicode-regex-split") {
        require_keys(operation, {"Kind", "Text", "Expressions", "ByteEncoding"});
        require(operation.at("Text").is_string() &&
                operation.at("Text").get_ref<const std::string &>().size() <= max_text_bytes,
                "Text must be a string of at most 65536 UTF-8 bytes");
        require(operation.at("ByteEncoding").is_boolean(), "ByteEncoding must be boolean");
        const auto & expressions = operation.at("Expressions");
        require(expressions.is_array() && !expressions.empty() && expressions.size() <= max_expressions,
                "Expressions must contain 1 through 16 strings");
        for (const auto & expression : expressions) {
            require(expression.is_string() && expression.get_ref<const std::string &>().size() <= max_expression_bytes,
                    "expression must be a string of at most 8192 UTF-8 bytes");
        }
    } else if (kind == "mtp-first-decode-fits") {
        require_keys(operation, {"Kind", "Batch", "Catchup", "Sequences"});
        integer(operation.at("Batch"));
        integer(operation.at("Catchup"));
        integer(operation.at("Sequences"));
    } else if (prism_jinja::handles(kind)) {
        prism_jinja::validate(operation);
    } else if (prism_mtmd::handles(kind)) {
        prism_mtmd::validate(operation);
    } else if (prism_grammar_parser::handles(kind)) {
        prism_grammar_parser::validate(operation);
    } else if (prism_reasoning_budget::handles(kind)) {
        prism_reasoning_budget::validate(operation);
    } else if (prism_tensor_graph::handles(kind)) {
        prism_tensor_graph::validate(operation);
    } else if (prism_grammar::handles(kind)) {
        prism_grammar::validate(operation);
    } else if (prism_codec::handles(kind)) {
        prism_codec::validate(operation);
    } else if (prism_peg::handles(kind)) {
        prism_peg::validate(operation);
    } else if (kind == "concurrent-log") {
        prism_concurrency::validate(operation);
    } else if (prism_arena::handles(kind)) {
        prism_arena::validate(operation);
    } else if (prism_batch_allocator::handles(kind)) {
        prism_batch_allocator::validate(operation);
    } else if (prism_allocation::handles(kind)) {
        prism_allocation::validate(operation);
    } else if (kind == "mtmd-chunks-trace") {
        prism_mtmd_chunks::validate(operation);
    } else if (prism_sampling::handles(kind)) {
        prism_sampling::validate(operation);
    } else if (prism_gguf::handles(kind)) {
        prism_gguf::validate(operation);
    } else if (prism_chat_templates::handles(kind)) {
        prism_chat_templates::validate(operation);
    } else if (prism_chat_parser::handles(kind)) {
        prism_chat_parser::validate(operation);
    } else if (prism_chat_auto::handles(kind)) {
        prism_chat_auto::validate(operation);
    } else if (prism_vocab::handles(kind)) {
        prism_vocab::validate(operation);
    } else if (prism_json_schema::handles(kind)) {
        prism_json_schema::validate(operation);
    } else if (prism_chat_analysis::handles(kind)) {
        prism_chat_analysis::validate(operation);
    } else if (prism_arg_parser::handles(kind)) {
        prism_arg_parser::validate(operation);
    } else if (prism_chat::handles(kind)) {
        prism_chat::validate(operation);
    } else if (prism_optimizer::handles(kind)) {
        prism_optimizer::validate(operation);
    } else if (prism_quantization::handles(kind)) {
        prism_quantization::validate(operation);
    } else if (prism_backend_ops::handles(kind)) {
        prism_backend_ops::validate(operation);
    } else if (kind == "cpu-threadpool") {
        prism_threadpool::validate(operation);
    } else if (prism_dspark::handles(kind)) {
        prism_dspark::validate(operation);
    } else {
        throw std::runtime_error("unknown operation Kind");
    }
}

wire_json execute(const wire_json & operation) {
    const auto & kind = operation.at("Kind").get_ref<const std::string &>();
    if (kind == "unicode-encode" || kind == "unicode-decode") {
        auto adapted = operation; adapted.erase("Kind"); adapted["Op"] = kind;
        return prism_vocab_session::execute(adapted, nullptr);
    }
    if (kind == "public-c-abi") {
        const auto value = prism_public_c_abi_observe();
        return wire_json{{"ModelParams", {{"Size", value.model_params_size}, {"Alignment", value.model_params_alignment}}},
                {"ContextParams", {{"Size", value.context_params_size}, {"Alignment", value.context_params_alignment}}},
                {"Batch", {{"Size", value.batch_size}, {"Alignment", value.batch_alignment}}},
                {"TokenSize", value.token_size}, {"PositionSize", value.position_size},
                {"SequenceIdSize", value.sequence_id_size}, {"CStandard", value.c_standard}};
    }
    if (prism_jinja::handles(kind)) {
        return prism_jinja::execute(operation);
    }
    if (prism_mtmd::handles(kind)) {
        return prism_mtmd::execute(operation);
    }
    if (prism_grammar_parser::handles(kind)) {
        return prism_grammar_parser::execute(operation);
    }
    if (prism_reasoning_budget::handles(kind)) {
        return prism_reasoning_budget::execute(operation);
    }
    if (prism_tensor_graph::handles(kind)) {
        return prism_tensor_graph::execute(operation);
    }
    if (prism_grammar::handles(kind)) {
        return prism_grammar::execute(operation);
    }
    if (prism_codec::handles(kind)) {
        return prism_codec::execute(operation);
    }
    if (prism_peg::handles(kind)) {
        return prism_peg::execute(operation);
    }
    if (kind == "concurrent-log") {
        return prism_concurrency::execute(operation);
    }
    if (prism_arena::handles(kind)) {
        return prism_arena::execute(operation);
    }
    if (prism_batch_allocator::handles(kind)) {
        return prism_batch_allocator::execute(operation);
    }
    if (prism_allocation::handles(kind)) {
        return prism_allocation::execute(operation);
    }
    if (kind == "mtmd-chunks-trace") {
        return prism_mtmd_chunks::execute(operation);
    }
    if (prism_sampling::handles(kind)) {
        return prism_sampling::execute(operation);
    }
    if (prism_gguf::handles(kind)) {
        return prism_gguf::execute(operation);
    }
    if (prism_chat_templates::handles(kind)) {
        return prism_chat_templates::execute(operation);
    }
    if (prism_chat_parser::handles(kind)) {
        return prism_chat_parser::execute(operation);
    }
    if (prism_chat_auto::handles(kind)) {
        return prism_chat_auto::execute(operation);
    }
    if (prism_vocab::handles(kind)) {
        return prism_vocab::execute(operation);
    }
    if (prism_json_schema::handles(kind)) {
        return prism_json_schema::execute(operation);
    }
    if (prism_chat_analysis::handles(kind)) {
        return prism_chat_analysis::execute(operation);
    }
    if (prism_arg_parser::handles(kind)) {
        return prism_arg_parser::execute(operation);
    }
    if (prism_chat::handles(kind)) {
        return prism_chat::execute(operation);
    }
    if (prism_optimizer::handles(kind)) {
        return prism_optimizer::execute(operation);
    }
    if (prism_quantization::handles(kind)) {
        return prism_quantization::execute(operation);
    }
    if (prism_backend_ops::handles(kind)) {
        return prism_backend_ops::execute(operation);
    }
    if (kind == "cpu-threadpool") {
        return prism_threadpool::execute(operation);
    }
    if (prism_dspark::handles(kind)) {
        return prism_dspark::execute(operation);
    }
    if (kind == "unicode-regex-split") {
        const auto pieces = unicode_regex_split(operation.at("Text").get<std::string>(),
                operation.at("Expressions").get<std::vector<std::string>>(),
                operation.at("ByteEncoding").get<bool>());
        return wire_json{{"Pieces", pieces}};
    }
    return wire_json{{"Fits", common_speculative_mtp_first_decode_fits(
            integer(operation.at("Batch")), integer(operation.at("Catchup")),
            integer(operation.at("Sequences")))}};
}
} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::strcmp(argv[1], "--backend-session") == 0) {
#ifdef _WIN32
            _setmode(_fileno(stdin), _O_BINARY);
            _setmode(_fileno(stdout), _O_BINARY);
#endif
            return prism_backend_session::serve();
        }
        if (argc == 2 && std::strcmp(argv[1], "--model-session") == 0) {
            return prism_model_runtime::serve();
        }
        require(argc == 5 && std::strcmp(argv[1], "--input") == 0 && std::strcmp(argv[3], "--output") == 0,
                "usage: prism-upstream-common-bridge --input ABSOLUTE_PATH --output ABSOLUTE_PATH");
        const std::filesystem::path input_path(argv[2]), output_path(argv[4]);
        require(input_path.is_absolute() && output_path.is_absolute(), "protocol paths must be absolute");
        require(input_path.lexically_normal() != output_path.lexically_normal(), "input and output paths must differ");
        const auto request = read_request(input_path);
        require_keys(request, {"SchemaVersion", "Operations"});
        require(integer(request.at("SchemaVersion")) == 1, "unsupported SchemaVersion");
        const auto & operations = request.at("Operations");
        require(operations.is_array() && !operations.empty() && operations.size() <= max_operations,
                "Operations must contain 1 through 128 operations");
        for (std::size_t index = 0; index < operations.size(); ++index) {
            const auto & operation = operations[index];
            validate_operation(operation);
            require(operation.at("Kind") != "concurrent-log" || index + 1 == operations.size(),
                    "concurrent-log must be the final operation because it frees the logger singleton");
        }
        wire_json results = wire_json::array();
        for (const auto & operation : operations) {
            results.push_back(execute(operation));
        }
        const auto response = wire_json{{"SchemaVersion", 1}, {"Results", results}}.dump();
        require(response.size() <= max_response_bytes, "response exceeds 16 MiB limit");
        std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
        require(bool(output), "cannot open response file");
        output.exceptions(std::ios::badbit | std::ios::failbit);
        output.write(response.data(), static_cast<std::streamsize>(response.size()));
        output.close();
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "prism-upstream-common-bridge: %s\n", error.what());
        return 1;
    }
}
