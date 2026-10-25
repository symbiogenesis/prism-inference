#pragma once

// Generic vocabulary loading and explicit worker schedules on one shared context.
// Corpus parsing, expected tokens, schedules and verdicts remain in safe Rust.
#include "llama.h"
#include "common.h"
#include "ggml.h"
#include <nlohmann/json.hpp>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace prism_vocab {
using json = nlohmann::ordered_json;
inline void require(bool condition, const char * message) { if (!condition) throw std::runtime_error(message); }
inline void fields(const json & value, std::initializer_list<const char *> names) {
    require(value.is_object() && value.size() == names.size(), "vocabulary fields mismatch");
    for (const auto * name : names) require(value.contains(name), "missing vocabulary field");
}
inline std::size_t integer(const json & value, std::size_t maximum) {
    require(value.is_number_integer() && (value.is_number_unsigned() || value.get<std::int64_t>() >= 0), "vocabulary integer must be nonnegative");
    const auto result = value.get<std::uint64_t>(); require(result <= maximum, "vocabulary integer exceeds bound"); return static_cast<std::size_t>(result);
}
inline bool handles(const std::string & kind) { return kind == "vocab-runtime-info" || kind == "vocab-evaluate"; }
inline void validate(const json & operation) {
    if (operation.at("Kind") == "vocab-runtime-info") { fields(operation, {"Kind"}); return; }
    fields(operation, {"Kind", "Path", "CreateContext", "Inputs", "Workers"});
    require(operation.at("Path").is_string(), "vocabulary path must be text"); const auto path = operation.at("Path").get<std::string>();
    require(path.size() <= 32768 && path.find('\0') == std::string::npos && std::filesystem::path(path).is_absolute(), "vocabulary path must be absolute and bounded");
    require(operation.at("CreateContext").is_boolean(), "CreateContext must be boolean");
    const auto & inputs = operation.at("Inputs"), & workers = operation.at("Workers");
    require(inputs.is_array() && inputs.size() <= 1024, "vocabulary input count exceeds bound");
    std::size_t total = 0;
    for (const auto & input : inputs) {
        fields(input, {"Bytes", "AddSpecial", "ParseSpecial"}); const auto & bytes = input.at("Bytes");
        require(bytes.is_array() && bytes.size() <= 65536, "vocabulary input bytes exceed bound"); total += bytes.size(); require(total <= 1024*1024, "vocabulary input byte total exceeds bound");
        for (const auto & byte : bytes) integer(byte, 255);
        require(input.at("AddSpecial").is_boolean() && input.at("ParseSpecial").is_boolean(), "tokenizer flags must be boolean");
    }
    require(workers.is_array() && workers.size() <= 256, "vocabulary worker count exceeds bound"); std::size_t scheduled = 0, captured = 0;
    for (const auto & worker : workers) {
        fields(worker, {"Actions"}); const auto & schedule = worker.at("Actions"); require(schedule.is_array() && schedule.size() <= 1024, "vocabulary worker schedule exceeds bound");
        for (const auto & action : schedule) {
            require(action.is_object() && action.contains("Kind") && action.at("Kind").is_string(), "vocabulary action Kind must be string");
            if (action.at("Kind") == "clock") { fields(action, {"Kind"}); continue; }
            fields(action, {"Kind", "Input", "Capture", "Detokenize"}); require(action.at("Kind") == "tokenize", "unknown vocabulary worker action");
            require(!inputs.empty(), "vocabulary tokenize action requires input"); integer(action.at("Input"), inputs.size() - 1);
            require(action.at("Capture").is_boolean() && action.at("Detokenize").is_boolean(), "vocabulary action flags must be boolean");
            if (action.at("Capture").get<bool>()) require(++captured <= 4096, "vocabulary capture count exceeds bound");
        }
        scheduled += schedule.size(); require(scheduled <= 262144, "vocabulary total schedule exceeds bound");
    }
}
inline std::string hex(const std::string & bytes) {
    require(bytes.size() <= 1024*1024, "vocabulary detokenized bytes exceed bound"); constexpr char digits[] = "0123456789abcdef";
    std::string result(bytes.size() * 2, '\0');
    for (std::size_t i = 0; i < bytes.size(); ++i) { const auto byte = static_cast<std::uint8_t>(bytes[i]); result[2*i] = digits[byte >> 4]; result[2*i+1] = digits[byte & 15]; } return result;
}
struct backend_scope { backend_scope() { llama_backend_init(); } ~backend_scope() { llama_backend_free(); } };
inline json execute(const json & operation) {
    if (operation.at("Kind") == "vocab-runtime-info") return json{{"HardwareConcurrency", std::thread::hardware_concurrency()}};
    const auto path = operation.at("Path").get<std::string>(); require(std::filesystem::is_regular_file(path) && std::filesystem::file_size(path) <= 64*1024*1024, "vocabulary file missing or exceeds 64 MiB");
    backend_scope backend;
    auto model_params = llama_model_default_params(); model_params.vocab_only = true;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(path.c_str(), model_params), llama_model_free);
    std::unique_ptr<llama_context, decltype(&llama_free)> context(nullptr, llama_free);
    const auto create_context = operation.at("CreateContext").get<bool>();
    if (model && create_context) context.reset(llama_init_from_model(model.get(), llama_context_default_params()));
    json response{{"ModelLoaded", bool(model)}, {"ContextLoaded", create_context ? json(bool(context)) : json(nullptr)}, {"Workers", json::array()}, {"Error", nullptr}};
    if (!model || (create_context && !context)) {
        response["Error"] = !model ? "production llama_model_load_from_file returned null" : "production llama_init_from_model returned null";
        return response;
    }
    const auto * vocab = llama_model_get_vocab(model.get()); require(vocab != nullptr, "production model has no vocabulary");
    std::vector<std::string> inputs;
    for (const auto & input : operation.at("Inputs")) {
        std::string bytes; bytes.reserve(input.at("Bytes").size()); for (const auto & byte : input.at("Bytes")) bytes.push_back(static_cast<char>(byte.get<int>())); inputs.push_back(std::move(bytes));
    }
    ggml_time_init();
    const auto & workers = operation.at("Workers"); std::vector<json> observations(workers.size()); std::vector<std::exception_ptr> errors(workers.size()); std::vector<std::thread> threads;
    const auto run = [&](std::size_t worker_index) {
        try {
            const auto & worker = workers[worker_index];
            json results = json::array(); std::size_t completed = 0, captured_tokens = 0;
            for (const auto & action : worker.at("Actions")) {
                if (action.at("Kind") == "clock") { results.push_back({{"Microseconds", ggml_time_us()}}); ++completed; continue; }
                const auto input_index = action.at("Input").get<std::size_t>(); const auto & options = operation.at("Inputs")[input_index];
                const auto tokens = context ? common_tokenize(context.get(), inputs[input_index], options.at("AddSpecial").get<bool>(), options.at("ParseSpecial").get<bool>()) : common_tokenize(vocab, inputs[input_index], options.at("AddSpecial").get<bool>(), options.at("ParseSpecial").get<bool>());
                require(tokens.size() <= 262144, "production tokenizer output exceeds bound");
                json text = nullptr;
                if (action.at("Detokenize").get<bool>()) text = hex(context ? common_detokenize(context.get(), tokens) : common_detokenize(vocab, tokens));
                if (action.at("Capture").get<bool>()) { captured_tokens += tokens.size(); require(captured_tokens <= 1048576, "vocabulary captured tokens exceed bound"); results.push_back({{"Tokens", tokens}, {"TextHex", text}}); }
                else results.push_back(json::object());
                ++completed;
            }
            observations[worker_index] = json{{"Completed", completed}, {"Results", results}};
        } catch (...) { errors[worker_index] = std::current_exception(); }
    };
    try { for (std::size_t i = 0; i < workers.size(); ++i) threads.emplace_back(run, i); }
    catch (...) { for (auto & thread : threads) thread.join(); throw; }
    for (auto & thread : threads) thread.join();
    for (const auto & error : errors) if (error) {
        try { std::rethrow_exception(error); }
        catch (const std::exception & failure) { response["Error"] = failure.what(); }
        catch (...) { response["Error"] = "unknown production vocabulary worker exception"; }
        return response;
    }
    for (auto & observation : observations) response["Workers"].push_back(std::move(observation));
    return response;
}
} // namespace prism_vocab
