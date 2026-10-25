// Bounded production chat rendering transport. Templates and comparisons live in Rust.
#pragma once
#include "chat.h"
#include "llama.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace prism_chat_templates {
using json = nlohmann::ordered_json;
inline void require(bool condition, const char * message) { if (!condition) throw std::runtime_error(message); }
inline void keys(const json & value, std::initializer_list<const char *> names) {
    require(value.is_object() && value.size() == names.size(), "chat render fields mismatch");
    for (const auto * name : names) require(value.contains(name), "missing chat render field");
}
inline std::size_t integer(const json & value, std::size_t maximum) {
    require(value.is_number_integer() && (value.is_number_unsigned() || value.get<std::int64_t>() >= 0), "chat integer must be nonnegative");
    const auto number = value.get<std::uint64_t>(); require(number <= maximum, "chat integer exceeds bound"); return static_cast<std::size_t>(number);
}
inline void text(const json & value, std::size_t maximum) {
    require(value.is_string(), "chat text must be string"); const auto & bytes = value.get_ref<const std::string &>();
    require(bytes.size() <= maximum && bytes.find('\0') == std::string::npos, "chat text exceeds byte bound or contains NUL");
}
inline bool handles(const std::string & kind) { return kind == "chat-builtins" || kind == "legacy-chat-render" || kind == "common-chat-render"; }
inline void validate(const json & operation) {
    const auto kind = operation.at("Kind").get<std::string>(); require(handles(kind), "unknown chat render kind");
    if (kind == "chat-builtins") { keys(operation, {"Kind", "Capacity"}); integer(operation.at("Capacity"), 4096); return; }
    if (kind == "legacy-chat-render") { keys(operation, {"Kind", "Template", "Messages", "AddGeneration", "Capacity"}); integer(operation.at("Capacity"), 131072); }
    else { keys(operation, {"Kind", "Template", "Messages", "AddGeneration", "Bos", "Eos", "UseJinja"}); text(operation.at("Bos"), 4096); text(operation.at("Eos"), 4096); require(operation.at("UseJinja").is_boolean(), "UseJinja must be boolean"); }
    text(operation.at("Template"), 256*1024); require(operation.at("AddGeneration").is_boolean(), "AddGeneration must be boolean");
    const auto & messages = operation.at("Messages"); require(messages.is_array() && messages.size() <= 1024, "chat message count exceeds bound");
    std::size_t total = 0;
    for (const auto & message : messages) {
        keys(message, {"Role", "Content"}); text(message.at("Role"), 4096); text(message.at("Content"), 65536);
        total += message.at("Role").get_ref<const std::string &>().size() + message.at("Content").get_ref<const std::string &>().size();
        require(total <= 1024*1024, "chat message byte total exceeds bound");
    }
}
inline std::string hex(const std::vector<char> & bytes) {
    constexpr char digits[] = "0123456789abcdef"; std::string output(bytes.size()*2, '\0');
    for (std::size_t i = 0; i < bytes.size(); ++i) { const auto value = static_cast<unsigned char>(bytes[i]); output[2*i] = digits[value>>4]; output[2*i+1] = digits[value&15]; }
    return output;
}
inline json execute(const json & operation) {
    const auto kind = operation.at("Kind").get<std::string>();
    if (kind == "chat-builtins") {
        const auto capacity = integer(operation.at("Capacity"), 4096); std::vector<const char *> names(capacity, nullptr);
        const auto count = llama_chat_builtin_templates(capacity ? names.data() : nullptr, capacity);
        json values = json::array(); for (const auto * name : names) values.push_back(name ? json(name) : json(nullptr));
        return json{{"Count", count}, {"Names", values}};
    }
    if (kind == "legacy-chat-render") {
        std::vector<std::pair<std::string, std::string>> storage;
        for (const auto & message : operation.at("Messages")) storage.emplace_back(message.at("Role").get<std::string>(), message.at("Content").get<std::string>());
        std::vector<llama_chat_message> messages;
        for (const auto & message : storage) messages.push_back({message.first.c_str(), message.second.c_str()});
        std::vector<char> buffer(integer(operation.at("Capacity"), 131072));
        const auto result = llama_chat_apply_template(operation.at("Template").get_ref<const std::string &>().c_str(), messages.data(), messages.size(), operation.at("AddGeneration").get<bool>(), buffer.empty() ? nullptr : buffer.data(), static_cast<std::int32_t>(buffer.size()));
        return json{{"ReturnCode", result}, {"BufferHex", hex(buffer)}};
    }
    try {
        auto templates = common_chat_templates_init(nullptr, operation.at("Template").get<std::string>(), operation.at("Bos").get<std::string>(), operation.at("Eos").get<std::string>());
        common_chat_templates_inputs inputs;
        inputs.use_jinja = operation.at("UseJinja").get<bool>(); inputs.add_generation_prompt = operation.at("AddGeneration").get<bool>();
        for (const auto & message : operation.at("Messages")) {
            common_chat_msg value; value.role = message.at("Role").get<std::string>(); value.content = message.at("Content").get<std::string>(); inputs.messages.push_back(std::move(value));
        }
        return json{{"Success", true}, {"Prompt", common_chat_templates_apply(templates.get(), inputs).prompt}, {"Error", nullptr}};
    } catch (const std::exception & error) { return json{{"Success", false}, {"Prompt", nullptr}, {"Error", error.what()}}; }
}
} // namespace prism_chat_templates
