#pragma once
// Production vocabulary/Unicode batch primitives for the persistent session.
// Rust owns batches, codepoint/token enumeration, dependencies and comparisons.
#include "llama.h"
#include "common.h"
#include "src/unicode.h"
#include <nlohmann/json.hpp>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace prism_vocab_session {
using json = nlohmann::ordered_json;
inline void require(bool condition, const char * message) { if (!condition) throw std::runtime_error(message); }
inline void fields(const json & object, std::initializer_list<const char *> keys) {
    require(object.is_object() && object.size() == keys.size(), "vocab session fields mismatch");
    for (const auto * key : keys) require(object.contains(key), "missing vocab session field");
}
inline std::uint32_t integer(const json & value, std::uint32_t maximum = UINT32_MAX) {
    require(value.is_number_integer() && (value.is_number_unsigned() || value.get<std::int64_t>() >= 0), "vocab session integer must be nonnegative");
    const auto number = value.get<std::uint64_t>(); require(number <= maximum, "vocab session integer outside bound"); return static_cast<std::uint32_t>(number);
}
inline std::uint8_t nibble(char value) {
    require((value >= '0' && value <= '9') || (value >= 'a' && value <= 'f'), "vocab session hex must be lowercase");
    return static_cast<std::uint8_t>(value <= '9' ? value - '0' : value - 'a' + 10);
}
inline std::string bytes(const json & value) {
    require(value.is_string(), "vocab session hex must be string"); const auto & hex = value.get_ref<const std::string &>();
    require(hex.size() % 2 == 0 && hex.size() <= 2*1024*1024, "vocab session input hex outside bound");
    std::string result(hex.size() / 2, '\0');
    for (std::size_t i = 0; i < result.size(); ++i) result[i] = static_cast<char>((nibble(hex[2*i]) << 4) | nibble(hex[2*i+1])); return result;
}
inline std::string hex(const std::string & bytes) {
    require(bytes.size() <= 1024*1024, "vocab session output bytes exceed bound"); constexpr char digits[] = "0123456789abcdef";
    std::string result(bytes.size()*2, '\0');
    for (std::size_t i = 0; i < bytes.size(); ++i) { const auto byte = static_cast<std::uint8_t>(bytes[i]); result[2*i] = digits[byte >> 4]; result[2*i+1] = digits[byte & 15]; } return result;
}
inline bool handles(const std::string & op) { return op == "vocab-tokenize" || op == "vocab-detokenize" || op == "raw-vocab-detokenize" || op == "unicode-encode" || op == "unicode-decode"; }
inline void validate_unicode(const json & operation) {
    const auto op = operation.at("Op").get<std::string>();
    const bool encode = op == "unicode-encode";
    require(encode || op == "unicode-decode", "unknown Unicode operation");
    if (encode) fields(operation, {"Op", "CodePoints"});
    else fields(operation, {"Op", "InputsHex"});
    const auto & items = operation.at(encode ? "CodePoints" : "InputsHex");
    require(items.is_array() && items.size() <= 512, "Unicode batch exceeds512");
    std::size_t supplied = 0;
    for (const auto & item : items) {
        if (encode) integer(item);
        else { supplied += bytes(item).size(); require(supplied <= 1024*1024, "Unicode input exceeds1MiB"); }
    }
}
inline json execute(const json & operation, llama_context * context) {
    const auto op = operation.at("Op").get<std::string>(); const json * items = nullptr;
    if (op == "vocab-tokenize") {
        fields(operation, {"Op", "Context", "InputsHex", "AddSpecial", "ParseSpecial"});
        require(operation.at("AddSpecial").is_boolean() && operation.at("ParseSpecial").is_boolean(), "vocab tokenizer flags must be boolean"); items = &operation.at("InputsHex");
    } else if (op == "vocab-detokenize") {
        fields(operation, {"Op", "Context", "TokenLists", "Special"}); require(operation.at("Special").is_boolean(), "vocab detokenizer Special must be boolean"); items = &operation.at("TokenLists");
    } else if (op == "raw-vocab-detokenize") { fields(operation,{"Op","Context","TokenLists","RemoveSpecial","UnparseSpecial"});require(operation.at("RemoveSpecial").is_boolean()&&operation.at("UnparseSpecial").is_boolean(),"raw detokenizer flags must be boolean");items=&operation.at("TokenLists");
    } else if (op == "unicode-encode") { fields(operation, {"Op", "CodePoints"}); items = &operation.at("CodePoints"); }
    else { require(op == "unicode-decode", "unknown vocab session operation"); fields(operation, {"Op", "InputsHex"}); items = &operation.at("InputsHex"); }
    require(items->is_array() && items->size() <= 512, "vocab session batch exceeds512");
    const llama_vocab * vocabulary = nullptr;
    if (op == "vocab-tokenize" || op == "vocab-detokenize" || op == "raw-vocab-detokenize") { require(context != nullptr, "vocab session requires shared context"); vocabulary = llama_model_get_vocab(llama_get_model(context)); require(vocabulary != nullptr, "model has no vocabulary"); }
    std::size_t supplied = 0;
    for (const auto & item : *items) {
        if (op == "unicode-encode") integer(item);
        else if (op == "vocab-detokenize" || op == "raw-vocab-detokenize") {
            require(item.is_array() && item.size() <= 65536, "vocab detokenizer token extent exceeds bound"); supplied += item.size()*sizeof(llama_token);
            for (const auto & token : item) { const auto id = integer(token, INT32_MAX); require(id < static_cast<std::uint32_t>(llama_vocab_n_tokens(vocabulary)), "detokenizer token outside vocabulary"); }
        } else supplied += bytes(item).size();
        require(supplied <= 1024*1024, "vocab session supplied bytes exceed1MiB");
    }
    json values = json::array(), errors = json::array(); std::size_t returned = 0;
    for (const auto & item : *items) {
        try {
            json value;
            if (op == "unicode-encode") { const auto output = unicode_cpt_to_utf8(integer(item)); returned += output.size(); value = hex(output); }
            else if (op == "unicode-decode") { const auto output = unicode_cpts_from_utf8(bytes(item)); returned += output.size()*sizeof(std::uint32_t); value = output; }
            else if (op == "vocab-tokenize") {
                const auto output = common_tokenize(context, bytes(item), operation.at("AddSpecial").get<bool>(), operation.at("ParseSpecial").get<bool>()); returned += output.size()*sizeof(llama_token); value = output;
            } else {
                std::vector<llama_token> tokens; for (const auto & token : item) tokens.push_back(static_cast<llama_token>(integer(token, INT32_MAX)));
                std::string output;if(op=="vocab-detokenize")output=common_detokenize(context,tokens,operation.at("Special").get<bool>());else{auto count=llama_detokenize(vocabulary,tokens.data(),static_cast<std::int32_t>(tokens.size()),nullptr,0,operation.at("RemoveSpecial").get<bool>(),operation.at("UnparseSpecial").get<bool>());require(count<=0&&count>=-1048576,"raw detokenizer required extent outside bound");if(count<0){output.resize(static_cast<std::size_t>(-count));count=llama_detokenize(vocabulary,tokens.data(),static_cast<std::int32_t>(tokens.size()),output.data(),static_cast<std::int32_t>(output.size()),operation.at("RemoveSpecial").get<bool>(),operation.at("UnparseSpecial").get<bool>());require(count>=0&&static_cast<std::size_t>(count)<=output.size(),"raw detokenizer second extent differs");output.resize(static_cast<std::size_t>(count));}} returned += output.size(); value = hex(output);
            }
            require(returned <= 4*1024*1024, "vocab session returned payload exceeds4MiB"); values.push_back(std::move(value)); errors.push_back(nullptr);
        } catch (const std::invalid_argument & error) { values.push_back(nullptr); errors.push_back({{"Kind", "invalid-argument"}, {"Message", std::string(error.what()).substr(0,4096)}}); }
        catch (const std::exception & error) { values.push_back(nullptr); errors.push_back({{"Kind", "standard"}, {"Message", std::string(error.what()).substr(0,4096)}}); }
        catch (...) { values.push_back(nullptr); errors.push_back({{"Kind", "unknown"}, {"Message", "non-standard production exception"}}); }
    }
    return json{{"Values", values}, {"Errors", errors}};
}
} // namespace prism_vocab_session
