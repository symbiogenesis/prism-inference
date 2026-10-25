// Serialization and the production grammar parser only; no expected rules.
#pragma once

#include "llama-grammar.h"
#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>

namespace prism_grammar_parser {
using json = nlohmann::ordered_json;

inline void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

inline bool handles(const std::string & kind) {
    return kind == "grammar-parser-parse";
}

inline void validate(const json & operation) {
    require(operation.is_object() && operation.size() == 2 && operation.contains("Kind") &&
            operation.contains("Grammar"), "invalid grammar parser operation fields");
    require(operation.at("Kind").is_string() && handles(operation.at("Kind").get<std::string>()),
            "unknown grammar parser Kind");
    require(operation.at("Grammar").is_string(), "Grammar must be UTF-8 text");
    const auto & grammar = operation.at("Grammar").get_ref<const std::string &>();
    require(!grammar.empty() && grammar.size() <= 65536 && grammar.find('\0') == std::string::npos,
            "Grammar must contain 1 through 65536 bytes without NUL");
}

inline json execute(const json & operation) {
    // A fresh production parser owns each operation's symbols and rules.
    llama_grammar_parser parser;
    const auto grammar = operation.at("Grammar").get<std::string>();
    const bool parsed = parser.parse(grammar.c_str());
    require(parser.symbol_ids.size() <= 4096 && parser.rules.size() <= 4096,
            "grammar parser result exceeds collection bound");
    json symbols = json::array();
    for (const auto & symbol : parser.symbol_ids) {
        symbols.push_back(json{{"Name", symbol.first}, {"Id", symbol.second}});
    }
    json rules = json::array();
    std::size_t total_elements = 0;
    for (const auto & rule : parser.rules) {
        require(rule.size() <= 65536 - total_elements, "grammar parser element count exceeds bound");
        total_elements += rule.size();
        json elements = json::array();
        for (const auto & element : rule) {
            elements.push_back(json{{"Type", static_cast<std::uint32_t>(element.type)}, {"Value", element.value}});
        }
        rules.push_back(std::move(elements));
    }
    return json{{"Parsed", parsed}, {"SymbolIds", symbols}, {"Rules", rules}};
}
} // namespace prism_grammar_parser
