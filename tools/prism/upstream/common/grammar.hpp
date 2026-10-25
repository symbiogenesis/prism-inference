#pragma once
// Generic production grammar lifecycle and pointer/value serialization only.
#include "llama-grammar.h"
#include "json-schema-to-grammar.h"
#include <nlohmann/json.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace prism_grammar {
using json = nlohmann::ordered_json;
inline void require(bool value, const char * message) { if (!value) throw std::runtime_error(message); }
inline void fields(const json & value, std::initializer_list<const char *> keys) {
    require(value.is_object() && value.size() == keys.size(), "grammar fields mismatch");
    for (const auto * key : keys) require(value.contains(key), "missing or unknown grammar field");
}
inline std::int64_t integer(const json & value, std::int64_t low, std::int64_t high) {
    require(value.is_number_integer(), "grammar value must be an integer");
    if (value.is_number_unsigned()) require(value.get<std::uint64_t>() <= std::uint64_t(high), "grammar integer exceeds bound");
    const auto result = value.get<std::int64_t>();
    require(result >= low && result <= high, "grammar integer outside bound");
    return result;
}
inline std::uint32_t u32(const json & value) { return static_cast<std::uint32_t>(integer(value, 0, 0xffffffffll)); }
inline void text(const json & value, std::size_t maximum, bool c_string = false) {
    require(value.is_string(), "grammar text must be a string");
    const auto & string = value.get_ref<const std::string &>();
    require(string.size() <= maximum && (!c_string || string.find('\0') == std::string::npos), "grammar text exceeds bound or contains NUL");
}
inline bool handles(const std::string & kind) { return kind == "grammar-trace" || kind == "json-schema-to-grammar"; }
inline void validate(const json & operation) {
    if (operation.at("Kind") == "json-schema-to-grammar") {
        fields(operation, {"Kind", "SchemaJson", "ForceGbnf"});
        text(operation.at("SchemaJson"), 262144);
        require(operation.at("ForceGbnf").is_boolean(), "ForceGbnf must be boolean");
        (void) common_json::parse(operation.at("SchemaJson").get<std::string>());
        return;
    }
    fields(operation, {"Kind", "Actions"});
    const auto & actions = operation.at("Actions");
    require(actions.is_array() && !actions.empty() && actions.size() <= 16384, "grammar trace requires 1..16384 actions");
    std::set<int> live;
    std::map<int, int> saved;
    for (const auto & action : actions) {
        require(action.is_object() && action.contains("Kind") && action.at("Kind").is_string() && action.contains("Grammar"), "invalid grammar action");
        const auto kind = action.at("Kind").get<std::string>();
        const auto handle = static_cast<int>(integer(action.at("Grammar"), 0, 31));
        if (kind == "init-text" || kind == "init-rules") {
            require(live.count(handle) == 0, "grammar handle already exists");
            if (kind == "init-text") {
                fields(action, {"Kind", "Grammar", "GrammarText", "Root"});
                text(action.at("GrammarText"), 262144, true);
                text(action.at("Root"), 4096, true);
                require(!action.at("Root").get_ref<const std::string &>().empty(), "grammar Root must not be empty");
            } else {
                fields(action, {"Kind", "Grammar", "Rules", "StartRule"});
                const auto & rules = action.at("Rules");
                require(rules.is_array() && !rules.empty() && rules.size() <= 4096, "grammar rules exceed bound");
                integer(action.at("StartRule"), 0, static_cast<std::int64_t>(rules.size()-1));
                std::size_t count = 0;
                for (const auto & rule : rules) {
                    require(rule.is_array() && !rule.empty(), "grammar rule must not be empty");
                    count += rule.size(); require(count <= 65536, "grammar rule elements exceed bound");
                    for (const auto & element : rule) {
                        fields(element, {"Type", "Value"});
                        const auto type = integer(element.at("Type"), 0, 9);
                        const auto value = u32(element.at("Value"));
                        require(type != LLAMA_GRETYPE_RULE_REF || value < rules.size(), "grammar rule reference exceeds extent");
                    }
                    require(rule.back().at("Type") == LLAMA_GRETYPE_END, "grammar rule must end with END");
                }
            }
            live.insert(handle); continue;
        }
        require(live.count(handle) != 0, "unknown or freed grammar handle");
        if (kind == "clone") {
            fields(action, {"Kind", "Grammar", "Target"});
            const auto target = static_cast<int>(integer(action.at("Target"), 0, 31));
            require(live.count(target) == 0, "grammar clone target exists"); live.insert(target);
        } else if (kind == "accept-codepoint") {
            fields(action, {"Kind", "Grammar", "CodePoint"}); u32(action.at("CodePoint"));
        } else if (kind == "accept-token") {
            require(action.contains("Piece") != action.contains("PieceBytes"), "accept requires Piece or PieceBytes");
            if (action.contains("Piece")) { fields(action, {"Kind", "Grammar", "Token", "Piece"}); text(action.at("Piece"), 65536); }
            else {
                fields(action, {"Kind", "Grammar", "Token", "PieceBytes"});
                const auto & bytes = action.at("PieceBytes"); require(bytes.is_array() && bytes.size() <= 65536, "piece bytes exceed bound");
                for (const auto & byte : bytes) integer(byte, 0, 255);
            }
            integer(action.at("Token"), std::numeric_limits<std::int32_t>::min(), std::numeric_limits<std::int32_t>::max());
        } else if (kind == "save-stacks" || kind == "restore-stacks") {
            fields(action, {"Kind", "Grammar", "StackHandle"});
            const auto stack = static_cast<int>(integer(action.at("StackHandle"), 0, 63));
            if (kind == "save-stacks") saved[stack] = handle;
            else require(saved.count(stack) != 0 && saved.at(stack) == handle, "saved stacks belong to a different grammar lifetime");
        } else if (kind == "reject-candidates") {
            fields(action, {"Kind", "Grammar", "Stack", "Candidates"}); integer(action.at("Stack"), 0, 8191);
            const auto & candidates = action.at("Candidates");
            require(candidates.is_array() && candidates.size() <= 4096, "grammar candidates exceed bound");
            for (const auto & candidate : candidates) {
                fields(candidate, {"Index", "CodePoints", "PartialUtf8", "Id"}); u32(candidate.at("Index"));
                integer(candidate.at("Id"), std::numeric_limits<std::int32_t>::min(), std::numeric_limits<std::int32_t>::max());
                fields(candidate.at("PartialUtf8"), {"Value", "NRemain"}); u32(candidate.at("PartialUtf8").at("Value")); integer(candidate.at("PartialUtf8").at("NRemain"), -1, 3);
                const auto & points = candidate.at("CodePoints");
                require(points.is_array() && !points.empty() && points.size() <= 1024, "candidate code points exceed bound");
                for (const auto & point : points) u32(point);
                require(points.back() == 0, "candidate code points need zero terminator");
            }
        } else {
            fields(action, {"Kind", "Grammar"});
            require(kind == "free" || kind == "state" || kind == "snapshot", "unknown grammar action Kind");
            if (kind == "free") {
                live.erase(handle);
                for (auto it = saved.begin(); it != saved.end();) { if (it->second == handle) it = saved.erase(it); else ++it; }
            }
        }
    }
}
inline json stack_sizes(llama_grammar * grammar) {
    const auto & stacks = llama_grammar_get_stacks(grammar);
    require(stacks.size() <= 8192, "grammar stacks exceed bound");
    json sizes = json::array();
    for (const auto & stack : stacks) { require(stack.size() <= 65536, "grammar stack size exceeds bound"); sizes.push_back(stack.size()); }
    return sizes;
}
inline json snapshot(llama_grammar * grammar) {
    const auto & rules = llama_grammar_get_rules(grammar);
    require(rules.size() <= 4096, "grammar rule count exceeds bound");
    json encoded = json::array(), addresses = json::array(), stacks = json::array();
    std::size_t count = 0;
    for (const auto & rule : rules) {
        count += rule.size(); require(count <= 65536, "grammar rule elements exceed bound");
        json elements = json::array();
        for (const auto & element : rule) elements.push_back(json{{"Type", static_cast<std::uint32_t>(element.type)}, {"Value", element.value}});
        encoded.push_back(std::move(elements)); addresses.push_back(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(rule.data())));
    }
    (void) stack_sizes(grammar);
    count = 0;
    for (const auto & stack : llama_grammar_get_stacks(grammar)) {
        count += stack.size(); require(count <= 65536, "grammar snapshot stack pointers exceed bound");
        json pointers = json::array();
        for (const auto * element : stack) pointers.push_back(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(element)));
        stacks.push_back(std::move(pointers));
    }
    return json{{"Rules", encoded}, {"RuleAddresses", addresses}, {"ElementSize", sizeof(llama_grammar_element)}, {"Stacks", stacks}};
}
struct deleter { void operator()(llama_grammar * grammar) const { llama_grammar_free_impl(grammar); } };
struct saved_stacks { llama_grammar * owner; llama_grammar_stacks stacks; };
inline json execute(const json & operation) {
    if (operation.at("Kind") == "json-schema-to-grammar") return json{{"GrammarText", json_schema_to_grammar(common_json::parse(operation.at("SchemaJson").get<std::string>()), operation.at("ForceGbnf").get<bool>())}};
    std::array<std::unique_ptr<llama_grammar, deleter>, 32> grammars;
    std::map<int, saved_stacks> saved;
    json trace = json::array();
    for (const auto & action : operation.at("Actions")) {
        const auto kind = action.at("Kind").get<std::string>();
        const auto handle = static_cast<std::size_t>(integer(action.at("Grammar"), 0, 31));
        auto * grammar = grammars[handle].get(); json event = json::object();
        if (kind == "init-text") {
            const auto source = action.at("GrammarText").get<std::string>(), root = action.at("Root").get<std::string>();
            grammars[handle].reset(llama_grammar_init_impl(nullptr, source.c_str(), root.c_str(), false, nullptr, 0, nullptr, 0));
            event = json{{"Created", bool(grammars[handle])}};
        } else if (kind == "init-rules") {
            llama_grammar_rules rules;
            for (const auto & input : action.at("Rules")) {
                llama_grammar_rule rule;
                for (const auto & element : input) rule.push_back({static_cast<llama_gretype>(u32(element.at("Type"))), u32(element.at("Value"))});
                rules.push_back(std::move(rule));
            }
            std::vector<const llama_grammar_element *> pointers;
            for (const auto & rule : rules) pointers.push_back(rule.data());
            grammars[handle].reset(llama_grammar_init_impl(nullptr, pointers.data(), pointers.size(), u32(action.at("StartRule"))));
            event = json{{"Created", bool(grammars[handle])}};
        } else if (kind == "free") {
            for (auto it = saved.begin(); it != saved.end();) { if (it->second.owner == grammar) it = saved.erase(it); else ++it; }
            grammars[handle].reset();
        } else {
            require(grammar != nullptr, "grammar action requires successful initialization");
            if (kind == "clone") {
                const auto target = static_cast<std::size_t>(integer(action.at("Target"), 0, 31));
                grammars[target].reset(llama_grammar_clone_impl(*grammar)); event = json{{"Created", bool(grammars[target])}};
            } else if (kind == "state") event = json{{"StackSizes", stack_sizes(grammar)}};
            else if (kind == "snapshot") event = snapshot(grammar);
            else if (kind == "save-stacks") saved[static_cast<int>(integer(action.at("StackHandle"), 0, 63))] = {grammar, llama_grammar_get_stacks(grammar)};
            else if (kind == "restore-stacks") {
                const auto & entry = saved.at(static_cast<int>(integer(action.at("StackHandle"), 0, 63)));
                require(entry.owner == grammar, "saved stacks lifetime mismatch"); llama_grammar_get_stacks(grammar) = entry.stacks;
            } else if (kind == "accept-codepoint") {
                json exception = nullptr; std::string message;
                try { llama_grammar_accept(grammar, u32(action.at("CodePoint"))); }
                catch (const std::runtime_error & error) { exception = "runtime"; message = error.what(); }
                catch (const std::exception & error) { exception = "standard"; message = error.what(); }
                catch (...) { exception = "unknown"; message = "non-standard production exception"; }
                event = json{{"ExceptionKind", exception}, {"Message", message}, {"StackSizes", stack_sizes(grammar)}};
            } else if (kind == "accept-token") {
                std::string piece;
                if (action.contains("Piece")) piece = action.at("Piece").get<std::string>();
                else for (const auto & byte : action.at("PieceBytes")) piece.push_back(static_cast<char>(integer(byte, 0, 255)));
                json exception = nullptr; std::string message;
                try { llama_grammar_accept_token(*grammar, static_cast<llama_token>(integer(action.at("Token"), std::numeric_limits<std::int32_t>::min(), std::numeric_limits<std::int32_t>::max())), piece); }
                catch (const std::runtime_error & error) { exception = "runtime"; message = error.what(); }
                catch (const std::exception & error) { exception = "standard"; message = error.what(); }
                catch (...) { exception = "unknown"; message = "non-standard exception"; }
                event = json{{"ExceptionKind", exception}, {"Message", message}, {"StackSizes", stack_sizes(grammar)}};
            } else if (kind == "reject-candidates") {
                const auto & stacks = llama_grammar_get_stacks(grammar);
                const auto stack = static_cast<std::size_t>(integer(action.at("Stack"), 0, 8191));
                require(stack < stacks.size(), "grammar stack index exceeds extent");
                std::vector<std::vector<std::uint32_t>> buffers;
                for (const auto & candidate : action.at("Candidates")) buffers.push_back(candidate.at("CodePoints").get<std::vector<std::uint32_t>>());
                llama_grammar_candidates candidates;
                std::size_t index = 0;
                for (const auto & candidate : action.at("Candidates")) {
                    candidates.push_back({u32(candidate.at("Index")), buffers[index++].data(),
                        {u32(candidate.at("PartialUtf8").at("Value")), static_cast<int>(integer(candidate.at("PartialUtf8").at("NRemain"), -1, 3))},
                        static_cast<llama_token>(integer(candidate.at("Id"), std::numeric_limits<std::int32_t>::min(), std::numeric_limits<std::int32_t>::max()))});
                }
                const auto rejected = llama_grammar_reject_candidates_for_stack(llama_grammar_get_rules(grammar), stacks[stack], candidates);
                json values = json::array();
                for (const auto & candidate : rejected) {
                    const auto pointer = reinterpret_cast<std::uintptr_t>(candidate.code_points);
                    json points; bool found = false;
                    for (const auto & buffer : buffers) {
                        const auto begin = reinterpret_cast<std::uintptr_t>(buffer.data()), end = begin + buffer.size() * sizeof(std::uint32_t);
                        if (pointer >= begin && pointer < end && (pointer-begin) % sizeof(std::uint32_t) == 0) {
                            points = json(std::vector<std::uint32_t>(buffer.begin() + (pointer-begin)/sizeof(std::uint32_t), buffer.end())); found = true; break;
                        }
                    }
                    require(found, "rejected candidate pointer exceeds input buffers");
                    values.push_back(json{{"Index", candidate.index}, {"CodePoints", points}, {"PartialUtf8", json{{"Value", candidate.partial_utf8.value}, {"NRemain", candidate.partial_utf8.n_remain}}}, {"Id", candidate.id}});
                }
                event = json{{"Rejected", values}};
            }
        }
        trace.push_back(std::move(event));
    }
    return json{{"Trace", trace}};
}
} // namespace prism_grammar
