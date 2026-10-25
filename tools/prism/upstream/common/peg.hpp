#pragma once
// Declarative transport for production PEG builder, parse and serialization APIs.
#include "peg-parser.h"
#include "json-schema-to-grammar.h"
#include <nlohmann/json.hpp>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace prism_peg {
using json = nlohmann::ordered_json;
inline void require(bool value, const char * message) { if (!value) throw std::runtime_error(message); }
inline void fields(const json & object, std::initializer_list<const char *> keys) {
    require(object.is_object() && object.size() == keys.size(), "PEG fields mismatch");
    for (const auto * key : keys) require(object.contains(key), "missing or unknown PEG field");
}
inline std::int64_t integer(const json & value, std::int64_t low, std::int64_t high) {
    require(value.is_number_integer(), "PEG argument must be an integer");
    if (value.is_number_unsigned()) require(value.get<std::uint64_t>() <= std::uint64_t(high), "PEG integer exceeds bound");
    const auto result = value.get<std::int64_t>(); require(result >= low && result <= high, "PEG integer outside bound"); return result;
}
inline void string(const json & value, std::size_t maximum = 65536) {
    require(value.is_string() && value.get_ref<const std::string &>().size() <= maximum, "PEG string exceeds bound");
}
inline bool handles(const std::string & kind) { return kind == "peg-evaluate" || kind == "peg-serialize" || kind == "peg-deserialize"; }
inline void validate_graph(const json &, std::set<std::string>, std::size_t, std::size_t &);
inline void validate_expr(const json & expression, const std::set<std::string> & locals, std::size_t depth, std::size_t & count) {
    require(depth <= 48 && ++count <= 16384 && expression.is_object(), "PEG expression depth/count exceeds bound");
    if (expression.contains("Ref")) { fields(expression, {"Ref"}); string(expression.at("Ref"), 128); require(locals.count(expression.at("Ref").get<std::string>()) != 0, "unknown PEG local Ref"); return; }
    fields(expression, {"Op", "Args"}); string(expression.at("Op"), 128);
    const auto op = expression.at("Op").get<std::string>(); const auto & args = expression.at("Args");
    require(args.is_array() && args.size() <= 64, "PEG Args must be an array of at most 64 values");
    static const std::set<std::string> zero = {"eps","end","any","space","double_quoted_string","single_quoted_string","json","python_value","marker"};
    const auto arity = [&](std::size_t low, std::size_t high) { require(args.size() >= low && args.size() <= high, "PEG argument count mismatch"); };
    const auto child = [&](std::size_t index) { validate_expr(args.at(index), locals, depth+1, count); };
    if (zero.count(op)) arity(0,0);
    else if (op == "literal" || op == "ref" || op == "until") { arity(1,1); string(args[0]); }
    else if (op == "chars") { arity(1,3); string(args[0]); for (std::size_t i=1;i<args.size();++i) integer(args[i], i==1?0:-1,65536); }
    else if (op == "one_or_more" || op == "zero_or_more" || op == "optional" || op == "atomic") { arity(1,1); child(0); }
    else if (op == "sequence" || op == "choice") { arity(1,1); require(args[0].is_array() && args[0].size() <= 1024, "PEG parser list exceeds bound"); for (const auto & item : args[0]) validate_expr(item,locals,depth+1,count); }
    else if (op == "until_one_of") { arity(1,1); require(args[0].is_array() && args[0].size() <= 1024, "PEG delimiter list exceeds bound"); for (const auto & item : args[0]) string(item); }
    else if (op == "add" || op == "or" || op == "space-add") { arity(2,2); require(!args[0].is_string() || !args[1].is_string(), "PEG operator requires a parser operand"); for (std::size_t i=0;i<2;++i) { if (args[i].is_string()) string(args[i]); else child(i); } }
    else if (op == "rule" || op == "rule-lazy") { arity(2,3); string(args[0],128); if (op == "rule-lazy") validate_graph(args[1],locals,depth+1,count); else child(1); if(args.size()==3) require(args[2].is_boolean(), "PEG trigger must be boolean"); }
    else if (op == "tag" || op == "json_member") { arity(2,2); string(args[0]); child(1); }
    else if (op == "gbnf") { arity(2,2); child(0); string(args[1]); }
    else if (op == "ac") { arity(2,2); child(0); if(args[1].is_string()) string(args[1]); else { require(args[1].is_array() && args[1].size()<=1024,"PEG AC delimiter list exceeds bound"); for(const auto & item:args[1]) string(item); } }
    else if (op == "string_content") { arity(1,1); integer(args[0],0,255); }

    else throw std::runtime_error("unknown PEG builder Op");
}
inline void validate_graph(const json & graph, std::set<std::string> locals, std::size_t depth, std::size_t & count) {
    require(depth<=48,"PEG graph depth exceeds bound"); fields(graph,{"Statements","Root"});
    const auto & statements=graph.at("Statements"); require(statements.is_array() && statements.size()<=4096,"PEG statements exceed bound");
    for(const auto & statement:statements) {
        if(statement.contains("Let")) { fields(statement,{"Let","Expr"}); string(statement.at("Let"),128); }
        else fields(statement,{"Expr"});
        validate_expr(statement.at("Expr"),locals,depth+1,count);
        if(statement.contains("Let")) locals.insert(statement.at("Let").get<std::string>());
    }
    validate_expr(graph.at("Root"),locals,depth+1,count);
}
inline void validate(const json & operation) {
    const auto kind=operation.at("Kind").get<std::string>();
    if(kind=="peg-deserialize") { fields(operation,{"Kind","SerializedJson"}); string(operation.at("SerializedJson"),8*1024*1024); (void)common_json::parse(operation.at("SerializedJson").get<std::string>()); return; }
    if(kind=="peg-serialize") fields(operation,{"Kind","Graph"});
    else fields(operation,{"Kind","Graph","Inputs","GenerateGrammar","RoundTrip"});
    std::size_t count=0; validate_graph(operation.at("Graph"),{},0,count);
    if(kind=="peg-serialize") return;
    const auto & inputs=operation.at("Inputs"); require(inputs.is_array() && inputs.size()<=1024,"PEG inputs exceed bound");
    for(const auto & input:inputs) { fields(input,{"Bytes","Flags"}); const auto & bytes=input.at("Bytes"); require(bytes.is_array() && bytes.size()<=65536,"PEG input bytes exceed bound"); for(const auto & byte:bytes) integer(byte,0,255); integer(input.at("Flags"),0,3); }
    const auto & modes=operation.at("GenerateGrammar"); require(modes.is_array() && modes.size()<=8,"PEG grammar modes exceed bound"); for(const auto & mode:modes) require(mode.is_boolean(),"PEG grammar mode must be boolean");
    require(operation.at("RoundTrip").is_boolean(),"PEG RoundTrip must be boolean");
}
using environment=std::map<std::string,common_peg_parser_id>;
inline common_peg_parser graph(common_peg_parser_builder &,const json &,environment);
inline common_peg_parser expr(common_peg_parser_builder & p,const json & expression,const environment & locals) {
    if(expression.contains("Ref")) return common_peg_parser(locals.at(expression.at("Ref").get<std::string>()),p);
    const auto op=expression.at("Op").get<std::string>(); const auto & a=expression.at("Args");
    if(op=="eps") return p.eps(); if(op=="end") return p.end(); if(op=="any") return p.any(); if(op=="space") return p.space();
    if(op=="double_quoted_string") return p.double_quoted_string(); if(op=="single_quoted_string") return p.single_quoted_string();
    if(op=="json") return p.json();
    if(op=="python_value") return p.python_value(); if(op=="marker") return p.marker();
    if(op=="literal") return p.literal(a[0].get<std::string>()); if(op=="ref") return p.ref(a[0].get<std::string>()); if(op=="until") return p.until(a[0].get<std::string>());
    if(op=="chars") return p.chars(a[0].get<std::string>(),a.size()>1?a[1].get<int>():1,a.size()>2?a[2].get<int>():-1);
    if(op=="string_content") return p.string_content(static_cast<char>(a[0].get<int>()));
    if(op=="until_one_of") return p.until_one_of(a[0].get<std::vector<std::string>>());
    if(op=="sequence" || op=="choice") { std::vector<common_peg_parser> children; for(const auto & item:a[0]) children.push_back(expr(p,item,locals)); return op=="sequence"?p.sequence(children):p.choice(children); }
    if(op=="rule-lazy") { const auto name=a[0].get<std::string>(); const bool trigger=a.size()>2?a[2].get<bool>():false; return p.rule(name,[&p,&a,locals](){return graph(p,a[1],locals);},trigger); }
    if(op=="add" || op=="or" || op=="space-add") {
        if(a[0].is_string()) { const auto left=a[0].get<std::string>(); const auto right=expr(p,a[1],locals); if(op=="add") return left+right; if(op=="or") return left|right; return left<<right; }
        const auto left=expr(p,a[0],locals);
        if(a[1].is_string()) { const auto right=a[1].get<std::string>(); if(op=="add") return left+right; if(op=="or") return left|right; return left<<right; }
        const auto right=expr(p,a[1],locals); if(op=="add") return left+right; if(op=="or") return left|right; return left<<right;
    }
    if(op=="tag" || op=="json_member" || op=="rule") { const auto name=a[0].get<std::string>(); const auto child=expr(p,a[1],locals); if(op=="tag") return p.tag(name,child); if(op=="json_member") return p.json_member(name,child); return p.rule(name,child,a.size()>2?a[2].get<bool>():false); }
    const auto child=expr(p,a[0],locals);
    if(op=="one_or_more") return p.one_or_more(child); if(op=="zero_or_more") return p.zero_or_more(child); if(op=="optional") return p.optional(child); if(op=="atomic") return p.atomic(child);

    if(op=="gbnf") return p.gbnf(child,a[1].get<std::string>());
    if(op=="ac") return a[1].is_string()?p.ac(child,a[1].get<std::string>()):p.ac(child,a[1].get<std::vector<std::string>>());
    throw std::runtime_error("unknown PEG builder Op");
}
inline common_peg_parser graph(common_peg_parser_builder & p,const json & input,environment locals) {
    for(const auto & statement:input.at("Statements")) { const auto value=expr(p,statement.at("Expr"),locals); if(statement.contains("Let")) locals.insert_or_assign(statement.at("Let").get<std::string>(),value.id()); }
    return expr(p,input.at("Root"),locals);
}
inline json parse(const common_peg_arena & arena,const json & inputs) {
    json results=json::array();
    for(const auto & input:inputs) { std::string bytes; for(const auto & byte:input.at("Bytes")) bytes.push_back(static_cast<char>(byte.get<int>())); common_peg_parse_context context(bytes,static_cast<common_peg_parse_flags>(input.at("Flags").get<int>())); const auto result=arena.parse(context); results.push_back(json{{"Type",static_cast<int>(result.type)},{"Start",result.start},{"End",result.end}}); }
    return results;
}
inline json execute(const json & operation) {
    const auto kind=operation.at("Kind").get<std::string>();
    if(kind=="peg-deserialize") { const auto arena=common_peg_arena::from_json(common_json::parse(operation.at("SerializedJson").get<std::string>())); return json{{"Root",arena.root()}}; }
    common_peg_parser_builder builder; const auto root=graph(builder,operation.at("Graph"),{}); builder.set_root(root); const auto arena=builder.build();
    if(kind=="peg-serialize") return json{{"SerializedJson",arena.to_json().dump()}};
    const auto results=parse(arena,operation.at("Inputs")); json grammars=json::array();
    for(const auto & mode:operation.at("GenerateGrammar")) { const bool lazy=mode.get<bool>(); grammars.push_back(build_grammar([&](const common_grammar_builder & grammar){arena.build_grammar(grammar,lazy);})); }
    json serialized=nullptr,roundtrip=json::array();
    if(operation.at("RoundTrip").get<bool>()) { const auto value=arena.to_json().dump(); serialized=value; const auto restored=common_peg_arena::from_json(common_json::parse(value)); roundtrip=parse(restored,operation.at("Inputs")); }
    return json{{"Results",results},{"GrammarTexts",grammars},{"SerializedJson",serialized},{"RoundTripResults",roundtrip}};
}
} // namespace prism_peg
