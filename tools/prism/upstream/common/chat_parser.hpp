#pragma once
// Generic production chat PEG transport. Rust supplies graphs, inputs, references and all assertions.
#include "peg.hpp"
#include "chat-peg-parser.h"
#include "chat-auto-parser.h"
#include "chat-auto-parser-helpers.h"
#include "ggml.h"

namespace prism_chat_parser {
using json = nlohmann::ordered_json;
using prism_peg::require; using prism_peg::fields; using prism_peg::integer; using prism_peg::string;
inline void validate_graph(const json &, std::set<std::string>, std::size_t, std::size_t &);
inline void validate_expr(const json & expression, const std::set<std::string> & locals, std::size_t depth, std::size_t & count) {
    require(depth <= 48 && ++count <= 16384 && expression.is_object(), "PEG expression depth/count exceeds bound");
    if (expression.contains("Ref")) { fields(expression, {"Ref"}); string(expression.at("Ref"), 128); require(locals.count(expression.at("Ref").get<std::string>()) != 0, "unknown PEG local Ref"); return; }
    fields(expression, {"Op", "Args"}); string(expression.at("Op"), 128);
    const auto op = expression.at("Op").get<std::string>(); const auto & args = expression.at("Args");
    require(args.is_array() && args.size() <= 64, "PEG Args must be an array of at most 64 values");
    static const std::set<std::string> zero = {"eps","start","end","any","space","rest","double_quoted_string","single_quoted_string","json","python_value","marker","choice-empty"};
    const auto arity = [&](std::size_t low, std::size_t high) { require(args.size() >= low && args.size() <= high, "PEG argument count mismatch"); };
    const auto child = [&](std::size_t index) { validate_expr(args.at(index), locals, depth+1, count); };
    if (op == "standard_json_tools") { arity(5,13); string(args[0]); string(args[1]); fields(args[2],{"Json"}); string(args[2].at("Json"),1024*1024); (void)common_json::parse(args[2].at("Json").get<std::string>()); require(args[3].is_boolean()&&args[4].is_boolean(),"chat tool options must be boolean"); for(std::size_t i=5;i<args.size();++i) { if(i==7||i==8||i==12) require(args[i].is_boolean(),"chat tool option must be boolean"); else if(i==11) {require(args[i].is_array()&&args[i].size()<=128,"chat parameter order exceeds bound");for(const auto & item:args[i])string(item);}else string(args[i]); } }
    else if (op == "standard_constructed_tools") { arity(4,4); for(std::size_t i=0;i<2;++i) {fields(args[i],{"Json"});string(args[i].at("Json"),1024*1024);(void)common_json::parse(args[i].at("Json").get<std::string>());}require(args[2].is_boolean()&&args[3].is_boolean(),"chat tool options must be boolean"); }
    else if(op == "schema") { arity(3,4); child(0); string(args[1]); fields(args[2],{"Json"});string(args[2].at("Json"),1024*1024);(void)common_json::parse(args[2].at("Json").get<std::string>());if(args.size()==4)require(args[3].is_boolean(),"chat schema raw must be boolean"); }
    else if(op == "permute") {arity(2,2);string(args[0],128);require(args[1].is_array()&&args[1].size()<=64,"chat permute list exceeds bound");for(const auto & item:args[1])validate_expr(item,locals,depth+1,count);}
    else if(op == "trigger_rule") {arity(2,2);string(args[0],128);child(1);}
    else if(op == "reasoning_block" || op == "reasoning" || op == "content" || op == "tool" || op == "tool_open" || op == "tool_close" || op == "tool_id" || op == "tool_name" || op == "tool_args" || op == "tool_arg" || op == "tool_arg_open" || op == "tool_arg_close" || op == "tool_arg_name" || op == "tool_arg_string_value" || op == "tool_arg_json_value") {arity(1,1);child(0);}
    else if (zero.count(op)) arity(0,0);
    else if (op == "literal" || op == "ref" || op == "until") { arity(1,1); string(args[0]); }
    else if (op == "chars") { arity(1,3); string(args[0]); for (std::size_t i=1;i<args.size();++i) integer(args[i], i==1?0:-1,65536); }
    else if(op=="or-assign"){arity(2,2);child(0);child(1);}
    else if (op == "one_or_more" || op == "zero_or_more" || op == "optional" || op == "atomic" || op == "peek" || op == "negate") { arity(1,1); child(0); }
    else if (op == "sequence" || op == "choice") { arity(1,1); require(args[0].is_array() && args[0].size() <= 1024, "PEG parser list exceeds bound"); for (const auto & item : args[0]) validate_expr(item,locals,depth+1,count); }
    else if (op == "until_one_of") { arity(1,1); require(args[0].is_array() && args[0].size() <= 1024, "PEG delimiter list exceeds bound"); for (const auto & item : args[0]) string(item); }
    else if (op == "add" || op == "or" || op == "space-add") { arity(2,2); require(!args[0].is_string() || !args[1].is_string(), "PEG operator requires a parser operand"); for (std::size_t i=0;i<2;++i) { if (args[i].is_string()) string(args[i]); else child(i); } }
    else if (op == "rule" || op == "rule-lazy") { arity(2,3); string(args[0],128); if (op == "rule-lazy") validate_graph(args[1],locals,depth+1,count); else child(1); if(args.size()==3) require(args[2].is_boolean(), "PEG trigger must be boolean"); }
    else if (op == "tag" || op == "json_member") { arity(2,2); string(args[0]); child(1); }
    else if (op == "gbnf") { arity(2,2); child(0); string(args[1]); }
    else if (op == "ac") { arity(2,2); child(0); if(args[1].is_string()) string(args[1]); else { require(args[1].is_array() && args[1].size()<=1024,"PEG AC delimiter list exceeds bound"); for(const auto & item:args[1]) string(item); } }
    else if (op == "string_content") { arity(1,1); integer(args[0],0,255); }
    else if (op == "repeat") { arity(2,3); child(0); integer(args[1],0,65536); if(args.size()==3) integer(args[2],-1,65536); }
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
using environment=std::map<std::string,common_peg_parser_id>;
inline common_peg_parser graph(common_chat_peg_builder &,const json &,environment);
inline common_peg_parser expr(common_chat_peg_builder & p,const json & expression,const environment & locals) {
    if(expression.contains("Ref")) return common_peg_parser(locals.at(expression.at("Ref").get<std::string>()),p);
    const auto op=expression.at("Op").get<std::string>(); const auto & a=expression.at("Args");

    if(op=="standard_json_tools") return p.standard_json_tools(a[0].get<std::string>(),a[1].get<std::string>(),common_json::parse(a[2].at("Json").get<std::string>()),a[3].get<bool>(),a[4].get<bool>(),a.size()>5?a[5].get<std::string>():"",a.size()>6?a[6].get<std::string>():"",a.size()>7?a[7].get<bool>():false,a.size()>8?a[8].get<bool>():false,a.size()>9?a[9].get<std::string>():"",a.size()>10?a[10].get<std::string>():"",a.size()>11?a[11].get<std::vector<std::string>>():std::vector<std::string>{},a.size()>12?a[12].get<bool>():false);
    if(op=="standard_constructed_tools") {std::map<std::string,std::string> markers;const auto object=common_json::parse(a[0].at("Json").get<std::string>());for(const auto & entry:object.items())markers.emplace(entry.key(),entry.value().get<std::string>());return p.standard_constructed_tools(markers,common_json::parse(a[1].at("Json").get<std::string>()),a[2].get<bool>(),a[3].get<bool>());}
    if(op=="permute") {std::vector<common_peg_parser> children;for(const auto & item:a[1])children.push_back(expr(p,item,locals));return p.permute(a[0].get<std::string>(),children);}
    if(op=="trigger_rule") {const auto child=expr(p,a[1],locals);return p.trigger_rule(a[0].get<std::string>(),child);}
    if(op=="schema") {const auto child=expr(p,a[0],locals);return p.schema(child,a[1].get<std::string>(),common_json::parse(a[2].at("Json").get<std::string>()),a.size()>3?a[3].get<bool>():false);}
    if(op=="reasoning_block") {const auto child=expr(p,a[0],locals);return p.reasoning_block(child);}
    if(op=="reasoning") {const auto child=expr(p,a[0],locals);return p.reasoning(child);}
    if(op=="content") {const auto child=expr(p,a[0],locals);return p.content(child);}
    if(op=="tool") {const auto child=expr(p,a[0],locals);return p.tool(child);}
    if(op=="tool_open") {const auto child=expr(p,a[0],locals);return p.tool_open(child);}
    if(op=="tool_close") {const auto child=expr(p,a[0],locals);return p.tool_close(child);}
    if(op=="tool_id") {const auto child=expr(p,a[0],locals);return p.tool_id(child);}
    if(op=="tool_name") {const auto child=expr(p,a[0],locals);return p.tool_name(child);}
    if(op=="tool_args") {const auto child=expr(p,a[0],locals);return p.tool_args(child);}
    if(op=="tool_arg") {const auto child=expr(p,a[0],locals);return p.tool_arg(child);}
    if(op=="tool_arg_open") {const auto child=expr(p,a[0],locals);return p.tool_arg_open(child);}
    if(op=="tool_arg_close") {const auto child=expr(p,a[0],locals);return p.tool_arg_close(child);}
    if(op=="tool_arg_name") {const auto child=expr(p,a[0],locals);return p.tool_arg_name(child);}
    if(op=="tool_arg_string_value") {const auto child=expr(p,a[0],locals);return p.tool_arg_string_value(child);}
    if(op=="tool_arg_json_value") {const auto child=expr(p,a[0],locals);return p.tool_arg_json_value(child);}
    if(op=="choice-empty") return p.choice();
    if(op=="or-assign"){auto left=expr(p,a[0],locals);const auto right=expr(p,a[1],locals);left|=right;return left;}
    if(op=="eps") return p.eps(); if(op=="start") return p.start(); if(op=="end") return p.end(); if(op=="any") return p.any(); if(op=="space") return p.space(); if(op=="rest") return p.rest();
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
    if(op=="one_or_more") return p.one_or_more(child); if(op=="zero_or_more") return p.zero_or_more(child); if(op=="optional") return p.optional(child); if(op=="atomic") return p.atomic(child); if(op=="peek") return p.peek(child); if(op=="negate") return p.negate(child);
    if(op=="repeat") return a.size()==2?p.repeat(child,a[1].get<int>()):p.repeat(child,a[1].get<int>(),a[2].get<int>());
    if(op=="gbnf") return p.gbnf(child,a[1].get<std::string>());
    if(op=="ac") return a[1].is_string()?p.ac(child,a[1].get<std::string>()):p.ac(child,a[1].get<std::vector<std::string>>());
    throw std::runtime_error("unknown PEG builder Op");
}
inline common_peg_parser graph(common_chat_peg_builder & p,const json & input,environment locals) {
    for(const auto & statement:input.at("Statements")) { const auto value=expr(p,statement.at("Expr"),locals); if(statement.contains("Let")) locals.insert_or_assign(statement.at("Let").get<std::string>(),value.id()); }
    return expr(p,input.at("Root"),locals);
}

inline bool handles(const std::string & kind) { return kind=="chat-peg-trace"; }
inline void validate(const json & operation) {
    fields(operation,{"Kind","Graph","ResolveRefs","Actions"});
    std::size_t count=0;validate_graph(operation.at("Graph"),{},0,count);
    const auto & refs=operation.at("ResolveRefs");require(refs.is_array()&&refs.size()<=128,"chat reference list exceeds bound");
    for(const auto & ref:refs){string(ref,1024*1024);(void)common_json::parse(ref.get<std::string>());}
    const auto & actions=operation.at("Actions");require(actions.is_array()&&actions.size()<=32768,"chat action list exceeds bound");
    std::set<std::int64_t> input_handles;
    for(std::size_t i=0;i<actions.size();++i) {
        const auto & action=actions[i];require(action.is_object()&&action.contains("Kind"),"missing chat action kind");string(action.at("Kind"),64);const auto kind=action.at("Kind").get<std::string>();
        if(kind=="grammar") {fields(action,{"Kind","Lazy"});require(action.at("Lazy").is_boolean(),"chat grammar lazy must be boolean");}
        else if(kind=="init-input"||kind=="append-input") {fields(action,{"Kind","Input","Bytes"});const auto handle=integer(action.at("Input"),0,31);if(kind=="append-input")require(input_handles.count(handle)!=0,"unknown chat input handle");else input_handles.insert(handle);const auto & bytes=action.at("Bytes");require(bytes.is_array()&&bytes.size()<=1024*1024,"chat input chunk exceeds bound");for(const auto & byte:bytes)integer(byte,0,255);}
        else if(kind=="free-input") {fields(action,{"Kind","Input"});require(input_handles.erase(integer(action.at("Input"),0,31))==1,"unknown chat input handle");}
        else if(kind=="parse") {if(action.contains("Input")){fields(action,{"Kind","Input","Flags","Mapper"});require(input_handles.count(integer(action.at("Input"),0,31))!=0,"unknown chat parse input handle");}else{fields(action,{"Kind","Bytes","Flags","Mapper"});const auto & bytes=action.at("Bytes");require(bytes.is_array()&&bytes.size()<=1024*1024,"chat parse input exceeds bound");for(const auto & byte:bytes)integer(byte,0,255);}integer(action.at("Flags"),0,3);string(action.at("Mapper"),32);const auto mapper=action.at("Mapper").get<std::string>();require(mapper=="chat"||mapper=="chat-unretained"||mapper=="tags"||mapper=="tags-anywhere"||mapper=="none","unknown chat mapper");}
        else if(kind=="diff") {fields(action,{"Kind","Before","After"});for(const auto * key:{"Before","After"}) {const auto & index=action.at(key);if(index.is_null()){require(std::string(key)=="Before","null chat After");continue;}const auto n=integer(index,0,32767);require(std::uint64_t(n)<i&&actions[std::size_t(n)].at("Kind")=="parse"&&actions[std::size_t(n)].at("Mapper")=="chat","chat diff must reference prior chat parses");}}
        else if(kind=="clock") fields(action,{"Kind"});else throw std::runtime_error("unknown chat trace action");
    }
}
inline json tool_call(const common_chat_tool_call & tool) {return json{{"Name",tool.name},{"Arguments",tool.arguments},{"Id",tool.id}};}
inline json message(const common_chat_msg & msg) {json tools=json::array();for(const auto & tool:msg.tool_calls)tools.push_back(tool_call(tool));return json{{"Role",msg.role},{"Content",msg.content},{"ReasoningContent",msg.reasoning_content},{"ToolCalls",tools}};}
inline json parse_result(const common_peg_parse_result & result) {return json{{"Type",static_cast<int>(result.type)},{"Start",result.start},{"End",result.end}};}
inline json execute(const json & operation) {
    common_chat_peg_builder builder;builder.set_root(graph(builder,operation.at("Graph"),{}));const auto arena=builder.build();tagged_peg_parser tagged{arena};
    std::map<std::size_t,common_chat_msg> messages;std::map<int,std::string> inputs;json trace=json::array();ggml_time_init();
    for(const auto & action:operation.at("Actions")) {
        const auto index=trace.size();const auto kind=action.at("Kind").get<std::string>();
        if(kind=="init-input"||kind=="append-input") {auto & input=inputs[action.at("Input").get<int>()];if(kind=="init-input")input=std::string();std::string chunk;for(const auto & byte:action.at("Bytes"))chunk.push_back(static_cast<char>(byte.get<int>()));input+=chunk;require(input.size()<=1024*1024,"assembled chat input exceeds bound");trace.push_back(json::object());continue;}
        if(kind=="free-input") {inputs.erase(action.at("Input").get<int>());trace.push_back(json::object());continue;}
        if(kind=="clock") {trace.push_back(json{{"Microseconds",ggml_time_us()}});continue;}
        if(kind=="grammar") {const auto value=build_grammar([&](const common_grammar_builder & grammar){for(const auto & ref:operation.at("ResolveRefs")){auto schema=common_json::parse(ref.get<std::string>());grammar.resolve_refs(schema);}arena.build_grammar(grammar,action.at("Lazy").get<bool>());});trace.push_back(json{{"GrammarText",value}});continue;}
        if(kind=="diff") {
            json value{{"ExceptionKind",nullptr},{"Message",""},{"Diffs",json::array()}};
            try {common_chat_msg before;if(!action.at("Before").is_null())before=messages.at(action.at("Before").get<std::size_t>());const auto diffs=common_chat_msg_diff::compute_diffs(before,messages.at(action.at("After").get<std::size_t>()));for(const auto & diff:diffs)value["Diffs"].push_back(json{{"ReasoningContentDelta",diff.reasoning_content_delta},{"ContentDelta",diff.content_delta},{"ToolCallIndex",diff.tool_call_index},{"ToolCallDelta",tool_call(diff.tool_call_delta)}});}
            catch(const std::exception & error){value["ExceptionKind"]="standard";value["Message"]=error.what();}catch(...){value["ExceptionKind"]="unknown";value["Message"]="";}
            trace.push_back(std::move(value));continue;
        }
        json value{{"Result",nullptr},{"Message",nullptr},{"Tags",nullptr},{"ExceptionKind",nullptr},{"ExceptionMessage",""}};
        try {
            std::string bytes;if(!action.contains("Input")){for(const auto & byte:action.at("Bytes"))bytes.push_back(static_cast<char>(byte.get<int>()));}const auto & input=action.contains("Input")?inputs.at(action.at("Input").get<int>()):bytes;const auto flags=static_cast<common_peg_parse_flags>(action.at("Flags").get<int>());const auto mapper=action.at("Mapper").get<std::string>();
            if(mapper=="tags"||mapper=="tags-anywhere") {const auto parsed=mapper=="tags"?tagged.parse_and_extract(input,flags):tagged.parse_anywhere_and_extract(input);value["Result"]=parse_result(parsed.result);value["Tags"]=parsed.tags;}
            else {common_peg_parse_context context(input,flags);const auto parsed=arena.parse(context);value["Result"]=parse_result(parsed);if(mapper=="chat"||mapper=="chat-unretained"){common_chat_msg msg;common_chat_peg_mapper(msg).from_ast(context.ast,parsed);value["Message"]=message(msg);if(mapper=="chat")messages.emplace(index,std::move(msg));}}
        }catch(const std::exception & error){value["ExceptionKind"]="standard";value["ExceptionMessage"]=error.what();}catch(...){value["ExceptionKind"]="unknown";}
        trace.push_back(std::move(value));
    }
    return json{{"Trace",trace}};
}
} // namespace prism_chat_parser
