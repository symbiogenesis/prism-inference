//! Generic tool-call parser construction from ordered function schemas.
use super::Arena;
use crate::{Result, chat::json::Ordered, invalid};
use serde_json::{Value, json};
fn op(name: &str, args: Vec<Value>) -> Value {
    Value::Object(serde_json::Map::from_iter([
        ("Op".into(), Value::String(name.into())),
        ("Args".into(), Value::Array(args)),
    ]))
}
fn lit(text: &str) -> Value {
    op("literal", vec![json!(text)])
}
fn seq(values: Vec<Value>) -> Value {
    op("sequence", vec![Value::Array(values)])
}
fn choice(values: Vec<Value>) -> Value {
    op("choice", vec![Value::Array(values)])
}
fn wrap(name: &str, child: Value) -> Value {
    op(name, vec![child])
}
fn space() -> Value {
    op("space", Vec::new())
}
fn field(name: &str, child: Value) -> Value {
    seq(vec![
        lit(&format!("\"{name}\"")),
        space(),
        lit(":"),
        space(),
        child,
    ])
}
fn id_value(number: bool) -> Value {
    let quoted = seq(vec![
        lit("\""),
        wrap("tool_id", op("string_content", vec![json!(34)])),
        lit("\""),
    ]);
    if number {
        choice(vec![
            quoted,
            wrap("tool_id", op("ref", vec![json!("json-number")])),
        ])
    } else {
        quoted
    }
}
fn tagged_schema(name: &str, params: &Ordered) -> Result<Value> {
    Ok(wrap(
        "tool_args",
        op(
            "schema",
            vec![
                op("json", Vec::new()),
                json!(format!("tool-{name}-schema")),
                json!({"Json":params.dump()?}),
            ],
        ),
    ))
}
fn tool_name(name: &str) -> Value {
    wrap(
        "atomic",
        seq(vec![lit("\""), wrap("tool_name", lit(name)), lit("\"")]),
    )
}
fn rule(name: &str, child: Value) -> Value {
    op(
        "rule",
        vec![json!(format!("tool-{name}")), wrap("tool", child)],
    )
}
fn optional(child: Value) -> Value {
    wrap("optional", child)
}
fn text_arg(args: &[Value], index: usize, default: &str) -> Result<String> {
    args.get(index).map_or_else(
        || Ok(default.into()),
        |value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| invalid("chat tool string argument invalid"))
        },
    )
}
fn flag(args: &[Value], index: usize, default: bool) -> Result<bool> {
    args.get(index).map_or(Ok(default), |value| {
        value
            .as_bool()
            .ok_or_else(|| invalid("chat tool flag invalid"))
    })
}
fn parse_tools(args: &[Value], index: usize) -> Result<Ordered> {
    let text = args
        .get(index)
        .and_then(|value| value.get("Json"))
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("chat tools JSON missing"))?;
    Ordered::parse(text)
}
impl Arena {
    pub(super) fn tools(&mut self, operation: &str, args: &[Value]) -> Result<usize> {
        let expression = if operation == "standard_json_tools" {
            self.json_tools(args)?
        } else {
            constructed_tools(args)?
        };
        self.expression(&expression, &std::collections::HashMap::new(), 0)
    }
    #[allow(
        clippy::too_many_lines,
        reason = "Keep source flat, nested and function-as-key tool layout construction together"
    )]
    fn json_tools(&mut self, args: &[Value]) -> Result<Value> {
        let start = text_arg(args, 0, "")?;
        let end = text_arg(args, 1, "")?;
        let tools = parse_tools(args, 2)?;
        if tools.array().is_empty() {
            return Ok(op("eps", Vec::new()));
        }
        let parallel = flag(args, 3, false)?;
        let force = flag(args, 4, false)?;
        let name_key = text_arg(args, 5, "")?;
        let args_key = text_arg(args, 6, "")?;
        let array = flag(args, 7, false)?;
        let function_is_key = flag(args, 8, false)?;
        let call_id = text_arg(args, 9, "")?;
        let generated_id = text_arg(args, 10, "")?;
        let order = args
            .get(11)
            .and_then(Value::as_array)
            .map_or_else(Vec::new, |values| {
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            });
        let openai = flag(args, 12, false)?;
        let effective_name = if name_key.is_empty() {
            "name"
        } else {
            &name_key
        };
        let effective_args = if args_key.is_empty() {
            "arguments"
        } else {
            &args_key
        };
        if !call_id.is_empty() || !generated_id.is_empty() {
            self.json_number()?;
        }
        let mut alternatives = Vec::new();
        for tool in tools.array() {
            let Some(function) = tool.get("function") else {
                continue;
            };
            let name = function
                .get("name")
                .and_then(Ordered::text)
                .ok_or_else(|| invalid("chat function name missing"))?;
            let empty = Ordered::Object(Vec::new());
            let params = function.get("parameters").unwrap_or(&empty);
            let arguments = tagged_schema(name, params)?;
            let body = if function_is_key {
                let mut fields = Vec::new();
                for (key, number) in [(&call_id, false), (&generated_id, true)] {
                    if !key.is_empty() {
                        fields.push(optional(seq(vec![
                            wrap("atomic", field(key, id_value(number))),
                            space(),
                            optional(seq(vec![lit(","), space()])),
                        ])));
                    }
                }
                fields.push(if args_key.is_empty() {
                    arguments
                } else {
                    field(effective_args, arguments)
                });
                let inner = if args_key.is_empty() && fields.len() == 1 {
                    fields.remove(0)
                } else {
                    let mut items = vec![lit("{"), space()];
                    let length = fields.len();
                    for (index, field) in fields.into_iter().enumerate() {
                        items.push(field);
                        if index + 1 < length {
                            items.push(space());
                        }
                    }
                    items.extend([space(), lit("}")]);
                    seq(items)
                };
                seq(vec![
                    wrap("tool_open", lit("{")),
                    space(),
                    lit("\""),
                    wrap("tool_name", lit(name)),
                    lit("\""),
                    space(),
                    lit(":"),
                    space(),
                    inner,
                    space(),
                    wrap("tool_close", lit("}")),
                ])
            } else if effective_name.contains('.') || effective_args.contains('.') {
                let name_spec = effective_name
                    .split_once('.')
                    .unwrap_or(("", effective_name));
                let args_spec = effective_args
                    .split_once('.')
                    .unwrap_or(("", effective_args));
                let prefix = if name_spec.0.is_empty() {
                    args_spec.0
                } else {
                    name_spec.0
                };
                let nested_name = if name_spec.0.is_empty() {
                    effective_name
                } else {
                    name_spec.1
                };
                let nested_args = if args_spec.0.is_empty() {
                    effective_args
                } else {
                    args_spec.1
                };
                let nested = seq(vec![
                    lit("{"),
                    space(),
                    field(nested_name, tool_name(name)),
                    space(),
                    lit(","),
                    space(),
                    field(nested_args, arguments),
                    space(),
                    lit("}"),
                ]);
                let mut items = vec![wrap("tool_open", lit("{")), space()];
                for (key, number) in [(&call_id, false), (&generated_id, true)] {
                    if !key.is_empty() && !key.contains('.') {
                        items.push(optional(seq(vec![
                            wrap("atomic", field(key, id_value(number))),
                            space(),
                            lit(","),
                            space(),
                        ])));
                    }
                }
                items.extend([field(prefix, nested), space(), wrap("tool_close", lit("}"))]);
                seq(items)
            } else {
                let mut fields = vec![
                    (
                        effective_name.to_owned(),
                        field(effective_name, tool_name(name)),
                    ),
                    (effective_args.to_owned(), field(effective_args, arguments)),
                ];
                for key in [&call_id, &generated_id] {
                    if !key.is_empty() {
                        fields.push((
                            key.clone(),
                            optional(wrap("atomic", field(key, id_value(true)))),
                        ));
                    }
                }
                fields.sort_by_key(|(key, _)| {
                    order
                        .iter()
                        .position(|candidate| candidate == key)
                        .unwrap_or(order.len())
                });
                let type_field = if openai {
                    optional(seq(vec![
                        field("type", lit("\"function\"")),
                        space(),
                        lit(","),
                        space(),
                    ]))
                } else {
                    op("eps", Vec::new())
                };
                let mut items = vec![wrap("tool_open", lit("{")), space(), type_field];
                let length = fields.len();
                for (index, (_, field)) in fields.into_iter().enumerate() {
                    items.push(field);
                    if index + 1 < length {
                        items.extend([space(), lit(","), space()]);
                    }
                }
                items.extend([space(), wrap("tool_close", lit("}"))]);
                seq(items)
            };
            alternatives.push(rule(name, body));
        }
        let mut calls = choice(alternatives);
        if parallel {
            calls = seq(vec![
                calls.clone(),
                wrap("zero_or_more", seq(vec![space(), lit(","), space(), calls])),
            ]);
        }
        if array {
            calls = seq(vec![lit("["), space(), calls, space(), lit("]")]);
        }
        let section = op(
            "trigger_rule",
            vec![
                json!("tool-call"),
                seq(vec![lit(&start), space(), calls, space(), lit(&end)]),
            ],
        );
        Ok(if force { section } else { optional(section) })
    }
}
fn constructed_tools(args: &[Value]) -> Result<Value> {
    let markers = parse_tools(args, 0)?;
    let tools = parse_tools(args, 1)?;
    if tools.array().is_empty() {
        return Ok(op("eps", Vec::new()));
    }
    let marker = |key: &str, default: &str| {
        markers
            .get(key)
            .and_then(Ordered::text)
            .unwrap_or(default)
            .to_owned()
    };
    let start = marker("tool_call_start_marker", "<tool_call>");
    let end = marker("tool_call_end_marker", "</tool_call>");
    let open = marker("function_opener", "<function=");
    let suffix = marker("function_name_suffix", ">");
    let close = marker("function_closer", "</function>");
    let key_prefix = marker("parameter_key_prefix", "<param=");
    let key_suffix = marker("parameter_key_suffix", ">");
    let parameter_close = marker("parameter_closer", "</param>");
    let mut alternatives = Vec::new();
    for tool in tools.array() {
        let Some(function) = tool.get("function") else {
            continue;
        };
        let name = function
            .get("name")
            .and_then(Ordered::text)
            .ok_or_else(|| invalid("chat function name missing"))?;
        let properties = function
            .get("parameters")
            .and_then(|params| params.get("properties"));
        let args = properties
            .filter(|properties| !properties.object().is_empty())
            .map_or_else(
                || op("eps", Vec::new()),
                |properties| {
                    let mut arguments = Vec::new();
                    for (property, _) in properties.object() {
                        let names = choice(vec![
                            lit(property),
                            lit(&format!("\"{property}\"")),
                            lit(&format!("'{property}'")),
                        ]);
                        arguments.push(wrap(
                            "tool_arg",
                            seq(vec![
                                wrap("tool_arg_open", lit(&key_prefix)),
                                wrap("tool_arg_name", names),
                                lit(&key_suffix),
                                wrap(
                                    "tool_arg_json_value",
                                    op("until", vec![json!(parameter_close)]),
                                ),
                                wrap("tool_arg_close", lit(&parameter_close)),
                            ]),
                        ));
                    }
                    wrap("zero_or_more", seq(vec![choice(arguments), space()]))
                },
            );
        let body = seq(vec![
            wrap(
                "tool_open",
                seq(vec![lit(&open), wrap("tool_name", lit(name)), lit(&suffix)]),
            ),
            space(),
            wrap("tool_args", args),
            space(),
            wrap("tool_close", lit(&close)),
        ]);
        alternatives.push(rule(name, body));
    }
    let calls = seq(vec![choice(alternatives), space()]);
    let calls = if flag(args, 2, false)? {
        wrap("one_or_more", calls)
    } else {
        calls
    };
    let section = op(
        "trigger_rule",
        vec![
            json!("tool-call"),
            seq(vec![lit(&start), space(), calls, lit(&end)]),
        ],
    );
    Ok(if flag(args, 3, false)? {
        section
    } else {
        optional(section)
    })
}
