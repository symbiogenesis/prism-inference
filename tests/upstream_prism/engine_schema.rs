//! JSON framing for production schema conversion and reference inspection.

use bonsai_prism_runtime::schema::{self, SchemaInfo};
use serde_json::{Value, json};

use super::llama_grammar::{array, exact_fields, field};
use super::{Result, invalid};

fn text<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
    field(value, name)?
        .as_str()
        .ok_or_else(|| invalid("schema field must be text"))
}

fn operation(operation: &Value) -> Result<Value> {
    let source = text(operation, "SchemaJson")?;
    if source.len() > 262_144 {
        return Err(invalid("schema JSON exceeds source transport bound"));
    }
    let kind = text(operation, "Kind")?;
    if kind == "json-schema-info" {
        exact_fields(operation, &["Kind", "SchemaJson", "Actions"])?;
        let actions = array(operation, "Actions")?;
        if actions.len() > 128 {
            return Err(invalid("schema info action count exceeds bound"));
        }
        let mut info = SchemaInfo::new(source)?;
        let mut trace = Vec::new();
        for action in actions {
            trace.push(match action.as_str() {
                Some("resolve-refs") => {
                    info.resolve_refs()?;
                    json!({"SchemaJson": info.schema_json()?})
                }
                Some("resolves-to-string") => {
                    json!({"ResolvesToString": info.resolves_to_string()})
                }
                _ => return Err(invalid("unknown schema info action")),
            });
        }
        return Ok(json!({"Trace": trace}));
    }
    exact_fields(operation, &["Kind", "SchemaJson", "ForceGbnf"])?;
    let force = field(operation, "ForceGbnf")?
        .as_bool()
        .ok_or_else(|| invalid("ForceGbnf must be boolean"))?;
    if kind == "json-schema-to-grammar" {
        return Ok(json!({"GrammarText": schema::to_grammar(source, force)?}));
    }
    match schema::to_grammar(source, force) {
        Ok(grammar) => Ok(json!({"Grammar": grammar, "ExceptionKind": null, "Message": ""})),
        Err(error) => Ok(
            json!({"Grammar": null, "ExceptionKind": "invalid-argument", "Message": error.to_string()}),
        ),
    }
}

pub(super) fn execute(request: &Value) -> Result<Option<Value>> {
    let Some(operations) = request["Operations"].as_array() else {
        return Ok(None);
    };
    if !operations.iter().all(|operation| {
        matches!(
            operation["Kind"].as_str(),
            Some("json-schema-convert" | "json-schema-info" | "json-schema-to-grammar")
        )
    }) {
        return Ok(None);
    }
    exact_fields(request, &["SchemaVersion", "Operations"])?;
    if request["SchemaVersion"] != 1 || operations.is_empty() || operations.len() > 128 {
        return Err(invalid("invalid schema transport framing"));
    }
    Ok(Some(
        json!({"SchemaVersion": 1, "Results": operations.iter().map(operation).collect::<Result<Vec<_>>>()?}),
    ))
}
