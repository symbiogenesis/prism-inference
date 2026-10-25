//! Ordered JSON Schema conversion to the pinned production GBNF contract.
//!
//! Ported from PrismML-Eng/llama.cpp `b87ca7225456b06b0975b8239c66b5aecf07e37c`,
//! `common/json-schema-to-grammar.cpp`; MIT attribution is retained in
//! `tools/prism/upstream/LICENSE`. Schema property order is preserved independently
//! of the reporting JSON representation.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::fmt::Write as _;

use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::ser::{SerializeMap, SerializeSeq};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{Result, invalid};

#[path = "schema_integer.rs"]
mod integer;
#[path = "schema_pattern.rs"]
mod pattern;

#[derive(Clone, Debug, PartialEq)]
enum Node {
    Null,
    Bool(bool),
    Number(serde_json::Number),
    String(String),
    Array(Vec<Self>),
    Object(Vec<(String, Self)>),
}

impl Node {
    fn get(&self, key: &str) -> &Self {
        if let Self::Object(values) = self {
            values
                .iter()
                .find(|(name, _)| name == key)
                .map_or(&Self::Null, |(_, value)| value)
        } else {
            &Self::Null
        }
    }
    fn has(&self, key: &str) -> bool {
        matches!(self, Self::Object(values) if values.iter().any(|(name, _)| name == key))
    }
    fn text(&self) -> Option<&str> {
        if let Self::String(value) = self {
            Some(value)
        } else {
            None
        }
    }
    fn array(&self) -> &[Self] {
        if let Self::Array(values) = self {
            values
        } else {
            &[]
        }
    }
    fn object(&self) -> &[(String, Self)] {
        if let Self::Object(values) = self {
            values
        } else {
            &[]
        }
    }
    fn is(&self, value: &str) -> bool {
        self.text() == Some(value)
    }
    fn number(&self) -> Option<i64> {
        if let Self::Number(value) = self {
            value.as_i64()
        } else {
            None
        }
    }
    fn set(&mut self, key: &str, value: Self) {
        if let Self::Object(values) = self {
            if let Some((_, existing)) = values.iter_mut().find(|(name, _)| name == key) {
                *existing = value;
            } else {
                values.push((key.to_owned(), value));
            }
        }
    }
    fn remove(&mut self, key: &str) {
        if let Self::Object(values) = self {
            values.retain(|(name, _)| name != key);
        }
    }
    const fn empty(&self) -> bool {
        matches!(self, Self::Null)
            || matches!(self, Self::Array(values) if values.is_empty())
            || matches!(self, Self::Object(values) if values.is_empty())
    }
    fn dump(&self) -> Result<String> {
        serde_json::to_string(self).map_err(Into::into)
    }
}

impl Serialize for Node {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        match self {
            Self::Null => serializer.serialize_none(),
            Self::Bool(value) => serializer.serialize_bool(*value),
            Self::Number(value) => value.serialize(serializer),
            Self::String(value) => serializer.serialize_str(value),
            Self::Array(values) => {
                let mut seq = serializer.serialize_seq(Some(values.len()))?;
                for value in values {
                    seq.serialize_element(value)?;
                }
                seq.end()
            }
            Self::Object(values) => {
                let mut map = serializer.serialize_map(Some(values.len()))?;
                for (key, value) in values {
                    map.serialize_entry(key, value)?;
                }
                map.end()
            }
        }
    }
}

struct OrderedVisitor;
impl<'de> Visitor<'de> for OrderedVisitor {
    type Value = Node;
    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ordered JSON value")
    }
    fn visit_unit<E: serde::de::Error>(self) -> std::result::Result<Node, E> {
        Ok(Node::Null)
    }
    fn visit_bool<E: serde::de::Error>(self, value: bool) -> std::result::Result<Node, E> {
        Ok(Node::Bool(value))
    }
    fn visit_i64<E: serde::de::Error>(self, value: i64) -> std::result::Result<Node, E> {
        Ok(Node::Number(value.into()))
    }
    fn visit_u64<E: serde::de::Error>(self, value: u64) -> std::result::Result<Node, E> {
        Ok(Node::Number(value.into()))
    }
    fn visit_f64<E: serde::de::Error>(self, value: f64) -> std::result::Result<Node, E> {
        serde_json::Number::from_f64(value)
            .map(Node::Number)
            .ok_or_else(|| E::custom("nonfinite JSON number"))
    }
    fn visit_str<E: serde::de::Error>(self, value: &str) -> std::result::Result<Node, E> {
        Ok(Node::String(value.to_owned()))
    }
    fn visit_string<E: serde::de::Error>(self, value: String) -> std::result::Result<Node, E> {
        Ok(Node::String(value))
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut access: A) -> std::result::Result<Node, A::Error> {
        let mut values = Vec::new();
        while let Some(value) = access.next_element()? {
            values.push(value);
        }
        Ok(Node::Array(values))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut access: A) -> std::result::Result<Node, A::Error> {
        let mut result = Node::Object(Vec::new());
        while let Some((name, value)) = access.next_entry::<String, Node>()? {
            result.set(&name, value);
        }
        Ok(result)
    }
}
impl<'de> Deserialize<'de> for Node {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        deserializer.deserialize_any(OrderedVisitor)
    }
}

fn parse(text: &str) -> Result<Node> {
    if text.len() > 262_144 {
        return Err(invalid("schema JSON exceeds byte limit"));
    }
    serde_json::from_str(text).map_err(Into::into)
}

/// Formats arbitrary text as one escaped GBNF literal.
#[must_use]
pub fn format_literal(text: &str) -> String {
    let mut result = String::from("\"");
    for value in text.chars() {
        match value {
            '\r' => result.push_str("\\r"),
            '\n' => result.push_str("\\n"),
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            _ => result.push(value),
        }
    }
    result.push('"');
    result
}

fn repetition(item: &str, minimum: i32, maximum: i32, separator: &str) -> String {
    if maximum == 0 {
        return String::new();
    }
    if minimum == 0 && maximum == 1 {
        return format!("{item}?");
    }
    if separator.is_empty() {
        return match (minimum, maximum) {
            (1, i32::MAX) => format!("{item}+"),
            (0, i32::MAX) => format!("{item}*"),
            _ => format!(
                "{item}{{{minimum},{}}}",
                if maximum == i32::MAX {
                    String::new()
                } else {
                    maximum.to_string()
                }
            ),
        };
    }
    let rest = repetition(
        &format!("({separator} {item})"),
        minimum.saturating_sub(1).max(0),
        if maximum == i32::MAX {
            maximum
        } else {
            maximum.saturating_sub(1)
        },
        "",
    );
    let result = format!("{item} {rest}");
    if minimum == 0 {
        format!("({result})?")
    } else {
        result
    }
}

fn builtin(name: &str) -> Option<(&'static str, &'static [&'static str])> {
    Some(match name {
        "boolean" => (r#"("true" | "false")"#, &[]),
        "decimal-part" => ("[0-9]{1,16}", &[]),
        "integral-part" => ("[0] | [1-9] [0-9]{0,15}", &[]),
        "number" => (
            r#"("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?"#,
            &["integral-part", "decimal-part"],
        ),
        "integer" => (r#"("-"? integral-part)"#, &["integral-part"]),
        "value" => (
            "object | array | string | number | boolean | null",
            &["object", "array", "string", "number", "boolean", "null"],
        ),
        "object" => (
            r#""{" space ( string ":" space value ("," space string ":" space value)* )? space "}""#,
            &["string", "value"],
        ),
        "array" => (
            r#""[" space ( value ("," space value)* )? space "]""#,
            &["value"],
        ),
        "uuid" => (
            r#""\"" [0-9a-fA-F]{8} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{12} "\"""#,
            &[],
        ),
        "char" => (
            r#"[^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})"#,
            &[],
        ),
        "string" => (r#""\"" char* "\"""#, &["char"]),
        "null" => (r#""null""#, &[]),
        "date" => (
            r#"[0-9]{4} "-" ( "0" [1-9] | "1" [0-2] ) "-" ( "0" [1-9] | [1-2] [0-9] | "3" [0-1] )"#,
            &[],
        ),
        "time" => (
            r#"([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9]{3} )? ( "Z" | ( "+" | "-" ) ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] )"#,
            &[],
        ),
        "date-time" => (r#"date "T" time"#, &["date", "time"]),
        "date-string" => (r#""\"" date "\"""#, &["date"]),
        "time-string" => (r#""\"" time "\"""#, &["time"]),
        "date-time-string" => (r#""\"" date-time "\"""#, &["date-time"]),
        _ => return None,
    })
}

fn escape_name(name: &str) -> String {
    let mut output = String::new();
    let mut invalid_run = false;
    for value in name.chars() {
        if value.is_ascii_alphanumeric() || value == '-' {
            output.push(value);
            invalid_run = false;
        } else if !invalid_run {
            output.push('-');
            invalid_run = true;
        }
    }
    output
}

struct Converter {
    rules: BTreeMap<String, String>,
    refs: HashMap<String, Node>,
    resolving: HashSet<String>,
    depth: usize,
    dotall: bool,
}
impl Converter {
    fn new() -> Self {
        Self {
            rules: BTreeMap::from([("space".into(), "| \" \" | \"\\n\"{1,2} [ \\t]{0,20}".into())]),
            refs: HashMap::new(),
            resolving: HashSet::new(),
            depth: 0,
            dotall: false,
        }
    }
    fn add(&mut self, name: &str, rule: &str) -> String {
        let name = escape_name(name);
        if self.rules.get(&name).is_none_or(|value| value == rule) {
            self.rules.insert(name.clone(), rule.to_owned());
            return name;
        }
        let mut index = 0;
        loop {
            let key = format!("{name}{index}");
            if self.rules.get(&key).is_none_or(|value| value == rule) {
                self.rules.insert(key.clone(), rule.to_owned());
                return key;
            }
            index += 1;
        }
    }
    fn primitive(&mut self, name: &str, kind: &str) -> Result<String> {
        let (rule, dependencies) =
            builtin(kind).ok_or_else(|| invalid(format!("unknown schema rule {kind}")))?;
        let name = self.add(name, rule);
        for dependency in dependencies {
            if !self.rules.contains_key(*dependency) {
                self.primitive(dependency, dependency)?;
            }
        }
        Ok(name)
    }
    fn union(&mut self, name: &str, alternatives: &[Node]) -> Result<String> {
        let mut rules = Vec::new();
        for (index, schema) in alternatives.iter().enumerate() {
            rules.push(self.visit(
                schema,
                &format!(
                    "{name}{}{index}",
                    if name.is_empty() { "alternative-" } else { "-" }
                ),
            )?);
        }
        Ok(rules.join(" | "))
    }
    fn resolve(&mut self, schema: &Node) -> Result<()> {
        fn walk(converter: &mut Converter, node: &Node, root: &Node, depth: usize) -> Result<()> {
            if depth > 128 {
                return Err(invalid("schema reference nesting exceeds bound"));
            }
            if let Node::Array(values) = node {
                for value in values {
                    walk(converter, value, root, depth + 1)?;
                }
            } else if node.has("$ref") {
                let reference = node
                    .get("$ref")
                    .text()
                    .ok_or_else(|| invalid("schema reference must be text"))?;
                if converter.refs.contains_key(reference) {
                    return Ok(());
                }
                if !reference.starts_with("#/") {
                    return Err(invalid(format!("unsupported schema ref {reference}")));
                }
                let mut target = root;
                for key in reference[2..].split('/') {
                    target = match target {
                        Node::Object(values) => values
                            .iter()
                            .find(|(name, _)| name == key)
                            .map(|(_, value)| value),
                        Node::Array(values) => key
                            .parse::<usize>()
                            .ok()
                            .and_then(|index| values.get(index)),
                        _ => None,
                    }
                    .ok_or_else(|| invalid(format!("unresolved schema ref {reference}")))?;
                }
                converter.refs.insert(reference.to_owned(), target.clone());
            } else {
                for (_, value) in node.object() {
                    walk(converter, value, root, depth + 1)?;
                }
            }
            Ok(())
        }
        walk(self, schema, schema, 0)
    }
    fn reference(&mut self, reference: &str) -> Result<String> {
        let fragment = reference
            .split_once('#')
            .map_or(reference, |(_, fragment)| fragment);
        let name = format!("ref{}", escape_name(fragment));
        if self.rules.contains_key(&name) || !self.resolving.insert(reference.to_owned()) {
            return Ok(name);
        }
        let resolved = self
            .refs
            .get(reference)
            .cloned()
            .ok_or_else(|| invalid(format!("unresolved schema ref {reference}")))?;
        let result = self.visit(&resolved, &name);
        self.resolving.remove(reference);
        result
    }
    #[allow(
        clippy::too_many_lines,
        reason = "Preserve both source object-order alternatives and optional-field recursion together"
    )]
    fn object_rule(
        &mut self,
        properties: &[(String, Node)],
        required: &HashSet<String>,
        name: &str,
        additional: &Node,
    ) -> Result<String> {
        if properties.len() > 256 || properties.iter().any(|(key, _)| key.len() > 256) {
            return Err(invalid("schema object property resource limit exceeded"));
        }
        let prefix = if name.is_empty() {
            String::new()
        } else {
            format!("{name}-")
        };
        let mut names = Vec::new();
        let mut mandatory = Vec::new();
        let mut optional = Vec::new();
        let mut rules = HashMap::new();
        for (key, schema) in properties {
            let value = self.visit(schema, &format!("{prefix}{key}"))?;
            let rule = self.add(
                &format!("{prefix}{key}-kv"),
                &format!(
                    "{} space \":\" space {value}",
                    format_literal(&Node::String(key.clone()).dump()?)
                ),
            );
            rules.insert(key.clone(), rule);
            names.push(key.clone());
            if required.contains(key) {
                mandatory.push(key.clone());
            } else {
                optional.push(key.clone());
            }
        }
        if matches!(additional, Node::Bool(true) | Node::Object(_)) {
            let subname = format!("{prefix}additional");
            let value = if matches!(additional, Node::Object(_)) {
                self.visit(additional, &format!("{subname}-value"))?
            } else {
                self.primitive("value", "value")?
            };
            let key = if names.is_empty() {
                self.primitive("string", "string")?
            } else {
                let rule = self.not_strings(&names)?;
                self.add(&format!("{subname}-k"), &rule)
            };
            let rule = self.add(
                &format!("{subname}-kv"),
                &format!("{key} \":\" space {value}"),
            );
            rules.insert("*".into(), rule);
            optional.push("*".into());
        }
        let mut seen_optional = false;
        let mut interleaved = false;
        for key in &names {
            if required.contains(key) {
                interleaved |= seen_optional;
            } else {
                seen_optional = true;
            }
        }
        let declared = if interleaved
            && names.first().is_some_and(|key| required.contains(key))
            && !rules.contains_key("*")
        {
            let mut result = format!("\"{{\" space {}", rules[&names[0]]);
            for key in &names[1..] {
                let kv = format!(" \",\" space {}", rules[key]);
                if required.contains(key) {
                    result.push_str(&kv);
                } else {
                    write!(result, " ({kv} )?")?;
                }
            }
            result.push_str(" space \"}\"");
            Some(result)
        } else {
            None
        };
        let mut result = format!(
            "\"{{\" space {}",
            mandatory
                .iter()
                .map(|key| rules[key].clone())
                .collect::<Vec<_>>()
                .join(" \",\" space ")
        );
        if !optional.is_empty() {
            result.push_str(" (");
            if !mandatory.is_empty() {
                result.push_str(" \",\" space ( ");
            }
            let mut alternatives = Vec::new();
            for index in 0..optional.len() {
                alternatives.push(self.optional_rest(&optional[index..], false, &rules, &prefix));
            }
            result.push_str(&alternatives.join(" | "));
            if !mandatory.is_empty() {
                result.push_str(" )");
            }
            result.push_str(" )?");
        }
        result.push_str(" space \"}\"");
        Ok(declared.map_or_else(
            || result.clone(),
            |declared| format!("{declared} | {result}"),
        ))
    }
    fn optional_rest(
        &mut self,
        keys: &[String],
        optional: bool,
        rules: &HashMap<String, String>,
        prefix: &str,
    ) -> String {
        let Some(key) = keys.first() else {
            return String::new();
        };
        let kv = &rules[key];
        let comma = format!("( \",\" space {kv} )");
        let mut result = if optional {
            format!("{comma}{}", if key == "*" { "*" } else { "?" })
        } else if key == "*" {
            format!("{kv} {comma}*")
        } else {
            kv.clone()
        };
        if keys.len() > 1 {
            let rest = self.optional_rest(&keys[1..], true, rules, prefix);
            let reference = self.add(&format!("{prefix}{key}-rest"), &rest);
            result.push(' ');
            result.push_str(&reference);
        }
        result
    }
    fn not_strings(&mut self, strings: &[String]) -> Result<String> {
        #[derive(Default)]
        struct Trie {
            children: BTreeMap<u8, Self>,
            end: bool,
        }
        fn render(node: &Trie) -> String {
            let mut result = Vec::new();
            let rejects: String = node
                .children
                .keys()
                .map(|&value| char::from(value))
                .collect();
            for (&value, child) in &node.children {
                let mut rule = format!("[{}]", char::from(value));
                if !child.children.is_empty() {
                    rule.push_str(" (");
                    rule.push_str(&render(child));
                    rule.push(')');
                } else if child.end {
                    rule.push_str(" char+");
                }
                result.push(rule);
            }
            if !node.children.is_empty() {
                result.push(format!("[^\"{rejects}] char*"));
            }
            result.join(" | ")
        }
        let mut trie = Trie::default();
        for text in strings {
            let mut node = &mut trie;
            for value in text.bytes() {
                node = node.children.entry(value).or_default();
            }
            node.end = true;
        }
        self.primitive("char", "char")?;
        Ok(format!(
            "[\"] ( {} ){} [\"]",
            render(&trie),
            if trie.end { "" } else { "?" }
        ))
    }
    fn visit(&mut self, schema: &Node, name: &str) -> Result<String> {
        self.depth += 1;
        if self.depth > 128 || self.rules.len() > 65536 {
            return Err(invalid("schema grammar nesting/rule bound exceeded"));
        }
        let result = self.visit_inner(schema, name);
        self.depth -= 1;
        result
    }
    #[allow(
        clippy::too_many_lines,
        reason = "Preserve the pinned converter's ordered schema keyword precedence"
    )]
    fn visit_inner(&mut self, schema: &Node, name: &str) -> Result<String> {
        for key in ["oneOf", "anyOf", "type"] {
            if matches!(schema.get(key), Node::Array(values) if values.is_empty()) {
                let mut rest = schema.clone();
                rest.remove(key);
                return self.visit(&rest, name);
            }
        }
        let kind = schema.get("type");
        let format = schema.get("format").text().unwrap_or("");
        let rule_name = if name == "root" || builtin(name).is_some() {
            format!("{name}-")
        } else if name.is_empty() {
            "root".to_owned()
        } else {
            name.to_owned()
        };
        if schema.has("$ref") {
            let value = self.reference(
                schema
                    .get("$ref")
                    .text()
                    .ok_or_else(|| invalid("schema ref must be text"))?,
            )?;
            return Ok(self.add(&rule_name, &value));
        }
        if schema.has("oneOf") || schema.has("anyOf") {
            let alternatives = schema
                .get(if schema.has("oneOf") {
                    "oneOf"
                } else {
                    "anyOf"
                })
                .array();
            let rule = self.union(name, alternatives)?;
            return Ok(self.add(&rule_name, &rule));
        }
        if let Node::Array(types) = kind {
            let alternatives: Vec<_> = types
                .iter()
                .map(|kind| {
                    let mut copy = schema.clone();
                    copy.set("type", kind.clone());
                    copy
                })
                .collect();
            let rule = self.union(name, &alternatives)?;
            return Ok(self.add(&rule_name, &rule));
        }
        if schema.has("const") {
            return Ok(self.add(&rule_name, &format_literal(&schema.get("const").dump()?)));
        }
        if schema.has("enum") {
            let values = schema
                .get("enum")
                .array()
                .iter()
                .map(|value| value.dump().map(|value| format_literal(&value)))
                .collect::<Result<Vec<_>>>()?;
            return Ok(self.add(&rule_name, &format!("({})", values.join(" | "))));
        }
        if (matches!(kind, Node::Null) || kind.is("object"))
            && (schema.has("properties")
                || schema.has("additionalProperties")
                    && schema.get("additionalProperties") != &Node::Bool(true))
        {
            let required: HashSet<_> = schema
                .get("required")
                .array()
                .iter()
                .filter_map(Node::text)
                .map(str::to_owned)
                .collect();
            let rule = self.object_rule(
                schema.get("properties").object(),
                &required,
                name,
                schema.get("additionalProperties"),
            )?;
            return Ok(self.add(&rule_name, &rule));
        }
        if (matches!(kind, Node::Null) || kind.is("object") || kind.is("string"))
            && schema.has("allOf")
        {
            let mut properties = Vec::new();
            let mut required = HashSet::new();
            let mut enums = BTreeMap::new();
            for component in schema.get("allOf").array() {
                let (components, mandatory) = if component.has("anyOf") {
                    (component.get("anyOf").array().to_vec(), false)
                } else {
                    (vec![component.clone()], true)
                };
                for component in components {
                    self.component(
                        &component,
                        mandatory,
                        &mut properties,
                        &mut required,
                        &mut enums,
                        0,
                    )?;
                }
            }
            if !enums.is_empty() {
                let intersection: Vec<_> = enums
                    .into_iter()
                    .filter_map(|(key, count)| {
                        (count == schema.get("allOf").array().len()).then_some(key)
                    })
                    .collect();
                if !intersection.is_empty() {
                    return Ok(self.add(&rule_name, &format!("({})", intersection.join(" | "))));
                }
            }
            let rule = self.object_rule(&properties, &required, name, &Node::Null)?;
            return Ok(self.add(&rule_name, &rule));
        }
        let prefix = if name.is_empty() {
            String::new()
        } else {
            format!("{name}-")
        };
        if (matches!(kind, Node::Null) || kind.is("array"))
            && (schema.has("items") || schema.has("prefixItems"))
        {
            let items = schema.get(if schema.has("items") {
                "items"
            } else {
                "prefixItems"
            });
            let rule = if let Node::Array(values) = items {
                let mut rules = Vec::new();
                for (index, value) in values.iter().enumerate() {
                    rules.push(self.visit(value, &format!("{prefix}tuple-{index}"))?);
                }
                format!("\"[\" space {} space \"]\"", rules.join(" \",\" space "))
            } else {
                let item = self.visit(items, &format!("{prefix}item"))?;
                format!(
                    "\"[\" space {} space \"]\"",
                    repetition(
                        &item,
                        bound(schema, "minItems", 0)?,
                        bound(schema, "maxItems", i32::MAX)?,
                        "\",\" space"
                    )
                )
            };
            return Ok(self.add(&rule_name, &rule));
        }
        if (matches!(kind, Node::Null) || kind.is("string")) && schema.has("pattern") {
            return self.pattern(
                schema
                    .get("pattern")
                    .text()
                    .ok_or_else(|| invalid("schema pattern must be text"))?,
                &rule_name,
            );
        }
        if matches!(kind, Node::Null) || kind.is("string") {
            if format == "uuid"
                || format.strip_prefix("uuid").is_some_and(|suffix| {
                    suffix.len() == 1 && matches!(suffix.as_bytes()[0], b'1'..=b'5')
                })
            {
                return self.primitive(if rule_name == "root" { "root" } else { format }, "uuid");
            }
            let primitive = format!("{format}-string");
            if builtin(&primitive).is_some() {
                let value = self.primitive(&primitive, &primitive)?;
                return Ok(self.add(&rule_name, &value));
            }
        }
        if kind.is("string") && (schema.has("minLength") || schema.has("maxLength")) {
            let character = self.primitive("char", "char")?;
            return Ok(self.add(
                &rule_name,
                &format!(
                    "\"\\\"\" {} \"\\\"\"",
                    repetition(
                        &character,
                        bound(schema, "minLength", 0)?,
                        bound(schema, "maxLength", i32::MAX)?,
                        ""
                    )
                ),
            ));
        }
        if kind.is("integer")
            && ["minimum", "exclusiveMinimum", "maximum", "exclusiveMaximum"]
                .iter()
                .any(|key| schema.has(key))
        {
            let minimum = schema.get("minimum").number().unwrap_or(
                schema
                    .get("exclusiveMinimum")
                    .number()
                    .map_or(Ok(i64::MIN), |v| {
                        v.checked_add(1)
                            .ok_or_else(|| invalid("schema integer lower bound overflow"))
                    })?,
            );
            let maximum = schema.get("maximum").number().unwrap_or(
                schema
                    .get("exclusiveMaximum")
                    .number()
                    .map_or(Ok(i64::MAX), |v| {
                        v.checked_sub(1)
                            .ok_or_else(|| invalid("schema integer upper bound overflow"))
                    })?,
            );
            return Ok(self.add(
                &rule_name,
                &format!("({})", integer::build(minimum, maximum, 16, true)?),
            ));
        }
        if schema.empty() || kind.is("object") {
            let value = self.primitive("object", "object")?;
            return Ok(self.add(&rule_name, &value));
        }
        if matches!(kind, Node::Null) && matches!(schema, Node::Object(_)) {
            let value = self.primitive("value", "value")?;
            return Ok(self.add(&rule_name, &value));
        }
        let kind = kind
            .text()
            .filter(|kind| builtin(kind).is_some())
            .ok_or_else(|| {
                invalid(format!(
                    "unrecognized schema {}",
                    schema.dump().unwrap_or_default()
                ))
            })?;
        self.primitive(if rule_name == "root" { "root" } else { kind }, kind)
    }
    fn component(
        &self,
        component: &Node,
        mandatory: bool,
        properties: &mut Vec<(String, Node)>,
        required: &mut HashSet<String>,
        enums: &mut BTreeMap<String, usize>,
        depth: usize,
    ) -> Result<()> {
        if depth > 128 {
            return Err(invalid("schema intersection ref depth exceeded"));
        }
        if component.has("$ref") {
            let target = self
                .refs
                .get(component.get("$ref").text().unwrap_or(""))
                .ok_or_else(|| invalid("unresolved intersection ref"))?;
            return self.component(target, mandatory, properties, required, enums, depth + 1);
        }
        if component.has("properties") {
            for (key, value) in component.get("properties").object() {
                properties.push((key.clone(), value.clone()));
                if mandatory {
                    required.insert(key.clone());
                }
            }
        } else if component.has("enum") {
            for value in component.get("enum").array() {
                *enums.entry(format_literal(&value.dump()?)).or_default() += 1;
            }
        }
        Ok(())
    }
}
fn bound(schema: &Node, name: &str, default: i32) -> Result<i32> {
    schema.get(name).number().map_or(Ok(default), |value| {
        i32::try_from(value).map_err(Into::into)
    })
}

/// Converts ordered JSON Schema text into deterministic GBNF rules.
///
/// # Errors
/// Rejects malformed/oversized JSON, unresolved references, invalid patterns,
/// unknown schemas, and bounded grammar recursion/output limits.
pub fn to_grammar(schema_json: &str, force_gbnf: bool) -> Result<String> {
    if !force_gbnf {
        return Err(invalid(
            "direct llguidance schema mode is unavailable; request GBNF conversion explicitly",
        ));
    }
    let schema = parse(schema_json)?;
    let mut converter = Converter::new();
    converter.resolve(&schema)?;
    converter.visit(&schema, "")?;
    let mut grammar = String::new();
    for (name, rule) in converter.rules {
        writeln!(grammar, "{name} ::= {rule}")?;
        if grammar.len() > 4 * 1024 * 1024 {
            return Err(invalid("schema grammar exceeds output bound"));
        }
    }
    Ok(grammar)
}

/// Shared schema lowering state used by composed production PEG grammars.
pub(crate) struct RuleBuilder {
    converter: Converter,
}
impl RuleBuilder {
    pub(crate) fn new() -> Self {
        Self {
            converter: Converter::new(),
        }
    }
    pub(crate) fn resolve(&mut self, schema_json: &str) -> Result<()> {
        self.converter.resolve(&parse(schema_json)?)
    }
    pub(crate) fn add(&mut self, name: &str, schema_json: &str) -> Result<String> {
        self.converter
            .visit(&parse(schema_json)?, if name == "root" { "" } else { name })
    }
    pub(crate) fn into_rules(self) -> BTreeMap<String, String> {
        self.converter.rules
    }
}

/// Ordered schema inspection with reusable resolved local references.
pub struct SchemaInfo {
    schema: Node,
    converter: Converter,
}
impl SchemaInfo {
    /// Parses bounded schema JSON while preserving property order.
    /// # Errors
    /// Rejects malformed or oversized JSON.
    pub fn new(schema_json: &str) -> Result<Self> {
        Ok(Self {
            schema: parse(schema_json)?,
            converter: Converter::new(),
        })
    }
    /// Resolves local references from the original schema document.
    /// # Errors
    /// Rejects missing or unsupported references and excessive nesting.
    pub fn resolve_refs(&mut self) -> Result<()> {
        self.converter.resolve(&self.schema)
    }
    /// Serializes the inspected schema in its original property order.
    /// # Errors
    /// Returns JSON serialization errors.
    pub fn schema_json(&self) -> Result<String> {
        self.schema.dump()
    }
    /// Reports whether a source schema branch permits string values.
    #[must_use]
    pub fn resolves_to_string(&self) -> bool {
        fn check(
            node: &Node,
            refs: &HashMap<String, Node>,
            visited: &mut HashSet<String>,
            depth: usize,
        ) -> bool {
            if depth > 128 || !matches!(node, Node::Object(_)) {
                return false;
            }
            if node.has("$ref") {
                let reference = node.get("$ref").text().unwrap_or("");
                return visited.insert(reference.to_owned())
                    && refs
                        .get(reference)
                        .is_some_and(|target| check(target, refs, visited, depth + 1));
            }
            if node.get("type").is("string")
                || node
                    .get("type")
                    .array()
                    .iter()
                    .any(|kind| kind.is("string"))
            {
                return true;
            }
            for key in ["oneOf", "anyOf"] {
                if node
                    .get(key)
                    .array()
                    .iter()
                    .any(|value| check(value, refs, visited, depth + 1))
                {
                    return true;
                }
            }
            if node.has("allOf")
                && node
                    .get("allOf")
                    .array()
                    .iter()
                    .all(|value| check(value, refs, visited, depth + 1))
            {
                return true;
            }
            if node.get("const").text().is_some()
                || node
                    .get("enum")
                    .array()
                    .iter()
                    .any(|value| value.text().is_some())
            {
                return true;
            }
            if ["pattern", "minLength", "maxLength"]
                .iter()
                .any(|key| node.has(key))
            {
                return true;
            }
            node.get("format").text().is_some_and(|format| {
                matches!(
                    format,
                    "date" | "time" | "date-time" | "uri" | "email" | "hostname" | "ipv4" | "ipv6"
                ) || format.starts_with("uuid")
            })
        }
        check(&self.schema, &self.converter.refs, &mut HashSet::new(), 0)
    }
}
