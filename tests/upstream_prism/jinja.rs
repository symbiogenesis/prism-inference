//! Safe Rust ownership of all default test-jinja.cpp contracts.
//!
//! Ported from PrismML-Eng/llama.cpp b87ca7225456b06b0975b8239c66b5aecf07e37c;
//! see tools/prism/upstream/LICENSE for the MIT terms. Literal expectations are
//! transcribed from source, never obtained from an engine. `VariablesJson` retains
//! source object insertion order and floating point literals such as 1.0.

use bonsai_prism_runtime::jinja::PathSegment;
use serde_json::{Value, json};

use super::engine::Engine;
use super::{Result, invalid, result};

const ID: &str = "tests/test-jinja.cpp";
const SOURCE_SHA256: &str = "ef2da16176a8e914cbd54341e4b77f675a8f3b5b15a8bfd0c9a7a2305c30c8ab";
const LITERALS: &str = include_str!("jinja-cases.json");
const VARIABLES: &[&str] = &[
    "x",
    "y",
    "z",
    "arr",
    "obj",
    "items",
    "foo",
    "bar",
    "undefined_var",
    "none",
    "true",
    "false",
    "None",
    "True",
    "False",
];
const FILTERS: &[&str] = &[
    "length", "first", "last", "reverse", "sort", "unique", "join", "upper", "lower", "trim",
    "default", "tojson", "string", "int", "float", "abs", "list", "dictsort",
];
const BUILTINS: &[&str] = &[
    "range",
    "len",
    "dict",
    "list",
    "join",
    "str",
    "int",
    "float",
    "namespace",
];
const COMMON_VARS: &str =
    r#"{"x":42,"y":"hello","arr":[1,2,3],"obj":{"a":1,"b":2},"items":["a","b","c"]}"#;

fn bytes(value: &Value) -> Result<Vec<u8>> {
    value
        .as_array()
        .ok_or("Jinja bytes must be an array")?
        .iter()
        .map(|value| {
            u8::try_from(value.as_u64().ok_or("Jinja byte must be unsigned")?).map_err(Into::into)
        })
        .collect()
}
fn path(value: &Value) -> Result<Vec<PathSegment>> {
    value
        .as_array()
        .ok_or("Jinja path must be an array")?
        .iter()
        .map(|value| {
            if let Some(key) = value.as_str() {
                Ok(PathSegment::Key(key.into()))
            } else {
                Ok(PathSegment::Index(usize::try_from(
                    value.as_u64().ok_or("Jinja path index must be unsigned")?,
                )?))
            }
        })
        .collect()
}

/// Observe the reusable production Jinja API without adding execution to the harness.
#[allow(
    clippy::unnecessary_wraps,
    reason = "Keep the fallible optional-operation dispatch signature shared by engine adapters"
)]
pub(super) fn execute(request: &Value) -> Result<Option<Value>> {
    use bonsai_prism_runtime::jinja::{Hasher, Template, builtin_inventory};
    let Some(operations) = request.get("Operations").and_then(Value::as_array) else {
        return Ok(None);
    };
    if !operations.iter().all(|operation| {
        operation
            .get("Kind")
            .and_then(Value::as_str)
            .is_some_and(|kind| kind.starts_with("jinja-"))
    }) {
        return Ok(None);
    }
    let mut results = Vec::new();
    for operation in operations {
        let observed = (|| -> Result<Value> {
            let kind = text(operation, "Kind")?;
            if kind == "jinja-builtins" {
                return Ok(
                    json!({"Builtins":builtin_inventory().iter().flat_map(|(category,names)|names.iter().map(move|name|json!({"Type":category,"Name":name}))).collect::<Vec<_>>()}),
                );
            }
            if kind == "jinja-hash" {
                let mut hash = Hasher::new();
                for chunk in array(operation, "Chunks")? {
                    hash.update(&bytes(chunk)?);
                }
                return Ok(json!({"Digest":hash.digest()}));
            }
            if kind == "jinja-hash-trace" {
                let mut hash = Hasher::new();
                let mut digests = Vec::new();
                for action in array(operation, "Actions")? {
                    if let Some(update) = action.get("Update") {
                        hash.update(&bytes(update)?);
                    } else if action.get("Digest") == Some(&Value::Bool(true)) {
                        digests.push(hash.digest());
                    } else {
                        return Err(invalid("invalid Jinja hash action"));
                    }
                }
                return Ok(json!({"Digests":digests}));
            }
            let template = Template::parse(text(operation, "Template")?)?;
            if kind == "jinja-caps" {
                let (string, typed) = template.content_capabilities()?;
                return Ok(json!({"SupportsStringContent":string,"SupportsTypedContent":typed}));
            }
            let variables = if let Some(json) = operation.get("VariablesJson") {
                json.as_str()
                    .ok_or("Jinja VariablesJson must be text")?
                    .to_owned()
            } else {
                operation
                    .get("Variables")
                    .map_or_else(|| Ok("{}".to_owned()), serde_json::to_string)?
            };
            let mut overlays = Vec::new();
            if let Some(values) = operation.get("ByteStrings") {
                for overlay in values
                    .as_array()
                    .ok_or("Jinja ByteStrings must be an array")?
                {
                    overlays.push((
                        path(field(overlay, "Path")?)?,
                        bytes(field(overlay, "Bytes")?)?,
                    ));
                }
            }
            let execution = template.execute_with_bytes(&variables, &overlays)?;
            if let Some(error) = &execution.error {
                return Err(invalid(error.clone()));
            }
            match kind {
                "jinja-render" => Ok(json!({"Rendered":String::from_utf8(execution.bytes())?})),
                "jinja-string-parts" => Ok(
                    json!({"Parts":execution.parts().into_iter().map(|part|Ok(json!({"Value":String::from_utf8(part.bytes)?,"IsInput":part.is_input}))).collect::<Result<Vec<_>>>()?}),
                ),
                "jinja-stats" => Ok(
                    json!({"Used":array(operation,"Paths")?.iter().map(|value|Ok(execution.used(&path(value)?))).collect::<Result<Vec<_>>>()?}),
                ),
                _ => Err(invalid("unknown Jinja observation")),
            }
        })();
        results.push(match observed {
            Ok(value) => value,
            Err(error) => json!({"ExceptionKind":"standard","Message":error.to_string()}),
        });
    }
    Ok(Some(json!({"SchemaVersion":1,"Results":results})))
}

#[derive(Default)]
struct Assertions {
    count: usize,
    skipped: usize,
    failures: Vec<String>,
}

impl Assertions {
    fn check(&mut self, name: impl Into<String>, passed: bool) {
        self.count += 1;
        if !passed {
            self.failures.push(name.into());
        }
    }
}

fn field<'a>(value: &'a Value, name: &str) -> Result<&'a Value> {
    value
        .get(name)
        .ok_or_else(|| invalid(format!("missing Jinja response field {name}")))
}

fn text<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
    field(value, name)?
        .as_str()
        .ok_or_else(|| invalid(format!("Jinja {name} must be text")))
}

fn boolean(value: &Value, name: &str) -> Result<bool> {
    field(value, name)?
        .as_bool()
        .ok_or_else(|| invalid(format!("Jinja {name} must be boolean")))
}

fn array<'a>(value: &'a Value, name: &str) -> Result<&'a [Value]> {
    field(value, name)?
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| invalid(format!("Jinja {name} must be an array")))
}

fn exact_fields(value: &Value, names: &[&str]) -> Result<()> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid("Jinja result must be an object"))?;
    if object.len() != names.len() || names.iter().any(|name| !object.contains_key(*name)) {
        return Err(invalid("Jinja result field mismatch"));
    }
    Ok(())
}

fn exchange(engine: &Engine, name: &str, operations: &[Value]) -> Result<Option<Vec<Value>>> {
    let mut output = Vec::with_capacity(operations.len());
    for chunk in operations.chunks(64) {
        let request = json!({"SchemaVersion": 1, "Operations": chunk});
        let Some(response) = engine.execute_common(name, &request)? else {
            return Ok(None);
        };
        exact_fields(&response, &["SchemaVersion", "Results"])?;
        if field(&response, "SchemaVersion")?.as_u64() != Some(1) {
            return Err(invalid("Jinja common response schema version mismatch"));
        }
        let results = array(&response, "Results")?;
        if results.len() != chunk.len() {
            return Err(invalid("Jinja common response result count mismatch"));
        }
        output.extend_from_slice(results);
    }
    Ok(Some(output))
}

fn render_operation(template: &str, variables_json: &str) -> Value {
    json!({"Kind": "jinja-render", "Template": template, "VariablesJson": variables_json})
}

enum Render<'a> {
    Text(&'a str),
    Exception(&'a str),
}

fn render_result(value: &Value) -> Result<Render<'_>> {
    if value.get("Rendered").is_some() {
        exact_fields(value, &["Rendered"])?;
        return Ok(Render::Text(text(value, "Rendered")?));
    }
    exact_fields(value, &["ExceptionKind", "Message"])?;
    let kind = text(value, "ExceptionKind")?;
    let _ = text(value, "Message")?;
    if !matches!(kind, "not-implemented" | "standard" | "unknown") {
        return Err(invalid("unknown Jinja exception kind"));
    }
    Ok(Render::Exception(kind))
}

fn literal_assertions(engine: &Engine, assertions: &mut Assertions) -> Result<()> {
    let fixture: Value = serde_json::from_str(LITERALS)?;
    if field(&fixture, "SchemaVersion")?.as_u64() != Some(1)
        || text(&fixture, "Source")? != ID
        || text(&fixture, "SourceSha256")? != SOURCE_SHA256
    {
        return Err(invalid("Jinja literal source provenance mismatch"));
    }
    let cases = array(&fixture, "Cases")?;
    if cases.len() != 278 {
        return Err(invalid("Jinja requires all 278 original literal cases"));
    }
    let operations = cases
        .iter()
        .map(|case| {
            Ok(render_operation(
                text(case, "Template")?,
                text(case, "VariablesJson")?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let values = exchange(engine, "jinja-literals", &operations)?
        .ok_or_else(|| invalid("Jinja production backend disappeared"))?;
    for (case, value) in cases.iter().zip(values) {
        match render_result(&value)? {
            Render::Text(actual) => assertions.check(
                format!(
                    "{}.{}: render mismatch",
                    text(case, "Group")?,
                    text(case, "Name")?
                ),
                actual == text(case, "Expected")?,
            ),
            // test_template_cpp explicitly skips only this exception category.
            Render::Exception("not-implemented") => assertions.skipped += 1,
            Render::Exception(kind) => assertions.check(
                format!(
                    "{}.{}: unexpected {kind} exception",
                    text(case, "Group")?,
                    text(case, "Name")?
                ),
                false,
            ),
        }
    }
    Ok(())
}

// Windows' original rand() default seed and recurrence. The unsigned wrapping
// and low eight bits reproduce test_hasher's rand()%256 on the target platform.
struct HasherRandom(u32);

impl HasherRandom {
    fn bytes(&mut self, count: usize) -> Vec<u8> {
        (0..count)
            .map(|_| {
                self.0 = self.0.wrapping_mul(214_013).wrapping_add(2_531_011);
                self.0.to_le_bytes()[2]
            })
            .collect()
    }
}

fn hash_operation(chunks: &[Vec<u8>]) -> Value {
    json!({"Kind": "jinja-hash", "Chunks": chunks})
}

fn empty_update_assertion(engine: &Engine, assertions: &mut Assertions) -> Result<()> {
    let trace = [json!({"Kind":"jinja-hash-trace", "Actions":[
        {"Update":b"some data".to_vec()}, {"Digest":true}, {"Update":[]}, {"Digest":true}
    ]})];
    let trace_values = exchange(engine, "jinja-hasher-empty-update", &trace)?
        .ok_or_else(|| invalid("Jinja production backend disappeared"))?;
    exact_fields(&trace_values[0], &["Digests"])?;
    let digests = array(&trace_values[0], "Digests")?;
    if digests.len() != 2 {
        return Err(invalid(
            "Jinja hash trace requires both digest observations",
        ));
    }
    let first = digests[0]
        .as_u64()
        .ok_or_else(|| invalid("Jinja digest must be uint64"))?;
    let second = digests[1]
        .as_u64()
        .ok_or_else(|| invalid("Jinja digest must be uint64"))?;
    assertions.check(
        "hasher.empty update leaves state unchanged after digest",
        first == second,
    );
    Ok(())
}

fn hash_assertions(engine: &Engine, assertions: &mut Assertions) -> Result<()> {
    empty_update_assertion(engine, assertions)?;
    let mut operations = Vec::new();
    let mut comparisons = Vec::new();
    let mut pair = |name: String, left: Vec<Vec<u8>>, right: Vec<Vec<u8>>, equal| {
        comparisons.push((name, operations.len(), operations.len() + 1, equal));
        operations.push(hash_operation(&left));
        operations.push(hash_operation(&right));
    };
    pair(
        "different inputs differ".to_owned(),
        vec![b"data one".to_vec()],
        vec![b"data two".to_vec()],
        false,
    );
    pair(
        "same inputs agree".to_owned(),
        vec![b"consistent data".to_vec()],
        vec![b"consistent data".to_vec()],
        true,
    );
    let sizes = [
        (1, 2),
        (1, 16),
        (8, 1),
        (1, 1024),
        (5, 512),
        (16, 256),
        (45, 122),
        (70, 634),
    ];
    let mut random = HasherRandom(1);
    for (left_size, right_size) in sizes {
        let left = random.bytes(left_size);
        let right = random.bytes(right_size);
        let combined = [left.as_slice(), right.as_slice()].concat();
        pair(
            format!("combined update ({left_size},{right_size})"),
            vec![left, right],
            vec![combined],
            true,
        );
    }
    let chunks: Vec<Vec<u8>> = [3, 732, 131, 13, 17, 256, 436, 99, 4]
        .into_iter()
        .map(|size| random.bytes(size))
        .collect();
    pair(
        "many update passes".to_owned(),
        chunks.clone(),
        vec![chunks.concat()],
        true,
    );
    for (left_size, right_size) in sizes {
        let left = random.bytes(left_size);
        let right = random.bytes(right_size);
        pair(
            format!("non-associativity ({left_size},{right_size})"),
            vec![left.clone(), right.clone()],
            vec![right, left],
            false,
        );
    }
    let mut bytes = random.bytes(64);
    for _ in 0..16 {
        let previous = bytes.clone();
        bytes.push(b'A');
        pair(
            format!("padding length {}", bytes.len()),
            vec![previous],
            vec![bytes.clone()],
            false,
        );
    }
    // Preserve the stronger all-byte/block-boundary coverage from the Rust
    // helper test alongside the original 36 source hash properties.
    let bytes: Vec<u8> = (0..=255).collect();
    for stride in 1..=16 {
        pair(
            format!("all-byte stream stride {stride}"),
            bytes.chunks(stride).map(<[u8]>::to_vec).collect(),
            vec![bytes.clone()],
            true,
        );
    }
    let values = exchange(engine, "jinja-hasher", &operations)?
        .ok_or_else(|| invalid("Jinja production backend disappeared"))?;
    let digests = values
        .iter()
        .map(|value| {
            exact_fields(value, &["Digest"])?;
            field(value, "Digest")?
                .as_u64()
                .ok_or_else(|| invalid("Jinja digest must be uint64"))
        })
        .collect::<Result<Vec<_>>>()?;
    for (name, left, right, equal) in comparisons {
        assertions.check(
            format!("hasher.{name}"),
            (digests[left] == digests[right]) == equal,
        );
    }
    Ok(())
}

fn stats_assertions(engine: &Engine, assertions: &mut Assertions) -> Result<()> {
    let template =
        "{{val.num}} {{val.str}} {{val.arr[0]}} {{val.obj.key1}} {{val.nested | tojson}}";
    let variables = r#"{"val":{"num":1,"str":"abc","arr":[1,2,3],"obj":{"key1":1,"key2":2,"key3":3},"nested":{"inner_key1":[1,2],"inner_key2":{"a":"x","b":"y"}},"mixed":{"used":1,"unused":2}}}"#;
    let paths = json!([
        ["val", "num"],
        ["val", "str"],
        ["val", "arr"],
        ["val", "arr", 0],
        ["val", "arr", 1],
        ["val", "obj"],
        ["val", "obj", "key1"],
        ["val", "obj", "key2"],
        ["val", "nested", "inner_key1", 0],
        ["val", "nested", "inner_key2", "a"],
    ]);
    let operations = [
        json!({"Kind":"jinja-stats", "Template":template, "VariablesJson":variables, "Paths":paths}),
    ];
    let results = exchange(engine, "jinja-stats", &operations)?
        .ok_or_else(|| invalid("Jinja production backend disappeared"))?;
    exact_fields(&results[0], &["Used"])?;
    let actual = array(&results[0], "Used")?;
    let expected = [true, true, true, true, false, true, true, false, true, true];
    if actual.len() != expected.len() {
        return Err(invalid("Jinja stats path count mismatch"));
    }
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assertions.check(
            format!("stats path {index}"),
            actual
                .as_bool()
                .ok_or_else(|| invalid("Jinja Used must contain booleans"))?
                == expected,
        );
    }
    Ok(())
}

fn caps_assertions(engine: &Engine, assertions: &mut Assertions) -> Result<()> {
    let cases = [
        (
            "{% for message in messages %}{{ message['role'] + ': ' + message['content'] }}{% endfor %}",
            true,
            false,
        ),
        (
            "{% for message in messages %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] }}{% endfor %}{% endfor %}",
            false,
            true,
        ),
        (
            "{% for message in messages %}{{ message['content'][0]['text'] }}{% endfor %}",
            false,
            true,
        ),
    ];
    let operations: Vec<Value> = cases
        .iter()
        .map(|(template, _, _)| json!({"Kind":"jinja-caps","Template":template}))
        .collect();
    let values = exchange(engine, "jinja-caps", &operations)?
        .ok_or_else(|| invalid("Jinja production backend disappeared"))?;
    for (index, ((_, string, typed), value)) in cases.iter().zip(values).enumerate() {
        exact_fields(&value, &["SupportsStringContent", "SupportsTypedContent"])?;
        assertions.check(
            format!("caps case {index} string"),
            boolean(&value, "SupportsStringContent")? == *string,
        );
        assertions.check(
            format!("caps case {index} typed"),
            boolean(&value, "SupportsTypedContent")? == *typed,
        );
    }
    Ok(())
}

fn parts_assertions(engine: &Engine, assertions: &mut Assertions) -> Result<()> {
    let operations = [
        json!({"Kind":"jinja-string-parts", "Template":"{{ val.a }}{{ val.b }}-{{ val.c }}", "VariablesJson":r#"{"val":{"a":"A","b":"B","c":"C"}}"#}),
    ];
    let results = exchange(engine, "jinja-parts", &operations)?
        .ok_or_else(|| invalid("Jinja production backend disappeared"))?;
    exact_fields(&results[0], &["Parts"])?;
    let parts = array(&results[0], "Parts")?;
    assertions.check("3 parts after merge", parts.len() == 3);
    // Preserve the original guard: inspect parts only after the extent passes.
    if parts.len() == 3 {
        for (index, (part, (value, input))) in parts
            .iter()
            .zip([("AB", true), ("-", false), ("C", true)])
            .enumerate()
        {
            exact_fields(part, &["Value", "IsInput"])?;
            assertions.check(
                format!("string part {index}"),
                text(part, "Value")? == value && boolean(part, "IsInput")? == input,
            );
        }
    }
    Ok(())
}

// Standard MT19937 recurrence and the MSVC uniform_int_distribution mapping
// used by the original Windows ARM64 suite. Every requested interval here is
// smaller than 2^32, so one draw suffices before Lemire rejection.
struct Random {
    state: [u32; 624],
    index: usize,
}

impl Random {
    fn new() -> Self {
        let mut state = [0_u32; 624];
        state[0] = 42;
        for (increment, index) in (1_u32..).zip(1..624) {
            let previous = state[index - 1];
            state[index] = 1_812_433_253_u32
                .wrapping_mul(previous ^ (previous >> 30))
                .wrapping_add(increment);
        }
        Self { state, index: 624 }
    }

    fn next(&mut self) -> u32 {
        if self.index == 624 {
            for index in 0..624 {
                let value = (self.state[index] & 0x8000_0000)
                    | (self.state[(index + 1) % 624] & 0x7fff_ffff);
                self.state[index] = self.state[(index + 397) % 624]
                    ^ (value >> 1)
                    ^ if value & 1 == 0 { 0 } else { 0x9908_b0df };
            }
            self.index = 0;
        }
        let mut value = self.state[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^ (value >> 18)
    }

    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "The distribution explicitly splits the 64-bit product into low and high 32-bit words."
    )]
    fn uniform(&mut self, maximum: u32) -> u32 {
        let range = maximum + 1;
        let threshold = range.wrapping_neg() % range;
        loop {
            let product = u64::from(self.next()) * u64::from(range);
            if product as u32 >= threshold {
                return (product >> 32) as u32;
            }
        }
    }

    fn choice(&mut self) -> usize {
        usize::from(self.uniform(100).to_le_bytes()[0])
    }

    fn integer(&mut self) -> i64 {
        i64::from(self.uniform(2000)) - 1000
    }

    fn string(&mut self, maximum: u8) -> String {
        const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_";
        let length = self.uniform(u32::from(maximum));
        (0..length)
            .map(|_| char::from(CHARSET[usize::from(self.uniform(62).to_le_bytes()[0])]))
            .collect()
    }
}

struct Fuzz {
    name: String,
    operation: Value,
}

fn fuzz(cases: &mut Vec<Fuzz>, name: impl Into<String>, template: &str, variables_json: &str) {
    cases.push(Fuzz {
        name: name.into(),
        operation: render_operation(template, variables_json),
    });
}

fn fuzz_random_access(cases: &mut Vec<Fuzz>, random: &mut Random) {
    for index in 0..100 {
        fuzz(
            cases,
            format!("out of bound array access {index}"),
            &format!("{{{{ arr[{}] }}}}", random.integer()),
            r#"{"arr":[1,2,3]}"#,
        );
    }
    for index in 0..100 {
        fuzz(
            cases,
            format!("non-existing variables {index}"),
            &format!("{{{{ {} }}}}", random.string(20)),
            "{}",
        );
    }
    for index in 0..100 {
        let var1 = VARIABLES[random.choice() % VARIABLES.len()];
        let var2 = random.string(10);
        let var3 = random.string(10);
        fuzz(
            cases,
            format!("non-existing nested attributes {index}"),
            &format!("{{{{ {var1}.{var2}.{var3} }}}}"),
            &format!(r#"{{"{var1}":{{"other":123}}}}"#),
        );
    }
}

fn fuzz_random_calls(cases: &mut Vec<Fuzz>, random: &mut Random) {
    for index in 0..100 {
        let filter = FILTERS[random.choice() % FILTERS.len()];
        fuzz(
            cases,
            format!("invalid filter arguments {index}"),
            &format!("{{{{ {} | {filter} }}}}", random.integer()),
            "{}",
        );
    }
    for index in 0..100 {
        let left = FILTERS[random.choice() % FILTERS.len()];
        let right = FILTERS[random.choice() % FILTERS.len()];
        let variable = VARIABLES[random.choice() % VARIABLES.len()];
        fuzz(
            cases,
            format!("chained filters {index}"),
            &format!("{{{{ {variable} | {left} | {right} }}}}"),
            COMMON_VARS,
        );
    }
    for index in 0..100 {
        let builtin = BUILTINS[random.choice() % BUILTINS.len()];
        let argument = match random.choice() % 4 {
            0 => "\"not a number\"".to_owned(),
            1 => "none".to_owned(),
            2 => random.integer().to_string(),
            _ => "[]".to_owned(),
        };
        fuzz(
            cases,
            format!("invalid builtin calls {index}"),
            &format!("{{{{ {builtin}({argument}) }}}}"),
            "{}",
        );
    }
}

fn fuzz_fixed(cases: &mut Vec<Fuzz>) {
    for (name, template) in [
        (
            "macro no args with args",
            "{% macro foo() %}hello{% endmacro %}{{ foo(1, 2, 3) }}",
        ),
        (
            "macro with args no args",
            "{% macro foo(a, b, c) %}{{ a }}{{ b }}{{ c }}{% endmacro %}{{ foo() }}",
        ),
        (
            "recursive macro",
            "{% macro foo(n) %}{% if n > 0 %}{{ foo(n - 1) }}{% endif %}{% endmacro %}{{ foo(5) }}",
        ),
    ] {
        fuzz(cases, name, template, "{}");
    }
    for _ in 0..10 {
        fuzz(
            cases,
            "nested macro",
            "{% macro outer() %}{% macro inner() %}x{% endmacro %}{{ inner() }}{% endmacro %}{{ outer() }}",
            "{}",
        );
    }
    for template in [
        r#"{{ "" | first }}"#,
        r#"{{ "" | last }}"#,
        "{{ [] | first }}",
        "{{ [] | last }}",
        "{{ none.attr }}",
        "{{ none | length }}",
        "{{ none | default('fallback') }}",
        "{{ {} | first }}",
        "{{ {} | dictsort }}",
    ] {
        fuzz(cases, format!("empty/none: {template}"), template, "{}");
    }
    for template in [
        "{{ 1 / 0 }}",
        "{{ 1 // 0 }}",
        "{{ 1 % 0 }}",
        "{{ 999999999999999999 * 999999999999999999 }}",
        "{{ -999999999999999999 - 999999999999999999 }}",
        "{{ 1.0 / 0.0 }}",
        "{{ 0.0 / 0.0 }}",
    ] {
        fuzz(cases, format!("arith: {template}"), template, "{}");
    }
    for depth in 1..=10 {
        let mut template = String::new();
        for index in 0..depth {
            template.push_str("{% for i");
            template.push_str(&index.to_string());
            template.push_str(" in arr %}");
        }
        template.push('x');
        template.push_str(&"{% endfor %}".repeat(depth));
        fuzz(
            cases,
            format!("nested loops depth {depth}"),
            &template,
            r#"{"arr":[1,2]}"#,
        );
    }
    for depth in 1..=10 {
        let template = format!(
            "{}x{}",
            "{% if true %}".repeat(depth),
            "{% endif %}".repeat(depth)
        );
        fuzz(cases, format!("nested ifs depth {depth}"), &template, "{}");
    }
    for template in [
        r#"{{ "}{%" }}"#,
        r#"{{ "}}{{" }}"#,
        r#"{{ "{%%}" }}"#,
        r#"{{ "\n\t\r" }}"#,
        r#"{{ "'\"'" }}"#,
        r#"{{ "hello\x00world" }}"#,
    ] {
        fuzz(cases, format!("special: {template}"), template, "{}");
    }
}

fn fuzz_templates(cases: &mut Vec<Fuzz>, random: &mut Random) {
    let fragments = [
        "{{ x }}",
        "{{ y }}",
        "{{ arr }}",
        "{{ obj }}",
        "{% if true %}a{% endif %}",
        "{% if false %}b{% else %}c{% endif %}",
        "{% for i in arr %}{{ i }}{% endfor %}",
        "{{ x | length }}",
        "{{ x | first }}",
        "{{ x | default(0) }}",
        "{{ x + y }}",
        "{{ x - y }}",
        "{{ x * y }}",
        "{{ x == y }}",
        "{{ x != y }}",
        "{{ x > y }}",
        "{{ range(3) }}",
        r#"{{ "hello" | upper }}"#,
        "text",
        " ",
        "\n",
    ];
    for index in 0..100 {
        let mut template = String::new();
        for _ in 0..=(random.choice() % 10) {
            template.push_str(fragments[random.choice() % fragments.len()]);
        }
        let variables = format!(
            r#"{{"x":{},"y":{},"arr":[1,2,3],"obj":{{"a":1,"b":2}}}}"#,
            random.integer(),
            random.integer()
        );
        fuzz(
            cases,
            format!("random template #{index}"),
            &template,
            &variables,
        );
    }
    for template in [
        "",
        "{{ x",
        "{% if %}",
        "{% for %}",
        "{% for x in %}",
        "{% endfor %}",
        "{% endif %}",
        "{{ | filter }}",
        "{% if x %}",
        "{% for i in x %}",
        "{{ x | }}",
        "{% macro %}{% endmacro %}",
        "{{{{",
        "}}}}",
        "{%%}",
        "{% set %}",
        "{% set x %}",
    ] {
        fuzz(cases, format!("malformed: {template}"), template, "{}");
    }
    let mut template = "{% for message in messages %}{{ message.role | string }} : {{ message.content if ('content' in message and message.content is not none) }}{% endfor %".to_owned();
    while !template.is_empty() {
        fuzz(cases, format!("malformed: {template}"), &template, "{}");
        template.pop();
    }
    let operators = ["+", "-", "*", "/", "==", "~"];
    for index in 0..100 {
        let operator = operators[random.choice() % operators.len()];
        let left = VARIABLES[random.choice() % VARIABLES.len()];
        let right = VARIABLES[random.choice() % VARIABLES.len()];
        fuzz(
            cases,
            format!("type coercion: #{index}"),
            &format!("{{{{ {left} {operator} {right} }}}}"),
            r#"{"x":42,"y":"hello","z":3.14,"arr":[1,2,3],"obj":{"a":1},"items":[],"foo":null,"bar":true}"#,
        );
    }
}

fn random_argument(random: &mut Random) -> String {
    match random.choice() % 8 {
        0 => random.integer().to_string(),
        1 => format!("{}.5", random.integer()),
        2 => format!("\"{}\"", random.string(10)),
        3 => "true".to_owned(),
        4 => "false".to_owned(),
        5 => "none".to_owned(),
        6 => "[1, 2, 3]".to_owned(),
        _ => "{\"a\": 1}".to_owned(),
    }
}

fn fuzz_builtins(cases: &mut Vec<Fuzz>, random: &mut Random, builtins: &[Value]) -> Result<()> {
    if builtins.is_empty() {
        return Err(invalid("Jinja builtin inventory must not be empty"));
    }
    let kwargs = [
        "base",
        "attribute",
        "default",
        "reverse",
        "case_sensitive",
        "by",
        "safe",
        "chars",
        "separators",
        "sort_keys",
        "indent",
        "ensure_ascii",
    ];
    for index in 0..100 {
        let builtin = &builtins[random.choice() % builtins.len()];
        exact_fields(builtin, &["Type", "Name"])?;
        let type_name = text(builtin, "Type")?;
        let function = text(builtin, "Name")?;
        let mut arguments = Vec::new();
        for _ in 0..random.choice() % 6 {
            arguments.push(if random.choice().is_multiple_of(3) {
                let name = kwargs[random.choice() % kwargs.len()];
                format!("{name}={}", random_argument(random))
            } else {
                random_argument(random)
            });
        }
        let arguments = arguments.join(", ");
        let template = if type_name == "global" {
            format!("{{{{ {function}({arguments}) }}}}")
        } else {
            let base = match type_name {
                "int" => random.integer().to_string(),
                "float" => format!("{}.5", random.integer()),
                "string" => "\"test_string\"".to_owned(),
                "array" => "[1, 2, 3, \"a\", \"b\"]".to_owned(),
                "object" => "{\"x\": 1, \"y\": 2}".to_owned(),
                _ => return Err(invalid("unknown Jinja builtin type")),
            };
            format!("{{{{ {base}.{function}({arguments}) }}}}")
        };
        fuzz(
            cases,
            format!("builtin {type_name}.{function} #{index}"),
            &template,
            r#"{"x":42,"y":"hello","arr":[1,2,3],"obj":{"a":1,"b":2}}"#,
        );
    }
    let mut operation = render_operation("{{ data|tojson(ensure_ascii=true) }}", r#"{"data":""}"#);
    operation["ByteStrings"] =
        json!([{"Path":["data"],"Bytes":[104,101,108,108,111,254,255,119,111,114,108,100]}]);
    cases.push(Fuzz {
        name: "invalid utf-8 does not crash".to_owned(),
        operation,
    });
    Ok(())
}

fn fuzz_assertions(
    engine: &Engine,
    assertions: &mut Assertions,
    builtins: &[Value],
) -> Result<usize> {
    let mut cases = Vec::new();
    let mut random = Random::new();
    fuzz_random_access(&mut cases, &mut random);
    fuzz_random_calls(&mut cases, &mut random);
    fuzz_fixed(&mut cases);
    fuzz_templates(&mut cases, &mut random);
    fuzz_builtins(&mut cases, &mut random, builtins)?;
    let operations: Vec<Value> = cases.iter().map(|case| case.operation.clone()).collect();
    let values = exchange(engine, "jinja-fuzz", &operations)?
        .ok_or_else(|| invalid("Jinja production backend disappeared"))?;
    for (case, value) in cases.iter().zip(values) {
        // The original fuzz helper accepts successful rendering and every
        // exception. A crash/timeout fails before a response can be validated.
        let _ = render_result(&value)?;
        assertions.check(format!("fuzz.{}", case.name), true);
    }
    Ok(cases.len())
}

pub(super) fn run(engine: &Engine) -> Result<Value> {
    let Some(inventory) = exchange(
        engine,
        "jinja-builtins",
        &[json!({"Kind":"jinja-builtins"})],
    )?
    else {
        return Ok(result(ID, false, "unsupported production API: Jinja lexer/parser/runtime/hasher/stats/caps/string-parts; all 19 original default groups required".to_owned()));
    };
    exact_fields(&inventory[0], &["Builtins"])?;
    let builtins = array(&inventory[0], "Builtins")?;
    let mut assertions = Assertions::default();
    literal_assertions(engine, &mut assertions)?;
    hash_assertions(engine, &mut assertions)?;
    stats_assertions(engine, &mut assertions)?;
    caps_assertions(engine, &mut assertions)?;
    parts_assertions(engine, &mut assertions)?;
    let fuzz_count = fuzz_assertions(engine, &mut assertions, builtins)?;
    Ok(result(
        ID,
        assertions.failures.is_empty(),
        format!(
            "19 original default groups; 278 literal renders; 36 original hash properties plus16 all-byte streaming checks; 10 stats flags; 6 caps flags; 4 guarded string-parts assertions; {fuzz_count} no-crash fuzz calls; assertions={}; original_not_implemented_skips={}; failed={}; {}",
            assertions.count,
            assertions.skipped,
            assertions.failures.len(),
            assertions.failures.join("; ")
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn production_jinja_resource_bounds() -> Result<()> {
        use bonsai_prism_runtime::jinja::Template;
        for source in [
            "{{ é }}".to_owned(),
            format!("{{{{ {}1 }}}}", "-".repeat(256)),
            format!("{{{{ {}1 }}}}", "1+".repeat(256)),
        ] {
            assert!(Template::parse(&source).is_err());
        }
        for source in [
            "{% set x=[] %}{% set y=x.append(x) %}",
            "{% set x=namespace() %}{% set x.a=x %}",
            "{% macro f() %}{{f()}}{% endmacro %}{{f()}}",
            "{{range(0,1000000)}}",
            "{% set x=['a' * 4096] * 65536 %}{{x}}",
        ] {
            let rendered = Template::parse(source)?.execute("{}")?;
            assert!(rendered.error.is_some(), "{source}");
        }
        let rendered = Template::parse("{{a + b}}")?.execute(r#"{"a":9007199254740992,"b":1}"#)?;
        assert!(rendered.error.is_none());
        assert_eq!(rendered.bytes(), b"9007199254740993");
        assert!(Template::parse("ok")?.execute(r#"{"a":1,"a":2}"#).is_err());
        let rendered = Template::parse("{{data}}")?.execute_with_bytes(
            r#"{"data":""}"#,
            &[(vec![PathSegment::Key("data".into())], vec![0xff, 0x80])],
        )?;
        assert_eq!(rendered.bytes(), [0xff, 0x80]);
        assert!(rendered.error.is_none());
        Ok(())
    }
    #[test]
    fn production_original_jinja_observations() -> Result<()> {
        let engine = Engine::new(
            super::super::engine::Target::Rust,
            None,
            None,
            None,
            None,
            None,
        )?;
        let mut assertions = Assertions::default();
        hash_assertions(&engine, &mut assertions)?;
        stats_assertions(&engine, &mut assertions)?;
        caps_assertions(&engine, &mut assertions)?;
        parts_assertions(&engine, &mut assertions)?;
        assert!(
            assertions.failures.is_empty(),
            "{}",
            assertions.failures.join("\n")
        );
        //56 original observations plus16 additional all-byte streaming checks.
        assert_eq!(assertions.count, 72);
        Ok(())
    }
    #[test]
    fn production_original_jinja_fuzz() -> Result<()> {
        let engine = Engine::new(
            super::super::engine::Target::Rust,
            None,
            None,
            None,
            None,
            None,
        )?;
        let inventory = exchange(
            &engine,
            "jinja-builtins",
            &[json!({"Kind":"jinja-builtins"})],
        )?
        .ok_or("Jinja builtin inventory missing")?;
        let mut assertions = Assertions::default();
        let count = fuzz_assertions(&engine, &mut assertions, array(&inventory[0], "Builtins")?)?;
        assert!(
            assertions.failures.is_empty(),
            "{}",
            assertions.failures.join("\n")
        );
        assert!(count > 500);
        Ok(())
    }
    #[test]
    fn production_original_jinja_literals() -> Result<()> {
        let fixture: Value = serde_json::from_str(LITERALS)?;
        let mut failures = Vec::new();
        for case in array(&fixture, "Cases")? {
            let operation = json!({"Kind":"jinja-render","Template":field(case,"Template")?,"VariablesJson":field(case,"VariablesJson")?});
            let response =
                execute(&json!({"Operations":[operation]}))?.ok_or("missing Jinja response")?;
            let observed = &response["Results"][0];
            if observed.get("Rendered") != case.get("Expected") {
                failures.push(format!(
                    "{} {}: {:?}, expected {:?}",
                    text(case, "Group")?,
                    text(case, "Name")?,
                    observed,
                    field(case, "Expected")?
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "{} failures:\n{}",
            failures.len(),
            failures.join("\n")
        );
        Ok(())
    }
}
