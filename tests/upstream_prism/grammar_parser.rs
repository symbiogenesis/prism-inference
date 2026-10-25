//! Complete safe Rust assertions from pinned `tests/test-grammar-parser.cpp`.
//!
//! Literal grammar inputs, symbol maps and flattened rule elements come only
//! from the MIT-licensed upstream source. See tools/prism/upstream/LICENSE.

use serde_json::{Map, Value, json};

use super::engine::Engine;
use super::{Result, invalid, result};

const ID: &str = "tests/test-grammar-parser.cpp";
const SOURCE_SHA256: &str = "4c3f3b8ba3f4aa426b750815c7420c5af88672037c98f324168038dedd913ece";
const INPUTS: &str = include_str!("grammar-parser-cases.json");
const ORIGINAL_ASSERTIONS: usize = 274;
type Symbol = (String, u32);
type Element = (u32, u32);

struct Case {
    grammar: String,
    parsed: bool,
    symbols: Vec<Symbol>,
    elements: Vec<Element>,
}

struct Observation {
    parsed: bool,
    symbols: Vec<Symbol>,
    elements: Vec<Element>,
    rule_count: usize,
}

fn fields<'a>(value: &'a Value, names: &[&str]) -> Result<&'a Map<String, Value>> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid("grammar parser value must be an object"))?;
    if object.len() != names.len() || names.iter().any(|name| !object.contains_key(*name)) {
        return Err(invalid("grammar parser object fields mismatch"));
    }
    Ok(object)
}

fn array(value: &Value) -> Result<&[Value]> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| invalid("grammar parser value must be an array"))
}

fn number(value: &Value) -> Result<u32> {
    u32::try_from(
        value
            .as_u64()
            .ok_or_else(|| invalid("grammar parser integer must be unsigned"))?,
    )
    .map_err(Into::into)
}

fn symbols(value: &Value) -> Result<Vec<Symbol>> {
    let values = array(value)?;
    if values.len() > 4096 {
        return Err(invalid("grammar parser symbol count exceeds bound"));
    }
    values
        .iter()
        .map(|value| {
            let object = fields(value, &["Name", "Id"])?;
            let name = object["Name"]
                .as_str()
                .ok_or_else(|| invalid("grammar symbol Name must be text"))?;
            Ok((name.to_owned(), number(&object["Id"])?))
        })
        .collect()
}

fn elements(values: &[Value]) -> Result<Vec<Element>> {
    if values.len() > 65_536 {
        return Err(invalid("grammar parser element count exceeds bound"));
    }
    values
        .iter()
        .map(|value| {
            let object = fields(value, &["Type", "Value"])?;
            let kind = number(&object["Type"])?;
            if kind > 9 {
                return Err(invalid("unknown pinned grammar element type"));
            }
            // Types 8 and 9 are TOKEN and TOKEN_NOT, not character elements.
            Ok((kind, number(&object["Value"])?))
        })
        .collect()
}

fn cases() -> Result<Vec<Case>> {
    let source: Value = serde_json::from_str(INPUTS)?;
    let object = fields(
        &source,
        &[
            "SchemaVersion",
            "UpstreamCommit",
            "SourceSha256",
            "OriginalAssertionCount",
            "SuccessfulVariants",
            "FailureVariants",
            "Cases",
        ],
    )?;
    if object["SchemaVersion"] != 1
        || object["UpstreamCommit"] != "b87ca7225456b06b0975b8239c66b5aecf07e37c"
        || object["SourceSha256"] != SOURCE_SHA256
        || object["OriginalAssertionCount"].as_u64() != Some(u64::try_from(ORIGINAL_ASSERTIONS)?)
        || object["SuccessfulVariants"] != 17
        || object["FailureVariants"] != 6
    {
        return Err(invalid("grammar parser source corpus provenance mismatch"));
    }
    let mut cases = Vec::new();
    let mut original_assertions = 0;
    for value in array(&object["Cases"])? {
        let object = fields(
            value,
            &[
                "Grammar",
                "ExpectParsed",
                "Symbols",
                "Elements",
                "OriginalAssertions",
            ],
        )?;
        let grammar = object["Grammar"]
            .as_str()
            .ok_or_else(|| invalid("grammar source must be text"))?;
        let parsed = object["ExpectParsed"]
            .as_bool()
            .ok_or_else(|| invalid("grammar expectation must be boolean"))?;
        let symbols = symbols(&object["Symbols"])?;
        let elements = elements(array(&object["Elements"])?)?;
        let count = if parsed {
            1 + symbols.len() + elements.len()
        } else {
            1
        };
        if object["OriginalAssertions"].as_u64() != Some(u64::try_from(count)?) {
            return Err(invalid("grammar source assertion count mismatch"));
        }
        original_assertions += count;
        cases.push(Case {
            grammar: grammar.to_owned(),
            parsed,
            symbols,
            elements,
        });
    }
    if cases.len() != 23
        || cases.iter().filter(|case| case.parsed).count() != 17
        || original_assertions != ORIGINAL_ASSERTIONS
    {
        return Err(invalid("grammar source variant inventory mismatch"));
    }
    Ok(cases)
}

fn observation(value: &Value) -> Result<Observation> {
    let object = fields(value, &["Parsed", "SymbolIds", "Rules"])?;
    let parsed = object["Parsed"]
        .as_bool()
        .ok_or_else(|| invalid("grammar Parsed must be boolean"))?;
    let rules = array(&object["Rules"])?;
    if rules.len() > 4096 {
        return Err(invalid("grammar rule count exceeds bound"));
    }
    let mut flattened = Vec::new();
    for rule in rules {
        let values = elements(array(rule)?)?;
        if values.len() > 65_536 - flattened.len() {
            return Err(invalid("grammar total element count exceeds bound"));
        }
        flattened.extend(values);
    }
    Ok(Observation {
        parsed,
        symbols: symbols(&object["SymbolIds"])?,
        elements: flattened,
        rule_count: rules.len(),
    })
}

fn evaluate(cases: &[Case], values: &[Value]) -> Result<Value> {
    if values.len() != cases.len() {
        return Err(invalid("grammar parser response count mismatch"));
    }
    let mut failures = Vec::new();
    for (index, (case, value)) in cases.iter().zip(values).enumerate() {
        let actual = observation(value)?;
        if actual.parsed != case.parsed {
            failures.push(format!(
                "case {index}: parse return {} != {}",
                actual.parsed, case.parsed
            ));
        }
        if !case.parsed {
            // The original failure contract checks rules.empty(), independent
            // of any partially assigned symbols left after the failed parse.
            if actual.rule_count != 0 {
                failures.push(format!("case {index}: failed grammar retained rules"));
            }
            continue;
        }
        if actual.symbols.len() != case.symbols.len() {
            failures.push(format!("case {index}: symbol count mismatch"));
        }
        for (symbol, (actual, expected)) in actual.symbols.iter().zip(&case.symbols).enumerate() {
            if actual != expected {
                failures.push(format!(
                    "case {index} symbol {symbol}: {actual:?} != {expected:?}"
                ));
            }
        }
        // Upstream compares flattened elements in rule/vector order. Exact
        // extent additionally prevents a dropped trailing rule from passing.
        if actual.elements.len() != case.elements.len() {
            failures.push(format!("case {index}: flattened rule extent mismatch"));
        }
        for (element, (actual, expected)) in actual.elements.iter().zip(&case.elements).enumerate()
        {
            if actual != expected {
                failures.push(format!(
                    "case {index} element {element}: {actual:?} != {expected:?}"
                ));
            }
        }
    }
    Ok(result(
        ID,
        failures.is_empty(),
        format!(
            "23 original variants (17 parse, 6 rejection); 274 source assertions; \
         exact symbol ID/order and flattened type/value pairs including TOKEN/TOKEN_NOT; \
         parse-return and complete-extent guards; failed={}; {}",
            failures.len(),
            failures.join("; ")
        ),
    ))
}

pub(super) fn run(engine: &Engine) -> Result<Value> {
    let cases = cases()?;
    let operations: Vec<Value> = cases
        .iter()
        .map(|case| json!({"Kind": "grammar-parser-parse", "Grammar": case.grammar}))
        .collect();
    let request = json!({"SchemaVersion": 1, "Operations": operations});
    let Some(response) = engine.execute_common("grammar-parser", &request)? else {
        return Ok(result(ID, false, "unsupported production API: llama_grammar_parser; all 23 variants and 274 source assertions required".to_owned()));
    };
    let object = fields(&response, &["SchemaVersion", "Results"])?;
    if object["SchemaVersion"] != 1 {
        return Err(invalid("grammar bridge version mismatch"));
    }
    evaluate(&cases, array(&object["Results"])?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_trailing_elements_and_wrong_token_types_cannot_pass() -> Result<()> {
        let case = Case {
            grammar: "root ::= <[1000]>".to_owned(),
            parsed: true,
            symbols: vec![("root".to_owned(), 0)],
            elements: vec![(8, 1000), (0, 0)],
        };
        let cases = [case];
        let valid = json!({"Parsed": true, "SymbolIds": [{"Name": "root", "Id": 0}], "Rules": [[{"Type": 8, "Value": 1000}, {"Type": 0, "Value": 0}]]});
        assert_eq!(
            evaluate(&cases, std::slice::from_ref(&valid))?["Status"],
            "pass"
        );
        let mut missing = valid.clone();
        missing["Rules"][0] = json!([{ "Type": 8, "Value": 1000 }]);
        assert_eq!(evaluate(&cases, &[missing])?["Status"], "fail");
        let mut wrong_type = valid;
        wrong_type["Rules"][0][0]["Type"] = json!(9);
        assert_eq!(evaluate(&cases, &[wrong_type])?["Status"], "fail");
        assert!(evaluate(&cases, &[]).is_err());
        assert!(
            observation(
                &json!({"Parsed": true, "SymbolIds": [], "Rules": [[{"Type": 10, "Value": 0}]]})
            )
            .is_err()
        );
        Ok(())
    }
}
