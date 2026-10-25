// SPDX-License-Identifier: MIT
//! Rust-owned C++ scalar, container and library operations used by the source program.

use super::super::{Result, invalid};
use super::interpreter::Interpreter;
use super::number::{Kind, Number};
use super::program::Expression;
use super::random::Mt19937;
use super::value::{Buffer, Element, Value};
use serde_json::Value as Json;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::rc::Rc;

pub(super) fn literal(value: &Json, kind: Option<&str>) -> Result<Value> {
    if value.is_null() {
        return Ok(Value::Null);
    }
    if let Some(value) = value.as_bool() {
        return Ok(Value::Bool(value));
    }
    if let Some(value) = value.as_str() {
        return Ok(Value::Bytes(value.as_bytes().to_vec()));
    }
    if let Some(values) = value.as_array() {
        return values
            .iter()
            .map(|value| literal(value, None))
            .collect::<Result<Vec<_>>>()
            .map(Value::array);
    }
    if let Some(record) = value.get("Record").and_then(Json::as_object) {
        let mut fields = std::collections::BTreeMap::new();
        for (name, value) in record {
            let value = literal(value, None)?;
            let value = match name.as_str() {
                "type" => value.coerce("ggml_type"),
                "ne" => value.coerce("std::array<int64_t,4>"),
                "nb" => value.coerce("std::array<size_t,4>"),
                _ => Ok(value),
            }
            .map_err(invalid)?;
            fields.insert(name.clone(), value);
        }
        return Ok(Value::Record(Rc::new(RefCell::new(fields))));
    }
    if let Some(name) = value.get("Function").and_then(Json::as_str) {
        return Ok(Value::Function(name.into()));
    }
    let number = if let Some(value) = value.as_f64().filter(|_| value.is_f64()) {
        Number::F64(value)
    } else {
        Number::from_json(value).map_err(invalid)?
    };
    let number = match kind {
        Some("float" | "f32") => number.cast(Kind::F32),
        Some("double" | "f64") => number.cast(Kind::F64),
        Some("u32") => number.cast(Kind::U32),
        Some("u64") => number.cast(Kind::U64),
        Some("i64") => number.cast(Kind::I64),
        _ => number,
    };
    Ok(Value::Number(number))
}

pub(super) fn zero(kind: &str) -> Value {
    // Template arguments can be pointers while the owning container is a value.
    let outer = kind.rsplit_once('>').map_or(kind, |(_, suffix)| suffix);
    if outer.contains('*') {
        return Value::Null;
    }
    if kind.contains("std::random_device") {
        return Value::RandomDevice;
    }
    if kind.contains("std::string") {
        return Value::Bytes(Vec::new());
    }
    if kind.contains("std::array<") || kind.contains("std::vector<") {
        return Value::array(Vec::new());
    }
    if kind == "bool" || kind == "constbool" {
        return Value::Bool(false);
    }
    Value::Number(Number::I32(0))
}

pub(super) fn cast(value: Value, kind: &str) -> Result<Value> {
    if kind.contains('*') {
        if matches!(value, Value::TensorBytes { .. }) {
            return Ok(value);
        }
        return Ok(match value {
            Value::Null => Value::Null,
            value => Value::Pointer {
                value: Box::new(value),
                index: 0,
            },
        });
    }
    if kind == "bool" {
        return value.truth().map(Value::Bool).map_err(invalid);
    }
    let number = value.number().map_err(invalid)?;
    let number = match kind {
        "char" | "int8_t" => Number::I32(i32::from(number.as_i64() as i8)),
        "uint8_t" => Number::I32(i32::from(number.as_u64() as u8)),
        "int16_t" => Number::I32(i32::from(number.as_i64() as i16)),
        "uint16_t" => Number::I32(i32::from(number.as_u64() as u16)),
        _ => number.cast(Kind::from_type(kind)),
    };
    Ok(Value::Number(number))
}

pub(super) fn size_of(runtime: &Interpreter<'_>, arguments: &[Expression]) -> Result<Value> {
    let [Expression::Id { name }] = arguments else {
        return Err(invalid("source sizeof operand differs"));
    };
    let size = match name.as_str() {
        "float" | "int" | "int32_t" | "uint32_t" => 4,
        "double" | "size_t" | "int64_t" | "uint64_t" => 8,
        "char" | "int8_t" | "uint8_t" => 1,
        "int16_t" | "uint16_t" => 2,
        _ => {
            let value = runtime.value(name)?;
            match value {
                Value::Buffer(buffer) => buffer.borrow().bytes.len(),
                Value::Number(_) => {
                    let cell = runtime
                        .cell(name)
                        .ok_or_else(|| invalid("source sizeof variable declaration missing"))?;
                    let kind = cell
                        .borrow()
                        .kind
                        .replace("constexpr", "")
                        .replace("const", "")
                        .replace("static", "");
                    Element::parse(&kind).map_err(invalid)?.bytes()
                }
                _ => {
                    return Err(invalid(
                        "source sizeof variable has no complete scalar-array type",
                    ));
                }
            }
        }
    };
    Ok(Value::Number(Number::U64(size as u64)))
}

fn text(value: &Value) -> Result<Vec<u8>> {
    match value {
        Value::Bytes(bytes) => Ok(bytes.clone()),
        Value::Pointer { value, index } => {
            let bytes = text(value)?;
            let bytes = bytes
                .get(*index..)
                .ok_or_else(|| invalid("source string pointer exceeds bytes"))?;
            let end = bytes
                .iter()
                .position(|&byte| byte == 0)
                .unwrap_or(bytes.len());
            Ok(bytes[..end].to_vec())
        }
        _ => Err(invalid("source string operation has no string bytes")),
    }
}

fn number_string(value: Number) -> String {
    match value {
        Number::F32(value) => format!("{value:.6}"),
        Number::F64(value) => format!("{value:.6}"),
        Number::I32(value) => value.to_string(),
        Number::U32(value) => value.to_string(),
        Number::I64(value) => value.to_string(),
        Number::U64(value) => value.to_string(),
    }
}

fn format_special(
    runtime: &mut Interpreter<'_>,
    value: &Value,
    kind: &str,
) -> Result<Option<String>> {
    let count = runtime.program.constants["GGML_MAX_OP_PARAMS"]
        .as_u64()
        .ok_or_else(|| invalid("source op-parameter diagnostic extent missing"))?
        / 4;
    let clean = kind.replace("const", "");
    if clean == format!("std::array<int32_t,{count}>") {
        let mut parts = Vec::new();
        for index in 0..value.len().map_err(invalid)? {
            let number = value
                .index(index)
                .map_err(invalid)?
                .number()
                .map_err(invalid)?
                .as_i64();
            if number != 0 {
                parts.push(format!("{index}:{number}"));
            }
        }
        return Ok(Some(format!("[{}]", parts.join(","))));
    }
    if clean == "std::vector<input_tensor>" {
        let mut parts = Vec::new();
        for index in 0..value.len().map_err(invalid)? {
            let input = value.index(index).map_err(invalid)?;
            let kind = member(&input, "type")?
                .ok_or_else(|| invalid("source input_tensor type missing"))?;
            let ne =
                member(&input, "ne")?.ok_or_else(|| invalid("source input_tensor ne missing"))?;
            let nb =
                member(&input, "nb")?.ok_or_else(|| invalid("source input_tensor nb missing"))?;
            let name = runtime.call("ggml_type_name", vec![kind])?;
            let mut text = String::from_utf8(text(&runtime.forced(name)?)?)?;
            text.push_str(&format_value(runtime, ne, "std::array<int64_t,4>")?);
            let noncontiguous = runtime
                .call("is_non_contiguous", vec![input])?
                .truth()
                .map_err(invalid)?;
            if noncontiguous {
                text.push_str("nb");
                text.push_str(&format_value(runtime, nb, "std::array<size_t,4>")?);
            }
            parts.push(text);
        }
        return Ok(Some(parts.join(",")));
    }

    Ok(None)
}

fn format_value(runtime: &mut Interpreter<'_>, value: Value, kind: &str) -> Result<String> {
    let value = runtime.forced(value)?;
    if let Some(formatted) = format_special(runtime, &value, kind)? {
        return Ok(formatted);
    }
    if kind.contains("std::array<") || matches!(value, Value::Buffer(_) | Value::Array(_)) {
        let mut parts = Vec::new();
        for index in 0..value.len().map_err(invalid)? {
            parts.push(format_value(
                runtime,
                value.index(index).map_err(invalid)?,
                "int64_t",
            )?);
        }
        return Ok(format!("[{}]", parts.join(",")));
    }
    let clean = kind.replace("const", "");
    if clean == "ggml_type" {
        let value = runtime.call("ggml_type_name", vec![value])?;
        return String::from_utf8(text(&runtime.forced(value)?)?).map_err(Into::into);
    }
    if clean == "ggml_prec" {
        return Ok(if value.number().map_err(invalid)?.as_i64()
            == runtime.program.constants["GGML_PREC_F32"]
                .as_i64()
                .ok_or_else(|| invalid("source precision constant differs"))?
        {
            "f32"
        } else {
            "def"
        }
        .into());
    }
    if clean == "ggml_op_pool" {
        let value = value.number().map_err(invalid)?.as_i64();
        for (name, text) in [("GGML_OP_POOL_AVG", "avg"), ("GGML_OP_POOL_MAX", "max")] {
            if runtime.program.constants[name].as_i64() == Some(value) {
                return Ok(text.into());
            }
        }
        return Ok(value.to_string());
    }
    if clean == "ggml_scale_mode" {
        let value = value.number().map_err(invalid)?.as_i64();
        let mut name = None;
        for (key, text) in [
            ("GGML_SCALE_MODE_NEAREST", "nearest"),
            ("GGML_SCALE_MODE_BILINEAR", "bilinear"),
            ("GGML_SCALE_MODE_BICUBIC", "bicubic"),
        ] {
            if runtime.program.constants[key].as_i64() == Some(value & 255) {
                name = Some(text.to_owned());
            }
        }
        let mut name = name.unwrap_or_else(|| value.to_string());
        for (key, suffix) in [
            ("GGML_SCALE_FLAG_ALIGN_CORNERS", "|align_corners"),
            ("GGML_SCALE_FLAG_ANTIALIAS", "|antialias"),
        ] {
            if value
                & runtime.program.constants[key]
                    .as_i64()
                    .ok_or_else(|| invalid("source scale flag differs"))?
                != 0
            {
                name.push_str(suffix);
            }
        }
        return Ok(name);
    }
    Ok(match value {
        Value::Bytes(bytes) => String::from_utf8(bytes)?,
        Value::Bool(value) => i32::from(value).to_string(),
        value => number_string(value.number().map_err(invalid)?),
    })
}

pub(super) fn vars(runtime: &mut Interpreter<'_>, arguments: &[Expression]) -> Result<Value> {
    let mut parts = Vec::new();
    for argument in arguments {
        let Expression::Id { name } = argument else {
            return Err(invalid("source VAR_TO_STR argument is not an identifier"));
        };
        let cell = runtime
            .cell(name)
            .ok_or_else(|| invalid("source diagnostic variable is missing"))?;
        let kind = cell.borrow().kind.clone();
        let value = runtime.eval(argument)?;
        parts.push(format!("{name}={}", format_value(runtime, value, &kind)?));
    }
    Ok(Value::Bytes(parts.join(",").into_bytes()))
}

pub(super) fn member(value: &Value, name: &str) -> Result<Option<Value>> {
    if let Value::Record(fields) = value {
        return Ok(fields.borrow().get(name).cloned());
    }
    if name == "first" || name == "second" {
        return value
            .index(usize::from(name == "second"))
            .map(Some)
            .map_err(invalid);
    }
    Ok(None)
}

fn pointer(value: &Value) -> Result<(&Value, usize)> {
    match value {
        Value::Pointer { value, index } => Ok((value, *index)),
        _ => Err(invalid("source iterator is not a pointer")),
    }
}
fn offset_pointer(value: &Value, offset: i64) -> Result<Value> {
    let (value, index) = pointer(value)?;
    let index = usize::try_from(index as i128 + i128::from(offset))?;
    if index > value.len().map_err(invalid)? {
        return Err(invalid("source pointer arithmetic exceeds storage"));
    }
    Ok(Value::Pointer {
        value: Box::new(value.clone()),
        index,
    })
}
fn same_storage(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Buffer(a), Value::Buffer(b)) => Rc::ptr_eq(a, b),
        (Value::Array(a), Value::Array(b)) => Rc::ptr_eq(a, b),
        (Value::Bytes(a), Value::Bytes(b)) => a == b,
        _ => false,
    }
}

fn numeric_array_equal(a: &Value, b: &Value) -> Result<bool> {
    let length = a.len().map_err(invalid)?;
    if length != b.len().map_err(invalid)? {
        return Ok(false);
    }
    for index in 0..length {
        let a = a.index(index).map_err(invalid)?.number().map_err(invalid)?;
        let b = b.index(index).map_err(invalid)?.number().map_err(invalid)?;
        if !a.equal(b) {
            return Ok(false);
        }
    }
    Ok(true)
}
pub(super) fn binary(op: &str, a: &Value, b: &Value) -> Result<Option<Value>> {
    if op == "+" && matches!((a, b), (Value::Bytes(_), Value::Bytes(_))) {
        let mut bytes = text(a)?;
        bytes.extend(text(b)?);
        return Ok(Some(Value::Bytes(bytes)));
    }
    if let (
        Value::Pointer {
            value: a,
            index: ai,
        },
        Value::Pointer {
            value: b,
            index: bi,
        },
    ) = (a, b)
        && same_storage(a, b)
    {
        let result = match op {
            "-" => Value::Number(Number::I64(*ai as i64 - *bi as i64)),
            "==" => Value::Bool(ai == bi),
            "!=" => Value::Bool(ai != bi),
            "<" => Value::Bool(ai < bi),
            "<=" => Value::Bool(ai <= bi),
            ">" => Value::Bool(ai > bi),
            ">=" => Value::Bool(ai >= bi),
            _ => return Err(invalid("source pointer operator differs")),
        };
        return Ok(Some(result));
    }
    if matches!(a, Value::Pointer { .. }) && matches!(op, "+" | "-") {
        let offset = b.number().map_err(invalid)?.as_i64();
        return offset_pointer(a, if op == "-" { -offset } else { offset }).map(Some);
    }
    if matches!(op, "==" | "!=") {
        let equal = match (a, b) {
            (Value::Null, Value::Null) => Some(true),
            (Value::Handle(a), Value::Handle(b)) => Some(a == b),
            (Value::Function(a), Value::Function(b)) => Some(a == b),
            (Value::Null, Value::Function(_) | Value::Handle(_))
            | (Value::Function(_) | Value::Handle(_), Value::Null) => Some(false),
            (Value::Bytes(a), Value::Bytes(b)) => Some(a == b),
            // Value containers compare elements; pointers keep their storage/offset identity above.
            (Value::Array(_) | Value::Buffer(_), Value::Array(_) | Value::Buffer(_)) => {
                Some(numeric_array_equal(a, b)?)
            }
            _ => None,
        };
        if let Some(equal) = equal {
            return Ok(Some(Value::Bool(if op == "==" { equal } else { !equal })));
        }
    }
    Ok(None)
}

fn arg(arguments: &[Value], index: usize) -> Result<&Value> {
    arguments
        .get(index)
        .ok_or_else(|| invalid("source library argument missing"))
}
fn numeric(arguments: &[Value], index: usize) -> Result<Number> {
    arg(arguments, index)?.number().map_err(invalid)
}
fn index(arguments: &[Value], index: usize) -> Result<usize> {
    numeric(arguments, index)?.index().map_err(invalid)
}

#[allow(
    clippy::too_many_lines,
    reason = "Container methods are an explicit finite source-language dispatch with shared error and ownership rules."
)]
pub(super) fn member_call(
    runtime: &mut Interpreter<'_>,
    object: Value,
    name: &str,
    arguments: Vec<Value>,
) -> Result<Value> {
    match name {
        "data" | "begin" | "c_str" => Ok(Value::Pointer {
            value: Box::new(object),
            index: 0,
        }),
        "end" => {
            let index = object.len().map_err(invalid)?;
            Ok(Value::Pointer {
                value: Box::new(object),
                index,
            })
        }
        "size" => Ok(Value::Number(Number::U64(
            object.len().map_err(invalid)? as u64
        ))),
        "empty" => Ok(Value::Bool(object.len().map_err(invalid)? == 0)),
        "back" => object
            .index(
                object
                    .len()
                    .map_err(invalid)?
                    .checked_sub(1)
                    .ok_or_else(|| invalid("source back on empty container"))?,
            )
            .map_err(invalid),
        "clear" => {
            match object {
                Value::Array(values) => values.borrow_mut().clear(),
                Value::Buffer(buffer) => buffer.borrow_mut().bytes.clear(),
                _ => return Err(invalid("source clear requires mutable container")),
            }
            Ok(Value::Null)
        }
        "push_back" => {
            let value = arg(&arguments, 0)?.clone();
            match object {
                Value::Array(values) => values.borrow_mut().push(value),
                Value::Buffer(buffer) => buffer
                    .borrow_mut()
                    .push(value.number().map_err(invalid)?)
                    .map_err(invalid)?,
                _ => return Err(invalid("source push_back requires mutable container")),
            }
            Ok(Value::Null)
        }
        "resize" => {
            let count = index(&arguments, 0)?;
            match object {
                Value::Array(values) => values.borrow_mut().resize(count, Value::Null),
                Value::Buffer(buffer) => {
                    let mut buffer = buffer.borrow_mut();
                    let width = buffer.element.bytes();
                    buffer.bytes.resize(
                        count
                            .checked_mul(width)
                            .ok_or_else(|| invalid("source resize byte overflow"))?,
                        0,
                    );
                }
                _ => return Err(invalid("source resize requires mutable container")),
            }
            Ok(Value::Null)
        }
        "substr" => {
            let bytes = text(&object)?;
            let start = index(&arguments, 0)?;
            let count = if arguments.len() > 1 {
                index(&arguments, 1)?
            } else {
                bytes.len()
            };
            let start = bytes
                .get(start..)
                .ok_or_else(|| invalid("source substr start exceeds string"))?;
            Ok(Value::Bytes(start[..count.min(start.len())].to_vec()))
        }
        "find" | "rfind" => {
            let bytes = text(&object)?;
            let needle = match arg(&arguments, 0)? {
                Value::Number(number) => vec![number.as_u64() as u8],
                value => text(value)?,
            };
            let position = arguments
                .get(1)
                .map(|_| index(&arguments, 1))
                .transpose()?
                .unwrap_or(if name == "find" { 0 } else { usize::MAX });
            let found = if needle.is_empty() {
                Some(position.min(bytes.len()))
            } else if name == "find" {
                bytes
                    .windows(needle.len())
                    .enumerate()
                    .skip(position)
                    .find(|(_, value)| *value == needle)
                    .map(|(index, _)| index)
            } else {
                bytes
                    .windows(needle.len())
                    .enumerate()
                    .take(position.saturating_add(1))
                    .filter(|(_, value)| *value == needle)
                    .map(|(index, _)| index)
                    .next_back()
            };
            Ok(Value::Number(Number::U64(
                found.map_or(u64::MAX, |index| index as u64),
            )))
        }
        _ => runtime
            .host
            .call(
                &format!("member:{name}"),
                &[vec![object], arguments].concat(),
            )?
            .ok_or_else(|| {
                invalid(format!(
                    "source container member {name} has no implementation"
                ))
            }),
    }
}

pub(super) fn call_value(
    runtime: &mut Interpreter<'_>,
    function: Value,
    arguments: &[Value],
) -> Result<Value> {
    match function {
        Value::RandomDevice => runtime
            .host
            .call("random-device", &[])?
            .ok_or_else(|| invalid("source entropy ledger unavailable")),
        Value::Engine(engine) => Ok(Value::Number(Number::U32(engine.borrow_mut().next()))),
        Value::Distribution {
            minimum,
            maximum,
            real,
        } => {
            let Value::Engine(engine) = arg(arguments, 0)? else {
                return Err(invalid("source distribution has no MT19937 engine"));
            };
            let mut engine = engine.borrow_mut();
            Ok(Value::Number(if real {
                Number::F32(engine.uniform_real(minimum.as_f32(), maximum.as_f32()))
            } else {
                Number::I32(
                    engine
                        .uniform_integer(minimum.as_i64(), maximum.as_i64())
                        .map_err(invalid)? as i32,
                )
            }))
        }
        _ => Err(invalid("source expression is not callable")),
    }
}

fn vector(name: &str, arguments: &[Value]) -> Result<Value> {
    let element = name
        .strip_prefix("std::vector<")
        .and_then(|name| name.strip_suffix('>'))
        .ok_or_else(|| invalid("source vector constructor type differs"))?;
    let count = if arguments.is_empty() {
        0
    } else {
        index(arguments, 0)?
    };
    if let Ok(element) = Element::parse(element) {
        let mut buffer = Buffer::new(element, count).map_err(invalid)?;
        if arguments.len() > 1 {
            let fill = numeric(arguments, 1)?;
            for index in 0..count {
                buffer.set(index, fill).map_err(invalid)?;
            }
        }
        Ok(Value::Buffer(Rc::new(RefCell::new(buffer))))
    } else {
        Ok(Value::array(vec![
            arguments
                .get(1)
                .cloned()
                .unwrap_or(Value::Null);
            count
        ]))
    }
}

fn order(a: &Value, b: &Value) -> Result<Ordering> {
    let a = a.number().map_err(invalid)?;
    let b = b.number().map_err(invalid)?;
    Ok(if a.less(b) {
        Ordering::Less
    } else if b.less(a) {
        Ordering::Greater
    } else {
        Ordering::Equal
    })
}
fn iterator_range(arguments: &[Value]) -> Result<(&Value, usize, usize)> {
    let (value, start) = pointer(arg(arguments, 0)?)?;
    let (other, end) = pointer(arg(arguments, 1)?)?;
    if !same_storage(value, other) || start > end || end > value.len().map_err(invalid)? {
        return Err(invalid("source iterator range differs"));
    }
    Ok((value, start, end))
}

fn iterator_call(name: &str, arguments: &[Value]) -> Result<Value> {
    let (value, start, end) = iterator_range(arguments)?;
    match name {
        "std::copy" => {
            let destination = arg(arguments, 2)?;
            for index in start..end {
                destination
                    .place(index - start)
                    .map_err(invalid)?
                    .set(value.index(index).map_err(invalid)?)
                    .map_err(invalid)?;
            }
            offset_pointer(destination, (end - start) as i64)
        }
        "std::adjacent_find" => {
            for index in start..end.saturating_sub(1) {
                if value
                    .index(index)
                    .map_err(invalid)?
                    .number()
                    .map_err(invalid)?
                    .equal(
                        value
                            .index(index + 1)
                            .map_err(invalid)?
                            .number()
                            .map_err(invalid)?,
                    )
                {
                    return Ok(Value::Pointer {
                        value: Box::new(value.clone()),
                        index,
                    });
                }
            }
            Ok(Value::Pointer {
                value: Box::new(value.clone()),
                index: end,
            })
        }
        "std::shuffle" => {
            let Value::Engine(engine) = arg(arguments, 2)? else {
                return Err(invalid("source shuffle lacks engine"));
            };
            for index in start + 1..end {
                let target = start
                    + engine
                        .borrow_mut()
                        .uniform_below((index - start + 1) as u64)
                        .map_err(invalid)? as usize;
                let a = value.index(index).map_err(invalid)?;
                let b = value.index(target).map_err(invalid)?;
                value
                    .place(index)
                    .map_err(invalid)?
                    .set(b)
                    .map_err(invalid)?;
                value
                    .place(target)
                    .map_err(invalid)?
                    .set(a)
                    .map_err(invalid)?;
            }
            Ok(Value::Null)
        }
        "std::sort" => {
            let mut values = (start..end)
                .map(|index| value.index(index).map_err(invalid))
                .collect::<Result<Vec<_>>>()?;
            for entry in &values {
                let _ = entry.number().map_err(invalid)?;
            }
            values.sort_unstable_by(|a, b| order(a, b).unwrap_or(Ordering::Equal));
            for (offset, replacement) in values.into_iter().enumerate() {
                value
                    .place(start + offset)
                    .map_err(invalid)?
                    .set(replacement)
                    .map_err(invalid)?;
            }
            Ok(Value::Null)
        }
        _ => Err(invalid("unknown source iterator operation")),
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "The finite library dispatch records every source library operation in one reviewable match."
)]
pub(super) fn call(
    runtime: &mut Interpreter<'_>,
    name: &str,
    arguments: &[Value],
) -> Result<Option<Value>> {
    if name.starts_with("std::vector<") {
        return vector(name, arguments).map(Some);
    }
    let value = match name {
        "GGML_UNUSED" | "printf" => Value::Null,
        "GGML_ASSERT" => {
            let _ = runtime.host.call("source-assert", arguments)?;
            if !arg(arguments, 0)?.truth().map_err(invalid)? {
                return Err(invalid(format!(
                    "original source assertion failed in {}",
                    runtime.class.name
                )));
            }
            Value::Null
        }
        "GGML_ABORT" => return Err(invalid("original source abort branch reached")),
        "std::move" => arg(arguments, 0)?.clone(),
        "std::default_random_engine" | "std::mt19937" => Value::Engine(Rc::new(RefCell::new(
            Mt19937::new(if arguments.is_empty() {
                5489
            } else {
                numeric(arguments, 0)?.as_u64() as u32
            }),
        ))),
        "std::uniform_real_distribution<float>" | "std::uniform_int_distribution<int32_t>" => {
            Value::Distribution {
                minimum: numeric(arguments, 0)?,
                maximum: numeric(arguments, 1)?,
                real: name.contains("real"),
            }
        }
        "float" => Value::Number(numeric(arguments, 0)?.cast(Kind::F32)),
        "std::string" => Value::Bytes(if arguments.is_empty() {
            Vec::new()
        } else {
            text(arg(arguments, 0)?)?
        }),
        "std::to_string" => Value::Bytes(number_string(numeric(arguments, 0)?).into_bytes()),
        "std::stoi" | "std::stoll" => {
            let text = String::from_utf8(text(arg(arguments, 0)?)?)?;
            let trimmed = text.trim_start();
            let end = trimmed
                .char_indices()
                .skip(usize::from(trimmed.starts_with(['+', '-'])))
                .find(|(_, character)| !character.is_ascii_digit())
                .map_or(trimmed.len(), |(index, _)| index);
            let value = trimmed[..end].parse::<i64>()?;
            Value::Number(if name == "std::stoi" {
                Number::I32(i32::try_from(value)?)
            } else {
                Number::I64(value)
            })
        }
        "std::min" | "std::max" => {
            let a = numeric(arguments, 0)?;
            let b = numeric(arguments, 1)?;
            Value::Number(
                if if name == "std::min" {
                    b.less(a)
                } else {
                    a.less(b)
                } {
                    b
                } else {
                    a
                },
            )
        }
        "sqrtf" => Value::Number(Number::F32(numeric(arguments, 0)?.as_f32().sqrt())),
        "fabsf" => Value::Number(Number::F32(numeric(arguments, 0)?.as_f32().abs())),
        "std::fabs" => {
            let value = numeric(arguments, 0)?;
            Value::Number(match value {
                Number::F32(value) => Number::F32(value.abs()),
                _ => Number::F64(value.as_f64().abs()),
            })
        }
        "pow" => Value::Number(Number::F64(
            numeric(arguments, 0)?
                .as_f64()
                .powf(numeric(arguments, 1)?.as_f64()),
        )),
        "strcmp" | "strncmp" => {
            let a = text(arg(arguments, 0)?)?;
            let b = text(arg(arguments, 1)?)?;
            let count = if name == "strncmp" {
                index(arguments, 2)?
            } else {
                usize::MAX
            };
            let mut result = 0;
            for index in 0..count.min(a.len().max(b.len()) + 1) {
                let a = a.get(index).copied().unwrap_or(0);
                let b = b.get(index).copied().unwrap_or(0);
                if a != b {
                    result = i32::from(a) - i32::from(b);
                    break;
                }
                if a == 0 {
                    break;
                }
            }
            Value::Number(Number::I32(result))
        }
        "memcmp" => {
            let count = index(arguments, 2)?;
            let a = arg(arguments, 0)?.bytes(count).map_err(invalid)?;
            let b = arg(arguments, 1)?.bytes(count).map_err(invalid)?;
            let value = a
                .iter()
                .zip(b)
                .find(|(a, b)| **a != *b)
                .map_or(0, |(a, b)| i32::from(*a) - i32::from(b));
            Value::Number(Number::I32(value))
        }
        "std::sort" | "std::shuffle" | "std::copy" | "std::adjacent_find" => {
            iterator_call(name, arguments)?
        }
        "is_pow2" => {
            let value = numeric(arguments, 0)?.as_u64();
            Value::Bool(value.is_power_of_two())
        }
        "arclab_popcount64" => Value::Number(Number::I32(
            numeric(arguments, 0)?.as_u64().count_ones() as i32,
        )),
        "GGML_PAD" => {
            let value = numeric(arguments, 0)?;
            let multiple = numeric(arguments, 1)?;
            Value::Number(
                value
                    .binary("+", multiple)
                    .map_err(invalid)?
                    .binary("-", Number::I32(1))
                    .map_err(invalid)?
                    .binary(
                        "&",
                        multiple
                            .binary("-", Number::I32(1))
                            .map_err(invalid)?
                            .complement()
                            .map_err(invalid)?,
                    )
                    .map_err(invalid)?,
            )
        }
        "calc_conv_output_size" => {
            let input = numeric(arguments, 0)?.as_f64();
            let kernel = numeric(arguments, 1)?.as_f64();
            let stride = numeric(arguments, 2)?.as_f64();
            let padding = numeric(arguments, 3)?.as_f64();
            let dilation = numeric(arguments, 4)?.as_f64();
            Value::Number(Number::I64(
                ((input + 2.0 * padding - dilation * (kernel - 1.0) - 1.0) / stride).floor() as i64
                    + 1,
            ))
        }
        _ => return Ok(None),
    };
    Ok(Some(value))
}

#[cfg(test)]
mod tests {
    use super::super::interpreter::Host;
    use super::super::value::Handle;
    use super::*;

    struct LanguageOnly;
    impl Host for LanguageOnly {
        fn call(&mut self, _name: &str, _arguments: &[Value]) -> Result<Option<Value>> {
            Err(invalid(
                "pointer-container guard must not invoke production",
            ))
        }
        fn member(&mut self, _value: &Value, _name: &str) -> Result<Option<Value>> {
            Err(invalid(
                "pointer-container guard must not observe production",
            ))
        }
        fn write(
            &mut self,
            _value: &Value,
            _name: &str,
            _index: Option<usize>,
            _replacement: Value,
        ) -> Result<()> {
            Err(invalid(
                "pointer-container guard must not mutate production",
            ))
        }
        fn force(&mut self, value: Value) -> Result<Value> {
            Ok(value)
        }
    }

    #[test]
    fn pointer_container_defaults_preserve_real_source_member_operations() -> Result<()> {
        let (program, geometries) = super::super::program::load().map_err(invalid)?;
        let geometry = geometries
            .eval
            .iter()
            .find(|geometry| geometry.class == "test_arclab_t2_multi")
            .ok_or_else(|| invalid("original T2 multi geometry missing"))?;
        let mut host = LanguageOnly;
        let mut runtime = Interpreter::new(program, &geometry.class, &mut host)?;
        runtime.instantiate(geometry)?;
        let outs = runtime.value("outs")?;
        assert_eq!(outs.len().map_err(invalid)?, 0);
        member_call(&mut runtime, outs.clone(), "clear", Vec::new())?;
        member_call(
            &mut runtime,
            outs,
            "push_back",
            vec![Value::Handle(Handle::Tensor(7))],
        )?;
        let outs = runtime.value("outs")?;
        assert_eq!(outs.len().map_err(invalid)?, 1);
        assert!(matches!(
            outs.index(0).map_err(invalid)?,
            Value::Handle(Handle::Tensor(7))
        ));
        let array = zero("std::array<ggml_tensor*,2>")
            .coerce("std::array<ggml_tensor*,2>")
            .map_err(invalid)?;
        assert_eq!(array.len().map_err(invalid)?, 2);
        assert!(matches!(array.index(1).map_err(invalid)?, Value::Null));
        assert_eq!(
            zero("conststd::vector<ggml_tensor*>")
                .len()
                .map_err(invalid)?,
            0
        );
        for kind in [
            "ggml_tensor*",
            "ggml_tensor**",
            "std::vector<ggml_tensor*>*",
            "std::array<ggml_tensor*,2>*const",
        ] {
            assert!(matches!(zero(kind), Value::Null));
        }
        Ok(())
    }
    #[test]
    fn named_source_scalar_sizeof_uses_declared_width() -> Result<()> {
        let (program, _) = super::super::program::load().map_err(invalid)?;
        let mut host = LanguageOnly;
        let mut runtime = Interpreter::new(program, "test_gdn_replay_chain", &mut host)?;
        let argument = [Expression::Id { name: "x".into() }];
        for (kind, value, expected) in [
            ("constint32_t", Number::I32(3), 4),
            ("constint64_t", Number::I64(3), 8),
            ("char", Number::I32(3), 1),
        ] {
            runtime.scopes[0].insert(
                "x".into(),
                super::super::value::variable(kind.into(), Value::Number(value)),
            );
            assert_eq!(
                size_of(&runtime, &argument)?.number().map_err(invalid)?,
                Number::U64(expected)
            );
        }
        let array = Value::array(vec![Value::Number(Number::I32(3)); 2])
            .coerce("std::array<int32_t,2>")
            .map_err(invalid)?;
        runtime.scopes[0].insert(
            "x".into(),
            super::super::value::variable("std::array<int32_t,2>".into(), array),
        );
        assert_eq!(
            size_of(&runtime, &argument)?.number().map_err(invalid)?,
            Number::U64(8)
        );
        Ok(())
    }
    #[test]
    fn source_function_identity_controls_original_bin_bcast_methods() -> Result<()> {
        let (program, geometries) = super::super::program::load().map_err(invalid)?;
        let names = [
            "ggml_add",
            "ggml_sub",
            "ggml_mul",
            "ggml_div",
            "ggml_add_inplace",
            "ggml_sub_inplace",
            "ggml_mul_inplace",
            "ggml_div_inplace",
        ];
        for name in names {
            let geometry = geometries
                .eval
                .iter()
                .find(|geometry| {
                    geometry.class == "test_bin_bcast"
                        && geometry
                            .arguments
                            .first()
                            .and_then(|value| value.get("Function"))
                            .and_then(Json::as_str)
                            == Some(name)
                })
                .ok_or_else(|| invalid("actual binary-broadcast function geometry missing"))?;
            let mut host = LanguageOnly;
            let mut runtime = Interpreter::new(program, &geometry.class, &mut host)?;
            runtime.instantiate(geometry)?;
            assert_eq!(
                runtime
                    .call("grad_precise", Vec::new())?
                    .truth()
                    .map_err(invalid)?,
                name == "ggml_div"
            );
            assert_eq!(
                runtime
                    .call("max_maa_err", Vec::new())?
                    .number()
                    .map_err(invalid)?,
                Number::F64(if name == "ggml_add" { 1e-4 } else { 1e-3 })
            );
            let function = runtime.value("op")?;
            for other in names {
                for (op, expected) in [("==", name == other), ("!=", name != other)] {
                    assert_eq!(
                        runtime
                            .binary_value(op, function.clone(), Value::Function(other.into()))?
                            .truth()
                            .map_err(invalid)?,
                        expected
                    );
                }
            }
            for (left, right) in [(function.clone(), Value::Null), (Value::Null, function)] {
                assert!(
                    !runtime
                        .binary_value("==", left.clone(), right.clone())?
                        .truth()
                        .map_err(invalid)?
                );
                assert!(
                    runtime
                        .binary_value("!=", left, right)?
                        .truth()
                        .map_err(invalid)?
                );
            }
        }
        assert!(
            binary(
                "<",
                &Value::Function("ggml_add".into()),
                &Value::Function("ggml_sub".into())
            )?
            .is_none()
        );
        Ok(())
    }
    #[test]
    fn original_flash_attention_array_comparison_preserves_value_and_pointer_semantics()
    -> Result<()> {
        let (program, geometries) = super::super::program::load().map_err(invalid)?;
        let geometry = geometries
            .eval
            .iter()
            .find(|geometry| geometry.class == "test_flash_attn_ext")
            .ok_or_else(|| invalid("actual flash-attention geometry missing"))?;
        let mut host = LanguageOnly;
        let mut runtime = Interpreter::new(program, &geometry.class, &mut host)?;
        runtime.instantiate(geometry)?;
        let identity = Value::array(
            (0..4)
                .map(|value| Value::Number(Number::I32(value)))
                .collect(),
        );
        let permute = runtime.value("permute")?;
        assert!(
            !runtime
                .binary_value("!=", permute.clone(), identity.clone())?
                .truth()
                .map_err(invalid)?
        );
        assert!(
            runtime
                .binary_value("==", identity.clone(), permute.clone())?
                .truth()
                .map_err(invalid)?
        );
        let changed = permute.copy_value();
        changed
            .place(1)
            .map_err(invalid)?
            .set(Value::Number(Number::I32(2)))
            .map_err(invalid)?;
        changed
            .place(2)
            .map_err(invalid)?
            .set(Value::Number(Number::I32(1)))
            .map_err(invalid)?;
        assert!(
            runtime
                .binary_value("!=", changed, identity)?
                .truth()
                .map_err(invalid)?
        );
        let shorter = Value::array(vec![Value::Number(Number::I32(0))]);
        assert!(
            !runtime
                .binary_value("==", permute.clone(), shorter)?
                .truth()
                .map_err(invalid)?
        );
        let floats = |values: &[f32]| {
            Value::array(
                values
                    .iter()
                    .copied()
                    .map(|value| Value::Number(Number::F32(value)))
                    .collect(),
            )
        };
        let zero = floats(&[-0.0])
            .coerce("std::array<float,1>")
            .map_err(invalid)?;
        assert!(
            runtime
                .binary_value("==", zero, floats(&[0.0]))?
                .truth()
                .map_err(invalid)?
        );
        let nan = floats(&[f32::NAN])
            .coerce("std::array<float,1>")
            .map_err(invalid)?;
        assert!(
            !runtime
                .binary_value("==", nan, floats(&[f32::NAN]))?
                .truth()
                .map_err(invalid)?
        );
        array_pointer_identity(&mut runtime, &permute)
    }

    fn array_pointer_identity(runtime: &mut Interpreter<'_>, permute: &Value) -> Result<()> {
        let first = Value::Pointer {
            value: Box::new(Value::clone(permute)),
            index: 0,
        };
        let next = Value::Pointer {
            value: Box::new(Value::clone(permute)),
            index: 1,
        };
        assert!(
            !runtime
                .binary_value("==", first.clone(), next)?
                .truth()
                .map_err(invalid)?
        );
        assert!(
            runtime
                .binary_value("==", first.clone(), first.clone())?
                .truth()
                .map_err(invalid)?
        );
        let separate = Value::Pointer {
            value: Box::new(permute.copy_value()),
            index: 0,
        };
        assert!(binary("==", &first, &separate)?.is_none());
        assert!(binary("<", permute, permute)?.is_none());
        Ok(())
    }
}
