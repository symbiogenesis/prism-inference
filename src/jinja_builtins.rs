//! Jinja value operations and callable builtins.
use super::{Kind, Part, Runtime, Val, value};
use crate::{Result, invalid};
use std::collections::BTreeMap;
pub(super) fn is_global(name: &str) -> bool {
    matches!(
        name,
        "range" | "namespace" | "raise_exception" | "tojson" | "strftime_now"
    ) || name.starts_with("test_is_")
}
pub(super) fn has_method(value: &Val, name: &str) -> bool {
    if value.0.borrow().namespace {
        return false;
    }
    match value.kind() {
        Kind::String(_) => matches!(
            name,
            "default"
                | "upper"
                | "lower"
                | "strip"
                | "rstrip"
                | "lstrip"
                | "title"
                | "capitalize"
                | "length"
                | "startswith"
                | "endswith"
                | "split"
                | "rsplit"
                | "replace"
                | "format"
                | "int"
                | "float"
                | "string"
                | "slice"
                | "safe"
                | "tojson"
                | "indent"
                | "join"
        ),
        Kind::Array(_) | Kind::Tuple(_) => matches!(
            name,
            "default"
                | "list"
                | "first"
                | "last"
                | "length"
                | "count"
                | "reverse"
                | "sort"
                | "join"
                | "pop"
                | "append"
                | "map"
                | "select"
                | "reject"
                | "selectattr"
                | "rejectattr"
                | "min"
                | "max"
                | "sum"
                | "unique"
                | "safe"
                | "string"
                | "tojson"
        ),
        Kind::Object(_) => matches!(
            name,
            "get"
                | "items"
                | "keys"
                | "values"
                | "dictsort"
                | "length"
                | "join"
                | "string"
                | "tojson"
        ),
        Kind::Int(_) | Kind::Float(_) | Kind::Bool(_) => matches!(
            name,
            "default" | "abs" | "int" | "float" | "safe" | "string" | "tojson"
        ),
        _ => false,
    }
}
fn arg(pos: &[Val], kw: &BTreeMap<String, Val>, name: &str, index: usize, default: Val) -> Val {
    kw.get(name)
        .or_else(|| pos.get(index))
        .cloned()
        .unwrap_or(default)
}
fn text_arg(
    pos: &[Val],
    kw: &BTreeMap<String, Val>,
    name: &str,
    index: usize,
    default: &[u8],
) -> Vec<u8> {
    arg(pos, kw, name, index, Val::bytes(default.to_vec(), false)).text()
}
fn find(bytes: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        Some(0)
    } else {
        bytes.windows(needle.len()).position(|part| part == needle)
    }
}
fn transformed(value: &Val, bytes: Vec<u8>) -> Val {
    Val::bytes(
        bytes,
        if let Kind::String(parts) = value.kind() {
            parts.iter().all(|part| part.input)
        } else {
            false
        },
    )
}
fn attribute(value: &Val, key: &Val) -> Val {
    if let Kind::String(_) = key.kind() {
        let mut result = value.clone();
        for name in key.text().split(|&byte| byte == b'.') {
            let name = String::from_utf8_lossy(name);
            let key = name
                .parse::<i64>()
                .map_or_else(|_| Val::string(name.into_owned()), Val::int);
            result = result.get(&key, true);
        }
        result
    } else {
        value.get(key, true)
    }
}
impl Runtime {
    pub(super) fn method(
        &mut self,
        value: Val,
        name: &str,
        mut pos: Vec<Val>,
        kw: BTreeMap<String, Val>,
    ) -> Result<Val> {
        pos.insert(0, value);
        self.builtin(name, pos, kw)
    }
    #[allow(
        clippy::too_many_lines,
        clippy::needless_pass_by_value,
        reason = "Keep source callable dispatch together with owned evaluated argument buffers"
    )]
    pub(super) fn builtin(
        &mut self,
        name: &str,
        pos: Vec<Val>,
        kw: BTreeMap<String, Val>,
    ) -> Result<Val> {
        self.tick()?;
        if let Some(name) = name.strip_prefix("test_is_") {
            let value = pos.first().cloned().unwrap_or_else(Val::undefined);
            return Ok(Val::boolean(self.test(
                &value,
                name,
                pos.get(1..).unwrap_or_default(),
            )?));
        }
        let value = pos.first().cloned().unwrap_or_else(Val::undefined);
        match name {
            "raise_exception" => Err(invalid(format!(
                "Jinja Exception: {}",
                String::from_utf8_lossy(&value.text())
            ))),
            "namespace" => {
                if !pos.is_empty() {
                    return Err(invalid("namespace accepts only keyword arguments"));
                }
                {
                    let namespace = Val::new(Kind::Object(
                        kw.into_iter()
                            .map(|(key, value)| (Val::string(key), value))
                            .collect(),
                    ));
                    namespace.0.borrow_mut().namespace = true;
                    Ok(namespace)
                }
            }
            "range" => {
                if pos.is_empty() || pos.len() > 3 || !kw.is_empty() {
                    return Err(invalid("Jinja range requires one to three arguments"));
                }
                let (start, stop) = if pos.len() == 1 {
                    (0, value.integer()?)
                } else {
                    (value.integer()?, pos[1].integer()?)
                };
                let step = if let Some(step) = pos.get(2) {
                    step.integer()?
                } else {
                    1
                };
                if step == 0 {
                    return Err(invalid("Jinja range step cannot be zero"));
                }
                let mut values = Vec::new();
                let mut current = start;
                while if step > 0 {
                    current < stop
                } else {
                    current > stop
                } {
                    if values.len() >= 65536 {
                        return Err(invalid("Jinja range bound exceeded"));
                    }
                    values.push(Val::int(current));
                    let Some(next) = current.checked_add(step) else {
                        break;
                    };
                    current = next;
                }
                Ok(Val::new(Kind::Array(values)))
            }
            "default" | "d" => {
                let fallback = arg(&pos, &kw, "default_value", 1, Val::string(""));
                let boolean = arg(&pos, &kw, "boolean", 2, Val::boolean(false)).truth();
                Ok(
                    if matches!(value.kind(), Kind::Undefined) || boolean && !value.truth() {
                        fallback
                    } else {
                        value
                    },
                )
            }
            "length" | "count" => Ok(Val::int(i64::try_from(value.len()?)?)),
            "string" | "safe" => {
                if matches!(value.kind(), Kind::String(_)) {
                    Ok(value)
                } else {
                    Ok(Val::bytes(value.text(), false))
                }
            }
            "upper" | "lower" | "capitalize" | "title" => {
                let mut bytes = value.text();
                match name {
                    "upper" => bytes.make_ascii_uppercase(),
                    "lower" => bytes.make_ascii_lowercase(),
                    "capitalize" => {
                        bytes.make_ascii_lowercase();
                        if let Some(first) = bytes.first_mut() {
                            first.make_ascii_uppercase();
                        }
                    }
                    _ => {
                        let mut previous = false;
                        for byte in &mut bytes {
                            let alpha = byte.is_ascii_alphabetic();
                            if alpha {
                                if previous {
                                    byte.make_ascii_lowercase();
                                } else {
                                    byte.make_ascii_uppercase();
                                }
                            }
                            previous = alpha;
                        }
                    }
                }
                Ok(transformed(&value, bytes))
            }
            "strip" | "trim" | "lstrip" | "rstrip" => {
                let chars = text_arg(&pos, &kw, "chars", 1, b" \t\n\r\x0b\x0c");
                let bytes = value.text();
                let mut start = 0;
                let mut stop = bytes.len();
                if name != "rstrip" {
                    while start < stop && chars.contains(&bytes[start]) {
                        start += 1;
                    }
                }
                if name != "lstrip" {
                    while stop > start && chars.contains(&bytes[stop - 1]) {
                        stop -= 1;
                    }
                }
                Ok(transformed(&value, bytes[start..stop].to_vec()))
            }
            "startswith" | "endswith" => {
                let needle = text_arg(&pos, &kw, "prefix", 1, b"");
                Ok(Val::boolean(if name == "startswith" {
                    value.text().starts_with(&needle)
                } else {
                    value.text().ends_with(&needle)
                }))
            }
            "replace" => {
                let bytes = value.text();
                let old = text_arg(&pos, &kw, "old", 1, b"");
                let new = text_arg(&pos, &kw, "new", 2, b"");
                let count = arg(&pos, &kw, "count", 3, Val::int(-1)).integer()?;
                let mut remaining = if count < 0 {
                    usize::MAX
                } else {
                    usize::try_from(count).unwrap_or(usize::MAX)
                };
                let mut result = Vec::new();
                let mut offset = 0;
                if old.is_empty() {
                    if remaining != 0 {
                        result.extend(&new);
                        remaining -= 1;
                    }
                    for byte in bytes {
                        result.push(byte);
                        if remaining != 0 {
                            result.extend(&new);
                            remaining -= 1;
                        }
                        if result.len() > 4 * 1024 * 1024 {
                            return Err(invalid("Jinja replace output bound exceeded"));
                        }
                    }
                } else {
                    while remaining != 0 {
                        let Some(index) = find(&bytes[offset..], &old) else {
                            break;
                        };
                        result.extend(&bytes[offset..offset + index]);
                        result.extend(&new);
                        offset += index + old.len();
                        remaining -= 1;
                        if result.len() > 4 * 1024 * 1024 {
                            return Err(invalid("Jinja replace output bound exceeded"));
                        }
                    }
                    result.extend(&bytes[offset..]);
                }
                Ok(transformed(&value, result))
            }
            "split" | "rsplit" => {
                let bytes = value.text();
                let delimiter = text_arg(&pos, &kw, "sep", 1, b" ");
                if delimiter.is_empty() {
                    return Err(invalid("Jinja empty split separator"));
                }
                let max = arg(&pos, &kw, "maxsplit", 2, Val::int(-1)).integer()?;
                let mut remaining = if max < 0 {
                    usize::MAX
                } else {
                    usize::try_from(max).unwrap_or(usize::MAX)
                };
                let mut values = Vec::new();
                if name == "split" {
                    let mut offset = 0;
                    while remaining != 0 {
                        let Some(index) = find(&bytes[offset..], &delimiter) else {
                            break;
                        };
                        values.push(Val::bytes(bytes[offset..offset + index].to_vec(), false));
                        offset += index + delimiter.len();
                        remaining -= 1;
                    }
                    values.push(transformed(&value, bytes[offset..].to_vec()));
                } else {
                    let mut stop = bytes.len();
                    while remaining != 0 {
                        let Some(index) = bytes[..stop]
                            .windows(delimiter.len())
                            .rposition(|part| part == delimiter)
                        else {
                            break;
                        };
                        values.push(Val::bytes(
                            bytes[index + delimiter.len()..stop].to_vec(),
                            false,
                        ));
                        stop = index;
                        remaining -= 1;
                    }
                    values.push(transformed(&value, bytes[..stop].to_vec()));
                    values.reverse();
                }
                Ok(Val::new(Kind::Array(values)))
            }
            "format" => {
                let bytes = value.text();
                let mut result = Vec::new();
                let mut literal = Vec::new();
                let input =
                    matches!(value.kind(),Kind::String(parts)if parts.iter().all(|part|part.input));
                let mut index = 0;
                let mut automatic = 1;
                while index < bytes.len() {
                    if bytes[index] == b'{' {
                        if bytes.get(index + 1) == Some(&b'{') {
                            literal.push(b'{');
                            index += 2;
                            continue;
                        }
                        if !literal.is_empty() {
                            result.push(Part {
                                bytes: std::mem::take(&mut literal),
                                input,
                            });
                        }
                        let end = bytes[index + 1..]
                            .iter()
                            .position(|&byte| byte == b'}')
                            .map(|offset| index + 1 + offset)
                            .ok_or_else(|| invalid("Jinja unclosed format placeholder"))?;
                        let key = String::from_utf8(bytes[index + 1..end].to_vec())?;
                        let argument = if key.is_empty() {
                            let argument = pos.get(automatic).cloned();
                            automatic += 1;
                            argument
                        } else if let Ok(number) = key.parse::<usize>() {
                            pos.get(number + 1).cloned()
                        } else {
                            kw.get(&key).cloned()
                        }
                        .ok_or_else(|| invalid("Jinja missing format argument"))?;
                        let parts = argument.parts();
                        let size = result
                            .iter()
                            .map(|part: &Part| part.bytes.len())
                            .sum::<usize>()
                            + parts.iter().map(|part| part.bytes.len()).sum::<usize>();
                        if size > 4 * 1024 * 1024 {
                            return Err(invalid("Jinja format output bound exceeded"));
                        }
                        result.extend(parts);
                        index = end + 1;
                    } else if bytes[index] == b'}' {
                        if bytes.get(index + 1) != Some(&b'}') {
                            return Err(invalid("Jinja unmatched format brace"));
                        }
                        literal.push(b'}');
                        index += 2;
                    } else {
                        literal.push(bytes[index]);
                        index += 1;
                    }
                }
                if !literal.is_empty() {
                    result.push(Part {
                        bytes: literal,
                        input,
                    });
                }
                Ok(Val::new(Kind::String(result)))
            }
            "int" => {
                if matches!(value.kind(), Kind::Undefined) {
                    return Ok(Val::int(0));
                }
                if !matches!(value.kind(), Kind::String(_)) {
                    return Ok(Val::int(value.integer()?));
                }
                let fallback = arg(&pos, &kw, "default", 1, Val::int(0));
                let base = arg(&pos, &kw, "base", 2, Val::int(10)).integer()?;
                if base != 0 && !(2..=36).contains(&base) {
                    return Err(invalid(
                        "Jinja integer base must be zero or between 2 and 36",
                    ));
                }
                let bytes = value.text();
                let text = String::from_utf8_lossy(&bytes);
                let mut text = text.trim_start();
                let negative = text.starts_with('-');
                if text.starts_with(['-', '+']) {
                    text = &text[1..];
                }
                let base = if base == 0 {
                    if text.starts_with("0x") {
                        16
                    } else if text.starts_with('0') {
                        8
                    } else {
                        10
                    }
                } else {
                    u32::try_from(base)?
                };
                if base == 16 && (text.starts_with("0x") || text.starts_with("0X")) {
                    text = &text[2..];
                }
                let count = text.chars().take_while(|ch| ch.is_digit(base)).count();
                let result = i64::from_str_radix(&text[..count], base)
                    .ok()
                    .and_then(|value| {
                        if negative {
                            value.checked_neg()
                        } else {
                            Some(value)
                        }
                    });
                Ok(result.map_or(fallback, Val::int))
            }
            "float" => {
                if !matches!(value.kind(), Kind::String(_)) {
                    return Ok(Val::new(Kind::Float(value.number()?)));
                }
                let text = String::from_utf8(value.text())?;
                let result = text
                    .trim()
                    .parse::<f64>()
                    .ok()
                    .filter(|number| number.is_finite());
                Ok(result.map_or_else(
                    || arg(&pos, &kw, "default", 1, Val::new(Kind::Float(0.0))),
                    |number| Val::new(Kind::Float(number)),
                ))
            }
            "abs" => {
                if let Kind::Int(number) = value.kind() {
                    Ok(Val::int(number.checked_abs().ok_or_else(|| {
                        invalid("Jinja absolute integer overflow")
                    })?))
                } else {
                    Ok(Val::new(Kind::Float(value.number()?.abs())))
                }
            }
            "list" => Ok(Val::new(Kind::Array(value.elements()?))),
            "first" | "last" => {
                let values = value.elements()?;
                Ok(if name == "first" {
                    values.first()
                } else {
                    values.last()
                }
                .cloned()
                .unwrap_or_else(Val::undefined))
            }
            "reverse" => {
                let mut values = value.elements()?;
                values.reverse();
                Ok(Val::new(Kind::Array(values)))
            }
            "join" => {
                let delimiter = arg(&pos, &kw, "d", 1, Val::string(""));
                let attribute_key = arg(&pos, &kw, "attribute", 2, Val::undefined());
                let mut parts = Vec::new();
                for (index, item) in value.elements()?.iter().enumerate() {
                    if index != 0 {
                        parts.extend(delimiter.parts());
                    }
                    let item = if matches!(attribute_key.kind(), Kind::Undefined) {
                        item.clone()
                    } else {
                        attribute(item, &attribute_key)
                    };
                    parts.extend(item.parts());
                }
                Ok(Val::new(Kind::String(parts)))
            }
            "append" => {
                let item = pos
                    .get(1)
                    .cloned()
                    .ok_or_else(|| invalid("Jinja append needs a value"))?;
                value.ensure_insert(&item)?;
                let mut data = value.0.borrow_mut();
                let Kind::Array(values) = &mut data.kind else {
                    return Err(invalid("Jinja append needs an array"));
                };
                if values.len() >= 65536 {
                    return Err(invalid("Jinja array size bound exceeded"));
                }
                values.push(item);
                Ok(Val::new(Kind::None))
            }
            "pop" => {
                let index = arg(&pos, &kw, "index", 1, Val::int(-1)).integer()?;
                let mut data = value.0.borrow_mut();
                let Kind::Array(values) = &mut data.kind else {
                    return Err(invalid("Jinja pop needs an array"));
                };
                let index = value::index_at(index, values.len())
                    .ok_or_else(|| invalid("Jinja pop index out of range"))?;
                Ok(values.remove(index))
            }
            "sort" | "min" | "max" => {
                let values = value.elements()?;
                let reverse = name == "max"
                    || name == "sort" && arg(&pos, &kw, "reverse", 1, Val::boolean(false)).truth();
                let sensitive = arg(&pos, &kw, "case_sensitive", 2, Val::boolean(false)).truth();
                let key = arg(
                    &pos,
                    &kw,
                    "attribute",
                    if name == "sort" { 3 } else { 2 },
                    Val::undefined(),
                );
                let mut keyed = Vec::new();
                for item in values {
                    let mut sort_key = if matches!(key.kind(), Kind::Undefined) {
                        item.clone()
                    } else {
                        attribute(&item, &key)
                    };
                    if !sensitive && matches!(sort_key.kind(), Kind::String(_)) {
                        let mut bytes = sort_key.text();
                        bytes.make_ascii_lowercase();
                        sort_key = Val::bytes(bytes, false);
                    }
                    keyed.push((sort_key, item));
                }
                let mut failure = None;
                keyed.sort_by(|(a, _), (b, _)| match value::compare(a, b) {
                    Ok(order) => {
                        if reverse {
                            order.reverse()
                        } else {
                            order
                        }
                    }
                    Err(error) => {
                        failure = Some(error);
                        std::cmp::Ordering::Equal
                    }
                });
                if let Some(error) = failure {
                    return Err(error);
                }
                let mut values: Vec<_> = keyed.into_iter().map(|(_, value)| value).collect();
                if name == "sort" {
                    Ok(Val::new(Kind::Array(values)))
                } else {
                    Ok(if values.is_empty() {
                        Val::undefined()
                    } else {
                        values.remove(0)
                    })
                }
            }
            "map" => {
                let key = arg(&pos, &kw, "attribute", usize::MAX, Val::undefined());
                let default = arg(&pos, &kw, "default", usize::MAX, Val::undefined());
                let mut values = Vec::new();
                for item in value.elements()? {
                    let mapped = if matches!(key.kind(), Kind::Undefined) {
                        let name = String::from_utf8(
                            pos.get(1)
                                .ok_or_else(|| invalid("Jinja map needs an attribute or filter"))?
                                .text(),
                        )?;
                        let mut arguments = vec![item];
                        arguments.extend(pos.iter().skip(2).cloned());
                        self.builtin(&name, arguments, BTreeMap::new())?
                    } else {
                        attribute(&item, &key)
                    };
                    values.push(if matches!(mapped.kind(), Kind::Undefined) {
                        default.clone()
                    } else {
                        mapped
                    });
                }
                Ok(Val::new(Kind::Array(values)))
            }
            "select" | "reject" | "selectattr" | "rejectattr" => {
                let attr = name.ends_with("attr");
                if attr {
                    value.0.borrow_mut().array_access = true;
                }
                let mut result = Vec::new();
                let test_index = if attr { 2 } else { 1 };
                let test = pos
                    .get(test_index)
                    .map(|value| String::from_utf8(value.text()))
                    .transpose()?;
                for item in value.elements()? {
                    let tested = if attr {
                        attribute(
                            &item,
                            pos.get(1)
                                .ok_or_else(|| invalid("Jinja attribute selection needs a name"))?,
                        )
                    } else {
                        item.clone()
                    };
                    let selected = if let Some(test) = &test {
                        self.test(&tested, test, pos.get(test_index + 1..).unwrap_or_default())?
                    } else {
                        tested.truth()
                    };
                    if selected != name.starts_with("reject") {
                        result.push(item);
                    }
                }
                Ok(Val::new(Kind::Array(result)))
            }
            "unique" => {
                let mut values = Vec::new();
                for item in value.elements()? {
                    if !values.iter().any(|previous| value::equal(previous, &item)) {
                        values.push(item);
                    }
                }
                Ok(Val::new(Kind::Array(values)))
            }
            "sum" => {
                let key = arg(&pos, &kw, "attribute", 1, Val::undefined());
                let mut total = arg(&pos, &kw, "start", 2, Val::int(0));
                for item in value.elements()? {
                    let item = if matches!(key.kind(), Kind::Undefined) {
                        item
                    } else {
                        attribute(&item, &key)
                    };
                    total = self.binary("+", total, item)?;
                }
                Ok(total)
            }
            "get" => {
                let key = pos.get(1).ok_or_else(|| invalid("Jinja get needs a key"))?;
                let result = value.get(key, true);
                Ok(if matches!(result.kind(), Kind::Undefined) {
                    arg(&pos, &kw, "default", 2, Val::new(Kind::None))
                } else {
                    result
                })
            }
            "items" | "keys" | "values" | "dictsort" => {
                if matches!(value.kind(), Kind::Undefined) {
                    return Ok(Val::new(Kind::Array(Vec::new())));
                }
                value.mark();
                let Kind::Object(mut values) = value.kind() else {
                    return Err(invalid("Jinja object builtin needs an object"));
                };
                if name == "dictsort" {
                    let sensitive =
                        arg(&pos, &kw, "case_sensitive", 1, Val::boolean(false)).truth();
                    let by = text_arg(&pos, &kw, "by", 2, b"key");
                    if by != b"key" && by != b"value" {
                        return Err(invalid("Jinja dictsort by must be key or value"));
                    }
                    let reverse = arg(&pos, &kw, "reverse", 3, Val::boolean(false)).truth();
                    let mut error = None;
                    values.sort_by(|a, b| {
                        let a = if by == b"key" { &a.0 } else { &a.1 };
                        let b = if by == b"key" { &b.0 } else { &b.1 };
                        let key = |value: &Val| {
                            if !sensitive && matches!(value.kind(), Kind::String(_)) {
                                let mut bytes = value.text();
                                bytes.make_ascii_lowercase();
                                Val::bytes(bytes, false)
                            } else {
                                value.clone()
                            }
                        };
                        match value::compare(&key(a), &key(b)) {
                            Ok(order) => {
                                if reverse {
                                    order.reverse()
                                } else {
                                    order
                                }
                            }
                            Err(failure) => {
                                error = Some(failure);
                                std::cmp::Ordering::Equal
                            }
                        }
                    });
                    if let Some(error) = error {
                        return Err(error);
                    }
                }
                Ok(Val::new(Kind::Array(
                    values
                        .into_iter()
                        .map(|(key, value)| match name {
                            "keys" => key,
                            "values" => value,
                            _ => Val::new(Kind::Tuple(vec![key, value])),
                        })
                        .collect(),
                )))
            }
            "tojson" => {
                let options = JsonOptions::new(&pos, &kw)?;
                let bytes = json_bytes(&value, &options, 0)?;
                Ok(Val::bytes(bytes, false))
            }
            "indent" => {
                let width = arg(&pos, &kw, "width", 1, Val::int(4));
                let indent = if matches!(width.kind(), Kind::String(_)) {
                    width.text()
                } else {
                    let count = width.integer()?;
                    if !(0..=65536).contains(&count) {
                        return Err(invalid("Jinja indent width bound exceeded"));
                    }
                    vec![b' '; usize::try_from(count).unwrap_or(usize::MAX)]
                };
                let first = arg(&pos, &kw, "first", 2, Val::boolean(false)).truth();
                let blank = arg(&pos, &kw, "blank", 3, Val::boolean(false)).truth();
                let bytes = value.text();
                let mut output = Vec::new();
                for (index, line) in bytes.split(|&byte| byte == b'\n').enumerate() {
                    if index != 0 {
                        output.push(b'\n');
                    }
                    if if index == 0 {
                        first
                    } else {
                        !line.is_empty() || blank
                    } {
                        output.extend(&indent);
                    }
                    output.extend(line);
                    if output.len() > 4 * 1024 * 1024 {
                        return Err(invalid("Jinja indent output bound exceeded"));
                    }
                }
                Ok(transformed(&value, output))
            }
            "wordcount" => Ok(Val::int(i64::try_from(
                value
                    .text()
                    .split(u8::is_ascii_whitespace)
                    .filter(|word| !word.is_empty())
                    .count(),
            )?)),
            "truncate" => {
                let mut bytes = value.text();
                let length =
                    usize::try_from(arg(&pos, &kw, "length", 1, Val::int(255)).integer()?.max(0))
                        .unwrap_or(usize::MAX);
                let end = text_arg(&pos, &kw, "end", 3, b"...");
                if bytes.len() > length {
                    bytes.truncate(length.saturating_sub(end.len()));
                    bytes.extend(end);
                }
                Ok(transformed(&value, bytes))
            }
            _ => Err(invalid(format!("unsupported Jinja builtin {name}"))),
        }
    }
    #[allow(
        clippy::too_many_lines,
        reason = "Keep the exhaustive source predicate name dispatch together"
    )]
    pub(super) fn test(&mut self, value: &Val, name: &str, args: &[Val]) -> Result<bool> {
        self.tick()?;
        value.mark();
        let other = args.first().cloned().unwrap_or_else(Val::undefined);
        Ok(match name {
            "defined" => !matches!(value.kind(), Kind::Undefined),
            "undefined" => matches!(value.kind(), Kind::Undefined),
            "none" => matches!(value.kind(), Kind::None),
            "boolean" => matches!(value.kind(), Kind::Bool(_)),
            "true" => matches!(value.kind(), Kind::Bool(true)),
            "false" => matches!(value.kind(), Kind::Bool(false)),
            "integer" => matches!(value.kind(), Kind::Int(_)),
            "float" => matches!(value.kind(), Kind::Float(_)),
            "number" => matches!(value.kind(), Kind::Int(_) | Kind::Float(_)),
            "string" => matches!(value.kind(), Kind::String(_)),
            "mapping" => matches!(value.kind(), Kind::Object(_)),
            "sequence" | "iterable" => matches!(
                value.kind(),
                Kind::Undefined | Kind::Array(_) | Kind::Tuple(_) | Kind::String(_)
            ),
            "callable" => matches!(
                value.kind(),
                Kind::Function(_) | Kind::Method(..) | Kind::Macro(..) | Kind::Caller(..)
            ),
            "odd" => value.integer()? % 2 != 0,
            "even" => value.integer()? % 2 == 0,
            "divisibleby" => {
                let divisor = other.integer()?;
                if divisor == 0 {
                    return Err(invalid("Jinja divisibleby zero"));
                }
                value
                    .integer()?
                    .checked_rem(divisor)
                    .ok_or_else(|| invalid("Jinja modulo overflow"))?
                    == 0
            }
            "eq" | "equalto" | "sameas" => value::equal(value, &other),
            "ne" => !value::equal(value, &other),
            "ge" => !value::compare(value, &other)?.is_lt(),
            "gt" | "greaterthan" => value::compare(value, &other)?.is_gt(),
            "lt" | "lessthan" => value::compare(value, &other)?.is_lt(),
            "le" => !value::compare(value, &other)?.is_gt(),
            "in" => value::contains(&other, value)?,
            "lower" => {
                let bytes = value.text();
                bytes.iter().any(u8::is_ascii_lowercase)
                    && !bytes.iter().any(u8::is_ascii_uppercase)
            }
            "upper" => {
                let bytes = value.text();
                bytes.iter().any(u8::is_ascii_uppercase)
                    && !bytes.iter().any(u8::is_ascii_lowercase)
            }

            "escaped" => true,
            "filter" => {
                let name = String::from_utf8(value.text())?;
                matches!(
                    name.as_str(),
                    "trim" | "upper" | "lower" | "default" | "tojson"
                ) || has_method(value, &name)
            }
            "test" => {
                let name = String::from_utf8(value.text())?;
                matches!(
                    name.as_str(),
                    "boolean"
                        | "callable"
                        | "odd"
                        | "even"
                        | "false"
                        | "true"
                        | "divisibleby"
                        | "string"
                        | "integer"
                        | "float"
                        | "number"
                        | "iterable"
                        | "sequence"
                        | "mapping"
                        | "lower"
                        | "upper"
                        | "none"
                        | "defined"
                        | "undefined"
                        | "eq"
                        | "equalto"
                        | "ge"
                        | "gt"
                        | "greaterthan"
                        | "lt"
                        | "lessthan"
                        | "ne"
                        | "in"
                        | "test"
                        | "sameas"
                        | "escaped"
                        | "filter"
                )
            }
            _ => return Err(invalid(format!("unknown Jinja test {name}"))),
        })
    }
}

struct JsonOptions {
    ascii: bool,
    sort: bool,
    indent: Option<usize>,
    item: Vec<u8>,
    colon: Vec<u8>,
}
impl JsonOptions {
    fn new(pos: &[Val], kw: &BTreeMap<String, Val>) -> Result<Self> {
        let indentation = arg(pos, kw, "indent", 1, Val::undefined());
        let indent = if matches!(indentation.kind(), Kind::Undefined | Kind::None) {
            None
        } else {
            let value = indentation.integer()?;
            if !(0..=256).contains(&value) {
                return Err(invalid("Jinja JSON indentation bound exceeded"));
            }
            Some(usize::try_from(value)?)
        };
        let separators = arg(pos, kw, "separators", usize::MAX, Val::undefined());
        let (item, colon) = if matches!(separators.kind(), Kind::Undefined) {
            (
                if indent.is_some() {
                    b",".to_vec()
                } else {
                    b", ".to_vec()
                },
                b": ".to_vec(),
            )
        } else {
            let values = separators.elements()?;
            if values.len() != 2 {
                return Err(invalid("Jinja JSON separators require a pair"));
            }
            (values[0].text(), values[1].text())
        };
        Ok(Self {
            ascii: arg(pos, kw, "ensure_ascii", usize::MAX, Val::boolean(false)).truth(),
            sort: arg(pos, kw, "sort_keys", usize::MAX, Val::boolean(false)).truth(),
            indent,
            item,
            colon,
        })
    }
}
fn json_string(bytes: &[u8], ascii: bool) -> Result<Vec<u8>> {
    let text = std::str::from_utf8(bytes).map_err(|_| invalid("Jinja JSON string is not UTF-8"))?;
    let mut output = vec![b'"'];
    for ch in text.chars() {
        match ch {
            '"' => output.extend(b"\\\""),
            '\\' => output.extend(b"\\\\"),
            '\n' => output.extend(b"\\n"),
            '\r' => output.extend(b"\\r"),
            '\t' => output.extend(b"\\t"),
            '\u{8}' => output.extend(b"\\b"),
            '\u{c}' => output.extend(b"\\f"),
            ch if ch < ' ' || ascii && !ch.is_ascii() => {
                let mut storage = [0; 2];
                for unit in ch.encode_utf16(&mut storage) {
                    output.extend(format!("\\u{unit:04x}").bytes());
                }
            }
            _ => {
                let mut storage = [0; 4];
                output.extend(ch.encode_utf8(&mut storage).as_bytes());
            }
        }
    }
    output.push(b'"');
    Ok(output)
}
fn json_bytes(value: &Val, options: &JsonOptions, depth: usize) -> Result<Vec<u8>> {
    if depth > 64 {
        return Err(invalid("Jinja JSON nesting bound exceeded"));
    }
    value.mark();
    let mut output = Vec::new();
    match value.kind() {
        Kind::Undefined => return Err(invalid("Jinja undefined is not JSON serializable")),
        Kind::None => output.extend(b"null"),
        Kind::Bool(value) => output.extend(if value { &b"true"[..] } else { b"false" }),
        Kind::Int(_) | Kind::Float(_) => output.extend(value.text()),
        Kind::String(_) => output.extend(json_string(&value.text(), options.ascii)?),
        Kind::Array(values) | Kind::Tuple(values) => {
            output.push(b'[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.extend(&options.item);
                }
                if let Some(indent) = options.indent {
                    output.push(b'\n');
                    output.resize(output.len() + indent * (depth + 1), b' ');
                }
                output.extend(json_bytes(value, options, depth + 1)?);
            }
            if !values.is_empty()
                && let Some(indent) = options.indent
            {
                output.push(b'\n');
                output.resize(output.len() + indent * depth, b' ');
            }
            output.push(b']');
        }
        Kind::Object(mut values) => {
            if options.sort {
                values.sort_by_key(|(a, _)| a.text());
            }
            output.push(b'{');
            for (index, (key, value)) in values.iter().enumerate() {
                if index != 0 {
                    output.extend(&options.item);
                }
                if let Some(indent) = options.indent {
                    output.push(b'\n');
                    output.resize(output.len() + indent * (depth + 1), b' ');
                }
                let key = match key.kind() {
                    Kind::String(_) | Kind::Int(_) | Kind::Float(_) => key.text(),
                    Kind::None => b"null".to_vec(),
                    Kind::Bool(value) => {
                        if value {
                            b"true".to_vec()
                        } else {
                            b"false".to_vec()
                        }
                    }

                    _ => return Err(invalid("Jinja invalid JSON object key")),
                };
                output.extend(json_string(&key, options.ascii)?);
                output.extend(&options.colon);
                output.extend(json_bytes(value, options, depth + 1)?);
            }
            if !values.is_empty()
                && let Some(indent) = options.indent
            {
                output.push(b'\n');
                output.resize(output.len() + indent * depth, b' ');
            }
            output.push(b'}');
        }
        _ => return Err(invalid("Jinja callable is not JSON serializable")),
    }
    if output.len() > 4 * 1024 * 1024 {
        return Err(invalid("Jinja JSON output bound exceeded"));
    }
    Ok(output)
}
