//! Ordered, shared Jinja values, byte-string provenance and usage observations.
use super::ast::{Expr, Stmt};
use crate::{Result, invalid};
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use std::{cell::RefCell, fmt, rc::Rc};

#[derive(Clone, Debug)]
pub(super) struct Part {
    pub bytes: Vec<u8>,
    pub input: bool,
}
#[derive(Clone, Debug)]
pub(super) struct Val(pub Rc<RefCell<Data>>);
#[derive(Clone, Debug)]
pub(super) struct Data {
    pub kind: Kind,
    pub used: bool,
    pub array_access: bool,
    pub namespace: bool,
}
#[derive(Clone, Debug)]
pub(super) enum Kind {
    Undefined,
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(Vec<Part>),
    Array(Vec<Val>),
    Tuple(Vec<Val>),
    Object(Vec<(Val, Val)>),
    Function(String),
    Method(Val, String),
    Macro(Vec<(String, Option<Expr>)>, Vec<Stmt>),
    Caller(
        Vec<String>,
        Vec<Stmt>,
        Vec<std::collections::BTreeMap<String, Val>>,
    ),
}
impl Val {
    pub(super) fn check_graph(&self) -> Result<()> {
        let mut pending = vec![(self.clone(), 0_usize)];
        let mut visited = 0;
        let mut bytes = 0_usize;
        while let Some((value, depth)) = pending.pop() {
            visited += 1;
            bytes += 32;
            if visited > 65536 || depth > 64 || bytes > 4 * 1024 * 1024 {
                return Err(invalid("Jinja value graph bound exceeded"));
            }
            match value.kind() {
                Kind::String(parts) => {
                    bytes += parts.iter().map(|part| part.bytes.len()).sum::<usize>();
                    if bytes > 4 * 1024 * 1024 {
                        return Err(invalid("Jinja retained byte bound exceeded"));
                    }
                }
                Kind::Array(values) | Kind::Tuple(values) => {
                    pending.extend(values.into_iter().map(|value| (value, depth + 1)));
                }
                Kind::Object(values) => {
                    for (key, value) in values {
                        pending.push((key, depth + 1));
                        pending.push((value, depth + 1));
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
    pub(super) fn ensure_insert(&self, value: &Self) -> Result<()> {
        value.check_graph()?;
        let mut pending = vec![value.clone()];
        while let Some(value) = pending.pop() {
            if Rc::ptr_eq(&self.0, &value.0) {
                return Err(invalid("cyclic Jinja container assignment"));
            }
            match value.kind() {
                Kind::Array(values) | Kind::Tuple(values) => pending.extend(values),
                Kind::Object(values) => {
                    for (key, value) in values {
                        pending.push(key);
                        pending.push(value);
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
    pub(super) fn new(kind: Kind) -> Self {
        Self(Rc::new(RefCell::new(Data {
            kind,
            used: false,
            array_access: false,
            namespace: false,
        })))
    }
    pub(super) fn kind(&self) -> Kind {
        self.0.borrow().kind.clone()
    }
    pub(super) fn undefined() -> Self {
        Self::new(Kind::Undefined)
    }
    pub(super) fn int(value: i64) -> Self {
        Self::new(Kind::Int(value))
    }
    pub(super) fn boolean(value: bool) -> Self {
        Self::new(Kind::Bool(value))
    }
    pub(super) fn bytes(bytes: Vec<u8>, input: bool) -> Self {
        Self::new(Kind::String(vec![Part { bytes, input }]))
    }
    pub(super) fn string(value: impl Into<String>) -> Self {
        Self::bytes(value.into().into_bytes(), false)
    }
    pub(super) fn mark(&self) {
        self.0.borrow_mut().used = true;
    }
    pub(super) fn recursive_mark(&self) {
        self.mark();
        match self.kind() {
            Kind::Array(values) | Kind::Tuple(values) => {
                for value in values {
                    value.recursive_mark();
                }
            }
            Kind::Object(values) => {
                for (key, value) in values {
                    key.recursive_mark();
                    value.recursive_mark();
                }
            }
            _ => {}
        }
    }
    pub(super) fn truth(&self) -> bool {
        self.mark();
        match self.kind() {
            Kind::Undefined | Kind::None => false,
            Kind::Bool(value) => value,
            Kind::Int(value) => value != 0,
            Kind::Float(value) => value != 0.0,
            Kind::String(parts) => parts.iter().any(|part| !part.bytes.is_empty()),
            Kind::Array(values) | Kind::Tuple(values) => !values.is_empty(),
            Kind::Object(values) => !values.is_empty(),
            _ => true,
        }
    }
    #[allow(
        clippy::cast_precision_loss,
        clippy::as_conversions,
        reason = "Jinja explicitly widens integer operands for floating-point operations; integer-only arithmetic remains checked i64"
    )]
    pub(super) fn number(&self) -> Result<f64> {
        self.mark();
        match self.kind() {
            Kind::Bool(value) => Ok(f64::from(value)),
            Kind::Int(value) => Ok(value as f64),
            Kind::Float(value) => Ok(value),
            _ => Err(invalid("Jinja value is not numeric")),
        }
    }
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::as_conversions,
        reason = "Jinja float-to-int conversion truncates only after finite and representable-range checks"
    )]
    pub(super) fn integer(&self) -> Result<i64> {
        self.mark();
        match self.kind() {
            Kind::Bool(value) => Ok(i64::from(value)),
            Kind::Int(value) => Ok(value),
            Kind::Float(value)
                if value.is_finite() && value >= i64::MIN as f64 && value < i64::MAX as f64 =>
            {
                Ok(value as i64)
            }
            _ => Err(invalid("Jinja value is not an integer")),
        }
    }
    pub(super) fn text(&self) -> Vec<u8> {
        self.recursive_mark();
        match self.kind() {
            Kind::Undefined => Vec::new(),
            Kind::None => b"None".to_vec(),
            Kind::Bool(value) => {
                if value {
                    b"True".to_vec()
                } else {
                    b"False".to_vec()
                }
            }
            Kind::Int(value) => value.to_string().into_bytes(),
            Kind::Float(value) => {
                let mut text = value.to_string();
                if !text.contains(['.', 'e', 'E']) {
                    text.push_str(".0");
                }
                text.into_bytes()
            }
            Kind::String(parts) => parts.into_iter().flat_map(|part| part.bytes).collect(),
            Kind::Array(values) => repr_sequence(&values, b"[", b"]", false),
            Kind::Tuple(values) => repr_sequence(&values, b"(", b")", true),
            Kind::Object(values) => {
                let mut result = b"{".to_vec();
                for (index, (key, value)) in values.iter().enumerate() {
                    if index != 0 {
                        result.extend(b", ");
                    }
                    result.extend(key.repr());
                    result.extend(b": ");
                    result.extend(value.repr());
                }
                result.push(b'}');
                result
            }
            _ => b"<function>".to_vec(),
        }
    }
    pub(super) fn repr(&self) -> Vec<u8> {
        if matches!(self.kind(), Kind::String(_)) {
            let mut result = b"'".to_vec();
            for byte in self.text() {
                match byte {
                    b'\'' => result.extend(b"\\'"),
                    b'\\' => result.extend(b"\\\\"),
                    b'\n' => result.extend(b"\\n"),
                    b'\r' => result.extend(b"\\r"),
                    b'\t' => result.extend(b"\\t"),
                    _ => result.push(byte),
                }
            }
            result.push(b'\'');
            result
        } else {
            self.text()
        }
    }
    pub(super) fn parts(&self) -> Vec<Part> {
        self.recursive_mark();
        if let Kind::String(parts) = self.kind() {
            parts
        } else {
            vec![Part {
                bytes: self.text(),
                input: false,
            }]
        }
    }
    pub(super) fn len(&self) -> Result<usize> {
        self.mark();
        match self.kind() {
            Kind::Undefined => Ok(0),
            Kind::String(parts) => Ok(parts.iter().map(|part| part.bytes.len()).sum()),
            Kind::Array(values) | Kind::Tuple(values) => Ok(values.len()),
            Kind::Object(values) => Ok(values.len()),
            _ => Err(invalid("Jinja value has no length")),
        }
    }
    pub(super) fn elements(&self) -> Result<Vec<Self>> {
        self.mark();
        match self.kind() {
            Kind::Undefined => Ok(Vec::new()),
            Kind::Array(values) | Kind::Tuple(values) => Ok(values),
            Kind::Object(values) => Ok(values.into_iter().map(|pair| pair.0).collect()),
            Kind::String(parts) => Ok(parts
                .into_iter()
                .flat_map(|part| {
                    part.bytes
                        .into_iter()
                        .map(move |byte| Self::bytes(vec![byte], part.input))
                })
                .collect()),
            _ => Err(invalid("Jinja value is not iterable")),
        }
    }
    pub(super) fn get(&self, key: &Self, track: bool) -> Self {
        if track {
            self.mark();
            self.0.borrow_mut().array_access = true;
        }
        match self.kind() {
            Kind::Object(values) => values
                .into_iter()
                .find(|(candidate, _)| equal(candidate, key))
                .map_or_else(Self::undefined, |pair| pair.1),
            Kind::Array(values) | Kind::Tuple(values) => key
                .integer()
                .ok()
                .and_then(|index| index_at(index, values.len()))
                .and_then(|index| values.get(index).cloned())
                .unwrap_or_else(Self::undefined),
            Kind::String(parts) => {
                let bytes: Vec<_> = parts
                    .iter()
                    .flat_map(|part| part.bytes.iter().copied())
                    .collect();
                key.integer()
                    .ok()
                    .and_then(|index| index_at(index, bytes.len()))
                    .map_or_else(Self::undefined, |index| {
                        Self::bytes(vec![bytes[index]], parts.iter().any(|part| part.input))
                    })
            }
            _ => Self::undefined(),
        }
    }
    pub(super) fn set(&self, key: Self, value: Self) -> Result<()> {
        self.ensure_insert(&value)?;
        self.ensure_insert(&key)?;
        let mut data = self.0.borrow_mut();
        match &mut data.kind {
            Kind::Object(values) => {
                if let Some((_, destination)) = values
                    .iter_mut()
                    .find(|(candidate, _)| equal(candidate, &key))
                {
                    *destination = value;
                } else {
                    values.push((key, value));
                }
                Ok(())
            }
            Kind::Array(values) => {
                let index = index_at(key.integer()?, values.len())
                    .ok_or_else(|| invalid("Jinja assignment index out of range"))?;
                values[index] = value;
                Ok(())
            }
            _ => Err(invalid("Jinja assignment requires a mutable container")),
        }
    }
}
fn repr_sequence(values: &[Val], begin: &[u8], end: &[u8], tuple: bool) -> Vec<u8> {
    let mut result = begin.to_vec();
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            result.extend(b", ");
        }
        result.extend(value.repr());
    }
    if tuple && values.len() == 1 {
        result.push(b',');
    }
    result.extend(end);
    result
}
pub(super) fn index_at(index: i64, len: usize) -> Option<usize> {
    let index = if index < 0 {
        i64::try_from(len).ok()?.checked_add(index)?
    } else {
        index
    };
    usize::try_from(index).ok().filter(|&index| index < len)
}
pub(super) fn equal(left: &Val, right: &Val) -> bool {
    match (left.kind(), right.kind()) {
        (Kind::Undefined, Kind::Undefined) | (Kind::None, Kind::None) => true,
        (
            Kind::Bool(_) | Kind::Int(_) | Kind::Float(_),
            Kind::Bool(_) | Kind::Int(_) | Kind::Float(_),
        ) => left.number().ok() == right.number().ok(),
        (Kind::String(_), Kind::String(_)) => left.text() == right.text(),
        (Kind::Array(a), Kind::Array(b)) | (Kind::Tuple(a), Kind::Tuple(b)) => {
            a.len() == b.len() && a.iter().zip(&b).all(|(a, b)| equal(a, b))
        }
        (Kind::Object(a), Kind::Object(b)) => {
            a.len() == b.len()
                && a.iter().all(|(key, value)| {
                    b.iter().any(|(other_key, other_value)| {
                        equal(key, other_key) && equal(value, other_value)
                    })
                })
        }
        _ => false,
    }
}
pub(super) fn compare(left: &Val, right: &Val) -> Result<std::cmp::Ordering> {
    if matches!(left.kind(), Kind::String(_)) && matches!(right.kind(), Kind::String(_)) {
        Ok(left.text().cmp(&right.text()))
    } else {
        left.number()?
            .partial_cmp(&right.number()?)
            .ok_or_else(|| invalid("Jinja unordered numeric comparison"))
    }
}
pub(super) fn contains(container: &Val, needle: &Val) -> Result<bool> {
    container.mark();
    Ok(match container.kind() {
        Kind::String(_) => {
            let needle = needle.text();
            needle.is_empty()
                || container
                    .text()
                    .windows(needle.len())
                    .any(|part| part == needle)
        }
        Kind::Object(values) => values.iter().any(|(key, _)| equal(key, needle)),
        _ => container
            .elements()?
            .iter()
            .any(|value| equal(value, needle)),
    })
}
pub(super) fn merge(parts: Vec<Part>) -> Vec<Part> {
    let mut result: Vec<Part> = Vec::new();
    for part in parts {
        if part.bytes.is_empty() {
            continue;
        }
        if let Some(last) = result.last_mut().filter(|last| last.input == part.input) {
            last.bytes.extend(part.bytes);
        } else {
            result.push(part);
        }
    }
    result
}

struct OrderedVisitor;
impl<'de> Visitor<'de> for OrderedVisitor {
    type Value = Val;
    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value")
    }
    fn visit_bool<E: de::Error>(self, value: bool) -> std::result::Result<Val, E> {
        Ok(Val::boolean(value))
    }
    fn visit_i64<E: de::Error>(self, value: i64) -> std::result::Result<Val, E> {
        Ok(Val::int(value))
    }
    fn visit_u64<E: de::Error>(self, value: u64) -> std::result::Result<Val, E> {
        Ok(Val::int(i64::try_from(value).map_err(E::custom)?))
    }
    fn visit_f64<E: de::Error>(self, value: f64) -> std::result::Result<Val, E> {
        Ok(Val::new(Kind::Float(value)))
    }
    fn visit_str<E: de::Error>(self, value: &str) -> std::result::Result<Val, E> {
        Ok(Val::bytes(value.as_bytes().to_vec(), true))
    }
    fn visit_none<E: de::Error>(self) -> std::result::Result<Val, E> {
        Ok(Val::new(Kind::None))
    }
    fn visit_unit<E: de::Error>(self) -> std::result::Result<Val, E> {
        Ok(Val::new(Kind::None))
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> std::result::Result<Val, A::Error> {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element()? {
            values.push(value);
        }
        Ok(Val::new(Kind::Array(values)))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> std::result::Result<Val, A::Error> {
        let mut values: Vec<(Val, Val)> = Vec::new();
        while let Some(key) = map.next_key::<String>()? {
            if values
                .iter()
                .any(|(other, _)| other.text() == key.as_bytes())
            {
                return Err(de::Error::custom("duplicate Jinja variable key"));
            }
            values.push((Val::string(key), map.next_value()?));
        }
        Ok(Val::new(Kind::Object(values)))
    }
}
impl<'de> Deserialize<'de> for Val {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        deserializer.deserialize_any(OrderedVisitor)
    }
}
pub(super) fn variables(json: &str) -> Result<Val> {
    fn depth(value: &Val, level: usize) -> Result<()> {
        if level > 32 {
            return Err(invalid("Jinja variables exceed nesting bound"));
        }
        match value.kind() {
            Kind::Array(values) => {
                for value in values {
                    depth(&value, level + 1)?;
                }
            }
            Kind::Object(values) => {
                for (_, value) in values {
                    depth(&value, level + 1)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
    if json.len() > 256 * 1024 {
        return Err(invalid("Jinja variables exceed byte bound"));
    }
    let value: Val = serde_json::from_str(json)?;
    if !matches!(value.kind(), Kind::Object(_)) {
        return Err(invalid("Jinja variables must be an object"));
    }
    depth(&value, 0)?;
    Ok(value)
}
