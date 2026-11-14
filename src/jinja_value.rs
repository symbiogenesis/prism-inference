//! Ordered, shared Jinja values, byte-string provenance and usage observations.
use super::ast::{Expr, Stmt};
use crate::{Result, invalid};
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use std::{cell::RefCell, collections::HashSet, fmt, rc::Rc};

const MAX_DAG_STEPS: usize = 65_536;
const MAX_TRAVERSAL_DEPTH: usize = 64;
const MAX_TEXT_OUTPUT_BYTES: usize = 1024 * 1024;

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
            if visited > MAX_DAG_STEPS || depth > MAX_TRAVERSAL_DEPTH || bytes > 4 * 1024 * 1024 {
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
        let mut seen = HashSet::new();
        let mut pending = vec![(value.clone(), 0_usize)];
        let mut steps = 0_usize;
        while let Some((node, depth)) = pending.pop() {
            steps += 1;
            if depth > MAX_TRAVERSAL_DEPTH || steps > MAX_DAG_STEPS {
                return Err(invalid("Jinja insertion traversal bound exceeded"));
            }
            if Rc::ptr_eq(&self.0, &node.0) {
                return Err(invalid("cyclic Jinja container assignment"));
            }
            if !seen.insert(Rc::as_ptr(&node.0)) {
                continue;
            }
            match node.kind() {
                Kind::Array(values) | Kind::Tuple(values) => {
                    pending.extend(values.into_iter().map(|child| (child, depth + 1)));
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
    pub(super) fn is_string(&self) -> bool {
        matches!(self.0.borrow().kind, Kind::String(_))
    }
    pub(super) fn is_int(&self) -> bool {
        matches!(self.0.borrow().kind, Kind::Int(_))
    }
    pub(super) fn raw_text_eq(&self, expected: &[u8]) -> bool {
        let borrowed = self.0.borrow();
        let Kind::String(parts) = &borrowed.kind else {
            return false;
        };
        let total: usize = parts.iter().map(|part| part.bytes.len()).sum();
        total == expected.len()
            && parts
                .iter()
                .flat_map(|part| part.bytes.iter().copied())
                .eq(expected.iter().copied())
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
        let mut seen = HashSet::new();
        let mut pending = vec![(self.clone(), 0_usize)];
        let mut steps = 0_usize;
        while let Some((value, depth)) = pending.pop() {
            if depth > MAX_TRAVERSAL_DEPTH || steps >= MAX_DAG_STEPS {
                break;
            }
            if !seen.insert(Rc::as_ptr(&value.0)) {
                continue;
            }
            steps += 1;
            value.mark();
            match value.kind() {
                Kind::Array(values) | Kind::Tuple(values) => {
                    pending.extend(values.into_iter().map(|child| (child, depth + 1)));
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
    }
    pub(super) fn truth(&self) -> bool {
        self.mark();
        match &self.0.borrow().kind {
            Kind::Undefined | Kind::None => false,
            Kind::Bool(value) => *value,
            Kind::Int(value) => *value != 0,
            Kind::Float(value) => *value != 0.0,
            Kind::String(parts) => parts.iter().any(|part| !part.bytes.is_empty()),
            Kind::Array(values) | Kind::Tuple(values) => !values.is_empty(),
            Kind::Object(values) => !values.is_empty(),
            _ => true,
        }
    }
    #[expect(
        clippy::cast_precision_loss,
        clippy::as_conversions,
        reason = "Jinja explicitly widens integer operands for floating-point operations; integer-only arithmetic remains checked i64"
    )]
    pub(super) fn number(&self) -> Result<f64> {
        self.mark();
        match self.0.borrow().kind {
            Kind::Bool(value) => Ok(f64::from(value)),
            Kind::Int(value) => Ok(value as f64),
            Kind::Float(value) => Ok(value),
            _ => Err(invalid("Jinja value is not numeric")),
        }
    }
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::as_conversions,
        reason = "Jinja float-to-int conversion truncates only after finite and representable-range checks"
    )]
    pub(super) fn integer(&self) -> Result<i64> {
        self.mark();
        match self.0.borrow().kind {
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
        let mut out = Vec::new();
        let mut steps = 0_usize;
        let _ = self.append_text(&mut out, 0, &mut steps);
        out
    }
    #[cfg(test)]
    pub(super) fn repr(&self) -> Vec<u8> {
        self.recursive_mark();
        let mut out = Vec::new();
        let mut steps = 0_usize;
        let _ = self.append_repr(&mut out, 0, &mut steps);
        out
    }
    fn append_text(&self, out: &mut Vec<u8>, depth: usize, steps: &mut usize) -> bool {
        *steps += 1;
        if depth > MAX_TRAVERSAL_DEPTH
            || *steps > MAX_DAG_STEPS
            || out.len() >= MAX_TEXT_OUTPUT_BYTES
        {
            return false;
        }
        match self.kind() {
            Kind::Undefined => true,
            Kind::None => push_bounded(out, b"None"),
            Kind::Bool(value) => push_bounded(out, if value { b"True" } else { b"False" }),
            Kind::Int(value) => push_bounded(out, value.to_string().as_bytes()),
            Kind::Float(value) => {
                let mut text = value.to_string();
                if !text.contains(['.', 'e', 'E']) {
                    text.push_str(".0");
                }
                push_bounded(out, text.as_bytes())
            }
            Kind::String(parts) => {
                for part in parts {
                    if !push_bounded(out, &part.bytes) {
                        return false;
                    }
                }
                true
            }
            Kind::Array(values) => {
                append_sequence(&values, b"[", b"]", false, out, depth + 1, steps)
            }
            Kind::Tuple(values) => {
                append_sequence(&values, b"(", b")", true, out, depth + 1, steps)
            }
            Kind::Object(values) => {
                if !push_bounded(out, b"{") {
                    return false;
                }
                for (index, (key, value)) in values.iter().enumerate() {
                    if index != 0 && !push_bounded(out, b", ") {
                        return false;
                    }
                    if !key.append_repr(out, depth + 1, steps)
                        || !push_bounded(out, b": ")
                        || !value.append_repr(out, depth + 1, steps)
                    {
                        return false;
                    }
                }
                push_bounded(out, b"}")
            }
            _ => push_bounded(out, b"<function>"),
        }
    }
    fn append_repr(&self, out: &mut Vec<u8>, depth: usize, steps: &mut usize) -> bool {
        *steps += 1;
        if depth > MAX_TRAVERSAL_DEPTH
            || *steps > MAX_DAG_STEPS
            || out.len() >= MAX_TEXT_OUTPUT_BYTES
        {
            return false;
        }
        if let Kind::String(parts) = self.kind() {
            if !push_bounded(out, b"'") {
                return false;
            }
            for part in parts {
                for byte in part.bytes {
                    let ok = match byte {
                        b'\'' => push_bounded(out, b"\\'"),
                        b'\\' => push_bounded(out, b"\\\\"),
                        b'\n' => push_bounded(out, b"\\n"),
                        b'\r' => push_bounded(out, b"\\r"),
                        b'\t' => push_bounded(out, b"\\t"),
                        _ => push_bounded(out, &[byte]),
                    };
                    if !ok {
                        return false;
                    }
                }
            }
            push_bounded(out, b"'")
        } else {
            self.append_text(out, depth, steps)
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
        match &self.0.borrow().kind {
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
fn push_bounded(out: &mut Vec<u8>, bytes: &[u8]) -> bool {
    let remaining = MAX_TEXT_OUTPUT_BYTES.saturating_sub(out.len());
    if remaining == 0 {
        return false;
    }
    let take = bytes.len().min(remaining);
    out.extend_from_slice(&bytes[..take]);
    take == bytes.len()
}

fn append_sequence(
    values: &[Val],
    begin: &[u8],
    end: &[u8],
    tuple: bool,
    out: &mut Vec<u8>,
    depth: usize,
    steps: &mut usize,
) -> bool {
    if !push_bounded(out, begin) {
        return false;
    }
    for (index, value) in values.iter().enumerate() {
        if index != 0 && !push_bounded(out, b", ") {
            return false;
        }
        if !value.append_repr(out, depth, steps) {
            return false;
        }
    }
    if tuple && values.len() == 1 && !push_bounded(out, b",") {
        return false;
    }
    push_bounded(out, end)
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
    let mut steps = 0_usize;
    equal_bounded(left, right, 0, &mut steps)
}

fn equal_bounded(left: &Val, right: &Val, depth: usize, steps: &mut usize) -> bool {
    *steps += 1;
    if depth > MAX_TRAVERSAL_DEPTH || *steps > MAX_DAG_STEPS {
        return false;
    }
    match (left.kind(), right.kind()) {
        (Kind::Undefined, Kind::Undefined) | (Kind::None, Kind::None) => true,
        (
            Kind::Bool(_) | Kind::Int(_) | Kind::Float(_),
            Kind::Bool(_) | Kind::Int(_) | Kind::Float(_),
        ) => left.number().ok() == right.number().ok(),
        (Kind::String(a), Kind::String(b)) => {
            left.mark();
            right.mark();
            a.iter()
                .flat_map(|part| part.bytes.iter().copied())
                .eq(b.iter().flat_map(|part| part.bytes.iter().copied()))
        }
        (Kind::Array(a), Kind::Array(b)) | (Kind::Tuple(a), Kind::Tuple(b)) => {
            a.len() == b.len()
                && a.iter()
                    .zip(&b)
                    .all(|(a, b)| equal_bounded(a, b, depth + 1, steps))
        }
        (Kind::Object(a), Kind::Object(b)) => {
            a.len() == b.len()
                && a.iter().all(|(key, value)| {
                    b.iter().any(|(other_key, other_value)| {
                        equal_bounded(key, other_key, depth + 1, steps)
                            && equal_bounded(value, other_value, depth + 1, steps)
                    })
                })
        }
        _ => false,
    }
}
pub(super) fn compare(left: &Val, right: &Val) -> Result<std::cmp::Ordering> {
    if let (Kind::String(a), Kind::String(b)) = (left.kind(), right.kind()) {
        left.mark();
        right.mark();
        Ok(a.iter()
            .flat_map(|part| part.bytes.iter().copied())
            .cmp(b.iter().flat_map(|part| part.bytes.iter().copied())))
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
                .any(|(other, _)| other.raw_text_eq(key.as_bytes()))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_and_repr_cap_output_to_one_mib_and_bound_shared_dag_steps() {
        let mut node = Val::string("x".repeat(1024));
        for _ in 0..18 {
            node = Val::new(Kind::Array(vec![node.clone(), node]));
        }
        let text = node.text();
        let repr = node.repr();
        assert!(text.len() <= MAX_TEXT_OUTPUT_BYTES);
        assert!(repr.len() <= MAX_TEXT_OUTPUT_BYTES);
        assert!(node.check_graph().is_err());
    }

    #[test]
    fn deep_dag_equal_and_recursive_mark_terminate_cleanly() {
        let mut left = Val::int(1);
        let mut right = Val::int(1);
        for _ in 0..70 {
            left = Val::new(Kind::Array(vec![left]));
            right = Val::new(Kind::Array(vec![right]));
        }
        left.recursive_mark();
        assert!(!equal(&left, &right));
    }
}
