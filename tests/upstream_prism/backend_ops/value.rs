// SPDX-License-Identifier: MIT
//! Safe values and addressable storage for source expressions.

use super::number::{Kind, Number};
use super::program::{Parameter, Statement};
use super::random::Mt19937;
use serde_json::Value as Json;
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

pub(super) type Cell = Rc<RefCell<Variable>>;
pub(super) type Scope = BTreeMap<String, Cell>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Handle {
    Context(u32),
    Tensor(u32),
    Graph(u32),
    Backend(u32),
    Buffer(u32),
}

impl Handle {
    pub(super) fn wire(self) -> Json {
        match self {
            Self::Context(id) => serde_json::json!({"Context":id}),
            Self::Tensor(id) => serde_json::json!({"Tensor":id}),
            Self::Graph(id) => serde_json::json!({"Graph":id}),
            Self::Backend(id) => serde_json::json!({"Backend":id}),
            Self::Buffer(id) => serde_json::json!({"Buffer":id}),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Element {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
}

impl Element {
    pub(super) fn parse(name: &str) -> Result<Self, String> {
        Ok(match name {
            "char" | "int8_t" => Self::I8,
            "uint8_t" => Self::U8,
            "int16_t" => Self::I16,
            "uint16_t" | "ggml_fp16_t" => Self::U16,
            "int" | "int32_t" => Self::I32,
            "uint32_t" => Self::U32,
            "int64_t" => Self::I64,
            "uint64_t" | "size_t" => Self::U64,
            "float" => Self::F32,
            "double" => Self::F64,
            _ => return Err(format!("source buffer element type {name}")),
        })
    }

    pub(super) const fn bytes(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct Buffer {
    pub(super) element: Element,
    pub(super) bytes: Vec<u8>,
}

impl Buffer {
    pub(super) fn new(element: Element, count: usize) -> Result<Self, String> {
        let size = count
            .checked_mul(element.bytes())
            .ok_or("source buffer size overflow")?;
        if size > 16 * 1024 * 1024 * 1024 {
            return Err("source buffer exceeds 16 GiB transport extent".into());
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size)
            .map_err(|error| format!("source buffer allocation: {error}"))?;
        bytes.resize(size, 0);
        Ok(Self { element, bytes })
    }

    pub(super) const fn len(&self) -> usize {
        self.bytes.len() / self.element.bytes()
    }

    fn slice<const N: usize>(&self, index: usize) -> Result<[u8; N], String> {
        let offset = index.checked_mul(N).ok_or("source buffer index overflow")?;
        self.bytes
            .get(offset..offset + N)
            .ok_or_else(|| "source buffer index out of bounds".into())
            .and_then(|bytes| {
                bytes
                    .try_into()
                    .map_err(|_| "source buffer element width".into())
            })
    }

    pub(super) fn get(&self, index: usize) -> Result<Number, String> {
        Ok(match self.element {
            Element::I8 => Number::I32(i32::from(i8::from_le_bytes(self.slice(index)?))),
            Element::U8 => Number::I32(i32::from(u8::from_le_bytes(self.slice(index)?))),
            Element::I16 => Number::I32(i32::from(i16::from_le_bytes(self.slice(index)?))),
            Element::U16 => Number::I32(i32::from(u16::from_le_bytes(self.slice(index)?))),
            Element::I32 => Number::I32(i32::from_le_bytes(self.slice(index)?)),
            Element::U32 => Number::U32(u32::from_le_bytes(self.slice(index)?)),
            Element::I64 => Number::I64(i64::from_le_bytes(self.slice(index)?)),
            Element::U64 => Number::U64(u64::from_le_bytes(self.slice(index)?)),
            Element::F32 => Number::F32(f32::from_le_bytes(self.slice(index)?)),
            Element::F64 => Number::F64(f64::from_le_bytes(self.slice(index)?)),
        })
    }

    pub(super) fn set(&mut self, index: usize, value: Number) -> Result<(), String> {
        let width = self.element.bytes();
        let offset = index
            .checked_mul(width)
            .ok_or("source buffer index overflow")?;
        let end = offset
            .checked_add(width)
            .ok_or("source buffer index overflow")?;
        let destination = self
            .bytes
            .get_mut(offset..end)
            .ok_or("source buffer index out of bounds")?;
        // Initializers and STL algorithms write every element through this
        // path. Encode on the stack instead of allocating a Vec per scalar.
        match self.element {
            Element::I8 | Element::U8 => destination[0] = value.as_u64() as u8,
            Element::I16 | Element::U16 => {
                destination.copy_from_slice(&(value.as_u64() as u16).to_le_bytes());
            }
            Element::I32 | Element::U32 => {
                destination.copy_from_slice(&(value.as_u64() as u32).to_le_bytes());
            }
            Element::I64 | Element::U64 => {
                destination.copy_from_slice(&value.as_u64().to_le_bytes());
            }
            Element::F32 => destination.copy_from_slice(&value.as_f32().to_le_bytes()),
            Element::F64 => destination.copy_from_slice(&value.as_f64().to_le_bytes()),
        }
        Ok(())
    }

    pub(super) fn push(&mut self, value: Number) -> Result<(), String> {
        let index = self.len();
        self.bytes
            .resize(self.bytes.len() + self.element.bytes(), 0);
        self.set(index, value)
    }
}

#[derive(Clone, Debug)]
pub(super) struct Variable {
    pub(super) kind: String,
    pub(super) value: Value,
}

#[derive(Clone, Debug)]
pub(super) enum Place {
    Variable(Cell),
    Array(Rc<RefCell<Vec<Value>>>, usize),
    Buffer(Rc<RefCell<Buffer>>, usize),
}

impl Place {
    pub(super) fn get(&self) -> Result<Value, String> {
        match self {
            Self::Variable(cell) => Ok(cell.borrow().value.clone()),
            Self::Array(array, index) => array
                .borrow()
                .get(*index)
                .cloned()
                .ok_or_else(|| "source array index out of bounds".into()),
            Self::Buffer(buffer, index) => buffer.borrow().get(*index).map(Value::Number),
        }
    }

    pub(super) fn set(&self, value: Value) -> Result<(), String> {
        match self {
            Self::Variable(cell) => {
                let value = value.coerce(&cell.borrow().kind)?;
                cell.borrow_mut().value = value;
            }
            Self::Array(array, index) => {
                *array
                    .borrow_mut()
                    .get_mut(*index)
                    .ok_or("source array index out of bounds")? = value;
            }
            Self::Buffer(buffer, index) => buffer.borrow_mut().set(*index, value.number()?)?,
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(super) struct Lambda {
    pub(super) parameters: Vec<Parameter>,
    pub(super) statements: Vec<Statement>,
    pub(super) scope: Scope,
}

#[derive(Clone, Debug)]
pub(super) enum Value {
    Null,
    Bool(bool),
    Number(Number),
    Bytes(Vec<u8>),
    Array(Rc<RefCell<Vec<Self>>>),
    Record(Rc<RefCell<BTreeMap<String, Self>>>),
    Buffer(Rc<RefCell<Buffer>>),
    Pointer {
        value: Box<Self>,
        index: usize,
    },
    Reference(Place),
    Address(Place),
    Handle(Handle),
    NativeInteger {
        kind: Kind,
        argument: Json,
    },
    Function(String),
    Lambda(Rc<Lambda>),
    Engine(Rc<RefCell<Mt19937>>),
    RandomDevice,
    Distribution {
        minimum: Number,
        maximum: Number,
        real: bool,
    },
    BackendDevice(u32),
    BackendRegistry(u32),
    TensorBytes {
        tensor: u32,
        field: String,
    },
    This,
}

impl Value {
    pub(super) fn number(&self) -> Result<Number, String> {
        match self {
            Self::Number(value) => Ok(*value),
            Self::Bool(value) => Ok(Number::I32(i32::from(*value))),
            Self::Reference(place) => place.get()?.number(),
            _ => Err(format!("source value is not numeric: {self:?}")),
        }
    }

    pub(super) fn truth(&self) -> Result<bool, String> {
        match self {
            Self::Null => Ok(false),
            Self::Bool(value) => Ok(*value),
            Self::Number(value) => Ok(value.truth()),
            Self::Handle(_) | Self::Pointer { .. } | Self::Address(_) => Ok(true),
            Self::Reference(place) => place.get()?.truth(),
            Self::NativeInteger { .. } => {
                Err("production metadata must be observed before source control flow".into())
            }
            _ => Err("source condition has no contextual bool conversion".into()),
        }
    }

    pub(super) fn array(values: Vec<Self>) -> Self {
        Self::Array(Rc::new(RefCell::new(values)))
    }

    pub(super) fn copy_value(&self) -> Self {
        match self {
            Self::Record(fields) => Self::Record(Rc::new(RefCell::new(
                fields
                    .borrow()
                    .iter()
                    .map(|(name, value)| (name.clone(), value.copy_value()))
                    .collect(),
            ))),
            Self::Array(values) => {
                Self::array(values.borrow().iter().map(Self::copy_value).collect())
            }
            Self::Buffer(buffer) => Self::Buffer(Rc::new(RefCell::new(buffer.borrow().clone()))),
            // References, data pointers, captured reference cells and engines retain their identity.
            value => value.clone(),
        }
    }

    pub(super) fn len(&self) -> Result<usize, String> {
        match self {
            Self::Address(_) => Ok(1),
            Self::Bytes(bytes) => Ok(bytes.len()),
            Self::Array(values) => Ok(values.borrow().len()),
            Self::Buffer(buffer) => Ok(buffer.borrow().len()),
            Self::Pointer { value, index } => value
                .len()?
                .checked_sub(*index)
                .ok_or_else(|| "source pointer past end".into()),
            Self::Reference(place) => place.get()?.len(),
            _ => Err("source value has no size".into()),
        }
    }

    pub(super) fn index(&self, index: usize) -> Result<Self, String> {
        match self {
            Self::Address(place) if index == 0 => place.get(),
            Self::Bytes(bytes) => bytes
                .get(index)
                .map(|&byte| Self::Number(Number::I32(i32::from(byte))))
                .ok_or_else(|| "source string index out of bounds".into()),
            Self::Array(values) => values
                .borrow()
                .get(index)
                .cloned()
                .ok_or_else(|| "source array index out of bounds".into()),
            Self::Buffer(buffer) => buffer.borrow().get(index).map(Self::Number),
            Self::Pointer {
                value,
                index: start,
            } => {
                let index = start
                    .checked_add(index)
                    .ok_or("source pointer index overflow")?;
                if matches!(value.as_ref(), Self::Bytes(bytes) if bytes.len() == index) {
                    Ok(Self::Number(Number::I32(0)))
                } else {
                    value.index(index)
                }
            }
            Self::Reference(place) => place.get()?.index(index),
            _ => Err("source value is not indexable".into()),
        }
    }

    pub(super) fn place(&self, index: usize) -> Result<Place, String> {
        match self {
            Self::Address(place) if index == 0 => Ok(place.clone()),
            Self::Array(values) if index < values.borrow().len() => {
                Ok(Place::Array(Rc::clone(values), index))
            }
            Self::Buffer(buffer) if index < buffer.borrow().len() => {
                Ok(Place::Buffer(Rc::clone(buffer), index))
            }
            Self::Pointer {
                value,
                index: start,
            } => value.place(
                start
                    .checked_add(index)
                    .ok_or("source pointer index overflow")?,
            ),
            Self::Reference(place) => place.get()?.place(index),
            _ => Err("source expression is not addressable".into()),
        }
    }

    pub(super) fn bytes(&self, count: usize) -> Result<Vec<u8>, String> {
        let (value, offset) = match self {
            Self::Pointer { value, index } => (value.as_ref(), *index),
            Self::Reference(place) | Self::Address(place) => return place.get()?.bytes(count),
            value => (value, 0),
        };
        match value {
            Self::Number(number) => {
                let bytes = match number {
                    Number::I32(value) => value.to_le_bytes().to_vec(),
                    Number::U32(value) => value.to_le_bytes().to_vec(),
                    Number::I64(value) => value.to_le_bytes().to_vec(),
                    Number::U64(value) => value.to_le_bytes().to_vec(),
                    Number::F32(value) => value.to_le_bytes().to_vec(),
                    Number::F64(value) => value.to_le_bytes().to_vec(),
                };
                bytes
                    .get(..count)
                    .map(<[u8]>::to_vec)
                    .ok_or_else(|| "source scalar byte read exceeds extent".into())
            }
            Self::Bytes(bytes) => bytes
                .get(offset..offset + count)
                .map(<[u8]>::to_vec)
                .ok_or_else(|| "source string byte read out of bounds".into()),
            Self::Buffer(buffer) => {
                let buffer = buffer.borrow();
                let offset = offset
                    .checked_mul(buffer.element.bytes())
                    .ok_or("source data pointer offset overflow")?;
                buffer
                    .bytes
                    .get(offset..offset + count)
                    .map(<[u8]>::to_vec)
                    .ok_or_else(|| "source buffer byte read out of bounds".into())
            }
            Self::Array(values) => {
                let values = values.borrow();
                let mut bytes = Vec::new();
                for value in values.iter().skip(offset) {
                    bytes.extend_from_slice(&value.number()?.as_i64().to_le_bytes());
                }
                bytes.truncate(count);
                if bytes.len() != count {
                    return Err("source array byte read out of bounds".into());
                }
                Ok(bytes)
            }
            _ => Err("source value has no byte representation".into()),
        }
    }

    pub(super) fn coerce(self, kind: &str) -> Result<Self, String> {
        // Scalar initializer loops repeatedly assign the same plain type.
        // Only qualified spellings need an owned normalization buffer.
        let clean = if kind.contains("const") || kind.contains("static") {
            Cow::Owned(
                kind.replace("constexpr", "")
                    .replace("const", "")
                    .replace("static", ""),
            )
        } else {
            Cow::Borrowed(kind)
        };
        if clean.starts_with("std::array<") {
            let definition = clean
                .strip_prefix("std::array<")
                .and_then(|text| text.strip_suffix('>'))
                .ok_or("invalid source array type")?;
            let (element, count) = definition
                .rsplit_once(',')
                .ok_or("source array type lacks extent")?;
            let count: usize = count
                .parse()
                .map_err(|_| "source array extent is not a literal")?;
            if element.contains('*') {
                let provided = self.len()?;
                if provided > count {
                    return Err("source pointer-array initializer exceeds extent".into());
                }
                let mut values = Vec::with_capacity(count);
                for index in 0..provided {
                    values.push(self.index(index)?);
                }
                values.resize(count, Self::Null);
                return Ok(Self::array(values));
            }
            let element = Element::parse(element)?;
            let mut buffer = Buffer::new(element, count)?;
            let provided = self.len()?;
            if provided > count {
                return Err("source array initializer exceeds extent".into());
            }
            for index in 0..provided {
                buffer.set(index, self.index(index)?.number()?)?;
            }
            return Ok(Self::Buffer(Rc::new(RefCell::new(buffer))));
        }
        if kind.contains('*') && matches!(self, Self::Bytes(_)) {
            return Ok(Self::Pointer {
                value: Box::new(self),
                index: 0,
            });
        }
        if kind == "auto" || kind.contains('*') || kind.contains('&') || clean.starts_with("std::")
        {
            return Ok(self);
        }
        if let Self::NativeInteger { argument, .. } = &self {
            return Ok(Self::NativeInteger {
                kind: Kind::from_type(&clean),
                argument: argument.clone(),
            });
        }
        if clean == "bool" {
            return self.truth().map(Self::Bool);
        }
        if matches!(self, Self::Number(_) | Self::Bool(_)) {
            return Ok(Self::Number(self.number()?.cast(Kind::from_type(&clean))));
        }
        Ok(self)
    }
}

pub(super) fn variable(kind: String, value: Value) -> Cell {
    Rc::new(RefCell::new(Variable { kind, value }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_writes_preserve_integer_truncation_float_bits_and_bounds() -> Result<(), String> {
        for (element, value, expected) in [
            (Element::I8, Number::I32(-1), vec![0xff]),
            (Element::U8, Number::U64(0x1234), vec![0x34]),
            (Element::I16, Number::I64(-2), vec![0xfe, 0xff]),
            (Element::U16, Number::U64(0x1234_abcd), vec![0xcd, 0xab]),
            (
                Element::I32,
                Number::I64(-3),
                (-3_i32).to_le_bytes().to_vec(),
            ),
            (
                Element::U32,
                Number::U64(0x1234_abcd_9876_5432),
                0x9876_5432_u32.to_le_bytes().to_vec(),
            ),
            (
                Element::I64,
                Number::I64(-4),
                (-4_i64).to_le_bytes().to_vec(),
            ),
            (
                Element::U64,
                Number::U64(u64::MAX),
                u64::MAX.to_le_bytes().to_vec(),
            ),
            (
                Element::F32,
                Number::F32(f32::from_bits(0xff80_0042)),
                0xff80_0042_u32.to_le_bytes().to_vec(),
            ),
            (
                Element::F64,
                Number::F64(f64::from_bits(0x8000_0000_0000_0000)),
                0x8000_0000_0000_0000_u64.to_le_bytes().to_vec(),
            ),
        ] {
            let mut buffer = Buffer::new(element, 3)?;
            buffer.set(1, value)?;
            assert_eq!(
                &buffer.bytes[element.bytes()..2 * element.bytes()],
                expected
            );
            assert!(
                buffer.bytes[..element.bytes()]
                    .iter()
                    .all(|&byte| byte == 0)
            );
            assert!(
                buffer.bytes[2 * element.bytes()..]
                    .iter()
                    .all(|&byte| byte == 0)
            );
            let unchanged = buffer.bytes.clone();
            assert!(buffer.set(3, value).is_err());
            assert!(buffer.set(usize::MAX, value).is_err());
            assert_eq!(buffer.bytes, unchanged);
        }
        Ok(())
    }

    #[test]
    fn qualified_scalar_assignments_preserve_casts_and_self_borrows() -> Result<(), String> {
        for (kind, value, expected) in [
            ("int32_t", Number::F64(-2.75), Number::I32(-2)),
            ("uint32_t", Number::I32(-1), Number::U32(u32::MAX)),
            (
                "int64_t",
                Number::U32(u32::MAX),
                Number::I64(i64::from(u32::MAX)),
            ),
            ("uint64_t", Number::I32(-1), Number::U64(u64::MAX)),
            ("float", Number::F64(-0.0), Number::F32(-0.0)),
            (
                "double",
                Number::F64(f64::from_bits(0x7ff0_0000_0000_0042)),
                Number::F64(f64::from_bits(0x7ff0_0000_0000_0042)),
            ),
        ] {
            for qualifier in ["", "const", "constexpr", "static", "staticconstexprconst"] {
                let cell = variable(format!("{qualifier}{kind}"), Value::Null);
                let place = Place::Variable(cell);
                place.set(Value::Number(value))?;
                assert_eq!(place.get()?.number()?.wire(), expected.wire());
                place.set(place.get()?)?;
                assert_eq!(place.get()?.number()?.wire(), expected.wire());
            }
        }
        let place = Place::Variable(variable("bool".into(), Value::Bool(true)));
        place.set(Value::Reference(place.clone()))?;
        assert!(place.get()?.truth()?);
        Ok(())
    }
}
