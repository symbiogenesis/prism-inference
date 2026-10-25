// SPDX-License-Identifier: MIT
//! Explicit C++ scalar promotions for the source-owned graph/input interpreter.

use serde_json::{Value, json};
use std::borrow::Cow;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Number {
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
    F32(f32),
    F64(f64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
}

impl Kind {
    pub(super) fn from_type(name: &str) -> Self {
        let name = if name.contains("const") {
            Cow::Owned(name.replace("const", "").replace("constexpr", ""))
        } else {
            Cow::Borrowed(name)
        };
        match name.as_ref() {
            "float" => Self::F32,
            "double" => Self::F64,
            "size_t" | "uint64_t" => Self::U64,
            "int64_t" => Self::I64,
            "uint32_t" => Self::U32,
            _ => Self::I32,
        }
    }
}

impl Number {
    pub(super) fn from_json(value: &Value) -> Result<Self, String> {
        if let Some(bits) = value.get("F32Bits").and_then(Value::as_u64) {
            let bits = u32::try_from(bits).map_err(|_| "F32 bits exceed u32")?;
            return Ok(Self::F32(f32::from_bits(bits)));
        }
        if let Some(bits) = value.get("F64Bits") {
            let bits = if let Some(bits) = bits.as_u64() {
                bits
            } else {
                bits.as_str()
                    .ok_or("invalid F64 bits")?
                    .parse()
                    .map_err(|_| "invalid F64 bits")?
            };
            return Ok(Self::F64(f64::from_bits(bits)));
        }
        if let Some(value) = value.get("U64").and_then(Value::as_str) {
            return Ok(Self::U64(value.parse().map_err(|_| "invalid U64")?));
        }
        if let Some(value) = value.as_i64() {
            return Ok(i32::try_from(value).map_or(Self::I64(value), Self::I32));
        }
        if let Some(value) = value.as_u64() {
            return Ok(Self::U64(value));
        }
        Err("source scalar is not a typed number".into())
    }

    pub(super) const fn kind(self) -> Kind {
        match self {
            Self::I32(_) => Kind::I32,
            Self::U32(_) => Kind::U32,
            Self::I64(_) => Kind::I64,
            Self::U64(_) => Kind::U64,
            Self::F32(_) => Kind::F32,
            Self::F64(_) => Kind::F64,
        }
    }

    pub(super) fn as_f64(self) -> f64 {
        match self {
            Self::I32(value) => f64::from(value),
            Self::U32(value) => f64::from(value),
            Self::I64(value) => value as f64,
            Self::U64(value) => value as f64,
            Self::F32(value) => f64::from(value),
            Self::F64(value) => value,
        }
    }

    pub(super) const fn as_f32(self) -> f32 {
        match self {
            Self::I32(value) => value as f32,
            Self::U32(value) => value as f32,
            Self::I64(value) => value as f32,
            Self::U64(value) => value as f32,
            Self::F32(value) => value,
            Self::F64(value) => value as f32,
        }
    }

    pub(super) fn as_i64(self) -> i64 {
        match self {
            Self::I32(value) => i64::from(value),
            Self::U32(value) => i64::from(value),
            Self::I64(value) => value,
            Self::U64(value) => value as i64,
            Self::F32(value) => value as i64,
            Self::F64(value) => value as i64,
        }
    }

    pub(super) fn as_u64(self) -> u64 {
        match self {
            Self::I32(value) => value as u64,
            Self::U32(value) => u64::from(value),
            Self::I64(value) => value as u64,
            Self::U64(value) => value,
            Self::F32(value) => value as u64,
            Self::F64(value) => value as u64,
        }
    }

    pub(super) fn index(self) -> Result<usize, String> {
        if matches!(self, Self::F32(_) | Self::F64(_)) || self.as_i64() < 0 {
            return Err("source array index is negative, floating, or exceeds signed range".into());
        }
        usize::try_from(self.as_u64()).map_err(|_| "source array index exceeds usize".into())
    }

    pub(super) fn truth(self) -> bool {
        match self {
            Self::F32(value) => value != 0.0,
            Self::F64(value) => value != 0.0,
            _ => self.as_u64() != 0,
        }
    }

    pub(super) fn cast(self, kind: Kind) -> Self {
        match kind {
            Kind::I32 => Self::I32(self.as_i64() as i32),
            Kind::U32 => Self::U32(self.as_u64() as u32),
            Kind::I64 => Self::I64(self.as_i64()),
            Kind::U64 => Self::U64(self.as_u64()),
            Kind::F32 => Self::F32(self.as_f32()),
            Kind::F64 => Self::F64(self.as_f64()),
        }
    }

    pub(super) fn wire(self) -> Value {
        match self {
            Self::I32(value) => json!({"I32":value}),
            Self::U32(value) => json!({"U64":value}),
            Self::I64(value) => json!({"I64":value}),
            Self::U64(value) => json!({"U64":value}),
            Self::F32(value) => json!({"F32Bits":value.to_bits()}),
            Self::F64(value) => json!({"F64Bits":value.to_bits()}),
        }
    }

    const fn common(self, other: Self) -> Kind {
        use Kind::{F32, F64, I32, I64, U32, U64};
        match (self.kind(), other.kind()) {
            (F64, _) | (_, F64) => F64,
            (F32, _) | (_, F32) => F32,
            (U64, _) | (_, U64) => U64,
            (I64, _) | (_, I64) => I64,
            (U32, _) | (_, U32) => U32,
            _ => I32,
        }
    }

    pub(super) fn equal(self, other: Self) -> bool {
        let kind = self.common(other);
        let (a, b) = (self.cast(kind), other.cast(kind));
        match kind {
            Kind::F32 => a.as_f32() == b.as_f32(),
            Kind::F64 => a.as_f64() == b.as_f64(),
            Kind::I32 | Kind::I64 => a.as_i64() == b.as_i64(),
            Kind::U32 | Kind::U64 => a.as_u64() == b.as_u64(),
        }
    }

    pub(super) fn less(self, other: Self) -> bool {
        let kind = self.common(other);
        let (a, b) = (self.cast(kind), other.cast(kind));
        match kind {
            Kind::F32 => a.as_f32() < b.as_f32(),
            Kind::F64 => a.as_f64() < b.as_f64(),
            Kind::I32 | Kind::I64 => a.as_i64() < b.as_i64(),
            Kind::U32 | Kind::U64 => a.as_u64() < b.as_u64(),
        }
    }

    pub(super) fn binary(self, op: &str, other: Self) -> Result<Self, String> {
        let kind = self.common(other);
        match kind {
            Kind::F32 => {
                let (a, b) = (self.as_f32(), other.as_f32());
                Ok(Self::F32(match op {
                    "+" => a + b,
                    "-" => a - b,
                    "*" => a * b,
                    "/" => a / b,
                    _ => return Err(format!("floating source operator {op}")),
                }))
            }
            Kind::F64 => {
                let (a, b) = (self.as_f64(), other.as_f64());
                Ok(Self::F64(match op {
                    "+" => a + b,
                    "-" => a - b,
                    "*" => a * b,
                    "/" => a / b,
                    _ => return Err(format!("floating source operator {op}")),
                }))
            }
            _ => {
                let a = self.cast(kind);
                let b = other.cast(kind);
                let signed = matches!(kind, Kind::I32 | Kind::I64);
                let value = match op {
                    "+" => a.as_u64().wrapping_add(b.as_u64()),
                    "-" => a.as_u64().wrapping_sub(b.as_u64()),
                    "*" => a.as_u64().wrapping_mul(b.as_u64()),
                    "/" | "%" if b.as_u64() == 0 => {
                        return Err("source integer division by zero".into());
                    }
                    "/" if signed => a.as_i64().wrapping_div(b.as_i64()) as u64,
                    "%" if signed => a.as_i64().wrapping_rem(b.as_i64()) as u64,
                    "/" => a.as_u64() / b.as_u64(),
                    "%" => a.as_u64() % b.as_u64(),
                    "&" => a.as_u64() & b.as_u64(),
                    "|" => a.as_u64() | b.as_u64(),
                    "^" => a.as_u64() ^ b.as_u64(),
                    "<<" | ">>" => {
                        // Shift result type comes from its promoted left operand, not usual conversions.
                        let width = if matches!(self.kind(), Kind::I32 | Kind::U32) {
                            32
                        } else {
                            64
                        };
                        let shift = other.as_u64();
                        if shift >= width {
                            return Err("source shift count exceeds promoted operand width".into());
                        }
                        let shifted = if op == "<<" {
                            self.as_u64() << shift
                        } else if matches!(self.kind(), Kind::I32 | Kind::I64) {
                            (self.as_i64() >> shift) as u64
                        } else {
                            self.as_u64() >> shift
                        };
                        return Ok(Self::U64(shifted).cast(self.kind()));
                    }
                    _ => return Err(format!("integer source operator {op}")),
                };
                Ok(Self::U64(value).cast(kind))
            }
        }
    }

    pub(super) fn negate(self) -> Self {
        match self {
            Self::I32(value) => Self::I32(value.wrapping_neg()),
            Self::U32(value) => Self::U32(value.wrapping_neg()),
            Self::I64(value) => Self::I64(value.wrapping_neg()),
            Self::U64(value) => Self::U64(value.wrapping_neg()),
            Self::F32(value) => Self::F32(-value),
            Self::F64(value) => Self::F64(-value),
        }
    }

    pub(super) fn complement(self) -> Result<Self, String> {
        Ok(match self {
            Self::I32(value) => Self::I32(!value),
            Self::U32(value) => Self::U32(!value),
            Self::I64(value) => Self::I64(!value),
            Self::U64(value) => Self::U64(!value),
            _ => return Err("source complement requires integer".into()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_type_mapping_keeps_plain_qualified_and_default_kinds() {
        for (name, expected) in [
            ("float", Kind::F32),
            ("double", Kind::F64),
            ("int", Kind::I32),
            ("int32_t", Kind::I32),
            ("uint32_t", Kind::U32),
            ("int64_t", Kind::I64),
            ("uint64_t", Kind::U64),
            ("size_t", Kind::U64),
            ("bool", Kind::I32),
            ("ggml_type", Kind::I32),
        ] {
            assert_eq!(Kind::from_type(name), expected);
            assert_eq!(Kind::from_type(&format!("const{name}")), expected);
        }
        // Declarations normalize constexpr before this mapper. Preserve its
        // existing direct-call fallback when const is removed first.
        assert_eq!(Kind::from_type("constexprfloat"), Kind::I32);
    }
}
