//! Ordered JSON shared by chat protocols and schema-bearing tool builders.
use crate::{Result, invalid};
use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::ser::{SerializeMap, SerializeSeq};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Number, Value};
use std::fmt;
#[derive(Clone, Debug)]
#[allow(
    clippy::redundant_pub_crate,
    reason = "Share ordered JSON with sibling PEG builders while keeping its representation private to the crate"
)]
pub(crate) enum Ordered {
    Scalar(Value),
    Array(Vec<Self>),
    Object(Vec<(String, Self)>),
}
impl Ordered {
    pub(crate) fn parse(text: &str) -> Result<Self> {
        if text.len() > 4 * 1024 * 1024 {
            return Err(invalid("chat JSON exceeds byte bound"));
        }
        serde_json::from_str(text).map_err(Into::into)
    }
    pub(crate) fn get(&self, key: &str) -> Option<&Self> {
        if let Self::Object(values) = self {
            values
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value)
        } else {
            None
        }
    }
    pub(crate) fn array(&self) -> &[Self] {
        if let Self::Array(values) = self {
            values
        } else {
            &[]
        }
    }
    pub(crate) fn object(&self) -> &[(String, Self)] {
        if let Self::Object(values) = self {
            values
        } else {
            &[]
        }
    }
    pub(crate) fn text(&self) -> Option<&str> {
        if let Self::Scalar(value) = self {
            value.as_str()
        } else {
            None
        }
    }
    pub(crate) fn dump(&self) -> Result<String> {
        serde_json::to_string(self).map_err(Into::into)
    }
}
impl Serialize for Ordered {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        match self {
            Self::Scalar(value) => value.serialize(serializer),
            Self::Array(values) => {
                let mut sequence = serializer.serialize_seq(Some(values.len()))?;
                for value in values {
                    sequence.serialize_element(value)?;
                }
                sequence.end()
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
    type Value = Ordered;
    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an ordered JSON value")
    }
    fn visit_unit<E: serde::de::Error>(self) -> std::result::Result<Ordered, E> {
        Ok(Ordered::Scalar(Value::Null))
    }
    fn visit_bool<E: serde::de::Error>(self, value: bool) -> std::result::Result<Ordered, E> {
        Ok(Ordered::Scalar(Value::Bool(value)))
    }
    fn visit_i64<E: serde::de::Error>(self, value: i64) -> std::result::Result<Ordered, E> {
        Ok(Ordered::Scalar(Value::Number(value.into())))
    }
    fn visit_u64<E: serde::de::Error>(self, value: u64) -> std::result::Result<Ordered, E> {
        Ok(Ordered::Scalar(Value::Number(value.into())))
    }
    fn visit_f64<E: serde::de::Error>(self, value: f64) -> std::result::Result<Ordered, E> {
        Ok(Ordered::Scalar(Value::Number(
            Number::from_f64(value).ok_or_else(|| E::custom("non-finite chat JSON number"))?,
        )))
    }
    fn visit_str<E: serde::de::Error>(self, value: &str) -> std::result::Result<Ordered, E> {
        Ok(Ordered::Scalar(Value::String(value.into())))
    }
    fn visit_seq<A: SeqAccess<'de>>(
        self,
        mut sequence: A,
    ) -> std::result::Result<Ordered, A::Error> {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element()? {
            values.push(value);
        }
        Ok(Ordered::Array(values))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> std::result::Result<Ordered, A::Error> {
        let mut values: Vec<(String, Ordered)> = Vec::new();
        while let Some((key, value)) = map.next_entry()? {
            if let Some((_, previous)) = values.iter_mut().find(|(name, _)| name == &key) {
                *previous = value;
            } else {
                values.push((key, value));
            }
        }
        Ok(Ordered::Object(values))
    }
}
impl<'de> Deserialize<'de> for Ordered {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        deserializer.deserialize_any(OrderedVisitor)
    }
}
