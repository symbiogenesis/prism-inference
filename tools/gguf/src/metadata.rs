//! Native GGUF metadata, independent of any reporting representation.

/// Complete metadata indexed by the unique GGUF key.
pub type Metadata = std::collections::BTreeMap<String, MetadataValue>;

/// An exact GGUF scalar or homogeneous array, retaining its serialized type.
#[derive(Clone, Debug, PartialEq)]
pub enum MetadataValue {
    /// Unsigned 8-bit metadata.
    U8(u8),
    /// Signed 8-bit metadata.
    I8(i8),
    /// Unsigned 16-bit metadata.
    U16(u16),
    /// Signed 16-bit metadata.
    I16(i16),
    /// Unsigned 32-bit metadata.
    U32(u32),
    /// Signed 32-bit metadata.
    I32(i32),
    /// Native binary32 floating-point metadata.
    F32(f32),
    /// Boolean metadata.
    Bool(bool),
    /// UTF-8 string metadata.
    String(String),
    /// Unsigned 64-bit metadata.
    U64(u64),
    /// Signed 64-bit metadata.
    I64(i64),
    /// Native binary64 floating-point metadata.
    F64(f64),
    /// Homogeneous values, including strings and nested arrays.
    Array(MetadataArray),
}

/// Compact homogeneous storage for a GGUF array.
#[derive(Clone, Debug, PartialEq)]
pub enum MetadataArray {
    /// Unsigned 8-bit elements.
    U8(Vec<u8>),
    /// Signed 8-bit elements.
    I8(Vec<i8>),
    /// Unsigned 16-bit elements.
    U16(Vec<u16>),
    /// Signed 16-bit elements.
    I16(Vec<i16>),
    /// Unsigned 32-bit elements.
    U32(Vec<u32>),
    /// Signed 32-bit elements.
    I32(Vec<i32>),
    /// Native binary32 floating-point elements.
    F32(Vec<f32>),
    /// Boolean elements.
    Bool(Vec<bool>),
    /// UTF-8 string elements.
    String(Vec<String>),
    /// Unsigned 64-bit elements.
    U64(Vec<u64>),
    /// Signed 64-bit elements.
    I64(Vec<i64>),
    /// Native binary64 floating-point elements.
    F64(Vec<f64>),
    /// Nested homogeneous GGUF arrays.
    Array(Vec<Self>),
}

/// A borrowed view of a scalar or nested array, without allocating.
#[derive(Clone, Copy, Debug)]
pub enum MetadataValueRef<'a> {
    /// An unsigned integer of any GGUF width.
    Unsigned(u64),
    /// A signed integer of any GGUF width.
    Signed(i64),
    /// A floating-point number promoted losslessly from its stored width.
    Float(f64),
    /// A boolean.
    Bool(bool),
    /// A UTF-8 string.
    String(&'a str),
    /// A nested array.
    Array(&'a MetadataArray),
}

impl MetadataValueRef<'_> {
    /// Returns a nonnegative integer; floats and booleans are not coerced.
    #[must_use]
    pub fn as_u64(self) -> Option<u64> {
        match self {
            Self::Unsigned(value) => Some(value),
            Self::Signed(value) => u64::try_from(value).ok(),
            _ => None,
        }
    }

    /// Returns a signed integer when it fits; floats and booleans are not coerced.
    #[must_use]
    pub fn as_i64(self) -> Option<i64> {
        match self {
            Self::Signed(value) => Some(value),
            Self::Unsigned(value) => i64::try_from(value).ok(),
            _ => None,
        }
    }

    /// Returns a number, promoting integers with ordinary binary64 rounding.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::as_conversions,
        reason = "Numeric metadata access explicitly permits integer-to-binary64 rounding."
    )]
    pub const fn as_f64(self) -> Option<f64> {
        match self {
            Self::Float(value) => Some(value),
            Self::Unsigned(value) => Some(value as f64),
            Self::Signed(value) => Some(value as f64),
            _ => None,
        }
    }

    /// Returns a boolean without coercing other types.
    #[must_use]
    pub const fn as_bool(self) -> Option<bool> {
        match self {
            Self::Bool(value) => Some(value),
            _ => None,
        }
    }
}

impl<'a> MetadataValueRef<'a> {
    /// Returns a string without copying its contents.
    #[must_use]
    pub const fn as_str(self) -> Option<&'a str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }
}

impl MetadataValue {
    /// Borrows a value without copying strings or arrays.
    #[must_use]
    pub fn as_ref(&self) -> MetadataValueRef<'_> {
        match self {
            Self::U8(value) => MetadataValueRef::Unsigned(u64::from(*value)),
            Self::I8(value) => MetadataValueRef::Signed(i64::from(*value)),
            Self::U16(value) => MetadataValueRef::Unsigned(u64::from(*value)),
            Self::I16(value) => MetadataValueRef::Signed(i64::from(*value)),
            Self::U32(value) => MetadataValueRef::Unsigned(u64::from(*value)),
            Self::I32(value) => MetadataValueRef::Signed(i64::from(*value)),
            Self::F32(value) => MetadataValueRef::Float(f64::from(*value)),
            Self::Bool(value) => MetadataValueRef::Bool(*value),
            Self::String(value) => MetadataValueRef::String(value),
            Self::U64(value) => MetadataValueRef::Unsigned(*value),
            Self::I64(value) => MetadataValueRef::Signed(*value),
            Self::F64(value) => MetadataValueRef::Float(*value),
            Self::Array(value) => MetadataValueRef::Array(value),
        }
    }

    /// Returns a nonnegative integer when supported by the stored type.
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        self.as_ref().as_u64()
    }

    /// Returns a signed integer when supported by the stored type.
    #[must_use]
    pub fn as_i64(&self) -> Option<i64> {
        self.as_ref().as_i64()
    }

    /// Returns a number promoted to binary64 when supported by the stored type.
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        self.as_ref().as_f64()
    }

    /// Returns a boolean when supported by the stored type.
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        self.as_ref().as_bool()
    }

    /// Returns a string without copying it.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        self.as_ref().as_str()
    }

    /// Returns the array's compact homogeneous storage.
    #[must_use]
    pub const fn as_array(&self) -> Option<&MetadataArray> {
        match self {
            Self::Array(value) => Some(value),
            _ => None,
        }
    }
}

impl MetadataArray {
    /// Returns the number of elements.
    #[must_use]
    pub const fn len(&self) -> usize {
        match self {
            Self::U8(values) => values.len(),
            Self::I8(values) => values.len(),
            Self::U16(values) => values.len(),
            Self::I16(values) => values.len(),
            Self::U32(values) => values.len(),
            Self::I32(values) => values.len(),
            Self::F32(values) => values.len(),
            Self::Bool(values) => values.len(),
            Self::String(values) => values.len(),
            Self::U64(values) => values.len(),
            Self::I64(values) => values.len(),
            Self::F64(values) => values.len(),
            Self::Array(values) => values.len(),
        }
    }

    /// Returns whether there are no elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrows one element without allocating.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<MetadataValueRef<'_>> {
        match self {
            Self::U8(values) => values
                .get(index)
                .map(|value| MetadataValueRef::Unsigned(u64::from(*value))),
            Self::I8(values) => values
                .get(index)
                .map(|value| MetadataValueRef::Signed(i64::from(*value))),
            Self::U16(values) => values
                .get(index)
                .map(|value| MetadataValueRef::Unsigned(u64::from(*value))),
            Self::I16(values) => values
                .get(index)
                .map(|value| MetadataValueRef::Signed(i64::from(*value))),
            Self::U32(values) => values
                .get(index)
                .map(|value| MetadataValueRef::Unsigned(u64::from(*value))),
            Self::I32(values) => values
                .get(index)
                .map(|value| MetadataValueRef::Signed(i64::from(*value))),
            Self::F32(values) => values
                .get(index)
                .map(|value| MetadataValueRef::Float(f64::from(*value))),
            Self::Bool(values) => values
                .get(index)
                .map(|value| MetadataValueRef::Bool(*value)),
            Self::String(values) => values
                .get(index)
                .map(|value| MetadataValueRef::String(value)),
            Self::U64(values) => values
                .get(index)
                .map(|value| MetadataValueRef::Unsigned(*value)),
            Self::I64(values) => values
                .get(index)
                .map(|value| MetadataValueRef::Signed(*value)),
            Self::F64(values) => values
                .get(index)
                .map(|value| MetadataValueRef::Float(*value)),
            Self::Array(values) => values.get(index).map(MetadataValueRef::Array),
        }
    }

    /// Iterates borrowed elements without constructing per-element storage.
    pub fn iter(&self) -> impl Iterator<Item = MetadataValueRef<'_>> {
        (0..self.len()).filter_map(|index| self.get(index))
    }
}

impl From<u8> for MetadataValue {
    fn from(value: u8) -> Self {
        Self::U8(value)
    }
}

impl From<Vec<u8>> for MetadataValue {
    fn from(value: Vec<u8>) -> Self {
        Self::Array(MetadataArray::U8(value))
    }
}

impl From<i8> for MetadataValue {
    fn from(value: i8) -> Self {
        Self::I8(value)
    }
}

impl From<Vec<i8>> for MetadataValue {
    fn from(value: Vec<i8>) -> Self {
        Self::Array(MetadataArray::I8(value))
    }
}

impl From<u16> for MetadataValue {
    fn from(value: u16) -> Self {
        Self::U16(value)
    }
}

impl From<Vec<u16>> for MetadataValue {
    fn from(value: Vec<u16>) -> Self {
        Self::Array(MetadataArray::U16(value))
    }
}

impl From<i16> for MetadataValue {
    fn from(value: i16) -> Self {
        Self::I16(value)
    }
}

impl From<Vec<i16>> for MetadataValue {
    fn from(value: Vec<i16>) -> Self {
        Self::Array(MetadataArray::I16(value))
    }
}

impl From<u32> for MetadataValue {
    fn from(value: u32) -> Self {
        Self::U32(value)
    }
}

impl From<Vec<u32>> for MetadataValue {
    fn from(value: Vec<u32>) -> Self {
        Self::Array(MetadataArray::U32(value))
    }
}

impl From<i32> for MetadataValue {
    fn from(value: i32) -> Self {
        Self::I32(value)
    }
}

impl From<Vec<i32>> for MetadataValue {
    fn from(value: Vec<i32>) -> Self {
        Self::Array(MetadataArray::I32(value))
    }
}

impl From<f32> for MetadataValue {
    fn from(value: f32) -> Self {
        Self::F32(value)
    }
}

impl From<Vec<f32>> for MetadataValue {
    fn from(value: Vec<f32>) -> Self {
        Self::Array(MetadataArray::F32(value))
    }
}

impl From<bool> for MetadataValue {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<Vec<bool>> for MetadataValue {
    fn from(value: Vec<bool>) -> Self {
        Self::Array(MetadataArray::Bool(value))
    }
}

impl From<String> for MetadataValue {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<Vec<String>> for MetadataValue {
    fn from(value: Vec<String>) -> Self {
        Self::Array(MetadataArray::String(value))
    }
}

impl From<u64> for MetadataValue {
    fn from(value: u64) -> Self {
        Self::U64(value)
    }
}

impl From<Vec<u64>> for MetadataValue {
    fn from(value: Vec<u64>) -> Self {
        Self::Array(MetadataArray::U64(value))
    }
}

impl From<i64> for MetadataValue {
    fn from(value: i64) -> Self {
        Self::I64(value)
    }
}

impl From<Vec<i64>> for MetadataValue {
    fn from(value: Vec<i64>) -> Self {
        Self::Array(MetadataArray::I64(value))
    }
}

impl From<f64> for MetadataValue {
    fn from(value: f64) -> Self {
        Self::F64(value)
    }
}

impl From<Vec<f64>> for MetadataValue {
    fn from(value: Vec<f64>) -> Self {
        Self::Array(MetadataArray::F64(value))
    }
}

impl From<&str> for MetadataValue {
    fn from(value: &str) -> Self {
        Self::String(value.to_owned())
    }
}

impl From<Vec<&str>> for MetadataValue {
    fn from(value: Vec<&str>) -> Self {
        Self::Array(MetadataArray::String(
            value.into_iter().map(str::to_owned).collect(),
        ))
    }
}
