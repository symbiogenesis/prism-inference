//! Validated PQ2 and PTQ1 codecs and shared numerical utilities.
//!
//! PQ2 stores a little-endian FP16 scale followed by 128 two-bit symbols
//! (`0 = -1`, `1 = 0`, `2 = +1`). Native PTQ1 stores 26 packed trit bytes
//! followed by that same scale. Packing preserves every valid source bit;
//! unpacking accepts only the canonical native encoding.
//!
//! The crate also supplies exact binary16 conversion and the native `Q8_K`
//! activation layout used by the Prism runtime. It has no model loader or
//! inference kernels.

#![forbid(unsafe_code)]

use std::error;
use std::fmt;

/// One native PQ2 group: a scale and 128 two-bit ternary symbols.
pub type Pq2Block = [u8; 34];

/// One native PTQ group: packed trits followed by an FP16 scale.
pub type PtqBlock = [u8; 28];

/// Native `Q8_K` activation layout, shared by two 128-weight groups.
///
/// The group sums are retained for layout compatibility but are not used by
/// the runtime scalar kernels. Values cover the entire `i8` range.
#[derive(Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Q8KBlock {
    /// Activation scale; the consuming kernel validates its supported range.
    pub scale: f32,
    /// Quantized activations, in weight order.
    pub values: [i8; 256],
    /// Native group sums, retained for layout compatibility.
    pub sums: [i16; 16],
}

const _: () = assert!(size_of::<Q8KBlock>() == 292);
const _: () = assert!(std::mem::offset_of!(Q8KBlock, values) == 4);

/// A rejected weight encoding or unsupported packed value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    /// Weight scales must be finite and nonzero FP16 values.
    InvalidWeightScale,
    /// A PQ2 block contains the reserved `+2` symbol.
    NonTernarySymbol,
    /// PTQ bytes do not use the unique encoding emitted by `pack_ptq`.
    NoncanonicalPtq,
    /// A packed value exceeds the supported integer range.
    SizeOverflow,
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidWeightScale => "finite nonzero FP16 scale required",
            Self::NonTernarySymbol => "PQ2 +2 symbol is not ternary",
            Self::NoncanonicalPtq => "noncanonical PTQ encoding",
            Self::SizeOverflow => "packed value capacity exceeded",
        })
    }
}

impl error::Error for Error {}

/// Converts every IEEE 754 binary16 bit pattern to binary32, preserving zero
/// signs and shifting NaN payload bits exactly as the C++ reference does.
#[must_use]
pub fn half_to_float(value: u16) -> f32 {
    let sign = u32::from(value & 0x8000) << 16;
    let mut mantissa = u32::from(value & 1023);
    let mut exponent = u32::from((value >> 10) & 31);
    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        exponent = 113;
        while mantissa & 1024 == 0 {
            mantissa <<= 1;
            exponent -= 1;
        }
        mantissa &= 1023;
    } else if exponent == 31 {
        exponent = 255;
    } else {
        exponent += 112;
    }
    f32::from_bits(sign | (exponent << 23) | (mantissa << 13))
}

const fn check_scale(bytes: &[u8]) -> Result<(), Error> {
    let scale = u16::from_le_bytes([bytes[0], bytes[1]]);
    if matches!(scale, 0 | 0x8000) || scale & 0x7c00 == 0x7c00 {
        return Err(Error::InvalidWeightScale);
    }
    Ok(())
}

const fn code(block: &Pq2Block, index: usize) -> u8 {
    (block[2 + index / 4] >> (2 * (index % 4))) & 3
}

fn check_pq2(block: &Pq2Block) -> Result<(), Error> {
    check_scale(block)?;
    if block[2..].iter().any(|byte| byte & (byte >> 1) & 0x55 != 0) {
        return Err(Error::NonTernarySymbol);
    }
    Ok(())
}

/// Packs one PQ2 group into the native PTQ byte layout.
///
/// # Errors
/// Returns an error for a zero or nonfinite weight scale or a reserved PQ2
/// symbol. All valid source bytes match the C++ reference.
pub fn pack_ptq(source: &Pq2Block) -> Result<PtqBlock, Error> {
    check_pq2(source)?;
    let mut result = [0; 28];
    let mut position = 0;
    for width in [16, 8] {
        let start = position / 5;
        for m in 0..width {
            let mut q = 0_u16;
            for n in 0..5 {
                q = q * 3 + u16::from(code(source, position + m + n * width));
            }
            result[start + m] =
                u8::try_from((q * 256).div_ceil(243)).map_err(|_| Error::SizeOverflow)?;
        }
        position += width * 5;
    }
    for h in 0..2 {
        let mut q = 0_u16;
        for n in 0..4 {
            q = q * 3 + u16::from(code(source, 120 + h + n * 2));
        }
        result[24 + h] =
            u8::try_from((q * 3 * 256).div_ceil(243)).map_err(|_| Error::SizeOverflow)?;
    }
    result[26..].copy_from_slice(&source[..2]);
    Ok(result)
}

// Each packed byte represents five base-three digits. The final two bytes
// represent four digits followed by zero. Store decoded digits and whether
// encoding them again would reproduce the input, so unpacking does not need
// a second full pass through pack_ptq to reject alternate encodings.
#[derive(Clone, Copy)]
struct PtqDigits {
    digits: [u8; 5],
    canonical_five: bool,
    canonical_four: bool,
}

const fn ptq_digits() -> [PtqDigits; 256] {
    let mut table = [PtqDigits {
        digits: [0; 5],
        canonical_five: false,
        canonical_four: false,
    }; 256];
    let mut index = 0;
    let mut byte = 0_u16;
    while byte < 256 {
        let mut fraction = byte;
        let mut value = 0;
        let mut digit = 0;
        while digit < 5 {
            fraction *= 3;
            let trit = fraction >> 8;
            // A trit is at most two, so its high byte is always zero.
            table[index].digits[digit] = trit.to_le_bytes()[0];
            value = value * 3 + trit;
            fraction &= 255;
            digit += 1;
        }
        table[index].canonical_five = (value * 256).div_ceil(243) == byte;
        table[index].canonical_four = (value / 3 * 3 * 256).div_ceil(243) == byte;
        byte += 1;
        index += 1;
    }
    table
}

const PTQ_DIGITS: [PtqDigits; 256] = ptq_digits();

/// Decodes the 128 signed trits in a native PTQ block.
///
/// Every packed byte has defined native decoder semantics, including alternate
/// fractional encodings. The scale bytes are not read. Use [`unpack_ptq`] when
/// validating a lossless, canonical wire conversion instead of a runtime tensor.
#[must_use]
pub fn decode_ptq_trits(source: &PtqBlock) -> [i16; 128] {
    std::array::from_fn(|index| {
        let (lane, digit) = if index < 80 {
            (index % 16, index / 16)
        } else if index < 120 {
            (16 + (index - 80) % 8, (index - 80) / 8)
        } else {
            (24 + (index - 120) % 2, (index - 120) / 2)
        };
        i16::from(PTQ_DIGITS[usize::from(source[lane])].digits[digit]) - 1
    })
}

/// Unpacks one native PTQ group to PQ2.
///
/// Unlike the C++ reference, this boundary rejects alternate encodings that
/// decode to the same ternary values: repacking must reproduce every byte.
///
/// # Errors
/// Returns an error if the stored weight scale is zero or nonfinite, or if
/// the encoding is noncanonical.
pub fn unpack_ptq(source: &PtqBlock) -> Result<Pq2Block, Error> {
    check_scale(&source[26..])?;
    let mut result = [0; 34];
    result[..2].copy_from_slice(&source[26..]);
    let mut position = 0;
    for width in [16, 8] {
        let start = position / 5;
        for m in 0..width {
            let digits = &PTQ_DIGITS[usize::from(source[start + m])];
            if !digits.canonical_five {
                return Err(Error::NoncanonicalPtq);
            }
            for (n, c) in digits.digits.iter().enumerate() {
                let index = position + n * width + m;
                result[2 + index / 4] |= c << (2 * (index % 4));
            }
        }
        position += width * 5;
    }
    for h in 0..2 {
        let digits = &PTQ_DIGITS[usize::from(source[24 + h])];
        if !digits.canonical_four {
            return Err(Error::NoncanonicalPtq);
        }
        for (n, c) in digits.digits[..4].iter().enumerate() {
            let index = 120 + n * 2 + h;
            result[2 + index / 4] |= c << (2 * (index % 4));
        }
    }
    Ok(result)
}
