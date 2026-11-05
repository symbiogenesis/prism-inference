//! Signed `libggml-htp-v81.so` operation codes, pass flags and `kparams` solvers.
//!
//! All layouts and solver formulas match `ggml-hexagon.dll` / `libggml-htp-v81.so`
//! from the signed package (`19cce6919992fdffd92e0ac152caaf1e2b8f254e`).

use crate::{Error, HMX_Q2_HVX_WORKERS, HMX_Q2_VTCM_REQUEST_BYTES};

/// Operation codes supported by the signed `libggml-htp-v81.so` CDSP binary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum HtpOpCode {
    /// Elementwise multiplication (`HTP_OP_MUL`).
    Mul = 0,
    /// Elementwise addition (`HTP_OP_ADD`).
    Add = 1,
    /// Elementwise subtraction (`HTP_OP_SUB`).
    Sub = 2,
    /// Elementwise division (`HTP_OP_DIV`).
    Div = 3,
    /// Matrix multiplication (`HTP_OP_MUL_MAT`).
    MulMat = 4,
    /// Matrix multiplication with bias/residual addition (`HTP_OP_MUL_MAT_ADD`).
    MulMatAdd = 8,
    /// Root-mean-square layer normalization (`HTP_OP_RMS_NORM`).
    RmsNorm = 9,
    /// Fused root-mean-square normalization and scale multiply (`HTP_OP_RMS_NORM_MUL`).
    RmsNormMul = 10,
    /// Unary `SiLU` activation (`HTP_OP_UNARY_SILU`).
    UnarySilu = 11,
    /// Gated Linear Unit with `SwiGLU` activation (`HTP_OP_GLU_SWIGLU`).
    GluSwiglu = 22,
    /// Rotary position embedding (`HTP_OP_ROPE`).
    Rope = 28,
    /// Extended `FlashAttention` (`HTP_OP_FLASH_ATTN_EXT`).
    FlashAttnExt = 29,
    /// State-space causal 1D convolution (`HTP_OP_SSM_CONV`).
    SsmConv = 41,
    /// Gated `DeltaNet` recurrent state update (`HTP_OP_GATED_DELTA_NET`).
    GatedDeltaNet = 48,
    /// Fast Walsh-Hadamard transform (`HTP_OP_HADAMARD`).
    Hadamard = 69,
}

impl HtpOpCode {
    /// Raw 32-bit opcode written at offset `0` of `htp_op_desc`.
    #[must_use]
    pub const fn id(self) -> u32 {
        match self {
            Self::Mul => 0,
            Self::Add => 1,
            Self::Sub => 2,
            Self::Div => 3,
            Self::MulMat => 4,
            Self::MulMatAdd => 8,
            Self::RmsNorm => 9,
            Self::RmsNormMul => 10,
            Self::UnarySilu => 11,
            Self::GluSwiglu => 22,
            Self::Rope => 28,
            Self::FlashAttnExt => 29,
            Self::SsmConv => 41,
            Self::GatedDeltaNet => 48,
            Self::Hadamard => 69,
        }
    }
}

/// VTCM resident-chaining pass flags written at offset `4` of `htp_op_desc`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HtpOpPassFlags(u32);

impl HtpOpPassFlags {
    /// No VTCM residency chaining (`0`).
    pub const NONE: Self = Self(0);
    /// Keep `dst` in VTCM after this operation completes (`OP_PASS_KEEP_DST_IN_VTCM = 1`).
    pub const KEEP_DST_IN_VTCM: Self = Self(1);
    /// Read `src0` directly from VTCM (`OP_PASS_READ_SRC0_FROM_VTCM = 2`).
    pub const READ_SRC0_FROM_VTCM: Self = Self(2);
    /// Read `src1` directly from VTCM (`OP_PASS_READ_SRC1_FROM_VTCM = 4`).
    pub const READ_SRC1_FROM_VTCM: Self = Self(4);

    /// Combine two pass flag masks.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Raw 32-bit flags field for `htp_op_desc.flags`.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }
}

/// Compute the 64-bit `(shift << 32) | magic` fast-division constant used by `htp_kparams_solve`.
#[must_use]
#[allow(
    clippy::as_conversions,
    reason = "u64::from(u32) is not yet usable in const fn on stable Rust; u32 -> u64 widening is lossless"
)]
pub const fn htp_fastdiv(divisor: u32) -> u64 {
    if divisor <= 1 {
        return 1;
    }
    let mut shift = 0_u32;
    while shift < 31 && (1_u32 << shift) < divisor {
        shift += 1;
    }
    let divisor_u64 = divisor as u64;
    let numerator = ((1_u64 << shift) - divisor_u64) << 32;
    let magic = (numerator / divisor_u64) + 1;
    ((shift as u64) << 32) | (magic & 0xffff_ffff)
}

/// Solve the 64-byte `kparams` block (`op_desc[72..136]`) for `HTP_OP_RMS_NORM` (`9`),
/// `HTP_OP_RMS_NORM_MUL` (`10`), `HTP_OP_UNARY_SILU` (`11`) and `HTP_OP_HADAMARD` (`69`).
///
/// Matches `htp_kparams_solve_unary` (`0x18002A41C`) in `ggml-hexagon.dll`.
///
/// # Errors
/// Rejects zero extents or configurations that exceed the 2 MiB VTCM budget.
pub fn solve_unary_kparams(
    op: HtpOpCode,
    columns: u32,
    rows: u32,
    src1_rows: u32,
) -> Result<[u8; 64], Error> {
    if !matches!(
        op,
        HtpOpCode::RmsNorm | HtpOpCode::RmsNormMul | HtpOpCode::UnarySilu | HtpOpCode::Hadamard
    ) || columns == 0
        || rows == 0
    {
        return Err(Error::Contract);
    }
    if op == HtpOpCode::Hadamard && (columns < 32 || !columns.is_power_of_two()) {
        return Err(Error::Contract);
    }
    let row_bytes = columns.checked_mul(4).ok_or(Error::Contract)?;
    let n_threads = rows.clamp(1, HMX_Q2_HVX_WORKERS);
    let nrows_per_thread = rows.div_ceil(n_threads);
    let bcast_src1 = u32::from(op == HtpOpCode::RmsNormMul && src1_rows <= 1);
    let buffers_per_row = match op {
        HtpOpCode::RmsNormMul if bcast_src1 == 0 => 3_u32,
        _ => 2_u32,
    };
    let aligned_row = row_bytes.next_multiple_of(128);
    let bcast_vtcm = if bcast_src1 != 0 { aligned_row } else { 0 };
    let per_thread_vtcm = aligned_row
        .checked_mul(buffers_per_row)
        .and_then(|bytes| bytes.checked_mul(nrows_per_thread))
        .and_then(|bytes| bytes.checked_mul(n_threads))
        .and_then(|bytes| bytes.checked_add(bcast_vtcm))
        .ok_or(Error::Contract)?;
    if per_thread_vtcm > HMX_Q2_VTCM_REQUEST_BYTES {
        return Err(Error::Contract);
    }
    let mut kp = [0_u8; 64];
    put32(&mut kp, 0, n_threads);
    put32(&mut kp, 4, 0); // nrows_per_chunk = 0 when all rows fit in VTCM.
    put32(&mut kp, 8, nrows_per_thread);
    put32(&mut kp, 12, bcast_src1);
    put32(&mut kp, 16, 1); // contiguous src0.
    put32(&mut kp, 20, 1); // contiguous dst/src1.
    put32(&mut kp, 24, per_thread_vtcm);
    put32(&mut kp, 28, 1); // n_chunks = 1.
    put64(&mut kp, 32, htp_fastdiv(rows));
    put64(&mut kp, 40, htp_fastdiv(1));
    put64(&mut kp, 48, htp_fastdiv(rows));
    put64(&mut kp, 56, htp_fastdiv(1));
    Ok(kp)
}

/// Solve the 12-byte `kparams` block (`op_desc[72..84]`) for `HTP_OP_GLU_SWIGLU` (`22`).
///
/// Matches `htp_kparams_solve_glu` (`0x18002AE5C`) in `ggml-hexagon.dll`.
///
/// # Errors
/// Rejects zero extents or row sizes that cannot fit at least 2 rows in VTCM.
pub fn solve_glu_kparams(columns: u32, rows: u32) -> Result<[u8; 12], Error> {
    if columns == 0 || rows == 0 {
        return Err(Error::Contract);
    }
    let row_bytes = columns.checked_mul(4).ok_or(Error::Contract)?;
    let aligned_row = row_bytes.next_multiple_of(128);
    let n_threads = rows.clamp(1, HMX_Q2_HVX_WORKERS);
    let per_step_bytes = aligned_row
        .checked_mul(n_threads)
        .and_then(|bytes| bytes.checked_mul(3))
        .ok_or(Error::Contract)?;
    let max_chunk_rows = HMX_Q2_VTCM_REQUEST_BYTES / per_step_bytes;
    if max_chunk_rows < 2 {
        return Err(Error::Contract);
    }
    let nrows_per_thread = rows.div_ceil(n_threads);
    let chunk_rows = if nrows_per_thread < 2 {
        max_chunk_rows
    } else {
        let half_even = (nrows_per_thread >> 1).max(2) & !1;
        if nrows_per_thread >= max_chunk_rows {
            max_chunk_rows
        } else {
            half_even.min(max_chunk_rows).max(2)
        }
    };
    let vtcm_size = per_step_bytes
        .checked_mul(chunk_rows)
        .ok_or(Error::Contract)?;
    let mut kp = [0_u8; 12];
    put32(&mut kp, 0, n_threads);
    put32(&mut kp, 4, chunk_rows);
    put32(&mut kp, 8, vtcm_size);
    Ok(kp)
}

/// Solve the 96-byte `kparams` block (`op_desc[72..168]`) for contiguous same-shape `F32`
/// binary operations (`HTP_OP_MUL = 0`, `HTP_OP_ADD = 1`, `HTP_OP_SUB = 2`, `HTP_OP_DIV = 3`).
///
/// Matches `htp_kparams_solve_binary` (`0x18002CB00`) in `ggml-hexagon.dll`.
///
/// # Errors
/// Rejects non-binary opcodes, zero extents or VTCM overflow.
pub fn solve_binary_kparams(op: HtpOpCode, columns: u32, rows: u32) -> Result<[u8; 96], Error> {
    if !matches!(
        op,
        HtpOpCode::Mul | HtpOpCode::Add | HtpOpCode::Sub | HtpOpCode::Div
    ) || columns == 0
        || rows == 0
    {
        return Err(Error::Contract);
    }
    let total_elems = columns.checked_mul(rows).ok_or(Error::Contract)?;
    let max_chunk_elems = 32_768_u32 / 4;
    let target = total_elems
        .div_ceil(HMX_Q2_HVX_WORKERS * 2)
        .next_multiple_of(32)
        .max(256)
        .min(max_chunk_elems);
    let n_chunks = total_elems.div_ceil(target);
    let n_threads = n_chunks.clamp(1, HMX_Q2_HVX_WORKERS);
    let chunk_bytes = (target * 4).next_multiple_of(128);
    let row_bytes = (columns * 4).next_multiple_of(128);
    let vtcm_size = chunk_bytes
        .checked_mul(3)
        .and_then(|bytes| bytes.checked_mul(n_threads))
        .ok_or(Error::Contract)?;
    if vtcm_size > HMX_Q2_VTCM_REQUEST_BYTES {
        return Err(Error::Contract);
    }
    let mut kp = [0_u8; 96];
    kp[0] = 6; // Flat contiguous F32 binary kernel.
    kp[4] = 2;
    kp[5] = u8::from(total_elems == 1);
    put32(&mut kp, 8, if n_chunks > 1 { n_threads } else { 1 });
    put32(&mut kp, 12, 1);
    put32(&mut kp, 16, row_bytes);
    put32(&mut kp, 20, if total_elems == 1 { 0 } else { row_bytes });
    put32(&mut kp, 24, row_bytes);
    put32(&mut kp, 28, 0);
    put32(&mut kp, 36, vtcm_size);
    put32(&mut kp, 40, target);
    put32(&mut kp, 44, chunk_bytes);
    for offset in [48, 56, 64, 72, 80, 88] {
        put64(&mut kp, offset, htp_fastdiv(1));
    }
    Ok(kp)
}

fn put32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}
fn put64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn htp_fastdiv_matches_ggml_hexagon_dll_values() {
        assert_eq!(htp_fastdiv(1), 1);
        assert_eq!(htp_fastdiv(8), (3_u64 << 32) | 1);
        assert_eq!(htp_fastdiv(128), (7_u64 << 32) | 1);
        assert_eq!(htp_fastdiv(5120), (13_u64 << 32) | 0x9999_999a);
    }

    #[test]
    fn unary_glu_and_binary_kparams_solve_for_qwen35_dimensions() -> Result<(), Error> {
        let rms_kp = solve_unary_kparams(HtpOpCode::RmsNormMul, 5120, 1, 1)?;
        assert_eq!(
            u32::from_le_bytes([rms_kp[0], rms_kp[1], rms_kp[2], rms_kp[3]]),
            1
        );
        assert_eq!(
            u32::from_le_bytes([rms_kp[12], rms_kp[13], rms_kp[14], rms_kp[15]]),
            1
        );

        let hadamard_kp = solve_unary_kparams(HtpOpCode::Hadamard, 1024, 5, 1)?;
        assert_eq!(
            u32::from_le_bytes([
                hadamard_kp[0],
                hadamard_kp[1],
                hadamard_kp[2],
                hadamard_kp[3]
            ]),
            5
        );

        let glu_kp = solve_glu_kparams(17_408, 1)?;
        assert_eq!(
            u32::from_le_bytes([glu_kp[0], glu_kp[1], glu_kp[2], glu_kp[3]]),
            1
        );

        let mul_kp = solve_binary_kparams(HtpOpCode::Mul, 17_408, 1)?;
        assert_eq!(mul_kp[0], 6);
        assert_eq!(
            u32::from_le_bytes([mul_kp[8], mul_kp[9], mul_kp[10], mul_kp[11]]),
            8
        );
        Ok(())
    }
}
