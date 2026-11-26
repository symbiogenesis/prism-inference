//! Direct-read NEON `F32` matvec and joined `F32` sibling execution.
//!
//! Preserves exact bit-for-bit (`0 ULP`) parity with the four-FMA-register
//! 16-element [`super::pinned_gdn::dot`] reduction when pinned accumulation is
//! active on 16-aligned columns, and with the left-to-right `f64` sum of
//! rounded `f32` products otherwise. Reads weight bytes and immutable input
//! activations directly without row-buffer or activation-scratch copies.

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
use fearless_simd::{Neon, f32x4, f64x2, prelude::*, u8x16};

use super::{ActivationContract, F32, PreparedMatvec, dimensions, finite};
use crate::{Result, invalid, model::Tensor};

impl PreparedMatvec<'_> {
    /// Multiplies one `F32` matrix directly from borrowed tensor bytes and `self.input`.
    pub(super) fn multiply_f32(&mut self, tensor: &Tensor, output: &mut Vec<f32>) -> Result<()> {
        let (columns, _rows, row_bytes) = dimensions(tensor)?;
        if !self.quantized
            || tensor.kind != F32
            || columns != self.input.len()
            || self.contract != ActivationContract::F32
        {
            return Err(invalid("incompatible prepared F32 matvec"));
        }
        let input = self.input;
        let scratch = &mut *self.scratch;
        scratch.diagnostics = scratch.diagnostics.select(false);
        let pinned = scratch.pinned_accumulation && columns.is_multiple_of(16);
        let use_neon = !matches!(scratch.requested_backend(), super::MatvecBackend::Scalar);
        #[cfg(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc",
            target_feature = "neon"
        ))]
        if use_neon {
            if pinned {
                return scratch.row_parallelism.fill_four_rows(
                    &tensor.data,
                    row_bytes,
                    output,
                    &mut scratch.row_errors,
                    |rows8| f32_pinned_dot_eight_rows(rows8, input),
                    |rows4| f32_pinned_dot_four_rows(rows4, input),
                    |row| f32_pinned_dot_row(row, input),
                );
            }
            return scratch.row_parallelism.fill_four_rows(
                &tensor.data,
                row_bytes,
                output,
                &mut scratch.row_errors,
                |rows8| f32_f64_dot_eight_rows(rows8, input),
                |rows4| f32_f64_dot_four_rows(rows4, input),
                |row| f32_f64_dot_row(row, input),
            );
        }
        let _ = use_neon;
        scratch.row_parallelism.fill(
            &tensor.data,
            row_bytes,
            output,
            &mut scratch.row_errors,
            |row| {
                if pinned {
                    f32_pinned_dot_row_scalar(row, input)
                } else {
                    f32_f64_dot_row_scalar(row, input)
                }
            },
        )
    }

    /// Executes 2..=4 compatible `F32` siblings sharing `self.input` and one pool join.
    pub(crate) fn matvec_f32_siblings_into(
        &mut self,
        tensors: &[&Tensor],
        outputs: &mut [&mut Vec<f32>],
    ) -> Result<()> {
        self.scratch.reset_diagnostics();
        let result = self.multiply_f32_siblings(tensors, outputs);
        if result.is_err() {
            for output in outputs {
                output.clear();
            }
        }
        self.scratch.diagnostics.succeeded = result.is_ok();
        for _ in tensors {
            self.scratch.statistics.record(self.scratch.diagnostics);
        }
        result
    }

    fn multiply_f32_siblings(
        &mut self,
        tensors: &[&Tensor],
        outputs: &mut [&mut Vec<f32>],
    ) -> Result<()> {
        if !(2..=4).contains(&tensors.len())
            || outputs.len() != tensors.len()
            || !self.quantized
            || self.contract != ActivationContract::F32
        {
            return Err(invalid("incompatible prepared F32 sibling group"));
        }
        let mut row_bytes = 0_usize;
        let columns = self.input.len();
        for (tensor, output) in tensors.iter().zip(outputs.iter_mut()) {
            let (tensor_cols, rows, bytes) = dimensions(tensor)?;
            if tensor.kind != F32 || tensor_cols != columns {
                return Err(invalid("incompatible prepared F32 sibling tensor"));
            }
            row_bytes = bytes;
            if output.len() != rows {
                output.clear();
                output.try_reserve_exact(rows)?;
                output.resize(rows, 0.0);
            }
        }
        let input = self.input;
        let scratch = &mut *self.scratch;
        scratch.diagnostics = scratch.diagnostics.select(false);
        if scratch.row_parallelism.workers() > 1 {
            scratch.statistics.grouped_row_dispatches =
                scratch.statistics.grouped_row_dispatches.saturating_add(1);
            for _ in tensors {
                scratch.statistics.parallel_row_operations =
                    scratch.statistics.parallel_row_operations.saturating_add(1);
            }
        }
        let mut data = [&[][..]; 4];
        for (slot, tensor) in data.iter_mut().zip(tensors) {
            *slot = &tensor.data;
        }
        let pinned = scratch.pinned_accumulation && columns.is_multiple_of(16);
        let use_neon = !matches!(scratch.requested_backend(), super::MatvecBackend::Scalar);
        scratch.row_parallelism.fill_siblings(
            &data[..tensors.len()],
            row_bytes,
            outputs,
            |bytes, output| {
                #[cfg(all(
                    feature = "experimental-neon",
                    target_arch = "aarch64",
                    target_os = "windows",
                    target_env = "msvc",
                    target_feature = "neon"
                ))]
                if use_neon {
                    if pinned {
                        return super::parallel_rows::fill_batched_chunk(
                            bytes,
                            row_bytes,
                            output,
                            &|rows8| f32_pinned_dot_eight_rows(rows8, input),
                            &|rows4| f32_pinned_dot_four_rows(rows4, input),
                            &|row| f32_pinned_dot_row(row, input),
                        );
                    }
                    return super::parallel_rows::fill_batched_chunk(
                        bytes,
                        row_bytes,
                        output,
                        &|rows8| f32_f64_dot_eight_rows(rows8, input),
                        &|rows4| f32_f64_dot_four_rows(rows4, input),
                        &|row| f32_f64_dot_row(row, input),
                    );
                }
                let _ = use_neon;
                for (row, value) in bytes.chunks_exact(row_bytes).zip(output) {
                    *value = if pinned {
                        f32_pinned_dot_row_scalar(row, input)?
                    } else {
                        f32_f64_dot_row_scalar(row, input)?
                    };
                }
                Ok(())
            },
        )
    }
}

fn f32_pinned_dot_row_scalar(row: &[u8], input: &[f32]) -> Result<f32> {
    let mut sums = [[0.0_f32; 4]; 4];
    for (w_blk, x_blk) in row
        .as_chunks::<64>()
        .0
        .iter()
        .zip(input.as_chunks::<16>().0)
    {
        let w_regs = w_blk.as_chunks::<4>().0;
        for (reg, lanes) in sums.iter_mut().enumerate() {
            for (lane, sum) in lanes.iter_mut().enumerate() {
                let idx = reg * 4 + lane;
                let weight = finite(f32::from_le_bytes(w_regs[idx]))?;
                *sum = weight.mul_add(x_blk[idx], *sum);
            }
        }
    }
    let mut lanes = [0.0_f32; 4];
    for (index, lane) in lanes.iter_mut().enumerate() {
        *lane = (sums[0][index] + sums[2][index]) + (sums[1][index] + sums[3][index]);
    }
    finite((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Retain the scalar floating dot's rounded F32 products, binary64 sum and final F32 rounding."
)]
fn f32_f64_dot_row_scalar(row: &[u8], input: &[f32]) -> Result<f32> {
    let mut sum = -0.0_f64;
    for (bytes, &act) in row.as_chunks::<4>().0.iter().zip(input) {
        let weight = finite(f32::from_le_bytes(*bytes))?;
        sum += f64::from(weight * act);
    }
    finite(sum as f32)
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
fn f32_pinned_dot_row(row: &[u8], input: &[f32]) -> Result<f32> {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return f32_pinned_dot_row_scalar(row, input);
    };
    let mut s0 = f32x4::splat(neon, 0.0);
    let mut s1 = f32x4::splat(neon, 0.0);
    let mut s2 = f32x4::splat(neon, 0.0);
    let mut s3 = f32x4::splat(neon, 0.0);
    let mut finite_mask = f32x4::splat(neon, 0.0).is_finite();
    for (blk, act) in row
        .as_chunks::<64>()
        .0
        .iter()
        .zip(input.as_chunks::<16>().0)
    {
        let b4 = blk.as_chunks::<16>().0;
        let w0 = u8x16::from_slice(neon, &b4[0]).bitcast::<f32x4<Neon>>();
        let w1 = u8x16::from_slice(neon, &b4[1]).bitcast::<f32x4<Neon>>();
        let w2 = u8x16::from_slice(neon, &b4[2]).bitcast::<f32x4<Neon>>();
        let w3 = u8x16::from_slice(neon, &b4[3]).bitcast::<f32x4<Neon>>();
        finite_mask =
            finite_mask & w0.is_finite() & w1.is_finite() & w2.is_finite() & w3.is_finite();
        s0 = w0.mul_add_precise(f32x4::from_slice(neon, &act[0..4]), s0);
        s1 = w1.mul_add_precise(f32x4::from_slice(neon, &act[4..8]), s1);
        s2 = w2.mul_add_precise(f32x4::from_slice(neon, &act[8..12]), s2);
        s3 = w3.mul_add_precise(f32x4::from_slice(neon, &act[12..16]), s3);
    }
    if !finite_mask.all_true() {
        return Err(invalid("nonfinite tensor value"));
    }
    finite(((s0 + s2) + (s1 + s3)).reduce_sum())
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
fn f32_pinned_dot_four_rows(rows: [&[u8]; 4], input: &[f32]) -> Result<[f32; 4]> {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return Ok([
            f32_pinned_dot_row_scalar(rows[0], input)?,
            f32_pinned_dot_row_scalar(rows[1], input)?,
            f32_pinned_dot_row_scalar(rows[2], input)?,
            f32_pinned_dot_row_scalar(rows[3], input)?,
        ]);
    };
    let in_chunks = input.as_chunks::<16>().0;
    let r_chunks = rows.map(|r| r.as_chunks::<64>().0);
    let mut acc = [[f32x4::splat(neon, 0.0); 4]; 4];
    let mut finite_mask = f32x4::splat(neon, 0.0).is_finite();
    for (idx, act) in in_chunks.iter().enumerate() {
        let a0 = f32x4::from_slice(neon, &act[0..4]);
        let a1 = f32x4::from_slice(neon, &act[4..8]);
        let a2 = f32x4::from_slice(neon, &act[8..12]);
        let a3 = f32x4::from_slice(neon, &act[12..16]);
        for (r_chunk, r_acc) in r_chunks.iter().zip(acc.iter_mut()) {
            let b4 = r_chunk[idx].as_chunks::<16>().0;
            let w0 = u8x16::from_slice(neon, &b4[0]).bitcast::<f32x4<Neon>>();
            let w1 = u8x16::from_slice(neon, &b4[1]).bitcast::<f32x4<Neon>>();
            let w2 = u8x16::from_slice(neon, &b4[2]).bitcast::<f32x4<Neon>>();
            let w3 = u8x16::from_slice(neon, &b4[3]).bitcast::<f32x4<Neon>>();
            finite_mask =
                finite_mask & w0.is_finite() & w1.is_finite() & w2.is_finite() & w3.is_finite();
            r_acc[0] = w0.mul_add_precise(a0, r_acc[0]);
            r_acc[1] = w1.mul_add_precise(a1, r_acc[1]);
            r_acc[2] = w2.mul_add_precise(a2, r_acc[2]);
            r_acc[3] = w3.mul_add_precise(a3, r_acc[3]);
        }
    }
    if !finite_mask.all_true() {
        return Err(invalid("nonfinite tensor value"));
    }
    Ok([
        finite(((acc[0][0] + acc[0][2]) + (acc[0][1] + acc[0][3])).reduce_sum())?,
        finite(((acc[1][0] + acc[1][2]) + (acc[1][1] + acc[1][3])).reduce_sum())?,
        finite(((acc[2][0] + acc[2][2]) + (acc[2][1] + acc[2][3])).reduce_sum())?,
        finite(((acc[3][0] + acc[3][2]) + (acc[3][1] + acc[3][3])).reduce_sum())?,
    ])
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
fn f32_pinned_dot_eight_rows(rows: [&[u8]; 8], input: &[f32]) -> Result<[f32; 8]> {
    const K_TILE: usize = 32;
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return Ok([
            f32_pinned_dot_row_scalar(rows[0], input)?,
            f32_pinned_dot_row_scalar(rows[1], input)?,
            f32_pinned_dot_row_scalar(rows[2], input)?,
            f32_pinned_dot_row_scalar(rows[3], input)?,
            f32_pinned_dot_row_scalar(rows[4], input)?,
            f32_pinned_dot_row_scalar(rows[5], input)?,
            f32_pinned_dot_row_scalar(rows[6], input)?,
            f32_pinned_dot_row_scalar(rows[7], input)?,
        ]);
    };
    let in_chunks = input.as_chunks::<16>().0;
    let r_chunks = rows.map(|r| r.as_chunks::<64>().0);
    let mut acc = [[f32x4::splat(neon, 0.0); 4]; 8];
    let mut finite_mask = f32x4::splat(neon, 0.0).is_finite();
    // Process 4 rows per half over an L1-resident K tile to keep all 16
    // accumulators + 4 input + 4 weight vectors pinned in 24 NEON registers.
    for k_base in (0..in_chunks.len()).step_by(K_TILE) {
        let k_end = (k_base + K_TILE).min(in_chunks.len());
        for half in 0..2_usize {
            let r_base = half * 4;
            let mut local_acc = [
                acc[r_base],
                acc[r_base + 1],
                acc[r_base + 2],
                acc[r_base + 3],
            ];
            for (idx, act) in in_chunks[k_base..k_end].iter().enumerate() {
                let col_blk = k_base + idx;
                let a0 = f32x4::from_slice(neon, &act[0..4]);
                let a1 = f32x4::from_slice(neon, &act[4..8]);
                let a2 = f32x4::from_slice(neon, &act[8..12]);
                let a3 = f32x4::from_slice(neon, &act[12..16]);
                for (row_off, r_acc) in local_acc.iter_mut().enumerate() {
                    let b4 = r_chunks[r_base + row_off][col_blk].as_chunks::<16>().0;
                    let w0 = u8x16::from_slice(neon, &b4[0]).bitcast::<f32x4<Neon>>();
                    let w1 = u8x16::from_slice(neon, &b4[1]).bitcast::<f32x4<Neon>>();
                    let w2 = u8x16::from_slice(neon, &b4[2]).bitcast::<f32x4<Neon>>();
                    let w3 = u8x16::from_slice(neon, &b4[3]).bitcast::<f32x4<Neon>>();
                    finite_mask = finite_mask
                        & w0.is_finite()
                        & w1.is_finite()
                        & w2.is_finite()
                        & w3.is_finite();
                    r_acc[0] = w0.mul_add_precise(a0, r_acc[0]);
                    r_acc[1] = w1.mul_add_precise(a1, r_acc[1]);
                    r_acc[2] = w2.mul_add_precise(a2, r_acc[2]);
                    r_acc[3] = w3.mul_add_precise(a3, r_acc[3]);
                }
            }
            acc[r_base..r_base + 4].copy_from_slice(&local_acc);
        }
    }
    if !finite_mask.all_true() {
        return Err(invalid("nonfinite tensor value"));
    }
    Ok([
        finite(((acc[0][0] + acc[0][2]) + (acc[0][1] + acc[0][3])).reduce_sum())?,
        finite(((acc[1][0] + acc[1][2]) + (acc[1][1] + acc[1][3])).reduce_sum())?,
        finite(((acc[2][0] + acc[2][2]) + (acc[2][1] + acc[2][3])).reduce_sum())?,
        finite(((acc[3][0] + acc[3][2]) + (acc[3][1] + acc[3][3])).reduce_sum())?,
        finite(((acc[4][0] + acc[4][2]) + (acc[4][1] + acc[4][3])).reduce_sum())?,
        finite(((acc[5][0] + acc[5][2]) + (acc[5][1] + acc[5][3])).reduce_sum())?,
        finite(((acc[6][0] + acc[6][2]) + (acc[6][1] + acc[6][3])).reduce_sum())?,
        finite(((acc[7][0] + acc[7][2]) + (acc[7][1] + acc[7][3])).reduce_sum())?,
    ])
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
#[inline]
fn add_f32_columns(sum: f64x2<Neon>, first: f32x4<Neon>, second: f32x4<Neon>) -> f64x2<Neon> {
    let (l0, l1) = first.zip_low(second).widen();
    let (h0, h1) = first.zip_high(second).widen();
    (((sum + l0) + l1) + h0) + h1
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Retain the scalar floating dot's rounded F32 products, binary64 sum and final F32 rounding."
)]
fn f32_f64_dot_row(row: &[u8], input: &[f32]) -> Result<f32> {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return f32_f64_dot_row_scalar(row, input);
    };
    let (w_chunks, w_rem) = row.as_chunks::<16>();
    let (in_chunks, in_rem) = input.as_chunks::<4>();
    let mut sum = -0.0_f64;
    let mut finite_mask = f32x4::splat(neon, 0.0).is_finite();
    for (w_chunk, in_chunk) in w_chunks.iter().zip(in_chunks) {
        let w = u8x16::from_slice(neon, w_chunk).bitcast::<f32x4<Neon>>();
        finite_mask &= w.is_finite();
        let prod = (w * f32x4::from_slice(neon, in_chunk)).to_array();
        sum += f64::from(prod[0]);
        sum += f64::from(prod[1]);
        sum += f64::from(prod[2]);
        sum += f64::from(prod[3]);
    }
    if !finite_mask.all_true() {
        return Err(invalid("nonfinite tensor value"));
    }
    for (bytes, &act) in w_rem.as_chunks::<4>().0.iter().zip(in_rem) {
        let weight = finite(f32::from_le_bytes(*bytes))?;
        sum += f64::from(weight * act);
    }
    finite(sum as f32)
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Retain the scalar floating dot's rounded F32 products, binary64 sum and final F32 rounding."
)]
fn f32_f64_dot_rows<const PAIRS: usize, const ROWS: usize>(
    rows: [&[u8]; ROWS],
    input: &[f32],
) -> Result<[f32; ROWS]> {
    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        let mut out = [0.0_f32; ROWS];
        for (dst, row) in out.iter_mut().zip(rows) {
            *dst = f32_f64_dot_row_scalar(row, input)?;
        }
        return Ok(out);
    };
    let (in_chunks, in_rem) = input.as_chunks::<4>();
    let r_chunks = rows.map(|r| r.as_chunks::<16>().0);
    let r_rems = rows.map(|r| r.as_chunks::<16>().1);
    let mut sums = [f64x2::splat(neon, -0.0); PAIRS];
    let mut finite_mask = f32x4::splat(neon, 0.0).is_finite();
    for (idx, act4) in in_chunks.iter().enumerate() {
        let act = f32x4::from_slice(neon, act4);
        for (p, pair_sum) in sums.iter_mut().enumerate() {
            let w0 = u8x16::from_slice(neon, &r_chunks[p * 2][idx]).bitcast::<f32x4<Neon>>();
            let w1 = u8x16::from_slice(neon, &r_chunks[p * 2 + 1][idx]).bitcast::<f32x4<Neon>>();
            finite_mask = finite_mask & w0.is_finite() & w1.is_finite();
            *pair_sum = add_f32_columns(*pair_sum, w0 * act, w1 * act);
        }
    }
    if !finite_mask.all_true() {
        return Err(invalid("nonfinite tensor value"));
    }
    let mut out = [0.0_f32; ROWS];
    for (p, pair_sum) in sums.into_iter().enumerate() {
        let [mut s0, mut s1] = pair_sum.to_array();
        for ((b0, b1), &act) in r_rems[p * 2]
            .as_chunks::<4>()
            .0
            .iter()
            .zip(r_rems[p * 2 + 1].as_chunks::<4>().0)
            .zip(in_rem)
        {
            let w0 = finite(f32::from_le_bytes(*b0))?;
            let w1 = finite(f32::from_le_bytes(*b1))?;
            s0 += f64::from(w0 * act);
            s1 += f64::from(w1 * act);
        }
        out[p * 2] = finite(s0 as f32)?;
        out[p * 2 + 1] = finite(s1 as f32)?;
    }
    Ok(out)
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
fn f32_f64_dot_four_rows(rows: [&[u8]; 4], input: &[f32]) -> Result<[f32; 4]> {
    f32_f64_dot_rows::<2, 4>(rows, input)
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
fn f32_f64_dot_eight_rows(rows: [&[u8]; 8], input: &[f32]) -> Result<[f32; 8]> {
    f32_f64_dot_rows::<4, 8>(rows, input)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    #[cfg(feature = "parallel-rows")]
    use super::super::RowParallelism;
    use super::super::{
        F32, MatvecBackend, MatvecScratch, float_dot, matvec_into, pinned_gdn, row_into,
    };
    use crate::{Result, model::Tensor};

    fn make_f32_tensor(name: &str, columns: usize, rows: usize, seed: u32) -> Tensor {
        let mut s = seed;
        let mut bytes = Vec::with_capacity(columns * rows * 4);
        for _ in 0..columns * rows {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let val = f32::from_bits((s & 0x807f_ffff) | 0x3d80_0000);
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        Tensor {
            name: name.to_owned(),
            dimensions: vec![columns, rows],
            kind: F32,
            data: Arc::<[u8]>::from(bytes).into(),
        }
    }

    #[test]
    fn f32_direct_matvec_matches_scalar_zero_ulp_and_skips_copies() -> Result<()> {
        for columns in [1, 3, 15, 16, 17, 32, 128, 272] {
            for rows in [1, 4, 7, 8, 9, 17] {
                let tensor = make_f32_tensor("f32-direct", columns, rows, 0x1357_2468);
                let input: Vec<f32> = (0..columns)
                    .map(|i| [0.0, -0.0, 0.25, -0.5, 1.125, -2.5, f32::from_bits(1)][i % 7])
                    .collect();
                let mut decoded = Vec::new();
                for pinned in [false, true] {
                    let mut expected = Vec::with_capacity(rows);
                    for r in 0..rows {
                        row_into(&tensor, r, &mut decoded)?;
                        expected.push(if pinned && columns.is_multiple_of(16) {
                            pinned_gdn::dot(&decoded, &input)?
                        } else {
                            super::super::finite(float_dot(&decoded, &input))?
                        });
                    }
                    for backend in [MatvecBackend::Scalar, MatvecBackend::Auto] {
                        let mut scratch = if pinned {
                            MatvecScratch::with_pinned_backend(backend)
                        } else {
                            MatvecScratch::with_backend(backend)
                        };
                        let mut actual = Vec::new();
                        matvec_into(&tensor, &input, &mut actual, &mut scratch)?;
                        assert_eq!(scratch.activations.capacity(), 0);
                        assert_eq!(scratch.row.capacity(), 0);
                        assert_eq!(
                            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            "mismatch for columns={columns} rows={rows} pinned={pinned} backend={backend:?}"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn f32_siblings_match_single_zero_ulp_and_reject_atomically() -> Result<()> {
        let first = make_f32_tensor("a", 64, 17, 0x1111_2222);
        let second = make_f32_tensor("b", 64, 9, 0x3333_4444);
        let third = make_f32_tensor("c", 64, 5, 0x5555_6666);
        let input: Vec<f32> = (0..64)
            .map(|i| [0.5, -1.25, 0.0, -0.0, 2.5][i % 5])
            .collect();
        for pinned in [false, true] {
            let mut single_scratch = if pinned {
                MatvecScratch::with_pinned_backend(MatvecBackend::Auto)
            } else {
                MatvecScratch::with_backend(MatvecBackend::Auto)
            };
            let mut exp0 = Vec::new();
            let mut exp1 = Vec::new();
            let mut exp2 = Vec::new();
            matvec_into(&first, &input, &mut exp0, &mut single_scratch)?;
            matvec_into(&second, &input, &mut exp1, &mut single_scratch)?;
            matvec_into(&third, &input, &mut exp2, &mut single_scratch)?;

            let mut sib_scratch = if pinned {
                MatvecScratch::with_pinned_backend(MatvecBackend::Auto)
            } else {
                MatvecScratch::with_backend(MatvecBackend::Auto)
            };
            #[cfg(feature = "parallel-rows")]
            sib_scratch.set_row_parallelism(RowParallelism::new(4)?);
            let mut outs = [Vec::new(), Vec::new(), Vec::new()];
            {
                let mut prep = sib_scratch.prepare_borrowed(&first, &input)?;
                prep.matvec_f32_siblings_into(&[&first, &second, &third], &mut outs.each_mut())?;
            }
            assert_eq!(sib_scratch.activations.capacity(), 0);
            assert_eq!(sib_scratch.row.capacity(), 0);
            for (actual, expected) in outs.iter().zip([&exp0, &exp1, &exp2]) {
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }

            let mut bad = third.clone();
            let mut bytes = bad.data.to_vec();
            let end = bytes.len();
            bytes[end - 4..].copy_from_slice(&f32::NAN.to_le_bytes());
            bad.data = Arc::<[u8]>::from(bytes).into();
            let mut prep = sib_scratch.prepare(&first, &input)?;
            assert!(
                prep.matvec_f32_siblings_into(&[&first, &second, &bad], &mut outs.each_mut())
                    .is_err()
            );
            assert!(outs.iter().all(Vec::is_empty));
        }
        Ok(())
    }
}
