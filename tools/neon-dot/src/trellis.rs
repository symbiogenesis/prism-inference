//! `NariQ` trellis tile decoding, scalar reference kernels, and ARM64 NEON dispatch.

#[cfg(target_arch = "aarch64")]
use super::aarch64;

/// Edge dimension of a square `NariQ` trellis weight tile (`16 × 16`).
pub const TILE_DIM: usize = 16;

/// Odd multiplier of the 16-bit circular trellis weight generator.
pub const LCG_MULTIPLIER: u32 = 0x83dc_d12d;

/// Number of bytes per `16 × 16` trellis tile at `bits` (`1..=6`) bits per weight.
#[must_use]
pub const fn tile_bytes(bits: usize) -> Option<usize> {
    if bits >= 1 && bits <= 6 {
        Some(32 * bits)
    } else {
        None
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ColHalfSchedule {
    pub(crate) w0_q: usize,
    pub(crate) w0_rem: usize,
    pub(crate) w1_q: usize,
    pub(crate) w1_rem: usize,
    pub(crate) shift0: i32,
    pub(crate) shift1: i32,
    pub(crate) mask1: u32,
}

pub(crate) fn build_schedule(bits: usize) -> [ColHalfSchedule; 32] {
    let empty = ColHalfSchedule {
        w0_q: 0,
        w0_rem: 0,
        w1_q: 0,
        w1_rem: 0,
        shift0: 0,
        shift1: 0,
        mask1: 0,
    };
    let mut out = [empty; 32];
    let total_bits = 256 * bits;
    let mut c = 0_usize;
    while c < 16 {
        let mut r_hi = 0_usize;
        while r_hi < 2 {
            let c_lo = c % 8;
            let c_hi = c / 8;
            let c_term = 8 * (c_lo / 2) + 4 * r_hi + 2 * c_hi + (c & 1);
            let start = ((c_term + 1) * bits + total_bits - 16) % total_bits;
            let w = start >> 5;
            let o = start & 31;
            let w0_q = w / bits;
            let w0_rem = w % bits;
            let (w1_q, w1_rem) = if w0_rem + 1 < bits {
                (w0_q, w0_rem + 1)
            } else {
                ((w0_q + 1) & 7, 0)
            };
            let o_i32 = i32::try_from(o).unwrap_or(0);
            let shift0 = o_i32 - 16;
            let (shift1, mask1) = if o > 16 {
                (o_i32 - 48, 0xffff_u32)
            } else {
                (0, 0_u32)
            };
            out[c * 2 + r_hi] = ColHalfSchedule {
                w0_q,
                w0_rem,
                w1_q,
                w1_rem,
                shift0,
                shift1,
                mask1,
            };
            r_hi += 1;
        }
        c += 1;
    }
    out
}

#[inline]
pub(crate) fn read_le_u32(bytes: &[u8], word_idx: usize) -> u32 {
    let start = word_idx * 4;
    bytes
        .get(start..start + 4)
        .and_then(|slice| <[u8; 4]>::try_from(slice).ok())
        .map_or(0, u32::from_le_bytes)
}

#[inline]
fn decode_weight_i16(state: u32) -> i16 {
    let x = state.wrapping_mul(LCG_MULTIPLIER);
    let sum = ((x >> 2) & 63) + ((x >> 10) & 63) + ((x >> 18) & 63) + ((x >> 26) & 63);
    i16::try_from(sum).map_or(0, |value| value - 126)
}

/// Decodes a single weight from one `16 × 16` trellis tile at `(row, col)`.
///
/// Returns `None` if `bits` is outside `1..=6`, `row`/`col >= 16`, or `tile.len() != 32 * bits`.
#[must_use]
pub fn decode_single_value(tile: &[u8], bits: usize, row: usize, col: usize) -> Option<f32> {
    let expected = tile_bytes(bits)?;
    if row >= TILE_DIM || col >= TILE_DIM || tile.len() != expected {
        return None;
    }
    let r_hi = row / 8;
    let r_lo = row % 8;
    let c_hi = col / 8;
    let c_lo = col % 8;
    let p = 8 * (4 * r_lo + (c_lo / 2)) + 4 * r_hi + 2 * c_hi + (col & 1);
    let total_bits = 256 * bits;
    let n_words = 8 * bits;
    let start = ((p + 1) * bits + total_bits - 16) % total_bits;
    let w = start >> 5;
    let o = start & 31;
    let w_next = (w + 1) % n_words;
    let pair = (u64::from(read_le_u32(tile, w)) << 32) | u64::from(read_le_u32(tile, w_next));
    let state = u32::try_from((pair >> (48 - o)) & 0xffff).ok()?;
    Some(f32::from(decode_weight_i16(state)))
}

pub(crate) fn scatter_tile_planar(tile: &[u8], bits: usize, planar: &mut [[u32; 16]; 6]) {
    for q in 0..8 {
        for (rem, plane) in planar.iter_mut().enumerate().take(bits) {
            let val = read_le_u32(tile, q * bits + rem);
            plane[q] = val;
            plane[q + 8] = val;
        }
    }
}

fn validate_tile_strip(row_tiles: &[u8], bits: usize, cols: usize) -> Result<usize, &'static str> {
    let per_tile = tile_bytes(bits).ok_or("trellis bit width must be between 1 and 6")?;
    if !cols.is_multiple_of(TILE_DIM) {
        return Err("trellis input columns must divide 16");
    }
    let tile_cols = cols / TILE_DIM;
    let expected_bytes = tile_cols
        .checked_mul(per_tile)
        .ok_or("trellis tile row byte length overflow")?;
    if row_tiles.len() != expected_bytes {
        return Err("trellis tile row byte length mismatch");
    }
    Ok(per_tile)
}

/// Evaluates one `16`-row trellis strip against `s16`-scaled inputs using portable scalar code.
///
/// # Errors
/// Returns an error if `bits` is outside `1..=6`, `x_scaled.len()` is not a multiple of `16`,
/// or `row_tiles.len()` does not equal `(x_scaled.len() / 16) * 32 * bits`.
pub fn tile_row_matvec_scalar(
    row_tiles: &[u8],
    bits: usize,
    x_scaled: &[f32],
    out: &mut [f32; TILE_DIM],
) -> Result<(), &'static str> {
    let per_tile = validate_tile_strip(row_tiles, bits, x_scaled.len())?;
    let schedule = build_schedule(bits);
    let mut acc = [0.0_f32; TILE_DIM];
    let mut planar = [[0_u32; 16]; 6];
    let tile_cols = x_scaled.len() / TILE_DIM;

    for tile_idx in 0..tile_cols {
        let tile = &row_tiles[tile_idx * per_tile..(tile_idx + 1) * per_tile];
        scatter_tile_planar(tile, bits, &mut planar);
        let x_tile = &x_scaled[tile_idx * TILE_DIM..(tile_idx + 1) * TILE_DIM];
        for (c, &xc) in x_tile.iter().enumerate() {
            for r_hi in 0..2 {
                let sched = &schedule[c * 2 + r_hi];
                let row_base = r_hi * 8;
                for r_lo in 0..8 {
                    let w0 = planar[sched.w0_rem][sched.w0_q + r_lo];
                    let w1 = planar[sched.w1_rem][sched.w1_q + r_lo];
                    let part0 = if sched.shift0 >= 0 {
                        w0.wrapping_shl(u32::try_from(sched.shift0).unwrap_or(0))
                    } else {
                        w0.wrapping_shr(u32::try_from(-sched.shift0).unwrap_or(0))
                    };
                    let part1 = if sched.mask1 != 0 {
                        w1.wrapping_shr(u32::try_from(-sched.shift1).unwrap_or(0))
                    } else {
                        0
                    };
                    let state = (part0 | part1) & 0xffff;
                    let wf = f32::from(decode_weight_i16(state));
                    acc[row_base + r_lo] = wf.mul_add(xc, acc[row_base + r_lo]);
                }
            }
        }
    }

    *out = acc;
    Ok(())
}

/// Evaluates one `16`-row trellis strip against `s16`-scaled inputs using ARM64 NEON when built
/// on `aarch64`, falling back to scalar elsewhere.
///
/// # Errors
/// Returns an error under the same geometry conditions as [`tile_row_matvec_scalar`].
pub fn tile_row_matvec(
    row_tiles: &[u8],
    bits: usize,
    x_scaled: &[f32],
    out: &mut [f32; TILE_DIM],
) -> Result<(), &'static str> {
    let per_tile = validate_tile_strip(row_tiles, bits, x_scaled.len())?;
    #[cfg(target_arch = "aarch64")]
    {
        aarch64::trellis_tile_row_matvec(row_tiles, bits, per_tile, x_scaled, out);
        Ok(())
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = per_tile;
        tile_row_matvec_scalar(row_tiles, bits, x_scaled, out)
    }
}

/// Evaluates one `16`-row trellis strip across a slice of `s16`-scaled token activation vectors,
/// sharing decoded weight tiles across `4`-token batches.
///
/// # Errors
/// Returns an error if `bits` is outside `1..=6`, `x_tokens.len() != out_tokens.len()`, any
/// token vector has a mismatched column count, or `row_tiles.len()` does not match.
pub fn tile_row_matmul_tokens(
    row_tiles: &[u8],
    bits: usize,
    x_tokens: &[&[f32]],
    out_tokens: &mut [&mut [f32; TILE_DIM]],
) -> Result<(), &'static str> {
    if x_tokens.len() != out_tokens.len() {
        return Err("trellis multi-token input and output slice counts differ");
    }
    let Some(first) = x_tokens.first() else {
        let _ = tile_bytes(bits).ok_or("trellis bit width must be between 1 and 6")?;
        return Ok(());
    };
    let cols = first.len();
    let per_tile = validate_tile_strip(row_tiles, bits, cols)?;
    for token in &x_tokens[1..] {
        if token.len() != cols {
            return Err("trellis multi-token activation lengths differ");
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        let mut token_idx = 0_usize;
        while token_idx + 4 <= x_tokens.len() {
            let x4 = [
                x_tokens[token_idx],
                x_tokens[token_idx + 1],
                x_tokens[token_idx + 2],
                x_tokens[token_idx + 3],
            ];
            let mut o4 = [[0.0_f32; TILE_DIM]; 4];
            aarch64::trellis_tile_row_matmul4(row_tiles, bits, per_tile, cols, x4, &mut o4);
            for lane in 0..4 {
                *out_tokens[token_idx + lane] = o4[lane];
            }
            token_idx += 4;
        }
        while token_idx < x_tokens.len() {
            aarch64::trellis_tile_row_matvec(
                row_tiles,
                bits,
                per_tile,
                x_tokens[token_idx],
                out_tokens[token_idx],
            );
            token_idx += 1;
        }
        Ok(())
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = per_tile;
        for (x_scaled, out) in x_tokens.iter().zip(out_tokens.iter_mut()) {
            tile_row_matvec_scalar(row_tiles, bits, x_scaled, out)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        TILE_DIM, decode_single_value, tile_bytes, tile_row_matmul_tokens, tile_row_matvec,
        tile_row_matvec_scalar,
    };

    fn synthetic_bytes(len: usize, seed: u64) -> Vec<u8> {
        let mut state = seed | 1;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            out.push(u8::try_from((state >> 24) & 0xff).unwrap_or(0));
        }
        out
    }

    fn naive_reference(row_tiles: &[u8], bits: usize, x_scaled: &[f32]) -> Option<[f32; TILE_DIM]> {
        let tile_len = tile_bytes(bits)?;
        let tile_cols = x_scaled.len() / TILE_DIM;
        let mut out = [0.0_f32; TILE_DIM];
        for i in 0..tile_cols {
            let tile = &row_tiles[i * tile_len..(i + 1) * tile_len];
            for (r, slot) in out.iter_mut().enumerate() {
                for c in 0..TILE_DIM {
                    let w = decode_single_value(tile, bits, r, c)?;
                    *slot = w.mul_add(x_scaled[i * TILE_DIM + c], *slot);
                }
            }
        }
        Some(out)
    }

    #[test]
    fn trellis_matvec_and_batch_match_single_value_reference_across_bit_widths()
    -> Result<(), &'static str> {
        for bits in 1..=6 {
            let cols = 64_usize;
            let row_bytes = (cols / TILE_DIM) * tile_bytes(bits).ok_or("invalid bits")?;
            let weights = synthetic_bytes(row_bytes, 0x9e37_79b9_u64 + u64::try_from(bits).unwrap_or(0));

            let x_data: Vec<Vec<f32>> = (0..6)
                .map(|t| {
                    (0..cols)
                        .map(|c| {
                            let signed = i16::try_from((c * 7 + t * 13) % 31).unwrap_or(0) - 15;
                            f32::from(signed) * 0.125
                        })
                        .collect()
                })
                .collect();

            for token in &x_data {
                let reference = naive_reference(&weights, bits, token).ok_or("naive ref failed")?;
                let mut scalar = [0.0_f32; TILE_DIM];
                let mut neon = [0.0_f32; TILE_DIM];
                tile_row_matvec_scalar(&weights, bits, token, &mut scalar)?;
                tile_row_matvec(&weights, bits, token, &mut neon)?;
                for row in 0..TILE_DIM {
                    assert!(
                        (scalar[row] - reference[row]).abs() <= 1e-3,
                        "scalar mismatch bits={bits} row={row}: {} vs {}",
                        scalar[row],
                        reference[row]
                    );
                    assert!(
                        (neon[row] - reference[row]).abs() <= 1e-3,
                        "neon mismatch bits={bits} row={row}: {} vs {}",
                        neon[row],
                        reference[row]
                    );
                }
            }

            let refs: Vec<&[f32]> = x_data.iter().map(Vec::as_slice).collect();
            let mut batch_bufs = [[0.0_f32; TILE_DIM]; 6];
            let mut out_refs: Vec<&mut [f32; TILE_DIM]> = batch_bufs.iter_mut().collect();
            tile_row_matmul_tokens(&weights, bits, &refs, &mut out_refs)?;
            for (t, token) in x_data.iter().enumerate() {
                let expected = naive_reference(&weights, bits, token).ok_or("ref failed")?;
                for row in 0..TILE_DIM {
                    assert!(
                        (batch_bufs[t][row] - expected[row]).abs() <= 1e-3,
                        "batch mismatch bits={bits} token={t} row={row}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn trellis_kernels_reject_malformed_geometries_and_bit_widths() {
        let mut out = [0.0_f32; TILE_DIM];
        assert!(tile_bytes(0).is_none());
        assert!(tile_bytes(7).is_none());
        assert!(tile_row_matvec(&[0_u8; 64], 0, &[0.0_f32; 16], &mut out).is_err());
        assert!(tile_row_matvec(&[0_u8; 64], 7, &[0.0_f32; 16], &mut out).is_err());
        assert!(tile_row_matvec(&[0_u8; 64], 2, &[0.0_f32; 15], &mut out).is_err());
        assert!(tile_row_matvec(&[0_u8; 63], 2, &[0.0_f32; 16], &mut out).is_err());
    }
}
