//! Opt-in bounded triangular GDN chunks; no cumulative-decay division.

use super::{TileReuse, TiledPrefill, project_tile, project_tile_pair};
use crate::graph::{
    ActiveAdapters, Arithmetic, Geometry, LayerDelta, LayerState, RecurrentAttention, Transforms,
    copy_into, finite, l2_norm, resize_into, sigmoid, small_float,
};
use crate::kernels::pinned_gdn::{ChunkInput as Input, StateStepInput};
use crate::kernels::prefill::ProjectionTile;
use crate::kernels::{
    RmsNormWeights, add_scaled_slice_in_place, gdn_wy_update_eight_rows, gdn_wy_update_four_rows,
    scale_slice_into,
};
use crate::model::lora::Scratch as AdapterScratch;
use crate::{Result, invalid};

#[cfg(test)]
#[path = "prefill_gdn_exact_tests.rs"]
mod exact_tests;

#[cfg(test)]
fn head(
    arithmetic: Arithmetic,
    previous: &[f32],
    inputs: &[Input<'_>],
    matrix: &mut [f32],
    output: &mut Vec<f32>,
) -> Result<()> {
    let Some(first) = inputs.first() else {
        return Err(invalid("empty GDN chunk"));
    };
    let width = first.key.len();
    let count = inputs.len();
    resize_into(
        count
            .checked_mul(width)
            .ok_or_else(|| invalid("GDN chunk output overflow"))?,
        output,
    )?;
    head_into(arithmetic, previous, inputs, None, matrix, output)
}

const WY_CHUNK_TOKENS: usize = 32;
const KEY_DOT_STRIDE: usize = 2 * WY_CHUNK_TOKENS * WY_CHUNK_TOKENS;
const GDN_INPUT_CAPACITY: usize = if crate::kernels::prefill::MAX_TILE_TOKENS > 64 {
    crate::kernels::prefill::MAX_TILE_TOKENS
} else {
    64
};

std::thread_local! {
    static WORKER_CARRY_STATE: std::cell::RefCell<Vec<f32>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

fn with_worker_carry_state<R>(f: impl FnOnce(&mut Vec<f32>) -> R) -> R {
    WORKER_CARRY_STATE.with(|cell| {
        if let Ok(mut guard) = cell.try_borrow_mut() {
            f(&mut guard)
        } else {
            let mut fallback = Vec::new();
            f(&mut fallback)
        }
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "Sub-chunk GDN tiles larger than 32 tokens into <=32-token WY chunks with one ping-pong carry buffer."
)]
fn chunkwise_subchunks_into(
    arithmetic: Arithmetic,
    previous: &[f32],
    inputs: &[Input<'_>],
    key_dots: Option<&[f32]>,
    width: usize,
    carry_state: &mut Vec<f32>,
    matrix: &mut [f32],
    output: &mut [f32],
) -> Result<()> {
    let count = inputs.len();
    if count <= WY_CHUNK_TOKENS {
        return head_into(arithmetic, previous, inputs, key_dots, matrix, output);
    }
    let state_len = matrix.len();
    resize_into(state_len, carry_state)?;
    let num_chunks = count.div_ceil(WY_CHUNK_TOKENS);
    let mut prev_in_matrix = false;
    for chunk_idx in 0..num_chunks {
        let start = chunk_idx * WY_CHUNK_TOKENS;
        let end = (start + WY_CHUNK_TOKENS).min(count);
        let chunk_inputs = &inputs[start..end];
        let chunk_out = &mut output[start * width..end * width];
        let chunk_dots = key_dots
            .map(|dots| &dots[chunk_idx * KEY_DOT_STRIDE..(chunk_idx + 1) * KEY_DOT_STRIDE]);
        let write_to_matrix = (num_chunks - 1 - chunk_idx).is_multiple_of(2);
        if chunk_idx == 0 {
            if write_to_matrix {
                head_into(
                    arithmetic,
                    previous,
                    chunk_inputs,
                    chunk_dots,
                    matrix,
                    chunk_out,
                )?;
            } else {
                head_into(
                    arithmetic,
                    previous,
                    chunk_inputs,
                    chunk_dots,
                    carry_state,
                    chunk_out,
                )?;
            }
        } else if prev_in_matrix {
            head_into(
                arithmetic,
                matrix,
                chunk_inputs,
                chunk_dots,
                carry_state,
                chunk_out,
            )?;
        } else {
            head_into(
                arithmetic,
                carry_state,
                chunk_inputs,
                chunk_dots,
                matrix,
                chunk_out,
            )?;
        }
        prev_in_matrix = write_to_matrix;
    }
    Ok(())
}

fn fill_key_head_dots(
    arithmetic: Arithmetic,
    mixed: &[Vec<f32>],
    key_width: usize,
    width: usize,
    key_head: usize,
    dots: &mut [f32],
) -> Result<()> {
    let q_range = key_head * width..(key_head + 1) * width;
    let k_range = key_width + key_head * width..key_width + (key_head + 1) * width;
    for (token, token_mixed) in mixed.iter().enumerate() {
        let row_base = token * 32;
        let query_token = &token_mixed[q_range.clone()];
        let key_token = &token_mixed[k_range.clone()];
        let eight_blocks = token / 8;
        for b in 0..eight_blocks {
            let p = b * 8;
            let keys: [&[f32]; 8] = std::array::from_fn(|i| &mixed[p + i][k_range.clone()]);
            let pairs = arithmetic.gdn_dot_pair_eight_rows(keys, query_token, key_token)?;
            for (i, &(q_dot, k_dot)) in pairs.iter().enumerate() {
                dots[row_base + p + i] = q_dot;
                dots[1024 + row_base + p + i] = k_dot;
            }
        }
        let four_blocks = token / 4;
        for b in (eight_blocks * 2)..four_blocks {
            let p = b * 4;
            let keys = [
                &mixed[p][k_range.clone()],
                &mixed[p + 1][k_range.clone()],
                &mixed[p + 2][k_range.clone()],
                &mixed[p + 3][k_range.clone()],
            ];
            let pairs = arithmetic.gdn_dot_pair_four_rows(keys, query_token, key_token)?;
            for (i, &(q_dot, k_dot)) in pairs.iter().enumerate() {
                dots[row_base + p + i] = q_dot;
                dots[1024 + row_base + p + i] = k_dot;
            }
        }
        for (prior, prior_mixed) in mixed[..token].iter().enumerate().skip(four_blocks * 4) {
            let key_prior = &prior_mixed[k_range.clone()];
            let (q_dot, k_dot) = arithmetic.gdn_dot_pair(key_prior, query_token, key_token)?;
            dots[row_base + prior] = q_dot;
            dots[1024 + row_base + prior] = k_dot;
        }
        dots[row_base + token] = arithmetic.gdn_dot(key_token, query_token)?;
    }
    Ok(())
}

fn fill_key_head_dots_head_major(
    arithmetic: Arithmetic,
    q_head: &[f32],
    k_head: &[f32],
    count: usize,
    width: usize,
    dots: &mut [f32],
) -> Result<()> {
    for token in 0..count {
        let row_base = token * 32;
        let query_token = &q_head[token * width..(token + 1) * width];
        let key_token = &k_head[token * width..(token + 1) * width];
        let eight_blocks = token / 8;
        for b in 0..eight_blocks {
            let p = b * 8;
            let keys: [&[f32]; 8] =
                std::array::from_fn(|i| &k_head[(p + i) * width..(p + i + 1) * width]);
            let pairs = arithmetic.gdn_dot_pair_eight_rows(keys, query_token, key_token)?;
            for (i, &(q_dot, k_dot)) in pairs.iter().enumerate() {
                dots[row_base + p + i] = q_dot;
                dots[1024 + row_base + p + i] = k_dot;
            }
        }
        let four_blocks = token / 4;
        for b in (eight_blocks * 2)..four_blocks {
            let p = b * 4;
            let keys = [
                &k_head[p * width..(p + 1) * width],
                &k_head[(p + 1) * width..(p + 2) * width],
                &k_head[(p + 2) * width..(p + 3) * width],
                &k_head[(p + 3) * width..(p + 4) * width],
            ];
            let pairs = arithmetic.gdn_dot_pair_four_rows(keys, query_token, key_token)?;
            for (i, &(q_dot, k_dot)) in pairs.iter().enumerate() {
                dots[row_base + p + i] = q_dot;
                dots[1024 + row_base + p + i] = k_dot;
            }
        }
        for prior in (four_blocks * 4)..token {
            let key_prior = &k_head[prior * width..(prior + 1) * width];
            let (q_dot, k_dot) = arithmetic.gdn_dot_pair(key_prior, query_token, key_token)?;
            dots[row_base + prior] = q_dot;
            dots[1024 + row_base + prior] = k_dot;
        }
        dots[row_base + token] = arithmetic.gdn_dot(key_token, query_token)?;
    }
    Ok(())
}

fn exact_head_into(
    arithmetic: Arithmetic,
    previous: &[f32],
    inputs: &[Input<'_>],
    matrix: &mut [f32],
    output: &mut [f32],
) -> Result<()> {
    let Some(first) = inputs.first() else {
        return Err(invalid("empty GDN chunk"));
    };
    let width = first.key.len();
    let scale = 1.0 / small_float(width)?.sqrt();
    if matches!(arithmetic, Arithmetic::PinnedSwift)
        && let Some(result) = crate::kernels::pinned_gdn::try_step_head_tokens_from(
            previous, matrix, inputs, scale, output,
        )
    {
        return result;
    }
    let row_count = previous.len() / width;
    let eight_row_blocks = row_count / 8;
    for block_idx in 0..eight_row_blocks {
        let r = block_idx * 8;
        let prev_rows: [&[f32]; 8] =
            std::array::from_fn(|i| &previous[(r + i) * width..(r + i + 1) * width]);
        let state_slice = &mut matrix[r * width..(r + 8) * width];
        let (h0, h1) = state_slice.split_at_mut(4 * width);
        let (s01, s23) = h0.split_at_mut(2 * width);
        let (s45, s67) = h1.split_at_mut(2 * width);
        let (s0, s1) = s01.split_at_mut(width);
        let (s2, s3) = s23.split_at_mut(width);
        let (s4, s5) = s45.split_at_mut(width);
        let (s6, s7) = s67.split_at_mut(width);
        arithmetic.gdn_step_eight_rows_tokens_from(
            prev_rows,
            [s0, s1, s2, s3, s4, s5, s6, s7],
            inputs,
            r,
            width,
            scale,
            output,
        )?;
    }
    let four_row_blocks = row_count / 4;
    for block_idx in (eight_row_blocks * 2)..four_row_blocks {
        let r = block_idx * 4;
        let prev_rows = [
            &previous[r * width..(r + 1) * width],
            &previous[(r + 1) * width..(r + 2) * width],
            &previous[(r + 2) * width..(r + 3) * width],
            &previous[(r + 3) * width..(r + 4) * width],
        ];
        let state_slice = &mut matrix[r * width..(r + 4) * width];
        let (s01, s23) = state_slice.split_at_mut(2 * width);
        let (s0, s1) = s01.split_at_mut(width);
        let (s2, s3) = s23.split_at_mut(width);
        arithmetic.gdn_step_four_rows_tokens_from(
            prev_rows,
            [s0, s1, s2, s3],
            inputs,
            r,
            width,
            scale,
            output,
        )?;
    }
    for r in (four_row_blocks * 4)..row_count {
        let prev_row = &previous[r * width..(r + 1) * width];
        let state_row = &mut matrix[r * width..(r + 1) * width];
        arithmetic
            .gdn_step_row_tokens_from(prev_row, state_row, inputs, r, width, scale, output)?;
    }
    Ok(())
}

#[expect(
    clippy::suboptimal_flops,
    clippy::too_many_arguments,
    reason = "Passes precomputed triangular WY tables and preserves exact separate FP32 multiply/add rounding."
)]
fn head_into_eight_row_block_scalar(
    arithmetic: Arithmetic,
    states: [&[f32]; 8],
    inputs: &[Input<'_>],
    cumulative: &[f32; 32],
    cross_key: &[[f32; 32]; 32],
    cross_query: &[[f32; 32]; 32],
    last_products: &[f32; 32],
    base_row: usize,
    width: usize,
    scale: f32,
    output: &mut [f32],
    scaled_deltas: &mut [[f32; 8]; 32],
) -> Result<()> {
    let count = inputs.len();
    let mut deltas = [[0.0_f32; 32]; 8];
    for token in 0..count {
        let pairs =
            arithmetic.gdn_dot_pair_eight_rows(states, inputs[token].key, inputs[token].query)?;
        let cum = cumulative[token];
        let beta = inputs[token].beta;
        let out_base = token * width + base_row;
        for sub in 0..8 {
            let (key_dot, query_dot) = pairs[sub];
            let mut predicted = cum * key_dot;
            for (prior, &delta) in deltas[sub][..token].iter().enumerate() {
                predicted += cross_key[token][prior] * delta;
            }
            deltas[sub][token] = (inputs[token].value[base_row + sub] - predicted) * beta;
            let mut value = cum * query_dot;
            for (prior, &delta) in deltas[sub][..=token].iter().enumerate() {
                value += cross_query[token][prior] * delta;
            }
            output[out_base + sub] = value * scale;
        }
    }
    for (prior, (&lp, sd_row)) in last_products[..count]
        .iter()
        .zip(&mut scaled_deltas[..count])
        .enumerate()
    {
        for sub in 0..8 {
            sd_row[sub] = lp * deltas[sub][prior];
        }
    }
    Ok(())
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
fn collect_eight_row_state_dots(
    arithmetic: Arithmetic,
    states: [&[f32]; 8],
    inputs: &[Input<'_>],
    key_dots: &mut [[f32; 8]; 32],
    query_dots: &mut [[f32; 8]; 32],
) -> Result<()> {
    match arithmetic {
        Arithmetic::PinnedSwift => {
            let count = inputs.len();
            let mut pairs = [(&[][..], &[][..]); 32];
            for (slot, input) in pairs[..count].iter_mut().zip(inputs) {
                *slot = (input.key, input.query);
            }
            crate::kernels::pinned_gdn::dot_pair_eight_rows_tokens_major(
                states,
                &pairs[..count],
                key_dots,
                query_dots,
            )
        }
        Arithmetic::Source => {
            for (token, input) in inputs.iter().enumerate() {
                let pairs = arithmetic.gdn_dot_pair_eight_rows(states, input.key, input.query)?;
                for row in 0..8 {
                    key_dots[token][row] = pairs[row].0;
                    query_dots[token][row] = pairs[row].1;
                }
            }
            Ok(())
        }
    }
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
#[expect(
    clippy::too_many_arguments,
    reason = "Matches the scalar 8-row triangular WY helper contract without altering buffer ownership."
)]
fn head_into_eight_row_block_neon(
    arithmetic: Arithmetic,
    states: [&[f32]; 8],
    inputs: &[Input<'_>],
    cumulative: &[f32; 32],
    cross_key: &[[f32; 32]; 32],
    cross_query: &[[f32; 32]; 32],
    last_products: &[f32; 32],
    base_row: usize,
    width: usize,
    scale: f32,
    output: &mut [f32],
    scaled_deltas: &mut [[f32; 8]; 32],
) -> Result<bool> {
    use fearless_simd::{f32x4, prelude::*};

    let Some(neon) = fearless_simd::Level::baseline().as_neon() else {
        return Ok(false);
    };
    let count = inputs.len();
    let zero = f32x4::splat(neon, 0.0);
    let mut deltas_lo = [zero; 32];
    let mut deltas_hi = [zero; 32];
    let scale_vec = f32x4::splat(neon, scale);
    let mut key_dots = [[0.0_f32; 8]; 32];
    let mut query_dots = [[0.0_f32; 8]; 32];
    collect_eight_row_state_dots(arithmetic, states, inputs, &mut key_dots, &mut query_dots)?;

    for token in 0..count {
        let cum_vec = f32x4::splat(neon, cumulative[token]);
        let beta_vec = f32x4::splat(neon, inputs[token].beta);
        let key_dots_lo = f32x4::from_slice(neon, &key_dots[token][0..4]);
        let key_dots_hi = f32x4::from_slice(neon, &key_dots[token][4..8]);
        let query_dots_lo = f32x4::from_slice(neon, &query_dots[token][0..4]);
        let query_dots_hi = f32x4::from_slice(neon, &query_dots[token][4..8]);

        let mut pred_lo = cum_vec * key_dots_lo;
        let mut pred_hi = cum_vec * key_dots_hi;
        let mut val_lo = cum_vec * query_dots_lo;
        let mut val_hi = cum_vec * query_dots_hi;
        for (((&ck_scalar, &cq_scalar), &d_lo), &d_hi) in cross_key[token][..token]
            .iter()
            .zip(&cross_query[token][..token])
            .zip(&deltas_lo[..token])
            .zip(&deltas_hi[..token])
        {
            let ck = f32x4::splat(neon, ck_scalar);
            let cq = f32x4::splat(neon, cq_scalar);
            pred_lo += ck * d_lo;
            pred_hi += ck * d_hi;
            val_lo += cq * d_lo;
            val_hi += cq * d_hi;
        }

        let target_vals = &inputs[token].value[base_row..base_row + 8];
        let v_lo = f32x4::from_slice(neon, &target_vals[0..4]);
        let v_hi = f32x4::from_slice(neon, &target_vals[4..8]);
        let d_tok_lo = (v_lo - pred_lo) * beta_vec;
        let d_tok_hi = (v_hi - pred_hi) * beta_vec;
        deltas_lo[token] = d_tok_lo;
        deltas_hi[token] = d_tok_hi;

        let cq_self = f32x4::splat(neon, cross_query[token][token]);
        let out_lo = (val_lo + (cq_self * d_tok_lo)) * scale_vec;
        let out_hi = (val_hi + (cq_self * d_tok_hi)) * scale_vec;
        let out_base = token * width + base_row;
        out_lo.store_slice(&mut output[out_base..out_base + 4]);
        out_hi.store_slice(&mut output[out_base + 4..out_base + 8]);
    }
    for (((sd_row, &lp_scalar), &d_lo), &d_hi) in scaled_deltas[..count]
        .iter_mut()
        .zip(&last_products[..count])
        .zip(&deltas_lo[..count])
        .zip(&deltas_hi[..count])
    {
        let lp = f32x4::splat(neon, lp_scalar);
        (lp * d_lo).store_slice(&mut sd_row[0..4]);
        (lp * d_hi).store_slice(&mut sd_row[4..8]);
    }
    Ok(true)
}

#[expect(
    clippy::suboptimal_flops,
    clippy::too_many_lines,
    reason = "The experimental FP32 chunk contract uses explicit separately rounded triangular arithmetic."
)]
fn head_into(
    arithmetic: Arithmetic,
    previous: &[f32],
    inputs: &[Input<'_>],
    key_dots: Option<&[f32]>,
    matrix: &mut [f32],
    output: &mut [f32],
) -> Result<()> {
    let Some(first) = inputs.first() else {
        return Err(invalid("empty GDN chunk"));
    };
    let width = first.key.len();
    let count = inputs.len();
    if count > 32
        || width == 0
        || width.checked_mul(width) != Some(previous.len())
        || matrix.len() != previous.len()
        || count.checked_mul(width) != Some(output.len())
        || inputs.iter().any(|input| {
            input.key.len() != width
                || input.query.len() != width
                || input.value.len() != width
                || !input.decay.is_finite()
                || !input.beta.is_finite()
                || !(0.0..=1.0).contains(&input.decay)
                || !(0.0..=1.0).contains(&input.beta)
        })
    {
        return Err(invalid("invalid GDN chunk geometry or gates"));
    }
    // products[t][j] is decay over (j,t], with an exact empty product of one.
    // This remains defined if a decay or a longer product underflows to zero.
    let mut products = [[0.0_f32; 32]; 32];
    let mut cumulative = [0.0_f32; 32];
    let mut cross_key = [[0.0_f32; 32]; 32];
    let mut cross_query = [[0.0_f32; 32]; 32];
    let mut decay = 1.0;
    for token in 0..count {
        decay *= inputs[token].decay;
        cumulative[token] = decay;
        products[token][token] = 1.0;
        for prior in (0..token).rev() {
            products[token][prior] = products[token][prior + 1] * inputs[prior + 1].decay;
        }
        if let Some(dots) = key_dots {
            let row_base = token * 32;
            for prior in 0..=token {
                cross_query[token][prior] = products[token][prior] * dots[row_base + prior];
                if prior < token {
                    cross_key[token][prior] =
                        products[token][prior] * dots[1024 + row_base + prior];
                }
            }
        } else {
            let eight_blocks = token / 8;
            for b in 0..eight_blocks {
                let p = b * 8;
                let keys: [&[f32]; 8] = std::array::from_fn(|i| inputs[p + i].key);
                let pairs = arithmetic.gdn_dot_pair_eight_rows(
                    keys,
                    inputs[token].query,
                    inputs[token].key,
                )?;
                for (i, &(q_dot, k_dot)) in pairs.iter().enumerate() {
                    let prior = p + i;
                    cross_query[token][prior] = products[token][prior] * q_dot;
                    cross_key[token][prior] = products[token][prior] * k_dot;
                }
            }
            let four_blocks = token / 4;
            for b in (eight_blocks * 2)..four_blocks {
                let p = b * 4;
                let keys = [
                    inputs[p].key,
                    inputs[p + 1].key,
                    inputs[p + 2].key,
                    inputs[p + 3].key,
                ];
                let pairs = arithmetic.gdn_dot_pair_four_rows(
                    keys,
                    inputs[token].query,
                    inputs[token].key,
                )?;
                for (i, &(q_dot, k_dot)) in pairs.iter().enumerate() {
                    let prior = p + i;
                    cross_query[token][prior] = products[token][prior] * q_dot;
                    cross_key[token][prior] = products[token][prior] * k_dot;
                }
            }
            for prior in (four_blocks * 4)..token {
                let (q_dot, k_dot) = arithmetic.gdn_dot_pair(
                    inputs[prior].key,
                    inputs[token].query,
                    inputs[token].key,
                )?;
                cross_query[token][prior] = products[token][prior] * q_dot;
                cross_key[token][prior] = products[token][prior] * k_dot;
            }
            cross_query[token][token] = products[token][token]
                * arithmetic.gdn_dot(inputs[token].key, inputs[token].query)?;
        }
    }
    let scale = 1.0 / small_float(width)?.sqrt();
    let last = count - 1;
    let last_cumulative = cumulative[last];
    let eight_row_blocks = width / 8;
    for block_idx in 0..eight_row_blocks {
        let base_row = block_idx * 8;
        let states: [&[f32]; 8] =
            std::array::from_fn(|r| &previous[(base_row + r) * width..(base_row + r + 1) * width]);
        let mut scaled_deltas = [[0.0_f32; 8]; 32];
        #[cfg(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc",
            target_feature = "neon"
        ))]
        let handled = head_into_eight_row_block_neon(
            arithmetic,
            states,
            inputs,
            &cumulative,
            &cross_key,
            &cross_query,
            &products[last],
            base_row,
            width,
            scale,
            output,
            &mut scaled_deltas,
        )?;
        #[cfg(not(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc",
            target_feature = "neon"
        )))]
        let handled = false;
        if !handled {
            head_into_eight_row_block_scalar(
                arithmetic,
                states,
                inputs,
                &cumulative,
                &cross_key,
                &cross_query,
                &products[last],
                base_row,
                width,
                scale,
                output,
                &mut scaled_deltas,
            )?;
        }
        let matrix_slice = &mut matrix[base_row * width..(base_row + 8) * width];
        let (h0, h1) = matrix_slice.split_at_mut(4 * width);
        let (m01, m23) = h0.split_at_mut(2 * width);
        let (m45, m67) = h1.split_at_mut(2 * width);
        let (m0, m1) = m01.split_at_mut(width);
        let (m2, m3) = m23.split_at_mut(width);
        let (m4, m5) = m45.split_at_mut(width);
        let (m6, m7) = m67.split_at_mut(width);
        if matches!(arithmetic, Arithmetic::PinnedSwift) {
            crate::kernels::pinned_gdn::gdn_wy_update_eight_rows_token_major(
                states,
                last_cumulative,
                inputs,
                &scaled_deltas,
                [m0, m1, m2, m3, m4, m5, m6, m7],
            );
        } else {
            let mut row_major_deltas = [[0.0_f32; 32]; 8];
            for (prior, sd_row) in scaled_deltas[..count].iter().enumerate() {
                for (row_deltas, &val) in row_major_deltas.iter_mut().zip(sd_row) {
                    row_deltas[prior] = val;
                }
            }
            gdn_wy_update_eight_rows(
                states,
                last_cumulative,
                inputs,
                &row_major_deltas,
                [m0, m1, m2, m3, m4, m5, m6, m7],
            );
        }
    }
    let four_row_blocks = width / 4;
    for block_idx in (eight_row_blocks * 2)..four_row_blocks {
        let base_row = block_idx * 4;
        let states = [
            &previous[base_row * width..(base_row + 1) * width],
            &previous[(base_row + 1) * width..(base_row + 2) * width],
            &previous[(base_row + 2) * width..(base_row + 3) * width],
            &previous[(base_row + 3) * width..(base_row + 4) * width],
        ];
        let mut deltas = [[0.0_f32; 32]; 4];
        for token in 0..count {
            let pairs = arithmetic.gdn_dot_pair_four_rows(
                states,
                inputs[token].key,
                inputs[token].query,
            )?;
            let cum = cumulative[token];
            let beta = inputs[token].beta;
            let out_base = token * width + base_row;
            for sub in 0..4 {
                let (key_dot, query_dot) = pairs[sub];
                let mut predicted = cum * key_dot;
                for (prior, &delta) in deltas[sub][..token].iter().enumerate() {
                    predicted += cross_key[token][prior] * delta;
                }
                deltas[sub][token] = (inputs[token].value[base_row + sub] - predicted) * beta;
                let mut value = cum * query_dot;
                for (prior, &delta) in deltas[sub][..=token].iter().enumerate() {
                    value += cross_query[token][prior] * delta;
                }
                output[out_base + sub] = value * scale;
            }
        }
        let mut scaled_deltas = [[0.0_f32; 32]; 4];
        for sub in 0..4 {
            for (prior, &delta) in deltas[sub][..count].iter().enumerate() {
                scaled_deltas[sub][prior] = products[last][prior] * delta;
            }
        }
        let matrix_slice = &mut matrix[base_row * width..(base_row + 4) * width];
        let (m01, m23) = matrix_slice.split_at_mut(2 * width);
        let (m0, m1) = m01.split_at_mut(width);
        let (m2, m3) = m23.split_at_mut(width);
        gdn_wy_update_four_rows(
            states,
            last_cumulative,
            inputs,
            &scaled_deltas,
            [m0, m1, m2, m3],
        );
    }
    for row in (four_row_blocks * 4)..width {
        let state = &previous[row * width..(row + 1) * width];
        let mut deltas = [0.0_f32; 32];
        for token in 0..count {
            let (key_dot, query_dot) =
                arithmetic.gdn_dot_pair(state, inputs[token].key, inputs[token].query)?;
            let mut predicted = cumulative[token] * key_dot;
            for (prior, &delta) in deltas[..token].iter().enumerate() {
                predicted += cross_key[token][prior] * delta;
            }
            deltas[token] = (inputs[token].value[row] - predicted) * inputs[token].beta;
            let mut value = cumulative[token] * query_dot;
            for (prior, &delta) in deltas[..=token].iter().enumerate() {
                value += cross_query[token][prior] * delta;
            }
            output[token * width + row] = value * scale;
        }
        let matrix_row = &mut matrix[row * width..(row + 1) * width];
        scale_slice_into(state, last_cumulative, matrix_row);
        for (prior, &delta) in deltas[..count].iter().enumerate() {
            let scaled_delta = products[last][prior] * delta;
            add_scaled_slice_in_place(matrix_row, inputs[prior].key, scaled_delta);
        }
    }
    finite(matrix, "chunkwise GDN state")?;
    finite(output, "chunkwise GDN output")
}

/// Preallocated per-layer input tape for folded speculative verification replay.
#[derive(Clone, Debug, Default)]
pub(super) struct FoldedRecurrentLayerTape {
    conv_width: usize,
    key_width: usize,
    value_width: usize,
    ssm_keys: usize,
    ssm_values: usize,
    ssm_head: usize,
    pending_len: usize,
    pending_raw_conv: Vec<f32>,
    pending_keys: Vec<f32>,
    pending_values: Vec<f32>,
    pending_decay: Vec<f32>,
    pending_beta: Vec<f32>,
    carried_len: usize,
    carried_raw_conv: Vec<f32>,
    carried_keys: Vec<f32>,
    carried_values: Vec<f32>,
    carried_decay: Vec<f32>,
    carried_beta: Vec<f32>,
}

impl FoldedRecurrentLayerTape {
    /// Conservative heap byte footprint of one layer's folded verification tape.
    pub(super) fn bytes_for_capacity(max_tokens: usize, geometry: &Geometry) -> Result<usize> {
        let per_token_floats = geometry
            .convolution_width()
            .checked_add(geometry.key_width())
            .and_then(|v| v.checked_add(geometry.value_width()))
            .and_then(|v| v.checked_add(geometry.ssm_values.saturating_mul(2)))
            .ok_or_else(|| invalid("folded GDN tape float count overflow"))?;
        let total_floats = per_token_floats
            .checked_mul(max_tokens)
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| invalid("folded GDN tape token capacity overflow"))?;
        total_floats
            .checked_mul(size_of::<f32>())
            .and_then(|v| v.checked_add(size_of::<Self>()))
            .ok_or_else(|| invalid("folded GDN tape byte size overflow"))
    }

    pub(super) fn new(max_tokens: usize, geometry: &Geometry) -> Result<Self> {
        let conv_width = geometry.convolution_width();
        let key_width = geometry.key_width();
        let value_width = geometry.value_width();
        let ssm_values = geometry.ssm_values;
        let alloc = |width: usize| -> Result<Vec<f32>> {
            let mut vec = Vec::new();
            let cap = width
                .checked_mul(max_tokens)
                .ok_or_else(|| invalid("folded GDN tape buffer overflow"))?;
            vec.try_reserve_exact(cap)?;
            Ok(vec)
        };
        Ok(Self {
            conv_width,
            key_width,
            value_width,
            ssm_keys: geometry.ssm_keys,
            ssm_values,
            ssm_head: geometry.ssm_head,
            pending_len: 0,
            pending_raw_conv: alloc(conv_width)?,
            pending_keys: alloc(key_width)?,
            pending_values: alloc(value_width)?,
            pending_decay: alloc(ssm_values)?,
            pending_beta: alloc(ssm_values)?,
            carried_len: 0,
            carried_raw_conv: alloc(conv_width)?,
            carried_keys: alloc(key_width)?,
            carried_values: alloc(value_width)?,
            carried_decay: alloc(ssm_values)?,
            carried_beta: alloc(ssm_values)?,
        })
    }

    pub(super) const fn has_pending(&self) -> bool {
        self.pending_len > 0
    }

    pub(super) const fn has_carried(&self) -> bool {
        self.carried_len > 0
    }

    pub(super) fn clear_pending(&mut self) {
        self.pending_len = 0;
        self.pending_raw_conv.clear();
        self.pending_keys.clear();
        self.pending_values.clear();
        self.pending_decay.clear();
        self.pending_beta.clear();
    }

    pub(super) fn clear_carried(&mut self) {
        self.carried_len = 0;
        self.carried_raw_conv.clear();
        self.carried_keys.clear();
        self.carried_values.clear();
        self.carried_decay.clear();
        self.carried_beta.clear();
    }

    pub(super) fn clear_all(&mut self) {
        self.clear_pending();
        self.clear_carried();
    }

    pub(super) fn validate_pending_promotion(&self, kept: usize) -> Result<()> {
        if self.carried_len != 0 {
            return Err(invalid(
                "cannot commit pending folded verification before flushing prior carry",
            ));
        }
        if kept > self.pending_len {
            return Err(invalid(
                "folded verification kept token count exceeds pending block length",
            ));
        }
        if kept > 0 {
            let conv_needed = kept
                .checked_mul(self.conv_width)
                .ok_or_else(|| invalid("folded raw convolution tape overflow"))?;
            let key_needed = kept
                .checked_mul(self.key_width)
                .ok_or_else(|| invalid("folded key tape overflow"))?;
            let value_needed = kept
                .checked_mul(self.value_width)
                .ok_or_else(|| invalid("folded value tape overflow"))?;
            let gate_needed = kept
                .checked_mul(self.ssm_values)
                .ok_or_else(|| invalid("folded gate tape overflow"))?;
            if self.pending_raw_conv.len() < conv_needed
                || self.pending_keys.len() < key_needed
                || self.pending_values.len() < value_needed
                || self.pending_decay.len() < gate_needed
                || self.pending_beta.len() < gate_needed
            {
                return Err(invalid("folded recurrent pending tape truncated"));
            }
        }
        Ok(())
    }

    pub(super) fn promote_pending_to_carried(&mut self, kept: usize) -> Result<()> {
        self.validate_pending_promotion(kept)?;
        if kept > 0 {
            std::mem::swap(&mut self.carried_raw_conv, &mut self.pending_raw_conv);
            self.carried_raw_conv.truncate(kept * self.conv_width);
            std::mem::swap(&mut self.carried_keys, &mut self.pending_keys);
            self.carried_keys.truncate(kept * self.key_width);
            std::mem::swap(&mut self.carried_values, &mut self.pending_values);
            self.carried_values.truncate(kept * self.value_width);
            std::mem::swap(&mut self.carried_decay, &mut self.pending_decay);
            self.carried_decay.truncate(kept * self.ssm_values);
            std::mem::swap(&mut self.carried_beta, &mut self.pending_beta);
            self.carried_beta.truncate(kept * self.ssm_values);
            self.carried_len = kept;
        }
        self.clear_pending();
        Ok(())
    }

    fn capture_raw_conv(&mut self, mixed: &[Vec<f32>]) -> Result<()> {
        self.clear_pending();
        let count = mixed.len();
        let needed = count
            .checked_mul(self.conv_width)
            .ok_or_else(|| invalid("folded raw convolution tape overflow"))?;
        if needed > self.pending_raw_conv.capacity() {
            self.pending_raw_conv
                .try_reserve(needed - self.pending_raw_conv.capacity())?;
        }
        for row in mixed {
            let slice = row
                .get(..self.conv_width)
                .ok_or_else(|| invalid("folded raw convolution width mismatch"))?;
            self.pending_raw_conv.extend_from_slice(slice);
        }
        Ok(())
    }

    fn capture_gdn_inputs(
        &mut self,
        count: usize,
        conv_head_major: bool,
        gdn_conv_outputs: &[f32],
        mixed: &[Vec<f32>],
        alpha: &[Vec<f32>],
        beta: &[Vec<f32>],
    ) -> Result<()> {
        self.pending_keys.clear();
        self.pending_values.clear();
        self.pending_decay.clear();
        self.pending_beta.clear();
        if self.ssm_keys.checked_mul(self.ssm_head) != Some(self.key_width)
            || self.ssm_values.checked_mul(self.ssm_head) != Some(self.value_width)
        {
            return Err(invalid("folded GDN key/value geometry mismatch"));
        }
        let head_out_len = count * self.ssm_head;
        for token in 0..count {
            if conv_head_major {
                for key_head in 0..self.ssm_keys {
                    let conv_head = self.ssm_keys + key_head;
                    let start = conv_head * head_out_len + token * self.ssm_head;
                    let slice = gdn_conv_outputs
                        .get(start..start + self.ssm_head)
                        .ok_or_else(|| invalid("folded head-major key slice out of bounds"))?;
                    self.pending_keys.extend_from_slice(slice);
                }
                for val_head in 0..self.ssm_values {
                    let conv_head = 2 * self.ssm_keys + val_head;
                    let start = conv_head * head_out_len + token * self.ssm_head;
                    let slice = gdn_conv_outputs
                        .get(start..start + self.ssm_head)
                        .ok_or_else(|| invalid("folded head-major value slice out of bounds"))?;
                    self.pending_values.extend_from_slice(slice);
                }
            } else {
                let row = mixed
                    .get(token)
                    .ok_or_else(|| invalid("folded token mixed row missing"))?;
                let keys = row
                    .get(self.key_width..2 * self.key_width)
                    .ok_or_else(|| invalid("folded token-major key slice out of bounds"))?;
                let values = row
                    .get(2 * self.key_width..2 * self.key_width + self.value_width)
                    .ok_or_else(|| invalid("folded token-major value slice out of bounds"))?;
                self.pending_keys.extend_from_slice(keys);
                self.pending_values.extend_from_slice(values);
            }
            let decay_row = alpha
                .get(token)
                .and_then(|row| row.get(..self.ssm_values))
                .ok_or_else(|| invalid("folded GDN decay row out of bounds"))?;
            let beta_row = beta
                .get(token)
                .and_then(|row| row.get(..self.ssm_values))
                .ok_or_else(|| invalid("folded GDN beta row out of bounds"))?;
            self.pending_decay.extend_from_slice(decay_row);
            self.pending_beta.extend_from_slice(beta_row);
        }
        self.pending_len = count;
        Ok(())
    }
}

fn advance_convolution_history(
    history: &mut [f32],
    raw_tokens: &[f32],
    conv_width: usize,
    hist_width: usize,
    kept: usize,
) -> Result<()> {
    if hist_width == 0 || kept == 0 {
        return Ok(());
    }
    if history.len() != conv_width * hist_width || raw_tokens.len() < kept * conv_width {
        return Err(invalid("folded convolution history dimension mismatch"));
    }
    for channel in 0..conv_width {
        let slot = &mut history[channel * hist_width..(channel + 1) * hist_width];
        if kept >= hist_width {
            let start = kept - hist_width;
            for (offset, dest) in slot.iter_mut().enumerate() {
                *dest = raw_tokens[(start + offset) * conv_width + channel];
            }
        } else {
            slot.rotate_left(kept);
            for offset in 0..kept {
                slot[hist_width - kept + offset] = raw_tokens[offset * conv_width + channel];
            }
        }
    }
    Ok(())
}

fn flush_carry_single_head(
    layer_tape: &FoldedRecurrentLayerTape,
    head_index: usize,
    kept: usize,
    arithmetic: Arithmetic,
    prev_head: &[f32],
    head_matrix: &mut [f32],
) -> Result<()> {
    let width = layer_tape.ssm_head;
    let num_keys = layer_tape.ssm_keys;
    let num_heads = layer_tape.ssm_values;
    let key_width = layer_tape.key_width;
    let value_width = layer_tape.value_width;
    let mut step_inputs = [StateStepInput {
        key: &[],
        value: &[],
        decay: 0.0,
        beta: 0.0,
    }; crate::speculative::MAX_SPECULATIVE_DRAFT_TOKENS];
    let key_head = head_index % num_keys;
    for (token, slot) in step_inputs[..kept].iter_mut().enumerate() {
        let k_start = token * key_width + key_head * width;
        let v_start = token * value_width + head_index * width;
        let decay = layer_tape.carried_decay[token * num_heads + head_index];
        let beta = layer_tape.carried_beta[token * num_heads + head_index];
        if !decay.is_finite()
            || !beta.is_finite()
            || !(0.0..=1.0).contains(&decay)
            || !(0.0..=1.0).contains(&beta)
        {
            return Err(invalid("invalid carried GDN gate bounds"));
        }
        *slot = StateStepInput {
            key: &layer_tape.carried_keys[k_start..k_start + width],
            value: &layer_tape.carried_values[v_start..v_start + width],
            decay,
            beta,
        };
    }
    if matches!(arithmetic, Arithmetic::PinnedSwift) && width >= 4 {
        super::step_recurrent_state_only_head_pinned(
            arithmetic,
            &step_inputs[..kept],
            width,
            prev_head,
            head_matrix,
        )
    } else {
        super::step_recurrent_state_only_rows_single(
            arithmetic,
            &step_inputs[..kept],
            0,
            width,
            prev_head,
            head_matrix,
        )
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Head replay consumes disjoint slices from the folded layer tape and parallel worker pool."
)]
fn flush_carry_heads(
    layer_tape: &FoldedRecurrentLayerTape,
    previous: &[f32],
    matrix: &mut [f32],
    geometry: &Geometry,
    arithmetic: Arithmetic,
    projector: &ProjectionTile,
    _head_outputs: &mut Vec<f32>,
    kept: usize,
) -> Result<()> {
    let width = geometry.ssm_head;
    let head_state_len = width * width;
    let num_heads = geometry.ssm_values;

    if projector.row_parallelism().workers() > 1 && num_heads > 1 {
        return projector.row_parallelism().update_chunks(
            matrix,
            head_state_len,
            |base_head, matrix_chunk| {
                for (offset, head_matrix) in
                    matrix_chunk.chunks_exact_mut(head_state_len).enumerate()
                {
                    let head_index = base_head + offset;
                    let prev_head =
                        &previous[head_index * head_state_len..(head_index + 1) * head_state_len];
                    flush_carry_single_head(
                        layer_tape,
                        head_index,
                        kept,
                        arithmetic,
                        prev_head,
                        head_matrix,
                    )?;
                }
                Ok(())
            },
        );
    }
    for (head_index, head_matrix) in matrix.chunks_exact_mut(head_state_len).enumerate() {
        let range = head_index * head_state_len..(head_index + 1) * head_state_len;
        flush_carry_single_head(
            layer_tape,
            head_index,
            kept,
            arithmetic,
            &previous[range],
            head_matrix,
        )?;
    }
    Ok(())
}

/// Replay deferred carried recurrent tokens into `state` without running layer projections.
pub(super) fn flush_recurrent_carry_layer(
    layer_tape: &mut FoldedRecurrentLayerTape,
    state: &mut LayerState,
    delta: &mut LayerDelta,
    geometry: &Geometry,
    arithmetic: Arithmetic,
    projector: &ProjectionTile,
    head_outputs: &mut Vec<f32>,
) -> Result<()> {
    let kept = layer_tape.carried_len;
    if kept == 0 {
        return Ok(());
    }
    if kept > crate::speculative::MAX_SPECULATIVE_DRAFT_TOKENS {
        return Err(invalid("folded carried token count exceeds draft limit"));
    }
    let (
        LayerState::Recurrent {
            convolution: history,
            matrix: previous,
        },
        LayerDelta::Recurrent {
            convolution: next_history,
            matrix,
        },
    ) = (state, delta)
    else {
        return Err(invalid("folded GDN carry flush requires recurrent state"));
    };
    let hist_width = geometry.convolution.saturating_sub(1);
    resize_into(history.len(), next_history)?;
    next_history.copy_from_slice(history);
    advance_convolution_history(
        next_history,
        &layer_tape.carried_raw_conv,
        geometry.convolution_width(),
        hist_width,
        kept,
    )?;
    finite(next_history, "folded convolution history")?;
    resize_into(previous.len(), matrix)?;
    flush_carry_heads(
        layer_tape,
        previous,
        matrix,
        geometry,
        arithmetic,
        projector,
        head_outputs,
        kept,
    )?;
    finite(matrix, "folded GDN state matrix")?;
    std::mem::swap(history, next_history);
    std::mem::swap(previous, matrix);
    layer_tape.clear_carried();
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "Keep tiled recurrent forward signature explicit alongside folded replay tape."
)]
pub(super) fn recurrent_tile<'a>(
    layer: &'a RecurrentAttention,
    state: &mut LayerState,
    delta: &mut LayerDelta,
    geometry: &Geometry,
    arithmetic: Arithmetic,
    transforms: &Transforms,
    adapters: ActiveAdapters<'_>,
    adapter_scratch: &mut AdapterScratch,
    projection_scratch: &mut crate::kernels::MatvecScratch,
    previous_norm: TileReuse<'a>,
    count: usize,
    check: &dyn Fn() -> Result<()>,
    tile: &mut TiledPrefill,
    executor: Option<&mut (dyn crate::kernels::Pq2ProjectionExecutor + '_)>,
) -> Result<()> {
    recurrent_tile_with_tape(
        layer,
        state,
        delta,
        geometry,
        arithmetic,
        transforms,
        adapters,
        adapter_scratch,
        projection_scratch,
        previous_norm,
        count,
        check,
        tile,
        executor,
        None,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "Keep folded recurrent forward aligned with standard tiled GDN execution."
)]
pub(super) fn recurrent_tile_folded<'a>(
    layer: &'a RecurrentAttention,
    state: &mut LayerState,
    delta: &mut LayerDelta,
    geometry: &Geometry,
    arithmetic: Arithmetic,
    transforms: &Transforms,
    adapters: ActiveAdapters<'_>,
    adapter_scratch: &mut AdapterScratch,
    projection_scratch: &mut crate::kernels::MatvecScratch,
    previous_norm: TileReuse<'a>,
    count: usize,
    check: &dyn Fn() -> Result<()>,
    tile: &mut TiledPrefill,
    executor: Option<&mut (dyn crate::kernels::Pq2ProjectionExecutor + '_)>,
    folded: &mut FoldedRecurrentLayerTape,
) -> Result<()> {
    recurrent_tile_with_tape(
        layer,
        state,
        delta,
        geometry,
        arithmetic,
        transforms,
        adapters,
        adapter_scratch,
        projection_scratch,
        previous_norm,
        count,
        check,
        tile,
        executor,
        Some(folded),
    )
}

#[inline]
#[expect(
    clippy::imprecise_flops,
    reason = "Match exact binary32 (1.0 + exp(x)).ln() softplus rounding."
)]
fn gdn_decay_beta(raw_alpha: f32, raw_beta: f32, time_bias: f32, decay_weight: f32) -> (f32, f32) {
    let activation = raw_alpha + time_bias;
    let softplus = if activation > 20.0 {
        activation
    } else {
        (1.0 + activation.exp()).ln()
    };
    ((softplus * decay_weight).exp(), sigmoid(raw_beta))
}

#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "Share exact GDN convolution, gating, and state stepping between standard and folded tiles."
)]
fn recurrent_tile_with_tape<'a>(
    layer: &'a RecurrentAttention,
    state: &mut LayerState,
    delta: &mut LayerDelta,
    geometry: &Geometry,
    arithmetic: Arithmetic,
    transforms: &Transforms,
    adapters: ActiveAdapters<'_>,
    adapter_scratch: &mut AdapterScratch,
    projection_scratch: &mut crate::kernels::MatvecScratch,
    previous_norm: TileReuse<'a>,
    count: usize,
    check: &dyn Fn() -> Result<()>,
    tile: &mut TiledPrefill,
    mut executor: Option<&mut (dyn crate::kernels::Pq2ProjectionExecutor + '_)>,
    mut folded: Option<&mut FoldedRecurrentLayerTape>,
) -> Result<()> {
    let (
        LayerState::Recurrent {
            convolution: history,
            matrix: previous,
        },
        LayerDelta::Recurrent {
            convolution,
            matrix,
        },
    ) = (state, delta)
    else {
        return Err(invalid("chunkwise GDN requires recurrent state"));
    };
    if count > GDN_INPUT_CAPACITY {
        return Err(invalid("GDN tile count exceeds static input capacity"));
    }
    check()?;
    let previous_weight = project_tile_pair(
        transforms,
        geometry,
        arithmetic,
        adapters,
        adapter_scratch,
        projection_scratch,
        previous_norm,
        [&layer.qkv, &layer.gate],
        &tile.normalized[..count],
        &mut tile.mixed[..count],
        &mut tile.recurrent_gate[..count],
        &mut tile.transformed[..count],
        &mut tile.projector,
        executor.as_deref_mut(),
        check,
    )?;
    check()?;
    let _ = project_tile_pair(
        transforms,
        geometry,
        arithmetic,
        adapters,
        adapter_scratch,
        projection_scratch,
        previous_weight,
        [&layer.beta, &layer.alpha],
        &tile.normalized[..count],
        &mut tile.beta[..count],
        &mut tile.alpha[..count],
        &mut tile.transformed[..count],
        &mut tile.projector,
        executor.as_deref_mut(),
        check,
    )?;
    let width = geometry.ssm_head;
    let hist_width = geometry.convolution - 1;
    let conv_aligned = hist_width > 0
        && geometry.convolution_width().is_multiple_of(width)
        && width.is_multiple_of(4);
    let parallelize_conv = conv_aligned && tile.projector.row_parallelism().workers() > 1;
    let parallelize_post_conv = tile.projector.row_parallelism().workers() > 1 && count > 1;
    let conv_head_major = conv_aligned && count > 1;
    if let Some(layer_tape) = folded.as_deref_mut() {
        layer_tape.capture_raw_conv(&tile.mixed[..count])?;
    }
    if conv_head_major && matches!(arithmetic, Arithmetic::PinnedSwift) {
        resize_into(history.len(), convolution)?;
    } else {
        copy_into(history, convolution)?;
    }
    resize_into(previous.len(), matrix)?;
    if conv_head_major {
        for attended in tile.attended.iter_mut().take(count) {
            check()?;
            resize_into(geometry.value_width(), attended)?;
            check()?;
        }
        let head_hist = width * hist_width;
        let head_coeff = width * geometry.convolution;
        let head_out_len = count * width;
        let num_conv_heads = geometry.convolution_width() / width;
        let qk_heads = (2 * geometry.key_width()) / width;
        resize_into(num_conv_heads * head_out_len, &mut tile.gdn_conv_outputs)?;
        let mixed_ref: &[Vec<f32>] = &tile.mixed[..count];
        tile.projector.row_parallelism().update_paired_chunks(
            convolution,
            head_hist,
            &mut tile.gdn_conv_outputs,
            head_out_len,
            |base_head, conv_chunk, out_chunk| {
                for (offset, (head_conv, head_out)) in conv_chunk
                    .chunks_exact_mut(head_hist)
                    .zip(out_chunk.chunks_exact_mut(head_out_len))
                    .enumerate()
                {
                    let head = base_head + offset;
                    let c_slice = layer
                        .convolution
                        .range(head * head_coeff..(head + 1) * head_coeff);
                    if matches!(arithmetic, Arithmetic::PinnedSwift) {
                        if head < qk_heads {
                            crate::kernels::pinned_convolution::step_silu_clamped_l2_token_rows_from_coefficients(
                                &history[head * head_hist..(head + 1) * head_hist],
                                head_conv,
                                mixed_ref,
                                &(head * width..(head + 1) * width),
                                c_slice,
                                geometry.convolution,
                                geometry.epsilon,
                                head_out,
                            )?;
                        } else {
                            crate::kernels::pinned_convolution::step_silu_token_rows_from_coefficients(
                                &history[head * head_hist..(head + 1) * head_hist],
                                head_conv,
                                mixed_ref,
                                &(head * width..(head + 1) * width),
                                c_slice,
                                geometry.convolution,
                                head_out,
                            )?;
                        }
                    } else {
                        for (token, token_mixed) in mixed_ref.iter().enumerate() {
                            head_out[token * width..(token + 1) * width]
                                .copy_from_slice(&token_mixed[head * width..(head + 1) * width]);
                        }
                        for head_mixed in head_out.chunks_exact_mut(width) {
                            for (channel, value) in head_mixed.iter_mut().enumerate() {
                                let projected = *value;
                                let start = channel * hist_width;
                                let coefficients = &c_slice.source()[channel * geometry.convolution
                                    ..(channel + 1) * geometry.convolution];
                                *value = arithmetic.convolution_row(
                                    &head_conv[start..start + hist_width],
                                    projected,
                                    coefficients,
                                )?;
                                head_conv[start..start + hist_width].rotate_left(1);
                                head_conv[start + hist_width - 1] = projected;
                            }
                            arithmetic.silu(head_mixed);
                        }
                        if head < qk_heads {
                            for head_mixed in head_out.chunks_exact_mut(width) {
                                l2_norm(head_mixed, geometry.epsilon);
                            }
                        }
                    }
                }
                Ok(())
            },
        )?;
    } else {
        for token in 0..count {
            check()?;
            let mixed = &mut tile.mixed[token];
            if parallelize_conv {
                let head_hist = width * hist_width;
                let head_coeff = width * geometry.convolution;
                let qk_heads = (2 * geometry.key_width()) / width;
                tile.projector.row_parallelism().update_paired_chunks(
                    mixed,
                    width,
                    convolution,
                    head_hist,
                    |base_head, mixed_chunk, conv_chunk| {
                        for (offset, (head_mixed, head_conv)) in mixed_chunk
                            .chunks_exact_mut(width)
                            .zip(conv_chunk.chunks_exact_mut(head_hist))
                            .enumerate()
                        {
                            let head = base_head + offset;
                            let c_slice = layer
                                .convolution
                                .range(head * head_coeff..(head + 1) * head_coeff);
                            if matches!(arithmetic, Arithmetic::PinnedSwift) {
                                crate::kernels::pinned_convolution::step_silu_in_place_coefficients(
                                    head_conv,
                                    head_mixed,
                                    c_slice,
                                    geometry.convolution,
                                )?;
                            } else {
                                for (channel, value) in head_mixed.iter_mut().enumerate() {
                                    let projected = *value;
                                    let start = channel * hist_width;
                                    let coefficients = &c_slice.source()[channel
                                        * geometry.convolution
                                        ..(channel + 1) * geometry.convolution];
                                    *value = arithmetic.convolution_row(
                                        &head_conv[start..start + hist_width],
                                        projected,
                                        coefficients,
                                    )?;
                                    head_conv[start..start + hist_width].rotate_left(1);
                                    head_conv[start + hist_width - 1] = projected;
                                }
                                arithmetic.silu(head_mixed);
                            }
                            if head < qk_heads {
                                l2_norm(head_mixed, geometry.epsilon);
                            }
                        }
                        Ok(())
                    },
                )?;
            } else if matches!(arithmetic, Arithmetic::PinnedSwift) {
                crate::kernels::pinned_convolution::step_silu_in_place_coefficients(
                    convolution,
                    mixed,
                    layer.convolution.view(),
                    geometry.convolution,
                )?;
            } else {
                for (channel, value) in mixed.iter_mut().enumerate() {
                    let projected = *value;
                    let start = channel * hist_width;
                    let coefficients = &layer.convolution.view().source()
                        [channel * geometry.convolution..(channel + 1) * geometry.convolution];
                    *value = arithmetic.convolution_row(
                        &convolution[start..start + hist_width],
                        projected,
                        coefficients,
                    )?;
                    if geometry.convolution > 1 {
                        convolution[start..start + hist_width].rotate_left(1);
                        convolution[start + hist_width - 1] = projected;
                    }
                }
                arithmetic.silu(mixed);
            }
            resize_into(geometry.value_width(), &mut tile.attended[token])?;
            if !parallelize_post_conv {
                if !parallelize_conv {
                    for head in mixed[..2 * geometry.key_width()].chunks_exact_mut(width) {
                        l2_norm(head, geometry.epsilon);
                    }
                }
                for head in 0..geometry.ssm_values {
                    let (decay, beta) = gdn_decay_beta(
                        tile.alpha[token][head],
                        tile.beta[token][head],
                        layer.time_bias[head],
                        layer.decay[head],
                    );
                    tile.alpha[token][head] = decay;
                    tile.beta[token][head] = beta;
                }
            }
            check()?;
        }
    }
    let parallel_gdn_heads =
        tile.projector.row_parallelism().workers() > 1 && geometry.ssm_values > 1;
    let defer_decay_beta =
        (conv_head_major || parallelize_post_conv) && parallel_gdn_heads && folded.is_none();
    if conv_head_major || parallelize_post_conv {
        if !conv_head_major && !parallelize_conv {
            tile.projector.row_parallelism().update_chunks(
                &mut tile.mixed[..count],
                1,
                |_, mixed_chunk| {
                    for mixed in mixed_chunk {
                        for head in mixed[..2 * geometry.key_width()].chunks_exact_mut(width) {
                            l2_norm(head, geometry.epsilon);
                        }
                    }
                    Ok(())
                },
            )?;
        }
        if !defer_decay_beta {
            for (alpha, beta) in tile.alpha[..count].iter_mut().zip(&mut tile.beta[..count]) {
                for head in 0..geometry.ssm_values {
                    let (decay, next_beta) = gdn_decay_beta(
                        alpha[head],
                        beta[head],
                        layer.time_bias[head],
                        layer.decay[head],
                    );
                    alpha[head] = decay;
                    beta[head] = next_beta;
                }
            }
        }
    }
    if let Some(layer_tape) = folded.as_deref_mut() {
        layer_tape.capture_gdn_inputs(
            count,
            conv_head_major,
            &tile.gdn_conv_outputs,
            &tile.mixed[..count],
            &tile.alpha[..count],
            &tile.beta[..count],
        )?;
    }
    let chunkwise = tile.chunkwise_gdn && folded.is_none();
    let wy_chunks = if chunkwise {
        count.div_ceil(WY_CHUNK_TOKENS)
    } else {
        1
    };
    let head_dot_stride = wy_chunks * KEY_DOT_STRIDE;
    let use_shared_key_dots =
        chunkwise && width > 0 && geometry.ssm_values > geometry.ssm_keys && geometry.ssm_keys > 0;
    if use_shared_key_dots {
        resize_into(geometry.ssm_keys * head_dot_stride, &mut tile.gdn_key_dots)?;
        let mixed_ref: &[Vec<f32>] = &tile.mixed[..count];
        let conv_ref: &[f32] = &tile.gdn_conv_outputs;
        let key_width = geometry.key_width();
        let head_out_len = count * width;
        tile.projector.row_parallelism().update_chunks(
            &mut tile.gdn_key_dots,
            head_dot_stride,
            |base_key, chunk| {
                for (offset, head_dots_all) in chunk.chunks_exact_mut(head_dot_stride).enumerate() {
                    let key_head = base_key + offset;
                    for (chunk_idx, dots) in head_dots_all
                        .as_chunks_mut::<KEY_DOT_STRIDE>()
                        .0
                        .iter_mut()
                        .enumerate()
                    {
                        let c_start = chunk_idx * WY_CHUNK_TOKENS;
                        let c_end = (c_start + WY_CHUNK_TOKENS).min(count);
                        let c_len = c_end - c_start;
                        if conv_head_major {
                            let q_head = &conv_ref[key_head * head_out_len + c_start * width
                                ..key_head * head_out_len + c_end * width];
                            let k_head_idx = geometry.ssm_keys + key_head;
                            let k_head = &conv_ref[k_head_idx * head_out_len + c_start * width
                                ..k_head_idx * head_out_len + c_end * width];
                            fill_key_head_dots_head_major(
                                arithmetic, q_head, k_head, c_len, width, dots,
                            )?;
                        } else {
                            fill_key_head_dots(
                                arithmetic,
                                &mixed_ref[c_start..c_end],
                                key_width,
                                width,
                                key_head,
                                dots,
                            )?;
                        }
                    }
                }
                Ok(())
            },
        )?;
    }
    let norm = RmsNormWeights::new(&layer.norm)?;
    let mut output_reuse = TileReuse::default();
    if parallel_gdn_heads {
        if chunkwise {
            for _ in 0..geometry.ssm_values {
                check()?;
            }
        }
        let head_state_len = width * width;
        let head_out_len = count * width;
        resize_into(
            geometry.ssm_values * head_out_len,
            &mut tile.gdn_head_outputs,
        )?;
        let mixed_ref: &[Vec<f32>] = &tile.mixed[..count];
        let conv_ref: &[f32] = &tile.gdn_conv_outputs;
        let alpha_ref: &[Vec<f32>] = &tile.alpha[..count];
        let beta_ref: &[Vec<f32>] = &tile.beta[..count];
        let gate_ref: &[Vec<f32>] = &tile.recurrent_gate[..count];
        let prev_ref: &[f32] = previous;
        let key_dots_ref: &[f32] = &tile.gdn_key_dots;
        tile.projector.row_parallelism().update_paired_chunks(
            matrix,
            head_state_len,
            &mut tile.gdn_head_outputs,
            head_out_len,
            |base_head, matrix_chunk, out_chunk| {
                with_worker_carry_state(|carry_state| {
                    let mut inputs: [Input<'_>; GDN_INPUT_CAPACITY] =
                        std::array::from_fn(|_| Input {
                            query: &[],
                            key: &[],
                            value: &[],
                            decay: 0.0,
                            beta: 0.0,
                        });
                    for (offset, (head_matrix, head_out)) in matrix_chunk
                        .chunks_exact_mut(head_state_len)
                        .zip(out_chunk.chunks_exact_mut(head_out_len))
                        .enumerate()
                    {
                        let head_index = base_head + offset;
                        let key_head = head_index % geometry.ssm_keys;
                        let head_time_bias = layer.time_bias[head_index];
                        let head_decay = layer.decay[head_index];
                        if conv_head_major {
                            let q_head =
                                &conv_ref[key_head * head_out_len..(key_head + 1) * head_out_len];
                            let k_head_idx = geometry.ssm_keys + key_head;
                            let k_head = &conv_ref
                                [k_head_idx * head_out_len..(k_head_idx + 1) * head_out_len];
                            let v_head_idx = 2 * geometry.ssm_keys + head_index;
                            let v_head = &conv_ref
                                [v_head_idx * head_out_len..(v_head_idx + 1) * head_out_len];
                            for (token, slot) in inputs[..count].iter_mut().enumerate() {
                                let (decay, beta) = if defer_decay_beta {
                                    gdn_decay_beta(
                                        alpha_ref[token][head_index],
                                        beta_ref[token][head_index],
                                        head_time_bias,
                                        head_decay,
                                    )
                                } else {
                                    (alpha_ref[token][head_index], beta_ref[token][head_index])
                                };
                                *slot = Input {
                                    query: &q_head[token * width..(token + 1) * width],
                                    key: &k_head[token * width..(token + 1) * width],
                                    value: &v_head[token * width..(token + 1) * width],
                                    decay,
                                    beta,
                                };
                            }
                        } else {
                            for (token, slot) in inputs[..count].iter_mut().enumerate() {
                                let mixed = &mixed_ref[token];
                                let (decay, beta) = if defer_decay_beta {
                                    gdn_decay_beta(
                                        alpha_ref[token][head_index],
                                        beta_ref[token][head_index],
                                        head_time_bias,
                                        head_decay,
                                    )
                                } else {
                                    (alpha_ref[token][head_index], beta_ref[token][head_index])
                                };
                                *slot = Input {
                                    query: &mixed[key_head * width..(key_head + 1) * width],
                                    key: &mixed[geometry.key_width() + key_head * width
                                        ..geometry.key_width() + (key_head + 1) * width],
                                    value: &mixed[2 * geometry.key_width() + head_index * width
                                        ..2 * geometry.key_width() + (head_index + 1) * width],
                                    decay,
                                    beta,
                                };
                            }
                        }
                        let prev_head = &prev_ref
                            [head_index * head_state_len..(head_index + 1) * head_state_len];
                        if chunkwise {
                            let head_dots = use_shared_key_dots.then(|| {
                                &key_dots_ref
                                    [key_head * head_dot_stride..(key_head + 1) * head_dot_stride]
                            });
                            chunkwise_subchunks_into(
                                arithmetic,
                                prev_head,
                                &inputs[..count],
                                head_dots,
                                width,
                                carry_state,
                                head_matrix,
                                head_out,
                            )?;
                        } else {
                            exact_head_into(
                                arithmetic,
                                prev_head,
                                &inputs[..count],
                                head_matrix,
                                head_out,
                            )?;
                        }
                        for token in 0..count {
                            let token_out = &mut head_out[token * width..(token + 1) * width];
                            arithmetic.normalize_swiglu_in_place(
                                token_out,
                                norm,
                                geometry.epsilon,
                                &gate_ref[token][head_index * width..(head_index + 1) * width],
                            )?;
                        }
                    }
                    Ok(())
                })
            },
        )?;
        if transforms.can_grouped_block_hadamard(&layer.output, geometry)
            && !adapters.has_target(&layer.output.name)
        {
            let head_outputs = &tile.gdn_head_outputs;
            let parallelism = tile.projector.row_parallelism();
            if count > 1 {
                let single_worker = crate::kernels::RowParallelism::default();
                parallelism.update_chunks(
                    &mut tile.transformed[..count],
                    1,
                    |base_tok, chunk| {
                        for (offset, target) in chunk.iter_mut().enumerate() {
                            let tok = base_tok + offset;
                            transforms.transform_grouped_strided_into_with_parallelism(
                                &head_outputs[tok * width..],
                                head_out_len,
                                geometry,
                                &single_worker,
                                target,
                            )?;
                        }
                        Ok(())
                    },
                )?;
            } else {
                transforms.transform_grouped_strided_into_with_parallelism(
                    head_outputs,
                    head_out_len,
                    geometry,
                    parallelism,
                    &mut tile.transformed[0],
                )?;
            }
            output_reuse = TileReuse {
                basis: Some(&layer.output),
                prepared: None,
            };
        } else if count > 1 {
            let head_outputs = &tile.gdn_head_outputs;
            let ssm_values = geometry.ssm_values;
            tile.projector.row_parallelism().update_chunks(
                &mut tile.attended[..count],
                1,
                |base_tok, chunk| {
                    for (offset, attended_tok) in chunk.iter_mut().enumerate() {
                        let tok = base_tok + offset;
                        for head_index in 0..ssm_values {
                            let src_start = head_index * head_out_len + tok * width;
                            attended_tok[head_index * width..(head_index + 1) * width]
                                .copy_from_slice(&head_outputs[src_start..src_start + width]);
                        }
                    }
                    Ok(())
                },
            )?;
        } else {
            for head_index in 0..geometry.ssm_values {
                let head_out = &tile.gdn_head_outputs
                    [head_index * head_out_len..(head_index + 1) * head_out_len];
                tile.attended[0][head_index * width..(head_index + 1) * width]
                    .copy_from_slice(&head_out[..width]);
            }
        }
        if chunkwise {
            tile.statistics.chunkwise_gdn_heads = tile
                .statistics
                .chunkwise_gdn_heads
                .saturating_add(u64::try_from(geometry.ssm_values)?);
        }
    } else {
        let mut inputs: [Input<'_>; GDN_INPUT_CAPACITY] = std::array::from_fn(|_| Input {
            query: &[],
            key: &[],
            value: &[],
            decay: 0.0,
            beta: 0.0,
        });
        let head_out_len = count * width;
        resize_into(head_out_len, &mut tile.gdn_head_outputs)?;
        with_worker_carry_state(|carry_state| -> Result<()> {
            for head_index in 0..geometry.ssm_values {
                if chunkwise {
                    check()?;
                }
                let key_head = head_index % geometry.ssm_keys;
                if conv_head_major {
                    let conv_ref: &[f32] = &tile.gdn_conv_outputs;
                    let q_head = &conv_ref[key_head * head_out_len..(key_head + 1) * head_out_len];
                    let k_head_idx = geometry.ssm_keys + key_head;
                    let k_head =
                        &conv_ref[k_head_idx * head_out_len..(k_head_idx + 1) * head_out_len];
                    let v_head_idx = 2 * geometry.ssm_keys + head_index;
                    let v_head =
                        &conv_ref[v_head_idx * head_out_len..(v_head_idx + 1) * head_out_len];
                    for (token, slot) in inputs[..count].iter_mut().enumerate() {
                        *slot = Input {
                            query: &q_head[token * width..(token + 1) * width],
                            key: &k_head[token * width..(token + 1) * width],
                            value: &v_head[token * width..(token + 1) * width],
                            decay: tile.alpha[token][head_index],
                            beta: tile.beta[token][head_index],
                        };
                    }
                } else {
                    for (token, slot) in inputs[..count].iter_mut().enumerate() {
                        let mixed = &tile.mixed[token];
                        *slot = Input {
                            query: &mixed[key_head * width..(key_head + 1) * width],
                            key: &mixed[geometry.key_width() + key_head * width
                                ..geometry.key_width() + (key_head + 1) * width],
                            value: &mixed[2 * geometry.key_width() + head_index * width
                                ..2 * geometry.key_width() + (head_index + 1) * width],
                            decay: tile.alpha[token][head_index],
                            beta: tile.beta[token][head_index],
                        };
                    }
                }
                let range = head_index * width * width..(head_index + 1) * width * width;
                if chunkwise {
                    let head_dots = use_shared_key_dots.then(|| {
                        &tile.gdn_key_dots
                            [key_head * head_dot_stride..(key_head + 1) * head_dot_stride]
                    });
                    chunkwise_subchunks_into(
                        arithmetic,
                        &previous[range.clone()],
                        &inputs[..count],
                        head_dots,
                        width,
                        carry_state,
                        &mut matrix[range],
                        &mut tile.gdn_head_outputs,
                    )?;
                } else {
                    exact_head_into(
                        arithmetic,
                        &previous[range.clone()],
                        &inputs[..count],
                        &mut matrix[range],
                        &mut tile.gdn_head_outputs,
                    )?;
                }
                for token in 0..count {
                    let token_out = &tile.gdn_head_outputs[token * width..(token + 1) * width];
                    arithmetic.normalize_swiglu_into(
                        token_out,
                        norm,
                        geometry.epsilon,
                        &tile.recurrent_gate[token][head_index * width..(head_index + 1) * width],
                        &mut tile.attended[token][head_index * width..(head_index + 1) * width],
                    )?;
                }
                if chunkwise {
                    tile.statistics.chunkwise_gdn_heads =
                        tile.statistics.chunkwise_gdn_heads.saturating_add(1);
                }
            }
            Ok(())
        })?;
    }
    check()?;
    let _ = project_tile(
        transforms,
        geometry,
        arithmetic,
        adapters,
        adapter_scratch,
        projection_scratch,
        output_reuse,
        &layer.output,
        &tile.attended[..count],
        &mut tile.attention[..count],
        &mut tile.transformed[..count],
        &mut tile.projector,
        executor,
        check,
    )?;
    if folded.is_none() {
        finite(convolution, "convolution state")?;
        std::mem::swap(history, convolution);
        std::mem::swap(previous, matrix);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn x1_temporal_gdn_prefill_consumer_matches_defensive_rows() -> Result<()> {
        let width = 128_usize;
        let previous: Vec<f32> = (0..width * width)
            .map(|i| {
                if i.is_multiple_of(3) {
                    -0.015_625
                } else {
                    0.03125
                }
            })
            .collect();
        let key = vec![0.25; width];
        let query = vec![-0.125; width];
        let values = vec![0.0625; width];
        let inputs = [
            Input {
                query: &query,
                key: &key,
                value: &values,
                decay: 0.875,
                beta: 0.625,
            },
            Input {
                query: &key,
                key: &query,
                value: &values,
                decay: -0.25,
                beta: 0.375,
            },
        ];
        let mut actual = vec![f32::NAN; previous.len()];
        let mut output = vec![f32::NAN; inputs.len() * width];
        exact_head_into(
            Arithmetic::PinnedSwift,
            &previous,
            &inputs,
            &mut actual,
            &mut output,
        )?;
        let mut expected = vec![0.0_f32; previous.len()];
        let mut expected_output = vec![0.0_f32; output.len()];
        let scale = 1.0 / small_float(width)?.sqrt();
        for (row, (previous, expected)) in previous
            .chunks_exact(width)
            .zip(expected.chunks_exact_mut(width))
            .enumerate()
        {
            Arithmetic::PinnedSwift.gdn_step_row_tokens_from(
                previous,
                expected,
                &inputs,
                row,
                width,
                scale,
                &mut expected_output,
            )?;
        }
        assert!(
            actual
                .iter()
                .zip(&expected)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        assert!(
            output
                .iter()
                .zip(&expected_output)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        Ok(())
    }

    #[test]
    #[expect(
        clippy::suboptimal_flops,
        reason = "Independent ordinary recurrence uses explicit multiply/add rather than chunk identities."
    )]
    fn chunkwise_gdn_triangular_solve_matches_sequential_signed_and_zero_decay() -> Result<()> {
        let queries = [[0.5, -0.25], [-0.25, 0.5], [0.25, 0.125], [-0.5, -0.25]];
        let keys = [[0.25, -0.5], [-0.5, 0.25], [0.5, 0.25], [-0.125, -0.5]];
        let values = [[0.5, -0.75], [-1.0, 0.25], [0.125, -0.5], [-0.75, -0.125]];
        for decays in [[0.5, 0.75, 0.25, 0.875], [0.5, 0.0, 0.25, 0.875], [0.0; 4]] {
            let inputs: Vec<_> = (0..4)
                .map(|token| Input {
                    query: &queries[token],
                    key: &keys[token],
                    value: &values[token],
                    decay: decays[token],
                    beta: [0.125, 0.25, 0.75, 0.5][token],
                })
                .collect();
            let previous = [0.125, -0.25, -0.5, 0.75];
            let mut expected = previous;
            let mut expected_outputs = Vec::new();
            for input in &inputs {
                for element in &mut expected {
                    *element *= input.decay;
                }
                for (row, &value) in expected.as_chunks_mut::<2>().0.iter_mut().zip(input.value) {
                    let predicted = row[0] * input.key[0] + row[1] * input.key[1];
                    let delta = (value - predicted) * input.beta;
                    for (element, &key) in row.iter_mut().zip(input.key) {
                        *element += delta * key;
                    }
                    expected_outputs
                        .push((row[0] * input.query[0] + row[1] * input.query[1]) / 2.0_f32.sqrt());
                }
            }
            for arithmetic in [Arithmetic::Source, Arithmetic::PinnedSwift] {
                let mut actual = [0.0; 4];
                let mut outputs = Vec::new();
                head(arithmetic, &previous, &inputs, &mut actual, &mut outputs)?;
                for (&actual, &expected) in actual
                    .iter()
                    .chain(&outputs)
                    .zip(expected.iter().chain(&expected_outputs))
                {
                    assert!((actual - expected).abs() <= 0.000_001);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn chunkwise_gdn_rejects_invalid_shapes_and_gates() {
        let input = Input {
            query: &[1.0],
            key: &[1.0],
            value: &[1.0],
            decay: -0.5,
            beta: 0.5,
        };
        assert!(
            head(
                Arithmetic::Source,
                &[0.0],
                &[input],
                &mut [0.0],
                &mut Vec::new()
            )
            .is_err()
        );
        assert!(head(Arithmetic::Source, &[], &[], &mut [], &mut Vec::new()).is_err());
    }

    #[test]
    #[expect(
        clippy::suboptimal_flops,
        reason = "Verifies bit-for-bit parity against the single-row WY reference."
    )]
    fn chunkwise_gdn_four_row_and_vector_update_matches_single_row_bits() -> Result<()> {
        for width in [6, 16, 18, 32] {
            let count = 6;
            let q_buf: Vec<f32> = (0..count * width)
                .map(|i| {
                    let s = i16::try_from((i * 37 + 11) % 64).unwrap_or(0) - 32;
                    f32::from(s) * 0.03125
                })
                .collect();
            let k_buf: Vec<f32> = (0..count * width)
                .map(|i| {
                    let s = i16::try_from((i * 53 + 7) % 64).unwrap_or(0) - 32;
                    f32::from(s) * 0.03125
                })
                .collect();
            let v_buf: Vec<f32> = (0..count * width)
                .map(|i| {
                    let s = i16::try_from((i * 29 + 3) % 64).unwrap_or(0) - 32;
                    f32::from(s) * 0.03125
                })
                .collect();
            let prev: Vec<f32> = (0..width * width)
                .map(|i| {
                    let s = i16::try_from((i * 19 + 5) % 64).unwrap_or(0) - 32;
                    f32::from(s) * 0.015_625
                })
                .collect();
            let decays = [0.875_f32, 0.75, 0.625, 0.5, 0.9375, 0.8125];
            let betas = [0.25_f32, 0.5, 0.375, 0.625, 0.75, 0.125];
            let inputs: Vec<_> = (0..count)
                .map(|t| Input {
                    query: &q_buf[t * width..(t + 1) * width],
                    key: &k_buf[t * width..(t + 1) * width],
                    value: &v_buf[t * width..(t + 1) * width],
                    decay: decays[t],
                    beta: betas[t],
                })
                .collect();

            for arithmetic in [Arithmetic::Source, Arithmetic::PinnedSwift] {
                let mut actual_mat = vec![0.0_f32; width * width];
                let mut actual_out = Vec::new();
                head(arithmetic, &prev, &inputs, &mut actual_mat, &mut actual_out)?;

                let mut ref_mat = vec![0.0_f32; width * width];
                for (r, state) in prev.chunks_exact(width).enumerate() {
                    let mut deltas = [0.0_f32; 32];
                    let mut cum = 1.0_f32;
                    for t in 0..count {
                        cum *= inputs[t].decay;
                        let (key_dot, _) =
                            arithmetic.gdn_dot_pair(state, inputs[t].key, inputs[t].query)?;
                        let mut predicted = cum * key_dot;
                        for (p, &d) in deltas[..t].iter().enumerate() {
                            let mut prod = 1.0_f32;
                            for step in ((p + 1)..=t).rev() {
                                prod *= inputs[step].decay;
                            }
                            let (_, k_dot) = arithmetic.gdn_dot_pair(
                                inputs[p].key,
                                inputs[t].query,
                                inputs[t].key,
                            )?;
                            predicted += (prod * k_dot) * d;
                        }
                        deltas[t] = (inputs[t].value[r] - predicted) * inputs[t].beta;
                    }
                    let row_slice = &mut ref_mat[r * width..(r + 1) * width];
                    scale_slice_into(state, cum, row_slice);
                    for (p, &d) in deltas[..count].iter().enumerate() {
                        let mut prod = 1.0_f32;
                        for step in ((p + 1)..count).rev() {
                            prod *= inputs[step].decay;
                        }
                        add_scaled_slice_in_place(row_slice, inputs[p].key, prod * d);
                    }
                }
                for (a, b) in actual_mat.iter().zip(&ref_mat) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
            }
        }
        Ok(())
    }

    fn verify_eight_row_neon_case(width: usize, count: usize) -> Result<()> {
        let prev: Vec<f32> = (0..width * width)
            .map(|i| [0.125_f32, -0.0625, 0.03125, -0.0, 0.25][i % 5])
            .collect();
        let q_buf: Vec<f32> = (0..count * width)
            .map(|i| f32::from(i8::try_from((i * 19) % 31).unwrap_or(0) - 15) * 0.03125)
            .collect();
        let k_buf: Vec<f32> = (0..count * width)
            .map(|i| f32::from(i8::try_from((i * 23) % 31).unwrap_or(0) - 15) * 0.03125)
            .collect();
        let v_buf: Vec<f32> = (0..count * width)
            .map(|i| f32::from(i8::try_from((i * 29) % 31).unwrap_or(0) - 15) * 0.03125)
            .collect();
        let inputs: Vec<_> = (0..count)
            .map(|t| Input {
                query: &q_buf[t * width..(t + 1) * width],
                key: &k_buf[t * width..(t + 1) * width],
                value: &v_buf[t * width..(t + 1) * width],
                decay: [0.875_f32, 0.75, 0.0, 0.9375, 0.5][t % 5],
                beta: [0.25_f32, 0.625, 1.0, 0.375, 0.5][t % 5],
            })
            .collect();
        let mut cross_key = [[0.0_f32; 32]; 32];
        let mut cross_query = [[0.0_f32; 32]; 32];
        let mut cumulative = [0.0_f32; 32];
        let mut last_products = [0.0_f32; 32];
        let mut acc = 1.0_f32;
        for t in 0..count {
            acc *= inputs[t].decay;
            cumulative[t] = acc;
            last_products[t] = if t == count - 1 { 1.0 } else { 0.85 };
            for p in 0..=t {
                cross_key[t][p] =
                    f32::from(i8::try_from((t * 7 + p * 3) % 9).unwrap_or(0) - 4) * 0.125;
                cross_query[t][p] =
                    f32::from(i8::try_from((t * 5 + p * 11) % 9).unwrap_or(0) - 4) * 0.125;
            }
        }
        let scale = 1.0 / small_float(width)?.sqrt();
        for arithmetic in [Arithmetic::Source, Arithmetic::PinnedSwift] {
            let mut scalar_out = vec![f32::NAN; count * width];
            let mut scalar_sd = [[0.0_f32; 8]; 32];
            let base_row = (width / 8 - 1) * 8;
            let states: [&[f32]; 8] =
                std::array::from_fn(|r| &prev[(base_row + r) * width..(base_row + r + 1) * width]);
            head_into_eight_row_block_scalar(
                arithmetic,
                states,
                &inputs,
                &cumulative,
                &cross_key,
                &cross_query,
                &last_products,
                base_row,
                width,
                scale,
                &mut scalar_out,
                &mut scalar_sd,
            )?;
            #[cfg(all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc",
                target_feature = "neon"
            ))]
            {
                let mut neon_out = vec![f32::NAN; count * width];
                let mut neon_sd = [[0.0_f32; 8]; 32];
                assert!(head_into_eight_row_block_neon(
                    arithmetic,
                    states,
                    &inputs,
                    &cumulative,
                    &cross_key,
                    &cross_query,
                    &last_products,
                    base_row,
                    width,
                    scale,
                    &mut neon_out,
                    &mut neon_sd,
                )?);
                for tok in 0..count {
                    for sub in 0..8 {
                        let idx = tok * width + base_row + sub;
                        assert_eq!(neon_out[idx].to_bits(), scalar_out[idx].to_bits());
                    }
                }
                for (n_row, s_row) in neon_sd[..count].iter().zip(&scalar_sd[..count]) {
                    for (n_val, s_val) in n_row.iter().zip(s_row) {
                        assert_eq!(n_val.to_bits(), s_val.to_bits());
                    }
                }
            }
            let mut head_mat = vec![0.0_f32; width * width];
            let mut head_out = Vec::new();
            head(arithmetic, &prev, &inputs, &mut head_mat, &mut head_out)?;
            assert_eq!(head_out.len(), count * width);
        }
        Ok(())
    }

    #[test]
    fn chunkwise_gdn_eight_row_neon_matches_scalar_at_zero_ulp() -> Result<()> {
        for width in [64_usize, 128] {
            for count in 1..=32_usize {
                verify_eight_row_neon_case(width, count)?;
            }
        }
        Ok(())
    }

    fn build_folded_carry_tape(
        kept: usize,
        geometry: &Geometry,
    ) -> Result<FoldedRecurrentLayerTape> {
        let max_tokens = crate::speculative::MAX_SPECULATIVE_DRAFT_TOKENS;
        let conv_width = geometry.convolution_width();
        let mut tape = FoldedRecurrentLayerTape::new(max_tokens, geometry)?;
        for tok in 0..kept {
            for c in 0..conv_width {
                tape.carried_raw_conv
                    .push((f32::from(i8::try_from((tok + c) % 7).unwrap_or(0)) - 3.0) * 0.125);
            }
            for c in 0..geometry.key_width() {
                tape.carried_keys
                    .push((f32::from(i8::try_from((tok * 3 + c) % 9).unwrap_or(0)) - 4.0) * 0.0625);
            }
            for c in 0..geometry.value_width() {
                tape.carried_values
                    .push((f32::from(i8::try_from((tok * 5 + c) % 9).unwrap_or(0)) - 4.0) * 0.0625);
            }
            for h in 0..geometry.ssm_values {
                tape.carried_decay
                    .push([0.875_f32, 0.0, 0.5, 1.0][(tok + h) % 4]);
                tape.carried_beta
                    .push([0.625_f32, 0.25, 0.0, 0.75][(tok + h) % 4]);
            }
        }
        tape.carried_len = kept;
        Ok(tape)
    }

    fn expected_folded_carry_matrix(
        arithmetic: Arithmetic,
        initial_matrix: &[f32],
        tape: &FoldedRecurrentLayerTape,
        geometry: &Geometry,
    ) -> Result<Vec<f32>> {
        let width = geometry.ssm_head;
        let num_keys = geometry.ssm_keys;
        let num_values = geometry.ssm_values;
        let head_state_len = width * width;
        let mut ref_matrix = vec![0.0_f32; initial_matrix.len()];
        let mut ref_head_out = vec![0.0_f32; tape.carried_len * width];
        for head_idx in 0..num_values {
            let key_head = head_idx % num_keys;
            let head_inputs: Vec<Input<'_>> = (0..tape.carried_len)
                .map(|tok| {
                    let ks = tok * tape.key_width + key_head * width;
                    let vs = tok * tape.value_width + head_idx * width;
                    let k = &tape.carried_keys[ks..ks + width];
                    Input {
                        query: k,
                        key: k,
                        value: &tape.carried_values[vs..vs + width],
                        decay: tape.carried_decay[tok * num_values + head_idx],
                        beta: tape.carried_beta[tok * num_values + head_idx],
                    }
                })
                .collect();
            let r = head_idx * head_state_len..(head_idx + 1) * head_state_len;
            exact_head_into(
                arithmetic,
                &initial_matrix[r.clone()],
                &head_inputs,
                &mut ref_matrix[r],
                &mut ref_head_out,
            )?;
        }
        Ok(ref_matrix)
    }

    fn check_folded_carry_flush_case(width: usize, kept: usize) -> Result<()> {
        #[cfg(feature = "parallel-rows")]
        let worker_counts = [1_usize, 4];
        #[cfg(not(feature = "parallel-rows"))]
        let worker_counts = [1_usize];
        let num_keys = 2;
        let num_values = 4;
        let geometry = Geometry {
            embedding: 16,
            feed_forward: 16,
            heads: 1,
            kv_heads: 1,
            head: 16,
            rope: 4,
            rope_sections: [2, 0, 0, 0],
            rope_base: 10_000.0,
            epsilon: 1e-6,
            ssm_head: width,
            ssm_keys: num_keys,
            ssm_values: num_values,
            convolution: 2,
        };
        let conv_width = geometry.convolution_width();
        for arithmetic in [Arithmetic::PinnedSwift, Arithmetic::Source] {
            let mut initial_conv = vec![0.125_f32; conv_width];
            initial_conv[0] = -0.0;
            let mut initial_matrix = vec![0.0_f32; num_values * width * width];
            for (i, slot) in initial_matrix.iter_mut().enumerate() {
                *slot = (f32::from(i8::try_from(i % 13).unwrap_or(0)) - 6.0) * 0.03125;
            }
            let tape = build_folded_carry_tape(kept, &geometry)?;
            let ref_matrix =
                expected_folded_carry_matrix(arithmetic, &initial_matrix, &tape, &geometry)?;

            for workers in worker_counts {
                let mut test_tape = tape.clone();
                let mut state = LayerState::Recurrent {
                    convolution: initial_conv.clone(),
                    matrix: initial_matrix.clone(),
                };
                let mut delta = LayerDelta::Recurrent {
                    convolution: Vec::new(),
                    matrix: Vec::new(),
                };
                let mut projector =
                    ProjectionTile::with_backend(crate::kernels::MatvecBackend::Scalar);
                projector.set_row_parallelism(crate::kernels::RowParallelism::new(workers)?);
                let mut scratch_out = Vec::new();
                flush_recurrent_carry_layer(
                    &mut test_tape,
                    &mut state,
                    &mut delta,
                    &geometry,
                    arithmetic,
                    &projector,
                    &mut scratch_out,
                )?;
                let LayerState::Recurrent { matrix, .. } = state else {
                    return Err(invalid("expected recurrent state"));
                };
                assert_eq!(scratch_out, Vec::<f32>::new());
                assert_eq!(
                    matrix.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    ref_matrix.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "carry flush mismatch width={width} kept={kept} workers={workers}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn folded_carry_flush_state_only_matches_exact_head_zero_ulp() -> Result<()> {
        for width in [32_usize, 64, 128, 131] {
            for kept in 1..=crate::speculative::MAX_SPECULATIVE_DRAFT_TOKENS {
                check_folded_carry_flush_case(width, kept)?;
            }
        }
        Ok(())
    }

    #[test]
    fn folded_carry_flush_preserves_history_and_matrix_on_failure() -> Result<()> {
        let geometry = Geometry {
            embedding: 16,
            feed_forward: 16,
            heads: 1,
            kv_heads: 1,
            head: 16,
            rope: 4,
            rope_sections: [2, 0, 0, 0],
            rope_base: 10_000.0,
            epsilon: 1e-6,
            ssm_head: 32,
            ssm_keys: 2,
            ssm_values: 4,
            convolution: 4,
        };
        let initial_conv = vec![0.125_f32; geometry.convolution_width() * (geometry.convolution - 1)];
        let initial_matrix = vec![0.25_f32; geometry.ssm_values * geometry.ssm_head * geometry.ssm_head];
        let mut tape = build_folded_carry_tape(2, &geometry)?;
        tape.carried_decay[0] = f32::NAN;

        let mut state = LayerState::Recurrent {
            convolution: initial_conv.clone(),
            matrix: initial_matrix.clone(),
        };
        let mut delta = LayerDelta::Recurrent {
            convolution: Vec::new(),
            matrix: Vec::new(),
        };
        let projector = ProjectionTile::with_backend(crate::kernels::MatvecBackend::Scalar);
        let mut scratch_out = Vec::new();
        let Err(err) = flush_recurrent_carry_layer(
            &mut tape,
            &mut state,
            &mut delta,
            &geometry,
            Arithmetic::Source,
            &projector,
            &mut scratch_out,
        ) else {
            return Err(invalid("invalid decay must fail carry flush"));
        };
        assert!(err.to_string().contains("gate bounds"));
        let LayerState::Recurrent {
            convolution: after_conv,
            matrix: after_matrix,
        } = state
        else {
            return Err(invalid("expected recurrent state"));
        };
        assert_eq!(after_conv, initial_conv);
        assert_eq!(after_matrix, initial_matrix);
        assert_eq!(tape.carried_len, 2);
        Ok(())
    }

    #[allow(
        clippy::too_many_lines,
        reason = "Chunkwise subchunk verification helper compares multiple state steps."
    )]
    fn check_chunkwise_subchunks_case(count: usize, arithmetic: Arithmetic) -> Result<()> {
        let width = 16_usize;
        let state_len = width * width;
        let is_pinned = matches!(arithmetic, Arithmetic::PinnedSwift);
        let prev: Vec<f32> = (0..state_len)
            .map(|i| (f32::from(i8::try_from(i % 17).unwrap_or(0)) - 8.0) * 0.015_625)
            .collect();
        let make_rows = |a: usize, b: usize, m: usize, bias: f32| -> Vec<Vec<f32>> {
            (0..count)
                .map(|t| {
                    (0..width)
                        .map(|c| {
                            (f32::from(i8::try_from((t * a + c * b) % m).unwrap_or(0)) - bias)
                                * 0.03125
                        })
                        .collect()
                })
                .collect()
        };
        let q = make_rows(5, 3, 15, 0.0);
        let mut k = make_rows(7, 11, 13, 6.0);
        for row in &mut k {
            l2_norm(row, 1e-6);
        }
        let v = make_rows(13, 5, 19, 0.0);
        let inputs: Vec<Input<'_>> = (0..count)
            .map(|t| Input {
                query: &q[t],
                key: &k[t],
                value: &v[t],
                decay: [0.9_f32, 0.875, 0.85, 0.825, 0.8][t % 5],
                beta: [0.125_f32, 0.156_25, 0.1875, 0.218_75, 0.25][t % 5],
            })
            .collect();
        let chunks = count.div_ceil(WY_CHUNK_TOKENS);
        let mut dots_all = vec![0.0_f32; chunks * KEY_DOT_STRIDE];
        let q_flat: Vec<f32> = q.iter().flat_map(|row| row.iter().copied()).collect();
        let k_flat: Vec<f32> = k.iter().flat_map(|row| row.iter().copied()).collect();
        for (chunk_idx, dots) in dots_all
            .as_chunks_mut::<KEY_DOT_STRIDE>()
            .0
            .iter_mut()
            .enumerate()
        {
            let c_start = chunk_idx * WY_CHUNK_TOKENS;
            let c_end = (c_start + WY_CHUNK_TOKENS).min(count);
            fill_key_head_dots_head_major(
                arithmetic,
                &q_flat[c_start * width..c_end * width],
                &k_flat[c_start * width..c_end * width],
                c_end - c_start,
                width,
                dots,
            )?;
        }
        let (mut ref_a, mut ref_b, mut ref_out) = (
            vec![0.0_f32; state_len],
            vec![0.0_f32; state_len],
            vec![0.0_f32; count * width],
        );
        head_into(
            arithmetic,
            &prev,
            &inputs[..WY_CHUNK_TOKENS],
            Some(&dots_all[..KEY_DOT_STRIDE]),
            &mut ref_a,
            &mut ref_out[..WY_CHUNK_TOKENS * width],
        )?;
        head_into(
            arithmetic,
            &ref_a,
            &inputs[WY_CHUNK_TOKENS..],
            Some(&dots_all[KEY_DOT_STRIDE..2 * KEY_DOT_STRIDE]),
            &mut ref_b,
            &mut ref_out[WY_CHUNK_TOKENS * width..],
        )?;
        let (mut carry_state, mut actual_state, mut actual_out) = (
            Vec::new(),
            vec![0.0_f32; state_len],
            vec![0.0_f32; count * width],
        );
        chunkwise_subchunks_into(
            arithmetic,
            &prev,
            &inputs,
            Some(&dots_all),
            width,
            &mut carry_state,
            &mut actual_state,
            &mut actual_out,
        )?;
        assert_eq!(
            actual_state.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            ref_b.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "state mismatch count={count} pinned={is_pinned}"
        );
        assert_eq!(
            actual_out.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            ref_out.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "output mismatch count={count} pinned={is_pinned}"
        );
        Ok(())
    }

    #[test]
    fn chunkwise_subchunks_64_matches_32_token_chunks_zero_ulp() -> Result<()> {
        for count in [33_usize, 40, 64] {
            for arithmetic in [Arithmetic::Source, Arithmetic::PinnedSwift] {
                check_chunkwise_subchunks_case(count, arithmetic)?;
            }
        }
        Ok(())
    }
}
