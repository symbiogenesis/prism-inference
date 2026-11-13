//! Exact pinned Windows ARM64 GDN arithmetic with a safe baseline NEON path.
//!
//! Reference: Prism `73bfe6cc485554822f80c473a3c0c4f25e80f41b`,
//! `ggml/src/ggml-cpu/vec.cpp`, `vec.h` and `simd-mappings.h`.
//! Upstream MIT terms are retained in `PRISM-LICENSE.txt`.

#![expect(
    clippy::redundant_pub_crate,
    reason = "Keep graph-only GDN helpers crate scoped while satisfying unreachable_pub."
)]

use crate::{Result, invalid};

/// Per-token GDN recurrence inputs for a single head chunk.
#[derive(Clone, Copy)]
pub(crate) struct ChunkInput<'a> {
    pub(crate) query: &'a [f32],
    pub(crate) key: &'a [f32],
    pub(crate) value: &'a [f32],
    pub(crate) decay: f32,
    pub(crate) beta: f32,
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
#[path = "pinned_gdn_neon.rs"]
mod neon;

#[cfg(test)]
#[path = "pinned_gdn_provisional_tests.rs"]
mod provisional_tests;

/// Matches the pinned NEON F32 dot, including its scalar remainder.
///
/// # Errors
/// Rejects unequal vector lengths or a nonfinite result.
pub(crate) fn dot(first: &[f32], second: &[f32]) -> Result<f32> {
    if first.len() != second.len() {
        return Err(invalid("pinned GDN dot requires matching vectors"));
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(result) = neon::dot(first, second) {
        return super::finite(result);
    }
    dot_scalar(first, second)
}

/// Computes `(dot(state, first), dot(state, second))` in a single pass over `state`.
///
/// # Errors
/// Rejects unequal vector lengths or a nonfinite result.
pub(crate) fn dot_pair(state: &[f32], first: &[f32], second: &[f32]) -> Result<(f32, f32)> {
    if state.len() != first.len() || state.len() != second.len() {
        return Err(invalid("pinned GDN dot requires matching vectors"));
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some((first_result, second_result)) = neon::dot_pair(state, first, second) {
        return Ok((super::finite(first_result)?, super::finite(second_result)?));
    }
    Ok((dot_scalar(state, first)?, dot_scalar(state, second)?))
}

/// Computes `[dot(state, inputs[0]), dot(state, inputs[1]), dot(state, inputs[2]), dot(state, inputs[3])]`
/// in a single pass over `state`.
///
/// # Errors
/// Rejects unequal vector lengths or a nonfinite result.
pub(crate) fn dot_four(state: &[f32], inputs: [&[f32]; 4]) -> Result<[f32; 4]> {
    let width = state.len();
    if inputs.iter().any(|input| input.len() != width) {
        return Err(invalid("pinned GDN dot requires matching vectors"));
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(values) = neon::dot_four(state, inputs) {
        return Ok([
            super::finite(values[0])?,
            super::finite(values[1])?,
            super::finite(values[2])?,
            super::finite(values[3])?,
        ]);
    }
    Ok([
        dot_scalar(state, inputs[0])?,
        dot_scalar(state, inputs[1])?,
        dot_scalar(state, inputs[2])?,
        dot_scalar(state, inputs[3])?,
    ])
}

/// Computes `[(dot(states[r], first), dot(states[r], second)); 4]` in a single pass
/// over `first` and `second`.
///
/// # Errors
/// Rejects unequal vector lengths or a nonfinite result.
pub(crate) fn dot_pair_four_rows(
    states: [&[f32]; 4],
    first: &[f32],
    second: &[f32],
) -> Result<[(f32, f32); 4]> {
    let width = first.len();
    if second.len() != width || states.iter().any(|state| state.len() != width) {
        return Err(invalid("pinned GDN dot requires matching vectors"));
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(pairs) = neon::dot_pair_four_rows(states, first, second) {
        return Ok([
            (super::finite(pairs[0].0)?, super::finite(pairs[0].1)?),
            (super::finite(pairs[1].0)?, super::finite(pairs[1].1)?),
            (super::finite(pairs[2].0)?, super::finite(pairs[2].1)?),
            (super::finite(pairs[3].0)?, super::finite(pairs[3].1)?),
        ]);
    }
    Ok([
        dot_pair(states[0], first, second)?,
        dot_pair(states[1], first, second)?,
        dot_pair(states[2], first, second)?,
        dot_pair(states[3], first, second)?,
    ])
}

/// Computes `[(dot(states[r], first), dot(states[r], second)); 8]` in a single pass
/// over `first` and `second`.
///
/// # Errors
/// Rejects unequal vector lengths or a nonfinite result.
pub(crate) fn dot_pair_eight_rows(
    states: [&[f32]; 8],
    first: &[f32],
    second: &[f32],
) -> Result<[(f32, f32); 8]> {
    let width = first.len();
    if second.len() != width || states.iter().any(|state| state.len() != width) {
        return Err(invalid("pinned GDN dot requires matching vectors"));
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(pairs) = neon::dot_pair_eight_rows(states, first, second) {
        return Ok([
            (super::finite(pairs[0].0)?, super::finite(pairs[0].1)?),
            (super::finite(pairs[1].0)?, super::finite(pairs[1].1)?),
            (super::finite(pairs[2].0)?, super::finite(pairs[2].1)?),
            (super::finite(pairs[3].0)?, super::finite(pairs[3].1)?),
            (super::finite(pairs[4].0)?, super::finite(pairs[4].1)?),
            (super::finite(pairs[5].0)?, super::finite(pairs[5].1)?),
            (super::finite(pairs[6].0)?, super::finite(pairs[6].1)?),
            (super::finite(pairs[7].0)?, super::finite(pairs[7].1)?),
        ]);
    }
    Ok([
        dot_pair(states[0], first, second)?,
        dot_pair(states[1], first, second)?,
        dot_pair(states[2], first, second)?,
        dot_pair(states[3], first, second)?,
        dot_pair(states[4], first, second)?,
        dot_pair(states[5], first, second)?,
        dot_pair(states[6], first, second)?,
        dot_pair(states[7], first, second)?,
    ])
}

// Floating matrix projections deliberately retain scalar diagnostics and their
// explicit scalar policy. Only the graph's pinned GDN helper selects NEON.
#[expect(
    clippy::suboptimal_flops,
    reason = "Keep the scalar oracle's exact multiply/add remainder rounding."
)]
pub(super) fn dot_scalar(first: &[f32], second: &[f32]) -> Result<f32> {
    if first.len() != second.len() {
        return Err(invalid("pinned GDN dot requires matching vectors"));
    }
    let vectorized = first.len() / 16 * 16;
    let mut sums = [[0.0_f32; 4]; 4];
    for offset in (0..vectorized).step_by(16) {
        for (register, lanes) in sums.iter_mut().enumerate() {
            for (lane, sum) in lanes.iter_mut().enumerate() {
                let index = offset + register * 4 + lane;
                *sum = first[index].mul_add(second[index], *sum);
            }
        }
    }
    // GGML_F32x4_REDUCE first combines registers 0+2 and 1+3. The
    // horizontal vaddvq_f32 then adds adjacent pairs of the resulting lanes.
    let mut lanes = [0.0; 4];
    for (index, lane) in lanes.iter_mut().enumerate() {
        *lane = (sums[0][index] + sums[2][index]) + (sums[1][index] + sums[3][index]);
    }
    let mut result = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    for (&first, &second) in first[vectorized..].iter().zip(&second[vectorized..]) {
        result += first * second;
    }
    super::finite(result)
}

/// Scales `state` by `decay` and computes its pinned NEON F32 dot with `key` in a single pass.
///
/// # Errors
/// Rejects unequal vector lengths before mutation, or a nonfinite dot result.
pub(crate) fn decay_and_dot(state: &mut [f32], key: &[f32], decay: f32) -> Result<f32> {
    if state.len() != key.len() {
        return Err(invalid("pinned GDN dot requires matching vectors"));
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(result) = neon::decay_and_dot(state, key, decay) {
        return super::finite(result);
    }
    for element in state.iter_mut() {
        *element *= decay;
    }
    dot_scalar(state, key)
}

/// Scales `previous` by `decay` into `state` and computes its pinned NEON F32 dot with `key` in a single pass.
///
/// # Errors
/// Rejects unequal vector lengths before mutation, or a nonfinite dot result.
pub(crate) fn decay_and_dot_from(
    previous: &[f32],
    state: &mut [f32],
    key: &[f32],
    decay: f32,
) -> Result<f32> {
    if previous.len() != state.len() || state.len() != key.len() {
        return Err(invalid("pinned GDN dot requires matching vectors"));
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(result) = neon::decay_and_dot_from(previous, state, key, decay) {
        return super::finite(result);
    }
    for (element, &prev) in state.iter_mut().zip(previous) {
        *element = prev * decay;
    }
    dot_scalar(state, key)
}

/// Matches the pinned NEON scaled update, including its scalar remainder.
///
/// # Errors
/// Rejects unequal lengths before mutation, or a nonfinite updated element.
#[cfg(test)]
pub(crate) fn add_scaled(target: &mut [f32], input: &[f32], scale: f32) -> Result<()> {
    if target.len() != input.len() {
        return Err(invalid("pinned GDN update requires matching vectors"));
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(result) = neon::add_scaled(target, input, scale) {
        return result;
    }
    scalar_add_scaled(target, input, scale)
}

/// Updates `target` with `scale * key` and computes its pinned NEON F32 dot with `query`.
///
/// # Errors
/// Rejects unequal lengths before mutation, or a nonfinite updated element or dot result.
pub(crate) fn update_and_dot(
    target: &mut [f32],
    key: &[f32],
    scale: f32,
    query: &[f32],
) -> Result<f32> {
    if target.len() != key.len() || target.len() != query.len() {
        return Err(invalid("pinned GDN update requires matching vectors"));
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(result) = neon::update_and_dot(target, key, scale, query) {
        return result;
    }
    scalar_add_scaled(target, key, scale)?;
    dot_scalar(target, query)
}

/// Fuses `decay_and_dot_from(previous, state, key, decay)` and
/// `update_and_dot(state, key, (value - dot_k) * beta, query)` into a single write to `state`.
///
/// # Errors
/// Rejects unequal vector lengths before mutation, or a nonfinite updated element or dot result.
pub(crate) fn step_row_from(
    previous: &[f32],
    state: &mut [f32],
    key: &[f32],
    query: &[f32],
    decay: f32,
    value: f32,
    beta: f32,
) -> Result<f32> {
    if previous.len() != state.len() || state.len() != key.len() || state.len() != query.len() {
        return Err(invalid("pinned GDN step requires matching vectors"));
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(result) = neon::step_row_from(previous, state, key, query, decay, value, beta) {
        return result;
    }
    let dot_k = decay_and_dot_from(previous, state, key, decay)?;
    let delta = (value - dot_k) * beta;
    update_and_dot(state, key, delta, query)
}

/// Fuses in-place `decay_and_dot(state, key, decay)` and
/// `update_and_dot(state, key, (value - dot_k) * beta, query)` into a single read and single write of `state`.
///
/// # Errors
/// Rejects unequal vector lengths before mutation, or a nonfinite updated element or dot result.
pub(crate) fn step_row_in_place(
    state: &mut [f32],
    key: &[f32],
    query: &[f32],
    decay: f32,
    value: f32,
    beta: f32,
) -> Result<f32> {
    if state.len() != key.len() || state.len() != query.len() {
        return Err(invalid("pinned GDN step requires matching vectors"));
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(result) = neon::step_row_in_place(state, key, query, decay, value, beta) {
        return result;
    }
    let dot_k = decay_and_dot(state, key, decay)?;
    let delta = (value - dot_k) * beta;
    update_and_dot(state, key, delta, query)
}

/// Steps one GDN state row across `inputs.len()` tokens from `previous` into `state`,
/// writing `out * scale` to `output[token * stride + row]` for each token.
///
/// # Errors
/// Rejects invalid geometry before mutation, or a nonfinite updated element or dot result.
pub(crate) fn step_row_tokens_from(
    previous: &[f32],
    state: &mut [f32],
    inputs: &[ChunkInput<'_>],
    row: usize,
    stride: usize,
    scale: f32,
    output: &mut [f32],
) -> Result<()> {
    let Some(first) = inputs.first() else {
        return Err(invalid("empty GDN chunk"));
    };
    if previous.len() != state.len()
        || state.len() != stride
        || row >= stride
        || inputs.len().checked_mul(stride) != Some(output.len())
        || inputs.iter().any(|input| {
            input.key.len() != stride || input.query.len() != stride || input.value.len() <= row
        })
    {
        return Err(invalid("pinned GDN step requires matching vectors"));
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(result) =
        neon::step_row_tokens_from(previous, state, inputs, row, stride, scale, output)
    {
        return result;
    }
    output[row] = step_row_from(
        previous,
        state,
        first.key,
        first.query,
        first.decay,
        first.value[row],
        first.beta,
    )? * scale;
    for (token, input) in inputs.iter().enumerate().skip(1) {
        output[token * stride + row] = step_row_in_place(
            state,
            input.key,
            input.query,
            input.decay,
            input.value[row],
            input.beta,
        )? * scale;
    }
    Ok(())
}

/// Steps 4 GDN state rows simultaneously from `previous` into `states`, sharing
/// `key` and `query` vector loads across all 4 rows.
///
/// # Errors
/// Rejects unequal vector lengths before mutation, or a nonfinite updated element or dot result.
pub(crate) fn step_four_rows_from(
    previous: [&[f32]; 4],
    states: [&mut [f32]; 4],
    key: &[f32],
    query: &[f32],
    decay: f32,
    values: [f32; 4],
    beta: f32,
) -> Result<[f32; 4]> {
    let width = key.len();
    if query.len() != width
        || previous.iter().any(|row| row.len() != width)
        || states.iter().any(|row| row.len() != width)
    {
        return Err(invalid("pinned GDN step requires matching vectors"));
    }
    let [s0, s1, s2, s3] = states;
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(outputs) =
        neon::step_four_rows_from(previous, [s0, s1, s2, s3], key, query, decay, values, beta)
    {
        return Ok(outputs);
    }
    Ok([
        step_row_from(previous[0], s0, key, query, decay, values[0], beta)?,
        step_row_from(previous[1], s1, key, query, decay, values[1], beta)?,
        step_row_from(previous[2], s2, key, query, decay, values[2], beta)?,
        step_row_from(previous[3], s3, key, query, decay, values[3], beta)?,
    ])
}

/// Steps 4 GDN state rows across `inputs.len()` tokens from `previous` into `states`,
/// writing `out * scale` to `output[token * stride + row + r]` for each token and row `r in 0..4`.
///
/// # Errors
/// Rejects invalid geometry before mutation, or a nonfinite updated element or dot result.
pub(crate) fn step_four_rows_tokens_from(
    previous: [&[f32]; 4],
    states: [&mut [f32]; 4],
    inputs: &[ChunkInput<'_>],
    row: usize,
    stride: usize,
    scale: f32,
    output: &mut [f32],
) -> Result<()> {
    if inputs.is_empty() {
        return Err(invalid("empty GDN chunk"));
    }
    if row.checked_add(4).is_none_or(|end| end > stride)
        || previous.iter().any(|prev| prev.len() != stride)
        || states.iter().any(|state| state.len() != stride)
        || inputs.len().checked_mul(stride) != Some(output.len())
        || inputs.iter().any(|input| {
            input.key.len() != stride
                || input.query.len() != stride
                || input.value.len() < row.saturating_add(4)
        })
    {
        return Err(invalid("pinned GDN step requires matching vectors"));
    }
    let [s0, s1, s2, s3] = states;
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if neon::step_four_rows_tokens_from(
        previous,
        [s0, s1, s2, s3],
        inputs,
        row,
        stride,
        scale,
        output,
    ) {
        return Ok(());
    }
    step_row_tokens_from(previous[0], s0, inputs, row, stride, scale, output)?;
    step_row_tokens_from(previous[1], s1, inputs, row + 1, stride, scale, output)?;
    step_row_tokens_from(previous[2], s2, inputs, row + 2, stride, scale, output)?;
    step_row_tokens_from(previous[3], s3, inputs, row + 3, stride, scale, output)
}

/// Steps 8 GDN state rows transactionally from `previous` into `states`.
/// Eligible NEON execution shares `key` and `query` loads within four-row
/// arithmetic phases while retaining atomic eight-row publication.
///
/// # Errors
/// Rejects unequal vector lengths before mutation, or a nonfinite updated element or dot result.
pub(crate) fn step_eight_rows_from(
    previous: [&[f32]; 8],
    states: [&mut [f32]; 8],
    key: &[f32],
    query: &[f32],
    decay: f32,
    values: [f32; 8],
    beta: f32,
) -> Result<[f32; 8]> {
    let width = key.len();
    if query.len() != width
        || previous.iter().any(|row| row.len() != width)
        || states.iter().any(|row| row.len() != width)
    {
        return Err(invalid("pinned GDN step requires matching vectors"));
    }
    let [s0, s1, s2, s3, s4, s5, s6, s7] = states;
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if let Some(outputs) = neon::step_eight_rows_from(
        previous,
        [s0, s1, s2, s3, s4, s5, s6, s7],
        key,
        query,
        decay,
        values,
        beta,
    ) {
        return Ok(outputs);
    }
    Ok([
        step_row_from(previous[0], s0, key, query, decay, values[0], beta)?,
        step_row_from(previous[1], s1, key, query, decay, values[1], beta)?,
        step_row_from(previous[2], s2, key, query, decay, values[2], beta)?,
        step_row_from(previous[3], s3, key, query, decay, values[3], beta)?,
        step_row_from(previous[4], s4, key, query, decay, values[4], beta)?,
        step_row_from(previous[5], s5, key, query, decay, values[5], beta)?,
        step_row_from(previous[6], s6, key, query, decay, values[6], beta)?,
        step_row_from(previous[7], s7, key, query, decay, values[7], beta)?,
    ])
}

/// Steps eight rows into the graph's unpublished recurrent-state delta.
///
/// The destination is disposable scratch: an error may leave any destination
/// row changed. The caller must reject the complete delta on failure and must
/// not use this operation for live state or an externally visible checkpoint.
/// The immutable source remains available for defensive replay and retry.
///
/// # Errors
/// Rejects unequal vector lengths before mutation, or a nonfinite intermediate.
pub(crate) fn step_eight_rows_provisional(
    previous: [&[f32]; 8],
    states: [&mut [f32]; 8],
    key: &[f32],
    query: &[f32],
    decay: f32,
    values: [f32; 8],
    beta: f32,
) -> Result<[f32; 8]> {
    let width = key.len();
    if query.len() != width
        || previous.iter().any(|row| row.len() != width)
        || states.iter().any(|row| row.len() != width)
    {
        return Err(invalid("pinned GDN step requires matching vectors"));
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    {
        neon::step_eight_rows_provisional(previous, states, key, query, decay, values, beta)
    }
    #[cfg(not(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    )))]
    {
        step_eight_rows_from(previous, states, key, query, decay, values, beta)
    }
}

/// Steps 8 GDN state rows across `inputs.len()` tokens from `previous` into `states`,
/// writing `out * scale` to `output[token * stride + row + r]` for each token and row `r in 0..8`.
///
/// # Errors
/// Rejects invalid geometry before mutation, or a nonfinite updated element or dot result.
pub(crate) fn step_eight_rows_tokens_from(
    previous: [&[f32]; 8],
    states: [&mut [f32]; 8],
    inputs: &[ChunkInput<'_>],
    row: usize,
    stride: usize,
    scale: f32,
    output: &mut [f32],
) -> Result<()> {
    if inputs.is_empty() {
        return Err(invalid("empty GDN chunk"));
    }
    if row.checked_add(8).is_none_or(|end| end > stride)
        || previous.iter().any(|prev| prev.len() != stride)
        || states.iter().any(|state| state.len() != stride)
        || inputs.len().checked_mul(stride) != Some(output.len())
        || inputs.iter().any(|input| {
            input.key.len() != stride
                || input.query.len() != stride
                || input.value.len() < row.saturating_add(8)
        })
    {
        return Err(invalid("pinned GDN step requires matching vectors"));
    }
    let [s0, s1, s2, s3, s4, s5, s6, s7] = states;
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if neon::step_eight_rows_tokens_from(
        previous,
        [s0, s1, s2, s3, s4, s5, s6, s7],
        inputs,
        row,
        stride,
        scale,
        output,
    ) {
        return Ok(());
    }
    step_row_tokens_from(previous[0], s0, inputs, row, stride, scale, output)?;
    step_row_tokens_from(previous[1], s1, inputs, row + 1, stride, scale, output)?;
    step_row_tokens_from(previous[2], s2, inputs, row + 2, stride, scale, output)?;
    step_row_tokens_from(previous[3], s3, inputs, row + 3, stride, scale, output)?;
    step_row_tokens_from(previous[4], s4, inputs, row + 4, stride, scale, output)?;
    step_row_tokens_from(previous[5], s5, inputs, row + 5, stride, scale, output)?;
    step_row_tokens_from(previous[6], s6, inputs, row + 6, stride, scale, output)?;
    step_row_tokens_from(previous[7], s7, inputs, row + 7, stride, scale, output)
}

/// Steps all `value_dim` rows of a GDN head state in-place across `inputs.len()` tokens,
/// batching 8 rows and 4 rows in registers/L1 when supported.
///
/// # Errors
/// Rejects invalid geometry or any non-finite intermediate/output value.
#[expect(
    clippy::too_many_lines,
    reason = "Keep 8-row, 4-row, and 1-row in-place head recurrence dispatch together."
)]
pub(crate) fn step_head_tokens_in_place(
    state_head: &mut [f32],
    value_dim: usize,
    key_dim: usize,
    inputs: &[ChunkInput<'_>],
) -> Result<()> {
    if inputs.is_empty()
        || value_dim.checked_mul(key_dim) != Some(state_head.len())
        || inputs.iter().any(|input| {
            input.key.len() != key_dim
                || input.query.len() != key_dim
                || input.value.len() != value_dim
        })
    {
        return Err(invalid("pinned GDN step requires matching vectors"));
    }
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    {
        let (rows8, rem8) = state_head.split_at_mut((value_dim / 8) * 8 * key_dim);
        for (chunk8_idx, chunk8) in rows8.chunks_exact_mut(8 * key_dim).enumerate() {
            let row = chunk8_idx * 8;
            let stepped = {
                let (s03, s47) = chunk8.split_at_mut(4 * key_dim);
                let (s01, s23) = s03.split_at_mut(2 * key_dim);
                let (s0, s1) = s01.split_at_mut(key_dim);
                let (s2, s3) = s23.split_at_mut(key_dim);
                let (s45, s67) = s47.split_at_mut(2 * key_dim);
                let (s4, s5) = s45.split_at_mut(key_dim);
                let (s6, s7) = s67.split_at_mut(key_dim);
                neon::step_eight_rows_tokens_in_place([s0, s1, s2, s3, s4, s5, s6, s7], inputs, row)
            };
            if !stepped {
                for input in inputs {
                    for (r, state_row) in chunk8.chunks_exact_mut(key_dim).enumerate() {
                        step_row_in_place(
                            state_row,
                            input.key,
                            input.query,
                            input.decay,
                            input.value[row + r],
                            input.beta,
                        )?;
                    }
                }
            }
        }
        let base4 = (value_dim / 8) * 8;
        let rem_rows = value_dim - base4;
        let (rows4, rem4) = rem8.split_at_mut((rem_rows / 4) * 4 * key_dim);
        for (chunk4_idx, chunk4) in rows4.chunks_exact_mut(4 * key_dim).enumerate() {
            let row = base4 + chunk4_idx * 4;
            let stepped = {
                let (s01, s23) = chunk4.split_at_mut(2 * key_dim);
                let (s0, s1) = s01.split_at_mut(key_dim);
                let (s2, s3) = s23.split_at_mut(key_dim);
                neon::step_four_rows_tokens_in_place([s0, s1, s2, s3], inputs, row)
            };
            if !stepped {
                for input in inputs {
                    for (r, state_row) in chunk4.chunks_exact_mut(key_dim).enumerate() {
                        step_row_in_place(
                            state_row,
                            input.key,
                            input.query,
                            input.decay,
                            input.value[row + r],
                            input.beta,
                        )?;
                    }
                }
            }
        }
        let base1 = base4 + (rem_rows / 4) * 4;
        for input in inputs {
            for (r, state_row) in rem4.chunks_exact_mut(key_dim).enumerate() {
                step_row_in_place(
                    state_row,
                    input.key,
                    input.query,
                    input.decay,
                    input.value[base1 + r],
                    input.beta,
                )?;
            }
        }
        Ok(())
    }
    #[cfg(not(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    )))]
    {
        for input in inputs {
            for (row_idx, &v_elem) in input.value.iter().enumerate() {
                let row_state = &mut state_head[row_idx * key_dim..(row_idx + 1) * key_dim];
                step_row_in_place(
                    row_state,
                    input.key,
                    input.query,
                    input.decay,
                    v_elem,
                    input.beta,
                )?;
            }
        }
        Ok(())
    }
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Keep the scalar oracle's exact multiply/add remainder rounding."
)]
fn scalar_add_scaled(target: &mut [f32], input: &[f32], scale: f32) -> Result<()> {
    let vectorized = target.len() / 16 * 16;
    for (index, (target, &input)) in target.iter_mut().zip(input).enumerate() {
        *target = super::finite(if index < vectorized {
            scale.mul_add(input, *target)
        } else {
            *target + scale * input
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        ChunkInput, add_scaled, decay_and_dot, decay_and_dot_from, dot, dot_four, dot_pair,
        dot_pair_eight_rows, dot_pair_four_rows, dot_scalar, scalar_add_scaled,
        step_eight_rows_from, step_eight_rows_tokens_from, step_four_rows_from,
        step_four_rows_tokens_from, step_row_from, step_row_in_place, step_row_tokens_from,
        update_and_dot,
    };
    use crate::Result;

    fn vectors(width: usize) -> (Vec<f32>, Vec<f32>) {
        let mut seed = 0x2468_ace1_u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            // Mixed signs and mantissas with a fixed finite exponent. These
            // distinguish the four-register tree from a scalar running sum.
            f32::from_bits((seed & 0x807f_ffff) | 0x3d80_0000)
        };
        (
            (0..width).map(|_| next()).collect(),
            (0..width).map(|_| next()).collect(),
        )
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "Verifies single-token and multi-token GDN parity across all widths and decay/beta cases."
    )]
    fn pinned_gdn_dispatch_matches_scalar_bits_and_tail() -> Result<()> {
        for width in (0..=33).chain([63, 64, 65, 127, 128, 129, 255, 256, 257, 5120]) {
            let (first, second) = vectors(width);
            assert_eq!(
                dot(&first, &second)?.to_bits(),
                dot_scalar(&first, &second)?.to_bits()
            );
            let (pair_first, pair_second) = dot_pair(&first, &second, &first)?;
            assert_eq!(pair_first.to_bits(), dot_scalar(&first, &second)?.to_bits());
            assert_eq!(pair_second.to_bits(), dot_scalar(&first, &first)?.to_bits());
            for scale in [0.0, -0.0, 0.375, -1.125, f32::from_bits(1)] {
                let mut expected = first.clone();
                let mut actual = first.clone();
                scalar_add_scaled(&mut expected, &second, scale)?;
                add_scaled(&mut actual, &second, scale)?;
                assert_eq!(
                    bits(&actual),
                    bits(&expected),
                    "width {width}, scale {scale}"
                );
                let mut decayed_expected = first.clone();
                for elem in &mut decayed_expected {
                    *elem *= scale;
                }
                let expected_dot = dot_scalar(&decayed_expected, &second)?;
                let mut decayed_actual = first.clone();
                let actual_dot = decay_and_dot(&mut decayed_actual, &second, scale)?;
                assert_eq!(actual_dot.to_bits(), expected_dot.to_bits());
                assert_eq!(bits(&decayed_actual), bits(&decayed_expected));
                let mut decayed_from = vec![f32::NAN; width];
                let from_dot = decay_and_dot_from(&first, &mut decayed_from, &second, scale)?;
                assert_eq!(from_dot.to_bits(), expected_dot.to_bits());
                assert_eq!(bits(&decayed_from), bits(&decayed_expected));

                let mut upd_expected = first.clone();
                scalar_add_scaled(&mut upd_expected, &second, scale)?;
                let expected_q_dot = dot_scalar(&upd_expected, &first)?;
                let mut upd_actual = first.clone();
                let actual_q_dot = update_and_dot(&mut upd_actual, &second, scale, &first)?;
                assert_eq!(actual_q_dot.to_bits(), expected_q_dot.to_bits());
                assert_eq!(bits(&upd_actual), bits(&upd_expected));

                let value = 0.75_f32;
                let beta = 0.625_f32;
                let mut step_ref_state = vec![f32::NAN; width];
                let ref_dot_k = decay_and_dot_from(&first, &mut step_ref_state, &second, scale)?;
                let ref_delta = (value - ref_dot_k) * beta;
                let ref_out = update_and_dot(&mut step_ref_state, &second, ref_delta, &first)?;

                let mut step_from_state = vec![f32::NAN; width];
                let from_out = step_row_from(
                    &first,
                    &mut step_from_state,
                    &second,
                    &first,
                    scale,
                    value,
                    beta,
                )?;
                assert_eq!(from_out.to_bits(), ref_out.to_bits());
                assert_eq!(bits(&step_from_state), bits(&step_ref_state));

                let mut step_in_place_state = first.clone();
                let in_place_out = step_row_in_place(
                    &mut step_in_place_state,
                    &second,
                    &first,
                    scale,
                    value,
                    beta,
                )?;
                assert_eq!(in_place_out.to_bits(), ref_out.to_bits());
                assert_eq!(bits(&step_in_place_state), bits(&step_ref_state));

                if width > 0 {
                    let row = width / 2;
                    let out_scale = 0.125_f32;
                    let chunk_inputs = [
                        ChunkInput {
                            query: &first,
                            key: &second,
                            value: &first,
                            decay: scale,
                            beta,
                        },
                        ChunkInput {
                            query: &second,
                            key: &first,
                            value: &second,
                            decay: 0.75,
                            beta: 0.375,
                        },
                        ChunkInput {
                            query: &first,
                            key: &second,
                            value: &first,
                            decay: 0.5,
                            beta: 0.875,
                        },
                    ];
                    let mut seq_state = vec![f32::NAN; width];
                    let mut seq_out = vec![0.0_f32; 3 * width];
                    seq_out[row] = step_row_from(
                        &first,
                        &mut seq_state,
                        chunk_inputs[0].key,
                        chunk_inputs[0].query,
                        chunk_inputs[0].decay,
                        chunk_inputs[0].value[row],
                        chunk_inputs[0].beta,
                    )? * out_scale;
                    for (tok, inp) in chunk_inputs.iter().enumerate().skip(1) {
                        seq_out[tok * width + row] = step_row_in_place(
                            &mut seq_state,
                            inp.key,
                            inp.query,
                            inp.decay,
                            inp.value[row],
                            inp.beta,
                        )? * out_scale;
                    }
                    let mut multi_state = vec![f32::NAN; width];
                    let mut multi_out = vec![0.0_f32; 3 * width];
                    step_row_tokens_from(
                        &first,
                        &mut multi_state,
                        &chunk_inputs,
                        row,
                        width,
                        out_scale,
                        &mut multi_out,
                    )?;
                    assert_eq!(bits(&multi_state), bits(&seq_state));
                    assert_eq!(bits(&multi_out), bits(&seq_out));
                }
            }
        }
        Ok(())
    }

    #[test]
    #[expect(
        clippy::suboptimal_flops,
        reason = "The regression must retain the unfused scalar remainder oracle."
    )]
    fn pinned_gdn_retains_fused_vector_and_nonfused_tail_rounding() -> Result<()> {
        let scale = f32::from_bits(0x3f80_0001);
        let input = f32::from_bits(0x3f7f_fffe);
        let fused = scale.mul_add(input, -1.0);
        let nonfused = scale * input - 1.0;
        assert_ne!(fused.to_bits(), nonfused.to_bits());
        for width in [15, 16, 17, 31, 32, 33, 128, 129] {
            let mut actual = vec![-1.0; width];
            add_scaled(&mut actual, &vec![input; width], scale)?;
            let vectorized = width / 16 * 16;
            assert!(
                actual[..vectorized]
                    .iter()
                    .all(|value| value.to_bits() == fused.to_bits())
            );
            assert!(
                actual[vectorized..]
                    .iter()
                    .all(|value| value.to_bits() == nonfused.to_bits())
            );
        }
        Ok(())
    }

    #[test]
    fn pinned_gdn_preserves_exceptional_bits_and_first_failure_mutation() -> Result<()> {
        let values = [
            0.0,
            -0.0,
            f32::from_bits(1),
            -f32::from_bits(1),
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            1.0,
            -1.0,
        ];
        for width in [0, 1, 15, 16, 17, 31, 32, 33, 128, 129] {
            let first = (0..width)
                .map(|index| values[index % values.len()])
                .collect::<Vec<_>>();
            let second = (0..width)
                .map(|index| values[(index + 3) % values.len()])
                .collect::<Vec<_>>();
            assert_eq!(
                dot(&first, &second)?.to_bits(),
                dot_scalar(&first, &second)?.to_bits()
            );
            for scale in [0.0, -0.0, 0.5, -1.0] {
                let mut actual = first.clone();
                let mut expected = first.clone();
                add_scaled(&mut actual, &second, scale)?;
                scalar_add_scaled(&mut expected, &second, scale)?;
                assert_eq!(bits(&actual), bits(&expected));
            }
        }
        for width in [17, 33] {
            for failing in 0..width {
                for invalid in [f32::MAX, f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
                    let mut input = vec![0.25; width];
                    input[failing] = invalid;
                    let mut actual = vec![1.0; width];
                    let mut expected = actual.clone();
                    assert!(scalar_add_scaled(&mut expected, &input, 2.0).is_err());
                    assert!(add_scaled(&mut actual, &input, 2.0).is_err());
                    assert_eq!(
                        bits(&actual),
                        bits(&expected),
                        "width {width}, failing lane {failing}"
                    );
                    assert_eq!(actual[failing].to_bits(), 1.0_f32.to_bits());
                    assert!(
                        actual[failing..]
                            .iter()
                            .all(|value| value.to_bits() == 1.0_f32.to_bits())
                    );
                }
            }
        }
        for nonfinite in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            let mut actual = [1.0; 17];
            assert!(add_scaled(&mut actual, &[0.0; 17], nonfinite).is_err());
            assert_eq!(actual.map(f32::to_bits), [1.0_f32.to_bits(); 17]);
            let mut input = [1.0; 17];
            input[3] = nonfinite;
            assert!(dot(&input, &[1.0; 17]).is_err());
        }
        Ok(())
    }

    #[test]
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    fn pinned_gdn_neon_executes_exact_vector_and_tail_contract() -> Result<()> {
        let (first, second) = vectors(129);
        let actual = super::neon::dot(&first, &second);
        assert!(
            actual.is_some(),
            "qualified baseline must provide a NEON token"
        );
        assert_eq!(
            actual.map(f32::to_bits),
            Some(dot_scalar(&first, &second)?.to_bits())
        );
        let mut expected = first.clone();
        let mut actual = first;
        scalar_add_scaled(&mut expected, &second, -0.375)?;
        let result = super::neon::add_scaled(&mut actual, &second, -0.375);
        assert!(result.is_some_and(|result| result.is_ok()));
        assert_eq!(bits(&actual), bits(&expected));
        Ok(())
    }

    #[test]
    fn pinned_gdn_rejects_mismatched_rows_and_nonfinite_results() {
        let mut target = [1.0, 2.0];
        assert!(add_scaled(&mut target, &[3.0], 4.0).is_err());
        assert_eq!(
            target.map(f32::to_bits),
            [1.0_f32.to_bits(), 2.0_f32.to_bits()]
        );
        assert!(dot(&[1.0], &[]).is_err());
        assert!(dot_scalar(&[1.0], &[]).is_err());
        assert!(dot(&[f32::MAX], &[2.0]).is_err());
        assert!(dot(&[f32::MAX; 16], &[2.0; 16]).is_err());
        assert!(add_scaled(&mut target, &[f32::MAX, 1.0], 2.0).is_err());
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        clippy::similar_names,
        reason = "Cover 4-row and 8-row GDN step and WY update bit parity and error rollback together."
    )]
    fn pinned_gdn_four_rows_matches_single_row_bits_and_error_state() -> Result<()> {
        for width in [16, 17, 64, 128, 129] {
            let (first, second) = vectors(width * 4);
            let rows = [
                &first[0..width],
                &first[width..2 * width],
                &first[2 * width..3 * width],
                &first[3 * width..4 * width],
            ];
            let k = &second[0..width];
            let q = &second[width..2 * width];
            let pairs = dot_pair_four_rows(rows, k, q)?;
            for (r, row) in rows.iter().enumerate() {
                let expected = dot_pair(row, k, q)?;
                assert_eq!(pairs[r].0.to_bits(), expected.0.to_bits());
                assert_eq!(pairs[r].1.to_bits(), expected.1.to_bits());
            }

            let values = [0.25, -0.5, 0.75, -0.125];
            let decay = 0.875;
            let beta = 0.625;
            let mut batch_states = vec![0.0_f32; width * 4];
            let [bs0, rest @ ..] = batch_states.as_chunks_mut::<1>().0 else {
                unreachable!()
            };
            let _ = (bs0, rest);
            let (s01, s23) = batch_states.split_at_mut(width * 2);
            let (s0, s1) = s01.split_at_mut(width);
            let (s2, s3) = s23.split_at_mut(width);
            let batch_out = step_four_rows_from(rows, [s0, s1, s2, s3], k, q, decay, values, beta)?;
            for r in 0..4 {
                let mut ref_state = vec![0.0_f32; width];
                let ref_out = step_row_from(rows[r], &mut ref_state, k, q, decay, values[r], beta)?;
                assert_eq!(batch_out[r].to_bits(), ref_out.to_bits());
                assert_eq!(
                    bits(&batch_states[r * width..(r + 1) * width]),
                    bits(&ref_state)
                );
            }

            let inputs = [
                ChunkInput {
                    query: q,
                    key: k,
                    value: &second[0..width],
                    decay: 0.875,
                    beta: 0.625,
                },
                ChunkInput {
                    query: k,
                    key: q,
                    value: &second[width..2 * width],
                    decay: 0.75,
                    beta: 0.375,
                },
            ];
            let mut tok_batch_states = vec![0.0_f32; width * 4];
            let mut tok_batch_out = vec![0.0_f32; inputs.len() * width];
            {
                let (low_pair, high_pair) = tok_batch_states.split_at_mut(width * 2);
                let (out_a, out_b) = low_pair.split_at_mut(width);
                let (out_c, out_d) = high_pair.split_at_mut(width);
                step_four_rows_tokens_from(
                    rows,
                    [out_a, out_b, out_c, out_d],
                    &inputs,
                    0,
                    width,
                    0.5,
                    &mut tok_batch_out,
                )?;
            }
            let mut tok_ref_states = vec![0.0_f32; width * 4];
            let mut tok_ref_out = vec![0.0_f32; inputs.len() * width];
            for r in 0..4 {
                step_row_tokens_from(
                    rows[r],
                    &mut tok_ref_states[r * width..(r + 1) * width],
                    &inputs,
                    r,
                    width,
                    0.5,
                    &mut tok_ref_out,
                )?;
            }
            assert_eq!(bits(&tok_batch_states), bits(&tok_ref_states));
            assert_eq!(bits(&tok_batch_out), bits(&tok_ref_out));

            let (first8, _) = vectors(width * 8);
            let rows8: [&[f32]; 8] = std::array::from_fn(|r| &first8[r * width..(r + 1) * width]);
            let pairs8 = dot_pair_eight_rows(rows8, k, q)?;
            for (r, row) in rows8.iter().enumerate() {
                let expected = dot_pair(row, k, q)?;
                assert_eq!(pairs8[r].0.to_bits(), expected.0.to_bits());
                assert_eq!(pairs8[r].1.to_bits(), expected.1.to_bits());
            }

            let dots4 = dot_four(k, [rows8[0], rows8[1], rows8[2], rows8[3]])?;
            for (r, &actual) in dots4.iter().enumerate() {
                assert_eq!(actual.to_bits(), dot(k, rows8[r])?.to_bits());
            }

            let values8 = [0.25, -0.5, 0.75, -0.125, 0.5, -0.25, 0.125, -0.75];
            let mut batch8_states = vec![0.0_f32; width * 8];
            let batch8_out = {
                let (h0, h1) = batch8_states.split_at_mut(width * 4);
                let (s01, s23) = h0.split_at_mut(width * 2);
                let (s45, s67) = h1.split_at_mut(width * 2);
                let (s0, s1) = s01.split_at_mut(width);
                let (s2, s3) = s23.split_at_mut(width);
                let (s4, s5) = s45.split_at_mut(width);
                let (s6, s7) = s67.split_at_mut(width);
                step_eight_rows_from(
                    rows8,
                    [s0, s1, s2, s3, s4, s5, s6, s7],
                    k,
                    q,
                    decay,
                    values8,
                    beta,
                )?
            };
            for r in 0..8 {
                let mut ref_state = vec![0.0_f32; width];
                let ref_out =
                    step_row_from(rows8[r], &mut ref_state, k, q, decay, values8[r], beta)?;
                assert_eq!(batch8_out[r].to_bits(), ref_out.to_bits());
                assert_eq!(
                    bits(&batch8_states[r * width..(r + 1) * width]),
                    bits(&ref_state)
                );
            }

            let mut tok8_states = vec![0.0_f32; width * 8];
            let mut tok8_out = vec![0.0_f32; inputs.len() * width];
            {
                let (h0, h1) = tok8_states.split_at_mut(width * 4);
                let (s01, s23) = h0.split_at_mut(width * 2);
                let (s45, s67) = h1.split_at_mut(width * 2);
                let (s0, s1) = s01.split_at_mut(width);
                let (s2, s3) = s23.split_at_mut(width);
                let (s4, s5) = s45.split_at_mut(width);
                let (s6, s7) = s67.split_at_mut(width);
                step_eight_rows_tokens_from(
                    rows8,
                    [s0, s1, s2, s3, s4, s5, s6, s7],
                    &inputs,
                    0,
                    width,
                    0.5,
                    &mut tok8_out,
                )?;
            }
            let mut tok8_ref_states = vec![0.0_f32; width * 8];
            let mut tok8_ref_out = vec![0.0_f32; inputs.len() * width];
            for r in 0..8 {
                step_row_tokens_from(
                    rows8[r],
                    &mut tok8_ref_states[r * width..(r + 1) * width],
                    &inputs,
                    r,
                    width,
                    0.5,
                    &mut tok8_ref_out,
                )?;
            }
            assert_eq!(bits(&tok8_states), bits(&tok8_ref_states));
            assert_eq!(bits(&tok8_out), bits(&tok8_ref_out));
            let inputs8 = inputs.map(|inp| ChunkInput {
                value: &inp.value[..8],
                ..inp
            });
            let mut in_place_head = first8.clone();
            super::step_head_tokens_in_place(&mut in_place_head, 8, width, &inputs8)?;
            assert_eq!(bits(&in_place_head), bits(&tok8_ref_states));
        }
        Ok(())
    }
}
