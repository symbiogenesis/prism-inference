//! Bounded Qwen3.5 text decoder with full attention and Gated Delta Net.
//!
//! The operation ordering follows Prism's `src/models/qwen35.cpp`. KV values
//! are stored as IEEE binary16 bits; convolution and recurrent state
//! remain binary32. This correctness path does not establish device parity.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::sync::Arc;

use bonsai_gguf::{Metadata, MetadataArray, MetadataValue, MetadataValueRef};
use bonsai_ternary::half_to_float;

use crate::kernels::pinned_activations::{silu_in_place, swiglu_in_place};
use crate::kernels::{
    MatvecBackend, MatvecDiagnostics, MatvecScratch, MatvecStatistics, RowParallelism,
    add_rms_norm_into, add_rms_norm_pinned_into, add_scaled_slice_in_place, add_slice_in_place,
    add_slice_in_place_all_finite, append_half_bits, f16_dot, float_dot, is_finite_slice,
    l2_norm_in_place, mul_sigmoid_in_place, mul_slice_in_place, mul_slice_into, rms_norm_into,
    rms_norm_pinned_into, rms_norm_swiglu_in_place, rms_norm_swiglu_into, round_f16_slice, row,
    row_into, scale_slice_into, silu, softmax_round_f16,
};
use crate::model::lora::{RankOneAdapter, Scratch as AdapterScratch};
use crate::model::{Model, Tensor};
use crate::{Decoder, Result, invalid};

mod projection;
pub use projection::ProjectedQwen35;
mod prefill;
pub use prefill::TiledPrefillStatistics;

/// Maximum admitted live KV/recurrent element payload, also the default budget.
///
/// This excludes model storage, scratch, checkpoints, page-table/Arc metadata
/// and allocator overhead.
pub const MAX_STATE_BYTES: usize = 4 * 1024 * 1024 * 1024;
/// Graph identity suffix for the generic source-order Qwen35 text decoder.
///
/// This identifies numerical/state layout, without portable checkpoint support
/// or model/device qualification. Complete decoder identities also include the
/// verified model SHA-256.
pub const QWEN35_GRAPH_CONTRACT: &str = "qwen35-text-v1:nonflash-f16:kv-f16:gdn-f32";
/// Graph identity suffix selected only for the exact pinned Swift representations.
///
/// This includes the pinned ARM64 arithmetic ordering. It does not qualify
/// other model artifacts or promise portable state or target-device parity.
pub const SWIFT_GRAPH_CONTRACT: &str =
    "qwen35-text-v1:nonflash-f16:kv-f16:gdn-f32:prism-arm64-fused-v1";
const MAX_WIDTH: usize = 1 << 20;

#[derive(Clone, Copy)]
enum Arithmetic {
    Source,
    PinnedSwift,
}

impl Arithmetic {
    fn normalize(
        self,
        input: &[f32],
        weight: &[f32],
        epsilon: f32,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        match self {
            Self::Source => rms_norm_into(input, weight, epsilon, output),
            Self::PinnedSwift => rms_norm_pinned_into(input, weight, epsilon, output),
        }
    }

    fn add_normalize(
        self,
        values: &mut [f32],
        delta: &[f32],
        weight: &[f32],
        epsilon: f32,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        match self {
            Self::Source => add_rms_norm_into(values, delta, weight, epsilon, output),
            Self::PinnedSwift => add_rms_norm_pinned_into(values, delta, weight, epsilon, output),
        }
    }

    fn silu(self, values: &mut [f32]) {
        match self {
            Self::Source => values.iter_mut().for_each(|value| *value = silu(*value)),
            Self::PinnedSwift => silu_in_place(values),
        }
    }

    fn swiglu_in_place(self, gate: &[f32], up: &mut [f32]) -> Result<()> {
        match self {
            Self::Source => {
                for (up, &gate) in up.iter_mut().zip(gate) {
                    *up *= silu(gate);
                }
                Ok(())
            }
            Self::PinnedSwift => swiglu_in_place(gate, up),
        }
    }

    fn swiglu_in_place_with_parallelism(
        self,
        gate: &[f32],
        up: &mut [f32],
        parallelism: &RowParallelism,
    ) -> Result<()> {
        if gate.len() != up.len() {
            return Err(invalid("SwiGLU requires equal gate and up vector lengths"));
        }
        if up.len() >= 1024 && parallelism.workers() > 1 {
            return parallelism.update_chunks(up, 256, |chunk_idx, up_chunk| {
                let start = chunk_idx * 256;
                let gate_chunk = &gate[start..start + up_chunk.len()];
                self.swiglu_in_place(gate_chunk, up_chunk)
            });
        }
        self.swiglu_in_place(gate, up)
    }

    fn normalize_swiglu_in_place(
        self,
        values: &mut [f32],
        weight: &[f32],
        epsilon: f32,
        gate: &[f32],
    ) -> Result<()> {
        match self {
            Self::Source => rms_norm_swiglu_in_place::<false>(values, weight, epsilon, gate),
            Self::PinnedSwift => rms_norm_swiglu_in_place::<true>(values, weight, epsilon, gate),
        }
    }

    fn normalize_swiglu_into(
        self,
        input: &[f32],
        weight: &[f32],
        epsilon: f32,
        gate: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        match self {
            Self::Source => rms_norm_swiglu_into::<false>(input, weight, epsilon, gate, output),
            Self::PinnedSwift => rms_norm_swiglu_into::<true>(input, weight, epsilon, gate, output),
        }
    }

    fn normalize_rope_round_f16_into(
        self,
        input: &[f32],
        weight: &[f32],
        epsilon: f32,
        geometry: &Geometry,
        frequencies: &[(f32, f32)],
        output: &mut [f32],
    ) -> Result<bool> {
        match self {
            Self::Source => crate::kernels::rms_norm_rope_round_f16_into::<false>(
                input,
                weight,
                epsilon,
                geometry.head,
                geometry.rope,
                frequencies,
                output,
            ),
            Self::PinnedSwift => crate::kernels::rms_norm_rope_round_f16_into::<true>(
                input,
                weight,
                epsilon,
                geometry.head,
                geometry.rope,
                frequencies,
                output,
            ),
        }
    }

    fn normalize_rope_round_f16_in_place(
        self,
        values: &mut [f32],
        weight: &[f32],
        epsilon: f32,
        geometry: &Geometry,
        frequencies: &[(f32, f32)],
    ) -> Result<bool> {
        match self {
            Self::Source => crate::kernels::rms_norm_rope_round_f16_in_place::<false>(
                values,
                weight,
                epsilon,
                geometry.head,
                geometry.rope,
                frequencies,
            ),
            Self::PinnedSwift => crate::kernels::rms_norm_rope_round_f16_in_place::<true>(
                values,
                weight,
                epsilon,
                geometry.head,
                geometry.rope,
                frequencies,
            ),
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Pass explicit RMSNorm parameters, Hadamard block geometry, and reusable output buffers."
    )]
    fn normalize_hadamard_into(
        self,
        input: &[f32],
        weight: &[f32],
        epsilon: f32,
        block: usize,
        signs: Option<&[f32]>,
        parallelism: &RowParallelism,
        norm_out: &mut Vec<f32>,
        rotated_out: &mut Vec<f32>,
    ) -> Result<()> {
        match self {
            Self::Source => crate::kernels::rms_norm_hadamard_into::<false>(
                input,
                weight,
                epsilon,
                block,
                signs,
                parallelism,
                norm_out,
                rotated_out,
            ),
            Self::PinnedSwift => crate::kernels::rms_norm_hadamard_into::<true>(
                input,
                weight,
                epsilon,
                block,
                signs,
                parallelism,
                norm_out,
                rotated_out,
            ),
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Pass explicit residual delta, RMSNorm parameters, Hadamard block geometry, and reusable output buffers."
    )]
    fn add_normalize_hadamard_into(
        self,
        values: &mut [f32],
        delta: &[f32],
        weight: &[f32],
        epsilon: f32,
        block: usize,
        signs: Option<&[f32]>,
        parallelism: &RowParallelism,
        norm_out: &mut Vec<f32>,
        rotated_out: &mut Vec<f32>,
    ) -> Result<()> {
        match self {
            Self::Source => crate::kernels::add_rms_norm_hadamard_into::<false>(
                values,
                delta,
                weight,
                epsilon,
                block,
                signs,
                parallelism,
                norm_out,
                rotated_out,
            ),
            Self::PinnedSwift => crate::kernels::add_rms_norm_hadamard_into::<true>(
                values,
                delta,
                weight,
                epsilon,
                block,
                signs,
                parallelism,
                norm_out,
                rotated_out,
            ),
        }
    }

    fn swiglu_hadamard_read_into_with_parallelism(
        self,
        gate: &[f32],
        up: &[f32],
        block: usize,
        signs: Option<&[f32]>,
        parallelism: &RowParallelism,
        rotated_out: &mut Vec<f32>,
    ) -> Result<()> {
        if gate.len() != up.len()
            || block == 0
            || !block.is_power_of_two()
            || !up.len().is_multiple_of(block)
            || signs.is_some_and(|s| s.len() < up.len())
        {
            return Err(invalid("Hadamard requires a finite power-of-two block"));
        }
        resize_into(up.len(), rotated_out)?;
        if parallelism.workers() > 1 && block >= 256 && up.len() >= block.saturating_mul(16) {
            return parallelism.update_chunks(rotated_out, block, |base_block, chunk| {
                for (local_idx, out_block) in chunk.chunks_exact_mut(block).enumerate() {
                    let offset = (base_block + local_idx) * block;
                    let g_block = &gate[offset..offset + block];
                    let u_block = &up[offset..offset + block];
                    let s_block = signs.map(|s| &s[offset..offset + block]);
                    match self {
                        Self::Source => crate::kernels::swiglu_hadamard_block_read_into::<false>(
                            g_block, u_block, s_block, out_block,
                        )?,
                        Self::PinnedSwift => {
                            crate::kernels::swiglu_hadamard_block_read_into::<true>(
                                g_block, u_block, s_block, out_block,
                            )?;
                        }
                    }
                }
                Ok(())
            });
        }
        for (block_idx, out_block) in rotated_out.chunks_exact_mut(block).enumerate() {
            let offset = block_idx * block;
            let g_block = &gate[offset..offset + block];
            let u_block = &up[offset..offset + block];
            let s_block = signs.map(|s| &s[offset..offset + block]);
            match self {
                Self::Source => crate::kernels::swiglu_hadamard_block_read_into::<false>(
                    g_block, u_block, s_block, out_block,
                )?,
                Self::PinnedSwift => crate::kernels::swiglu_hadamard_block_read_into::<true>(
                    g_block, u_block, s_block, out_block,
                )?,
            }
        }
        Ok(())
    }

    fn scratch(self, backend: MatvecBackend) -> MatvecScratch {
        match self {
            Self::Source => MatvecScratch::with_backend(backend),
            Self::PinnedSwift => MatvecScratch::with_pinned_backend(backend),
        }
    }

    fn gdn_dot(self, first: &[f32], second: &[f32]) -> Result<f32> {
        match self {
            Self::Source => Ok(dot(first, second)),
            Self::PinnedSwift => crate::kernels::pinned_gdn::dot(first, second),
        }
    }

    fn gdn_dot_pair(self, state: &[f32], first: &[f32], second: &[f32]) -> Result<(f32, f32)> {
        match self {
            Self::Source => Ok(crate::kernels::float_dot_pair(state, first, second)),
            Self::PinnedSwift => crate::kernels::pinned_gdn::dot_pair(state, first, second),
        }
    }

    fn gdn_dot_pair_four_rows(
        self,
        states: [&[f32]; 4],
        first: &[f32],
        second: &[f32],
    ) -> Result<[(f32, f32); 4]> {
        match self {
            Self::Source => Ok(crate::kernels::float_dot_pair_four_rows(
                states, first, second,
            )),
            Self::PinnedSwift => {
                crate::kernels::pinned_gdn::dot_pair_four_rows(states, first, second)
            }
        }
    }

    fn gdn_dot_pair_eight_rows(
        self,
        states: [&[f32]; 8],
        first: &[f32],
        second: &[f32],
    ) -> Result<[(f32, f32); 8]> {
        match self {
            Self::Source => Ok(crate::kernels::float_dot_pair_eight_rows(
                states, first, second,
            )),
            Self::PinnedSwift => {
                crate::kernels::pinned_gdn::dot_pair_eight_rows(states, first, second)
            }
        }
    }

    fn gdn_decay_and_dot(self, state: &mut [f32], key: &[f32], decay: f32) -> Result<f32> {
        match self {
            Self::Source => {
                for element in state.iter_mut() {
                    *element *= decay;
                }
                Ok(dot(state, key))
            }
            Self::PinnedSwift => crate::kernels::pinned_gdn::decay_and_dot(state, key, decay),
        }
    }

    fn gdn_decay_and_dot_from(
        self,
        previous: &[f32],
        state: &mut [f32],
        key: &[f32],
        decay: f32,
    ) -> Result<f32> {
        match self {
            Self::Source => {
                scale_slice_into(previous, decay, state);
                Ok(dot(state, key))
            }
            Self::PinnedSwift => {
                crate::kernels::pinned_gdn::decay_and_dot_from(previous, state, key, decay)
            }
        }
    }

    #[expect(
        clippy::suboptimal_flops,
        reason = "Preserve separate multiply/add rounding for generic graph convolution."
    )]
    fn convolution_row(self, history: &[f32], projected: f32, coefficients: &[f32]) -> Result<f32> {
        match self {
            Self::Source => {
                let mut sum = 0.0;
                for (&old, &coefficient) in history.iter().zip(coefficients) {
                    sum += old * coefficient;
                }
                sum += projected * coefficients[history.len()];
                Ok(sum)
            }
            Self::PinnedSwift => {
                crate::kernels::pinned_convolution::row(history, projected, coefficients)
            }
        }
    }

    fn attention_dot(self, query: &[f32], key: &[u16]) -> Result<f32> {
        match self {
            Self::Source => Ok(dot_f16(query, key)),
            Self::PinnedSwift => crate::kernels::pinned_attention::dot_f16(key, query),
        }
    }

    fn attention_current_dot(self, query: &[f32], key: &[f32]) -> Result<f32> {
        match self {
            Self::Source => Ok(dot(query, key)),
            Self::PinnedSwift => crate::kernels::pinned_attention::dot_f32(key, query),
        }
    }

    fn attention_dot_four(self, queries: [&[f32]; 4], key: &[u16]) -> Result<[f32; 4]> {
        match self {
            Self::Source => Ok(crate::kernels::f16_dot_four(queries, key)),
            Self::PinnedSwift => crate::kernels::pinned_attention::dot_f16_four(key, queries),
        }
    }

    fn attention_dot_two_token_four_queries(
        self,
        queries0: [&[f32]; 4],
        queries1: [&[f32]; 4],
        key: &[u16],
    ) -> Result<([f32; 4], [f32; 4])> {
        match self {
            Self::Source => Ok(crate::kernels::f16_dot_two_token_four_queries(
                queries0, queries1, key,
            )),
            Self::PinnedSwift => crate::kernels::pinned_attention::dot_f16_two_token_four_queries(
                key, queries0, queries1,
            ),
        }
    }

    fn attention_dot_two_keys_four_queries(
        self,
        queries: [&[f32]; 4],
        keys: [&[u16]; 2],
    ) -> Result<[[f32; 4]; 2]> {
        match self {
            Self::Source => Ok(crate::kernels::f16_dot_two_keys_four_queries(queries, keys)),
            Self::PinnedSwift => {
                crate::kernels::pinned_attention::dot_f16_two_keys_four_queries(keys, queries)
            }
        }
    }

    fn attention_dot_two_keys_two_token_four_queries(
        self,
        queries0: [&[f32]; 4],
        queries1: [&[f32]; 4],
        keys: [&[u16]; 2],
    ) -> Result<(
        crate::kernels::pinned_attention::TwoKeyFourQueryScores,
        crate::kernels::pinned_attention::TwoKeyFourQueryScores,
    )> {
        match self {
            Self::Source => Ok(crate::kernels::f16_dot_two_keys_two_token_four_queries(
                queries0, queries1, keys,
            )),
            Self::PinnedSwift => {
                crate::kernels::pinned_attention::dot_f16_two_keys_two_token_four_queries(
                    keys, queries0, queries1,
                )
            }
        }
    }

    fn attention_dot_four_keys_four_queries(
        self,
        queries: [&[f32]; 4],
        keys: [&[u16]; 4],
    ) -> Result<[[f32; 4]; 4]> {
        match self {
            Self::Source => Ok(crate::kernels::f16_dot_four_keys_four_queries(
                queries, keys,
            )),
            Self::PinnedSwift => {
                crate::kernels::pinned_attention::dot_f16_four_keys_four_queries(keys, queries)
            }
        }
    }

    fn attention_dot_four_keys_two_token_four_queries(
        self,
        queries0: [&[f32]; 4],
        queries1: [&[f32]; 4],
        keys: [&[u16]; 4],
    ) -> Result<(
        crate::kernels::pinned_attention::FourKeyFourQueryScores,
        crate::kernels::pinned_attention::FourKeyFourQueryScores,
    )> {
        match self {
            Self::Source => Ok(crate::kernels::f16_dot_four_keys_two_token_four_queries(
                queries0, queries1, keys,
            )),
            Self::PinnedSwift => {
                crate::kernels::pinned_attention::dot_f16_four_keys_two_token_four_queries(
                    keys, queries0, queries1,
                )
            }
        }
    }

    fn attention_current_dot_four(self, queries: [&[f32]; 4], key: &[f32]) -> Result<[f32; 4]> {
        match self {
            Self::Source => Ok(crate::kernels::float_dot_four(queries, key)),
            Self::PinnedSwift => crate::kernels::pinned_attention::dot_f32_four(key, queries),
        }
    }

    fn attention_softmax_round_f16(self, scores: &mut [f32]) -> Result<()> {
        match self {
            Self::Source => softmax_round_f16(scores),
            Self::PinnedSwift => crate::kernels::pinned_attention::softmax_round_f16(scores),
        }
    }

    #[cfg(test)]
    fn attention_rope(
        self,
        input: &mut [f32],
        geometry: &Geometry,
        frequencies: &[(f32, f32)],
    ) -> Result<()> {
        match self {
            Self::Source => rope(input, geometry, frequencies),
            Self::PinnedSwift => crate::kernels::pinned_attention::rope_in_place(
                input,
                geometry.head,
                geometry.rope,
                frequencies,
            ),
        }
    }

    #[cfg(test)]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::as_conversions,
        reason = "Preserve the generic graph's binary64 accumulation and final binary32 rounding."
    )]
    fn attention_values(
        self,
        values: &KvCache,
        current: f32,
        dimension: usize,
        probabilities: &[f32],
    ) -> Result<f32> {
        match self {
            Self::Source => {
                let past_sum = values
                    .tokens()
                    .zip(probabilities.iter())
                    .map(|(past, &probability)| {
                        f64::from(probability * half_to_float(past[dimension]))
                    })
                    .sum::<f64>();
                Ok((past_sum + f64::from(probabilities[probabilities.len() - 1] * current)) as f32)
            }
            Self::PinnedSwift => crate::kernels::pinned_attention::dot(
                values
                    .tokens()
                    .map(|past| half_to_float(past[dimension]))
                    .chain(std::iter::once(current)),
                probabilities,
            ),
        }
    }

    fn attention_value_tile(
        self,
        values: &KvCache,
        current: &[f32],
        offset: usize,
        probabilities: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        let last = probabilities.len().saturating_sub(1);
        match self {
            Self::Source => {
                crate::kernels::source_value_tile(
                    values.tokens().take(last),
                    current,
                    offset,
                    probabilities,
                    output,
                );
                Ok(())
            }
            Self::PinnedSwift => crate::kernels::pinned_attention::value_tile(
                values.tokens().take(last),
                current,
                probabilities,
                offset,
                output,
            ),
        }
    }

    fn attention_value_tile_four(
        self,
        values: &KvCache,
        current: &[f32],
        offset: usize,
        probabilities: [&[f32]; 4],
        outputs: [&mut [f32]; 4],
    ) -> Result<()> {
        let last = probabilities[0].len().saturating_sub(1);
        match self {
            Self::Source => {
                crate::kernels::source_value_tile_four(
                    values.tokens().take(last),
                    current,
                    offset,
                    probabilities,
                    outputs,
                );
                Ok(())
            }
            Self::PinnedSwift => crate::kernels::pinned_attention::value_tile_four(
                values.tokens().take(last),
                current,
                probabilities,
                offset,
                outputs,
            ),
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Preserve each channel's binary64 sum and final binary32 rounding across two tokens."
    )]
    fn attention_value_tile_four_two_tokens(
        self,
        values: &KvCache,
        currents: [&[f32]; 2],
        offset: usize,
        probabilities0: [&[f32]; 4],
        probabilities1: [&[f32]; 4],
        outputs0: [&mut [f32]; 4],
        outputs1: [&mut [f32]; 4],
    ) -> Result<()> {
        let last1 = probabilities1[0].len().saturating_sub(1);
        match self {
            Self::Source => {
                crate::kernels::source_value_tile_four_two_tokens(
                    values.tokens().take(last1),
                    currents,
                    offset,
                    probabilities0,
                    probabilities1,
                    outputs0,
                    outputs1,
                );
                Ok(())
            }
            Self::PinnedSwift => crate::kernels::pinned_attention::value_tile_four_two_tokens(
                values.tokens().take(last1),
                currents,
                probabilities0,
                probabilities1,
                offset,
                outputs0,
                outputs1,
            ),
        }
    }

    #[cfg(test)]
    fn gdn_add_scaled(self, target: &mut [f32], input: &[f32], scale: f32) -> Result<()> {
        match self {
            Self::Source => {
                add_scaled_slice_in_place(target, input, scale);
                Ok(())
            }
            Self::PinnedSwift => crate::kernels::pinned_gdn::add_scaled(target, input, scale),
        }
    }

    fn gdn_update_and_dot(
        self,
        target: &mut [f32],
        key: &[f32],
        scale: f32,
        query: &[f32],
    ) -> Result<f32> {
        match self {
            Self::Source => {
                add_scaled_slice_in_place(target, key, scale);
                finite(target, "GDN state")?;
                Ok(dot(target, query))
            }
            Self::PinnedSwift => {
                crate::kernels::pinned_gdn::update_and_dot(target, key, scale, query)
            }
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Keep immutable previous state, mutable state output and scalar GDN step parameters explicit."
    )]
    fn gdn_step_row_from(
        self,
        previous: &[f32],
        state: &mut [f32],
        key: &[f32],
        query: &[f32],
        decay: f32,
        value: f32,
        beta: f32,
    ) -> Result<f32> {
        match self {
            Self::Source => {
                let dot_k = self.gdn_decay_and_dot_from(previous, state, key, decay)?;
                let delta = (value - dot_k) * beta;
                self.gdn_update_and_dot(state, key, delta, query)
            }
            Self::PinnedSwift => crate::kernels::pinned_gdn::step_row_from(
                previous, state, key, query, decay, value, beta,
            ),
        }
    }

    fn gdn_step_row_in_place(
        self,
        state: &mut [f32],
        key: &[f32],
        query: &[f32],
        decay: f32,
        value: f32,
        beta: f32,
    ) -> Result<f32> {
        match self {
            Self::Source => {
                let dot_k = self.gdn_decay_and_dot(state, key, decay)?;
                let delta = (value - dot_k) * beta;
                self.gdn_update_and_dot(state, key, delta, query)
            }
            Self::PinnedSwift => {
                crate::kernels::pinned_gdn::step_row_in_place(state, key, query, decay, value, beta)
            }
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Keep multi-token GDN row recurrence parameters and exact single-token rounding order explicit."
    )]
    fn gdn_step_row_tokens_from(
        self,
        previous: &[f32],
        state: &mut [f32],
        inputs: &[crate::kernels::pinned_gdn::ChunkInput<'_>],
        row: usize,
        stride: usize,
        scale: f32,
        output: &mut [f32],
    ) -> Result<()> {
        match self {
            Self::Source => {
                let Some(first) = inputs.first() else {
                    return Err(invalid("empty GDN chunk"));
                };
                output[row] = self.gdn_step_row_from(
                    previous,
                    state,
                    first.key,
                    first.query,
                    first.decay,
                    first.value[row],
                    first.beta,
                )? * scale;
                for (token, input) in inputs.iter().enumerate().skip(1) {
                    output[token * stride + row] = self.gdn_step_row_in_place(
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
            Self::PinnedSwift => crate::kernels::pinned_gdn::step_row_tokens_from(
                previous, state, inputs, row, stride, scale, output,
            ),
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Keep 4-row GDN step parameters explicit."
    )]
    fn gdn_step_four_rows_from(
        self,
        previous: [&[f32]; 4],
        states: [&mut [f32]; 4],
        key: &[f32],
        query: &[f32],
        decay: f32,
        values: [f32; 4],
        beta: f32,
    ) -> Result<[f32; 4]> {
        match self {
            Self::Source => {
                let [s0, s1, s2, s3] = states;
                Ok([
                    self.gdn_step_row_from(previous[0], s0, key, query, decay, values[0], beta)?,
                    self.gdn_step_row_from(previous[1], s1, key, query, decay, values[1], beta)?,
                    self.gdn_step_row_from(previous[2], s2, key, query, decay, values[2], beta)?,
                    self.gdn_step_row_from(previous[3], s3, key, query, decay, values[3], beta)?,
                ])
            }
            Self::PinnedSwift => crate::kernels::pinned_gdn::step_four_rows_from(
                previous, states, key, query, decay, values, beta,
            ),
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Keep 4-row multi-token GDN recurrence parameters explicit."
    )]
    fn gdn_step_four_rows_tokens_from(
        self,
        previous: [&[f32]; 4],
        states: [&mut [f32]; 4],
        inputs: &[crate::kernels::pinned_gdn::ChunkInput<'_>],
        row: usize,
        stride: usize,
        scale: f32,
        output: &mut [f32],
    ) -> Result<()> {
        match self {
            Self::Source => {
                let [s0, s1, s2, s3] = states;
                self.gdn_step_row_tokens_from(previous[0], s0, inputs, row, stride, scale, output)?;
                self.gdn_step_row_tokens_from(
                    previous[1],
                    s1,
                    inputs,
                    row + 1,
                    stride,
                    scale,
                    output,
                )?;
                self.gdn_step_row_tokens_from(
                    previous[2],
                    s2,
                    inputs,
                    row + 2,
                    stride,
                    scale,
                    output,
                )?;
                self.gdn_step_row_tokens_from(
                    previous[3],
                    s3,
                    inputs,
                    row + 3,
                    stride,
                    scale,
                    output,
                )
            }
            Self::PinnedSwift => crate::kernels::pinned_gdn::step_four_rows_tokens_from(
                previous, states, inputs, row, stride, scale, output,
            ),
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Keep 8-row GDN step parameters explicit."
    )]
    fn gdn_step_eight_rows_from(
        self,
        previous: [&[f32]; 8],
        states: [&mut [f32]; 8],
        key: &[f32],
        query: &[f32],
        decay: f32,
        values: [f32; 8],
        beta: f32,
    ) -> Result<[f32; 8]> {
        match self {
            Self::Source => {
                let [s0, s1, s2, s3, s4, s5, s6, s7] = states;
                Ok([
                    self.gdn_step_row_from(previous[0], s0, key, query, decay, values[0], beta)?,
                    self.gdn_step_row_from(previous[1], s1, key, query, decay, values[1], beta)?,
                    self.gdn_step_row_from(previous[2], s2, key, query, decay, values[2], beta)?,
                    self.gdn_step_row_from(previous[3], s3, key, query, decay, values[3], beta)?,
                    self.gdn_step_row_from(previous[4], s4, key, query, decay, values[4], beta)?,
                    self.gdn_step_row_from(previous[5], s5, key, query, decay, values[5], beta)?,
                    self.gdn_step_row_from(previous[6], s6, key, query, decay, values[6], beta)?,
                    self.gdn_step_row_from(previous[7], s7, key, query, decay, values[7], beta)?,
                ])
            }
            Self::PinnedSwift => crate::kernels::pinned_gdn::step_eight_rows_from(
                previous, states, key, query, decay, values, beta,
            ),
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Keep 8-row multi-token GDN recurrence parameters explicit."
    )]
    fn gdn_step_eight_rows_tokens_from(
        self,
        previous: [&[f32]; 8],
        states: [&mut [f32]; 8],
        inputs: &[crate::kernels::pinned_gdn::ChunkInput<'_>],
        row: usize,
        stride: usize,
        scale: f32,
        output: &mut [f32],
    ) -> Result<()> {
        match self {
            Self::Source => {
                for (r, state_row) in states.into_iter().enumerate() {
                    self.gdn_step_row_tokens_from(
                        previous[r],
                        state_row,
                        inputs,
                        row + r,
                        stride,
                        scale,
                        output,
                    )?;
                }
                Ok(())
            }
            Self::PinnedSwift => crate::kernels::pinned_gdn::step_eight_rows_tokens_from(
                previous, states, inputs, row, stride, scale, output,
            ),
        }
    }
}
const MAX_LAYERS: usize = 256;
const KV_PAGE_TOKENS: usize = 64;

/// Completed pages never mutate. Checkpoints share them and independently copy
/// only the bounded tail; appending and restoring never copy a sealed payload.
struct KvCache {
    pages: Vec<Arc<Vec<u16>>>,
    tail: Vec<u16>,
    width: usize,
    page_elements: usize,
    maximum: usize,
    len: usize,
}

impl KvCache {
    fn new(width: usize, context: usize) -> Result<Self> {
        let maximum = checked_product(&[width, context])?;
        if width == 0
            || context == 0
            || checked_product(&[maximum, size_of::<u16>()])? > MAX_STATE_BYTES
        {
            return Err(invalid("invalid paged KV cache bounds"));
        }
        Ok(Self {
            pages: Vec::new(),
            tail: Vec::new(),
            width,
            page_elements: checked_product(&[width, context.min(KV_PAGE_TOKENS)])?,
            maximum,
            len: 0,
        })
    }

    fn tokens(&self) -> impl Iterator<Item = &[u16]> {
        self.pages
            .iter()
            .flat_map(|page| page.chunks_exact(self.width))
            .chain(self.tail.chunks_exact(self.width))
    }

    // Reserve every fallible payload/table growth before any token commits.
    // Sealing moves the tail allocation into an Arc; there is no payload COW
    // or second page-sized allocation, even while snapshots retain old pages.
    // The standard Arc constructor still allocates its fixed-size header.
    fn reserve_append(&mut self) -> Result<()> {
        let next = checked_sum(self.len, self.width)?;
        if next > self.maximum {
            return Err(invalid("KV cache append exceeds context bound"));
        }
        let required = checked_sum(self.tail.len(), self.width)?;
        let remaining = self.maximum - (self.len - self.tail.len());
        let limit = self.page_elements.min(remaining);
        if required > self.tail.capacity() {
            let target = self
                .tail
                .capacity()
                .saturating_mul(2)
                .min(limit)
                .max(required);
            self.tail.try_reserve_exact(target - self.tail.len())?;
        }
        if required == self.page_elements || next == self.maximum {
            self.pages.try_reserve_exact(1)?;
        }
        Ok(())
    }

    fn append(&mut self, values: &[f32]) {
        debug_assert_eq!(values.len(), self.width);
        append_half_bits(values, &mut self.tail);
        self.len += self.width;
        if self.tail.len() == self.page_elements || self.len == self.maximum {
            self.pages.push(Arc::new(std::mem::take(&mut self.tail)));
        }
    }

    fn copy(&self) -> Result<Self> {
        let mut pages = capacity(self.pages.len())?;
        pages.extend(self.pages.iter().map(Arc::clone));
        Ok(Self {
            pages,
            tail: copy_values(&self.tail)?,
            width: self.width,
            page_elements: self.page_elements,
            maximum: self.maximum,
            len: self.len,
        })
    }

    fn clear(&mut self) {
        self.pages.clear();
        self.tail.clear();
        self.len = 0;
    }

    fn bytes(&self) -> usize {
        self.pages.capacity() * size_of::<Arc<Vec<u16>>>()
            + self.tail.capacity() * size_of::<u16>()
            + self
                .pages
                .iter()
                .map(|page| {
                    page.capacity() * size_of::<u16>()
                        + size_of::<Vec<u16>>()
                        + 2 * size_of::<usize>()
                })
                .sum::<usize>()
    }
}

#[derive(Clone)]
struct Geometry {
    embedding: usize,
    feed_forward: usize,
    heads: usize,
    kv_heads: usize,
    head: usize,
    rope: usize,
    rope_sections: [usize; 4],
    rope_base: f32,
    epsilon: f32,
    ssm_head: usize,
    ssm_keys: usize,
    ssm_values: usize,
    convolution: usize,
}

impl Geometry {
    const fn key_width(&self) -> usize {
        self.ssm_head * self.ssm_keys
    }

    const fn value_width(&self) -> usize {
        self.ssm_head * self.ssm_values
    }

    const fn convolution_width(&self) -> usize {
        2 * self.key_width() + self.value_width()
    }

    const fn query_gate_width(&self) -> usize {
        2 * self.head * self.heads
    }

    const fn kv_width(&self) -> usize {
        self.head * self.kv_heads
    }
}

struct FullAttention {
    query: Tensor,
    key: Tensor,
    value: Tensor,
    output: Tensor,
    query_norm: Vec<f32>,
    key_norm: Vec<f32>,
}

struct RecurrentAttention {
    qkv: Tensor,
    gate: Tensor,
    beta: Tensor,
    alpha: Tensor,
    convolution: Vec<f32>,
    time_bias: Vec<f32>,
    decay: Vec<f32>,
    norm: Vec<f32>,
    output: Tensor,
}

enum Attention {
    Full(FullAttention),
    Recurrent(RecurrentAttention),
}

struct Layer {
    norm: Vec<f32>,
    post_norm: Vec<f32>,
    attention: Attention,
    up: Tensor,
    gate: Tensor,
    down: Tensor,
}

enum LayerState {
    Full {
        keys: KvCache,
        values: KvCache,
    },
    Recurrent {
        convolution: Vec<f32>,
        matrix: Vec<f32>,
    },
}

impl LayerState {
    fn bytes(&self) -> usize {
        match self {
            Self::Full { keys, values } => keys.bytes() + values.bytes(),
            Self::Recurrent {
                convolution,
                matrix,
            } => (convolution.capacity() + matrix.capacity()) * size_of::<f32>(),
        }
    }
}

enum LayerDelta {
    Full {
        // Keep the current token in reusable binary32 scratch while the graph
        // runs; only a successful evaluation appends its rounded binary16 bits.
        key: Vec<f32>,
        value: Vec<f32>,
    },
    Recurrent {
        convolution: Vec<f32>,
        matrix: Vec<f32>,
    },
}

#[derive(Default)]
struct ProjectionScratch {
    transformed: Vec<f32>,
    pre_transformed_len: Option<usize>,
    matvec: MatvecScratch,
    #[cfg(test)]
    transforms: usize,
}

#[derive(Default)]
struct AttentionScratch {
    // Attention projections finish before the FFN starts. Reuse these two
    // vectors for joint Q/gate or recurrent QKV, then FFN up/gate. Recurrent
    // convolution also replaces each QKV element only after its last raw use.
    projected: Vec<f32>,
    query: Vec<f32>,
    gate: Vec<f32>,
    attended: Vec<f32>,
    scores: Vec<f32>,
    frequencies: Vec<(f32, f32)>,
    beta: Vec<f32>,
    alpha: Vec<f32>,
}

#[derive(Default)]
struct Workspace {
    hidden: Vec<f32>,
    normalized: Vec<f32>,
    attention: Vec<f32>,
    projection: ProjectionScratch,
    adapter: AdapterScratch,
    attention_scratch: AttentionScratch,
    deltas: Vec<LayerDelta>,
}

struct Evaluation<'a, 'executor> {
    geometry: &'a Geometry,
    arithmetic: Arithmetic,
    transforms: &'a Transforms,
    position: usize,
    frequencies_ready: bool,
    projection: &'a mut ProjectionScratch,
    adapter: Option<&'a RankOneAdapter>,
    adapter_scale: f32,
    adapter_scratch: &'a mut AdapterScratch,
    executor: Option<&'executor mut dyn crate::kernels::Pq2ProjectionExecutor>,
}

#[derive(Default)]
struct Transforms {
    block: usize,
    weights: BTreeSet<String>,
    inverse_embedding: bool,
    grouped_values: bool,
    signs: BTreeMap<usize, Vec<f32>>,
}

impl Transforms {
    #[expect(
        clippy::too_many_lines,
        reason = "Keep validation and graph operation order auditable together."
    )]
    fn read(model: &Model, geometry: &Geometry) -> Result<Self> {
        let metadata = &model.metadata;
        let Some(version) = metadata.get("prism.hadamard.version") else {
            if metadata
                .keys()
                .any(|key| key.starts_with("prism.hadamard."))
            {
                return Err(invalid("Hadamard metadata requires prism.hadamard.version"));
            }
            return Ok(Self::default());
        };
        let version = version
            .as_u64()
            .ok_or_else(|| invalid("invalid Hadamard version"))?;
        let tied = optional_bool(metadata, "prism.hadamard.tied_output", false)?;
        if !matches!(version, 1 | 2) || (version == 2) != tied {
            return Err(invalid(
                "Hadamard version 2 requires tied_output; version 1 forbids it",
            ));
        }
        if tied && model.tensors.contains_key("output.weight") {
            return Err(invalid(
                "Hadamard tied output requires output.weight to be absent",
            ));
        }
        let block = integer(metadata, "prism.hadamard.block_size")?;
        if !block.is_power_of_two() || block > 65_536 {
            return Err(invalid(
                "Hadamard block size must be a bounded power of two",
            ));
        }
        if string(metadata, "prism.hadamard.transform")? != "normalized-sylvester-walsh-hadamard"
            || string(metadata, "prism.hadamard.axis")? != "input-last-dimension"
        {
            return Err(invalid("unsupported Hadamard transform or axis"));
        }
        let mut weights = BTreeSet::new();
        for name in array(metadata, "prism.hadamard.weight_names")?.iter() {
            let name = name
                .as_str()
                .ok_or_else(|| invalid("Hadamard weight names must be strings"))?;
            if !foldable(name) || !weights.insert(name.to_owned()) {
                return Err(invalid(format!(
                    "unsupported or duplicate Hadamard weight: {name}"
                )));
            }
        }
        if weights.is_empty() {
            return Err(invalid("Hadamard weight_names must not be empty"));
        }
        let mut inverse_embedding = false;
        if metadata.contains_key("prism.hadamard.inverse_weight_names") {
            for name in array(metadata, "prism.hadamard.inverse_weight_names")?.iter() {
                if name.as_str() != Some("token_embd.weight") || inverse_embedding {
                    return Err(invalid(
                        "only one token embedding inverse transform is supported",
                    ));
                }
                inverse_embedding = true;
            }
        }
        if tied {
            if !inverse_embedding {
                return Err(invalid(
                    "Hadamard tied output requires inverse token embedding transform",
                ));
            }
            weights.insert("token_embd.weight".to_owned());
        } else if inverse_embedding && !model.tensors.contains_key("output.weight") {
            return Err(invalid("latent tied output requires Hadamard version 2"));
        }
        let mut signs = BTreeMap::new();
        match string(metadata, "prism.hadamard.sign_mode")? {
            "identity" => {}
            "explicit" => {
                let widths = array(metadata, "prism.hadamard.sign_widths")?;
                let values = array(metadata, "prism.hadamard.sign_values")?;
                if widths.is_empty() {
                    return Err(invalid("explicit Hadamard sign widths must not be empty"));
                }
                let mut offset = 0_usize;
                for width in widths.iter() {
                    let width = metadata_usize(width, "Hadamard sign width")?;
                    let end = offset
                        .checked_add(width)
                        .ok_or_else(|| invalid("sign length overflow"))?;
                    if width == 0
                        || width > MAX_WIDTH
                        || !width.is_multiple_of(block)
                        || end > values.len()
                    {
                        return Err(invalid("invalid Hadamard sign width"));
                    }
                    let vector = (offset..end)
                        .map(
                            |index| match values.get(index).and_then(MetadataValueRef::as_i64) {
                                Some(-1) => Ok(-1.0),
                                Some(1) => Ok(1.0),
                                _ => Err(invalid("Hadamard signs must be +1 or -1")),
                            },
                        )
                        .collect::<Result<Vec<_>>>()?;
                    if signs.insert(width, vector).is_some() {
                        return Err(invalid("duplicate Hadamard sign width"));
                    }
                    offset = end;
                }
                if offset != values.len() {
                    return Err(invalid("Hadamard sign_values length mismatch"));
                }
            }
            _ => return Err(invalid("unsupported Hadamard sign mode")),
        }
        for name in weights
            .iter()
            .map(String::as_str)
            .chain(inverse_embedding.then_some("token_embd.weight"))
        {
            let tensor = model
                .tensors
                .get(name)
                .ok_or_else(|| invalid(format!("Hadamard tensor absent: {name}")))?;
            let width = tensor
                .dimensions
                .first()
                .copied()
                .ok_or_else(|| invalid("Hadamard tensor has no dimensions"))?;
            if !width.is_multiple_of(block) || (!signs.is_empty() && !signs.contains_key(&width)) {
                return Err(invalid(format!(
                    "Hadamard width or signs do not match {name}"
                )));
            }
        }
        let grouped_values = optional_bool(metadata, "prism.hadamard.gdn_v_grouped", false)?;
        if grouped_values && !geometry.ssm_values.is_multiple_of(geometry.ssm_keys) {
            return Err(invalid(
                "grouped GDN transform has incompatible head geometry",
            ));
        }
        Ok(Self {
            block,
            weights,
            inverse_embedding,
            grouped_values,
            signs,
        })
    }

    #[cfg(test)]
    fn project_into(
        &self,
        weight: &Tensor,
        input: &[f32],
        geometry: &Geometry,
        output: &mut Vec<f32>,
        scratch: &mut ProjectionScratch,
    ) -> Result<()> {
        self.project_group_into([weight], input, geometry, [output], scratch)
    }

    fn basis(&self, weight: &Tensor) -> (bool, bool) {
        let rotated = self.weights.contains(&weight.name);
        (
            rotated,
            rotated && self.grouped_values && weight.name.ends_with(".ssm_out.weight"),
        )
    }

    fn can_block_hadamard(&self, len: usize) -> bool {
        self.block >= 16
            && self.block.is_power_of_two()
            && len.is_multiple_of(self.block)
            && self.signs.get(&len).is_none_or(|signs| signs.len() >= len)
    }

    #[expect(
        clippy::option_option,
        reason = "Outer Option indicates whether the FFN chain is eligible; inner Option<u32> carries the optional down-projection Hadamard block size."
    )]
    pub(super) fn ffn_chain_hadamard_block_for(&self, down_weight: &Tensor) -> Option<Option<u32>> {
        let (rotated, grouped) = self.basis(down_weight);
        if !rotated {
            return Some(None);
        }
        if grouped {
            return None;
        }
        let len = down_weight.dimensions.first().copied()?;
        if !self.can_block_hadamard(len) || self.block < 32 {
            return None;
        }
        if self.signs.get(&len).is_some_and(|signs| {
            signs[..len]
                .iter()
                .any(|&s| s.to_bits() != 1.0_f32.to_bits())
        }) {
            return None;
        }
        u32::try_from(self.block).ok().map(Some)
    }

    // The fixed-size group borrows one exact input slice for its whole scope.
    // Nothing is reused across calls, tokens, layers or failed evaluations.
    #[cfg(test)]
    fn project_group_into<const N: usize>(
        &self,
        weights: [&Tensor; N],
        input: &[f32],
        geometry: &Geometry,
        outputs: [&mut Vec<f32>; N],
        scratch: &mut ProjectionScratch,
    ) -> Result<()> {
        self.project_group_into_with_executor(weights, input, geometry, outputs, scratch, &mut None)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Keep scoped execution separate from CPU scratch and exact transform ownership."
    )]
    fn project_group_into_with_executor<const N: usize>(
        &self,
        weights: [&Tensor; N],
        input: &[f32],
        geometry: &Geometry,
        mut outputs: [&mut Vec<f32>; N],
        scratch: &mut ProjectionScratch,
        executor: &mut Option<&mut dyn crate::kernels::Pq2ProjectionExecutor>,
    ) -> Result<()> {
        let mut pre_transformed = scratch.pre_transformed_len.take();
        let result = (|| {
            if N > 4 {
                return Err(invalid("projection group exceeds four siblings"));
            }
            let mut completed = [false; N];
            for index in 0..N {
                if completed[index] {
                    continue;
                }
                let basis = self.basis(weights[index]);
                let activation = if basis.0 {
                    if basis.1
                        || pre_transformed.take() != Some(input.len())
                        || scratch.transformed.len() != input.len()
                    {
                        self.transform_into_with_parallelism(
                            weights[index],
                            input,
                            geometry,
                            scratch.matvec.row_parallelism(),
                            &mut scratch.transformed,
                        )?;
                    }
                    #[cfg(test)]
                    {
                        scratch.transforms += 1;
                    }
                    scratch.transformed.as_slice()
                } else {
                    input
                };
                let skip_cpu_quantization = executor.as_deref().is_some_and(|exec| {
                    weights
                        .iter()
                        .enumerate()
                        .skip(index)
                        .all(|(candidate, weight)| {
                            completed[candidate]
                                || self.basis(weight) != basis
                                || !scratch
                                    .matvec
                                    .shares_contract(weights[index], weight)
                                    .unwrap_or(false)
                                || (weight.kind == crate::kernels::PQ2
                                    && weight
                                        .dimensions
                                        .first()
                                        .is_some_and(|&cols| cols.is_multiple_of(128))
                                    && exec.supports_f32(weight))
                        })
                });
                let mut prepared = if skip_cpu_quantization {
                    scratch
                        .matvec
                        .prepare_f32_executor(weights[index], activation)?
                } else {
                    scratch.matvec.prepare(weights[index], activation)?
                };
                let group_len = weights[index..]
                    .iter()
                    .enumerate()
                    .take_while(|&(offset, w)| {
                        !completed[index + offset]
                            && self.basis(w) == basis
                            && prepared.is_compatible(w).unwrap_or(false)
                    })
                    .count();
                if group_len >= 2
                    && skip_cpu_quantization
                    && let Some(executor) = executor.as_deref_mut()
                    && executor.supports_f32_group(&weights[index..index + group_len])
                {
                    prepared.matvec_group_with_executor(
                        &weights[index..index + group_len],
                        &mut outputs[index..index + group_len],
                        executor,
                    )?;
                    completed[index..index + group_len].fill(true);
                    continue;
                }
                for (candidate, (weight, output)) in
                    weights.iter().zip(outputs.iter_mut()).enumerate()
                {
                    if !completed[candidate]
                        && self.basis(weight) == basis
                        && prepared.is_compatible(weight)?
                    {
                        if let Some(executor) = executor.as_deref_mut() {
                            prepared.matvec_with_executor(weight, output, executor)?;
                        } else {
                            prepared.matvec_into(weight, output)?;
                        }
                        completed[candidate] = true;
                    }
                }
            }
            Ok(())
        })();
        if result.is_err() {
            for output in &mut outputs {
                output.clear();
            }
        }
        result
    }

    fn transform_into(
        &self,
        weight: &Tensor,
        input: &[f32],
        geometry: &Geometry,
        transformed: &mut Vec<f32>,
    ) -> Result<()> {
        self.transform_into_with_parallelism(
            weight,
            input,
            geometry,
            &RowParallelism::default(),
            transformed,
        )
    }

    pub(super) fn can_grouped_block_hadamard(&self, weight: &Tensor, geometry: &Geometry) -> bool {
        let len = geometry.value_width();
        self.weights.contains(&weight.name)
            && self.grouped_values
            && weight.name.ends_with(".ssm_out.weight")
            && self.block >= 64
            && self.block.is_power_of_two()
            && geometry.ssm_head > 0
            && self.block.is_multiple_of(geometry.ssm_head)
            && len > 0
            && len.is_multiple_of(self.block)
            && self.signs.get(&len).is_none_or(|signs| signs.len() >= len)
    }

    pub(super) fn transform_grouped_strided_into_with_parallelism(
        &self,
        input: &[f32],
        head_stride: usize,
        geometry: &Geometry,
        parallelism: &RowParallelism,
        transformed: &mut Vec<f32>,
    ) -> Result<()> {
        let len = geometry.value_width();
        resize_into(len, transformed)?;
        let repetitions = geometry.ssm_values / geometry.ssm_keys;
        let valid_signs = self
            .signs
            .get(&len)
            .map(Vec::as_slice)
            .filter(|signs| signs.len() >= len);
        let heads_per_block = self.block / geometry.ssm_head;
        if parallelism.workers() > 1 && self.block >= 256 && len >= self.block.saturating_mul(2) {
            return parallelism.update_chunks(transformed, self.block, |base_block, chunk| {
                for (local_idx, out_block) in chunk.chunks_exact_mut(self.block).enumerate() {
                    crate::kernels::hadamard_grouped_strided_block_into(
                        input,
                        head_stride,
                        valid_signs,
                        base_block + local_idx,
                        heads_per_block,
                        repetitions,
                        geometry.ssm_keys,
                        geometry.ssm_head,
                        out_block,
                    )?;
                }
                Ok(())
            });
        }
        for (block_idx, out_block) in transformed.chunks_exact_mut(self.block).enumerate() {
            crate::kernels::hadamard_grouped_strided_block_into(
                input,
                head_stride,
                valid_signs,
                block_idx,
                heads_per_block,
                repetitions,
                geometry.ssm_keys,
                geometry.ssm_head,
                out_block,
            )?;
        }
        Ok(())
    }

    fn transform_into_with_parallelism(
        &self,
        weight: &Tensor,
        input: &[f32],
        geometry: &Geometry,
        parallelism: &RowParallelism,
        transformed: &mut Vec<f32>,
    ) -> Result<()> {
        if self.grouped_values && weight.name.ends_with(".ssm_out.weight") {
            if self.can_grouped_block_hadamard(weight, geometry)
                && input.len() == geometry.value_width()
            {
                return self.transform_grouped_strided_into_with_parallelism(
                    input,
                    geometry.ssm_head,
                    geometry,
                    parallelism,
                    transformed,
                );
            }
            resize_into(input.len(), transformed)?;
            let repetitions = geometry.ssm_values / geometry.ssm_keys;
            let valid_signs = self
                .signs
                .get(&input.len())
                .map(Vec::as_slice)
                .filter(|signs| signs.len() >= input.len());
            if let Some(signs) = valid_signs {
                for key in 0..geometry.ssm_keys {
                    for repeat in 0..repetitions {
                        let source = (repeat * geometry.ssm_keys + key) * geometry.ssm_head;
                        let target = (key * repetitions + repeat) * geometry.ssm_head;
                        mul_slice_into(
                            &input[source..source + geometry.ssm_head],
                            &signs[target..target + geometry.ssm_head],
                            &mut transformed[target..target + geometry.ssm_head],
                        );
                    }
                }
            } else {
                for key in 0..geometry.ssm_keys {
                    for repeat in 0..repetitions {
                        let source = (repeat * geometry.ssm_keys + key) * geometry.ssm_head;
                        let target = (key * repetitions + repeat) * geometry.ssm_head;
                        transformed[target..target + geometry.ssm_head]
                            .copy_from_slice(&input[source..source + geometry.ssm_head]);
                    }
                }
                self.sign(transformed);
            }
        } else if self.block > 0
            && input.len().is_multiple_of(self.block)
            && self
                .signs
                .get(&input.len())
                .is_none_or(|signs| signs.len() >= input.len())
        {
            resize_into(input.len(), transformed)?;
            let signs = self.signs.get(&input.len()).map(Vec::as_slice);
            if parallelism.workers() > 1
                && self.block >= 256
                && input.len() >= self.block.saturating_mul(16)
            {
                return parallelism.update_chunks(transformed, self.block, |base_block, chunk| {
                    for (local_idx, out_block) in chunk.chunks_exact_mut(self.block).enumerate() {
                        let offset = (base_block + local_idx) * self.block;
                        let in_block = &input[offset..offset + self.block];
                        let sign_block = signs.map(|s| &s[offset..offset + self.block]);
                        crate::kernels::hadamard_into(in_block, sign_block, out_block)?;
                    }
                    Ok(())
                });
            }
            for (block_idx, out_block) in transformed.chunks_exact_mut(self.block).enumerate() {
                let offset = block_idx * self.block;
                let in_block = &input[offset..offset + self.block];
                let sign_block = signs.map(|s| &s[offset..offset + self.block]);
                crate::kernels::hadamard_into(in_block, sign_block, out_block)?;
            }
            return Ok(());
        } else if let Some(signs) = self.signs.get(&input.len()) {
            resize_into(input.len(), transformed)?;
            let overlap = input.len().min(signs.len());
            mul_slice_into(
                &input[..overlap],
                &signs[..overlap],
                &mut transformed[..overlap],
            );
            if overlap < input.len() {
                transformed[overlap..].copy_from_slice(&input[overlap..]);
            }
        } else {
            copy_into(input, transformed)?;
        }
        self.rotate_with_parallelism(transformed, parallelism)?;
        Ok(())
    }

    fn sign(&self, input: &mut [f32]) {
        if let Some(signs) = self.signs.get(&input.len()) {
            let overlap = input.len().min(signs.len());
            mul_slice_in_place(&mut input[..overlap], &signs[..overlap]);
        }
    }

    fn rotate(&self, input: &mut [f32]) -> Result<()> {
        for block in input.chunks_exact_mut(self.block) {
            crate::kernels::hadamard(block)?;
        }
        Ok(())
    }

    fn rotate_with_parallelism(
        &self,
        input: &mut [f32],
        parallelism: &RowParallelism,
    ) -> Result<()> {
        if parallelism.workers() > 1
            && self.block >= 256
            && input.len() >= self.block.saturating_mul(16)
            && input.len().is_multiple_of(self.block)
        {
            return parallelism.update_chunks(input, self.block, |_, chunk| self.rotate(chunk));
        }
        self.rotate(input)
    }
}

/// Exact conversation state with shared immutable KV pages and an owned tail.
///
/// Convolution and recurrent payloads are copied exactly, without F16 rounding.
/// A zero-position checkpoint retains only provenance and metadata: construction
/// and reset establish empty KV and positive-zero recurrent/convolution state.
pub struct Qwen35Checkpoint {
    provenance: Arc<()>,
    identity: String,
    context_limit: usize,
    position: usize,
    layers: Vec<LayerState>,
}

/// One logical state component observed for full-model numerical qualification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StateComponent {
    /// One cached key token, promoted exactly from stored binary16 bits.
    Key(usize),
    /// One cached value token, promoted exactly from stored binary16 bits.
    Value(usize),
    /// Channel-major convolution history, oldest value first per channel.
    Convolution,
    /// GDN state in value-head, value-row, contiguous-key-column order.
    Recurrent,
}

/// Native decoder for the supported Qwen3.5 dense text graph.
///
/// The exact pinned Swift artifact uses its compiled Windows ARM64 arithmetic
/// contract. Other model identities retain the scalar source-order reference.
pub struct Qwen35 {
    provenance: Arc<()>,
    identity: String,
    geometry: Geometry,
    arithmetic: Arithmetic,
    embedding: Tensor,
    output: Tensor,
    output_norm: Vec<f32>,
    transforms: Transforms,
    adapter: Option<RankOneAdapter>,
    adapter_scale: f32,
    layers: Vec<Layer>,
    state: Vec<LayerState>,
    workspace: Workspace,
    context_limit: usize,
    position: usize,
    vocabulary: usize,
    narrow_last_layer: bool,
    tiled_prefill: Option<prefill::TiledPrefill>,
    projection_executor: Option<Box<dyn crate::kernels::Pq2ProjectionExecutor + Send>>,
}

impl Qwen35 {
    /// Borrows each logical state component for numerical qualification.
    ///
    /// The callback receives the block index and component. Recurrent and
    /// convolution slices borrow committed state directly. Each cached token
    /// uses one reusable, fallibly allocated row to promote FP16 bits to F32.
    /// No observation can mutate state, and callback errors leave state intact.
    /// This supplies no state-import or portable checkpoint contract.
    ///
    /// # Errors
    /// Returns allocation or observer errors without modifying decoder state.
    pub fn visit_state(
        &self,
        observer: &mut impl FnMut(usize, StateComponent, &[f32]) -> Result<()>,
    ) -> Result<()> {
        let mut row = Vec::new();
        for (index, state) in self.state.iter().enumerate() {
            match state {
                LayerState::Full { keys, values } => {
                    for (cache, key) in [(keys, true), (values, false)] {
                        for (position, values) in cache.tokens().enumerate() {
                            row.clear();
                            row.try_reserve_exact(values.len())?;
                            row.extend(values.iter().copied().map(half_to_float));
                            let component = if key {
                                StateComponent::Key(position)
                            } else {
                                StateComponent::Value(position)
                            };
                            observer(index, component, &row)?;
                        }
                    }
                }
                LayerState::Recurrent {
                    convolution,
                    matrix,
                } => {
                    observer(index, StateComponent::Convolution, convolution)?;
                    observer(index, StateComponent::Recurrent, matrix)?;
                }
            }
        }
        Ok(())
    }

    /// Validates model contracts and cache bounds before allocating decoder state.
    ///
    /// Uses [`MatvecBackend::Auto`] with per-operation scalar fallback.
    ///
    /// # Errors
    /// Returns an error for malformed tensors, unsupported graph variants, or
    /// a context/state allocation beyond the native runtime's explicit bounds.
    pub fn new(model: Model, context_limit: usize) -> Result<Self> {
        Self::with_backend(model, context_limit, MatvecBackend::Auto)
    }

    /// Validates and creates a decoder with an explicit packed-matrix kernel request.
    ///
    /// Selection applies to every projection, including grouped siblings and the
    /// output head. Unsupported builds, targets and tensor formats fall back to
    /// scalar for each operation, matching [`MatvecScratch::with_backend`]. Other
    /// graph operations retain their reference arithmetic. The request does not
    /// establish whole-model SIMD parity or performance qualification.
    ///
    /// # Errors
    /// Returns the same model validation and allocation errors as [`Self::new`].
    pub fn with_backend(
        model: Model,
        context_limit: usize,
        backend: MatvecBackend,
    ) -> Result<Self> {
        Self::with_state_budget(model, context_limit, MAX_STATE_BYTES, backend)
    }

    /// Creates an explicit rank-one adapter graph for numerical qualification.
    ///
    /// Base projections retain their existing transforms; corrections receive
    /// the original activations. Embedding corrections are added after inverse
    /// transformation. Scale zero bypasses all correction arithmetic exactly.
    /// Adapter identity and scale bits are sealed into the graph identity.
    /// [`Self::reset_with_adapter_scale`] changes scale only after clearing state
    /// and invalidates every previously created checkpoint. Service/default
    /// qualification remains independent of this explicit graph support.
    ///
    /// # Errors
    /// Rejects a different base identity, unsupported/non-live targets, scales
    /// outside finite 0..2, an embedding adapter on a tied output head, and
    /// ordinary graph/storage validation failures.
    pub fn with_rank_one_adapter(
        model: Model,
        context_limit: usize,
        state_budget_bytes: usize,
        backend: MatvecBackend,
        adapter: RankOneAdapter,
        scale: f32,
    ) -> Result<Self> {
        RankOneAdapter::validate_scale(scale)?;
        if adapter.base_identity() != model.identity {
            return Err(invalid("rank-one adapter belongs to another base model"));
        }
        let mut graph = Self::with_state_budget(model, context_limit, state_budget_bytes, backend)?;
        for target in adapter.targets() {
            if target == graph.embedding.name && graph.output.name == graph.embedding.name {
                return Err(invalid(
                    "rank-one embedding adapter requires a separate output head",
                ));
            }
            if target != graph.embedding.name
                && !graph.layers.iter().any(|layer| {
                    target == layer.down.name
                        || target
                            == match &layer.attention {
                                Attention::Full(weights) => &weights.output.name,
                                Attention::Recurrent(weights) => &weights.output.name,
                            }
                })
            {
                return Err(invalid("rank-one adapter target is outside the live graph"));
            }
        }
        write!(
            graph.identity,
            ":rank-one-lora-v1:{}:{:08x}",
            adapter.identity(),
            scale.to_bits()
        )?;
        graph.adapter = Some(adapter);
        graph.adapter_scale = scale;
        Ok(graph)
    }

    /// Current rank-one correction scale, or `None` when no adapter is loaded.
    #[must_use]
    pub const fn adapter_scale(&self) -> Option<f32> {
        if self.adapter.is_some() {
            Some(self.adapter_scale)
        } else {
            None
        }
    }

    /// Clear all live state and establish a fresh rank-one scale identity.
    ///
    /// Immutable weights and factors remain shared. Every successful call renews
    /// checkpoint provenance, including the same scale and changes back to a
    /// previous scale, so old checkpoints can never restore across this boundary.
    /// Invalid scale or allocation failure leaves the previous graph unchanged.
    ///
    /// # Errors
    /// Rejects a graph without an adapter, scales outside finite 0..2, malformed
    /// graph identity and allocation failure while preparing the new identity.
    pub fn reset_with_adapter_scale(&mut self, scale: f32) -> Result<()> {
        RankOneAdapter::validate_scale(scale)?;
        let adapter = self
            .adapter
            .as_ref()
            .ok_or_else(|| invalid("adapter scale requires a loaded rank-one adapter"))?;
        let (base_identity, _) = self
            .identity
            .rsplit_once(":rank-one-lora-v1:")
            .ok_or_else(|| invalid("rank-one graph identity is missing its adapter contract"))?;
        let mut identity = String::new();
        identity.try_reserve_exact(base_identity.len() + adapter.identity().len() + 32)?;
        write!(
            identity,
            "{base_identity}:rank-one-lora-v1:{}:{:08x}",
            adapter.identity(),
            scale.to_bits()
        )?;
        self.append_tiled_prefill_identity(&mut identity)?;
        let provenance = Arc::new(());
        self.reset()?;
        self.identity = identity;
        self.adapter_scale = scale;
        self.provenance = provenance;
        Ok(())
    }

    /// Creates a decoder within an explicit maximum live-state payload budget.
    ///
    /// Admission charges full-context FP16 K/V plus F32 convolution/recurrent
    /// elements before allocating decoder state. KV allocation remains lazy;
    /// completed pages hold up to 64 tokens and never copy their payload when
    /// sealed or restored. Checkpoints share those immutable pages and copy the
    /// bounded tail plus exact F32 recurrent state. Snapshot admission charges
    /// each referenced page in full, conservatively overcounting shared pages.
    /// The live-state budget excludes model storage, graph scratch, checkpoints,
    /// page-table/Arc metadata and allocator overhead; it does not bound process
    /// memory. Kernel selection follows [`Self::with_backend`].
    ///
    /// # Errors
    /// Rejects zero budgets, budgets above [`MAX_STATE_BYTES`], a graph whose
    /// maximum live-state element payload exceeds the budget, and the same
    /// malformed model/allocation errors as [`Self::new`].
    #[expect(
        clippy::too_many_lines,
        reason = "Keep validation and graph operation order auditable together."
    )]
    pub fn with_state_budget(
        model: Model,
        context_limit: usize,
        state_budget_bytes: usize,
        backend: MatvecBackend,
    ) -> Result<Self> {
        if state_budget_bytes == 0 || state_budget_bytes > MAX_STATE_BYTES {
            return Err(invalid("invalid native live-state payload budget"));
        }
        let metadata = &model.metadata;
        if string(metadata, "general.architecture")? != "qwen35" {
            return Err(invalid(
                "the native graph supports general.architecture=qwen35 only",
            ));
        }
        reject_unsupported(metadata)?;
        let block_count = integer(metadata, "qwen35.block_count")?;
        let mtp = optional_integer(metadata, "qwen35.nextn_predict_layers", 0)?;
        if block_count == 0 || block_count > MAX_LAYERS || mtp >= block_count {
            return Err(invalid("invalid Qwen35 block or MTP count"));
        }
        let layer_count = block_count - mtp;
        let train_context = integer(metadata, "qwen35.context_length")?;
        if context_limit == 0 || context_limit > train_context || context_limit > 1_048_576 {
            return Err(invalid(
                "context limit must be positive and within the unscaled training context",
            ));
        }
        let mut geometry = Geometry {
            embedding: integer(metadata, "qwen35.embedding_length")?,
            feed_forward: integer(metadata, "qwen35.feed_forward_length")?,
            heads: integer(metadata, "qwen35.attention.head_count")?,
            kv_heads: integer(metadata, "qwen35.attention.head_count_kv")?,
            head: integer(metadata, "qwen35.attention.key_length")?,
            rope: integer(metadata, "qwen35.rope.dimension_count")?,
            rope_sections: [0; 4],
            rope_base: float(metadata, "qwen35.rope.freq_base")?,
            epsilon: float(metadata, "qwen35.attention.layer_norm_rms_epsilon")?,
            ssm_head: integer(metadata, "qwen35.ssm.state_size")?,
            ssm_keys: integer(metadata, "qwen35.ssm.group_count")?,
            ssm_values: integer(metadata, "qwen35.ssm.time_step_rank")?,
            convolution: integer(metadata, "qwen35.ssm.conv_kernel")?,
        };
        let sizes = [
            geometry.embedding,
            geometry.feed_forward,
            geometry.heads,
            geometry.kv_heads,
            geometry.head,
            geometry.rope,
            geometry.ssm_head,
            geometry.ssm_keys,
            geometry.ssm_values,
            geometry.convolution,
        ];
        if sizes.iter().any(|&value| value == 0 || value > 65_535)
            || !geometry.heads.is_multiple_of(geometry.kv_heads)
            || !geometry.ssm_values.is_multiple_of(geometry.ssm_keys)
            || geometry.rope > geometry.head
            || !geometry.rope.is_multiple_of(2)
            || geometry.rope_base <= 0.0
            || geometry.epsilon <= 0.0
            || integer(metadata, "qwen35.attention.value_length")? != geometry.head
            || integer(metadata, "qwen35.ssm.inner_size")? != geometry.value_width()
            || geometry.convolution_width() > MAX_WIDTH
            || geometry.convolution_width() * geometry.convolution > 16 * MAX_WIDTH
            || geometry.heads * geometry.head > MAX_WIDTH
        {
            return Err(invalid(
                "unsupported or invalid Qwen35 head, RoPE, normalization, or convolution geometry",
            ));
        }
        let sections = array(metadata, "qwen35.rope.dimension_sections")?;
        if sections.len() != 4
            || sections.iter().try_fold(0_usize, |sum, value| {
                checked_sum(sum, metadata_usize(value, "RoPE section")?)
            })? != geometry.rope / 2
        {
            return Err(invalid(
                "RoPE sections must describe exactly half the rotated dimensions",
            ));
        }
        for (destination, value) in geometry.rope_sections.iter_mut().zip(sections.iter()) {
            *destination = metadata_usize(value, "RoPE section")?;
        }
        let embedding = tensor(&model, "token_embd.weight", &[geometry.embedding, 0])?;
        let vocabulary = embedding.dimensions[1];
        if vocabulary == 0 || vocabulary > 2_097_152 {
            return Err(invalid("vocabulary size exceeds decoder bounds"));
        }
        let output = if model.tensors.contains_key("output.weight") {
            tensor(&model, "output.weight", &[geometry.embedding, vocabulary])?
        } else {
            embedding.clone()
        };
        let output_norm = vector(&model, "output_norm.weight", geometry.embedding)?;
        let transforms = Transforms::read(&model, &geometry)?;
        let full_interval = optional_integer(metadata, "qwen35.full_attention_interval", 4)?;
        if full_interval == 0 {
            return Err(invalid("full attention interval must be positive"));
        }
        let recurrent_flags = if metadata.contains_key("qwen35.attention.recurrent_layers") {
            let flags = array(metadata, "qwen35.attention.recurrent_layers")?;
            if flags.len() != block_count {
                return Err(invalid("recurrent layer flags must cover all blocks"));
            }
            flags
                .iter()
                .map(|value| {
                    value
                        .as_bool()
                        .ok_or_else(|| invalid("recurrent layer flags must be boolean"))
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            (0..block_count)
                .map(|index| index < layer_count && !(index + 1).is_multiple_of(full_interval))
                .collect()
        };
        let mut state_bytes = 0_usize;
        for &recurrent in recurrent_flags.iter().take(layer_count) {
            let elements = if recurrent {
                checked_sum(
                    checked_product(&[geometry.ssm_head, geometry.ssm_head, geometry.ssm_values])?,
                    checked_product(&[geometry.convolution - 1, geometry.convolution_width()])?,
                )?
            } else {
                checked_product(&[2, geometry.head, geometry.kv_heads, context_limit])?
            };
            let element_bytes = if recurrent {
                size_of::<f32>()
            } else {
                size_of::<u16>()
            };
            state_bytes = checked_sum(state_bytes, checked_product(&[elements, element_bytes])?)?;
        }
        if state_bytes > state_budget_bytes {
            return Err(invalid(
                "KV and recurrent state exceed the native state budget",
            ));
        }
        // Model owns immutable Arc payloads throughout construction. Validate
        // every payload once, including allowed unused MTP tensors, instead of
        // scanning each large projection again when binding its graph shape.
        // Keep this pass after the cheap geometry and state-budget checks.
        for tensor in model.tensors.values() {
            crate::kernels::validate_tensor(tensor)?;
        }
        let mut layers = capacity(layer_count)?;
        for (index, &recurrent) in recurrent_flags.iter().take(layer_count).enumerate() {
            let prefix = format!("blk.{index}.");
            let attention = if recurrent {
                let channels = geometry.convolution_width();
                let convolution = flatten(&tensor(
                    &model,
                    &format!("{prefix}ssm_conv1d.weight"),
                    &[geometry.convolution, channels],
                )?)?;
                Attention::Recurrent(RecurrentAttention {
                    qkv: tensor(
                        &model,
                        &format!("{prefix}attn_qkv.weight"),
                        &[geometry.embedding, channels],
                    )?,
                    gate: tensor(
                        &model,
                        &format!("{prefix}attn_gate.weight"),
                        &[geometry.embedding, geometry.value_width()],
                    )?,
                    beta: tensor(
                        &model,
                        &format!("{prefix}ssm_beta.weight"),
                        &[geometry.embedding, geometry.ssm_values],
                    )?,
                    alpha: tensor(
                        &model,
                        &format!("{prefix}ssm_alpha.weight"),
                        &[geometry.embedding, geometry.ssm_values],
                    )?,
                    convolution,
                    time_bias: vector(
                        &model,
                        &format!("{prefix}ssm_dt.bias"),
                        geometry.ssm_values,
                    )?,
                    decay: vector(&model, &format!("{prefix}ssm_a"), geometry.ssm_values)?,
                    norm: vector(
                        &model,
                        &format!("{prefix}ssm_norm.weight"),
                        geometry.ssm_head,
                    )?,
                    output: tensor(
                        &model,
                        &format!("{prefix}ssm_out.weight"),
                        &[geometry.value_width(), geometry.embedding],
                    )?,
                })
            } else {
                Attention::Full(FullAttention {
                    query: tensor(
                        &model,
                        &format!("{prefix}attn_q.weight"),
                        &[geometry.embedding, geometry.head * geometry.heads * 2],
                    )?,
                    key: tensor(
                        &model,
                        &format!("{prefix}attn_k.weight"),
                        &[geometry.embedding, geometry.head * geometry.kv_heads],
                    )?,
                    value: tensor(
                        &model,
                        &format!("{prefix}attn_v.weight"),
                        &[geometry.embedding, geometry.head * geometry.kv_heads],
                    )?,
                    output: tensor(
                        &model,
                        &format!("{prefix}attn_output.weight"),
                        &[geometry.head * geometry.heads, geometry.embedding],
                    )?,
                    query_norm: vector(
                        &model,
                        &format!("{prefix}attn_q_norm.weight"),
                        geometry.head,
                    )?,
                    key_norm: vector(
                        &model,
                        &format!("{prefix}attn_k_norm.weight"),
                        geometry.head,
                    )?,
                })
            };
            layers.push(Layer {
                norm: vector(
                    &model,
                    &format!("{prefix}attn_norm.weight"),
                    geometry.embedding,
                )?,
                post_norm: vector(
                    &model,
                    &format!("{prefix}post_attention_norm.weight"),
                    geometry.embedding,
                )?,
                attention,
                up: tensor(
                    &model,
                    &format!("{prefix}ffn_up.weight"),
                    &[geometry.embedding, geometry.feed_forward],
                )?,
                gate: tensor(
                    &model,
                    &format!("{prefix}ffn_gate.weight"),
                    &[geometry.embedding, geometry.feed_forward],
                )?,
                down: tensor(
                    &model,
                    &format!("{prefix}ffn_down.weight"),
                    &[geometry.feed_forward, geometry.embedding],
                )?,
            });
        }
        reject_unconsumed_tensors(&model, &layers, block_count, mtp)?;
        let mut state = capacity(layer_count)?;
        for layer in &layers {
            state.push(match &layer.attention {
                Attention::Full(_) => LayerState::Full {
                    keys: KvCache::new(geometry.head * geometry.kv_heads, context_limit)?,
                    values: KvCache::new(geometry.head * geometry.kv_heads, context_limit)?,
                },
                Attention::Recurrent(_) => LayerState::Recurrent {
                    convolution: zeros((geometry.convolution - 1) * geometry.convolution_width())?,
                    matrix: zeros(geometry.ssm_head * geometry.ssm_head * geometry.ssm_values)?,
                },
            });
        }
        let mut deltas = capacity(layer_count)?;
        for state in &state {
            deltas.push(match state {
                LayerState::Full { .. } => LayerDelta::Full {
                    key: Vec::new(),
                    value: Vec::new(),
                },
                LayerState::Recurrent { .. } => LayerDelta::Recurrent {
                    convolution: Vec::new(),
                    matrix: Vec::new(),
                },
            });
        }
        // Verified loaders set identity to the complete file SHA256. Select the
        // compiled native contract only for the exact current-weight PQ2 and
        // lossless PTQ1 artifacts. Packing does not change the graph's native
        // arithmetic; generic artifacts retain source-order controls.
        let arithmetic = if matches!(
            model.identity.as_str(),
            crate::model::swift::MODEL_SHA256 | crate::model::swift::REPACKED_PTQ1_SHA256
        ) {
            Arithmetic::PinnedSwift
        } else {
            Arithmetic::Source
        };
        let contract = if matches!(arithmetic, Arithmetic::PinnedSwift) {
            SWIFT_GRAPH_CONTRACT
        } else {
            QWEN35_GRAPH_CONTRACT
        };
        let identity = model.identity + ":" + contract;
        Ok(Self {
            provenance: Arc::new(()),
            identity,
            geometry,
            arithmetic,
            embedding,
            output,
            output_norm,
            transforms,
            adapter: None,
            adapter_scale: 0.0,
            layers,
            state,
            workspace: Workspace {
                deltas,
                adapter: AdapterScratch::new(
                    backend,
                    matches!(arithmetic, Arithmetic::PinnedSwift),
                ),
                projection: ProjectionScratch {
                    matvec: arithmetic.scratch(backend),
                    ..ProjectionScratch::default()
                },
                ..Workspace::default()
            },
            context_limit,
            position: 0,
            vocabulary,
            narrow_last_layer: layer_count >= 4,
            tiled_prefill: None,
            projection_executor: None,
        })
    }

    /// Returns the configured packed-matrix kernel request.
    ///
    /// This is a dispatch policy, not a whole-graph effective backend. Individual
    /// projections may fall back to scalar even after a NEON projection succeeds.
    /// Checkpoint restoration and reset preserve this request.
    #[must_use]
    pub const fn requested_backend(&self) -> MatvecBackend {
        self.workspace.projection.matvec.requested_backend()
    }

    /// Sets explicit packed-base-projection row scheduling for subsequent tokens.
    ///
    /// Weights and prepared Q8 activations remain borrowed and immutable; rows
    /// keep their exact kernel arithmetic. Other graph work remains serial.
    /// Reset and checkpoint restoration preserve this resource policy. It does
    /// not change numerical/state identity or preempt in-flight projections;
    /// ordinary decode cancellation is checked between synchronous operations.
    /// Explicit CPU tiles share this pool and keep cooperative cancellation on
    /// their calling thread while workers finish or stop before the next row.
    pub fn set_row_parallelism(&mut self, parallelism: RowParallelism) {
        if let Some(tile) = &mut self.tiled_prefill {
            tile.set_row_parallelism(parallelism.clone());
        }
        self.workspace
            .projection
            .matvec
            .set_row_parallelism(parallelism);
    }

    /// Configured packed-projection row-worker budget, serial by default.
    #[must_use]
    pub const fn row_workers(&self) -> usize {
        self.workspace.projection.matvec.row_workers()
    }

    /// Cloned row-parallelism handle for external projection executors.
    #[must_use]
    pub fn clone_row_parallelism(&self) -> RowParallelism {
        self.workspace.projection.matvec.clone_row_parallelism()
    }

    /// Install or clear an optional persistent PQ2 projection executor.
    pub fn set_projection_executor(
        &mut self,
        executor: Option<Box<dyn crate::kernels::Pq2ProjectionExecutor + Send>>,
    ) {
        self.projection_executor = executor;
    }

    /// Enable or disable last-layer (lk.N-1) state-only prefill row narrowing.
    ///
    /// Enabled by default for graphs with at least 4 layers (such as the 64-layer
    /// Qwen3.5-27B hybrid stack). When enabled, state-only token/tile ingestion
    /// (`ingest` and `ingest_block` without output logits) computes only the final
    /// layer's committed state (K/V cache append on a full-attention final
    /// layer or convolution/matrix update on a recurrent final layer), skipping
    /// the unused final-layer query, attention, output, and FFN projections with
    /// zero change to committed state or subsequent token logits.
    pub const fn set_last_layer_prefill_narrowing(&mut self, enabled: bool) {
        self.narrow_last_layer = enabled;
    }

    /// Returns whether last-layer state-only prefill row narrowing is enabled.
    #[must_use]
    pub const fn last_layer_prefill_narrowing(&self) -> bool {
        self.narrow_last_layer
    }

    /// Look up a named projection tensor across all decoder layers.
    #[must_use]
    pub fn projection_tensor(&self, name: &str) -> Option<&Tensor> {
        for layer in &self.layers {
            match &layer.attention {
                Attention::Full(attention) => {
                    for tensor in [
                        &attention.query,
                        &attention.key,
                        &attention.value,
                        &attention.output,
                    ] {
                        if tensor.name == name {
                            return Some(tensor);
                        }
                    }
                }
                Attention::Recurrent(attention) => {
                    for tensor in [
                        &attention.qkv,
                        &attention.gate,
                        &attention.beta,
                        &attention.alpha,
                        &attention.output,
                    ] {
                        if tensor.name == name {
                            return Some(tensor);
                        }
                    }
                }
            }
            for tensor in [&layer.up, &layer.gate, &layer.down] {
                if tensor.name == name {
                    return Some(tensor);
                }
            }
        }
        if self.output.name == name {
            return Some(&self.output);
        }
        None
    }

    /// Diagnostics for the latest matrix attempt, including provisional work.
    ///
    /// This describes one projection, not every projection in a token or graph.
    /// Failure before matrix preparation, reset and checkpoint restoration leave
    /// the record unchanged. A successful matrix may precede a later graph
    /// failure; this record does not establish committed conversation state.
    #[must_use]
    pub const fn backend_diagnostics(&self) -> MatvecDiagnostics {
        self.workspace.projection.matvec.diagnostics()
    }

    /// Aggregate matrix dispatch and failure counts for this decoder's lifetime.
    ///
    /// Reset and checkpoint restoration preserve these counts, including failed
    /// and provisional graph work. They do not describe committed token counts
    /// or establish whole-model backend coverage or performance.
    #[must_use]
    pub const fn backend_statistics(&self) -> MatvecStatistics {
        self.workspace.projection.matvec.statistics()
    }

    fn evaluate_token_into(&mut self, token: u32, logits: Option<&mut Vec<f32>>) -> Result<()> {
        const fn coerce_executor<'a>(
            executor: &'a mut Box<dyn crate::kernels::Pq2ProjectionExecutor + Send>,
        ) -> &'a mut (dyn crate::kernels::Pq2ProjectionExecutor + 'a) {
            &mut **executor
        }
        let mut executor = self.projection_executor.take();
        let result = self.evaluate_token_with_executor(
            token,
            logits,
            executor.as_mut().map(coerce_executor),
        );
        self.projection_executor = executor;
        result
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Keep validation and transactional graph operation order auditable together."
    )]
    fn evaluate_token_with_executor(
        &mut self,
        token: u32,
        logits: Option<&mut Vec<f32>>,
        executor: Option<&mut dyn crate::kernels::Pq2ProjectionExecutor>,
    ) -> Result<()> {
        let token = usize::try_from(token)?;
        if token >= self.vocabulary || self.position >= self.context_limit {
            return Err(invalid(
                "token is outside the vocabulary or context is exhausted",
            ));
        }
        let Workspace {
            hidden,
            normalized,
            attention,
            projection,
            adapter: adapter_scratch,
            attention_scratch,
            deltas,
        } = &mut self.workspace;
        let mut evaluation = Evaluation {
            geometry: &self.geometry,
            arithmetic: self.arithmetic,
            transforms: &self.transforms,
            position: self.position,
            frequencies_ready: false,
            projection,
            adapter: self.adapter.as_ref(),
            adapter_scale: self.adapter_scale,
            adapter_scratch,
            executor,
        };
        row_into(&self.embedding, token, hidden)?;
        if self.transforms.inverse_embedding {
            self.transforms
                .rotate_with_parallelism(hidden, evaluation.projection.matvec.row_parallelism())?;
            self.transforms.sign(hidden);
        }
        if let Some(adapter) = &self.adapter {
            adapter.embedding(
                token,
                self.adapter_scale,
                hidden,
                evaluation.adapter_scratch,
            )?;
        }
        let total_layers = self.layers.len();
        let narrow_final = logits.is_none() && self.narrow_last_layer;
        for (layer_idx, ((layer, state), delta)) in self
            .layers
            .iter()
            .zip(&self.state)
            .zip(deltas.iter_mut())
            .enumerate()
        {
            let is_narrow_final = narrow_final && layer_idx + 1 == total_layers;
            let first_attn_target = match &layer.attention {
                Attention::Full(weights) if is_narrow_final => &weights.key,
                Attention::Full(weights) => &weights.query,
                Attention::Recurrent(weights) => &weights.qkv,
            };
            evaluation.normalize_for_into(hidden, &layer.norm, normalized, first_attn_target)?;
            if is_narrow_final {
                match (&layer.attention, state) {
                    (Attention::Full(weights), LayerState::Full { .. }) => {
                        evaluation.full_attention_kv_only(
                            weights,
                            normalized,
                            attention_scratch,
                            delta,
                        )?;
                    }
                    (
                        Attention::Recurrent(weights),
                        LayerState::Recurrent {
                            convolution,
                            matrix,
                        },
                    ) => {
                        evaluation.recurrent_attention_state_only(
                            weights,
                            normalized,
                            convolution,
                            matrix,
                            attention_scratch,
                            delta,
                        )?;
                    }
                    _ => return Err(invalid("decoder layer/cache kind mismatch")),
                }
                continue;
            }
            match (&layer.attention, state) {
                (Attention::Full(weights), LayerState::Full { keys, values }) => {
                    evaluation.full_attention(
                        weights,
                        normalized,
                        keys,
                        values,
                        attention_scratch,
                        delta,
                        attention,
                    )?;
                }
                (
                    Attention::Recurrent(weights),
                    LayerState::Recurrent {
                        convolution,
                        matrix,
                    },
                ) => evaluation.recurrent_attention(
                    weights,
                    normalized,
                    convolution,
                    matrix,
                    attention_scratch,
                    delta,
                    attention,
                )?,
                _ => return Err(invalid("decoder layer/cache kind mismatch")),
            }
            evaluation.add_normalize_for_into(
                hidden,
                attention,
                &layer.post_norm,
                normalized,
                &layer.up,
            )?;
            // Attention no longer reads its projections or gates. Keep the
            // exact FFN operations while reusing their high-water allocations.
            let up = &mut attention_scratch.projected;
            let gate = &mut attention_scratch.gate;
            if evaluation.try_project_ffn_chain_into(
                &layer.gate,
                &layer.up,
                &layer.down,
                normalized,
                Some(hidden.as_slice()),
                attention,
            )? {
                std::mem::swap(hidden, attention);
                finite(hidden, "decoder layer output")?;
            } else {
                evaluation.project_ffn_swiglu_into(
                    &layer.gate,
                    &layer.up,
                    &layer.down,
                    normalized,
                    gate,
                    up,
                )?;
                evaluation.project_into(&layer.down, up, attention)?;
                add_finite(hidden, attention, "decoder layer output")?;
            }
        }
        if let Some(logits) = logits {
            evaluation.normalize_for_into(hidden, &self.output_norm, normalized, &self.output)?;
            evaluation.project_into(&self.output, normalized, logits)?;
            finite(logits, "output logits")?;
        }
        // Reserve every append before changing numerical state. Deltas retain
        // their allocations across tokens but never alias checkpoint storage.
        for state in &mut self.state {
            if let LayerState::Full { keys, values } = state {
                keys.reserve_append()?;
                values.reserve_append()?;
            }
        }
        for (state, delta) in self.state.iter_mut().zip(deltas.iter_mut()) {
            match (state, delta) {
                (LayerState::Full { keys, values }, LayerDelta::Full { key, value }) => {
                    keys.append(key);
                    values.append(value);
                }
                (
                    LayerState::Recurrent {
                        convolution,
                        matrix,
                    },
                    LayerDelta::Recurrent {
                        convolution: new_convolution,
                        matrix: new_matrix,
                    },
                ) => {
                    std::mem::swap(convolution, new_convolution);
                    std::mem::swap(matrix, new_matrix);
                }
                // Every kind was checked by the corresponding attention call.
                _ => {}
            }
        }
        self.position += 1;
        Ok(())
    }
}

impl Evaluation<'_, '_> {
    fn project_group_into<const N: usize>(
        &mut self,
        tensors: [&Tensor; N],
        input: &[f32],
        mut outputs: [&mut Vec<f32>; N],
    ) -> Result<()> {
        let result = (|| {
            self.transforms.project_group_into_with_executor(
                tensors,
                input,
                self.geometry,
                outputs.each_mut().map(|output| &mut **output),
                self.projection,
                &mut self.executor,
            )?;
            if let Some(adapter) = self.adapter {
                for (tensor, output) in tensors.into_iter().zip(&mut outputs) {
                    adapter.linear(
                        &tensor.name,
                        input,
                        self.adapter_scale,
                        output,
                        self.adapter_scratch,
                    )?;
                }
            }
            Ok(())
        })();
        if result.is_err() {
            for output in outputs {
                output.clear();
            }
        }
        result
    }

    fn project_into(
        &mut self,
        tensor: &Tensor,
        input: &[f32],
        output: &mut Vec<f32>,
    ) -> Result<()> {
        self.project_group_into([tensor], input, [output])
    }

    fn normalize_for_into(
        &mut self,
        input: &[f32],
        weight: &[f32],
        output: &mut Vec<f32>,
        first_target: &Tensor,
    ) -> Result<()> {
        self.projection.pre_transformed_len = None;
        if self.transforms.basis(first_target) == (true, false)
            && self.transforms.can_block_hadamard(input.len())
            && first_target.dimensions.first().copied() == Some(input.len())
        {
            let signs = self.transforms.signs.get(&input.len()).map(Vec::as_slice);
            let parallelism = self.projection.matvec.clone_row_parallelism();
            if self
                .arithmetic
                .normalize_hadamard_into(
                    input,
                    weight,
                    self.geometry.epsilon,
                    self.transforms.block,
                    signs,
                    &parallelism,
                    output,
                    &mut self.projection.transformed,
                )
                .is_ok()
            {
                self.projection.pre_transformed_len = Some(input.len());
                return Ok(());
            }
        }
        self.arithmetic
            .normalize(input, weight, self.geometry.epsilon, output)
    }

    fn add_normalize_for_into(
        &mut self,
        values: &mut [f32],
        delta: &[f32],
        weight: &[f32],
        output: &mut Vec<f32>,
        first_target: &Tensor,
    ) -> Result<()> {
        self.projection.pre_transformed_len = None;
        if self.transforms.basis(first_target) == (true, false)
            && self.transforms.can_block_hadamard(values.len())
            && first_target.dimensions.first().copied() == Some(values.len())
        {
            let signs = self.transforms.signs.get(&values.len()).map(Vec::as_slice);
            let parallelism = self.projection.matvec.clone_row_parallelism();
            if self
                .arithmetic
                .add_normalize_hadamard_into(
                    values,
                    delta,
                    weight,
                    self.geometry.epsilon,
                    self.transforms.block,
                    signs,
                    &parallelism,
                    output,
                    &mut self.projection.transformed,
                )
                .is_ok()
            {
                self.projection.pre_transformed_len = Some(values.len());
                return Ok(());
            }
            return self
                .arithmetic
                .normalize(values, weight, self.geometry.epsilon, output);
        }
        self.arithmetic
            .add_normalize(values, delta, weight, self.geometry.epsilon, output)
    }

    fn try_project_ffn_chain_into(
        &mut self,
        gate_weight: &Tensor,
        up_weight: &Tensor,
        down_weight: &Tensor,
        input: &[f32],
        residual: Option<&[f32]>,
        output: &mut Vec<f32>,
    ) -> Result<bool> {
        let Some(hadamard_block) = self.transforms.ffn_chain_hadamard_block_for(down_weight) else {
            return Ok(false);
        };
        let tensors = [gate_weight, up_weight, down_weight];
        if self.adapter.is_some()
            || self.transforms.basis(gate_weight) != self.transforms.basis(up_weight)
            || !self
                .executor
                .as_deref()
                .is_some_and(|exec| exec.supports_f32_ffn_chain(tensors))
        {
            return Ok(false);
        }
        let pre_transformed = self.projection.pre_transformed_len.take();
        let basis = self.transforms.basis(gate_weight);
        let activation = if basis.0 {
            if basis.1
                || pre_transformed != Some(input.len())
                || self.projection.transformed.len() != input.len()
            {
                self.transforms.transform_into_with_parallelism(
                    gate_weight,
                    input,
                    self.geometry,
                    self.projection.matvec.row_parallelism(),
                    &mut self.projection.transformed,
                )?;
            }
            #[cfg(test)]
            {
                self.projection.transforms += 1;
            }
            self.projection.transformed.as_slice()
        } else {
            input
        };
        let mut prepared = self
            .projection
            .matvec
            .prepare_f32_executor(gate_weight, activation)?;
        if let Some(executor) = self.executor.as_deref_mut() {
            prepared.matvec_ffn_chain_with_executor(
                tensors,
                residual,
                hadamard_block,
                output,
                executor,
            )?;
            return Ok(true);
        }
        Ok(false)
    }

    fn project_ffn_swiglu_into(
        &mut self,
        gate_weight: &Tensor,
        up_weight: &Tensor,
        down_weight: &Tensor,
        input: &[f32],
        gate: &mut Vec<f32>,
        up: &mut Vec<f32>,
    ) -> Result<()> {
        if self.adapter.is_none()
            && self.transforms.basis(gate_weight) == self.transforms.basis(up_weight)
            && self
                .executor
                .as_deref()
                .is_some_and(|exec| exec.supports_f32_ffn_swiglu(gate_weight, up_weight))
        {
            let pre_transformed = self.projection.pre_transformed_len.take();
            let basis = self.transforms.basis(gate_weight);
            let activation = if basis.0 {
                if basis.1
                    || pre_transformed != Some(input.len())
                    || self.projection.transformed.len() != input.len()
                {
                    self.transforms.transform_into_with_parallelism(
                        gate_weight,
                        input,
                        self.geometry,
                        self.projection.matvec.row_parallelism(),
                        &mut self.projection.transformed,
                    )?;
                }
                #[cfg(test)]
                {
                    self.projection.transforms += 1;
                }
                self.projection.transformed.as_slice()
            } else {
                input
            };
            let mut prepared = self
                .projection
                .matvec
                .prepare_f32_executor(gate_weight, activation)?;
            if let Some(executor) = self.executor.as_deref_mut() {
                return prepared.matvec_ffn_swiglu_with_executor(
                    gate_weight,
                    up_weight,
                    up,
                    executor,
                );
            }
        }
        self.project_group_into([up_weight, gate_weight], input, [up, gate])?;
        if self.adapter.is_none()
            && self.transforms.basis(down_weight) == (true, false)
            && self.transforms.can_block_hadamard(up.len())
            && down_weight.dimensions.first().copied() == Some(up.len())
        {
            let signs = self.transforms.signs.get(&up.len()).map(Vec::as_slice);
            if self
                .arithmetic
                .swiglu_hadamard_read_into_with_parallelism(
                    gate,
                    up,
                    self.transforms.block,
                    signs,
                    self.projection.matvec.row_parallelism(),
                    &mut self.projection.transformed,
                )
                .is_ok()
            {
                self.projection.pre_transformed_len = Some(up.len());
                return Ok(());
            }
        }
        self.arithmetic.swiglu_in_place_with_parallelism(
            gate,
            up,
            self.projection.matvec.row_parallelism(),
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Separate immutable committed state from reusable transactional scratch."
    )]
    fn full_attention(
        &mut self,
        layer: &FullAttention,
        input: &[f32],
        keys: &KvCache,
        values: &KvCache,
        scratch: &mut AttentionScratch,
        delta: &mut LayerDelta,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        let LayerDelta::Full { key, value } = delta else {
            return Err(invalid("decoder full-attention delta kind mismatch"));
        };
        self.project_group_into(
            [&layer.query, &layer.key, &layer.value],
            input,
            [&mut scratch.projected, key, value],
        )?;
        self.full_attention_state(
            layer,
            &scratch.projected,
            key,
            value,
            keys,
            values,
            &mut scratch.query,
            &mut scratch.scores,
            &mut scratch.frequencies,
            &mut scratch.attended,
        )?;
        self.project_into(&layer.output, &scratch.attended, output)
    }

    fn full_attention_kv_only(
        &mut self,
        layer: &FullAttention,
        input: &[f32],
        scratch: &mut AttentionScratch,
        delta: &mut LayerDelta,
    ) -> Result<()> {
        let LayerDelta::Full { key, value } = delta else {
            return Err(invalid("decoder full-attention delta kind mismatch"));
        };
        self.project_group_into([&layer.key, &layer.value], input, [key, value])?;
        let g = self.geometry;
        if !self.frequencies_ready {
            rope_frequencies(g, self.position, &mut scratch.frequencies)?;
            self.frequencies_ready = true;
        }
        let arithmetic = self.arithmetic;
        let key_norm: &[f32] = &layer.key_norm;
        let frequencies: &[(f32, f32)] = &scratch.frequencies;
        self.projection
            .matvec
            .row_parallelism()
            .update_paired_chunks(key, g.head, value, g.head, |_, key_chunk, value_chunk| {
                let mut key_finite = true;
                let mut value_finite = true;
                for (key_head, value_head) in key_chunk
                    .chunks_exact_mut(g.head)
                    .zip(value_chunk.chunks_exact_mut(g.head))
                {
                    key_finite &= arithmetic.normalize_rope_round_f16_in_place(
                        key_head,
                        key_norm,
                        g.epsilon,
                        g,
                        frequencies,
                    )?;
                    value_finite &= round_f16_slice(value_head);
                }
                if !key_finite {
                    return Err(invalid("nonfinite values in binary16 key cache"));
                }
                if !value_finite {
                    return Err(invalid("nonfinite values in binary16 value cache"));
                }
                Ok(())
            })
    }

    #[expect(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "Preserve scalar nonflash F16 matmul inputs and binary64 accumulation rounded to binary32."
    )]
    fn full_attention_state(
        &mut self,
        layer: &FullAttention,
        projected: &[f32],
        key: &mut [f32],
        value: &mut [f32],
        keys: &KvCache,
        values: &KvCache,
        query_scratch: &mut Vec<f32>,
        scores_scratch: &mut Vec<f32>,
        frequencies_scratch: &mut Vec<(f32, f32)>,
        attended: &mut Vec<f32>,
    ) -> Result<()> {
        let g = self.geometry;
        // Geometry and position are shared by every full-attention layer in
        // this token evaluation. Reuse the exact scalar frequencies for Q/K
        // and subsequent layers; the flag resets for every evaluation attempt.
        if !self.frequencies_ready {
            rope_frequencies(g, self.position, frequencies_scratch)?;
            self.frequencies_ready = true;
        }
        resize_into(g.head * g.heads, query_scratch)?;
        let arithmetic = self.arithmetic;
        let query_norm: &[f32] = &layer.query_norm;
        let frequencies: &[(f32, f32)] = frequencies_scratch;
        self.projection.matvec.row_parallelism().update_chunks(
            query_scratch,
            g.head,
            |base_head, query_chunk| {
                let mut query_finite = true;
                for (head_offset, query_head) in query_chunk.chunks_exact_mut(g.head).enumerate() {
                    let head_idx = base_head + head_offset;
                    let src = &projected[head_idx * 2 * g.head..(head_idx + 1) * 2 * g.head];
                    // F16 K x Q converts the F32 query to F16 even with F32 precision.
                    query_finite &= arithmetic.normalize_rope_round_f16_into(
                        &src[..g.head],
                        query_norm,
                        g.epsilon,
                        g,
                        frequencies,
                        query_head,
                    )?;
                }
                if !query_finite {
                    return Err(invalid("nonfinite values in binary16 attention query"));
                }
                Ok(())
            },
        )?;
        let key_norm: &[f32] = &layer.key_norm;
        self.projection
            .matvec
            .row_parallelism()
            .update_paired_chunks(key, g.head, value, g.head, |_, key_chunk, value_chunk| {
                let mut key_finite = true;
                let mut value_finite = true;
                for (key_head, value_head) in key_chunk
                    .chunks_exact_mut(g.head)
                    .zip(value_chunk.chunks_exact_mut(g.head))
                {
                    key_finite &= arithmetic.normalize_rope_round_f16_in_place(
                        key_head,
                        key_norm,
                        g.epsilon,
                        g,
                        frequencies,
                    )?;
                    value_finite &= round_f16_slice(value_head);
                }
                if !key_finite {
                    return Err(invalid("nonfinite values in binary16 key cache"));
                }
                if !value_finite {
                    return Err(invalid("nonfinite values in binary16 value cache"));
                }
                Ok(())
            })?;
        resize_into(g.head * g.heads, attended)?;
        let score_len = self.position + 1;
        resize_into(g.heads * score_len, scores_scratch)?;
        let scale = 1.0 / small_float(g.head)?.sqrt();
        let arithmetic = self.arithmetic;
        let query_ref: &[f32] = query_scratch;
        let key_ref: &[f32] = key;
        let value_ref: &[f32] = value;
        let heads_per_kv = g.heads / g.kv_heads;
        self.projection
            .matvec
            .row_parallelism()
            .update_paired_chunks(
                attended,
                g.head,
                scores_scratch,
                score_len,
                |base_head, attended_chunk, scores_chunk| {
                    let chunk_heads = attended_chunk.len() / g.head;
                    let mut head_offset = 0;
                    while head_offset < chunk_heads {
                        let head = base_head + head_offset;
                        let kv_head = head / heads_per_kv;
                        let offset = kv_head * g.head;
                        if head_offset + 4 <= chunk_heads && kv_head == (head + 3) / heads_per_kv {
                            let (r01, r23) = attended_chunk
                                [head_offset * g.head..(head_offset + 4) * g.head]
                                .split_at_mut(2 * g.head);
                            let (r0, r1) = r01.split_at_mut(g.head);
                            let (r2, r3) = r23.split_at_mut(g.head);
                            let (s01, s23) = scores_chunk
                                [head_offset * score_len..(head_offset + 4) * score_len]
                                .split_at_mut(2 * score_len);
                            let (s0, s1) = s01.split_at_mut(score_len);
                            let (s2, s3) = s23.split_at_mut(score_len);
                            let queries = std::array::from_fn(|i| {
                                let h = head + i;
                                &query_ref[h * g.head..(h + 1) * g.head]
                            });
                            let mut past_iter = keys.tokens().enumerate();
                            while let Some((pos0, p0)) = past_iter.next() {
                                if let Some((_, p1)) = past_iter.next() {
                                    if let Some((_, p2)) = past_iter.next() {
                                        if let Some((_, p3)) = past_iter.next() {
                                            let dots4 = arithmetic
                                                .attention_dot_four_keys_four_queries(
                                                    queries,
                                                    [
                                                        &p0[offset..offset + g.head],
                                                        &p1[offset..offset + g.head],
                                                        &p2[offset..offset + g.head],
                                                        &p3[offset..offset + g.head],
                                                    ],
                                                )?;
                                            for (k_off, dots) in dots4.into_iter().enumerate() {
                                                let pos = pos0 + k_off;
                                                s0[pos] = dots[0] * scale;
                                                s1[pos] = dots[1] * scale;
                                                s2[pos] = dots[2] * scale;
                                                s3[pos] = dots[3] * scale;
                                            }
                                        } else {
                                            let dots2 = arithmetic
                                                .attention_dot_two_keys_four_queries(
                                                    queries,
                                                    [
                                                        &p0[offset..offset + g.head],
                                                        &p1[offset..offset + g.head],
                                                    ],
                                                )?;
                                            for (k_off, dots) in dots2.into_iter().enumerate() {
                                                let pos = pos0 + k_off;
                                                s0[pos] = dots[0] * scale;
                                                s1[pos] = dots[1] * scale;
                                                s2[pos] = dots[2] * scale;
                                                s3[pos] = dots[3] * scale;
                                            }
                                            let pos2 = pos0 + 2;
                                            let dots = arithmetic.attention_dot_four(
                                                queries,
                                                &p2[offset..offset + g.head],
                                            )?;
                                            s0[pos2] = dots[0] * scale;
                                            s1[pos2] = dots[1] * scale;
                                            s2[pos2] = dots[2] * scale;
                                            s3[pos2] = dots[3] * scale;
                                        }
                                    } else {
                                        let dots2 = arithmetic
                                            .attention_dot_two_keys_four_queries(
                                                queries,
                                                [
                                                    &p0[offset..offset + g.head],
                                                    &p1[offset..offset + g.head],
                                                ],
                                            )?;
                                        for (k_off, dots) in dots2.into_iter().enumerate() {
                                            let pos = pos0 + k_off;
                                            s0[pos] = dots[0] * scale;
                                            s1[pos] = dots[1] * scale;
                                            s2[pos] = dots[2] * scale;
                                            s3[pos] = dots[3] * scale;
                                        }
                                    }
                                } else {
                                    let dots = arithmetic.attention_dot_four(
                                        queries,
                                        &p0[offset..offset + g.head],
                                    )?;
                                    s0[pos0] = dots[0] * scale;
                                    s1[pos0] = dots[1] * scale;
                                    s2[pos0] = dots[2] * scale;
                                    s3[pos0] = dots[3] * scale;
                                }
                            }
                            let current_dots = arithmetic.attention_current_dot_four(
                                queries,
                                &key_ref[offset..offset + g.head],
                            )?;
                            s0[score_len - 1] = current_dots[0] * scale;
                            s1[score_len - 1] = current_dots[1] * scale;
                            s2[score_len - 1] = current_dots[2] * scale;
                            s3[score_len - 1] = current_dots[3] * scale;
                            for scores in [&mut *s0, &mut *s1, &mut *s2, &mut *s3] {
                                arithmetic.attention_softmax_round_f16(scores)?;
                            }
                            let probs: [&[f32]; 4] = [s0, s1, s2, s3];
                            let g0 = attention_gate(projected, head, g.head);
                            let g1 = attention_gate(projected, head + 1, g.head);
                            let g2 = attention_gate(projected, head + 2, g.head);
                            let g3 = attention_gate(projected, head + 3, g.head);
                            for (tile, (((out0, out1), out2), out3)) in
                                r0.chunks_mut(crate::kernels::pinned_attention::VALUE_TILE_FOUR)
                                    .zip(r1.chunks_mut(
                                        crate::kernels::pinned_attention::VALUE_TILE_FOUR,
                                    ))
                                    .zip(r2.chunks_mut(
                                        crate::kernels::pinned_attention::VALUE_TILE_FOUR,
                                    ))
                                    .zip(r3.chunks_mut(
                                        crate::kernels::pinned_attention::VALUE_TILE_FOUR,
                                    ))
                                    .enumerate()
                            {
                                let dimension =
                                    tile * crate::kernels::pinned_attention::VALUE_TILE_FOUR;
                                let len = out0.len();
                                arithmetic.attention_value_tile_four(
                                    values,
                                    &value_ref[offset + dimension..offset + dimension + len],
                                    offset + dimension,
                                    probs,
                                    [out0, out1, out2, out3],
                                )?;
                                mul_sigmoid_in_place(out0, &g0[dimension..dimension + len]);
                                mul_sigmoid_in_place(out1, &g1[dimension..dimension + len]);
                                mul_sigmoid_in_place(out2, &g2[dimension..dimension + len]);
                                mul_sigmoid_in_place(out3, &g3[dimension..dimension + len]);
                            }
                            head_offset += 4;
                            continue;
                        }
                        let result =
                            &mut attended_chunk[head_offset * g.head..(head_offset + 1) * g.head];
                        let scores = &mut scores_chunk
                            [head_offset * score_len..(head_offset + 1) * score_len];
                        let q = &query_ref[head * g.head..(head + 1) * g.head];
                        for (pos, past) in keys.tokens().enumerate() {
                            scores[pos] = arithmetic
                                .attention_dot(q, &past[offset..offset + g.head])?
                                * scale;
                        }
                        scores[score_len - 1] = arithmetic
                            .attention_current_dot(q, &key_ref[offset..offset + g.head])?
                            * scale;
                        // Nonflash V(F16) x softmax(F16) also converts its RHS to F16.
                        arithmetic.attention_softmax_round_f16(scores)?;
                        let g_head = attention_gate(projected, head, g.head);
                        for (tile, out) in result
                            .chunks_mut(crate::kernels::pinned_attention::VALUE_TILE)
                            .enumerate()
                        {
                            let dimension = tile * crate::kernels::pinned_attention::VALUE_TILE;
                            let len = out.len();
                            arithmetic.attention_value_tile(
                                values,
                                &value_ref[offset + dimension..offset + dimension + len],
                                offset + dimension,
                                scores,
                                out,
                            )?;
                            mul_sigmoid_in_place(out, &g_head[dimension..dimension + len]);
                        }
                        head_offset += 1;
                    }
                    Ok(())
                },
            )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Separate immutable committed state from reusable transactional scratch."
    )]
    fn recurrent_attention(
        &mut self,
        layer: &RecurrentAttention,
        input: &[f32],
        history: &[f32],
        previous: &[f32],
        scratch: &mut AttentionScratch,
        delta: &mut LayerDelta,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        self.project_group_into(
            [&layer.qkv, &layer.gate, &layer.beta, &layer.alpha],
            input,
            [
                &mut scratch.projected,
                &mut scratch.gate,
                &mut scratch.beta,
                &mut scratch.alpha,
            ],
        )?;
        self.recurrent_attention_state(
            layer,
            &mut scratch.projected,
            &scratch.gate,
            &scratch.beta,
            &scratch.alpha,
            history,
            previous,
            delta,
            &mut scratch.attended,
        )?;
        self.project_into(&layer.output, &scratch.attended, output)
    }

    fn recurrent_attention_state_only(
        &mut self,
        layer: &RecurrentAttention,
        input: &[f32],
        history: &[f32],
        previous: &[f32],
        scratch: &mut AttentionScratch,
        delta: &mut LayerDelta,
    ) -> Result<()> {
        self.project_group_into(
            [&layer.qkv, &layer.beta, &layer.alpha],
            input,
            [
                &mut scratch.projected,
                &mut scratch.beta,
                &mut scratch.alpha,
            ],
        )?;
        resize_into(self.geometry.value_width(), &mut scratch.gate)?;
        self.recurrent_attention_state(
            layer,
            &mut scratch.projected,
            &scratch.gate,
            &scratch.beta,
            &scratch.alpha,
            history,
            previous,
            delta,
            &mut scratch.attended,
        )
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Keep validation and graph operation order auditable together."
    )]
    #[expect(
        clippy::imprecise_flops,
        reason = "Match the pinned scalar GDN operations, including logf(1+expf(x))."
    )]
    #[expect(
        clippy::too_many_arguments,
        reason = "Separate immutable committed state from reusable transactional scratch."
    )]
    fn recurrent_attention_state(
        &self,
        layer: &RecurrentAttention,
        mixed: &mut [f32],
        gate_ref: &[f32],
        beta_ref: &[f32],
        alpha_ref: &[f32],
        history: &[f32],
        previous: &[f32],
        delta: &mut LayerDelta,
        attended: &mut Vec<f32>,
    ) -> Result<()> {
        let LayerDelta::Recurrent {
            convolution,
            matrix,
        } = delta
        else {
            return Err(invalid("decoder recurrent-attention delta kind mismatch"));
        };
        let g = self.geometry;
        let hist_width = g.convolution - 1;
        if hist_width > 0 {
            resize_into(history.len(), convolution)?;
        } else {
            zeros_into(history.len(), convolution)?;
        }
        let key_width = g.key_width();
        let arithmetic = self.arithmetic;
        let parallelism = self.projection.matvec.row_parallelism();
        if hist_width > 0
            && parallelism.workers() > 1
            && mixed.len().is_multiple_of(g.ssm_head)
            && g.ssm_head.is_multiple_of(4)
        {
            let head_hist = g.ssm_head * hist_width;
            let head_coeff = g.ssm_head * g.convolution;
            let qk_heads = (2 * key_width) / g.ssm_head;
            parallelism.update_paired_chunks(
                mixed,
                g.ssm_head,
                convolution,
                head_hist,
                |base_head, mixed_chunk, conv_chunk| {
                    for (offset, (head_mixed, head_conv)) in mixed_chunk
                        .chunks_exact_mut(g.ssm_head)
                        .zip(conv_chunk.chunks_exact_mut(head_hist))
                        .enumerate()
                    {
                        let head = base_head + offset;
                        let h_slice = &history[head * head_hist..(head + 1) * head_hist];
                        let c_slice =
                            &layer.convolution[head * head_coeff..(head + 1) * head_coeff];
                        if matches!(arithmetic, Arithmetic::PinnedSwift) {
                            crate::kernels::pinned_convolution::step_silu_into(
                                h_slice,
                                head_mixed,
                                c_slice,
                                g.convolution,
                                head_conv,
                            )?;
                        } else {
                            for (channel, mixed_value) in head_mixed.iter_mut().enumerate() {
                                let projected = *mixed_value;
                                let start = channel * hist_width;
                                let coefficients = &c_slice
                                    [channel * g.convolution..(channel + 1) * g.convolution];
                                *mixed_value = arithmetic.convolution_row(
                                    &h_slice[start..start + hist_width],
                                    projected,
                                    coefficients,
                                )?;
                                if hist_width == 3 {
                                    head_conv[start] = h_slice[start + 1];
                                    head_conv[start + 1] = h_slice[start + 2];
                                    head_conv[start + 2] = projected;
                                } else {
                                    head_conv[start..start + hist_width - 1]
                                        .copy_from_slice(&h_slice[start + 1..start + hist_width]);
                                    head_conv[start + hist_width - 1] = projected;
                                }
                            }
                            arithmetic.silu(head_mixed);
                        }
                        if head < qk_heads {
                            l2_norm(head_mixed, g.epsilon);
                        }
                    }
                    Ok(())
                },
            )?;
        } else {
            if matches!(arithmetic, Arithmetic::PinnedSwift) {
                crate::kernels::pinned_convolution::step_silu_into(
                    history,
                    mixed,
                    &layer.convolution,
                    g.convolution,
                    convolution,
                )?;
            } else {
                for (channel, mixed_value) in mixed.iter_mut().enumerate() {
                    let projected = *mixed_value;
                    let start = channel * hist_width;
                    let coefficients =
                        &layer.convolution[channel * g.convolution..(channel + 1) * g.convolution];
                    *mixed_value = arithmetic.convolution_row(
                        &history[start..start + hist_width],
                        projected,
                        coefficients,
                    )?;
                    if hist_width == 3 {
                        convolution[start] = history[start + 1];
                        convolution[start + 1] = history[start + 2];
                        convolution[start + 2] = projected;
                    } else if hist_width > 0 {
                        convolution[start..start + hist_width - 1]
                            .copy_from_slice(&history[start + 1..start + hist_width]);
                        convolution[start + hist_width - 1] = projected;
                    }
                }
                arithmetic.silu(mixed);
            }
            for head in mixed[..2 * key_width].chunks_exact_mut(g.ssm_head) {
                l2_norm(head, g.epsilon);
            }
        }
        resize_into(previous.len(), matrix)?;
        resize_into(g.value_width(), attended)?;
        let scale = 1.0 / small_float(g.ssm_head)?.sqrt();
        let head_state_len = g.ssm_head * g.ssm_head;
        let arithmetic = self.arithmetic;
        let mixed_ref: &[f32] = mixed;
        self.projection
            .matvec
            .row_parallelism()
            .update_paired_chunks(
                matrix,
                head_state_len,
                attended,
                g.ssm_head,
                |base_head, state_chunk, attended_chunk| {
                    for (offset, (state, output)) in state_chunk
                        .chunks_exact_mut(head_state_len)
                        .zip(attended_chunk.chunks_exact_mut(g.ssm_head))
                        .enumerate()
                    {
                        let head = base_head + offset;
                        let prev_state =
                            &previous[head * head_state_len..(head + 1) * head_state_len];
                        // ggml repeat tiles key groups; it does not repeat each key head contiguously.
                        let key_head = head % g.ssm_keys;
                        let q = &mixed_ref[key_head * g.ssm_head..(key_head + 1) * g.ssm_head];
                        let k = &mixed_ref[key_width + key_head * g.ssm_head
                            ..key_width + (key_head + 1) * g.ssm_head];
                        let v = &mixed_ref[2 * key_width + head * g.ssm_head
                            ..2 * key_width + (head + 1) * g.ssm_head];
                        let activation = alpha_ref[head] + layer.time_bias[head];
                        let softplus = if activation > 20.0 {
                            activation
                        } else {
                            (1.0 + activation.exp()).ln()
                        };
                        let decay = (softplus * layer.decay[head]).exp();
                        let update = sigmoid(beta_ref[head]);
                        let row_stride = g.ssm_head;
                        let row_count = v.len();
                        let eight_row_blocks = row_count / 8;
                        for block_idx in 0..eight_row_blocks {
                            let row_idx = block_idx * 8;
                            let prev_rows: [&[f32]; 8] = std::array::from_fn(|r| {
                                &prev_state
                                    [(row_idx + r) * row_stride..(row_idx + r + 1) * row_stride]
                            });
                            let state_slice =
                                &mut state[row_idx * row_stride..(row_idx + 8) * row_stride];
                            let (h0, h1) = state_slice.split_at_mut(4 * row_stride);
                            let (s01, s23) = h0.split_at_mut(2 * row_stride);
                            let (s45, s67) = h1.split_at_mut(2 * row_stride);
                            let (s0, s1) = s01.split_at_mut(row_stride);
                            let (s2, s3) = s23.split_at_mut(row_stride);
                            let (s4, s5) = s45.split_at_mut(row_stride);
                            let (s6, s7) = s67.split_at_mut(row_stride);
                            let values: [f32; 8] = std::array::from_fn(|r| v[row_idx + r]);
                            let outs = arithmetic.gdn_step_eight_rows_from(
                                prev_rows,
                                [s0, s1, s2, s3, s4, s5, s6, s7],
                                k,
                                q,
                                decay,
                                values,
                                update,
                            )?;
                            for (r, &out_val) in outs.iter().enumerate() {
                                output[row_idx + r] = out_val * scale;
                            }
                        }
                        let four_row_blocks = row_count / 4;
                        for block_idx in (eight_row_blocks * 2)..four_row_blocks {
                            let row_idx = block_idx * 4;
                            let prev_rows = [
                                &prev_state[row_idx * row_stride..(row_idx + 1) * row_stride],
                                &prev_state[(row_idx + 1) * row_stride..(row_idx + 2) * row_stride],
                                &prev_state[(row_idx + 2) * row_stride..(row_idx + 3) * row_stride],
                                &prev_state[(row_idx + 3) * row_stride..(row_idx + 4) * row_stride],
                            ];
                            let state_slice =
                                &mut state[row_idx * row_stride..(row_idx + 4) * row_stride];
                            let (s01, s23) = state_slice.split_at_mut(2 * row_stride);
                            let (s0, s1) = s01.split_at_mut(row_stride);
                            let (s2, s3) = s23.split_at_mut(row_stride);
                            let values =
                                [v[row_idx], v[row_idx + 1], v[row_idx + 2], v[row_idx + 3]];
                            let outs = arithmetic.gdn_step_four_rows_from(
                                prev_rows,
                                [s0, s1, s2, s3],
                                k,
                                q,
                                decay,
                                values,
                                update,
                            )?;
                            output[row_idx] = outs[0] * scale;
                            output[row_idx + 1] = outs[1] * scale;
                            output[row_idx + 2] = outs[2] * scale;
                            output[row_idx + 3] = outs[3] * scale;
                        }
                        for row_idx in (four_row_blocks * 4)..row_count {
                            let prev_row =
                                &prev_state[row_idx * row_stride..(row_idx + 1) * row_stride];
                            let state_row =
                                &mut state[row_idx * row_stride..(row_idx + 1) * row_stride];
                            output[row_idx] = arithmetic.gdn_step_row_from(
                                prev_row, state_row, k, q, decay, v[row_idx], update,
                            )? * scale;
                        }
                        // Pinned qwen35.cpp build_norm_gated explicitly uses
                        // ggml_swiglu_split(gate, normalized), not separate SiLU/MUL nodes.
                        arithmetic.normalize_swiglu_in_place(
                            output,
                            &layer.norm,
                            g.epsilon,
                            &gate_ref[head * g.ssm_head..(head + 1) * g.ssm_head],
                        )?;
                    }
                    Ok(())
                },
            )?;
        finite(convolution, "convolution state")
    }
}

impl Decoder for Qwen35 {
    type Checkpoint = Qwen35Checkpoint;

    fn identity(&self) -> &str {
        &self.identity
    }

    fn vocab_size(&self) -> usize {
        self.vocabulary
    }

    fn reset(&mut self) -> Result<()> {
        // Construction and successful reset establish canonical empty state.
        // Evaluation writes only provisional deltas until its infallible commit
        // advances position, and failed restore staging never changes live state.
        // Avoid refilling the recurrent buffers when callers reset twice (for
        // example request-boundary reset followed by runtime prefix preparation).
        if self.position == 0 {
            return Ok(());
        }
        for state in &mut self.state {
            match state {
                LayerState::Full { keys, values } => {
                    keys.clear();
                    values.clear();
                }
                LayerState::Recurrent {
                    convolution,
                    matrix,
                } => {
                    convolution.fill(0.0);
                    matrix.fill(0.0);
                }
            }
        }
        self.position = 0;
        Ok(())
    }

    fn evaluate(&mut self, token: u32) -> Result<Vec<f32>> {
        let mut logits = Vec::new();
        self.evaluate_into(token, &mut logits)?;
        Ok(logits)
    }

    fn evaluate_into(&mut self, token: u32, logits: &mut Vec<f32>) -> Result<()> {
        let result = self.evaluate_token_into(token, Some(logits));
        if result.is_err() {
            logits.clear();
        }
        result
    }
    fn ingest(&mut self, token: u32) -> Result<()> {
        self.evaluate_token_into(token, None)
    }
    fn ingest_block(&mut self, tokens: &[u32], check: &dyn Fn() -> Result<()>) -> Result<()> {
        self.ingest_prefill_block(tokens, check)
    }
    fn checkpoint_bytes(&self) -> usize {
        // Deliberately charge each referenced page in full for every snapshot,
        // including its Arc header and page table. Runtime admission therefore
        // overcounts shared pages rather than undercounting unique pages or
        // retained recurrent snapshots. Tails are private: subsequent append
        // needs no payload COW. Allocator overhead and graph scratch remain
        // outside this element/metadata storage bound, as for live state.
        // Empty checkpoints retain no layer payload. Continue charging their
        // unused layer-table capacity conservatively, without charging the
        // canonical zero recurrent buffers or spare KV capacity after reset.
        size_of::<Qwen35Checkpoint>()
            + self.identity.capacity()
            + self.state.capacity() * size_of::<LayerState>()
            + if self.position == 0 {
                0
            } else {
                self.state.iter().map(LayerState::bytes).sum::<usize>()
            }
    }

    fn checkpoint(&self) -> Result<Self::Checkpoint> {
        Ok(Qwen35Checkpoint {
            provenance: self.provenance.clone(),
            identity: copy_string(&self.identity)?,
            context_limit: self.context_limit,
            position: self.position,
            // Graph evaluation commits all layer deltas and position together.
            // Position zero therefore has one exact representation, even after
            // a failed first evaluation has populated workspace scratch.
            layers: if self.position == 0 {
                Vec::new()
            } else {
                copy_state(&self.state)?
            },
        })
    }

    fn restore(&mut self, checkpoint: &Self::Checkpoint) -> Result<()> {
        if !Arc::ptr_eq(&checkpoint.provenance, &self.provenance)
            || checkpoint.identity != self.identity
            || checkpoint.context_limit != self.context_limit
            || if checkpoint.position == 0 {
                !checkpoint.layers.is_empty()
            } else {
                checkpoint.layers.len() != self.layers.len()
            }
            || checkpoint.position > self.context_limit
        {
            return Err(invalid(
                "checkpoint belongs to a different model or context",
            ));
        }
        if checkpoint.position == 0 {
            // All provenance and shape checks precede this infallible reset of
            // existing live storage; scratch and other snapshots remain private.
            return self.reset();
        }
        restore_state(&mut self.state, &checkpoint.layers, KvCache::copy)?;
        self.position = checkpoint.position;
        Ok(())
    }
}

fn foldable(name: &str) -> bool {
    if name == "output.weight" {
        return true;
    }
    let Some(rest) = name.strip_prefix("blk.") else {
        return false;
    };
    let Some((index, suffix)) = rest.split_once('.') else {
        return false;
    };
    !index.is_empty()
        && index.bytes().all(|byte| byte.is_ascii_digit())
        && matches!(
            suffix,
            "attn_q.weight"
                | "attn_k.weight"
                | "attn_v.weight"
                | "attn_qkv.weight"
                | "attn_gate.weight"
                | "attn_output.weight"
                | "ffn_gate.weight"
                | "ffn_up.weight"
                | "ffn_down.weight"
                | "ssm_out.weight"
        )
}

fn reject_unsupported(metadata: &Metadata) -> Result<()> {
    for key in metadata.keys() {
        if let Some(key) = key.strip_prefix("qwen35.")
            && !matches!(
                key,
                "block_count"
                    | "nextn_predict_layers"
                    | "context_length"
                    | "embedding_length"
                    | "feed_forward_length"
                    | "attention.head_count"
                    | "attention.head_count_kv"
                    | "attention.key_length"
                    | "attention.value_length"
                    | "attention.layer_norm_rms_epsilon"
                    | "attention.recurrent_layers"
                    | "attention.causal"
                    | "rope.dimension_count"
                    | "rope.dimension_sections"
                    | "rope.freq_base"
                    | "rope.scaling.type"
                    | "ssm.state_size"
                    | "ssm.group_count"
                    | "ssm.time_step_rank"
                    | "ssm.inner_size"
                    | "ssm.conv_kernel"
                    | "full_attention_interval"
                    | "expert_count"
                    | "expert_used_count"
            )
        {
            return Err(invalid(format!("unsupported Qwen35 metadata: {key}")));
        }
        if let Some(key) = key.strip_prefix("prism.hadamard.")
            && !matches!(
                key,
                "version"
                    | "tied_output"
                    | "block_size"
                    | "transform"
                    | "axis"
                    | "sign_mode"
                    | "weight_names"
                    | "inverse_weight_names"
                    | "sign_widths"
                    | "sign_values"
                    | "gdn_v_grouped"
            )
        {
            return Err(invalid(format!("unsupported Hadamard metadata: {key}")));
        }
    }
    for key in ["qwen35.expert_count", "qwen35.expert_used_count"] {
        if optional_integer(metadata, key, 0)? != 0 {
            return Err(invalid("MoE graph is not supported"));
        }
    }
    for key in [
        "qwen35.rope.scaling.factor",
        "qwen35.rope.scaling.attn_factor",
        "qwen35.attention.scale",
    ] {
        if metadata.contains_key(key) {
            return Err(invalid(format!("unsupported graph override: {key}")));
        }
    }
    if let Some(scaling) = metadata.get("qwen35.rope.scaling.type")
        && scaling.as_str() != Some("none")
    {
        return Err(invalid("scaled RoPE is not supported"));
    }
    if !optional_bool(metadata, "qwen35.attention.causal", true)? {
        return Err(invalid("noncausal Qwen35 attention is not supported"));
    }
    Ok(())
}

fn reject_unconsumed_tensors(
    model: &Model,
    layers: &[Layer],
    block_count: usize,
    mtp: usize,
) -> Result<()> {
    let mut consumed: BTreeSet<String> =
        ["token_embd.weight", "output.weight", "output_norm.weight"]
            .into_iter()
            .map(str::to_owned)
            .collect();
    for (index, layer) in layers.iter().enumerate() {
        for suffix in [
            "attn_norm.weight",
            "post_attention_norm.weight",
            "ffn_up.weight",
            "ffn_gate.weight",
            "ffn_down.weight",
        ] {
            consumed.insert(format!("blk.{index}.{suffix}"));
        }
        let attention: &[&str] = match &layer.attention {
            Attention::Full(_) => &[
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "attn_q_norm.weight",
                "attn_k_norm.weight",
            ],
            Attention::Recurrent(_) => &[
                "attn_qkv.weight",
                "attn_gate.weight",
                "ssm_beta.weight",
                "ssm_alpha.weight",
                "ssm_conv1d.weight",
                "ssm_dt.bias",
                "ssm_a",
                "ssm_norm.weight",
                "ssm_out.weight",
            ],
        };
        for suffix in attention {
            consumed.insert(format!("blk.{index}.{suffix}"));
        }
    }
    for (name, tensor) in &model.tensors {
        let mtp_tensor = mtp != 0
            && (name.starts_with("nextn.")
                || name.contains(".nextn.")
                || name
                    .strip_prefix("blk.")
                    .and_then(|rest| rest.split_once('.'))
                    .and_then(|(index, _)| index.parse::<usize>().ok())
                    .is_some_and(|index| index >= layers.len() && index < block_count));
        if tensor.name != *name || (!consumed.contains(name) && !mtp_tensor) {
            return Err(invalid(format!("unsupported or unconsumed tensor: {name}")));
        }
    }
    Ok(())
}

fn tensor(model: &Model, name: &str, shape: &[usize]) -> Result<Tensor> {
    let tensor = model
        .tensors
        .get(name)
        .ok_or_else(|| invalid(format!("required tensor absent: {name}")))?;
    if tensor.name != name
        || tensor.dimensions.len() != shape.len()
        || tensor
            .dimensions
            .iter()
            .zip(shape)
            .any(|(&actual, &expected)| actual == 0 || (expected != 0 && actual != expected))
    {
        return Err(invalid(format!("tensor shape mismatch: {name}")));
    }
    // This private constructor lookup checks graph geometry only. The complete
    // model pass validates every immutable payload before graph publication;
    // initial embedding/output bindings are not executed before that pass.
    Ok(tensor.clone())
}

fn vector(model: &Model, name: &str, width: usize) -> Result<Vec<f32>> {
    let tensor = tensor(model, name, &[width])?;
    if !matches!(tensor.kind, 0 | 1 | 30) {
        return Err(invalid(format!(
            "normalization/state tensor must be floating point: {name}"
        )));
    }
    let result = row(&tensor, 0)?;
    finite(&result, name)?;
    Ok(result)
}

fn flatten(tensor: &Tensor) -> Result<Vec<f32>> {
    if !matches!(tensor.kind, 0 | 1 | 30) {
        return Err(invalid("convolution weights must be floating point"));
    }
    let mut values = Vec::new();
    for index in 0..tensor.dimensions[1] {
        values.extend(row(tensor, index)?);
    }
    finite(&values, &tensor.name)?;
    Ok(values)
}

fn integer(metadata: &Metadata, key: &str) -> Result<usize> {
    metadata_usize(
        metadata
            .get(key)
            .ok_or_else(|| invalid(format!("required metadata absent: {key}")))?
            .as_ref(),
        key,
    )
}

fn optional_integer(metadata: &Metadata, key: &str, default: usize) -> Result<usize> {
    metadata
        .get(key)
        .map_or(Ok(default), |value| metadata_usize(value.as_ref(), key))
}

fn metadata_usize(value: MetadataValueRef<'_>, name: &str) -> Result<usize> {
    usize::try_from(
        value
            .as_u64()
            .ok_or_else(|| invalid(format!("metadata must be an unsigned integer: {name}")))?,
    )
    .map_err(Into::into)
}

fn optional_bool(metadata: &Metadata, key: &str, default: bool) -> Result<bool> {
    metadata.get(key).map_or(Ok(default), |value| {
        value
            .as_bool()
            .ok_or_else(|| invalid(format!("metadata must be boolean: {key}")))
    })
}

fn string<'a>(metadata: &'a Metadata, key: &str) -> Result<&'a str> {
    metadata
        .get(key)
        .and_then(MetadataValue::as_str)
        .ok_or_else(|| invalid(format!("metadata must be a string: {key}")))
}

fn array<'a>(metadata: &'a Metadata, key: &str) -> Result<&'a MetadataArray> {
    metadata
        .get(key)
        .and_then(MetadataValue::as_array)
        .ok_or_else(|| invalid(format!("metadata must be an array: {key}")))
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::as_conversions,
    reason = "The pinned scalar numerical contract explicitly rounds this value to binary32."
)]
fn float(metadata: &Metadata, key: &str) -> Result<f32> {
    let value = metadata
        .get(key)
        .and_then(MetadataValue::as_f64)
        .ok_or_else(|| invalid(format!("metadata must be numeric: {key}")))?;
    let result = value as f32;
    if !result.is_finite() {
        return Err(invalid(format!("nonfinite metadata: {key}")));
    }
    Ok(result)
}

fn checked_product(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1_usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(|| invalid("decoder allocation size overflow"))
    })
}

fn checked_sum(first: usize, second: usize) -> Result<usize> {
    first
        .checked_add(second)
        .ok_or_else(|| invalid("decoder allocation size overflow"))
}

fn zeros(count: usize) -> Result<Vec<f32>> {
    if count > MAX_STATE_BYTES / size_of::<f32>() {
        return Err(invalid("decoder allocation exceeds state budget"));
    }
    let mut result = Vec::new();
    result.try_reserve_exact(count)?;
    result.resize(count, 0.0);
    Ok(result)
}

fn prepare<T>(output: &mut Vec<T>, count: usize) -> Result<()> {
    output.clear();
    if count
        .checked_mul(size_of::<T>())
        .is_none_or(|bytes| bytes > MAX_STATE_BYTES)
    {
        return Err(invalid("decoder scratch exceeds allocation budget"));
    }
    output.try_reserve_exact(count)?;
    Ok(())
}

fn zeros_into(count: usize, output: &mut Vec<f32>) -> Result<()> {
    prepare(output, count)?;
    output.resize(count, 0.0);
    Ok(())
}

fn resize_into(count: usize, output: &mut Vec<f32>) -> Result<()> {
    if output.len() != count {
        zeros_into(count, output)?;
    }
    Ok(())
}

// Full attention projects [query_head, gate_head] pairs. Queries are normalized
// into separate scratch, while each immutable gate remains directly borrowable
// until attention completes. Geometry and projection lengths are validated by
// graph construction/projection before this private view is used.
fn attention_gate(projected: &[f32], head: usize, width: usize) -> &[f32] {
    &projected[(2 * head + 1) * width..(2 * head + 2) * width]
}

fn copy_into(input: &[f32], output: &mut Vec<f32>) -> Result<()> {
    if output.len() == input.len() {
        output.copy_from_slice(input);
        return Ok(());
    }
    prepare(output, input.len())?;
    output.extend_from_slice(input);
    Ok(())
}

// Retain the unqualified copy/decay candidate in model-free controls only.
// The complete matrix must be reserved fallibly by its test caller. Production
// retains the previously released copy-then-scale path pending device evidence.
#[cfg(test)]
fn append_decayed_head<'a>(
    previous: &[f32],
    decay: f32,
    output: &'a mut Vec<f32>,
) -> &'a mut [f32] {
    let start = output.len();
    output.extend(previous.iter().map(|&element| element * decay));
    &mut output[start..]
}

fn capacity<T>(count: usize) -> Result<Vec<T>> {
    if count
        .checked_mul(size_of::<T>())
        .is_none_or(|bytes| bytes > MAX_STATE_BYTES)
    {
        return Err(invalid("decoder vector exceeds allocation budget"));
    }
    let mut result = Vec::new();
    result.try_reserve_exact(count)?;
    Ok(result)
}

fn copy_values<T: Copy>(values: &[T]) -> Result<Vec<T>> {
    let mut result = capacity(values.len())?;
    result.extend_from_slice(values);
    Ok(result)
}

fn copy_string(value: &str) -> Result<String> {
    let mut result = String::new();
    result.try_reserve_exact(value.len())?;
    result.push_str(value);
    Ok(result)
}

fn copy_state(states: &[LayerState]) -> Result<Vec<LayerState>> {
    let mut result = capacity(states.len())?;
    for state in states {
        result.push(match state {
            LayerState::Full { keys, values } => LayerState::Full {
                keys: keys.copy()?,
                values: values.copy()?,
            },
            LayerState::Recurrent {
                convolution,
                matrix,
            } => LayerState::Recurrent {
                convolution: copy_values(convolution)?,
                matrix: copy_values(matrix)?,
            },
        });
    }
    Ok(result)
}

/// Stage only the fallible KV copies before writing any live state. Recurrent
/// payloads have fixed geometry and restore into their existing allocations;
/// copying a complete replacement would transiently retain another full set of
/// convolution and GDN buffers while live state and checkpoints coexist.
fn restore_state(
    states: &mut [LayerState],
    checkpoint: &[LayerState],
    mut copy_cache: impl FnMut(&KvCache) -> Result<KvCache>,
) -> Result<()> {
    enum Restore<'a> {
        Full {
            keys: &'a mut KvCache,
            values: &'a mut KvCache,
            saved_keys: KvCache,
            saved_values: KvCache,
        },
        Recurrent {
            convolution: &'a mut [f32],
            matrix: &'a mut [f32],
            saved_convolution: &'a [f32],
            saved_matrix: &'a [f32],
        },
    }

    if states.len() != checkpoint.len() {
        return Err(invalid("checkpoint has incompatible layer state"));
    }
    let mut restores = capacity(states.len())?;
    for (state, saved) in states.iter_mut().zip(checkpoint) {
        restores.push(match (state, saved) {
            (
                LayerState::Full { keys, values },
                LayerState::Full {
                    keys: saved_keys,
                    values: saved_values,
                },
            ) => Restore::Full {
                keys,
                values,
                saved_keys: copy_cache(saved_keys)?,
                saved_values: copy_cache(saved_values)?,
            },
            (
                LayerState::Recurrent {
                    convolution,
                    matrix,
                },
                LayerState::Recurrent {
                    convolution: saved_convolution,
                    matrix: saved_matrix,
                },
            ) if convolution.len() == saved_convolution.len()
                && matrix.len() == saved_matrix.len() =>
            {
                Restore::Recurrent {
                    convolution,
                    matrix,
                    saved_convolution,
                    saved_matrix,
                }
            }
            _ => return Err(invalid("checkpoint has incompatible layer state")),
        });
    }
    // All allocation, shape checks and fallible work completed above. Applying
    // the prepared writes cannot fail, so a staging error leaves live state
    // unchanged, including recurrent buffers preceding the failed KV copy.
    for restore in restores {
        match restore {
            Restore::Full {
                keys,
                values,
                saved_keys,
                saved_values,
            } => {
                *keys = saved_keys;
                *values = saved_values;
            }
            Restore::Recurrent {
                convolution,
                matrix,
                saved_convolution,
                saved_matrix,
            } => {
                convolution.copy_from_slice(saved_convolution);
                matrix.copy_from_slice(saved_matrix);
            }
        }
    }
    Ok(())
}

fn small_float(value: usize) -> Result<f32> {
    Ok(f32::from(u16::try_from(value)?))
}

fn finite(values: &[f32], operation: &str) -> Result<()> {
    if !is_finite_slice(values) {
        return Err(invalid(format!("nonfinite values in {operation}")));
    }
    Ok(())
}

fn dot(first: &[f32], second: &[f32]) -> f32 {
    float_dot(first, second)
}

fn dot_f16(first: &[f32], second: &[u16]) -> f32 {
    f16_dot(first, second)
}

fn l2_norm(input: &mut [f32], epsilon: f32) {
    l2_norm_in_place(input, epsilon);
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn add(target: &mut [f32], other: &[f32]) {
    add_slice_in_place(target, other);
}

fn add_finite(target: &mut [f32], other: &[f32], operation: &str) -> Result<()> {
    if !add_slice_in_place_all_finite(target, other) {
        return Err(invalid(format!("nonfinite values in {operation}")));
    }
    Ok(())
}

#[expect(
    clippy::cast_precision_loss,
    clippy::as_conversions,
    reason = "Bounded token positions are converted to the same binary32 RoPE input as Prism."
)]
fn rope_frequencies(
    geometry: &Geometry,
    position: usize,
    frequencies: &mut Vec<(f32, f32)>,
) -> Result<()> {
    let theta_scale = geometry.rope_base.powf(-2.0 / small_float(geometry.rope)?);
    prepare(frequencies, geometry.rope / 2)?;
    let mut theta = position as f32;
    for index in 0..geometry.rope / 2 {
        // Text positions are [p,p,p,0]. Interleaved MRoPE falls back to the
        // fourth (zero) position for sectors outside their declared section.
        let angle = if index < 3 * geometry.rope_sections[index % 3] {
            theta
        } else {
            0.0
        };
        frequencies.push((angle.cos(), angle.sin()));
        theta *= theta_scale;
    }
    Ok(())
}

#[cfg(test)]
#[expect(
    clippy::suboptimal_flops,
    reason = "Preserve separate multiply/add rounding in the scalar RoPE reference."
)]
fn rope(input: &mut [f32], geometry: &Geometry, frequencies: &[(f32, f32)]) -> Result<()> {
    for head in input.chunks_exact_mut(geometry.head) {
        for (index, &(cosine, sine)) in frequencies.iter().enumerate() {
            let first = head[index];
            let second = head[index + geometry.rope / 2];
            head[index] = first * cosine - second * sine;
            head[index + geometry.rope / 2] = first * sine + second * cosine;
        }
    }
    finite(input, "RoPE output")
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::io::{Cursor, Read};
    use std::sync::Arc;

    use sha2::{Digest, Sha256};

    use super::{
        Arithmetic, Geometry, KV_PAGE_TOKENS, KvCache, LayerState, MAX_STATE_BYTES,
        ProjectionScratch, Transforms, append_decayed_head, copy_state, dot, dot_f16,
        half_to_float, restore_state, rope, rope_frequencies,
    };
    use crate::Result;
    use crate::kernels::half_bits;
    use crate::model::Tensor;

    fn read_gdn_word(input: &mut Cursor<&[u8]>) -> Result<u32> {
        let mut bytes = [0; 4];
        input.read_exact(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_gdn_vector(input: &mut Cursor<&[u8]>, width: usize) -> Result<Vec<f32>> {
        (0..width)
            .map(|_| read_gdn_word(input).map(f32::from_bits))
            .collect()
    }

    #[test]
    fn pinned_gdn_arithmetic_matches_independent_native_capture() -> Result<()> {
        // Expected buffers come from the pinned DLL's one-head D128 GDN graph,
        // g=0, beta=1, raw gates disabled. Keep the independent native rounding
        // oracle rather than manufacturing expected values from Rust formulas.
        let bytes = include_bytes!("../tests/fixtures/swift-gdn-parity.bin");
        let provenance: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/swift-gdn-parity.json"))?;
        let hash: String = Sha256::digest(bytes)
            .into_iter()
            .flat_map(|byte| [byte >> 4, byte & 15])
            .map(|nibble| char::from(b"0123456789abcdef"[usize::from(nibble)]))
            .collect();
        assert_eq!(Some(hash.as_str()), provenance["fixture_sha256"].as_str());
        assert_eq!(
            provenance["runtime_commit"].as_str(),
            Some("73bfe6cc485554822f80c473a3c0c4f25e80f41b")
        );
        assert_eq!(
            provenance["ggml_cpu_sha256"].as_str(),
            Some("701b354a38b25e5c6dc57a1559f71cd26b2d0f8d69c1576a0e77a59a4273d2ab")
        );
        let mut input = Cursor::new(bytes.as_slice());
        let mut header = [0; 8];
        input.read_exact(&mut header)?;
        assert_eq!(&header, b"PGDN001\0");
        let dots = read_gdn_word(&mut input)?;
        let rows = read_gdn_word(&mut input)?;
        assert_eq!((dots, rows), (1, 2));
        for _ in 0..dots {
            let width = usize::try_from(read_gdn_word(&mut input)?)?;
            assert_eq!(width, 128);
            let native = read_gdn_word(&mut input)?;
            let first = read_gdn_vector(&mut input, width)?;
            let second = read_gdn_vector(&mut input, width)?;
            assert_eq!(
                Arithmetic::PinnedSwift.gdn_dot(&first, &second)?.to_bits(),
                native
            );
            assert_ne!(
                Arithmetic::Source.gdn_dot(&first, &second)?.to_bits(),
                native
            );
        }
        for _ in 0..rows {
            let width = usize::try_from(read_gdn_word(&mut input)?)?;
            assert_eq!(width, 128);
            let value = f32::from_bits(read_gdn_word(&mut input)?);
            let native = read_gdn_word(&mut input)?;
            let query = read_gdn_vector(&mut input, width)?;
            let key = read_gdn_vector(&mut input, width)?;
            let mut state = read_gdn_vector(&mut input, width)?;
            let expected = read_gdn_vector(&mut input, width)?;
            let mut source = state.clone();
            let delta = value - Arithmetic::PinnedSwift.gdn_dot(&state, &key)?;
            Arithmetic::PinnedSwift.gdn_add_scaled(&mut state, &key, delta)?;
            let scale = 1.0 / 128.0_f32.sqrt();
            assert_eq!(
                (Arithmetic::PinnedSwift.gdn_dot(&state, &query)? * scale).to_bits(),
                native
            );
            assert_eq!(
                state
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
            let delta = value - Arithmetic::Source.gdn_dot(&source, &key)?;
            Arithmetic::Source.gdn_add_scaled(&mut source, &key, delta)?;
            assert_ne!(
                (Arithmetic::Source.gdn_dot(&source, &query)? * scale).to_bits(),
                native
            );
        }
        assert_eq!(input.position(), u64::try_from(bytes.len())?);
        Ok(())
    }

    #[test]
    fn decayed_heads_preserve_bits_and_independent_state_across_scratch_reuse() -> Result<()> {
        let values = [
            0.0,
            -0.0,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::from_bits(1),
            -f32::from_bits(1),
            f32::from_bits(3),
            -f32::from_bits(3),
            f32::MAX,
            -f32::MAX,
            f32::from_bits(0x3f80_0001),
            -f32::from_bits(0x3f80_0001),
        ];
        let previous = values.repeat(3);
        let original: Vec<_> = previous.iter().map(|value| value.to_bits()).collect();
        let mut scratch = vec![f32::NAN; previous.len() + 7];
        let allocation = scratch.as_ptr();
        for decays in [[1.0, 0.5, 0.0], [0.0, 1.0, 0.5]] {
            // Keep the prior copy-then-scale algorithm as a bitwise reference,
            // with different decays per head and a partly filled old workspace.
            let mut reference = previous.clone();
            for (head, decay) in reference.chunks_exact_mut(values.len()).zip(decays) {
                for element in head {
                    *element *= decay;
                }
            }
            super::prepare(&mut scratch, previous.len())?;
            for (head, decay) in previous.chunks_exact(values.len()).zip(decays) {
                let staged = append_decayed_head(head, decay, &mut scratch);
                assert_eq!(staged.len(), values.len());
            }
            assert_eq!(scratch.as_ptr(), allocation);
            assert_eq!(
                scratch
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                reference
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                previous
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                original
            );
        }
        // A later state-validation error may leave provisional scratch dirty;
        // the immutable committed matrix and a subsequent retry remain exact.
        super::prepare(&mut scratch, previous.len())?;
        append_decayed_head(&previous, f32::MAX, &mut scratch);
        assert!(super::finite(&scratch, "provisional GDN state").is_err());
        assert_eq!(
            previous
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            original
        );
        super::prepare(&mut scratch, previous.len())?;
        append_decayed_head(&previous, 1.0, &mut scratch);
        assert_eq!(
            scratch
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            original
        );
        assert_eq!(scratch.as_ptr(), allocation);
        Ok(())
    }

    fn projection_geometry(width: usize) -> Geometry {
        Geometry {
            embedding: width,
            feed_forward: width,
            heads: 1,
            kv_heads: 1,
            head: width,
            rope: width,
            rope_sections: [width / 2, 0, 0, 0],
            rope_base: 10_000.0,
            epsilon: 0.000_001,
            ssm_head: width / 4,
            ssm_keys: 2,
            ssm_values: 4,
            convolution: 1,
        }
    }

    fn projection_weight(name: &str, kind: u32, width: usize) -> Tensor {
        let data: Vec<u8> = match kind {
            142 => {
                let mut block = [0x66; 34];
                block[..2].copy_from_slice(&0x3555_u16.to_le_bytes());
                block.repeat(width / 128)
            }
            8 => {
                let mut block = [1; 34];
                block[..2].copy_from_slice(&0x3555_u16.to_le_bytes());
                block.repeat(width / 32)
            }
            _ => (0..width)
                .flat_map(|index| {
                    [0.125_f32, -0.25, 0.5, -1.0, 2.0, -4.0, 0.0, -0.0][index % 8].to_le_bytes()
                })
                .collect(),
        };
        Tensor {
            name: name.to_owned(),
            dimensions: vec![width],
            kind,
            data: Arc::from(data),
        }
    }

    #[test]
    fn grouped_projection_shares_only_matching_basis_permutation_and_quantizer() -> Result<()> {
        for width in [8, 256] {
            let geometry = projection_geometry(width);
            let kind = if width == 256 { 142 } else { 0 };
            let weights = [
                projection_weight("up.weight", kind, width),
                projection_weight("gate.weight", kind, width),
                projection_weight("blk.0.ssm_out.weight", kind, width),
                projection_weight("plain.weight", kind, width),
            ];
            let mut transforms = Transforms {
                block: 4,
                weights: weights[..3]
                    .iter()
                    .map(|weight| weight.name.clone())
                    .collect::<BTreeSet<_>>(),
                inverse_embedding: false,
                grouped_values: true,
                signs: BTreeMap::from([(
                    width,
                    (0..width)
                        .map(|index| if index % 3 == 0 { -1.0 } else { 1.0 })
                        .collect(),
                )]),
            };
            let input: Vec<f32> = (0..width)
                .map(|index| [127.0, 0.5, 1.5, -63.5, 2.5, -0.5, 4.0, -16.0][index % 8])
                .collect();
            let mut expected = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
            let mut independent = ProjectionScratch::default();
            for (weight, output) in weights.iter().zip(&mut expected) {
                transforms.project_into(weight, &input, &geometry, output, &mut independent)?;
            }
            let mut actual = [
                vec![f32::NAN],
                vec![f32::NAN],
                vec![f32::NAN],
                vec![f32::NAN],
            ];
            let mut scratch = ProjectionScratch::default();
            transforms.project_group_into(
                weights.each_ref(),
                &input,
                &geometry,
                actual.each_mut(),
                &mut scratch,
            )?;
            for (actual, expected) in actual.iter().zip(&expected) {
                assert!(
                    actual
                        .iter()
                        .map(|value| value.to_bits())
                        .eq(expected.iter().map(|value| value.to_bits()))
                );
            }
            assert_eq!(scratch.transforms, 2);
            assert_eq!(scratch.matvec.preparation_count(), 3);
            assert_eq!(independent.transforms, 3);
            assert_eq!(independent.matvec.preparation_count(), 4);

            // Same basis is still insufficient for PQ2 Q8_K vs legacy Q8_0.
            if width == 256 {
                transforms.grouped_values = false;
                let legacy = projection_weight("gate.weight", 8, width);
                let before = scratch.matvec.preparation_count();
                let transforms_before = scratch.transforms;
                let [first, second, ..] = &mut actual;
                transforms.project_group_into(
                    [&weights[0], &legacy],
                    &input,
                    &geometry,
                    [first, second],
                    &mut scratch,
                )?;
                assert_eq!(scratch.matvec.preparation_count() - before, 2);
                assert_eq!(scratch.transforms - transforms_before, 2);
                transforms.project_into(
                    &legacy,
                    &input,
                    &geometry,
                    &mut expected[1],
                    &mut independent,
                )?;
                assert!(
                    actual[1]
                        .iter()
                        .map(|value| value.to_bits())
                        .eq(expected[1].iter().map(|value| value.to_bits()))
                );
            }
        }
        Ok(())
    }

    #[test]
    fn grouped_projection_failure_clears_all_siblings_and_reprepares() -> Result<()> {
        let geometry = projection_geometry(8);
        let first = projection_weight("first", 0, 8);
        let mut second = projection_weight("second", 0, 8);
        second.data = Arc::from(f32::NAN.to_le_bytes().repeat(8));
        let transforms = Transforms::default();
        let mut scratch = ProjectionScratch::default();
        let mut output = [vec![1.0], vec![2.0]];
        assert!(
            transforms
                .project_group_into(
                    [&first, &second],
                    &[1.0; 8],
                    &geometry,
                    output.each_mut(),
                    &mut scratch
                )
                .is_err()
        );
        assert!(output.iter().all(Vec::is_empty));
        assert_eq!(scratch.matvec.preparation_count(), 1);
        transforms.project_group_into(
            [&first, &first],
            &[-1.0; 8],
            &geometry,
            output.each_mut(),
            &mut scratch,
        )?;
        assert_eq!(scratch.matvec.preparation_count(), 2);
        assert_eq!(output[0], output[1]);
        assert_eq!(output[0].len(), 1);
        Ok(())
    }

    struct F32OnlyExecutor {
        offloaded_names: BTreeSet<String>,
        allow_group: bool,
        f32_calls: usize,
        f32_group_calls: usize,
    }

    impl crate::kernels::Pq2ProjectionExecutor for F32OnlyExecutor {
        fn identity(&self) -> &'static str {
            "f32-only-test"
        }
        fn supports(&self, _tensor: &Tensor) -> bool {
            false
        }
        fn supports_f32(&self, tensor: &Tensor) -> bool {
            self.offloaded_names.contains(&tensor.name)
        }
        fn supports_f32_group(&self, tensors: &[&Tensor]) -> bool {
            self.allow_group
                && matches!(tensors.len(), 2 | 3)
                && tensors.iter().all(|tensor| self.supports_f32(tensor))
        }
        fn project(
            &mut self,
            _tensor: &Tensor,
            _activation: &[crate::kernels::Q8KBlock],
            _pinned_accumulation: bool,
            _output: &mut Vec<f32>,
        ) -> Result<()> {
            Err(crate::invalid("unexpected Q8_K executor call"))
        }
        fn project_f32(
            &mut self,
            tensor: &Tensor,
            activation: &[f32],
            _pinned_accumulation: bool,
            output: &mut Vec<f32>,
        ) -> Result<()> {
            self.f32_calls += 1;
            let mut cpu_scratch = crate::kernels::MatvecScratch::default();
            crate::kernels::matvec_into(tensor, activation, output, &mut cpu_scratch)
        }
        fn project_f32_group(
            &mut self,
            tensors: &[&Tensor],
            activation: &[f32],
            _pinned_accumulation: bool,
            outputs: &mut [&mut Vec<f32>],
        ) -> Result<()> {
            self.f32_group_calls += 1;
            let mut cpu_scratch = crate::kernels::MatvecScratch::default();
            for (tensor, output) in tensors.iter().zip(outputs.iter_mut()) {
                crate::kernels::matvec_into(tensor, activation, output, &mut cpu_scratch)?;
            }
            Ok(())
        }
    }

    #[test]
    fn grouped_projection_skips_cpu_q8k_quantization_when_all_pq2_siblings_use_f32_executor()
    -> Result<()> {
        let width = 256;
        let geometry = projection_geometry(width);
        let qkv = projection_weight("blk.0.attn_qkv.weight", 142, width);
        let gate = projection_weight("blk.0.attn_gate.weight", 142, width);
        let bf16 = |name: &str, bits: u16| Tensor {
            name: name.to_owned(),
            dimensions: vec![width],
            kind: 30,
            data: Arc::from(bits.to_le_bytes().repeat(width)),
        };
        let (beta, alpha) = (
            bf16("blk.0.ssm_beta.weight", 0x3f80),
            bf16("blk.0.ssm_alpha.weight", 0x3f00),
        );
        let transforms = Transforms {
            block: 4,
            weights: BTreeSet::from([
                qkv.name.clone(),
                gate.name.clone(),
                beta.name.clone(),
                alpha.name.clone(),
            ]),
            inverse_embedding: false,
            grouped_values: false,
            signs: BTreeMap::from([(width, vec![1.0; width])]),
        };
        let input: Vec<f32> = (0..width)
            .map(|index| [1.0, -0.5, 0.25, -0.125][index % 4])
            .collect();

        // 1. All PQ2 siblings offloaded via supports_f32 + BF16 siblings on CPU:
        // Q8_K quantization is skipped completely while BF16 activations are prepared.
        let mut executor = F32OnlyExecutor {
            offloaded_names: BTreeSet::from([qkv.name.clone(), gate.name.clone()]),
            allow_group: false,
            f32_calls: 0,
            f32_group_calls: 0,
        };
        let mut outputs = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut scratch = ProjectionScratch::default();
        let mut executor_opt: Option<&mut dyn crate::kernels::Pq2ProjectionExecutor> =
            Some(&mut executor);
        transforms.project_group_into_with_executor(
            [&qkv, &gate, &beta, &alpha],
            &input,
            &geometry,
            outputs.each_mut(),
            &mut scratch,
            &mut executor_opt,
        )?;
        assert_eq!(executor.f32_calls, 2);
        assert_eq!(executor.f32_group_calls, 0);
        assert_eq!(scratch.transforms, 2);
        assert_eq!(scratch.matvec.preparation_count(), 2);
        assert_eq!(scratch.matvec.statistics().matrix_successes, 2);

        // 2. Partial PQ2 offload (qkv on f32 executor, gate on CPU):
        // Q8_K quantization is performed once so the CPU sibling succeeds.
        let mut partial_executor = F32OnlyExecutor {
            offloaded_names: BTreeSet::from([qkv.name.clone()]),
            allow_group: true,
            f32_calls: 0,
            f32_group_calls: 0,
        };
        let mut partial_outputs = [Vec::new(), Vec::new()];
        let mut partial_scratch = ProjectionScratch::default();
        let mut partial_opt: Option<&mut dyn crate::kernels::Pq2ProjectionExecutor> =
            Some(&mut partial_executor);
        transforms.project_group_into_with_executor(
            [&qkv, &gate],
            &input,
            &geometry,
            partial_outputs.each_mut(),
            &mut partial_scratch,
            &mut partial_opt,
        )?;
        assert_eq!(partial_executor.f32_calls, 1);
        assert_eq!(partial_executor.f32_group_calls, 0);
        assert_eq!(partial_scratch.matvec.preparation_count(), 1);
        assert_eq!(partial_scratch.matvec.statistics().matrix_successes, 1);
        assert_eq!(partial_outputs, [outputs[0].clone(), outputs[1].clone()]);
        Ok(())
    }

    #[test]
    fn grouped_projection_batches_compatible_pq2_prefix_with_f32_group_executor() -> Result<()> {
        let width = 256;
        let geometry = projection_geometry(width);
        let qkv = projection_weight("blk.0.attn_qkv.weight", 142, width);
        let gate = projection_weight("blk.0.attn_gate.weight", 142, width);
        let bf16 = |name: &str, bits: u16| Tensor {
            name: name.to_owned(),
            dimensions: vec![width],
            kind: 30,
            data: Arc::from(bits.to_le_bytes().repeat(width)),
        };
        let (beta, alpha) = (
            bf16("blk.0.ssm_beta.weight", 0x3f80),
            bf16("blk.0.ssm_alpha.weight", 0x3f00),
        );
        let transforms = Transforms {
            block: 4,
            weights: BTreeSet::from([
                qkv.name.clone(),
                gate.name.clone(),
                beta.name.clone(),
                alpha.name.clone(),
            ]),
            inverse_embedding: false,
            grouped_values: false,
            signs: BTreeMap::from([(width, vec![1.0; width])]),
        };
        let input: Vec<f32> = (0..width)
            .map(|index| [1.0, -0.5, 0.25, -0.125][index % 4])
            .collect();
        let mut expected = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut cpu_scratch = ProjectionScratch::default();
        transforms.project_group_into(
            [&qkv, &gate, &beta, &alpha],
            &input,
            &geometry,
            expected.each_mut(),
            &mut cpu_scratch,
        )?;

        let mut grouped_executor = F32OnlyExecutor {
            offloaded_names: BTreeSet::from([qkv.name.clone(), gate.name.clone()]),
            allow_group: true,
            f32_calls: 0,
            f32_group_calls: 0,
        };
        let mut grouped_outputs = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut grouped_scratch = ProjectionScratch::default();
        let mut grouped_opt: Option<&mut dyn crate::kernels::Pq2ProjectionExecutor> =
            Some(&mut grouped_executor);
        transforms.project_group_into_with_executor(
            [&qkv, &gate, &beta, &alpha],
            &input,
            &geometry,
            grouped_outputs.each_mut(),
            &mut grouped_scratch,
            &mut grouped_opt,
        )?;
        assert_eq!(grouped_executor.f32_calls, 0);
        assert_eq!(grouped_executor.f32_group_calls, 1);
        assert_eq!(grouped_scratch.transforms, 2);
        assert_eq!(grouped_scratch.matvec.preparation_count(), 2);
        assert_eq!(grouped_scratch.matvec.statistics().matrix_successes, 2);
        assert_eq!(grouped_outputs, expected);
        Ok(())
    }

    fn assert_state_bits_eq(expected: &[LayerState], actual: &[LayerState]) {
        assert_eq!(expected.len(), actual.len());
        for (expected, actual) in expected.iter().zip(actual) {
            match (expected, actual) {
                (
                    LayerState::Full { keys, values },
                    LayerState::Full {
                        keys: actual_keys,
                        values: actual_values,
                    },
                ) => {
                    assert!(keys.tokens().eq(actual_keys.tokens()));
                    assert!(values.tokens().eq(actual_values.tokens()));
                }
                (
                    LayerState::Recurrent {
                        convolution,
                        matrix,
                    },
                    LayerState::Recurrent {
                        convolution: actual_convolution,
                        matrix: actual_matrix,
                    },
                ) => {
                    assert!(
                        convolution
                            .iter()
                            .map(|value| value.to_bits())
                            .eq(actual_convolution.iter().map(|value| value.to_bits()))
                    );
                    assert!(
                        matrix
                            .iter()
                            .map(|value| value.to_bits())
                            .eq(actual_matrix.iter().map(|value| value.to_bits()))
                    );
                }
                _ => assert_eq!(
                    std::mem::discriminant(expected),
                    std::mem::discriminant(actual)
                ),
            }
        }
    }

    #[test]
    fn packed_cache_roundtrip_preserves_every_finite_binary16_value() {
        for bits in 0..=u16::MAX {
            if bits & 0x7c00 != 0x7c00 {
                assert_eq!(half_bits(half_to_float(bits)), bits);
            }
        }
    }

    #[test]
    fn packed_cache_dot_preserves_product_rounding_and_wide_accumulation() {
        // Promoting operands before multiplication changes the first result;
        // accumulating products in f32 loses the middle unit in the second.
        assert_eq!(
            dot_f16(&[f32::from_bits(0x3f80_0001), -1.0], &[0x3c01; 2]).to_bits(),
            0x3400_0000
        );
        assert_eq!(
            dot_f16(&[16_777_216.0, 1.0, -16_777_216.0], &[0x3c00; 3]).to_bits(),
            1.0_f32.to_bits()
        );
        let bits = [0x8000, 0x0001, 0x03ff, 0x0400, 0x7bff, 0xfbff];
        let expanded = bits.map(half_to_float);
        let query = [-0.0, 0.25, -0.5, 1.0, 0.5, -0.5];
        assert_eq!(
            dot_f16(&query, &bits).to_bits(),
            dot(&query, &expanded).to_bits()
        );
    }

    fn check_original_attention_gates(
        geometry: &Geometry,
        layer: &super::FullAttention,
        projected: &[f32],
        original_values: [f32; 8],
        expected: &[f32],
        arithmetic: Arithmetic,
        workers: usize,
    ) -> Result<()> {
        let mut projection = ProjectionScratch::default();
        projection
            .matvec
            .set_row_parallelism(crate::kernels::RowParallelism::new(workers)?);
        let transforms = Transforms::default();
        let mut adapter = crate::model::lora::Scratch::default();
        let mut evaluation = super::Evaluation {
            geometry,
            arithmetic,
            transforms: &transforms,
            position: 0,
            frequencies_ready: false,
            projection: &mut projection,
            adapter: None,
            adapter_scale: 0.0,
            adapter_scratch: &mut adapter,
            executor: None,
        };
        let keys = KvCache::new(8, 2)?;
        let values = KvCache::new(8, 2)?;
        let mut scratch = super::AttentionScratch::default();
        let mut key = [0.0; 8];
        let mut value = original_values;
        // A failed rounding attempt must not publish attended output
        // or append state; reuse the same scratch for the valid retry.
        value[0] = 70_000.0;
        assert!(
            evaluation
                .full_attention_state(
                    layer,
                    projected,
                    &mut key,
                    &mut value,
                    &keys,
                    &values,
                    &mut scratch.query,
                    &mut scratch.scores,
                    &mut scratch.frequencies,
                    &mut scratch.attended,
                )
                .is_err()
        );
        assert_eq!(scratch.attended, [] as [f32; 0]);
        assert_eq!(scratch.gate, [] as [f32; 0]);
        assert_eq!(keys.tokens().count(), 0);
        assert_eq!(values.tokens().count(), 0);
        let mut value = original_values;
        evaluation.full_attention_state(
            layer,
            projected,
            &mut key,
            &mut value,
            &keys,
            &values,
            &mut scratch.query,
            &mut scratch.scores,
            &mut scratch.frequencies,
            &mut scratch.attended,
        )?;
        assert_eq!(scratch.gate, [] as [f32; 0]);
        assert!(
            scratch
                .attended
                .iter()
                .map(|value| value.to_bits())
                .eq(expected.iter().map(|value| value.to_bits()))
        );
        Ok(())
    }

    #[test]
    fn full_attention_reads_original_interleaved_gates_across_worker_groups() -> Result<()> {
        let mut geometry = projection_geometry(4);
        geometry.heads = 12;
        geometry.kv_heads = 2;
        let layer = super::FullAttention {
            query: projection_weight("q", 0, 4),
            key: projection_weight("k", 0, 4),
            value: projection_weight("v", 0, 4),
            output: projection_weight("out", 0, 4),
            query_norm: vec![1.0; 4],
            key_norm: vec![1.0; 4],
        };
        let mut projected = vec![0.0_f32; geometry.query_gate_width()];
        let mut expected_gates = Vec::new();
        for (head, pair) in projected.as_chunks_mut::<8>().0.iter_mut().enumerate() {
            for (channel, gate) in pair[4..].iter_mut().enumerate() {
                *gate = [-4.0, -1.0, 0.0, 1.0, 4.0][(head + channel) % 5];
                expected_gates.push(*gate);
            }
        }
        let original_values = [0.25, -0.5, 1.0, 2.0, -0.25, 0.5, -1.0, -2.0];
        // With zero Q/K and no prior tokens, attention has exactly unit
        // probability. Independently apply each head's known gate to its KV
        // head's rounded values; this exercises both 4-head and tail workers.
        let expected: Vec<f32> = expected_gates
            .iter()
            .enumerate()
            .map(|(index, &gate)| {
                let head = index / 4;
                let kv = head / (geometry.heads / geometry.kv_heads);
                let value = half_to_float(half_bits(original_values[kv * 4 + index % 4]));
                value * (1.0 / (1.0 + (-gate).exp()))
            })
            .collect();
        #[cfg(feature = "parallel-rows")]
        let worker_counts = [1, 2, 4];
        #[cfg(not(feature = "parallel-rows"))]
        let worker_counts = [1];
        for arithmetic in [Arithmetic::Source, Arithmetic::PinnedSwift] {
            for workers in worker_counts {
                check_original_attention_gates(
                    &geometry,
                    &layer,
                    &projected,
                    original_values,
                    &expected,
                    arithmetic,
                    workers,
                )?;
            }
        }
        Ok(())
    }

    #[test]
    fn paged_cache_copy_shares_only_sealed_payload_and_copies_exact_recurrent_bits() -> Result<()> {
        let bits = [0x8000, 0x0001, 0x03ff, 0x0400, 0x7bff, 0xfbff];
        let values = bits.map(half_to_float);
        let mut cache = KvCache::new(bits.len(), 3 * KV_PAGE_TOKENS)?;
        for _ in 0..=KV_PAGE_TOKENS {
            cache.reserve_append()?;
            cache.append(&values);
        }
        assert_eq!(cache.pages.len(), 1);
        assert_eq!(cache.tail, bits);
        let mut states = vec![
            LayerState::Full {
                keys: cache.copy()?,
                values: cache,
            },
            LayerState::Recurrent {
                convolution: vec![-0.0, f32::from_bits(1)],
                matrix: vec![f32::MIN_POSITIVE, f32::MAX],
            },
        ];
        let checkpoint = copy_state(&states)?;
        let expected = copy_state(&states)?;
        assert_state_bits_eq(&states, &checkpoint);
        if let (
            LayerState::Full { keys, values },
            LayerState::Full {
                keys: saved_keys,
                values: saved_values,
            },
        ) = (&mut states[0], &checkpoint[0])
        {
            assert!(Arc::ptr_eq(&keys.pages[0], &saved_keys.pages[0]));
            assert!(Arc::ptr_eq(&values.pages[0], &saved_values.pages[0]));
            assert_ne!(keys.tail.as_ptr(), saved_keys.tail.as_ptr());
            assert_ne!(values.tail.as_ptr(), saved_values.tail.as_ptr());
            // Each snapshot charges all shared payloads conservatively.
            assert_eq!(keys.bytes(), saved_keys.bytes());
            assert!(keys.bytes() >= (KV_PAGE_TOKENS + 1) * bits.len() * size_of::<u16>());
            keys.tail[0] = 0;
            values.tail[1] = 0;
            keys.clear();
        }
        if let LayerState::Recurrent {
            convolution,
            matrix,
        } = &mut states[1]
        {
            convolution[0] = 1.0;
            matrix[0] = 0.0;
        }
        assert_state_bits_eq(&expected, &checkpoint);
        Ok(())
    }

    fn restore_fixture() -> Result<Vec<LayerState>> {
        let mut cache = KvCache::new(2, 3 * KV_PAGE_TOKENS)?;
        for _ in 0..=KV_PAGE_TOKENS {
            cache.reserve_append()?;
            cache.append(&[-0.0, half_to_float(1)]);
        }
        Ok(vec![
            LayerState::Recurrent {
                convolution: vec![-0.0, f32::from_bits(1)],
                matrix: vec![f32::MIN_POSITIVE, f32::MAX],
            },
            LayerState::Full {
                keys: cache.copy()?,
                values: cache,
            },
        ])
    }

    #[test]
    fn checkpoint_restore_reuses_recurrent_storage_and_preserves_independent_bits() -> Result<()> {
        let mut states = restore_fixture()?;
        let checkpoint = copy_state(&states)?;
        let expected = copy_state(&states)?;
        let mut storage = Vec::new();
        for state in &mut states {
            match state {
                LayerState::Recurrent {
                    convolution,
                    matrix,
                } => {
                    storage.push((convolution.as_ptr(), convolution.capacity()));
                    storage.push((matrix.as_ptr(), matrix.capacity()));
                    convolution.fill(1.0);
                    matrix.fill(2.0);
                }
                LayerState::Full { keys, values } => {
                    keys.clear();
                    values.clear();
                }
            }
        }
        restore_state(&mut states, &checkpoint, KvCache::copy)?;
        assert_state_bits_eq(&expected, &states);
        assert_state_bits_eq(&expected, &checkpoint);
        if let LayerState::Recurrent {
            convolution,
            matrix,
        } = &mut states[0]
        {
            assert_eq!(storage[0], (convolution.as_ptr(), convolution.capacity()));
            assert_eq!(storage[1], (matrix.as_ptr(), matrix.capacity()));
            convolution[0] = 3.0;
            matrix[0] = 4.0;
        }
        if let (
            LayerState::Full { keys, values },
            LayerState::Full {
                keys: saved_keys,
                values: saved_values,
            },
        ) = (&mut states[1], &checkpoint[1])
        {
            assert!(Arc::ptr_eq(&keys.pages[0], &saved_keys.pages[0]));
            assert!(Arc::ptr_eq(&values.pages[0], &saved_values.pages[0]));
            assert_ne!(keys.tail.as_ptr(), saved_keys.tail.as_ptr());
            assert_ne!(values.tail.as_ptr(), saved_values.tail.as_ptr());
            keys.tail[0] = 0;
            values.tail[0] = 0;
        }
        assert_state_bits_eq(&expected, &checkpoint);
        restore_state(&mut states, &checkpoint, KvCache::copy)?;
        assert_state_bits_eq(&expected, &states);
        Ok(())
    }

    #[test]
    fn checkpoint_restore_staging_failure_leaves_all_live_payloads_unchanged() -> Result<()> {
        let mut states = restore_fixture()?;
        let mut checkpoint = copy_state(&states)?;
        if let LayerState::Recurrent {
            convolution,
            matrix,
        } = &mut checkpoint[0]
        {
            convolution.fill(1.0);
            matrix.fill(2.0);
        }
        let expected = copy_state(&states)?;
        let mut copies = 0;
        let result = restore_state(&mut states, &checkpoint, |cache| {
            copies += 1;
            if copies == 2 {
                return Err(std::io::Error::other("injected KV staging allocation failure").into());
            }
            cache.copy()
        });
        assert!(result.is_err());
        assert_eq!(copies, 2);
        assert_state_bits_eq(&expected, &states);

        // A malformed later layer likewise cannot partially restore the earlier
        // recurrent payload, even though that payload needs no allocation.
        checkpoint[1] = LayerState::Recurrent {
            convolution: vec![0.0],
            matrix: vec![0.0],
        };
        assert!(restore_state(&mut states, &checkpoint, KvCache::copy).is_err());
        assert_state_bits_eq(&expected, &states);
        assert!(restore_state(&mut states, &checkpoint[..1], KvCache::copy).is_err());
        assert_state_bits_eq(&expected, &states);

        // Complete KV staging before discovering a later recurrent size
        // mismatch; neither the earlier cache nor any recurrent bits change.
        states.swap(0, 1);
        let expected = copy_state(&states)?;
        let mut checkpoint = copy_state(&states)?;
        if let LayerState::Full { keys, values } = &mut checkpoint[0] {
            keys.tail[0] = 0x3c00;
            values.tail[0] = 0x3c00;
        }
        if let LayerState::Recurrent { convolution, .. } = &mut checkpoint[1] {
            convolution.push(0.0);
        }
        assert!(restore_state(&mut states, &checkpoint, KvCache::copy).is_err());
        assert_state_bits_eq(&expected, &states);
        Ok(())
    }

    #[test]
    fn text_imrope_uses_zero_fourth_position_for_unbalanced_sections() -> Result<()> {
        let geometry = Geometry {
            embedding: 12,
            feed_forward: 12,
            heads: 1,
            kv_heads: 1,
            head: 12,
            rope: 12,
            rope_sections: [4, 1, 1, 0],
            rope_base: 10_000.0,
            epsilon: 0.000_001,
            ssm_head: 1,
            ssm_keys: 1,
            ssm_values: 1,
            convolution: 1,
        };
        let mut input = [0.0; 12];
        input[0] = 1.0;
        input[4] = 1.0;
        input[10] = 2.0;
        let mut frequencies = Vec::new();
        rope_frequencies(&geometry, 3, &mut frequencies)?;
        rope(&mut input, &geometry, &frequencies)?;
        // Sector four is outside the height section, so [p,p,p,0] selects 0.
        assert_eq!(input[4].to_bits(), 1.0_f32.to_bits());
        assert_eq!(input[10].to_bits(), 2.0_f32.to_bits());
        assert!((input[0] - 3.0_f32.cos()).abs() < 0.000_001);
        assert!((input[6] - 3.0_f32.sin()).abs() < 0.000_001);
        Ok(())
    }

    #[derive(serde::Deserialize)]
    struct RopeCapture {
        runtime_commit: String,
        ggml_cpu_sha256: String,
        head: usize,
        rotary_dimensions: usize,
        sections: [usize; 4],
        freq_base_bits: u32,
        input_bits: Vec<u32>,
        cases: Vec<RopeCase>,
    }

    #[derive(serde::Deserialize)]
    struct RopeCase {
        position: usize,
        native_bits: Vec<u32>,
    }

    #[derive(serde::Deserialize)]
    struct RopeCoefficients {
        runtime_commit: String,
        runtime_dll_sha256: BTreeMap<String, String>,
        cases: Vec<RopeCoefficientCase>,
    }

    #[derive(serde::Deserialize)]
    struct RopeCoefficientCase {
        position: usize,
        cosine_bits: Vec<u32>,
        sine_bits: Vec<u32>,
    }

    fn pinned_rope_captures() -> Result<(RopeCapture, RopeCoefficients)> {
        let capture: RopeCapture =
            serde_json::from_str(include_str!("../tests/fixtures/swift-attention-rope.json"))?;
        let coefficients = include_str!("../tests/fixtures/swift-rope-basis.json");
        // Git may check text fixtures out with CRLF. Pin the captured contents
        // with LF normalization while preserving every numeric oracle word.
        let normalized = coefficients.replace("\r\n", "\n");
        let hash: String = Sha256::digest(normalized.as_bytes())
            .into_iter()
            .flat_map(|byte| [byte >> 4, byte & 15])
            .map(|nibble| char::from(b"0123456789abcdef"[usize::from(nibble)]))
            .collect();
        assert_eq!(
            hash,
            "e58b3c96b6ce704361d7a397775817959793e2a6cbaa78598e8a07075548ae93"
        );
        let coefficients: RopeCoefficients = serde_json::from_str(coefficients)?;
        assert_eq!(
            capture.runtime_commit,
            "73bfe6cc485554822f80c473a3c0c4f25e80f41b"
        );
        assert_eq!(
            capture.ggml_cpu_sha256,
            "701b354a38b25e5c6dc57a1559f71cd26b2d0f8d69c1576a0e77a59a4273d2ab"
        );
        assert_eq!(coefficients.runtime_commit, capture.runtime_commit);
        assert_eq!(
            coefficients.runtime_dll_sha256.get("ggml-cpu.dll"),
            Some(&capture.ggml_cpu_sha256)
        );
        assert_eq!(capture.cases.len(), coefficients.cases.len());
        Ok((capture, coefficients))
    }

    fn captured_rope_geometry(capture: &RopeCapture) -> Geometry {
        let mut geometry = projection_geometry(capture.head);
        geometry.rope = capture.rotary_dimensions;
        geometry.rope_sections = capture.sections;
        geometry.rope_base = f32::from_bits(capture.freq_base_bits);
        geometry
    }

    #[test]
    fn pinned_attention_out_of_place_rope_matches_native_and_keeps_source_path() -> Result<()> {
        let (capture, coefficients) = pinned_rope_captures()?;
        let geometry = captured_rope_geometry(&capture);
        let input = capture
            .input_bits
            .iter()
            .copied()
            .map(f32::from_bits)
            .collect::<Vec<_>>();
        for (case, coefficient) in capture.cases.iter().zip(&coefficients.cases) {
            assert_eq!(case.position, coefficient.position);
            assert_eq!(coefficient.cosine_bits.len(), geometry.rope / 2);
            assert_eq!(coefficient.sine_bits.len(), geometry.rope / 2);
            // These are fresh native basis-vector outputs, independent of the
            // host's powf/sin/cos implementations. The full native output oracle
            // remains bit-exact on every host that checks the FMA arithmetic.
            let frequencies = coefficient
                .cosine_bits
                .iter()
                .zip(&coefficient.sine_bits)
                .map(|(&cosine, &sine)| (f32::from_bits(cosine), f32::from_bits(sine)))
                .collect::<Vec<_>>();
            let mut pinned = input.clone();
            crate::kernels::pinned_attention::rope(
                &mut pinned,
                geometry.head,
                geometry.rope,
                &frequencies,
            )?;
            assert_eq!(
                pinned.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                case.native_bits,
                "position {}",
                case.position
            );
            let mut source = input.clone();
            Arithmetic::Source.attention_rope(&mut source, &geometry, &frequencies)?;
            let mut baseline = input.clone();
            rope(&mut baseline, &geometry, &frequencies)?;
            assert!(
                source
                    .iter()
                    .zip(&baseline)
                    .all(|(source, baseline)| source.to_bits() == baseline.to_bits())
            );
            assert_ne!(
                source.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                case.native_bits
            );
        }
        Ok(())
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"))]
    fn pinned_attention_rope_target_frequencies_and_full_rotation_match_native() -> Result<()> {
        let (capture, coefficients) = pinned_rope_captures()?;
        let geometry = captured_rope_geometry(&capture);
        let input = capture
            .input_bits
            .iter()
            .copied()
            .map(f32::from_bits)
            .collect::<Vec<_>>();
        for (case, coefficient) in capture.cases.iter().zip(&coefficients.cases) {
            assert_eq!(case.position, coefficient.position);
            let mut frequencies = Vec::new();
            rope_frequencies(&geometry, case.position, &mut frequencies)?;
            assert_eq!(
                frequencies
                    .iter()
                    .map(|&(cosine, _)| cosine.to_bits())
                    .collect::<Vec<_>>(),
                coefficient.cosine_bits,
                "native target cosine coefficients at position {}",
                case.position
            );
            assert_eq!(
                frequencies
                    .iter()
                    .map(|&(_, sine)| sine.to_bits())
                    .collect::<Vec<_>>(),
                coefficient.sine_bits,
                "native target sine coefficients at position {}",
                case.position
            );
            let mut pinned = input.clone();
            crate::kernels::pinned_attention::rope(
                &mut pinned,
                geometry.head,
                geometry.rope,
                &frequencies,
            )?;
            assert_eq!(
                pinned.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                case.native_bits,
                "native target full rotation at position {}",
                case.position
            );
        }
        Ok(())
    }

    fn pinned_rope_inplace_capture() -> Result<Cursor<&'static [u8]>> {
        let bytes = include_bytes!("../tests/fixtures/swift-rope-inplace.bin");
        let provenance: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/swift-rope-inplace.json"))?;
        let hash: String = Sha256::digest(bytes)
            .into_iter()
            .flat_map(|byte| [byte >> 4, byte & 15])
            .map(|nibble| char::from(b"0123456789abcdef"[usize::from(nibble)]))
            .collect();
        assert_eq!(Some(hash.as_str()), provenance["fixture_sha256"].as_str());
        assert_eq!(
            hash,
            "efda5a7620adad6ac1735357bb20d6ea8663092164d9a3d221afa2822226fbc1"
        );
        assert_eq!(
            provenance["runtime_commit"].as_str(),
            Some("73bfe6cc485554822f80c473a3c0c4f25e80f41b")
        );
        assert_eq!(
            provenance["runtime_dll_sha256"]["ggml-cpu.dll"].as_str(),
            Some("701b354a38b25e5c6dc57a1559f71cd26b2d0f8d69c1576a0e77a59a4273d2ab")
        );
        let mut input = Cursor::new(bytes.as_slice());
        let mut magic = [0; 8];
        input.read_exact(&mut magic)?;
        assert_eq!(&magic, b"SWROPIP1");
        assert_eq!(read_gdn_word(&mut input)?, 3);
        Ok(input)
    }

    #[test]
    fn pinned_attention_inplace_rope_matches_native_and_fp16_boundary() -> Result<()> {
        let (capture, coefficients) = pinned_rope_captures()?;
        let geometry = captured_rope_geometry(&capture);
        let original = capture
            .input_bits
            .iter()
            .copied()
            .map(f32::from_bits)
            .collect::<Vec<_>>();
        let mut input = pinned_rope_inplace_capture()?;
        for coefficient in &coefficients.cases {
            assert_eq!(
                usize::try_from(read_gdn_word(&mut input)?)?,
                coefficient.position
            );
            let native = read_gdn_vector(&mut input, 64)?;
            let frequencies = coefficient
                .cosine_bits
                .iter()
                .zip(&coefficient.sine_bits)
                .map(|(&cosine, &sine)| (f32::from_bits(cosine), f32::from_bits(sine)))
                .collect::<Vec<_>>();
            let mut actual = original.clone();
            Arithmetic::PinnedSwift.attention_rope(&mut actual, &geometry, &frequencies)?;
            let mut expected = original.clone();
            expected[..64].copy_from_slice(&native);
            assert_eq!(
                actual.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                expected.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                "in-place position {}",
                coefficient.position
            );
        }
        assert_eq!(read_gdn_word(&mut input)?, 42);
        let pair = usize::try_from(read_gdn_word(&mut input)?)?;
        assert_eq!(pair, 19);
        let first = f32::from_bits(read_gdn_word(&mut input)?);
        let second = f32::from_bits(read_gdn_word(&mut input)?);
        let cosine = f32::from_bits(read_gdn_word(&mut input)?);
        let sine = f32::from_bits(read_gdn_word(&mut input)?);
        let native = [read_gdn_word(&mut input)?, read_gdn_word(&mut input)?];
        let separate = [read_gdn_word(&mut input)?, read_gdn_word(&mut input)?];
        let native_half = read_gdn_word(&mut input)?;
        let mut frequencies = vec![(1.0, 0.0); 32];
        frequencies[pair] = (cosine, sine);
        let mut actual = vec![0.0; 256];
        actual[pair] = first;
        actual[pair + 32] = second;
        let mut out_of_place = actual.clone();
        Arithmetic::PinnedSwift.attention_rope(&mut actual, &geometry, &frequencies)?;
        crate::kernels::pinned_attention::rope(&mut out_of_place, 256, 64, &frequencies)?;
        assert_eq!(
            [actual[pair].to_bits(), actual[pair + 32].to_bits()],
            native
        );
        assert_eq!(
            [
                out_of_place[pair].to_bits(),
                out_of_place[pair + 32].to_bits()
            ],
            separate
        );
        assert_eq!(
            u32::from(half_bits(actual[pair])) | (u32::from(half_bits(actual[pair + 32])) << 16),
            native_half
        );
        assert_ne!(half_bits(actual[pair]), half_bits(out_of_place[pair]));
        assert_eq!(input.position(), u64::try_from(input.get_ref().len())?);
        Ok(())
    }

    #[test]
    fn attention_value_tiles_preserve_strided_oracle_bits_across_cache_pages() -> Result<()> {
        let width = 71;
        let tile_width = crate::kernels::pinned_attention::VALUE_TILE;
        let cases = [0_u16, 0x8000, 1, 0x0400, 0x3c00, 0xbc00, 0x7bff, 0xfbff];
        for history in [
            0, 1, 14, 15, 16, 17, 63, 64, 65, 127, 128, 129, 254, 255, 256, 257, 8191,
        ] {
            let mut cache = KvCache::new(width, history + 1)?;
            for position in 0..history {
                let values: Vec<f32> = (0..width)
                    .map(|dimension| half_to_float(cases[(position + dimension) % cases.len()]))
                    .collect();
                cache.reserve_append()?;
                cache.append(&values);
            }
            let current: Vec<f32> = (0..width)
                .map(|dimension| half_to_float(cases[(history + dimension) % cases.len()]))
                .collect();
            for weights in [[0.5_f32, -0.25, 0.125, 0.0, -0.0], [-0.0; 5]] {
                let probabilities: Vec<f32> = (0..=history)
                    .map(|position| weights[position % weights.len()])
                    .collect();
                for arithmetic in [Arithmetic::Source, Arithmetic::PinnedSwift] {
                    let mut actual = vec![f32::NAN; width];
                    for (tile, output) in actual.chunks_mut(tile_width).enumerate() {
                        let offset = tile * tile_width;
                        arithmetic.attention_value_tile(
                            &cache,
                            &current[offset..offset + output.len()],
                            offset,
                            &probabilities,
                            output,
                        )?;
                    }
                    for (dimension, actual) in actual.into_iter().enumerate() {
                        let expected = arithmetic.attention_values(
                            &cache,
                            current[dimension],
                            dimension,
                            &probabilities,
                        )?;
                        assert_eq!(
                            actual.to_bits(),
                            expected.to_bits(),
                            "history {history}, channel {dimension}"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn paged_cache_iteration_keeps_flat_f16_dot_and_weighted_sum_order() -> Result<()> {
        let mut cache = KvCache::new(3, 3 * KV_PAGE_TOKENS)?;
        let mut flat = Vec::new();
        for position in 0..=2 * KV_PAGE_TOKENS {
            let values = match position % 3 {
                0 => [65_504.0, -0.0, 1.0],
                1 => [0.000_061_035_156, 0.5, -65_504.0],
                _ => [-65_504.0, -0.25, 0.000_061_035_156],
            };
            cache.reserve_append()?;
            cache.append(&values);
            flat.extend(values.map(half_bits));
        }
        let query = [1.000_000_1, -0.5, 0.25];
        assert!(
            cache
                .tokens()
                .map(|token| dot_f16(&query, token).to_bits())
                .eq(flat
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|token| dot_f16(&query, token).to_bits()))
        );
        for dimension in 0..3 {
            let weighted = |token: &[u16], index: usize| {
                let probability = [0.5, -0.25, 0.125][index % 3];
                f64::from(probability * half_to_float(token[dimension]))
            };
            let paged = cache
                .tokens()
                .enumerate()
                .map(|(index, token)| weighted(token, index))
                .sum::<f64>();
            let contiguous = flat
                .as_chunks::<3>()
                .0
                .iter()
                .enumerate()
                .map(|(index, token)| weighted(token, index))
                .sum::<f64>();
            assert_eq!(paged.to_bits(), contiguous.to_bits());
        }
        Ok(())
    }

    #[test]
    fn paged_cache_growth_seals_without_payload_copy_and_remains_context_bounded() -> Result<()> {
        let width = 3;
        let context = 2 * KV_PAGE_TOKENS + 5;
        let mut cache = KvCache::new(width, context)?;
        for position in 0..context {
            cache.reserve_append()?;
            let prepared_pointer = cache.tail.as_ptr();
            cache.append(&[1.0; 3]);
            assert_eq!(cache.len, (position + 1) * width);
            assert_eq!(cache.tokens().count(), position + 1);
            assert!(cache.tokens().all(|token| token == [0x3c00; 3]));
            assert!(cache.tail.capacity() <= width * KV_PAGE_TOKENS);
            let allocated = cache
                .pages
                .iter()
                .map(|page| page.capacity())
                .sum::<usize>()
                + cache.tail.capacity();
            assert!(allocated <= width * context);
            if cache.tail.is_empty() {
                let sealed = cache
                    .pages
                    .last()
                    .ok_or_else(|| crate::invalid("missing sealed page"))?;
                assert_eq!(sealed.as_ptr(), prepared_pointer);
            }
        }
        let saved = cache.copy()?;
        assert!(cache.reserve_append().is_err());
        assert!(cache.tokens().eq(saved.tokens()));
        assert_eq!(cache.pages.len(), 3);
        assert_eq!(cache.pages[2].len(), 5 * width);
        assert_eq!(cache.tail, Vec::<u16>::new());
        cache.clear();
        cache.reserve_append()?;
        cache.append(&[-1.0; 3]);
        assert_eq!(cache.tokens().collect::<Vec<_>>(), vec![&[0xbc00; 3][..]]);
        assert!(saved.tokens().all(|token| token == [0x3c00; 3]));
        assert!(KvCache::new(usize::MAX, 2).is_err());
        assert!(KvCache::new(1, MAX_STATE_BYTES).is_err());
        assert!(KvCache::new(0, 1).is_err());
        Ok(())
    }
}
