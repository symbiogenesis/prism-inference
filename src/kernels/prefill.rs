//! Bounded multi-token projection with a single decoded packed weight block.
//!
//! The block/row loop surrounds the token loop. Unlike repeated matvec calls,
//! every PQ2/PTQ1 block is decoded once for the entire tile. Each token retains
//! its original activation quantizer, ordered block accumulation and pinned
//! PTQ1 four-lane reduction. No decoded matrix survives this call.

#![expect(
    clippy::redundant_pub_crate,
    reason = "Keep graph-consumed tile types crate scoped inside the private kernel module."
)]

#[cfg(test)]
use std::sync::Arc;

use super::{
    ActivationContract, BF16, F16, F32, MatvecBackend, MatvecScratch, PQ2, PTQ1, Q4_0, Q4_1, Q5_0,
    Q5_1, Q8_0, RowParallelism, bf16_dot, dimensions, dot_additional, dot_legacy,
    f16_f64_dot_scalar, f16_pinned_dot_scalar, finite, integer_float, kquants, matvec_into, scale,
};
use crate::{Result, invalid, model::Tensor};

#[path = "prefill_dot.rs"]
mod dot;

pub(crate) const MAX_TILE_TOKENS: usize = 64;
pub(crate) const ROW_TASK_SCRATCH_BYTES: usize = 32 * 1024;

const fn supports_tiled_rows(kind: u32) -> bool {
    matches!(
        kind,
        PQ2 | PTQ1
            | F32
            | F16
            | BF16
            | Q4_0
            | Q4_1
            | Q5_0
            | Q5_1
            | Q8_0
            | 10..=14
            | 16..=23
            | 29
            | 34
            | 35
            | 39..=42
    )
}

fn dot_quantized_tile_row(
    kind: u32,
    row: &[u8],
    preparations: &[MatvecScratch],
    backend: MatvecBackend,
    decoded_blocks: &mut u64,
) -> Result<[f32; MAX_TILE_TOKENS]> {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if backend != MatvecBackend::Scalar {
        if kind == Q8_0
            && let Some(res) = super::neon::dot_q8_0_tile_row::<MAX_TILE_TOKENS>(row, preparations)
        {
            return res;
        }
        if kind == Q4_0
            && let Some(res) = super::neon::dot_q4_0_tile_row::<MAX_TILE_TOKENS>(row, preparations)
        {
            return res;
        }
    }
    let _ = backend;
    if matches!(kind, Q4_0 | Q8_0) {
        return dot::dot_legacy_tile_row(kind, row, preparations, decoded_blocks);
    }
    let mut sums = [0.0_f32; MAX_TILE_TOKENS];
    for (sum, scratch) in sums.iter_mut().zip(preparations) {
        *sum = match kind {
            Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 => dot_legacy(kind, row, &scratch.q8)?,
            10..=14 => kquants::dot(kind, row, &scratch.q8k)?,
            _ => dot_additional(kind, row, &scratch.q8, &scratch.q8k)?,
        };
    }
    Ok(sums)
}

#[derive(Clone)]
struct PreparedBatch {
    tokens: usize,
    columns: usize,
    pinned: bool,
    contract: ActivationContract,
    tensors: [Option<crate::model::WeakTensorBytes>; 4],
}

#[derive(Default)]
pub(crate) struct ProjectionTile {
    backend: MatvecBackend,
    preparations: Vec<MatvecScratch>,
    last_preparation: Option<PreparedBatch>,
    row_parallelism: RowParallelism,
    pub(crate) packed_projections: u64,
    pub(crate) decoded_blocks: u64,
    pub(crate) trellis: super::trellis::TrellisScratch,
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    pq2_panels: Vec<bonsai_neon_dot::Pq2I8mmPairBlock>,
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    pq2_panels_tokens: usize,
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    pq2_panels_blocks: usize,
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    pq2_panels_valid: bool,
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    ptq1_panels: Vec<bonsai_neon_dot::Ptq1PinnedI8mmPairBlock>,
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    ptq1_panels_tokens: usize,
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    ptq1_panels_groups: usize,
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    ptq1_panels_valid: bool,
}

impl ProjectionTile {
    pub(crate) fn with_backend(backend: MatvecBackend) -> Self {
        Self {
            backend,
            ..Self::default()
        }
    }

    #[cfg_attr(
        not(feature = "parallel-rows"),
        allow(
            clippy::missing_const_for_fn,
            reason = "The parallel-rows build drops the replaced pool owner; retain one non-const signature across feature sets."
        )
    )]
    pub(crate) fn set_row_parallelism(&mut self, parallelism: RowParallelism) {
        self.row_parallelism = parallelism;
    }

    pub(crate) const fn row_parallelism(&self) -> &RowParallelism {
        &self.row_parallelism
    }

    pub(crate) const fn requested_backend(&self) -> MatvecBackend {
        self.backend
    }

    pub(crate) const fn row_parallelism_and_trellis(
        &mut self,
    ) -> (&RowParallelism, &mut super::trellis::TrellisScratch) {
        (&self.row_parallelism, &mut self.trellis)
    }

    #[cfg(test)]
    pub(crate) fn project(
        &mut self,
        tensor: &Tensor,
        inputs: &[Vec<f32>],
        outputs: &mut [Vec<f32>],
        pinned: bool,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        self.project_with_previous(None, tensor, inputs, outputs, pinned, check)
    }

    pub(crate) fn project_with_previous(
        &mut self,
        previous: Option<&Tensor>,
        tensor: &Tensor,
        inputs: &[Vec<f32>],
        outputs: &mut [Vec<f32>],
        pinned: bool,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        self.project_group_internal(previous, &[tensor], inputs, &mut [outputs], pinned, check)
    }

    pub(crate) fn supports_siblings(tensors: &[&Tensor]) -> bool {
        let Some(first) = tensors.first() else {
            return false;
        };
        let Ok((columns, _, row_bytes)) = dimensions(first) else {
            return false;
        };
        (2..=3).contains(&tensors.len())
            // The joined scheduler retains the existing per-format row
            // kernels, including Q8_0 for odd PQ2 and PTQ1. Row tails use the
            // same one-row kernel; joining does not require a four-row head.
            && matches!(first.kind, F32 | BF16 | PQ2 | PTQ1)
            && tensors.iter().all(|tensor| {
                tensor.kind == first.kind
                    && dimensions(tensor).is_ok_and(|(cols, rows, bytes)| {
                        cols == columns && rows > 0 && bytes == row_bytes
                    })
            })
    }

    pub(crate) fn project_siblings_with_previous(
        &mut self,
        previous: Option<&Tensor>,
        tensors: &[&Tensor],
        inputs: &[Vec<f32>],
        outputs: &mut [&mut [Vec<f32>]],
        pinned: bool,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        if !Self::supports_siblings(tensors) || tensors.len() != outputs.len() {
            for group in outputs {
                for output in group.iter_mut() {
                    output.clear();
                }
            }
            return Err(invalid("incompatible CPU projection siblings"));
        }
        self.project_group_internal(previous, tensors, inputs, outputs, pinned, check)
    }

    #[expect(
        clippy::suboptimal_flops,
        clippy::too_many_lines,
        reason = "Retain the separately rounded source arithmetic and auditable packed block order."
    )]
    fn project_group_internal(
        &mut self,
        previous: Option<&Tensor>,
        tensors: &[&Tensor],
        inputs: &[Vec<f32>],
        all_outputs: &mut [&mut [Vec<f32>]],
        pinned: bool,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        let tensor = tensors[0];
        for group in all_outputs.iter_mut() {
            for output in group.iter_mut() {
                output.clear();
            }
        }
        let previous_preparation = self.last_preparation.take();
        let mut reused_batch = false;
        let result = (|| {
            let (columns, rows, block_bytes) = dimensions(tensor)?;
            for (sibling, outputs) in tensors.iter().zip(all_outputs.iter_mut()).skip(1) {
                if outputs.len() != inputs.len() {
                    return Err(invalid("invalid bounded projection tile"));
                }
                let (_, sibling_rows, _) = dimensions(sibling)?;
                for output in outputs.iter_mut() {
                    output.try_reserve_exact(sibling_rows)?;
                }
            }
            let outputs = &mut *all_outputs[0];
            if inputs.is_empty() || inputs.len() > MAX_TILE_TOKENS || outputs.len() != inputs.len()
            {
                return Err(invalid("invalid bounded projection tile"));
            }
            let current_contract = ActivationContract::for_tensor(tensor, columns, pinned);
            let reuse_prepared = supports_tiled_rows(tensor.kind)
                && self.preparations.len() >= inputs.len()
                && previous_preparation.as_ref().is_some_and(|prepared| {
                    prepared.tokens == inputs.len()
                        && prepared.columns == columns
                        && prepared.pinned == pinned
                        && prepared.contract == current_contract
                        && previous.is_some_and(|prev| {
                            supports_tiled_rows(prev.kind)
                                && dimensions(prev).is_ok_and(|(prev_cols, _, _)| {
                                    prev_cols == columns
                                        && ActivationContract::for_tensor(prev, prev_cols, pinned)
                                            == current_contract
                                })
                                && prepared.tensors.iter().flatten().any(|weak| {
                                    weak.upgrade().is_some_and(|data| {
                                        crate::model::TensorBytes::ptr_eq(&data, &prev.data)
                                    })
                                })
                        })
                });
            reused_batch = reuse_prepared;
            while self.preparations.len() < inputs.len() {
                self.preparations.try_reserve_exact(1)?;
                self.preparations.push(if pinned {
                    MatvecScratch::with_pinned_backend(self.backend)
                } else {
                    MatvecScratch::with_backend(self.backend)
                });
            }
            let parallel_prepare = !reuse_prepared && self.row_parallelism.uses_pool(inputs.len());
            for ((scratch, input), output) in self
                .preparations
                .iter_mut()
                .zip(inputs)
                .zip(outputs.iter_mut())
            {
                // Preparation also validates every input before publishing any
                // tile output, including unsupported-format fallback inputs.
                check()?;
                scratch.pinned_accumulation = pinned;
                if !reuse_prepared && !parallel_prepare {
                    scratch.prepare_input(tensor, input)?;
                    #[cfg(all(
                        feature = "experimental-neon",
                        target_arch = "aarch64",
                        target_os = "windows",
                        target_env = "msvc",
                        target_feature = "neon"
                    ))]
                    if tensor.kind == PQ2
                        && rows >= 4
                        && columns.is_multiple_of(256)
                        && !matches!(self.backend, MatvecBackend::Scalar)
                    {
                        scratch.prepare_pq2_consumption()?;
                    }
                }
                output.try_reserve_exact(rows)?;
            }
            if parallel_prepare {
                #[cfg(all(
                    feature = "experimental-neon",
                    target_arch = "aarch64",
                    target_os = "windows",
                    target_env = "msvc",
                    target_feature = "neon"
                ))]
                let prepare_pq2 = tensor.kind == PQ2
                    && rows >= 4
                    && columns.is_multiple_of(256)
                    && !matches!(self.backend, MatvecBackend::Scalar);
                self.row_parallelism.update_chunks(
                    &mut self.preparations[..inputs.len()],
                    1,
                    |base, chunk| {
                        for (offset, scratch) in chunk.iter_mut().enumerate() {
                            scratch.prepare_input(tensor, &inputs[base + offset])?;
                            #[cfg(all(
                                feature = "experimental-neon",
                                target_arch = "aarch64",
                                target_os = "windows",
                                target_env = "msvc",
                                target_feature = "neon"
                            ))]
                            if prepare_pq2 {
                                scratch.prepare_pq2_consumption()?;
                            }
                        }
                        Ok(())
                    },
                )?;
            }
            #[cfg(all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc",
                target_feature = "neon"
            ))]
            if reuse_prepared
                && tensor.kind == PQ2
                && rows >= 4
                && columns.is_multiple_of(256)
                && !matches!(self.backend, MatvecBackend::Scalar)
            {
                for scratch in &mut self.preparations[..inputs.len()] {
                    if scratch.pq2_consumption.is_empty() {
                        scratch.prepare_pq2_consumption()?;
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
            if !reuse_prepared {
                self.pq2_panels_valid = false;
                self.ptq1_panels_valid = false;
            }
            #[cfg(all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc",
                target_feature = "neon"
            ))]
            if tensor.kind == PQ2
                && inputs.len() >= 8
                && columns.is_multiple_of(256)
                && !matches!(self.backend, MatvecBackend::Scalar)
                && (rows >= 8
                    || tensors[1..]
                        .iter()
                        .any(|t| dimensions(t).is_ok_and(|(_, r, _)| r >= 8)))
            {
                let block_count = columns / 128;
                if self.preparations[..inputs.len()]
                    .iter()
                    .all(|scratch| scratch.pq2_consumption.len() >= block_count)
                    && let Some(dotprod) = self.preparations[0].pq2_consumption.dotprod
                    && dotprod.has_i8mm()
                    && (!self.pq2_panels_valid
                        || self.pq2_panels_tokens != inputs.len()
                        || self.pq2_panels_blocks != block_count)
                {
                    let chunks8_len = inputs.len() / 8;
                    self.pq2_panels.clear();
                    let mut packed_ok = true;
                    for c in 0..chunks8_len {
                        for p in 0..4 {
                            let t0 =
                                &self.preparations[c * 8 + p * 2].pq2_consumption[..block_count];
                            let t1 = &self.preparations[c * 8 + p * 2 + 1].pq2_consumption
                                [..block_count];
                            if dotprod
                                .append_i8mm_pair_blocks(t0, t1, &mut self.pq2_panels)
                                .is_none()
                            {
                                packed_ok = false;
                                break;
                            }
                        }
                        if !packed_ok {
                            break;
                        }
                    }
                    if packed_ok && self.pq2_panels.len() == chunks8_len * 4 * block_count {
                        self.pq2_panels_valid = true;
                        self.pq2_panels_tokens = inputs.len();
                        self.pq2_panels_blocks = block_count;
                    } else {
                        self.pq2_panels.clear();
                        self.pq2_panels_valid = false;
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
            if tensor.kind == PTQ1
                && pinned
                && inputs.len() >= 8
                && columns.is_multiple_of(128)
                && !matches!(self.backend, MatvecBackend::Scalar)
                && (rows >= 8
                    || tensors[1..]
                        .iter()
                        .any(|t| dimensions(t).is_ok_and(|(_, r, _)| r >= 8)))
            {
                let q8_len = columns / 32;
                if self.preparations[..inputs.len()]
                    .iter()
                    .all(|scratch| scratch.q8.len() >= q8_len)
                    && let Some(ptq1_dotprod) = bonsai_neon_dot::Ptq1Dotprod::detect()
                    && ptq1_dotprod.has_i8mm()
                    && (!self.ptq1_panels_valid
                        || self.ptq1_panels_tokens != inputs.len()
                        || self.ptq1_panels_groups != q8_len)
                {
                    let chunks8_len = inputs.len() / 8;
                    self.ptq1_panels.clear();
                    let mut packed_ok = true;
                    for c in 0..chunks8_len {
                        for p in 0..4 {
                            let t0 = &self.preparations[c * 8 + p * 2].q8[..q8_len];
                            let t1 = &self.preparations[c * 8 + p * 2 + 1].q8[..q8_len];
                            if ptq1_dotprod
                                .append_pinned_i8mm_pair_blocks(t0, t1, &mut self.ptq1_panels)
                                .is_none()
                            {
                                packed_ok = false;
                                break;
                            }
                        }
                        if !packed_ok {
                            break;
                        }
                    }
                    if packed_ok && self.ptq1_panels.len() == chunks8_len * 4 * q8_len {
                        self.ptq1_panels_valid = true;
                        self.ptq1_panels_tokens = inputs.len();
                        self.ptq1_panels_groups = q8_len;
                    } else {
                        self.ptq1_panels.clear();
                        self.ptq1_panels_valid = false;
                    }
                }
            }
            if tensor.kind == F32 {
                let backend = self.backend;
                let preparations = &self.preparations[..inputs.len()];
                let mut unused_blocks = 0_u64;
                let dot_f32_one = |row: &[u8], _: &mut u64| -> Result<[f32; MAX_TILE_TOKENS]> {
                    dot::dot_f32_row::<MAX_TILE_TOKENS>(row, preparations, columns, pinned, backend)
                };
                let dot_eight = |rows8: [&[u8]; 8], _: &mut u64| {
                    dot::dot_f32_eight_rows::<MAX_TILE_TOKENS>(
                        rows8,
                        preparations,
                        columns,
                        pinned,
                        backend,
                    )
                };
                let dot_four = |rows4: [&[u8]; 4], _: &mut u64| {
                    dot::dot_f32_four_rows::<MAX_TILE_TOKENS>(
                        rows4,
                        preparations,
                        columns,
                        pinned,
                        backend,
                    )
                };
                if tensors.len() == 1 {
                    return self.row_parallelism.fill_tile_four_rows(
                        &tensor.data,
                        block_bytes,
                        all_outputs[0],
                        &mut unused_blocks,
                        check,
                        dot_eight,
                        dot_four,
                        dot_f32_one,
                    );
                }
                let mut data = [&[][..]; 3];
                for (slot, tensor) in data.iter_mut().zip(tensors) {
                    *slot = &tensor.data;
                }
                return self.row_parallelism.fill_tile_siblings_four_rows(
                    &data[..tensors.len()],
                    block_bytes,
                    all_outputs,
                    &mut unused_blocks,
                    check,
                    dot_eight,
                    dot_four,
                    dot_f32_one,
                );
            }
            if tensor.kind == BF16 {
                let backend = self.backend;
                let preparations = &self.preparations[..inputs.len()];
                let mut unused_blocks = 0_u64;
                let dot_bf16_one = |row: &[u8], _: &mut u64| -> Result<[f32; MAX_TILE_TOKENS]> {
                    let mut sums = [0.0_f32; MAX_TILE_TOKENS];
                    for (sum, scratch) in sums.iter_mut().zip(preparations) {
                        *sum = bf16_dot(row, &scratch.activations)?;
                    }
                    Ok(sums)
                };
                let dot_eight = |rows8: [&[u8]; 8], decoded: &mut u64| {
                    if let Some(res) =
                        dot::dot_bf16_eight_rows::<MAX_TILE_TOKENS>(rows8, preparations, backend)
                    {
                        return res;
                    }
                    Ok([
                        dot_bf16_one(rows8[0], decoded)?,
                        dot_bf16_one(rows8[1], decoded)?,
                        dot_bf16_one(rows8[2], decoded)?,
                        dot_bf16_one(rows8[3], decoded)?,
                        dot_bf16_one(rows8[4], decoded)?,
                        dot_bf16_one(rows8[5], decoded)?,
                        dot_bf16_one(rows8[6], decoded)?,
                        dot_bf16_one(rows8[7], decoded)?,
                    ])
                };
                let dot_four = |rows4: [&[u8]; 4], decoded: &mut u64| {
                    if let Some(res) =
                        dot::dot_bf16_four_rows::<MAX_TILE_TOKENS>(rows4, preparations, backend)
                    {
                        return res;
                    }
                    Ok([
                        dot_bf16_one(rows4[0], decoded)?,
                        dot_bf16_one(rows4[1], decoded)?,
                        dot_bf16_one(rows4[2], decoded)?,
                        dot_bf16_one(rows4[3], decoded)?,
                    ])
                };
                if tensors.len() == 1 {
                    return self.row_parallelism.fill_tile_four_rows(
                        &tensor.data,
                        block_bytes,
                        all_outputs[0],
                        &mut unused_blocks,
                        check,
                        dot_eight,
                        dot_four,
                        dot_bf16_one,
                    );
                }
                let mut data = [&[][..]; 3];
                for (slot, tensor) in data.iter_mut().zip(tensors) {
                    *slot = &tensor.data;
                }
                return self.row_parallelism.fill_tile_siblings_four_rows(
                    &data[..tensors.len()],
                    block_bytes,
                    all_outputs,
                    &mut unused_blocks,
                    check,
                    dot_eight,
                    dot_four,
                    dot_bf16_one,
                );
            }
            if tensor.kind == F16 {
                let backend = self.backend;
                let preparations = &self.preparations[..inputs.len()];
                let use_pinned = pinned && columns.is_multiple_of(16);
                let mut act_refs: [&[f32]; MAX_TILE_TOKENS] = [&[]; MAX_TILE_TOKENS];
                for (slot, scratch) in act_refs.iter_mut().zip(preparations) {
                    *slot = &scratch.activations;
                }
                let active_acts = &act_refs[..preparations.len()];
                let mut unused_blocks = 0_u64;
                let dot_f16_one_scalar = |row: &[u8]| -> Result<[f32; MAX_TILE_TOKENS]> {
                    let mut sums = [0.0_f32; MAX_TILE_TOKENS];
                    for (sum, &acts) in sums.iter_mut().zip(active_acts) {
                        *sum = if use_pinned {
                            f16_pinned_dot_scalar(row, acts)?
                        } else {
                            f16_f64_dot_scalar(row, acts)?
                        };
                    }
                    Ok(sums)
                };
                let dot_f16_one = |row: &[u8], _: &mut u64| -> Result<[f32; MAX_TILE_TOKENS]> {
                    #[cfg(all(
                        feature = "experimental-neon",
                        target_arch = "aarch64",
                        target_os = "windows",
                        target_env = "msvc",
                        target_feature = "neon"
                    ))]
                    if backend != MatvecBackend::Scalar {
                        let mut sums = [0.0_f32; MAX_TILE_TOKENS];
                        let ok = if use_pinned {
                            bonsai_neon_dot::dot_f16_tile_row_pinned(row, active_acts, &mut sums)
                        } else {
                            bonsai_neon_dot::dot_f16_tile_row(row, active_acts, &mut sums)
                        };
                        return ok
                            .map(|()| sums)
                            .ok_or_else(|| invalid("nonfinite tensor or activation value"));
                    }
                    let _ = backend;
                    dot_f16_one_scalar(row)
                };
                let dot_eight = |rows8: [&[u8]; 8],
                                 _: &mut u64|
                 -> Result<[[f32; MAX_TILE_TOKENS]; 8]> {
                    #[cfg(all(
                        feature = "experimental-neon",
                        target_arch = "aarch64",
                        target_os = "windows",
                        target_env = "msvc",
                        target_feature = "neon"
                    ))]
                    if backend != MatvecBackend::Scalar {
                        let mut sums = [[0.0_f32; MAX_TILE_TOKENS]; 8];
                        let ok = if use_pinned {
                            bonsai_neon_dot::dot_f16_tile_rows_pinned(rows8, active_acts, &mut sums)
                        } else {
                            bonsai_neon_dot::dot_f16_tile_rows(rows8, active_acts, &mut sums)
                        };
                        return ok
                            .map(|()| sums)
                            .ok_or_else(|| invalid("nonfinite tensor or activation value"));
                    }
                    let _ = backend;
                    Ok([
                        dot_f16_one_scalar(rows8[0])?,
                        dot_f16_one_scalar(rows8[1])?,
                        dot_f16_one_scalar(rows8[2])?,
                        dot_f16_one_scalar(rows8[3])?,
                        dot_f16_one_scalar(rows8[4])?,
                        dot_f16_one_scalar(rows8[5])?,
                        dot_f16_one_scalar(rows8[6])?,
                        dot_f16_one_scalar(rows8[7])?,
                    ])
                };
                let dot_four = |rows4: [&[u8]; 4],
                                _: &mut u64|
                 -> Result<[[f32; MAX_TILE_TOKENS]; 4]> {
                    #[cfg(all(
                        feature = "experimental-neon",
                        target_arch = "aarch64",
                        target_os = "windows",
                        target_env = "msvc",
                        target_feature = "neon"
                    ))]
                    if backend != MatvecBackend::Scalar {
                        let mut sums = [[0.0_f32; MAX_TILE_TOKENS]; 4];
                        let ok = if use_pinned {
                            bonsai_neon_dot::dot_f16_tile_rows_pinned(rows4, active_acts, &mut sums)
                        } else {
                            bonsai_neon_dot::dot_f16_tile_rows(rows4, active_acts, &mut sums)
                        };
                        return ok
                            .map(|()| sums)
                            .ok_or_else(|| invalid("nonfinite tensor or activation value"));
                    }
                    let _ = backend;
                    Ok([
                        dot_f16_one_scalar(rows4[0])?,
                        dot_f16_one_scalar(rows4[1])?,
                        dot_f16_one_scalar(rows4[2])?,
                        dot_f16_one_scalar(rows4[3])?,
                    ])
                };
                return self.row_parallelism.fill_tile_four_rows(
                    &tensor.data,
                    block_bytes,
                    all_outputs[0],
                    &mut unused_blocks,
                    check,
                    dot_eight,
                    dot_four,
                    dot_f16_one,
                );
            }
            if matches!(
                tensor.kind,
                Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | 10..=14 | 16..=23 | 29 | 34 | 35 | 39..=42
            ) {
                let backend = self.backend;
                let preparations = &self.preparations[..inputs.len()];
                let mut unused_blocks = 0_u64;
                if matches!(tensor.kind, Q4_0 | Q8_0) {
                    let dot_quant_one =
                        |row: &[u8], decoded: &mut u64| -> Result<[f32; MAX_TILE_TOKENS]> {
                            dot_quantized_tile_row(tensor.kind, row, preparations, backend, decoded)
                        };
                    return self.row_parallelism.fill_tile_four_rows(
                        &tensor.data,
                        block_bytes,
                        outputs,
                        &mut unused_blocks,
                        check,
                        |rows8, decoded| {
                            if let Some(res) = dot::dot_quantized_eight_rows::<MAX_TILE_TOKENS>(
                                tensor.kind,
                                rows8,
                                preparations,
                                backend,
                            ) {
                                return res;
                            }
                            Ok([
                                dot_quant_one(rows8[0], decoded)?,
                                dot_quant_one(rows8[1], decoded)?,
                                dot_quant_one(rows8[2], decoded)?,
                                dot_quant_one(rows8[3], decoded)?,
                                dot_quant_one(rows8[4], decoded)?,
                                dot_quant_one(rows8[5], decoded)?,
                                dot_quant_one(rows8[6], decoded)?,
                                dot_quant_one(rows8[7], decoded)?,
                            ])
                        },
                        |rows4, decoded| {
                            if let Some(res) = dot::dot_quantized_four_rows::<MAX_TILE_TOKENS>(
                                tensor.kind,
                                rows4,
                                preparations,
                                backend,
                            ) {
                                return res;
                            }
                            Ok([
                                dot_quant_one(rows4[0], decoded)?,
                                dot_quant_one(rows4[1], decoded)?,
                                dot_quant_one(rows4[2], decoded)?,
                                dot_quant_one(rows4[3], decoded)?,
                            ])
                        },
                        dot_quant_one,
                    );
                }
                return self.row_parallelism.fill_tile(
                    &tensor.data,
                    block_bytes,
                    outputs,
                    &mut unused_blocks,
                    check,
                    |row, decoded| {
                        dot_quantized_tile_row(tensor.kind, row, preparations, backend, decoded)
                    },
                );
            }
            if !matches!(tensor.kind, PQ2 | PTQ1) {
                for ((input, output), scratch) in inputs
                    .iter()
                    .zip(outputs.iter_mut())
                    .zip(self.preparations.iter_mut())
                {
                    check()?;
                    scratch.set_row_parallelism(self.row_parallelism.clone());
                    matvec_into(tensor, input, output, scratch)?;
                }
                return Ok(());
            }
            self.packed_projections = self
                .packed_projections
                .saturating_add(u64::try_from(tensors.len())?);
            let backend = self.backend;
            #[cfg(all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc",
                target_feature = "neon"
            ))]
            let pq2_panels_slice = &self.pq2_panels;
            #[cfg(all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc",
                target_feature = "neon"
            ))]
            let pq2_panels_valid = self.pq2_panels_valid
                && self.pq2_panels_tokens == inputs.len()
                && self.pq2_panels_blocks == columns / 128;
            #[cfg(all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc",
                target_feature = "neon"
            ))]
            let ptq1_panels_slice = &self.ptq1_panels;
            #[cfg(all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc",
                target_feature = "neon"
            ))]
            let ptq1_panels_valid = self.ptq1_panels_valid
                && self.ptq1_panels_tokens == inputs.len()
                && self.ptq1_panels_groups == columns / 32;
            let preparations = &self.preparations[..inputs.len()];
            #[cfg(all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc",
                target_feature = "neon"
            ))]
            let pq2_block_count = columns / 128;
            #[cfg(all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc",
                target_feature = "neon"
            ))]
            let (pq2_chunks8, pq2_rem8) = preparations.as_chunks::<8>();
            #[cfg(all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc",
                target_feature = "neon"
            ))]
            let mut pq2_arg_chunks: [[&[bonsai_neon_dot::Pq2Activation]; 8]; 8] = [[&[]; 8]; 8];
            #[cfg(all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc",
                target_feature = "neon"
            ))]
            let mut pq2_rem_acts: [&[bonsai_neon_dot::Pq2Activation]; 8] = [&[]; 8];
            #[cfg(all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc",
                target_feature = "neon"
            ))]
            let mut pq2_panel_chunks: [[&[bonsai_neon_dot::Pq2I8mmPairBlock]; 4]; 8] =
                [[&[]; 4]; 8];
            #[cfg(all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc",
                target_feature = "neon"
            ))]
            let (pq2_prepared_chunks, pq2_prepared_rem, pq2_has_panels) = if tensor.kind == PQ2
                && backend != MatvecBackend::Scalar
                && columns.is_multiple_of(256)
                && pq2_block_count != 0
                && !preparations.is_empty()
                && preparations
                    .iter()
                    .all(|scratch| scratch.pq2_consumption.len() >= pq2_block_count)
                && let Some(dotprod) = preparations[0].pq2_consumption.dotprod
            {
                for (c, chunk) in pq2_chunks8.iter().enumerate() {
                    pq2_arg_chunks[c] =
                        std::array::from_fn(|t| &chunk[t].pq2_consumption[..pq2_block_count]);
                }
                for (t, scratch) in pq2_rem8.iter().enumerate() {
                    pq2_rem_acts[t] = &scratch.pq2_consumption[..pq2_block_count];
                }
                let prep_chunks = if pq2_chunks8.is_empty() {
                    None
                } else {
                    dotprod.prepare_scaled_chunks(&pq2_arg_chunks[..pq2_chunks8.len()])
                };
                let prep_rem = if pq2_rem8.is_empty() {
                    None
                } else {
                    dotprod.prepare_scaled_tokens(&pq2_rem_acts[..pq2_rem8.len()])
                };
                let has_panels = pq2_panels_valid
                    && !pq2_chunks8.is_empty()
                    && pq2_panels_slice.len() >= pq2_chunks8.len() * 4 * pq2_block_count;
                if has_panels {
                    for (c, panel_chunk) in
                        pq2_panel_chunks[..pq2_chunks8.len()].iter_mut().enumerate()
                    {
                        for (p, slot) in panel_chunk.iter_mut().enumerate() {
                            let start = (c * 4 + p) * pq2_block_count;
                            *slot = &pq2_panels_slice[start..start + pq2_block_count];
                        }
                    }
                }
                (prep_chunks, prep_rem, has_panels)
            } else {
                (None, None, false)
            };
            #[cfg(all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc",
                target_feature = "neon"
            ))]
            let ptq1_q8_len = columns / 32;
            #[cfg(all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc",
                target_feature = "neon"
            ))]
            let mut ptq1_arg_chunks: [[&[super::Q8Block]; 8]; 8] = [[&[]; 8]; 8];
            #[cfg(all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc",
                target_feature = "neon"
            ))]
            let mut ptq1_rem_acts: [&[super::Q8Block]; 8] = [&[]; 8];
            #[cfg(all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc",
                target_feature = "neon"
            ))]
            let mut ptq1_panel_chunks: [[&[bonsai_neon_dot::Ptq1PinnedI8mmPairBlock];
                4]; 8] = [[&[]; 4]; 8];
            #[cfg(all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc",
                target_feature = "neon"
            ))]
            let ptq1_pinned_dotprod = if tensor.kind == PTQ1
                && pinned
                && backend != MatvecBackend::Scalar
                && columns.is_multiple_of(128)
                && ptq1_panels_valid
                && !pq2_chunks8.is_empty()
                && ptq1_panels_slice.len() >= pq2_chunks8.len() * 4 * ptq1_q8_len
                && preparations
                    .iter()
                    .all(|scratch| scratch.q8.len() >= ptq1_q8_len)
                && let Some(dotprod) = bonsai_neon_dot::Ptq1Dotprod::detect()
                && dotprod.has_i8mm()
            {
                for (c, chunk) in pq2_chunks8.iter().enumerate() {
                    ptq1_arg_chunks[c] = std::array::from_fn(|t| &chunk[t].q8[..ptq1_q8_len]);
                }
                for (t, scratch) in pq2_rem8.iter().enumerate() {
                    ptq1_rem_acts[t] = &scratch.q8[..ptq1_q8_len];
                }
                for (c, panel_chunk) in ptq1_panel_chunks[..pq2_chunks8.len()]
                    .iter_mut()
                    .enumerate()
                {
                    for (p, slot) in panel_chunk.iter_mut().enumerate() {
                        let start = (c * 4 + p) * ptq1_q8_len;
                        *slot = &ptq1_panels_slice[start..start + ptq1_q8_len];
                    }
                }
                Some(dotprod)
            } else {
                None
            };
            let dot_one = |row: &[u8],
                           decoded_blocks: &mut u64|
             -> Result<[f32; MAX_TILE_TOKENS]> {
                let mut sums = [0.0_f32; MAX_TILE_TOKENS];
                let mut lanes = [[0.0_f32; 4]; MAX_TILE_TOKENS];
                let bytes = if tensor.kind == PQ2 { 34 } else { 28 };
                for (index, block) in row.chunks_exact(bytes).enumerate() {
                    let weight_scale = if tensor.kind == PQ2 {
                        scale(block)?
                    } else {
                        scale(&block[26..])?
                    };
                    // Prepare one SIMD/scalar block for all tokens. This stays
                    // stack-scoped and never expands or retains a matrix.
                    let weights = if tensor.kind == PQ2 {
                        dot::Block::from_pq2(block[2..34].try_into()?, backend)
                    } else {
                        dot::Block::from_ptq1(block.try_into()?, backend)?
                    };
                    *decoded_blocks = decoded_blocks.saturating_add(1);
                    if tensor.kind == PQ2 && columns.is_multiple_of(256) {
                        let start = (index % 2) * 128;
                        let (chunks8, rem8) = preparations.as_chunks::<8>();
                        for (chunk_idx, chunk) in chunks8.iter().enumerate() {
                            let base = chunk_idx * 8;
                            let a0 = &chunk[0].q8k[index / 2];
                            let a1 = &chunk[1].q8k[index / 2];
                            let a2 = &chunk[2].q8k[index / 2];
                            let a3 = &chunk[3].q8k[index / 2];
                            let a4 = &chunk[4].q8k[index / 2];
                            let a5 = &chunk[5].q8k[index / 2];
                            let a6 = &chunk[6].q8k[index / 2];
                            let a7 = &chunk[7].q8k[index / 2];
                            let dots = weights.dot128_eight([
                                a0.values[start..start + 128].try_into()?,
                                a1.values[start..start + 128].try_into()?,
                                a2.values[start..start + 128].try_into()?,
                                a3.values[start..start + 128].try_into()?,
                                a4.values[start..start + 128].try_into()?,
                                a5.values[start..start + 128].try_into()?,
                                a6.values[start..start + 128].try_into()?,
                                a7.values[start..start + 128].try_into()?,
                            ]);
                            for (offset, (act, dot_int)) in [a0, a1, a2, a3, a4, a5, a6, a7]
                                .into_iter()
                                .zip(dots)
                                .enumerate()
                            {
                                let d = weight_scale * act.scale;
                                let dot = integer_float(dot_int);
                                let token = base + offset;
                                sums[token] = if pinned {
                                    d.mul_add(dot, sums[token])
                                } else {
                                    sums[token] + d * dot
                                };
                            }
                        }
                        let base4 = chunks8.len() * 8;
                        let (chunks4, rem) = rem8.as_chunks::<4>();
                        for (chunk_idx, chunk) in chunks4.iter().enumerate() {
                            let base = base4 + chunk_idx * 4;
                            let a0 = &chunk[0].q8k[index / 2];
                            let a1 = &chunk[1].q8k[index / 2];
                            let a2 = &chunk[2].q8k[index / 2];
                            let a3 = &chunk[3].q8k[index / 2];
                            let dots = weights.dot128_four([
                                a0.values[start..start + 128].try_into()?,
                                a1.values[start..start + 128].try_into()?,
                                a2.values[start..start + 128].try_into()?,
                                a3.values[start..start + 128].try_into()?,
                            ]);
                            for (offset, (act, dot_int)) in
                                [a0, a1, a2, a3].into_iter().zip(dots).enumerate()
                            {
                                let d = weight_scale * act.scale;
                                let dot = integer_float(dot_int);
                                let token = base + offset;
                                sums[token] = if pinned {
                                    d.mul_add(dot, sums[token])
                                } else {
                                    sums[token] + d * dot
                                };
                            }
                        }
                        let rem_base = base4 + chunks4.len() * 4;
                        for (offset, scratch) in rem.iter().enumerate() {
                            let activation = &scratch.q8k[index / 2];
                            let d = weight_scale * activation.scale;
                            let dot = integer_float(
                                weights.dot128(activation.values[start..start + 128].try_into()?),
                            );
                            let token = rem_base + offset;
                            sums[token] = if pinned {
                                d.mul_add(dot, sums[token])
                            } else {
                                sums[token] + d * dot
                            };
                        }
                    } else if tensor.kind == PTQ1 && pinned {
                        for (token, scratch) in preparations.iter().enumerate() {
                            let activations = <&[super::Q8Block; 4]>::try_from(
                                &scratch.q8[index * 4..index * 4 + 4],
                            )?;
                            weights.accumulate_lanes128(
                                weight_scale,
                                activations,
                                &mut lanes[token],
                            );
                        }
                    } else {
                        for (token, scratch) in preparations.iter().enumerate() {
                            let activations = <&[super::Q8Block; 4]>::try_from(
                                &scratch.q8[index * 4..index * 4 + 4],
                            )?;
                            sums[token] += weight_scale * weights.dot128_q8(activations);
                        }
                    }
                }
                for (sum, lanes) in sums.iter_mut().zip(lanes).take(inputs.len()) {
                    *sum = finite(if tensor.kind == PTQ1 && pinned {
                        (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
                    } else {
                        *sum
                    })?;
                }
                Ok(sums)
            };
            let dot_eight = |rows8: [&[u8]; 8], decoded_blocks: &mut u64| {
                #[cfg(all(
                    feature = "experimental-neon",
                    target_arch = "aarch64",
                    target_os = "windows",
                    target_env = "msvc",
                    target_feature = "neon"
                ))]
                if pq2_prepared_chunks.is_some() || pq2_prepared_rem.is_some() {
                    let mut sums = [[0.0_f32; MAX_TILE_TOKENS]; 8];
                    if let Some(prep_chunks) = pq2_prepared_chunks {
                        let mut out_buf = [[[0.0_f32; 8]; 8]; 8];
                        let out_slice = &mut out_buf[..pq2_chunks8.len()];
                        let ok = match (pq2_has_panels, pinned) {
                            (true, true) => prep_chunks.dot_scaled_8x8_chunks_with_panels::<true>(
                                rows8,
                                &pq2_panel_chunks[..pq2_chunks8.len()],
                                out_slice,
                            ),
                            (true, false) => prep_chunks
                                .dot_scaled_8x8_chunks_with_panels::<false>(
                                    rows8,
                                    &pq2_panel_chunks[..pq2_chunks8.len()],
                                    out_slice,
                                ),
                            (false, true) => {
                                prep_chunks.dot_scaled_8x8_chunks::<true>(rows8, out_slice)
                            }
                            (false, false) => {
                                prep_chunks.dot_scaled_8x8_chunks::<false>(rows8, out_slice)
                            }
                        };
                        let Some(()) = ok else {
                            return Err(invalid("nonfinite tensor or activation value"));
                        };
                        for (c, chunk_out) in out_slice.iter().enumerate() {
                            let base = c * 8;
                            for row in 0..8 {
                                sums[row][base..base + 8].copy_from_slice(&chunk_out[row]);
                            }
                        }
                    }
                    if let Some(prep_rem) = pq2_prepared_rem {
                        let mut rem_out = [[0.0_f32; 8]; 8];
                        let ok = if pinned {
                            prep_rem.dot_scaled_8x_tokens::<true>(rows8, &mut rem_out)
                        } else {
                            prep_rem.dot_scaled_8x_tokens::<false>(rows8, &mut rem_out)
                        };
                        let Some(()) = ok else {
                            return Err(invalid("nonfinite tensor or activation value"));
                        };
                        let base = pq2_chunks8.len() * 8;
                        let rem_len = pq2_rem8.len();
                        for row in 0..8 {
                            sums[row][base..base + rem_len]
                                .copy_from_slice(&rem_out[row][..rem_len]);
                        }
                    }
                    let block_count_u64 = u64::try_from(pq2_block_count)?;
                    *decoded_blocks =
                        decoded_blocks.saturating_add(block_count_u64.saturating_mul(8));
                    return Ok(sums);
                }
                #[cfg(all(
                    feature = "experimental-neon",
                    target_arch = "aarch64",
                    target_os = "windows",
                    target_env = "msvc",
                    target_feature = "neon"
                ))]
                if let Some(ptq1_dotprod) = ptq1_pinned_dotprod {
                    let mut sums = [[0.0_f32; MAX_TILE_TOKENS]; 8];
                    let mut out_buf = [[[0.0_f32; 8]; 8]; 8];
                    let out_slice = &mut out_buf[..pq2_chunks8.len()];
                    let Some(()) = ptq1_dotprod.dot_scaled_8x8_chunks_with_pinned_panels(
                        rows8,
                        &ptq1_arg_chunks[..pq2_chunks8.len()],
                        &ptq1_panel_chunks[..pq2_chunks8.len()],
                        out_slice,
                    ) else {
                        return Err(invalid("nonfinite tensor or activation value"));
                    };
                    for (c, chunk_out) in out_slice.iter().enumerate() {
                        let base = c * 8;
                        for row in 0..8 {
                            sums[row][base..base + 8].copy_from_slice(&chunk_out[row]);
                        }
                    }
                    if !pq2_rem8.is_empty() {
                        let mut rem_out = [[0.0_f32; 8]; 8];
                        let Some(()) = ptq1_dotprod.dot_scaled_8x_tokens_view::<true, _>(
                            rows8,
                            &ptq1_rem_acts[..pq2_rem8.len()],
                            &mut rem_out,
                        ) else {
                            return Err(invalid("nonfinite tensor or activation value"));
                        };
                        let base = pq2_chunks8.len() * 8;
                        let rem_len = pq2_rem8.len();
                        for row in 0..8 {
                            sums[row][base..base + rem_len]
                                .copy_from_slice(&rem_out[row][..rem_len]);
                        }
                    }
                    let block_count_u64 = u64::try_from(pq2_block_count)?;
                    *decoded_blocks =
                        decoded_blocks.saturating_add(block_count_u64.saturating_mul(8));
                    return Ok(sums);
                }
                if let Some(res) = dot::dot_eight_rows::<MAX_TILE_TOKENS>(
                    rows8,
                    preparations,
                    tensor.kind,
                    columns,
                    pinned,
                    backend,
                    decoded_blocks,
                ) {
                    return res;
                }
                Ok([
                    dot_one(rows8[0], decoded_blocks)?,
                    dot_one(rows8[1], decoded_blocks)?,
                    dot_one(rows8[2], decoded_blocks)?,
                    dot_one(rows8[3], decoded_blocks)?,
                    dot_one(rows8[4], decoded_blocks)?,
                    dot_one(rows8[5], decoded_blocks)?,
                    dot_one(rows8[6], decoded_blocks)?,
                    dot_one(rows8[7], decoded_blocks)?,
                ])
            };
            let dot_four = |rows4: [&[u8]; 4], decoded_blocks: &mut u64| {
                if let Some(res) = dot::dot_four_rows::<MAX_TILE_TOKENS>(
                    rows4,
                    preparations,
                    tensor.kind,
                    columns,
                    pinned,
                    backend,
                    decoded_blocks,
                ) {
                    return res;
                }
                Ok([
                    dot_one(rows4[0], decoded_blocks)?,
                    dot_one(rows4[1], decoded_blocks)?,
                    dot_one(rows4[2], decoded_blocks)?,
                    dot_one(rows4[3], decoded_blocks)?,
                ])
            };
            if tensors.len() == 1 {
                return self.row_parallelism.fill_tile_four_rows(
                    &tensor.data,
                    block_bytes,
                    all_outputs[0],
                    &mut self.decoded_blocks,
                    check,
                    dot_eight,
                    dot_four,
                    dot_one,
                );
            }
            let mut data = [&[][..]; 3];
            for (slot, tensor) in data.iter_mut().zip(tensors) {
                *slot = &tensor.data;
            }
            self.row_parallelism.fill_tile_siblings_four_rows(
                &data[..tensors.len()],
                block_bytes,
                all_outputs,
                &mut self.decoded_blocks,
                check,
                dot_eight,
                dot_four,
                dot_one,
            )
        })();
        if result.is_err() {
            for group in all_outputs {
                for output in group.iter_mut() {
                    output.clear();
                }
            }
        } else if supports_tiled_rows(tensor.kind)
            && let Ok((columns, _, _)) = dimensions(tensor)
        {
            let mut prepared_tensors = [None, None, None, None];
            for (slot, sibling) in prepared_tensors.iter_mut().zip(tensors) {
                *slot = Some(crate::model::TensorBytes::downgrade(&sibling.data));
            }
            if reused_batch
                && tensors.len() < prepared_tensors.len()
                && let Some(prev) = previous
            {
                prepared_tensors[tensors.len()] =
                    Some(crate::model::TensorBytes::downgrade(&prev.data));
            }
            self.last_preparation = Some(PreparedBatch {
                tokens: inputs.len(),
                columns,
                pinned,
                contract: ActivationContract::for_tensor(tensor, columns, pinned),
                tensors: prepared_tensors,
            });
        }
        result
    }
}

#[cfg(test)]
#[path = "prefill_sibling_tests.rs"]
mod sibling_tests;

#[cfg(test)]
#[path = "prefill_few_token_tests.rs"]
mod few_token_tests;

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    #[cfg(feature = "parallel-rows")]
    #[test]
    fn tiled_parallel_projection_matches_serial_and_recovers() -> Result<()> {
        for workers in [2, 4] {
            let parallelism = RowParallelism::new(workers)?;
            for (kind, columns) in [(PQ2, 256), (PQ2, 384), (PTQ1, 256)] {
                let mut packed = [if kind == PTQ1 { 0x66 } else { 0xe4 }; 34];
                packed[..2].copy_from_slice(&0x3555_u16.to_le_bytes());
                let block = if kind == PTQ1 {
                    bonsai_ternary::pack_ptq(&packed)?.to_vec()
                } else {
                    packed.to_vec()
                };
                let mut tensor = Tensor {
                    name: "parallel-packed-tile".into(),
                    dimensions: vec![columns, 7],
                    kind,
                    data: Arc::<[u8]>::from(block.repeat(columns / 128 * 7)).into(),
                };
                for count in [8, 16, 32] {
                    let inputs: Vec<Vec<f32>> = (0..count)
                        .map(|token| {
                            (0..columns)
                                .map(|index| {
                                    [1.5, -128.0, 0.0, -0.0, 63.5, -3.25][(index + token) % 6]
                                })
                                .collect()
                        })
                        .collect();
                    for backend in [
                        MatvecBackend::Scalar,
                        MatvecBackend::Auto,
                        MatvecBackend::ExperimentalNeon,
                    ] {
                        let mut tile = ProjectionTile::with_backend(backend);
                        tile.set_row_parallelism(parallelism.clone());
                        for pinned in [false, true] {
                            let mut actual = vec![vec![f32::NAN]; count];
                            let previous_blocks = tile.decoded_blocks;
                            tile.project(&tensor, &inputs, &mut actual, pinned, &|| Ok(()))?;
                            assert_eq!(
                                tile.decoded_blocks - previous_blocks,
                                u64::try_from(columns / 128 * 7)?
                            );
                            for (input, output) in inputs.iter().zip(&actual) {
                                let mut scratch = if pinned {
                                    MatvecScratch::with_pinned_backend(MatvecBackend::Scalar)
                                } else {
                                    MatvecScratch::with_backend(MatvecBackend::Scalar)
                                };
                                let mut expected = Vec::new();
                                matvec_into(&tensor, input, &mut expected, &mut scratch)?;
                                assert_eq!(
                                    output
                                        .iter()
                                        .map(|value| value.to_bits())
                                        .collect::<Vec<_>>(),
                                    expected
                                        .iter()
                                        .map(|value| value.to_bits())
                                        .collect::<Vec<_>>()
                                );
                            }
                            let valid = tensor.data.clone();
                            let mut bad = valid.to_vec();
                            let offset =
                                bad.len() - block.len() + if kind == PTQ1 { 26 } else { 0 };
                            bad[offset..offset + 2].copy_from_slice(&0x7c00_u16.to_le_bytes());
                            tensor.data = Arc::<[u8]>::from(bad).into();
                            assert!(
                                tile.project(&tensor, &inputs, &mut actual, pinned, &|| Ok(()))
                                    .is_err()
                            );
                            assert!(actual.iter().all(Vec::is_empty));
                            tensor.data = valid;
                            tile.project(&tensor, &inputs, &mut actual, pinned, &|| Ok(()))?;
                            assert!(actual.iter().all(|output| output.len() == 7));
                        }
                        assert_eq!(tile.row_parallelism.workers(), workers);
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn tiled_packed_projection_reuses_blocks_and_preserves_independent_rows() -> Result<()> {
        for (kind, columns) in [(142, 256), (142, 384), (143, 256)] {
            let mut block = [if kind == 143 { 0x66 } else { 0xe4 }; 34];
            block[..2].copy_from_slice(&0x3555_u16.to_le_bytes());
            let encoded = if kind == 143 {
                bonsai_ternary::pack_ptq(&block)?.to_vec()
            } else {
                block.to_vec()
            };
            let tensor = Tensor {
                name: "tile-packed-reference".into(),
                dimensions: vec![columns, 3],
                kind,
                data: Arc::<[u8]>::from(encoded.repeat(columns / 128 * 3)).into(),
            };
            for count in [1, 8, 16, 32] {
                let inputs: Vec<Vec<f32>> = (0..count)
                    .map(|token| {
                        (0..columns)
                            .map(|index| [1.5, -128.0, 0.0, -0.0, 63.5, -3.25][(index + token) % 6])
                            .collect()
                    })
                    .collect();
                for backend in [
                    MatvecBackend::Scalar,
                    MatvecBackend::Auto,
                    MatvecBackend::ExperimentalNeon,
                ] {
                    let mut tiled = ProjectionTile::with_backend(backend);
                    for (pass, pinned) in [false, true, false].into_iter().enumerate() {
                        let mut actual = vec![vec![f32::NAN]; count];
                        tiled.project(&tensor, &inputs, &mut actual, pinned, &|| Ok(()))?;
                        assert_eq!(
                            tiled.decoded_blocks,
                            u64::try_from(columns / 128 * 3 * (pass + 1))?
                        );
                        assert!(tiled.preparations.iter().all(|scratch| {
                            scratch.requested_backend() == backend
                                && scratch.pinned_accumulation == pinned
                                && scratch.pq2_consumption.capacity() == 0
                        }));
                        for (input, actual) in inputs.iter().zip(actual) {
                            let mut scratch = if pinned {
                                MatvecScratch::with_pinned_backend(MatvecBackend::Scalar)
                            } else {
                                MatvecScratch::with_backend(MatvecBackend::Scalar)
                            };
                            let mut expected = Vec::new();
                            matvec_into(&tensor, input, &mut expected, &mut scratch)?;
                            assert_eq!(
                                actual.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                                expected.into_iter().map(f32::to_bits).collect::<Vec<_>>()
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn tiled_projection_clears_all_outputs_on_late_rejection() {
        let tensor = Tensor {
            name: "reject".into(),
            dimensions: vec![128, 2],
            kind: 142,
            data: Arc::<[u8]>::from([0_u8; 68]).into(),
        };
        let mut scratch = ProjectionTile::default();
        for count in [0, 8, MAX_TILE_TOKENS + 1] {
            let mut inputs = vec![vec![0.0; 128]; count];
            if count == 8 {
                inputs[7][127] = f32::NAN;
            }
            let mut outputs = vec![vec![1.0]; count];
            assert!(
                scratch
                    .project(&tensor, &inputs, &mut outputs, true, &|| Ok(()))
                    .is_err()
            );
            assert!(outputs.iter().all(Vec::is_empty));
        }
    }

    #[test]
    fn tiled_projection_clears_completed_rows_on_cancel_and_bad_scale() -> Result<()> {
        for backend in [
            MatvecBackend::Scalar,
            MatvecBackend::Auto,
            MatvecBackend::ExperimentalNeon,
        ] {
            for kind in [PQ2, PTQ1] {
                let mut packed = [0x66; 34];
                packed[..2].copy_from_slice(&0x3555_u16.to_le_bytes());
                let block = if kind == PTQ1 {
                    bonsai_ternary::pack_ptq(&packed)?.to_vec()
                } else {
                    packed.to_vec()
                };
                let mut tensor = Tensor {
                    name: "late-tile-error".into(),
                    dimensions: vec![256, 2],
                    kind,
                    data: Arc::<[u8]>::from(block.repeat(4)).into(),
                };
                let inputs = vec![vec![1.5; 256]; 8];
                let mut output = vec![vec![1.0]; 8];
                let mut tile = ProjectionTile::with_backend(backend);
                let calls = Cell::new(0);
                assert!(
                    tile.project(&tensor, &inputs, &mut output, true, &|| {
                        calls.set(calls.get() + 1);
                        // Eight preparations, first row, then cancel at row two.
                        if calls.get() == 10 {
                            Err(invalid("cancel after completed tile row"))
                        } else {
                            Ok(())
                        }
                    })
                    .is_err()
                );
                assert_eq!(tile.decoded_blocks, 2);
                assert!(output.iter().all(Vec::is_empty));
                tile.project(&tensor, &inputs, &mut output, true, &|| Ok(()))?;
                assert!(output.iter().all(|row| row.len() == 2));
                let mut bad = tensor.data.to_vec();
                let offset = block.len() * 3 + if kind == PTQ1 { 26 } else { 0 };
                bad[offset..offset + 2].copy_from_slice(&0x7c00_u16.to_le_bytes());
                tensor.data = Arc::<[u8]>::from(bad).into();
                assert!(
                    tile.project(&tensor, &inputs, &mut output, false, &|| Ok(()))
                        .is_err()
                );
                assert!(output.iter().all(Vec::is_empty));
            }
        }
        Ok(())
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "Cover 8-row and 4-row PQ2/PTQ1 GEMM groups, remainder rows and mid-group error/cancel recovery together."
    )]
    fn tiled_four_row_gemm_matches_scalar_and_recovers_mid_group_failure() -> Result<()> {
        let rows = 13_usize;
        for (kind, columns) in [(PQ2, 256_usize), (PQ2, 384), (PTQ1, 256)] {
            let blocks_per_row = columns / 128;
            let mut data = Vec::new();
            for row in 0..rows {
                for blk in 0..blocks_per_row {
                    let mut raw = [0_u8; 34];
                    let scale = u16::try_from(0x3400_usize + ((row * 3 + blk) % 16) * 0x10)?;
                    raw[..2].copy_from_slice(&scale.to_le_bytes());
                    for (idx, slot) in raw[2..].iter_mut().enumerate() {
                        *slot = if kind == PTQ1 {
                            [0x00, 0x55, 0x66, 0x99, 0xaa][(row + blk + idx) % 5]
                        } else {
                            u8::try_from((row * 29 + blk * 13 + idx * 7) & 0xff)?
                        };
                    }
                    if kind == PTQ1 {
                        data.extend_from_slice(&bonsai_ternary::pack_ptq(&raw)?);
                    } else {
                        data.extend_from_slice(&raw);
                    }
                }
            }
            let tensor = Tensor {
                name: "tile-four-row-gemm".into(),
                dimensions: vec![columns, rows],
                kind,
                data: Arc::<[u8]>::from(data).into(),
            };
            for count in [1, 4, 8, 16, 32] {
                let inputs: Vec<Vec<f32>> = (0..count)
                    .map(|token| {
                        (0..columns)
                            .map(|idx| {
                                [1.5, -64.0, 0.25, -0.0, 31.75, -3.25, 7.125][(idx + token * 3) % 7]
                            })
                            .collect()
                    })
                    .collect();
                for pinned in [false, true] {
                    let expected: Vec<Vec<u32>> = inputs
                        .iter()
                        .map(|input| {
                            let mut scratch = if pinned {
                                MatvecScratch::with_pinned_backend(MatvecBackend::Scalar)
                            } else {
                                MatvecScratch::with_backend(MatvecBackend::Scalar)
                            };
                            let mut out = Vec::new();
                            matvec_into(&tensor, input, &mut out, &mut scratch)?;
                            Ok(out.into_iter().map(f32::to_bits).collect())
                        })
                        .collect::<Result<_>>()?;
                    for backend in [
                        MatvecBackend::Scalar,
                        MatvecBackend::Auto,
                        MatvecBackend::ExperimentalNeon,
                    ] {
                        let mut tile = ProjectionTile::with_backend(backend);
                        let mut actual = vec![Vec::new(); count];
                        tile.project(&tensor, &inputs, &mut actual, pinned, &|| Ok(()))?;
                        assert_eq!(tile.decoded_blocks, u64::try_from(rows * blocks_per_row)?);
                        #[cfg(all(
                            feature = "experimental-neon",
                            target_arch = "aarch64",
                            target_os = "windows",
                            target_env = "msvc",
                            target_feature = "neon"
                        ))]
                        if kind == PQ2
                            && columns.is_multiple_of(256)
                            && backend != MatvecBackend::Scalar
                        {
                            assert!(
                                tile.preparations[..count]
                                    .iter()
                                    .all(|s| s.pq2_consumption.len() == blocks_per_row)
                            );
                        }
                        let actual_bits: Vec<Vec<u32>> = actual
                            .into_iter()
                            .map(|row| row.into_iter().map(f32::to_bits).collect())
                            .collect();
                        assert_eq!(actual_bits, expected);
                    }
                }
            }
            // Verify mid-8-row-group cancellation and mid-group bad scale rewind.
            let inputs = vec![vec![1.25; columns]; 8];
            let mut output = vec![vec![1.0]; 8];
            let mut tile = ProjectionTile::with_backend(MatvecBackend::ExperimentalNeon);
            let calls = Cell::new(0);
            assert!(
                tile.project(&tensor, &inputs, &mut output, true, &|| {
                    calls.set(calls.get() + 1);
                    // Eight preparations + row 0 ok + cancel at row 1 inside first 8-row group.
                    if calls.get() == 10 {
                        Err(invalid("cancel inside 8-row group"))
                    } else {
                        Ok(())
                    }
                })
                .is_err()
            );
            assert_eq!(tile.decoded_blocks, u64::try_from(blocks_per_row)?);
            assert!(output.iter().all(Vec::is_empty));

            let block_bytes = if kind == PTQ1 { 28 } else { 34 };
            let mut bad = tensor.data.to_vec();
            // Corrupt row 2 (third row inside the first 8-row group).
            let offset = (2 * blocks_per_row) * block_bytes + if kind == PTQ1 { 26 } else { 0 };
            bad[offset..offset + 2].copy_from_slice(&0x7c00_u16.to_le_bytes());
            let bad_tensor = Tensor {
                data: Arc::<[u8]>::from(bad).into(),
                ..tensor
            };
            let mut bad_tile = ProjectionTile::with_backend(MatvecBackend::ExperimentalNeon);
            assert!(
                bad_tile
                    .project(&bad_tensor, &inputs, &mut output, true, &|| Ok(()))
                    .is_err()
            );
            assert_eq!(bad_tile.decoded_blocks, u64::try_from(2 * blocks_per_row)?);
            assert!(output.iter().all(Vec::is_empty));
        }
        Ok(())
    }

    #[test]
    fn tiled_bf16_gemm_matches_scalar_and_reuses_prepared_activations() -> Result<()> {
        let columns = 20_usize;
        let rows = 13_usize;
        let mut data = Vec::with_capacity(columns * rows * 2);
        for row in 0..rows {
            for col in 0..columns {
                let bits = u16::try_from(0x3c00_usize + ((row * 5 + col * 3) % 64) * 4)?;
                data.extend_from_slice(&bits.to_le_bytes());
            }
        }
        let tensor = Tensor {
            name: "tile-bf16-gemm".into(),
            dimensions: vec![columns, rows],
            kind: BF16,
            data: Arc::<[u8]>::from(data).into(),
        };
        let inputs: Vec<Vec<f32>> = (0..8)
            .map(|token| {
                (0..columns)
                    .map(|idx| [1.25, -3.5, 0.0, -0.0, 7.125, -0.375][(idx + token) % 6])
                    .collect()
            })
            .collect();
        let expected: Vec<Vec<u32>> = inputs
            .iter()
            .map(|input| {
                let mut scratch = MatvecScratch::with_backend(MatvecBackend::Scalar);
                let mut out = Vec::new();
                matvec_into(&tensor, input, &mut out, &mut scratch)?;
                Ok(out.into_iter().map(f32::to_bits).collect())
            })
            .collect::<Result<_>>()?;
        for backend in [
            MatvecBackend::Scalar,
            MatvecBackend::Auto,
            MatvecBackend::ExperimentalNeon,
        ] {
            let mut tile = ProjectionTile::with_backend(backend);
            #[cfg(feature = "parallel-rows")]
            tile.set_row_parallelism(RowParallelism::new(2)?);
            let mut actual = vec![Vec::new(); inputs.len()];
            tile.project(&tensor, &inputs, &mut actual, true, &|| Ok(()))?;
            assert_eq!(tile.packed_projections, 0);
            assert_eq!(tile.decoded_blocks, 0);
            let actual_bits: Vec<Vec<u32>> = actual
                .iter()
                .map(|row| row.iter().copied().map(f32::to_bits).collect())
                .collect();
            assert_eq!(actual_bits, expected);
            let preps_before: Vec<usize> = tile
                .preparations
                .iter()
                .map(MatvecScratch::preparation_count)
                .collect();
            tile.project_with_previous(
                Some(&tensor),
                &tensor,
                &inputs,
                &mut actual,
                true,
                &|| Ok(()),
            )?;
            let preps_after: Vec<usize> = tile
                .preparations
                .iter()
                .map(MatvecScratch::preparation_count)
                .collect();
            assert_eq!(preps_after, preps_before);
        }
        Ok(())
    }

    #[test]
    fn tiled_quantized_projections_match_scalar_and_reuse_prepared_activations() -> Result<()> {
        let columns = 256_usize;
        let rows = 19_usize;
        let inputs: Vec<Vec<f32>> = (0..9)
            .map(|token| {
                (0..columns)
                    .map(|idx| [1.25, -3.5, 0.0, -0.0, 7.125, -0.375][(idx + token) % 6])
                    .collect()
            })
            .collect();
        for kind in [Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, 12, 14, 34, 35, 39, 41, 42] {
            let mut data = Vec::new();
            for row in 0..rows {
                let row_f32: Vec<f32> = (0..columns)
                    .map(|col| {
                        let s = (row * 17 + col * 11) % 31;
                        (f32::from(u8::try_from(s).unwrap_or(0)) - 15.0) * 0.125
                    })
                    .collect();
                data.extend_from_slice(&super::super::quantize_row(kind, &row_f32)?);
            }
            let tensor = Tensor {
                name: format!("tile-quant-{kind}"),
                dimensions: vec![columns, rows],
                kind,
                data: Arc::<[u8]>::from(data).into(),
            };
            let expected: Vec<Vec<u32>> = inputs
                .iter()
                .map(|input| {
                    let mut scratch = MatvecScratch::with_backend(MatvecBackend::Scalar);
                    let mut out = Vec::new();
                    matvec_into(&tensor, input, &mut out, &mut scratch)?;
                    Ok(out.into_iter().map(f32::to_bits).collect())
                })
                .collect::<Result<_>>()?;
            for backend in [
                MatvecBackend::Scalar,
                MatvecBackend::Auto,
                MatvecBackend::ExperimentalNeon,
            ] {
                let mut tile = ProjectionTile::with_backend(backend);
                #[cfg(feature = "parallel-rows")]
                tile.set_row_parallelism(RowParallelism::new(2)?);
                let mut actual = vec![Vec::new(); inputs.len()];
                tile.project(&tensor, &inputs, &mut actual, true, &|| Ok(()))?;
                assert_eq!(tile.packed_projections, 0);
                assert_eq!(tile.decoded_blocks, 0);
                let actual_bits: Vec<Vec<u32>> = actual
                    .iter()
                    .map(|row| row.iter().copied().map(f32::to_bits).collect())
                    .collect();
                assert_eq!(actual_bits, expected, "mismatch for kind={kind}");
                let preps_before: Vec<usize> = tile
                    .preparations
                    .iter()
                    .map(MatvecScratch::preparation_count)
                    .collect();
                tile.project_with_previous(
                    Some(&tensor),
                    &tensor,
                    &inputs,
                    &mut actual,
                    true,
                    &|| Ok(()),
                )?;
                let preps_after: Vec<usize> = tile
                    .preparations
                    .iter()
                    .map(MatvecScratch::preparation_count)
                    .collect();
                assert_eq!(preps_after, preps_before, "prep not reused for kind={kind}");
            }
        }
        Ok(())
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "Regression verifies 32->4->8 stale-slot, pinned, tensor-id, and error invalidation together."
    )]
    fn projection_tile_invalidates_stale_tokens_pinned_tensor_and_errors() -> Result<()> {
        let columns = 256_usize;
        let rows = 8_usize;
        let make_pq2 = |name: &str, seed: u8| -> Tensor {
            let mut data = Vec::with_capacity(rows * (columns / 128) * 34);
            for row in 0..rows {
                for block_idx in 0..(columns / 128) {
                    let mut block = [0_u8; 34];
                    block[..2].copy_from_slice(&0x3555_u16.to_le_bytes());
                    for byte in &mut block[2..] {
                        *byte = seed.wrapping_add(u8::try_from(row + block_idx).unwrap_or(0));
                    }
                    data.extend_from_slice(&block);
                }
            }
            Tensor {
                name: name.to_owned(),
                dimensions: vec![columns, rows],
                kind: PQ2,
                data: Arc::<[u8]>::from(data).into(),
            }
        };
        let t0 = make_pq2("t0", 0x1b);
        let t1 = make_pq2("t1", 0x72);
        let t2 = make_pq2("t2", 0xe4);
        let make_inputs = |count: usize, offset: usize| -> Vec<Vec<f32>> {
            (0..count)
                .map(|token| {
                    (0..columns)
                        .map(|idx| {
                            [1.25, -3.5, 0.5, -2.25, 7.125, -0.375][(idx + token + offset) % 6]
                        })
                        .collect()
                })
                .collect()
        };
        let scalar_bits =
            |tensor: &Tensor, inputs: &[Vec<f32>], pinned: bool| -> Result<Vec<Vec<u32>>> {
                inputs
                    .iter()
                    .map(|input| {
                        let mut scratch = if pinned {
                            MatvecScratch::with_pinned_backend(MatvecBackend::Scalar)
                        } else {
                            MatvecScratch::with_backend(MatvecBackend::Scalar)
                        };
                        let mut out = Vec::new();
                        matvec_into(tensor, input, &mut out, &mut scratch)?;
                        Ok(out.into_iter().map(f32::to_bits).collect())
                    })
                    .collect()
            };
        let inputs_32 = make_inputs(32, 0);
        let inputs_4 = make_inputs(4, 100);
        let inputs_8 = make_inputs(8, 200);
        let inputs_8_alt = make_inputs(8, 300);

        let mut tile = ProjectionTile::with_backend(MatvecBackend::Scalar);
        let mut out32 = vec![Vec::new(); 32];
        tile.project(&t0, &inputs_32, &mut out32, true, &|| Ok(()))?;
        assert_eq!(tile.preparations.len(), 32);

        let mut out4 = vec![Vec::new(); 4];
        tile.project(&t1, &inputs_4, &mut out4, true, &|| Ok(()))?;
        let out4_bits: Vec<Vec<u32>> = out4
            .iter()
            .map(|r| r.iter().copied().map(f32::to_bits).collect())
            .collect();
        assert_eq!(out4_bits, scalar_bits(&t1, &inputs_4, true)?);

        // 32 -> 4 -> 8 regression: must NOT reuse stale entries in slots 4..8.
        let mut out8 = vec![Vec::new(); 8];
        tile.project_with_previous(Some(&t1), &t2, &inputs_8, &mut out8, true, &|| Ok(()))?;
        let out8_bits: Vec<Vec<u32>> = out8
            .iter()
            .map(|r| r.iter().copied().map(f32::to_bits).collect())
            .collect();
        assert_eq!(out8_bits, scalar_bits(&t2, &inputs_8, true)?);

        // Pinning mode mismatch must invalidate cached preparations.
        let mut out8_unpinned = vec![Vec::new(); 8];
        let preps_before: Vec<usize> = tile
            .preparations
            .iter()
            .take(8)
            .map(MatvecScratch::preparation_count)
            .collect();
        tile.project_with_previous(
            Some(&t2),
            &t0,
            &inputs_8_alt,
            &mut out8_unpinned,
            false,
            &|| Ok(()),
        )?;
        let preps_after: Vec<usize> = tile
            .preparations
            .iter()
            .take(8)
            .map(MatvecScratch::preparation_count)
            .collect();
        assert!(
            preps_before
                .iter()
                .zip(&preps_after)
                .all(|(before, after)| *after > *before)
        );
        let unpinned_bits: Vec<Vec<u32>> = out8_unpinned
            .iter()
            .map(|r| r.iter().copied().map(f32::to_bits).collect())
            .collect();
        assert_eq!(unpinned_bits, scalar_bits(&t0, &inputs_8_alt, false)?);

        // Tensor identity mismatch (`t1` was not last prepared; `t0` was) must invalidate.
        tile.project_with_previous(Some(&t1), &t2, &inputs_8, &mut out8, false, &|| Ok(()))?;
        let mismatched_bits: Vec<Vec<u32>> = out8
            .iter()
            .map(|r| r.iter().copied().map(f32::to_bits).collect())
            .collect();
        assert_eq!(mismatched_bits, scalar_bits(&t2, &inputs_8, false)?);

        // Failure invalidation: failed call must clear valid_tokens so subsequent call re-prepares.
        assert!(
            tile.project_with_previous(Some(&t2), &t0, &inputs_8, &mut out8, false, &|| Err(
                invalid("cancel")
            ))
            .is_err()
        );
        tile.project_with_previous(Some(&t2), &t0, &inputs_8_alt, &mut out8, false, &|| Ok(()))?;
        let recovered_bits: Vec<Vec<u32>> = out8
            .iter()
            .map(|r| r.iter().copied().map(f32::to_bits).collect())
            .collect();
        assert_eq!(recovered_bits, scalar_bits(&t0, &inputs_8_alt, false)?);
        Ok(())
    }

    fn assert_tiled_16_32_rejects_nonfinite(
        kind: u32,
        columns: usize,
        tensor: &mut Tensor,
        max_inputs: &[Vec<f32>],
    ) {
        let block_bytes = if kind == PTQ1 { 28 } else { 34 };
        let scale_off = if kind == PTQ1 { 26 } else { 0 };
        let valid = tensor.data.clone();
        let mut bad_weight = valid.to_vec();
        let corrupt_off = 3 * (columns / if kind == PTQ1 { 128 } else { 32 }) * block_bytes
            + block_bytes
            + scale_off;
        bad_weight[corrupt_off..corrupt_off + 2].copy_from_slice(&0x7e00_u16.to_le_bytes());
        tensor.data = Arc::<[u8]>::from(bad_weight).into();
        for count in [16_usize, 32] {
            let mut tile = ProjectionTile::with_backend(MatvecBackend::ExperimentalNeon);
            let mut out = vec![vec![1.0_f32]; count];
            assert!(
                tile.project(
                    tensor,
                    &max_inputs[..count],
                    &mut out,
                    kind == PTQ1,
                    &|| Ok(())
                )
                .is_err()
            );
            assert!(out.iter().all(Vec::is_empty));
        }

        tensor.data = valid;
        for count in [16_usize, 32] {
            let mut bad_inputs = max_inputs[..count].to_vec();
            bad_inputs[count - 3][columns - 1] = f32::NAN;
            let mut tile = ProjectionTile::with_backend(MatvecBackend::ExperimentalNeon);
            let mut out = vec![vec![1.0_f32]; count];
            assert!(
                tile.project(tensor, &bad_inputs, &mut out, kind == PTQ1, &|| Ok(()))
                    .is_err()
            );
            assert!(out.iter().all(Vec::is_empty));
        }
    }

    #[test]
    fn tiled_ptq1_and_q8_0_16_and_32_token_tiles_match_scalar_bits_and_reject_nonfinite()
    -> Result<()> {
        for (kind, columns, rows) in [(PTQ1, 256_usize, 19_usize), (Q8_0, 192_usize, 21_usize)] {
            let mut data = Vec::new();
            for row in 0..rows {
                let values: Vec<f32> = (0..columns)
                    .map(|col| {
                        [1.25, -2.5, 0.0, -0.0, 0.5, -0.375, 3.0, -0.125][(col * 5 + row * 3) % 8]
                    })
                    .collect();
                data.extend_from_slice(&super::super::quantize_row(kind, &values)?);
            }
            let mut tensor = Tensor {
                name: format!("tile-16-32-{kind}"),
                dimensions: vec![columns, rows],
                kind,
                data: Arc::<[u8]>::from(data).into(),
            };
            let max_inputs: Vec<Vec<f32>> = (0..32)
                .map(|token| {
                    (0..columns)
                        .map(|col| {
                            [1.5, -128.0, 0.0, -0.0, 63.5, -3.25, 7.125][(col + token * 7) % 7]
                        })
                        .collect()
                })
                .collect();

            for pinned in [false, true] {
                if kind == Q8_0 && pinned {
                    continue;
                }
                let expected: Vec<Vec<u32>> = max_inputs
                    .iter()
                    .map(|input| {
                        let mut scratch = if pinned {
                            MatvecScratch::with_pinned_backend(MatvecBackend::Scalar)
                        } else {
                            MatvecScratch::with_backend(MatvecBackend::Scalar)
                        };
                        let mut out = Vec::new();
                        matvec_into(&tensor, input, &mut out, &mut scratch)?;
                        Ok(out.into_iter().map(f32::to_bits).collect())
                    })
                    .collect::<Result<_>>()?;

                for count in [16_usize, 19, 24, 32] {
                    let inputs = &max_inputs[..count];
                    for backend in [
                        MatvecBackend::Scalar,
                        MatvecBackend::Auto,
                        MatvecBackend::ExperimentalNeon,
                    ] {
                        let mut tile = ProjectionTile::with_backend(backend);
                        let mut actual = vec![vec![f32::NAN]; count];
                        tile.project(&tensor, inputs, &mut actual, pinned, &|| Ok(()))?;
                        let actual_bits: Vec<Vec<u32>> = actual
                            .iter()
                            .map(|row| row.iter().copied().map(f32::to_bits).collect())
                            .collect();
                        assert_eq!(
                            actual_bits,
                            expected[..count],
                            "mismatch kind={kind} count={count} pinned={pinned}"
                        );
                    }
                }
            }

            assert_tiled_16_32_rejects_nonfinite(kind, columns, &mut tensor, &max_inputs);
        }
        Ok(())
    }

    fn make_f32_tensor(name: &str, columns: usize, rows: usize, seed: usize) -> Tensor {
        let mut data = Vec::with_capacity(columns * rows * 4);
        for row in 0..rows {
            for col in 0..columns {
                let value = [
                    0.0_f32,
                    -0.0,
                    f32::from_bits(1),
                    -f32::from_bits(1),
                    1.25,
                    -3.5,
                    7.125,
                    -0.375,
                    0.0625,
                ][(row * 7 + col * 5 + seed) % 9];
                data.extend_from_slice(&value.to_le_bytes());
            }
        }
        Tensor {
            name: name.to_owned(),
            dimensions: vec![columns, rows],
            kind: F32,
            data: Arc::<[u8]>::from(data).into(),
        }
    }

    fn make_f32_inputs(count: usize, columns: usize, seed: usize) -> Vec<Vec<f32>> {
        (0..count)
            .map(|token| {
                (0..columns)
                    .map(|col| {
                        [1.5_f32, -128.0, 0.0, -0.0, 63.5, -3.25, 0.125, -0.5]
                            [(col + token * 3 + seed) % 8]
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "Verify 0 ULP F32 parity across all 1..=32 token counts, shapes, backends, pinned modes, serial/parallel rows, and sibling pairs."
    )]
    fn tiled_f32_single_and_siblings_match_scalar_zero_ulp_across_tokens_and_workers() -> Result<()>
    {
        #[cfg(feature = "parallel-rows")]
        let worker_counts = [1_usize, 4];
        #[cfg(not(feature = "parallel-rows"))]
        let worker_counts = [1_usize];

        for (columns, rows_a, rows_b) in [
            (1_usize, 9_usize, 5_usize),
            (3, 13, 4),
            (5, 8, 1),
            (16, 17, 9),
            (17, 13, 5),
            (32, 21, 12),
            (35, 9, 4),
        ] {
            let tensor_a = make_f32_tensor("f32-a", columns, rows_a, 1);
            let tensor_b = make_f32_tensor("f32-b", columns, rows_b, 2);
            let max_inputs = make_f32_inputs(32, columns, 0);

            for pinned in [false, true] {
                let expected_a: Vec<Vec<u32>> = max_inputs
                    .iter()
                    .map(|input| {
                        let mut scratch = if pinned {
                            MatvecScratch::with_pinned_backend(MatvecBackend::Scalar)
                        } else {
                            MatvecScratch::with_backend(MatvecBackend::Scalar)
                        };
                        let mut out = Vec::new();
                        matvec_into(&tensor_a, input, &mut out, &mut scratch)?;
                        Ok(out.into_iter().map(f32::to_bits).collect())
                    })
                    .collect::<Result<_>>()?;
                let expected_b: Vec<Vec<u32>> = max_inputs
                    .iter()
                    .map(|input| {
                        let mut scratch = if pinned {
                            MatvecScratch::with_pinned_backend(MatvecBackend::Scalar)
                        } else {
                            MatvecScratch::with_backend(MatvecBackend::Scalar)
                        };
                        let mut out = Vec::new();
                        matvec_into(&tensor_b, input, &mut out, &mut scratch)?;
                        Ok(out.into_iter().map(f32::to_bits).collect())
                    })
                    .collect::<Result<_>>()?;

                let counts: &[usize] = if columns == 16 || columns == 17 {
                    &[
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                    ]
                } else {
                    &[1, 2, 3, 4, 7, 8, 9, 16, 17, 24, 31, 32]
                };

                for &count in counts {
                    let inputs = &max_inputs[..count];
                    for &workers in &worker_counts {
                        let parallelism = RowParallelism::new(workers)?;
                        for backend in [
                            MatvecBackend::Scalar,
                            MatvecBackend::Auto,
                            MatvecBackend::ExperimentalNeon,
                        ] {
                            let mut tile = ProjectionTile::with_backend(backend);
                            tile.set_row_parallelism(parallelism.clone());

                            // Single projection (project_tile path) + reuse on second tensor.
                            let mut out_a = vec![vec![f32::NAN]; count];
                            tile.project_with_previous(
                                None,
                                &tensor_a,
                                inputs,
                                &mut out_a,
                                pinned,
                                &|| Ok(()),
                            )?;
                            assert_eq!(tile.packed_projections, 0);
                            assert_eq!(tile.decoded_blocks, 0);
                            let bits_a: Vec<Vec<u32>> = out_a
                                .iter()
                                .map(|row| row.iter().copied().map(f32::to_bits).collect())
                                .collect();
                            assert_eq!(
                                bits_a,
                                expected_a[..count],
                                "single f32 mismatch cols={columns} count={count} pinned={pinned} workers={workers}"
                            );

                            let preps_before: Vec<usize> = tile
                                .preparations
                                .iter()
                                .take(count)
                                .map(MatvecScratch::preparation_count)
                                .collect();
                            let mut out_b = vec![vec![f32::NAN]; count];
                            tile.project_with_previous(
                                Some(&tensor_a),
                                &tensor_b,
                                inputs,
                                &mut out_b,
                                pinned,
                                &|| Ok(()),
                            )?;
                            let preps_after: Vec<usize> = tile
                                .preparations
                                .iter()
                                .take(count)
                                .map(MatvecScratch::preparation_count)
                                .collect();
                            assert_eq!(preps_after, preps_before);
                            let bits_b: Vec<Vec<u32>> = out_b
                                .iter()
                                .map(|row| row.iter().copied().map(f32::to_bits).collect())
                                .collect();
                            assert_eq!(
                                bits_b,
                                expected_b[..count],
                                "reused f32 mismatch cols={columns} count={count} pinned={pinned} workers={workers}"
                            );

                            // Sibling pair projection (project_tile_pair path).
                            assert!(ProjectionTile::supports_siblings(&[&tensor_a, &tensor_b]));
                            let mut sib_a = vec![vec![f32::NAN]; count];
                            let mut sib_b = vec![vec![f32::NAN]; count];
                            tile.project_siblings_with_previous(
                                Some(&tensor_b),
                                &[&tensor_a, &tensor_b],
                                inputs,
                                &mut [&mut sib_a, &mut sib_b],
                                pinned,
                                &|| Ok(()),
                            )?;
                            let sib_bits_a: Vec<Vec<u32>> = sib_a
                                .iter()
                                .map(|row| row.iter().copied().map(f32::to_bits).collect())
                                .collect();
                            let sib_bits_b: Vec<Vec<u32>> = sib_b
                                .iter()
                                .map(|row| row.iter().copied().map(f32::to_bits).collect())
                                .collect();
                            assert_eq!(sib_bits_a, expected_a[..count]);
                            assert_eq!(sib_bits_b, expected_b[..count]);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "Tests signed-zero preservation, unaligned rows, and nonfinite rejection across shapes."
    )]
    fn tiled_f32_preserves_signed_zero_unaligned_bytes_and_rejects_nonfinite() -> Result<()> {
        for columns in [1_usize, 3, 16, 19] {
            let row_bytes = columns * 4;
            let mut neg_zero_bytes = Vec::with_capacity(9 * row_bytes);
            for _ in 0..9 * columns {
                neg_zero_bytes.extend_from_slice(&(-0.0_f32).to_le_bytes());
            }
            let neg_tensor = Tensor {
                name: "f32-neg-zero".into(),
                dimensions: vec![columns, 9],
                kind: F32,
                data: Arc::<[u8]>::from(neg_zero_bytes).into(),
            };
            let pos_inputs = vec![vec![1.0_f32; columns]; 9];
            for pinned in [false, true] {
                let mut scalar_scratch = if pinned {
                    MatvecScratch::with_pinned_backend(MatvecBackend::Scalar)
                } else {
                    MatvecScratch::with_backend(MatvecBackend::Scalar)
                };
                let mut expected = Vec::new();
                matvec_into(
                    &neg_tensor,
                    &pos_inputs[0],
                    &mut expected,
                    &mut scalar_scratch,
                )?;
                let expected_bits: Vec<u32> = expected.into_iter().map(f32::to_bits).collect();
                for backend in [MatvecBackend::Scalar, MatvecBackend::ExperimentalNeon] {
                    let mut tile = ProjectionTile::with_backend(backend);
                    let mut outputs = vec![Vec::new(); 9];
                    tile.project(&neg_tensor, &pos_inputs, &mut outputs, pinned, &|| Ok(()))?;
                    for out in &outputs {
                        let bits: Vec<u32> = out.iter().copied().map(f32::to_bits).collect();
                        assert_eq!(bits, expected_bits);
                    }
                }
            }

            // Unaligned `&[u8]` slice check directly on dot_f32_eight_rows / four_rows / row.
            let mut unaligned = vec![0xaa_u8; 1 + 8 * row_bytes];
            for row in 0..8 {
                for col in 0..columns {
                    let val = [1.25_f32, -2.5, -0.0, 0.5, -0.125][(row + col) % 5];
                    let off = 1 + row * row_bytes + col * 4;
                    unaligned[off..off + 4].copy_from_slice(&val.to_le_bytes());
                }
            }
            let row_slices: [&[u8]; 8] = std::array::from_fn(|r| {
                let start = 1 + r * row_bytes;
                &unaligned[start..start + row_bytes]
            });
            let mut preps = Vec::with_capacity(9);
            for input in &make_f32_inputs(9, columns, 7) {
                let mut s = MatvecScratch::with_pinned_backend(MatvecBackend::Scalar);
                let _ = s.prepare(&neg_tensor, input)?;
                preps.push(s);
            }
            for pinned in [false, true] {
                let scalar8 = dot::dot_f32_eight_rows::<MAX_TILE_TOKENS>(
                    row_slices,
                    &preps,
                    columns,
                    pinned,
                    MatvecBackend::Scalar,
                )?;
                let neon8 = dot::dot_f32_eight_rows::<MAX_TILE_TOKENS>(
                    row_slices,
                    &preps,
                    columns,
                    pinned,
                    MatvecBackend::ExperimentalNeon,
                )?;
                for r in 0..8 {
                    assert_eq!(
                        neon8[r][..9]
                            .iter()
                            .copied()
                            .map(f32::to_bits)
                            .collect::<Vec<_>>(),
                        scalar8[r][..9]
                            .iter()
                            .copied()
                            .map(f32::to_bits)
                            .collect::<Vec<_>>()
                    );
                }
                let neon4 = dot::dot_f32_four_rows::<MAX_TILE_TOKENS>(
                    [row_slices[0], row_slices[1], row_slices[2], row_slices[3]],
                    &preps,
                    columns,
                    pinned,
                    MatvecBackend::ExperimentalNeon,
                )?;
                let neon1 = dot::dot_f32_row::<MAX_TILE_TOKENS>(
                    row_slices[7],
                    &preps,
                    columns,
                    pinned,
                    MatvecBackend::ExperimentalNeon,
                )?;
                for r in 0..4 {
                    assert_eq!(
                        neon4[r][..9]
                            .iter()
                            .copied()
                            .map(f32::to_bits)
                            .collect::<Vec<_>>(),
                        scalar8[r][..9]
                            .iter()
                            .copied()
                            .map(f32::to_bits)
                            .collect::<Vec<_>>()
                    );
                }
                assert_eq!(
                    neon1[..9]
                        .iter()
                        .copied()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>(),
                    scalar8[7][..9]
                        .iter()
                        .copied()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>()
                );
            }

            // Reject nonfinite weights even when activations are all zero, and recover atomically.
            let mut bad_bytes = neg_tensor.data.to_vec();
            let last_elem = bad_bytes.len() - 4;
            bad_bytes[last_elem..].copy_from_slice(&f32::INFINITY.to_le_bytes());
            let bad_tensor = Tensor {
                name: "f32-bad".into(),
                dimensions: vec![columns, 9],
                kind: F32,
                data: Arc::<[u8]>::from(bad_bytes).into(),
            };
            let zero_inputs = vec![vec![0.0_f32; columns]; 9];
            for backend in [MatvecBackend::Scalar, MatvecBackend::ExperimentalNeon] {
                for pinned in [false, true] {
                    let mut tile = ProjectionTile::with_backend(backend);
                    let mut outs = vec![vec![1.0_f32]; 9];
                    assert!(
                        tile.project(&bad_tensor, &zero_inputs, &mut outs, pinned, &|| Ok(()))
                            .is_err()
                    );
                    assert!(outs.iter().all(Vec::is_empty));
                    tile.project(&neg_tensor, &zero_inputs, &mut outs, pinned, &|| Ok(()))?;
                    assert!(outs.iter().all(|row| row.len() == 9));

                    let mut sib_a = vec![vec![1.0_f32]; 9];
                    let mut sib_b = vec![vec![1.0_f32]; 9];
                    assert!(
                        tile.project_siblings_with_previous(
                            None,
                            &[&neg_tensor, &bad_tensor],
                            &zero_inputs,
                            &mut [&mut sib_a, &mut sib_b],
                            pinned,
                            &|| Ok(()),
                        )
                        .is_err()
                    );
                    assert!(sib_a.iter().chain(&sib_b).all(Vec::is_empty));
                }
            }
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Verify 64-token tiled projections across BF16 float64x2, F32, PQ2, PTQ1, and Q8_0 against scalar reference."
    )]
    #[test]
    fn tiled_64_token_gemm_and_bf16_match_scalar_zero_ulp() -> Result<()> {
        let scalar_bits_ref =
            |tensor: &Tensor, inputs: &[Vec<f32>], pinned: bool| -> Result<Vec<Vec<u32>>> {
                inputs
                    .iter()
                    .map(|input| {
                        let mut s = if pinned {
                            MatvecScratch::with_pinned_backend(MatvecBackend::Scalar)
                        } else {
                            MatvecScratch::with_backend(MatvecBackend::Scalar)
                        };
                        let mut out = Vec::new();
                        matvec_into(tensor, input, &mut out, &mut s)?;
                        Ok(out.into_iter().map(f32::to_bits).collect())
                    })
                    .collect()
            };

        for &(columns, rows) in &[(20_usize, 13_usize), (67, 17)] {
            let mut bf16_data = Vec::with_capacity(columns * rows * 2);
            for row in 0..rows {
                for col in 0..columns {
                    let bits = u16::try_from(0x3c00_usize + ((row * 7 + col * 5) % 64) * 4)?;
                    bf16_data.extend_from_slice(&bits.to_le_bytes());
                }
            }
            let bf16_tensor = Tensor {
                name: "tile-bf16-64".into(),
                dimensions: vec![columns, rows],
                kind: BF16,
                data: Arc::<[u8]>::from(bf16_data).into(),
            };
            for count in [1_usize, 5, 17, 48, MAX_TILE_TOKENS] {
                let inputs: Vec<Vec<f32>> = (0..count)
                    .map(|tok| {
                        (0..columns)
                            .map(|idx| [1.25, -3.5, 0.0, -0.0, 7.125, -0.375][(idx + tok * 3) % 6])
                            .collect()
                    })
                    .collect();
                let expected = scalar_bits_ref(&bf16_tensor, &inputs, true)?;
                for backend in [MatvecBackend::Scalar, MatvecBackend::ExperimentalNeon] {
                    let mut tile = ProjectionTile::with_backend(backend);
                    #[cfg(feature = "parallel-rows")]
                    tile.set_row_parallelism(RowParallelism::new(4)?);
                    let mut actual = vec![Vec::new(); count];
                    tile.project(&bf16_tensor, &inputs, &mut actual, true, &|| Ok(()))?;
                    let actual_bits: Vec<Vec<u32>> = actual
                        .iter()
                        .map(|row| row.iter().copied().map(f32::to_bits).collect())
                        .collect();
                    assert_eq!(actual_bits, expected, "bf16 count={count} cols={columns}");
                }
            }
        }

        let columns = 256_usize;
        let rows = 9_usize;
        let f32_tensor = make_f32_tensor("f32-64", columns, rows, 31);
        let f32_inputs = make_f32_inputs(MAX_TILE_TOKENS, columns, 19);
        for pinned in [false, true] {
            let expected = scalar_bits_ref(&f32_tensor, &f32_inputs, pinned)?;
            let mut tile = ProjectionTile::with_backend(MatvecBackend::ExperimentalNeon);
            #[cfg(feature = "parallel-rows")]
            tile.set_row_parallelism(RowParallelism::new(4)?);
            let mut actual = vec![Vec::new(); MAX_TILE_TOKENS];
            tile.project(&f32_tensor, &f32_inputs, &mut actual, pinned, &|| Ok(()))?;
            let actual_bits: Vec<Vec<u32>> = actual
                .iter()
                .map(|r| r.iter().copied().map(f32::to_bits).collect())
                .collect();
            assert_eq!(actual_bits, expected, "f32 64-tok pinned={pinned}");
        }

        for kind in [PQ2, PTQ1, Q8_0] {
            let block_width = if kind == Q8_0 { 32 } else { 128 };
            let blocks_per_row = columns / block_width;
            let mut data = Vec::new();
            for row in 0..rows {
                for blk in 0..blocks_per_row {
                    let mut raw = [0_u8; 34];
                    let scale = u16::try_from(0x3400_usize + ((row * 3 + blk) % 16) * 0x10)?;
                    raw[..2].copy_from_slice(&scale.to_le_bytes());
                    for (idx, slot) in raw[2..].iter_mut().enumerate() {
                        *slot = if kind == PTQ1 {
                            [0x00, 0x55, 0x66, 0x99, 0xaa][(row + blk + idx) % 5]
                        } else {
                            u8::try_from((row * 29 + blk * 13 + idx * 7) & 0xff)?
                        };
                    }
                    if kind == PTQ1 {
                        data.extend_from_slice(&bonsai_ternary::pack_ptq(&raw)?);
                    } else {
                        data.extend_from_slice(&raw);
                    }
                }
            }
            let tensor = Tensor {
                name: format!("tile-64-kind-{kind}"),
                dimensions: vec![columns, rows],
                kind,
                data: Arc::<[u8]>::from(data).into(),
            };
            let inputs = make_f32_inputs(MAX_TILE_TOKENS, columns, 23);
            for pinned in [false, true] {
                let expected = scalar_bits_ref(&tensor, &inputs, pinned)?;
                let mut tile = ProjectionTile::with_backend(MatvecBackend::ExperimentalNeon);
                #[cfg(feature = "parallel-rows")]
                tile.set_row_parallelism(RowParallelism::new(4)?);
                let mut actual = vec![Vec::new(); MAX_TILE_TOKENS];
                tile.project(&tensor, &inputs, &mut actual, pinned, &|| Ok(()))?;
                let actual_bits: Vec<Vec<u32>> = actual
                    .iter()
                    .map(|r| r.iter().copied().map(f32::to_bits).collect())
                    .collect();
                assert_eq!(actual_bits, expected, "kind={kind} 64-tok pinned={pinned}");
            }
        }
        Ok(())
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "Verify F16 matvec and 64-token tiled GEMM across unpinned float_dot, pinned_gdn::dot, odd/tail columns, and NaN/Inf recovery."
    )]
    fn tiled_f16_gemm_matches_scalar_zero_ulp_and_pinned_contract() -> Result<()> {
        let make_f16_tensor = |columns: usize, rows: usize| -> Result<Tensor> {
            let bits = [
                0x0000_u16, 0x8000, 0x0001, 0x8001, 0x3c00, 0xbc00, 0x3e00, 0x4200,
            ];
            let mut data = Vec::with_capacity(columns * rows * 2);
            for row in 0..rows {
                for col in 0..columns {
                    data.extend_from_slice(&bits[(row * 5 + col * 3) % bits.len()].to_le_bytes());
                }
            }
            Ok(Tensor {
                name: format!("tile-f16-{columns}x{rows}"),
                dimensions: vec![columns, rows],
                kind: F16,
                data: Arc::<[u8]>::from(data).into(),
            })
        };
        let make_inputs = |count: usize, columns: usize| -> Vec<Vec<f32>> {
            let vals = [0.0_f32, -0.0, 1.25, -3.5, 0.125, 7.125, -0.375, -0.0625];
            (0..count)
                .map(|tok| {
                    (0..columns)
                        .map(|idx| vals[(idx + tok * 3) % vals.len()])
                        .collect()
                })
                .collect()
        };

        for &(columns, rows) in &[
            (1_usize, 5_usize),
            (7, 1),
            (15, 4),
            (16, 13),
            (17, 9),
            (20, 13),
            (67, 17),
            (256, 13),
        ] {
            let tensor = make_f16_tensor(columns, rows)?;
            let max_tokens = if columns >= 16 { MAX_TILE_TOKENS } else { 17 };
            let all_inputs = make_inputs(max_tokens, columns);
            for pinned in [false, true] {
                let mut ref_row = Vec::new();
                for count in [1_usize, 5, 17, max_tokens] {
                    let inputs = &all_inputs[..count];
                    let expected: Vec<Vec<u32>> = inputs
                        .iter()
                        .map(|input| {
                            let mut s = if pinned {
                                MatvecScratch::with_pinned_backend(MatvecBackend::Scalar)
                            } else {
                                MatvecScratch::with_backend(MatvecBackend::Scalar)
                            };
                            let mut out = Vec::new();
                            matvec_into(&tensor, input, &mut out, &mut s)?;
                            // Also verify explicit contract against float_dot vs pinned_gdn::dot.
                            let rounded_act: Vec<f32> =
                                input.iter().map(|&v| super::super::round_f16(v)).collect();
                            for r in 0..rows {
                                super::super::row_into(&tensor, r, &mut ref_row)?;
                                let ref_val = if pinned && columns.is_multiple_of(16) {
                                    super::super::pinned_gdn::dot(&ref_row, &rounded_act)?
                                } else {
                                    finite(super::super::float_dot(&ref_row, &rounded_act))?
                                };
                                assert_eq!(
                                    out[r].to_bits(),
                                    ref_val.to_bits(),
                                    "scalar matvec contract cols={columns} rows={rows} r={r} pinned={pinned}"
                                );
                            }
                            Ok(out.into_iter().map(f32::to_bits).collect())
                        })
                        .collect::<Result<_>>()?;

                    for backend in [
                        MatvecBackend::Scalar,
                        MatvecBackend::Auto,
                        MatvecBackend::ExperimentalNeon,
                    ] {
                        for input_idx in [0_usize, count - 1] {
                            let mut scratch = if pinned {
                                MatvecScratch::with_pinned_backend(backend)
                            } else {
                                MatvecScratch::with_backend(backend)
                            };
                            #[cfg(feature = "parallel-rows")]
                            scratch.set_row_parallelism(RowParallelism::new(2)?);
                            let mut mv_out = Vec::new();
                            matvec_into(&tensor, &inputs[input_idx], &mut mv_out, &mut scratch)?;
                            assert_eq!(
                                mv_out.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                                expected[input_idx],
                                "matvec mismatch cols={columns} rows={rows} pinned={pinned} backend={backend:?}"
                            );
                        }

                        let mut tile = ProjectionTile::with_backend(backend);
                        #[cfg(feature = "parallel-rows")]
                        tile.set_row_parallelism(RowParallelism::new(4)?);
                        let mut actual = vec![Vec::new(); count];
                        tile.project(&tensor, inputs, &mut actual, pinned, &|| Ok(()))?;
                        let actual_bits: Vec<Vec<u32>> = actual
                            .iter()
                            .map(|row| row.iter().copied().map(f32::to_bits).collect())
                            .collect();
                        assert_eq!(
                            actual_bits, expected,
                            "tiled f16 mismatch cols={columns} rows={rows} count={count} pinned={pinned} backend={backend:?}"
                        );
                        let preps_before: Vec<usize> = tile
                            .preparations
                            .iter()
                            .take(count)
                            .map(MatvecScratch::preparation_count)
                            .collect();
                        tile.project_with_previous(
                            Some(&tensor),
                            &tensor,
                            inputs,
                            &mut actual,
                            pinned,
                            &|| Ok(()),
                        )?;
                        let preps_after: Vec<usize> = tile
                            .preparations
                            .iter()
                            .take(count)
                            .map(MatvecScratch::preparation_count)
                            .collect();
                        assert_eq!(preps_after, preps_before);
                    }
                }
            }
        }

        // NaN/Inf weight rejection (even with zero activations) must clear outputs and recover.
        let valid_tensor = make_f16_tensor(32, 13)?;
        let zero_inputs = vec![vec![0.0_f32; 32]; 9];
        for bad_f16 in [0x7c00_u16, 0x7e00, 0xfc00] {
            let mut bad_bytes = valid_tensor.data.to_vec();
            let corrupt_off = bad_bytes.len() - 2;
            bad_bytes[corrupt_off..].copy_from_slice(&bad_f16.to_le_bytes());
            let bad_tensor = Tensor {
                name: "bad-f16".into(),
                dimensions: vec![32, 13],
                kind: F16,
                data: Arc::<[u8]>::from(bad_bytes).into(),
            };
            for pinned in [false, true] {
                for backend in [MatvecBackend::Scalar, MatvecBackend::ExperimentalNeon] {
                    let mut scratch = if pinned {
                        MatvecScratch::with_pinned_backend(backend)
                    } else {
                        MatvecScratch::with_backend(backend)
                    };
                    let mut mv_out = vec![1.0_f32];
                    assert!(
                        matvec_into(&bad_tensor, &zero_inputs[0], &mut mv_out, &mut scratch)
                            .is_err()
                    );
                    assert!(mv_out.is_empty());
                    matvec_into(&valid_tensor, &zero_inputs[0], &mut mv_out, &mut scratch)?;
                    assert_eq!(mv_out.len(), 13);

                    let mut tile = ProjectionTile::with_backend(backend);
                    #[cfg(feature = "parallel-rows")]
                    tile.set_row_parallelism(RowParallelism::new(2)?);
                    let mut outs = vec![vec![1.0_f32]; 9];
                    assert!(
                        tile.project(&bad_tensor, &zero_inputs, &mut outs, pinned, &|| Ok(()))
                            .is_err()
                    );
                    assert!(outs.iter().all(Vec::is_empty));
                    tile.project(&valid_tensor, &zero_inputs, &mut outs, pinned, &|| Ok(()))?;
                    assert!(outs.iter().all(|out| out.len() == 13));
                }
            }
        }
        Ok(())
    }
}
