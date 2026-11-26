//! Adreno `#40` attention tile planner, deterministic cache keys, and CPU reference oracle.
//!
//! Selects between single-dispatch [`AdrenoAttentionPath::Fused`] and 3-stage
//! [`AdrenoAttentionPath::Decomposed3Stage`] execution according to Adreno shared-memory
//! pressure, scratch buffer limits, and Windows TDR MAC bounds, while enforcing Prism's
//! IEEE-754 binary16 probability-rounding contract prior to multiplying `V`.
#![forbid(unsafe_code)]

use bonsai_native_hash::Hasher;
use bonsai_ternary::{float_to_half, half_to_float};

use crate::{
    AdrenoAttentionPath, AttentionStrategy, Error, VulkanDeviceCapabilities, hash_bytes, hash_str,
    hash_u32, hash_u64,
};

/// Maximum `total_seq` supported by [`crate::spirv::attn_fused_f16prob`] shared storage (`[f32; 512]`).
pub const FUSED_MAX_TOTAL_SEQ: u32 = 512;

/// Workgroup shared-memory footprint (`(512 + 64) * 4 = 2304` bytes) for fused attention.
pub const FUSED_SHARED_MEMORY_BYTES: u32 = (FUSED_MAX_TOTAL_SEQ + 64) * 4;

/// Conservative per-dispatch MAC cap bounding Adreno compute time well below Windows TDR limits.
pub const TDR_MAC_LIMIT: u64 = 64_000_000;

/// Validated geometry and normalization parameters for causal/non-causal GQA attention.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttentionDescriptor {
    /// Number of active query tokens processed (`query_len >= 1`).
    pub query_len: u32,
    /// Total active key/value sequence positions (`total_seq >= 1`).
    pub total_seq: u32,
    /// Number of past prefix tokens preceding query token `0`.
    pub past_len: u32,
    /// Number of query attention heads (`num_heads % num_kv_heads == 0`).
    pub num_heads: u32,
    /// Number of key/value attention heads (`num_kv_heads >= 1`).
    pub num_kv_heads: u32,
    /// Element dimension per head (`head_dim >= 1`).
    pub head_dim: u32,
    /// Finite score scale (`1.0 / sqrt(head_dim)` in standard scaled dot-product attention).
    pub scale: f32,
    /// Whether query token `qi` masks positions `>= past_len + qi + 1`.
    pub causal: bool,
}

impl AttentionDescriptor {
    /// Validates geometry, GQA divisibility, finite scale, causal bounds, and buffer sizes.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] for invalid dimensions, non-finite scale, or arithmetic
    /// overflow when computing tensor element counts.
    pub fn validate(&self) -> Result<(usize, usize, usize), Error> {
        if self.query_len == 0
            || self.total_seq == 0
            || self.num_heads == 0
            || self.num_kv_heads == 0
            || self.head_dim == 0
        {
            return Err(Error::InvalidInput(
                "attention dimensions must all be non-zero".to_owned(),
            ));
        }
        if !self.num_heads.is_multiple_of(self.num_kv_heads) {
            return Err(Error::InvalidInput(format!(
                "num_heads ({}) must be a multiple of num_kv_heads ({})",
                self.num_heads, self.num_kv_heads
            )));
        }
        if !self.scale.is_finite() {
            return Err(Error::InvalidInput(
                "attention scale must be finite".to_owned(),
            ));
        }
        if self.causal {
            let max_valid = self.past_len.checked_add(self.query_len).ok_or_else(|| {
                Error::InvalidInput("past_len + query_len overflows u32".to_owned())
            })?;
            if max_valid > self.total_seq {
                return Err(Error::InvalidInput(format!(
                    "causal attention past_len ({}) + query_len ({}) exceeds total_seq ({})",
                    self.past_len, self.query_len, self.total_seq
                )));
            }
        }
        let q_elems = checked_prod_usize(&[self.query_len, self.num_heads, self.head_dim])?;
        let kv_elems = checked_prod_usize(&[self.total_seq, self.num_kv_heads, self.head_dim])?;
        let score_elems = checked_prod_usize(&[self.query_len, self.num_heads, self.total_seq])?;
        Ok((q_elems, kv_elems, score_elems))
    }

    /// Computes expected `Q`, `K/V`, and `Scores/Probs` slice lengths in `f32` elements.
    ///
    /// # Errors
    /// Propagates validation or overflow errors from [`Self::validate`].
    pub fn element_counts(&self) -> Result<(usize, usize, usize), Error> {
        self.validate()
    }
}

/// Selected Adreno attention execution plan and dispatch geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionTilePlan {
    /// Selected execution path (`Fused` or `Decomposed3Stage`).
    pub path: AdrenoAttentionPath,
    /// Global scratch bytes required across intermediate score/probability buffers.
    pub scratch_bytes: u64,
    /// Per-workgroup shared storage footprint in bytes (`0` for Stage 1/3, `256` for Stage 2, `2304` for Fused).
    pub shared_memory_bytes: u32,
    /// Estimated total MAC count across `QK` and `PV`.
    pub mac_ops: u64,
    /// Workgroup grid for Stage 1 (`QK` scores) or `[1, num_heads, query_len]` for `Fused`.
    pub qk_workgroups: [u32; 3],
    /// Workgroup grid for Stage 2 (`Softmax+F16`), or `[0; 3]` when `Fused`.
    pub softmax_workgroups: [u32; 3],
    /// Workgroup grid for Stage 3 (`PV` accumulate), or `[0; 3]` when `Fused`.
    pub pv_workgroups: [u32; 3],
}

/// Captured intermediate `#40` attention tensors for white-box binary16 verification.
#[derive(Clone, Debug, PartialEq)]
pub struct AttentionStageOutputs {
    /// Stage-1 scaled `QK^T` scores shaped `[query_len, num_heads, total_seq]`.
    pub scores: Vec<f32>,
    /// Stage-2 softmax probabilities explicitly rounded to IEEE binary16 `[query_len, num_heads, total_seq]`.
    pub rounded_probs: Vec<f32>,
}

/// Execution provenance witness attached to every [`AttentionOutput`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AttentionExecutionWitness {
    /// Planned tiling, path, scratch footprint, and dispatch counts.
    pub plan: AttentionTilePlan,
    /// Deterministic cache key binding device identity, geometry, path, and shader digests.
    pub cache_key: String,
    /// Actual GPU stage submissions executed (`1` for `Fused`, `3` for `Decomposed3Stage`).
    pub stage_count: u32,
    /// SHA-256 digest of the Stage-1 `QK` SPIR-V module.
    pub qk_shader_digest: String,
    /// SHA-256 digest of the Stage-2 `Softmax+F16` SPIR-V module.
    pub softmax_shader_digest: String,
    /// SHA-256 digest of the Stage-3 `PV` SPIR-V module.
    pub pv_shader_digest: String,
    /// SHA-256 digest of the `Fused` SPIR-V module.
    pub fused_shader_digest: String,
}

/// Result returned by `VulkanAttentionExecutor::execute`.
#[derive(Clone, Debug, PartialEq)]
pub struct AttentionOutput {
    /// Computed attention output shaped `[query_len, num_heads, head_dim]`.
    pub output: Vec<f32>,
    /// Execution provenance and resource plan witness.
    pub witness: AttentionExecutionWitness,
    /// Optional intermediate Stage-1 scores and Stage-2 rounded probabilities when captured.
    pub stages: Option<AttentionStageOutputs>,
}

/// Selects [`AdrenoAttentionPath::Fused`] vs [`AdrenoAttentionPath::Decomposed3Stage`] from
/// adapter compute limits, scratch budget, and TDR MAC bounds.
///
/// # Errors
/// Returns [`Error::InvalidInput`] on invalid geometry or [`Error::ResourceLimit`] when
/// inputs, intermediate buffers, workgroup grids, shared memory, or TDR MACs exceed bounds.
pub fn select_attention_plan(
    desc: &AttentionDescriptor,
    caps: &VulkanDeviceCapabilities,
    strategy: AttentionStrategy,
    scratch_budget_bytes: u64,
) -> Result<AttentionTilePlan, Error> {
    let (q_elems, kv_elems, score_elems) = desc.validate()?;
    let max_range = u64::from(caps.compute_limits.max_storage_buffer_range);
    let q_bytes = u64::try_from(q_elems)
        .ok()
        .and_then(|elems| elems.checked_mul(4))
        .ok_or_else(|| Error::ResourceLimit("Q buffer byte size overflow".to_owned()))?;
    let kv_bytes = u64::try_from(kv_elems)
        .ok()
        .and_then(|elems| elems.checked_mul(4))
        .ok_or_else(|| Error::ResourceLimit("K/V buffer byte size overflow".to_owned()))?;
    let single_scratch_bytes = u64::try_from(score_elems)
        .ok()
        .and_then(|elems| elems.checked_mul(4))
        .ok_or_else(|| Error::ResourceLimit("scratch buffer byte size overflow".to_owned()))?;

    if q_bytes > max_range || kv_bytes > max_range {
        return Err(Error::ResourceLimit(format!(
            "Q ({q_bytes} B) or K/V ({kv_bytes} B) exceeds maxStorageBufferRange ({max_range} B)"
        )));
    }

    let qk_macs = checked_prod_u64(&[
        desc.query_len,
        desc.num_heads,
        desc.total_seq,
        desc.head_dim,
    ])?;
    let total_macs = qk_macs
        .checked_mul(2)
        .ok_or_else(|| Error::ResourceLimit("attention MAC count overflow".to_owned()))?;

    let max_wg = caps.compute_limits.max_compute_work_group_count;
    let fused_wg = [1_u32, desc.num_heads, desc.query_len];
    let qk_wg = [desc.total_seq.div_ceil(64), desc.num_heads, desc.query_len];
    let softmax_wg = [1_u32, desc.num_heads, desc.query_len];
    let pv_wg = [desc.head_dim.div_ceil(64), desc.num_heads, desc.query_len];

    let fused_shared_ok = desc.total_seq <= FUSED_MAX_TOTAL_SEQ
        && FUSED_SHARED_MEMORY_BYTES <= caps.compute_limits.max_compute_shared_memory_size / 2;
    let fused_tdr_ok = total_macs <= TDR_MAC_LIMIT;
    let fused_wg_ok = fits_workgroups(fused_wg, max_wg);

    let decomposed_scratch_bytes = single_scratch_bytes
        .checked_mul(2)
        .ok_or_else(|| Error::ResourceLimit("decomposed scratch byte overflow".to_owned()))?;
    let decomposed_shared_ok = 256 <= caps.compute_limits.max_compute_shared_memory_size;
    let decomposed_range_ok = single_scratch_bytes <= max_range;
    let decomposed_budget_ok = decomposed_scratch_bytes <= scratch_budget_bytes;
    let decomposed_tdr_ok = qk_macs <= TDR_MAC_LIMIT;
    let decomposed_wg_ok = fits_workgroups(qk_wg, max_wg)
        && fits_workgroups(softmax_wg, max_wg)
        && fits_workgroups(pv_wg, max_wg);

    let decomposed = DecomposedPlanInputs {
        decomposed_scratch_bytes,
        total_macs,
        qk_wg,
        softmax_wg,
        pv_wg,
        shared_ok: decomposed_shared_ok,
        range_ok: decomposed_range_ok,
        budget_ok: decomposed_budget_ok,
        tdr_ok: decomposed_tdr_ok,
        wg_ok: decomposed_wg_ok,
        scratch_budget_bytes,
        max_range,
    };

    match strategy {
        AttentionStrategy::ForceFused => {
            if !(fused_shared_ok && fused_tdr_ok && fused_wg_ok) {
                return Err(Error::ResourceLimit(format!(
                    "forced fused attention exceeds limits: total_seq={} (max {FUSED_MAX_TOTAL_SEQ}), macs={total_macs} (cap {TDR_MAC_LIMIT})",
                    desc.total_seq
                )));
            }
            Ok(AttentionTilePlan {
                path: AdrenoAttentionPath::Fused,
                scratch_bytes: 0,
                shared_memory_bytes: FUSED_SHARED_MEMORY_BYTES,
                mac_ops: total_macs,
                qk_workgroups: fused_wg,
                softmax_workgroups: [0; 3],
                pv_workgroups: [0; 3],
            })
        }
        AttentionStrategy::ForceDecomposed => build_decomposed_plan(&decomposed),
        AttentionStrategy::Auto => {
            if fused_shared_ok && fused_tdr_ok && fused_wg_ok {
                Ok(AttentionTilePlan {
                    path: AdrenoAttentionPath::Fused,
                    scratch_bytes: 0,
                    shared_memory_bytes: FUSED_SHARED_MEMORY_BYTES,
                    mac_ops: total_macs,
                    qk_workgroups: fused_wg,
                    softmax_workgroups: [0; 3],
                    pv_workgroups: [0; 3],
                })
            } else {
                build_decomposed_plan(&decomposed)
            }
        }
    }
}

#[expect(
    clippy::struct_excessive_bools,
    reason = "Aggregates independent resource-limit validation flags."
)]
struct DecomposedPlanInputs {
    decomposed_scratch_bytes: u64,
    total_macs: u64,
    qk_wg: [u32; 3],
    softmax_wg: [u32; 3],
    pv_wg: [u32; 3],
    shared_ok: bool,
    range_ok: bool,
    budget_ok: bool,
    tdr_ok: bool,
    wg_ok: bool,
    scratch_budget_bytes: u64,
    max_range: u64,
}

fn build_decomposed_plan(input: &DecomposedPlanInputs) -> Result<AttentionTilePlan, Error> {
    if !input.shared_ok {
        return Err(Error::ResourceLimit(
            "adapter shared memory < 256 bytes for softmax reduction".to_owned(),
        ));
    }
    if !input.range_ok {
        return Err(Error::ResourceLimit(format!(
            "attention score/probability scratch exceeds maxStorageBufferRange ({} B)",
            input.max_range
        )));
    }
    if !input.budget_ok {
        return Err(Error::ResourceLimit(format!(
            "attention scratch ({} B) exceeds caller budget ({} B)",
            input.decomposed_scratch_bytes, input.scratch_budget_bytes
        )));
    }
    if !input.tdr_ok {
        return Err(Error::ResourceLimit(format!(
            "single-stage MAC count ({}) exceeds TDR guard ({TDR_MAC_LIMIT})",
            input.total_macs / 2
        )));
    }
    if !input.wg_ok {
        return Err(Error::ResourceLimit(
            "attention workgroup grid exceeds maxComputeWorkGroupCount".to_owned(),
        ));
    }
    Ok(AttentionTilePlan {
        path: AdrenoAttentionPath::Decomposed3Stage,
        scratch_bytes: input.decomposed_scratch_bytes,
        shared_memory_bytes: 256,
        mac_ops: input.total_macs,
        qk_workgroups: input.qk_wg,
        softmax_workgroups: input.softmax_wg,
        pv_workgroups: input.pv_wg,
    })
}

const fn fits_workgroups(grid: [u32; 3], limit: [u32; 3]) -> bool {
    grid[0] >= 1
        && grid[1] >= 1
        && grid[2] >= 1
        && grid[0] <= limit[0]
        && grid[1] <= limit[1]
        && grid[2] <= limit[2]
}

fn checked_prod_u64(dims: &[u32]) -> Result<u64, Error> {
    let mut acc = 1_u64;
    for &dim in dims {
        acc = acc.checked_mul(u64::from(dim)).ok_or_else(|| {
            Error::ResourceLimit("attention dimension product overflow".to_owned())
        })?;
    }
    Ok(acc)
}

fn checked_prod_usize(dims: &[u32]) -> Result<usize, Error> {
    let prod = checked_prod_u64(dims)?;
    usize::try_from(prod)
        .map_err(|_| Error::ResourceLimit("attention slice length exceeds usize".to_owned()))
}

/// Computes a deterministic SHA-256 pipeline/cache key binding device identity,
/// attention geometry, planned execution path, and SPIR-V shader digests.
///
/// # Errors
/// Returns [`Error::InvalidInput`] if hashing fails.
pub fn attention_cache_key(
    desc: &AttentionDescriptor,
    plan: &AttentionTilePlan,
    caps: &VulkanDeviceCapabilities,
    qk_shader_digest: &str,
    softmax_shader_digest: &str,
    pv_shader_digest: &str,
    fused_shader_digest: &str,
) -> Result<String, Error> {
    let mut hasher = Hasher::new().map_err(|err| Error::InvalidInput(err.to_string()))?;
    hash_str(
        &mut hasher,
        b"bonsai-adreno-attn-cache-v1\0",
        &caps.identity_sha256,
    )?;
    hash_str(&mut hasher, b"", plan.path.as_tag())?;
    hash_u64(&mut hasher, plan.scratch_bytes)?;
    hash_u32(&mut hasher, plan.shared_memory_bytes)?;
    hash_u32(&mut hasher, desc.query_len)?;
    hash_u32(&mut hasher, desc.total_seq)?;
    hash_u32(&mut hasher, desc.past_len)?;
    hash_u32(&mut hasher, desc.num_heads)?;
    hash_u32(&mut hasher, desc.num_kv_heads)?;
    hash_u32(&mut hasher, desc.head_dim)?;
    hash_u32(&mut hasher, desc.scale.to_bits())?;
    hash_bytes(&mut hasher, &[u8::from(desc.causal)])?;
    match plan.path {
        AdrenoAttentionPath::Fused => hash_str(&mut hasher, b"", fused_shader_digest)?,
        AdrenoAttentionPath::Decomposed3Stage => {
            hash_str(&mut hasher, b"", qk_shader_digest)?;
            hash_str(&mut hasher, b"", softmax_shader_digest)?;
            hash_str(&mut hasher, b"", pv_shader_digest)?;
        }
    }
    let digest = hasher
        .finish()
        .map_err(|err| Error::InvalidInput(err.to_string()))?;
    Ok(hex::encode(digest))
}

/// Safe CPU reference implementation following Prism's `softmax_round_f16` contract.
///
/// Computes left-to-right binary32 `QK^T * scale`, global row max, binary32 `exp` and
/// reciprocal normalization, explicit IEEE binary16 rounding (`float_to_half` -> `half_to_float`),
/// and left-to-right `probs * V` accumulation.
///
/// # Errors
/// Returns [`Error::InvalidInput`] for invalid dimensions, mismatched slice lengths, or non-finite inputs/outputs.
#[expect(
    clippy::suboptimal_flops,
    reason = "Preserve unfused multiply-then-add rounding matching NoContraction SPIR-V shaders."
)]
pub fn cpu_attention_reference(
    desc: &AttentionDescriptor,
    q: &[f32],
    k: &[f32],
    v: &[f32],
) -> Result<(Vec<f32>, AttentionStageOutputs), Error> {
    let (q_elems, kv_elems, score_elems) = desc.validate()?;
    if q.len() != q_elems || k.len() != kv_elems || v.len() != kv_elems {
        return Err(Error::InvalidInput(format!(
            "mismatched Q/K/V slice lengths: got ({}, {}, {}), expected ({q_elems}, {kv_elems}, {kv_elems})",
            q.len(),
            k.len(),
            v.len()
        )));
    }
    if q.iter().chain(k).chain(v).any(|val| !val.is_finite()) {
        return Err(Error::InvalidInput(
            "attention inputs contain non-finite f32 values".to_owned(),
        ));
    }

    let q_len = usize::try_from(desc.query_len).unwrap_or(0);
    let total_seq = usize::try_from(desc.total_seq).unwrap_or(0);
    let past_len = usize::try_from(desc.past_len).unwrap_or(0);
    let heads = usize::try_from(desc.num_heads).unwrap_or(0);
    let kv_heads = usize::try_from(desc.num_kv_heads).unwrap_or(0);
    let dim = usize::try_from(desc.head_dim).unwrap_or(0);
    let gqa = heads / kv_heads;

    let mut scores = vec![0.0_f32; score_elems];
    let mut rounded_probs = vec![0.0_f32; score_elems];
    let mut output = vec![0.0_f32; q_elems];

    for qi in 0..q_len {
        let valid_len = if desc.causal {
            (past_len + qi + 1).min(total_seq)
        } else {
            total_seq
        };
        for hi in 0..heads {
            let kv_h = hi / gqa;
            let row_head = qi * heads + hi;
            let q_base = row_head * dim;
            let s_base = row_head * total_seq;

            let mut row_max = f32::NEG_INFINITY;
            for ki in 0..valid_len {
                let k_base = (ki * kv_heads + kv_h) * dim;
                let mut dot = 0.0_f32;
                for d in 0..dim {
                    dot += q[q_base + d] * k[k_base + d];
                }
                let scaled = desc.scale * dot;
                scores[s_base + ki] = scaled;
                if scaled > row_max {
                    row_max = scaled;
                }
            }

            let mut sum = 0.0_f32;
            for ki in 0..valid_len {
                let e = (scores[s_base + ki] - row_max).exp();
                rounded_probs[s_base + ki] = e;
                sum += e;
            }
            if !sum.is_finite() || sum <= 0.0 {
                return Err(Error::InvalidInput(
                    "non-finite or non-positive softmax denominator".to_owned(),
                ));
            }
            let inv_sum = 1.0_f32 / sum;
            for ki in 0..valid_len {
                let raw_p = rounded_probs[s_base + ki] * inv_sum;
                let half = float_to_half(raw_p);
                rounded_probs[s_base + ki] = half_to_float(half);
            }

            for d in 0..dim {
                let mut acc = 0.0_f32;
                for ki in 0..valid_len {
                    let v_idx = (ki * kv_heads + kv_h) * dim + d;
                    acc += rounded_probs[s_base + ki] * v[v_idx];
                }
                if !acc.is_finite() {
                    return Err(Error::InvalidInput(
                        "non-finite attention output element".to_owned(),
                    ));
                }
                output[q_base + d] = acc;
            }
        }
    }

    Ok((
        output,
        AttentionStageOutputs {
            scores,
            rounded_probs,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AdrenoFamily, VulkanComputeLimits};

    fn test_caps(shared_mem: u32, max_range: u32) -> VulkanDeviceCapabilities {
        let mut caps = VulkanDeviceCapabilities {
            device_name: "Qualcomm(R) Adreno(TM) X2-90 GPU".to_owned(),
            family: AdrenoFamily::X2,
            vendor_id: 0x5143,
            device_id: 0x3630_0000,
            driver_version: 1,
            api_version: 4_210_983,
            subgroup_size: 64,
            wave64_subgroup_shuffle: true,
            shader_float16: true,
            shader_integer_dot_product: true,
            integer_dot_4x8_packed_signed_accelerated: true,
            cooperative_matrix_shapes: Vec::new(),
            qcom_cooperative_matrix_conversion: true,
            compute_limits: VulkanComputeLimits {
                max_compute_shared_memory_size: shared_mem,
                max_compute_work_group_count: [65_535, 65_535, 65_535],
                max_compute_work_group_invocations: 1024,
                max_compute_work_group_size: [1024, 1024, 1024],
                max_storage_buffer_range: max_range,
                max_push_constants_size: 256,
            },
            memory_heaps: Vec::new(),
            identity_sha256: String::new(),
        };
        let _ = caps.recompute_identity_sha256();
        caps
    }

    #[test]
    fn planner_switches_between_fused_and_decomposed_on_sequence_and_shared_pressure()
    -> Result<(), Error> {
        let caps = test_caps(32_768, 2_147_483_647);
        let short_desc = AttentionDescriptor {
            query_len: 1,
            total_seq: 64,
            past_len: 63,
            num_heads: 24,
            num_kv_heads: 4,
            head_dim: 256,
            scale: 1.0 / 16.0,
            causal: true,
        };
        let short_plan = select_attention_plan(
            &short_desc,
            &caps,
            AttentionStrategy::Auto,
            64 * 1024 * 1024,
        )?;
        assert_eq!(short_plan.path, AdrenoAttentionPath::Fused);

        let long_desc = AttentionDescriptor {
            total_seq: 600,
            past_len: 599,
            ..short_desc
        };
        let long_plan =
            select_attention_plan(&long_desc, &caps, AttentionStrategy::Auto, 64 * 1024 * 1024)?;
        assert_eq!(long_plan.path, AdrenoAttentionPath::Decomposed3Stage);
        assert!(
            select_attention_plan(
                &long_desc,
                &caps,
                AttentionStrategy::ForceFused,
                64 * 1024 * 1024
            )
            .is_err()
        );

        let tight_shared = test_caps(4096, 2_147_483_647);
        let fallback_plan = select_attention_plan(
            &short_desc,
            &tight_shared,
            AttentionStrategy::Auto,
            64 * 1024 * 1024,
        )?;
        assert_eq!(fallback_plan.path, AdrenoAttentionPath::Decomposed3Stage);

        assert!(matches!(
            select_attention_plan(&long_desc, &caps, AttentionStrategy::Auto, 1024),
            Err(Error::ResourceLimit(_))
        ));
        Ok(())
    }

    #[test]
    fn cache_key_changes_with_path_geometry_and_device_and_oracle_rounds_f16_probs()
    -> Result<(), Error> {
        let caps_a = test_caps(32_768, 2_147_483_647);
        let mut caps_b = test_caps(32_768, 2_147_483_647);
        caps_b.device_id += 1;
        caps_b.recompute_identity_sha256()?;

        let desc = AttentionDescriptor {
            query_len: 2,
            total_seq: 6,
            past_len: 4,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 4,
            scale: 0.5,
            causal: true,
        };
        let plan_f =
            select_attention_plan(&desc, &caps_a, AttentionStrategy::ForceFused, 1024 * 1024)?;
        let plan_d = select_attention_plan(
            &desc,
            &caps_a,
            AttentionStrategy::ForceDecomposed,
            1024 * 1024,
        )?;

        let k1 = attention_cache_key(&desc, &plan_f, &caps_a, "qk", "sm", "pv", "fu")?;
        let k2 = attention_cache_key(&desc, &plan_d, &caps_a, "qk", "sm", "pv", "fu")?;
        let k3 = attention_cache_key(&desc, &plan_f, &caps_b, "qk", "sm", "pv", "fu")?;
        assert_ne!(k1, k2);
        assert_ne!(k1, k3);

        let q = vec![0.3_f32; 2 * 2 * 4];
        let k = (0..6 * 4)
            .map(|i| f32::from(u8::try_from(i).unwrap_or(0)) * 0.05)
            .collect::<Vec<_>>();
        let v = (0..6 * 4)
            .map(|i| f32::mul_add(f32::from(u8::try_from(i).unwrap_or(0)), -0.04, 0.5))
            .collect::<Vec<_>>();
        let (out, stages) = cpu_attention_reference(&desc, &q, &k, &v)?;
        assert_eq!(out.len(), 2 * 2 * 4);
        assert!(out.iter().all(|val| val.is_finite()));
        for &p in &stages.rounded_probs {
            assert_eq!(half_to_float(float_to_half(p)).to_bits(), p.to_bits());
        }
        assert_eq!(stages.rounded_probs[5], 0.0);
        Ok(())
    }
}
