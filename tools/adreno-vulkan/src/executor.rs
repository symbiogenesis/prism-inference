//! Native Adreno Vulkan projection (`#39`) and attention (`#40`) executors.
//!
//! Implements [`Pq2ProjectionExecutor`] for `PQ2_0` and `PTQ1_0` weights (`Q8_K` and `Q8_0`
//! activations, fused/unfused/pinned modes) plus [`VulkanAttentionExecutor`] supporting both
//! single-dispatch fused attention and 3-stage decomposed attention with explicit Vulkan
//! compute barriers, deterministic cache keys, optional stage capture, and cooperative cancellation.
#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use bonsai_prism_runtime::{
    kernels::{Pq2ProjectionExecutor, Q8Block},
    model::{Tensor, TensorBytes, WeakTensorBytes},
};
use bonsai_ternary::Q8KBlock;

use crate::{
    AdrenoAttentionPath, AttentionDescriptor, AttentionExecutionWitness, AttentionOutput,
    AttentionStageOutputs, AttentionStrategy, Error, ProjectionKernelPath,
    VulkanDeviceCapabilities, attention_cache_key,
    device::{AdrenoVulkanDevice, BufferId, ComputeStage, PipelineId},
    select_attention_plan, spirv,
};

const KIND_PQ2: u32 = 142;
const KIND_PTQ1: u32 = 143;
const PQ2_BLOCK_BYTES: usize = 34;
const PTQ1_BLOCK_BYTES: usize = 28;
const Q8K_WORDS_PER_BLOCK: usize = 65;
const Q8_0_WORDS_PER_BLOCK: usize = 9;
const DEFAULT_OPERATION_CAPACITY: u64 = 1_u64 << 40;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct WeightCacheKey {
    name: String,
    kind: u32,
    cols: usize,
    rows: usize,
}

struct CachedWeight {
    buffer: BufferId,
    bytes: usize,
    weak_source: WeakTensorBytes,
    last_used: u64,
}

enum WeightBufferLease {
    Cached(BufferId),
    Transient(BufferId),
}

impl WeightBufferLease {
    const fn id(&self) -> BufferId {
        match *self {
            Self::Cached(id) | Self::Transient(id) => id,
        }
    }
}

/// Scoped GPU projection executor (`#39`) backed by native Adreno Vulkan SPIR-V 1.6 shaders.
///
/// `fused = false` reproduces CPU `dot_pq2::<false>`, `dot_pq2_q8_0`, `dot_ptq1`, and `dot_ptq1_q8k`
/// bit-for-bit (`0 ULP`) on Adreno X2-90; `fused = true` emits `GLSL.std.450` `Fma` (`NoContraction`),
/// whose Vulkan precision follows GPU multiply-add hardware (`<= 1 ULP` vs single-rounded `f32::mul_add`).
pub struct VulkanPq2Executor {
    device: AdrenoVulkanDevice,
    kernel_path: ProjectionKernelPath,
    identity: String,
    pipe_pq2_q8k_fused: PipelineId,
    pipe_pq2_q8k_unfused: PipelineId,
    pipe_pq2_q8_0: PipelineId,
    pipe_ptq1_q8_0: PipelineId,
    pipe_ptq1_q8_0_pinned: PipelineId,
    pipe_ptq1_q8k: PipelineId,
    weight_cache: BTreeMap<WeightCacheKey, CachedWeight>,
    cache_access_counter: u64,
    cached_bytes: usize,
    max_cached_weight_bytes: usize,
}

impl VulkanPq2Executor {
    /// Opens the native Adreno Vulkan device, auto-selecting [`ProjectionKernelPath::IntegerDotPacked4x8`]
    /// when `VK_KHR_shader_integer_dot_product` (`4x8BitPackedSignedAccelerated`) is enabled, or
    /// [`ProjectionKernelPath::PortableIntegerFallback`] otherwise.
    ///
    /// # Errors
    /// Propagates device discovery, loader, or SPIR-V pipeline creation errors.
    pub fn open_default(max_cached_weight_bytes: usize) -> Result<Self, Error> {
        let device = AdrenoVulkanDevice::open()?;
        let path = if device.integer_dot_product_enabled()
            && device.capabilities().supports_integer_dot_4x8()
        {
            ProjectionKernelPath::IntegerDotPacked4x8
        } else {
            ProjectionKernelPath::PortableIntegerFallback
        };
        Self::from_device(device, path, max_cached_weight_bytes)
    }

    /// Opens the native Adreno Vulkan device requiring a specific [`ProjectionKernelPath`]
    /// (enabling live A/B comparison between `OpSDot` and portable integer fallback).
    ///
    /// # Errors
    /// Returns [`Error::Unsupported`] if [`ProjectionKernelPath::IntegerDotPacked4x8`] is requested
    /// on a device lacking `VK_KHR_shader_integer_dot_product`, or propagates device/pipeline errors.
    pub fn open_with_path(
        path: ProjectionKernelPath,
        max_cached_weight_bytes: usize,
    ) -> Result<Self, Error> {
        let device = AdrenoVulkanDevice::open()?;
        if path == ProjectionKernelPath::IntegerDotPacked4x8
            && !device.integer_dot_product_enabled()
        {
            return Err(Error::Unsupported);
        }
        Self::from_device(device, path, max_cached_weight_bytes)
    }

    fn from_device(
        mut device: AdrenoVulkanDevice,
        kernel_path: ProjectionKernelPath,
        max_cached_weight_bytes: usize,
    ) -> Result<Self, Error> {
        let use_sdot = spirv::path_uses_sdot(kernel_path);
        let pipe_pq2_q8k_fused =
            device.create_compute_pipeline(&spirv::pq2_q8k_gemv(use_sdot, true), 3, 4)?;
        let pipe_pq2_q8k_unfused =
            device.create_compute_pipeline(&spirv::pq2_q8k_gemv(use_sdot, false), 3, 4)?;
        let pipe_pq2_q8_0 =
            device.create_compute_pipeline(&spirv::pq2_q8_0_gemv(use_sdot), 3, 4)?;
        let pipe_ptq1_q8_0 =
            device.create_compute_pipeline(&spirv::ptq1_q8_0_gemv(use_sdot, false), 3, 4)?;
        let pipe_ptq1_q8_0_pinned =
            device.create_compute_pipeline(&spirv::ptq1_q8_0_gemv(use_sdot, true), 3, 4)?;
        let pipe_ptq1_q8k =
            device.create_compute_pipeline(&spirv::ptq1_q8k_gemv(use_sdot), 3, 4)?;

        let caps = device.capabilities();
        let short_hash = &caps.identity_sha256[..16.min(caps.identity_sha256.len())];
        let identity = format!(
            "adreno-vulkan-v1:{}:{}:{}",
            caps.family.as_tag(),
            short_hash,
            kernel_path.as_tag()
        );
        if identity.is_empty()
            || identity.len() > 256
            || !identity.bytes().all(|byte| byte.is_ascii_graphic())
        {
            return Err(Error::InvalidInput(
                "constructed executor identity is invalid".to_owned(),
            ));
        }

        Ok(Self {
            device,
            kernel_path,
            identity,
            pipe_pq2_q8k_fused,
            pipe_pq2_q8k_unfused,
            pipe_pq2_q8_0,
            pipe_ptq1_q8_0,
            pipe_ptq1_q8_0_pinned,
            pipe_ptq1_q8k,
            weight_cache: BTreeMap::new(),
            cache_access_counter: 0,
            cached_bytes: 0,
            max_cached_weight_bytes,
        })
    }

    /// Returns the discovered Qualcomm Adreno device capability snapshot.
    #[must_use]
    pub const fn capabilities(&self) -> &VulkanDeviceCapabilities {
        self.device.capabilities()
    }

    /// Returns the active [`ProjectionKernelPath`].
    #[must_use]
    pub const fn kernel_path(&self) -> ProjectionKernelPath {
        self.kernel_path
    }

    /// Returns the currently resident cached GPU weight payload size in bytes.
    #[must_use]
    pub const fn cached_weight_bytes(&self) -> usize {
        self.cached_bytes
    }

    /// Returns `true` if a device timeout or fault has latched terminal teardown.
    #[must_use]
    pub const fn is_faulted(&self) -> bool {
        self.device.is_faulted()
    }

    /// Explicitly releases all resident cached GPU weight buffers and returns freed bytes.
    pub fn release_resident_weights(&mut self) -> usize {
        let released = self.cached_bytes;
        let entries = std::mem::take(&mut self.weight_cache);
        for (_, cached) in entries {
            self.device.free_buffer(cached.buffer);
        }
        self.cached_bytes = 0;
        released
    }

    /// Projects a `PTQ1_0` (`kind == 143`) weight matrix directly against prepared `Q8_K` blocks
    /// (supports odd `128`-column multiples via half-block indexing).
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] on mismatched dimensions or non-finite values,
    /// [`Error::ResourceLimit`] if extents exceed device storage bounds, or [`Error::DeviceFault`]/[`Error::Vulkan`].
    pub fn project_ptq1_q8k(
        &mut self,
        tensor: &Tensor,
        activation: &[Q8KBlock],
        output: &mut Vec<f32>,
    ) -> Result<(), Error> {
        output.clear();
        let (cols, rows) = validate_matrix(tensor, KIND_PTQ1, 128)?;
        let blocks_per_row = cols / 128;
        let expected_q8k = blocks_per_row.div_ceil(2);
        if activation.len() != expected_q8k {
            return Err(Error::InvalidInput(format!(
                "PTQ1 x Q8K expected {expected_q8k} activation blocks, got {}",
                activation.len()
            )));
        }
        let act_words = pack_q8k_blocks(activation)?;
        let act_words_per_token = u32::try_from(act_words.len())
            .map_err(|_| Error::ResourceLimit("activation word count overflow".to_owned()))?;
        self.dispatch_gemv(
            tensor,
            rows,
            blocks_per_row,
            1,
            act_words_per_token,
            &act_words,
            self.pipe_ptq1_q8k,
            output,
        )
    }

    /// Batched multi-token `PQ2_0` × `Q8_K` projection (`[tokens, q8k_blocks_per_token]` laid
    /// out contiguously per token, writing `[tokens, rows]` into `output`).
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`], [`Error::ResourceLimit`], [`Error::DeviceFault`], or [`Error::Vulkan`].
    pub fn project_pq2_q8k_tokens(
        &mut self,
        tensor: &Tensor,
        activations: &[Q8KBlock],
        tokens: usize,
        fused: bool,
        output: &mut Vec<f32>,
    ) -> Result<(), Error> {
        output.clear();
        if tokens == 0 {
            return Err(Error::InvalidInput(
                "multi-token projection requires tokens >= 1".to_owned(),
            ));
        }
        let (cols, rows) = validate_matrix(tensor, KIND_PQ2, 256)?;
        let blocks_per_row = cols / 128;
        let q8k_per_token = cols / 256;
        let expected = q8k_per_token
            .checked_mul(tokens)
            .ok_or_else(|| Error::InvalidInput("activation token count overflow".to_owned()))?;
        if activations.len() != expected {
            return Err(Error::InvalidInput(format!(
                "PQ2 x Q8K multi-token expected {expected} blocks, got {}",
                activations.len()
            )));
        }
        let act_words = pack_q8k_blocks(activations)?;
        let act_words_per_token = u32::try_from(q8k_per_token * Q8K_WORDS_PER_BLOCK)
            .map_err(|_| Error::ResourceLimit("activation word count overflow".to_owned()))?;
        let pipeline = if fused {
            self.pipe_pq2_q8k_fused
        } else {
            self.pipe_pq2_q8k_unfused
        };
        self.dispatch_gemv(
            tensor,
            rows,
            blocks_per_row,
            tokens,
            act_words_per_token,
            &act_words,
            pipeline,
            output,
        )
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "Centralizes weight caching, buffer lifecycle, push constants, and finite checks across projection kernels."
    )]
    fn dispatch_gemv(
        &mut self,
        tensor: &Tensor,
        rows: usize,
        blocks_per_row: usize,
        tokens: usize,
        act_words_per_token: u32,
        act_words: &[u32],
        pipeline: PipelineId,
        output: &mut Vec<f32>,
    ) -> Result<(), Error> {
        if self.is_faulted() {
            return Err(Error::DeviceFault(
                "Vulkan executor faulted and requires terminal teardown".to_owned(),
            ));
        }
        let rows_u32 = u32::try_from(rows)
            .map_err(|_| Error::ResourceLimit("projection row count exceeds u32".to_owned()))?;
        let blocks_u32 = u32::try_from(blocks_per_row)
            .map_err(|_| Error::ResourceLimit("projection block count exceeds u32".to_owned()))?;
        let tokens_u32 = u32::try_from(tokens)
            .map_err(|_| Error::ResourceLimit("projection token count exceeds u32".to_owned()))?;
        let total_outputs = rows
            .checked_mul(tokens)
            .ok_or_else(|| Error::ResourceLimit("projection output count overflow".to_owned()))?;

        let max_range = u64::from(
            self.device
                .capabilities()
                .compute_limits
                .max_storage_buffer_range,
        );
        let act_bytes = u64::try_from(act_words.len())
            .ok()
            .and_then(|w| w.checked_mul(4))
            .ok_or_else(|| Error::ResourceLimit("activation buffer size overflow".to_owned()))?;
        let out_bytes = u64::try_from(total_outputs)
            .ok()
            .and_then(|o| o.checked_mul(4))
            .ok_or_else(|| Error::ResourceLimit("output buffer size overflow".to_owned()))?;
        if act_bytes > max_range || out_bytes > max_range {
            return Err(Error::ResourceLimit(format!(
                "projection buffer exceeds maxStorageBufferRange ({max_range} B)"
            )));
        }

        let grid = [rows_u32.div_ceil(64), tokens_u32, 1];
        let max_wg = self
            .device
            .capabilities()
            .compute_limits
            .max_compute_work_group_count;
        if grid[0] > max_wg[0] || grid[1] > max_wg[1] {
            return Err(Error::ResourceLimit(
                "projection workgroup count exceeds maxComputeWorkGroupCount".to_owned(),
            ));
        }

        let weight_lease = self.acquire_weight_buffer(tensor, blocks_per_row * 128, rows)?;
        let act_buf = match self.device.create_storage_buffer(act_bytes) {
            Ok(id) => id,
            Err(err) => {
                self.release_transient_weight(&weight_lease);
                return Err(err);
            }
        };
        let out_buf = match self.device.create_storage_buffer(out_bytes) {
            Ok(id) => id,
            Err(err) => {
                self.device.free_buffer(act_buf);
                self.release_transient_weight(&weight_lease);
                return Err(err);
            }
        };

        let run_result = (|| -> Result<(), Error> {
            self.device.write_buffer_u32(act_buf, act_words)?;
            let push = [rows_u32, blocks_u32, tokens_u32, act_words_per_token];
            self.device.submit_compute(
                pipeline,
                &[weight_lease.id(), act_buf, out_buf],
                &push,
                grid,
            )?;
            output.resize(total_outputs, 0.0_f32);
            self.device.read_buffer_f32(out_buf, output)?;
            if output.iter().any(|val| !val.is_finite()) {
                return Err(Error::InvalidInput(
                    "GPU projection produced non-finite output".to_owned(),
                ));
            }
            Ok(())
        })();

        self.device.free_buffer(out_buf);
        self.device.free_buffer(act_buf);
        self.release_transient_weight(&weight_lease);
        if run_result.is_err() {
            output.clear();
        }
        run_result
    }

    fn release_transient_weight(&mut self, lease: &WeightBufferLease) {
        if let WeightBufferLease::Transient(buf) = *lease {
            self.device.free_buffer(buf);
        }
    }

    fn acquire_weight_buffer(
        &mut self,
        tensor: &Tensor,
        cols: usize,
        rows: usize,
    ) -> Result<WeightBufferLease, Error> {
        let key = WeightCacheKey {
            name: tensor.name.clone(),
            kind: tensor.kind,
            cols,
            rows,
        };
        if let Some(cached) = self.weight_cache.get_mut(&key)
            && cached
                .weak_source
                .upgrade()
                .is_some_and(|live| TensorBytes::ptr_eq(&live, &tensor.data))
        {
            self.cache_access_counter += 1;
            cached.last_used = self.cache_access_counter;
            return Ok(WeightBufferLease::Cached(cached.buffer));
        }

        if let Some(stale) = self.weight_cache.remove(&key) {
            self.cached_bytes = self.cached_bytes.saturating_sub(stale.bytes);
            self.device.free_buffer(stale.buffer);
        }

        validate_weight_scales(tensor.kind, &tensor.data)?;
        let (buf, upload_bytes) = upload_weight_bytes(&mut self.device, tensor.kind, &tensor.data)?;
        let max_range = u64::from(
            self.device
                .capabilities()
                .compute_limits
                .max_storage_buffer_range,
        );
        let upload_u64 = u64::try_from(upload_bytes)
            .map_err(|_| Error::ResourceLimit("weight byte length overflow".to_owned()))?;
        if upload_u64 > max_range {
            self.device.free_buffer(buf);
            return Err(Error::ResourceLimit(format!(
                "weight tensor '{}' ({upload_u64} B) exceeds maxStorageBufferRange ({max_range} B)",
                tensor.name
            )));
        }

        if self.max_cached_weight_bytes == 0 || upload_bytes > self.max_cached_weight_bytes {
            return Ok(WeightBufferLease::Transient(buf));
        }

        while self.cached_bytes + upload_bytes > self.max_cached_weight_bytes {
            let Some(evict_key) = self
                .weight_cache
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(k, _)| k.clone())
            else {
                break;
            };
            if let Some(removed) = self.weight_cache.remove(&evict_key) {
                self.cached_bytes = self.cached_bytes.saturating_sub(removed.bytes);
                self.device.free_buffer(removed.buffer);
            }
        }

        self.cache_access_counter += 1;
        self.cached_bytes += upload_bytes;
        self.weight_cache.insert(
            key,
            CachedWeight {
                buffer: buf,
                bytes: upload_bytes,
                weak_source: TensorBytes::downgrade(&tensor.data),
                last_used: self.cache_access_counter,
            },
        );
        Ok(WeightBufferLease::Cached(buf))
    }
}

impl Drop for VulkanPq2Executor {
    fn drop(&mut self) {
        let _ = self.release_resident_weights();
    }
}

impl Pq2ProjectionExecutor for VulkanPq2Executor {
    fn identity(&self) -> &str {
        &self.identity
    }

    fn operation_capacity(&self) -> Option<u64> {
        if self.is_faulted() {
            Some(0)
        } else {
            Some(DEFAULT_OPERATION_CAPACITY)
        }
    }

    fn operation_count(&self, tensor: &Tensor) -> Option<u64> {
        if !(self.supports(tensor) || self.supports_q8(tensor)) {
            return None;
        }
        let cols = u64::try_from(*tensor.dimensions.first()?).ok()?;
        let rows = u64::try_from(*tensor.dimensions.get(1)?).ok()?;
        cols.checked_mul(rows)
    }

    fn supports(&self, tensor: &Tensor) -> bool {
        !self.is_faulted() && validate_matrix(tensor, KIND_PQ2, 256).is_ok()
    }

    fn project(
        &mut self,
        tensor: &Tensor,
        activation: &[Q8KBlock],
        fused: bool,
        output: &mut Vec<f32>,
    ) -> bonsai_prism_runtime::Result<()> {
        output.clear();
        let (cols, rows) =
            validate_matrix(tensor, KIND_PQ2, 256).map_err(Error::into_runtime_error)?;
        let blocks_per_row = cols / 128;
        let expected_q8k = cols / 256;
        if activation.len() != expected_q8k {
            return Err(Error::InvalidInput(format!(
                "PQ2 x Q8K expected {expected_q8k} activation blocks, got {}",
                activation.len()
            ))
            .into_runtime_error());
        }
        let act_words = pack_q8k_blocks(activation).map_err(Error::into_runtime_error)?;
        let act_words_per_token = u32::try_from(act_words.len()).map_err(|_| {
            Error::ResourceLimit("activation word count overflow".to_owned()).into_runtime_error()
        })?;
        let pipeline = if fused {
            self.pipe_pq2_q8k_fused
        } else {
            self.pipe_pq2_q8k_unfused
        };
        self.dispatch_gemv(
            tensor,
            rows,
            blocks_per_row,
            1,
            act_words_per_token,
            &act_words,
            pipeline,
            output,
        )
        .map_err(Error::into_runtime_error)
    }

    fn supports_q8(&self, tensor: &Tensor) -> bool {
        !self.is_faulted()
            && (validate_matrix(tensor, KIND_PQ2, 128).is_ok()
                || validate_matrix(tensor, KIND_PTQ1, 128).is_ok())
    }

    fn project_q8(
        &mut self,
        tensor: &Tensor,
        activation: &[Q8Block],
        pinned: bool,
        output: &mut Vec<f32>,
    ) -> bonsai_prism_runtime::Result<()> {
        output.clear();
        let (cols, rows) = match tensor.kind {
            KIND_PQ2 => validate_matrix(tensor, KIND_PQ2, 128),
            KIND_PTQ1 => validate_matrix(tensor, KIND_PTQ1, 128),
            other => Err(Error::InvalidInput(format!(
                "unsupported Q8 projection tensor kind {other}"
            ))),
        }
        .map_err(Error::into_runtime_error)?;

        let blocks_per_row = cols / 128;
        let expected_q8 = cols / 32;
        if activation.len() != expected_q8 {
            return Err(Error::InvalidInput(format!(
                "Q8 projection expected {expected_q8} activation blocks, got {}",
                activation.len()
            ))
            .into_runtime_error());
        }
        let act_words = pack_q8_blocks(activation).map_err(Error::into_runtime_error)?;
        let act_words_per_token = u32::try_from(act_words.len()).map_err(|_| {
            Error::ResourceLimit("activation word count overflow".to_owned()).into_runtime_error()
        })?;
        let pipeline = match tensor.kind {
            KIND_PQ2 => self.pipe_pq2_q8_0,
            KIND_PTQ1 if pinned => self.pipe_ptq1_q8_0_pinned,
            KIND_PTQ1 => self.pipe_ptq1_q8_0,
            _ => unreachable!(),
        };
        self.dispatch_gemv(
            tensor,
            rows,
            blocks_per_row,
            1,
            act_words_per_token,
            &act_words,
            pipeline,
            output,
        )
        .map_err(Error::into_runtime_error)
    }
}

/// Adreno Vulkan attention executor (`#40`) implementing fused and 3-stage decomposed kernels
/// with explicit pipeline barriers, IEEE-754 binary16 probability rounding, and cancellation.
pub struct VulkanAttentionExecutor {
    device: AdrenoVulkanDevice,
    scratch_budget_bytes: u64,
    pipe_qk: PipelineId,
    pipe_softmax: PipelineId,
    pipe_pv: PipelineId,
    pipe_fused: PipelineId,
    qk_digest: String,
    softmax_digest: String,
    pv_digest: String,
    fused_digest: String,
}

impl VulkanAttentionExecutor {
    /// Opens the native Adreno Vulkan device and compiles `#40` fused & decomposed pipelines.
    ///
    /// # Errors
    /// Propagates loader, device selection, or SPIR-V compilation errors.
    pub fn open(scratch_budget_bytes: u64) -> Result<Self, Error> {
        let mut device = AdrenoVulkanDevice::open()?;
        let qk_spv = spirv::attn_qk_scores();
        let softmax_spv = spirv::attn_softmax_f16();
        let pv_spv = spirv::attn_pv_accum();
        let fused_spv = spirv::attn_fused_f16prob();

        let qk_digest = spirv::shader_digest(&qk_spv)?;
        let softmax_digest = spirv::shader_digest(&softmax_spv)?;
        let pv_digest = spirv::shader_digest(&pv_spv)?;
        let fused_digest = spirv::shader_digest(&fused_spv)?;

        let pipe_qk = device.create_compute_pipeline(&qk_spv, 3, 8)?;
        let pipe_softmax = device.create_compute_pipeline(&softmax_spv, 2, 8)?;
        let pipe_pv = device.create_compute_pipeline(&pv_spv, 3, 8)?;
        let pipe_fused = device.create_compute_pipeline(&fused_spv, 4, 8)?;

        Ok(Self {
            device,
            scratch_budget_bytes,
            pipe_qk,
            pipe_softmax,
            pipe_pv,
            pipe_fused,
            qk_digest,
            softmax_digest,
            pv_digest,
            fused_digest,
        })
    }

    /// Returns the discovered Qualcomm Adreno capability snapshot.
    #[must_use]
    pub const fn capabilities(&self) -> &VulkanDeviceCapabilities {
        self.device.capabilities()
    }

    /// Executes `#40` attention using [`select_attention_plan`], running either the single-dispatch
    /// fused kernel or staged 3-dispatch execution (`QK -> Softmax+F16 -> PV`) with inter-stage
    /// cooperative cancellation checks and optional Stage-1/Stage-2 readback.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] for non-finite inputs/mismatched slices, [`Error::ResourceLimit`]
    /// if device/scratch/TDR bounds reject the workload, [`Error::Cancelled`] if cancelled, or
    /// [`Error::DeviceFault`]/[`Error::Vulkan`].
    #[expect(
        clippy::too_many_arguments,
        reason = "Explicitly separates Q/K/V slices, descriptor, strategy, witness capture, and cancellation hook."
    )]
    pub fn execute<F>(
        &mut self,
        desc: &AttentionDescriptor,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        strategy: AttentionStrategy,
        capture_stages: bool,
        mut check_cancelled: F,
    ) -> Result<AttentionOutput, Error>
    where
        F: FnMut() -> bool,
    {
        let (q_elems, kv_elems, score_elems) = validate_attention_inputs(desc, q, k, v)?;
        let plan = select_attention_plan(
            desc,
            self.device.capabilities(),
            strategy,
            self.scratch_budget_bytes,
        )?;
        let cache_key = attention_cache_key(
            desc,
            &plan,
            self.device.capabilities(),
            &self.qk_digest,
            &self.softmax_digest,
            &self.pv_digest,
            &self.fused_digest,
        )?;
        if check_cancelled() {
            return Err(Error::Cancelled);
        }

        let q_buf = self.alloc_and_write_f32(q)?;
        let k_buf = match self.alloc_and_write_f32(k) {
            Ok(id) => id,
            Err(err) => {
                self.device.free_buffer(q_buf);
                return Err(err);
            }
        };
        let v_buf = match self.alloc_and_write_f32(v) {
            Ok(id) => id,
            Err(err) => {
                self.device.free_buffer(k_buf);
                self.device.free_buffer(q_buf);
                return Err(err);
            }
        };
        let out_bytes = u64::try_from(q_elems)
            .ok()
            .and_then(|e| e.checked_mul(4))
            .ok_or_else(|| Error::ResourceLimit("output bytes overflow".to_owned()))?;
        let out_buf = match self.device.create_storage_buffer(out_bytes) {
            Ok(id) => id,
            Err(err) => {
                self.device.free_buffer(v_buf);
                self.device.free_buffer(k_buf);
                self.device.free_buffer(q_buf);
                return Err(err);
            }
        };

        let push = attention_push_constants(desc);
        let res = match plan.path {
            AdrenoAttentionPath::Fused => self.execute_fused(
                plan, cache_key, q_buf, k_buf, v_buf, out_buf, &push, q_elems,
            ),
            AdrenoAttentionPath::Decomposed3Stage => self.execute_decomposed_staged(
                plan,
                cache_key,
                q_buf,
                k_buf,
                v_buf,
                out_buf,
                &push,
                q_elems,
                score_elems,
                kv_elems,
                capture_stages,
                &mut check_cancelled,
            ),
        };

        self.device.free_buffer(out_buf);
        self.device.free_buffer(v_buf);
        self.device.free_buffer(k_buf);
        self.device.free_buffer(q_buf);
        res
    }

    /// Executes 3-stage decomposed attention in a single Vulkan command buffer with explicit
    /// intra-command-buffer `COMPUTE_SHADER -> COMPUTE_SHADER` (`SHADER_WRITE -> SHADER_READ`)
    /// pipeline barriers and zero inter-stage host wait round-trips.
    ///
    /// # Errors
    /// Propagates validation, planning, or Vulkan execution errors.
    pub fn execute_pipelined_decomposed(
        &mut self,
        desc: &AttentionDescriptor,
        q: &[f32],
        k: &[f32],
        v: &[f32],
    ) -> Result<AttentionOutput, Error> {
        let (q_elems, _kv_elems, score_elems) = validate_attention_inputs(desc, q, k, v)?;
        let plan = select_attention_plan(
            desc,
            self.device.capabilities(),
            AttentionStrategy::ForceDecomposed,
            self.scratch_budget_bytes,
        )?;
        let cache_key = attention_cache_key(
            desc,
            &plan,
            self.device.capabilities(),
            &self.qk_digest,
            &self.softmax_digest,
            &self.pv_digest,
            &self.fused_digest,
        )?;

        let scratch_bytes = u64::try_from(score_elems)
            .ok()
            .and_then(|e| e.checked_mul(4))
            .ok_or_else(|| Error::ResourceLimit("scratch bytes overflow".to_owned()))?;
        let out_bytes = u64::try_from(q_elems)
            .ok()
            .and_then(|e| e.checked_mul(4))
            .ok_or_else(|| Error::ResourceLimit("output bytes overflow".to_owned()))?;

        let q_buf = self.alloc_and_write_f32(q)?;
        let k_buf = self
            .alloc_and_write_f32(k)
            .inspect_err(|_| self.device.free_buffer(q_buf))?;
        let v_buf = self.alloc_and_write_f32(v).inspect_err(|_| {
            self.device.free_buffer(k_buf);
            self.device.free_buffer(q_buf);
        })?;
        let scores_buf = self.device.create_storage_buffer(scratch_bytes)?;
        let probs_buf = match self.device.create_storage_buffer(scratch_bytes) {
            Ok(id) => id,
            Err(err) => {
                self.free_buffers(&[scores_buf, v_buf, k_buf, q_buf]);
                return Err(err);
            }
        };
        let out_buf = match self.device.create_storage_buffer(out_bytes) {
            Ok(id) => id,
            Err(err) => {
                self.free_buffers(&[probs_buf, scores_buf, v_buf, k_buf, q_buf]);
                return Err(err);
            }
        };

        let push = attention_push_constants(desc);
        let stages = [
            ComputeStage {
                pipeline: self.pipe_qk,
                buffers: &[q_buf, k_buf, scores_buf],
                push_u32s: &push,
                workgroups: plan.qk_workgroups,
            },
            ComputeStage {
                pipeline: self.pipe_softmax,
                buffers: &[scores_buf, probs_buf],
                push_u32s: &push,
                workgroups: plan.softmax_workgroups,
            },
            ComputeStage {
                pipeline: self.pipe_pv,
                buffers: &[probs_buf, v_buf, out_buf],
                push_u32s: &push,
                workgroups: plan.pv_workgroups,
            },
        ];

        let exec = (|| -> Result<AttentionOutput, Error> {
            self.device.submit_compute_stages(&stages)?;
            let mut output = vec![0.0_f32; q_elems];
            self.device.read_buffer_f32(out_buf, &mut output)?;
            if output.iter().any(|val| !val.is_finite()) {
                return Err(Error::InvalidInput(
                    "GPU pipelined attention produced non-finite output".to_owned(),
                ));
            }
            Ok(AttentionOutput {
                output,
                witness: self.witness(plan, cache_key, 3),
                stages: None,
            })
        })();

        self.free_buffers(&[out_buf, probs_buf, scores_buf, v_buf, k_buf, q_buf]);
        exec
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "Encapsulates fused submission, readback, finite verification, and witness emission."
    )]
    fn execute_fused(
        &mut self,
        plan: crate::AttentionTilePlan,
        cache_key: String,
        q_buf: BufferId,
        k_buf: BufferId,
        v_buf: BufferId,
        out_buf: BufferId,
        push: &[u32; 8],
        q_elems: usize,
    ) -> Result<AttentionOutput, Error> {
        self.device.submit_compute(
            self.pipe_fused,
            &[q_buf, k_buf, v_buf, out_buf],
            push,
            plan.qk_workgroups,
        )?;
        let mut output = vec![0.0_f32; q_elems];
        self.device.read_buffer_f32(out_buf, &mut output)?;
        if output.iter().any(|val| !val.is_finite()) {
            return Err(Error::InvalidInput(
                "GPU fused attention produced non-finite output".to_owned(),
            ));
        }
        Ok(AttentionOutput {
            output,
            witness: self.witness(plan, cache_key, 1),
            stages: None,
        })
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "Coordinates 3-stage buffers, cancellation checkpoints, optional stage capture, and cleanup."
    )]
    fn execute_decomposed_staged<F>(
        &mut self,
        plan: crate::AttentionTilePlan,
        cache_key: String,
        q_buf: BufferId,
        k_buf: BufferId,
        v_buf: BufferId,
        out_buf: BufferId,
        push: &[u32; 8],
        q_elems: usize,
        score_elems: usize,
        _kv_elems: usize,
        capture_stages: bool,
        check_cancelled: &mut F,
    ) -> Result<AttentionOutput, Error>
    where
        F: FnMut() -> bool,
    {
        let scratch_bytes = u64::try_from(score_elems)
            .ok()
            .and_then(|e| e.checked_mul(4))
            .ok_or_else(|| Error::ResourceLimit("scratch bytes overflow".to_owned()))?;
        let scores_buf = self.device.create_storage_buffer(scratch_bytes)?;
        let probs_buf = match self.device.create_storage_buffer(scratch_bytes) {
            Ok(id) => id,
            Err(err) => {
                self.device.free_buffer(scores_buf);
                return Err(err);
            }
        };

        let result = (|| -> Result<AttentionOutput, Error> {
            self.device.submit_compute(
                self.pipe_qk,
                &[q_buf, k_buf, scores_buf],
                push,
                plan.qk_workgroups,
            )?;
            if check_cancelled() {
                return Err(Error::Cancelled);
            }

            self.device.submit_compute(
                self.pipe_softmax,
                &[scores_buf, probs_buf],
                push,
                plan.softmax_workgroups,
            )?;
            if check_cancelled() {
                return Err(Error::Cancelled);
            }

            self.device.submit_compute(
                self.pipe_pv,
                &[probs_buf, v_buf, out_buf],
                push,
                plan.pv_workgroups,
            )?;

            let mut output = vec![0.0_f32; q_elems];
            self.device.read_buffer_f32(out_buf, &mut output)?;
            if output.iter().any(|val| !val.is_finite()) {
                return Err(Error::InvalidInput(
                    "GPU decomposed attention produced non-finite output".to_owned(),
                ));
            }

            let stages = if capture_stages {
                let mut scores = vec![0.0_f32; score_elems];
                let mut rounded_probs = vec![0.0_f32; score_elems];
                self.device.read_buffer_f32(scores_buf, &mut scores)?;
                self.device.read_buffer_f32(probs_buf, &mut rounded_probs)?;
                if scores
                    .iter()
                    .chain(&rounded_probs)
                    .any(|val| !val.is_finite())
                {
                    return Err(Error::InvalidInput(
                        "GPU attention intermediate stage produced non-finite values".to_owned(),
                    ));
                }
                Some(AttentionStageOutputs {
                    scores,
                    rounded_probs,
                })
            } else {
                None
            };

            Ok(AttentionOutput {
                output,
                witness: self.witness(plan, cache_key, 3),
                stages,
            })
        })();

        self.device.free_buffer(probs_buf);
        self.device.free_buffer(scores_buf);
        result
    }

    fn alloc_and_write_f32(&mut self, values: &[f32]) -> Result<BufferId, Error> {
        let bytes = u64::try_from(values.len())
            .ok()
            .and_then(|e| e.checked_mul(4))
            .ok_or_else(|| Error::ResourceLimit("buffer size overflow".to_owned()))?;
        let id = self.device.create_storage_buffer(bytes)?;
        if let Err(err) = self.device.write_buffer_f32(id, values) {
            self.device.free_buffer(id);
            return Err(err);
        }
        Ok(id)
    }

    fn free_buffers(&mut self, ids: &[BufferId]) {
        for &id in ids {
            self.device.free_buffer(id);
        }
    }

    fn witness(
        &self,
        plan: crate::AttentionTilePlan,
        cache_key: String,
        stage_count: u32,
    ) -> AttentionExecutionWitness {
        AttentionExecutionWitness {
            plan,
            cache_key,
            stage_count,
            qk_shader_digest: self.qk_digest.clone(),
            softmax_shader_digest: self.softmax_digest.clone(),
            pv_shader_digest: self.pv_digest.clone(),
            fused_shader_digest: self.fused_digest.clone(),
        }
    }
}

const fn attention_push_constants(desc: &AttentionDescriptor) -> [u32; 8] {
    [
        desc.query_len,
        desc.total_seq,
        desc.past_len,
        desc.num_heads,
        desc.num_kv_heads,
        desc.head_dim,
        desc.scale.to_bits(),
        if desc.causal { 1 } else { 0 },
    ]
}

fn validate_attention_inputs(
    desc: &AttentionDescriptor,
    q: &[f32],
    k: &[f32],
    v: &[f32],
) -> Result<(usize, usize, usize), Error> {
    let (q_elems, kv_elems, score_elems) = desc.validate()?;
    if q.len() != q_elems || k.len() != kv_elems || v.len() != kv_elems {
        return Err(Error::InvalidInput(format!(
            "mismatched Q/K/V lengths: got ({}, {}, {}), expected ({q_elems}, {kv_elems}, {kv_elems})",
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
    Ok((q_elems, kv_elems, score_elems))
}

fn validate_matrix(
    tensor: &Tensor,
    expected_kind: u32,
    col_alignment: usize,
) -> Result<(usize, usize), Error> {
    if tensor.kind != expected_kind {
        return Err(Error::InvalidInput(format!(
            "tensor '{}' has kind {}, expected {expected_kind}",
            tensor.name, tensor.kind
        )));
    }
    let [cols, rows] = tensor.dimensions.as_slice() else {
        return Err(Error::InvalidInput(format!(
            "tensor '{}' must be rank 2, got {:?}",
            tensor.name, tensor.dimensions
        )));
    };
    let (cols, rows) = (*cols, *rows);
    if cols == 0 || rows == 0 || !cols.is_multiple_of(col_alignment) {
        return Err(Error::InvalidInput(format!(
            "tensor '{}' has invalid extents [{cols}, {rows}] for alignment {col_alignment}",
            tensor.name
        )));
    }
    let block_bytes = match expected_kind {
        KIND_PQ2 => PQ2_BLOCK_BYTES,
        KIND_PTQ1 => PTQ1_BLOCK_BYTES,
        _ => unreachable!(),
    };
    let expected_bytes = (cols / 128)
        .checked_mul(block_bytes)
        .and_then(|row_bytes| row_bytes.checked_mul(rows))
        .ok_or_else(|| Error::ResourceLimit("tensor byte size overflow".to_owned()))?;
    if tensor.data.len() != expected_bytes {
        return Err(Error::InvalidInput(format!(
            "tensor '{}' byte length {} != expected {expected_bytes}",
            tensor.name,
            tensor.data.len()
        )));
    }
    Ok((cols, rows))
}

fn validate_weight_scales(kind: u32, data: &[u8]) -> Result<(), Error> {
    match kind {
        KIND_PQ2 => {
            for block in data.as_chunks::<PQ2_BLOCK_BYTES>().0 {
                let bits = u16::from_le_bytes([block[0], block[1]]);
                check_f16_scale(bits)?;
            }
        }
        KIND_PTQ1 => {
            for block in data.as_chunks::<PTQ1_BLOCK_BYTES>().0 {
                let bits = u16::from_le_bytes([block[26], block[27]]);
                check_f16_scale(bits)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn check_f16_scale(bits: u16) -> Result<(), Error> {
    if bits.trailing_zeros() >= 15 || (bits & 0x7C00) == 0x7C00 {
        return Err(Error::InvalidInput(
            "quantized weight block has zero or non-finite f16 scale".to_owned(),
        ));
    }
    Ok(())
}

fn upload_weight_bytes(
    device: &mut AdrenoVulkanDevice,
    kind: u32,
    raw: &[u8],
) -> Result<(BufferId, usize), Error> {
    let aligned = raw.len().next_multiple_of(4);
    let total_bytes = if kind == KIND_PQ2 {
        aligned + 4
    } else {
        aligned
    }
    .max(4);
    let total_u64 = u64::try_from(total_bytes)
        .map_err(|_| Error::ResourceLimit("weight buffer byte size overflow".to_owned()))?;
    let buf = device.create_storage_buffer(total_u64)?;
    let mut padded = vec![0_u8; total_bytes];
    padded[..raw.len()].copy_from_slice(raw);
    if let Err(err) = device.write_buffer_bytes(buf, &padded) {
        device.free_buffer(buf);
        return Err(err);
    }
    Ok((buf, total_bytes))
}

fn pack_q8k_blocks(blocks: &[Q8KBlock]) -> Result<Vec<u32>, Error> {
    let mut words = Vec::with_capacity(blocks.len() * Q8K_WORDS_PER_BLOCK);
    for block in blocks {
        if !block.scale.is_finite() {
            return Err(Error::InvalidInput(
                "Q8_K activation block has non-finite scale".to_owned(),
            ));
        }
        words.push(block.scale.to_bits());
        for chunk in block.values.as_chunks::<4>().0 {
            words.push(u32::from_le_bytes([
                chunk[0].cast_unsigned(),
                chunk[1].cast_unsigned(),
                chunk[2].cast_unsigned(),
                chunk[3].cast_unsigned(),
            ]));
        }
    }
    Ok(words)
}

fn pack_q8_blocks(blocks: &[Q8Block]) -> Result<Vec<u32>, Error> {
    let mut words = Vec::with_capacity(blocks.len() * Q8_0_WORDS_PER_BLOCK);
    for block in blocks {
        if !block.scale.is_finite() {
            return Err(Error::InvalidInput(
                "Q8_0 activation block has non-finite scale".to_owned(),
            ));
        }
        words.push(block.scale.to_bits());
        for chunk in block.values.as_chunks::<4>().0 {
            words.push(u32::from_le_bytes([
                chunk[0].cast_unsigned(),
                chunk[1].cast_unsigned(),
                chunk[2].cast_unsigned(),
                chunk[3].cast_unsigned(),
            ]));
        }
    }
    Ok(words)
}
