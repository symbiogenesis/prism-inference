//! Native Rust SPIR-V Vulkan compute backend for Qualcomm Adreno GPUs.
//!
//! Targets Windows ARM64 Snapdragon X Elite (`X1`) and Snapdragon X2 Elite Extreme
//! (`X2`) Adreno GPUs via dynamic loading of `%SystemRoot%\System32\vulkan-1.dll`,
//! runtime-emitted SPIR-V 1.3 compute kernels, `PQ2_0`/`PTQ1_0` integer projection
//! execution, and resource-pressure-selected fused vs. 3-stage decomposed attention
//! preserving Prism binary16 probability rounding.

use std::fmt;

use bonsai_native_hash::Hasher;

pub mod spirv;

/// Error returned by native Adreno Vulkan initialization, planning, or execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Error {
    /// Required Vulkan driver, Qualcomm Adreno GPU, or feature is unavailable.
    Unsupported,
    /// Caller supplied invalid tensor geometry, non-finite inputs, or unsupported encoding.
    InvalidInput(String),
    /// Workload exceeds device storage buffer range, shared memory, scratch budget, or TDR guard.
    ResourceLimit(String),
    /// Operation was cancelled cooperatively between execution stages.
    Cancelled,
    /// GPU wait timed out or reported device loss; owning device requires terminal teardown.
    DeviceFault(String),
    /// Underlying Vulkan API call failed.
    Vulkan(String),
}

impl Error {
    /// Converts this error into a boxed [`bonsai_prism_runtime::Error`].
    #[must_use]
    pub fn into_runtime_error(self) -> bonsai_prism_runtime::Error {
        Box::new(self)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported => {
                formatter.write_str("native Qualcomm Adreno Vulkan compute support is unavailable")
            }
            Self::InvalidInput(message) => write!(formatter, "invalid input: {message}"),
            Self::ResourceLimit(message) => write!(formatter, "resource limit exceeded: {message}"),
            Self::Cancelled => formatter.write_str("GPU operation cancelled between stages"),
            Self::DeviceFault(message) => {
                write!(
                    formatter,
                    "Vulkan device fault (teardown required): {message}"
                )
            }
            Self::Vulkan(message) => write!(formatter, "Vulkan driver error: {message}"),
        }
    }
}

impl std::error::Error for Error {}

/// Qualcomm Snapdragon Adreno GPU generation classification.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum AdrenoFamily {
    /// First-generation Snapdragon X Elite (`Adreno X1-85` / `X1`).
    X1,
    /// Second-generation Snapdragon X2 Elite Extreme (`Adreno X2-90` / `X2`).
    X2,
    /// Another Qualcomm Vulkan adapter (`vendor_id == 0x5143`).
    OtherQualcomm,
}

impl AdrenoFamily {
    /// Classifies an Adreno generation conservatively from the reported Vulkan device name.
    #[must_use]
    pub fn from_device_name(name: &str) -> Self {
        if name.contains("X2-") || name.contains("Adreno(TM) X2") || name.contains("Adreno X2") {
            Self::X2
        } else if name.contains("X1-")
            || name.contains("Adreno(TM) X1")
            || name.contains("Adreno X1")
        {
            Self::X1
        } else {
            Self::OtherQualcomm
        }
    }

    /// Short ASCII-graphic tag suitable for executor identity strings.
    #[must_use]
    pub const fn as_tag(self) -> &'static str {
        match self {
            Self::X1 => "x1",
            Self::X2 => "x2",
            Self::OtherQualcomm => "qcom",
        }
    }
}

/// Supported Vulkan `VK_KHR_cooperative_matrix` tile shape reported by the adapter.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct CoopMatShape {
    /// Tile row count `M`.
    pub m_size: u32,
    /// Tile column count `N`.
    pub n_size: u32,
    /// Tile inner reduction count `K`.
    pub k_size: u32,
    /// Component type name for matrix `A`.
    pub a_type: String,
    /// Component type name for matrix `B`.
    pub b_type: String,
    /// Component type name for accumulator matrix `C`.
    pub c_type: String,
    /// Component type name for output matrix `Result`.
    pub result_type: String,
    /// SPIR-V execution scope ID (`3` = Subgroup).
    pub scope: u32,
    /// Whether saturating integer accumulation is supported for this shape.
    pub saturating_accumulation: bool,
}

/// Compute queue and memory limits reported by the selected Vulkan physical device.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct VulkanComputeLimits {
    /// `maxComputeSharedMemorySize` in bytes.
    pub max_compute_shared_memory_size: u32,
    /// `maxComputeWorkGroupCount` (`[x, y, z]`).
    pub max_compute_work_group_count: [u32; 3],
    /// `maxComputeWorkGroupInvocations`.
    pub max_compute_work_group_invocations: u32,
    /// `maxComputeWorkGroupSize` (`[x, y, z]`).
    pub max_compute_work_group_size: [u32; 3],
    /// `maxStorageBufferRange` in bytes.
    pub max_storage_buffer_range: u32,
    /// `maxPushConstantsSize` in bytes.
    pub max_push_constants_size: u32,
}

/// Per-heap budget and usage observation from `VK_EXT_memory_budget` or heap totals.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct MemoryHeapBudget {
    /// Vulkan physical memory heap index.
    pub heap_index: usize,
    /// Total heap size reported in `VkMemoryHeap.size`.
    pub size_bytes: u64,
    /// OS-estimated budget in bytes (falls back to `size_bytes` when budget extension is absent).
    pub budget_bytes: u64,
    /// Current usage estimate in bytes.
    pub usage_bytes: u64,
    /// Whether `VK_MEMORY_HEAP_DEVICE_LOCAL_BIT` is set on this heap.
    pub device_local: bool,
}

/// Complete capability snapshot and deterministic SHA-256 identity of an Adreno Vulkan adapter.
#[derive(Clone, Debug, Eq, PartialEq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "Represents independent Vulkan physical-device feature flags."
)]
pub struct VulkanDeviceCapabilities {
    /// Physical device name string (`deviceName`).
    pub device_name: String,
    /// Classified Adreno generation.
    pub family: AdrenoFamily,
    /// PCI/Khronos vendor ID (`0x5143` for native Qualcomm Adreno).
    pub vendor_id: u32,
    /// Adapter device ID.
    pub device_id: u32,
    /// Driver version word.
    pub driver_version: u32,
    /// Physical device Vulkan API version word.
    pub api_version: u32,
    /// Preferred/native compute subgroup width (`64` on Adreno X1/X2).
    pub subgroup_size: u32,
    /// Whether `shaderFloat16` (`VK_KHR_shader_float16_int8`) is supported.
    pub shader_float16: bool,
    /// Whether `VK_KHR_shader_integer_dot_product` is supported.
    pub shader_integer_dot_product: bool,
    /// Whether `integerDotProduct4x8BitPackedSignedAccelerated` is reported `true`.
    pub integer_dot_4x8_packed_signed_accelerated: bool,
    /// Enumerated `VK_KHR_cooperative_matrix` tile shapes.
    pub cooperative_matrix_shapes: Vec<CoopMatShape>,
    /// Whether `VK_QCOM_cooperative_matrix_conversion` is advertised.
    pub qcom_cooperative_matrix_conversion: bool,
    /// Device compute workgroup, shared-memory, storage, and push-constant limits.
    pub compute_limits: VulkanComputeLimits,
    /// Observed memory heaps and budgets.
    pub memory_heaps: Vec<MemoryHeapBudget>,
    /// Canonical lowercase SHA-256 digest of the capability fields above.
    pub identity_sha256: String,
}

impl VulkanDeviceCapabilities {
    /// Returns `true` when hardware-accelerated packed `4x8` signed integer dot product is available.
    #[must_use]
    pub const fn supports_integer_dot_4x8(&self) -> bool {
        self.shader_integer_dot_product && self.integer_dot_4x8_packed_signed_accelerated
    }

    /// Recomputes [`Self::identity_sha256`] canonically from all capability fields.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] if hashing fails or a length exceeds `u64`.
    pub fn recompute_identity_sha256(&mut self) -> Result<(), Error> {
        let mut hasher = Hasher::new().map_err(|err| Error::InvalidInput(err.to_string()))?;
        hash_str(
            &mut hasher,
            b"bonsai-adreno-vulkan-caps-v1\0",
            &self.device_name,
        )?;
        hash_u32(&mut hasher, self.vendor_id)?;
        hash_u32(&mut hasher, self.device_id)?;
        hash_u32(&mut hasher, self.driver_version)?;
        hash_u32(&mut hasher, self.api_version)?;
        hash_u32(&mut hasher, self.subgroup_size)?;
        hash_bytes(
            &mut hasher,
            &[
                u8::from(self.shader_float16),
                u8::from(self.shader_integer_dot_product),
                u8::from(self.integer_dot_4x8_packed_signed_accelerated),
                u8::from(self.qcom_cooperative_matrix_conversion),
            ],
        )?;
        hash_u32(
            &mut hasher,
            self.compute_limits.max_compute_shared_memory_size,
        )?;
        for value in self.compute_limits.max_compute_work_group_count {
            hash_u32(&mut hasher, value)?;
        }
        hash_u32(
            &mut hasher,
            self.compute_limits.max_compute_work_group_invocations,
        )?;
        for value in self.compute_limits.max_compute_work_group_size {
            hash_u32(&mut hasher, value)?;
        }
        hash_u32(&mut hasher, self.compute_limits.max_storage_buffer_range)?;
        hash_u32(&mut hasher, self.compute_limits.max_push_constants_size)?;
        hash_usize(&mut hasher, self.cooperative_matrix_shapes.len())?;
        for shape in &self.cooperative_matrix_shapes {
            hash_u32(&mut hasher, shape.m_size)?;
            hash_u32(&mut hasher, shape.n_size)?;
            hash_u32(&mut hasher, shape.k_size)?;
            hash_str(&mut hasher, b"", &shape.a_type)?;
            hash_str(&mut hasher, b"", &shape.b_type)?;
            hash_str(&mut hasher, b"", &shape.c_type)?;
            hash_str(&mut hasher, b"", &shape.result_type)?;
            hash_u32(&mut hasher, shape.scope)?;
            hash_bytes(&mut hasher, &[u8::from(shape.saturating_accumulation)])?;
        }
        hash_usize(&mut hasher, self.memory_heaps.len())?;
        for heap in &self.memory_heaps {
            hash_usize(&mut hasher, heap.heap_index)?;
            hash_u64(&mut hasher, heap.size_bytes)?;
            hash_u32(&mut hasher, u32::from(heap.device_local))?;
        }
        let digest = hasher
            .finish()
            .map_err(|err| Error::InvalidInput(err.to_string()))?;
        self.identity_sha256 = hex::encode(digest);
        Ok(())
    }
}

/// Selected SPIR-V integer dot-product execution path for `#39` projections.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ProjectionKernelPath {
    /// Hardware-accelerated `OpSDot` (`SPV_KHR_integer_dot_product`, `PackedVectorFormat4x8Bit`).
    IntegerDotPacked4x8,
    /// Portable SPIR-V core integer arithmetic fallback without `SPV_KHR_integer_dot_product`.
    PortableIntegerFallback,
}

impl ProjectionKernelPath {
    /// Short ASCII-graphic tag identifying the active projection kernel path.
    #[must_use]
    pub const fn as_tag(self) -> &'static str {
        match self {
            Self::IntegerDotPacked4x8 => "sdot4x8",
            Self::PortableIntegerFallback => "portable",
        }
    }
}

/// Selected Adreno attention kernel architecture for `#40`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum AdrenoAttentionPath {
    /// Single-dispatch fused workgroup attention with binary16 probability rounding.
    Fused,
    /// Three-stage decomposed compute dispatches (`QK -> Softmax+F16 -> PV`) with barriers.
    Decomposed3Stage,
}

impl AdrenoAttentionPath {
    /// Short ASCII-graphic tag identifying the selected attention path.
    #[must_use]
    pub const fn as_tag(self) -> &'static str {
        match self {
            Self::Fused => "fused",
            Self::Decomposed3Stage => "decomposed3",
        }
    }
}

/// Caller policy controlling resource-pressure selection between fused and decomposed attention.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum AttentionStrategy {
    /// Automatically choose [`AdrenoAttentionPath::Fused`] within shared-memory/TDR limits and
    /// fall back to [`AdrenoAttentionPath::Decomposed3Stage`] under sequence or register pressure.
    #[default]
    Auto,
    /// Require the single-dispatch fused attention kernel, failing if hardware/TDR limits reject it.
    ForceFused,
    /// Require the 3-stage decomposed attention pipeline.
    ForceDecomposed,
}

fn hash_bytes(hasher: &mut Hasher, bytes: &[u8]) -> Result<(), Error> {
    hasher
        .update(bytes)
        .map_err(|err| Error::InvalidInput(err.to_string()))
}

fn hash_u32(hasher: &mut Hasher, value: u32) -> Result<(), Error> {
    hash_bytes(hasher, &value.to_le_bytes())
}

fn hash_u64(hasher: &mut Hasher, value: u64) -> Result<(), Error> {
    hash_bytes(hasher, &value.to_le_bytes())
}

fn hash_usize(hasher: &mut Hasher, value: usize) -> Result<(), Error> {
    let encoded = u64::try_from(value)
        .map_err(|err| Error::InvalidInput(format!("usize overflow during hash: {err}")))?;
    hash_u64(hasher, encoded)
}

fn hash_str(hasher: &mut Hasher, prefix: &[u8], text: &str) -> Result<(), Error> {
    if !prefix.is_empty() {
        hash_bytes(hasher, prefix)?;
    }
    hash_usize(hasher, text.len())?;
    hash_bytes(hasher, text.as_bytes())
}
