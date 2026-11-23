//! Exact immutable identities for the explicitly selected private service profile.

use bonsai_prism_runtime::{
    graph::KvCachePolicy,
    model::{
        lora::SWIFT_ADAPTER_SHA256,
        swift::{self, Representation},
    },
    runtime::PromptReuse,
    swift_session::{CpuPrefill, CpuSpeculation},
};
use serde_json::{Value, json};
use std::sync::Arc;

use super::Placement;

/// Model-thread observations frozen after accelerator initialization completes.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) enum ProjectionStartup {
    #[default]
    Cpu,
    External(Arc<str>),
    #[cfg_attr(
        not(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc")),
        allow(
            dead_code,
            reason = "Only Windows ARM64 initializes a native Hexagon executor; other targets retain the wire-validation controls."
        )
    )]
    NativeQ2 {
        resident_tensors: u64,
        resident_weight_bytes: u64,
        host_metadata_bytes: usize,
    },
    #[cfg_attr(
        not(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc")),
        allow(
            dead_code,
            reason = "Only Windows ARM64 initializes a native Hexagon executor; other targets retain the wire-validation controls."
        )
    )]
    NativeQ2Partial {
        architecture: u32,
        resident_tensors: u64,
        resident_weight_bytes: u64,
        host_metadata_bytes: usize,
    },
    #[cfg_attr(
        not(windows),
        allow(
            dead_code,
            reason = "Only Windows builds initialize the Adreno Vulkan projection executor."
        )
    )]
    AdrenoVulkan {
        identity: Arc<str>,
        resident_tensors: u64,
        resident_weight_bytes: u64,
    },
    #[cfg_attr(
        not(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc")),
        allow(
            dead_code,
            reason = "Only Windows ARM64 builds initialize the Hexagon + Adreno Vulkan composite executor."
        )
    )]
    Composite {
        identity: Arc<str>,
        npu_resident_tensors: u64,
        gpu_resident_tensors: u64,
        resident_weight_bytes: u64,
        host_metadata_bytes: usize,
    },
}

fn valid_ascii_graphic_identity(identity: &str) -> bool {
    !identity.is_empty()
        && identity.len() <= 256
        && identity.bytes().all(|byte| byte.is_ascii_graphic())
}

impl ProjectionStartup {
    pub(super) fn valid(&self, placement: Placement) -> bool {
        match self {
            Self::Cpu => matches!(placement, Placement::Auto | Placement::Cpu),
            Self::External(identity) => {
                placement == Placement::Auto && valid_ascii_graphic_identity(identity)
            }
            Self::NativeQ2 {
                resident_tensors,
                resident_weight_bytes,
                host_metadata_bytes,
            } => {
                placement == Placement::Auto
                    && *resident_tensors == 168
                    && *resident_weight_bytes == 2_888_663_040
                    && (1..=524_288).contains(host_metadata_bytes)
            }
            Self::NativeQ2Partial {
                architecture,
                resident_tensors,
                resident_weight_bytes,
                host_metadata_bytes,
            } => {
                placement == Placement::Auto
                    && matches!(*architecture, 73 | 81)
                    && (1..168).contains(resident_tensors)
                    && (1..2_888_663_040).contains(resident_weight_bytes)
                    && (1..=524_288).contains(host_metadata_bytes)
            }
            Self::AdrenoVulkan {
                identity,
                resident_tensors,
                resident_weight_bytes,
            } => {
                placement == Placement::Vulkan
                    && valid_ascii_graphic_identity(identity)
                    && *resident_tensors >= 1
                    && *resident_weight_bytes >= 1
            }
            Self::Composite {
                identity,
                npu_resident_tensors,
                gpu_resident_tensors,
                resident_weight_bytes,
                host_metadata_bytes,
            } => {
                placement == Placement::Composite
                    && valid_ascii_graphic_identity(identity)
                    && *npu_resident_tensors >= 1
                    && *gpu_resident_tensors >= 1
                    && *resident_weight_bytes >= 1
                    && (1..=524_288).contains(host_metadata_bytes)
            }
        }
    }

    fn readiness(&self, placement: Placement) -> Value {
        let (effective, identity, tensors, weights, metadata) = match self {
            Self::Cpu => ("cpu", None, 0, 0, 0),
            Self::External(identity) => ("external", Some(identity.as_ref()), 0, 0, 0),
            Self::NativeQ2 {
                resident_tensors,
                resident_weight_bytes,
                host_metadata_bytes,
            } => (
                "hexagon-native-q2",
                Some("hexagon-hvx-native-q2-resident-v1-experimental"),
                *resident_tensors,
                *resident_weight_bytes,
                *host_metadata_bytes,
            ),
            Self::NativeQ2Partial {
                architecture,
                resident_tensors,
                resident_weight_bytes,
                host_metadata_bytes,
            } => (
                "hexagon-native-q2-partial",
                match *architecture {
                    73 => Some("hexagon-v73-hvx-native-q2-partial-v1-experimental"),
                    _ => Some("hexagon-v81-hvx-native-q2-partial-v1-experimental"),
                },
                *resident_tensors,
                *resident_weight_bytes,
                *host_metadata_bytes,
            ),
            Self::AdrenoVulkan {
                identity,
                resident_tensors,
                resident_weight_bytes,
            } => (
                "adreno-vulkan",
                Some(identity.as_ref()),
                *resident_tensors,
                *resident_weight_bytes,
                0,
            ),
            Self::Composite {
                identity,
                npu_resident_tensors,
                gpu_resident_tensors,
                resident_weight_bytes,
                host_metadata_bytes,
            } => (
                "composite-hexagon-vulkan",
                Some(identity.as_ref()),
                npu_resident_tensors.saturating_add(*gpu_resident_tensors),
                *resident_weight_bytes,
                *host_metadata_bytes,
            ),
        };
        let requested = match placement {
            Placement::Auto => "auto",
            Placement::Cpu => "cpu",
            Placement::Vulkan => "vulkan",
            Placement::Composite => "composite",
        };
        json!({
            "schema_version": 1,
            "requested": requested,
            "effective": effective,
            "identity": identity,
            "resident_tensors": tensors,
            "resident_weight_bytes": weights,
            "host_metadata_bytes": metadata,
        })
    }
}

#[derive(Clone)]
pub(super) struct ModelProfile {
    pub placement: Placement,
    pub projection_startup: ProjectionStartup,
    pub representation: Representation,
    pub adapter: bool,
    pub daisyrose_adapter: bool,
    pub daisyrose_adapter_sha256: Option<Arc<str>>,
    pub daisyrose_adapter_max_rank: Option<u16>,
    pub vision: bool,
    pub image_max_tokens: Option<usize>,
    pub prompt_reuse: PromptReuse,
    pub prompt_reuse_checkpoints: usize,
    pub row_workers: usize,
    pub context_tokens: usize,
    pub kv_cache: KvCachePolicy,
    pub cpu_prefill: Option<CpuPrefill>,
    pub cpu_speculation: Option<CpuSpeculation>,
    pub startup_graph_identity: Option<Arc<str>>,
}

impl Default for ModelProfile {
    fn default() -> Self {
        Self {
            placement: Placement::Auto,
            projection_startup: ProjectionStartup::default(),
            representation: Representation::default(),
            adapter: false,
            daisyrose_adapter: false,
            daisyrose_adapter_sha256: None,
            daisyrose_adapter_max_rank: None,
            vision: false,
            image_max_tokens: None,
            prompt_reuse: PromptReuse::default(),
            prompt_reuse_checkpoints: 1,
            row_workers: 1,
            context_tokens: swift::CONTEXT_TOKENS,
            kv_cache: KvCachePolicy::F16,
            cpu_prefill: None,
            cpu_speculation: None,
            startup_graph_identity: None,
        }
    }
}

impl ModelProfile {
    pub(super) fn readiness(&self, seed: u32) -> Value {
        let mut model = json!({
            "id": match self.representation {
                Representation::Pq2 => "Swift-Bonsai-2-PQ2_0",
                Representation::CurrentRepackedPtq1 => "Swift-Bonsai-2-current-repacked-PTQ1_0",
            },
            "object": "model", "model_sha256": self.representation.model_sha256(),
            "template_sha256": swift::TEMPLATE_SHA256, "runtime": "prism-inference",
            "supports_cognitive_controls": self.placement != Placement::Vulkan,
            "context_tokens": self.context_tokens, "seed": seed,
            "row_workers": self.row_workers,
            "startup_projection": self.projection_startup.readiness(self.placement),
        });
        match self.placement {
            Placement::Auto => {}
            Placement::Cpu => {
                model["placement"] = json!("cpu");
                model["projection_backend"] = json!("cpu");
            }
            Placement::Vulkan => {
                model["placement"] = json!("vulkan");
                model["projection_backend"] = json!("adreno-vulkan");
            }
            Placement::Composite => {
                model["placement"] = json!("composite");
                model["projection_backend"] = json!("composite-hexagon-vulkan");
            }
        }
        if self.representation != Representation::Pq2 || self.adapter {
            model["weight_representation"] = json!(self.representation.name());
            model["adapter_sha256"] = json!(self.adapter.then_some(SWIFT_ADAPTER_SHA256));
            model["supports_abliteration_scale"] = json!(self.adapter);
            if self.adapter {
                model["abliteration_scale_range"] = json!([0, 2]);
            }
        }
        if self.daisyrose_adapter {
            model["supports_daisyrose_scale"] = json!(true);
            model["daisyrose_scale_range"] = json!([0, 2]);
            model["daisyrose_adapter_sha256"] = json!(self.daisyrose_adapter_sha256.as_deref());
            model["daisyrose_adapter_max_rank"] = json!(self.daisyrose_adapter_max_rank);
        }
        if self.vision {
            model["vision"] = json!(true);
            model["mmproj_sha256"] = json!(swift::MMPROJ_SHA256);
            if let Some(max_tokens) = self.image_max_tokens {
                model["image_max_tokens"] = json!(max_tokens);
            }
        }
        match self.prompt_reuse {
            PromptReuse::Disabled => {}
            PromptReuse::Latest => model["prompt_reuse"] = json!("latest"),
            PromptReuse::PromptEndAndLatest => {
                model["prompt_reuse"] = json!("prompt-end-and-latest");
            }
        }
        if self.prompt_reuse_checkpoints > 1 {
            model["prompt_reuse_checkpoints"] = json!(self.prompt_reuse_checkpoints);
        }
        if self.kv_cache != KvCachePolicy::F16 {
            model["kv_cache"] = json!(self.kv_cache.name());
        }
        if let Some(prefill) = self.cpu_prefill {
            model["cpu_prefill_tokens"] = json!(prefill.tokens());
            model["cpu_prefill_scratch_bytes"] = json!(prefill.scratch_bytes());
            if prefill.chunkwise_gdn() {
                model["cpu_prefill_gdn"] = json!("wy-f32");
            }
        }
        if let Some(speculation) = self.cpu_speculation {
            model["cpu_speculative_tokens"] = json!(speculation.tokens());
            model["cpu_verification_scratch_bytes"] =
                json!(speculation.verification_scratch_bytes());
            model["cpu_speculative_checkpoint_bytes"] = json!(speculation.checkpoint_bytes());
        }
        if self.kv_cache != KvCachePolicy::F16 || self.cpu_prefill.is_some() {
            // Adapter scales may change between requests. This records the
            // actual startup graph, never a claim about its current scale.
            model["startup_graph_identity"] = json!(self.startup_graph_identity.as_deref());
        }
        json!({"data": [model]})
    }
}
