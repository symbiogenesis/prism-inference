//! Exact immutable identities for the explicitly selected private service profile.

use bonsai_prism_runtime::{
    model::{
        lora::SWIFT_ADAPTER_SHA256,
        swift::{self, Representation},
    },
    runtime::PromptReuse,
};
use serde_json::{Value, json};

#[derive(Clone, Copy, Default)]
pub(super) struct ModelProfile {
    pub representation: Representation,
    pub adapter: bool,
    pub prompt_reuse: PromptReuse,
}

impl ModelProfile {
    pub(super) fn readiness(self, seed: u32) -> Value {
        let mut model = json!({
            "id": match self.representation {
                Representation::Pq2 => "Swift-Bonsai-2-PQ2_0",
                Representation::CurrentRepackedPtq1 => "Swift-Bonsai-2-current-repacked-PTQ1_0",
            },
            "object": "model", "model_sha256": self.representation.model_sha256(),
            "template_sha256": swift::TEMPLATE_SHA256, "runtime": "prism-inference",
            "context_tokens": swift::CONTEXT_TOKENS, "seed": seed,
        });
        if self.representation != Representation::Pq2 || self.adapter {
            model["weight_representation"] = json!(self.representation.name());
            model["adapter_sha256"] = json!(self.adapter.then_some(SWIFT_ADAPTER_SHA256));
            model["supports_abliteration_scale"] = json!(self.adapter);
            if self.adapter {
                model["abliteration_scale_range"] = json!([0, 2]);
            }
        }
        if self.prompt_reuse == PromptReuse::Latest {
            model["prompt_reuse"] = json!("latest");
        }
        json!({"data": [model]})
    }
}
