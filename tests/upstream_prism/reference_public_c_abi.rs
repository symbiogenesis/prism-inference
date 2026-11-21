//! Clean-room independent safe Rust C17 public ABI layout reference provider.
//!
//! No production `bonsai_prism_runtime` code, native C bridge or precomputed
//! answer fixture is called here. Declares clean-room `#[repr(C)]` mirror
//! descriptors for C17 public parameter and batch layouts and scalar widths.

use serde::Deserialize;
use serde_json::{Value, json};

use super::{Result, invalid};

pub(super) const ID: &str = "tests/test-c.c";
const SOURCE_SHA: &str = "c973865e5e86263a71a1210003e80cffa6269572de31cfa647cc3d01b4bea99e";
const MAX_OPERATIONS: usize = 128;
const C_STANDARD: u64 = 201_710;

type Token = i32;
type Position = i32;
type SequenceId = i32;

#[repr(C)]
struct ModelParams {
    devices: usize,
    tensor_buft_overrides: usize,
    n_gpu_layers: i32,
    split_mode: i32,
    main_gpu: i32,
    tensor_split: usize,
    progress_callback: usize,
    progress_callback_user_data: usize,
    kv_overrides: usize,
    vocab_only: bool,
    use_mmap: bool,
    use_direct_io: bool,
    use_mlock: bool,
    check_tensors: bool,
    use_extra_bufts: bool,
    no_host: bool,
}

#[repr(C)]
struct ContextParams {
    n_ctx: u32,
    n_batch: u32,
    n_ubatch: u32,
    n_seq_max: u32,
    n_threads: i32,
    n_threads_batch: i32,
    rope_scaling_type: i32,
    pooling_type: i32,
    attention_type: i32,
    flash_attn_type: i32,
    rope_freq_base: f32,
    rope_freq_scale: f32,
    yarn_ext_factor: f32,
    yarn_attn_factor: f32,
    yarn_beta_fast: f32,
    yarn_beta_slow: f32,
    yarn_orig_ctx: u32,
    defrag_thold: f32,
    cb_eval: usize,
    cb_eval_user_data: usize,
    type_k: i32,
    type_v: i32,
    abort_callback: usize,
    abort_callback_data: usize,
    embeddings: bool,
    offload_kqv: bool,
    no_perf: bool,
    op_offload: bool,
    swa_full: bool,
    kv_unified: bool,
    samplers: usize,
    n_samplers: usize,
}

#[repr(C)]
struct Batch {
    n_tokens: i32,
    token: usize,
    embd: usize,
    pos: usize,
    n_seq_id: usize,
    seq_id: usize,
    logits: usize,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
struct Request {
    schema_version: u32,
    operations: Vec<Operation>,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
struct Operation {
    kind: String,
}

fn layout_observation<T>() -> Result<Value> {
    let size = u64::try_from(size_of::<T>())?;
    let alignment = u64::try_from(align_of::<T>())?;
    Ok(json!({
        "Size": size,
        "Alignment": alignment,
    }))
}

fn observe_abi() -> Result<Value> {
    let token_size = u64::try_from(size_of::<Token>())?;
    let position_size = u64::try_from(size_of::<Position>())?;
    let sequence_id_size = u64::try_from(size_of::<SequenceId>())?;
    Ok(json!({
        "ModelParams": layout_observation::<ModelParams>()?,
        "ContextParams": layout_observation::<ContextParams>()?,
        "Batch": layout_observation::<Batch>()?,
        "TokenSize": token_size,
        "PositionSize": position_size,
        "SequenceIdSize": sequence_id_size,
        "CStandard": C_STANDARD,
    }))
}

pub(super) fn handles(request: &Value) -> bool {
    request["Operations"].as_array().is_some_and(|operations| {
        !operations.is_empty()
            && operations
                .iter()
                .all(|operation| operation["Kind"] == "public-c-abi")
    })
}

pub(super) fn execute(request: &Value) -> Result<Value> {
    if !request["Operations"]
        .as_array()
        .is_some_and(|operations| (1..=MAX_OPERATIONS).contains(&operations.len()))
    {
        return Err(invalid(
            "independent public-c-abi requires 1 through 128 operations",
        ));
    }
    let parsed: Request = serde_json::from_value(request.clone())?;
    if parsed.schema_version != 1
        || parsed
            .operations
            .iter()
            .any(|operation| operation.kind != "public-c-abi")
    {
        return Err(invalid(
            "unsupported independent public-c-abi schema version or operation kind",
        ));
    }
    let observation = observe_abi()?;
    let results = vec![observation; parsed.operations.len()];
    Ok(json!({
        "SchemaVersion": 1,
        "Results": results,
    }))
}

pub(super) fn identity() -> Value {
    json!({
        "Kind": "independent-rust-public-c-abi",
        "SupportedSourceIds": [ID],
        "UpstreamCommit": super::UPSTREAM,
        "AssertionSource": {"Path": ID, "Sha256": SOURCE_SHA},
        "Architecture": "clean-room-repr-c-c17-public-abi-layout-descriptor-v1",
        "NativeReferenceReplay": false,
        "ProductionRuntimeCalled": false,
        "CompleteSourceGate": false,
        "NativeRetirementQualified": false
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, Target};

    #[test]
    fn independent_public_c_abi_passes_on_reference_and_cpp_while_rust_fails() -> Result<()> {
        let reference = Engine::new(Target::RustReference, None, None, None, None)?;
        let reference_report = crate::public_c::run(&reference)?;
        assert_eq!(reference_report["Id"], ID);
        assert_eq!(reference_report["Status"], "pass", "{reference_report}");

        let cpp = Engine::new(Target::Cpp, None, None, None, None)?;
        let cpp_report = crate::public_c::run(&cpp)?;
        assert_eq!(cpp_report["Id"], ID);
        assert_eq!(cpp_report["Status"], "pass", "{cpp_report}");
        assert_eq!(cpp_report, reference_report);

        let rust = Engine::new(Target::Rust, None, None, None, None)?;
        let rust_report = crate::public_c::run(&rust)?;
        assert_eq!(rust_report["Id"], ID);
        assert_eq!(rust_report["Status"], "fail", "{rust_report}");

        assert!(crate::reference_provider::supports(ID));
        let info = identity();
        assert_eq!(info["Kind"], "independent-rust-public-c-abi");
        assert_eq!(info["SupportedSourceIds"], json!([ID]));
        assert_eq!(info["UpstreamCommit"], crate::UPSTREAM);
        assert_eq!(
            info["AssertionSource"],
            json!({"Path": ID, "Sha256": SOURCE_SHA})
        );
        assert_eq!(info["NativeReferenceReplay"], false);
        assert_eq!(info["ProductionRuntimeCalled"], false);
        assert_eq!(info["CompleteSourceGate"], false);
        assert_eq!(info["NativeRetirementQualified"], false);
        Ok(())
    }

    #[test]
    fn independent_public_c_abi_observes_multiple_ops_and_rejects_malformed_requests() -> Result<()>
    {
        let multi = json!({
            "SchemaVersion": 1,
            "Operations": [
                {"Kind": "public-c-abi"},
                {"Kind": "public-c-abi"}
            ]
        });
        assert!(handles(&multi));
        let response = execute(&multi)?;
        assert_eq!(response["SchemaVersion"], 1);
        let results = response["Results"]
            .as_array()
            .ok_or_else(|| invalid("missing Results array"))?;
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], results[1]);
        assert_eq!(results[0]["TokenSize"], 4);
        assert_eq!(results[0]["PositionSize"], 4);
        assert_eq!(results[0]["SequenceIdSize"], 4);
        assert_eq!(results[0]["CStandard"], C_STANDARD);

        assert!(!handles(&json!({"SchemaVersion": 1, "Operations": []})));
        assert!(!handles(
            &json!({"SchemaVersion": 1, "Operations": [{"Kind": "other"}]})
        ));
        assert!(!handles(&json!({
            "SchemaVersion": 1,
            "Operations": [{"Kind": "public-c-abi"}, {"Kind": "other"}]
        })));
        assert!(!handles(&json!({"Operations": "invalid"})));

        for invalid_request in [
            json!({"SchemaVersion": 0, "Operations": [{"Kind": "public-c-abi"}]}),
            json!({"SchemaVersion": 2, "Operations": [{"Kind": "public-c-abi"}]}),
            json!({"SchemaVersion": true, "Operations": [{"Kind": "public-c-abi"}]}),
            json!({"SchemaVersion": 1.0, "Operations": [{"Kind": "public-c-abi"}]}),
            json!({"SchemaVersion": 1, "Operations": []}),
            json!({"SchemaVersion": 1, "Operations": vec![json!({"Kind": "public-c-abi"}); MAX_OPERATIONS + 1]}),
            json!({"SchemaVersion": 1, "Operations": [{"Kind": "public-c-abi", "Extra": 1}]}),
            json!({"SchemaVersion": 1, "Operations": [{"Kind": "unknown"}]}),
            json!({"SchemaVersion": 1, "Operations": [{}]}),
            json!({"SchemaVersion": 1, "Operations": [{"Kind": "public-c-abi"}], "Extra": 1}),
        ] {
            assert!(execute(&invalid_request).is_err(), "{invalid_request}");
        }
        Ok(())
    }
}
