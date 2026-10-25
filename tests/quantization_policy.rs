//! Focused policy resets, tied weights, expert roles and shape fallbacks.
#![forbid(unsafe_code)]

use bonsai_prism_runtime::model::quantization_policy::{
    ModelDescriptor, Parameters, Policy, TYPE_COUNT, TensorDescriptor, default_type,
};

type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

fn model(architecture: &str) -> ModelDescriptor {
    ModelDescriptor {
        architecture: architecture.to_owned(),
        embedding: 4096,
        feed_forward: 14336,
        layers: 16,
        heads: 16,
        kv_heads: 4,
        experts: 0,
        head_key: 256,
        head_value: 256,
    }
}

#[test]
fn tied_embedding_detection_and_precision_counters_reset_for_each_computation() -> TestResult {
    let policy = Policy::new(model("qwen3"), Parameters::default())?;
    let embedding = TensorDescriptor::new("token_embd.weight", [256, 1024, 1, 1])?;
    let output = TensorDescriptor::new("output.weight", [256, 1024, 1, 1])?;
    let value = TensorDescriptor::new("blk.0.attn_v.weight", [256, 256, 1, 1])?;
    let down = TensorDescriptor::new("blk.0.ffn_down.weight", [256, 256, 1, 1])?;
    assert_eq!(policy.compute(15, &[&embedding])?, [14]);
    assert_eq!(
        policy.compute(15, &[&embedding, &output, &value, &down])?,
        [12, 14, 14, 14]
    );
    assert_eq!(policy.compute(15, &[&embedding])?, [14]);
    assert_eq!(
        policy.compute(24, &[&embedding, &output, &value, &down])?,
        [10, 13, 12, 10]
    );
    assert_eq!(policy.compute(0, &[&embedding, &output])?, [0, 0]);
    assert_eq!(policy.compute(32, &[&embedding, &output])?, [30, 30]);
    Ok(())
}

#[test]
fn shape_fallbacks_and_importance_policy_preserve_native_order() -> TestResult {
    let policy = Policy::new(model("llama"), Parameters::default())?;
    let aligned = TensorDescriptor::new("blk.0.other.weight", [64, 2, 1, 1])?;
    let unusual = TensorDescriptor::new("blk.0.other.weight", [33, 2, 1, 1])?;
    let down = TensorDescriptor::new("blk.0.ffn_down.weight", [256, 2, 1, 1])?;
    assert_eq!(policy.compute(36, &[&aligned, &unusual])?, [2, 1]);
    assert_eq!(policy.compute(2, &[&down])?, [2]);
    let importance = Policy::new(
        model("llama"),
        Parameters {
            has_importance: true,
            ..Parameters::default()
        },
    )?;
    assert_eq!(importance.compute(2, &[&down])?, [3]);
    let pure = Policy::new(
        model("llama"),
        Parameters {
            pure: true,
            ..Parameters::default()
        },
    )?;
    assert_eq!(pure.compute(15, &[&down])?, [12]);
    assert_eq!(default_type(39), TYPE_COUNT);
    assert_eq!(default_type(142), TYPE_COUNT);
    assert_eq!(default_type(141), 142);
    assert!(policy.compute(39, &[&aligned]).is_err());
    Ok(())
}

#[test]
fn expert_tensor_roles_names_and_exclusions_are_checked() -> TestResult {
    let mut descriptor = model("bailingmoe3");
    descriptor.experts = 8;
    let policy = Policy::new(descriptor, Parameters::default())?;
    let expert = TensorDescriptor::new("blk.0.ffn_down_exps.weight", [256, 256, 8, 1])?;
    let projection = TensorDescriptor::new("blk.0.attn_kv_b.weight", [256, 256, 8, 1])?;
    assert_eq!(policy.compute(38, &[&expert, &projection])?, [39, 8]);
    let bad_layer = TensorDescriptor::new("blk.16.ffn_down_exps.weight", [256, 256, 8, 1])?;
    assert!(policy.compute(15, &[&bad_layer]).is_err());
    for name in [
        "output_norm.weight",
        "position_embd.weight",
        "token_types.weight",
        "blk.0.ffn_gate_inp.weight",
        "blk.0.ssm_conv1d.weight",
        "v.position_embd.weight",
        "mm.a.code_embd.weight",
    ] {
        assert!(!policy.allows(&TensorDescriptor::new(name, [256, 256, 1, 1])?));
    }
    assert!(!policy.allows(&TensorDescriptor::new("blk.0.norm.weight", [256, 1, 1, 1])?));
    // Native size_t subtraction gives this unusual five-byte name eligibility.
    assert!(policy.allows(&TensorDescriptor::new("abcde", [32, 2, 1, 1])?));
    assert!(TensorDescriptor::new("x.weight", [0, 2, 1, 1]).is_err());
    assert!(Policy::new(model("unknown-architecture"), Parameters::default()).is_err());
    Ok(())
}
