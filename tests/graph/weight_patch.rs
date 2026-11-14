//! Original-basis observation, additive composition and checkpoint isolation.

use super::{
    Decoder, MatvecBackend, Qwen35, Result, fixture, rank_one_factors, rank_one_graph, weight,
};
use bonsai_gguf::MetadataValue;
use bonsai_prism_runtime::graph::weight_patch::{FfnPatch, FfnProbe};

fn active_ffn() -> super::Model {
    let mut model = fixture(false);
    for target in ["blk.0.ffn_up.weight", "blk.0.ffn_gate.weight"] {
        model.tensors.insert(
            target.into(),
            weight(target, &[2, 2], &[1.0, 0.0, 0.0, 1.0]),
        );
    }
    model
}

#[test]
fn ffn_probe_zero_preserves_unpatched_bits_and_observes_gated_basis() -> Result<()> {
    let graph = || Qwen35::with_backend(active_ffn(), 8, MatvecBackend::Scalar);
    let mut plain = FfnProbe::new(graph()?, 0, None)?;
    let patch = FfnPatch::new(vec![f32::MAX; 2], vec![1.0, -1.0], 0.0)?;
    let mut zero = FfnProbe::new(graph()?, 0, Some(patch))?;
    assert_ne!(plain.identity(), zero.identity());
    let mut observed = false;
    let mut plain_logits = Vec::new();
    plain.evaluate_observed_into(0, &mut plain_logits, &mut |row| {
        assert_eq!(row.layer, 0);
        assert_eq!(row.token_index, 0);
        assert_eq!(row.activation.len(), 2);
        assert_eq!(row.residual.len(), 2);
        assert!(row.activation.iter().any(|value| *value > 0.0));
        observed = true;
        Ok(())
    })?;
    assert!(observed);
    let zero_logits = zero.evaluate(0)?;
    assert_eq!(
        plain_logits
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        zero_logits
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
    );
    assert_eq!(plain.evaluate(1)?, zero.evaluate(1)?);
    Ok(())
}

#[test]
fn ffn_probe_composes_after_adapter_and_rejects_foreign_state() -> Result<()> {
    let base = active_ffn();
    let factors = rank_one_factors(&base, &["blk.0.ffn_down.weight"], 0.25);
    let graph = rank_one_graph(base, factors, 1.3)?;
    let old_checkpoint = graph.checkpoint()?;
    let patch = FfnPatch::new(vec![0.0, 1.0], vec![0.0, 0.5], 1.0)?;
    let mut probe = FfnProbe::new(graph, 0, Some(patch))?;
    assert!(probe.restore(&old_checkpoint).is_err());
    let mut original = FfnProbe::new(
        {
            let base = active_ffn();
            let factors = rank_one_factors(&base, &["blk.0.ffn_down.weight"], 0.25);
            rank_one_graph(base, factors, 1.3)?
        },
        0,
        None,
    )?;
    let mut plain_residual = Vec::new();
    let mut hidden = Vec::new();
    original.evaluate_observed_into(1, &mut Vec::new(), &mut |row| {
        plain_residual.extend_from_slice(row.residual);
        hidden.extend_from_slice(row.activation);
        Ok(())
    })?;
    probe.evaluate_observed_into(1, &mut Vec::new(), &mut |row| {
        assert_eq!(row.activation, hidden);
        assert_eq!(row.residual[0], plain_residual[0]);
        // The base fixture down projection is zero. Existing adapter output is
        // nonzero, so the rank-one branch must add rather than replace it.
        assert!(plain_residual[1] > 1.0);
        assert!(row.residual[1] > plain_residual[1]);
        Ok(())
    })?;
    let checkpoint = probe.checkpoint()?;
    let next = probe.evaluate(0)?;
    probe.restore(&checkpoint)?;
    assert_eq!(next, probe.evaluate(0)?);
    assert_eq!(probe.batched_evaluation_capacity(), 0);
    assert!(
        probe
            .evaluate_batch_into(&[0], &mut Vec::new(), &|| Ok(()))
            .is_err()
    );
    Ok(())
}

#[test]
fn ffn_probe_callback_failure_commits_no_token_and_bounds_reject() -> Result<()> {
    let graph = || Qwen35::with_backend(active_ffn(), 8, MatvecBackend::Scalar);
    let mut probe = FfnProbe::new(graph()?, 0, None)?;
    let before = probe.checkpoint()?;
    let mut output = vec![1.0];
    assert!(
        probe
            .evaluate_observed_into(0, &mut output, &mut |_| {
                Err(std::io::Error::other("selected capture failure").into())
            })
            .is_err()
    );
    assert_eq!(output, Vec::<f32>::new());
    probe.restore(&before)?;
    let mut fresh = FfnProbe::new(graph()?, 0, None)?;
    assert_eq!(probe.evaluate(0)?, fresh.evaluate(0)?);
    assert!(FfnProbe::new(graph()?, 1, None).is_err());
    assert!(
        FfnProbe::new(
            graph()?,
            0,
            Some(FfnPatch::new(vec![1.0], vec![1.0, 0.0], 1.0)?),
        )
        .is_err()
    );
    let mut nonempty = graph()?;
    nonempty.evaluate(0)?;
    assert!(FfnProbe::new(nonempty, 0, None).is_err());
    Ok(())
}

#[test]
fn ffn_probe_captures_original_gated_values_before_hadamard() -> Result<()> {
    let mut model = active_ffn();
    model
        .metadata
        .insert("qwen35.feed_forward_length".into(), metadata!(16));
    for target in ["blk.0.ffn_up.weight", "blk.0.ffn_gate.weight"] {
        model.tensors.insert(
            target.into(),
            weight(target, &[2, 16], &[1.0, 0.0].repeat(16)),
        );
    }
    model.tensors.insert(
        "blk.0.ffn_down.weight".into(),
        weight("blk.0.ffn_down.weight", &[16, 2], &[0.0; 32]),
    );
    let mut rotated = active_ffn();
    rotated.metadata = model.metadata.clone();
    rotated.tensors = model.tensors.clone();
    for (name, value) in [
        ("prism.hadamard.version", metadata!(1)),
        ("prism.hadamard.block_size", metadata!(16)),
        (
            "prism.hadamard.transform",
            metadata!("normalized-sylvester-walsh-hadamard"),
        ),
        ("prism.hadamard.axis", metadata!("input-last-dimension")),
        ("prism.hadamard.sign_mode", metadata!("identity")),
        (
            "prism.hadamard.weight_names",
            metadata!(["blk.0.ffn_down.weight"]),
        ),
    ] {
        rotated.metadata.insert(name.into(), value);
    }
    let mut plain = FfnProbe::new(
        Qwen35::with_backend(model, 8, MatvecBackend::Scalar)?,
        0,
        None,
    )?;
    let mut transformed = FfnProbe::new(
        Qwen35::with_backend(rotated, 8, MatvecBackend::Scalar)?,
        0,
        None,
    )?;
    let mut hidden = Vec::new();
    plain.evaluate_observed_into(0, &mut Vec::new(), &mut |row| {
        hidden.extend_from_slice(row.activation);
        Ok(())
    })?;
    transformed.evaluate_observed_into(0, &mut Vec::new(), &mut |row| {
        assert_eq!(row.activation, hidden);
        assert!(row.activation.iter().all(|value| *value > 0.0));
        Ok(())
    })?;
    Ok(())
}
