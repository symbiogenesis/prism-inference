//! Timed evaluation must preserve numerical ordering and transactional state.

use super::{
    Decoder, MatvecBackend, Qwen35, Result, fixture, graph_bits, rank_one_factors, rank_one_graph,
    weight,
};
use bonsai_gguf::MetadataValue;
use bonsai_prism_runtime::graph::TokenTimings;

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn layer_timing_preserves_adapter_logits_complete_state_and_checkpoint_replay() -> Result<()> {
    for recurrent in [false, true] {
        let graph = || {
            let mut model = fixture(recurrent);
            for target in [
                "blk.0.ffn_up.weight",
                "blk.0.ffn_gate.weight",
                "blk.0.ffn_down.weight",
            ] {
                model.tensors.insert(
                    target.into(),
                    weight(target, &[2, 2], &[0.25, 0.0, 0.0, 0.5]),
                );
            }
            let factors = rank_one_factors(&model, &["blk.0.ffn_down.weight"], 0.125);
            rank_one_graph(model, factors, 1.3)
        };
        let mut plain = graph()?;
        let mut timed = graph()?;
        let mut timings = TokenTimings::default();
        for token in [0, 1, 0] {
            let mut actual = Vec::new();
            timed.evaluate_with_timings_into(token, &mut actual, &mut timings)?;
            assert_eq!(bits(&actual), bits(&plain.evaluate(token)?));
            assert_eq!(graph_bits(&timed)?, graph_bits(&plain)?);
            assert_eq!(timed.identity(), plain.identity());
            assert!(timings.entered && timings.succeeded);
            assert_eq!(timings.layers().len(), 1);
            let layer = timings.layers()[0];
            assert!(layer.attention.attempted && layer.attention.completed);
            assert!(layer.ffn.attempted && layer.ffn.completed);
            assert!(layer.attention.elapsed + layer.ffn.elapsed <= timings.elapsed);
        }
        let checkpoint = timed.checkpoint()?;
        let expected = timed.evaluate(1)?;
        timed.restore(&checkpoint)?;
        let mut actual = Vec::new();
        timed.evaluate_with_timings_into(1, &mut actual, &mut timings)?;
        assert_eq!(bits(&actual), bits(&expected));
    }
    Ok(())
}

#[test]
fn layer_timing_distinguishes_completed_brackets_from_failed_token_commit() -> Result<()> {
    let mut model = fixture(true);
    model.tensors.insert(
        "output.weight".into(),
        weight("output.weight", &[2, 2], &[f32::MAX; 4]),
    );
    let mut graph = Qwen35::with_backend(model, 8, MatvecBackend::Scalar)?;
    let before = graph_bits(&graph)?;
    let mut timings = TokenTimings::default();
    let mut logits = vec![123.0];
    assert!(
        graph
            .evaluate_with_timings_into(0, &mut logits, &mut timings)
            .is_err()
    );
    assert!(timings.entered && !timings.succeeded);
    assert!(timings.layers()[0].attention.completed);
    assert!(timings.layers()[0].ffn.completed);
    assert_eq!(logits, Vec::<f32>::new());
    assert_eq!(before, graph_bits(&graph)?);

    // A rejected next call must replace the previous successful layer brackets.
    assert!(
        graph
            .evaluate_with_timings_into(u32::MAX, &mut logits, &mut timings)
            .is_err()
    );
    assert!(!timings.succeeded);
    assert!(!timings.layers()[0].attention.attempted);
    assert!(!timings.layers()[0].ffn.attempted);
    Ok(())
}

#[test]
fn layer_timing_retains_failed_ffn_attempt_and_bounds_fixed_storage() -> Result<()> {
    let mut model = fixture(false);
    for target in ["blk.0.ffn_up.weight", "blk.0.ffn_gate.weight"] {
        model
            .tensors
            .insert(target.into(), weight(target, &[2, 2], &[f32::MAX; 4]));
    }
    let mut graph = Qwen35::with_backend(model, 8, MatvecBackend::Scalar)?;
    let before = graph_bits(&graph)?;
    let mut timings = TokenTimings::default();
    let mut logits = Vec::new();
    assert!(
        graph
            .evaluate_with_timings_into(0, &mut logits, &mut timings)
            .is_err()
    );
    assert!(timings.layers()[0].attention.completed);
    assert!(timings.layers()[0].ffn.attempted);
    assert!(!timings.layers()[0].ffn.completed);
    assert!(!timings.succeeded);
    assert_eq!(before, graph_bits(&graph)?);

    let mut model = fixture(false);
    let layer: Vec<_> = model
        .tensors
        .iter()
        .filter(|(name, _)| name.starts_with("blk.0."))
        .map(|(name, tensor)| (name.clone(), tensor.clone()))
        .collect();
    for index in 1..65 {
        for (name, tensor) in &layer {
            let name = name.replacen("blk.0.", &format!("blk.{index}."), 1);
            let mut tensor = tensor.clone();
            tensor.name.clone_from(&name);
            model.tensors.insert(name, tensor);
        }
    }
    model
        .metadata
        .insert("qwen35.block_count".into(), MetadataValue::from(65_u32));
    model.metadata.insert(
        "qwen35.attention.recurrent_layers".into(),
        MetadataValue::from(vec![false; 65]),
    );
    let mut graph = Qwen35::with_backend(model, 8, MatvecBackend::Scalar)?;
    logits.push(1.0);
    assert!(
        graph
            .evaluate_with_timings_into(0, &mut logits, &mut timings)
            .is_err()
    );
    assert!(!timings.entered && !timings.succeeded);
    assert!(timings.layers().is_empty() && logits.is_empty());
    assert_eq!(graph_bits(&graph)?, Vec::<u32>::new());
    Ok(())
}
