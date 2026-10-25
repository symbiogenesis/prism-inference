//! Focused arbitrary broadcast gradients, classification metrics and dataset bounds.
#![forbid(unsafe_code)]

use bonsai_prism_runtime::optimization::{
    Dataset, Graph, Kind, Loss, Metrics, Operator, Optimizer, Parameters, Training,
};

type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

#[test]
fn broadcast_reverse_derivatives_accumulate_before_one_sgd_update() -> TestResult {
    let mut graph = Graph::new();
    let input = graph.tensor(0, &[2, 2], false, "input")?;
    let weight = graph.tensor(0, &[2], true, "weight")?;
    let bias = graph.tensor(0, &[1], true, "bias")?;
    graph.set_values(input, &[1.0, 2.0, 3.0, 4.0])?;
    graph.set_values(weight, &[1.0, 1.0])?;
    let product = graph.operation(1, Operator::Multiply, &[input, weight], "product")?;
    let output = graph.operation(1, Operator::Add, &[product, bias], "output")?;
    let mut optimizer = Optimizer::new(
        &mut graph,
        Training {
            context: 1,
            inputs: input,
            outputs: output,
            loss: Loss::MeanSquaredError,
            kind: Kind::Sgd,
            period: 2,
            parameters: Parameters {
                sgd: [0.1, 0.0],
                ..Parameters::default()
            },
        },
    )?;
    let mut metrics = Metrics::default();
    optimizer.allocate(&mut graph, true)?;
    optimizer.evaluate(&mut graph, Some(&mut metrics))?;
    assert_eq!(graph.values(weight)?, [1.0, 1.0]);
    assert_eq!(
        graph.values(optimizer.gradient_tensor(weight)?)?,
        [2.5, 5.0]
    );
    optimizer.allocate(&mut graph, true)?;
    optimizer.evaluate(&mut graph, Some(&mut metrics))?;
    assert_eq!(graph.values(weight)?, [0.5, 0.0]);
    assert_eq!(graph.values(bias)?, [-0.5]);
    assert_eq!(
        graph.values(optimizer.gradient_tensor(weight)?)?,
        [5.0, 10.0]
    );
    assert_eq!(metrics.count(), 4);
    assert!((metrics.loss().0 - 7.5).abs() < 1e-8);
    assert!(metrics.loss().1.abs() < 1e-8);
    assert!(metrics.accuracy().0.is_nan());
    optimizer.reset(&mut graph, false)?;
    assert_eq!(
        graph.values(optimizer.gradient_tensor(weight)?)?,
        [0.0, 0.0]
    );
    assert!(optimizer.evaluate(&mut graph, None).is_err());
    Ok(())
}

#[test]
fn cross_entropy_reports_real_row_accuracy_and_loss_uncertainty() -> TestResult {
    let mut graph = Graph::new();
    let input = graph.tensor(0, &[2, 2], false, "logits")?;
    graph.set_values(input, &[0.0, 0.0, 3.0_f32.ln(), 0.0])?;
    let mut optimizer = Optimizer::new(
        &mut graph,
        Training {
            context: 1,
            inputs: input,
            outputs: input,
            loss: Loss::CrossEntropy,
            kind: Kind::Sgd,
            period: 1,
            parameters: Parameters::default(),
        },
    )?;
    graph.set_values(
        optimizer.labels_tensor().ok_or("missing labels")?,
        &[1.0, 0.0, 1.0, 0.0],
    )?;
    let mut metrics = Metrics::default();
    optimizer.allocate(&mut graph, false)?;
    optimizer.evaluate(&mut graph, Some(&mut metrics))?;
    assert_eq!(metrics.count(), 2);
    assert!((metrics.loss().0 - 2.0_f64.ln().midpoint((4.0_f64 / 3.0).ln())).abs() < 1e-6);
    assert!(metrics.loss().1.is_nan());
    assert!((metrics.accuracy().0 - 1.0).abs() < f64::EPSILON);
    assert!(metrics.accuracy().1.abs() < f64::EPSILON);
    let mut combined = Metrics::default();
    combined.append(&metrics)?;
    combined.append(&metrics)?;
    assert_eq!(combined.count(), 4);
    assert!((combined.loss().0 - metrics.loss().0).abs() < f64::EPSILON);
    Ok(())
}

#[test]
fn classification_metrics_observe_forward_predictions_before_parameter_updates() -> TestResult {
    let mut graph = Graph::new();
    let output = graph.tensor(0, &[2, 1], true, "class logits")?;
    graph.set_values(output, &[1.0, 0.0])?;
    let mut optimizer = Optimizer::new(
        &mut graph,
        Training {
            context: 1,
            inputs: output,
            outputs: output,
            loss: Loss::CrossEntropy,
            kind: Kind::Sgd,
            period: 1,
            parameters: Parameters {
                sgd: [100.0, 0.0],
                ..Parameters::default()
            },
        },
    )?;
    graph.set_values(
        optimizer.labels_tensor().ok_or("missing labels")?,
        &[0.0, 1.0],
    )?;
    let mut metrics = Metrics::default();
    optimizer.allocate(&mut graph, true)?;
    optimizer.evaluate(&mut graph, Some(&mut metrics))?;
    let updated = graph.values(output)?;
    assert!(updated[1] > updated[0]);
    assert!(metrics.accuracy().0.abs() < f64::EPSILON);
    Ok(())
}

#[test]
fn sharded_batches_own_values_and_reject_overflow_and_incomplete_shards() -> TestResult {
    let dataset = Dataset::new(
        2,
        1,
        6,
        2,
        (0_u16..12).map(f32::from).collect(),
        (0_u16..6).map(f32::from).collect(),
    )?;
    let (mut data, labels) = dataset.batch(2, 2)?;
    assert_eq!(data, [8.0, 9.0, 10.0, 11.0]);
    assert_eq!(labels, [4.0, 5.0]);
    data[0] = 99.0;
    assert_eq!(dataset.batch(2, 2)?.0[0].to_bits(), 8.0_f32.to_bits());
    assert!(dataset.batch(3, 0).is_err());
    assert!(dataset.batch(2, usize::MAX).is_err());
    assert!(dataset.batch(usize::MAX, 0).is_err());
    let incomplete = Dataset::new(1, 0, 6, 4, vec![0.0; 6], vec![])?;
    assert_eq!(incomplete.batch(4, 0)?.0.len(), 4);
    assert!(incomplete.batch(4, 1).is_err());
    Ok(())
}
