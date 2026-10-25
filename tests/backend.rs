#![forbid(unsafe_code)]
//! Focused graph traversal correctness and bounded-stack regressions.

use bonsai_prism_runtime::Result;
use bonsai_prism_runtime::backend::{Argument, CpuBackend, GraphId, Observation, TensorId};

fn graph(backend: &mut CpuBackend, capacity: usize) -> Result<GraphId> {
    let Observation::Graph(graph) = backend.call(
        "ggml_new_graph_custom",
        &[
            Argument::Context(0),
            Argument::Integer(i64::try_from(capacity)?),
            Argument::Bool(false),
        ],
    )?
    else {
        return Err("graph construction returned a different type".into());
    };
    Ok(graph)
}

fn add(backend: &mut CpuBackend, a: TensorId, b: TensorId) -> Result<TensorId> {
    let Observation::Tensor(Some(tensor)) = backend.call(
        "ggml_add",
        &[
            Argument::Context(0),
            Argument::Tensor(a),
            Argument::Tensor(b),
        ],
    )?
    else {
        return Err("addition construction returned a different type".into());
    };
    Ok(tensor)
}

fn expand(backend: &mut CpuBackend, graph: GraphId, output: TensorId) -> Result<()> {
    backend.call(
        "ggml_build_forward_expand",
        &[Argument::Graph(graph), Argument::Tensor(output)],
    )?;
    Ok(())
}

#[test]
fn deep_graph_expands_and_computes_on_a_small_stack() -> Result<()> {
    // No timer or benchmark workload: scalar additions verify that graph depth
    // affects heap traversal storage rather than native stack consumption.
    let worker = std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(|| -> Result<()> {
            let mut backend = CpuBackend::new();
            let one = backend.tensor(0, 0, [1; 4])?;
            let zero = backend.tensor(0, 0, [1; 4])?;
            let mut output = one;
            for _ in 0..4096 {
                output = add(&mut backend, output, zero)?;
            }
            let graph = graph(&mut backend, 4096)?;
            expand(&mut backend, graph, output)?;
            let (nodes, leaves) = backend.graph_nodes(graph)?;
            assert_eq!(nodes.len(), 4096);
            assert_eq!(nodes.last(), Some(&output));
            assert_eq!(leaves, [one, zero]);
            backend.allocate(0)?;
            backend.set_values(one, &[1.0])?;
            backend.compute(graph)?;
            assert_eq!(backend.values(output)?, [1.0]);
            Ok(())
        })?;
    worker
        .join()
        .map_err(|_| "deep graph correctness worker panicked")??;
    Ok(())
}

#[test]
fn shared_sources_preserve_order_and_incremental_expansion() -> Result<()> {
    let mut backend = CpuBackend::new();
    let a = backend.tensor(0, 0, [1; 4])?;
    let b = backend.tensor(0, 0, [1; 4])?;
    let c = backend.tensor(0, 0, [1; 4])?;
    let left = add(&mut backend, a, b)?;
    let right = add(&mut backend, c, a)?;
    let root = add(&mut backend, left, right)?;
    let repeated = add(&mut backend, root, root)?;
    let graph = graph(&mut backend, 4)?;
    expand(&mut backend, graph, root)?;
    assert_eq!(
        backend.graph_nodes(graph)?,
        ([left, right, root].as_slice(), [a, b, c].as_slice())
    );
    expand(&mut backend, graph, repeated)?;
    expand(&mut backend, graph, root)?;
    assert_eq!(
        backend.graph_nodes(graph)?,
        (
            [left, right, root, repeated].as_slice(),
            [a, b, c].as_slice()
        )
    );
    let empty = self::graph(&mut backend, 1)?;
    backend.call(
        "ggml_graph_cpy",
        &[Argument::Graph(empty), Argument::Graph(graph)],
    )?;
    expand(&mut backend, graph, right)?;
    assert_eq!(
        backend.graph_nodes(graph)?,
        ([right].as_slice(), [c, a].as_slice())
    );
    Ok(())
}

#[test]
fn cycles_unknown_sources_and_node_capacity_remain_errors() -> Result<()> {
    let mut backend = CpuBackend::new();
    let input = backend.tensor(0, 0, [1; 4])?;
    let first = add(&mut backend, input, input)?;
    let second = add(&mut backend, first, input)?;
    let graph = graph(&mut backend, 1)?;
    let error = expand(&mut backend, graph, second)
        .err()
        .ok_or("excess nodes were admitted")?;
    assert_eq!(
        error.to_string(),
        "CPU graph exceeds requested node capacity"
    );
    assert_eq!(
        backend.graph_nodes(graph)?,
        ([first].as_slice(), [input].as_slice())
    );

    let cyclic = self::graph(&mut backend, 3)?;
    backend.info_mut(first)?.sources = vec![second];
    let error = expand(&mut backend, cyclic, second)
        .err()
        .ok_or("cycle was admitted")?;
    assert_eq!(error.to_string(), "cyclic or excessive CPU graph");
    backend.info_mut(first)?.sources = vec![TensorId(usize::MAX)];
    let error = expand(&mut backend, cyclic, first)
        .err()
        .ok_or("unknown source was admitted")?;
    assert_eq!(error.to_string(), "unknown CPU tensor identity");

    backend.info_mut(first)?.sources = vec![input, input];
    backend.call("ggml_set_param", &[Argument::Tensor(input)])?;
    expand(&mut backend, cyclic, second)?;
    assert_eq!(
        backend.graph_nodes(cyclic)?,
        ([input, first, second].as_slice(), [].as_slice())
    );
    Ok(())
}
