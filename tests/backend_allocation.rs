//! Focused generic allocation lifetime and numerical aliasing regressions.

use bonsai_prism_runtime::Result;
use bonsai_prism_runtime::backend::allocator::{BufferType, GraphAllocator};
use bonsai_prism_runtime::backend::{Argument, CpuBackend, GraphId, Observation, TensorId};

fn tensor(value: &Observation) -> Result<TensorId> {
    match value {
        Observation::Tensor(Some(id)) => Ok(*id),
        _ => Err(std::io::Error::other("expected tensor").into()),
    }
}

fn graph(backend: &mut CpuBackend, output: TensorId) -> Result<GraphId> {
    let Observation::Graph(graph) = backend.call("ggml_new_graph", &[Argument::Context(0)])? else {
        return Err(std::io::Error::other("expected graph").into());
    };
    backend.info_mut(output)?.flags |= 2;
    backend.call(
        "ggml_build_forward_expand",
        &[Argument::Graph(graph), Argument::Tensor(output)],
    )?;
    Ok(graph)
}

#[test]
fn graph_arena_keeps_live_inputs_and_view_metadata() -> Result<()> {
    let mut backend = CpuBackend::new();
    let input = backend.tensor(0, 0, [4, 1, 1, 1])?;
    backend.info_mut(input)?.flags |= 1;
    let scaled = tensor(&backend.call(
        "ggml_scale",
        &[
            Argument::Context(0),
            Argument::Tensor(input),
            Argument::Float(2.0),
        ],
    )?)?;
    let output = tensor(&backend.call(
        "ggml_add",
        &[
            Argument::Context(0),
            Argument::Tensor(scaled),
            Argument::Tensor(input),
        ],
    )?)?;
    let graph = graph(&mut backend, output)?;
    let mut allocator = GraphAllocator::new(vec![BufferType {
        alignment: 8,
        maximum: Some(16),
    }])?;
    let allocation = allocator.reserve(&backend, graph, &[], &[])?;
    assert_ne!(allocation.tensors[input.0], allocation.tensors[scaled.0]);
    assert_eq!(allocation.tensors[output.0], allocation.tensors[scaled.0]);
    assert_eq!(
        allocation.buffers.iter().map(|b| b.bytes).sum::<usize>(),
        32
    );
    GraphAllocator::allocate(&mut backend, &allocation)?;
    assert!(backend.info(input)?.view.is_none());
    assert!(backend.info(output)?.view.is_none());
    backend.set_values(input, &[1.0, 2.0, 3.0, 4.0])?;
    backend.compute(graph)?;
    assert_eq!(backend.values(output)?, [3.0, 6.0, 9.0, 12.0]);
    Ok(())
}

#[test]
fn external_storage_and_output_flags_prevent_reuse() -> Result<()> {
    let mut backend = CpuBackend::new();
    let input = backend.tensor(7, 0, [3, 1, 1, 1])?;
    backend.info_mut(input)?.flags |= 1;
    let output = tensor(&backend.call(
        "ggml_scale",
        &[
            Argument::Context(0),
            Argument::Tensor(input),
            Argument::Float(3.0),
        ],
    )?)?;
    let graph = graph(&mut backend, output)?;
    let mut allocator = GraphAllocator::new(vec![BufferType {
        alignment: 4,
        maximum: None,
    }])?;
    let allocation = allocator.reserve(&backend, graph, &[], &[(7, 0)])?;
    assert_ne!(allocation.tensors[input.0], allocation.tensors[output.0]);
    GraphAllocator::allocate(&mut backend, &allocation)?;
    backend.set_values(input, &[2.0, 3.0, 5.0])?;
    backend.compute(graph)?;
    assert_eq!(backend.values(input)?, [2.0, 3.0, 5.0]);
    assert_eq!(backend.values(output)?, [6.0, 9.0, 15.0]);
    Ok(())
}
