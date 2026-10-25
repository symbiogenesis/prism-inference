//! Non-timed graph switching and active-worker synchronization regressions.

use bonsai_prism_runtime::Result;
use bonsai_prism_runtime::backend::threadpool::CpuThreadPool;
use bonsai_prism_runtime::backend::{Argument, CpuBackend, GraphId, Observation, TensorId};

fn matrix_graph(
    backend: &mut CpuBackend,
    width: usize,
    rows: usize,
    packed: bool,
) -> Result<(GraphId, TensorId)> {
    let matrix = backend.tensor(0, if packed { 2 } else { 0 }, [width, rows, 1, 1])?;
    let input = backend.tensor(0, 0, [width, 1, 1, 1])?;
    backend.set_values(matrix, &vec![1.0; width * rows])?;
    backend.set_values(input, &vec![2.0; width])?;
    let Observation::Tensor(Some(output)) = backend.call(
        "ggml_mul_mat",
        &[
            Argument::Context(0),
            Argument::Tensor(matrix),
            Argument::Tensor(input),
        ],
    )?
    else {
        return Err(std::io::Error::other("expected matrix tensor").into());
    };
    let Observation::Graph(graph) = backend.call("ggml_new_graph", &[Argument::Context(0)])? else {
        return Err(std::io::Error::other("expected graph").into());
    };
    backend.call(
        "ggml_build_forward_expand",
        &[Argument::Graph(graph), Argument::Tensor(output)],
    )?;
    Ok((graph, output))
}

#[test]
fn plans_coexist_and_switch_active_workers_without_losing_rows() -> Result<()> {
    let mut backend = CpuBackend::new();
    let (small_graph, small_output) = matrix_graph(&mut backend, 3, 5, false)?;
    let (large_graph, large_output) = matrix_graph(&mut backend, 32, 9, true)?;
    let pool = CpuThreadPool::new(4)?;
    for threads in [1, 4, 2, 4, 1] {
        let small = pool.plan(&backend, small_graph, threads)?;
        let large = pool.plan(&backend, large_graph, threads)?;
        assert_eq!(small.threads(), threads);
        pool.compute(&mut backend, &small)?;
        assert_eq!(backend.values(small_output)?, vec![6.0; 5]);
        pool.compute(&mut backend, &large)?;
        // Native Q8_0 activation scales are rounded to F16 before the dot.
        assert_eq!(backend.values(large_output)?, vec![63.996_094; 9]);
    }
    assert!(pool.plan(&backend, small_graph, 0).is_err());
    assert!(pool.plan(&backend, small_graph, 5).is_err());
    Ok(())
}

#[test]
fn dependent_operations_observe_completed_parallel_outputs() -> Result<()> {
    let mut backend = CpuBackend::new();
    let (graph, first) = matrix_graph(&mut backend, 4, 3, false)?;
    let weights = backend.tensor(0, 0, [3, 2, 1, 1])?;
    backend.set_values(weights, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let Observation::Tensor(Some(output)) = backend.call(
        "ggml_mul_mat",
        &[
            Argument::Context(0),
            Argument::Tensor(weights),
            Argument::Tensor(first),
        ],
    )?
    else {
        return Err(std::io::Error::other("expected dependent tensor").into());
    };
    backend.call(
        "ggml_build_forward_expand",
        &[Argument::Graph(graph), Argument::Tensor(output)],
    )?;
    let pool = CpuThreadPool::new(4)?;
    let plan = pool.plan(&backend, graph, 4)?;
    pool.compute(&mut backend, &plan)?;
    assert_eq!(backend.values(output)?, [48.0, 120.0]);
    Ok(())
}

#[test]
fn failed_worker_group_drains_before_the_plan_is_reused() -> Result<()> {
    let mut backend = CpuBackend::new();
    let (graph, output) = matrix_graph(&mut backend, 3, 5, false)?;
    let input = TensorId(1);
    backend.set_values(input, &[f32::INFINITY, 2.0, 2.0])?;
    let pool = CpuThreadPool::new(4)?;
    let plan = pool.plan(&backend, graph, 4)?;
    assert!(pool.compute(&mut backend, &plan).is_err());
    backend.set_values(input, &[2.0, 2.0, 2.0])?;
    pool.compute(&mut backend, &plan)?;
    assert_eq!(backend.values(output)?, vec![6.0; 5]);
    Ok(())
}
