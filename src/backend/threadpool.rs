//! Persistent CPU workers and independent graph plans with operation barriers.
//!
//! Matrix rows use the production packed kernels. Every operation completes on
//! all active workers before its outputs become inputs to the following node.
//! Plans own distinct, reusable kernel scratch even when sharing the same pool.

use super::{CpuBackend, GraphId, TensorId, linear, type_layout};
use crate::{Result, invalid};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};

type Job = Box<dyn FnOnce() + Send + 'static>;

/// A persistent set of CPU workers shared by independently owned graph plans.
pub struct CpuThreadPool {
    senders: Vec<mpsc::Sender<Job>>,
    workers: Vec<JoinHandle<()>>,
}

/// One graph's immutable schedule and independent per-worker kernel scratch.
pub struct CpuPlan {
    graph: GraphId,
    nodes: Vec<TensorId>,
    scratch: Arc<Vec<Mutex<crate::kernels::MatvecScratch>>>,
}

impl CpuPlan {
    /// Number of active workers selected when constructing this plan.
    #[must_use]
    pub fn threads(&self) -> usize {
        self.scratch.len()
    }
}

impl CpuThreadPool {
    /// Start a bounded set of workers, retained until the pool is dropped.
    ///
    /// # Errors
    /// Rejects zero or more than 256 workers, or operating-system spawn failures.
    pub fn new(threads: usize) -> Result<Self> {
        if !(1..=256).contains(&threads) {
            return Err(invalid("CPU pool thread count outside bound"));
        }
        let mut pool = Self {
            senders: Vec::new(),
            workers: Vec::new(),
        };
        for index in 0..threads {
            let (sender, receiver) = mpsc::channel::<Job>();
            let worker = thread::Builder::new()
                .name(format!("prism-cpu-{index}"))
                .spawn(move || {
                    while let Ok(job) = receiver.recv() {
                        job();
                    }
                })?;
            pool.senders.push(sender);
            pool.workers.push(worker);
        }
        Ok(pool)
    }

    /// Construct a graph plan with its own reusable workspace.
    /// Other plans may coexist and use different active worker counts.
    ///
    /// # Errors
    /// Rejects unknown graphs, unsupported kernels, or more workers than the pool.
    pub fn plan(&self, backend: &CpuBackend, graph: GraphId, threads: usize) -> Result<CpuPlan> {
        if threads == 0 || threads > self.senders.len() {
            return Err(invalid("CPU plan exceeds available workers"));
        }
        let nodes = backend.graph_nodes(graph)?.0.to_vec();
        for &node in &nodes {
            if !backend.supports(node)? {
                return Err(invalid("CPU plan contains unsupported operation"));
            }
        }
        Ok(CpuPlan {
            graph,
            nodes,
            scratch: Arc::new(
                (0..threads)
                    .map(|_| Mutex::new(crate::kernels::MatvecScratch::default()))
                    .collect(),
            ),
        })
    }

    fn map<T: Send + 'static>(
        &self,
        threads: usize,
        operation: impl Fn(usize) -> Result<T> + Send + Sync + 'static,
    ) -> Result<Vec<T>> {
        let operation = Arc::new(operation);
        let (sender, receiver) = mpsc::channel();
        for (index, worker) in self.senders.iter().enumerate().take(threads) {
            let operation = Arc::clone(&operation);
            let sender = sender.clone();
            worker
                .send(Box::new(move || {
                    let result =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| operation(index)))
                            .unwrap_or_else(|_| Err(invalid("CPU worker panicked")));
                    let _ = sender.send((index, result));
                }))
                .map_err(|_| invalid("CPU worker terminated"))?;
        }
        drop(sender);
        let mut output: Vec<Option<T>> = (0..threads).map(|_| None).collect();
        let mut failure = None;
        // Always drain every active worker before returning, including failures.
        for _ in 0..threads {
            let (index, result) = receiver
                .recv()
                .map_err(|_| invalid("CPU barrier lost a worker result"))?;
            match result {
                Ok(value) => output[index] = Some(value),
                Err(error) => {
                    if failure.is_none() {
                        failure = Some(error);
                    }
                }
            }
        }
        if let Some(error) = failure {
            return Err(error);
        }
        output
            .into_iter()
            .map(|value| value.ok_or_else(|| invalid("CPU barrier has incomplete output")))
            .collect()
    }

    /// Execute a plan, completing all active workers between dependent operations.
    /// Matrix rows run concurrently; other reference kernels run on the caller.
    ///
    /// # Errors
    /// Rejects graph mutation after planning, invalid numerical input, kernel
    /// failures, poisoned workspace, worker termination, or allocation failure.
    pub fn compute(&self, backend: &mut CpuBackend, plan: &CpuPlan) -> Result<()> {
        if backend.graph_nodes(plan.graph)?.0 != plan.nodes {
            return Err(invalid("CPU graph changed after planning"));
        }
        for &node in &plan.nodes {
            if !backend.supports(node)? {
                return Err(invalid("CPU planned operation is no longer supported"));
            }
            if backend.info(node)?.operation == 29 {
                self.matmul(backend, node, plan)?;
            } else {
                backend.evaluate(node)?;
            }
        }
        Ok(())
    }

    fn matmul(&self, backend: &mut CpuBackend, node: TensorId, plan: &CpuPlan) -> Result<()> {
        let info = backend.info(node)?.clone();
        let left = *info
            .sources
            .first()
            .ok_or_else(|| invalid("CPU matrix source missing"))?;
        let right = *info
            .sources
            .get(1)
            .ok_or_else(|| invalid("CPU matrix activation missing"))?;
        let left_info = backend.info(left)?.clone();
        let right_shape = backend.info(right)?.shape;
        let input = Arc::new(backend.values(right)?);
        let shape = info.shape;
        if left_info.shape[0] != right_shape[0]
            || !right_shape[2].is_multiple_of(left_info.shape[2])
            || !right_shape[3].is_multiple_of(left_info.shape[3])
        {
            return Err(invalid("CPU pool matrix shapes differ"));
        }
        let row_bytes = left_info.shape[0] / type_layout(left_info.kind)?.block
            * type_layout(left_info.kind)?.bytes;
        let mut output = vec![0.0; info.elements()];
        let threads = plan.threads();
        for batch3 in 0..shape[3] {
            for batch2 in 0..shape[2] {
                let matrix = backend.matrix(
                    left,
                    batch2 / (right_shape[2] / left_info.shape[2]),
                    batch3 / (right_shape[3] / left_info.shape[3]),
                )?;
                let scratch = Arc::clone(&plan.scratch);
                let input = Arc::clone(&input);
                let rows = self.map(threads, move |worker| {
                    let first = worker * shape[0] / threads;
                    let end = (worker + 1) * shape[0] / threads;
                    if first == end {
                        return Ok((first, Vec::new()));
                    }
                    let matrix = crate::model::Tensor {
                        name: matrix.name.clone(),
                        dimensions: vec![right_shape[0], end - first],
                        kind: matrix.kind,
                        data: Arc::from(&matrix.data[first * row_bytes..end * row_bytes]),
                    };
                    let mut workspace = scratch[worker]
                        .lock()
                        .map_err(|_| invalid("CPU plan workspace is poisoned"))?;
                    let mut results = Vec::new();
                    results.try_reserve_exact((end - first) * shape[1])?;
                    let mut column_output = Vec::new();
                    for column in 0..shape[1] {
                        let offset = linear([0, column, batch2, batch3], right_shape);
                        crate::kernels::matvec_into(
                            &matrix,
                            &input[offset..offset + right_shape[0]],
                            &mut column_output,
                            &mut workspace,
                        )?;
                        results.extend_from_slice(&column_output);
                    }
                    Ok((first, results))
                })?;
                for (first, values) in rows {
                    let count = values.len().checked_div(shape[1]).unwrap_or(0);
                    if count == 0 {
                        continue;
                    }
                    for (column, values) in values.chunks_exact(count).enumerate() {
                        let offset = linear([first, column, batch2, batch3], shape);
                        output[offset..offset + count].copy_from_slice(values);
                    }
                }
            }
        }
        backend.set_values(node, &output)
    }
}

impl Drop for CpuThreadPool {
    fn drop(&mut self) {
        self.senders.clear();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}
