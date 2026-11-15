//! Independent scoped-OS threadpool and DAG barrier observations.
//!
//! This module implements bounded graph validation, iterative post-order node
//! counting, and persistent scoped OS worker threads with step barriers without
//! invoking `bonsai_prism_runtime`, any native bridge, or expected-answer fixtures.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Instant;

use serde::Deserialize;
use serde_json::{Value, json};

use super::{Result, invalid};

pub(super) const ID: &str = "tests/test-barrier.cpp";

const MAX_OPERATIONS: usize = 128;
const MIN_CONTEXT_BYTES: u64 = 1024;
const MAX_CONTEXT_BYTES: u64 = 1 << 30;
const MAX_POOL_THREADS: usize = 256;
const MAX_GRAPHS: usize = 16;
const MAX_ACTIONS: usize = 25_000;
const MAX_NODES_PER_GRAPH: usize = 8192;
const MAX_TOTAL_NODES: usize = 16_384;
const MAX_DIM: u64 = 1_048_576;
const MAX_SLOTS: usize = 16;
const MAX_WORKSPACE_BYTES: usize = 1 << 28;
const Q4_0_BLOCK: usize = 32;
const Q4_0_BLOCK_BYTES: usize = 18;
const NODE_OVERHEAD_BYTES: usize = 256;
const GRAPH_OVERHEAD_BYTES: usize = 512;

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
    context_bytes: u64,
    pool_threads: usize,
    graphs: Vec<GraphDesc>,
    actions: Vec<ActionDesc>,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
struct GraphDesc {
    nodes: Vec<NodeDesc>,
    output: usize,
}

#[derive(Deserialize)]
#[serde(tag = "Op", deny_unknown_fields)]
enum NodeDesc {
    #[serde(rename = "input")]
    Input {
        #[serde(rename = "Type")]
        kind: TensorKind,
        #[serde(rename = "Dimensions")]
        dimensions: Vec<u64>,
    },
    #[serde(rename = "mul_mat")]
    MulMat {
        #[serde(rename = "A")]
        a: usize,
        #[serde(rename = "B")]
        b: usize,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, Deserialize)]
enum TensorKind {
    F32,
    Q4_0,
}

impl TensorKind {
    const fn row_bytes(self, dim0: usize) -> Option<usize> {
        match self {
            Self::F32 => dim0.checked_mul(4),
            Self::Q4_0 => {
                if !dim0.is_multiple_of(Q4_0_BLOCK) {
                    return None;
                }
                (dim0 / Q4_0_BLOCK).checked_mul(Q4_0_BLOCK_BYTES)
            }
        }
    }
}

#[derive(Deserialize)]
#[serde(tag = "Op", deny_unknown_fields)]
enum ActionDesc {
    #[serde(rename = "plan")]
    Plan {
        #[serde(rename = "Slot")]
        slot: usize,
        #[serde(rename = "Graph")]
        graph: usize,
        #[serde(rename = "Threads")]
        threads: usize,
    },
    #[serde(rename = "compute")]
    Compute {
        #[serde(rename = "Slot")]
        slot: usize,
    },
    #[serde(rename = "timer-start")]
    TimerStart,
    #[serde(rename = "timer-stop")]
    TimerStop,
}

struct GraphInfo {
    node_count: usize,
}

struct StepBarrier {
    arrived: AtomicUsize,
    generation: AtomicUsize,
}

impl StepBarrier {
    const fn new() -> Self {
        Self {
            arrived: AtomicUsize::new(0),
            generation: AtomicUsize::new(0),
        }
    }

    fn wait(&self, participants: usize) {
        if participants <= 1 {
            return;
        }
        let current = self.generation.load(Ordering::SeqCst);
        if self.arrived.fetch_add(1, Ordering::SeqCst) + 1 == participants {
            self.arrived.store(0, Ordering::SeqCst);
            self.generation.store(current + 1, Ordering::SeqCst);
            return;
        }
        let mut spins = 0_u32;
        while self.generation.load(Ordering::SeqCst) == current {
            spins = spins.wrapping_add(1);
            if spins < 256 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }
}

struct CompiledPlan {
    threads: usize,
    node_count: usize,
    workspace: Vec<AtomicU64>,
    barrier: StepBarrier,
}

struct DispatchTask {
    epoch: u64,
    active_threads: usize,
    plan: Arc<CompiledPlan>,
}

struct PoolState {
    shutdown: bool,
    current: Option<DispatchTask>,
}

struct PoolShared {
    lock: Mutex<PoolState>,
    cvar: Condvar,
}

struct PoolGuard<'a> {
    shared: &'a PoolShared,
}

impl Drop for PoolGuard<'_> {
    fn drop(&mut self) {
        {
            let mut guard = lock_unpoisoned(&self.shared.lock);
            guard.shutdown = true;
        }
        self.shared.cvar.notify_all();
    }
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn wait_unpoisoned<'a, T>(cvar: &Condvar, guard: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
    match cvar.wait(guard) {
        Ok(next) => next,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn validate_node(
    node: &NodeDesc,
    index: usize,
    shapes: &mut Vec<[usize; 2]>,
    kinds: &mut Vec<TensorKind>,
    children: &mut Vec<Option<(usize, usize)>>,
) -> Result<()> {
    let (shape, kind, edge) = match node {
        NodeDesc::Input { kind, dimensions } => {
            if dimensions.is_empty()
                || dimensions.len() > 2
                || dimensions.iter().any(|&dim| !(1..=MAX_DIM).contains(&dim))
            {
                return Err(invalid(
                    "independent threadpool input dimensions outside bound",
                ));
            }
            let dim0 = usize::try_from(dimensions[0])?;
            let dim1 = if dimensions.len() == 2 {
                usize::try_from(dimensions[1])?
            } else {
                1
            };
            ([dim0, dim1], *kind, None)
        }
        NodeDesc::MulMat { a, b } => {
            if index == 0 || *a >= index || *b >= index {
                return Err(invalid(
                    "independent threadpool mul_mat index outside bound",
                ));
            }
            if shapes[*a][0] != shapes[*b][0] || kinds[*b] != TensorKind::F32 {
                return Err(invalid(
                    "independent threadpool matrix dimensions or right-hand type mismatch",
                ));
            }
            (
                [shapes[*a][1], shapes[*b][1]],
                TensorKind::F32,
                Some((*a, *b)),
            )
        }
    };
    shapes.push(shape);
    kinds.push(kind);
    children.push(edge);
    Ok(())
}

fn count_reachable_ops(children: &[Option<(usize, usize)>], output: usize) -> usize {
    let mut visited = vec![false; children.len()];
    let mut count = 0_usize;
    let mut stack = vec![output];
    while let Some(node) = stack.pop() {
        if visited[node] {
            continue;
        }
        visited[node] = true;
        if let Some((a, b)) = children[node] {
            count += 1;
            if !visited[b] {
                stack.push(b);
            }
            if !visited[a] {
                stack.push(a);
            }
        }
    }
    count
}

fn validate_graph(
    graph: &GraphDesc,
    context_bytes: usize,
    total_nodes: &mut usize,
    storage_bytes: &mut usize,
) -> Result<GraphInfo> {
    if graph.nodes.is_empty() || graph.nodes.len() > MAX_NODES_PER_GRAPH {
        return Err(invalid(
            "independent threadpool graph node count outside bound",
        ));
    }
    *total_nodes += graph.nodes.len();
    if *total_nodes > MAX_TOTAL_NODES {
        return Err(invalid(
            "independent threadpool total node count outside bound",
        ));
    }
    if graph.output >= graph.nodes.len() {
        return Err(invalid(
            "independent threadpool graph output index outside bound",
        ));
    }
    let mut shapes = Vec::with_capacity(graph.nodes.len());
    let mut kinds = Vec::with_capacity(graph.nodes.len());
    let mut children = Vec::with_capacity(graph.nodes.len());
    let limit = context_bytes / 2;
    for (index, node) in graph.nodes.iter().enumerate() {
        validate_node(node, index, &mut shapes, &mut kinds, &mut children)?;
        let shape = shapes[index];
        let row = kinds[index]
            .row_bytes(shape[0])
            .ok_or_else(|| invalid("independent threadpool quantized row is incomplete"))?;
        if row > context_bytes / shape[1] {
            return Err(invalid(
                "independent threadpool tensor exceeds context bound",
            ));
        }
        let bytes = row
            .checked_mul(shape[1])
            .ok_or_else(|| invalid("independent threadpool tensor storage overflow"))?;
        if bytes > limit || *storage_bytes > limit - bytes {
            return Err(invalid("independent threadpool data exceeds context bound"));
        }
        *storage_bytes += bytes;
    }
    Ok(GraphInfo {
        node_count: count_reachable_ops(&children, graph.output),
    })
}

fn validate_actions(actions: &[ActionDesc], graph_count: usize, pool_threads: usize) -> Result<()> {
    if actions.is_empty() || actions.len() > MAX_ACTIONS {
        return Err(invalid("independent threadpool action count outside bound"));
    }
    let mut planned = [false; MAX_SLOTS];
    let mut timer_running = false;
    for action in actions {
        match action {
            ActionDesc::Plan {
                slot,
                graph,
                threads,
            } => {
                if *slot >= MAX_SLOTS
                    || *graph >= graph_count
                    || !(1..=pool_threads).contains(threads)
                {
                    return Err(invalid(
                        "independent threadpool plan parameter outside bound",
                    ));
                }
                planned[*slot] = true;
            }
            ActionDesc::Compute { slot } => {
                if *slot >= MAX_SLOTS || !planned[*slot] {
                    return Err(invalid(
                        "independent threadpool plan used before construction",
                    ));
                }
            }
            ActionDesc::TimerStart => {
                if timer_running {
                    return Err(invalid("independent threadpool timer already running"));
                }
                timer_running = true;
            }
            ActionDesc::TimerStop => {
                if !timer_running {
                    return Err(invalid("independent threadpool timer not running"));
                }
                timer_running = false;
            }
        }
    }
    if timer_running {
        return Err(invalid("independent threadpool timer left running"));
    }
    Ok(())
}

fn validate_operation(operation: &Operation) -> Result<Vec<GraphInfo>> {
    if operation.kind != "cpu-threadpool"
        || !(MIN_CONTEXT_BYTES..=MAX_CONTEXT_BYTES).contains(&operation.context_bytes)
        || !(1..=MAX_POOL_THREADS).contains(&operation.pool_threads)
        || operation.graphs.is_empty()
        || operation.graphs.len() > MAX_GRAPHS
    {
        return Err(invalid(
            "independent threadpool configuration outside bound",
        ));
    }
    let context_bytes = usize::try_from(operation.context_bytes)?;
    let mut total_nodes = 0_usize;
    let mut storage_bytes = 0_usize;
    let graphs = operation
        .graphs
        .iter()
        .map(|graph| validate_graph(graph, context_bytes, &mut total_nodes, &mut storage_bytes))
        .collect::<Result<Vec<_>>>()?;
    let metadata = total_nodes
        .checked_mul(NODE_OVERHEAD_BYTES)
        .and_then(|nodes| {
            operation
                .graphs
                .len()
                .checked_mul(GRAPH_OVERHEAD_BYTES)
                .and_then(|headers| nodes.checked_add(headers))
        })
        .ok_or_else(|| invalid("independent threadpool metadata overflow"))?;
    if metadata > context_bytes / 4 {
        return Err(invalid(
            "independent threadpool metadata exceeds context bound",
        ));
    }
    validate_actions(&operation.actions, graphs.len(), operation.pool_threads)?;
    Ok(graphs)
}

fn run_plan(thread_id: usize, plan: &CompiledPlan) {
    let participants = plan.threads;
    for step in 0..plan.node_count {
        let step_tag = u64::try_from(step).map_or(0, |value| value + 1);
        let tid_tag = u64::try_from(thread_id).map_or(0, |value| value + 1);
        if let Some(slot) = plan.workspace.get(thread_id) {
            let peer = plan
                .workspace
                .get((thread_id + 1) % participants)
                .map_or(0, |entry| entry.load(Ordering::Relaxed));
            slot.store(step_tag ^ tid_tag ^ peer, Ordering::Relaxed);
        }
        plan.barrier.wait(participants);
    }
    // Final rendezvous guarantees all participating OS threads finish before
    // the coordinator moves to another slot, workspace replacement, or teardown.
    plan.barrier.wait(participants);
}

fn next_task(
    thread_id: usize,
    last_epoch: &mut u64,
    shared: &PoolShared,
) -> Option<(bool, Arc<CompiledPlan>)> {
    let mut guard = lock_unpoisoned(&shared.lock);
    loop {
        if guard.shutdown {
            return None;
        }
        if let Some(task) = &guard.current
            && task.epoch > *last_epoch
        {
            *last_epoch = task.epoch;
            return Some((thread_id < task.active_threads, Arc::clone(&task.plan)));
        }
        guard = wait_unpoisoned(&shared.cvar, guard);
    }
}

fn worker_loop(thread_id: usize, shared: &PoolShared) {
    let mut last_epoch = 0_u64;
    while let Some((participate, plan)) = next_task(thread_id, &mut last_epoch, shared) {
        if participate {
            run_plan(thread_id, &plan);
        }
    }
}

fn dispatch_plan(epoch: u64, plan: &Arc<CompiledPlan>, shared: &PoolShared) {
    {
        let mut guard = lock_unpoisoned(&shared.lock);
        guard.current = Some(DispatchTask {
            epoch,
            active_threads: plan.threads,
            plan: Arc::clone(plan),
        });
    }
    shared.cvar.notify_all();
}

fn execute_schedule(
    operation: &Operation,
    graphs: &[GraphInfo],
    shared: &PoolShared,
) -> Result<Value> {
    let mut plans: [Option<Arc<CompiledPlan>>; MAX_SLOTS] = std::array::from_fn(|_| None);
    let mut planned = 0_usize;
    let mut statuses = Vec::new();
    let mut nanoseconds = Vec::new();
    let mut timer: Option<Instant> = None;
    let mut epoch = 0_u64;
    for action in &operation.actions {
        match action {
            ActionDesc::Plan {
                slot,
                graph,
                threads,
            } => {
                let workspace_bytes = threads
                    .checked_mul(size_of::<AtomicU64>())
                    .ok_or_else(|| invalid("independent threadpool workspace overflow"))?;
                if workspace_bytes > MAX_WORKSPACE_BYTES {
                    return Err(invalid("independent threadpool workspace outside bound"));
                }
                let workspace = (0..*threads).map(|_| AtomicU64::new(0)).collect();
                plans[*slot] = Some(Arc::new(CompiledPlan {
                    threads: *threads,
                    node_count: graphs[*graph].node_count,
                    workspace,
                    barrier: StepBarrier::new(),
                }));
                planned += 1;
            }
            ActionDesc::Compute { slot } => {
                let plan = plans[*slot].as_ref().ok_or_else(|| {
                    invalid("independent threadpool plan used before construction")
                })?;
                if plan.threads > 1 {
                    epoch += 1;
                    dispatch_plan(epoch, plan, shared);
                }
                run_plan(0, plan);
                statuses.push(0_i32);
            }
            ActionDesc::TimerStart => {
                if timer.is_some() {
                    return Err(invalid("independent threadpool timer already running"));
                }
                timer = Some(Instant::now());
            }
            ActionDesc::TimerStop => {
                let started = timer
                    .take()
                    .ok_or_else(|| invalid("independent threadpool timer not running"))?;
                nanoseconds.push(u64::try_from(started.elapsed().as_nanos())?);
            }
        }
    }
    let node_counts = graphs
        .iter()
        .map(|graph| graph.node_count)
        .collect::<Vec<_>>();
    Ok(json!({
        "GraphNodeCounts": node_counts,
        "Plans": planned,
        "ComputeStatuses": statuses,
        "Nanoseconds": nanoseconds,
    }))
}

fn run_operation(operation: &Operation) -> Result<Value> {
    let graphs = validate_operation(operation)?;
    let worker_count = operation.pool_threads - 1;
    let shared = PoolShared {
        lock: Mutex::new(PoolState {
            shutdown: false,
            current: None,
        }),
        cvar: Condvar::new(),
    };
    std::thread::scope(|scope| {
        let guard = PoolGuard { shared: &shared };
        for thread_id in 1..=worker_count {
            let shared_ref = &shared;
            scope.spawn(move || worker_loop(thread_id, shared_ref));
        }
        let outcome = execute_schedule(operation, &graphs, &shared);
        drop(guard);
        outcome
    })
}

pub(super) fn handles(value: &Value) -> bool {
    value["Operations"].as_array().is_some_and(|operations| {
        !operations.is_empty()
            && operations
                .iter()
                .all(|operation| operation["Kind"] == "cpu-threadpool")
    })
}

pub(super) fn execute(value: &Value) -> Result<Value> {
    if !value["Operations"]
        .as_array()
        .is_some_and(|operations| (1..=MAX_OPERATIONS).contains(&operations.len()))
    {
        return Err(invalid(
            "independent threadpool requires 1 through 128 operations",
        ));
    }
    let request: Request = serde_json::from_value(value.clone())?;
    if request.schema_version != 1 {
        return Err(invalid("unsupported independent threadpool schema version"));
    }
    let results = request
        .operations
        .iter()
        .map(run_operation)
        .collect::<Result<Vec<_>>>()?;
    Ok(json!({"SchemaVersion": 1, "Results": results}))
}

pub(super) fn identity() -> Value {
    json!({"Kind": "independent-rust-threadpool", "SupportedSourceIds": [ID],
        "UpstreamCommit": super::UPSTREAM,
        "AssertionSource": {"Path": ID, "Sha256": "44a60801134d178c8e2c6789c471909ed83c0b8c71055b259478b4d396220cee"},
        "Architecture": "scoped-os-workers-with-epoch-condvar-and-step-barrier-v1",
        "NativeReferenceReplay": false, "ProductionRuntimeCalled": false,
        "CompleteSourceGate": false, "NativeRetirementQualified": false})
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_graph(width: usize, repetitions: usize) -> Value {
        let mut nodes = vec![json!({"Op": "input", "Type": "F32", "Dimensions": [width]})];
        let mut output = 0_usize;
        for _ in 0..repetitions {
            let a = nodes.len();
            nodes.push(json!({"Op": "input", "Type": "Q4_0", "Dimensions": [width, 128]}));
            output = nodes.len();
            nodes.push(json!({"Op": "mul_mat", "A": a, "B": output - 2}));
            let d = nodes.len();
            nodes.push(json!({"Op": "input", "Type": "Q4_0", "Dimensions": [128, width]}));
            let previous = output;
            output = nodes.len();
            nodes.push(json!({"Op": "mul_mat", "A": d, "B": previous}));
        }
        json!({"Nodes": nodes, "Output": output})
    }

    fn wrap(operation: Value) -> Value {
        let mut object = serde_json::Map::new();
        object.insert("SchemaVersion".into(), json!(1));
        object.insert("Operations".into(), Value::Array(vec![operation]));
        Value::Object(object)
    }

    #[test]
    fn independent_threadpool_preserves_untimed_coexisting_and_timed_schedules() -> Result<()> {
        let generic = wrap(json!({
            "Kind": "cpu-threadpool",
            "ContextBytes": 65_536,
            "PoolThreads": 3,
            "Graphs": [{
                "Nodes": [
                    {"Op": "input", "Type": "Q4_0", "Dimensions": [32, 5]},
                    {"Op": "input", "Type": "F32", "Dimensions": [32]},
                    {"Op": "mul_mat", "A": 0, "B": 1}
                ],
                "Output": 2
            }],
            "Actions": [
                {"Op": "plan", "Slot": 0, "Graph": 0, "Threads": 1},
                {"Op": "plan", "Slot": 1, "Graph": 0, "Threads": 3},
                {"Op": "compute", "Slot": 1},
                {"Op": "compute", "Slot": 0}
            ]
        }));
        assert!(handles(&generic));
        assert_eq!(
            execute(&generic)?["Results"][0],
            json!({"GraphNodeCounts": [1], "Plans": 2, "ComputeStatuses": [0, 0], "Nanoseconds": []})
        );

        let constant = wrap(json!({
            "Kind": "cpu-threadpool",
            "ContextBytes": 1_u64 << 30,
            "PoolThreads": 4,
            "Graphs": [test_graph(64, 1000)],
            "Actions": [
                {"Op": "plan", "Slot": 0, "Graph": 0, "Threads": 4},
                {"Op": "compute", "Slot": 0},
                {"Op": "timer-start"},
                {"Op": "compute", "Slot": 0},
                {"Op": "compute", "Slot": 0},
                {"Op": "timer-stop"}
            ]
        }));
        let constant_result = &execute(&constant)?["Results"][0];
        assert_eq!(constant_result["GraphNodeCounts"], json!([2000]));
        assert_eq!(constant_result["Plans"], 1);
        assert_eq!(constant_result["ComputeStatuses"], json!([0, 0, 0]));
        assert_eq!(
            constant_result["Nanoseconds"].as_array().map(Vec::len),
            Some(1)
        );

        let multi = wrap(json!({
            "Kind": "cpu-threadpool",
            "ContextBytes": 1_u64 << 30,
            "PoolThreads": 4,
            "Graphs": [test_graph(64, 2), test_graph(256, 4)],
            "Actions": [
                {"Op": "plan", "Slot": 0, "Graph": 0, "Threads": 1},
                {"Op": "plan", "Slot": 1, "Graph": 1, "Threads": 1},
                {"Op": "compute", "Slot": 0},
                {"Op": "compute", "Slot": 1},
                {"Op": "plan", "Slot": 0, "Graph": 0, "Threads": 4},
                {"Op": "plan", "Slot": 1, "Graph": 1, "Threads": 4},
                {"Op": "compute", "Slot": 0},
                {"Op": "compute", "Slot": 1},
                {"Op": "plan", "Slot": 0, "Graph": 0, "Threads": 1},
                {"Op": "compute", "Slot": 0}
            ]
        }));
        assert_eq!(
            execute(&multi)?["Results"][0],
            json!({"GraphNodeCounts": [4, 8], "Plans": 5, "ComputeStatuses": [0, 0, 0, 0, 0], "Nanoseconds": []})
        );
        Ok(())
    }

    #[test]
    fn independent_threadpool_counts_dag_and_leaf_outputs_correctly() -> Result<()> {
        let single_thread = wrap(json!({
            "Kind": "cpu-threadpool",
            "ContextBytes": 65_536,
            "PoolThreads": 1,
            "Graphs": [
                {
                    "Nodes": [
                        {"Op": "input", "Type": "F32", "Dimensions": [4]},
                        {"Op": "input", "Type": "F32", "Dimensions": [4]},
                        {"Op": "mul_mat", "A": 0, "B": 1},
                        {"Op": "mul_mat", "A": 2, "B": 2},
                        {"Op": "mul_mat", "A": 0, "B": 1}
                    ],
                    "Output": 3
                },
                {
                    "Nodes": [{"Op": "input", "Type": "F32", "Dimensions": [4]}],
                    "Output": 0
                }
            ],
            "Actions": [
                {"Op": "plan", "Slot": 0, "Graph": 0, "Threads": 1},
                {"Op": "plan", "Slot": 1, "Graph": 1, "Threads": 1},
                {"Op": "compute", "Slot": 0},
                {"Op": "compute", "Slot": 1}
            ]
        }));
        assert_eq!(
            execute(&single_thread)?["Results"][0],
            json!({"GraphNodeCounts": [2, 0], "Plans": 2, "ComputeStatuses": [0, 0], "Nanoseconds": []})
        );

        let info = identity();
        assert_eq!(info["SupportedSourceIds"], json!([ID]));
        assert_eq!(info["NativeReferenceReplay"], false);
        assert_eq!(info["ProductionRuntimeCalled"], false);
        Ok(())
    }

    #[test]
    fn independent_threadpool_rejects_invalid_bounds_types_and_schedules() {
        assert!(!handles(&wrap(json!({"Kind": "cpu-buffer-info"}))));
        assert!(!handles(&wrap(json!({"Kind": "cpu-arena-graph"}))));
        assert!(!handles(&json!({"SchemaVersion": 1, "Operations": []})));

        let valid_graph = json!({
            "Nodes": [
                {"Op": "input", "Type": "Q4_0", "Dimensions": [32]},
                {"Op": "input", "Type": "F32", "Dimensions": [32]},
                {"Op": "mul_mat", "A": 0, "B": 1}
            ],
            "Output": 2
        });
        for operation in [
            json!({"Kind": "cpu-threadpool", "ContextBytes": 1023, "PoolThreads": 1, "Graphs": [valid_graph], "Actions": [{"Op": "plan", "Slot": 0, "Graph": 0, "Threads": 1}]}),
            json!({"Kind": "cpu-threadpool", "ContextBytes": 1024, "PoolThreads": 1, "Graphs": [valid_graph], "Actions": [{"Op": "plan", "Slot": 0, "Graph": 0, "Threads": 1}]}),
            json!({"Kind": "cpu-threadpool", "ContextBytes": 65_536, "PoolThreads": 0, "Graphs": [valid_graph], "Actions": [{"Op": "plan", "Slot": 0, "Graph": 0, "Threads": 1}]}),
            json!({"Kind": "cpu-threadpool", "ContextBytes": 65_536, "PoolThreads": 257, "Graphs": [valid_graph], "Actions": [{"Op": "plan", "Slot": 0, "Graph": 0, "Threads": 1}]}),
            json!({"Kind": "cpu-threadpool", "ContextBytes": 65_536, "PoolThreads": 2, "Graphs": [{"Nodes": [{"Op": "input", "Type": "Q4_0", "Dimensions": [33]}], "Output": 0}], "Actions": [{"Op": "plan", "Slot": 0, "Graph": 0, "Threads": 1}]}),
            json!({"Kind": "cpu-threadpool", "ContextBytes": 65_536, "PoolThreads": 2, "Graphs": [{"Nodes": [{"Op": "input", "Type": "F32", "Dimensions": [32]}, {"Op": "input", "Type": "Q4_0", "Dimensions": [32]}, {"Op": "mul_mat", "A": 0, "B": 1}], "Output": 2}], "Actions": [{"Op": "plan", "Slot": 0, "Graph": 0, "Threads": 1}]}),
            json!({"Kind": "cpu-threadpool", "ContextBytes": 65_536, "PoolThreads": 2, "Graphs": [{"Nodes": [{"Op": "input", "Type": "F32", "Dimensions": [32]}, {"Op": "input", "Type": "F32", "Dimensions": [64]}, {"Op": "mul_mat", "A": 0, "B": 1}], "Output": 2}], "Actions": [{"Op": "plan", "Slot": 0, "Graph": 0, "Threads": 1}]}),
            json!({"Kind": "cpu-threadpool", "ContextBytes": 65_536, "PoolThreads": 2, "Graphs": [{"Nodes": [{"Op": "mul_mat", "A": 0, "B": 0}], "Output": 0}], "Actions": [{"Op": "plan", "Slot": 0, "Graph": 0, "Threads": 1}]}),
            json!({"Kind": "cpu-threadpool", "ContextBytes": 65_536, "PoolThreads": 2, "Graphs": [valid_graph], "Actions": [{"Op": "compute", "Slot": 0}]}),
            json!({"Kind": "cpu-threadpool", "ContextBytes": 65_536, "PoolThreads": 2, "Graphs": [valid_graph], "Actions": [{"Op": "plan", "Slot": 0, "Graph": 0, "Threads": 3}]}),
            json!({"Kind": "cpu-threadpool", "ContextBytes": 65_536, "PoolThreads": 2, "Graphs": [valid_graph], "Actions": [{"Op": "plan", "Slot": 16, "Graph": 0, "Threads": 1}]}),
            json!({"Kind": "cpu-threadpool", "ContextBytes": 65_536, "PoolThreads": 2, "Graphs": [valid_graph], "Actions": [{"Op": "timer-start"}]}),
            json!({"Kind": "cpu-threadpool", "ContextBytes": 65_536, "PoolThreads": 2, "Graphs": [valid_graph], "Actions": [{"Op": "timer-stop"}]}),
            json!({"Kind": "cpu-threadpool", "ContextBytes": 65_536, "PoolThreads": 2, "Graphs": [valid_graph], "Actions": [{"Op": "timer-start"}, {"Op": "timer-start"}, {"Op": "timer-stop"}]}),
            json!({"Kind": "cpu-threadpool", "ContextBytes": 65_536, "PoolThreads": 2, "Graphs": [valid_graph], "Actions": [{"Op": "plan", "Slot": 0, "Graph": 0, "Threads": 1}], "Extra": 1}),
        ] {
            assert!(execute(&wrap(operation)).is_err());
        }
        assert!(execute(&json!({"SchemaVersion": 2, "Operations": []})).is_err());
    }
}
