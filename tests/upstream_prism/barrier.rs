//! Rust-owned CPU barrier workload; the source has no numeric output assertions.
//! Every execution is CI-only because the original includes timed graph work.
use super::engine::Engine;
use super::{Result, invalid, result};
use serde_json::{Value, json};
use std::env;
use std::time::Instant;

const ID: &str = "tests/test-barrier.cpp";

fn number(value: &Value, low: usize, high: usize) -> Result<usize> {
    let number = usize::try_from(
        value
            .as_u64()
            .ok_or_else(|| invalid("threadpool unsigned integer required"))?,
    )?;
    if !(low..=high).contains(&number) {
        return Err(invalid("threadpool integer outside bound"));
    }
    Ok(number)
}

fn array(value: &Value) -> Result<&[Value]> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| invalid("threadpool array required"))
}

/// Marshal graph, plan and compute actions to the production persistent worker pool.
#[allow(
    clippy::too_many_lines,
    reason = "Only graph construction and supplied action serialization occur in this adapter."
)]
pub(super) fn execute(operation: &Value) -> Result<Value> {
    use bonsai_prism_runtime::backend::threadpool::{CpuPlan, CpuThreadPool};
    use bonsai_prism_runtime::backend::{Argument, CpuBackend, Observation, TensorId};
    let memory = number(&operation["ContextBytes"], 1024, 1 << 30)?;
    let threads = number(&operation["PoolThreads"], 1, 256)?;
    let descriptions = array(&operation["Graphs"])?;
    let actions = array(&operation["Actions"])?;
    if descriptions.is_empty()
        || descriptions.len() > 16
        || actions.is_empty()
        || actions.len() > 25_000
    {
        return Err(invalid("threadpool graph/action count outside bound"));
    }
    if actions
        .iter()
        .any(|action| matches!(action["Op"].as_str(), Some("timer-start" | "timer-stop")))
        && (env::var("GITHUB_ACTIONS").as_deref() != Ok("true")
            || env::var("GITHUB_REF").as_deref() != Ok("refs/heads/main"))
    {
        return Err(invalid(
            "timed CPU threadpool work runs only in GitHub Actions on main",
        ));
    }
    let mut backend = CpuBackend::new();
    let mut graphs = Vec::new();
    let mut counts = Vec::new();
    let mut bytes = 0_usize;
    let mut total_nodes = 0_usize;
    for description in descriptions {
        let nodes = array(&description["Nodes"])?;
        if nodes.is_empty() || nodes.len() > 8192 {
            return Err(invalid("threadpool graph node count outside bound"));
        }
        total_nodes += nodes.len();
        if total_nodes > 16384 {
            return Err(invalid("threadpool total graph size outside bound"));
        }
        let mut tensors: Vec<TensorId> = Vec::new();
        for node in nodes {
            let tensor = match node["Op"].as_str() {
                Some("input") => {
                    let kind = match node["Type"].as_str() {
                        Some("F32") => 0,
                        Some("Q4_0") => 2,
                        _ => return Err(invalid("threadpool input type unsupported")),
                    };
                    let dimensions = array(&node["Dimensions"])?;
                    if dimensions.is_empty() || dimensions.len() > 2 {
                        return Err(invalid("threadpool input rank differs"));
                    }
                    let mut shape = [1; 4];
                    for (dimension, value) in shape.iter_mut().zip(dimensions) {
                        *dimension = number(value, 1, 1_048_576)?;
                    }
                    backend.tensor(0, kind, shape)?
                }
                Some("mul_mat") => {
                    if tensors.is_empty() {
                        return Err(invalid("threadpool operation has no preceding input"));
                    }
                    let left = tensors[number(&node["A"], 0, tensors.len() - 1)?];
                    let right = tensors[number(&node["B"], 0, tensors.len() - 1)?];
                    let Observation::Tensor(Some(id)) = backend.call(
                        "ggml_mul_mat",
                        &[
                            Argument::Context(0),
                            Argument::Tensor(left),
                            Argument::Tensor(right),
                        ],
                    )?
                    else {
                        return Err(invalid("threadpool matmul construction failed"));
                    };
                    id
                }
                _ => return Err(invalid("threadpool operation unsupported")),
            };
            let extent = backend.info(tensor)?.bytes()?;
            bytes = bytes
                .checked_add(extent)
                .ok_or_else(|| invalid("threadpool storage extent overflow"))?;
            if bytes > memory / 2 {
                return Err(invalid("threadpool tensor storage exceeds context bound"));
            }
            tensors.push(tensor);
        }
        let output = tensors[number(&description["Output"], 0, tensors.len() - 1)?];
        let Observation::Graph(graph) = backend.call(
            "ggml_new_graph_custom",
            &[
                Argument::Context(0),
                Argument::Integer(i64::try_from(nodes.len())?),
                Argument::Bool(false),
            ],
        )?
        else {
            return Err(invalid("threadpool graph construction failed"));
        };
        backend.call(
            "ggml_build_forward_expand",
            &[Argument::Graph(graph), Argument::Tensor(output)],
        )?;
        counts.push(backend.graph_nodes(graph)?.0.len());
        graphs.push(graph);
    }
    backend.allocate(0)?;
    let pool = CpuThreadPool::new(threads)?;
    let mut plans: Vec<Option<CpuPlan>> = (0..16).map(|_| None).collect();
    let mut planned = 0_usize;
    let mut statuses = Vec::new();
    let mut times = Vec::new();
    let mut started = None;
    for action in actions {
        match action["Op"].as_str() {
            Some("plan") => {
                let slot = number(&action["Slot"], 0, 15)?;
                let graph = graphs[number(&action["Graph"], 0, graphs.len() - 1)?];
                plans[slot] =
                    Some(pool.plan(&backend, graph, number(&action["Threads"], 1, threads)?)?);
                planned += 1;
            }
            Some("compute") => {
                let plan = plans[number(&action["Slot"], 0, 15)?]
                    .as_ref()
                    .ok_or_else(|| invalid("threadpool plan used before construction"))?;
                pool.compute(&mut backend, plan)?;
                statuses.push(0);
            }
            Some("timer-start") => {
                if started.is_some() {
                    return Err(invalid("threadpool timer already running"));
                }
                started = Some(Instant::now());
            }
            Some("timer-stop") => {
                times.push(u64::try_from(
                    started
                        .take()
                        .ok_or_else(|| invalid("threadpool timer not running"))?
                        .elapsed()
                        .as_nanos(),
                )?);
            }
            _ => return Err(invalid("threadpool action unsupported")),
        }
    }
    if started.is_some() {
        return Err(invalid("threadpool timer left running"));
    }
    Ok(
        json!({"GraphNodeCounts":counts,"Plans":planned,"ComputeStatuses":statuses,"Nanoseconds":times}),
    )
}

fn graph(width: usize, repetitions: usize) -> Value {
    let mut nodes = vec![json!({"Op":"input","Type":"F32","Dimensions":[width]})];
    let mut output = 0;
    for _ in 0..repetitions {
        let a = nodes.len();
        nodes.push(json!({"Op":"input","Type":"Q4_0","Dimensions":[width,128]}));
        output = nodes.len();
        nodes.push(json!({"Op":"mul_mat","A":a,"B":output-2}));
        let d = nodes.len();
        nodes.push(json!({"Op":"input","Type":"Q4_0","Dimensions":[128,width]}));
        let previous = output;
        output = nodes.len();
        nodes.push(json!({"Op":"mul_mat","A":d,"B":previous}));
    }
    json!({"Nodes":nodes,"Output":output})
}

fn plan(slot: usize, graph: usize, threads: usize) -> Value {
    json!({"Op":"plan","Slot":slot,"Graph":graph,"Threads":threads})
}

fn schedule(kind: usize, threads: usize, rounds: usize) -> (Value, usize, usize) {
    let mut actions = Vec::new();
    let (graphs, plans, computes) = match kind {
        0 => {
            actions.push(plan(0, 0, threads));
            actions.push(json!({"Op":"compute","Slot":0})); // Original untimed warmup.
            actions.push(json!({"Op":"timer-start"}));
            actions.extend((0..rounds).map(|_| json!({"Op":"compute","Slot":0})));
            actions.push(json!({"Op":"timer-stop"}));
            (vec![graph(64, 1000)], 1, rounds + 1)
        }
        1 => {
            for i in 0..rounds * 100 {
                actions.push(plan(0, 0, if i % 4 == 0 { 1 } else { threads }));
                actions.push(json!({"Op":"compute","Slot":0}));
            }
            (vec![graph(64, 2)], rounds * 100, rounds * 100)
        }
        _ => {
            for i in 0..rounds * 10 {
                let count = if i % 4 == 0 { 1 } else { threads };
                // Both plans/workspaces coexist before either graph computes.
                actions.push(plan(0, 0, count));
                actions.push(plan(1, 1, count));
                actions.push(json!({"Op":"compute","Slot":0}));
                actions.push(json!({"Op":"compute","Slot":1}));
            }
            (vec![graph(64, 2), graph(256, 4)], rounds * 20, rounds * 20)
        }
    };
    (
        json!({"Kind":"cpu-threadpool","ContextBytes":1_u64<<30,"PoolThreads":threads,"Graphs":graphs,"Actions":actions}),
        plans,
        computes,
    )
}

pub(super) fn run_with_parameters(engine: &Engine, threads: usize, rounds: usize) -> Result<Value> {
    if env::var("GITHUB_ACTIONS").as_deref() != Ok("true")
        || env::var("GITHUB_REF").as_deref() != Ok("refs/heads/main")
    {
        return Err(invalid(
            "The timed barrier source runs only in GitHub Actions on main",
        ));
    }
    if !(1..=256).contains(&threads) || !(1..=125).contains(&rounds) {
        return Err(invalid(
            "barrier parameters exceed bounded original workload transport",
        ));
    }
    let mut observations = Vec::new();
    for (kind, name) in ["constant-threads", "changing-threads", "multiple-graphs"]
        .into_iter()
        .enumerate()
    {
        let (operation, plans, computes) = schedule(kind, threads, rounds);
        let started = Instant::now();
        let Some(response) =
            engine.execute_common(name, &json!({"SchemaVersion":1,"Operations":[operation]}))?
        else {
            return Ok(result(ID, false, "Rust runtime has no production graph/threadpool API for the original barrier workload".into()));
        };
        if response["SchemaVersion"] != 1 || response["Results"].as_array().map(Vec::len) != Some(1)
        {
            return Err(invalid("barrier response framing differs"));
        }
        let value = &response["Results"][0];
        super::llama_grammar::exact_fields(
            value,
            &["GraphNodeCounts", "Plans", "ComputeStatuses", "Nanoseconds"],
        )?;
        let expected_nodes = if kind == 0 {
            vec![2000]
        } else if kind == 1 {
            vec![4]
        } else {
            vec![4, 8]
        };
        let statuses = value["ComputeStatuses"]
            .as_array()
            .ok_or_else(|| invalid("barrier compute statuses missing"))?;
        if value["GraphNodeCounts"] != json!(expected_nodes)
            || value["Plans"] != plans
            || statuses.len() != computes
            || statuses.iter().any(|status| status.as_i64() != Some(0))
            || value["Nanoseconds"].as_array().map(Vec::len) != Some(usize::from(kind == 0))
        {
            return Ok(result(
                ID,
                false,
                format!("{name}: additional workload completion guard failed: {value}"),
            ));
        }
        observations.push(json!({"Name":name,"Plans":plans,"Computes":computes,"GraphNodes":expected_nodes,"NativeNanoseconds":value["Nanoseconds"],"WallNanosecondsIncludingTransport":started.elapsed().as_nanos().to_string()}));
    }
    Ok(result(ID, true, json!({"SourceSha256":"44a60801134d178c8e2c6789c471909ed83c0b8c71055b259478b4d396220cee","Category":"CI-only timed workload completion","OriginalNumericAssertions":0,"AdditionalGuards":"graph extent, exact supplied plan/compute counts, production success statuses","Threads":threads,"Rounds":rounds,"InputInitialization":"source leaves GGML tensor data unspecified; no output values are read","TimingThreshold":null,"Results":observations}).to_string()))
}

pub(super) fn run(engine: &Engine) -> Result<Value> {
    let threads = std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .clamp(1, 4);
    run_with_parameters(engine, threads, 100)
}

#[cfg(test)]
mod tests {
    #[test]
    fn generic_untimed_plans_call_the_production_pool() -> super::Result<()> {
        let operation = serde_json::json!({"Kind":"cpu-threadpool","ContextBytes":65536,"PoolThreads":3,
            "Graphs":[{"Nodes":[{"Op":"input","Type":"Q4_0","Dimensions":[32,5]}, {"Op":"input","Type":"F32","Dimensions":[32]}, {"Op":"mul_mat","A":0,"B":1}],"Output":2}],
            "Actions":[{"Op":"plan","Slot":0,"Graph":0,"Threads":1},{"Op":"plan","Slot":1,"Graph":0,"Threads":3},{"Op":"compute","Slot":1},{"Op":"compute","Slot":0}]} );
        let observation = super::execute(&operation)?;
        assert_eq!(
            observation,
            serde_json::json!({"GraphNodeCounts":[1],"Plans":2,"ComputeStatuses":[0,0],"Nanoseconds":[]})
        );
        Ok(())
    }

    #[test]
    fn original_graph_edges_and_switch_lifetimes_are_retained() {
        let graph = super::graph(64, 2);
        assert_eq!(graph["Nodes"][2]["B"], 0);
        assert_eq!(graph["Nodes"][6]["B"], 4);
        assert_eq!(graph["Output"], 8);
        let (operation, plans, computes) = super::schedule(2, 4, 1);
        assert_eq!((plans, computes), (20, 20));
        assert_eq!(operation["Actions"][0]["Threads"], 1);
        assert_eq!(operation["Actions"][1]["Threads"], 1);
        assert_eq!(operation["Actions"][4]["Threads"], 4);
        assert_eq!(operation["Actions"][16]["Threads"], 1);
    }
}
