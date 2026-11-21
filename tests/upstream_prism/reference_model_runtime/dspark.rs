//! Clean-room `DSpark`, speculative decoding, trace capture, and common-sampler session ops.
//!
//! No `bonsai_prism_runtime`, `bonsai_gguf`, or native GGML libraries are linked or called.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use serde_json::{Value, json};

use super::forward::{
    BatchData, BatchRow, ContextState, DsparkStaged, RuntimeEnv, TraceCollector,
    check_trace_limits, run_decode,
};
use super::gguf_model::ModelData;
use super::gguf_model::checked_session_file;
use super::sampler::SamplerNode;
use crate::{Result, invalid};

const MAX_SPECULATIVE: usize = 64;
const MAX_CAPTURE_LAYERS: usize = 16;
const MAX_CAPTURE_BYTES: usize = 16 * 1024 * 1024;
const MAX_LOGIT_ROWS: usize = 512;

#[derive(Clone)]
pub(super) struct SpeculativeState {
    pub(super) draft_ctx: u64,
    pub(super) target_ctx: u64,
    pub(super) n_max: usize,
    pub(super) expected_prefix: Option<usize>,
    pub(super) prompts: BTreeMap<i32, Vec<i32>>,
    pub(super) staged: BTreeMap<i32, DsparkStaged>,
    pub(super) draft_calls: u64,
    pub(super) total_drafted: u64,
    pub(super) total_accepted: u64,
    pub(super) accept_calls: u64,
}

#[derive(Default)]
pub(super) struct DsparkState {
    pub(super) specs: BTreeMap<u64, SpeculativeState>,
    pub(super) traces: BTreeMap<u64, TraceCollector>,
    pub(super) next_id: u64,
}

impl DsparkState {
    pub(super) const fn new() -> Self {
        Self {
            specs: BTreeMap::new(),
            traces: BTreeMap::new(),
            next_id: 1,
        }
    }

    fn alloc_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
}

pub(super) struct DsparkEnv<'a> {
    pub(super) models: &'a mut BTreeMap<u64, Arc<ModelData>>,
    pub(super) contexts: &'a mut BTreeMap<u64, ContextState>,
    pub(super) batches: &'a mut BTreeMap<u64, BatchData>,
    pub(super) samplers: &'a mut BTreeMap<u64, SamplerNode>,
    pub(super) state: &'a mut DsparkState,
    pub(super) env: &'a mut RuntimeEnv,
    pub(super) next_handle: &'a mut dyn FnMut() -> u64,
    pub(super) session_dir: &'a Path,
}

pub(super) fn handles(op: &str) -> bool {
    matches!(
        op,
        "dspark-meta"
            | "dspark-stage"
            | "dspark-stage-clear"
            | "dspark-context-info"
            | "speculative-init"
            | "speculative-needs-capture"
            | "speculative-begin"
            | "speculative-stage"
            | "speculative-process"
            | "speculative-draft"
            | "speculative-accept"
            | "speculative-stats"
            | "speculative-release"
            | "common-sampler-new"
            | "common-sampler-free"
            | "common-sampler-sample"
            | "common-sampler-accept"
            | "common-sampler-sample-and-accept-n"
            | "greedy-sampler-new"
            | "context-set-sampler"
            | "capture-layers"
            | "causal-attention"
            | "observe-decode"
            | "decode-output-logits"
            | "sampled-tokens"
            | "model-vocabulary"
            | "model-tensor-info"
            | "model-tensor-file"
            | "model-output-identity"
            | "trace-new"
            | "context-traces"
            | "graph-allow-reuse"
            | "process-environment"
            | "process-environment-restore"
    )
}

pub(super) fn execute(op: &str, req: &Value, ctx: &mut DsparkEnv<'_>) -> Result<Value> {
    match op {
        "dspark-meta" => exec_dspark_meta(req, ctx.models),
        "dspark-stage" => exec_dspark_stage(req, ctx.contexts),
        "dspark-stage-clear" => exec_dspark_stage_clear(req, ctx.contexts),
        "dspark-context-info" => exec_context_info(req, ctx.contexts),
        "speculative-init" => exec_spec_init(req, ctx),
        "speculative-needs-capture" => exec_spec_needs_capture(req, ctx),
        "speculative-begin" => exec_spec_begin(req, ctx.state),
        "speculative-stage" => exec_spec_stage(req, ctx.state),
        "speculative-process" => exec_spec_process(req, ctx),
        "speculative-draft" => exec_spec_draft(req, ctx),
        "speculative-accept" => exec_spec_accept(req, ctx.state),
        "speculative-stats" => exec_spec_stats(req, ctx.state),
        "speculative-release" => exec_spec_release(req, ctx.state),
        "common-sampler-new" => exec_common_sampler_new(req, ctx),
        "common-sampler-free" => exec_common_sampler_free(req, ctx.samplers),
        "common-sampler-sample" => exec_common_sampler_sample(req, ctx),
        "common-sampler-accept" => exec_common_sampler_accept(req, ctx.samplers),
        "common-sampler-sample-and-accept-n" => exec_sample_accept_n(req, ctx),
        "greedy-sampler-new" => Ok(exec_greedy_sampler_new(ctx)),
        "context-set-sampler" => exec_context_set_sampler(req, ctx),
        "capture-layers" => exec_capture_layers(req, ctx.contexts),
        "causal-attention" => exec_causal_attention(req, ctx.contexts),
        "observe-decode" => exec_observe_decode(req, ctx),
        "decode-output-logits" => exec_decode_output_logits(req, ctx.contexts),
        "sampled-tokens" => exec_sampled_tokens(req, ctx.contexts),
        "model-vocabulary" => exec_model_vocabulary(req, ctx.models),
        "model-tensor-info" => exec_model_tensor_info(req, ctx.models),
        "model-tensor-file" => exec_model_tensor_file(req, ctx.models, ctx.session_dir),
        "model-output-identity" => exec_model_output_identity(req, ctx.models),
        "trace-new" => exec_trace_new(req, ctx.state),
        "context-traces" => exec_context_traces(req, ctx.state),
        "graph-allow-reuse" => Ok(exec_graph_allow_reuse(req)),
        "process-environment" => exec_process_env(req, ctx.env),
        "process-environment-restore" => exec_process_env_restore(req, ctx.env),
        _ => Err(invalid(format!("unknown dspark operation {op}"))),
    }
}

pub(super) fn observe_production(operation: &Value, ctx: &mut DsparkEnv<'_>) -> Result<Value> {
    let inner_op = operation["Op"]
        .as_str()
        .ok_or_else(|| invalid("observe-production Op missing"))?
        .to_owned();
    let res = if handles(&inner_op) {
        execute(&inner_op, operation, ctx)
    } else {
        Err(invalid(format!(
            "invalid-argument:unsupported observed operation {inner_op}"
        )))
    };
    match res {
        Ok(val) => Ok(json!({ "Value": val, "Exception": Value::Null })),
        Err(err) => Ok(json!({
            "Value": Value::Null,
            "Exception": classify_exception(&err.to_string()),
        })),
    }
}

pub(super) fn classify_exception(msg: &str) -> Value {
    let (kind, clean) = msg.strip_prefix("invalid-argument:").map_or_else(
        || {
            msg.strip_prefix("runtime-error:")
                .map_or(("runtime-error", msg), |rest| ("runtime-error", rest))
        },
        |rest| ("invalid-argument", rest),
    );
    json!({
        "Kind": kind,
        "Message": clean,
    })
}

fn get_model(models: &BTreeMap<u64, Arc<ModelData>>, id: u64) -> Result<&Arc<ModelData>> {
    models
        .get(&id)
        .ok_or_else(|| invalid("unknown Model handle"))
}

fn get_ctx(contexts: &BTreeMap<u64, ContextState>, id: u64) -> Result<&ContextState> {
    contexts
        .get(&id)
        .ok_or_else(|| invalid("unknown Context handle"))
}

fn get_ctx_mut(contexts: &mut BTreeMap<u64, ContextState>, id: u64) -> Result<&mut ContextState> {
    contexts
        .get_mut(&id)
        .ok_or_else(|| invalid("unknown Context handle"))
}

fn req_u64(req: &Value, field: &str) -> Result<u64> {
    req[field]
        .as_u64()
        .ok_or_else(|| invalid(format!("missing integer field {field}")))
}

fn exec_dspark_meta(req: &Value, models: &BTreeMap<u64, Arc<ModelData>>) -> Result<Value> {
    let model = get_model(models, req_u64(req, "Model")?)?;
    let cap_count = if model.dspark_target_layers.is_empty() {
        (model.n_embd_cap / model.n_embd.max(1)).max(1)
    } else {
        model.dspark_target_layers.len()
    };
    Ok(json!({
        "Found": model.is_dspark,
        "Embedding": model.n_embd,
        "CaptureCount": cap_count,
        "CaptureEmbedding": model.n_embd_cap,
        "BlockSize": model.block_size,
        "Vocabulary": model.vocab_size,
        "MaskToken": model.mask_token_id,
        "MarkovRank": model.markov_rank,
        "GraphCorrected": model.graph_corrected,
        "TargetLayers": model.dspark_target_layers,
        "LayerCount": model.n_layer,
        "MaxCaptureLayers": MAX_CAPTURE_LAYERS,
    }))
}

fn decode_f32_hex(hex_str: &str) -> Result<Vec<f32>> {
    let bytes = hex::decode(hex_str)?;
    if bytes.len() > MAX_CAPTURE_BYTES || !bytes.len().is_multiple_of(4) {
        return Err(invalid("float buffer size invalid"));
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect())
}

fn encode_f32_hex(values: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for &v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    hex::encode(bytes)
}

fn parse_staged(req: &Value) -> Result<DsparkStaged> {
    let rows = usize::try_from(req_u64(req, "Rows")?)?;
    let width = usize::try_from(req_u64(req, "Width")?)?;
    let features = decode_f32_hex(
        req["FeaturesHex"]
            .as_str()
            .ok_or_else(|| invalid("FeaturesHex missing"))?,
    )?;
    if features.len() != rows * width {
        return Err(invalid("staged feature count mismatch"));
    }
    let pos_arr = req["Positions"]
        .as_array()
        .ok_or_else(|| invalid("Positions missing"))?;
    if pos_arr.len() != rows {
        return Err(invalid("staged position count mismatch"));
    }
    let mut positions = Vec::with_capacity(rows);
    for p in pos_arr {
        positions.push(i32::try_from(
            p.as_i64().ok_or_else(|| invalid("position invalid"))?,
        )?);
    }
    Ok(DsparkStaged {
        width,
        features,
        positions,
    })
}

fn exec_dspark_stage(req: &Value, contexts: &mut BTreeMap<u64, ContextState>) -> Result<Value> {
    let staged = parse_staged(req)?;
    let ctx = get_ctx_mut(contexts, req_u64(req, "Context")?)?;
    ctx.staged = Some(staged);
    Ok(json!({ "Staged": true }))
}

fn exec_dspark_stage_clear(
    req: &Value,
    contexts: &mut BTreeMap<u64, ContextState>,
) -> Result<Value> {
    let ctx = get_ctx_mut(contexts, req_u64(req, "Context")?)?;
    ctx.staged = None;
    Ok(Value::Null)
}

fn exec_context_info(req: &Value, contexts: &BTreeMap<u64, ContextState>) -> Result<Value> {
    let ctx = get_ctx(contexts, req_u64(req, "Context")?)?;
    let rs_seqs = if ctx.model.is_hybrid || ctx.model.is_recurrent {
        ctx.n_seq_max
    } else {
        0
    };
    Ok(json!({
        "ContextSize": ctx.n_ctx,
        "BatchSize": ctx.n_batch,
        "UbatchSize": ctx.n_ubatch,
        "Sequences": ctx.n_seq_max,
        "CaptureCount": ctx.capture_layers.len(),
        "Hybrid": ctx.model.is_hybrid,
        "Recurrent": ctx.model.is_recurrent,
        "RsSequences": rs_seqs,
    }))
}

fn exec_spec_init(req: &Value, env: &mut DsparkEnv<'_>) -> Result<Value> {
    if env.state.specs.len() >= MAX_SPECULATIVE {
        return Err(invalid("speculative handle bound exceeded"));
    }
    let draft_ctx_id = req_u64(req, "DraftContext")?;
    let target_ctx_id = req_u64(req, "TargetContext")?;
    let n_max = usize::try_from(req_u64(req, "Max")?)?;
    let _ = req_u64(req, "Min")?;
    let n_seq_max = usize::try_from(req_u64(req, "Sequences")?)?;
    let draft_ctx = get_ctx(env.contexts, draft_ctx_id)?;
    let _target = get_ctx(env.contexts, target_ctx_id)?;
    let expected_prefix = env
        .env
        .parse_correction_prefix(draft_ctx.model.block_size)?;
    if let Some(prefix) = expected_prefix
        && (n_max > prefix || n_seq_max > 1)
    {
        return Err(invalid(
            "runtime-error:LLAMA_DSPARK_CORRECTION_PREFIX incompatible with speculative bounds",
        ));
    }
    let id = env.state.alloc_id();
    env.state.specs.insert(
        id,
        SpeculativeState {
            draft_ctx: draft_ctx_id,
            target_ctx: target_ctx_id,
            n_max,
            expected_prefix,
            prompts: BTreeMap::new(),
            staged: BTreeMap::new(),
            draft_calls: 0,
            total_drafted: 0,
            total_accepted: 0,
            accept_calls: 0,
        },
    );
    Ok(json!({ "Speculative": id }))
}

fn exec_spec_needs_capture(req: &Value, env: &DsparkEnv<'_>) -> Result<Value> {
    let id = req_u64(req, "Speculative")?;
    let spec = env
        .state
        .specs
        .get(&id)
        .ok_or_else(|| invalid("unknown Speculative handle"))?;
    let draft_ctx = get_ctx(env.contexts, spec.draft_ctx)?;
    let needed = draft_ctx.model.is_dspark && draft_ctx.model.n_embd_cap > 0;
    Ok(json!({ "Needed": needed }))
}

fn exec_spec_begin(req: &Value, state: &mut DsparkState) -> Result<Value> {
    let id = req_u64(req, "Speculative")?;
    let seq = i32::try_from(req["Sequence"].as_i64().unwrap_or(0))?;
    let prompt_arr = req["Prompt"]
        .as_array()
        .ok_or_else(|| invalid("speculative-begin Prompt must be array"))?;
    let mut prompt = Vec::with_capacity(prompt_arr.len());
    for t in prompt_arr {
        prompt.push(i32::try_from(
            t.as_i64().ok_or_else(|| invalid("prompt token invalid"))?,
        )?);
    }
    let spec = state
        .specs
        .get_mut(&id)
        .ok_or_else(|| invalid("unknown Speculative handle"))?;
    spec.prompts.insert(seq, prompt);
    Ok(Value::Null)
}

fn exec_spec_stage(req: &Value, state: &mut DsparkState) -> Result<Value> {
    let id = req_u64(req, "Speculative")?;
    let seq = i32::try_from(req["Sequence"].as_i64().unwrap_or(0))?;
    let staged = parse_staged(req)?;
    let spec = state
        .specs
        .get_mut(&id)
        .ok_or_else(|| invalid("unknown Speculative handle"))?;
    spec.staged.insert(seq, staged);
    Ok(json!({ "Staged": true }))
}

fn exec_spec_process(req: &Value, env: &mut DsparkEnv<'_>) -> Result<Value> {
    let id = req_u64(req, "Speculative")?;
    let batch_id = req_u64(req, "Batch")?;
    let spec = env
        .state
        .specs
        .get_mut(&id)
        .ok_or_else(|| invalid("unknown Speculative handle"))?;
    let target_ctx = get_ctx(env.contexts, spec.target_ctx)?;
    let batch = env
        .batches
        .get(&batch_id)
        .ok_or_else(|| invalid("unknown Batch handle"))?;
    let mut seq_features: BTreeMap<i32, (Vec<f32>, Vec<i32>, usize)> = BTreeMap::new();
    for (row_idx, row) in batch.rows.iter().enumerate() {
        let seq = *row.seq_ids.first().unwrap_or(&0);
        if let Some(layers) = target_ctx.captured_rows.get(row_idx) {
            let flat: Vec<f32> = layers.iter().flatten().copied().collect();
            let width = flat.len();
            let entry = seq_features
                .entry(seq)
                .or_insert((Vec::new(), Vec::new(), width));
            entry.0.extend_from_slice(&flat);
            entry.1.push(row.pos);
        }
    }
    for (seq, (features, positions, width)) in seq_features {
        if width > 0 {
            spec.staged.insert(
                seq,
                DsparkStaged {
                    width,
                    features,
                    positions,
                },
            );
        }
    }
    Ok(json!({ "Processed": true }))
}

fn exec_spec_draft(req: &Value, env: &mut DsparkEnv<'_>) -> Result<Value> {
    let id = req_u64(req, "Speculative")?;
    let seq = i32::try_from(req["Sequence"].as_i64().unwrap_or(0))?;
    let max_arg = req["Max"].as_i64().unwrap_or(-1);
    let past = i32::try_from(req["Past"].as_i64().unwrap_or(0))?;
    let last = i32::try_from(req["Last"].as_i64().unwrap_or(0))?;
    let spec = env
        .state
        .specs
        .get_mut(&id)
        .ok_or_else(|| invalid("unknown Speculative handle"))?;
    let draft_ctx_id = spec.draft_ctx;
    let draft_ctx = get_ctx_mut(env.contexts, draft_ctx_id)?;
    let current_prefix = env
        .env
        .parse_correction_prefix(draft_ctx.model.block_size)?;
    if current_prefix != spec.expected_prefix {
        return Err(invalid(
            "runtime-error:LLAMA_DSPARK_CORRECTION_PREFIX changed after speculative-init",
        ));
    }
    let staged_opt = spec.staged.get(&seq).cloned();
    if let Some(st) = staged_opt {
        draft_ctx.staged = Some(st);
    }
    let block = draft_ctx.model.block_size;
    let count = if max_arg < 0 {
        spec.n_max.min(block)
    } else {
        usize::try_from(max_arg)?.min(spec.n_max).min(block)
    };
    let count = current_prefix.map_or(count, |p| count.min(p)).max(1);
    let mut rows = Vec::new();
    let ctx_pos_len = draft_ctx.staged.as_ref().map_or(0, |st| {
        for &p in &st.positions {
            rows.push(BatchRow {
                token: 0,
                pos: p,
                seq_ids: vec![seq],
                logits: false,
            });
        }
        st.positions.len()
    });
    let mask = draft_ctx.model.mask_token_id;
    let full_depth = if draft_ctx.model.is_dspark && draft_ctx.model.graph_corrected {
        block
    } else {
        count
    };
    for idx in 0..full_depth {
        rows.push(BatchRow {
            token: if idx == 0 { last } else { mask },
            pos: past + i32::try_from(idx)?,
            seq_ids: vec![seq],
            logits: idx < count,
        });
    }
    let _ = ctx_pos_len;
    let synth_batch = BatchData {
        capacity: rows.len(),
        rows,
    };
    draft_ctx.causal_attention = false;
    run_decode(draft_ctx, &synth_batch, &mut env.state.traces, env.env)?;
    let tokens: Vec<i32> = draft_ctx
        .sampled_tokens
        .iter()
        .take(count)
        .copied()
        .collect();
    spec.draft_calls += 1;
    spec.total_drafted += u64::try_from(tokens.len())?;
    Ok(json!({ "Tokens": tokens }))
}

fn exec_spec_accept(req: &Value, state: &mut DsparkState) -> Result<Value> {
    let id = req_u64(req, "Speculative")?;
    let accepted = req["Accepted"].as_u64().unwrap_or(0);
    let spec = state
        .specs
        .get_mut(&id)
        .ok_or_else(|| invalid("unknown Speculative handle"))?;
    spec.total_accepted += accepted;
    spec.accept_calls += 1;
    Ok(Value::Null)
}

fn exec_spec_stats(req: &Value, state: &DsparkState) -> Result<Value> {
    let id = req_u64(req, "Speculative")?;
    let spec = state
        .specs
        .get(&id)
        .ok_or_else(|| invalid("unknown Speculative handle"))?;
    Ok(json!({
        "DraftCalls": spec.draft_calls,
        "TotalDrafted": spec.total_drafted,
        "TotalAccepted": spec.total_accepted,
        "AcceptCalls": spec.accept_calls,
    }))
}

fn exec_spec_release(req: &Value, state: &mut DsparkState) -> Result<Value> {
    let id = req_u64(req, "Speculative")?;
    state
        .specs
        .remove(&id)
        .ok_or_else(|| invalid("unknown Speculative handle"))?;
    Ok(Value::Null)
}

fn exec_common_sampler_new(req: &Value, env: &mut DsparkEnv<'_>) -> Result<Value> {
    let seed = u32::try_from(req["Seed"].as_u64().unwrap_or(0))?;
    let temp = f32::from_bits(u32::try_from(req["TemperatureBits"].as_u64().unwrap_or(0))?);
    let node = if temp == 0.0 {
        SamplerNode::Greedy
    } else {
        SamplerNode::Chain(vec![
            SamplerNode::Temp(temp),
            SamplerNode::Dist(super::sampler::Mt19937::new(seed)),
        ])
    };
    let id = (env.next_handle)();
    env.samplers.insert(id, node);
    Ok(json!({ "Sampler": id }))
}

fn exec_common_sampler_free(
    req: &Value,
    samplers: &mut BTreeMap<u64, SamplerNode>,
) -> Result<Value> {
    let id = req_u64(req, "Sampler")?;
    samplers.remove(&id);
    Ok(Value::Null)
}

fn exec_common_sampler_sample(req: &Value, env: &mut DsparkEnv<'_>) -> Result<Value> {
    let s_id = req_u64(req, "Sampler")?;
    let c_id = req_u64(req, "Context")?;
    let index = req["Index"].as_i64().unwrap_or(-1);
    let ctx = get_ctx(env.contexts, c_id)?;
    let logits = resolve_logit_row(&ctx.output_logits, index)?;
    let mut sampler = env
        .samplers
        .get(&s_id)
        .ok_or_else(|| invalid("unknown Sampler handle"))?
        .clone();
    let (tok, _) = sampler.sample_logits(logits, &mut None)?;
    env.samplers.insert(s_id, sampler);
    Ok(json!({ "Token": tok }))
}

fn exec_common_sampler_accept(
    req: &Value,
    samplers: &mut BTreeMap<u64, SamplerNode>,
) -> Result<Value> {
    let s_id = req_u64(req, "Sampler")?;
    let token = i32::try_from(
        req["Token"]
            .as_i64()
            .ok_or_else(|| invalid("accept Token missing"))?,
    )?;
    if let Some(sampler) = samplers.get_mut(&s_id) {
        sampler.accept(token);
    }
    Ok(Value::Null)
}

fn exec_sample_accept_n(req: &Value, env: &mut DsparkEnv<'_>) -> Result<Value> {
    let s_id = req_u64(req, "Sampler")?;
    let c_id = req_u64(req, "Context")?;
    let draft_arr = req["Draft"]
        .as_array()
        .ok_or_else(|| invalid("sample-and-accept-n Draft must be array"))?;
    let ctx = get_ctx(env.contexts, c_id)?;
    let mut sampler = env
        .samplers
        .get(&s_id)
        .ok_or_else(|| invalid("unknown Sampler handle"))?
        .clone();
    let mut accepted = Vec::new();
    for (i, d_tok) in draft_arr.iter().enumerate() {
        let draft_id = i32::try_from(
            d_tok
                .as_i64()
                .ok_or_else(|| invalid("draft token not int"))?,
        )?;
        let Some(row_logits) = ctx.output_logits.get(i) else {
            break;
        };
        let (tok, _) = sampler.sample_logits(row_logits, &mut None)?;
        sampler.accept(tok);
        accepted.push(tok);
        if tok != draft_id {
            break;
        }
    }
    if accepted.len() == draft_arr.len() {
        let bonus_row = ctx
            .output_logits
            .get(draft_arr.len())
            .or_else(|| ctx.output_logits.last())
            .ok_or_else(|| invalid("target logits absent for sample-and-accept-n"))?;
        let (bonus, _) = sampler.sample_logits(bonus_row, &mut None)?;
        sampler.accept(bonus);
        accepted.push(bonus);
    }
    if accepted.is_empty() {
        let last_row = ctx
            .output_logits
            .last()
            .ok_or_else(|| invalid("target logits absent"))?;
        let (tok, _) = sampler.sample_logits(last_row, &mut None)?;
        sampler.accept(tok);
        accepted.push(tok);
    }
    env.samplers.insert(s_id, sampler);
    Ok(json!({ "Tokens": accepted }))
}

fn exec_greedy_sampler_new(env: &mut DsparkEnv<'_>) -> Value {
    let id = (env.next_handle)();
    env.samplers.insert(id, SamplerNode::Greedy);
    json!({ "Sampler": id })
}

fn exec_context_set_sampler(req: &Value, env: &mut DsparkEnv<'_>) -> Result<Value> {
    let c_id = req_u64(req, "Context")?;
    let seq = i32::try_from(req["Sequence"].as_i64().unwrap_or(0))?;
    let sampler_opt = req["Sampler"].as_u64();
    let ctx = get_ctx_mut(env.contexts, c_id)?;
    if let Some(sid) = sampler_opt {
        if !env.samplers.contains_key(&sid) {
            return Ok(json!({ "Set": false }));
        }
        ctx.seq_samplers.insert(seq, sid);
    } else {
        ctx.seq_samplers.remove(&seq);
    }
    Ok(json!({ "Set": true }))
}

fn exec_capture_layers(req: &Value, contexts: &mut BTreeMap<u64, ContextState>) -> Result<Value> {
    let c_id = req_u64(req, "Context")?;
    let arr = req["Layers"]
        .as_array()
        .ok_or_else(|| invalid("invalid-argument:Layers must be array"))?;
    let ctx = get_ctx(contexts, c_id)?;
    let n_layer = i32::try_from(ctx.model.n_layer)?;
    if arr.len() > MAX_CAPTURE_LAYERS {
        return Err(invalid(
            "invalid-argument:capture layer list exceeds maximum",
        ));
    }
    let mut parsed = Vec::with_capacity(arr.len());
    for v in arr {
        let idx = i32::try_from(
            v.as_i64()
                .ok_or_else(|| invalid("invalid-argument:layer index not integer"))?,
        )?;
        if idx < 0 || idx >= n_layer {
            return Err(invalid("invalid-argument:layer index out of range"));
        }
        parsed.push(idx);
    }
    let ctx_mut = get_ctx_mut(contexts, c_id)?;
    ctx_mut.capture_layers = parsed;
    Ok(Value::Null)
}

fn exec_causal_attention(req: &Value, contexts: &mut BTreeMap<u64, ContextState>) -> Result<Value> {
    let c_id = req_u64(req, "Context")?;
    let causal = req["Value"]
        .as_bool()
        .ok_or_else(|| invalid("causal-attention Value must be bool"))?;
    let ctx = get_ctx_mut(contexts, c_id)?;
    ctx.causal_attention = causal;
    Ok(Value::Null)
}

fn exec_observe_decode(req: &Value, env: &mut DsparkEnv<'_>) -> Result<Value> {
    let c_id = req_u64(req, "Context")?;
    let b_id = req_u64(req, "Batch")?;
    let batch = env
        .batches
        .get(&b_id)
        .ok_or_else(|| invalid("unknown Batch handle"))?
        .clone();
    let ctx = get_ctx_mut(env.contexts, c_id)?;
    match run_decode(ctx, &batch, &mut env.state.traces, env.env) {
        Ok(()) => Ok(json!({ "Code": 0, "Exception": Value::Null })),
        Err(err) => Ok(json!({
            "Code": 1,
            "Exception": classify_exception(&err.to_string()),
        })),
    }
}

fn exec_decode_output_logits(req: &Value, contexts: &BTreeMap<u64, ContextState>) -> Result<Value> {
    let ctx = get_ctx(contexts, req_u64(req, "Context")?)?;
    if ctx.output_logits.is_empty() {
        return Ok(json!({ "Present": false, "Rows": 0, "DataHex": "" }));
    }
    let flat: Vec<f32> = ctx.output_logits.iter().flatten().copied().collect();
    Ok(json!({
        "Present": true,
        "Rows": ctx.output_logits.len(),
        "DataHex": encode_f32_hex(&flat),
    }))
}

fn exec_sampled_tokens(req: &Value, contexts: &BTreeMap<u64, ContextState>) -> Result<Value> {
    let ctx = get_ctx(contexts, req_u64(req, "Context")?)?;
    let indices = req["Indices"]
        .as_array()
        .ok_or_else(|| invalid("sampled-tokens Indices must be array"))?;
    let len = i64::try_from(ctx.sampled_tokens.len())?;
    let mut tokens = Vec::with_capacity(indices.len());
    for raw in indices {
        let idx_i64 = raw
            .as_i64()
            .ok_or_else(|| invalid("sampled index must be integer"))?;
        let pos = if idx_i64 < 0 { len + idx_i64 } else { idx_i64 };
        let u_pos = usize::try_from(pos)
            .ok()
            .filter(|&p| p < ctx.sampled_tokens.len())
            .ok_or_else(|| invalid("sampled token index out of range"))?;
        tokens.push(ctx.sampled_tokens[u_pos]);
    }
    Ok(json!({ "Tokens": tokens }))
}

fn exec_model_vocabulary(req: &Value, models: &BTreeMap<u64, Arc<ModelData>>) -> Result<Value> {
    let model = get_model(models, req_u64(req, "Model")?)?;
    let mut eog = Vec::new();
    for tok in 0..model.vocab.count().min(1_000_000) {
        let tok_i = i32::try_from(tok)?;
        if model.vocab.is_eog(tok_i) {
            eog.push(tok_i);
        }
    }
    Ok(json!({
        "Count": model.vocab.count(),
        "Bos": model.vocab.bos(),
        "Eos": model.vocab.eos(),
        "EogTokenIds": eog,
    }))
}

fn exec_model_tensor_info(req: &Value, models: &BTreeMap<u64, Arc<ModelData>>) -> Result<Value> {
    let model = get_model(models, req_u64(req, "Model")?)?;
    let names = req["Names"]
        .as_array()
        .ok_or_else(|| invalid("model-tensor-info Names must be array"))?;
    let mut rows = Vec::with_capacity(names.len());
    for item in names {
        let name = item
            .as_str()
            .ok_or_else(|| invalid("tensor name must be string"))?;
        if let Some(entry) = model.tensors.get(name) {
            rows.push(json!({
                "Name": name,
                "Found": true,
                "Type": entry.kind,
                "Dimensions": entry.ne,
                "Bytes": entry.nbytes()?,
            }));
        } else {
            rows.push(json!({ "Name": name, "Found": false }));
        }
    }
    Ok(json!({ "Tensors": rows }))
}

fn exec_model_tensor_file(
    req: &Value,
    models: &BTreeMap<u64, Arc<ModelData>>,
    session_dir: &Path,
) -> Result<Value> {
    let model = get_model(models, req_u64(req, "Model")?)?;
    let name = req["Name"]
        .as_str()
        .ok_or_else(|| invalid("model-tensor-file Name must be string"))?;
    let out_name = super::params::validate_state_name(&req["Output"])?;
    let entry = model
        .tensors
        .get(name)
        .ok_or_else(|| invalid(format!("missing model tensor {name}")))?;
    let out_path = checked_session_file(
        session_dir
            .join(&out_name)
            .to_str()
            .ok_or_else(|| invalid("non-UTF-8 session path"))?,
        session_dir,
        false,
    )?;
    fs::write(&out_path, &entry.raw)?;
    Ok(json!({
        "Tensors": [{
            "Name": name,
            "Type": entry.kind,
            "DataFile": out_path,
        }]
    }))
}

fn exec_model_output_identity(
    req: &Value,
    models: &BTreeMap<u64, Arc<ModelData>>,
) -> Result<Value> {
    let a = get_model(models, req_u64(req, "Model")?)?;
    let b = get_model(models, req_u64(req, "Other")?)?;
    let eq =
        a.output_identity == b.output_identity && Arc::ptr_eq(&a.output_buffer, &b.output_buffer);
    Ok(json!({
        "TensorEqual": eq,
        "BufferEqual": eq,
    }))
}

fn exec_trace_new(req: &Value, state: &mut DsparkState) -> Result<Value> {
    check_trace_limits(&state.traces)?;
    let mut collector = TraceCollector::default();
    if let Some(names) = req["Names"].as_array() {
        for n in names {
            if let Some(s) = n.as_str() {
                collector.exact_names.insert(s.to_owned());
            }
        }
    }
    if let Some(prefixes) = req["Prefixes"].as_array() {
        for p in prefixes {
            if let Some(s) = p.as_str() {
                collector.prefixes.push(s.to_owned());
            }
        }
    }
    let id = state.alloc_id();
    state.traces.insert(id, collector);
    Ok(json!({ "Trace": id }))
}

fn exec_context_traces(req: &Value, state: &DsparkState) -> Result<Value> {
    let tid = req_u64(req, "Trace")?;
    let collector = state
        .traces
        .get(&tid)
        .ok_or_else(|| invalid("unknown Trace handle"))?;
    let rows: Vec<Value> = collector
        .values
        .iter()
        .map(|(name, vals)| {
            json!({
                "Name": name,
                "DataHex": encode_f32_hex(vals),
            })
        })
        .collect();
    Ok(json!({
        "Failed": collector.failed,
        "Values": rows,
    }))
}

fn exec_graph_allow_reuse(req: &Value) -> Value {
    let prev = &req["Previous"];
    let next = &req["Next"];
    let live_rows = req["LiveRows"].as_i64().unwrap_or(-1);
    let live_width = req["LiveWidth"].as_i64().unwrap_or(-1);
    let ok = prev["HasContext"].as_bool() == Some(true)
        && next["HasContext"].as_bool() == Some(true)
        && prev["Rows"].as_i64() == next["Rows"].as_i64()
        && prev["Width"].as_i64() == next["Width"].as_i64()
        && next["Rows"].as_i64() == Some(live_rows)
        && next["Width"].as_i64() == Some(live_width);
    json!({ "AllowReuse": ok })
}

fn exec_process_env(req: &Value, env: &mut RuntimeEnv) -> Result<Value> {
    let save = req["Save"].as_bool().unwrap_or(false);
    let saved = if save {
        let mut map = serde_json::Map::new();
        for k in req["Changes"]
            .as_object()
            .into_iter()
            .flat_map(|o| o.keys())
        {
            map.insert(k.clone(), env.vars.get(k).cloned().unwrap_or(Value::Null));
        }
        Value::Object(map)
    } else {
        Value::Null
    };
    env.apply(&req["Changes"])?;
    Ok(json!({ "Environment": saved }))
}

fn exec_process_env_restore(req: &Value, env: &mut RuntimeEnv) -> Result<Value> {
    env.apply(&req["Environment"])?;
    Ok(Value::Null)
}

pub(super) fn resolve_logit_row(rows: &[Vec<f32>], index: i64) -> Result<&[f32]> {
    let len = i64::try_from(rows.len())?;
    let pos = if index < 0 { len + index } else { index };
    let u_pos = usize::try_from(pos)
        .ok()
        .filter(|&p| p < rows.len())
        .ok_or_else(|| invalid("logit row index out of range"))?;
    Ok(&rows[u_pos])
}

pub(super) fn logits_rows_slice(ctx: &ContextState, indices: &[Value]) -> Result<Value> {
    if indices.len() > MAX_LOGIT_ROWS {
        return Err(invalid("context-logits row count exceeds bound"));
    }
    let mut rows = Vec::with_capacity(indices.len());
    for item in indices {
        let idx = item
            .as_i64()
            .ok_or_else(|| invalid("logit row index must be int"))?;
        let slice = resolve_logit_row(&ctx.output_logits, idx)?;
        rows.push(json!({ "DataHex": encode_f32_hex(slice) }));
    }
    let first_hex = rows
        .first()
        .and_then(|r| r["DataHex"].as_str())
        .unwrap_or("")
        .to_owned();
    Ok(json!({ "Rows": rows, "DataHex": first_hex }))
}
