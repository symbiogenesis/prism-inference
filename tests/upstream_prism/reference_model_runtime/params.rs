//! Clean-room parameter parsing and runtime initialization handlers.
//!
//! No `bonsai_prism_runtime`, `bonsai_gguf`, or native GGML libraries are linked or called.

use std::path::{Path, PathBuf};
use std::thread;

use serde_json::{Value, json};

use crate::{Result, invalid};

#[derive(Clone)]
pub(super) struct SessionParams {
    pub(super) seed: u32,
    pub(super) temp: f32,
    pub(super) n_predict: i32,
    pub(super) n_batch: u32,
    pub(super) n_ubatch: u32,
    pub(super) n_ctx: u32,
    pub(super) n_parallel: u32,
    pub(super) kv_unified: bool,
    pub(super) n_gpu_layers: i32,
    pub(super) fit_params: bool,
    pub(super) warmup: bool,
    pub(super) n_threads: i32,
    pub(super) prompt: String,
    pub(super) out_file: String,
    pub(super) model_path: Option<PathBuf>,
    pub(super) lora_path: Option<PathBuf>,
    pub(super) devices: Vec<u32>,
}

impl Default for SessionParams {
    fn default() -> Self {
        Self {
            seed: 0xFFFF_FFFF,
            temp: 0.8,
            n_predict: -1,
            n_batch: 2048,
            n_ubatch: 512,
            n_ctx: 512,
            n_parallel: 1,
            kv_unified: false,
            n_gpu_layers: 0,
            fit_params: true,
            warmup: false,
            n_threads: 4,
            prompt: String::new(),
            out_file: String::new(),
            model_path: None,
            lora_path: None,
            devices: Vec::new(),
        }
    }
}

pub(super) fn parse_params(operation: &Value) -> Result<SessionParams> {
    if let Some(ex) = operation.get("Example") {
        let s = ex.as_str().unwrap_or("");
        if s != "completion" && s != "export-graph-ops" {
            return Err(invalid("unsupported parameter example"));
        }
    }
    let mut params = SessionParams::default();
    let initial = operation["Initial"]
        .as_object()
        .ok_or_else(|| invalid("params Initial must be object"))?;
    if initial.len() > 8 {
        return Err(invalid("params Initial exceeds 8 fields"));
    }
    for (key, val) in initial {
        apply_initial_field(&mut params, key, val)?;
    }
    let args_arr = operation["Arguments"]
        .as_array()
        .ok_or_else(|| invalid("params Arguments must be array"))?;
    if args_arr.is_empty() || args_arr.len() > 32 {
        return Err(invalid("params Arguments count outside 1..=32"));
    }
    let mut strings = Vec::with_capacity(args_arr.len());
    for item in args_arr {
        strings.push(
            item.as_str()
                .ok_or_else(|| invalid("param argument must be string"))?
                .to_owned(),
        );
    }
    apply_cli_args(&mut params, &strings)?;
    Ok(params)
}

fn apply_initial_field(params: &mut SessionParams, key: &str, val: &Value) -> Result<()> {
    match key {
        "Predict" => {
            params.n_predict = i32::try_from(
                val.as_i64()
                    .filter(|&v| (-1..=2048).contains(&v))
                    .ok_or_else(|| invalid("Predict outside -1..=2048"))?,
            )?;
        }
        "Batch" => {
            params.n_batch = u32::try_from(
                val.as_u64()
                    .filter(|&v| (1..=2048).contains(&v))
                    .ok_or_else(|| invalid("Batch outside 1..=2048"))?,
            )?;
            params.n_ubatch = params.n_ubatch.min(params.n_batch);
        }
        "Context" => {
            params.n_ctx = u32::try_from(
                val.as_u64()
                    .filter(|&v| v <= 4096)
                    .ok_or_else(|| invalid("Context outside 0..=4096"))?,
            )?;
        }
        "Parallel" => {
            params.n_parallel = u32::try_from(
                val.as_u64()
                    .filter(|&v| (1..=16).contains(&v))
                    .ok_or_else(|| invalid("Parallel outside 1..=16"))?,
            )?;
        }
        "Seed" => {
            params.seed = u32::try_from(
                val.as_u64()
                    .ok_or_else(|| invalid("Seed must be unsigned u32"))?,
            )?;
        }
        "Unified" => {
            params.kv_unified = val
                .as_bool()
                .ok_or_else(|| invalid("Unified must be boolean"))?;
        }
        "Prompt" => {
            params.prompt = val
                .as_str()
                .ok_or_else(|| invalid("Prompt must be string"))?
                .to_owned();
        }
        "OutputFile" => {
            let name = val
                .as_str()
                .ok_or_else(|| invalid("OutputFile must be string"))?;
            if name.is_empty()
                || name.contains("..")
                || !name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
            {
                return Err(invalid("OutputFile must be an owned plain filename"));
            }
            params.out_file = name.to_owned();
        }
        _ => return Err(invalid(format!("unknown initial field {key}"))),
    }
    Ok(())
}

fn apply_cli_args(params: &mut SessionParams, args: &[String]) -> Result<()> {
    let mut i = 1_usize;
    while i < args.len() {
        let flag = args[i].as_str();
        if flag == "-no-cnv" {
            i += 1;
            continue;
        }
        if i + 1 >= args.len() {
            return Err(invalid("argument flag missing value"));
        }
        let val = &args[i + 1];
        match flag {
            "-m" => {
                let p = PathBuf::from(val);
                if !p.is_absolute() {
                    return Err(invalid("-m model path must be absolute"));
                }
                params.model_path = Some(p);
            }
            "--lora" => {
                let p = PathBuf::from(val);
                if !p.is_absolute() {
                    return Err(invalid("--lora path must be absolute"));
                }
                params.lora_path = Some(p);
            }
            "-ngl" => params.n_gpu_layers = val.parse::<i32>()?,
            "-p" => params.prompt = val.clone(),
            "-n" => params.n_predict = val.parse::<i32>()?,
            "-c" => params.n_ctx = val.parse::<u32>()?,
            "-ub" => params.n_ubatch = val.parse::<u32>()?,
            "-np" => params.n_parallel = val.parse::<u32>()?,
            "-t" => params.n_threads = val.parse::<i32>()?,
            "--seed" => params.seed = val.parse::<u32>()?,
            "--temp" => params.temp = val.parse::<f32>()?,
            _ => return Err(invalid(format!("unsupported common argument {flag}"))),
        }
        i += 2;
    }
    Ok(())
}

pub(super) fn update_params(params: &mut SessionParams, fields: &Value) -> Result<()> {
    let obj = fields
        .as_object()
        .ok_or_else(|| invalid("params-update Fields must be object"))?;
    if obj.len() > 4 {
        return Err(invalid("params-update field count exceeds 4"));
    }
    for (key, val) in obj {
        match key.as_str() {
            "Devices" => {
                let arr = val
                    .as_array()
                    .ok_or_else(|| invalid("Devices must be array"))?;
                let mut devs = Vec::with_capacity(arr.len());
                for item in arr {
                    let idx = u32::try_from(
                        item.as_u64()
                            .filter(|&n| n == 0)
                            .ok_or_else(|| invalid("device index outside CPU inventory"))?,
                    )?;
                    devs.push(idx);
                }
                params.devices = devs;
            }
            "GpuLayers" => {
                params.n_gpu_layers = i32::try_from(
                    val.as_i64()
                        .filter(|&n| (-1..=999).contains(&n))
                        .ok_or_else(|| invalid("GpuLayers outside -1..=999"))?,
                )?;
            }
            "FitParams" => {
                params.fit_params = val
                    .as_bool()
                    .ok_or_else(|| invalid("FitParams must be boolean"))?;
            }
            "Warmup" => {
                params.warmup = val
                    .as_bool()
                    .ok_or_else(|| invalid("Warmup must be boolean"))?;
            }
            _ => return Err(invalid(format!("unknown params-update field {key}"))),
        }
    }
    Ok(())
}

pub(super) fn hardware_concurrency() -> Value {
    let count = thread::available_parallelism().map_or(4, std::num::NonZeroUsize::get);
    json!({ "Count": count })
}

pub(super) fn validate_state_name(raw: &Value) -> Result<String> {
    let s = raw
        .as_str()
        .ok_or_else(|| invalid("state filename must be string"))?;
    if s.is_empty()
        || s.len() > 64
        || s.contains("..")
        || !s
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
    {
        return Err(invalid("state file must be an owned plain filename"));
    }
    Ok(s.to_owned())
}

pub(super) fn checked_model_path(raw: &Value) -> Result<PathBuf> {
    let s = raw
        .as_str()
        .ok_or_else(|| invalid("model path must be string"))?;
    if s.len() > 32768 || s.contains('\0') {
        return Err(invalid("model path string violation"));
    }
    let p = Path::new(s);
    if !p.is_absolute() {
        return Err(invalid("model path must be absolute"));
    }
    Ok(p.to_path_buf())
}
