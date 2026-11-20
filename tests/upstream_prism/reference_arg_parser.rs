//! Clean-room independent safe Rust CLI argument parser, speculative limits, and HTTP/1.1 reference provider.
//!
//! No production `bonsai_prism_runtime` code, native bridge or precomputed
//! answer fixture is called here. Implements argument-option inventory,
//! stateful CLI parsing with RFC-4180/unquoted CSV `--lora` splitting,
//! speculative output-limit arithmetic and a bounded plain-HTTP/1.1 client.

use std::io::{ErrorKind, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

use serde_json::{Map, Value, json};

use super::{Result, invalid};

pub(super) const ID: &str = "tests/test-arg-parser.cpp";
const SOURCE_SHA: &str = "935b2cd08b63799a26fb804991380c541a45406c76dceb38589d78c8b28befc1";
const EXAMPLE_COUNT: i32 = 27;
const COMMON: i32 = 2;
const SPECULATIVE: i32 = 3;
const MAX_OPERATIONS: usize = 128;
const MAX_SLOTS: usize = 16;
const MAX_ACTIONS: usize = 256;
const MAX_ARGV_COUNT: usize = 64;
const MAX_ARGV_ITEM_BYTES: usize = 16_384;
const MAX_ARGV_TOTAL_BYTES: usize = 65_536;
const MAX_URL_BYTES: usize = 4096;
const MAX_HEADERS: usize = 32;
const MAX_HEADER_FIELD_BYTES: usize = 4096;
const MAX_TIMEOUT_SECONDS: i32 = 60;
const MAX_REMOTE_SIZE: usize = 8 * 1024 * 1024;
const MAX_REMOTE_SIZE_I32: i32 = 8 * 1024 * 1024;
const MAX_REDIRECTS: usize = 5;
const MAX_HTTP_HEADER_BYTES: usize = 64 * 1024;
const MAX_HTTP_LINE_BYTES: usize = 8192;
const MAX_ERROR_BYTES: usize = 4096;
const DEFAULT_TIMEOUT_SECONDS: u64 = 15;

#[derive(Clone)]
struct Params {
    model_path: String,
    threads: i32,
    verbosity: i32,
    predict: i32,
    batch: i32,
    draft_max: i32,
    load_mode: i32,
    penalty_last: i32,
    dry_penalty_last: i32,
    lora_paths: Vec<String>,
    outputs_max: i32,
    outputs_max_per_seq: i32,
    parallel: i32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            threads: -1,
            verbosity: 0,
            predict: -1,
            batch: 2048,
            draft_max: 16,
            load_mode: 1,
            penalty_last: 64,
            dry_penalty_last: 64,
            lora_paths: Vec::new(),
            outputs_max: 1,
            outputs_max_per_seq: 1,
            parallel: 1,
        }
    }
}

impl Params {
    fn snapshot(&self) -> Value {
        json!({
            "ModelPath": self.model_path,
            "Threads": self.threads,
            "Verbosity": self.verbosity,
            "Predict": self.predict,
            "Batch": self.batch,
            "DraftMax": self.draft_max,
            "LoadMode": self.load_mode,
            "PenaltyLast": self.penalty_last,
            "DryPenaltyLast": self.dry_penalty_last,
            "LoraPaths": self.lora_paths,
            "OutputsMax": self.outputs_max,
            "OutputsMaxPerSequence": self.outputs_max_per_seq,
        })
    }

    fn to_speculative(&self) -> Self {
        let mut copy = self.clone();
        copy.outputs_max_per_seq = 1;
        copy.outputs_max = copy.parallel;
        copy
    }
}

#[derive(Clone, Copy)]
enum OptionId {
    Help,
    Verbose,
    Threads,
    Model,
    Predict,
    Batch,
    GpuLayers,
    SplitMode,
    LoadMode,
    KvOffload,
    RepeatLastN,
    DryPenaltyLastN,
    RepeatPenalty,
    FrequencyPenalty,
    PresencePenalty,
    Lora,
    DraftMax,
}

#[derive(Clone, Copy)]
enum Scope {
    All,
    CommonAndSpeculative,
}

impl Scope {
    const fn includes(self, example: i32) -> bool {
        match self {
            Self::All => true,
            Self::CommonAndSpeculative => example == COMMON || example == SPECULATIVE,
        }
    }
}

#[derive(Clone, Copy)]
struct OptionSpec {
    id: OptionId,
    args: &'static [&'static str],
    negated_args: &'static [&'static str],
    environment: &'static [&'static str],
    speculative: bool,
    scope: Scope,
}

const OPTIONS: &[OptionSpec] = &[
    OptionSpec {
        id: OptionId::Help,
        args: &["-h", "--help"],
        negated_args: &[],
        environment: &[],
        speculative: false,
        scope: Scope::All,
    },
    OptionSpec {
        id: OptionId::Verbose,
        args: &["-v", "--verbose"],
        negated_args: &[],
        environment: &[],
        speculative: false,
        scope: Scope::All,
    },
    OptionSpec {
        id: OptionId::Threads,
        args: &["-t", "--threads"],
        negated_args: &[],
        environment: &["LLAMA_ARG_THREADS"],
        speculative: false,
        scope: Scope::All,
    },
    OptionSpec {
        id: OptionId::Model,
        args: &["-m", "--model"],
        negated_args: &[],
        environment: &["LLAMA_ARG_MODEL"],
        speculative: false,
        scope: Scope::All,
    },
    OptionSpec {
        id: OptionId::Predict,
        args: &["-n", "--predict"],
        negated_args: &[],
        environment: &["LLAMA_ARG_N_PREDICT"],
        speculative: false,
        scope: Scope::All,
    },
    OptionSpec {
        id: OptionId::Batch,
        args: &["-b", "--batch-size"],
        negated_args: &[],
        environment: &["LLAMA_ARG_BATCH"],
        speculative: false,
        scope: Scope::All,
    },
    OptionSpec {
        id: OptionId::GpuLayers,
        args: &["-ngl", "--gpu-layers", "--n-gpu-layers"],
        negated_args: &[],
        environment: &["LLAMA_ARG_N_GPU_LAYERS"],
        speculative: false,
        scope: Scope::All,
    },
    OptionSpec {
        id: OptionId::SplitMode,
        args: &["-sm", "--split-mode"],
        negated_args: &[],
        environment: &["LLAMA_ARG_SPLIT_MODE"],
        speculative: false,
        scope: Scope::All,
    },
    OptionSpec {
        id: OptionId::LoadMode,
        args: &["-lm", "--load-mode"],
        negated_args: &[],
        environment: &["LLAMA_ARG_LOAD_MODE"],
        speculative: false,
        scope: Scope::All,
    },
    OptionSpec {
        id: OptionId::KvOffload,
        args: &["-kvo", "--kv-offload"],
        negated_args: &["-nkvo", "--no-kv-offload"],
        environment: &["LLAMA_ARG_KV_OFFLOAD"],
        speculative: false,
        scope: Scope::All,
    },
    OptionSpec {
        id: OptionId::RepeatLastN,
        args: &["--repeat-last-n"],
        negated_args: &[],
        environment: &["LLAMA_ARG_REPEAT_LAST_N"],
        speculative: false,
        scope: Scope::All,
    },
    OptionSpec {
        id: OptionId::DryPenaltyLastN,
        args: &["--dry-penalty-last-n"],
        negated_args: &[],
        environment: &["LLAMA_ARG_DRY_PENALTY_LAST_N"],
        speculative: false,
        scope: Scope::All,
    },
    OptionSpec {
        id: OptionId::RepeatPenalty,
        args: &["--repeat-penalty"],
        negated_args: &[],
        environment: &["LLAMA_ARG_REPEAT_PENALTY"],
        speculative: false,
        scope: Scope::All,
    },
    OptionSpec {
        id: OptionId::FrequencyPenalty,
        args: &["--frequency-penalty"],
        negated_args: &[],
        environment: &["LLAMA_ARG_FREQUENCY_PENALTY"],
        speculative: false,
        scope: Scope::All,
    },
    OptionSpec {
        id: OptionId::PresencePenalty,
        args: &["--presence-penalty"],
        negated_args: &[],
        environment: &["LLAMA_ARG_PRESENCE_PENALTY"],
        speculative: false,
        scope: Scope::All,
    },
    OptionSpec {
        id: OptionId::Lora,
        args: &["--lora"],
        negated_args: &[],
        environment: &["LLAMA_ARG_LORA"],
        speculative: false,
        scope: Scope::All,
    },
    OptionSpec {
        id: OptionId::DraftMax,
        args: &["--draft", "--draft-max", "--spec-draft-n-max"],
        negated_args: &[],
        environment: &["LLAMA_ARG_DRAFT_MAX"],
        speculative: true,
        scope: Scope::CommonAndSpeculative,
    },
];

fn fields<'a>(value: &'a Value, names: &[&str]) -> Result<&'a Map<String, Value>> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid("argument observation object required"))?;
    if object.len() != names.len() || names.iter().any(|name| !object.contains_key(*name)) {
        return Err(invalid("argument observation fields mismatch"));
    }
    Ok(object)
}

fn array(value: &Value) -> Result<&[Value]> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| invalid("argument observation array required"))
}

fn integer(value: &Value, low: i32, high: i32) -> Result<i32> {
    let number = value
        .as_i64()
        .ok_or_else(|| invalid("argument observation integer required"))?;
    if number < i64::from(low) || number > i64::from(high) {
        return Err(invalid("argument observation integer outside bounds"));
    }
    Ok(i32::try_from(number)?)
}

fn text(value: &Value, maximum: usize) -> Result<&str> {
    let text = value
        .as_str()
        .ok_or_else(|| invalid("argument observation string required"))?;
    if text.len() > maximum || text.contains('\0') {
        return Err(invalid("argument observation string extent/NUL violation"));
    }
    Ok(text)
}

fn slot(value: &Value, count: usize) -> Result<usize> {
    let index = usize::try_from(integer(value, 0, 15)?)?;
    if index >= count {
        return Err(invalid("argument params slot outside inventory"));
    }
    Ok(index)
}

fn truncate_utf8(message: &str, max_bytes: usize) -> &str {
    if message.len() <= max_bytes {
        return message;
    }
    let mut end = max_bytes;
    while end > 0 && !message.is_char_boundary(end) {
        end -= 1;
    }
    &message[..end]
}

fn exception_object(message: &str) -> Value {
    json!({
        "Kind": "standard",
        "Message": truncate_utf8(message, MAX_ERROR_BYTES),
    })
}

fn validate_remote_operation(operation: &Value) -> Result<()> {
    fields(
        operation,
        &["Kind", "Url", "Headers", "TimeoutSeconds", "MaxSize"],
    )?;
    let url = text(&operation["Url"], MAX_URL_BYTES)?;
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Err(invalid("remote content URL must be HTTP(S)"));
    }
    integer(&operation["TimeoutSeconds"], 0, MAX_TIMEOUT_SECONDS)?;
    integer(&operation["MaxSize"], 0, MAX_REMOTE_SIZE_I32)?;
    let headers = array(&operation["Headers"])?;
    if headers.len() > MAX_HEADERS {
        return Err(invalid("remote header count exceeds 32"));
    }
    for header in headers {
        fields(header, &["Name", "Value"])?;
        for name in ["Name", "Value"] {
            let value = text(&header[name], MAX_HEADER_FIELD_BYTES)?;
            if value.contains(['\r', '\n']) {
                return Err(invalid("remote header newline violation"));
            }
        }
    }
    Ok(())
}

fn validate_trace_action(action: &Value, slot_count: usize) -> Result<()> {
    let object = action
        .as_object()
        .ok_or_else(|| invalid("argument action must be an object"))?;
    let action_name = object
        .get("Action")
        .ok_or_else(|| invalid("argument action required"))?;
    match text(action_name, 64)? {
        "limits" => {
            fields(action, &["Action", "Batch", "Parallel", "Draft"])?;
            integer(&action["Batch"], 1, i32::MAX)?;
            integer(&action["Parallel"], 1, i32::MAX)?;
            integer(&action["Draft"], -1, i32::MAX)?;
        }
        "snapshot" | "speculative" => {
            fields(action, &["Action", "Slot"])?;
            slot(&action["Slot"], slot_count)?;
        }
        "inventory" => {
            fields(action, &["Action", "Slot", "Example"])?;
            slot(&action["Slot"], slot_count)?;
            integer(&action["Example"], 0, EXAMPLE_COUNT - 1)?;
        }
        "parse" => {
            fields(action, &["Action", "Slot", "Example", "Arguments"])?;
            slot(&action["Slot"], slot_count)?;
            integer(&action["Example"], 0, EXAMPLE_COUNT - 1)?;
            let arguments = array(&action["Arguments"])?;
            if arguments.is_empty() || arguments.len() > MAX_ARGV_COUNT {
                return Err(invalid("argument argv extent outside bounds"));
            }
            let mut total = 0_usize;
            for argument in arguments {
                total += text(argument, MAX_ARGV_ITEM_BYTES)?.len();
                if total > MAX_ARGV_TOTAL_BYTES {
                    return Err(invalid("argument argv bytes exceed 64KiB"));
                }
            }
        }
        _ => return Err(invalid("unknown argument observation action")),
    }
    Ok(())
}

fn validate_trace_operation(operation: &Value) -> Result<()> {
    fields(operation, &["Kind", "Initials", "Actions"])?;
    let initials = array(&operation["Initials"])?;
    if initials.is_empty() || initials.len() > MAX_SLOTS {
        return Err(invalid("argument params slots exceed bound"));
    }
    for initial in initials {
        let object = initial
            .as_object()
            .ok_or_else(|| invalid("argument initial must be an object"))?;
        if object.len() > 2 {
            return Err(invalid("argument initial fields exceed bound"));
        }
        for (key, value) in object {
            if key != "Parallel" && key != "OutputsMaxPerSequence" {
                return Err(invalid("unknown argument initial field"));
            }
            integer(value, 1, i32::MAX)?;
        }
    }
    let actions = array(&operation["Actions"])?;
    if actions.len() > MAX_ACTIONS {
        return Err(invalid("argument actions exceed 256"));
    }
    for action in actions {
        validate_trace_action(action, initials.len())?;
    }
    Ok(())
}

fn validate_operation(operation: &Value) -> Result<()> {
    let object = operation
        .as_object()
        .ok_or_else(|| invalid("argument observation operation must be an object"))?;
    let kind_value = object
        .get("Kind")
        .ok_or_else(|| invalid("argument observation kind missing"))?;
    match text(kind_value, 64)? {
        "argument-enums" => {
            fields(operation, &["Kind"])?;
            Ok(())
        }
        "remote-get-content" => validate_remote_operation(operation),
        "argument-parser-trace" => validate_trace_operation(operation),
        _ => Err(invalid("unknown argument observation kind")),
    }
}

fn speculative_limits(batch: i32, parallel: i32, draft: i32) -> (i64, i64) {
    let draft_nonneg = i64::from(draft.max(0));
    let batch_i64 = i64::from(batch);
    let per_seq = (1_i64 + draft_nonneg).min(batch_i64);
    let total = i64::from(parallel).saturating_mul(per_seq).min(batch_i64);
    (total, per_seq)
}

fn inventory_for_example(example: i32) -> Value {
    let options = OPTIONS
        .iter()
        .filter(|option| option.scope.includes(example))
        .map(|option| {
            let mut all_args = Vec::with_capacity(option.args.len() + option.negated_args.len());
            all_args.extend_from_slice(option.args);
            all_args.extend_from_slice(option.negated_args);
            json!({
                "Args": option.args,
                "NegatedArgs": option.negated_args,
                "AllArgs": all_args,
                "Environment": option.environment,
                "Speculative": option.speculative,
            })
        })
        .collect::<Vec<_>>();
    json!({"Options": options})
}

fn parse_csv(input: &str) -> Vec<String> {
    if input.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut at_field_start = true;
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        if in_quotes {
            if ch == '"' {
                if chars.peek() == Some(&'"') {
                    let _ = chars.next();
                    current.push('"');
                } else {
                    in_quotes = false;
                }
            } else {
                current.push(ch);
            }
        } else if at_field_start && ch == '"' {
            in_quotes = true;
            at_field_start = false;
        } else if ch == ',' {
            out.push(std::mem::take(&mut current));
            at_field_start = true;
        } else {
            current.push(ch);
            at_field_start = false;
        }
    }
    out.push(current);
    out
}

fn next_i32<'a, I: Iterator<Item = &'a str>>(iter: &mut I) -> Option<i32> {
    iter.next()?.parse::<i32>().ok()
}

fn next_nonneg_i32<'a, I: Iterator<Item = &'a str>>(iter: &mut I) -> Option<i32> {
    next_i32(iter).filter(|&number| number >= 0)
}

fn next_finite_f32<'a, I: Iterator<Item = &'a str>>(iter: &mut I) -> Option<f32> {
    iter.next()?.parse::<f32>().ok().filter(|n| n.is_finite())
}

fn apply_option<'a, I>(candidate: &mut Params, id: OptionId, iter: &mut I) -> bool
where
    I: Iterator<Item = &'a str>,
{
    match id {
        OptionId::Help => return false,
        OptionId::Verbose => candidate.verbosity = i32::MAX,
        OptionId::KvOffload => {}
        OptionId::Model => {
            let Some(value) = iter.next() else {
                return false;
            };
            value.clone_into(&mut candidate.model_path);
        }
        OptionId::Threads => {
            let Some(number) = next_i32(iter) else {
                return false;
            };
            candidate.threads = number;
        }
        OptionId::Predict => {
            let Some(number) = next_i32(iter) else {
                return false;
            };
            candidate.predict = number;
        }
        OptionId::Batch => {
            let Some(number) = next_i32(iter) else {
                return false;
            };
            candidate.batch = number;
        }
        OptionId::DraftMax => {
            let Some(number) = next_i32(iter) else {
                return false;
            };
            candidate.draft_max = number;
        }
        OptionId::GpuLayers => {
            let Some(value) = iter.next() else {
                return false;
            };
            if !value.eq_ignore_ascii_case("auto")
                && !value.eq_ignore_ascii_case("all")
                && !value.eq_ignore_ascii_case("none")
                && value.parse::<i32>().is_err()
            {
                return false;
            }
        }
        OptionId::SplitMode => {
            if !matches!(iter.next(), Some("none" | "layer" | "row")) {
                return false;
            }
        }
        OptionId::LoadMode => {
            let Some(mode) = iter.next().and_then(|value| match value {
                "none" => Some(0),
                "mmap" => Some(1),
                "mlock" => Some(2),
                "mmap+mlock" => Some(3),
                "dio" => Some(4),
                _ => None,
            }) else {
                return false;
            };
            candidate.load_mode = mode;
        }
        OptionId::RepeatLastN => {
            let Some(number) = next_nonneg_i32(iter) else {
                return false;
            };
            candidate.penalty_last = number;
        }
        OptionId::DryPenaltyLastN => {
            let Some(number) = next_nonneg_i32(iter) else {
                return false;
            };
            candidate.dry_penalty_last = number;
        }
        OptionId::RepeatPenalty => {
            if next_finite_f32(iter).is_none_or(|number| number <= 0.0) {
                return false;
            }
        }
        OptionId::FrequencyPenalty | OptionId::PresencePenalty => {
            if next_finite_f32(iter).is_none() {
                return false;
            }
        }
        OptionId::Lora => {
            let Some(value) = iter.next() else {
                return false;
            };
            candidate.lora_paths = parse_csv(value);
        }
    }
    true
}

fn parse_arguments(params: &mut Params, example: i32, arguments: &[&str]) -> bool {
    let mut candidate = params.clone();
    let mut iter = arguments.iter().copied().skip(1);
    while let Some(flag) = iter.next() {
        let Some(spec) = OPTIONS.iter().find(|option| {
            option.scope.includes(example)
                && (option.args.contains(&flag) || option.negated_args.contains(&flag))
        }) else {
            return false;
        };
        if !apply_option(&mut candidate, spec.id, &mut iter) {
            return false;
        }
    }
    *params = candidate;
    true
}

fn execute_trace_operation(operation: &Value) -> Result<Value> {
    let initials = array(&operation["Initials"])?;
    let mut slots = Vec::with_capacity(initials.len());
    for initial in initials {
        let mut params = Params::default();
        if let Some(parallel) = initial.get("Parallel") {
            params.parallel = integer(parallel, 1, i32::MAX)?;
        }
        if let Some(outputs) = initial.get("OutputsMaxPerSequence") {
            params.outputs_max_per_seq = integer(outputs, 1, i32::MAX)?;
        }
        slots.push(params);
    }
    let actions = array(&operation["Actions"])?;
    let mut trace = Vec::with_capacity(actions.len());
    for action in actions {
        let action_name = text(&action["Action"], 64)?;
        if action_name == "limits" {
            let batch = integer(&action["Batch"], 1, i32::MAX)?;
            let parallel = integer(&action["Parallel"], 1, i32::MAX)?;
            let draft = integer(&action["Draft"], -1, i32::MAX)?;
            let (total, per_seq) = speculative_limits(batch, parallel, draft);
            trace.push(json!({"Total": total, "PerSequence": per_seq}));
            continue;
        }
        let slot_index = slot(&action["Slot"], slots.len())?;
        let params = &mut slots[slot_index];
        match action_name {
            "snapshot" => trace.push(json!({"Params": params.snapshot()})),
            "speculative" => trace.push(json!({"Params": params.to_speculative().snapshot()})),
            "inventory" => {
                let example = integer(&action["Example"], 0, EXAMPLE_COUNT - 1)?;
                trace.push(inventory_for_example(example));
            }
            "parse" => {
                let example = integer(&action["Example"], 0, EXAMPLE_COUNT - 1)?;
                let arguments = array(&action["Arguments"])?
                    .iter()
                    .map(|value| text(value, MAX_ARGV_ITEM_BYTES))
                    .collect::<Result<Vec<_>>>()?;
                let parsed = parse_arguments(params, example, &arguments);
                trace.push(json!({"Parsed": parsed, "Params": params.snapshot()}));
            }
            _ => return Err(invalid("unknown argument observation action")),
        }
    }
    Ok(json!({"Trace": trace}))
}

struct HeaderEntry<'a> {
    name: &'a str,
    value: &'a str,
}

struct ParsedUrl {
    connect_host: String,
    host_header: String,
    port: u16,
    target: String,
}

fn parse_port(text: &str) -> std::result::Result<u16, String> {
    let port = text
        .parse::<u16>()
        .map_err(|_| "invalid port in HTTP URL".to_owned())?;
    if port == 0 {
        return Err("HTTP URL port must be nonzero".to_owned());
    }
    Ok(port)
}

fn parse_http_url(url: &str) -> std::result::Result<ParsedUrl, String> {
    if url.starts_with("https://") {
        return Err("HTTPS URLs require TLS and are unsupported by plain HTTP client".to_owned());
    }
    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| "remote content URL must start with http://".to_owned())?;
    let without_fragment = rest.split_once('#').map_or(rest, |(before, _)| before);
    let (authority, target) = match without_fragment.find(['/', '?']) {
        Some(index) if without_fragment.as_bytes()[index] == b'/' => (
            &without_fragment[..index],
            without_fragment[index..].to_owned(),
        ),
        Some(index) => (
            &without_fragment[..index],
            format!("/{}", &without_fragment[index..]),
        ),
        None => (without_fragment, "/".to_owned()),
    };
    if authority.is_empty()
        || authority.contains('@')
        || authority.chars().any(|ch| ch.is_control() || ch == ' ')
        || target.chars().any(|ch| ch.is_control() || ch == ' ')
    {
        return Err("malformed HTTP URL".to_owned());
    }
    let (connect_host, host_header, port) = if let Some(after_open) = authority.strip_prefix('[') {
        let (inside, after) = after_open
            .split_once(']')
            .ok_or_else(|| "invalid IPv6 host in HTTP URL".to_owned())?;
        if inside.is_empty() {
            return Err("empty IPv6 host in HTTP URL".to_owned());
        }
        let port = if after.is_empty() {
            80
        } else if let Some(port_text) = after.strip_prefix(':') {
            parse_port(port_text)?
        } else {
            return Err("invalid port after IPv6 host in HTTP URL".to_owned());
        };
        let header = if port == 80 {
            format!("[{inside}]")
        } else {
            format!("[{inside}]:{port}")
        };
        (inside.to_owned(), header, port)
    } else if let Some((host, port_text)) = authority.split_once(':') {
        if host.is_empty() {
            return Err("empty host in HTTP URL".to_owned());
        }
        let port = parse_port(port_text)?;
        let header = if port == 80 {
            host.to_owned()
        } else {
            format!("{host}:{port}")
        };
        (host.to_owned(), header, port)
    } else {
        (authority.to_owned(), authority.to_owned(), 80)
    };
    Ok(ParsedUrl {
        connect_host,
        host_header,
        port,
        target,
    })
}

fn resolve_redirect(base: &ParsedUrl, location: &str) -> std::result::Result<String, String> {
    let location = location.trim();
    if location.is_empty() || location.chars().any(char::is_control) {
        return Err("invalid HTTP redirect Location header".to_owned());
    }
    if location.starts_with("http://") || location.starts_with("https://") {
        return Ok(location.to_owned());
    }
    if location.starts_with("//") {
        return Ok(format!("http:{location}"));
    }
    if location.starts_with('/') {
        return Ok(format!("http://{}{location}", base.host_header));
    }
    let path_only = base
        .target
        .split_once('?')
        .map_or(base.target.as_str(), |(path, _)| path);
    let prefix = path_only.rfind('/').map_or("/", |idx| &path_only[..=idx]);
    Ok(format!("http://{}{prefix}{location}", base.host_header))
}

const fn exceeded_size_message(max_size: usize) -> &'static str {
    if max_size > 0 {
        "remote content exceeds maximum size"
    } else {
        "remote observation body exceeds 8MiB"
    }
}

fn find_header_end(data: &[u8]) -> Option<(usize, usize)> {
    let mut index = 0_usize;
    while index < data.len() {
        if data[index..].starts_with(b"\r\n\r\n") {
            return Some((index, 4));
        }
        if data[index..].starts_with(b"\n\n") {
            return Some((index, 2));
        }
        index += 1;
    }
    None
}

fn read_stream_chunk(stream: &mut TcpStream, buf: &mut [u8]) -> std::io::Result<usize> {
    loop {
        match stream.read(buf) {
            Ok(count) => return Ok(count),
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
}

struct BufferedReader<'a> {
    stream: &'a mut TcpStream,
    buf: Vec<u8>,
    pos: usize,
}

impl<'a> BufferedReader<'a> {
    const fn new(stream: &'a mut TcpStream) -> Self {
        Self {
            stream,
            buf: Vec::new(),
            pos: 0,
        }
    }

    fn read_header_block(&mut self) -> std::result::Result<Vec<u8>, String> {
        if self.pos > 0 {
            self.buf.drain(..self.pos);
            self.pos = 0;
        }
        loop {
            if let Some((end, sep_len)) = find_header_end(&self.buf) {
                if end > MAX_HTTP_HEADER_BYTES {
                    return Err("HTTP response headers exceed 64 KiB".to_owned());
                }
                let block = self.buf[..end].to_vec();
                self.pos = end + sep_len;
                return Ok(block);
            }
            if self.buf.len() >= MAX_HTTP_HEADER_BYTES {
                return Err("HTTP response headers exceed 64 KiB".to_owned());
            }
            let mut chunk = [0_u8; 4096];
            let count = read_stream_chunk(self.stream, &mut chunk)
                .map_err(|error| format!("HTTP header read failed: {error}"))?;
            if count == 0 {
                return Err("unexpected EOF reading HTTP response headers".to_owned());
            }
            self.buf.extend_from_slice(&chunk[..count]);
        }
    }

    fn read_line_crlf(&mut self) -> std::result::Result<String, String> {
        loop {
            if let Some(rel_end) = self.buf[self.pos..].iter().position(|&byte| byte == b'\n') {
                let line_end = self.pos + rel_end;
                let raw = &self.buf[self.pos..line_end];
                let trimmed = raw.strip_suffix(b"\r").unwrap_or(raw);
                if trimmed.len() > MAX_HTTP_LINE_BYTES {
                    return Err("HTTP chunk line exceeds 8 KiB".to_owned());
                }
                let line = std::str::from_utf8(trimmed)
                    .map_err(|_| "non-UTF-8 HTTP chunk line".to_owned())?
                    .to_owned();
                self.pos = line_end + 1;
                return Ok(line);
            }
            if self.buf.len() - self.pos > MAX_HTTP_LINE_BYTES {
                return Err("HTTP chunk line exceeds 8 KiB".to_owned());
            }
            if self.pos > 0 {
                self.buf.drain(..self.pos);
                self.pos = 0;
            }
            let mut chunk = [0_u8; 4096];
            let count = read_stream_chunk(self.stream, &mut chunk)
                .map_err(|error| format!("HTTP chunk read failed: {error}"))?;
            if count == 0 {
                return Err("unexpected EOF reading HTTP chunked response".to_owned());
            }
            self.buf.extend_from_slice(&chunk[..count]);
        }
    }
}

impl Read for BufferedReader<'_> {
    fn read(&mut self, dst: &mut [u8]) -> std::io::Result<usize> {
        if self.pos < self.buf.len() {
            let available = &self.buf[self.pos..];
            let count = dst.len().min(available.len());
            dst[..count].copy_from_slice(&available[..count]);
            self.pos += count;
            if self.pos == self.buf.len() {
                self.buf.clear();
                self.pos = 0;
            }
            return Ok(count);
        }
        read_stream_chunk(self.stream, dst)
    }
}

struct ResponseHead {
    status_code: u16,
    content_length: Option<usize>,
    is_chunked: bool,
    location: Option<String>,
}

fn parse_response_headers(block: &[u8]) -> std::result::Result<ResponseHead, String> {
    let lossy = String::from_utf8_lossy(block);
    let mut lines = lossy
        .split('\n')
        .map(|line| line.strip_suffix('\r').unwrap_or(line));
    let status_line = lines
        .next()
        .ok_or_else(|| "missing HTTP status line".to_owned())?;
    let rest = status_line
        .strip_prefix("HTTP/1.1 ")
        .or_else(|| status_line.strip_prefix("HTTP/1.0 "))
        .ok_or_else(|| format!("unsupported HTTP status line: {status_line}"))?;
    let code_str = rest
        .split_whitespace()
        .next()
        .ok_or_else(|| "missing HTTP status code".to_owned())?;
    if code_str.len() != 3 || !code_str.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(format!("invalid HTTP status code: {code_str}"));
    }
    let status_code = code_str
        .parse::<u16>()
        .map_err(|_| "invalid HTTP status code".to_owned())?;
    let mut content_length: Option<usize> = None;
    let mut is_chunked = false;
    let mut location: Option<String> = None;
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let Some((name, value)) = line.split_once(':') else {
            return Err("malformed HTTP response header".to_owned());
        };
        let name = name.trim();
        let value = value.trim();
        if name.eq_ignore_ascii_case("content-length") {
            let parsed = value
                .parse::<usize>()
                .map_err(|_| "invalid HTTP Content-Length header".to_owned())?;
            if content_length.is_some_and(|prev| prev != parsed) {
                return Err("conflicting HTTP Content-Length headers".to_owned());
            }
            content_length = Some(parsed);
        } else if name.eq_ignore_ascii_case("transfer-encoding") {
            if value
                .split(',')
                .any(|token| token.trim().eq_ignore_ascii_case("chunked"))
            {
                is_chunked = true;
            }
        } else if name.eq_ignore_ascii_case("location") && location.is_none() {
            location = Some(value.to_owned());
        }
    }
    Ok(ResponseHead {
        status_code,
        content_length,
        is_chunked,
        location,
    })
}

fn read_chunked_body(
    reader: &mut BufferedReader<'_>,
    max_allowed: usize,
    max_size: usize,
) -> std::result::Result<Vec<u8>, String> {
    let mut body = Vec::new();
    loop {
        let size_line = reader.read_line_crlf()?;
        let hex_part = size_line
            .split_once(';')
            .map_or(size_line.as_str(), |(size, _)| size)
            .trim();
        let chunk_size = usize::from_str_radix(hex_part, 16)
            .map_err(|_| "invalid HTTP chunk size".to_owned())?;
        if chunk_size == 0 {
            loop {
                if reader.read_line_crlf()?.is_empty() {
                    break;
                }
            }
            return Ok(body);
        }
        let next_len = body
            .len()
            .checked_add(chunk_size)
            .ok_or_else(|| exceeded_size_message(max_size).to_owned())?;
        if next_len > max_allowed {
            return Err(exceeded_size_message(max_size).to_owned());
        }
        let start = body.len();
        body.resize(next_len, 0);
        reader
            .read_exact(&mut body[start..])
            .map_err(|error| format!("HTTP chunk payload read failed: {error}"))?;
        if !reader.read_line_crlf()?.is_empty() {
            return Err("invalid HTTP chunk terminator".to_owned());
        }
    }
}

fn read_exact_bounded(
    reader: &mut BufferedReader<'_>,
    content_length: usize,
    max_allowed: usize,
    max_size: usize,
) -> std::result::Result<Vec<u8>, String> {
    if content_length > max_allowed {
        return Err(exceeded_size_message(max_size).to_owned());
    }
    let mut body = vec![0_u8; content_length];
    reader
        .read_exact(&mut body)
        .map_err(|error| format!("HTTP response body read failed: {error}"))?;
    Ok(body)
}

fn read_to_eof_bounded(
    reader: &mut BufferedReader<'_>,
    max_allowed: usize,
    max_size: usize,
) -> std::result::Result<Vec<u8>, String> {
    let mut body = Vec::new();
    let mut chunk = [0_u8; 8192];
    loop {
        let count = match reader.read(&mut chunk) {
            Ok(0) => return Ok(body),
            Ok(n) => n,
            Err(error) if error.kind() == ErrorKind::Interrupted => continue,
            Err(error) => return Err(format!("HTTP response body read failed: {error}")),
        };
        let next_len = body
            .len()
            .checked_add(count)
            .ok_or_else(|| exceeded_size_message(max_size).to_owned())?;
        if next_len > max_allowed {
            return Err(exceeded_size_message(max_size).to_owned());
        }
        body.extend_from_slice(&chunk[..count]);
    }
}

fn connect_http(parsed: &ParsedUrl, timeout: Duration) -> std::result::Result<TcpStream, String> {
    let addresses = (parsed.connect_host.as_str(), parsed.port)
        .to_socket_addrs()
        .map_err(|error| format!("HTTP DNS resolution failed: {error}"))?;
    let mut last_error = None;
    for address in addresses {
        match TcpStream::connect_timeout(&address, timeout) {
            Ok(stream) => {
                stream
                    .set_read_timeout(Some(timeout))
                    .map_err(|error| format!("failed to configure read timeout: {error}"))?;
                stream
                    .set_write_timeout(Some(timeout))
                    .map_err(|error| format!("failed to configure write timeout: {error}"))?;
                return Ok(stream);
            }
            Err(error) => last_error = Some(error),
        }
    }
    Err(last_error.map_or_else(
        || "no socket addresses resolved for HTTP host".to_owned(),
        |error| format!("HTTP connect failed: {error}"),
    ))
}

fn build_http_request(
    parsed: &ParsedUrl,
    headers: &[HeaderEntry<'_>],
    forward_host: bool,
) -> String {
    use std::fmt::Write as _;

    let mut request = format!("GET {} HTTP/1.1\r\n", parsed.target);
    let custom_host = forward_host
        && headers
            .iter()
            .any(|header| header.name.eq_ignore_ascii_case("host"));
    if !custom_host {
        let _ = write!(request, "Host: {}\r\n", parsed.host_header);
    }
    let custom_ua = headers
        .iter()
        .any(|header| header.name.eq_ignore_ascii_case("user-agent"));
    if !custom_ua {
        request.push_str("User-Agent: prism-upstream-reference/1.0\r\n");
    }
    request.push_str("Accept: */*\r\nConnection: close\r\n");
    for header in headers {
        if header.name.eq_ignore_ascii_case("connection")
            || (!forward_host && header.name.eq_ignore_ascii_case("host"))
        {
            continue;
        }
        let _ = write!(request, "{}: {}\r\n", header.name, header.value);
    }
    request.push_str("\r\n");
    request
}

enum FetchOutcome {
    Redirect(String),
    Complete { status_code: u16, body: Vec<u8> },
}

fn fetch_http_once(
    parsed: &ParsedUrl,
    headers: &[HeaderEntry<'_>],
    forward_host: bool,
    timeout: Duration,
    max_size: usize,
) -> std::result::Result<FetchOutcome, String> {
    let mut stream = connect_http(parsed, timeout)?;
    let request = build_http_request(parsed, headers, forward_host);
    stream
        .write_all(request.as_bytes())
        .map_err(|error| format!("HTTP request write failed: {error}"))?;
    stream
        .flush()
        .map_err(|error| format!("HTTP request flush failed: {error}"))?;

    let mut reader = BufferedReader::new(&mut stream);
    let head = loop {
        let block = reader.read_header_block()?;
        let parsed_head = parse_response_headers(&block)?;
        if parsed_head.status_code / 100 != 1 {
            break parsed_head;
        }
    };

    if matches!(head.status_code, 301 | 302 | 303 | 307 | 308)
        && let Some(location) = head.location.as_deref()
    {
        return Ok(FetchOutcome::Redirect(resolve_redirect(parsed, location)?));
    }

    let max_allowed = if max_size > 0 {
        max_size
    } else {
        MAX_REMOTE_SIZE
    };
    let body = if head.status_code == 204 || head.status_code == 304 {
        Vec::new()
    } else if head.is_chunked {
        read_chunked_body(&mut reader, max_allowed, max_size)?
    } else if let Some(content_length) = head.content_length {
        read_exact_bounded(&mut reader, content_length, max_allowed, max_size)?
    } else {
        read_to_eof_bounded(&mut reader, max_allowed, max_size)?
    };

    Ok(FetchOutcome::Complete {
        status_code: head.status_code,
        body,
    })
}

fn fetch_http(
    initial_url: &str,
    headers: &[HeaderEntry<'_>],
    timeout_seconds: i32,
    max_size: usize,
) -> std::result::Result<(u16, Vec<u8>), String> {
    let timeout = Duration::from_secs(if timeout_seconds > 0 {
        u64::try_from(timeout_seconds).unwrap_or(DEFAULT_TIMEOUT_SECONDS)
    } else {
        DEFAULT_TIMEOUT_SECONDS
    });
    let mut current_url = initial_url.to_owned();
    let mut forward_host = true;
    let initial_authority = parse_http_url(initial_url)?.host_header;
    for redirect in 0..=MAX_REDIRECTS {
        let parsed = parse_http_url(&current_url)?;
        if parsed.host_header != initial_authority {
            forward_host = false;
        }
        match fetch_http_once(&parsed, headers, forward_host, timeout, max_size)? {
            FetchOutcome::Complete { status_code, body } => return Ok((status_code, body)),
            FetchOutcome::Redirect(next_url) => {
                if redirect == MAX_REDIRECTS {
                    return Err("HTTP redirect limit exceeded".to_owned());
                }
                current_url = next_url;
            }
        }
    }
    Err("HTTP redirect limit exceeded".to_owned())
}

fn execute_remote_operation(operation: &Value) -> Result<Value> {
    let url = text(&operation["Url"], MAX_URL_BYTES)?;
    let timeout_seconds = integer(&operation["TimeoutSeconds"], 0, MAX_TIMEOUT_SECONDS)?;
    let max_size = usize::try_from(integer(&operation["MaxSize"], 0, MAX_REMOTE_SIZE_I32)?)?;
    let headers = array(&operation["Headers"])?
        .iter()
        .map(|header| {
            Ok(HeaderEntry {
                name: text(&header["Name"], MAX_HEADER_FIELD_BYTES)?,
                value: text(&header["Value"], MAX_HEADER_FIELD_BYTES)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    match fetch_http(url, &headers, timeout_seconds, max_size) {
        Ok((status_code, body)) => Ok(json!({
            "HttpCode": status_code,
            "BodyHex": hex::encode(body),
            "Exception": Value::Null,
        })),
        Err(message) => Ok(json!({
            "HttpCode": Value::Null,
            "BodyHex": Value::Null,
            "Exception": exception_object(&message),
        })),
    }
}

fn execute_operation(operation: &Value) -> Result<Value> {
    match text(&operation["Kind"], 64)? {
        "argument-enums" => Ok(json!({
            "Count": EXAMPLE_COUNT,
            "Common": COMMON,
            "Speculative": SPECULATIVE,
            "Embedding": 6,
        })),
        "argument-parser-trace" => execute_trace_operation(operation),
        "remote-get-content" => execute_remote_operation(operation),
        _ => Err(invalid("unknown argument observation kind")),
    }
}

pub(super) fn handles(request: &Value) -> bool {
    request["Operations"].as_array().is_some_and(|operations| {
        !operations.is_empty()
            && operations.iter().all(|operation| {
                matches!(
                    operation["Kind"].as_str(),
                    Some("argument-enums" | "argument-parser-trace" | "remote-get-content")
                )
            })
    })
}

pub(super) fn execute(request: &Value) -> Result<Value> {
    fields(request, &["SchemaVersion", "Operations"])?;
    integer(&request["SchemaVersion"], 1, 1)?;
    let operations = array(&request["Operations"])?;
    if operations.is_empty() || operations.len() > MAX_OPERATIONS {
        return Err(invalid("argument operations must number 1 through 128"));
    }
    for operation in operations {
        validate_operation(operation)?;
    }
    let results = operations
        .iter()
        .map(execute_operation)
        .collect::<Result<Vec<_>>>()?;
    Ok(json!({"SchemaVersion": 1, "Results": results}))
}

pub(super) fn identity() -> Value {
    json!({
        "Kind": "independent-rust-arg-parser",
        "SupportedSourceIds": [ID],
        "UpstreamCommit": super::UPSTREAM,
        "AssertionSource": {"Path": ID, "Sha256": SOURCE_SHA},
        "Architecture": "clean-room-cli-argument-parser-speculative-limits-and-http11-client-v1",
        "NativeReferenceReplay": false,
        "ProductionRuntimeCalled": false,
        "CompleteSourceGate": false,
        "NativeRetirementQualified": false
    })
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;

    use super::*;
    use crate::engine::{Engine, Target};

    fn serve_once(response: Vec<u8>) -> Result<(String, std::thread::JoinHandle<()>)> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let handle = std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut reader = BufReader::new(&stream);
                let mut line = String::new();
                while reader.read_line(&mut line).is_ok_and(|n| n > 0) {
                    if line == "\r\n" || line == "\n" {
                        break;
                    }
                    line.clear();
                }
                let _ = stream.write_all(&response);
                let _ = stream.flush();
            }
        });
        Ok((format!("http://127.0.0.1:{port}/test"), handle))
    }

    #[test]
    fn independent_arg_parser_passes_complete_test_arg_parser_suite() -> Result<()> {
        for target in [Target::RustReference, Target::Cpp] {
            let engine = Engine::new(target, None, None, None, None, None)?;
            let report = crate::arg_parser::run(&engine)?;
            assert_eq!(report["Id"], ID);
            assert_eq!(report["Status"], "pass", "{report}");
        }
        assert!(crate::reference_provider::supports(ID));
        let info = identity();
        assert_eq!(info["SupportedSourceIds"], json!([ID]));
        assert_eq!(info["NativeReferenceReplay"], false);
        assert_eq!(info["ProductionRuntimeCalled"], false);
        assert_eq!(info["CompleteSourceGate"], false);
        assert_eq!(info["NativeRetirementQualified"], false);
        Ok(())
    }

    #[test]
    fn independent_remote_get_content_handles_loopback_200_404_chunked_redirect_and_maxsize()
    -> Result<()> {
        let (ok_url, ok_thread) =
            serve_once(b"HTTP/1.1 200 OK\r\nContent-Length: 16\r\n\r\nhello llama.cpp!".to_vec())?;
        let (not_found_url, not_found_thread) =
            serve_once(b"HTTP/1.1 404 Not Found\r\nContent-Length: 7\r\n\r\nmissing".to_vec())?;
        let (chunked_url, chunked_thread) = serve_once(
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nllama\r\n4\r\n.cpp\r\n0\r\n\r\n"
                .to_vec(),
        )?;
        let (max_url, max_thread) =
            serve_once(b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\n0123456789".to_vec())?;

        let response = execute(&json!({
            "SchemaVersion": 1,
            "Operations": [
                {"Kind": "remote-get-content", "Url": ok_url, "Headers": [], "TimeoutSeconds": 5, "MaxSize": 0},
                {"Kind": "remote-get-content", "Url": not_found_url, "Headers": [], "TimeoutSeconds": 5, "MaxSize": 0},
                {"Kind": "remote-get-content", "Url": chunked_url, "Headers": [], "TimeoutSeconds": 5, "MaxSize": 0},
                {"Kind": "remote-get-content", "Url": max_url, "Headers": [], "TimeoutSeconds": 5, "MaxSize": 1},
                {"Kind": "remote-get-content", "Url": "https://127.0.0.1:1/no-tls", "Headers": [], "TimeoutSeconds": 1, "MaxSize": 0},
            ]
        }))?;
        for handle in [ok_thread, not_found_thread, chunked_thread, max_thread] {
            let _ = handle.join();
        }
        let results = array(&response["Results"])?;
        assert_eq!(results[0]["HttpCode"], 200);
        assert_eq!(results[0]["BodyHex"], hex::encode(b"hello llama.cpp!"));
        assert!(results[0]["Exception"].is_null());

        assert_eq!(results[1]["HttpCode"], 404);
        assert_eq!(results[1]["BodyHex"], hex::encode(b"missing"));
        assert!(results[1]["Exception"].is_null());

        assert_eq!(results[2]["HttpCode"], 200);
        assert_eq!(results[2]["BodyHex"], hex::encode(b"llama.cpp"));
        assert!(results[2]["Exception"].is_null());

        assert!(results[3]["HttpCode"].is_null());
        assert!(results[3]["BodyHex"].is_null());
        assert_eq!(results[3]["Exception"]["Kind"], "standard");
        assert_eq!(
            results[3]["Exception"]["Message"],
            "remote content exceeds maximum size"
        );

        assert!(results[4]["HttpCode"].is_null());
        assert!(results[4]["BodyHex"].is_null());
        assert_eq!(results[4]["Exception"]["Kind"], "standard");

        let (target_url, target_thread) =
            serve_once(b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\nredirected".to_vec())?;
        let redirect_raw =
            format!("HTTP/1.1 302 Found\r\nLocation: {target_url}\r\nContent-Length: 0\r\n\r\n");
        let (redir_url, redir_thread) = serve_once(redirect_raw.into_bytes())?;
        let redir_response = execute(&json!({
            "SchemaVersion": 1,
            "Operations": [
                {"Kind": "remote-get-content", "Url": redir_url, "Headers": [{"Name": "X-Probe", "Value": "1"}], "TimeoutSeconds": 5, "MaxSize": 0},
            ]
        }))?;
        let _ = redir_thread.join();
        let _ = target_thread.join();
        assert_eq!(redir_response["Results"][0]["HttpCode"], 200);
        assert_eq!(
            redir_response["Results"][0]["BodyHex"],
            hex::encode(b"redirected")
        );
        Ok(())
    }

    #[test]
    fn independent_parser_handles_csv_rollback_and_rejects_malformed_requests() -> Result<()> {
        assert_eq!(
            parse_csv(r#"file1.gguf,"file2,2.gguf","file3""3"".gguf",file4".gguf"#),
            vec![
                "file1.gguf",
                "file2,2.gguf",
                "file3\"3\".gguf",
                "file4\".gguf",
            ]
        );
        assert_eq!(parse_csv(""), Vec::<String>::new());

        let trace_response = execute(&json!({
            "SchemaVersion": 1,
            "Operations": [{
                "Kind": "argument-parser-trace",
                "Initials": [{}],
                "Actions": [
                    {"Action": "parse", "Slot": 0, "Example": COMMON, "Arguments": ["bin", "-t", "99", "--repeat-last-n", "-1"]},
                    {"Action": "snapshot", "Slot": 0},
                ]
            }]
        }))?;
        assert_eq!(trace_response["Results"][0]["Trace"][0]["Parsed"], false);
        assert_eq!(
            trace_response["Results"][0]["Trace"][1]["Params"]["Threads"],
            -1
        );

        assert!(!handles(&json!({"SchemaVersion": 1, "Operations": []})));
        assert!(!handles(
            &json!({"SchemaVersion": 1, "Operations": [{"Kind": "gguf-inventory"}]})
        ));

        for invalid_request in [
            json!({"SchemaVersion": 2, "Operations": [{"Kind": "argument-enums"}]}),
            json!({"SchemaVersion": true, "Operations": [{"Kind": "argument-enums"}]}),
            json!({"SchemaVersion": 1, "Operations": []}),
            json!({"SchemaVersion": 1, "Operations": vec![json!({"Kind": "argument-enums"}); 129]}),
            json!({"SchemaVersion": 1, "Operations": [{"Kind": "argument-enums", "Extra": 1}]}),
            json!({"SchemaVersion": 1, "Operations": [{"Kind": "unknown-op"}]}),
            json!({"SchemaVersion": 1, "Operations": [{"Kind": "remote-get-content", "Url": "ftp://example.com/", "Headers": [], "TimeoutSeconds": 0, "MaxSize": 0}]}),
            json!({"SchemaVersion": 1, "Operations": [{"Kind": "remote-get-content", "Url": "http://example.com/", "Headers": [{"Name": "X-Bad\r\n", "Value": "1"}], "TimeoutSeconds": 0, "MaxSize": 0}]}),
            json!({"SchemaVersion": 1, "Operations": [{"Kind": "remote-get-content", "Url": "http://example.com/", "Headers": [], "TimeoutSeconds": 61, "MaxSize": 0}]}),
            json!({"SchemaVersion": 1, "Operations": [{"Kind": "remote-get-content", "Url": "http://example.com/", "Headers": [], "TimeoutSeconds": 0, "MaxSize": MAX_REMOTE_SIZE + 1}]}),
            json!({"SchemaVersion": 1, "Operations": [{"Kind": "argument-parser-trace", "Initials": [], "Actions": []}]}),
            json!({"SchemaVersion": 1, "Operations": [{"Kind": "argument-parser-trace", "Initials": [{"Parallel": 0}], "Actions": []}]}),
            json!({"SchemaVersion": 1, "Operations": [{"Kind": "argument-parser-trace", "Initials": [{}], "Actions": [{"Action": "snapshot", "Slot": 1}]}]}),
            json!({"SchemaVersion": 1, "Operations": [{"Kind": "argument-parser-trace", "Initials": [{}], "Actions": [{"Action": "inventory", "Slot": 0, "Example": 27}]}]}),
            json!({"SchemaVersion": 1, "Operations": [{"Kind": "argument-parser-trace", "Initials": [{}], "Actions": [{"Action": "parse", "Slot": 0, "Example": 2, "Arguments": []}]}]}),
            json!({"SchemaVersion": 1, "Operations": [{"Kind": "argument-parser-trace", "Initials": [{}], "Actions": [{"Action": "limits", "Batch": 0, "Parallel": 1, "Draft": 1}]}]}),
        ] {
            assert!(execute(&invalid_request).is_err(), "{invalid_request}");
        }
        Ok(())
    }
}
