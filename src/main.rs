#![forbid(unsafe_code)]
//! Explicit token and plain-text driver for the unqualified CPU reference engine.

use bonsai_prism_runtime::{
    MAX_PREFILL_BLOCK_TOKENS, Result,
    graph::{MAX_STATE_BYTES, Qwen35},
    kernels::{MatvecBackend, MatvecDiagnostics, MatvecFallbackReason, MatvecStatistics},
    model::{Model, ModelMetadata},
    prompt_input::read_prompt,
    runtime::{Cancellation, Finish, Generation, Limits, Prefill, Runtime},
    sampling::{Greedy, ReferenceSampler, Sampler, Sampling},
    special_input::SpecialInputPolicy,
    special_tokens::SpecialTokens,
    text_output::Utf8Output,
    token_input::read_tokens,
    tokenization::{
        SpecialTokenInsertion, TextTokenizationPolicy, TokenBudget, TokenizationSession,
    },
    tokenizer::Tokenizer,
    vocabulary::Vocabulary,
};
use serde_json::json;
use std::{
    collections::BTreeMap,
    io::Write,
    path::PathBuf,
    process::ExitCode,
    time::{Duration, Instant},
};

const USAGE: &str = "Native CPU reference; full-model parity and performance are unqualified.\n\
Usage: prism-inference --capabilities\n\
Reports versioned compiled support as one JSON record without loading a model.\n\
Build eligibility and graph contract IDs do not qualify a hardware profile.\n\
Usage: prism-inference --model <absolute-gguf-path> --bytes <u64>\n\
  --sha256 <64-hex> --memory-bytes <usize> --context <1..262144>\n\
  [--mode <generate|tokenize>]\n\
  [--checkpoint-bytes <usize>] [--max-tokens <usize>]\n\
  (--tokens <id,id,...> | --tokens-file <absolute-path|-> |\n\
   --prompt <utf8-text> | --prompt-file <absolute-path|->)\n\
  [--special-input <literal|parse>] [--insert-special <none|model>]\n\
  [--stop-tokens <id,id,...>] [--backend <auto|scalar|neon>]\n\
  [--timeout-seconds <1..3600>] [--output <tokens|text>]\n\
  [--state-bytes <usize>]\n\
  [--prefill-block-tokens <1..256>]\n\
  [--sampling <greedy|reference>] [--seed <u64>]\n\
  [--temperature <positive-f32>] [--top-k <0..1048576>] [--top-p <0..1>]\n\
Mode defaults to generate, which requires checkpoint-bytes and positive max-tokens.\n\
Tokenize verifies the artifact and bounded metadata without loading weights or\n\
constructing a graph, then emits one JSON record with exact input IDs and count.\n\
Its optional max-tokens reserves context for output (default zero); generation-only\n\
checkpoint, stop, backend, timeout, output, state, prefill and sampling options reject.\n\
The memory budget covers retained metadata only in tokenize mode; tokenizer\n\
storage and input/output obey their separate fixed resource bounds.\n\
State budget defaults to 4 GiB for live KV/recurrent element payload only;\n\
model storage, graph scratch, checkpoints and allocator overhead are separate.\n\
Timeout covers generation after loading. Checks are cooperative between operations;\n\
streamed tokens are provisional until the final completion record.\n\
Backend defaults to auto. NEON requires the experimental-neon feature and\n\
native Windows ARM64 MSVC; unsupported builds, targets and formats use scalar.\n\
Selection is per matrix operation; backend_requested reports only the request.\n\
backend_last_operation records just the latest projection, including scalar fallback.\n\
backend_statistics counts all attempted matrix work, including provisional work.\n\
Prefill blocks default to one token and preserve sequential reference arithmetic.\n\
Input is exact token IDs or bounded plain text; no chat template is applied.\n\
Text BOS/EOS insertion defaults to none; model explicitly follows insertion flags.\n\
Special input spellings stay literal by default; special-input and insert-special\n\
require text input. Prompt files and stdin (-) contain bounded exact UTF-8;\n\
newlines and a UTF-8 BOM are preserved. Input is read before model verification.\n\
Token files and stdin (-) contain a nonempty strict JSON array of u32 IDs,\n\
bounded to 1 MiB and the available prompt context; IDs are preserved exactly.\n\
Text input is experimental; full-model tokenizer parity remains unqualified.\n\
Decoding defaults to greedy. Output is one JSON object per line.\n\
Reference sampling is experimental, requires --seed, and uses temperature, top-k,\n\
then top-p with SplitMix64. Defaults are 1, 20 and 0.95; top-k 0 retains all IDs.\n\
Top-p must be greater than zero. Sampling controls require --sampling reference.\n\
This reference RNG/sampler does not establish Prism sampling or quality parity.\n\
Model-declared EOS/EOT/EOM always stop generation; --stop-tokens adds IDs.\n\
Text output adds lossless token bytes and incremental UTF-8 text (invalid bytes\n\
display as replacement characters). Control token spellings remain visible.\n";

struct Arguments {
    mode: Mode,
    path: PathBuf,
    bytes: u64,
    sha256: String,
    memory: usize,
    context: usize,
    checkpoint_bytes: usize,
    state_bytes: usize,
    generation: Generation,
    tokens: Vec<u32>,
    prompt: Option<String>,
    special_input: SpecialInputPolicy,
    insert_special: InsertSpecial,
    backend: MatvecBackend,
    timeout: Option<Duration>,
    prefill: Prefill,
    text_output: bool,
    sampling: SamplingMode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Generate,
    Tokenize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InsertSpecial {
    None,
    Model,
}

impl InsertSpecial {
    const fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Model => "model",
        }
    }
}

#[derive(Clone, Copy)]
enum SamplingMode {
    Greedy,
    Reference { options: Sampling, seed: u64 },
}

impl SamplingMode {
    fn sampler(self) -> Result<DriverSampler> {
        match self {
            Self::Greedy => Ok(DriverSampler::Greedy(Greedy)),
            Self::Reference { options, seed } => Ok(DriverSampler::Reference(
                ReferenceSampler::new(options, seed)?,
            )),
        }
    }

    fn record(self) -> serde_json::Value {
        match self {
            Self::Greedy => json!({"mode": "greedy"}),
            Self::Reference { options, seed } => json!({
                "mode": "reference",
                "rng": "splitmix64",
                "seed": seed,
                "temperature": options.temperature,
                "top_k": options.top_k,
                "top_p": options.top_p,
            }),
        }
    }
}

#[derive(Clone)]
enum DriverSampler {
    Greedy(Greedy),
    Reference(ReferenceSampler),
}

impl Sampler for DriverSampler {
    fn sample(&mut self, logits: &[f32]) -> Result<u32> {
        match self {
            Self::Greedy(sampler) => sampler.sample(logits),
            Self::Reference(sampler) => sampler.sample(logits),
        }
    }
}

fn bad(message: &'static str) -> bonsai_prism_runtime::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message).into()
}

fn option_values(arguments: impl IntoIterator<Item = String>) -> Result<BTreeMap<String, String>> {
    let mut values = BTreeMap::new();
    let mut args = arguments.into_iter();
    while let Some(key) = args.next() {
        if !matches!(
            key.as_str(),
            "--model"
                | "--mode"
                | "--bytes"
                | "--sha256"
                | "--memory-bytes"
                | "--context"
                | "--checkpoint-bytes"
                | "--max-tokens"
                | "--tokens"
                | "--tokens-file"
                | "--prompt"
                | "--prompt-file"
                | "--special-input"
                | "--insert-special"
                | "--stop-tokens"
                | "--backend"
                | "--timeout-seconds"
                | "--state-bytes"
                | "--prefill-block-tokens"
                | "--output"
                | "--sampling"
                | "--seed"
                | "--temperature"
                | "--top-k"
                | "--top-p"
        ) || values.contains_key(&key)
        {
            return Err(bad("unknown or repeated option"));
        }
        let value = args.next().ok_or_else(|| bad("missing option value"))?;
        if value.len() > 4 * 1024 * 1024 {
            return Err(bad("option exceeds bounded input size"));
        }
        values.insert(key, value);
    }
    Ok(values)
}

fn mode_options(values: &mut BTreeMap<String, String>) -> Result<Mode> {
    let mode = match values.remove("--mode").as_deref().unwrap_or("generate") {
        "generate" => Mode::Generate,
        "tokenize" => Mode::Tokenize,
        _ => return Err(bad("mode must be generate or tokenize")),
    };
    if mode == Mode::Tokenize
        && [
            "--checkpoint-bytes",
            "--stop-tokens",
            "--backend",
            "--timeout-seconds",
            "--output",
            "--state-bytes",
            "--prefill-block-tokens",
            "--sampling",
            "--seed",
            "--temperature",
            "--top-k",
            "--top-p",
        ]
        .iter()
        .any(|key| values.contains_key(*key))
    {
        return Err(bad(
            "generation-only option is incompatible with tokenize mode",
        ));
    }
    Ok(mode)
}

fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Arguments> {
    let mut values = option_values(arguments)?;
    let mode = mode_options(&mut values)?;
    let mut required = |key: &str| {
        values
            .remove(key)
            .ok_or_else(|| bad("missing required option"))
    };
    let path = PathBuf::from(required("--model")?);
    if !path.is_absolute() {
        return Err(bad("model path must be absolute"));
    }
    let bytes = required("--bytes")?.parse::<u64>()?;
    let sha256 = required("--sha256")?;
    let memory = required("--memory-bytes")?.parse::<usize>()?;
    let context = required("--context")?.parse::<usize>()?;
    let (checkpoint_bytes, max_tokens) = match mode {
        Mode::Generate => (
            required("--checkpoint-bytes")?.parse::<usize>()?,
            required("--max-tokens")?.parse::<usize>()?,
        ),
        Mode::Tokenize => (
            0,
            values
                .remove("--max-tokens")
                .map(|value| value.parse::<usize>())
                .transpose()?
                .unwrap_or(0),
        ),
    };
    // Reject model-independent errors before opening/hashing a multi-GiB file
    // or allocating model and graph storage. Vocabulary checks still require
    // the verified model and remain the runtime's responsibility.
    if bytes == 0
        || sha256.len() != 64
        || !sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        || memory == 0
        || !(1..=262_144).contains(&context)
        || (mode == Mode::Generate && (checkpoint_bytes == 0 || max_tokens == 0))
        || max_tokens >= context
    {
        return Err(bad("invalid artifact identity or native resource bounds"));
    }
    let stop_tokens = values
        .remove("--stop-tokens")
        .map(|text| token_ids(&text, 256))
        .transpose()?
        .unwrap_or_default();
    let state_bytes = values
        .remove("--state-bytes")
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(MAX_STATE_BYTES);
    if state_bytes == 0 || state_bytes > MAX_STATE_BYTES {
        return Err(bad("invalid native live-state payload budget"));
    }
    let backend = match values.remove("--backend").as_deref().unwrap_or("auto") {
        "auto" => MatvecBackend::Auto,
        "scalar" => MatvecBackend::Scalar,
        "neon" => MatvecBackend::ExperimentalNeon,
        _ => return Err(bad("backend must be auto, scalar or neon")),
    };
    let timeout = values
        .remove("--timeout-seconds")
        .as_deref()
        .map(timeout_seconds)
        .transpose()?;
    let prefill = prefill_options(values.remove("--prefill-block-tokens").as_deref())?;
    let text_output = match values.remove("--output").as_deref().unwrap_or("tokens") {
        "tokens" => false,
        "text" => true,
        _ => return Err(bad("output must be tokens or text")),
    };
    let sampling = sampling_options(&mut values)?;
    // Validate all non-input options before a possible blocking stdin read.
    let text_input = values.contains_key("--prompt") || values.contains_key("--prompt-file");
    let insert_special = insertion_options(&mut values, text_input)?;
    let (tokens, prompt, special_input) = input_options(&mut values, context - max_tokens)?;
    if !values.is_empty() {
        return Err(bad("incompatible option"));
    }
    Ok(Arguments {
        mode,
        path,
        bytes,
        sha256,
        memory,
        context,
        checkpoint_bytes,
        state_bytes,
        generation: Generation {
            max_tokens,
            stop_tokens,
        },
        tokens,
        prompt,
        special_input,
        insert_special,
        backend,
        timeout,
        prefill,
        text_output,
        sampling,
    })
}

fn insertion_options(
    values: &mut BTreeMap<String, String>,
    text_input: bool,
) -> Result<InsertSpecial> {
    let insertion = values.remove("--insert-special");
    if insertion.is_some() && !text_input {
        return Err(bad("special token insertion requires text input"));
    }
    match insertion.as_deref().unwrap_or("none") {
        "none" => Ok(InsertSpecial::None),
        "model" => Ok(InsertSpecial::Model),
        _ => Err(bad("special token insertion must be none or model")),
    }
}

fn input_options(
    values: &mut BTreeMap<String, String>,
    max_tokens: usize,
) -> Result<(Vec<u32>, Option<String>, SpecialInputPolicy)> {
    let tokens = values.remove("--tokens");
    let tokens_file = values.remove("--tokens-file");
    let prompt = values.remove("--prompt");
    let prompt_file = values.remove("--prompt-file");
    let policy = values.remove("--special-input");
    match (tokens, tokens_file, prompt, prompt_file) {
        (Some(tokens), None, None, None) if policy.is_none() => Ok((
            token_ids(&tokens, max_tokens)?,
            None,
            SpecialInputPolicy::Literal,
        )),
        (None, Some(source), None, None) if policy.is_none() => {
            let tokens = if source == "-" {
                read_tokens(&mut std::io::stdin().lock(), max_tokens)?
            } else {
                let path = PathBuf::from(source);
                if !path.is_absolute() {
                    return Err(bad("token file path must be absolute or - for stdin"));
                }
                let mut file = std::fs::File::open(path).map_err(|error| {
                    std::io::Error::new(error.kind(), "failed to open raw token input")
                })?;
                read_tokens(&mut file, max_tokens)?
            };
            if tokens.is_empty() {
                return Err(bad("raw token input must not be empty"));
            }
            Ok((tokens, None, SpecialInputPolicy::Literal))
        }
        (None, None, prompt, prompt_file) if prompt.is_some() != prompt_file.is_some() => {
            let policy = match policy.as_deref().unwrap_or("literal") {
                "literal" => SpecialInputPolicy::Literal,
                "parse" => SpecialInputPolicy::Parse,
                _ => return Err(bad("special input must be literal or parse")),
            };
            let prompt = if let Some(prompt) = prompt {
                prompt
            } else {
                let source = prompt_file.ok_or_else(|| bad("missing text input source"))?;
                if source == "-" {
                    read_prompt(&mut std::io::stdin().lock())?
                } else {
                    let path = PathBuf::from(source);
                    if !path.is_absolute() {
                        return Err(bad("prompt file path must be absolute or - for stdin"));
                    }
                    read_prompt(&mut std::fs::File::open(path)?)?
                }
            };
            if prompt.is_empty() || prompt.len() > bonsai_prism_runtime::bpe::MAX_BPE_INPUT_BYTES {
                return Err(bad(
                    "text prompt must contain 1 through 1048576 UTF-8 bytes",
                ));
            }
            Ok((Vec::new(), Some(prompt), policy))
        }
        _ => Err(bad(
            "provide exactly one of --tokens, --tokens-file, --prompt or --prompt-file; special input policy requires text input",
        )),
    }
}

fn sampling_options(values: &mut BTreeMap<String, String>) -> Result<SamplingMode> {
    match values.remove("--sampling").as_deref().unwrap_or("greedy") {
        "greedy" => {
            if ["--seed", "--temperature", "--top-k", "--top-p"]
                .iter()
                .any(|key| values.contains_key(*key))
            {
                return Err(bad("sampling controls require --sampling reference"));
            }
            Ok(SamplingMode::Greedy)
        }
        "reference" => {
            let seed = values
                .remove("--seed")
                .ok_or_else(|| bad("reference sampling requires an explicit seed"))?;
            if seed.is_empty() || !seed.bytes().all(|byte| byte.is_ascii_digit()) {
                return Err(bad("seed must be an unsigned decimal integer"));
            }
            let seed = seed.parse::<u64>()?;
            let defaults = Sampling::default();
            let temperature = values
                .remove("--temperature")
                .map(|value| value.parse::<f32>())
                .transpose()?
                .unwrap_or(defaults.temperature);
            let top_k = values
                .remove("--top-k")
                .map(|value| {
                    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
                        return Err(bad("top-k must be an unsigned decimal integer"));
                    }
                    Ok(value.parse::<usize>()?)
                })
                .transpose()?
                .unwrap_or(defaults.top_k);
            let top_p = values
                .remove("--top-p")
                .map(|value| value.parse::<f32>())
                .transpose()?
                .unwrap_or(defaults.top_p);
            let sampling = SamplingMode::Reference {
                options: Sampling {
                    temperature,
                    top_k,
                    top_p,
                },
                seed,
            };
            // Reject configuration errors before model hashing or graph allocation.
            let _validated = sampling.sampler()?;
            Ok(sampling)
        }
        _ => Err(bad("sampling must be greedy or reference")),
    }
}

fn prefill_options(value: Option<&str>) -> Result<Prefill> {
    let value = value.unwrap_or("1");
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(bad("prefill block tokens must be from 1 through 256"));
    }
    let block_tokens = value.parse::<usize>()?;
    if !(1..=MAX_PREFILL_BLOCK_TOKENS).contains(&block_tokens) {
        return Err(bad("prefill block tokens must be from 1 through 256"));
    }
    Ok(Prefill { block_tokens })
}

fn timeout_seconds(value: &str) -> Result<Duration> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(bad(
            "timeout seconds must be an integer from 1 through 3600",
        ));
    }
    let seconds = value.parse::<u64>()?;
    if !(1..=3600).contains(&seconds) {
        return Err(bad(
            "timeout seconds must be an integer from 1 through 3600",
        ));
    }
    Ok(Duration::from_secs(seconds))
}

fn token_ids(text: &str, limit: usize) -> Result<Vec<u32>> {
    let mut result = Vec::new();
    for token in text.split(',') {
        if result.len() == limit
            || token.is_empty()
            || !token.bytes().all(|byte| byte.is_ascii_digit())
        {
            return Err(bad("invalid or oversized token sequence"));
        }
        result.try_reserve(1)?;
        result.push(token.parse()?);
    }
    Ok(result)
}

struct Loaded {
    decoder: Qwen35,
    generation: Generation,
    vocabulary: Option<Vocabulary>,
    tokens: Vec<u32>,
}

fn load_decoder(arguments: &Arguments) -> Result<Loaded> {
    let model = Model::load(
        &arguments.path,
        arguments.bytes,
        &arguments.sha256,
        arguments.memory,
    )?;
    let vocab_size = model
        .tensors
        .get("token_embd.weight")
        .filter(|tensor| tensor.dimensions.len() == 2)
        .and_then(|tensor| tensor.dimensions.get(1))
        .copied()
        .ok_or_else(|| bad("missing token embedding vocabulary dimension"))?;
    let special_tokens = SpecialTokens::from_metadata(&model.metadata, vocab_size)?;
    let generation = special_tokens.generation(
        arguments.generation.max_tokens,
        &arguments.generation.stop_tokens,
    )?;
    let tokens = if let Some(prompt) = &arguments.prompt {
        encode_prompt(
            arguments,
            prompt,
            &model.metadata,
            vocab_size,
            &special_tokens,
        )?
    } else {
        arguments.tokens.clone()
    };
    let vocabulary = arguments
        .text_output
        .then(|| Vocabulary::from_metadata(&model.metadata, vocab_size))
        .transpose()?;
    let decoder = Qwen35::with_state_budget(
        model,
        arguments.context,
        arguments.state_bytes,
        arguments.backend,
    )?;
    Ok(Loaded {
        decoder,
        generation,
        vocabulary,
        tokens,
    })
}

fn execute(arguments: &Arguments) -> Result<()> {
    if arguments.mode == Mode::Tokenize {
        let record = tokenize_record(arguments)?;
        let mut output = std::io::stdout().lock();
        serde_json::to_writer(&mut output, &record)?;
        output.write_all(b"\n")?;
        output.flush()?;
        return Ok(());
    }
    let mut sampler = arguments.sampling.sampler()?;
    let Loaded {
        decoder,
        generation,
        vocabulary,
        tokens,
    } = load_decoder(arguments)?;
    let backend_requested = backend_name(decoder.requested_backend());
    let mut runtime = Runtime::new(
        decoder,
        Limits {
            context_tokens: arguments.context,
            checkpoints: 0,
            checkpoint_bytes: arguments.checkpoint_bytes,
        },
    )?;
    let mut output = std::io::stdout().lock();
    let cancellation = Cancellation::default();
    let mut text_output = Utf8Output::default();
    let emit = |token| {
        let record = if let Some(vocabulary) = &vocabulary {
            let bytes = vocabulary.piece(token)?;
            json!({
                "token": token,
                "bytes": bytes,
                "text": text_output.push(bytes)?,
                "special": vocabulary.is_control(token)?,
            })
        } else {
            json!({"token": token})
        };
        serde_json::to_writer(&mut output, &record)?;
        output.write_all(b"\n")?;
        output.flush()?;
        Ok(())
    };
    let result = if let Some(timeout) = arguments.timeout {
        let deadline = Instant::now()
            .checked_add(timeout)
            .ok_or_else(|| bad("generation deadline exceeds monotonic clock bounds"))?;
        runtime.generate_sampled_until(
            &tokens,
            &generation,
            arguments.prefill,
            &cancellation,
            deadline,
            &mut sampler,
            emit,
        )?
    } else {
        runtime.generate_sampled(
            &tokens,
            &generation,
            arguments.prefill,
            &cancellation,
            &mut sampler,
            emit,
        )?
    };
    let mut completion = json!({
        "finish_reason": match result.finish { Finish::Stop => "stop", Finish::Length => "length" },
        "prompt_tokens": result.prompt_tokens,
        "generated_tokens": result.generated_tokens,
        "reused_tokens": result.reused_tokens,
        "backend_requested": backend_requested,
        "state_budget_bytes": arguments.state_bytes,
        "prefill_block_tokens": arguments.prefill.block_tokens,
        "stop_tokens": generation.stop_tokens,
        "sampling": arguments.sampling.record(),
        "backend_last_operation": backend_record(runtime.decoder().backend_diagnostics()),
        "backend_statistics": statistics_record(runtime.decoder().backend_statistics()),
    });
    if arguments.text_output {
        completion["text"] = json!(text_output.finish());
    }
    if arguments.prompt.is_some() {
        completion["input_mode"] = json!("text");
        completion["insert_special"] = json!(arguments.insert_special.name());
        completion["special_input"] = json!(match arguments.special_input {
            SpecialInputPolicy::Literal => "literal",
            SpecialInputPolicy::Parse => "parse",
        });
    }
    serde_json::to_writer(&mut output, &completion)?;
    output.write_all(b"\n")?;
    output.flush()?;
    Ok(())
}

fn tokenize_record(arguments: &Arguments) -> Result<serde_json::Value> {
    let (tokens, identity) =
        if let Some(prompt) = &arguments.prompt {
            let session = TokenizationSession::load(
                &arguments.path,
                arguments.bytes,
                &arguments.sha256,
                arguments.memory,
            )?;
            let result = session.encode(
                prompt,
                TextTokenizationPolicy {
                    special_input: arguments.special_input,
                    insert_special: match arguments.insert_special {
                        InsertSpecial::None => SpecialTokenInsertion::None,
                        InsertSpecial::Model => SpecialTokenInsertion::Model,
                    },
                    budget: TokenBudget {
                        context: arguments.context,
                        reserved_output_tokens: arguments.generation.max_tokens,
                    },
                },
            )?;
            (result.token_ids, session.identity().to_owned())
        } else {
            // Preserve raw-ID mode's metadata-only contract: unlike a text session,
            // it does not require a supported pretokenizer or BPE merge metadata.
            let model = ModelMetadata::load(
                &arguments.path,
                arguments.bytes,
                &arguments.sha256,
                arguments.memory,
            )?;
            let _special_tokens = SpecialTokens::from_metadata(&model.metadata, model.vocab_size)?;
            if arguments.tokens.iter().any(|&token| {
                usize::try_from(token).map_or(true, |token| token >= model.vocab_size)
            }) {
                return Err(bad("input token is outside the model vocabulary"));
            }
            (arguments.tokens.clone(), model.identity)
        };
    Ok(json!({
        "mode": "tokenize",
        "input_mode": if arguments.prompt.is_some() { "text" } else { "tokens" },
        "token_count": tokens.len(),
        "token_ids": tokens,
        "special_input": arguments.prompt.as_ref().map(|_| match arguments.special_input {
            SpecialInputPolicy::Literal => "literal",
            SpecialInputPolicy::Parse => "parse",
        }),
        "insert_special": arguments.insert_special.name(),
        "artifact": { "bytes": arguments.bytes, "sha256": identity },
        "context": arguments.context,
        "reserved_output_tokens": arguments.generation.max_tokens,
    }))
}

fn encode_prompt(
    arguments: &Arguments,
    prompt: &str,
    metadata: &bonsai_gguf::Metadata,
    vocab_size: usize,
    special_tokens: &SpecialTokens,
) -> Result<Vec<u32>> {
    let tokenizer = Tokenizer::from_metadata(metadata, vocab_size)?;
    let limit = arguments.context - arguments.generation.max_tokens;
    match arguments.insert_special {
        InsertSpecial::None => tokenizer.encode(prompt, limit, arguments.special_input),
        InsertSpecial::Model => tokenizer.encode_with_special_tokens(
            prompt,
            limit,
            arguments.special_input,
            special_tokens,
        ),
    }
}

const fn backend_name(backend: MatvecBackend) -> &'static str {
    match backend {
        MatvecBackend::Auto => "auto",
        MatvecBackend::Scalar => "scalar",
        MatvecBackend::ExperimentalNeon => "neon",
    }
}

fn backend_record(diagnostics: MatvecDiagnostics) -> serde_json::Value {
    json!({
        "selected": backend_name(diagnostics.selected_backend),
        "neon_build_eligible": diagnostics.neon_build_eligible,
        "fallback_reason": diagnostics.fallback_reason.map(|reason| match reason {
            MatvecFallbackReason::ScalarRequested => "scalar_requested",
            MatvecFallbackReason::TensorKindUnsupported => "tensor_kind_unsupported",
            MatvecFallbackReason::FeatureDisabled => "feature_disabled",
            MatvecFallbackReason::TargetUnsupported => "target_unsupported",
        }),
        "dispatched": diagnostics.dispatched,
        "succeeded": diagnostics.succeeded,
    })
}

fn statistics_record(statistics: MatvecStatistics) -> serde_json::Value {
    json!({
        "scope": "decoder_lifetime_including_provisional_work",
        "preparations": statistics.preparations,
        "preparation_failures": statistics.preparation_failures,
        "matrix_attempts": statistics.matrix_attempts,
        "matrix_successes": statistics.matrix_successes,
        "rejected_before_dispatch": statistics.rejected_before_dispatch,
        "scalar_operations": statistics.scalar_operations,
        "neon_operations": statistics.neon_operations,
        "fallbacks": {
            "scalar_requested": statistics.scalar_requested,
            "tensor_kind_unsupported": statistics.tensor_kind_unsupported,
            "feature_disabled": statistics.feature_disabled,
            "target_unsupported": statistics.target_unsupported,
        },
    })
}

fn main() -> ExitCode {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() == 1 && args[0] == "--help" {
        return match std::io::stdout().lock().write_all(USAGE.as_bytes()) {
            Ok(()) => ExitCode::SUCCESS,
            Err(_) => ExitCode::FAILURE,
        };
    }
    if args.len() == 1 && args[0] == "--capabilities" {
        let mut output = std::io::stdout().lock();
        return match serde_json::to_writer(
            &mut output,
            &bonsai_prism_runtime::capabilities::RuntimeCapabilities::current(),
        )
        .map_err(std::io::Error::other)
        .and_then(|()| output.write_all(b"\n"))
        {
            Ok(()) => ExitCode::SUCCESS,
            Err(_) => ExitCode::FAILURE,
        };
    }
    match parse(args).and_then(|arguments| execute(&arguments)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            // No argument, prompt, token sequence or credential is logged.
            let _written = writeln!(std::io::stderr().lock(), "native runtime failed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Duration, InsertSpecial, MAX_STATE_BYTES, MatvecBackend, Mode, Result, SamplingMode,
        SpecialInputPolicy, bad, parse,
    };

    fn arguments() -> Vec<String> {
        [
            "--model".to_owned(),
            std::env::temp_dir()
                .join("unopened-native-model.gguf")
                .to_string_lossy()
                .into_owned(),
            "--bytes".to_owned(),
            "32".to_owned(),
            "--sha256".to_owned(),
            "ab".repeat(32),
            "--memory-bytes".to_owned(),
            "4096".to_owned(),
            "--context".to_owned(),
            "8".to_owned(),
            "--checkpoint-bytes".to_owned(),
            "4096".to_owned(),
            "--max-tokens".to_owned(),
            "1".to_owned(),
            "--tokens".to_owned(),
            "1,2".to_owned(),
        ]
        .into()
    }

    fn replace(arguments: &mut [String], key: &str, value: &str) -> Result<()> {
        let position = arguments
            .iter()
            .position(|argument| argument == key)
            .ok_or_else(|| bad("missing test argument"))?;
        arguments[position + 1] = value.to_owned();
        Ok(())
    }

    fn tokenize_arguments() -> Vec<String> {
        let mut input = arguments();
        for key in ["--checkpoint-bytes", "--max-tokens"] {
            if let Some(index) = input.iter().position(|value| value == key) {
                input.drain(index..=index + 1);
            }
        }
        input.extend(["--mode", "tokenize"].map(str::to_owned));
        input
    }

    #[test]
    fn cli_prompt_file_options_reject_before_reading_stdin() -> Result<()> {
        for pair in [
            ("--backend", "invalid"),
            ("--insert-special", "invalid"),
            ("--special-input", "invalid"),
            ("--prompt", "private prompt sentinel"),
            ("--tokens", "1"),
        ] {
            let mut input = arguments();
            let index = input
                .iter()
                .position(|key| key == "--tokens")
                .ok_or("missing input")?;
            input[index] = "--prompt-file".to_owned();
            input[index + 1] = "-".to_owned();
            input.extend(<[&str; 2]>::from(pair).map(str::to_owned));
            assert!(parse(input).is_err());
        }
        Ok(())
    }

    #[test]
    fn cli_token_file_options_reject_before_reading_stdin() -> Result<()> {
        for pair in [
            ("--backend", "invalid"),
            ("--insert-special", "none"),
            ("--insert-special", "model"),
            ("--special-input", "literal"),
            ("--special-input", "parse"),
            ("--prompt", "private prompt sentinel"),
            ("--prompt-file", "-"),
            ("--tokens", "1"),
            ("--tokens-file", "-"),
        ] {
            let mut input = arguments();
            let index = input
                .iter()
                .position(|key| key == "--tokens")
                .ok_or("missing input")?;
            input[index] = "--tokens-file".to_owned();
            input[index + 1] = "-".to_owned();
            input.extend(<[&str; 2]>::from(pair).map(str::to_owned));
            assert!(parse(input).is_err());
        }
        Ok(())
    }

    #[test]
    fn cli_mode_defaults_to_generation_and_accepts_explicit_selection() -> Result<()> {
        assert_eq!(parse(arguments())?.mode, Mode::Generate);
        let mut input = arguments();
        input.extend(["--mode", "generate"].map(str::to_owned));
        assert_eq!(parse(input)?.mode, Mode::Generate);
        let parsed = parse(tokenize_arguments())?;
        assert_eq!(parsed.mode, Mode::Tokenize);
        assert_eq!(parsed.generation.max_tokens, 0);
        assert_eq!(parsed.checkpoint_bytes, 0);
        for tail in [
            vec!["--mode"],
            vec!["--mode", ""],
            vec!["--mode", "Tokenize"],
            vec!["--mode", "unknown"],
            vec!["--mode", "generate", "--mode", "tokenize"],
        ] {
            let mut input = arguments();
            input.extend(tail.into_iter().map(str::to_owned));
            assert!(parse(input).is_err());
        }
        Ok(())
    }

    #[test]
    fn cli_tokenize_reserves_optional_output_and_accepts_full_input_context() -> Result<()> {
        let mut input = tokenize_arguments();
        replace(&mut input, "--tokens", "0,1,2,3,4,5,6,7")?;
        assert_eq!(parse(input.clone())?.tokens.len(), 8);
        input.extend(["--max-tokens", "0"].map(str::to_owned));
        assert_eq!(parse(input.clone())?.tokens.len(), 8);
        for reserve in ["1", "8", "9", "18446744073709551616"] {
            replace(&mut input, "--max-tokens", reserve)?;
            assert!(parse(input.clone()).is_err());
        }
        replace(&mut input, "--tokens", "0,1")?;
        replace(&mut input, "--max-tokens", "6")?;
        assert_eq!(parse(input.clone())?.generation.max_tokens, 6);
        replace(&mut input, "--max-tokens", "7")?;
        assert!(parse(input).is_err());
        Ok(())
    }

    #[test]
    fn cli_tokenize_rejects_generation_controls_before_model_loading() {
        for (key, value) in [
            ("--checkpoint-bytes", "4096"),
            ("--stop-tokens", "1"),
            ("--backend", "auto"),
            ("--timeout-seconds", "1"),
            ("--output", "tokens"),
            ("--state-bytes", "4096"),
            ("--prefill-block-tokens", "1"),
            ("--sampling", "greedy"),
            ("--seed", "0"),
            ("--temperature", "1"),
            ("--top-k", "20"),
            ("--top-p", "0.95"),
            ("--unknown", "value"),
        ] {
            let mut input = tokenize_arguments();
            input.extend([key, value].map(str::to_owned));
            assert!(parse(input).is_err(), "accepted incompatible {key}");
        }
    }

    #[test]
    fn cli_special_insertion_is_explicit_and_requires_text_in_both_modes() -> Result<()> {
        for mode in ["generate", "tokenize"] {
            for option in ["none", "model"] {
                let mut input = if mode == "generate" {
                    arguments()
                } else {
                    tokenize_arguments()
                };
                input.extend(["--insert-special", option].map(str::to_owned));
                assert!(parse(input.clone()).is_err());
                let index = input
                    .iter()
                    .position(|value| value == "--tokens")
                    .ok_or("missing fixture input")?;
                input[index] = "--prompt".to_owned();
                input[index + 1] = "ab".to_owned();
                assert_eq!(
                    parse(input)?.insert_special,
                    if option == "model" {
                        InsertSpecial::Model
                    } else {
                        InsertSpecial::None
                    }
                );
            }
        }
        let mut input = arguments();
        let index = input
            .iter()
            .position(|value| value == "--tokens")
            .ok_or("missing fixture input")?;
        input[index] = "--prompt".to_owned();
        input[index + 1] = "ab".to_owned();
        assert_eq!(parse(input.clone())?.insert_special, InsertSpecial::None);
        for tail in [
            vec!["--insert-special"],
            vec!["--insert-special", ""],
            vec!["--insert-special", "Model"],
            vec!["--insert-special", "invalid"],
            vec!["--insert-special", "none", "--insert-special", "model"],
        ] {
            let mut invalid = input.clone();
            invalid.extend(tail.into_iter().map(str::to_owned));
            assert!(parse(invalid).is_err());
        }
        Ok(())
    }

    #[test]
    fn cli_text_input_requires_one_input_and_explicit_special_policy() -> Result<()> {
        let mut input = arguments();
        let position = input
            .iter()
            .position(|key| key == "--tokens")
            .ok_or("missing tokens")?;
        input[position] = "--prompt".to_owned();
        input[position + 1] = "a<stop>".to_owned();
        let parsed = parse(input.clone())?;
        assert_eq!(parsed.prompt.as_deref(), Some("a<stop>"));
        assert_eq!(parsed.tokens, Vec::<u32>::new());
        assert!(matches!(parsed.special_input, SpecialInputPolicy::Literal));
        input.extend(["--special-input", "parse"].map(str::to_owned));
        assert!(matches!(
            parse(input)?.special_input,
            SpecialInputPolicy::Parse
        ));
        for tail in [
            vec!["--prompt", "a"],
            vec!["--special-input", "literal"],
            vec!["--special-input", "parse"],
        ] {
            let mut input = arguments();
            input.extend(tail.into_iter().map(str::to_owned));
            assert!(parse(input).is_err());
        }
        for text in [String::new(), "a".repeat(1_048_577)] {
            let mut input = arguments();
            input[position] = "--prompt".to_owned();
            input[position + 1] = text;
            assert!(parse(input).is_err());
        }
        let mut input = arguments();
        input.truncate(position);
        assert!(parse(input.clone()).is_err());
        input.extend(["--prompt", "a", "--special-input", "auto"].map(str::to_owned));
        assert!(parse(input).is_err());
        Ok(())
    }

    #[test]
    fn cli_sampling_requires_explicit_mode_and_seed_and_records_effective_options() -> Result<()> {
        assert!(matches!(parse(arguments())?.sampling, SamplingMode::Greedy));
        let mut input = arguments();
        input.extend(["--sampling", "reference", "--seed", "42"].map(str::to_owned));
        let mode = parse(input)?.sampling;
        assert_eq!(mode.record()["mode"], "reference");
        assert_eq!(mode.record()["rng"], "splitmix64");
        assert_eq!(mode.record()["seed"], 42);
        assert_eq!(mode.record()["temperature"], 1.0);
        assert_eq!(mode.record()["top_k"], 20);
        assert_eq!(mode.record()["top_p"], serde_json::json!(0.95_f32));
        let mut input = arguments();
        input.extend(
            [
                "--sampling",
                "reference",
                "--seed",
                "18446744073709551615",
                "--temperature",
                "0.7",
                "--top-k",
                "0",
                "--top-p",
                "1",
            ]
            .map(str::to_owned),
        );
        let mode = parse(input)?.sampling;
        assert_eq!(mode.record()["seed"], u64::MAX);
        assert_eq!(mode.record()["temperature"], serde_json::json!(0.7_f32));
        assert_eq!(mode.record()["top_k"], 0);
        assert_eq!(mode.record()["top_p"], 1.0);
        Ok(())
    }

    #[test]
    fn cli_sampling_rejects_inactive_controls_and_invalid_configuration_before_loading() {
        for tail in [
            vec!["--sampling", "reference"],
            vec!["--sampling", "Reference", "--seed", "1"],
            vec!["--seed", "1"],
            vec!["--sampling", "greedy", "--temperature", "1"],
            vec!["--sampling", "greedy", "--top-k", "1"],
            vec!["--sampling", "greedy", "--top-p", "1"],
            vec!["--sampling", "reference", "--seed", "-1"],
            vec!["--sampling", "reference", "--seed", "+1"],
            vec!["--sampling", "reference", "--seed", "18446744073709551616"],
            vec!["--sampling", "reference", "--seed", "1", "--seed", "2"],
        ] {
            let mut input = arguments();
            input.extend(tail.into_iter().map(str::to_owned));
            assert!(parse(input).is_err());
        }
        for (key, value) in [
            ("--temperature", "0"),
            ("--temperature", "NaN"),
            ("--temperature", "inf"),
            ("--top-k", "1048577"),
            ("--top-k", "+1"),
            ("--top-k", "-1"),
            ("--top-p", "0"),
            ("--top-p", "1.01"),
            ("--top-p", "NaN"),
        ] {
            let mut input = arguments();
            input.extend(["--sampling", "reference", "--seed", "1", key, value].map(str::to_owned));
            assert!(parse(input).is_err(), "accepted invalid {key}");
        }
    }

    #[test]
    fn cli_text_output_is_explicit_and_rejects_invalid_options() -> Result<()> {
        assert!(!parse(arguments())?.text_output);
        for (value, expected) in [("tokens", false), ("text", true)] {
            let mut input = arguments();
            input.extend(["--output".to_owned(), value.to_owned()]);
            assert_eq!(parse(input)?.text_output, expected);
        }
        for tail in [
            vec!["--output"],
            vec!["--output", ""],
            vec!["--output", "Text"],
            vec!["--output", "text", "--output", "tokens"],
        ] {
            let mut input = arguments();
            input.extend(tail.into_iter().map(str::to_owned));
            assert!(parse(input).is_err());
        }
        Ok(())
    }

    #[test]
    fn cli_preflight_rejects_invalid_identity_and_resource_bounds() -> Result<()> {
        for (key, value) in [
            ("--bytes", "0"),
            ("--sha256", "bad"),
            ("--sha256", &"z".repeat(64)),
            ("--memory-bytes", "0"),
            ("--context", "0"),
            ("--context", "262145"),
            ("--checkpoint-bytes", "0"),
            ("--max-tokens", "0"),
            ("--max-tokens", "8"),
            ("--max-tokens", &usize::MAX.to_string()),
        ] {
            let mut input = arguments();
            replace(&mut input, key, value)?;
            assert!(parse(input).is_err(), "accepted invalid {key}");
        }
        Ok(())
    }

    #[test]
    fn cli_preflight_bounds_prompt_and_stop_lists_before_model_loading() -> Result<()> {
        let mut input = arguments();
        replace(&mut input, "--tokens", "0,1,2,3,4,5,6")?;
        assert!(parse(input.clone()).is_ok());
        replace(&mut input, "--tokens", "0,1,2,3,4,5,6,7")?;
        assert!(parse(input).is_err());

        for count in [256, 257] {
            let mut input = arguments();
            input.extend(["--stop-tokens".to_owned(), vec!["1"; count].join(",")]);
            assert_eq!(parse(input).is_ok(), count == 256);
        }
        for tokens in ["", "1,", "-1", "4294967296"] {
            let mut input = arguments();
            replace(&mut input, "--tokens", tokens)?;
            assert!(parse(input).is_err());
        }
        Ok(())
    }

    #[test]
    fn cli_preflight_accepts_exact_context_boundary_and_uppercase_digest() -> Result<()> {
        let mut input = arguments();
        replace(&mut input, "--sha256", &"AB".repeat(32))?;
        replace(&mut input, "--max-tokens", "6")?;
        let parsed = parse(input)?;
        assert_eq!(parsed.tokens.len() + parsed.generation.max_tokens, 8);
        assert_eq!(parsed.generation.stop_tokens.len(), 0);
        Ok(())
    }

    #[test]
    fn cli_backend_selection_defaults_to_auto_and_accepts_explicit_requests() -> Result<()> {
        assert_eq!(parse(arguments())?.backend, MatvecBackend::Auto);
        for (value, expected) in [
            ("auto", MatvecBackend::Auto),
            ("scalar", MatvecBackend::Scalar),
            ("neon", MatvecBackend::ExperimentalNeon),
        ] {
            let mut input = arguments();
            input.extend(["--backend".to_owned(), value.to_owned()]);
            assert_eq!(parse(input)?.backend, expected);
        }
        Ok(())
    }

    #[test]
    fn cli_prefill_blocks_validate_bounds_before_model_loading() -> Result<()> {
        assert_eq!(parse(arguments())?.prefill.block_tokens, 1);
        for tokens in ["1", "2", "256"] {
            let mut input = arguments();
            input.extend(["--prefill-block-tokens".to_owned(), tokens.to_owned()]);
            assert_eq!(parse(input)?.prefill.block_tokens, tokens.parse::<usize>()?);
        }
        for tokens in ["0", "257", "-1", "+1", "1.5", "", "18446744073709551616"] {
            let mut input = arguments();
            input.extend(["--prefill-block-tokens".to_owned(), tokens.to_owned()]);
            assert!(parse(input).is_err());
        }
        let mut input = arguments();
        input.push("--prefill-block-tokens".to_owned());
        assert!(parse(input).is_err());
        let mut input = arguments();
        input.extend(
            ["--prefill-block-tokens", "2", "--prefill-block-tokens", "4"].map(str::to_owned),
        );
        assert!(parse(input).is_err());
        Ok(())
    }

    #[test]
    fn cli_backend_selection_rejects_invalid_repeated_and_missing_values() {
        for value in ["", "NEON", "Scalar", "avx2", "neon ", "--scalar"] {
            let mut input = arguments();
            input.extend(["--backend".to_owned(), value.to_owned()]);
            assert!(parse(input).is_err(), "accepted invalid backend {value}");
        }
        let mut input = arguments();
        input.push("--backend".to_owned());
        assert!(parse(input).is_err());
        for second in ["auto", "scalar", "neon"] {
            let mut input = arguments();
            input.extend(["--backend", "scalar", "--backend", second].map(str::to_owned));
            assert!(parse(input).is_err());
        }
    }

    #[test]
    fn cli_state_budget_accepts_explicit_bounds_and_preserves_default() -> Result<()> {
        assert_eq!(parse(arguments())?.state_bytes, MAX_STATE_BYTES);
        for bytes in [1, 4096, MAX_STATE_BYTES] {
            let mut input = arguments();
            input.extend(["--state-bytes".to_owned(), bytes.to_string()]);
            assert_eq!(parse(input)?.state_bytes, bytes);
        }
        Ok(())
    }

    #[test]
    fn cli_state_budget_rejects_invalid_repeated_and_missing_values() {
        for value in [
            "",
            "0",
            "-1",
            "1.5",
            " 1",
            "4294967297",
            "18446744073709551616",
        ] {
            let mut input = arguments();
            input.extend(["--state-bytes".to_owned(), value.to_owned()]);
            assert!(parse(input).is_err());
        }
        let mut input = arguments();
        input.push("--state-bytes".to_owned());
        assert!(parse(input).is_err());
        let mut input = arguments();
        input.extend(["--state-bytes", "4096", "--state-bytes", "8192"].map(str::to_owned));
        assert!(parse(input).is_err());
    }

    #[test]
    fn cli_timeout_accepts_bounded_seconds_and_preserves_default() -> Result<()> {
        assert_eq!(parse(arguments())?.timeout, None);
        for seconds in [1, 3600] {
            let mut input = arguments();
            input.extend(["--timeout-seconds".to_owned(), seconds.to_string()]);
            assert_eq!(parse(input)?.timeout, Some(Duration::from_secs(seconds)));
        }
        Ok(())
    }

    #[test]
    fn cli_timeout_rejects_invalid_repeated_and_missing_values() {
        for value in [
            "",
            "0",
            "3601",
            "-1",
            "+1",
            "1.5",
            " 1",
            "1s",
            "18446744073709551616",
        ] {
            let mut input = arguments();
            input.extend(["--timeout-seconds".to_owned(), value.to_owned()]);
            assert!(parse(input).is_err());
        }
        let mut input = arguments();
        input.push("--timeout-seconds".to_owned());
        assert!(parse(input).is_err());
        let mut input = arguments();
        input.extend(["--timeout-seconds", "1", "--timeout-seconds", "2"].map(str::to_owned));
        assert!(parse(input).is_err());
    }
}
