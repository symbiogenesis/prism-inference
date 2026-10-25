//! Complete safe Rust controller for the pinned `LoRA` conversion/inference shell.
//! Original expected texts remain hash-locked external assets, not source code.
use super::engine::Engine;
use super::lora_inputs::{Inputs, digest, hash_bytes};
use super::model_session::{Client, ModelSession};
use super::{Result, invalid, result};
use serde_json::{Value, json};
use std::fs;
use std::io::Read;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};
const ID: &str = "tests/test-lora-conversion-inference.sh";
const SOURCE_SHA: &str = "69584616380bdaa7f6ee1e53d165408118fb0897e1280f964eb7bc3b932b9c9a";
const ARCHITECTURES: [&str; 3] = ["Gemma2ForCausalLM", "LlamaForCausalLM", "Phi3ForCausalLM"];
const OUTPUT_CAP: u64 = 1024 * 1024;
struct Process(Child);
impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn invoke(client: &Client, operation: Value) -> Result<Value> {
    Ok(client.call(0, operation, None)?.value)
}
fn handle(value: &Value, key: &str) -> Result<u64> {
    value[key]
        .as_u64()
        .filter(|&v| v > 0)
        .ok_or_else(|| invalid(format!("production LoRA {key} unavailable")))
}
fn decode_hex(value: &Value) -> Result<Vec<u8>> {
    let text = value
        .as_str()
        .filter(|v| v.len() <= 2 * 1024 * 1024 && v.len() % 2 == 0)
        .ok_or_else(|| invalid("LoRA token-piece extent differs"))?;
    text.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| {
            let digit = |c: u8| match c {
                b'0'..=b'9' => Ok(c - b'0'),
                b'a'..=b'f' => Ok(c - b'a' + 10),
                _ => Err(invalid("invalid token piece hex")),
            };
            Ok((digit(pair[0])? << 4) | digit(pair[1])?)
        })
        .collect()
}
fn encoded_bytes(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    bytes
        .iter()
        .flat_map(|&v| {
            [
                char::from(DIGITS[usize::from(v >> 4)]),
                char::from(DIGITS[usize::from(v & 15)]),
            ]
        })
        .collect()
}
fn merged(
    engine: &Engine,
    base: &Path,
    adapter: &Path,
    output: &Path,
    directory: &Path,
) -> Result<Value> {
    let executable = engine
        .production_tool("llama-export-lora")?
        .ok_or_else(|| invalid("target lacks production LoRA merger"))?;
    let mut process = Process(
        Command::new(&executable)
            .args(["-m"])
            .arg(base)
            .arg("-o")
            .arg(output)
            .arg("--lora")
            .arg(adapter)
            .current_dir(directory)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?,
    );
    let child = &mut process.0;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| invalid("LoRA exporter stdout missing"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| invalid("LoRA exporter stderr missing"))?;
    let overflow = Arc::new(AtomicBool::new(false));
    let capture = |stream: Box<dyn Read + Send>, overflow: Arc<AtomicBool>| {
        thread::spawn(move || -> std::io::Result<Vec<u8>> {
            let mut data = Vec::new();
            stream.take(OUTPUT_CAP + 1).read_to_end(&mut data)?;
            if u64::try_from(data.len()).unwrap_or(u64::MAX) > OUTPUT_CAP {
                overflow.store(true, Ordering::Release);
            }
            Ok(data)
        })
    };
    let out = capture(Box::new(stdout), Arc::clone(&overflow));
    let err = capture(Box::new(stderr), Arc::clone(&overflow));
    let started = Instant::now();
    let mut limit = None;
    let status = loop {
        if overflow.load(Ordering::Acquire) || started.elapsed() > Duration::from_secs(180) {
            limit = Some("production LoRA exporter exceeded output/time bound");
            let _ = child.kill();
            break child.wait()?;
        }
        if let Some(status) = child.try_wait()? {
            break status;
        }
        thread::sleep(Duration::from_millis(10));
    };
    let stdout = out
        .join()
        .map_err(|_| invalid("LoRA stdout collector panicked"))??;
    let stderr = err
        .join()
        .map_err(|_| invalid("LoRA stderr collector panicked"))??;
    fs::write(directory.join("export.stdout"), &stdout)?;
    fs::write(directory.join("export.stderr"), &stderr)?;
    if limit.is_some()
        || stdout.len() > usize::try_from(OUTPUT_CAP)?
        || stderr.len() > usize::try_from(OUTPUT_CAP)?
        || !status.success()
    {
        return Err(invalid(format!(
            "production LoRA merger failed: {status}; {}",
            limit.unwrap_or("see external exporter logs")
        )));
    }
    let (length, sha) = digest(output)?;
    if !(24..=256 * 1024 * 1024).contains(&length) {
        return Err(invalid("merged production GGUF extent differs"));
    }
    Ok(
        json!({"Tool":executable,"Status":status.code(),"Bytes":length,"Sha256":sha,"StdoutBytes":stdout.len(),"StderrBytes":stderr.len()}),
    )
}
/// Bash command substitution drops trailing LF, then the source trim removes
/// leading POSIX whitespace. Expected inputs are UTF-8, so lengths are characters.
pub(super) fn normalized(value: &str) -> &str {
    value
        .trim_end_matches('\n')
        .trim_start_matches([' ', '\t', '\n', '\r', '\u{b}', '\u{c}'])
        .trim_end_matches('\n')
}
fn first_word(value: &str) -> Result<&str> {
    value
        .lines()
        .next()
        .unwrap_or("")
        .split([' ', '\t', '\n'])
        .find(|s| !s.is_empty())
        .ok_or_else(|| invalid("source expected text has no first word"))
}
pub(super) fn source_prefix(reference: &str, output: &str) -> String {
    reference
        .chars()
        .take(output.chars().count())
        .collect::<String>()
        .trim_end_matches('\n')
        .into()
}
#[allow(
    clippy::too_many_lines,
    reason = "preserve original completion prompt/decode/accept/echo/free schedule as one source path"
)]
fn generate(
    engine: &Engine,
    name: &str,
    model_path: &Path,
    adapter: Option<&Path>,
    prompt: &str,
) -> Result<(String, Value)> {
    let session = ModelSession::open(engine, name)?
        .ok_or_else(|| invalid("target lacks production completion model API"))?;
    let outcome = (|| -> Result<(String, Value)> {
        let client = session.client();
        invoke(&client, json!({"Op":"common-init"}))?;
        let mut arguments = vec![
            "llama-completion".to_owned(),
            "-no-cnv".into(),
            "-m".into(),
            model_path.to_string_lossy().into_owned(),
        ];
        if let Some(path) = adapter {
            arguments.extend(["--lora".into(), path.to_string_lossy().into_owned()]);
        }
        arguments.extend([
            "-p".into(),
            prompt.into(),
            "-n".into(),
            "50".into(),
            "--seed".into(),
            "42".into(),
            "--temp".into(),
            "0".into(),
        ]);
        let params = invoke(
            &client,
            json!({"Op":"params","Initial":{},"Arguments":arguments,"Example":"completion"}),
        )?;
        if params["Parsed"] != true {
            return Err(invalid(
                "production completion argument parser rejected source argv",
            ));
        }
        let params = handle(&params, "Handle")?;
        invoke(&client, json!({"Op":"backend-init"}))?;
        invoke(&client, json!({"Op":"numa-init","Params":params}))?;
        let loaded = invoke(
            &client,
            json!({"Op":"common-model","Params":params,"ModelOnly":false}),
        )?;
        let model = handle(&loaded, "Model")?;
        let context = handle(&loaded, "Context")?;
        let sampler = handle(
            &invoke(
                &client,
                json!({"Op":"common-model-sampler","Model":model,"Sequence":0}),
            )?,
            "Sampler",
        )?;
        let tokenized = invoke(
            &client,
            json!({"Op":"vocab-tokenize","Context":context,"InputsHex":[encoded_bytes(prompt.as_bytes())],"AddSpecial":true,"ParseSpecial":true}),
        )?;
        if !tokenized["Errors"][0].is_null() {
            return Err(invalid("production completion tokenizer failed"));
        }
        let tokens = tokenized["Values"][0]
            .as_array()
            .filter(|v| !v.is_empty() && v.len() <= 64)
            .ok_or_else(|| invalid("source prompt token extent differs"))?
            .clone();
        let mut output = Vec::new();
        for token in &tokens {
            invoke(
                &client,
                json!({"Op":"sampler-accept","Sampler":sampler,"Token":token,"Grammar":false}),
            )?;
            output.extend(decode_hex(
                &invoke(
                    &client,
                    json!({"Op":"token-info","Context":context,"Token":token,"Special":false}),
                )?["PieceHex"],
            )?);
        }
        let batch = handle(
            &invoke(
                &client,
                json!({"Op":"batch-one","Batch":null,"Tokens":tokens,"Positions":null}),
            )?,
            "Batch",
        )?;
        let mut generated = Vec::new();
        let mut decode_count = 0;
        let mut stopped_eog = false;
        for index in 0..50 {
            let decoded = invoke(
                &client,
                json!({"Op":"decode","Context":context,"Batch":batch}),
            )?;
            decode_count += 1;
            if decoded["Code"] != 0 {
                return Err(invalid("production completion decode failed"));
            }
            let token = invoke(
                &client,
                json!({"Op":"sample","Sampler":sampler,"Context":context,"Index":-1}),
            )?["Token"]
                .as_i64()
                .ok_or_else(|| invalid("production completion token absent"))?;
            invoke(
                &client,
                json!({"Op":"sampler-accept","Sampler":sampler,"Token":token,"Grammar":true}),
            )?;
            let info = invoke(
                &client,
                json!({"Op":"token-info","Context":context,"Token":token,"Special":false}),
            )?;
            output.extend(decode_hex(&info["PieceHex"])?);
            generated.push(token);
            if output.len() > 1024 * 1024 {
                return Err(invalid("completion text exceeded1MiB"));
            }
            // The original CLI prints this marker after echoing the EOG piece.
            if info["Eog"] == true {
                output.extend_from_slice(b" [end of text]\n");
                stopped_eog = true;
                break;
            }
            if index != 49 {
                invoke(
                    &client,
                    json!({"Op":"batch-one","Batch":batch,"Tokens":[token],"Positions":null}),
                )?;
            }
        }
        invoke(&client, json!({"Op":"backend-free"}))?;
        for (kind, id) in [
            ("sampler", sampler),
            ("batch", batch),
            ("context", context),
            ("model", model),
            ("params", params),
        ] {
            invoke(&client, json!({"Op":"release","Kind":kind,"Handle":id}))?;
        }
        let output = String::from_utf8(output)?;
        let receipt = json!({"PromptTokens":tokens,"GeneratedTokens":generated,"DecodeCalls":decode_count,"StoppedEog":stopped_eog,"Seed":42,"Temperature":0,"Predict":50,"CompletionExample":true,"ProductionCommonInitialization":true,"OutputBytes":output.len(),"OutputSha256":hash_bytes(output.as_bytes())});
        Ok((normalized(&output).into(), receipt))
    })();
    let finish = session.finish();
    let output = outcome?;
    finish?;
    Ok(output)
}
#[allow(
    clippy::unnecessary_wraps,
    reason = "all source ports expose the same Result interface and retain measured failures"
)]
pub(super) fn run(engine: &Engine) -> Result<Value> {
    let outcome = (|| -> Result<Value> {
        let inputs = Inputs::verify(engine)?;
        let base_full = fs::read_to_string(inputs.path("data/pale_blue_dot.txt")?)?;
        let base_full = base_full.trim_end_matches('\n');
        let lora_full = fs::read_to_string(inputs.path("data/bohemian_rhapsody.txt")?)?;
        let lora_full = lora_full.trim_end_matches('\n');
        let base_prompt = first_word(base_full)?;
        let lora_prompt = first_word(lora_full)?;
        let mut receipts = Vec::new();
        let mut assertions = 0;
        for architecture in ARCHITECTURES {
            let directory = engine.evidence_path(&format!("lora-{architecture}"))?;
            let converted = super::lora_convert::convert(&inputs, architecture, &directory)?;
            let output_path = directory.join("Base-F32-lora-merged.gguf");
            let merge = merged(
                engine,
                &converted.base,
                &converted.adapter,
                &output_path,
                &directory,
            )?;
            let (base, base_receipt) = generate(
                engine,
                &format!("lora-{architecture}-base"),
                &converted.base,
                None,
                base_prompt,
            )?;
            let (hot, hot_receipt) = generate(
                engine,
                &format!("lora-{architecture}-hot"),
                &converted.base,
                Some(&converted.adapter),
                lora_prompt,
            )?;
            let (merged, merged_receipt) = generate(
                engine,
                &format!("lora-{architecture}-merged"),
                &output_path,
                None,
                lora_prompt,
            )?;
            let expected_base = source_prefix(base_full, &base);
            let expected_lora = source_prefix(lora_full, &hot);
            for (label, actual, expected) in [
                ("base", &base, &expected_base),
                ("hot", &hot, &expected_lora),
                ("merged", &merged, &expected_lora),
            ] {
                assertions += 1;
                if actual != expected {
                    return Err(invalid(format!(
                        "{architecture} {label}: original expected text prefix differs (actual {} characters, expected {}); see token/hash receipts",
                        actual.chars().count(),
                        expected.chars().count()
                    )));
                }
            }
            receipts.push(json!({"Conversion":converted.receipt,"Merge":merge,"Base":base_receipt,"Hot":hot_receipt,"Merged":merged_receipt,"PrefixPredicates":3,"MergedExpectedLengthUsesHotOutput":true}));
            fs::write(
                directory.join("receipt.json"),
                serde_json::to_vec_pretty(
                    receipts.last().ok_or_else(|| invalid("receipt missing"))?,
                )?,
            )?;
        }
        Ok(
            json!({"SourceSHA256":SOURCE_SHA,"SourcePrefixPredicates":assertions,"Inputs":inputs.receipt,"Architectures":receipts,"PythonTestExecution":false,"InferenceOwner":"safe-rust","ProductionMergeTool":"llama-export-lora"}),
        )
    })();
    match outcome {
        Ok(value) => Ok(result(ID, true, value.to_string())),
        Err(error) => Ok(result(
            ID,
            false,
            format!("complete LoRA conversion/inference source contract: {error}"),
        )),
    }
}
