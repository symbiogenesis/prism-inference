//! Bounded correctness capture against an already owned released private service.
//!
//! The external owner retains artifact handles, listener/process identity, the
//! sandbox, Job and aggregate setup/loading/check/cleanup deadline. This checker
//! starts no server, loads no weights and changes no runtime selection.
#![forbid(unsafe_code)]

use bonsai_prism_runtime::model::{lora::SWIFT_ADAPTER_SHA256, swift};
use serde_json::{Value, json};
use std::{
    fs::{File, OpenOptions},
    io::{self, Read, Seek, Write},
    net::{Ipv4Addr, SocketAddr, TcpStream},
    path::Path,
    process::ExitCode,
    time::{Duration, Instant},
};

const MAX_REPLY: usize = 1024 * 1024;
const HEADER_LIMIT: usize = 16 * 1024;
const PROMPT: &str = "Continue this story for several paragraphs: The courier opened the blue door and found a letter addressed to tomorrow.";
type Check<T> = io::Result<T>;

fn invalid() -> io::Error {
    io::Error::other("private service acceptance check failed")
}

fn require(condition: bool) -> Check<()> {
    if condition { Ok(()) } else { Err(invalid()) }
}

struct Client<'a> {
    port: u16,
    key: &'a str,
    deadline: Instant,
}

impl Client<'_> {
    fn remaining(&self) -> Check<Duration> {
        self.deadline
            .checked_duration_since(Instant::now())
            .filter(|value| !value.is_zero())
            .ok_or_else(invalid)
    }

    fn start(&self, route: &str, body: Option<&Value>) -> Check<TcpStream> {
        let mut stream = TcpStream::connect_timeout(
            &SocketAddr::from((Ipv4Addr::LOCALHOST, self.port)),
            self.remaining()?.min(Duration::from_secs(10)),
        )?;
        let payload = body
            .map(serde_json::to_vec)
            .transpose()
            .map_err(|_| invalid())?;
        let method = if route == "/reset" || body.is_some() {
            "POST"
        } else {
            "GET"
        };
        let framing = payload.as_ref().map_or_else(String::new, |bytes| {
            format!(
                "Content-Type: application/json\r\nContent-Length: {}\r\n",
                bytes.len()
            )
        });
        let head = format!(
            "{method} {route} HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nAuthorization: Bearer {}\r\nConnection: close\r\n{framing}\r\n",
            self.port, self.key
        );
        self.write(&mut stream, head.as_bytes())?;
        if let Some(bytes) = payload {
            self.write(&mut stream, &bytes)?;
        }
        Ok(stream)
    }

    fn write(&self, stream: &mut TcpStream, mut bytes: &[u8]) -> Check<()> {
        while !bytes.is_empty() {
            stream.set_write_timeout(Some(self.remaining()?))?;
            let count = stream.write(bytes)?;
            require(count > 0)?;
            bytes = &bytes[count..];
        }
        self.remaining().map(|_| ())
    }

    fn read(&self, stream: &mut TcpStream, bytes: &mut Vec<u8>) -> Check<usize> {
        require(bytes.len() < MAX_REPLY)?;
        stream.set_read_timeout(Some(self.remaining()?))?;
        let mut buffer = [0; 4096];
        let capacity = (MAX_REPLY - bytes.len()).min(buffer.len());
        let count = stream.read(&mut buffer[..capacity])?;
        self.remaining()?;
        bytes.extend_from_slice(&buffer[..count]);
        Ok(count)
    }

    fn request(&self, route: &str, body: Option<&Value>, status: u16) -> Check<Value> {
        let mut stream = self.start(route, body)?;
        let mut bytes = Vec::new();
        while self.read(&mut stream, &mut bytes)? != 0 {}
        let (actual, payload, complete) = decode(&bytes)?;
        require(actual == status && complete)?;
        serde_json::from_slice(&payload).map_err(|_| invalid())
    }

    fn count(&self, body: &Value) -> Check<u64> {
        let value = self.request("/v1/chat/completions/input_tokens", Some(body), 200)?;
        value["input_tokens"]
            .as_u64()
            .filter(|n| *n > 0 && *n < 8192)
            .ok_or_else(invalid)
    }

    fn complete(
        &self,
        body: &Value,
        evidence: &mut Evidence,
        count_step: &'static str,
    ) -> Check<Value> {
        let count = self.count(body)?;
        evidence.step(count_step)?;
        let value = self.request("/v1/chat/completions", Some(body), 200)?;
        require(value["usage"]["prompt_tokens"].as_u64() == Some(count))?;
        let output = value["usage"]["completion_tokens"]
            .as_u64()
            .ok_or_else(invalid)?;
        require(output <= body["max_completion_tokens"].as_u64().ok_or_else(invalid)?)?;
        require(value["usage"]["total_tokens"].as_u64() == count.checked_add(output))?;
        require(matches!(
            value["choices"][0]["finish_reason"].as_str(),
            Some("stop" | "length")
        ))?;
        require(value["choices"][0]["message"]["content"].is_string())?;
        Ok(value)
    }

    fn abandon_provisional_stream(&self, body: &Value, evidence: &mut Evidence) -> Check<()> {
        self.count(body)?;
        evidence.step("cancellation-stream-canonical-count")?;
        let mut stream = self.start("/v1/chat/completions", Some(body))?;
        let mut bytes = Vec::new();
        loop {
            require(self.read(&mut stream, &mut bytes)? > 0)?;
            let Some(head_end) = boundary(&bytes, b"\r\n\r\n") else {
                require(bytes.len() < HEADER_LIMIT)?;
                continue;
            };
            require(head_end <= HEADER_LIMIT)?;
            let (status, payload, complete) = decode(&bytes)?;
            require(status == 200 && !complete)?;
            if provisional(&payload)? {
                return Ok(());
            }
        }
    }
}

fn boundary(bytes: &[u8], needle: &[u8]) -> Option<usize> {
    bytes
        .windows(needle.len())
        .position(|part| part == needle)
        .map(|n| n + needle.len())
}

/// Decode complete chunks only; incomplete payloads stay provisional.
fn chunks(mut bytes: &[u8]) -> Check<(Vec<u8>, bool)> {
    let mut output = Vec::new();
    loop {
        let Some(end) = boundary(bytes, b"\r\n") else {
            return Ok((output, false));
        };
        let line = &bytes[..end - 2];
        require(!line.is_empty() && line.len() <= 8 && line.iter().all(u8::is_ascii_hexdigit))?;
        let count = usize::from_str_radix(std::str::from_utf8(line).map_err(|_| invalid())?, 16)
            .map_err(|_| invalid())?;
        require(count <= MAX_REPLY)?;
        bytes = &bytes[end..];
        if bytes.len() < count + 2 {
            return Ok((output, false));
        }
        require(&bytes[count..count + 2] == b"\r\n")?;
        if count == 0 {
            require(bytes.len() == 2)?;
            return Ok((output, true));
        }
        require(output.len() + count <= MAX_REPLY)?;
        output.extend_from_slice(&bytes[..count]);
        bytes = &bytes[count + 2..];
    }
}

fn decode(bytes: &[u8]) -> Check<(u16, Vec<u8>, bool)> {
    let end = boundary(bytes, b"\r\n\r\n").ok_or_else(invalid)?;
    require(end <= HEADER_LIMIT)?;
    let mut headers = [httparse::EMPTY_HEADER; 64];
    let mut response = httparse::Response::new(&mut headers);
    require(
        matches!(response.parse(&bytes[..end]), Ok(httparse::Status::Complete(n)) if n == end),
    )?;
    require(response.version == Some(1))?;
    let status = response.code.ok_or_else(invalid)?;
    let field = |name: &str| -> Check<Option<&[u8]>> {
        let mut found = response
            .headers
            .iter()
            .filter(|h| h.name.eq_ignore_ascii_case(name));
        let value = found.next().map(|h| h.value);
        require(found.next().is_none())?;
        Ok(value)
    };
    let payload = &bytes[end..];
    if let Some(encoding) = field("transfer-encoding")? {
        require(encoding.eq_ignore_ascii_case(b"chunked") && field("content-length")?.is_none())?;
        require(field("content-type")? == Some(b"text/event-stream".as_slice()))?;
        let (decoded, complete) = chunks(payload)?;
        return Ok((status, decoded, complete));
    }
    let length = field("content-length")?.ok_or_else(invalid)?;
    require(!length.is_empty() && length.iter().all(u8::is_ascii_digit))?;
    let length = std::str::from_utf8(length)
        .map_err(|_| invalid())?
        .parse::<usize>()
        .map_err(|_| invalid())?;
    require(payload.len() <= length && length <= MAX_REPLY)?;
    Ok((status, payload.to_vec(), payload.len() == length))
}

fn provisional(bytes: &[u8]) -> Check<bool> {
    let text = std::str::from_utf8(bytes).map_err(|_| invalid())?;
    let mut observed = false;
    for frame in text.split_inclusive("\n\n").filter(|s| s.ends_with("\n\n")) {
        let data = frame
            .trim_end_matches('\n')
            .strip_prefix("data: ")
            .ok_or_else(invalid)?;
        require(data != "[DONE]")?;
        let event: Value = serde_json::from_str(data).map_err(|_| invalid())?;
        require(event["choices"][0]["finish_reason"].is_null() && event.get("usage").is_none())?;
        observed |= ["content", "reasoning_content"].into_iter().any(|field| {
            event["choices"][0]["delta"][field]
                .as_str()
                .is_some_and(|s| !s.is_empty())
        });
    }
    Ok(observed)
}

fn body(scale: f64, stream: bool, limit: u32) -> Value {
    json!({"messages":[{"role":"user","content":PROMPT}],"max_completion_tokens":limit,
        "stream":stream,"n":1,"cache_prompt":false,"n_cache_reuse":0,"ignore_eos":false,
        "temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,
        "frequency_penalty":0.0,"repeat_penalty":1.0,
        "chat_template_kwargs":{"enable_thinking":false,"reasoning_effort":"xhigh","preserve_thinking":true},
        "lora":[{"id":0,"scale":scale}]})
}

struct Evidence {
    file: File,
    steps: Vec<&'static str>,
    readiness: Option<Value>,
}

impl Evidence {
    fn write(&mut self, status: &str) -> Check<()> {
        let value = json!({"schemaVersion":1,"status":status,"completedSteps":self.steps,
            "readiness":self.readiness,
            "scope":"original-pq2-approved-adapter-reset-every-request-private-http",
            "fullStateNumericalParityQualified":false,"performanceQualified":false,
            "activeGraphCancellationQualified":false,"completeStateRollbackQualified":false,
            "processOwnershipQualified":false,"defaultPromotionQualified":false});
        self.file.rewind()?;
        serde_json::to_writer_pretty(&mut self.file, &value).map_err(|_| invalid())?;
        let length = self.file.stream_position()?;
        self.file.set_len(length)?;
        self.file.sync_all()
    }

    fn step(&mut self, name: &'static str) -> Check<()> {
        self.steps.push(name);
        self.write("pending")
    }
}

fn readiness_identity(ready: &Value) -> Check<Value> {
    let model = &ready["data"][0];
    require(ready["data"].as_array().is_some_and(|a| a.len() == 1))?;
    require(
        model["model_sha256"] == swift::MODEL_SHA256
            && model["template_sha256"] == swift::TEMPLATE_SHA256,
    )?;
    require(
        model["adapter_sha256"] == SWIFT_ADAPTER_SHA256
            && model["weight_representation"] == "PQ2_0",
    )?;
    require(
        model["supports_abliteration_scale"] == true
            && model["abliteration_scale_range"] == json!([0, 2]),
    )?;
    require(model.get("prompt_reuse").is_none())?;
    require(
        model["id"] == "Swift-Bonsai-2-PQ2_0"
            && model["runtime"] == "prism-inference"
            && model["context_tokens"] == 8192,
    )?;
    let seed = model["seed"].as_u64().ok_or_else(invalid)?;
    require(u32::try_from(seed).is_ok() && seed != u64::from(u32::MAX))?;
    Ok(
        json!({"modelSha256":swift::MODEL_SHA256,"templateSha256":swift::TEMPLATE_SHA256,
        "adapterSha256":SWIFT_ADAPTER_SHA256,"seed":seed,"contextTokens":8192,
        "weightRepresentation":"PQ2_0","servingStatePolicy":"reset-every-request"}),
    )
}

fn exercise(client: &Client<'_>, evidence: &mut Evidence) -> Check<()> {
    let ready = client.request("/v1/models", None, 200)?;
    evidence.readiness = Some(readiness_identity(&ready)?);
    evidence.step("exact-pq2-adapter-readiness")?;
    let baseline = body(1.3, false, 4);
    client.request("/reset", None, 200)?;
    evidence.step("initial-explicit-reset")?;
    let first = client.complete(&baseline, evidence, "scale-1.3-canonical-count")?;
    evidence.step("scale-1.3-count-generation-usage")?;
    client.complete(&body(0.0, false, 4), evidence, "scale-zero-canonical-count")?;
    evidence.step("scale-zero-count-generation-usage")?;
    client.request(
        "/v1/chat/completions/input_tokens",
        Some(&body(2.01, false, 4)),
        400,
    )?;
    evidence.step("invalid-scale-rejection")?;
    client.count(&baseline)?;
    evidence.step("stale-control-scale-1.3-count")?;
    client.count(&body(0.0, false, 4))?;
    evidence.step("stale-control-replacement-scale-zero-count")?;
    client.request("/v1/chat/completions", Some(&baseline), 400)?;
    evidence.step("replaced-count-stale-generation-rejection")?;
    client.request("/reset", None, 200)?;
    evidence.step("stale-control-explicit-reset")?;
    client.request("/v1/chat/completions", Some(&baseline), 400)?;
    evidence.step("reset-invalidates-counted-input")?;
    client.abandon_provisional_stream(&body(1.3, true, 4096), evidence)?;
    evidence.step("provisional-stream-disconnected-before-received-terminal")?;
    client.request("/v1/chat/completions/input_tokens", Some(&baseline), 400)?;
    evidence.step("provisional-disconnect-closes-worker-admission")?;
    client.request("/reset", None, 200)?;
    evidence.step("recovery-explicit-reset")?;
    let replay = client.complete(&baseline, evidence, "recovery-scale-1.3-canonical-count")?;
    require(replay == first)?;
    evidence.step("explicit-reset-seeded-response-replay")?;
    client.remaining()?;
    evidence.write("passed")?;
    client.remaining().map(|_| ())
}

fn run() -> Check<()> {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    require(arguments.len() == 4)?;
    let port = arguments[0]
        .to_str()
        .ok_or_else(invalid)?
        .parse::<u16>()
        .map_err(|_| invalid())?;
    require(port >= 1024)?;
    let key_path = Path::new(&arguments[1]);
    let evidence_path = Path::new(&arguments[2]);
    require(key_path.is_absolute() && evidence_path.is_absolute())?;
    let seconds = arguments[3]
        .to_str()
        .ok_or_else(invalid)?
        .parse::<u64>()
        .map_err(|_| invalid())?;
    require((1..=240).contains(&seconds))?;
    let deadline = Instant::now() + Duration::from_secs(seconds);
    let key = credential(key_path)?;
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(evidence_path)?;
    let mut evidence = Evidence {
        file,
        steps: Vec::new(),
        readiness: None,
    };
    evidence.write("pending")?;
    let result = exercise(
        &Client {
            port,
            key: key.as_hex(),
            deadline,
        },
        &mut evidence,
    );
    if result.is_err() {
        evidence.write("failed")?;
    }
    result
}

#[cfg(windows)]
fn credential(path: &Path) -> Check<bonsai_windows_credentials::RetainedCredential> {
    bonsai_windows_credentials::RetainedCredential::open(path)
}

#[cfg(not(windows))]
struct Credential(String);

#[cfg(not(windows))]
impl Credential {
    fn as_hex(&self) -> &str {
        &self.0
    }
}

#[cfg(not(windows))]
fn credential(path: &Path) -> Check<Credential> {
    let mut bytes = Vec::new();
    File::open(path)?.take(65).read_to_end(&mut bytes)?;
    require(
        bytes.len() == 64
            && bytes
                .iter()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b)),
    )?;
    Ok(Credential(String::from_utf8(bytes).map_err(|_| invalid())?))
}

fn main() -> ExitCode {
    if run().is_ok() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readiness_binds_the_exact_profile_and_observed_seed() -> Check<()> {
        let ready = json!({"data":[{"id":"Swift-Bonsai-2-PQ2_0","runtime":"prism-inference",
            "model_sha256":swift::MODEL_SHA256,"template_sha256":swift::TEMPLATE_SHA256,
            "adapter_sha256":SWIFT_ADAPTER_SHA256,"weight_representation":"PQ2_0",
            "supports_abliteration_scale":true,"abliteration_scale_range":[0,2],"context_tokens":8192,"seed":42}]});
        assert_eq!(readiness_identity(&ready)?["seed"], 42);
        for (field, value) in [
            ("model_sha256", json!("wrong")),
            ("adapter_sha256", Value::Null),
            ("runtime", json!("other")),
            ("context_tokens", json!(4096)),
            ("seed", json!(u32::MAX)),
            ("prompt_reuse", json!("latest")),
        ] {
            let mut changed = ready.clone();
            changed["data"][0][field] = value;
            assert!(readiness_identity(&changed).is_err());
        }
        Ok(())
    }

    #[test]
    fn failed_capture_preserves_completed_steps_without_qualifying_missing_work() -> Check<()> {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|_| invalid())?
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "prism-service-check-{}-{unique}.json",
            std::process::id()
        ));
        let result = (|| {
            let file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)?;
            let mut evidence = Evidence {
                file,
                steps: Vec::new(),
                readiness: None,
            };
            evidence.step("exact-pq2-adapter-readiness")?;
            evidence.write("failed")?;
            drop(evidence);
            let value: Value =
                serde_json::from_reader(File::open(&path)?).map_err(|_| invalid())?;
            assert_eq!(value["status"], "failed");
            assert_eq!(
                value["completedSteps"],
                json!(["exact-pq2-adapter-readiness"])
            );
            for field in [
                "activeGraphCancellationQualified",
                "completeStateRollbackQualified",
                "defaultPromotionQualified",
            ] {
                assert_eq!(value[field], false);
            }
            Ok(())
        })();
        let cleanup = std::fs::remove_file(path);
        result.and(cleanup)
    }

    #[test]
    fn incomplete_chunks_never_count_as_terminal_or_provisional_text() -> Check<()> {
        let frame =
            b"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}\n\n";
        let mut chunk = format!("{:x}\r\n", frame.len()).into_bytes();
        chunk.extend_from_slice(frame);
        chunk.extend_from_slice(b"\r\n");
        for end in 0..chunk.len() {
            let (payload, complete) = chunks(&chunk[..end])?;
            assert!(!complete);
            assert!(!provisional(&payload)?);
        }
        let (payload, complete) = chunks(&chunk)?;
        assert!(!complete);
        assert!(provisional(&payload)?);
        chunk.extend_from_slice(b"0\r\n\r\n");
        assert!(chunks(&chunk)?.1);
        assert!(chunks(b"ffffffff\r\n").is_err());
        assert!(chunks(b"0\r\n\r\ntrailing").is_err());
        Ok(())
    }

    #[test]
    fn natural_completion_is_never_qualified_as_active_cancellation() -> Check<()> {
        for payload in [
            "data: [DONE]\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":null}],\"usage\":{}}\n\n",
        ] {
            assert!(provisional(payload.as_bytes()).is_err());
        }
        assert!(!provisional(b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n")?);
        Ok(())
    }

    #[test]
    fn response_framing_rejects_ambiguous_duplicate_and_overflow_headers() -> Check<()> {
        let response = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}";
        assert_eq!(decode(response)?, (200, b"{}".to_vec(), true));
        for response in [
            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nContent-Length: 2\r\n\r\n{}".as_slice(),
            b"HTTP/1.1 200 OK\r\nContent-Length: 99999999\r\n\r\n",
            b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\n{}",
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Length: 0\r\n\r\n",
        ] {
            assert!(decode(response).is_err());
        }
        Ok(())
    }
}
