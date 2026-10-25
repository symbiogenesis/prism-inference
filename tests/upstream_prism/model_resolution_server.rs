//! Bounded safe Rust HF fixture server, with per-request transport evidence.

use super::super::{Result, invalid};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

pub(super) type Repositories = BTreeMap<String, Vec<String>>;
struct State {
    repos: Repositories,
    requests: Vec<Value>,
}
pub(super) struct FixtureServer {
    address: SocketAddr,
    state: Arc<Mutex<State>>,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<Result<()>>>,
}
impl FixtureServer {
    pub(super) fn start(commit: &str) -> Result<Self> {
        if commit.len() != 40 || !commit.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(invalid("invalid fixture commit"));
        }
        let listener = TcpListener::bind("127.0.0.1:0")?;
        listener.set_nonblocking(true)?;
        let address = listener.local_addr()?;
        let state = Arc::new(Mutex::new(State {
            repos: BTreeMap::new(),
            requests: Vec::new(),
        }));
        let stop = Arc::new(AtomicBool::new(false));
        let shared = Arc::clone(&state);
        let stopping = Arc::clone(&stop);
        let commit = commit.to_owned();
        let worker = std::thread::spawn(move || {
            while !stopping.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((mut stream, peer)) => {
                        if !peer.ip().is_loopback() {
                            return Err(invalid("nonloopback fixture peer"));
                        }
                        // WinSock accepts inherit the listener's nonblocking
                        // mode. Request I/O uses the bounded blocking deadlines.
                        stream.set_nonblocking(false)?;
                        stream.set_read_timeout(Some(Duration::from_secs(5)))?;
                        stream.set_write_timeout(Some(Duration::from_secs(5)))?;
                        serve(&mut stream, &shared, &commit)?;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(2));
                    }
                    Err(error) => return Err(error.into()),
                }
            }
            Ok(())
        });
        Ok(Self {
            address,
            state,
            stop,
            worker: Some(worker),
        })
    }
    pub(super) fn endpoint(&self) -> String {
        format!("http://{}/", self.address)
    }
    pub(super) fn replace(&self, repos: Repositories) -> Result<()> {
        if repos.len() > 32 || repos.values().any(|files| files.len() > 128) {
            return Err(invalid("fixture repository inventory exceeds bound"));
        }
        self.state
            .lock()
            .map_err(|_| invalid("fixture state poisoned"))?
            .repos = repos;
        Ok(())
    }
    pub(super) fn request_count(&self) -> Result<usize> {
        Ok(self
            .state
            .lock()
            .map_err(|_| invalid("fixture state poisoned"))?
            .requests
            .len())
    }
    pub(super) fn finish(mut self) -> Result<Vec<Value>> {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            worker
                .join()
                .map_err(|_| invalid("fixture server panicked"))??;
        }
        let requests = self
            .state
            .lock()
            .map_err(|_| invalid("fixture state poisoned"))?
            .requests
            .clone();
        Ok(requests)
    }
}
impl Drop for FixtureServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _joined = worker.join();
        }
    }
}
fn response(target: &str, repos: &Repositories, commit: &str) -> Result<(u16, String)> {
    let Some(api) = target
        .split('?')
        .next()
        .and_then(|path| path.strip_prefix("/api/models/"))
    else {
        return Ok((404, String::new()));
    };
    let (repo, kind) = if let Some(repo) = api.strip_suffix("/refs") {
        (repo, "refs")
    } else if let Some((repo, _revision)) = api.rsplit_once("/tree/") {
        (repo, "tree")
    } else {
        return Ok((404, String::new()));
    };
    let Some(files) = repos.get(repo) else {
        return Ok((404, String::new()));
    };
    if kind == "refs" {
        return Ok((
            200,
            format!(r#"{{"branches":[{{"name":"main","targetCommit":"{commit}"}}]}}"#),
        ));
    }
    let files = files
        .iter()
        .enumerate()
        .map(|(index, path)| {
            Ok(format!(
                r#"{{"type":"file","path":{},"size":1,"oid":"{:040x}"}}"#,
                serde_json::to_string(path)?,
                index + 1
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((200, format!("[{}]", files.join(","))))
}
fn request(stream: &mut TcpStream) -> Result<(String, String)> {
    let mut bytes = Vec::new();
    let mut buffer = [0_u8; 1024];
    while !bytes.windows(4).any(|w| w == b"\r\n\r\n") {
        let count = stream.read(&mut buffer)?;
        if count == 0 {
            return Err(invalid("truncated fixture HTTP request"));
        }
        bytes.extend_from_slice(&buffer[..count]);
        if bytes.len() > 32 * 1024 {
            return Err(invalid("fixture HTTP header exceeds32KiB"));
        }
    }
    let header = std::str::from_utf8(&bytes)?;
    let mut fields = header
        .lines()
        .next()
        .ok_or_else(|| invalid("missing HTTP request line"))?
        .split_whitespace();
    let method = fields
        .next()
        .ok_or_else(|| invalid("missing HTTP method"))?;
    let target = fields
        .next()
        .ok_or_else(|| invalid("missing HTTP target"))?;
    if fields.next() != Some("HTTP/1.1") || fields.next().is_some() || target.len() > 8192 {
        return Err(invalid("invalid fixture HTTP request line"));
    }
    Ok((method.to_owned(), target.to_owned()))
}
fn serve(stream: &mut TcpStream, state: &Mutex<State>, commit: &str) -> Result<()> {
    let (method, target) = request(stream)?;
    let (status, body) = {
        let mut state = state
            .lock()
            .map_err(|_| invalid("fixture state poisoned"))?;
        if state.requests.len() >= 1024 {
            return Err(invalid("fixture request count exceeds1024"));
        }
        let response = if method == "GET" {
            response(&target, &state.repos, commit)?
        } else {
            (404, String::new())
        };
        state
            .requests
            .push(json!({"Method":method,"Target":target,"Status":response.0}));
        drop(state);
        response
    };
    if body.len() > 128 * 1024 {
        return Err(invalid("fixture response exceeds128KiB"));
    }
    let phrase = if status == 200 { "OK" } else { "Not Found" };
    write!(
        stream,
        "HTTP/1.1 {status} {phrase}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body.as_bytes())?;
    stream.flush()?;
    Ok(())
}

#[cfg(test)]
mod focused_guards {
    use super::*;
    #[test]
    fn delayed_http_header_does_not_terminate_fixture() -> Result<()> {
        let server = FixtureServer::start("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")?;
        server.replace(BTreeMap::from([(
            "probe/repo".into(),
            vec!["input.gguf".into()],
        )]))?;
        let mut stream = TcpStream::connect(server.address)?;
        stream.set_read_timeout(Some(Duration::from_secs(2)))?;
        // A client may connect before it has sent its request. The fixture
        // must wait within its read deadline and keep serving later clients.
        std::thread::sleep(Duration::from_millis(100));
        stream.write_all(b"GET /api/models/probe/repo/refs HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")?;
        let mut response = String::new();
        stream.take(8192).read_to_string(&mut response)?;
        assert!(response.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(response.contains("targetCommit"));
        let requests = server.finish()?;
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0]["Status"], 200);
        Ok(())
    }
    #[test]
    fn bounded_loopback_transport_serves_custom_fixture() -> Result<()> {
        let server = FixtureServer::start("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")?;
        server.replace(BTreeMap::from([(
            "probe/repo".into(),
            vec!["input.gguf".into()],
        )]))?;
        let mut stream = TcpStream::connect(server.address)?;
        stream.set_read_timeout(Some(Duration::from_secs(2)))?;
        stream.write_all(b"GET /api/models/probe/repo/tree/main?recursive=true HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")?;
        let mut response = String::new();
        stream.take(8192).read_to_string(&mut response)?;
        assert!(response.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(response.contains("input.gguf"));
        let requests = server.finish()?;
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0]["Status"], 200);
        Ok(())
    }
    #[test]
    fn fixture_oid_order_and_missing_repo_are_explicit() -> Result<()> {
        let repos = BTreeMap::from([("test/repo".into(), vec!["b.gguf".into(), "a.gguf".into()])]);
        let (status, body) = response(
            "/api/models/test/repo/tree/main?recursive=true",
            &repos,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )?;
        assert_eq!(status, 200);
        let files: Value = serde_json::from_str(&body)?;
        assert_eq!(files[0]["path"], "b.gguf");
        assert_eq!(files[1]["oid"], "0000000000000000000000000000000000000002");
        assert_eq!(
            response(
                "/api/models/missing/repo/refs",
                &repos,
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            )?
            .0,
            404
        );
        Ok(())
    }
}
