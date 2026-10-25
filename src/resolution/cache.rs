//! Cache ownership and bounded HTTP metadata transport.

use super::{File, Model, Options, Plan, select, split_repo_tag};
use crate::{Result, invalid};
use serde_json::Value;
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

const MAX_HTTP: u64 = 64 * 1024 * 1024;
const MAX_HEADER: usize = 65536;
const TIMEOUT: Duration = Duration::from_secs(30);
static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

/// Explicit cache/endpoint owner; does not mutate process environment.
#[derive(Clone, Debug)]
pub struct Resolver {
    cache: PathBuf,
    endpoint: String,
}
fn valid_hash(value: &str, lengths: &[usize]) -> bool {
    lengths.contains(&value.len()) && value.bytes().all(|b| b.is_ascii_hexdigit())
}
fn valid_repo(value: &str) -> bool {
    let mut special = true;
    let mut slash = 0;
    if value.is_empty() || value.len() > 256 {
        return false;
    }
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || byte == b'_' {
            special = false;
        } else if matches!(byte, b'/' | b'.' | b'-') {
            if special {
                return false;
            }
            slash += usize::from(byte == b'/');
            special = true;
        } else {
            return false;
        }
    }
    !special && slash == 1
}
fn subpath(value: &str) -> Option<PathBuf> {
    let mut result = PathBuf::new();
    for component in Path::new(value).components() {
        match component {
            Component::Normal(part) => result.push(part),
            Component::CurDir => (),
            Component::ParentDir => {
                if !result.pop() {
                    return None;
                }
            }
            Component::Prefix(_) | Component::RootDir => return None,
        }
    }
    Some(result)
}
fn path_text(path: &Path) -> Result<String> {
    path.to_str()
        .map(str::to_owned)
        .ok_or_else(|| invalid("cache path is not UTF-8"))
}
fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension(format!(
        "tmp-{}-{}",
        std::process::id(),
        NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
    ));
    fs::write(&tmp, bytes)?;
    let outcome = fs::rename(&tmp, path);
    if outcome.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    Ok(outcome?)
}
fn decode_response(bytes: &[u8], max_body: usize) -> Result<(u16, Vec<u8>)> {
    let end = bytes
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or_else(|| invalid("HTTP response lacks header terminator"))?;
    if end > MAX_HEADER {
        return Err(invalid("HTTP response headers exceed bound"));
    }
    let header = std::str::from_utf8(&bytes[..end])?;
    let mut lines = header.split("\r\n");
    let mut status_line = lines
        .next()
        .ok_or_else(|| invalid("HTTP status missing"))?
        .split_whitespace();
    let version = status_line.next().unwrap_or("");
    let code = status_line.next().unwrap_or("");
    if !matches!(version, "HTTP/1.0" | "HTTP/1.1")
        || code.len() != 3
        || !code.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(invalid("invalid HTTP status line"));
    }
    let status = code.parse()?;
    let mut content_length = None;
    let mut chunked = false;
    for line in lines {
        let (key, value) = line
            .split_once(':')
            .filter(|(key, value)| valid_header(key, value))
            .ok_or_else(|| invalid("invalid HTTP header"))?;
        let value = value.trim();
        if key.eq_ignore_ascii_case("content-length") {
            if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
                return Err(invalid("invalid HTTP Content-Length"));
            }
            let length = value.parse::<usize>()?;
            if length > max_body || content_length.replace(length).is_some() {
                return Err(invalid("HTTP Content-Length exceeds bound or is repeated"));
            }
        } else if key.eq_ignore_ascii_case("transfer-encoding") {
            if chunked || !value.eq_ignore_ascii_case("chunked") {
                return Err(invalid("unsupported or repeated HTTP Transfer-Encoding"));
            }
            chunked = true;
        }
    }
    if chunked && content_length.is_some() {
        return Err(invalid("ambiguous HTTP response framing"));
    }
    let body = &bytes[end + 4..];
    // Content-Length framing and decoded-body bounds adapt Camelid's
    // src/fabric/http.rs parse_response/dechunk at 1829a7e73775eb829efe8fabf46a6680897a975b.
    // Copyright (c) 2026 Tim Toole, MIT; see the root NOTICE.
    if !chunked {
        if body.len() > max_body || content_length.is_some_and(|length| body.len() != length) {
            return Err(invalid(
                "HTTP body exceeds bound or differs from Content-Length",
            ));
        }
        return Ok((status, body.to_vec()));
    }
    let mut result = Vec::new();
    let mut at = 0;
    loop {
        let end = body[at..]
            .windows(2)
            .position(|w| w == b"\r\n")
            .ok_or_else(|| invalid("HTTP chunk header truncated"))?
            + at;
        let size = std::str::from_utf8(&body[at..end])?
            .split(';')
            .next()
            .unwrap_or("");
        if size.is_empty() || !size.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(invalid("invalid HTTP chunk size"));
        }
        let length = usize::from_str_radix(size, 16)?;
        at = end + 2;
        if length == 0 {
            validate_trailers(&body[at..])?;
            break;
        }
        if length > max_body.saturating_sub(result.len()) {
            return Err(invalid("decoded HTTP body exceeds bound"));
        }
        let end = at
            .checked_add(length)
            .ok_or_else(|| invalid("HTTP chunk overflow"))?;
        let after = end
            .checked_add(2)
            .ok_or_else(|| invalid("HTTP chunk overflow"))?;
        if body.get(end..after) != Some(b"\r\n") {
            return Err(invalid("HTTP chunk truncated"));
        }
        result.extend_from_slice(&body[at..end]);
        at = after;
    }
    Ok((status, result))
}
fn valid_header(key: &str, value: &str) -> bool {
    !key.is_empty()
        && key
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&byte))
        && value
            .bytes()
            .all(|byte| byte == b'\t' || (byte >= b' ' && byte != 127))
}
fn validate_trailers(bytes: &[u8]) -> Result<()> {
    if bytes.len() > MAX_HEADER {
        return Err(invalid("HTTP trailers exceed bound"));
    }
    let trailers = std::str::from_utf8(bytes)?;
    let mut lines = trailers.split("\r\n");
    loop {
        let line = lines
            .next()
            .ok_or_else(|| invalid("HTTP terminal chunk lacks trailer terminator"))?;
        if line.is_empty() {
            return if lines.next() == Some("") && lines.next().is_none() {
                Ok(())
            } else {
                Err(invalid(
                    "HTTP trailers truncated or followed by unexpected bytes",
                ))
            };
        }
        let (key, value) = line
            .split_once(':')
            .ok_or_else(|| invalid("invalid HTTP trailer"))?;
        if !valid_header(key, value)
            || key.eq_ignore_ascii_case("content-length")
            || key.eq_ignore_ascii_case("transfer-encoding")
        {
            return Err(invalid("invalid HTTP trailer"));
        }
    }
}
fn valid_token(token: &str) -> bool {
    (37..=256).contains(&token.len())
        && token.starts_with("hf_")
        && token[3..].bytes().all(|b| b.is_ascii_alphanumeric())
}
fn http_get(url: &str, token: &str, depth: usize) -> Result<Vec<u8>> {
    if depth > 5 || url.len() > 16384 || url.contains(['\r', '\n', '\0']) {
        return Err(invalid("invalid HTTP URL or redirect depth"));
    }
    // HTTPS delegates TLS/certificate handling to the operating system curl client.
    if url.starts_with("https://") {
        let mut command =
            std::process::Command::new(if cfg!(windows) { "curl.exe" } else { "curl" });
        command.args([
            "--silent",
            "--show-error",
            "--fail",
            "--location",
            "--max-redirs",
            "5",
            "--max-time",
            "30",
            "--max-filesize",
            "67108864",
            "--proto",
            "=https,http",
        ]);
        if valid_token(token) {
            command.args(["--header", &format!("Authorization: Bearer {token}")]);
        }
        let mut child = command
            .arg("--")
            .arg(url)
            .current_dir(std::env::temp_dir())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let mut bytes = Vec::new();
        let Some(stdout) = child.stdout.take() else {
            let _ = child.kill();
            let _ = child.wait();
            return Err(invalid("HTTPS response pipe missing"));
        };
        let read = stdout.take(MAX_HTTP + 1).read_to_end(&mut bytes);
        let exceeded = u64::try_from(bytes.len())? > MAX_HTTP;
        if read.is_err() || exceeded {
            let _ = child.kill();
        }
        let status = child.wait()?;
        read?;
        if !status.success() || exceeded {
            return Err(invalid("HTTPS request failed or exceeded body bound"));
        }
        return Ok(bytes);
    }
    let authority = url
        .strip_prefix("http://")
        .ok_or_else(|| invalid("URL requires HTTP or HTTPS"))?;
    let (authority, target) = authority
        .find('/')
        .map_or((authority, "/"), |at| (&authority[..at], &authority[at..]));
    if authority.is_empty() || authority.contains('@') {
        return Err(invalid("invalid HTTP authority"));
    }
    let address = if authority.starts_with('[') {
        if authority.contains("]:") {
            authority.to_owned()
        } else {
            format!("{authority}:80")
        }
    } else if authority.contains(':') {
        authority.to_owned()
    } else {
        format!("{authority}:80")
    };
    let mut stream = address
        .to_socket_addrs()?
        .find_map(|address| TcpStream::connect_timeout(&address, TIMEOUT).ok())
        .ok_or_else(|| invalid("HTTP connection failed"))?;
    stream.set_read_timeout(Some(TIMEOUT))?;
    stream.set_write_timeout(Some(TIMEOUT))?;
    let auth = if valid_token(token) {
        format!("Authorization: Bearer {token}\r\n")
    } else {
        String::new()
    };
    write!(
        stream,
        "GET {target} HTTP/1.1\r\nHost: {authority}\r\nAccept: application/json\r\nUser-Agent: bonsai-prism\r\nConnection: close\r\n{auth}\r\n"
    )?;
    let mut bytes = Vec::new();
    stream
        .take(MAX_HTTP + u64::try_from(MAX_HEADER)? + 1)
        .read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len())? > MAX_HTTP + u64::try_from(MAX_HEADER)? {
        return Err(invalid("HTTP response exceeds bound"));
    }
    let (status, body) = decode_response(&bytes, usize::try_from(MAX_HTTP)?)?;
    if status != 200 {
        return Err(invalid(format!("HTTP GET failed with status {status}")));
    }
    Ok(body)
}
impl Resolver {
    /// Create a resolver with an absolute cache root and model endpoint.
    ///
    /// # Errors
    /// Rejects relative cache roots and unsupported or malformed endpoint URLs.
    pub fn new(cache: PathBuf, mut endpoint: String) -> Result<Self> {
        if !cache.is_absolute()
            || !(endpoint.starts_with("http://") || endpoint.starts_with("https://"))
            || endpoint.contains(['\r', '\n', '\0'])
        {
            return Err(invalid(
                "resolver requires absolute cache and HTTP endpoint",
            ));
        }
        if !endpoint.ends_with('/') {
            endpoint.push('/');
        }
        Ok(Self { cache, endpoint })
    }
    /// Cache root supplied when constructing this owner.
    #[must_use]
    pub fn cache(&self) -> &Path {
        &self.cache
    }
    fn repo_root(&self, repo: &str) -> PathBuf {
        self.cache
            .join(format!("models--{}", repo.replace('/', "--")))
    }
    /// Read immutable live repository metadata and persist its selected branch ref.
    ///
    /// # Errors
    /// Reports transport, malformed JSON and filesystem failures.
    pub fn remote_files(&self, repo: &str, token: &str) -> Result<Vec<File>> {
        if !valid_repo(repo) {
            return Ok(Vec::new());
        }
        let refs: Value = serde_json::from_slice(&http_get(
            &format!("{}api/models/{repo}/refs", self.endpoint),
            token,
            0,
        )?)?;
        let branches = refs["branches"]
            .as_array()
            .ok_or_else(|| invalid("HF refs branches missing"))?;
        let mut selected = None;
        for branch in branches {
            let (Some(name), Some(commit)) =
                (branch["name"].as_str(), branch["targetCommit"].as_str())
            else {
                continue;
            };
            if subpath(name).is_none() || !valid_hash(commit, &[40]) {
                continue;
            }
            if name == "main" || selected.is_none() {
                selected = Some((name, commit));
            }
            if name == "main" {
                break;
            }
        }
        let Some((name, commit)) = selected else {
            return Ok(Vec::new());
        };
        let root = self.repo_root(repo);
        atomic_write(
            &root
                .join("refs")
                .join(subpath(name).ok_or_else(|| invalid("invalid HF branch"))?),
            commit.as_bytes(),
        )?;
        let tree: Value = serde_json::from_slice(&http_get(
            &format!(
                "{}api/models/{repo}/tree/{commit}?recursive=true",
                self.endpoint
            ),
            token,
            0,
        )?)?;
        let tree = tree
            .as_array()
            .ok_or_else(|| invalid("HF tree must be an array"))?;
        if tree.len() > 1_000_000 {
            return Err(invalid("HF tree exceeds item bound"));
        }
        let mut files = Vec::new();
        for item in tree {
            if item["type"] != "file" {
                continue;
            }
            let Some(path) = item["path"].as_str() else {
                continue;
            };
            let Some(relative) = subpath(path) else {
                continue;
            };
            let oid = if item["lfs"].is_object() {
                item["lfs"]["oid"].as_str().unwrap_or("")
            } else {
                item["oid"].as_str().unwrap_or("")
            };
            if !oid.is_empty() && !valid_hash(oid, &[40, 64]) {
                continue;
            }
            let final_path = root.join("snapshots").join(commit).join(relative);
            let local_path = if !oid.is_empty() && !final_path.exists() {
                root.join("blobs").join(oid)
            } else {
                final_path.clone()
            };
            files.push(File {
                path: path.into(),
                url: format!("{}{repo}/resolve/{commit}/{path}", self.endpoint),
                local_path,
                final_path,
                oid: oid.into(),
                repo: repo.into(),
            });
        }
        Ok(files)
    }
    /// Walk the snapshot identified by a cached main or fallback valid ref.
    ///
    /// # Errors
    /// Reports unreadable cache directories and files.
    pub fn cached_files(&self, repo: &str) -> Result<Vec<File>> {
        if !valid_repo(repo) {
            return Ok(Vec::new());
        }
        let root = self.repo_root(repo);
        let refs = root.join("refs");
        if !refs.is_dir() {
            return Ok(Vec::new());
        }
        let mut commit = String::new();
        for entry in fs::read_dir(refs)? {
            let entry = entry?;
            if !entry.file_type()?.is_file() || entry.metadata()?.len() > 128 {
                continue;
            }
            let bytes = fs::read_to_string(entry.path())?;
            let value = bytes.lines().next().unwrap_or("");
            if !valid_hash(value, &[40]) {
                continue;
            }
            if commit.is_empty() || entry.file_name() == "main" {
                value.clone_into(&mut commit);
            }
            if entry.file_name() == "main" {
                break;
            }
        }
        if commit.is_empty() {
            return Ok(Vec::new());
        }
        let snapshot = root.join("snapshots").join(commit);
        if !snapshot.is_dir() {
            return Ok(Vec::new());
        }
        let mut files = Vec::new();
        let mut directories = vec![snapshot.clone()];
        while let Some(directory) = directories.pop() {
            for entry in fs::read_dir(directory)? {
                let entry = entry?;
                let kind = entry.file_type()?;
                if kind.is_dir() {
                    directories.push(entry.path());
                } else if kind.is_file() || kind.is_symlink() {
                    let final_path = entry.path();
                    let path = path_text(final_path.strip_prefix(&snapshot)?)?.replace('\\', "/");
                    files.push(File {
                        path,
                        local_path: final_path.clone(),
                        final_path,
                        repo: repo.into(),
                        ..File::default()
                    });
                }
                if files.len() + directories.len() > 1_000_000 {
                    return Err(invalid("HF cache exceeds item bound"));
                }
            }
        }
        Ok(files)
    }
    /// Resolve online metadata, falling back to locally cached descriptors.
    ///
    /// # Errors
    /// Rejects invalid repository/tag syntax or selection patterns.
    pub fn plan(&self, model: &Model, options: &Options) -> Result<Plan> {
        let (repo, _) = split_repo_tag(&model.hf_repo)?;
        let mut files = if options.offline {
            Vec::new()
        } else {
            self.remote_files(repo, &options.token).unwrap_or_default()
        };
        if files.is_empty() {
            files = self.cached_files(repo).unwrap_or_default();
        }
        select(&files, model, options)
    }
    /// Download one missing blob, then expose its immutable snapshot path.
    ///
    /// # Errors
    /// Reports download, path ownership and filesystem errors.
    pub fn finalize(&self, file: &File, offline: bool, token: &str) -> Result<String> {
        if file.path.is_empty() {
            return Ok(String::new());
        }
        if !file.local_path.starts_with(&self.cache) || !file.final_path.starts_with(&self.cache) {
            return Err(invalid("HF file lies outside owned cache"));
        }
        if !offline && !file.local_path.exists() && !file.url.is_empty() {
            atomic_write(&file.local_path, &http_get(&file.url, token, 0)?)?;
        }
        if file.local_path != file.final_path
            && !file.final_path.exists()
            && file.local_path.exists()
        {
            if let Some(parent) = file.final_path.parent() {
                fs::create_dir_all(parent)?;
            }
            if fs::hard_link(&file.local_path, &file.final_path).is_err() {
                fs::copy(&file.local_path, &file.final_path)?;
            }
        }
        path_text(&file.final_path)
    }
    pub(super) fn plain_url(&self, model: &mut Model, offline: bool, token: &str) -> Result<()> {
        if model.url.is_empty() {
            return Ok(());
        }
        if model.path.is_empty() {
            let url = model.url.split(['#', '?']).next().unwrap_or("");
            let filename = url.rsplit('/').next().unwrap_or("");
            if filename.is_empty() || subpath(filename).is_none() {
                return Err(invalid("download URL filename missing"));
            }
            model.path = path_text(&self.cache.join(filename))?;
        }
        if !offline && !Path::new(&model.path).exists() {
            atomic_write(Path::new(&model.path), &http_get(&model.url, token, 0)?)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    fn with_loopback<T>(response: &[u8], request: impl FnOnce(&str) -> Result<T>) -> Result<T> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let response = response.to_vec();
        let server = std::thread::spawn(move || -> std::io::Result<()> {
            let (mut stream, _) = listener.accept()?;
            stream.set_read_timeout(Some(Duration::from_secs(5)))?;
            let mut request = Vec::new();
            let mut buffer = [0_u8; 512];
            while !request.windows(4).any(|bytes| bytes == b"\r\n\r\n") {
                let count = stream.read(&mut buffer)?;
                if count == 0 || request.len() > MAX_HEADER {
                    return Err(std::io::Error::other("invalid test request"));
                }
                request.extend_from_slice(&buffer[..count]);
            }
            stream.write_all(&response)
        });
        let body = request(&format!("http://{address}/model.gguf"));
        server
            .join()
            .map_err(|_| invalid("loopback response worker failed"))??;
        body
    }
    fn loopback_get(response: &[u8]) -> Result<Vec<u8>> {
        with_loopback(response, |url| http_get(url, "", 0))
    }
    #[test]
    fn hostile_repository_paths_and_hashes_are_rejected() {
        assert!(valid_repo("owner/model-v1.0"));
        for repo in [
            "../model",
            "owner//model",
            "owner/model..gguf",
            "/owner/model",
            "owner/model-",
        ] {
            assert!(!valid_repo(repo));
        }
        assert!(subpath("a/../b").is_some());
        for path in ["../escape", "a/../../escape", "/absolute"] {
            assert!(subpath(path).is_none());
        }
        assert!(!valid_hash(&"x".repeat(40), &[40]));
    }
    #[test]
    fn chunked_binary_response_is_decoded_without_text_conversion() -> Result<()> {
        let (_, bytes) = decode_response(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\na\0b\r\n2;ext=x\r\ncd\r\n0\r\n\r\n", 5)?;
        assert_eq!(bytes, b"a\0bcd");
        assert!(
            decode_response(
                b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n9\r\nshort\r\n",
                64
            )
            .is_err()
        );
        Ok(())
    }
    #[test]
    fn live_http_preserves_binary_content_and_complete_chunk_trailers() -> Result<()> {
        assert_eq!(
            loopback_get(b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\na\0b")?,
            b"a\0b"
        );
        assert_eq!(
            loopback_get(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3;ext=x\r\na\0b\r\n0\r\nDigest: opaque\r\n\r\n")?,
            b"a\0b"
        );
        assert_eq!(
            loopback_get(b"HTTP/1.0 200 OK\r\n\r\nclose-delimited")?,
            b"close-delimited"
        );
        Ok(())
    }
    #[test]
    fn live_http_rejects_truncated_oversized_or_ambiguous_framing() {
        for response in [
            &b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\nabc"[..],
            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nabc",
            b"HTTP/1.1 200 OK\r\nContent-Length: 67108865\r\n\r\n",
            b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\nContent-Length: 2\r\n\r\na",
            b"HTTP/1.1 200 OK\r\nContent-Length: invalid\r\n\r\na",
            b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n",
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: gzip\r\n\r\na",
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n1\r\na\r\n0\r\n",
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n1\r\naXX0\r\n\r\n",
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n0\r\nDigest: incomplete",
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n0\r\nContent-Length: 0\r\n\r\n",
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n+1\r\na\r\n0\r\n\r\n",
            b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n",
            b"NOT-HTTP 200 OK\r\nContent-Length: 0\r\n\r\n",
        ] {
            assert!(loopback_get(response).is_err(), "accepted {response:?}");
        }
    }
    #[test]
    fn decoded_body_limit_applies_to_every_framing_mode() {
        for response in [
            &b"HTTP/1.1 200 OK\r\n\r\nabc"[..],
            b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nabc",
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n2\r\nab\r\n1\r\nc\r\n0\r\n\r\n",
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nffffffffffffffff\r\n",
        ] {
            assert!(decode_response(response, 2).is_err());
        }
    }
    #[test]
    fn rejected_transfer_never_publishes_a_cached_model() -> Result<()> {
        let cache = std::env::temp_dir().join(format!(
            "prism-http-{}-{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        assert!(!cache.exists());
        with_loopback(
            b"HTTP/1.1 200 OK\r\nContent-Length: 8\r\n\r\npartial",
            |url| {
                let resolver = Resolver::new(cache.clone(), "http://127.0.0.1/".into())?;
                let file = File {
                    path: "model.gguf".into(),
                    url: url.into(),
                    local_path: cache.join("blobs").join("model"),
                    final_path: cache.join("snapshots").join("model.gguf"),
                    ..File::default()
                };
                assert!(resolver.finalize(&file, false, "").is_err());
                assert!(!file.local_path.exists());
                assert!(!file.final_path.exists());
                Ok(())
            },
        )?;
        assert!(!cache.exists());
        Ok(())
    }
}
