//! Clean-room independent safe Rust model resolution, HTTP/1.1, and cache reference provider.
//!
//! Independently implements bounded HTTP/1.1 transport over `std::net::TcpStream`
//! (`GET`, `HEAD`, `Range`/`206`, redirects, `If-None-Match`/`ETag`/`304`,
//! `Content-Disposition`, `Content-Range`, `Content-Length`, chunked transfer),
//! atomic Hugging Face / URL cache writes with SHA-256 verification, Hugging Face
//! repository listing and quantization/shard/sidecar planning, and model CLI
//! handler assembly. Never imports `bonsai_prism_runtime`, calls any native bridge,
//! or reads expected-answer test fixtures.

use std::fs::{self, OpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use super::{Result, invalid};

/// Supported upstream assertion source identifier for `tests/test-model-resolution.cpp`.
pub(super) const ID: &str = "tests/test-model-resolution.cpp";
const SOURCE_SHA: &str = "d596a6befca98035acfab4e71e72f50a698a11062fcac138735b3fe1aa9af593";

const EXAMPLE_COUNT: u32 = 27;
const DEFAULT_QUANT: &str = "Q4_K_M";
const MAX_OPERATIONS: usize = 128;
const MAX_REPO_BYTES: usize = 256;
const MAX_PATH_BYTES: usize = 4096;
const MAX_URL_BYTES: usize = 4096;
const MAX_TOKEN_BYTES: usize = 4096;
const MAX_HEADERS: usize = 32;
const MAX_HEADER_FIELD_BYTES: usize = 4096;
const MAX_ARGV_COUNT: usize = 64;
const MAX_ARGV_ITEM_BYTES: usize = 16_384;
const MAX_ARGV_TOTAL_BYTES: usize = 65_536;
const MAX_HTTP_HEADER_BYTES: usize = 64 * 1024;
const MAX_HTTP_LINE_BYTES: usize = 8192;
const MAX_HTTP_BODY_BYTES: usize = 16 * 1024 * 1024;
const MAX_REDIRECTS: usize = 5;
const DEFAULT_TIMEOUT_SECS: u64 = 15;
const MAX_TIMEOUT_SECS: u64 = 60;

const SPEC_TYPE_DRAFT: u32 = 0;
const SPEC_TYPE_EAGLE3: u32 = 2;
const SPEC_TYPE_MTP: u32 = 3;
const SPEC_TYPE_DFLASH: u32 = 4;
const SPEC_TYPE_DSPARK: u32 = 5;

/// Resolved repository or cached file metadata matching `hf_cache::hf_file`.
#[derive(Clone, Default)]
pub(super) struct ResolvedFile {
    pub(super) path: String,
    pub(super) url: String,
    pub(super) local_path: String,
    pub(super) final_path: String,
    pub(super) oid: String,
    pub(super) repo: String,
}

/// Complete Hugging Face resolution plan for a primary model and sidecars.
#[derive(Clone, Default)]
pub(super) struct Plan {
    pub(super) primary: ResolvedFile,
    pub(super) model_files: Vec<ResolvedFile>,
    pub(super) mmproj: ResolvedFile,
    pub(super) mtp: ResolvedFile,
    pub(super) dflash: ResolvedFile,
    pub(super) eagle3: ResolvedFile,
    pub(super) dspark: ResolvedFile,
    pub(super) preset: ResolvedFile,
}

/// Model selector fields parsed from CLI arguments or API calls.
#[derive(Clone, Default)]
pub(super) struct ModelSpec {
    pub(super) path: String,
    pub(super) url: String,
    pub(super) hf_repo: String,
    pub(super) hf_file: String,
}

/// Sidecar discovery toggles matching `common_download_opts`.
#[expect(
    clippy::struct_excessive_bools,
    reason = "Mirrors the five independent upstream sidecar resolution flags."
)]
#[derive(Clone, Copy, Default)]
pub(super) struct Options {
    pub(super) mmproj: bool,
    pub(super) mtp: bool,
    pub(super) dflash: bool,
    pub(super) eagle3: bool,
    pub(super) dspark: bool,
}

/// Subset of `common_params` observed by model-resolution tests.
#[derive(Clone, Default)]
pub(super) struct Params {
    pub(super) model: ModelSpec,
    pub(super) mmproj: ModelSpec,
    pub(super) draft: ModelSpec,
    pub(super) spec_types: Vec<u32>,
    pub(super) preset: String,
    pub(super) offline: bool,
    pub(super) no_mmproj: bool,
    pub(super) hf_token: String,
}

/// Planned model and draft-model resolution handler state.
#[derive(Clone, Default)]
pub(super) struct Handler {
    pub(super) plan: Plan,
    pub(super) draft_plan: Plan,
}

/// Independent clean-room Hugging Face / HTTP model resolver.
#[derive(Clone)]
pub(super) struct Resolver {
    cache_dir: PathBuf,
    endpoint: String,
    offline: bool,
    hf_token: Option<String>,
}

impl Resolver {
    /// Creates a new clean-room resolver bound to a cache directory and HTTP endpoint.
    pub(super) fn new(cache_dir: PathBuf, endpoint: String) -> Result<Self> {
        Self::with_options(cache_dir, endpoint, false, None)
    }

    /// Creates a resolver with explicit offline and bearer-token settings.
    pub(super) fn with_options(
        cache_dir: PathBuf,
        endpoint: String,
        offline: bool,
        hf_token: Option<String>,
    ) -> Result<Self> {
        let cache_str = cache_dir
            .to_str()
            .ok_or_else(|| invalid("cache directory must be valid UTF-8"))?;
        if !cache_dir.is_absolute() || cache_str.contains('%') || cache_str.contains('\0') {
            return Err(invalid("cache directory must be an expanded absolute path"));
        }
        if !cache_dir.is_dir() {
            return Err(invalid("cache directory must exist"));
        }
        if endpoint.len() > MAX_URL_BYTES || endpoint.contains('\0') {
            return Err(invalid("resolver endpoint exceeds bound or contains NUL"));
        }
        let _ = parse_http_url(&endpoint)?;
        if let Some(token) = hf_token.as_deref()
            && (token.len() > MAX_TOKEN_BYTES || token.contains(['\0', '\r', '\n']))
        {
            return Err(invalid("invalid HF token"));
        }
        Ok(Self {
            cache_dir,
            endpoint,
            offline,
            hf_token,
        })
    }

    /// Executes a `model_resolution.hpp` JSON wire operation.
    pub(super) fn invoke(&self, operation: &Value) -> Result<Value> {
        let op = text(&operation["Op"], 64)?;
        match op {
            "resolution-pause-log" => {
                exact_fields(operation, &["Op"])?;
                Ok(Value::Null)
            }
            "resolution-plan" => {
                exact_fields(operation, &["Op", "HfRepo", "HfFile", "Options"])?;
                let hf_repo = text(&operation["HfRepo"], MAX_REPO_BYTES)?.to_owned();
                let hf_file = text(&operation["HfFile"], MAX_PATH_BYTES)?.to_owned();
                let options = parse_plan_options(&operation["Options"])?;
                let model = ModelSpec {
                    hf_repo,
                    hf_file,
                    ..ModelSpec::default()
                };
                Ok(plan_to_json(&self.plan(&model, options)?))
            }
            "resolution-handler" => {
                exact_fields(
                    operation,
                    &["Op", "Arguments", "Example", "OfflineAfterInit", "Apply"],
                )?;
                let example = parse_example(&operation["Example"])?;
                let offline_after_init = flag(&operation["OfflineAfterInit"], "OfflineAfterInit")?;
                let apply = flag(&operation["Apply"], "Apply")?;
                let arguments = parse_argv(&operation["Arguments"])?;
                execute_handler(self, &arguments, example, offline_after_init, apply)
            }
            _ => Err(invalid("unknown resolution operation")),
        }
    }

    /// Resolves a Hugging Face repository listing and constructs a `Plan`.
    pub(super) fn plan(&self, model: &ModelSpec, opts: Options) -> Result<Plan> {
        if model.hf_repo.is_empty() {
            return Ok(Plan::default());
        }
        let (repo_id, user_tag) = parse_hf_repo(&model.hf_repo)?;
        let (commit, entries) = self.fetch_hf_tree(&repo_id)?;
        let mut plan = Plan::default();

        if model.hf_file.is_empty()
            && let Some(preset_entry) = entries.iter().find(|entry| entry.path == "preset.ini")
        {
            plan.preset = self.build_resolved_file(&repo_id, &commit, preset_entry)?;
        }

        if !model.hf_file.is_empty() {
            if let Some(matched) = entries.iter().find(|entry| entry.path == model.hf_file) {
                self.assign_primary(&repo_id, &commit, &entries, matched, &mut plan)?;
            }
            return Ok(plan);
        }

        let mut main_entries = Vec::new();
        let mut mmproj_entries = Vec::new();
        let mut mtp_entries = Vec::new();
        let mut dflash_entries = Vec::new();
        let mut eagle3_entries = Vec::new();
        let mut dspark_entries = Vec::new();

        for entry in &entries {
            match classify_repo_file(&entry.path) {
                RepoFileKind::MainGguf => main_entries.push(entry),
                RepoFileKind::Mmproj => mmproj_entries.push(entry),
                RepoFileKind::Mtp => mtp_entries.push(entry),
                RepoFileKind::Dflash => dflash_entries.push(entry),
                RepoFileKind::Eagle3 => eagle3_entries.push(entry),
                RepoFileKind::Dspark => dspark_entries.push(entry),
                RepoFileKind::Other => {}
            }
        }

        let selected_main = user_tag.as_deref().map_or_else(
            || {
                main_entries
                    .iter()
                    .copied()
                    .find(|entry| matches_quant(&entry.path, DEFAULT_QUANT))
                    .or_else(|| main_entries.first().copied())
            },
            |tag| {
                main_entries
                    .iter()
                    .copied()
                    .find(|entry| matches_quant(&entry.path, tag))
            },
        );

        if let Some(primary_entry) = selected_main {
            self.assign_primary(&repo_id, &commit, &entries, primary_entry, &mut plan)?;
        }

        let anchor_tag = user_tag.or_else(|| {
            if plan.primary.path.is_empty() {
                None
            } else {
                extract_quant_tag(&plan.primary.path)
            }
        });

        self.assign_sidecars(
            &repo_id,
            &commit,
            opts,
            anchor_tag.as_deref(),
            &SidecarCandidates {
                mmproj: &mmproj_entries,
                mtp: &mtp_entries,
                dflash: &dflash_entries,
                eagle3: &eagle3_entries,
                dspark: &dspark_entries,
            },
            &mut plan,
        )?;

        Ok(plan)
    }

    fn assign_primary(
        &self,
        repo_id: &str,
        commit: &str,
        entries: &[TreeEntry],
        matched: &TreeEntry,
        plan: &mut Plan,
    ) -> Result<()> {
        let group = expand_shards(entries, matched);
        let files = group
            .iter()
            .map(|entry| self.build_resolved_file(repo_id, commit, entry))
            .collect::<Result<Vec<_>>>()?;
        if let Some(first) = files.first() {
            plan.primary = first.clone();
        }
        plan.model_files = files;
        Ok(())
    }

    fn assign_sidecars(
        &self,
        repo_id: &str,
        commit: &str,
        opts: Options,
        anchor_tag: Option<&str>,
        candidates: &SidecarCandidates<'_>,
        plan: &mut Plan,
    ) -> Result<()> {
        if opts.mmproj
            && let Some(entry) = select_mmproj(candidates.mmproj, anchor_tag)
        {
            plan.mmproj = self.build_resolved_file(repo_id, commit, entry)?;
        }
        if opts.mtp
            && let Some(entry) = select_draft_sidecar(candidates.mtp, anchor_tag)
        {
            plan.mtp = self.build_resolved_file(repo_id, commit, entry)?;
        }
        if opts.dflash
            && let Some(entry) = select_draft_sidecar(candidates.dflash, anchor_tag)
        {
            plan.dflash = self.build_resolved_file(repo_id, commit, entry)?;
        }
        if opts.eagle3
            && let Some(entry) = select_draft_sidecar(candidates.eagle3, anchor_tag)
        {
            plan.eagle3 = self.build_resolved_file(repo_id, commit, entry)?;
        }
        if opts.dspark
            && let Some(entry) = select_draft_sidecar(candidates.dspark, anchor_tag)
        {
            plan.dspark = self.build_resolved_file(repo_id, commit, entry)?;
        }
        Ok(())
    }

    fn fetch_hf_tree(&self, repo_id: &str) -> Result<(String, Vec<TreeEntry>)> {
        let repo_slug = format!("models--{}", repo_id.replace('/', "--"));
        let ref_path = self.cache_dir.join(&repo_slug).join("refs").join("main");
        if self.offline {
            let commit = fs::read_to_string(&ref_path)
                .map_err(|_| invalid(format!("offline HF ref missing in cache for {repo_id}")))?;
            let commit = commit.trim().to_owned();
            validate_commit(&commit)?;
            let snapshot_dir = self
                .cache_dir
                .join(&repo_slug)
                .join("snapshots")
                .join(&commit);
            let mut entries = Vec::new();
            collect_cached_tree(&snapshot_dir, &snapshot_dir, &mut entries)?;
            entries.sort_by(|a, b| a.path.cmp(&b.path));
            return Ok((commit, entries));
        }

        let base = self.endpoint.trim_end_matches('/');
        let mut auth_headers = Vec::new();
        let auth_header_value;
        if let Some(token) = self.hf_token.as_deref().filter(|t| !t.is_empty()) {
            auth_header_value = format!("Bearer {token}");
            auth_headers.push(HeaderEntry {
                name: "Authorization",
                value: &auth_header_value,
            });
        }

        let refs_url = format!("{base}/api/models/{repo_id}/refs");
        let refs_resp = perform_http_request(
            HttpMethod::Get,
            &refs_url,
            &auth_headers,
            Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            MAX_HTTP_BODY_BYTES,
        )?;
        if refs_resp.status_code != 200 {
            return Err(invalid(format!(
                "HF refs request failed with status {}",
                refs_resp.status_code
            )));
        }
        let refs_json: Value = serde_json::from_slice(&refs_resp.body)?;
        let commit = extract_main_commit(&refs_json)?;
        validate_commit(&commit)?;
        write_atomic(&ref_path, commit.as_bytes())?;

        let tree_url = format!("{base}/api/models/{repo_id}/tree/{commit}?recursive=true");
        let tree_resp = perform_http_request(
            HttpMethod::Get,
            &tree_url,
            &auth_headers,
            Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            MAX_HTTP_BODY_BYTES,
        )?;
        if tree_resp.status_code != 200 {
            return Err(invalid(format!(
                "HF tree request failed with status {}",
                tree_resp.status_code
            )));
        }
        let tree_json: Value = serde_json::from_slice(&tree_resp.body)?;
        let items = tree_json
            .as_array()
            .ok_or_else(|| invalid("HF tree response must be a JSON array"))?;
        let mut entries = Vec::with_capacity(items.len());
        for item in items {
            if item["type"].as_str() != Some("file") {
                continue;
            }
            let path = item["path"]
                .as_str()
                .ok_or_else(|| invalid("HF tree entry path must be a string"))?;
            validate_relative_repo_path(path)?;
            let oid = item["oid"].as_str().unwrap_or_default().to_owned();
            entries.push(TreeEntry {
                path: path.to_owned(),
                oid,
            });
        }
        Ok((commit, entries))
    }

    fn build_resolved_file(
        &self,
        repo_id: &str,
        commit: &str,
        entry: &TreeEntry,
    ) -> Result<ResolvedFile> {
        let base = self.endpoint.trim_end_matches('/');
        let local = self
            .cache_dir
            .join(format!("models--{}", repo_id.replace('/', "--")))
            .join("snapshots")
            .join(commit)
            .join(&entry.path);
        let local_str = local
            .to_str()
            .ok_or_else(|| invalid("resolved local cache path is not UTF-8"))?
            .to_owned();
        Ok(ResolvedFile {
            path: entry.path.clone(),
            url: format!("{base}/{repo_id}/resolve/{commit}/{}", entry.path),
            local_path: local_str.clone(),
            final_path: local_str,
            oid: entry.oid.clone(),
            repo: repo_id.to_owned(),
        })
    }

    fn download_plan_files(&self, plan: &Plan, files: &[&ResolvedFile]) -> Result<()> {
        for file in files {
            if file.final_path.is_empty() {
                continue;
            }
            self.download_single(file)?;
        }
        if !plan.primary.final_path.is_empty() && files.is_empty() {
            self.download_single(&plan.primary)?;
        }
        Ok(())
    }

    fn download_single(&self, file: &ResolvedFile) -> Result<()> {
        let dest = Path::new(&file.final_path);
        if dest.is_file() {
            return Ok(());
        }
        if self.offline {
            return Err(invalid(format!(
                "offline mode and resolved file is not cached: {}",
                file.final_path
            )));
        }
        let mut auth_headers = Vec::new();
        let auth_header_value;
        if let Some(token) = self.hf_token.as_deref().filter(|t| !t.is_empty()) {
            auth_header_value = format!("Bearer {token}");
            auth_headers.push(HeaderEntry {
                name: "Authorization",
                value: &auth_header_value,
            });
        }
        let response = perform_http_request(
            HttpMethod::Get,
            &file.url,
            &auth_headers,
            Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            MAX_HTTP_BODY_BYTES,
        )?;
        if response.status_code != 200 {
            return Err(invalid(format!(
                "HF file download failed with status {}",
                response.status_code
            )));
        }
        if file.oid.len() == 64 && file.oid.bytes().all(|b| b.is_ascii_hexdigit()) {
            let actual = sha256_hex(&response.body);
            if !actual.eq_ignore_ascii_case(&file.oid) {
                return Err(invalid("HF downloaded file SHA-256 mismatch"));
            }
        }
        write_atomic(dest, &response.body)?;
        Ok(())
    }
}

impl Params {
    /// Parses CLI argument vectors for model, mmproj, draft, preset, offline, and spec-type flags.
    pub(super) fn parse(arguments: &[String]) -> Result<Self> {
        if arguments.is_empty() {
            return Err(invalid("empty CLI argv"));
        }
        let mut params = Self::default();
        let mut iter = arguments.iter().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "-hf" | "--hf-repo" => {
                    let val = iter
                        .next()
                        .ok_or_else(|| invalid("missing value for -hf"))?;
                    params.model.hf_repo.clone_from(val);
                }
                "-hff" | "--hf-file" => {
                    let val = iter
                        .next()
                        .ok_or_else(|| invalid("missing value for -hff"))?;
                    params.model.hf_file.clone_from(val);
                }
                "-m" | "--model" => {
                    let val = iter.next().ok_or_else(|| invalid("missing value for -m"))?;
                    params.model.path.clone_from(val);
                }
                "-mu" | "--model-url" => {
                    let val = iter
                        .next()
                        .ok_or_else(|| invalid("missing value for -mu"))?;
                    params.model.url.clone_from(val);
                }
                "-hfd" | "--hf-repo-draft" => {
                    let val = iter
                        .next()
                        .ok_or_else(|| invalid("missing value for -hfd"))?;
                    params.draft.hf_repo.clone_from(val);
                }
                "-hffd" | "--hf-file-draft" => {
                    let val = iter
                        .next()
                        .ok_or_else(|| invalid("missing value for -hffd"))?;
                    params.draft.hf_file.clone_from(val);
                }
                "-md" | "--model-draft" => {
                    let val = iter
                        .next()
                        .ok_or_else(|| invalid("missing value for -md"))?;
                    params.draft.path.clone_from(val);
                }
                "--mmproj" => {
                    let val = iter
                        .next()
                        .ok_or_else(|| invalid("missing value for --mmproj"))?;
                    params.mmproj.path.clone_from(val);
                    params.no_mmproj = false;
                }
                "--mmproj-url" => {
                    let val = iter
                        .next()
                        .ok_or_else(|| invalid("missing value for --mmproj-url"))?;
                    params.mmproj.url.clone_from(val);
                }
                "--no-mmproj" => {
                    params.no_mmproj = true;
                    params.mmproj.path.clear();
                }
                "--spec-type" => {
                    let val = iter
                        .next()
                        .ok_or_else(|| invalid("missing value for --spec-type"))?;
                    params.spec_types = parse_spec_types(val)?;
                }
                "--offline" => {
                    params.offline = true;
                }
                "-hft" | "--hf-token" => {
                    let val = iter
                        .next()
                        .ok_or_else(|| invalid("missing value for --hf-token"))?;
                    params.hf_token.clone_from(val);
                }
                "--models-preset" => {
                    let val = iter
                        .next()
                        .ok_or_else(|| invalid("missing value for --models-preset"))?;
                    params.preset.clone_from(val);
                }
                _ => return Err(invalid(format!("unknown CLI flag: {arg}"))),
            }
        }
        Ok(params)
    }
}

impl Handler {
    /// Initializes HF primary and draft resolution plans from parsed CLI parameters.
    pub(super) fn initialize(resolver: &Resolver, params: &Params, example: u32) -> Result<Self> {
        if example >= EXAMPLE_COUNT {
            return Err(invalid("example index outside enum"));
        }
        let active_resolver = if params.offline || !params.hf_token.is_empty() {
            Resolver::with_options(
                resolver.cache_dir.clone(),
                resolver.endpoint.clone(),
                params.offline || resolver.offline,
                if params.hf_token.is_empty() {
                    resolver.hf_token.clone()
                } else {
                    Some(params.hf_token.clone())
                },
            )?
        } else {
            resolver.clone()
        };

        let plan = if !params.model.hf_repo.is_empty()
            && params.model.path.is_empty()
            && params.model.url.is_empty()
        {
            let same_repo_draft = params.draft.hf_repo.is_empty() && params.draft.path.is_empty();
            let opts = Options {
                mmproj: !params.no_mmproj
                    && params.mmproj.path.is_empty()
                    && params.mmproj.url.is_empty(),
                mtp: same_repo_draft && params.spec_types.contains(&SPEC_TYPE_MTP),
                dflash: same_repo_draft && params.spec_types.contains(&SPEC_TYPE_DFLASH),
                eagle3: same_repo_draft && params.spec_types.contains(&SPEC_TYPE_EAGLE3),
                dspark: same_repo_draft && params.spec_types.contains(&SPEC_TYPE_DSPARK),
            };
            active_resolver.plan(&params.model, opts)?
        } else {
            Plan::default()
        };

        let draft_plan = if params.draft.hf_repo.is_empty() {
            Plan::default()
        } else {
            let mut draft_model = params.draft.clone();
            if draft_model.hf_file.is_empty() && !draft_model.path.is_empty() {
                draft_model.hf_file = draft_model.path.clone();
            }
            let draft_opts = if !draft_model.hf_file.is_empty() || !params.spec_types.is_empty() {
                Options {
                    mmproj: false,
                    mtp: params.spec_types.contains(&SPEC_TYPE_MTP),
                    dflash: params.spec_types.contains(&SPEC_TYPE_DFLASH),
                    eagle3: params.spec_types.contains(&SPEC_TYPE_EAGLE3),
                    dspark: params.spec_types.contains(&SPEC_TYPE_DSPARK),
                }
            } else {
                Options {
                    mmproj: false,
                    mtp: true,
                    dflash: true,
                    eagle3: true,
                    dspark: true,
                }
            };
            active_resolver.plan(&draft_model, draft_opts)?
        };

        Ok(Self { plan, draft_plan })
    }

    /// Applies planned local/cached paths (and downloads if online) onto `params`.
    pub(super) fn apply(&self, resolver: &Resolver, params: &mut Params) -> Result<()> {
        let effective_resolver = Resolver::with_options(
            resolver.cache_dir.clone(),
            resolver.endpoint.clone(),
            params.offline || resolver.offline,
            if params.hf_token.is_empty() {
                resolver.hf_token.clone()
            } else {
                Some(params.hf_token.clone())
            },
        )?;

        if !self.plan.preset.final_path.is_empty()
            && params.model.hf_file.is_empty()
            && params.model.path.is_empty()
        {
            if !effective_resolver.offline {
                effective_resolver.download_single(&self.plan.preset)?;
            }
            params.preset.clone_from(&self.plan.preset.final_path);
            params.model.path.clear();
            params.model.hf_repo.clear();
            return Ok(());
        }

        if !self.plan.primary.final_path.is_empty() && params.model.path.is_empty() {
            if !effective_resolver.offline {
                let refs: Vec<&ResolvedFile> = self.plan.model_files.iter().collect();
                effective_resolver.download_plan_files(&self.plan, &refs)?;
            }
            params.model.path.clone_from(&self.plan.primary.final_path);
        }

        if !params.no_mmproj
            && params.mmproj.path.is_empty()
            && params.mmproj.url.is_empty()
            && !self.plan.mmproj.final_path.is_empty()
        {
            if !effective_resolver.offline {
                effective_resolver.download_single(&self.plan.mmproj)?;
            }
            params.mmproj.path.clone_from(&self.plan.mmproj.final_path);
        }

        if !params.draft.hf_repo.is_empty() {
            let selected = select_applied_draft(&self.draft_plan, params);
            if let Some(file) = selected {
                if !effective_resolver.offline {
                    effective_resolver.download_single(file)?;
                }
                params.draft.path.clone_from(&file.final_path);
            }
        } else if params.draft.path.is_empty() && !params.spec_types.is_empty() {
            for &spec_type in &params.spec_types {
                let candidate = match spec_type {
                    SPEC_TYPE_MTP => &self.plan.mtp,
                    SPEC_TYPE_DSPARK => &self.plan.dspark,
                    SPEC_TYPE_DFLASH => &self.plan.dflash,
                    SPEC_TYPE_EAGLE3 => &self.plan.eagle3,
                    _ => continue,
                };
                if !candidate.final_path.is_empty() {
                    if !effective_resolver.offline {
                        effective_resolver.download_single(candidate)?;
                    }
                    params.draft.path.clone_from(&candidate.final_path);
                    break;
                }
            }
        }

        Ok(())
    }
}

fn select_applied_draft<'a>(draft_plan: &'a Plan, params: &mut Params) -> Option<&'a ResolvedFile> {
    let explicit_file = !params.draft.hf_file.is_empty()
        || (!params.draft.path.is_empty() && params.draft.path == draft_plan.primary.path);
    if explicit_file && !draft_plan.primary.final_path.is_empty() {
        return Some(&draft_plan.primary);
    }
    if !params.spec_types.is_empty() {
        for &spec_type in &params.spec_types {
            let candidate = match spec_type {
                SPEC_TYPE_MTP => &draft_plan.mtp,
                SPEC_TYPE_DSPARK => &draft_plan.dspark,
                SPEC_TYPE_DFLASH => &draft_plan.dflash,
                SPEC_TYPE_EAGLE3 => &draft_plan.eagle3,
                SPEC_TYPE_DRAFT => &draft_plan.primary,
                _ => continue,
            };
            if !candidate.final_path.is_empty() {
                return Some(candidate);
            }
        }
        return None;
    }

    for (spec_type, candidate) in [
        (SPEC_TYPE_MTP, &draft_plan.mtp),
        (SPEC_TYPE_DSPARK, &draft_plan.dspark),
        (SPEC_TYPE_DFLASH, &draft_plan.dflash),
        (SPEC_TYPE_EAGLE3, &draft_plan.eagle3),
        (SPEC_TYPE_DRAFT, &draft_plan.primary),
    ] {
        if !candidate.final_path.is_empty() {
            params.spec_types.push(spec_type);
            return Some(candidate);
        }
    }
    None
}

fn parse_spec_types(value: &str) -> Result<Vec<u32>> {
    if value.is_empty() {
        return Err(invalid("empty --spec-type value"));
    }
    let mut out = Vec::new();
    for token in value.split(',') {
        let code = match token.trim().to_ascii_lowercase().as_str() {
            "draft" => SPEC_TYPE_DRAFT,
            "eagle3" | "draft-eagle3" => SPEC_TYPE_EAGLE3,
            "mtp" | "draft-mtp" => SPEC_TYPE_MTP,
            "dflash" | "draft-dflash" => SPEC_TYPE_DFLASH,
            "dspark" | "draft-dspark" => SPEC_TYPE_DSPARK,
            _ => return Err(invalid(format!("unknown --spec-type: {token}"))),
        };
        if !out.contains(&code) {
            out.push(code);
        }
    }
    Ok(out)
}

struct SidecarCandidates<'a> {
    mmproj: &'a [&'a TreeEntry],
    mtp: &'a [&'a TreeEntry],
    dflash: &'a [&'a TreeEntry],
    eagle3: &'a [&'a TreeEntry],
    dspark: &'a [&'a TreeEntry],
}

#[derive(Clone)]
struct TreeEntry {
    path: String,
    oid: String,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RepoFileKind {
    MainGguf,
    Mmproj,
    Mtp,
    Dflash,
    Eagle3,
    Dspark,
    Other,
}

fn classify_repo_file(path: &str) -> RepoFileKind {
    if path.len() < 5 || !path[path.len() - 5..].eq_ignore_ascii_case(".gguf") {
        return RepoFileKind::Other;
    }
    let basename = path.rsplit('/').next().unwrap_or(path);
    if basename.to_ascii_lowercase().contains("mmproj") {
        return RepoFileKind::Mmproj;
    }
    // Preserve upstream case-sensitive "mtp" sidecar recognition so tags such as
    // "-MTP-" remain ordinary model candidates rather than MTP sidecars.
    if basename.contains("mtp") {
        return RepoFileKind::Mtp;
    }
    if basename.contains("dflash") {
        return RepoFileKind::Dflash;
    }
    if basename.contains("eagle3") {
        return RepoFileKind::Eagle3;
    }
    if basename.contains("dspark") {
        return RepoFileKind::Dspark;
    }
    RepoFileKind::MainGguf
}

fn strip_gguf_and_shard(basename: &str) -> (&str, Option<(u32, u32)>) {
    let without_ext =
        if basename.len() >= 5 && basename[basename.len() - 5..].eq_ignore_ascii_case(".gguf") {
            &basename[..basename.len() - 5]
        } else {
            basename
        };
    if let Some((stem, idx, total)) = parse_shard_suffix(without_ext) {
        (stem, Some((idx, total)))
    } else {
        (without_ext, None)
    }
}

fn parse_shard_suffix(stem: &str) -> Option<(&str, u32, u32)> {
    // Expected suffix: -XXXXX-of-YYYYY (15 ASCII bytes)
    if stem.len() < 16 {
        return None;
    }
    let split_pos = stem.len() - 15;
    if !stem.is_char_boundary(split_pos) {
        return None;
    }
    let (prefix, suffix) = stem.split_at(split_pos);
    let bytes = suffix.as_bytes();
    if bytes[0] != b'-' || &bytes[6..10] != b"-of-" {
        return None;
    }
    let idx_str = &suffix[1..6];
    let total_str = &suffix[10..15];
    if !idx_str.bytes().all(|b| b.is_ascii_digit())
        || !total_str.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let idx = idx_str.parse::<u32>().ok()?;
    let total = total_str.parse::<u32>().ok()?;
    if total == 0 || idx == 0 || idx > total {
        return None;
    }
    Some((prefix, idx, total))
}

fn expand_shards<'a>(entries: &'a [TreeEntry], selected: &'a TreeEntry) -> Vec<&'a TreeEntry> {
    let selected_stem = selected
        .path
        .strip_suffix(".gguf")
        .or_else(|| selected.path.strip_suffix(".GGUF"))
        .unwrap_or(&selected.path);
    let Some((prefix, _, total)) = parse_shard_suffix(selected_stem) else {
        return vec![selected];
    };
    let mut matched: Vec<(u32, &TreeEntry)> = Vec::new();
    for entry in entries {
        let entry_stem = if entry.path.len() >= 5
            && entry.path[entry.path.len() - 5..].eq_ignore_ascii_case(".gguf")
        {
            &entry.path[..entry.path.len() - 5]
        } else {
            continue;
        };
        if let Some((entry_prefix, idx, entry_total)) = parse_shard_suffix(entry_stem)
            && entry_prefix == prefix
            && entry_total == total
        {
            matched.push((idx, entry));
        }
    }
    if matched.is_empty() {
        return vec![selected];
    }
    matched.sort_by_key(|(idx, _)| *idx);
    matched.into_iter().map(|(_, entry)| entry).collect()
}

fn is_token_delimiter(byte: u8) -> bool {
    matches!(byte, b'-' | b'.' | b'_')
}

fn matches_quant(path: &str, quant: &str) -> bool {
    if quant.is_empty() {
        return false;
    }
    let basename = path.rsplit('/').next().unwrap_or(path);
    let (stem, _) = strip_gguf_and_shard(basename);
    if stem.eq_ignore_ascii_case(quant) {
        return true;
    }
    let stem_lower = stem.to_ascii_lowercase();
    let quant_lower = quant.to_ascii_lowercase();
    let stem_bytes = stem_lower.as_bytes();
    let qlen = quant_lower.len();
    let mut start = 0_usize;
    while let Some(offset) = stem_lower[start..].find(&quant_lower) {
        let idx = start + offset;
        let left_ok = idx == 0 || is_token_delimiter(stem_bytes[idx - 1]);
        let right_idx = idx + qlen;
        let right_ok = right_idx == stem_bytes.len() || is_token_delimiter(stem_bytes[right_idx]);
        if left_ok && right_ok {
            return true;
        }
        start = idx + 1;
    }
    if let Some((dir, _)) = path.rsplit_once('/') {
        for segment in dir.split('/') {
            if segment.eq_ignore_ascii_case(quant) {
                return true;
            }
        }
    }
    false
}

fn extract_quant_tag(path: &str) -> Option<String> {
    let basename = path.rsplit('/').next().unwrap_or(path);
    let (stem, _) = strip_gguf_and_shard(basename);
    if let Some(pos) = stem.rfind(['-', '.']) {
        let tail = &stem[pos + 1..];
        if !tail.is_empty() {
            return Some(tail.to_owned());
        }
    }
    if !stem.is_empty() {
        return Some(stem.to_owned());
    }
    None
}

fn select_mmproj<'a>(
    candidates: &[&'a TreeEntry],
    requested_tag: Option<&str>,
) -> Option<&'a TreeEntry> {
    if candidates.is_empty() {
        return None;
    }
    if let Some(tag) = requested_tag
        && let Some(found) = candidates
            .iter()
            .copied()
            .find(|e| matches_quant(&e.path, tag))
    {
        return Some(found);
    }
    for priority in [
        "F16", "BF16", "F32", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0",
    ] {
        if let Some(found) = candidates
            .iter()
            .copied()
            .find(|e| matches_quant(&e.path, priority))
        {
            return Some(found);
        }
    }
    candidates.first().copied()
}

fn select_draft_sidecar<'a>(
    candidates: &[&'a TreeEntry],
    requested_tag: Option<&str>,
) -> Option<&'a TreeEntry> {
    if candidates.is_empty() {
        return None;
    }
    if let Some(tag) = requested_tag {
        if let Some(found) = candidates
            .iter()
            .copied()
            .find(|e| matches_quant(&e.path, tag))
        {
            return Some(found);
        }
        if tag.to_ascii_uppercase().starts_with("Q4") {
            for family in [
                "Q4_K_M", "Q4_K_S", "Q4_K", "Q4_0", "Q4_1", "IQ4_NL", "IQ4_XS",
            ] {
                if let Some(found) = candidates
                    .iter()
                    .copied()
                    .find(|e| matches_quant(&e.path, family))
                {
                    return Some(found);
                }
            }
        }
    }
    for priority in [
        "Q8_0", "Q4_K_M", "Q4_K_S", "Q4_0", "Q5_K_M", "Q6_K", "Q3_K_M", "Q2_K", "MXFP4", "BF16",
        "F16", "F32",
    ] {
        if let Some(found) = candidates
            .iter()
            .copied()
            .find(|e| matches_quant(&e.path, priority))
        {
            return Some(found);
        }
    }
    candidates.first().copied()
}

fn parse_hf_repo(raw: &str) -> Result<(String, Option<String>)> {
    if raw.is_empty() || raw.len() > MAX_REPO_BYTES || raw.contains('\0') {
        return Err(invalid("invalid HF repo length or NUL byte"));
    }
    let trimmed = raw
        .strip_prefix("https://huggingface.co/")
        .or_else(|| raw.strip_prefix("http://huggingface.co/"))
        .or_else(|| raw.strip_prefix("hf.co/"))
        .unwrap_or(raw)
        .trim_matches('/');
    let (repo_part, tag) = match trimmed.rsplit_once(':') {
        Some((repo, tag)) => {
            if tag.is_empty() || tag.contains('/') || tag.contains("..") {
                return Err(invalid("invalid HF quant tag"));
            }
            (repo, Some(tag.to_owned()))
        }
        None => (trimmed, None),
    };
    let mut parts = repo_part.split('/');
    let (Some(owner), Some(name), None) = (parts.next(), parts.next(), parts.next()) else {
        return Err(invalid("HF repo must have owner/name format"));
    };
    if !is_safe_repo_component(owner) || !is_safe_repo_component(name) {
        return Err(invalid("HF repo contains invalid path characters"));
    }
    Ok((format!("{owner}/{name}"), tag))
}

fn is_safe_repo_component(component: &str) -> bool {
    !component.is_empty()
        && component != "."
        && component != ".."
        && !component.contains("..")
        && component
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
}

fn validate_commit(commit: &str) -> Result<()> {
    if commit.is_empty()
        || commit.len() > 128
        || commit.contains(['/', '\\', '\0'])
        || commit.contains("..")
    {
        return Err(invalid("invalid HF commit hash"));
    }
    Ok(())
}

fn extract_main_commit(refs_json: &Value) -> Result<String> {
    let branches = refs_json["branches"]
        .as_array()
        .ok_or_else(|| invalid("HF refs response missing branches array"))?;
    for branch in branches {
        if branch["name"].as_str() == Some("main")
            && let Some(commit) = branch["targetCommit"].as_str()
        {
            return Ok(commit.to_owned());
        }
    }
    if let Some(tags) = refs_json["tags"].as_array() {
        for tag in tags {
            if tag["name"].as_str() == Some("main")
                && let Some(commit) = tag["targetCommit"].as_str()
            {
                return Ok(commit.to_owned());
            }
        }
    }
    Err(invalid("HF refs response missing main targetCommit"))
}

fn validate_relative_repo_path(path: &str) -> Result<()> {
    if path.is_empty()
        || path.len() > MAX_PATH_BYTES
        || path.contains(['\0', '\\'])
        || path.starts_with('/')
    {
        return Err(invalid("invalid repository file path"));
    }
    for segment in path.split('/') {
        if segment.is_empty() || segment == "." || segment == ".." {
            return Err(invalid("repository file path contains traversal segment"));
        }
    }
    Ok(())
}

fn collect_cached_tree(root: &Path, dir: &Path, out: &mut Vec<TreeEntry>) -> Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_cached_tree(root, &path, out)?;
        } else if path.is_file() {
            let rel = path
                .strip_prefix(root)
                .map_err(|_| invalid("failed to relativize cached snapshot file"))?;
            let rel_str = rel
                .iter()
                .map(|part| {
                    part.to_str()
                        .ok_or_else(|| invalid("non-UTF-8 cached file component"))
                })
                .collect::<Result<Vec<_>>>()?
                .join("/");
            if !rel_str.starts_with(".tmp-") {
                out.push(TreeEntry {
                    path: rel_str,
                    oid: String::new(),
                });
            }
        }
    }
    Ok(())
}

// ============================================================================
// Atomic cache file writes & SHA-256 verification
// ============================================================================

struct TempFileGuard {
    path: PathBuf,
    committed: bool,
}

impl TempFileGuard {
    const fn new(path: PathBuf) -> Self {
        Self {
            path,
            committed: false,
        }
    }

    const fn disarm(&mut self) {
        self.committed = true;
    }
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        if !self.committed {
            let _ = fs::remove_file(&self.path);
        }
    }
}

fn write_atomic(dest: &Path, bytes: &[u8]) -> Result<()> {
    static NEXT_NONCE: AtomicU64 = AtomicU64::new(1);
    let parent = dest
        .parent()
        .ok_or_else(|| invalid("cache target path has no parent"))?;
    fs::create_dir_all(parent)?;
    for _ in 0..64 {
        let seq = NEXT_NONCE.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let tmp_path = parent.join(format!(".tmp-{}-{nanos}-{seq}", std::process::id()));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp_path)
        {
            Ok(mut file) => {
                let mut guard = TempFileGuard::new(tmp_path.clone());
                file.write_all(bytes)?;
                file.flush()?;
                file.sync_all()?;
                drop(file);
                if let Err(error) = fs::rename(&tmp_path, dest) {
                    if error.kind() == ErrorKind::AlreadyExists {
                        fs::remove_file(dest)?;
                        fs::rename(&tmp_path, dest)?;
                    } else {
                        return Err(error.into());
                    }
                }
                guard.disarm();
                return Ok(());
            }
            Err(error) if error.kind() == ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error.into()),
        }
    }
    Err(invalid("unable to allocate unique temporary cache file"))
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

// ============================================================================
// Clean-room HTTP/1.1 client over std::net::TcpStream
// ============================================================================

#[derive(Clone, Copy, PartialEq, Eq)]
enum HttpMethod {
    Get,
    Head,
}

impl HttpMethod {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Get => "GET",
            Self::Head => "HEAD",
        }
    }
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

struct ParsedContentRange {
    start: u64,
    end: u64,
    total: Option<u64>,
}

struct HttpResponse {
    status_code: u16,
    content_length: Option<usize>,
    etag: Option<String>,
    content_disposition: Option<String>,
    disposition_filename: Option<String>,
    content_range: Option<String>,
    parsed_range: Option<ParsedContentRange>,
    body: Vec<u8>,
}

struct ResponseHead {
    status_code: u16,
    content_length: Option<usize>,
    is_chunked: bool,
    location: Option<String>,
    etag: Option<String>,
    content_disposition: Option<String>,
    content_range: Option<String>,
}

fn parse_port(text: &str) -> Result<u16> {
    let port = text
        .parse::<u16>()
        .map_err(|_| invalid("invalid port in HTTP URL"))?;
    if port == 0 {
        return Err(invalid("HTTP URL port must be nonzero"));
    }
    Ok(port)
}

fn parse_http_url(url: &str) -> Result<ParsedUrl> {
    if url.starts_with("https://") {
        return Err(invalid(
            "HTTPS URLs require TLS and are unsupported by plain HTTP/1.1 client",
        ));
    }
    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| invalid("HTTP URL must start with http://"))?;
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
        return Err(invalid("malformed HTTP URL"));
    }
    let (connect_host, host_header, port) = if let Some(after_open) = authority.strip_prefix('[') {
        let (inside, after) = after_open
            .split_once(']')
            .ok_or_else(|| invalid("invalid IPv6 host in HTTP URL"))?;
        if inside.is_empty() {
            return Err(invalid("empty IPv6 host in HTTP URL"));
        }
        let port = if after.is_empty() {
            80
        } else if let Some(port_text) = after.strip_prefix(':') {
            parse_port(port_text)?
        } else {
            return Err(invalid("invalid port after IPv6 host in HTTP URL"));
        };
        let header = if port == 80 {
            format!("[{inside}]")
        } else {
            format!("[{inside}]:{port}")
        };
        (inside.to_owned(), header, port)
    } else if let Some((host, port_text)) = authority.split_once(':') {
        if host.is_empty() {
            return Err(invalid("empty host in HTTP URL"));
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

fn resolve_redirect(base: &ParsedUrl, location: &str) -> Result<String> {
    let location = location.trim();
    if location.is_empty() || location.chars().any(char::is_control) {
        return Err(invalid("invalid HTTP redirect Location header"));
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

    fn read_header_block(&mut self) -> Result<Vec<u8>> {
        if self.pos > 0 {
            self.buf.drain(..self.pos);
            self.pos = 0;
        }
        loop {
            if let Some((end, sep_len)) = find_header_end(&self.buf) {
                if end > MAX_HTTP_HEADER_BYTES {
                    return Err(invalid("HTTP response headers exceed 64 KiB"));
                }
                let block = self.buf[..end].to_vec();
                self.pos = end + sep_len;
                return Ok(block);
            }
            if self.buf.len() >= MAX_HTTP_HEADER_BYTES {
                return Err(invalid("HTTP response headers exceed 64 KiB"));
            }
            let mut chunk = [0_u8; 4096];
            let count = read_stream_chunk(self.stream, &mut chunk)
                .map_err(|error| invalid(format!("HTTP header read failed: {error}")))?;
            if count == 0 {
                return Err(invalid("unexpected EOF reading HTTP response headers"));
            }
            self.buf.extend_from_slice(&chunk[..count]);
        }
    }

    fn read_line_crlf(&mut self) -> Result<String> {
        loop {
            if let Some(rel_end) = self.buf[self.pos..].iter().position(|&byte| byte == b'\n') {
                let line_end = self.pos + rel_end;
                let raw = &self.buf[self.pos..line_end];
                let trimmed = raw.strip_suffix(b"\r").unwrap_or(raw);
                if trimmed.len() > MAX_HTTP_LINE_BYTES {
                    return Err(invalid("HTTP chunk line exceeds 8 KiB"));
                }
                let line = std::str::from_utf8(trimmed)
                    .map_err(|_| invalid("non-UTF-8 HTTP chunk line"))?
                    .to_owned();
                self.pos = line_end + 1;
                return Ok(line);
            }
            if self.buf.len() - self.pos > MAX_HTTP_LINE_BYTES {
                return Err(invalid("HTTP chunk line exceeds 8 KiB"));
            }
            if self.pos > 0 {
                self.buf.drain(..self.pos);
                self.pos = 0;
            }
            let mut chunk = [0_u8; 4096];
            let count = read_stream_chunk(self.stream, &mut chunk)
                .map_err(|error| invalid(format!("HTTP chunk read failed: {error}")))?;
            if count == 0 {
                return Err(invalid("unexpected EOF reading HTTP chunked response"));
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

fn parse_response_headers(block: &[u8]) -> Result<ResponseHead> {
    let lossy = String::from_utf8_lossy(block);
    let mut lines = lossy
        .split('\n')
        .map(|line| line.strip_suffix('\r').unwrap_or(line));
    let status_line = lines
        .next()
        .ok_or_else(|| invalid("missing HTTP status line"))?;
    let rest = status_line
        .strip_prefix("HTTP/1.1 ")
        .or_else(|| status_line.strip_prefix("HTTP/1.0 "))
        .ok_or_else(|| invalid(format!("unsupported HTTP status line: {status_line}")))?;
    let code_str = rest
        .split_whitespace()
        .next()
        .ok_or_else(|| invalid("missing HTTP status code"))?;
    if code_str.len() != 3 || !code_str.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(invalid(format!("invalid HTTP status code: {code_str}")));
    }
    let status_code = code_str
        .parse::<u16>()
        .map_err(|_| invalid("invalid HTTP status code"))?;
    let mut content_length: Option<usize> = None;
    let mut is_chunked = false;
    let mut location: Option<String> = None;
    let mut etag: Option<String> = None;
    let mut content_disposition: Option<String> = None;
    let mut content_range: Option<String> = None;
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let Some((name, value)) = line.split_once(':') else {
            return Err(invalid("malformed HTTP response header"));
        };
        let name = name.trim();
        let value = value.trim();
        if name.eq_ignore_ascii_case("content-length") {
            let parsed = value
                .parse::<usize>()
                .map_err(|_| invalid("invalid HTTP Content-Length header"))?;
            if content_length.is_some_and(|prev| prev != parsed) {
                return Err(invalid("conflicting HTTP Content-Length headers"));
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
        } else if name.eq_ignore_ascii_case("etag") && etag.is_none() {
            etag = Some(value.to_owned());
        } else if name.eq_ignore_ascii_case("content-disposition") && content_disposition.is_none()
        {
            content_disposition = Some(value.to_owned());
        } else if name.eq_ignore_ascii_case("content-range") && content_range.is_none() {
            content_range = Some(value.to_owned());
        }
    }
    Ok(ResponseHead {
        status_code,
        content_length,
        is_chunked,
        location,
        etag,
        content_disposition,
        content_range,
    })
}

fn percent_decode_utf8(raw: &str) -> Option<String> {
    let bytes = raw.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0_usize;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            if i + 2 >= bytes.len() {
                return None;
            }
            let hex_pair = std::str::from_utf8(&bytes[i + 1..i + 3]).ok()?;
            let byte = u8::from_str_radix(hex_pair, 16).ok()?;
            out.push(byte);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

fn sanitize_disposition_filename(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.chars().any(char::is_control) {
        return None;
    }
    let basename = trimmed.rsplit(['/', '\\']).next()?;
    if basename.is_empty() || basename == "." || basename == ".." {
        return None;
    }
    Some(basename.to_owned())
}

fn parse_content_disposition_filename(header: &str) -> Option<String> {
    let mut ext_name: Option<String> = None;
    let mut plain_name: Option<String> = None;
    for part in header.split(';') {
        let trimmed = part.trim();
        let Some((key, value)) = trimmed.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim();
        if key.eq_ignore_ascii_case("filename*") {
            let unquoted = value
                .strip_prefix('"')
                .and_then(|s| s.strip_suffix('"'))
                .unwrap_or(value);
            if let Some((_, encoded)) = unquoted.split_once("''")
                && let Some(decoded) = percent_decode_utf8(encoded)
                && let Some(clean) = sanitize_disposition_filename(&decoded)
            {
                ext_name = Some(clean);
            }
        } else if key.eq_ignore_ascii_case("filename") {
            let unquoted = value
                .strip_prefix('"')
                .and_then(|s| s.strip_suffix('"'))
                .unwrap_or(value);
            if let Some(clean) = sanitize_disposition_filename(unquoted) {
                plain_name = Some(clean);
            }
        }
    }
    ext_name.or(plain_name)
}

fn parse_content_range(header: &str) -> Result<ParsedContentRange> {
    let trimmed = header.trim();
    let rest = trimmed
        .strip_prefix("bytes ")
        .or_else(|| trimmed.strip_prefix("BYTES "))
        .ok_or_else(|| invalid("Content-Range must use bytes unit"))?;
    let (span, total_str) = rest
        .split_once('/')
        .ok_or_else(|| invalid("Content-Range missing '/' separator"))?;
    let (start_str, end_str) = span
        .split_once('-')
        .ok_or_else(|| invalid("Content-Range missing byte span"))?;
    let start = start_str
        .trim()
        .parse::<u64>()
        .map_err(|_| invalid("invalid Content-Range start"))?;
    let end = end_str
        .trim()
        .parse::<u64>()
        .map_err(|_| invalid("invalid Content-Range end"))?;
    if start > end {
        return Err(invalid("Content-Range start exceeds end"));
    }
    let total = if total_str.trim() == "*" {
        None
    } else {
        let parsed = total_str
            .trim()
            .parse::<u64>()
            .map_err(|_| invalid("invalid Content-Range total"))?;
        if end >= parsed {
            return Err(invalid("Content-Range end exceeds total length"));
        }
        Some(parsed)
    };
    Ok(ParsedContentRange { start, end, total })
}

fn read_chunked_body(reader: &mut BufferedReader<'_>, max_allowed: usize) -> Result<Vec<u8>> {
    let mut body = Vec::new();
    loop {
        let size_line = reader.read_line_crlf()?;
        let hex_part = size_line
            .split_once(';')
            .map_or(size_line.as_str(), |(size, _)| size)
            .trim();
        let chunk_size =
            usize::from_str_radix(hex_part, 16).map_err(|_| invalid("invalid HTTP chunk size"))?;
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
            .ok_or_else(|| invalid("HTTP chunked response exceeds size bound"))?;
        if next_len > max_allowed {
            return Err(invalid("HTTP chunked response exceeds size bound"));
        }
        let start = body.len();
        body.resize(next_len, 0);
        reader
            .read_exact(&mut body[start..])
            .map_err(|error| invalid(format!("HTTP chunk payload read failed: {error}")))?;
        if !reader.read_line_crlf()?.is_empty() {
            return Err(invalid("invalid HTTP chunk terminator"));
        }
    }
}

fn read_exact_bounded(
    reader: &mut BufferedReader<'_>,
    content_length: usize,
    max_allowed: usize,
) -> Result<Vec<u8>> {
    if content_length > max_allowed {
        return Err(invalid("HTTP response body exceeds size bound"));
    }
    let mut body = vec![0_u8; content_length];
    reader
        .read_exact(&mut body)
        .map_err(|error| invalid(format!("HTTP response body read failed: {error}")))?;
    Ok(body)
}

fn read_to_eof_bounded(reader: &mut BufferedReader<'_>, max_allowed: usize) -> Result<Vec<u8>> {
    let mut body = Vec::new();
    let mut chunk = [0_u8; 8192];
    loop {
        let count = match reader.read(&mut chunk) {
            Ok(0) => return Ok(body),
            Ok(n) => n,
            Err(error) if error.kind() == ErrorKind::Interrupted => continue,
            Err(error) => return Err(invalid(format!("HTTP response body read failed: {error}"))),
        };
        let next_len = body
            .len()
            .checked_add(count)
            .ok_or_else(|| invalid("HTTP response body exceeds size bound"))?;
        if next_len > max_allowed {
            return Err(invalid("HTTP response body exceeds size bound"));
        }
        body.extend_from_slice(&chunk[..count]);
    }
}

fn connect_http(parsed: &ParsedUrl, timeout: Duration) -> Result<TcpStream> {
    let addresses = (parsed.connect_host.as_str(), parsed.port)
        .to_socket_addrs()
        .map_err(|error| invalid(format!("HTTP DNS resolution failed: {error}")))?;
    let mut last_error = None;
    for address in addresses {
        match TcpStream::connect_timeout(&address, timeout) {
            Ok(stream) => {
                stream
                    .set_read_timeout(Some(timeout))
                    .map_err(|error| invalid(format!("failed to set read timeout: {error}")))?;
                stream
                    .set_write_timeout(Some(timeout))
                    .map_err(|error| invalid(format!("failed to set write timeout: {error}")))?;
                return Ok(stream);
            }
            Err(error) => last_error = Some(error),
        }
    }
    Err(last_error.map_or_else(
        || invalid("no socket addresses resolved for HTTP host"),
        |error| invalid(format!("HTTP connect failed: {error}")),
    ))
}

fn build_http_request(
    method: HttpMethod,
    parsed: &ParsedUrl,
    headers: &[HeaderEntry<'_>],
    same_origin: bool,
) -> String {
    use std::fmt::Write as _;

    let mut request = format!("{} {} HTTP/1.1\r\n", method.as_str(), parsed.target);
    let custom_host = same_origin
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
            || (!same_origin
                && (header.name.eq_ignore_ascii_case("host")
                    || header.name.eq_ignore_ascii_case("authorization")))
        {
            continue;
        }
        let _ = write!(request, "{}: {}\r\n", header.name, header.value);
    }
    request.push_str("\r\n");
    request
}

enum FetchOutcome {
    Redirect { status: u16, location: String },
    Complete(HttpResponse),
}

fn perform_http_once(
    method: HttpMethod,
    parsed: &ParsedUrl,
    headers: &[HeaderEntry<'_>],
    same_origin: bool,
    timeout: Duration,
    max_allowed: usize,
) -> Result<FetchOutcome> {
    let mut stream = connect_http(parsed, timeout)?;
    let request = build_http_request(method, parsed, headers, same_origin);
    stream
        .write_all(request.as_bytes())
        .map_err(|error| invalid(format!("HTTP request write failed: {error}")))?;
    stream
        .flush()
        .map_err(|error| invalid(format!("HTTP request flush failed: {error}")))?;

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
        return Ok(FetchOutcome::Redirect {
            status: head.status_code,
            location: resolve_redirect(parsed, location)?,
        });
    }

    let body = if method == HttpMethod::Head || matches!(head.status_code, 204 | 304) {
        Vec::new()
    } else if head.is_chunked {
        read_chunked_body(&mut reader, max_allowed)?
    } else if let Some(content_length) = head.content_length {
        read_exact_bounded(&mut reader, content_length, max_allowed)?
    } else {
        read_to_eof_bounded(&mut reader, max_allowed)?
    };

    let disposition_filename = head
        .content_disposition
        .as_deref()
        .and_then(parse_content_disposition_filename);
    let parsed_range = head
        .content_range
        .as_deref()
        .map(parse_content_range)
        .transpose()?;

    Ok(FetchOutcome::Complete(HttpResponse {
        status_code: head.status_code,
        content_length: head.content_length,
        etag: head.etag,
        content_disposition: head.content_disposition,
        disposition_filename,
        content_range: head.content_range,
        parsed_range,
        body,
    }))
}

fn perform_http_request(
    mut method: HttpMethod,
    initial_url: &str,
    headers: &[HeaderEntry<'_>],
    timeout: Duration,
    max_allowed: usize,
) -> Result<HttpResponse> {
    let mut current_url = initial_url.to_owned();
    let mut same_origin = true;
    let initial_authority = parse_http_url(initial_url)?.host_header;
    for redirect in 0..=MAX_REDIRECTS {
        let parsed = parse_http_url(&current_url)?;
        if parsed.host_header != initial_authority {
            same_origin = false;
        }
        match perform_http_once(method, &parsed, headers, same_origin, timeout, max_allowed)? {
            FetchOutcome::Complete(response) => return Ok(response),
            FetchOutcome::Redirect { status, location } => {
                if redirect == MAX_REDIRECTS {
                    return Err(invalid("HTTP redirect limit exceeded"));
                }
                if status == 303 && method != HttpMethod::Head {
                    method = HttpMethod::Get;
                }
                current_url = location;
            }
        }
    }
    Err(invalid("HTTP redirect limit exceeded"))
}

// ============================================================================
// Wire validation and clean-room JSON execution
// ============================================================================

fn exact_fields<'a>(value: &'a Value, names: &[&str]) -> Result<&'a Map<String, Value>> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid("resolution object required"))?;
    if object.len() != names.len() || names.iter().any(|name| !object.contains_key(*name)) {
        return Err(invalid("resolution object fields mismatch"));
    }
    Ok(object)
}

fn text(value: &Value, maximum: usize) -> Result<&str> {
    let s = value
        .as_str()
        .ok_or_else(|| invalid("resolution string required"))?;
    if s.len() > maximum || s.contains('\0') {
        return Err(invalid("resolution string extent or NUL violation"));
    }
    Ok(s)
}

fn flag(value: &Value, name: &str) -> Result<bool> {
    value
        .as_bool()
        .ok_or_else(|| invalid(format!("resolution {name} must be boolean")))
}

fn parse_example(value: &Value) -> Result<u32> {
    let raw = value
        .as_u64()
        .ok_or_else(|| invalid("resolution Example must be unsigned integer"))?;
    let example = u32::try_from(raw)?;
    if example >= EXAMPLE_COUNT {
        return Err(invalid("resolution Example outside enum"));
    }
    Ok(example)
}

fn parse_argv(value: &Value) -> Result<Vec<String>> {
    let array = value
        .as_array()
        .ok_or_else(|| invalid("resolution Arguments must be an array"))?;
    if array.is_empty() || array.len() > MAX_ARGV_COUNT {
        return Err(invalid("resolution argv count outside bound"));
    }
    let mut total = 0_usize;
    let mut out = Vec::with_capacity(array.len());
    for item in array {
        let s = text(item, MAX_ARGV_ITEM_BYTES)?;
        total += s.len();
        if total > MAX_ARGV_TOTAL_BYTES {
            return Err(invalid("resolution argv bytes exceed 64 KiB"));
        }
        out.push(s.to_owned());
    }
    Ok(out)
}

fn parse_plan_options(value: &Value) -> Result<Options> {
    exact_fields(value, &["Mmproj", "Mtp", "Dflash", "Eagle3", "Dspark"])?;
    Ok(Options {
        mmproj: flag(&value["Mmproj"], "Mmproj")?,
        mtp: flag(&value["Mtp"], "Mtp")?,
        dflash: flag(&value["Dflash"], "Dflash")?,
        eagle3: flag(&value["Eagle3"], "Eagle3")?,
        dspark: flag(&value["Dspark"], "Dspark")?,
    })
}

fn file_to_json(file: &ResolvedFile) -> Value {
    json!({
        "Path": file.path,
        "Url": file.url,
        "LocalPath": file.local_path,
        "FinalPath": file.final_path,
        "Oid": file.oid,
        "Repo": file.repo,
    })
}

fn plan_to_json(plan: &Plan) -> Value {
    json!({
        "Primary": file_to_json(&plan.primary),
        "ModelFiles": plan.model_files.iter().map(file_to_json).collect::<Vec<_>>(),
        "Mmproj": file_to_json(&plan.mmproj),
        "Mtp": file_to_json(&plan.mtp),
        "Dflash": file_to_json(&plan.dflash),
        "Eagle3": file_to_json(&plan.eagle3),
        "Dspark": file_to_json(&plan.dspark),
        "Preset": file_to_json(&plan.preset),
    })
}

fn params_to_json(params: &Params) -> Value {
    json!({
        "ModelPath": params.model.path,
        "ModelHfRepo": params.model.hf_repo,
        "MmprojPath": params.mmproj.path,
        "DraftPath": params.draft.path,
        "SpecTypes": params.spec_types,
        "Preset": params.preset,
        "Offline": params.offline,
    })
}

fn execute_handler(
    resolver: &Resolver,
    arguments: &[String],
    example: u32,
    offline_after_init: bool,
    apply: bool,
) -> Result<Value> {
    let Ok(mut params) = Params::parse(arguments) else {
        return Ok(json!({
            "Parsed": false,
            "Params": params_to_json(&Params::default()),
            "Plan": Value::Null,
            "DraftPlan": Value::Null,
        }));
    };
    let handler = Handler::initialize(resolver, &params, example)?;
    params.offline = offline_after_init;
    if apply {
        handler.apply(resolver, &mut params)?;
    }
    Ok(json!({
        "Parsed": true,
        "Params": params_to_json(&params),
        "Plan": plan_to_json(&handler.plan),
        "DraftPlan": plan_to_json(&handler.draft_plan),
    }))
}

fn validate_relative_cache_path(raw: &str) -> Result<PathBuf> {
    if raw.is_empty() || raw.len() > MAX_PATH_BYTES || raw.contains('\0') {
        return Err(invalid("cache relative path empty or exceeds bound"));
    }
    let candidate = Path::new(raw);
    if candidate.is_absolute() {
        return Err(invalid("cache relative path must not be absolute"));
    }
    for component in candidate.components() {
        if !matches!(component, Component::Normal(_)) {
            return Err(invalid(
                "cache relative path must use normal components only",
            ));
        }
    }
    Ok(candidate.to_path_buf())
}

fn validate_optional_u64(value: &Value, name: &str) -> Result<Option<u64>> {
    if value.is_null() {
        return Ok(None);
    }
    value
        .as_u64()
        .map(Some)
        .ok_or_else(|| invalid(format!("{name} must be null or unsigned integer")))
}

fn validate_optional_sha256(value: &Value) -> Result<Option<String>> {
    if value.is_null() {
        return Ok(None);
    }
    let hex_str = text(value, 64)?;
    if hex_str.len() != 64 || !hex_str.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(invalid("ExpectedSha256 must be 64 hex digits"));
    }
    Ok(Some(hex_str.to_ascii_lowercase()))
}

fn validate_optional_string(value: &Value, name: &str, max_len: usize) -> Result<Option<String>> {
    if value.is_null() {
        return Ok(None);
    }
    let s = text(value, max_len)?;
    if s.contains(['\r', '\n']) {
        return Err(invalid(format!("{name} contains newline")));
    }
    Ok(Some(s.to_owned()))
}

fn verify_cached_bytes(
    bytes: &[u8],
    expected_bytes: Option<u64>,
    expected_sha256: Option<&str>,
) -> Result<String> {
    let actual_len = u64::try_from(bytes.len())?;
    if let Some(expected_len) = expected_bytes
        && actual_len != expected_len
    {
        return Err(invalid(format!(
            "cached/downloaded byte count {actual_len} differs from ExpectedBytes {expected_len}"
        )));
    }
    let digest = sha256_hex(bytes);
    if let Some(expected_hash) = expected_sha256
        && digest != expected_hash
    {
        return Err(invalid(format!(
            "cached/downloaded SHA-256 {digest} differs from ExpectedSha256 {expected_hash}"
        )));
    }
    Ok(digest)
}

fn parse_http_headers_field(value: &Value) -> Result<Vec<(String, String)>> {
    let array = value
        .as_array()
        .ok_or_else(|| invalid("Headers must be an array"))?;
    if array.len() > MAX_HEADERS {
        return Err(invalid("Headers count exceeds 32"));
    }
    let mut headers = Vec::with_capacity(array.len());
    for entry in array {
        exact_fields(entry, &["Name", "Value"])?;
        let name = text(&entry["Name"], MAX_HEADER_FIELD_BYTES)?;
        let val = text(&entry["Value"], MAX_HEADER_FIELD_BYTES)?;
        if name.is_empty() || name.contains(['\r', '\n', ':']) || val.contains(['\r', '\n']) {
            return Err(invalid(
                "HTTP header name/value contains invalid characters",
            ));
        }
        headers.push((name.to_owned(), val.to_owned()));
    }
    Ok(headers)
}

fn validate_http_operation(operation: &Value) -> Result<()> {
    exact_fields(
        operation,
        &[
            "Kind",
            "Method",
            "Url",
            "CacheDir",
            "CachePath",
            "Headers",
            "RangeStart",
            "IfNoneMatch",
            "Offline",
            "ExpectExistingFile",
            "ExpectedBytes",
            "ExpectedSha256",
            "TimeoutSeconds",
        ],
    )?;
    let method = text(&operation["Method"], 16)?;
    if !matches!(method, "GET" | "HEAD") {
        return Err(invalid("model-resolve-http Method must be GET or HEAD"));
    }
    let url = text(&operation["Url"], MAX_URL_BYTES)?;
    let _ = parse_http_url(url)?;
    let cache_dir = Path::new(text(&operation["CacheDir"], MAX_PATH_BYTES)?);
    if !cache_dir.is_absolute() {
        return Err(invalid("model-resolve-http CacheDir must be absolute"));
    }
    let _ = validate_relative_cache_path(text(&operation["CachePath"], MAX_PATH_BYTES)?)?;
    let _ = parse_http_headers_field(&operation["Headers"])?;
    let _ = validate_optional_u64(&operation["RangeStart"], "RangeStart")?;
    let _ = validate_optional_string(
        &operation["IfNoneMatch"],
        "IfNoneMatch",
        MAX_HEADER_FIELD_BYTES,
    )?;
    let _ = flag(&operation["Offline"], "Offline")?;
    let _ = flag(&operation["ExpectExistingFile"], "ExpectExistingFile")?;
    let _ = validate_optional_u64(&operation["ExpectedBytes"], "ExpectedBytes")?;
    let _ = validate_optional_sha256(&operation["ExpectedSha256"])?;
    if !operation["TimeoutSeconds"].is_null() {
        let secs = operation["TimeoutSeconds"]
            .as_u64()
            .ok_or_else(|| invalid("TimeoutSeconds must be unsigned integer or null"))?;
        if secs == 0 || secs > MAX_TIMEOUT_SECS {
            return Err(invalid("TimeoutSeconds must be between 1 and 60"));
        }
    }
    Ok(())
}

fn cached_hit_response(
    dest_path: &Path,
    http_code: Option<u16>,
    response: Option<&HttpResponse>,
    expected_bytes: Option<u64>,
    expected_sha256: Option<&str>,
) -> Result<Value> {
    let existing = fs::read(dest_path)?;
    let digest = verify_cached_bytes(&existing, expected_bytes, expected_sha256)?;
    Ok(json!({
        "HttpCode": http_code,
        "Cached": true,
        "Path": dest_path,
        "Bytes": existing.len(),
        "Sha256": digest,
        "ETag": response.and_then(|r| r.etag.clone()),
        "ContentDisposition": response.and_then(|r| r.content_disposition.clone()),
        "DispositionFilename": response.and_then(|r| r.disposition_filename.clone()),
        "ContentRange": response.and_then(|r| r.content_range.clone()),
        "ContentLength": response.and_then(|r| r.content_length),
    }))
}

fn assemble_partial_206(
    dest_path: &Path,
    range_start: Option<u64>,
    response: HttpResponse,
) -> Result<(Vec<u8>, HttpResponseMeta)> {
    let start = range_start.ok_or_else(|| invalid("HTTP 206 returned without Range request"))?;
    let parsed_range = response
        .parsed_range
        .as_ref()
        .ok_or_else(|| invalid("HTTP 206 missing Content-Range header"))?;
    if parsed_range.start != start {
        return Err(invalid("HTTP 206 Content-Range start mismatch"));
    }
    let expected_slice_len = parsed_range.end - parsed_range.start + 1;
    if u64::try_from(response.body.len())? != expected_slice_len {
        return Err(invalid("HTTP 206 body length differs from Content-Range"));
    }
    if let Some(total) = parsed_range.total
        && parsed_range.end + 1 > total
    {
        return Err(invalid("HTTP 206 Content-Range end exceeds total"));
    }
    let meta = HttpResponseMeta {
        status_code: response.status_code,
        etag: response.etag,
        content_disposition: response.content_disposition,
        disposition_filename: response.disposition_filename,
        content_range: response.content_range,
        content_length: response.content_length,
    };
    if start == 0 {
        return Ok((response.body, meta));
    }
    let prefix = fs::read(dest_path)
        .map_err(|_| invalid("HTTP 206 ranged resume requires existing partial cache file"))?;
    let prefix_len = usize::try_from(start)?;
    if prefix.len() < prefix_len {
        return Err(invalid("existing cached prefix shorter than RangeStart"));
    }
    let mut merged = Vec::with_capacity(prefix_len + response.body.len());
    merged.extend_from_slice(&prefix[..prefix_len]);
    merged.extend_from_slice(&response.body);
    Ok((merged, meta))
}

struct HttpResponseMeta {
    status_code: u16,
    etag: Option<String>,
    content_disposition: Option<String>,
    disposition_filename: Option<String>,
    content_range: Option<String>,
    content_length: Option<usize>,
}

struct HttpOpRequest<'a> {
    method: HttpMethod,
    url: &'a str,
    dest_path: PathBuf,
    headers: Vec<(String, String)>,
    range_start: Option<u64>,
    if_none_match: Option<String>,
    offline: bool,
    expect_existing: bool,
    expected_bytes: Option<u64>,
    expected_sha256: Option<String>,
    timeout_secs: u64,
}

fn parse_http_op_request(operation: &Value) -> Result<HttpOpRequest<'_>> {
    let method = match text(&operation["Method"], 16)? {
        "GET" => HttpMethod::Get,
        "HEAD" => HttpMethod::Head,
        _ => return Err(invalid("unsupported HTTP method")),
    };
    let url = text(&operation["Url"], MAX_URL_BYTES)?;
    let cache_dir = PathBuf::from(text(&operation["CacheDir"], MAX_PATH_BYTES)?);
    let dest_path = cache_dir.join(validate_relative_cache_path(text(
        &operation["CachePath"],
        MAX_PATH_BYTES,
    )?)?);
    Ok(HttpOpRequest {
        method,
        url,
        dest_path,
        headers: parse_http_headers_field(&operation["Headers"])?,
        range_start: validate_optional_u64(&operation["RangeStart"], "RangeStart")?,
        if_none_match: validate_optional_string(
            &operation["IfNoneMatch"],
            "IfNoneMatch",
            MAX_HEADER_FIELD_BYTES,
        )?,
        offline: flag(&operation["Offline"], "Offline")?,
        expect_existing: flag(&operation["ExpectExistingFile"], "ExpectExistingFile")?,
        expected_bytes: validate_optional_u64(&operation["ExpectedBytes"], "ExpectedBytes")?,
        expected_sha256: validate_optional_sha256(&operation["ExpectedSha256"])?,
        timeout_secs: operation["TimeoutSeconds"]
            .as_u64()
            .unwrap_or(DEFAULT_TIMEOUT_SECS),
    })
}

fn finalize_http_download(req: &HttpOpRequest<'_>, response: HttpResponse) -> Result<Value> {
    if response.status_code == 304 {
        if !req.dest_path.is_file() {
            return Err(invalid(
                "HTTP 304 Not Modified without existing cached file",
            ));
        }
        return cached_hit_response(
            &req.dest_path,
            Some(304),
            Some(&response),
            req.expected_bytes,
            req.expected_sha256.as_deref(),
        );
    }
    if req.method == HttpMethod::Head {
        if response.status_code >= 400 {
            return Err(invalid(format!(
                "HTTP HEAD failed with status {}",
                response.status_code
            )));
        }
        return Ok(json!({
            "HttpCode": response.status_code,
            "Cached": false,
            "Path": Value::Null,
            "Bytes": 0,
            "Sha256": Value::Null,
            "ETag": response.etag,
            "ContentDisposition": response.content_disposition,
            "DispositionFilename": response.disposition_filename,
            "ContentRange": response.content_range,
            "ContentLength": response.content_length,
        }));
    }
    if !matches!(response.status_code, 200 | 206) {
        return Err(invalid(format!(
            "HTTP GET failed with status {}",
            response.status_code
        )));
    }
    let (final_bytes, meta) = if response.status_code == 206 {
        assemble_partial_206(&req.dest_path, req.range_start, response)?
    } else {
        let meta = HttpResponseMeta {
            status_code: response.status_code,
            etag: response.etag,
            content_disposition: response.content_disposition,
            disposition_filename: response.disposition_filename,
            content_range: response.content_range,
            content_length: response.content_length,
        };
        (response.body, meta)
    };
    let digest = verify_cached_bytes(
        &final_bytes,
        req.expected_bytes,
        req.expected_sha256.as_deref(),
    )?;
    write_atomic(&req.dest_path, &final_bytes)?;
    Ok(json!({
        "HttpCode": meta.status_code,
        "Cached": false,
        "Path": req.dest_path,
        "Bytes": final_bytes.len(),
        "Sha256": digest,
        "ETag": meta.etag,
        "ContentDisposition": meta.content_disposition,
        "DispositionFilename": meta.disposition_filename,
        "ContentRange": meta.content_range,
        "ContentLength": meta.content_length,
    }))
}

fn execute_http_operation(operation: &Value) -> Result<Value> {
    let mut req = parse_http_op_request(operation)?;
    if req.expect_existing {
        if !req.dest_path.is_file() {
            return Err(invalid("expected existing cached file is missing"));
        }
        if req.if_none_match.is_none() {
            return cached_hit_response(
                &req.dest_path,
                None,
                None,
                req.expected_bytes,
                req.expected_sha256.as_deref(),
            );
        }
    }
    if req.offline {
        if !req.dest_path.is_file() {
            return Err(invalid("offline mode and cached file is missing"));
        }
        return cached_hit_response(
            &req.dest_path,
            None,
            None,
            req.expected_bytes,
            req.expected_sha256.as_deref(),
        );
    }
    if let Some(start) = req.range_start {
        req.headers
            .push(("Range".to_owned(), format!("bytes={start}-")));
    }
    if let Some(etag) = req.if_none_match.as_deref() {
        req.headers
            .push(("If-None-Match".to_owned(), etag.to_owned()));
    }
    let header_refs: Vec<HeaderEntry<'_>> = req
        .headers
        .iter()
        .map(|(name, value)| HeaderEntry { name, value })
        .collect();
    let response = perform_http_request(
        req.method,
        req.url,
        &header_refs,
        Duration::from_secs(req.timeout_secs),
        MAX_HTTP_BODY_BYTES,
    )?;
    finalize_http_download(&req, response)
}

fn validate_huggingface_operation(operation: &Value) -> Result<()> {
    exact_fields(
        operation,
        &[
            "Kind", "Endpoint", "CacheDir", "HfRepo", "HfFile", "HfToken", "Offline", "Download",
            "Options",
        ],
    )?;
    let _ = parse_http_url(text(&operation["Endpoint"], MAX_URL_BYTES)?)?;
    let cache_dir = Path::new(text(&operation["CacheDir"], MAX_PATH_BYTES)?);
    if !cache_dir.is_absolute() {
        return Err(invalid(
            "model-resolve-huggingface CacheDir must be absolute",
        ));
    }
    let _ = parse_hf_repo(text(&operation["HfRepo"], MAX_REPO_BYTES)?)?;
    let _ = text(&operation["HfFile"], MAX_PATH_BYTES)?;
    let _ = validate_optional_string(&operation["HfToken"], "HfToken", MAX_TOKEN_BYTES)?;
    let _ = flag(&operation["Offline"], "Offline")?;
    let _ = flag(&operation["Download"], "Download")?;
    let _ = parse_plan_options(&operation["Options"])?;
    Ok(())
}

fn execute_huggingface_operation(operation: &Value) -> Result<Value> {
    let endpoint = text(&operation["Endpoint"], MAX_URL_BYTES)?.to_owned();
    let cache_dir = PathBuf::from(text(&operation["CacheDir"], MAX_PATH_BYTES)?);
    let hf_repo = text(&operation["HfRepo"], MAX_REPO_BYTES)?.to_owned();
    let hf_file = text(&operation["HfFile"], MAX_PATH_BYTES)?.to_owned();
    let hf_token = validate_optional_string(&operation["HfToken"], "HfToken", MAX_TOKEN_BYTES)?;
    let offline = flag(&operation["Offline"], "Offline")?;
    let download = flag(&operation["Download"], "Download")?;
    let options = parse_plan_options(&operation["Options"])?;
    let resolver = Resolver::with_options(cache_dir, endpoint, offline, hf_token)?;
    let model = ModelSpec {
        hf_repo,
        hf_file,
        ..ModelSpec::default()
    };
    let plan = resolver.plan(&model, options)?;
    if download {
        let mut to_fetch: Vec<&ResolvedFile> = plan.model_files.iter().collect();
        for sidecar in [
            &plan.mmproj,
            &plan.mtp,
            &plan.dflash,
            &plan.eagle3,
            &plan.dspark,
            &plan.preset,
        ] {
            if !sidecar.final_path.is_empty() {
                to_fetch.push(sidecar);
            }
        }
        resolver.download_plan_files(&plan, &to_fetch)?;
    }
    Ok(plan_to_json(&plan))
}

fn validate_cli_operation(operation: &Value) -> Result<()> {
    exact_fields(
        operation,
        &[
            "Kind",
            "Endpoint",
            "CacheDir",
            "Arguments",
            "Example",
            "OfflineAfterInit",
            "Apply",
        ],
    )?;
    let _ = parse_http_url(text(&operation["Endpoint"], MAX_URL_BYTES)?)?;
    let cache_dir = Path::new(text(&operation["CacheDir"], MAX_PATH_BYTES)?);
    if !cache_dir.is_absolute() {
        return Err(invalid("model-resolve-cli CacheDir must be absolute"));
    }
    let _ = parse_argv(&operation["Arguments"])?;
    let _ = parse_example(&operation["Example"])?;
    let _ = flag(&operation["OfflineAfterInit"], "OfflineAfterInit")?;
    let _ = flag(&operation["Apply"], "Apply")?;
    Ok(())
}

fn execute_cli_operation(operation: &Value) -> Result<Value> {
    let endpoint = text(&operation["Endpoint"], MAX_URL_BYTES)?.to_owned();
    let cache_dir = PathBuf::from(text(&operation["CacheDir"], MAX_PATH_BYTES)?);
    let arguments = parse_argv(&operation["Arguments"])?;
    let example = parse_example(&operation["Example"])?;
    let offline_after_init = flag(&operation["OfflineAfterInit"], "OfflineAfterInit")?;
    let apply = flag(&operation["Apply"], "Apply")?;
    let resolver = Resolver::new(cache_dir, endpoint)?;
    execute_handler(&resolver, &arguments, example, offline_after_init, apply)
}

fn validate_operation(operation: &Value) -> Result<()> {
    let object = operation
        .as_object()
        .ok_or_else(|| invalid("model resolution operation must be an object"))?;
    let kind = object
        .get("Kind")
        .ok_or_else(|| invalid("model resolution operation Kind required"))?;
    match text(kind, 64)? {
        "model-resolve-http" => validate_http_operation(operation),
        "model-resolve-huggingface" => validate_huggingface_operation(operation),
        "model-resolve-cli" => validate_cli_operation(operation),
        _ => Err(invalid("unknown model resolution operation Kind")),
    }
}

fn execute_operation(operation: &Value) -> Result<Value> {
    match text(&operation["Kind"], 64)? {
        "model-resolve-http" => execute_http_operation(operation),
        "model-resolve-huggingface" => execute_huggingface_operation(operation),
        "model-resolve-cli" => execute_cli_operation(operation),
        _ => Err(invalid("unknown model resolution operation Kind")),
    }
}

/// Returns `true` when `request` is a batch whose operations are all handled by this provider.
pub(super) fn handles(request: &Value) -> bool {
    request["Operations"].as_array().is_some_and(|operations| {
        !operations.is_empty()
            && operations.iter().all(|operation| {
                matches!(
                    operation["Kind"].as_str(),
                    Some("model-resolve-http" | "model-resolve-huggingface" | "model-resolve-cli")
                )
            })
    })
}

/// Returns `true` when `id` names the supported upstream source case.
pub(super) fn supports(id: &str) -> bool {
    id == ID
}

/// Validates and executes a `model-resolve-*` reference batch request.
pub(super) fn execute(request: &Value) -> Result<Value> {
    exact_fields(request, &["SchemaVersion", "Operations"])?;
    if request["SchemaVersion"].as_u64() != Some(1) {
        return Err(invalid("unsupported model resolution SchemaVersion"));
    }
    let operations = request["Operations"]
        .as_array()
        .ok_or_else(|| invalid("model resolution Operations array required"))?;
    if operations.is_empty() || operations.len() > MAX_OPERATIONS {
        return Err(invalid(
            "model resolution operations must number 1 through 128",
        ));
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

/// Discloses the clean-room independent provider metadata for `tests/test-model-resolution.cpp`.
pub(super) fn identity() -> Value {
    json!({
        "Kind": "independent-rust-model-resolution",
        "SupportedSourceIds": [ID],
        "UpstreamCommit": super::UPSTREAM,
        "AssertionSource": {"Path": ID, "Sha256": SOURCE_SHA},
        "Architecture": "clean-room-http11-cache-huggingface-and-cli-resolution-v1",
        "NativeReferenceReplay": false,
        "ProductionRuntimeCalled": false,
        "CompleteSourceGate": false,
        "NativeRetirementQualified": false
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;

    use super::*;
    use crate::engine::{Engine, Target};

    struct TempDirGuard {
        path: PathBuf,
    }

    impl TempDirGuard {
        fn new(prefix: &str) -> Result<Self> {
            static SEQ: AtomicU64 = AtomicU64::new(1);
            let root = std::env::temp_dir();
            fs::create_dir_all(&root)?;
            for _ in 0..64 {
                let id = SEQ.fetch_add(1, Ordering::Relaxed);
                let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
                let candidate = root.join(format!("{prefix}-{}-{nanos}-{id}", std::process::id()));
                match fs::create_dir(&candidate) {
                    Ok(()) => return Ok(Self { path: candidate }),
                    Err(error) if error.kind() == ErrorKind::AlreadyExists => {}
                    Err(error) => return Err(error.into()),
                }
            }
            Err(invalid("unable to create unique test temp directory"))
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    struct CapturedRequest {
        request_line: String,
        headers: BTreeMap<String, String>,
    }

    fn spawn_scripted_http<F>(
        handler: F,
    ) -> Result<(
        String,
        std::thread::JoinHandle<Result<Vec<CapturedRequest>>>,
    )>
    where
        F: Fn(&CapturedRequest) -> Vec<u8> + Send + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let addr = listener.local_addr()?;
        let handle = std::thread::spawn(move || {
            let mut captured = Vec::new();
            let (mut stream, _) = listener.accept()?;
            let mut reader = BufReader::new(stream.try_clone()?);
            let mut request_line = String::new();
            let _ = reader.read_line(&mut request_line)?;
            let mut headers = BTreeMap::new();
            loop {
                let mut line = String::new();
                let read = reader.read_line(&mut line)?;
                if read == 0 || line == "\r\n" || line == "\n" {
                    break;
                }
                if let Some((k, v)) = line.trim_end_matches(['\r', '\n']).split_once(':') {
                    headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_owned());
                }
            }
            let req = CapturedRequest {
                request_line: request_line.trim_end_matches(['\r', '\n']).to_owned(),
                headers,
            };
            let reply = handler(&req);
            captured.push(req);
            stream.write_all(&reply)?;
            stream.flush()?;
            Ok(captured)
        });
        Ok((format!("http://{addr}"), handle))
    }

    #[test]
    fn independent_model_resolution_passes_complete_suite_on_rust_reference_and_cpp() -> Result<()>
    {
        for target in [Target::RustReference, Target::Cpp] {
            let engine = Engine::new(target, None, None, None, None)?;
            let report = crate::model_resolution::run(&engine)?;
            assert_eq!(report["Id"], ID);
            assert_eq!(report["Status"], "pass", "{report}");
        }
        assert!(crate::reference_provider::supports(ID));
        assert!(supports(ID));
        let info = identity();
        assert_eq!(info["SupportedSourceIds"], json!([ID]));
        assert_eq!(info["NativeReferenceReplay"], false);
        assert_eq!(info["ProductionRuntimeCalled"], false);
        assert_eq!(info["CompleteSourceGate"], false);
        assert_eq!(info["NativeRetirementQualified"], false);
        let provider_info = crate::reference_provider::identity();
        let supported = provider_info["SupportedSourceIds"]
            .as_array()
            .ok_or_else(|| invalid("missing SupportedSourceIds"))?;
        assert!(supported.iter().any(|entry| entry == ID));
        Ok(())
    }

    fn http_op(
        method: &str,
        url: &str,
        cache_dir: &Path,
        cache_path: &str,
        offline: bool,
        expect_existing: bool,
    ) -> Value {
        json!({
            "Kind": "model-resolve-http",
            "Method": method,
            "Url": url,
            "CacheDir": cache_dir,
            "CachePath": cache_path,
            "Headers": [],
            "RangeStart": null,
            "IfNoneMatch": null,
            "Offline": offline,
            "ExpectExistingFile": expect_existing,
            "ExpectedBytes": null,
            "ExpectedSha256": null,
            "TimeoutSeconds": 5
        })
    }

    #[test]
    fn http_client_handles_head_and_redirect_chain_with_sha256() -> Result<()> {
        let cache = TempDirGuard::new("prism-mr-http-1")?;
        let payload = b"0123456789abcdef";
        let full_sha = sha256_hex(payload);

        let (head_base, head_thread) = spawn_scripted_http(|req| {
            if req.request_line.starts_with("HEAD ") {
                b"HTTP/1.1 200 OK\r\nContent-Length: 16\r\nETag: \"v1\"\r\nContent-Disposition: attachment; filename=\"weights.gguf\"\r\n\r\n".to_vec()
            } else {
                b"HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\n\r\n".to_vec()
            }
        })?;
        let head_res = execute(&json!({
            "SchemaVersion": 1,
            "Operations": [http_op("HEAD", &format!("{head_base}/file"), cache.path(), "head.bin", false, false)]
        }))?;
        let _ = head_thread.join();
        assert_eq!(head_res["Results"][0]["HttpCode"], 200);
        assert_eq!(head_res["Results"][0]["ContentLength"], 16);
        assert_eq!(head_res["Results"][0]["ETag"], "\"v1\"");
        assert_eq!(
            head_res["Results"][0]["DispositionFilename"],
            "weights.gguf"
        );
        assert_eq!(head_res["Results"][0]["Bytes"], 0);

        let (target_base, target_thread) = spawn_scripted_http(|req| {
            assert!(!req.headers.contains_key("authorization"));
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nETag: \"v1\"\r\n\r\n6\r\n012345\r\na\r\n6789abcdef\r\n0\r\n\r\n".to_vec()
        })?;
        let target_dest = format!("{target_base}/final");
        let redir_listener = TcpListener::bind("127.0.0.1:0")?;
        let redir_addr = redir_listener.local_addr()?;
        let redir2_thread = std::thread::spawn(move || -> Result<()> {
            for idx in 0..2 {
                let (mut stream, _) = redir_listener.accept()?;
                let mut reader = BufReader::new(stream.try_clone()?);
                let mut line = String::new();
                while reader.read_line(&mut line)? > 0 {
                    if line == "\r\n" || line == "\n" {
                        break;
                    }
                    line.clear();
                }
                let resp = if idx == 0 {
                    "HTTP/1.1 308 Permanent Redirect\r\nLocation: /hop\r\nContent-Length: 0\r\n\r\n"
                        .to_owned()
                } else {
                    format!(
                        "HTTP/1.1 303 See Other\r\nLocation: {target_dest}\r\nContent-Length: 0\r\n\r\n"
                    )
                };
                stream.write_all(resp.as_bytes())?;
                stream.flush()?;
            }
            Ok(())
        });

        let mut get_op = http_op(
            "GET",
            &format!("http://{redir_addr}/start"),
            cache.path(),
            "sub/model.bin",
            false,
            false,
        );
        get_op["Headers"] = json!([{"Name": "Authorization", "Value": "Bearer secret"}]);
        get_op["ExpectedBytes"] = json!(16);
        get_op["ExpectedSha256"] = json!(full_sha);
        let get_res = execute(&json!({"SchemaVersion": 1, "Operations": [get_op]}))?;
        let _ = redir2_thread.join();
        let _ = target_thread.join();
        assert_eq!(get_res["Results"][0]["HttpCode"], 200);
        assert_eq!(get_res["Results"][0]["Bytes"], 16);
        assert_eq!(get_res["Results"][0]["Sha256"], full_sha);
        assert_eq!(fs::read(cache.path().join("sub/model.bin"))?, payload);
        Ok(())
    }

    #[test]
    fn http_client_handles_etag_304_range_206_offline_and_sha_mismatch() -> Result<()> {
        let cache = TempDirGuard::new("prism-mr-http-2")?;
        let payload = b"0123456789abcdef";
        let full_sha = sha256_hex(payload);
        fs::create_dir_all(cache.path().join("sub"))?;
        fs::write(cache.path().join("sub/model.bin"), payload)?;

        let (not_mod_base, not_mod_thread) = spawn_scripted_http(|req| {
            assert_eq!(
                req.headers.get("if-none-match").map(String::as_str),
                Some("\"v1\"")
            );
            b"HTTP/1.1 304 Not Modified\r\nETag: \"v1\"\r\n\r\n".to_vec()
        })?;
        let mut reval_op = http_op(
            "GET",
            &format!("{not_mod_base}/file"),
            cache.path(),
            "sub/model.bin",
            false,
            true,
        );
        reval_op["IfNoneMatch"] = json!("\"v1\"");
        reval_op["ExpectedBytes"] = json!(16);
        reval_op["ExpectedSha256"] = json!(full_sha);
        let reval_res = execute(&json!({"SchemaVersion": 1, "Operations": [reval_op]}))?;
        let _ = not_mod_thread.join();
        assert_eq!(reval_res["Results"][0]["HttpCode"], 304);
        assert_eq!(reval_res["Results"][0]["Cached"], true);

        fs::write(cache.path().join("sub/resume.bin"), &payload[..6])?;
        let (range_base, range_thread) = spawn_scripted_http(|req| {
            assert_eq!(
                req.headers.get("range").map(String::as_str),
                Some("bytes=6-")
            );
            b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 6-15/16\r\nContent-Length: 10\r\n\r\n6789abcdef".to_vec()
        })?;
        let mut range_op = http_op(
            "GET",
            &format!("{range_base}/file"),
            cache.path(),
            "sub/resume.bin",
            false,
            false,
        );
        range_op["RangeStart"] = json!(6);
        range_op["ExpectedBytes"] = json!(16);
        range_op["ExpectedSha256"] = json!(full_sha);
        let range_res = execute(&json!({"SchemaVersion": 1, "Operations": [range_op]}))?;
        let _ = range_thread.join();
        assert_eq!(range_res["Results"][0]["HttpCode"], 206);
        assert_eq!(range_res["Results"][0]["Bytes"], 16);
        assert_eq!(fs::read(cache.path().join("sub/resume.bin"))?, payload);

        let mut hit_op = http_op(
            "GET",
            "http://127.0.0.1:1/unused",
            cache.path(),
            "sub/resume.bin",
            true,
            true,
        );
        hit_op["ExpectedBytes"] = json!(16);
        hit_op["ExpectedSha256"] = json!(full_sha);
        let offline_hit = execute(&json!({"SchemaVersion": 1, "Operations": [hit_op]}))?;
        assert_eq!(offline_hit["Results"][0]["Cached"], true);

        let offline_miss = execute(&json!({
            "SchemaVersion": 1,
            "Operations": [http_op("GET", "http://127.0.0.1:1/unused", cache.path(), "sub/missing.bin", true, false)]
        }));
        assert!(offline_miss.is_err());

        let (bad_sha_base, bad_sha_thread) =
            spawn_scripted_http(|_| b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\nbad!".to_vec())?;
        let mut bad_sha_op = http_op(
            "GET",
            &format!("{bad_sha_base}/file"),
            cache.path(),
            "sub/corrupt.bin",
            false,
            false,
        );
        bad_sha_op["ExpectedBytes"] = json!(4);
        bad_sha_op["ExpectedSha256"] = json!("0".repeat(64));
        let bad_sha = execute(&json!({"SchemaVersion": 1, "Operations": [bad_sha_op]}));
        let _ = bad_sha_thread.join();
        assert!(bad_sha.is_err());
        assert!(!cache.path().join("sub/corrupt.bin").exists());
        Ok(())
    }

    #[test]
    fn huggingface_and_cli_operations_work_end_to_end() -> Result<()> {
        let cache = TempDirGuard::new("prism-mr-hf")?;
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let addr = listener.local_addr()?;
        let endpoint = format!("http://{addr}/");
        let server_thread = std::thread::spawn(move || -> Result<()> {
            for _ in 0..4 {
                let (mut stream, _) = listener.accept()?;
                let mut reader = BufReader::new(stream.try_clone()?);
                let mut req_line = String::new();
                let _ = reader.read_line(&mut req_line)?;
                loop {
                    let mut line = String::new();
                    let read = reader.read_line(&mut line)?;
                    if read == 0 || line == "\r\n" || line == "\n" {
                        break;
                    }
                }
                let body = if req_line.contains("/refs ") {
                    r#"{"branches":[{"name":"main","targetCommit":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]}"#
                } else {
                    r#"[{"type":"file","path":"model-Q4_K_M.gguf","size":1,"oid":"0000000000000000000000000000000000000001"},{"type":"file","path":"mmproj-model-f16.gguf","size":1,"oid":"0000000000000000000000000000000000000002"}]"#
                };
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                stream.write_all(resp.as_bytes())?;
                stream.flush()?;
            }
            Ok(())
        });

        let hf_res = execute(&json!({
            "SchemaVersion": 1,
            "Operations": [{
                "Kind": "model-resolve-huggingface",
                "Endpoint": endpoint,
                "CacheDir": cache.path(),
                "HfRepo": "org/model",
                "HfFile": "",
                "HfToken": "hf_test_token",
                "Offline": false,
                "Download": false,
                "Options": {"Mmproj": true, "Mtp": false, "Dflash": false, "Eagle3": false, "Dspark": false}
            }]
        }))?;
        assert_eq!(hf_res["Results"][0]["Primary"]["Path"], "model-Q4_K_M.gguf");
        assert_eq!(
            hf_res["Results"][0]["Mmproj"]["Path"],
            "mmproj-model-f16.gguf"
        );

        let cli_res = execute(&json!({
            "SchemaVersion": 1,
            "Operations": [
                {
                    "Kind": "model-resolve-cli",
                    "Endpoint": endpoint,
                    "CacheDir": cache.path(),
                    "Arguments": ["server", "-hf", "org/model"],
                    "Example": 12,
                    "OfflineAfterInit": true,
                    "Apply": true
                },
                {
                    "Kind": "model-resolve-cli",
                    "Endpoint": endpoint,
                    "CacheDir": cache.path(),
                    "Arguments": ["server", "--unknown-flag"],
                    "Example": 12,
                    "OfflineAfterInit": false,
                    "Apply": false
                }
            ]
        }))?;
        assert_eq!(cli_res["Results"][0]["Parsed"], true);
        assert_eq!(cli_res["Results"][1]["Parsed"], false);
        server_thread
            .join()
            .map_err(|_| invalid("HF test server thread panicked"))??;
        Ok(())
    }

    #[test]
    fn rejects_malformed_inputs_and_unknown_ops() -> Result<()> {
        let cache = TempDirGuard::new("prism-mr-reject")?;
        assert!(!handles(&json!({"SchemaVersion": 1, "Operations": []})));
        assert!(!handles(
            &json!({"SchemaVersion": 1, "Operations": [{"Kind": "unknown-op"}]})
        ));
        for bad in [
            json!({"SchemaVersion": 2, "Operations": []}),
            json!({"SchemaVersion": 1, "Operations": []}),
            json!({"SchemaVersion": 1, "Operations": [{"Kind": "unknown-op"}]}),
            json!({"SchemaVersion": 1, "Operations": [http_op("POST", "http://127.0.0.1:80/a", cache.path(), "a", false, false)]}),
            json!({"SchemaVersion": 1, "Operations": [http_op("GET", "https://127.0.0.1:443/a", cache.path(), "a", false, false)]}),
            json!({"SchemaVersion": 1, "Operations": [http_op("GET", "http://127.0.0.1:80/a", cache.path(), "../escape", false, false)]}),
            json!({"SchemaVersion": 1, "Operations": [{"Kind": "model-resolve-cli", "Endpoint": "http://127.0.0.1:80/", "CacheDir": cache.path(), "Arguments": [], "Example": 12, "OfflineAfterInit": false, "Apply": false}]}),
            json!({"SchemaVersion": 1, "Operations": [{"Kind": "model-resolve-cli", "Endpoint": "http://127.0.0.1:80/", "CacheDir": cache.path(), "Arguments": ["server"], "Example": 27, "OfflineAfterInit": false, "Apply": false}]}),
        ] {
            assert!(execute(&bad).is_err(), "{bad}");
        }
        let resolver = Resolver::new(
            cache.path().to_path_buf(),
            "http://127.0.0.1:80/".to_owned(),
        )?;
        assert!(resolver.invoke(&json!({"Op": "unknown-op"})).is_err());
        assert!(
            resolver
                .invoke(&json!({"Op": "resolution-pause-log", "Extra": 1}))
                .is_err()
        );
        Ok(())
    }
}
