//! Internal native Swift backend entry point for the existing Windows Job owner.

use std::{ffi::OsString, io, path::PathBuf};

use bonsai_prism_runtime::{
    graph::KvCachePolicy,
    kernels::{MAX_ROW_WORKERS, MatvecBackend, RowParallelism},
    model::swift::{MMPROJ_BYTES, MMPROJ_SHA256, Representation},
    runtime::PromptReuse,
    swift_session::{
        BranchCheckpoints, CpuPrefill, CpuSpeculation, DEFAULT_MAX_IMAGE_TOKENS, MAX_IMAGE_TOKENS,
        MIN_IMAGE_TOKENS, MMPROJ_MEMORY_BYTES, MultiRankAdapterSpec, NON_CONTEXT_MEMORY_BYTES,
        SwiftSession, calibrated_non_context_memory_bytes,
        validate_calibrated_cow_execution_capacity, validate_context_capacity_with_kv,
        validate_execution_capacity_with_reuse,
    },
};
use bonsai_windows_credentials::{validate_external_path, validate_read_path};

use crate::{
    native_backend::{Placement, WorkerOptions},
    service::bound_loopback_socket,
    startup::{self, Stage},
};

#[derive(Debug, Eq, PartialEq)]
enum ChildTransport {
    Loopback { port: u16, key: PathBuf },
    BrokerPipe,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum AdmissionPolicy {
    #[default]
    Conservative,
    CalibratedCow,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DaisyRoseConfig {
    path: PathBuf,
    bytes: u64,
    sha256: String,
}

struct Config {
    placement: Placement,
    model: PathBuf,
    transport: ChildTransport,
    verified_sections: Option<usize>,
    admission_policy: AdmissionPolicy,
    seed: u32,
    row_workers: usize,
    prefill_row_workers: usize,
    representation: Representation,
    daisyrose_lora: Option<DaisyRoseConfig>,
    mmproj: Option<PathBuf>,
    image_max_tokens: usize,
    prompt_reuse: PromptReuse,
    branch_checkpoints: BranchCheckpoints,
    kv_cache: KvCachePolicy,
    cpu_prefill: Option<CpuPrefill>,
    requested_prefill_tokens: Option<usize>,
    cpu_speculation: Option<CpuSpeculation>,
    capacity: Option<(usize, u64)>,
    hexagon_arena_cache: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CpuFamily {
    X1,
    X2,
}

impl CpuFamily {
    const fn default_row_workers(self) -> usize {
        match self {
            Self::X1 => 8,
            Self::X2 => 6,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SpecType {
    None,
    Suffix,
}

fn invalid() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        "invalid native child configuration",
    )
}

#[expect(
    clippy::too_many_lines,
    reason = "Keep trusted profile parsing and pre-I/O memory admission visibly ordered."
)]
fn parse(arguments: impl IntoIterator<Item = OsString>) -> io::Result<Config> {
    let mut arguments = arguments.into_iter();
    let mut child = false;
    let mut broker_pipe = false;
    let mut verified_sections = None;
    let mut admission_policy = None;
    let (mut model, mut key, mut port, mut seed) = (None, None, None, None);
    let mut row_workers = None;
    let mut prefill_row_workers = None;
    let mut cpu_family = None;
    let mut nariq: Option<Representation> = None;
    let mut bonsai25: Option<Representation> = None;
    let (mut daisyrose_path, mut daisyrose_bytes, mut daisyrose_sha256) = (None, None, None);
    let mut mmproj = None;
    let mut image_max_tokens = None;
    let mut prompt_reuse = None;
    let (mut reuse_checkpoints, mut reuse_checkpoint_bytes) = (None, None);
    let mut kv_cache = None;
    let mut placement = None;
    let mut hexagon_arena_cache = None;
    let (mut prefill_tokens, mut prefill_scratch, mut prefill_gdn) = (None, None, None);
    let (mut spec_type, mut draft_max) = (None, None);
    let (mut speculative_tokens, mut verification_scratch, mut speculative_checkpoint) =
        (None, None, None);
    let (mut context_tokens, mut memory_limit_bytes) = (None, None);
    while let Some(argument) = arguments.next() {
        if argument == "--native-engine-child" && !child {
            child = true;
            continue;
        }
        if argument == "--broker-pipe" && !broker_pipe {
            broker_pipe = true;
            continue;
        }
        let value = arguments.next().ok_or_else(invalid)?;
        match argument.to_str() {
            Some("--nariq") if nariq.is_none() && bonsai25.is_none() => {
                nariq = Some(match value.to_str() {
                    Some("2.5bpw") => Representation::NariQ2_5Bpw,
                    Some("3.0bpw") => Representation::NariQ3_0Bpw,
                    Some("2.0bpw") => Representation::NariQ2_0Bpw,
                    _ => return Err(invalid()),
                });
            }
            Some("--bonsai-2-5") if bonsai25.is_none() && nariq.is_none() => {
                bonsai25 = Some(match value.to_str() {
                    Some("pq2") => Representation::Bonsai25Pq2,
                    Some("ptq1") => Representation::Bonsai25Ptq1,
                    _ => return Err(invalid()),
                });
            }
            Some("--model") if model.is_none() => model = Some(PathBuf::from(value)),
            Some("--key-file") if key.is_none() => key = Some(PathBuf::from(value)),
            Some("--daisyrose-lora") if daisyrose_path.is_none() => {
                daisyrose_path = Some(PathBuf::from(value));
            }
            Some("--daisyrose-lora-bytes") if daisyrose_bytes.is_none() => {
                daisyrose_bytes = Some(value);
            }
            Some("--daisyrose-lora-sha256") if daisyrose_sha256.is_none() => {
                daisyrose_sha256 = Some(value);
            }
            Some("--mmproj") if mmproj.is_none() => mmproj = Some(PathBuf::from(value)),
            Some("--image-max-tokens") if image_max_tokens.is_none() => {
                image_max_tokens = Some(value);
            }
            Some("--context-tokens") if context_tokens.is_none() => context_tokens = Some(value),
            Some("--memory-limit-bytes") if memory_limit_bytes.is_none() => {
                memory_limit_bytes = Some(value);
            }
            Some("--cpu-prefill-tokens") if prefill_tokens.is_none() => {
                prefill_tokens = Some(value);
            }
            Some("--cpu-prefill-scratch-bytes") if prefill_scratch.is_none() => {
                prefill_scratch = Some(value);
            }
            Some("--cpu-prefill-gdn") if prefill_gdn.is_none() => {
                prefill_gdn = Some(match value.to_str() {
                    Some("sequential") => false,
                    Some("wy-f32") => true,
                    _ => return Err(invalid()),
                });
            }
            Some("--cpu-speculative-tokens") if speculative_tokens.is_none() => {
                speculative_tokens = Some(value);
            }
            Some("--cpu-verification-scratch-bytes") if verification_scratch.is_none() => {
                verification_scratch = Some(value);
            }
            Some("--cpu-speculative-checkpoint-bytes") if speculative_checkpoint.is_none() => {
                speculative_checkpoint = Some(value);
            }
            Some("--port") if port.is_none() => {
                port = Some(
                    value
                        .to_str()
                        .ok_or_else(invalid)?
                        .parse::<u16>()
                        .map_err(|_| invalid())?,
                );
            }
            Some("--seed") if seed.is_none() => {
                seed = Some(
                    value
                        .to_str()
                        .ok_or_else(invalid)?
                        .parse::<u32>()
                        .map_err(|_| invalid())?,
                );
            }
            Some("--row-workers") if row_workers.is_none() => {
                row_workers = Some(
                    value
                        .to_str()
                        .ok_or_else(invalid)?
                        .parse::<usize>()
                        .map_err(|_| invalid())?,
                );
            }
            Some("--prefill-row-workers") if prefill_row_workers.is_none() => {
                prefill_row_workers = Some(
                    value
                        .to_str()
                        .ok_or_else(invalid)?
                        .parse::<usize>()
                        .map_err(|_| invalid())?,
                );
            }
            Some("--prompt-reuse") if prompt_reuse.is_none() => {
                prompt_reuse = Some(match value.to_str() {
                    Some("disabled") => PromptReuse::Disabled,
                    Some("latest") => PromptReuse::Latest,
                    Some("prompt-end-and-latest") => PromptReuse::PromptEndAndLatest,
                    _ => return Err(invalid()),
                });
            }
            Some("--prompt-reuse-checkpoints") if reuse_checkpoints.is_none() => {
                reuse_checkpoints = Some(value);
            }
            Some("--prompt-reuse-checkpoint-bytes") if reuse_checkpoint_bytes.is_none() => {
                reuse_checkpoint_bytes = Some(value);
            }
            Some("--kv-cache") if kv_cache.is_none() => {
                kv_cache = Some(match value.to_str() {
                    Some("f16") => KvCachePolicy::F16,
                    Some("rsq4") => KvCachePolicy::Rotated4,
                    Some("rsq4vf") => KvCachePolicy::Rotated4FittedValues,
                    Some("rsq4kvf") => KvCachePolicy::Rotated4FittedKv,
                    Some("rsq4k3v") => KvCachePolicy::Rotated4Keys3Values,
                    _ => return Err(invalid()),
                });
            }
            Some("--cpu-family") if cpu_family.is_none() => {
                cpu_family = Some(match value.to_str() {
                    Some("x1") => CpuFamily::X1,
                    Some("x2") => CpuFamily::X2,
                    _ => return Err(invalid()),
                });
            }
            Some("--spec-type") if spec_type.is_none() => {
                spec_type = Some(match value.to_str() {
                    Some("none") => SpecType::None,
                    Some("suffix") => SpecType::Suffix,
                    _ => return Err(invalid()),
                });
            }
            Some("--draft-max") if draft_max.is_none() => draft_max = Some(value),
            Some("--verified-sections") if verified_sections.is_none() => {
                let parsed = value
                    .to_str()
                    .ok_or_else(invalid)?
                    .parse::<usize>()
                    .map_err(|_| invalid())?;
                if !(1..=8).contains(&parsed) {
                    return Err(invalid());
                }
                verified_sections = Some(parsed);
            }
            Some("--admission-policy") if admission_policy.is_none() => {
                admission_policy = Some(match value.to_str() {
                    Some("conservative") => AdmissionPolicy::Conservative,
                    Some("calibrated-cow") => AdmissionPolicy::CalibratedCow,
                    _ => return Err(invalid()),
                });
            }
            Some("--placement") if placement.is_none() => {
                placement = Some(match value.to_str() {
                    Some("auto") => Placement::Auto,
                    Some("cpu") => Placement::Cpu,
                    Some("vulkan") => Placement::Vulkan,
                    Some("composite") => Placement::Composite,
                    _ => return Err(invalid()),
                });
            }
            Some("--hexagon-arena-cache") if hexagon_arena_cache.is_none() => {
                hexagon_arena_cache = Some(PathBuf::from(value));
            }
            _ => return Err(invalid()),
        }
    }
    let resolved_image_max_tokens = match image_max_tokens {
        Some(raw) => {
            if mmproj.is_none() {
                return Err(invalid());
            }
            let parsed = raw
                .to_str()
                .ok_or_else(invalid)?
                .parse::<usize>()
                .map_err(|_| invalid())?;
            if !(MIN_IMAGE_TOKENS..=MAX_IMAGE_TOKENS).contains(&parsed) {
                return Err(invalid());
            }
            parsed
        }
        None => DEFAULT_MAX_IMAGE_TOKENS,
    };
    let transport = if broker_pipe {
        if key.is_some() || port.is_some() {
            return Err(invalid());
        }
        ChildTransport::BrokerPipe
    } else {
        if verified_sections.is_some() {
            return Err(invalid());
        }
        let key = key.ok_or_else(invalid)?;
        let port = port.filter(|&n| n >= 1024).ok_or_else(invalid)?;
        validate_read_path(&key)?;
        ChildTransport::Loopback { port, key }
    };
    let resolved_placement = placement.unwrap_or_default();
    let resolved_kv_cache =
        resolve_kv_cache_for_placement(resolved_placement, kv_cache).ok_or_else(invalid)?;
    let resolved_policy = admission_policy.unwrap_or_default();
    let daisyrose_lora =
        resolve_daisyrose_config(daisyrose_path, daisyrose_bytes, daisyrose_sha256)?;
    let resolved_row_workers =
        row_workers.unwrap_or_else(|| cpu_family.map_or(1, CpuFamily::default_row_workers));
    let resolved_prefill_row_workers = prefill_row_workers.unwrap_or(resolved_row_workers);
    let (cpu_prefill, requested_prefill_tokens) =
        parse_prefill(prefill_tokens, prefill_scratch, prefill_gdn)?;
    let config = Config {
        placement: resolved_placement,
        model: model.ok_or_else(invalid)?,
        transport,
        verified_sections,
        admission_policy: resolved_policy,
        seed: seed.filter(|&n| n != u32::MAX).ok_or_else(invalid)?,
        row_workers: resolved_row_workers,
        prefill_row_workers: resolved_prefill_row_workers,
        representation: nariq.or(bonsai25).unwrap_or(Representation::Bonsai25Pq2),
        daisyrose_lora,
        mmproj,
        image_max_tokens: resolved_image_max_tokens,
        prompt_reuse: prompt_reuse.unwrap_or_default(),
        branch_checkpoints: parse_branch_checkpoints(reuse_checkpoints, reuse_checkpoint_bytes)?,
        kv_cache: resolved_kv_cache,
        cpu_prefill,
        requested_prefill_tokens,
        cpu_speculation: resolve_speculation(
            spec_type,
            draft_max,
            speculative_tokens,
            verification_scratch,
            speculative_checkpoint,
        )?,
        capacity: parse_capacity_with_policy(
            context_tokens,
            memory_limit_bytes,
            resolved_kv_cache,
            resolved_policy,
        )?,
        hexagon_arena_cache,
    };
    if let Some(expected_sections) = config.verified_sections {
        let configured_sections = 1_usize
            + usize::from(config.daisyrose_lora.is_some())
            + usize::from(config.mmproj.is_some());
        if expected_sections != configured_sections {
            return Err(invalid());
        }
    }
    if config.mmproj.is_some() && config.cpu_speculation.is_some() {
        return Err(invalid());
    }
    if config.representation.is_nariq()
        && (config.mmproj.is_some()
            || config.hexagon_arena_cache.is_some()
            || matches!(config.placement, Placement::Vulkan | Placement::Composite))
    {
        return Err(invalid());
    }
    if !placement_supports_config(&config) {
        return Err(invalid());
    }
    let requires_reuse_capacity = config.prompt_reuse == PromptReuse::PromptEndAndLatest
        || config.branch_checkpoints.checkpoints() > 1
        || config.branch_checkpoints.checkpoint_bytes().is_some();
    if config.prompt_reuse == PromptReuse::Disabled
        && (config.branch_checkpoints.checkpoints() > 1
            || config.branch_checkpoints.checkpoint_bytes().is_some())
    {
        return Err(invalid());
    }
    if (config.cpu_prefill.is_some() || config.cpu_speculation.is_some()) && kv_cache.is_none() {
        return Err(invalid());
    }
    match config.admission_policy {
        AdmissionPolicy::Conservative => {
            if config.cpu_prefill.is_some()
                || config.cpu_speculation.is_some()
                || requires_reuse_capacity
            {
                let (context, memory) = config.capacity.ok_or_else(invalid)?;
                validate_execution_capacity_with_reuse(
                    context,
                    memory,
                    config.cpu_prefill,
                    config.cpu_speculation,
                    config.kv_cache,
                    config.prompt_reuse,
                    config.branch_checkpoints,
                )
                .map_err(|_| invalid())?;
            }
        }
        AdmissionPolicy::CalibratedCow => {
            let (context, memory) = config.capacity.ok_or_else(invalid)?;
            validate_calibrated_cow_execution_capacity(
                context,
                memory,
                config.representation,
                config.cpu_prefill,
                config.cpu_speculation,
                config.kv_cache,
                config.prompt_reuse,
                config.branch_checkpoints,
            )
            .map_err(|_| invalid())?;
        }
    }
    if !(1..=MAX_ROW_WORKERS).contains(&config.row_workers)
        || !(1..=MAX_ROW_WORKERS).contains(&config.prefill_row_workers)
        || ((config.row_workers > 1 || config.prefill_row_workers > 1)
            && !cfg!(feature = "parallel-rows"))
    {
        return Err(invalid());
    }
    if !child {
        return Err(invalid());
    }
    for path in std::iter::once(&config.model)
        .chain(config.daisyrose_lora.as_ref().map(|item| &item.path))
        .chain(config.mmproj.as_ref())
    {
        validate_read_path(path)?;
    }
    if let Some(ref cache_dir) = config.hexagon_arena_cache {
        validate_hexagon_arena_cache_dir(cache_dir)?;
    }
    Ok(config)
}

fn validate_hexagon_arena_cache_dir(dir: &std::path::Path) -> io::Result<()> {
    validate_external_path(dir)?;
    match std::fs::symlink_metadata(dir) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => Err(invalid()),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

fn resolve_daisyrose_config(
    path: Option<PathBuf>,
    bytes: Option<OsString>,
    sha256: Option<OsString>,
) -> io::Result<Option<DaisyRoseConfig>> {
    match (path, bytes, sha256) {
        (None, None, None) => Ok(None),
        (Some(path), Some(raw_bytes), Some(raw_sha256)) => {
            let bytes = raw_bytes
                .to_str()
                .ok_or_else(invalid)?
                .parse::<u64>()
                .ok()
                .filter(|&n| n > 0)
                .ok_or_else(invalid)?;
            let sha256 = raw_sha256
                .to_str()
                .ok_or_else(invalid)?
                .to_ascii_lowercase();
            let _ = decode_sha256_hex(&sha256)?;
            Ok(Some(DaisyRoseConfig {
                path,
                bytes,
                sha256,
            }))
        }
        _ => Err(invalid()),
    }
}

fn resolve_speculation(
    spec_type: Option<SpecType>,
    draft_max: Option<OsString>,
    speculative_tokens: Option<OsString>,
    verification_scratch: Option<OsString>,
    speculative_checkpoint: Option<OsString>,
) -> io::Result<Option<CpuSpeculation>> {
    if draft_max.is_some() && speculative_tokens.is_some() {
        return Err(invalid());
    }
    if draft_max.is_some() && spec_type != Some(SpecType::Suffix) {
        return Err(invalid());
    }
    match spec_type {
        Some(SpecType::None) => {
            if speculative_tokens.is_some()
                || verification_scratch.is_some()
                || speculative_checkpoint.is_some()
            {
                return Err(invalid());
            }
            Ok(None)
        }
        Some(SpecType::Suffix) => {
            let tokens = draft_max
                .or(speculative_tokens)
                .unwrap_or_else(|| OsString::from("4"));
            parse_speculation(Some(tokens), verification_scratch, speculative_checkpoint)
        }
        None => parse_speculation(
            speculative_tokens,
            verification_scratch,
            speculative_checkpoint,
        ),
    }
}

fn parse_prefill(
    tokens: Option<OsString>,
    scratch: Option<OsString>,
    chunkwise_gdn: Option<bool>,
) -> io::Result<(Option<CpuPrefill>, Option<usize>)> {
    match (tokens, scratch) {
        (None, None) => {
            if chunkwise_gdn.is_some() {
                return Err(invalid());
            }
            Ok((None, None))
        }
        (Some(tokens), Some(scratch)) => {
            let tokens = tokens
                .to_str()
                .ok_or_else(invalid)?
                .parse::<usize>()
                .map_err(|_| invalid())?;
            let scratch = scratch
                .to_str()
                .ok_or_else(invalid)?
                .parse::<usize>()
                .map_err(|_| invalid())?;
            let prefill = CpuPrefill::new(tokens, scratch).map_err(|_| invalid())?;
            let prefill = if chunkwise_gdn == Some(true) {
                prefill.with_chunkwise_gdn()
            } else {
                prefill
            };
            Ok((Some(prefill), Some(tokens)))
        }
        _ => Err(invalid()),
    }
}

fn parse_branch_checkpoints(
    checkpoints: Option<OsString>,
    checkpoint_bytes: Option<OsString>,
) -> io::Result<BranchCheckpoints> {
    let parse_usize = |raw: OsString| -> io::Result<usize> {
        raw.to_str()
            .ok_or_else(invalid)?
            .parse()
            .map_err(|_| invalid())
    };
    match (checkpoints, checkpoint_bytes) {
        (None, None) => Ok(BranchCheckpoints::default()),
        (Some(slots), None) => BranchCheckpoints::new(parse_usize(slots)?).map_err(|_| invalid()),
        (None, Some(bytes)) => {
            BranchCheckpoints::with_checkpoint_bytes(1, parse_usize(bytes)?).map_err(|_| invalid())
        }
        (Some(slots), Some(bytes)) => {
            BranchCheckpoints::with_checkpoint_bytes(parse_usize(slots)?, parse_usize(bytes)?)
                .map_err(|_| invalid())
        }
    }
}

fn parse_speculation(
    tokens: Option<OsString>,
    scratch: Option<OsString>,
    checkpoint: Option<OsString>,
) -> io::Result<Option<CpuSpeculation>> {
    match (tokens, scratch, checkpoint) {
        (None, None, None) => Ok(None),
        (Some(tokens), Some(scratch), Some(checkpoint)) => {
            let parse_number = |value: OsString| -> io::Result<usize> {
                value
                    .to_str()
                    .ok_or_else(invalid)?
                    .parse()
                    .map_err(|_| invalid())
            };
            CpuSpeculation::new(
                parse_number(tokens)?,
                parse_number(scratch)?,
                parse_number(checkpoint)?,
            )
            .map(Some)
            .map_err(|_| invalid())
        }
        _ => Err(invalid()),
    }
}

const fn resolve_kv_cache_for_placement(
    placement: Placement,
    kv_cache: Option<KvCachePolicy>,
) -> Option<KvCachePolicy> {
    match (placement, kv_cache) {
        (Placement::Vulkan, None | Some(KvCachePolicy::F16)) => Some(KvCachePolicy::F16),
        (Placement::Vulkan, Some(_)) => None,
        (_, Some(policy)) => Some(policy),
        (_, None) => Some(bonsai_prism_runtime::swift_session::DEFAULT_KV_CACHE),
    }
}

const fn placement_supports_config(config: &Config) -> bool {
    !matches!(
        (
            config.placement,
            config.mmproj.is_some(),
            config.cpu_prefill.is_some(),
            config.cpu_speculation.is_some(),
        ),
        (Placement::Vulkan, true, _, _)
            | (Placement::Vulkan, _, true, _)
            | (Placement::Vulkan, _, _, true)
    )
}

fn parse_capacity_with_policy(
    context_tokens: Option<OsString>,
    memory_limit_bytes: Option<OsString>,
    kv_cache: KvCachePolicy,
    admission_policy: AdmissionPolicy,
) -> io::Result<Option<(usize, u64)>> {
    match (context_tokens, memory_limit_bytes) {
        (None, None) => Ok(None),
        (Some(context), Some(memory)) => {
            let context = context
                .to_str()
                .ok_or_else(invalid)?
                .parse::<usize>()
                .map_err(|_| invalid())?;
            let memory = memory
                .to_str()
                .ok_or_else(invalid)?
                .parse::<u64>()
                .map_err(|_| invalid())?;
            if context < 4_352
                || !context.is_multiple_of(256)
                || memory.checked_add(4 * 1024 * 1024 * 1024).is_none()
            {
                return Err(invalid());
            }
            match admission_policy {
                AdmissionPolicy::Conservative => {
                    validate_context_capacity_with_kv(context, memory, kv_cache)
                        .map_err(|_| invalid())?;
                }
                AdmissionPolicy::CalibratedCow => {
                    validate_context_capacity_with_kv(context, u64::MAX, kv_cache)
                        .map_err(|_| invalid())?;
                }
            }
            Ok(Some((context, memory)))
        }
        _ => Err(invalid()),
    }
}

#[cfg(test)]
fn parse_capacity(
    context_tokens: Option<OsString>,
    memory_limit_bytes: Option<OsString>,
    kv_cache: KvCachePolicy,
) -> io::Result<Option<(usize, u64)>> {
    parse_capacity_with_policy(
        context_tokens,
        memory_limit_bytes,
        kv_cache,
        AdmissionPolicy::Conservative,
    )
}

fn decode_sha256_hex(hex: &str) -> io::Result<[u8; 32]> {
    let bytes = hex.as_bytes();
    if bytes.len() != 64 {
        return Err(invalid());
    }
    let mut digest = [0_u8; 32];
    for (index, pair) in bytes.as_chunks::<2>().0.iter().enumerate() {
        let hi = match pair[0] {
            b'0'..=b'9' => pair[0] - b'0',
            b'a'..=b'f' => pair[0] - b'a' + 10,
            _ => return Err(invalid()),
        };
        let lo = match pair[1] {
            b'0'..=b'9' => pair[1] - b'0',
            b'a'..=b'f' => pair[1] - b'a' + 10,
            _ => return Err(invalid()),
        };
        let slot = digest.get_mut(index).ok_or_else(invalid)?;
        *slot = (hi << 4) | lo;
    }
    Ok(digest)
}

fn expected_verified_sections(config: &Config) -> io::Result<Vec<(u32, u64, [u8; 32])>> {
    let mut expected = Vec::with_capacity(4);
    expected.push((
        0_u32,
        config.representation.model_bytes(),
        decode_sha256_hex(config.representation.model_sha256())?,
    ));
    if let Some(ref daisyrose) = config.daisyrose_lora {
        let slot = u32::try_from(expected.len()).map_err(|_| invalid())?;
        expected.push((slot, daisyrose.bytes, decode_sha256_hex(&daisyrose.sha256)?));
    }
    if config.mmproj.is_some() {
        let slot = u32::try_from(expected.len()).map_err(|_| invalid())?;
        expected.push((slot, MMPROJ_BYTES, decode_sha256_hex(MMPROJ_SHA256)?));
    }
    Ok(expected)
}

fn verify_broker_pipe_sections(
    endpoint: &crate::broker_pipe::ServerEndpoint,
    config: &Config,
    expected_count: usize,
) -> io::Result<()> {
    let expected = expected_verified_sections(config)
        .map_err(|error| startup::at(Stage::Arguments, &error))?;
    if expected.len() != expected_count {
        return Err(startup::at(Stage::Arguments, &invalid()));
    }
    endpoint
        .wait_for_verified_sections(expected_count, std::time::Duration::from_secs(15))
        .map_err(|error| startup::at(Stage::Model, &error))?;
    let received = endpoint.verified_sections();
    if received.len() != expected.len() {
        return Err(startup::at(
            Stage::Model,
            &io::Error::new(
                io::ErrorKind::InvalidData,
                "broker-pipe verified section count mismatch",
            ),
        ));
    }
    for (slot, byte_len, sha256) in expected {
        let matched = received
            .iter()
            .find(|descriptor| descriptor.slot == slot)
            .ok_or_else(|| {
                startup::at(
                    Stage::Model,
                    &io::Error::new(
                        io::ErrorKind::InvalidData,
                        "missing broker-pipe verified section slot",
                    ),
                )
            })?;
        if matched.handle == 0 || matched.byte_len != byte_len || matched.sha256 != sha256 {
            return Err(startup::at(
                Stage::Model,
                &io::Error::new(
                    io::ErrorKind::InvalidData,
                    "broker-pipe verified section attestation mismatch",
                ),
            ));
        }
    }
    Ok(())
}

enum PreparedTransport {
    Loopback {
        port: u16,
        guard: bonsai_windows_credentials::RetainedCredential,
        key: String,
    },
    BrokerPipe {
        endpoint: crate::broker_pipe::ServerEndpoint,
        key: crate::broker_pipe::ScrubbedKey,
    },
}

fn prepare_transport(transport: &ChildTransport) -> io::Result<PreparedTransport> {
    match transport {
        ChildTransport::Loopback {
            port,
            key: key_path,
        } => {
            let (guard, key) = crate::configuration::read_key(key_path)
                .map_err(|error| startup::at(Stage::Key, &error))?;
            Ok(PreparedTransport::Loopback {
                port: *port,
                guard,
                key,
            })
        }
        ChildTransport::BrokerPipe => {
            let (endpoint, key) = crate::broker_pipe::ServerEndpoint::accept_stdio()
                .map_err(|error| startup::at(Stage::Key, &error))?;
            Ok(PreparedTransport::BrokerPipe { endpoint, key })
        }
    }
}

fn required_memory_bytes(config: &Config) -> io::Result<u64> {
    const DAISYROSE_SCRATCH_HEADROOM_BYTES: u64 = 16 * 1024 * 1024;
    let base_required_bytes = if let Some((_, memory)) = config.capacity {
        memory + 4 * 1024 * 1024 * 1024
    } else {
        let extra = if config.representation.is_nariq() || config.representation.is_bonsai_2_5() {
            calibrated_non_context_memory_bytes(config.representation)
                .saturating_sub(NON_CONTEXT_MEMORY_BYTES)
        } else {
            0
        };
        crate::configuration::REQUIRED_AVAILABLE_BYTES
            .checked_add(extra)
            .ok_or_else(|| {
                startup::at(
                    Stage::Memory,
                    &io::Error::other("native Swift startup memory limit overflow"),
                )
            })?
    };
    let vision_reserve_bytes = if config.mmproj.is_some() {
        u64::try_from(MMPROJ_MEMORY_BYTES).map_err(|_| startup::at(Stage::Arguments, &invalid()))?
    } else {
        0
    };
    let daisyrose_reserve_bytes = config
        .daisyrose_lora
        .as_ref()
        .and_then(|item| item.bytes.checked_add(DAISYROSE_SCRATCH_HEADROOM_BYTES))
        .unwrap_or(0);
    base_required_bytes
        .checked_add(vision_reserve_bytes)
        .and_then(|bytes| bytes.checked_add(daisyrose_reserve_bytes))
        .ok_or_else(|| {
            startup::at(
                Stage::Memory,
                &io::Error::other("native Swift startup memory limit overflow"),
            )
        })
}

enum PreparedBackend {
    Cpu {
        session: Box<SwiftSession>,
        placement: Placement,
    },
    Vulkan(Box<crate::native_backend::VulkanSwiftSession>),
}

async fn serve_prepared(
    prepared_transport: PreparedTransport,
    session: SwiftSession,
    placement: Placement,
    options: WorkerOptions,
    prefill_row_workers: Option<usize>,
) -> io::Result<()> {
    serve_prepared_backend(
        prepared_transport,
        PreparedBackend::Cpu {
            session: Box::new(session),
            placement,
        },
        options,
        prefill_row_workers,
    )
    .await
}

async fn serve_prepared_vulkan(
    prepared_transport: PreparedTransport,
    session: crate::native_backend::VulkanSwiftSession,
    options: WorkerOptions,
    prefill_row_workers: Option<usize>,
) -> io::Result<()> {
    serve_prepared_backend(
        prepared_transport,
        PreparedBackend::Vulkan(Box::new(session)),
        options,
        prefill_row_workers,
    )
    .await
}

async fn serve_prepared_backend(
    prepared_transport: PreparedTransport,
    backend: PreparedBackend,
    options: WorkerOptions,
    prefill_row_workers: Option<usize>,
) -> io::Result<()> {
    let served = match prepared_transport {
        PreparedTransport::Loopback { port, guard, key } => {
            let listener = bound_loopback_socket(port)
                .and_then(|socket| socket.listen(1))
                .map_err(|error| startup::at(Stage::Listener, &error))?;
            let result = match backend {
                PreparedBackend::Cpu { session, placement } => {
                    Box::pin(crate::native_backend::serve_with_prefill_workers(
                        listener,
                        *session,
                        key,
                        placement,
                        options,
                        prefill_row_workers,
                    ))
                    .await
                }
                PreparedBackend::Vulkan(session) => {
                    Box::pin(crate::native_backend::serve_vulkan_with_prefill_workers(
                        listener,
                        *session,
                        key,
                        options,
                        prefill_row_workers,
                    ))
                    .await
                }
            };
            drop(guard);
            result
        }
        PreparedTransport::BrokerPipe { endpoint, key } => match backend {
            PreparedBackend::Cpu { session, placement } => {
                Box::pin(
                    crate::native_backend::serve_broker_pipe_with_prefill_workers(
                        endpoint,
                        *session,
                        key.as_str(),
                        placement,
                        options,
                        prefill_row_workers,
                    ),
                )
                .await
            }
            PreparedBackend::Vulkan(session) => {
                Box::pin(
                    crate::native_backend::serve_broker_pipe_vulkan_with_prefill_workers(
                        endpoint,
                        *session,
                        key.as_str(),
                        options,
                        prefill_row_workers,
                    ),
                )
                .await
            }
        },
    };
    served.map_err(|_| {
        startup::at(
            Stage::Backend,
            &io::Error::other("native Swift backend admission closed"),
        )
    })
}

fn load_prepared_backend(config: &Config) -> io::Result<PreparedBackend> {
    if config
        .requested_prefill_tokens
        .is_some_and(|requested| config.cpu_prefill.map(CpuPrefill::tokens) != Some(requested))
    {
        return Err(startup::at(
            Stage::Arguments,
            &io::Error::new(
                io::ErrorKind::Unsupported,
                "requested CPU prefill tile size requires runtime support",
            ),
        ));
    }
    let context_tokens = config.capacity.map_or(
        bonsai_prism_runtime::model::swift::CONTEXT_TOKENS,
        |(tokens, _)| tokens,
    );
    let daisyrose = config
        .daisyrose_lora
        .as_ref()
        .map(|item| MultiRankAdapterSpec {
            path: &item.path,
            bytes: item.bytes,
            sha256: &item.sha256,
        });
    if config.placement == Placement::Vulkan {
        let session = crate::native_backend::VulkanSwiftSession::load_profile_with_adapters(
            &config.model,
            config.representation,
            daisyrose,
            config.seed,
            config.row_workers,
            context_tokens,
            config.prompt_reuse,
            config.branch_checkpoints,
        )
        .map_err(|error| startup::at(Stage::Model, error.as_ref()))?;
        return Ok(PreparedBackend::Vulkan(Box::new(session)));
    }
    let (parallelism, prefill_parallelism) =
        RowParallelism::new_pair(config.row_workers, config.prefill_row_workers)
            .map_err(|error| startup::at(Stage::Arguments, error.as_ref()))?;
    let mut session = SwiftSession::load_profile_with_adapters(
        &config.model,
        config.representation,
        daisyrose,
        MatvecBackend::Auto,
        config.seed,
        parallelism,
        config.kv_cache,
        context_tokens,
        config.cpu_prefill,
        config.cpu_speculation,
        config.prompt_reuse,
        config.branch_checkpoints,
    )
    .map_err(|error| startup::at(Stage::Model, error.as_ref()))?;
    crate::native_backend::set_swift_prefill_row_parallelism(&mut session, prefill_parallelism);
    if let Some(ref mmproj) = config.mmproj {
        session
            .load_vision(mmproj, config.image_max_tokens)
            .map_err(|error| startup::at(Stage::Model, error.as_ref()))?;
    }
    Ok(PreparedBackend::Cpu {
        session: Box::new(session),
        placement: config.placement,
    })
}

/// Start only the private backend selected by `--native-engine-child`.
///
/// The Windows development launcher supplies its protected key, external model,
/// inherited environment and Job ownership. Loading precedes listener creation;
/// the parent retains artifact/executable guards and monitors memory and PID.
///
/// # Errors
/// Rejects arguments, environment overrides, low startup memory, key access,
/// verified model loading, listener setup and backend lifecycle failures.
pub fn run(arguments: impl IntoIterator<Item = OsString>) -> io::Result<()> {
    let config = parse(arguments).map_err(|error| startup::at(Stage::Arguments, &error))?;
    if matches!(config.transport, ChildTransport::BrokerPipe)
        && !bonsai_windows_credentials::is_app_container()
            .map_err(|error| startup::at(Stage::Environment, &error))?
    {
        return Err(startup::at(
            Stage::Environment,
            &io::Error::new(
                io::ErrorKind::PermissionDenied,
                "broker-pipe worker requires an AppContainer token",
            ),
        ));
    }
    if matches!(config.placement, Placement::Auto | Placement::Composite) {
        crate::native_backend::register_hexagon_model_path(&config.model);
    }
    crate::configuration::validate_environment(std::env::vars_os().map(|(name, _)| name))
        .map_err(|error| startup::at(Stage::Environment, &error))?;
    let required_available_bytes = required_memory_bytes(&config)?;
    if bonsai_windows_process::available_memory_bytes()
        .map_err(|error| startup::at(Stage::Memory, &error))?
        < required_available_bytes
    {
        return Err(startup::at(
            Stage::Memory,
            &io::Error::other("native Swift startup memory admission rejected"),
        ));
    }
    let prepared_transport = prepare_transport(&config.transport)?;
    if let (PreparedTransport::BrokerPipe { endpoint, .. }, Some(expected_sections)) =
        (&prepared_transport, config.verified_sections)
    {
        verify_broker_pipe_sections(endpoint, &config, expected_sections)?;
    }
    let backend = load_prepared_backend(&config)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| startup::at(Stage::Runtime, &error))?;
    let options = WorkerOptions {
        hexagon_arena_cache: config.hexagon_arena_cache,
    };
    let prefill_row_workers = Some(config.prefill_row_workers);
    match backend {
        PreparedBackend::Vulkan(session) => runtime.block_on(serve_prepared_vulkan(
            prepared_transport,
            *session,
            options,
            prefill_row_workers,
        )),
        PreparedBackend::Cpu { session, placement } => runtime.block_on(serve_prepared(
            prepared_transport,
            *session,
            placement,
            options,
            prefill_row_workers,
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files() -> io::Result<(bonsai_windows_credentials::PrivateDirectory, [OsString; 3])> {
        let directory =
            bonsai_windows_credentials::PrivateDirectory::create(&std::env::temp_dir())?;
        let paths =
            ["model.gguf", "backend.key", "adapter.gguf"].map(|name| directory.path().join(name));
        for path in &paths {
            std::fs::write(path, [])?;
        }
        Ok((directory, paths.map(PathBuf::into_os_string)))
    }

    #[test]
    fn native_child_placement_is_explicit_unique_and_does_not_change_capacity() -> io::Result<()> {
        let (_directory, [model, key, _]) = files()?;
        let base: Vec<OsString> = vec![
            "--native-engine-child".into(),
            "--model".into(),
            model,
            "--key-file".into(),
            key,
            "--port".into(),
            "8080".into(),
            "--seed".into(),
            "42".into(),
        ];
        assert_eq!(parse(base.clone())?.placement, Placement::Auto);
        for (name, expected) in [
            ("auto", Placement::Auto),
            ("cpu", Placement::Cpu),
            ("vulkan", Placement::Vulkan),
            ("composite", Placement::Composite),
        ] {
            let mut selected = base.clone();
            selected.extend(["--placement".into(), name.into()]);
            let config = parse(selected.clone())?;
            assert_eq!(config.placement, expected);
            assert_eq!(config.capacity, None);
            let expected_kv_cache = if expected == Placement::Vulkan {
                KvCachePolicy::F16
            } else {
                KvCachePolicy::Rotated4
            };
            assert_eq!(config.kv_cache, expected_kv_cache);
            selected.extend(["--placement".into(), name.into()]);
            assert!(parse(selected).is_err());
        }
        for rejected in ["", "CPU", "npu", "gpu", "cpu ", "automatic"] {
            let mut selected = base.clone();
            selected.extend(["--placement".into(), rejected.into()]);
            assert!(parse(selected).is_err());
        }
        let mut missing = base;
        missing.push("--placement".into());
        assert!(parse(missing).is_err());
        Ok(())
    }

    #[test]
    fn native_child_context_capacity_is_paired_and_checked() -> io::Result<()> {
        use bonsai_prism_runtime::swift_session::{
            CONTEXT_TOKEN_MEMORY_BYTES, MAX_CONTEXT_TOKENS, NON_CONTEXT_MEMORY_BYTES,
        };
        assert_eq!(parse_capacity(None, None, KvCachePolicy::F16)?, None);
        for tokens in [4_352, 8_192, 32_768, MAX_CONTEXT_TOKENS] {
            let memory = NON_CONTEXT_MEMORY_BYTES
                + u64::try_from(tokens).map_err(io::Error::other)? * CONTEXT_TOKEN_MEMORY_BYTES;
            assert_eq!(
                parse_capacity(
                    Some(tokens.to_string().into()),
                    Some(memory.to_string().into()),
                    KvCachePolicy::F16,
                )?,
                Some((tokens, memory)),
            );
            assert!(
                parse_capacity(
                    Some(tokens.to_string().into()),
                    Some((memory - 1).to_string().into()),
                    KvCachePolicy::F16,
                )
                .is_err()
            );
        }
        for (tokens, memory) in [
            (Some("8192"), None),
            (None, Some("20000000000")),
            (Some("0"), Some("20000000000")),
            (Some("4351"), Some("20000000000")),
            (Some("8193"), Some("20000000000")),
            (Some("63232"), Some("20000000000")),
            (Some("8192"), Some("18446744073709551615")),
        ] {
            assert!(
                parse_capacity(
                    tokens.map(OsString::from),
                    memory.map(OsString::from),
                    KvCachePolicy::F16,
                )
                .is_err()
            );
        }
        Ok(())
    }

    #[test]
    fn native_child_capacity_uses_final_cache_selection_in_either_argument_order() -> io::Result<()>
    {
        use bonsai_prism_runtime::swift_session::NON_CONTEXT_MEMORY_BYTES;
        let (_directory, [model, key, _adapter]) = files()?;
        let base = vec![
            OsString::from("--native-engine-child"),
            "--model".into(),
            model,
            "--key-file".into(),
            key,
            "--port".into(),
            "8081".into(),
            "--seed".into(),
            "42".into(),
        ];
        for context in [65_536_u64, 131_072, 262_144] {
            let memory = NON_CONTEXT_MEMORY_BYTES + context * 34_016;
            for cache_first in [false, true] {
                let capacity: [OsString; 4] = [
                    "--context-tokens".into(),
                    context.to_string().into(),
                    "--memory-limit-bytes".into(),
                    memory.to_string().into(),
                ];
                let cache = [OsString::from("--kv-cache"), "rsq4".into()];
                let mut args = base.clone();
                if cache_first {
                    args.extend(cache);
                    args.extend(capacity);
                } else {
                    args.extend(capacity);
                    args.extend(cache);
                }
                let selected = parse(args.clone())?;
                assert_eq!(
                    selected.capacity,
                    Some((usize::try_from(context).map_err(io::Error::other)?, memory))
                );
                let cache_index = if cache_first { 10 } else { 14 };
                args[cache_index] = "f16".into();
                assert!(parse(args).is_err());
            }
        }
        Ok(())
    }

    #[test]
    fn native_child_k4v3_capacity_covers_projected_and_full_model_windows() -> io::Result<()> {
        use bonsai_prism_runtime::swift_session::NON_CONTEXT_MEMORY_BYTES;
        let (_directory, [model, key, _adapter]) = files()?;
        let base = vec![
            OsString::from("--native-engine-child"),
            "--model".into(),
            model,
            "--key-file".into(),
            key,
            "--port".into(),
            "8081".into(),
            "--seed".into(),
            "42".into(),
        ];
        let cache = KvCachePolicy::Rotated4Keys3Values;
        // Derive the process charge from the codec's records (including scales),
        // two independently live caches, 24 score rows and token/page metadata.
        let kv = cache
            .layer_payload_bytes(256, 4, 1)
            .map_err(io::Error::other)?
            * 16;
        let per_token = u64::try_from(2 * kv + 24 * 4 + 128).map_err(io::Error::other)?;
        let gib = 1024 * 1024 * 1024;
        let usable_at_20_gib = 20 * gib - 4 * gib - gib / 4 - NON_CONTEXT_MEMORY_BYTES;
        assert!(170_240 * per_token <= usable_at_20_gib);
        assert!(170_496 * per_token > usable_at_20_gib);
        for context in [170_240, 262_144] {
            let memory = NON_CONTEXT_MEMORY_BYTES + context * per_token;
            for cache_first in [false, true] {
                let capacity: [OsString; 4] = [
                    "--context-tokens".into(),
                    context.to_string().into(),
                    "--memory-limit-bytes".into(),
                    memory.to_string().into(),
                ];
                let cache_args = [OsString::from("--kv-cache"), "rsq4k3v".into()];
                let mut args = base.clone();
                if cache_first {
                    args.extend(cache_args);
                    args.extend(capacity);
                } else {
                    args.extend(capacity);
                    args.extend(cache_args);
                }
                let selected = parse(args.clone())?;
                assert_eq!(selected.kv_cache, cache);
                assert_eq!(
                    selected.capacity,
                    Some((usize::try_from(context).map_err(io::Error::other)?, memory)),
                );
                let memory_index = if cache_first { 14 } else { 12 };
                args[memory_index] = (memory - 1).to_string().into();
                assert!(parse(args.clone()).is_err());
                args[memory_index] = memory.to_string().into();
                let cache_index = if cache_first { 10 } else { 14 };
                for wrong_cache in ["f16", "rsq4"] {
                    args[cache_index] = wrong_cache.into();
                    assert!(parse(args.clone()).is_err());
                }
            }
        }
        let maximum_memory = NON_CONTEXT_MEMORY_BYTES + 262_144 * per_token;
        for (tokens, memory) in [
            (262_400, maximum_memory + 256 * per_token),
            (262_144, u64::MAX),
        ] {
            assert!(
                parse_capacity(
                    Some(tokens.to_string().into()),
                    Some(memory.to_string().into()),
                    cache,
                )
                .is_err()
            );
        }
        Ok(())
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "One admission control retains valid tile/reuse profiles beside all paired-option rejections."
    )]
    fn native_child_cpu_prefill_is_paired_and_memory_admitted() -> io::Result<()> {
        use bonsai_prism_runtime::swift_session::{
            CONTEXT_TOKEN_MEMORY_BYTES, NON_CONTEXT_MEMORY_BYTES,
        };
        let (_directory, [model, key, _adapter]) = files()?;
        let base = vec![
            OsString::from("--native-engine-child"),
            "--model".into(),
            model,
            "--key-file".into(),
            key,
            "--port".into(),
            "8081".into(),
            "--seed".into(),
            "42".into(),
        ];
        assert!(parse(base.clone())?.cpu_prefill.is_none());
        let reserve = 64 * 1024 * 1024;
        let memory = NON_CONTEXT_MEMORY_BYTES + 8_192 * CONTEXT_TOKEN_MEMORY_BYTES + reserve;
        for tokens in ["8", "16", "32"] {
            let mut selected = base.clone();
            selected.extend(
                [
                    "--kv-cache",
                    "f16",
                    "--prompt-reuse",
                    "latest",
                    "--context-tokens",
                    "8192",
                    "--memory-limit-bytes",
                    &memory.to_string(),
                    "--cpu-prefill-tokens",
                    tokens,
                    "--cpu-prefill-scratch-bytes",
                    &reserve.to_string(),
                ]
                .map(OsString::from),
            );
            let config = parse(selected.clone())?;
            assert_eq!(
                config.cpu_prefill.map(CpuPrefill::tokens),
                Some(tokens.parse().map_err(io::Error::other)?)
            );
            assert_eq!(
                config.cpu_prefill.map(CpuPrefill::scratch_bytes),
                Some(usize::try_from(reserve).map_err(io::Error::other)?)
            );
            assert_eq!(config.prompt_reuse, PromptReuse::Latest);
            assert_eq!(config.kv_cache, KvCachePolicy::F16);
            for cache in ["rsq4", "rsq4k3v"] {
                let mut rotated = selected.clone();
                rotated[10] = cache.into();
                assert_eq!(parse(rotated)?.kv_cache.name(), cache);
            }
            let mut insufficient = selected.clone();
            insufficient[16] = OsString::from((memory - 1).to_string());
            assert!(parse(insufficient).is_err());
            selected.extend(["--cpu-prefill-tokens", tokens].map(OsString::from));
            assert!(parse(selected).is_err());
        }
        for options in [
            vec!["--cpu-prefill-tokens", "8"],
            vec!["--cpu-prefill-scratch-bytes", "67108864"],
            vec![
                "--cpu-prefill-tokens",
                "8",
                "--cpu-prefill-scratch-bytes",
                "67108864",
            ],
            vec![
                "--kv-cache",
                "f16",
                "--cpu-prefill-tokens",
                "8",
                "--cpu-prefill-scratch-bytes",
                "67108864",
            ],
            vec![
                "--context-tokens",
                "8192",
                "--memory-limit-bytes",
                "20000000000",
                "--cpu-prefill-tokens",
                "8",
                "--cpu-prefill-scratch-bytes",
                "67108864",
            ],
            vec![
                "--kv-cache",
                "f16",
                "--context-tokens",
                "8192",
                "--memory-limit-bytes",
                "20000000000",
                "--cpu-prefill-tokens",
                "7",
                "--cpu-prefill-scratch-bytes",
                "67108864",
            ],
            vec![
                "--kv-cache",
                "f16",
                "--context-tokens",
                "8192",
                "--memory-limit-bytes",
                "20000000000",
                "--cpu-prefill-tokens",
                "8",
                "--cpu-prefill-scratch-bytes",
                "0",
            ],
        ] {
            let mut arguments = base.clone();
            arguments.extend(options.into_iter().map(OsString::from));
            assert!(parse(arguments).is_err());
        }
        Ok(())
    }

    fn speculation_arguments() -> io::Result<(
        bonsai_windows_credentials::PrivateDirectory,
        Vec<OsString>,
        u64,
    )> {
        use bonsai_prism_runtime::swift_session::{CONTEXT_TOKEN_MEMORY_BYTES, STATE_MEMORY_BYTES};
        let (directory, [model, key, _adapter]) = files()?;
        let tile = 64 * 1024 * 1024;
        let logits = 16 * 1024 * 1024;
        let checkpoint = STATE_MEMORY_BYTES;
        let memory = NON_CONTEXT_MEMORY_BYTES
            + 8_192 * CONTEXT_TOKEN_MEMORY_BYTES
            + u64::try_from(tile + logits + checkpoint).map_err(io::Error::other)?;
        let base = vec![
            OsString::from("--native-engine-child"),
            "--model".into(),
            model,
            "--key-file".into(),
            key,
            "--port".into(),
            "8081".into(),
            "--seed".into(),
            "42".into(),
            "--kv-cache".into(),
            "f16".into(),
            "--context-tokens".into(),
            "8192".into(),
            "--memory-limit-bytes".into(),
            memory.to_string().into(),
            "--cpu-prefill-tokens".into(),
            "8".into(),
            "--cpu-prefill-scratch-bytes".into(),
            tile.to_string().into(),
        ];
        Ok((directory, base, memory))
    }

    #[test]
    fn native_child_cpu_speculation_requires_complete_triple_and_both_reserves() -> io::Result<()> {
        let (_directory, base, memory) = speculation_arguments()?;
        let logits = 16 * 1024 * 1024;
        let checkpoint = bonsai_prism_runtime::swift_session::STATE_MEMORY_BYTES;
        assert!(parse(base.clone())?.cpu_speculation.is_none());
        for tokens in [1, 4, 8] {
            let mut selected = base.clone();
            selected.extend([
                "--cpu-speculative-tokens".into(),
                tokens.to_string().into(),
                "--cpu-verification-scratch-bytes".into(),
                logits.to_string().into(),
                "--cpu-speculative-checkpoint-bytes".into(),
                checkpoint.to_string().into(),
            ]);
            for cache in ["f16", "rsq4", "rsq4k3v"] {
                selected[10] = cache.into();
                let config = parse(selected.clone())?;
                let speculation = config.cpu_speculation.ok_or_else(invalid)?;
                assert_eq!(speculation.tokens(), tokens);
                assert_eq!(speculation.verification_scratch_bytes(), logits);
                assert_eq!(speculation.checkpoint_bytes(), checkpoint);
                assert_eq!(config.kv_cache.name(), cache);
            }
            selected[10] = "f16".into();
            let mut insufficient = selected.clone();
            insufficient[14] = (memory - 1).to_string().into();
            assert!(parse(insufficient).is_err());
            let mut no_tiles = selected.clone();
            no_tiles.drain(15..19);
            assert!(parse(no_tiles).is_err());
            let mut small_checkpoint = selected.clone();
            small_checkpoint[24] = (checkpoint - 1).to_string().into();
            assert!(parse(small_checkpoint).is_err());
            selected.extend(["--cpu-speculative-tokens".into(), "4".into()]);
            assert!(parse(selected).is_err());
        }
        for (tokens, scratch, checkpoint) in [
            (None, Some("1"), Some("1073741824")),
            (Some("4"), None, Some("1073741824")),
            (Some("4"), Some("1"), None),
            (Some("0"), Some("1"), Some("1073741824")),
            (Some("9"), Some("1"), Some("1073741824")),
            (Some("4"), Some("0"), Some("1073741824")),
            (Some("4"), Some("1"), Some("0")),
        ] {
            assert!(
                parse_speculation(
                    tokens.map(OsString::from),
                    scratch.map(OsString::from),
                    checkpoint.map(OsString::from)
                )
                .is_err()
            );
        }
        Ok(())
    }

    #[test]
    fn native_startup_rejects_missing_paths_without_loading_or_leaking_values() -> io::Result<()> {
        let (directory, [_model, key, _adapter]) = files()?;
        let arguments = [
            OsString::from("--native-engine-child"),
            "--model".into(),
            directory
                .path()
                .join("PRIVATE_MISSING_MODEL.gguf")
                .into_os_string(),
            "--key-file".into(),
            key,
            "--port".into(),
            "8081".into(),
            "--seed".into(),
            "42".into(),
        ];
        let error = run(arguments)
            .err()
            .ok_or_else(|| io::Error::other("missing file was accepted"))?;
        let mut output = Vec::new();
        startup::write_failure(&mut output, &error)?;
        let output = String::from_utf8(output).map_err(io::Error::other)?;
        assert!(output.starts_with(
            "private-inference-startup-failed stage=arguments errorKind=NotFound osCode="
        ));
        assert!(!output.contains("PRIVATE_MISSING_MODEL"));
        Ok(())
    }

    #[test]
    fn native_child_rejects_unknown_duplicate_relative_and_unbounded_options() -> io::Result<()> {
        let good = [
            "--native-engine-child",
            "--model",
            "C:\\External\\model.gguf",
            "--port",
            "8081",
            "--key-file",
            "C:\\External\\backend.key",
            "--seed",
            "42",
        ]
        .map(OsString::from);
        let (_directory, [model, key, _adapter]) = files()?;
        let mut good = good;
        good[2] = model;
        good[6] = key;
        assert!(parse(good.clone()).is_ok());
        assert_eq!(parse(good.clone())?.row_workers, 1);
        assert_eq!(parse(good.clone())?.prompt_reuse, PromptReuse::Disabled);
        for (value, expected) in [
            ("disabled", PromptReuse::Disabled),
            ("latest", PromptReuse::Latest),
        ] {
            let mut selected = good.to_vec();
            selected.extend(["--prompt-reuse", value].map(OsString::from));
            assert_eq!(parse(selected.clone())?.prompt_reuse, expected);
            selected.extend(["--prompt-reuse", value].map(OsString::from));
            assert!(parse(selected).is_err());
        }
        for value in ["", "true", "Latest", "snapshot", "approximate"] {
            let mut selected = good.to_vec();
            selected.extend(["--prompt-reuse", value].map(OsString::from));
            assert!(parse(selected).is_err());
        }
        for value in ["0", "19", "invalid"] {
            let mut args = good.to_vec();
            args.extend(["--row-workers", value].map(OsString::from));
            assert!(parse(args).is_err());
        }
        let mut parallel = good.to_vec();
        parallel.extend(["--row-workers", "2"].map(OsString::from));
        assert_eq!(parse(parallel).is_ok(), cfg!(feature = "parallel-rows"));
        for extra in [
            vec!["--seed", "1"],
            vec!["--profile", "swift-pq2-assessment"],
            vec!["--native-engine-child"],
        ] {
            let mut args = good.to_vec();
            args.extend(extra.into_iter().map(OsString::from));
            assert!(parse(args).is_err());
        }
        let mut args = good.clone();
        args[2] = OsString::from("relative.gguf");
        assert!(parse(args).is_err());
        for value in ["0", "1023", "65536"] {
            let mut args = good.clone();
            args[4] = OsString::from(value);
            assert!(parse(args).is_err());
        }
        let mut args = good;
        args[8] = OsString::from(u32::MAX.to_string());
        assert!(parse(args).is_err());
        Ok(())
    }

    #[test]
    fn native_child_selects_bonsai25_ptq1_and_rejects_removed_legacy_options() -> io::Result<()> {
        let (directory, [model, key, adapter]) = files()?;
        let base = vec![
            OsString::from("--native-engine-child"),
            "--model".into(),
            model,
            "--key-file".into(),
            key,
            "--port".into(),
            "8081".into(),
            "--seed".into(),
            "42".into(),
        ];
        let default = parse(base.clone())?;
        assert_eq!(default.representation, Representation::Bonsai25Pq2);
        let mut missing = base.clone();
        missing[2] = directory.path().join("missing.gguf").into_os_string();
        assert!(parse(missing).is_err());
        let mut nonregular = base.clone();
        nonregular[2] = directory.path().as_os_str().to_owned();
        assert!(parse(nonregular).is_err());
        let mut ptq1 = base.clone();
        ptq1.extend([OsString::from("--bonsai-2-5"), "ptq1".into()]);
        let config = parse(ptq1)?;
        assert_eq!(config.representation, Representation::Bonsai25Ptq1);
        for extra in [
            vec![OsString::from("--use-repacked-ptq1")],
            vec![OsString::from("--abliterate-lora"), adapter],
            vec![OsString::from("--abliteration-scale"), "1.3".into()],
        ] {
            let mut invalid = base.clone();
            invalid.extend(extra);
            assert!(parse(invalid).is_err());
        }
        Ok(())
    }

    #[test]
    fn native_child_selects_nariq_with_daisyrose_and_rejects_incompatible_modes() -> io::Result<()>
    {
        let (directory, [model, key, _adapter]) = files()?;
        let daisy = directory.path().join("daisyrose.gguf");
        std::fs::write(&daisy, [])?;
        let mmproj = directory.path().join("mmproj.gguf");
        std::fs::write(&mmproj, [])?;
        let cache_dir = directory.path().join("cache");
        let base = vec![
            OsString::from("--native-engine-child"),
            "--model".into(),
            model,
            "--key-file".into(),
            key,
            "--port".into(),
            "8080".into(),
            "--seed".into(),
            "42".into(),
        ];

        for (flag_val, expected_repr) in [
            ("2.5bpw", Representation::NariQ2_5Bpw),
            ("3.0bpw", Representation::NariQ3_0Bpw),
            ("2.0bpw", Representation::NariQ2_0Bpw),
        ] {
            let mut args = base.clone();
            args.extend([
                OsString::from("--nariq"),
                OsString::from(flag_val),
                OsString::from("--daisyrose-lora"),
                daisy.clone().into_os_string(),
                OsString::from("--daisyrose-lora-bytes"),
                "128".into(),
                OsString::from("--daisyrose-lora-sha256"),
                "ab".repeat(32).into(),
                OsString::from("--kv-cache"),
                "rsq4".into(),
                OsString::from("--placement"),
                "cpu".into(),
            ]);
            let cfg = parse(args)?;
            assert_eq!(cfg.representation, expected_repr);
            assert!(cfg.daisyrose_lora.is_some());
            assert!(cfg.mmproj.is_none());
        }

        // Uncalibrated required_memory_bytes accounts for 2.5bpw / 3.0bpw size + 256 MiB headroom.
        let mut args_30 = base.clone();
        args_30.extend([OsString::from("--nariq"), "3.0bpw".into()]);
        let cfg_30 = parse(args_30)?;
        assert!(required_memory_bytes(&cfg_30)? > crate::configuration::REQUIRED_AVAILABLE_BYTES);

        for bad in [
            vec!["--nariq", "2.5bpw", "--use-repacked-ptq1"],
            vec!["--use-repacked-ptq1", "--nariq", "2.5bpw"],
            vec!["--nariq", "2.5bpw", "--nariq", "3.0bpw"],
            vec!["--nariq", "4.0bpw"],
            vec!["--nariq", "2.5bpw", "--placement", "vulkan"],
            vec!["--nariq", "2.5bpw", "--placement", "composite"],
        ] {
            let mut invalid_args = base.clone();
            invalid_args.extend(bad.into_iter().map(OsString::from));
            assert!(parse(invalid_args).is_err());
        }
        let mut with_mmproj = base.clone();
        with_mmproj.extend([
            OsString::from("--nariq"),
            "2.5bpw".into(),
            OsString::from("--mmproj"),
            mmproj.into(),
        ]);
        assert!(parse(with_mmproj).is_err());

        let mut with_cache = base;
        with_cache.extend([
            OsString::from("--nariq"),
            "2.5bpw".into(),
            OsString::from("--hexagon-arena-cache"),
            cache_dir.into_os_string(),
        ]);
        assert!(parse(with_cache).is_err());
        Ok(())
    }

    #[test]
    fn native_child_selects_bonsai25_with_daisyrose_mmproj_and_placements() -> io::Result<()> {
        let (directory, [model, key, _adapter]) = files()?;
        let daisy = directory.path().join("daisyrose.gguf");
        std::fs::write(&daisy, [])?;
        let mmproj = directory.path().join("mmproj.gguf");
        std::fs::write(&mmproj, [])?;
        let cache_dir = directory.path().join("cache");
        let base = vec![
            OsString::from("--native-engine-child"),
            "--model".into(),
            model,
            "--key-file".into(),
            key,
            "--port".into(),
            "8080".into(),
            "--seed".into(),
            "42".into(),
        ];

        for (flag, expected_repr) in [
            ("pq2", Representation::Bonsai25Pq2),
            ("ptq1", Representation::Bonsai25Ptq1),
        ] {
            for (placement_opt, kv_cache, with_mmproj) in [
                (None, "rsq4", true),
                (Some("cpu"), "rsq4", true),
                (Some("vulkan"), "f16", false),
                (Some("composite"), "rsq4", true),
            ] {
                let mut args = base.clone();
                args.extend([
                    OsString::from("--bonsai-2-5"),
                    OsString::from(flag),
                    OsString::from("--daisyrose-lora"),
                    daisy.clone().into_os_string(),
                    OsString::from("--daisyrose-lora-bytes"),
                    "128".into(),
                    OsString::from("--daisyrose-lora-sha256"),
                    "ab".repeat(32).into(),
                    OsString::from("--kv-cache"),
                    kv_cache.into(),
                ]);
                if with_mmproj {
                    args.extend([OsString::from("--mmproj"), mmproj.clone().into_os_string()]);
                }
                if let Some(p) = placement_opt {
                    args.extend([OsString::from("--placement"), OsString::from(p)]);
                }
                let cfg = parse(args)?;
                assert_eq!(cfg.representation, expected_repr);
                assert!(cfg.daisyrose_lora.is_some());
                assert_eq!(cfg.mmproj.is_some(), with_mmproj);
            }
            let mut with_cache_b25 = base.clone();
            with_cache_b25.extend([
                OsString::from("--bonsai-2-5"),
                OsString::from(flag),
                OsString::from("--hexagon-arena-cache"),
                cache_dir.clone().into_os_string(),
            ]);
            assert_eq!(parse(with_cache_b25)?.representation, expected_repr);
        }
        let mut b25_pq2_args = base.clone();
        b25_pq2_args.extend([OsString::from("--bonsai-2-5"), "pq2".into()]);
        assert_eq!(
            required_memory_bytes(&parse(b25_pq2_args)?)?,
            crate::configuration::REQUIRED_AVAILABLE_BYTES + 47_185_920
        );
        for bad in [
            vec!["--bonsai-2-5", "pq2", "--nariq", "2.5bpw"],
            vec!["--nariq", "2.5bpw", "--bonsai-2-5", "ptq1"],
            vec!["--bonsai-2-5", "pq2", "--bonsai-2-5", "ptq1"],
            vec!["--bonsai-2-5", "q4_0"],
        ] {
            let mut invalid_args = base.clone();
            invalid_args.extend(bad.into_iter().map(OsString::from));
            assert!(parse(invalid_args).is_err());
        }
        Ok(())
    }

    #[test]
    fn native_child_kv_cache_defaults_to_k4v4_with_explicit_unique_recovery() -> io::Result<()> {
        let (_directory, [model, key, _adapter]) = files()?;
        let base = vec![
            OsString::from("--native-engine-child"),
            "--model".into(),
            model,
            "--key-file".into(),
            key,
            "--port".into(),
            "8081".into(),
            "--seed".into(),
            "42".into(),
        ];
        assert_eq!(parse(base.clone())?.kv_cache, KvCachePolicy::Rotated4);
        for (name, expected) in [
            ("f16", KvCachePolicy::F16),
            ("rsq4", KvCachePolicy::Rotated4),
            ("rsq4vf", KvCachePolicy::Rotated4FittedValues),
            ("rsq4kvf", KvCachePolicy::Rotated4FittedKv),
            ("rsq4k3v", KvCachePolicy::Rotated4Keys3Values),
        ] {
            let mut selected = base.clone();
            selected.extend(["--kv-cache", name].map(OsString::from));
            assert_eq!(parse(selected.clone())?.kv_cache, expected);
            selected.extend(["--kv-cache", name].map(OsString::from));
            assert!(parse(selected).is_err());
        }
        for name in ["", "F16", "q4_0", "turboquant", "rsq3", "rsq4 "] {
            let mut selected = base.clone();
            selected.extend(["--kv-cache", name].map(OsString::from));
            assert!(parse(selected).is_err());
        }
        Ok(())
    }

    #[test]
    fn native_child_cpu_family_and_spec_type_flags_enforce_bounds_and_defaults() -> io::Result<()> {
        let (_directory, [model, key, _adapter]) = files()?;
        let base = vec![
            OsString::from("--native-engine-child"),
            "--model".into(),
            model,
            "--key-file".into(),
            key,
            "--port".into(),
            "8081".into(),
            "--seed".into(),
            "42".into(),
        ];

        for (family, expected_workers) in [("x1", 8_usize), ("x2", 6_usize)] {
            let mut family_only = base.clone();
            family_only.extend(["--cpu-family", family].map(OsString::from));
            let parsed = parse(family_only.clone());
            assert_eq!(
                parsed.as_ref().map(|config| config.row_workers).ok(),
                cfg!(feature = "parallel-rows").then_some(expected_workers)
            );
            let mut overridden = family_only.clone();
            overridden.extend(["--row-workers", "1"].map(OsString::from));
            assert_eq!(parse(overridden)?.row_workers, 1);
            family_only.extend(["--cpu-family", family].map(OsString::from));
            assert!(parse(family_only).is_err());
        }
        for invalid_family in ["", "x3", "X1", "x2 "] {
            let mut args = base.clone();
            args.extend(["--cpu-family", invalid_family].map(OsString::from));
            assert!(parse(args).is_err());
        }

        let mut none_args = base.clone();
        none_args.extend(["--spec-type", "none"].map(OsString::from));
        assert!(parse(none_args.clone())?.cpu_speculation.is_none());
        for extra in [
            vec!["--spec-type", "none"],
            vec!["--draft-max", "4"],
            vec!["--cpu-speculative-tokens", "4"],
            vec!["--cpu-verification-scratch-bytes", "16777216"],
            vec!["--cpu-speculative-checkpoint-bytes", "1073741824"],
        ] {
            let mut args = none_args.clone();
            args.extend(extra.into_iter().map(OsString::from));
            assert!(parse(args).is_err());
        }
        let mut bare_draft = base;
        bare_draft.extend(["--draft-max", "4"].map(OsString::from));
        assert!(parse(bare_draft).is_err());

        let (_dir2, spec_base, _memory) = speculation_arguments()?;
        let logits = 16 * 1024 * 1024;
        let checkpoint = bonsai_prism_runtime::swift_session::STATE_MEMORY_BYTES;
        let mut suffix_default = spec_base;
        suffix_default.extend([
            "--spec-type".into(),
            "suffix".into(),
            "--cpu-verification-scratch-bytes".into(),
            logits.to_string().into(),
            "--cpu-speculative-checkpoint-bytes".into(),
            checkpoint.to_string().into(),
        ]);
        let default_cfg = parse(suffix_default.clone())?;
        assert_eq!(default_cfg.cpu_speculation.ok_or_else(invalid)?.tokens(), 4);

        for draft_max in 1..=8 {
            let mut bounded = suffix_default.clone();
            bounded.extend(["--draft-max", &draft_max.to_string()].map(OsString::from));
            let cfg = parse(bounded)?;
            assert_eq!(cfg.cpu_speculation.ok_or_else(invalid)?.tokens(), draft_max);
        }
        for bad_draft in ["0", "9", "abc"] {
            let mut invalid_draft = suffix_default.clone();
            invalid_draft.extend(["--draft-max", bad_draft].map(OsString::from));
            assert!(parse(invalid_draft).is_err());
        }
        let mut conflict = suffix_default.clone();
        conflict.extend(["--draft-max", "4", "--cpu-speculative-tokens", "4"].map(OsString::from));
        assert!(parse(conflict).is_err());
        let mut dup_draft = suffix_default.clone();
        dup_draft.extend(["--draft-max", "4", "--draft-max", "2"].map(OsString::from));
        assert!(parse(dup_draft).is_err());
        suffix_default.drain(15..19);
        assert!(parse(suffix_default).is_err());
        Ok(())
    }

    #[test]
    fn native_child_mmproj_and_image_max_tokens_parse_and_validate_bounds() -> io::Result<()> {
        let (directory, [model, key, _]) = files()?;
        let mmproj = directory.path().join("mmproj.gguf");
        std::fs::write(&mmproj, [])?;
        let base: Vec<OsString> = vec![
            "--native-engine-child".into(),
            "--model".into(),
            model,
            "--key-file".into(),
            key,
            "--port".into(),
            "8080".into(),
            "--seed".into(),
            "42".into(),
        ];
        // --image-max-tokens requires --mmproj.
        let mut orphan_tokens = base.clone();
        orphan_tokens.extend(["--image-max-tokens", "1024"].map(OsString::from));
        assert!(parse(orphan_tokens).is_err());

        let mut with_mmproj = base.clone();
        with_mmproj.extend(["--mmproj".into(), mmproj.clone().into_os_string()]);
        let default_cfg = parse(with_mmproj.clone())?;
        assert_eq!(default_cfg.mmproj.as_deref(), Some(mmproj.as_path()));
        assert_eq!(default_cfg.image_max_tokens, DEFAULT_MAX_IMAGE_TOKENS);

        let mut bounded = with_mmproj.clone();
        bounded.extend(["--image-max-tokens", "512"].map(OsString::from));
        assert_eq!(parse(bounded)?.image_max_tokens, 512);

        for invalid_tokens in ["0", "63", "4097", "abc"] {
            let mut bad = with_mmproj.clone();
            bad.extend(["--image-max-tokens", invalid_tokens].map(OsString::from));
            assert!(parse(bad).is_err());
        }
        let mut duplicate = with_mmproj;
        duplicate.extend(["--mmproj".into(), mmproj.into_os_string()]);
        assert!(parse(duplicate).is_err());

        let (_dir2, spec_base, _memory) = speculation_arguments()?;
        let mut conflict = spec_base;
        let spec_mmproj = directory.path().join("spec-mmproj.gguf");
        std::fs::write(&spec_mmproj, [])?;
        conflict.extend([
            "--spec-type".into(),
            "suffix".into(),
            "--cpu-verification-scratch-bytes".into(),
            "16777216".into(),
            "--cpu-speculative-checkpoint-bytes".into(),
            bonsai_prism_runtime::swift_session::STATE_MEMORY_BYTES
                .to_string()
                .into(),
            "--mmproj".into(),
            spec_mmproj.into_os_string(),
        ]);
        assert!(parse(conflict).is_err());
        drop(base);
        Ok(())
    }

    #[test]
    fn native_child_broker_pipe_is_exclusive_and_requires_app_container() -> io::Result<()> {
        let (_directory, [model, key, _adapter]) = files()?;
        let base: Vec<OsString> = vec![
            "--native-engine-child".into(),
            "--broker-pipe".into(),
            "--model".into(),
            model.clone(),
            "--seed".into(),
            "42".into(),
        ];
        let parsed = parse(base.clone())?;
        assert_eq!(parsed.transport, ChildTransport::BrokerPipe);

        let mut duplicate = base.clone();
        duplicate.push("--broker-pipe".into());
        assert!(parse(duplicate).is_err());

        for conflict in [
            vec![OsString::from("--port"), OsString::from("8081")],
            vec![OsString::from("--key-file"), key],
        ] {
            let mut invalid_mix = base.clone();
            invalid_mix.extend(conflict);
            assert!(parse(invalid_mix).is_err());
        }

        if !bonsai_windows_credentials::is_app_container()? {
            let error = run(base)
                .err()
                .ok_or_else(|| io::Error::other("broker-pipe ran without AppContainer"))?;
            let mut output = Vec::new();
            startup::write_failure(&mut output, &error)?;
            let output = String::from_utf8(output).map_err(io::Error::other)?;
            assert!(output.starts_with(
                "private-inference-startup-failed stage=environment errorKind=PermissionDenied osCode="
            ));
            assert!(!output.contains(model.to_string_lossy().as_ref()));
        }
        Ok(())
    }

    #[test]
    fn native_child_cpu_prefill_gdn_allows_f16_and_rotated_and_enforces_non_speculative_pairing()
    -> io::Result<()> {
        use bonsai_prism_runtime::swift_session::{CONTEXT_TOKEN_MEMORY_BYTES, STATE_MEMORY_BYTES};
        let (_directory, [model, key, _adapter]) = files()?;
        let reserve = 64 * 1024 * 1024_u64;
        let prefill_mem = NON_CONTEXT_MEMORY_BYTES + 8_192 * CONTEXT_TOKEN_MEMORY_BYTES + reserve;
        let base = vec![
            OsString::from("--native-engine-child"),
            "--model".into(),
            model,
            "--key-file".into(),
            key,
            "--port".into(),
            "8081".into(),
            "--seed".into(),
            "42".into(),
        ];

        for kv_flag in ["f16", "rsq4", "rsq4vf", "rsq4kvf", "rsq4k3v"] {
            for (mode, expected_chunkwise) in [("sequential", false), ("wy-f32", true)] {
                let mut args = base.clone();
                args.extend(
                    [
                        "--kv-cache",
                        kv_flag,
                        "--context-tokens",
                        "8192",
                        "--memory-limit-bytes",
                        &prefill_mem.to_string(),
                        "--cpu-prefill-tokens",
                        "16",
                        "--cpu-prefill-scratch-bytes",
                        &reserve.to_string(),
                        "--cpu-prefill-gdn",
                        mode,
                    ]
                    .map(OsString::from),
                );
                let parsed = parse(args.clone())?;
                assert_eq!(
                    parsed.cpu_prefill.map(CpuPrefill::chunkwise_gdn),
                    Some(expected_chunkwise)
                );
                args.extend(["--cpu-prefill-gdn", mode].map(OsString::from));
                assert!(parse(args).is_err());
            }
        }

        for bad in [
            vec!["--cpu-prefill-gdn", "wy-f32"],
            vec!["--cpu-prefill-gdn", "sequential"],
            vec![
                "--kv-cache",
                "f16",
                "--context-tokens",
                "8192",
                "--memory-limit-bytes",
                &prefill_mem.to_string(),
                "--cpu-prefill-tokens",
                "16",
                "--cpu-prefill-scratch-bytes",
                &reserve.to_string(),
                "--cpu-prefill-gdn",
                "triangular",
            ],
        ] {
            let mut args = base.clone();
            args.extend(bad.into_iter().map(OsString::from));
            assert!(parse(args).is_err());
        }
        let (_dir2, spec_base, _spec_mem) = speculation_arguments()?;
        let mut spec_gdn = spec_base;
        spec_gdn.extend(
            [
                "--cpu-prefill-gdn",
                "wy-f32",
                "--cpu-speculative-tokens",
                "4",
                "--cpu-verification-scratch-bytes",
                "16777216",
                "--cpu-speculative-checkpoint-bytes",
                &STATE_MEMORY_BYTES.to_string(),
            ]
            .map(OsString::from),
        );
        assert!(parse(spec_gdn).is_err());
        Ok(())
    }

    #[test]
    fn native_child_prompt_end_and_branch_checkpoints_enforce_budgets() -> io::Result<()> {
        use bonsai_prism_runtime::swift_session::{
            CONTEXT_TOKEN_MEMORY_BYTES, MAX_REUSABLE_CHECKPOINTS, NON_CONTEXT_MEMORY_BYTES,
            STATE_MEMORY_BYTES,
        };
        let (_directory, [model, key, _adapter]) = files()?;
        let base_mem = NON_CONTEXT_MEMORY_BYTES + 8_192 * CONTEXT_TOKEN_MEMORY_BYTES;
        let base = vec![
            OsString::from("--native-engine-child"),
            "--model".into(),
            model,
            "--key-file".into(),
            key,
            "--port".into(),
            "8081".into(),
            "--seed".into(),
            "42".into(),
        ];

        let default_cfg = parse(base.clone())?;
        assert_eq!(default_cfg.prompt_reuse, PromptReuse::Disabled);
        assert_eq!(default_cfg.branch_checkpoints.checkpoints(), 1);
        assert_eq!(default_cfg.branch_checkpoints.checkpoint_bytes(), None);

        let state_u64 = u64::try_from(STATE_MEMORY_BYTES).map_err(io::Error::other)?;
        let prompt_end_mem = base_mem + state_u64;
        let mut prompt_end_args = base.clone();
        prompt_end_args.extend(
            [
                "--kv-cache",
                "f16",
                "--context-tokens",
                "8192",
                "--memory-limit-bytes",
                &prompt_end_mem.to_string(),
                "--prompt-reuse",
                "prompt-end-and-latest",
            ]
            .map(OsString::from),
        );
        let prompt_end_cfg = parse(prompt_end_args.clone())?;
        assert_eq!(prompt_end_cfg.prompt_reuse, PromptReuse::PromptEndAndLatest);
        assert_eq!(prompt_end_cfg.branch_checkpoints.checkpoints(), 1);
        prompt_end_args[14] = (prompt_end_mem - 1).to_string().into();
        assert!(parse(prompt_end_args).is_err());

        for slots in 2..=MAX_REUSABLE_CHECKPOINTS {
            let slots_u64 = u64::try_from(slots).map_err(io::Error::other)?;
            let branch_bytes = slots_u64 * state_u64;
            let total_mem = base_mem + branch_bytes;
            for mode in ["latest", "prompt-end-and-latest"] {
                let mut args = base.clone();
                args.extend(
                    [
                        "--kv-cache",
                        "f16",
                        "--context-tokens",
                        "8192",
                        "--memory-limit-bytes",
                        &total_mem.to_string(),
                        "--prompt-reuse",
                        mode,
                        "--prompt-reuse-checkpoints",
                        &slots.to_string(),
                        "--prompt-reuse-checkpoint-bytes",
                        &branch_bytes.to_string(),
                    ]
                    .map(OsString::from),
                );
                let cfg = parse(args.clone())?;
                assert_eq!(cfg.branch_checkpoints.checkpoints(), slots);
                assert_eq!(
                    cfg.branch_checkpoints.checkpoint_bytes(),
                    Some(usize::try_from(branch_bytes).map_err(io::Error::other)?)
                );
                let mut low_mem = args.clone();
                low_mem[14] = (total_mem - 1).to_string().into();
                assert!(parse(low_mem).is_err());
                let mut low_branch = args;
                low_branch[20] = (branch_bytes - 1).to_string().into();
                assert!(parse(low_branch).is_err());
            }
        }
        Ok(())
    }

    #[test]
    fn native_child_branch_checkpoints_reject_missing_capacity_and_invalid_ranges() -> io::Result<()>
    {
        let (_directory, [model, key, _adapter]) = files()?;
        let base = vec![
            OsString::from("--native-engine-child"),
            "--model".into(),
            model,
            "--key-file".into(),
            key,
            "--port".into(),
            "8081".into(),
            "--seed".into(),
            "42".into(),
        ];
        for bad in [
            vec!["--prompt-reuse", "prompt-end-and-latest"],
            vec![
                "--prompt-reuse",
                "latest",
                "--prompt-reuse-checkpoints",
                "4",
            ],
            vec![
                "--kv-cache",
                "f16",
                "--context-tokens",
                "8192",
                "--memory-limit-bytes",
                "30000000000",
                "--prompt-reuse",
                "disabled",
                "--prompt-reuse-checkpoints",
                "2",
            ],
            vec![
                "--kv-cache",
                "f16",
                "--context-tokens",
                "8192",
                "--memory-limit-bytes",
                "30000000000",
                "--prompt-reuse",
                "latest",
                "--prompt-reuse-checkpoints",
                "0",
            ],
            vec![
                "--kv-cache",
                "f16",
                "--context-tokens",
                "8192",
                "--memory-limit-bytes",
                "30000000000",
                "--prompt-reuse",
                "latest",
                "--prompt-reuse-checkpoints",
                "9",
            ],
        ] {
            let mut args = base.clone();
            args.extend(bad.into_iter().map(OsString::from));
            assert!(parse(args).is_err());
        }
        Ok(())
    }

    #[test]
    fn native_child_verified_sections_and_calibrated_cow_policy_enforce_contracts() -> io::Result<()>
    {
        use bonsai_prism_runtime::swift_session::{
            calibrated_cow_execution_memory_bytes, selected_state_budget,
        };
        let (directory, [model, key, _adapter]) = files()?;
        let mmproj = directory.path().join("mmproj.gguf");
        std::fs::write(&mmproj, [])?;

        let pipe_base: Vec<OsString> = vec![
            "--native-engine-child".into(),
            "--broker-pipe".into(),
            "--model".into(),
            model.clone(),
            "--seed".into(),
            "42".into(),
        ];

        let mut sec1 = pipe_base.clone();
        sec1.extend(["--verified-sections", "1"].map(OsString::from));
        assert_eq!(parse(sec1.clone())?.verified_sections, Some(1));
        sec1.extend(["--verified-sections", "1"].map(OsString::from));
        assert!(parse(sec1).is_err());

        let mut sec2 = pipe_base.clone();
        sec2.extend([
            "--mmproj".into(),
            mmproj.into_os_string(),
            "--verified-sections".into(),
            "2".into(),
        ]);
        assert_eq!(parse(sec2.clone())?.verified_sections, Some(2));
        // Wrong count against 2 artifacts must fail.
        sec2.pop();
        sec2.push("3".into());
        assert!(parse(sec2).is_err());

        for bad_count in ["0", "9", "abc"] {
            let mut bad = pipe_base.clone();
            bad.extend(["--verified-sections", bad_count].map(OsString::from));
            assert!(parse(bad).is_err());
        }
        let loopback_verified = vec![
            OsString::from("--native-engine-child"),
            "--model".into(),
            model,
            "--key-file".into(),
            key,
            "--port".into(),
            "8081".into(),
            "--seed".into(),
            "42".into(),
            "--verified-sections".into(),
            "1".into(),
        ];
        assert!(parse(loopback_verified).is_err());

        // Calibrated CoW policy admits calibrated PTQ1 + shared KV budgets that conservative mode rejects.
        let context = 262_144_usize;
        let ptq1_cow_mem = calibrated_cow_execution_memory_bytes(
            context,
            Representation::Bonsai25Ptq1,
            None,
            None,
            KvCachePolicy::Rotated4,
            PromptReuse::Latest,
            BranchCheckpoints::default(),
        )
        .map_err(io::Error::other)?;
        let mut calibrated = pipe_base.clone();
        calibrated.extend([
            "--bonsai-2-5".into(),
            "ptq1".into(),
            "--kv-cache".into(),
            "rsq4".into(),
            "--prompt-reuse".into(),
            "latest".into(),
            "--context-tokens".into(),
            context.to_string().into(),
            "--memory-limit-bytes".into(),
            ptq1_cow_mem.to_string().into(),
            "--admission-policy".into(),
            "calibrated-cow".into(),
        ]);
        let parsed_cow = parse(calibrated.clone())?;
        assert_eq!(parsed_cow.admission_policy, AdmissionPolicy::CalibratedCow);
        assert_eq!(parsed_cow.capacity, Some((context, ptq1_cow_mem)));

        let mut below_cow = calibrated.clone();
        below_cow[15] = (ptq1_cow_mem - 1).to_string().into();
        assert!(parse(below_cow).is_err());

        let mut conservative = calibrated.clone();
        conservative[17] = "conservative".into();
        assert!(parse(conservative).is_err());

        // Missing capacity with calibrated-cow or unknown policy string must fail.
        let mut missing_cap = pipe_base;
        missing_cap.extend(["--admission-policy", "calibrated-cow"].map(OsString::from));
        assert!(parse(missing_cap).is_err());
        let mut dup_policy = calibrated.clone();
        dup_policy.extend(["--admission-policy", "calibrated-cow"].map(OsString::from));
        assert!(parse(dup_policy).is_err());
        let mut unknown_policy = calibrated;
        unknown_policy[17] = "auto".into();
        assert!(parse(unknown_policy).is_err());

        // Ensure spec checkpoint helper aligns with selected_state_budget.
        assert!(
            selected_state_budget(context, KvCachePolicy::Rotated4).map_err(io::Error::other)?
                > bonsai_prism_runtime::swift_session::STATE_MEMORY_BYTES
        );
        Ok(())
    }

    #[test]
    fn native_child_daisyrose_lora_verified_sections_and_partial_flag_rejection() -> io::Result<()>
    {
        let (directory, [model, _key, _adapter]) = files()?;
        let mmproj = directory.path().join("mmproj.gguf");
        std::fs::write(&mmproj, [])?;
        let daisy = directory.path().join("daisyrose.gguf");
        std::fs::write(&daisy, b"placeholder")?;
        let daisy_hash = "ab".repeat(32);

        let pipe_base: Vec<OsString> = vec![
            "--native-engine-child".into(),
            "--broker-pipe".into(),
            "--model".into(),
            model,
            "--seed".into(),
            "42".into(),
        ];

        let mut sec3 = pipe_base.clone();
        sec3.extend([
            "--daisyrose-lora".into(),
            daisy.clone().into_os_string(),
            "--daisyrose-lora-bytes".into(),
            "4096".into(),
            "--daisyrose-lora-sha256".into(),
            daisy_hash.into(),
            "--mmproj".into(),
            mmproj.into_os_string(),
            "--verified-sections".into(),
            "3".into(),
        ]);
        let parsed_sec3 = parse(sec3.clone())?;
        assert_eq!(parsed_sec3.verified_sections, Some(3));
        let expected_sec3 = expected_verified_sections(&parsed_sec3)?;
        assert_eq!(expected_sec3.len(), 3);
        assert_eq!(expected_sec3[1].0, 1);
        assert_eq!(expected_sec3[1].1, 4096);
        assert_eq!(expected_sec3[2].0, 2);

        let last_idx = sec3.len() - 1;
        sec3[last_idx] = "2".into();
        assert!(parse(sec3).is_err());

        let mut partial = pipe_base;
        partial.extend([
            "--daisyrose-lora".into(),
            daisy.into_os_string(),
            "--daisyrose-lora-bytes".into(),
            "4096".into(),
        ]);
        assert!(parse(partial).is_err());
        Ok(())
    }

    #[test]
    fn native_child_hexagon_arena_cache_flag_parsing_and_validation() -> io::Result<()> {
        let (directory, [model, _key, _adapter]) = files()?;
        let base: Vec<OsString> = vec![
            "--native-engine-child".into(),
            "--broker-pipe".into(),
            "--model".into(),
            model,
            "--seed".into(),
            "42".into(),
        ];

        assert_eq!(parse(base.clone())?.hexagon_arena_cache, None);

        let existing_dir = directory.path().join("arena-cache-existing");
        std::fs::create_dir(&existing_dir)?;
        let mut with_existing = base.clone();
        with_existing.extend(["--hexagon-arena-cache".into(), existing_dir.clone().into()]);
        assert_eq!(
            parse(with_existing.clone())?.hexagon_arena_cache,
            Some(existing_dir)
        );

        let leaf_dir = directory.path().join("arena-cache-leaf");
        let mut with_leaf = base.clone();
        with_leaf.extend(["--hexagon-arena-cache".into(), leaf_dir.clone().into()]);
        assert_eq!(parse(with_leaf)?.hexagon_arena_cache, Some(leaf_dir));

        let mut duplicate = with_existing;
        duplicate.extend([
            "--hexagon-arena-cache".into(),
            directory.path().join("other").into(),
        ]);
        assert!(parse(duplicate).is_err());

        let mut missing_value = base.clone();
        missing_value.push("--hexagon-arena-cache".into());
        assert!(parse(missing_value).is_err());

        let existing_file = directory.path().join("not-a-directory");
        std::fs::write(&existing_file, b"file")?;
        let mut with_file = base.clone();
        with_file.extend(["--hexagon-arena-cache".into(), existing_file.into()]);
        assert!(parse(with_file).is_err());

        let mut relative = base;
        relative.extend(["--hexagon-arena-cache".into(), "relative-cache".into()]);
        assert!(parse(relative).is_err());
        Ok(())
    }

    #[test]
    fn native_child_vulkan_placement_defaults_and_accepts_f16_configurations() -> io::Result<()> {
        use bonsai_prism_runtime::swift_session::{CONTEXT_TOKEN_MEMORY_BYTES, STATE_MEMORY_BYTES};

        let (_directory, [model, key, _adapter]) = files()?;
        let base: Vec<OsString> = vec![
            "--native-engine-child".into(),
            "--model".into(),
            model,
            "--key-file".into(),
            key,
            "--port".into(),
            "8080".into(),
            "--seed".into(),
            "42".into(),
            "--placement".into(),
            "vulkan".into(),
        ];

        let default_cfg = parse(base.clone())?;
        assert_eq!(default_cfg.placement, Placement::Vulkan);
        assert_eq!(default_cfg.kv_cache, KvCachePolicy::F16);
        assert!(default_cfg.mmproj.is_none());
        assert!(default_cfg.cpu_prefill.is_none());
        assert!(default_cfg.cpu_speculation.is_none());

        let mut explicit_f16 = base.clone();
        explicit_f16.extend(["--kv-cache".into(), "f16".into()]);
        assert_eq!(parse(explicit_f16)?.kv_cache, KvCachePolicy::F16);

        let f16_mem = NON_CONTEXT_MEMORY_BYTES + 8_192 * CONTEXT_TOKEN_MEMORY_BYTES;
        let state_u64 = u64::try_from(STATE_MEMORY_BYTES).map_err(io::Error::other)?;
        let prompt_end_mem = f16_mem + state_u64;
        let mut capacity = base.clone();
        capacity.extend([
            "--context-tokens".into(),
            "8192".into(),
            "--memory-limit-bytes".into(),
            f16_mem.to_string().into(),
        ]);
        let capacity_cfg = parse(capacity)?;
        assert_eq!(capacity_cfg.kv_cache, KvCachePolicy::F16);
        assert_eq!(capacity_cfg.capacity, Some((8192, f16_mem)));

        let mut variant = base;
        variant.extend([
            "--bonsai-2-5".into(),
            "ptq1".into(),
            "--prompt-reuse".into(),
            "prompt-end-and-latest".into(),
            "--context-tokens".into(),
            "8192".into(),
            "--memory-limit-bytes".into(),
            prompt_end_mem.to_string().into(),
        ]);
        let variant_cfg = parse(variant)?;
        assert_eq!(variant_cfg.placement, Placement::Vulkan);
        assert_eq!(variant_cfg.representation, Representation::Bonsai25Ptq1);
        assert_eq!(variant_cfg.prompt_reuse, PromptReuse::PromptEndAndLatest);
        Ok(())
    }

    fn vulkan_cpu_only_cases(
        prefill_scratch: &str,
        verify_scratch: &str,
        checkpoint: &str,
        memory: &str,
    ) -> Vec<Vec<String>> {
        let cap = [
            "--kv-cache",
            "f16",
            "--context-tokens",
            "8192",
            "--memory-limit-bytes",
            memory,
        ];
        let prefill = [
            "--cpu-prefill-tokens",
            "16",
            "--cpu-prefill-scratch-bytes",
            prefill_scratch,
        ];
        let mut suffix_case: Vec<&str> = Vec::from(cap);
        suffix_case.extend(prefill);
        suffix_case.extend([
            "--spec-type",
            "suffix",
            "--cpu-verification-scratch-bytes",
            verify_scratch,
            "--cpu-speculative-checkpoint-bytes",
            checkpoint,
        ]);
        let mut spec_tokens_case: Vec<&str> = Vec::from(cap);
        spec_tokens_case.extend(prefill);
        spec_tokens_case.extend([
            "--cpu-speculative-tokens",
            "4",
            "--cpu-verification-scratch-bytes",
            verify_scratch,
            "--cpu-speculative-checkpoint-bytes",
            checkpoint,
        ]);
        let mut gdn_case: Vec<&str> = Vec::from(cap);
        gdn_case.extend(["--prompt-reuse", "latest"]);
        gdn_case.extend(prefill);
        gdn_case.extend(["--cpu-prefill-gdn", "wy-f32"]);
        let mut prefill_only_case: Vec<&str> = Vec::from(cap);
        prefill_only_case.extend(["--prompt-reuse", "latest"]);
        prefill_only_case.extend(prefill);
        [
            vec!["--kv-cache", "rsq4"],
            vec!["--kv-cache", "rsq4vf"],
            vec!["--kv-cache", "rsq4kvf"],
            vec!["--kv-cache", "rsq4k3v"],
            suffix_case,
            spec_tokens_case,
            gdn_case,
            prefill_only_case,
        ]
        .into_iter()
        .map(|case| case.into_iter().map(str::to_owned).collect())
        .collect()
    }

    #[test]
    fn native_child_vulkan_placement_rejects_unsupported_flags() -> io::Result<()> {
        let (directory, [model, key, _adapter]) = files()?;
        let mmproj = directory.path().join("mmproj.gguf");
        std::fs::write(&mmproj, [])?;
        let prefill_scratch = (64_usize * 1024 * 1024).to_string();
        let verify_scratch = (16_usize * 1024 * 1024).to_string();
        let checkpoint = bonsai_prism_runtime::swift_session::STATE_MEMORY_BYTES.to_string();
        let base = |placement: &str| -> Vec<OsString> {
            vec![
                "--native-engine-child".into(),
                "--model".into(),
                model.clone(),
                "--key-file".into(),
                key.clone(),
                "--port".into(),
                "8080".into(),
                "--seed".into(),
                "42".into(),
                "--placement".into(),
                placement.into(),
            ]
        };

        for extra in vulkan_cpu_only_cases(
            &prefill_scratch,
            &verify_scratch,
            &checkpoint,
            "68719476736",
        ) {
            let mut cpu_args = base("cpu");
            cpu_args.extend(extra.iter().map(OsString::from));
            assert!(parse(cpu_args).is_ok());

            let mut vk_args = base("vulkan");
            vk_args.extend(extra.into_iter().map(OsString::from));
            assert!(parse(vk_args).is_err());
        }

        for incomplete in [
            vec!["--spec-type", "suffix"],
            vec!["--cpu-speculative-tokens", "4"],
            vec!["--cpu-prefill-gdn", "wy-f32"],
        ] {
            let mut vk_args = base("vulkan");
            vk_args.extend(incomplete.into_iter().map(OsString::from));
            assert!(parse(vk_args).is_err());
        }

        let mut cpu_vision = base("cpu");
        cpu_vision.extend(["--mmproj".into(), mmproj.as_os_str().to_owned()]);
        assert!(parse(cpu_vision).is_ok());

        let mut vk_vision = base("vulkan");
        vk_vision.extend(["--mmproj".into(), mmproj.into_os_string()]);
        assert!(parse(vk_vision).is_err());
        Ok(())
    }

    #[test]
    fn native_child_prefill_row_workers_and_64_tile_tokens_cli_admission() -> io::Result<()> {
        use bonsai_prism_runtime::swift_session::{
            CONTEXT_TOKEN_MEMORY_BYTES, NON_CONTEXT_MEMORY_BYTES,
        };
        let (_directory, [model, key, _adapter]) = files()?;
        let base: Vec<OsString> = vec![
            "--native-engine-child".into(),
            "--model".into(),
            model,
            "--key-file".into(),
            key,
            "--port".into(),
            "8080".into(),
            "--seed".into(),
            "42".into(),
        ];

        let default_cfg = parse(base.clone())?;
        assert_eq!(default_cfg.row_workers, 1);
        assert_eq!(default_cfg.prefill_row_workers, 1);

        let mut single_override = base.clone();
        single_override.extend(["--prefill-row-workers", "1"].map(OsString::from));
        let single_cfg = parse(single_override.clone())?;
        assert_eq!(single_cfg.row_workers, 1);
        assert_eq!(single_cfg.prefill_row_workers, 1);

        let mut duplicate = single_override;
        duplicate.extend(["--prefill-row-workers", "1"].map(OsString::from));
        assert!(parse(duplicate).is_err());

        let mut x1_args = base.clone();
        x1_args.extend(["--cpu-family", "x1"].map(OsString::from));
        #[cfg(feature = "parallel-rows")]
        {
            let x1_cfg = parse(x1_args.clone())?;
            assert_eq!(x1_cfg.row_workers, 8);
            assert_eq!(x1_cfg.prefill_row_workers, 8);

            let mut override_args = x1_args;
            override_args.extend(["--prefill-row-workers", "16"].map(OsString::from));
            let override_cfg = parse(override_args.clone())?;
            assert_eq!(override_cfg.row_workers, 8);
            assert_eq!(override_cfg.prefill_row_workers, 16);

            let mut dup_multi = override_args;
            dup_multi.extend(["--prefill-row-workers", "12"].map(OsString::from));
            assert!(parse(dup_multi).is_err());
        }
        #[cfg(not(feature = "parallel-rows"))]
        {
            assert!(parse(x1_args).is_err());
            let mut multi_prefill = base.clone();
            multi_prefill.extend(["--prefill-row-workers", "16"].map(OsString::from));
            assert!(parse(multi_prefill).is_err());
        }
        for bad in ["0", &(MAX_ROW_WORKERS + 1).to_string()] {
            let mut bad_args = base.clone();
            bad_args.extend(["--prefill-row-workers", bad].map(OsString::from));
            assert!(parse(bad_args).is_err());
        }

        // `--cpu-prefill-tokens 64` parses cleanly alongside paired KV/context/memory admission.
        let reserve = 64 * 1024 * 1024;
        let memory = NON_CONTEXT_MEMORY_BYTES + 8_192 * CONTEXT_TOKEN_MEMORY_BYTES + reserve;
        let mut tile64 = base;
        tile64.extend(
            [
                "--kv-cache",
                "rsq4",
                "--context-tokens",
                "8192",
                "--memory-limit-bytes",
                &memory.to_string(),
                "--cpu-prefill-tokens",
                "64",
                "--cpu-prefill-scratch-bytes",
                &reserve.to_string(),
            ]
            .map(OsString::from),
        );
        let tile64_cfg = parse(tile64.clone())?;
        assert_eq!(tile64_cfg.requested_prefill_tokens, Some(64));
        assert!(tile64_cfg.cpu_prefill.is_some());

        // Zero scratch still fails even with `--cpu-prefill-tokens 64`.
        let last = tile64.len() - 1;
        tile64[last] = "0".into();
        assert!(parse(tile64).is_err());
        Ok(())
    }
}
