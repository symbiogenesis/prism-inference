//! Ordered Hugging Face model, shard and speculative-sidecar resolution.
//!
//! Ports the selection policy in pinned `common/download.cpp`. Metadata comes
//! from callers or a cache/HTTP owner, never from model-specific registries.

#![allow(
    clippy::struct_excessive_bools,
    clippy::case_sensitive_file_extension_comparisons,
    reason = "Independent native discovery flags and case-sensitive GGUF filename policy."
)]

use crate::{Result, invalid};
use regex::RegexBuilder;
use std::path::PathBuf;

/// One remote file and its independent cache destinations.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct File {
    /// Repository-relative path.
    pub path: String,
    /// Resolved immutable download URL, empty for an offline listing.
    pub url: String,
    /// Blob or snapshot path receiving downloaded bytes.
    pub local_path: PathBuf,
    /// Snapshot path loaded by callers.
    pub final_path: PathBuf,
    /// Optional Git/LFS object hash.
    pub oid: String,
    /// Repository owner/name.
    pub repo: String,
}

/// Local/URL/HF model arguments shared by model and draft handlers.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Model {
    /// Local filename.
    pub path: String,
    /// Explicit download URL.
    pub url: String,
    /// Owner/repository and optional quantization tag.
    pub hf_repo: String,
    /// Exact repository-relative filename override.
    pub hf_file: String,
}
impl Model {
    /// Whether neither a local model nor HF repository was provided.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.path.is_empty() && self.hf_repo.is_empty()
    }
}

/// Requested optional companions and online/cache policy.
#[derive(Clone, Debug, Default)]
pub struct Options {
    /// Discover multimodal projection weights.
    pub mmproj: bool,
    /// Discover an MTP head.
    pub mtp: bool,
    /// Discover a `DFlash` draft.
    pub dflash: bool,
    /// Discover an Eagle3 draft.
    pub eagle3: bool,
    /// Discover a `DSpark` draft.
    pub dspark: bool,
    /// Use cached metadata exclusively.
    pub offline: bool,
    /// Optional Hugging Face bearer credential.
    pub token: String,
}

/// Selected primary shard, ordered shard inventory and independent companions.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Plan {
    /// Primary model, always first shard for automatically selected models.
    pub primary: File,
    /// All matching shards in repository listing order.
    pub model_files: Vec<File>,
    /// Multimodal projector.
    pub mmproj: File,
    /// MTP head.
    pub mtp: File,
    /// `DFlash` draft.
    pub dflash: File,
    /// Eagle3 draft.
    pub eagle3: File,
    /// `DSpark` draft.
    pub dspark: File,
    /// Root preset, which takes precedence over model files.
    pub preset: File,
}

/// Separate repository and optional quantization suffix.
///
/// # Errors
/// Rejects identifiers without exactly two slash-separated components.
pub fn split_repo_tag(value: &str) -> Result<(&str, &str)> {
    let parts: Vec<_> = value.split(':').collect();
    let repo = parts[0];
    if repo.split('/').count() != 2 {
        return Err(invalid("HF repository requires owner/model[:quant]"));
    }
    Ok((
        repo,
        if parts.len() > 1 {
            parts[parts.len() - 1]
        } else {
            ""
        },
    ))
}

#[derive(Default)]
struct Split<'a> {
    prefix: &'a str,
    tag: String,
    index: u32,
    count: u32,
}
fn split(path: &str) -> Split<'_> {
    let Some(mut prefix) = path.strip_suffix(".gguf") else {
        return Split::default();
    };
    let mut index = 1;
    let mut count = 1;
    if prefix.len() >= 16 {
        let start = prefix.len() - 15;
        let tail = &prefix.as_bytes()[start..];
        if start > 0
            && tail[0] == b'-'
            && tail[1..6].iter().all(u8::is_ascii_digit)
            && tail[6..10].eq_ignore_ascii_case(b"-of-")
            && tail[10..15].iter().all(u8::is_ascii_digit)
        {
            index = prefix[start + 1..start + 6].parse().unwrap_or(0);
            count = prefix[start + 10..].parse().unwrap_or(0);
            prefix = &prefix[..start];
        }
    }
    let tag = prefix.rsplit_once(['-', '.']).map_or("", |(_, tag)| tag);
    let tag = if !tag.is_empty() && tag.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
        tag.to_ascii_uppercase()
    } else {
        String::new()
    };
    Split {
        prefix,
        tag,
        index,
        count,
    }
}
fn bits(tag: &str) -> i32 {
    let Some(pos) = tag.find(|c: char| c.is_ascii_digit()) else {
        return 0;
    };
    tag[pos..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .unwrap_or(0)
}
fn is_model(path: &str) -> bool {
    let basename = path.rsplit('/').next().unwrap_or(path);
    path.ends_with(".gguf")
        && ["mmproj", "imatrix", "mtp-", "eagle3-", "dflash-", "dspark-"]
            .iter()
            .all(|word| !basename.contains(word))
}
fn find_model(files: &[File], tag: &str) -> Result<File> {
    let tags = if tag.is_empty() {
        vec!["Q4_K_M", "Q8_0"]
    } else {
        vec![tag]
    };
    for tag in tags {
        let pattern = RegexBuilder::new(&format!("{tag}[.-]"))
            .case_insensitive(true)
            .build()?;
        for file in files {
            let info = split(&file.path);
            if is_model(&file.path)
                && pattern.is_match(&file.path)
                && (info.count <= 1 || info.index == 1)
            {
                return Ok(file.clone());
            }
        }
    }
    Ok(if tag.is_empty() {
        files
            .iter()
            .find(|f| {
                let info = split(&f.path);
                is_model(&f.path) && (info.count <= 1 || info.index == 1)
            })
            .cloned()
            .unwrap_or_default()
    } else {
        File::default()
    })
}
fn sibling(files: &[File], model: &str, keyword: &str, tag: &str) -> File {
    let tag = tag.to_ascii_uppercase();
    let model_tag = split(model).tag;
    let model_bits = bits(if tag.is_empty() { &model_tag } else { &tag });
    let model_dir: Vec<_> = model.split('/').collect();
    let model_dir = &model_dir[..model_dir.len() - 1];
    let mut best: Option<(&File, usize, bool, i32)> = None;
    for file in files {
        if !file.path.ends_with(".gguf") || !file.path.contains(keyword) {
            continue;
        }
        let parts: Vec<_> = file.path.split('/').collect();
        let dir = &parts[..parts.len() - 1];
        if !model_dir.starts_with(dir) {
            continue;
        }
        let depth = dir.len();
        let exact = !tag.is_empty() && file.path.to_ascii_uppercase().contains(&format!("-{tag}."));
        let diff = (bits(&split(&file.path).tag) - model_bits).abs();
        if best.is_none_or(|(_, bd, be, bf)| {
            depth > bd || (depth == bd && exact && !be) || (depth == bd && exact == be && diff < bf)
        }) {
            best = Some((file, depth, exact, diff));
        }
    }
    best.map_or_else(File::default, |(file, _, _, _)| file.clone())
}

/// Select a plan from arbitrary ordered remote or cached descriptors.
///
/// # Errors
/// Rejects malformed repository/tag syntax or invalid quantization regular expressions.
pub fn select(files: &[File], model: &Model, options: &Options) -> Result<Plan> {
    let (_, tag) = split_repo_tag(&model.hf_repo)?;
    let mut plan = Plan::default();
    if let Some(preset) = files.iter().find(|f| f.path == "preset.ini") {
        plan.preset = preset.clone();
        return Ok(plan);
    }
    let primary = if model.hf_file.is_empty() {
        find_model(files, tag)?
    } else {
        let Some(file) = files.iter().find(|f| f.path == model.hf_file) else {
            return Ok(plan);
        };
        file.clone()
    };
    if primary.path.is_empty()
        && !(options.mtp || options.dflash || options.eagle3 || options.dspark)
    {
        return Ok(plan);
    }
    if !primary.path.is_empty() {
        let info = split(&primary.path);
        plan.model_files = if info.count <= 1 {
            vec![primary.clone()]
        } else {
            files
                .iter()
                .filter(|f| {
                    let other = split(&f.path);
                    other.count == info.count && other.prefix == info.prefix
                })
                .cloned()
                .collect()
        };
        plan.primary = primary.clone();
        if options.mmproj {
            plan.mmproj = sibling(files, &primary.path, "mmproj", "");
        }
    }
    if options.mtp {
        plan.mtp = sibling(files, &primary.path, "mtp-", tag);
    }
    if options.dflash {
        plan.dflash = sibling(files, &primary.path, "dflash-", tag);
    }
    if options.eagle3 {
        plan.eagle3 = sibling(files, &primary.path, "eagle3-", tag);
    }
    if options.dspark {
        plan.dspark = sibling(files, &primary.path, "dspark-", tag);
    }
    Ok(plan)
}

#[path = "resolution/cache.rs"]
mod cache;
#[path = "resolution/params.rs"]
mod params;
pub use cache::Resolver;
pub use params::{Handler, Params, SpeculativeType};

#[cfg(test)]
mod tests {
    use super::*;
    fn files(paths: &[&str]) -> Vec<File> {
        paths
            .iter()
            .map(|path| File {
                path: (*path).into(),
                ..File::default()
            })
            .collect()
    }
    #[test]
    fn shard_order_and_ancestor_sidecars_follow_metadata() -> Result<()> {
        let source = files(&[
            "x/a-Q8_0-00002-of-00002.gguf",
            "mtp-a-Q8_0.gguf",
            "x/mtp-a-F16.gguf",
            "x/a-Q8_0-00001-of-00002.gguf",
            "x/y/mtp-a-Q8_0.gguf",
        ]);
        let model = Model {
            hf_repo: "owner/repo:Q8_0".into(),
            ..Model::default()
        };
        let plan = select(
            &source,
            &model,
            &Options {
                mtp: true,
                ..Options::default()
            },
        )?;
        assert_eq!(plan.primary.path, "x/a-Q8_0-00001-of-00002.gguf");
        assert_eq!(plan.model_files.len(), 2);
        assert_eq!(plan.model_files[0].path, source[0].path);
        assert_eq!(plan.mtp.path, "x/mtp-a-F16.gguf");
        Ok(())
    }
    #[test]
    fn explicit_missing_filename_prevents_sidecar_fallback() -> Result<()> {
        let source = files(&["mtp-a-Q8_0.gguf"]);
        let mut model = Model {
            hf_repo: "o/r:Q8_0".into(),
            ..Model::default()
        };
        let options = Options {
            mtp: true,
            ..Options::default()
        };
        assert_eq!(select(&source, &model, &options)?.mtp.path, source[0].path);
        model.hf_file = "missing.gguf".into();
        assert_eq!(select(&source, &model, &options)?, Plan::default());
        Ok(())
    }
}
