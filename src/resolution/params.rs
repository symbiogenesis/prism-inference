//! Model-related argument handlers and deterministic assembly precedence.

use super::{File, Model, Options, Plan, Resolver};
use crate::model::gguf_context::Context;
use crate::{Result, invalid};
use std::path::Path;

/// Native speculative decoding policy identifiers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum SpeculativeType {
    /// No speculative decoding policy.
    None = 0,
    /// Independent draft model.
    DraftSimple = 1,
    /// Eagle3 draft.
    DraftEagle3 = 2,
    /// MTP head.
    DraftMtp = 3,
    /// `DFlash` draft.
    DraftDflash = 4,
    /// `DSpark` draft with a Markov head.
    DraftDspark = 5,
    /// Simple ngram speculation.
    NgramSimple = 6,
    /// Ngram map with keys.
    NgramMapK = 7,
    /// Ngram map with four values.
    NgramMapK4v = 8,
    /// Modified ngram strategy.
    NgramMod = 9,
    /// Cached ngram strategy.
    NgramCache = 10,
}
impl SpeculativeType {
    fn parse(value: &str) -> Result<Self> {
        Ok(match value {
            "none" => Self::None,
            "draft-simple" => Self::DraftSimple,
            "draft-eagle3" => Self::DraftEagle3,
            "draft-mtp" => Self::DraftMtp,
            "draft-dflash" => Self::DraftDflash,
            "draft-dspark" => Self::DraftDspark,
            "ngram-simple" => Self::NgramSimple,
            "ngram-map-k" => Self::NgramMapK,
            "ngram-map-k4v" => Self::NgramMapK4v,
            "ngram-mod" => Self::NgramMod,
            "ngram-cache" => Self::NgramCache,
            _ => return Err(invalid(format!("unknown speculative type {value}"))),
        })
    }
}

/// Model-related portion of ordinary CLI parameters, reusable by a wider parser.
#[derive(Clone, Debug)]
pub struct Params {
    /// Main model parameters.
    pub model: Model,
    /// Multimodal projector parameters.
    pub mmproj: Model,
    /// Draft model parameters.
    pub draft: Model,
    /// Ordered requested speculative policies.
    pub spec_types: Vec<SpeculativeType>,
    /// Disable automatic multimodal projector discovery.
    pub no_mmproj: bool,
    /// Suppress network access and actual downloads.
    pub offline: bool,
    /// Optional bearer credential.
    pub hf_token: String,
    /// Selected preset path for server router mode.
    pub preset: String,
}
impl Default for Params {
    fn default() -> Self {
        Self {
            model: Model::default(),
            mmproj: Model::default(),
            draft: Model::default(),
            spec_types: vec![SpeculativeType::None],
            no_mmproj: false,
            offline: false,
            hf_token: String::new(),
            preset: String::new(),
        }
    }
}
impl Params {
    /// Apply one model-related option to this owned parameter set.
    ///
    /// Returns false for an option belonging to another parser section.
    ///
    /// # Errors
    /// Rejects missing values, malformed speculative names and NUL text.
    pub fn apply_option(&mut self, option: &str, value: Option<&str>) -> Result<bool> {
        if option.contains('\0') || value.is_some_and(|v| v.contains('\0')) {
            return Err(invalid("NUL argument"));
        }
        let required = || value.ok_or_else(|| invalid(format!("option {option} requires a value")));
        match option {
            "-m" | "--model" => self.model.path = required()?.into(),
            "-mu" | "--model-url" => self.model.url = required()?.into(),
            "-hf" | "-hfr" | "--hf-repo" => self.model.hf_repo = required()?.into(),
            "-hff" | "--hf-file" => self.model.hf_file = required()?.into(),
            "-hft" | "--hf-token" => self.hf_token = required()?.into(),
            "--spec-draft-hf" | "-hfd" | "-hfrd" | "--hf-repo-draft" => {
                self.draft.hf_repo = required()?.into();
            }
            "--spec-draft-model" | "-md" | "--model-draft" => {
                self.draft.path = required()?.into();
                self.draft.hf_file.clone_from(&self.draft.path);
            }
            "--mmproj" => self.mmproj.path = required()?.into(),
            "--mmproj-url" => self.mmproj.url = required()?.into(),
            "--no-mmproj" => self.no_mmproj = true,
            "--offline" => self.offline = true,
            "--spec-type" => {
                let mut types = Vec::new();
                for name in required()?.split(',') {
                    let kind = SpeculativeType::parse(name)?;
                    if kind == SpeculativeType::None {
                        types = vec![kind];
                        break;
                    }
                    types.push(kind);
                }
                self.spec_types.extend(types);
            }
            "--mtp" => self.spec_types.push(SpeculativeType::DraftMtp),
            "--dflash" => self.spec_types.push(SpeculativeType::DraftDflash),
            "--eagle3" => self.spec_types.push(SpeculativeType::DraftEagle3),
            _ => return Ok(false),
        }
        Ok(true)
    }
    /// Parse an argument vector consisting of model-resolution options.
    ///
    /// The first item is the executable name. Other CLI sections can call
    /// [`Self::apply_option`] from their own ordinary option dispatch.
    ///
    /// # Errors
    /// Rejects unknown options, missing values and argument extent violations.
    pub fn parse(arguments: &[String]) -> Result<Self> {
        if arguments.is_empty()
            || arguments.len() > 4096
            || arguments.iter().map(String::len).sum::<usize>() > 1024 * 1024
        {
            return Err(invalid("model argument extent outside bound"));
        }
        let mut params = Self::default();
        let mut at = 1;
        while at < arguments.len() {
            let argument = if arguments[at].starts_with("--") {
                arguments[at].replace('_', "-")
            } else {
                arguments[at].clone()
            };
            let flag = matches!(
                argument.as_str(),
                "--no-mmproj" | "--offline" | "--mtp" | "--dflash" | "--eagle3"
            );
            let value = if flag {
                None
            } else {
                arguments.get(at + 1).map(String::as_str)
            };
            if !params.apply_option(&argument, value)? {
                return Err(invalid(format!("unknown model argument {argument}")));
            }
            at += if flag { 1 } else { 2 };
        }
        Ok(params)
    }
    fn default_spec(&self) -> bool {
        self.spec_types == [SpeculativeType::None]
    }
}

/// One independently resolved main/draft pair and their discovery options.
#[derive(Clone, Debug)]
pub struct Handler {
    /// Main plan, updated when applied.
    pub plan: Plan,
    /// Draft plan, with lower-priority sidecars removed during inference.
    pub draft_plan: Plan,
    /// Captured initialization policy.
    pub options: Options,
}
fn metadata_spec(path: &str) -> Option<SpeculativeType> {
    let context = Context::read_file(Path::new(path), false).ok()?;
    let arch = context
        .entries()
        .iter()
        .find(|entry| entry.key == "general.architecture" && entry.kind == 8)?
        .strings
        .first()?;
    let arch = std::str::from_utf8(arch).ok()?;
    if arch == "dflash" {
        return Some(
            if context
                .tensors()
                .iter()
                .any(|t| t.name == "markov_w1.weight")
            {
                SpeculativeType::DraftDspark
            } else {
                SpeculativeType::DraftDflash
            },
        );
    }
    let count = context
        .entries()
        .iter()
        .find(|entry| entry.key == format!("{arch}.block_count") && entry.kind == 4)?
        .data
        .as_slice();
    let count = u32::from_le_bytes(count.try_into().ok()?);
    let name = format!("blk.{}.nextn.eh_proj.weight", count.checked_sub(1)?);
    context
        .tensors()
        .iter()
        .any(|t| t.name == name)
        .then_some(SpeculativeType::DraftMtp)
}
impl Handler {
    /// Discover main and draft plans for the native example identifier.
    ///
    /// # Errors
    /// Rejects invalid example values or malformed repository selection syntax.
    pub fn initialize(resolver: &Resolver, params: &Params, example: u32) -> Result<Self> {
        if example >= 27 {
            return Err(invalid("example identifier outside enum"));
        }
        let options = Options {
            mmproj: matches!(example, 5 | 12 | 15 | 18)
                && !params.no_mmproj
                && params.mmproj.path.is_empty()
                && params.mmproj.url.is_empty(),
            mtp: params.spec_types.contains(&SpeculativeType::DraftMtp),
            dflash: params.spec_types.contains(&SpeculativeType::DraftDflash),
            eagle3: params.spec_types.contains(&SpeculativeType::DraftEagle3),
            dspark: params.spec_types.contains(&SpeculativeType::DraftDspark),
            offline: params.offline,
            token: params.hf_token.clone(),
        };
        let plan = if params.model.hf_repo.is_empty() {
            Plan::default()
        } else {
            resolver.plan(&params.model, &options)?
        };
        let mut draft_options = options.clone();
        if params.default_spec() {
            draft_options.mtp = true;
            draft_options.dflash = true;
            draft_options.eagle3 = true;
            draft_options.dspark = true;
        }
        let draft_plan = if params.draft.hf_repo.is_empty() {
            Plan::default()
        } else {
            resolver.plan(&params.draft, &draft_options)?
        };
        Ok(Self {
            plan,
            draft_plan,
            options,
        })
    }
    fn finalize(resolver: &Resolver, params: &Params, file: &File) -> Result<String> {
        resolver.finalize(file, params.offline, &params.hf_token)
    }
    /// Apply explicit filename, URL, sidecar and preset precedence to parameters.
    ///
    /// Offline application still finalizes paths, matching native callbacks.
    ///
    /// # Errors
    /// Reports failed downloads or filesystem operations.
    pub fn apply(&mut self, resolver: &Resolver, params: &mut Params) -> Result<()> {
        resolver.plain_url(&mut params.model, params.offline, &params.hf_token)?;
        resolver.plain_url(&mut params.mmproj, params.offline, &params.hf_token)?;
        resolver.plain_url(&mut params.draft, params.offline, &params.hf_token)?;
        let mut explicit = !params.draft.url.is_empty();
        let draft = &mut self.draft_plan;
        if !params.draft.hf_file.is_empty() {
            draft.mtp = File::default();
            draft.dflash = File::default();
            draft.eagle3 = File::default();
            draft.dspark = File::default();
        }
        if params.default_spec() {
            if !draft.mtp.path.is_empty() {
                params.spec_types = vec![SpeculativeType::DraftMtp];
                draft.dspark = File::default();
                draft.dflash = File::default();
                draft.eagle3 = File::default();
            } else if !draft.dspark.path.is_empty() {
                params.spec_types = vec![SpeculativeType::DraftDspark];
                draft.dflash = File::default();
                draft.eagle3 = File::default();
            } else if !draft.dflash.path.is_empty() {
                params.spec_types = vec![SpeculativeType::DraftDflash];
                draft.eagle3 = File::default();
            } else if !draft.eagle3.path.is_empty() {
                params.spec_types = vec![SpeculativeType::DraftEagle3];
            }
        }
        if params.default_spec()
            && !params.draft.path.is_empty()
            && let Some(kind) = metadata_spec(&params.draft.path)
        {
            params.spec_types = vec![kind];
        }
        let sidecar = [&draft.mtp, &draft.dflash, &draft.eagle3, &draft.dspark]
            .iter()
            .any(|file| !file.path.is_empty());
        if !explicit {
            for file in [&draft.mtp, &draft.dflash, &draft.eagle3, &draft.dspark] {
                if !file.path.is_empty() {
                    let path = Self::finalize(resolver, params, file)?;
                    if params.draft.path.is_empty() {
                        params.draft.path = path;
                    }
                }
            }
        }
        explicit |= sidecar;
        if !draft.model_files.is_empty() && !explicit {
            for file in &draft.model_files {
                let path = Self::finalize(resolver, params, file)?;
                if file.path == draft.primary.path {
                    params.draft.path = path;
                }
            }
            explicit = true;
        }
        for file in &self.plan.model_files {
            let path = Self::finalize(resolver, params, file)?;
            if file.path == self.plan.primary.path {
                params.model.path = path;
            }
        }
        if !self.plan.mmproj.path.is_empty() {
            params.mmproj.path = Self::finalize(resolver, params, &self.plan.mmproj)?;
        }
        if !explicit {
            for file in [
                &self.plan.mtp,
                &self.plan.dflash,
                &self.plan.eagle3,
                &self.plan.dspark,
            ] {
                if !file.path.is_empty() {
                    let path = Self::finalize(resolver, params, file)?;
                    if params.draft.is_empty() {
                        params.draft.path = path;
                    }
                }
            }
        }
        if !self.plan.preset.path.is_empty() {
            params.preset = Self::finalize(resolver, params, &self.plan.preset)?;
            params.model = Model::default();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn model_options_share_owned_typed_handlers() -> Result<()> {
        let args: Vec<_> = [
            "app",
            "--hf_repo",
            "owner/model:Q8_0",
            "--spec-draft-model",
            "draft.gguf",
            "--spec-type",
            "draft-mtp,draft-dspark",
            "--offline",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        let params = Params::parse(&args)?;
        assert_eq!(params.model.hf_repo, "owner/model:Q8_0");
        assert_eq!(params.draft.hf_file, "draft.gguf");
        assert_eq!(
            params.spec_types,
            [
                SpeculativeType::None,
                SpeculativeType::DraftMtp,
                SpeculativeType::DraftDspark
            ]
        );
        assert!(params.offline);
        assert!(Params::parse(&["app".into(), "--hf-repo".into()]).is_err());
        Ok(())
    }
}
