//! Thin transport for the source's owned Rust loading and cancellation calls.

use bonsai_prism_runtime::backend::CpuBackend;
use bonsai_prism_runtime::model::Model;
use serde_json::{Value, json};
use std::path::Path;

use super::super::llama_grammar::exact_fields;
use super::{ASSET_LENGTH, ASSET_SHA, ProgressPolicy, Reply, Result, invalid};

const MEMORY_LIMIT: usize = 256 * 1024 * 1024;

#[derive(Default)]
pub(super) struct Client {
    backend: Option<CpuBackend>,
    model: Option<Model>,
}

impl Client {
    pub(super) fn call(
        &mut self,
        worker: u32,
        operation: &Value,
        progress: Option<ProgressPolicy>,
    ) -> Result<Reply> {
        if worker != 0 {
            return Err(invalid("Rust storage adapter only owns worker zero"));
        }
        match operation["Op"].as_str() {
            Some("backend-init") => {
                exact_fields(operation, &["Op"])?;
                if self.backend.is_some() {
                    return Err(invalid("Rust storage backend already initialized"));
                }
                self.backend = Some(CpuBackend::new());
            }
            Some("backend-free") => {
                exact_fields(operation, &["Op"])?;
                if self.backend.take().is_none() {
                    return Err(invalid("Rust storage backend is not initialized"));
                }
                self.model = None;
            }
            Some("model-load") => return self.load(operation, progress),
            _ => return Err(invalid("Rust storage operation is unavailable")),
        }
        Ok(Reply {
            value: json!({}),
            progress: Vec::new(),
        })
    }

    fn load(&mut self, operation: &Value, policy: Option<ProgressPolicy>) -> Result<Reply> {
        exact_fields(
            operation,
            &["Op", "Path", "Base", "Params", "Overrides", "Progress"],
        )?;
        if self.backend.is_none()
            || self.model.is_some()
            || operation["Base"] != "zero"
            || !operation["Params"].is_null()
            || operation["Overrides"] != json!({"LoadMode":"none"})
            || operation["Progress"] != true
        {
            return Err(invalid("Rust storage load configuration is unavailable"));
        }
        let path = Path::new(
            operation["Path"]
                .as_str()
                .ok_or_else(|| invalid("Rust storage model path must be text"))?,
        );
        if !path.is_absolute() {
            return Err(invalid("Rust storage model path must be absolute"));
        }
        let policy = policy.ok_or_else(|| invalid("Rust storage progress policy missing"))?;
        let mut observations = Vec::new();
        let mut cancelled = false;
        let loaded = Model::load_with_progress(
            path,
            ASSET_LENGTH,
            ASSET_SHA,
            MEMORY_LIMIT,
            &mut |progress| {
                let decision = policy(progress);
                observations.push((progress.to_bits(), decision));
                cancelled |= !decision;
                Ok(decision)
            },
        );
        let value = match loaded {
            Ok(model) => {
                if cancelled {
                    return Err(invalid("Rust loader published a cancelled model"));
                }
                self.model = Some(model);
                json!({"Model":1})
            }
            Err(_) if cancelled => json!({"Model":Value::Null}),
            Err(error) => return Err(error),
        };
        Ok(Reply {
            value,
            progress: observations,
        })
    }

    pub(super) fn finish(self) -> Result<()> {
        if self.backend.is_some() || self.model.is_some() {
            return Err(invalid("Rust storage adapter retained an unclosed backend"));
        }
        Ok(())
    }
}
