//! Literal prompts, dataset handling and source scalar oracle for real `DSpark` eval.
#![allow(
    clippy::suboptimal_flops,
    reason = "Retain the original ordered scalar KLD reduction."
)]
use super::{Result, invalid};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

pub(super) const SOURCE_SHA: &str =
    "d420abbcdf9d7c191d0fa1ac2e9859ab9c67e09a3f59e9b79db45875f3dbc7c9";

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "Independent source command-line diagnostic switches."
)]
pub(super) struct Options {
    pub(super) target_path: Option<PathBuf>,
    pub(super) drafter_path: Option<PathBuf>,
    pub(super) prediction_limit: usize,
    pub(super) gpu_layers: i32,
    pub(super) dataset_path: Option<PathBuf>,
    pub(super) prompt_limit: usize,
    pub(super) max_draft: Option<usize>,
    pub(super) oracle_rows: Option<usize>,
    pub(super) same_path: bool,
    pub(super) measure_kld: bool,
    pub(super) split_prefill: bool,
    pub(super) parity_continue: bool,
    pub(super) context: usize,
    pub(super) batch: usize,
    pub(super) shared_head: bool,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            target_path: None,
            drafter_path: None,
            prediction_limit: 96,
            gpu_layers: 999,
            dataset_path: None,
            prompt_limit: 24,
            max_draft: None,
            oracle_rows: None,
            same_path: false,
            measure_kld: false,
            split_prefill: false,
            parity_continue: false,
            context: 4096,
            batch: 2048,
            shared_head: false,
        }
    }
}
impl Options {
    pub(super) fn validate(&self) -> Result<()> {
        if self.target_path.is_some() != self.drafter_path.is_some()
            || !(1..=2048).contains(&self.prediction_limit)
            || !(0..=999).contains(&self.gpu_layers)
            || !(128..=4096).contains(&self.context)
            || !(16..=2048).contains(&self.batch)
            || self.prompt_limit > 256
            || self.oracle_rows.is_some_and(|n| !(1..=9).contains(&n))
            || self.max_draft.is_some_and(|n| !(1..=64).contains(&n))
        {
            return Err(invalid(
                "DSpark options exceed source or bounded transport domains (context128..4096, prediction1..2048, prompts0..256)",
            ));
        }
        for path in [&self.target_path, &self.drafter_path, &self.dataset_path]
            .into_iter()
            .flatten()
        {
            if !path.is_absolute() || !path.is_file() {
                return Err(invalid("DSpark inputs must be existing absolute files"));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
pub(super) struct Prompt {
    pub(super) category: String,
    pub(super) text: String,
    pub(super) messages: Vec<Value>,
}
pub(super) fn prompts(options: &Options) -> Result<Vec<Prompt>> {
    let Some(path) = &options.dataset_path else {
        let value: Value = serde_json::from_str(include_str!("dspark-real-prompts.json"))?;
        if value["SourceSha256"] != SOURCE_SHA
            || value["Prompts"].as_array().map(Vec::len) != Some(24)
        {
            return Err(invalid("DSpark literal prompt provenance differs"));
        }
        // As in the source, n_prompts limits external datasets only.
        return Ok(serde_json::from_value(value["Prompts"].clone())?);
    };
    let mut result = Vec::new();
    let mut bytes = 0_usize;
    let mut reader = BufReader::new(File::open(path)?);
    loop {
        if options.prompt_limit > 0 && result.len() >= options.prompt_limit {
            break;
        }
        let mut line = String::new();
        let count = reader.by_ref().take(65537).read_line(&mut line)?;
        bytes = bytes
            .checked_add(count)
            .ok_or_else(|| invalid("dataset byte count overflow"))?;
        if count == 0 {
            break;
        }
        if count > 65536 || bytes > 16 * 1024 * 1024 || result.len() >= 256 {
            return Err(invalid(
                "DSpark dataset exceeds bounded line/file/prompt extent",
            ));
        }
        if line.ends_with('\n') {
            line.pop();
            if line.ends_with('\r') {
                line.pop();
            }
        }
        if line.is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(&line)?;
        if let Some(messages) = value.get("messages") {
            let mut messages = messages
                .as_array()
                .ok_or_else(|| invalid("dataset messages must be an array"))?
                .iter()
                .map(|message| -> Result<Value> {
                    let role = message["role"]
                        .as_str()
                        .ok_or_else(|| invalid("dataset role must be text"))?;
                    let content = message["content"]
                        .as_str()
                        .ok_or_else(|| invalid("dataset content must be text"))?;
                    Ok(json!({"role":role,"content":content}))
                })
                .collect::<Result<Vec<_>>>()?;
            if messages
                .last()
                .is_some_and(|message| message["role"] == "assistant")
            {
                messages.pop();
            }
            if messages
                .last()
                .is_none_or(|message| message["role"] != "user")
            {
                return Err(invalid(
                    "held-out prompt must end in a user turn after removing its reference answer",
                ));
            }
            result.push(Prompt {
                category: "clean-heldout".into(),
                text: String::new(),
                messages,
            });
        } else {
            let text = value["turns"][0]
                .as_str()
                .ok_or_else(|| invalid("dataset first turn must be text"))?;
            result.push(Prompt {
                category: "deepspec".into(),
                text: text.into(),
                messages: Vec::new(),
            });
        }
    }
    Ok(result)
}

pub(super) fn target_layers(path: &Path) -> Result<Vec<i32>> {
    // Header-only reading is independent of the loaded model's hparams, as in
    // the source's gguf_get_arr_data path. No tensor weights are decoded here.
    let mut bytes = Vec::new();
    File::open(path)?
        .take(64 * 1024 * 1024)
        .read_to_end(&mut bytes)?;
    let model = super::gguf_model_data::parse_meta(&bytes)
        .ok_or_else(|| invalid("drafter GGUF metadata could not be parsed within64MiB"))?;
    let arch = std::str::from_utf8(&model.architecture)?;
    let key = format!("{arch}.dspark.target_layers");
    let entry = model
        .entries
        .iter()
        .find(|entry| entry.key == key.as_bytes())
        .ok_or_else(|| invalid("drafter target_layers missing"))?;
    if entry.kind != 9 || entry.payload.len() < 12 {
        return Err(invalid("target_layers is not an array-typed KV"));
    }
    let kind = u32::from_le_bytes(entry.payload[0..4].try_into()?);
    let count = usize::try_from(u64::from_le_bytes(entry.payload[4..12].try_into()?))?;
    let width = match kind {
        4 | 5 => 4,
        10 | 11 => 8,
        _ => return Err(invalid("unexpected target_layers array element type")),
    };
    if count > 256
        || count.checked_mul(width).and_then(|n| n.checked_add(12)) != Some(entry.payload.len())
    {
        return Err(invalid("target layer array extent differs"));
    }
    // Windows C++ narrows INT64/UINT64 by retaining the low32bits. Keeping raw
    // little-endian bytes models that conversion without an unsafe cast.
    (0..count)
        .map(|i| {
            Ok(i32::from_le_bytes(
                entry.payload[12 + i * width..16 + i * width].try_into()?,
            ))
        })
        .collect()
}

pub(super) struct OracleComparison {
    pub(super) top: usize,
    pub(super) max_abs: f32,
    pub(super) kld: Option<f64>,
}
pub(super) fn oracle_compare(
    reference: &[f32],
    actual: &[f32],
    measure_kld: bool,
) -> Result<OracleComparison> {
    if reference.is_empty() || reference.len() != actual.len() {
        return Err(invalid("oracle vocabulary extent differs"));
    }
    let mut top = 0;
    let mut max_abs = 0_f32;
    for (i, (&a, &b)) in reference.iter().zip(actual).enumerate() {
        if !a.is_finite() || !b.is_finite() {
            return Err(invalid("oracle non-finite logit"));
        }
        if actual[i] > actual[top] {
            top = i;
        }
        max_abs = max_abs.max((b - a).abs());
    }
    let kld = if measure_kld {
        let ref_max = f64::from(reference.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        let test_max = f64::from(actual[top]);
        let (mut ref_sum, mut test_sum, mut weighted_delta) = (0_f64, 0_f64, 0_f64);
        for (&a, &b) in reference.iter().zip(actual) {
            let weight = (f64::from(a) - ref_max).exp();
            ref_sum += weight;
            test_sum += (f64::from(b) - test_max).exp();
            weighted_delta += weight * (f64::from(a) - f64::from(b));
        }
        let value = weighted_delta / ref_sum + (test_sum / ref_sum).ln() + test_max - ref_max;
        if !value.is_finite() || value < -1e-10 {
            return Err(invalid("invalid oracle KLD"));
        }
        Some(value.max(0.0))
    } else {
        None
    };
    Ok(OracleComparison { top, max_abs, kld })
}

#[cfg(test)]
mod tests {
    #[test]
    fn source_oracle_ties_offsets_and_nonfinite_refusals() -> super::Result<()> {
        let value = super::oracle_compare(&[1.0, 1.0, -2.0], &[9.0, 9.0, 6.0], true)?;
        assert_eq!(value.top, 0);
        assert_eq!(value.max_abs, 8.0);
        assert!(value.kld.is_some_and(|kld| kld.abs() < 1e-14));
        assert!(super::oracle_compare(&[f32::NAN], &[0.0], true).is_err());
        assert!(super::oracle_compare(&[0.0], &[f32::INFINITY], false).is_err());
        assert!(super::oracle_compare(&[0.0], &[], false).is_err());
        assert_eq!(super::prompts(&super::Options::default())?.len(), 24);
        Ok(())
    }
}
