//! Safe Rust GGUF input writer for the pinned `DSpark` companion generator.
//! Frozen bytes contain only generator-produced tensor inputs, never SUT output.
use super::{Result, invalid};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::Path;
const DATA: &[u8] = include_bytes!("dspark-inputs.bin");
const SHA: &str = "be829f6051708499ad7ace92e4db5703acf1774d6feea5b8ec9dd0dca7fb00e9";
fn check(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(invalid(message)) }
}
fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    bytes
        .iter()
        .flat_map(|&b| {
            [
                char::from(DIGITS[usize::from(b >> 4)]),
                char::from(DIGITS[usize::from(b & 15)]),
            ]
        })
        .collect()
}
fn string(bytes: &mut Vec<u8>, value: &str) -> Result<()> {
    bytes.extend_from_slice(&u64::try_from(value.len())?.to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
    Ok(())
}
fn value(name: &str, kind: u32, data: &[u8]) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    string(&mut bytes, name)?;
    bytes.extend_from_slice(&kind.to_le_bytes());
    bytes.extend_from_slice(data);
    Ok(bytes)
}
fn text(name: &str, data: &str) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    string(&mut bytes, data)?;
    value(name, 8, &bytes)
}
fn u32_value(name: &str, data: u32) -> Result<Vec<u8>> {
    value(name, 4, &data.to_le_bytes())
}
fn f32_bits(name: &str, bits: u32) -> Result<Vec<u8>> {
    value(name, 6, &bits.to_le_bytes())
}
fn boolean(name: &str, data: bool) -> Result<Vec<u8>> {
    value(name, 7, &[u8::from(data)])
}
fn aligned(bytes: &mut Vec<u8>) -> Result<()> {
    let end = bytes
        .len()
        .checked_add(31)
        .ok_or_else(|| invalid("DSpark alignment overflow"))?
        & !31;
    bytes.resize(end, 0);
    Ok(())
}
fn number(v: &Value, key: &str) -> Result<usize> {
    usize::try_from(
        v[key]
            .as_u64()
            .ok_or_else(|| invalid(format!("missing DSpark input number {key}")))?,
    )
    .map_err(Into::into)
}
pub(super) struct Corpus {
    manifest: Value,
}
impl Corpus {
    pub(super) fn load() -> Result<Self> {
        check(
            DATA.len() == 1_236_480 && hex(&Sha256::digest(DATA)) == SHA,
            "DSpark frozen tensor corpus hash/length differs",
        )?;
        let manifest: Value = serde_json::from_str(include_str!("dspark-inputs.json"))?;
        check(
            manifest["InputsOnly"] == true
                && manifest["Seed"] == 1234
                && manifest["BlobSHA256"] == SHA,
            "DSpark tensor provenance differs",
        )?;
        Ok(Self { manifest })
    }
    #[allow(
        clippy::too_many_lines,
        reason = "keep the literal source metadata and independent binary writer together"
    )]
    pub(super) fn write(
        &self,
        path: &Path,
        conditioning: bool,
        minimum: Option<u32>,
        maximum: Option<u32>,
        correction: bool,
    ) -> Result<()> {
        let label = if correction {
            "corrected"
        } else if conditioning {
            "conditioned"
        } else {
            "unconditioned"
        };
        check(
            !correction || conditioning,
            "corrected input corpus requires conditioning",
        )?;
        let layout = self.manifest["Layouts"]
            .as_array()
            .ok_or_else(|| invalid("DSpark input layouts absent"))?
            .iter()
            .find(|v| v["Name"] == label)
            .ok_or_else(|| invalid("DSpark tensor layout absent"))?;
        let tensors = layout["Tensors"]
            .as_array()
            .ok_or_else(|| invalid("DSpark tensor inventory absent"))?;
        let mut entries = vec![
            text("general.architecture", "dspark")?,
            text("general.name", "tiny-random-dspark")?,
            u32_value("general.file_type", 0)?,
        ];
        for (key, n) in [
            ("block_count", 2),
            ("context_length", 512),
            ("embedding_length", 64),
            ("feed_forward_length", 96),
            ("attention.head_count", 2),
            ("attention.head_count_kv", 1),
        ] {
            entries.push(u32_value(&format!("dspark.{key}"), n)?);
        }
        entries.push(f32_bits(
            "dspark.attention.layer_norm_rms_epsilon",
            1e-6_f32.to_bits(),
        )?);
        entries.push(f32_bits("dspark.rope.freq_base", 10000_f32.to_bits())?);
        entries.push(u32_value("dspark.dspark.block_size", 4)?);
        entries.push(u32_value("dspark.dspark.mask_token_id", 255)?);
        let mut layers = 4_u32.to_le_bytes().to_vec();
        layers.extend_from_slice(&2_u64.to_le_bytes());
        layers.extend_from_slice(&1_u32.to_le_bytes());
        layers.extend_from_slice(&16_u32.to_le_bytes());
        entries.push(value("dspark.dspark.target_layers", 9, &layers)?);
        entries.push(boolean("dspark.dspark.log_snr_conditioning", conditioning)?);
        if let Some(bits) = minimum {
            entries.push(f32_bits("dspark.dspark.min_log_snr", bits)?);
        }
        if let Some(bits) = maximum {
            entries.push(f32_bits("dspark.dspark.max_log_snr", bits)?);
        }
        entries.push(text("tokenizer.ggml.model", "none")?);
        entries.push(u32_value("dspark.vocab_size", 256)?);
        if correction {
            entries.push(boolean("dspark.dspark.hidden_correction", true)?);
            entries.push(boolean("dspark.dspark.markov_deploy_exposure", true)?);
            entries.push(u32_value("dspark.dspark.correction_size", 64)?);
            entries.push(u32_value("dspark.dspark.markov_rank", 8)?);
        }
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&u64::try_from(tensors.len())?.to_le_bytes());
        bytes.extend_from_slice(&u64::try_from(entries.len())?.to_le_bytes());
        for entry in entries {
            bytes.extend(entry);
        }
        let mut tensor_data = Vec::new();
        let mut names = std::collections::BTreeSet::new();
        for tensor in tensors {
            let name = tensor["Name"]
                .as_str()
                .ok_or_else(|| invalid("DSpark tensor name absent"))?;
            check(names.insert(name), "duplicate DSpark input tensor name")?;
            let dimensions = tensor["Dimensions"]
                .as_array()
                .ok_or_else(|| invalid("DSpark tensor shape absent"))?;
            check(
                !dimensions.is_empty() && dimensions.len() <= 2,
                "DSpark tensor input rank differs",
            )?;
            let extent = dimensions.iter().try_fold(4_usize, |n, v| {
                n.checked_mul(usize::try_from(
                    v.as_u64()
                        .ok_or_else(|| invalid("DSpark invalid tensor extent"))?,
                )?)
                .ok_or_else(|| invalid("DSpark tensor extent overflow"))
            })?;
            check(
                extent == number(tensor, "Bytes")?,
                "DSpark tensor shape/byte count differs",
            )?;
            let offset = number(tensor, "Offset")?;
            let end = offset
                .checked_add(extent)
                .ok_or_else(|| invalid("DSpark corpus range overflow"))?;
            let data = DATA
                .get(offset..end)
                .ok_or_else(|| invalid("DSpark corpus range outside bytes"))?;
            check(
                tensor["SHA256"].as_str() == Some(hex(&Sha256::digest(data)).as_str()),
                "DSpark tensor input digest differs",
            )?;
            aligned(&mut tensor_data)?;
            string(&mut bytes, name)?;
            bytes.extend_from_slice(&u32::try_from(dimensions.len())?.to_le_bytes());
            for n in dimensions {
                bytes.extend_from_slice(
                    &n.as_u64()
                        .ok_or_else(|| invalid("tensor dimension absent"))?
                        .to_le_bytes(),
                );
            }
            bytes.extend_from_slice(&0_u32.to_le_bytes());
            bytes.extend_from_slice(&u64::try_from(tensor_data.len())?.to_le_bytes());
            tensor_data.extend_from_slice(data);
        }
        aligned(&mut bytes)?;
        bytes.extend(tensor_data);
        check(
            bytes.len() <= 1024 * 1024,
            "DSpark generated model exceeds1MiB",
        )?;
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(path)?;
        file.write_all(&bytes)?;
        Ok(())
    }
}
