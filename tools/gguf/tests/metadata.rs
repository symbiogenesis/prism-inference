#![forbid(unsafe_code)]
//! Focused native metadata representation, retention-budget and report regressions.

use std::fs;
use std::path::PathBuf;

use bonsai_gguf::{MetadataArray, MetadataValue, Result, VerifiedModel};
use sha2::{Digest, Sha256};

mod support;
use support::ScratchDirectory;

fn text(value: &str) -> Result<Vec<u8>> {
    let mut bytes = u64::try_from(value.len())?.to_le_bytes().to_vec();
    bytes.extend(value.as_bytes());
    Ok(bytes)
}

fn array(kind: u32, count: u64, payload: impl IntoIterator<Item = u8>) -> Vec<u8> {
    let mut bytes = kind.to_le_bytes().to_vec();
    bytes.extend(count.to_le_bytes());
    bytes.extend(payload);
    bytes
}

struct Artifact {
    directory: Option<ScratchDirectory>,
    path: PathBuf,
    bytes: Vec<u8>,
    hash: String,
}

impl Artifact {
    fn new(records: &[(&str, u32, Vec<u8>)]) -> Result<Self> {
        let directory = ScratchDirectory::new()?;
        let path = directory.path().join("metadata.gguf");
        let mut bytes = 0x4655_4747_u32.to_le_bytes().to_vec();
        bytes.extend(3_u32.to_le_bytes());
        bytes.extend(0_u64.to_le_bytes());
        bytes.extend(u64::try_from(records.len() + 1)?.to_le_bytes());
        bytes.extend(text("general.architecture")?);
        bytes.extend(8_u32.to_le_bytes());
        bytes.extend(text("synthetic")?);
        for (key, kind, value) in records {
            bytes.extend(text(key)?);
            bytes.extend(kind.to_le_bytes());
            bytes.extend(value);
        }
        fs::write(&path, &bytes)?;
        let hash = Sha256::digest(&bytes)
            .into_iter()
            .flat_map(|byte| [byte >> 4, byte & 15])
            .map(|nibble| char::from(b"0123456789abcdef"[usize::from(nibble)]))
            .collect();
        Ok(Self {
            directory: Some(directory),
            path,
            bytes,
            hash,
        })
    }

    fn open(&self, budget: usize) -> Result<VerifiedModel> {
        VerifiedModel::open(
            &self.path,
            "Stock",
            u64::try_from(self.bytes.len())?,
            &self.hash,
            budget,
        )
    }

    #[cfg(feature = "report")]
    fn report(&self) -> Result<serde_json::Value> {
        bonsai_gguf::inspect(
            &self.path,
            "Stock",
            false,
            u64::try_from(self.bytes.len())?,
            &self.hash,
        )
    }
}

impl Drop for Artifact {
    fn drop(&mut self) {
        if let Some(directory) = self.directory.take() {
            let _ = directory.close();
        }
    }
}

#[test]
fn typed_scalars_preserve_wire_width_extremes_and_native_float_bits() -> Result<()> {
    let artifact = Artifact::new(&[
        ("u8", 0, u8::MAX.to_le_bytes().to_vec()),
        ("i8", 1, i8::MIN.to_le_bytes().to_vec()),
        ("u16", 2, u16::MAX.to_le_bytes().to_vec()),
        ("i16", 3, i16::MIN.to_le_bytes().to_vec()),
        ("u32", 4, u32::MAX.to_le_bytes().to_vec()),
        ("i32", 5, i32::MIN.to_le_bytes().to_vec()),
        ("f32", 6, 0.1_f32.to_le_bytes().to_vec()),
        ("bool", 7, vec![1]),
        ("string", 8, text("wide λ 🦀")?),
        ("u64", 10, u64::MAX.to_le_bytes().to_vec()),
        ("i64", 11, i64::MIN.to_le_bytes().to_vec()),
        ("f64", 12, f64::MIN_POSITIVE.to_le_bytes().to_vec()),
        ("negative_zero", 6, (-0.0_f32).to_le_bytes().to_vec()),
    ])?;
    let model = artifact.open(16_384)?;
    let metadata = model.metadata();
    for (key, expected) in [
        ("u8", MetadataValue::U8(u8::MAX)),
        ("i8", MetadataValue::I8(i8::MIN)),
        ("u16", MetadataValue::U16(u16::MAX)),
        ("i16", MetadataValue::I16(i16::MIN)),
        ("u32", MetadataValue::U32(u32::MAX)),
        ("i32", MetadataValue::I32(i32::MIN)),
        ("bool", MetadataValue::Bool(true)),
        ("string", MetadataValue::String("wide λ 🦀".to_owned())),
        ("u64", MetadataValue::U64(u64::MAX)),
        ("i64", MetadataValue::I64(i64::MIN)),
    ] {
        assert_eq!(metadata[key], expected);
    }
    assert!(
        matches!(metadata["f32"], MetadataValue::F32(value) if value.to_bits() == 0.1_f32.to_bits())
    );
    assert!(
        matches!(metadata["negative_zero"], MetadataValue::F32(value) if value.to_bits() == (-0.0_f32).to_bits())
    );
    assert!(
        matches!(metadata["f64"], MetadataValue::F64(value) if value.to_bits() == f64::MIN_POSITIVE.to_bits())
    );
    assert_eq!(MetadataValue::I8(3).as_u64(), Some(3));
    assert_eq!(MetadataValue::I64(-1).as_u64(), None);
    assert_eq!(MetadataValue::U64(u64::MAX).as_i64(), None);
    assert_eq!(metadata["f32"].as_f64(), Some(f64::from(0.1_f32)));
    Ok(())
}

#[test]
fn typed_arrays_are_compact_and_preserve_strings_and_nested_arrays() -> Result<()> {
    let values = [0.1_f32, -0.0, f32::MIN_POSITIVE, f32::MAX];
    let strings = [text("first")?, text("λ")?].concat();
    let nested = [
        array(
            3,
            2,
            [i16::MIN, i16::MAX].into_iter().flat_map(i16::to_le_bytes),
        ),
        array(3, 0, []),
    ]
    .concat();
    let artifact = Artifact::new(&[
        ("u8s", 9, array(0, 5000, vec![u8::MAX; 5000])),
        (
            "signed",
            9,
            array(
                11,
                2,
                [i64::MIN, i64::MAX].into_iter().flat_map(i64::to_le_bytes),
            ),
        ),
        (
            "unsigned",
            9,
            array(10, 2, [0, u64::MAX].into_iter().flat_map(u64::to_le_bytes)),
        ),
        (
            "floats",
            9,
            array(6, 4, values.into_iter().flat_map(f32::to_le_bytes)),
        ),
        ("strings", 9, array(8, 2, strings)),
        ("nested", 9, array(9, 2, nested)),
        ("flags", 9, array(7, 3, [0, 1, 0])),
    ])?;
    // The byte array fits this budget in its native representation; a per-item
    // metadata enum for 5000 values would exceed the whole budget.
    let model = artifact.open(16_384)?;
    let metadata = model.metadata();
    assert!(
        matches!(&metadata["u8s"], MetadataValue::Array(MetadataArray::U8(values))
        if values.len() == 5000 && values[4999] == u8::MAX)
    );
    assert_eq!(
        metadata["signed"],
        MetadataValue::Array(MetadataArray::I64(vec![i64::MIN, i64::MAX]))
    );
    assert_eq!(
        metadata["unsigned"],
        MetadataValue::Array(MetadataArray::U64(vec![0, u64::MAX]))
    );
    let Some(MetadataArray::F32(floats)) = metadata["floats"].as_array() else {
        return Err("F32 array lost its native representation".into());
    };
    assert_eq!(
        floats
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        metadata["strings"],
        MetadataValue::Array(MetadataArray::String(vec![
            "first".to_owned(),
            "λ".to_owned()
        ]))
    );
    assert_eq!(
        metadata["nested"],
        MetadataValue::Array(MetadataArray::Array(vec![
            MetadataArray::I16(vec![i16::MIN, i16::MAX]),
            MetadataArray::I16(Vec::new())
        ]))
    );
    assert_eq!(
        metadata["flags"],
        MetadataValue::Array(MetadataArray::Bool(vec![false, true, false]))
    );
    let charged = model.retained_bytes();
    assert!(charged < 16_384);
    drop(model);
    assert_eq!(artifact.open(charged)?.retained_bytes(), charged);
    assert!(artifact.open(charged - 1).is_err());
    Ok(())
}

#[test]
fn typed_array_budget_is_checked_before_reservation_and_payload_reads() -> Result<()> {
    // The truncated payload is deliberate. A retained-size failure must occur
    // before trying to allocate or read this declared two-million-item array.
    let artifact = Artifact::new(&[("huge", 9, array(12, 2_000_000, []))])?;
    let Err(error) = artifact.open(1024) else {
        return Err("over-budget array was accepted".into());
    };
    assert!(error.to_string().contains("exceeds memory budget"));
    Ok(())
}

#[test]
fn typed_retention_rejects_nonfinite_values_invalid_booleans_and_excessive_nesting() -> Result<()> {
    for (kind, payload) in [
        (6, f32::NAN.to_le_bytes().to_vec()),
        (12, f64::INFINITY.to_le_bytes().to_vec()),
        (7, vec![2]),
        (9, array(6, 1, f32::NEG_INFINITY.to_le_bytes())),
        (9, array(7, 1, [2])),
    ] {
        assert!(
            Artifact::new(&[("invalid", kind, payload)])?
                .open(4096)
                .is_err()
        );
    }
    let mut nested = array(0, 1, [0]);
    for _ in 0..4 {
        nested = array(9, 1, nested);
    }
    assert!(Artifact::new(&[("nested", 9, nested)])?.open(4096).is_err());
    Ok(())
}

#[cfg(feature = "report")]
#[test]
fn report_schema_preserves_short_f32_decimals_and_array_summaries() -> Result<()> {
    let artifact = Artifact::new(&[
        ("f32", 6, 0.1_f32.to_le_bytes().to_vec()),
        ("f64", 12, f64::from(0.1_f32).to_le_bytes().to_vec()),
        ("u64", 10, u64::MAX.to_le_bytes().to_vec()),
        ("i64", 11, i64::MIN.to_le_bytes().to_vec()),
        (
            "floats",
            9,
            array(6, 2, [0.1_f32, 0.2].into_iter().flat_map(f32::to_le_bytes)),
        ),
        ("strings", 9, array(8, 1, text("token")?)),
        ("large", 9, array(0, 4097, vec![0; 4097])),
    ])?;
    let report = artifact.report()?;
    assert_eq!(report["SchemaVersion"], 1);
    assert_eq!(report["Scalars"]["f32"], serde_json::json!(0.1));
    assert_eq!(
        report["Scalars"]["f64"],
        serde_json::json!(f64::from(0.1_f32))
    );
    assert_eq!(report["Scalars"]["u64"], serde_json::json!(u64::MAX));
    assert_eq!(report["Scalars"]["i64"], serde_json::json!(i64::MIN));
    assert_eq!(report["Arrays"]["floats"], serde_json::json!([0.1, 0.2]));
    assert_eq!(
        report["Arrays"]["strings"],
        serde_json::json!({"ElementTypeId":8,"Count":1})
    );
    assert_eq!(
        report["Arrays"]["large"],
        serde_json::json!({"ElementTypeId":0,"Count":4097})
    );
    assert_eq!(report["TensorCount"], 0);
    assert_eq!(report["TensorPayloadRead"], false);
    Ok(())
}

#[test]
fn architecture_count_conversion_uses_native_floats_and_existing_rounding_rules() -> Result<()> {
    for (kind, payload) in [
        (7, vec![0]),
        (8, text(" \t-000\r\n")?),
        (8, text("+000")?),
        (6, (-0.5_f32).to_le_bytes().to_vec()),
        (12, (-0.0_f64).to_le_bytes().to_vec()),
    ] {
        assert!(
            Artifact::new(&[("synthetic.block_count", kind, payload)])?
                .open(4096)
                .is_ok()
        );
    }
    // 2.5 rounds to 2, whereas 3.5 rounds to 4.
    for (count, accepted) in [(2.5_f32, false), (3.5_f32, true)] {
        let artifact = Artifact::new(&[
            ("synthetic.block_count", 6, count.to_le_bytes().to_vec()),
            (
                "synthetic.nextn_predict_layers",
                4,
                3_u32.to_le_bytes().to_vec(),
            ),
        ])?;
        assert_eq!(artifact.open(4096).is_ok(), accepted);
    }
    for (kind, payload) in [
        (11, (-1_i64).to_le_bytes().to_vec()),
        (12, 18_446_744_073_709_551_616.0_f64.to_le_bytes().to_vec()),
    ] {
        assert!(
            Artifact::new(&[("synthetic.block_count", kind, payload)])?
                .open(4096)
                .is_err()
        );
    }
    Ok(())
}
