#![forbid(unsafe_code)]
//! Focused generated-model, callback and streaming saver contracts.

use std::collections::BTreeMap;
use std::fs;
use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use bonsai_gguf::{Metadata, MetadataArray, MetadataValue};
use bonsai_prism_runtime::Result;
use bonsai_prism_runtime::model::Model;
use bonsai_prism_runtime::model::gguf_context::tensor_layout;
use bonsai_prism_runtime::model::owned::TensorSpec;
use sha2::{Digest, Sha256};

const BUDGET: usize = 2 * 1024 * 1024;
static NEXT: AtomicU64 = AtomicU64::new(0);

struct Artifact {
    root: PathBuf,
    path: PathBuf,
    bytes: Vec<u8>,
}

impl Artifact {
    fn new(bytes: Vec<u8>) -> Result<Self> {
        let root = std::env::temp_dir().join(format!(
            "SwiftBonsai-OwnedModel-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root)?;
        let path = root.join("model.gguf");
        fs::write(&path, &bytes)?;
        Ok(Self { root, path, bytes })
    }

    fn load(&self, progress: &mut impl FnMut(f32) -> Result<bool>) -> Result<Model> {
        Model::load_with_progress(
            &self.path,
            u64::try_from(self.bytes.len())?,
            &hash(&self.bytes),
            BUDGET,
            progress,
        )
    }
}

impl Drop for Artifact {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
        let _ = fs::remove_dir(&self.root);
    }
}

fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .into_iter()
        .flat_map(|byte| [byte >> 4, byte & 15])
        .map(|value| char::from(b"0123456789abcdef"[usize::from(value)]))
        .collect()
}

fn metadata() -> Metadata {
    BTreeMap::from([
        (
            "general.architecture".into(),
            MetadataValue::String("llama".into()),
        ),
        ("general.alignment".into(), MetadataValue::U32(64)),
    ])
}

fn spec(name: &str, dimensions: &[usize], kind: u32) -> TensorSpec {
    TensorSpec {
        name: name.into(),
        kind,
        dimensions: dimensions.to_vec(),
    }
}

fn generated() -> Result<Model> {
    Model::generate(
        metadata(),
        &[spec("a", &[1], 0), spec("b", &[3], 0)],
        BUDGET,
        |tensor, bytes| {
            for (index, destination) in bytes.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                let value = if tensor.name == "a" {
                    0.25
                } else {
                    [-1.0, 2.0, 4.0][index]
                };
                *destination = f32::to_le_bytes(value);
            }
            Ok(())
        },
    )
}

fn encoded(model: &Model) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    assert_eq!(model.write_to(&mut bytes)?, u64::try_from(bytes.len())?);
    Ok(bytes)
}

#[test]
fn generated_storage_roundtrips_every_metadata_type_nested_arrays_and_tensor_rank() -> Result<()> {
    let mut metadata = metadata();
    let scalars = [
        MetadataValue::U8(251),
        MetadataValue::I8(-113),
        MetadataValue::U16(61_234),
        MetadataValue::I16(-30_000),
        MetadataValue::U32(4_000_000_001),
        MetadataValue::I32(-2_000_000_001),
        MetadataValue::F32(-0.0),
        MetadataValue::Bool(true),
        MetadataValue::String("a\0λ".into()),
        MetadataValue::U64(u64::MAX - 1),
        MetadataValue::I64(i64::MIN + 1),
        MetadataValue::F64(-1.25),
    ];
    for (index, value) in scalars.into_iter().enumerate() {
        metadata.insert(format!("scalar.{index}"), value);
    }
    let arrays = [
        MetadataArray::U8(vec![0, 255]),
        MetadataArray::I8(vec![-128, 127]),
        MetadataArray::U16(vec![0, u16::MAX]),
        MetadataArray::I16(vec![i16::MIN, i16::MAX]),
        MetadataArray::U32(vec![0, u32::MAX]),
        MetadataArray::I32(vec![i32::MIN, i32::MAX]),
        MetadataArray::F32(vec![-0.0, 1.25]),
        MetadataArray::Bool(vec![true, false]),
        MetadataArray::String(vec!["λ".into(), "a\0b".into(), String::new()]),
        MetadataArray::U64(vec![0, u64::MAX]),
        MetadataArray::I64(vec![i64::MIN, i64::MAX]),
        MetadataArray::F64(vec![-0.0, 1.25]),
        MetadataArray::Array(vec![
            MetadataArray::U32(vec![1, 2]),
            MetadataArray::Array(vec![MetadataArray::String(vec!["nested".into()])]),
            MetadataArray::F32(vec![]),
        ]),
    ];
    for (index, value) in arrays.into_iter().enumerate() {
        metadata.insert(format!("array.{index}"), MetadataValue::Array(value));
    }
    let expected = metadata.clone();
    let mut callbacks = Vec::new();
    let model = Model::generate(
        metadata,
        &[spec("z", &[2, 1, 1, 1], 0), spec("a", &[1], 1)],
        BUDGET,
        |spec, data| {
            callbacks.push(spec.name.clone());
            if spec.name == "z" {
                data.copy_from_slice(&[1.25_f32.to_le_bytes(), (-0.0_f32).to_le_bytes()].concat());
            } else {
                data.copy_from_slice(&0x3c00_u16.to_le_bytes());
            }
            Ok(())
        },
    )?;
    assert_eq!(callbacks, ["z", "a"]);
    let bytes = encoded(&model)?;
    assert_eq!(model.identity, hash(&bytes));
    let fixture = Artifact::new(bytes)?;
    let loaded = fixture.load(&mut |_| Ok(true))?;
    assert_eq!(loaded.metadata, expected);
    assert_eq!(loaded.identity, model.identity);
    for (name, source) in &model.tensors {
        let saved = loaded.tensors.get(name).ok_or("missing saved tensor")?;
        assert_eq!(saved.dimensions, source.dimensions);
        assert_eq!(saved.kind, source.kind);
        assert_eq!(saved.data, source.data);
    }
    assert_eq!(encoded(&loaded)?, fixture.bytes);
    Ok(())
}

#[test]
fn generated_and_verified_loaders_accept_every_production_decoder_without_expansion() -> Result<()>
{
    let mut specs = Vec::new();
    for kind in (0..=3)
        .chain(6..=23)
        .chain([29, 30, 34, 35, 39, 40, 41, 42, 142, 143])
    {
        let (block, _) = tensor_layout(kind).ok_or("missing production layout")?;
        specs.push(spec(
            &format!("type.{kind}"),
            &[usize::try_from(block)?, 2],
            kind,
        ));
    }
    let model = Model::generate(metadata(), &specs, BUDGET, |_, _| Ok(()))?;
    let fixture = Artifact::new(encoded(&model)?)?;
    let loaded = fixture.load(&mut |_| Ok(true))?;
    assert_eq!(loaded.tensors.len(), specs.len());
    for spec in specs {
        let tensor = loaded
            .tensors
            .get(&spec.name)
            .ok_or("missing decoder tensor")?;
        assert_eq!(tensor.kind, spec.kind);
        assert_eq!(tensor.dimensions, spec.dimensions);
        assert!(tensor.data.iter().all(|&byte| byte == 0));
    }
    Ok(())
}

#[test]
fn generated_preflight_rejects_budget_duplicate_and_shape_errors_before_callbacks() {
    for specs in [
        vec![spec("a", &[1], 0), spec("a", &[1], 0)],
        vec![spec("a", &[1], 0), spec("bad", &[31], 2)],
        vec![spec("a", &[1], 0), spec("bad", &[usize::MAX, 2], 0)],
        vec![spec("a", &[1], 0), spec("bad", &[1], 24)],
        vec![spec("a", &[0], 0)],
    ] {
        let mut calls = 0;
        assert!(
            Model::generate(metadata(), &specs, BUDGET, |_, _| {
                calls += 1;
                Ok(())
            })
            .is_err()
        );
        assert_eq!(calls, 0);
    }
    let mut calls = 0;
    assert!(
        Model::generate(metadata(), &[spec("a", &[128], 0)], 1, |_, _| {
            calls += 1;
            Ok(())
        })
        .is_err()
    );
    assert_eq!(calls, 0);
}

#[test]
fn metadata_preflight_rejects_reserved_types_nonfinite_arrays_and_excessive_nesting() {
    let mut nested = MetadataArray::U8(vec![1]);
    for _ in 0..4 {
        nested = MetadataArray::Array(vec![nested]);
    }
    for (key, value) in [
        ("general.architecture", MetadataValue::String(String::new())),
        ("general.alignment", MetadataValue::U32(3)),
        ("tokenizer.ggml.bos_token_id", MetadataValue::U8(1)),
        ("nested", MetadataValue::Array(nested)),
        (
            "nonfinite",
            MetadataValue::Array(MetadataArray::F32(vec![f32::NAN])),
        ),
    ] {
        let mut meta = metadata();
        meta.insert(key.into(), value);
        let mut calls = 0;
        assert!(
            Model::generate(meta, &[spec("a", &[1], 0)], BUDGET, |_, _| {
                calls += 1;
                Ok(())
            })
            .is_err()
        );
        assert_eq!(calls, 0);
    }
}

#[test]
fn callback_errors_and_nonfinite_generated_payloads_never_publish_partial_models() {
    let mut calls = 0;
    let result = Model::generate(
        metadata(),
        &[spec("a", &[1], 0), spec("b", &[1], 0)],
        BUDGET,
        |_, _| {
            calls += 1;
            if calls == 2 {
                return Err(std::io::Error::other("callback cancellation").into());
            }
            Ok(())
        },
    );
    assert!(result.is_err());
    assert_eq!(calls, 2);
    assert!(
        Model::generate(metadata(), &[spec("a", &[1], 0)], BUDGET, |_, bytes| {
            bytes.copy_from_slice(&f32::INFINITY.to_le_bytes());
            Ok(())
        })
        .is_err()
    );
}

#[test]
fn decoded_q8k_overflow_is_rejected_by_generation_loading_and_saving() -> Result<()> {
    let specifications = [spec("a", &[256], 15)];
    let mut model = Model::generate(metadata(), &specifications, BUDGET, |_, bytes| {
        bytes[..4].copy_from_slice(&f32::MAX.to_le_bytes());
        bytes[4] = 1;
        Ok(())
    })?;
    let mut overflowing = model
        .tensors
        .get("a")
        .ok_or("missing Q8_K tensor")?
        .data
        .to_vec();
    overflowing[4] = 2;
    assert!(
        Model::generate(metadata(), &specifications, BUDGET, |_, bytes| {
            bytes.copy_from_slice(&overflowing);
            Ok(())
        })
        .is_err()
    );
    let mut serialized = encoded(&model)?;
    // One tensor, with the fixture's 64-byte alignment and padded final payload.
    let offset = serialized.len() - overflowing.len().div_ceil(64) * 64;
    serialized[offset..offset + overflowing.len()].copy_from_slice(&overflowing);
    let fixture = Artifact::new(serialized)?;
    assert!(fixture.load(&mut |_| Ok(true)).is_err());
    model
        .tensors
        .get_mut("a")
        .ok_or("missing Q8_K tensor")?
        .data = std::sync::Arc::from(overflowing);
    let mut output = Vec::new();
    assert!(model.write_to(&mut output).is_err());
    assert_eq!(output, Vec::<u8>::new());
    Ok(())
}

#[test]
fn file_progress_is_byte_weighted_and_cancellation_includes_the_final_callback() -> Result<()> {
    let fixture = Artifact::new(encoded(&generated()?)?)?;
    let mut progress = Vec::new();
    fixture.load(&mut |value| {
        progress.push(value.to_bits());
        Ok(true)
    })?;
    assert_eq!(
        progress,
        [0.0_f32.to_bits(), 0.25_f32.to_bits(), 1.0_f32.to_bits()]
    );
    for stop in 1..=3 {
        let mut calls = 0;
        assert!(
            fixture
                .load(&mut |_| {
                    calls += 1;
                    Ok(calls != stop)
                })
                .is_err()
        );
        assert_eq!(calls, stop);
    }
    let mut calls = 0;
    assert!(
        fixture
            .load(&mut |_| {
                calls += 1;
                Err(std::io::Error::other("progress failure").into())
            })
            .is_err()
    );
    assert_eq!(calls, 1);
    Ok(())
}

#[test]
fn saver_preserves_stream_position_and_prevalidates_public_model_fields() -> Result<()> {
    let mut model = generated()?;
    let bytes = encoded(&model)?;
    let mut stream = Cursor::new(vec![0xaa; 7]);
    stream.set_position(7);
    assert_eq!(model.write_to(&mut stream)?, u64::try_from(bytes.len())?);
    assert_eq!(&stream.get_ref()[..7], &[0xaa; 7]);
    assert_eq!(&stream.get_ref()[7..], &bytes);
    model
        .metadata
        .insert("bad".into(), MetadataValue::F64(f64::NAN));
    let mut output = Vec::new();
    assert!(model.write_to(&mut output).is_err());
    assert_eq!(output, Vec::<u8>::new());
    model.metadata.remove("bad");
    model
        .tensors
        .get_mut("a")
        .ok_or("missing tensor")?
        .dimensions[0] = 2;
    assert!(model.write_to(&mut output).is_err());
    assert_eq!(output, Vec::<u8>::new());
    Ok(())
}

struct FailingWriter;

impl Write for FailingWriter {
    fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
        Err(std::io::Error::other("owned output failure"))
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn saver_propagates_owned_stream_failures() -> Result<()> {
    assert!(generated()?.write_to(&mut FailingWriter).is_err());
    Ok(())
}
