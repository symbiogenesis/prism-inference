#![forbid(unsafe_code)]
//! Synthetic, focused checks of verified native model storage.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use bonsai_gguf::{MetadataArray, MetadataValue, PRISM_DIALECT, VerifiedModel};
use bonsai_prism_runtime::Result;
use bonsai_prism_runtime::model::{Model, ModelMetadata, Tensor};
use sha2::{Digest, Sha256};

const BUDGET: usize = 2 * 1024 * 1024;
static NEXT_ARTIFACT: AtomicU64 = AtomicU64::new(0);

struct Spec<'a> {
    name: &'a str,
    dimensions: &'a [u64],
    kind: u32,
    data: Vec<u8>,
}

struct Artifact {
    path: PathBuf,
    root: PathBuf,
    bytes: Vec<u8>,
}

impl Artifact {
    fn new(tensors: &[Spec<'_>], tokenizer: bool) -> Result<Self> {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        Self::with_timestamp(tensors, tokenizer, nonce)
    }

    fn with_timestamp(tensors: &[Spec<'_>], tokenizer: bool, nonce: u128) -> Result<Self> {
        // Windows clock ticks can be shared by concurrent test fixtures.
        let sequence = NEXT_ARTIFACT.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "SwiftBonsai-NativeModel-{}-{nonce}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&root)?;
        let path = root.join("model.gguf");
        let mut bytes = Vec::new();
        u32_bytes(&mut bytes, 0x4655_4747);
        u32_bytes(&mut bytes, 3);
        u64_bytes(&mut bytes, u64::try_from(tensors.len())?);
        u64_bytes(&mut bytes, if tokenizer { 7 } else { 1 });
        string_metadata(&mut bytes, "general.architecture", "synthetic")?;
        if tokenizer {
            text(&mut bytes, "tokenizer.ggml.tokens")?;
            u32_bytes(&mut bytes, 9);
            u32_bytes(&mut bytes, 8);
            u64_bytes(&mut bytes, 4100);
            for index in 0..4100 {
                text(&mut bytes, &format!("token{index}"))?;
            }
            text(&mut bytes, "tokenizer.ggml.scores")?;
            u32_bytes(&mut bytes, 9);
            u32_bytes(&mut bytes, 6);
            u64_bytes(&mut bytes, 4100);
            for _ in 0..4100 {
                bytes.extend(1.25_f32.to_le_bytes());
            }
            text(&mut bytes, "tokenizer.ggml.token_type")?;
            u32_bytes(&mut bytes, 9);
            u32_bytes(&mut bytes, 5);
            u64_bytes(&mut bytes, 4100);
            for _ in 0..4100 {
                bytes.extend(1_i32.to_le_bytes());
            }
            string_metadata(&mut bytes, "tokenizer.chat_template", &"x".repeat(65_537))?;
            text(&mut bytes, "tokenizer.ggml.bos_token_id")?;
            u32_bytes(&mut bytes, 4);
            u32_bytes(&mut bytes, 3);
            string_metadata(&mut bytes, "tokenizer.ggml.model", "gpt2")?;
        }
        let mut offsets = Vec::new();
        let mut end = 0_usize;
        for tensor in tensors {
            let offset = end.div_ceil(32) * 32;
            offsets.push(offset);
            end = offset + tensor.data.len();
            text(&mut bytes, tensor.name)?;
            u32_bytes(&mut bytes, u32::try_from(tensor.dimensions.len())?);
            for dimension in tensor.dimensions {
                u64_bytes(&mut bytes, *dimension);
            }
            u32_bytes(&mut bytes, tensor.kind);
            u64_bytes(&mut bytes, u64::try_from(offset)?);
        }
        let data_offset = bytes.len().div_ceil(32) * 32;
        bytes.resize(data_offset + end, 0);
        for (tensor, offset) in tensors.iter().zip(offsets) {
            bytes[data_offset + offset..data_offset + offset + tensor.data.len()]
                .copy_from_slice(&tensor.data);
        }
        fs::write(&path, &bytes)?;
        Ok(Self { path, root, bytes })
    }

    fn load(&self, budget: usize) -> Result<Model> {
        Model::load(
            &self.path,
            u64::try_from(self.bytes.len())?,
            &sha(&self.bytes),
            budget,
        )
    }
}

impl Drop for Artifact {
    fn drop(&mut self) {
        // Both paths were created and are owned by this fixture. Avoid recursive
        // cleanup so an unexpected file can never remove another session's work.
        let _ = fs::remove_file(&self.path);
        let _ = fs::remove_dir(&self.root);
    }
}

fn u32_bytes(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend(value.to_le_bytes());
}
fn u64_bytes(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend(value.to_le_bytes());
}
fn text(bytes: &mut Vec<u8>, value: &str) -> Result<()> {
    u64_bytes(bytes, u64::try_from(value.len())?);
    bytes.extend(value.as_bytes());
    Ok(())
}
fn string_metadata(bytes: &mut Vec<u8>, key: &str, value: &str) -> Result<()> {
    text(bytes, key)?;
    u32_bytes(bytes, 8);
    text(bytes, value)
}
fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .into_iter()
        .flat_map(|byte| [byte >> 4, byte & 15])
        .map(|nibble| char::from(b"0123456789abcdef"[usize::from(nibble)]))
        .collect()
}

#[test]
fn metadata_only_loader_preserves_identity_and_skips_executable_storage() -> Result<()> {
    let fixture = Artifact::new(
        &[
            Spec {
                name: "token_embd.weight",
                dimensions: &[1024, 3],
                kind: 24,
                data: vec![0; 3072],
            },
            Spec {
                name: "not_executable",
                dimensions: &[1],
                kind: 0,
                data: f32::NAN.to_le_bytes().to_vec(),
            },
        ],
        false,
    )?;
    let reader = VerifiedModel::open(
        &fixture.path,
        PRISM_DIALECT,
        u64::try_from(fixture.bytes.len())?,
        &sha(&fixture.bytes),
        BUDGET,
    )?;
    let header_budget = reader.retained_bytes();
    drop(reader);
    let model = ModelMetadata::load(
        &fixture.path,
        u64::try_from(fixture.bytes.len())?,
        &sha(&fixture.bytes).to_uppercase(),
        header_budget,
    )?;
    assert_eq!(model.identity, sha(&fixture.bytes));
    assert_eq!(model.vocab_size, 3);
    assert_eq!(
        model.metadata["general.architecture"].as_str(),
        Some("synthetic")
    );
    assert!(fixture.load(BUDGET).is_err());
    assert!(
        ModelMetadata::load(
            &fixture.path,
            u64::try_from(fixture.bytes.len())?,
            &sha(&fixture.bytes),
            header_budget - 1,
        )
        .is_err()
    );
    for (bytes, hash) in [
        (u64::try_from(fixture.bytes.len())? + 1, sha(&fixture.bytes)),
        (u64::try_from(fixture.bytes.len())?, "ab".repeat(32)),
    ] {
        assert!(ModelMetadata::load(&fixture.path, bytes, &hash, BUDGET).is_err());
    }
    Ok(())
}

#[test]
fn metadata_only_loader_requires_a_rank_two_embedding_descriptor() -> Result<()> {
    for (name, dimensions) in [
        ("other", &[1, 3][..]),
        ("token_embd.weight", &[3][..]),
        ("token_embd.weight", &[1, 1, 3][..]),
    ] {
        let fixture = Artifact::new(
            &[Spec {
                name,
                dimensions,
                kind: 24,
                data: vec![0; 3],
            }],
            false,
        )?;
        assert!(
            ModelMetadata::load(
                &fixture.path,
                u64::try_from(fixture.bytes.len())?,
                &sha(&fixture.bytes),
                BUDGET,
            )
            .is_err()
        );
    }
    Ok(())
}

#[test]
fn concurrent_artifacts_with_identical_timestamps_remain_independent() -> Result<()> {
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..8)
            .map(|_| scope.spawn(|| Artifact::with_timestamp(&[], false, 0)))
            .collect();
        let mut artifacts = Vec::new();
        for worker in workers {
            artifacts.push(worker.join().map_err(|_| "fixture worker panicked")??);
        }
        for (index, artifact) in artifacts.iter().enumerate() {
            fs::write(&artifact.path, index.to_string())?;
        }
        drop(artifacts.remove(0));
        for (index, artifact) in artifacts.iter().enumerate() {
            assert_eq!(fs::read_to_string(&artifact.path)?, (index + 1).to_string());
        }
        Ok(())
    })
}

#[test]
fn retains_exact_tokenizer_arrays_templates_and_tensor_dimensions() -> Result<()> {
    let fixture = Artifact::new(
        &[Spec {
            name: "weight",
            dimensions: &[2, 2],
            kind: 0,
            data: [1_f32, 2., 3., 4.]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        }],
        true,
    )?;
    let model = fixture.load(BUDGET)?;
    assert_eq!(model.identity, sha(&fixture.bytes));
    assert!(matches!(&model.metadata["tokenizer.ggml.tokens"],
        MetadataValue::Array(MetadataArray::String(values)) if values[4099] == "token4099"));
    assert!(matches!(&model.metadata["tokenizer.ggml.scores"],
        MetadataValue::Array(MetadataArray::F32(values)) if values[4099].to_bits() == 1.25_f32.to_bits()));
    assert!(matches!(&model.metadata["tokenizer.ggml.token_type"],
        MetadataValue::Array(MetadataArray::I32(values)) if values[4099] == 1));
    assert_eq!(
        model.metadata["tokenizer.ggml.bos_token_id"].as_u64(),
        Some(3)
    );
    assert_eq!(
        model.metadata["tokenizer.chat_template"]
            .as_str()
            .map(str::len),
        Some(65_537)
    );
    assert_eq!(model.tensors["weight"].dimensions, [2, 2]);
    assert_eq!(
        &*model.tensors["weight"].data,
        &[1_f32, 2., 3., 4.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn rejects_wrong_identity_and_cumulative_retained_budget() -> Result<()> {
    let fixture = Artifact::new(
        &[Spec {
            name: "weight",
            dimensions: &[2],
            kind: 0,
            data: vec![0; 8],
        }],
        true,
    )?;
    assert!(
        Model::load(
            &fixture.path,
            u64::try_from(fixture.bytes.len())? + 1,
            &sha(&fixture.bytes),
            BUDGET
        )
        .is_err()
    );
    assert!(
        Model::load(
            &fixture.path,
            u64::try_from(fixture.bytes.len())?,
            &"0".repeat(64),
            BUDGET
        )
        .is_err()
    );
    assert!(fixture.load(1024).is_err());
    let reader = VerifiedModel::open(
        &fixture.path,
        PRISM_DIALECT,
        u64::try_from(fixture.bytes.len())?,
        &sha(&fixture.bytes),
        BUDGET,
    )?;
    assert!(fixture.load(reader.retained_bytes() + 7).is_err());
    Ok(())
}

#[test]
fn reads_only_budgeted_payloads_through_the_verified_handle() -> Result<()> {
    let fixture = Artifact::new(
        &[Spec {
            name: "weight",
            dimensions: &[2],
            kind: 30,
            data: vec![0x80, 0x3f, 0, 0x40],
        }],
        false,
    )?;
    let mut reader = VerifiedModel::open(
        &fixture.path,
        PRISM_DIALECT,
        u64::try_from(fixture.bytes.len())?,
        &sha(&fixture.bytes),
        BUDGET,
    )?;
    assert_eq!(reader.tensors()[0].dimensions, [2]);
    assert!(reader.read_tensor(0, 3).is_err());
    assert!(reader.read_tensor(1, BUDGET).is_err());
    #[cfg(windows)]
    assert!(
        fs::OpenOptions::new()
            .write(true)
            .open(&fixture.path)
            .is_err()
    );
    assert_eq!(reader.read_tensor(0, 4)?, [0x80, 0x3f, 0, 0x40]);
    Ok(())
}

#[test]
fn accepts_f16_bf16_and_the_two_exact_prism_encodings() -> Result<()> {
    let mut pq = [0x55; 34];
    pq[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
    let ptq = bonsai_ternary::pack_ptq(&pq)?;
    let fixture = Artifact::new(
        &[
            Spec {
                name: "f16",
                dimensions: &[2],
                kind: 1,
                data: vec![0, 0x3c, 0, 0xbc],
            },
            Spec {
                name: "bf16",
                dimensions: &[2],
                kind: 30,
                data: vec![0x80, 0x3f, 0x80, 0xbf],
            },
            Spec {
                name: "pq",
                dimensions: &[128],
                kind: 142,
                data: pq.to_vec(),
            },
            Spec {
                name: "ptq",
                dimensions: &[128],
                kind: 143,
                data: ptq.to_vec(),
            },
        ],
        false,
    )?;
    assert_eq!(fixture.load(BUDGET)?.tensors.len(), 4);
    Ok(())
}

#[test]
fn accepts_signed_zero_prism_scales_without_altering_stored_bytes() -> Result<()> {
    for scale in [0_u16, 0x8000] {
        let mut pq = [0; 34];
        pq[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        let mut ptq = bonsai_ternary::pack_ptq(&pq)?;
        pq[..2].copy_from_slice(&scale.to_le_bytes());
        ptq[26..].copy_from_slice(&scale.to_le_bytes());
        let fixture = Artifact::new(
            &[
                Spec {
                    name: "pq",
                    dimensions: &[128],
                    kind: 142,
                    data: pq.to_vec(),
                },
                Spec {
                    name: "ptq",
                    dimensions: &[128],
                    kind: 143,
                    data: ptq.to_vec(),
                },
            ],
            false,
        )?;
        let model = fixture.load(BUDGET)?;
        assert_eq!(&*model.tensors["pq"].data, &pq);
        assert_eq!(&*model.tensors["ptq"].data, &ptq);
        let expected = -bonsai_ternary::half_to_float(scale);
        for name in ["pq", "ptq"] {
            let decoded = bonsai_prism_runtime::kernels::row(&model.tensors[name], 0)?;
            assert_eq!(decoded.len(), 128);
            assert!(
                decoded
                    .iter()
                    .all(|value| value.to_bits() == expected.to_bits())
            );
        }
    }
    Ok(())
}

#[test]
fn accepts_all_native_pq2_codes_with_finite_scales() -> Result<()> {
    for scale in [0x3c00_u16, 0xbc00, 0, 0x8000] {
        // Each byte contains native codes 0, 1, 2 and 3, in order.
        let mut block = [0xe4; 34];
        block[..2].copy_from_slice(&scale.to_le_bytes());
        let data = [block, block].concat();
        let fixture = Artifact::new(
            &[Spec {
                name: "native",
                dimensions: &[256],
                kind: 142,
                data: data.clone(),
            }],
            false,
        )?;
        let model = fixture.load(BUDGET)?;
        let tensor = &model.tensors["native"];
        assert_eq!(&*tensor.data, &data);
        let decoded = bonsai_prism_runtime::kernels::row(tensor, 0)?;
        for (index, actual) in decoded.iter().enumerate() {
            let symbol = [-1.0_f32, 0.0, 1.0, 2.0][index % 4];
            let expected = symbol * bonsai_ternary::half_to_float(scale);
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        if scale & 0x7fff != 0 {
            let mut input = [0.0; 256];
            input[3] = 127.0;
            let output = bonsai_prism_runtime::kernels::matvec(tensor, &input)?;
            let expected = 254.0 * bonsai_ternary::half_to_float(scale);
            assert_eq!(output[0].to_bits(), expected.to_bits());
        }
    }
    Ok(())
}

#[test]
fn rejects_nonfinite_prism_scales_and_accepts_native_zero_scale_ptq1_aliases() -> Result<()> {
    let mut source = [0; 34];
    source[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
    let canonical = bonsai_ternary::pack_ptq(&source)?;
    for scale in [0x7c00_u16, 0xfc00, 0x7e00, 0xfe00] {
        let mut pq = source;
        pq[..2].copy_from_slice(&scale.to_le_bytes());
        let mut ptq = canonical;
        ptq[26..].copy_from_slice(&scale.to_le_bytes());
        for (kind, data) in [(142, pq.to_vec()), (143, ptq.to_vec())] {
            assert!(
                Artifact::new(
                    &[Spec {
                        name: "nonfinite",
                        dimensions: &[128],
                        kind,
                        data,
                    }],
                    false,
                )?
                .load(BUDGET)
                .is_err()
            );
        }
    }
    for scale in [0_u16, 0x8000] {
        let mut ptq = canonical;
        ptq[26..].copy_from_slice(&scale.to_le_bytes());
        // Runtime tensor decoding accepts fractional aliases at zero scale.
        ptq[25] = 1;
        assert!(
            Artifact::new(
                &[Spec {
                    name: "malformed",
                    dimensions: &[128],
                    kind: 143,
                    data: ptq.to_vec(),
                }],
                false,
            )?
            .load(BUDGET)
            .is_ok()
        );
    }
    Ok(())
}

#[test]
fn rejects_nonfinite_and_invalid_executable_payloads() -> Result<()> {
    for (kind, data) in [
        (0, f32::NAN.to_le_bytes().to_vec()),
        (1, 0x7c00_u16.to_le_bytes().to_vec()),
        (30, 0x7f80_u16.to_le_bytes().to_vec()),
    ] {
        assert!(
            Artifact::new(
                &[Spec {
                    name: "bad",
                    dimensions: &[1],
                    kind,
                    data
                }],
                false
            )?
            .load(BUDGET)
            .is_err()
        );
    }
    let mut pq = [0xff; 34];
    pq[..2].copy_from_slice(&0x7c00_u16.to_le_bytes());
    assert!(
        Artifact::new(
            &[Spec {
                name: "bad",
                dimensions: &[128],
                kind: 142,
                data: pq.to_vec()
            }],
            false
        )?
        .load(BUDGET)
        .is_err()
    );
    let mut ptq = [0; 28];
    ptq[0] = 1;
    // Arbitrary native PTQ bytes are admitted; an infinite scale is invalid.
    ptq[26..].copy_from_slice(&0x7c00_u16.to_le_bytes());
    assert!(
        Artifact::new(
            &[Spec {
                name: "bad",
                dimensions: &[128],
                kind: 143,
                data: ptq.to_vec()
            }],
            false
        )?
        .load(BUDGET)
        .is_err()
    );
    assert!(
        Artifact::new(
            &[Spec {
                name: "unsupported",
                dimensions: &[1],
                kind: 24,
                data: vec![0]
            }],
            false
        )?
        .load(BUDGET)
        .is_err()
    );
    Ok(())
}

#[test]
fn rejects_duplicate_descriptors_truncation_and_dimension_overflow() -> Result<()> {
    assert!(
        Artifact::new(
            &[
                Spec {
                    name: "same",
                    dimensions: &[1],
                    kind: 0,
                    data: vec![0; 4]
                },
                Spec {
                    name: "same",
                    dimensions: &[1],
                    kind: 0,
                    data: vec![0; 4]
                },
            ],
            false
        )?
        .load(BUDGET)
        .is_err()
    );
    let mut fixture = Artifact::new(
        &[Spec {
            name: "short",
            dimensions: &[2],
            kind: 0,
            data: vec![0; 8],
        }],
        false,
    )?;
    fixture.bytes.pop();
    fs::write(&fixture.path, &fixture.bytes)?;
    assert!(fixture.load(BUDGET).is_err());
    assert!(
        Artifact::new(
            &[Spec {
                name: "huge",
                dimensions: &[i64::MAX.unsigned_abs(), 3],
                kind: 0,
                data: Vec::new()
            }],
            false
        )?
        .load(BUDGET)
        .is_err()
    );
    Ok(())
}

#[test]
fn lora_b_sharing_preserves_layout_bytes_and_owner_lifetime() {
    let tensor = |name: &str, dimensions: Vec<usize>, value: u8| Tensor {
        name: name.into(),
        dimensions,
        kind: 0,
        data: Arc::from(vec![value; 8]),
    };
    let mut model = Model {
        metadata: BTreeMap::new(),
        identity: String::new(),
        tensors: [
            ("a.lora_b".into(), tensor("a.lora_b", vec![1, 2], 1)),
            ("b.lora_b".into(), tensor("b.lora_b", vec![1, 2], 1)),
            ("c.lora_b".into(), tensor("c.lora_b", vec![2, 1], 1)),
            ("d.lora_b".into(), tensor("d.lora_b", vec![1, 2], 2)),
            ("other".into(), tensor("other", vec![1, 2], 1)),
        ]
        .into(),
    };
    assert_eq!(model.share_lora_b(), 1);
    assert!(Arc::ptr_eq(
        &model.tensors["a.lora_b"].data,
        &model.tensors["b.lora_b"].data
    ));
    assert!(!Arc::ptr_eq(
        &model.tensors["a.lora_b"].data,
        &model.tensors["c.lora_b"].data
    ));
    let saved = Arc::clone(&model.tensors["b.lora_b"].data);
    drop(model);
    assert_eq!(&*saved, &[1; 8]);
}
