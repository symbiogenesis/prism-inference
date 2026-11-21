//! Integration tests for bounded cancellable `Read + Seek` GGUF index parsing.
#![expect(
    clippy::unwrap_used,
    reason = "Integration tests construct synthetic GGUF streams and assert expected fields."
)]

use std::io::{self, Cursor};

use bonsai_gguf::{
    IndexPolicy, IndexValue, MetadataArray, MetadataValue, ParsedIndex, read_index,
    read_index_cancellable,
};

fn push_bytes_str(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&u64::try_from(bytes.len()).unwrap().to_le_bytes());
    out.extend_from_slice(bytes);
}

fn kv_u32(val: u32) -> Vec<u8> {
    [4_u32.to_le_bytes().as_slice(), &val.to_le_bytes()].concat()
}

fn kv_i32(val: i32) -> Vec<u8> {
    [5_u32.to_le_bytes().as_slice(), &val.to_le_bytes()].concat()
}

fn kv_bool(val: u8) -> Vec<u8> {
    [7_u32.to_le_bytes().as_slice(), &[val]].concat()
}

fn kv_str_array(items: &[&[u8]]) -> Vec<u8> {
    let mut out = [9_u32.to_le_bytes(), 8_u32.to_le_bytes()].concat();
    out.extend_from_slice(&u64::try_from(items.len()).unwrap().to_le_bytes());
    for item in items {
        push_bytes_str(&mut out, item);
    }
    out
}

fn kv_i32_array(items: &[i32]) -> Vec<u8> {
    let mut out = [9_u32.to_le_bytes(), 5_u32.to_le_bytes()].concat();
    out.extend_from_slice(&u64::try_from(items.len()).unwrap().to_le_bytes());
    for &item in items {
        out.extend_from_slice(&item.to_le_bytes());
    }
    out
}

fn build_gguf(
    version: u32,
    alignment: usize,
    kv: &[(&[u8], Vec<u8>)],
    tensors: &[(&[u8], &[u64], u32, u64)],
    data_bytes: usize,
) -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(&u64::try_from(tensors.len()).unwrap().to_le_bytes());
    out.extend_from_slice(&u64::try_from(kv.len()).unwrap().to_le_bytes());
    for (k, v) in kv {
        push_bytes_str(&mut out, k);
        out.extend_from_slice(v);
    }
    for &(name, shape, kind, offset) in tensors {
        push_bytes_str(&mut out, name);
        out.extend_from_slice(&u32::try_from(shape.len()).unwrap().to_le_bytes());
        for &dim in shape {
            out.extend_from_slice(&dim.to_le_bytes());
        }
        out.extend_from_slice(&kind.to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
    }
    if !tensors.is_empty() {
        out.resize(out.len().next_multiple_of(alignment) + data_bytes, 0);
    }
    out
}

const fn asr_policy<'a>() -> IndexPolicy<'a> {
    IndexPolicy {
        allow_v2: false,
        allow_zero_tensors: false,
        allow_zero_metadata: false,
        max_file_bytes: 8 * 1024 * 1024 * 1024,
        max_tensors: 8192,
        max_metadata: 65_536,
        metadata_budget: 64 * 1024 * 1024,
        max_key_bytes: 1024,
        max_string_bytes: 65_536,
        max_array_string_bytes: 4096,
        max_retained_string_array_bytes: None,
        max_array_items: 262_144,
        max_alignment: 4096,
        allow_positive_i32_alignment: false,
        reject_nonfinite_floats: false,
        reject_interior_nul_keys: true,
        reject_interior_nul_strings: false,
        kv_overhead_bytes: 128,
        string_overhead_bytes: 32,
        tensor_overhead_bytes: 512,
        charge_string_array_slots: true,
        charge_skipped_scalar_arrays: true,
        prism_dialect: false,
        allowed_tensor_types: Some(&[0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 30]),
        ..IndexPolicy::new()
    }
}

fn tts_retain(key: &str) -> bool {
    matches!(key, "tokenizer.ggml.tokens" | "tokenizer.ggml.merges")
}

const fn tts_scalar_array(_key: &str, element_type_id: u32) -> bool {
    matches!(element_type_id, 4 | 5)
}

fn tts_policy<'a>() -> IndexPolicy<'a> {
    IndexPolicy {
        allow_v2: true,
        allow_zero_tensors: false,
        allow_zero_metadata: true,
        max_file_bytes: 16 * 1024 * 1024 * 1024,
        max_tensors: 131_072,
        max_metadata: 131_072,
        metadata_budget: 128 * 1024 * 1024,
        max_key_bytes: 65_536,
        max_string_bytes: 16 * 1024 * 1024,
        max_array_string_bytes: 16 * 1024 * 1024,
        max_retained_string_array_bytes: Some(64 * 1024 * 1024),
        max_array_items: 1_000_000,
        max_alignment: 1_048_576,
        allow_positive_i32_alignment: true,
        reject_nonfinite_floats: true,
        reject_interior_nul_keys: true,
        reject_interior_nul_strings: true,
        kv_overhead_bytes: 0,
        string_overhead_bytes: 0,
        tensor_overhead_bytes: 0,
        charge_string_array_slots: false,
        charge_skipped_scalar_arrays: false,
        prism_dialect: false,
        allowed_tensor_types: Some(&[0, 1, 8, 12, 14]),
        retain_string_array: tts_retain,
        retain_scalar_array: tts_scalar_array,
    }
}

#[test]
fn stream_index_preserves_precancelled_cursor_and_recovers_after_interruption() -> io::Result<()> {
    let mut source = Cursor::new(vec![0_u8; 32]);
    source.set_position(7);
    let policy = asr_policy();
    let err = read_index_cancellable(&mut source, 32, &policy, &mut || true).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::Interrupted);
    assert_eq!(source.position(), 7);

    let tokens = vec![b"literal".as_slice(); 200];
    let bytes = build_gguf(
        3,
        32,
        &[(b"tokens", kv_str_array(&tokens))],
        &[(b"weight", &[1], 0, 0)],
        4,
    );
    let len = u64::try_from(bytes.len()).unwrap();
    let mut stream = Cursor::new(bytes);
    let mut polls = 0_usize;
    let interrupted = read_index_cancellable(&mut stream, len, &policy, &mut || {
        polls += 1;
        polls == 50
    })
    .unwrap_err();
    assert_eq!(interrupted.kind(), io::ErrorKind::Interrupted);
    assert_eq!(polls, 50);

    let parsed: ParsedIndex = read_index(&mut stream, len, &policy)?;
    assert_eq!(parsed.tensors["weight"].bytes, 4);
    assert!(matches!(
        parsed.metadata.get("tokens").and_then(IndexValue::as_metadata),
        Some(MetadataValue::Array(MetadataArray::String(items))) if items.len() == 200
    ));
    Ok(())
}

#[test]
fn stream_index_validates_layout_skips_arrays_and_enforces_budgets() -> io::Result<()> {
    let bytes = build_gguf(
        3,
        32,
        &[
            (b"unrelated", kv_u32(7)),
            (b"skipped_i32", kv_i32_array(&[1, 2, 3, 4])),
        ],
        &[(b"q8", &[32, 2], 8, 32), (b"f32", &[3], 0, 0)],
        100,
    );
    let policy = asr_policy();
    let parsed = read_index(
        &mut Cursor::new(&bytes),
        u64::try_from(bytes.len()).unwrap(),
        &policy,
    )?;
    assert_eq!(parsed.tensors["q8"].bytes, 68);
    assert_eq!(parsed.tensors["f32"].bytes, 12);
    assert_eq!(
        parsed.tensors["q8"].offset - parsed.tensors["f32"].offset,
        32
    );
    assert_eq!(parsed.metadata["skipped_i32"].skipped_array(), Some((5, 4)));

    // Charges empty string arrays and skipped scalar arrays against remaining_budget.
    let small_budget = IndexPolicy {
        metadata_budget: 63,
        allow_zero_metadata: true,
        ..asr_policy()
    };
    let empty_strs = build_gguf(
        3,
        32,
        &[(b"t", kv_str_array(&[b"", b""]))],
        &[(b"w", &[1], 0, 0)],
        4,
    );
    assert!(
        read_index(
            &mut Cursor::new(&empty_strs),
            u64::try_from(empty_strs.len()).unwrap(),
            &small_budget
        )
        .is_err()
    );

    // Rejects overlapping, unaligned, truncated, fractional, disallowed type, duplicate key, nested array.
    for bad in [
        build_gguf(
            3,
            32,
            &[(b"k", kv_u32(1))],
            &[(b"a", &[2], 0, 0), (b"b", &[32], 8, 0)],
            100,
        ),
        build_gguf(3, 32, &[(b"k", kv_u32(1))], &[(b"b", &[32], 8, 1)], 100),
        build_gguf(3, 32, &[(b"k", kv_u32(1))], &[(b"b", &[32], 8, 32)], 65),
        build_gguf(3, 32, &[(b"k", kv_u32(1))], &[(b"b", &[31], 8, 32)], 100),
        build_gguf(3, 32, &[(b"k", kv_u32(1))], &[(b"b", &[128], 142, 0)], 100),
        build_gguf(
            3,
            32,
            &[(b"k", kv_u32(1)), (b"k", kv_u32(2))],
            &[(b"w", &[1], 0, 0)],
            4,
        ),
        build_gguf(
            3,
            32,
            &[(
                b"nested",
                [
                    9_u32.to_le_bytes().as_slice(),
                    &9_u32.to_le_bytes(),
                    &0_u64.to_le_bytes(),
                ]
                .concat(),
            )],
            &[(b"w", &[1], 0, 0)],
            4,
        ),
    ] {
        assert!(
            read_index(
                &mut Cursor::new(&bad),
                u64::try_from(bad.len()).unwrap(),
                &policy
            )
            .is_err()
        );
    }
    Ok(())
}

#[test]
fn stream_index_tts_policy_retains_selectively_and_validates_skipped_values() -> io::Result<()> {
    let policy = tts_policy();
    let tokens = kv_str_array(&[b"alpha", b"beta"]);
    let ignored = kv_str_array(&[b"ignored"]);
    let i32_arr = kv_i32_array(&[8, 5, 4, 3]);
    let g2 = build_gguf(
        2,
        1_048_576,
        &[
            (b"general.alignment", kv_i32(1_048_576)),
            (b"tokenizer.ggml.tokens", tokens),
            (b"other.strings", ignored),
            (b"codec.rates", i32_arr),
            (b"flag", kv_bool(1)),
        ],
        &[(b"w", &[1], 0, 0)],
        4,
    );
    let parsed = read_index(
        &mut Cursor::new(&g2),
        u64::try_from(g2.len()).unwrap(),
        &policy,
    )?;
    assert_eq!(parsed.version, 2);
    assert_eq!(parsed.alignment, 1_048_576);
    assert_eq!(
        parsed.metadata["other.strings"].skipped_array(),
        Some((8, 1))
    );
    assert!(matches!(
        parsed.metadata["tokenizer.ggml.tokens"].as_metadata(),
        Some(MetadataValue::Array(MetadataArray::String(v))) if v == &["alpha", "beta"]
    ));
    assert!(matches!(
        parsed.metadata["codec.rates"].as_metadata(),
        Some(MetadataValue::Array(MetadataArray::I32(v))) if v == &[8, 5, 4, 3]
    ));

    // Skipped string arrays still reject interior NUL and invalid UTF-8.
    for bad_item in [b"bad\0nul".as_slice(), b"\xfe\xff".as_slice()] {
        let bad = build_gguf(
            3,
            32,
            &[(b"other.strings", kv_str_array(&[bad_item]))],
            &[(b"w", &[1], 0, 0)],
            4,
        );
        assert!(
            read_index(
                &mut Cursor::new(&bad),
                u64::try_from(bad.len()).unwrap(),
                &policy
            )
            .is_err()
        );
    }
    Ok(())
}

#[test]
fn stream_index_rejects_bad_booleans_alignment_and_zero_tensors() -> io::Result<()> {
    let policy = tts_policy();
    // Bad booleans (scalar and skipped array) and invalid alignment are rejected.
    let bad_bool_arr = [
        9_u32.to_le_bytes().as_slice(),
        &7_u32.to_le_bytes(),
        &2_u64.to_le_bytes(),
        &[0_u8, 255],
    ]
    .concat();
    for bad in [
        build_gguf(3, 32, &[(b"flag", kv_bool(2))], &[(b"w", &[1], 0, 0)], 4),
        build_gguf(3, 32, &[(b"flags", bad_bool_arr)], &[(b"w", &[1], 0, 0)], 4),
        build_gguf(
            3,
            32,
            &[(b"general.alignment", kv_i32(-32))],
            &[(b"w", &[1], 0, 0)],
            4,
        ),
        build_gguf(
            3,
            32,
            &[(b"general.alignment", kv_u32(3))],
            &[(b"w", &[1], 0, 0)],
            4,
        ),
        build_gguf(
            3,
            32,
            &[(b"general.alignment", kv_u32(2_097_152))],
            &[(b"w", &[1], 0, 0)],
            4,
        ),
    ] {
        assert!(
            read_index(
                &mut Cursor::new(&bad),
                u64::try_from(bad.len()).unwrap(),
                &policy
            )
            .is_err()
        );
    }

    // 24-byte zero-tensor GGUF rejected by default, admitted with allow_zero_tensors.
    let zero_tensor = build_gguf(3, 32, &[], &[], 0);
    assert_eq!(zero_tensor.len(), 24);
    assert!(read_index(&mut Cursor::new(&zero_tensor), 24, &policy).is_err());
    let allow_zero = IndexPolicy {
        allow_zero_tensors: true,
        ..policy
    };
    let z = read_index(&mut Cursor::new(&zero_tensor), 24, &allow_zero)?;
    assert!(z.tensors.is_empty());
    assert_eq!(z.data_offset, 24);
    Ok(())
}

#[test]
fn read_index_cancellable_accepts_concrete_closure_and_dyn_callback() -> io::Result<()> {
    let bytes = build_gguf(3, 32, &[(b"k", kv_u32(1))], &[(b"weight", &[1], 0, 0)], 4);
    let len = u64::try_from(bytes.len()).unwrap();
    let policy = asr_policy();

    let mut concrete_polls = 0_usize;
    let parsed_concrete =
        read_index_cancellable(&mut Cursor::new(&bytes), len, &policy, &mut || {
            concrete_polls += 1;
            false
        })?;
    assert_eq!(parsed_concrete.tensors["weight"].bytes, 4);
    assert!(concrete_polls > 0);

    let mut dyn_polls = 0_usize;
    let mut dyn_cb = || {
        dyn_polls += 1;
        false
    };
    let cancel_dyn: &mut dyn FnMut() -> bool = &mut dyn_cb;
    let parsed_dyn = read_index_cancellable(&mut Cursor::new(&bytes), len, &policy, cancel_dyn)?;
    assert_eq!(parsed_dyn.tensors["weight"].bytes, 4);
    assert!(dyn_polls > 0);
    Ok(())
}
