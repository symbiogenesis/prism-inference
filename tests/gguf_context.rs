//! Focused mutable GGUF ownership, native byte coercion and rejection contracts.
#![forbid(unsafe_code)]

use bonsai_prism_runtime::model::gguf_context::{Context, Entry, tensor_bytes};

type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

fn string(output: &mut Vec<u8>, value: &[u8]) -> TestResult {
    output.extend(u64::try_from(value.len())?.to_le_bytes());
    output.extend(value);
    Ok(())
}

fn header(tensors: u64, keys: u64) -> Vec<u8> {
    let mut output = b"GGUF".to_vec();
    output.extend(3_u32.to_le_bytes());
    output.extend(tensors.to_le_bytes());
    output.extend(keys.to_le_bytes());
    output
}

#[test]
fn replacement_copy_and_roundtrip_preserve_independent_tensor_storage() -> TestResult {
    let entry = |key: &str, value| Entry {
        key: key.to_owned(),
        kind: 0,
        array_kind: None,
        count: 1,
        data: vec![value],
        strings: vec![],
    };
    let mut context = Context::new();
    context.set_entry(entry("first", 1))?;
    context.set_entry(entry("second", 2))?;
    context.set_entry(entry("first", 3))?;
    assert_eq!(context.entries()[0].key, "second");
    assert_eq!(context.entries()[1].key, "first");
    context.add_tensor("weight".to_owned(), 2, [32, 1, 1, 1], vec![0x81; 18])?;
    context.add_tensor("bias".to_owned(), 0, [1, 1, 1, 1], vec![0, 0, 128, 63])?;
    assert_eq!(context.tensors()[1].offset, 32);
    let bytes = context.encode(false)?;
    let loaded = Context::read_buffer(&bytes, true)?;
    assert_eq!(loaded.entries(), context.entries());
    assert_eq!(loaded.tensors(), context.tensors());
    assert_eq!(loaded.blob().ok_or("missing blob")?.len(), 64);
    let mut destination = Context::new();
    destination.set_entry(entry("retained", 9))?;
    destination.set_entry(entry("second", 7))?;
    destination.copy_metadata(&loaded)?;
    assert_eq!(destination.entries()[0], entry("retained", 9));
    assert_eq!(destination.entries()[1..], *loaded.entries());
    assert_eq!(destination.tensors(), []);
    assert_eq!(loaded.entries()[0], entry("second", 2));
    let metadata = Context::read_buffer(&context.encode(true)?, false)?;
    assert!(!metadata.has_tensor_context());
    assert_eq!(metadata.tensors()[0].data, None);
    assert!(metadata.encode(false).is_err());
    Ok(())
}

#[test]
fn native_boolean_reads_normalize_and_metadata_copy_truncates_c_strings() -> TestResult {
    let mut bytes = header(0, 3);
    string(&mut bytes, b"scalar")?;
    bytes.extend(7_u32.to_le_bytes());
    bytes.push(255);
    string(&mut bytes, b"array")?;
    bytes.extend(9_u32.to_le_bytes());
    bytes.extend(7_u32.to_le_bytes());
    bytes.extend(3_u64.to_le_bytes());
    bytes.extend([0, 2, 128]);
    string(&mut bytes, b"string")?;
    bytes.extend(8_u32.to_le_bytes());
    string(&mut bytes, b"prefix\0suffix")?;
    let context = Context::read_buffer(&bytes, false)?;
    assert_eq!(context.entries()[0].data, [1]);
    assert_eq!(context.entries()[1].data, [0, 1, 1]);
    assert_eq!(context.entries()[2].strings, [b"prefix\0suffix".to_vec()]);
    let mut destination = Context::new();
    destination.copy_metadata(&context)?;
    assert_eq!(destination.entries()[2].strings, [b"prefix".to_vec()]);
    let copied = Context::read_buffer(&destination.encode(true)?, false)?;
    assert_eq!(copied.entries(), destination.entries());
    Ok(())
}

#[test]
fn malformed_offsets_shapes_payloads_and_short_callbacks_are_rejected() -> TestResult {
    let mut bytes = header(1, 0);
    string(&mut bytes, b"q")?;
    bytes.extend(1_u32.to_le_bytes());
    let dimension_offset = bytes.len();
    bytes.extend(32_u64.to_le_bytes());
    bytes.extend(2_u32.to_le_bytes());
    let offset_offset = bytes.len();
    bytes.extend(0_u64.to_le_bytes());
    bytes.resize(bytes.len().next_multiple_of(32), 0);
    let metadata = bytes.len();
    bytes.extend([0x55; 32]);
    assert_eq!(Context::read_buffer(&bytes, true)?.tensors()[0].bytes, 18);
    assert!(Context::read_buffer(&bytes[..bytes.len() - 1], true).is_err());
    assert!(Context::read_buffer(&bytes[..metadata], false).is_ok());
    let mut bad_offset = bytes.clone();
    bad_offset[offset_offset..offset_offset + 8].copy_from_slice(&32_u64.to_le_bytes());
    assert!(Context::read_buffer(&bad_offset, false).is_err());
    let mut bad_shape = bytes.clone();
    bad_shape[dimension_offset..dimension_offset + 8].copy_from_slice(&33_u64.to_le_bytes());
    assert!(Context::read_buffer(&bad_shape, false).is_err());
    assert!(tensor_bytes(4, [32, 1, 1, 1]).is_err());
    assert!(tensor_bytes(0, [u64::MAX, 1, 1, 1]).is_err());
    assert_eq!(tensor_bytes(0, [0, 1, 1, 1])?, 0);
    assert!(
        Context::read_callback(u64::try_from(bytes.len())?, true, |offset, output| {
            let offset = usize::try_from(offset)?;
            let count = output.len() - 1;
            output[..count].copy_from_slice(&bytes[offset..offset + count]);
            Ok(count)
        })
        .is_err()
    );
    let loaded = Context::read_callback(u64::try_from(bytes.len())?, true, |offset, output| {
        assert!(output.len() <= 4096);
        let offset = usize::try_from(offset)?;
        output.copy_from_slice(&bytes[offset..offset + output.len()]);
        Ok(output.len())
    })?;
    assert_eq!(
        loaded.tensors()[0].data.as_deref(),
        Some([0x55; 18].as_slice())
    );
    Ok(())
}
