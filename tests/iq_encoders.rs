//! Focused pinned `IQ2_S` and IQ3 reference/importance encoder observations.

use bonsai_prism_runtime::{Result, codec, kernels};
use serde_json::Value;

fn persistent_reference(kind: u32, input: &[u8], expected: &[u8]) -> Result<()> {
    let mut owner = codec::CodecBuffers::default();
    let buffer = owner.new_buffer(input.len() + 3, 32, 1)?;
    let source = codec::BufferRegion { buffer, offset: 1 };
    let destination = codec::BufferRegion { buffer, offset: 3 };
    owner.write(source, input)?;
    owner.call(
        codec::CodecFunction::FromFloatReference,
        kind,
        input.len() / 4,
        source,
        None,
        destination,
    )?;
    assert_eq!(owner.read(destination, expected.len())?, expected);
    Ok(())
}

fn bytes(value: &Value) -> Result<Vec<u8>> {
    let text = value
        .as_str()
        .ok_or_else(|| std::io::Error::other("IQ encoder fixture hex absent"))?;
    if !text.len().is_multiple_of(2) {
        return Err(std::io::Error::other("IQ encoder fixture hex is truncated").into());
    }
    text.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| Ok(u8::from_str_radix(std::str::from_utf8(pair)?, 16)?))
        .collect()
}

fn floats(value: &Value) -> Result<Vec<f32>> {
    let bytes = bytes(value)?;
    if !bytes.len().is_multiple_of(4) {
        return Err(std::io::Error::other("IQ encoder float extent is truncated").into());
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect())
}

#[test]
fn pinned_reference_and_importance_encoders_preserve_every_byte() -> Result<()> {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/upstream-iq-encoders.json"))?;
    assert_eq!(
        fixture["UpstreamCommit"],
        "b87ca7225456b06b0975b8239c66b5aecf07e37c"
    );
    let cases = fixture["Cases"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("IQ encoder cases absent"))?;
    assert_eq!(cases.len(), 144);
    let inputs = fixture["Inputs"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("IQ encoder input rows absent"))?;
    assert_eq!(inputs.len(), 12);
    for case in cases {
        let kind = u32::try_from(
            case["TypeId"]
                .as_u64()
                .ok_or_else(|| std::io::Error::other("IQ encoder fixture type absent"))?,
        )?;
        let shape = usize::try_from(
            case["Shape"]
                .as_u64()
                .ok_or_else(|| std::io::Error::other("IQ encoder fixture shape absent"))?,
        )?;
        let row = inputs
            .get(shape)
            .ok_or_else(|| std::io::Error::other("IQ encoder input row absent"))?;
        assert_eq!(row["Shape"], case["Shape"]);
        let input = floats(&row["InputHex"])?;
        let packed = match case["Mode"].as_u64() {
            Some(0) => kernels::quantize_row(kind, &input)?,
            Some(1) => {
                kernels::quantize_row_weighted(kind, &input, &floats(&row["ImportanceHex"])?)?
            }
            Some(2) => kernels::quantize_row_weighted(kind, &input, &vec![0.0; input.len()])?,
            Some(3) => {
                let importance = (0..input.len())
                    .map(|index| if index % 8 == 0 { 1.0 } else { 0.0 })
                    .collect::<Vec<_>>();
                kernels::quantize_row_weighted(kind, &input, &importance)?
            }
            _ => return Err(std::io::Error::other("IQ encoder fixture mode absent").into()),
        };
        let expected = bytes(&case["OutputHex"])?;
        assert_eq!(
            packed, expected,
            "type {kind}, shape {}, importance mode {}",
            case["Shape"], case["Mode"]
        );
        if case["Mode"] == 0 {
            persistent_reference(kind, &bytes(&row["InputHex"])?, &expected)?;
        }
    }
    Ok(())
}

#[test]
fn iq_fits_reject_invalid_domains_and_preserve_cpu_reference_distinction() -> Result<()> {
    for kind in [18, 21, 22] {
        let input = [1.0_f32; 256];
        let weights = [1.0_f32; 256];
        let traits = codec::traits(kind)?;
        assert!(traits.native.from_float_reference);
        assert!(traits.rust.from_float_reference);
        assert!(!traits.native.from_float_cpu);
        assert!(!traits.rust.from_float_cpu);
        assert!(kernels::quantize_row(kind, &[]).is_err());
        assert!(kernels::quantize_row(kind, &input[..255]).is_err());
        assert!(kernels::quantize_row(kind, &[f32::INFINITY; 256]).is_err());
        assert!(kernels::quantize_row(kind, &[f32::NAN; 256]).is_err());
        assert!(kernels::quantize_row(kind, &[f32::MAX; 256]).is_err());
        assert!(kernels::quantize_row_weighted(kind, &input, &weights[..255]).is_err());
        assert!(kernels::quantize_row_weighted(kind, &input, &[-1.0; 256]).is_err());
        assert!(kernels::quantize_row_weighted(kind, &input, &[f32::INFINITY; 256]).is_err());
        assert!(kernels::quantize_row_weighted(kind, &input, &[f32::NAN; 256]).is_err());
        assert!(kernels::quantize_row_weighted(kind, &input, &[f32::MAX; 256]).is_err());
    }
    for kind in [16, 17, 19, 29] {
        assert!(!codec::traits(kind)?.rust.from_float_reference);
        assert!(kernels::quantize_row(kind, &[1.0; 256]).is_err());
    }
    Ok(())
}
