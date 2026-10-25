#![forbid(unsafe_code)]
//! Focused scalar kernel fixtures pinned to Prism revision
//! `73bfe6cc485554822f80c473a3c0c4f25e80f41b`.
//! Expected quantized results below follow the C reference integer sums,
//! independently of float row decoding, and deliberately differ from float dots.

#![allow(
    clippy::float_cmp,
    reason = "Fixtures require exact representable reference values."
)]

use std::sync::Arc;

use bonsai_prism_runtime::kernels::{
    MatvecBackend, MatvecScratch, hadamard, matvec, matvec_into, rms_norm, rms_norm_into,
    round_f16, row, row_into, softmax, validate_tensor,
};
use bonsai_prism_runtime::model::Tensor;
use bonsai_ternary::pack_ptq;

type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

fn tensor(kind: u32, dimensions: Vec<usize>, data: Vec<u8>) -> Tensor {
    Tensor {
        name: "fixture".to_owned(),
        dimensions,
        kind,
        data: Arc::from(data),
    }
}

fn pq2(scale: u16, symbols: &[(usize, u8)]) -> [u8; 34] {
    let mut block = [0x55; 34];
    block[..2].copy_from_slice(&scale.to_le_bytes());
    for &(index, symbol) in symbols {
        let shift = 2 * (index % 4);
        block[2 + index / 4] &= !(3 << shift);
        block[2 + index / 4] |= symbol << shift;
    }
    block
}

fn assert_matvec_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn prepared_activations_match_independent_calls_and_split_quantizers() -> TestResult {
    let first = pq2(0x3c00, &[(0, 0), (1, 2), (2, 2), (33, 2)]);
    let second = pq2(0x3800, &[(0, 2), (1, 0), (3, 2), (127, 0)]);
    let pq2_weights = tensor(142, vec![256, 2], [first, second, second, first].concat());
    let sibling = tensor(142, vec![256], [second, first].concat());
    let ptq1 = tensor(
        143,
        vec![256],
        [pack_ptq(&first)?, pack_ptq(&second)?].concat(),
    );
    let mut input = vec![0.0; 256];
    for (index, value) in input.iter_mut().enumerate() {
        *value = [127.0, 0.5, 1.5, 2.5, -63.5, -0.5][index % 6];
    }
    let mut scratch = MatvecScratch::default();
    let mut output = vec![f32::NAN];
    {
        let mut prepared = scratch.prepare(&pq2_weights, &input)?;
        for weights in [&pq2_weights, &sibling] {
            assert!(prepared.is_compatible(weights)?);
            prepared.matvec_into(weights, &mut output)?;
            assert_matvec_bits(&output, &matvec(weights, &input)?);
        }
        assert!(!prepared.is_compatible(&ptq1)?);
        assert!(prepared.matvec_into(&ptq1, &mut output).is_err());
        assert_eq!(output, Vec::<f32>::new());
        prepared.matvec_into(&sibling, &mut output)?;
        assert_matvec_bits(&output, &matvec(&sibling, &input)?);
    }
    // The same allocation now contains a different input. A fresh scope must
    // quantize its contents again, including after a failed preparation.
    input.fill(f32::NAN);
    assert!(matvec_into(&pq2_weights, &input, &mut output, &mut scratch).is_err());
    assert_eq!(output, Vec::<f32>::new());
    assert_eq!(scratch.selected_backend(), MatvecBackend::Scalar);
    input.fill(1.0);
    scratch
        .prepare(&pq2_weights, &input)?
        .matvec_into(&sibling, &mut output)?;
    assert_matvec_bits(&output, &matvec(&sibling, &input)?);

    // A 128-column PQ2 fallback, PTQ1 and legacy Q kinds have the same Q8_0
    // activation contract, despite using distinct packed row kernels.
    let odd_pq2 = tensor(142, vec![128], first.to_vec());
    let odd_ptq1 = tensor(143, vec![128], pack_ptq(&first)?.to_vec());
    let q4 = tensor(
        2,
        vec![128],
        [vec![0x3c, 0x3c], vec![0x98; 16]].concat().repeat(4),
    );
    let q8 = tensor(
        8,
        vec![128],
        [vec![0, 0x3c], vec![1; 32]].concat().repeat(4),
    );
    let mut prepared = scratch.prepare(&odd_pq2, &input[..128])?;
    for weights in [&odd_pq2, &odd_ptq1, &q4, &q8] {
        assert!(prepared.is_compatible(weights)?);
        prepared.matvec_into(weights, &mut output)?;
        assert_matvec_bits(&output, &matvec(weights, &input[..128])?);
    }
    assert!(!prepared.is_compatible(&pq2_weights)?);
    Ok(())
}

#[test]
fn prepared_floating_precision_and_failures_preserve_output_contract() -> TestResult {
    let f32_weights = tensor(
        0,
        vec![2],
        [1.0_f32, -0.5]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect(),
    );
    let f16_weights = tensor(
        1,
        vec![2],
        [0x3c00_u16, 0xb800]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect(),
    );
    let bf16_weights = tensor(
        30,
        vec![2],
        [0x3f80_u16, 0xbf00]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect(),
    );
    let weights = [&f32_weights, &f16_weights, &bf16_weights];
    let input = [1.000_7, 0.000_001];
    let mut scratch = MatvecScratch::default();
    let mut output = vec![1.0];
    for (index, weight) in weights.iter().enumerate() {
        let mut prepared = scratch.prepare(weight, &input)?;
        for (candidate, other) in weights.iter().enumerate() {
            assert_eq!(prepared.is_compatible(other)?, index == candidate);
            if index != candidate {
                assert!(prepared.matvec_into(other, &mut output).is_err());
                assert_eq!(output, Vec::<f32>::new());
            }
        }
        prepared.matvec_into(weight, &mut output)?;
        assert_matvec_bits(&output, &matvec(weight, &input)?);
    }
    let mut bad = f32_weights.clone();
    bad.data = Arc::from(f32::NAN.to_le_bytes().repeat(2));
    let mut prepared = scratch.prepare(&f32_weights, &input)?;
    assert!(prepared.matvec_into(&bad, &mut output).is_err());
    assert_eq!(output, Vec::<f32>::new());
    bad.dimensions = vec![1];
    assert!(prepared.is_compatible(&bad).is_err());
    assert!(prepared.matvec_into(&bad, &mut output).is_err());
    assert_eq!(output, Vec::<f32>::new());
    prepared.matvec_into(&f32_weights, &mut output)?;
    assert_matvec_bits(&output, &matvec(&f32_weights, &input)?);
    Ok(())
}

#[test]
fn preparation_rejects_activation_precision_overflow_and_recovers() -> TestResult {
    let mut scratch = MatvecScratch::default();
    let mut output = vec![99.0];
    for (kind, maximum, overflow) in [
        (1, 65_504.0, 65_520.0),
        (30, f32::from_bits(0x7f7f_0000), f32::MAX),
    ] {
        let weights = tensor(kind, vec![1], 0_u16.to_le_bytes().to_vec());
        for input in [overflow, -overflow] {
            // Finite F32 inputs may become infinite in the tensor's activation
            // precision. The preparation itself must reject that conversion.
            assert!(scratch.prepare(&weights, &[input]).is_err());
            assert!(matvec_into(&weights, &[input], &mut output, &mut scratch).is_err());
            assert_eq!(output, [0.0; 0]);
            assert_eq!(scratch.selected_backend(), MatvecBackend::Scalar);
        }
        for input in [maximum, -maximum, 0.0, -0.0] {
            scratch
                .prepare(&weights, &[input])?
                .matvec_into(&weights, &mut output)?;
            assert_eq!(output, [0.0]);
        }
    }
    let weights = tensor(0, vec![1], 0_f32.to_le_bytes().to_vec());
    scratch
        .prepare(&weights, &[f32::MAX])?
        .matvec_into(&weights, &mut output)?;
    assert_eq!(output, [0.0]);
    Ok(())
}

#[test]
fn pq2_matvec_uses_q8k_ties_even_and_per_half_scales() -> TestResult {
    let first = pq2(0x3c00, &[(1, 2), (2, 2), (3, 2)]);
    let second = pq2(0x3800, &[(0, 0)]);
    let packed = [first.as_slice(), second.as_slice()].concat();
    let weights = tensor(142, vec![256, 1], packed);
    let mut input = vec![0.0; 256];
    input[0] = 127.0;
    input[1] = 0.5;
    input[2] = 1.5;
    input[3] = 2.5;
    input[128] = -1.5;
    // Equal magnitudes preserve the first maximum's sign.
    input[255] = -127.0;
    // Signed maximum +127 gives iscale=-1, d=-1. Native nearest_int
    // maps -.5,-1.5,-2.5 to 0,-2,-2; second half contributes +1.
    assert_eq!(matvec(&weights, &input)?, [5.0]);
    let decoded = row(&weights, 0)?;
    let float_dot: f32 = decoded.iter().zip(&input).map(|(&a, &b)| a * b).sum();
    assert_eq!(float_dot, 5.25);
    Ok(())
}

#[test]
fn pq2_native_symbol_three_zero_scale_and_zero_activations() -> TestResult {
    let first = pq2(0x3c00, &[(0, 3)]);
    let second = pq2(0, &[(0, 3)]);
    let weights = tensor(
        142,
        vec![256],
        [first.as_slice(), second.as_slice()].concat(),
    );
    assert_eq!(row(&weights, 0)?[0], 2.0);
    assert_eq!(row(&weights, 0)?[128], 0.0);
    let mut input = vec![0.0; 256];
    input[0] = -127.0;
    assert_eq!(matvec(&weights, &input)?, [-254.0]);
    assert_eq!(matvec(&weights, &[0.0; 256])?, [0.0]);
    Ok(())
}

#[test]
fn ptq1_matvec_uses_q8_0_round_away_and_32_wide_scales() -> TestResult {
    let block = pack_ptq(&pq2(0x4000, &[(1, 2), (2, 2), (33, 2)]))?;
    let weights = tensor(143, vec![128, 1], block.to_vec());
    let mut input = vec![0.0; 128];
    input[0] = 127.0;
    input[1] = 0.5;
    input[2] = 1.5;
    input[32] = 63.5;
    input[33] = 0.25;
    // Q8_0 rounds .5 and 1.5 away from zero. Blocks 0 and 1 have
    // scales 1 and .5. The PTQ1 weight scale is 2: 2*(1+2+.5)=7.
    assert_eq!(matvec(&weights, &input)?, [7.0]);
    let decoded = row(&weights, 0)?;
    let float_dot: f32 = decoded.iter().zip(&input).map(|(&a, &b)| a * b).sum();
    assert_eq!(float_dot, 4.5);
    Ok(())
}

#[test]
fn ptq1_activation_scales_are_rounded_to_fp16() -> TestResult {
    let block = pack_ptq(&pq2(0x3c00, &[(0, 2)]))?;
    let weights = tensor(143, vec![128], block.to_vec());
    let mut input = [0.0; 128];
    input[0] = 1.0;
    // Q8_0 stores half bits 0x2008 for 1/127, then multiplies by 127.
    assert_eq!(matvec(&weights, &input)?, [0.999_938_96]);
    Ok(())
}

#[test]
fn pq2_odd_group_rows_use_native_q8_0_fallback() -> TestResult {
    let first = pq2(0x4000, &[(1, 2), (2, 3), (33, 2)]);
    let mut input = [0.0; 128];
    input[0] = 127.0;
    input[1] = 0.5;
    input[2] = 1.5;
    input[32] = 63.5;
    input[33] = 0.25;
    let weights = tensor(142, vec![128], first.to_vec());
    // Native Q8_0 rounds .5 and 1.5 away from zero; second subblock
    // scale is .5. Native symbol3 is +2: 2*(1+2*2+.5)=11.
    assert_eq!(matvec(&weights, &input)?, [11.0]);
    let decoded = row(&weights, 0)?;
    assert_eq!(
        decoded.iter().zip(input).map(|(&a, b)| a * b).sum::<f32>(),
        7.5
    );

    let second = pq2(0x3800, &[(1, 2), (2, 3), (33, 2)]);
    let third = pq2(0x3c00, &[(1, 2), (2, 3), (33, 2)]);
    let weights = tensor(
        142,
        vec![384],
        [first.as_slice(), second.as_slice(), third.as_slice()].concat(),
    );
    let input = [input.as_slice(), input.as_slice(), input.as_slice()].concat();
    assert_eq!(matvec(&weights, &input)?, [19.25]);
    assert!(matvec(&weights, &input[..383]).is_err());
    Ok(())
}

#[test]
fn packed_rows_match_lossless_ptq1_codec() -> TestResult {
    let symbols: Vec<(usize, u8)> = (0..128)
        .map(|index| Ok((index, u8::try_from((index * 7 + index / 3) % 3)?)))
        .collect::<Result<_, std::num::TryFromIntError>>()?;
    let native = pq2(0x3555, &symbols);
    let packed = pack_ptq(&native)?;
    let source = tensor(142, vec![128], native.to_vec());
    let destination = tensor(143, vec![128], packed.to_vec());
    assert_eq!(row(&source, 0)?, row(&destination, 0)?);
    assert_eq!(matvec(&source, &[0.0; 128])?, [0.0]);
    assert!(row(&source, 1).is_err());
    Ok(())
}

#[test]
fn floating_rows_and_matvec_preserve_native_activation_precision() -> TestResult {
    let f32_tensor = tensor(
        0,
        vec![2, 2],
        [1.0_f32, 2.0, -1.0, 0.5]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect(),
    );
    assert_eq!(row(&f32_tensor, 1)?, [-1.0, 0.5]);
    assert_eq!(matvec(&f32_tensor, &[2.0, 3.0])?, [8.0, -0.5]);
    let half = tensor(
        1,
        vec![2],
        [0x3c00_u16, 0x4000]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect(),
    );
    assert_eq!(row(&half, 0)?, [1.0, 2.0]);
    assert_eq!(matvec(&half, &[1.000_1, 0.0])?, [1.0]);
    let brain = tensor(
        30,
        vec![2],
        [0x3f80_u16, 0xc000]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect(),
    );
    assert_eq!(row(&brain, 0)?, [1.0, -2.0]);
    assert_eq!(matvec(&brain, &[1.001, 0.0])?, [1.0]);
    Ok(())
}

#[test]
fn half_roundtrip_covers_ties_subnormals_and_signed_zero() {
    assert_eq!(round_f16(1.000_488_3), 1.0);
    assert_eq!(round_f16(1.001_464_8), 1.001_953_1);
    assert_eq!(
        round_f16(f32::from_bits(0x3380_0000)).to_bits(),
        0x3380_0000
    );
    assert_eq!(round_f16(f32::from_bits(0x3300_0000)).to_bits(), 0);
    assert_eq!(round_f16(-0.0).to_bits(), (-0.0_f32).to_bits());
    assert_eq!(round_f16(65_504.0), 65_504.0);
    assert_eq!(round_f16(65_520.0), f32::INFINITY);
    assert!(round_f16(f32::NAN).is_nan());
}

#[test]
fn malformed_packed_tensors_and_nonfinite_inputs_fail_closed() -> TestResult {
    assert!(row(&tensor(142, vec![128], vec![0; 33]), 0).is_err());
    assert!(row(&tensor(142, vec![127], vec![0; 34]), 0).is_err());
    assert!(row(&tensor(999, vec![1], vec![0]), 0).is_err());
    assert!(row(&tensor(0, vec![1], f32::NAN.to_le_bytes().to_vec()), 0).is_err());
    let block = pack_ptq(&pq2(0x3c00, &[]))?;
    let weights = tensor(143, vec![128], block.to_vec());
    assert!(matvec(&weights, &[0.0; 127]).is_err());
    assert!(matvec(&weights, &[f32::NAN; 128]).is_err());
    let mut corrupted = block;
    corrupted[24] = 255;
    // Native PTQ admits every packed byte; malformed scales still fail closed.
    assert!(row(&tensor(143, vec![128], corrupted.to_vec()), 0).is_ok());
    corrupted[26..].copy_from_slice(&0x7c00_u16.to_le_bytes());
    assert!(row(&tensor(143, vec![128], corrupted.to_vec()), 0).is_err());
    Ok(())
}

#[test]
fn normalization_and_attention_mask_fixture() -> TestResult {
    assert_eq!(
        rms_norm(&[3.0, 4.0], &[1.0, 2.0], 0.0)?,
        [3.0 / 12.5_f32.sqrt(), 8.0 / 12.5_f32.sqrt()]
    );
    assert!(rms_norm(&[0.0], &[1.0], 0.0).is_err());
    let mut values = [0.0, f32::NEG_INFINITY, 0.0];
    softmax(&mut values)?;
    assert_eq!(values, [0.5, 0.0, 0.5]);
    assert!(softmax(&mut [f32::NEG_INFINITY]).is_err());
    Ok(())
}

#[test]
fn hadamard_and_nonexpanding_validation_fixture() -> TestResult {
    let mut values = [1.0, 2.0, 3.0, 4.0];
    hadamard(&mut values)?;
    assert_eq!(values, [5.0, -1.0, -2.0, 0.0]);
    hadamard(&mut values)?;
    assert_eq!(values, [1.0, 2.0, 3.0, 4.0]);
    assert!(hadamard(&mut []).is_err());
    assert!(hadamard(&mut [1.0; 3]).is_err());
    let valid = tensor(142, vec![128], pq2(0, &[(0, 3)]).to_vec());
    validate_tensor(&valid)?;
    assert!(validate_tensor(&tensor(30, vec![1], 0x7f80_u16.to_le_bytes().to_vec())).is_err());
    assert!(validate_tensor(&tensor(142, vec![128], vec![0; 33])).is_err());
    Ok(())
}

#[test]
fn reusable_rows_and_normalization_clear_and_recover_without_shrinking() -> TestResult {
    let weights = tensor(
        0,
        vec![2, 2],
        [1.0_f32, 2.0, -1.0, 0.5]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect(),
    );
    let mut output = vec![99.0; 256];
    let pointer = output.as_ptr();
    row_into(&weights, 1, &mut output)?;
    assert_eq!(output, [-1.0, 0.5]);
    assert_eq!(output.as_ptr(), pointer);
    assert!(row_into(&weights, 2, &mut output).is_err());
    assert_eq!(output, [0.0; 0]);
    assert_eq!(output.as_ptr(), pointer);
    let invalid = tensor(
        0,
        vec![2],
        [1.0_f32, f32::NAN]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect(),
    );
    assert!(row_into(&invalid, 0, &mut output).is_err());
    assert_eq!(output, [0.0; 0]);
    rms_norm_into(&[3.0, 4.0], &[1.0, 2.0], 0.0, &mut output)?;
    assert_eq!(output, [3.0 / 12.5_f32.sqrt(), 8.0 / 12.5_f32.sqrt()]);
    assert_eq!(output.as_ptr(), pointer);
    assert!(rms_norm_into(&[0.0, 2.0], &[1.0, f32::MAX], 0.0, &mut output).is_err());
    assert_eq!(output, [0.0; 0]);
    assert!(rms_norm_into(&[], &[], 0.0, &mut output).is_err());
    assert_eq!(output, [0.0; 0]);
    rms_norm_into(&[2.0], &[3.0], 0.0, &mut output)?;
    assert_eq!(output, [3.0]);
    assert_eq!(output.as_ptr(), pointer);
    Ok(())
}

#[test]
fn reusable_matvec_crosses_quantization_formats_and_retains_output_storage() -> TestResult {
    let mut scratch = MatvecScratch::default();
    let mut output = vec![99.0; 16];
    let pointer = output.as_ptr();
    let first = pq2(0x3c00, &[(1, 2), (2, 2), (3, 2)]);
    let second = pq2(0x3800, &[(0, 0)]);
    let packed = [first.as_slice(), second.as_slice()].concat();
    let weights = tensor(142, vec![256, 2], [packed.as_slice(); 2].concat());
    let mut input = [0.0; 256];
    input[0] = 127.0;
    input[1] = 0.5;
    input[2] = 1.5;
    input[3] = 2.5;
    input[128] = -1.5;
    input[255] = -127.0;
    matvec_into(&weights, &input, &mut output, &mut scratch)?;
    assert_eq!(output, [5.0, 5.0]);

    let block = pack_ptq(&pq2(0x4000, &[(1, 2), (2, 2), (33, 2)]))?;
    let weights = tensor(143, vec![128], block.to_vec());
    let mut input = [0.0; 128];
    input[0] = 127.0;
    input[1] = 0.5;
    input[2] = 1.5;
    input[32] = 63.5;
    input[33] = 0.25;
    matvec_into(&weights, &input, &mut output, &mut scratch)?;
    assert_eq!(output, [7.0]);

    let fallback = tensor(
        142,
        vec![128],
        pq2(0x4000, &[(1, 2), (2, 3), (33, 2)]).to_vec(),
    );
    matvec_into(&fallback, &input, &mut output, &mut scratch)?;
    assert_eq!(output, [11.0]);
    let block = pack_ptq(&pq2(0x3c00, &[(0, 2)]))?;
    let rounded = tensor(143, vec![128], block.to_vec());
    let mut input = [0.0; 128];
    input[0] = 1.0;
    matvec_into(&rounded, &input, &mut output, &mut scratch)?;
    assert_eq!(output, [0.999_938_96]);

    let mut q4 = [0x99; 18];
    q4[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
    let mut q8 = [1; 34];
    q8[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
    let mut input = [0.0; 32];
    input[0] = 127.0;
    input[1] = 0.5;
    input[2] = 1.5;
    for weights in [
        tensor(2, vec![32], q4.to_vec()),
        tensor(8, vec![32], q8.to_vec()),
    ] {
        matvec_into(&weights, &input, &mut output, &mut scratch)?;
        assert_eq!(output, [130.0]);
    }
    assert_eq!(output.as_ptr(), pointer);
    Ok(())
}

#[test]
fn reusable_matvec_preserves_float_order_and_recovers_from_partial_errors() -> TestResult {
    let mut scratch = MatvecScratch::default();
    let mut output = vec![99.0; 16];
    let pointer = output.as_ptr();
    let weights = tensor(
        0,
        vec![3],
        [16_777_216.0_f32, 1.0, -16_777_216.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect(),
    );
    // Rounded f32 products accumulate in f64, preserving the middle unit.
    matvec_into(&weights, &[1.0; 3], &mut output, &mut scratch)?;
    assert_eq!(output, [1.0]);
    let invalid = tensor(
        0,
        vec![1, 2],
        [1.0_f32, f32::INFINITY]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect(),
    );
    assert!(matvec_into(&invalid, &[1.0], &mut output, &mut scratch).is_err());
    assert_eq!(output, [0.0; 0]);
    assert!(matvec_into(&weights, &[1.0; 2], &mut output, &mut scratch).is_err());
    assert_eq!(output, [0.0; 0]);
    for (kind, bits, input) in [(1, 0x3c00_u16, 1.000_1), (30, 0x3f80_u16, 1.001)] {
        let weights = tensor(kind, vec![1], bits.to_le_bytes().to_vec());
        matvec_into(&weights, &[input], &mut output, &mut scratch)?;
        assert_eq!(output, [1.0]);
    }
    let block = pack_ptq(&pq2(0x3c00, &[(0, 2)]))?;
    let mut corrupted = block;
    corrupted[24] = 255;
    corrupted[26..].copy_from_slice(&0x7c00_u16.to_le_bytes());
    let invalid = tensor(143, vec![128, 2], [block, corrupted].concat());
    assert!(matvec_into(&invalid, &[1.0; 128], &mut output, &mut scratch).is_err());
    assert_eq!(output, [0.0; 0]);
    let valid = tensor(143, vec![128], block.to_vec());
    matvec_into(&valid, &[0.0; 128], &mut output, &mut scratch)?;
    assert_eq!(output, [0.0]);
    assert_eq!(output.as_ptr(), pointer);
    Ok(())
}
