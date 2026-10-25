#![forbid(unsafe_code)]
//! Exact differential checks for the automatic packed PQ2/PTQ1 backend.
//! The scalar runtime remains the arithmetic and failure-behavior oracle.

use std::sync::Arc;

use bonsai_prism_runtime::kernels::{
    MatvecBackend, MatvecFallbackReason, MatvecScratch, MatvecStatistics, matvec_into,
};
use bonsai_prism_runtime::model::Tensor;
use bonsai_ternary::pack_ptq;

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

const SCALE_BITS: [u16; 10] = [
    0x0000, 0x8000, 0x0001, 0x8001, 0x0400, 0x8400, 0x3555, 0xb555, 0x7bff, 0xfbff,
];

fn tensor(kind: u32, columns: usize, rows: usize, data: Vec<u8>) -> Tensor {
    Tensor {
        name: "simd-differential".to_owned(),
        dimensions: vec![columns, rows],
        kind,
        data: Arc::from(data),
    }
}

fn packed_block(
    scale: u16,
    phase: usize,
    symbol_count: usize,
) -> Result<[u8; 34], std::num::TryFromIntError> {
    let mut block = [0; 34];
    block[..2].copy_from_slice(&scale.to_le_bytes());
    for index in 0..128 {
        let symbol = u8::try_from((index * 7 + index / 11 + phase) % symbol_count)?;
        block[2 + index / 4] |= symbol << (2 * (index % 4));
    }
    Ok(block)
}

fn pq2_tensor(
    columns: usize,
    rows: usize,
    scales: &[u16],
) -> Result<Tensor, std::num::TryFromIntError> {
    let blocks = columns / 128;
    let mut data = Vec::new();
    for row in 0..rows {
        for block in 0..blocks {
            let index = row * blocks + block;
            data.extend_from_slice(&packed_block(
                scales[index % scales.len()],
                row * 3 + block * 5,
                4,
            )?);
        }
    }
    Ok(tensor(142, columns, rows, data))
}

fn ptq1_tensor(columns: usize, rows: usize, scales: &[u16]) -> TestResult<Tensor> {
    let blocks = columns / 128;
    let mut data = Vec::new();
    for row in 0..rows {
        for block in 0..blocks {
            let index = row * blocks + block;
            let mut packed = pack_ptq(&packed_block(0x3c00, row * 3 + block * 5, 3)?)?;
            // Native tensors allow both signed zero scales; the exporter does
            // not, so set scale bits after constructing canonical trit bytes.
            packed[26..].copy_from_slice(&scales[index % scales.len()].to_le_bytes());
            data.extend_from_slice(&packed);
        }
    }
    Ok(tensor(143, columns, rows, data))
}

fn activations(columns: usize, negative_first: bool) -> Vec<f32> {
    const VALUES: [f32; 12] = [
        0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 0.0, -0.0, 63.5, -63.5, 17.25, -31.75,
    ];
    let mut values: Vec<f32> = (0..columns)
        .map(|index| VALUES[index % VALUES.len()])
        .collect();
    for (index, group) in values.chunks_mut(256).enumerate() {
        let maximum = if negative_first ^ (index % 2 == 1) {
            -127.0
        } else {
            127.0
        };
        group[0] = maximum;
        // The first strict absolute maximum determines Q8_K's scale sign.
        if let Some(last) = group.last_mut() {
            *last = -maximum;
        }
    }
    values
}

const fn eligible_backend() -> MatvecBackend {
    if cfg!(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    )) {
        MatvecBackend::ExperimentalNeon
    } else {
        MatvecBackend::Scalar
    }
}

const fn build_fallback() -> Option<MatvecFallbackReason> {
    if !cfg!(feature = "experimental-neon") {
        Some(MatvecFallbackReason::FeatureDisabled)
    } else if matches!(eligible_backend(), MatvecBackend::Scalar) {
        Some(MatvecFallbackReason::TargetUnsupported)
    } else {
        None
    }
}

fn assert_bits(actual: &[f32], expected: &[f32]) {
    let actual: Vec<u32> = actual.iter().map(|value| value.to_bits()).collect();
    let expected: Vec<u32> = expected.iter().map(|value| value.to_bits()).collect();
    assert_eq!(actual, expected);
}

fn compare(
    weights: &Tensor,
    input: &[f32],
    scratch: &mut MatvecScratch,
    output: &mut Vec<f32>,
    expected_backend: MatvecBackend,
) -> TestResult {
    let mut scalar = Vec::new();
    let mut reference = MatvecScratch::with_backend(MatvecBackend::Scalar);
    matvec_into(weights, input, &mut scalar, &mut reference)?;
    assert_eq!(reference.selected_backend(), MatvecBackend::Scalar);
    matvec_into(weights, input, output, scratch)?;
    assert_eq!(scratch.selected_backend(), expected_backend);
    assert_bits(output, &scalar);
    Ok(())
}

#[test]
fn simd_diagnostics_explain_build_tensor_and_explicit_scalar_selection() -> TestResult {
    let mut output = Vec::new();
    let mut legacy = [0x99; 18];
    legacy[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
    for request in [
        MatvecBackend::Auto,
        MatvecBackend::ExperimentalNeon,
        MatvecBackend::Scalar,
    ] {
        let mut scratch = MatvecScratch::with_backend(request);
        let initial = scratch.diagnostics();
        assert_eq!(initial.requested_backend, request);
        assert_eq!(initial.selected_backend, MatvecBackend::Scalar);
        assert_eq!(
            initial.neon_build_eligible,
            matches!(eligible_backend(), MatvecBackend::ExperimentalNeon)
        );
        assert_eq!(initial.fallback_reason, None);
        assert!(!initial.dispatched);
        assert!(!initial.succeeded);
        for (weights, supports_neon) in [
            (pq2_tensor(256, 2, &[0x3c00])?, true),
            (pq2_tensor(384, 2, &[0x3c00])?, true),
            (ptq1_tensor(128, 2, &[0x3c00])?, true),
            (tensor(2, 32, 1, legacy.to_vec()), false),
            (tensor(0, 1, 1, 1.0_f32.to_le_bytes().to_vec()), false),
        ] {
            matvec_into(
                &weights,
                &activations(weights.dimensions[0], false),
                &mut output,
                &mut scratch,
            )?;
            let report = scratch.diagnostics();
            let reason = if matches!(request, MatvecBackend::Scalar) {
                Some(MatvecFallbackReason::ScalarRequested)
            } else if !supports_neon {
                Some(MatvecFallbackReason::TensorKindUnsupported)
            } else {
                build_fallback()
            };
            assert_eq!(report.requested_backend, request);
            assert_eq!(report.selected_backend, scratch.selected_backend());
            assert_eq!(report.fallback_reason, reason);
            assert_eq!(report.neon_build_eligible, initial.neon_build_eligible);
            assert!(report.dispatched);
            assert!(report.succeeded);
            assert_eq!(
                report.selected_backend,
                if reason.is_none() {
                    MatvecBackend::ExperimentalNeon
                } else {
                    MatvecBackend::Scalar
                }
            );
        }
    }
    Ok(())
}

#[test]
fn simd_diagnostics_replace_success_on_preparation_and_row_failure() -> TestResult {
    let weights = pq2_tensor(256, 2, &[0x3c00])?;
    let input = activations(256, false);
    let mut scratch = MatvecScratch::with_backend(MatvecBackend::ExperimentalNeon);
    let mut output = Vec::new();
    matvec_into(&weights, &input, &mut output, &mut scratch)?;
    let successful = scratch.diagnostics();
    assert!(successful.dispatched);
    assert!(successful.succeeded);

    // Even a successful preparation clears the last matrix result.
    {
        let _prepared = scratch.prepare(&weights, &input)?;
    }
    assert!(!scratch.diagnostics().dispatched);
    assert!(!scratch.diagnostics().succeeded);
    assert_eq!(scratch.diagnostics().fallback_reason, None);

    let mut invalid_scale = weights.data.to_vec();
    invalid_scale[68..70].copy_from_slice(&0x7c00_u16.to_le_bytes());
    let bad_rows = tensor(142, 256, 2, invalid_scale);
    let malformed = tensor(142, 255, 2, weights.data.to_vec());
    for (case_weights, case_input, dispatched) in [
        (&bad_rows, input.clone(), true),
        (&malformed, input.clone(), false),
        (&weights, vec![f32::NAN; 256], false),
        (&weights, input[..255].to_vec(), false),
    ] {
        assert!(matvec_into(case_weights, &case_input, &mut output, &mut scratch).is_err());
        let failed = scratch.diagnostics();
        assert_eq!(failed.requested_backend, successful.requested_backend);
        assert_eq!(failed.neon_build_eligible, successful.neon_build_eligible);
        assert_eq!(failed.dispatched, dispatched);
        assert!(!failed.succeeded);
        assert_eq!(
            failed.fallback_reason,
            if dispatched { build_fallback() } else { None }
        );
        assert_eq!(
            failed.selected_backend,
            if dispatched {
                eligible_backend()
            } else {
                MatvecBackend::Scalar
            }
        );
        assert_eq!(output, Vec::<f32>::new());
        matvec_into(&weights, &input, &mut output, &mut scratch)?;
        assert_eq!(scratch.diagnostics(), successful);
    }
    Ok(())
}

#[test]
fn simd_diagnostics_track_latest_prepared_sibling_attempt() -> TestResult {
    let weights = pq2_tensor(256, 2, &[0x3c00])?;
    let incompatible = ptq1_tensor(256, 2, &[0x3c00])?;
    let input = activations(256, false);
    let mut scratch = MatvecScratch::default();
    let mut output = Vec::new();
    {
        let mut prepared = scratch.prepare(&weights, &input)?;
        prepared.matvec_into(&weights, &mut output)?;
        assert!(prepared.matvec_into(&incompatible, &mut output).is_err());
    }
    assert!(!scratch.diagnostics().dispatched);
    assert!(!scratch.diagnostics().succeeded);
    assert_eq!(
        scratch.diagnostics().selected_backend,
        MatvecBackend::Scalar
    );
    assert_eq!(scratch.diagnostics().fallback_reason, None);
    assert_eq!(output, Vec::<f32>::new());
    {
        let mut prepared = scratch.prepare(&weights, &input)?;
        assert!(prepared.matvec_into(&incompatible, &mut output).is_err());
        prepared.matvec_into(&weights, &mut output)?;
    }
    assert!(scratch.diagnostics().dispatched);
    assert!(scratch.diagnostics().succeeded);
    assert_eq!(scratch.diagnostics().selected_backend, eligible_backend());
    assert_eq!(scratch.diagnostics().fallback_reason, build_fallback());
    assert_eq!(output.len(), 2);
    Ok(())
}

#[test]
fn simd_statistics_retain_mixed_dispatch_and_failed_work() -> TestResult {
    let weights = pq2_tensor(256, 2, &[0x3c00])?;
    let floating = tensor(0, 1, 1, 1.0_f32.to_le_bytes().to_vec());
    let mut invalid = weights.clone();
    let mut bytes = invalid.data.to_vec();
    bytes[..2].copy_from_slice(&0x7c00_u16.to_le_bytes());
    invalid.data = Arc::from(bytes);
    let incompatible = ptq1_tensor(256, 2, &[0x3c00])?;
    let input = activations(256, false);
    for request in [MatvecBackend::Auto, MatvecBackend::Scalar] {
        let mut scratch = MatvecScratch::with_backend(request);
        let mut output = Vec::new();
        assert_eq!(scratch.statistics(), MatvecStatistics::default());
        matvec_into(&weights, &input, &mut output, &mut scratch)?;
        matvec_into(&floating, &[1.0], &mut output, &mut scratch)?;
        assert!(matvec_into(&weights, &[], &mut output, &mut scratch).is_err());
        assert_eq!(output, Vec::<f32>::new());
        {
            let mut prepared = scratch.prepare(&weights, &input)?;
            prepared.matvec_into(&weights, &mut output)?;
            assert!(prepared.matvec_into(&incompatible, &mut output).is_err());
            assert_eq!(output, Vec::<f32>::new());
            assert!(prepared.matvec_into(&invalid, &mut output).is_err());
            assert_eq!(output, Vec::<f32>::new());
            prepared.matvec_into(&weights, &mut output)?;
        }
        let statistics = scratch.statistics();
        assert_eq!(statistics.preparations, 4);
        assert_eq!(statistics.preparation_failures, 1);
        assert_eq!(statistics.matrix_attempts, 6);
        assert_eq!(statistics.matrix_successes, 4);
        assert_eq!(statistics.rejected_before_dispatch, 1);
        let neon = if request == MatvecBackend::Auto
            && eligible_backend() == MatvecBackend::ExperimentalNeon
        {
            4
        } else {
            0
        };
        assert_eq!(statistics.neon_operations, neon);
        assert_eq!(statistics.scalar_operations, 5 - neon);
        if request == MatvecBackend::Scalar {
            assert_eq!(statistics.scalar_requested, 5);
            assert_eq!(statistics.tensor_kind_unsupported, 0);
        } else {
            assert_eq!(statistics.scalar_requested, 0);
            assert_eq!(statistics.tensor_kind_unsupported, 1);
            assert_eq!(
                statistics.feature_disabled,
                if build_fallback() == Some(MatvecFallbackReason::FeatureDisabled) {
                    4
                } else {
                    0
                }
            );
            assert_eq!(
                statistics.target_unsupported,
                if build_fallback() == Some(MatvecFallbackReason::TargetUnsupported) {
                    4
                } else {
                    0
                }
            );
        }
        assert_eq!(
            statistics.scalar_operations,
            statistics.scalar_requested
                + statistics.tensor_kind_unsupported
                + statistics.feature_disabled
                + statistics.target_unsupported
        );
    }
    Ok(())
}

#[test]
fn simd_prepared_siblings_match_scalar_and_reset_after_rejection() -> TestResult {
    let mut scratch = MatvecScratch::default();
    let mut output = vec![f32::NAN];
    for columns in [128, 256, 384, 512] {
        let pq2 = pq2_tensor(columns, 3, &[0x3555, 0xb555, 0x0400])?;
        let ptq1 = ptq1_tensor(columns, 2, &[0x3555, 0xb555])?;
        let input = activations(columns, true);
        for weights in [&pq2, &ptq1] {
            let mut reference = MatvecScratch::with_backend(MatvecBackend::Scalar);
            let mut expected = Vec::new();
            matvec_into(weights, &input, &mut expected, &mut reference)?;
            {
                let mut prepared = scratch.prepare(weights, &input)?;
                for _ in 0..3 {
                    prepared.matvec_into(weights, &mut output)?;
                    assert_bits(&output, &expected);
                }
            }
            assert_eq!(scratch.selected_backend(), eligible_backend());
        }
        if columns.is_multiple_of(256) {
            {
                let mut prepared = scratch.prepare(&pq2, &input)?;
                prepared.matvec_into(&pq2, &mut output)?;
                assert!(prepared.matvec_into(&ptq1, &mut output).is_err());
                assert_eq!(output, Vec::<f32>::new());
            }
            assert_eq!(scratch.selected_backend(), MatvecBackend::Scalar);
        }
        assert!(scratch.prepare(&pq2, &input[..columns - 1]).is_err());
        assert_eq!(scratch.selected_backend(), MatvecBackend::Scalar);
    }
    Ok(())
}

#[test]
fn simd_default_selects_available_kernel_and_explicit_scalar_overrides() -> TestResult {
    assert_eq!(MatvecBackend::default(), MatvecBackend::Auto);
    let mut scratch = MatvecScratch::default();
    assert_eq!(scratch.selected_backend(), MatvecBackend::Scalar);
    let mut automatic = MatvecScratch::with_backend(MatvecBackend::Auto);
    let mut scalar = MatvecScratch::with_backend(MatvecBackend::Scalar);
    let mut output = Vec::new();
    let mut legacy = [0x99; 18];
    legacy[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
    for (weights, expected_backend) in [
        (pq2_tensor(256, 3, &SCALE_BITS)?, eligible_backend()),
        (pq2_tensor(384, 3, &SCALE_BITS)?, eligible_backend()),
        (ptq1_tensor(384, 3, &SCALE_BITS)?, eligible_backend()),
        (tensor(2, 32, 1, legacy.to_vec()), MatvecBackend::Scalar),
        (
            tensor(0, 1, 1, 1.0_f32.to_le_bytes().to_vec()),
            MatvecBackend::Scalar,
        ),
    ] {
        let input = activations(weights.dimensions[0], false);
        for request in [&mut scratch, &mut automatic] {
            compare(&weights, &input, request, &mut output, expected_backend)?;
        }
        compare(
            &weights,
            &input,
            &mut scalar,
            &mut output,
            MatvecBackend::Scalar,
        )?;
        assert!(matvec_into(&weights, &[], &mut output, &mut scratch).is_err());
        assert_eq!(scratch.selected_backend(), MatvecBackend::Scalar);
        assert_eq!(output, Vec::<f32>::new());
    }
    Ok(())
}

#[test]
fn simd_pq2_matches_scalar_for_multiple_rows_blocks_and_scale_bits() -> TestResult {
    let mut scratch = MatvecScratch::with_backend(MatvecBackend::ExperimentalNeon);
    let mut output = Vec::new();
    for columns in [256, 512, 1_024] {
        for rows in [1, 3, 7] {
            let weights = pq2_tensor(columns, rows, &SCALE_BITS)?;
            for negative_first in [false, true] {
                compare(
                    &weights,
                    &activations(columns, negative_first),
                    &mut scratch,
                    &mut output,
                    eligible_backend(),
                )?;
            }
        }
    }
    let weights = pq2_tensor(256, 3, &SCALE_BITS)?;
    let input = activations(256, false);
    let mut default = MatvecScratch::default();
    compare(
        &weights,
        &input,
        &mut default,
        &mut output,
        eligible_backend(),
    )?;
    Ok(())
}

#[test]
fn simd_pq2_preserves_quantization_edges_and_ordered_block_sums() -> TestResult {
    let mut scratch = MatvecScratch::with_backend(MatvecBackend::ExperimentalNeon);
    let mut output = Vec::new();
    let weights = pq2_tensor(512, 3, &[0x3c00, 0xbc00, 0x3555, 0xb555])?;
    for input in [
        vec![0.0; 512],
        vec![-0.0; 512],
        activations(512, false),
        activations(512, true),
        vec![1.0e-20; 512],
    ] {
        compare(
            &weights,
            &input,
            &mut scratch,
            &mut output,
            eligible_backend(),
        )?;
    }

    // The smallest finite half scale keeps even the largest finite activations
    // finite, while exercising the Q8_K inverse-scale and product extremes.
    let tiny_weights = pq2_tensor(512, 3, &[0x0001, 0x8001])?;
    for input in [vec![f32::MAX; 512], vec![-f32::MAX; 512]] {
        compare(
            &tiny_weights,
            &input,
            &mut scratch,
            &mut output,
            eligible_backend(),
        )?;
    }

    // Saturated positive/negative weights exercise the proven integer bounds,
    // including PQ2's native code 3 (+2) rather than a ternary interpretation.
    let mut saturated_data = Vec::new();
    for byte in [0xff, 0x00] {
        let mut block = [byte; 34];
        block[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        saturated_data.extend_from_slice(&block);
        saturated_data.extend_from_slice(&block);
    }
    let saturated = tensor(142, 256, 2, saturated_data);
    for value in [127.0, -127.0] {
        compare(
            &saturated,
            &[value; 256],
            &mut scratch,
            &mut output,
            eligible_backend(),
        )?;
    }

    // Adding a small block between large opposite blocks loses that first small
    // contribution. Reassociation or changing the per-block scale order changes
    // this result even when every integer partial sum is exact.
    let mut data = Vec::new();
    for bits in [0x7bff_u16, 0x0400, 0xfbff, 0x0400] {
        let mut block = [0xaa; 34];
        block[..2].copy_from_slice(&bits.to_le_bytes());
        data.extend_from_slice(&block);
    }
    let ordered = tensor(142, 512, 1, data);
    compare(
        &ordered,
        &[127.0; 512],
        &mut scratch,
        &mut output,
        eligible_backend(),
    )?;
    Ok(())
}

#[test]
fn simd_requested_backend_falls_back_and_reuses_scratch_across_formats() -> TestResult {
    let mut scratch = MatvecScratch::with_backend(MatvecBackend::ExperimentalNeon);
    let mut output = vec![99.0; 16];
    let pointer = output.as_ptr();
    for columns in [256, 128, 512, 384] {
        let weights = pq2_tensor(columns, 3, &[0x3c00, 0xb800])?;
        compare(
            &weights,
            &activations(columns, false),
            &mut scratch,
            &mut output,
            eligible_backend(),
        )?;
    }

    let mut packed = Vec::new();
    for index in 0..6 {
        packed.extend_from_slice(&pack_ptq(&packed_block(0x3555, index, 3)?)?);
    }
    let ptq1 = tensor(143, 256, 3, packed);
    compare(
        &ptq1,
        &activations(256, true),
        &mut scratch,
        &mut output,
        eligible_backend(),
    )?;

    let floating = tensor(
        0,
        3,
        2,
        [16_777_216.0_f32, 1.0, -16_777_216.0, -1.0, 0.5, 2.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect(),
    );
    compare(
        &floating,
        &[1.0; 3],
        &mut scratch,
        &mut output,
        MatvecBackend::Scalar,
    )?;
    let weights = pq2_tensor(256, 3, &SCALE_BITS)?;
    compare(
        &weights,
        &[0.0; 256],
        &mut scratch,
        &mut output,
        eligible_backend(),
    )?;
    assert_eq!(output.as_ptr(), pointer);
    Ok(())
}

#[test]
fn simd_q8_groups_match_scalar_for_ptq1_and_odd_pq2() -> TestResult {
    let mut scratch = MatvecScratch::with_backend(MatvecBackend::ExperimentalNeon);
    let mut output = Vec::new();
    for columns in [128, 256, 384, 640] {
        for rows in [1, 3, 5] {
            let ptq1 = ptq1_tensor(columns, rows, &SCALE_BITS)?;
            let pq2 = pq2_tensor(columns, rows, &SCALE_BITS)?;
            for weights in [&ptq1, &pq2] {
                for negative_first in [false, true] {
                    let mut input = activations(columns, negative_first);
                    // Every 32-value group has its own stored FP16 Q8_0 scale;
                    // retain unequal scales and ties-away quantization cases.
                    for (group, values) in input.as_chunks_mut::<32>().0.iter_mut().enumerate() {
                        let factor = f32::from(u16::try_from(group % 7 + 1)?);
                        for value in values {
                            *value *= factor;
                        }
                    }
                    compare(
                        weights,
                        &input,
                        &mut scratch,
                        &mut output,
                        eligible_backend(),
                    )?;
                }
                for input in [
                    vec![0.0; columns],
                    vec![-0.0; columns],
                    vec![1.0e-20; columns],
                ] {
                    compare(
                        weights,
                        &input,
                        &mut scratch,
                        &mut output,
                        eligible_backend(),
                    )?;
                }
                compare(
                    weights,
                    &activations(columns, false),
                    &mut MatvecScratch::default(),
                    &mut output,
                    eligible_backend(),
                )?;
            }
        }
    }
    // Legacy packed formats keep their scalar kernel even when NEON is requested.
    for (kind, bytes) in [(2, 18), (8, 34)] {
        let mut data = vec![0xff; bytes];
        data[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        compare(
            &tensor(kind, 32, 1, data),
            &[127.0; 32],
            &mut scratch,
            &mut output,
            MatvecBackend::Scalar,
        )?;
    }
    Ok(())
}

#[test]
fn simd_q8_groups_preserve_ordered_float_sums() -> TestResult {
    let mut scratch = MatvecScratch::with_backend(MatvecBackend::ExperimentalNeon);
    let mut output = Vec::new();
    for kind in [142, 143] {
        let mut data = Vec::new();
        for bits in [0x7bff_u16, 0x0400, 0xfbff] {
            let mut block = [0xaa; 34];
            block[..2].copy_from_slice(&bits.to_le_bytes());
            if kind == 143 {
                data.extend_from_slice(&pack_ptq(&block)?);
            } else {
                data.extend_from_slice(&block);
            }
        }
        let weights = tensor(kind, 384, 1, data);
        let input: Vec<f32> = (0..384)
            .map(|index| match index / 32 % 4 {
                0 => 127.0 * 65_504.0,
                1 | 3 => 127.0 / 16_384.0,
                _ => -127.0 * 65_504.0,
            })
            .collect();
        compare(
            &weights,
            &input,
            &mut scratch,
            &mut output,
            eligible_backend(),
        )?;
        compare(
            &weights,
            &[127.0; 384],
            &mut scratch,
            &mut output,
            eligible_backend(),
        )?;
    }
    Ok(())
}

#[test]
fn simd_q8_groups_accept_native_aliases_and_reject_nonfinite_payloads() -> TestResult {
    let mut scratch = MatvecScratch::with_backend(MatvecBackend::ExperimentalNeon);
    let mut scalar = MatvecScratch::with_backend(MatvecBackend::Scalar);
    let mut output = vec![99.0; 16];
    let pointer = output.as_ptr();
    let mut scalar_output = Vec::new();
    let valid = ptq1_tensor(384, 3, &[0x0000, 0x8000, 0x3555])?;
    let mut cases = Vec::new();
    for position in 0..26 {
        let mut data = valid.data.to_vec();
        // Native decoding accepts fractional aliases in every packed lane,
        // including in later blocks with zero scale.
        data[84 + position] = 1;
        compare(
            &tensor(143, 384, 3, data),
            &activations(384, true),
            &mut scratch,
            &mut output,
            eligible_backend(),
        )?;
    }
    for kind in [142, 143] {
        let weights = if kind == 142 {
            pq2_tensor(384, 3, &[0x3c00])?
        } else {
            ptq1_tensor(384, 3, &[0x3c00])?
        };
        for bits in [0x7c00_u16, 0xfc00, 0x7e00] {
            let mut data = weights.data.to_vec();
            let offset = if kind == 142 { 102 } else { 84 + 26 };
            data[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
            cases.push((tensor(kind, 384, 3, data), vec![0.0; 384]));
        }
        for value in [f32::NAN, f32::INFINITY, f32::MAX] {
            cases.push((weights.clone(), vec![value; 384]));
        }
    }
    for (weights, input) in cases {
        let scalar_error = matvec_into(&weights, &input, &mut scalar_output, &mut scalar)
            .err()
            .ok_or("scalar accepted invalid Q8_0 group input")?;
        let error = matvec_into(&weights, &input, &mut output, &mut scratch)
            .err()
            .ok_or("NEON request accepted invalid Q8_0 group input")?;
        assert_eq!(error.to_string(), scalar_error.to_string());
        assert_eq!(output.len(), 0);
        assert_eq!(scalar_output.len(), 0);
        compare(
            &valid,
            &activations(384, true),
            &mut scratch,
            &mut output,
            eligible_backend(),
        )?;
        assert_eq!(output.as_ptr(), pointer);
    }
    Ok(())
}

#[test]
fn simd_errors_clear_output_and_recover_with_retained_storage() -> TestResult {
    let valid = pq2_tensor(256, 3, &[0x3c00, 0xb800])?;
    let input = activations(256, false);
    let mut cases = Vec::new();
    for length in [valid.data.len() - 1, valid.data.len() + 1] {
        let mut data = valid.data.to_vec();
        data.resize(length, 0);
        cases.push((tensor(142, 256, 3, data), input.clone()));
    }
    cases.push((tensor(142, 255, 3, valid.data.to_vec()), input.clone()));
    cases.push((valid.clone(), input[..255].to_vec()));
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut invalid_input = input.clone();
        invalid_input[255] = value;
        cases.push((valid.clone(), invalid_input));
    }
    // Finite values may still fail checked quantization or output arithmetic.
    cases.push((valid.clone(), vec![f32::from_bits(1); 256]));
    cases.push((pq2_tensor(256, 3, &[0x7bff])?, vec![f32::MAX; 256]));
    for bits in [0x7c00_u16, 0xfc00, 0x7e00] {
        let mut data = valid.data.to_vec();
        // Corruption in a later row tests clearing partially written output.
        data[68..70].copy_from_slice(&bits.to_le_bytes());
        cases.push((tensor(142, 256, 3, data.clone()), input.clone()));
        // A zero activation block must still validate its weight scale.
        cases.push((tensor(142, 256, 3, data), vec![0.0; 256]));
    }
    let mut canonical = pack_ptq(&packed_block(0x3c00, 0, 3)?)?;
    // The exporter rejects zero scales, while the native runtime accepts them.
    canonical[26..].copy_from_slice(&0_u16.to_le_bytes());
    let mut invalid_ptq1 = [canonical, canonical].concat();
    invalid_ptq1[28 + 24] = 255;
    // Native aliases remain valid, but even zero inputs must reject infinity.
    invalid_ptq1[28 + 26..28 + 28].copy_from_slice(&0x7c00_u16.to_le_bytes());
    cases.push((tensor(143, 128, 2, invalid_ptq1), vec![0.0; 128]));

    let mut scalar_scratch = MatvecScratch::with_backend(MatvecBackend::Scalar);
    let mut scratch = MatvecScratch::with_backend(MatvecBackend::ExperimentalNeon);
    let mut scalar_output = vec![99.0; 16];
    let mut output = vec![99.0; 16];
    let pointer = output.as_ptr();
    for (weights, invalid_input) in cases {
        assert!(
            matvec_into(
                &weights,
                &invalid_input,
                &mut scalar_output,
                &mut scalar_scratch
            )
            .is_err()
        );
        assert!(matvec_into(&weights, &invalid_input, &mut output, &mut scratch).is_err());
        assert_eq!(scalar_output.len(), 0);
        assert_eq!(output.len(), 0);
        compare(
            &valid,
            &input,
            &mut scratch,
            &mut output,
            eligible_backend(),
        )?;
        assert_eq!(output.as_ptr(), pointer);
    }
    Ok(())
}
