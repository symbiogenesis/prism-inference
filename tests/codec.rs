//! Focused persistent-buffer ownership, native capability and packed-byte checks.
#![forbid(unsafe_code)]

use bonsai_prism_runtime::codec::{
    BufferRegion, CodecBuffers, CodecFunction, MAX_BUFFER_BYTES, MAX_BUFFERS, TYPE_COUNT,
    inventory, traits,
};
use bonsai_prism_runtime::kernels::decode_row_into;

type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

const fn region(buffer: u32, offset: usize) -> BufferRegion {
    BufferRegion { buffer, offset }
}

#[test]
fn inventory_retains_native_holes_and_distinguishes_unimplemented_cpu_operations() -> TestResult {
    let values = inventory().collect::<Vec<_>>();
    assert_eq!(values.len(), usize::try_from(TYPE_COUNT)?);
    assert_eq!(values[4].name, Some("DEPRECATED"));
    assert!(values[31].name.is_some_and(|name| name.contains("REMOVED")));
    assert_eq!(values[100].name, None);
    assert_eq!(values[100].block_size, 0);
    assert_eq!(values[142].name, Some("pq2_0"));
    assert_eq!(values[143].type_size, 28);
    assert!(traits(TYPE_COUNT).is_err());
    assert!(values[4].row_bytes(0).is_err());
    assert_eq!(values[142].row_bytes(0)?, 0);
    assert_eq!(values[142].row_bytes(256)?, 68);
    assert!(values[142].row_bytes(127).is_err());
    assert!(values[0].row_bytes(usize::MAX).is_err());
    assert!(values[0].row_bytes(MAX_BUFFER_BYTES).is_err());
    assert!(values[1].native.from_float_cpu);
    assert!(!values[1].rust.from_float_cpu);
    assert!(values[18].native.from_float_reference);
    assert!(values[18].rust.from_float_reference);
    assert!(values[18].rust.to_float);
    assert!(values[18].rust.vec_dot);
    assert!(values[26].native.from_float_cpu);
    assert!(!values[26].rust.from_float_cpu);
    assert!(!values[15].native.from_float_reference);
    assert!(!values[15].native.to_float);
    assert!(values[15].rust.from_float_reference);
    assert!(values[15].rust.to_float);
    assert_eq!(values[142].vec_dot_type, 15);
    assert_eq!(values[143].vec_dot_type, 8);
    Ok(())
}

#[test]
fn released_handles_stay_invalid_and_allocation_failures_preserve_live_buffers() -> TestResult {
    let mut owner = CodecBuffers::default();
    assert!(owner.is_empty());
    let first = owner.new_buffer(16, 64, 1)?;
    let second = owner.new_buffer(8, 8, 8)?;
    assert_eq!(owner.read(region(first, 0), 16)?.as_ptr().addr() % 64, 1);
    owner.write(region(first, 3), &[4, 5, 6])?;
    assert_eq!(owner.read(region(first, 0), 8)?, &[0, 0, 0, 4, 5, 6, 0, 0]);
    assert_eq!(owner.allocated_bytes(), 24);
    assert!(owner.new_buffer(1, 3, 0).is_err());
    assert!(owner.new_buffer(1, 0, 0).is_err());
    assert!(owner.new_buffer(1, 8192, 0).is_err());
    assert!(owner.new_buffer(1, 64, 65).is_err());
    assert!(owner.new_buffer(MAX_BUFFER_BYTES + 1, 64, 0).is_err());
    assert!(owner.read(region(first, usize::MAX), 1).is_err());
    assert!(owner.read(region(first, 15), usize::MAX).is_err());
    assert!(owner.write(region(first, 15), &[9, 9]).is_err());
    assert_eq!(owner.allocated_bytes(), 24);
    owner.free_buffer(first)?;
    assert!(owner.free_buffer(first).is_err());
    assert!(owner.read(region(first, 0), 0).is_err());
    assert_eq!(owner.allocated_bytes(), 8);
    let replacement = owner.new_buffer(0, 1, 0)?;
    assert!(replacement > second);
    assert_eq!(owner.read(region(replacement, 0), 0)?, &[0_u8; 0]);
    assert!(owner.read(region(replacement, 1), 0).is_err());
    while owner.len() < MAX_BUFFERS {
        owner.new_buffer(0, 1, 0)?;
    }
    assert!(owner.new_buffer(0, 1, 0).is_err());
    assert_eq!(owner.len(), MAX_BUFFERS);
    assert_eq!(owner.allocated_bytes(), 8);
    owner.free_buffer(second)?;
    assert_eq!(owner.allocated_bytes(), 0);
    Ok(())
}

#[test]
fn unaligned_overlapping_rows_preserve_ternary_format_and_exact_packed_dot() -> TestResult {
    let mut owner = CodecBuffers::default();
    let data = owner.new_buffer(1040, 64, 1)?;
    owner.write(region(data, 0), &[0xa5; 1040])?;
    let values = [-1.0_f32, 0.0, 1.0, 1.0].repeat(64);
    let source = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    owner.write(region(data, 1), &source)?;
    owner.call(
        CodecFunction::FromFloatReference,
        142,
        256,
        region(data, 1),
        None,
        region(data, 1),
    )?;
    let block = [&[0, 0x3c][..], &[0xa4; 32][..]].concat();
    let expected = block.repeat(2);
    assert_eq!(owner.read(region(data, 1), 68)?, expected);
    assert_eq!(owner.read(region(data, 0), 1)?, &[0xa5]);
    assert_eq!(owner.read(region(data, 1025), 15)?, &[0xa5; 15]);

    // A manually serialized Q8_K row: scale 1, 256 signed ones and sixteen
    // block sums of 16. The ternary pattern sums to 64 over this row.
    let rhs = owner.new_buffer(293, 32, 1)?;
    let mut activations = 1.0_f32.to_le_bytes().to_vec();
    activations.extend([1_u8; 256]);
    activations.extend([16_i16; 16].iter().flat_map(|sum| sum.to_le_bytes()));
    owner.write(region(rhs, 1), &activations)?;
    owner.call(
        CodecFunction::VecDot,
        142,
        256,
        region(data, 1),
        Some(region(rhs, 1)),
        region(data, 1000),
    )?;
    assert_eq!(owner.read(region(data, 1000), 4)?, 64.0_f32.to_le_bytes());
    owner.call(
        CodecFunction::ToFloat,
        142,
        256,
        region(data, 1),
        None,
        region(data, 1),
    )?;
    assert_eq!(owner.read(region(data, 1), 1024)?, source);
    Ok(())
}

#[test]
fn rejected_calls_leave_output_intact_without_cpu_reference_substitution() -> TestResult {
    let mut owner = CodecBuffers::default();
    let data = owner.new_buffer(1024, 1, 0)?;
    let output = owner.new_buffer(1024, 1, 0)?;
    owner.write(region(output, 0), &[0x5a; 1024])?;
    for (function, kind, count, input_offset, with_rhs, output_offset) in [
        (CodecFunction::FromFloatCpu, 1, 256, 0, false, 0),
        (CodecFunction::FromFloatReference, 16, 256, 0, false, 0),
        (CodecFunction::FromFloatCpu, 18, 256, 0, false, 0),
        (CodecFunction::FromFloatCpu, 21, 256, 0, false, 0),
        (CodecFunction::FromFloatCpu, 22, 256, 0, false, 0),
        (CodecFunction::FromFloatReference, 142, 127, 0, false, 0),
        (CodecFunction::FromFloatReference, 142, 256, 1, false, 0),
        (CodecFunction::FromFloatReference, 142, 256, 0, true, 0),
        (CodecFunction::VecDot, 142, 256, 0, false, 0),
        (CodecFunction::ToFloat, 142, 256, 0, false, 1),
        (CodecFunction::ToFloat, TYPE_COUNT, 256, 0, false, 0),
        (CodecFunction::FromFloatCpu, 0, 0, 0, false, 0),
    ] {
        assert!(
            owner
                .call(
                    function,
                    kind,
                    count,
                    region(data, input_offset),
                    with_rhs.then_some(region(data, 0)),
                    region(output, output_offset)
                )
                .is_err()
        );
        assert_eq!(owner.read(region(output, 0), 1024)?, &[0x5a; 1024]);
    }
    owner.write(region(data, 0), &f32::NAN.to_le_bytes())?;
    assert!(
        owner
            .call(
                CodecFunction::FromFloatReference,
                142,
                256,
                region(data, 0),
                None,
                region(output, 0)
            )
            .is_err()
    );
    assert_eq!(owner.read(region(output, 0), 1024)?, &[0x5a; 1024]);
    assert!(
        owner
            .call(
                CodecFunction::FromFloatCpu,
                0,
                256,
                region(data, 0),
                None,
                region(output, 0)
            )
            .is_err()
    );
    assert_eq!(owner.read(region(output, 0), 1024)?, &[0x5a; 1024]);
    Ok(())
}

#[test]
fn borrowed_decoder_clears_output_on_every_failure_and_preserves_capacity() -> TestResult {
    let mut output = Vec::with_capacity(512);
    for (kind, count, bytes) in [
        (TYPE_COUNT, 1, &[][..]),
        (142, 0, &[][..]),
        (142, 127, &[][..]),
        (142, 128, &[][..]),
        (0, 1, &f32::INFINITY.to_le_bytes()[..]),
    ] {
        output.push(1.0);
        assert!(decode_row_into(kind, count, bytes, &mut output).is_err());
        assert_eq!(output, [0.0_f32; 0]);
        assert_eq!(output.capacity(), 512);
    }
    let input = [&[0xff][..], &1.25_f32.to_le_bytes()[..]].concat();
    decode_row_into(0, 1, &input[1..], &mut output)?;
    assert_eq!(output, [1.25]);
    assert_eq!(output.capacity(), 512);
    let partial_failure = [1.0_f32, f32::NAN]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    assert!(decode_row_into(0, 2, &partial_failure, &mut output).is_err());
    assert_eq!(output, [0.0_f32; 0]);
    assert_eq!(output.capacity(), 512);
    Ok(())
}

#[test]
fn cpu_f32_identity_preserves_unaligned_tail_bits_with_overlapping_output() -> TestResult {
    let mut owner = CodecBuffers::default();
    let data = owner.new_buffer(24, 32, 1)?;
    let source = [0x0000_0001_u32, 0x8000_0000, 0x3fa0_0000]
        .iter()
        .flat_map(|bits| bits.to_le_bytes())
        .collect::<Vec<_>>();
    owner.write(region(data, 3), &source)?;
    owner.call(
        CodecFunction::FromFloatCpu,
        0,
        3,
        region(data, 3),
        None,
        region(data, 7),
    )?;
    assert_eq!(owner.read(region(data, 7), 12)?, source);
    Ok(())
}

#[test]
fn decode_and_dot_numerical_failures_preserve_persistent_output() -> TestResult {
    let mut owner = CodecBuffers::default();
    let input = owner.new_buffer(1024, 1, 0)?;
    let rhs = owner.new_buffer(292, 1, 0)?;
    let output = owner.new_buffer(8, 1, 0)?;
    owner.write(region(output, 0), &[0x5a; 8])?;
    let invalid = [1.0_f32, f32::NAN]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    owner.write(region(input, 0), &invalid)?;
    assert!(
        owner
            .call(
                CodecFunction::ToFloat,
                0,
                2,
                region(input, 0),
                None,
                region(output, 0),
            )
            .is_err()
    );
    assert_eq!(owner.read(region(output, 0), 8)?, &[0x5a; 8]);
    owner.write(region(input, 0), &[0; 68])?;
    owner.write(region(rhs, 0), &f32::NAN.to_le_bytes())?;
    assert!(
        owner
            .call(
                CodecFunction::VecDot,
                142,
                256,
                region(input, 0),
                Some(region(rhs, 0)),
                region(output, 0),
            )
            .is_err()
    );
    assert_eq!(owner.read(region(output, 0), 8)?, &[0x5a; 8]);
    Ok(())
}

#[test]
fn scalar_dot_formats_preserve_exact_tail_arithmetic() -> TestResult {
    let f32_bytes = [1.0_f32, -2.0, 0.5]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    for (kind, bytes) in [
        (0, f32_bytes),
        (
            1,
            [0x3c00_u16, 0xc000, 0x3800]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect(),
        ),
        (
            30,
            [0x3f80_u16, 0xc000, 0x3f00]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect(),
        ),
    ] {
        let mut owner = CodecBuffers::default();
        let data = owner.new_buffer(bytes.len() + 1, 32, 1)?;
        owner.write(region(data, 1), &bytes)?;
        owner.call(
            CodecFunction::VecDot,
            kind,
            3,
            region(data, 1),
            Some(region(data, 1)),
            region(data, 1),
        )?;
        assert_eq!(owner.read(region(data, 1), 4)?, 5.25_f32.to_le_bytes());
    }
    Ok(())
}
