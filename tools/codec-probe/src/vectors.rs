//! Fixed synthetic corpus and an FP32 oracle independent of production codecs.

use bonsai_ternary::{Pq2Block, PtqBlock, pack_ptq, unpack_ptq};

// Reviewed IEEE 754 binary16/binary32 bit pairs. Do not call the production
// half_to_float here: this independently anchors the native numerical checks.
const SCALES: [(u16, u32); 12] = [
    (0x3c00, 0x3f80_0000),
    (0xbc00, 0xbf80_0000),
    (0x0001, 0x3380_0000),
    (0x8001, 0xb380_0000),
    (0x03ff, 0x387f_c000),
    (0x83ff, 0xb87f_c000),
    (0x0400, 0x3880_0000),
    (0x8400, 0xb880_0000),
    (0x7bff, 0x477f_e000),
    (0xfbff, 0xc77f_e000),
    (0x3555, 0x3eaa_a000),
    (0xb555, 0xbeaa_a000),
];

fn block(scale: u16) -> Pq2Block {
    let mut block = [0x55; 34];
    block[..2].copy_from_slice(&scale.to_le_bytes());
    block
}

const fn set(block: &mut Pq2Block, index: usize, code: u8) {
    let shift = 2 * (index % 4);
    block[2 + index / 4] = (block[2 + index / 4] & !(3 << shift)) | (code << shift);
}

type NativeValues = ([f32; 128], [f32; 128]);

fn compare(
    source: &Pq2Block,
    decode: &mut impl FnMut(&Pq2Block, &PtqBlock) -> Result<NativeValues, String>,
) -> Result<(), String> {
    let packed = pack_ptq(source).map_err(|error| error.to_string())?;
    if unpack_ptq(&packed).map_err(|error| error.to_string())? != *source {
        return Err("The production codec changed a synthetic symbol or scale bit.".into());
    }
    let (original, repacked) = decode(source, &packed)?;
    let scale = u16::from_le_bytes([source[0], source[1]]);
    let bits = SCALES
        .iter()
        .find(|pair| pair.0 == scale)
        .ok_or("The synthetic scale has no reviewed independent FP32 oracle.")?
        .1;
    let scale = f32::from_bits(bits);
    for index in 0..128 {
        let code = (source[2 + index / 4] >> (2 * (index % 4))) & 3;
        let expected = (scale * (f32::from(code) - 1.0)).to_bits();
        if original[index].to_bits() != expected || repacked[index].to_bits() != expected {
            return Err(format!(
                "Native PQ2/PTQ1 dequantization changed FP32 bits at index {index}, including signed zero."
            ));
        }
    }
    Ok(())
}

// Frozen from the original System.Random(7310787) corpus; see fixture provenance.
const RANDOM_BLOCKS: &[u8; 34 * 256] = include_bytes!("../tests/fixtures/random-pq2.bin");

/// Checks every fixed block against the independent expected numerical bits.
#[expect(
    clippy::redundant_pub_crate,
    reason = "binary implementation modules intentionally expose only to their parent"
)]
pub(super) fn check_vectors(
    mut decode: impl FnMut(&Pq2Block, &PtqBlock) -> Result<NativeValues, String>,
) -> Result<usize, String> {
    let mut count = 0;
    for (scale, _) in SCALES {
        for code in 0..3 {
            let mut source = block(scale);
            for index in 0..128 {
                set(&mut source, index, code);
            }
            compare(&source, &mut decode)?;
            count += 1;
        }
    }
    for index in 0..128 {
        for code in 0..3 {
            let mut source = block(0x3555);
            set(&mut source, index, code);
            compare(&source, &mut decode)?;
            count += 1;
        }
    }
    for lane in 0..26 {
        let (digits, words, first, stride) = match lane {
            0..16 => (5, 243_u32, lane, 16),
            16..24 => (5, 243, 80 + lane - 16, 8),
            _ => (4, 81, 120 + lane - 24, 2),
        };
        for word in 0..words {
            let mut source = block(if lane < 24 { 0xbc00 } else { 0x0001 });
            let mut remaining = word;
            for trit in (0..digits).rev() {
                set(
                    &mut source,
                    first + trit * stride,
                    u8::try_from(remaining % 3).map_err(|error| error.to_string())?,
                );
                remaining /= 3;
            }
            compare(&source, &mut decode)?;
            count += 1;
        }
    }
    for source in RANDOM_BLOCKS.as_chunks::<34>().0 {
        compare(source, &mut decode)?;
        count += 1;
    }
    if count != 6670 {
        return Err("The fixed native codec vector count changed.".into());
    }
    Ok(count)
}
