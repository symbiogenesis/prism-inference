#![forbid(unsafe_code)]
//! Independent wire-format, numerical-layout, and malformed-input regressions.
#![expect(
    clippy::unwrap_used,
    reason = "test construction and successful-operation assertions must fail visibly"
)]

use bonsai_ternary::{Error, Pq2Block, Q8KBlock, pack_ptq, unpack_ptq};

fn zeros(scale: u16) -> Pq2Block {
    let mut block = [0x55; 34];
    block[..2].copy_from_slice(&scale.to_le_bytes());
    block
}

const fn set_code(block: &mut Pq2Block, index: usize, code: u8) {
    let shift = 2 * (index % 4);
    block[2 + index / 4] = (block[2 + index / 4] & !(3 << shift)) | (code << shift);
}

#[test]
fn q8k_block_preserves_native_abi_layout() {
    // Exact binary16 edge bits are already checked against all 26 captured
    // native reference cases in cpp_reference_ptq_parity.
    assert_eq!(size_of::<Q8KBlock>(), 292);
    assert_eq!(std::mem::offset_of!(Q8KBlock, values), 4);
    assert_eq!(std::mem::offset_of!(Q8KBlock, sums), 260);
}

#[test]
fn ptq_has_known_wire_bytes() {
    for (symbols, trit_byte, tail_byte) in [
        ([0; 32], 0, 0),
        ([0x55; 32], 128, 127),
        ([0xaa; 32], 255, 253),
    ] {
        let mut block = zeros(0xbc00);
        block[2..].copy_from_slice(&symbols);
        let packed = pack_ptq(&block).unwrap();
        assert_eq!(&packed[..24], &[trit_byte; 24]);
        assert_eq!(&packed[24..26], &[tail_byte; 2]);
        assert_eq!(&packed[26..], &[0, 0xbc]);
        assert_eq!(unpack_ptq(&packed).unwrap(), block);
    }
}

#[test]
fn every_density_preserves_ptq_round_trip() {
    for count in 0..=128 {
        let mut source = zeros(0x83ff);
        for index in 0..count {
            set_code(
                &mut source,
                (index * 37) % 128,
                if index % 2 == 0 { 0 } else { 2 },
            );
        }
        assert_eq!(unpack_ptq(&pack_ptq(&source).unwrap()).unwrap(), source);
    }
}

#[test]
fn every_pq2_byte_is_classified_by_the_ternary_symbol_rule() {
    // Every single-symbol mutation is included among these complete byte
    // patterns, alongside combinations of all four neighboring symbols.
    for byte in 0..=255_u8 {
        let valid = (0..4).all(|index| (byte >> (2 * index)) & 3 != 3);
        for position in 2..34 {
            let mut source = zeros(0x3c00);
            source[position] = byte;
            if valid {
                assert_eq!(unpack_ptq(&pack_ptq(&source).unwrap()).unwrap(), source);
            } else {
                assert_eq!(pack_ptq(&source), Err(Error::NonTernarySymbol));
            }
        }
    }
}

#[test]
fn reviewed_finite_scales_preserve_exact_wire_bits() {
    for scale in [
        0x0001, 0x03ff, 0x0400, 0x3555, 0x3c00, 0x7bff, 0x8001, 0x83ff, 0x8400, 0xbc00, 0xfbff,
    ] {
        let mut source = zeros(scale);
        source[2] = 0x24;
        let packed = pack_ptq(&source).unwrap();
        assert_eq!(&packed[26..], &scale.to_le_bytes());
        assert_eq!(unpack_ptq(&packed).unwrap(), source);
    }
}

#[test]
fn every_native_lane_word_has_independent_base_three_symbols() {
    let mut checked = 0;
    let mut rejected = 0;
    for lane in 0..26 {
        let (digits, words, first, stride) = match lane {
            0..16 => (5, 243_u32, lane, 16),
            16..24 => (5, 243, 80 + lane - 16, 8),
            _ => (4, 81, 120 + lane - 24, 2),
        };
        let mut canonical = [false; 256];
        for word in 0..words {
            let numerator = if digits == 4 { word * 3 } else { word };
            let stored = u8::try_from((numerator * 256).div_ceil(243)).unwrap();
            assert!(!canonical[usize::from(stored)], "lane {lane}, word {word}");
            canonical[usize::from(stored)] = true;
            let mut packed = [0; 28];
            packed[27] = 0x3c;
            packed[lane] = stored;
            let mut expected = zeros(0x3c00);
            expected[2..].fill(0);
            // The oracle uses independent integer base-three division and
            // pinned native lane positions, rather than production unpacking
            // or a pack/unpack round trip to derive the expected symbols.
            for trit in 0..digits {
                let digit = word / 3_u32.pow(u32::try_from(digits - 1 - trit).unwrap()) % 3;
                set_code(
                    &mut expected,
                    first + trit * stride,
                    u8::try_from(digit).unwrap(),
                );
            }
            assert_eq!(
                unpack_ptq(&packed).unwrap(),
                expected,
                "lane {lane}, word {word}"
            );
            assert_eq!(
                pack_ptq(&expected).unwrap(),
                packed,
                "lane {lane}, word {word}"
            );
            checked += 1;
        }
        for byte in 0..=255_u8 {
            if !canonical[usize::from(byte)] {
                let mut packed = [0; 28];
                packed[27] = 0x3c;
                packed[lane] = byte;
                assert_eq!(
                    unpack_ptq(&packed),
                    Err(Error::NoncanonicalPtq),
                    "lane {lane}, byte {byte}"
                );
                rejected += 1;
            }
        }
    }
    // These two counts cover every byte at every position, including aliases
    // such as byte 1 that decode like a canonical encoding but must be rejected.
    assert_eq!((checked, rejected), (5994, 662));
}
