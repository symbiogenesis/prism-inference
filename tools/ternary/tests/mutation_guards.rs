#![forbid(unsafe_code)]
//! Tiny independent oracles for the bounded source-mutation catalog.
#![expect(
    clippy::unwrap_used,
    reason = "successful codec operations are explicit test assertions"
)]

use bonsai_ternary::{Error, Pq2Block, PtqBlock, pack_ptq, unpack_ptq};

fn negative_weights(scale: u16) -> Pq2Block {
    let mut source = [0; 34];
    source[..2].copy_from_slice(&scale.to_le_bytes());
    source
}

fn negative_wire(scale: u16) -> PtqBlock {
    let mut wire = [0; 28];
    wire[26..].copy_from_slice(&scale.to_le_bytes());
    wire
}

fn assert_scale_round_trip(scale: u16) {
    let source = negative_weights(scale);
    let wire = negative_wire(scale);
    assert_eq!(pack_ptq(&source).unwrap(), wire);
    assert_eq!(unpack_ptq(&wire).unwrap(), source);
}

fn assert_scale_rejected(scale: u16) {
    assert_eq!(
        pack_ptq(&negative_weights(scale)),
        Err(Error::InvalidWeightScale)
    );
    assert_eq!(
        unpack_ptq(&negative_wire(scale)),
        Err(Error::InvalidWeightScale)
    );
}

#[test]
fn guard_zero_scale_boundary() {
    // Both signs of the smallest subnormal remain valid beside signed zero.
    for scale in [0x0001, 0x8001] {
        assert_scale_round_trip(scale);
    }
    for scale in [0x0000, 0x8000] {
        assert_scale_rejected(scale);
    }
}

#[test]
fn guard_nonfinite_scale_boundary() {
    for scale in [0x7bff, 0xfbff] {
        assert_scale_round_trip(scale);
    }
    // Preserve both infinity signs and the reviewed signaling/quiet NaN
    // payloads formerly repeated in the general codec regressions.
    for scale in [
        0x7c00, 0xfc00, 0x7c01, 0xfc01, 0x7e00, 0xfe00, 0x7fff, 0xffff, 0x7e01,
    ] {
        assert_scale_rejected(scale);
    }
}

#[test]
fn guard_isolated_reserved_symbol() {
    // A single forbidden symbol must fail even when its three neighbors are -1.
    // Valid neighbors +1 and zero must continue to pack successfully.
    for index in [0, 3, 64, 79, 80, 119, 120, 127] {
        for symbol in [1_u8, 2] {
            let mut source = negative_weights(0x3c00);
            source[2 + index / 4] |= symbol << (2 * (index % 4));
            assert_eq!(unpack_ptq(&pack_ptq(&source).unwrap()).unwrap(), source);
        }
        let mut source = negative_weights(0x3c00);
        source[2 + index / 4] |= 3 << (2 * (index % 4));
        assert_eq!(pack_ptq(&source), Err(Error::NonTernarySymbol));
    }
}

#[test]
fn guard_five_trit_canonical_alias() {
    // Wire byte 0 is canonical. Byte 1 decodes identically but is forbidden.
    let canonical = negative_wire(0xbc00);
    assert_eq!(unpack_ptq(&canonical).unwrap(), negative_weights(0xbc00));
    for position in [0, 15, 16, 23] {
        let mut alias = canonical;
        alias[position] = 1;
        assert_eq!(unpack_ptq(&alias), Err(Error::NoncanonicalPtq));
    }
}

#[test]
fn guard_four_trit_implicit_zero() {
    // Five-trit wire byte 2 encodes 00001 and is invalid in a four-trit tail.
    // Tail byte 4 encodes 00010, whose implicit final zero is canonical.
    for (tail, index) in [(24, 126), (25, 127)] {
        let mut canonical = negative_wire(0x3c00);
        canonical[tail] = 4;
        let mut source = negative_weights(0x3c00);
        source[2 + index / 4] |= 1 << (2 * (index % 4));
        assert_eq!(pack_ptq(&source).unwrap(), canonical);
        assert_eq!(unpack_ptq(&canonical).unwrap(), source);
        canonical[tail] = 2;
        assert_eq!(unpack_ptq(&canonical), Err(Error::NoncanonicalPtq));
    }
}

#[test]
fn guard_pack_rounds_up() {
    // Base-three value 00001 has the independently calculated wire byte
    // ceil(256 / 243) = 2, rather than the truncated byte 1.
    let mut source = negative_weights(0x3c00);
    source[2 + 64 / 4] = 1;
    let mut expected = negative_wire(0x3c00);
    expected[0] = 2;
    assert_eq!(pack_ptq(&source).unwrap(), expected);
    assert_eq!(unpack_ptq(&expected).unwrap(), source);
    expected[0] = 1;
    assert_eq!(unpack_ptq(&expected), Err(Error::NoncanonicalPtq));
}
