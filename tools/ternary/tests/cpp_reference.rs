#![forbid(unsafe_code)]
//! Exact PQ2/PTQ and binary16 parity against captured native C++ results.
//!
//! Every read is bounded, fields use little endian byte order, and unexpected
//! trailing bytes are rejected. The fixture contains only codec evidence.
#![expect(
    clippy::unwrap_used,
    reason = "corrupt fixtures and failed parity assertions must fail visibly"
)]

use bonsai_ternary::{Pq2Block, PtqBlock, half_to_float, pack_ptq, unpack_ptq};

struct Reader<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> &'a [u8] {
        let end = self.cursor.checked_add(count).unwrap();
        let result = self.bytes.get(self.cursor..end).unwrap();
        self.cursor = end;
        result
    }

    fn array<const N: usize>(&mut self) -> [u8; N] {
        self.take(N).try_into().unwrap()
    }

    fn u16(&mut self) -> u16 {
        u16::from_le_bytes(self.array())
    }

    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.array())
    }

    fn count(&mut self) -> usize {
        usize::try_from(self.u32()).unwrap()
    }
}

#[test]
fn cpp_reference_ptq_parity() {
    let fixture = include_bytes!("fixtures/ptq-reference.bin");
    let mut reader = Reader {
        bytes: fixture,
        cursor: 0,
    };
    assert_eq!(reader.take(8), b"PTPAR001");
    let half_count = reader.count();
    assert_eq!(half_count, 26);
    for _ in 0..half_count {
        let input = reader.u16();
        let expected = reader.u32();
        assert_eq!(half_to_float(input).to_bits(), expected, "half {input:04x}");
    }
    let count = reader.count();
    assert_eq!(count, 161);
    let mut row_count = 0;
    let mut widths = [false; 4];
    for index in 0..count {
        let rows = reader.count();
        let blocks = reader.count();
        assert!((1..=3).contains(&rows));
        assert!([2, 40, 48, 136].contains(&blocks));
        row_count += rows;
        widths[[2, 40, 48, 136]
            .iter()
            .position(|width| *width == blocks)
            .unwrap()] = true;
        let length = rows.checked_mul(blocks).unwrap();
        let pq2 = (0..length)
            .map(|_| reader.array::<34>())
            .collect::<Vec<Pq2Block>>();
        let ptq = (0..length)
            .map(|_| reader.array::<28>())
            .collect::<Vec<PtqBlock>>();
        for (pq2, ptq) in pq2.iter().zip(&ptq) {
            assert_eq!(pack_ptq(pq2).unwrap(), *ptq, "case {index}");
            assert_eq!(unpack_ptq(ptq).unwrap(), *pq2, "case {index}");
        }
    }
    assert_eq!(row_count, 169);
    assert!(widths.into_iter().all(|present| present));
    assert_eq!(reader.cursor, fixture.len(), "unparsed C++ fixture bytes");
}
