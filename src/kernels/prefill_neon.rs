//! Baseline NEON over one predecoded PQ2/PTQ1 block, reused across token dots.

use fearless_simd::{Neon, i8x16, i16x8, i32x4, prelude::*};

pub(super) struct Block {
    neon: Neon,
    weights: [i16x8<Neon>; 16],
}

impl Block {
    pub(super) fn new(weights: &[i16; 128]) -> Option<Self> {
        let neon = fearless_simd::Level::baseline().as_neon()?;
        Some(Self {
            neon,
            weights: std::array::from_fn(|index| {
                i16x8::from_slice(neon, &weights[index * 8..index * 8 + 8])
            }),
        })
    }

    /// Each signed product is in -256..=254 for accepted -1..=2 weights.
    /// Widen before summation; no i16 accumulation or optional ISA is used.
    #[inline]
    fn products16(&self, start: usize, values: &[i8; 16]) -> [i32x4<Neon>; 4] {
        let (low, high) = i8x16::from_slice(self.neon, values).widen();
        let (a, b) = (self.weights[start] * low).widen();
        let (c, d) = (self.weights[start + 1] * high).widen();
        [a, b, c, d]
    }

    #[inline]
    pub(super) fn dot128(&self, values: &[i8; 128]) -> i32 {
        let mut even = i32x4::splat(self.neon, 0);
        let mut odd = i32x4::splat(self.neon, 0);
        for (index, values) in values.as_chunks::<16>().0.iter().enumerate() {
            let [a, b, c, d] = self.products16(index * 2, values);
            even += a + c;
            odd += b + d;
        }
        // The complete dot's absolute bound is 128 * 2 * 128 = 32,768.
        (even + odd).reduce_sum()
    }

    #[inline]
    pub(super) fn dot32(&self, group: usize, values: &[i8; 32]) -> i32 {
        let halves = values.as_chunks::<16>().0;
        let [a, b, c, d] = self.products16(group * 4, &halves[0]);
        let [e, f, g, h] = self.products16(group * 4 + 2, &halves[1]);
        ((a + b) + (c + d) + (e + f) + (g + h)).reduce_sum()
    }

    #[inline]
    pub(super) fn lanes32(&self, group: usize, values: &[i8; 32]) -> [i32; 4] {
        let halves = values.as_chunks::<16>().0;
        let low = self.products16(group * 4, &halves[0]);
        let high = self.products16(group * 4 + 2, &halves[1]);
        // Native DOTPROD lane n sums adjacent groups 4*n and 16+4*n.
        // Keep these four integer sums separate for the original FP32 FMAs.
        std::array::from_fn(|lane| (low[lane] + high[lane]).reduce_sum())
    }
}
