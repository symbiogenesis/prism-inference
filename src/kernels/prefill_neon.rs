//! Baseline NEON over one predecoded PQ2/PTQ1 block, reused across token dots.

use fearless_simd::{Neon, i8x16, i16x8, prelude::*, u8x16, u16x8};

pub(in crate::kernels::prefill) struct Block {
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

    pub(super) fn from_pq2(packed: &[u8; 32]) -> Option<Self> {
        let neon = fearless_simd::Level::baseline().as_neon()?;
        let mut weights = [i16x8::splat(neon, 0); 16];
        for (half, bytes) in packed.as_chunks::<16>().0.iter().enumerate() {
            let raw = u8x16::from_slice(neon, bytes);
            let a = raw & 3;
            let b = (raw >> 2) & 3;
            let c = (raw >> 4) & 3;
            let d = raw >> 6;
            let ab_low: u16x8<Neon> = a.zip_low(b).bitcast();
            let ab_high: u16x8<Neon> = a.zip_high(b).bitcast();
            let cd_low: u16x8<Neon> = c.zip_low(d).bitcast();
            let cd_high: u16x8<Neon> = c.zip_high(d).bitcast();
            let decoded = [
                ab_low.zip_low(cd_low).bitcast::<i8x16<Neon>>() - 1,
                ab_low.zip_high(cd_low).bitcast::<i8x16<Neon>>() - 1,
                ab_high.zip_low(cd_high).bitcast::<i8x16<Neon>>() - 1,
                ab_high.zip_high(cd_high).bitcast::<i8x16<Neon>>() - 1,
            ];
            for (group, vector) in decoded.into_iter().enumerate() {
                let (low, high) = vector.widen();
                weights[half * 8 + group * 2] = low;
                weights[half * 8 + group * 2 + 1] = high;
            }
        }
        Some(Self { neon, weights })
    }

    #[inline]
    pub(super) fn dot128(&self, values: &[i8; 128]) -> i32 {
        // Each signed product is in -256..=254 for accepted -1..=2 weights.
        // Sixteen products per i16 lane are bounded by 4,096 before widening.
        let mut acc = i16x8::splat(self.neon, 0);
        for (index, values) in values.as_chunks::<16>().0.iter().enumerate() {
            let (low, high) = i8x16::from_slice(self.neon, values).widen();
            acc += self.weights[index * 2] * low + self.weights[index * 2 + 1] * high;
        }
        // The complete dot's absolute bound is 128 * 2 * 128 = 32,768.
        let (sum_low, sum_high) = acc.widen();
        (sum_low + sum_high).reduce_sum()
    }

    #[inline]
    pub(super) fn dot128_four(&self, values: [&[i8; 128]; 4]) -> [i32; 4] {
        let mut acc = [i16x8::splat(self.neon, 0); 4];
        let v0 = values[0].as_chunks::<16>().0;
        let v1 = values[1].as_chunks::<16>().0;
        let v2 = values[2].as_chunks::<16>().0;
        let v3 = values[3].as_chunks::<16>().0;
        for index in 0..8 {
            let w_low = self.weights[index * 2];
            let w_high = self.weights[index * 2 + 1];
            let (l0, h0) = i8x16::from_slice(self.neon, &v0[index]).widen();
            let (l1, h1) = i8x16::from_slice(self.neon, &v1[index]).widen();
            let (l2, h2) = i8x16::from_slice(self.neon, &v2[index]).widen();
            let (l3, h3) = i8x16::from_slice(self.neon, &v3[index]).widen();
            acc[0] += w_low * l0 + w_high * h0;
            acc[1] += w_low * l1 + w_high * h1;
            acc[2] += w_low * l2 + w_high * h2;
            acc[3] += w_low * l3 + w_high * h3;
        }
        let (s0_l, s0_h) = acc[0].widen();
        let (s1_l, s1_h) = acc[1].widen();
        let (s2_l, s2_h) = acc[2].widen();
        let (s3_l, s3_h) = acc[3].widen();
        let r0 = s0_l + s0_h;
        let r1 = s1_l + s1_h;
        let r2 = s2_l + s2_h;
        let r3 = s3_l + s3_h;
        let p01 = r0.unzip_low(r1) + r0.unzip_high(r1);
        let p23 = r2.unzip_low(r3) + r2.unzip_high(r3);
        (p01.unzip_low(p23) + p01.unzip_high(p23)).to_array()
    }

    #[inline]
    pub(super) fn dot32(&self, group: usize, values: &[i8; 32]) -> i32 {
        let halves = values.as_chunks::<16>().0;
        let (l0, h0) = i8x16::from_slice(self.neon, &halves[0]).widen();
        let (l1, h1) = i8x16::from_slice(self.neon, &halves[1]).widen();
        let pairs = (self.weights[group * 4] * l0 + self.weights[group * 4 + 1] * h0)
            + (self.weights[group * 4 + 2] * l1 + self.weights[group * 4 + 3] * h1);
        let (sum_low, sum_high) = pairs.widen();
        (sum_low + sum_high).reduce_sum()
    }

    #[inline]
    pub(super) fn lanes32(&self, group: usize, values: &[i8; 32]) -> [i32; 4] {
        let halves = values.as_chunks::<16>().0;
        let (l0, h0) = i8x16::from_slice(self.neon, &halves[0]).widen();
        let (l1, h1) = i8x16::from_slice(self.neon, &halves[1]).widen();
        // Native DOTPROD lane n sums adjacent groups 4*n and 16+4*n.
        // Keep these four integer sums separate for the original FP32 FMAs.
        let sum_01 = self.weights[group * 4] * l0 + self.weights[group * 4 + 2] * l1;
        let sum_23 = self.weights[group * 4 + 1] * h0 + self.weights[group * 4 + 3] * h1;
        let (lane0, lane1) = sum_01.widen();
        let (lane2, lane3) = sum_23.widen();
        [
            lane0.reduce_sum(),
            lane1.reduce_sum(),
            lane2.reduce_sum(),
            lane3.reduce_sum(),
        ]
    }
}
