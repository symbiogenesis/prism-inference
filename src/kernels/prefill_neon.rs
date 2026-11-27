//! Baseline NEON over one predecoded PQ2/PTQ1 block, reused across token dots.

use fearless_simd::{Neon, f32x4, f64x2, i8x16, i16x8, i32x4, prelude::*, u8x16, u16x8};

#[inline]
fn reduce4_i32x4_vec(
    r0: i32x4<Neon>,
    r1: i32x4<Neon>,
    r2: i32x4<Neon>,
    r3: i32x4<Neon>,
) -> i32x4<Neon> {
    let p01 = r0.unzip_low(r1) + r0.unzip_high(r1);
    let p23 = r2.unzip_low(r3) + r2.unzip_high(r3);
    p01.unzip_low(p23) + p01.unzip_high(p23)
}

#[inline]
fn reduce4_i32x4(r0: i32x4<Neon>, r1: i32x4<Neon>, r2: i32x4<Neon>, r3: i32x4<Neon>) -> [i32; 4] {
    reduce4_i32x4_vec(r0, r1, r2, r3).to_array()
}

pub(in crate::kernels::prefill) struct Block {
    neon: Neon,
    weights: [i16x8<Neon>; 16],
}

impl Block {
    #[cfg(test)]
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
        Some(Self {
            neon,
            weights: decode_pq2_sequential(neon, packed),
        })
    }

    pub(super) fn from_ptq1(packed: &[u8; 28]) -> Option<Self> {
        let neon = fearless_simd::Level::baseline().as_neon()?;
        let weights = crate::kernels::neon::decode_ptq1_i16(neon, packed);
        Some(Self { neon, weights })
    }

    #[inline]
    pub(super) fn accumulate_lanes128(
        &self,
        weight_scale: f32,
        activations: &[crate::kernels::Q8Block; 4],
        lanes: &mut [f32; 4],
    ) {
        let mut acc = f32x4::from_slice(self.neon, lanes);
        for (group, act) in activations.iter().enumerate() {
            let halves = act.values.as_chunks::<16>().0;
            let (l0, h0) = i8x16::from_slice(self.neon, &halves[0]).widen();
            let (l1, h1) = i8x16::from_slice(self.neon, &halves[1]).widen();
            let sum_01 = self.weights[group * 4] * l0 + self.weights[group * 4 + 2] * l1;
            let sum_23 = self.weights[group * 4 + 1] * h0 + self.weights[group * 4 + 3] * h1;
            let (q0, q1) = sum_01.widen();
            let (q2, q3) = sum_23.widen();
            let floats = reduce4_i32x4_vec(q0, q1, q2, q3).to_float::<f32x4<Neon>>();
            let d = f32x4::splat(self.neon, weight_scale * act.scale);
            acc = floats.mul_add_precise(d, acc);
        }
        acc.store_slice(lanes);
    }

    #[inline]
    pub(super) fn dot128_q8(&self, activations: &[crate::kernels::Q8Block; 4]) -> f32 {
        let group_sums: [i32x4<Neon>; 4] = std::array::from_fn(|group| {
            let halves = activations[group].values.as_chunks::<16>().0;
            let (l0, h0) = i8x16::from_slice(self.neon, &halves[0]).widen();
            let (l1, h1) = i8x16::from_slice(self.neon, &halves[1]).widen();
            let pairs = (self.weights[group * 4] * l0 + self.weights[group * 4 + 1] * h0)
                + (self.weights[group * 4 + 2] * l1 + self.weights[group * 4 + 3] * h1);
            let (sum_low, sum_high) = pairs.widen();
            sum_low + sum_high
        });
        let dots = reduce4_i32x4_vec(group_sums[0], group_sums[1], group_sums[2], group_sums[3])
            .to_float::<f32x4<Neon>>();
        let scales = f32x4::from_slice(
            self.neon,
            &[
                activations[0].scale,
                activations[1].scale,
                activations[2].scale,
                activations[3].scale,
            ],
        );
        let prod = (dots * scales).to_array();
        (((0.0 + prod[0]) + prod[1]) + prod[2]) + prod[3]
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
        reduce4_i32x4(s0_l + s0_h, s1_l + s1_h, s2_l + s2_h, s3_l + s3_h)
    }

    #[inline]
    pub(super) fn dot128_eight(&self, values: [&[i8; 128]; 8]) -> [i32; 8] {
        let mut acc = [i16x8::splat(self.neon, 0); 8];
        let chunks = values.map(|token| token.as_chunks::<16>().0);
        for index in 0..8 {
            let w_low = self.weights[index * 2];
            let w_high = self.weights[index * 2 + 1];
            for token in 0..8 {
                let (low, high) = i8x16::from_slice(self.neon, &chunks[token][index]).widen();
                acc[token] += w_low * low + w_high * high;
            }
        }
        let (s0_l, s0_h) = acc[0].widen();
        let (s1_l, s1_h) = acc[1].widen();
        let (s2_l, s2_h) = acc[2].widen();
        let (s3_l, s3_h) = acc[3].widen();
        let (s4_l, s4_h) = acc[4].widen();
        let (s5_l, s5_h) = acc[5].widen();
        let (s6_l, s6_h) = acc[6].widen();
        let (s7_l, s7_h) = acc[7].widen();
        let first = reduce4_i32x4(s0_l + s0_h, s1_l + s1_h, s2_l + s2_h, s3_l + s3_h);
        let second = reduce4_i32x4(s4_l + s4_h, s5_l + s5_h, s6_l + s6_h, s7_l + s7_h);
        [
            first[0], first[1], first[2], first[3], second[0], second[1], second[2], second[3],
        ]
    }

    #[cfg(test)]
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

    #[cfg(test)]
    #[inline]
    pub(super) fn dot32_four(&self, group: usize, values: [&[i8; 32]; 4]) -> [i32; 4] {
        let w0 = self.weights[group * 4];
        let w1 = self.weights[group * 4 + 1];
        let w2 = self.weights[group * 4 + 2];
        let w3 = self.weights[group * 4 + 3];
        let sums = values.map(|token| {
            let halves = token.as_chunks::<16>().0;
            let (l0, h0) = i8x16::from_slice(self.neon, &halves[0]).widen();
            let (l1, h1) = i8x16::from_slice(self.neon, &halves[1]).widen();
            let pairs = (w0 * l0 + w1 * h0) + (w2 * l1 + w3 * h1);
            let (sum_low, sum_high) = pairs.widen();
            sum_low + sum_high
        });
        reduce4_i32x4(sums[0], sums[1], sums[2], sums[3])
    }

    #[cfg(test)]
    #[inline]
    pub(super) fn dot32_eight(&self, group: usize, values: [&[i8; 32]; 8]) -> [i32; 8] {
        let w0 = self.weights[group * 4];
        let w1 = self.weights[group * 4 + 1];
        let w2 = self.weights[group * 4 + 2];
        let w3 = self.weights[group * 4 + 3];
        let sums = values.map(|token| {
            let halves = token.as_chunks::<16>().0;
            let (l0, h0) = i8x16::from_slice(self.neon, &halves[0]).widen();
            let (l1, h1) = i8x16::from_slice(self.neon, &halves[1]).widen();
            let pairs = (w0 * l0 + w1 * h0) + (w2 * l1 + w3 * h1);
            let (sum_low, sum_high) = pairs.widen();
            sum_low + sum_high
        });
        let first = reduce4_i32x4(sums[0], sums[1], sums[2], sums[3]);
        let second = reduce4_i32x4(sums[4], sums[5], sums[6], sums[7]);
        [
            first[0], first[1], first[2], first[3], second[0], second[1], second[2], second[3],
        ]
    }

    #[cfg(test)]
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
        reduce4_i32x4(lane0, lane1, lane2, lane3)
    }

    #[cfg(test)]
    #[inline]
    pub(super) fn lanes32_four(&self, group: usize, values: [&[i8; 32]; 4]) -> [[i32; 4]; 4] {
        let w0 = self.weights[group * 4];
        let w1 = self.weights[group * 4 + 1];
        let w2 = self.weights[group * 4 + 2];
        let w3 = self.weights[group * 4 + 3];
        values.map(|token| {
            let halves = token.as_chunks::<16>().0;
            let (l0, h0) = i8x16::from_slice(self.neon, &halves[0]).widen();
            let (l1, h1) = i8x16::from_slice(self.neon, &halves[1]).widen();
            let sum_01 = w0 * l0 + w2 * l1;
            let sum_23 = w1 * h0 + w3 * h1;
            let (lane0, lane1) = sum_01.widen();
            let (lane2, lane3) = sum_23.widen();
            reduce4_i32x4(lane0, lane1, lane2, lane3)
        })
    }

    #[cfg(test)]
    #[inline]
    pub(super) fn lanes32_eight(&self, group: usize, values: [&[i8; 32]; 8]) -> [[i32; 4]; 8] {
        let w0 = self.weights[group * 4];
        let w1 = self.weights[group * 4 + 1];
        let w2 = self.weights[group * 4 + 2];
        let w3 = self.weights[group * 4 + 3];
        values.map(|token| {
            let halves = token.as_chunks::<16>().0;
            let (l0, h0) = i8x16::from_slice(self.neon, &halves[0]).widen();
            let (l1, h1) = i8x16::from_slice(self.neon, &halves[1]).widen();
            let sum_01 = w0 * l0 + w2 * l1;
            let sum_23 = w1 * h0 + w3 * h1;
            let (lane0, lane1) = sum_01.widen();
            let (lane2, lane3) = sum_23.widen();
            reduce4_i32x4(lane0, lane1, lane2, lane3)
        })
    }
}

#[inline]
fn decode_pq2_sequential(neon: Neon, packed: &[u8; 32]) -> [i16x8<Neon>; 16] {
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
    weights
}

#[inline]
fn decode_pq2_unsigned(neon: Neon, packed: &[u8; 32]) -> [i16x8<Neon>; 16] {
    let mut weights = [i16x8::splat(neon, 0); 16];
    let mask = u8x16::splat(neon, 0x03);
    for (half, bytes) in packed.as_chunks::<16>().0.iter().enumerate() {
        let w = u8x16::from_slice(neon, bytes);
        let (q0_low, q0_high) = (w & mask).widen();
        let (q1_low, q1_high) = ((w >> 2) & mask).widen();
        let (q2_low, q2_high) = ((w >> 4) & mask).widen();
        let (q3_low, q3_high) = (w >> 6).widen();
        let base = half * 8;
        weights[base] = q0_low.bitcast();
        weights[base + 1] = q0_high.bitcast();
        weights[base + 2] = q1_low.bitcast();
        weights[base + 3] = q1_high.bitcast();
        weights[base + 4] = q2_low.bitcast();
        weights[base + 5] = q2_high.bitcast();
        weights[base + 6] = q3_low.bitcast();
        weights[base + 7] = q3_high.bitcast();
    }
    weights
}

struct Pq2Activation {
    planes: [i16x8<Neon>; 16],
    sum128: i32,
}

#[inline]
fn q8k_half_sum(neon: Neon, sums: &[i16; 16], half: usize) -> i32 {
    let offset = half * 8;
    let (low, high) = i16x8::from_slice(neon, &sums[offset..offset + 8]).widen();
    (low + high).reduce_sum()
}

#[inline]
fn load_pq2_activation_precomputed(
    neon: Neon,
    prepared: &crate::kernels::packed_pq2::Activations,
    index: usize,
) -> Option<Pq2Activation> {
    let act = prepared.get(index)?;
    let chunks = prepared.widened.get(index)?.as_chunks::<8>().0;
    Some(Pq2Activation {
        planes: std::array::from_fn(|k| i16x8::from_slice(neon, &chunks[k])),
        sum128: act.sum,
    })
}

#[inline]
fn prepare_pq2_activation(neon: Neon, values: &[i8; 128], sum128: i32) -> Pq2Activation {
    let mut planes = [i16x8::splat(neon, 0); 16];
    for (half, bytes) in values.as_chunks::<64>().0.iter().enumerate() {
        let chunks = bytes.as_chunks::<16>().0;
        let c0 = i8x16::from_slice(neon, &chunks[0]);
        let c1 = i8x16::from_slice(neon, &chunks[1]);
        let c2 = i8x16::from_slice(neon, &chunks[2]);
        let c3 = i8x16::from_slice(neon, &chunks[3]);
        let e01 = c0.unzip_low(c1);
        let o01 = c0.unzip_high(c1);
        let e23 = c2.unzip_low(c3);
        let o23 = c2.unzip_high(c3);
        let p0 = e01.unzip_low(e23);
        let p1 = o01.unzip_low(o23);
        let p2 = e01.unzip_high(e23);
        let p3 = o01.unzip_high(o23);
        let (p0_low, p0_high) = p0.widen();
        let (p1_low, p1_high) = p1.widen();
        let (p2_low, p2_high) = p2.widen();
        let (p3_low, p3_high) = p3.widen();
        let base = half * 8;
        planes[base] = p0_low;
        planes[base + 1] = p0_high;
        planes[base + 2] = p1_low;
        planes[base + 3] = p1_high;
        planes[base + 4] = p2_low;
        planes[base + 5] = p2_high;
        planes[base + 6] = p3_low;
        planes[base + 7] = p3_high;
    }
    Pq2Activation { planes, sum128 }
}

#[inline]
fn dot128_four_rows_four_tokens(
    neon: Neon,
    weights: &[[i16x8<Neon>; 16]; 4],
    acts: &[Pq2Activation; 4],
) -> [i32x4<Neon>; 4] {
    let mut acc00 = i16x8::splat(neon, 0);
    let mut acc01 = i16x8::splat(neon, 0);
    let mut acc02 = i16x8::splat(neon, 0);
    let mut acc03 = i16x8::splat(neon, 0);
    let mut acc10 = i16x8::splat(neon, 0);
    let mut acc11 = i16x8::splat(neon, 0);
    let mut acc12 = i16x8::splat(neon, 0);
    let mut acc13 = i16x8::splat(neon, 0);
    let mut acc20 = i16x8::splat(neon, 0);
    let mut acc21 = i16x8::splat(neon, 0);
    let mut acc22 = i16x8::splat(neon, 0);
    let mut acc23 = i16x8::splat(neon, 0);
    let mut acc30 = i16x8::splat(neon, 0);
    let mut acc31 = i16x8::splat(neon, 0);
    let mut acc32 = i16x8::splat(neon, 0);
    let mut acc33 = i16x8::splat(neon, 0);
    for k in 0..16 {
        let w0 = weights[0][k];
        let w1 = weights[1][k];
        let w2 = weights[2][k];
        let w3 = weights[3][k];
        let a0 = acts[0].planes[k];
        let a1 = acts[1].planes[k];
        let a2 = acts[2].planes[k];
        let a3 = acts[3].planes[k];
        acc00 += w0 * a0;
        acc01 += w0 * a1;
        acc02 += w0 * a2;
        acc03 += w0 * a3;
        acc10 += w1 * a0;
        acc11 += w1 * a1;
        acc12 += w1 * a2;
        acc13 += w1 * a3;
        acc20 += w2 * a0;
        acc21 += w2 * a1;
        acc22 += w2 * a2;
        acc23 += w2 * a3;
        acc30 += w3 * a0;
        acc31 += w3 * a1;
        acc32 += w3 * a2;
        acc33 += w3 * a3;
    }
    let sums = i32x4::from_slice(
        neon,
        &[
            acts[0].sum128,
            acts[1].sum128,
            acts[2].sum128,
            acts[3].sum128,
        ],
    );
    let reduce_row = |c0: i16x8<Neon>, c1: i16x8<Neon>, c2: i16x8<Neon>, c3: i16x8<Neon>| {
        let (l0, h0) = c0.widen();
        let (l1, h1) = c1.widen();
        let (l2, h2) = c2.widen();
        let (l3, h3) = c3.widen();
        reduce4_i32x4_vec(l0 + h0, l1 + h1, l2 + h2, l3 + h3) - sums
    };
    [
        reduce_row(acc00, acc01, acc02, acc03),
        reduce_row(acc10, acc11, acc12, acc13),
        reduce_row(acc20, acc21, acc22, acc23),
        reduce_row(acc30, acc31, acc32, acc33),
    ]
}

#[inline]
fn dot128_four_rows_one_token(
    neon: Neon,
    weights: &[[i16x8<Neon>; 16]; 4],
    act: &Pq2Activation,
) -> [i32; 4] {
    let mut acc0 = i16x8::splat(neon, 0);
    let mut acc1 = i16x8::splat(neon, 0);
    let mut acc2 = i16x8::splat(neon, 0);
    let mut acc3 = i16x8::splat(neon, 0);
    for k in 0..16 {
        let a = act.planes[k];
        acc0 += weights[0][k] * a;
        acc1 += weights[1][k] * a;
        acc2 += weights[2][k] * a;
        acc3 += weights[3][k] * a;
    }
    let (l0, h0) = acc0.widen();
    let (l1, h1) = acc1.widen();
    let (l2, h2) = acc2.widen();
    let (l3, h3) = acc3.widen();
    let sums = i32x4::splat(neon, act.sum128);
    (reduce4_i32x4_vec(l0 + h0, l1 + h1, l2 + h2, l3 + h3) - sums).to_array()
}

#[expect(
    clippy::suboptimal_flops,
    clippy::too_many_lines,
    reason = "Preserve separately rounded source arithmetic and four-row × four-token register blocking."
)]
pub(in crate::kernels::prefill) fn dot_pq2_four_rows<const TOKENS: usize>(
    rows: [&[u8]; 4],
    preparations: &[crate::kernels::MatvecScratch],
    pinned: bool,
    decoded_blocks: &mut u64,
) -> Option<crate::Result<[[f32; TOKENS]; 4]>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let mut sums = [[0.0_f32; TOKENS]; 4];
    let r0 = rows[0].as_chunks::<34>().0;
    let r1 = rows[1].as_chunks::<34>().0;
    let r2 = rows[2].as_chunks::<34>().0;
    let r3 = rows[3].as_chunks::<34>().0;
    let has_precomputed = preparations
        .iter()
        .all(|scratch| scratch.pq2_consumption.len() >= r0.len());
    let dotprod = if has_precomputed {
        preparations.first()?.pq2_consumption.dotprod
    } else {
        bonsai_neon_dot::Pq2Dotprod::detect()
    };
    for (index, (((b0, b1), b2), b3)) in r0.iter().zip(r1).zip(r2).zip(r3).enumerate() {
        let scales_vec = crate::kernels::neon::f16_bits_to_f32_x4(
            neon,
            [
                u16::from_le_bytes([b0[0], b0[1]]),
                u16::from_le_bytes([b1[0], b1[1]]),
                u16::from_le_bytes([b2[0], b2[1]]),
                u16::from_le_bytes([b3[0], b3[1]]),
            ],
        );
        if !scales_vec.is_finite().all_true() {
            return Some(Err(crate::invalid("nonfinite tensor or activation value")));
        }
        let row_scales = scales_vec.to_array();
        *decoded_blocks = decoded_blocks.saturating_add(4);
        let packed_rows: [&[u8; 32]; 4] = [
            b0[2..34].try_into().ok()?,
            b1[2..34].try_into().ok()?,
            b2[2..34].try_into().ok()?,
            b3[2..34].try_into().ok()?,
        ];
        let half = index % 2;
        let start = half * 128;
        let q8k_idx = index / 2;
        let (chunks4, rem4) = preparations.as_chunks::<4>();
        if let Some(dotprod) = dotprod {
            for (chunk_idx, chunk) in chunks4.iter().enumerate() {
                let base = chunk_idx * 4;
                let (dots, act_scales) = if has_precomputed {
                    let a0 = &chunk[0].pq2_consumption[index];
                    let a1 = &chunk[1].pq2_consumption[index];
                    let a2 = &chunk[2].pq2_consumption[index];
                    let a3 = &chunk[3].pq2_consumption[index];
                    (
                        dotprod.dot_four_tokens(
                            packed_rows,
                            [&a0.values, &a1.values, &a2.values, &a3.values],
                            [a0.sum, a1.sum, a2.sum, a3.sum],
                        ),
                        f32x4::from_slice(neon, &[a0.scale, a1.scale, a2.scale, a3.scale]),
                    )
                } else {
                    let a0 = &chunk[0].q8k[q8k_idx];
                    let a1 = &chunk[1].q8k[q8k_idx];
                    let a2 = &chunk[2].q8k[q8k_idx];
                    let a3 = &chunk[3].q8k[q8k_idx];
                    (
                        bonsai_neon_dot::dot_pq2_four_rows_four_tokens(
                            packed_rows,
                            [
                                a0.values[start..start + 128].try_into().ok()?,
                                a1.values[start..start + 128].try_into().ok()?,
                                a2.values[start..start + 128].try_into().ok()?,
                                a3.values[start..start + 128].try_into().ok()?,
                            ],
                            [
                                q8k_half_sum(neon, &a0.sums, half),
                                q8k_half_sum(neon, &a1.sums, half),
                                q8k_half_sum(neon, &a2.sums, half),
                                q8k_half_sum(neon, &a3.sums, half),
                            ],
                        ),
                        f32x4::from_slice(neon, &[a0.scale, a1.scale, a2.scale, a3.scale]),
                    )
                };
                for r in 0..4 {
                    let d = f32x4::splat(neon, row_scales[r]) * act_scales;
                    let dot_f32 = i32x4::from_slice(neon, &dots[r]).to_float::<f32x4<Neon>>();
                    let cur = f32x4::from_slice(neon, &sums[r][base..base + 4]);
                    let next = if pinned {
                        d.mul_add_precise(dot_f32, cur)
                    } else {
                        cur + d * dot_f32
                    };
                    next.store_slice(&mut sums[r][base..base + 4]);
                }
            }
            let base2 = chunks4.len() * 4;
            let (chunks2, rem2) = rem4.as_chunks::<2>();
            for (chunk_idx, chunk) in chunks2.iter().enumerate() {
                let token0 = base2 + chunk_idx * 2;
                let (dots, act_scales) = if has_precomputed {
                    let a0 = &chunk[0].pq2_consumption[index];
                    let a1 = &chunk[1].pq2_consumption[index];
                    (
                        dotprod.dot_four_rows_two_tokens(
                            packed_rows,
                            [&a0.values, &a1.values],
                            [a0.sum, a1.sum],
                        ),
                        [a0.scale, a1.scale],
                    )
                } else {
                    let a0 = &chunk[0].q8k[q8k_idx];
                    let a1 = &chunk[1].q8k[q8k_idx];
                    (
                        bonsai_neon_dot::dot_pq2_four_rows_two_tokens(
                            packed_rows,
                            [
                                a0.values[start..start + 128].try_into().ok()?,
                                a1.values[start..start + 128].try_into().ok()?,
                            ],
                            [
                                q8k_half_sum(neon, &a0.sums, half),
                                q8k_half_sum(neon, &a1.sums, half),
                            ],
                        ),
                        [a0.scale, a1.scale],
                    )
                };
                for r in 0..4 {
                    for (t, &scale) in act_scales.iter().enumerate() {
                        let d = row_scales[r] * scale;
                        let dot_f32 = crate::kernels::integer_float(dots[r][t]);
                        let token = token0 + t;
                        sums[r][token] = if pinned {
                            d.mul_add(dot_f32, sums[r][token])
                        } else {
                            sums[r][token] + d * dot_f32
                        };
                    }
                }
            }
            let rem_base = base2 + chunks2.len() * 2;
            for (offset, scratch) in rem2.iter().enumerate() {
                let token = rem_base + offset;
                let (dots, act_scale) = if has_precomputed {
                    let act_block = &scratch.pq2_consumption[index];
                    (
                        dotprod.dot_four(packed_rows, &act_block.values, act_block.sum),
                        act_block.scale,
                    )
                } else {
                    let act_block = &scratch.q8k[q8k_idx];
                    (
                        bonsai_neon_dot::dot_pq2_four_rows_one_token(
                            packed_rows,
                            act_block.values[start..start + 128].try_into().ok()?,
                            q8k_half_sum(neon, &act_block.sums, half),
                        ),
                        act_block.scale,
                    )
                };
                for r in 0..4 {
                    let d = row_scales[r] * act_scale;
                    let dot_f32 = crate::kernels::integer_float(dots[r]);
                    sums[r][token] = if pinned {
                        d.mul_add(dot_f32, sums[r][token])
                    } else {
                        sums[r][token] + d * dot_f32
                    };
                }
            }
        } else {
            let weights = packed_rows.map(|row| decode_pq2_unsigned(neon, row));
            for (chunk_idx, chunk) in chunks4.iter().enumerate() {
                let base = chunk_idx * 4;
                let (acts, act_scales) = if has_precomputed {
                    let a0 = &chunk[0].pq2_consumption[index];
                    let a1 = &chunk[1].pq2_consumption[index];
                    let a2 = &chunk[2].pq2_consumption[index];
                    let a3 = &chunk[3].pq2_consumption[index];
                    (
                        [
                            load_pq2_activation_precomputed(
                                neon,
                                &chunk[0].pq2_consumption,
                                index,
                            )?,
                            load_pq2_activation_precomputed(
                                neon,
                                &chunk[1].pq2_consumption,
                                index,
                            )?,
                            load_pq2_activation_precomputed(
                                neon,
                                &chunk[2].pq2_consumption,
                                index,
                            )?,
                            load_pq2_activation_precomputed(
                                neon,
                                &chunk[3].pq2_consumption,
                                index,
                            )?,
                        ],
                        f32x4::from_slice(neon, &[a0.scale, a1.scale, a2.scale, a3.scale]),
                    )
                } else {
                    let a0 = &chunk[0].q8k[q8k_idx];
                    let a1 = &chunk[1].q8k[q8k_idx];
                    let a2 = &chunk[2].q8k[q8k_idx];
                    let a3 = &chunk[3].q8k[q8k_idx];
                    (
                        [
                            prepare_pq2_activation(
                                neon,
                                a0.values[start..start + 128].try_into().ok()?,
                                q8k_half_sum(neon, &a0.sums, half),
                            ),
                            prepare_pq2_activation(
                                neon,
                                a1.values[start..start + 128].try_into().ok()?,
                                q8k_half_sum(neon, &a1.sums, half),
                            ),
                            prepare_pq2_activation(
                                neon,
                                a2.values[start..start + 128].try_into().ok()?,
                                q8k_half_sum(neon, &a2.sums, half),
                            ),
                            prepare_pq2_activation(
                                neon,
                                a3.values[start..start + 128].try_into().ok()?,
                                q8k_half_sum(neon, &a3.sums, half),
                            ),
                        ],
                        f32x4::from_slice(neon, &[a0.scale, a1.scale, a2.scale, a3.scale]),
                    )
                };
                let dots = dot128_four_rows_four_tokens(neon, &weights, &acts);
                for r in 0..4 {
                    let d = f32x4::splat(neon, row_scales[r]) * act_scales;
                    let dot_f32 = dots[r].to_float::<f32x4<Neon>>();
                    let cur = f32x4::from_slice(neon, &sums[r][base..base + 4]);
                    let next = if pinned {
                        d.mul_add_precise(dot_f32, cur)
                    } else {
                        cur + d * dot_f32
                    };
                    next.store_slice(&mut sums[r][base..base + 4]);
                }
            }
            let rem_base = chunks4.len() * 4;
            for (offset, scratch) in rem4.iter().enumerate() {
                let token = rem_base + offset;
                let (act, act_scale) = if has_precomputed {
                    let act_block = &scratch.pq2_consumption[index];
                    (
                        load_pq2_activation_precomputed(neon, &scratch.pq2_consumption, index)?,
                        act_block.scale,
                    )
                } else {
                    let act_block = &scratch.q8k[q8k_idx];
                    (
                        prepare_pq2_activation(
                            neon,
                            act_block.values[start..start + 128].try_into().ok()?,
                            q8k_half_sum(neon, &act_block.sums, half),
                        ),
                        act_block.scale,
                    )
                };
                let dots = dot128_four_rows_one_token(neon, &weights, &act);
                for r in 0..4 {
                    let d = row_scales[r] * act_scale;
                    let dot_f32 = crate::kernels::integer_float(dots[r]);
                    sums[r][token] = if pinned {
                        d.mul_add(dot_f32, sums[r][token])
                    } else {
                        sums[r][token] + d * dot_f32
                    };
                }
            }
        }
    }
    for row_sums in &mut sums {
        for sum in row_sums.iter_mut().take(preparations.len()) {
            if let Err(err) = crate::kernels::finite(*sum) {
                return Some(Err(err));
            }
        }
    }
    Some(Ok(sums))
}

#[expect(
    clippy::suboptimal_flops,
    clippy::too_many_lines,
    reason = "Preserve separately rounded source arithmetic and eight-row × eight-token register blocking."
)]
pub(in crate::kernels::prefill) fn dot_pq2_eight_rows<const TOKENS: usize>(
    rows: [&[u8]; 8],
    preparations: &[crate::kernels::MatvecScratch],
    pinned: bool,
    decoded_blocks: &mut u64,
) -> Option<crate::Result<[[f32; TOKENS]; 8]>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let mut sums = [[0.0_f32; TOKENS]; 8];
    let chunks = rows.map(|row| row.as_chunks::<34>().0);
    let has_precomputed = preparations
        .iter()
        .all(|scratch| scratch.pq2_consumption.len() >= chunks[0].len());
    let dotprod = if has_precomputed {
        preparations.first()?.pq2_consumption.dotprod
    } else {
        bonsai_neon_dot::Pq2Dotprod::detect()
    };
    let (chunks8, rem8) = preparations.as_chunks::<8>();
    let block_count = chunks[0].len();
    if has_precomputed
        && !preparations.is_empty()
        && preparations.len() <= TOKENS
        && block_count != 0
        && rows.iter().all(|row| row.len() == block_count * 34)
        && let Some(dotprod) = dotprod
    {
        if !chunks8.is_empty() {
            let mut out_buf = [[[0.0_f32; 8]; 8]; 8];
            for (batch_idx, batch) in chunks8.chunks(8).enumerate() {
                let mut arg_chunks: [[&[bonsai_neon_dot::Pq2Activation]; 8]; 8] = [[&[]; 8]; 8];
                for (c, chunk) in batch.iter().enumerate() {
                    arg_chunks[c] =
                        std::array::from_fn(|t| &chunk[t].pq2_consumption[..block_count]);
                }
                let Some(prepared) = dotprod.prepare_scaled_chunks(&arg_chunks[..batch.len()])
                else {
                    return Some(Err(crate::invalid("nonfinite tensor or activation value")));
                };
                let out_slice = &mut out_buf[..batch.len()];
                let result = if pinned {
                    prepared.dot_scaled_8x8_chunks::<true>(rows, out_slice)
                } else {
                    prepared.dot_scaled_8x8_chunks::<false>(rows, out_slice)
                };
                let Some(()) = result else {
                    return Some(Err(crate::invalid("nonfinite tensor or activation value")));
                };
                for (c, chunk_out) in out_slice.iter().enumerate() {
                    let base = (batch_idx * 8 + c) * 8;
                    for row in 0..8 {
                        sums[row][base..base + 8].copy_from_slice(&chunk_out[row]);
                    }
                }
            }
        }
        if !rem8.is_empty() {
            let mut rem_acts: [&[bonsai_neon_dot::Pq2Activation]; 8] = [&[]; 8];
            for (t, scratch) in rem8.iter().enumerate() {
                rem_acts[t] = &scratch.pq2_consumption[..block_count];
            }
            let Some(prepared_rem) = dotprod.prepare_scaled_tokens(&rem_acts[..rem8.len()]) else {
                return Some(Err(crate::invalid("nonfinite tensor or activation value")));
            };
            let mut rem_out = [[0.0_f32; 8]; 8];
            let result = if pinned {
                prepared_rem.dot_scaled_8x_tokens::<true>(rows, &mut rem_out)
            } else {
                prepared_rem.dot_scaled_8x_tokens::<false>(rows, &mut rem_out)
            };
            let Some(()) = result else {
                return Some(Err(crate::invalid("nonfinite tensor or activation value")));
            };
            let base = chunks8.len() * 8;
            for row in 0..8 {
                sums[row][base..base + rem8.len()].copy_from_slice(&rem_out[row][..rem8.len()]);
            }
        }
        let block_count_u64 = u64::try_from(block_count).ok()?;
        *decoded_blocks = decoded_blocks.saturating_add(block_count_u64.saturating_mul(8));
        return Some(Ok(sums));
    }
    for index in 0..chunks[0].len() {
        let b0 = &chunks[0][index];
        let b1 = &chunks[1][index];
        let b2 = &chunks[2][index];
        let b3 = &chunks[3][index];
        let b4 = &chunks[4][index];
        let b5 = &chunks[5][index];
        let b6 = &chunks[6][index];
        let b7 = &chunks[7][index];
        let (scales_lo, scales_hi) = crate::kernels::neon::f16_bits_to_f32_x8(
            neon,
            &[
                u16::from_le_bytes([b0[0], b0[1]]),
                u16::from_le_bytes([b1[0], b1[1]]),
                u16::from_le_bytes([b2[0], b2[1]]),
                u16::from_le_bytes([b3[0], b3[1]]),
                u16::from_le_bytes([b4[0], b4[1]]),
                u16::from_le_bytes([b5[0], b5[1]]),
                u16::from_le_bytes([b6[0], b6[1]]),
                u16::from_le_bytes([b7[0], b7[1]]),
            ],
        );
        if !(scales_lo.is_finite() & scales_hi.is_finite()).all_true() {
            return Some(Err(crate::invalid("nonfinite tensor or activation value")));
        }
        let scales_lo_arr = scales_lo.to_array();
        let scales_hi_arr = scales_hi.to_array();
        let row_scales = [
            scales_lo_arr[0],
            scales_lo_arr[1],
            scales_lo_arr[2],
            scales_lo_arr[3],
            scales_hi_arr[0],
            scales_hi_arr[1],
            scales_hi_arr[2],
            scales_hi_arr[3],
        ];
        *decoded_blocks = decoded_blocks.saturating_add(8);
        let packed_lo: [&[u8; 32]; 4] = [
            b0[2..34].try_into().ok()?,
            b1[2..34].try_into().ok()?,
            b2[2..34].try_into().ok()?,
            b3[2..34].try_into().ok()?,
        ];
        let packed_hi: [&[u8; 32]; 4] = [
            b4[2..34].try_into().ok()?,
            b5[2..34].try_into().ok()?,
            b6[2..34].try_into().ok()?,
            b7[2..34].try_into().ok()?,
        ];
        let half = index % 2;
        let start = half * 128;
        let q8k_idx = index / 2;
        let (chunks8, rem8) = preparations.as_chunks::<8>();
        if let Some(dotprod) = dotprod {
            let packed_all: [&[u8; 32]; 8] = [
                packed_lo[0],
                packed_lo[1],
                packed_lo[2],
                packed_lo[3],
                packed_hi[0],
                packed_hi[1],
                packed_hi[2],
                packed_hi[3],
            ];
            for (chunk_idx, chunk) in chunks8.iter().enumerate() {
                let base = chunk_idx * 8;
                let (dots8, act_scales_lo, act_scales_hi) = if has_precomputed {
                    let a0 = &chunk[0].pq2_consumption[index];
                    let a1 = &chunk[1].pq2_consumption[index];
                    let a2 = &chunk[2].pq2_consumption[index];
                    let a3 = &chunk[3].pq2_consumption[index];
                    let a4 = &chunk[4].pq2_consumption[index];
                    let a5 = &chunk[5].pq2_consumption[index];
                    let a6 = &chunk[6].pq2_consumption[index];
                    let a7 = &chunk[7].pq2_consumption[index];
                    (
                        dotprod.dot_eight_tokens(
                            packed_all,
                            [
                                &a0.values, &a1.values, &a2.values, &a3.values, &a4.values,
                                &a5.values, &a6.values, &a7.values,
                            ],
                            [
                                a0.sum, a1.sum, a2.sum, a3.sum, a4.sum, a5.sum, a6.sum, a7.sum,
                            ],
                        ),
                        f32x4::from_slice(neon, &[a0.scale, a1.scale, a2.scale, a3.scale]),
                        f32x4::from_slice(neon, &[a4.scale, a5.scale, a6.scale, a7.scale]),
                    )
                } else {
                    let a0 = &chunk[0].q8k[q8k_idx];
                    let a1 = &chunk[1].q8k[q8k_idx];
                    let a2 = &chunk[2].q8k[q8k_idx];
                    let a3 = &chunk[3].q8k[q8k_idx];
                    let a4 = &chunk[4].q8k[q8k_idx];
                    let a5 = &chunk[5].q8k[q8k_idx];
                    let a6 = &chunk[6].q8k[q8k_idx];
                    let a7 = &chunk[7].q8k[q8k_idx];
                    (
                        bonsai_neon_dot::dot_pq2_eight_rows_eight_tokens(
                            packed_all,
                            [
                                a0.values[start..start + 128].try_into().ok()?,
                                a1.values[start..start + 128].try_into().ok()?,
                                a2.values[start..start + 128].try_into().ok()?,
                                a3.values[start..start + 128].try_into().ok()?,
                                a4.values[start..start + 128].try_into().ok()?,
                                a5.values[start..start + 128].try_into().ok()?,
                                a6.values[start..start + 128].try_into().ok()?,
                                a7.values[start..start + 128].try_into().ok()?,
                            ],
                            [
                                q8k_half_sum(neon, &a0.sums, half),
                                q8k_half_sum(neon, &a1.sums, half),
                                q8k_half_sum(neon, &a2.sums, half),
                                q8k_half_sum(neon, &a3.sums, half),
                                q8k_half_sum(neon, &a4.sums, half),
                                q8k_half_sum(neon, &a5.sums, half),
                                q8k_half_sum(neon, &a6.sums, half),
                                q8k_half_sum(neon, &a7.sums, half),
                            ],
                        ),
                        f32x4::from_slice(neon, &[a0.scale, a1.scale, a2.scale, a3.scale]),
                        f32x4::from_slice(neon, &[a4.scale, a5.scale, a6.scale, a7.scale]),
                    )
                };
                for row in 0..8 {
                    let r_scale = f32x4::splat(neon, row_scales[row]);
                    let d_lo = r_scale * act_scales_lo;
                    let d_hi = r_scale * act_scales_hi;
                    let dot_lo_f32 =
                        i32x4::from_slice(neon, &dots8[row][..4]).to_float::<f32x4<Neon>>();
                    let dot_hi_f32 =
                        i32x4::from_slice(neon, &dots8[row][4..8]).to_float::<f32x4<Neon>>();
                    let cur_lo = f32x4::from_slice(neon, &sums[row][base..base + 4]);
                    let cur_hi = f32x4::from_slice(neon, &sums[row][base + 4..base + 8]);
                    let (next_lo, next_hi) = if pinned {
                        (
                            d_lo.mul_add_precise(dot_lo_f32, cur_lo),
                            d_hi.mul_add_precise(dot_hi_f32, cur_hi),
                        )
                    } else {
                        (cur_lo + d_lo * dot_lo_f32, cur_hi + d_hi * dot_hi_f32)
                    };
                    next_lo.store_slice(&mut sums[row][base..base + 4]);
                    next_hi.store_slice(&mut sums[row][base + 4..base + 8]);
                }
            }
            let base4 = chunks8.len() * 8;
            let (chunks4, rem4) = rem8.as_chunks::<4>();
            for (chunk_idx, chunk) in chunks4.iter().enumerate() {
                let base = base4 + chunk_idx * 4;
                if has_precomputed {
                    let a0 = &chunk[0].pq2_consumption[index];
                    let a1 = &chunk[1].pq2_consumption[index];
                    let a2 = &chunk[2].pq2_consumption[index];
                    let a3 = &chunk[3].pq2_consumption[index];
                    let act_scales =
                        f32x4::from_slice(neon, &[a0.scale, a1.scale, a2.scale, a3.scale]);
                    let dots8 = dotprod.dot_eight_rows_four_tokens(
                        packed_all,
                        [&a0.values, &a1.values, &a2.values, &a3.values],
                        [a0.sum, a1.sum, a2.sum, a3.sum],
                    );
                    for (row, dot) in dots8.iter().enumerate() {
                        let d = f32x4::splat(neon, row_scales[row]) * act_scales;
                        let dot_f32 = i32x4::from_slice(neon, dot).to_float::<f32x4<Neon>>();
                        let cur = f32x4::from_slice(neon, &sums[row][base..base + 4]);
                        let next = if pinned {
                            d.mul_add_precise(dot_f32, cur)
                        } else {
                            cur + d * dot_f32
                        };
                        next.store_slice(&mut sums[row][base..base + 4]);
                    }
                } else {
                    let a0 = &chunk[0].q8k[q8k_idx];
                    let a1 = &chunk[1].q8k[q8k_idx];
                    let a2 = &chunk[2].q8k[q8k_idx];
                    let a3 = &chunk[3].q8k[q8k_idx];
                    let act_slices: [&[i8; 128]; 4] = [
                        a0.values[start..start + 128].try_into().ok()?,
                        a1.values[start..start + 128].try_into().ok()?,
                        a2.values[start..start + 128].try_into().ok()?,
                        a3.values[start..start + 128].try_into().ok()?,
                    ];
                    let act_sums = [
                        q8k_half_sum(neon, &a0.sums, half),
                        q8k_half_sum(neon, &a1.sums, half),
                        q8k_half_sum(neon, &a2.sums, half),
                        q8k_half_sum(neon, &a3.sums, half),
                    ];
                    let act_scales =
                        f32x4::from_slice(neon, &[a0.scale, a1.scale, a2.scale, a3.scale]);
                    for (row_group, weights_half) in [(packed_lo, 0_usize), (packed_hi, 4_usize)] {
                        let dots = bonsai_neon_dot::dot_pq2_four_rows_four_tokens(
                            row_group, act_slices, act_sums,
                        );
                        for (r, dot) in dots.iter().enumerate() {
                            let row = weights_half + r;
                            let d = f32x4::splat(neon, row_scales[row]) * act_scales;
                            let dot_f32 = i32x4::from_slice(neon, dot).to_float::<f32x4<Neon>>();
                            let cur = f32x4::from_slice(neon, &sums[row][base..base + 4]);
                            let next = if pinned {
                                d.mul_add_precise(dot_f32, cur)
                            } else {
                                cur + d * dot_f32
                            };
                            next.store_slice(&mut sums[row][base..base + 4]);
                        }
                    }
                }
            }
            let base2 = base4 + chunks4.len() * 4;
            let (chunks2, rem2) = rem4.as_chunks::<2>();
            for (chunk_idx, chunk) in chunks2.iter().enumerate() {
                let token0 = base2 + chunk_idx * 2;
                let (dots8, act_scales) = if has_precomputed {
                    let a0 = &chunk[0].pq2_consumption[index];
                    let a1 = &chunk[1].pq2_consumption[index];
                    (
                        dotprod.dot_eight_rows_two_tokens(
                            packed_all,
                            [&a0.values, &a1.values],
                            [a0.sum, a1.sum],
                        ),
                        [a0.scale, a1.scale],
                    )
                } else {
                    let a0 = &chunk[0].q8k[q8k_idx];
                    let a1 = &chunk[1].q8k[q8k_idx];
                    (
                        bonsai_neon_dot::dot_pq2_eight_rows_two_tokens(
                            packed_all,
                            [
                                a0.values[start..start + 128].try_into().ok()?,
                                a1.values[start..start + 128].try_into().ok()?,
                            ],
                            [
                                q8k_half_sum(neon, &a0.sums, half),
                                q8k_half_sum(neon, &a1.sums, half),
                            ],
                        ),
                        [a0.scale, a1.scale],
                    )
                };
                for row in 0..8 {
                    for (t, &scale) in act_scales.iter().enumerate() {
                        let d = row_scales[row] * scale;
                        let dot_f32 = crate::kernels::integer_float(dots8[row][t]);
                        let token = token0 + t;
                        sums[row][token] = if pinned {
                            d.mul_add(dot_f32, sums[row][token])
                        } else {
                            sums[row][token] + d * dot_f32
                        };
                    }
                }
            }
            let rem_base = base2 + chunks2.len() * 2;
            for (offset, scratch) in rem2.iter().enumerate() {
                let token = rem_base + offset;
                if has_precomputed {
                    let act_block = &scratch.pq2_consumption[index];
                    let dots8 = dotprod.dot_eight(packed_all, &act_block.values, act_block.sum);
                    for row in 0..8 {
                        let d = row_scales[row] * act_block.scale;
                        let dot_f32 = crate::kernels::integer_float(dots8[row]);
                        sums[row][token] = if pinned {
                            d.mul_add(dot_f32, sums[row][token])
                        } else {
                            sums[row][token] + d * dot_f32
                        };
                    }
                } else {
                    let act_block = &scratch.q8k[q8k_idx];
                    let act_slice: &[i8; 128] =
                        act_block.values[start..start + 128].try_into().ok()?;
                    let act_sum = q8k_half_sum(neon, &act_block.sums, half);
                    let dots_lo =
                        bonsai_neon_dot::dot_pq2_four_rows_one_token(packed_lo, act_slice, act_sum);
                    let dots_hi =
                        bonsai_neon_dot::dot_pq2_four_rows_one_token(packed_hi, act_slice, act_sum);
                    for r in 0..4 {
                        let d_lo = row_scales[r] * act_block.scale;
                        let d_hi = row_scales[r + 4] * act_block.scale;
                        let dot_lo_f32 = crate::kernels::integer_float(dots_lo[r]);
                        let dot_hi_f32 = crate::kernels::integer_float(dots_hi[r]);
                        sums[r][token] = if pinned {
                            d_lo.mul_add(dot_lo_f32, sums[r][token])
                        } else {
                            sums[r][token] + d_lo * dot_lo_f32
                        };
                        sums[r + 4][token] = if pinned {
                            d_hi.mul_add(dot_hi_f32, sums[r + 4][token])
                        } else {
                            sums[r + 4][token] + d_hi * dot_hi_f32
                        };
                    }
                }
            }
        } else {
            let weights_lo = packed_lo.map(|row| decode_pq2_unsigned(neon, row));
            let weights_hi = packed_hi.map(|row| decode_pq2_unsigned(neon, row));
            for (chunk_idx, chunk) in chunks8.iter().enumerate() {
                let base = chunk_idx * 8;
                let (acts_lo, acts_hi, act_scales_lo, act_scales_hi) = if has_precomputed {
                    let a0 = &chunk[0].pq2_consumption[index];
                    let a1 = &chunk[1].pq2_consumption[index];
                    let a2 = &chunk[2].pq2_consumption[index];
                    let a3 = &chunk[3].pq2_consumption[index];
                    let a4 = &chunk[4].pq2_consumption[index];
                    let a5 = &chunk[5].pq2_consumption[index];
                    let a6 = &chunk[6].pq2_consumption[index];
                    let a7 = &chunk[7].pq2_consumption[index];
                    (
                        [
                            load_pq2_activation_precomputed(
                                neon,
                                &chunk[0].pq2_consumption,
                                index,
                            )?,
                            load_pq2_activation_precomputed(
                                neon,
                                &chunk[1].pq2_consumption,
                                index,
                            )?,
                            load_pq2_activation_precomputed(
                                neon,
                                &chunk[2].pq2_consumption,
                                index,
                            )?,
                            load_pq2_activation_precomputed(
                                neon,
                                &chunk[3].pq2_consumption,
                                index,
                            )?,
                        ],
                        [
                            load_pq2_activation_precomputed(
                                neon,
                                &chunk[4].pq2_consumption,
                                index,
                            )?,
                            load_pq2_activation_precomputed(
                                neon,
                                &chunk[5].pq2_consumption,
                                index,
                            )?,
                            load_pq2_activation_precomputed(
                                neon,
                                &chunk[6].pq2_consumption,
                                index,
                            )?,
                            load_pq2_activation_precomputed(
                                neon,
                                &chunk[7].pq2_consumption,
                                index,
                            )?,
                        ],
                        f32x4::from_slice(neon, &[a0.scale, a1.scale, a2.scale, a3.scale]),
                        f32x4::from_slice(neon, &[a4.scale, a5.scale, a6.scale, a7.scale]),
                    )
                } else {
                    let a0 = &chunk[0].q8k[q8k_idx];
                    let a1 = &chunk[1].q8k[q8k_idx];
                    let a2 = &chunk[2].q8k[q8k_idx];
                    let a3 = &chunk[3].q8k[q8k_idx];
                    let a4 = &chunk[4].q8k[q8k_idx];
                    let a5 = &chunk[5].q8k[q8k_idx];
                    let a6 = &chunk[6].q8k[q8k_idx];
                    let a7 = &chunk[7].q8k[q8k_idx];
                    (
                        [
                            prepare_pq2_activation(
                                neon,
                                a0.values[start..start + 128].try_into().ok()?,
                                q8k_half_sum(neon, &a0.sums, half),
                            ),
                            prepare_pq2_activation(
                                neon,
                                a1.values[start..start + 128].try_into().ok()?,
                                q8k_half_sum(neon, &a1.sums, half),
                            ),
                            prepare_pq2_activation(
                                neon,
                                a2.values[start..start + 128].try_into().ok()?,
                                q8k_half_sum(neon, &a2.sums, half),
                            ),
                            prepare_pq2_activation(
                                neon,
                                a3.values[start..start + 128].try_into().ok()?,
                                q8k_half_sum(neon, &a3.sums, half),
                            ),
                        ],
                        [
                            prepare_pq2_activation(
                                neon,
                                a4.values[start..start + 128].try_into().ok()?,
                                q8k_half_sum(neon, &a4.sums, half),
                            ),
                            prepare_pq2_activation(
                                neon,
                                a5.values[start..start + 128].try_into().ok()?,
                                q8k_half_sum(neon, &a5.sums, half),
                            ),
                            prepare_pq2_activation(
                                neon,
                                a6.values[start..start + 128].try_into().ok()?,
                                q8k_half_sum(neon, &a6.sums, half),
                            ),
                            prepare_pq2_activation(
                                neon,
                                a7.values[start..start + 128].try_into().ok()?,
                                q8k_half_sum(neon, &a7.sums, half),
                            ),
                        ],
                        f32x4::from_slice(neon, &[a0.scale, a1.scale, a2.scale, a3.scale]),
                        f32x4::from_slice(neon, &[a4.scale, a5.scale, a6.scale, a7.scale]),
                    )
                };
                for (row_group, weights_half) in [(&weights_lo, 0_usize), (&weights_hi, 4_usize)] {
                    let dots_lo = dot128_four_rows_four_tokens(neon, row_group, &acts_lo);
                    let dots_hi = dot128_four_rows_four_tokens(neon, row_group, &acts_hi);
                    for r in 0..4 {
                        let row = weights_half + r;
                        let r_scale = f32x4::splat(neon, row_scales[row]);
                        let d_lo = r_scale * act_scales_lo;
                        let d_hi = r_scale * act_scales_hi;
                        let dot_lo_f32 = dots_lo[r].to_float::<f32x4<Neon>>();
                        let dot_hi_f32 = dots_hi[r].to_float::<f32x4<Neon>>();
                        let cur_lo = f32x4::from_slice(neon, &sums[row][base..base + 4]);
                        let cur_hi = f32x4::from_slice(neon, &sums[row][base + 4..base + 8]);
                        let (next_lo, next_hi) = if pinned {
                            (
                                d_lo.mul_add_precise(dot_lo_f32, cur_lo),
                                d_hi.mul_add_precise(dot_hi_f32, cur_hi),
                            )
                        } else {
                            (cur_lo + d_lo * dot_lo_f32, cur_hi + d_hi * dot_hi_f32)
                        };
                        next_lo.store_slice(&mut sums[row][base..base + 4]);
                        next_hi.store_slice(&mut sums[row][base + 4..base + 8]);
                    }
                }
            }
            let base4 = chunks8.len() * 8;
            let (chunks4, rem4) = rem8.as_chunks::<4>();
            for (chunk_idx, chunk) in chunks4.iter().enumerate() {
                let base = base4 + chunk_idx * 4;
                let (acts, act_scales) = if has_precomputed {
                    let a0 = &chunk[0].pq2_consumption[index];
                    let a1 = &chunk[1].pq2_consumption[index];
                    let a2 = &chunk[2].pq2_consumption[index];
                    let a3 = &chunk[3].pq2_consumption[index];
                    (
                        [
                            load_pq2_activation_precomputed(
                                neon,
                                &chunk[0].pq2_consumption,
                                index,
                            )?,
                            load_pq2_activation_precomputed(
                                neon,
                                &chunk[1].pq2_consumption,
                                index,
                            )?,
                            load_pq2_activation_precomputed(
                                neon,
                                &chunk[2].pq2_consumption,
                                index,
                            )?,
                            load_pq2_activation_precomputed(
                                neon,
                                &chunk[3].pq2_consumption,
                                index,
                            )?,
                        ],
                        f32x4::from_slice(neon, &[a0.scale, a1.scale, a2.scale, a3.scale]),
                    )
                } else {
                    let a0 = &chunk[0].q8k[q8k_idx];
                    let a1 = &chunk[1].q8k[q8k_idx];
                    let a2 = &chunk[2].q8k[q8k_idx];
                    let a3 = &chunk[3].q8k[q8k_idx];
                    (
                        [
                            prepare_pq2_activation(
                                neon,
                                a0.values[start..start + 128].try_into().ok()?,
                                q8k_half_sum(neon, &a0.sums, half),
                            ),
                            prepare_pq2_activation(
                                neon,
                                a1.values[start..start + 128].try_into().ok()?,
                                q8k_half_sum(neon, &a1.sums, half),
                            ),
                            prepare_pq2_activation(
                                neon,
                                a2.values[start..start + 128].try_into().ok()?,
                                q8k_half_sum(neon, &a2.sums, half),
                            ),
                            prepare_pq2_activation(
                                neon,
                                a3.values[start..start + 128].try_into().ok()?,
                                q8k_half_sum(neon, &a3.sums, half),
                            ),
                        ],
                        f32x4::from_slice(neon, &[a0.scale, a1.scale, a2.scale, a3.scale]),
                    )
                };
                for (row_group, weights_half) in [(&weights_lo, 0_usize), (&weights_hi, 4_usize)] {
                    let dots = dot128_four_rows_four_tokens(neon, row_group, &acts);
                    for (r, &dot) in dots.iter().enumerate() {
                        let row = weights_half + r;
                        let d = f32x4::splat(neon, row_scales[row]) * act_scales;
                        let dot_f32 = dot.to_float::<f32x4<Neon>>();
                        let cur = f32x4::from_slice(neon, &sums[row][base..base + 4]);
                        let next = if pinned {
                            d.mul_add_precise(dot_f32, cur)
                        } else {
                            cur + d * dot_f32
                        };
                        next.store_slice(&mut sums[row][base..base + 4]);
                    }
                }
            }
            let rem_base = base4 + chunks4.len() * 4;
            for (offset, scratch) in rem4.iter().enumerate() {
                let token = rem_base + offset;
                let (act, act_scale) = if has_precomputed {
                    let act_block = &scratch.pq2_consumption[index];
                    (
                        load_pq2_activation_precomputed(neon, &scratch.pq2_consumption, index)?,
                        act_block.scale,
                    )
                } else {
                    let act_block = &scratch.q8k[q8k_idx];
                    (
                        prepare_pq2_activation(
                            neon,
                            act_block.values[start..start + 128].try_into().ok()?,
                            q8k_half_sum(neon, &act_block.sums, half),
                        ),
                        act_block.scale,
                    )
                };
                let dots_lo = dot128_four_rows_one_token(neon, &weights_lo, &act);
                let dots_hi = dot128_four_rows_one_token(neon, &weights_hi, &act);
                for r in 0..4 {
                    let d_lo = row_scales[r] * act_scale;
                    let d_hi = row_scales[r + 4] * act_scale;
                    let dot_lo_f32 = crate::kernels::integer_float(dots_lo[r]);
                    let dot_hi_f32 = crate::kernels::integer_float(dots_hi[r]);
                    sums[r][token] = if pinned {
                        d_lo.mul_add(dot_lo_f32, sums[r][token])
                    } else {
                        sums[r][token] + d_lo * dot_lo_f32
                    };
                    sums[r + 4][token] = if pinned {
                        d_hi.mul_add(dot_hi_f32, sums[r + 4][token])
                    } else {
                        sums[r + 4][token] + d_hi * dot_hi_f32
                    };
                }
            }
        }
    }
    for row_sums in &mut sums {
        for sum in row_sums.iter_mut().take(preparations.len()) {
            if let Err(err) = crate::kernels::finite(*sum) {
                return Some(Err(err));
            }
        }
    }
    Some(Ok(sums))
}

#[expect(
    clippy::suboptimal_flops,
    clippy::too_many_lines,
    reason = "Preserve separately rounded source arithmetic and four-row Q8 activation sharing."
)]
pub(in crate::kernels::prefill) fn dot_q8_four_rows<const TOKENS: usize>(
    rows: [&[u8]; 4],
    preparations: &[crate::kernels::MatvecScratch],
    is_ptq1: bool,
    pinned: bool,
    decoded_blocks: &mut u64,
) -> Option<crate::Result<[[f32; TOKENS]; 4]>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let use_ptq1_dotprod = is_ptq1 && bonsai_neon_dot::has_dotprod();
    let mut sums = [[0.0_f32; TOKENS]; 4];
    let mut lanes = [[f32x4::splat(neon, 0.0); TOKENS]; 4];
    let block_bytes = if is_ptq1 { 28 } else { 34 };
    let r0 = rows[0].chunks_exact(block_bytes);
    let r1 = rows[1].chunks_exact(block_bytes);
    let r2 = rows[2].chunks_exact(block_bytes);
    let r3 = rows[3].chunks_exact(block_bytes);
    for (index, (((b0, b1), b2), b3)) in r0.zip(r1).zip(r2).zip(r3).enumerate() {
        let scale_offset = if is_ptq1 { 26 } else { 0 };
        let scales_vec = crate::kernels::neon::f16_bits_to_f32_x4(
            neon,
            [
                u16::from_le_bytes([b0[scale_offset], b0[scale_offset + 1]]),
                u16::from_le_bytes([b1[scale_offset], b1[scale_offset + 1]]),
                u16::from_le_bytes([b2[scale_offset], b2[scale_offset + 1]]),
                u16::from_le_bytes([b3[scale_offset], b3[scale_offset + 1]]),
            ],
        );
        if !scales_vec.is_finite().all_true() {
            return Some(Err(crate::invalid("nonfinite tensor or activation value")));
        }
        let row_scales = scales_vec.to_array();
        *decoded_blocks = decoded_blocks.saturating_add(4);
        if use_ptq1_dotprod {
            let trits = [
                bonsai_neon_dot::decode_ptq1_trits_i8(b0.try_into().ok()?),
                bonsai_neon_dot::decode_ptq1_trits_i8(b1.try_into().ok()?),
                bonsai_neon_dot::decode_ptq1_trits_i8(b2.try_into().ok()?),
                bonsai_neon_dot::decode_ptq1_trits_i8(b3.try_into().ok()?),
            ];
            let mut token = 0;
            while token + 4 <= preparations.len() {
                let acts: [&[crate::kernels::Q8Block; 4]; 4] = [
                    <&[crate::kernels::Q8Block; 4]>::try_from(
                        &preparations[token].q8[index * 4..index * 4 + 4],
                    )
                    .ok()?,
                    <&[crate::kernels::Q8Block; 4]>::try_from(
                        &preparations[token + 1].q8[index * 4..index * 4 + 4],
                    )
                    .ok()?,
                    <&[crate::kernels::Q8Block; 4]>::try_from(
                        &preparations[token + 2].q8[index * 4..index * 4 + 4],
                    )
                    .ok()?,
                    <&[crate::kernels::Q8Block; 4]>::try_from(
                        &preparations[token + 3].q8[index * 4..index * 4 + 4],
                    )
                    .ok()?,
                ];
                let act_values: [[&[i8; 32]; 4]; 4] = std::array::from_fn(|t| {
                    [
                        &acts[t][0].values,
                        &acts[t][1].values,
                        &acts[t][2].values,
                        &acts[t][3].values,
                    ]
                });
                if pinned {
                    let tile_lanes = bonsai_neon_dot::dot_ptq1_4x4_lanes(&trits, act_values);
                    for r in 0..4 {
                        let rs = row_scales[r];
                        for t in 0..4 {
                            let mut acc = lanes[r][token + t];
                            for (group, lane_sums) in tile_lanes[r][t].iter().enumerate() {
                                let floats =
                                    i32x4::from_slice(neon, lane_sums).to_float::<f32x4<Neon>>();
                                let d = f32x4::splat(neon, rs * acts[t][group].scale);
                                acc = floats.mul_add_precise(d, acc);
                            }
                            lanes[r][token + t] = acc;
                        }
                    }
                } else {
                    let tile_groups = bonsai_neon_dot::dot_ptq1_4x4_groups(&trits, act_values);
                    let act_scales: [f32x4<Neon>; 4] = std::array::from_fn(|t| {
                        f32x4::from_slice(
                            neon,
                            &[
                                acts[t][0].scale,
                                acts[t][1].scale,
                                acts[t][2].scale,
                                acts[t][3].scale,
                            ],
                        )
                    });
                    for r in 0..4 {
                        let rs = row_scales[r];
                        for t in 0..4 {
                            let dots = i32x4::from_slice(neon, &tile_groups[r][t])
                                .to_float::<f32x4<Neon>>();
                            let prod = (dots * act_scales[t]).to_array();
                            let dot_sum = (((0.0 + prod[0]) + prod[1]) + prod[2]) + prod[3];
                            sums[r][token + t] += rs * dot_sum;
                        }
                    }
                }
                token += 4;
            }
            for (token, scratch) in preparations.iter().enumerate().skip(token) {
                let activations = <&[crate::kernels::Q8Block; 4]>::try_from(
                    &scratch.q8[index * 4..index * 4 + 4],
                )
                .ok()?;
                let act_values = [
                    &activations[0].values,
                    &activations[1].values,
                    &activations[2].values,
                    &activations[3].values,
                ];
                if pinned {
                    let row_lanes = bonsai_neon_dot::dot_ptq1_4x1_lanes(&trits, act_values);
                    for r in 0..4 {
                        let rs = row_scales[r];
                        let mut acc = lanes[r][token];
                        for (group, lane_sums) in row_lanes[r].iter().enumerate() {
                            let floats =
                                i32x4::from_slice(neon, lane_sums).to_float::<f32x4<Neon>>();
                            let d = f32x4::splat(neon, rs * activations[group].scale);
                            acc = floats.mul_add_precise(d, acc);
                        }
                        lanes[r][token] = acc;
                    }
                } else {
                    let act_scales = f32x4::from_slice(
                        neon,
                        &[
                            activations[0].scale,
                            activations[1].scale,
                            activations[2].scale,
                            activations[3].scale,
                        ],
                    );
                    let row_groups = bonsai_neon_dot::dot_ptq1_4x1_groups(&trits, act_values);
                    for r in 0..4 {
                        let dots =
                            i32x4::from_slice(neon, &row_groups[r]).to_float::<f32x4<Neon>>();
                        let prod = (dots * act_scales).to_array();
                        let dot_sum = (((0.0 + prod[0]) + prod[1]) + prod[2]) + prod[3];
                        sums[r][token] += row_scales[r] * dot_sum;
                    }
                }
            }
            continue;
        }
        let weights = if is_ptq1 {
            [
                crate::kernels::neon::decode_ptq1_i16(neon, b0.try_into().ok()?),
                crate::kernels::neon::decode_ptq1_i16(neon, b1.try_into().ok()?),
                crate::kernels::neon::decode_ptq1_i16(neon, b2.try_into().ok()?),
                crate::kernels::neon::decode_ptq1_i16(neon, b3.try_into().ok()?),
            ]
        } else {
            [
                decode_pq2_sequential(neon, b0[2..34].try_into().ok()?),
                decode_pq2_sequential(neon, b1[2..34].try_into().ok()?),
                decode_pq2_sequential(neon, b2[2..34].try_into().ok()?),
                decode_pq2_sequential(neon, b3[2..34].try_into().ok()?),
            ]
        };
        let mut token = 0;
        while token + 4 <= preparations.len() {
            let acts: [&[crate::kernels::Q8Block; 4]; 4] = [
                <&[crate::kernels::Q8Block; 4]>::try_from(
                    &preparations[token].q8[index * 4..index * 4 + 4],
                )
                .ok()?,
                <&[crate::kernels::Q8Block; 4]>::try_from(
                    &preparations[token + 1].q8[index * 4..index * 4 + 4],
                )
                .ok()?,
                <&[crate::kernels::Q8Block; 4]>::try_from(
                    &preparations[token + 2].q8[index * 4..index * 4 + 4],
                )
                .ok()?,
                <&[crate::kernels::Q8Block; 4]>::try_from(
                    &preparations[token + 3].q8[index * 4..index * 4 + 4],
                )
                .ok()?,
            ];
            if is_ptq1 && pinned {
                for group in 0..4 {
                    let act_group: [_; 4] = std::array::from_fn(|t| {
                        let halves = acts[t][group].values.as_chunks::<16>().0;
                        let (l0, h0) = i8x16::from_slice(neon, &halves[0]).widen();
                        let (l1, h1) = i8x16::from_slice(neon, &halves[1]).widen();
                        (l0, h0, l1, h1, acts[t][group].scale)
                    });
                    for r in 0..4 {
                        let w0 = weights[r][group * 4];
                        let w1 = weights[r][group * 4 + 1];
                        let w2 = weights[r][group * 4 + 2];
                        let w3 = weights[r][group * 4 + 3];
                        let rs = row_scales[r];
                        for (t, &(l0, h0, l1, h1, scale)) in act_group.iter().enumerate() {
                            let sum_01 = w0 * l0 + w2 * l1;
                            let sum_23 = w1 * h0 + w3 * h1;
                            let (q0, q1) = sum_01.widen();
                            let (q2, q3) = sum_23.widen();
                            let floats =
                                reduce4_i32x4_vec(q0, q1, q2, q3).to_float::<f32x4<Neon>>();
                            let d = f32x4::splat(neon, rs * scale);
                            lanes[r][token + t] = floats.mul_add_precise(d, lanes[r][token + t]);
                        }
                    }
                }
            } else {
                let mut group_sums = [[[i32x4::splat(neon, 0); 4]; 4]; 4];
                for group in 0..4 {
                    let act_group: [_; 4] = std::array::from_fn(|t| {
                        let halves = acts[t][group].values.as_chunks::<16>().0;
                        let (l0, h0) = i8x16::from_slice(neon, &halves[0]).widen();
                        let (l1, h1) = i8x16::from_slice(neon, &halves[1]).widen();
                        (l0, h0, l1, h1)
                    });
                    for r in 0..4 {
                        let w0 = weights[r][group * 4];
                        let w1 = weights[r][group * 4 + 1];
                        let w2 = weights[r][group * 4 + 2];
                        let w3 = weights[r][group * 4 + 3];
                        for (t, &(l0, h0, l1, h1)) in act_group.iter().enumerate() {
                            let pairs = (w0 * l0 + w1 * h0) + (w2 * l1 + w3 * h1);
                            let (sum_low, sum_high) = pairs.widen();
                            group_sums[r][t][group] = sum_low + sum_high;
                        }
                    }
                }
                let act_scales: [f32x4<Neon>; 4] = std::array::from_fn(|t| {
                    f32x4::from_slice(
                        neon,
                        &[
                            acts[t][0].scale,
                            acts[t][1].scale,
                            acts[t][2].scale,
                            acts[t][3].scale,
                        ],
                    )
                });
                for r in 0..4 {
                    let rs = row_scales[r];
                    for t in 0..4 {
                        let gs = group_sums[r][t];
                        let dots =
                            reduce4_i32x4_vec(gs[0], gs[1], gs[2], gs[3]).to_float::<f32x4<Neon>>();
                        let prod = (dots * act_scales[t]).to_array();
                        let dot_sum = (((0.0 + prod[0]) + prod[1]) + prod[2]) + prod[3];
                        sums[r][token + t] += rs * dot_sum;
                    }
                }
            }
            token += 4;
        }
        for (token, scratch) in preparations.iter().enumerate().skip(token) {
            let activations =
                <&[crate::kernels::Q8Block; 4]>::try_from(&scratch.q8[index * 4..index * 4 + 4])
                    .ok()?;
            let act_groups: [_; 4] = std::array::from_fn(|group| {
                let halves = activations[group].values.as_chunks::<16>().0;
                let (l0, h0) = i8x16::from_slice(neon, &halves[0]).widen();
                let (l1, h1) = i8x16::from_slice(neon, &halves[1]).widen();
                (l0, h0, l1, h1)
            });
            if is_ptq1 && pinned {
                for r in 0..4 {
                    let mut acc = lanes[r][token];
                    for (group, &(l0, h0, l1, h1)) in act_groups.iter().enumerate() {
                        let sum_01 = weights[r][group * 4] * l0 + weights[r][group * 4 + 2] * l1;
                        let sum_23 =
                            weights[r][group * 4 + 1] * h0 + weights[r][group * 4 + 3] * h1;
                        let (q0, q1) = sum_01.widen();
                        let (q2, q3) = sum_23.widen();
                        let floats = reduce4_i32x4_vec(q0, q1, q2, q3).to_float::<f32x4<Neon>>();
                        let d = f32x4::splat(neon, row_scales[r] * activations[group].scale);
                        acc = floats.mul_add_precise(d, acc);
                    }
                    lanes[r][token] = acc;
                }
            } else {
                let act_scales = f32x4::from_slice(
                    neon,
                    &[
                        activations[0].scale,
                        activations[1].scale,
                        activations[2].scale,
                        activations[3].scale,
                    ],
                );
                for r in 0..4 {
                    let group_sums: [i32x4<Neon>; 4] = std::array::from_fn(|group| {
                        let (l0, h0, l1, h1) = act_groups[group];
                        let pairs = (weights[r][group * 4] * l0 + weights[r][group * 4 + 1] * h0)
                            + (weights[r][group * 4 + 2] * l1 + weights[r][group * 4 + 3] * h1);
                        let (sum_low, sum_high) = pairs.widen();
                        sum_low + sum_high
                    });
                    let dots = reduce4_i32x4_vec(
                        group_sums[0],
                        group_sums[1],
                        group_sums[2],
                        group_sums[3],
                    )
                    .to_float::<f32x4<Neon>>();
                    let prod = (dots * act_scales).to_array();
                    let dot_sum = (((0.0 + prod[0]) + prod[1]) + prod[2]) + prod[3];
                    sums[r][token] += row_scales[r] * dot_sum;
                }
            }
        }
    }
    for r in 0..4 {
        for (sum, lane_vec) in sums[r].iter_mut().zip(lanes[r]).take(preparations.len()) {
            let value = if is_ptq1 && pinned {
                let lane = lane_vec.to_array();
                (lane[0] + lane[1]) + (lane[2] + lane[3])
            } else {
                *sum
            };
            match crate::kernels::finite(value) {
                Ok(checked) => *sum = checked,
                Err(err) => return Some(Err(err)),
            }
        }
    }
    Some(Ok(sums))
}

#[expect(
    clippy::suboptimal_flops,
    clippy::too_many_lines,
    reason = "Preserve separately rounded source arithmetic and eight-row Q8 activation sharing."
)]
pub(in crate::kernels::prefill) fn dot_q8_eight_rows<const TOKENS: usize>(
    rows: [&[u8]; 8],
    preparations: &[crate::kernels::MatvecScratch],
    is_ptq1: bool,
    pinned: bool,
    decoded_blocks: &mut u64,
) -> Option<crate::Result<[[f32; TOKENS]; 8]>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let mut sums = [[0.0_f32; TOKENS]; 8];
    let block_bytes = if is_ptq1 { 28 } else { 34 };
    let block_count = rows[0].len() / block_bytes;
    if is_ptq1
        && !preparations.is_empty()
        && preparations.len() <= TOKENS
        && block_count != 0
        && rows.iter().all(|row| row.len() == block_count * 28)
        && preparations
            .iter()
            .all(|scratch| scratch.q8.len() >= block_count * 4)
        && let Some(dotprod) = bonsai_neon_dot::Ptq1Dotprod::detect()
    {
        let q8_len = block_count * 4;
        let (chunks8, rem8) = preparations.as_chunks::<8>();
        if !chunks8.is_empty() {
            let mut out_buf = [[[0.0_f32; 8]; 8]; 8];
            for (batch_idx, batch) in chunks8.chunks(8).enumerate() {
                let mut arg_chunks: [[&[crate::kernels::Q8Block]; 8]; 8] = [[&[]; 8]; 8];
                for (c, chunk) in batch.iter().enumerate() {
                    arg_chunks[c] = std::array::from_fn(|t| &chunk[t].q8[..q8_len]);
                }
                let out_slice = &mut out_buf[..batch.len()];
                let result = if pinned {
                    dotprod.dot_scaled_8x8_chunks_up_to_8::<true, _>(
                        rows,
                        &arg_chunks[..batch.len()],
                        out_slice,
                    )
                } else {
                    dotprod.dot_scaled_8x8_chunks_up_to_8::<false, _>(
                        rows,
                        &arg_chunks[..batch.len()],
                        out_slice,
                    )
                };
                let Some(()) = result else {
                    return Some(Err(crate::invalid("nonfinite tensor or activation value")));
                };
                for (c, chunk_out) in out_slice.iter().enumerate() {
                    let base = (batch_idx * 8 + c) * 8;
                    for row in 0..8 {
                        sums[row][base..base + 8].copy_from_slice(&chunk_out[row]);
                    }
                }
            }
        }
        if !rem8.is_empty() {
            let mut tok_refs: [&[crate::kernels::Q8Block]; 8] = [&[]; 8];
            for (t, scratch) in rem8.iter().enumerate() {
                tok_refs[t] = &scratch.q8[..q8_len];
            }
            let mut out_chunk = [[0.0_f32; 8]; 8];
            let result = if pinned {
                dotprod.dot_scaled_8x_tokens_view::<true, _>(
                    rows,
                    &tok_refs[..rem8.len()],
                    &mut out_chunk,
                )
            } else {
                dotprod.dot_scaled_8x_tokens_view::<false, _>(
                    rows,
                    &tok_refs[..rem8.len()],
                    &mut out_chunk,
                )
            };
            let Some(()) = result else {
                return Some(Err(crate::invalid("nonfinite tensor or activation value")));
            };
            let base = chunks8.len() * 8;
            for row in 0..8 {
                sums[row][base..base + rem8.len()].copy_from_slice(&out_chunk[row][..rem8.len()]);
            }
        }
        let block_count_u64 = u64::try_from(block_count).ok()?;
        *decoded_blocks = decoded_blocks.saturating_add(block_count_u64.saturating_mul(8));
        return Some(Ok(sums));
    }
    let use_ptq1_dotprod = is_ptq1 && bonsai_neon_dot::has_dotprod();
    let mut lanes = [[f32x4::splat(neon, 0.0); TOKENS]; 8];
    let scale_offset = if is_ptq1 { 26 } else { 0 };
    for index in 0..block_count {
        let start_byte = index * block_bytes;
        let end_byte = start_byte + block_bytes;
        let blocks: [&[u8]; 8] = std::array::from_fn(|r| &rows[r][start_byte..end_byte]);
        let (scales_lo, scales_hi) = crate::kernels::neon::f16_bits_to_f32_x8(
            neon,
            &[
                u16::from_le_bytes([blocks[0][scale_offset], blocks[0][scale_offset + 1]]),
                u16::from_le_bytes([blocks[1][scale_offset], blocks[1][scale_offset + 1]]),
                u16::from_le_bytes([blocks[2][scale_offset], blocks[2][scale_offset + 1]]),
                u16::from_le_bytes([blocks[3][scale_offset], blocks[3][scale_offset + 1]]),
                u16::from_le_bytes([blocks[4][scale_offset], blocks[4][scale_offset + 1]]),
                u16::from_le_bytes([blocks[5][scale_offset], blocks[5][scale_offset + 1]]),
                u16::from_le_bytes([blocks[6][scale_offset], blocks[6][scale_offset + 1]]),
                u16::from_le_bytes([blocks[7][scale_offset], blocks[7][scale_offset + 1]]),
            ],
        );
        if !(scales_lo.is_finite() & scales_hi.is_finite()).all_true() {
            return Some(Err(crate::invalid("nonfinite tensor or activation value")));
        }
        let scales_lo_arr = scales_lo.to_array();
        let scales_hi_arr = scales_hi.to_array();
        let row_scales = [
            scales_lo_arr[0],
            scales_lo_arr[1],
            scales_lo_arr[2],
            scales_lo_arr[3],
            scales_hi_arr[0],
            scales_hi_arr[1],
            scales_hi_arr[2],
            scales_hi_arr[3],
        ];
        *decoded_blocks = decoded_blocks.saturating_add(8);
        if use_ptq1_dotprod {
            let trits = [
                bonsai_neon_dot::decode_ptq1_trits_i8(blocks[0].try_into().ok()?),
                bonsai_neon_dot::decode_ptq1_trits_i8(blocks[1].try_into().ok()?),
                bonsai_neon_dot::decode_ptq1_trits_i8(blocks[2].try_into().ok()?),
                bonsai_neon_dot::decode_ptq1_trits_i8(blocks[3].try_into().ok()?),
                bonsai_neon_dot::decode_ptq1_trits_i8(blocks[4].try_into().ok()?),
                bonsai_neon_dot::decode_ptq1_trits_i8(blocks[5].try_into().ok()?),
                bonsai_neon_dot::decode_ptq1_trits_i8(blocks[6].try_into().ok()?),
                bonsai_neon_dot::decode_ptq1_trits_i8(blocks[7].try_into().ok()?),
            ];
            let mut token = 0;
            while token + 8 <= preparations.len() {
                let acts: [&[crate::kernels::Q8Block; 4]; 8] = [
                    <&[crate::kernels::Q8Block; 4]>::try_from(
                        &preparations[token].q8[index * 4..index * 4 + 4],
                    )
                    .ok()?,
                    <&[crate::kernels::Q8Block; 4]>::try_from(
                        &preparations[token + 1].q8[index * 4..index * 4 + 4],
                    )
                    .ok()?,
                    <&[crate::kernels::Q8Block; 4]>::try_from(
                        &preparations[token + 2].q8[index * 4..index * 4 + 4],
                    )
                    .ok()?,
                    <&[crate::kernels::Q8Block; 4]>::try_from(
                        &preparations[token + 3].q8[index * 4..index * 4 + 4],
                    )
                    .ok()?,
                    <&[crate::kernels::Q8Block; 4]>::try_from(
                        &preparations[token + 4].q8[index * 4..index * 4 + 4],
                    )
                    .ok()?,
                    <&[crate::kernels::Q8Block; 4]>::try_from(
                        &preparations[token + 5].q8[index * 4..index * 4 + 4],
                    )
                    .ok()?,
                    <&[crate::kernels::Q8Block; 4]>::try_from(
                        &preparations[token + 6].q8[index * 4..index * 4 + 4],
                    )
                    .ok()?,
                    <&[crate::kernels::Q8Block; 4]>::try_from(
                        &preparations[token + 7].q8[index * 4..index * 4 + 4],
                    )
                    .ok()?,
                ];
                let act_values: [[&[i8; 32]; 4]; 8] = std::array::from_fn(|t| {
                    [
                        &acts[t][0].values,
                        &acts[t][1].values,
                        &acts[t][2].values,
                        &acts[t][3].values,
                    ]
                });
                if pinned {
                    let tile_lanes = bonsai_neon_dot::dot_ptq1_8x8_lanes(&trits, act_values);
                    for r in 0..8 {
                        let rs = row_scales[r];
                        for t in 0..8 {
                            let mut acc = lanes[r][token + t];
                            for (group, lane_sums) in tile_lanes[r][t].iter().enumerate() {
                                let floats =
                                    i32x4::from_slice(neon, lane_sums).to_float::<f32x4<Neon>>();
                                let d = f32x4::splat(neon, rs * acts[t][group].scale);
                                acc = floats.mul_add_precise(d, acc);
                            }
                            lanes[r][token + t] = acc;
                        }
                    }
                } else {
                    let act_scales: [f32x4<Neon>; 8] = std::array::from_fn(|t| {
                        f32x4::from_slice(
                            neon,
                            &[
                                acts[t][0].scale,
                                acts[t][1].scale,
                                acts[t][2].scale,
                                acts[t][3].scale,
                            ],
                        )
                    });
                    let tile_groups = bonsai_neon_dot::dot_ptq1_8x8_groups(&trits, act_values);
                    for r in 0..8 {
                        let rs = row_scales[r];
                        for t in 0..8 {
                            let dots = i32x4::from_slice(neon, &tile_groups[r][t])
                                .to_float::<f32x4<Neon>>();
                            let prod = (dots * act_scales[t]).to_array();
                            let dot_sum = (((0.0 + prod[0]) + prod[1]) + prod[2]) + prod[3];
                            sums[r][token + t] += rs * dot_sum;
                        }
                    }
                }
                token += 8;
            }
            while token + 4 <= preparations.len() {
                let acts: [&[crate::kernels::Q8Block; 4]; 4] = [
                    <&[crate::kernels::Q8Block; 4]>::try_from(
                        &preparations[token].q8[index * 4..index * 4 + 4],
                    )
                    .ok()?,
                    <&[crate::kernels::Q8Block; 4]>::try_from(
                        &preparations[token + 1].q8[index * 4..index * 4 + 4],
                    )
                    .ok()?,
                    <&[crate::kernels::Q8Block; 4]>::try_from(
                        &preparations[token + 2].q8[index * 4..index * 4 + 4],
                    )
                    .ok()?,
                    <&[crate::kernels::Q8Block; 4]>::try_from(
                        &preparations[token + 3].q8[index * 4..index * 4 + 4],
                    )
                    .ok()?,
                ];
                let act_values: [[&[i8; 32]; 4]; 4] = std::array::from_fn(|t| {
                    [
                        &acts[t][0].values,
                        &acts[t][1].values,
                        &acts[t][2].values,
                        &acts[t][3].values,
                    ]
                });
                if pinned {
                    let tile_lanes = bonsai_neon_dot::dot_ptq1_8x4_lanes(&trits, act_values);
                    for r in 0..8 {
                        let rs = row_scales[r];
                        for t in 0..4 {
                            let mut acc = lanes[r][token + t];
                            for (group, lane_sums) in tile_lanes[r][t].iter().enumerate() {
                                let floats =
                                    i32x4::from_slice(neon, lane_sums).to_float::<f32x4<Neon>>();
                                let d = f32x4::splat(neon, rs * acts[t][group].scale);
                                acc = floats.mul_add_precise(d, acc);
                            }
                            lanes[r][token + t] = acc;
                        }
                    }
                } else {
                    let act_scales: [f32x4<Neon>; 4] = std::array::from_fn(|t| {
                        f32x4::from_slice(
                            neon,
                            &[
                                acts[t][0].scale,
                                acts[t][1].scale,
                                acts[t][2].scale,
                                acts[t][3].scale,
                            ],
                        )
                    });
                    let tile_groups = bonsai_neon_dot::dot_ptq1_8x4_groups(&trits, act_values);
                    for r in 0..8 {
                        let rs = row_scales[r];
                        for t in 0..4 {
                            let dots = i32x4::from_slice(neon, &tile_groups[r][t])
                                .to_float::<f32x4<Neon>>();
                            let prod = (dots * act_scales[t]).to_array();
                            let dot_sum = (((0.0 + prod[0]) + prod[1]) + prod[2]) + prod[3];
                            sums[r][token + t] += rs * dot_sum;
                        }
                    }
                }
                token += 4;
            }
            for (token, scratch) in preparations.iter().enumerate().skip(token) {
                let activations = <&[crate::kernels::Q8Block; 4]>::try_from(
                    &scratch.q8[index * 4..index * 4 + 4],
                )
                .ok()?;
                let act_values = [
                    &activations[0].values,
                    &activations[1].values,
                    &activations[2].values,
                    &activations[3].values,
                ];
                if pinned {
                    let row_lanes = bonsai_neon_dot::dot_ptq1_8x1_lanes(&trits, act_values);
                    for r in 0..8 {
                        let rs = row_scales[r];
                        let mut acc = lanes[r][token];
                        for (group, lane_sums) in row_lanes[r].iter().enumerate() {
                            let floats =
                                i32x4::from_slice(neon, lane_sums).to_float::<f32x4<Neon>>();
                            let d = f32x4::splat(neon, rs * activations[group].scale);
                            acc = floats.mul_add_precise(d, acc);
                        }
                        lanes[r][token] = acc;
                    }
                } else {
                    let act_scales = f32x4::from_slice(
                        neon,
                        &[
                            activations[0].scale,
                            activations[1].scale,
                            activations[2].scale,
                            activations[3].scale,
                        ],
                    );
                    let row_groups = bonsai_neon_dot::dot_ptq1_8x1_groups(&trits, act_values);
                    for r in 0..8 {
                        let dots =
                            i32x4::from_slice(neon, &row_groups[r]).to_float::<f32x4<Neon>>();
                        let prod = (dots * act_scales).to_array();
                        let dot_sum = (((0.0 + prod[0]) + prod[1]) + prod[2]) + prod[3];
                        sums[r][token] += row_scales[r] * dot_sum;
                    }
                }
            }
            continue;
        }
        let weights = if is_ptq1 {
            [
                crate::kernels::neon::decode_ptq1_i16(neon, blocks[0].try_into().ok()?),
                crate::kernels::neon::decode_ptq1_i16(neon, blocks[1].try_into().ok()?),
                crate::kernels::neon::decode_ptq1_i16(neon, blocks[2].try_into().ok()?),
                crate::kernels::neon::decode_ptq1_i16(neon, blocks[3].try_into().ok()?),
                crate::kernels::neon::decode_ptq1_i16(neon, blocks[4].try_into().ok()?),
                crate::kernels::neon::decode_ptq1_i16(neon, blocks[5].try_into().ok()?),
                crate::kernels::neon::decode_ptq1_i16(neon, blocks[6].try_into().ok()?),
                crate::kernels::neon::decode_ptq1_i16(neon, blocks[7].try_into().ok()?),
            ]
        } else {
            [
                decode_pq2_sequential(neon, blocks[0][2..34].try_into().ok()?),
                decode_pq2_sequential(neon, blocks[1][2..34].try_into().ok()?),
                decode_pq2_sequential(neon, blocks[2][2..34].try_into().ok()?),
                decode_pq2_sequential(neon, blocks[3][2..34].try_into().ok()?),
                decode_pq2_sequential(neon, blocks[4][2..34].try_into().ok()?),
                decode_pq2_sequential(neon, blocks[5][2..34].try_into().ok()?),
                decode_pq2_sequential(neon, blocks[6][2..34].try_into().ok()?),
                decode_pq2_sequential(neon, blocks[7][2..34].try_into().ok()?),
            ]
        };
        let mut token = 0;
        while token + 4 <= preparations.len() {
            let acts: [&[crate::kernels::Q8Block; 4]; 4] = [
                <&[crate::kernels::Q8Block; 4]>::try_from(
                    &preparations[token].q8[index * 4..index * 4 + 4],
                )
                .ok()?,
                <&[crate::kernels::Q8Block; 4]>::try_from(
                    &preparations[token + 1].q8[index * 4..index * 4 + 4],
                )
                .ok()?,
                <&[crate::kernels::Q8Block; 4]>::try_from(
                    &preparations[token + 2].q8[index * 4..index * 4 + 4],
                )
                .ok()?,
                <&[crate::kernels::Q8Block; 4]>::try_from(
                    &preparations[token + 3].q8[index * 4..index * 4 + 4],
                )
                .ok()?,
            ];
            if is_ptq1 && pinned {
                for group in 0..4 {
                    let act_group: [_; 4] = std::array::from_fn(|t| {
                        let halves = acts[t][group].values.as_chunks::<16>().0;
                        let (l0, h0) = i8x16::from_slice(neon, &halves[0]).widen();
                        let (l1, h1) = i8x16::from_slice(neon, &halves[1]).widen();
                        (l0, h0, l1, h1, acts[t][group].scale)
                    });
                    for r in 0..8 {
                        let w0 = weights[r][group * 4];
                        let w1 = weights[r][group * 4 + 1];
                        let w2 = weights[r][group * 4 + 2];
                        let w3 = weights[r][group * 4 + 3];
                        let rs = row_scales[r];
                        for (t, &(l0, h0, l1, h1, scale)) in act_group.iter().enumerate() {
                            let sum_01 = w0 * l0 + w2 * l1;
                            let sum_23 = w1 * h0 + w3 * h1;
                            let (q0, q1) = sum_01.widen();
                            let (q2, q3) = sum_23.widen();
                            let floats =
                                reduce4_i32x4_vec(q0, q1, q2, q3).to_float::<f32x4<Neon>>();
                            let d = f32x4::splat(neon, rs * scale);
                            lanes[r][token + t] = floats.mul_add_precise(d, lanes[r][token + t]);
                        }
                    }
                }
            } else {
                let mut group_sums = [[[i32x4::splat(neon, 0); 4]; 4]; 8];
                for group in 0..4 {
                    let act_group: [_; 4] = std::array::from_fn(|t| {
                        let halves = acts[t][group].values.as_chunks::<16>().0;
                        let (l0, h0) = i8x16::from_slice(neon, &halves[0]).widen();
                        let (l1, h1) = i8x16::from_slice(neon, &halves[1]).widen();
                        (l0, h0, l1, h1)
                    });
                    for r in 0..8 {
                        let w0 = weights[r][group * 4];
                        let w1 = weights[r][group * 4 + 1];
                        let w2 = weights[r][group * 4 + 2];
                        let w3 = weights[r][group * 4 + 3];
                        for (t, &(l0, h0, l1, h1)) in act_group.iter().enumerate() {
                            let pairs = (w0 * l0 + w1 * h0) + (w2 * l1 + w3 * h1);
                            let (sum_low, sum_high) = pairs.widen();
                            group_sums[r][t][group] = sum_low + sum_high;
                        }
                    }
                }
                let act_scales: [f32x4<Neon>; 4] = std::array::from_fn(|t| {
                    f32x4::from_slice(
                        neon,
                        &[
                            acts[t][0].scale,
                            acts[t][1].scale,
                            acts[t][2].scale,
                            acts[t][3].scale,
                        ],
                    )
                });
                for r in 0..8 {
                    let rs = row_scales[r];
                    for t in 0..4 {
                        let gs = group_sums[r][t];
                        let dots =
                            reduce4_i32x4_vec(gs[0], gs[1], gs[2], gs[3]).to_float::<f32x4<Neon>>();
                        let prod = (dots * act_scales[t]).to_array();
                        let dot_sum = (((0.0 + prod[0]) + prod[1]) + prod[2]) + prod[3];
                        sums[r][token + t] += rs * dot_sum;
                    }
                }
            }
            token += 4;
        }
        for (token, scratch) in preparations.iter().enumerate().skip(token) {
            let activations =
                <&[crate::kernels::Q8Block; 4]>::try_from(&scratch.q8[index * 4..index * 4 + 4])
                    .ok()?;
            let act_groups: [_; 4] = std::array::from_fn(|group| {
                let halves = activations[group].values.as_chunks::<16>().0;
                let (l0, h0) = i8x16::from_slice(neon, &halves[0]).widen();
                let (l1, h1) = i8x16::from_slice(neon, &halves[1]).widen();
                (l0, h0, l1, h1)
            });
            if is_ptq1 && pinned {
                for r in 0..8 {
                    let mut acc = lanes[r][token];
                    for (group, &(l0, h0, l1, h1)) in act_groups.iter().enumerate() {
                        let sum_01 = weights[r][group * 4] * l0 + weights[r][group * 4 + 2] * l1;
                        let sum_23 =
                            weights[r][group * 4 + 1] * h0 + weights[r][group * 4 + 3] * h1;
                        let (q0, q1) = sum_01.widen();
                        let (q2, q3) = sum_23.widen();
                        let floats = reduce4_i32x4_vec(q0, q1, q2, q3).to_float::<f32x4<Neon>>();
                        let d = f32x4::splat(neon, row_scales[r] * activations[group].scale);
                        acc = floats.mul_add_precise(d, acc);
                    }
                    lanes[r][token] = acc;
                }
            } else {
                let act_scales = f32x4::from_slice(
                    neon,
                    &[
                        activations[0].scale,
                        activations[1].scale,
                        activations[2].scale,
                        activations[3].scale,
                    ],
                );
                for r in 0..8 {
                    let group_sums: [i32x4<Neon>; 4] = std::array::from_fn(|group| {
                        let (l0, h0, l1, h1) = act_groups[group];
                        let pairs = (weights[r][group * 4] * l0 + weights[r][group * 4 + 1] * h0)
                            + (weights[r][group * 4 + 2] * l1 + weights[r][group * 4 + 3] * h1);
                        let (sum_low, sum_high) = pairs.widen();
                        sum_low + sum_high
                    });
                    let dots = reduce4_i32x4_vec(
                        group_sums[0],
                        group_sums[1],
                        group_sums[2],
                        group_sums[3],
                    )
                    .to_float::<f32x4<Neon>>();
                    let prod = (dots * act_scales).to_array();
                    let dot_sum = (((0.0 + prod[0]) + prod[1]) + prod[2]) + prod[3];
                    sums[r][token] += row_scales[r] * dot_sum;
                }
            }
        }
    }
    for r in 0..8 {
        for (sum, lane_vec) in sums[r].iter_mut().zip(lanes[r]).take(preparations.len()) {
            let value = if is_ptq1 && pinned {
                let lane = lane_vec.to_array();
                (lane[0] + lane[1]) + (lane[2] + lane[3])
            } else {
                *sum
            };
            match crate::kernels::finite(value) {
                Ok(checked) => *sum = checked,
                Err(err) => return Some(Err(err)),
            }
        }
    }
    Some(Ok(sums))
}

#[inline]
fn decode_bf16_pair(neon: Neon, zero: u16x8<Neon>, bytes: &[u8; 16]) -> (f32x4<Neon>, f32x4<Neon>) {
    let raw = u8x16::from_slice(neon, bytes).bitcast::<u16x8<Neon>>();
    (
        zero.zip_low(raw).bitcast::<f32x4<Neon>>(),
        zero.zip_high(raw).bitcast::<f32x4<Neon>>(),
    )
}

#[inline]
fn decode_bf16_tail_weights(rem: &[u8]) -> crate::Result<([f32; 7], usize)> {
    let mut weights = [0.0_f32; 7];
    let chunks = rem.as_chunks::<2>().0;
    if chunks.len() > weights.len() {
        return Err(crate::invalid("invalid BF16 tail extent"));
    }
    for (dst, bytes) in weights.iter_mut().zip(chunks) {
        *dst = crate::kernels::finite(f32::from_bits(u32::from(u16::from_le_bytes(*bytes)) << 16))?;
    }
    Ok((weights, chunks.len()))
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    reason = "Vectorize paired BF16 rows with float64x2 accumulation while preserving exact scalar 0 ULP sum order."
)]
fn dot_bf16_row_pairs<const PAIRS: usize, const ROWS: usize, const TOKENS: usize>(
    neon: Neon,
    rows: [&[u8]; ROWS],
    preparations: &[crate::kernels::MatvecScratch],
) -> crate::Result<[[f32; TOKENS]; ROWS]> {
    const K_TILE: usize = 16;
    let token_count = preparations.len();
    if token_count == 0
        || token_count > TOKENS
        || !rows[0].len().is_multiple_of(2)
        || rows.iter().any(|row| row.len() != rows[0].len())
        || preparations
            .iter()
            .any(|prep| prep.activations.len() * 2 != rows[0].len())
    {
        return Err(crate::invalid("invalid BF16 tile geometry"));
    }

    let zero = u16x8::splat(neon, 0);
    let mut finite_mask = f32x4::splat(neon, 0.0).is_finite();
    let b_chunks = rows.map(|row| row.as_chunks::<16>().0);
    let b_rems = rows.map(|row| row.as_chunks::<16>().1);
    let chunk_count = b_chunks[0].len();
    let mut sums = [[f64x2::splat(neon, -0.0); TOKENS]; PAIRS];
    let (tok_quads, tok_rem) = preparations.as_chunks::<4>();

    for k_base in (0..chunk_count).step_by(K_TILE) {
        let tile_len = (chunk_count - k_base).min(K_TILE);
        for (p, pair_sums) in sums.iter_mut().enumerate() {
            let r_even = &b_chunks[p * 2];
            let r_odd = &b_chunks[p * 2 + 1];
            let mut even_tile = [[f32x4::splat(neon, 0.0); 2]; K_TILE];
            let mut odd_tile = [[f32x4::splat(neon, 0.0); 2]; K_TILE];
            for k_off in 0..tile_len {
                let idx = k_base + k_off;
                let (we0, we1) = decode_bf16_pair(neon, zero, &r_even[idx]);
                let (wo0, wo1) = decode_bf16_pair(neon, zero, &r_odd[idx]);
                finite_mask = finite_mask
                    & we0.is_finite()
                    & we1.is_finite()
                    & wo0.is_finite()
                    & wo1.is_finite();
                even_tile[k_off] = [we0, we1];
                odd_tile[k_off] = [wo0, wo1];
            }

            for (quad_idx, quad) in tok_quads.iter().enumerate() {
                let base_t = quad_idx * 4;
                let mut acc = [
                    pair_sums[base_t],
                    pair_sums[base_t + 1],
                    pair_sums[base_t + 2],
                    pair_sums[base_t + 3],
                ];
                for k_off in 0..tile_len {
                    let col = (k_base + k_off) * 8;
                    let [we0, we1] = even_tile[k_off];
                    let [wo0, wo1] = odd_tile[k_off];
                    for (t, acc_t) in acc.iter_mut().enumerate() {
                        let a_lo = f32x4::from_slice(neon, &quad[t].activations[col..col + 4]);
                        let a_hi = f32x4::from_slice(neon, &quad[t].activations[col + 4..col + 8]);
                        let s = add_f32_columns(*acc_t, we0 * a_lo, wo0 * a_lo);
                        *acc_t = add_f32_columns(s, we1 * a_hi, wo1 * a_hi);
                    }
                }
                pair_sums[base_t..base_t + 4].copy_from_slice(&acc);
            }
            if !tok_rem.is_empty() {
                let rem_base = tok_quads.len() * 4;
                let mut acc = [f64x2::splat(neon, -0.0); 3];
                acc[..tok_rem.len()]
                    .copy_from_slice(&pair_sums[rem_base..rem_base + tok_rem.len()]);
                for k_off in 0..tile_len {
                    let col = (k_base + k_off) * 8;
                    let [we0, we1] = even_tile[k_off];
                    let [wo0, wo1] = odd_tile[k_off];
                    for (t_off, scratch) in tok_rem.iter().enumerate() {
                        let a_lo = f32x4::from_slice(neon, &scratch.activations[col..col + 4]);
                        let a_hi = f32x4::from_slice(neon, &scratch.activations[col + 4..col + 8]);
                        let s = add_f32_columns(acc[t_off], we0 * a_lo, wo0 * a_lo);
                        acc[t_off] = add_f32_columns(s, we1 * a_hi, wo1 * a_hi);
                    }
                }
                pair_sums[rem_base..rem_base + tok_rem.len()]
                    .copy_from_slice(&acc[..tok_rem.len()]);
            }
        }
    }

    if !finite_mask.all_true() {
        return Err(crate::invalid("nonfinite tensor or activation value"));
    }
    let rem_base = chunk_count * 8;
    let mut out = [[0.0_f32; TOKENS]; ROWS];
    for (p, pair_sums) in sums.iter().enumerate() {
        let r0 = p * 2;
        let r1 = r0 + 1;
        let (tail_w0, tail_len) = decode_bf16_tail_weights(b_rems[r0])?;
        let (tail_w1, tail_len1) = decode_bf16_tail_weights(b_rems[r1])?;
        if tail_len != tail_len1 {
            return Err(crate::invalid("invalid BF16 tail geometry"));
        }
        for (tok, (prep, pair_sum)) in preparations.iter().zip(pair_sums).enumerate() {
            let [mut s0, mut s1] = pair_sum.to_array();
            for (idx, (&w0, &w1)) in tail_w0[..tail_len]
                .iter()
                .zip(&tail_w1[..tail_len])
                .enumerate()
            {
                let act = prep.activations[rem_base + idx];
                s0 += f64::from(w0 * act);
                s1 += f64::from(w1 * act);
            }
            out[r0][tok] = crate::kernels::finite(s0 as f32)?;
            out[r1][tok] = crate::kernels::finite(s1 as f32)?;
        }
    }
    Ok(out)
}

pub(in crate::kernels::prefill) fn dot_bf16_rows<const ROWS: usize, const TOKENS: usize>(
    rows: [&[u8]; ROWS],
    preparations: &[crate::kernels::MatvecScratch],
) -> Option<crate::Result<[[f32; TOKENS]; ROWS]>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    match ROWS {
        4 => Some(dot_bf16_row_pairs::<2, ROWS, TOKENS>(
            neon,
            rows,
            preparations,
        )),
        8 => Some(dot_bf16_row_pairs::<4, ROWS, TOKENS>(
            neon,
            rows,
            preparations,
        )),
        _ => None,
    }
}

#[inline]
fn add_f32_columns(sum: f64x2<Neon>, first: f32x4<Neon>, second: f32x4<Neon>) -> f64x2<Neon> {
    let (l0, l1) = first.zip_low(second).widen();
    let (h0, h1) = first.zip_high(second).widen();
    (((sum + l0) + l1) + h0) + h1
}

#[inline]
fn decode_f32_tail_weights(rem: &[u8]) -> crate::Result<([f32; 3], usize)> {
    let mut weights = [0.0_f32; 3];
    let chunks = rem.as_chunks::<4>().0;
    if chunks.len() > weights.len() {
        return Err(crate::invalid("invalid F32 tail extent"));
    }
    for (dst, bytes) in weights.iter_mut().zip(chunks) {
        *dst = crate::kernels::finite(f32::from_le_bytes(*bytes))?;
    }
    Ok((weights, chunks.len()))
}

fn dot_f32_row_pinned<const TOKENS: usize>(
    neon: Neon,
    row: &[u8],
    preparations: &[crate::kernels::MatvecScratch],
) -> crate::Result<[f32; TOKENS]> {
    const K_TILE: usize = 16;
    let chunks16 = row.as_chunks::<64>().0;
    let mut lanes = [[f32x4::splat(neon, 0.0); 4]; TOKENS];
    let mut finite_mask = f32x4::splat(neon, 0.0).is_finite();
    let (tok_quads, tok_rem) = preparations.as_chunks::<4>();

    for k_base in (0..chunks16.len()).step_by(K_TILE) {
        let k_end = (k_base + K_TILE).min(chunks16.len());
        for (quad_idx, quad) in tok_quads.iter().enumerate() {
            let base_t = quad_idx * 4;
            let mut acc = [
                lanes[base_t],
                lanes[base_t + 1],
                lanes[base_t + 2],
                lanes[base_t + 3],
            ];
            for (blk_off, blk) in chunks16[k_base..k_end].iter().enumerate() {
                let col = (k_base + blk_off) * 16;
                let b4 = blk.as_chunks::<16>().0;
                let w0 = u8x16::from_slice(neon, &b4[0]).bitcast::<f32x4<Neon>>();
                let w1 = u8x16::from_slice(neon, &b4[1]).bitcast::<f32x4<Neon>>();
                let w2 = u8x16::from_slice(neon, &b4[2]).bitcast::<f32x4<Neon>>();
                let w3 = u8x16::from_slice(neon, &b4[3]).bitcast::<f32x4<Neon>>();
                if quad_idx == 0 {
                    finite_mask = finite_mask
                        & w0.is_finite()
                        & w1.is_finite()
                        & w2.is_finite()
                        & w3.is_finite();
                }
                for (scratch, acc_tok) in quad.iter().zip(acc.iter_mut()) {
                    let act = &scratch.activations[col..col + 16];
                    acc_tok[0] =
                        w0.mul_add_precise(f32x4::from_slice(neon, &act[0..4]), acc_tok[0]);
                    acc_tok[1] =
                        w1.mul_add_precise(f32x4::from_slice(neon, &act[4..8]), acc_tok[1]);
                    acc_tok[2] =
                        w2.mul_add_precise(f32x4::from_slice(neon, &act[8..12]), acc_tok[2]);
                    acc_tok[3] =
                        w3.mul_add_precise(f32x4::from_slice(neon, &act[12..16]), acc_tok[3]);
                }
            }
            lanes[base_t..base_t + 4].copy_from_slice(&acc);
        }
        if !tok_rem.is_empty() {
            let rem_base = tok_quads.len() * 4;
            let check_weights = tok_quads.is_empty();
            for (t_off, scratch) in tok_rem.iter().enumerate() {
                let mut acc = lanes[rem_base + t_off];
                for (blk_off, blk) in chunks16[k_base..k_end].iter().enumerate() {
                    let col = (k_base + blk_off) * 16;
                    let b4 = blk.as_chunks::<16>().0;
                    let w0 = u8x16::from_slice(neon, &b4[0]).bitcast::<f32x4<Neon>>();
                    let w1 = u8x16::from_slice(neon, &b4[1]).bitcast::<f32x4<Neon>>();
                    let w2 = u8x16::from_slice(neon, &b4[2]).bitcast::<f32x4<Neon>>();
                    let w3 = u8x16::from_slice(neon, &b4[3]).bitcast::<f32x4<Neon>>();
                    if check_weights && t_off == 0 {
                        finite_mask = finite_mask
                            & w0.is_finite()
                            & w1.is_finite()
                            & w2.is_finite()
                            & w3.is_finite();
                    }
                    let act = &scratch.activations[col..col + 16];
                    acc[0] = w0.mul_add_precise(f32x4::from_slice(neon, &act[0..4]), acc[0]);
                    acc[1] = w1.mul_add_precise(f32x4::from_slice(neon, &act[4..8]), acc[1]);
                    acc[2] = w2.mul_add_precise(f32x4::from_slice(neon, &act[8..12]), acc[2]);
                    acc[3] = w3.mul_add_precise(f32x4::from_slice(neon, &act[12..16]), acc[3]);
                }
                lanes[rem_base + t_off] = acc;
            }
        }
    }

    if !finite_mask.all_true() {
        return Err(crate::invalid("nonfinite tensor or activation value"));
    }
    let mut out = [0.0_f32; TOKENS];
    for (dst, [s0, s1, s2, s3]) in out.iter_mut().zip(lanes).take(preparations.len()) {
        let sum = ((s0 + s2) + (s1 + s3)).reduce_sum();
        *dst = crate::kernels::finite(sum)?;
    }
    Ok(out)
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Retain the scalar floating dot's rounded F32 products, binary64 sum and final F32 rounding."
)]
fn dot_f32_row_f64<const TOKENS: usize>(
    neon: Neon,
    row: &[u8],
    preparations: &[crate::kernels::MatvecScratch],
) -> crate::Result<[f32; TOKENS]> {
    const K_TILE: usize = 64;
    let (chunks4, rem_bytes) = row.as_chunks::<16>();
    let mut finite_mask = f32x4::splat(neon, 0.0).is_finite();
    let (tok_pairs, tok_rem) = preparations.as_chunks::<2>();
    let mut pair_sums = [f64x2::splat(neon, -0.0); TOKENS];
    let mut odd_sum = -0.0_f64;

    for k_base in (0..chunks4.len()).step_by(K_TILE) {
        let k_end = (k_base + K_TILE).min(chunks4.len());
        for (batch_idx, pair_batch) in tok_pairs.chunks(4).enumerate() {
            let pair_base = batch_idx * 4;
            let mut acc = [f64x2::splat(neon, -0.0); 4];
            acc[..pair_batch.len()]
                .copy_from_slice(&pair_sums[pair_base..pair_base + pair_batch.len()]);
            for (off, w_bytes) in chunks4[k_base..k_end].iter().enumerate() {
                let col = (k_base + off) * 4;
                let w = u8x16::from_slice(neon, w_bytes).bitcast::<f32x4<Neon>>();
                if batch_idx == 0 {
                    finite_mask &= w.is_finite();
                }
                for ([prep0, prep1], acc_pair) in pair_batch.iter().zip(acc.iter_mut()) {
                    let a0 = f32x4::from_slice(neon, &prep0.activations[col..col + 4]);
                    let a1 = f32x4::from_slice(neon, &prep1.activations[col..col + 4]);
                    *acc_pair = add_f32_columns(*acc_pair, w * a0, w * a1);
                }
            }
            pair_sums[pair_base..pair_base + pair_batch.len()]
                .copy_from_slice(&acc[..pair_batch.len()]);
        }
        if let [odd_prep] = tok_rem {
            let check_weights = tok_pairs.is_empty();
            for (off, w_bytes) in chunks4[k_base..k_end].iter().enumerate() {
                let col = (k_base + off) * 4;
                let w = u8x16::from_slice(neon, w_bytes).bitcast::<f32x4<Neon>>();
                if check_weights {
                    finite_mask &= w.is_finite();
                }
                let p =
                    (w * f32x4::from_slice(neon, &odd_prep.activations[col..col + 4])).to_array();
                odd_sum += f64::from(p[0]);
                odd_sum += f64::from(p[1]);
                odd_sum += f64::from(p[2]);
                odd_sum += f64::from(p[3]);
            }
        }
    }

    if !finite_mask.all_true() {
        return Err(crate::invalid("nonfinite tensor or activation value"));
    }
    let (tail_w, tail_len) = decode_f32_tail_weights(rem_bytes)?;
    let rem_base = chunks4.len() * 4;
    let mut out = [0.0_f32; TOKENS];
    for (pair_idx, [prep0, prep1]) in tok_pairs.iter().enumerate() {
        let [mut s0, mut s1] = pair_sums[pair_idx].to_array();
        for (idx, &w) in tail_w[..tail_len].iter().enumerate() {
            let col = rem_base + idx;
            s0 += f64::from(w * prep0.activations[col]);
            s1 += f64::from(w * prep1.activations[col]);
        }
        let base_t = pair_idx * 2;
        out[base_t] = crate::kernels::finite(s0 as f32)?;
        out[base_t + 1] = crate::kernels::finite(s1 as f32)?;
    }
    if let [odd_prep] = tok_rem {
        for (idx, &w) in tail_w[..tail_len].iter().enumerate() {
            odd_sum += f64::from(w * odd_prep.activations[rem_base + idx]);
        }
        out[tok_pairs.len() * 2] = crate::kernels::finite(odd_sum as f32)?;
    }
    Ok(out)
}

#[expect(
    clippy::too_many_lines,
    reason = "Keep 2x2 F32 pinned FMA register-blocked tile and weight validation together."
)]
fn dot_f32_rows_pinned<const PAIRS: usize, const ROWS: usize, const TOKENS: usize>(
    neon: Neon,
    rows: [&[u8]; ROWS],
    preparations: &[crate::kernels::MatvecScratch],
) -> crate::Result<[[f32; TOKENS]; ROWS]> {
    const K_TILE: usize = 8;
    let r_chunks = rows.map(|row| row.as_chunks::<64>().0);
    let block_count = r_chunks[0].len();
    let mut lanes = [[[f32x4::splat(neon, 0.0); 4]; TOKENS]; ROWS];
    let mut finite_mask = f32x4::splat(neon, 0.0).is_finite();
    let (tok_pairs, tok_rem) = preparations.as_chunks::<2>();

    for k_base in (0..block_count).step_by(K_TILE) {
        let k_end = (k_base + K_TILE).min(block_count);
        for (pair_idx, [prep0, prep1]) in tok_pairs.iter().enumerate() {
            let t0 = pair_idx * 2;
            let t1 = t0 + 1;
            for row_pair in 0..PAIRS {
                let r0 = row_pair * 2;
                let r1 = r0 + 1;
                let mut acc00 = lanes[r0][t0];
                let mut acc01 = lanes[r0][t1];
                let mut acc10 = lanes[r1][t0];
                let mut acc11 = lanes[r1][t1];
                for blk_idx in k_base..k_end {
                    let col = blk_idx * 16;
                    let act0 = &prep0.activations[col..col + 16];
                    let act1 = &prep1.activations[col..col + 16];
                    let a00 = f32x4::from_slice(neon, &act0[0..4]);
                    let a01 = f32x4::from_slice(neon, &act0[4..8]);
                    let a02 = f32x4::from_slice(neon, &act0[8..12]);
                    let a03 = f32x4::from_slice(neon, &act0[12..16]);
                    let a10 = f32x4::from_slice(neon, &act1[0..4]);
                    let a11 = f32x4::from_slice(neon, &act1[4..8]);
                    let a12 = f32x4::from_slice(neon, &act1[8..12]);
                    let a13 = f32x4::from_slice(neon, &act1[12..16]);

                    let b0 = r_chunks[r0][blk_idx].as_chunks::<16>().0;
                    let w00 = u8x16::from_slice(neon, &b0[0]).bitcast::<f32x4<Neon>>();
                    let w01 = u8x16::from_slice(neon, &b0[1]).bitcast::<f32x4<Neon>>();
                    let w02 = u8x16::from_slice(neon, &b0[2]).bitcast::<f32x4<Neon>>();
                    let w03 = u8x16::from_slice(neon, &b0[3]).bitcast::<f32x4<Neon>>();
                    if pair_idx == 0 {
                        finite_mask = finite_mask
                            & w00.is_finite()
                            & w01.is_finite()
                            & w02.is_finite()
                            & w03.is_finite();
                    }
                    acc00[0] = w00.mul_add_precise(a00, acc00[0]);
                    acc01[0] = w00.mul_add_precise(a10, acc01[0]);
                    acc00[1] = w01.mul_add_precise(a01, acc00[1]);
                    acc01[1] = w01.mul_add_precise(a11, acc01[1]);
                    acc00[2] = w02.mul_add_precise(a02, acc00[2]);
                    acc01[2] = w02.mul_add_precise(a12, acc01[2]);
                    acc00[3] = w03.mul_add_precise(a03, acc00[3]);
                    acc01[3] = w03.mul_add_precise(a13, acc01[3]);

                    let b1 = r_chunks[r1][blk_idx].as_chunks::<16>().0;
                    let w10 = u8x16::from_slice(neon, &b1[0]).bitcast::<f32x4<Neon>>();
                    let w11 = u8x16::from_slice(neon, &b1[1]).bitcast::<f32x4<Neon>>();
                    let w12 = u8x16::from_slice(neon, &b1[2]).bitcast::<f32x4<Neon>>();
                    let w13 = u8x16::from_slice(neon, &b1[3]).bitcast::<f32x4<Neon>>();
                    if pair_idx == 0 {
                        finite_mask = finite_mask
                            & w10.is_finite()
                            & w11.is_finite()
                            & w12.is_finite()
                            & w13.is_finite();
                    }
                    acc10[0] = w10.mul_add_precise(a00, acc10[0]);
                    acc11[0] = w10.mul_add_precise(a10, acc11[0]);
                    acc10[1] = w11.mul_add_precise(a01, acc10[1]);
                    acc11[1] = w11.mul_add_precise(a11, acc11[1]);
                    acc10[2] = w12.mul_add_precise(a02, acc10[2]);
                    acc11[2] = w12.mul_add_precise(a12, acc11[2]);
                    acc10[3] = w13.mul_add_precise(a03, acc10[3]);
                    acc11[3] = w13.mul_add_precise(a13, acc11[3]);
                }
                lanes[r0][t0] = acc00;
                lanes[r0][t1] = acc01;
                lanes[r1][t0] = acc10;
                lanes[r1][t1] = acc11;
            }
        }
        if let [odd_prep] = tok_rem {
            let t_last = tok_pairs.len() * 2;
            let check_weights = tok_pairs.is_empty();
            for row_pair in 0..PAIRS {
                let r0 = row_pair * 2;
                let r1 = r0 + 1;
                let mut acc0 = lanes[r0][t_last];
                let mut acc1 = lanes[r1][t_last];
                for blk_idx in k_base..k_end {
                    let col = blk_idx * 16;
                    let act = &odd_prep.activations[col..col + 16];
                    let a0 = f32x4::from_slice(neon, &act[0..4]);
                    let a1 = f32x4::from_slice(neon, &act[4..8]);
                    let a2 = f32x4::from_slice(neon, &act[8..12]);
                    let a3 = f32x4::from_slice(neon, &act[12..16]);

                    let b0 = r_chunks[r0][blk_idx].as_chunks::<16>().0;
                    let w00 = u8x16::from_slice(neon, &b0[0]).bitcast::<f32x4<Neon>>();
                    let w01 = u8x16::from_slice(neon, &b0[1]).bitcast::<f32x4<Neon>>();
                    let w02 = u8x16::from_slice(neon, &b0[2]).bitcast::<f32x4<Neon>>();
                    let w03 = u8x16::from_slice(neon, &b0[3]).bitcast::<f32x4<Neon>>();
                    if check_weights {
                        finite_mask = finite_mask
                            & w00.is_finite()
                            & w01.is_finite()
                            & w02.is_finite()
                            & w03.is_finite();
                    }
                    acc0[0] = w00.mul_add_precise(a0, acc0[0]);
                    acc0[1] = w01.mul_add_precise(a1, acc0[1]);
                    acc0[2] = w02.mul_add_precise(a2, acc0[2]);
                    acc0[3] = w03.mul_add_precise(a3, acc0[3]);

                    let b1 = r_chunks[r1][blk_idx].as_chunks::<16>().0;
                    let w10 = u8x16::from_slice(neon, &b1[0]).bitcast::<f32x4<Neon>>();
                    let w11 = u8x16::from_slice(neon, &b1[1]).bitcast::<f32x4<Neon>>();
                    let w12 = u8x16::from_slice(neon, &b1[2]).bitcast::<f32x4<Neon>>();
                    let w13 = u8x16::from_slice(neon, &b1[3]).bitcast::<f32x4<Neon>>();
                    if check_weights {
                        finite_mask = finite_mask
                            & w10.is_finite()
                            & w11.is_finite()
                            & w12.is_finite()
                            & w13.is_finite();
                    }
                    acc1[0] = w10.mul_add_precise(a0, acc1[0]);
                    acc1[1] = w11.mul_add_precise(a1, acc1[1]);
                    acc1[2] = w12.mul_add_precise(a2, acc1[2]);
                    acc1[3] = w13.mul_add_precise(a3, acc1[3]);
                }
                lanes[r0][t_last] = acc0;
                lanes[r1][t_last] = acc1;
            }
        }
    }

    if !finite_mask.all_true() {
        return Err(crate::invalid("nonfinite tensor or activation value"));
    }
    let mut out = [[0.0_f32; TOKENS]; ROWS];
    for (out_row, row_lanes) in out.iter_mut().zip(lanes) {
        for (dst, [s0, s1, s2, s3]) in out_row.iter_mut().zip(row_lanes).take(preparations.len()) {
            let sum = ((s0 + s2) + (s1 + s3)).reduce_sum();
            *dst = crate::kernels::finite(sum)?;
        }
    }
    Ok(out)
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    reason = "Retain exact scalar binary64 reduction order across paired F32 rows and register-blocked token quads."
)]
fn dot_f32_rows_f64<const PAIRS: usize, const ROWS: usize, const TOKENS: usize>(
    neon: Neon,
    rows: [&[u8]; ROWS],
    preparations: &[crate::kernels::MatvecScratch],
) -> crate::Result<[[f32; TOKENS]; ROWS]> {
    const K_TILE: usize = 32;
    let r_chunks = rows.map(|row| row.as_chunks::<16>().0);
    let r_rems = rows.map(|row| row.as_chunks::<16>().1);
    let chunk_count = r_chunks[0].len();
    let mut sums = [[f64x2::splat(neon, -0.0); TOKENS]; PAIRS];
    let mut finite_mask = f32x4::splat(neon, 0.0).is_finite();
    let (tok_quads, tok_rem) = preparations.as_chunks::<4>();

    for k_base in (0..chunk_count).step_by(K_TILE) {
        let k_end = (k_base + K_TILE).min(chunk_count);
        for (quad_idx, quad) in tok_quads.iter().enumerate() {
            let base_t = quad_idx * 4;
            let mut acc: [[f64x2<Neon>; 4]; PAIRS] = std::array::from_fn(|p| {
                [
                    sums[p][base_t],
                    sums[p][base_t + 1],
                    sums[p][base_t + 2],
                    sums[p][base_t + 3],
                ]
            });
            for idx in k_base..k_end {
                let col = idx * 4;
                let a0 = f32x4::from_slice(neon, &quad[0].activations[col..col + 4]);
                let a1 = f32x4::from_slice(neon, &quad[1].activations[col..col + 4]);
                let a2 = f32x4::from_slice(neon, &quad[2].activations[col..col + 4]);
                let a3 = f32x4::from_slice(neon, &quad[3].activations[col..col + 4]);
                for (p, acc_pair) in acc.iter_mut().enumerate() {
                    let we =
                        u8x16::from_slice(neon, &r_chunks[p * 2][idx]).bitcast::<f32x4<Neon>>();
                    let wo =
                        u8x16::from_slice(neon, &r_chunks[p * 2 + 1][idx]).bitcast::<f32x4<Neon>>();
                    if quad_idx == 0 {
                        finite_mask = finite_mask & we.is_finite() & wo.is_finite();
                    }
                    acc_pair[0] = add_f32_columns(acc_pair[0], we * a0, wo * a0);
                    acc_pair[1] = add_f32_columns(acc_pair[1], we * a1, wo * a1);
                    acc_pair[2] = add_f32_columns(acc_pair[2], we * a2, wo * a2);
                    acc_pair[3] = add_f32_columns(acc_pair[3], we * a3, wo * a3);
                }
            }
            for (p, acc_pair) in acc.iter().enumerate() {
                sums[p][base_t..base_t + 4].copy_from_slice(acc_pair);
            }
        }
        if !tok_rem.is_empty() {
            let rem_base = tok_quads.len() * 4;
            let check_weights = tok_quads.is_empty();
            let mut acc = [[f64x2::splat(neon, -0.0); 3]; PAIRS];
            for (p, acc_pair) in acc.iter_mut().enumerate() {
                acc_pair[..tok_rem.len()]
                    .copy_from_slice(&sums[p][rem_base..rem_base + tok_rem.len()]);
            }
            for idx in k_base..k_end {
                let col = idx * 4;
                let mut acts = [f32x4::splat(neon, 0.0); 3];
                for (t_off, scratch) in tok_rem.iter().enumerate() {
                    acts[t_off] = f32x4::from_slice(neon, &scratch.activations[col..col + 4]);
                }
                for (p, acc_pair) in acc.iter_mut().enumerate() {
                    let we =
                        u8x16::from_slice(neon, &r_chunks[p * 2][idx]).bitcast::<f32x4<Neon>>();
                    let wo =
                        u8x16::from_slice(neon, &r_chunks[p * 2 + 1][idx]).bitcast::<f32x4<Neon>>();
                    if check_weights {
                        finite_mask = finite_mask & we.is_finite() & wo.is_finite();
                    }
                    for (acc_tok, &act) in acc_pair[..tok_rem.len()].iter_mut().zip(&acts) {
                        *acc_tok = add_f32_columns(*acc_tok, we * act, wo * act);
                    }
                }
            }
            for (p, acc_pair) in acc.iter().enumerate() {
                sums[p][rem_base..rem_base + tok_rem.len()]
                    .copy_from_slice(&acc_pair[..tok_rem.len()]);
            }
        }
    }

    if !finite_mask.all_true() {
        return Err(crate::invalid("nonfinite tensor or activation value"));
    }
    let rem_base = chunk_count * 4;
    let mut out = [[0.0_f32; TOKENS]; ROWS];
    for (p, pair_sums) in sums.iter().enumerate() {
        let r0 = p * 2;
        let r1 = r0 + 1;
        let (tail_w0, tail_len) = decode_f32_tail_weights(r_rems[r0])?;
        let (tail_w1, tail_len1) = decode_f32_tail_weights(r_rems[r1])?;
        if tail_len != tail_len1 {
            return Err(crate::invalid("invalid F32 tail geometry"));
        }
        for (tok, (prep, pair_sum)) in preparations.iter().zip(pair_sums).enumerate() {
            let [mut s0, mut s1] = pair_sum.to_array();
            for (idx, (&w0, &w1)) in tail_w0[..tail_len]
                .iter()
                .zip(&tail_w1[..tail_len])
                .enumerate()
            {
                let act = prep.activations[rem_base + idx];
                s0 += f64::from(w0 * act);
                s1 += f64::from(w1 * act);
            }
            out[r0][tok] = crate::kernels::finite(s0 as f32)?;
            out[r1][tok] = crate::kernels::finite(s1 as f32)?;
        }
    }
    Ok(out)
}

#[inline]
fn validate_f32_tile_inputs<const ROWS: usize, const TOKENS: usize>(
    rows: [&[u8]; ROWS],
    preparations: &[crate::kernels::MatvecScratch],
    columns: usize,
) -> Option<()> {
    let token_count = preparations.len();
    if token_count == 0 || token_count > TOKENS || columns == 0 {
        return None;
    }
    let row_bytes = columns.checked_mul(4)?;
    if rows.iter().any(|row| row.len() != row_bytes)
        || preparations
            .iter()
            .any(|prep| prep.activations.len() != columns)
    {
        return None;
    }
    Some(())
}

pub(in crate::kernels::prefill) fn dot_f32_row<const TOKENS: usize>(
    row: &[u8],
    preparations: &[crate::kernels::MatvecScratch],
    columns: usize,
    pinned: bool,
) -> Option<crate::Result<[f32; TOKENS]>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    if validate_f32_tile_inputs::<1, TOKENS>([row], preparations, columns).is_none() {
        return Some(Err(crate::invalid("invalid F32 tile geometry")));
    }
    Some(if pinned && columns.is_multiple_of(16) {
        dot_f32_row_pinned(neon, row, preparations)
    } else {
        dot_f32_row_f64(neon, row, preparations)
    })
}

pub(in crate::kernels::prefill) fn dot_f32_four_rows<const TOKENS: usize>(
    rows: [&[u8]; 4],
    preparations: &[crate::kernels::MatvecScratch],
    columns: usize,
    pinned: bool,
) -> Option<crate::Result<[[f32; TOKENS]; 4]>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    if validate_f32_tile_inputs::<4, TOKENS>(rows, preparations, columns).is_none() {
        return Some(Err(crate::invalid("invalid F32 tile geometry")));
    }
    Some(if pinned && columns.is_multiple_of(16) {
        dot_f32_rows_pinned::<2, 4, TOKENS>(neon, rows, preparations)
    } else {
        dot_f32_rows_f64::<2, 4, TOKENS>(neon, rows, preparations)
    })
}

pub(in crate::kernels::prefill) fn dot_f32_eight_rows<const TOKENS: usize>(
    rows: [&[u8]; 8],
    preparations: &[crate::kernels::MatvecScratch],
    columns: usize,
    pinned: bool,
) -> Option<crate::Result<[[f32; TOKENS]; 8]>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    if validate_f32_tile_inputs::<8, TOKENS>(rows, preparations, columns).is_none() {
        return Some(Err(crate::invalid("invalid F32 tile geometry")));
    }
    Some(if pinned && columns.is_multiple_of(16) {
        dot_f32_rows_pinned::<4, 8, TOKENS>(neon, rows, preparations)
    } else {
        dot_f32_rows_f64::<4, 8, TOKENS>(neon, rows, preparations)
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::Result;
    use crate::kernels::prefill::MAX_TILE_TOKENS;
    use crate::kernels::{MatvecBackend, MatvecScratch, PQ2, PTQ1, matvec_into, quantize_row};
    use crate::model::Tensor;

    fn build_pq2_test_fixture(
        columns: usize,
        max_tokens: usize,
    ) -> Result<(Tensor, Vec<Vec<f32>>)> {
        let block_count = columns / 128;
        let row_bytes = block_count * 34;
        let mut data = Vec::with_capacity(8 * row_bytes);
        for row in 0..8_usize {
            let values: Vec<f32> = (0..columns)
                .map(|col| {
                    [
                        0.0_f32,
                        -0.0,
                        f32::from_bits(1),
                        -f32::from_bits(1),
                        1.25,
                        -2.5,
                        0.375,
                        -0.0625,
                    ][(col * 3 + row * 5) % 8]
                })
                .collect();
            let mut encoded = quantize_row(PQ2, &values)?;
            if row == 0 {
                encoded[..2].copy_from_slice(&0x8000_u16.to_le_bytes());
            } else if row == 1 {
                encoded[34..36].copy_from_slice(&0x0001_u16.to_le_bytes());
            } else if row == 2 {
                encoded[68..70].copy_from_slice(&0x8001_u16.to_le_bytes());
            }
            data.extend_from_slice(&encoded);
        }
        let tensor = Tensor {
            name: "prefill-pq2-8x-tokens".into(),
            dimensions: vec![columns, 8],
            kind: PQ2,
            data: Arc::<[u8]>::from(data).into(),
        };
        let inputs: Vec<Vec<f32>> = (0..max_tokens)
            .map(|tok| {
                (0..columns)
                    .map(|col| {
                        [
                            0.0_f32,
                            -0.0,
                            f32::from_bits(1),
                            -128.0,
                            127.0,
                            63.5,
                            -31.25,
                            0.5,
                        ][(col + tok * 7) % 8]
                    })
                    .collect()
            })
            .collect();
        Ok((tensor, inputs))
    }

    #[test]
    fn prefill_pq2_eight_rows_full_k_tails_and_extremes_match_scalar_bits() -> Result<()> {
        let columns = 512_usize;
        let block_count = columns / 128;
        let row_bytes = block_count * 34;
        let max_tokens = MAX_TILE_TOKENS;
        let (tensor, inputs) = build_pq2_test_fixture(columns, max_tokens)?;
        let row_refs: [&[u8]; 8] =
            std::array::from_fn(|r| &tensor.data[r * row_bytes..(r + 1) * row_bytes]);

        for pinned in [false, true] {
            let mut scalar_outputs: Vec<Vec<u32>> = Vec::with_capacity(max_tokens);
            let mut preps: Vec<MatvecScratch> = Vec::with_capacity(max_tokens);
            for input in &inputs {
                let mut scalar_scratch = if pinned {
                    MatvecScratch::with_pinned_backend(MatvecBackend::Scalar)
                } else {
                    MatvecScratch::with_backend(MatvecBackend::Scalar)
                };
                let mut out = Vec::new();
                matvec_into(&tensor, input, &mut out, &mut scalar_scratch)?;
                scalar_outputs.push(out.into_iter().map(f32::to_bits).collect());

                let mut neon_scratch = if pinned {
                    MatvecScratch::with_pinned_backend(MatvecBackend::ExperimentalNeon)
                } else {
                    MatvecScratch::with_backend(MatvecBackend::ExperimentalNeon)
                };
                let _ = neon_scratch.prepare(&tensor, input)?;
                preps.push(neon_scratch);
            }

            for count in [1_usize, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17, 24, 32, 48, 64] {
                let mut decoded = 0_u64;
                let actual = super::dot_pq2_eight_rows::<MAX_TILE_TOKENS>(
                    row_refs,
                    &preps[..count],
                    pinned,
                    &mut decoded,
                )
                .ok_or_else(|| crate::invalid("missing neon prefill 8-row kernel"))??;
                assert_eq!(decoded, u64::try_from(block_count * 8)?);
                for (row, out_row) in actual.iter().enumerate() {
                    let got: Vec<u32> =
                        out_row[..count].iter().copied().map(f32::to_bits).collect();
                    let expected: Vec<u32> =
                        scalar_outputs[..count].iter().map(|tok| tok[row]).collect();
                    assert_eq!(
                        got, expected,
                        "mismatch count={count} pinned={pinned} row={row}"
                    );
                }
            }
        }

        let mut bad_data = tensor.data.to_vec();
        bad_data[7 * row_bytes + 3 * 34..7 * row_bytes + 3 * 34 + 2]
            .copy_from_slice(&0x7e00_u16.to_le_bytes());
        let bad_rows: [&[u8]; 8] =
            std::array::from_fn(|r| &bad_data[r * row_bytes..(r + 1) * row_bytes]);
        let mut preps_bad = Vec::with_capacity(10);
        for input in &inputs[..10] {
            let mut s = MatvecScratch::with_pinned_backend(MatvecBackend::ExperimentalNeon);
            let _ = s.prepare(&tensor, input)?;
            preps_bad.push(s);
        }
        for count in [3_usize, 10] {
            let mut decoded = 0_u64;
            let res = super::dot_pq2_eight_rows::<MAX_TILE_TOKENS>(
                bad_rows,
                &preps_bad[..count],
                true,
                &mut decoded,
            )
            .ok_or_else(|| crate::invalid("missing neon prefill 8-row kernel"))?;
            assert!(res.is_err());
        }
        Ok(())
    }

    fn sample_ptq1_eight_row_tensor(
        columns: usize,
        max_tokens: usize,
    ) -> Result<(Tensor, Vec<Vec<f32>>)> {
        let block_count = columns / 128;
        let row_bytes = block_count * 28;
        let mut data = Vec::with_capacity(8 * row_bytes);
        for row in 0..8_usize {
            let values: Vec<f32> = (0..columns)
                .map(|col| {
                    [
                        0.0_f32,
                        -0.0,
                        f32::from_bits(1),
                        -f32::from_bits(1),
                        1.25,
                        -2.5,
                        0.375,
                        -0.0625,
                    ][(col * 3 + row * 5) % 8]
                })
                .collect();
            let mut encoded = quantize_row(PTQ1, &values)?;
            if row == 0 {
                encoded[26..28].copy_from_slice(&0x8000_u16.to_le_bytes());
            } else if row == 1 {
                encoded[28 + 26..28 + 28].copy_from_slice(&0x0001_u16.to_le_bytes());
            } else if row == 2 {
                encoded[56 + 26..56 + 28].copy_from_slice(&0x8001_u16.to_le_bytes());
            }
            data.extend_from_slice(&encoded);
        }
        let tensor = Tensor {
            name: "prefill-ptq1-8x-tokens".into(),
            dimensions: vec![columns, 8],
            kind: PTQ1,
            data: Arc::<[u8]>::from(data).into(),
        };
        let inputs: Vec<Vec<f32>> = (0..max_tokens)
            .map(|tok| {
                (0..columns)
                    .map(|col| {
                        [
                            0.0_f32,
                            -0.0,
                            f32::from_bits(1),
                            -128.0,
                            127.0,
                            63.5,
                            -31.25,
                            0.5,
                        ][(col + tok * 7) % 8]
                    })
                    .collect()
            })
            .collect();
        Ok((tensor, inputs))
    }

    fn verify_ptq1_eight_rows_mode(
        tensor: &Tensor,
        row_refs: [&[u8]; 8],
        inputs: &[Vec<f32>],
        block_count: usize,
        pinned: bool,
    ) -> Result<()> {
        let mut scalar_outputs: Vec<Vec<u32>> = Vec::with_capacity(inputs.len());
        let mut preps: Vec<MatvecScratch> = Vec::with_capacity(inputs.len());
        for input in inputs {
            let mut scalar_scratch = if pinned {
                MatvecScratch::with_pinned_backend(MatvecBackend::Scalar)
            } else {
                MatvecScratch::with_backend(MatvecBackend::Scalar)
            };
            let mut out = Vec::new();
            matvec_into(tensor, input, &mut out, &mut scalar_scratch)?;
            scalar_outputs.push(out.into_iter().map(f32::to_bits).collect());

            let mut neon_scratch = if pinned {
                MatvecScratch::with_pinned_backend(MatvecBackend::ExperimentalNeon)
            } else {
                MatvecScratch::with_backend(MatvecBackend::ExperimentalNeon)
            };
            let _ = neon_scratch.prepare(tensor, input)?;
            preps.push(neon_scratch);
        }

        for count in [1_usize, 2, 3, 4, 5, 7, 8, 9, 16, 24, 32, 48, 64] {
            let mut decoded = 0_u64;
            let actual = super::dot_q8_eight_rows::<MAX_TILE_TOKENS>(
                row_refs,
                &preps[..count],
                true,
                pinned,
                &mut decoded,
            )
            .ok_or_else(|| crate::invalid("missing neon prefill ptq1 8-row kernel"))??;
            assert_eq!(decoded, u64::try_from(block_count * 8)?);
            for (row, out_row) in actual.iter().enumerate() {
                let got: Vec<u32> = out_row[..count].iter().copied().map(f32::to_bits).collect();
                let expected: Vec<u32> =
                    scalar_outputs[..count].iter().map(|tok| tok[row]).collect();
                assert_eq!(
                    got, expected,
                    "ptq1 count={count} pinned={pinned} row={row}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn prefill_ptq1_eight_rows_full_k_matches_scalar_bits() -> Result<()> {
        let columns = 384_usize;
        let block_count = columns / 128;
        let row_bytes = block_count * 28;
        let (tensor, inputs) = sample_ptq1_eight_row_tensor(columns, MAX_TILE_TOKENS)?;
        let row_refs: [&[u8]; 8] =
            std::array::from_fn(|r| &tensor.data[r * row_bytes..(r + 1) * row_bytes]);
        for pinned in [false, true] {
            verify_ptq1_eight_rows_mode(&tensor, row_refs, &inputs, block_count, pinned)?;
        }

        let mut bad_data = tensor.data.to_vec();
        bad_data[6 * row_bytes + 2 * 28 + 26..6 * row_bytes + 2 * 28 + 28]
            .copy_from_slice(&0x7e00_u16.to_le_bytes());
        let bad_rows: [&[u8]; 8] =
            std::array::from_fn(|r| &bad_data[r * row_bytes..(r + 1) * row_bytes]);
        let mut bad_prep = MatvecScratch::with_pinned_backend(MatvecBackend::ExperimentalNeon);
        let _ = bad_prep.prepare(&tensor, &inputs[0])?;
        let mut decoded = 0_u64;
        assert!(
            super::dot_q8_eight_rows::<MAX_TILE_TOKENS>(
                bad_rows,
                &[bad_prep],
                true,
                true,
                &mut decoded,
            )
            .ok_or_else(|| crate::invalid("missing neon prefill ptq1 8-row kernel"))?
            .is_err()
        );
        Ok(())
    }
}
