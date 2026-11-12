//! Baseline NEON over one predecoded PQ2/PTQ1 block, reused across token dots.

use fearless_simd::{Neon, f32x4, i8x16, i16x8, i32x4, prelude::*, u8x16, u16x8};

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
            let rem_base = chunks4.len() * 4;
            for (offset, scratch) in rem4.iter().enumerate() {
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
            let rem_base = base4 + chunks4.len() * 4;
            for (offset, scratch) in rem4.iter().enumerate() {
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
    let use_ptq1_dotprod = is_ptq1 && bonsai_neon_dot::has_dotprod();
    let mut sums = [[0.0_f32; TOKENS]; 8];
    let mut lanes = [[f32x4::splat(neon, 0.0); TOKENS]; 8];
    let block_bytes = if is_ptq1 { 28 } else { 34 };
    let scale_offset = if is_ptq1 { 26 } else { 0 };
    let block_count = rows[0].len() / block_bytes;
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

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    reason = "Retain the scalar floating dot's rounded F32 products, binary64 sum and final F32 rounding."
)]
pub(in crate::kernels::prefill) fn dot_bf16_rows<const ROWS: usize, const TOKENS: usize>(
    rows: [&[u8]; ROWS],
    preparations: &[crate::kernels::MatvecScratch],
) -> Option<crate::Result<[[f32; TOKENS]; ROWS]>> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let zero = u16x8::splat(neon, 0);
    let mut finite_mask = f32x4::splat(neon, 0.0).is_finite();
    let b_chunks = rows.map(|row| row.as_chunks::<16>().0);
    let b_rems = rows.map(|row| row.as_chunks::<16>().1);
    let token_count = preparations.len();
    let mut sums = [[-0.0_f64; TOKENS]; ROWS];
    let chunk_count = b_chunks[0].len();
    for idx in 0..chunk_count {
        let mut w_pairs = [(f32x4::splat(neon, 0.0), f32x4::splat(neon, 0.0)); ROWS];
        for r in 0..ROWS {
            let raw = u8x16::from_slice(neon, &b_chunks[r][idx]).bitcast::<u16x8<Neon>>();
            let w0 = zero.zip_low(raw).bitcast::<f32x4<Neon>>();
            let w1 = zero.zip_high(raw).bitcast::<f32x4<Neon>>();
            finite_mask = finite_mask & w0.is_finite() & w1.is_finite();
            w_pairs[r] = (w0, w1);
        }
        let col_base = idx * 8;
        for (tok, prep) in preparations.iter().enumerate() {
            let in_chunk = &prep.activations[col_base..col_base + 8];
            let a0 = f32x4::from_slice(neon, &in_chunk[..4]);
            let a1 = f32x4::from_slice(neon, &in_chunk[4..]);
            for r in 0..ROWS {
                let (w0, w1) = w_pairs[r];
                let prod0 = (w0 * a0).to_array();
                let prod1 = (w1 * a1).to_array();
                sums[r][tok] += f64::from(prod0[0]);
                sums[r][tok] += f64::from(prod0[1]);
                sums[r][tok] += f64::from(prod0[2]);
                sums[r][tok] += f64::from(prod0[3]);
                sums[r][tok] += f64::from(prod1[0]);
                sums[r][tok] += f64::from(prod1[1]);
                sums[r][tok] += f64::from(prod1[2]);
                sums[r][tok] += f64::from(prod1[3]);
            }
        }
    }
    if !finite_mask.all_true() {
        return Some(Err(crate::invalid("nonfinite tensor or activation value")));
    }
    let rem_base = chunk_count * 8;
    for r in 0..ROWS {
        for (rem_idx, b) in b_rems[r].as_chunks::<2>().0.iter().enumerate() {
            let w = match crate::kernels::finite(f32::from_bits(
                u32::from(u16::from_le_bytes(*b)) << 16,
            )) {
                Ok(val) => val,
                Err(err) => return Some(Err(err)),
            };
            let col = rem_base + rem_idx;
            for (tok, prep) in preparations.iter().enumerate() {
                sums[r][tok] += f64::from(w * prep.activations[col]);
            }
        }
    }
    let mut out = [[0.0_f32; TOKENS]; ROWS];
    for r in 0..ROWS {
        for tok in 0..token_count {
            match crate::kernels::finite(sums[r][tok] as f32) {
                Ok(val) => out[r][tok] = val,
                Err(err) => return Some(Err(err)),
            }
        }
    }
    Some(Ok(out))
}
