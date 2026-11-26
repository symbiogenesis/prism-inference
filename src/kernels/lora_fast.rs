//! Fast register-pinned 4-row decode and token-parallel in-place prefill `LoRA` B kernels.
//!
//! Preserves strict `0 ULP` bit-exact parity with the 8-element FMA reduction tree
//! in [`super::dot_multi_scalar`] and [`super::neon`], eliminating single-row
//! rank reloads in decode and the serial transpose + `scratch.correction` copy in
//! multi-token prefill.

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
use fearless_simd::{Neon, f32x4, prelude::*};

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
pub(super) trait FastLoraWeight: Copy {
    fn load_x8(neon: Neon, slice: &[Self; 8]) -> (f32x4<Neon>, f32x4<Neon>);
    fn load_x4(neon: Neon, slice: &[Self; 4]) -> f32x4<Neon>;
    fn to_f32(self) -> f32;
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
impl FastLoraWeight for u16 {
    #[inline]
    fn load_x8(neon: Neon, slice: &[Self; 8]) -> (f32x4<Neon>, f32x4<Neon>) {
        super::neon::f16_bits_to_f32_x8(neon, slice)
    }

    #[inline]
    fn load_x4(neon: Neon, slice: &[Self; 4]) -> f32x4<Neon> {
        super::neon::f16_bits_to_f32_x4(neon, *slice)
    }

    #[inline]
    fn to_f32(self) -> f32 {
        super::half_to_float(self)
    }
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
impl FastLoraWeight for f32 {
    #[inline]
    fn load_x8(neon: Neon, slice: &[Self; 8]) -> (f32x4<Neon>, f32x4<Neon>) {
        (
            f32x4::from_slice(neon, &slice[..4]),
            f32x4::from_slice(neon, &slice[4..]),
        )
    }

    #[inline]
    fn load_x4(neon: Neon, slice: &[Self; 4]) -> f32x4<Neon> {
        f32x4::from_slice(neon, slice)
    }

    #[inline]
    fn to_f32(self) -> f32 {
        self
    }
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
#[inline]
fn reduce_acc_pair(acc0: f32x4<Neon>, acc1: f32x4<Neon>) -> f32 {
    let mut lanes = [0.0_f32; 4];
    (acc0 + acc1).store_slice(&mut lanes);
    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
#[inline]
fn dot_row_pinned_rank16<W: FastLoraWeight>(
    neon: Neon,
    b_row: &[W],
    r0: f32x4<Neon>,
    r1: f32x4<Neon>,
    r2: f32x4<Neon>,
    r3: f32x4<Neon>,
) -> f32 {
    let zero = f32x4::splat(neon, 0.0);
    let (chunks, _) = b_row.as_chunks::<8>();
    let (w0, w1) = W::load_x8(neon, &chunks[0]);
    let (w2, w3) = W::load_x8(neon, &chunks[1]);
    let acc0 = w2.mul_add_precise(r2, w0.mul_add_precise(r0, zero));
    let acc1 = w3.mul_add_precise(r3, w1.mul_add_precise(r1, zero));
    reduce_acc_pair(acc0, acc1)
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
#[inline]
fn dot_row_general<W: FastLoraWeight>(neon: Neon, b_row: &[W], ranks: &[f32]) -> f32 {
    let mut acc0 = f32x4::splat(neon, 0.0);
    let mut acc1 = f32x4::splat(neon, 0.0);
    let (w_chunks8, w_rem8) = b_row.as_chunks::<8>();
    let (x_chunks8, x_rem8) = ranks.as_chunks::<8>();
    for (w8, x8) in w_chunks8.iter().zip(x_chunks8) {
        let (w_lo, w_hi) = W::load_x8(neon, w8);
        let x_lo = f32x4::from_slice(neon, &x8[..4]);
        let x_hi = f32x4::from_slice(neon, &x8[4..]);
        acc0 = w_lo.mul_add_precise(x_lo, acc0);
        acc1 = w_hi.mul_add_precise(x_hi, acc1);
    }
    let (w_chunks4, w_tail) = w_rem8.as_chunks::<4>();
    let (x_chunks4, x_tail) = x_rem8.as_chunks::<4>();
    if let (Some(w4), Some(x4)) = (w_chunks4.first(), x_chunks4.first()) {
        let w = W::load_x4(neon, w4);
        let x = f32x4::from_slice(neon, x4);
        acc0 = w.mul_add_precise(x, acc0);
    }
    let mut sum = reduce_acc_pair(acc0, acc1);
    for (&w, &x) in w_tail.iter().zip(x_tail) {
        sum = w.to_f32().mul_add(x, sum);
    }
    sum
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
#[inline]
fn dot_four_rows_general<W: FastLoraWeight>(
    neon: Neon,
    rows: [&[W]; 4],
    ranks: &[f32],
) -> [f32; 4] {
    let (x_chunks8, x_rem8) = ranks.as_chunks::<8>();
    let (x_chunks4, x_tail) = x_rem8.as_chunks::<4>();
    let r_chunks8 = rows.map(|r| r.as_chunks::<8>().0);
    let r_rem8 = rows.map(|r| r.as_chunks::<8>().1);
    let mut acc0 = [f32x4::splat(neon, 0.0); 4];
    let mut acc1 = [f32x4::splat(neon, 0.0); 4];
    for (idx, x8) in x_chunks8.iter().enumerate() {
        let x_lo = f32x4::from_slice(neon, &x8[..4]);
        let x_hi = f32x4::from_slice(neon, &x8[4..]);
        for (rc, (a0, a1)) in r_chunks8.iter().zip(acc0.iter_mut().zip(acc1.iter_mut())) {
            let (w_lo, w_hi) = W::load_x8(neon, &rc[idx]);
            *a0 = w_lo.mul_add_precise(x_lo, *a0);
            *a1 = w_hi.mul_add_precise(x_hi, *a1);
        }
    }
    let r_chunks4 = r_rem8.map(|r| r.as_chunks::<4>().0);
    let r_tails = r_rem8.map(|r| r.as_chunks::<4>().1);
    if let Some(x4) = x_chunks4.first() {
        let x = f32x4::from_slice(neon, x4);
        for (rc4, a0) in r_chunks4.iter().zip(acc0.iter_mut()) {
            let w = W::load_x4(neon, &rc4[0]);
            *a0 = w.mul_add_precise(x, *a0);
        }
    }
    let mut sums = [
        reduce_acc_pair(acc0[0], acc1[0]),
        reduce_acc_pair(acc0[1], acc1[1]),
        reduce_acc_pair(acc0[2], acc1[2]),
        reduce_acc_pair(acc0[3], acc1[3]),
    ];
    for (i, &x) in x_tail.iter().enumerate() {
        for (sum, tail) in sums.iter_mut().zip(r_tails) {
            *sum = tail[i].to_f32().mul_add(x, *sum);
        }
    }
    sums
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
fn lora_multi_b_generic<W: FastLoraWeight>(
    b_weights: &[W],
    ranks: &[f32],
    scale: f32,
    base_output: &[f32],
    correction: &mut [f32],
) -> Option<bool> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let rank = ranks.len();
    if rank == 0
        || b_weights.len() != base_output.len().saturating_mul(rank)
        || correction.len() != base_output.len()
    {
        return Some(false);
    }
    let (base_quads, base_tail) = base_output.as_chunks::<4>();
    let (corr_quads, corr_tail) = correction.as_chunks_mut::<4>();
    let quad_weights_len = base_quads.len() * 4 * rank;
    let (b_quads, b_tail) = b_weights.split_at(quad_weights_len);

    if rank == 16 {
        let r0 = f32x4::from_slice(neon, &ranks[0..4]);
        let r1 = f32x4::from_slice(neon, &ranks[4..8]);
        let r2 = f32x4::from_slice(neon, &ranks[8..12]);
        let r3 = f32x4::from_slice(neon, &ranks[12..16]);
        for ((b_quad, base4), corr4) in b_quads
            .as_chunks::<64>()
            .0
            .iter()
            .zip(base_quads)
            .zip(corr_quads.iter_mut())
        {
            for (b_row, (&base, dst)) in b_quad
                .as_chunks::<16>()
                .0
                .iter()
                .zip(base4.iter().zip(corr4.iter_mut()))
            {
                let dot = dot_row_pinned_rank16(neon, b_row, r0, r1, r2, r3);
                let updated = dot.mul_add(scale, base);
                if !updated.is_finite() {
                    return Some(false);
                }
                *dst = updated;
            }
        }
        for (b_row, (&base, dst)) in b_tail
            .as_chunks::<16>()
            .0
            .iter()
            .zip(base_tail.iter().zip(corr_tail.iter_mut()))
        {
            let dot = dot_row_pinned_rank16(neon, b_row, r0, r1, r2, r3);
            let updated = dot.mul_add(scale, base);
            if !updated.is_finite() {
                return Some(false);
            }
            *dst = updated;
        }
        return Some(true);
    }

    for ((b_quad, base4), corr4) in b_quads
        .chunks_exact(4 * rank)
        .zip(base_quads)
        .zip(corr_quads.iter_mut())
    {
        let (r0, rest) = b_quad.split_at(rank);
        let (r1, rest) = rest.split_at(rank);
        let (r2, r3) = rest.split_at(rank);
        let dots = dot_four_rows_general(neon, [r0, r1, r2, r3], ranks);
        for ((dot, &base), dst) in dots.into_iter().zip(base4).zip(corr4.iter_mut()) {
            let updated = dot.mul_add(scale, base);
            if !updated.is_finite() {
                return Some(false);
            }
            *dst = updated;
        }
    }
    for (b_row, (&base, dst)) in b_tail
        .chunks_exact(rank)
        .zip(base_tail.iter().zip(corr_tail.iter_mut()))
    {
        let dot = dot_row_general(neon, b_row, ranks);
        let updated = dot.mul_add(scale, base);
        if !updated.is_finite() {
            return Some(false);
        }
        *dst = updated;
    }
    Some(true)
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
fn lora_multi_b_inplace_one<W: FastLoraWeight>(
    neon: Neon,
    b_weights: &[W],
    ranks: &[f32],
    scale: f32,
    output: &mut [f32],
) -> bool {
    let rank = ranks.len();
    let (out_quads, out_tail) = output.as_chunks_mut::<4>();
    let quad_weights_len = out_quads.len() * 4 * rank;
    let (b_quads, b_tail) = b_weights.split_at(quad_weights_len);

    if rank == 16 {
        let r0 = f32x4::from_slice(neon, &ranks[0..4]);
        let r1 = f32x4::from_slice(neon, &ranks[4..8]);
        let r2 = f32x4::from_slice(neon, &ranks[8..12]);
        let r3 = f32x4::from_slice(neon, &ranks[12..16]);
        for (b_quad, out4) in b_quads.as_chunks::<64>().0.iter().zip(out_quads.iter_mut()) {
            for (b_row, slot) in b_quad.as_chunks::<16>().0.iter().zip(out4.iter_mut()) {
                let dot = dot_row_pinned_rank16(neon, b_row, r0, r1, r2, r3);
                let updated = dot.mul_add(scale, *slot);
                if !updated.is_finite() {
                    return false;
                }
                *slot = updated;
            }
        }
        for (b_row, slot) in b_tail.as_chunks::<16>().0.iter().zip(out_tail.iter_mut()) {
            let dot = dot_row_pinned_rank16(neon, b_row, r0, r1, r2, r3);
            let updated = dot.mul_add(scale, *slot);
            if !updated.is_finite() {
                return false;
            }
            *slot = updated;
        }
        return true;
    }

    for (b_quad, out4) in b_quads.chunks_exact(4 * rank).zip(out_quads.iter_mut()) {
        let (r0, rest) = b_quad.split_at(rank);
        let (r1, rest) = rest.split_at(rank);
        let (r2, r3) = rest.split_at(rank);
        let dots = dot_four_rows_general(neon, [r0, r1, r2, r3], ranks);
        for (dot, slot) in dots.into_iter().zip(out4.iter_mut()) {
            let updated = dot.mul_add(scale, *slot);
            if !updated.is_finite() {
                return false;
            }
            *slot = updated;
        }
    }
    for (b_row, slot) in b_tail.chunks_exact(rank).zip(out_tail.iter_mut()) {
        let dot = dot_row_general(neon, b_row, ranks);
        let updated = dot.mul_add(scale, *slot);
        if !updated.is_finite() {
            return false;
        }
        *slot = updated;
    }
    true
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
fn lora_multi_b_inplace_two_rank16<W: FastLoraWeight>(
    neon: Neon,
    b_weights: &[W],
    ranks0: &[f32],
    ranks1: &[f32],
    scale: f32,
    out0: &mut [f32],
    out1: &mut [f32],
) -> bool {
    let r00 = f32x4::from_slice(neon, &ranks0[0..4]);
    let r01 = f32x4::from_slice(neon, &ranks0[4..8]);
    let r02 = f32x4::from_slice(neon, &ranks0[8..12]);
    let r03 = f32x4::from_slice(neon, &ranks0[12..16]);
    let r10 = f32x4::from_slice(neon, &ranks1[0..4]);
    let r11 = f32x4::from_slice(neon, &ranks1[4..8]);
    let r12 = f32x4::from_slice(neon, &ranks1[8..12]);
    let r13 = f32x4::from_slice(neon, &ranks1[12..16]);
    let zero = f32x4::splat(neon, 0.0);

    for (b_row, (slot0, slot1)) in b_weights
        .as_chunks::<16>()
        .0
        .iter()
        .zip(out0.iter_mut().zip(out1.iter_mut()))
    {
        let chunks = b_row.as_chunks::<8>().0;
        let (w0, w1) = W::load_x8(neon, &chunks[0]);
        let (w2, w3) = W::load_x8(neon, &chunks[1]);
        let acc00 = w2.mul_add_precise(r02, w0.mul_add_precise(r00, zero));
        let acc01 = w3.mul_add_precise(r03, w1.mul_add_precise(r01, zero));
        let acc10 = w2.mul_add_precise(r12, w0.mul_add_precise(r10, zero));
        let acc11 = w3.mul_add_precise(r13, w1.mul_add_precise(r11, zero));
        let dot0 = reduce_acc_pair(acc00, acc01);
        let dot1 = reduce_acc_pair(acc10, acc11);
        let u0 = dot0.mul_add(scale, *slot0);
        let u1 = dot1.mul_add(scale, *slot1);
        if !u0.is_finite() || !u1.is_finite() {
            return false;
        }
        *slot0 = u0;
        *slot1 = u1;
    }
    true
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
fn lora_multi_b_inplace_two_general<W: FastLoraWeight>(
    neon: Neon,
    b_weights: &[W],
    ranks0: &[f32],
    ranks1: &[f32],
    scale: f32,
    out0: &mut [f32],
    out1: &mut [f32],
) -> bool {
    let rank = ranks0.len();
    let (x0_chunks8, x0_rem8) = ranks0.as_chunks::<8>();
    let (x1_chunks8, x1_rem8) = ranks1.as_chunks::<8>();
    let (x0_chunks4, x0_tail) = x0_rem8.as_chunks::<4>();
    let (x1_chunks4, x1_tail) = x1_rem8.as_chunks::<4>();
    for (b_row, (slot0, slot1)) in b_weights
        .chunks_exact(rank)
        .zip(out0.iter_mut().zip(out1.iter_mut()))
    {
        let (w_chunks8, w_rem8) = b_row.as_chunks::<8>();
        let mut acc00 = f32x4::splat(neon, 0.0);
        let mut acc01 = f32x4::splat(neon, 0.0);
        let mut acc10 = f32x4::splat(neon, 0.0);
        let mut acc11 = f32x4::splat(neon, 0.0);
        for ((w8, x0_8), x1_8) in w_chunks8.iter().zip(x0_chunks8).zip(x1_chunks8) {
            let (w_lo, w_hi) = W::load_x8(neon, w8);
            acc00 = w_lo.mul_add_precise(f32x4::from_slice(neon, &x0_8[..4]), acc00);
            acc01 = w_hi.mul_add_precise(f32x4::from_slice(neon, &x0_8[4..]), acc01);
            acc10 = w_lo.mul_add_precise(f32x4::from_slice(neon, &x1_8[..4]), acc10);
            acc11 = w_hi.mul_add_precise(f32x4::from_slice(neon, &x1_8[4..]), acc11);
        }
        let (w_chunks4, w_tail) = w_rem8.as_chunks::<4>();
        if let (Some(w4), Some(x0_4), Some(x1_4)) =
            (w_chunks4.first(), x0_chunks4.first(), x1_chunks4.first())
        {
            let w = W::load_x4(neon, w4);
            acc00 = w.mul_add_precise(f32x4::from_slice(neon, x0_4), acc00);
            acc10 = w.mul_add_precise(f32x4::from_slice(neon, x1_4), acc10);
        }
        let mut d0 = reduce_acc_pair(acc00, acc01);
        let mut d1 = reduce_acc_pair(acc10, acc11);
        for ((&w, &x0), &x1) in w_tail.iter().zip(x0_tail).zip(x1_tail) {
            let wf = w.to_f32();
            d0 = wf.mul_add(x0, d0);
            d1 = wf.mul_add(x1, d1);
        }
        let u0 = d0.mul_add(scale, *slot0);
        let u1 = d1.mul_add(scale, *slot1);
        if !u0.is_finite() || !u1.is_finite() {
            return false;
        }
        *slot0 = u0;
        *slot1 = u1;
    }
    true
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
fn lora_multi_b_tokens_in_place_generic<W: FastLoraWeight>(
    b_weights: &[W],
    ranks: &[f32],
    rank: usize,
    scale: f32,
    outputs: &mut [Vec<f32>],
) -> Option<bool> {
    let neon = fearless_simd::Level::baseline().as_neon()?;
    let n_tokens = outputs.len();
    if rank == 0 || n_tokens == 0 || ranks.len() != n_tokens.saturating_mul(rank) {
        return Some(false);
    }
    let out_features = outputs[0].len();
    if b_weights.len() != out_features.saturating_mul(rank)
        || outputs.iter().any(|out| out.len() != out_features)
    {
        return Some(false);
    }
    let (tok_pairs, tok_rem) = outputs.as_chunks_mut::<2>();
    for (pair_idx, [out0, out1]) in tok_pairs.iter_mut().enumerate() {
        let base_t = pair_idx * 2;
        let ranks0 = &ranks[base_t * rank..(base_t + 1) * rank];
        let ranks1 = &ranks[(base_t + 1) * rank..(base_t + 2) * rank];
        let ok = if rank == 16 {
            lora_multi_b_inplace_two_rank16(neon, b_weights, ranks0, ranks1, scale, out0, out1)
        } else {
            lora_multi_b_inplace_two_general(neon, b_weights, ranks0, ranks1, scale, out0, out1)
        };
        if !ok {
            return Some(false);
        }
    }
    if let [last_out] = tok_rem {
        let last_ranks = &ranks[tok_pairs.len() * 2 * rank..];
        if !lora_multi_b_inplace_one(neon, b_weights, last_ranks, scale, last_out) {
            return Some(false);
        }
    }
    Some(true)
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
pub(super) fn lora_multi_b_f16(
    b_weights: &[u16],
    ranks: &[f32],
    scale: f32,
    base_output: &[f32],
    correction: &mut [f32],
) -> Option<bool> {
    lora_multi_b_generic(b_weights, ranks, scale, base_output, correction)
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
pub(super) fn lora_multi_b_f32(
    b_weights: &[f32],
    ranks: &[f32],
    scale: f32,
    base_output: &[f32],
    correction: &mut [f32],
) -> Option<bool> {
    lora_multi_b_generic(b_weights, ranks, scale, base_output, correction)
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
pub(super) fn lora_multi_b_tokens_in_place_f16(
    b_weights: &[u16],
    ranks: &[f32],
    rank: usize,
    scale: f32,
    outputs: &mut [Vec<f32>],
) -> Option<bool> {
    lora_multi_b_tokens_in_place_generic(b_weights, ranks, rank, scale, outputs)
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
pub(super) fn lora_multi_b_tokens_in_place_f32(
    b_weights: &[f32],
    ranks: &[f32],
    rank: usize,
    scale: f32,
    outputs: &mut [Vec<f32>],
) -> Option<bool> {
    lora_multi_b_tokens_in_place_generic(b_weights, ranks, rank, scale, outputs)
}
