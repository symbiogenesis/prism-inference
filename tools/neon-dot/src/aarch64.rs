#![allow(
    clippy::redundant_pub_crate,
    reason = "Explicit parent-only visibility keeps the ARM64 intrinsics inside the audited module boundary."
)]

use core::arch::aarch64::{
    int8x16_t, int32x4_t, uint8x8_t, uint8x16_t, vaddq_s32, vaddvq_s32, vandq_u8, vcgeq_u8,
    vcltq_u8, vcombine_s8, vcombine_s32, vcombine_u8, vdotq_s32, vdupq_n_s32, vdupq_n_u8,
    vget_high_s8, vget_high_s32, vget_low_s8, vget_low_s32, vld1_u8, vld1q_s8, vld1q_s32, vld1q_u8,
    vmulq_u8, vpaddlq_s8, vpaddlq_s16, vreinterpretq_s8_u8, vshrq_n_u8, vst1q_s8, vst1q_s32,
    vsubq_s8, vsubq_s32, vuzp1q_s8, vuzp1q_s32, vuzp2q_s8, vuzp2q_s32,
};
use std::sync::atomic::{AtomicU8, Ordering};

const UNINIT: u8 = 0;
const ABSENT: u8 = 1;
const PRESENT: u8 = 2;

static DOTPROD_STATE: AtomicU8 = AtomicU8::new(UNINIT);
static I8MM_STATE: AtomicU8 = AtomicU8::new(UNINIT);

#[inline]
pub(super) fn has_dotprod() -> bool {
    if cfg!(target_feature = "dotprod") {
        return true;
    }
    match DOTPROD_STATE.load(Ordering::Relaxed) {
        PRESENT => true,
        ABSENT => false,
        _ => {
            let detected = std::arch::is_aarch64_feature_detected!("dotprod");
            DOTPROD_STATE.store(if detected { PRESENT } else { ABSENT }, Ordering::Relaxed);
            detected
        }
    }
}

#[inline]
pub(super) fn has_i8mm() -> bool {
    if cfg!(target_feature = "i8mm") {
        return true;
    }
    match I8MM_STATE.load(Ordering::Relaxed) {
        PRESENT => true,
        ABSENT => false,
        _ => {
            let detected = std::arch::is_aarch64_feature_detected!("i8mm");
            I8MM_STATE.store(if detected { PRESENT } else { ABSENT }, Ordering::Relaxed);
            detected
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_u8x16(bytes: &[u8; 16]) -> uint8x16_t {
    // SAFETY: `bytes` is a valid 16-byte array reference and `vld1q_u8` requires no alignment.
    unsafe { vld1q_u8(bytes.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_u8x8(bytes: &[u8; 8]) -> uint8x8_t {
    // SAFETY: `bytes` is a valid 8-byte array reference and `vld1_u8` requires no alignment.
    unsafe { vld1_u8(bytes.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_s8x16(bytes: &[i8; 16]) -> int8x16_t {
    // SAFETY: `bytes` is a valid 16-byte `i8` array reference and `vld1q_s8` requires no alignment.
    unsafe { vld1q_s8(bytes.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_s8x16(vec: int8x16_t, out: &mut [i8; 16]) {
    // SAFETY: `out` is a valid mutable 16-byte `i8` array reference and `vst1q_s8` requires no alignment.
    unsafe { vst1q_s8(out.as_mut_ptr(), vec) };
}

#[inline]
#[target_feature(enable = "neon")]
fn load_s32x4(values: &[i32; 4]) -> int32x4_t {
    // SAFETY: `values` is a valid 4-element `i32` array reference and `vld1q_s32` requires no alignment.
    unsafe { vld1q_s32(values.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_s32x4(vec: int32x4_t, out: &mut [i32; 4]) {
    // SAFETY: `out` is a valid mutable 4-element `i32` array reference and `vst1q_s32` requires no alignment.
    unsafe { vst1q_s32(out.as_mut_ptr(), vec) };
}

/// Extracts signed trits (`-1..=1`) from unsigned fractional base-3 residues `v`:
/// `((u16::from(v) * 3) >> 8) as i8 - 1` equals `-1` for `v < 86`, `0` for `86..171`,
/// and `+1` for `v >= 171`. Since NEON `vcltq_u8` / `vcgeq_u8` yield `0xFF` (`-1_i8`)
/// when true and `0x00` (`0_i8`) when false, `lt_86 - ge_171` computes all 16 signed
/// trits in 3 instructions without widening.
#[inline]
#[target_feature(enable = "neon")]
fn extract_trit_s8x16(v: uint8x16_t) -> int8x16_t {
    let lt_86 = vreinterpretq_s8_u8(vcltq_u8(v, vdupq_n_u8(86)));
    let ge_171 = vreinterpretq_s8_u8(vcgeq_u8(v, vdupq_n_u8(171)));
    vsubq_s8(lt_86, ge_171)
}

#[inline]
#[target_feature(enable = "neon")]
fn unpack_ptq1_vectors(packed: &[u8; 28]) -> [int8x16_t; 8] {
    const POW3_MID_01: [u8; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3];
    const POW3_MID_23: [u8; 16] = [9, 9, 9, 9, 9, 9, 9, 9, 27, 27, 27, 27, 27, 27, 27, 27];
    const POW3_TAIL: [u8; 16] = [81, 81, 81, 81, 81, 81, 81, 81, 1, 1, 3, 3, 9, 9, 27, 27];

    let first_bytes: [u8; 16] = [
        packed[0], packed[1], packed[2], packed[3], packed[4], packed[5], packed[6], packed[7],
        packed[8], packed[9], packed[10], packed[11], packed[12], packed[13], packed[14],
        packed[15],
    ];
    let q0 = load_u8x16(&first_bytes);
    let t0 = extract_trit_s8x16(q0);
    let t1 = extract_trit_s8x16(vmulq_u8(q0, vdupq_n_u8(3)));
    let t2 = extract_trit_s8x16(vmulq_u8(q0, vdupq_n_u8(9)));
    let t3 = extract_trit_s8x16(vmulq_u8(q0, vdupq_n_u8(27)));
    let t4 = extract_trit_s8x16(vmulq_u8(q0, vdupq_n_u8(81)));

    let mid_half: [u8; 8] = [
        packed[16], packed[17], packed[18], packed[19], packed[20], packed[21], packed[22],
        packed[23],
    ];
    let q_mid_half = load_u8x8(&mid_half);
    let q_mid = vcombine_u8(q_mid_half, q_mid_half);
    let t5 = extract_trit_s8x16(vmulq_u8(q_mid, load_u8x16(&POW3_MID_01)));
    let t6 = extract_trit_s8x16(vmulq_u8(q_mid, load_u8x16(&POW3_MID_23)));

    let qh0 = packed[24];
    let qh1 = packed[25];
    let tail_half: [u8; 8] = [qh0, qh1, qh0, qh1, qh0, qh1, qh0, qh1];
    let q_tail = vcombine_u8(q_mid_half, load_u8x8(&tail_half));
    let t7 = extract_trit_s8x16(vmulq_u8(q_tail, load_u8x16(&POW3_TAIL)));

    [t0, t1, t2, t3, t4, t5, t6, t7]
}

#[target_feature(enable = "neon")]
fn decode_ptq1_trits_neon(packed: &[u8; 28]) -> [[i8; 16]; 8] {
    let vectors = unpack_ptq1_vectors(packed);
    let mut out = [[0_i8; 16]; 8];
    for (vec, dst) in vectors.into_iter().zip(&mut out) {
        store_s8x16(vec, dst);
    }
    out
}

#[inline]
pub(super) fn decode_ptq1_trits_i8(packed: &[u8; 28]) -> [[i8; 16]; 8] {
    // SAFETY: Baseline NEON is mandatory on all 64-bit ARM (`aarch64`) targets.
    unsafe { decode_ptq1_trits_neon(packed) }
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
fn sdot32_vec(w0: int8x16_t, w1: int8x16_t, a0: int8x16_t, a1: int8x16_t) -> int32x4_t {
    let acc0 = vdotq_s32(vdupq_n_s32(0), w0, a0);
    vdotq_s32(acc0, w1, a1)
}

#[target_feature(enable = "neon,dotprod")]
fn dot32_s8_lanes_dotprod(w0: &[i8; 16], w1: &[i8; 16], a0: &[i8; 16], a1: &[i8; 16]) -> [i32; 4] {
    let acc = sdot32_vec(
        load_s8x16(w0),
        load_s8x16(w1),
        load_s8x16(a0),
        load_s8x16(a1),
    );
    let mut out = [0_i32; 4];
    store_s32x4(acc, &mut out);
    out
}

#[target_feature(enable = "neon,dotprod")]
fn dot32_s8_sum_dotprod(w0: &[i8; 16], w1: &[i8; 16], a0: &[i8; 16], a1: &[i8; 16]) -> i32 {
    let acc = sdot32_vec(
        load_s8x16(w0),
        load_s8x16(w1),
        load_s8x16(a0),
        load_s8x16(a1),
    );
    vaddvq_s32(acc)
}

#[inline]
pub(super) fn dot32_s8_lanes(
    w0: &[i8; 16],
    w1: &[i8; 16],
    a0: &[i8; 16],
    a1: &[i8; 16],
) -> Option<[i32; 4]> {
    if !has_dotprod() {
        return None;
    }
    // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
    Some(unsafe { dot32_s8_lanes_dotprod(w0, w1, a0, a1) })
}

#[inline]
pub(super) fn dot32_s8_sum(
    w0: &[i8; 16],
    w1: &[i8; 16],
    a0: &[i8; 16],
    a1: &[i8; 16],
) -> Option<i32> {
    if !has_dotprod() {
        return None;
    }
    // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
    Some(unsafe { dot32_s8_sum_dotprod(w0, w1, a0, a1) })
}

#[target_feature(enable = "neon,dotprod")]
fn dot_ptq1_block_lanes_dotprod(packed: &[u8; 28], activations: [&[i8; 32]; 4]) -> [[i32; 4]; 4] {
    let [t0, t1, t2, t3, t4, t5, t6, t7] = unpack_ptq1_vectors(packed);
    let pairs = [(t0, t1), (t2, t3), (t4, t5), (t6, t7)];
    let mut out = [[0_i32; 4]; 4];
    for (group, ((w0, w1), act)) in pairs.into_iter().zip(activations).enumerate() {
        let a_halves = act.as_chunks::<16>().0;
        let acc = sdot32_vec(w0, w1, load_s8x16(&a_halves[0]), load_s8x16(&a_halves[1]));
        store_s32x4(acc, &mut out[group]);
    }
    out
}

#[target_feature(enable = "neon,dotprod")]
fn dot_ptq1_block_groups_dotprod(packed: &[u8; 28], activations: [&[i8; 32]; 4]) -> [i32; 4] {
    let [t0, t1, t2, t3, t4, t5, t6, t7] = unpack_ptq1_vectors(packed);
    let pairs = [(t0, t1), (t2, t3), (t4, t5), (t6, t7)];
    let mut out = [0_i32; 4];
    for (group, ((w0, w1), act)) in pairs.into_iter().zip(activations).enumerate() {
        let a_halves = act.as_chunks::<16>().0;
        let acc = sdot32_vec(w0, w1, load_s8x16(&a_halves[0]), load_s8x16(&a_halves[1]));
        out[group] = vaddvq_s32(acc);
    }
    out
}

#[inline]
pub(super) fn dot_ptq1_block_lanes(
    packed: &[u8; 28],
    activations: [&[i8; 32]; 4],
) -> Option<[[i32; 4]; 4]> {
    if !has_dotprod() {
        return None;
    }
    // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
    Some(unsafe { dot_ptq1_block_lanes_dotprod(packed, activations) })
}

#[inline]
pub(super) fn dot_ptq1_block_groups(
    packed: &[u8; 28],
    activations: [&[i8; 32]; 4],
) -> Option<[i32; 4]> {
    if !has_dotprod() {
        return None;
    }
    // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
    Some(unsafe { dot_ptq1_block_groups_dotprod(packed, activations) })
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
fn pq2_half_unsigned_sdot(
    acc: int32x4_t,
    w: uint8x16_t,
    p0: int8x16_t,
    p1: int8x16_t,
    p2: int8x16_t,
    p3: int8x16_t,
) -> int32x4_t {
    let mask = vdupq_n_u8(3);
    let w0 = vreinterpretq_s8_u8(vandq_u8(w, mask));
    let w1 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<2>(w), mask));
    let w2 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<4>(w), mask));
    let w3 = vreinterpretq_s8_u8(vshrq_n_u8::<6>(w));
    let acc = vdotq_s32(acc, w0, p0);
    let acc = vdotq_s32(acc, w1, p1);
    let acc = vdotq_s32(acc, w2, p2);
    vdotq_s32(acc, w3, p3)
}

#[target_feature(enable = "neon,dotprod")]
fn dot_pq2_consumption_dotprod(packed: &[u8; 32], planes: &[i8; 128], activation_sum: i32) -> i32 {
    let mut acc = vdupq_n_s32(0);
    for (bytes, half_planes) in packed
        .as_chunks::<16>()
        .0
        .iter()
        .zip(planes.as_chunks::<64>().0)
    {
        let w = load_u8x16(bytes);
        let p = half_planes.as_chunks::<16>().0;
        acc = pq2_half_unsigned_sdot(
            acc,
            w,
            load_s8x16(&p[0]),
            load_s8x16(&p[1]),
            load_s8x16(&p[2]),
            load_s8x16(&p[3]),
        );
    }
    vaddvq_s32(acc) - activation_sum
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
fn reduce4_s32x4(v0: int32x4_t, v1: int32x4_t, v2: int32x4_t, v3: int32x4_t) -> int32x4_t {
    let p01 = vaddq_s32(vuzp1q_s32(v0, v1), vuzp2q_s32(v0, v1));
    let p23 = vaddq_s32(vuzp1q_s32(v2, v3), vuzp2q_s32(v2, v3));
    vaddq_s32(vuzp1q_s32(p01, p23), vuzp2q_s32(p01, p23))
}

#[target_feature(enable = "neon,dotprod")]
fn dot_pq2_four_consumption_dotprod(
    packed: [&[u8; 32]; 4],
    planes: &[i8; 128],
    activation_sum: i32,
) -> [i32; 4] {
    let mut acc = [vdupq_n_s32(0); 4];
    for (half, half_planes) in planes.as_chunks::<64>().0.iter().enumerate() {
        let p = half_planes.as_chunks::<16>().0;
        let p0 = load_s8x16(&p[0]);
        let p1 = load_s8x16(&p[1]);
        let p2 = load_s8x16(&p[2]);
        let p3 = load_s8x16(&p[3]);
        for row in 0..4 {
            let bytes = &packed[row].as_chunks::<16>().0[half];
            acc[row] = pq2_half_unsigned_sdot(acc[row], load_u8x16(bytes), p0, p1, p2, p3);
        }
    }
    let reduced = reduce4_s32x4(acc[0], acc[1], acc[2], acc[3]);
    let mut out = [0_i32; 4];
    store_s32x4(reduced, &mut out);
    out.map(|v| v - activation_sum)
}

#[target_feature(enable = "neon")]
fn sum_s8_128_neon(planes: &[i8; 128]) -> i32 {
    let mut acc = vdupq_n_s32(0);
    for chunk in planes.as_chunks::<16>().0 {
        let v = load_s8x16(chunk);
        acc = vaddq_s32(acc, vpaddlq_s16(vpaddlq_s8(v)));
    }
    vaddvq_s32(acc)
}

#[inline]
pub(super) fn dot_pq2_consumption_block(
    packed: &[u8; 32],
    planes: &[i8; 128],
    activation_sum: i32,
) -> Option<i32> {
    if !has_dotprod() {
        return None;
    }
    // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
    Some(unsafe { dot_pq2_consumption_dotprod(packed, planes, activation_sum) })
}

#[inline]
pub(super) fn dot_pq2_four_consumption_precomputed(
    packed: [&[u8; 32]; 4],
    planes: &[i8; 128],
    activation_sum: i32,
) -> Option<[i32; 4]> {
    if !has_dotprod() {
        return None;
    }
    // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
    Some(unsafe { dot_pq2_four_consumption_dotprod(packed, planes, activation_sum) })
}

#[inline]
pub(super) fn dot_pq2_four_consumption(
    packed: [&[u8; 32]; 4],
    planes: &[i8; 128],
) -> Option<[i32; 4]> {
    if !has_dotprod() {
        return None;
    }
    // SAFETY: Baseline NEON is mandatory on `aarch64`.
    let activation_sum = unsafe { sum_s8_128_neon(planes) };
    // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
    Some(unsafe { dot_pq2_four_consumption_dotprod(packed, planes, activation_sum) })
}

#[inline]
#[target_feature(enable = "neon,i8mm")]
fn smmla_s32(mut acc: int32x4_t, w: int8x16_t, a: int8x16_t) -> int32x4_t {
    // SAFETY: The caller ensures `i8mm` is supported; `smmla` is a pure register-to-register instruction.
    unsafe {
        core::arch::asm!(
            "smmla {acc:v}.4s, {w:v}.16b, {a:v}.16b",
            acc = inout(vreg) acc,
            w = in(vreg) w,
            a = in(vreg) a,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    acc
}

#[target_feature(enable = "neon,i8mm")]
fn smmla_dot32_2x2_i8mm(
    r0: &[i8; 32],
    r1: &[i8; 32],
    c0: &[i8; 32],
    c1: &[i8; 32],
) -> [[i32; 2]; 2] {
    let r0_halves = r0.as_chunks::<16>().0;
    let r1_halves = r1.as_chunks::<16>().0;
    let c0_halves = c0.as_chunks::<16>().0;
    let c1_halves = c1.as_chunks::<16>().0;

    let mut acc = vdupq_n_s32(0);
    for half in 0..2 {
        let w0 = load_s8x16(&r0_halves[half]);
        let w1 = load_s8x16(&r1_halves[half]);
        let a0 = load_s8x16(&c0_halves[half]);
        let a1 = load_s8x16(&c1_halves[half]);

        let w_lo = vcombine_s8(vget_low_s8(w0), vget_low_s8(w1));
        let a_lo = vcombine_s8(vget_low_s8(a0), vget_low_s8(a1));
        acc = smmla_s32(acc, w_lo, a_lo);

        let w_hi = vcombine_s8(vget_high_s8(w0), vget_high_s8(w1));
        let a_hi = vcombine_s8(vget_high_s8(a0), vget_high_s8(a1));
        acc = smmla_s32(acc, w_hi, a_hi);
    }
    let mut flat = [0_i32; 4];
    store_s32x4(acc, &mut flat);
    [[flat[0], flat[1]], [flat[2], flat[3]]]
}

#[target_feature(enable = "neon,dotprod")]
fn dot_q8_0_block_dotprod(weights: &[u8; 32], activations: &[i8; 32]) -> i32 {
    let w_halves = weights.as_chunks::<16>().0;
    let a_halves = activations.as_chunks::<16>().0;
    let w0 = vreinterpretq_s8_u8(load_u8x16(&w_halves[0]));
    let w1 = vreinterpretq_s8_u8(load_u8x16(&w_halves[1]));
    let acc = sdot32_vec(w0, w1, load_s8x16(&a_halves[0]), load_s8x16(&a_halves[1]));
    vaddvq_s32(acc)
}

#[inline]
pub(super) fn dot_q8_0_block(weights: &[u8; 32], activations: &[i8; 32]) -> Option<i32> {
    if !has_dotprod() {
        return None;
    }
    // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
    Some(unsafe { dot_q8_0_block_dotprod(weights, activations) })
}

#[target_feature(enable = "neon,i8mm")]
fn smmla_q8_0_2x2_i8mm(
    r0: &[u8; 32],
    r1: &[u8; 32],
    c0: &[i8; 32],
    c1: &[i8; 32],
) -> [[i32; 2]; 2] {
    let r0_halves = r0.as_chunks::<16>().0;
    let r1_halves = r1.as_chunks::<16>().0;
    let c0_halves = c0.as_chunks::<16>().0;
    let c1_halves = c1.as_chunks::<16>().0;

    let mut acc = vdupq_n_s32(0);
    for half in 0..2 {
        let w0 = vreinterpretq_s8_u8(load_u8x16(&r0_halves[half]));
        let w1 = vreinterpretq_s8_u8(load_u8x16(&r1_halves[half]));
        let a0 = load_s8x16(&c0_halves[half]);
        let a1 = load_s8x16(&c1_halves[half]);

        let w_lo = vcombine_s8(vget_low_s8(w0), vget_low_s8(w1));
        let a_lo = vcombine_s8(vget_low_s8(a0), vget_low_s8(a1));
        acc = smmla_s32(acc, w_lo, a_lo);

        let w_hi = vcombine_s8(vget_high_s8(w0), vget_high_s8(w1));
        let a_hi = vcombine_s8(vget_high_s8(a0), vget_high_s8(a1));
        acc = smmla_s32(acc, w_hi, a_hi);
    }
    let mut flat = [0_i32; 4];
    store_s32x4(acc, &mut flat);
    [[flat[0], flat[1]], [flat[2], flat[3]]]
}

#[inline]
pub(super) fn smmla_q8_0_2x2(
    r0: &[u8; 32],
    r1: &[u8; 32],
    c0: &[i8; 32],
    c1: &[i8; 32],
) -> Option<[[i32; 2]; 2]> {
    if !has_i8mm() {
        return None;
    }
    // SAFETY: `has_i8mm()` verified that the CPU supports `i8mm` and baseline `neon`.
    Some(unsafe { smmla_q8_0_2x2_i8mm(r0, r1, c0, c1) })
}

#[inline]
pub(super) fn smmla_dot32_2x2(
    r0: &[i8; 32],
    r1: &[i8; 32],
    c0: &[i8; 32],
    c1: &[i8; 32],
) -> Option<[[i32; 2]; 2]> {
    if !has_i8mm() {
        return None;
    }
    // SAFETY: `has_i8mm()` verified that the CPU supports `i8mm` and baseline `neon`.
    Some(unsafe { smmla_dot32_2x2_i8mm(r0, r1, c0, c1) })
}

#[inline]
#[target_feature(enable = "neon")]
fn unpack_pq2_unsigned_s8(packed: &[u8; 32]) -> [int8x16_t; 8] {
    let halves = packed.as_chunks::<16>().0;
    let mask = vdupq_n_u8(3);
    let w_lo = load_u8x16(&halves[0]);
    let w_hi = load_u8x16(&halves[1]);
    [
        vreinterpretq_s8_u8(vandq_u8(w_lo, mask)),
        vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<2>(w_lo), mask)),
        vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<4>(w_lo), mask)),
        vreinterpretq_s8_u8(vshrq_n_u8::<6>(w_lo)),
        vreinterpretq_s8_u8(vandq_u8(w_hi, mask)),
        vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<2>(w_hi), mask)),
        vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<4>(w_hi), mask)),
        vreinterpretq_s8_u8(vshrq_n_u8::<6>(w_hi)),
    ]
}

#[inline]
#[target_feature(enable = "neon")]
fn deinterleave_pq2_planes_s8(values: &[i8; 128]) -> [int8x16_t; 8] {
    let chunks = values.as_chunks::<16>().0;
    let c0 = load_s8x16(&chunks[0]);
    let c1 = load_s8x16(&chunks[1]);
    let c2 = load_s8x16(&chunks[2]);
    let c3 = load_s8x16(&chunks[3]);
    let e01 = vuzp1q_s8(c0, c1);
    let o01 = vuzp2q_s8(c0, c1);
    let e23 = vuzp1q_s8(c2, c3);
    let o23 = vuzp2q_s8(c2, c3);

    let c4 = load_s8x16(&chunks[4]);
    let c5 = load_s8x16(&chunks[5]);
    let c6 = load_s8x16(&chunks[6]);
    let c7 = load_s8x16(&chunks[7]);
    let e45 = vuzp1q_s8(c4, c5);
    let o45 = vuzp2q_s8(c4, c5);
    let e67 = vuzp1q_s8(c6, c7);
    let o67 = vuzp2q_s8(c6, c7);

    [
        vuzp1q_s8(e01, e23),
        vuzp1q_s8(o01, o23),
        vuzp2q_s8(e01, e23),
        vuzp2q_s8(o01, o23),
        vuzp1q_s8(e45, e67),
        vuzp1q_s8(o45, o67),
        vuzp2q_s8(e45, e67),
        vuzp2q_s8(o45, o67),
    ]
}

#[target_feature(enable = "neon,dotprod")]
fn dot_pq2_four_rows_one_token_dotprod(
    packed: [&[u8; 32]; 4],
    activation: &[i8; 128],
    activation_sum: i32,
) -> [i32; 4] {
    let planes = deinterleave_pq2_planes_s8(activation);
    let mut acc = [vdupq_n_s32(0); 4];
    for (row_idx, row_bytes) in packed.into_iter().enumerate() {
        let weights = unpack_pq2_unsigned_s8(row_bytes);
        for (w, p) in weights.into_iter().zip(planes) {
            acc[row_idx] = vdotq_s32(acc[row_idx], w, p);
        }
    }
    let reduced = vsubq_s32(
        reduce4_s32x4(acc[0], acc[1], acc[2], acc[3]),
        vdupq_n_s32(activation_sum),
    );
    let mut out = [0_i32; 4];
    store_s32x4(reduced, &mut out);
    out
}

#[inline]
#[target_feature(enable = "neon,i8mm")]
fn dot_pq2_4x4_tile_i8mm(
    weights: &[[int8x16_t; 8]; 4],
    acts: &[[int8x16_t; 8]; 4],
    sums_vec: int32x4_t,
) -> [[i32; 4]; 4] {
    let mut acc00 = vdupq_n_s32(0);
    let mut acc01 = vdupq_n_s32(0);
    let mut acc10 = vdupq_n_s32(0);
    let mut acc11 = vdupq_n_s32(0);
    for k in 0..8 {
        let w0 = weights[0][k];
        let w1 = weights[1][k];
        let w2 = weights[2][k];
        let w3 = weights[3][k];
        let a0 = acts[0][k];
        let a1 = acts[1][k];
        let a2 = acts[2][k];
        let a3 = acts[3][k];

        let w01_lo = vcombine_s8(vget_low_s8(w0), vget_low_s8(w1));
        let w01_hi = vcombine_s8(vget_high_s8(w0), vget_high_s8(w1));
        let w23_lo = vcombine_s8(vget_low_s8(w2), vget_low_s8(w3));
        let w23_hi = vcombine_s8(vget_high_s8(w2), vget_high_s8(w3));

        let a01_lo = vcombine_s8(vget_low_s8(a0), vget_low_s8(a1));
        let a01_hi = vcombine_s8(vget_high_s8(a0), vget_high_s8(a1));
        let a23_lo = vcombine_s8(vget_low_s8(a2), vget_low_s8(a3));
        let a23_hi = vcombine_s8(vget_high_s8(a2), vget_high_s8(a3));

        acc00 = smmla_s32(smmla_s32(acc00, w01_lo, a01_lo), w01_hi, a01_hi);
        acc01 = smmla_s32(smmla_s32(acc01, w01_lo, a23_lo), w01_hi, a23_hi);
        acc10 = smmla_s32(smmla_s32(acc10, w23_lo, a01_lo), w23_hi, a01_hi);
        acc11 = smmla_s32(smmla_s32(acc11, w23_lo, a23_lo), w23_hi, a23_hi);
    }
    let r0 = vsubq_s32(
        vcombine_s32(vget_low_s32(acc00), vget_low_s32(acc01)),
        sums_vec,
    );
    let r1 = vsubq_s32(
        vcombine_s32(vget_high_s32(acc00), vget_high_s32(acc01)),
        sums_vec,
    );
    let r2 = vsubq_s32(
        vcombine_s32(vget_low_s32(acc10), vget_low_s32(acc11)),
        sums_vec,
    );
    let r3 = vsubq_s32(
        vcombine_s32(vget_high_s32(acc10), vget_high_s32(acc11)),
        sums_vec,
    );
    let mut out = [[0_i32; 4]; 4];
    store_s32x4(r0, &mut out[0]);
    store_s32x4(r1, &mut out[1]);
    store_s32x4(r2, &mut out[2]);
    store_s32x4(r3, &mut out[3]);
    out
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
fn dot_pq2_4x4_tile_dotprod(
    weights: &[[int8x16_t; 8]; 4],
    acts: &[[int8x16_t; 8]; 4],
    sums_vec: int32x4_t,
) -> [[i32; 4]; 4] {
    let mut acc = [[vdupq_n_s32(0); 4]; 4];
    for k in 0..8 {
        let a0 = acts[0][k];
        let a1 = acts[1][k];
        let a2 = acts[2][k];
        let a3 = acts[3][k];
        for r in 0..4 {
            let w = weights[r][k];
            acc[r][0] = vdotq_s32(acc[r][0], w, a0);
            acc[r][1] = vdotq_s32(acc[r][1], w, a1);
            acc[r][2] = vdotq_s32(acc[r][2], w, a2);
            acc[r][3] = vdotq_s32(acc[r][3], w, a3);
        }
    }
    let mut out = [[0_i32; 4]; 4];
    for r in 0..4 {
        let reduced = vsubq_s32(
            reduce4_s32x4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]),
            sums_vec,
        );
        store_s32x4(reduced, &mut out[r]);
    }
    out
}

#[target_feature(enable = "neon,i8mm")]
fn dot_pq2_four_rows_four_tokens_i8mm(
    packed: [&[u8; 32]; 4],
    activations: [&[i8; 128]; 4],
    activation_sums: [i32; 4],
) -> [[i32; 4]; 4] {
    let weights = [
        unpack_pq2_unsigned_s8(packed[0]),
        unpack_pq2_unsigned_s8(packed[1]),
        unpack_pq2_unsigned_s8(packed[2]),
        unpack_pq2_unsigned_s8(packed[3]),
    ];
    let acts = [
        deinterleave_pq2_planes_s8(activations[0]),
        deinterleave_pq2_planes_s8(activations[1]),
        deinterleave_pq2_planes_s8(activations[2]),
        deinterleave_pq2_planes_s8(activations[3]),
    ];
    dot_pq2_4x4_tile_i8mm(&weights, &acts, load_s32x4(&activation_sums))
}

#[target_feature(enable = "neon,dotprod")]
fn dot_pq2_four_rows_four_tokens_dotprod(
    packed: [&[u8; 32]; 4],
    activations: [&[i8; 128]; 4],
    activation_sums: [i32; 4],
) -> [[i32; 4]; 4] {
    let weights = [
        unpack_pq2_unsigned_s8(packed[0]),
        unpack_pq2_unsigned_s8(packed[1]),
        unpack_pq2_unsigned_s8(packed[2]),
        unpack_pq2_unsigned_s8(packed[3]),
    ];
    let acts = [
        deinterleave_pq2_planes_s8(activations[0]),
        deinterleave_pq2_planes_s8(activations[1]),
        deinterleave_pq2_planes_s8(activations[2]),
        deinterleave_pq2_planes_s8(activations[3]),
    ];
    dot_pq2_4x4_tile_dotprod(&weights, &acts, load_s32x4(&activation_sums))
}

#[target_feature(enable = "neon,i8mm")]
fn dot_pq2_eight_rows_eight_tokens_i8mm(
    packed: [&[u8; 32]; 8],
    activations: [&[i8; 128]; 8],
    activation_sums: [i32; 8],
) -> [[i32; 8]; 8] {
    let weights_lo = [
        unpack_pq2_unsigned_s8(packed[0]),
        unpack_pq2_unsigned_s8(packed[1]),
        unpack_pq2_unsigned_s8(packed[2]),
        unpack_pq2_unsigned_s8(packed[3]),
    ];
    let weights_hi = [
        unpack_pq2_unsigned_s8(packed[4]),
        unpack_pq2_unsigned_s8(packed[5]),
        unpack_pq2_unsigned_s8(packed[6]),
        unpack_pq2_unsigned_s8(packed[7]),
    ];
    let acts_lo = [
        deinterleave_pq2_planes_s8(activations[0]),
        deinterleave_pq2_planes_s8(activations[1]),
        deinterleave_pq2_planes_s8(activations[2]),
        deinterleave_pq2_planes_s8(activations[3]),
    ];
    let acts_hi = [
        deinterleave_pq2_planes_s8(activations[4]),
        deinterleave_pq2_planes_s8(activations[5]),
        deinterleave_pq2_planes_s8(activations[6]),
        deinterleave_pq2_planes_s8(activations[7]),
    ];
    let sums_lo = load_s32x4(&[
        activation_sums[0],
        activation_sums[1],
        activation_sums[2],
        activation_sums[3],
    ]);
    let sums_hi = load_s32x4(&[
        activation_sums[4],
        activation_sums[5],
        activation_sums[6],
        activation_sums[7],
    ]);
    let q00 = dot_pq2_4x4_tile_i8mm(&weights_lo, &acts_lo, sums_lo);
    let q01 = dot_pq2_4x4_tile_i8mm(&weights_lo, &acts_hi, sums_hi);
    let q10 = dot_pq2_4x4_tile_i8mm(&weights_hi, &acts_lo, sums_lo);
    let q11 = dot_pq2_4x4_tile_i8mm(&weights_hi, &acts_hi, sums_hi);
    let mut out = [[0_i32; 8]; 8];
    for r in 0..4 {
        out[r][..4].copy_from_slice(&q00[r]);
        out[r][4..].copy_from_slice(&q01[r]);
        out[r + 4][..4].copy_from_slice(&q10[r]);
        out[r + 4][4..].copy_from_slice(&q11[r]);
    }
    out
}

#[target_feature(enable = "neon,dotprod")]
fn dot_pq2_eight_rows_eight_tokens_dotprod(
    packed: [&[u8; 32]; 8],
    activations: [&[i8; 128]; 8],
    activation_sums: [i32; 8],
) -> [[i32; 8]; 8] {
    let weights_lo = [
        unpack_pq2_unsigned_s8(packed[0]),
        unpack_pq2_unsigned_s8(packed[1]),
        unpack_pq2_unsigned_s8(packed[2]),
        unpack_pq2_unsigned_s8(packed[3]),
    ];
    let weights_hi = [
        unpack_pq2_unsigned_s8(packed[4]),
        unpack_pq2_unsigned_s8(packed[5]),
        unpack_pq2_unsigned_s8(packed[6]),
        unpack_pq2_unsigned_s8(packed[7]),
    ];
    let acts_lo = [
        deinterleave_pq2_planes_s8(activations[0]),
        deinterleave_pq2_planes_s8(activations[1]),
        deinterleave_pq2_planes_s8(activations[2]),
        deinterleave_pq2_planes_s8(activations[3]),
    ];
    let acts_hi = [
        deinterleave_pq2_planes_s8(activations[4]),
        deinterleave_pq2_planes_s8(activations[5]),
        deinterleave_pq2_planes_s8(activations[6]),
        deinterleave_pq2_planes_s8(activations[7]),
    ];
    let sums_lo = load_s32x4(&[
        activation_sums[0],
        activation_sums[1],
        activation_sums[2],
        activation_sums[3],
    ]);
    let sums_hi = load_s32x4(&[
        activation_sums[4],
        activation_sums[5],
        activation_sums[6],
        activation_sums[7],
    ]);
    let q00 = dot_pq2_4x4_tile_dotprod(&weights_lo, &acts_lo, sums_lo);
    let q01 = dot_pq2_4x4_tile_dotprod(&weights_lo, &acts_hi, sums_hi);
    let q10 = dot_pq2_4x4_tile_dotprod(&weights_hi, &acts_lo, sums_lo);
    let q11 = dot_pq2_4x4_tile_dotprod(&weights_hi, &acts_hi, sums_hi);
    let mut out = [[0_i32; 8]; 8];
    for r in 0..4 {
        out[r][..4].copy_from_slice(&q00[r]);
        out[r][4..].copy_from_slice(&q01[r]);
        out[r + 4][..4].copy_from_slice(&q10[r]);
        out[r + 4][4..].copy_from_slice(&q11[r]);
    }
    out
}

#[inline]
pub(super) fn dot_pq2_four_rows_one_token(
    packed: [&[u8; 32]; 4],
    activation: &[i8; 128],
    activation_sum: i32,
) -> Option<[i32; 4]> {
    if !has_dotprod() {
        return None;
    }
    // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
    Some(unsafe { dot_pq2_four_rows_one_token_dotprod(packed, activation, activation_sum) })
}

#[inline]
pub(super) fn dot_pq2_four_rows_four_tokens(
    packed: [&[u8; 32]; 4],
    activations: [&[i8; 128]; 4],
    activation_sums: [i32; 4],
) -> Option<[[i32; 4]; 4]> {
    if has_i8mm() {
        // SAFETY: `has_i8mm()` verified that the CPU supports `i8mm` and baseline `neon`.
        return Some(unsafe {
            dot_pq2_four_rows_four_tokens_i8mm(packed, activations, activation_sums)
        });
    }
    if has_dotprod() {
        // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
        return Some(unsafe {
            dot_pq2_four_rows_four_tokens_dotprod(packed, activations, activation_sums)
        });
    }
    None
}

#[inline]
pub(super) fn dot_pq2_eight_rows_eight_tokens(
    packed: [&[u8; 32]; 8],
    activations: [&[i8; 128]; 8],
    activation_sums: [i32; 8],
) -> Option<[[i32; 8]; 8]> {
    if has_i8mm() {
        // SAFETY: `has_i8mm()` verified that the CPU supports `i8mm` and baseline `neon`.
        return Some(unsafe {
            dot_pq2_eight_rows_eight_tokens_i8mm(packed, activations, activation_sums)
        });
    }
    if has_dotprod() {
        // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
        return Some(unsafe {
            dot_pq2_eight_rows_eight_tokens_dotprod(packed, activations, activation_sums)
        });
    }
    None
}
