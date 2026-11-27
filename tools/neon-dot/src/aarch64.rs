#![expect(
    clippy::redundant_pub_crate,
    reason = "Explicit parent-only visibility keeps the ARM64 intrinsics inside the audited module boundary."
)]

use core::arch::aarch64::{
    int8x16_t, int32x4_t, uint8x8_t, uint8x16_t, vaddq_s32, vaddvq_s32, vandq_u8, vcgeq_u8,
    vcltq_u8, vcombine_s8, vcombine_s32, vcombine_u8, vdotq_s32, vdupq_n_s32, vdupq_n_u8,
    vget_high_s8, vget_high_s32, vget_low_s8, vget_low_s32, vld1_u8, vld1q_s8, vld1q_s32, vld1q_u8,
    vmulq_u8, vnegq_s32, vpaddlq_s8, vpaddlq_s16, vreinterpret_u8_u16, vreinterpretq_s8_u8,
    vshrq_n_u8, vst1q_s8, vst1q_s32, vsubq_s8, vsubq_s32, vuzp1q_s8, vuzp2q_s8,
};
use std::sync::atomic::{AtomicU8, Ordering};

const UNINIT: u8 = 0;
const ABSENT: u8 = 1;
const PRESENT: u8 = 2;

static DOTPROD_STATE: AtomicU8 = AtomicU8::new(UNINIT);
static I8MM_STATE: AtomicU8 = AtomicU8::new(UNINIT);
static SVE_STATE: AtomicU8 = AtomicU8::new(UNINIT);

// FCVTL is part of baseline AArch64 NEON, independently of the optional FP16
// arithmetic extension. Its exact finite widening ignores FZ16. Sanitizing via
// the integer exceptional check also avoids AHP/DN and signaling-NaN changes.
#[cfg(target_feature = "neon")]
#[inline]
pub(super) fn widen_f16_four(bits: [u16; 4]) -> [f32; 4] {
    // SAFETY: this wrapper is compiled only for targets with baseline NEON.
    // The fixed array supplies every input element and no memory is retained.
    unsafe { widen_f16_four_neon(bits) }
}

#[cfg(target_feature = "neon")]
#[inline]
#[target_feature(enable = "neon")]
fn widen_f16_four_neon(bits: [u16; 4]) -> [f32; 4] {
    use core::arch::aarch64::{
        vand_u16, vcvt_f32_f16, vdup_n_u16, vld1_u16, vmaxv_u16, vreinterpret_f16_u16, vst1q_f32,
    };
    // SAFETY: fixed-array extent covers the unaligned load. This function is
    // compiled only when baseline NEON is enabled for its AArch64 target.
    let raw = unsafe { vld1_u16(bits.as_ptr()) };
    if vmaxv_u16(vand_u16(raw, vdup_n_u16(0x7fff))) >= 0x7c00 {
        return bits.map(bonsai_ternary::half_to_float);
    }
    let value = vcvt_f32_f16(vreinterpret_f16_u16(raw));
    let mut output = [0.0; 4];
    // SAFETY: output is a unique valid four-element array; unaligned stores
    // are supported, and the vector contains four initialized values.
    unsafe { vst1q_f32(output.as_mut_ptr(), value) };
    output
}

#[cfg(target_feature = "neon")]
#[inline]
pub(super) fn widen_f16_eight(bits: &[u16; 8]) -> [[f32; 4]; 2] {
    // SAFETY: this wrapper is compiled only when baseline NEON is available;
    // fixed-size input and output arrays cover every unaligned access.
    unsafe { widen_f16_eight_neon(bits) }
}

#[cfg(target_feature = "neon")]
#[inline]
#[target_feature(enable = "neon")]
fn widen_f16_eight_neon(bits: &[u16; 8]) -> [[f32; 4]; 2] {
    use core::arch::aarch64::{
        vandq_u16, vcvt_f32_f16, vcvt_high_f32_f16, vdupq_n_u16, vget_low_f16, vld1q_u16,
        vmaxvq_u16, vreinterpretq_f16_u16, vst1q_f32,
    };
    // SAFETY: fixed-array extent covers the unaligned load. The enclosing
    // cfg ensures baseline AArch64 NEON; no optional FP16 arithmetic is used.
    let raw = unsafe { vld1q_u16(bits.as_ptr()) };
    if vmaxvq_u16(vandq_u16(raw, vdupq_n_u16(0x7fff))) >= 0x7c00 {
        return super::half::portable_eight(bits);
    }
    let half = vreinterpretq_f16_u16(raw);
    let first = vcvt_f32_f16(vget_low_f16(half));
    let second = vcvt_high_f32_f16(half);
    let mut output = [[0.0; 4]; 2];
    // SAFETY: the first output is a valid unique four-element array, and
    // `first` contains four initialized values; unaligned stores are valid.
    unsafe { vst1q_f32(output[0].as_mut_ptr(), first) };
    // SAFETY: the second output is a disjoint valid four-element array, and
    // `second` contains four initialized values; unaligned stores are valid.
    unsafe { vst1q_f32(output[1].as_mut_ptr(), second) };
    output
}

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

#[cfg(target_os = "windows")]
fn probe_sve() -> bool {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn IsProcessorFeaturePresent(processor_feature: u32) -> i32;
    }
    const PF_ARM_SVE_INSTRUCTIONS_AVAILABLE: u32 = 46;
    // SAFETY: `IsProcessorFeaturePresent` is a read-only Windows OS query.
    unsafe { IsProcessorFeaturePresent(PF_ARM_SVE_INSTRUCTIONS_AVAILABLE) != 0 }
}

#[cfg(not(target_os = "windows"))]
fn probe_sve() -> bool {
    std::arch::is_aarch64_feature_detected!("sve")
}

#[inline]
pub(super) fn has_sve() -> bool {
    if cfg!(target_feature = "sve") {
        return true;
    }
    match SVE_STATE.load(Ordering::Relaxed) {
        PRESENT => true,
        ABSENT => false,
        _ => {
            let detected = probe_sve();
            SVE_STATE.store(if detected { PRESENT } else { ABSENT }, Ordering::Relaxed);
            detected
        }
    }
}

#[inline]
fn prefetch_l1(ptr: *const u8) {
    // SAFETY: `prfm pldl1keep` is a non-faulting architectural read prefetch hint.
    unsafe {
        core::arch::asm!(
            "prfm pldl1keep, [{ptr}]",
            ptr = in(reg) ptr,
            options(nostack, readonly, preserves_flags),
        );
    }
}

#[inline]
fn prefetch_row_blocks<const ROWS: usize, const BYTES: usize>(
    row_chunks: &[&[[u8; BYTES]]; ROWS],
    block_idx: usize,
    block_count: usize,
) {
    if block_idx + 2 < block_count {
        for chunk in row_chunks {
            prefetch_l1(chunk[block_idx + 2].as_ptr());
        }
    }
}

#[inline]
fn prefetch_pq2_tokens_block(
    tokens: &[&[super::Pq2Activation]],
    block_idx: usize,
    block_count: usize,
) {
    if block_idx + 1 < block_count {
        for tok in tokens {
            let ptr = tok[block_idx + 1].values.as_ptr().cast::<u8>();
            prefetch_l1(ptr);
            prefetch_l1(ptr.wrapping_add(64));
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

#[inline]
#[target_feature(enable = "neon")]
fn load_f32x4_neon(values: &[f32; 4]) -> core::arch::aarch64::float32x4_t {
    // SAFETY: `values` is a valid 4-element `f32` array reference and `vld1q_f32` requires no alignment.
    unsafe { core::arch::aarch64::vld1q_f32(values.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_f32x4_neon(vec: core::arch::aarch64::float32x4_t, out: &mut [f32; 4]) {
    // SAFETY: `out` is a valid mutable 4-element `f32` array reference and `vst1q_f32` requires no alignment.
    unsafe { core::arch::aarch64::vst1q_f32(out.as_mut_ptr(), vec) };
}

/// Extracts signed trits (`-1..=1`) from unsigned fractional base-3 residues `v`:
/// `((u16::from(v) * 3) >> 8) as i8 - 1` equals `-1` for `v < 86`, `0` for `86..171`,
/// and `+1` for `v >= 171`. Since NEON `vcltq_u8` / `vcgeq_u8` yield `0xFF` (`-1_i8`)
/// when true and `0x00` (`0_i8`) when false, `lt_86 - ge_171` computes all 16 signed
/// trits in 3 instructions without widening.
#[inline]
#[target_feature(enable = "neon")]
fn extract_trit_s8x16_consts(v: uint8x16_t, c86: uint8x16_t, c171: uint8x16_t) -> int8x16_t {
    let lt_86 = vreinterpretq_s8_u8(vcltq_u8(v, c86));
    let ge_171 = vreinterpretq_s8_u8(vcgeq_u8(v, c171));
    vsubq_s8(lt_86, ge_171)
}

#[inline]
#[target_feature(enable = "neon")]
fn extract_trit_s8x16(v: uint8x16_t) -> int8x16_t {
    extract_trit_s8x16_consts(v, vdupq_n_u8(86), vdupq_n_u8(171))
}

#[inline]
#[target_feature(enable = "neon")]
fn unpack_ptq1_vectors(packed: &[u8; 28]) -> [int8x16_t; 8] {
    use core::arch::aarch64::vdup_n_u16;
    const POW3_MID_01: [u8; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3];
    const POW3_MID_23: [u8; 16] = [9, 9, 9, 9, 9, 9, 9, 9, 27, 27, 27, 27, 27, 27, 27, 27];
    const POW3_TAIL: [u8; 16] = [81, 81, 81, 81, 81, 81, 81, 81, 1, 1, 3, 3, 9, 9, 27, 27];

    let c86 = vdupq_n_u8(86);
    let c171 = vdupq_n_u8(171);
    let q0 = load_u8x16(&packed[..16].as_chunks::<16>().0[0]);
    let t0 = extract_trit_s8x16_consts(q0, c86, c171);
    let t1 = extract_trit_s8x16_consts(vmulq_u8(q0, vdupq_n_u8(3)), c86, c171);
    let t2 = extract_trit_s8x16_consts(vmulq_u8(q0, vdupq_n_u8(9)), c86, c171);
    let t3 = extract_trit_s8x16_consts(vmulq_u8(q0, vdupq_n_u8(27)), c86, c171);
    let t4 = extract_trit_s8x16_consts(vmulq_u8(q0, vdupq_n_u8(81)), c86, c171);

    let q_mid_half = load_u8x8(&packed[16..24].as_chunks::<8>().0[0]);
    let q_mid = vcombine_u8(q_mid_half, q_mid_half);
    let t5 = extract_trit_s8x16_consts(vmulq_u8(q_mid, load_u8x16(&POW3_MID_01)), c86, c171);
    let t6 = extract_trit_s8x16_consts(vmulq_u8(q_mid, load_u8x16(&POW3_MID_23)), c86, c171);

    let tail_half = vreinterpret_u8_u16(vdup_n_u16(u16::from_le_bytes([packed[24], packed[25]])));
    let q_tail = vcombine_u8(q_mid_half, tail_half);
    let t7 = extract_trit_s8x16_consts(vmulq_u8(q_tail, load_u8x16(&POW3_TAIL)), c86, c171);

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
    let a0 = activations[0].as_chunks::<16>().0;
    let a1 = activations[1].as_chunks::<16>().0;
    let a2 = activations[2].as_chunks::<16>().0;
    let a3 = activations[3].as_chunks::<16>().0;
    let g0 = sdot32_vec(t0, t1, load_s8x16(&a0[0]), load_s8x16(&a0[1]));
    let g1 = sdot32_vec(t2, t3, load_s8x16(&a1[0]), load_s8x16(&a1[1]));
    let g2 = sdot32_vec(t4, t5, load_s8x16(&a2[0]), load_s8x16(&a2[1]));
    let g3 = sdot32_vec(t6, t7, load_s8x16(&a3[0]), load_s8x16(&a3[1]));
    let mut out = [0_i32; 4];
    store_s32x4(reduce4_s32x4_addp(g0, g1, g2, g3), &mut out);
    out
}

#[target_feature(enable = "neon,dotprod")]
fn dot_ptq1_rows_lanes_dotprod<const ROWS: usize>(
    packed: [&[u8; 28]; ROWS],
    activations: [&[i8; 32]; 4],
) -> [[[i32; 4]; 4]; ROWS] {
    let a0 = activations[0].as_chunks::<16>().0;
    let a1 = activations[1].as_chunks::<16>().0;
    let a2 = activations[2].as_chunks::<16>().0;
    let a3 = activations[3].as_chunks::<16>().0;
    let act_pairs = [
        (load_s8x16(&a0[0]), load_s8x16(&a0[1])),
        (load_s8x16(&a1[0]), load_s8x16(&a1[1])),
        (load_s8x16(&a2[0]), load_s8x16(&a2[1])),
        (load_s8x16(&a3[0]), load_s8x16(&a3[1])),
    ];
    let mut out = [[[0_i32; 4]; 4]; ROWS];
    for (row, block) in packed.into_iter().enumerate() {
        let [t0, t1, t2, t3, t4, t5, t6, t7] = unpack_ptq1_vectors(block);
        let pairs = [(t0, t1), (t2, t3), (t4, t5), (t6, t7)];
        for (group, ((w0, w1), (act0, act1))) in pairs.into_iter().zip(act_pairs).enumerate() {
            let acc = sdot32_vec(w0, w1, act0, act1);
            store_s32x4(acc, &mut out[row][group]);
        }
    }
    out
}

#[target_feature(enable = "neon,dotprod")]
fn dot_ptq1_rows_groups_dotprod<const ROWS: usize>(
    packed: [&[u8; 28]; ROWS],
    activations: [&[i8; 32]; 4],
) -> [[i32; 4]; ROWS] {
    let a0 = activations[0].as_chunks::<16>().0;
    let a1 = activations[1].as_chunks::<16>().0;
    let a2 = activations[2].as_chunks::<16>().0;
    let a3 = activations[3].as_chunks::<16>().0;
    let act_pairs = [
        (load_s8x16(&a0[0]), load_s8x16(&a0[1])),
        (load_s8x16(&a1[0]), load_s8x16(&a1[1])),
        (load_s8x16(&a2[0]), load_s8x16(&a2[1])),
        (load_s8x16(&a3[0]), load_s8x16(&a3[1])),
    ];
    let mut out = [[0_i32; 4]; ROWS];
    for (row, block) in packed.into_iter().enumerate() {
        let [t0, t1, t2, t3, t4, t5, t6, t7] = unpack_ptq1_vectors(block);
        let g0 = sdot32_vec(t0, t1, act_pairs[0].0, act_pairs[0].1);
        let g1 = sdot32_vec(t2, t3, act_pairs[1].0, act_pairs[1].1);
        let g2 = sdot32_vec(t4, t5, act_pairs[2].0, act_pairs[2].1);
        let g3 = sdot32_vec(t6, t7, act_pairs[3].0, act_pairs[3].1);
        store_s32x4(reduce4_s32x4_addp(g0, g1, g2, g3), &mut out[row]);
    }
    out
}

#[inline]
pub(super) fn dot_ptq1_sixteen_rows_lanes(
    packed: [&[u8; 28]; 16],
    activations: [&[i8; 32]; 4],
) -> Option<[[[i32; 4]; 4]; 16]> {
    if !has_dotprod() {
        return None;
    }
    // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
    Some(unsafe { dot_ptq1_rows_lanes_dotprod(packed, activations) })
}

#[inline]
pub(super) fn dot_ptq1_sixteen_rows_groups(
    packed: [&[u8; 28]; 16],
    activations: [&[i8; 32]; 4],
) -> Option<[[i32; 4]; 16]> {
    if !has_dotprod() {
        return None;
    }
    // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
    Some(unsafe { dot_ptq1_rows_groups_dotprod(packed, activations) })
}

#[target_feature(enable = "neon,dotprod")]
fn dot_ptq1_four_rows_lanes_dotprod(
    packed: [&[u8; 28]; 4],
    activations: [&[i8; 32]; 4],
) -> [[[i32; 4]; 4]; 4] {
    let a0 = activations[0].as_chunks::<16>().0;
    let a1 = activations[1].as_chunks::<16>().0;
    let a2 = activations[2].as_chunks::<16>().0;
    let a3 = activations[3].as_chunks::<16>().0;
    let act_pairs = [
        (load_s8x16(&a0[0]), load_s8x16(&a0[1])),
        (load_s8x16(&a1[0]), load_s8x16(&a1[1])),
        (load_s8x16(&a2[0]), load_s8x16(&a2[1])),
        (load_s8x16(&a3[0]), load_s8x16(&a3[1])),
    ];
    let mut out = [[[0_i32; 4]; 4]; 4];
    for (row, block) in packed.into_iter().enumerate() {
        let [t0, t1, t2, t3, t4, t5, t6, t7] = unpack_ptq1_vectors(block);
        let pairs = [(t0, t1), (t2, t3), (t4, t5), (t6, t7)];
        for (group, ((w0, w1), (act0, act1))) in pairs.into_iter().zip(act_pairs).enumerate() {
            let acc = sdot32_vec(w0, w1, act0, act1);
            store_s32x4(acc, &mut out[row][group]);
        }
    }
    out
}

#[target_feature(enable = "neon,dotprod")]
fn dot_ptq1_four_rows_groups_dotprod(
    packed: [&[u8; 28]; 4],
    activations: [&[i8; 32]; 4],
) -> [[i32; 4]; 4] {
    let a0 = activations[0].as_chunks::<16>().0;
    let a1 = activations[1].as_chunks::<16>().0;
    let a2 = activations[2].as_chunks::<16>().0;
    let a3 = activations[3].as_chunks::<16>().0;
    let act00 = load_s8x16(&a0[0]);
    let act01 = load_s8x16(&a0[1]);
    let act10 = load_s8x16(&a1[0]);
    let act11 = load_s8x16(&a1[1]);
    let act20 = load_s8x16(&a2[0]);
    let act21 = load_s8x16(&a2[1]);
    let act30 = load_s8x16(&a3[0]);
    let act31 = load_s8x16(&a3[1]);
    let mut out = [[0_i32; 4]; 4];
    for (row, block) in packed.into_iter().enumerate() {
        let [t0, t1, t2, t3, t4, t5, t6, t7] = unpack_ptq1_vectors(block);
        let g0 = sdot32_vec(t0, t1, act00, act01);
        let g1 = sdot32_vec(t2, t3, act10, act11);
        let g2 = sdot32_vec(t4, t5, act20, act21);
        let g3 = sdot32_vec(t6, t7, act30, act31);
        store_s32x4(reduce4_s32x4_addp(g0, g1, g2, g3), &mut out[row]);
    }
    out
}

#[target_feature(enable = "neon,dotprod")]
fn dot_ptq1_eight_rows_lanes_dotprod(
    packed: [&[u8; 28]; 8],
    activations: [&[i8; 32]; 4],
) -> [[[i32; 4]; 4]; 8] {
    let a0 = activations[0].as_chunks::<16>().0;
    let a1 = activations[1].as_chunks::<16>().0;
    let a2 = activations[2].as_chunks::<16>().0;
    let a3 = activations[3].as_chunks::<16>().0;
    let act_pairs = [
        (load_s8x16(&a0[0]), load_s8x16(&a0[1])),
        (load_s8x16(&a1[0]), load_s8x16(&a1[1])),
        (load_s8x16(&a2[0]), load_s8x16(&a2[1])),
        (load_s8x16(&a3[0]), load_s8x16(&a3[1])),
    ];
    let mut out = [[[0_i32; 4]; 4]; 8];
    for (row, block) in packed.into_iter().enumerate() {
        let [t0, t1, t2, t3, t4, t5, t6, t7] = unpack_ptq1_vectors(block);
        let pairs = [(t0, t1), (t2, t3), (t4, t5), (t6, t7)];
        for (group, ((w0, w1), (act0, act1))) in pairs.into_iter().zip(act_pairs).enumerate() {
            let acc = sdot32_vec(w0, w1, act0, act1);
            store_s32x4(acc, &mut out[row][group]);
        }
    }
    out
}

#[target_feature(enable = "neon,dotprod")]
fn dot_ptq1_eight_rows_groups_dotprod(
    packed: [&[u8; 28]; 8],
    activations: [&[i8; 32]; 4],
) -> [[i32; 4]; 8] {
    let a0 = activations[0].as_chunks::<16>().0;
    let a1 = activations[1].as_chunks::<16>().0;
    let a2 = activations[2].as_chunks::<16>().0;
    let a3 = activations[3].as_chunks::<16>().0;
    let act00 = load_s8x16(&a0[0]);
    let act01 = load_s8x16(&a0[1]);
    let act10 = load_s8x16(&a1[0]);
    let act11 = load_s8x16(&a1[1]);
    let act20 = load_s8x16(&a2[0]);
    let act21 = load_s8x16(&a2[1]);
    let act30 = load_s8x16(&a3[0]);
    let act31 = load_s8x16(&a3[1]);
    let mut out = [[0_i32; 4]; 8];
    for (row, block) in packed.into_iter().enumerate() {
        let [t0, t1, t2, t3, t4, t5, t6, t7] = unpack_ptq1_vectors(block);
        let g0 = sdot32_vec(t0, t1, act00, act01);
        let g1 = sdot32_vec(t2, t3, act10, act11);
        let g2 = sdot32_vec(t4, t5, act20, act21);
        let g3 = sdot32_vec(t6, t7, act30, act31);
        store_s32x4(reduce4_s32x4_addp(g0, g1, g2, g3), &mut out[row]);
    }
    out
}

#[inline]
pub(super) fn dot_ptq1_eight_rows_lanes(
    packed: [&[u8; 28]; 8],
    activations: [&[i8; 32]; 4],
) -> Option<[[[i32; 4]; 4]; 8]> {
    if !has_dotprod() {
        return None;
    }
    // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
    Some(unsafe { dot_ptq1_eight_rows_lanes_dotprod(packed, activations) })
}

#[inline]
pub(super) fn dot_ptq1_eight_rows_groups(
    packed: [&[u8; 28]; 8],
    activations: [&[i8; 32]; 4],
) -> Option<[[i32; 4]; 8]> {
    if !has_dotprod() {
        return None;
    }
    // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
    Some(unsafe { dot_ptq1_eight_rows_groups_dotprod(packed, activations) })
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
pub(super) fn dot_ptq1_four_rows_lanes(
    packed: [&[u8; 28]; 4],
    activations: [&[i8; 32]; 4],
) -> Option<[[[i32; 4]; 4]; 4]> {
    if !has_dotprod() {
        return None;
    }
    // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
    Some(unsafe { dot_ptq1_four_rows_lanes_dotprod(packed, activations) })
}

#[inline]
pub(super) fn dot_ptq1_four_rows_groups(
    packed: [&[u8; 28]; 4],
    activations: [&[i8; 32]; 4],
) -> Option<[[i32; 4]; 4]> {
    if !has_dotprod() {
        return None;
    }
    // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
    Some(unsafe { dot_ptq1_four_rows_groups_dotprod(packed, activations) })
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
fn pq2_half_unsigned_sdot_masked(
    acc: int32x4_t,
    w: uint8x16_t,
    p0: int8x16_t,
    p1: int8x16_t,
    p2: int8x16_t,
    p3: int8x16_t,
    mask: uint8x16_t,
) -> int32x4_t {
    let w0 = vreinterpretq_s8_u8(vandq_u8(w, mask));
    let w1 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<2>(w), mask));
    let w2 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<4>(w), mask));
    let w3 = vreinterpretq_s8_u8(vshrq_n_u8::<6>(w));
    let acc = vdotq_s32(acc, w0, p0);
    let acc = vdotq_s32(acc, w1, p1);
    let acc = vdotq_s32(acc, w2, p2);
    vdotq_s32(acc, w3, p3)
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
    pq2_half_unsigned_sdot_masked(acc, w, p0, p1, p2, p3, vdupq_n_u8(3))
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
#[target_feature(enable = "neon")]
fn reduce4_s32x4_addp(v0: int32x4_t, v1: int32x4_t, v2: int32x4_t, v3: int32x4_t) -> int32x4_t {
    use core::arch::aarch64::vpaddq_s32;
    vpaddq_s32(vpaddq_s32(v0, v1), vpaddq_s32(v2, v3))
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
    let reduced = reduce4_s32x4_addp(acc[0], acc[1], acc[2], acc[3]);
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

#[target_feature(enable = "neon,dotprod")]
fn dot_pq2_eight_consumption_dotprod(
    packed: [&[u8; 32]; 8],
    planes: &[i8; 128],
    activation_sum: i32,
) -> [i32; 8] {
    let mut acc = [vdupq_n_s32(0); 8];
    for (half, half_planes) in planes.as_chunks::<64>().0.iter().enumerate() {
        let p = half_planes.as_chunks::<16>().0;
        let p0 = load_s8x16(&p[0]);
        let p1 = load_s8x16(&p[1]);
        let p2 = load_s8x16(&p[2]);
        let p3 = load_s8x16(&p[3]);
        for row in 0..8 {
            let bytes = &packed[row].as_chunks::<16>().0[half];
            acc[row] = pq2_half_unsigned_sdot(acc[row], load_u8x16(bytes), p0, p1, p2, p3);
        }
    }
    let sum_vec = vdupq_n_s32(activation_sum);
    let reduced_lo = vsubq_s32(reduce4_s32x4_addp(acc[0], acc[1], acc[2], acc[3]), sum_vec);
    let reduced_hi = vsubq_s32(reduce4_s32x4_addp(acc[4], acc[5], acc[6], acc[7]), sum_vec);
    let mut out_lo = [0_i32; 4];
    let mut out_hi = [0_i32; 4];
    store_s32x4(reduced_lo, &mut out_lo);
    store_s32x4(reduced_hi, &mut out_hi);
    [
        out_lo[0], out_lo[1], out_lo[2], out_lo[3], out_hi[0], out_hi[1], out_hi[2], out_hi[3],
    ]
}

macro_rules! pq2_scale_accum_asm {
    ($sum:literal, $scale:literal, $dot:literal, fused) => {
        concat!("fmla ", $sum, ".4s, ", $scale, ".4s, ", $dot, ".4s\n")
    };
    ($sum:literal, $scale:literal, $dot:literal, nonfused) => {
        concat!(
            "fmul ", $dot, ".4s, ", $scale, ".4s, ", $dot, ".4s\n", "fadd ", $sum, ".4s, ", $sum,
            ".4s, ", $dot, ".4s\n",
        )
    };
}

macro_rules! pq2_load_act_asm {
    () => {
        concat!(
            "ldr s6, [{act}], #4\n",
            "ldp q8, q9, [{act}]\n",
            "ldp q10, q11, [{act}, #32]\n",
            "ldp q12, q13, [{act}, #64]\n",
            "ldp q14, q15, [{act}, #96]\n",
            "ldr s7, [{act}, #128]\n",
            "add {act}, {act}, #132\n",
        )
    };
}

macro_rules! pq2_row_sdot_asm {
    ($p:literal, $acc:literal) => {
        concat!(
            "ldp q24, q25, [{",
            $p,
            "}], #32\n",
            "movi ",
            $acc,
            ".2d, #0\n",
            "and v26.16b, v24.16b, v5.16b\n",
            "ushr v27.16b, v24.16b, #2\n",
            "ushr v28.16b, v24.16b, #4\n",
            "ushr v29.16b, v24.16b, #6\n",
            "and v27.16b, v27.16b, v5.16b\n",
            "and v28.16b, v28.16b, v5.16b\n",
            "sdot ",
            $acc,
            ".4s, v26.16b, v8.16b\n",
            "sdot ",
            $acc,
            ".4s, v27.16b, v9.16b\n",
            "sdot ",
            $acc,
            ".4s, v28.16b, v10.16b\n",
            "sdot ",
            $acc,
            ".4s, v29.16b, v11.16b\n",
            "and v26.16b, v25.16b, v5.16b\n",
            "ushr v27.16b, v25.16b, #2\n",
            "ushr v28.16b, v25.16b, #4\n",
            "ushr v29.16b, v25.16b, #6\n",
            "and v27.16b, v27.16b, v5.16b\n",
            "and v28.16b, v28.16b, v5.16b\n",
            "sdot ",
            $acc,
            ".4s, v26.16b, v12.16b\n",
            "sdot ",
            $acc,
            ".4s, v27.16b, v13.16b\n",
            "sdot ",
            $acc,
            ".4s, v28.16b, v14.16b\n",
            "sdot ",
            $acc,
            ".4s, v29.16b, v15.16b\n",
        )
    };
}

macro_rules! pq2_octet_block_asm {
    (
        $p0:literal,
        $p1:literal,
        $p2:literal,
        $p3:literal,
        $p4:literal,
        $p5:literal,
        $p6:literal,
        $p7:literal,
        $sum0:literal,
        $sum1:literal,
        $fused:ident
    ) => {
        concat!(
            "ldrh {t0:w}, [{",
            $p0,
            "}], #2\n",
            "ldrh {t1:w}, [{",
            $p1,
            "}], #2\n",
            "fmov s30, {t0:w}\n",
            "ldrh {t0:w}, [{",
            $p2,
            "}], #2\n",
            "ins v30.h[1], {t1:w}\n",
            "ldrh {t1:w}, [{",
            $p3,
            "}], #2\n",
            "ins v30.h[2], {t0:w}\n",
            "ldrh {t0:w}, [{",
            $p4,
            "}], #2\n",
            "ins v30.h[3], {t1:w}\n",
            "ldrh {t1:w}, [{",
            $p5,
            "}], #2\n",
            "ins v30.h[4], {t0:w}\n",
            "ldrh {t0:w}, [{",
            $p6,
            "}], #2\n",
            "ins v30.h[5], {t1:w}\n",
            "ldrh {t1:w}, [{",
            $p7,
            "}], #2\n",
            "ins v30.h[6], {t0:w}\n",
            "ins v30.h[7], {t1:w}\n",
            "mov v31.16b, v30.16b\n",
            "bic v31.8h, #0x80, lsl #8\n",
            "umax v4.8h, v4.8h, v31.8h\n",
            pq2_row_sdot_asm!($p0, "v16"),
            pq2_row_sdot_asm!($p1, "v17"),
            pq2_row_sdot_asm!($p2, "v18"),
            pq2_row_sdot_asm!($p3, "v19"),
            pq2_row_sdot_asm!($p4, "v20"),
            pq2_row_sdot_asm!($p5, "v21"),
            pq2_row_sdot_asm!($p6, "v22"),
            pq2_row_sdot_asm!($p7, "v23"),
            "addp v16.4s, v16.4s, v17.4s\n",
            "addp v18.4s, v18.4s, v19.4s\n",
            "addp v16.4s, v16.4s, v18.4s\n",
            "sub v16.4s, v16.4s, v7.4s\n",
            "scvtf v16.4s, v16.4s\n",
            "fcvtl v31.4s, v30.4h\n",
            "fmul v31.4s, v31.4s, v6.s[0]\n",
            pq2_scale_accum_asm!($sum0, "v31", "v16", $fused),
            "addp v20.4s, v20.4s, v21.4s\n",
            "addp v22.4s, v22.4s, v23.4s\n",
            "addp v20.4s, v20.4s, v22.4s\n",
            "sub v20.4s, v20.4s, v7.4s\n",
            "scvtf v20.4s, v20.4s\n",
            "fcvtl2 v30.4s, v30.8h\n",
            "fmul v30.4s, v30.4s, v6.s[0]\n",
            pq2_scale_accum_asm!($sum1, "v30", "v20", $fused),
        )
    };
}

macro_rules! pq2_1row_block_asm {
    (fused) => {
        concat!(
            "ldrh {t0:w}, [{p0}], #2\n",
            "fmov s30, {t0:w}\n",
            "mov v31.8b, v30.8b\n",
            "bic v31.4h, #0x80, lsl #8\n",
            "umax v4.4h, v4.4h, v31.4h\n",
            pq2_load_act_asm!(),
            pq2_row_sdot_asm!("p0", "v16"),
            "addv s16, v16.4s\n",
            "sub v16.2s, v16.2s, v7.2s\n",
            "scvtf s16, s16\n",
            "fcvt s30, h30\n",
            "fmul s30, s30, s6\n",
            "fmadd s0, s30, s16, s0\n",
        )
    };
    (nonfused) => {
        concat!(
            "ldrh {t0:w}, [{p0}], #2\n",
            "fmov s30, {t0:w}\n",
            "mov v31.8b, v30.8b\n",
            "bic v31.4h, #0x80, lsl #8\n",
            "umax v4.4h, v4.4h, v31.4h\n",
            pq2_load_act_asm!(),
            pq2_row_sdot_asm!("p0", "v16"),
            "addv s16, v16.4s\n",
            "sub v16.2s, v16.2s, v7.2s\n",
            "scvtf s16, s16\n",
            "fcvt s30, h30\n",
            "fmul s30, s30, s6\n",
            "fmul s16, s30, s16\n",
            "fadd s0, s0, s16\n",
        )
    };
}

macro_rules! pq2_4rows_block_asm {
    ($fused:ident) => {
        concat!(
            "ldrh {t0:w}, [{p0}], #2\n",
            "ldrh {t1:w}, [{p1}], #2\n",
            "fmov s30, {t0:w}\n",
            "ldrh {t0:w}, [{p2}], #2\n",
            "ins v30.h[1], {t1:w}\n",
            "ldrh {t1:w}, [{p3}], #2\n",
            "ins v30.h[2], {t0:w}\n",
            "ins v30.h[3], {t1:w}\n",
            "mov v31.8b, v30.8b\n",
            "bic v31.4h, #0x80, lsl #8\n",
            "umax v4.4h, v4.4h, v31.4h\n",
            pq2_load_act_asm!(),
            "dup v7.4s, v7.s[0]\n",
            pq2_row_sdot_asm!("p0", "v16"),
            pq2_row_sdot_asm!("p1", "v17"),
            pq2_row_sdot_asm!("p2", "v18"),
            pq2_row_sdot_asm!("p3", "v19"),
            "addp v16.4s, v16.4s, v17.4s\n",
            "addp v18.4s, v18.4s, v19.4s\n",
            "addp v16.4s, v16.4s, v18.4s\n",
            "sub v16.4s, v16.4s, v7.4s\n",
            "scvtf v16.4s, v16.4s\n",
            "fcvtl v30.4s, v30.4h\n",
            "fmul v30.4s, v30.4s, v6.s[0]\n",
            pq2_scale_accum_asm!("v0", "v30", "v16", $fused),
        )
    };
}

macro_rules! pq2_8rows_block_asm {
    ($fused:ident) => {
        concat!(
            pq2_load_act_asm!(),
            "dup v7.4s, v7.s[0]\n",
            pq2_octet_block_asm!(
                "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "v0", "v1", $fused
            ),
        )
    };
}

macro_rules! pq2_16rows_block_asm {
    ($fused:ident) => {
        concat!(
            pq2_load_act_asm!(),
            "dup v7.4s, v7.s[0]\n",
            pq2_octet_block_asm!(
                "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "v0", "v1", $fused
            ),
            pq2_octet_block_asm!(
                "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "v2", "v3", $fused
            ),
        )
    };
}

#[target_feature(enable = "neon,dotprod")]
fn pq2_scaled_1row_asm<const FUSED: bool>(
    row: &[u8],
    activation: &[super::Pq2Activation],
) -> Option<[f32; 1]> {
    use core::arch::aarch64::{vdupq_n_f32, vgetq_lane_f32};
    let block_count = activation.len();
    if block_count == 0 || row.len() != block_count * 34 {
        return None;
    }
    let mut sum0 = vdupq_n_f32(0.0);
    let p0 = row.as_ptr();
    let act = activation.as_ptr();
    let cnt = block_count;
    let ok: usize;
    macro_rules! run_1row_asm {
        ($fused:ident) => {
            // SAFETY: Caller verified `row` covers `block_count * 34` bytes and
            // `activation` covers `block_count` `Pq2Activation` elements.
            unsafe {
                core::arch::asm!(
                    "movi v4.2d, #0",
                    "movi v5.16b, #3",
                    "tst {cnt}, #1",
                    "b.eq 1f",
                    pq2_1row_block_asm!($fused),
                    "sub {cnt}, {cnt}, #1",
                    "cbz {cnt}, 2f",
                    "1:",
                    "prfm pldl1keep, [{p0}, #136]",
                    "prfm pldl1keep, [{act}, #272]",
                    "prfm pldl1keep, [{act}, #336]",
                    pq2_1row_block_asm!($fused),
                    pq2_1row_block_asm!($fused),
                    "subs {cnt}, {cnt}, #2",
                    "b.ne 1b",
                    "2:",
                    "umaxv h28, v4.4h",
                    "fmov {t0:w}, s28",
                    "lsr {t0:w}, {t0:w}, #10",
                    "cmp {t0:w}, #30",
                    "cset {ok}, ls",
                    p0 = inout(reg) p0 => _,
                    act = inout(reg) act => _,
                    cnt = inout(reg) cnt => _,
                    t0 = out(reg) _,
                    ok = out(reg) ok,
                    inout("v0") sum0,
                    out("v4") _,
                    out("v5") _,
                    out("v6") _,
                    out("v7") _,
                    out("v8") _,
                    out("v9") _,
                    out("v10") _,
                    out("v11") _,
                    out("v12") _,
                    out("v13") _,
                    out("v14") _,
                    out("v15") _,
                    out("v16") _,
                    out("v24") _,
                    out("v25") _,
                    out("v26") _,
                    out("v27") _,
                    out("v28") _,
                    out("v29") _,
                    out("v30") _,
                    out("v31") _,
                    options(nostack, readonly),
                );
            }
        };
    }
    if FUSED {
        run_1row_asm!(fused);
    } else {
        run_1row_asm!(nonfused);
    }
    if ok == 0 {
        return None;
    }
    let out = vgetq_lane_f32::<0>(sum0);
    out.is_finite().then_some([out])
}

#[target_feature(enable = "neon,dotprod")]
fn pq2_scaled_4rows_asm<const FUSED: bool>(
    rows: [&[u8]; 4],
    activation: &[super::Pq2Activation],
) -> Option<[f32; 4]> {
    use core::arch::aarch64::vdupq_n_f32;
    let block_count = activation.len();
    let expected = block_count.checked_mul(34)?;
    if block_count == 0 || rows.iter().any(|r| r.len() != expected) {
        return None;
    }
    let mut sum0 = vdupq_n_f32(0.0);
    let p0 = rows[0].as_ptr();
    let p1 = rows[1].as_ptr();
    let p2 = rows[2].as_ptr();
    let p3 = rows[3].as_ptr();
    let act = activation.as_ptr();
    let cnt = block_count;
    let ok: usize;
    macro_rules! run_4rows_asm {
        ($fused:ident) => {
            // SAFETY: Caller verified every row covers `block_count * 34` bytes and
            // `activation` covers `block_count` `Pq2Activation` elements.
            unsafe {
                core::arch::asm!(
                    "movi v4.2d, #0",
                    "movi v5.16b, #3",
                    "tst {cnt}, #1",
                    "b.eq 1f",
                    pq2_4rows_block_asm!($fused),
                    "sub {cnt}, {cnt}, #1",
                    "cbz {cnt}, 2f",
                    "1:",
                    "prfm pldl1keep, [{p0}, #136]",
                    "prfm pldl1keep, [{p1}, #136]",
                    "prfm pldl1keep, [{p2}, #136]",
                    "prfm pldl1keep, [{p3}, #136]",
                    "prfm pldl1keep, [{act}, #272]",
                    "prfm pldl1keep, [{act}, #336]",
                    pq2_4rows_block_asm!($fused),
                    pq2_4rows_block_asm!($fused),
                    "subs {cnt}, {cnt}, #2",
                    "b.ne 1b",
                    "2:",
                    "umaxv h28, v4.4h",
                    "fmov {t0:w}, s28",
                    "lsr {t0:w}, {t0:w}, #10",
                    "cmp {t0:w}, #30",
                    "cset {ok}, ls",
                    p0 = inout(reg) p0 => _,
                    p1 = inout(reg) p1 => _,
                    p2 = inout(reg) p2 => _,
                    p3 = inout(reg) p3 => _,
                    act = inout(reg) act => _,
                    cnt = inout(reg) cnt => _,
                    t0 = out(reg) _,
                    t1 = out(reg) _,
                    ok = out(reg) ok,
                    inout("v0") sum0,
                    out("v4") _,
                    out("v5") _,
                    out("v6") _,
                    out("v7") _,
                    out("v8") _,
                    out("v9") _,
                    out("v10") _,
                    out("v11") _,
                    out("v12") _,
                    out("v13") _,
                    out("v14") _,
                    out("v15") _,
                    out("v16") _,
                    out("v17") _,
                    out("v18") _,
                    out("v19") _,
                    out("v24") _,
                    out("v25") _,
                    out("v26") _,
                    out("v27") _,
                    out("v28") _,
                    out("v29") _,
                    out("v30") _,
                    out("v31") _,
                    options(nostack, readonly),
                );
            }
        };
    }
    if FUSED {
        run_4rows_asm!(fused);
    } else {
        run_4rows_asm!(nonfused);
    }
    if ok == 0 {
        return None;
    }
    let mut output = [0.0_f32; 4];
    store_f32x4_neon(sum0, &mut output);
    output.iter().all(|v| v.is_finite()).then_some(output)
}

#[expect(
    clippy::too_many_lines,
    reason = "Zero-spill 8-row PQ2 inline-asm decode kernel keeps activation.values in v8..v15 across 2x K-unroll."
)]
#[target_feature(enable = "neon,dotprod")]
fn pq2_scaled_8rows_asm<const FUSED: bool>(
    rows: [&[u8]; 8],
    activation: &[super::Pq2Activation],
) -> Option<[f32; 8]> {
    use core::arch::aarch64::vdupq_n_f32;
    let block_count = activation.len();
    let expected = block_count.checked_mul(34)?;
    if block_count == 0 || rows.iter().any(|r| r.len() != expected) {
        return None;
    }
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let p0 = rows[0].as_ptr();
    let p1 = rows[1].as_ptr();
    let p2 = rows[2].as_ptr();
    let p3 = rows[3].as_ptr();
    let p4 = rows[4].as_ptr();
    let p5 = rows[5].as_ptr();
    let p6 = rows[6].as_ptr();
    let p7 = rows[7].as_ptr();
    let act = activation.as_ptr();
    let cnt = block_count;
    let ok: usize;
    macro_rules! run_8rows_asm {
        ($fused:ident) => {
            // SAFETY: Caller verified every row covers `block_count * 34` bytes and
            // `activation` covers `block_count` `Pq2Activation` elements.
            unsafe {
                core::arch::asm!(
                    "movi v4.2d, #0",
                    "movi v5.16b, #3",
                    "tst {cnt}, #1",
                    "b.eq 1f",
                    pq2_8rows_block_asm!($fused),
                    "sub {cnt}, {cnt}, #1",
                    "cbz {cnt}, 2f",
                    "1:",
                    "prfm pldl1keep, [{p0}, #136]",
                    "prfm pldl1keep, [{p1}, #136]",
                    "prfm pldl1keep, [{p2}, #136]",
                    "prfm pldl1keep, [{p3}, #136]",
                    "prfm pldl1keep, [{p4}, #136]",
                    "prfm pldl1keep, [{p5}, #136]",
                    "prfm pldl1keep, [{p6}, #136]",
                    "prfm pldl1keep, [{p7}, #136]",
                    "prfm pldl1keep, [{act}, #272]",
                    "prfm pldl1keep, [{act}, #336]",
                    pq2_8rows_block_asm!($fused),
                    pq2_8rows_block_asm!($fused),
                    "subs {cnt}, {cnt}, #2",
                    "b.ne 1b",
                    "2:",
                    "umaxv h28, v4.8h",
                    "fmov {t0:w}, s28",
                    "lsr {t0:w}, {t0:w}, #10",
                    "cmp {t0:w}, #30",
                    "cset {ok}, ls",
                    p0 = inout(reg) p0 => _,
                    p1 = inout(reg) p1 => _,
                    p2 = inout(reg) p2 => _,
                    p3 = inout(reg) p3 => _,
                    p4 = inout(reg) p4 => _,
                    p5 = inout(reg) p5 => _,
                    p6 = inout(reg) p6 => _,
                    p7 = inout(reg) p7 => _,
                    act = inout(reg) act => _,
                    cnt = inout(reg) cnt => _,
                    t0 = out(reg) _,
                    t1 = out(reg) _,
                    ok = out(reg) ok,
                    inout("v0") sum0,
                    inout("v1") sum1,
                    out("v4") _,
                    out("v5") _,
                    out("v6") _,
                    out("v7") _,
                    out("v8") _,
                    out("v9") _,
                    out("v10") _,
                    out("v11") _,
                    out("v12") _,
                    out("v13") _,
                    out("v14") _,
                    out("v15") _,
                    out("v16") _,
                    out("v17") _,
                    out("v18") _,
                    out("v19") _,
                    out("v20") _,
                    out("v21") _,
                    out("v22") _,
                    out("v23") _,
                    out("v24") _,
                    out("v25") _,
                    out("v26") _,
                    out("v27") _,
                    out("v28") _,
                    out("v29") _,
                    out("v30") _,
                    out("v31") _,
                    options(nostack, readonly),
                );
            }
        };
    }
    if FUSED {
        run_8rows_asm!(fused);
    } else {
        run_8rows_asm!(nonfused);
    }
    if ok == 0 {
        return None;
    }
    let mut lo = [0.0_f32; 4];
    let mut hi = [0.0_f32; 4];
    store_f32x4_neon(sum0, &mut lo);
    store_f32x4_neon(sum1, &mut hi);
    let output = [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]];
    output.iter().all(|v| v.is_finite()).then_some(output)
}

#[expect(
    clippy::too_many_lines,
    reason = "Zero-spill 16-row PQ2 inline-asm decode kernel keeps activation.values in v8..v15 across both octets and 2x K-unroll."
)]
#[target_feature(enable = "neon,dotprod")]
fn pq2_scaled_16rows_asm<const FUSED: bool>(
    rows: [&[u8]; 16],
    activation: &[super::Pq2Activation],
) -> Option<[f32; 16]> {
    use core::arch::aarch64::vdupq_n_f32;
    let block_count = activation.len();
    let expected = block_count.checked_mul(34)?;
    if block_count == 0 || rows.iter().any(|r| r.len() != expected) {
        return None;
    }
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);
    let p0 = rows[0].as_ptr();
    let p1 = rows[1].as_ptr();
    let p2 = rows[2].as_ptr();
    let p3 = rows[3].as_ptr();
    let p4 = rows[4].as_ptr();
    let p5 = rows[5].as_ptr();
    let p6 = rows[6].as_ptr();
    let p7 = rows[7].as_ptr();
    let p8 = rows[8].as_ptr();
    let p9 = rows[9].as_ptr();
    let p10 = rows[10].as_ptr();
    let p11 = rows[11].as_ptr();
    let p12 = rows[12].as_ptr();
    let p13 = rows[13].as_ptr();
    let p14 = rows[14].as_ptr();
    let p15 = rows[15].as_ptr();
    let act = activation.as_ptr();
    let cnt = block_count;
    let ok: usize;
    macro_rules! run_16rows_asm {
        ($fused:ident) => {
            // SAFETY: Caller verified every row covers `block_count * 34` bytes and
            // `activation` covers `block_count` `Pq2Activation` elements.
            unsafe {
                core::arch::asm!(
                    "movi v4.2d, #0",
                    "movi v5.16b, #3",
                    "tst {cnt}, #1",
                    "b.eq 1f",
                    pq2_16rows_block_asm!($fused),
                    "sub {cnt}, {cnt}, #1",
                    "cbz {cnt}, 2f",
                    "1:",
                    "prfm pldl1keep, [{p0}, #136]",
                    "prfm pldl1keep, [{p1}, #136]",
                    "prfm pldl1keep, [{p2}, #136]",
                    "prfm pldl1keep, [{p3}, #136]",
                    "prfm pldl1keep, [{p4}, #136]",
                    "prfm pldl1keep, [{p5}, #136]",
                    "prfm pldl1keep, [{p6}, #136]",
                    "prfm pldl1keep, [{p7}, #136]",
                    "prfm pldl1keep, [{p8}, #136]",
                    "prfm pldl1keep, [{p9}, #136]",
                    "prfm pldl1keep, [{p10}, #136]",
                    "prfm pldl1keep, [{p11}, #136]",
                    "prfm pldl1keep, [{p12}, #136]",
                    "prfm pldl1keep, [{p13}, #136]",
                    "prfm pldl1keep, [{p14}, #136]",
                    "prfm pldl1keep, [{p15}, #136]",
                    "prfm pldl1keep, [{act}, #272]",
                    "prfm pldl1keep, [{act}, #336]",
                    pq2_16rows_block_asm!($fused),
                    pq2_16rows_block_asm!($fused),
                    "subs {cnt}, {cnt}, #2",
                    "b.ne 1b",
                    "2:",
                    "umaxv h28, v4.8h",
                    "fmov {t0:w}, s28",
                    "lsr {t0:w}, {t0:w}, #10",
                    "cmp {t0:w}, #30",
                    "cset {ok}, ls",
                    p0 = inout(reg) p0 => _,
                    p1 = inout(reg) p1 => _,
                    p2 = inout(reg) p2 => _,
                    p3 = inout(reg) p3 => _,
                    p4 = inout(reg) p4 => _,
                    p5 = inout(reg) p5 => _,
                    p6 = inout(reg) p6 => _,
                    p7 = inout(reg) p7 => _,
                    p8 = inout(reg) p8 => _,
                    p9 = inout(reg) p9 => _,
                    p10 = inout(reg) p10 => _,
                    p11 = inout(reg) p11 => _,
                    p12 = inout(reg) p12 => _,
                    p13 = inout(reg) p13 => _,
                    p14 = inout(reg) p14 => _,
                    p15 = inout(reg) p15 => _,
                    act = inout(reg) act => _,
                    cnt = inout(reg) cnt => _,
                    t0 = out(reg) _,
                    t1 = out(reg) _,
                    ok = out(reg) ok,
                    inout("v0") sum0,
                    inout("v1") sum1,
                    inout("v2") sum2,
                    inout("v3") sum3,
                    out("v4") _,
                    out("v5") _,
                    out("v6") _,
                    out("v7") _,
                    out("v8") _,
                    out("v9") _,
                    out("v10") _,
                    out("v11") _,
                    out("v12") _,
                    out("v13") _,
                    out("v14") _,
                    out("v15") _,
                    out("v16") _,
                    out("v17") _,
                    out("v18") _,
                    out("v19") _,
                    out("v20") _,
                    out("v21") _,
                    out("v22") _,
                    out("v23") _,
                    out("v24") _,
                    out("v25") _,
                    out("v26") _,
                    out("v27") _,
                    out("v28") _,
                    out("v29") _,
                    out("v30") _,
                    out("v31") _,
                    options(nostack, readonly),
                );
            }
        };
    }
    if FUSED {
        run_16rows_asm!(fused);
    } else {
        run_16rows_asm!(nonfused);
    }
    if ok == 0 {
        return None;
    }
    let mut q0 = [0.0_f32; 4];
    let mut q1 = [0.0_f32; 4];
    let mut q2 = [0.0_f32; 4];
    let mut q3 = [0.0_f32; 4];
    store_f32x4_neon(sum0, &mut q0);
    store_f32x4_neon(sum1, &mut q1);
    store_f32x4_neon(sum2, &mut q2);
    store_f32x4_neon(sum3, &mut q3);
    let output = [
        q0[0], q0[1], q0[2], q0[3], q1[0], q1[1], q1[2], q1[3], q2[0], q2[1], q2[2], q2[3], q3[0],
        q3[1], q3[2], q3[3],
    ];
    output.iter().all(|v| v.is_finite()).then_some(output)
}

/// All extents and activation scales were checked by the safe parent boundary.
#[target_feature(enable = "neon,dotprod")]
fn pq2_scaled_rows<const ROWS: usize, const FUSED: bool>(
    rows: [&[u8]; ROWS],
    activation: &[super::Pq2Activation],
) -> Option<[f32; ROWS]> {
    match ROWS {
        1 => {
            let r1: [&[u8]; 1] = rows.as_slice().try_into().ok()?;
            let res = pq2_scaled_1row_asm::<FUSED>(r1[0], activation)?;
            res.as_slice().try_into().ok()
        }
        4 => {
            let r4: [&[u8]; 4] = rows.as_slice().try_into().ok()?;
            let res = pq2_scaled_4rows_asm::<FUSED>(r4, activation)?;
            res.as_slice().try_into().ok()
        }
        8 => {
            let r8: [&[u8]; 8] = rows.as_slice().try_into().ok()?;
            let res = pq2_scaled_8rows_asm::<FUSED>(r8, activation)?;
            res.as_slice().try_into().ok()
        }
        16 => {
            let r16: [&[u8]; 16] = rows.as_slice().try_into().ok()?;
            let res = pq2_scaled_16rows_asm::<FUSED>(r16, activation)?;
            res.as_slice().try_into().ok()
        }
        _ => None,
    }
}

#[inline]
pub(super) fn pq2_detected_scaled_rows<const ROWS: usize, const FUSED: bool>(
    _detected: super::Pq2Dotprod,
    rows: [&[u8]; ROWS],
    activation: &[super::Pq2Activation],
) -> Option<[f32; ROWS]> {
    // SAFETY: the unforgeable capability establishes CPU/OS `SDOT` eligibility;
    // the safe parent checked complete row extents and finite activation scales.
    unsafe { pq2_scaled_rows::<ROWS, FUSED>(rows, activation) }
}

#[inline]
pub(super) fn pq2_detected_dot(
    _detected: super::Pq2Dotprod,
    packed: &[u8; 32],
    planes: &[i8; 128],
    activation_sum: i32,
) -> i32 {
    // SAFETY: The unforgeable capability was constructed only after CPU/OS
    // `dotprod` detection. Array references provide every required input byte.
    unsafe { dot_pq2_consumption_dotprod(packed, planes, activation_sum) }
}

#[inline]
pub(super) fn pq2_detected_dot_four(
    _detected: super::Pq2Dotprod,
    packed: [&[u8; 32]; 4],
    planes: &[i8; 128],
    activation_sum: i32,
) -> [i32; 4] {
    // SAFETY: The unforgeable capability retains verified CPU/OS `dotprod`
    // eligibility, and all inputs are valid fixed-size array references.
    unsafe { dot_pq2_four_consumption_dotprod(packed, planes, activation_sum) }
}

#[inline]
pub(super) fn pq2_detected_dot_eight(
    _detected: super::Pq2Dotprod,
    packed: [&[u8; 32]; 8],
    planes: &[i8; 128],
    activation_sum: i32,
) -> [i32; 8] {
    // SAFETY: The unforgeable capability retains verified CPU/OS `dotprod`
    // eligibility, and all inputs are valid fixed-size array references.
    unsafe { dot_pq2_eight_consumption_dotprod(packed, planes, activation_sum) }
}

#[inline]
pub(super) fn dot_pq2_eight_consumption_precomputed(
    packed: [&[u8; 32]; 8],
    planes: &[i8; 128],
    activation_sum: i32,
) -> Option<[i32; 8]> {
    if !has_dotprod() {
        return None;
    }
    // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
    Some(unsafe { dot_pq2_eight_consumption_dotprod(packed, planes, activation_sum) })
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

#[target_feature(enable = "neon,i8mm")]
fn smmla_q8_0_4x2_i8mm(rows: [&[u8; 32]; 4], cols: [&[i8; 32]; 2]) -> [[i32; 2]; 4] {
    let r_halves = rows.map(|r| r.as_chunks::<16>().0);
    let c0_halves = cols[0].as_chunks::<16>().0;
    let c1_halves = cols[1].as_chunks::<16>().0;

    let mut acc0 = vdupq_n_s32(0);
    let mut acc1 = vdupq_n_s32(0);
    for half in 0..2 {
        let a0 = load_s8x16(&c0_halves[half]);
        let a1 = load_s8x16(&c1_halves[half]);
        let a_lo = vcombine_s8(vget_low_s8(a0), vget_low_s8(a1));
        let a_hi = vcombine_s8(vget_high_s8(a0), vget_high_s8(a1));

        let w0 = vreinterpretq_s8_u8(load_u8x16(&r_halves[0][half]));
        let w1 = vreinterpretq_s8_u8(load_u8x16(&r_halves[1][half]));
        let w2 = vreinterpretq_s8_u8(load_u8x16(&r_halves[2][half]));
        let w3 = vreinterpretq_s8_u8(load_u8x16(&r_halves[3][half]));

        let w01_lo = vcombine_s8(vget_low_s8(w0), vget_low_s8(w1));
        let w01_hi = vcombine_s8(vget_high_s8(w0), vget_high_s8(w1));
        let w23_lo = vcombine_s8(vget_low_s8(w2), vget_low_s8(w3));
        let w23_hi = vcombine_s8(vget_high_s8(w2), vget_high_s8(w3));

        acc0 = smmla_s32(smmla_s32(acc0, w01_lo, a_lo), w01_hi, a_hi);
        acc1 = smmla_s32(smmla_s32(acc1, w23_lo, a_lo), w23_hi, a_hi);
    }
    let mut q0 = [0_i32; 4];
    let mut q1 = [0_i32; 4];
    store_s32x4(acc0, &mut q0);
    store_s32x4(acc1, &mut q1);
    [
        [q0[0], q0[1]],
        [q0[2], q0[3]],
        [q1[0], q1[1]],
        [q1[2], q1[3]],
    ]
}

#[target_feature(enable = "neon,dotprod")]
fn smmla_q8_0_4x2_dotprod(rows: [&[u8; 32]; 4], cols: [&[i8; 32]; 2]) -> [[i32; 2]; 4] {
    let c0_halves = cols[0].as_chunks::<16>().0;
    let c1_halves = cols[1].as_chunks::<16>().0;
    let a00 = load_s8x16(&c0_halves[0]);
    let a01 = load_s8x16(&c0_halves[1]);
    let a10 = load_s8x16(&c1_halves[0]);
    let a11 = load_s8x16(&c1_halves[1]);
    let mut out = [[0_i32; 2]; 4];
    for (r, row) in rows.into_iter().enumerate() {
        let w_halves = row.as_chunks::<16>().0;
        let w0 = vreinterpretq_s8_u8(load_u8x16(&w_halves[0]));
        let w1 = vreinterpretq_s8_u8(load_u8x16(&w_halves[1]));
        out[r] = [
            vaddvq_s32(sdot32_vec(w0, w1, a00, a01)),
            vaddvq_s32(sdot32_vec(w0, w1, a10, a11)),
        ];
    }
    out
}

#[target_feature(enable = "neon,i8mm")]
fn smmla_q8_0_4x4_i8mm(rows: [&[u8; 32]; 4], cols: [&[i8; 32]; 4]) -> [[i32; 4]; 4] {
    let r_halves = rows.map(|r| r.as_chunks::<16>().0);
    let c_halves = cols.map(|c| c.as_chunks::<16>().0);

    let mut acc00 = vdupq_n_s32(0);
    let mut acc01 = vdupq_n_s32(0);
    let mut acc10 = vdupq_n_s32(0);
    let mut acc11 = vdupq_n_s32(0);
    for half in 0..2 {
        let a0 = load_s8x16(&c_halves[0][half]);
        let a1 = load_s8x16(&c_halves[1][half]);
        let a2 = load_s8x16(&c_halves[2][half]);
        let a3 = load_s8x16(&c_halves[3][half]);
        let a01_lo = vcombine_s8(vget_low_s8(a0), vget_low_s8(a1));
        let a01_hi = vcombine_s8(vget_high_s8(a0), vget_high_s8(a1));
        let a23_lo = vcombine_s8(vget_low_s8(a2), vget_low_s8(a3));
        let a23_hi = vcombine_s8(vget_high_s8(a2), vget_high_s8(a3));

        let w0 = vreinterpretq_s8_u8(load_u8x16(&r_halves[0][half]));
        let w1 = vreinterpretq_s8_u8(load_u8x16(&r_halves[1][half]));
        let w2 = vreinterpretq_s8_u8(load_u8x16(&r_halves[2][half]));
        let w3 = vreinterpretq_s8_u8(load_u8x16(&r_halves[3][half]));
        let w01_lo = vcombine_s8(vget_low_s8(w0), vget_low_s8(w1));
        let w01_hi = vcombine_s8(vget_high_s8(w0), vget_high_s8(w1));
        let w23_lo = vcombine_s8(vget_low_s8(w2), vget_low_s8(w3));
        let w23_hi = vcombine_s8(vget_high_s8(w2), vget_high_s8(w3));

        acc00 = smmla_s32(smmla_s32(acc00, w01_lo, a01_lo), w01_hi, a01_hi);
        acc01 = smmla_s32(smmla_s32(acc01, w01_lo, a23_lo), w01_hi, a23_hi);
        acc10 = smmla_s32(smmla_s32(acc10, w23_lo, a01_lo), w23_hi, a01_hi);
        acc11 = smmla_s32(smmla_s32(acc11, w23_lo, a23_lo), w23_hi, a23_hi);
    }
    let mut out = [[0_i32; 4]; 4];
    store_s32x4(
        vcombine_s32(vget_low_s32(acc00), vget_low_s32(acc01)),
        &mut out[0],
    );
    store_s32x4(
        vcombine_s32(vget_high_s32(acc00), vget_high_s32(acc01)),
        &mut out[1],
    );
    store_s32x4(
        vcombine_s32(vget_low_s32(acc10), vget_low_s32(acc11)),
        &mut out[2],
    );
    store_s32x4(
        vcombine_s32(vget_high_s32(acc10), vget_high_s32(acc11)),
        &mut out[3],
    );
    out
}

#[target_feature(enable = "neon,dotprod")]
fn smmla_q8_0_4x4_dotprod(rows: [&[u8; 32]; 4], cols: [&[i8; 32]; 4]) -> [[i32; 4]; 4] {
    let acts: [(int8x16_t, int8x16_t); 4] = cols.map(|col| {
        let halves = col.as_chunks::<16>().0;
        (load_s8x16(&halves[0]), load_s8x16(&halves[1]))
    });
    let mut out = [[0_i32; 4]; 4];
    for (r, row) in rows.into_iter().enumerate() {
        let w_halves = row.as_chunks::<16>().0;
        let w0 = vreinterpretq_s8_u8(load_u8x16(&w_halves[0]));
        let w1 = vreinterpretq_s8_u8(load_u8x16(&w_halves[1]));
        let l0 = sdot32_vec(w0, w1, acts[0].0, acts[0].1);
        let l1 = sdot32_vec(w0, w1, acts[1].0, acts[1].1);
        let l2 = sdot32_vec(w0, w1, acts[2].0, acts[2].1);
        let l3 = sdot32_vec(w0, w1, acts[3].0, acts[3].1);
        store_s32x4(reduce4_s32x4_addp(l0, l1, l2, l3), &mut out[r]);
    }
    out
}

#[target_feature(enable = "neon,i8mm")]
fn smmla_q8_0_8x2_i8mm(rows: [&[u8; 32]; 8], cols: [&[i8; 32]; 2]) -> [[i32; 2]; 8] {
    let r_halves = rows.map(|r| r.as_chunks::<16>().0);
    let c0_halves = cols[0].as_chunks::<16>().0;
    let c1_halves = cols[1].as_chunks::<16>().0;

    let mut acc0 = vdupq_n_s32(0);
    let mut acc1 = vdupq_n_s32(0);
    let mut acc2 = vdupq_n_s32(0);
    let mut acc3 = vdupq_n_s32(0);
    for half in 0..2 {
        let a0 = load_s8x16(&c0_halves[half]);
        let a1 = load_s8x16(&c1_halves[half]);
        let a_lo = vcombine_s8(vget_low_s8(a0), vget_low_s8(a1));
        let a_hi = vcombine_s8(vget_high_s8(a0), vget_high_s8(a1));

        let w0 = vreinterpretq_s8_u8(load_u8x16(&r_halves[0][half]));
        let w1 = vreinterpretq_s8_u8(load_u8x16(&r_halves[1][half]));
        let w2 = vreinterpretq_s8_u8(load_u8x16(&r_halves[2][half]));
        let w3 = vreinterpretq_s8_u8(load_u8x16(&r_halves[3][half]));
        let w01_lo = vcombine_s8(vget_low_s8(w0), vget_low_s8(w1));
        let w01_hi = vcombine_s8(vget_high_s8(w0), vget_high_s8(w1));
        let w23_lo = vcombine_s8(vget_low_s8(w2), vget_low_s8(w3));
        let w23_hi = vcombine_s8(vget_high_s8(w2), vget_high_s8(w3));

        acc0 = smmla_s32(smmla_s32(acc0, w01_lo, a_lo), w01_hi, a_hi);
        acc1 = smmla_s32(smmla_s32(acc1, w23_lo, a_lo), w23_hi, a_hi);

        let w4 = vreinterpretq_s8_u8(load_u8x16(&r_halves[4][half]));
        let w5 = vreinterpretq_s8_u8(load_u8x16(&r_halves[5][half]));
        let w6 = vreinterpretq_s8_u8(load_u8x16(&r_halves[6][half]));
        let w7 = vreinterpretq_s8_u8(load_u8x16(&r_halves[7][half]));
        let w45_lo = vcombine_s8(vget_low_s8(w4), vget_low_s8(w5));
        let w45_hi = vcombine_s8(vget_high_s8(w4), vget_high_s8(w5));
        let w67_lo = vcombine_s8(vget_low_s8(w6), vget_low_s8(w7));
        let w67_hi = vcombine_s8(vget_high_s8(w6), vget_high_s8(w7));

        acc2 = smmla_s32(smmla_s32(acc2, w45_lo, a_lo), w45_hi, a_hi);
        acc3 = smmla_s32(smmla_s32(acc3, w67_lo, a_lo), w67_hi, a_hi);
    }
    let mut q0 = [0_i32; 4];
    let mut q1 = [0_i32; 4];
    let mut q2 = [0_i32; 4];
    let mut q3 = [0_i32; 4];
    store_s32x4(acc0, &mut q0);
    store_s32x4(acc1, &mut q1);
    store_s32x4(acc2, &mut q2);
    store_s32x4(acc3, &mut q3);
    [
        [q0[0], q0[1]],
        [q0[2], q0[3]],
        [q1[0], q1[1]],
        [q1[2], q1[3]],
        [q2[0], q2[1]],
        [q2[2], q2[3]],
        [q3[0], q3[1]],
        [q3[2], q3[3]],
    ]
}

#[target_feature(enable = "neon,dotprod")]
fn smmla_q8_0_8x2_dotprod(rows: [&[u8; 32]; 8], cols: [&[i8; 32]; 2]) -> [[i32; 2]; 8] {
    let c0_halves = cols[0].as_chunks::<16>().0;
    let c1_halves = cols[1].as_chunks::<16>().0;
    let a00 = load_s8x16(&c0_halves[0]);
    let a01 = load_s8x16(&c0_halves[1]);
    let a10 = load_s8x16(&c1_halves[0]);
    let a11 = load_s8x16(&c1_halves[1]);
    let mut out = [[0_i32; 2]; 8];
    for (r, row) in rows.into_iter().enumerate() {
        let w_halves = row.as_chunks::<16>().0;
        let w0 = vreinterpretq_s8_u8(load_u8x16(&w_halves[0]));
        let w1 = vreinterpretq_s8_u8(load_u8x16(&w_halves[1]));
        out[r] = [
            vaddvq_s32(sdot32_vec(w0, w1, a00, a01)),
            vaddvq_s32(sdot32_vec(w0, w1, a10, a11)),
        ];
    }
    out
}

#[target_feature(enable = "neon,i8mm")]
fn smmla_q8_0_8x4_i8mm(rows: [&[u8; 32]; 8], cols: [&[i8; 32]; 4]) -> [[i32; 4]; 8] {
    let r_halves = rows.map(|r| r.as_chunks::<16>().0);
    let c_halves = cols.map(|c| c.as_chunks::<16>().0);

    let mut acc00 = vdupq_n_s32(0);
    let mut acc01 = vdupq_n_s32(0);
    let mut acc10 = vdupq_n_s32(0);
    let mut acc11 = vdupq_n_s32(0);
    let mut acc20 = vdupq_n_s32(0);
    let mut acc21 = vdupq_n_s32(0);
    let mut acc30 = vdupq_n_s32(0);
    let mut acc31 = vdupq_n_s32(0);
    for half in 0..2 {
        let a0 = load_s8x16(&c_halves[0][half]);
        let a1 = load_s8x16(&c_halves[1][half]);
        let a2 = load_s8x16(&c_halves[2][half]);
        let a3 = load_s8x16(&c_halves[3][half]);
        let a01_lo = vcombine_s8(vget_low_s8(a0), vget_low_s8(a1));
        let a01_hi = vcombine_s8(vget_high_s8(a0), vget_high_s8(a1));
        let a23_lo = vcombine_s8(vget_low_s8(a2), vget_low_s8(a3));
        let a23_hi = vcombine_s8(vget_high_s8(a2), vget_high_s8(a3));

        let w0 = vreinterpretq_s8_u8(load_u8x16(&r_halves[0][half]));
        let w1 = vreinterpretq_s8_u8(load_u8x16(&r_halves[1][half]));
        let w2 = vreinterpretq_s8_u8(load_u8x16(&r_halves[2][half]));
        let w3 = vreinterpretq_s8_u8(load_u8x16(&r_halves[3][half]));
        let w01_lo = vcombine_s8(vget_low_s8(w0), vget_low_s8(w1));
        let w01_hi = vcombine_s8(vget_high_s8(w0), vget_high_s8(w1));
        let w23_lo = vcombine_s8(vget_low_s8(w2), vget_low_s8(w3));
        let w23_hi = vcombine_s8(vget_high_s8(w2), vget_high_s8(w3));

        acc00 = smmla_s32(smmla_s32(acc00, w01_lo, a01_lo), w01_hi, a01_hi);
        acc01 = smmla_s32(smmla_s32(acc01, w01_lo, a23_lo), w01_hi, a23_hi);
        acc10 = smmla_s32(smmla_s32(acc10, w23_lo, a01_lo), w23_hi, a01_hi);
        acc11 = smmla_s32(smmla_s32(acc11, w23_lo, a23_lo), w23_hi, a23_hi);

        let w4 = vreinterpretq_s8_u8(load_u8x16(&r_halves[4][half]));
        let w5 = vreinterpretq_s8_u8(load_u8x16(&r_halves[5][half]));
        let w6 = vreinterpretq_s8_u8(load_u8x16(&r_halves[6][half]));
        let w7 = vreinterpretq_s8_u8(load_u8x16(&r_halves[7][half]));
        let w45_lo = vcombine_s8(vget_low_s8(w4), vget_low_s8(w5));
        let w45_hi = vcombine_s8(vget_high_s8(w4), vget_high_s8(w5));
        let w67_lo = vcombine_s8(vget_low_s8(w6), vget_low_s8(w7));
        let w67_hi = vcombine_s8(vget_high_s8(w6), vget_high_s8(w7));

        acc20 = smmla_s32(smmla_s32(acc20, w45_lo, a01_lo), w45_hi, a01_hi);
        acc21 = smmla_s32(smmla_s32(acc21, w45_lo, a23_lo), w45_hi, a23_hi);
        acc30 = smmla_s32(smmla_s32(acc30, w67_lo, a01_lo), w67_hi, a01_hi);
        acc31 = smmla_s32(smmla_s32(acc31, w67_lo, a23_lo), w67_hi, a23_hi);
    }
    let mut out = [[0_i32; 4]; 8];
    for (pair_idx, (acc_lo, acc_hi)) in [
        (acc00, acc01),
        (acc10, acc11),
        (acc20, acc21),
        (acc30, acc31),
    ]
    .into_iter()
    .enumerate()
    {
        let r0 = pair_idx * 2;
        store_s32x4(
            vcombine_s32(vget_low_s32(acc_lo), vget_low_s32(acc_hi)),
            &mut out[r0],
        );
        store_s32x4(
            vcombine_s32(vget_high_s32(acc_lo), vget_high_s32(acc_hi)),
            &mut out[r0 + 1],
        );
    }
    out
}

#[target_feature(enable = "neon,dotprod")]
fn smmla_q8_0_8x4_dotprod(rows: [&[u8; 32]; 8], cols: [&[i8; 32]; 4]) -> [[i32; 4]; 8] {
    let acts: [(int8x16_t, int8x16_t); 4] = cols.map(|col| {
        let halves = col.as_chunks::<16>().0;
        (load_s8x16(&halves[0]), load_s8x16(&halves[1]))
    });
    let mut out = [[0_i32; 4]; 8];
    for (r, row) in rows.into_iter().enumerate() {
        let w_halves = row.as_chunks::<16>().0;
        let w0 = vreinterpretq_s8_u8(load_u8x16(&w_halves[0]));
        let w1 = vreinterpretq_s8_u8(load_u8x16(&w_halves[1]));
        let l0 = sdot32_vec(w0, w1, acts[0].0, acts[0].1);
        let l1 = sdot32_vec(w0, w1, acts[1].0, acts[1].1);
        let l2 = sdot32_vec(w0, w1, acts[2].0, acts[2].1);
        let l3 = sdot32_vec(w0, w1, acts[3].0, acts[3].1);
        store_s32x4(reduce4_s32x4_addp(l0, l1, l2, l3), &mut out[r]);
    }
    out
}

#[inline]
pub(super) fn smmla_q8_0_4x2(rows: [&[u8; 32]; 4], cols: [&[i8; 32]; 2]) -> Option<[[i32; 2]; 4]> {
    if has_i8mm() {
        // SAFETY: `has_i8mm()` verified that the CPU supports `i8mm` and baseline `neon`.
        return Some(unsafe { smmla_q8_0_4x2_i8mm(rows, cols) });
    }
    if has_dotprod() {
        // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
        return Some(unsafe { smmla_q8_0_4x2_dotprod(rows, cols) });
    }
    None
}

#[inline]
pub(super) fn smmla_q8_0_4x4(rows: [&[u8; 32]; 4], cols: [&[i8; 32]; 4]) -> Option<[[i32; 4]; 4]> {
    if has_i8mm() {
        // SAFETY: `has_i8mm()` verified that the CPU supports `i8mm` and baseline `neon`.
        return Some(unsafe { smmla_q8_0_4x4_i8mm(rows, cols) });
    }
    if has_dotprod() {
        // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
        return Some(unsafe { smmla_q8_0_4x4_dotprod(rows, cols) });
    }
    None
}

#[inline]
pub(super) fn smmla_q8_0_8x2(rows: [&[u8; 32]; 8], cols: [&[i8; 32]; 2]) -> Option<[[i32; 2]; 8]> {
    if has_i8mm() {
        // SAFETY: `has_i8mm()` verified that the CPU supports `i8mm` and baseline `neon`.
        return Some(unsafe { smmla_q8_0_8x2_i8mm(rows, cols) });
    }
    if has_dotprod() {
        // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
        return Some(unsafe { smmla_q8_0_8x2_dotprod(rows, cols) });
    }
    None
}

#[inline]
pub(super) fn smmla_q8_0_8x4(rows: [&[u8; 32]; 8], cols: [&[i8; 32]; 4]) -> Option<[[i32; 4]; 8]> {
    if has_i8mm() {
        // SAFETY: `has_i8mm()` verified that the CPU supports `i8mm` and baseline `neon`.
        return Some(unsafe { smmla_q8_0_8x4_i8mm(rows, cols) });
    }
    if has_dotprod() {
        // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
        return Some(unsafe { smmla_q8_0_8x4_dotprod(rows, cols) });
    }
    None
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
        reduce4_s32x4_addp(acc[0], acc[1], acc[2], acc[3]),
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
            reduce4_s32x4_addp(acc[r][0], acc[r][1], acc[r][2], acc[r][3]),
            sums_vec,
        );
        store_s32x4(reduced, &mut out[r]);
    }
    out
}

#[inline]
#[target_feature(enable = "neon")]
fn pair_s8_lo_hi(first: int8x16_t, second: int8x16_t) -> (int8x16_t, int8x16_t) {
    (
        vcombine_s8(vget_low_s8(first), vget_low_s8(second)),
        vcombine_s8(vget_high_s8(first), vget_high_s8(second)),
    )
}

#[target_feature(enable = "neon,i8mm")]
fn dot_pq2_4x2_act_pairs_i8mm(
    packed: [&[u8; 32]; 4],
    act_pairs: &[(int8x16_t, int8x16_t); 8],
    sums_vec: int32x4_t,
) -> [[i32; 2]; 4] {
    let weights0 = unpack_pq2_unsigned_s8(packed[0]);
    let weights1 = unpack_pq2_unsigned_s8(packed[1]);
    let weights2 = unpack_pq2_unsigned_s8(packed[2]);
    let weights3 = unpack_pq2_unsigned_s8(packed[3]);
    let mut acc01 = vdupq_n_s32(0);
    let mut acc23 = vdupq_n_s32(0);
    for (plane, &(act_lo, act_hi)) in act_pairs.iter().enumerate() {
        let (w01_lo, w01_hi) = pair_s8_lo_hi(weights0[plane], weights1[plane]);
        let (w23_lo, w23_hi) = pair_s8_lo_hi(weights2[plane], weights3[plane]);
        acc01 = smmla_s32(smmla_s32(acc01, w01_lo, act_lo), w01_hi, act_hi);
        acc23 = smmla_s32(smmla_s32(acc23, w23_lo, act_lo), w23_hi, act_hi);
    }
    let mut tile01 = [0_i32; 4];
    let mut tile23 = [0_i32; 4];
    store_s32x4(vsubq_s32(acc01, sums_vec), &mut tile01);
    store_s32x4(vsubq_s32(acc23, sums_vec), &mut tile23);
    [
        [tile01[0], tile01[1]],
        [tile01[2], tile01[3]],
        [tile23[0], tile23[1]],
        [tile23[2], tile23[3]],
    ]
}

#[target_feature(enable = "neon,i8mm")]
fn dot_pq2_4x2_planes_i8mm(
    packed: [&[u8; 32]; 4],
    planes: [&[int8x16_t; 8]; 2],
    activation_sums: [i32; 2],
) -> [[i32; 2]; 4] {
    let act_pairs: [(int8x16_t, int8x16_t); 8] =
        std::array::from_fn(|plane| pair_s8_lo_hi(planes[0][plane], planes[1][plane]));
    let sums_vec = load_s32x4(&[
        activation_sums[0],
        activation_sums[1],
        activation_sums[0],
        activation_sums[1],
    ]);
    dot_pq2_4x2_act_pairs_i8mm(packed, &act_pairs, sums_vec)
}

#[target_feature(enable = "neon,i8mm")]
fn dot_pq2_8x2_planes_i8mm(
    packed: [&[u8; 32]; 8],
    planes: [&[int8x16_t; 8]; 2],
    activation_sums: [i32; 2],
) -> [[i32; 2]; 8] {
    let act_pairs: [(int8x16_t, int8x16_t); 8] =
        std::array::from_fn(|plane| pair_s8_lo_hi(planes[0][plane], planes[1][plane]));
    let sums_vec = load_s32x4(&[
        activation_sums[0],
        activation_sums[1],
        activation_sums[0],
        activation_sums[1],
    ]);
    let lo = dot_pq2_4x2_act_pairs_i8mm(
        [packed[0], packed[1], packed[2], packed[3]],
        &act_pairs,
        sums_vec,
    );
    let hi = dot_pq2_4x2_act_pairs_i8mm(
        [packed[4], packed[5], packed[6], packed[7]],
        &act_pairs,
        sums_vec,
    );
    [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]]
}

#[target_feature(enable = "neon,dotprod")]
fn dot_pq2_4x2_planes_dotprod(
    packed: [&[u8; 32]; 4],
    planes: [&[int8x16_t; 8]; 2],
    activation_sums: [i32; 2],
) -> [[i32; 2]; 4] {
    let weights = [
        unpack_pq2_unsigned_s8(packed[0]),
        unpack_pq2_unsigned_s8(packed[1]),
        unpack_pq2_unsigned_s8(packed[2]),
        unpack_pq2_unsigned_s8(packed[3]),
    ];
    let mut acc = [[vdupq_n_s32(0); 2]; 4];
    for plane in 0..8 {
        let act0 = planes[0][plane];
        let act1 = planes[1][plane];
        for (row_weights, row_acc) in weights.iter().zip(&mut acc) {
            row_acc[0] = vdotq_s32(row_acc[0], row_weights[plane], act0);
            row_acc[1] = vdotq_s32(row_acc[1], row_weights[plane], act1);
        }
    }
    let sums_vec = load_s32x4(&[
        activation_sums[0],
        activation_sums[1],
        activation_sums[0],
        activation_sums[1],
    ]);
    let mut tile01 = [0_i32; 4];
    let mut tile23 = [0_i32; 4];
    store_s32x4(
        vsubq_s32(
            reduce4_s32x4_addp(acc[0][0], acc[0][1], acc[1][0], acc[1][1]),
            sums_vec,
        ),
        &mut tile01,
    );
    store_s32x4(
        vsubq_s32(
            reduce4_s32x4_addp(acc[2][0], acc[2][1], acc[3][0], acc[3][1]),
            sums_vec,
        ),
        &mut tile23,
    );
    [
        [tile01[0], tile01[1]],
        [tile01[2], tile01[3]],
        [tile23[0], tile23[1]],
        [tile23[2], tile23[3]],
    ]
}

#[target_feature(enable = "neon,dotprod")]
fn dot_pq2_8x2_planes_dotprod(
    packed: [&[u8; 32]; 8],
    planes: [&[int8x16_t; 8]; 2],
    activation_sums: [i32; 2],
) -> [[i32; 2]; 8] {
    let lo = dot_pq2_4x2_planes_dotprod(
        [packed[0], packed[1], packed[2], packed[3]],
        planes,
        activation_sums,
    );
    let hi = dot_pq2_4x2_planes_dotprod(
        [packed[4], packed[5], packed[6], packed[7]],
        planes,
        activation_sums,
    );
    [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]]
}

#[target_feature(enable = "neon,i8mm")]
fn dot_pq2_four_rows_two_tokens_precomputed_i8mm(
    packed: [&[u8; 32]; 4],
    planes: [&[i8; 128]; 2],
    activation_sums: [i32; 2],
) -> [[i32; 2]; 4] {
    let p0 = load_pq2_planes_s8(planes[0]);
    let p1 = load_pq2_planes_s8(planes[1]);
    dot_pq2_4x2_planes_i8mm(packed, [&p0, &p1], activation_sums)
}

#[target_feature(enable = "neon,dotprod")]
fn dot_pq2_four_rows_two_tokens_precomputed_dotprod(
    packed: [&[u8; 32]; 4],
    planes: [&[i8; 128]; 2],
    activation_sums: [i32; 2],
) -> [[i32; 2]; 4] {
    let p0 = load_pq2_planes_s8(planes[0]);
    let p1 = load_pq2_planes_s8(planes[1]);
    dot_pq2_4x2_planes_dotprod(packed, [&p0, &p1], activation_sums)
}

#[target_feature(enable = "neon,i8mm")]
fn dot_pq2_eight_rows_two_tokens_precomputed_i8mm(
    packed: [&[u8; 32]; 8],
    planes: [&[i8; 128]; 2],
    activation_sums: [i32; 2],
) -> [[i32; 2]; 8] {
    let p0 = load_pq2_planes_s8(planes[0]);
    let p1 = load_pq2_planes_s8(planes[1]);
    dot_pq2_8x2_planes_i8mm(packed, [&p0, &p1], activation_sums)
}

#[target_feature(enable = "neon,dotprod")]
fn dot_pq2_eight_rows_two_tokens_precomputed_dotprod(
    packed: [&[u8; 32]; 8],
    planes: [&[i8; 128]; 2],
    activation_sums: [i32; 2],
) -> [[i32; 2]; 8] {
    let p0 = load_pq2_planes_s8(planes[0]);
    let p1 = load_pq2_planes_s8(planes[1]);
    dot_pq2_8x2_planes_dotprod(packed, [&p0, &p1], activation_sums)
}

#[target_feature(enable = "neon,i8mm")]
fn dot_pq2_four_rows_two_tokens_i8mm(
    packed: [&[u8; 32]; 4],
    acts: [&[i8; 128]; 2],
    activation_sums: [i32; 2],
) -> [[i32; 2]; 4] {
    let p0 = deinterleave_pq2_planes_s8(acts[0]);
    let p1 = deinterleave_pq2_planes_s8(acts[1]);
    dot_pq2_4x2_planes_i8mm(packed, [&p0, &p1], activation_sums)
}

#[target_feature(enable = "neon,dotprod")]
fn dot_pq2_four_rows_two_tokens_dotprod(
    packed: [&[u8; 32]; 4],
    acts: [&[i8; 128]; 2],
    activation_sums: [i32; 2],
) -> [[i32; 2]; 4] {
    let p0 = deinterleave_pq2_planes_s8(acts[0]);
    let p1 = deinterleave_pq2_planes_s8(acts[1]);
    dot_pq2_4x2_planes_dotprod(packed, [&p0, &p1], activation_sums)
}

#[target_feature(enable = "neon,i8mm")]
fn dot_pq2_eight_rows_two_tokens_i8mm(
    packed: [&[u8; 32]; 8],
    acts: [&[i8; 128]; 2],
    activation_sums: [i32; 2],
) -> [[i32; 2]; 8] {
    let p0 = deinterleave_pq2_planes_s8(acts[0]);
    let p1 = deinterleave_pq2_planes_s8(acts[1]);
    dot_pq2_8x2_planes_i8mm(packed, [&p0, &p1], activation_sums)
}

#[target_feature(enable = "neon,dotprod")]
fn dot_pq2_eight_rows_two_tokens_dotprod(
    packed: [&[u8; 32]; 8],
    acts: [&[i8; 128]; 2],
    activation_sums: [i32; 2],
) -> [[i32; 2]; 8] {
    let p0 = deinterleave_pq2_planes_s8(acts[0]);
    let p1 = deinterleave_pq2_planes_s8(acts[1]);
    dot_pq2_8x2_planes_dotprod(packed, [&p0, &p1], activation_sums)
}

#[inline]
pub(super) fn pq2_detected_dot_four_rows_two_tokens(
    detected: super::Pq2Dotprod,
    packed: [&[u8; 32]; 4],
    planes: [&[i8; 128]; 2],
    activation_sums: [i32; 2],
) -> [[i32; 2]; 4] {
    if detected.i8mm {
        // SAFETY: `detected.i8mm` was established by `has_dotprod() && has_i8mm()`.
        return unsafe {
            dot_pq2_four_rows_two_tokens_precomputed_i8mm(packed, planes, activation_sums)
        };
    }
    // SAFETY: `Pq2Dotprod` is only constructed after `has_dotprod()` succeeds.
    unsafe { dot_pq2_four_rows_two_tokens_precomputed_dotprod(packed, planes, activation_sums) }
}

#[inline]
pub(super) fn pq2_detected_dot_eight_rows_two_tokens(
    detected: super::Pq2Dotprod,
    packed: [&[u8; 32]; 8],
    planes: [&[i8; 128]; 2],
    activation_sums: [i32; 2],
) -> [[i32; 2]; 8] {
    if detected.i8mm {
        // SAFETY: `detected.i8mm` was established by `has_dotprod() && has_i8mm()`.
        return unsafe {
            dot_pq2_eight_rows_two_tokens_precomputed_i8mm(packed, planes, activation_sums)
        };
    }
    // SAFETY: `Pq2Dotprod` is only constructed after `has_dotprod()` succeeds.
    unsafe { dot_pq2_eight_rows_two_tokens_precomputed_dotprod(packed, planes, activation_sums) }
}

#[inline]
pub(super) fn dot_pq2_four_rows_two_tokens(
    packed: [&[u8; 32]; 4],
    acts: [&[i8; 128]; 2],
    activation_sums: [i32; 2],
) -> Option<[[i32; 2]; 4]> {
    if has_i8mm() && has_dotprod() {
        // SAFETY: `has_i8mm() && has_dotprod()` verified CPU support.
        return Some(unsafe { dot_pq2_four_rows_two_tokens_i8mm(packed, acts, activation_sums) });
    }
    if has_dotprod() {
        // SAFETY: `has_dotprod()` verified CPU support.
        return Some(unsafe {
            dot_pq2_four_rows_two_tokens_dotprod(packed, acts, activation_sums)
        });
    }
    None
}

#[inline]
pub(super) fn dot_pq2_eight_rows_two_tokens(
    packed: [&[u8; 32]; 8],
    acts: [&[i8; 128]; 2],
    activation_sums: [i32; 2],
) -> Option<[[i32; 2]; 8]> {
    if has_i8mm() && has_dotprod() {
        // SAFETY: `has_i8mm() && has_dotprod()` verified CPU support.
        return Some(unsafe { dot_pq2_eight_rows_two_tokens_i8mm(packed, acts, activation_sums) });
    }
    if has_dotprod() {
        // SAFETY: `has_dotprod()` verified CPU support.
        return Some(unsafe {
            dot_pq2_eight_rows_two_tokens_dotprod(packed, acts, activation_sums)
        });
    }
    None
}

#[inline]
#[target_feature(enable = "neon")]
fn load_pq2_planes_s8(planes: &[i8; 128]) -> [int8x16_t; 8] {
    let chunks = planes.as_chunks::<16>().0;
    [
        load_s8x16(&chunks[0]),
        load_s8x16(&chunks[1]),
        load_s8x16(&chunks[2]),
        load_s8x16(&chunks[3]),
        load_s8x16(&chunks[4]),
        load_s8x16(&chunks[5]),
        load_s8x16(&chunks[6]),
        load_s8x16(&chunks[7]),
    ]
}

#[target_feature(enable = "neon,i8mm")]
fn dot_pq2_four_rows_four_tokens_precomputed_i8mm(
    packed: [&[u8; 32]; 4],
    planes: [&[i8; 128]; 4],
    activation_sums: [i32; 4],
) -> [[i32; 4]; 4] {
    let weights = [
        unpack_pq2_unsigned_s8(packed[0]),
        unpack_pq2_unsigned_s8(packed[1]),
        unpack_pq2_unsigned_s8(packed[2]),
        unpack_pq2_unsigned_s8(packed[3]),
    ];
    let acts = [
        load_pq2_planes_s8(planes[0]),
        load_pq2_planes_s8(planes[1]),
        load_pq2_planes_s8(planes[2]),
        load_pq2_planes_s8(planes[3]),
    ];
    dot_pq2_4x4_tile_i8mm(&weights, &acts, load_s32x4(&activation_sums))
}

#[target_feature(enable = "neon,dotprod")]
fn dot_pq2_four_rows_four_tokens_precomputed_dotprod(
    packed: [&[u8; 32]; 4],
    planes: [&[i8; 128]; 4],
    activation_sums: [i32; 4],
) -> [[i32; 4]; 4] {
    let weights = [
        unpack_pq2_unsigned_s8(packed[0]),
        unpack_pq2_unsigned_s8(packed[1]),
        unpack_pq2_unsigned_s8(packed[2]),
        unpack_pq2_unsigned_s8(packed[3]),
    ];
    let acts = [
        load_pq2_planes_s8(planes[0]),
        load_pq2_planes_s8(planes[1]),
        load_pq2_planes_s8(planes[2]),
        load_pq2_planes_s8(planes[3]),
    ];
    dot_pq2_4x4_tile_dotprod(&weights, &acts, load_s32x4(&activation_sums))
}

#[inline]
#[target_feature(enable = "neon")]
fn plane_s8x16(planes: &[i8; 128], chunk_idx: usize) -> int8x16_t {
    load_s8x16(&planes.as_chunks::<16>().0[chunk_idx])
}

#[inline]
#[target_feature(enable = "neon")]
fn pq2_half_u8x16(packed: &[u8; 32], half: usize) -> uint8x16_t {
    load_u8x16(&packed.as_chunks::<16>().0[half])
}

#[inline]
#[target_feature(enable = "neon")]
fn unpack_pq2_plane_s8(raw: uint8x16_t, plane: usize, mask: uint8x16_t) -> int8x16_t {
    match plane {
        0 => vreinterpretq_s8_u8(vandq_u8(raw, mask)),
        1 => vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<2>(raw), mask)),
        2 => vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<4>(raw), mask)),
        _ => vreinterpretq_s8_u8(vshrq_n_u8::<6>(raw)),
    }
}

#[inline]
#[target_feature(enable = "neon,i8mm")]
fn finish_i8mm_4x4(
    acc00: int32x4_t,
    acc01: int32x4_t,
    acc10: int32x4_t,
    acc11: int32x4_t,
    sums_vec: int32x4_t,
) -> [[i32; 4]; 4] {
    let mut out = [[0_i32; 4]; 4];
    store_s32x4(
        vsubq_s32(
            vcombine_s32(vget_low_s32(acc00), vget_low_s32(acc01)),
            sums_vec,
        ),
        &mut out[0],
    );
    store_s32x4(
        vsubq_s32(
            vcombine_s32(vget_high_s32(acc00), vget_high_s32(acc01)),
            sums_vec,
        ),
        &mut out[1],
    );
    store_s32x4(
        vsubq_s32(
            vcombine_s32(vget_low_s32(acc10), vget_low_s32(acc11)),
            sums_vec,
        ),
        &mut out[2],
    );
    store_s32x4(
        vsubq_s32(
            vcombine_s32(vget_high_s32(acc10), vget_high_s32(acc11)),
            sums_vec,
        ),
        &mut out[3],
    );
    out
}

#[inline]
#[target_feature(enable = "neon,i8mm")]
fn accumulate_pq2_half_4x4_i8mm(
    raw_rows: [uint8x16_t; 4],
    planes: [&[i8; 128]; 4],
    half: usize,
    mask: uint8x16_t,
    acc: &mut [int32x4_t; 4],
) {
    let plane_base = half * 4;
    for plane in 0..4 {
        let w0 = unpack_pq2_plane_s8(raw_rows[0], plane, mask);
        let w1 = unpack_pq2_plane_s8(raw_rows[1], plane, mask);
        let w2 = unpack_pq2_plane_s8(raw_rows[2], plane, mask);
        let w3 = unpack_pq2_plane_s8(raw_rows[3], plane, mask);
        let (w01_lo, w01_hi) = pair_s8_lo_hi(w0, w1);
        let (w23_lo, w23_hi) = pair_s8_lo_hi(w2, w3);

        let chunk = plane_base + plane;
        let a0 = plane_s8x16(planes[0], chunk);
        let a1 = plane_s8x16(planes[1], chunk);
        let a2 = plane_s8x16(planes[2], chunk);
        let a3 = plane_s8x16(planes[3], chunk);
        let (a01_lo, a01_hi) = pair_s8_lo_hi(a0, a1);
        let (a23_lo, a23_hi) = pair_s8_lo_hi(a2, a3);

        acc[0] = smmla_s32(smmla_s32(acc[0], w01_lo, a01_lo), w01_hi, a01_hi);
        acc[1] = smmla_s32(smmla_s32(acc[1], w01_lo, a23_lo), w01_hi, a23_hi);
        acc[2] = smmla_s32(smmla_s32(acc[2], w23_lo, a01_lo), w23_hi, a01_hi);
        acc[3] = smmla_s32(smmla_s32(acc[3], w23_lo, a23_lo), w23_hi, a23_hi);
    }
}

#[inline]
#[target_feature(enable = "neon,i8mm")]
fn dot_pq2_4x4_stream_i8mm(
    packed: [&[u8; 32]; 4],
    planes: [&[i8; 128]; 4],
    sums_vec: int32x4_t,
) -> [[i32; 4]; 4] {
    let mask = vdupq_n_u8(3);
    let mut acc = [vdupq_n_s32(0); 4];
    for half in 0..2 {
        let raw_rows = [
            pq2_half_u8x16(packed[0], half),
            pq2_half_u8x16(packed[1], half),
            pq2_half_u8x16(packed[2], half),
            pq2_half_u8x16(packed[3], half),
        ];
        accumulate_pq2_half_4x4_i8mm(raw_rows, planes, half, mask, &mut acc);
    }
    finish_i8mm_4x4(acc[0], acc[1], acc[2], acc[3], sums_vec)
}

#[inline]
#[target_feature(enable = "neon")]
fn unpack_pq2_paired_plane(
    raw_rows: [uint8x16_t; 4],
    plane: usize,
    mask: uint8x16_t,
) -> (int8x16_t, int8x16_t, int8x16_t, int8x16_t) {
    let w0 = unpack_pq2_plane_s8(raw_rows[0], plane, mask);
    let w1 = unpack_pq2_plane_s8(raw_rows[1], plane, mask);
    let w2 = unpack_pq2_plane_s8(raw_rows[2], plane, mask);
    let w3 = unpack_pq2_plane_s8(raw_rows[3], plane, mask);
    let (w01_lo, w01_hi) = pair_s8_lo_hi(w0, w1);
    let (w23_lo, w23_hi) = pair_s8_lo_hi(w2, w3);
    (w01_lo, w01_hi, w23_lo, w23_hi)
}

#[inline]
#[target_feature(enable = "neon")]
fn unpack_pq2_paired_half(
    raw_rows: [uint8x16_t; 4],
    mask: uint8x16_t,
) -> [(int8x16_t, int8x16_t, int8x16_t, int8x16_t); 4] {
    [
        unpack_pq2_paired_plane(raw_rows, 0, mask),
        unpack_pq2_paired_plane(raw_rows, 1, mask),
        unpack_pq2_paired_plane(raw_rows, 2, mask),
        unpack_pq2_paired_plane(raw_rows, 3, mask),
    ]
}

#[inline]
#[target_feature(enable = "neon,i8mm")]
fn accumulate_pq2_paired_half_4x8_i8mm(
    w_planes: &[(int8x16_t, int8x16_t, int8x16_t, int8x16_t); 4],
    planes: [&[i8; 128]; 8],
    half: usize,
    acc_lo: &mut [int32x4_t; 4],
    acc_hi: &mut [int32x4_t; 4],
) {
    let plane_base = half * 4;
    for (plane, &(w01_lo, w01_hi, w23_lo, w23_hi)) in w_planes.iter().enumerate() {
        let chunk = plane_base + plane;
        let (a01_lo, a01_hi) =
            pair_s8_lo_hi(plane_s8x16(planes[0], chunk), plane_s8x16(planes[1], chunk));
        let (a23_lo, a23_hi) =
            pair_s8_lo_hi(plane_s8x16(planes[2], chunk), plane_s8x16(planes[3], chunk));
        acc_lo[0] = smmla_s32(smmla_s32(acc_lo[0], w01_lo, a01_lo), w01_hi, a01_hi);
        acc_lo[1] = smmla_s32(smmla_s32(acc_lo[1], w01_lo, a23_lo), w01_hi, a23_hi);
        acc_lo[2] = smmla_s32(smmla_s32(acc_lo[2], w23_lo, a01_lo), w23_hi, a01_hi);
        acc_lo[3] = smmla_s32(smmla_s32(acc_lo[3], w23_lo, a23_lo), w23_hi, a23_hi);

        let (b01_lo, b01_hi) =
            pair_s8_lo_hi(plane_s8x16(planes[4], chunk), plane_s8x16(planes[5], chunk));
        let (b23_lo, b23_hi) =
            pair_s8_lo_hi(plane_s8x16(planes[6], chunk), plane_s8x16(planes[7], chunk));
        acc_hi[0] = smmla_s32(smmla_s32(acc_hi[0], w01_lo, b01_lo), w01_hi, b01_hi);
        acc_hi[1] = smmla_s32(smmla_s32(acc_hi[1], w01_lo, b23_lo), w01_hi, b23_hi);
        acc_hi[2] = smmla_s32(smmla_s32(acc_hi[2], w23_lo, b01_lo), w23_hi, b01_hi);
        acc_hi[3] = smmla_s32(smmla_s32(acc_hi[3], w23_lo, b23_lo), w23_hi, b23_hi);
    }
}

#[inline]
#[target_feature(enable = "neon,i8mm")]
fn accumulate_pq2_half_4x8_i8mm(
    raw_rows: [uint8x16_t; 4],
    planes: [&[i8; 128]; 8],
    half: usize,
    mask: uint8x16_t,
    acc_lo: &mut [int32x4_t; 4],
    acc_hi: &mut [int32x4_t; 4],
) {
    let w_planes = unpack_pq2_paired_half(raw_rows, mask);
    accumulate_pq2_paired_half_4x8_i8mm(&w_planes, planes, half, acc_lo, acc_hi);
}

#[inline]
#[target_feature(enable = "neon,i8mm")]
fn dot_pq2_4x8_stream_i8mm(
    packed: [&[u8; 32]; 4],
    planes: [&[i8; 128]; 8],
    sums_lo: int32x4_t,
    sums_hi: int32x4_t,
) -> ([[i32; 4]; 4], [[i32; 4]; 4]) {
    let mask = vdupq_n_u8(3);
    let mut acc_lo = [vdupq_n_s32(0); 4];
    let mut acc_hi = [vdupq_n_s32(0); 4];
    for half in 0..2 {
        let raw_rows = [
            pq2_half_u8x16(packed[0], half),
            pq2_half_u8x16(packed[1], half),
            pq2_half_u8x16(packed[2], half),
            pq2_half_u8x16(packed[3], half),
        ];
        accumulate_pq2_half_4x8_i8mm(raw_rows, planes, half, mask, &mut acc_lo, &mut acc_hi);
    }
    (
        finish_i8mm_4x4(acc_lo[0], acc_lo[1], acc_lo[2], acc_lo[3], sums_lo),
        finish_i8mm_4x4(acc_hi[0], acc_hi[1], acc_hi[2], acc_hi[3], sums_hi),
    )
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
fn dot_pq2_4x4_stream_dotprod(
    packed: [&[u8; 32]; 4],
    planes: [&[i8; 128]; 4],
    sums_vec: int32x4_t,
) -> [[i32; 4]; 4] {
    let mask = vdupq_n_u8(3);
    let mut acc = [[vdupq_n_s32(0); 4]; 4];
    for half in 0..2 {
        let raw_rows = [
            pq2_half_u8x16(packed[0], half),
            pq2_half_u8x16(packed[1], half),
            pq2_half_u8x16(packed[2], half),
            pq2_half_u8x16(packed[3], half),
        ];
        let plane_base = half * 4;
        for plane in 0..4 {
            let chunk = plane_base + plane;
            let a0 = plane_s8x16(planes[0], chunk);
            let a1 = plane_s8x16(planes[1], chunk);
            let a2 = plane_s8x16(planes[2], chunk);
            let a3 = plane_s8x16(planes[3], chunk);
            for r in 0..4 {
                let w = unpack_pq2_plane_s8(raw_rows[r], plane, mask);
                acc[r][0] = vdotq_s32(acc[r][0], w, a0);
                acc[r][1] = vdotq_s32(acc[r][1], w, a1);
                acc[r][2] = vdotq_s32(acc[r][2], w, a2);
                acc[r][3] = vdotq_s32(acc[r][3], w, a3);
            }
        }
    }
    let mut out = [[0_i32; 4]; 4];
    for r in 0..4 {
        let reduced = vsubq_s32(
            reduce4_s32x4_addp(acc[r][0], acc[r][1], acc[r][2], acc[r][3]),
            sums_vec,
        );
        store_s32x4(reduced, &mut out[r]);
    }
    out
}

#[target_feature(enable = "neon,i8mm")]
fn dot_pq2_eight_rows_four_tokens_precomputed_i8mm(
    packed: [&[u8; 32]; 8],
    planes: [&[i8; 128]; 4],
    activation_sums: [i32; 4],
) -> [[i32; 4]; 8] {
    let sums_vec = load_s32x4(&activation_sums);
    let q0 = dot_pq2_4x4_stream_i8mm(
        [packed[0], packed[1], packed[2], packed[3]],
        planes,
        sums_vec,
    );
    let q1 = dot_pq2_4x4_stream_i8mm(
        [packed[4], packed[5], packed[6], packed[7]],
        planes,
        sums_vec,
    );
    [q0[0], q0[1], q0[2], q0[3], q1[0], q1[1], q1[2], q1[3]]
}

#[target_feature(enable = "neon,dotprod")]
fn dot_pq2_eight_rows_four_tokens_precomputed_dotprod(
    packed: [&[u8; 32]; 8],
    planes: [&[i8; 128]; 4],
    activation_sums: [i32; 4],
) -> [[i32; 4]; 8] {
    let sums_vec = load_s32x4(&activation_sums);
    let q0 = dot_pq2_4x4_stream_dotprod(
        [packed[0], packed[1], packed[2], packed[3]],
        planes,
        sums_vec,
    );
    let q1 = dot_pq2_4x4_stream_dotprod(
        [packed[4], packed[5], packed[6], packed[7]],
        planes,
        sums_vec,
    );
    [q0[0], q0[1], q0[2], q0[3], q1[0], q1[1], q1[2], q1[3]]
}

#[target_feature(enable = "neon,i8mm")]
fn dot_pq2_eight_rows_eight_tokens_precomputed_i8mm(
    packed: [&[u8; 32]; 8],
    planes: [&[i8; 128]; 8],
    activation_sums: [i32; 8],
) -> [[i32; 8]; 8] {
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
    let (q00, q01) = dot_pq2_4x8_stream_i8mm(
        [packed[0], packed[1], packed[2], packed[3]],
        planes,
        sums_lo,
        sums_hi,
    );
    let (q10, q11) = dot_pq2_4x8_stream_i8mm(
        [packed[4], packed[5], packed[6], packed[7]],
        planes,
        sums_lo,
        sums_hi,
    );
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
fn dot_pq2_eight_rows_eight_tokens_precomputed_dotprod(
    packed: [&[u8; 32]; 8],
    planes: [&[i8; 128]; 8],
    activation_sums: [i32; 8],
) -> [[i32; 8]; 8] {
    let acts_lo = [planes[0], planes[1], planes[2], planes[3]];
    let acts_hi = [planes[4], planes[5], planes[6], planes[7]];
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
    let w_lo = [packed[0], packed[1], packed[2], packed[3]];
    let w_hi = [packed[4], packed[5], packed[6], packed[7]];
    let q00 = dot_pq2_4x4_stream_dotprod(w_lo, acts_lo, sums_lo);
    let q01 = dot_pq2_4x4_stream_dotprod(w_lo, acts_hi, sums_hi);
    let q10 = dot_pq2_4x4_stream_dotprod(w_hi, acts_lo, sums_lo);
    let q11 = dot_pq2_4x4_stream_dotprod(w_hi, acts_hi, sums_hi);
    let mut out = [[0_i32; 8]; 8];
    for r in 0..4 {
        out[r][..4].copy_from_slice(&q00[r]);
        out[r][4..].copy_from_slice(&q01[r]);
        out[r + 4][..4].copy_from_slice(&q10[r]);
        out[r + 4][4..].copy_from_slice(&q11[r]);
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
pub(super) fn pq2_detected_dot_four_tokens(
    detected: super::Pq2Dotprod,
    packed: [&[u8; 32]; 4],
    planes: [&[i8; 128]; 4],
    activation_sums: [i32; 4],
) -> [[i32; 4]; 4] {
    if detected.i8mm {
        // SAFETY: This private capability flag was set only after CPU/OS I8MM
        // detection. Fixed array references provide every required input byte.
        unsafe { dot_pq2_four_rows_four_tokens_precomputed_i8mm(packed, planes, activation_sums) }
    } else {
        // SAFETY: Constructing the unforgeable capability verified CPU/OS
        // dotprod eligibility. All inputs have the kernel's fixed array sizes.
        unsafe {
            dot_pq2_four_rows_four_tokens_precomputed_dotprod(packed, planes, activation_sums)
        }
    }
}

#[inline]
pub(super) fn pq2_detected_dot_eight_rows_four_tokens(
    detected: super::Pq2Dotprod,
    packed: [&[u8; 32]; 8],
    planes: [&[i8; 128]; 4],
    activation_sums: [i32; 4],
) -> [[i32; 4]; 8] {
    if detected.i8mm {
        // SAFETY: This private capability flag was set only after CPU/OS I8MM
        // detection. Fixed array references provide every required input byte.
        unsafe { dot_pq2_eight_rows_four_tokens_precomputed_i8mm(packed, planes, activation_sums) }
    } else {
        // SAFETY: Constructing the unforgeable capability verified CPU/OS
        // dotprod eligibility. All inputs have the kernel's fixed array sizes.
        unsafe {
            dot_pq2_eight_rows_four_tokens_precomputed_dotprod(packed, planes, activation_sums)
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn update_scaled_row4<const FUSED: bool>(
    dots: int32x4_t,
    row_scale: f32,
    act_scales: core::arch::aarch64::float32x4_t,
    dst: &mut [f32; 4],
) {
    use core::arch::aarch64::{
        vaddq_f32, vcvtq_f32_s32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vmulq_f32, vst1q_f32,
    };
    let floats = vcvtq_f32_s32(dots);
    let d = vmulq_f32(vdupq_n_f32(row_scale), act_scales);
    // SAFETY: `dst` references an initialized four-element `f32` array.
    let cur = unsafe { vld1q_f32(dst.as_ptr()) };
    let next = if FUSED {
        vfmaq_f32(cur, d, floats)
    } else {
        vaddq_f32(cur, vmulq_f32(d, floats))
    };
    // SAFETY: `dst` references a unique writable four-element `f32` array.
    unsafe { vst1q_f32(dst.as_mut_ptr(), next) };
}

#[inline]
#[target_feature(enable = "neon")]
fn load_pq2_block_halves(block: &[u8; 34]) -> (uint8x16_t, uint8x16_t) {
    let [_, _, ref payload @ ..] = *block;
    let halves = payload.as_chunks::<16>().0;
    (load_u8x16(&halves[0]), load_u8x16(&halves[1]))
}

#[inline]
#[target_feature(enable = "neon")]
fn widen_row_scales_f16(blocks: &[&[u8; 34]; 8]) -> Option<[[f32; 4]; 2]> {
    use core::arch::aarch64::{
        vandq_u16, vcvt_f32_f16, vcvt_high_f32_f16, vdupq_n_u16, vget_low_f16, vld1q_u16,
        vmaxvq_u16, vreinterpretq_f16_u16, vst1q_f32,
    };
    let scale_bits: [u16; 8] =
        std::array::from_fn(|r| u16::from_le_bytes([blocks[r][0], blocks[r][1]]));
    // SAFETY: `scale_bits` is an initialized eight-element `u16` array.
    let scale_vec = unsafe { vld1q_u16(scale_bits.as_ptr()) };
    if vmaxvq_u16(vandq_u16(scale_vec, vdupq_n_u16(0x7fff))) >= 0x7c00 {
        return None;
    }
    let half_scales = vreinterpretq_f16_u16(scale_vec);
    let lo = vcvt_f32_f16(vget_low_f16(half_scales));
    let hi = vcvt_high_f32_f16(half_scales);
    let mut row_scales = [[0.0_f32; 4]; 2];
    // SAFETY: `row_scales[0]` is a valid 4-element `f32` array.
    unsafe { vst1q_f32(row_scales[0].as_mut_ptr(), lo) };
    // SAFETY: `row_scales[1]` is a valid 4-element `f32` array.
    unsafe { vst1q_f32(row_scales[1].as_mut_ptr(), hi) };
    Some(row_scales)
}

#[inline]
#[target_feature(enable = "neon")]
fn load_f16_scales_x4_pq2(blocks: [&[u8; 34]; 4]) -> core::arch::aarch64::uint16x4_t {
    use core::arch::aarch64::{vdup_n_u16, vset_lane_u16};
    let v = vdup_n_u16(u16::from_le_bytes([blocks[0][0], blocks[0][1]]));
    let v = vset_lane_u16::<1>(u16::from_le_bytes([blocks[1][0], blocks[1][1]]), v);
    let v = vset_lane_u16::<2>(u16::from_le_bytes([blocks[2][0], blocks[2][1]]), v);
    vset_lane_u16::<3>(u16::from_le_bytes([blocks[3][0], blocks[3][1]]), v)
}

#[inline]
#[target_feature(enable = "neon,i8mm")]
fn scaled_4x4_dots_and_scales_i8mm(
    acc: [int32x4_t; 4],
    sums_vec: int32x4_t,
    row_scales: core::arch::aarch64::float32x4_t,
    act_scales: core::arch::aarch64::float32x4_t,
) -> (
    [core::arch::aarch64::float32x4_t; 4],
    [core::arch::aarch64::float32x4_t; 4],
) {
    use core::arch::aarch64::{vcvtq_f32_s32, vmulq_laneq_f32};
    let dots = [
        vcvtq_f32_s32(vsubq_s32(
            vcombine_s32(vget_low_s32(acc[0]), vget_low_s32(acc[1])),
            sums_vec,
        )),
        vcvtq_f32_s32(vsubq_s32(
            vcombine_s32(vget_high_s32(acc[0]), vget_high_s32(acc[1])),
            sums_vec,
        )),
        vcvtq_f32_s32(vsubq_s32(
            vcombine_s32(vget_low_s32(acc[2]), vget_low_s32(acc[3])),
            sums_vec,
        )),
        vcvtq_f32_s32(vsubq_s32(
            vcombine_s32(vget_high_s32(acc[2]), vget_high_s32(acc[3])),
            sums_vec,
        )),
    ];
    let scales = [
        vmulq_laneq_f32::<0>(act_scales, row_scales),
        vmulq_laneq_f32::<1>(act_scales, row_scales),
        vmulq_laneq_f32::<2>(act_scales, row_scales),
        vmulq_laneq_f32::<3>(act_scales, row_scales),
    ];
    (dots, scales)
}

#[inline]
#[target_feature(enable = "neon,i8mm")]
fn finish_reg_scaled_4x4_i8mm<const FUSED: bool>(
    acc: [int32x4_t; 4],
    sums_vec: int32x4_t,
    row_scales: core::arch::aarch64::float32x4_t,
    act_scales: core::arch::aarch64::float32x4_t,
    f_acc: &mut [[core::arch::aarch64::float32x4_t; 2]; 4],
    hi_quad: usize,
) {
    let (dots, d) = scaled_4x4_dots_and_scales_i8mm(acc, sums_vec, row_scales, act_scales);
    for r in 0..4 {
        f_acc[r][hi_quad] = f32x4_scale_step::<FUSED>(f_acc[r][hi_quad], d[r], dots[r]);
    }
}

#[inline]
#[target_feature(enable = "neon,i8mm")]
fn finish_reg_scaled_4x4_out_i8mm<const FUSED: bool>(
    acc: [int32x4_t; 4],
    sums_vec: int32x4_t,
    row_scales: core::arch::aarch64::float32x4_t,
    act_scales: core::arch::aarch64::float32x4_t,
    out_rows: &mut [[f32; 8]; 4],
    hi_quad: usize,
) {
    let (dots, d) = scaled_4x4_dots_and_scales_i8mm(acc, sums_vec, row_scales, act_scales);
    for (r, out_row) in out_rows.iter_mut().enumerate() {
        let quads = out_row.as_chunks_mut::<4>().0;
        let cur = load_f32x4_neon(&quads[hi_quad]);
        let next = f32x4_scale_step::<FUSED>(cur, d[r], dots[r]);
        store_f32x4_neon(next, &mut quads[hi_quad]);
    }
}

#[inline]
#[target_feature(enable = "neon,i8mm")]
fn accumulate_pq2_half_4x8_streaming_i8mm(
    raw_rows: [uint8x16_t; 4],
    planes: [&[i8; 128]; 8],
    half: usize,
    mask: uint8x16_t,
    acc_lo: &mut [int32x4_t; 4],
    acc_hi: &mut [int32x4_t; 4],
) {
    let plane_base = half * 4;
    for plane in 0..4 {
        let (w01_lo, w01_hi, w23_lo, w23_hi) = unpack_pq2_paired_plane(raw_rows, plane, mask);
        let chunk = plane_base + plane;
        let (a01_lo, a01_hi) =
            pair_s8_lo_hi(plane_s8x16(planes[0], chunk), plane_s8x16(planes[1], chunk));
        acc_lo[0] = smmla_s32(smmla_s32(acc_lo[0], w01_lo, a01_lo), w01_hi, a01_hi);
        acc_lo[2] = smmla_s32(smmla_s32(acc_lo[2], w23_lo, a01_lo), w23_hi, a01_hi);

        let (a23_lo, a23_hi) =
            pair_s8_lo_hi(plane_s8x16(planes[2], chunk), plane_s8x16(planes[3], chunk));
        acc_lo[1] = smmla_s32(smmla_s32(acc_lo[1], w01_lo, a23_lo), w01_hi, a23_hi);
        acc_lo[3] = smmla_s32(smmla_s32(acc_lo[3], w23_lo, a23_lo), w23_hi, a23_hi);

        let (b01_lo, b01_hi) =
            pair_s8_lo_hi(plane_s8x16(planes[4], chunk), plane_s8x16(planes[5], chunk));
        acc_hi[0] = smmla_s32(smmla_s32(acc_hi[0], w01_lo, b01_lo), w01_hi, b01_hi);
        acc_hi[2] = smmla_s32(smmla_s32(acc_hi[2], w23_lo, b01_lo), w23_hi, b01_hi);

        let (b23_lo, b23_hi) =
            pair_s8_lo_hi(plane_s8x16(planes[6], chunk), plane_s8x16(planes[7], chunk));
        acc_hi[1] = smmla_s32(smmla_s32(acc_hi[1], w01_lo, b23_lo), w01_hi, b23_hi);
        acc_hi[3] = smmla_s32(smmla_s32(acc_hi[3], w23_lo, b23_lo), w23_hi, b23_hi);
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn pack_pq2_i8mm_pair_block(
    b0: &super::Pq2Activation,
    b1: &super::Pq2Activation,
) -> super::Pq2I8mmPairBlock {
    let mut planes = [([0_i8; 16], [0_i8; 16]); 8];
    for (plane_idx, dst) in planes.iter_mut().enumerate() {
        let (lo, hi) = pair_s8_lo_hi(
            plane_s8x16(&b0.values, plane_idx),
            plane_s8x16(&b1.values, plane_idx),
        );
        store_s8x16(lo, &mut dst.0);
        store_s8x16(hi, &mut dst.1);
    }
    super::Pq2I8mmPairBlock {
        planes,
        sums: [b0.sum, b1.sum],
        scales: [b0.scale, b1.scale],
    }
}

#[inline]
pub(super) fn pq2_append_i8mm_pair_blocks(
    t0: &[super::Pq2Activation],
    t1: &[super::Pq2Activation],
    out: &mut Vec<super::Pq2I8mmPairBlock>,
) {
    // SAFETY: Baseline NEON is mandatory on `aarch64` targets and caller verified matching lengths.
    unsafe { append_pq2_i8mm_pair_blocks(t0, t1, out) };
}

#[inline]
pub(super) fn pq2_pack_i8mm_pair_blocks_into(
    t0: &[super::Pq2Activation],
    t1: &[super::Pq2Activation],
    out: &mut [super::Pq2I8mmPairBlock],
) {
    // SAFETY: Baseline NEON is mandatory on `aarch64` targets and caller verified matching lengths.
    unsafe { pack_pq2_i8mm_pair_blocks_into_neon(t0, t1, out) };
}

#[target_feature(enable = "neon")]
pub(super) fn append_pq2_i8mm_pair_blocks(
    t0: &[super::Pq2Activation],
    t1: &[super::Pq2Activation],
    out: &mut Vec<super::Pq2I8mmPairBlock>,
) {
    out.reserve(t0.len());
    for (b0, b1) in t0.iter().zip(t1.iter()) {
        out.push(pack_pq2_i8mm_pair_block(b0, b1));
    }
}

#[target_feature(enable = "neon")]
fn pack_pq2_i8mm_pair_blocks_into_neon(
    t0: &[super::Pq2Activation],
    t1: &[super::Pq2Activation],
    out: &mut [super::Pq2I8mmPairBlock],
) {
    for (dst, (b0, b1)) in out.iter_mut().zip(t0.iter().zip(t1.iter())) {
        *dst = pack_pq2_i8mm_pair_block(b0, b1);
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "K-tiled register-resident 4x8 I8MM kernel preserves exact block_idx order without 32-register spills."
)]
#[target_feature(enable = "neon,i8mm")]
fn dot_scaled_8x8_chunks_i8mm<const FUSED: bool>(
    rows: [&[u8]; 8],
    chunks: &[[&[super::Pq2Activation]; 8]],
    outputs: &mut [[[f32; 8]; 8]],
) -> Option<()> {
    use core::arch::aarch64::{
        vand_u16, vcvt_f32_f16, vdup_n_u16, vdupq_n_f32, vld1q_f32, vmax_u16, vmaxv_u16,
        vreinterpret_f16_u16,
    };
    const K_TILE: usize = 16;
    outputs.fill([[0.0_f32; 8]; 8]);
    let block_count = chunks[0][0].len();
    let mask = vdupq_n_u8(3);
    let mask_7fff = vdup_n_u16(0x7fff);
    let mut umax_scales = vdup_n_u16(0);
    let row_chunks = rows.map(|row| row.as_chunks::<34>().0);

    for k_start in (0..block_count).step_by(K_TILE) {
        let k_end = (k_start + K_TILE).min(block_count);
        for row_half in 0..2 {
            let rbase = row_half * 4;
            for (chunk, out_chunk) in chunks.iter().zip(outputs.iter_mut()) {
                let out_halves = out_chunk.as_chunks_mut::<4>().0;
                let mut f_acc: [[core::arch::aarch64::float32x4_t; 2]; 4] = if k_start == 0 {
                    [[vdupq_n_f32(0.0); 2]; 4]
                } else {
                    std::array::from_fn(|r| {
                        let quads = out_halves[row_half][r].as_chunks::<4>().0;
                        [load_f32x4_neon(&quads[0]), load_f32x4_neon(&quads[1])]
                    })
                };
                for block_idx in k_start..k_end {
                    let blocks: [&[u8; 34]; 4] = [
                        &row_chunks[rbase][block_idx],
                        &row_chunks[rbase + 1][block_idx],
                        &row_chunks[rbase + 2][block_idx],
                        &row_chunks[rbase + 3][block_idx],
                    ];
                    if block_idx + 2 < k_end {
                        for block in &row_chunks[rbase..rbase + 4] {
                            prefetch_l1(block[block_idx + 2].as_ptr());
                        }
                        prefetch_pq2_tokens_block(chunk, block_idx + 1, k_end);
                    }
                    let scale_u16 = load_f16_scales_x4_pq2(blocks);
                    umax_scales = vmax_u16(umax_scales, vand_u16(scale_u16, mask_7fff));
                    let row_scales = vcvt_f32_f16(vreinterpret_f16_u16(scale_u16));

                    let a = [
                        &chunk[0][block_idx],
                        &chunk[1][block_idx],
                        &chunk[2][block_idx],
                        &chunk[3][block_idx],
                        &chunk[4][block_idx],
                        &chunk[5][block_idx],
                        &chunk[6][block_idx],
                        &chunk[7][block_idx],
                    ];
                    let planes: [&[i8; 128]; 8] = [
                        &a[0].values,
                        &a[1].values,
                        &a[2].values,
                        &a[3].values,
                        &a[4].values,
                        &a[5].values,
                        &a[6].values,
                        &a[7].values,
                    ];
                    let mut acc_lo = [vdupq_n_s32(0); 4];
                    let mut acc_hi = [vdupq_n_s32(0); 4];
                    for half in 0..2 {
                        let raw_rows = [
                            load_u8x16(&blocks[0][2..].as_chunks::<16>().0[half]),
                            load_u8x16(&blocks[1][2..].as_chunks::<16>().0[half]),
                            load_u8x16(&blocks[2][2..].as_chunks::<16>().0[half]),
                            load_u8x16(&blocks[3][2..].as_chunks::<16>().0[half]),
                        ];
                        accumulate_pq2_half_4x8_streaming_i8mm(
                            raw_rows,
                            planes,
                            half,
                            mask,
                            &mut acc_lo,
                            &mut acc_hi,
                        );
                    }

                    let sums_lo = load_s32x4(&[a[0].sum, a[1].sum, a[2].sum, a[3].sum]);
                    let sums_hi = load_s32x4(&[a[4].sum, a[5].sum, a[6].sum, a[7].sum]);
                    let scales_lo_arr = [a[0].scale, a[1].scale, a[2].scale, a[3].scale];
                    let scales_hi_arr = [a[4].scale, a[5].scale, a[6].scale, a[7].scale];
                    // SAFETY: fixed 4-element initialized arrays.
                    let act_scales_lo = unsafe { vld1q_f32(scales_lo_arr.as_ptr()) };
                    // SAFETY: fixed 4-element initialized array.
                    let act_scales_hi = unsafe { vld1q_f32(scales_hi_arr.as_ptr()) };

                    finish_reg_scaled_4x4_i8mm::<FUSED>(
                        acc_lo,
                        sums_lo,
                        row_scales,
                        act_scales_lo,
                        &mut f_acc,
                        0,
                    );
                    finish_reg_scaled_4x4_i8mm::<FUSED>(
                        acc_hi,
                        sums_hi,
                        row_scales,
                        act_scales_hi,
                        &mut f_acc,
                        1,
                    );
                }
                for (r, acc_pair) in f_acc.iter().enumerate() {
                    let quads = out_halves[row_half][r].as_chunks_mut::<4>().0;
                    store_f32x4_neon(acc_pair[0], &mut quads[0]);
                    store_f32x4_neon(acc_pair[1], &mut quads[1]);
                }
            }
        }
    }
    if vmaxv_u16(umax_scales) >= 0x7c00 {
        return None;
    }
    outputs
        .iter()
        .flatten()
        .flatten()
        .all(|v| v.is_finite())
        .then_some(())
}

#[expect(
    clippy::too_many_lines,
    reason = "K-tiled 8x8 I8MM kernel unpacks all 8 rows once per K_TILE and streams activation panels once per chunk."
)]
#[target_feature(enable = "neon,i8mm")]
fn dot_scaled_8x8_chunks_panels_i8mm<const FUSED: bool>(
    rows: [&[u8]; 8],
    panels: &[[&[super::Pq2I8mmPairBlock]; 4]],
    outputs: &mut [[[f32; 8]; 8]],
) -> Option<()> {
    use core::arch::aarch64::{
        vand_u16, vcvt_f32_f16, vdup_n_u16, vdupq_n_f32, vdupq_n_s8, vmax_u16, vmaxv_u16,
        vreinterpret_f16_u16,
    };
    const K_TILE: usize = 4;
    outputs.fill([[0.0_f32; 8]; 8]);
    let block_count = panels[0][0].len();
    let mask = vdupq_n_u8(3);
    let mask_7fff = vdup_n_u16(0x7fff);
    let mut umax_scales = vdup_n_u16(0);
    let row_chunks = rows.map(|row| row.as_chunks::<34>().0);

    for k_start in (0..block_count).step_by(K_TILE) {
        let k_end = (k_start + K_TILE).min(block_count);
        let tile_len = k_end - k_start;
        let mut tile_scales = [[vdupq_n_f32(0.0); 2]; K_TILE];
        let mut tile_weights = [[([vdupq_n_s8(0); 4], [vdupq_n_s8(0); 4]); 8]; K_TILE];

        for k_off in 0..tile_len {
            let block_idx = k_start + k_off;
            if block_idx + 2 < block_count {
                for row in &row_chunks {
                    prefetch_l1(row[block_idx + 2].as_ptr());
                }
            }
            for row_half in 0..2 {
                let rbase = row_half * 4;
                let blocks: [&[u8; 34]; 4] = [
                    &row_chunks[rbase][block_idx],
                    &row_chunks[rbase + 1][block_idx],
                    &row_chunks[rbase + 2][block_idx],
                    &row_chunks[rbase + 3][block_idx],
                ];
                let scale_u16 = load_f16_scales_x4_pq2(blocks);
                umax_scales = vmax_u16(umax_scales, vand_u16(scale_u16, mask_7fff));
                tile_scales[k_off][row_half] = vcvt_f32_f16(vreinterpret_f16_u16(scale_u16));
                for half in 0..2 {
                    let raw_rows = [
                        load_u8x16(&blocks[0][2..].as_chunks::<16>().0[half]),
                        load_u8x16(&blocks[1][2..].as_chunks::<16>().0[half]),
                        load_u8x16(&blocks[2][2..].as_chunks::<16>().0[half]),
                        load_u8x16(&blocks[3][2..].as_chunks::<16>().0[half]),
                    ];
                    for plane in 0..4 {
                        let (w_first_lo, w_first_hi, w_second_lo, w_second_hi) =
                            unpack_pq2_paired_plane(raw_rows, plane, mask);
                        let slot = &mut tile_weights[k_off][half * 4 + plane];
                        slot.0[row_half * 2] = w_first_lo;
                        slot.0[row_half * 2 + 1] = w_second_lo;
                        slot.1[row_half * 2] = w_first_hi;
                        slot.1[row_half * 2 + 1] = w_second_hi;
                    }
                }
            }
        }

        for (chunk_panels, out_chunk) in panels.iter().zip(outputs.iter_mut()) {
            let out_halves = out_chunk.as_chunks_mut::<4>().0;
            for k_off in 0..tile_len {
                let block_idx = k_start + k_off;
                if block_idx + 2 < block_count {
                    for pair_slice in chunk_panels {
                        prefetch_l1(pair_slice[block_idx + 2].planes.as_ptr().cast::<u8>());
                    }
                }
                let pairs = [
                    &chunk_panels[0][block_idx],
                    &chunk_panels[1][block_idx],
                    &chunk_panels[2][block_idx],
                    &chunk_panels[3][block_idx],
                ];
                let mut acc = [[vdupq_n_s32(0); 4]; 4];
                for (chunk, (w_lo, w_hi)) in tile_weights[k_off].iter().enumerate() {
                    let [w01_lo, w23_lo, w45_lo, w67_lo] = *w_lo;
                    for (tp, pair) in pairs.iter().enumerate() {
                        let a_lo = load_s8x16(&pair.planes[chunk].0);
                        acc[0][tp] = smmla_s32(acc[0][tp], w01_lo, a_lo);
                        acc[1][tp] = smmla_s32(acc[1][tp], w23_lo, a_lo);
                        acc[2][tp] = smmla_s32(acc[2][tp], w45_lo, a_lo);
                        acc[3][tp] = smmla_s32(acc[3][tp], w67_lo, a_lo);
                    }
                    let [w01_hi, w23_hi, w45_hi, w67_hi] = *w_hi;
                    for (tp, pair) in pairs.iter().enumerate() {
                        let a_hi = load_s8x16(&pair.planes[chunk].1);
                        acc[0][tp] = smmla_s32(acc[0][tp], w01_hi, a_hi);
                        acc[1][tp] = smmla_s32(acc[1][tp], w23_hi, a_hi);
                        acc[2][tp] = smmla_s32(acc[2][tp], w45_hi, a_hi);
                        acc[3][tp] = smmla_s32(acc[3][tp], w67_hi, a_hi);
                    }
                }

                let sums_lo = load_s32x4(&[
                    pairs[0].sums[0],
                    pairs[0].sums[1],
                    pairs[1].sums[0],
                    pairs[1].sums[1],
                ]);
                let sums_hi = load_s32x4(&[
                    pairs[2].sums[0],
                    pairs[2].sums[1],
                    pairs[3].sums[0],
                    pairs[3].sums[1],
                ]);
                let act_scales_lo = load_f32x4_neon(&[
                    pairs[0].scales[0],
                    pairs[0].scales[1],
                    pairs[1].scales[0],
                    pairs[1].scales[1],
                ]);
                let act_scales_hi = load_f32x4_neon(&[
                    pairs[2].scales[0],
                    pairs[2].scales[1],
                    pairs[3].scales[0],
                    pairs[3].scales[1],
                ]);

                for (row_half, out_half) in out_halves.iter_mut().enumerate() {
                    let row_scales = tile_scales[k_off][row_half];
                    let rp0 = row_half * 2;
                    let rp1 = rp0 + 1;
                    finish_reg_scaled_4x4_out_i8mm::<FUSED>(
                        [acc[rp0][0], acc[rp0][1], acc[rp1][0], acc[rp1][1]],
                        sums_lo,
                        row_scales,
                        act_scales_lo,
                        out_half,
                        0,
                    );
                    finish_reg_scaled_4x4_out_i8mm::<FUSED>(
                        [acc[rp0][2], acc[rp0][3], acc[rp1][2], acc[rp1][3]],
                        sums_hi,
                        row_scales,
                        act_scales_hi,
                        out_half,
                        1,
                    );
                }
            }
        }
    }
    if vmaxv_u16(umax_scales) >= 0x7c00 {
        return None;
    }
    outputs
        .iter()
        .flatten()
        .flatten()
        .all(|v| v.is_finite())
        .then_some(())
}

#[inline]
pub(super) fn pq2_detected_dot_scaled_8x8_chunks_with_panels<const FUSED: bool>(
    detected: super::Pq2Dotprod,
    rows: [&[u8]; 8],
    chunks: &[[&[super::Pq2Activation]; 8]],
    panels: &[[&[super::Pq2I8mmPairBlock]; 4]],
    outputs: &mut [[[f32; 8]; 8]],
) -> Option<()> {
    if detected.i8mm {
        // SAFETY: `detected.i8mm` was enabled after CPU/OS I8MM detection, and
        // the safe parent checked chunk/panel extents and finite activation scales.
        unsafe { dot_scaled_8x8_chunks_panels_i8mm::<FUSED>(rows, panels, outputs) }
    } else {
        // SAFETY: Constructing the unforgeable `Pq2Dotprod` verified CPU/OS
        // `dotprod` support, and the safe parent checked all slice invariants.
        unsafe { dot_scaled_8x8_chunks_dotprod::<FUSED>(rows, chunks, outputs) }
    }
}

#[target_feature(enable = "neon,dotprod")]
fn dot_scaled_8x8_chunks_dotprod<const FUSED: bool>(
    rows: [&[u8]; 8],
    chunks: &[[&[super::Pq2Activation]; 8]],
    outputs: &mut [[[f32; 8]; 8]],
) -> Option<()> {
    use core::arch::aarch64::vld1q_f32;
    outputs.fill([[0.0_f32; 8]; 8]);
    let block_count = chunks[0][0].len();
    let mask = vdupq_n_u8(3);
    let row_chunks = rows.map(|row| row.as_chunks::<34>().0);

    for block_idx in 0..block_count {
        prefetch_row_blocks(&row_chunks, block_idx, block_count);
        let blocks: [&[u8; 34]; 8] = std::array::from_fn(|r| &row_chunks[r][block_idx]);
        let row_scales = widen_row_scales_f16(&blocks)?;

        for (row_half, half_scales_arr) in row_scales.iter().enumerate() {
            let rbase = row_half * 4;
            let (r0_h0, r0_h1) = load_pq2_block_halves(blocks[rbase]);
            let (r1_h0, r1_h1) = load_pq2_block_halves(blocks[rbase + 1]);
            let (r2_h0, r2_h1) = load_pq2_block_halves(blocks[rbase + 2]);
            let (r3_h0, r3_h1) = load_pq2_block_halves(blocks[rbase + 3]);
            let raw_halves = [[r0_h0, r1_h0, r2_h0, r3_h0], [r0_h1, r1_h1, r2_h1, r3_h1]];
            for (chunk, out_chunk) in chunks.iter().zip(outputs.iter_mut()) {
                if row_half == 0 {
                    prefetch_pq2_tokens_block(chunk, block_idx, block_count);
                }
                let a = [
                    &chunk[0][block_idx],
                    &chunk[1][block_idx],
                    &chunk[2][block_idx],
                    &chunk[3][block_idx],
                    &chunk[4][block_idx],
                    &chunk[5][block_idx],
                    &chunk[6][block_idx],
                    &chunk[7][block_idx],
                ];
                let quads = [
                    (
                        [&a[0].values, &a[1].values, &a[2].values, &a[3].values],
                        load_s32x4(&[a[0].sum, a[1].sum, a[2].sum, a[3].sum]),
                        [a[0].scale, a[1].scale, a[2].scale, a[3].scale],
                    ),
                    (
                        [&a[4].values, &a[5].values, &a[6].values, &a[7].values],
                        load_s32x4(&[a[4].sum, a[5].sum, a[6].sum, a[7].sum]),
                        [a[4].scale, a[5].scale, a[6].scale, a[7].scale],
                    ),
                ];
                let out_halves = out_chunk.as_chunks_mut::<4>().0;
                for (tok_quad, (planes, sums_vec, scales_arr)) in quads.into_iter().enumerate() {
                    let mut acc = [[vdupq_n_s32(0); 4]; 4];
                    for (half, &raw_rows) in raw_halves.iter().enumerate() {
                        let plane_base = half * 4;
                        for plane in 0..4 {
                            let chunk_idx = plane_base + plane;
                            let a0 = plane_s8x16(planes[0], chunk_idx);
                            let a1 = plane_s8x16(planes[1], chunk_idx);
                            let a2 = plane_s8x16(planes[2], chunk_idx);
                            let a3 = plane_s8x16(planes[3], chunk_idx);
                            for r in 0..4 {
                                let w = unpack_pq2_plane_s8(raw_rows[r], plane, mask);
                                acc[r][0] = vdotq_s32(acc[r][0], w, a0);
                                acc[r][1] = vdotq_s32(acc[r][1], w, a1);
                                acc[r][2] = vdotq_s32(acc[r][2], w, a2);
                                acc[r][3] = vdotq_s32(acc[r][3], w, a3);
                            }
                        }
                    }
                    // SAFETY: `scales_arr` is an initialized 4-element `f32` array.
                    let act_scales = unsafe { vld1q_f32(scales_arr.as_ptr()) };
                    for r in 0..4 {
                        let reduced = vsubq_s32(
                            reduce4_s32x4_addp(acc[r][0], acc[r][1], acc[r][2], acc[r][3]),
                            sums_vec,
                        );
                        let row_quads = out_halves[row_half][r].as_chunks_mut::<4>().0;
                        update_scaled_row4::<FUSED>(
                            reduced,
                            half_scales_arr[r],
                            act_scales,
                            &mut row_quads[tok_quad],
                        );
                    }
                }
            }
        }
    }
    outputs
        .iter()
        .flatten()
        .flatten()
        .all(|v| v.is_finite())
        .then_some(())
}

#[inline]
pub(super) fn pq2_detected_dot_scaled_8x8_chunks<const FUSED: bool>(
    detected: super::Pq2Dotprod,
    rows: [&[u8]; 8],
    chunks: &[[&[super::Pq2Activation]; 8]],
    outputs: &mut [[[f32; 8]; 8]],
) -> Option<()> {
    if detected.i8mm {
        // SAFETY: `detected.i8mm` was enabled after CPU/OS I8MM detection, and
        // the safe parent checked chunk extents and finite activation scales.
        unsafe { dot_scaled_8x8_chunks_i8mm::<FUSED>(rows, chunks, outputs) }
    } else {
        // SAFETY: Constructing the unforgeable `Pq2Dotprod` verified CPU/OS
        // `dotprod` support, and the safe parent checked all slice invariants.
        unsafe { dot_scaled_8x8_chunks_dotprod::<FUSED>(rows, chunks, outputs) }
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn f32x4_scale_step<const FUSED: bool>(
    cur: core::arch::aarch64::float32x4_t,
    d: core::arch::aarch64::float32x4_t,
    dots: core::arch::aarch64::float32x4_t,
) -> core::arch::aarch64::float32x4_t {
    use core::arch::aarch64::{vaddq_f32, vfmaq_f32, vmulq_f32};
    if FUSED {
        vfmaq_f32(cur, d, dots)
    } else {
        vaddq_f32(cur, vmulq_f32(d, dots))
    }
}

#[inline]
#[target_feature(enable = "neon,i8mm,dotprod")]
fn pq2_4x_pairs_block_i8mm<const PAIRS: usize, const HAS_ODD: bool>(
    blocks: [&[u8; 34]; 4],
    tokens: &[&[super::Pq2Activation]],
    block_idx: usize,
    mask: uint8x16_t,
) -> ([[int32x4_t; 2]; PAIRS], [int32x4_t; 4]) {
    let mut acc_pair = [[vdupq_n_s32(0); 2]; PAIRS];
    let mut acc_odd = [vdupq_n_s32(0); 4];
    for half in 0..2 {
        let raw_rows = [
            load_u8x16(&blocks[0][2..].as_chunks::<16>().0[half]),
            load_u8x16(&blocks[1][2..].as_chunks::<16>().0[half]),
            load_u8x16(&blocks[2][2..].as_chunks::<16>().0[half]),
            load_u8x16(&blocks[3][2..].as_chunks::<16>().0[half]),
        ];
        let plane_base = half * 4;
        for plane in 0..4 {
            let chunk = plane_base + plane;
            let w0 = unpack_pq2_plane_s8(raw_rows[0], plane, mask);
            let w1 = unpack_pq2_plane_s8(raw_rows[1], plane, mask);
            let w2 = unpack_pq2_plane_s8(raw_rows[2], plane, mask);
            let w3 = unpack_pq2_plane_s8(raw_rows[3], plane, mask);
            if HAS_ODD {
                let a_odd = plane_s8x16(&tokens[PAIRS * 2][block_idx].values, chunk);
                acc_odd[0] = vdotq_s32(acc_odd[0], w0, a_odd);
                acc_odd[1] = vdotq_s32(acc_odd[1], w1, a_odd);
                acc_odd[2] = vdotq_s32(acc_odd[2], w2, a_odd);
                acc_odd[3] = vdotq_s32(acc_odd[3], w3, a_odd);
            }
            if PAIRS > 0 {
                let (w01_lo, w01_hi) = pair_s8_lo_hi(w0, w1);
                let (w23_lo, w23_hi) = pair_s8_lo_hi(w2, w3);
                for (p, pair_acc) in acc_pair.iter_mut().enumerate() {
                    let a0 = plane_s8x16(&tokens[2 * p][block_idx].values, chunk);
                    let a1 = plane_s8x16(&tokens[2 * p + 1][block_idx].values, chunk);
                    let (a_lo, a_hi) = pair_s8_lo_hi(a0, a1);
                    pair_acc[0] = smmla_s32(smmla_s32(pair_acc[0], w01_lo, a_lo), w01_hi, a_hi);
                    pair_acc[1] = smmla_s32(smmla_s32(pair_acc[1], w23_lo, a_lo), w23_hi, a_hi);
                }
            }
        }
    }
    (acc_pair, acc_odd)
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
fn pq2_4x_pair_scale_update<const FUSED: bool>(
    acc_pair: [int32x4_t; 2],
    t0: &super::Pq2Activation,
    t1: &super::Pq2Activation,
    rs01: core::arch::aarch64::float32x4_t,
    rs23: core::arch::aarch64::float32x4_t,
    dst01: &mut core::arch::aarch64::float32x4_t,
    dst23: &mut core::arch::aarch64::float32x4_t,
) {
    use core::arch::aarch64::{vcvtq_f32_s32, vmulq_f32};
    let sums_01 = load_s32x4(&[t0.sum, t1.sum, t0.sum, t1.sum]);
    let act_scales_01 = load_f32x4_neon(&[t0.scale, t1.scale, t0.scale, t1.scale]);
    let d01 = vmulq_f32(rs01, act_scales_01);
    let d23 = vmulq_f32(rs23, act_scales_01);
    let dots01 = vcvtq_f32_s32(vsubq_s32(acc_pair[0], sums_01));
    let dots23 = vcvtq_f32_s32(vsubq_s32(acc_pair[1], sums_01));
    *dst01 = f32x4_scale_step::<FUSED>(*dst01, d01, dots01);
    *dst23 = f32x4_scale_step::<FUSED>(*dst23, d23, dots23);
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
fn pq2_4x_odd_scale_update<const FUSED: bool>(
    acc_odd: [int32x4_t; 4],
    t_odd: &super::Pq2Activation,
    rs_vec: core::arch::aarch64::float32x4_t,
    dst_odd: &mut core::arch::aarch64::float32x4_t,
) {
    use core::arch::aarch64::{vcvtq_f32_s32, vdupq_n_f32, vmulq_f32};
    let d_odd = vmulq_f32(rs_vec, vdupq_n_f32(t_odd.scale));
    let reduced = vsubq_s32(
        reduce4_s32x4_addp(acc_odd[0], acc_odd[1], acc_odd[2], acc_odd[3]),
        vdupq_n_s32(t_odd.sum),
    );
    let dots_odd = vcvtq_f32_s32(reduced);
    *dst_odd = f32x4_scale_step::<FUSED>(*dst_odd, d_odd, dots_odd);
}

#[target_feature(enable = "neon,i8mm,dotprod")]
fn dot_scaled_8x_pairs_reg_i8mm<const FUSED: bool, const PAIRS: usize, const HAS_ODD: bool>(
    rows: [&[u8]; 8],
    tokens: &[&[super::Pq2Activation]],
    out: &mut [[f32; 8]; 8],
) -> Option<()> {
    use core::arch::aarch64::{vdupq_n_f32, vzip1q_f32, vzip2q_f32};
    let block_count = tokens[0].len();
    let mask = vdupq_n_u8(3);
    let row_chunks = rows.map(|row| row.as_chunks::<34>().0);
    let mut pair_acc = [[[vdupq_n_f32(0.0); 2]; PAIRS]; 2];
    let mut odd_acc = [vdupq_n_f32(0.0); 2];

    for block_idx in 0..block_count {
        prefetch_row_blocks(&row_chunks, block_idx, block_count);
        prefetch_pq2_tokens_block(tokens, block_idx, block_count);
        let blocks: [&[u8; 34]; 8] = std::array::from_fn(|r| &row_chunks[r][block_idx]);
        let row_scales = widen_row_scales_f16(&blocks)?;
        for (row_half, half_scales_arr) in row_scales.iter().enumerate() {
            let rbase = row_half * 4;
            let half_blocks = [
                blocks[rbase],
                blocks[rbase + 1],
                blocks[rbase + 2],
                blocks[rbase + 3],
            ];
            let (acc_pair, acc_odd) =
                pq2_4x_pairs_block_i8mm::<PAIRS, HAS_ODD>(half_blocks, tokens, block_idx, mask);
            let rs_vec = load_f32x4_neon(half_scales_arr);
            let rs01 = vzip1q_f32(rs_vec, rs_vec);
            let rs23 = vzip2q_f32(rs_vec, rs_vec);
            for p in 0..PAIRS {
                let [mut dst01, mut dst23] = pair_acc[row_half][p];
                pq2_4x_pair_scale_update::<FUSED>(
                    acc_pair[p],
                    &tokens[2 * p][block_idx],
                    &tokens[2 * p + 1][block_idx],
                    rs01,
                    rs23,
                    &mut dst01,
                    &mut dst23,
                );
                pair_acc[row_half][p] = [dst01, dst23];
            }
            if HAS_ODD {
                pq2_4x_odd_scale_update::<FUSED>(
                    acc_odd,
                    &tokens[PAIRS * 2][block_idx],
                    rs_vec,
                    &mut odd_acc[row_half],
                );
            }
        }
    }

    *out = [[0.0_f32; 8]; 8];
    for row_half in 0..2 {
        let rbase = row_half * 4;
        for p in 0..PAIRS {
            let mut v01 = [0.0_f32; 4];
            let mut v23 = [0.0_f32; 4];
            store_f32x4_neon(pair_acc[row_half][p][0], &mut v01);
            store_f32x4_neon(pair_acc[row_half][p][1], &mut v23);
            out[rbase][2 * p] = v01[0];
            out[rbase][2 * p + 1] = v01[1];
            out[rbase + 1][2 * p] = v01[2];
            out[rbase + 1][2 * p + 1] = v01[3];
            out[rbase + 2][2 * p] = v23[0];
            out[rbase + 2][2 * p + 1] = v23[1];
            out[rbase + 3][2 * p] = v23[2];
            out[rbase + 3][2 * p + 1] = v23[3];
        }
        if HAS_ODD {
            let mut vo = [0.0_f32; 4];
            store_f32x4_neon(odd_acc[row_half], &mut vo);
            for r in 0..4 {
                out[rbase + r][PAIRS * 2] = vo[r];
            }
        }
    }
    out.iter().flatten().all(|v| v.is_finite()).then_some(())
}

#[target_feature(enable = "neon,i8mm,dotprod")]
fn dot_scaled_8x_pairs_stack_i8mm<const FUSED: bool, const PAIRS: usize, const HAS_ODD: bool>(
    rows: [&[u8]; 8],
    tokens: &[&[super::Pq2Activation]],
    out: &mut [[f32; 8]; 8],
) -> Option<()> {
    use core::arch::aarch64::{vzip1q_f32, vzip2q_f32};
    let block_count = tokens[0].len();
    let mask = vdupq_n_u8(3);
    let row_chunks = rows.map(|row| row.as_chunks::<34>().0);
    let mut pair_buf = [[[[0.0_f32; 4]; 2]; PAIRS]; 2];
    let mut odd_buf = [[0.0_f32; 4]; 2];

    for block_idx in 0..block_count {
        prefetch_row_blocks(&row_chunks, block_idx, block_count);
        prefetch_pq2_tokens_block(tokens, block_idx, block_count);
        let blocks: [&[u8; 34]; 8] = std::array::from_fn(|r| &row_chunks[r][block_idx]);
        let row_scales = widen_row_scales_f16(&blocks)?;
        for (row_half, half_scales_arr) in row_scales.iter().enumerate() {
            let rbase = row_half * 4;
            let half_blocks = [
                blocks[rbase],
                blocks[rbase + 1],
                blocks[rbase + 2],
                blocks[rbase + 3],
            ];
            let (acc_pair, acc_odd) =
                pq2_4x_pairs_block_i8mm::<PAIRS, HAS_ODD>(half_blocks, tokens, block_idx, mask);
            let rs_vec = load_f32x4_neon(half_scales_arr);
            let rs01 = vzip1q_f32(rs_vec, rs_vec);
            let rs23 = vzip2q_f32(rs_vec, rs_vec);
            for p in 0..PAIRS {
                let mut dst01 = load_f32x4_neon(&pair_buf[row_half][p][0]);
                let mut dst23 = load_f32x4_neon(&pair_buf[row_half][p][1]);
                pq2_4x_pair_scale_update::<FUSED>(
                    acc_pair[p],
                    &tokens[2 * p][block_idx],
                    &tokens[2 * p + 1][block_idx],
                    rs01,
                    rs23,
                    &mut dst01,
                    &mut dst23,
                );
                store_f32x4_neon(dst01, &mut pair_buf[row_half][p][0]);
                store_f32x4_neon(dst23, &mut pair_buf[row_half][p][1]);
            }
            if HAS_ODD {
                let mut dst_odd = load_f32x4_neon(&odd_buf[row_half]);
                pq2_4x_odd_scale_update::<FUSED>(
                    acc_odd,
                    &tokens[PAIRS * 2][block_idx],
                    rs_vec,
                    &mut dst_odd,
                );
                store_f32x4_neon(dst_odd, &mut odd_buf[row_half]);
            }
        }
    }

    *out = [[0.0_f32; 8]; 8];
    for row_half in 0..2 {
        let rbase = row_half * 4;
        for p in 0..PAIRS {
            let v01 = pair_buf[row_half][p][0];
            let v23 = pair_buf[row_half][p][1];
            out[rbase][2 * p] = v01[0];
            out[rbase][2 * p + 1] = v01[1];
            out[rbase + 1][2 * p] = v01[2];
            out[rbase + 1][2 * p + 1] = v01[3];
            out[rbase + 2][2 * p] = v23[0];
            out[rbase + 2][2 * p + 1] = v23[1];
            out[rbase + 3][2 * p] = v23[2];
            out[rbase + 3][2 * p + 1] = v23[3];
        }
        if HAS_ODD {
            for r in 0..4 {
                out[rbase + r][PAIRS * 2] = odd_buf[row_half][r];
            }
        }
    }
    out.iter().flatten().all(|v| v.is_finite()).then_some(())
}

#[target_feature(enable = "neon,i8mm,dotprod")]
fn dot_scaled_8x_tokens_i8mm<const FUSED: bool>(
    rows: [&[u8]; 8],
    tokens: &[&[super::Pq2Activation]],
    out: &mut [[f32; 8]; 8],
) -> Option<()> {
    match tokens.len() {
        1 => dot_scaled_8x_1_dotprod::<FUSED>(rows, tokens[0], out),
        2 => dot_scaled_8x_pairs_reg_i8mm::<FUSED, 1, false>(rows, tokens, out),
        3 => dot_scaled_8x_pairs_reg_i8mm::<FUSED, 1, true>(rows, tokens, out),
        4 => dot_scaled_8x_pairs_reg_i8mm::<FUSED, 2, false>(rows, tokens, out),
        5 => dot_scaled_8x_pairs_stack_i8mm::<FUSED, 2, true>(rows, tokens, out),
        6 => dot_scaled_8x_pairs_stack_i8mm::<FUSED, 3, false>(rows, tokens, out),
        7 => dot_scaled_8x_pairs_stack_i8mm::<FUSED, 3, true>(rows, tokens, out),
        8 => dot_scaled_8x_pairs_stack_i8mm::<FUSED, 4, false>(rows, tokens, out),
        _ => None,
    }
}

#[target_feature(enable = "neon,dotprod")]
fn dot_scaled_8x_1_dotprod<const FUSED: bool>(
    rows: [&[u8]; 8],
    activation: &[super::Pq2Activation],
    out: &mut [[f32; 8]; 8],
) -> Option<()> {
    use core::arch::aarch64::{vaddq_f32, vcvtq_f32_s32, vdupq_n_f32, vfmaq_f32, vmulq_f32};
    let mask = vdupq_n_u8(3);
    let row_chunks = rows.map(|row| row.as_chunks::<34>().0);
    let mut sum_lo = vdupq_n_f32(0.0);
    let mut sum_hi = vdupq_n_f32(0.0);
    for (block_idx, act) in activation.iter().enumerate() {
        let blocks: [&[u8; 34]; 8] = std::array::from_fn(|r| &row_chunks[r][block_idx]);
        let [scales_lo_arr, scales_hi_arr] = widen_row_scales_f16(&blocks)?;
        let act_scale = vdupq_n_f32(act.scale);
        let d_lo = vmulq_f32(load_f32x4_neon(&scales_lo_arr), act_scale);
        let d_hi = vmulq_f32(load_f32x4_neon(&scales_hi_arr), act_scale);
        let mut acc = [vdupq_n_s32(0); 8];
        for (half, values) in act.values.as_chunks::<64>().0.iter().enumerate() {
            let planes = values.as_chunks::<16>().0;
            let p0 = load_s8x16(&planes[0]);
            let p1 = load_s8x16(&planes[1]);
            let p2 = load_s8x16(&planes[2]);
            let p3 = load_s8x16(&planes[3]);
            for (r, acc_slot) in acc.iter_mut().enumerate() {
                let [_, _, ref packed @ ..] = *blocks[r];
                let bytes = &packed.as_chunks::<16>().0[half];
                *acc_slot = pq2_half_unsigned_sdot_masked(
                    *acc_slot,
                    load_u8x16(bytes),
                    p0,
                    p1,
                    p2,
                    p3,
                    mask,
                );
            }
        }
        let correction = vdupq_n_s32(act.sum);
        let dot_lo = vcvtq_f32_s32(vsubq_s32(
            reduce4_s32x4_addp(acc[0], acc[1], acc[2], acc[3]),
            correction,
        ));
        let dot_hi = vcvtq_f32_s32(vsubq_s32(
            reduce4_s32x4_addp(acc[4], acc[5], acc[6], acc[7]),
            correction,
        ));
        if FUSED {
            sum_lo = vfmaq_f32(sum_lo, d_lo, dot_lo);
            sum_hi = vfmaq_f32(sum_hi, d_hi, dot_hi);
        } else {
            sum_lo = vaddq_f32(sum_lo, vmulq_f32(d_lo, dot_lo));
            sum_hi = vaddq_f32(sum_hi, vmulq_f32(d_hi, dot_hi));
        }
    }
    *out = [[0.0_f32; 8]; 8];
    let mut lo_arr = [0.0_f32; 4];
    let mut hi_arr = [0.0_f32; 4];
    store_f32x4_neon(sum_lo, &mut lo_arr);
    store_f32x4_neon(sum_hi, &mut hi_arr);
    for r in 0..4 {
        out[r][0] = lo_arr[r];
        out[r + 4][0] = hi_arr[r];
    }
    out.iter().flatten().all(|v| v.is_finite()).then_some(())
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
fn pq2_4x_n_block_dotprod<const N: usize>(
    raw_halves: [&[u8; 34]; 4],
    tokens: &[&[super::Pq2Activation]],
    block_idx: usize,
    mask: uint8x16_t,
) -> [int32x4_t; 4] {
    let zero = vdupq_n_s32(0);
    let mut acc = [[zero; N]; 4];
    for half in 0..2 {
        let raw_rows = [
            load_u8x16(&raw_halves[0][2..].as_chunks::<16>().0[half]),
            load_u8x16(&raw_halves[1][2..].as_chunks::<16>().0[half]),
            load_u8x16(&raw_halves[2][2..].as_chunks::<16>().0[half]),
            load_u8x16(&raw_halves[3][2..].as_chunks::<16>().0[half]),
        ];
        let plane_base = half * 4;
        for plane in 0..4 {
            let chunk = plane_base + plane;
            let acts: [int8x16_t; N] =
                std::array::from_fn(|t| plane_s8x16(&tokens[t][block_idx].values, chunk));
            for (r, &raw_row) in raw_rows.iter().enumerate() {
                let w = unpack_pq2_plane_s8(raw_row, plane, mask);
                for (t, &a) in acts.iter().enumerate() {
                    acc[r][t] = vdotq_s32(acc[r][t], w, a);
                }
            }
        }
    }
    let sums_arr: [i32; 4] =
        std::array::from_fn(|t| if t < N { tokens[t][block_idx].sum } else { 0 });
    let sums_vec = load_s32x4(&sums_arr);
    std::array::from_fn(|r| {
        let v0 = acc[r][0];
        let v1 = acc[r].get(1).copied().unwrap_or(zero);
        let v2 = acc[r].get(2).copied().unwrap_or(zero);
        let v3 = acc[r].get(3).copied().unwrap_or(zero);
        vsubq_s32(reduce4_s32x4_addp(v0, v1, v2, v3), sums_vec)
    })
}

#[target_feature(enable = "neon,dotprod")]
fn dot_scaled_8x_quad_dotprod<const FUSED: bool, const N_TOK: usize>(
    rows: [&[u8]; 8],
    tokens: &[&[super::Pq2Activation]],
    out: &mut [[f32; 8]; 8],
) -> Option<()> {
    use core::arch::aarch64::{vcvtq_f32_s32, vdupq_n_f32, vmulq_f32};
    let block_count = tokens[0].len();
    let mask = vdupq_n_u8(3);
    let row_chunks = rows.map(|row| row.as_chunks::<34>().0);
    let mut f_acc = [[vdupq_n_f32(0.0); 4]; 2];
    for block_idx in 0..block_count {
        prefetch_row_blocks(&row_chunks, block_idx, block_count);
        prefetch_pq2_tokens_block(tokens, block_idx, block_count);
        let blocks: [&[u8; 34]; 8] = std::array::from_fn(|r| &row_chunks[r][block_idx]);
        let row_scales = widen_row_scales_f16(&blocks)?;
        let act_scales_arr: [f32; 4] = std::array::from_fn(|t| {
            if t < N_TOK {
                tokens[t][block_idx].scale
            } else {
                0.0
            }
        });
        let act_scales = load_f32x4_neon(&act_scales_arr);
        for (row_half, half_scales_arr) in row_scales.iter().enumerate() {
            let rbase = row_half * 4;
            let half_blocks = [
                blocks[rbase],
                blocks[rbase + 1],
                blocks[rbase + 2],
                blocks[rbase + 3],
            ];
            let dots =
                pq2_4x_n_block_dotprod::<N_TOK>(half_blocks, &tokens[..N_TOK], block_idx, mask);
            for (r, (&dot_vec, &row_scale)) in dots.iter().zip(half_scales_arr.iter()).enumerate() {
                let d = vmulq_f32(vdupq_n_f32(row_scale), act_scales);
                let floats = vcvtq_f32_s32(dot_vec);
                f_acc[row_half][r] = f32x4_scale_step::<FUSED>(f_acc[row_half][r], d, floats);
            }
        }
    }
    *out = [[0.0_f32; 8]; 8];
    for (row_half, half_acc) in f_acc.iter().enumerate() {
        let rbase = row_half * 4;
        for (r, &vec) in half_acc.iter().enumerate() {
            let mut tmp = [0.0_f32; 4];
            store_f32x4_neon(vec, &mut tmp);
            out[rbase + r][..N_TOK].copy_from_slice(&tmp[..N_TOK]);
        }
    }
    out.iter().flatten().all(|v| v.is_finite()).then_some(())
}

#[target_feature(enable = "neon,dotprod")]
fn dot_scaled_8x_oct_dotprod<const FUSED: bool, const TAIL_TOK: usize>(
    rows: [&[u8]; 8],
    tokens: &[&[super::Pq2Activation]],
    out: &mut [[f32; 8]; 8],
) -> Option<()> {
    *out = [[0.0_f32; 8]; 8];
    let block_count = tokens[0].len();
    let mask = vdupq_n_u8(3);
    let row_chunks = rows.map(|row| row.as_chunks::<34>().0);
    for block_idx in 0..block_count {
        prefetch_row_blocks(&row_chunks, block_idx, block_count);
        prefetch_pq2_tokens_block(tokens, block_idx, block_count);
        let blocks: [&[u8; 34]; 8] = std::array::from_fn(|r| &row_chunks[r][block_idx]);
        let row_scales = widen_row_scales_f16(&blocks)?;
        let act_lo_arr: [f32; 4] = std::array::from_fn(|t| tokens[t][block_idx].scale);
        let act_hi_arr: [f32; 4] = std::array::from_fn(|t| {
            if t < TAIL_TOK {
                tokens[4 + t][block_idx].scale
            } else {
                0.0
            }
        });
        let act_lo = load_f32x4_neon(&act_lo_arr);
        let act_hi = load_f32x4_neon(&act_hi_arr);
        for (row_half, half_scales_arr) in row_scales.iter().enumerate() {
            let rbase = row_half * 4;
            let half_blocks = [
                blocks[rbase],
                blocks[rbase + 1],
                blocks[rbase + 2],
                blocks[rbase + 3],
            ];
            let dots_lo = pq2_4x_n_block_dotprod::<4>(half_blocks, &tokens[..4], block_idx, mask);
            for (r, (&dot_vec, &row_scale)) in
                dots_lo.iter().zip(half_scales_arr.iter()).enumerate()
            {
                let quads = out[rbase + r].as_chunks_mut::<4>().0;
                update_scaled_row4::<FUSED>(dot_vec, row_scale, act_lo, &mut quads[0]);
            }
            let dots_hi = pq2_4x_n_block_dotprod::<TAIL_TOK>(
                half_blocks,
                &tokens[4..4 + TAIL_TOK],
                block_idx,
                mask,
            );
            for (r, (&dot_vec, &row_scale)) in
                dots_hi.iter().zip(half_scales_arr.iter()).enumerate()
            {
                let quads = out[rbase + r].as_chunks_mut::<4>().0;
                update_scaled_row4::<FUSED>(dot_vec, row_scale, act_hi, &mut quads[1]);
            }
        }
    }
    let total_tok = 4 + TAIL_TOK;
    for row in out.iter_mut() {
        row[total_tok..].fill(0.0);
    }
    out.iter().flatten().all(|v| v.is_finite()).then_some(())
}

#[target_feature(enable = "neon,dotprod")]
fn dot_scaled_8x_tokens_dotprod<const FUSED: bool>(
    rows: [&[u8]; 8],
    tokens: &[&[super::Pq2Activation]],
    out: &mut [[f32; 8]; 8],
) -> Option<()> {
    match tokens.len() {
        1 => dot_scaled_8x_1_dotprod::<FUSED>(rows, tokens[0], out),
        2 => dot_scaled_8x_quad_dotprod::<FUSED, 2>(rows, tokens, out),
        3 => dot_scaled_8x_quad_dotprod::<FUSED, 3>(rows, tokens, out),
        4 => dot_scaled_8x_quad_dotprod::<FUSED, 4>(rows, tokens, out),
        5 => dot_scaled_8x_oct_dotprod::<FUSED, 1>(rows, tokens, out),
        6 => dot_scaled_8x_oct_dotprod::<FUSED, 2>(rows, tokens, out),
        7 => dot_scaled_8x_oct_dotprod::<FUSED, 3>(rows, tokens, out),
        8 => dot_scaled_8x_oct_dotprod::<FUSED, 4>(rows, tokens, out),
        _ => None,
    }
}

#[inline]
pub(super) fn pq2_detected_dot_scaled_8x_tokens<const FUSED: bool>(
    detected: super::Pq2Dotprod,
    rows: [&[u8]; 8],
    tokens: &[&[super::Pq2Activation]],
    out: &mut [[f32; 8]; 8],
) -> Option<()> {
    if detected.i8mm {
        // SAFETY: `detected.i8mm` was enabled after CPU/OS `i8mm` and `dotprod`
        // detection, and the safe parent checked token/row extents and finite scales.
        unsafe { dot_scaled_8x_tokens_i8mm::<FUSED>(rows, tokens, out) }
    } else {
        // SAFETY: Constructing the unforgeable `Pq2Dotprod` verified CPU/OS
        // `dotprod` support, and the safe parent checked all slice invariants.
        unsafe { dot_scaled_8x_tokens_dotprod::<FUSED>(rows, tokens, out) }
    }
}

#[inline]
pub(super) fn pq2_detected_dot_eight_tokens(
    detected: super::Pq2Dotprod,
    packed: [&[u8; 32]; 8],
    planes: [&[i8; 128]; 8],
    activation_sums: [i32; 8],
) -> [[i32; 8]; 8] {
    if detected.i8mm {
        // SAFETY: This private capability flag was set only after CPU/OS I8MM
        // detection. Fixed array references provide every required input byte.
        unsafe { dot_pq2_eight_rows_eight_tokens_precomputed_i8mm(packed, planes, activation_sums) }
    } else {
        // SAFETY: Constructing the unforgeable capability verified CPU/OS
        // dotprod eligibility. All inputs have the kernel's fixed array sizes.
        unsafe {
            dot_pq2_eight_rows_eight_tokens_precomputed_dotprod(packed, planes, activation_sums)
        }
    }
}

#[inline]
pub(super) fn dot_pq2_four_rows_four_tokens_precomputed(
    packed: [&[u8; 32]; 4],
    planes: [&[i8; 128]; 4],
    activation_sums: [i32; 4],
) -> Option<[[i32; 4]; 4]> {
    if has_i8mm() {
        // SAFETY: `has_i8mm()` verified that the CPU supports `i8mm` and baseline `neon`.
        return Some(unsafe {
            dot_pq2_four_rows_four_tokens_precomputed_i8mm(packed, planes, activation_sums)
        });
    }
    if has_dotprod() {
        // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
        return Some(unsafe {
            dot_pq2_four_rows_four_tokens_precomputed_dotprod(packed, planes, activation_sums)
        });
    }
    None
}

#[inline]
pub(super) fn dot_pq2_eight_rows_four_tokens_precomputed(
    packed: [&[u8; 32]; 8],
    planes: [&[i8; 128]; 4],
    activation_sums: [i32; 4],
) -> Option<[[i32; 4]; 8]> {
    if has_i8mm() {
        // SAFETY: `has_i8mm()` verified that the CPU supports `i8mm` and baseline `neon`.
        return Some(unsafe {
            dot_pq2_eight_rows_four_tokens_precomputed_i8mm(packed, planes, activation_sums)
        });
    }
    if has_dotprod() {
        // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
        return Some(unsafe {
            dot_pq2_eight_rows_four_tokens_precomputed_dotprod(packed, planes, activation_sums)
        });
    }
    None
}

#[inline]
pub(super) fn dot_pq2_eight_rows_eight_tokens_precomputed(
    packed: [&[u8; 32]; 8],
    planes: [&[i8; 128]; 8],
    activation_sums: [i32; 8],
) -> Option<[[i32; 8]; 8]> {
    if has_i8mm() {
        // SAFETY: `has_i8mm()` verified that the CPU supports `i8mm` and baseline `neon`.
        return Some(unsafe {
            dot_pq2_eight_rows_eight_tokens_precomputed_i8mm(packed, planes, activation_sums)
        });
    }
    if has_dotprod() {
        // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
        return Some(unsafe {
            dot_pq2_eight_rows_eight_tokens_precomputed_dotprod(packed, planes, activation_sums)
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

#[target_feature(enable = "neon,dotprod")]
fn dot_ptq1_4x4_lanes_dotprod(
    trits: &[[[i8; 16]; 8]; 4],
    activations: [[&[i8; 32]; 4]; 4],
) -> super::Ptq1FourRowFourTokenLanes {
    let mut out = [[[[0_i32; 4]; 4]; 4]; 4];
    for group in 0..4 {
        let w0 = [
            load_s8x16(&trits[0][group * 2]),
            load_s8x16(&trits[1][group * 2]),
            load_s8x16(&trits[2][group * 2]),
            load_s8x16(&trits[3][group * 2]),
        ];
        let w1 = [
            load_s8x16(&trits[0][group * 2 + 1]),
            load_s8x16(&trits[1][group * 2 + 1]),
            load_s8x16(&trits[2][group * 2 + 1]),
            load_s8x16(&trits[3][group * 2 + 1]),
        ];
        let acts: [(int8x16_t, int8x16_t); 4] = std::array::from_fn(|t| {
            let halves = activations[t][group].as_chunks::<16>().0;
            (load_s8x16(&halves[0]), load_s8x16(&halves[1]))
        });
        for r in 0..4 {
            for t in 0..4 {
                let lanes = sdot32_vec(w0[r], w1[r], acts[t].0, acts[t].1);
                store_s32x4(lanes, &mut out[r][t][group]);
            }
        }
    }
    out
}

#[target_feature(enable = "neon,i8mm")]
fn dot_ptq1_4x4_groups_i8mm(
    trits: &[[[i8; 16]; 8]; 4],
    activations: [[&[i8; 32]; 4]; 4],
) -> super::Ptq1FourRowFourTokenGroups {
    let mut out = [[[0_i32; 4]; 4]; 4];
    for group in 0..4 {
        let mut acc00 = vdupq_n_s32(0);
        let mut acc01 = vdupq_n_s32(0);
        let mut acc10 = vdupq_n_s32(0);
        let mut acc11 = vdupq_n_s32(0);
        let a0_halves = activations[0][group].as_chunks::<16>().0;
        let a1_halves = activations[1][group].as_chunks::<16>().0;
        let a2_halves = activations[2][group].as_chunks::<16>().0;
        let a3_halves = activations[3][group].as_chunks::<16>().0;
        for half in 0..2 {
            let w0 = load_s8x16(&trits[0][group * 2 + half]);
            let w1 = load_s8x16(&trits[1][group * 2 + half]);
            let w2 = load_s8x16(&trits[2][group * 2 + half]);
            let w3 = load_s8x16(&trits[3][group * 2 + half]);
            let a0 = load_s8x16(&a0_halves[half]);
            let a1 = load_s8x16(&a1_halves[half]);
            let a2 = load_s8x16(&a2_halves[half]);
            let a3 = load_s8x16(&a3_halves[half]);

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
        let mut q00 = [0_i32; 4];
        let mut q01 = [0_i32; 4];
        let mut q10 = [0_i32; 4];
        let mut q11 = [0_i32; 4];
        store_s32x4(acc00, &mut q00);
        store_s32x4(acc01, &mut q01);
        store_s32x4(acc10, &mut q10);
        store_s32x4(acc11, &mut q11);
        out[0][0][group] = q00[0];
        out[0][1][group] = q00[1];
        out[0][2][group] = q01[0];
        out[0][3][group] = q01[1];
        out[1][0][group] = q00[2];
        out[1][1][group] = q00[3];
        out[1][2][group] = q01[2];
        out[1][3][group] = q01[3];
        out[2][0][group] = q10[0];
        out[2][1][group] = q10[1];
        out[2][2][group] = q11[0];
        out[2][3][group] = q11[1];
        out[3][0][group] = q10[2];
        out[3][1][group] = q10[3];
        out[3][2][group] = q11[2];
        out[3][3][group] = q11[3];
    }
    out
}

#[target_feature(enable = "neon,dotprod")]
fn dot_ptq1_4x4_groups_dotprod(
    trits: &[[[i8; 16]; 8]; 4],
    activations: [[&[i8; 32]; 4]; 4],
) -> super::Ptq1FourRowFourTokenGroups {
    let mut out = [[[0_i32; 4]; 4]; 4];
    for group in 0..4 {
        let w0 = [
            load_s8x16(&trits[0][group * 2]),
            load_s8x16(&trits[1][group * 2]),
            load_s8x16(&trits[2][group * 2]),
            load_s8x16(&trits[3][group * 2]),
        ];
        let w1 = [
            load_s8x16(&trits[0][group * 2 + 1]),
            load_s8x16(&trits[1][group * 2 + 1]),
            load_s8x16(&trits[2][group * 2 + 1]),
            load_s8x16(&trits[3][group * 2 + 1]),
        ];
        let acts: [(int8x16_t, int8x16_t); 4] = std::array::from_fn(|t| {
            let halves = activations[t][group].as_chunks::<16>().0;
            (load_s8x16(&halves[0]), load_s8x16(&halves[1]))
        });
        for r in 0..4 {
            let l0 = sdot32_vec(w0[r], w1[r], acts[0].0, acts[0].1);
            let l1 = sdot32_vec(w0[r], w1[r], acts[1].0, acts[1].1);
            let l2 = sdot32_vec(w0[r], w1[r], acts[2].0, acts[2].1);
            let l3 = sdot32_vec(w0[r], w1[r], acts[3].0, acts[3].1);
            let mut sums = [0_i32; 4];
            store_s32x4(reduce4_s32x4_addp(l0, l1, l2, l3), &mut sums);
            for t in 0..4 {
                out[r][t][group] = sums[t];
            }
        }
    }
    out
}

#[target_feature(enable = "neon,dotprod")]
fn dot_ptq1_4x1_lanes_dotprod(
    trits: &[[[i8; 16]; 8]; 4],
    activations: [&[i8; 32]; 4],
) -> [[[i32; 4]; 4]; 4] {
    let a0 = activations[0].as_chunks::<16>().0;
    let a1 = activations[1].as_chunks::<16>().0;
    let a2 = activations[2].as_chunks::<16>().0;
    let a3 = activations[3].as_chunks::<16>().0;
    let act_pairs = [
        (load_s8x16(&a0[0]), load_s8x16(&a0[1])),
        (load_s8x16(&a1[0]), load_s8x16(&a1[1])),
        (load_s8x16(&a2[0]), load_s8x16(&a2[1])),
        (load_s8x16(&a3[0]), load_s8x16(&a3[1])),
    ];
    let mut out = [[[0_i32; 4]; 4]; 4];
    for r in 0..4 {
        for (group, (act0, act1)) in act_pairs.into_iter().enumerate() {
            let acc = sdot32_vec(
                load_s8x16(&trits[r][group * 2]),
                load_s8x16(&trits[r][group * 2 + 1]),
                act0,
                act1,
            );
            store_s32x4(acc, &mut out[r][group]);
        }
    }
    out
}

#[target_feature(enable = "neon,dotprod")]
fn dot_ptq1_4x1_groups_dotprod(
    trits: &[[[i8; 16]; 8]; 4],
    activations: [&[i8; 32]; 4],
) -> [[i32; 4]; 4] {
    let a0 = activations[0].as_chunks::<16>().0;
    let a1 = activations[1].as_chunks::<16>().0;
    let a2 = activations[2].as_chunks::<16>().0;
    let a3 = activations[3].as_chunks::<16>().0;
    let act00 = load_s8x16(&a0[0]);
    let act01 = load_s8x16(&a0[1]);
    let act10 = load_s8x16(&a1[0]);
    let act11 = load_s8x16(&a1[1]);
    let act20 = load_s8x16(&a2[0]);
    let act21 = load_s8x16(&a2[1]);
    let act30 = load_s8x16(&a3[0]);
    let act31 = load_s8x16(&a3[1]);
    let mut out = [[0_i32; 4]; 4];
    for r in 0..4 {
        let g0 = sdot32_vec(
            load_s8x16(&trits[r][0]),
            load_s8x16(&trits[r][1]),
            act00,
            act01,
        );
        let g1 = sdot32_vec(
            load_s8x16(&trits[r][2]),
            load_s8x16(&trits[r][3]),
            act10,
            act11,
        );
        let g2 = sdot32_vec(
            load_s8x16(&trits[r][4]),
            load_s8x16(&trits[r][5]),
            act20,
            act21,
        );
        let g3 = sdot32_vec(
            load_s8x16(&trits[r][6]),
            load_s8x16(&trits[r][7]),
            act30,
            act31,
        );
        store_s32x4(reduce4_s32x4_addp(g0, g1, g2, g3), &mut out[r]);
    }
    out
}

#[target_feature(enable = "neon,dotprod")]
fn dot_ptq1_8x1_lanes_dotprod(
    trits: &[[[i8; 16]; 8]; 8],
    activations: [&[i8; 32]; 4],
) -> [[[i32; 4]; 4]; 8] {
    let a0 = activations[0].as_chunks::<16>().0;
    let a1 = activations[1].as_chunks::<16>().0;
    let a2 = activations[2].as_chunks::<16>().0;
    let a3 = activations[3].as_chunks::<16>().0;
    let act_pairs = [
        (load_s8x16(&a0[0]), load_s8x16(&a0[1])),
        (load_s8x16(&a1[0]), load_s8x16(&a1[1])),
        (load_s8x16(&a2[0]), load_s8x16(&a2[1])),
        (load_s8x16(&a3[0]), load_s8x16(&a3[1])),
    ];
    let mut out = [[[0_i32; 4]; 4]; 8];
    for r in 0..8 {
        for (group, (act0, act1)) in act_pairs.into_iter().enumerate() {
            let acc = sdot32_vec(
                load_s8x16(&trits[r][group * 2]),
                load_s8x16(&trits[r][group * 2 + 1]),
                act0,
                act1,
            );
            store_s32x4(acc, &mut out[r][group]);
        }
    }
    out
}

#[target_feature(enable = "neon,dotprod")]
fn dot_ptq1_8x1_groups_dotprod(
    trits: &[[[i8; 16]; 8]; 8],
    activations: [&[i8; 32]; 4],
) -> [[i32; 4]; 8] {
    let a0 = activations[0].as_chunks::<16>().0;
    let a1 = activations[1].as_chunks::<16>().0;
    let a2 = activations[2].as_chunks::<16>().0;
    let a3 = activations[3].as_chunks::<16>().0;
    let act00 = load_s8x16(&a0[0]);
    let act01 = load_s8x16(&a0[1]);
    let act10 = load_s8x16(&a1[0]);
    let act11 = load_s8x16(&a1[1]);
    let act20 = load_s8x16(&a2[0]);
    let act21 = load_s8x16(&a2[1]);
    let act30 = load_s8x16(&a3[0]);
    let act31 = load_s8x16(&a3[1]);
    let mut out = [[0_i32; 4]; 8];
    for r in 0..8 {
        let g0 = sdot32_vec(
            load_s8x16(&trits[r][0]),
            load_s8x16(&trits[r][1]),
            act00,
            act01,
        );
        let g1 = sdot32_vec(
            load_s8x16(&trits[r][2]),
            load_s8x16(&trits[r][3]),
            act10,
            act11,
        );
        let g2 = sdot32_vec(
            load_s8x16(&trits[r][4]),
            load_s8x16(&trits[r][5]),
            act20,
            act21,
        );
        let g3 = sdot32_vec(
            load_s8x16(&trits[r][6]),
            load_s8x16(&trits[r][7]),
            act30,
            act31,
        );
        store_s32x4(reduce4_s32x4_addp(g0, g1, g2, g3), &mut out[r]);
    }
    out
}

#[target_feature(enable = "neon,dotprod")]
fn dot_ptq1_8x4_lanes_dotprod(
    trits: &[[[i8; 16]; 8]; 8],
    activations: [[&[i8; 32]; 4]; 4],
) -> super::Ptq1EightRowFourTokenLanes {
    let mut out = [[[[0_i32; 4]; 4]; 4]; 8];
    for group in 0..4 {
        let acts: [(int8x16_t, int8x16_t); 4] = std::array::from_fn(|t| {
            let halves = activations[t][group].as_chunks::<16>().0;
            (load_s8x16(&halves[0]), load_s8x16(&halves[1]))
        });
        for r in 0..8 {
            let w0 = load_s8x16(&trits[r][group * 2]);
            let w1 = load_s8x16(&trits[r][group * 2 + 1]);
            for t in 0..4 {
                let lanes = sdot32_vec(w0, w1, acts[t].0, acts[t].1);
                store_s32x4(lanes, &mut out[r][t][group]);
            }
        }
    }
    out
}

#[target_feature(enable = "neon,i8mm")]
fn dot_ptq1_8x4_groups_i8mm(
    trits: &[[[i8; 16]; 8]; 8],
    activations: [[&[i8; 32]; 4]; 4],
) -> super::Ptq1EightRowFourTokenGroups {
    let mut out = [[[0_i32; 4]; 4]; 8];
    for group in 0..4 {
        let mut acc00 = vdupq_n_s32(0);
        let mut acc01 = vdupq_n_s32(0);
        let mut acc10 = vdupq_n_s32(0);
        let mut acc11 = vdupq_n_s32(0);
        let mut acc20 = vdupq_n_s32(0);
        let mut acc21 = vdupq_n_s32(0);
        let mut acc30 = vdupq_n_s32(0);
        let mut acc31 = vdupq_n_s32(0);
        let a0_halves = activations[0][group].as_chunks::<16>().0;
        let a1_halves = activations[1][group].as_chunks::<16>().0;
        let a2_halves = activations[2][group].as_chunks::<16>().0;
        let a3_halves = activations[3][group].as_chunks::<16>().0;
        for half in 0..2 {
            let a0 = load_s8x16(&a0_halves[half]);
            let a1 = load_s8x16(&a1_halves[half]);
            let a2 = load_s8x16(&a2_halves[half]);
            let a3 = load_s8x16(&a3_halves[half]);
            let a01_lo = vcombine_s8(vget_low_s8(a0), vget_low_s8(a1));
            let a01_hi = vcombine_s8(vget_high_s8(a0), vget_high_s8(a1));
            let a23_lo = vcombine_s8(vget_low_s8(a2), vget_low_s8(a3));
            let a23_hi = vcombine_s8(vget_high_s8(a2), vget_high_s8(a3));

            let w0 = load_s8x16(&trits[0][group * 2 + half]);
            let w1 = load_s8x16(&trits[1][group * 2 + half]);
            let w2 = load_s8x16(&trits[2][group * 2 + half]);
            let w3 = load_s8x16(&trits[3][group * 2 + half]);
            let w01_lo = vcombine_s8(vget_low_s8(w0), vget_low_s8(w1));
            let w01_hi = vcombine_s8(vget_high_s8(w0), vget_high_s8(w1));
            let w23_lo = vcombine_s8(vget_low_s8(w2), vget_low_s8(w3));
            let w23_hi = vcombine_s8(vget_high_s8(w2), vget_high_s8(w3));

            acc00 = smmla_s32(smmla_s32(acc00, w01_lo, a01_lo), w01_hi, a01_hi);
            acc01 = smmla_s32(smmla_s32(acc01, w01_lo, a23_lo), w01_hi, a23_hi);
            acc10 = smmla_s32(smmla_s32(acc10, w23_lo, a01_lo), w23_hi, a01_hi);
            acc11 = smmla_s32(smmla_s32(acc11, w23_lo, a23_lo), w23_hi, a23_hi);

            let w4 = load_s8x16(&trits[4][group * 2 + half]);
            let w5 = load_s8x16(&trits[5][group * 2 + half]);
            let w6 = load_s8x16(&trits[6][group * 2 + half]);
            let w7 = load_s8x16(&trits[7][group * 2 + half]);
            let w45_lo = vcombine_s8(vget_low_s8(w4), vget_low_s8(w5));
            let w45_hi = vcombine_s8(vget_high_s8(w4), vget_high_s8(w5));
            let w67_lo = vcombine_s8(vget_low_s8(w6), vget_low_s8(w7));
            let w67_hi = vcombine_s8(vget_high_s8(w6), vget_high_s8(w7));

            acc20 = smmla_s32(smmla_s32(acc20, w45_lo, a01_lo), w45_hi, a01_hi);
            acc21 = smmla_s32(smmla_s32(acc21, w45_lo, a23_lo), w45_hi, a23_hi);
            acc30 = smmla_s32(smmla_s32(acc30, w67_lo, a01_lo), w67_hi, a01_hi);
            acc31 = smmla_s32(smmla_s32(acc31, w67_lo, a23_lo), w67_hi, a23_hi);
        }
        for (pair_idx, (acc_lo, acc_hi)) in [
            (acc00, acc01),
            (acc10, acc11),
            (acc20, acc21),
            (acc30, acc31),
        ]
        .into_iter()
        .enumerate()
        {
            let mut q_lo = [0_i32; 4];
            let mut q_hi = [0_i32; 4];
            store_s32x4(acc_lo, &mut q_lo);
            store_s32x4(acc_hi, &mut q_hi);
            let r0 = pair_idx * 2;
            let r1 = r0 + 1;
            out[r0][0][group] = q_lo[0];
            out[r0][1][group] = q_lo[1];
            out[r0][2][group] = q_hi[0];
            out[r0][3][group] = q_hi[1];
            out[r1][0][group] = q_lo[2];
            out[r1][1][group] = q_lo[3];
            out[r1][2][group] = q_hi[2];
            out[r1][3][group] = q_hi[3];
        }
    }
    out
}

#[target_feature(enable = "neon,dotprod")]
fn dot_ptq1_8x4_groups_dotprod(
    trits: &[[[i8; 16]; 8]; 8],
    activations: [[&[i8; 32]; 4]; 4],
) -> super::Ptq1EightRowFourTokenGroups {
    let mut out = [[[0_i32; 4]; 4]; 8];
    for group in 0..4 {
        let acts: [(int8x16_t, int8x16_t); 4] = std::array::from_fn(|t| {
            let halves = activations[t][group].as_chunks::<16>().0;
            (load_s8x16(&halves[0]), load_s8x16(&halves[1]))
        });
        for r in 0..8 {
            let w0 = load_s8x16(&trits[r][group * 2]);
            let w1 = load_s8x16(&trits[r][group * 2 + 1]);
            let l0 = sdot32_vec(w0, w1, acts[0].0, acts[0].1);
            let l1 = sdot32_vec(w0, w1, acts[1].0, acts[1].1);
            let l2 = sdot32_vec(w0, w1, acts[2].0, acts[2].1);
            let l3 = sdot32_vec(w0, w1, acts[3].0, acts[3].1);
            let mut sums = [0_i32; 4];
            store_s32x4(reduce4_s32x4_addp(l0, l1, l2, l3), &mut sums);
            for t in 0..4 {
                out[r][t][group] = sums[t];
            }
        }
    }
    out
}

#[target_feature(enable = "neon,dotprod")]
fn dot_ptq1_8x8_lanes_dotprod(
    trits: &[[[i8; 16]; 8]; 8],
    activations: [[&[i8; 32]; 4]; 8],
) -> super::Ptq1EightRowEightTokenLanes {
    let mut out = [[[[0_i32; 4]; 4]; 8]; 8];
    for group in 0..4 {
        let acts: [(int8x16_t, int8x16_t); 8] = std::array::from_fn(|t| {
            let halves = activations[t][group].as_chunks::<16>().0;
            (load_s8x16(&halves[0]), load_s8x16(&halves[1]))
        });
        for r in 0..8 {
            let w0 = load_s8x16(&trits[r][group * 2]);
            let w1 = load_s8x16(&trits[r][group * 2 + 1]);
            for t in 0..8 {
                let lanes = sdot32_vec(w0, w1, acts[t].0, acts[t].1);
                store_s32x4(lanes, &mut out[r][t][group]);
            }
        }
    }
    out
}

#[target_feature(enable = "neon,i8mm")]
#[expect(
    clippy::too_many_lines,
    reason = "Keep 8 weight rows in registers across both 4-token halves of the 8x8 I8MM tile."
)]
fn dot_ptq1_8x8_groups_i8mm(
    trits: &[[[i8; 16]; 8]; 8],
    activations: [[&[i8; 32]; 4]; 8],
) -> super::Ptq1EightRowEightTokenGroups {
    let mut out = [[[0_i32; 4]; 8]; 8];
    for group in 0..4 {
        let mut acc00 = vdupq_n_s32(0);
        let mut acc01 = vdupq_n_s32(0);
        let mut acc02 = vdupq_n_s32(0);
        let mut acc03 = vdupq_n_s32(0);
        let mut acc10 = vdupq_n_s32(0);
        let mut acc11 = vdupq_n_s32(0);
        let mut acc12 = vdupq_n_s32(0);
        let mut acc13 = vdupq_n_s32(0);
        let mut acc20 = vdupq_n_s32(0);
        let mut acc21 = vdupq_n_s32(0);
        let mut acc22 = vdupq_n_s32(0);
        let mut acc23 = vdupq_n_s32(0);
        let mut acc30 = vdupq_n_s32(0);
        let mut acc31 = vdupq_n_s32(0);
        let mut acc32 = vdupq_n_s32(0);
        let mut acc33 = vdupq_n_s32(0);
        let a_halves: [&[[i8; 16]]; 8] =
            std::array::from_fn(|t| activations[t][group].as_chunks::<16>().0);
        for half in 0..2 {
            let w0 = load_s8x16(&trits[0][group * 2 + half]);
            let w1 = load_s8x16(&trits[1][group * 2 + half]);
            let w2 = load_s8x16(&trits[2][group * 2 + half]);
            let w3 = load_s8x16(&trits[3][group * 2 + half]);
            let w4 = load_s8x16(&trits[4][group * 2 + half]);
            let w5 = load_s8x16(&trits[5][group * 2 + half]);
            let w6 = load_s8x16(&trits[6][group * 2 + half]);
            let w7 = load_s8x16(&trits[7][group * 2 + half]);
            let w01_lo = vcombine_s8(vget_low_s8(w0), vget_low_s8(w1));
            let w01_hi = vcombine_s8(vget_high_s8(w0), vget_high_s8(w1));
            let w23_lo = vcombine_s8(vget_low_s8(w2), vget_low_s8(w3));
            let w23_hi = vcombine_s8(vget_high_s8(w2), vget_high_s8(w3));
            let w45_lo = vcombine_s8(vget_low_s8(w4), vget_low_s8(w5));
            let w45_hi = vcombine_s8(vget_high_s8(w4), vget_high_s8(w5));
            let w67_lo = vcombine_s8(vget_low_s8(w6), vget_low_s8(w7));
            let w67_hi = vcombine_s8(vget_high_s8(w6), vget_high_s8(w7));

            let a0 = load_s8x16(&a_halves[0][half]);
            let a1 = load_s8x16(&a_halves[1][half]);
            let a2 = load_s8x16(&a_halves[2][half]);
            let a3 = load_s8x16(&a_halves[3][half]);
            let a01_lo = vcombine_s8(vget_low_s8(a0), vget_low_s8(a1));
            let a01_hi = vcombine_s8(vget_high_s8(a0), vget_high_s8(a1));
            let a23_lo = vcombine_s8(vget_low_s8(a2), vget_low_s8(a3));
            let a23_hi = vcombine_s8(vget_high_s8(a2), vget_high_s8(a3));

            acc00 = smmla_s32(smmla_s32(acc00, w01_lo, a01_lo), w01_hi, a01_hi);
            acc01 = smmla_s32(smmla_s32(acc01, w01_lo, a23_lo), w01_hi, a23_hi);
            acc10 = smmla_s32(smmla_s32(acc10, w23_lo, a01_lo), w23_hi, a01_hi);
            acc11 = smmla_s32(smmla_s32(acc11, w23_lo, a23_lo), w23_hi, a23_hi);
            acc20 = smmla_s32(smmla_s32(acc20, w45_lo, a01_lo), w45_hi, a01_hi);
            acc21 = smmla_s32(smmla_s32(acc21, w45_lo, a23_lo), w45_hi, a23_hi);
            acc30 = smmla_s32(smmla_s32(acc30, w67_lo, a01_lo), w67_hi, a01_hi);
            acc31 = smmla_s32(smmla_s32(acc31, w67_lo, a23_lo), w67_hi, a23_hi);

            let a4 = load_s8x16(&a_halves[4][half]);
            let a5 = load_s8x16(&a_halves[5][half]);
            let a6 = load_s8x16(&a_halves[6][half]);
            let a7 = load_s8x16(&a_halves[7][half]);
            let a45_lo = vcombine_s8(vget_low_s8(a4), vget_low_s8(a5));
            let a45_hi = vcombine_s8(vget_high_s8(a4), vget_high_s8(a5));
            let a67_lo = vcombine_s8(vget_low_s8(a6), vget_low_s8(a7));
            let a67_hi = vcombine_s8(vget_high_s8(a6), vget_high_s8(a7));

            acc02 = smmla_s32(smmla_s32(acc02, w01_lo, a45_lo), w01_hi, a45_hi);
            acc03 = smmla_s32(smmla_s32(acc03, w01_lo, a67_lo), w01_hi, a67_hi);
            acc12 = smmla_s32(smmla_s32(acc12, w23_lo, a45_lo), w23_hi, a45_hi);
            acc13 = smmla_s32(smmla_s32(acc13, w23_lo, a67_lo), w23_hi, a67_hi);
            acc22 = smmla_s32(smmla_s32(acc22, w45_lo, a45_lo), w45_hi, a45_hi);
            acc23 = smmla_s32(smmla_s32(acc23, w45_lo, a67_lo), w45_hi, a67_hi);
            acc32 = smmla_s32(smmla_s32(acc32, w67_lo, a45_lo), w67_hi, a45_hi);
            acc33 = smmla_s32(smmla_s32(acc33, w67_lo, a67_lo), w67_hi, a67_hi);
        }
        for (pair_idx, [acc_0, acc_1, acc_2, acc_3]) in [
            [acc00, acc01, acc02, acc03],
            [acc10, acc11, acc12, acc13],
            [acc20, acc21, acc22, acc23],
            [acc30, acc31, acc32, acc33],
        ]
        .into_iter()
        .enumerate()
        {
            let mut q0 = [0_i32; 4];
            let mut q1 = [0_i32; 4];
            let mut q2 = [0_i32; 4];
            let mut q3 = [0_i32; 4];
            store_s32x4(acc_0, &mut q0);
            store_s32x4(acc_1, &mut q1);
            store_s32x4(acc_2, &mut q2);
            store_s32x4(acc_3, &mut q3);
            let r0 = pair_idx * 2;
            let r1 = r0 + 1;
            out[r0][0][group] = q0[0];
            out[r0][1][group] = q0[1];
            out[r0][2][group] = q1[0];
            out[r0][3][group] = q1[1];
            out[r0][4][group] = q2[0];
            out[r0][5][group] = q2[1];
            out[r0][6][group] = q3[0];
            out[r0][7][group] = q3[1];
            out[r1][0][group] = q0[2];
            out[r1][1][group] = q0[3];
            out[r1][2][group] = q1[2];
            out[r1][3][group] = q1[3];
            out[r1][4][group] = q2[2];
            out[r1][5][group] = q2[3];
            out[r1][6][group] = q3[2];
            out[r1][7][group] = q3[3];
        }
    }
    out
}

#[target_feature(enable = "neon,dotprod")]
fn dot_ptq1_8x8_groups_dotprod(
    trits: &[[[i8; 16]; 8]; 8],
    activations: [[&[i8; 32]; 4]; 8],
) -> super::Ptq1EightRowEightTokenGroups {
    let mut out = [[[0_i32; 4]; 8]; 8];
    for group in 0..4 {
        let acts: [(int8x16_t, int8x16_t); 8] = std::array::from_fn(|t| {
            let halves = activations[t][group].as_chunks::<16>().0;
            (load_s8x16(&halves[0]), load_s8x16(&halves[1]))
        });
        for r in 0..8 {
            let w0 = load_s8x16(&trits[r][group * 2]);
            let w1 = load_s8x16(&trits[r][group * 2 + 1]);
            let l0 = sdot32_vec(w0, w1, acts[0].0, acts[0].1);
            let l1 = sdot32_vec(w0, w1, acts[1].0, acts[1].1);
            let l2 = sdot32_vec(w0, w1, acts[2].0, acts[2].1);
            let l3 = sdot32_vec(w0, w1, acts[3].0, acts[3].1);
            let l4 = sdot32_vec(w0, w1, acts[4].0, acts[4].1);
            let l5 = sdot32_vec(w0, w1, acts[5].0, acts[5].1);
            let l6 = sdot32_vec(w0, w1, acts[6].0, acts[6].1);
            let l7 = sdot32_vec(w0, w1, acts[7].0, acts[7].1);
            let mut sums_lo = [0_i32; 4];
            let mut sums_hi = [0_i32; 4];
            store_s32x4(reduce4_s32x4_addp(l0, l1, l2, l3), &mut sums_lo);
            store_s32x4(reduce4_s32x4_addp(l4, l5, l6, l7), &mut sums_hi);
            for t in 0..4 {
                out[r][t][group] = sums_lo[t];
                out[r][t + 4][group] = sums_hi[t];
            }
        }
    }
    out
}

#[inline]
pub(super) fn dot_ptq1_8x8_lanes(
    trits: &[[[i8; 16]; 8]; 8],
    activations: [[&[i8; 32]; 4]; 8],
) -> Option<super::Ptq1EightRowEightTokenLanes> {
    if !has_dotprod() {
        return None;
    }
    // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
    Some(unsafe { dot_ptq1_8x8_lanes_dotprod(trits, activations) })
}

#[inline]
pub(super) fn dot_ptq1_8x8_groups(
    trits: &[[[i8; 16]; 8]; 8],
    activations: [[&[i8; 32]; 4]; 8],
) -> Option<super::Ptq1EightRowEightTokenGroups> {
    if has_i8mm() {
        // SAFETY: `has_i8mm()` verified that the CPU supports `i8mm` and baseline `neon`.
        return Some(unsafe { dot_ptq1_8x8_groups_i8mm(trits, activations) });
    }
    if has_dotprod() {
        // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
        return Some(unsafe { dot_ptq1_8x8_groups_dotprod(trits, activations) });
    }
    None
}

#[inline]
pub(super) fn dot_ptq1_4x1_lanes(
    trits: &[[[i8; 16]; 8]; 4],
    activations: [&[i8; 32]; 4],
) -> Option<[[[i32; 4]; 4]; 4]> {
    if !has_dotprod() {
        return None;
    }
    // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
    Some(unsafe { dot_ptq1_4x1_lanes_dotprod(trits, activations) })
}

#[inline]
pub(super) fn dot_ptq1_4x1_groups(
    trits: &[[[i8; 16]; 8]; 4],
    activations: [&[i8; 32]; 4],
) -> Option<[[i32; 4]; 4]> {
    if !has_dotprod() {
        return None;
    }
    // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
    Some(unsafe { dot_ptq1_4x1_groups_dotprod(trits, activations) })
}

#[inline]
pub(super) fn dot_ptq1_8x1_lanes(
    trits: &[[[i8; 16]; 8]; 8],
    activations: [&[i8; 32]; 4],
) -> Option<[[[i32; 4]; 4]; 8]> {
    if !has_dotprod() {
        return None;
    }
    // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
    Some(unsafe { dot_ptq1_8x1_lanes_dotprod(trits, activations) })
}

#[inline]
pub(super) fn dot_ptq1_8x1_groups(
    trits: &[[[i8; 16]; 8]; 8],
    activations: [&[i8; 32]; 4],
) -> Option<[[i32; 4]; 8]> {
    if !has_dotprod() {
        return None;
    }
    // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
    Some(unsafe { dot_ptq1_8x1_groups_dotprod(trits, activations) })
}

#[inline]
pub(super) fn dot_ptq1_8x4_lanes(
    trits: &[[[i8; 16]; 8]; 8],
    activations: [[&[i8; 32]; 4]; 4],
) -> Option<super::Ptq1EightRowFourTokenLanes> {
    if !has_dotprod() {
        return None;
    }
    // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
    Some(unsafe { dot_ptq1_8x4_lanes_dotprod(trits, activations) })
}

#[inline]
pub(super) fn dot_ptq1_8x4_groups(
    trits: &[[[i8; 16]; 8]; 8],
    activations: [[&[i8; 32]; 4]; 4],
) -> Option<super::Ptq1EightRowFourTokenGroups> {
    if has_i8mm() {
        // SAFETY: `has_i8mm()` verified that the CPU supports `i8mm` and baseline `neon`.
        return Some(unsafe { dot_ptq1_8x4_groups_i8mm(trits, activations) });
    }
    if has_dotprod() {
        // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
        return Some(unsafe { dot_ptq1_8x4_groups_dotprod(trits, activations) });
    }
    None
}

#[inline]
pub(super) fn dot_ptq1_4x4_lanes(
    trits: &[[[i8; 16]; 8]; 4],
    activations: [[&[i8; 32]; 4]; 4],
) -> Option<super::Ptq1FourRowFourTokenLanes> {
    if !has_dotprod() {
        return None;
    }
    // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
    Some(unsafe { dot_ptq1_4x4_lanes_dotprod(trits, activations) })
}

#[inline]
pub(super) fn dot_ptq1_4x4_groups(
    trits: &[[[i8; 16]; 8]; 4],
    activations: [[&[i8; 32]; 4]; 4],
) -> Option<super::Ptq1FourRowFourTokenGroups> {
    if has_i8mm() {
        // SAFETY: `has_i8mm()` verified that the CPU supports `i8mm` and baseline `neon`.
        return Some(unsafe { dot_ptq1_4x4_groups_i8mm(trits, activations) });
    }
    if has_dotprod() {
        // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
        return Some(unsafe { dot_ptq1_4x4_groups_dotprod(trits, activations) });
    }
    None
}

#[inline]
const fn ptq1_scale_u16(block: &[u8; 28]) -> u16 {
    u16::from_le_bytes([block[26], block[27]])
}

#[inline]
#[target_feature(enable = "neon")]
fn ptq1_validate_row_scales<const ROWS: usize>(blocks: &[&[u8; 28]; ROWS]) -> Option<[f32; ROWS]> {
    if ROWS == 1 {
        let s = super::widen_f16_four(&[ptq1_scale_u16(blocks[0]), 0, 0, 0])[0];
        if !s.is_finite() {
            return None;
        }
        return Some([s; ROWS]);
    }
    if ROWS == 4 {
        let bits = [
            ptq1_scale_u16(blocks[0]),
            ptq1_scale_u16(blocks[1]),
            ptq1_scale_u16(blocks[2]),
            ptq1_scale_u16(blocks[3]),
        ];
        let s = super::widen_f16_four(&bits);
        if !s.iter().all(|v| v.is_finite()) {
            return None;
        }
        return Some(std::array::from_fn(|r| s[r]));
    }
    if ROWS == 8 {
        let bits: [u16; 8] = std::array::from_fn(|r| ptq1_scale_u16(blocks[r]));
        let [lo, hi] = super::widen_f16_eight(&bits);
        if !lo.iter().chain(hi.iter()).all(|v| v.is_finite()) {
            return None;
        }
        return Some(std::array::from_fn(
            |r| if r < 4 { lo[r] } else { hi[r - 4] },
        ));
    }
    let bits_lo: [u16; 8] = std::array::from_fn(|r| ptq1_scale_u16(blocks[r]));
    let bits_hi: [u16; 8] = std::array::from_fn(|r| ptq1_scale_u16(blocks[8 + r]));
    let [s0, s1] = super::widen_f16_eight(&bits_lo);
    let [s2, s3] = super::widen_f16_eight(&bits_hi);
    if !s0
        .iter()
        .chain(s1.iter())
        .chain(s2.iter())
        .chain(s3.iter())
        .all(|v| v.is_finite())
    {
        return None;
    }
    let quads = [s0, s1, s2, s3];
    Some(std::array::from_fn(|r| quads[r / 4][r % 4]))
}

#[inline]
#[target_feature(enable = "neon")]
fn unpack_ptq1_group(packed: &[u8; 28], group: usize) -> (int8x16_t, int8x16_t) {
    use core::arch::aarch64::vdup_n_u16;
    const POW3_MID_01: [u8; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3];
    const POW3_MID_23: [u8; 16] = [9, 9, 9, 9, 9, 9, 9, 9, 27, 27, 27, 27, 27, 27, 27, 27];
    const POW3_TAIL: [u8; 16] = [81, 81, 81, 81, 81, 81, 81, 81, 1, 1, 3, 3, 9, 9, 27, 27];

    match group {
        0 => {
            let q0 = load_u8x16(&packed[..16].as_chunks::<16>().0[0]);
            (
                extract_trit_s8x16(q0),
                extract_trit_s8x16(vmulq_u8(q0, vdupq_n_u8(3))),
            )
        }
        1 => {
            let q0 = load_u8x16(&packed[..16].as_chunks::<16>().0[0]);
            (
                extract_trit_s8x16(vmulq_u8(q0, vdupq_n_u8(9))),
                extract_trit_s8x16(vmulq_u8(q0, vdupq_n_u8(27))),
            )
        }
        2 => {
            let q0 = load_u8x16(&packed[..16].as_chunks::<16>().0[0]);
            let q_mid_half = load_u8x8(&packed[16..24].as_chunks::<8>().0[0]);
            let q_mid = vcombine_u8(q_mid_half, q_mid_half);
            (
                extract_trit_s8x16(vmulq_u8(q0, vdupq_n_u8(81))),
                extract_trit_s8x16(vmulq_u8(q_mid, load_u8x16(&POW3_MID_01))),
            )
        }
        _ => {
            let q_mid_half = load_u8x8(&packed[16..24].as_chunks::<8>().0[0]);
            let q_mid = vcombine_u8(q_mid_half, q_mid_half);
            let tail_half =
                vreinterpret_u8_u16(vdup_n_u16(u16::from_le_bytes([packed[24], packed[25]])));
            let q_tail = vcombine_u8(q_mid_half, tail_half);
            (
                extract_trit_s8x16(vmulq_u8(q_mid, load_u8x16(&POW3_MID_23))),
                extract_trit_s8x16(vmulq_u8(q_tail, load_u8x16(&POW3_TAIL))),
            )
        }
    }
}

const PTQ1_POW3_MID_01: [u8; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3];
const PTQ1_POW3_MID_23: [u8; 16] = [9, 9, 9, 9, 9, 9, 9, 9, 27, 27, 27, 27, 27, 27, 27, 27];
const PTQ1_POW3_TAIL: [u8; 16] = [81, 81, 81, 81, 81, 81, 81, 81, 1, 1, 3, 3, 9, 9, 27, 27];

macro_rules! ptq1_row_pinned_asm {
    ($p:literal, $lane:literal, $rs_reg:literal, $rs_lane:literal, $step4:literal) => {
        concat!(
            "ldr q26, [{",
            $p,
            "}], #16\n",
            "fmul v25.4s, v16.4s, ",
            $rs_reg,
            ".s[",
            $rs_lane,
            "]\n",
            "cmhi v28.16b, v19.16b, v26.16b\n",
            "cmhi v29.16b, v26.16b, v20.16b\n",
            "mul v30.16b, v26.16b, v21.16b\n",
            "movi v27.2d, #0\n",
            "sub v28.16b, v28.16b, v29.16b\n",
            "cmhi v29.16b, v19.16b, v30.16b\n",
            "cmhi v31.16b, v30.16b, v20.16b\n",
            "sdot v27.4s, v28.16b, v8.16b\n",
            "mul v26.16b, v30.16b, v21.16b\n",
            "sub v29.16b, v29.16b, v31.16b\n",
            "sdot v27.4s, v29.16b, v9.16b\n",
            "scvtf v27.4s, v27.4s\n",
            "fmla ",
            $lane,
            ".4s, v27.4s, v25.s[0]\n",
            "cmhi v28.16b, v19.16b, v26.16b\n",
            "cmhi v29.16b, v26.16b, v20.16b\n",
            "mul v30.16b, v26.16b, v21.16b\n",
            "movi v27.2d, #0\n",
            "sub v28.16b, v28.16b, v29.16b\n",
            "cmhi v29.16b, v19.16b, v30.16b\n",
            "cmhi v31.16b, v30.16b, v20.16b\n",
            "sdot v27.4s, v28.16b, v10.16b\n",
            "mul v26.16b, v30.16b, v21.16b\n",
            "sub v29.16b, v29.16b, v31.16b\n",
            "sdot v27.4s, v29.16b, v11.16b\n",
            "scvtf v27.4s, v27.4s\n",
            "fmla ",
            $lane,
            ".4s, v27.4s, v25.s[1]\n",
            "ld1r {{v30.2d}}, [{",
            $p,
            "}], #8\n",
            "ld1r {{v31.4h}}, [{",
            $p,
            "}], {",
            $step4,
            "}\n",
            "cmhi v28.16b, v19.16b, v26.16b\n",
            "cmhi v29.16b, v26.16b, v20.16b\n",
            "mul v26.16b, v30.16b, v22.16b\n",
            "movi v27.2d, #0\n",
            "sub v28.16b, v28.16b, v29.16b\n",
            "cmhi v29.16b, v19.16b, v26.16b\n",
            "sdot v27.4s, v28.16b, v12.16b\n",
            "cmhi v28.16b, v26.16b, v20.16b\n",
            "mul v26.16b, v30.16b, v23.16b\n",
            "mov v30.d[1], v31.d[0]\n",
            "sub v29.16b, v29.16b, v28.16b\n",
            "sdot v27.4s, v29.16b, v13.16b\n",
            "scvtf v27.4s, v27.4s\n",
            "fmla ",
            $lane,
            ".4s, v27.4s, v25.s[2]\n",
            "cmhi v28.16b, v19.16b, v26.16b\n",
            "cmhi v29.16b, v26.16b, v20.16b\n",
            "mul v30.16b, v30.16b, v24.16b\n",
            "movi v27.2d, #0\n",
            "sub v28.16b, v28.16b, v29.16b\n",
            "cmhi v29.16b, v19.16b, v30.16b\n",
            "cmhi v31.16b, v30.16b, v20.16b\n",
            "sdot v27.4s, v28.16b, v14.16b\n",
            "sub v29.16b, v29.16b, v31.16b\n",
            "sdot v27.4s, v29.16b, v15.16b\n",
            "scvtf v27.4s, v27.4s\n",
            "fmla ",
            $lane,
            ".4s, v27.4s, v25.s[3]\n",
        )
    };
}

macro_rules! ptq1_row_source_asm {
    ($p:literal, $dst_vec:literal, $dst_lane:literal, $step4:literal) => {
        concat!(
            "ldr q26, [{", $p, "}], #16\n",
            "cmhi v28.16b, v19.16b, v26.16b\n",
            "cmhi v29.16b, v26.16b, v20.16b\n",
            "mul v30.16b, v26.16b, v21.16b\n",
            "movi v2.2d, #0\n",
            "sub v28.16b, v28.16b, v29.16b\n",
            "cmhi v29.16b, v19.16b, v30.16b\n",
            "cmhi v31.16b, v30.16b, v20.16b\n",
            "sdot v2.4s, v28.16b, v8.16b\n",
            "mul v26.16b, v30.16b, v21.16b\n",
            "sub v29.16b, v29.16b, v31.16b\n",
            "sdot v2.4s, v29.16b, v9.16b\n",
            "cmhi v28.16b, v19.16b, v26.16b\n",
            "cmhi v29.16b, v26.16b, v20.16b\n",
            "mul v30.16b, v26.16b, v21.16b\n",
            "movi v3.2d, #0\n",
            "sub v28.16b, v28.16b, v29.16b\n",
            "cmhi v29.16b, v19.16b, v30.16b\n",
            "cmhi v31.16b, v30.16b, v20.16b\n",
            "sdot v3.4s, v28.16b, v10.16b\n",
            "mul v26.16b, v30.16b, v21.16b\n",
            "sub v29.16b, v29.16b, v31.16b\n",
            "sdot v3.4s, v29.16b, v11.16b\n",
            "ld1r {{v30.2d}}, [{", $p, "}], #8\n",
            "ld1r {{v31.4h}}, [{", $p, "}], {", $step4, "}\n",
            "cmhi v28.16b, v19.16b, v26.16b\n",
            "cmhi v29.16b, v26.16b, v20.16b\n",
            "mul v26.16b, v30.16b, v22.16b\n",
            "movi v4.2d, #0\n",
            "sub v28.16b, v28.16b, v29.16b\n",
            "cmhi v29.16b, v19.16b, v26.16b\n",
            "sdot v4.4s, v28.16b, v12.16b\n",
            "cmhi v28.16b, v26.16b, v20.16b\n",
            "mul v26.16b, v30.16b, v23.16b\n",
            "mov v30.d[1], v31.d[0]\n",
            "sub v29.16b, v29.16b, v28.16b\n",
            "sdot v4.4s, v29.16b, v13.16b\n",
            "cmhi v28.16b, v19.16b, v26.16b\n",
            "cmhi v29.16b, v26.16b, v20.16b\n",
            "mul v30.16b, v30.16b, v24.16b\n",
            "movi v5.2d, #0\n",
            "sub v28.16b, v28.16b, v29.16b\n",
            "cmhi v29.16b, v19.16b, v30.16b\n",
            "cmhi v31.16b, v30.16b, v20.16b\n",
            "sdot v5.4s, v28.16b, v14.16b\n",
            "sub v29.16b, v29.16b, v31.16b\n",
            "sdot v5.4s, v29.16b, v15.16b\n",
            "uzp1 v27.4s, v2.4s, v3.4s\n",
            "uzp2 v28.4s, v2.4s, v3.4s\n",
            "uzp1 v29.4s, v4.4s, v5.4s\n",
            "uzp2 v30.4s, v4.4s, v5.4s\n",
            "add v27.4s, v27.4s, v28.4s\n",
            "add v29.4s, v29.4s, v30.4s\n",
            "uzp1 v2.4s, v27.4s, v29.4s\n",
            "uzp2 v28.4s, v27.4s, v29.4s\n",
            "add v2.4s, v2.4s, v28.4s\n",
            "scvtf v2.4s, v2.4s\n",
            "fmul v2.4s, v2.4s, v16.4s\n",
            "movi d28, #0\n",
            "fadd s28, s28, s2\n",
            "dup s29, v2.s[1]\n",
            "fadd s28, s28, s29\n",
            "dup s29, v2.s[2]\n",
            "fadd s28, s28, s29\n",
            "dup s29, v2.s[3]\n",
            "fadd s28, s28, s29\n",
            "mov ", $dst_vec, ".s[", $dst_lane, "], v28.s[0]\n",
        )
    };
}

#[inline]
pub(super) fn ptq1_neg_lane_sums(values: &[i8; 128]) -> [[i32; 4]; 4] {
    // SAFETY: Baseline NEON is always available on `aarch64`.
    unsafe { ptq1_neg_lane_sums_neon(values) }
}

#[inline]
#[target_feature(enable = "neon")]
fn ptq1_neg_lane_sums_neon(values: &[i8; 128]) -> [[i32; 4]; 4] {
    let halves = values.as_chunks::<16>().0;
    let mut out = [[0_i32; 4]; 4];
    for (g, dst) in out.iter_mut().enumerate() {
        let s_lo = vpaddlq_s16(vpaddlq_s8(load_s8x16(&halves[2 * g])));
        let s_hi = vpaddlq_s16(vpaddlq_s8(load_s8x16(&halves[2 * g + 1])));
        store_s32x4(vnegq_s32(vaddq_s32(s_lo, s_hi)), dst);
    }
    out
}

macro_rules! ptq1_row_pinned_v2_asm {
    ($p:literal, $lane:literal, $lane_idx:literal) => {
        concat!(
            "ldr q26, [{",
            $p,
            "}]\n",
            "fmul v25.4s, v16.4s, v17.s[",
            $lane_idx,
            "]\n",
            "cmhi v28.16b, v19.16b, v26.16b\n",
            "cmhi v29.16b, v26.16b, v20.16b\n",
            "mul v30.16b, v26.16b, v21.16b\n",
            "movi v27.2d, #0\n",
            "sub v28.16b, v28.16b, v29.16b\n",
            "cmhi v29.16b, v19.16b, v30.16b\n",
            "sdot v27.4s, v28.16b, v8.16b\n",
            "cmhi v28.16b, v30.16b, v20.16b\n",
            "mul v26.16b, v30.16b, v21.16b\n",
            "sub v29.16b, v29.16b, v28.16b\n",
            "sdot v27.4s, v29.16b, v9.16b\n",
            "scvtf v27.4s, v27.4s\n",
            "fmla ",
            $lane,
            ".4s, v27.4s, v25.s[0]\n",
            "cmhi v28.16b, v19.16b, v26.16b\n",
            "cmhi v29.16b, v26.16b, v20.16b\n",
            "mul v30.16b, v26.16b, v21.16b\n",
            "movi v27.2d, #0\n",
            "sub v28.16b, v28.16b, v29.16b\n",
            "cmhi v29.16b, v19.16b, v30.16b\n",
            "sdot v27.4s, v28.16b, v10.16b\n",
            "cmhi v28.16b, v30.16b, v20.16b\n",
            "mul v26.16b, v30.16b, v21.16b\n",
            "sub v29.16b, v29.16b, v28.16b\n",
            "sdot v27.4s, v29.16b, v11.16b\n",
            "scvtf v27.4s, v27.4s\n",
            "fmla ",
            $lane,
            ".4s, v27.4s, v25.s[1]\n",
            "ldr d30, [{",
            $p,
            "}, #16]\n",
            "add {",
            $p,
            "}, {",
            $p,
            "}, #28\n",
            "dup v30.2d, v30.d[0]\n",
            "cmhi v28.16b, v19.16b, v26.16b\n",
            "cmhi v29.16b, v26.16b, v20.16b\n",
            "mul v26.16b, v30.16b, v22.16b\n",
            "movi v27.2d, #0\n",
            "sub v28.16b, v28.16b, v29.16b\n",
            "cmhi v29.16b, v19.16b, v26.16b\n",
            "sdot v27.4s, v28.16b, v12.16b\n",
            "cmhi v28.16b, v26.16b, v20.16b\n",
            "mul v26.16b, v30.16b, v23.16b\n",
            "sub v29.16b, v29.16b, v28.16b\n",
            "sdot v27.4s, v29.16b, v13.16b\n",
            "scvtf v27.4s, v27.4s\n",
            "fmla ",
            $lane,
            ".4s, v27.4s, v25.s[2]\n",
            "cmhi v28.16b, v19.16b, v26.16b\n",
            "cmhi v29.16b, v26.16b, v20.16b\n",
            "movi v27.2d, #0\n",
            "sub v28.16b, v28.16b, v29.16b\n",
            "sdot v27.4s, v28.16b, v14.16b\n",
            "dup v28.4h, v18.h[",
            $lane_idx,
            "]\n",
            "mov v30.d[1], v28.d[0]\n",
            "mul v30.16b, v30.16b, v24.16b\n",
            "cmhi v28.16b, v19.16b, v30.16b\n",
            "cmhi v29.16b, v30.16b, v20.16b\n",
            "sub v28.16b, v28.16b, v29.16b\n",
            "sdot v27.4s, v28.16b, v15.16b\n",
            "scvtf v27.4s, v27.4s\n",
            "fmla ",
            $lane,
            ".4s, v27.4s, v25.s[3]\n",
        )
    };
}

macro_rules! ptq1_8rows_unpack_scales_tails_asm {
    ($p0:literal, $p1:literal, $p2:literal, $p3:literal) => {
        concat!(
            "ldr s27, [{",
            $p0,
            "}, #24]\n",
            "ldr s28, [{",
            $p1,
            "}, #24]\n",
            "ldr s29, [{",
            $p2,
            "}, #24]\n",
            "ldr s30, [{",
            $p3,
            "}, #24]\n",
            "trn1 v27.2s, v27.2s, v28.2s\n",
            "trn1 v29.2s, v29.2s, v30.2s\n",
            "uzp1 v18.4h, v27.4h, v29.4h\n",
            "uzp2 v17.4h, v27.4h, v29.4h\n",
            "fmov d27, d17\n",
            "bic v27.4h, #0x80, lsl #8\n",
            "umax v31.4h, v31.4h, v27.4h\n",
            "fcvtl v17.4s, v17.4h\n",
        )
    };
}

macro_rules! ptq1_8rows_pinned_block_asm {
    () => {
        concat!(
            "ldr q16, [{act}]\n",
            "ldp q8, q9, [{act}, #16]\n",
            "ldp q10, q11, [{act}, #48]\n",
            "ldp q12, q13, [{act}, #80]\n",
            "ldp q14, q15, [{act}, #112]\n",
            "add {act}, {act}, #208\n",
            ptq1_8rows_unpack_scales_tails_asm!("p0", "p1", "p2", "p3"),
            ptq1_row_pinned_v2_asm!("p0", "v0", "0"),
            ptq1_row_pinned_v2_asm!("p1", "v1", "1"),
            ptq1_row_pinned_v2_asm!("p2", "v2", "2"),
            ptq1_row_pinned_v2_asm!("p3", "v3", "3"),
            ptq1_8rows_unpack_scales_tails_asm!("p4", "p5", "p6", "p7"),
            ptq1_row_pinned_v2_asm!("p4", "v4", "0"),
            ptq1_row_pinned_v2_asm!("p5", "v5", "1"),
            ptq1_row_pinned_v2_asm!("p6", "v6", "2"),
            ptq1_row_pinned_v2_asm!("p7", "v7", "3"),
        )
    };
}

/// Caller must guarantee `block_count > 0`, every `row_ptrs[r]` covers
/// `block_count * 28` readable bytes, and `act_ptr` covers `block_count`
/// `Ptq1Activation` elements.
#[expect(
    clippy::too_many_lines,
    reason = "Register-resident 8-row PTQ1 pinned asm kernel with 2x unrolling avoids stack spills."
)]
#[target_feature(enable = "neon,dotprod")]
unsafe fn ptq1_packed_8rows_pinned_accum(
    row_ptrs: [*const u8; 8],
    act_ptr: *const super::Ptq1Activation,
    block_count: usize,
    lanes: &mut [[f32; 4]; 8],
) -> bool {
    if block_count == 0 {
        return true;
    }
    let pow_mid01 = load_u8x16(&PTQ1_POW3_MID_01);
    let pow_mid23 = load_u8x16(&PTQ1_POW3_MID_23);
    let pow_tail = load_u8x16(&PTQ1_POW3_TAIL);
    let mut l0 = load_f32x4_neon(&lanes[0]);
    let mut l1 = load_f32x4_neon(&lanes[1]);
    let mut l2 = load_f32x4_neon(&lanes[2]);
    let mut l3 = load_f32x4_neon(&lanes[3]);
    let mut l4 = load_f32x4_neon(&lanes[4]);
    let mut l5 = load_f32x4_neon(&lanes[5]);
    let mut l6 = load_f32x4_neon(&lanes[6]);
    let mut l7 = load_f32x4_neon(&lanes[7]);
    let p0 = row_ptrs[0];
    let p1 = row_ptrs[1];
    let p2 = row_ptrs[2];
    let p3 = row_ptrs[3];
    let p4 = row_ptrs[4];
    let p5 = row_ptrs[5];
    let p6 = row_ptrs[6];
    let p7 = row_ptrs[7];
    let act = act_ptr;
    let cnt = block_count;
    let ok: usize;

    // SAFETY: Caller guarantees `row_ptrs[0..8]` and `act_ptr` remain valid for
    // `block_count` blocks; all clobbered SIMD registers are listed in operands.
    unsafe {
        core::arch::asm!(
            "movi v31.2d, #0",
            "movi v19.16b, #86",
            "movi v20.16b, #170",
            "movi v21.16b, #3",
            "tbz {cnt}, #0, 2f",
            ptq1_8rows_pinned_block_asm!(),
            "sub {cnt}, {cnt}, #1",
            "cbz {cnt}, 4f",
            "2:",
            "prfm pldl1keep, [{act}, #416]",
            "prfm pldl1keep, [{p0}, #112]",
            "prfm pldl1keep, [{p1}, #112]",
            "prfm pldl1keep, [{p2}, #112]",
            "prfm pldl1keep, [{p3}, #112]",
            "prfm pldl1keep, [{p4}, #112]",
            "prfm pldl1keep, [{p5}, #112]",
            "prfm pldl1keep, [{p6}, #112]",
            "prfm pldl1keep, [{p7}, #112]",
            ptq1_8rows_pinned_block_asm!(),
            "prfm pldl1keep, [{act}, #416]",
            ptq1_8rows_pinned_block_asm!(),
            "subs {cnt}, {cnt}, #2",
            "b.ne 2b",
            "4:",
            "umaxv h28, v31.4h",
            "fmov {t0:w}, s28",
            "lsr {t0:w}, {t0:w}, #10",
            "cmp {t0:w}, #30",
            "cset {ok}, ls",
            p0 = inout(reg) p0 => _,
            p1 = inout(reg) p1 => _,
            p2 = inout(reg) p2 => _,
            p3 = inout(reg) p3 => _,
            p4 = inout(reg) p4 => _,
            p5 = inout(reg) p5 => _,
            p6 = inout(reg) p6 => _,
            p7 = inout(reg) p7 => _,
            act = inout(reg) act => _,
            cnt = inout(reg) cnt => _,
            t0 = out(reg) _,
            ok = out(reg) ok,
            inout("v0") l0,
            inout("v1") l1,
            inout("v2") l2,
            inout("v3") l3,
            inout("v4") l4,
            inout("v5") l5,
            inout("v6") l6,
            inout("v7") l7,
            out("v8") _,
            out("v9") _,
            out("v10") _,
            out("v11") _,
            out("v12") _,
            out("v13") _,
            out("v14") _,
            out("v15") _,
            out("v16") _,
            out("v17") _,
            out("v18") _,
            out("v19") _,
            out("v20") _,
            out("v21") _,
            in("v22") pow_mid01,
            in("v23") pow_mid23,
            in("v24") pow_tail,
            out("v25") _,
            out("v26") _,
            out("v27") _,
            out("v28") _,
            out("v29") _,
            out("v30") _,
            out("v31") _,
            options(nostack, readonly),
        );
    }
    if ok == 0 {
        return false;
    }
    store_f32x4_neon(l0, &mut lanes[0]);
    store_f32x4_neon(l1, &mut lanes[1]);
    store_f32x4_neon(l2, &mut lanes[2]);
    store_f32x4_neon(l3, &mut lanes[3]);
    store_f32x4_neon(l4, &mut lanes[4]);
    store_f32x4_neon(l5, &mut lanes[5]);
    store_f32x4_neon(l6, &mut lanes[6]);
    store_f32x4_neon(l7, &mut lanes[7]);
    true
}

macro_rules! ptq1_row_pinned_fast_asm {
    ($p:literal, $lane:literal, $rs_lane:literal, $qh_lane:literal) => {
        concat!(
            "ldr q26, [{",
            $p,
            "}]\n",
            "ldr d30, [{",
            $p,
            "}, #16]\n",
            "fmul v25.4s, v16.4s, v17.s[",
            $rs_lane,
            "]\n",
            "add {",
            $p,
            "}, {",
            $p,
            "}, #28\n",
            "cmhi v28.16b, v19.16b, v26.16b\n",
            "cmhi v29.16b, v26.16b, v20.16b\n",
            "mul v27.16b, v26.16b, v21.16b\n",
            "sub v28.16b, v28.16b, v29.16b\n",
            "mul v29.16b, v26.16b, v4.16b\n",
            "movi v31.2d, #0\n",
            "sdot v31.4s, v28.16b, v8.16b\n",
            "cmhi v28.16b, v19.16b, v27.16b\n",
            "cmhi v27.16b, v27.16b, v20.16b\n",
            "sub v28.16b, v28.16b, v27.16b\n",
            "mul v27.16b, v26.16b, v5.16b\n",
            "mul v26.16b, v26.16b, v6.16b\n",
            "sdot v31.4s, v28.16b, v9.16b\n",
            "scvtf v31.4s, v31.4s\n",
            "fmla ",
            $lane,
            ".4s, v31.4s, v25.s[0]\n",
            "cmhi v28.16b, v19.16b, v29.16b\n",
            "cmhi v29.16b, v29.16b, v20.16b\n",
            "movi v31.2d, #0\n",
            "sub v28.16b, v28.16b, v29.16b\n",
            "sdot v31.4s, v28.16b, v10.16b\n",
            "cmhi v28.16b, v19.16b, v27.16b\n",
            "cmhi v29.16b, v27.16b, v20.16b\n",
            "sub v28.16b, v28.16b, v29.16b\n",
            "dup v30.2d, v30.d[0]\n",
            "sdot v31.4s, v28.16b, v11.16b\n",
            "scvtf v31.4s, v31.4s\n",
            "fmla ",
            $lane,
            ".4s, v31.4s, v25.s[1]\n",
            "mul v27.16b, v30.16b, v22.16b\n",
            "cmhi v28.16b, v19.16b, v26.16b\n",
            "cmhi v29.16b, v26.16b, v20.16b\n",
            "movi v31.2d, #0\n",
            "sub v28.16b, v28.16b, v29.16b\n",
            "mul v26.16b, v30.16b, v23.16b\n",
            "sdot v31.4s, v28.16b, v12.16b\n",
            "cmhi v28.16b, v19.16b, v27.16b\n",
            "cmhi v29.16b, v27.16b, v20.16b\n",
            "dup v27.4h, v18.h[",
            $qh_lane,
            "]\n",
            "sub v28.16b, v28.16b, v29.16b\n",
            "mov v30.d[1], v27.d[0]\n",
            "sdot v31.4s, v28.16b, v13.16b\n",
            "scvtf v31.4s, v31.4s\n",
            "fmla ",
            $lane,
            ".4s, v31.4s, v25.s[2]\n",
            "mul v30.16b, v30.16b, v24.16b\n",
            "cmhi v28.16b, v19.16b, v26.16b\n",
            "cmhi v29.16b, v26.16b, v20.16b\n",
            "movi v31.2d, #0\n",
            "sub v28.16b, v28.16b, v29.16b\n",
            "sdot v31.4s, v28.16b, v14.16b\n",
            "cmhi v28.16b, v19.16b, v30.16b\n",
            "cmhi v29.16b, v30.16b, v20.16b\n",
            "sub v28.16b, v28.16b, v29.16b\n",
            "sdot v31.4s, v28.16b, v15.16b\n",
            "scvtf v31.4s, v31.4s\n",
            "fmla ",
            $lane,
            ".4s, v31.4s, v25.s[3]\n",
        )
    };
}

macro_rules! ptq1_4rows_pinned_block_asm {
    () => {
        concat!(
            "ldr s28, [{p0}, #24]\n",
            "ldr s29, [{p1}, #24]\n",
            "ldr s30, [{p2}, #24]\n",
            "ldr s31, [{p3}, #24]\n",
            "trn1 v28.2s, v28.2s, v29.2s\n",
            "trn1 v30.2s, v30.2s, v31.2s\n",
            "uzp1 v18.4h, v28.4h, v30.4h\n",
            "uzp2 v17.4h, v28.4h, v30.4h\n",
            "fmov d28, d17\n",
            "bic v28.4h, #0x80, lsl #8\n",
            "umax v7.4h, v7.4h, v28.4h\n",
            "fcvtl v17.4s, v17.4h\n",
            "ldr q16, [{act}]\n",
            "ldp q8, q9, [{act}, #16]\n",
            "ldp q10, q11, [{act}, #48]\n",
            "ldp q12, q13, [{act}, #80]\n",
            "ldp q14, q15, [{act}, #112]\n",
            "add {act}, {act}, #208\n",
            ptq1_row_pinned_fast_asm!("p0", "v0", "0", "0"),
            ptq1_row_pinned_fast_asm!("p1", "v1", "1", "1"),
            ptq1_row_pinned_fast_asm!("p2", "v2", "2", "2"),
            ptq1_row_pinned_fast_asm!("p3", "v3", "3", "3"),
        )
    };
}

/// Caller must guarantee `block_count > 0`, every `row_ptrs[r]` covers
/// `block_count * 28` readable bytes, and `act_ptr` covers `block_count`
/// `Ptq1Activation` elements.
#[target_feature(enable = "neon,dotprod")]
unsafe fn ptq1_packed_4rows_pinned_accum(
    row_ptrs: [*const u8; 4],
    act_ptr: *const super::Ptq1Activation,
    block_count: usize,
    lanes: &mut [[f32; 4]; 4],
) -> bool {
    if block_count == 0 {
        return true;
    }
    let pow_mid01 = load_u8x16(&PTQ1_POW3_MID_01);
    let pow_mid23 = load_u8x16(&PTQ1_POW3_MID_23);
    let pow_tail = load_u8x16(&PTQ1_POW3_TAIL);
    let mut l0 = load_f32x4_neon(&lanes[0]);
    let mut l1 = load_f32x4_neon(&lanes[1]);
    let mut l2 = load_f32x4_neon(&lanes[2]);
    let mut l3 = load_f32x4_neon(&lanes[3]);
    let p0 = row_ptrs[0];
    let p1 = row_ptrs[1];
    let p2 = row_ptrs[2];
    let p3 = row_ptrs[3];
    let act = act_ptr;
    let cnt = block_count;
    let ok: usize;

    // SAFETY: Caller guarantees `row_ptrs[0..4]` and `act_ptr` remain valid for
    // `block_count` blocks; all clobbered SIMD registers are listed in operands.
    unsafe {
        core::arch::asm!(
            "movi v4.16b, #9",
            "movi v5.16b, #27",
            "movi v6.16b, #81",
            "movi v7.2d, #0",
            "movi v19.16b, #86",
            "movi v20.16b, #170",
            "movi v21.16b, #3",
            "tbz {cnt}, #0, 2f",
            ptq1_4rows_pinned_block_asm!(),
            "sub {cnt}, {cnt}, #1",
            "cbz {cnt}, 4f",
            "2:",
            "prfm pldl1keep, [{act}, #416]",
            "prfm pldl1keep, [{p0}, #112]",
            "prfm pldl1keep, [{p1}, #112]",
            "prfm pldl1keep, [{p2}, #112]",
            "prfm pldl1keep, [{p3}, #112]",
            ptq1_4rows_pinned_block_asm!(),
            "prfm pldl1keep, [{act}, #416]",
            ptq1_4rows_pinned_block_asm!(),
            "subs {cnt}, {cnt}, #2",
            "b.ne 2b",
            "4:",
            "umaxv h28, v7.4h",
            "fmov {t0:w}, s28",
            "lsr {t0:w}, {t0:w}, #10",
            "cmp {t0:w}, #30",
            "cset {ok}, ls",
            p0 = inout(reg) p0 => _,
            p1 = inout(reg) p1 => _,
            p2 = inout(reg) p2 => _,
            p3 = inout(reg) p3 => _,
            act = inout(reg) act => _,
            cnt = inout(reg) cnt => _,
            t0 = out(reg) _,
            ok = out(reg) ok,
            inout("v0") l0,
            inout("v1") l1,
            inout("v2") l2,
            inout("v3") l3,
            out("v4") _,
            out("v5") _,
            out("v6") _,
            out("v7") _,
            out("v8") _,
            out("v9") _,
            out("v10") _,
            out("v11") _,
            out("v12") _,
            out("v13") _,
            out("v14") _,
            out("v15") _,
            out("v16") _,
            out("v17") _,
            out("v18") _,
            out("v19") _,
            out("v20") _,
            out("v21") _,
            in("v22") pow_mid01,
            in("v23") pow_mid23,
            in("v24") pow_tail,
            out("v25") _,
            out("v26") _,
            out("v27") _,
            out("v28") _,
            out("v29") _,
            out("v30") _,
            out("v31") _,
            options(nostack, readonly),
        );
    }
    if ok == 0 {
        return false;
    }
    store_f32x4_neon(l0, &mut lanes[0]);
    store_f32x4_neon(l1, &mut lanes[1]);
    store_f32x4_neon(l2, &mut lanes[2]);
    store_f32x4_neon(l3, &mut lanes[3]);
    true
}

/// Caller must guarantee `block_count > 0`, `row_ptr` covers `block_count * 28`
/// readable bytes, and `act_ptr` covers `block_count` `Ptq1Activation` elements.
#[target_feature(enable = "neon,dotprod")]
unsafe fn ptq1_packed_1row_pinned_accum(
    row_ptr: *const u8,
    act_ptr: *const super::Ptq1Activation,
    block_count: usize,
    lanes: &mut [f32; 4],
) -> bool {
    if block_count == 0 {
        return true;
    }
    let pow_mid01 = load_u8x16(&PTQ1_POW3_MID_01);
    let pow_mid23 = load_u8x16(&PTQ1_POW3_MID_23);
    let pow_tail = load_u8x16(&PTQ1_POW3_TAIL);
    let mut l0 = load_f32x4_neon(lanes);
    let p0 = row_ptr;
    let act = act_ptr;
    let cnt = block_count;
    let ok: usize;

    // SAFETY: Caller guarantees `row_ptr` and `act_ptr` remain valid for
    // `block_count` blocks; all clobbered SIMD registers are listed in operands.
    unsafe {
        core::arch::asm!(
            "movi v19.16b, #86",
            "movi v20.16b, #170",
            "movi v21.16b, #3",
            "1:",
            "add {t0}, {p0}, #26",
            "ldr h17, [{t0}]",
            "umov {t0:w}, v17.h[0]",
            "and {t0:w}, {t0:w}, #0x7fff",
            "lsr {t0:w}, {t0:w}, #10",
            "cmp {t0:w}, #30",
            "b.hi 2f",
            "fcvtl v17.4s, v17.4h",
            "ldr q16, [{act}]",
            "ldp q8, q9, [{act}, #16]",
            "ldp q10, q11, [{act}, #48]",
            "ldp q12, q13, [{act}, #80]",
            "ldp q14, q15, [{act}, #112]",
            "add {act}, {act}, #208",
            ptq1_row_pinned_asm!("p0", "v0", "v17", "0", "step4"),
            "subs {cnt}, {cnt}, #1",
            "b.ne 1b",
            "mov {ok}, #1",
            "b 3f",
            "2:",
            "mov {ok}, #0",
            "3:",
            p0 = inout(reg) p0 => _,
            act = inout(reg) act => _,
            cnt = inout(reg) cnt => _,
            step4 = in(reg) 4_usize,
            t0 = out(reg) _,
            ok = out(reg) ok,
            inout("v0") l0,
            out("v8") _,
            out("v9") _,
            out("v10") _,
            out("v11") _,
            out("v12") _,
            out("v13") _,
            out("v14") _,
            out("v15") _,
            out("v16") _,
            out("v17") _,
            out("v19") _,
            out("v20") _,
            out("v21") _,
            in("v22") pow_mid01,
            in("v23") pow_mid23,
            in("v24") pow_tail,
            out("v25") _,
            out("v26") _,
            out("v27") _,
            out("v28") _,
            out("v29") _,
            out("v30") _,
            out("v31") _,
            options(nostack, readonly),
        );
    }
    if ok == 0 {
        return false;
    }
    store_f32x4_neon(l0, lanes);
    true
}

/// Caller must guarantee `block_count > 0`, every `row_ptrs[r]` covers
/// `block_count * 28` readable bytes, and `act_ptr` covers `block_count`
/// `Ptq1Activation` elements.
#[expect(
    clippy::too_many_lines,
    reason = "Register-resident 8-row PTQ1 source asm kernel avoids stack spills."
)]
#[target_feature(enable = "neon,dotprod")]
unsafe fn ptq1_packed_8rows_source_accum(
    row_ptrs: [*const u8; 8],
    act_ptr: *const super::Ptq1Activation,
    block_count: usize,
    sums: &mut [f32; 8],
) -> bool {
    if block_count == 0 {
        return true;
    }
    let pow_mid01 = load_u8x16(&PTQ1_POW3_MID_01);
    let pow_mid23 = load_u8x16(&PTQ1_POW3_MID_23);
    let pow_tail = load_u8x16(&PTQ1_POW3_TAIL);
    let [sum_lo_arr, sum_hi_arr] = sums.as_chunks::<4>().0 else {
        return false;
    };
    let mut sum_lo = load_f32x4_neon(sum_lo_arr);
    let mut sum_hi = load_f32x4_neon(sum_hi_arr);
    let p0 = row_ptrs[0];
    let p1 = row_ptrs[1];
    let p2 = row_ptrs[2];
    let p3 = row_ptrs[3];
    let p4 = row_ptrs[4];
    let p5 = row_ptrs[5];
    let p6 = row_ptrs[6];
    let p7 = row_ptrs[7];
    let act = act_ptr;
    let cnt = block_count;
    let ok: usize;

    // SAFETY: Caller guarantees `row_ptrs[0..8]` and `act_ptr` remain valid for
    // `block_count` blocks; all clobbered SIMD registers are listed in operands.
    unsafe {
        core::arch::asm!(
            "movi v19.16b, #86",
            "movi v20.16b, #170",
            "movi v21.16b, #3",
            "1:",
            "ldrh {t0:w}, [{p0}, #26]",
            "ldrh {t1:w}, [{p1}, #26]",
            "fmov s17, {t0:w}",
            "ldrh {t0:w}, [{p4}, #26]",
            "ins v17.h[1], {t1:w}",
            "ldrh {t1:w}, [{p5}, #26]",
            "fmov s18, {t0:w}",
            "ldrh {t0:w}, [{p2}, #26]",
            "ins v18.h[1], {t1:w}",
            "ldrh {t1:w}, [{p3}, #26]",
            "ins v17.h[2], {t0:w}",
            "ldrh {t0:w}, [{p6}, #26]",
            "ins v17.h[3], {t1:w}",
            "ldrh {t1:w}, [{p7}, #26]",
            "ins v18.h[2], {t0:w}",
            "ins v18.h[3], {t1:w}",
            "mov v28.16b, v17.16b",
            "mov v28.d[1], v18.d[0]",
            "bic v28.8h, #0x80, lsl #8",
            "umaxv h28, v28.8h",
            "fmov {t0:w}, s28",
            "lsr {t0:w}, {t0:w}, #10",
            "cmp {t0:w}, #30",
            "b.hi 2f",
            "fcvtl v17.4s, v17.4h",
            "fcvtl v18.4s, v18.4h",
            "ldr q16, [{act}]",
            "ldp q8, q9, [{act}, #16]",
            "ldp q10, q11, [{act}, #48]",
            "ldp q12, q13, [{act}, #80]",
            "ldp q14, q15, [{act}, #112]",
            "add {act}, {act}, #208",
            ptq1_row_source_asm!("p0", "v6", "0", "step4"),
            ptq1_row_source_asm!("p1", "v6", "1", "step4"),
            ptq1_row_source_asm!("p2", "v6", "2", "step4"),
            ptq1_row_source_asm!("p3", "v6", "3", "step4"),
            "fmul v6.4s, v6.4s, v17.4s",
            "fadd v0.4s, v0.4s, v6.4s",
            ptq1_row_source_asm!("p4", "v6", "0", "step4"),
            ptq1_row_source_asm!("p5", "v6", "1", "step4"),
            ptq1_row_source_asm!("p6", "v6", "2", "step4"),
            ptq1_row_source_asm!("p7", "v6", "3", "step4"),
            "fmul v6.4s, v6.4s, v18.4s",
            "fadd v1.4s, v1.4s, v6.4s",
            "subs {cnt}, {cnt}, #1",
            "b.ne 1b",
            "mov {ok}, #1",
            "b 3f",
            "2:",
            "mov {ok}, #0",
            "3:",
            p0 = inout(reg) p0 => _,
            p1 = inout(reg) p1 => _,
            p2 = inout(reg) p2 => _,
            p3 = inout(reg) p3 => _,
            p4 = inout(reg) p4 => _,
            p5 = inout(reg) p5 => _,
            p6 = inout(reg) p6 => _,
            p7 = inout(reg) p7 => _,
            act = inout(reg) act => _,
            cnt = inout(reg) cnt => _,
            step4 = in(reg) 4_usize,
            t0 = out(reg) _,
            t1 = out(reg) _,
            ok = out(reg) ok,
            inout("v0") sum_lo,
            inout("v1") sum_hi,
            out("v2") _,
            out("v3") _,
            out("v4") _,
            out("v5") _,
            out("v6") _,
            out("v8") _,
            out("v9") _,
            out("v10") _,
            out("v11") _,
            out("v12") _,
            out("v13") _,
            out("v14") _,
            out("v15") _,
            out("v16") _,
            out("v17") _,
            out("v18") _,
            out("v19") _,
            out("v20") _,
            out("v21") _,
            in("v22") pow_mid01,
            in("v23") pow_mid23,
            in("v24") pow_tail,
            out("v26") _,
            out("v27") _,
            out("v28") _,
            out("v29") _,
            out("v30") _,
            out("v31") _,
            options(nostack, readonly),
        );
    }
    if ok == 0 {
        return false;
    }
    let out_halves = sums.as_chunks_mut::<4>().0;
    store_f32x4_neon(sum_lo, &mut out_halves[0]);
    store_f32x4_neon(sum_hi, &mut out_halves[1]);
    true
}

/// Caller must guarantee `block_count > 0`, every `row_ptrs[r]` covers
/// `block_count * 28` readable bytes, and `act_ptr` covers `block_count`
/// `Ptq1Activation` elements.
#[expect(
    clippy::too_many_lines,
    reason = "Register-resident 4-row PTQ1 source asm kernel avoids stack spills."
)]
#[target_feature(enable = "neon,dotprod")]
unsafe fn ptq1_packed_4rows_source_accum(
    row_ptrs: [*const u8; 4],
    act_ptr: *const super::Ptq1Activation,
    block_count: usize,
    sums: &mut [f32; 4],
) -> bool {
    if block_count == 0 {
        return true;
    }
    let pow_mid01 = load_u8x16(&PTQ1_POW3_MID_01);
    let pow_mid23 = load_u8x16(&PTQ1_POW3_MID_23);
    let pow_tail = load_u8x16(&PTQ1_POW3_TAIL);
    let mut sum_v = load_f32x4_neon(sums);
    let p0 = row_ptrs[0];
    let p1 = row_ptrs[1];
    let p2 = row_ptrs[2];
    let p3 = row_ptrs[3];
    let act = act_ptr;
    let cnt = block_count;
    let ok: usize;

    // SAFETY: Caller guarantees `row_ptrs[0..4]` and `act_ptr` remain valid for
    // `block_count` blocks; all clobbered SIMD registers are listed in operands.
    unsafe {
        core::arch::asm!(
            "movi v19.16b, #86",
            "movi v20.16b, #170",
            "movi v21.16b, #3",
            "1:",
            "ldrh {t0:w}, [{p0}, #26]",
            "ldrh {t1:w}, [{p1}, #26]",
            "fmov s17, {t0:w}",
            "ldrh {t0:w}, [{p2}, #26]",
            "ins v17.h[1], {t1:w}",
            "ldrh {t1:w}, [{p3}, #26]",
            "ins v17.h[2], {t0:w}",
            "ins v17.h[3], {t1:w}",
            "fmov d28, d17",
            "bic v28.4h, #0x80, lsl #8",
            "umaxv h28, v28.4h",
            "fmov {t0:w}, s28",
            "lsr {t0:w}, {t0:w}, #10",
            "cmp {t0:w}, #30",
            "b.hi 2f",
            "fcvtl v17.4s, v17.4h",
            "ldr q16, [{act}]",
            "ldp q8, q9, [{act}, #16]",
            "ldp q10, q11, [{act}, #48]",
            "ldp q12, q13, [{act}, #80]",
            "ldp q14, q15, [{act}, #112]",
            "add {act}, {act}, #208",
            ptq1_row_source_asm!("p0", "v6", "0", "step4"),
            ptq1_row_source_asm!("p1", "v6", "1", "step4"),
            ptq1_row_source_asm!("p2", "v6", "2", "step4"),
            ptq1_row_source_asm!("p3", "v6", "3", "step4"),
            "fmul v6.4s, v6.4s, v17.4s",
            "fadd v0.4s, v0.4s, v6.4s",
            "subs {cnt}, {cnt}, #1",
            "b.ne 1b",
            "mov {ok}, #1",
            "b 3f",
            "2:",
            "mov {ok}, #0",
            "3:",
            p0 = inout(reg) p0 => _,
            p1 = inout(reg) p1 => _,
            p2 = inout(reg) p2 => _,
            p3 = inout(reg) p3 => _,
            act = inout(reg) act => _,
            cnt = inout(reg) cnt => _,
            step4 = in(reg) 4_usize,
            t0 = out(reg) _,
            t1 = out(reg) _,
            ok = out(reg) ok,
            inout("v0") sum_v,
            out("v2") _,
            out("v3") _,
            out("v4") _,
            out("v5") _,
            out("v6") _,
            out("v8") _,
            out("v9") _,
            out("v10") _,
            out("v11") _,
            out("v12") _,
            out("v13") _,
            out("v14") _,
            out("v15") _,
            out("v16") _,
            out("v17") _,
            out("v19") _,
            out("v20") _,
            out("v21") _,
            in("v22") pow_mid01,
            in("v23") pow_mid23,
            in("v24") pow_tail,
            out("v26") _,
            out("v27") _,
            out("v28") _,
            out("v29") _,
            out("v30") _,
            out("v31") _,
            options(nostack, readonly),
        );
    }
    if ok == 0 {
        return false;
    }
    store_f32x4_neon(sum_v, sums);
    true
}

/// Caller must guarantee `block_count > 0`, `row_ptr` covers `block_count * 28`
/// readable bytes, and `act_ptr` covers `block_count` `Ptq1Activation` elements.
#[target_feature(enable = "neon,dotprod")]
unsafe fn ptq1_packed_1row_source_accum(
    row_ptr: *const u8,
    act_ptr: *const super::Ptq1Activation,
    block_count: usize,
    sum: &mut f32,
) -> bool {
    if block_count == 0 {
        return true;
    }
    let pow_mid01 = load_u8x16(&PTQ1_POW3_MID_01);
    let pow_mid23 = load_u8x16(&PTQ1_POW3_MID_23);
    let pow_tail = load_u8x16(&PTQ1_POW3_TAIL);
    let mut sum_buf = [*sum, 0.0, 0.0, 0.0];
    let mut sum_v = load_f32x4_neon(&sum_buf);
    let p0 = row_ptr;
    let act = act_ptr;
    let cnt = block_count;
    let ok: usize;

    // SAFETY: Caller guarantees `row_ptr` and `act_ptr` remain valid for
    // `block_count` blocks; all clobbered SIMD registers are listed in operands.
    unsafe {
        core::arch::asm!(
            "movi v19.16b, #86",
            "movi v20.16b, #170",
            "movi v21.16b, #3",
            "1:",
            "add {t0}, {p0}, #26",
            "ldr h17, [{t0}]",
            "umov {t0:w}, v17.h[0]",
            "and {t0:w}, {t0:w}, #0x7fff",
            "lsr {t0:w}, {t0:w}, #10",
            "cmp {t0:w}, #30",
            "b.hi 2f",
            "fcvtl v17.4s, v17.4h",
            "ldr q16, [{act}]",
            "ldp q8, q9, [{act}, #16]",
            "ldp q10, q11, [{act}, #48]",
            "ldp q12, q13, [{act}, #80]",
            "ldp q14, q15, [{act}, #112]",
            "add {act}, {act}, #208",
            ptq1_row_source_asm!("p0", "v6", "0", "step4"),
            "fmul s6, s6, s17",
            "fadd s0, s0, s6",
            "subs {cnt}, {cnt}, #1",
            "b.ne 1b",
            "mov {ok}, #1",
            "b 3f",
            "2:",
            "mov {ok}, #0",
            "3:",
            p0 = inout(reg) p0 => _,
            act = inout(reg) act => _,
            cnt = inout(reg) cnt => _,
            step4 = in(reg) 4_usize,
            t0 = out(reg) _,
            ok = out(reg) ok,
            inout("v0") sum_v,
            out("v2") _,
            out("v3") _,
            out("v4") _,
            out("v5") _,
            out("v6") _,
            out("v8") _,
            out("v9") _,
            out("v10") _,
            out("v11") _,
            out("v12") _,
            out("v13") _,
            out("v14") _,
            out("v15") _,
            out("v16") _,
            out("v17") _,
            out("v19") _,
            out("v20") _,
            out("v21") _,
            in("v22") pow_mid01,
            in("v23") pow_mid23,
            in("v24") pow_tail,
            out("v26") _,
            out("v27") _,
            out("v28") _,
            out("v29") _,
            out("v30") _,
            out("v31") _,
            options(nostack, readonly),
        );
    }
    if ok == 0 {
        return false;
    }
    store_f32x4_neon(sum_v, &mut sum_buf);
    *sum = sum_buf[0];
    true
}

macro_rules! ptq1_row_pinned_sve_asm {
    ($p:literal, $lane:literal, $rs_reg:literal, $rs_lane:literal, $row_idx:literal) => {
        concat!(
            "ldr q26, [{",
            $p,
            "}], #16\n",
            "fmul v25.4s, v16.4s, ",
            $rs_reg,
            ".s[",
            $rs_lane,
            "]\n",
            "shl v28.16b, v26.16b, #1\n",
            "shl v29.16b, v26.16b, #3\n",
            "add v28.16b, v28.16b, v26.16b\n",
            "add v29.16b, v29.16b, v26.16b\n",
            "mov v27.16b, v30.16b\n",
            "umulh z26.b, p0/m, z26.b, z21.b\n",
            "umulh z28.b, p0/m, z28.b, z21.b\n",
            "sdot v27.4s, v26.16b, v8.16b\n",
            "sdot v27.4s, v28.16b, v9.16b\n",
            "scvtf v27.4s, v27.4s\n",
            "fmla ",
            $lane,
            ".4s, v27.4s, v25.s[0]\n",
            "shl v28.16b, v29.16b, #1\n",
            "shl v26.16b, v29.16b, #3\n",
            "add v28.16b, v28.16b, v29.16b\n",
            "add v26.16b, v26.16b, v29.16b\n",
            "mov v27.16b, v31.16b\n",
            "umulh z29.b, p0/m, z29.b, z21.b\n",
            "umulh z28.b, p0/m, z28.b, z21.b\n",
            "sdot v27.4s, v29.16b, v10.16b\n",
            "sdot v27.4s, v28.16b, v11.16b\n",
            "scvtf v27.4s, v27.4s\n",
            "fmla ",
            $lane,
            ".4s, v27.4s, v25.s[1]\n",
            "ldr d28, [{",
            $p,
            "}], #12\n",
            "dup v28.2d, v28.d[0]\n",
            "mul v29.16b, v28.16b, v22.16b\n",
            "ldr q27, [{act}, #176]\n",
            "umulh z26.b, p0/m, z26.b, z21.b\n",
            "umulh z29.b, p0/m, z29.b, z21.b\n",
            "sdot v27.4s, v26.16b, v12.16b\n",
            "sdot v27.4s, v29.16b, v13.16b\n",
            "scvtf v27.4s, v27.4s\n",
            "fmla ",
            $lane,
            ".4s, v27.4s, v25.s[2]\n",
            "mul v26.16b, v28.16b, v23.16b\n",
            "dup v29.4h, v20.h[",
            $row_idx,
            "]\n",
            "mov v28.d[1], v29.d[0]\n",
            "mul v28.16b, v28.16b, v24.16b\n",
            "ldr q27, [{act}, #192]\n",
            "umulh z26.b, p0/m, z26.b, z21.b\n",
            "umulh z28.b, p0/m, z28.b, z21.b\n",
            "sdot v27.4s, v26.16b, v14.16b\n",
            "sdot v27.4s, v28.16b, v15.16b\n",
            "scvtf v27.4s, v27.4s\n",
            "fmla ",
            $lane,
            ".4s, v27.4s, v25.s[3]\n",
        )
    };
}

macro_rules! ptq1_8rows_pinned_sve_block_asm {
    () => {
        concat!(
            "ldr q16, [{act}]\n",
            "ldp q8, q9, [{act}, #16]\n",
            "ldp q10, q11, [{act}, #48]\n",
            "ldp q12, q13, [{act}, #80]\n",
            "ldp q14, q15, [{act}, #112]\n",
            "ldp q30, q31, [{act}, #144]\n",
            "ldr s25, [{p0}, #24]\n",
            "ldr s26, [{p1}, #24]\n",
            "ldr s27, [{p2}, #24]\n",
            "ldr s28, [{p3}, #24]\n",
            "trn1 v25.4s, v25.4s, v26.4s\n",
            "trn1 v27.4s, v27.4s, v28.4s\n",
            "trn1 v25.2d, v25.2d, v27.2d\n",
            "ldr s26, [{p4}, #24]\n",
            "ldr s27, [{p5}, #24]\n",
            "ldr s28, [{p6}, #24]\n",
            "ldr s29, [{p7}, #24]\n",
            "trn1 v26.4s, v26.4s, v27.4s\n",
            "trn1 v28.4s, v28.4s, v29.4s\n",
            "trn1 v26.2d, v26.2d, v28.2d\n",
            "uzp1 v20.8h, v25.8h, v26.8h\n",
            "uzp2 v28.8h, v25.8h, v26.8h\n",
            "fcvtl v17.4s, v28.4h\n",
            "fcvtl2 v18.4s, v28.8h\n",
            "bic v28.8h, #0x80, lsl #8\n",
            "umax v19.8h, v19.8h, v28.8h\n",
            ptq1_row_pinned_sve_asm!("p0", "v0", "v17", "0", "0"),
            ptq1_row_pinned_sve_asm!("p1", "v1", "v17", "1", "1"),
            ptq1_row_pinned_sve_asm!("p2", "v2", "v17", "2", "2"),
            ptq1_row_pinned_sve_asm!("p3", "v3", "v17", "3", "3"),
            ptq1_row_pinned_sve_asm!("p4", "v4", "v18", "0", "4"),
            ptq1_row_pinned_sve_asm!("p5", "v5", "v18", "1", "5"),
            ptq1_row_pinned_sve_asm!("p6", "v6", "v18", "2", "6"),
            ptq1_row_pinned_sve_asm!("p7", "v7", "v18", "3", "7"),
            "add {act}, {act}, #208\n",
        )
    };
}

macro_rules! ptq1_row_pinned_4sve_asm {
    ($p:literal, $lane:literal, $row_idx:literal) => {
        concat!(
            "ldr q26, [{",
            $p,
            "}], #16\n",
            "fmul v25.4s, v16.4s, v17.s[",
            $row_idx,
            "]\n",
            "shl v28.16b, v26.16b, #1\n",
            "shl v29.16b, v26.16b, #3\n",
            "add v28.16b, v28.16b, v26.16b\n",
            "add v29.16b, v29.16b, v26.16b\n",
            "mov v27.16b, v4.16b\n",
            "umulh z26.b, p0/m, z26.b, z21.b\n",
            "umulh z28.b, p0/m, z28.b, z21.b\n",
            "sdot v27.4s, v26.16b, v8.16b\n",
            "sdot v27.4s, v28.16b, v9.16b\n",
            "scvtf v27.4s, v27.4s\n",
            "fmla ",
            $lane,
            ".4s, v27.4s, v25.s[0]\n",
            "shl v28.16b, v29.16b, #1\n",
            "shl v26.16b, v29.16b, #3\n",
            "add v28.16b, v28.16b, v29.16b\n",
            "add v26.16b, v26.16b, v29.16b\n",
            "mov v27.16b, v5.16b\n",
            "umulh z29.b, p0/m, z29.b, z21.b\n",
            "umulh z28.b, p0/m, z28.b, z21.b\n",
            "sdot v27.4s, v29.16b, v10.16b\n",
            "sdot v27.4s, v28.16b, v11.16b\n",
            "scvtf v27.4s, v27.4s\n",
            "fmla ",
            $lane,
            ".4s, v27.4s, v25.s[1]\n",
            "ldr d28, [{",
            $p,
            "}], #12\n",
            "dup v28.2d, v28.d[0]\n",
            "mul v29.16b, v28.16b, v22.16b\n",
            "mov v27.16b, v6.16b\n",
            "umulh z26.b, p0/m, z26.b, z21.b\n",
            "umulh z29.b, p0/m, z29.b, z21.b\n",
            "sdot v27.4s, v26.16b, v12.16b\n",
            "sdot v27.4s, v29.16b, v13.16b\n",
            "scvtf v27.4s, v27.4s\n",
            "fmla ",
            $lane,
            ".4s, v27.4s, v25.s[2]\n",
            "mul v26.16b, v28.16b, v23.16b\n",
            "dup v29.4h, v20.h[",
            $row_idx,
            "]\n",
            "mov v28.d[1], v29.d[0]\n",
            "mul v28.16b, v28.16b, v24.16b\n",
            "mov v27.16b, v7.16b\n",
            "umulh z26.b, p0/m, z26.b, z21.b\n",
            "umulh z28.b, p0/m, z28.b, z21.b\n",
            "sdot v27.4s, v26.16b, v14.16b\n",
            "sdot v27.4s, v28.16b, v15.16b\n",
            "scvtf v27.4s, v27.4s\n",
            "fmla ",
            $lane,
            ".4s, v27.4s, v25.s[3]\n",
        )
    };
}

macro_rules! ptq1_4rows_pinned_sve_block_asm {
    () => {
        concat!(
            "ldr q16, [{act}]\n",
            "ldp q8, q9, [{act}, #16]\n",
            "ldp q10, q11, [{act}, #48]\n",
            "ldp q12, q13, [{act}, #80]\n",
            "ldp q14, q15, [{act}, #112]\n",
            "ldp q4, q5, [{act}, #144]\n",
            "ldp q6, q7, [{act}, #176]\n",
            "add {act}, {act}, #208\n",
            "ldr s25, [{p0}, #24]\n",
            "ldr s26, [{p1}, #24]\n",
            "ldr s27, [{p2}, #24]\n",
            "ldr s28, [{p3}, #24]\n",
            "trn1 v25.2s, v25.2s, v26.2s\n",
            "trn1 v27.2s, v27.2s, v28.2s\n",
            "uzp1 v20.4h, v25.4h, v27.4h\n",
            "uzp2 v28.4h, v25.4h, v27.4h\n",
            "fcvtl v17.4s, v28.4h\n",
            "bic v28.4h, #0x80, lsl #8\n",
            "umax v19.4h, v19.4h, v28.4h\n",
            ptq1_row_pinned_4sve_asm!("p0", "v0", "0"),
            ptq1_row_pinned_4sve_asm!("p1", "v1", "1"),
            ptq1_row_pinned_4sve_asm!("p2", "v2", "2"),
            ptq1_row_pinned_4sve_asm!("p3", "v3", "3"),
        )
    };
}

/// Caller must guarantee `has_sve()` and `dotprod`, `block_count > 0`, every
/// `row_ptrs[r]` covers `block_count * 28` readable bytes, and `act_ptr` covers
/// `block_count` `Ptq1Activation` elements.
#[expect(
    clippy::too_many_lines,
    reason = "Non-streaming SVE+NEON 8-row PTQ1 pinned asm kernel avoids stack spills."
)]
#[target_feature(enable = "neon,dotprod")]
unsafe fn ptq1_packed_8rows_pinned_sve(
    row_ptrs: [*const u8; 8],
    act_ptr: *const super::Ptq1Activation,
    block_count: usize,
    lanes: &mut [[f32; 4]; 8],
) -> bool {
    if block_count == 0 {
        return true;
    }
    let pow_mid01 = load_u8x16(&PTQ1_POW3_MID_01);
    let pow_mid23 = load_u8x16(&PTQ1_POW3_MID_23);
    let pow_tail = load_u8x16(&PTQ1_POW3_TAIL);
    let mut l0 = load_f32x4_neon(&lanes[0]);
    let mut l1 = load_f32x4_neon(&lanes[1]);
    let mut l2 = load_f32x4_neon(&lanes[2]);
    let mut l3 = load_f32x4_neon(&lanes[3]);
    let mut l4 = load_f32x4_neon(&lanes[4]);
    let mut l5 = load_f32x4_neon(&lanes[5]);
    let mut l6 = load_f32x4_neon(&lanes[6]);
    let mut l7 = load_f32x4_neon(&lanes[7]);
    let p0 = row_ptrs[0];
    let p1 = row_ptrs[1];
    let p2 = row_ptrs[2];
    let p3 = row_ptrs[3];
    let p4 = row_ptrs[4];
    let p5 = row_ptrs[5];
    let p6 = row_ptrs[6];
    let p7 = row_ptrs[7];
    let act = act_ptr;
    let cnt = block_count;
    let ok: usize;

    // SAFETY: Caller verified non-streaming SVE + dotprod and valid extents
    // for all 8 rows and activations.
    unsafe {
        core::arch::asm!(
            ".arch_extension sve",
            "mov {t0}, #16",
            "whilelt p0.b, xzr, {t0}",
            "movi v19.2d, #0",
            "movi v21.16b, #3",
            "tst {cnt}, #1",
            "b.eq 1f",
            ptq1_8rows_pinned_sve_block_asm!(),
            "sub {cnt}, {cnt}, #1",
            "cbz {cnt}, 2f",
            "1:",
            "prfm pldl1keep, [{p0}, #112]",
            "prfm pldl1keep, [{p1}, #112]",
            "prfm pldl1keep, [{p2}, #112]",
            "prfm pldl1keep, [{p3}, #112]",
            "prfm pldl1keep, [{p4}, #112]",
            "prfm pldl1keep, [{p5}, #112]",
            "prfm pldl1keep, [{p6}, #112]",
            "prfm pldl1keep, [{p7}, #112]",
            ptq1_8rows_pinned_sve_block_asm!(),
            ptq1_8rows_pinned_sve_block_asm!(),
            "subs {cnt}, {cnt}, #2",
            "b.ne 1b",
            "2:",
            "umaxv h28, v19.8h",
            "fmov {t0:w}, s28",
            "lsr {t0:w}, {t0:w}, #10",
            "cmp {t0:w}, #30",
            "cset {ok}, ls",
            p0 = inout(reg) p0 => _,
            p1 = inout(reg) p1 => _,
            p2 = inout(reg) p2 => _,
            p3 = inout(reg) p3 => _,
            p4 = inout(reg) p4 => _,
            p5 = inout(reg) p5 => _,
            p6 = inout(reg) p6 => _,
            p7 = inout(reg) p7 => _,
            act = inout(reg) act => _,
            cnt = inout(reg) cnt => _,
            t0 = out(reg) _,
            ok = out(reg) ok,
            inout("v0") l0,
            inout("v1") l1,
            inout("v2") l2,
            inout("v3") l3,
            inout("v4") l4,
            inout("v5") l5,
            inout("v6") l6,
            inout("v7") l7,
            out("v8") _,
            out("v9") _,
            out("v10") _,
            out("v11") _,
            out("v12") _,
            out("v13") _,
            out("v14") _,
            out("v15") _,
            out("v16") _,
            out("v17") _,
            out("v18") _,
            out("v19") _,
            out("v20") _,
            out("v21") _,
            in("v22") pow_mid01,
            in("v23") pow_mid23,
            in("v24") pow_tail,
            out("v25") _,
            out("v26") _,
            out("v27") _,
            out("v28") _,
            out("v29") _,
            out("v30") _,
            out("v31") _,
            out("p0") _,
            options(nostack, readonly),
        );
    }
    if ok == 0 {
        return false;
    }
    store_f32x4_neon(l0, &mut lanes[0]);
    store_f32x4_neon(l1, &mut lanes[1]);
    store_f32x4_neon(l2, &mut lanes[2]);
    store_f32x4_neon(l3, &mut lanes[3]);
    store_f32x4_neon(l4, &mut lanes[4]);
    store_f32x4_neon(l5, &mut lanes[5]);
    store_f32x4_neon(l6, &mut lanes[6]);
    store_f32x4_neon(l7, &mut lanes[7]);
    true
}

/// Caller must guarantee `has_sve()` and `dotprod`, `block_count > 0`, every
/// `row_ptrs[r]` covers `block_count * 28` readable bytes, and `act_ptr` covers
/// `block_count` `Ptq1Activation` elements.
#[target_feature(enable = "neon,dotprod")]
unsafe fn ptq1_packed_4rows_pinned_sve(
    row_ptrs: [*const u8; 4],
    act_ptr: *const super::Ptq1Activation,
    block_count: usize,
    lanes: &mut [[f32; 4]; 4],
) -> bool {
    if block_count == 0 {
        return true;
    }
    let pow_mid01 = load_u8x16(&PTQ1_POW3_MID_01);
    let pow_mid23 = load_u8x16(&PTQ1_POW3_MID_23);
    let pow_tail = load_u8x16(&PTQ1_POW3_TAIL);
    let mut l0 = load_f32x4_neon(&lanes[0]);
    let mut l1 = load_f32x4_neon(&lanes[1]);
    let mut l2 = load_f32x4_neon(&lanes[2]);
    let mut l3 = load_f32x4_neon(&lanes[3]);
    let p0 = row_ptrs[0];
    let p1 = row_ptrs[1];
    let p2 = row_ptrs[2];
    let p3 = row_ptrs[3];
    let act = act_ptr;
    let cnt = block_count;
    let ok: usize;

    // SAFETY: Caller verified non-streaming SVE + dotprod and valid extents
    // for all 4 rows and activations.
    unsafe {
        core::arch::asm!(
            ".arch_extension sve",
            "mov {t0}, #16",
            "whilelt p0.b, xzr, {t0}",
            "movi v19.2d, #0",
            "movi v21.16b, #3",
            "tst {cnt}, #1",
            "b.eq 1f",
            ptq1_4rows_pinned_sve_block_asm!(),
            "sub {cnt}, {cnt}, #1",
            "cbz {cnt}, 2f",
            "1:",
            "prfm pldl1keep, [{p0}, #112]",
            "prfm pldl1keep, [{p1}, #112]",
            "prfm pldl1keep, [{p2}, #112]",
            "prfm pldl1keep, [{p3}, #112]",
            ptq1_4rows_pinned_sve_block_asm!(),
            ptq1_4rows_pinned_sve_block_asm!(),
            "subs {cnt}, {cnt}, #2",
            "b.ne 1b",
            "2:",
            "umaxv h28, v19.4h",
            "fmov {t0:w}, s28",
            "lsr {t0:w}, {t0:w}, #10",
            "cmp {t0:w}, #30",
            "cset {ok}, ls",
            p0 = inout(reg) p0 => _,
            p1 = inout(reg) p1 => _,
            p2 = inout(reg) p2 => _,
            p3 = inout(reg) p3 => _,
            act = inout(reg) act => _,
            cnt = inout(reg) cnt => _,
            t0 = out(reg) _,
            ok = out(reg) ok,
            inout("v0") l0,
            inout("v1") l1,
            inout("v2") l2,
            inout("v3") l3,
            out("v4") _,
            out("v5") _,
            out("v6") _,
            out("v7") _,
            out("v8") _,
            out("v9") _,
            out("v10") _,
            out("v11") _,
            out("v12") _,
            out("v13") _,
            out("v14") _,
            out("v15") _,
            out("v16") _,
            out("v17") _,
            out("v19") _,
            out("v20") _,
            out("v21") _,
            in("v22") pow_mid01,
            in("v23") pow_mid23,
            in("v24") pow_tail,
            out("v25") _,
            out("v26") _,
            out("v27") _,
            out("v28") _,
            out("v29") _,
            out("v30") _,
            out("v31") _,
            out("p0") _,
            options(nostack, readonly),
        );
    }
    if ok == 0 {
        return false;
    }
    store_f32x4_neon(l0, &mut lanes[0]);
    store_f32x4_neon(l1, &mut lanes[1]);
    store_f32x4_neon(l2, &mut lanes[2]);
    store_f32x4_neon(l3, &mut lanes[3]);
    true
}

#[target_feature(enable = "neon,dotprod")]
fn ptq1_packed_rows_pinned_dotprod<const ROWS: usize>(
    rows: [&[u8]; ROWS],
    activation: &[super::Ptq1Activation],
    sve: bool,
) -> Option<[f32; ROWS]> {
    let block_count = activation.len();
    let act_ptr = activation.as_ptr();
    let mut out = [0.0_f32; ROWS];
    if ROWS == 1 {
        let mut lanes = [0.0_f32; 4];
        // SAFETY: Safe parent checked `rows[0].len() == block_count * 28`.
        let ok = unsafe {
            ptq1_packed_1row_pinned_accum(rows[0].as_ptr(), act_ptr, block_count, &mut lanes)
        };
        if !ok {
            return None;
        }
        let val = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
        out[0] = val;
        return val.is_finite().then_some(out);
    }
    if ROWS == 4 {
        let mut lanes = [[0.0_f32; 4]; 4];
        let ptrs = [
            rows[0].as_ptr(),
            rows[1].as_ptr(),
            rows[2].as_ptr(),
            rows[3].as_ptr(),
        ];
        let ok = if sve {
            // SAFETY: Safe parent checked `rows[0..4]` extents; `sve` was runtime-verified.
            unsafe { ptq1_packed_4rows_pinned_sve(ptrs, act_ptr, block_count, &mut lanes) }
        } else {
            // SAFETY: Safe parent checked `rows[0..4]` each have `block_count * 28` bytes.
            unsafe { ptq1_packed_4rows_pinned_accum(ptrs, act_ptr, block_count, &mut lanes) }
        };
        if !ok {
            return None;
        }
        for (dst, l) in out.iter_mut().zip(lanes) {
            *dst = (l[0] + l[1]) + (l[2] + l[3]);
        }
        return out.iter().all(|v| v.is_finite()).then_some(out);
    }
    for (rows8, dst8) in rows
        .as_chunks::<8>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<8>().0)
    {
        let mut lanes = [[0.0_f32; 4]; 8];
        let ptrs: [*const u8; 8] = std::array::from_fn(|r| rows8[r].as_ptr());
        let ok = if sve {
            // SAFETY: Safe parent checked every row in `rows8` extents; `sve` was runtime-verified.
            unsafe { ptq1_packed_8rows_pinned_sve(ptrs, act_ptr, block_count, &mut lanes) }
        } else {
            // SAFETY: Safe parent checked every row in `rows8` has `block_count * 28` bytes.
            unsafe { ptq1_packed_8rows_pinned_accum(ptrs, act_ptr, block_count, &mut lanes) }
        };
        if !ok {
            return None;
        }
        for (dst, l) in dst8.iter_mut().zip(lanes) {
            *dst = (l[0] + l[1]) + (l[2] + l[3]);
        }
    }
    out.iter().all(|v| v.is_finite()).then_some(out)
}

#[target_feature(enable = "neon,dotprod")]
fn ptq1_packed_rows_source_dotprod<const ROWS: usize>(
    rows: [&[u8]; ROWS],
    activation: &[super::Ptq1Activation],
) -> Option<[f32; ROWS]> {
    let block_count = activation.len();
    let act_ptr = activation.as_ptr();
    let mut out = [0.0_f32; ROWS];
    if ROWS == 1 {
        let mut sum = 0.0_f32;
        // SAFETY: Safe parent checked `rows[0].len() == block_count * 28`.
        let ok = unsafe {
            ptq1_packed_1row_source_accum(rows[0].as_ptr(), act_ptr, block_count, &mut sum)
        };
        if !ok {
            return None;
        }
        out[0] = sum;
        return sum.is_finite().then_some(out);
    }
    if ROWS == 4 {
        let ptrs = [
            rows[0].as_ptr(),
            rows[1].as_ptr(),
            rows[2].as_ptr(),
            rows[3].as_ptr(),
        ];
        let mut sums = [0.0_f32; 4];
        // SAFETY: Safe parent checked `rows[0..4]` each have `block_count * 28` bytes.
        let ok = unsafe { ptq1_packed_4rows_source_accum(ptrs, act_ptr, block_count, &mut sums) };
        if !ok {
            return None;
        }
        out.copy_from_slice(&sums);
        return out.iter().all(|v| v.is_finite()).then_some(out);
    }
    for (rows8, dst8) in rows
        .as_chunks::<8>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<8>().0)
    {
        let ptrs: [*const u8; 8] = std::array::from_fn(|r| rows8[r].as_ptr());
        let mut sums = [0.0_f32; 8];
        // SAFETY: Safe parent checked every row in `rows8` has `block_count * 28` bytes.
        let ok = unsafe { ptq1_packed_8rows_source_accum(ptrs, act_ptr, block_count, &mut sums) };
        if !ok {
            return None;
        }
        dst8.copy_from_slice(&sums);
    }
    out.iter().all(|v| v.is_finite()).then_some(out)
}

#[target_feature(enable = "neon,dotprod")]
fn ptq1_packed_rows_dotprod<const ROWS: usize, const PINNED: bool>(
    rows: [&[u8]; ROWS],
    activation: &[super::Ptq1Activation],
    sve: bool,
) -> Option<[f32; ROWS]> {
    if PINNED {
        ptq1_packed_rows_pinned_dotprod(rows, activation, sve)
    } else {
        ptq1_packed_rows_source_dotprod(rows, activation)
    }
}

fn ptq1_pack_chunk_activations<A: super::Q8ActivationView>(
    activation: &[A],
    chunk_start: usize,
    dst: &mut [super::Ptq1Activation],
) {
    for (k, slot) in dst.iter_mut().enumerate() {
        let g_base = (chunk_start + k) * 4;
        let g0 = &activation[g_base];
        let g1 = &activation[g_base + 1];
        let g2 = &activation[g_base + 2];
        let g3 = &activation[g_base + 3];
        *slot = super::Ptq1Activation::from_groups_without_lane_sums(
            [g0.scale(), g1.scale(), g2.scale(), g3.scale()],
            [g0.values(), g1.values(), g2.values(), g3.values()],
        );
    }
}

#[target_feature(enable = "neon,dotprod")]
fn ptq1_scaled_pinned_chunk<const ROWS: usize>(
    rows: [&[u8]; ROWS],
    byte_off: usize,
    act_ptr: *const super::Ptq1Activation,
    chunk_blocks: usize,
    lanes: &mut [[f32; 4]; ROWS],
) -> bool {
    if ROWS == 1 {
        let mut l1 = lanes[0];
        let p0 = rows[0][byte_off..].as_ptr();
        // SAFETY: `rows[0][byte_off..]` has at least `chunk_blocks * 28` readable bytes.
        let ok = unsafe { ptq1_packed_1row_pinned_accum(p0, act_ptr, chunk_blocks, &mut l1) };
        if !ok {
            return false;
        }
        lanes[0] = l1;
        return true;
    }
    if ROWS == 4 {
        let mut l4: [[f32; 4]; 4] = [lanes[0], lanes[1], lanes[2], lanes[3]];
        let ptrs: [*const u8; 4] = std::array::from_fn(|r| rows[r][byte_off..].as_ptr());
        // SAFETY: Each `rows[r][byte_off..]` has at least `chunk_blocks * 28` readable bytes.
        let ok = unsafe { ptq1_packed_4rows_pinned_accum(ptrs, act_ptr, chunk_blocks, &mut l4) };
        if !ok {
            return false;
        }
        lanes[..4].copy_from_slice(&l4);
        return true;
    }
    for (rows8, lanes8) in rows
        .as_chunks::<8>()
        .0
        .iter()
        .zip(lanes.as_chunks_mut::<8>().0)
    {
        let mut l8 = *lanes8;
        let ptrs: [*const u8; 8] = std::array::from_fn(|r| rows8[r][byte_off..].as_ptr());
        // SAFETY: Each `rows8[r][byte_off..]` has at least `chunk_blocks * 28` readable bytes.
        let ok = unsafe { ptq1_packed_8rows_pinned_accum(ptrs, act_ptr, chunk_blocks, &mut l8) };
        if !ok {
            return false;
        }
        lanes8.copy_from_slice(&l8);
    }
    true
}

#[target_feature(enable = "neon,dotprod")]
fn ptq1_scaled_source_chunk<const ROWS: usize>(
    rows: [&[u8]; ROWS],
    byte_off: usize,
    act_ptr: *const super::Ptq1Activation,
    chunk_blocks: usize,
    out: &mut [f32; ROWS],
) -> bool {
    if ROWS == 1 {
        let mut s1 = out[0];
        let p0 = rows[0][byte_off..].as_ptr();
        // SAFETY: `rows[0][byte_off..]` has at least `chunk_blocks * 28` readable bytes.
        let ok = unsafe { ptq1_packed_1row_source_accum(p0, act_ptr, chunk_blocks, &mut s1) };
        if !ok {
            return false;
        }
        out[0] = s1;
        return true;
    }
    if ROWS == 4 {
        let mut s4 = [out[0], out[1], out[2], out[3]];
        let ptrs: [*const u8; 4] = std::array::from_fn(|r| rows[r][byte_off..].as_ptr());
        // SAFETY: Each `rows[r][byte_off..]` has at least `chunk_blocks * 28` readable bytes.
        let ok = unsafe { ptq1_packed_4rows_source_accum(ptrs, act_ptr, chunk_blocks, &mut s4) };
        if !ok {
            return false;
        }
        out[..4].copy_from_slice(&s4);
        return true;
    }
    for (rows8, out8) in rows
        .as_chunks::<8>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<8>().0)
    {
        let mut s8 = *out8;
        let ptrs: [*const u8; 8] = std::array::from_fn(|r| rows8[r][byte_off..].as_ptr());
        // SAFETY: Each `rows8[r][byte_off..]` has at least `chunk_blocks * 28` readable bytes.
        let ok = unsafe { ptq1_packed_8rows_source_accum(ptrs, act_ptr, chunk_blocks, &mut s8) };
        if !ok {
            return false;
        }
        out8.copy_from_slice(&s8);
    }
    true
}

#[target_feature(enable = "neon,dotprod")]
fn ptq1_scaled_rows_dotprod<const ROWS: usize, const PINNED: bool, A: super::Q8ActivationView>(
    rows: [&[u8]; ROWS],
    activation: &[A],
) -> Option<[f32; ROWS]> {
    const CHUNK_BLOCKS: usize = 16;
    let total_blocks = rows[0].len() / 28;
    if total_blocks == 0 {
        return None;
    }
    let mut act_buf = [super::Ptq1Activation::zeroed(); CHUNK_BLOCKS];
    let mut out = [0.0_f32; ROWS];
    let mut lanes = [[0.0_f32; 4]; ROWS];

    for chunk_start in (0..total_blocks).step_by(CHUNK_BLOCKS) {
        let chunk_blocks = (total_blocks - chunk_start).min(CHUNK_BLOCKS);
        ptq1_pack_chunk_activations(activation, chunk_start, &mut act_buf[..chunk_blocks]);
        let byte_off = chunk_start * 28;
        let act_ptr = act_buf.as_ptr();
        let ok = if PINNED {
            ptq1_scaled_pinned_chunk(rows, byte_off, act_ptr, chunk_blocks, &mut lanes)
        } else {
            ptq1_scaled_source_chunk(rows, byte_off, act_ptr, chunk_blocks, &mut out)
        };
        if !ok {
            return None;
        }
    }
    if PINNED {
        for (dst, l) in out.iter_mut().zip(lanes) {
            *dst = (l[0] + l[1]) + (l[2] + l[3]);
        }
    }
    out.iter().all(|v| v.is_finite()).then_some(out)
}

#[target_feature(enable = "neon,dotprod")]
fn ptq1_scaled_8x_2_pinned_dotprod<A: super::Q8ActivationView>(
    rows: [&[u8]; 8],
    tokens: &[&[A]],
    out: &mut [[f32; 8]; 8],
) -> Option<()> {
    use core::arch::aarch64::{vcvtq_f32_s32, vdupq_n_f32, vfmaq_f32};

    let row_chunks = rows.map(|row| row.as_chunks::<28>().0);
    let block_count = row_chunks[0].len();
    let mut lanes = [[vdupq_n_f32(0.0); 2]; 8];
    for block_idx in 0..block_count {
        let blocks: [&[u8; 28]; 8] = std::array::from_fn(|r| &row_chunks[r][block_idx]);
        let row_scales = ptq1_validate_row_scales(&blocks)?;
        for group in 0..4 {
            let act0 = &tokens[0][block_idx * 4 + group];
            let act1 = &tokens[1][block_idx * 4 + group];
            let ah0 = act0.values().as_chunks::<16>().0;
            let ah1 = act1.values().as_chunks::<16>().0;
            let a00 = load_s8x16(&ah0[0]);
            let a01 = load_s8x16(&ah0[1]);
            let a10 = load_s8x16(&ah1[0]);
            let a11 = load_s8x16(&ah1[1]);
            let s0 = act0.scale();
            let s1 = act1.scale();
            for (r, (&block, &rs)) in blocks.iter().zip(row_scales.iter()).enumerate() {
                let (w0, w1) = unpack_ptq1_group(block, group);
                let d0 = sdot32_vec(w0, w1, a00, a01);
                let d1 = sdot32_vec(w0, w1, a10, a11);
                lanes[r][0] = vfmaq_f32(lanes[r][0], vcvtq_f32_s32(d0), vdupq_n_f32(rs * s0));
                lanes[r][1] = vfmaq_f32(lanes[r][1], vcvtq_f32_s32(d1), vdupq_n_f32(rs * s1));
            }
        }
    }
    *out = [[0.0_f32; 8]; 8];
    for (r, row_lanes) in lanes.into_iter().enumerate() {
        for (t, lane) in row_lanes.into_iter().enumerate() {
            let mut l = [0.0_f32; 4];
            store_f32x4_neon(lane, &mut l);
            out[r][t] = (l[0] + l[1]) + (l[2] + l[3]);
        }
    }
    out.iter().flatten().all(|v| v.is_finite()).then_some(())
}

#[target_feature(enable = "neon,dotprod")]
fn ptq1_scaled_8x_n_pinned_dotprod<const N_TOK: usize, A: super::Q8ActivationView>(
    rows: [&[u8]; 8],
    tokens: &[&[A]],
    out: &mut [[f32; 8]; 8],
) -> Option<()> {
    use core::arch::aarch64::{vcvtq_f32_s32, vdupq_n_f32, vfmaq_f32};

    let row_chunks = rows.map(|row| row.as_chunks::<28>().0);
    let block_count = row_chunks[0].len();
    let mut lanes = [[[0.0_f32; 4]; N_TOK]; 8];
    for block_idx in 0..block_count {
        let blocks: [&[u8; 28]; 8] = std::array::from_fn(|r| &row_chunks[r][block_idx]);
        let row_scales = ptq1_validate_row_scales(&blocks)?;
        for group in 0..4 {
            let a_scales: [f32; N_TOK] =
                std::array::from_fn(|t| tokens[t][block_idx * 4 + group].scale());
            let acts: [(int8x16_t, int8x16_t); N_TOK] = std::array::from_fn(|t| {
                let ah = tokens[t][block_idx * 4 + group]
                    .values()
                    .as_chunks::<16>()
                    .0;
                (load_s8x16(&ah[0]), load_s8x16(&ah[1]))
            });
            for (r, (&block, &rs)) in blocks.iter().zip(row_scales.iter()).enumerate() {
                let (w0, w1) = unpack_ptq1_group(block, group);
                for (t, ((&(a0, a1), &a_s), lane_buf)) in acts
                    .iter()
                    .zip(a_scales.iter())
                    .zip(lanes[r].iter_mut())
                    .enumerate()
                {
                    let _ = t;
                    let sdot = sdot32_vec(w0, w1, a0, a1);
                    let cur = load_f32x4_neon(lane_buf);
                    let next = vfmaq_f32(cur, vcvtq_f32_s32(sdot), vdupq_n_f32(rs * a_s));
                    store_f32x4_neon(next, lane_buf);
                }
            }
        }
    }
    *out = [[0.0_f32; 8]; 8];
    for (r, row_lanes) in lanes.iter().enumerate() {
        for (t, l) in row_lanes.iter().enumerate() {
            out[r][t] = (l[0] + l[1]) + (l[2] + l[3]);
        }
    }
    out.iter().flatten().all(|v| v.is_finite()).then_some(())
}

#[target_feature(enable = "neon,i8mm,dotprod")]
fn ptq1_scaled_8x_pairs_source_i8mm<
    const PAIRS: usize,
    const HAS_ODD: bool,
    A: super::Q8ActivationView,
>(
    rows: [&[u8]; 8],
    tokens: &[&[A]],
    out: &mut [[f32; 8]; 8],
) -> Option<()> {
    use core::arch::aarch64::{vaddq_f32, vcvtq_f32_s32, vdupq_n_f32, vmulq_f32};

    let row_chunks = rows.map(|row| row.as_chunks::<28>().0);
    let block_count = row_chunks[0].len();
    let mut sums_pair = [[[0.0_f32; 4]; PAIRS]; 4];
    let mut sums_odd = [[0.0_f32; 4]; 4];
    let zero_s32 = vdupq_n_s32(0);

    for block_idx in 0..block_count {
        let blocks: [&[u8; 28]; 8] = std::array::from_fn(|r| &row_chunks[r][block_idx]);
        let row_scales = ptq1_validate_row_scales(&blocks)?;
        for row_pair in 0..4 {
            let r0 = row_pair * 2;
            let r1 = r0 + 1;
            let t_r0 = unpack_ptq1_vectors(blocks[r0]);
            let t_r1 = unpack_ptq1_vectors(blocks[r1]);
            let mut dot_sum_pairs = [vdupq_n_f32(0.0); PAIRS];
            let mut dot_sum_odd = vdupq_n_f32(0.0);
            for group in 0..4 {
                let (w_h0_lo, w_h0_hi) = pair_s8_lo_hi(t_r0[group * 2], t_r1[group * 2]);
                let (w_h1_lo, w_h1_hi) = pair_s8_lo_hi(t_r0[group * 2 + 1], t_r1[group * 2 + 1]);
                for (p, dot_sum) in dot_sum_pairs.iter_mut().enumerate() {
                    let act0 = &tokens[2 * p][block_idx * 4 + group];
                    let act1 = &tokens[2 * p + 1][block_idx * 4 + group];
                    let a0_h = act0.values().as_chunks::<16>().0;
                    let a1_h = act1.values().as_chunks::<16>().0;
                    let (a_h0_lo, a_h0_hi) =
                        pair_s8_lo_hi(load_s8x16(&a0_h[0]), load_s8x16(&a1_h[0]));
                    let (a_h1_lo, a_h1_hi) =
                        pair_s8_lo_hi(load_s8x16(&a0_h[1]), load_s8x16(&a1_h[1]));
                    let mut acc = smmla_s32(zero_s32, w_h0_lo, a_h0_lo);
                    acc = smmla_s32(acc, w_h0_hi, a_h0_hi);
                    acc = smmla_s32(acc, w_h1_lo, a_h1_lo);
                    acc = smmla_s32(acc, w_h1_hi, a_h1_hi);
                    let as_vec =
                        load_f32x4_neon(&[act0.scale(), act1.scale(), act0.scale(), act1.scale()]);
                    *dot_sum = vaddq_f32(*dot_sum, vmulq_f32(vcvtq_f32_s32(acc), as_vec));
                }
                if HAS_ODD {
                    let act_odd = &tokens[PAIRS * 2][block_idx * 4 + group];
                    let ao_h = act_odd.values().as_chunks::<16>().0;
                    let ao0 = load_s8x16(&ao_h[0]);
                    let ao1 = load_s8x16(&ao_h[1]);
                    let s0 = sdot32_vec(t_r0[group * 2], t_r0[group * 2 + 1], ao0, ao1);
                    let s1 = sdot32_vec(t_r1[group * 2], t_r1[group * 2 + 1], ao0, ao1);
                    let dots_01 = reduce4_s32x4_addp(s0, s1, zero_s32, zero_s32);
                    dot_sum_odd = vaddq_f32(
                        dot_sum_odd,
                        vmulq_f32(vcvtq_f32_s32(dots_01), vdupq_n_f32(act_odd.scale())),
                    );
                }
            }
            let rs0 = row_scales[r0];
            let rs1 = row_scales[r1];
            let rs_01 = load_f32x4_neon(&[rs0, rs0, rs1, rs1]);
            for (p, &dot_sum) in dot_sum_pairs.iter().enumerate() {
                let cur = load_f32x4_neon(&sums_pair[row_pair][p]);
                let next = vaddq_f32(cur, vmulq_f32(rs_01, dot_sum));
                store_f32x4_neon(next, &mut sums_pair[row_pair][p]);
            }
            if HAS_ODD {
                let rs_odd = load_f32x4_neon(&[rs0, rs1, 0.0, 0.0]);
                let cur = load_f32x4_neon(&sums_odd[row_pair]);
                let next = vaddq_f32(cur, vmulq_f32(rs_odd, dot_sum_odd));
                store_f32x4_neon(next, &mut sums_odd[row_pair]);
            }
        }
    }

    *out = [[0.0_f32; 8]; 8];
    for row_pair in 0..4 {
        let r0 = row_pair * 2;
        let r1 = r0 + 1;
        for (p, pair_vals) in sums_pair[row_pair].iter().enumerate() {
            out[r0][2 * p] = pair_vals[0];
            out[r0][2 * p + 1] = pair_vals[1];
            out[r1][2 * p] = pair_vals[2];
            out[r1][2 * p + 1] = pair_vals[3];
        }
        if HAS_ODD {
            out[r0][PAIRS * 2] = sums_odd[row_pair][0];
            out[r1][PAIRS * 2] = sums_odd[row_pair][1];
        }
    }
    out.iter().flatten().all(|v| v.is_finite()).then_some(())
}

#[target_feature(enable = "neon,dotprod")]
fn ptq1_scaled_8x_n_source_dotprod<const N_TOK: usize, A: super::Q8ActivationView>(
    rows: [&[u8]; 8],
    tokens: &[&[A]],
    out: &mut [[f32; 8]; 8],
) -> Option<()> {
    use core::arch::aarch64::{vaddq_f32, vcvtq_f32_s32, vdupq_n_f32, vmulq_f32};

    let row_chunks = rows.map(|row| row.as_chunks::<28>().0);
    let block_count = row_chunks[0].len();
    let mut sums = [[0.0_f32; 8]; 8];
    let zero_s32 = vdupq_n_s32(0);

    for block_idx in 0..block_count {
        let blocks: [&[u8; 28]; 8] = std::array::from_fn(|r| &row_chunks[r][block_idx]);
        let row_scales = ptq1_validate_row_scales(&blocks)?;
        let mut dot_sum_lo = [vdupq_n_f32(0.0); 8];
        let mut dot_sum_hi = [vdupq_n_f32(0.0); 8];
        for group in 0..4 {
            let a_scales: [f32; 8] = std::array::from_fn(|t| {
                if t < N_TOK {
                    tokens[t][block_idx * 4 + group].scale()
                } else {
                    0.0
                }
            });
            let acts: [(int8x16_t, int8x16_t); N_TOK] = std::array::from_fn(|t| {
                let ah = tokens[t][block_idx * 4 + group]
                    .values()
                    .as_chunks::<16>()
                    .0;
                (load_s8x16(&ah[0]), load_s8x16(&ah[1]))
            });
            let as_lo = load_f32x4_neon(&[a_scales[0], a_scales[1], a_scales[2], a_scales[3]]);
            let as_hi = load_f32x4_neon(&[a_scales[4], a_scales[5], a_scales[6], a_scales[7]]);
            for (r, &block) in blocks.iter().enumerate() {
                let (w0, w1) = unpack_ptq1_group(block, group);
                let l0 = sdot32_vec(w0, w1, acts[0].0, acts[0].1);
                let l1 = if N_TOK > 1 {
                    sdot32_vec(w0, w1, acts[1].0, acts[1].1)
                } else {
                    zero_s32
                };
                let l2 = if N_TOK > 2 {
                    sdot32_vec(w0, w1, acts[2].0, acts[2].1)
                } else {
                    zero_s32
                };
                let l3 = if N_TOK > 3 {
                    sdot32_vec(w0, w1, acts[3].0, acts[3].1)
                } else {
                    zero_s32
                };
                let d_lo = vcvtq_f32_s32(reduce4_s32x4_addp(l0, l1, l2, l3));
                dot_sum_lo[r] = vaddq_f32(dot_sum_lo[r], vmulq_f32(d_lo, as_lo));
                if N_TOK > 4 {
                    let l4 = sdot32_vec(w0, w1, acts[4].0, acts[4].1);
                    let l5 = if N_TOK > 5 {
                        sdot32_vec(w0, w1, acts[5].0, acts[5].1)
                    } else {
                        zero_s32
                    };
                    let l6 = if N_TOK > 6 {
                        sdot32_vec(w0, w1, acts[6].0, acts[6].1)
                    } else {
                        zero_s32
                    };
                    let l7 = if N_TOK > 7 {
                        sdot32_vec(w0, w1, acts[7].0, acts[7].1)
                    } else {
                        zero_s32
                    };
                    let d_hi = vcvtq_f32_s32(reduce4_s32x4_addp(l4, l5, l6, l7));
                    dot_sum_hi[r] = vaddq_f32(dot_sum_hi[r], vmulq_f32(d_hi, as_hi));
                }
            }
        }
        for (r, &rs) in row_scales.iter().enumerate() {
            let rs_vec = vdupq_n_f32(rs);
            let quads = sums[r].as_chunks_mut::<4>().0;
            let cur_lo = load_f32x4_neon(&quads[0]);
            let next_lo = vaddq_f32(cur_lo, vmulq_f32(rs_vec, dot_sum_lo[r]));
            store_f32x4_neon(next_lo, &mut quads[0]);
            if N_TOK > 4 {
                let cur_hi = load_f32x4_neon(&quads[1]);
                let next_hi = vaddq_f32(cur_hi, vmulq_f32(rs_vec, dot_sum_hi[r]));
                store_f32x4_neon(next_hi, &mut quads[1]);
            }
        }
    }
    *out = sums;
    for row in out.iter_mut() {
        row[N_TOK..].fill(0.0);
    }
    out.iter().flatten().all(|v| v.is_finite()).then_some(())
}

#[target_feature(enable = "neon,i8mm,dotprod")]
fn ptq1_scaled_8x_tokens_i8mm<const PINNED: bool, A: super::Q8ActivationView>(
    rows: [&[u8]; 8],
    tokens: &[&[A]],
    out: &mut [[f32; 8]; 8],
) -> Option<()> {
    if PINNED {
        return ptq1_scaled_8x_tokens_dotprod::<true, A>(rows, tokens, out);
    }
    match tokens.len() {
        1 => {
            let col = ptq1_scaled_rows_dotprod::<8, false, A>(rows, tokens[0])?;
            *out = [[0.0_f32; 8]; 8];
            for (dst, &val) in out.iter_mut().zip(col.iter()) {
                dst[0] = val;
            }
            Some(())
        }
        2 => ptq1_scaled_8x_pairs_source_i8mm::<1, false, A>(rows, tokens, out),
        3 => ptq1_scaled_8x_pairs_source_i8mm::<1, true, A>(rows, tokens, out),
        4 => ptq1_scaled_8x_pairs_source_i8mm::<2, false, A>(rows, tokens, out),
        5 => ptq1_scaled_8x_pairs_source_i8mm::<2, true, A>(rows, tokens, out),
        6 => ptq1_scaled_8x_pairs_source_i8mm::<3, false, A>(rows, tokens, out),
        7 => ptq1_scaled_8x_pairs_source_i8mm::<3, true, A>(rows, tokens, out),
        8 => ptq1_scaled_8x_pairs_source_i8mm::<4, false, A>(rows, tokens, out),
        _ => None,
    }
}

#[target_feature(enable = "neon,dotprod")]
fn ptq1_scaled_8x_tokens_dotprod<const PINNED: bool, A: super::Q8ActivationView>(
    rows: [&[u8]; 8],
    tokens: &[&[A]],
    out: &mut [[f32; 8]; 8],
) -> Option<()> {
    if tokens.len() == 1 {
        let col = ptq1_scaled_rows_dotprod::<8, PINNED, A>(rows, tokens[0])?;
        *out = [[0.0_f32; 8]; 8];
        for (dst, &val) in out.iter_mut().zip(col.iter()) {
            dst[0] = val;
        }
        return Some(());
    }
    if PINNED {
        match tokens.len() {
            2 => ptq1_scaled_8x_2_pinned_dotprod(rows, tokens, out),
            3 => ptq1_scaled_8x_n_pinned_dotprod::<3, A>(rows, tokens, out),
            4 => ptq1_scaled_8x_n_pinned_dotprod::<4, A>(rows, tokens, out),
            5 => ptq1_scaled_8x_n_pinned_dotprod::<5, A>(rows, tokens, out),
            6 => ptq1_scaled_8x_n_pinned_dotprod::<6, A>(rows, tokens, out),
            7 => ptq1_scaled_8x_n_pinned_dotprod::<7, A>(rows, tokens, out),
            8 => ptq1_scaled_8x_n_pinned_dotprod::<8, A>(rows, tokens, out),
            _ => None,
        }
    } else {
        match tokens.len() {
            2 => ptq1_scaled_8x_n_source_dotprod::<2, A>(rows, tokens, out),
            3 => ptq1_scaled_8x_n_source_dotprod::<3, A>(rows, tokens, out),
            4 => ptq1_scaled_8x_n_source_dotprod::<4, A>(rows, tokens, out),
            5 => ptq1_scaled_8x_n_source_dotprod::<5, A>(rows, tokens, out),
            6 => ptq1_scaled_8x_n_source_dotprod::<6, A>(rows, tokens, out),
            7 => ptq1_scaled_8x_n_source_dotprod::<7, A>(rows, tokens, out),
            8 => ptq1_scaled_8x_n_source_dotprod::<8, A>(rows, tokens, out),
            _ => None,
        }
    }
}

#[inline]
pub(super) fn ptq1_detected_dot_scaled_rows<
    const ROWS: usize,
    const PINNED: bool,
    A: super::Q8ActivationView,
>(
    _detected: super::Ptq1Dotprod,
    rows: [&[u8]; ROWS],
    activation: &[A],
) -> Option<[f32; ROWS]> {
    // SAFETY: constructing `Ptq1Dotprod` verified CPU/OS `dotprod` and baseline `neon`;
    // the safe parent checked row extents, group count, and finite activation scales.
    unsafe { ptq1_scaled_rows_dotprod::<ROWS, PINNED, A>(rows, activation) }
}

#[inline]
pub(super) fn ptq1_detected_dot_packed_rows<const ROWS: usize, const PINNED: bool>(
    detected: super::Ptq1Dotprod,
    rows: [&[u8]; ROWS],
    activation: &[super::Ptq1Activation],
) -> Option<[f32; ROWS]> {
    // SAFETY: constructing `Ptq1Dotprod` verified CPU/OS `dotprod` and baseline `neon`;
    // the safe parent checked row extents, block count, and finite activation scales.
    unsafe { ptq1_packed_rows_dotprod::<ROWS, PINNED>(rows, activation, detected.sve) }
}

#[inline]
pub(super) fn ptq1_detected_dot_scaled_8x_tokens<const PINNED: bool, A: super::Q8ActivationView>(
    detected: super::Ptq1Dotprod,
    rows: [&[u8]; 8],
    tokens: &[&[A]],
    out: &mut [[f32; 8]; 8],
) -> Option<()> {
    if detected.i8mm && !PINNED {
        // SAFETY: `detected.i8mm` was established by `has_dotprod() && has_i8mm()`,
        // and the safe parent checked token/row extents and finite activation scales.
        unsafe { ptq1_scaled_8x_tokens_i8mm::<PINNED, A>(rows, tokens, out) }
    } else {
        // SAFETY: constructing `Ptq1Dotprod` verified CPU/OS `dotprod` and baseline `neon`,
        // and the safe parent checked all slice invariants.
        unsafe { ptq1_scaled_8x_tokens_dotprod::<PINNED, A>(rows, tokens, out) }
    }
}

#[inline]
fn prefetch_q8_quad_window<A: super::Q8ActivationView>(tok_quad: [&[A]; 4], act_idx: usize) {
    if act_idx < tok_quad[0].len() {
        for tok in tok_quad {
            prefetch_l1(tok[act_idx].values().as_ptr().cast::<u8>());
        }
    }
    if act_idx + 2 < tok_quad[0].len() {
        for tok in tok_quad {
            prefetch_l1(tok[act_idx + 2].values().as_ptr().cast::<u8>());
        }
    }
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
fn ptq1_pinned_quad_block<A: super::Q8ActivationView>(
    block_trits: &[[int8x16_t; 8]; 4],
    rs: &[f32; 4],
    tok_quad: [&[A]; 4],
    act_base: usize,
    acc: &mut [[core::arch::aarch64::float32x4_t; 4]; 4],
) {
    use core::arch::aarch64::{vcvtq_f32_s32, vdupq_n_f32, vfmaq_laneq_f32, vmulq_f32};

    for group in 0..4 {
        let act_idx = act_base + group;
        let a_scales = load_f32x4_neon(&[
            tok_quad[0][act_idx].scale(),
            tok_quad[1][act_idx].scale(),
            tok_quad[2][act_idx].scale(),
            tok_quad[3][act_idx].scale(),
        ]);
        let a_pairs: [(int8x16_t, int8x16_t); 4] = std::array::from_fn(|t| {
            let ah = tok_quad[t][act_idx].values().as_chunks::<16>().0;
            (load_s8x16(&ah[0]), load_s8x16(&ah[1]))
        });
        for (acc_row, (&row_rs, row_trits)) in acc.iter_mut().zip(rs.iter().zip(block_trits.iter()))
        {
            let rs_a = vmulq_f32(a_scales, vdupq_n_f32(row_rs));
            let w0 = row_trits[group * 2];
            let w1 = row_trits[group * 2 + 1];
            let s0 = sdot32_vec(w0, w1, a_pairs[0].0, a_pairs[0].1);
            let s1 = sdot32_vec(w0, w1, a_pairs[1].0, a_pairs[1].1);
            let s2 = sdot32_vec(w0, w1, a_pairs[2].0, a_pairs[2].1);
            let s3 = sdot32_vec(w0, w1, a_pairs[3].0, a_pairs[3].1);
            acc_row[0] = vfmaq_laneq_f32::<0>(acc_row[0], vcvtq_f32_s32(s0), rs_a);
            acc_row[1] = vfmaq_laneq_f32::<1>(acc_row[1], vcvtq_f32_s32(s1), rs_a);
            acc_row[2] = vfmaq_laneq_f32::<2>(acc_row[2], vcvtq_f32_s32(s2), rs_a);
            acc_row[3] = vfmaq_laneq_f32::<3>(acc_row[3], vcvtq_f32_s32(s3), rs_a);
        }
    }
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
fn ptq1_pinned_quad_k_tile<A: super::Q8ActivationView>(
    half_trits: &[[[int8x16_t; 8]; 4]],
    half_scales: &[[f32; 4]],
    tok_quad: [&[A]; 4],
    k_base: usize,
    mut acc: [[core::arch::aarch64::float32x4_t; 4]; 4],
) -> [[core::arch::aarch64::float32x4_t; 4]; 4] {
    prefetch_q8_quad_window(tok_quad, k_base * 4);
    let (trit_pairs, trit_rem) = half_trits.as_chunks::<2>();
    let (scale_pairs, scale_rem) = half_scales.as_chunks::<2>();

    for (pair_idx, (trit_pair, scale_pair)) in trit_pairs.iter().zip(scale_pairs).enumerate() {
        let act_base = (k_base + pair_idx * 2) * 4;
        prefetch_q8_quad_window(tok_quad, act_base + 4);
        prefetch_q8_quad_window(tok_quad, act_base + 8);
        ptq1_pinned_quad_block(&trit_pair[0], &scale_pair[0], tok_quad, act_base, &mut acc);
        ptq1_pinned_quad_block(
            &trit_pair[1],
            &scale_pair[1],
            tok_quad,
            act_base + 4,
            &mut acc,
        );
    }

    let rem_base = (k_base + trit_pairs.len() * 2) * 4;
    for (rem_off, (block_trits, rs)) in trit_rem.iter().zip(scale_rem).enumerate() {
        let act_base = rem_base + rem_off * 4;
        prefetch_q8_quad_window(tok_quad, act_base + 4);
        ptq1_pinned_quad_block(block_trits, rs, tok_quad, act_base, &mut acc);
    }
    acc
}

#[target_feature(enable = "neon,dotprod")]
fn ptq1_scaled_8x8_chunks_pinned_dotprod<const CHUNKS: usize, A: super::Q8ActivationView>(
    rows: [&[u8]; 8],
    chunks: &[[&[A]; 8]],
    outputs: &mut [[[f32; 8]; 8]],
) -> Option<()> {
    use core::arch::aarch64::{vdupq_n_f32, vdupq_n_s8};

    const K_TILE: usize = 8;
    let row_chunks = rows.map(|row| row.as_chunks::<28>().0);
    let block_count = row_chunks[0].len();
    let mut lanes = [[[[0.0_f32; 4]; 8]; 8]; CHUNKS];
    let mut out_buf = [[[0.0_f32; 8]; 8]; CHUNKS];

    for k_base in (0..block_count).step_by(K_TILE) {
        let tile_len = (block_count - k_base).min(K_TILE);
        let is_first = k_base == 0;
        let is_last = k_base + tile_len == block_count;

        let mut scales_lo = [[0.0_f32; 4]; K_TILE];
        let mut scales_hi = [[0.0_f32; 4]; K_TILE];
        for k_off in 0..tile_len {
            let block_idx = k_base + k_off;
            prefetch_row_blocks(&row_chunks, block_idx, block_count);
            let blocks: [&[u8; 28]; 8] = std::array::from_fn(|r| &row_chunks[r][block_idx]);
            let row_scales = ptq1_validate_row_scales(&blocks)?;
            scales_lo[k_off] = [row_scales[0], row_scales[1], row_scales[2], row_scales[3]];
            scales_hi[k_off] = [row_scales[4], row_scales[5], row_scales[6], row_scales[7]];
        }

        for row_half in 0..2 {
            let rbase = row_half * 4;
            let half_scales = if row_half == 0 {
                &scales_lo[..tile_len]
            } else {
                &scales_hi[..tile_len]
            };
            let mut half_trits = [[[vdupq_n_s8(0); 8]; 4]; K_TILE];
            for (k_off, dst_block) in half_trits.iter_mut().take(tile_len).enumerate() {
                let block_idx = k_base + k_off;
                for (r, dst_row) in dst_block.iter_mut().enumerate() {
                    *dst_row = unpack_ptq1_vectors(&row_chunks[rbase + r][block_idx]);
                }
            }

            for (chunk_idx, chunk) in chunks.iter().take(CHUNKS).enumerate() {
                for tok_half in 0..2 {
                    let tbase = tok_half * 4;
                    let tok_quad = [
                        chunk[tbase],
                        chunk[tbase + 1],
                        chunk[tbase + 2],
                        chunk[tbase + 3],
                    ];
                    let acc_init = if is_first {
                        [[vdupq_n_f32(0.0); 4]; 4]
                    } else {
                        std::array::from_fn(|r| {
                            std::array::from_fn(|t| {
                                load_f32x4_neon(&lanes[chunk_idx][rbase + r][tbase + t])
                            })
                        })
                    };
                    let acc = ptq1_pinned_quad_k_tile(
                        &half_trits[..tile_len],
                        half_scales,
                        tok_quad,
                        k_base,
                        acc_init,
                    );
                    if is_last {
                        for r in 0..4 {
                            for t in 0..4 {
                                let mut l = [0.0_f32; 4];
                                store_f32x4_neon(acc[r][t], &mut l);
                                let val = (l[0] + l[1]) + (l[2] + l[3]);
                                if !val.is_finite() {
                                    return None;
                                }
                                out_buf[chunk_idx][rbase + r][tbase + t] = val;
                            }
                        }
                    } else {
                        for r in 0..4 {
                            for t in 0..4 {
                                store_f32x4_neon(
                                    acc[r][t],
                                    &mut lanes[chunk_idx][rbase + r][tbase + t],
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    for (dst, src) in outputs.iter_mut().zip(out_buf.iter()) {
        *dst = *src;
    }
    Some(())
}

#[target_feature(enable = "neon,i8mm")]
fn ptq1_scaled_8x8_chunks_source_i8mm<const CHUNKS: usize, A: super::Q8ActivationView>(
    rows: [&[u8]; 8],
    chunks: &[[&[A]; 8]],
    outputs: &mut [[[f32; 8]; 8]],
) -> Option<()> {
    use core::arch::aarch64::{vaddq_f32, vcvtq_f32_s32, vdupq_n_f32, vmulq_f32};

    let row_chunks = rows.map(|row| row.as_chunks::<28>().0);
    let block_count = row_chunks[0].len();
    let mut sums_pair = [[[[0.0_f32; 4]; 4]; CHUNKS]; 4];
    let zero_s32 = vdupq_n_s32(0);

    for block_idx in 0..block_count {
        prefetch_row_blocks(&row_chunks, block_idx, block_count);
        let blocks: [&[u8; 28]; 8] = std::array::from_fn(|r| &row_chunks[r][block_idx]);
        let row_scales = ptq1_validate_row_scales(&blocks)?;
        let act_base = block_idx * 4;
        for ((&[block0, block1], &[rs0, rs1]), chunk_sums) in blocks
            .as_chunks::<2>()
            .0
            .iter()
            .zip(row_scales.as_chunks::<2>().0)
            .zip(sums_pair.iter_mut())
        {
            let t_r0 = unpack_ptq1_vectors(block0);
            let t_r1 = unpack_ptq1_vectors(block1);
            let w_groups: [(int8x16_t, int8x16_t, int8x16_t, int8x16_t); 4] =
                std::array::from_fn(|group| {
                    let (w_h0_lo, w_h0_hi) = pair_s8_lo_hi(t_r0[group * 2], t_r1[group * 2]);
                    let (w_h1_lo, w_h1_hi) =
                        pair_s8_lo_hi(t_r0[group * 2 + 1], t_r1[group * 2 + 1]);
                    (w_h0_lo, w_h0_hi, w_h1_lo, w_h1_hi)
                });
            let rs_01 = load_f32x4_neon(&[rs0, rs0, rs1, rs1]);
            for (chunk, pair_sums) in chunks.iter().take(CHUNKS).zip(chunk_sums.iter_mut()) {
                let mut dot_sum_pairs = [vdupq_n_f32(0.0); 4];
                for (group, &(w_h0_lo, w_h0_hi, w_h1_lo, w_h1_hi)) in w_groups.iter().enumerate() {
                    let act_idx = act_base + group;
                    for (p, dot_sum) in dot_sum_pairs.iter_mut().enumerate() {
                        let act0 = &chunk[2 * p][act_idx];
                        let act1 = &chunk[2 * p + 1][act_idx];
                        let a0_h = act0.values().as_chunks::<16>().0;
                        let a1_h = act1.values().as_chunks::<16>().0;
                        let (a_h0_lo, a_h0_hi) =
                            pair_s8_lo_hi(load_s8x16(&a0_h[0]), load_s8x16(&a1_h[0]));
                        let (a_h1_lo, a_h1_hi) =
                            pair_s8_lo_hi(load_s8x16(&a0_h[1]), load_s8x16(&a1_h[1]));
                        let mut acc = smmla_s32(zero_s32, w_h0_lo, a_h0_lo);
                        acc = smmla_s32(acc, w_h0_hi, a_h0_hi);
                        acc = smmla_s32(acc, w_h1_lo, a_h1_lo);
                        acc = smmla_s32(acc, w_h1_hi, a_h1_hi);
                        let as_vec = load_f32x4_neon(&[
                            act0.scale(),
                            act1.scale(),
                            act0.scale(),
                            act1.scale(),
                        ]);
                        *dot_sum = vaddq_f32(*dot_sum, vmulq_f32(vcvtq_f32_s32(acc), as_vec));
                    }
                }
                for (&dot_sum, slot) in dot_sum_pairs.iter().zip(pair_sums.iter_mut()) {
                    let cur = load_f32x4_neon(slot);
                    let next = vaddq_f32(cur, vmulq_f32(rs_01, dot_sum));
                    store_f32x4_neon(next, slot);
                }
            }
        }
    }

    let mut out_buf = [[[0.0_f32; 8]; 8]; CHUNKS];
    for (row_pair, pair_chunks) in sums_pair.iter().enumerate() {
        let r0 = row_pair * 2;
        let r1 = r0 + 1;
        for (chunk_arr, chunk_out) in pair_chunks.iter().zip(out_buf.iter_mut()) {
            for (p, vals) in chunk_arr.iter().enumerate() {
                chunk_out[r0][2 * p] = vals[0];
                chunk_out[r0][2 * p + 1] = vals[1];
                chunk_out[r1][2 * p] = vals[2];
                chunk_out[r1][2 * p + 1] = vals[3];
            }
        }
    }
    if !out_buf.iter().flatten().flatten().all(|v| v.is_finite()) {
        return None;
    }
    for (dst, src) in outputs.iter_mut().zip(out_buf.iter()) {
        *dst = *src;
    }
    Some(())
}

#[target_feature(enable = "neon,dotprod")]
fn ptq1_scaled_8x8_chunks_source_dotprod<const CHUNKS: usize, A: super::Q8ActivationView>(
    rows: [&[u8]; 8],
    chunks: &[[&[A]; 8]],
    outputs: &mut [[[f32; 8]; 8]],
) -> Option<()> {
    use core::arch::aarch64::{vaddq_f32, vcvtq_f32_s32, vdupq_n_f32, vmulq_f32};

    let row_chunks = rows.map(|row| row.as_chunks::<28>().0);
    let block_count = row_chunks[0].len();
    let mut sums = [[[0.0_f32; 8]; 8]; CHUNKS];

    for block_idx in 0..block_count {
        prefetch_row_blocks(&row_chunks, block_idx, block_count);
        let blocks: [&[u8; 28]; 8] = std::array::from_fn(|r| &row_chunks[r][block_idx]);
        let row_scales = ptq1_validate_row_scales(&blocks)?;
        let act_base = block_idx * 4;
        for (r, (&block, &rs)) in blocks.iter().zip(row_scales.iter()).enumerate() {
            let [t0, t1, t2, t3, t4, t5, t6, t7] = unpack_ptq1_vectors(block);
            let pairs = [(t0, t1), (t2, t3), (t4, t5), (t6, t7)];
            let rs_vec = vdupq_n_f32(rs);
            for (c, chunk) in chunks.iter().take(CHUNKS).enumerate() {
                let mut dot_sum_lo = vdupq_n_f32(0.0);
                let mut dot_sum_hi = vdupq_n_f32(0.0);
                for (group, &(w0, w1)) in pairs.iter().enumerate() {
                    let act_idx = act_base + group;
                    let acts: [&A; 8] = std::array::from_fn(|t| &chunk[t][act_idx]);
                    let as_lo = load_f32x4_neon(&[
                        acts[0].scale(),
                        acts[1].scale(),
                        acts[2].scale(),
                        acts[3].scale(),
                    ]);
                    let as_hi = load_f32x4_neon(&[
                        acts[4].scale(),
                        acts[5].scale(),
                        acts[6].scale(),
                        acts[7].scale(),
                    ]);
                    let a_pairs: [(int8x16_t, int8x16_t); 8] = std::array::from_fn(|t| {
                        let ah = acts[t].values().as_chunks::<16>().0;
                        (load_s8x16(&ah[0]), load_s8x16(&ah[1]))
                    });
                    let l0 = sdot32_vec(w0, w1, a_pairs[0].0, a_pairs[0].1);
                    let l1 = sdot32_vec(w0, w1, a_pairs[1].0, a_pairs[1].1);
                    let l2 = sdot32_vec(w0, w1, a_pairs[2].0, a_pairs[2].1);
                    let l3 = sdot32_vec(w0, w1, a_pairs[3].0, a_pairs[3].1);
                    let d_lo = vcvtq_f32_s32(reduce4_s32x4_addp(l0, l1, l2, l3));
                    dot_sum_lo = vaddq_f32(dot_sum_lo, vmulq_f32(d_lo, as_lo));

                    let l4 = sdot32_vec(w0, w1, a_pairs[4].0, a_pairs[4].1);
                    let l5 = sdot32_vec(w0, w1, a_pairs[5].0, a_pairs[5].1);
                    let l6 = sdot32_vec(w0, w1, a_pairs[6].0, a_pairs[6].1);
                    let l7 = sdot32_vec(w0, w1, a_pairs[7].0, a_pairs[7].1);
                    let d_hi = vcvtq_f32_s32(reduce4_s32x4_addp(l4, l5, l6, l7));
                    dot_sum_hi = vaddq_f32(dot_sum_hi, vmulq_f32(d_hi, as_hi));
                }
                let quads = sums[c][r].as_chunks_mut::<4>().0;
                let cur_lo = load_f32x4_neon(&quads[0]);
                let next_lo = vaddq_f32(cur_lo, vmulq_f32(rs_vec, dot_sum_lo));
                store_f32x4_neon(next_lo, &mut quads[0]);
                let cur_hi = load_f32x4_neon(&quads[1]);
                let next_hi = vaddq_f32(cur_hi, vmulq_f32(rs_vec, dot_sum_hi));
                store_f32x4_neon(next_hi, &mut quads[1]);
            }
        }
    }

    if !sums.iter().flatten().flatten().all(|v| v.is_finite()) {
        return None;
    }
    for (dst, src) in outputs.iter_mut().zip(sums.iter()) {
        *dst = *src;
    }
    Some(())
}

#[inline]
#[target_feature(enable = "neon")]
fn pack_pinned_lane_pair(
    lo0: int8x16_t,
    hi0: int8x16_t,
    lo1: int8x16_t,
    hi1: int8x16_t,
) -> [int8x16_t; 4] {
    use core::arch::aarch64::{
        vreinterpretq_s8_s64, vreinterpretq_s32_s8, vreinterpretq_s64_s32, vzip1q_s32, vzip1q_s64,
        vzip2q_s32, vzip2q_s64,
    };
    let l0 = vreinterpretq_s32_s8(lo0);
    let h0 = vreinterpretq_s32_s8(hi0);
    let l1 = vreinterpretq_s32_s8(lo1);
    let h1 = vreinterpretq_s32_s8(hi1);
    let p0_01 = vreinterpretq_s64_s32(vzip1q_s32(l0, h0));
    let p0_23 = vreinterpretq_s64_s32(vzip2q_s32(l0, h0));
    let p1_01 = vreinterpretq_s64_s32(vzip1q_s32(l1, h1));
    let p1_23 = vreinterpretq_s64_s32(vzip2q_s32(l1, h1));
    [
        vreinterpretq_s8_s64(vzip1q_s64(p0_01, p1_01)),
        vreinterpretq_s8_s64(vzip2q_s64(p0_01, p1_01)),
        vreinterpretq_s8_s64(vzip1q_s64(p0_23, p1_23)),
        vreinterpretq_s8_s64(vzip2q_s64(p0_23, p1_23)),
    ]
}

#[inline]
#[target_feature(enable = "neon")]
fn transpose_4x4_s32(d0: int32x4_t, d1: int32x4_t, d2: int32x4_t, d3: int32x4_t) -> [int32x4_t; 4] {
    use core::arch::aarch64::{
        vreinterpretq_s32_s64, vreinterpretq_s64_s32, vtrn1q_s32, vtrn1q_s64, vtrn2q_s32,
        vtrn2q_s64,
    };
    let t01_lo = vreinterpretq_s64_s32(vtrn1q_s32(d0, d1));
    let t01_hi = vreinterpretq_s64_s32(vtrn2q_s32(d0, d1));
    let t23_lo = vreinterpretq_s64_s32(vtrn1q_s32(d2, d3));
    let t23_hi = vreinterpretq_s64_s32(vtrn2q_s32(d2, d3));
    [
        vreinterpretq_s32_s64(vtrn1q_s64(t01_lo, t23_lo)),
        vreinterpretq_s32_s64(vtrn1q_s64(t01_hi, t23_hi)),
        vreinterpretq_s32_s64(vtrn2q_s64(t01_lo, t23_lo)),
        vreinterpretq_s32_s64(vtrn2q_s64(t01_hi, t23_hi)),
    ]
}

#[inline]
#[target_feature(enable = "neon")]
fn pack_ptq1_pinned_i8mm_pair_block<A: super::Q8ActivationView>(
    b0: &A,
    b1: &A,
) -> super::Ptq1PinnedI8mmPairBlock {
    let h0 = b0.values().as_chunks::<16>().0;
    let h1 = b1.values().as_chunks::<16>().0;
    let packed = pack_pinned_lane_pair(
        load_s8x16(&h0[0]),
        load_s8x16(&h0[1]),
        load_s8x16(&h1[0]),
        load_s8x16(&h1[1]),
    );
    let mut planes = [[0_i8; 16]; 4];
    for (vec, dst) in packed.into_iter().zip(&mut planes) {
        store_s8x16(vec, dst);
    }
    super::Ptq1PinnedI8mmPairBlock {
        planes,
        scales: [b0.scale(), b1.scale()],
    }
}

#[inline]
pub(super) fn ptq1_append_pinned_i8mm_pair_blocks<A: super::Q8ActivationView>(
    t0: &[A],
    t1: &[A],
    out: &mut Vec<super::Ptq1PinnedI8mmPairBlock>,
) {
    // SAFETY: Baseline NEON is mandatory on `aarch64` targets and caller verified matching extents.
    unsafe { append_ptq1_pinned_i8mm_pair_blocks(t0, t1, out) };
}

#[inline]
pub(super) fn ptq1_pack_pinned_i8mm_pair_blocks_into<A: super::Q8ActivationView>(
    t0: &[A],
    t1: &[A],
    out: &mut [super::Ptq1PinnedI8mmPairBlock],
) {
    // SAFETY: Baseline NEON is mandatory on `aarch64` targets and caller verified matching extents.
    unsafe { pack_ptq1_pinned_i8mm_pair_blocks_into_neon(t0, t1, out) };
}

#[target_feature(enable = "neon")]
fn append_ptq1_pinned_i8mm_pair_blocks<A: super::Q8ActivationView>(
    t0: &[A],
    t1: &[A],
    out: &mut Vec<super::Ptq1PinnedI8mmPairBlock>,
) {
    out.reserve(t0.len());
    for (b0, b1) in t0.iter().zip(t1.iter()) {
        out.push(pack_ptq1_pinned_i8mm_pair_block(b0, b1));
    }
}

#[target_feature(enable = "neon")]
fn pack_ptq1_pinned_i8mm_pair_blocks_into_neon<A: super::Q8ActivationView>(
    t0: &[A],
    t1: &[A],
    out: &mut [super::Ptq1PinnedI8mmPairBlock],
) {
    for (dst, (b0, b1)) in out.iter_mut().zip(t0.iter().zip(t1.iter())) {
        *dst = pack_ptq1_pinned_i8mm_pair_block(b0, b1);
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "K-tiled PTQ1 pinned I8MM kernel hoists 2-row weight unpacking across chunks and token pairs with 16 register accumulators."
)]
#[target_feature(enable = "neon,i8mm")]
fn ptq1_scaled_8x8_chunks_pinned_panels_i8mm<const CHUNKS: usize>(
    rows: [&[u8]; 8],
    panels: &[[&[super::Ptq1PinnedI8mmPairBlock]; 4]],
    outputs: &mut [[[f32; 8]; 8]],
) -> Option<()> {
    use core::arch::aarch64::{
        vandq_u16, vcvt_f32_f16, vcvt_high_f32_f16, vcvtq_f32_s32, vdupq_n_f32, vdupq_n_s8,
        vdupq_n_u16, vfmaq_laneq_f32, vget_low_f16, vld1q_u16, vmaxq_u16, vmaxvq_u16, vmulq_f32,
        vreinterpretq_f16_u16, vzip1q_f32, vzip2q_f32,
    };

    const K_TILE: usize = 8;
    let row_chunks = rows.map(|row| row.as_chunks::<28>().0);
    let block_count = row_chunks[0].len();
    let mut lanes = [[[[0.0_f32; 4]; 8]; 8]; CHUNKS];
    let zero_s32 = vdupq_n_s32(0);
    let mask_7fff = vdupq_n_u16(0x7fff);
    let mut umax_scales = vdupq_n_u16(0);

    for k_base in (0..block_count).step_by(K_TILE) {
        let tile_len = (block_count - k_base).min(K_TILE);
        let is_first = k_base == 0;
        let is_last = k_base + tile_len == block_count;

        let mut tile_rs = [[vdupq_n_f32(0.0); 4]; K_TILE];
        for (k_off, rs_slot) in tile_rs.iter_mut().enumerate().take(tile_len) {
            let block_idx = k_base + k_off;
            prefetch_row_blocks(&row_chunks, block_idx, block_count);
            let bits: [u16; 8] = std::array::from_fn(|r| ptq1_scale_u16(&row_chunks[r][block_idx]));
            // SAFETY: `bits` is an initialized eight-element `u16` array.
            let raw = unsafe { vld1q_u16(bits.as_ptr()) };
            umax_scales = vmaxq_u16(umax_scales, vandq_u16(raw, mask_7fff));
            let half = vreinterpretq_f16_u16(raw);
            let f_lo = vcvt_f32_f16(vget_low_f16(half));
            let f_hi = vcvt_high_f32_f16(half);
            *rs_slot = [
                vzip1q_f32(f_lo, f_lo),
                vzip2q_f32(f_lo, f_lo),
                vzip1q_f32(f_hi, f_hi),
                vzip2q_f32(f_hi, f_hi),
            ];
        }
        if is_last && vmaxvq_u16(umax_scales) >= 0x7c00 {
            return None;
        }

        for (rp, row_pair) in row_chunks.as_chunks::<2>().0.iter().enumerate() {
            let r0 = rp * 2;
            let mut tile_w = [[[vdupq_n_s8(0); 4]; 4]; K_TILE];
            for (k_off, w_block) in tile_w.iter_mut().enumerate().take(tile_len) {
                let block_idx = k_base + k_off;
                let t_r0 = unpack_ptq1_vectors(&row_pair[0][block_idx]);
                let t_r1 = unpack_ptq1_vectors(&row_pair[1][block_idx]);
                for (group, group_w) in w_block.iter_mut().enumerate() {
                    *group_w = pack_pinned_lane_pair(
                        t_r0[group * 2],
                        t_r0[group * 2 + 1],
                        t_r1[group * 2],
                        t_r1[group * 2 + 1],
                    );
                }
            }

            for (c, chunk_panels) in panels.iter().take(CHUNKS).enumerate() {
                let mut acc: [[core::arch::aarch64::float32x4_t; 4]; 4] = if is_first {
                    [[vdupq_n_f32(0.0); 4]; 4]
                } else {
                    std::array::from_fn(|p| {
                        [
                            load_f32x4_neon(&lanes[c][r0][2 * p]),
                            load_f32x4_neon(&lanes[c][r0][2 * p + 1]),
                            load_f32x4_neon(&lanes[c][r0 + 1][2 * p]),
                            load_f32x4_neon(&lanes[c][r0 + 1][2 * p + 1]),
                        ]
                    })
                };
                for (k_off, (rs_quad, w_block)) in
                    tile_rs.iter().zip(tile_w.iter()).enumerate().take(tile_len)
                {
                    let act_base = (k_base + k_off) * 4;
                    if k_off + 1 < tile_len {
                        for pair_slice in chunk_panels {
                            prefetch_l1(pair_slice[act_base + 4].planes.as_ptr().cast::<u8>());
                        }
                    }
                    let rs = rs_quad[rp];
                    for (group, &w) in w_block.iter().enumerate() {
                        for (p, pair_slice) in chunk_panels.iter().enumerate() {
                            let panel = &pair_slice[act_base + group];
                            let d0 = smmla_s32(zero_s32, w[0], load_s8x16(&panel.planes[0]));
                            let d1 = smmla_s32(zero_s32, w[1], load_s8x16(&panel.planes[1]));
                            let d2 = smmla_s32(zero_s32, w[2], load_s8x16(&panel.planes[2]));
                            let d3 = smmla_s32(zero_s32, w[3], load_s8x16(&panel.planes[3]));
                            let [r0t0, r0t1, r1t0, r1t1] = transpose_4x4_s32(d0, d1, d2, d3);
                            let as_vec = load_f32x4_neon(&[
                                panel.scales[0],
                                panel.scales[1],
                                panel.scales[0],
                                panel.scales[1],
                            ]);
                            let d = vmulq_f32(rs, as_vec);
                            acc[p][0] = vfmaq_laneq_f32::<0>(acc[p][0], vcvtq_f32_s32(r0t0), d);
                            acc[p][1] = vfmaq_laneq_f32::<1>(acc[p][1], vcvtq_f32_s32(r0t1), d);
                            acc[p][2] = vfmaq_laneq_f32::<2>(acc[p][2], vcvtq_f32_s32(r1t0), d);
                            acc[p][3] = vfmaq_laneq_f32::<3>(acc[p][3], vcvtq_f32_s32(r1t1), d);
                        }
                    }
                }
                if is_last {
                    for (p, pair_acc) in acc.iter().enumerate() {
                        for (cell, &lane_vec) in pair_acc.iter().enumerate() {
                            let r = r0 + cell / 2;
                            let t = 2 * p + (cell % 2);
                            let mut l = [0.0_f32; 4];
                            store_f32x4_neon(lane_vec, &mut l);
                            let val = (l[0] + l[1]) + (l[2] + l[3]);
                            if !val.is_finite() {
                                return None;
                            }
                            outputs[c][r][t] = val;
                        }
                    }
                } else {
                    for (p, pair_acc) in acc.iter().enumerate() {
                        store_f32x4_neon(pair_acc[0], &mut lanes[c][r0][2 * p]);
                        store_f32x4_neon(pair_acc[1], &mut lanes[c][r0][2 * p + 1]);
                        store_f32x4_neon(pair_acc[2], &mut lanes[c][r0 + 1][2 * p]);
                        store_f32x4_neon(pair_acc[3], &mut lanes[c][r0 + 1][2 * p + 1]);
                    }
                }
            }
        }
    }

    Some(())
}

#[inline]
pub(super) fn ptq1_detected_dot_scaled_8x8_chunks_with_pinned_panels(
    detected: super::Ptq1Dotprod,
    rows: [&[u8]; 8],
    panels: &[[&[super::Ptq1PinnedI8mmPairBlock]; 4]],
    outputs: &mut [[[f32; 8]; 8]],
) -> Option<()> {
    if !detected.i8mm {
        return None;
    }
    match panels.len() {
        // SAFETY: `detected.i8mm` was established by `has_dotprod() && has_i8mm()`,
        // and the safe parent checked chunk/panel/row extents and finite activation scales.
        1 => unsafe { ptq1_scaled_8x8_chunks_pinned_panels_i8mm::<1>(rows, panels, outputs) },
        // SAFETY: see arm `1` above.
        2 => unsafe { ptq1_scaled_8x8_chunks_pinned_panels_i8mm::<2>(rows, panels, outputs) },
        // SAFETY: see arm `1` above.
        3 => unsafe { ptq1_scaled_8x8_chunks_pinned_panels_i8mm::<3>(rows, panels, outputs) },
        // SAFETY: see arm `1` above.
        4 => unsafe { ptq1_scaled_8x8_chunks_pinned_panels_i8mm::<4>(rows, panels, outputs) },
        // SAFETY: see arm `1` above.
        5 => unsafe { ptq1_scaled_8x8_chunks_pinned_panels_i8mm::<5>(rows, panels, outputs) },
        // SAFETY: see arm `1` above.
        6 => unsafe { ptq1_scaled_8x8_chunks_pinned_panels_i8mm::<6>(rows, panels, outputs) },
        // SAFETY: see arm `1` above.
        7 => unsafe { ptq1_scaled_8x8_chunks_pinned_panels_i8mm::<7>(rows, panels, outputs) },
        // SAFETY: see arm `1` above.
        8 => unsafe { ptq1_scaled_8x8_chunks_pinned_panels_i8mm::<8>(rows, panels, outputs) },
        _ => None,
    }
}

#[inline]
pub(super) fn ptq1_detected_dot_scaled_8x8_chunks<
    const PINNED: bool,
    A: super::Q8ActivationView,
>(
    detected: super::Ptq1Dotprod,
    rows: [&[u8]; 8],
    chunks: &[[&[A]; 8]],
    outputs: &mut [[[f32; 8]; 8]],
) -> Option<()> {
    if PINNED {
        if detected.i8mm && (1..=8).contains(&chunks.len()) {
            let q8_len = chunks[0][0].len();
            let mut packed = Vec::with_capacity(chunks.len() * 4 * q8_len);
            for chunk in chunks {
                for p in 0..4 {
                    ptq1_append_pinned_i8mm_pair_blocks(
                        chunk[2 * p],
                        chunk[2 * p + 1],
                        &mut packed,
                    );
                }
            }
            let mut panel_chunks: [[&[super::Ptq1PinnedI8mmPairBlock]; 4]; 8] = [[&[]; 4]; 8];
            for (c, panel_chunk) in panel_chunks[..chunks.len()].iter_mut().enumerate() {
                for (p, slot) in panel_chunk.iter_mut().enumerate() {
                    let start = (c * 4 + p) * q8_len;
                    *slot = &packed[start..start + q8_len];
                }
            }
            return ptq1_detected_dot_scaled_8x8_chunks_with_pinned_panels(
                detected,
                rows,
                &panel_chunks[..chunks.len()],
                outputs,
            );
        }
        match chunks.len() {
            // SAFETY: constructing `Ptq1Dotprod` verified CPU/OS `dotprod` and baseline `neon`,
            // and the safe parent checked chunk/row extents and finite activation scales.
            1 => unsafe { ptq1_scaled_8x8_chunks_pinned_dotprod::<1, A>(rows, chunks, outputs) },
            // SAFETY: see arm `1` above.
            2 => unsafe { ptq1_scaled_8x8_chunks_pinned_dotprod::<2, A>(rows, chunks, outputs) },
            // SAFETY: see arm `1` above.
            3 => unsafe { ptq1_scaled_8x8_chunks_pinned_dotprod::<3, A>(rows, chunks, outputs) },
            // SAFETY: see arm `1` above.
            4 => unsafe { ptq1_scaled_8x8_chunks_pinned_dotprod::<4, A>(rows, chunks, outputs) },
            // SAFETY: see arm `1` above.
            5 => unsafe { ptq1_scaled_8x8_chunks_pinned_dotprod::<5, A>(rows, chunks, outputs) },
            // SAFETY: see arm `1` above.
            6 => unsafe { ptq1_scaled_8x8_chunks_pinned_dotprod::<6, A>(rows, chunks, outputs) },
            // SAFETY: see arm `1` above.
            7 => unsafe { ptq1_scaled_8x8_chunks_pinned_dotprod::<7, A>(rows, chunks, outputs) },
            // SAFETY: see arm `1` above.
            8 => unsafe { ptq1_scaled_8x8_chunks_pinned_dotprod::<8, A>(rows, chunks, outputs) },
            _ => None,
        }
    } else if detected.i8mm {
        match chunks.len() {
            // SAFETY: `detected.i8mm` was established by `has_dotprod() && has_i8mm()`,
            // and the safe parent checked chunk/row extents and finite activation scales.
            1 => unsafe { ptq1_scaled_8x8_chunks_source_i8mm::<1, A>(rows, chunks, outputs) },
            // SAFETY: see arm `1` above.
            2 => unsafe { ptq1_scaled_8x8_chunks_source_i8mm::<2, A>(rows, chunks, outputs) },
            // SAFETY: see arm `1` above.
            3 => unsafe { ptq1_scaled_8x8_chunks_source_i8mm::<3, A>(rows, chunks, outputs) },
            // SAFETY: see arm `1` above.
            4 => unsafe { ptq1_scaled_8x8_chunks_source_i8mm::<4, A>(rows, chunks, outputs) },
            // SAFETY: see arm `1` above.
            5 => unsafe { ptq1_scaled_8x8_chunks_source_i8mm::<5, A>(rows, chunks, outputs) },
            // SAFETY: see arm `1` above.
            6 => unsafe { ptq1_scaled_8x8_chunks_source_i8mm::<6, A>(rows, chunks, outputs) },
            // SAFETY: see arm `1` above.
            7 => unsafe { ptq1_scaled_8x8_chunks_source_i8mm::<7, A>(rows, chunks, outputs) },
            // SAFETY: see arm `1` above.
            8 => unsafe { ptq1_scaled_8x8_chunks_source_i8mm::<8, A>(rows, chunks, outputs) },
            _ => None,
        }
    } else {
        match chunks.len() {
            // SAFETY: constructing `Ptq1Dotprod` verified CPU/OS `dotprod` and baseline `neon`,
            // and the safe parent checked chunk/row extents and finite activation scales.
            1 => unsafe { ptq1_scaled_8x8_chunks_source_dotprod::<1, A>(rows, chunks, outputs) },
            // SAFETY: see arm `1` above.
            2 => unsafe { ptq1_scaled_8x8_chunks_source_dotprod::<2, A>(rows, chunks, outputs) },
            // SAFETY: see arm `1` above.
            3 => unsafe { ptq1_scaled_8x8_chunks_source_dotprod::<3, A>(rows, chunks, outputs) },
            // SAFETY: see arm `1` above.
            4 => unsafe { ptq1_scaled_8x8_chunks_source_dotprod::<4, A>(rows, chunks, outputs) },
            // SAFETY: see arm `1` above.
            5 => unsafe { ptq1_scaled_8x8_chunks_source_dotprod::<5, A>(rows, chunks, outputs) },
            // SAFETY: see arm `1` above.
            6 => unsafe { ptq1_scaled_8x8_chunks_source_dotprod::<6, A>(rows, chunks, outputs) },
            // SAFETY: see arm `1` above.
            7 => unsafe { ptq1_scaled_8x8_chunks_source_dotprod::<7, A>(rows, chunks, outputs) },
            // SAFETY: see arm `1` above.
            8 => unsafe { ptq1_scaled_8x8_chunks_source_dotprod::<8, A>(rows, chunks, outputs) },
            _ => None,
        }
    }
}

#[inline]
#[target_feature(enable = "neon,i8mm")]
fn load_q8_0_row_pair_packed(b0: &[u8; 34], b1: &[u8; 34]) -> [int8x16_t; 4] {
    let h0 = b0[2..34].as_chunks::<16>().0;
    let h1 = b1[2..34].as_chunks::<16>().0;
    let (w_h0_lo, w_h0_hi) = pair_s8_lo_hi(
        vreinterpretq_s8_u8(load_u8x16(&h0[0])),
        vreinterpretq_s8_u8(load_u8x16(&h1[0])),
    );
    let (w_h1_lo, w_h1_hi) = pair_s8_lo_hi(
        vreinterpretq_s8_u8(load_u8x16(&h0[1])),
        vreinterpretq_s8_u8(load_u8x16(&h1[1])),
    );
    [w_h0_lo, w_h0_hi, w_h1_lo, w_h1_hi]
}

#[inline]
#[target_feature(enable = "neon,i8mm")]
fn q8_0_row_pair_i8mm<const PAIRS: usize, const HAS_ODD: bool, A: super::Q8ActivationView>(
    pair_blocks: [&[u8; 34]; 2],
    pair_scales: [f32; 2],
    tokens: &[&[A]],
    block_idx: usize,
    mut acc_pairs: [core::arch::aarch64::float32x4_t; PAIRS],
    mut acc_odd: core::arch::aarch64::float32x4_t,
) -> (
    [core::arch::aarch64::float32x4_t; PAIRS],
    core::arch::aarch64::float32x4_t,
) {
    use core::arch::aarch64::{vcvtq_f32_s32, vdupq_n_s8, vmulq_f32};
    let [w_h0_lo, w_h0_hi, w_h1_lo, w_h1_hi] =
        load_q8_0_row_pair_packed(pair_blocks[0], pair_blocks[1]);
    let rs = load_f32x4_neon(&[
        pair_scales[0],
        pair_scales[0],
        pair_scales[1],
        pair_scales[1],
    ]);
    for p in 0..PAIRS {
        let act0 = &tokens[2 * p][block_idx];
        let act1 = &tokens[2 * p + 1][block_idx];
        let a0_halves = act0.values().as_chunks::<16>().0;
        let a1_halves = act1.values().as_chunks::<16>().0;
        let (a_h0_lo, a_h0_hi) =
            pair_s8_lo_hi(load_s8x16(&a0_halves[0]), load_s8x16(&a1_halves[0]));
        let mut int_dot = smmla_s32(vdupq_n_s32(0), w_h0_lo, a_h0_lo);
        int_dot = smmla_s32(int_dot, w_h0_hi, a_h0_hi);
        let (a_h1_lo, a_h1_hi) =
            pair_s8_lo_hi(load_s8x16(&a0_halves[1]), load_s8x16(&a1_halves[1]));
        int_dot = smmla_s32(int_dot, w_h1_lo, a_h1_lo);
        int_dot = smmla_s32(int_dot, w_h1_hi, a_h1_hi);
        let as0 = act0.scale();
        let as1 = act1.scale();
        let d = vmulq_f32(rs, load_f32x4_neon(&[as0, as1, as0, as1]));
        acc_pairs[p] = f32x4_scale_step::<false>(acc_pairs[p], d, vcvtq_f32_s32(int_dot));
    }
    if HAS_ODD {
        let odd_act = &tokens[PAIRS * 2][block_idx];
        let a_halves = odd_act.values().as_chunks::<16>().0;
        let zero = vdupq_n_s8(0);
        let (a_h0_lo, a_h0_hi) = pair_s8_lo_hi(load_s8x16(&a_halves[0]), zero);
        let mut int_dot = smmla_s32(vdupq_n_s32(0), w_h0_lo, a_h0_lo);
        int_dot = smmla_s32(int_dot, w_h0_hi, a_h0_hi);
        let (a_h1_lo, a_h1_hi) = pair_s8_lo_hi(load_s8x16(&a_halves[1]), zero);
        int_dot = smmla_s32(int_dot, w_h1_lo, a_h1_lo);
        int_dot = smmla_s32(int_dot, w_h1_hi, a_h1_hi);
        let as_odd = odd_act.scale();
        let d = vmulq_f32(rs, load_f32x4_neon(&[as_odd, 0.0, as_odd, 0.0]));
        acc_odd = f32x4_scale_step::<false>(acc_odd, d, vcvtq_f32_s32(int_dot));
    }
    (acc_pairs, acc_odd)
}

#[inline]
#[target_feature(enable = "neon")]
fn store_q8_0_pair_outputs<const PAIRS: usize, const HAS_ODD: bool>(
    row_base: usize,
    acc_pairs: [core::arch::aarch64::float32x4_t; PAIRS],
    acc_odd: core::arch::aarch64::float32x4_t,
    out: &mut [[f32; 8]; 8],
) {
    for (p, acc) in acc_pairs.into_iter().enumerate() {
        let mut vals = [0.0_f32; 4];
        store_f32x4_neon(acc, &mut vals);
        out[row_base][2 * p] = vals[0];
        out[row_base][2 * p + 1] = vals[1];
        out[row_base + 1][2 * p] = vals[2];
        out[row_base + 1][2 * p + 1] = vals[3];
    }
    if HAS_ODD {
        let mut vals = [0.0_f32; 4];
        store_f32x4_neon(acc_odd, &mut vals);
        out[row_base][PAIRS * 2] = vals[0];
        out[row_base + 1][PAIRS * 2] = vals[2];
    }
}

#[target_feature(enable = "neon,i8mm")]
fn q8_0_scaled_8x_pairs_i8mm<
    const PAIRS: usize,
    const HAS_ODD: bool,
    A: super::Q8ActivationView,
>(
    rows: [&[u8]; 8],
    tokens: &[&[A]],
    out: &mut [[f32; 8]; 8],
) -> Option<()> {
    use core::arch::aarch64::vdupq_n_f32;
    let block_count = tokens[0].len();
    let row_chunks = rows.map(|row| row.as_chunks::<34>().0);
    let mut acc0 = [vdupq_n_f32(0.0); PAIRS];
    let mut acc1 = [vdupq_n_f32(0.0); PAIRS];
    let mut acc2 = [vdupq_n_f32(0.0); PAIRS];
    let mut acc3 = [vdupq_n_f32(0.0); PAIRS];
    let mut odd0 = vdupq_n_f32(0.0);
    let mut odd1 = vdupq_n_f32(0.0);
    let mut odd2 = vdupq_n_f32(0.0);
    let mut odd3 = vdupq_n_f32(0.0);
    for (block_idx, _) in row_chunks[0].iter().enumerate().take(block_count) {
        let blocks: [&[u8; 34]; 8] = std::array::from_fn(|r| &row_chunks[r][block_idx]);
        let [rs_lo, rs_hi] = widen_row_scales_f16(&blocks)?;
        (acc0, odd0) = q8_0_row_pair_i8mm::<PAIRS, HAS_ODD, A>(
            [blocks[0], blocks[1]],
            [rs_lo[0], rs_lo[1]],
            tokens,
            block_idx,
            acc0,
            odd0,
        );
        (acc1, odd1) = q8_0_row_pair_i8mm::<PAIRS, HAS_ODD, A>(
            [blocks[2], blocks[3]],
            [rs_lo[2], rs_lo[3]],
            tokens,
            block_idx,
            acc1,
            odd1,
        );
        (acc2, odd2) = q8_0_row_pair_i8mm::<PAIRS, HAS_ODD, A>(
            [blocks[4], blocks[5]],
            [rs_hi[0], rs_hi[1]],
            tokens,
            block_idx,
            acc2,
            odd2,
        );
        (acc3, odd3) = q8_0_row_pair_i8mm::<PAIRS, HAS_ODD, A>(
            [blocks[6], blocks[7]],
            [rs_hi[2], rs_hi[3]],
            tokens,
            block_idx,
            acc3,
            odd3,
        );
    }
    let mut tmp = [[0.0_f32; 8]; 8];
    store_q8_0_pair_outputs::<PAIRS, HAS_ODD>(0, acc0, odd0, &mut tmp);
    store_q8_0_pair_outputs::<PAIRS, HAS_ODD>(2, acc1, odd1, &mut tmp);
    store_q8_0_pair_outputs::<PAIRS, HAS_ODD>(4, acc2, odd2, &mut tmp);
    store_q8_0_pair_outputs::<PAIRS, HAS_ODD>(6, acc3, odd3, &mut tmp);
    if !tmp.iter().flatten().all(|v| v.is_finite()) {
        return None;
    }
    *out = tmp;
    Some(())
}

#[target_feature(enable = "neon,i8mm")]
fn q8_0_scaled_8x_tokens_i8mm<A: super::Q8ActivationView>(
    rows: [&[u8]; 8],
    tokens: &[&[A]],
    out: &mut [[f32; 8]; 8],
) -> Option<()> {
    match tokens.len() {
        1 => q8_0_scaled_8x_pairs_i8mm::<0, true, A>(rows, tokens, out),
        2 => q8_0_scaled_8x_pairs_i8mm::<1, false, A>(rows, tokens, out),
        3 => q8_0_scaled_8x_pairs_i8mm::<1, true, A>(rows, tokens, out),
        4 => q8_0_scaled_8x_pairs_i8mm::<2, false, A>(rows, tokens, out),
        5 => q8_0_scaled_8x_pairs_i8mm::<2, true, A>(rows, tokens, out),
        6 => q8_0_scaled_8x_pairs_i8mm::<3, false, A>(rows, tokens, out),
        7 => q8_0_scaled_8x_pairs_i8mm::<3, true, A>(rows, tokens, out),
        8 => q8_0_scaled_8x_pairs_i8mm::<4, false, A>(rows, tokens, out),
        _ => None,
    }
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
fn q8_0_4rows_sdot<const TOKENS: usize, A: super::Q8ActivationView>(
    blocks: [&[u8; 34]; 4],
    row_scales: [f32; 4],
    tokens: &[&[A]],
    block_idx: usize,
    mut acc: [core::arch::aarch64::float32x4_t; TOKENS],
) -> [core::arch::aarch64::float32x4_t; TOKENS] {
    use core::arch::aarch64::{vcvtq_f32_s32, vdupq_n_f32, vmulq_f32};
    let rs_vec = load_f32x4_neon(&row_scales);
    let (w0_0, w0_1) = {
        let h = blocks[0][2..34].as_chunks::<16>().0;
        (
            vreinterpretq_s8_u8(load_u8x16(&h[0])),
            vreinterpretq_s8_u8(load_u8x16(&h[1])),
        )
    };
    let (w1_0, w1_1) = {
        let h = blocks[1][2..34].as_chunks::<16>().0;
        (
            vreinterpretq_s8_u8(load_u8x16(&h[0])),
            vreinterpretq_s8_u8(load_u8x16(&h[1])),
        )
    };
    let (w2_0, w2_1) = {
        let h = blocks[2][2..34].as_chunks::<16>().0;
        (
            vreinterpretq_s8_u8(load_u8x16(&h[0])),
            vreinterpretq_s8_u8(load_u8x16(&h[1])),
        )
    };
    let (w3_0, w3_1) = {
        let h = blocks[3][2..34].as_chunks::<16>().0;
        (
            vreinterpretq_s8_u8(load_u8x16(&h[0])),
            vreinterpretq_s8_u8(load_u8x16(&h[1])),
        )
    };
    for (t, tok) in tokens.iter().enumerate() {
        let act = &tok[block_idx];
        let a_halves = act.values().as_chunks::<16>().0;
        let a0 = load_s8x16(&a_halves[0]);
        let a1 = load_s8x16(&a_halves[1]);
        let d0 = sdot32_vec(w0_0, w0_1, a0, a1);
        let d1 = sdot32_vec(w1_0, w1_1, a0, a1);
        let d2 = sdot32_vec(w2_0, w2_1, a0, a1);
        let d3 = sdot32_vec(w3_0, w3_1, a0, a1);
        let dots = vcvtq_f32_s32(reduce4_s32x4_addp(d0, d1, d2, d3));
        let d = vmulq_f32(rs_vec, vdupq_n_f32(act.scale()));
        acc[t] = f32x4_scale_step::<false>(acc[t], d, dots);
    }
    acc
}

#[target_feature(enable = "neon,dotprod")]
fn q8_0_scaled_8x_n_dotprod<const TOKENS: usize, A: super::Q8ActivationView>(
    rows: [&[u8]; 8],
    tokens: &[&[A]],
    out: &mut [[f32; 8]; 8],
) -> Option<()> {
    use core::arch::aarch64::vdupq_n_f32;
    let block_count = tokens[0].len();
    let row_chunks = rows.map(|row| row.as_chunks::<34>().0);
    let mut acc_lo = [vdupq_n_f32(0.0); TOKENS];
    let mut acc_hi = [vdupq_n_f32(0.0); TOKENS];
    for (block_idx, _) in row_chunks[0].iter().enumerate().take(block_count) {
        let blocks: [&[u8; 34]; 8] = std::array::from_fn(|r| &row_chunks[r][block_idx]);
        let [rs_lo, rs_hi] = widen_row_scales_f16(&blocks)?;
        acc_lo = q8_0_4rows_sdot::<TOKENS, A>(
            [blocks[0], blocks[1], blocks[2], blocks[3]],
            rs_lo,
            tokens,
            block_idx,
            acc_lo,
        );
        acc_hi = q8_0_4rows_sdot::<TOKENS, A>(
            [blocks[4], blocks[5], blocks[6], blocks[7]],
            rs_hi,
            tokens,
            block_idx,
            acc_hi,
        );
    }
    let mut tmp = [[0.0_f32; 8]; 8];
    for t in 0..TOKENS {
        let mut lo = [0.0_f32; 4];
        let mut hi = [0.0_f32; 4];
        store_f32x4_neon(acc_lo[t], &mut lo);
        store_f32x4_neon(acc_hi[t], &mut hi);
        for r in 0..4 {
            tmp[r][t] = lo[r];
            tmp[4 + r][t] = hi[r];
        }
    }
    if !tmp.iter().flatten().all(|v| v.is_finite()) {
        return None;
    }
    *out = tmp;
    Some(())
}

#[target_feature(enable = "neon,dotprod")]
fn q8_0_scaled_8x_tokens_dotprod<A: super::Q8ActivationView>(
    rows: [&[u8]; 8],
    tokens: &[&[A]],
    out: &mut [[f32; 8]; 8],
) -> Option<()> {
    match tokens.len() {
        1 => q8_0_scaled_8x_n_dotprod::<1, A>(rows, tokens, out),
        2 => q8_0_scaled_8x_n_dotprod::<2, A>(rows, tokens, out),
        3 => q8_0_scaled_8x_n_dotprod::<3, A>(rows, tokens, out),
        4 => q8_0_scaled_8x_n_dotprod::<4, A>(rows, tokens, out),
        5 => q8_0_scaled_8x_n_dotprod::<5, A>(rows, tokens, out),
        6 => q8_0_scaled_8x_n_dotprod::<6, A>(rows, tokens, out),
        7 => q8_0_scaled_8x_n_dotprod::<7, A>(rows, tokens, out),
        8 => q8_0_scaled_8x_n_dotprod::<8, A>(rows, tokens, out),
        _ => None,
    }
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
fn load_f16_scales_x4(blocks: [&[u8; 34]; 4]) -> core::arch::aarch64::uint16x4_t {
    use core::arch::aarch64::{vdup_n_u16, vset_lane_u16};
    let v = vdup_n_u16(u16::from_le_bytes([blocks[0][0], blocks[0][1]]));
    let v = vset_lane_u16::<1>(u16::from_le_bytes([blocks[1][0], blocks[1][1]]), v);
    let v = vset_lane_u16::<2>(u16::from_le_bytes([blocks[2][0], blocks[2][1]]), v);
    vset_lane_u16::<3>(u16::from_le_bytes([blocks[3][0], blocks[3][1]]), v)
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
fn widen_f16_one_finite(bits: u16) -> Option<f32> {
    use core::arch::aarch64::{vcvt_f32_f16, vdup_n_u16, vgetq_lane_f32, vreinterpret_f16_u16};
    if (bits & 0x7fff) >= 0x7c00 {
        return None;
    }
    let v = vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(bits)));
    Some(vgetq_lane_f32::<0>(v))
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
fn q8_0_block_sdot<A: super::Q8ActivationView>(block: &[u8; 34], act: &A) -> int32x4_t {
    let wh = block[2..34].as_chunks::<16>().0;
    let ah = act.values().as_chunks::<16>().0;
    sdot32_vec(
        vreinterpretq_s8_u8(load_u8x16(&wh[0])),
        vreinterpretq_s8_u8(load_u8x16(&wh[1])),
        load_s8x16(&ah[0]),
        load_s8x16(&ah[1]),
    )
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
fn q8_0_quad_sdot_step(
    blocks: [&[u8; 34]; 4],
    a0: int8x16_t,
    a1: int8x16_t,
    rs_vec: core::arch::aarch64::float32x4_t,
    act_scale_vec: core::arch::aarch64::float32x4_t,
    sum: core::arch::aarch64::float32x4_t,
) -> core::arch::aarch64::float32x4_t {
    use core::arch::aarch64::{vcvtq_f32_s32, vmulq_f32};
    let w0 = blocks[0][2..34].as_chunks::<16>().0;
    let w1 = blocks[1][2..34].as_chunks::<16>().0;
    let w2 = blocks[2][2..34].as_chunks::<16>().0;
    let w3 = blocks[3][2..34].as_chunks::<16>().0;
    let mut d0 = vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(load_u8x16(&w0[0])), a0);
    let mut d1 = vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(load_u8x16(&w1[0])), a0);
    let mut d2 = vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(load_u8x16(&w2[0])), a0);
    let mut d3 = vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(load_u8x16(&w3[0])), a0);
    d0 = vdotq_s32(d0, vreinterpretq_s8_u8(load_u8x16(&w0[1])), a1);
    d1 = vdotq_s32(d1, vreinterpretq_s8_u8(load_u8x16(&w1[1])), a1);
    d2 = vdotq_s32(d2, vreinterpretq_s8_u8(load_u8x16(&w2[1])), a1);
    d3 = vdotq_s32(d3, vreinterpretq_s8_u8(load_u8x16(&w3[1])), a1);
    let dots = vcvtq_f32_s32(reduce4_s32x4_addp(d0, d1, d2, d3));
    let d = vmulq_f32(rs_vec, act_scale_vec);
    f32x4_scale_step::<false>(sum, d, dots)
}

#[target_feature(enable = "neon,dotprod")]
fn q8_0_1row_sdot<A: super::Q8ActivationView>(row: &[u8], activation: &[A]) -> Option<[f32; 1]> {
    use core::arch::aarch64::{
        vand_u16, vcvt_f32_f16, vcvtq_f32_s32, vdup_n_u16, vgetq_lane_f32, vmaxv_u16, vmulq_f32,
        vreinterpret_f16_u16,
    };
    let row_chunks = row.as_chunks::<34>().0;
    let block_count = activation.len();
    if block_count == 0 || row_chunks.len() != block_count {
        return None;
    }
    let (row_quads, row_tail) = row_chunks.as_chunks::<4>();
    let (act_quads, act_tail) = activation.as_chunks::<4>();
    let mask_7fff = vdup_n_u16(0x7fff);
    let mut sum = 0.0_f32;
    for (w_quad, a_quad) in row_quads.iter().zip(act_quads.iter()) {
        let s = load_f16_scales_x4([&w_quad[0], &w_quad[1], &w_quad[2], &w_quad[3]]);
        if vmaxv_u16(vand_u16(s, mask_7fff)) >= 0x7c00 {
            return None;
        }
        let rs = vcvt_f32_f16(vreinterpret_f16_u16(s));
        let as_vec = load_f32x4_neon(&[
            a_quad[0].scale(),
            a_quad[1].scale(),
            a_quad[2].scale(),
            a_quad[3].scale(),
        ]);
        let d_vec = vmulq_f32(rs, as_vec);
        let d0 = q8_0_block_sdot(&w_quad[0], &a_quad[0]);
        let d1 = q8_0_block_sdot(&w_quad[1], &a_quad[1]);
        let d2 = q8_0_block_sdot(&w_quad[2], &a_quad[2]);
        let d3 = q8_0_block_sdot(&w_quad[3], &a_quad[3]);
        let dots = vcvtq_f32_s32(reduce4_s32x4_addp(d0, d1, d2, d3));
        let prods = vmulq_f32(d_vec, dots);
        sum = (((sum + vgetq_lane_f32::<0>(prods)) + vgetq_lane_f32::<1>(prods))
            + vgetq_lane_f32::<2>(prods))
            + vgetq_lane_f32::<3>(prods);
    }
    for (wb, ab) in row_tail.iter().zip(act_tail.iter()) {
        let bits = u16::from_le_bytes([wb[0], wb[1]]);
        let rs = widen_f16_one_finite(bits)?;
        let dot = vaddvq_s32(q8_0_block_sdot(wb, ab));
        #[expect(
            clippy::cast_precision_loss,
            clippy::as_conversions,
            clippy::suboptimal_flops,
            reason = "32-element i8 dot fits exactly in f32 and non-FMA order matches dot_legacy."
        )]
        {
            sum += (rs * ab.scale()) * (dot as f32);
        }
    }
    sum.is_finite().then_some([sum])
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
fn q8_0_4rows_pair_step<A: super::Q8ActivationView>(
    b0: [&[u8; 34]; 4],
    b1: [&[u8; 34]; 4],
    act0: &A,
    act1: &A,
    mask_7fff: core::arch::aarch64::uint16x8_t,
    mut sum0: core::arch::aarch64::float32x4_t,
) -> Option<core::arch::aarch64::float32x4_t> {
    use core::arch::aarch64::{
        vandq_u16, vcombine_u16, vcvt_f32_f16, vcvt_high_f32_f16, vdupq_n_f32, vget_low_u16,
        vmaxvq_u16, vreinterpret_f16_u16, vreinterpretq_f16_u16,
    };
    let s_pair = vcombine_u16(load_f16_scales_x4(b0), load_f16_scales_x4(b1));
    if vmaxvq_u16(vandq_u16(s_pair, mask_7fff)) >= 0x7c00 {
        return None;
    }
    let rs0 = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(s_pair)));
    let rs1 = vcvt_high_f32_f16(vreinterpretq_f16_u16(s_pair));

    let a0_h = act0.values().as_chunks::<16>().0;
    let a1_h = act1.values().as_chunks::<16>().0;
    let a00 = load_s8x16(&a0_h[0]);
    let a01 = load_s8x16(&a0_h[1]);
    let a10 = load_s8x16(&a1_h[0]);
    let a11 = load_s8x16(&a1_h[1]);

    sum0 = q8_0_quad_sdot_step(b0, a00, a01, rs0, vdupq_n_f32(act0.scale()), sum0);
    sum0 = q8_0_quad_sdot_step(b1, a10, a11, rs1, vdupq_n_f32(act1.scale()), sum0);
    Some(sum0)
}

#[target_feature(enable = "neon,dotprod")]
fn q8_0_4rows_sdot_full<A: super::Q8ActivationView>(
    rows: [&[u8]; 4],
    activation: &[A],
) -> Option<[f32; 4]> {
    use core::arch::aarch64::{
        vand_u16, vcvt_f32_f16, vdup_n_u16, vdupq_n_f32, vdupq_n_u16, vmaxv_u16,
        vreinterpret_f16_u16,
    };
    let block_count = activation.len();
    let row_chunks: [&[[u8; 34]]; 4] = rows.map(|row| row.as_chunks::<34>().0);
    if block_count == 0 || row_chunks.iter().any(|c| c.len() != block_count) {
        return None;
    }
    let (act_pairs, act_tail) = activation.as_chunks::<2>();
    let r0 = row_chunks[0].as_chunks::<2>();
    let r1 = row_chunks[1].as_chunks::<2>();
    let r2 = row_chunks[2].as_chunks::<2>();
    let r3 = row_chunks[3].as_chunks::<2>();
    let mut sum0 = vdupq_n_f32(0.0);
    let mask_7fff = vdupq_n_u16(0x7fff);
    for ((((p0, p1), p2), p3), act_pair) in r0.0.iter().zip(r1.0).zip(r2.0).zip(r3.0).zip(act_pairs)
    {
        sum0 = q8_0_4rows_pair_step(
            [&p0[0], &p1[0], &p2[0], &p3[0]],
            [&p0[1], &p1[1], &p2[1], &p3[1]],
            &act_pair[0],
            &act_pair[1],
            mask_7fff,
            sum0,
        )?;
    }
    if let (Some(b0), Some(b1), Some(b2), Some(b3), Some(act0)) = (
        r0.1.first(),
        r1.1.first(),
        r2.1.first(),
        r3.1.first(),
        act_tail.first(),
    ) {
        let blocks = [b0, b1, b2, b3];
        let s = load_f16_scales_x4(blocks);
        if vmaxv_u16(vand_u16(s, vdup_n_u16(0x7fff))) >= 0x7c00 {
            return None;
        }
        let rs0 = vcvt_f32_f16(vreinterpret_f16_u16(s));
        let a0_h = act0.values().as_chunks::<16>().0;
        sum0 = q8_0_quad_sdot_step(
            blocks,
            load_s8x16(&a0_h[0]),
            load_s8x16(&a0_h[1]),
            rs0,
            vdupq_n_f32(act0.scale()),
            sum0,
        );
    }
    let mut out = [0.0_f32; 4];
    store_f32x4_neon(sum0, &mut out);
    out.iter().all(|v| v.is_finite()).then_some(out)
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
fn q8_0_8rows_step<A: super::Q8ActivationView>(
    blocks: [&[u8; 34]; 8],
    act: &A,
    mask_7fff: core::arch::aarch64::uint16x8_t,
    sum0: core::arch::aarch64::float32x4_t,
    sum1: core::arch::aarch64::float32x4_t,
) -> Option<(
    core::arch::aarch64::float32x4_t,
    core::arch::aarch64::float32x4_t,
)> {
    use core::arch::aarch64::{
        vandq_u16, vcombine_u16, vcvt_f32_f16, vcvt_high_f32_f16, vdupq_n_f32, vget_low_u16,
        vmaxvq_u16, vreinterpret_f16_u16, vreinterpretq_f16_u16,
    };
    let s0 = vcombine_u16(
        load_f16_scales_x4([blocks[0], blocks[1], blocks[2], blocks[3]]),
        load_f16_scales_x4([blocks[4], blocks[5], blocks[6], blocks[7]]),
    );
    if vmaxvq_u16(vandq_u16(s0, mask_7fff)) >= 0x7c00 {
        return None;
    }
    let rs_lo = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(s0)));
    let rs_hi = vcvt_high_f32_f16(vreinterpretq_f16_u16(s0));
    let a_h = act.values().as_chunks::<16>().0;
    let a0 = load_s8x16(&a_h[0]);
    let a1 = load_s8x16(&a_h[1]);
    let act_scale = vdupq_n_f32(act.scale());
    Some((
        q8_0_quad_sdot_step(
            [blocks[0], blocks[1], blocks[2], blocks[3]],
            a0,
            a1,
            rs_lo,
            act_scale,
            sum0,
        ),
        q8_0_quad_sdot_step(
            [blocks[4], blocks[5], blocks[6], blocks[7]],
            a0,
            a1,
            rs_hi,
            act_scale,
            sum1,
        ),
    ))
}

#[inline]
fn q8_activation_offsets<A: super::Q8ActivationView>(
    activation: &[A],
) -> Option<(usize, usize, usize)> {
    let first = activation.first()?;
    let scale_ref = first.scale_ref()?;
    let base = core::ptr::from_ref(first).addr();
    let val_off = core::ptr::from_ref(first.values())
        .addr()
        .checked_sub(base)?;
    let scale_off = core::ptr::from_ref(scale_ref).addr().checked_sub(base)?;
    let stride = size_of::<A>();
    if val_off.checked_add(32)? <= stride && scale_off.checked_add(4)? <= stride {
        Some((stride, val_off, scale_off))
    } else {
        None
    }
}

macro_rules! q8_0_row_sdot_asm {
    ($p:literal, $acc:literal) => {
        concat!(
            "ldp q28, q29, [{",
            $p,
            "}], #32\n",
            "movi ",
            $acc,
            ".2d, #0\n",
            "sdot ",
            $acc,
            ".4s, v28.16b, v4.16b\n",
            "sdot ",
            $acc,
            ".4s, v29.16b, v5.16b\n",
        )
    };
}

macro_rules! q8_0_8rows_block_asm {
    () => {
        concat!(
            "ldr q4, [{act}, {val_off}]\n",
            "ldr q5, [{act}, {val_off_hi}]\n",
            "ldr s3, [{act}, {scale_off}]\n",
            "add {act}, {act}, {stride}\n",
            "ldrh {t0:w}, [{p0}], #2\n",
            "ldrh {t1:w}, [{p1}], #2\n",
            "fmov s30, {t0:w}\n",
            "ldrh {t0:w}, [{p2}], #2\n",
            "ins v30.h[1], {t1:w}\n",
            "ldrh {t1:w}, [{p3}], #2\n",
            "ins v30.h[2], {t0:w}\n",
            "ldrh {t0:w}, [{p4}], #2\n",
            "ins v30.h[3], {t1:w}\n",
            "ldrh {t1:w}, [{p5}], #2\n",
            "ins v30.h[4], {t0:w}\n",
            "ldrh {t0:w}, [{p6}], #2\n",
            "ins v30.h[5], {t1:w}\n",
            "ldrh {t1:w}, [{p7}], #2\n",
            "ins v30.h[6], {t0:w}\n",
            "ins v30.h[7], {t1:w}\n",
            "mov v31.16b, v30.16b\n",
            "bic v31.8h, #0x80, lsl #8\n",
            "umax v2.8h, v2.8h, v31.8h\n",
            "fcvtl v16.4s, v30.4h\n",
            "fcvtl2 v17.4s, v30.8h\n",
            "fmul v16.4s, v16.4s, v3.s[0]\n",
            "fmul v17.4s, v17.4s, v3.s[0]\n",
            q8_0_row_sdot_asm!("p0", "v20"),
            q8_0_row_sdot_asm!("p1", "v21"),
            q8_0_row_sdot_asm!("p2", "v22"),
            q8_0_row_sdot_asm!("p3", "v23"),
            q8_0_row_sdot_asm!("p4", "v24"),
            q8_0_row_sdot_asm!("p5", "v25"),
            q8_0_row_sdot_asm!("p6", "v26"),
            q8_0_row_sdot_asm!("p7", "v27"),
            "addp v20.4s, v20.4s, v21.4s\n",
            "addp v22.4s, v22.4s, v23.4s\n",
            "addp v20.4s, v20.4s, v22.4s\n",
            "scvtf v20.4s, v20.4s\n",
            "fmul v20.4s, v16.4s, v20.4s\n",
            "fadd v0.4s, v0.4s, v20.4s\n",
            "addp v24.4s, v24.4s, v25.4s\n",
            "addp v26.4s, v26.4s, v27.4s\n",
            "addp v24.4s, v24.4s, v26.4s\n",
            "scvtf v24.4s, v24.4s\n",
            "fmul v24.4s, v17.4s, v24.4s\n",
            "fadd v1.4s, v1.4s, v24.4s\n",
        )
    };
}

#[target_feature(enable = "neon,dotprod")]
fn q8_0_8rows_sdot<A: super::Q8ActivationView>(
    rows: [&[u8]; 8],
    activation: &[A],
    stride: usize,
    val_off: usize,
    scale_off: usize,
) -> Option<[f32; 8]> {
    use core::arch::aarch64::vdupq_n_f32;
    let block_count = activation.len();
    let expected = block_count.checked_mul(34)?;
    if block_count == 0 || rows.iter().any(|r| r.len() != expected) {
        return None;
    }
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let p0 = rows[0].as_ptr();
    let p1 = rows[1].as_ptr();
    let p2 = rows[2].as_ptr();
    let p3 = rows[3].as_ptr();
    let p4 = rows[4].as_ptr();
    let p5 = rows[5].as_ptr();
    let p6 = rows[6].as_ptr();
    let p7 = rows[7].as_ptr();
    let act = activation.as_ptr().cast::<u8>();
    let val_off_hi = val_off + 16;
    let cnt = block_count;
    let ok: usize;
    // SAFETY: Caller verified every row covers `block_count * 34` bytes,
    // `activation` covers `block_count` elements of `A`, and `val_off..+32` /
    // `scale_off..+4` lie within `size_of::<A>()`.
    unsafe {
        core::arch::asm!(
            "movi v2.2d, #0",
            "tst {cnt}, #1",
            "b.eq 1f",
            q8_0_8rows_block_asm!(),
            "sub {cnt}, {cnt}, #1",
            "cbz {cnt}, 2f",
            "1:",
            "prfm pldl1keep, [{p0}, #136]",
            "prfm pldl1keep, [{p1}, #136]",
            "prfm pldl1keep, [{p2}, #136]",
            "prfm pldl1keep, [{p3}, #136]",
            "prfm pldl1keep, [{p4}, #136]",
            "prfm pldl1keep, [{p5}, #136]",
            "prfm pldl1keep, [{p6}, #136]",
            "prfm pldl1keep, [{p7}, #136]",
            "prfm pldl1keep, [{act}, #128]",
            q8_0_8rows_block_asm!(),
            q8_0_8rows_block_asm!(),
            "subs {cnt}, {cnt}, #2",
            "b.ne 1b",
            "2:",
            "umaxv h28, v2.8h",
            "fmov {t0:w}, s28",
            "lsr {t0:w}, {t0:w}, #10",
            "cmp {t0:w}, #30",
            "cset {ok}, ls",
            p0 = inout(reg) p0 => _,
            p1 = inout(reg) p1 => _,
            p2 = inout(reg) p2 => _,
            p3 = inout(reg) p3 => _,
            p4 = inout(reg) p4 => _,
            p5 = inout(reg) p5 => _,
            p6 = inout(reg) p6 => _,
            p7 = inout(reg) p7 => _,
            act = inout(reg) act => _,
            stride = in(reg) stride,
            val_off = in(reg) val_off,
            val_off_hi = in(reg) val_off_hi,
            scale_off = in(reg) scale_off,
            cnt = inout(reg) cnt => _,
            t0 = out(reg) _,
            t1 = out(reg) _,
            ok = out(reg) ok,
            inout("v0") sum0,
            inout("v1") sum1,
            out("v2") _,
            out("v3") _,
            out("v4") _,
            out("v5") _,
            out("v16") _,
            out("v17") _,
            out("v20") _,
            out("v21") _,
            out("v22") _,
            out("v23") _,
            out("v24") _,
            out("v25") _,
            out("v26") _,
            out("v27") _,
            out("v28") _,
            out("v29") _,
            out("v30") _,
            out("v31") _,
            options(nostack, readonly),
        );
    }
    if ok == 0 {
        return None;
    }
    let mut lo = [0.0_f32; 4];
    let mut hi = [0.0_f32; 4];
    store_f32x4_neon(sum0, &mut lo);
    store_f32x4_neon(sum1, &mut hi);
    let out = [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]];
    out.iter().all(|v| v.is_finite()).then_some(out)
}

#[target_feature(enable = "neon,dotprod")]
fn q8_0_8rows_sdot_full<A: super::Q8ActivationView>(
    rows: [&[u8]; 8],
    activation: &[A],
) -> Option<[f32; 8]> {
    use core::arch::aarch64::{vdupq_n_f32, vdupq_n_u16};
    if let Some((stride, val_off, scale_off)) = q8_activation_offsets(activation) {
        return q8_0_8rows_sdot(rows, activation, stride, val_off, scale_off);
    }
    let block_count = activation.len();
    let row_chunks: [&[[u8; 34]]; 8] = rows.map(|row| row.as_chunks::<34>().0);
    if block_count == 0 || row_chunks.iter().any(|c| c.len() != block_count) {
        return None;
    }
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mask_7fff = vdupq_n_u16(0x7fff);
    for (block_idx, act) in activation.iter().enumerate() {
        prefetch_row_blocks(&row_chunks, block_idx, block_count);
        let blocks: [&[u8; 34]; 8] = std::array::from_fn(|r| &row_chunks[r][block_idx]);
        (sum0, sum1) = q8_0_8rows_step(blocks, act, mask_7fff, sum0, sum1)?;
    }
    let mut lo = [0.0_f32; 4];
    let mut hi = [0.0_f32; 4];
    store_f32x4_neon(sum0, &mut lo);
    store_f32x4_neon(sum1, &mut hi);
    let out = [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]];
    out.iter().all(|v| v.is_finite()).then_some(out)
}

#[target_feature(enable = "neon,dotprod")]
fn q8_0_16rows_sdot_full<A: super::Q8ActivationView>(
    rows: [&[u8]; 16],
    activation: &[A],
) -> Option<[f32; 16]> {
    let lo = q8_0_8rows_sdot_full(
        [
            rows[0], rows[1], rows[2], rows[3], rows[4], rows[5], rows[6], rows[7],
        ],
        activation,
    )?;
    let hi = q8_0_8rows_sdot_full(
        [
            rows[8], rows[9], rows[10], rows[11], rows[12], rows[13], rows[14], rows[15],
        ],
        activation,
    )?;
    Some([
        lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7], hi[0], hi[1], hi[2], hi[3], hi[4],
        hi[5], hi[6], hi[7],
    ])
}

#[inline]
pub(super) fn q8_0_detected_dot_scaled_rows<const ROWS: usize, A: super::Q8ActivationView>(
    detected: super::Q8Dotprod,
    rows: [&[u8]; ROWS],
    activation: &[A],
) -> Option<[f32; ROWS]> {
    if !detected.dotprod {
        return super::scaled::portable_q8_0_rows_validated::<ROWS, A>(rows, activation);
    }
    match ROWS {
        1 => {
            let r1: [&[u8]; 1] = rows.as_slice().try_into().ok()?;
            // SAFETY: `detected.dotprod` was established by `has_dotprod()` and baseline `neon`
            // on AArch64; the safe caller validated row/activation extents and finite scales.
            let res = unsafe { q8_0_1row_sdot::<A>(r1[0], activation)? };
            res.as_slice().try_into().ok()
        }
        4 => {
            let r4: [&[u8]; 4] = rows.as_slice().try_into().ok()?;
            // SAFETY: `detected.dotprod` was established by `has_dotprod()` and baseline `neon`
            // on AArch64; the safe caller validated row/activation extents and finite scales.
            let res = unsafe { q8_0_4rows_sdot_full::<A>(r4, activation)? };
            res.as_slice().try_into().ok()
        }
        8 => {
            let r8: [&[u8]; 8] = rows.as_slice().try_into().ok()?;
            // SAFETY: `detected.dotprod` was established by `has_dotprod()` and baseline `neon`
            // on AArch64; the safe caller validated row/activation extents and finite scales.
            let res = unsafe { q8_0_8rows_sdot_full::<A>(r8, activation)? };
            res.as_slice().try_into().ok()
        }
        16 => {
            let r16: [&[u8]; 16] = rows.as_slice().try_into().ok()?;
            // SAFETY: `detected.dotprod` was established by `has_dotprod()` and baseline `neon`
            // on AArch64; the safe caller validated row/activation extents and finite scales.
            let res = unsafe { q8_0_16rows_sdot_full::<A>(r16, activation)? };
            res.as_slice().try_into().ok()
        }
        _ => None,
    }
}

#[inline]
pub(super) fn q8_0_detected_dot_scaled_8x_tokens<A: super::Q8ActivationView>(
    detected: super::Q8Dotprod,
    rows: [&[u8]; 8],
    tokens: &[&[A]],
    out: &mut [[f32; 8]; 8],
) -> Option<()> {
    if detected.i8mm && (tokens.len() >= 2 || !detected.dotprod) {
        // SAFETY: `detected.i8mm` was established by `has_i8mm()` and baseline `neon` on AArch64;
        // the safe caller validated row/token extents, token count `1..=8`, and finite scales.
        unsafe { q8_0_scaled_8x_tokens_i8mm::<A>(rows, tokens, out) }
    } else if detected.dotprod {
        // SAFETY: `detected.dotprod` was established by `has_dotprod()` and baseline `neon` on
        // AArch64; the safe caller validated row/token extents, token count `1..=8`, and finite scales.
        unsafe { q8_0_scaled_8x_tokens_dotprod::<A>(rows, tokens, out) }
    } else {
        None
    }
}

#[target_feature(enable = "neon,i8mm")]
fn q8_0_scaled_8x8_chunks_i8mm<const CHUNKS: usize, A: super::Q8ActivationView>(
    rows: [&[u8]; 8],
    chunks: &[[&[A]; 8]],
    outputs: &mut [[[f32; 8]; 8]],
) -> Option<()> {
    use core::arch::aarch64::{vcvtq_f32_s32, vdupq_n_f32, vdupq_n_s8, vmulq_f32};

    const K_TILE: usize = 8;
    let block_count = chunks[0][0].len();
    let row_chunks = rows.map(|row| row.as_chunks::<34>().0);
    let mut acc_buf = [[[[0.0_f32; 4]; 4]; 4]; CHUNKS];
    let mut out_buf = [[[0.0_f32; 8]; 8]; CHUNKS];
    let zero_s32 = vdupq_n_s32(0);

    for k_base in (0..block_count).step_by(K_TILE) {
        let tile_len = (block_count - k_base).min(K_TILE);
        let is_first = k_base == 0;
        let is_last = k_base + tile_len == block_count;
        let mut wp_tile = [[[vdupq_n_s8(0); 4]; 4]; K_TILE];
        let mut rs_tile = [[vdupq_n_f32(0.0); 4]; K_TILE];
        for k_off in 0..tile_len {
            let block_idx = k_base + k_off;
            prefetch_row_blocks(&row_chunks, block_idx, block_count);
            let blocks: [&[u8; 34]; 8] = std::array::from_fn(|r| &row_chunks[r][block_idx]);
            let [rs_lo, rs_hi] = widen_row_scales_f16(&blocks)?;
            wp_tile[k_off] = [
                load_q8_0_row_pair_packed(blocks[0], blocks[1]),
                load_q8_0_row_pair_packed(blocks[2], blocks[3]),
                load_q8_0_row_pair_packed(blocks[4], blocks[5]),
                load_q8_0_row_pair_packed(blocks[6], blocks[7]),
            ];
            rs_tile[k_off] = [
                load_f32x4_neon(&[rs_lo[0], rs_lo[0], rs_lo[1], rs_lo[1]]),
                load_f32x4_neon(&[rs_lo[2], rs_lo[2], rs_lo[3], rs_lo[3]]),
                load_f32x4_neon(&[rs_hi[0], rs_hi[0], rs_hi[1], rs_hi[1]]),
                load_f32x4_neon(&[rs_hi[2], rs_hi[2], rs_hi[3], rs_hi[3]]),
            ];
        }
        for (c, chunk) in chunks.iter().take(CHUNKS).enumerate() {
            for p in 0..4 {
                let mut acc = if is_first {
                    [vdupq_n_f32(0.0); 4]
                } else {
                    std::array::from_fn(|rp| load_f32x4_neon(&acc_buf[c][p][rp]))
                };
                for k_off in 0..tile_len {
                    let block_idx = k_base + k_off;
                    if k_off + 2 < tile_len {
                        prefetch_l1(chunk[2 * p][block_idx + 2].values().as_ptr().cast::<u8>());
                        prefetch_l1(
                            chunk[2 * p + 1][block_idx + 2]
                                .values()
                                .as_ptr()
                                .cast::<u8>(),
                        );
                    }
                    let act0 = &chunk[2 * p][block_idx];
                    let act1 = &chunk[2 * p + 1][block_idx];
                    let a0_halves = act0.values().as_chunks::<16>().0;
                    let a1_halves = act1.values().as_chunks::<16>().0;
                    let (a_h0_lo, a_h0_hi) =
                        pair_s8_lo_hi(load_s8x16(&a0_halves[0]), load_s8x16(&a1_halves[0]));
                    let (a_h1_lo, a_h1_hi) =
                        pair_s8_lo_hi(load_s8x16(&a0_halves[1]), load_s8x16(&a1_halves[1]));
                    let as_vec =
                        load_f32x4_neon(&[act0.scale(), act1.scale(), act0.scale(), act1.scale()]);
                    for ((&w_packed, &rs), acc_reg) in wp_tile[k_off]
                        .iter()
                        .zip(rs_tile[k_off].iter())
                        .zip(acc.iter_mut())
                    {
                        let mut int_dot = smmla_s32(zero_s32, w_packed[0], a_h0_lo);
                        int_dot = smmla_s32(int_dot, w_packed[1], a_h0_hi);
                        int_dot = smmla_s32(int_dot, w_packed[2], a_h1_lo);
                        int_dot = smmla_s32(int_dot, w_packed[3], a_h1_hi);
                        let d = vmulq_f32(rs, as_vec);
                        *acc_reg = f32x4_scale_step::<false>(*acc_reg, d, vcvtq_f32_s32(int_dot));
                    }
                }
                if is_last {
                    for (rp, &acc_reg) in acc.iter().enumerate() {
                        let mut vals = [0.0_f32; 4];
                        store_f32x4_neon(acc_reg, &mut vals);
                        let r0 = rp * 2;
                        let r1 = r0 + 1;
                        out_buf[c][r0][2 * p] = vals[0];
                        out_buf[c][r0][2 * p + 1] = vals[1];
                        out_buf[c][r1][2 * p] = vals[2];
                        out_buf[c][r1][2 * p + 1] = vals[3];
                    }
                } else {
                    for (rp, &acc_reg) in acc.iter().enumerate() {
                        store_f32x4_neon(acc_reg, &mut acc_buf[c][p][rp]);
                    }
                }
            }
        }
    }

    if !out_buf.iter().flatten().flatten().all(|v| v.is_finite()) {
        return None;
    }
    for (dst, src) in outputs.iter_mut().zip(out_buf.iter()) {
        *dst = *src;
    }
    Some(())
}

#[target_feature(enable = "neon,dotprod")]
fn q8_0_scaled_8x8_chunks_dotprod<const CHUNKS: usize, A: super::Q8ActivationView>(
    rows: [&[u8]; 8],
    chunks: &[[&[A]; 8]],
    outputs: &mut [[[f32; 8]; 8]],
) -> Option<()> {
    use core::arch::aarch64::{vcvtq_f32_s32, vdupq_n_f32, vmulq_f32};

    let block_count = chunks[0][0].len();
    let row_chunks = rows.map(|row| row.as_chunks::<34>().0);
    let mut acc_buf = [[[[0.0_f32; 4]; 8]; 2]; CHUNKS];

    for block_idx in 0..block_count {
        prefetch_row_blocks(&row_chunks, block_idx, block_count);
        let blocks: [&[u8; 34]; 8] = std::array::from_fn(|r| &row_chunks[r][block_idx]);
        let [rs_lo, rs_hi] = widen_row_scales_f16(&blocks)?;
        let halves = [
            ([blocks[0], blocks[1], blocks[2], blocks[3]], rs_lo),
            ([blocks[4], blocks[5], blocks[6], blocks[7]], rs_hi),
        ];
        for (row_half, (row_blocks, row_scales)) in halves.into_iter().enumerate() {
            let rs_vec = load_f32x4_neon(&row_scales);
            let w_pairs: [(int8x16_t, int8x16_t); 4] = row_blocks.map(|block| {
                let h = block[2..34].as_chunks::<16>().0;
                (
                    vreinterpretq_s8_u8(load_u8x16(&h[0])),
                    vreinterpretq_s8_u8(load_u8x16(&h[1])),
                )
            });
            for (c, chunk) in chunks.iter().take(CHUNKS).enumerate() {
                for (t, tok) in chunk.iter().enumerate() {
                    let act = &tok[block_idx];
                    let a_halves = act.values().as_chunks::<16>().0;
                    let a0 = load_s8x16(&a_halves[0]);
                    let a1 = load_s8x16(&a_halves[1]);
                    let d0 = sdot32_vec(w_pairs[0].0, w_pairs[0].1, a0, a1);
                    let d1 = sdot32_vec(w_pairs[1].0, w_pairs[1].1, a0, a1);
                    let d2 = sdot32_vec(w_pairs[2].0, w_pairs[2].1, a0, a1);
                    let d3 = sdot32_vec(w_pairs[3].0, w_pairs[3].1, a0, a1);
                    let dots = vcvtq_f32_s32(reduce4_s32x4_addp(d0, d1, d2, d3));
                    let d = vmulq_f32(rs_vec, vdupq_n_f32(act.scale()));
                    let cur = load_f32x4_neon(&acc_buf[c][row_half][t]);
                    let next = f32x4_scale_step::<false>(cur, d, dots);
                    store_f32x4_neon(next, &mut acc_buf[c][row_half][t]);
                }
            }
        }
    }

    let mut out_buf = [[[0.0_f32; 8]; 8]; CHUNKS];
    for (chunk_acc, chunk_out) in acc_buf.iter().zip(out_buf.iter_mut()) {
        for (row_half, half_acc) in chunk_acc.iter().enumerate() {
            let rbase = row_half * 4;
            for (t, vals) in half_acc.iter().enumerate() {
                for r in 0..4 {
                    chunk_out[rbase + r][t] = vals[r];
                }
            }
        }
    }
    if !out_buf.iter().flatten().flatten().all(|v| v.is_finite()) {
        return None;
    }
    for (dst, src) in outputs.iter_mut().zip(out_buf.iter()) {
        *dst = *src;
    }
    Some(())
}

#[inline]
pub(super) fn q8_0_detected_dot_scaled_8x8_chunks<A: super::Q8ActivationView>(
    detected: super::Q8Dotprod,
    rows: [&[u8]; 8],
    chunks: &[[&[A]; 8]],
    outputs: &mut [[[f32; 8]; 8]],
) -> Option<()> {
    if detected.i8mm {
        match chunks.len() {
            // SAFETY: `detected.i8mm` was established by `has_i8mm()` and baseline `neon` on AArch64;
            // the safe caller validated row/chunk extents and finite activation scales.
            1 => unsafe { q8_0_scaled_8x8_chunks_i8mm::<1, A>(rows, chunks, outputs) },
            // SAFETY: see arm `1` above.
            2 => unsafe { q8_0_scaled_8x8_chunks_i8mm::<2, A>(rows, chunks, outputs) },
            // SAFETY: see arm `1` above.
            3 => unsafe { q8_0_scaled_8x8_chunks_i8mm::<3, A>(rows, chunks, outputs) },
            // SAFETY: see arm `1` above.
            4 => unsafe { q8_0_scaled_8x8_chunks_i8mm::<4, A>(rows, chunks, outputs) },
            // SAFETY: see arm `1` above.
            5 => unsafe { q8_0_scaled_8x8_chunks_i8mm::<5, A>(rows, chunks, outputs) },
            // SAFETY: see arm `1` above.
            6 => unsafe { q8_0_scaled_8x8_chunks_i8mm::<6, A>(rows, chunks, outputs) },
            // SAFETY: see arm `1` above.
            7 => unsafe { q8_0_scaled_8x8_chunks_i8mm::<7, A>(rows, chunks, outputs) },
            // SAFETY: see arm `1` above.
            8 => unsafe { q8_0_scaled_8x8_chunks_i8mm::<8, A>(rows, chunks, outputs) },
            _ => None,
        }
    } else if detected.dotprod {
        match chunks.len() {
            // SAFETY: `detected.dotprod` was established by `has_dotprod()` and baseline `neon` on AArch64;
            // the safe caller validated row/chunk extents and finite activation scales.
            1 => unsafe { q8_0_scaled_8x8_chunks_dotprod::<1, A>(rows, chunks, outputs) },
            // SAFETY: see arm `1` above.
            2 => unsafe { q8_0_scaled_8x8_chunks_dotprod::<2, A>(rows, chunks, outputs) },
            // SAFETY: see arm `1` above.
            3 => unsafe { q8_0_scaled_8x8_chunks_dotprod::<3, A>(rows, chunks, outputs) },
            // SAFETY: see arm `1` above.
            4 => unsafe { q8_0_scaled_8x8_chunks_dotprod::<4, A>(rows, chunks, outputs) },
            // SAFETY: see arm `1` above.
            5 => unsafe { q8_0_scaled_8x8_chunks_dotprod::<5, A>(rows, chunks, outputs) },
            // SAFETY: see arm `1` above.
            6 => unsafe { q8_0_scaled_8x8_chunks_dotprod::<6, A>(rows, chunks, outputs) },
            // SAFETY: see arm `1` above.
            7 => unsafe { q8_0_scaled_8x8_chunks_dotprod::<7, A>(rows, chunks, outputs) },
            // SAFETY: see arm `1` above.
            8 => unsafe { q8_0_scaled_8x8_chunks_dotprod::<8, A>(rows, chunks, outputs) },
            _ => None,
        }
    } else {
        None
    }
}

#[cfg(target_feature = "neon")]
#[inline]
#[target_feature(enable = "neon")]
fn load_f32x4(values: &[f32; 4]) -> core::arch::aarch64::float32x4_t {
    // SAFETY: `values` is a valid 4-element `f32` array reference and `vld1q_f32` requires no alignment.
    unsafe { core::arch::aarch64::vld1q_f32(values.as_ptr()) }
}

#[cfg(target_feature = "neon")]
#[inline]
#[target_feature(enable = "neon")]
fn store_u16x4(vec: core::arch::aarch64::uint16x4_t, out: &mut [u16; 4]) {
    // SAFETY: `out` is a valid mutable 4-element `u16` array reference and `vst1_u16` requires no alignment.
    unsafe { core::arch::aarch64::vst1_u16(out.as_mut_ptr(), vec) };
}

#[cfg(target_feature = "neon")]
#[inline]
#[target_feature(enable = "neon")]
fn store_u16x8(vec: core::arch::aarch64::uint16x8_t, out: &mut [u16; 8]) {
    // SAFETY: `out` is a valid mutable 8-element `u16` array reference and `vst1q_u16` requires no alignment.
    unsafe { core::arch::aarch64::vst1q_u16(out.as_mut_ptr(), vec) };
}

#[cfg(target_feature = "neon")]
#[inline]
pub(super) fn narrow_f32_four(values: [f32; 4]) -> [u16; 4] {
    // SAFETY: compiled only with baseline AArch64 NEON; fixed-array extents cover all accesses.
    unsafe { narrow_f32_four_neon(values) }
}

#[cfg(target_feature = "neon")]
#[inline]
#[target_feature(enable = "neon")]
fn narrow_f32_four_neon(values: [f32; 4]) -> [u16; 4] {
    use core::arch::aarch64::{
        vandq_u32, vcvt_f16_f32, vdupq_n_u32, vmaxvq_u32, vreinterpret_u16_f16,
        vreinterpretq_u32_f32,
    };
    let raw = load_f32x4(&values);
    let abs_bits = vandq_u32(vreinterpretq_u32_f32(raw), vdupq_n_u32(0x7fff_ffff));
    if vmaxvq_u32(abs_bits) > 0x7f80_0000 {
        return values.map(bonsai_ternary::float_to_half);
    }
    let mut out = [0_u16; 4];
    store_u16x4(vreinterpret_u16_f16(vcvt_f16_f32(raw)), &mut out);
    out
}

#[cfg(target_feature = "neon")]
#[inline]
pub(super) fn narrow_f32_eight(values: &[f32; 8]) -> [u16; 8] {
    // SAFETY: compiled only with baseline AArch64 NEON; fixed-array extents cover all accesses.
    unsafe { narrow_f32_eight_neon(values) }
}

#[cfg(target_feature = "neon")]
#[inline]
#[target_feature(enable = "neon")]
fn narrow_f32_eight_neon(values: &[f32; 8]) -> [u16; 8] {
    use core::arch::aarch64::{
        vandq_u32, vcvt_f16_f32, vcvt_high_f16_f32, vdupq_n_u32, vmaxq_u32, vmaxvq_u32,
        vreinterpretq_u16_f16, vreinterpretq_u32_f32,
    };
    let (halves, _) = values.as_chunks::<4>();
    let lo = load_f32x4(&halves[0]);
    let hi = load_f32x4(&halves[1]);
    let mask = vdupq_n_u32(0x7fff_ffff);
    let abs_lo = vandq_u32(vreinterpretq_u32_f32(lo), mask);
    let abs_hi = vandq_u32(vreinterpretq_u32_f32(hi), mask);
    if vmaxvq_u32(vmaxq_u32(abs_lo, abs_hi)) > 0x7f80_0000 {
        return values.map(bonsai_ternary::float_to_half);
    }
    let half_lo = vcvt_f16_f32(lo);
    let half = vcvt_high_f16_f32(half_lo, hi);
    let mut out = [0_u16; 8];
    store_u16x8(vreinterpretq_u16_f16(half), &mut out);
    out
}

#[cfg(target_feature = "neon")]
#[inline]
pub(super) fn narrow_f32_slice_checked(src: &[f32], dst: &mut [u16]) -> bool {
    // SAFETY: compiled only with baseline AArch64 NEON; `as_chunks` bounds all array accesses.
    unsafe { narrow_f32_slice_checked_neon(src, dst) }
}

#[cfg(target_feature = "neon")]
#[target_feature(enable = "neon")]
fn narrow_f32_slice_checked_neon(src: &[f32], dst: &mut [u16]) -> bool {
    use core::arch::aarch64::{
        vandq_u32, vcvt_f16_f32, vcvt_high_f16_f32, vdupq_n_u32, vmaxq_u32, vmaxvq_u32,
        vreinterpret_u16_f16, vreinterpretq_u16_f16, vreinterpretq_u32_f32,
    };
    const OVERFLOW_BOUND: u32 = 0x477f_f000;
    if src.len() != dst.len() {
        return false;
    }
    let abs_mask = vdupq_n_u32(0x7fff_ffff);
    let (src_chunks, src_tail) = src.as_chunks::<8>();
    let (dst_chunks, dst_tail) = dst.as_chunks_mut::<8>();
    for (s, d) in src_chunks.iter().zip(dst_chunks.iter_mut()) {
        let (halves, _) = s.as_chunks::<4>();
        let lo = load_f32x4(&halves[0]);
        let hi = load_f32x4(&halves[1]);
        let abs_lo = vandq_u32(vreinterpretq_u32_f32(lo), abs_mask);
        let abs_hi = vandq_u32(vreinterpretq_u32_f32(hi), abs_mask);
        if vmaxvq_u32(vmaxq_u32(abs_lo, abs_hi)) >= OVERFLOW_BOUND {
            return false;
        }
        let half = vcvt_high_f16_f32(vcvt_f16_f32(lo), hi);
        store_u16x8(vreinterpretq_u16_f16(half), d);
    }
    let (src_four, src_rem) = src_tail.as_chunks::<4>();
    let (dst_four, dst_rem) = dst_tail.as_chunks_mut::<4>();
    if let (Some(s), Some(d)) = (src_four.first(), dst_four.first_mut()) {
        let v = load_f32x4(s);
        let abs = vandq_u32(vreinterpretq_u32_f32(v), abs_mask);
        if vmaxvq_u32(abs) >= OVERFLOW_BOUND {
            return false;
        }
        store_u16x4(vreinterpret_u16_f16(vcvt_f16_f32(v)), d);
    }
    for (&value, slot) in src_rem.iter().zip(dst_rem.iter_mut()) {
        let Some(bits) = bonsai_ternary::checked_float_to_half(value) else {
            return false;
        };
        *slot = bits;
    }
    true
}

#[cfg(target_feature = "neon")]
#[inline]
#[target_feature(enable = "neon")]
fn accum_8lanes_16(
    acc_lo: &mut int32x4_t,
    acc_hi: &mut int32x4_t,
    w_vec: int8x16_t,
    a_vec: int8x16_t,
    scale_i16: i16,
) {
    use core::arch::aarch64::{
        vdupq_n_s16, vget_high_s16, vget_low_s16, vmlal_high_s8, vmlal_s16, vmull_s8,
    };
    let prod0 = vmull_s8(vget_low_s8(w_vec), vget_low_s8(a_vec));
    let sum16 = vmlal_high_s8(prod0, w_vec, a_vec);
    let scale_vec = vdupq_n_s16(scale_i16);
    *acc_lo = vmlal_s16(*acc_lo, vget_low_s16(sum16), vget_low_s16(scale_vec));
    *acc_hi = vmlal_s16(*acc_hi, vget_high_s16(sum16), vget_low_s16(scale_vec));
}

#[cfg(target_feature = "neon")]
#[inline]
pub(super) fn dot_kquant_8lanes_q4_k(
    block: &[u8; 144],
    values: &[i8; 256],
    sums: &[i16; 16],
) -> ([i32; 8], i32) {
    // SAFETY: compiled only with baseline AArch64 NEON; fixed-size array extents guarantee valid bounds.
    unsafe { dot_kquant_8lanes_q4_k_neon(block, values, sums) }
}

#[cfg(target_feature = "neon")]
#[target_feature(enable = "neon")]
fn dot_kquant_8lanes_q4_k_neon(
    block: &[u8; 144],
    values: &[i8; 256],
    sums: &[i16; 16],
) -> ([i32; 8], i32) {
    use core::arch::aarch64::{
        vaddq_s16, vdupq_n_s16, vget_high_s16, vget_low_s16, vmlal_high_s8, vmlal_s16, vmull_s8,
    };
    let mut acc_lo = vdupq_n_s32(0);
    let mut acc_hi = vdupq_n_s32(0);
    let mut min_sum = 0_i32;
    let Ok(scales) = <&[u8; 12]>::try_from(&block[4..16]) else {
        return ([0; 8], 0);
    };
    let a_groups = values.as_chunks::<16>().0;
    let mask_0f = vdupq_n_u8(0x0f);
    for chunk in 0..4 {
        let group_lo = chunk * 2;
        let group_hi = group_lo + 1;
        let (sc_lo, min_lo) = super::kquants::unpack_q4_k_scale_min(group_lo, scales);
        let (sc_hi, min_hi) = super::kquants::unpack_q4_k_scale_min(group_hi, scales);
        min_sum +=
            i32::from(min_lo) * (i32::from(sums[2 * group_lo]) + i32::from(sums[2 * group_lo + 1]));
        min_sum +=
            i32::from(min_hi) * (i32::from(sums[2 * group_hi]) + i32::from(sums[2 * group_hi + 1]));

        let qs_off = 16 + chunk * 32;
        let raw0 = load_u8x16(&block[qs_off..qs_off + 16].as_chunks::<16>().0[0]);
        let raw1 = load_u8x16(&block[qs_off + 16..qs_off + 32].as_chunks::<16>().0[0]);

        let lo0 = vreinterpretq_s8_u8(vandq_u8(raw0, mask_0f));
        let lo1 = vreinterpretq_s8_u8(vandq_u8(raw1, mask_0f));
        let a_lo0 = load_s8x16(&a_groups[chunk * 4]);
        let a_lo1 = load_s8x16(&a_groups[chunk * 4 + 1]);
        let sum_lo0 = vmlal_high_s8(vmull_s8(vget_low_s8(lo0), vget_low_s8(a_lo0)), lo0, a_lo0);
        let sum_lo1 = vmlal_high_s8(vmull_s8(vget_low_s8(lo1), vget_low_s8(a_lo1)), lo1, a_lo1);
        let sum_lo = vaddq_s16(sum_lo0, sum_lo1);
        let sc_lo_vec = vdupq_n_s16(i16::from(sc_lo));
        acc_lo = vmlal_s16(acc_lo, vget_low_s16(sum_lo), vget_low_s16(sc_lo_vec));
        acc_hi = vmlal_s16(acc_hi, vget_high_s16(sum_lo), vget_low_s16(sc_lo_vec));

        let hi0 = vreinterpretq_s8_u8(vshrq_n_u8(raw0, 4));
        let hi1 = vreinterpretq_s8_u8(vshrq_n_u8(raw1, 4));
        let a_hi0 = load_s8x16(&a_groups[chunk * 4 + 2]);
        let a_hi1 = load_s8x16(&a_groups[chunk * 4 + 3]);
        let sum_hi0 = vmlal_high_s8(vmull_s8(vget_low_s8(hi0), vget_low_s8(a_hi0)), hi0, a_hi0);
        let sum_hi1 = vmlal_high_s8(vmull_s8(vget_low_s8(hi1), vget_low_s8(a_hi1)), hi1, a_hi1);
        let sum_hi = vaddq_s16(sum_hi0, sum_hi1);
        let sc_hi_vec = vdupq_n_s16(i16::from(sc_hi));
        acc_lo = vmlal_s16(acc_lo, vget_low_s16(sum_hi), vget_low_s16(sc_hi_vec));
        acc_hi = vmlal_s16(acc_hi, vget_high_s16(sum_hi), vget_low_s16(sc_hi_vec));
    }
    let mut lo_out = [0_i32; 4];
    let mut hi_out = [0_i32; 4];
    store_s32x4(acc_lo, &mut lo_out);
    store_s32x4(acc_hi, &mut hi_out);
    (
        [
            lo_out[0], lo_out[1], lo_out[2], lo_out[3], hi_out[0], hi_out[1], hi_out[2], hi_out[3],
        ],
        min_sum,
    )
}

#[cfg(target_feature = "neon")]
#[inline]
pub(super) fn dot_kquant_8lanes_q6_k(block: &[u8; 210], values: &[i8; 256]) -> [i32; 8] {
    // SAFETY: compiled only with baseline AArch64 NEON; fixed-size array extents guarantee valid bounds.
    unsafe { dot_kquant_8lanes_q6_k_neon(block, values) }
}

#[cfg(target_feature = "neon")]
#[target_feature(enable = "neon")]
fn dot_kquant_8lanes_q6_k_neon(block: &[u8; 210], values: &[i8; 256]) -> [i32; 8] {
    let mut acc_lo = vdupq_n_s32(0);
    let mut acc_hi = vdupq_n_s32(0);
    let a_groups = values.as_chunks::<16>().0;
    for half_idx in 0..2 {
        for quarter in 0..4 {
            let g_base = half_idx * 8 + quarter * 2;
            let (w0, w1, s0, s1) = unpack_q6_k_32(block, half_idx, quarter);
            #[expect(
                clippy::cast_possible_truncation,
                clippy::as_conversions,
                reason = "s0 originates from a signed i8 Q6_K sub-scale in -128..=127."
            )]
            accum_8lanes_16(
                &mut acc_lo,
                &mut acc_hi,
                w0,
                load_s8x16(&a_groups[g_base]),
                s0 as i16,
            );
            #[expect(
                clippy::cast_possible_truncation,
                clippy::as_conversions,
                reason = "s1 originates from a signed i8 Q6_K sub-scale in -128..=127."
            )]
            accum_8lanes_16(
                &mut acc_lo,
                &mut acc_hi,
                w1,
                load_s8x16(&a_groups[g_base + 1]),
                s1 as i16,
            );
        }
    }
    let mut lo_out = [0_i32; 4];
    let mut hi_out = [0_i32; 4];
    store_s32x4(acc_lo, &mut lo_out);
    store_s32x4(acc_hi, &mut hi_out);
    [
        lo_out[0], lo_out[1], lo_out[2], lo_out[3], hi_out[0], hi_out[1], hi_out[2], hi_out[3],
    ]
}

#[cfg(target_feature = "neon")]
#[inline]
#[target_feature(enable = "neon")]
fn unpack_q6_k_32(
    block: &[u8; 210],
    half_idx: usize,
    quarter: usize,
) -> (int8x16_t, int8x16_t, i32, i32) {
    use core::arch::aarch64::{vdupq_n_s8, vorrq_u8, vshlq_n_u8};
    let low_off = half_idx * 64 + (quarter % 2) * 32;
    let high_off = 128 + half_idx * 32;
    let ql0 = load_u8x16(&block[low_off..low_off + 16].as_chunks::<16>().0[0]);
    let ql1 = load_u8x16(&block[low_off + 16..low_off + 32].as_chunks::<16>().0[0]);
    let low0 = if quarter < 2 {
        vandq_u8(ql0, vdupq_n_u8(0x0f))
    } else {
        vshrq_n_u8(ql0, 4)
    };
    let low1 = if quarter < 2 {
        vandq_u8(ql1, vdupq_n_u8(0x0f))
    } else {
        vshrq_n_u8(ql1, 4)
    };
    let qh0 = load_u8x16(&block[high_off..high_off + 16].as_chunks::<16>().0[0]);
    let qh1 = load_u8x16(&block[high_off + 16..high_off + 32].as_chunks::<16>().0[0]);
    let mask_03 = vdupq_n_u8(0x03);
    let hi0 = match quarter {
        0 => vandq_u8(qh0, mask_03),
        1 => vandq_u8(vshrq_n_u8(qh0, 2), mask_03),
        2 => vandq_u8(vshrq_n_u8(qh0, 4), mask_03),
        _ => vandq_u8(vshrq_n_u8(qh0, 6), mask_03),
    };
    let hi1 = match quarter {
        0 => vandq_u8(qh1, mask_03),
        1 => vandq_u8(vshrq_n_u8(qh1, 2), mask_03),
        2 => vandq_u8(vshrq_n_u8(qh1, 4), mask_03),
        _ => vandq_u8(vshrq_n_u8(qh1, 6), mask_03),
    };
    let bias = vdupq_n_s8(32);
    let w0 = vsubq_s8(
        vreinterpretq_s8_u8(vorrq_u8(low0, vshlq_n_u8(hi0, 4))),
        bias,
    );
    let w1 = vsubq_s8(
        vreinterpretq_s8_u8(vorrq_u8(low1, vshlq_n_u8(hi1, 4))),
        bias,
    );
    let s_base = 192 + half_idx * 8 + quarter * 2;
    let s0 = i32::from(block[s_base].cast_signed());
    let s1 = i32::from(block[s_base + 1].cast_signed());
    (w0, w1, s0, s1)
}

#[cfg(target_feature = "neon")]
pub(super) fn quantize_row_q8_0(input: &[f32], out: &mut [super::kquants::Q8_0ActBlock]) {
    // SAFETY: compiled only with baseline AArch64 NEON.
    unsafe { quantize_row_q8_0_neon(input, out) }
}

#[cfg(target_feature = "neon")]
#[target_feature(enable = "neon")]
fn quantize_row_q8_0_neon(input: &[f32], out: &mut [super::kquants::Q8_0ActBlock]) {
    use core::arch::aarch64::{
        vabsq_f32, vcombine_s16, vcvtnq_s32_f32, vdupq_n_f32, vdupq_n_s8, vmaxq_f32, vmaxq_s8,
        vmaxvq_f32, vmulq_f32, vqmovn_s16, vqmovn_s32,
    };
    let (chunks, _) = input.as_chunks::<32>();
    for (src, block) in chunks.iter().zip(out.iter_mut()) {
        let quads = src.as_chunks::<4>().0;
        let max_v0 = vmaxq_f32(
            vmaxq_f32(
                vabsq_f32(load_f32x4(&quads[0])),
                vabsq_f32(load_f32x4(&quads[1])),
            ),
            vmaxq_f32(
                vabsq_f32(load_f32x4(&quads[2])),
                vabsq_f32(load_f32x4(&quads[3])),
            ),
        );
        let max_v1 = vmaxq_f32(
            vmaxq_f32(
                vabsq_f32(load_f32x4(&quads[4])),
                vabsq_f32(load_f32x4(&quads[5])),
            ),
            vmaxq_f32(
                vabsq_f32(load_f32x4(&quads[6])),
                vabsq_f32(load_f32x4(&quads[7])),
            ),
        );
        let amax = vmaxvq_f32(vmaxq_f32(max_v0, max_v1));
        if amax == 0.0 || !amax.is_finite() {
            block.d = 0.0;
            block.qs = [0; 32];
            continue;
        }
        let d_val = amax / 127.0_f32;
        let id_v = vdupq_n_f32(127.0_f32 / amax);
        let min_q = vdupq_n_s8(-127);
        let (out_halves, _) = block.qs.as_chunks_mut::<16>();
        for (half_idx, out_half) in out_halves.iter_mut().enumerate() {
            let base = half_idx * 4;
            let q0 = vcvtnq_s32_f32(vmulq_f32(load_f32x4(&quads[base]), id_v));
            let q1 = vcvtnq_s32_f32(vmulq_f32(load_f32x4(&quads[base + 1]), id_v));
            let q2 = vcvtnq_s32_f32(vmulq_f32(load_f32x4(&quads[base + 2]), id_v));
            let q3 = vcvtnq_s32_f32(vmulq_f32(load_f32x4(&quads[base + 3]), id_v));
            let i16_01 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
            let i16_23 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
            let packed = vmaxq_s8(vcombine_s8(vqmovn_s16(i16_01), vqmovn_s16(i16_23)), min_q);
            store_s8x16(packed, out_half);
        }
        block.d = d_val;
    }
}

#[cfg(target_feature = "neon")]
pub(super) fn quantize_row_q8_k(input: &[f32], out: &mut [super::kquants::Q8_KActBlock]) {
    // SAFETY: compiled only with baseline AArch64 NEON.
    unsafe { quantize_row_q8_k_neon(input, out) }
}

#[cfg(target_feature = "neon")]
#[target_feature(enable = "neon")]
fn quantize_row_q8_k_neon(input: &[f32], out: &mut [super::kquants::Q8_KActBlock]) {
    use core::arch::aarch64::{
        vabsq_f32, vaddvq_s16, vcombine_s16, vcvtnq_s32_f32, vdupq_n_f32, vdupq_n_s8, vmaxq_f32,
        vmaxq_s8, vmaxvq_f32, vmulq_f32, vqmovn_s16, vqmovn_s32,
    };
    let (chunks, _) = input.as_chunks::<256>();
    for (src, block) in chunks.iter().zip(out.iter_mut()) {
        let quads = src.as_chunks::<4>().0;
        let mut max_v0 = vdupq_n_f32(0.0);
        let mut max_v1 = vdupq_n_f32(0.0);
        for pair in quads.as_chunks::<2>().0 {
            max_v0 = vmaxq_f32(max_v0, vabsq_f32(load_f32x4(&pair[0])));
            max_v1 = vmaxq_f32(max_v1, vabsq_f32(load_f32x4(&pair[1])));
        }
        let amax = vmaxvq_f32(vmaxq_f32(max_v0, max_v1));
        if amax == 0.0 || !amax.is_finite() {
            block.d = 0.0;
            block.qs = [0; 256];
            block.bsums = [0; 16];
            continue;
        }
        let d_val = amax / 127.0_f32;
        let id_v = vdupq_n_f32(127.0_f32 / amax);
        let min_q = vdupq_n_s8(-127);
        let qs_groups = block.qs.as_chunks_mut::<16>().0;
        for (group_idx, (q_out, bsum_slot)) in
            qs_groups.iter_mut().zip(block.bsums.iter_mut()).enumerate()
        {
            let base = group_idx * 4;
            let q0 = vcvtnq_s32_f32(vmulq_f32(load_f32x4(&quads[base]), id_v));
            let q1 = vcvtnq_s32_f32(vmulq_f32(load_f32x4(&quads[base + 1]), id_v));
            let q2 = vcvtnq_s32_f32(vmulq_f32(load_f32x4(&quads[base + 2]), id_v));
            let q3 = vcvtnq_s32_f32(vmulq_f32(load_f32x4(&quads[base + 3]), id_v));
            let i16_01 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
            let i16_23 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
            let packed = vmaxq_s8(vcombine_s8(vqmovn_s16(i16_01), vqmovn_s16(i16_23)), min_q);
            store_s8x16(packed, q_out);
            *bsum_slot = vaddvq_s16(vpaddlq_s8(packed));
        }
        block.d = d_val;
    }
}

pub(super) fn quant_dot_q8_0_row(
    w_row: &[u8],
    act: &[super::kquants::Q8_0ActBlock],
) -> Option<f32> {
    if !has_dotprod() {
        return None;
    }
    // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
    Some(unsafe { quant_dot_q8_0_row_dotprod(w_row, act) })
}

#[target_feature(enable = "neon,dotprod")]
fn quant_dot_q8_0_row_dotprod(w_row: &[u8], act: &[super::kquants::Q8_0ActBlock]) -> f32 {
    let mut sum = 0.0_f32;
    for (wb, ab) in w_row.as_chunks::<34>().0.iter().zip(act.iter()) {
        let w_d = super::kquants::f16_bits_to_f32([wb[0], wb[1]]);
        let a_halves = ab.qs.as_chunks::<16>().0;
        let q0 = vreinterpretq_s8_u8(load_u8x16(&wb[2..18].as_chunks::<16>().0[0]));
        let q1 = vreinterpretq_s8_u8(load_u8x16(&wb[18..34].as_chunks::<16>().0[0]));
        let dot = vaddvq_s32(sdot32_vec(
            q0,
            q1,
            load_s8x16(&a_halves[0]),
            load_s8x16(&a_halves[1]),
        ));
        #[expect(
            clippy::suboptimal_flops,
            reason = "Matches non-FMA Q8_0 accumulation semantics."
        )]
        {
            sum += w_d * (ab.d * super::kquants::i32_to_f32(dot));
        }
    }
    sum
}

pub(super) fn quant_dot_q8_0_4rows(
    rows: [&[u8]; 4],
    act: &[super::kquants::Q8_0ActBlock],
) -> Option<[f32; 4]> {
    if !has_dotprod() {
        return None;
    }
    // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
    Some(unsafe { quant_dot_q8_0_4rows_dotprod(rows, act) })
}

#[target_feature(enable = "neon,dotprod")]
fn quant_dot_q8_0_4rows_dotprod(
    rows: [&[u8]; 4],
    act: &[super::kquants::Q8_0ActBlock],
) -> [f32; 4] {
    let mut acc = [0.0_f32; 4];
    let r0 = rows[0].as_chunks::<34>().0;
    let r1 = rows[1].as_chunks::<34>().0;
    let r2 = rows[2].as_chunks::<34>().0;
    let r3 = rows[3].as_chunks::<34>().0;
    for (idx, ab) in act.iter().enumerate() {
        let a_halves = ab.qs.as_chunks::<16>().0;
        let a0 = load_s8x16(&a_halves[0]);
        let a1 = load_s8x16(&a_halves[1]);
        let blocks = [&r0[idx], &r1[idx], &r2[idx], &r3[idx]];
        let d_bits: [u16; 4] = blocks.map(|b| u16::from_le_bytes([b[0], b[1]]));
        let d_vals = super::widen_f16_four(&d_bits);
        for row_idx in 0..4 {
            let blk = blocks[row_idx];
            let q0 = vreinterpretq_s8_u8(load_u8x16(&blk[2..18].as_chunks::<16>().0[0]));
            let q1 = vreinterpretq_s8_u8(load_u8x16(&blk[18..34].as_chunks::<16>().0[0]));
            let dot = vaddvq_s32(sdot32_vec(q0, q1, a0, a1));
            #[expect(
                clippy::suboptimal_flops,
                reason = "Matches non-FMA Q8_0 accumulation semantics."
            )]
            {
                acc[row_idx] += d_vals[row_idx] * (ab.d * super::kquants::i32_to_f32(dot));
            }
        }
    }
    acc
}

pub(super) fn quant_gemm_q8_0_4x2(
    rows: [&[u8]; 4],
    act0: &[super::kquants::Q8_0ActBlock],
    act1: &[super::kquants::Q8_0ActBlock],
) -> Option<[[f32; 2]; 4]> {
    if !has_i8mm() {
        return None;
    }
    // SAFETY: `has_i8mm()` verified that the CPU supports `i8mm`, `dotprod`, and baseline `neon`.
    Some(unsafe { quant_gemm_q8_0_4x2_i8mm(rows, act0, act1) })
}

#[target_feature(enable = "neon,i8mm")]
fn quant_gemm_q8_0_4x2_i8mm(
    rows: [&[u8]; 4],
    act0: &[super::kquants::Q8_0ActBlock],
    act1: &[super::kquants::Q8_0ActBlock],
) -> [[f32; 2]; 4] {
    let mut acc = [[0.0_f32; 2]; 4];
    let r0 = rows[0].as_chunks::<34>().0;
    let r1 = rows[1].as_chunks::<34>().0;
    let r2 = rows[2].as_chunks::<34>().0;
    let r3 = rows[3].as_chunks::<34>().0;
    for (idx, (ab0, ab1)) in act0.iter().zip(act1.iter()).enumerate() {
        let blocks = [&r0[idx], &r1[idx], &r2[idx], &r3[idx]];
        let w_quants = blocks.map(|b| &b[2..34].as_chunks::<32>().0[0]);
        let tile = smmla_q8_0_4x2_i8mm(w_quants, [&ab0.qs, &ab1.qs]);
        let d_bits: [u16; 4] = blocks.map(|b| u16::from_le_bytes([b[0], b[1]]));
        let d_vals = super::widen_f16_four(&d_bits);
        for row_idx in 0..4 {
            #[expect(
                clippy::suboptimal_flops,
                reason = "Matches non-FMA Q8_0 accumulation semantics."
            )]
            {
                acc[row_idx][0] +=
                    d_vals[row_idx] * (ab0.d * super::kquants::i32_to_f32(tile[row_idx][0]));
                acc[row_idx][1] +=
                    d_vals[row_idx] * (ab1.d * super::kquants::i32_to_f32(tile[row_idx][1]));
            }
        }
    }
    acc
}

pub(super) fn quant_dot_q4_k_row(
    w_row: &[u8],
    act: &[super::kquants::Q8_KActBlock],
) -> Option<f32> {
    if !has_dotprod() {
        return None;
    }
    // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
    Some(unsafe { quant_dot_q4_k_row_dotprod(w_row, act) })
}

#[target_feature(enable = "neon,dotprod")]
fn quant_dot_q4_k_row_dotprod(w_row: &[u8], act: &[super::kquants::Q8_KActBlock]) -> f32 {
    let mut acc = 0.0_f64;
    let mask_0f = vdupq_n_u8(0x0f);
    for (block, ab) in w_row.as_chunks::<144>().0.iter().zip(act.iter()) {
        let scales = &block[4..16];
        let a_groups = ab.qs.as_chunks::<16>().0;
        let mut mint = 0_i32;
        let mut sumi = 0_i32;
        let d_act = f64::from(ab.d);
        let [d_w_f, dmin_w_f, _, _] = super::widen_f16_four(&[
            u16::from_le_bytes([block[0], block[1]]),
            u16::from_le_bytes([block[2], block[3]]),
            0,
            0,
        ]);
        let d_w = f64::from(d_w_f);
        let dmin_w = f64::from(dmin_w_f);
        let mut sc_arr = [0_i32; 8];
        for (group, sc_slot) in sc_arr.iter_mut().enumerate() {
            let (sc_val, min_val) = super::kquants::unpack_q4_k_scale_min(group, scales);
            *sc_slot = i32::from(sc_val);
            let b_pair = i32::from(ab.bsums[2 * group]) + i32::from(ab.bsums[2 * group + 1]);
            mint += i32::from(min_val) * b_pair;
        }
        for chunk in 0..4 {
            let qs_off = 16 + chunk * 32;
            let raw0 = load_u8x16(&block[qs_off..qs_off + 16].as_chunks::<16>().0[0]);
            let raw1 = load_u8x16(&block[qs_off + 16..qs_off + 32].as_chunks::<16>().0[0]);
            let lo0 = vreinterpretq_s8_u8(vandq_u8(raw0, mask_0f));
            let lo1 = vreinterpretq_s8_u8(vandq_u8(raw1, mask_0f));
            let hi0 = vreinterpretq_s8_u8(vshrq_n_u8(raw0, 4));
            let hi1 = vreinterpretq_s8_u8(vshrq_n_u8(raw1, 4));
            let d_lo = vaddvq_s32(sdot32_vec(
                lo0,
                lo1,
                load_s8x16(&a_groups[chunk * 4]),
                load_s8x16(&a_groups[chunk * 4 + 1]),
            ));
            let d_hi = vaddvq_s32(sdot32_vec(
                hi0,
                hi1,
                load_s8x16(&a_groups[chunk * 4 + 2]),
                load_s8x16(&a_groups[chunk * 4 + 3]),
            ));
            sumi += d_lo * sc_arr[chunk * 2] + d_hi * sc_arr[chunk * 2 + 1];
        }
        #[expect(
            clippy::suboptimal_flops,
            reason = "Matches non-FMA Q4_K block accumulation order."
        )]
        {
            acc += (d_w * d_act) * f64::from(sumi) - (dmin_w * d_act) * f64::from(mint);
        }
    }
    super::kquants::f64_to_f32(acc)
}

pub(super) fn quant_dot_q4_k_4rows(
    rows: [&[u8]; 4],
    act: &[super::kquants::Q8_KActBlock],
) -> Option<[f32; 4]> {
    if !has_dotprod() {
        return None;
    }
    // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
    Some(unsafe { quant_dot_q4_k_4rows_dotprod(rows, act) })
}

#[target_feature(enable = "neon,dotprod")]
fn quant_dot_q4_k_4rows_dotprod(
    rows: [&[u8]; 4],
    act: &[super::kquants::Q8_KActBlock],
) -> [f32; 4] {
    let mut acc = [0.0_f64; 4];
    let r0 = rows[0].as_chunks::<144>().0;
    let r1 = rows[1].as_chunks::<144>().0;
    let r2 = rows[2].as_chunks::<144>().0;
    let r3 = rows[3].as_chunks::<144>().0;
    let mask_0f = vdupq_n_u8(0x0f);
    for (idx, ab) in act.iter().enumerate() {
        let blocks = [&r0[idx], &r1[idx], &r2[idx], &r3[idx]];
        let a_groups = ab.qs.as_chunks::<16>().0;
        let bsum_pairs: [i32; 8] =
            std::array::from_fn(|g| i32::from(ab.bsums[2 * g]) + i32::from(ab.bsums[2 * g + 1]));
        let d_act = f64::from(ab.d);
        let d_f16: [u16; 8] = [
            u16::from_le_bytes([blocks[0][0], blocks[0][1]]),
            u16::from_le_bytes([blocks[1][0], blocks[1][1]]),
            u16::from_le_bytes([blocks[2][0], blocks[2][1]]),
            u16::from_le_bytes([blocks[3][0], blocks[3][1]]),
            u16::from_le_bytes([blocks[0][2], blocks[0][3]]),
            u16::from_le_bytes([blocks[1][2], blocks[1][3]]),
            u16::from_le_bytes([blocks[2][2], blocks[2][3]]),
            u16::from_le_bytes([blocks[3][2], blocks[3][3]]),
        ];
        let [d_weights, dmin_weights] = super::widen_f16_eight(&d_f16);
        let scales = blocks.map(|b| {
            let mut sc_arr = [0_i32; 8];
            let mut min_arr = [0_i32; 8];
            let raw_sc = &b[4..16];
            for g in 0..8 {
                let (sc_v, m_v) = super::kquants::unpack_q4_k_scale_min(g, raw_sc);
                sc_arr[g] = i32::from(sc_v);
                min_arr[g] = i32::from(m_v);
            }
            (sc_arr, min_arr)
        });
        let mut mint = [0_i32; 4];
        for r in 0..4 {
            let mut m_sum = 0_i32;
            for g in 0..8 {
                m_sum += scales[r].1[g] * bsum_pairs[g];
            }
            mint[r] = m_sum;
        }
        let mut sumi = [0_i32; 4];
        for chunk in 0..4 {
            let a_lo0 = load_s8x16(&a_groups[chunk * 4]);
            let a_lo1 = load_s8x16(&a_groups[chunk * 4 + 1]);
            let a_hi0 = load_s8x16(&a_groups[chunk * 4 + 2]);
            let a_hi1 = load_s8x16(&a_groups[chunk * 4 + 3]);
            let qs_off = 16 + chunk * 32;
            for r in 0..4 {
                let raw0 = load_u8x16(&blocks[r][qs_off..qs_off + 16].as_chunks::<16>().0[0]);
                let raw1 = load_u8x16(&blocks[r][qs_off + 16..qs_off + 32].as_chunks::<16>().0[0]);
                let lo0 = vreinterpretq_s8_u8(vandq_u8(raw0, mask_0f));
                let lo1 = vreinterpretq_s8_u8(vandq_u8(raw1, mask_0f));
                let hi0 = vreinterpretq_s8_u8(vshrq_n_u8(raw0, 4));
                let hi1 = vreinterpretq_s8_u8(vshrq_n_u8(raw1, 4));
                let dot_lo = vaddvq_s32(sdot32_vec(lo0, lo1, a_lo0, a_lo1));
                let dot_hi = vaddvq_s32(sdot32_vec(hi0, hi1, a_hi0, a_hi1));
                sumi[r] += dot_lo * scales[r].0[chunk * 2] + dot_hi * scales[r].0[chunk * 2 + 1];
            }
        }
        for r in 0..4 {
            #[expect(
                clippy::suboptimal_flops,
                reason = "Matches non-FMA Q4_K block accumulation order."
            )]
            {
                acc[r] += (f64::from(d_weights[r]) * d_act) * f64::from(sumi[r])
                    - (f64::from(dmin_weights[r]) * d_act) * f64::from(mint[r]);
            }
        }
    }
    acc.map(super::kquants::f64_to_f32)
}

pub(super) fn quant_gemm_q4_k_4x2(
    rows: [&[u8]; 4],
    act0: &[super::kquants::Q8_KActBlock],
    act1: &[super::kquants::Q8_KActBlock],
) -> Option<[[f32; 2]; 4]> {
    if !has_i8mm() {
        return None;
    }
    // SAFETY: `has_i8mm()` verified that the CPU supports `i8mm`, `dotprod`, and baseline `neon`.
    Some(unsafe { quant_gemm_q4_k_4x2_i8mm(rows, act0, act1) })
}

#[expect(
    clippy::too_many_lines,
    reason = "Unrolls two-token SMMLA accumulators for four Q4_K weight rows in registers."
)]
#[target_feature(enable = "neon,i8mm")]
fn quant_gemm_q4_k_4x2_i8mm(
    rows: [&[u8]; 4],
    act0: &[super::kquants::Q8_KActBlock],
    act1: &[super::kquants::Q8_KActBlock],
) -> [[f32; 2]; 4] {
    use core::arch::aarch64::vmulq_s32;
    let mut acc = [[0.0_f64; 2]; 4];
    let r0 = rows[0].as_chunks::<144>().0;
    let r1 = rows[1].as_chunks::<144>().0;
    let r2 = rows[2].as_chunks::<144>().0;
    let r3 = rows[3].as_chunks::<144>().0;
    let mask_0f = vdupq_n_u8(0x0f);
    for (idx, (ab0, ab1)) in act0.iter().zip(act1.iter()).enumerate() {
        let blocks = [&r0[idx], &r1[idx], &r2[idx], &r3[idx]];
        let a0_g = ab0.qs.as_chunks::<16>().0;
        let a1_g = ab1.qs.as_chunks::<16>().0;
        let bsum0: [i32; 8] =
            std::array::from_fn(|g| i32::from(ab0.bsums[2 * g]) + i32::from(ab0.bsums[2 * g + 1]));
        let bsum1: [i32; 8] =
            std::array::from_fn(|g| i32::from(ab1.bsums[2 * g]) + i32::from(ab1.bsums[2 * g + 1]));
        let d_act0 = f64::from(ab0.d);
        let d_act1 = f64::from(ab1.d);
        let d_f16: [u16; 8] = [
            u16::from_le_bytes([blocks[0][0], blocks[0][1]]),
            u16::from_le_bytes([blocks[1][0], blocks[1][1]]),
            u16::from_le_bytes([blocks[2][0], blocks[2][1]]),
            u16::from_le_bytes([blocks[3][0], blocks[3][1]]),
            u16::from_le_bytes([blocks[0][2], blocks[0][3]]),
            u16::from_le_bytes([blocks[1][2], blocks[1][3]]),
            u16::from_le_bytes([blocks[2][2], blocks[2][3]]),
            u16::from_le_bytes([blocks[3][2], blocks[3][3]]),
        ];
        let [d_weights, dmin_weights] = super::widen_f16_eight(&d_f16);
        let scales = blocks.map(|b| {
            let mut sc_arr = [0_i32; 8];
            let mut min_arr = [0_i32; 8];
            let raw_sc = &b[4..16];
            for g in 0..8 {
                let (sc_v, m_v) = super::kquants::unpack_q4_k_scale_min(g, raw_sc);
                sc_arr[g] = i32::from(sc_v);
                min_arr[g] = i32::from(m_v);
            }
            (sc_arr, min_arr)
        });
        let mut mint = [[0_i32; 2]; 4];
        for r in 0..4 {
            let mut m0 = 0_i32;
            let mut m1 = 0_i32;
            for g in 0..8 {
                m0 += scales[r].1[g] * bsum0[g];
                m1 += scales[r].1[g] * bsum1[g];
            }
            mint[r] = [m0, m1];
        }
        let mut sum01 = vdupq_n_s32(0);
        let mut sum23 = vdupq_n_s32(0);
        for chunk in 0..4 {
            let qs_off = 16 + chunk * 32;
            let mut lo01 = vdupq_n_s32(0);
            let mut lo23 = vdupq_n_s32(0);
            let mut hi01 = vdupq_n_s32(0);
            let mut hi23 = vdupq_n_s32(0);
            for sub in 0..2 {
                let a0_lo = load_s8x16(&a0_g[chunk * 4 + sub]);
                let a1_lo = load_s8x16(&a1_g[chunk * 4 + sub]);
                let a_lo_0 = vcombine_s8(vget_low_s8(a0_lo), vget_low_s8(a1_lo));
                let a_lo_1 = vcombine_s8(vget_high_s8(a0_lo), vget_high_s8(a1_lo));

                let a0_hi = load_s8x16(&a0_g[chunk * 4 + 2 + sub]);
                let a1_hi = load_s8x16(&a1_g[chunk * 4 + 2 + sub]);
                let a_hi_0 = vcombine_s8(vget_low_s8(a0_hi), vget_low_s8(a1_hi));
                let a_hi_1 = vcombine_s8(vget_high_s8(a0_hi), vget_high_s8(a1_hi));

                let off = qs_off + sub * 16;
                let r0_raw = load_u8x16(&blocks[0][off..off + 16].as_chunks::<16>().0[0]);
                let r1_raw = load_u8x16(&blocks[1][off..off + 16].as_chunks::<16>().0[0]);
                let r2_raw = load_u8x16(&blocks[2][off..off + 16].as_chunks::<16>().0[0]);
                let r3_raw = load_u8x16(&blocks[3][off..off + 16].as_chunks::<16>().0[0]);

                let l0 = vreinterpretq_s8_u8(vandq_u8(r0_raw, mask_0f));
                let l1 = vreinterpretq_s8_u8(vandq_u8(r1_raw, mask_0f));
                let l2 = vreinterpretq_s8_u8(vandq_u8(r2_raw, mask_0f));
                let l3 = vreinterpretq_s8_u8(vandq_u8(r3_raw, mask_0f));
                lo01 = smmla_s32(lo01, vcombine_s8(vget_low_s8(l0), vget_low_s8(l1)), a_lo_0);
                lo01 = smmla_s32(
                    lo01,
                    vcombine_s8(vget_high_s8(l0), vget_high_s8(l1)),
                    a_lo_1,
                );
                lo23 = smmla_s32(lo23, vcombine_s8(vget_low_s8(l2), vget_low_s8(l3)), a_lo_0);
                lo23 = smmla_s32(
                    lo23,
                    vcombine_s8(vget_high_s8(l2), vget_high_s8(l3)),
                    a_lo_1,
                );

                let h0 = vreinterpretq_s8_u8(vshrq_n_u8(r0_raw, 4));
                let h1 = vreinterpretq_s8_u8(vshrq_n_u8(r1_raw, 4));
                let h2 = vreinterpretq_s8_u8(vshrq_n_u8(r2_raw, 4));
                let h3 = vreinterpretq_s8_u8(vshrq_n_u8(r3_raw, 4));
                hi01 = smmla_s32(hi01, vcombine_s8(vget_low_s8(h0), vget_low_s8(h1)), a_hi_0);
                hi01 = smmla_s32(
                    hi01,
                    vcombine_s8(vget_high_s8(h0), vget_high_s8(h1)),
                    a_hi_1,
                );
                hi23 = smmla_s32(hi23, vcombine_s8(vget_low_s8(h2), vget_low_s8(h3)), a_hi_0);
                hi23 = smmla_s32(
                    hi23,
                    vcombine_s8(vget_high_s8(h2), vget_high_s8(h3)),
                    a_hi_1,
                );
            }
            let g_lo = chunk * 2;
            let g_hi = chunk * 2 + 1;
            let sc_lo01 = load_s32x4(&[
                scales[0].0[g_lo],
                scales[0].0[g_lo],
                scales[1].0[g_lo],
                scales[1].0[g_lo],
            ]);
            let sc_hi01 = load_s32x4(&[
                scales[0].0[g_hi],
                scales[0].0[g_hi],
                scales[1].0[g_hi],
                scales[1].0[g_hi],
            ]);
            let sc_lo23 = load_s32x4(&[
                scales[2].0[g_lo],
                scales[2].0[g_lo],
                scales[3].0[g_lo],
                scales[3].0[g_lo],
            ]);
            let sc_hi23 = load_s32x4(&[
                scales[2].0[g_hi],
                scales[2].0[g_hi],
                scales[3].0[g_hi],
                scales[3].0[g_hi],
            ]);
            sum01 = vaddq_s32(sum01, vmulq_s32(lo01, sc_lo01));
            sum01 = vaddq_s32(sum01, vmulq_s32(hi01, sc_hi01));
            sum23 = vaddq_s32(sum23, vmulq_s32(lo23, sc_lo23));
            sum23 = vaddq_s32(sum23, vmulq_s32(hi23, sc_hi23));
        }
        let mut f01 = [0_i32; 4];
        let mut f23 = [0_i32; 4];
        store_s32x4(sum01, &mut f01);
        store_s32x4(sum23, &mut f23);
        let sums = [
            [f01[0], f01[1]],
            [f01[2], f01[3]],
            [f23[0], f23[1]],
            [f23[2], f23[3]],
        ];
        for r in 0..4 {
            let dw = f64::from(d_weights[r]);
            let dm = f64::from(dmin_weights[r]);
            #[expect(
                clippy::suboptimal_flops,
                reason = "Matches non-FMA Q4_K block accumulation order."
            )]
            {
                acc[r][0] +=
                    (dw * d_act0) * f64::from(sums[r][0]) - (dm * d_act0) * f64::from(mint[r][0]);
                acc[r][1] +=
                    (dw * d_act1) * f64::from(sums[r][1]) - (dm * d_act1) * f64::from(mint[r][1]);
            }
        }
    }
    acc.map(|pair| pair.map(super::kquants::f64_to_f32))
}

pub(super) fn quant_dot_q4_k_repacked8(
    strip: &[u8],
    act: &[super::kquants::Q8_KActBlock],
) -> Option<[f32; 8]> {
    if !has_dotprod() {
        return None;
    }
    // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
    Some(unsafe { quant_dot_q4_k_repacked8_dotprod(strip, act) })
}

#[target_feature(enable = "neon,dotprod")]
fn quant_dot_q4_k_repacked8_dotprod(
    strip: &[u8],
    act: &[super::kquants::Q8_KActBlock],
) -> [f32; 8] {
    let mut acc = [0.0_f64; 8];
    let mask_0f = vdupq_n_u8(0x0f);
    for (block, ab) in strip.as_chunks::<1184>().0.iter().zip(act.iter()) {
        let a_groups = ab.qs.as_chunks::<16>().0;
        let bsum_pairs: [i32; 8] =
            std::array::from_fn(|g| i32::from(ab.bsums[2 * g]) + i32::from(ab.bsums[2 * g + 1]));
        let d_act = f64::from(ab.d);
        let d_f16: [u16; 8] =
            std::array::from_fn(|r| u16::from_le_bytes([block[r * 2], block[r * 2 + 1]]));
        let dmin_f16: [u16; 8] =
            std::array::from_fn(|r| u16::from_le_bytes([block[16 + r * 2], block[16 + r * 2 + 1]]));
        let sc_bytes = &block[32..96];
        let m_bytes = &block[96..160];
        let mut mint = [0_i32; 8];
        for g in 0..8 {
            let b_pair = bsum_pairs[g];
            let m_row = &m_bytes[g * 8..(g + 1) * 8];
            for r in 0..8 {
                mint[r] += i32::from(m_row[r]) * b_pair;
            }
        }
        let mut sumi = [0_i32; 8];
        for chunk in 0..4 {
            let a_lo0 = load_s8x16(&a_groups[chunk * 4]);
            let a_lo1 = load_s8x16(&a_groups[chunk * 4 + 1]);
            let a_hi0 = load_s8x16(&a_groups[chunk * 4 + 2]);
            let a_hi1 = load_s8x16(&a_groups[chunk * 4 + 3]);
            let qs_base = 160 + chunk * 256;
            let sc_lo = &sc_bytes[(chunk * 2) * 8..(chunk * 2 + 1) * 8];
            let sc_hi = &sc_bytes[(chunk * 2 + 1) * 8..(chunk * 2 + 2) * 8];
            for r in 0..8 {
                let row_qs = qs_base + r * 32;
                let raw0 = load_u8x16(&block[row_qs..row_qs + 16].as_chunks::<16>().0[0]);
                let raw1 = load_u8x16(&block[row_qs + 16..row_qs + 32].as_chunks::<16>().0[0]);
                let lo0 = vreinterpretq_s8_u8(vandq_u8(raw0, mask_0f));
                let lo1 = vreinterpretq_s8_u8(vandq_u8(raw1, mask_0f));
                let hi0 = vreinterpretq_s8_u8(vshrq_n_u8(raw0, 4));
                let hi1 = vreinterpretq_s8_u8(vshrq_n_u8(raw1, 4));
                let d_lo = vaddvq_s32(sdot32_vec(lo0, lo1, a_lo0, a_lo1));
                let d_hi = vaddvq_s32(sdot32_vec(hi0, hi1, a_hi0, a_hi1));
                sumi[r] += d_lo * i32::from(sc_lo[r]) + d_hi * i32::from(sc_hi[r]);
            }
        }
        let d_w = super::widen_f16_eight(&d_f16);
        let dmin_w = super::widen_f16_eight(&dmin_f16);
        for r in 0..8 {
            let dw = f64::from(d_w[r / 4][r % 4]) * d_act;
            let dm = f64::from(dmin_w[r / 4][r % 4]) * d_act;
            #[expect(
                clippy::suboptimal_flops,
                reason = "Matches non-FMA repacked Q4_K block accumulation order."
            )]
            {
                acc[r] += dw * f64::from(sumi[r]) - dm * f64::from(mint[r]);
            }
        }
    }
    acc.map(super::kquants::f64_to_f32)
}

pub(super) fn quant_gemm_q4_k_repacked8_8x2(
    strip: &[u8],
    act0: &[super::kquants::Q8_KActBlock],
    act1: &[super::kquants::Q8_KActBlock],
) -> Option<[[f32; 2]; 8]> {
    if !has_i8mm() {
        return None;
    }
    // SAFETY: `has_i8mm()` verified that the CPU supports `i8mm`, `dotprod`, and baseline `neon`.
    Some(unsafe { quant_gemm_q4_k_repacked8_8x2_i8mm(strip, act0, act1) })
}

#[expect(
    clippy::too_many_lines,
    reason = "Unrolls two-token SMMLA accumulators for an 8-row repacked strip in registers."
)]
#[target_feature(enable = "neon,i8mm")]
fn quant_gemm_q4_k_repacked8_8x2_i8mm(
    strip: &[u8],
    act0: &[super::kquants::Q8_KActBlock],
    act1: &[super::kquants::Q8_KActBlock],
) -> [[f32; 2]; 8] {
    use core::arch::aarch64::vmulq_s32;
    let mut acc = [[0.0_f64; 2]; 8];
    let mask_0f = vdupq_n_u8(0x0f);
    for (block, (ab0, ab1)) in strip
        .as_chunks::<1184>()
        .0
        .iter()
        .zip(act0.iter().zip(act1.iter()))
    {
        let a0_g = ab0.qs.as_chunks::<16>().0;
        let a1_g = ab1.qs.as_chunks::<16>().0;
        let bsum0: [i32; 8] =
            std::array::from_fn(|g| i32::from(ab0.bsums[2 * g]) + i32::from(ab0.bsums[2 * g + 1]));
        let bsum1: [i32; 8] =
            std::array::from_fn(|g| i32::from(ab1.bsums[2 * g]) + i32::from(ab1.bsums[2 * g + 1]));
        let d_act0 = f64::from(ab0.d);
        let d_act1 = f64::from(ab1.d);
        let d_f16: [u16; 8] =
            std::array::from_fn(|r| u16::from_le_bytes([block[r * 2], block[r * 2 + 1]]));
        let dmin_f16: [u16; 8] =
            std::array::from_fn(|r| u16::from_le_bytes([block[16 + r * 2], block[16 + r * 2 + 1]]));
        let sc_bytes = &block[32..96];
        let m_bytes = &block[96..160];
        let mut mint = [[0_i32; 2]; 8];
        for g in 0..8 {
            let b0 = bsum0[g];
            let b1 = bsum1[g];
            let m_row = &m_bytes[g * 8..(g + 1) * 8];
            for r in 0..8 {
                let m_val = i32::from(m_row[r]);
                mint[r][0] += m_val * b0;
                mint[r][1] += m_val * b1;
            }
        }
        let mut sum_pairs = [vdupq_n_s32(0); 4];
        for chunk in 0..4 {
            let mut lo_acc = [vdupq_n_s32(0); 4];
            let mut hi_acc = [vdupq_n_s32(0); 4];
            let qs_base = 160 + chunk * 256;
            for sub in 0..2 {
                let a0_lo = load_s8x16(&a0_g[chunk * 4 + sub]);
                let a1_lo = load_s8x16(&a1_g[chunk * 4 + sub]);
                let a_lo_0 = vcombine_s8(vget_low_s8(a0_lo), vget_low_s8(a1_lo));
                let a_lo_1 = vcombine_s8(vget_high_s8(a0_lo), vget_high_s8(a1_lo));

                let a0_hi = load_s8x16(&a0_g[chunk * 4 + 2 + sub]);
                let a1_hi = load_s8x16(&a1_g[chunk * 4 + 2 + sub]);
                let a_hi_0 = vcombine_s8(vget_low_s8(a0_hi), vget_low_s8(a1_hi));
                let a_hi_1 = vcombine_s8(vget_high_s8(a0_hi), vget_high_s8(a1_hi));

                for pair in 0..4 {
                    let o0 = qs_base + (pair * 2) * 32 + sub * 16;
                    let o1 = qs_base + (pair * 2 + 1) * 32 + sub * 16;
                    let raw0 = load_u8x16(&block[o0..o0 + 16].as_chunks::<16>().0[0]);
                    let raw1 = load_u8x16(&block[o1..o1 + 16].as_chunks::<16>().0[0]);
                    let l0 = vreinterpretq_s8_u8(vandq_u8(raw0, mask_0f));
                    let l1 = vreinterpretq_s8_u8(vandq_u8(raw1, mask_0f));
                    lo_acc[pair] = smmla_s32(
                        lo_acc[pair],
                        vcombine_s8(vget_low_s8(l0), vget_low_s8(l1)),
                        a_lo_0,
                    );
                    lo_acc[pair] = smmla_s32(
                        lo_acc[pair],
                        vcombine_s8(vget_high_s8(l0), vget_high_s8(l1)),
                        a_lo_1,
                    );

                    let h0 = vreinterpretq_s8_u8(vshrq_n_u8(raw0, 4));
                    let h1 = vreinterpretq_s8_u8(vshrq_n_u8(raw1, 4));
                    hi_acc[pair] = smmla_s32(
                        hi_acc[pair],
                        vcombine_s8(vget_low_s8(h0), vget_low_s8(h1)),
                        a_hi_0,
                    );
                    hi_acc[pair] = smmla_s32(
                        hi_acc[pair],
                        vcombine_s8(vget_high_s8(h0), vget_high_s8(h1)),
                        a_hi_1,
                    );
                }
            }
            let sc_lo = &sc_bytes[(chunk * 2) * 8..(chunk * 2 + 1) * 8];
            let sc_hi = &sc_bytes[(chunk * 2 + 1) * 8..(chunk * 2 + 2) * 8];
            for pair in 0..4 {
                let s0 = i32::from(sc_lo[pair * 2]);
                let s1 = i32::from(sc_lo[pair * 2 + 1]);
                let h0 = i32::from(sc_hi[pair * 2]);
                let h1 = i32::from(sc_hi[pair * 2 + 1]);
                let vec_scale_low = load_s32x4(&[s0, s0, s1, s1]);
                let vec_scale_high = load_s32x4(&[h0, h0, h1, h1]);
                sum_pairs[pair] =
                    vaddq_s32(sum_pairs[pair], vmulq_s32(lo_acc[pair], vec_scale_low));
                sum_pairs[pair] =
                    vaddq_s32(sum_pairs[pair], vmulq_s32(hi_acc[pair], vec_scale_high));
            }
        }
        let d_w = super::widen_f16_eight(&d_f16);
        let dmin_w = super::widen_f16_eight(&dmin_f16);
        for (pair, &pair_vec) in sum_pairs.iter().enumerate() {
            let mut vals = [0_i32; 4];
            store_s32x4(pair_vec, &mut vals);
            let r0 = pair * 2;
            let r1 = pair * 2 + 1;
            let dw0 = f64::from(d_w[r0 / 4][r0 % 4]);
            let dm0 = f64::from(dmin_w[r0 / 4][r0 % 4]);
            let dw1 = f64::from(d_w[r1 / 4][r1 % 4]);
            let dm1 = f64::from(dmin_w[r1 / 4][r1 % 4]);
            #[expect(
                clippy::suboptimal_flops,
                reason = "Matches non-FMA repacked Q4_K block accumulation order."
            )]
            {
                acc[r0][0] +=
                    (dw0 * d_act0) * f64::from(vals[0]) - (dm0 * d_act0) * f64::from(mint[r0][0]);
                acc[r0][1] +=
                    (dw0 * d_act1) * f64::from(vals[1]) - (dm0 * d_act1) * f64::from(mint[r0][1]);
                acc[r1][0] +=
                    (dw1 * d_act0) * f64::from(vals[2]) - (dm1 * d_act0) * f64::from(mint[r1][0]);
                acc[r1][1] +=
                    (dw1 * d_act1) * f64::from(vals[3]) - (dm1 * d_act1) * f64::from(mint[r1][1]);
            }
        }
    }
    acc.map(|pair| pair.map(super::kquants::f64_to_f32))
}

pub(super) fn quant_dot_q6_k_row(
    w_row: &[u8],
    act: &[super::kquants::Q8_KActBlock],
) -> Option<f32> {
    if !has_dotprod() {
        return None;
    }
    // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
    Some(unsafe { quant_dot_q6_k_row_dotprod(w_row, act) })
}

#[target_feature(enable = "neon,dotprod")]
fn quant_dot_q6_k_row_dotprod(w_row: &[u8], act: &[super::kquants::Q8_KActBlock]) -> f32 {
    let mut acc = 0.0_f64;
    let zero = vdupq_n_s32(0);
    for (block, ab) in w_row.as_chunks::<210>().0.iter().zip(act.iter()) {
        let a_groups = ab.qs.as_chunks::<16>().0;
        let d_act = f64::from(ab.d);
        let d_w = f64::from(super::kquants::f16_bits_to_f32([block[208], block[209]]));
        let mut sumi = 0_i32;
        for half_idx in 0..2 {
            for quarter in 0..4 {
                let g_base = half_idx * 8 + quarter * 2;
                let a0 = load_s8x16(&a_groups[g_base]);
                let a1 = load_s8x16(&a_groups[g_base + 1]);
                let (w0, w1, s0, s1) = unpack_q6_k_32(block, half_idx, quarter);
                let d0 = vaddvq_s32(vdotq_s32(zero, w0, a0));
                let d1 = vaddvq_s32(vdotq_s32(zero, w1, a1));
                sumi += d0 * s0 + d1 * s1;
            }
        }
        #[expect(
            clippy::suboptimal_flops,
            reason = "Matches non-FMA Q6_K block accumulation order."
        )]
        {
            acc += (d_w * d_act) * f64::from(sumi);
        }
    }
    super::kquants::f64_to_f32(acc)
}

pub(super) fn quant_dot_q6_k_4rows(
    rows: [&[u8]; 4],
    act: &[super::kquants::Q8_KActBlock],
) -> Option<[f32; 4]> {
    if !has_dotprod() {
        return None;
    }
    // SAFETY: `has_dotprod()` verified that the CPU supports `dotprod` and baseline `neon`.
    Some(unsafe { quant_dot_q6_k_4rows_dotprod(rows, act) })
}

#[target_feature(enable = "neon,dotprod")]
fn quant_dot_q6_k_4rows_dotprod(
    rows: [&[u8]; 4],
    act: &[super::kquants::Q8_KActBlock],
) -> [f32; 4] {
    let mut acc = [0.0_f64; 4];
    let r0 = rows[0].as_chunks::<210>().0;
    let r1 = rows[1].as_chunks::<210>().0;
    let r2 = rows[2].as_chunks::<210>().0;
    let r3 = rows[3].as_chunks::<210>().0;
    let zero = vdupq_n_s32(0);
    for (idx, ab) in act.iter().enumerate() {
        let blocks = [&r0[idx], &r1[idx], &r2[idx], &r3[idx]];
        let a_groups = ab.qs.as_chunks::<16>().0;
        let d_act = f64::from(ab.d);
        let d_bits: [u16; 4] = blocks.map(|b| u16::from_le_bytes([b[208], b[209]]));
        let d_weights = super::widen_f16_four(&d_bits);
        let mut sumi = [0_i32; 4];
        for half_idx in 0..2 {
            for quarter in 0..4 {
                let g_base = half_idx * 8 + quarter * 2;
                let a0 = load_s8x16(&a_groups[g_base]);
                let a1 = load_s8x16(&a_groups[g_base + 1]);
                for r in 0..4 {
                    let (w0, w1, s0, s1) = unpack_q6_k_32(blocks[r], half_idx, quarter);
                    let d0 = vaddvq_s32(vdotq_s32(zero, w0, a0));
                    let d1 = vaddvq_s32(vdotq_s32(zero, w1, a1));
                    sumi[r] += d0 * s0 + d1 * s1;
                }
            }
        }
        for r in 0..4 {
            #[expect(
                clippy::suboptimal_flops,
                reason = "Matches non-FMA Q6_K block accumulation order."
            )]
            {
                acc[r] += (f64::from(d_weights[r]) * d_act) * f64::from(sumi[r]);
            }
        }
    }
    acc.map(super::kquants::f64_to_f32)
}

pub(super) fn quant_gemm_q6_k_4x2(
    rows: [&[u8]; 4],
    act0: &[super::kquants::Q8_KActBlock],
    act1: &[super::kquants::Q8_KActBlock],
) -> Option<[[f32; 2]; 4]> {
    if !has_i8mm() {
        return None;
    }
    // SAFETY: `has_i8mm()` verified that the CPU supports `i8mm`, `dotprod`, and baseline `neon`.
    Some(unsafe { quant_gemm_q6_k_4x2_i8mm(rows, act0, act1) })
}

#[expect(
    clippy::too_many_lines,
    reason = "Unrolls two-token SMMLA accumulators for four Q6_K weight rows in registers."
)]
#[target_feature(enable = "neon,i8mm")]
fn quant_gemm_q6_k_4x2_i8mm(
    rows: [&[u8]; 4],
    act0: &[super::kquants::Q8_KActBlock],
    act1: &[super::kquants::Q8_KActBlock],
) -> [[f32; 2]; 4] {
    use core::arch::aarch64::vmulq_s32;
    let mut acc = [[0.0_f64; 2]; 4];
    let r0 = rows[0].as_chunks::<210>().0;
    let r1 = rows[1].as_chunks::<210>().0;
    let r2 = rows[2].as_chunks::<210>().0;
    let r3 = rows[3].as_chunks::<210>().0;
    for (idx, (ab0, ab1)) in act0.iter().zip(act1.iter()).enumerate() {
        let blocks = [&r0[idx], &r1[idx], &r2[idx], &r3[idx]];
        let a0_g = ab0.qs.as_chunks::<16>().0;
        let a1_g = ab1.qs.as_chunks::<16>().0;
        let d_act0 = f64::from(ab0.d);
        let d_act1 = f64::from(ab1.d);
        let d_bits: [u16; 4] = blocks.map(|b| u16::from_le_bytes([b[208], b[209]]));
        let d_weights = super::widen_f16_four(&d_bits);
        let mut sum01 = vdupq_n_s32(0);
        let mut sum23 = vdupq_n_s32(0);
        for half_idx in 0..2 {
            for quarter in 0..4 {
                let g_base = half_idx * 8 + quarter * 2;
                let a0_lo = load_s8x16(&a0_g[g_base]);
                let a1_lo = load_s8x16(&a1_g[g_base]);
                let a_g0_lo = vcombine_s8(vget_low_s8(a0_lo), vget_low_s8(a1_lo));
                let a_g0_hi = vcombine_s8(vget_high_s8(a0_lo), vget_high_s8(a1_lo));

                let a0_hi = load_s8x16(&a0_g[g_base + 1]);
                let a1_hi = load_s8x16(&a1_g[g_base + 1]);
                let a_g1_lo = vcombine_s8(vget_low_s8(a0_hi), vget_low_s8(a1_hi));
                let a_g1_hi = vcombine_s8(vget_high_s8(a0_hi), vget_high_s8(a1_hi));

                let (w0_0, w0_1, s0_0, s0_1) = unpack_q6_k_32(blocks[0], half_idx, quarter);
                let (w1_0, w1_1, s1_0, s1_1) = unpack_q6_k_32(blocks[1], half_idx, quarter);
                let (w2_0, w2_1, s2_0, s2_1) = unpack_q6_k_32(blocks[2], half_idx, quarter);
                let (w3_0, w3_1, s3_0, s3_1) = unpack_q6_k_32(blocks[3], half_idx, quarter);

                let g0_01 = smmla_s32(
                    smmla_s32(
                        vdupq_n_s32(0),
                        vcombine_s8(vget_low_s8(w0_0), vget_low_s8(w1_0)),
                        a_g0_lo,
                    ),
                    vcombine_s8(vget_high_s8(w0_0), vget_high_s8(w1_0)),
                    a_g0_hi,
                );
                let g1_01 = smmla_s32(
                    smmla_s32(
                        vdupq_n_s32(0),
                        vcombine_s8(vget_low_s8(w0_1), vget_low_s8(w1_1)),
                        a_g1_lo,
                    ),
                    vcombine_s8(vget_high_s8(w0_1), vget_high_s8(w1_1)),
                    a_g1_hi,
                );
                sum01 = vaddq_s32(
                    sum01,
                    vmulq_s32(g0_01, load_s32x4(&[s0_0, s0_0, s1_0, s1_0])),
                );
                sum01 = vaddq_s32(
                    sum01,
                    vmulq_s32(g1_01, load_s32x4(&[s0_1, s0_1, s1_1, s1_1])),
                );

                let g0_23 = smmla_s32(
                    smmla_s32(
                        vdupq_n_s32(0),
                        vcombine_s8(vget_low_s8(w2_0), vget_low_s8(w3_0)),
                        a_g0_lo,
                    ),
                    vcombine_s8(vget_high_s8(w2_0), vget_high_s8(w3_0)),
                    a_g0_hi,
                );
                let g1_23 = smmla_s32(
                    smmla_s32(
                        vdupq_n_s32(0),
                        vcombine_s8(vget_low_s8(w2_1), vget_low_s8(w3_1)),
                        a_g1_lo,
                    ),
                    vcombine_s8(vget_high_s8(w2_1), vget_high_s8(w3_1)),
                    a_g1_hi,
                );
                sum23 = vaddq_s32(
                    sum23,
                    vmulq_s32(g0_23, load_s32x4(&[s2_0, s2_0, s3_0, s3_0])),
                );
                sum23 = vaddq_s32(
                    sum23,
                    vmulq_s32(g1_23, load_s32x4(&[s2_1, s2_1, s3_1, s3_1])),
                );
            }
        }
        let mut f01 = [0_i32; 4];
        let mut f23 = [0_i32; 4];
        store_s32x4(sum01, &mut f01);
        store_s32x4(sum23, &mut f23);
        let sums = [
            [f01[0], f01[1]],
            [f01[2], f01[3]],
            [f23[0], f23[1]],
            [f23[2], f23[3]],
        ];
        for r in 0..4 {
            let dw = f64::from(d_weights[r]);
            #[expect(
                clippy::suboptimal_flops,
                reason = "Matches non-FMA Q6_K block accumulation order."
            )]
            {
                acc[r][0] += (dw * d_act0) * f64::from(sums[r][0]);
                acc[r][1] += (dw * d_act1) * f64::from(sums[r][1]);
            }
        }
    }
    acc.map(|pair| pair.map(super::kquants::f64_to_f32))
}

#[inline]
#[target_feature(enable = "neon")]
fn load_u32x4(values: &[u32; 4]) -> core::arch::aarch64::uint32x4_t {
    // SAFETY: `values` is a valid 4-element `u32` array reference and `vld1q_u32` requires no alignment.
    unsafe { core::arch::aarch64::vld1q_u32(values.as_ptr()) }
}

#[inline]
const fn trellis_quad(planar: &[u32; 16], offset: usize) -> [u32; 4] {
    [
        planar[offset],
        planar[offset + 1],
        planar[offset + 2],
        planar[offset + 3],
    ]
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
fn trellis_decode_half_column_dotprod(
    planar: &[[u32; 16]; 6],
    sched: &super::trellis::ColHalfSchedule,
) -> (
    core::arch::aarch64::float32x4_t,
    core::arch::aarch64::float32x4_t,
) {
    use core::arch::aarch64::{
        vandq_u32, vcvtq_f32_s32, vdupq_n_s8, vdupq_n_u32, vmulq_u32, vorrq_u32,
        vreinterpretq_u8_u32, vshlq_u32,
    };

    let s0 = vdupq_n_s32(sched.shift0);
    let s1 = vdupq_n_s32(sched.shift1);
    let m1 = vdupq_n_u32(sched.mask1);
    let low16 = vdupq_n_u32(0xffff);
    let lcg = vdupq_n_u32(super::trellis::LCG_MULTIPLIER);
    let bias_neg126 = vdupq_n_s32(-126);
    let ones = vdupq_n_s8(1);

    let decode_four = |row_offset: usize| {
        let w0 = load_u32x4(&trellis_quad(
            &planar[sched.w0_rem],
            sched.w0_q + row_offset,
        ));
        let w1 = load_u32x4(&trellis_quad(
            &planar[sched.w1_rem],
            sched.w1_q + row_offset,
        ));
        let part0 = vshlq_u32(w0, s0);
        let part1 = vandq_u32(vshlq_u32(w1, s1), m1);
        let state = vandq_u32(vorrq_u32(part0, part1), low16);
        let x = vmulq_u32(state, lcg);
        let bytes = vreinterpretq_s8_u8(vshrq_n_u8::<2>(vreinterpretq_u8_u32(x)));
        let signed = vdotq_s32(bias_neg126, bytes, ones);
        vcvtq_f32_s32(signed)
    };

    (decode_four(0), decode_four(4))
}

#[inline]
#[target_feature(enable = "neon")]
fn trellis_decode_half_column(
    planar: &[[u32; 16]; 6],
    sched: &super::trellis::ColHalfSchedule,
) -> (
    core::arch::aarch64::float32x4_t,
    core::arch::aarch64::float32x4_t,
) {
    use core::arch::aarch64::{
        vaddq_u32, vandq_u32, vcvtq_f32_s32, vdupq_n_u32, vmulq_u32, vorrq_u32,
        vreinterpretq_s32_u32, vshlq_u32, vshrq_n_u32,
    };

    let s0 = vdupq_n_s32(sched.shift0);
    let s1 = vdupq_n_s32(sched.shift1);
    let m1 = vdupq_n_u32(sched.mask1);
    let low16 = vdupq_n_u32(0xffff);
    let lcg = vdupq_n_u32(super::trellis::LCG_MULTIPLIER);
    let mask63 = vdupq_n_u32(63);
    let bias126 = vdupq_n_s32(126);

    let decode_four = |row_offset: usize| {
        let w0 = load_u32x4(&trellis_quad(
            &planar[sched.w0_rem],
            sched.w0_q + row_offset,
        ));
        let w1 = load_u32x4(&trellis_quad(
            &planar[sched.w1_rem],
            sched.w1_q + row_offset,
        ));
        let part0 = vshlq_u32(w0, s0);
        let part1 = vandq_u32(vshlq_u32(w1, s1), m1);
        let state = vandq_u32(vorrq_u32(part0, part1), low16);
        let x = vmulq_u32(state, lcg);
        let b0 = vandq_u32(vshrq_n_u32::<2>(x), mask63);
        let b1 = vandq_u32(vshrq_n_u32::<10>(x), mask63);
        let b2 = vandq_u32(vshrq_n_u32::<18>(x), mask63);
        let b3 = vandq_u32(vshrq_n_u32::<26>(x), mask63);
        let sum = vaddq_u32(vaddq_u32(b0, b1), vaddq_u32(b2, b3));
        let signed = vsubq_s32(vreinterpretq_s32_u32(sum), bias126);
        vcvtq_f32_s32(signed)
    };

    (decode_four(0), decode_four(4))
}

#[inline]
#[target_feature(enable = "neon")]
fn trellis_decode_tile_neon(
    planar: &[[u32; 16]; 6],
    schedule: &[super::trellis::ColHalfSchedule; 32],
) -> [[core::arch::aarch64::float32x4_t; 4]; 16] {
    std::array::from_fn(|c| {
        let (w0, w1) = trellis_decode_half_column(planar, &schedule[c * 2]);
        let (w2, w3) = trellis_decode_half_column(planar, &schedule[c * 2 + 1]);
        [w0, w1, w2, w3]
    })
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
fn trellis_decode_tile_dotprod(
    planar: &[[u32; 16]; 6],
    schedule: &[super::trellis::ColHalfSchedule; 32],
) -> [[core::arch::aarch64::float32x4_t; 4]; 16] {
    std::array::from_fn(|c| {
        let (w0, w1) = trellis_decode_half_column_dotprod(planar, &schedule[c * 2]);
        let (w2, w3) = trellis_decode_half_column_dotprod(planar, &schedule[c * 2 + 1]);
        [w0, w1, w2, w3]
    })
}

#[inline]
#[target_feature(enable = "neon")]
fn trellis_accumulate_tile_tokens(
    w_cols: &[[core::arch::aarch64::float32x4_t; 4]; 16],
    x_base: usize,
    x_tokens: &[&[f32]],
    out_tokens: &mut [&mut [f32; 16]],
) {
    use core::arch::aarch64::{vdupq_n_f32, vfmaq_f32};

    let mut t = 0_usize;
    while t + 4 <= x_tokens.len() {
        let [o0, o1, o2, o3] = &mut out_tokens[t..t + 4] else {
            break;
        };
        let q0 = o0.as_chunks_mut::<4>().0;
        let q1 = o1.as_chunks_mut::<4>().0;
        let q2 = o2.as_chunks_mut::<4>().0;
        let q3 = o3.as_chunks_mut::<4>().0;
        let mut a0 = [
            load_f32x4_neon(&q0[0]),
            load_f32x4_neon(&q0[1]),
            load_f32x4_neon(&q0[2]),
            load_f32x4_neon(&q0[3]),
        ];
        let mut a1 = [
            load_f32x4_neon(&q1[0]),
            load_f32x4_neon(&q1[1]),
            load_f32x4_neon(&q1[2]),
            load_f32x4_neon(&q1[3]),
        ];
        let mut a2 = [
            load_f32x4_neon(&q2[0]),
            load_f32x4_neon(&q2[1]),
            load_f32x4_neon(&q2[2]),
            load_f32x4_neon(&q2[3]),
        ];
        let mut a3 = [
            load_f32x4_neon(&q3[0]),
            load_f32x4_neon(&q3[1]),
            load_f32x4_neon(&q3[2]),
            load_f32x4_neon(&q3[3]),
        ];
        let x0 = &x_tokens[t][x_base..x_base + 16];
        let x1 = &x_tokens[t + 1][x_base..x_base + 16];
        let x2 = &x_tokens[t + 2][x_base..x_base + 16];
        let x3 = &x_tokens[t + 3][x_base..x_base + 16];
        for (c, w) in w_cols.iter().enumerate() {
            let xt0 = vdupq_n_f32(x0[c]);
            let xt1 = vdupq_n_f32(x1[c]);
            let xt2 = vdupq_n_f32(x2[c]);
            let xt3 = vdupq_n_f32(x3[c]);
            for q in 0..4 {
                a0[q] = vfmaq_f32(a0[q], w[q], xt0);
                a1[q] = vfmaq_f32(a1[q], w[q], xt1);
                a2[q] = vfmaq_f32(a2[q], w[q], xt2);
                a3[q] = vfmaq_f32(a3[q], w[q], xt3);
            }
        }
        for q in 0..4 {
            store_f32x4_neon(a0[q], &mut q0[q]);
            store_f32x4_neon(a1[q], &mut q1[q]);
            store_f32x4_neon(a2[q], &mut q2[q]);
            store_f32x4_neon(a3[q], &mut q3[q]);
        }
        t += 4;
    }
    while t < x_tokens.len() {
        let q_out = out_tokens[t].as_chunks_mut::<4>().0;
        let mut acc = [
            load_f32x4_neon(&q_out[0]),
            load_f32x4_neon(&q_out[1]),
            load_f32x4_neon(&q_out[2]),
            load_f32x4_neon(&q_out[3]),
        ];
        let x = &x_tokens[t][x_base..x_base + 16];
        for (c, w) in w_cols.iter().enumerate() {
            let xt = vdupq_n_f32(x[c]);
            for q in 0..4 {
                acc[q] = vfmaq_f32(acc[q], w[q], xt);
            }
        }
        for q in 0..4 {
            store_f32x4_neon(acc[q], &mut q_out[q]);
        }
        t += 1;
    }
}

/// Evaluates one `16`-row `NariQ` trellis strip against a single `s16`-scaled activation vector.
pub(crate) fn trellis_tile_row_matvec(
    row_tiles: &[u8],
    bits: usize,
    per_tile: usize,
    x_scaled: &[f32],
    out: &mut [f32; 16],
) {
    #[target_feature(enable = "neon")]
    fn inner_neon(
        row_tiles: &[u8],
        bits: usize,
        per_tile: usize,
        x_scaled: &[f32],
        out: &mut [f32; 16],
    ) {
        use core::arch::aarch64::{vdupq_n_f32, vfmaq_f32};

        let schedule = super::trellis::schedule_for(bits);
        let mut planar = [[0_u32; 16]; 6];
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let tile_cols = x_scaled.len() / 16;

        for tile_idx in 0..tile_cols {
            let tile = &row_tiles[tile_idx * per_tile..(tile_idx + 1) * per_tile];
            super::trellis::scatter_tile_planar(tile, bits, &mut planar);
            let x_base = tile_idx * 16;
            for c in 0..16 {
                let xc = vdupq_n_f32(x_scaled[x_base + c]);
                let (w0, w1) = trellis_decode_half_column(&planar, &schedule[c * 2]);
                let (w2, w3) = trellis_decode_half_column(&planar, &schedule[c * 2 + 1]);
                acc0 = vfmaq_f32(acc0, w0, xc);
                acc1 = vfmaq_f32(acc1, w1, xc);
                acc2 = vfmaq_f32(acc2, w2, xc);
                acc3 = vfmaq_f32(acc3, w3, xc);
            }
        }

        let quads = out.as_chunks_mut::<4>().0;
        store_f32x4_neon(acc0, &mut quads[0]);
        store_f32x4_neon(acc1, &mut quads[1]);
        store_f32x4_neon(acc2, &mut quads[2]);
        store_f32x4_neon(acc3, &mut quads[3]);
    }

    #[target_feature(enable = "neon,dotprod")]
    fn inner_dotprod(
        row_tiles: &[u8],
        bits: usize,
        per_tile: usize,
        x_scaled: &[f32],
        out: &mut [f32; 16],
    ) {
        use core::arch::aarch64::{vdupq_n_f32, vfmaq_f32};

        let schedule = super::trellis::schedule_for(bits);
        let mut planar = [[0_u32; 16]; 6];
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let tile_cols = x_scaled.len() / 16;

        for tile_idx in 0..tile_cols {
            let tile = &row_tiles[tile_idx * per_tile..(tile_idx + 1) * per_tile];
            super::trellis::scatter_tile_planar(tile, bits, &mut planar);
            let x_base = tile_idx * 16;
            for c in 0..16 {
                let xc = vdupq_n_f32(x_scaled[x_base + c]);
                let (w0, w1) = trellis_decode_half_column_dotprod(&planar, &schedule[c * 2]);
                let (w2, w3) = trellis_decode_half_column_dotprod(&planar, &schedule[c * 2 + 1]);
                acc0 = vfmaq_f32(acc0, w0, xc);
                acc1 = vfmaq_f32(acc1, w1, xc);
                acc2 = vfmaq_f32(acc2, w2, xc);
                acc3 = vfmaq_f32(acc3, w3, xc);
            }
        }

        let quads = out.as_chunks_mut::<4>().0;
        store_f32x4_neon(acc0, &mut quads[0]);
        store_f32x4_neon(acc1, &mut quads[1]);
        store_f32x4_neon(acc2, &mut quads[2]);
        store_f32x4_neon(acc3, &mut quads[3]);
    }

    if has_dotprod() {
        // SAFETY: `has_dotprod()` verified CPU/OS `dotprod` and baseline `neon`.
        unsafe { inner_dotprod(row_tiles, bits, per_tile, x_scaled, out) };
    } else {
        // SAFETY: ARMv8-A always provides baseline NEON instructions on `aarch64` targets.
        unsafe { inner_neon(row_tiles, bits, per_tile, x_scaled, out) };
    }
}

/// Evaluates one `16`-row `NariQ` trellis strip across all `s16`-scaled tokens,
/// decoding each `16 × 16` tile once and reusing it across every token.
pub(crate) fn trellis_tile_row_matmul_tokens(
    row_tiles: &[u8],
    bits: usize,
    per_tile: usize,
    cols: usize,
    x_tokens: &[&[f32]],
    out_tokens: &mut [&mut [f32; 16]],
) {
    #[target_feature(enable = "neon")]
    fn inner_neon(
        row_tiles: &[u8],
        bits: usize,
        per_tile: usize,
        cols: usize,
        x_tokens: &[&[f32]],
        out_tokens: &mut [&mut [f32; 16]],
    ) {
        let schedule = super::trellis::schedule_for(bits);
        let mut planar = [[0_u32; 16]; 6];
        for out in out_tokens.iter_mut() {
            **out = [0.0_f32; 16];
        }
        let tile_cols = cols / 16;
        for tile_idx in 0..tile_cols {
            let tile = &row_tiles[tile_idx * per_tile..(tile_idx + 1) * per_tile];
            super::trellis::scatter_tile_planar(tile, bits, &mut planar);
            let w_cols = trellis_decode_tile_neon(&planar, schedule);
            trellis_accumulate_tile_tokens(&w_cols, tile_idx * 16, x_tokens, out_tokens);
        }
    }

    #[target_feature(enable = "neon,dotprod")]
    fn inner_dotprod(
        row_tiles: &[u8],
        bits: usize,
        per_tile: usize,
        cols: usize,
        x_tokens: &[&[f32]],
        out_tokens: &mut [&mut [f32; 16]],
    ) {
        let schedule = super::trellis::schedule_for(bits);
        let mut planar = [[0_u32; 16]; 6];
        for out in out_tokens.iter_mut() {
            **out = [0.0_f32; 16];
        }
        let tile_cols = cols / 16;
        for tile_idx in 0..tile_cols {
            let tile = &row_tiles[tile_idx * per_tile..(tile_idx + 1) * per_tile];
            super::trellis::scatter_tile_planar(tile, bits, &mut planar);
            let w_cols = trellis_decode_tile_dotprod(&planar, schedule);
            trellis_accumulate_tile_tokens(&w_cols, tile_idx * 16, x_tokens, out_tokens);
        }
    }

    if has_dotprod() {
        // SAFETY: `has_dotprod()` verified CPU/OS `dotprod` and baseline `neon`.
        unsafe { inner_dotprod(row_tiles, bits, per_tile, cols, x_tokens, out_tokens) };
    } else {
        // SAFETY: ARMv8-A always provides baseline NEON instructions on `aarch64` targets.
        unsafe { inner_neon(row_tiles, bits, per_tile, cols, x_tokens, out_tokens) };
    }
}
