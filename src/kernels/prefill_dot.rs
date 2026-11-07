//! Exact integer reductions over one decoded block shared by a prompt tile.

use super::super::{MatvecBackend, pq2_values, ptq1_values, q8_native_lanes};
use crate::Result;

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
#[path = "prefill_neon.rs"]
mod neon;

/// Exactly one block, scoped to the packed row loop; no heap allocation.
pub(super) enum Block {
    Scalar([i16; 128]),
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    Neon(neon::Block),
}

impl Block {
    #[cfg(test)]
    #[allow(
        clippy::missing_const_for_fn,
        reason = "Only the scalar fallback is constant; retain one signature across targets and feature sets."
    )]
    pub(super) fn new(weights: [i16; 128], backend: MatvecBackend) -> Self {
        #[cfg(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc",
            target_feature = "neon"
        ))]
        if backend != MatvecBackend::Scalar
            && let Some(block) = neon::Block::new(&weights)
        {
            return Self::Neon(block);
        }
        // Unsupported builds retain the scalar result for every request policy.
        let _ = backend;
        Self::Scalar(weights)
    }

    pub(super) fn from_pq2(packed: &[u8; 32], backend: MatvecBackend) -> Self {
        #[cfg(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc",
            target_feature = "neon"
        ))]
        if backend != MatvecBackend::Scalar
            && let Some(block) = neon::Block::from_pq2(packed)
        {
            return Self::Neon(block);
        }
        let _ = backend;
        let mut full = [0_u8; 34];
        full[2..].copy_from_slice(packed);
        Self::Scalar(pq2_values(&full))
    }

    pub(super) fn from_ptq1(packed: &[u8; 28], backend: MatvecBackend) -> Result<Self> {
        #[cfg(all(
            feature = "experimental-neon",
            target_arch = "aarch64",
            target_os = "windows",
            target_env = "msvc",
            target_feature = "neon"
        ))]
        if backend != MatvecBackend::Scalar
            && let Some(block) = neon::Block::from_ptq1(packed)
        {
            return Ok(Self::Neon(block));
        }
        let _ = backend;
        Ok(Self::Scalar(ptq1_values(packed)?))
    }

    #[inline]
    pub(super) fn dot128(&self, values: &[i8; 128]) -> i32 {
        match self {
            Self::Scalar(weights) => scalar(weights, values),
            #[cfg(all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc",
                target_feature = "neon"
            ))]
            Self::Neon(block) => block.dot128(values),
        }
    }

    #[inline]
    pub(super) fn dot128_four(&self, values: [&[i8; 128]; 4]) -> [i32; 4] {
        match self {
            Self::Scalar(weights) => values.map(|token| scalar(weights, token)),
            #[cfg(all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc",
                target_feature = "neon"
            ))]
            Self::Neon(block) => block.dot128_four(values),
        }
    }

    #[inline]
    pub(super) fn dot128_eight(&self, values: [&[i8; 128]; 8]) -> [i32; 8] {
        match self {
            Self::Scalar(weights) => values.map(|token| scalar(weights, token)),
            #[cfg(all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc",
                target_feature = "neon"
            ))]
            Self::Neon(block) => block.dot128_eight(values),
        }
    }

    #[cfg(test)]
    #[inline]
    pub(super) fn dot32(&self, group: usize, values: &[i8; 32]) -> i32 {
        match self {
            Self::Scalar(weights) => scalar(&weights[group * 32..group * 32 + 32], values),
            #[cfg(all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc",
                target_feature = "neon"
            ))]
            Self::Neon(block) => block.dot32(group, values),
        }
    }

    #[cfg(test)]
    #[inline]
    pub(super) fn dot32_four(&self, group: usize, values: [&[i8; 32]; 4]) -> [i32; 4] {
        match self {
            Self::Scalar(weights) => {
                let slice = &weights[group * 32..group * 32 + 32];
                values.map(|token| scalar(slice, token))
            }
            #[cfg(all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc",
                target_feature = "neon"
            ))]
            Self::Neon(block) => block.dot32_four(group, values),
        }
    }

    #[cfg(test)]
    #[inline]
    pub(super) fn dot32_eight(&self, group: usize, values: [&[i8; 32]; 8]) -> [i32; 8] {
        match self {
            Self::Scalar(weights) => {
                let slice = &weights[group * 32..group * 32 + 32];
                values.map(|token| scalar(slice, token))
            }
            #[cfg(all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc",
                target_feature = "neon"
            ))]
            Self::Neon(block) => block.dot32_eight(group, values),
        }
    }

    #[cfg(test)]
    #[inline]
    pub(super) fn lanes32(&self, group: usize, values: &[i8; 32]) -> [i32; 4] {
        match self {
            Self::Scalar(weights) => q8_native_lanes(&weights[group * 32..group * 32 + 32], values),
            #[cfg(all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc",
                target_feature = "neon"
            ))]
            Self::Neon(block) => block.lanes32(group, values),
        }
    }

    #[cfg(test)]
    #[inline]
    pub(super) fn lanes32_four(&self, group: usize, values: [&[i8; 32]; 4]) -> [[i32; 4]; 4] {
        match self {
            Self::Scalar(weights) => {
                let slice = &weights[group * 32..group * 32 + 32];
                values.map(|token| q8_native_lanes(slice, token))
            }
            #[cfg(all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc",
                target_feature = "neon"
            ))]
            Self::Neon(block) => block.lanes32_four(group, values),
        }
    }

    #[cfg(test)]
    #[inline]
    pub(super) fn lanes32_eight(&self, group: usize, values: [&[i8; 32]; 8]) -> [[i32; 4]; 8] {
        match self {
            Self::Scalar(weights) => {
                let slice = &weights[group * 32..group * 32 + 32];
                values.map(|token| q8_native_lanes(slice, token))
            }
            #[cfg(all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc",
                target_feature = "neon"
            ))]
            Self::Neon(block) => block.lanes32_eight(group, values),
        }
    }

    #[inline]
    pub(super) fn accumulate_lanes128(
        &self,
        weight_scale: f32,
        activations: &[super::super::Q8Block; 4],
        lanes: &mut [f32; 4],
    ) {
        match self {
            Self::Scalar(weights) => {
                for (group, activation) in activations.iter().enumerate() {
                    let d = weight_scale * activation.scale;
                    let integers =
                        q8_native_lanes(&weights[group * 32..group * 32 + 32], &activation.values);
                    for (lane, integer) in lanes.iter_mut().zip(integers) {
                        *lane = super::super::integer_float(integer).mul_add(d, *lane);
                    }
                }
            }
            #[cfg(all(
                feature = "experimental-neon",
                target_arch = "aarch64",
                target_os = "windows",
                target_env = "msvc",
                target_feature = "neon"
            ))]
            Self::Neon(block) => block.accumulate_lanes128(weight_scale, activations, lanes),
        }
    }

    #[allow(
        clippy::suboptimal_flops,
        reason = "Preserve scalar reference multiply-then-add rounding."
    )]
    #[inline]
    pub(super) fn dot128_q8(&self, activations: &[super::super::Q8Block; 4]) -> f32 {
        match self {
            Self::Scalar(weights) => {
                let mut sum = 0.0;
                for (group, activation) in activations.iter().enumerate() {
                    sum += activation.scale
                        * super::super::integer_float(scalar(
                            &weights[group * 32..group * 32 + 32],
                            &activation.values,
                        ));
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
            Self::Neon(block) => block.dot128_q8(activations),
        }
    }
}

#[allow(
    clippy::missing_const_for_fn,
    reason = "Retain one signature across targets and feature sets."
)]
pub(super) fn dot_eight_rows<const TOKENS: usize>(
    rows: [&[u8]; 8],
    preparations: &[super::super::MatvecScratch],
    kind: u32,
    columns: usize,
    pinned: bool,
    backend: MatvecBackend,
    decoded_blocks: &mut u64,
) -> Option<Result<[[f32; TOKENS]; 8]>> {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if backend != MatvecBackend::Scalar {
        if kind == super::super::PQ2 && columns.is_multiple_of(256) {
            return neon::dot_pq2_eight_rows(rows, preparations, pinned, decoded_blocks);
        }
        return neon::dot_q8_eight_rows(
            rows,
            preparations,
            kind == super::super::PTQ1,
            pinned,
            decoded_blocks,
        );
    }
    let _ = (
        rows,
        preparations,
        kind,
        columns,
        pinned,
        backend,
        decoded_blocks,
    );
    None
}

#[allow(
    clippy::missing_const_for_fn,
    reason = "Retain one signature across targets and feature sets."
)]
pub(super) fn dot_four_rows<const TOKENS: usize>(
    rows: [&[u8]; 4],
    preparations: &[super::super::MatvecScratch],
    kind: u32,
    columns: usize,
    pinned: bool,
    backend: MatvecBackend,
    decoded_blocks: &mut u64,
) -> Option<Result<[[f32; TOKENS]; 4]>> {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if backend != MatvecBackend::Scalar {
        if kind == super::super::PQ2 && columns.is_multiple_of(256) {
            return neon::dot_pq2_four_rows(rows, preparations, pinned, decoded_blocks);
        }
        return neon::dot_q8_four_rows(
            rows,
            preparations,
            kind == super::super::PTQ1,
            pinned,
            decoded_blocks,
        );
    }
    let _ = (
        rows,
        preparations,
        kind,
        columns,
        pinned,
        backend,
        decoded_blocks,
    );
    None
}

#[allow(
    clippy::missing_const_for_fn,
    reason = "Retain one signature across targets and feature sets."
)]
pub(super) fn dot_bf16_eight_rows<const TOKENS: usize>(
    rows: [&[u8]; 8],
    preparations: &[super::super::MatvecScratch],
    backend: MatvecBackend,
) -> Option<Result<[[f32; TOKENS]; 8]>> {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if backend != MatvecBackend::Scalar {
        return neon::dot_bf16_rows::<8, TOKENS>(rows, preparations);
    }
    let _ = (rows, preparations, backend);
    None
}

#[allow(
    clippy::missing_const_for_fn,
    reason = "Retain one signature across targets and feature sets."
)]
pub(super) fn dot_bf16_four_rows<const TOKENS: usize>(
    rows: [&[u8]; 4],
    preparations: &[super::super::MatvecScratch],
    backend: MatvecBackend,
) -> Option<Result<[[f32; TOKENS]; 4]>> {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if backend != MatvecBackend::Scalar {
        return neon::dot_bf16_rows::<4, TOKENS>(rows, preparations);
    }
    let _ = (rows, preparations, backend);
    None
}

fn scalar(weights: &[i16], values: &[i8]) -> i32 {
    weights
        .iter()
        .zip(values)
        .map(|(&weight, &value)| i32::from(weight) * i32::from(value))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiled_integer_reductions_preserve_extreme_values_and_native_lanes() {
        for seed in 0_u8..=255 {
            let weights = std::array::from_fn(|index: usize| {
                i16::from(seed.wrapping_add(index.to_le_bytes()[0]) & 3) - 1
            });
            let values = std::array::from_fn(|index: usize| {
                i8::from_ne_bytes([seed.wrapping_mul(197).wrapping_add(index.to_le_bytes()[0])])
            });
            for backend in [
                MatvecBackend::Scalar,
                MatvecBackend::Auto,
                MatvecBackend::ExperimentalNeon,
            ] {
                let block = Block::new(weights, backend);
                if backend == MatvecBackend::Scalar {
                    assert!(matches!(block, Block::Scalar(_)));
                }
                #[cfg(all(
                    feature = "experimental-neon",
                    target_arch = "aarch64",
                    target_os = "windows",
                    target_env = "msvc",
                    target_feature = "neon"
                ))]
                if backend != MatvecBackend::Scalar {
                    assert!(matches!(block, Block::Neon(_)));
                }
                let expected128 = scalar(&weights, &values);
                assert_eq!(block.dot128(&values), expected128);
                assert_eq!(block.dot128_four([&values; 4]), [expected128; 4]);
                assert_eq!(block.dot128_eight([&values; 8]), [expected128; 8]);
                for (group, values) in values.as_chunks::<32>().0.iter().enumerate() {
                    let weights = &weights[group * 32..group * 32 + 32];
                    let expected32 = scalar(weights, values);
                    let expected_lanes = q8_native_lanes(weights, values);
                    assert_eq!(block.dot32(group, values), expected32);
                    assert_eq!(block.dot32_four(group, [values; 4]), [expected32; 4]);
                    assert_eq!(block.dot32_eight(group, [values; 8]), [expected32; 8]);
                    assert_eq!(block.lanes32(group, values), expected_lanes);
                    assert_eq!(block.lanes32_four(group, [values; 4]), [expected_lanes; 4]);
                    assert_eq!(block.lanes32_eight(group, [values; 8]), [expected_lanes; 8]);
                }
            }
        }
    }
}
