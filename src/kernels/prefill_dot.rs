//! Exact integer reductions over one decoded block shared by a prompt tile.

use super::super::{MatvecBackend, q8_native_lanes};

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
                assert_eq!(block.dot128(&values), scalar(&weights, &values));
                for (group, values) in values.as_chunks::<32>().0.iter().enumerate() {
                    let weights = &weights[group * 32..group * 32 + 32];
                    assert_eq!(block.dot32(group, values), scalar(weights, values));
                    assert_eq!(
                        block.lanes32(group, values),
                        q8_native_lanes(weights, values)
                    );
                }
            }
        }
    }
}
