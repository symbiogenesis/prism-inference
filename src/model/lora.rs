//! `LoRA` adapter storage and multi-rank correction support.

use crate::Result;
use crate::kernels::{MatvecBackend, MatvecScratch};

mod multi_rank;
pub use multi_rank::{MultiRankAdapter, MultiRankPolicy};

/// Reusable bounded correction storage, independent of base projection scratch.
#[derive(Default)]
pub(crate) struct Scratch {
    pub(crate) multi_ranks: Vec<f32>,
    pub(crate) correction: Vec<f32>,
    pub(crate) matvec: MatvecScratch,
}

impl Scratch {
    pub(crate) fn new(backend: MatvecBackend, pinned: bool) -> Self {
        Self {
            multi_ranks: Vec::new(),
            correction: Vec::new(),
            matvec: if pinned {
                MatvecScratch::with_pinned_backend(backend)
            } else {
                MatvecScratch::with_backend(backend)
            },
        }
    }

    pub(crate) fn set_row_parallelism(&mut self, parallelism: crate::kernels::RowParallelism) {
        self.matvec.set_row_parallelism(parallelism);
    }

    #[cfg(all(test, feature = "parallel-rows"))]
    pub(crate) const fn row_workers(&self) -> usize {
        self.matvec.row_workers()
    }
}

fn hash_len_bytes(hasher: &mut bonsai_native_hash::Hasher, bytes: &[u8]) -> Result<()> {
    hasher.update(&u64::try_from(bytes.len())?.to_le_bytes())?;
    hasher.update(bytes)?;
    Ok(())
}
