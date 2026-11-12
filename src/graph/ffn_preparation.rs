//! Block-local FFN preparation retaining the original activation basis for adapters.

use super::{Arithmetic, resize_into};
use crate::{Result, invalid, kernels::RowParallelism};

impl Arithmetic {
    // Unlike the adapter-free read-only fusion, this writes the exact SwiGLU
    // result back to `up`: the down adapter must never consume signed/rotated
    // activations. Both outputs reuse existing graph scratch. Once mutation
    // starts, failures propagate without falling back or applying SwiGLU twice.
    pub(super) fn swiglu_hadamard_preserving_up_into(
        self,
        gate: &[f32],
        up: &mut [f32],
        block: usize,
        signs: Option<&[f32]>,
        parallelism: &RowParallelism,
        transformed: &mut Vec<f32>,
    ) -> Result<()> {
        let result = (|| {
            if gate.len() != up.len()
                || block < 16
                || !block.is_power_of_two()
                || !up.len().is_multiple_of(block)
                || signs.is_some_and(|s| s.len() < up.len())
            {
                return Err(invalid("Hadamard requires a finite power-of-two block"));
            }
            resize_into(up.len(), transformed)?;
            let update = |base_block: usize, up_chunk: &mut [f32], out_chunk: &mut [f32]| {
                for (local_block, (up_block, out_block)) in up_chunk
                    .chunks_exact_mut(block)
                    .zip(out_chunk.chunks_exact_mut(block))
                    .enumerate()
                {
                    let offset = (base_block + local_block) * block;
                    // Equal lengths were checked before any mutation. This
                    // retains the same four-lane pinned SwiGLU boundaries as
                    // the original 256-element worker chunks and scalar path.
                    self.swiglu_in_place(&gate[offset..offset + block], up_block)?;
                    crate::kernels::hadamard_into(
                        up_block,
                        signs.map(|s| &s[offset..offset + block]),
                        out_block,
                    )?;
                }
                Ok(())
            };
            if parallelism.workers() > 1 && block >= 256 && up.len() >= block.saturating_mul(16) {
                parallelism.update_paired_chunks(up, block, transformed, block, update)
            } else {
                update(0, up, transformed)
            }
        })();
        if result.is_err() {
            transformed.clear();
        }
        result
    }
}
