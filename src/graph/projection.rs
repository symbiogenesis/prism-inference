//! Scoped projection execution with unchanged graph-state transactions.

use super::{Qwen35, Qwen35Checkpoint};
use crate::{Decoder, Result, invalid, kernels::Pq2ProjectionExecutor};

/// Experimental decoder borrowing one synchronous projection executor.
///
/// Construct this inside the owned native session callback, then place it in an
/// ordinary [`crate::runtime::Runtime`] for complete request checkpoints. The
/// borrow keeps one session alive across all eligible projections, tokens and
/// request phases without per-projection module open/close. It neither extends a
/// terminal native session's operation/deadline limits nor enables a service
/// profile. Unsupported projection shapes stay on CPU before submission.
///
/// Any evaluation error permanently closes this scope. Reset or restoration
/// cannot authorize further native work; the owner must finish verified teardown
/// or end its supervised process. Device consumers and resources are outside
/// graph checkpoints and must never be restored by replaying a failed projection.
/// Checkpoints intentionally contain the same CPU-owned committed state as the
/// underlying graph. A checkpoint taken from that graph before wrapping remains
/// valid while this scope is healthy; foreign graph/model provenance still
/// rejects. Native queues, in-flight buffers and device resources are excluded.
pub struct ProjectedQwen35<'a> {
    decoder: Qwen35,
    executor: &'a mut dyn Pq2ProjectionExecutor,
    identity: String,
    faulted: bool,
}

impl Qwen35 {
    /// Borrow one explicit executor while retaining native graph and adapter order.
    ///
    /// This API selects no device, loads no driver and admits no production
    /// accelerator. The caller independently owns signed deployment, exact device
    /// identity, resource admission, an absolute process deadline and teardown.
    /// The executor identity distinguishes scoped runtime caches from CPU placement.
    ///
    /// # Errors
    /// Rejects empty, non-ASCII or oversized executor identities and allocation.
    pub fn with_projection_executor(
        self,
        executor: &mut dyn Pq2ProjectionExecutor,
    ) -> Result<ProjectedQwen35<'_>> {
        let placement = executor.identity();
        if placement.is_empty()
            || placement.len() > 256
            || !placement.bytes().all(|byte| byte.is_ascii_graphic())
        {
            return Err(invalid("invalid projection executor identity"));
        }
        let mut identity = String::new();
        identity.try_reserve_exact(self.identity.len() + placement.len() + 32)?;
        identity.push_str(&self.identity);
        identity.push_str(":external-pq2-v1:");
        identity.push_str(placement);
        Ok(ProjectedQwen35 {
            decoder: self,
            executor,
            identity,
            faulted: false,
        })
    }
}

impl ProjectedQwen35<'_> {
    /// Read-only underlying graph for state and CPU-fallback diagnostics.
    #[must_use]
    pub const fn decoder(&self) -> &Qwen35 {
        &self.decoder
    }

    /// Whether this scope rejected work and requires terminal owner teardown.
    #[must_use]
    pub const fn is_faulted(&self) -> bool {
        self.faulted
    }

    fn available(&self) -> Result<()> {
        if self.faulted {
            Err(invalid(
                "projection executor scope requires terminal teardown",
            ))
        } else {
            Ok(())
        }
    }

    fn evaluate_token(&mut self, token: u32, logits: Option<&mut Vec<f32>>) -> Result<()> {
        self.available()?;
        let result = self
            .decoder
            .evaluate_token_with_executor(token, logits, Some(self.executor));
        self.faulted = result.is_err();
        result
    }
}

impl Decoder for ProjectedQwen35<'_> {
    type Checkpoint = Qwen35Checkpoint;

    fn identity(&self) -> &str {
        &self.identity
    }
    fn vocab_size(&self) -> usize {
        self.decoder.vocab_size()
    }
    fn checkpoint_bytes(&self) -> usize {
        self.decoder.checkpoint_bytes()
    }
    fn checkpoint(&self) -> Result<Self::Checkpoint> {
        self.available()?;
        self.decoder.checkpoint()
    }
    fn restore(&mut self, checkpoint: &Self::Checkpoint) -> Result<()> {
        self.available()?;
        self.decoder.restore(checkpoint)
    }
    fn reset(&mut self) -> Result<()> {
        self.available()?;
        self.decoder.reset()
    }
    fn evaluate(&mut self, token: u32) -> Result<Vec<f32>> {
        let mut logits = Vec::new();
        self.evaluate_into(token, &mut logits)?;
        Ok(logits)
    }
    fn evaluate_into(&mut self, token: u32, output: &mut Vec<f32>) -> Result<()> {
        output.clear();
        let result = self.evaluate_token(token, Some(output));
        if result.is_err() {
            output.clear();
        }
        result
    }
    fn ingest(&mut self, token: u32) -> Result<()> {
        self.evaluate_token(token, None)
    }
}
