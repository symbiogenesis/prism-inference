//! One pinned Swift conversation joining exact chat input and transactional text output.
//!
//! Readiness verifies identity and executable graph construction. Full-model
//! correctness and device qualification require separately retained comparisons.

use std::{path::Path, time::Instant};

use serde_json::Value;

use crate::{
    Result,
    graph::Qwen35,
    invalid,
    kernels::MatvecBackend,
    model::swift::{self, LoadReceipt},
    runtime::{Cancellation, Completion, Limits, Prefill, Runtime},
    sampling::SwiftSampler,
    swift_chat::SwiftChatSession,
    text_output::Utf8Output,
    vocabulary::Vocabulary,
};

/// Maximum retained model storage admitted by this Swift session.
pub const MODEL_MEMORY_BYTES: usize = 8 * 1024 * 1024 * 1024;
/// Maximum live FP16 KV and F32 recurrent/convolution payload.
pub const STATE_MEMORY_BYTES: usize = 1024 * 1024 * 1024;
/// Maximum complete transactional rollback snapshot storage.
pub const CHECKPOINT_MEMORY_BYTES: usize = 1024 * 1024 * 1024;
/// Largest complete displayed output retained or delivered by one request.
pub const MAX_OUTPUT_BYTES: usize = 4 * 1024 * 1024;

/// Exact rendered and counted input, sealed by its originating verified session.
pub struct PreparedChat {
    tokens: Vec<u32>,
    output_tokens: usize,
}

impl PreparedChat {
    /// Exact number of tokens in the complete rendered prompt.
    #[must_use]
    pub const fn prompt_tokens(&self) -> usize {
        self.tokens.len()
    }

    /// Complete output allowance reserved before model execution.
    #[must_use]
    pub const fn output_tokens(&self) -> usize {
        self.output_tokens
    }
}

/// A provisional text event; callers commit displayed output only on completion.
#[derive(Debug)]
pub enum TextEvent<'a> {
    /// Incrementally decoded UTF-8, including a possible final replacement suffix.
    Delta(&'a str),
    /// Terminal accounting. Failure to deliver this event closes admission.
    Complete(Completion),
}

/// Single-owner verified Swift runtime, tokenizer, output vocabulary and sampler.
///
/// Mutable access serializes generation and reset. Every request starts model
/// state from empty, disabling prompt reuse. Sampler draws persist until reset;
/// cancelled or failed generation restores the pre-request sampler. A failed
/// terminal delivery closes admission until an explicit successful reset.
pub struct SwiftSession {
    runtime: Runtime<Qwen35>,
    chat: SwiftChatSession,
    vocabulary: Vocabulary,
    sampler: SwiftSampler,
    seed: u32,
    delivery_fault: bool,
    receipt: LoadReceipt,
}

impl SwiftSession {
    /// Load the pinned immutable artifact and establish bounded executable state.
    ///
    /// The separate storage/state/checkpoint caps exclude allocator overhead,
    /// bounded graph scratch and temporary loader copies. They do not replace
    /// the existing process-memory admission and watchdog policies.
    ///
    /// # Errors
    /// Rejects artifact/metadata/inventory drift, graph errors or allocation bounds.
    pub fn load(path: &Path, backend: MatvecBackend, seed: u32) -> Result<Self> {
        let sampler = SwiftSampler::new(seed)?;
        let (model, receipt) = swift::load(path, MODEL_MEMORY_BYTES)?;
        let chat = SwiftChatSession::from_model(&model)?;
        let vocabulary = Vocabulary::from_metadata(&model.metadata, swift::VOCAB_SIZE)?;
        let decoder =
            Qwen35::with_state_budget(model, swift::CONTEXT_TOKENS, STATE_MEMORY_BYTES, backend)?;
        let runtime = Runtime::new(
            decoder,
            Limits {
                context_tokens: swift::CONTEXT_TOKENS,
                checkpoints: 0,
                checkpoint_bytes: CHECKPOINT_MEMORY_BYTES,
            },
        )?;
        Ok(Self {
            runtime,
            chat,
            vocabulary,
            sampler,
            seed,
            delivery_fault: false,
            receipt,
        })
    }

    /// Verified loading and bounded storage accounting for this model.
    #[must_use]
    pub const fn receipt(&self) -> &LoadReceipt {
        &self.receipt
    }

    /// Explicit reproducible seed restored by a successful reset.
    #[must_use]
    pub const fn seed(&self) -> u32 {
        self.seed
    }

    /// Requested CPU dispatch policy; individual projections may fall back.
    #[must_use]
    pub const fn requested_backend(&self) -> MatvecBackend {
        self.runtime.decoder().requested_backend()
    }

    /// Whether reset is required before another completion can start.
    #[must_use]
    pub const fn is_faulted(&self) -> bool {
        self.delivery_fault || self.runtime.is_faulted()
    }

    /// Render and encode once, reserving the entire output allowance.
    ///
    /// # Errors
    /// Rejects malformed history, output/context overflow and closed admission.
    pub fn prepare(&self, messages: &Value, output_tokens: usize) -> Result<PreparedChat> {
        if self.is_faulted() {
            return Err(invalid("Swift session requires reset"));
        }
        let prompt = self.chat.prepare(messages, output_tokens)?;
        Ok(PreparedChat {
            tokens: prompt.tokenization.token_ids,
            output_tokens,
        })
    }

    /// Clear model and display state and restore the initial seeded sampler.
    ///
    /// # Errors
    /// Failed reset keeps admission closed until a subsequent reset succeeds.
    pub fn reset(&mut self) -> Result<()> {
        self.delivery_fault = true;
        self.runtime.reset()?;
        self.sampler = SwiftSampler::new(self.seed)?;
        self.delivery_fault = false;
        Ok(())
    }

    /// Generate sampled text under the complete output cap and absolute deadline.
    ///
    /// Callbacks provide backpressure and must report failed delivery. Deadlines
    /// and cancellation are cooperative between synchronous decoder operations.
    /// Provisional deltas are abandoned when the method fails. Failure during
    /// terminal UTF-8/accounting delivery latches admission closed until reset.
    ///
    /// # Errors
    /// Rejects closed admission, cancellation, expiry, output limits and I/O or
    /// model failures. Generation failures transactionally restore model/sampler.
    pub fn complete(
        &mut self,
        prompt: &PreparedChat,
        cancellation: &Cancellation,
        deadline: Instant,
        mut emit: impl FnMut(TextEvent<'_>) -> Result<()>,
    ) -> Result<Completion> {
        if self.is_faulted() {
            return Err(invalid("Swift session requires reset"));
        }
        // An empty runtime at every request forbids both live-prefix and saved
        // checkpoint reuse while retaining the transactional failure contract.
        self.runtime.reset()?;
        let options = self.chat.generation(prompt.output_tokens)?;
        let mut display = Utf8Output::new();
        let mut output_bytes = 0usize;
        let completion = self.runtime.generate_sampled_until(
            &prompt.tokens,
            &options,
            Prefill { block_tokens: 64 },
            cancellation,
            deadline,
            &mut self.sampler,
            |token| {
                let text = display.push(self.vocabulary.piece(token)?)?;
                charge_output(&mut output_bytes, text.len())?;
                if !text.is_empty() {
                    emit(TextEvent::Delta(&text))?;
                }
                Ok(())
            },
        )?;
        self.delivery_fault = true;
        let suffix = display.finish();
        charge_output(&mut output_bytes, suffix.len())?;
        if !suffix.is_empty() {
            emit(TextEvent::Delta(suffix))?;
        }
        check_request(cancellation, deadline)?;
        emit(TextEvent::Complete(completion))?;
        check_request(cancellation, deadline)?;
        self.delivery_fault = false;
        Ok(completion)
    }
}

fn charge_output(total: &mut usize, bytes: usize) -> Result<()> {
    *total = total
        .checked_add(bytes)
        .filter(|&n| n <= MAX_OUTPUT_BYTES)
        .ok_or_else(|| invalid("Swift output exceeds bounded byte allowance"))?;
    Ok(())
}

fn check_request(cancellation: &Cancellation, deadline: Instant) -> Result<()> {
    if cancellation.is_cancelled() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Interrupted,
            "Swift request cancelled",
        )
        .into());
    }
    if Instant::now() >= deadline {
        return Err(
            std::io::Error::new(std::io::ErrorKind::TimedOut, "Swift request expired").into(),
        );
    }
    Ok(())
}
