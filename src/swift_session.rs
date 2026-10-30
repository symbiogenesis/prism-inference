//! One pinned Swift conversation joining exact chat input and transactional text output.
//!
//! Readiness verifies identity and executable graph construction. Full-model
//! correctness and device qualification require separately retained comparisons.

use std::{path::Path, sync::Arc, time::Instant};

use serde_json::Value;

use crate::{
    Result,
    graph::Qwen35,
    invalid,
    kernels::{MatvecBackend, RowParallelism},
    model::{
        lora::RankOneAdapter,
        swift::{self, LoadReceipt, Representation},
    },
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
    owner: Arc<()>,
    adapter_scale: Option<f32>,
    tokens: Vec<u32>,
    output_tokens: usize,
    enable_thinking: bool,
}

impl PreparedChat {
    /// Exact correction scale reserved by this request, when an adapter is loaded.
    #[must_use]
    pub const fn adapter_scale(&self) -> Option<f32> {
        self.adapter_scale
    }

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

    /// Whether the exact prepared prompt starts an open reasoning block.
    #[must_use]
    pub const fn thinking_enabled(&self) -> bool {
        self.enable_thinking
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
    owner: Arc<()>,
    representation: Representation,
    adapter_identity: Option<String>,
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
        Self::load_with_row_parallelism(path, backend, seed, RowParallelism::default())
    }

    /// Loads the pinned session with an explicitly constructed row-worker pool.
    ///
    /// Default loading remains serial. The pool is shared by packed projections
    /// and survives reset; existing transaction/cancellation behavior is retained.
    ///
    /// # Errors
    /// Returns the same loading, identity and state errors as [`Self::load`].
    pub fn load_with_row_parallelism(
        path: &Path,
        backend: MatvecBackend,
        seed: u32,
        parallelism: RowParallelism,
    ) -> Result<Self> {
        Self::load_profile_with_row_parallelism(
            path,
            Representation::Pq2,
            None,
            backend,
            seed,
            parallelism,
        )
    }

    /// Load one exact Swift representation and an optional approved rank-one adapter.
    ///
    /// Selection is explicit and immutable for this session. Only the approved
    /// adapter artifact is admitted. This executable path does not qualify model
    /// arithmetic, authoring basis, device resources, quality or a serving default.
    ///
    /// # Errors
    /// Rejects artifact, representation, adapter, graph and storage-budget drift.
    pub fn load_profile_with_row_parallelism(
        path: &Path,
        representation: Representation,
        adapter_path: Option<&Path>,
        backend: MatvecBackend,
        seed: u32,
        parallelism: RowParallelism,
    ) -> Result<Self> {
        let sampler = SwiftSampler::new(seed)?;
        let (model, receipt) =
            swift::load_representation(path, MODEL_MEMORY_BYTES, representation)?;
        let chat = SwiftChatSession::from_model_representation(&model, representation)?;
        let vocabulary = Vocabulary::from_metadata(&model.metadata, swift::VOCAB_SIZE)?;
        let adapter = adapter_path
            .map(|path| RankOneAdapter::load_swift_representation(path, &model, representation))
            .transpose()?;
        let adapter_identity = adapter
            .as_ref()
            .map(|_| crate::model::lora::SWIFT_ADAPTER_SHA256.to_owned());
        let mut decoder = if let Some(adapter) = adapter {
            Qwen35::with_rank_one_adapter(
                model,
                swift::CONTEXT_TOKENS,
                STATE_MEMORY_BYTES,
                backend,
                adapter,
                1.3,
            )?
        } else {
            Qwen35::with_state_budget(model, swift::CONTEXT_TOKENS, STATE_MEMORY_BYTES, backend)?
        };
        decoder.set_row_parallelism(parallelism);
        let runtime = Runtime::new(
            decoder,
            Limits {
                context_tokens: swift::CONTEXT_TOKENS,
                checkpoints: 0,
                checkpoint_bytes: CHECKPOINT_MEMORY_BYTES,
            },
        )?;
        Ok(Self {
            owner: Arc::new(()),
            representation,
            adapter_identity,
            runtime,
            chat,
            vocabulary,
            sampler,
            seed,
            delivery_fault: false,
            receipt,
        })
    }

    /// Exact immutable model representation selected for this session.
    #[must_use]
    pub const fn representation(&self) -> Representation {
        self.representation
    }

    /// Exact approved adapter artifact SHA-256, when one was explicitly loaded.
    #[must_use]
    pub fn adapter_identity(&self) -> Option<&str> {
        self.adapter_identity.as_deref()
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
        self.prepare_with_thinking(messages, output_tokens, true)
    }

    /// Render and encode once with the selected thinking mode.
    ///
    /// The prepared prompt retains its mode for matching response parsing.
    ///
    /// # Errors
    /// Returns the same admission and prompt errors as [`Self::prepare`].
    pub fn prepare_with_thinking(
        &self,
        messages: &Value,
        output_tokens: usize,
        enable_thinking: bool,
    ) -> Result<PreparedChat> {
        self.prepare_with_settings(
            messages,
            output_tokens,
            enable_thinking,
            self.adapter_identity.as_ref().map(|_| 1.3),
        )
    }

    /// Seal exact prompt tokens, thinking mode and an explicit adapter scale.
    ///
    /// A loaded adapter requires `Some` finite scale in 0..2; an adapter-free
    /// session requires `None`. The prompt belongs only to this session, across
    /// explicit resets; it cannot be completed by another session or profile.
    ///
    /// # Errors
    /// Rejects closed admission, mismatched settings and all normal prompt bounds.
    pub fn prepare_with_settings(
        &self,
        messages: &Value,
        output_tokens: usize,
        enable_thinking: bool,
        adapter_scale: Option<f32>,
    ) -> Result<PreparedChat> {
        validate_adapter_scale(self.adapter_identity.is_some(), adapter_scale)?;
        if self.is_faulted() {
            return Err(invalid("Swift session requires reset"));
        }
        let prompt = self
            .chat
            .prepare_with_thinking(messages, output_tokens, enable_thinking)?;
        Ok(PreparedChat {
            owner: self.owner.clone(),
            adapter_scale,
            tokens: prompt.tokenization.token_ids,
            output_tokens,
            enable_thinking,
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
        validate_prompt_owner(&self.owner, prompt)?;
        validate_adapter_scale(self.adapter_identity.is_some(), prompt.adapter_scale)?;
        check_request(cancellation, deadline)?;
        // Every request starts with empty model state. The adapter path also
        // renews graph identity/provenance, so no scale's checkpoints survive.
        // Sampler draws retain the existing session behavior until explicit reset.
        if let Some(scale) = prompt.adapter_scale {
            self.runtime.reset_with_adapter_scale(scale)?;
        } else {
            self.runtime.reset()?;
        }
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

fn validate_prompt_owner(owner: &Arc<()>, prompt: &PreparedChat) -> Result<()> {
    if Arc::ptr_eq(owner, &prompt.owner) {
        Ok(())
    } else {
        Err(invalid("prepared Swift prompt belongs to another session"))
    }
}

fn validate_adapter_scale(loaded: bool, scale: Option<f32>) -> Result<()> {
    match (loaded, scale) {
        (false, None) => Ok(()),
        (true, Some(scale)) => RankOneAdapter::validate_scale(scale),
        _ => Err(invalid(
            "Swift adapter scale does not match the loaded profile",
        )),
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

#[cfg(test)]
mod tests {
    use super::{PreparedChat, validate_adapter_scale, validate_prompt_owner};
    use crate::Result;
    use std::sync::Arc;

    #[test]
    fn prepared_chat_seals_session_and_exact_adapter_settings() -> Result<()> {
        let owner = Arc::new(());
        let prompt = PreparedChat {
            owner: owner.clone(),
            adapter_scale: Some(-0.0),
            tokens: vec![1, 2],
            output_tokens: 3,
            enable_thinking: false,
        };
        validate_prompt_owner(&owner, &prompt)?;
        assert!(validate_prompt_owner(&Arc::new(()), &prompt).is_err());
        assert_eq!(prompt.prompt_tokens(), 2);
        assert_eq!(prompt.output_tokens(), 3);
        assert!(!prompt.thinking_enabled());
        assert_eq!(
            prompt.adapter_scale().map(f32::to_bits),
            Some((-0.0_f32).to_bits())
        );
        assert!(validate_adapter_scale(false, None).is_ok());
        assert!(validate_adapter_scale(true, None).is_err());
        for scale in [0.0, -0.0, 1.3, 2.0] {
            validate_adapter_scale(true, Some(scale))?;
            assert!(validate_adapter_scale(false, Some(scale)).is_err());
        }
        for scale in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.01, 2.01] {
            assert!(validate_adapter_scale(true, Some(scale)).is_err());
        }
        Ok(())
    }
}
