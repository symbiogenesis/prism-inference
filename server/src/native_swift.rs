//! Explicit native Swift adapter behind the existing authorized request policy.
//!
//! This synchronous boundary owns no listener, process or task. The caller keeps
//! existing authentication, reservation, process-memory and ownership policies.
//! Borrowing a mutable session permits only one active native generation.

use std::time::Instant;

use bonsai_prism_runtime::{
    runtime::{Cancellation, Completion},
    swift_session::{PreparedChat, SwiftSession, TextEvent},
};

use crate::{AdmittedChat, PendingChat, Rejection};

/// Exact-count admission bound to the same prepared input used for completion.
pub struct NativeChat {
    admitted: AdmittedChat,
    prompt: PreparedChat,
}

impl NativeChat {
    /// Canonical authorized request retained for the response policy.
    #[must_use]
    pub const fn admitted(&self) -> &AdmittedChat {
        &self.admitted
    }

    /// Exact rendered prompt count used by context admission.
    #[must_use]
    pub const fn prompt_tokens(&self) -> usize {
        self.prompt.prompt_tokens()
    }

    /// Thinking mode used to render the exact counted prompt.
    #[must_use]
    pub const fn thinking_enabled(&self) -> bool {
        self.prompt.thinking_enabled()
    }
}

/// Render, tokenize and admit an already authenticated strict broker request.
///
/// The existing broker canonicalization fixes sampling and selects thinking.
/// Counting and execution share the prepared token allocation, and preserve the
/// existing conservative template margin in addition to exact accounting.
///
/// # Errors
/// Returns value-free schema/count/context rejection; no prompt escapes in errors.
pub fn prepare(session: &SwiftSession, pending: PendingChat) -> Result<NativeChat, Rejection> {
    let mut prepared = None;
    let admitted = pending.admit_with_context(session.context_tokens(), |body| {
        let value: serde_json::Value =
            serde_json::from_slice(body).map_err(|_| Rejection::ChatSchema)?;
        let messages = value.get("messages").ok_or(Rejection::ChatSchema)?;
        let enable_thinking = value["chat_template_kwargs"]["enable_thinking"]
            .as_bool()
            .ok_or(Rejection::ChatSchema)?;
        let output = value
            .get("max_completion_tokens")
            .and_then(serde_json::Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or(Rejection::CompletionLimit)?;
        let adapter_scale = value
            .get("lora")
            .map(|lora| {
                serde_json::from_value::<f32>(lora[0]["scale"].clone())
                    .map_err(|_| Rejection::ChatSchema)
            })
            .transpose()?;
        let prompt = session
            .prepare_with_settings(messages, output, enable_thinking, adapter_scale)
            .map_err(|error| classify_preparation_error(error.as_ref()))?;
        let count = prompt.prompt_tokens();
        prepared = Some(prompt);
        Ok(count)
    })?;
    let prompt = prepared.ok_or(Rejection::TokenCountUnavailable)?;
    Ok(NativeChat { admitted, prompt })
}

fn classify_preparation_error(
    error: &(dyn std::error::Error + Send + Sync + 'static),
) -> Rejection {
    if error.is::<bonsai_prism_runtime::TokenLimitExceeded>() {
        Rejection::ContextLimit
    } else {
        Rejection::TokenCountUnavailable
    }
}

/// Stream one admitted request on its original verified native session.
///
/// The owner propagates disconnects through `cancellation` or callback errors,
/// and supplies the same optional deadline used during HTTP intake and admission.
/// Deltas remain provisional until terminal completion succeeds. Failed terminal
/// delivery requires a successful session reset before another request.
///
/// # Errors
/// Returns bounded session, cancellation, deadline, execution or delivery errors.
pub fn complete(
    session: &mut SwiftSession,
    chat: &NativeChat,
    cancellation: &Cancellation,
    deadline: impl Into<Option<Instant>>,
    emit: impl FnMut(TextEvent<'_>) -> bonsai_prism_runtime::Result<()>,
) -> bonsai_prism_runtime::Result<Completion> {
    session.complete(&chat.prompt, cancellation, deadline, emit)
}

/// Complete with value-free phase attribution retained on returned failures.
///
/// Uses the same ownership, cancellation and delivery contract as [`complete`].
/// Loading, request rendering and outer HTTP delivery remain outside this record.
///
/// # Errors
/// Returns the errors of [`complete`]; CPU speculation does not support attribution.
pub fn complete_with_timings(
    session: &mut SwiftSession,
    chat: &NativeChat,
    cancellation: &Cancellation,
    deadline: impl Into<Option<Instant>>,
    timings: &mut bonsai_prism_runtime::runtime::GenerationTimings,
    emit: impl FnMut(TextEvent<'_>) -> bonsai_prism_runtime::Result<()>,
) -> bonsai_prism_runtime::Result<Completion> {
    session.complete_with_timings(&chat.prompt, cancellation, deadline, timings, emit)
}

#[cfg(test)]
mod tests {
    use super::classify_preparation_error;
    use crate::Rejection;

    #[test]
    fn native_swift_context_classification_requires_concrete_token_limit_error() {
        assert_eq!(
            classify_preparation_error(&bonsai_prism_runtime::TokenLimitExceeded),
            Rejection::ContextLimit
        );
        for message in [
            "token output exceeds token limit",
            "invalid UTF-8",
            "missing byte vocabulary",
            "byte limit exceeded",
            "allocation failed",
        ] {
            assert_eq!(
                classify_preparation_error(&std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    message
                )),
                Rejection::TokenCountUnavailable
            );
        }
        let bytes = std::hint::black_box([u8::MAX]);
        let utf8 = std::str::from_utf8(&bytes).err();
        assert!(utf8.is_some());
        if let Some(error) = utf8 {
            assert_eq!(
                classify_preparation_error(&error),
                Rejection::TokenCountUnavailable
            );
        }
    }
}
