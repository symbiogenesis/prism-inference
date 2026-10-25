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
}

/// Render, tokenize and admit an already authenticated strict broker request.
///
/// The existing broker canonicalization fixes sampling and thinking settings.
/// Counting and execution share the prepared token allocation, and preserve the
/// existing conservative template margin in addition to exact accounting.
///
/// # Errors
/// Returns value-free schema/count/context rejection; no prompt escapes in errors.
pub fn prepare(session: &SwiftSession, pending: PendingChat) -> Result<NativeChat, Rejection> {
    let mut prepared = None;
    let admitted = pending.admit(|body| {
        let value: serde_json::Value =
            serde_json::from_slice(body).map_err(|_| Rejection::ChatSchema)?;
        let messages = value.get("messages").ok_or(Rejection::ChatSchema)?;
        let output = value
            .get("max_completion_tokens")
            .and_then(serde_json::Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or(Rejection::CompletionLimit)?;
        let prompt = session
            .prepare(messages, output)
            .map_err(|_| Rejection::TokenCountUnavailable)?;
        let count = prompt.prompt_tokens();
        prepared = Some(prompt);
        Ok(count)
    })?;
    let prompt = prepared.ok_or(Rejection::TokenCountUnavailable)?;
    Ok(NativeChat { admitted, prompt })
}

/// Stream one admitted request on its original verified native session.
///
/// The owner propagates disconnects through `cancellation` or callback errors,
/// and supplies the same absolute deadline used during HTTP intake and admission.
/// Deltas remain provisional until terminal completion succeeds. Failed terminal
/// delivery requires a successful session reset before another request.
///
/// # Errors
/// Returns bounded session, cancellation, deadline, execution or delivery errors.
pub fn complete(
    session: &mut SwiftSession,
    chat: &NativeChat,
    cancellation: &Cancellation,
    deadline: Instant,
    emit: impl FnMut(TextEvent<'_>) -> bonsai_prism_runtime::Result<()>,
) -> bonsai_prism_runtime::Result<Completion> {
    session.complete(&chat.prompt, cancellation, deadline, emit)
}
