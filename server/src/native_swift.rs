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

pub(crate) struct CanonicalRequest<'a> {
    pub(crate) messages: &'a serde_json::Value,
    pub(crate) output_tokens: usize,
    pub(crate) enable_thinking: bool,
    pub(crate) adapter_scale: Option<f32>,
    pub(crate) multi_rank_scale: Option<f32>,
}

pub(crate) fn parse_canonical_request(
    value: &serde_json::Value,
    has_rank_one: bool,
    has_multi_rank: bool,
) -> Result<CanonicalRequest<'_>, Rejection> {
    let messages = value.get("messages").ok_or(Rejection::ChatSchema)?;
    let enable_thinking = value["chat_template_kwargs"]["enable_thinking"]
        .as_bool()
        .ok_or(Rejection::ChatSchema)?;
    let output_tokens = value
        .get("max_completion_tokens")
        .and_then(serde_json::Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or(Rejection::CompletionLimit)?;
    let (adapter_scale, multi_rank_scale) =
        parse_adapter_scales(value, has_rank_one, has_multi_rank)?;
    Ok(CanonicalRequest {
        messages,
        output_tokens,
        enable_thinking,
        adapter_scale,
        multi_rank_scale,
    })
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
        let request = parse_canonical_request(
            &value,
            session.adapter_identity().is_some(),
            session.secondary_adapter_identity().is_some(),
        )?;
        let prompt = session
            .prepare_with_adapter_scales(
                request.messages,
                request.output_tokens,
                request.enable_thinking,
                request.adapter_scale,
                request.multi_rank_scale,
            )
            .map_err(|error| classify_preparation_error(error.as_ref()))?;
        let count = prompt.prompt_tokens();
        prepared = Some(prompt);
        Ok(count)
    })?;
    let prompt = prepared.ok_or(Rejection::TokenCountUnavailable)?;
    Ok(NativeChat { admitted, prompt })
}

fn parse_scale_value(raw_val: &serde_json::Value) -> Result<f32, Rejection> {
    let raw = serde_json::from_value::<f64>(raw_val.clone()).map_err(|_| Rejection::ChatSchema)?;
    if !raw.is_finite() || !(0.0..=f64::from(f32::MAX)).contains(&raw) {
        return Err(Rejection::ChatSchema);
    }
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "Scale is validated finite and within [0.0, f32::MAX] before narrowing to binary32."
    )]
    let scale = raw as f32;
    if !scale.is_finite() || scale < 0.0 {
        return Err(Rejection::ChatSchema);
    }
    Ok(scale)
}

#[cfg(test)]
fn parse_adapter_scale(value: &serde_json::Value) -> Result<Option<f32>, Rejection> {
    let Some(lora) = value.get("lora") else {
        return Ok(None);
    };
    let Some(entries) = lora.as_array() else {
        return Err(Rejection::ChatSchema);
    };
    if entries.len() != 1 {
        return Err(Rejection::ChatSchema);
    }
    Ok(Some(parse_scale_value(&entries[0]["scale"])?))
}

pub(crate) fn parse_adapter_scales(
    value: &serde_json::Value,
    has_rank_one: bool,
    has_multi_rank: bool,
) -> Result<(Option<f32>, Option<f32>), Rejection> {
    let Some(lora) = value.get("lora") else {
        return Ok((None, has_multi_rank.then_some(0.0)));
    };
    let Some(entries) = lora.as_array() else {
        return Err(Rejection::ChatSchema);
    };
    if !(1..=2).contains(&entries.len()) {
        return Err(Rejection::ChatSchema);
    }
    let mut slot0: Option<f32> = None;
    let mut slot1: Option<f32> = None;
    for entry in entries {
        let id = entry
            .get("id")
            .and_then(serde_json::Value::as_u64)
            .ok_or(Rejection::ChatSchema)?;
        let scale = parse_scale_value(&entry["scale"])?;
        match id {
            0 if slot0.is_none() => slot0 = Some(scale),
            1 if slot1.is_none() => slot1 = Some(scale),
            _ => return Err(Rejection::ChatSchema),
        }
    }
    match (has_rank_one, has_multi_rank) {
        (false, false) => Err(Rejection::ChatSchema),
        (true, false) => match (slot0, slot1) {
            (Some(r1), None) => Ok((Some(r1), None)),
            _ => Err(Rejection::ChatSchema),
        },
        (false, true) => match (slot0, slot1) {
            (Some(mr), None) => Ok((None, Some(mr))),
            _ => Err(Rejection::ChatSchema),
        },
        (true, true) => match (slot0, slot1) {
            (Some(r1), Some(mr)) => Ok((Some(r1), Some(mr))),
            (Some(r1), None) => Ok((Some(r1), Some(0.0))),
            _ => Err(Rejection::ChatSchema),
        },
    }
}

pub(crate) fn classify_preparation_error(
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
    use super::{classify_preparation_error, parse_adapter_scale, parse_adapter_scales};
    use crate::Rejection;
    use serde_json::json;

    #[test]
    fn parse_adapter_scales_maps_stacked_ids_and_defaults_daisyrose_to_zero() {
        assert_eq!(
            parse_adapter_scales(&json!({}), false, false),
            Ok((None, None))
        );
        assert_eq!(
            parse_adapter_scales(&json!({}), false, true),
            Ok((None, Some(0.0)))
        );
        assert_eq!(
            parse_adapter_scales(&json!({"lora": [{"id": 0, "scale": 1.3}]}), true, true),
            Ok((Some(1.3), Some(0.0)))
        );
        assert_eq!(
            parse_adapter_scales(
                &json!({"lora": [{"id": 0, "scale": 1.3}, {"id": 1, "scale": 0.5}]}),
                true,
                true
            ),
            Ok((Some(1.3), Some(0.5)))
        );
        assert_eq!(
            parse_adapter_scales(
                &json!({"lora": [{"id": 1, "scale": 0.5}, {"id": 0, "scale": 1.3}]}),
                true,
                true
            ),
            Ok((Some(1.3), Some(0.5)))
        );
        assert_eq!(
            parse_adapter_scales(&json!({"lora": [{"id": 0, "scale": 0.75}]}), false, true),
            Ok((None, Some(0.75)))
        );
        for (body, r1, mr) in [
            (json!({"lora": [{"id": 1, "scale": 0.5}]}), true, true),
            (
                json!({"lora": [{"id": 0, "scale": 1.3}, {"id": 0, "scale": 0.5}]}),
                true,
                true,
            ),
            (
                json!({"lora": [{"id": 0, "scale": 1.3}, {"id": 1, "scale": 0.5}]}),
                true,
                false,
            ),
            (json!({"lora": [{"id": 0, "scale": 0.5}]}), false, false),
        ] {
            assert_eq!(
                parse_adapter_scales(&body, r1, mr),
                Err(Rejection::ChatSchema),
                "expected rejection for {body} (r1={r1}, mr={mr})"
            );
        }
    }

    #[test]
    fn parse_adapter_scale_validates_f64_before_binary32_narrowing() {
        assert_eq!(parse_adapter_scale(&json!({})), Ok(None));
        assert_eq!(
            parse_adapter_scale(&json!({"lora": [{"scale": 0.0}]})),
            Ok(Some(0.0))
        );
        assert_eq!(
            parse_adapter_scale(&json!({"lora": [{"scale": 1.25}]})),
            Ok(Some(1.25))
        );
        for invalid in [
            json!({"lora": null}),
            json!({"lora": []}),
            json!({"lora": [{}]}),
            json!({"lora": [{"scale": null}]}),
            json!({"lora": [{"scale": "1.0"}]}),
            json!({"lora": [{"scale": -0.001}]}),
            json!({"lora": [{"scale": -1.0}]}),
            json!({"lora": [{"scale": 1e39}]}),
            json!({"lora": [{"scale": 3.402_823_6e38_f64}]}),
        ] {
            assert_eq!(
                parse_adapter_scale(&invalid),
                Err(Rejection::ChatSchema),
                "unexpected acceptance for {invalid}"
            );
        }
    }

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
