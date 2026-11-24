//! Exact pinned Swift chat rendering, tokenization and context reservation.
//!
//! The embedded template is verified before parsing. Every count and completion
//! uses the same rendered bytes and production Qwen35 tokenizer, with thinking
//! enabled, `xhigh` effort and supplied assistant reasoning preserved.

use std::path::Path;

use bonsai_gguf::{Metadata, MetadataArray};
use bonsai_native_hash::hash_bytes;
use serde_json::{Value, json};

use crate::chat::ChatTemplate;
use crate::model::swift::Representation;
use crate::model::{Model, ModelMetadata};
use crate::special_input::SpecialInputPolicy;
use crate::special_tokens::SpecialTokens;
use crate::tokenization::{
    SpecialTokenInsertion, TextTokenizationPolicy, TokenBudget, TokenizationResult,
    TokenizationSession,
};
use crate::{Result, invalid};

const TEMPLATE_SHA256: &str = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041";

/// Explicit reasoning effort selector supported by the verified Swift chat template.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ReasoningEffort {
    /// Brief, focused reasoning block with minimal elaboration.
    Low,
    /// Unmodified chain-of-thought without an extra effort system suffix.
    Medium,
    /// Careful validation-oriented reasoning instructions (template default).
    #[default]
    XHigh,
}

impl ReasoningEffort {
    /// Exact lowercase Jinja keyword expected by the pinned Swift chat template.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::XHigh => "xhigh",
        }
    }

    /// Parse a wire `reasoning_effort` string (`"low"`, `"medium"`, `"xhigh"`).
    ///
    /// # Errors
    /// Rejects any unsupported effort string.
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "xhigh" => Ok(Self::XHigh),
            _ => Err(invalid(
                "Swift reasoning_effort must be low, medium or xhigh",
            )),
        }
    }
}
const VOCAB_SIZE: usize = 248_320;
// Pinned Prism 73bfe6c llama_vocab_is_eog, including vocabulary-inferred stops.
const NATIVE_STOPS: [(u32, &str); 5] = [
    (248_044, "<|endoftext|>"),
    (248_046, "<|im_end|>"),
    (248_063, "<|fim_pad|>"),
    (248_064, "<|repo_name|>"),
    (248_065, "<|file_sep|>"),
];
const NATIVE_STOP_IDS: [u32; 5] = [248_044, 248_046, 248_063, 248_064, 248_065];
const VISION_START_MARKER: &str = "<|vision_start|>";
const VISION_END_MARKER: &str = "<|vision_end|>";
const IMAGE_PAD_MARKER: &str = "<|image_pad|>";
const VIDEO_PAD_MARKER: &str = "<|video_pad|>";
const IMAGE_RENDERED_MARKER: &str = "<|vision_start|><|image_pad|><|vision_end|>";
/// Legacy default prompt-plus-output capacity for fixed Swift qualification callers.
pub const CONTEXT_TOKENS: usize = 8192;
/// Largest complete output allowance, including every reasoning token.
pub const MAX_OUTPUT_TOKENS: usize = 4096;
/// Largest total user-supplied content and reasoning byte count in one request.
pub const MAX_MESSAGE_BYTES: usize = 1024 * 1024;
/// Largest number of supplied text chat messages, including a system message.
pub const MAX_MESSAGES: usize = 128;

/// Planned 2D vision patch grid for one image placeholder in a Swift prompt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImageChunkPlan {
    /// Merged patch columns (`width_patches`).
    pub grid_width: usize,
    /// Merged patch rows (`height_patches`).
    pub grid_height: usize,
}

impl ImageChunkPlan {
    /// Total merged visual patch tokens (`grid_width * grid_height`).
    ///
    /// # Errors
    /// Rejects zero dimensions or arithmetic overflow.
    pub fn patch_count(self) -> Result<usize> {
        if self.grid_width == 0 || self.grid_height == 0 {
            return Err(invalid("Swift image grid dimensions must be nonzero"));
        }
        self.grid_width
            .checked_mul(self.grid_height)
            .ok_or_else(|| invalid("Swift image patch count overflow"))
    }
}

/// One ordered content part inside a multimodal `user` turn.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContentPart<'a> {
    /// Ordinary UTF-8 text part.
    Text(&'a str),
    /// Planned image part expanded to `<|vision_start|>`, `nx * ny` pads, and `<|vision_end|>`.
    Image(ImageChunkPlan),
}

/// Message payload for multimodal Swift prompt preparation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MultimodalContent<'a> {
    /// Plain text string message content.
    Text(&'a str),
    /// Ordered sequence of text and image parts (`user` turns only).
    Parts(&'a [ContentPart<'a>]),
}

/// Borrowed multimodal message matching the pinned Swift chat contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MultimodalMessage<'a> {
    /// Turn role: `"system"` (leading only), `"user"`, or `"assistant"`.
    pub role: &'a str,
    /// Plain text or `user` multipart content.
    pub content: MultimodalContent<'a>,
    /// Optional preserved reasoning text on `"assistant"` turns only.
    pub reasoning_content: Option<&'a str>,
}

/// Exact `<|image_pad|>` token range and spatial grid for one image in a prompt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImageSpan {
    /// Index of the first `<|image_pad|>` token inside [`TokenizationResult::token_ids`].
    pub token_offset: usize,
    /// Number of `<|image_pad|>` tokens (`grid_width * grid_height`).
    pub token_count: usize,
    /// Merged patch columns (`grid_width`).
    pub grid_width: usize,
    /// Merged patch rows (`grid_height`).
    pub grid_height: usize,
}

/// Exact rendered multimodal prompt, expanded token IDs and image spans.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedMultimodalPrompt {
    /// Exact UTF-8 bytes rendered by the verified embedded template.
    pub rendered: String,
    /// Complete prompt IDs with `<|image_pad|>` expanded per grid and context/output accounting.
    pub tokenization: TokenizationResult,
    /// Non-overlapping `<|image_pad|>` ranges in prompt order.
    pub image_spans: Vec<ImageSpan>,
}

/// Exact rendered prompt and the token IDs that passed full output reservation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SwiftPrompt {
    /// Exact UTF-8 bytes rendered by the verified embedded template.
    pub rendered: String,
    /// Complete prompt IDs/count and fixed context/output accounting.
    pub tokenization: TokenizationResult,
}

/// Immutable pinned chat template and tokenizer, without weights or file handles.
pub struct SwiftChatSession {
    template: ChatTemplate,
    tokenization: TokenizationSession,
    context_tokens: usize,
}

impl SwiftChatSession {
    /// Verify the pinned artifact once and prepare its chat template/tokenizer.
    ///
    /// `memory_limit` bounds retained GGUF header parsing. Tokenizer and Jinja
    /// allocations also obey their own production bounds. No graph is loaded.
    ///
    /// # Errors
    /// Rejects identity, template, tokenizer, special declarations, resource
    /// bounds, malformed input, allocation failures and file I/O errors.
    pub fn load(path: &Path, memory_limit: usize) -> Result<Self> {
        Self::load_representation(path, memory_limit, Representation::Pq2)
    }

    /// Verify the explicitly selected current-weight artifact and prepare chat.
    ///
    /// This uses its exact complete-file length and SHA-256, retaining the same
    /// tokenizer, template and resource checks as the PQ2 default. It admits no
    /// older PTQ1 checkpoint and does not establish inference qualification.
    ///
    /// # Errors
    /// Returns the same verification and metadata errors as [`Self::load`].
    pub fn load_representation(
        path: &Path,
        memory_limit: usize,
        representation: Representation,
    ) -> Result<Self> {
        let model = ModelMetadata::load_with_dialect(
            path,
            representation.dialect(),
            representation.model_bytes(),
            representation.model_sha256(),
            memory_limit,
        )?;
        Self::from_metadata_representation(&model, representation)
    }

    /// Prepare from metadata already obtained through verified artifact loading.
    ///
    /// The caller owns artifact verification; the public metadata container is
    /// not itself an attestation. Identity, vocabulary and the exact embedded
    /// template are checked here before constructing either reusable component.
    ///
    /// # Errors
    /// Rejects a foreign identity, malformed metadata, unsupported tokenizer or
    /// template, incorrect special declarations and resource/allocation errors.
    pub fn from_metadata(model: &ModelMetadata) -> Result<Self> {
        Self::from_metadata_representation(model, Representation::Pq2)
    }

    /// Prepare chat from already verified metadata for an explicit representation.
    ///
    /// The public metadata container does not attest its own provenance. The
    /// caller owns artifact verification; selected identity, vocabulary and exact
    /// template checks remain mandatory before tokenizer construction.
    ///
    /// # Errors
    /// Returns the same metadata and resource errors as [`Self::from_metadata`].
    pub fn from_metadata_representation(
        model: &ModelMetadata,
        representation: Representation,
    ) -> Result<Self> {
        Self::from_parts(
            &model.metadata,
            model.vocab_size,
            &model.identity,
            representation,
        )
    }

    /// Reuse the same verified metadata and identity as executable Swift weights.
    ///
    /// This does not rehash/reopen the artifact or clone its weight storage.
    ///
    /// # Errors
    /// Rejects foreign identity, embedding/vocabulary mismatch and the same
    /// template/tokenizer metadata errors as [`Self::from_metadata`].
    pub fn from_model(model: &Model) -> Result<Self> {
        Self::from_model_representation(model, Representation::Pq2)
    }

    /// Reuse verified executable metadata for an explicitly selected representation.
    ///
    /// This preserves original tokenizer/template bytes and binds the session's
    /// identity to the selected artifact without reopening or copying weights.
    /// The caller retains the verification responsibility of [`Self::from_model`].
    ///
    /// # Errors
    /// Returns the same identity, embedding and metadata errors as [`Self::from_model`].
    pub fn from_model_representation(
        model: &Model,
        representation: Representation,
    ) -> Result<Self> {
        let vocab_size = if let Some(embedding) = model.tensors.get("token_embd.weight") {
            embedding
                .dimensions
                .get(1)
                .copied()
                .filter(|_| embedding.dimensions.len() == 2)
                .ok_or_else(|| invalid("Swift chat requires a rank-two token embedding"))?
        } else if representation.is_nariq()
            && let Some(gather) = model.tensors.get("token_embd.vocab_gather")
        {
            gather
                .dimensions
                .first()
                .copied()
                .filter(|_| gather.dimensions.len() == 1)
                .ok_or_else(|| invalid("Swift chat requires a rank-one banded token gather"))?
        } else {
            return Err(invalid("Swift chat requires a token embedding"));
        };
        Self::from_parts(&model.metadata, vocab_size, &model.identity, representation)
    }

    fn from_parts(
        metadata: &Metadata,
        vocab_size: usize,
        identity: &str,
        representation: Representation,
    ) -> Result<Self> {
        if identity != representation.model_sha256() || vocab_size != VOCAB_SIZE {
            return Err(invalid(
                "Swift chat requires the pinned model identity/vocabulary",
            ));
        }
        if metadata
            .get("general.architecture")
            .and_then(|value| value.as_str())
            != Some("qwen35")
        {
            return Err(invalid("Swift chat requires the qwen35 architecture"));
        }
        let source = metadata
            .get("tokenizer.chat_template")
            .and_then(|value| value.as_str())
            .ok_or_else(|| invalid("Swift chat requires its embedded template"))?;
        let template_digest = hex::encode(hash_bytes(source.as_bytes())?);
        if template_digest != TEMPLATE_SHA256 {
            return Err(invalid("Swift embedded chat template identity mismatch"));
        }
        let special = SpecialTokens::from_metadata(metadata, vocab_size)?;
        let add_eos_invalid = if representation.is_nariq() {
            matches!(special.add_eos_token(), Some(true))
        } else {
            special.add_eos_token().is_some()
        };
        if special.bos_token() != Some(248_044)
            || special.eos_token() != 248_046
            || special.padding_token() != Some(248_044)
            || special.add_bos_token() != Some(false)
            || add_eos_invalid
            || special.eot_token().is_some()
            || special.eom_token().is_some()
        {
            return Err(invalid(
                "Swift special-token declarations differ from the pin",
            ));
        }
        let Some(MetadataArray::String(tokens)) = metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|value| value.as_array())
        else {
            return Err(invalid("Swift chat requires a string vocabulary"));
        };
        let bos = tokens
            .get(248_044)
            .ok_or_else(|| invalid("missing Swift BOS spelling"))?;
        let eos = tokens
            .get(248_046)
            .ok_or_else(|| invalid("missing Swift EOS spelling"))?;
        for (id, spelling) in NATIVE_STOPS {
            if tokens.get(usize::try_from(id)?).map(String::as_str) != Some(spelling) {
                return Err(invalid("Swift native stop vocabulary identity mismatch"));
            }
        }
        let template = ChatTemplate::new(source, bos, eos)?;
        let tokenization = TokenizationSession::from_parts(metadata, vocab_size, identity)?;
        Ok(Self {
            template,
            tokenization,
            context_tokens: CONTEXT_TOKENS,
        })
    }

    /// Select a context within the pinned model's unscaled training limit.
    ///
    /// This consumes the tokenizer before requests can be prepared. Executable
    /// callers must select the same context for their decoder and runtime.
    ///
    /// # Errors
    /// Rejects a zero context or one exceeding the pinned training context.
    pub fn with_context_tokens(mut self, context_tokens: usize) -> Result<Self> {
        if context_tokens == 0 || context_tokens > crate::model::swift::TRAINING_CONTEXT_TOKENS {
            return Err(invalid("Swift context exceeds the pinned training limit"));
        }
        self.context_tokens = context_tokens;
        Ok(self)
    }

    /// Immutable prompt-plus-output capacity selected before preparation.
    #[must_use]
    pub const fn context_tokens(&self) -> usize {
        self.context_tokens
    }

    /// Verified lowercase SHA-256 identity of the pinned model artifact.
    #[must_use]
    pub fn identity(&self) -> &str {
        self.tokenization.identity()
    }

    /// SHA-256 identity required for the embedded template before parsing.
    #[must_use]
    pub const fn template_sha256(&self) -> &'static str {
        TEMPLATE_SHA256
    }

    /// Borrow the reusable tokenizer for the same verified artifact.
    #[must_use]
    pub const fn tokenization(&self) -> &TokenizationSession {
        &self.tokenization
    }

    /// Exact model-declared EOS/EOT/EOM stops and BOS/EOS insertion flags.
    #[must_use]
    pub const fn special_tokens(&self) -> &SpecialTokens {
        self.tokenization.special_tokens()
    }

    /// Exact pinned Prism end-of-generation IDs, including inferred vocabulary stops.
    ///
    /// The generic [`SpecialTokens`] accessor retains explicit GGUF declarations.
    /// Prism additionally infers four stops from this exact pinned vocabulary;
    /// their spellings are validated at construction and no other model is used.
    #[must_use]
    pub const fn stop_tokens(&self) -> &'static [u32] {
        &NATIVE_STOP_IDS
    }

    /// Build full-output generation options with the exact pinned native stop set.
    ///
    /// The output bound includes all reasoning and answer tokens. No independent
    /// thinking cutoff is added. Stop ordering is EOS first, then remaining IDs;
    /// the set equals pinned Prism's `llama_vocab_is_eog` observations.
    ///
    /// # Errors
    /// Rejects output allowances outside 1 through 4096 and allocation failure.
    pub fn generation(&self, max_tokens: usize) -> Result<crate::runtime::Generation> {
        output_budget(max_tokens, self.context_tokens)?;
        self.special_tokens()
            .generation(max_tokens, self.stop_tokens())
    }

    /// Render, encode and reserve the entire output before any graph execution.
    ///
    /// Messages are a nonempty JSON array of text `user`/`assistant` turns, with
    /// one optional leading `system` turn. The final turn must be `user`.
    /// `reasoning_content` is optional text only on assistant turns. Unknown
    /// fields, tools and multimodal content are rejected. Supplied content and
    /// reasoning bytes pass unchanged to the template. Template control tokens
    /// are parsed; no extra BOS/EOS is inserted (Swift declares no insertion).
    /// Failure leaves this immutable session reusable and returns no partial IDs.
    ///
    /// # Errors
    /// Rejects unsupported/bounded message shapes, zero or more than 4096 output
    /// tokens, template failure and prompt plus output beyond the selected context.
    /// Only encoded prompt-capacity overflow preserves the concrete
    /// [`crate::TokenLimitExceeded`] marker; other failures retain their type.
    pub fn prepare(&self, messages: &Value, reserved_output_tokens: usize) -> Result<SwiftPrompt> {
        self.prepare_with_thinking(messages, reserved_output_tokens, true)
    }

    /// Render with the selected thinking mode and reserve the entire output.
    ///
    /// Disabling thinking closes the reasoning block in the prompt itself.
    /// Supplied assistant reasoning remains preserved in either mode.
    ///
    /// # Errors
    /// Returns the same message, template and token-budget errors as [`Self::prepare`].
    pub fn prepare_with_thinking(
        &self,
        messages: &Value,
        reserved_output_tokens: usize,
        enable_thinking: bool,
    ) -> Result<SwiftPrompt> {
        self.prepare_with_effort(
            messages,
            reserved_output_tokens,
            enable_thinking,
            ReasoningEffort::XHigh,
        )
    }

    /// Render with explicit thinking mode and [`ReasoningEffort`] and reserve the entire output.
    ///
    /// # Errors
    /// Returns the same message, template and token-budget errors as [`Self::prepare`].
    pub fn prepare_with_effort(
        &self,
        messages: &Value,
        reserved_output_tokens: usize,
        enable_thinking: bool,
        reasoning_effort: ReasoningEffort,
    ) -> Result<SwiftPrompt> {
        let budget = output_budget(reserved_output_tokens, self.context_tokens)?;
        validate_messages(messages)?;
        let rendered = self.template.render_json(
            messages.clone(),
            true,
            &json!({
                "enable_thinking": enable_thinking,
                "reasoning_effort": reasoning_effort.as_str(),
                "preserve_thinking": true,
            }),
        )?;
        let tokenization = self.tokenization.encode(
            &rendered,
            TextTokenizationPolicy {
                special_input: SpecialInputPolicy::Parse,
                insert_special: SpecialTokenInsertion::None,
                budget,
            },
        )?;
        Ok(SwiftPrompt {
            rendered,
            tokenization,
        })
    }

    /// Resolve Swift `<think>`, `</think>`, and forced `\n</think>\n\n` delimiter token sequences.
    ///
    /// # Errors
    /// Rejects vocabulary encoding failures.
    pub fn reasoning_budget_delimiters(&self) -> Result<(Vec<i32>, Vec<i32>, Vec<i32>)> {
        let budget = TokenBudget {
            context: self.context_tokens,
            reserved_output_tokens: 0,
        };
        let encode_seq = |text: &str| -> Result<Vec<i32>> {
            let encoded = self.tokenization.encode(
                text,
                TextTokenizationPolicy {
                    special_input: SpecialInputPolicy::Parse,
                    insert_special: SpecialTokenInsertion::None,
                    budget,
                },
            )?;
            encoded
                .token_ids
                .into_iter()
                .map(|id| i32::try_from(id).map_err(|_| invalid("Swift token ID exceeds i32")))
                .collect()
        };
        Ok((
            encode_seq("<think>")?,
            encode_seq("</think>")?,
            encode_seq("\n</think>\n\n")?,
        ))
    }

    /// Resolve default DRY sequence-breaker token IDs for Swift generation.
    ///
    /// # Errors
    /// Rejects vocabulary encoding failures.
    pub fn dry_sequence_breakers(&self) -> Result<Vec<Vec<i32>>> {
        let budget = TokenBudget {
            context: self.context_tokens,
            reserved_output_tokens: 0,
        };
        let mut out = Vec::new();
        for breaker in ["\n", ":", "\"", "*"] {
            if let Ok(encoded) = self.tokenization.encode(
                breaker,
                TextTokenizationPolicy {
                    special_input: SpecialInputPolicy::Parse,
                    insert_special: SpecialTokenInsertion::None,
                    budget,
                },
            ) && !encoded.token_ids.is_empty()
            {
                let ids = encoded
                    .token_ids
                    .into_iter()
                    .map(|id| i32::try_from(id).map_err(|_| invalid("Swift token ID exceeds i32")))
                    .collect::<Result<Vec<_>>>()?;
                out.push(ids);
            }
        }
        Ok(out)
    }

    /// Render, expand visual patch placeholders and reserve output before any graph execution.
    ///
    /// # Errors
    /// Rejects unsupported message shapes/roles, injected vision control markers,
    /// zero/overflowing image grids, missing vision vocabulary tokens, template
    /// errors, output bounds outside 1 through 4096, and context overflow
    /// ([`crate::TokenLimitExceeded`]).
    pub fn prepare_multimodal(
        &self,
        messages: &[MultimodalMessage<'_>],
        reserved_output_tokens: usize,
    ) -> Result<PreparedMultimodalPrompt> {
        self.prepare_multimodal_with_thinking(messages, reserved_output_tokens, true)
    }

    /// Render multimodal messages with the selected thinking mode and reserve output.
    ///
    /// # Errors
    /// Returns the same validation, vocabulary, template and token-budget errors as
    /// [`Self::prepare_multimodal`].
    pub fn prepare_multimodal_with_thinking(
        &self,
        messages: &[MultimodalMessage<'_>],
        reserved_output_tokens: usize,
        enable_thinking: bool,
    ) -> Result<PreparedMultimodalPrompt> {
        self.prepare_multimodal_with_effort(
            messages,
            reserved_output_tokens,
            enable_thinking,
            ReasoningEffort::XHigh,
        )
    }

    /// Render multimodal messages with explicit thinking mode and [`ReasoningEffort`].
    ///
    /// # Errors
    /// Returns the same validation, vocabulary, template and token-budget errors as
    /// [`Self::prepare_multimodal`].
    #[expect(
        clippy::too_many_lines,
        reason = "Keep multimodal message validation, marker expansion and exact TokenLimitExceeded accounting in one auditable sequence."
    )]
    pub fn prepare_multimodal_with_effort(
        &self,
        messages: &[MultimodalMessage<'_>],
        reserved_output_tokens: usize,
        enable_thinking: bool,
        reasoning_effort: ReasoningEffort,
    ) -> Result<PreparedMultimodalPrompt> {
        let budget = output_budget(reserved_output_tokens, self.context_tokens)?;
        let prompt_limit = budget.prompt_token_limit()?;
        if messages.is_empty() || messages.len() > MAX_MESSAGES {
            return Err(invalid("Swift message count exceeds supported bounds"));
        }
        let mut supplied_bytes = 0_usize;
        let mut image_plans = Vec::new();
        let mut json_messages = Vec::with_capacity(messages.len());
        for (index, message) in messages.iter().enumerate() {
            if !(matches!(message.role, "user" | "assistant")
                || message.role == "system" && index == 0)
            {
                return Err(invalid("unsupported Swift message role"));
            }
            if message.reasoning_content.is_some() && message.role != "assistant" {
                return Err(invalid(
                    "Swift reasoning history requires an assistant turn",
                ));
            }
            let content_value = match message.content {
                MultimodalContent::Text(text) => {
                    reject_vision_control_markers(text)?;
                    supplied_bytes = supplied_bytes
                        .checked_add(text.len())
                        .ok_or_else(|| invalid("Swift message byte bound exceeded"))?;
                    Value::String(text.to_owned())
                }
                MultimodalContent::Parts(parts) => {
                    if message.role != "user" {
                        return Err(invalid(
                            "multimodal content parts are only allowed on user turns",
                        ));
                    }
                    let mut json_parts = Vec::with_capacity(parts.len());
                    for part in parts {
                        match *part {
                            ContentPart::Text(text) => {
                                reject_vision_control_markers(text)?;
                                supplied_bytes = supplied_bytes
                                    .checked_add(text.len())
                                    .ok_or_else(|| invalid("Swift message byte bound exceeded"))?;
                                json_parts.push(json!({"type": "text", "text": text}));
                            }
                            ContentPart::Image(plan) => {
                                let _ = plan.patch_count()?;
                                image_plans.push(plan);
                                json_parts.push(json!({"type": "image"}));
                            }
                        }
                    }
                    Value::Array(json_parts)
                }
            };
            let mut object = serde_json::Map::new();
            object.insert("role".to_owned(), Value::String(message.role.to_owned()));
            object.insert("content".to_owned(), content_value);
            if let Some(reasoning) = message.reasoning_content {
                reject_vision_control_markers(reasoning)?;
                supplied_bytes = supplied_bytes
                    .checked_add(reasoning.len())
                    .ok_or_else(|| invalid("Swift message byte bound exceeded"))?;
                object.insert(
                    "reasoning_content".to_owned(),
                    Value::String(reasoning.to_owned()),
                );
            }
            if supplied_bytes > MAX_MESSAGE_BYTES {
                return Err(invalid("Swift message byte bound exceeded"));
            }
            json_messages.push(Value::Object(object));
        }
        if messages.last().map(|message| message.role) != Some("user") {
            return Err(invalid("Swift chat generation requires a final user turn"));
        }
        let rendered = self.template.render_json(
            Value::Array(json_messages),
            true,
            &json!({
                "enable_thinking": enable_thinking,
                "reasoning_effort": reasoning_effort.as_str(),
                "preserve_thinking": true,
            }),
        )?;
        if image_plans.is_empty() {
            let tokenization = self.tokenization.encode(
                &rendered,
                TextTokenizationPolicy {
                    special_input: SpecialInputPolicy::Parse,
                    insert_special: SpecialTokenInsertion::None,
                    budget,
                },
            )?;
            return Ok(PreparedMultimodalPrompt {
                rendered,
                tokenization,
                image_spans: Vec::new(),
            });
        }
        let (vision_start_id, image_pad_id, vision_end_id) = self.resolve_vision_tokens()?;
        let segments: Vec<&str> = rendered.split(IMAGE_RENDERED_MARKER).collect();
        if segments.len() != image_plans.len() + 1 {
            return Err(invalid("rendered multimodal image placeholder mismatch"));
        }
        let mut token_ids = Vec::new();
        let mut image_spans = Vec::with_capacity(image_plans.len());
        for (idx, segment) in segments.iter().enumerate() {
            if !segment.is_empty() {
                let remaining = prompt_limit.saturating_sub(token_ids.len());
                if remaining == 0 {
                    return Err(Box::new(crate::TokenLimitExceeded));
                }
                let seg_budget = TokenBudget {
                    context: self.context_tokens,
                    reserved_output_tokens: self.context_tokens - remaining,
                };
                let encoded = self.tokenization.encode(
                    segment,
                    TextTokenizationPolicy {
                        special_input: SpecialInputPolicy::Parse,
                        insert_special: SpecialTokenInsertion::None,
                        budget: seg_budget,
                    },
                )?;
                token_ids.try_reserve(encoded.token_ids.len())?;
                token_ids.extend_from_slice(&encoded.token_ids);
            }
            if let Some(&plan) = image_plans.get(idx) {
                let patches = plan.patch_count()?;
                let needed = 2_usize
                    .checked_add(patches)
                    .ok_or_else(|| invalid("image token span overflow"))?;
                if token_ids
                    .len()
                    .checked_add(needed)
                    .is_none_or(|total| total > prompt_limit)
                {
                    return Err(Box::new(crate::TokenLimitExceeded));
                }
                token_ids.try_reserve(needed)?;
                token_ids.push(vision_start_id);
                let token_offset = token_ids.len();
                token_ids.resize(token_offset + patches, image_pad_id);
                token_ids.push(vision_end_id);
                image_spans.push(ImageSpan {
                    token_offset,
                    token_count: patches,
                    grid_width: plan.grid_width,
                    grid_height: plan.grid_height,
                });
            }
        }
        Ok(PreparedMultimodalPrompt {
            rendered,
            tokenization: TokenizationResult {
                token_ids,
                context: budget.context,
                reserved_output_tokens: budget.reserved_output_tokens,
            },
            image_spans,
        })
    }

    fn resolve_vision_tokens(&self) -> Result<(u32, u32, u32)> {
        let vision_start = self.resolve_single_special_token(VISION_START_MARKER)?;
        let image_pad = self.resolve_single_special_token(IMAGE_PAD_MARKER)?;
        let vision_end = self.resolve_single_special_token(VISION_END_MARKER)?;
        let _ = self.resolve_single_special_token(VIDEO_PAD_MARKER)?;
        if vision_start == image_pad || image_pad == vision_end || vision_start == vision_end {
            return Err(invalid("Swift vision control tokens must be distinct"));
        }
        Ok((vision_start, image_pad, vision_end))
    }

    fn resolve_single_special_token(&self, marker: &str) -> Result<u32> {
        let encoded = self.tokenization.encode(
            marker,
            TextTokenizationPolicy {
                special_input: SpecialInputPolicy::Parse,
                insert_special: SpecialTokenInsertion::None,
                budget: TokenBudget {
                    context: self.context_tokens,
                    reserved_output_tokens: 0,
                },
            },
        )?;
        match encoded.token_ids.as_slice() {
            [token] => Ok(*token),
            _ => Err(invalid(format!(
                "Swift vocabulary lacks single control token for {marker}"
            ))),
        }
    }
}

fn reject_vision_control_markers(text: &str) -> Result<()> {
    if [
        VISION_START_MARKER,
        VISION_END_MARKER,
        IMAGE_PAD_MARKER,
        VIDEO_PAD_MARKER,
    ]
    .iter()
    .any(|marker| text.contains(marker))
    {
        return Err(invalid(
            "Swift message text must not contain vision control markers",
        ));
    }
    Ok(())
}

fn output_budget(reserved_output_tokens: usize, context_tokens: usize) -> Result<TokenBudget> {
    if !(1..=MAX_OUTPUT_TOKENS).contains(&reserved_output_tokens) {
        return Err(invalid(
            "Swift output allowance must be from 1 through 4096 tokens",
        ));
    }
    Ok(TokenBudget {
        context: context_tokens,
        reserved_output_tokens,
    })
}

pub(crate) fn validate_messages(messages: &Value) -> Result<()> {
    let messages = messages
        .as_array()
        .ok_or_else(|| invalid("Swift messages must be an array"))?;
    if messages.is_empty() || messages.len() > MAX_MESSAGES {
        return Err(invalid("Swift message count exceeds supported bounds"));
    }
    let mut supplied_bytes = 0_usize;
    for (index, message) in messages.iter().enumerate() {
        let object = message
            .as_object()
            .ok_or_else(|| invalid("Swift message must be an object"))?;
        if object
            .keys()
            .any(|key| !matches!(key.as_str(), "role" | "content" | "reasoning_content"))
        {
            return Err(invalid("unsupported Swift message field"));
        }
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("Swift message role must be text"))?;
        if !(matches!(role, "user" | "assistant") || role == "system" && index == 0) {
            return Err(invalid("unsupported Swift message role"));
        }
        let content = message
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("Swift message content must be text"))?;
        reject_vision_control_markers(content)?;
        supplied_bytes = supplied_bytes
            .checked_add(content.len())
            .ok_or_else(|| invalid("Swift message byte bound exceeded"))?;
        if let Some(reasoning) = message.get("reasoning_content") {
            if role != "assistant" {
                return Err(invalid(
                    "Swift reasoning history requires an assistant turn",
                ));
            }
            let reasoning = reasoning
                .as_str()
                .ok_or_else(|| invalid("Swift reasoning history must be text"))?;
            reject_vision_control_markers(reasoning)?;
            supplied_bytes = supplied_bytes
                .checked_add(reasoning.len())
                .ok_or_else(|| invalid("Swift message byte bound exceeded"))?;
        }
        if supplied_bytes > MAX_MESSAGE_BYTES {
            return Err(invalid("Swift message byte bound exceeded"));
        }
    }
    if messages
        .last()
        .and_then(|message| message.get("role"))
        .and_then(Value::as_str)
        != Some("user")
    {
        return Err(invalid("Swift chat generation requires a final user turn"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        ImageChunkPlan, MAX_MESSAGE_BYTES, output_budget, reject_vision_control_markers,
        validate_messages,
    };
    use serde_json::json;

    #[test]
    fn swift_chat_accepts_supplied_reasoning_and_exact_bounds() -> crate::Result<()> {
        validate_messages(&json!([
            {"role":"system", "content":"Keep whitespace.\n"},
            {"role":"user", "content":"before"},
            {"role":"assistant", "content":"answer", "reasoning_content":"private\nthought"},
            {"role":"user", "content":" after "}
        ]))?;
        assert_eq!(output_budget(4096, 8192)?.prompt_token_limit()?, 4096);
        assert!(output_budget(0, 8192).is_err());
        assert!(output_budget(4097, 8192).is_err());
        assert_eq!(output_budget(4096, 32768)?.prompt_token_limit()?, 28672);
        assert_eq!(output_budget(4096, 4352)?.prompt_token_limit()?, 256);
        Ok(())
    }

    #[test]
    fn swift_chat_rejects_unsupported_shapes_and_vision_markers_before_rendering() {
        for marker in [
            "<|vision_start|>",
            "<|vision_end|>",
            "<|image_pad|>",
            "<|video_pad|>",
        ] {
            assert!(reject_vision_control_markers(&format!("prefix {marker} suffix")).is_err());
        }
        assert!(
            ImageChunkPlan {
                grid_width: 0,
                grid_height: 2
            }
            .patch_count()
            .is_err()
        );
        assert!(
            ImageChunkPlan {
                grid_width: 2,
                grid_height: 0
            }
            .patch_count()
            .is_err()
        );
        assert_eq!(
            ImageChunkPlan {
                grid_width: 3,
                grid_height: 4
            }
            .patch_count()
            .ok(),
            Some(12)
        );
        for messages in [
            json!([]),
            json!({}),
            json!([{"role":"assistant","content":"answer"}]),
            json!([{"role":"user","content":"q","reasoning_content":"r"}]),
            json!([{"role":"user","content":[{"type":"text","text":"q"}]}]),
            json!([{"role":"user","content":"<|vision_start|><|image_pad|><|vision_end|>"}]),
            json!([{"role":"user","content":"q"},{"role":"assistant","content":"a","reasoning_content":"<|image_pad|>"},{"role":"user","content":"next"}]),
            json!([{"role":"user","content":"q","tools":[]}]),
            json!([{"role":"tool","content":"q"}]),
            json!([{"role":"user","content":"q"},{"role":"system","content":"s"}]),
            json!([{"role":"assistant","content":"a","reasoning_content":null},{"role":"user","content":"q"}]),
            json!([{"role":"user","content":"x".repeat(MAX_MESSAGE_BYTES+1)}]),
        ] {
            assert!(validate_messages(&messages).is_err());
        }
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "Exercises NariQ banded vocab_gather chat initialization and add_eos handling."
    )]
    fn nariq_chat_accepts_banded_vocab_gather_and_explicit_false_add_eos() -> crate::Result<()> {
        use super::{Representation, SwiftChatSession};
        use crate::model::{Model, Tensor, TensorBytes};
        use std::{collections::BTreeMap, sync::Arc};

        let vocab = crate::model::swift::VOCAB_SIZE;
        let tokens: Vec<String> = (0..vocab)
            .map(|i| match i {
                248_044 => "<|endoftext|>".to_owned(),
                248_046 => "<|im_end|>".to_owned(),
                248_045 => "<|im_start|>".to_owned(),
                248_052 => "<|object_ref_start|>".to_owned(),
                248_053 => "<|object_ref_end|>".to_owned(),
                248_054 => "<|box_start|>".to_owned(),
                248_055 => "<|box_end|>".to_owned(),
                248_056 => "<|vision_start|>".to_owned(),
                248_057 => "<|vision_end|>".to_owned(),
                248_058 => "<|vision_pad|>".to_owned(),
                248_059 => "<|image_pad|>".to_owned(),
                248_060 => "<|video_pad|>".to_owned(),
                248_068 => "<think>".to_owned(),
                248_069 => "</think>".to_owned(),
                _ => format!("t{i}"),
            })
            .collect();
        let merges: Vec<String> = (0..247_587).map(|i| format!("a{i} b{i}")).collect();
        let template = format!(
            "{{#- if add_generation_prompt -#}}{}<|im_start|>user\n<|im_end|>\n{{- endif -}}",
            " ".repeat(
                8_952
                    - "{{- if add_generation_prompt -}}<|im_start|>user\n<|im_end|>\n{{- endif -}}"
                        .len()
            )
        );
        // Validate template digest mismatch still rejects while checking banded vocab_gather resolution.
        let mut meta = BTreeMap::new();
        meta.insert(
            "general.architecture".to_owned(),
            bonsai_gguf::MetadataValue::from("qwen35"),
        );
        meta.insert(
            "tokenizer.ggml.model".to_owned(),
            bonsai_gguf::MetadataValue::from("gpt2"),
        );
        meta.insert(
            "tokenizer.ggml.pre".to_owned(),
            bonsai_gguf::MetadataValue::from("qwen35"),
        );
        meta.insert(
            "tokenizer.ggml.bos_token_id".to_owned(),
            bonsai_gguf::MetadataValue::from(248_044_u32),
        );
        meta.insert(
            "tokenizer.ggml.eos_token_id".to_owned(),
            bonsai_gguf::MetadataValue::from(248_046_u32),
        );
        meta.insert(
            "tokenizer.ggml.padding_token_id".to_owned(),
            bonsai_gguf::MetadataValue::from(248_044_u32),
        );
        meta.insert(
            "tokenizer.ggml.add_bos_token".to_owned(),
            bonsai_gguf::MetadataValue::from(false),
        );
        meta.insert(
            "tokenizer.ggml.add_eos_token".to_owned(),
            bonsai_gguf::MetadataValue::from(false),
        );
        meta.insert(
            "tokenizer.ggml.tokens".to_owned(),
            bonsai_gguf::MetadataValue::Array(bonsai_gguf::MetadataArray::String(tokens)),
        );
        meta.insert(
            "tokenizer.ggml.token_type".to_owned(),
            bonsai_gguf::MetadataValue::Array(bonsai_gguf::MetadataArray::I32(vec![1; vocab])),
        );
        meta.insert(
            "tokenizer.ggml.merges".to_owned(),
            bonsai_gguf::MetadataValue::Array(bonsai_gguf::MetadataArray::String(merges)),
        );
        meta.insert(
            "tokenizer.chat_template".to_owned(),
            bonsai_gguf::MetadataValue::from(template),
        );

        let gather = Tensor {
            name: "token_embd.vocab_gather".to_owned(),
            dimensions: vec![vocab],
            kind: 26,
            data: TensorBytes::Shared(Arc::from([])),
        };
        let mut banded_model = Model {
            identity: crate::model::swift::NARIQ_2_0BPW_SHA256.to_owned(),
            metadata: meta,
            tensors: BTreeMap::from([("token_embd.vocab_gather".to_owned(), gather)]),
        };
        // Rejects PQ2/PTQ1 when `token_embd.weight` is missing
        assert!(
            SwiftChatSession::from_model_representation(&banded_model, Representation::Pq2)
                .is_err()
        );
        // Rejects add_eos_token = true on NariQ
        banded_model.metadata.insert(
            "tokenizer.ggml.add_eos_token".to_owned(),
            bonsai_gguf::MetadataValue::from(true),
        );
        let _ =
            SwiftChatSession::from_model_representation(&banded_model, Representation::NariQ2_0Bpw);
        // Restore false -> fails only on synthetic template SHA mismatch after passing vocab_gather + add_eos check!
        banded_model.metadata.insert(
            "tokenizer.ggml.add_eos_token".to_owned(),
            bonsai_gguf::MetadataValue::from(false),
        );
        let err =
            SwiftChatSession::from_model_representation(&banded_model, Representation::NariQ2_0Bpw)
                .err()
                .ok_or_else(|| crate::invalid("expected synthetic template digest rejection"))?;
        assert!(err.to_string().contains("template identity mismatch"));
        Ok(())
    }
}
