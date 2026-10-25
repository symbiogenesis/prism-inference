//! Exact pinned Swift chat rendering, tokenization and context reservation.
//!
//! The embedded template is verified before parsing. Every count and completion
//! uses the same rendered bytes and production Qwen35 tokenizer, with thinking
//! enabled, `xhigh` effort and supplied assistant reasoning preserved.

use std::path::Path;

use bonsai_gguf::{Metadata, MetadataArray};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::chat::ChatTemplate;
use crate::model::{Model, ModelMetadata};
use crate::special_input::SpecialInputPolicy;
use crate::special_tokens::SpecialTokens;
use crate::tokenization::{
    SpecialTokenInsertion, TextTokenizationPolicy, TokenBudget, TokenizationResult,
    TokenizationSession,
};
use crate::{Result, invalid};

const MODEL_BYTES: u64 = 7_206_168_928;
const MODEL_SHA256: &str = "5912bb739217cf25b4283a098c7b893d9baff36bdf2e72d3dbf4cb999d5633d5";
const TEMPLATE_SHA256: &str = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041";
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
/// Fixed prompt-plus-output context capacity for the Swift CPU MVP.
pub const CONTEXT_TOKENS: usize = 8192;
/// Largest complete output allowance, including every reasoning token.
pub const MAX_OUTPUT_TOKENS: usize = 4096;
/// Largest total user-supplied content and reasoning byte count in one request.
pub const MAX_MESSAGE_BYTES: usize = 1024 * 1024;
/// Largest number of supplied text chat messages, including a system message.
pub const MAX_MESSAGES: usize = 128;

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
        let model = ModelMetadata::load(path, MODEL_BYTES, MODEL_SHA256, memory_limit)?;
        Self::from_metadata(&model)
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
        Self::from_parts(&model.metadata, model.vocab_size, &model.identity)
    }

    /// Reuse the same verified metadata and identity as executable Swift weights.
    ///
    /// This does not rehash/reopen the artifact or clone its weight storage.
    ///
    /// # Errors
    /// Rejects foreign identity, embedding/vocabulary mismatch and the same
    /// template/tokenizer metadata errors as [`Self::from_metadata`].
    pub fn from_model(model: &Model) -> Result<Self> {
        let embedding = model
            .tensors
            .get("token_embd.weight")
            .ok_or_else(|| invalid("Swift chat requires a token embedding"))?;
        let vocab_size = embedding
            .dimensions
            .get(1)
            .copied()
            .filter(|_| embedding.dimensions.len() == 2)
            .ok_or_else(|| invalid("Swift chat requires a rank-two token embedding"))?;
        Self::from_parts(&model.metadata, vocab_size, &model.identity)
    }

    fn from_parts(metadata: &Metadata, vocab_size: usize, identity: &str) -> Result<Self> {
        if identity != MODEL_SHA256 || vocab_size != VOCAB_SIZE {
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
        let template_digest: String = Sha256::digest(source.as_bytes())
            .into_iter()
            .flat_map(|byte| [byte >> 4, byte & 15])
            .map(|nibble| char::from(b"0123456789abcdef"[usize::from(nibble)]))
            .collect();
        if template_digest != TEMPLATE_SHA256 {
            return Err(invalid("Swift embedded chat template identity mismatch"));
        }
        let special = SpecialTokens::from_metadata(metadata, vocab_size)?;
        if special.bos_token() != Some(248_044)
            || special.eos_token() != 248_046
            || special.padding_token() != Some(248_044)
            || special.add_bos_token() != Some(false)
            || special.add_eos_token().is_some()
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
        })
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
        output_budget(max_tokens)?;
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
    /// tokens, template failure and prompt plus output beyond 8192 tokens.
    pub fn prepare(&self, messages: &Value, reserved_output_tokens: usize) -> Result<SwiftPrompt> {
        let budget = output_budget(reserved_output_tokens)?;
        validate_messages(messages)?;
        let rendered = self.template.render_json(
            messages.clone(),
            true,
            &json!({"enable_thinking": true, "reasoning_effort": "xhigh", "preserve_thinking": true}),
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
}

fn output_budget(reserved_output_tokens: usize) -> Result<TokenBudget> {
    if !(1..=MAX_OUTPUT_TOKENS).contains(&reserved_output_tokens) {
        return Err(invalid(
            "Swift output allowance must be from 1 through 4096 tokens",
        ));
    }
    Ok(TokenBudget {
        context: CONTEXT_TOKENS,
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
    use super::{MAX_MESSAGE_BYTES, output_budget, validate_messages};
    use serde_json::json;

    #[test]
    fn swift_chat_accepts_supplied_reasoning_and_exact_bounds() -> crate::Result<()> {
        validate_messages(&json!([
            {"role":"system", "content":"Keep whitespace.\n"},
            {"role":"user", "content":"before"},
            {"role":"assistant", "content":"answer", "reasoning_content":"private\nthought"},
            {"role":"user", "content":" after "}
        ]))?;
        assert_eq!(output_budget(4096)?.prompt_token_limit()?, 4096);
        assert!(output_budget(0).is_err());
        assert!(output_budget(4097).is_err());
        Ok(())
    }

    #[test]
    fn swift_chat_rejects_unsupported_shapes_before_rendering() {
        for messages in [
            json!([]),
            json!({}),
            json!([{"role":"assistant","content":"answer"}]),
            json!([{"role":"user","content":"q","reasoning_content":"r"}]),
            json!([{"role":"user","content":[{"type":"text","text":"q"}]}]),
            json!([{"role":"user","content":"q","tools":[]}]),
            json!([{"role":"tool","content":"q"}]),
            json!([{"role":"user","content":"q"},{"role":"system","content":"s"}]),
            json!([{"role":"assistant","content":"a","reasoning_content":null},{"role":"user","content":"q"}]),
            json!([{"role":"user","content":"x".repeat(MAX_MESSAGE_BYTES+1)}]),
        ] {
            assert!(validate_messages(&messages).is_err());
        }
    }
}
