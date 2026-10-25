//! Artifact-verified output pieces for reusable service-side detokenization.
//!
//! Loading verifies the complete artifact, builds the bounded vocabulary, then
//! releases its metadata and file handle. The session retains only the verified
//! identity and immutable decoded pieces. It allocates no graph or weights and
//! owns no request text, generated-token history, or display-stream state.

use std::path::Path;

use crate::Result;
use crate::model::ModelMetadata;
use crate::vocabulary::Vocabulary;

/// Reusable lossless output vocabulary from one verified model artifact.
///
/// Borrowed pieces preserve every byte, including incomplete UTF-8, NUL bytes
/// and declared control spellings. The caller chooses whether to display a
/// control token and retains original bytes when lossless output is required.
/// This session does not apply stop rules, chat templates or normalization.
/// Artifact identity does not establish inference validity or full-model parity.
///
/// Each request should own an independent [`crate::text_output::Utf8Output`].
/// Feed admitted pieces in order, finish at successful end of stream, and reset
/// on cancellation before reusing a stream. The session has no mutable request
/// state, so a failed lookup cannot affect it or any caller-owned stream:
///
/// ```no_run
/// # use bonsai_prism_runtime::{Result, text_output::Utf8Output,
/// #     text_session::TextOutputSession};
/// # fn example(session: &TextOutputSession, token: u32) -> Result<()> {
/// let mut display = Utf8Output::new();
/// let bytes = session.piece(token)?;
/// // Explicit policy: display control spellings as well as normal pieces.
/// let chunk = display.push(bytes)?;
/// let final_chunk = display.finish();
/// # let _ = (chunk, final_chunk);
/// # Ok(())
/// # }
/// ```
pub struct TextOutputSession {
    identity: String,
    vocabulary: Vocabulary,
}

impl std::fmt::Debug for TextOutputSession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TextOutputSession")
            .field("identity", &self.identity)
            .field("vocab_size", &self.vocab_size())
            .finish_non_exhaustive()
    }
}

impl TextOutputSession {
    /// Verify a complete artifact once and retain its bounded output pieces.
    ///
    /// The embedding descriptor supplies the vocabulary size. All pieces are
    /// built from the same verified metadata before it is dropped; neither a
    /// model file handle nor tensor payloads remain in the returned session.
    /// Unsupported graph architectures and nonfinite executable tensor values
    /// do not prevent output-piece construction from valid tokenizer metadata.
    ///
    /// `memory_limit` bounds retained GGUF header storage during loading, not
    /// process memory or the vocabulary. [`Vocabulary::from_metadata`] enforces
    /// independent token-count, per-piece and aggregate decoded-byte limits.
    /// Loading temporarily holds both the header and the constructed vocabulary.
    /// No caller output bytes are accepted, retained or included in load errors.
    /// Debug output includes only artifact identity and vocabulary size.
    ///
    /// # Errors
    /// Rejects artifact size/hash mismatches, I/O failures, invalid or over-budget
    /// headers, missing/rank-mismatched embeddings, unsupported tokenizer
    /// metadata, malformed pieces, resource limits and allocation failures.
    pub fn load(
        path: &Path,
        expected_bytes: u64,
        expected_sha256: &str,
        memory_limit: usize,
    ) -> Result<Self> {
        let model = ModelMetadata::load(path, expected_bytes, expected_sha256, memory_limit)?;
        let vocabulary = Vocabulary::from_metadata(&model.metadata, model.vocab_size)?;
        Ok(Self {
            identity: model.identity,
            vocabulary,
        })
    }

    /// Lowercase SHA-256 identity verified at session construction.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Verified embedding vocabulary size used for output-ID validation.
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.vocabulary.vocab_size()
    }

    /// Borrow exact output bytes without allocating or interpreting UTF-8.
    ///
    /// The slice belongs to this immutable session and may be held across other
    /// lookups. Control spellings are returned without omission. Lookup errors
    /// contain no decoded piece or caller output, and leave the session intact.
    ///
    /// # Errors
    /// Rejects out-of-range, unknown and unused IDs. Call this before mutating
    /// a caller-owned display stream so a rejected ID preserves its suffix.
    pub fn piece(&self, id: u32) -> Result<&[u8]> {
        self.vocabulary.piece(id)
    }

    /// Report a declared control type without guessing from its spelling.
    ///
    /// This does not decide whether an ID should stop generation or be displayed.
    /// Unknown and unused IDs are in range but do not have renderable pieces.
    ///
    /// # Errors
    /// Rejects IDs outside the verified vocabulary without changing the session.
    pub fn is_control(&self, id: u32) -> Result<bool> {
        self.vocabulary.is_control(id)
    }
}
