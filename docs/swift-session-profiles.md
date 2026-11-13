# Swift session profiles

`SwiftSession::load` and `load_with_row_parallelism` load the pinned original
PQ2 model without an adapter. `load_profile_with_row_parallelism` explicitly
selects original PQ2 or its exact current-weight PTQ1 export and an optional
approved [rank-one adapter](rank-one-adapter.md). Each selection verifies the
complete artifact and metadata; arbitrary model-path fallback is unavailable.

Ordinary loaders select K4/V4 (`rsq4`). Use
`load_profile_with_kv_cache(..., KvCachePolicy::F16)` for FP16 comparison.
[KV precision](quantized-kv.md), representation and adapter identity are immutable
for a session. `representation()` and `adapter_identity()` expose the latter two.
These experimental profiles require actual-model and device qualification.

## Request settings and ownership

`prepare_with_settings(messages, output_tokens, enable_thinking, adapter_scale)`
renders and counts once. An adapter requires an explicit finite `Some(scale)`
within 0..2; an adapter-free session requires `None`. Both signed zeros bypass
correction arithmetic but retain distinct graph identities. `prepare` and
`prepare_with_thinking` use scale 1.3 when an adapter is loaded.

`PreparedChat` seals tokens, output reservation, thinking mode and scale to its
originating session. It carries no mutable graph state and remains replayable
after that session's explicit reset. Another session cannot complete it.

By default, completions start from empty model state.
`Runtime<Qwen35>::reset_with_adapter_scale` clears tokens, logits and prefixes
after renewing decoder provenance. `Qwen35::reset_with_adapter_scale` retains
immutable weights/factors, clears selected KV and F32 recurrent/convolution state,
and invalidates every old checkpoint, including at the same scale or after
changing back. Invalid scales and absent adapters reject before mutation.
Identity-allocation failure preserves the graph but closes runtime admission
until reset succeeds.

Cancellation/expiry is checked before request reset. Generation rolls back model
and sampler state on failure; failed final text/accounting delivery closes
admission until reset. Successful requests and scale changes preserve sampler
progression. `SwiftSession::reset` clears model state and restores the initial
seeded sampler.

## Latest-prefix reuse

`set_prompt_reuse(PromptReuse::Latest)` enables the latest committed live sequence
as an exact prefix candidate. The private server accepts `--prompt-reuse latest`;
omission or `disabled` selects fresh recomputation. JSONL workers always recompute.
Enabled HTTP readiness at `/v1/models` and `/props` includes
`"prompt_reuse":"latest"`. Counting never mutates live state.

Reuse requires the same model, adapter and bit-identical scale. Any scale-bit
change clears state and renews checkpoint provenance. Edited or switched histories
without the complete live prefix recompute from empty; recurrent state cannot
be truncated. Cancellation or provisional-delivery failure restores the committed
state and sampler at an unchanged scale; after a scale change, rollback preserves
the new scale's empty state. Changing reuse policy clears live state while
preserving sampler progression. No saved-prefix cache or budget discount is added.

Graph and private-service controls cover fresh/reused state and logits, RNG
progression, page boundaries, edited histories, scale changes, cancellation,
count seals, delivery failure and explicit reset. Actual-model turn re-encoding,
service lifecycle, resources and latency remain separate qualification gates.

## Terminal state-only ingestion

With `PromptReuse::Disabled`, the final non-stop token at the output cap uses
state-only ingestion: complete KV/recurrent/convolution state is checked, while
unused output normalization/projection is omitted. Earlier tokens compute logits.
A stop token consumes its sampler draw but enters neither state nor output.

Callback, cancellation and deadline checks surround ingestion within the request
transaction. Failure clears logits and restores the committed snapshot; failed
restore closes admission. The crate-private service path requires zero retained
prefix checkpoints and clears terminal logits on success. An internal continuation
therefore recomputes its prompt. Public Runtime APIs, latest-prefix reuse and the
greedy suffix path retain complete terminal logits.

Timing adds `terminal-ingestion` after the eleven existing phases.
`decode-evaluation` counts only generated-token evaluations that compute logits.
Conservative work and memory admission still reserve complete evaluation costs.

## Bounded request preparation

Special-token splitting uses reverse-trie windows of at most 4,096 positions in
an 8 KiB array of 16-bit indices. Each window scans up to 255 overlap bytes
(the longest spelling minus one), preserving earliest-start/longest-match
selection across boundaries. A match crossing a boundary advances the next window
past the complete spelling. Byte scanning may cross UTF-8 scalars; only validated
complete spellings create text-slice boundaries.

Segments grow fallibly within the limits: 1 MiB input, 256 bytes per spelling,
4,096 specials and 65,536 segments. Text segments borrow the input; literal mode
uses its direct path. Recognition finishes before BPE, preserving segmentation
errors ahead of token-output-limit errors. No match cache survives a request.

See [runtime memory accounting](runtime-contracts.md#memory-and-ownership) and
[validation](build-and-validation.md) for shared limits and qualification procedures.
