# Explicit Swift session profiles

`SwiftSession::load` and `load_with_row_parallelism` retain the original pinned
PQ2 model without an adapter. The explicit
`load_profile_with_row_parallelism(path, representation, adapter_path, backend,
seed, parallelism)` API admits either original PQ2 or its exact current-weight
PTQ1 export, with an optional approved rank-one adapter. It uses each selected
representation's full artifact and metadata verification. There is no arbitrary
checkpoint or model-path fallback.

Representation and adapter identity are immutable for the session. The accessors
`representation()` and `adapter_identity()` expose the selected format and exact
approved adapter artifact SHA-256. Model receipt accounting retains its original
scope; adapter storage, graph scratch and process overhead are separate. Source
support does not lower process admission or establish native authoring basis,
full-model arithmetic, resource, speed, quality or release qualification.

## Request settings and ownership

`prepare_with_settings(messages, output_tokens, enable_thinking, adapter_scale)`
renders and counts once. A loaded adapter requires an explicit `Some(scale)`
finite within 0..2; an adapter-free session requires `None`, including when the
caller would otherwise supply zero. Both signed zero scales bypass correction
arithmetic while retaining their exact scale bits in graph identity. Existing
`prepare` and `prepare_with_thinking` use scale 1.3 when an adapter was explicitly
loaded; their original adapter-free behavior is unchanged.

`PreparedChat` seals the exact token sequence, output reservation, thinking mode
and scale to its originating session. Another session cannot complete it, even
when both sessions selected identical files and settings. A prepared prompt can
be replayed after an explicit reset of its original session, preserving the
existing seeded replay API. It never carries mutable graph state.

## Fresh state and experimental exact reuse

By default every completion starts from empty model state. Adapter requests use
`Runtime<Qwen35>::reset_with_adapter_scale`, which clears committed tokens,
logits and saved prefixes only after renewing the decoder's empty-state identity.
`Qwen35::reset_with_adapter_scale` reuses immutable weights and factors, clears
FP16 KV and F32 convolution/recurrent state, and creates fresh checkpoint
provenance. No prior checkpoint can restore afterward, including after the same
scale or a change back to an earlier scale. Neither API exposes general mutable
decoder access or duplicates the model.

Invalid scales and absent adapters reject before mutation. Graph identity
allocation failure preserves the old graph; the owning runtime closes admission
until successful explicit reset. A cancelled or expired request is checked before
request-state reset. Generation retains transactional model and sampler rollback;
failed final text/accounting delivery closes session admission until reset.

Sampler draws retain the established session behavior across successful requests
and scale changes. Scale reset does not silently reseed the sampler. Explicit
`SwiftSession::reset` clears model state and restores the initial seeded sampler.
The per-session prepared-prompt seal remains valid across that reset.

`set_prompt_reuse(PromptReuse::Latest)` explicitly enables the latest committed
live sequence as an exact prefix candidate. The private server accepts
`--native-engine-child ... --prompt-reuse latest`; omission and explicit
`--prompt-reuse disabled` retain fresh recomputation. Unknown values and duplicate
options reject before loading. The JSONL Swift worker retains fresh recomputation.
Only selected native HTTP readiness adds `"prompt_reuse":"latest"`, allowing an
owner to verify the explicit policy without changing the default response shape.

The live candidate is eligible only within its original immutable model/adapter
profile and at bit-identical scale. A scale change, including positive/negative
zero or change-back, clears state and renews checkpoint provenance before execution.
An edited or switched token sequence without the complete live prefix recomputes
from empty under the runtime transaction; recurrent state is never truncated.
Cancellation or provisional delivery failure at an unchanged scale restores the
previous committed state and sampler. At a changed scale, rollback retains the
new scale's empty state, matching the existing fresh-request contract. Failed
terminal delivery closes admission until explicit reset. Changing reuse policy
clears the live sequence while preserving sampler progression; an explicit reset
clears it and reseeds. No retained snapshot cache is added, and snapshot budget
admission remains unchanged.

Focused synthetic full-attention and recurrent graph tests compare exact live
reuse, edited/switched histories and provisional failure/cancellation replay with
fresh logits/state and a cloned reference sampler. Scale tests cover signed zero,
change-back, invalid-setting atomicity, disabled mode and explicit reset. These
are model-free controls; they do not qualify Swift token re-encoding across turn
boundaries, complete-model arithmetic, production sampling, resource or latency
benefit. Reuse remains opt-in pending actual-model private-service qualification.

Focused synthetic graph checks cover same-scale/change-back checkpoint rejection,
complete state clearing, invalid-setting atomicity, prefix eviction, cancellation
and failed delivery at a new scale. These checks do not load the full Swift model.
Full-model numerical comparisons, actual private-service lifecycle/device
qualification and release integration remain independent requirements under the
[rank-one contract](rank-one-adapter.md) and [validation policy](build-and-validation.md).
