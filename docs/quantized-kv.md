# Rotated KV cache

Swift generation, sessions and workers default to K4/V4 (`rsq4`); generic graph
constructors retain FP16. The immutable graph policy is
`KvCachePolicy::{F16, Rotated4, Rotated4FittedValues, Rotated4Keys3Values}`.
Client requests cannot change precision, and a codec failure never switches to
FP16. Actual-model quality, long-context benefit and separate X1/X2 device
qualification remain open.

## Algorithm and CPU implementation

For Swift's 256-channel full-attention heads, the codec applies three deterministic
signed normalized Walsh-Hadamard passes, then symmetric Gaussian Lloyd-Max
levels. Four-bit records occupy 132 bytes/head and three-bit records 100 bytes,
including a little-endian F32 norm scale. Codes pack least-significant bits first.
Zero has a canonical representation. Version 1 binds seeds, tables, packing and
rounding; the sampler seed does not affect rotations.

This original implementation draws mathematical ideas from
[TurboQuant](https://arxiv.org/abs/2504.19874),
[Qdrant's CPU discussion](https://qdrant.tech/articles/turboquant-quantization/)
and [turbovec](https://github.com/RyanCodrai/turbovec). It imports no code/dependency
from them and implements no QJL. Gaussian levels approximate the rotated
distribution; unbiased attention and negligible quality loss are not established.

The safe codec uses an audited `bonsai-neon-dot` boundary for detected 128-bit
NEON, with an explicit scalar fallback. X1/X2 use the same numerical identity.
Queries rotate once/head; keys are scored directly from packed codes, values
accumulate in the rotated basis, then receive one inverse transform.
Current-token K/V use the same codec. No context-sized expanded cache is allocated.
Existing RMS, RoPE and FP16 preparation remain; pinned rotated softmax admits up
to 262,144 tokens, while ordinary FP16 softmax retains its 8K bound.

Independent query heads can share the [row-worker pool](parallel-rows.md).
Tasks borrow packed records, own disjoint score/output slices and join before
returning the earliest head-chunk error. No additional pool is created;
cancellation remains at synchronous token boundaries.

## Selection, identity and recovery

| Selector | Policy | Use |
| --- | --- | --- |
| `f16` | `F16` | Reference/comparison |
| `rsq4` | `Rotated4` | Swift development default |
| `rsq4vf` | `Rotated4FittedValues` | Opt-in fitted value scales |
| `rsq4k3v` | `Rotated4Keys3Values` | Opt-in capacity experiment |

Generation's `--kv-cache` default becomes `rsq4` only after exact Swift PQ2/PTQ1
verification; other models retain FP16. Tokenize mode rejects the option.
Unsupported geometry rejects. Completion metadata reports `kv_cache` and
`graph_identity`. `Qwen35::with_state_budget_and_kv` and
`with_rank_one_adapter_and_kv` expose explicit library selection.

`SwiftSession::load_profile_with_kv_cache` and trusted worker/server
`--kv-cache` options select a profile before loading.
Non-FP16 readiness includes `kv_cache` and `startup_graph_identity`.
`SwiftClient::connect` verifies the default profile;
`connect_with_kv_cache` verifies an explicit alternative.
Changing precision requires a fresh session.

Immutable 64-token pages retain packed records; checkpoints copy private tails.
Every encode and storage reservation completes before graph state is committed.
Restoration preserves exact packed bytes. Profile identities prevent cross-policy
reuse; F32 recurrent/convolution state remains independent.
[CPU tiles](cpu-tiled-prefill.md) retain causal per-token codec attention.
Triangular GDN is a separate FP16 profile.

### Guarded fitted-value scale candidate

`rsq4vf` keeps version-1 four-bit keys and fits only value scales
(`Quantizer::with_fitted_scale`, codec version 2).
After ordinary rotation and code selection it proposes
`s = s_norm * dot(y,c) / dot(c,c)` using ordered F64 FMA sums and F32 scale
rounding. Nonpositive, nonfinite or reconstruction-overflowing proposals retain
the original scale; codes never change.

Acceptance requires strictly lower squared error against
`f64(s_norm) * f64(y[i])`, using actual F32 reconstruction products and ordered
F64 error sums. Ties retain baseline. This guard precedes inverse transformation
and does not guarantee better original-space error, logits or answers.
Encoding adds two 256-coordinate scans and a division per nonzero value head.

Records remain 132 bytes/head. Identity `kv-rsq4vf-d256-rht3-v1` binds version-1
keys and version-2 value scales. Its 4 GiB state ceiling admits up to 241,664
tokens, without qualifying that context. Compare encoding cost and held-out
quality before default selection.

## Memory observations

`Qwen35::state_memory` and `SwiftSession::state_memory` report committed KV
payload/scales, calculated full-context payload, F32 recurrent/convolution
payload and live capacities. CLI completion emits `state_memory` separately
from `state_budget_bytes`.

Live storage includes tables, metadata and retained capacity, charging shared
pages in full. It excludes weights, workspace, external checkpoints, allocator
overhead and device/process allocations. Zero committed payload after reset does
not mean buffers were freed. [Runtime accounting](runtime-contracts.md#memory-and-ownership)
defines process admission.

For 16 attention layers and four KV heads, calculated maximum cache payload is:

| Context | FP16 | K4/V4 | K4/V3 |
| --- | ---: | ---: | ---: |
| 8,192 | 512 MiB | 132 MiB | 116 MiB |
| 32,768 | 2,048 MiB | 528 MiB | 464 MiB |
| 65,536 | 4,096 MiB | 1,056 MiB | 928 MiB |

## Qualification

Required controls include scalar/NEON equality, layout/overflow, independent
decoded attention/GQA, page boundaries, cancellation and failed delivery.
Model-free controls cannot establish retrieval, answer/adapter quality or device
speed. Qualify exact source/model/adapter/profile identities and X1/X2
independently under the [validation procedures](build-and-validation.md) and
[owner-device budget](../AGENTS.md#owner-device-execution-budgets).
