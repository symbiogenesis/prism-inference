# Inference execution

This reference describes CPU/host execution and its invariants. Source work and
storage counts do not establish machine traffic, latency, process-memory savings
or permission to lower admission limits. Selection/resource contracts live in
[runtime contracts](runtime-contracts.md), [row workers](parallel-rows.md),
[CPU tiles](cpu-tiled-prefill.md) and [session profiles](swift-session-profiles.md).

## Projection preparation

Scoped prepared matvecs share activation preparation only for identical borrowed
input, transform/permutation, width and quantizer contract. Complete-input finite
validation precedes mutation. Its borrowed proof permits Q8_K/Q8_0/pinned PTQ1
maximum scans to omit repeated finite predicates; scale/code/rounding checks
remain. Standalone encoders validate their own input.

BF16 projection reads immutable two-byte weights after geometry validation.
Activation rounding, separately rounded F32 products, F64 accumulation and final
F32 rounding retain their order. Every stored weight receives a finite check;
malformed/nonfinite/overflowing work clears output. One-column F32 adapter
projections use the same streaming principle without decoded-row materialization.

### Packed PQ2 consumption layout

Eligible baseline-NEON PQ2 projections prepare activation bytes in weight-digit
consumption order once per `PreparedMatvec`. Four two-bit fields of sixteen
packed weight bytes pair directly with four activation vectors. Two integer
accumulators preserve exact dots without DOTPROD/I8MM.

Selection requires original PQ2 and columns divisible by 256. Original Q8_K
blocks/scales/sums remain available to other formats/executors; PQ2's fourth
symbol is +2. Odd-group PQ2 and PTQ1 use Q8_0. Unsupported/scalar builds allocate
no derived layout.

Preparation resets logical layout length even before rejected dispatch; addresses
or retained capacity never authorize content reuse. Growth is fallible and errors
clear layout/output. Row workers borrow the single immutable allocation.

Each 128-weight block retains its scale order (pinned FMA; generic multiply/add)
and complete-dot absolute bound 32,768. Derived payload for `C` columns is
`C + 4 * C / 128` bytes: 5,280 bytes at width 5,120, additional to Q8_K scratch.
Capacity remains at its high water; weights are neither repacked nor expanded.

## Packed KV validation

Packed-row encoding checks each input coordinate during the existing ordered
binary64 norm accumulation, before normalization or record publication. Portable
value accumulation checks each binary32 result during its update traversal,
omitting a separate read of the complete output row. Neither changes arithmetic
order, record bytes, cache precision, scratch admission or native ISA selection.
The public single-row helper still stages its complete result; scan-owned sums
remain private and permanently reject publication after a failed append.

After weight, complete record and weighted-scale validation, a scan-owned value
sum skips code decoding and channel updates when the weighted scale rounds to
either signed zero, including product underflow. Its initial positive-zero sum
and separate finite binary32 additions cannot create a negative-zero accumulator.
Public accumulation retains its ordinary arithmetic for caller-owned negative
zero. Corrupt zero-probability rows still reject and poison the private scan.

These source reductions require exact record/arithmetic and failure controls,
complete-model state/adapters/cancellation acceptance and matched whole-request
measurement before claiming X1 or X2 benefit.

## Finite logits and sampling

Complete decoder-output validation creates an immutable `FiniteLogits` view.
`Decoder::evaluate_finite_into` lets native/projected Qwen35 pass that proof to
Runtime, which verifies storage pointer, full extent and vocabulary size.
Custom decoders use a validating default. Foreign/partial views reject before
sampling/publication.

Greedy and fixed Swift top-20 selection can omit duplicate finite predicates.
Public sampling validates the entire slice, including losing candidates, before
RNG mutation. Custom `Sampler` implementations retain `sample` through the
default `sample_finite` adapter. Ingestion invalidates logits; failed continuation
restores validated output and sampler state.

Swift's bounded heap retains token-order ties and native probability/RNG order.
NEON visits winner-mask lanes in increasing token-ID order; nonfinite blocks
restore scalar rejection behavior. The fixed chain retains validated top-p
configuration, skips identity min-p-zero/temperature-one stages and reuses an
80-byte stack array of 20 unnormalized softmax weights. Top-p sums/divides in
F32; distribution resums retained weights in F64 and consumes its ordinary draw.
Singleton distributions still draw once. Generic stages remain independent.

## Attention locality

Pinned Swift has 24 query heads and four KV heads. Decode partitions workers by
complete six-head groups when the worker count does not exceed KV-head count;
larger policies retain finer partitions. Each group shares cached key conversion
and value traversal. Other geometries/source arithmetic use their own dispatch.

Six-head values use 64-channel tiles and 24 KiB of temporal accumulators.
Paired prompt tokens use twelve 32-channel rows with the same payload. They share
the common FP16 prefix, while token one uses its F32 current value and token two
sees that position through committed FP16 plus its own F32 current value.
A single-token tail uses the six-head route. No future row is reachable.

Every channel retains sixteen temporal FMA lanes, padding signed-zero effects,
the native reduction tree, globally normalized F16 probabilities and sigmoid
gate order. Source arithmetic retains F32 products/F64 accumulation. Masked
attention runs the first zero FMA per affected lane (at most sixteen padding
steps); repeated zero FMAs are idempotent for finite values. Softmax retains the
partially masked score group and omits later all-zero groups.

Grouped reduction validates all heads before gated publication. Paired tiles
validate both rows before either output is published; malformed geometry rejects
before mutation. Eligible four-head NEON value tiles fuse the gate into their
final store, retaining their ordered partial-error publication. Scalar atomic
fallback retains its own failure boundary. These destinations are private graph
scratch; complete-request rollback governs committed state.

## Recurrent RMS weight admission

Each recurrent layer borrows its immutable RMS weights through one finite-weight
scope before traversing its value heads. Ordinary decode shares this borrow
across all heads; tiled prefill shares it across all heads and tokens in the
tile, including existing worker partitions. The square-accumulation pass omits
repeated weight loads and finite predicates. At Swift's 48 value heads and
128-channel normalization, ordinary decode replaces 6,144 such checks with 128
per recurrent layer. This source count establishes no device timing benefit.

The scope owns no allocation and keeps no pointer-keyed cache. Its immutable
borrow prevents weight mutation until all consumers join. Geometry, epsilon,
activation and result checks remain per operation; binary32 products, ordered
binary64 square sums, scale rounding, source/pinned multiply order and SwiGLU
remain unchanged. Public normalization retains defensive validation. Graph
loading already rejects nonfinite normalization vectors; private malformed
weight injection now rejects before recurrent matrix work. Failed disposable
deltas never alter committed state and valid retry replaces dirty scratch.

Focused controls compare both arithmetic profiles with independent normalization,
exercise scalar/NEON tails, signed zero/subnormals, invalid input/weights/epsilon,
failure publication and stale admission. A direct recurrent-layer control uses
defensive per-row/per-head kernels, one/four workers, malformed weights and late
recurrence failure followed by recovery. Native/fallback compilation, complete
model state/logit/adapter/cancellation parity, codegen/resources and matched
complete-request benefit remain independent qualification gates.

## Provisional GDN decode state

Pinned Swift CPU decode writes eight recurrent rows into unpublished
`LayerDelta::Recurrent.matrix`. Eligible baseline NEON uses that destination
for decay/update, avoiding a separate staging array and flush. Committed state
and checkpoints remain immutable sources. Unsupported/portable/tiled recurrence
uses defensive helpers.

Native refusal/nonfinite work replays the defensive helper from committed rows,
preserving error precedence. Failed deltas never become recurrence inputs;
retry overwrites dirty scratch. Arithmetic retains separate decay, four-register
FMA dots, `(value - dot_k) * beta`, fused update/query, with all eight key dots
before updates. Two four-row phases reuse sixteen accumulators.

Other one/four/eight-row helpers use staging capacity eight through width 128
and sixteen through width 256. Multi-token helpers reuse staging for decay/
update and combine finite masks before grouped publication. Floating pass/order
and scalar partial-failure replay remain intact. Direct convolution tiles read
immutable projection rows and write convolved SiLU values; unsupported shapes
retain complete raw staging. Conservative stack/scratch charges are unchanged.

Exact pinned tiled prefill admits each complete square 128/256-row head once,
including every immutable token's Q/K/V extent, and selects baseline NEON once.
Eight-row groups reuse the unchanged temporal kernel and private staging. A
failed group replays the existing defensive row-major path from committed source;
prior successful groups retain their ordinary publication. Source arithmetic,
unsupported geometry and feature-disabled targets retain their existing grouping.

## Suffix proposals

The opt-in suffix drafter uses reverse-Z matching with a maximum 64-token window
and 64-byte prefix table. It preserves overlap, longest match and most-recent
start on ties, bounding token equalities below
`2 * (history tokens + maximum match tokens)`.

Runtime copies at most eight proposals into a reused stack buffer; public
`SuffixDrafter::propose` returns independently owned fallible storage.
A rejected proposal retains its validated correction token without a second
greedy vocabulary scan. Cancellation remains checked around proposal/verification/
correction, and request rollback includes sampler state.
[Sampled suffix verification](cpu-tiled-prefill.md#explicit-sampled-suffix-consumer)
defines the separate target-sampling contract.

## Native-Q2 prefill packets

Resident native-Q2 parameters are solved at preparation and reused by immutable
handles. Packet construction serializes each distinct resident geometry once,
then relocates member offsets/indices. Validation, activation deduplication,
output poisoning and operation order remain mandatory.

`NativeQ2Submitter::maximum_resident_token_batch` advertises single-packet
support: default one for arbitrary submitters, eight for a healthy native session.
Only 1/2/3/4/8 are accepted; other values choose scalar submission without probing
and retry. Resident tiles batch ordered `N=1` operations across prompt tokens.
Groups of eight use `submit_resident_multi8`; tails use four/three/two/one.

For `T` weight tiles and `n` tokens, queue writes are
`T * (n / 8 + ceil((n % 8) / 4))` (integer division), while logical operations
remain `T * n`. Each token preserves column-tile addition order. Transient weights
pack once per token tile but retain single-operation queue submission.

Preparation releases packed scratch before allocating up to eight padded
output rows: at most `32 * P` bytes for padded extent `P`, within the existing
`36 * P + 4 * P` conservative reservation. Caller output/metadata remain charged.
Native descriptors validate dimensions, registry ownership, operation capacity,
sequence, ranges and finite output. Runtime checks padding and partial sums.
Counters advance only after packet validation/accumulation; later failure clears
all output and faults the executor without CPU replay.

These packets do not implement `N > 1` matrix solving.
`hvx_mv_params`, kernel `HTP_MM_KERNEL_HVX_QUANT_ROW_N1` (7), eight HVX workers,
VTCM layout and one-row descriptors are specific to `N=1`. Multirow solving
requires separate kernel/solver/stride/scratch/numerical and signed-device review.

### Native-Q2 finite-output evidence

Scalar and two/three-sibling resident single-token projections carry borrowed
evidence from the session's checked output copy. Exact ordered pointers/lengths
must match every output. Arbitrary submitters use a validating default; foreign,
reordered or partial evidence cannot waive checks. Nonfinite reported success
rejects as `native-rpc-contract-failed`, clears output and publishes no successful
counters. Submission errors preserve their own diagnostics.

Transient, multitoken, fused-FFN and partial-row paths retain independent
validation. Fake-arena/queue controls do not open a driver or execute a DSP graph.

## Exact hosted correctness gate

The owner-operated [Source checks workflow](../.github/workflows/source-checks.yml)
with `model_free: efficiency` selects hosted Windows ARM64/x64 and Linux ARM64.
Its exact-name inventory rejects missing, ignored, duplicate or failed controls.
Receipts bind revision, source/executable hashes and raw results. Strict Release
all-target Clippy and production linking remain separate workflow steps.

Independent controls cover decoded-row projections, full-vocabulary sampling,
attention padding/grouped values, checkpoint ownership, adapter provenance,
suffix search, GDN arithmetic/failure, bounded tokenization and mock native-Q2
layout/queue behavior. Synthetic controls do not establish full-model or
signed-device qualification.

## Qualification

Before selecting an execution profile, retain exact-source PQ2 and separately
scoped PTQ1/adapter logits and complete-state comparisons, zero/nonzero adapter
scales, seeded output, reset, rollback and original-PQ2 recovery. Measure matched
complete requests/resources on the intended target, separating prompt ingestion,
first-token latency and generation intervals. ARM64 code generation/spills and
effective accelerator placement require independent review.

Use [building and validation](build-and-validation.md) for acceptance procedures
and [AGENTS.md](../AGENTS.md) for CI ownership and owner-device limits.
