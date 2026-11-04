# Terminal real-model projection probe

`prism-hexagon-model-probe` is an unpublished qualification worker. It runs one
or two tokens of the exact original Swift PQ2 artifact, offloading only
`blk.0.ffn_up.weight` through the existing guarded HVX or explicit HMX bitplane
signed-package boundary.
The graph supplies its actual transformed, quantized activation. No external
activation file substitutes for live graph execution. The base model runs without
an adapter; adapter qualification and effective HMX placement remain separate.

The Runner owner must verify the source-bound executable, original model,
signed package/catalog and driver, create the child suspended inside its owned
kill-on-close Job, retain the owner PID and creation FILETIME, and enforce one
absolute outer deadline including setup, CPU control, native work and cleanup.
The worker retains that owner identity before loading the model. It changes no
OS signing, certificate, boot or power policy. The reference native initializer
retains its documented DSP-process-class request; the package must still pass
the independent signed deployment gate.

Invoke `prism-hexagon-model-probe --request <absolute-json-file>`. Its strict
camelCase request contains `schemaVersion:1`, `modelPath`, `packageDirectory`,
`tokenId`, `contextLimit` (1..16), `modelMemoryLimitBytes` (at most 12 GiB),
`stateBudgetBytes` (at most 4 GiB), `sessionSeconds` (4..120),
`maximumOperations` (40..24,000), `maximumOwnedPayloadBytes` (at most 64 MiB),
`expectedParentPid` and `expectedParentCreationTime`. Unknown fields reject;
model bytes/hash come from the compiled original-Swift contract. Storage must
be on an expanded local drive outside Git and without reparse ancestors.

Optional `executionBackend` is `guarded-hvx` (the default) or `hmx-bitplane`.
Optional `tokenIds` contains one or two valid token IDs and must start with the
required legacy `tokenId`. Without it the worker evaluates only `tokenId`.
`contextLimit` must fit the entire sequence and `maximumOperations` must reserve
forty submissions per token before execution. For example, `tokenId:42`,
`tokenIds:[42,43]`, `executionBackend:"hmx-bitplane"`, `contextLimit:2` and
`maximumOperations:80` select one two-token HMX control in one terminal session.
No backend substitution or repeated native session occurs.

The fixed first projection needs forty native submissions per token. A scalar
CPU graph runs the entire sequence first, sharing immutable tensor storage with
the device graph. It is dropped before native setup; every token's logits and
logical-state bits remain for direct comparison. Retained state payload across
all CPU snapshots is capped in aggregate by `stateBudgetBytes`, separately from
the graph's own state payload bound. Model storage, transient conversion, graph scratch, retained state,
logits, native mapping and all allocator/metadata costs remain outer-owner
admission charges. The state budget is a per-graph payload bound, not a process
memory ceiling. The native 120-second cap starts at native setup and never
extends the owner's aggregate allowance.

Each native result is compared with an independent scalar integer-dot oracle.
The guarded HVX route checks both integer limbs; the HMX route independently
decodes serialized Q2 tile symbols, unit scales and all eight binary activation
planes, checking every returned plane value. Those eight planes represent the
bits of one token's original Q8_K activation, rather than eight prefill tokens.
The complete accelerated graph's logits and every logical state component must
then match the corresponding CPU control bitwise after each token. Complete-sequence
operation admission runs before the first evaluation. Native or oracle failure
closes the executor without replay. The final `event:"result"` JSON line is emitted
only after all forty or eighty submissions,
the comparisons and the native explicit cleanup cycle have succeeded. Progress
lines carry no completed model output. Runner must also establish exit and an
empty owned Job before accepting or publishing the result.

The result binds request/model identities, selected placement, operation counts,
integer comparison count, per-token `tokenComparisons` and per-arm endpoint power
snapshots. `tokenId` retains the first requested token and the top-level
logits/state digests describe the final token. Each `tokenComparisons` entry
binds its zero-based `index`, `tokenId`, both bitwise equality flags and both
digests. `nativeCleanup` retains individual cleanup return codes and
`nativeResources` retains mapped allocation and available domain metadata.
The HMX control marker describes the requested signed path; effective HMX
placement and granted VTCM remain unknown. Unknown power remains explicit; equal endpoints establish no
uninterrupted stability. This is a correctness capture with
`profileEligible:false` and `performanceMeasured:false`, including when it
passes. No default backend or service admission changes.

`--inspect` reports the system driver path without loading it. `--self-test`
checks model-free cap boundaries without loading a model or device. Full-model
placement, persistent serving, efficient fused kernels, signed HMX device qualification,
adapters, actual resources and same-artifact benefit remain independently pending.
