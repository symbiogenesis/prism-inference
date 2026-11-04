# Terminal real-model projection probe

`prism-hexagon-model-probe` is an unpublished qualification worker. It runs one
token of the exact original Swift PQ2 artifact, offloading only
`blk.0.ffn_up.weight` through the existing guarded HVX signed-package boundary.
The graph supplies its actual transformed, quantized activation. No external
activation file substitutes for live graph execution. The base model runs without
an adapter; adapter qualification and HMX execution remain separate.

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

The fixed first projection needs forty native submissions. A scalar CPU graph
runs first, sharing immutable tensor storage with the device graph. It is dropped
before native setup; its logits and bounded logical-state bits remain for direct
comparison. Model storage, transient conversion, graph scratch, retained state,
logits, native mapping and all allocator/metadata costs remain outer-owner
admission charges. The state budget is a per-graph payload bound, not a process
memory ceiling. The native 120-second cap starts at native setup and never
extends the owner's aggregate allowance.

Each native result is compared with an independent scalar integer-dot oracle.
The complete accelerated graph's logits and every logical state component must
then match the CPU control bitwise. Complete-request operation admission runs
before evaluation. Native or oracle failure closes the executor without replay.
The final `event:"result"` JSON line is emitted only after all forty submissions,
the comparisons and the native explicit cleanup cycle have succeeded. Progress
lines carry no completed model output. Runner must also establish exit and an
empty owned Job before accepting or publishing the result.

The result binds request/model identities, selected placement, operation counts,
integer comparison count, logits/state digests and per-arm endpoint power
snapshots. Unknown power remains explicit; equal endpoints establish no
uninterrupted stability. This is a correctness capture with
`profileEligible:false` and `performanceMeasured:false`, including when it
passes. No default backend or service admission changes.

`--inspect` reports the system driver path without loading it. `--self-test`
checks model-free cap boundaries without loading a model or device. Full-model
placement, persistent serving, efficient fused kernels, signed HMX execution,
adapters, actual resources and same-artifact benefit remain independently pending.
