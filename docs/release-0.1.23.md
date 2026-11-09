# Prism Inference 0.1.23 source delivery

This prepares an owner-requested crates.io source delivery after 0.1.22. It is
separate from a qualified application release or a new five-executable GitHub
bundle. Hosted conformance, complete model/state/adapter acceptance and measured
request benefit remain pending; source reductions are not speed measurements.

## Source changes

The library includes the local CPU efficiency candidates documented in
[inference efficiency](inference-efficiency.md): direct floating projections,
compact empty rollback checkpoints, bounded native-order sampling, exact masked
attention traversal, original projection gate views, grouped recurrent staging,
direct projected convolution rows, and linear bounded suffix matching with
verified correction reuse. Explicit tiled prefill and speculative paths retain
their separate numerical and lifecycle contracts.

This delivery also includes additional multi-token GDN staging, validated-logit
sampling and resident native-Q2 queue-tail work, with exact focused controls in
the owning efficiency inventory. Accelerator crates and the private server
remain unpublished; their local source changes do not become registry runtime
dependencies or qualify signed-device execution.

## Dependencies and delivery boundaries

The five registry helper crates are unchanged: `bonsai-ternary =0.1.2`,
`bonsai-prism =0.1.4`, `bonsai-native-hash =0.1.2`, `bonsai-neon-dot =0.1.1`, and
`bonsai-gguf =0.1.4`. Only `prism-inference` requires a new registry version.
The unpublished server's package version and exact runtime dependency advance
together to 0.1.23; Cargo.lock retains every other resolved version.

Runner continues to select its verified 0.1.22 binary bundle until the owner
supplies a new qualified immutable source/archive/executable identity set.
Publishing the library cannot supply the unpublished private server or replace
that binary integrity contract. No Git ref is pushed, no GitHub release is
created, and no CI is dispatched or monitored by this source delivery.

Default arithmetic, representation, adapter selection, worker policy, prompt
reuse, speculative selection, resource admission and containment are unchanged.
Registry publication does not promote an opt-in path or qualify throughput,
quality, sustained comfort, startup, or application acceptance.
