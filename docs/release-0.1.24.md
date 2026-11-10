# Prism Inference 0.1.24 registry delivery

This release prepares the runtime changes after 0.1.23 for crates.io. Publication
and its verified registry checksum will be recorded separately after Cargo's
normal package verification succeeds. No performance improvement is claimed.

The runtime includes paired and six-head prefill attention, terminal state-only
work, bounded request-local BPE reuse and special-token recognition, validated
logit sampling, and finite-activation controls. The local Hexagon sources also
include resident multi-tile prefill batching. The latter are outside the
published runtime package and need their own source-bound binary delivery.

The five published helper crates are unchanged from the 0.1.23 release source:
`bonsai-ternary =0.1.2`, `bonsai-prism =0.1.4`, `bonsai-native-hash =0.1.2`,
`bonsai-neon-dot =0.1.1` and `bonsai-gguf =0.1.4`. Releasing duplicate helper
versions would provide no source change. The Hexagon crates and private server
retain their existing unpublished status; the server's local package version and
exact runtime dependency advance together to 0.1.24.

Cargo publication does not supply Runner's five-executable Windows ARM64 bundle.
The owner must publish the prepared source before a hosted Foundation producer
can bind a new bundle to it. Keep the existing 0.1.23 GitHub artifacts and Runner
pin until a real 0.1.24 source archive, bundle and producer receipt are verified.
No Git push, tag, GitHub release or CI operation is authorized by this document.

Full Foundation, source-package consumer variants, numerical correctness,
server/Hexagon acceptance, actual-model quality and sustained performance remain
owner-operated CI or separately declared device work. Existing closed diagnostic
clocks stay closed. README timings must remain attached to the runtime actually
measured until a new binary repin and fresh bounded capture succeed.
