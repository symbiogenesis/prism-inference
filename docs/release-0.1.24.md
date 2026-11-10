# Prism Inference 0.1.24 registry delivery

The [runtime crate](https://crates.io/crates/prism-inference/0.1.24) was published
on November 9, 2026, from local source
`f80f90e72c8a15b0a05e251fc49b354f6b7201ff`. The 3,708,626-byte archive contains
555 members and has SHA-256
`9262f8e737323cd66020cd5d72982e14ccdfd667b9fc0da0a0108014449fbd75`.
Both normal Cargo package and publication verification passed, including the
normalized source identity and exact registry dependency audit. The public
registry returned the same checksum. No performance improvement is claimed.

Runner retains the declaration, driver and raw receipts at
`docs/evidence/prism-0124-publish-20261109/`. The single five-minute operation
closed in 112.274 seconds, including a pre-entry launcher-copy failure and its
correction under the original clock. Power at that original start is unknown;
all observations after corrected entry reported AC connected, battery saver off
and 99% charge. Samples do not prove uninterrupted power stability. No model,
benchmark, complete suite, Git push or CI action ran.

The runtime includes paired and six-head prefill attention, terminal state-only
work, bounded request-local BPE reuse and special-token recognition, validated
logit sampling, and finite-activation controls. The local Hexagon sources also
include resident multi-tile prefill batching. The latter are outside the
published runtime package and need their own source-bound binary delivery.

The five published helper crates are unchanged from the 0.1.23 release source
and their checksum-verified registry archives matched the committed files:
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
