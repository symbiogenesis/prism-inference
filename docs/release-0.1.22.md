# Prism Inference 0.1.22

This prepares the Rust library, source tools and Windows ARM64/x64 CPU bundles
for publication after exact-source release checks pass. It follows `v0.1.21`
(`90eb97481a8df532cc0375660a1533bb0c5fe4cb`); the earlier release and its retained
model/device observations keep their original identities.

## Changes since 0.1.21

CPU inference efficiency, kernel, prefill, attention, GDN, tokenizer and
server additions include:

- Expanded `bonsai-neon-dot` (`0.1.1`) multi-row and tiled ARMv8.2-A `dotprod`
  (`SDOT`) and `i8mm` (`SMMLA`) primitives across `PQ2_0`, `PTQ1_0`, `Q8_0` and
  `Q4_0`, including `4x4`, `8x4` and `8x8` GEMM prefill tiles, `4x1` and `8x1`
  multi-row decode kernels, pre-deinterleaved `PQ2_0` activation plane reuse and
  cached `Q8_K` half-sums;
- Fused residual add, `RMSNorm`, block Hadamard rotation (`Sign * FWHT`),
  `SwiGLU + Sign * FWHT` and output finiteness checks across single-token and
  parallel prefill tiles;
- Full-attention and causal scoring optimizations including fused per-head
  `RMSNorm + RoPE + FP16` rounding, `4x4` (`4-key x 4-query`) and `2-token x
  4-key x 4-query` shared-FP16 NEON scoring kernels, causal multi-key softmax
  tiling with fused FP16 rounding, widened multi-head value accumulation tiles
  and fused per-tile gate sigmoid;
- Gated Delta Net (`GDN`) decode, prefill and speculative replay optimizations
  including `4-row` and `8-row` register-resident NEON recurrence kernels, fused
  `4-row` and `8-row` WY chunk solvers, multi-token register-sliding causal
  `conv1d + SiLU`, `16-lane` L2 normalization, fused post-recurrence
  `RMSNorm + SwiGLU`, direct head-major grouped Hadamard rotation, paired
  `beta`/`alpha` prefill projections and batched `4-token` rank-one LoRA
  projections;
- Zero-copy and borrowed-slice paths across the tokenizer, grammar/schema/Jinja
  parsers, JSONL worker/client transports and private HTTP server, including
  direct `Qwen35PreTokenizer` construction without per-call `Metadata`
  `BTreeMap` allocation, borrowed UTF-8 piece decoding and higher-ranked lending
  streams, single-lookup PEG automaton failure transitions, borrowed `httparse`
  header validation (dropping the `http` crate from `prism-inference-server`),
  and Windows CNG `bonsai-native-hash` (`0.1.2`) across production crates;
- Advancement of the pinned Rust compiler to `nightly-2026-11-09` (`rustc
  1.102.0-nightly`, `5b7b37aef 2026-11-08`) with Cargo-level Polonius Alpha and
  next-generation trait-solver gates in `bonsai-rust-trial` and scoped
  `#[expect(..., reason = "...")]` lint attributes across the workspace.

## Supported release scope

Default CPU arithmetic, model/adapter pins, sampling, context/output bounds,
admission, memory reserves, required containment and rollback remain unchanged.
Original PQ2, one row worker, no adapter and fresh private-service request state
remain the defaults. PTQ1, approved adapters, parallel row workers,
latest-prefix reuse, tiled prefill and experimental Hexagon NPU probes still
require explicit selection and independent qualification. This release does not
qualify a new application default, numerical parity, adapter provenance,
performance, quality or comfort.

Worker names and the five packaged CPU executable names are unchanged.
Applications must explicitly verify the new release before selecting it.

## Installation and provenance

Publication requires successful Foundation, explicitly dispatched full Runtime
conformance (not the push Runtime regression tier), and source-package checks
on the exact release commit. Attach the unchanged ARM64 and x64 Foundation
archives and matching integrity receipts from that commit. Receipts bind source,
compiler, target, all five executable hashes, the four license/notice files and
the archive digest. Release notes must link the successful owning runs.

The [build guide](build-and-validation.md) documents versioned source installation
and bundles. Dependency notices and pins are updated to published registry
releases: `bonsai-native-hash` is `0.1.2`, `bonsai-neon-dot` is `0.1.1`,
`bonsai-ternary` is `0.1.2`, `bonsai-gguf` is `0.1.4`, `bonsai-prism` is `0.1.4`,
`bonsai-windows-process` is `0.1.2`, and the server pins published
`bonsai-windows-credentials` `0.1.4` while retaining its unpublished status due
to the local Hexagon path dependencies. Registry publication of
`prism-inference` remains separate from bundle attachment. A Runner repin must
independently verify the source, archive and file identities; passing hosted
checks does not reopen a closed laptop allowance or qualify this release on an
actual model.
