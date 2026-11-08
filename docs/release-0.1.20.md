# Prism Inference 0.1.20

This prepares the Rust library, source tools and Windows ARM64/x64 CPU bundles
for publication after exact-source release checks pass. It follows `v0.1.19`
(`951cc88486fdc5609a3aa57aad3b9e92216de2b1`); the earlier release and its retained
model/device observations keep their original identities.

## Changes since 0.1.19

CPU inference efficiency, kernel, speculative-decoding and tiled-prefill
additions include:

- Eighteen-worker CPU row-parallelism ceiling (`MAX_ROW_WORKERS = 18` matching
  18-core Snapdragon X2 Elite / X2 Elite Extreme topology), `TL1`-style 16-entry
  `INT16` activation-pair nibble lookup tables (`Tl1BlockLut` for contiguous and
  32-strided 128-element ternary blocks), compile-time `TQ1_BYTE_LUT` and
  `TQ2_BYTE_LUT` byte tables, 128-bit NEON `TQ2_0` block dot products
  (`neon::dot_tq2_block`), and direct `PTQ1 × Q8_K` dot products
  (`dot_ptq1_q8k`);
- Training-free prompt/history suffix speculative decoding (`SuffixDraftConfig`,
  `SuffixDrafter` / `SuffixDecoding`), Leviathan stochastic verification
  (`verify_stochastic_step`), and transactional short-chain block verification
  (`Runtime::generate_speculative`, `Runtime::verify_draft_block`, and
  `SwiftSession::complete_speculative`);
- Single-write fused GDN recurrent-state row step (`pinned_gdn::step_row_from`
  and `pinned_gdn::step_row_in_place`), keeping each decayed 128-element state
  row in NEON registers across `dot_k` and `update_and_dot` to eliminate one
  full state write-and-reread pass per token across all 48 GDN layers;
- 1/4/8-row NEON integer dot kernels for `Q8_0`, `Q4_0`, `Q4_1`, and 5-bit
  `Q5_0`/`Q5_1` (`dot_q8_0_block`, `dot_q8_0_eight_rows`, `dot_q4_0_block`,
  `dot_q4_0_eight_rows`, `dot_q4_1_block`, `dot_i16_block`), multi-worker
  `RowParallelism::fill` dispatch across all legacy, `K`-quant, `I`-quant, `TQ`,
  `FP4`, and `Q1_0`/`Q2_0` matvec rows, and single-pass row-parallel tiled
  prefill for all non-`PQ2`/`PTQ1` quantized projections
  (`supports_tiled_rows`, `dot_quantized_tile_row`, `dot_q8_0_tile_row`, and
  `dot_q4_0_tile_row`) with sibling prepared-`Q8`/`Q8_K` activation reuse.

Experimental native Hexagon NPU (`v81`) crates (`bonsai-hexagon-native` and
`bonsai-hexagon-runtime`, both `publish = false`) and Windows ARM64 worker hooks
add:

- Full six-arena `total_used` accounting (`arena0_used` through `arena5_used`),
  8-op multi-operation resident `dspqueue` batching (`submit_resident_multi8`),
  and intra-layer fused 1-token 3-op (`submit_resident_ffn_swiglu`) and 2-token
  6-op (`submit_resident_ffn_swiglu_multi2`) `ffn_gate` + `ffn_up` +
  `HTP_OP_GLU_SWIGLU` VTCM tensor chaining (`project_f32_ffn_swiglu` and
  `project_f32_ffn_swiglu_tokens` wired into `Evaluation::project_ffn_swiglu_into`
  and `PrefillWorkspace::project_tile_ffn_swiglu` when no rank-one LoRA adapter
  is active).

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
releases: `bonsai-native-hash` remains `0.1.0`, `bonsai-ternary` is `0.1.1`,
`bonsai-gguf` is `0.1.2`, `bonsai-prism` is `0.1.3`, and the server pins
published `bonsai-windows-credentials` `0.1.2` while retaining its unpublished
status due to the local Hexagon path dependencies. Registry publication of
`prism-inference` remains separate from bundle attachment. A Runner repin must
independently verify the source, archive and file identities; passing hosted
checks does not reopen a closed laptop allowance or qualify this release on an
actual model.
