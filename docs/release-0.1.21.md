# Prism Inference 0.1.21

This prepares the Rust library, source tools and Windows ARM64/x64 CPU bundles
for publication after exact-source release checks pass. It follows `v0.1.20`
(`3a22fefc497da3d571c07bba072f270fc9c4b135`); the earlier release and its retained
model/device observations keep their original identities.

## Changes since 0.1.20

CPU inference efficiency, kernel, sampling, speculative-decoding and prefill
additions include:

- Isolated `bonsai-neon-dot` (`0.1.0`) safe boundary crate wrapping ARMv8.2-A
  `vdotq_s32` (`dotprod` / `SDOT`) and `vmmlaq_s32` (`i8mm` / `SMMLA` `2x2`
  row-by-token tiles) behind runtime feature checks (`has_dotprod`, `has_i8mm`),
  plus non-widening `u8x16` base-3 division-by-multiplication `PTQ1_0` trit
  unpacking (`decode_ptq1_trits_i8`, `dot_ptq1_q80_block`, `dot_ptq1_q8k`);
- Single-pass 16-register NEON fused block helpers for `RMSNorm + Sign * FWHT`
  (`rms_norm_signed_fwht_in_place`) and `SwiGLU + Sign * FWHT`
  (`swiglu_signed_fwht_into`), eliminating intermediate activation scratch
  round-trips before `PQ2_0`/`PTQ1_0` projection quantization;
- Last-layer state-only prefill narrowing (`Evaluation::full_attention_kv_only`,
  `Evaluation::recurrent_attention_state_only`, and
  `PrefillWorkspace::evaluate_Prompt_tile`), skipping dead output-projection and
  FFN work on the final layer (`layer 63`) for intermediate prompt tokens whose
  logits are not consumed;
- Streaming NEON top-k sampler pre-filter (`TopK::apply`, scanning `f32x4`
  lanes with `vmaxvq_f32` cutoff rejection to avoid full-vocabulary `248,320`
  candidate heap allocation) and a low-jitter spin-then-park `parallel_tile`
  completion barrier (`TileBarrier::wait`);
- GDN rollback-by-replay speculative verification tape (`GdnVerifyTape`,
  reducing per-layer speculative rollback storage from `4,194,304` state bytes
  to `12,288` activation bytes across `48` GDN layers), `DFlash2DraftConfig`
  (`5` block-diffusion layers, `block_size = 16`, `[16, 30, 44, 58]` target
  feature extraction layers), and checkpoint-free greedy step verification in
  `Runtime::verify_draft_block`;
- Stateful streaming `Hasher` (`new`, `update`, `finalize`) in
  `bonsai-native-hash` (`0.1.1`) backed by Windows CNG (`BCRYPT_SHA256_ALGORITHM`)
  on Windows and `sha2::Sha256` on non-Windows targets.

Experimental native Hexagon NPU (`v81`) and heterogeneous placement additions in
`bonsai-hexagon-native` and `bonsai-hexagon-runtime` (both `publish = false`)
include:

- Full intra-layer FFN VTCM chain (`submit_resident_ffn_full_chain`,
  `image_ffn_full_chain_resident`, and `project_f32_ffn_full_chain`, chaining
  `ffn_gate + ffn_up + HTP_OP_GLU_SWIGLU + HTP_OP_FWHT_I8 + ffn_down` in one
  FastRPC submission), `DMA64` stride guard (`supports_dma64_desc`), hybrid
  CPU/NPU prefill chunk scheduler (`plan_hybrid_prefill_tile`), and the Adreno
  `Q2_0+FWHT` 3-processor placement planner (`plan_three_processor_placement`,
  `HeterogeneousPlacementPlan`, `AdrenoQ2KernelKind::WgslSubgroupShuffle`, and
  `AdrenoQ2KernelKind::OpenClSubgroupBlockRead`).

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
releases: `bonsai-native-hash` is `0.1.1`, `bonsai-neon-dot` is `0.1.0`,
`bonsai-ternary` is `0.1.1`, `bonsai-gguf` is `0.1.3`, `bonsai-prism` is `0.1.3`,
and the server pins published `bonsai-windows-credentials` `0.1.3` while
retaining its unpublished status due to the local Hexagon path dependencies.
Registry publication of `prism-inference` remains separate from bundle
attachment. A Runner repin must independently verify the source, archive and
file identities; passing hosted checks does not reopen a closed laptop
allowance or qualify this release on an actual model.
