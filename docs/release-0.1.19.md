# Prism Inference 0.1.19

This prepares the Rust library, source tools and Windows ARM64/x64 CPU bundles
for publication after exact-source release checks pass. It follows `v0.1.18`
(`c4cfe40eedd3d7913a225cfa494ac596c948c9c9`); the earlier release and its retained
model/device observations keep their original identities.

## Changes since 0.1.18

CPU inference efficiency, kernel and tiled-prefill additions include:

- Batched non-FFN tiled CPU prefill projections across 8/16/32-token tiles with
  `TileReuse` separating Hadamard basis reuse from quantized `ProjectionTile`
  preparation reuse, and token-parallel prefill normalizations across
  `RowParallelism` workers;
- Four-row × four-token (`4x4`) and eight-row × eight-token (`8x8`)
  consumption-order `PQ2_0` and 4-token-per-group register-blocked `PTQ1`/`Q8_0`
  NEON prefill GEMM kernels wired through `RowParallelism::fill_tile_four_rows`,
  8/4-row multi-token `BF16` prefill GEMM with shared sibling `ssm_beta`/
  `ssm_alpha` preparation, and multi-worker parallel tile activation quantization;
- Four-head grouped-KV full-attention score and value-tile reuse (`dot_f16_four`,
  `dot_f32_four` and `value_tile_four`), reading and converting each cached
  `FP16` key and value row once per four query heads sharing a KV head, plus
  batched multi-token full-attention tiled prefill (`full_attention_tile`) with
  shared per-tile `rope_frequencies` and parallel KV head RMSNorm/RoPE/FP16
  preparation;
- Parallel chunkwise GDN prefill causal short-convolution, shared 16-key-head
  `(K·Q, K·K)` triangular dot-matrix precomputation with zero per-layer heap
  allocations, single-dispatch head-major multi-token GDN causal short-convolution
  + `SiLU` + Q/K `l2_norm` (`conv_head_major`), L1-resident row-major exact GDN
  recurrent tiled prefill (`exact_head_into` using `gdn_decay_and_dot`), and
  paired NEON GDN dot kernels (`pinned_gdn::dot_pair`).

Experimental native Hexagon NPU (`v81`) crates (`bonsai-hexagon-native` and
`bonsai-hexagon-runtime`, both `publish = false`) and Windows ARM64 worker hooks
add:

- Three-sibling full-attention tiled-prefill NPU batching
  (`project_f32_triple_tokens` and `project_tile_triple`), packing
  `[attn_q, attn_k, attn_v]` across 4-token windows into 4-op
  `submit_resident_multi4` `dspqueue` submissions with pointer-deduplicated
  `rpcmem` activations and `submit_resident_batch3`/`batch2` tail batches;
- 168-projection six-arena resident coverage (`DEFAULT_NATIVE_Q2_PROJECTIONS`,
  `2,888,663,040` bytes / `2,754.84` MiB across six alternating `<= 480` MiB
  `FastRPC` weight arenas covering all 27 initial layers `blk.0..=blk.26` with
  zero wasted bytes within CDSP `v-size = 3,000` MiB).

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
`bonsai-gguf` is `0.1.2`, `bonsai-prism` is `0.1.2`, and the server pins
published `bonsai-windows-credentials` `0.1.1` while retaining its unpublished
status due to the local Hexagon path dependencies. Registry publication of
`prism-inference` remains separate from bundle attachment. A Runner repin must
independently verify the source, archive and file identities; passing hosted
checks does not reopen a closed laptop allowance or qualify this release on an
actual model.
