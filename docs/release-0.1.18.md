# Prism Inference 0.1.18

This prepares the Rust library, source tools and Windows ARM64/x64 CPU bundles
for publication after exact-source release checks pass. It follows `v0.1.17`
(`958fd4a8bc3feb2ae1f923d15d279eb191a4fd6f`); the earlier release and its retained
model/device observations keep their original identities.

## Changes since 0.1.17

CPU inference efficiency, kernel and server additions include:

- Fused 64-element and final radix-4 NEON Hadamard stages, fused sign/rounding
  passes, parallelized Hadamard/GDN head preparation and FFN SwiGLU/tiled-prefill
  inverse-embedding rotation across row workers, and fused recurrent causal
  convolution with SiLU;
- Single-widen `PQ2_0` NEON consumption kernels, 16-bit lane `PTQ1` NEON
  unpacking without stack spills, vectorized multi-row scale decode, multi-row
  NEON `PTQ1` source-contract kernels with 4-group SIMD reduction, 8-row NEON
  `BF16` dot kernels for parallel row dispatch, and vectorized tiled-prefill
  `PTQ1`/`Q8_0` four-group lane accumulations.

Experimental native Hexagon NPU (`v81`) crates (`bonsai-hexagon-native` and
`bonsai-hexagon-runtime`, both `publish = false`) and Windows ARM64 worker hooks
add:

- SWAR `FP16` scale validation, 16-lane coalesced `F32` input/output finiteness
  verification, fused `rpcmem` output validation, and zero-churn registry lookups;
- Multi-op resident `HTP_TYPE_Q2_0` batch submission (`submit_resident_multi2..4`
  across `NativeQ2Submitter` and `NativeQ2Session`) with direct in-place
  descriptor construction and pointer-deduplicated multi-token/pair activation
  staging, amortizing FastRPC queue synchronization across grouped projections
  and tiled prefill batches (`project_f32_tokens` and `project_f32_pair_tokens`);
- 112-projection multi-arena resident coverage (`DEFAULT_NATIVE_Q2_PROJECTIONS`,
  `1,925,775,360` bytes / `1,836.56` MiB across four alternating `<= 480` MiB
  `FastRPC` weight arenas covering all 18 initial layers `blk.0..=blk.17` via
  `req.n_bufs > 1` and `tensor.bi > 0`), `O(1)` address-range + binary-search
  tensor eligibility and resident cache lookups, and reusable pre-warming scratch.

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
