# Prism Inference 0.1.15

This prepares the Rust library, source tools and Windows ARM64/x64 CPU bundles
for publication after exact-source release checks pass. It follows `v0.1.14`
(`9e8ea3e2c042ab0c3f1a8e9596ff155e202c3c57`); the earlier release and its retained
model/device observations keep their original identities.

## Changes since 0.1.14

CPU inference efficiency and kernel additions include compact reset-state
rollback checkpoints, bounded top-20 Swift sampler candidate storage with fused
finite-logit validation, direct one-column `F32` adapter projections, redundant
empty-reset elimination, exact masked-attention padding pruning, scoped packed
`PQ2_0` digit-plane activation preparation, bounded pair-accumulator `PQ2_0`
NEON decode reductions, baseline NEON GDN dot/update and adjacent-channel
attention value tiles, streamed `BF16` projection weights without decoded row
scratch, explicit 8/16/32-token CPU tiled prefill sharing the private
row-worker pool, and `RowParallelism::update_chunks` for fixed-chunk parallel
slice updates.

Experimental native Hexagon NPU (`v81`) crates (`bonsai-hexagon-native` and
`bonsai-hexagon-runtime`, both `publish = false`) add:

- 8-HVX (`HMX_Q2_HVX_WORKERS = 8`, `HMX_Q2_ACT_WORKERS = 4`), 2 MiB VTCM
  chunking (`n_chunk = 4352`), Burst power mode (`HTP_POWER_MODE_BURST = 7`),
  batch-4 and batch-10 HMX bitplane operation framing, 15 ms bounded queue
  spin-polling before 1 ms timer sleep fallback, and 4-worker L1-tiled host
  bitplane reconstruction;
- 1-op full-`K` (`K <= 17,408`, `M <= 17,408`, `N = 1`, `4` MiB VTCM for
  `K > 6,144`) native `HTP_TYPE_Q2_0` HVX projection execution
  (`NativeQ2Executor` and `ExecutionBackend::NativeQ2`) with parallel
  4-worker `PQ2_0 -> HTP_TYPE_Q2_0` tile packing, signed `v81` HTP opcode
  descriptors (`htp_ops.rs`), and bounded floating-point logit tolerance audit
  in `prism-hexagon-model-probe`;
- Scoped `QueueBuffer.size` (`dspqueue_buffer.size`) cache-flush tracking
  (`dirty_resident_weights`) so warm resident `Weights::NativeResident`
  submissions flush only the page-aligned active working prefix (`<= 92` KiB)
  while preserving full-allocation `bufs[0].size` (`image[8..16]`), plus a
  `480` MiB resident weight arena (`NATIVE_Q2_MAX_RESIDENT_WEIGHT_BYTES`) and
  `26`-projection probe admission covering all 25 `PQ2_0` projections of the
  complete 4-layer repeating block (`blk.0..blk.3`).

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
and bundles. Dependency notices and pins are unchanged: `bonsai-native-hash`
remains `0.1.0`, `bonsai-gguf` remains `0.1.1`, and the server retains its exact
Git credential-helper dependency and unpublished status. Registry publication
remains separate. A Runner repin must independently verify the source, archive
and file identities; passing hosted checks does not reopen a closed laptop
allowance or qualify this release on an actual model.
