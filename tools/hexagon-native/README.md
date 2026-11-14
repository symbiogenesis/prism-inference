# Native Rust Hexagon boundary

This experimental, unpublished crate queries Qualcomm's Windows FastRPC driver
and exposes bounded projection sessions. The main CPU runtime does not depend on
it and reports Hexagon unavailable. Public ABI and signed-reference host pins
are in [sources.json](sources.json); no vendor implementation is vendored.

## Driver ownership

SCM discovery selects `qcnspmcdm` and its `libcdsprpc.dll` in the Windows
DriverStore. Callers cannot supply a DLL, arbitrary RPC request or process flag.
Device/UNC paths, traversal and reparse ancestors reject. A deny-write/delete
file guard spans cache-only, no-UI Authenticode validation, restricted loading,
native calls and unload. Only the Windows boundary permits documented unsafe
operations; the main runtime retains `forbid(unsafe_code)`.

`Driver` is neither `Send` nor `Sync` and exports no raw symbols. One process-wide
nonblocking mutex lease excludes other native owners, including across independent
objects. Contention or poisoning rejects. The lease covers unload/file close and
remains pinned with terminal sessions.

The trusted supervisor must independently verify the driver SHA-256/publisher,
exact package/catalog identities and child environment, and enforce an outer
process deadline. Native synchronous calls have no in-process cancellation
guarantee. Cache-only Authenticode checks do not refresh revocation or identify
the DSP-selected file. Unknown capability results remain unknown; reported
architecture, packed-decimal decoding and granted resources are distinct.

## Capability and mapping APIs

The ordinary API issues only fixed CDSP capability/default-session controls.
`prepare_session_metadata` checks the effective domain and bundled v73/v81 URI
using public request IDs 14 and 15. Returned pointers, lengths, bytes, termination,
version, domain and session zero must match. It opens no module or mapping and
confers no execution authority.

Mapping probes consume the owner. The scratch probe CPU-checks and attempts one
4,096-byte STATIC flag-0 mapping cycle. `session_mapping_probe` opens the fixed
module from an explicitly checked `ADSP_LIBRARY_PATH` directory, then uses FD=2.
Module/catalog/INF guards and the driver remain pinned through process exit after
any open attempt. Uncertain map/unmap retains both allocation and handle; no later
driver admission or retry is allowed. Delayed-FD=3 is descriptive only. A STATIC
refusal says nothing about established-session FD mapping, DMA or computation.

`reference_session_mapping_probe` explicitly adds the reference initializer's
unsigned DSP process-class request. It still requires the exact Windows
catalog-signed package; package signing and DSP process class are separate.
The fixed request is `DSPRPC_CONTROL_UNSIGNED_MODULE=2`, size 8, enable 1 and the
checked default-CDSP domain, after effective-domain lookup and before GET_URI.
See the pinned [public header](https://github.com/qualcomm/fastrpc/blob/a56e9d4de3614a8d8f4b0e15ff561350c4b686e3/inc/remote.h#L603-L622)
and [reference initializer](https://github.com/ggml-org/llama.cpp/blob/94256114c229674ef96e76eb2dea596e65b43818/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L3875-L3912).

Owners are terminally pinned before this request. Its exact result must pass
before module open. Writable returned fields are recorded as raw
`processInitializationDomainAfter`/`processInitializationEnableAfter` observations;
they never configure later URI/mapping or prove an effective process class.
`unsignedProcessRequested` records invocation only. Refusal has no weaker retry.
The diagnostic changes no OS signing, certificate trust, boot or power policy.

## Fixed queue controls

Queue controls consume one independently verified GenieX v0.10.0 v81 package.
The supervisor must first pass the independent Windows ARM64 C ABI/start-wire
controls and establish owned-process containment. Public headers alone do not
satisfy those prerequisites.

| API | Fixed workload |
| --- | --- |
| `reference_queue_compute_probe` | K=32, M=32, N=2 F32 matrix; all 64 output bits checked independently |
| `reference_queue_q2_probe` | K=256, M=32, N=2 guarded Q2; four 576-byte host tiles, 640-byte VTCM slots, 61,440-byte scratch |
| `reference_queue_noblock_probe` | F32 control with one `dspqueue_write_noblock` call |
| `reference_queue_q2_noblock_probe` | Guarded-Q2 control with one nonblocking write |

The F32 packet uses a seven-field 48-byte start frame, one 24-byte request, three
tensor descriptors, one HVX worker and a 16 KiB mapped allocation. HMX is disabled
and balanced power is explicit. Output poison and results are included in cache
extents. Replies must match sequence, status, counts, message length,
FD/address/offset/extent, cache flags and trace counts before output is read.
Reads share a three-second absolute deadline; an outer process deadline is still
required. Controls are selected individually, without fallback or repeat writes.

A failed write may have advanced the queue. At most one correlated reply is
observed afterward: valid numeric output may be recorded, but the write error
and failed complete-cycle verdict remain. Such paths retain all owners and skip
cleanup. Missing, invalid or expired replies supply no numerical result.

## Projection sessions

`reference_q2_projection` exposes guarded HVX K=256/N=2 operations.
`reference_hmx_q2_projection` exposes K=128/N=8 bitplanes through
`HmxQ2Submitter`. Both admit M multiples of 32 through 17,408, preserve response
correlation and require caller-owned exact integer-dot comparison before
publication. The HMX eight rows are one token's activation bits, not eight tokens.

| Property | Guarded HVX | HMX bitplanes |
| --- | --- | --- |
| Workers/resources requested | One HVX, 61,440-byte VTCM scratch | One HMX, one HVX, 65,536-byte VTCM budget |
| Returned integer range | Finite, absolute value at most 16,384 | Finite, -128 through 256 |
| Full-M solver | Scratch independent of M | `m_chunk=32`, `n_chunk=128`, 57,344 VTCM bytes |

Requests establish neither effective HMX placement nor granted VTCM. Each session
has one absolute 4..=120-second allowance, including preparation and a three-second
cleanup reserve, and at most 24,000 total write attempts across all executors.
Reset, failed writes and executor replacement never replenish it. These terminal
limits do not constitute a persistent-service contract.

Every request rebuilds mutable descriptors, advances its sequence and poisons
output. No host slice spans device work. Failed writes, stale/malformed replies,
invalid output, expired limits, callback errors or unwinding pin uncertain owners
through supervised exit, without retry or CPU replay. Output stays provisional
until caller comparisons and `completed_cycle()` both pass.

Successful cleanup orders DSP stop, host queue close, same-tuple unmap, authoritative
u64 module close and allocation free. Unmap failure skips close/free; close failure
skips free. Even after successful explicit cleanup, driver/package/lease owners
remain pinned through exit, and complete vendor-session teardown is unqualified.

## Resident weights and fused packets

`HmxQ2ProjectionLimits::with_resident_weights(bytes)` optionally reserves an
additional immutable host-mapped arena, capped at 64 MiB. Callers separately admit
its aligned `allocation_bytes()`. Append-only preparation validates/copies each
extent once and returns session-owned handles. There is no eviction; capacity
refusal preserves a healthy session and its operation allowance. Foreign handles
and native failures close admission. Handles expire at teardown.

`HmxQ2WeightRegistry` supplies metadata-only allocation for custom submitters.
Resident Q2 handles retain validated HVX solver parameters; callers charge their
metadata payload. Full-allocation FD/cache coverage includes the extra arena on
every submission. Host residency avoids repeated CPU packing/copying; it does
not establish DSP residency, reduced DMA/cache work or measured speed.

Native Q2 secondary arenas must be fully prepared before their first DSP import.
Their initial host writes are cleaned before import; the supported mapping API
supplies no later cache-clean fence for a persistently imported arena. Late
preparation therefore faults the session before invoking an initializer, changing
bytes or publishing a reservation. A dirty already-imported arena also rejects
before submission. Primary-arena appends retain their existing queue-buffer
flush: preparation after a warm operation restores the full populated-prefix
dependency on the next submission. Unused secondary arenas are neither cleaned
nor imported by an unrelated projection.

Borrowed and owned `NativeQ2Session` forward direct in-place weight preparation,
eight-operation packets, fused SwiGLU and complete FFN chains. The single-token
chain is gate/up, SwiGLU, optional plain Hadamard, down and optional residual add.
Its supported `(operations, tensors)` pairs are `(4,12)`, `(5,14)`, `(5,15)` and
`(6,17)`. Adapters and grouped rotations disable it; down Hadamard requires every
authored sign to be positive one.

`supports_ffn_chain_multi2` explicitly enables paired token chains in one
8/10/12-operation packet with 24/28/30/34 tensors and at most four buffers.
Otherwise custom submitters retain per-token calls. Both outputs must pass before
either is copied; any failure clears the whole token tile and closes admission.
The working allocation reserves at least `8192 + 2 * 8 * 4 * max(K, M)` bytes,
page aligned. Operation caps, deadlines and retained-buffer rules still apply.
Cancellation is observed at synchronous graph-operation boundaries.

These are ordered `N=1` operations. True multi-token matrix prefill is not
implemented. See [graph integration](../hexagon-runtime/README.md) for placement,
resource accounting and complete-request admission.

## Validation

[Foundation CI](../../.github/workflows/foundation.yml) owns complete contract
tests, strict Clippy and documentation checks on Windows ARM64/x64, plus a
Linux-hosted Windows ARM64 compile/lint lane. Unsupported targets expose explicit
refusal stubs. Model-free `self_check()` loads no service, file or library.

Follow the [owner-device policy](../../AGENTS.md#owner-device-execution-budgets)
for any local capture. ABI, wire, cleanup and mock arithmetic controls do not
qualify signed-device numerics, full-model state/adapters, actual resources,
vendor teardown or sustained request benefit.
