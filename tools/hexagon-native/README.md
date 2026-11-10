# Native Rust Hexagon capability boundary

This std-only crate queries the installed Qualcomm Windows FastRPC user driver
directly. It has no ggml, GenieX, QNN, C/C++ inference or vendor-SDK build
dependency. It is an isolated experimental workspace crate, with `publish=false`;
the main runtime does not depend on or re-export it. A reviewed registry release
of this boundary must precede any main-library dependency so existing runtime
publication is preserved. The runtime continues to report Hexagon unavailable.

The safe API restricts ordinary native calls to fixed CDSP capability and
default-session metadata controls. A consuming terminal diagnostic also opens
the fixed bundled module and attempts one established-session FD mapping cycle.
SCM discovery selects `qcnspmcdm` and its `libcdsprpc.dll` inside the Windows
DriverStore; callers cannot supply a DLL, RPC request or unsigned-process flag.
Path traversal, device/UNC paths and reparse ancestors reject. A retained
deny-write/delete file handle covers cache-only, no-UI Authenticode validation,
restricted loading, the synchronous query calls and unload. The driver object
is deliberately neither Send nor Sync and exports no raw symbol pointer.
A process-wide nonblocking mutex lease restricts all threads and independent
objects to one native owner. Contention or lock poisoning rejects immediately;
the lease covers unload/file close and is pinned with any terminal mapping owner.

The supervisor must separately bind the resolved DLL SHA-256 and publisher to
reviewed driver evidence and enforce an aggregate process deadline. Cached
Authenticode acceptance does not pin Microsoft, refresh revocation evidence or
qualify installed kernels. Native synchronous calls have no in-process
cancellation guarantee. Capability values are observations, not throughput or
allocation guarantees. Raw architecture remains distinct from packed-decimal
decoding and unknown/failure results must stay unknown.

Trusted callers must also exclude vendor/runtime search-path, unsigned-process,
power and backend overrides in their child environment before loading the driver.
Runner's standalone query/mapping CLI rejects the fixed vendor/runtime prefixes
without logging values or mutating the parent environment.

The ordinary API opens no DSP module, creates no queue/session, maps no buffers
and submits no kernels. An explicit consuming terminal scratch probe can allocate,
CPU-check, map and unmap exactly 4096 bytes using fixed public vendor flags. The
mapping API may implicitly open a domain session. Every mapping attempt pins the
DLL/file owner until supervised process death; any mapping/unmapping failure pins
the uncertain buffer as well. No later driver admission or retry is allowed in
that process. Host allocation/registration proves neither DSP DMA nor compute;
implicit-session teardown remains unqualified. Direct native Rust inference still needs safe asynchronous
buffer/queue ownership and qualified packed computation through a reviewed signed
generic-kernel composition or supported custom-kernel distribution, followed
by numerical, complete-state, lifecycle, resource and measured benefit gates.

The terminal STATIC flag-0 control differs from the signed-reference ggml host's
FD flag-2 or delayed-FD flag-3 flows inside an established HTP session. A STATIC
refusal supplies no verdict on those modes, DMA64/VTCM or NPU execution. There
is no automatic alternate-flag retry. In the referenced public Qualcomm host
[`AEEStdErr.h`](https://github.com/qualcomm/fastrpc/blob/a56e9d4de3614a8d8f4b0e15ff561350c4b686e3/inc/AEEStdErr.h#L92),
512 matches generic `AEE_ERPC`; the native return remains authoritative and its
underlying cause remains unknown.
It preserves the main runtime crate's `forbid(unsafe_code)` boundary: only this
crate's isolated Windows module permits documented unsafe native operations.

`sources.json` pins the exact public Qualcomm ABI header and corresponding
signed-reference ggml host-driver convention. No implementation is vendored.
The narrow candidate is developed alongside Runner's standalone capability
fixture; the production engine remains Prism and no Runner runtime pin changes.

The native library source files and `sources.json` follow Runner's reviewed
[`bounded projection boundary`](https://github.com/symbiogenesis/SwiftBonsai2Runner/tree/bc45fcfc853916e1b78f1d788d7655db87e668fe/tools/hexagon/native_rpc).
Fixed module names use typed `CStr` literals. The vendor request length still
excludes the terminating NUL, while its independently writable backing buffer
includes that NUL and is checked after return. URI, ABI, request ordering and
terminal ownership contracts are unchanged. Prior device observations remain
bound to their original source/build receipts; this synchronization supplies no
new device or inference qualification.

## Guarded original-projection sessions

`Driver::reference_q2_projection` adds a separate experimental terminal API for
caller-prepared guarded Q2 integer operations. One owned module, bounded mapped
allocation and queue serve synchronous operations. The pinned K=256/N=2
quant-row solver uses one HVX thread and 61,440 bytes of VTCM scratch. M admits
multiples of 32 through 17,408. The bound Swift up projection can therefore
preserve its forty original per-128 reductions in forty submissions rather than
21,760 small-row requests; this is derived scheduling, not measured performance.

The exact pinned `htp-kparams.cpp` and `matmul-ops.h` layout keep scratch
independent of weight M for N=2, with no destination VTCM. Full-M host mapping
is bounded at 1,396,736 bytes and its start-frame mapping budget equals that
allocation. Every request regenerates its mutable command descriptors, updates
the monotonic sequence and poisons output. Reuse requires an exact correlated
response, complete FD/extent/address/cache checks and finite integral outputs
bounded by 16,384. No borrowed host slice spans device work.

`Q2ProjectionLimits` supplies one absolute 4..=120-second session allowance,
including setup, caller preparation, all operations and a three-second cleanup
reserve. The owner-authorized session cap is 24,000 total write attempts for
both HVX and HMX; attempts consumed by earlier executors remain charged, and
failed writes cannot replenish capacity. The cap does not extend the deadline
or establish complete-request admission. It also bounds M. Native calls still
require an external owned process deadline. Every failed write, stale or malformed
reply, invalid result, expired bound, callback error or unwinding pins uncertain
consumers through supervised exit with no retry or CPU replay. Successful
caller work must additionally verify exact integer dots against an independent
oracle; integer-shaped wrong outputs establish no numerical parity. Publish
only after the entire caller projection and `completed_cycle()` pass, including
explicit stop, host queue close, host unmap, u64 module close and allocation free.

Source-bound model-free controls cover dynamic packet extents, unchanged solver
fields, guard/scales, stale correlation, repeated successful operations, write
refusal, invalid numeric output, deadlines, operation caps and poisoned retention.
They do not qualify full-M device solver behavior, selected DSP bytes, complete
projection/model/state parity, vendor teardown or speed. The crate stays
unpublished, with no main-runtime dependency, backend availability or profile
promotion change. Existing device observations retain their original identities.

`Driver::prepare_session_metadata` now prepares the default `cdsp` session's
effective domain and exact bundled v73/v81 module URI through pinned public
`remote_session_control` request IDs 14 and 15. Native ARM64 ABI sizes, alignment
and field offsets are checked. Each input pointer, length and byte array is
checked after return; output must be terminated and match the fixed module,
version, domain and session zero. Failed or unexpected results retain exact
return codes and completed stages without retry or a fabricated domain/URI.
The method issues no remote-handle open, mapping or unsigned-process request;
its report is descriptive and supplies no admission or mapping authority.

`Driver::session_mapping_probe` now consumes the owner, validates the fixed
v73/v81 URI returned unchanged by the driver, and opens that module through one
explicit checked `ADSP_LIBRARY_PATH` directory. The trusted supervisor must first
verify the exact package hashes, catalog membership/chain and service-derived
driver identity. Native code also retains module/catalog/INF deny-write/delete
file guards. A basename URI and guarded directory do not independently prove the
actual DSP-selected file; signed deployment identity remains pending.

The probe uses public FD=2 only after a successful module open, and retains exact
open/map/unmap/close results. Any uncertain map/unmap retains the allocation and
handle without free/close. Every open attempt pins the driver, package guards and
process lease until supervised process death. Even successful explicit handle
close does not qualify complete vendor session teardown. Fixed callback progress
records entered operations only. That original probe submits no computation or
unsigned-process request; the separately named reference probes below do.
The authoritative public `remote.h` declares a u64 close handle; the reference
host wrapper's inconsistent 32-bit function-pointer typedef is not reused.

The terminal probe implements FD=2; delayed-FD=3 remains descriptive only.
Delayed registration does not prove DSP-side `HAP_mmap`, cache ownership or DMA.
The extended delayed mode's matching SDK enum/ABI is absent from the pinned
public header and remains pending. Asynchronous submission/completion ownership,
cancellation and in-use buffer retention require the verified signed kernel and
queue ABI. Metadata preparation neither replaces the retained scoped STATIC
failure nor establishes those execution/lifecycle gates.

Library regressions include exact URI/family boundaries, partial metadata and
mocked native refusals and mutated pointer responses. They never load a driver.

Hosted CI owns the complete tests and package checks. A focused named local ABI
check is allowed under the shared owner-device aggregate budget, including
setup, compilation, loading/querying when requested and cleanup. This library's
model-free `self_check()` opens no services, files or libraries.

The main and experiment-branch hosted `hexagon-native.yml` workflow runs formatting, all native
contract tests, strict release Clippy, doctests and documentation compilation on
Windows ARM64 and x86_64 without a device. Unsupported-target stubs retain the
same runtime API and reject explicitly; only their constant-stub Clippy suggestion
is conditionally suppressed. ARM64 native lint checks remain unchanged. The
private mapping-receipt constructor is compiled only for native ARM64 or tests,
so ordinary unsupported-target libraries and documentation omit unused code.
Actual device execution belongs to Runner's separate source-bound supervisor;
this unpublished crate does not change production runtime dependencies or pins.

An additional Ubuntu-hosted lane uses the same dated Rust release to check and
lint the actual Windows ARM64 target without linking or executing its binaries.
This reaches native-only FFI code when ARM64 runner capacity is queued, while
native ARM64 tests, documentation and device acceptance remain separate gates.

## Fixed X2 matrix queue control

`Driver::reference_queue_compute_probe` is a separate consuming diagnostic for
the exact reviewed newer GenieX v0.10.0 package, restricted to v81. It prepares
one F32 matrix operation with K=32, M=32 and N=2. The matrix contains diagonal
weights scaled by row index and two distinct signed integer activation rows;
all 64 outputs must exactly replace their NaN poison with independently computed
F32 bit patterns. This verifies neither packed Swift arithmetic nor speed.

The implementation encodes the newer seven-field 48-byte start frame, one
24-byte request, three tensor descriptors and one precomputed HVX matrix
descriptor directly in Rust. There is one queue slot, one HVX worker, HMX
disabled, explicit balanced power and no caller-selected native fields. A
single 16 KiB mapped allocation contains command, input and output. Its entire
extent participates in the queue cache flags so output poison and returned
results are included. The response must match the fixed sequence, status,
counts, 88-byte message, same FD/address/offset/extent, cache flags and zero
trace counts before the CPU reads output. The write has a finite timeout; reads
use the explicit nonblocking export under one three-second absolute deadline.
Native calls still require an independent external process deadline.

Before invocation, the trusted supervisor must independently pass source-bound
Windows ARM64 C queue-layout/prototype and generated-start-wire controls, verify
the exact package/catalog/driver identities, guard package files and establish
owned-process containment. Public queue declarations and a passed generic access
fixture alone do not satisfy those prerequisites. The implementation uses no
SDK, generated C build dependency or C/C++ inference library. Its independent
controls remain outside the Rust production runtime.

Every native-affecting attempt pins the driver, package guards and process lease.
No failed, cancelled, expired or malformed submission releases its allocation,
queue or module. A complete validated numeric response followed by successful
DSP stop, host queue close, same-tuple host unmap and authoritative u64 module
close permits normal allocation free. The module/session stays open through
host unmap, and the allocation stays retained through module close. A refused
unmap skips module close and free; a refused module close skips free. Hosted
mocks call the actual cleanup function and verify the exact release order and
each stop/queue/unmap/module failure boundary. The DLL and package owners remain pinned
through process exit even then. Explicit cleanup does not establish complete
vendor-session teardown. Selected DSP bytes, actual-model numerics/state,
resource behavior, sustained benefit and automatic selection remain unqualified;
`profileEligible` remains false and the main runtime does not consume this crate.

`Driver::reference_queue_q2_probe` is an explicitly separate fixed control for
the newer signed Q2 kernel. Its K=256/M=32/N=2 packet uses four tightly packed
576-byte host weight tiles and two guarded F32 integer-limb rows. DSP DMA stages
each host tile in a 640-byte VTCM slot; host payload contains no such padding.
The exact source-derived HVX solver requests 61,440 bytes of scratch. The 64
independent integer results cover all four symbols, positive/negative guards,
maximum corrected group sums and odd values beyond FP16's exact-integer range.
Every output must match exactly before the same conservative cleanup can run.
This controls the signed quantizer/integer primitive only: actual Swift weights,
activation preparation, original per-128 reduction, logits/state, adapters and
end-to-end benefit remain separate qualification gates. It runs no whole-model
conversion and changes no backend default. Callers select one fixed control;
there is no automatic fallback or retry between F32 and Q2.

`Driver::reference_queue_noblock_probe` is a separate F32 diagnostic using one
call to `dspqueue_write_noblock`. The packet, mapped FD/extent, cache flags,
numeric checks and conservative cleanup match the timed F32 control exactly.
Its purpose is to isolate the submission API after an observed timed-write
refusal; it asserts no cause or fix. The existing timed F32 and Q2 methods keep
their original seven-argument API and finite timeout. A would-block or any other
nonzero result is terminal and retains uncertain owners. There is no fallback,
repeat write or reuse of a queue after failure. Hosted independent C controls
check both six- and seven-argument prototypes; Rust controls verify that a
refused nonblocking call never invokes the timed API.

`Driver::reference_queue_q2_noblock_probe` explicitly selects the same one-call
nonblocking API with the fixed guarded-Q2 packet instead. It performs no F32
control automatically and supplies no fallback from a failed timed Q2 call.
Each control is separately selected, source-bound and charged to the same
owner-device allowance; missing remaining time leaves later controls pending.

On **2026-11-01**, the bounded X2 guarded-Q2 nonblocking control passed against
that exact Runner source using the same-run [ARM64 ABI and source-bound fixture
preparation](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/41318454484).
All 64 literal integer outputs matched bitwise for K=256/M=32/N=2. Write, read,
DSP stop, host queue close, host FD unmap and module close returned zero; the
allocation was freed, `poisoned` was false and the supervisor confirmed an empty
process Job. This establishes the fixed guarded integer primitive and its
explicit cleanup cycle. Actual Swift projections, complete logits/state,
adapters, memory/resource limits, selected DSP-byte identity, complete vendor
teardown, sustained performance and backend/default promotion remain pending.

The matching owning [Prism hosted
workflow](https://github.com/symbiogenesis/prism-inference/actions/runs/41318461417)
passed 19 Windows ARM64 and 12 Windows x64 tests, strict release Clippy,
documentation checks and the Linux-hosted Windows ARM64 cross-target check.

A nonzero write can follow an already advanced packet queue in the pinned public
reference implementation. After any write result, the diagnostic therefore
observes at most one strictly correlated reply under the existing three-second
read deadline. It never writes again or changes the selected submission API.
Exact numeric output may be recorded even when the original write returned an
error, while that error remains unchanged and the complete-cycle verdict stays
false. Every write-error path keeps all owners pinned and skips Stop, close,
unmap and free, including after valid numeric output. An empty, invalid or
expired read supplies no numeric evidence. Hosted failure-state controls verify
that numeric evidence and stale cleanup fields cannot erase the write error or
authorize cleanup/admission. Actual closed Windows behavior remains independent.

## Bounded HMX bitplane session

`Driver::reference_hmx_q2_projection` consumes the same independently verified
signed X2 package through one bounded queue and mapped allocation. Its separate
`HmxQ2Submitter` accepts fixed K=128/N=8 binary activation planes and unit-scale
576-byte Q2 host tiles, with M a multiple of 32 up to 17,408. The session requests
one HMX, one HVX worker and 65,536 VTCM bytes. Requests do not establish effective
HMX placement or a granted VTCM allocation; signed-device qualification remains
pending.

The pinned nonpipelined HMX2D source solver uses m_chunk=32 and chooses n_chunk
under that fixed budget, including its cost, tail-waste and size tie rules. At
full M it chooses n_chunk=128 and requires 57,344 VTCM bytes; at M=32 it needs
22,528 bytes. Internal 32-row activation padding stays in VTCM; the host receives
exactly eight contiguous M-row F32 plane results. All returned values must be
finite integers in -128..=256, with exact-dot comparison still caller-owned.

HMX and guarded HVX use the same queue ownership, monotonic response correlation,
poisoned output, no-retry and ordered stop/close/unmap/module-close/free protocol.
The shared report explicitly marks HMX through `is_hmx_bitplane_control()` and
excludes it from `is_guarded_q2_control()`. Every owner remains pinned after any
uncertain call, malformed response or deadline until supervised process exit.
The complete session allowance remains 4..=120 seconds with three seconds
reserved for cleanup, and an external Job deadline remains mandatory. No model,
PTQ1, performance or production eligibility follows from model-free controls.

The optional `HmxQ2ProjectionLimits::with_resident_weights(bytes)` reserves an
additional host-mapped immutable packed-weight arena, disabled by default and
bounded by 64 MiB. Its aligned allocation is reported by `allocation_bytes()`;
these bytes require separate resource admission. `prepare_weights` validates and
copies each packed extent once, issuing an opaque session-owned handle.
`submit_resident` uses that handle without copying its weights into the mutable
working tile again. Preparations are append-only, consume no kernel operations
and never evict an earlier extent. Capacity refusal leaves a healthy session and
its remaining operation allowance unchanged. Foreign-session handles, malformed
activation, queue failures and deadlines close the owner with the existing
poisoned-buffer lifetime rule. Handles do not retain native memory after teardown.

`HmxQ2WeightRegistry` exposes the metadata-only owner/range allocator for custom
model-free submitters. The native session keeps the existing full-allocation FD
descriptor and cache flags: adding the arena enlarges the extent covered by
**every** submission and may increase cache-maintenance work. Signed kernels
still stream their weights into VTCM. This infrastructure avoids repeated CPU
packing/copying only; it establishes no DSP residency, reduced DMA/cache traffic,
speedup, persistent serving or increased operation/deadline capacity. The report
retains mapped descriptor extent, prepared payload/count and resident write
attempts as work/resource observations, not measured traffic.

Native `Q2_0` resident handles also retain the immutable HVX solver result for
their validated `(K, M)` geometry. Single, batched and fused resident packets
reuse these parameters; transient packets still solve their active geometry.
The registry holds no additional per-handle allocation. Callers must charge the
larger handle payload; the runtime's existing `size_of`-based metadata reservation
does so within its unchanged cap. Wire bytes, signed kernels, placement and
session limits remain unchanged. Source work avoided is not measured latency.

The native borrowed session and owned session implement the complete single-token
resident FFN chain: gate/up matmuls, `SwiGLU`, optional plain block Hadamard, down
matmul and optional residual add. The four supported shapes have exactly
`(operations, tensors) = (4, 12), (5, 14), (5, 15), (6, 17)`; Hadamard has two
tensors, so a response assuming three tensors per operation cannot qualify it.
Each chain validates all three session-owned weight handles and geometry before
one nonblocking queue write, charges every operation and all three resident
projections, flushes dirty secondary arenas, and retains the existing primary
active-prefix cache and mapped-allocation bounds. Final output is copied only
after the complete correlated response and finite-result check. Any malformed
input, foreign handle, exhausted budget, deadline, write/reply or output failure
poisons the owner and clears the caller output without replay or early teardown.
Cancellation remains at the existing synchronous graph-operation boundary;
requesting cancellation does not release an in-flight allocation.

This is the existing explicit graph placement and supported transform subset.
Adapters disable the chain, grouped rotations reject it, and down-projection
Hadamard is eligible only when every authored sign is exactly positive one.
The packet cannot encode arbitrary signs or adapter corrections. No new default,
placement, resource discount, kernel, session allowance or release eligibility is
introduced. Model-free queue controls cover all optional shapes, warm cache
flushes, exact tensor counts and terminal failures; signed arithmetic, complete
model/state/adapter parity, cancellation/teardown and measured request benefit
remain separate qualification gates. True `N > 1` native-Q2 prefill remains open:
the current native solver is specialized to the `N = 1` HVX matvec contract.

Resident multi-operation packets also reuse already serialized tensor and HVX
operation geometry inside their own command buffer. Members with the same row
count and retained solver parameters share the first member's descriptors after
the existing common-column and complete arena/extent validation. Each copied
member relocates all weight/activation/output offsets, weight buffer index and
tensor indices. Only the 187 populated immutable bytes are copied; the already
zeroed command area supplies padding. Including relocation, each reused member
writes the same 225 bytes as the original sparse serializer, rather than the
438 bytes required by a complete 400-byte descriptor copy and relocation.
This reduces full descriptor construction from eight times to
once for an eight-token projection packet, and shares matching sibling geometry
in ordinary and fused SwiGLU packets. Mixed geometries retain the serial writer.
No template or allocation survives the packet; metadata, command bounds, flush
extents, submission order, numerics, failure handling and admission are unchanged.
Complete-frame model-free controls compare the copied path with independent
serial construction, including multiple arenas, changed solver parameters,
activation sharing, padding, output poison and invalid later members. Source
serialization counts establish no measured request or device benefit.

The public owned `NativeQ2Session` forwards the complete submitter contract to
its native owner, including direct in-place weight preparation, eight-operation
projection packets, single/two-token fused SwiGLU and complete FFN chains. Previously its
inherited trait defaults reported direct preparation unavailable, split an
8-operation packet into two 4-operation submissions, and refused fused SwiGLU
before reaching the existing backend. Normal owned-session prewarm can now use
the admitted mapped extent directly instead of allocating/copying a full packed
tensor through host scratch; existing capacity, scale validation and terminal
failure rules remain in the backend. Eight-operation prefill now reaches one
native submission instead of two. No operation is merged arithmetically: these
remain ordered `N = 1` projections, not true multi-token matrix multiplication.

Portable model-free controls instantiate the identical forwarding implementation
with a recording submitter. They require exactly one direct initializer and one
8-operation call, preserve fused call boundaries, original output bits and
slice-bound finite-output evidence, and propagate native refusals. Full FFN
calls reach the native session execution and exact packet-correlation path
described above. These source fixes do not qualify signed-device
execution, end-to-end performance, reduced admission or a release/default change.

## Explicit reference process initialization

The separate `--reference-session-mapping-probe ABSOLUTE_PACKAGE_DIRECTORY` and
`Driver::reference_session_mapping_probe` select one fixed additional native step.
The original `--session-mapping-probe` still requests no unsigned process and its
retained observation remains independent. There is no automatic retry between them.
The reference mode requests an **unsigned DSP process class**, even though only
the exact Windows catalog-signed package is eligible for this diagnostic. It must
not be described as signed-PD-only access or authorization for arbitrary DSP code.

The [pinned public header](https://github.com/qualcomm/fastrpc/blob/a56e9d4de3614a8d8f4b0e15ff561350c4b686e3/inc/remote.h#L603-L622)
defines `DSPRPC_CONTROL_UNSIGNED_MODULE=2`; its two-int structure at lines389-392
has `domain` and `enable`. The fixed control uses enable1 and the checked effective
default-CDSP domain. It occurs after the effective-domain request, before GET_URI
and module open, matching the [bundled reference initializer](https://github.com/ggml-org/llama.cpp/blob/94256114c229674ef96e76eb2dea596e65b43818/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L3875-L3912).
That source unconditionally selects the unsigned PD, including Windows builds.
This difference does not establish why an earlier module open was refused.

The public [Linux implementation](https://github.com/qualcomm/fastrpc/blob/a56e9d4de3614a8d8f4b0e15ff561350c4b686e3/src/fastrpc_apps_user.c#L2434-L2454)
changes a process-local per-domain attribute before opening a session; zero selects
signed and nonzero unsigned. Once open, changing its class rejects. The array is
zero-initialized at line4116. Lines2596-2616 dispatch this request; lines3238-3244
use its flag when constructing process attributes and lines3688/3735 use those
attributes for shell/session initialization. This public Linux behavior does not
prove the exact closed Windows driver implementation or complete remote teardown.
The source manifest records its exact digest and that limitation.

Windows package catalog signing is a separate requirement. The
[pinned GenieX run guide](https://github.com/qualcomm/GenieX/blob/a5c8e7fb4a364ddcccefbf187afd1bac52d88de5/notes/run.md#L303-L342)
requires no extra setup for its Microsoft-signed SDK, while its self-signed fallback
requires certificate trust and test-signing changes. The reference diagnostic makes
none of those OS, certificate or boot-policy changes and continues to require
independent exact package/catalog checks. It still explicitly records
`unsignedProcessRequested=true` only after its fixed runtime request is invoked.

Driver, package guards and process lease become terminally pinned **before** that
initialization attempt. Its exact return must pass before GET_URI/module open.
The ABI accepts a writable payload and the reference caller does not inspect its
post-call fields. The receipt retains raw `processInitializationDomainAfter` and
`processInitializationEnableAfter` values after every returned call, including a
refusal. They remain null when no call was attempted. These bytes never configure
URI preparation or mapping and never prove an effective DSP process class. All
subsequent operations use the original independently validated metadata/domain.
The earlier source's unchanged-input rejection remains a separate observation;
it was not a vendor refusal or evidence that the returned bytes were configuration.
Refusal or later incomplete URI returns a separate stage with no weaker retry. The flushed
`hexagon-session:stage=reference-process-init-entered` marker establishes entry only.
The receipt records explicit mode selection, actual request/result and fixed request2,
size8/enable1 independently. Tests mutate both returned fields, retain their raw
values and prove that subsequent URI/mapping still uses the original domain3.
Tests also cover ordering, refusal and retained
ownership; hosted builds and independent device observations remain required.
