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

All seven native library source files match Runner revision
[`7373c020c928a6d188db628bef465f94b612ab9f`](https://github.com/symbiogenesis/SwiftBonsai2Runner/tree/7373c020c928a6d188db628bef465f94b612ab9f/tools/hexagon/native_rpc).
Fixed module names use typed `CStr` literals. The vendor request length still
excludes the terminating NUL, while its independently writable backing buffer
includes that NUL and is checked after return. URI, ABI, request ordering and
terminal ownership contracts are unchanged. Prior device observations remain
bound to their original source/build receipts; this synchronization supplies no
new device or inference qualification.

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
records entered operations only. No compute or unsigned-process control exists.
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

The branch-only hosted `hexagon-native.yml` workflow runs formatting, all native
contract tests, strict release Clippy, doctests and documentation compilation on
Windows ARM64 and x86_64 without a device. Unsupported-target stubs retain the
same runtime API and reject explicitly; only their constant-stub Clippy suggestion
is conditionally suppressed. ARM64 native lint checks remain unchanged. The
private mapping-receipt constructor is compiled only for native ARM64 or tests,
so ordinary unsupported-target libraries and documentation omit unused code.
Actual device execution belongs to Runner's separate source-bound supervisor;
this unpublished crate does not change production runtime dependencies or pins.

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
