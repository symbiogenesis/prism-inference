# Native Rust Hexagon capability boundary

This std-only crate queries the installed Qualcomm Windows FastRPC user driver
directly. It has no ggml, GenieX, QNN, C/C++ inference or vendor-SDK build
dependency. It is an isolated experimental workspace crate, with `publish=false`;
the main runtime does not depend on or re-export it. A reviewed registry release
of this boundary must precede any main-library dependency so existing runtime
publication is preserved. The runtime continues to report Hexagon unavailable.

The safe API restricts ordinary native calls to the fixed CDSP capability and
default-session metadata control requests.
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
buffer/queue ownership and separately signed PQ2/PTQ1 device kernels, followed
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

`Driver::prepare_session_metadata` now prepares the default `cdsp` session's
effective domain and exact bundled v73/v81 module URI through pinned public
`remote_session_control` request IDs 14 and 15. Native ARM64 ABI sizes, alignment
and field offsets are checked. Each input pointer, length and byte array is
checked after return; output must be terminated and match the fixed module,
version, domain and session zero. Failed or unexpected results retain exact
return codes and completed stages without retry or a fabricated domain/URI.
The method issues no remote-handle open, mapping or unsigned-process request;
its report is descriptive and supplies no admission or mapping authority.

Actual signed module opening remains pending independent verification of the
installed ELF bytes, catalog membership, supported Windows deployment/search
route and retained file lifetime through open/teardown. A basename URI does not
bind an installed module to the caller's checked package. An absolute URI has
not been established as a supported Windows deployment route. The referenced
ggml initializer requests unsigned-process control before opening its interface;
this crate omits that request and does not infer a usable signed-session policy
from source or a bundled catalog. No catalog installation, certificate changes
or search-path overrides are supplied here.

The public FD=2 and delayed-FD=3 modes are recorded without executing them.
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
