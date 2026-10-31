# Native Rust Hexagon capability boundary

This std-only crate queries the installed Qualcomm Windows FastRPC user driver
directly. It has no ggml, GenieX, QNN, C/C++ inference or vendor-SDK build
dependency. It is an isolated experimental workspace crate, with `publish=false`;
the main runtime does not depend on or re-export it. A reviewed registry release
of this boundary must precede any main-library dependency so existing runtime
publication is preserved. The runtime continues to report Hexagon unavailable.

The safe API restricts native calls to the fixed CDSP capability control request.
SCM discovery selects `qcnspmcdm` and its `libcdsprpc.dll` inside the Windows
DriverStore; callers cannot supply a DLL, RPC request or unsigned-process flag.
Path traversal, device/UNC paths and reparse ancestors reject. A retained
deny-write/delete file handle covers cache-only, no-UI Authenticode validation,
restricted loading, the synchronous query calls and unload. The driver object
is deliberately neither Send nor Sync and exports no raw symbol pointer.

The supervisor must separately bind the resolved DLL SHA-256 and publisher to
reviewed driver evidence and enforce an aggregate process deadline. Cached
Authenticode acceptance does not pin Microsoft, refresh revocation evidence or
qualify installed kernels. Native synchronous calls have no in-process
cancellation guarantee. Capability values are observations, not throughput or
allocation guarantees. Raw architecture remains distinct from packed-decimal
decoding and unknown/failure results must stay unknown.

This crate opens no DSP module, creates no queue/session, maps no buffers and
submits no kernels. Direct native Rust inference still needs safe asynchronous
buffer/queue ownership and separately signed PQ2/PTQ1 device kernels, followed
by numerical, complete-state, lifecycle, resource and measured benefit gates.
It preserves the main runtime crate's `forbid(unsafe_code)` boundary: only this
crate's isolated Windows module permits documented unsafe native operations.

`sources.json` pins the exact public Qualcomm ABI header and corresponding
signed-reference ggml host-driver convention. No implementation is vendored.
The narrow candidate is developed alongside Runner's standalone capability
fixture; the production engine remains Prism and no Runner runtime pin changes.

Hosted CI owns the complete tests and package checks. A focused named local ABI
check is allowed under the shared owner-device aggregate budget, including
setup, compilation, loading/querying when requested and cleanup. This library's
model-free `self_check()` opens no services, files or libraries.
