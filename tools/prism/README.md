# Rust LoRA storage sharing for pinned Prism

`bonsai-prism` implements bounded LoRA B-vector sharing and the pinned engine's
file-loading/tensor-view boundary. The policy in `src/pool.rs` forbids unsafe code;
`src/loader.rs` owns file reads, bounds checks, metadata extraction, views and
backend writes. There are no runtime dependencies. `engine-loader` enables the
pinned optional cbindgen build dependency to generate C declarations in Cargo's
external output directory. Safe-policy builds need no external dependencies or
header. The static library links into the Prism revision in
[source.lock.json](source.lock.json).

Only ordinary CPU-buffer, contiguous rank-one F32 B vectors are eligible.
Context, buffer type, dimensions, strides and every payload byte must match;
signed zero and NaN payload bits remain distinct. Each candidate is at most
64 KiB; the cache holds at most 64 entries and 1 MiB. Existing matches remain
usable at either cap. Failed cache allocations retain independent storage.
Views keep their own tensor names; only canonical owners are loaded.

The approved adapter's 129 identical 20,480-byte B vectors avoid **2.5 MiB** of
native payload allocation. The native regression checks that exact difference
and byte-identical ggml MUL_MAT output. Whole-model throughput/quality remain
unqualified.

## Native boundary

The external Prism/llama.cpp engine remains C/C++. The source applicator inserts
ownership, Rust calls and exception translation into its loader. Generated C
declarations preserve the interop contract; inference replacement is separate
[native Rust engine work](../../docs/native-rust-engine.md). The experimental
[CPU reference](../../src/prism-runtime/README.md) reuses the safe sharing policy
but does not replace this overlay or the publisher profile.

`src/loader.rs` is the audited FFI boundary. It reads the pinned tensor's initial
C layout fields and calls public ggml/gguf APIs. It retains one file handle and
a fallible reusable read buffer; Windows denies source writes/deletion during
loading. Other platforms require caller-enforced immutability. Views skip loading
because canonical owners load separately. Null inputs, missing descriptors,
changed file length, invalid offsets and size mismatches reject.

Foreign callers must provide valid pointers, the exact pinned layout, exclusive
handle access and live native objects disjoint from Rust state. `tests/native.rs`
has separate documented unsafe calls into pinned ggml. The rest of the crate
denies unsafe code and the policy forbids it.

## Build and checks

Use PowerShell 7.6, pinned nightly Rust and a Clang-compatible CMake environment
for the external engine. Download and verify the ZIP in `source.lock.json` into
external storage. From the checkout, capture the repository path and use an
external native working directory for dependency fetches:

```powershell
$repo = (Get-Location).Path
$scratch = Join-Path ([IO.Path]::GetTempPath()) ('SwiftBonsai-Prism-' + [guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($scratch)
$previous = [Environment]::CurrentDirectory
Push-Location $scratch
try {
    [Environment]::CurrentDirectory = $scratch
    cargo +nightly-2026-10-22 fetch --locked --manifest-path (Join-Path $repo 'tools/prism/Cargo.toml')
    if ($LASTEXITCODE) { throw 'Prism dependency fetch failed.' }
    $result = & (Join-Path $repo 'scripts/Build-PrismEngine.ps1') -SourceArchive $absoluteZip -BuildMode All
    ctest --test-dir (Join-Path $result.EvidenceDirectory 'build-probe') -C Release -R '^prism-lora-storage-alias$' --output-on-failure
    if ($LASTEXITCODE) { throw 'Focused native regression failed.' }
} finally {
    [Environment]::CurrentDirectory = $previous
    Pop-Location
}
```

Set `$absoluteZip` to the verified archive path before running. The builder checks
the whole ZIP, extracts a fresh external lane, verifies original and overlaid
source hashes, builds locked/offline Rust and ggml, and links completion/server
to Rust. Receipts record toolchain and library hash. It runs no tests or model
and installs/promotes no binaries. `-PrepareOnly` compiles Rust/generates headers
without a C++ compiler. `-BuildMode Probe` builds ggml plus the Rust native test.
Visual Studio generators select ClangCL on Windows.

For a focused policy check, use the external working-directory convention above:

```powershell
cargo +nightly-2026-10-22 test --locked --offline --release --manifest-path <repo>/tools/prism/Cargo.toml --target-dir <temp>/build --test policy pool_policy_preserves_existing_cpp_cases -- --exact
```

Native tests require `--features native-tests --test native` and absolute ggml
library directories in `BONSAI_PRISM_NATIVE_LIB_DIR`, separated by the OS path
separator. Static linking is default. Reviewed Windows import libraries also need
`BONSAI_PRISM_NATIVE_LINK_KIND=dylib` and matching DLLs on PATH. Exact cases are
`native_storage_and_math` and
`native_loader_rejects_invalid_reads_and_keeps_fallbacks`.

Full suites and benchmarks remain CI-only. Foundation checks formatting, Release
Clippy/documentation, policy/FFI behavior, source drift/repeat rejection, native
allocator/math correctness and engine Rust linkage on Linux ARM64. The upstream
MIT notice remains in `PRISM-LICENSE.txt`; full engine sources stay in the verified
external archive.
