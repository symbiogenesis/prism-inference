# Safe Rust assertions against pinned Prism

The `upstream_prism` example owns inputs, references and assertions in safe Rust.
The same tests target production Prism through a native process bridge and this
repository's Rust runtime. Both results are retained; native reference failure
fails the final gate. The source pin is
`PrismML-Eng/llama.cpp@b87ca7225456b06b0975b8239c66b5aecf07e37c`.

The [conformance guide](../../../docs/prism-upstream-tests.md) is the canonical
source inventory, assertion scope and acceptance policy. The Apple wired-memory
source is an explicit exclusion from Windows execution. Unsupported Rust APIs
remain failures rather than invented equivalents. Upstream MIT attribution is
in [LICENSE](LICENSE).

## Bridge boundary

The Rust PQ2/PTQ1 adapter in `bridge/` and separate native `common/` adapter contain
serialization, input validation and production calls. Generators, expected values,
assertions and tolerances belong to the safe Rust harness. Protocols are defined
in [PROTOCOL.md](PROTOCOL.md) and [common/PROTOCOL.md](common/PROTOCOL.md).

`reference-patches.lock.json` binds production saver and Llama 3 corrections by
original/corrected/patch hashes. Only verified external translation units are
substituted; original archives and test predicates stay intact. Saver correction
preserves DSpark metadata and sixteen tensors. Llama 3 correction disables inherited
punctuation cleanup for its three canonical aliases; see
[BPE acceptance](BPE-ACCEPTANCE.md). Receipts identify the corrected reference.

Input locks retain independent corpora, immutable publisher revisions, bounded
GGUF prefixes and HTTP range/size/hash checks. Expected schemas never generate
header inputs. Assets remain outside Git. The vector API runs once per original
activation column and compares every `M*N` result; this establishes no batched
execution or throughput equivalence.

## Build and run

Configure this directory against a verified source extraction with
`-DPRISM_SOURCE_DIR=<absolute upstream path>`, then build CMake target
`prism-upstream-bridge`. `RustBridge.cmake` uses the pinned workspace compiler and
production ggml libraries through a reviewed C ABI, without dependency downloads.
`PRISM_RUST_TOOLCHAIN` selects another explicitly installed compiler. Windows Rust
target and CMake architecture must agree. Build `common/` separately for its API.

```text
cargo run -p prism-inference --example upstream_prism -- \
  --target cpp --case test-pq2-row-shapes \
  --bridge <absolute native bridge executable> \
  --evidence-directory <fresh absolute evidence directory> \
  --output <absolute report path>
```

Use `--target rust` without bridge options for the runtime arm. PTQ1's selector is
`test-ptq1_0-element-map`; Unicode/MTP use `--common-bridge`. The supervising
`tests/Test-PrismUpstream.ps1` retains required requests/responses and checks all
native results in the final gate. GGML adapter use also requires
`tests/Test-PrismGgmlBridge.ps1` protocol/resource rejection checks; optional
`-ReferenceBridge` compares adapters independently.

Reports use `SchemaVersion: 1`, `UpstreamCommit`, `Target` and `Cases` with
`Id`, `Status`, `Detail`. Conformance failures are gate data; infrastructure
failures exit nonzero. Full catalogs, timed sources and all benchmarks are CI-only.
See [building and validation](../../../docs/build-and-validation.md) for execution
policy and the workflow inventory.
