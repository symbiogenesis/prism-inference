# Prism conformance policy

This safe offline JSON tool owns the runtime catalog, baseline floor, deterministic
ordinary-source partitioning and bounded revision-bound artifact fan-in. It
needs no Runner binary, native runtime archive, installer or model.

Build with `cargo +nightly-2026-11-09 build -p prism-conformance-policy --locked --release`.
Invoke `prism-conformance-policy --version`, or pass `prism-baseline`/`prism-source`
followed by `--request ABSOLUTE_JSON_PATH`. Requests are bounded to 16 MiB. Source
result fan-in retains its separate 1 MiB receipt and 128 MiB report bounds.

`.github/workflows/runtime-conformance.yml` is explicitly dispatched on `main`, preserving all
four ordinary source partitions, 16 backend planning partitions, 32 execution
partitions and the complete baseline gate. Benchmarks and gradients are explicit
manual opt-ins. Routine pushes use the regression jobs in
`.github/workflows/foundation.yml`: its compact Rust corpus plus affected native
codec parity. Regression receipts
record `FullSourceGate=false` and cannot replace the complete conformance gate.
`.github/workflows/prism-focused.yml` diagnoses one named source
without claiming complete coverage. `scripts/Bonsai.PrismSource.psm1` always
selects this checkout; only the separately hash-pinned C++ reference is downloaded.

See [conformance](../../docs/prism-upstream-tests.md) for coverage and
[validation](../../docs/build-and-validation.md) for workflow ownership.
