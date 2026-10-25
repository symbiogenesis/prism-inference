# Prism conformance policy

This safe offline JSON tool owns the runtime catalog, baseline floor, deterministic
ordinary-source partitioning and bounded revision-bound artifact fan-in. It
preserves the policy and frozen fixtures moved from SwiftBonsai2Runner; no Runner
binary, released runtime source archive, installer or model is needed.

Build with `cargo +nightly-2026-10-22 build -p prism-conformance-policy --locked --release`.
Invoke `prism-conformance-policy --version`, or pass `prism-baseline`/`prism-source`
followed by `--request ABSOLUTE_JSON_PATH`. Requests are bounded to 16 MiB. Source
result fan-in retains its separate 1 MiB receipt and 128 MiB report bounds.

`.github/workflows/runtime-conformance.yml` runs only on `main`, preserving all
four ordinary source partitions, 16 backend planning partitions, 32 execution
partitions and the complete baseline gate. Benchmarks and gradients are explicit
manual opt-ins. `.github/workflows/prism-focused.yml` diagnoses one named source
without claiming complete coverage. `scripts/Bonsai.PrismSource.psm1` always
selects this checkout; only the separately hash-pinned C++ reference is downloaded.

The existing Foundation workflow continues to own workspace Release analyzers,
complete Rust tests, immutable codec fixtures, mutation checks, both Windows
architectures and the Linux ARM64 native storage overlay regression.