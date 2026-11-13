# GGUF tools

`bonsai-gguf` provides bounded little-endian GGUF v3 inspection, Prism PQ2 census
and experimental weight-lossless PTQ1 export using the shared Rust codecs.
Every operation verifies complete input size/SHA-256; Windows handles deny source
writes/deletion through verification and use. This is not a validator for every
GGUF extension.

`Metadata`, `MetadataValue` and `MetadataArray` preserve native scalar types and
compact homogeneous arrays. The default `report` feature includes the CLI, JSON
reports, census and export; `--no-default-features --lib` omits that tooling.
Runtime metadata loading uses the same parser but omits report-only hashes,
histograms and name lists. Header parsing uses one bounded 64 KiB buffer and
does not interpret/materialize tensor payloads.

## CLI

Build `bonsai-gguf` using the [workspace build instructions](../../docs/build-and-validation.md).
Use model/output paths outside Git and supply the expected artifact identity:

```text
bonsai-gguf inspect --model PATH --expected-bytes N --expected-sha256 HASH [--dialect Stock|Prism-73bfe6cc485554822f80c473a3c0c4f25e80f41b] [--census-pq2] [--output PATH]
bonsai-gguf export-ptq1 --model PATH --output PATH --expected-bytes N --expected-sha256 HASH [--timeout-seconds N]
```

Inspection skips payloads after hashing; census streams PQ2 in bounded batches.
Optional report output is create-new. Success writes one JSON object; failure
writes stderr and returns nonzero.

Export preserves raw scales, ternary symbols, other tensor payloads and metadata,
then verifies byte-identical PQ2 reconstruction. It writes a unique partial,
validates it and publishes by exclusive hard link, requiring a hard-link-capable
filesystem. Failures retain partials. A post-publication partial cleanup failure
adds `RetainedPartialPath` and `PartialCleanupError` to the report.

`--timeout-seconds` accepts 1..3600 (default 1800), covering conversion, copying,
padding and initial output hashing. Source verification and final validation are
additional work. This tool limit does not extend
[owner-device allowances](../../AGENTS.md#owner-device-execution-budgets).
Weight-lossless export does not qualify PTQ1 activation arithmetic, logits,
quality or speed; see [validation](../../docs/build-and-validation.md).
