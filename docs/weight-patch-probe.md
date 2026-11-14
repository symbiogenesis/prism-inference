# Experimental FFN capture and patches

`graph::weight_patch::FfnProbe` is an explicit research wrapper around an empty
`Qwen35`. It accepts one selected FFN-down site and an optional independent,
immutable rank-one patch. The base and approved adapter files remain unchanged.
The operation order is base projection, existing adapter correction, new patch,
then residual addition. A patch consumes the original post-SwiGLU FFN activation,
before any base-weight Hadamard rotation; it never consumes the old LoRA's output.

The wrapper accepts only one row worker with a scalar projection request and no
tiled-prefill or accelerator executor. It exposes sequential evaluation and
ingestion, preserving the patch on every token, and rejects batched evaluation.
There is no service option, dynamic factor/scale setter or installed-model change.
The site, factor bits, scale and separate FFN execution profile enter a fresh
checkpoint identity. Old or foreign checkpoints cannot restore into the probe.
Scale zero bypasses all patch arithmetic exactly within this profile.

An observer receives original FFN hidden values and the post-FFN residual stream.
The callback runs before token commit. Its values are provisional until the whole
evaluation succeeds; consumers must discard staged rows on later failure. Failed
evaluation does not advance the token's numerical state. Capturing all layers or
tokens is not implied by this API.

## Bounded native collector

Build the explicit `ffn_probe` example from reviewed source before any
owner-device allowance starts:

```text
cargo +nightly-2026-11-09 build --locked --offline --release --example ffn_probe --manifest-path <absolute-Cargo.toml> --target-dir <external-build>
ffn_probe <absolute-request.json>
```

The request has `schemaVersion:1`, absolute `modelPath`, optional absolute
`adapterPath`, eight lowercase hexadecimal `adapterScaleBits`,
`weightRepresentation` (`PQ2_0` or `current-repacked-PTQ1_0`), zero-based `layer`,
40-character lowercase `runtimeSourceCommit`, `absoluteDeadlineUnixMs`, a new
absolute `capturePath`, optional absolute `patchPath`, and `sequences`.
Each sequence supplies unique `sampleId`, `groupId`, `split` (`fit` or `holdout`),
`label` (`target` or `preserve`), `tokenIds` and `captureTokenIndices`. Identifiers
contain only ASCII letters, digits, period, underscore or hyphen, at most 128
bytes. A trace group cannot appear on both sides of the split.

The fixed collector bounds are 16 sequences, 256 aggregate teacher-forced tokens,
64 selected capture rows, 256 KiB request JSON, 4 MiB incoming patch JSON and
32 MiB serialized capture. Only the exact pinned Swift representations and
approved adapter are accepted. No adapter requires zero adapter scale. Labels
and token selections are supplied evidence, not an automatic failure diagnosis.

The capture contains `source`, `site` and `rows`. Source identities include the
verified model/adapter hashes, exact adapter scale bits, representation, actual
executable SHA-256, declared source commit, execution profile and actual complete
graph identity. Each row contains sample/group/split/label, token index,
`activation` and `residual`. Both vectors are original-basis F32 values; the
selected Swift site is `[17408,5120]` with `inputBasis:original-ffn-hidden`.
Complete capture publication uses a same-directory create-new temporary and
atomic, nonoverwriting hard-link publication; temporary files are removed.

An incoming patch must match the exact source and site, factor hashes, scale-one
bits, `after-standard-adapter` composition, research-only markers, and a certified
F32 optimization gap within the declared tolerance. This certificate establishes
an offline optimization result, not model quality. A capture taken with a patch
records the patched graph identity, so a subsequent fit cannot masquerade as one
made on an unpatched graph. Iterative multi-patch composition is not supported.

The original absolute deadline must leave at most 240 seconds on entry. Checks
cover intake, hashing, loading, token boundaries and publication. Loading and
synchronous graph operations require an external native Job/process supervisor
to enforce the hard deadline and memory admission. Follow the repository's
aggregate allowance, before/after and per-arm AC-power observations, cleanup and
evidence-retention rules. A returned failure is emitted on stderr and no success
receipt is implied by a missing capture.

Teacher-forced captures establish neither spontaneous repetition nor improved
correctness, creativity, performance, memory use or daily-service qualification.
Diagnose a real failure, separate fitting/tuning/held-out material, and compare
complete answers and simpler controls before applying a fitted correction.
