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

## Entire-FFN surrogate teacher pairs

Set `captureKind:"ffn-surrogate"` to prepare the separate Runner
`bonsai-tools ffn-surrogate` fitting input. Omitting `captureKind`, or selecting
`hidden-patch`, preserves the existing post-SwiGLU capture format above.
Surrogate capture rejects `patchPath`; its teacher includes only the packed
base and the approved adapter at the declared frozen scale. An absent adapter
requires positive-zero scale bits `00000000`.

Each surrogate sequence uses `split:"fit"` or `"development"`; `label` can be
omitted. At least two fitting rows and one development row are required, with
the same 64-row aggregate bound. Final held-out answer material is rejected
from this fitting capture. Distinct group names do not prove content independence.

`FfnProbe::evaluate_surrogate_into` returns an owned teacher pair only after
the complete token succeeds. Its input is the original normalized pre-FFN
residual, before Hadamard rotation. Its target is the complete FFN output,
including the approved FFN-down adapter, captured **before residual addition**.
It never reconstructs the target by subtracting rounded residual streams.
Input/target copies are provisional until output projection and state staging
succeed; an error drops both, clears logits and commits no token state.

The surrogate JSON contains exactly `schemaVersion:1`, `source`, `site` and
`rows`. `source` retains the executable/source/base/adapter/scale/profile/graph
identities described above. `site` is:

```json
{
  "layer": 0,
  "inputBasis": "original-ffn-normalized-input",
  "targetBasis": "complete-ffn-branch-delta",
  "inputWidth": 5120,
  "outputWidth": 5120
}
```

Rows contain only `sampleId`, `groupId`, `tokenIndex`, `split`, `input` and
`target`. The numerical profile remains
`experimental-ffn-cpu-scalar-unfused-v1`; this is no production NEON capture.
Before fitting, add the fitter's bounded `controls` and new `outputPath` if
needed. In particular, select `controls.rank` no greater than the number of
fitting rows; the fitter's default rank eight needs at least eight fitting rows.

This operation still executes and loads the complete FFN. It supplies no skip,
unloading, latency, process-memory or cognitive-quality result. Whole-FFN
replacement/loading and independent complete-answer acceptance remain separate.

## One-site structured omission pairs

Set `captureKind:"ffn-group-omission"` and supply `omittedGroups`, a strictly
increasing nonempty list of original 1,024-coordinate group indices from 0 to 16.
At least one group must remain. The request version remains 1; the resulting
fitting packet uses schema 2. This mode rejects `patchPath` and uses the same
fit/development-only split rules as the surrogate collector. Other modes reject
`omittedGroups`, preserving the existing schema-1 capture formats.

The collector first replays the supplied fixed token IDs through the original
teacher and retains the requested complete pre-add FFN targets. It then consumes
and resets that same graph into one immutable omission profile, resets before
each sequence, and replays exactly the same tokens. Both passes share the
original absolute deadline and the 256-token aggregate ceiling, so the input
sequences may contain at most 128 tokens in total. There is one model load and
no second model payload copy. Both passes, reset, hashes, retention and cleanup
belong in the predeclared owner allowance.

At the selected site, the omission graph computes the complete FFN branch on
its current normalized original-basis input. It then zeroes each omitted
**original hidden group**, evaluates down projection and standard LoRA on that
same zeroed hidden vector, and propagates the retained branch through later
layers. The teacher and retained branch are copied before residual addition and
returned only after the whole token succeeds. Failed-token rows are discarded.
The scalar profile is
`experimental-ffn-group-omission-cpu-scalar-unfused-v1`; its group/sign recipe
enters a fresh checkpoint identity, invalidating original teacher checkpoints.

The schema-2 packet matches Runner's structured `ffn-surrogate` input:

- `source` identifies the original teacher profile, model, executable, source,
  approved adapter and exact scale bits.
- `studentExecution` identifies the changed graph/profile and `prefixPolicySha256`.
- `site.targetBasis` is `teacher-minus-retained-ffn-branch`; the input basis remains
  `original-ffn-normalized-input` and both vector widths are 5,120.
- `omission` records original groups, width 1,024, original hidden width 17,408,
  `transform:independent-contiguous-1024-signed-hadamard-v1` and
  `retainedAdapterPolicy:zero-omitted-original-hidden-before-standard-adapter`.
- `omission.signIdentity.fullSignsSha256` hashes the actual complete explicit
  metadata sign values, in Swift's width order `[5120,6144,17408]`, as canonical
  I32-LE values +1/-1. The down slice starts at element 11,264 and contains
  17,408 values. Signs precede independent block transforms; omission never
  compacts or reindexes their original slices or the down-projection columns.
- Each row contains `sampleId`, `groupId`, `tokenIndex`, `split`, `input`,
  `teacherTarget`, `retainedBranch`, `originalPrefixTeacherTarget`,
  `sharedInputSha256`, `studentPrefixSha256` and `originalPrefixSha256`.

`teacherTarget` and `retainedBranch` use exactly the same changed-graph input.
`originalPrefixTeacherTarget` comes from the matched first-pass sample/token;
its separate error diagnostic exposes local input drift rather than silently
substituting an original-graph target for a changed-graph target. An unchanged
input at this sole edited site during fixed-token replay does not establish
multi-site composition or freely generated continuation quality.

`sharedInputSha256` hashes the exact F32-LE normalized input. Prefix hashes use
domain `prism-ffn-capture-token-prefix-v1\0`, then graph-identity UTF-8 length as
U64-LE, graph bytes, token count as U64-LE, and ordered U32-LE token IDs through
the current token. They identify graph and token history, not a cryptographic
digest of all numerical state. The prefix policy hashes domain
`prism-ffn-student-prefix-policy-v1\0` plus canonical serialized JSON describing
the graph, group selection, no-recovery policy, teacher-forced rollout, reset
policy and exact sequence identities/token IDs. `captureSha256` hashes domain
`prism-ffn-structured-capture-packet-v1\0` plus the serialized packet before
inserting that field; it is not the final capture file's SHA-256. Retain the
exact request and final file digest with any required independent replay inputs.
The fitter still treats supplied vectors and identities as unauthenticated.

Add bounded fitting `controls` and optional new `outputPath` before using the
packet with `bonsai-tools ffn-surrogate`. Rank must not exceed fit rows. This
collector runs **without compensation**, computes the complete gate/up branch
and loads every base weight. It provides neither selective loading nor saved
computation, actual RAM or latency. Compensated-prefix capture, multiple edited
sites, freely generated answers and production arithmetic remain separate work.
