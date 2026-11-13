# Runtime conformance

Prism owns the upstream source/backend catalog, codec parity and CI orchestration.
Runner owns application integration. Source-level conformance, full-model
numerical parity and target-device performance are separate gates.

## Source and assertion ownership

[source.lock.json](../tools/prism/source.lock.json) pins the upstream archive,
length and SHA-256. CI verifies and extracts it into fresh external storage.
[PRISM-LICENSE.txt](../tools/prism/PRISM-LICENSE.txt) supplies upstream terms.

The [catalog](../tools/prism/upstream/catalog.json) records all 71 recursive
`tests/**/test-*.cpp`, `.c`, `.py` and `.sh` files, their identities and
digests, and the test-registration CMake digest. This measures source-file
coverage, not assertion counts. Helpers, diagnostics and benchmark-only sources
remain visible.

The [migration inventory](../tools/prism/upstream/migration.json) is the canonical
per-source account of assertions, prerequisites and adapter status. The executable
adapter list in [harness.rs](../tests/upstream_prism/harness.rs) must match it.
All 70 Windows-applicable sources have safe Rust adapters; Apple-only wired
memory remains unsupported and never counts as a pass.

Safe Rust owns inputs, reference calculations, assertions and tolerances.
Each adapter runs the same assertions against pinned native Prism and the Rust
runtime. Native bridges provide bounded serialization and production calls;
they contain no expected outputs, random generators or acceptance policy.
Upstream test executables and fixture exporters are not built.

The PQ2/PTQ1 GGML bridge is Rust linked to pinned production ggml; its reference
target is named `cpp`. The separate common API bridge is native.
[Bridge documentation](../tools/prism/upstream/README.md) covers build commands,
protocols and boundary checks.

## What runs

| Lane | Scope | Exclusions |
| --- | --- | --- |
| [Foundation regression](../.github/workflows/foundation.yml) | 23 selected Rust source contracts; affected native codec parity | No native reference replay, external model/tokenizer corpus, exhaustive backend matrix or long timed schedules |
| [Runtime conformance](../.github/workflows/runtime-conformance.yml) | Complete source/reference and CPU backend gate | Benchmarks and gradients require explicit selection |
| [Focused reference](../.github/workflows/prism-focused.yml) | One named source or `BackendPlan` diagnostic | Does not satisfy the complete gate |

The [regression selector](../.github/ci/runtime_regression.py) covers codecs,
tensor arithmetic, sampling, grammar, templates, allocation, GGUF and PEG.
It permits 120 seconds after compilation and 30 seconds per source.
Every selected result must pass; missing, duplicate, failed or unsupported
results reject the job. Receipts explicitly record `FullSourceGate=false`
and `NativeReferenceReplay=false`.

Complete conformance runs on `main` with four ordinary source partitions,
16 backend planning partitions and 32 execution partitions. It requires every
source group and backend partition. A green routine or focused result cannot
replace it. CI dispatch and local budgets follow [AGENTS.md](../AGENTS.md);
see [workflow selection](build-and-validation.md#owning-workflows).

## Backend matrix

The backend source includes 112 active CPU classes and 22,817 ordered geometries
in each evaluation and support mode. Rust interprets the pinned source program,
builds inputs and computes reference metrics. Plans bind complete geometry sets,
source-order random offsets and logged reproducible seeds before execution.

CPU selection is explicit. The selected runtime receives the source thread
count; the reference retains its production default. Each native partition uses
a bounded persistent process with fresh state per geometry. Transport controls
check file/pipe equivalence, state isolation and malformed-input rejection.

Compact `record.json` handoffs bind revision, identities, ordering and
full-result digests. Complete case receipts and diagnostics remain in the
separate backend evidence artifacts. Fan-in rejects absent, duplicate,
changed or incomplete results.

The default gate requires evaluation and support preflight. The 1,530 performance
workloads and complete gradient matrix are explicit CI opt-ins. The pinned
backward builder supports split SWIGLU but aborts on GEGLU; that upstream failure
remains a diagnostic, not a passing gradient result.

## Reference corrections and scope

The reference applies two hash-verified production corrections to external
translation units; receipts identify their resulting bytes:

- [DSpark saver](../tools/prism/upstream/patches/dspark-saver.patch) adds metadata
  and global tensors needed to reload a saved DSpark model.
- [Llama 3 decoding](../tools/prism/upstream/patches/llama3-cleanup.patch)
  preserves punctuation bytes for `llama3`, `llama-v3` and `llama-bpe`.

The immutable archive and original assertions remain unchanged.
[Input provenance](../tests/upstream_prism/INPUTS.md) and the
[locked BPE acceptance inputs](../tools/prism/upstream/BPE-ACCEPTANCE.md)
define generated fixtures, native random streams and tokenizer exceptions.

Interpret source results according to the inventory:

- A compile-only or diagnostic source with zero numerical assertions does not
  establish numerical correctness. Reports distinguish workload completion,
  diagnostic mismatches and strict reference equality.
- Model sources need their exact model, vocabulary and reference assets.
  A current Swift artifact cannot substitute for another source's fixture.
- Benchmark timings include the reported transport/metadata work and do not
  measure isolated native kernels. All timed sources run only in CI.
- CPU transcriptions of CUDA/Metal-oriented sources qualify only their stated
  host-side assertions. GPU execution remains separate.
- Optional upstream Python diagnostics report availability and skips explicitly;
  their absence cannot count as executed coverage.
- Defined corrections, platform exclusions and capability skips remain visible.
  Missing prerequisites or unsupported Rust operations stay red.

## Floor and retained evidence

[baseline.json](../tools/prism/upstream/baseline.json) binds the catalog digest,
upstream revision, minimum green count and required passing IDs.
`RequiredReferencePassedIds` protects native controls independently of Rust
progress. For each complete run:

1. Verify the unchanged source inventory and catalog binding.
2. Require one Rust and one reference result for every implemented adapter.
3. Expand remaining sources to `unsupported` for Rust.
4. Count only `pass` as green; `fail`, `skip` and `unsupported` stay red.
5. Require every protected ID and the minimum integer ratio. New passes cannot
   compensate for an existing regression.

Missing adapters, duplicate/unknown IDs, malformed reports, compilation/process
failures and timeouts reject the gate. Expected Rust failures are tolerated only
within the committed floor; reference failures always reject it.

Retain `report.json`, `gate.json`, source/runtime receipts, separate target
results and bridge inputs needed for replay. Keep nonempty compiler/process
diagnostics when required to diagnose failure. Cache only verified immutable
inputs or compiler outputs; results and generated observations must be fresh.

## Growing the baseline

Preserve original inputs, assertions and tolerances when adding an adapter.
Keep policy in safe Rust and extend native bridges only with production calls
and serialization. Prove the shared assertions against pinned Prism, add the
stable ID to `AdapterIds`, and update catalog/baseline bindings together.

After complete hosted acceptance, review the retained report before increasing
`MinimumPassed` and `RequiredPassedIds`. Keep every protected ID and record
the tested revision/run identity in baseline provenance. CI never rewrites the
floor automatically. Review upstream pin changes and exceptions explicitly.

## Focused local checks

Full catalogs and benchmarks remain CI-only. A focused policy check is:

```powershell
./tests/Test-PrismBaseline.ps1 -Case Ratchet
```

One adapter can run with the pinned archive, locked dependencies and a C++
toolchain. Expand all paths below to absolute paths outside Git:

```powershell
./tests/Test-PrismUpstream.ps1 -SourceArchive $archive `
    -EvidenceDirectory $freshEvidence -TargetDirectory $externalCargoTarget `
    -Case Pq2RowShapes -Generator 'Visual Studio 18 2026' -NativePlatform x64
```

An x64 reference under Windows ARM64 emulation supplies correctness evidence,
not native-device performance. Hosted acceptance uses native Windows ARM64.
Use `Ninja` for Linux or a configured compiler environment.
