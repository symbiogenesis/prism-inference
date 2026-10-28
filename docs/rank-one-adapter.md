# Rank-one adapter correctness path

The source graph implements an explicit rank-one F32 adapter path for numerical
qualification. The private HTTP server, Swift JSONL worker and ordinary token
driver continue to reject adapters. This source support is not a qualified
Runner default, a released-runtime claim or evidence of adapter effectiveness.

`model::lora::RankOneAdapter::load_swift` verifies the approved Orca file's exact
9,682,464-byte length and SHA-256
`f1669534803d340a496015f5c45125f3437b4d13ec764f40e34488ce83967f42`.
It requires the original pinned Swift PQ2 base and all 129 rank-one pairs with
alpha one. PTQ1 remains outside this adapter capture path. The general
`RankOneAdapter::from_model` constructor admits a bounded explicit subset after
checking metadata, complete F32 factor pairs, finite payloads and matching base
shapes; callers of that constructor own artifact authenticity.

Only `token_embd.weight`, `blk.N.attn_output.weight`, `blk.N.ssm_out.weight` and
`blk.N.ffn_down.weight` are admitted. `Qwen35::with_rank_one_adapter` checks that
every target is in the live graph. It seals the exact adapter identity and scale
bits into the graph identity. Scale is fixed for that graph and must be finite
within 0..2; a different scale requires a fresh graph, preventing reuse of state
computed under another correction.

The [pinned native graph](https://github.com/PrismML-Eng/llama.cpp/blob/73bfe6cc485554822f80c473a3c0c4f25e80f41b/src/llama-graph.cpp)
defines the operation stages: base projection, independent A/B products on the
original activation, scale, then add. Folded base Hadamard and grouped SSM
permutations stay on the base branch. Embedding corrections follow inverse
embedding transformation. The implementation retains each F32 rounding stage,
uses the existing pinned Swift accumulation contract for aligned factor rows
and bypasses correction arithmetic entirely at either signed zero scale.

The loader shares identical immutable B payloads only after kind, shape and
bytes match. Correction scratch is separate from base projection scratch and
reuses one rank value and one output-width vector. No dense vocabulary-sized
correction is constructed. These storage properties do not establish process
peak RAM, a lower admission budget or a speed improvement.

## Capture actual-model correctness evidence

Build the `swift_model_check` example through the external working-directory and
pinned compiler convention in [building and validation](build-and-validation.md).
From a new external native working directory, use the absolute executable path:

```powershell
& $qualifier --rank-one-adapter $approvedAdapter 0 forward $originalPq2 $zeroEvidence scalar '248045'
& $qualifier --rank-one-adapter $approvedAdapter 1.3 forward $originalPq2 $nonzeroEvidence scalar '248045'
& $qualifier --rank-one-adapter $approvedAdapter 1.3 trajectory $originalPq2 $trajectoryEvidence auto $promptIds 16
```

Every path variable must contain an expanded absolute path; evidence directories must
already exist and be distinct. The adapter must be the exact approved file.
`load` also accepts the prefix; `inspect`, PTQ1, relative adapter paths, unresolved
environment tokens and invalid scales reject. The example remains a source
correctness tool and is not added to release bundles or ordinary service launch.
It reports adapter hash, target count and scale bits and retains existing
create-new logits, complete logical KV/recurrent/convolution snapshots, raw
greedy IDs and evidence-byte bounds. It produces no timing or benchmark verdict.
No full-model adapter capture was performed as part of the implementation.

Before service integration, compare the same approved artifacts against pinned
native execution at loaded scale zero and intended nonzero scales. Establish
residual-writer inputs/corrections, especially grouped SSM authoring order;
complete logits and live state must retain the existing
`1e-3 + 1e-3 * abs(reference)` bound. Synthetic tests prove that this Rust branch
uses original input order; they cannot establish the publisher's authoring basis.
Final logits/state captures alone do not replace native writer-level evidence.

After those numerical checks, add a separately reviewed immutable release and
explicit private-service adapter readiness/request identities; qualify scale
changes with fresh state, exact count/generation policy, cancellation, reset,
streaming and rollback. Then run full-context/long-output, resource, same-engine
TPS and independent human quality gates on the actual target before promoting
Runner's adapter-enabled default. All benchmarks and complete suites remain
main-only CI work; focused synthetic tests and builds may run locally.

## Focused regressions

`rank_one_` tests cover malformed factors/metadata/base identities, finite scale
bounds, both signed zero bypasses, separately rounded products and addition,
post-inverse embedding correction, unrotated attention/recurrent writers,
untiled grouped SSM input order, and checkpoint/request rollback. They use small
synthetic graphs and make no full-model or native-adapter qualification claim.
