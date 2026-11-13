# Rank-one adapters

The graph and private HTTP server admit an explicitly selected approved rank-one
F32 adapter. JSONL workers and the ordinary token driver remain adapter-free.
Actual-model correctness, authoring basis and adapter effectiveness require
independent qualification.

## Admission and arithmetic

`model::lora::RankOneAdapter::load_swift` verifies the approved Orca artifact:

- Length: 9,682,464 bytes.
- SHA-256: `f1669534803d340a496015f5c45125f3437b4d13ec764f40e34488ce83967f42`.
- Original pinned Swift PQ2 base, 129 rank-one pairs, alpha one.

`load_swift_representation` also admits the exact current-weight PTQ1 export
when explicitly selected, with a matching base hash. Other PTQ1 artifacts reject.
General `RankOneAdapter::from_model` checks bounded metadata, complete finite
F32 pairs and base shapes; its caller owns artifact authenticity.

Targets are limited to `token_embd.weight`, `blk.N.attn_output.weight`,
`blk.N.ssm_out.weight` and `blk.N.ffn_down.weight`.
`Qwen35::with_rank_one_adapter` requires every target in the live graph and seals
adapter identity and scale bits into graph identity. Scale must be finite in
0..2. Either signed zero bypasses corrections. Scale reset clears state and
renews checkpoint provenance; see [session ownership](swift-session-profiles.md).

Embedding adapters require a separate output head. A tied
`token_embd.weight` output rejects them even at zero scale. Both pinned Swift
representations have separate output heads.

The [native graph reference](https://github.com/PrismML-Eng/llama.cpp/blob/73bfe6cc485554822f80c473a3c0c4f25e80f41b/src/llama-graph.cpp)
orders base projection, independent A/B products on original activations, scale,
then addition. Folded base Hadamard/grouped SSM permutations stay on the base
branch; embedding correction follows inverse transformation. F32 rounding stages
and pinned aligned-row accumulation are preserved.

Identical immutable B payloads share storage only after kind/shape/byte equality.
Correction scratch holds one rank value and one output-width vector, separate
from base scratch. One-column B projections stream immutable F32 scalars with
separately rounded products and binary64 reduction. No dense correction matrix
is constructed or process-memory benefit assumed.

## Capture actual-model correctness evidence

Use the [source-bound capture procedure](build-and-validation.md) and absolute paths:

```powershell
& $qualifier --rank-one-adapter $approvedAdapter 0 forward $originalPq2 $zeroEvidence scalar '248045'
& $qualifier --rank-one-adapter $approvedAdapter 1.3 forward $originalPq2 $nonzeroEvidence scalar '248045'
& $qualifier --rank-one-adapter $approvedAdapter 1.3 trajectory $originalPq2 $trajectoryEvidence auto $promptIds 16
& $qualifier --use-repacked-ptq1 --rank-one-adapter $approvedAdapter 1.3 forward $currentPtq1 $ptq1Evidence neon '248045'
```

Evidence directories must exist and differ. `load` accepts the adapter prefix;
`inspect`, relative paths and invalid scales reject. Captures include exact base/
adapter hashes, target count, scale bits, create-new logits, complete logical
KV/recurrent/convolution state and raw greedy IDs. They provide no benchmark verdict.

Compare loaded scale zero and intended nonzero scales against pinned native
execution. Capture residual-writer inputs/corrections, especially grouped SSM
authoring order, plus complete logits/state within
`1e-3 + 1e-3 * abs(reference)`. Synthetic input-order tests and final logits alone
cannot establish native authoring basis. Qualify private-service scale changes,
reset, streaming, cancellation and rollback separately on the actual target.

## Focused regressions

`rank_one_` controls cover metadata/factor/base rejection, scale bounds, signed
zero, rounded arithmetic, post-inverse embeddings, original writer input order,
checkpoint isolation and request rollback. Full-context, quality, resource and
performance acceptance follow the [validation guide](build-and-validation.md).
