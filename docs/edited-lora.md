# Edited LoRA research profile

`RankOneAdapter::load_edited_swift` and
`SwiftSession::load_edited_profile_with_context` admit one explicitly selected
edited Orca artifact against an exact supported current Swift base. This is a
source-only research entry point. Ordinary loaders and the released service
continue to require the approved adapter's fixed hash.

The candidate has an absolute path, exact lowercase SHA-256 and the original
9,682,464-byte GGUF length. It must retain alpha one and all 129 F32 rank-one
sites: embedding, 64 FFN-down, 48 SSM-output and 16 attention-output factors.
Shapes match the authenticated base. This admission verifies the candidate
bytes and structural compatibility; the offline editor's independent report
must establish which coefficients changed. It does not authenticate authoring
history or resolve the original adapter's SSM input-basis uncertainty.

An edited linear A vector may contain exact zeros. If its nonzero coefficients
plus U16 input indices occupy fewer bytes than the dense F32 vector, the loader
retains those coefficients and indices and drops the decoded dense vector.
Serialized linear A tensors are also released. Every linear site must retain
a nonzero coefficient. Embedding lookup and shared B vectors retain their
existing storage and arithmetic. The GGUF itself remains an ordinary dense F32
adapter, so sparse runtime storage does not reduce this candidate's file size.

Sparse dots consume the original activation basis, validate every activation
for finiteness including omitted inputs, accumulate retained products in F64
in ascending input-index order, and round once to F32 before the ordinary B,
scale and base-add stages. This is a distinct experimental numerical profile;
it is not bit parity with the dense scalar or NEON reduction. Both initial and
reset graph identities include `edited-sparse-a-u16-f64-v1`, so the same file
cannot share a graph identity with generic dense execution. Zero adapter scale
still bypasses the correction exactly. Multi-token correction uses the same
sparse operation for each token.

`AdapterStorage` and `SwiftSession::adapter_storage()` report retained factor
payload, separate from process commit, peak memory or timing. Their sum includes
dense/sparse linear coefficients, sparse indices, serialized embedding A and
unique decoded B allocations, and excludes map/vector headers and allocator
overhead. A smaller factor payload or fewer nonzero multiplies does not prove
faster useful answers or preserved accuracy. Those require frozen-candidate
comparisons with complete answers, original aggregate deadlines, power evidence
and external process supervision. No production default changes follow from
source checks or a small factual smoke.
