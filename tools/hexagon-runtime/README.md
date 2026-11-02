# Experimental native Hexagon graph integration

This unpublished Rust crate joins Prism's ordinary Qwen3.5 graph to a single
caller-owned signed X2 projection session. The published CPU runtime depends on
neither this crate nor the unpublished native driver boundary.

`GuardedQ2Executor` uses the existing guarded HVX K=256/N=2 primitive.
`HmxQ2Executor` uses the separate K=128/N=8 binary-plane HMX primitive. Both
implement the public CPU runtime's `Pq2ProjectionExecutor` extension and consume
the graph's existing prepared Q8_K codes, without another activation quantizer.
Original PQ2 scale bits and per-128-block fused/nonfused reduction order remain on
CPU. The HMX rows are the eight bits of one token's codes, with signed high-bit
coefficient -128; they are not eight prefill tokens.

The graph applies the existing Hadamard/sign/grouped transformations before
projection and the existing approved rank-one adapter corrections afterward.
Embedding, attention, GDN, convolution, sampling and complete committed state
remain in their original Rust graph. PTQ1 and odd-block PQ2/Q8_0 use the configured
CPU implementation. Any submitted error clears provisional output and permanently
closes the executor and scoped decoder; no failed work is replayed on CPU.

Use `Qwen35::with_projection_executor` inside
`Driver::reference_q2_projection` or `Driver::reference_hmx_q2_projection`'s work
callback. Place the resulting borrowed `ProjectedQwen35` in an ordinary `Runtime`
to retain request checkpoints, cancellation checks and prefix policies. The
borrow spans every projection and token in that callback, so module/session
open and close occur once for the whole explicitly bounded workload. Keep all
outputs provisional until the native report's `completed_cycle()` succeeds.
Only then may a trusted supervisor publish a terminal qualification result.

`maximum_owned_payload_bytes` separately bounds original-F32 provisional output,
one padded packed-weight tile, returned limb/plane results and activation payload.
The original model/prepared activations/caller output, allocator and stack overhead,
native shared allocation, VTCM and resources outside the process are additional.
Insufficient host payload or unsupported geometry selects CPU before submission;
native operation/deadline exhaustion fails the scope. Native bounds remain
4..=120 seconds with a three-second cleanup reserve and at most 21,760 operations.
They are terminal experiment limits, not a daily service session contract; a
complete model/token may exceed them. An outer process deadline, exact signed
package/driver verification, child-only environment and verified teardown remain
the trusted caller's responsibility.

For the retained Swift geometry, source arithmetic at the maximum 17,408-row
tile requires 22,656 submissions for all eligible projections of one ingestion
token and another 600 for its vocabulary head (23,256 for evaluation). This
exceeds the terminal limit before even one complete token. Those derived counts
are not a device observation: full-model use needs an admitted explicit partial
placement or a separately reviewed persistent-session/operation contract.

Model-free controls compare complete fused and nonfused matvec bits, multiple and
partial row tiles, all i8 bitplane codes and refusal/poisoned output behavior.
The CPU runtime's `projection_scope` graph regressions cover attention/recurrent
state, transforms, loaded-zero/nonzero adapter corrections, checkpoints, reset,
unsupported CPU paths and terminal failure after provisional attention state.
Hosted Hexagon and Foundation CI own these tests and Clippy/documentation checks.

This source integration enables no Runner profile or automatic backend selection.
Signed HMX execution and actual shapes, PQ2/PTQ1 full logits/state and adapter
qualification, model operation/resource admission, cancellation/device loss,
sustained use and paired end-to-end benefit remain pending. Native invocation,
successful loading or an exact CPU/mock control does not establish NPU speed.
