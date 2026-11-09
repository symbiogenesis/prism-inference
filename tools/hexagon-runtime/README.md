# Experimental native Hexagon graph integration

This unpublished Rust crate joins Prism's ordinary Qwen3.5 graph to a single
caller-owned signed X2 projection session. The published CPU runtime depends on
neither this crate nor the unpublished native driver boundary.

`NativeQ2Executor` prewarming can initialize a fresh resident `Q2_0` extent directly
through the native submitter's synchronous, exact-length callback. Packing then
writes into its admitted `rpcmem` arena, retaining the same byte transcode and
row-worker pool. The prospective registry and handle stay private until packing,
all packed scales and the existing deadline pass. Callback errors close admission;
unwinding publishes no handle or registry growth. Existing handles and neighboring
extents remain untouched. Submitters without this optional capability retain the
scratch-and-copy API and its original input validation.

For the current 168-projection selection, this removes a nominal
`2,888,663,040` bytes of startup host copying and the temporary packed-weight
scratch from the direct prewarm path. This is a source work reduction, with no
measured startup, RAM, request-throughput or DSP-transfer verdict. The existing
host/resource admission, six mapped arenas, signed kernels, queue/cache protocol,
operation limits and CPU placement remain unchanged. Hosted model-free controls
compare direct and copied preparation across partial-row/column tiles and retain
failure, unwinding, extent-bound and secondary-arena controls. Signed-device,
actual-model state/adapter parity and matched whole-request benefit remain pending.

`GuardedQ2Executor` uses the existing guarded HVX K=256/N=2 primitive.
`HmxQ2Executor` uses the separate K=128/N=8 binary-plane HMX primitive. Both
implement the public CPU runtime's `Pq2ProjectionExecutor` extension and consume
the graph's existing prepared Q8_K codes, without another activation quantizer.
Original PQ2 scale bits and per-128-block fused/nonfused reduction order remain on
CPU. The HMX rows are the eight bits of one token's codes, with signed high-bit
coefficient -128; they are not eight prefill tokens.

Original-PQ2 `HmxQ2Executor` additionally supports explicit immutable weight
residency. The caller first reserves a bounded native arena with
`HmxQ2ProjectionLimits::with_resident_weights`, then enables the executor's
`with_resident_weights(maximum_cache_metadata_bytes)`. The host metadata
reservation is charged inside `maximum_owned_payload_bytes` for every supported
projection, so eligibility and request operation counts stay stable as the cache
fills. Equal numbers of projection records and tile handles are allocated up
front; each tensor consumes one record and one handle per block/row-tile operation.
Original immutable source bytes are retained through strong `Arc` ownership,
without another model copy. Original model retention, allocator overhead and the
separately bounded native mapping remain additional resource charges.

An entire projection's padded packed bytes and metadata must fit before its first
preparation. All its tiles are then prepared before any kernel submission. A
capacity miss uses the ordinary streaming HMX path for the entire projection;
unexpected preparation refusal or error closes the executor with no partial
fallback. Cache identity checks shared immutable source ownership plus tensor
kind/dimensions, avoiding raw pointer keys, address reuse and per-token hashing.
Repeated projections reuse the native handles and skip weight packing/copying.
There is no eviction and no renewal of operation/deadline limits. The native
arena still participates in the existing mapping/descriptor/cache protocol: this
does not establish reduced DSP DMA traffic, signed-device parity or speed.
`residency_statistics` reports complete preparations, cache hits, copied packed
bytes and reserved host metadata payload independently of kernel work counts.

`HmxQ8Executor` opts into PTQ1 and odd-block PQ2 using the graph's original
prepared Q8_0 codes and per-32 scales. Four sparse output rows preserve the four
integer DOTPROD lanes of one original row, with four K128/N8 submissions per
128-weight block. Pinned PTQ1 keeps four independent FMA accumulators across the
complete row and the final adjacent-pair reduction. Generic PTQ1 and odd-block
PQ2 retain their original per-32 and per-128 nonfused reduction order. Native
PTQ1 fractional-byte aliases retain decoder semantics; no canonical conversion
or expanded model copy is required. The fourfold row duplication reduces the
original-row tile capacity to the native limit divided by four. This exact
composition is an experimental compatibility path, not an efficiency claim.

The graph applies the existing Hadamard/sign/grouped transformations before projection and the
existing approved rank-one adapter corrections afterward. Embedding, attention, GDN, convolution,
sampling and complete committed state remain in their original Rust graph. PTQ1 and odd-block
PQ2/Q8_0 use the configured CPU implementation unless the executor explicitly opts in. Any submitted
error clears provisional output and permanently closes the executor and scoped decoder; no failed
work is replayed on CPU.

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
4..=120 seconds with a three-second cleanup reserve and at most 24,000 write
attempts.
They are terminal experiment limits, not a daily service session contract; a
complete model/token may exceed them. An outer process deadline, exact signed
package/driver verification, child-only environment and verified teardown remain
the trusted caller's responsibility.

Every executor reports its exact source-derived projection operation count and
remaining caller-bound work allowance. The native submitter additionally reports
the actual session's remaining writes, including any work preceding the executor
borrow; the effective allowance is the minimum of both bounds. Custom submitters
with opaque capacity still require independently matched caller limits.

`ProjectedQwen35` derives per-ingestion and per-evaluation costs from the actual
graph projections, excluding embedding row gathers and CPU placements. Bounded
executors must provide stable, nonzero counts for every selected projection.
The ordinary `Runtime` checks all unreused prompt forwards and the complete
reserved output length through `Decoder::admit_work` before starting new
inference or sampling. Early stop is not assumed. Direct graph calls also check
one complete token before executing its first projection. Capacity refusal
leaves the scope healthy; errors after admitted evaluation starts remain terminal.
Reset and checkpoints never replenish the native operation allowance. Operation
admission establishes no duration, memory, device placement or performance gate;
outputs still stay provisional until terminal native cleanup succeeds.

For the retained Swift geometry, source arithmetic at the maximum 17,408-row
tile requires 22,656 submissions for all eligible projections of one ingestion
token and another 600 for its vocabulary head (23,256 for evaluation). This
fits the owner-authorized 24,000-write cap for one evaluation token, with 744
writes left. One ingestion token plus one evaluation needs 45,912 writes and
still exceeds the cap. Complete-request admission continues to reserve all
unreused prompt work and the maximum output before the first submission;
the higher cap is not a complete-chat or persistent-serving qualification.
Those derived counts are not a device observation: useful full-model requests
need an admitted explicit partial placement or a separately reviewed
persistent-session/operation contract.

Model-free controls compare complete fused and nonfused matvec bits, multiple and
partial row tiles, all i8 bitplane codes, native PTQ1 DOTPROD captures and refusal/poisoned output behavior.
The CPU runtime's `projection_scope` graph regressions cover attention/recurrent
state, transforms, loaded-zero/nonzero adapter corrections, checkpoints, reset,
unsupported CPU paths and terminal failure after provisional attention state.
Hosted Hexagon and Foundation CI own these tests and Clippy/documentation checks.

This source integration enables no Runner profile or automatic backend selection.
`PlacedProjectionExecutor` freezes an explicit model-tensor subset before graph
construction. For example, select `blk.0.ffn_up.weight` while leaving the other
projections on CPU. Unknown model names, duplicates and unsupported arithmetic
reject at construction, and immutable storage/geometry checks prevent another
tensor from borrowing the selection. Its deterministic identity binds model,
executor, selected names and operation counts. The existing graph counts this
placement for complete-request admission before new inference; reset never
restores native capacity. Selection does not establish that an otherwise unused
model tensor is an executed graph projection: qualification must observe the
expected nonzero submissions and exact placement. Hosted controls exercise real
graph admission, unchanged logits/state and refusal before state mutation.

Signed HMX execution and actual shapes, PQ2/PTQ1 full logits/state and adapter
qualification, actual resource admission, cancellation/device loss,
sustained use and paired end-to-end benefit remain pending. Native invocation,
successful loading or an exact CPU/mock control does not establish NPU speed.
