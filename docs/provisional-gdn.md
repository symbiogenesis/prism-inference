# Provisional GDN decode state

The ordinary pinned Swift CPU graph evaluates eight recurrent rows directly
into its unpublished `LayerDelta::Recurrent.matrix`. Eligible Windows ARM64
baseline NEON uses those rows for both decayed and updated state. This removes
the helper's private eight-row staging array and final flush into the same
graph delta. Other arithmetic, feature-disabled/portable builds, unsupported
row widths and tiled multi-token recurrence retain their existing helpers.

The new helper accepts only the graph's disposable destination. Committed
recurrent state and checkpoints remain immutable sources. Any failure rejects
the complete graph evaluation before the final state swaps; dirty scratch is
never a recurrence input. A native refusal or nonfinite intermediate replays
the original defensive helper from immutable source rows, preserving its error
precedence. Existing row-helper transactional and partial-error contracts are
unchanged. Retry overwrites every accepted destination from committed state.

Arithmetic retains separate decay multiplication, the original four-register
FMA dot/reduction tree, `(value - dot_k) * beta`, fused update and query FMA.
All eight key dots precede updates. Two four-row phases retain sixteen dot
accumulators rather than thirty-two; the small key/query inputs are read again
for each phase, as in the previous eight-row helper. No precision, model state,
profile, cancellation boundary, checkpoint layout or admission limit changes.

At Swift's checked 48 recurrent layers, 48 value heads and 128-element rows,
one full final-flush copy is `48 * 48 * 128 * 128 * 4 = 144 MiB` of payload per
complete token. Eliminating that source-level copy omits 288 MiB of logical
read/write operations. The former eight-row staging payload is 4 KiB at this
width. These counts establish neither machine traffic nor stack size, usable
memory savings or latency benefit; cache locality and compiler output require
independent review. Conservative memory charges stay unchanged.

The focused `x1_provisional_gdn_` controls compare every row/output bit against
an independent scalar recurrence across successive deltas, signed zero,
subnormals, supported native capacities and awkward portable widths. They
reject shape errors before mutation, compare late-failure errors with the
defensive control and retry dirty scratch from unchanged committed sources.
A native-only control calls the direct work path so successful fallback cannot
substitute for exercising it. These controls do not qualify complete model
logits/state, adapters, cancellation or performance. Final-source strict native
and x64 Release analysis/linking, feature-disabled controls, ARM64 codegen and
matched CPU-only complete-request benefit remain required before promotion.
