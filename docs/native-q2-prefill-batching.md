# Resident multi-tile native-Q2 prefill packets

The production `NativeQ2Executor::project_f32_tokens` route now batches resident
weight tiles across independent prompt tokens. This closes a separate queue-cost
gap left after tile-major packing reuse: a projection spanning several native
row or column operations previously sent one queue packet per tile per token.
The existing native session executes up to eight ordered `N=1` operations in one
correlated packet. Complete groups of eight use `submit_resident_multi8`; tails
use four, three, two or one operation. Every token retains its original tile
accumulation order and original single-token solver arithmetic.

For `T` weight tiles and `n` prompt tokens, native queue writes become
`T * (n / 8 + ceil((n % 8) / 4))` with integer division, instead of `T * n`.
An 8/16/32-token tile therefore uses one eighth as many queue writes. Logical
operations remain `T * n`; session work admission, absolute deadlines, native
response validation and retained conservative resource charges do not change.
These are source operation counts, not measured request benefit.

`NativeQ2Submitter::maximum_resident_token_batch` explicitly identifies genuine
single-packet support. Arbitrary submitters default to one; the real borrowed
session and private owned session advertise eight while healthy. The public
owned facade and borrowed trait forwarding preserve that result. Accepted
limits are 1, 2, 3, 4 and 8; other values conservatively select scalar submission.
No unsupported method is discovered by submitting and retrying it.

Resident preparation releases its temporary packed-weight vector and retained
packing scratch before allocating at most eight padded output rows. For a
padded row extent `P`, the outputs cost at most `32 * P` bytes, fitting within
the existing reservation of at least `36 * P` packed-weight bytes plus `4 * P`
partial-output bytes. Caller output and metadata charges remain unchanged.
The path allocates only enough rows for the actual token count up to this cap.
Transient weights retain the original single partial-output allocation and
scalar queue path. Every native destination is a disjoint mutable slice of
one bounded allocation; repeated immutable weight handles are intentional.

The native packet validates all dimensions, resident registry ownership,
aggregate logical-operation capacity, sequence, descriptor range, tensor and
operation counts, and finite returned output before success. Runtime partials
retain finite checks and zero-padding checks while accumulating. Runtime
completion counters advance only after every member of a packet validates and
accumulates successfully. A later failure clears all caller outputs and faults
the executor; provisional completed-packet counts remain diagnostic. No failed
work is replayed or continued on CPU.

## True multi-token solver boundary

This implementation is not true `N > 1` native-Q2 matrix prefill. The local
source audit identifies `hvx_mv_params` as a specialization for `N=1`, kernel
`HTP_MM_KERNEL_HVX_QUANT_ROW_N1` (7), eight HVX workers and its declared VTCM
layout. Packet activation/output descriptors also explicitly contain one row.
Changing their dimensions alone would reuse invalid solver and scratch data.
The retained Runner signed-source audit records an HMX eligibility transition
above four activation rows and different internal quantization/reduction
contracts. Thus 8/16/32 matrix rows require a separately reviewed kernel choice,
solver, tensor strides, bounded VTCM and host scratch, numerical contract and
signed-device evidence. No authenticated local `N > 1` solver closure is claimed.

## Controls and pending acceptance

The focused numerical control uses an independent original-PQ2 decode oracle,
partial row tiles, column splits, 2/3/5/7/8/9/16/32 tokens, capability limits
0/1/2/3/4/5/7/8/9, preserved caller output
storage, tile-major activation order, cached repeats and exact packet counts.
The failure control covers early and late refusal, nonfinite results, nonzero
padded rows, complete output clearing and permanent refusal without replay.
Existing scalar/transient admission and malformed-input controls remain required.
Owned-facade controls verify capability forwarding. The actual native fake-queue
control checks four/eight-operation packets, repeated resident handles, every
output member and aggregate-cap refusal before any further queue write.

These model-free controls do not establish signed arithmetic, effective DSP
placement, full-model state/adapter parity, cancellation/device loss, vendor
teardown/resource acceptance or matched whole-request benefit. Owning hosted
checks and those qualification gates remain pending before release and Runner
repinning. No model, accelerator, benchmark or CI operation is part of source
implementation.
