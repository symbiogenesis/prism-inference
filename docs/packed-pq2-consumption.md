# Packed PQ2 activation consumption layout

The native Windows ARM64 baseline NEON PQ2 path prepares activation bytes in
weight-digit consumption order once per scoped `PreparedMatvec`. The low,
second, third and high two-bit fields of sixteen packed bytes then pair directly
with four activation vectors. This removes the weight-unpack interleave network
from every output row. Two integer accumulators keep the per-block dot exact;
there is no optional DOTPROD or I8MM requirement.

Only original PQ2 projections with input columns divisible by 256 select this
path. The original Q8_K quantizer, Q8_K blocks and their scales/sums remain
unchanged and available to other formats and explicit external executors. PQ2's
fourth symbol remains +2. Odd-group PQ2 and PTQ1 retain their existing Q8_0
paths. Scalar forcing, unsupported targets and builds without NEON allocate no
consumption-layout storage.

The derived layout is reused across rows and compatible sibling projections in
one preparation scope. Beginning any new preparation clears its logical length,
including attempts rejected before dispatch. Input addresses and retained
capacity do not authorize reuse. Preparation growth is fallible and failures
clear provisional layout/output. Row workers share immutable activation storage;
the persistent worker budget, row error ordering and joining stay unchanged.

## Arithmetic and resources

Each original 128-weight block retains the original activation scale and the
original per-128 weight scale multiplication. Integer regrouping is exact, with
absolute complete-dot bound 32,768 including activation -128 and weight +2.
Pinned mode retains one FMA per block; generic mode retains multiply then add.
The final finite-result check is unchanged. No weight tensor is repacked or
expanded.

For C input columns, live derived bytes are C activation bytes plus one f32 scale
per 128 columns, before Vec capacity and allocator overhead. A 5,120-column
projection therefore needs 5,280 logical payload bytes. This is additional to
the original Q8_K scratch, not a replacement or a reduction in admission. Each
scratch owner retains its maximum capacity across preparations; there is no
cross-request activation-content cache. Row workers borrow this one allocation
and do not retain separate copies. Full-model scratch high-water accounting and
actual process resources remain qualification requirements.

## Validation and delivery

The layout tests independently enumerate original positions, preserve source
Q8_K codes/sums/scales, reject nonfinite preparation and verify error clearing.
Production-dispatch tests compare scalar and automatic/native paths bitwise for
generic and pinned accumulation, complete/partial row-worker tiles, +2 symbols,
signed-zero scales, multiple input widths and changed input in reused storage.
ARM64 NEON tests independently decode exhaustive packed bytes and isolate every
activation lane including extremes.

Foundation's existing native variants lane explicitly runs the packed layout
tests with `parallel-rows`, and additionally runs the new NEON integer controls
on Windows ARM64. Existing full graph/state/adapter and scalar fallback controls
remain required. No CI execution or actual-model performance is implied by the
source change; source-bound owning CI, actual-model logits/state and matched
whole-request throughput/resource evidence remain pending before release use.
