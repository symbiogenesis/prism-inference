# Paired causal prefill values

Swift's six-head tiled-prefill consumer now shares cached value conversion
across two adjacent prompt tokens as well as cached key conversion. It computes
the common cached prefix once for twelve query rows. The first token uses its
original F32 current value, while the second sees that same position through
the committed F16 cache and uses its own original F32 current value.

Paired values use 32-channel tiles. Twelve heads times 32 channels times sixteen
temporal FMA lanes times four bytes retains the existing 24 KiB accumulator
payload. A single-token tail retains the six-head, 64-channel path. Scalar and
baseline NEON preserve each output channel's temporal FMA order, padding, native
reduction tree, globally normalized F16 probabilities and separate gate
arithmetic. No retained score reservation or conservative admission changes.

For a 256-channel KV head and a paired prefix with `P` common cached positions,
the previous separate paths converted `256 * (2 * P + 1)` F16 values. Paired
values convert `256 * (P + 1)`: the common prefix is shared, and the first
token's cached position belongs only to the second. There are eight 32-channel
iterator sweeps instead of eight 64-channel sweeps across the two tokens. These
are source work counts; code generation and matched whole-request timing must
establish whether this improves latency.

Malformed row geometry rejects before output mutation. Both rows' reductions
within a channel tile validate before either tile output is published. Compared
with separate per-token value traversal, a bad second row may therefore reject
before the first row's provisional value tile is published. Tiles and head
groups remain private graph scratch; complete-request checkpoint rollback stays
the committed-state/publication boundary. A future cache row remains unreachable.

Independent controls compare every scalar/dispatched channel bit against the
single-head native-dot oracle, including F32/F16 asymmetry, negative zero,
subnormals, temporal padding, partial tiles and page crossings. Cache visitation
is counted, and malformed extents, short/long iterators and a late second-token
failure must preserve both outputs. Production prefill controls retain worker,
causal-prefix, cancellation and complete-state restore coverage. Actual-model
logit/state/adapter parity, ARM64 codegen/spills and matched request benefit stay
pending before release or Runner repinning.
