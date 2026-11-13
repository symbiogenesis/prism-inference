# Upstream process adapter

`prism-upstream-bridge --input ABSOLUTE_PATH --output ABSOLUTE_PATH` calls the
production ggml library from the pinned Prism source. The Rust adapter serializes
production calls; the [safe Rust harness](README.md) owns inputs and assertions.
Exit zero means completion and a written response; errors use nonzero exit and
stderr. Discard any response from a failing process.

All integers and IEEE-754 binary32 floats are little-endian. Packed tensor bytes
use the pinned upstream ggml layout on a little-endian host. No implicit padding
is present in the protocol.

The request begins with the eight bytes `BSPREQ1\n`, then a `u32` request count.
Each request consists of six `u32` values: operation, K, M, N, packed byte count,
activation float count; followed by packed bytes and activation floats.

| Operation | Shape and payload | Returned floats |
| --- | --- | --- |
| 1: PQ2 matmul | K is a positive multiple of 128; M rows of K/128 34-byte PQ2 blocks; N columns of K f32 activations | M*N, column-major (`column*M + row`) |
| 2: PTQ1 decode | K=128, M is the block count, N=0; M 28-byte PTQ1 blocks; no activations | 128*M, block order |

The response begins with the eight bytes `BSPRES1\n`, then the matching `u32`
request count. Each response consists of a `u32` float count and that many f32
values, in request order. There is no other response data.

The adapter accepts 1 through 64 requests and at most 128 MiB of request data,
16 MiB packed bytes and 4,194,304 activation floats per request, 4,194,304 returned
floats per request and 16,777,216 returned floats overall. PQ2 dimensions are
bounded by K<=8192, M<=4096, N<=256 and K*M<=8,388,608. PTQ1 batches have at most
20,000 blocks. Unsupported operations, mismatched lengths, trailing data and
relative paths are rejected before execution. Each operation receives a separate
128 MiB ggml context and matmul uses four threads. Process timeout and scratch-file
lifecycle belong to the caller.

## Backend partition sessions

`prism-upstream-common-bridge --backend-session` accepts a sequence of framed
JSON messages on binary stdin and writes framed JSON replies to binary stdout.
Each frame starts with a four-byte little-endian byte count. Requests are bounded
by 128 MiB, replies by 16 MiB; zero-length, truncated and oversized frames fail.
Requests contain exactly `Sequence` and `Operation`; replies contain exactly
`Sequence` and `Result`. Sequence numbers start at one and increase by one.
Unknown fields, duplicate request keys and invalid sequences fail the process.

The Rust controller opens a geometry with `open` (`Directory`, `MaxFileBytes`),
sends batches through `actions` (`Actions`), and calls `close` before opening the
next geometry. Each geometry receives a fresh production state; closing destroys
all tensors, contexts, buffers and backends before acknowledging completion.
No handles or graph state survive between geometries. A partition accepts at
most 22,817 geometry scopes, each with at most 524,288 actions and the existing
4 GiB file bound. `shutdown` requires no open geometry and exits successfully.
Production actions and their observations use the existing backend schema.

Rust buffers each geometry's `actions.jsonl` and `observations.jsonl` in memory
up to 256 KiB per stream, then spills larger transcripts to disk. Successful
planner/execution cases discard these diagnostics only after controller and
session completion; returned errors and unwinding retain them for the existing
failure digest/retention policy. A hard process kill or abort can lose the
unflushed in-memory prefix. Default transport diagnostics retain transcripts.
Partition stderr and process-exit receipts remain separate diagnostics. Framing,
count, I/O, deadline or process failures poison the session and abort further
geometries. The file/checkpoint transport remains available for equivalence checks
of raw observations, output bytes, input receipts, seeds and state isolation.
