# Documentation

[Project overview](../README.md) · [Download releases](https://github.com/symbiogenesis/prism-inference/releases)

Prism Inference supplies CPU inference tools and a Rust library. Its initial
generation path targets Swift/Bonsai; broader Prism models are planned.

## Start here

| Task | Guide |
| --- | --- |
| Run a downloaded executable | [CLI and workers](cli-and-workers.md) |
| Choose text, token IDs or tokenization-only mode | [Driver input](cli-and-workers.md#generate-from-text-or-token-ids) |
| Keep a tokenizer process alive | [Persistent token worker](cli-and-workers.md#persistent-token-worker) |
| Integrate a worker or the private HTTP server | [Swift worker and private server](cli-and-workers.md#swift-worker-and-private-http-server) |
| Add the Rust library to an application | [Library dependency and tokenizer example](build-and-validation.md#use-the-rust-library) |
| Understand budgets and failure recovery | [Runtime contracts](runtime-contracts.md) |
| Configure opt-in packed projection workers | [Explicit row workers](parallel-rows.md) |
| Review CPU work/storage reductions and qualification limits | [Inference efficiency candidate](inference-efficiency.md) |
| Prepare a source-bound ordinary CPU bundle comparison | [CPU bundle comparison](cpu-bundle-comparison.md) |
| Review scoped packed activation preparation | [PQ2 consumption layout](packed-pq2-consumption.md) |
| Select separately qualified CPU prompt tiles or GDN chunks | [CPU tiled prefill](cpu-tiled-prefill.md) |
| Evaluate explicit Snapdragon compressed KV profiles | [Rotated KV candidate](quantized-kv.md) |
| Qualify exact latest-state service reuse | [Latest-prefix service](latest-prefix-service.md) |
| Compile, check or package the workspace | [Building and validation](build-and-validation.md) |
| Run upstream source/backend and codec checks | [Runtime conformance](prism-upstream-tests.md) |
| Capture explicit rank-one adapter correctness evidence | [Adapter correctness path](rank-one-adapter.md) |
| Check support and remaining qualification | [Roadmap](roadmap.md) |
| Review the unreleased static-runtime and full K4/V4 candidate | [0.1.27 preparation](release-0.1.27.md) |
| Review the latest registry runtime and development ARM64 bundle | [0.1.26 delivery and runtime prerequisite](release-0.1.26.md) |

## Technical reference

The [runtime contracts](runtime-contracts.md) retain the implemented API behavior,
model verification, transactional checkpoints, memory accounting and SIMD
diagnostics. The [CLI guide](cli-and-workers.md) covers exact file/stream input,
generation controls, JSONL framing, client failure policy and tokenizer APIs.

The [build guide](build-and-validation.md) describes Cargo features, the pinned
compiler, local focused checks, hosted correctness gates and release receipts.
The [roadmap](roadmap.md) links the inherited research and qualification evidence.

The [GGUF tool](../tools/gguf/README.md), [ternary codecs](../tools/ternary/README.md)
and [Prism storage overlay](../tools/prism/README.md) retain their own reference
documentation. See [NOTICE](../NOTICE) for attribution and source revisions.
