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
| Understand budgets and failure recovery | [Runtime contracts](runtime-contracts.md) |
| Configure opt-in packed projection workers | [Explicit row workers](parallel-rows.md) |
| Compile, check or package the workspace | [Building and validation](build-and-validation.md) |
| Run upstream source/backend and codec checks | [Runtime conformance](prism-upstream-tests.md) |
| Capture explicit rank-one adapter correctness evidence | [Adapter correctness path](rank-one-adapter.md) |
| Check support and remaining qualification | [Roadmap](roadmap.md) |

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
