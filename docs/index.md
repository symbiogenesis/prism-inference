# Documentation

[Project overview](../README.md) · [Roadmap](roadmap.md)

## Use and integrate

| Task | Guide |
| --- | --- |
| Run the CLI or JSONL workers | [CLI and workers](cli-and-workers.md) |
| Build, use the Rust library or package binaries | [Building and validation](build-and-validation.md) |
| Integrate the private HTTP server | [Server contract](../server/README.md) |
| Understand state, budgets and failure recovery | [Runtime contracts](runtime-contracts.md) |
| Select Swift profiles, adapter scale or prefix reuse | [Swift sessions](swift-session-profiles.md) |

## Execution and validation

| Topic | Guide |
| --- | --- |
| CPU kernels and work/storage reductions | [Inference execution](inference-efficiency.md) |
| Packed projection worker pools | [Parallel rows](parallel-rows.md) |
| Prompt tiles and GDN chunks | [CPU tiled prefill](cpu-tiled-prefill.md) |
| Rotated compressed KV profiles | [Quantized KV](quantized-kv.md) |
| Rank-one adapter execution and correctness capture | [Rank-one adapters](rank-one-adapter.md) |
| Upstream numerical and codec checks | [Runtime conformance](prism-upstream-tests.md) |
| Source-bound CPU request comparison | [CPU bundle comparison](cpu-bundle-comparison.md) |

## Tools

See the [GGUF tool](../tools/gguf/README.md), [ternary codecs](../tools/ternary/README.md),
[native storage overlay](../tools/prism/README.md) and [Hexagon runtime](../tools/hexagon-runtime/README.md)
for their contracts and commands. Fixture provenance stays with the
[conformance inputs](../tests/upstream_prism/INPUTS.md); attribution is in [NOTICE](../NOTICE).
