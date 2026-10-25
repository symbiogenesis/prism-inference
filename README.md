<div align="center">

# 🔷 Prism Inference

**A Rust CPU runtime for Prism-compatible language models.**

GGUF models · Packed quantization · Local inference

![Platform: Windows ARM64 and x64](https://img.shields.io/badge/Windows-ARM64_%26_x64-0078D4?style=flat-square)
![Status: early development](https://img.shields.io/badge/status-early_development-D99A28?style=flat-square)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-50855B?style=flat-square)](LICENSE)

[Download](https://github.com/symbiogenesis/prism-inference/releases) · [Documentation](docs/index.md) · [Roadmap](docs/roadmap.md)

</div>

---

Prism Inference runs language models on the CPU using Rust, with packed
quantized weights, GGUF loading, tokenization and streamed output. It links no
C/C++ inference library.

The first generation path supports the dense Qwen35 architecture used by
**Swift/Bonsai**. Broader Prism architectures and general model selection are
in scope and planned. Prism Inference is an independent implementation, separate
from PrismML's upstream runtime.

> [!IMPORTANT]
> **Early development.** Full-model numerical parity and device qualification
> remain open. See the [current status and roadmap](docs/roadmap.md) before
> choosing it for an application.

## At a glance

| | Current support |
| --- | --- |
| **Runtime** | Rust CPU inference with bounded model and conversation storage |
| **First model path** | Swift/Bonsai · dense Qwen35 · Prism GGUF |
| **Acceleration** | Scalar reference kernels; eligible Windows ARM64 NEON |
| **Windows bundles** | ARM64 and x64 executables, licenses and integrity receipts |

## Get started

[Download a release bundle](https://github.com/symbiogenesis/prism-inference/releases)
for your Windows architecture, extract it, and start with:

```powershell
.\prism-inference.exe --help
```

Bundles contain prebuilt executables, so running them needs no Rust build.
Models are separate downloads; the [CLI guide](docs/cli-and-workers.md) explains
model identity, input options and resource budgets. SwiftBonsai provides model
setup and a guarded launcher that consumes the prebuilt private server.

For a source installation, use the pinned Rust nightly. This command compiles
locally:

```powershell
cargo +nightly-2026-10-22 install prism-inference --locked
```

See [building and validation](docs/build-and-validation.md) for all workspace
tools, Cargo features and development checks.

## Included tools

| Executable | Purpose |
| --- | --- |
| `prism-inference` | Generate from text or token IDs; inspect tokenization |
| `bonsai-token-worker` | Persistent token counting and encoding over JSONL |
| `bonsai-swift-worker` | Swift model inference over JSONL |
| `prism-inference-server` | Private HTTP inference for an owning launcher |
| `bonsai-gguf` | GGUF inspection, PQ2 census and PTQ1 export |

## Documentation

| Guide | What you'll find |
| --- | --- |
| [Documentation index](docs/index.md) | Where to start for each task |
| [CLI and workers](docs/cli-and-workers.md) | Inputs, streaming, tokenization and service protocols |
| [Runtime contracts](docs/runtime-contracts.md) | Implemented behavior, memory, ownership and SIMD |
| [Building and validation](docs/build-and-validation.md) | Source builds, tests, CI and release bundles |
| [Roadmap](docs/roadmap.md) | Current evidence, broader model support and remaining work |

## License

Original code and documentation use the [Apache License 2.0](LICENSE).
Third-party material retains its terms, including Prism/ggml and Camelid MIT
attributions in [NOTICE](NOTICE). Model weights and adapters are separate
artifacts governed by their publishers' licenses.
