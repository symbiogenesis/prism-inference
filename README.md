# Prism Inference

A Rust CPU runtime for Prism-compatible GGUF language models, with packed
quantized weights, tokenization and streamed output. The first generation path
supports the dense Qwen35 architecture used by Swift/Bonsai. The runtime is an
independent implementation and links no C/C++ inference library.

**Experimental:** broader model support, full-model numerical qualification and
device acceptance remain open. See the [roadmap](docs/roadmap.md).

## Get started

Follow the [build guide](docs/build-and-validation.md#build-a-checkout), then run:

```powershell
.\prism-inference.exe --help
```

Models are separate downloads. The [CLI guide](docs/cli-and-workers.md) explains
artifact verification, input formats and resource budgets. For Rust integration,
see the [library example](docs/build-and-validation.md#use-the-rust-library).

## Executables

| Executable | Purpose |
| --- | --- |
| `prism-inference` | Generate from text or token IDs; inspect tokenization |
| `bonsai-token-worker` | Persistent token counting and encoding over JSONL |
| `bonsai-swift-worker` | Swift model inference over JSONL |
| `prism-inference-server` | Private HTTP inference for an owning launcher |
| `bonsai-gguf` | GGUF inspection, PQ2 census and PTQ1 export |

## Documentation

- [Documentation index](docs/index.md): task and reference guides.
- [Runtime contracts](docs/runtime-contracts.md): API behavior, memory and ownership.
- [Building and validation](docs/build-and-validation.md): compiler, checks and packaging.
- [Roadmap](docs/roadmap.md): remaining work and qualification limits.

## License

Original code and documentation use the [Apache License 2.0](LICENSE).
Third-party terms and attributions are listed in [NOTICE](NOTICE). Model weights
and adapters are governed by their publishers' licenses.
