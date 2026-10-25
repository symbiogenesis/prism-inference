# Pinned native codec probe

`bonsai-codec-probe` checks production `bonsai-ternary` PQ2/PTQ1 conversion
against both native dequantizers from the released Windows ARM64 Prism runtime
`73bfe6cc485554822f80c473a3c0c4f25e80f41b`, without model weights. The preserved
6,670 blocks cover 853,760 FP32 values and the original fixed random corpus.
The independent bit oracle and native ABI boundary moved unchanged from Runner.

Runtime conformance CI owns the complete probe. Its wrapper accepts an already
downloaded archive; `native.lock.json` pins all 12,412,126 bytes and SHA-256:

```powershell
./tests/Test-PrismCodec.ps1 -RuntimeArchive C:\Evidence\runtime.zip `
  -ProbeToolPath C:\Build\release\bonsai-codec-probe.exe `
  -EvidenceDirectory C:\Evidence\fresh-codec
```

The wrapper verifies the entire archive, bounds expansion, compares every
extracted file with its ZIP entry, verifies all native PE headers and retains
read handles denying mutation through child completion. It uses neither the
Runner installer nor its maintenance CLI. Version and codec children run in
separate one-process kill-on-close Windows Jobs via the already published
`bonsai-windows-process =0.1.0`, each capped to 1 GiB commit and 30 seconds, with
parent-process lifetime checks, inherited environment and an explicit external
native working directory. Evidence includes exact package file hashes, version,
ABI, corpus counts, stdout/stderr and peak Job memory.

Unsafe code stays confined to `src/native.rs`: restricted absolute DLL loading,
export ABI conversion, static traits and bounded decoders. The reviewed ABI is
56-byte `ggml_type_traits`, `to_float` at byte 40, PQ2 type 142 (34 bytes) and PTQ1
type 143 (28 bytes), each with 128 values. Inputs and results remain independently
anchored by frozen FP16/FP32 pairs and integer ternary arithmetic; no production
`half_to_float` oracle or round-trip-only expected symbols are used.

Passing proves synthetic format consumption and exact symbol/scale packing.
Activation quantization, model logits and performance remain unqualified.