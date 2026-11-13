# Validated ternary codecs

[`bonsai-ternary`](Cargo.toml) provides PQ2/PTQ1 codecs and numerical utilities for
GGUF export and the Prism runtime. It has no dependencies, forbids unsafe code,
uses checked Release arithmetic and returns typed encoding errors.

PQ2 stores a little-endian FP16 scale and 128 two-bit symbols (`0 = -1`, `1 = 0`,
`2 = +1`). Native PTQ1 stores 26 packed trit bytes followed by the unchanged scale
in 28 bytes. Packing rejects reserved symbol 3 and zero/nonfinite weight scales;
finite negative/subnormal scale bits are preserved. Unpacking accepts only
canonical bytes: repacking must reproduce the input exactly.

`half_to_float` converts every binary16 pattern, preserving signed zero and NaN
payload bits. `Q8KBlock` preserves the native 292-byte activation ABI, including
group sums; the consuming kernel validates activation scales. This crate supplies
no inference kernel or model loader.

The codec contract is [Prism 73bfe6cc](https://github.com/PrismML-Eng/llama.cpp/tree/73bfe6cc485554822f80c473a3c0c4f25e80f41b);
[PRISM-NOTICE.txt](PRISM-NOTICE.txt) retains its MIT notice. Regressions cover packed
bytes/positions, all symbol bytes/nonzero counts, malformed encodings, binary16
edges and activation ABI. The [native C++ fixture](tests/fixtures/README.md) adds
161 captured codec cases and 26 binary16 cases.

Independent base-3 expectations check all 5,994 lane words against exact symbol
positions, so a matching encoder/decoder permutation cannot pass by round trip
alone. The [native probe](../codec-probe/README.md) compares
production Rust packing against the pinned native PQ2/PTQ1 decoders without
adding unsafe code to this crate.

Use the [pinned compiler and build setup](../../docs/build-and-validation.md).
One focused parity check, from an external working directory with absolute paths:

```text
cargo +nightly-2026-11-09 test --locked --offline --release --manifest-path <repo>/tools/ternary/Cargo.toml --target-dir <temp>/build --test cpp_reference cpp_reference_ptq_parity -- --exact
```

Full suites and benchmarks follow the [validation policy](../../docs/build-and-validation.md).
