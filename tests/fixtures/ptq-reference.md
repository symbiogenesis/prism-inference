# Native C++ codec fixture

`ptq-reference.bin` preserves native ARM64 C++ PQ2/PTQ bytes and binary16
conversions captured at revision `d7bab354d0848215f0a9bb4d6699b218cc1e90f7`.
The codec contract is [Prism 73bfe6cc](https://github.com/PrismML-Eng/llama.cpp/tree/73bfe6cc485554822f80c473a3c0c4f25e80f41b).

The 161 cases contain 169 rows at widths 2, 40, 48 and 136 blocks, every nonzero
count from 0 through 128, and positive/negative single-weight encoding boundaries.
Scales include negatives, FP16 subnormals and extrema. Another table contains
26 binary16 cases covering signed zero, infinities and NaN payloads.

The fixture is 62,964 bytes with SHA-256
`de86d0ff0e31037635ae5f607bcb68d4e62ee7cb5d1b5db5e739ef900114fd79`.
It retains codec fields from the original fixture in Git history; experimental
storage/arithmetic fields are omitted. The capture used strict floating-point
ARM64 MSVC Release mode and required native PTQ decoding to reproduce PQ2 bytes.

All multibyte values are little endian:

1. Eight-byte magic `PTPAR001`.
2. `u32` conversion count, then `(u16 FP16 input, u32 FP32 result bits)` records.
3. `u32` case count, then `u32 rows`, `u32 blocks_per_row`, row-major PQ2 blocks
   (34 bytes each) and row-major PTQ blocks (28 bytes each) for each case.

`cpp_reference_ptq_parity` checks bounds, dimensions, counts, widths, end of input,
exact packed/decoded bytes and half-conversion bits. This is correctness evidence,
not an inference-performance result.
