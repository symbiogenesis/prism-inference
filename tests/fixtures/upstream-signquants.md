`upstream-signquants.json` contains 18 model-free correctness observations for
Prism `Q1_0` (type 41, 128 values/18 bytes) and `Q2_0` (type 42, 64 values/18 bytes).
Its reference encodings, decoded binary32 bytes, packed `Q8_0` activations and
generic dots come from unchanged functions extracted from PrismML-Eng/llama.cpp
revision `73bfe6cc485554822f80c473a3c0c4f25e80f41b`:

- `ggml/src/ggml-quants.c`: `quantize_row_q1_0_ref`, `quantize_row_q2_0_ref`,
  `quantize_row_q8_0_ref`, `dequantize_row_q1_0`, `dequantize_row_q2_0`.
- `ggml/src/ggml-cpu/quants.c`: `ggml_vec_dot_q1_0_q8_0_generic` and
  `ggml_vec_dot_q2_0_q8_0_generic`.
- `ggml/src/ggml-impl.h`: its scalar half-conversion functions.

The native probe was built with MSVC 14.51.36231 for Windows ARM64 using
`/O2 /fp:strict /std:c11`. The fixture retains the source and executable hashes.
Scenarios cover positive and negative zero, integer rounding ties, finite half
boundaries, tiny scales, and deterministic random rows. Native CPU encoders
delegate to the same reference functions; native importance values are unused
for these two formats.

Two additional records retain SHA-256 hashes of native decoded bytes and native
dot results for 4,096 blocks per codec. Each of the 16 payload byte positions
takes all 256 byte values. Eight scales cycle through positive/negative zero,
minimum positive/negative subnormal, positive/negative one, and positive/negative
maximum finite half. Packed activations include every signed byte, including
`-128`, and use independent positive/negative scales. The focused test rebuilds
these inputs and checks the hashes; no floating tolerance replaces byte equality.

The production decoder adaptation was informed by Camelid revision
`1829a7e73775eb829efe8fabf46a6680897a975b`, `src/tensor/mod.rs`,
`decode_q1_0_tensor` and `decode_q2_0_tensor` (MIT, Copyright (c) 2026 Tim Toole).
Prism's exact 64-element type-42 geometry is retained. Camelid's additional
128-element type-42 dialect is not admitted by these kernels.

These observations establish focused codec correctness. They contain no timing,
full-model, full-suite or hosted-qualification evidence.
