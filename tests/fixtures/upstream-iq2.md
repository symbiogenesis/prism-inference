`upstream-iq2.json` retains focused native correctness observations for
`IQ2_XXS` (type 16, 66 bytes), `IQ2_XS` (type 17, 74 bytes), and `IQ2_S`
(type 22, 82 bytes), each encoding 256 values per block.

The oracle executes unchanged functions extracted from PrismML-Eng/llama.cpp
revision `73bfe6cc485554822f80c473a3c0c4f25e80f41b` (MIT):

- `ggml/src/ggml-quants.c`: the three `dequantize_row_iq2_*` functions,
  `quantize_row_q8_K_ref`, and its `nearest_int` helper.
- `ggml/src/ggml-cpu/quants.c`: the three `ggml_vec_dot_iq2_*_q8_K_generic`
  functions.
- `ggml/src/ggml-common.h`: original codebooks, sign table, and sign mask.
- `ggml/src/ggml-impl.h`: original scalar half-conversion functions.

The standalone native probe was compiled for Windows ARM64 with MSVC
14.51.36231 using `/O2 /fp:strict /std:c11`. The fixture records hashes of
those four source files, the extracted probe, and its executable.

The 24 two-block records contain arbitrary packed payloads with positive and
negative zero, minimum subnormal, unit, and maximum finite half scales. They
retain native decoded binary32 bytes, native `Q8_K` activation bytes, the
floating activation input, and the exact native generic-dot binary32 result.

Three additional records retain SHA-256 hashes of native decoded bytes and
native dots for every payload byte position taking all 256 possible values:
16,384 `IQ2_XXS` blocks, 18,432 `IQ2_XS` blocks, and 20,480 `IQ2_S` blocks.
Together these exercise 14,155,776 decoded values and 55,296 single-block dots.
The test reconstructs the same input schedule independently of the decoder.
Weight scales cycle through the same eight finite half values. Activations
include every signed byte, including `-128`, with positive/negative zero,
minimum normal binary32, unit, and `123.125` scales. Their stored group sums
are deliberately unrelated to the activation bytes; native IQ2 dots do not
consume those sums.

The production codebooks are mechanically extracted from the same pinned
`ggml-common.h`. The seven-bit sign mapping is derived from even parity and
checked against the oracle's original 128-entry sign table through the full
payload domains. Native integer block accumulation and the final row-wide
`0.125` multiplication are preserved. No floating expansion substitutes for
the production packed dot. `IQ2_XXS/XS` importance encoders remain explicitly
unavailable; the separately qualified `IQ2_S` encoder is documented in
`upstream-iq-encoders.md`.

External Rust candidates were inspected but no external code was imported:
`https://github.com/InfiniTensor/gguf` at
`98f180387df864b9492eb4d262ed8d88c19d57de` contains IQ2 codec stubs;
`https://github.com/rexlunae/joshua` at
`a0ec8a21c0816512f808e96824e45aa16b64a38b` has a safe little-endian decoder
in `vendor/candle-core/src/quantized/iq2xxs.rs`, but its fallback dot expands
binary32 values and changes the required accumulation arithmetic.

These are model-free focused correctness observations, with no timings,
full-suite, full-model, or hosted-qualification claim.
