`upstream-iq3.json` retains independent native correctness observations for
Prism `IQ3_XXS` (type 18, 256 values/98 bytes) and `IQ3_S` (type 21,
256 values/110 bytes). Production decoding, integer codebooks and scalar dots
are ported directly from PrismML-Eng/llama.cpp revision
`73bfe6cc485554822f80c473a3c0c4f25e80f41b` (MIT; root NOTICE):

- `ggml/src/ggml-common.h`: original block layouts, `iq3xxs_grid`,
  `iq3s_grid`, `ksigns_iq2xs` and `kmask_iq2xs`.
- `ggml/src/ggml-quants.c`: unchanged `dequantize_row_iq3_xxs` and
  `dequantize_row_iq3_s`.
- `ggml/src/ggml-cpu/quants.c`: unchanged
  `ggml_vec_dot_iq3_xxs_q8_K_generic` and `ggml_vec_dot_iq3_s_q8_K_generic`.
- `ggml/src/ggml-impl.h`: original scalar half conversion functions.

The native probe uses MSVC 14.51.36231 for Windows ARM64 with
`/O2 /fp:strict /std:c11`. Source/header, generated probe and executable hashes
are recorded in the fixture. Its sixteen cases each exercise three consecutive
blocks, preserving native floating accumulation order. Decoded float bytes and
native packed `Q8_K` dot bytes are compared without a numerical tolerance.

The arbitrary-byte records cover every byte value at every payload position:
24,576 IQ3_XXS blocks and 27,648 IQ3_S blocks, totaling 13,369,344 decoded
values and 52,224 independent native dot results. Eight half scales cycle
through both signed zeroes, positive/negative minimum subnormals, one and maximum
finite half. Activation bytes include all signed values, including -128;
their F32 scales vary independently. Deliberately unrelated `Q8_K.bsums`
confirm these native IQ3 dots do not consume that field. The original 128-entry
sign table was separately verified against the production parity derivation.

Additional focused regressions cover malformed blocks, partial rows, invalid
stored scales/activation scales, direct/prepared matvec, CPU graph routing and
the explicit unavailable encoder/importance-encoder results. No quantizer,
timing, full-model, full-suite or hosted qualification is claimed.
