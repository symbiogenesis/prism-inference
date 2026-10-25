# IQ1 native codec evidence

The adjacent JSON was produced before executing the Rust implementation by a
standalone native probe containing unchanged production functions and the
unchanged 2,048-entry `iq1s_grid` from
[`PrismML-Eng/llama.cpp@73bfe6cc485554822f80c473a3c0c4f25e80f41b`](https://github.com/PrismML-Eng/llama.cpp/tree/73bfe6cc485554822f80c473a3c0c4f25e80f41b).
These are MIT sources; the repository retains the upstream license in
`PRISM-LICENSE.txt`. Source hashes and native compiler flags are in the JSON.

The probe calls `dequantize_row_iq1_s`, `dequantize_row_iq1_m`,
`ggml_vec_dot_iq1_s_q8_K_generic`, `ggml_vec_dot_iq1_m_q8_K_generic`, and
`quantize_row_q8_K_ref`. The wrapper only creates input bytes and serializes
production observations. It uses MSVC 14.51.36231 ARM64, `/O2 /fp:strict
/std:c11`, and Windows SDK 10.0.26100.0. It neither executes benchmark schedules
nor measures timings. The probe source hash is retained in the JSON.

Sixteen literal cases cover both types, signed zero and subnormal scales,
finite scale extremes, arbitrary packed activation integers and sums, and
two-block accumulation. Four cases additionally retain native Q8_K activation
encoding from supplied F32 inputs for production matvec/prepared-matvec checks.

Six SHA-256 records cover 157,696 independently decoded blocks and generic
packed dots: every payload byte position swept over 256 values (with finite
scale bits restored), every one of the 2,048 grid indices, and every one of the
63,488 finite binary16 scale patterns for each type. Native output bytes were
hashed externally; Rust reconstructs the supplied inputs and must match those
hashes. All 2,048 nonfinite scale patterns are rejected separately for each type.

IQ1_S native dots consume the supplied Q8_K block sums, including arbitrary
values that differ from recomputed integer sums. IQ1_M native dots ignore the
supplied sums and recompute local sums. The fixtures preserve this distinction.

This evidence qualifies these focused scalar codecs and packed dot contracts.
It does not qualify the complete upstream source suite, native SIMD arithmetic,
or performance. IQ1 reference and importance encoders remain unavailable.
