# Checked ARM64 arithmetic and packed KV primitives

`bonsai-neon-dot` provides isolated, audited ARMv8.2-A `dotprod` (`vdotq_s32` /
`SDOT`) and `i8mm` (`vmmlaq_s32` / `SMMLA` `2x2` row-by-token tiles) SIMD
dot-product primitives, plus non-widening `u8x16` base-3 division-by-multiplication
`PTQ1_0` trit unpacking (`decode_ptq1_trits_i8`).

The crate also provides exact binary16 widening, ordered BF16 row reductions
through `dot_bf16_rows`, and bounded packed rotated-scalar KV operations in
`kv_quant`. Floating-point paths retain their documented scale, reduction-order
and finite-value checks.

Public entry points use fixed arrays or checked borrowed slices. Optional
`dotprod` and `i8mm` paths require runtime capability checks (`has_dotprod`,
`has_i8mm`); baseline ARM64 and portable implementations preserve their separate
arithmetic contracts. Publishing the package does not qualify complete-model
behavior or establish measured Snapdragon performance.

Original code uses Apache-2.0. The included `LICENSE`, `NOTICE`,
`PRISM-LICENSE.txt` and `CAMELID-LICENSE.txt` preserve the repository's license
and attribution records.
