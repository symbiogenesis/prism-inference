# Audited ARMv8.2-A NEON dot-product and PTQ1 unpacking primitives

`bonsai-neon-dot` provides isolated, audited ARMv8.2-A `dotprod` (`vdotq_s32` /
`SDOT`) and `i8mm` (`vmmlaq_s32` / `SMMLA` `2x2` row-by-token tiles) SIMD
dot-product primitives, plus non-widening `u8x16` base-3 division-by-multiplication
`PTQ1_0` trit unpacking (`decode_ptq1_trits_i8`).

All public entry points accept fixed-size array references, gate target
feature intrinsics behind runtime CPU capability checks (`has_dotprod`,
`has_i8mm`), and fall back cleanly to exact portable integer arithmetic when
`aarch64`, `dotprod`, or `i8mm` is unavailable.
