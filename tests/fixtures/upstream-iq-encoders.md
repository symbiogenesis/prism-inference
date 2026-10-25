`upstream-iq-encoders.json` contains 144 focused native observations for
`IQ3_XXS` (type 18), `IQ3_S` (21), and `IQ2_S` (22), using Prism revision
`b87ca7225456b06b0975b8239c66b5aecf07e37c`. Each type uses twelve unchanged
input rows with four importance modes: absent, the supplied nonnegative vector,
all zero, and one unit entry per eight elements. Repeated rows are stored once;
the complete supplied input and importance bytes remain in `Inputs`.

Rows include positive/negative zero, positive/negative constants, alternating
and odd-parity signs, the original `1e-8` group cutoffs, zero groups followed by
nonzero groups, sparse impulses, and two-block pseudorandom inputs with different
scales. Every expected output byte comes from actual native production calls.

The oracle compiled the unchanged, hash-verified `ggml/src/ggml-quants.c` for
Windows x64 with MSVC 14.51.36231, `/O2 /fp:strict /std:c11`, then called
`ggml_quantize_chunk` using the pinned ggml-base library for runtime dispatch.
The fixture records source/archive, probe and executable hashes. These are
scalar fitting observations; they establish neither CPU quantizer availability
nor target-device performance. Native CPU encoder entries remain unavailable
for all three types. No benchmark or full suite was run locally.

The native integer fitting grids are reconstructed from the existing exact
decode codebooks. All 256/512/1,024 original native compact-grid identities were
independently compared against that reconstruction. Geometric neighbour shells
are sorted by distance and original grid index; weighted errors use the original
square-root weights, scale trials and refit decisions. `IQ3_XXS` retains its
even-sign-parity correction, including selection of the least weighted lane.

The pinned `IQ3_S` encoder advances low-code/sign pointers only for nonzero
groups, while high bits/scales retain original group indices. These defined
native bytes are preserved and covered by mixed-zero-group observations.
No upstream production correction is implied. All-zero importance also retains
the actual native output instead of imposing a zero-output shortcut.
