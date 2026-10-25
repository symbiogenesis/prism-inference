`row-quantizers.json` retains 28 focused production quantizer records captured
from the real pinned Prism common bridge. Inputs cover signed zero, maximum
ties, binary16/bfloat16 boundaries and bounded LCG values. Each has 256 values;
the six reference quantizers use `codec-quantize` in `reference` mode. Q8_K uses
the native CPU quantizer, which has no reference trait entry.

The file records the upstream commit, production bridge binary SHA-256 and
quantizer source SHA-256. Its SHA-256 is
`7b520199c64e045fb2b153df38fe488f3043557c04248727766bd2a6b8944d54`.
The bridge was an x64 Release build executed under Windows ARM64 emulation.
These are model-free codec correctness records; they contain no timings.

F32 encoding is a direct byte-preserving scalar copy. The fixture covers F16,
Q4_0, Q8_0, Q8_K, BF16, PQ2 and PTQ1. It does not qualify the other native
quantizers, full-model inference, or hardware performance.
