The Rust suite ports `tests/test-pq2-row-shapes.cpp` and
`tests/test-ptq1_0-element-map.cpp` from PrismML-Eng/llama.cpp revision
`b87ca7225456b06b0975b8239c66b5aecf07e37c`, under the MIT license retained in
`tools/prism/upstream/LICENSE`.

`pq2-inputs.bin` freezes the original 15 PQ2 cases' prequantized weight bytes and
activation floats. These came from the original upstream `std::mt19937` and
`std::normal_distribution<float>` generators: weights use seed `1234 + K` and
activations use seed `99`. The shape order is K = 128/256/384/512/640, M = 4,
and N = 1/4/16. No decoded weights, reference answers, or C++ outputs are stored.
Rust independently decodes the packed weights and computes every reference dot.

The input-only file has 167,652 bytes and SHA256
`a37e4e61333647ff528ac597e8bba75b945ac0255430a87332719f34c56b5ef1`.
It was obtained by stripping decoded weights and target outputs from the
261,492-byte original `BSPQ2V1\n` fixture whose SHA256 is
`279cac324e1382b93650dcfe3b4e2a1a80c346e7c481ba8ef966f47f6f7b1621`.
The retained bytes are unchanged. The input-only format is `BSPQ2I1\n`, a
little-endian u32 shape count, then for every shape u32 K/M/N/packed-byte-count,
the packed bytes, and K*N little-endian binary32 activations. Parsing rejects
wrong counts, extents, order, nonfinite activations, truncation and trailing data.

PTQ1 needs no binary fixture: Rust reproduces the original wrapping 32-bit LCG
from seed 12345 and keeps all 20,000 arbitrary 26-byte packed blocks. Unit FP16
scale allows the original integer assertions to target float production
dequantization. Rust compares both original independent traversals (the CPU
stages and the CUDA accessor's host-checkable element indexing) for all 2,560,000
positions, then requires the target decoder to match exactly. Rejected blocks
remain failures; they are never discarded or canonicalized.
