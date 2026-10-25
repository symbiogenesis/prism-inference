The immutable sampler RNG record contains 20 correctness observations from the
real Release x64 production bridge, executing the pinned
`llama-sampler.cpp` implementation. It contains eight successive distribution
selections and twelve XTC filter sizes, including singleton draw consumption.
The source revision and production binary SHA-256 are retained in the record.
The input consists of equal zero logits, distribution seed 4242 with candidate
counts `[1,2,17,257,257,257,257,257]`, and XTC seed 12345 with five candidates,
probability 0.5, threshold 0.1 and minimum keep 1. Fresh candidate buffers are
used for every apply; the sampler RNG persists between applies.

File SHA-256: `5e728e604357e474898180cfb866bdba62102d95048ff8bf79a8fcea12938310`.
This is a focused Windows x64-emulated correctness reference. No benchmark was
executed and the capture makes no physical Snapdragon performance claim.
The original C++ test executables remain retired. Production sampling never
reads this record.
