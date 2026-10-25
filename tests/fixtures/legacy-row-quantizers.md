`legacy-row-quantizers.json` retains 16 native reference encodings for Q4_1,
Q5_0, Q5_1 and Q8_1. `weighted-legacy.json` retains 24 native weighted Q4_1,
Q5_0 and Q5_1 rows with ones and varied importance. Inputs cover signed zero,
rounding ties, scalar precision boundaries and bounded LCG values. Each row has
256 elements. The files retain production bridge and quantizer source hashes
at upstream revision `b87ca7225456b06b0975b8239c66b5aecf07e37c`.

The 36 weight rows additionally retain native decoded bytes and native packed
dot results using independently captured Q8_0 or Q8_1 activation bytes. Q8_1
has no native decoder or weight-dot trait. These model-free correctness records
were captured from an x64 Release production bridge under Windows ARM64
emulation. They contain no performance measurements.

SHA-256 is `085a94cd73be5fe410faa6149264a2f6e724bf7ff47412fec969ffaf2e8029dc`
for the reference records and
`b9fd8de01c5b7705e373bae311b899bbacec397d2e32db3fcdd7bd52e20a7e52` for the
weighted records.
