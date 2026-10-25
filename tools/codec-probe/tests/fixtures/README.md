# Frozen native codec corpus

`random-pq2.bin` contains the original 256 random PQ2 blocks from
`tests/PrismCodecProbe.cs` in SwiftBonsai2Runner at repository revision
`9dd5a81f2af11d399d92d23a02f3a3027d91e925`. The file is 8,704 bytes,
SHA-256 `0f0405ee9d335b5cb42df4c2ac95c1adb6bf272fb6380f1c9ba890e7c725b64f`.

It was captured with .NET `System.Random(7310787)` on Windows ARM64. In sample
order 0 through 255, each 34-byte PQ2 block starts with the little-endian FP16
scale `0x3c00` for even samples or `0xb555` for odd samples. Positions 0 through
127 receive successive `Random.Next(3)` values, stored low two bits first in
the 32 symbol bytes. The first 24 generated symbols are:

```text
2 2 0 1 0 0 2 0 1 0 0 1 2 2 0 0 1 0 1 2 0 0 1 1
```

This binary freezes the existing inputs without implementing a .NET random
generator in Rust. Expected PTQ bytes and native FP32 values are not generated
by this fixture: the probe checks them independently against source symbols,
frozen IEEE 754 scale-bit pairs, and both native dequantizers.
