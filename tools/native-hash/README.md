# Bounded SHA-256 streams

`bonsai-native-hash` hashes an exact-length byte stream using a caller-owned read
buffer and a cancellation checkpoint before every read. Read, length, checkpoint
and hash failures return no digest. It retains no payload and owns no file paths.

Windows uses the system CNG SHA-256 provider, whose CPU dispatch avoids requiring
a global target-feature flag. The small private native module owns one
thread-confined hash handle, borrows bytes only during synchronous calls and
destroys the handle on success, read failure, cancellation and finalization
failure. Other targets use the portable RustCrypto `sha2` baseline. CNG failures
fail closed rather than publishing an unchecked or partial digest.

The GGUF loader uses this stream for exact artifact/range verification while
retaining its original file handle, Windows sharing protection, 64 KiB scratch
and cursor restoration. Header parsing and payload ownership remain in the
unsafe-free GGUF crate. Mandatory full-file verification still reads all model
bytes; this change establishes no measured startup or model-performance gain.

Independent vectors, binary block boundaries, interruption, read errors,
cancellation, short/long input and bounded-range tests run in hosted CI alongside
the GGUF final-byte-corruption regression. Publish this dependency before any
registry release of a GGUF package that requires it.
