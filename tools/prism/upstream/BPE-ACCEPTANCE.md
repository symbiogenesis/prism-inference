# BPE roundtrip acceptance inputs

The complete safe Rust port of `tests/test-tokenizer-1-bpe.cpp` accepts a
vocabulary and the original optional `--ignore-merges` flag. The eight example
invocations in the pinned CMake file are commented out. Default acceptance uses
the original `aquila`, `gpt-2`, `gpt-neox`, `mpt`, `refact`, and `starcoder` inputs
with the false flag, plus the original `llama-bpe` input with the true flag.

The Llama-BPE run uses the explicitly hashed production correction in
[`patches/llama3-cleanup.patch`](patches/llama3-cleanup.patch). The pinned decoder
inherits `clean_spaces=true`, which removes legitimate spaces before punctuation.
For example, raw token 24268 is `202124` (` !$`); the unpatched decoder produces
`2124` (`!$`), which retokenizes as two tokens. This is a production decoding
defect, not an assertion that should be skipped.

Meta's canonical [Llama3 decoder](https://github.com/meta-llama/llama-models/blob/01dc8ce46fecf06b639598f715efbb4ab981fb4c/models/llama3/tokenizer.py#L174)
delegates to [tiktoken](https://github.com/openai/tiktoken/blob/fa8b65d062fb6a656ac3810c89efde4c8ab999e2/tiktoken/core.py#L275),
which decodes token bytes without punctuation cleanup. The official
[Transformers regression](https://github.com/huggingface/transformers/blob/bbb51c83c151c3285d7f0e63ea6a68165c61cc58/tests/models/llama/test_tokenization_llama.py#L57)
also verifies that Llama3 skips cleanup despite the shipped configuration flag.
The correction disables cleanup only for metadata aliases `llama3`, `llama-v3`
and `llama-bpe`; other families keep their configured cleanup behavior.
The manifest records the exact original source, corrected source and patch
hashes.

Every original predicate remains: vocabulary type and loading checks; all
vocabulary tokens; invalid-argument handling; the approved strict invalid-UTF8
piece skip; `tokens.size() <= 1` when requested; exact detokenized-byte equality;
both original Unicode exclusions; the U+2581 exemption; worker strides and shared
stop behavior. Successful full runs require 456,703 Unicode rows and 456,702
equality predicates per vocabulary. The true-flag receipt must contain a positive
actual count of cardinality predicates.

Falcon remains an explicit diagnostic with its actual failure status. Its 65,012
normal and 12 control tokens have no representation for byte 04, so the source's
universal Unicode equality assertion cannot pass for that original input.

```json
{"Kind":"tokenizer-bpe-roundtrip","Vocabulary":"falcon","IgnoreMerges":false}
```

Any of the eight original pinned BPE vocabularies and either flag remain selectable
through that diagnostic. It runs the same assertions with the real result; its
distinct diagnostic ID, canonical `SourceId`, `SourceCount:0`, and diagnostic
measure prevent it from counting as a source pass. Input hashes and exact caller
arguments remain in every receipt.

Qwen2 with the true flag is excluded: its
[Qwen2 pretokenizer](https://github.com/huggingface/transformers/blob/a30413b78feed68da5c486746f745db092bfdf9a/src/transformers/models/qwen2/tokenization_qwen2.py)
intentionally separates numeric codepoints; Qwen2 also leaves native
`ignore_merges` false. The source CLI flag enables a check rather than changing
that native configuration.
