# Safe Rust assertions against pinned Prism

The `upstream_prism` example owns inputs, references and assertions in safe Rust.
The same tests target an independent reference and this repository's Rust runtime.
References use pinned native Prism except explicitly migrated model-free families.
Both results are retained; reference failure
fails the final gate. The source pin is
`PrismML-Eng/llama.cpp@b87ca7225456b06b0975b8239c66b5aecf07e37c`.

The [conformance guide](../../../docs/prism-upstream-tests.md) is the canonical
source inventory, assertion scope and acceptance policy. The Apple wired-memory
source is an explicit exclusion from Windows execution. Unsupported Rust APIs
remain failures rather than invented equivalents. Upstream MIT attribution is
in [LICENSE](LICENSE).

## Bridge boundary

The Rust PQ2/PTQ1 adapter in `bridge/` and separate native `common/` adapter contain
serialization, input validation and production calls. Generators, expected values,
assertions and tolerances belong to the safe Rust harness. Protocols are defined
in [PROTOCOL.md](PROTOCOL.md) and [common/PROTOCOL.md](common/PROTOCOL.md).

`reference-patches.lock.json` binds production saver and Llama 3 corrections by
original/corrected/patch hashes. Only verified external translation units are
substituted; original archives and test predicates stay intact. Saver correction
preserves DSpark metadata and sixteen tensors. Llama 3 correction disables inherited
punctuation cleanup for its three canonical aliases; see
[BPE acceptance](BPE-ACCEPTANCE.md). Receipts identify the corrected reference.

Input locks retain independent corpora, immutable publisher revisions, bounded
GGUF prefixes and HTTP range/size/hash checks. Expected schemas never generate
header inputs. Assets remain outside Git. The vector API runs once per original
activation column and compares every `M*N` result; this establishes no batched
execution or throughput equivalence.

## Build and run

Configure this directory against a verified source extraction with
`-DPRISM_SOURCE_DIR=<absolute upstream path>`, then build CMake target
`prism-upstream-bridge`. `RustBridge.cmake` uses the pinned workspace compiler and
production ggml libraries through a reviewed C ABI, without dependency downloads.
`PRISM_RUST_TOOLCHAIN` selects another explicitly installed compiler. Windows Rust
target and CMake architecture must agree. Build `common/` separately for its API.

```text
cargo run -p prism-inference --example upstream_prism -- \
  --target cpp --case test-pq2-row-shapes \
  --bridge <absolute native bridge executable> \
  --evidence-directory <fresh absolute evidence directory> \
  --output <absolute report path>
```

Use `--target rust` without bridge options for the runtime arm. PTQ1's selector is
`test-ptq1_0-element-map`; Unicode uses `--common-bridge`. The supervising
`tests/Test-PrismUpstream.ps1` retains required requests/responses and checks all
reference results in the final gate. GGML adapter use also requires
`tests/Test-PrismGgmlBridge.ps1` protocol/resource rejection checks; optional
`-ReferenceBridge` compares adapters independently.

Reports use `SchemaVersion: 1`, `UpstreamCommit`, `Target` and `Cases` with
`Id`, `Status`, `Detail`. Conformance failures are gate data; infrastructure
failures exit nonzero. Full catalogs, timed sources and all benchmarks are CI-only.
See [building and validation](../../../docs/build-and-validation.md) for execution
policy and the workflow inventory.

## Independent Rust observation provider

`--target rust-reference --case test-mtp-catchup-batch` selects the independent
mathematical MTP bounds provider. It runs the same six original ordered assertions
through the shared source adapter. The provider reserves nonnegative anchors from
the available batch and compares catch-up demand with the remainder; it calls
neither production implementation and performs no fixture lookup. Signed boundary
controls include negative inputs and the complete `i32` extrema.

The implementation binds reviewed upstream revision
`b87ca7225456b06b0975b8239c66b5aecf07e37c`:

| Source | Bytes | SHA-256 |
| --- | ---: | --- |
| [common/speculative.cpp](https://github.com/PrismML-Eng/llama.cpp/blob/b87ca7225456b06b0975b8239c66b5aecf07e37c/common/speculative.cpp) | 183,232 | `c8cad0219bc9c8c11c04574d3020876646835966412cc9e39eaae5aca805d92d` |
| [tests/test-mtp-catchup-batch.cpp](https://github.com/PrismML-Eng/llama.cpp/blob/b87ca7225456b06b0975b8239c66b5aecf07e37c/tests/test-mtp-catchup-batch.cpp) | 812 | `48d136ab03706c888eb393a771a6e1c7ec0c4ac0be1d7581d58a35bb57f1aae0` |

Retain the upstream MIT terms in [LICENSE](LICENSE). Source identities are reviewed
source observations, not captured native output. Reports identify target
`rust-reference` and `ObservationProvider.Kind: independent-rust-reference`,
with `NativeReferenceReplay`, `CompleteSourceGate` and `NativeRetirementQualified`
false. The selector also admits the explicitly migrated reasoning-budget,
concurrent logger, MTMD geometry and public chunk sources below; it rejects other sources,
full catalogs, benchmarks, diagnostics
and native bridge options. The ordinary owning `cpp` reference arm also uses
this provider for the MTP source, preserving all six source predicates and the
protected green ID. Its native include and dispatch are retired; the Rust
production arm remains unchanged. Reports and per-case details identify the
independent provider without claiming native replay. This provider does not
replace the complete owning reference gate or authorize deleting the common
dispatcher. Backend geometries,
source inventories, approved green IDs and separate performance schedules remain
required.

## Retired reasoning-budget native provider

`--target rust-reference --case test-reasoning-budget` runs all 71 original
semantic conditions through the unchanged source adapter against
`tests/upstream_prism/reference_reasoning.rs`. Its delimiter recognizer tracks
active prefix positions and its countdown derives from accepted-event ordinals.
It calls neither the production sampler nor native libraries and reads no
expected-answer fixtures. The assertions retain their original test source
identity, eight basic sequences, mid-count/mid-force clone, manual force/null,
end match/rearm/overlap/null and fifteen UTF-8 boundaries. Candidate and probability
bits, selected index and sorted flag survive unchanged except the specified
negative-infinity forcing mask.

The ordinary owning reference arm's historic `--target cpp` selector now uses
this independent provider for that same source ID. Its case `Detail` records the
provider and `nativeReferenceReplay=false`, so existing baseline/fan-in records
retain that distinction; the top-level report also identifies the migrated
provider. All required reference green IDs remain protected, including this ID.
The inventory's `LegacyReferenceRequired` field continues to mean the protected
reference-result obligation, without requiring the retired implementation.

`common/reasoning_budget.hpp`, its operative include/dispatch and its CMake edge
are removed. Direct native requests for these operations reject. Complete owning
hosted acceptance and the remaining C++ files are separate gates; this removal
does not claim complete native retirement, model/codec parity or device behavior.

## Retired concurrent logger native provider

`--target rust-reference --case test-log` runs the unchanged complete Windows
source schedule through `tests/upstream_prism/reference_logging.rs`. Eight
producers issue 1,000 messages each with the original per-thread Windows CRT
random severity and timestamp/prefix changes. They feed a bounded channel to
a separate owner that writes, flushes and destroys its sink. The production Rust
arm independently exercises its shared-mutex logger. All producers and the
consumer join before success; writer failures remain sticky while pending
channel messages drain. The original source's explicit destruction addresses
Windows thread-startup/DLL-teardown hangs. It asserts no output contents.

The reviewed assertion identity remains `tests/test-log.cpp` at upstream commit
`b87ca7225456b06b0975b8239c66b5aecf07e37c`, SHA-256
`f668c6d6987b0e8888a3823c0425971356c16ca77b44819236fb00c30e4b60c1`.
The existing source ID, required reference green ID and MIT attribution remain.
The ordinary `cpp` reference selector uses this separate Rust provider; reports
and case details name it with `NativeReferenceReplay: false`. No content parity,
fresh native execution, whole-source gate or full-retirement verdict is implied.

`common/concurrency.hpp` and its native include, dispatch and build edges are
removed. Direct native logger requests reject. Complete owning hosted acceptance
and the remaining C++ files are separate gates; backend geometry and separate
performance schedules are unchanged.

## Retired MTMD tiling and temporal grouping provider

`--target rust-reference --case test-mtmd-impl` executes all 29 original source
assertions: five LFM2 tiling decisions, twelve group-size results and twelve
independent remaining-bitmap counts. The original case order, five geometries
(including the 720 tie), twelve text/image/video/audio sequences, source ID and
immutable assertion corpus are unchanged.

`reference_mtmd.rs` independently minimizes distance on a bounded integer lattice,
preferring even indices for ties. It applies the original patch-size minimum and
strict two-times token-area threshold. Temporal grouping partitions original
equal-frame runs into pairs and separately counts surviving original bitmap
positions; it does not call production's quotient/remainder tiler or mutate the
production media vector. Audio, plain images, dimensions and intervening text
retain their original group boundaries. Whole-request typed bounds and owned
metadata replace native temporary bitmap pools; rejected input produces no partial
observation and subsequent requests use fresh state.

Provider identities bind upstream commit
`b87ca7225456b06b0975b8239c66b5aecf07e37c`, the complete source test SHA-256
`b6cffd8cbe193c590cb25b98e480f5db6615f8da3ef7e732ca85b19a2345388b` and the
original `mtmd.cpp`, `mtmd-image.cpp`, `mtmd-image.h` and `clip-model.h` identities.
The upstream MIT terms remain in `LICENSE`. The ordinary `cpp` reference arm
uses this independent provider and discloses `NativeReferenceReplay: false`.

`common/mtmd.hpp` and its native include/dispatch/CMake edge are removed; direct
native requests reject. Complete owning hosted acceptance and the remaining
34 C++ files stay separate, with catalog, required-green-ID and complete backend
and performance schedules unchanged. These metadata controls supply no vision or
audio-model inference, native arithmetic replay or device-performance verdict.

## Retired public MTMD chunk native provider

`--target rust-reference --case test-mtmd-c-api` runs all 46 original source
conditions through the unchanged assertion schedule against
`tests/upstream_prism/reference_mtmd_chunks.rs`. The ordinary `cpp` reference
selector uses the same independent provider. It constructs the source-defined
five-token text and 4x4 image with sixteen placeholder frames using flat wire
field recipes, then decodes with a bounded cursor. It calls neither production
Rust chunk APIs nor a native library. The reference supports the original
factory and all states reachable through its bounded chunk transport; it does
not claim general model/media preprocessing or allocation-failure equivalence.

Default `image_marker` remains null. Exact 42/226-byte cache lengths, all 261
aggregate tokens, ordered text IDs, image identity/position metadata, tiny saves,
every truncated prefix, successful roundtrips, trailing capacity and borrowed/
owned/null-handle lifetimes have focused controls. Pixel buffers are absent from
placeholder metadata as in upstream. The original required green ID and all
source predicates remain protected; backend geometries and separate performance
schedules are unchanged.

All reviewed sources use upstream commit
`b87ca7225456b06b0975b8239c66b5aecf07e37c`; MIT terms remain in [LICENSE](LICENSE).

| Source | Bytes | SHA-256 |
| --- | ---: | --- |
| `tests/test-mtmd-c-api.c` | 5,735 | `bdeb47199c1bc878566c7b47dd4905e6b82fb42586ad92aabc6ee27e1647d884` |
| `tools/mtmd/mtmd.cpp` | 103,968 | `9300de0676f5a3100ab67b0194e4999a57b217d12ef5c0c933069d7bf9592925` |
| `tools/mtmd/mtmd-helper.cpp` | 34,595 | `de3eb3ae5d261635051bc50d3dae89f5ff8420e06ab616dedfca2d7348ea4eaf` |
| `tools/mtmd/clip-impl.h` | 43,760 | `23060c1dea9dec19a722d1ff7f5639996826a49be6ab5446aec5b0a7e0d6c335` |

Reports identify the independent provider and `nativeReferenceReplay=false`.
`common/mtmd_chunks.hpp`, its operative include/dispatch and CMake edge are
removed. Complete owning hosted acceptance and remaining native families stay
separate; source-derived controls supply no fresh native execution, full-source
gate, model/codec parity, device behavior or complete retirement verdict.

## Retired grammar-parser native provider

`--target rust-reference --case test-grammar-parser` runs every original grammar
through `tests/upstream_prism/reference_grammar_parser.rs`. A character lexer
produces a token stream, an owned syntax tree captures alternatives/groups and
postfix repetition, and a separate pass allocates encounter-order symbols and
lowers instructions. Production Rust independently emits instructions while
scanning bytes. The reference imports neither production grammar code nor the
expected-answer corpus.

The original 23 variants, including six rejections, preserve all 274 literal
source assertions: exact lexical symbol order/encounter IDs, generated-rule and
flattened instruction order, character classes/ranges, finite and unbounded
repetition, explicit TOKEN/TOKEN_NOT and the pinned expansion threshold. Invalid
grammar returns no rules; each operation starts with fresh state. Additional
literal Unicode/escape/group/repetition and typed-request controls exercise the
independent generic protocol without supplying expected observations from the
production parser.

The assertion identity remains `tests/test-grammar-parser.cpp`, upstream commit
`b87ca7225456b06b0975b8239c66b5aecf07e37c`, SHA-256
`4c3f3b8ba3f4aa426b750815c7420c5af88672037c98f324168038dedd913ece`.
Its immutable corpus, source/required green IDs and MIT attribution remain.
The ordinary `cpp` reference selector uses the independent Rust provider and
discloses it in case details/provider identities with native replay false.

`common/grammar_parser.hpp`, its include/dispatch and its CMake build edge are
removed. Selecting this source alone needs no native common bridge. Generic
grammar-parser calls from other source families use the same independent
provider; their other native operations and independent obligations remain.
This port supplies no hosted full-source, model, grammar-runtime/token-mask,
device or complete C++ retirement verdict. Backend geometry and performance
schedules are unchanged.

## Retired threadpool native provider

`--target rust-reference --case test-barrier` runs the `cpu-threadpool` schedule
through `tests/upstream_prism/reference_threadpool.rs` with scoped OS worker
threads, mutex/condvar epoch dispatch and per-step barrier synchronization across
active threads and independent slot plans, without calling `bonsai_prism_runtime`
or any native bridge. Timing modes retain CI-only timing separation rules, while
ordinary `cpp` and `rust-reference` selections disclose `nativeReferenceReplay=false`.

`common/threadpool.hpp`, its include/dispatch and its CMake build edge are removed
(leaving 34 checked-in C++ files across the bridge snapshot). Selecting
`test-barrier` alone needs no native common bridge.

## Retired batch-allocator native provider

`--target rust-reference --case test-batch-alloc` runs all 26 cases and 198
original assertions through `tests/upstream_prism/reference_batch_allocator.rs`
using independent `SeqSet` bitsets, sequence-order checks, `KeepTail`
prefix/single-sequence fallback and `ubatch` snapshot serialization, without
calling `bonsai_prism_runtime` or any native bridge. Ordinary `cpp` and
`rust-reference` selections disclose `nativeReferenceReplay=false`.

`common/batch_allocator.hpp`, its include/dispatch and its CMake build edge are
removed (leaving 33 checked-in C++ files across the bridge snapshot). Selecting
`test-batch-alloc` alone needs no native common bridge.

## Retired allocation-graph native provider

`--target rust-reference --case test-alloc` runs all 13 scenarios and 468+
original assertions through `tests/upstream_prism/reference_allocation.rs` using
an independent post-order graph builder, view-aware liveness/in-place allocator
and sorted free-block chunk manager, without calling `bonsai_prism_runtime` or
any native bridge. Ordinary `cpp` and `rust-reference` selections disclose
`nativeReferenceReplay=false`.

`common/allocation.hpp`, its include/dispatch and its CMake build edge are
removed (leaving 32 checked-in C++ files across the bridge snapshot). Selecting
`test-alloc` alone needs no native common bridge.

## Retired quantization-policy native provider

`--target rust-reference --case test-quant-type-selection` runs all 363 pinned
model-ftype schema sections and per-tensor storage type assertions through
`tests/upstream_prism/reference_quantization.rs` using clean-room architecture
predicates, role classification and fallback selection, without calling
`bonsai_prism_runtime`, any native bridge or expected-answer fixtures (it reads
only the frozen GGUF model metadata header inputs and asserts against the pinned
`tests/snapshots/*.schema` suites as before). Ordinary `cpp` and
`rust-reference` selections disclose `nativeReferenceReplay=false`.

`common/quantization.hpp`, its include/dispatch and its CMake build edge are
removed (leaving 31 checked-in C++ files across the bridge snapshot). Selecting
`test-quant-type-selection` alone needs no native common bridge.

## Retired Unicode regex-split native provider

`--target rust-reference --case test-unicode` runs the original regex-split
assertion through `tests/upstream_prism/reference_unicode.rs` using Rust's
standard library and `fancy_regex`, without calling `bonsai_prism_runtime` or
any native bridge. It also supplies top-level `unicode-encode` and
`unicode-decode` operations for independent reference execution while leaving
persistent `--model-session` primitives in `common/vocab_session.hpp` untouched.
Ordinary `cpp` and `rust-reference` selections disclose
`nativeReferenceReplay=false`.

The `#include "unicode.h"` header include and top-level `unicode-*` dispatch
in `common/bridge.cpp` are removed (31 checked-in C++ files remain across the
bridge snapshot). Selecting `test-unicode` alone needs no native common bridge.

## Retired arena-graph native provider

`--target rust-reference --case test-moe-weighted-reduce` runs all 48 source
geometries/layouts across manual `W_IN_DST` / `X_IN_DST` aliased and plain
reference graphs through `tests/upstream_prism/reference_arena_graph.rs` using
an independent flat byte-arena evaluator, without calling `bonsai_prism_runtime`
or any native bridge. Ordinary `cpp` and `rust-reference` selections disclose
`nativeReferenceReplay=false`.

`common/arena_graph.hpp`, its include/dispatch and its CMake build edge are
removed (leaving 30 checked-in C++ files across the bridge snapshot). Selecting
`test-moe-weighted-reduce` alone needs no native common bridge.

## Retired tensor-graph native provider

`--target rust-reference` supports `test-col2im-1d`, `test-dfly-fusion` and
`test-rope` through `tests/upstream_prism/reference_tensor_graph.rs` using an
independent strided tensor-graph evaluator (covering `F32`/`F16`/`BF16` RNE
conversions, `col2im_1d`, `conv_transpose_1d`, `rms_norm`, `soft_max`, `mul_mat`,
`view_2d` and normal/NeoX/MRoPE/vision/IMRoPE rotations), without calling
`bonsai_prism_runtime` or any native bridge. Ordinary `cpp` and
`rust-reference` selections disclose `nativeReferenceReplay=false`.

`common/tensor_graph.hpp`, its include/dispatch and its CMake build edge are
removed (leaving 29 checked-in C++ files across the bridge snapshot). Selecting
`test-col2im-1d`, `test-dfly-fusion` or `test-rope` alone needs no native
common bridge.

## Retired GGUF, sampling, grammar and JSON-schema native providers

`--target rust-reference` supports `test-gguf` through
`tests/upstream_prism/reference_gguf.rs`, `test-sampling` through
`tests/upstream_prism/reference_sampling.rs`, `test-llama-grammar`,
`test-gbnf-validator` and `test-grammar-integration` through
`tests/upstream_prism/reference_grammar.rs`, and `test-json-schema-to-grammar`
(plus schema conversion in `test-grammar-integration`) through
`tests/upstream_prism/reference_json_schema.rs`, without calling
`bonsai_prism_runtime` or any native bridge. Ordinary `cpp` and
`rust-reference` selections disclose `nativeReferenceReplay=false`.

`common/gguf.hpp`, `common/sampling.hpp`, `common/grammar.hpp` and
`common/json_schema.hpp`, along with their bridge includes/dispatch and CMake
build edges, are removed (leaving 25 checked-in C++ files and 2 checked-in C
files, 27 native files total, across the bridge snapshot). Selecting those cases
needs no native common bridge.

## Retired public C ABI, DSpark metadata, argument parser, optimizer, codec and Jinja native providers

`--target rust-reference` and the ordinary `cpp` reference selector now route
seven additional source contracts through independent clean-room safe Rust
providers (disclosing `NativeReferenceReplay: false` without importing
`bonsai_prism_runtime`, calling native bridges, or reading precomputed expected
answer fixtures):

- `test-c` (`tests/test-c.c`, SHA-256
  `c973865e5e86263a71a1210003e80cffa6269572de31cfa647cc3d01b4bea99e`) via
  `tests/upstream_prism/reference_public_c_abi.rs`
- `test-dspark-logsnr-meta` (`tests/test-dspark-logsnr-meta.cpp`, SHA-256
  `2fef2521dbc0985a3aed6bbe50de5a54b7a75dc6c3c5933e04fc8ceb4c7e3112`) via
  `tests/upstream_prism/reference_dspark_logsnr.rs`
- `test-arg-parser` (`tests/test-arg-parser.cpp`, SHA-256
  `935b2cd08b63799a26fb804991380c541a45406c76dceb38589d78c8b28befc1`) via
  `tests/upstream_prism/reference_arg_parser.rs`
- `test-opt` (`tests/test-opt.cpp`, SHA-256
  `24af6c26293d328ad42820cc817ab891869c93ec744950e6505018cfd33d04bb`) via
  `tests/upstream_prism/reference_optimizer.rs`
- `test-quantize-fns` (`tests/test-quantize-fns.cpp`, SHA-256
  `830e197b34efc57030cef3b83449264f8b2d4db76121796ae962602cdf675d6b`) and
  `test-ptq1_0-cuda-dot` (`tests/test-ptq1_0-cuda-dot.cpp`, SHA-256
  `17d72e91495a7ffe4e726e90ebe99bf9d7b9acbe1d451a38c865352c3559092a`) via
  `tests/upstream_prism/reference_codec.rs` (with `tests/upstream_prism/engine.rs`
  routing `Target::RustReference` codec calls to `reference_provider`)
- `test-jinja` (`tests/test-jinja.cpp`, SHA-256
  `ef2da16176a8e914cbd54341e4b77f675a8f3b5b15a8bfd0c9a7a2305c30c8ab`) via
  `tests/upstream_prism/reference_jinja.rs`

All seven sources bind pinned upstream commit
`b87ca7225456b06b0975b8239c66b5aecf07e37c` (MIT terms retained in [LICENSE](LICENSE)).
`common/public_c_abi.c`, `common/public_c_abi.h`, `common/dspark.hpp`,
`common/arg_parser.hpp`, `common/optimizer.hpp`, `common/codec.hpp`, and
`common/jinja.hpp`, together with their bridge includes, dispatch branches and
CMake build edges, are removed (with `require`, `keys` and `integer` helpers
inlined into `common/codec_buffers.hpp`), leaving 20 checked-in C++ files and
zero checked-in C files (20 native files total) across the common bridge snapshot
and placing all 19 `RequiredReferencePassedIds` in `baseline.json` off
checked-in C++. Direct native requests for those retired operations reject;
complete Prism suite execution, native bridge compilation and owning hosted
acceptance remain pending.

## Retired model-resolution, codec-buffers, PEG and chat native providers

`--target rust-reference` and the ordinary `cpp` reference selector now route
fifteen additional source contracts across eight retired common bridge headers
through independent clean-room safe Rust providers (disclosing
`NativeReferenceReplay: false` without importing `bonsai_prism_runtime`,
calling native bridges, or reading precomputed expected-answer fixtures):

- `test-model-resolution` (`tests/test-model-resolution.cpp`, SHA-256
  `d596a6befca98035acfab4e71e72f50a698a11062fcac138735b3fe1aa9af593`) via
  `tests/upstream_prism/reference_model_resolution.rs`
- `test-quantize-perf` (`tests/test-quantize-perf.cpp`, SHA-256
  `09de1812b11836bd899a63dfa6b06927a2d0a012edc1e702a33fb51c95d63563`) and
  `test-quantize-stats` (`tests/test-quantize-stats.cpp`, SHA-256
  `1866b300a8966c3797740a5b4543236d92d1c7b8f2ca67829f453ffcd5b9707b`) via
  `tests/upstream_prism/reference_codec_buffers.rs`
- `test-peg-parser` (`tests/test-peg-parser.cpp`, SHA-256
  `52ebb967feae3aec877a1d3cb1c509b2598a9c799e60401a82253a5d4844038b`) and its
  six nested suites (`tests/peg-parser/test-basic.cpp`, SHA-256
  `24c5c31c65b2f5fc88919a4b5d06506ae644f8f0460c14c0134cd5439c5a7694`;
  `tests/peg-parser/test-gbnf-generation.cpp`, SHA-256
  `d14973bb98faaa87d1e17af30b76d413e2ac6541ab800c20eb55d48c8fa1952a`;
  `tests/peg-parser/test-json-parser.cpp`, SHA-256
  `2731298bf7f28047852e4f420243fc533bdb1408611257f47a5fff2adacb36fc`;
  `tests/peg-parser/test-json-serialization.cpp`, SHA-256
  `276f50faa0f1f50da7d5289b3e13110c26cf9120ff23bbdb4a6a8c12c384a028`;
  `tests/peg-parser/test-python-dict-parser.cpp`, SHA-256
  `ab6b20f141fed193bc93e810608c1b6800415d14ff832ad3ff84671594d18d4f`; and
  `tests/peg-parser/test-unicode.cpp`, SHA-256
  `5eda6c6457af1ff1a62081dadac597d7706bb656e9941bb828de6421d9c563cf`) via
  `tests/upstream_prism/reference_peg.rs`
- `test-chat-template` (`tests/test-chat-template.cpp`, SHA-256
  `7a790605222858ca5d5a2de94de27115d693e2c293638c392b7726a44ad3451f`) via
  `tests/upstream_prism/reference_chat_templates.rs`
- `test-chat-analysis` (`tests/test-chat-analysis.cpp`, SHA-256
  `d895ae1736845640b8dab486c68189726821abfbbfc0ca96f98cd419633ad1a1`) via
  `tests/upstream_prism/reference_chat_analysis.rs`
- `test-chat-peg-parser` (`tests/test-chat-peg-parser.cpp`, SHA-256
  `c0583c716c65fa7c06fc9b282fc8d24aecde29fa55271536e8c23dfbbc0429e0`) via
  `tests/upstream_prism/reference_chat_parser.rs`
- `test-chat-auto-parser` (`tests/test-chat-auto-parser.cpp`, SHA-256
  `d69af3154f3ed738e47a74280bcd63d2e90cd60b05dff7f5f1f7a5b26c59478c`) via
  `tests/upstream_prism/reference_chat_auto.rs`
- `test-chat` (`tests/test-chat.cpp`, SHA-256
  `9920d484dfe35d8ccec4e34f3dd6a23e1788a479a419cab231c361d3cad4e8b4`) via
  `tests/upstream_prism/reference_chat.rs`

All fifteen sources bind pinned upstream commit
`b87ca7225456b06b0975b8239c66b5aecf07e37c` (MIT terms retained in [LICENSE](LICENSE)).
`common/model_resolution.hpp`, `common/codec_buffers.hpp`, `common/peg.hpp`,
`common/chat_templates.hpp`, `common/chat_analysis.hpp`, `common/chat_parser.hpp`,
`common/chat_auto.hpp`, and `common/chat.hpp`, together with their bridge
includes, dispatch branches, CMake build edges, and the
`prism-upstream-server-context` CMake library target (`server-chat.cpp`/
`server-common.cpp`), are removed, leaving 12 checked-in C++ files and zero
checked-in C files (12 native files total) across the common bridge snapshot.
Direct native requests for those retired operations reject; complete Prism suite
execution, native bridge compilation and owning hosted acceptance remain
pending.
