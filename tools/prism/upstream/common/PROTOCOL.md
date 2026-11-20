# Production common-library process adapter

Build this separate CMake tree with `PRISM_SOURCE_DIR` naming the verified pinned
upstream root, and target `prism-upstream-common-bridge`. It statically links the
production `llama`, `llama-common` and `mtmd` libraries. Tests, tools, examples, server,
apps, TLS, optional external libraries and dependency downloads are disabled.
The [safe Rust harness](../README.md) owns every generated input, reference,
schedule, tolerance and verdict. This adapter validates and serializes production
calls and raw observations; it contains no upstream test sources or assertions.

Run `prism-upstream-common-bridge --input ABSOLUTE_PATH --output ABSOLUTE_PATH`.
Input and output are bounded UTF-8 JSON objects:

```json
{"SchemaVersion":1,"Operations":[{"Kind":"<operation-kind>"}]}
```

Every operation is validated before any production API executes. Operation fields are exact; unknown
fields, duplicate object keys, unknown kinds and incorrect types are rejected.
JSON nesting is limited to 160 levels for declarative
graph transport. Request and response files are at most 16 MiB, and operations number 1 through 128.
Paths must be absolute and distinct.

A zero process exit means completion and a written
response (`{"SchemaVersion":1,"Results":[...]}`). Errors return a nonzero exit and a message on
stderr. Discard any response from a failing process. The caller owns the timeout and file lifecycle.

The former top-level `unicode-regex-split`, `unicode-encode` and `unicode-decode` file operations are
retired. Those operations now belong to the independent Rust reference provider
(`tests/upstream_prism/reference_unicode.rs`) selected by the source harness; the native common
executable rejects them in file mode. Session-mode `unicode-encode`/`unicode-decode` primitives
inside `--model-session` (`vocab_session.hpp`) remain unchanged.

The former `mtp-first-decode-fits` native operation is retired. Its exact
`Batch`, `Catchup`, `Sequences` i32 protocol and `{"Fits":boolean}` result now
belong to the independent Rust reference selected by the source harness. The
native executable rejects this operation as unknown.

## Generic Jinja operations

`common/jinja.hpp` is retired from the native bridge and direct native requests
for `jinja-*` reject. `--target rust-reference` and the ordinary `cpp` reference
selector route `tests/test-jinja.cpp` through
`tests/upstream_prism/reference_jinja.rs` (`nativeReferenceReplay=false`) using an
independent safe Rust Jinja2 lexer, parser, evaluator, stats/caps/string-parts
engine and 64-bit hasher. Production Rust retains its separate runtime implementation.

Jinja operations execute the pinned Jinja APIs. A
template is at most 65,536 UTF-8 bytes. Variable-bearing operations accept exactly
one of `Variables` (an object) or `VariablesJson` (a string containing an object).
`VariablesJson` preserves object insertion order and numeric types such as `1.0`
across callers whose JSON DOM sorts keys or normalizes numbers. Variables are
limited to 256 KiB and 32 nesting levels, with duplicate object keys rejected.
Strings are marked as input when passed to the production runtime.

| Kind | Additional request fields | Successful result fields |
| --- | --- | --- |
| `jinja-render` | `Template`, variables, optional `ByteStrings` | `Rendered`: string |
| `jinja-hash` | `Chunks`: array of byte arrays | `Digest`: production 64-bit unsigned digest |
| `jinja-hash-trace` | `Actions`: array of `{"Update":[bytes]}` or `{"Digest":true}` | `Digests`: production 64-bit unsigned digests in action order |
| `jinja-stats` | `Template`, variables, `Paths`: array of paths | `Used`: booleans in path order |
| `jinja-caps` | `Template` | `SupportsStringContent`, `SupportsTypedContent`: booleans |
| `jinja-string-parts` | `Template`, variables | `Parts`: array of objects with `Value` string and `IsInput` boolean |
| `jinja-builtins` | none | `Builtins`: array of objects with `Type`, `Name` strings |

Hash chunks number at most 1,024 and contain at most 256 KiB of bytes combined.
Every byte must be an integer from 0 through 255. Builtin inventory is read from
the production global, int, float, string, array and object builtin maps, in that
group order and each map's iteration order.
Hash traces contain at most 2,048 actions and 256 KiB of update bytes combined;
they retain the same production hasher instance across all updates and digests,
including digest's mutation of buffered state.

A path is an array of 1 through 32 string keys or nonnegative integer indices,
beginning with a global variable name. String keys are at most 4,096 UTF-8 bytes
and indices at most 65,535. Stats accepts at most 1,024 paths. Optional render-only
`ByteStrings` contains at most 16 objects with exact fields `Path` and `Bytes`.
Each replaces an existing string at that path with at most 65,536 raw bytes after
the JSON variables have been parsed, permitting invalid UTF-8 inputs to production
code without placing invalid text on the JSON wire.

Production Jinja exceptions are observations returned per operation as
`{"ExceptionKind":"not-implemented"|"standard"|"unknown","Message":"..."}`;
they do not abort the batch. The caller decides which outcomes satisfy each case.
`not-implemented` identifies the production exception type only during runtime
execution and string gathering; the same type from setup, lexing or parsing is
reported as `standard`, preserving the caller's ability to follow the original
phase-specific exception contract.
Invalid protocol inputs still fail the process before execution. Rendered text
that cannot be encoded as JSON yields a standard encoding exception observation.

## Generic MTMD operations

These operations are retired from the native bridge and reject there. The Rust
production and independent reference adapters preserve the exact protocol below.

`mtmd-image-should-tile` takes exact fields `Kind`, `PatchSize`, `Merge`,
`MinImageTokens`, `MaxImageTokens`, `Width`, `Height`, returning
`{"ShouldTile":boolean}`. The independent reference minimizes exact distance to
the alignment lattice with even-index tie breaking, applies the patch-size clamp,
and compares area strictly against twice the maximum token-area allowance.
Patch size ranges from 1 through 64, merge from 1 through 8, token bounds from 1
through 4,096 in ascending order, and dimensions from 1 through 4,096.

`mtmd-group-bitmaps` takes exact fields `Kind`, `Merge`, `Parts`. Merge ranges
from 1 through 64 and parts contains at most 64 descriptors:

- `{"Kind":"text","Text":"..."}`: at most 65,536 UTF-8 bytes.
- `{"Kind":"bitmap","Width":100,"Height":100,"Mergeable":true}`: dimensions 1 through 4,096.
- `{"Kind":"audio","Samples":100,"Mergeable":true}`: sample count 1 through 1,048,576.

The independent reference partitions original runs of equal-sized mergeable
images into pairs, even when `Merge` exceeds two. Text, audio and nonmergeable or
different-sized images break runs. It returns
`{"GroupSizes":[...],"RemainingBitmapParts":number}` and counts original-vector
survivors independently of group sizes. Each request owns its metadata and drops
it after the observation; no native payload or borrowed handle survives. Every
operation is validated before producing results. No fresh native execution,
model/vision/audio support or whole-retirement verdict is claimed.

## Generic grammar-parser operation

This adapter is retired from the native bridge. The ordinary `cpp` reference
selector and explicit `rust-reference` selection use
`tests/upstream_prism/reference_grammar_parser.rs`, a character lexer, owned syntax
tree and separate instruction lowering pass. Production Rust retains its direct
byte parser. The independent provider imports neither production grammar code nor
the literal assertion fixture; reports claim no native replay. Direct native
`grammar-parser-parse` requests reject.

The Rust-owned `grammar-parser-parse` takes exact fields `Kind` and `Grammar`. Grammar is a
nonempty UTF-8 string of at most 65,536 bytes without NUL characters, since the
original production parser consumed a C string. Every operation owns fresh
parser/lowering state. The result is `{"Parsed":boolean,"SymbolIds":[{"Name":"...",
"Id":number}],"Rules":[[{"Type":number,"Value":number}]]}`. Symbol IDs retain
the production map's name order, and rules retain the production vector order.
The result is limited to 4,096 symbols, 4,096 rules and 65,536 total rule elements.

## Generic reasoning-budget and UTF-8 operations

This source route is retired from the native bridge. The Rust harness owns the
bounded protocol below for both production Rust and a separately implemented
reference. The legacy `cpp` reference-arm selector dispatches these operations
to `reference_reasoning.rs`; sending them to the C++ executable rejects as unknown.
No native execution or captured native output is claimed for this family.

`utf8-is-complete` takes exact fields `Kind`, `Bytes` (0 through 65,536 integer
bytes) and returns `{"Complete":boolean}` under the pinned trailing-boundary
contract. Raw bytes permit incomplete and invalid UTF-8 inputs.

`reasoning-budget-trace` takes exact fields `Kind`, `Actions`, with 1 through
4,096 actions. Every action has a `Kind` and a `Sampler` handle (0 through 31),
or `null` for the pinned null-sampler behavior. Handles are owned only for
that operation and freed on all exit paths. Handle lifetimes are checked before
an observable result is published. Additional fields and observations are:

| Action Kind | Additional fields | Trace event |
| --- | --- | --- |
| `init` | `Vocabulary`: null; `StartSequences`, `EndSequences`: token sequence arrays; `ForcedTokens`: token array; `Budget`: i32; `InitialState`: enum 0 through 4 | `{}` |
| `apply` | `Candidates`: objects with `Id` i32, `LogitBits` u32, `ProbabilityBits` u32; `Selected`: valid index or -1; `Sorted`: boolean | `Candidates`, `Selected`, `Sorted` after forcing |
| `accept` | `Token`: i32 | `{}` |
| `clone` | `Target`: unused handle | `{}` |
| `state` | none | `{"State":enum}` |
| `force` | none | `{"Forced":boolean}` |
| `end-match` | none | `{"EndMatch":null}` or `{"EndMatch":[tokens]}` |
| `free` | none | `{}` |

The response is `{"Trace":[events in action order]}`. Start and end sequence
collections contain at most 64 sequences, each of at most 1,024 tokens, and the
forced sequence at most 1,024 tokens. Token IDs and budgets are signed 32-bit
integers. Each apply has at most 4,096 candidates, with at most 65,536 candidates
combined across the trace. Floats use their raw IEEE binary32 bits, preserving
infinities and NaNs without JSON number conversions. The complete backing
candidate vector is returned. The original source has no reset calls; this
transport rejects reset rather than claiming additional source coverage.

## Generic tensor graphs

This adapter (`common/tensor_graph.hpp`) is retired from the native bridge. The ordinary `cpp`
reference selector and explicit `rust-reference` selection route `tensor-graph` to
`tests/upstream_prism/reference_tensor_graph.rs` with `nativeReferenceReplay=false`. Direct native
`tensor-graph` requests reject.

## Generic stateful grammar operations

`common/grammar.hpp` and `common/json_schema.hpp` are retired from the native
bridge. `--target rust-reference` routes grammar and JSON-schema cases through
`tests/upstream_prism/reference_grammar.rs` and
`tests/upstream_prism/reference_json_schema.rs` (`nativeReferenceReplay=false`),
and direct native bridge requests for `grammar-trace`, `json-schema-to-grammar`,
`json-schema-convert`, `json-schema-info`, and `gbnf-validator` reject.

`json-schema-to-grammar` takes exact fields `Kind`, `SchemaJson` (a JSON string
of at most 256 KiB), `ForceGbnf` (boolean), and returns `GrammarText` directly from
the production converter. Parsing the raw JSON text preserves property ordering.

`grammar-trace` takes `Kind` and `Actions` (1 through 16,384). Each action has
`Kind` and a `Grammar` handle from 0 through 31. Additional fields and results:

| Action Kind | Additional fields | Trace event |
| --- | --- | --- |
| `init-text` | `GrammarText` (at most 256 KiB, no NUL), `Root` (1..4096 bytes, no NUL) | `Created`: boolean |
| `init-rules` | `Rules`: arrays of `{Type,Value}` u32 elements, `StartRule`: index | `Created`: boolean |
| `clone` | `Target`: unused grammar handle | `Created`: boolean |
| `free` | none | `{}` |
| `state` | none | `StackSizes`: array of actual lengths |
| `save-stacks` | `StackHandle`: 0..63 | `{}` |
| `restore-stacks` | `StackHandle`: saved for the same live grammar | `{}` |
| `accept-token` | `Token`: i32, and exactly one of `Piece` string or `PieceBytes` byte array, at most 65,536 bytes | `ExceptionKind`: null, `runtime`, `standard` or `unknown`; `Message`: string; `StackSizes`: array |
| `snapshot` | none | `Rules`, `RuleAddresses`, `ElementSize`, `Stacks` |
| `reject-candidates` | `Stack`: current stack index; `Candidates`: array of `{Index,CodePoints,PartialUtf8:{Value,NRemain},Id}` | `Rejected`: same candidate fields |

Rule arrays number 1 through 4,096, contain at most 65,536 elements combined,
end with `END`, and use valid rule-reference indices. Element types range from
0 through 9. Snapshots return actual rule values, rule data addresses, element
size, and raw stack pointer addresses. They do not dereference stack pointers or
calculate pointer ownership; those comparisons belong to Rust. Snapshot stacks
contain at most 65,536 pointers combined, with at most 8,192 stacks.

Candidate arrays contain at most 4,096 entries, each with 1 through 1,024 u32
code points ending in zero, a u32 index, an i32 token ID, and partial UTF-8 state
with u32 `Value` and `NRemain` from -1 through 3. Rejected code-point pointers are
serialized as suffix arrays bounded by the original buffers.
Saved stacks copy only the production stack collection and are invalidated when
their owning grammar is freed. Grammar handles are freed on every exit path.

## Generic production codecs

`common/codec.hpp` and `common/codec_buffers.hpp` are retired from the native
bridge and direct native requests for `codec-inventory`, `codec-quantize`,
`codec-dequantize`, `codec-vec-dot`, and persistent `qb-*` buffer operations
reject. `--target rust-reference` and the ordinary `cpp` reference selector
route `tests/test-quantize-fns.cpp` and `tests/test-ptq1_0-cuda-dot.cpp` through
`tests/upstream_prism/reference_codec.rs`, and `tests/test-quantize-perf.cpp`
and `tests/test-quantize-stats.cpp` through
`tests/upstream_prism/reference_codec_buffers.rs`
(`nativeReferenceReplay=false`). Production Rust retains its separate
`engine_codec` transport.

Codec byte transport uses lowercase hex containing little-endian packed blocks
or IEEE binary32 values. Type IDs are the pinned production ggml enum values.
Counts are positive, at most 8,192, and must consist of whole blocks for every
participating type. Packed payloads are at most 1 MiB and must have the exact
production block extent. Quantizer float inputs must be finite. Codec payloads
receive native struct alignment before production calls.

| Kind | Additional request fields | Response |
| --- | --- | --- |
| `codec-inventory` | none | `Types`: production metadata and function availability |
| `codec-quantize` | `TypeId`, `Mode`: `cpu` or `reference`, `Count`, `DataHex`: F32 input | `DataHex`: packed output |
| `codec-dequantize` | `TypeId`, `Count`, `DataHex`: packed input | `DataHex`: F32 output |
| `codec-vec-dot` | `TypeId`, `Count`, `WeightsHex`, `ActivationsHex`: production vec-dot activation layout | `DataHex`: one F32 output |

Inventory returns exact fields `TypeId`, `Name` (string or null), `BlockSize`,
`TypeSize`, `HasFromFloatCpu`, `HasFromFloatRef`, `HasToFloat`, `HasVecDot`, and
`VecDotType` for every enum slot, including unused slots. Availability and block
alignment are checked before calling production CPU/reference quantizers,
decoders or CPU vector-dot functions. CPU initialization and required quantizer
table initialization use production APIs.

These operations do not execute CUDA or another GPU backend. A Rust port of a
CUDA-named source may compare its original host-side transcription against raw
production CPU observations; that establishes host coverage and must not be
reported as GPU execution.

## Generic PEG builder and parse transport

`common/peg.hpp` is retired from the native bridge and direct native requests
for `peg-evaluate`, `peg-serialize`, and `peg-deserialize` reject.
`--target rust-reference` and the ordinary `cpp` reference selector route
`tests/test-peg-parser.cpp` and its six nested `tests/peg-parser/*.cpp` suites
through `tests/upstream_prism/reference_peg.rs` (`nativeReferenceReplay=false`).

`peg-evaluate` takes exact fields `Kind`, `Graph`, `Inputs`, `GenerateGrammar`,
`RoundTrip`. A graph is `{Statements:[{Let?:name,Expr:expression}],Root:expression}`.
Expressions are `{Ref:localName}` or `{Op:method,Args:[arguments]}`. Method names
and argument order follow the production PEG builder. `add`, `or`, `space-add`
call its actual `+`, `|`, `<<` overloads; raw string operands retain the string
overloads. `rule-lazy` takes a name, a nested graph body, and optional trigger
boolean, invoking the production callback overload. Other supported methods were
previously exposed by retired `common/peg.hpp`; they include `marker`,
JSON/Python value parsers, character classes, rules, tagging, repetition,
delimiters and GBNF wrappers.

Construction is in explicit statement and child-array order, with at most 4,096
statements per graph, 16,384 expression nodes and 48 expression/graph nesting
levels. Local names are at most 128 bytes. Strings are at most 65,536 bytes,
parser/delimiter lists at most 1,024 entries and argument arrays at most 64 entries.
`Inputs` contains at most 1,024 `{Bytes:[0..255],Flags:0..3}` descriptors, each at
most 65,536 bytes. `GenerateGrammar` contains at most eight booleans passed as
production trigger-only/lazy modes. `RoundTrip` is boolean.

The result has exact fields `Results` (raw `{Type,Start,End}` parse results),
`GrammarTexts`, `SerializedJson` (string or null), `RoundTripResults`. A requested
round trip serializes the actual arena once, deserializes it with the production
API and parses the same supplied inputs.
`peg-serialize` takes `Kind`, `Graph` and returns `SerializedJson`.
`peg-deserialize` takes `Kind`, `SerializedJson` (at most 8 MiB) and returns the
production `Root` parser ID. Benchmark repetitions and timing belong to an
explicit CI-only Rust benchmark entry, using these ordinary production operations.

## Retired native concurrent logger

The native bridge rejects `concurrent-log`; its header, include, dispatch and
build edge are retired. The production Rust and separate Rust reference adapters
retain the following protocol and complete source schedule.

`concurrent-log` takes exact fields `Kind`, `Workers`, `Finalize`. One through 32
workers execute their supplied action arrays concurrently against the same
logger owner. Worker arrays contain at most 16,384 actions, and
all workers at most 65,536 combined. Actions are `emit` with exact `Level`
(`info`, `warning`, `error`, `debug`) and `Text` (at most 1,024 bytes, no NUL), or
`timestamps`/`prefix` with boolean `Value`. Each action also has its `Kind`.
`Finalize` contains at most eight `flush`/`free` strings and must end with `free`.
Workers join before finalization; captured worker exceptions fail the operation.
The result is `Completed`, containing action completion counts per worker.
The independent reference accepts exactly one terminal logger operation per
request, preserving the original singleton destruction boundary. Its bounded
channel joins all producers before flush/free and joins the consumer before
returning. Writer failures remain failures while queued producers are drained.
Reports identify the independent provider and claim no native replay or
output-content parity; the original source has no content assertions.

## CPU arena placement transport

This adapter (`common/arena_graph.hpp`) is retired from the native bridge. The ordinary `cpp`
reference selector and explicit `rust-reference` selection route `cpu-buffer-info` and
`cpu-arena-graph` to `tests/upstream_prism/reference_arena_graph.rs` with
`nativeReferenceReplay=false`. Direct native `cpu-buffer-info` and `cpu-arena-graph` requests
reject.

## Generic batch allocator trace

This adapter is retired from the native bridge. The ordinary `cpp` reference
selector and explicit `rust-reference` selection route `batch-allocator` to
`tests/upstream_prism/reference_batch_allocator.rs` with `nativeReferenceReplay=false`.
Direct native `batch-allocator` requests reject.

`batch-allocator` takes exact fields `Kind`, `PositionsPerEmbedding` (1 or 4),
`Actions` (1 through 64). An init action supplies raw token or embedding arrays,
optional positions/sequence IDs/logit flags, embedding width, sequence limit,
output-all flag, and optional `{Sequence,Min,Max}` input memory ranges. Other
actions run `split-simple`, `split-equal`, `split-seq`, `split-reset` or
`reserve` operations against an independent safe Rust state machine.
Inputs are limited to 64 tokens, embedding width 8 and eight sequence IDs per
token; negative IDs and an oversized sequence limit reach explicit validation
failure paths. All requested array extents and scalar types are checked first.

The response is `Actions`, containing raw initialization status, allocator state
and ubatch observations in action order. Snapshots include every supplied/derived
batch field, output IDs, used/output counts, sequence position ranges, ubatch
counts, pointer presence and pointed-to content. The optional range-only memory
adapter returns caller-supplied position ranges; unused methods throw.

## Generic graph allocator observations

This adapter is retired from the native bridge. The ordinary `cpp` reference
selector and explicit `rust-reference` selection route `allocation-graph` to
`tests/upstream_prism/reference_allocation.rs` with `nativeReferenceReplay=false`.
Direct native `allocation-graph` requests reject.

`allocation-graph` takes exact fields `Kind`, `Backends`, `Steps`. Up to three
backend descriptors supply `Alignment` (power of two, 1..64) and `MaxBufferSize`
(null for unlimited, or 1..1 MiB). The independent Rust allocator state machine
emulates aligned logical buffers, chunk limits and view/liveness metadata without
allocating native device storage.

One through four steps share the same allocator state, with fresh graph
contexts per step. Steps supply `Nodes`, `Output`, per-node `BufferTypes`,
`ExternalAllocations`, and `Reserve`. Up to 40 nodes describe F32 input, add,
multiply, concatenate, scale, pad, mean, reshape, permute and view operations.
Each tensor has at most 16,384 elements and each axis at most 1,024.

Results contain raw reserve/allocation status, live buffer sizes, tensor buffer
and relative address offsets, byte extents, source/view references, actual op and
flags, production in-place capability, output flags, graph nodes and leaves.
This operation does not execute tensor computation.

`mtmd-chunks-trace` is retired from this native bridge and direct requests reject.
The owning Rust reference arm uses a separate source-derived factory and bounded
wire-packet decoder for the complete `tests/test-mtmd-c-api.c` contract. The Rust
production arm retains its independent media chunk APIs. Both preserve the
original source assertions; this replacement supplies no captured native output.

`common/sampling.hpp` and `common/gguf.hpp` are retired from the native bridge.
`--target rust-reference` routes sampling and GGUF cases through
`tests/upstream_prism/reference_sampling.rs` and
`tests/upstream_prism/reference_gguf.rs` (`nativeReferenceReplay=false`), and
direct native bridge requests for `sampler-trace`, `gguf-inventory`, `gguf-read`,
and `gguf-context-trace` reject.

The `sampler-trace` transport owns scoped production sampler and candidate handles. Rust supplies
constructors, raw IEEE binary32 candidate fields and ordered apply, accept, reset, clone, copy and
free actions. Candidate copies retain the complete backing vector and rebase the production
active-data pointer, size, selection and sorted metadata; snapshots return the active view. There
are at most 4096 actions, 32 handles of each kind, 131072 entries per vector and 262144 retained
entries. Explicit `clock` actions return `ggml_time_us`. Rust supplies any timed action interval;
dispatch and metadata checks within that interval are included, while construction, request parsing
and warmup can be excluded. Benchmark execution remains CI-only.

`gguf-inventory` returns every compiled production backend device. `gguf-read` forwards supplied
bytes unchanged into the selected production file, buffer or callback reader. `gguf-context-trace`
applies ordered typed KV setters, creates supplied tensor descriptors on the selected backend,
snapshots metadata and selected tensor bytes, copies KV pairs, and writes/reloads through the
selected reader. Numeric payloads use raw little-endian bytes; string values use hex bytes,
preserving invalid UTF-8. The callback is bounded to 4096-byte reads and the original 4 GiB
expected-file limit. Descriptor content and random generation remain in Rust. At most eight
contexts, 256 setters/tensors per descriptor, 32 actions and 8 MiB supplied bytes are accepted; each
snapshot selects at most 2 MiB tensor data. Native files use exclusive owned temporary directories
beneath the caller's absolute process working directory, and failure to create a file fails the
operation.

`common/chat_templates.hpp` and `common/chat_parser.hpp` are retired from the
native bridge and direct native requests for `chat-builtins`,
`legacy-chat-render`, `common-chat-render`, and `chat-peg-trace` reject.
`--target rust-reference` and the ordinary `cpp` reference selector route
`tests/test-chat-template.cpp` through
`tests/upstream_prism/reference_chat_templates.rs` and
`tests/test-chat-peg-parser.cpp` through
`tests/upstream_prism/reference_chat_parser.rs` (`nativeReferenceReplay=false`).
For `chat-builtins`, `legacy-chat-render`, and `common-chat-render`, templates,
normalization and comparisons are Rust-owned; for `chat-peg-trace`, the Rust
reference evaluates a supplied declarative graph, retains its arena across
explicit actions, and returns raw parse/mapping/diff observations.

`vocab-runtime-info` returns production hardware concurrency. `vocab-evaluate` loads a supplied
absolute GGUF path with vocabulary-only model parameters and optionally creates the production
context with its defaults. Rust supplies byte inputs, special-token flags, and up to256 explicit
worker action schedules operating on that same shared context. Tokenize actions choose capture and
diagnostic detokenization; clock actions return production microseconds. The exact response fields
are `ModelLoaded`, `ContextLoaded`, `Workers` and `Error`; Error is null or a production load/worker
failure message. The Rust source port verifies all fifteen pinned `.inp`/`.out` corpora,
reproduces Windows text I/O and map behavior, and captures source worker zero. Text-file export
is a separate bounded CI-only diagnostic; its timer includes action dispatch and token
serialization. External vocabularies remain under the runner's verified source directory. The Rust
target uses existing production `Tokenizer`, `Vocabulary` and optional `SpecialTokens` construction
as its vocabulary context, and preserves unsupported metadata errors without relabeling formats.

`common/chat_auto.hpp` is retired from the native bridge and direct native
requests reject. `--target rust-reference` and the ordinary `cpp` reference
selector route `tests/test-chat-auto-parser.cpp` through
`tests/upstream_prism/reference_chat_auto.rs` (`nativeReferenceReplay=false`).

The separate `--model-session` mode provides bounded JSON-lines production model/context operations
on persistent native worker lanes. `vocab_session.hpp` provides batch
primitives `vocab-tokenize`, `vocab-detokenize`, `unicode-encode` and `unicode-decode`, each at
most512 supplied items and1MiB supplied bytes. Values and per-item exception metadata return in
parallel arrays.
Vocabulary-only model loading and shared read-only context access preserve the original concurrent
tokenizer domain. Complete roundtrip coverage requires the safe Rust source ports, not merely these
native calls.

`json-schema-convert` and `json-schema-info` return production grammar/exception
metadata and ordered schema-information traces.

`public-c-abi` (`common/public_c_abi.c`, `common/public_c_abi.h`) is retired from
the native bridge and direct native requests reject; `--target rust-reference` and
the ordinary `cpp` reference selector route `tests/test-c.c` through
`tests/upstream_prism/reference_public_c_abi.rs` (`nativeReferenceReplay=false`),
observing `#[repr(C)]` ABI mirror sizes/alignments, token/position/sequence-ID
scalar widths and the C17 standard constant without invoking native code or
asserting pass verdicts outside Rust.

`common/chat_analysis.hpp` is retired from the native bridge and direct native
requests for `chat-template-trace` reject. `--target rust-reference` and the
ordinary `cpp` reference selector route `tests/test-chat-analysis.cpp` through
`tests/upstream_prism/reference_chat_analysis.rs` (`nativeReferenceReplay=false`).
Rust supplies action order, the shared timestamp and structural comparisons;
returned rendering bytes are hex so partial UTF-8 is retained.

`grammar-trace` also accepts explicit `accept-codepoint` actions (`Grammar`, `CodePoint`), calling
`llama_grammar_accept` and returning the same exception, message and stack-size fields as token
acceptance. Top-level file-mode `unicode-encode` and `unicode-decode` are retired alongside
`unicode-regex-split` (handled by `reference_unicode.rs`); session-mode `vocab_session.hpp`
primitives remain only on `--model-session`.

`common/arg_parser.hpp` and `common/model_resolution.hpp` are retired from the
native bridge and direct native requests for `argument-enums`,
`argument-parser-trace`, `remote-get-content`, and session resolution operations
reject. `--target rust-reference` and the ordinary `cpp` reference selector route
`tests/test-arg-parser.cpp` through `tests/upstream_prism/reference_arg_parser.rs`
and `tests/test-model-resolution.cpp` through
`tests/upstream_prism/reference_model_resolution.rs`
(`nativeReferenceReplay=false`).

`common/chat.hpp` and the `prism-upstream-server-context` CMake library target
(`server-chat.cpp`/`server-common.cpp`) are retired from the native bridge, and
direct native requests for `chat-trace`, `chat-value`, `chat-server-params`,
`chat-generic-parse`, and `chat-regex` reject. `--target rust-reference` and
the ordinary `cpp` reference selector route `tests/test-chat.cpp` through
`tests/upstream_prism/reference_chat.rs` (`nativeReferenceReplay=false`).

The session `model-user` operation calls production `llama_model_init_from_user` using
caller-supplied metadata. Each production tensor-initialization callback sends its actual TensorSpec
to Rust and reads the caller's bounded data file; native code performs only the requested F32-to-F16
conversion when supplied as float32 bytes. Callback failures are latched and reported
after the C API returns, without unwinding
across the C callback. Context options and raw KV-center/logit observations expose production
behavior without a native training or generation policy.

`common/optimizer.hpp` is retired from the native bridge and direct native requests
for `optimizer-info` and `optimizer-trace` reject. `--target rust-reference` and the
ordinary `cpp` reference selector route `tests/test-opt.cpp` through
`tests/upstream_prism/reference_optimizer.rs` (`nativeReferenceReplay=false`) using an
independent CPU training graph, autodiff, AdamW/SGD updater and MT19937 shuffler.
`optimizer-info` returns backend inventory, hardware concurrency, optimizer defaults and
context-overhead metadata. `optimizer-trace` accepts caller-supplied devices/scheduler order,
bounded F32 graph and dataset descriptors, and explicit production allocator, optimizer, result,
epoch and fit actions. Tensor values and result doubles use raw little-endian bytes/bits, retaining
NaNs. Fit callbacks select the supplied per-epoch parameter struct. Raw saved tensor
bytes support caller-requested save/restore
without introducing expected values. At most8192 actions, sixteen live
contexts/buffers/datasets/optimizers/results,256 tensor descriptors,1MiB per data payload and4096
fit epochs are accepted. Context and buffer lifetimes are validated separately from numerical test
predicates.

The native `quantization-info` and `quantization-model` bridge operations (formerly provided by retired
`common/quantization.hpp`) are removed. Independent `test-quant-type-selection` checks run in-process
through `tests/upstream_prism/reference_quantization.rs` and disclose `nativeReferenceReplay=false`;
direct native requests reject as unsupported.

`backend-graph` accepts `Directory`, `MaxFileBytes` and either an `Actions` array or an absolute
`ActionsFile` beneath that directory. Native `call` actions
dispatch an explicit allowlist of production GGML functions using typed
context/tensor/graph/backend/buffer/scalar handles, signed/unsigned integers, raw float bits and
bounded arrays. Integer argument marshalling can read supplied tensor metadata and checked binary
arithmetic. It does not implement class-specific graph construction or control flow.

The result is `Trace`, one raw result per requested action. Context snapshots enumerate actual
production tensors in context order; tensor metadata includes canonical ID and all bound aliases,
dimensions, strides, operation parameters, sources, view root, byte/element/row counts and
production flags/descriptions. Graph snapshots return ordered nodes, leaves and
gradient/gradient-accumulator handles. `allocate`, `supports`, tensor-file I/O, caller-supplied
mutable tensor fields, clocks and production quantization/dequantization primitives are explicit
actions. Binary files contain native little-endian observations on the supported Windows ARM64
platform and must remain beneath `Directory` after canonicalization. `tensor-get` accepts
`FileOffset` and optional `Truncate`; `scalar` accepts optional `Release` so an observed scalar
handle can be reused.

`compare` calls the actual production backend comparison API, with a serialization-only callback
that always continues. It writes independent raw left/right tensor files and metadata for every
selected callback, plus explicitly requested captures. It performs no equality, tolerance,
finite-value or sentinel assertions. Callback exceptions are latched and reported after the
production C API returns. CPU backend creation returns whether the production `set_use_ref`
procedure is available, and applies the caller's reference-mode choice when available.

Inline actions are capped at65,536. External action arrays are capped at128MiB,524,288 cumulative
actions and160 JSON nesting levels, rejecting duplicate object keys. There are at most16 contexts,
four CPU backends,32,768 handles per table,1GiB total context arenas and4GiB per binary file; a
comparison captures at most8GiB. The ordinary16MiB response limit remains in force. A final
`checkpoint` action atomically publishes the current raw `Trace` (including its empty
acknowledgment) and waits up to the supplied1..60,000 milliseconds for a fresh next-actions file.
That supplied array continues in the same scope; an empty array ends it. Rust owns every
continuation, including source gradient and timing stop conditions. This transport wait neither
generates actions nor decides numerical results. Full source matrices and benchmark execution remain
CI-only.

`model-load-observe` (`common/dspark.hpp`) is retired from the native bridge and direct
native requests reject; `--target rust-reference` and the ordinary `cpp` reference selector
route `tests/test-dspark-logsnr-meta.cpp` through `tests/upstream_prism/reference_dspark_logsnr.rs`
(`nativeReferenceReplay=false`) with an independent bounded GGUF v2/v3 parser and log-SNR guard.
The persistent DSpark operations (`common/dspark_runtime.hpp`) remain unchanged and expose model loading/logging, context
staging/capture, tensor observations, samplers and speculation primitives. The provider
retains production owners and callback data for their required lifetimes.

Persistent sessions accept at most 65,536 requests, with a 16 MiB frame, 64 MiB
stream, 90-second operation and 240-second session limit.
`context-logits` accepts row indices from -1 through 2047, including the
production API's last-row selector. `model-chat-render` applies the loaded model's
own production template to supplied messages and explicit thinking/generation flags.

The persistent `llg-*` operations construct production LLGuidance, distribution and chain samplers and
retain caller-owned candidate vectors. Candidate files supply exact little-endian
IDs/logits/probabilities; updates may replace only logits or IDs/logits while retaining the original
probability fields. Explicit patches, active-view metadata, apply probes, reset, accept and free
operations return raw observations. Candidates are capped at 1,048,576 entries, files at 12 MiB
and live sampler/candidate tables at64 entries each, within the existing model-session request and
lifetime bounds.

`llg-available` reports the actual `LLAMA_USE_LLGUIDANCE` feature definition. A complete source
contract requires it to be true; disabled builds cannot count as passing coverage. The build
consumes separately verified absolute `PRISM_LLGUIDANCE_LIBRARY` and `PRISM_LLGUIDANCE_INCLUDE_DIR`
paths, enabling the pristine production common source without upstream network build steps.
`Initialize-BonsaiPrismGrammarDependency` verifies the locked source archive, extracts fresh source
and invokes only the pinned production Cargo package with `--locked` for the explicit target triple.
All dependency sources, targets, temporary files and receipts remain external to Git worktrees.
