# Production common-library process adapter

Build this separate CMake tree with `PRISM_SOURCE_DIR` naming the verified pinned
upstream root, and target `prism-upstream-common-bridge`. It statically links the
production `llama`, `llama-common` and `mtmd` libraries. Tests, tools, examples, server,
apps, TLS, optional external libraries and dependency downloads are disabled.
No upstream test source is compiled or included.

Run `prism-upstream-common-bridge --input ABSOLUTE_PATH --output ABSOLUTE_PATH`.
Input and output are bounded UTF-8 JSON files with this schema:

```json
{"SchemaVersion":1,"Operations":[
  {"Kind":"unicode-regex-split","Text":"text","Expressions":["expression"],"ByteEncoding":false},
  {"Kind":"mtp-first-decode-fits","Batch":32,"Catchup":31,"Sequences":1}
]}
```

Every operation is validated before any production API executes. Operation fields
are exact; unknown fields, duplicate object keys, unknown kinds and incorrect
types are rejected. `Batch`, `Catchup` and `Sequences` are signed 32-bit integers.
JSON nesting is limited to 160 levels for declarative graph transport. Request and response files are at most
16 MiB, and operations number 1 through 128. Text is at most 65,536 UTF-8 bytes; there are 1 through 16
expressions, each at most 8,192 UTF-8 bytes. Paths must be absolute and distinct.

The adapter forwards inputs directly to `unicode_regex_split` and
`common_speculative_mtp_first_decode_fits`. Results preserve input order:

```json
{"SchemaVersion":1,"Results":[{"Pieces":["piece"]},{"Fits":true}]}
```

`Pieces` contains the production split result without further encoding changes;
the production function receives `ByteEncoding` exactly as supplied. `Fits` is
the production boolean result. The Rust suite owns all expected results and
assertions. A zero process exit means the operations completed and the response
was written. Errors return a nonzero exit and a message on stderr. Discard any
response from a failing process. The caller owns the timeout and file lifecycle.

## Generic Jinja operations

Jinja operations call the production lexer, parser, runtime and value APIs. A
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

`mtmd-image-should-tile` takes exact fields `Kind`, `PatchSize`, `Merge`,
`MinImageTokens`, `MaxImageTokens`, `Width`, `Height`. It forwards the populated
production hyperparameters and image size to
`mtmd_image_preprocessor_lfm2::should_tile`, returning `{"ShouldTile":boolean}`.
Patch size ranges from 1 through 64, merge from 1 through 8, token bounds from 1
through 4,096 in ascending order, and dimensions from 1 through 4,096.

`mtmd-group-bitmaps` takes exact fields `Kind`, `Merge`, `Parts`. Merge ranges
from 1 through 64 and parts contains at most 64 descriptors:

- `{"Kind":"text","Text":"..."}`: at most 65,536 UTF-8 bytes.
- `{"Kind":"bitmap","Width":100,"Height":100,"Mergeable":true}`: dimensions 1 through 4,096.
- `{"Kind":"audio","Samples":100,"Mergeable":true}`: sample count 1 through 1,048,576.

The adapter constructs production bitmap handles with null payload pointers,
sets the supplied mergeability flags, and calls `mtmd_group_mergeable_bitmaps`.
It returns `{"GroupSizes":[...],"RemainingBitmapParts":number}` from the actual
groups and mutated part list. Descriptor generation and interpretation of any
compact test specifications belong to the Rust suite.

## Generic grammar-parser operation

`grammar-parser-parse` takes exact fields `Kind` and `Grammar`. Grammar is a
nonempty UTF-8 string of at most 65,536 bytes without NUL characters, since the
production parser consumes a C string. A fresh `llama_grammar_parser` receives
each request. The result is `{"Parsed":boolean,"SymbolIds":[{"Name":"...",
"Id":number}],"Rules":[[{"Type":number,"Value":number}]]}`. Symbol IDs retain
the production map's name order, and rules retain the production vector order.
The result is limited to 4,096 symbols, 4,096 rules and 65,536 total rule elements.
The Rust caller owns every grammar input, expected symbol and expected rule.

## Generic reasoning-budget and UTF-8 operations

`utf8-is-complete` takes exact fields `Kind`, `Bytes` (0 through 65,536 integer
bytes) and returns `{"Complete":boolean}` directly from
`common_utf8_is_complete`. Raw bytes permit incomplete and invalid UTF-8 inputs.

`reasoning-budget-trace` takes exact fields `Kind`, `Actions`, with 1 through
4,096 actions. Every action has a `Kind` and a `Sampler` handle (0 through 31),
or `null` for native APIs that accept a null sampler. Handles are owned only for
that operation and freed on all exit paths. Handle lifetimes are checked before
any native action executes. Additional fields and production calls are:

| Action Kind | Additional fields | Trace event |
| --- | --- | --- |
| `init` | `Vocabulary`: null; `StartSequences`, `EndSequences`: token sequence arrays; `ForcedTokens`: token array; `Budget`: i32; `InitialState`: enum 0 through 4 | `{}` |
| `apply` | `Candidates`: objects with `Id` i32, `LogitBits` u32, `ProbabilityBits` u32; `Selected`: valid index or -1; `Sorted`: boolean | `Candidates`, `Selected`, `Sorted` after `llama_sampler_apply` |
| `accept` | `Token`: i32 | `{}` |
| `clone` | `Target`: unused handle | `{}` |
| `state` | none | `{"State":enum}` |
| `force` | none | `{"Forced":boolean}` |
| `end-match` | none | `{"EndMatch":null}` or `{"EndMatch":[tokens]}` |
| `reset` | none | `{}` |
| `free` | none | `{}` |

The response is `{"Trace":[events in action order]}`. Start and end sequence
collections contain at most 64 sequences, each of at most 1,024 tokens, and the
forced sequence at most 1,024 tokens. Token IDs and budgets are signed 32-bit
integers. Each apply has at most 4,096 candidates, with at most 65,536 candidates
combined across the trace. Floats use their raw IEEE binary32 bits, preserving
infinities and NaNs without JSON number conversions. The complete backing
candidate vector is returned. Every state transition, clone, reset, forcing
operation and end match comes from the production sampler API; the adapter has
no test sequence generation, finite-count calculation or forcing assertions.

## Generic tensor graphs

`tensor-graph` takes exact fields `Kind`, `Threads` (1 through 16), `Nodes`
(1 through 96) and `Outputs` (1 through 16 prior node indices). An input node is
`{"Op":"input","Type":"F32"|"I32","Dimensions":[...],"DataHex":"..."}`.
Dimensions number 1 through 4, each 1 through 8,192, with at most 2,097,152
elements per tensor. Raw bytes are little-endian hex in ggml storage order and
must match the shape and type exactly. F32 inputs must contain finite values.

Other nodes take `Op` and `Inputs` (prior node indices), plus the exact operation
parameters accepted in `tensor_graph.hpp`. Supported production operations are
`cont`, `transpose`, `soft_max`, `permute`, `reshape`, `cast`, `mul_mat`, `add`,
`mul`, `rms_norm`, `view_2d`, `conv_transpose_1d`, `col2im_1d`, `rope` and
`rope_multi`. Cast outputs additionally support F16 and BF16. Shape, type,
broadcast, view extent and operation parameter validity are checked before any
ggml graph is constructed.

The adapter allocates a 128 MiB ggml context, with a conservative cumulative
storage estimate capped at 96 MiB, then executes the graph using the supplied
thread count. The response is `{"Outputs":[{"Type":"...","Dimensions":[four
padded dimensions],"DataHex":"..."}]}` in requested output order. Outputs must
be contiguous, and their combined hex data is limited to 15 MiB. Inputs, graph
specifications, random streams, reference calculations, tolerances and all
pass/fail decisions remain in the Rust suite.

## Generic stateful grammar operations

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
serialized as suffix arrays bounded by the original buffers. The Rust suite owns
input text decoding, encoded token IDs, expected rejections and match decisions.
Saved stacks copy only the production stack collection and are invalidated when
their owning grammar is freed. Grammar handles are freed on every exit path.

## Generic production codecs

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
reported as GPU execution. All input generation, independent transcription,
reference algorithms, tolerances and pass/fail decisions remain in Rust.

## Generic PEG builder and parse transport

`peg-evaluate` takes exact fields `Kind`, `Graph`, `Inputs`, `GenerateGrammar`,
`RoundTrip`. A graph is `{Statements:[{Let?:name,Expr:expression}],Root:expression}`.
Expressions are `{Ref:localName}` or `{Op:method,Args:[arguments]}`. Method names
and argument order follow the production PEG builder. `add`, `or`, `space-add`
call its actual `+`, `|`, `<<` overloads; raw string operands retain the string
overloads. `rule-lazy` takes a name, a nested graph body, and optional trigger
boolean, invoking the production callback overload. Other supported methods are
listed in `peg.hpp`; they include `marker`, JSON/Python value parsers, character
classes, rules, tagging, repetition, delimiters and GBNF wrappers.

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
API and parses the same supplied inputs. Native code does not compare outcomes.
`peg-serialize` takes `Kind`, `Graph` and returns `SerializedJson`.
`peg-deserialize` takes `Kind`, `SerializedJson` (at most 8 MiB) and returns the
production `Root` parser ID. Benchmark repetitions and timing belong to an
explicit CI-only Rust benchmark entry, using these ordinary production operations.

## Concurrent production logger

`concurrent-log` takes exact fields `Kind`, `Workers`, `Finalize`. One through 32
workers execute their supplied action arrays concurrently against the same
production logger singleton. Worker arrays contain at most 16,384 actions, and
all workers at most 65,536 combined. Actions are `emit` with exact `Level`
(`info`, `warning`, `error`, `debug`) and `Text` (at most 1,024 bytes, no NUL), or
`timestamps`/`prefix` with boolean `Value`. Each action also has its `Kind`.
`Finalize` contains at most eight `flush`/`free` strings and must end with `free`.
Workers join before finalization; captured worker exceptions fail the operation.
The result is `Completed`, containing action completion counts per worker.
Schedules and verdicts belong to Rust. This must be the final operation in a
request because the production logger singleton cache is not reset after free.

## CPU arena placement transport

`cpu-buffer-info` takes only `Kind` and returns production CPU buffer `Alignment`.
`cpu-arena-graph` takes `Kind`, `Graph`, `Forward`, `Allocation`. Graph is the
existing tensor-graph descriptor, restricted to F32 input, multiply, add and 2-D
views. Forward supplies explicit graph expansion indices. Allocation is null for
the production context allocator, or `{BufferBytes,Offsets}` for manual placement;
view offsets are null, deriving storage from their source. Offsets and extents
must fit the supplied buffer and meet the actual backend alignment.
The result is raw production `Status` and output `DataHex` values. Input creation,
overlap layouts, operation ordering and result comparisons remain in Rust. CPU
thread count comes from Graph; the pinned CPU backend defaults to four threads.

## Generic batch allocator trace

`batch-allocator` takes exact fields `Kind`, `PositionsPerEmbedding` (1 or 4),
`Actions` (1 through 64). An init action supplies raw token or embedding arrays,
optional positions/sequence IDs/logit flags, embedding width, sequence limit,
output-all flag, and optional `{Sequence,Min,Max}` input memory ranges. Other
actions call production `split-simple`, `split-equal`, `split-seq`, `split-reset`
or `reserve` APIs. Exact action fields are declared in `batch_allocator.hpp`.
Inputs are limited to 64 tokens, embedding width 8 and eight sequence IDs per
token; negative IDs and an oversized sequence limit may reach production's own
rejection paths. All requested array extents and scalar types are checked first.

The response is `Actions`, containing raw initialization status, allocator state
and ubatch observations in action order. Snapshots include every supplied/derived
batch field, output IDs, used/output counts, sequence position ranges, ubatch
counts, pointer presence and pointed-to content. The optional range-only memory
adapter returns caller-supplied position ranges; unused methods throw. Native
code has no original case selector, expected split table or allocator assertions.

## Generic graph allocator observations

`allocation-graph` takes exact fields `Kind`, `Backends`, `Steps`. Up to three
backend descriptors supply `Alignment` (power of two, 1..64) and `MaxBufferSize`
(null for unlimited, or 1..1 MiB). The native input backend supplies real aligned
backing storage, bounded to 4 MiB per allocation, and production gallocr behavior
uses the supplied logical limits. Buffer callbacks carry data/size metadata only.

One through four steps share the same production gallocr, with fresh graph
contexts per step. Steps supply `Nodes`, `Output`, per-node `BufferTypes`,
`ExternalAllocations`, and `Reserve`. Up to 40 nodes describe F32 input, add,
multiply, concatenate, scale, pad, mean, reshape, permute and view operations.
Each tensor has at most 16,384 elements and each axis at most 1,024. Exact node
and context fields are declared in `allocation.hpp`.

Results contain raw reserve/allocation status, live buffer sizes, tensor buffer
and relative address offsets, byte extents, source/view references, actual op and
flags, production in-place capability, output flags, graph nodes and leaves.
The Rust suite owns graph scenarios, memory overlap/liveness analysis and every
allocation assertion. The adapter does not execute tensor computation here.

The generic `mtmd-chunks-trace` transport exposes production public chunk collection, serialization, reload and decoder-position APIs with bounded handles and buffers. Its explicit `create-test-inputs` action calls the pinned production public `mtmd_test_create_input_chunks` factory, as the original source does: Rust owns the invocation and predicates, while the opaque populated chunks originate in that existing SUT factory. The adapter does not duplicate its contents or compile a C++ test source.

The `sampler-trace` transport owns scoped production sampler and candidate handles. Rust supplies constructors, raw IEEE binary32 candidate fields and ordered apply, accept, reset, clone, copy and free actions. Candidate copies retain the complete backing vector and rebase the production active-data pointer, size, selection and sorted metadata; snapshots return the active view. There are at most 4096 actions, 32 handles of each kind, 131072 entries per vector and 262144 retained entries. Explicit `clock` actions return `ggml_time_us`. Rust supplies any timed action interval; dispatch and metadata checks within that interval are included, while construction, request parsing and warmup can be excluded. Benchmark execution remains CI-only.

`gguf-inventory` returns every compiled production backend device. `gguf-read` forwards supplied bytes unchanged into the selected production file, buffer or callback reader. `gguf-context-trace` applies ordered typed KV setters, creates supplied tensor descriptors on the selected backend, snapshots metadata and selected tensor bytes, copies KV pairs, and writes/reloads through the selected reader. Numeric payloads use raw little-endian bytes; string values use hex bytes, preserving invalid UTF-8. The callback is bounded to 4096-byte reads and the original 4 GiB expected-file limit. Descriptor content and random generation remain in Rust. At most eight contexts, 256 setters/tensors per descriptor, 32 actions and 8 MiB supplied bytes are accepted; each snapshot selects at most 2 MiB tensor data. Native files use exclusive owned temporary directories beneath the caller's absolute process working directory, and failure to create a file fails the operation.

`chat-builtins`, `legacy-chat-render` and `common-chat-render` expose production template inventory and rendering. Legacy rendering returns the raw return code and supplied-capacity byte buffer; common rendering returns the production prompt or exception metadata. Templates, normalization and comparisons are Rust-owned. The preliminary `chat-peg-trace` transport evaluates a supplied declarative graph with the production chat PEG builder, retains its arena across explicit actions, and returns raw parse/mapping/diff observations. Source registration requires the corresponding complete Rust port; compiling this adapter does not establish source coverage.

`vocab-runtime-info` returns production hardware concurrency. `vocab-evaluate` loads a supplied absolute GGUF path with vocabulary-only model parameters and optionally creates the production context with its defaults. Rust supplies byte inputs, special-token flags, and up to256 explicit worker action schedules operating on that same shared context. Tokenize actions choose capture and diagnostic detokenization; clock actions return production microseconds. The exact response fields are `ModelLoaded`, `ContextLoaded`, `Workers` and `Error`; Error is null or a production load/worker failure message. The adapter returns raw per-action token observations, never expected tokens or comparisons. The default Rust source port reads all15 pinned `.inp`/`.out` corpora, verifies asset digests, reproduces Windows text I/O and map behavior, and captures source worker0. Text-file export is a separate bounded CI-only diagnostic; its timer includes action dispatch and token serialization. External vocabularies remain under the runner's verified source directory. The Rust target uses existing production `Tokenizer`, `Vocabulary` and optional `SpecialTokens` construction as its vocabulary context, and preserves unsupported metadata errors without relabeling formats.

`chat_auto.hpp` adds preliminary generic production chat helper, template-variant comparison and analyzer/parser traces. Its Rust source port must be complete before source registration; native compilation alone establishes no source coverage.

The separate `--model-session` mode provides bounded JSON-lines production model/context operations on persistent native worker lanes. Rust owns operation schedules, callbacks, token selection, stopping and all verdicts. Ordinary file-mode protocol is unchanged. `vocab_session.hpp` adds batch primitives `vocab-tokenize`, `vocab-detokenize`, `unicode-encode` and `unicode-decode`, each at most512 supplied items and1MiB supplied bytes. Values and per-item exception metadata return in parallel arrays; no native roundtrip comparison or codepoint/token enumeration occurs. Vocabulary-only model loading and shared read-only context access preserve the original concurrent tokenizer domain. Complete roundtrip coverage requires the safe Rust source ports, not merely these native calls.

The generic `json-schema-convert` and `json-schema-info` operations return production grammar/exception metadata and ordered schema-information traces. Rust owns positive/negative schemas, expected grammar text, validation and optional independent process comparison.

`public-c-abi` returns raw size/alignment observations from a separate C17 translation unit including production `llama.h`: model/context parameters, batch, token/position/sequence scalar widths and the C standard macro. The C file contains no expected sizes, assertions or constant compilation-success response.

`chat-template-trace` exposes supplied production template capabilities, direct applies, diffs and undefined-variable statistics. Rust supplies action order, the shared timestamp and structural comparisons; returned rendering bytes are hex so partial UTF-8 is retained.

`grammar-trace` also accepts explicit `accept-codepoint` actions (`Grammar`, `CodePoint`), calling `llama_grammar_accept` and returning the same exception, message and stack-size fields as token acceptance. File-mode `unicode-encode` and `unicode-decode` expose the existing session primitives with `Kind` replacing `Op`, returning parallel Values/Errors arrays. These calls perform no enumeration or acceptance comparisons.

`argument-enums`, `argument-parser-trace` and `remote-get-content` expose bounded production argument parsing and remote-file observations. The session-only resolution operations expose caller-supplied model-resolution plans and handler actions; Rust owns fixtures, loopback HTTP responses, reordering and comparisons.

`chat-trace`, `chat-value`, `chat-server-params`, `chat-generic-parse` and `chat-regex` expose production chat/response conversions, parsers and raw regex observations. The build links the upstream production server-context source list as a static library without enabling the HTTP server executable or UI. Inputs, templates, trigger selection, streaming schedules and all assertions remain in Rust.

The session `model-user` operation calls production `llama_model_init_from_user` using caller-supplied metadata. Each production tensor-initialization callback sends its actual TensorSpec to Rust and reads the caller's bounded data file; native code performs only the requested F32-to-F16 conversion when supplied as float32 bytes. Tensor contents, RNG and model schedules remain Rust-owned. Callback failures are latched and reported after the C API returns, without unwinding across the C callback. Context options and raw KV-center/logit observations expose production behavior without a native training or generation policy.

`optimizer-info` returns production backend inventory, hardware concurrency, optimizer defaults and context-overhead metadata. `optimizer-trace` accepts caller-supplied devices/scheduler order, bounded F32 graph and dataset descriptors, and explicit production allocator, optimizer, result, epoch and fit actions. Tensor values and result doubles use raw little-endian bytes/bits, retaining NaNs. Fit callbacks only select the supplied per-epoch parameter struct; RNG, rates, epochs, reset choices and assertions are Rust-owned. Raw saved tensor bytes support caller-requested save/restore without introducing expected values. At most8192 actions, sixteen live contexts/buffers/datasets/optimizers/results,256 tensor descriptors,1MiB per data payload and4096 fit epochs are accepted. Context and buffer lifetimes are validated separately from numerical test predicates.

`quantization-info` returns production GGML type names and default types for supplied ftype IDs. `quantization-model` constructs a production metadata-only quantization model and F32 tensor descriptors from supplied dimensions/names. Explicit `allows` actions return actual quantizability flags and production tensor names; `types` actions classify the caller's ordered tensor indices. At most32768 tensor descriptors and128 actions are accepted, with two million returned classifications. No expected schema, model catalog, sorting rule, filtering workload or ftype-name table exists in C++. The original test-only GGUF parser/cache/split library is rewritten entirely in safe Rust and never included or compiled here.

`backend-graph` accepts `Directory`, `MaxFileBytes` and either an `Actions` array or an absolute `ActionsFile` beneath that directory. All graph descriptors, input bytes, execution schedules, references, sentinels, comparisons and stopping decisions are Rust-owned. Native `call` actions dispatch an explicit allowlist of production GGML functions using typed context/tensor/graph/backend/buffer/scalar handles, signed/unsigned integers, raw float bits and bounded arrays. Integer argument marshalling can read supplied tensor metadata and checked binary arithmetic. It does not implement class-specific graph construction or control flow.

The result is `Trace`, one raw result per requested action. Context snapshots enumerate actual production tensors in context order; tensor metadata includes canonical ID and all bound aliases, dimensions, strides, operation parameters, sources, view root, byte/element/row counts and production flags/descriptions. Graph snapshots return ordered nodes, leaves and gradient/gradient-accumulator handles. `allocate`, `supports`, tensor-file I/O, caller-supplied mutable tensor fields, clocks and production quantization/dequantization primitives are explicit actions. Binary files contain native little-endian observations on the supported Windows ARM64 platform and must remain beneath `Directory` after canonicalization. `tensor-get` accepts `FileOffset` and optional `Truncate`; `scalar` accepts optional `Release` so an observed scalar handle can be reused.

`compare` calls the actual production backend comparison API, with a serialization-only callback that always continues. It writes independent raw left/right tensor files and metadata for every selected callback, plus explicitly requested captures. It performs no equality, tolerance, finite-value or sentinel assertions. Callback exceptions are latched and reported after the production C API returns. CPU backend creation returns whether the production `set_use_ref` procedure is available, and applies the caller's reference-mode choice when available.

Inline actions are capped at65,536. External action arrays are capped at128MiB,524,288 cumulative actions and160 JSON nesting levels, rejecting duplicate object keys. There are at most16 contexts, four CPU backends,32,768 handles per table,1GiB total context arenas and4GiB per binary file; a comparison captures at most8GiB. The ordinary16MiB response limit remains in force. A final `checkpoint` action atomically publishes the current raw `Trace` (including its empty acknowledgment) and waits up to the supplied1..60,000 milliseconds for a fresh next-actions file. That supplied array continues in the same scope; an empty array ends it. Rust owns every continuation, including source gradient and timing stop conditions. This transport wait neither generates actions nor decides numerical results. Full source matrices and benchmark execution remain CI-only.

`model-load-observe` and the persistent DSpark operations expose production model loading/logging, context staging/capture, tensor observations, samplers and speculation primitives. Their inputs, source schedules, environment choices and all predicates remain Rust-owned. The persistent provider retains production owners and callback data for their required lifetimes; compilation or malformed-request probes do not establish model or source coverage.

The persistent session accepts at most65,536 requests so the complete24-prompt
real-evaluation source can retain its model and speculative state across prompts.
The16MiB frame,64MiB stream,90-second operation and240-second session limits remain
unchanged. `context-logits` accepts row indices from-1 through2047, including the
production API's last-row selector. `model-chat-render` applies the loaded model's
own production template to supplied messages and explicit thinking/generation flags.

The persistent `llg-*` operations construct production LLGuidance, distribution and chain samplers and retain caller-owned candidate vectors. Candidate files supply exact little-endian IDs/logits/probabilities; updates may replace only logits or IDs/logits while retaining the original probability fields. Explicit patches, active-view metadata, apply probes, reset, accept and free operations return raw observations. Rust owns tokenization order, full-vector initialization, one-hot inputs, negative-logit decisions and EOS fallback. No native matching helper or grammar/schema test vectors are compiled. Candidates are capped at1,048,576 entries, files at12MiB and live sampler/candidate tables at64 entries each, within the existing model-session request and lifetime bounds.

`llg-available` reports the actual `LLAMA_USE_LLGUIDANCE` feature definition. A complete source contract requires it to be true; disabled builds cannot count as passing coverage. The build consumes separately verified absolute `PRISM_LLGUIDANCE_LIBRARY` and `PRISM_LLGUIDANCE_INCLUDE_DIR` paths, enabling the pristine production common source without upstream network build steps. `Initialize-BonsaiPrismGrammarDependency` verifies the locked source archive, extracts fresh source and invokes only the pinned production Cargo package with `--locked` for the explicit target triple. All dependency sources, targets, temporary files and receipts remain external to Git worktrees.
