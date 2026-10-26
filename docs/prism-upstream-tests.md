# Runtime conformance

The PQ2/PTQ1 GGML process adapter is now Rust (`tools/prism/upstream/bridge/`),
linked to the unchanged pinned production ggml library. The reference target
remains `cpp` because it observes production Prism. Its binary protocol, numerical
calls, resource limits and executable name are preserved. The separate common API
bridge remains native. Focused protocol checks run before GGML tools are used or
published; `tests/Test-PrismGgmlBridge.ps1 -ReferenceBridge <old-executable>` also
compares old/new response bytes and rejection behavior.

This repository owns the complete upstream source/backend catalog, codec parity
and their CI orchestration. SwiftBonsai Runner keeps app and released-runtime
integration checks. The [Swift MVP checklist](https://github.com/symbiogenesis/SwiftBonsai2Runner/blob/main/docs/prism-upstream-todo.md)
remains narrower than this complete runtime contract. Existing required green
source identities, reference assertions and opt-in measurement policy are retained.

[Runtime conformance](../.github/workflows/runtime-conformance.yml) imports tests from the exact
Prism archive in [source.lock.json](https://github.com/symbiogenesis/prism-inference/blob/main/tools/prism/source.lock.json), verifies its
length and SHA-256, and extracts it into fresh external storage. The reference
build verifies the archive and records the explicit production corrections
described below. No engine source or build tree is
vendored here.
The upstream MIT terms remain in [PRISM-LICENSE.txt](https://github.com/symbiogenesis/prism-inference/blob/main/tools/prism/PRISM-LICENSE.txt).

The [catalog](https://github.com/symbiogenesis/prism-inference/blob/main/tools/prism/upstream/catalog.json) records all **71** recursive
`tests/**/test-*.cpp`, `.c`, `.py` and `.sh` source files, their full relative
identities and SHA-256 digests, plus the test-registration CMake digest. This is
a source-file coverage measure: files contain different numbers of assertions,
and some are helpers, manual tests, model tests or benchmarks. It must not be
read as an assertion-level pass rate or full-model qualification.

## What runs

All original C++ test executables and fixture exporters are retired. Their
importer and replay build were removed after Windows ARM64 CI proved the safe
Rust replacements for every previously required C++ control. Native builds
disable upstream tests and expose production operations through bounded bridges.
The Windows bridge preserves synchronous exceptions from public C ABI calls;
this is required to observe the original negative-input rejection checks.
A Rust-owned reference failure still fails CI independently of the runtime floor.
The manual [focused reference workflow](../.github/workflows/prism-focused.yml)
runs one selected source on Windows ARM64 for diagnostics. Its result does not
replace Runtime conformance's complete source and backend gates.
The `BackendPlan` choice instead prepares revision-bound tools and plans
partition 0 of 16 for both targets in evaluation, support and performance modes.
Its optional `gradients` input also plans gradient mode. It produces no source
verdict or benchmark measurement; this short diagnostic checks backend planning
without first running the ordinary source catalog.

The migration inventories **all 71 catalog sources**, including model and
benchmark cases. Acceptance targets **Snapdragon X Elite on Windows ARM64**.
The Apple-only wired-memory source is explicitly outside that target and never
counts as a pass. Every Windows-applicable source now has a safe Rust adapter. Passing reference
qualification is tracked separately. Every previously passing source
remains required after replacement with safe Rust assertions against pinned Prism.
The Apple-only source remains visible as unsupported and outside the selected target.

The pinned C++ saver advertises DSpark support but omits required metadata and
global tensors. A focused load/save/reload reproduction confirms that its saved
model cannot reload. The reference applies an explicit, hash-verified
[production saver correction](https://github.com/symbiogenesis/prism-inference/blob/main/tools/prism/upstream/patches/dspark-saver.patch)
to an external copy of that translation unit. The immutable archive stays
unchanged, and receipts identify the patch and resulting source. All architecture
save/load assertions remain required; this reference is qualified with the
recorded production correction.
The original Llama-BPE roundtrip also exposes inherited punctuation cleanup:
token 24268 decodes from ` !$` to `!$`, then retokenizes into two tokens. The
explicit [Llama 3 decoding correction](https://github.com/symbiogenesis/prism-inference/blob/main/tools/prism/upstream/patches/llama3-cleanup.patch)
preserves bytes for the canonical `llama3`, `llama-v3` and `llama-bpe` metadata
names, following the publisher's tokenizer. Other pretokenizer aliases retain
their behavior. Both fixes use verified external translation units, and every
original assertion remains required.
The [migration inventory](https://github.com/symbiogenesis/prism-inference/blob/main/tools/prism/upstream/migration.json) lists every
source's assertions and prerequisites, bound to its pinned digest. Its port flags
must match the executable adapter list; they are not test verdicts.

The Rust harness, adapters and fixtures live together under
[`prism-inference/tests/upstream_prism/`](https://github.com/symbiogenesis/prism-inference/blob/main/tests/upstream_prism/).
An explicit Cargo example target retains the `upstream_prism` command-line runner
and its embedded unit tests, so the existing build and CI commands remain valid.

Seventy Windows-applicable source contracts currently have safe Rust adapters in
[harness.rs](https://github.com/symbiogenesis/prism-inference/blob/main/tests/upstream_prism/harness.rs):

- `test-tokenizer-0.py`, `test-tokenizer-0.sh` and `test-tokenizer-random.py`:
  independent safe Rust HF/SentencePiece references, all 26 original families,
  five default exhaustive schedules and seven optional generator families.
  All 122 input files are immutable and hash checked. Thirteen original
  vocabularies remain unchanged; safe Rust constructs the thirteen missing
  fixtures from original publisher inputs. Exact original configurations are
  verified even when supplied by public byte-identical mirrors or reconstruction.
  The file helpers retain caller-supplied inputs; acceptance uses an explicitly
  selected pinned source text. These original scripts have zero numerical
  assertions and non-failing mismatch diagnostics. Reports preserve that verdict
  and expose `StrictReferenceMatch` separately. Stateless process rotation keeps
  the complete schedules within bounded transport limits.
  Production load or operation rejections produce failed source results with
  their actual reasons. The shell workflow stops at the original failure point
  and retains completed rows; malformed transport remains an infrastructure error.
- `test-quantize-perf.cpp`: all five original operation schedules, eligible
  types, sizes, alignment and iteration options. This benchmark-only source has
  zero numerical assertions; passing means the workload completed. The source's
  wrong-type vector-dot operand preparation is explicitly preserved.
- `test-quantize-stats.cpp`: all float layers and eligible codecs, complete
  150-bucket histograms and original error/reduction equations. Safe Rust owns
  chunk scheduling and worker-local reductions. Its default compatible input
  copies all 75 original tensors from a pinned publisher F32 checkpoint; no layer
  is dropped for block-size compatibility. Both quantization sources are CI-only.
  Windows continuation reads allow file deletion sharing after atomic publication
  and report I/O failures directly; Rust retains the first native/controller error.
- `test-backend-ops.cpp`: all 112 active CPU classes and 22,817 ordered
  geometries in each of evaluation and support modes. Rust interprets
  the complete pinned source program, builds inputs and computes every reference
  and metric. Sixteen hosted planning partitions require complete plans and
  source-order random offsets before 32 execution partitions run. A final gate rejects missing or duplicate
  results. The 1,530 original performance workloads remain a separate requested
  CI measurement. CPU selection is explicit; source random-device seeds become
  logged reproducible Rust seeds.
  The original CTest registration supplies no arguments and defaults to
  evaluation (`test`) mode. Gradient mode is a separate source CLI option;
  Foundation's `prism_gradients` input additionally requires its complete matrix.
  The safe Rust gradient implementation is retained without added skips. Among
  GLU variants, the pinned production backward builder supports only split
  SWIGLU and aborts on the source's GEGLU gradient graph. That upstream failure
  is retained as a diagnostic; it is not reported as a passing gradient check. The default
  source gate requires every evaluation result plus the support preflight.
  The selected CPU receives the source thread count; the reference CPU retains
  its production default, matching the original source.
  Bounded summaries retain each case's identity and full-result hash. Complete
  metrics and input receipts remain in the uploaded case receipts; aggregation
  preserves the ordered digest of those full results.
  Plan and execution handoff artifacts contain only their revision-bound
  `record.json`. Complete case receipts and native diagnostics remain in the
  separate `prism-backend-evidence-plan-*` and `prism-backend-evidence-run-*`
  artifacts. Fan-in jobs read the compact records and retain every identity,
  digest, ordering and completeness check.
  Each C++ plan/execution partition reuses one bounded pipe process while every
  geometry receives fresh production state. Default evaluation/support planning
  and execution use 96 native partition workers instead of 91,268 per-geometry
  process launches. Source-order random offsets, input receipts, geometry counts,
  tolerances and result hashes remain required. CI compares the original file
  transport with the pipe transport on focused numerical/source geometries and
  checks state isolation and malformed-protocol rejection before shard execution.
- `test-tokenizers-repo.sh`: all six publisher vocabulary/input/output triples
  in the complete compatible 2026-05-22 repository snapshot,
  with every corpus attempted even after a failure. The existing Rust corpus
  predicates are shared with `test-tokenizer-0.cpp`; immutable provisioning
  replaces the source shell script's mutable repository update.
  The source accepts a repository argument; this pin keeps every independent
  reference token unchanged. Later publisher additions include tokenizers absent
  from the pinned fork and are outside this selected input revision.
- `test-lora-conversion-inference.sh`: all three original Gemma, Llama and
  Phi models, all nine base/hot/merged output-prefix predicates, and the same
  greedy 50-token schedules. Safe Rust converts the 28 pinned original inputs;
  merging invokes the actual production LoRA exporter. No Python or C++ test
  executable remains in this path.
- `test-export-graph-ops.cpp`: both local-model and pinned HF metadata paths, all
  prompt/token-generation reserve calls, structural set ordering and first-name
  retention, and original text export. Rust owns metadata reconstruction and
  graph filtering; the bridge only exposes production reserve observations.
  The source has zero numerical output assertions.
- `test-backend-sampler.cpp`: all 19 source scenarios, penalties/filter matrices,
  CDF comparisons, multi-output transactions and CPU-suffix callback contracts.
  Generic native callbacks report observations to Rust, which owns acceptance
  decisions, counters and tolerances. The original MSVC double RNG is reproduced
  in safe Rust and independently matched for 3,000 draws and float narrowings.
  Its default input is a Rust-generated F32 Llama with the first 4,096 vocabulary
  entries from the locked model. This covers the source's fixed token 3,834.
  Fixed seed 12,345 and ordinary fan-in initialization produce pinned model
  bytes; receipts retain metadata and all 21 tensor hashes for both targets.
  The earlier stories15M input fails the unchanged bias-10/seed-88 condition
  with both production CPU and backend sampling. The generated input retains
  that condition and every original schedule; caller-selected models remain
  available for diagnostics.
- `test-dspark-forward.cpp`: all six Windows CPU modes, including scalar/head
  oracles, SNR/output A/B, runtime/prefix grids, replay, capture atomicity and
  negative-prefix contracts. Tier2 uses the independent Rust reference and
  preserves the source's diagnostic-only numeric differences. Original Metal and
  CUDA modes are explicitly outside the selected Snapdragon CPU backend scope.
- `test-dspark-loop.cpp`: the full seven-entry acceptance cycle, 49 exact token
  comparisons and 95 original primitive predicates. Safe Rust computes the tiny
  fixture's independent reference before opening the production session, using
  pinned published DeepSpec equations and no recorded engine outputs. External
  model/reference pairs remain available as typed diagnostics.
- `test-dspark-real-eval.cpp`: all 24 literal prompts, separate plain AR and
  speculative schedules, exact output parity, cache rollback refusals, category
  and conditional-depth rollups. Optional teacher forcing retains split/same-path
  modes, finite-logit checks and ordered KLD calculations. Routine CI uses a small
  Rust-generated compatible pair; caller-supplied pairs and JSONL remain explicit
  diagnostics. Timings include bridge dispatch and qualify workload completion,
  with no publisher model quality or Snapdragon throughput claim. Every execution
  is CI-only because the source includes benchmark timings.
  The complete prompt loop retains one bounded 900-second session on main CI
  (960-second outer process limit); individual operations still have 90 seconds.
  Native ARM64 evidence reached prompt 13 at the ordinary 240-second lifetime.
- `test-grammar-llguidance.cpp`: all 42 grammar/schema groups, 344 literal
  predicates, and every token/EOS sampler-chain check. Rust owns candidate
  construction and matching decisions. The reference links the exact production
  LLGuidance commit pinned by upstream, with its unchanged Cargo lock.
  One model/context spans every group and the sampler chain, matching the
  original source and retaining the production vocabulary tokenizer cache.
  Rust supplies one complete candidate template; field-selective restores
  preserve logits-only versus ID/logit resets and the original probability and
  active-view metadata. Native equivalence checks compare all three encodings
  with the original file imports, including exceptional float bit patterns.
- `test-dspark-logsnr-meta.cpp`: all eight generated checkpoint variants and
  fourteen original loader predicates, including required error-log checks.
  Safe Rust writes every GGUF byte; an input-only tensor corpus preserves the
  pinned companion generator's exact NumPy data. All eight generated files were
  independently verified byte for byte against that generator.
- `test-gguf-model-data.cpp` and `test-quant-type-selection.cpp`: the complete
  GGUF test helper is rewritten in safe Rust, including progressive prefixes,
  cache handling, shard merging and binary metadata parsing. The helper's 34
  predicates are target independent. Production quantization selection compares
  all eleven models and 363 untouched expected schema sections. Raw publisher
  inputs are independently pinned; the broken Nemotron repository and schema
  paths have explicit corrections. Schema generation is a separate, explicitly
  confirmed CI diagnostic that writes only external evidence.
- `test-barrier.cpp`: all three CPU threadpool schedules, including the
  2,000-node graph, 10,000 changing-thread iterations and 1,000 two-graph
  iterations. The source has zero numerical output assertions. Its workload
  completion, additional API success guards and timings are reported explicitly;
  every execution of this timed source is restricted to CI.
- `test-llama-archs.cpp`: all 151 source variants, the production architecture
  inventory and capability skips, backend comparisons, and model export/load
  roundtrips. Rust creates metadata and tensor weights, retains the original
  numeric predicates, and records seed 1234 as a repeatable replacement for the
  source's unpinned random seed. The export phase runs separately from backend
  comparison and preserves saver-capability and temporary-file skips.
- `test-recurrent-state-rollback.cpp`: all three generated architecture fixtures and
  their full, partial and dirty restores, two-sequence replay, independent tails,
  and pending-fork byte equality. Fixtures are generated within this case's
  evidence directory; it does not depend on an earlier test having run.
- `test-chat.cpp`: all 14 default groups and 431 cases, including every byte
  prefix, final-message updates and reconstruction checks. A separate diagnostic
  retains source filters and forced reconstruction, including the filtered-only
  regression. All 59 template assets retain their original bytes.
- `test-arg-parser.cpp`: all 27 example inventories, option and environment
  uniqueness, repeated valid/invalid parses, load modes, LoRA CSV settings and
  the original HTTP cases. POSIX-only environment mutations remain excluded on
  Windows, as in the source.
- `test-model-resolution.cpp`: all 14 cases, 94 deterministic reorderings and
  13 assembly cases. A bounded Rust loopback server and private cache provide
  the source responses; the child invokes the production resolver. Accepted
  Windows sockets use blocking I/O with five-second read and write deadlines,
  so a client may connect before sending its HTTP header.
- Save/load state and thread safety preserve their complete live model scopes.
  Main CI allows 600 and 1,800 seconds respectively (660 and 1,860 seconds for
  their outer processes). Hosted ARM64 traces reached the ordinary 240-second
  lifetime while making valid decode progress. The 90-second operation deadline
  and all transport bounds remain in effect.
- `test-kv-mean-center.cpp`: all three generated-model groups and original
  numeric predicates. Rust supplies GGUF metadata, Windows random inputs and
  tensor bytes through bounded file callbacks to production model construction.
- `test-opt.cpp`: all 118 AdamW and 46 SGD result subtests, including dataset
  batching, shuffling, accumulated gradients, epoch/fit equivalence and training
  regressions. The original diagnostic-only SGD tolerance policy is explicit.
- `test-double-float.cpp`: the complete binary32 domain is a CI-only arithmetic
  proof. This source calls no engine API. Rust compares defined floating-point
  results and reports the original out-of-range integer-cast domain explicitly.
- `test-gbnf-validator.cpp`: the manual validator retains Windows text reading,
  acceptance tracing and source highlight behavior. Its zero fixed source
  assertions are explicit; 119 independent source grammar vectors add regression
  coverage. A CI-only diagnostic accepts grammar and input paths.
- Both tokenizer roundtrip sources traverse every token and the entire original
  Unicode domain in two SPM and seven explicitly selected BPE fixtures. The
  [versioned acceptance inputs](https://github.com/symbiogenesis/prism-inference/blob/main/tools/prism/upstream/BPE-ACCEPTANCE.md)
  preserve both optional-flag branches, including the corrected original
  Llama-BPE input with `--ignore-merges`; other named variants remain
  available as diagnostics with their actual verdicts. Per fixture this retains 456,703 Unicode calls and
  456,702 equality predicates, source exceptions, worker strides and shared stop.
  The BPE rewrite restores the source's intended invalid-UTF-8 skip with safe
  Rust validation: the pinned native decoder consumes errors that the original
  test expected to catch. Skip counts and this correction are reported explicitly;
  every valid token piece and original Unicode predicate remain required.
- `test-c.c`: the empty C main is retired. A C17 adapter includes the actual
  public header and exposes raw ABI layouts to seven additional Rust guards.
  The original contract is compile-only with zero behavioral assertions.
- `test-chat-analysis.cpp`: the full 37-template diagnostic, all paired render
  probes and undefined-variable analysis. Original accepted exceptions and zero
  correctness assertions are explicit; additional Rust diff guards are separate.
- Five model-lifecycle sources preserve cancellation, same-thread destruction,
  state-file continuation, sequence isolation and host/device state copies,
  fragmented-cache restoration, and concurrent model/context decoding. Their
  TinyStories fixture has pinned length and SHA-256 in
  [model-assets.lock.json](https://github.com/symbiogenesis/prism-inference/blob/main/tools/prism/upstream/model-assets.lock.json).
  Rust verifies its bytes before each schedule and controls individual native
  operations and progress callbacks through a bounded persistent process.
- `test-json-schema-to-grammar.cpp`: all 269 mandatory conditions and the
  original optional independent Python path, which adds 156 conditions when
  available. Python availability and skips are reported explicitly. Production
  Python conversion remains upstream code; inputs and comparisons are Rust.
- `test-tokenizer-0.cpp`: all 694 independent token-vector predicates across
  fifteen bundled vocabularies. Rust preserves Windows text input, map ordering,
  shared-context hardware workers and the original worker-zero comparisons.
  Optional timed text export is a separate CI-only diagnostic.
- `test-chat-auto-parser.cpp`: all seventeen groups and 524 executed conditions,
  including the source quote normalizer and 42 role-template rows.
- `test-pq2-row-shapes.cpp`: a committed input-only corpus preserves the
  original generated packed weights and activations for all 15 shapes. Rust
  independently decodes the packed weights, calculates reference products and
  checks the original relative-RMS threshold. The corpus contains no expected
  outputs; its provenance identifies the original archive and source digests.
  This path exercises pre-existing encoded weights; per-run quantizer checks
  remain in the retained `test-quantize-fns.cpp` control.
- `test-ptq1_0-element-map.cpp`: Rust generates the original seed-12345 LCG
  inputs across all 20,000 arbitrary-byte blocks, and independently computes
  the original CPU traversal. It compares all 2,560,000 positions against each
  engine's production decoder. Rejection of any input is red, including
  encodings that the Rust decoder intentionally does not admit.
- `test-unicode.cpp`: the exact original regex, input and two-piece expected
  vector are checked against production `unicode_regex_split`.
- `test-mtp-catchup-batch.cpp`: all six first-decode bounds assertions,
  including negative catch-up and zero-batch rejection, target the production
  speculative-decoding helper.
- `test-jinja.cpp`: all 19 default groups, including 278 literal renders,
  stateful hash properties, statistics, capabilities, string parts and all
  1,122 crash-resistance fuzz calls. Rust generates the inputs and checks the
  results. The Windows random stream was independently matched against MSVC;
  expected strings come from the original source. The optional Python mode is
  an upstream diagnostic cross-check, outside its default correctness suite.
- `test-mtmd-impl.cpp`: all five tiling cases and 12 temporal-grouping cases,
  with both group-size and mutated remaining-part assertions (29 in total).
- `test-grammar-parser.cpp`: all 23 variants, including six rejected grammars,
  ordered symbol IDs and flattened element types/values (274 original checks).
- `test-reasoning-budget.cpp`: all 71 original conditions for basic sequences,
  cloning, forcing, end matching and UTF-8 boundaries. Raw candidate float bits
  preserve finite-logit checks. Static transport traces preserve the original
  analysis break and may add an unobserved accept on a completed sampler before
  freeing it; no extra assertion coverage is claimed.
- `test-col2im-1d.cpp`: all 11 geometries across three precisions, with 33
  original NMSE checks and 22 shape checks.
- `test-dfly-fusion.cpp`: all three aggregate contracts, including full and
  layer-sliced outputs and eight layer separation norms.
- `test-rope.cpp`: the original five modes and ten relative-L1 checks over
  complete tensor extents, retaining the continuous Windows random stream.
- `test-llama-grammar.cpp`: all 174 original stack-frame and rejected-candidate
  comparisons. The source's unused expected-row tails remain recorded.
- `test-grammar-integration.cpp`: all 11 groups and 504 conditions, including
  408 literal string expectations, parser/build errors and 86 clone checks.
- `test-quantize-fns.cpp`: all 1,108 original predicates. The source's 144 enum
  slots, 37 active formats and 22 eligible quantizers are pinned independently of
  observed outputs, preserving genuine skips without hiding missing formats.
- `test-ptq1_0-cuda-dot.cpp`: all 53,192 original host-side predicates plus 14,096
  production decoder/CPU-dot checks. Unit scales preserve exact packed integer
  observations, while Rust retains the original floating scales and diagnostics.
  These host transcriptions do not qualify GPU execution.
- `test-log.cpp`: eight concurrent workers and all 8,000 messages, random
  timestamp/prefix changes, joins, flush and explicit logger destruction.
- `test-moe-weighted-reduce.cpp`: all 48 shapes/layouts, including weights inside
  the destination and destination inside the input. Rust owns manual placement
  and bitwise comparison against plain CPU allocation. CPU fallback is tested;
  CUDA fusion is not qualified.
- All six nested `peg-parser` sources: 170 builder workloads and 241 executed
  original conditions across basic parsing, grammar generation, JSON, Python
  dictionaries, serialization and Unicode. Malformed UTF-8 remains byte data.
  The umbrella `test-peg-parser.cpp` preserves group order and summary semantics.
- `test-batch-alloc.cpp`: all 26 named cases and 198 original assertions, covering
  initialization, splitting, tail retention and multidimensional positions.
- `test-alloc.cpp`: all 13 scenarios and 14 graph allocations, preserving 468
  unconditional predicates and actual overlap-dependent lifetime checks.
- `test-gguf.cpp`: all 34 crafted variants, all CPU file/buffer/callback
  roundtrips and KV-copy/merge checks. Source-only input parity was independently
  verified against MSVC for every crafted file, two generated contexts and RNG state.
- `test-sampling.cpp`: all 343 correctness conditions, preserving stateful
  candidate backing storage across filters, distributions, copying and resetting.
- `test-chat-template.cpp`: all 73 automated conditions, including 36 legacy
  renders and 35 Jinja renders, extra conversations and Windows newline handling.
- `test-chat-peg-parser.cpp`: all seven groups, 197 fixed assertions, conditional
  short-name checks and 105 diff crash-resistance conditions.
- `test-mtmd-c-api.c`: all 46 original conditions. The pinned library's public
  opaque fixture factory supplies chunk contents, as in the original source.
  Rust owns the invocation, traversal, serialization requests and comparisons.

The Rust runtime adapters use production tokenizer, Unicode, MTP, Jinja,
multimodal, grammar/schema/PEG, sampling, graph allocator, optimizer and codec
APIs. Focused implementation evidence and the remaining production families are
listed in [the deferred migration backlog](https://github.com/symbiogenesis/SwiftBonsai2Runner/blob/main/docs/prism-migration-backlog.md). Complete
hosted source, numerical and performance qualification remains separate;
unavailable operations continue to fail explicitly rather than testing substitute
helpers.

For each port, the same safe Rust inputs and assertions run against pinned Prism
first, then against the Rust runtime. A native process bridge performs bounded
serialization and calls production Prism functions. It imports no test sources
and contains no random generators, expected outputs, reference arithmetic,
tolerances or test assertions. The C++ exporters and original test executables
are no longer built. A separate C17 production ABI adapter proves public-header
compilation without retaining the original empty C test main.

Catalog runs include the pinned model-lifecycle workloads and the CI-only
barrier timing source. Foundation's
manual `prism_benchmarks` input executes separately reported benchmark workloads
after correctness. PEG serialization retains all 100 deserializations. Its
reported wall time includes native process transport and does not qualify native
kernel performance or affect the correctness denominator. Sampling retains five
32-iteration workloads; chat PEG retains complete/incremental parse workloads.
Their reports state which native dispatch and metadata handling is timed.
The GGUF test uses its supported explicit seed `12345` for repeatable inputs.

## Floor and retained evidence

The initial measured Rust baseline is **1 green / 71 source files (1.41%)**:
PQ2 row shapes pass, PTQ1 element mapping fails, and 69 source files are
unported. Equivalently the green:red count is **1:70**. The failed PTQ1 port
reports 19,434 rejected blocks out of the original 20,000; admitted positions
match exactly. The baseline retains this original domain instead of narrowing
the test to canonical bytes.

The initial [hosted Foundation run](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/40112751630)
passed on revision `afe66d881b8828183b9ffe7ef3119463b5d58bc2`. All **19 selected
C++ controls** passed (17 original suites plus the two paired adapter controls).
Their IDs are recorded in the baseline, so none can disappear from later runs.
The ported reference results now come from safe Rust assertions calling
the engine through the process bridge. The original GGUF qualification replay
was retired after its replacement passed native acceptance.
That initial run left 52 sources without an executed C++ control. The MTMD
Rust port adds another upstream control; it is separate from the original 19.

The first native Windows ARM64 acceptance run passed on revision `84ad97d`,
[Foundation run 40116792959](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/40116792959).
Its ClangCL bridges and Rust harness were verified as ARM64 PE files. All eight
ports published at that revision and 12 retained original controls passed on
Prism. Rust recorded one passing source, seven failed ports and 63 unsupported
sources. Later port registrations require their own hosted reference acceptance.

The next [Windows ARM64 run, 40118410264](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/40118410264),
passed on `18d2c18`: all 25 ports registered at that revision and two original
controls passed. Rust recorded one passing source, 24 failed ports and 46
unsupported sources. The baseline retains the original green source and all 19
required reference identities independently of these increasing port counts.

[Windows ARM64 run 40120815419](https://github.com/symbiogenesis/SwiftBonsai2Runner/actions/runs/40120815419)
passed on `2daa94c`: all33 Rust-owned source contracts passed on upstream,
including all694 tokenizer predicates and the complete GGUF replacement.
Rust recorded one passing source,32 failed ports and38 unsupported sources.
This provided the final evidence needed to retire the original C++ replay.

[baseline.json](https://github.com/symbiogenesis/prism-inference/blob/main/tools/prism/upstream/baseline.json) holds the recorded minimum
green count and every required green source ID. `RequiredReferencePassedIds`
also preserves each already-passing C++ control independently of Rust progress.
Its integer fraction is the
minimum green ratio. The exact catalog digest and upstream revision bind the
baseline to the denominator. Each run:

1. Verifies that the source inventory is complete and unchanged.
2. Requires all implemented adapters to emit one Rust and one C++ result.
3. Expands every remaining catalog file to `unsupported` for Rust.
4. Counts `fail`, `skip` and `unsupported` as red; only `pass` is green.
5. Requires every previously green ID to remain green, as well as the minimum
   fraction. A new pass cannot compensate for a regression in an existing pass.

Missing implemented adapters, duplicate/unknown IDs, malformed reports,
compilation errors, process failures and timeouts fail the pipeline. Expected
Rust conformance failures are recorded and tolerated only within the committed
floor. C++ reference failures always fail the gate.

Every run uploads `report.json`, `gate.json`, the exact source/runtime receipt,
separate C++ and Rust results, bridge requests/responses and compiler/test logs.
The job summary shows the full inventory with independent Rust and C++ statuses.
Git history records changes to the approved baseline; Actions artifacts retain
per-run observations under the repository's normal artifact retention policy.

The Windows ARM64 lane verifies the archive and cached source/adapter bytes
before configuring production libraries with ClangCL. Rust inputs and results
are fresh. Receipts record compiler arguments, binary hashes and ARM64 PE
identities. Ordinary workloads run in four independent groups: randomized
tokenization, thread safety, quantization statistics with LLGuidance, and every
remaining adapter. Each group retains its own result files and receipt. Fan-in
checks the revision, exact assigned source set and both targets for all four
groups before combining every checked backend partition and applying the final
unified source gate; no deferred result counts as passed. Native feature builds
may differ between groups. A separate build-only job supplies revision-bound,
hash-checked ARM64 backend tools, so backend planning and execution can overlap
every ordinary group. It uses the same verified source and native build cache
rules and produces no source verdict. Missing tools prevent backend execution;
a failed ordinary job still fails final Foundation acceptance, and the final
gate still requires all four ordinary groups and all backend partitions.

## Growing the baseline

Add an adapter using the original inputs, assertions and tolerances. Keep input
generation, reference calculations and assertions in safe Rust; extend the
native bridge only with production calls and serialization. Prove the
shared assertions on pinned C++ before applying them to Rust. Add its stable ID
to `AdapterIds`, so it cannot disappear into the unsupported backlog. Update
catalog identity and baseline binding together when this changes the catalog.

After a complete hosted run, review its retained report and increase
`MinimumPassed` and `RequiredPassedIds` to include the newly passing files.
Record the tested revision and run URL in `Provenance`. CI never rewrites the
baseline. Keep every existing required green ID. Review upstream pin changes
and any exception explicitly instead of deleting difficult cases.

The original C++ importer is retired; retain every Rust-owned source contract
and its upstream validation evidence.
Keep the full 71-source provenance inventory, including the explicit Apple
exclusion. Model tests need
their actual model/vocabulary/reference assets; platform tests need the relevant
host; embedded benchmarks need CI execution lanes. Missing prerequisites and
unsupported Rust features are red outcomes. Current-weight Swift artifacts
cannot substitute for unrelated upstream model fixtures.

## Focused local checks

Full catalogs and all benchmarks remain CI-only. A focused policy check is:

```powershell
./tests/Test-PrismBaseline.ps1 -Case Ratchet
```

One imported adapter can run locally with the pinned archive, locked Rust
dependencies and an available C++ toolchain. All paths below must be expanded
absolute paths outside Git:

```powershell
./tests/Test-PrismUpstream.ps1 -SourceArchive $archive `
    -EvidenceDirectory $freshEvidence -TargetDirectory $externalCargoTarget `
    -Case Pq2RowShapes -Generator 'Visual Studio 18 2026' -NativePlatform x64
```

The x64 C++ control can run under Windows ARM64 emulation; this is correctness
evidence and makes no target-device performance claim. Hosted acceptance uses native Windows ARM64. For Linux or a configured compiler environment, use `Ninja`.
