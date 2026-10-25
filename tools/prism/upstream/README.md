# Safe Rust assertions against pinned Prism

The `upstream_prism` example owns test inputs, references and assertions in
safe Rust (`#![forbid(unsafe_code)]`). It uses the same test code with two
targets: production Prism through a native process bridge, and this repository's
production Rust runtime. Both results are retained for each source; every
upstream reference failure fails the final gate.

The current ports preserve all seventy Windows-applicable source contracts from
`PrismML-Eng/llama.cpp@b87ca7225456b06b0975b8239c66b5aecf07e37c`:

- Backend sampling retains all19 original scenarios, complete penalties/filter
  comparisons, multi-output transactions and CPU-suffix observers. Callback
  decisions and source predicates belong to Rust; native callbacks only relay
  raw events. Safe Rust reproduces the original MSVC double RNG exactly.
- DSpark forward retains all six Windows CPU modes and their runtime/prefix
  grids, A/B checks, replays, capture atomicity and rejection contracts. Tier2
  preserves diagnostic-only numeric differences against independent Rust math.
- DSpark loop retains all seven acceptance cases,49 exact token comparisons and
  95 primitive predicates. Independent safe Rust math follows pinned published
  DeepSpec equations; expected tokens are computed before production opens.
- Real DSpark evaluation retains all24 prompts, plain AR/speculative parity and
  rollback refusals, diagnostic teacher forcing/KLD and source statistic rollups.
  The small compatible CI pair is generated in safe Rust. External pairs/JSONL
  remain selectable. All executions are CI-only; bridge timings carry no model
  quality or publisher throughput claim.
- LLGuidance retains 42 grammar/schema groups, 344 literal predicates and all
  dynamic token/EOS chain checks. Its reference enables the pinned production
  dependency rather than accepting the source's disabled-feature skip.
- DSpark log-SNR metadata retains all eight checkpoint variants and fourteen
  original loader/log predicates. A safe Rust writer reproduces every original
  input file byte for byte, using a pinned input-only tensor corpus.
- GGUF metadata helper coverage includes all 34 original predicates in safe
  Rust. Quantization selection calls production APIs for eleven models and all
  363 original expected schema sections. Inputs and expected schemas remain
  independent; source fixture-name corrections are reported explicitly.
- The barrier source retains all three graph/threadpool schedules. It has zero
  output assertions; reports distinguish workload completion and additional API
  guards from numerical coverage. All its executions, including named cases,
  are CI-only because the original source measures timings.
- Architecture coverage retains every source variant and backend/export phase;
  Rust generates the small model inputs and records capability skips. Recurrent
  rollback creates its own three fixtures and retains every replay/fork schedule.
- Full chat, argument parsing and model resolution retain all source cases and
  optional diagnostic controls. Resolution uses a Rust loopback fixture server.
- KV mean centering and optimizer schedules use Rust-owned metadata, tensors,
  random inputs, callbacks and numeric predicates. Native model construction and
  optimizer dispatch call production APIs only.
- The binary32 arithmetic proof runs only in CI and is explicitly independent
  of engine APIs. The manual grammar validator preserves its zero fixed source
  assertions; independent grammar vectors provide additional regression checks.
- Both tokenizer roundtrip sources retain every vocabulary token and the entire
  source Unicode domain across two SPM and seven selected BPE fixtures. The
  [versioned input selection](BPE-ACCEPTANCE.md) preserves both flag branches;
  original incompatible variants remain explicit diagnostics. Rust owns exclusions, exceptions, worker strides,
  stop policy and comparisons; native batches contain supplied inputs only.
  Safe Rust restores BPE's intended malformed-UTF-8 skip because the pinned
  decoder consumes its own exceptions. Reports count the corrected skips;
  valid token pieces and all Unicode comparisons remain required.
- `tests/test-c.c`: compile-only C17 public-header coverage via raw native ABI
  observations and additional Rust layout guards; no C test main remains.
- `tests/test-chat-analysis.cpp`: all37 diagnostic templates and render/stat
  probes; source-accepted exceptions and zero source assertions are explicit.
- Five model-lifecycle sources use the externally stored, hash-pinned
  TinyStories fixture: cancellation, worker-thread destruction, state-file and
  sequence restoration, fragmented cache, and concurrent contexts. Rust owns
  every request, callback decision and comparison. The persistent native
  session only dispatches production operations on bounded worker lanes.
- `tests/test-json-schema-to-grammar.cpp`: all 269 mandatory conditions,
  plus 156 independent Python comparisons when the original availability
  rules enable them; skip status remains visible.
- `tests/test-tokenizer-0.cpp`: all 694 independent token-vector predicates
  across the fifteen bundled vocabulary fixtures, including shared-context
  hardware workers and the original worker-zero comparisons. Fixtures remain
  in the verified external source archive. Optional timed text export uses
  a separate CI-only `--diagnostic` configuration.
- `tests/test-chat-auto-parser.cpp`: all seventeen groups, including 524
  executed conditions, the source quote normalizer and 42 role templates.

- `tests/test-pq2-row-shapes.cpp`: all 15 shapes, `K` in
  `{128,256,384,512,640}`, `M=4`, `N` in `{1,4,16}`, and the original
  `rel_rms(computed, dequantized-reference) < 0.05` assertion. The frozen
  input-only corpus preserves original packed weights and activations from
  the recorded hosted baseline. Rust independently decodes weights and computes
  references; it never reads a saved native output as the engine under test.
- `tests/test-ptq1_0-element-map.cpp`: the original seed-12345 LCG, all 20,000
  arbitrary-byte blocks and all 2,560,000 element equality assertions. Rust
  generates the bytes and computes the independent CPU traversal. Scale is
  exactly one because the original integer test leaves it unused. Noncanonical
  inputs remain in the test; the Rust runtime's rejection remains a failure.
- `tests/test-unicode.cpp`: the exact original regex, input and expected pieces.
- `tests/test-mtp-catchup-batch.cpp`: all six original first-decode bounds,
  including zero-batch and negative catch-up rejection.
- `tests/test-jinja.cpp`: every default literal, hash, stats, caps, string-parts
  and fuzz group. Expected literals are source data; all generators and
  assertions are Rust. Stateful hash traces preserve intermediate digest calls.
- `tests/test-mtmd-impl.cpp`: five tiling cases and 12 grouping cases, including
  both grouping outputs and the mutation of the remaining input parts.
- `tests/test-grammar-parser.cpp`: all 23 parse/reject variants and 274 original
  symbol-map and rule-element checks, including token and token-not elements.
- `tests/test-reasoning-budget.cpp`: all 71 original sequence, cloning, forcing,
  end-match and UTF-8 conditions, retaining candidate float bits and the original
  analysis break. Static transport may accept one unobserved completed token
  before freeing a sampler, without claiming additional assertion coverage.
- `tests/test-col2im-1d.cpp`: all 11 geometries at three precisions, preserving
  the 33 original NMSE thresholds and 22 shape checks.
- `tests/test-dfly-fusion.cpp`: all three aggregate contracts, covering the full
  output, layer slices and eight layer separation norms.
- `tests/test-rope.cpp`: all five modes and ten relative-L1 checks over the
  original tensor extents and continuous Windows random stream.
- `tests/test-llama-grammar.cpp`: all 174 original stack-frame and rejected
  candidate pair comparisons. Unused expected-row tails remain source data.
- `tests/test-grammar-integration.cpp`: all 11 groups and 504 original conditions,
  including schema conversion, parser failures, recursive roots and cloning.
- `tests/test-quantize-fns.cpp`: all 1,108 original predicates, preserving
  source-pinned quantizer eligibility, F32 tails and packed ternary dot patterns.
- `tests/test-ptq1_0-cuda-dot.cpp`: all 53,192 host-side predicates, with 14,096
  additional checks against production CPU decoding and packed dots. This source
  uses host transcriptions; it establishes no GPU execution coverage.
- `tests/test-log.cpp`: all eight concurrent workers and 8,000 messages,
  timestamp/prefix changes, joining, flushing and explicit singleton destruction.
- `tests/test-moe-weighted-reduce.cpp`: all 48 cases, preserving manual aliased
  buffer layouts and bitwise comparison against plain CPU allocation. Acceptance
  uses the source's CPU fallback; CUDA fusion remains unqualified.
- All six `tests/peg-parser/` sources preserve their 170 builder workloads and
  241 executed original conditions, including malformed UTF-8 byte inputs. The
  umbrella `tests/test-peg-parser.cpp` preserves its group order and summary.
- `tests/test-batch-alloc.cpp`: all 26 named cases and 198 assertions, including
  invalid inputs, sequence splitting, tail retention and multidimensional positions.
- `tests/test-alloc.cpp`: all 13 scenarios, 14 graph allocations, 468 unconditional
  predicates and every overlap-dependent lifetime check, including multi-buffer
  external allocations and reuse of one allocator across graphs.
- `tests/test-gguf.cpp`: all 34 crafted variants, all CPU file/buffer/callback
  roundtrips and metadata copying/merging. Input-only MSVC parity covers every
  crafted byte file, both generated contexts and subsequent RNG state.
- `tests/test-sampling.cpp`: all 343 correctness conditions. Candidate backing
  storage remains stateful across filters, distributions, copies and resets.
- `tests/test-chat-template.cpp`: all 73 default automated conditions, including
  36 legacy renders and 35 Jinja renders with Windows newline normalization.
- `tests/test-chat-peg-parser.cpp`: all seven groups, 197 fixed assertions,
  conditional name checks and 105 diff crash-resistance conditions.
- `tests/test-mtmd-c-api.c`: all 46 original conditions. Rust explicitly invokes
  the public upstream opaque fixture factory; that production function supplies
  chunk contents. Rust owns traversal, save/load requests and every predicate.

Unicode, MTP, Jinja, MTMD and grammar use a [common bridge](common/PROTOCOL.md).
The Rust runtime has no equivalent production APIs; their Rust results fail.

The native bridge contains only serialization, input validation and production
Prism calls. It does not include upstream test sources or contain generators,
references, expected values, assertions or tolerances. The C++ fixture exporters
and original source files are no longer compiled as tests. The original importer
and replay build were removed after all33 Rust-owned contracts, including GGUF,
passed [native Windows ARM64 CI](https://github.com/symbiogenesis/SwiftBonsai2/actions/runs/40120815419).

The immutable source archive is unchanged. The pinned saver omits
DSpark metadata and global tensors. The reference applies explicit,
hash-verified production corrections in `reference-patches.lock.json`.
The saver correction preserves the metadata and sixteen tensors. The Llama 3
correction disables inherited punctuation cleanup for its three canonical
metadata names, preserving decoded token bytes and the original single-token
roundtrip assertion. Only verified external translation units are substituted;
the archive and every original test
predicate remain unchanged. Receipts identify this corrected reference variant.

The runtime's vector API runs once for each original activation column,
preserving the complete `M*N` comparison. This tests numerical compatibility,
without claiming batched execution or throughput equivalence.

The metadata fixture lock pins eleven model selections at immutable publisher
revisions. GGUF inputs are bounded raw prefixes, downloaded with exact HTTP
range, size and SHA-256 checks. Raw API listings are an input-only corpus because
their activity fields change even at revision URLs. No header input is derived
from expected schemas. The lock explicitly records the upstream Nemotron
repository-name correction. Assets are materialized only in external evidence.

From a verified, unmodified source extraction, configure this directory with
CMake and `-DPRISM_SOURCE_DIR=<absolute upstream path>`, then build the
`prism-upstream-bridge` target. Run the safe Rust suite with:

```text
cargo run -p bonsai-prism-runtime --example upstream_prism -- \
  --target cpp --case test-pq2-row-shapes \
  --bridge <absolute native bridge executable> \
  --evidence-directory <fresh absolute evidence directory> \
  --output <absolute report path>
```

Repeat with `--target rust`, omitting the bridge options. The PTQ1 selector is
`test-ptq1_0-element-map`. For `test-unicode` or `test-mtp-catchup-batch`, use
`--common-bridge` with the executable built from `common/` instead of `--bridge`.
`tests/Test-PrismUpstream.ps1` supervises both targets,
retains bridge requests/responses and logs, and checks every upstream result in
the final gate. Full catalogs and all benchmarks run only in CI.
The Foundation manual input `prism_benchmarks` runs the ported benchmark paths
after correctness. PEG deserialization retains its 100-iteration workload;
sampling retains five 32-iteration workloads; chat PEG retains complete and
incremental parse workloads. Each report states its timing boundary and has no
correctness threshold.

Reports preserve `SchemaVersion: 1`, `UpstreamCommit`, `Target`, and `Cases`
containing `Id`, `Status` and `Detail`. Measured conformance failures are data for
the baseline gate; infrastructure failures exit nonzero. The excluded Apple source remains
unsupported in the 71-source denominator.

Native acceptance targets Windows ARM64 for Snapdragon X Elite. The Apple-only
wired-memory source remains inventoried as an explicit exclusion. The complete
71-source migration inventory is tracked in the
[baseline guide](../../../docs/prism-upstream-tests.md). These derivatives retain
the upstream MIT attribution in [LICENSE](LICENSE).
