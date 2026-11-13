# CPU bundle comparison

This protocol compares one frozen baseline with one reviewed candidate for the
ordinary CPU serving path. Explicit LUT, tiled/triangular prefill, prefix reuse
and speculative paths are outside its scope. A bundle comparison cannot
attribute benefit to individual source changes.

## Hosted preparation

The [preparation tool](../.github/ci/cpu_bundle.py) defines the accepted baseline,
registry checksums and source identities. Use the same reviewed collector sources
for both arms in pristine hosted Windows ARM64 checkouts. Its focused
[controls](../.github/ci/test_cpu_bundle.py) must pass before preparation.

Preparation verifies source cleanliness, compiler/target, locked dependencies,
registry archive bytes against Cargo's extracted sources, and generated consumer
locks. It never modifies the baseline. Source candidates require an exact clean
revision; registry mode accepts only the tool's frozen package set.

Both arms use `cli`, `experimental-neon` and `parallel-rows`, with the pinned
compiler. The baseline's intrinsic FP16 loader and candidate's explicit FP16
selection are matched. Strict runtime/helper/collector analysis, scalar compilation,
named graph and collector controls, and production linking must succeed.
Test receipts must contain the exact positive inventory; exit zero with missing,
ignored or different tests rejects preparation.

The packet contains `baseline.exe`, `candidate.exe`, `messages.json`,
`evaluator.py` and `preparation.json`. Its receipt binds sources, compiler,
locks, collector/build inputs, features, executable hashes and named commands.
The public fixture asks for the numbers one through ten with a 32-token cap.
Verify the hosted source/run identity independently before intake; JSON hashes
alone do not authenticate the build.

## One fixed device operation

Runner owns native supervision and admission. Predeclare the complete workload
under [owner-device policy](../AGENTS.md#owner-device-execution-budgets);
preparation grants no device allowance. The fixed plan uses one 600-second
aggregate clock: four arms reserve 110 seconds each, bootstrap/intake sixty,
and final cleanup/evaluation/retention one hundred. Stop on failure without retry.

Run four fresh processes in order baseline, candidate, candidate, baseline.
Every arm uses the frozen messages, original PQ2, approved adapter at 1.3,
four workers, SwiftSampler seed 42, thinking disabled, FP16 and reset-per-request
serving. A direct SwiftSession without a projection executor establishes CPU-only
construction; it does not establish which individual operations use NEON.

Collector arguments are absolute model path, adapter path, a new empty arm
directory, messages path, output cap, absolute Unix-millisecond deadline and
exact source revision. Derive each deadline from the original aggregate clock
and the sixty-second collector limit; never pass the aggregate deadline directly.
Entry permits at most 240 seconds remaining and reserves two seconds for its
receipt. The generic supervisor permits ninety seconds of completion and
ninety-five seconds total; the collector deadline remains sixty.

Native supervision separately bounds blocking load, retains 15.5 GiB admission,
an 11.5 GiB Job ceiling and 4 GiB reserve, and confirms teardown within the
aggregate deadline. Capture GetSystemPowerStatus before/after the operation and
each arm: AC, battery saver and available charge. Enable native CPU-condition
observations. `ciOnly: false` selects raw diagnostic output; the enclosing owner
must enforce CI ownership without manufacturing CI environment variables.

## Offline admission and interpretation

Evaluate the original preparation receipt and schema-1
`prism-cpu-bundle-owner` envelope with the packet's evaluator. The
[validator](../.github/ci/cpu_bundle.py) is authoritative for field names and
acceptance predicates. Preserve:

- Preparation, approved model/adapter and input hashes; fixed budget, original
  monotonic/Unix deadlines, clock frequency and retention-complete tick.
- Observed host identity and aggregate/arm power endpoints, with IDs
  `arm-1` through `arm-4` in the fixed role order.
- Arm start/deadline/cleanup ticks and the derived collector deadline.
- Original `cpu-bundle-started.json`, `cpu-bundle.json` and
  `supervision.json`, referenced by confined relative paths and SHA-256.

Admission requires complete successful accounting, matching text/digests and
positive prompt/output counts, approved identities/profile, enclosed timings,
and confirmed empty owned Jobs. Missing, changed, escaping or reused receipts
reject admission. Unknown AC/saver or mismatched power/CPU conditions reject
timing qualification. Equal endpoints do not prove uninterrupted stability.

Report load-inclusive process wall time for each reversed candidate/baseline
pair, preparation-plus-request ratios and cleanup-inclusive arm ratios separately.
`short_request_benefit_observed` requires both process-wall ratios at most 0.90
and both owned-arm ratios at most 1.10. Retain losing outcomes. Two pairs establish
neither statistical significance nor sustained-use benefit. Promotion, default
changes and per-change benefit remain unqualified.

Source-request TPS divides committed output by complete SwiftSession duration,
including prefill and sampling. A text callback is neither a token timestamp nor
HTTP TTFT. Job commit and process residency overlap and must not be summed as
total RAM. A length-limited response is not a completed answer.

Retain the descriptor's selected preparation files, owner envelope and referenced
raw arm proofs for independent replay. Executables are needed for verified
device intake, not ratio arithmetic. Keep extra diagnostics only for actionable
failures; discard scratch source trees, generated consumers and build chatter.
