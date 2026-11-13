# CI failure review

The review selected the newest 30 runs with conclusion `failure` from
`symbiogenesis/prism-inference`, ordered by creation time. The fixed snapshot spans
2026-11-12 10:25:20 UTC through 2026-11-13 19:13:46 UTC: 17 Foundation runs, eight
Hexagon boundary runs, four Foundation bundle runs and one Source checks run.
All 30 job inventories and failed-step logs were available and read. Later failures
are outside this snapshot. Cancelled runs and individual failed jobs are not separate
entries in this run-level sample.

The repeated shard rejection is a CI inventory false positive: optional or unrelated
workspace feature forwarding is treated as proof that a selected default test needs
an unresolved feature. Fourteen runs contain it, often alongside genuine failures.
One run also exposes a stale exact-output fixture after useful CLI diagnostics were
added. Compiler, lint, formatting and documentation failures remain real source
failures under the repository's strict contracts. The sample supplies no reason to
retry whole tests automatically, weaken lint policy or accept incomplete coverage.

## Run inventory

Findings use these codes:

- **S**: conservative dependency-feature scan rejects the correctness inventory.
- **J**: exact CLI fixture omits graph identity, cache policy and state memory.
- **D**: unresolved `Decoder::ingest_block` rustdoc link in `src/graph/prefill.rs`.
- **C**: Swift state constant/re-export visibility, dead-code or import lints.
- **U**: `unused_self` in the feature-disabled sibling-matrix implementation.
- **T**: strict test lints: empty assertions, raw-pointer borrow or unfulfilled expectation.
- **E**: CPU bundle example lints: unchecked deadline subtraction and event ownership.
- **N**: ARM64 helper takes a trivially copied array by reference.
- **F**: Rust formatting differences.
- **M**: Markdown line-length violation.

| Run | Workflow | Source revision | Findings |
| --- | --- | --- | --- |
| [43490203477](https://github.com/symbiogenesis/prism-inference/actions/runs/43490203477) | Foundation | `1429a33772089dfd3a50a96c698b9dbe0b36836a` | S, D |
| [43489026858](https://github.com/symbiogenesis/prism-inference/actions/runs/43489026858) | Bundle | `40118969e69d490634a9cd480ea68bd69758bfc8` | C |
| [43488946876](https://github.com/symbiogenesis/prism-inference/actions/runs/43488946876) | Hexagon | `40118969e69d490634a9cd480ea68bd69758bfc8` | C |
| [43488946565](https://github.com/symbiogenesis/prism-inference/actions/runs/43488946565) | Foundation | `40118969e69d490634a9cd480ea68bd69758bfc8` | C, S, D |
| [43488012054](https://github.com/symbiogenesis/prism-inference/actions/runs/43488012054) | Bundle | `45724e50cbcfee309f558d08d9e5fd99e9518eef` | C |
| [43487921415](https://github.com/symbiogenesis/prism-inference/actions/runs/43487921415) | Hexagon | `45724e50cbcfee309f558d08d9e5fd99e9518eef` | C |
| [43487921405](https://github.com/symbiogenesis/prism-inference/actions/runs/43487921405) | Foundation | `45724e50cbcfee309f558d08d9e5fd99e9518eef` | C, S |
| [43487080842](https://github.com/symbiogenesis/prism-inference/actions/runs/43487080842) | Bundle | `52af7e4db462f4ef59bfc8e0489125a2e6bb4188` | F |
| [43486884103](https://github.com/symbiogenesis/prism-inference/actions/runs/43486884103) | Hexagon | `52af7e4db462f4ef59bfc8e0489125a2e6bb4188` | F |
| [43486884116](https://github.com/symbiogenesis/prism-inference/actions/runs/43486884116) | Foundation | `52af7e4db462f4ef59bfc8e0489125a2e6bb4188` | F, C, S |
| [43480372939](https://github.com/symbiogenesis/prism-inference/actions/runs/43480372939) | Foundation | `ac941f2668ae93dba9824002b4597081457aa09f` | S, D |
| [43476952268](https://github.com/symbiogenesis/prism-inference/actions/runs/43476952268) | Foundation | `3d735627df43056d3bf569002af2c23af30a3e6a` | S, D |
| [43475906736](https://github.com/symbiogenesis/prism-inference/actions/runs/43475906736) | Foundation | `a578fbc5ce6ede60bb5e15a03af98d31a3bc3541` | S, T |
| [43459959206](https://github.com/symbiogenesis/prism-inference/actions/runs/43459959206) | Hexagon | `4cd754c99b99502e8b3d6f2b6b017baf2ac5ebc6` | U |
| [43459959192](https://github.com/symbiogenesis/prism-inference/actions/runs/43459959192) | Foundation | `4cd754c99b99502e8b3d6f2b6b017baf2ac5ebc6` | S, U, T |
| [43455360376](https://github.com/symbiogenesis/prism-inference/actions/runs/43455360376) | Foundation | `e40d30c163403b7ffc5a4d506655f70c0b0a7d15` | S, U, T |
| [43455360688](https://github.com/symbiogenesis/prism-inference/actions/runs/43455360688) | Hexagon | `e40d30c163403b7ffc5a4d506655f70c0b0a7d15` | U |
| [43455360151](https://github.com/symbiogenesis/prism-inference/actions/runs/43455360151) | Source checks | `e40d30c163403b7ffc5a4d506655f70c0b0a7d15` | M |
| [43370473736](https://github.com/symbiogenesis/prism-inference/actions/runs/43370473736) | Foundation | `bb8d30d1042377060c39525eff5c57769523e78d` | S, U |
| [43369035808](https://github.com/symbiogenesis/prism-inference/actions/runs/43369035808) | Foundation | `0a57f49cd105af63722360661a6a69ed60565950` | S, U |
| [43243562670](https://github.com/symbiogenesis/prism-inference/actions/runs/43243562670) | Foundation | `ead6ededefe14cd6147a250be08e25ffa66914d4` | S, U |
| [43239852828](https://github.com/symbiogenesis/prism-inference/actions/runs/43239852828) | Bundle | `f9b5eb2f61bbb53d2129cededb3434c80b8dbcf9` | E |
| [43235944827](https://github.com/symbiogenesis/prism-inference/actions/runs/43235944827) | Foundation | `f9b5eb2f61bbb53d2129cededb3434c80b8dbcf9` | S, U |
| [43231093290](https://github.com/symbiogenesis/prism-inference/actions/runs/43231093290) | Hexagon | `d31a196cc68030723297a1440945c0c401145f35` | U |
| [43231092924](https://github.com/symbiogenesis/prism-inference/actions/runs/43231092924) | Foundation | `d31a196cc68030723297a1440945c0c401145f35` | S, U |
| [43227626826](https://github.com/symbiogenesis/prism-inference/actions/runs/43227626826) | Foundation | `8d2b3067e5f17369f579857a180c01a422a58211` | U, J |
| [43227626785](https://github.com/symbiogenesis/prism-inference/actions/runs/43227626785) | Hexagon | `8d2b3067e5f17369f579857a180c01a422a58211` | U |
| [43220096954](https://github.com/symbiogenesis/prism-inference/actions/runs/43220096954) | Hexagon | `85048102d133dc649029ca2d0e49910814681410` | U |
| [43220096847](https://github.com/symbiogenesis/prism-inference/actions/runs/43220096847) | Foundation | `85048102d133dc649029ca2d0e49910814681410` | N, U, T |
| [43200223412](https://github.com/symbiogenesis/prism-inference/actions/runs/43200223412) | Foundation | `a5ee5b6d47fb0a2031c3e6bd410a875f0afbf0a7` | D |

## Improvements selected from the evidence

1. Resolve the actual selected Cargo unit graph and target before sharding in
   [the shard planner](../.github/ci/foundation_shards.py).
   Inventory every enabled test exactly once, retain disabled-target reasons, and
   reject malformed or incomplete resolution. Optional forwarding alone must not
   make an ordinary default-feature test inventory fail.
2. Keep the [CLI fixture](../tests/graph.rs)'s full exact JSON assertion and add
   the diagnostic fields.
   Assert the fixed fixture's graph identity, FP16 policy and memory geometry.
   Obtain only layout-sensitive live storage from a separate scalar decoder, so
   compiler layout changes do not stale the protocol fixture.
3. Qualify the rustdoc link through the crate's public decoder trait. Keep strict
   documentation and cfg-sensitive lint gates enabled; historical lint fixes
   already present in the reviewed source need no additional suppression.
4. Use [the command recorder](../.github/ci/run_ci_command.py) to stream exact
   release-control output as each control executes and name its
   command, feature selection and exit result. Retain failure context even
   when an early compiler or control failure prevents bundle creation.
5. Surface source/run/attempt, runner image, compiler, target, selected packages,
   feature flags and lane alongside failures. Distinguish the failing command
   from downstream aggregate rejection and preserve the original nonzero result.

These changes preserve the original correctness obligations and the bundle lane's
limited qualification scope. They add no model run, benchmark, push or CI dispatch.
Hosted acceptance on the final integrated source remains owner-operated.

## Evidence and validation limits

The run links above retain independent failure provenance and exact source
identities. This concise inventory replaces a duplicate raw-log archive; transient
job metadata and logs are review inputs, not new replay fixtures. Historical power
conditions were not captured and remain unknown. These hosted failures establish
no owner-device performance comparison.

Focused local checks and their results are recorded in the delivery message.
Local source checks do not establish a hosted CI verdict or qualify a release,
actual-model correctness or performance. Full suites remain CI-only.
