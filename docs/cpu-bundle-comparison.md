# Native CPU bundle comparison preparation

This lane prepares a source-bound comparison of the reviewed Prism 0.1.23
baseline, `055c7430cb4771dfddb24ddaa30494af6ad182da`, and one settled candidate.
The selected published candidate is registry `prism-inference =0.1.25`, with
embedded VCS identity `99bab71270b1e6a6fb6d47eec2e34b8097cdcd4c`. Its verified
archive and helpers are separate from the current 0.1.23 GitHub binary bundle.
It does not execute models, change Runner's release pin or qualify promotion.
The first comparison concerns the ordinary shared serving CPU bundle. Explicit
LUT, tiled/triangular prefill, prefix reuse and speculative paths remain outside
its exercised scope. One bundle result cannot attribute benefit to each change.

## Hosted preparation

The owner publishes the collector/preparation tooling first. From an owning
repository's main-only hosted Windows ARM64 workflow, check out the baseline
and the selected tooling source into separate pristine directories. Use the
same collector sources for both builds and an external artifact directory:

```powershell
python -B -m unittest discover -s "$tooling/.github/ci" -p test_cpu_bundle.py
python -B "$tooling/.github/ci/cpu_bundle.py" prepare `
  --baseline-root $baseline --tooling-root $tooling `
  --candidate-registry-version 0.1.25 --runner-revision $env:GITHUB_SHA `
  --output $packet
```

Preparation rejects local/self-hosted execution, changed source checkouts,
unexpected compiler/target and a baseline other than the reviewed revision.
Temporary standalone Cargo consumers use the pristine baseline source path and
the exact candidate registry dependency. Cargo verifies and extracts registry
archives; this tool never manually unpacks untrusted tar paths. It checks the
runtime 0.1.25 checksum `0a117db7589a821b27c224e41aaababa5ddd2ac9852344c0a8121a0849046363`,
`bonsai-neon-dot` 0.1.2 checksum
`f932c3309ef3645b055b91bdd9ee6986b7ef98c24bc1831701138e3ab9d3a31b`, and
`bonsai-native-hash` 0.1.3 checksum
`315284b1f3fc7e98f2cb0f749e63690853f1c441d2b3a3eec2ad68abe45180a2`, plus embedded
VCS identities before building. Before and after building, stream each verified
Cargo cache archive read-only and compare its regular-file bytes with the source
files Cargo extracted; never extract paths supplied by the archive.
They never transplant files into or edit the baseline. The same collector bytes
use the baseline's intrinsic FP16 loader under `cpu_bundle_legacy_f16`; the
candidate explicitly selects FP16. The fixed features are `cli`,
`experimental-neon` and `parallel-rows`; both compile with nightly-2026-11-09.
The generated consumer lock may select only versions/checksums already in the
corresponding source lock. No registry helper publication is needed for these
source-path baseline build. The newly published helper prerequisites are already
available for the selected registry candidate. An alternative source comparison
uses `--candidate-root` and `--candidate-revision` instead of the two registry-mode
options; it requires that exact candidate checkout to be clean.

Each arm requires strict Release analysis of the runtime and changed native
helpers, portable-library checking, the named graph state/row-worker control,
strict collector analysis, its focused model-free controls and production
linking. Failure stops preparation. These source controls do not establish
actual-model state/logit/adapter/cancellation parity. Owning hosted x64 fallback
and other affected source checks remain required independently.

The artifact contains `baseline.exe`, `candidate.exe`, `messages.json`,
`evaluator.py` and `preparation.json`. The identity receipt binds compiler,
source trees/manifests/locks, exact collector hashes, generated build-input
hashes, features/cfg, executable bytes/digests and successful named commands.
Its public fixture asks for the numbers one through ten, with a 32-token cap.
Preserve the original preparation receipt and verify its hosted source/run
identity independently before device intake; JSON hashes do not authenticate CI.

## One fixed device operation

Runner owns device process supervision and CI admission. This script has no
device execution command. The owner must predeclare one aggregate workload and
allowance, covering setup, all arms, every power observation, cleanup and selected
evidence retention. Existing narrower limits and the fifteen-minute ceiling
remain effective; preparation grants no new device allowance or retry.

Execute exactly four sequential fresh processes in order baseline, candidate,
candidate, baseline. Each arm uses the identical frozen public messages,
original PQ2, approved adapter at 1.3, four workers, production `SwiftSampler`,
seed 42, thinking disabled, FP16 and reset-per-request serving. A direct
`SwiftSession` with no projection executor establishes CPU-only construction.
Per-operation scalar versus NEON execution is unknown; neither backend request
labels nor this packet claim that every operation uses NEON.

The collector's seven arguments are absolute model path, approved adapter path,
new existing empty arm directory, absolute messages path, output cap, the arm's
absolute Unix-millisecond deadline, and its exact source revision. Bound that
deadline once from the original aggregate clock and the at-most-90-second arm
limit. The collector accepts at most 240 seconds remaining and reserves two
seconds for its final receipt; a 600-second aggregate deadline must not be
passed directly to the first arm. Native supervision must separately bound
blocking load, retain 15.5 GiB admission, 11.5 GiB Job ceiling and 4 GiB reserve,
and confirm owned teardown under the same aggregate deadline.

Capture `GetSystemPowerStatus` before and after the aggregate and each arm,
including AC, battery saver and available charge observations. Enable native
CPU-condition observations in the generic completion supervisor. Its
`ciOnly: false` selects raw diagnostic output rather than the publisher timing
grammar; the enclosing Runner workflow/script must independently enforce CI
ownership and must not manufacture CI environment variables.

## Offline admission and interpretation

```powershell
python -B "$packet/evaluator.py" evaluate `
  --preparation "$packet/preparation.json" --owner "$evidence/owner.json"
```

The owner envelope is schema 1, kind `prism-cpu-bundle-owner`. It records:

- The preparation-file SHA-256, approved model/adapter SHA-256, predeclared-work
  and completion flags, fixed budget seconds, monotonic clock frequency,
  original start/deadline ticks, original aggregate Unix deadline and the tick
  after evidence retention. Retention must finish before that original deadline.
- One complete observed native `host_identity`, aggregate `power_before` and
  `power_after`, and four `arms` with IDs `arm-1` through `arm-4` in the fixed
  role order. Each arm records start, execution-deadline and cleanup-complete
  ticks, its derived `collector_deadline_unix_milliseconds`, matching host and
  power endpoints, successful cleanup and confirmed empty owned Job.
- Each arm's `receipt`, `started_receipt` and `supervision` as objects containing
  a relative `path` and SHA-256. Paths resolve under the owner envelope directory;
  missing, reused, changed or escaping members reject admission. Retain original
  `cpu-bundle-started.json`, `cpu-bundle.json` and native `supervision.json` bytes.

Power objects use observed `GetSystemPowerStatus`, boolean `ac_online` and
`battery_saver`, and a numeric or explicitly unknown `battery_percent`.
Unknown AC/saver or mismatched endpoints reject timing admission. Raw supervisor
CPU snapshots must preserve known host, topology and effective power policy
throughout the selected endpoints; a supplied derived comparison flag is ignored.
Equal snapshots do not prove uninterrupted stability or measure energy.

Admission requires matching successful terminal accounting, exact returned text
and UTF-8 digest, positive matching prompt/output counts, approved profile and
source/binary/input identities, positive enclosed timings and successful native
resource/cleanup observations. Empty output, zero generated tokens, failed or
missing receipts and incomplete four-arm captures supply no benefit pass.

The primary observation is load-inclusive supervised process wall time. Report
each reversed-pair candidate/baseline ratio, preparation-plus-request ratios and
cleanup-inclusive owned-arm ratios separately. `short_request_benefit_observed`
requires both process-wall ratios at most 0.90 and both owned-arm ratios at most
1.10. Losing results remain admitted observations. Two short pairs supply no
statistical-significance or sustained-use result. A length-limited answer is not
reported as a complete answer. Every result keeps `benefit_gate_qualified`,
`promotion_qualified`, `DefaultMayChange` and per-change verdicts false. Existing
uncertainty, independent state, quality, lifecycle and release gates remain open.

Per-arm observations retain committed output count, load/preparation/request and
process wall times, optional first nonempty text callback and phase records,
Job commit peak and available process memory counters. Source-request TPS divides
committed output count by the whole `SwiftSession` completion duration, including
prefill and sampling. It is different from historical broker HTTP post-first-token
decode TPS. A text callback is not a model token timestamp or HTTP TTFT; phase
inventories may differ between source versions. Job commit and process residency
are separate overlapping domains, not total RAM. Publish README values only from
an actual successful capture under these explicit metric scopes.

Retain only the descriptor's `retained_files` selection (`preparation.json`,
`messages.json`, `evaluator.py`), the owner envelope and referenced raw arm proofs
for offline admission replay. The two executables are required during verified
device intake, not for later ratio arithmetic. Keep additional nonempty failure
diagnostics only when they diagnose the failure; do not archive source trees,
temporary Cargo consumers, blank logs or successful build chatter.
