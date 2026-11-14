# Prism private inference server

`prism-inference-server` serves authenticated HTTP on numeric IPv4 loopback after
loading the exact Swift model. The owning launcher verifies executable/model
bytes and retains credential guards, process/Job ownership, memory/watchdog policy,
authenticated readiness and exclusive listener-PID checks. The crate is
`publish=false`; build it from the exact checkout.

## Startup profiles

Original PQ2 without an adapter is the default. `--use-repacked-ptq1` selects the
verified current-weight export; `--abliterate-lora <absolute-path>` loads the
approved rank-one adapter. See [adapter semantics](../docs/rank-one-adapter.md).

K4/V4 `rsq4` is the source development KV default; `--kv-cache f16` selects the
comparison/recovery profile and `rsq4k3v` is opt-in. Requests cannot select
precision, and failures never change it. Non-FP16 readiness reports `kv_cache`
and `startup_graph_identity`, both of which strict owners must validate.
See [KV profiles and accounting](../docs/quantized-kv.md).

Paired `--context-tokens`/`--memory-limit-bytes` options select immutable capacity
before model I/O. Context, exact tokenization, request admission and authenticated
`context_tokens` readiness must agree. Partial, duplicate, unaligned, overflowing
or underfunded options reject. Owners enforce the selected Job allowance and
monitor pressure; observed free memory is not a reservation. The
[memory contract](../docs/runtime-contracts.md#memory-and-ownership) defines current
context ranges and per-profile charges. Without those options,
the server uses 8,192 tokens and 16 GiB startup admission.

Explicit CPU prefill requires `--cpu-prefill-tokens <8|16|32>` and
`--cpu-prefill-scratch-bytes`, paired capacity options and explicit `--kv-cache`.
The selected cache and full scratch charge are retained, and readiness binds tile,
reserve and graph identity. Sampled suffix generation additionally requires
`--cpu-speculative-tokens <1..8>`, `--cpu-verification-scratch-bytes` and
`--cpu-speculative-checkpoint-bytes`. Readiness binds every reserve and
`:cpu-tiled-verify-v1`; phase-timing capture rejects that selection.
See [CPU prefill and suffix options](../docs/cpu-tiled-prefill.md).

Read-only model/adapter/key validation uses the exact Runner credential-helper
dependency. AppContainer checks preserve local-drive, non-root ancestry, reparse
and regular-file restrictions; the owner validates full ancestry before granting
only selected files. The key remains held against writes/deletion.

## Requests and readiness

`/v1/models` and `/props` identify model ID/hash, weight representation, approved
adapter hash or null, scale support, configured `row_workers` (1..16), and selected
startup policies. Worker count is a budget, not measured parallel utilization.
Requests cannot override it. Strict owning brokers must validate the complete
selected profile before forwarding work.

Adapter requests require exactly `lora: [{"id": 0, "scale": 1.3}]`, with a finite
JSON scale in 0..2 before binary32 conversion. Missing/null/extra slots, other IDs
and overrides reject. No-adapter requests reject `lora`. Public
`abliteration_scale` belongs to the outer broker and is rejected here.

Exact count/completion canonical bytes bind scale bits, including signed zero.
Prepared input cannot survive reset or a replacement count. Requests recompute
from fresh model state unless latest-prefix reuse is explicitly enabled.
See [session profiles](../docs/swift-session-profiles.md) for scale changes,
prefix eligibility, sampler progression, cancellation and reset semantics.
Ordinary private HTTP counting and generation have no total elapsed-time cutoff.
Header/body intake remain bounded to 5/30 seconds, and each response write retains
its 5-second allowance. Explicit library/diagnostic deadlines and owner-process
lifetime limits still apply; Stop/disconnect cancellation remains active during
prefill and generation. JSONL worker timeout settings are unchanged.

One acknowledged event bounds provisional streaming; failed delivery or
cancellation closes admission until explicit reset.

`chat_template_kwargs.enable_thinking` accepts either boolean value in exact
counting and generation. Disabled thinking closes the reasoning block before
generation; generated text is visible answer content. Supplied assistant reasoning
history is preserved. Enabled thinking uses `xhigh`; other template/sampler
overrides reject. Rust `prepare_with_thinking` selects this mode explicitly;
ordinary `prepare` and the JSONL worker enable thinking.

Startup failure emits one value-free diagnostic:
`private-inference-startup-failed stage=arguments errorKind=PermissionDenied osCode=5`.
Stages are `arguments`, `environment`, `memory`, `key`, `model`, `runtime`,
`listener`, `backend` or `unknown`; absent numeric OS codes use `none`.
Diagnostics omit paths, arguments, credentials, model values and original error
chains. The launcher retains the line before temporary storage cleanup.

## Service correctness capture

The owner-operated [acceptance workflow](../.github/workflows/private-service-acceptance.yml)
packages source/run-bound `private-service-check.exe` and `private-service-owner.exe`
after hosted model-free controls. Default preparation runs no model. Explicit
device acceptance requires the external owner's ephemeral run-specific runner
label and complete `BONSAI_NATIVE_OWNER_*` source/run/deadline binding.

This workflow has a fixed 210-second work cutoff, 240-second owner process lifetime
and at most 60 further seconds for intake: five minutes total, including setup,
package verification, model loading and cleanup. No retry extends it. Follow
[owner-device budgets](../AGENTS.md#owner-device-execution-budgets), including power
observations. Narrower checker/session limits still apply.

The trusted owner retains the exact server archive/members, original model,
approved adapter, seeded sampler, explicit row-worker count and disabled prompt
reuse, using the reviewed Runner launcher and its sandbox/Job/listener controls.
The checker starts no process; invoke it outside Git against that owned test
service with a fresh absolute output path and the remaining allowance:

```powershell
& $checker $privatePort $privateKeyPath $newEvidencePath $remainingCheckSeconds
```

The port must be numeric loopback; the key stays in its protected file. Never
target an independently running daily service: the checker resets state and
abandons output. Its fixed workload checks readiness, canonical counts and usage
at scales 1.3/zero, invalid scale, stale count/reset, provisional-stream disconnect,
closed admission and seeded reset/replay. Completed observations flush before the
next step; failures retain partial results. Responses/credentials are excluded.

Only `host.json`, `hosted-build.json`, `owner.json` and `service.json` are public
retention/upload candidates. Staging verifies schema/source/run, non-reparse
external ancestry and held-file identities, with 64 KiB per file and create-new
destinations under `%LOCALAPPDATA%\SwiftBonsai2\qualification`. It never copies
extra files or protected child directories. Missing/malformed receipts cannot
become success; failed/pending receipts retain their status. Success additionally
requires an empty owned Job and released listeners. Retain the checker package,
tested server identities and owner artifact/cleanup reports for independent review.

The checker establishes visible interruption/recovery, not graph execution at the
disconnect instant, cancellation acknowledgement, complete-state rollback or
performance. It does not exercise the JSONL worker. Model/state, lifecycle,
long-context, quality and device qualification remain separate gates.

Request failures retain one value-free `private-inference-request-failed error=Deadline`
line (or another fixed error category) in private stderr; no prompt, generated text
or credential is included. The uncapped HTTP change needs a published source-bound
bundle, Runner repin and actual-model owner acceptance before installed use changes.

Native generation also retains one fixed metadata line on completion or returned
failure: prompt/reuse/committed-output counts, total and active-phase elapsed time,
phase call/token attempts and completed lower bounds, and unknown partial progress.
Missing counts remain `unknown`. `firstTextMicros` observes the first nonempty text
delta before its HTTP write; it is not first sampled-token or delivered-text latency.
The line excludes prompt text, generated text, token IDs, paths and credentials.
Runtime timing includes instrumentation and rollback, excludes loading/rendering,
and does not qualify performance. Session elapsed time includes profile setup and
callback acknowledgements; phase sums omit other runtime overhead. CPU speculation
uses its unchanged path and reports attribution unavailable. Abrupt process exits
cannot publish a completed summary. Matched actual-model measurements and owning
hosted checks remain required before selecting an optimization.