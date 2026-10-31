# Prism inference private server

The independently built prism-inference-server executable supplies the same strict private HTTP contract used by SwiftBonsai's guarded development launcher. The owner verifies executable and model bytes before launch and retains process/Job ownership, the fixed memory limit, authenticated readiness and listener-PID checks.

The server loads the selected exact Swift artifact before binding numeric IPv4 loopback. PQ2 without an adapter remains the default. Explicit `--use-repacked-ptq1` selects only the verified current-weight export; `--abliterate-lora <absolute-path>` loads only the approved rank-one adapter. These options preserve the existing process admission and do not qualify full-model parity, creative quality, device performance or lower memory limits.

Read-only model, adapter and key validation uses the credential helper pinned to
Runner revision `abe04b4a8eff18388af96f0e75c8a8d180b81989`. Inside an actual Windows
AppContainer, that helper preserves local-drive, non-root ancestry, reparse and
regular-file checks while avoiding denied volume-root metadata queries. The
trusted owner still validates and retains complete ancestry before granting only
the selected files. The key remains held by a write/delete-denying reader. No
filesystem grant is expanded.

The server temporarily has `publish = false`: its reviewed Git-only helper
dependency must remain exact. Delivery uses checked GitHub binary prereleases or
the exact source checkout, not crates.io server installation. Publishing a shared
helper release and qualifying a registry-backed server remain pending; the root
runtime library's separate publication contract is unchanged.

Failed startup emits one value-free stderr line, for example
`private-inference-startup-failed stage=arguments errorKind=PermissionDenied osCode=5`.
The fixed stages are `arguments`, `environment`, `memory`, `key`, `model`,
`runtime`, `listener` and `backend`; an unclassified error uses `unknown`.
`osCode=none` means no numeric OS code was available. No original error message,
source chain, path, argument, credential or model value enters this diagnostic.
The owning launcher must retain it before deleting temporary CI storage.

Extended `/v1/models` and `/props` readiness names the exact model ID/hash and
`weight_representation`, the approved `adapter_sha256` or null, and
`supports_abliteration_scale`. Adapter profiles also report
`abliteration_scale_range: [0, 2]`. Legacy PQ2/no-adapter readiness is unchanged.
Adapter-loaded private requests require exactly `lora: [{"id": 0, "scale": 1.3}]`;
the scale must be a finite JSON number in 0..2 before conversion to binary32.
The outer broker owns the public `abliteration_scale` setting; that field is
rejected directly on this private endpoint. Missing/null/extra adapter slots,
other IDs and overrides reject; no-adapter requests reject `lora` entirely.

Exact count/completion canonical bytes bind the selected scale, including signed
zero. Every generation starts fresh model state at that scale; prior prepared
input cannot be reused after reset or a different count. One acknowledged output
event bounds provisional streaming, and cancellation/deadline or delivery failure
closes admission until explicit reset. Reset preserves loaded artifact identities
and restores the seeded sampler. Successful requests keep the existing sampler
draw sequence, including across scale changes.

The private canonical request accepts either boolean value for
`chat_template_kwargs.enable_thinking`. Both exact counting and generation use
that selected template mode. With thinking disabled, the template closes the
reasoning block before generation and all generated text is visible answer
content. Supplied assistant reasoning history remains preserved. Effort stays
`xhigh` when thinking is enabled, and other template and sampler overrides remain
rejected. Existing public Rust `prepare` methods and the JSONL worker retain their
thinking-enabled behavior; `prepare_with_thinking` selects the mode explicitly.

## Released-service correctness capture

The main-only [acceptance workflow](../.github/workflows/private-service-acceptance.yml)
builds `private-service-check.exe` and runs its model-free framing and cancellation
observation controls on hosted Windows ARM64. Its artifact is named
`private-service-check-<main-revision>-<run-id>-<attempt>`. The packaged receipt binds
the checker bytes to its exact owning main source, workflow and run. Its version-2
package also contains `private-service-owner.exe`, built from Prism's Rust owner
example against exact Runner revision `1914330f8a31769bfe8368ca9728fe7a418ed84e`,
and the owner source digest. Hosted preparation checks the owner, package intake
and mandatory deadline controls without models. Preparation remains the default;
it does not qualify a released service.

Explicit `run_device: true` adds one Prism-owned `acceptance` job. An external
trusted owner must register an ephemeral runner with only the run-specific
`private-service-acceptance-<run-id>` label and supply the complete
`BONSAI_NATIVE_OWNER_*` run/revision/start-tick/start-UTC/fixed-210-second deadline
binding. Missing, partial, stale or exhausted bindings fail closed. The external
owner has a 240-second process lifetime, including thirty seconds after the work
cutoff, and at most sixty further seconds for evidence intake: five minutes total.
No retry or matrix extends that allowance.

The job starts its own monotonic clock before checkout and same-run package
intake. It verifies the exact member inventory, source/run receipt, native ARM64
PE identities and held-file digests. The Rust owner additionally retains every
installed release member and the released archive, then uses the exact reviewed
Runner launcher for model/adapter verification, native sandbox, Job, memory,
authenticated readiness and exclusive listener monitoring. All setup reduces the
remaining model/checker allowance, with independent checker, cleanup and outer
process limits. Only `host.json`, `hosted-build.json`, `owner.json` and
`service.json` may be uploaded; protected child directories contain credentials.
Successful owner receipts require an empty owned Job and released listeners.

An existing trusted launcher must start the independently verified released
`prism-inference-server.exe` with the original PQ2 model, approved adapter, seed,
explicit row-worker count and omitted prompt reuse. It retains the executable,
model, adapter and private credential guards, AppContainer/Job, exclusive listener
PID, memory/watchdog policy and teardown. The checker starts no process and accepts
only a numeric loopback port. It uses the same protected private backend key as the
owner; do not pass key bytes on the command line or save them in evidence.

```powershell
# Run outside every checkout while the existing trusted owner retains the server.
# The new evidence file must be absolute and must not already exist.
& $checker $privatePort $privateKeyPath $newEvidencePath $remainingCheckSeconds
```

The owner supplies a positive allowance of at most 240 seconds, shortened to the
actual time remaining after setup, artifact verification and model loading, with
time reserved for owned cleanup. All owner-device diagnostics and captures share
one five-minute aggregate budget; this command does not grant another allowance.
Socket operations check one absolute checker deadline. No automatic retry or model
download occurs. Never use the checker against an independently running daily
service: it explicitly resets state and abandons a generated response.

The fixed workload checks exact PQ2/template/adapter readiness, canonical counts
and terminal generation usage at scales 1.3 and zero, invalid-scale rejection,
stale canonical generation after a replacement count and reset, then disconnects
after receiving a nonempty provisional streamed delta. A subsequent count must
be rejected until explicit reset; reset must reproduce the initial seeded short
response. Every completed observation is flushed to the held evidence file before
the next step. Natural completion, missing provisional output, a deadline or any
failed required observation leaves a failed receipt with the completed steps.
Responses and credentials are never written to that receipt.

Receiving provisional output and observing closed admission establishes externally
visible interruption and recovery. It does not establish graph execution at the
instant of disconnect, a cancellation acknowledgement, logits or complete-state
rollback. Those fields remain explicitly unqualified in the receipt. The checker
also establishes no sandbox/process ownership, performance, long-context/output,
creative quality or default-promotion evidence. Runner lifecycle receipts remain
separate from this Prism-owned capture.

Archive the checker package and receipt alongside the actual-model result and
the trusted owner's process/cleanup and artifact-verification reports. Bind the
tested released source, version, archive/server/worker SHA-256 identities separately
from the later owning main checker source/workflow/run and retained package artifact.
An internal model thread is exercised through the released HTTP service; the
separate JSONL `bonsai-swift-worker` binary is not executed by this checker. Exact
source Foundation/conformance and independent review remain required, and a passed
checker receipt alone cannot close the released-service promotion gate.
