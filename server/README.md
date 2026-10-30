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
