# Prism Inference 0.1.11 candidate

This GitHub binary candidate adds the effective row-worker budget to private
native HTTP readiness. The release review compares the complete source delta
against `749a5f5b43bf77764a29682557ec91a8de8dcdef`, the pinned `0.1.10` source.
It preserves explicit original-PQ2/current-weight-PTQ1 selection, approved-adapter
identity and scale bounds, fresh state for every request, the one-worker default
and existing process ownership and admission requirements.

## Production execution and protocol changes

The `row_workers` readiness field comes from the instantiated session rather
than echoing command-line arguments. Owners must compare it with their selected
worker policy. The private worker rejects zero or over-limit profile identities
before spawning its model-owner thread. The field reports the effective budget;
it does not count completed parallel operations, select an ISA or establish a
performance gain. Foundation's feature-variant controls cover matching serial
and four-worker budgets plus rejection of invalid identities.

The unreleased fused recurrent copy/decay change from `244df2e` was discovered in
the full release delta and removed from production before this release candidate.
The production recurrent path again uses the exact `749a5f5` copy-then-scale
sequence. The independent native GDN fixture keeps its raw captured state inputs;
the fusion candidate's model-free bitwise/scratch controls remain test-only.
Git history and the runtime contract retain the candidate, pending actual-model
logits/live-state, throughput and resource qualification before release/repinning.

The runtime/server package versions, exact server runtime dependency, two lockfile
package versions and Foundation archive version become `0.1.11`. No third-party
dependency version or checksum changes. Kernels, sampler, tokenizer, adapter
arithmetic and authoring basis are unchanged from the selected `0.1.10` source.
This source review does not establish equality of newly compiled binary bytes or
replace exact artifact verification and target-device qualification.

Windows CNG GGUF verification and opt-in `PromptReuse::Latest` were already
present in `749a5f5`; they are retained capabilities, not new `0.1.11` changes.
Serving retains reset-every-request. Exact length/SHA-256 checks and read-only
file guards remain mandatory. No startup, throughput or resource improvement is
claimed by this candidate.

## Source tools outside the five-executable bundle

The source-only `swift_model_check rollback` mode adds bounded complete-state
cancellation/continuation captures. It retains exact tokens, logits, logical
KV/convolution/recurrent state and create-new evidence under its 2 GiB payload
cap. Its capture-stop token and cooperative cancellation do not qualify production
sampling, native HTTP cancellation or a full Swift workload.

The new private-service checker, source-bound owner workflow and intake/archive
helpers prepare one externally supervised released-service capture. Setup,
loading, work, comparison, retention and cleanup share the existing five-minute
owner budget; partial captures cannot continue through later owner jobs. Their
pinned Runner owner and its selected release lock are independent identities.
Preparing these tools does not automatically retarget an existing pinned owner
to `0.1.11` or qualify the new binary release.

The optional unpublished Hexagon source crate adds fixed signed X2 F32/guarded-Q2
queue controls, explicit nonblocking submission, mapping-before-module teardown
and conservative retention after uncertain native results. It remains a separate
terminal diagnostic surface requiring reviewed package/driver/ABI inputs and an
external process owner. None of the five bundled CPU executables depends on this
crate; these additions do not enable model offload or a Hexagon runtime profile.
Its owning source/ABI/device gates remain separate from CPU release qualification.

## Delivery and remaining qualification

Release only the exact Foundation-produced ARM64 and x64 archives and receipts
after the owning main Foundation and runtime conformance gates pass. Preserve
all older release bytes. Runner integration needs a reviewed exact source/archive
lock, every bundled file digest and explicit version admission for four workers.
The changed readiness must pass actual released-service lifecycle acceptance;
previous worker-policy device results do not transfer to this candidate.

New-source full-context logits/live-state, loaded-zero/nonzero application,
adapter authoring provenance, generated-history lifecycle, paired benefit,
independent human quality/comfort, resource, isolation, installation and rollback
gates remain required for default promotion. A source-equivalent CPU path and
hosted controls do not close these device/application gates.

Registry publication is separate: the new native hash crate, a new GGUF registry
version and an updated exact runtime dependency remain prerequisites. The server
still has an exact Git credential-helper dependency and is not registry-published.
This binary release changes no PTQ1 default, adapter qualification, model quality
or device-resource claim.
