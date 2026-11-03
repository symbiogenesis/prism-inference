# Exact latest-prefix private-service integration

The existing `--prompt-reuse latest` native child option selects exact current
committed state at bit-identical adapter scale; omission retains per-request
reset. The policy appears in authenticated `/v1/models` and `/props` readiness
only when enabled. It adds no saved-prefix cache or per-request cache override.

Graph controls `profile_reuse_*` compare fresh recomputation with reused
complete attention, recurrent and convolution state, sampled logits and RNG
progression. They cover edited/switched history, scale-bit changes and change-back,
provisional failures, cancellation, explicit reset and attention-page boundaries.

The focused private-service control
`private_latest_prefix_policy_preserves_counting_and_transactional_reset_boundaries`
extends the transport controls to the selected latest policy: readiness,
counting without state mutation, successive acknowledged commits without
implicit reset, failed output rollback, closed admission, explicit reset,
count-seal invalidation and fresh success. Its synthetic engine supplies
transport/lifecycle evidence; graph fixtures supply the independent numerical
controls. Neither substitutes for actual-model service qualification.

Runner's source integration accepts this policy only explicitly and binds it to
the exact reviewed 0.1.14 release source and locked files. Default selection,
artifacts, admission, numerical contracts, complete-state rollback and native
containment remain unchanged. Hosted checks and actual-model latency/resource
qualification remain owner-operated and pending.
