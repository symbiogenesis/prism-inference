# Local delivery workflow

Never push anything, including `main`, branches or tags, directly or through
scripts. The owner handles every push from the
`C:\Users\symbi\source\repos\prism-inference` clone. Keep the primary checkout on
local `main`, review and integrate completed changes there, and leave them local.
Do not create Prism pull requests as the routine delivery workflow. Parallel
workers use individual Git worktrees and commit each logical change separately.

# CI ownership and completion

The owner handles CI, including Host CI and Guest CI dispatch, monitoring and
verdicts, except for the Foundation release-bundle operation below.
Other CI needs the owner's explicit direction. Fan-out scripts complete locally
without pushing, dispatching, querying, polling or waiting for CI. Record local
checks and pending CI acceptance accurately; pending CI does not block local
completion.

Agents may dispatch, monitor, download and verify artifacts from, and cancel a
Foundation run needed to produce a release bundle. Hosted release builds use
GitHub's own execution cap, without a repository-imposed build timeout or local
aggregate wall-clock cancellation deadline. Prefer a dedicated bundle-only lane;
its result does not qualify the complete Foundation suite. Cancel invalid-source,
failed or otherwise infeasible work when identified; elapsed local wall time alone
does not require cancellation. Retain unsuccessful attempts and their actual
outcomes without treating them as successful builds or extending closed diagnostic
trials. Existing owner-device, model and diagnostic execution budgets and power
observations remain unchanged. This exception does not authorize Git pushes or
other CI workflows.

# Owner-device execution budgets

Owner-device diagnostics, model qualification and benchmarks default to five
minutes total, preferably less. A dedicated, predeclared workload may use the
smallest justified longer allowance, up to fifteen minutes total. Record its
fixed work, conservative time estimate and reason before starting; fifteen
minutes is a ceiling, not a target. Charge setup, intake, loading, every comparison
arm, power observations, cleanup and evidence retention to one original clock.
Existing narrower capture, helper and workflow limits remain effective until
explicitly extended. Stop early on completion or a known no-go. Do not restart
clocks, retry a failed trial or divide a long matrix into repeated owner runs.
Historical trial deadlines stay closed. Full suites and benchmarks run only in
CI; hosted cloud CI retains its own workflow limits.

Runner's explicit 5/10/15-minute native-runtime acceptance option does not extend
Prism's numerical/accelerator captures or existing external owner helpers.

# Cross-target lint checks

For cfg-sensitive Rust changes or compiler/lint configuration changes, identify
the affected native and fallback branches before local delivery. Select each
changed helper package explicitly with pinned Release Clippy, `--all-targets`
and `-D warnings`; a dependency build does not lint the helper's test targets.
Include an applicable non-ARM target when changing NEON dispatch or stubs.
Use the [focused check recipes](docs/build-and-validation.md#focused-checks-before-local-delivery)
within the existing aggregate owner-device budget. Record unavailable targets
and pending hosted coverage accurately; never widen local work to a full suite
or restart a closed validation allowance.
