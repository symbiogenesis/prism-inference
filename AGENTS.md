# Local delivery workflow

Never push anything, including `main`, branches or tags, directly or through
scripts. The owner handles every push from the
`C:\Users\symbi\source\repos\prism-inference` clone. Keep the primary checkout on
local `main`, review and integrate completed changes there, and leave them local.
Do not create Prism pull requests as the routine delivery workflow. Parallel
workers use individual Git worktrees and commit each logical change separately.

# CI ownership and completion

The owner handles all CI, including Host CI and Guest CI, dispatch, monitoring
and verdicts. Do not dispatch or monitor CI unless the owner explicitly directs
that action. Never wait for CI as part of task or fan-out completion. Fan-out
scripts must complete locally without pushing, dispatching, querying, polling or
waiting for CI. Record local checks and pending CI acceptance without claiming
pending checks have passed; pending CI does not block local completion.

# Owner-device execution budgets

Owner-device diagnostics, model qualification and benchmarks share a five-minute
total ceiling, including setup, loading, comparison and cleanup. Full suites
and benchmarks run only in CI; do not split a long matrix into repeated owner runs.
