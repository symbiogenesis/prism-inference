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
