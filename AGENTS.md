# Local delivery workflow

Never push anything, including `main`, branches or tags, directly or through
scripts. The owner handles every push from the
`C:\Users\symbi\source\repos\prism-inference` clone. Keep the primary checkout on
local `main`, review and integrate completed changes there, and leave them local.
Do not create Prism pull requests as the routine delivery workflow. Parallel
workers use individual Git worktrees and commit each logical change separately.

# CI ownership and completion

The owner handles CI, including Host CI and Guest CI dispatch, monitoring and
verdicts, except for the bounded Foundation release-bundle operation below.
Other CI needs the owner's explicit direction. Fan-out scripts complete locally
without pushing, dispatching, querying, polling or waiting for CI. Record local
checks and pending CI acceptance accurately; pending CI does not block local
completion.

Agents may dispatch, monitor, download and verify artifacts from, and cancel a
Foundation run needed to produce a release bundle. Predeclare one fixed operation
with a maximum of five minutes aggregate wall time, including setup, queue time,
dispatch, monitoring, artifact verification, cancellation and evidence retention.
Prefer a dedicated bundle-only lane; its result does not qualify the complete
Foundation suite. Reserve time to cancel the run before the aggregate deadline if
it fails or cannot finish. Stop early when done or infeasible. Do not restart the
clock, retry a failed operation or split a longer run into repeated allowances.
Existing owner-device power and execution-budget rules still apply. This exception
does not authorize Git pushes or other CI workflows.

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
