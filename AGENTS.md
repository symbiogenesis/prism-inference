# Local delivery workflow

On this computer, the owner authorizes direct pushes to `main` from the
`C:\Users\symbi\source\repos\prism-inference` clone. Do not create Prism pull
requests as the routine delivery workflow. Review changes before integrating
and pushing. Parallel workers use individual Git worktrees and commit each
logical change separately.

Owner-device diagnostics, model qualification and benchmarks share a five-minute
total ceiling, including setup, loading, comparison and cleanup. Full suites
and benchmarks run only in CI; do not split a long matrix into repeated owner runs.
