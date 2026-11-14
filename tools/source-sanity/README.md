# Native source sanity

`prism-source-sanity` screens the exact Git objects an owner is about to push.
It is adapted from Runner's native push guard at
[`20c4064`](https://github.com/symbiogenesis/SwiftBonsai2Runner/tree/20c4064ddf840c79d4bcc06fa46a2ba24c1414b8/tools/ci/src),
under the same Apache-2.0 license. Prism uses its own workspace, compiler pin and
workflow policy; Runner's application and owned-language rules do not apply.

The tracked `.githooks/pre-push` invokes
[Check-PushSanity.ps1](../../.github/ci/Check-PushSanity.ps1). The owner handles
pushes and CI. Installing the hook or preparing the helper authorizes neither.

## Prepare and use

Prerequisites are native Windows ARM64, stable PowerShell 7.6 or later, Git,
the repository's pinned Rust toolchain and locked dependencies already cached.
The native process boundary is the existing published
`bonsai-windows-process` dependency. Initial preparation compiles the helper
from committed `HEAD`; declare its source-build allowance and supervise that
build externally before invoking:

```powershell
pwsh -NoLogo -NoProfile -File .github/ci/Check-PushSanity.ps1 -Prepare
git config core.hooksPath .githooks
```

Preparation retains a receipt binding the ordered committed build inputs and
the executable's SHA-256. Missing, stale or modified receipts or helper bytes
block checking until explicit preparation succeeds. No automatic compilation
runs during a push. A preparation receipt is not a source-check verdict.

For a read-only full-tree check of a specific commit, use its full object ID:

```powershell
pwsh -NoLogo -NoProfile -File .github/ci/Check-PushSanity.ps1 -Commit <full-object-id>
```

The ordinary hook consumes Git's pre-push input and checks each distinct
outgoing commit once. It peels annotated tags, ignores deletions and unions the
changes selected by every remote base for that commit. A new ref or unavailable
remote base selects the complete tree. It does not contact the remote or query CI.

## Scope and limits

Each commit is materialized through a private Git index in invocation-private
temporary storage. The owner's index and working files do not supply source and
cannot hide a broken historical or non-HEAD tip. An empty `.git` directory lets
actionlint inspect local composite-action inputs without copying live metadata.
Git repository-selection environment variables are removed from child commands,
and local replacement refs cannot substitute other objects for outgoing IDs.

Rust/build/formatter changes select `cargo fmt --all -- --check` with the
outgoing snapshot's dated compiler pin and its resolved Rustfmt executable.
Toolchain auto-installation and Cargo network access are disabled. Formatting
precedes the trusted PowerShell policy, which parses selected PowerShell files
without executing candidate scripts and routes workflow changes through Prism's
existing workflow validator. Workflow checks require its pinned actionlint;
Python and model work are never invoked by this guard.

Native child processes and descendants enter an owned Windows Job before
execution. The original bootstrap elapsed time shares a 270-second admission
and work deadline, leaving reserve inside the five-minute owner allowance for
termination, cleanup and observations. Each command has bounded output and
fails closed on missing tools, invalid input, nonzero exit or exhausted time.
The adapter records power before bootstrap and after cleanup; the native helper
also records its own endpoint observations. Unknown observations stay unknown.
Endpoint snapshots do not establish uninterrupted power stability or qualify a
performance comparison.

This screen does not compile candidate application code, run Clippy, run complete
tests, audit dependencies, benchmark inference or qualify models. Foundation and
the other owning CI lanes retain those verdicts. Local hooks are bypassable by
an owner and do not enforce remote repository policy.
