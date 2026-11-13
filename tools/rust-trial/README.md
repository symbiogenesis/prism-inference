# Rust compiler capability checks

This dependency-free crate compares stable 1.99.1 with the nightly pinned in
[configuration](../../config/rust-nightly-trial.json). The workspace nightly enables
Polonius Alpha and the next trait solver; stable is the probe control.

## Borrowing fixtures

`LogitsSource` ties its associated output to an exclusive source borrow.
`LogitsCache` reuses constructor-owned storage, returning cached slices or refilling
on changed tokens/invalidation. Its token-only key is a fixture, not a production
decoder-state key. `maximum_logit` releases the view before reevaluation;
`with_logits` permits borrowed inputs and consumers returning a valid view.
`SupportsLogitsView` exercises higher-ranked associated-view normalization.

The regressions compare independent numerical output, cache hits, invalidation,
empty vocabulary and stable storage address/capacity. They establish ownership and
reuse, without model-parity or speed claims.

`fixtures/` compile directly with `rustc --edition=2024 --crate-type=lib
--emit=metadata`, outside Cargo discovery. They forbid unsafe code and deny warnings
and missing documentation.

| Fixture | Assertion | Stable | Nightly component |
| --- | --- | --- | --- |
| `stable_control.rs` | Refill before lending; generic consumer | Accept | Accept |
| `invalid_alias.rs` | Overlapping exclusive references | Reject `E0499` | Reject `E0499` |
| `polonius.rs` | Cached borrow or alternate-branch refill | Reject `E0499` | Accept with Polonius Alpha |
| `next_solver.rs` | Lifetime-generic view in higher-ranked callback | Reject `E0277` | Accept with next solver |

Controls distinguish capability from broken invocation or invalid aliasing. The
hosted matrix isolates all four nightly component combinations and stable. See
Rust's [Polonius Alpha](https://blog.rust-lang.org/2026/08/04/enabling-polonius-alpha-on-nightly/)
and [next solver](https://blog.rust-lang.org/2026/08/21/enabling-next-solver-on-nightly/)
descriptions for the language features.

## Run a focused check

Supply a fresh absolute evidence directory outside Git:

```powershell
./tests/Test-RustNightly.ps1 -Mode Probe -ProbeCase Polonius -EvidenceDirectory C:/absolute/temp/rust-trial-evidence
```

`-Mode Compatibility -PackageGroup Runtime` or `Broker` selects formatting and
Release Clippy/documentation; the default selects all packages. Full correctness
and the complete compiler matrix require hosted `-FullTests`.
[Source checks](../../.github/workflows/source-checks.yml) requires successful
selected probes and retained compiler identity, arguments, source hashes and
verdicts; failed, cancelled or skipped probes reject. Foundation owns workspace
correctness and native Prism linking. Follow the
[validation policy](../../docs/build-and-validation.md) for local budgets.
