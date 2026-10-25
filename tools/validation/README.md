# Script standards

`tests/Test-Standards.ps1` checks every repository PowerShell source with
PSScriptAnalyzer 1.25.0 and explicit parameter, output, parser and error-handling
contracts. The main-only `Script standards` workflow runs these checks in Prism;
it does not build Runner tools or run model/test/benchmark workloads.

Every literal `Add-Type -TypeDefinition` helper is extracted into external
temporary storage and compiled through `CSharp.Strict.csproj`. The project uses
the pinned .NET 10 SDK, nullable and overflow checks, warnings as errors, all
enabled .NET analyzers, documentation and build-time code-style diagnostics.
Scripts must also supply the matching strict `Add-Type` compiler options.

When the repository contains no embedded C# helper, validation reports that fact
and performs no C# build. Adding a helper automatically enables its extraction
and strict analyzer build. The standalone project rejects missing or empty
extracted-source input.

Formatting, analyzers and focused compiler checks may run locally. Complete
correctness suites and every benchmark remain restricted to hosted CI.
