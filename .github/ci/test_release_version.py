"""Model-free bundle-version and hosted-release workflow contract controls."""

from pathlib import Path
import re
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import release_version as bundle  # noqa: E402


class ReleaseVersionTests(unittest.TestCase):
    def fixture(self, root, runtime="0.1.23", server="0.1.23", dependency="=0.1.23"):
        (root / "server").mkdir(exist_ok=True)
        (root / "Cargo.toml").write_text(
            f'[package]\nname = "prism-inference"\nversion = "{runtime}"\n', encoding="utf-8")
        (root / "server/Cargo.toml").write_text(
            f'[package]\nname = "prism-inference-server"\nversion = "{server}"\n'
            '[dependencies]\nbonsai-prism-runtime = { package = "prism-inference", '
            f'path = "..", version = "{dependency}" }}\n', encoding="utf-8")

    def test_matching_version_tracks_manifests_instead_of_workflow_literal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for version in ("0.1.23", "0.1.24"):
                self.fixture(root, version, version, "=" + version)
                self.assertEqual(bundle.release_version(root), version)

    def test_server_drift_nonexact_pin_and_invalid_versions_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for runtime, server, dependency in (
                    ("0.1.23", "0.1.22", "=0.1.23"),
                    ("0.1.23", "0.1.23", "^0.1.23"),
                    ("0.1.23", "0.1.23", "=0.1.22"),
                    ("0.1.23-rc.1", "0.1.23-rc.1", "=0.1.23-rc.1")):
                self.fixture(root, runtime, server, dependency)
                with self.subTest(runtime=runtime, server=server, dependency=dependency):
                    with self.assertRaises(ValueError):
                        bundle.release_version(root)

    def test_workflow_is_a_separate_manual_hosted_bundle_using_the_github_job_cap(self):
        root = Path(__file__).resolve().parents[2]
        workflow = (root / ".github/workflows/foundation-bundle.yml").read_text(encoding="utf-8")
        self.assertTrue(workflow.startswith("name: Foundation bundle\n"))
        self.assertIn("on:\n  workflow_dispatch:", workflow)
        self.assertIn("run-name: Foundation bundle / ${{ inputs.operation_id }}", workflow)
        self.assertIn("expected_revision:\n", workflow)
        self.assertIn("operation_id:\n", workflow)
        self.assertIn("$env:GITHUB_SHA -cne $env:EXPECTED_REVISION", workflow)
        self.assertNotIn("  push:", workflow)
        self.assertRegex(workflow, r"(?m)^    runs-on: windows-11-arm$")
        self.assertNotIn("timeout-minutes:", workflow)
        self.assertNotIn("timeout_minutes:", workflow)
        self.assertNotIn("BUNDLE_TIMEOUT", workflow)
        self.assertIn("CARGO_BUILD_TARGET: aarch64-pc-windows-msvc", workflow)
        self.assertIn("CARGO_TARGET_AARCH64_PC_WINDOWS_MSVC_RUSTFLAGS: '-C target-feature=+crt-static'", workflow)
        self.assertNotRegex(workflow, r"(?m)^\s+(?:RUSTFLAGS|CARGO_ENCODED_RUSTFLAGS):")
        self.assertEqual(re.findall(r"(?m)^  ([a-z][a-z0-9_-]*):$", workflow.split("jobs:\n", 1)[1]),
                         ["bundle"])
        self.assertNotIn("matrix:", workflow)
        self.assertIn(".github/ci/release_version.py", workflow)
        self.assertIn("scripts/New-ReleaseBundle.ps1", workflow)
        self.assertIn("-RequireStaticRuntime | ConvertTo-Json", workflow)
        self.assertIn(".github/ci/test_release_version.py", workflow)
        self.assertIn(".github/ci/test_inference_efficiency.py", workflow)
        self.assertIn(".github/ci/test_release_imports.py", workflow)
        self.assertIn("--expect-unittest-count 3 --", workflow)
        self.assertIn("--expect-unittest-count 6 --", workflow)
        self.assertIn("--expect-unittest-count 5 --", workflow)
        self.assertIn("fmt --manifest-path $manifest --all -- --check", workflow)
        self.assertIn("--all-targets --features prism-inference/parallel-rows,prism-inference-server/parallel-rows -- -D warnings", workflow)
        self.assertIn("--release -p prism-inference --lib --no-default-features", workflow)
        self.assertIn("FinalSourceChecksPassed = $true", workflow)
        self.assertIn("ExternalVcRuntimeImportsAbsent = $true", workflow)
        self.assertIn("FullFoundationQualified = $false", workflow)
        test_commands = re.findall(r"cargo \+nightly-2026-11-09 test[^\n]+", workflow)
        prefix = ("cargo +nightly-2026-11-09 test --manifest-path $manifest --locked --offline "
                  "--release -p prism-inference --features parallel-rows ")
        self.assertEqual(test_commands, [
            prefix + "--lib kernels::parallel_rows::tile::tests::tiled_parallel_rows_worker_panic_cannot_strand_caller -- --exact",
            prefix + "--test backend_threadpool failed_worker_group_drains_before_the_plan_is_reused -- --exact",
            *[prefix + "--lib " + name + " -- --exact" for name in (
                "graph::swift_state::tests::full_context_state_scope_is_geometry_derived_and_preserves_generic_cap",
                "graph::swift_state::tests::state_scope_rejects_changed_counts_recurrence_and_mtp",
                "graph::swift_state::tests::generic_constructors_and_excess_selected_budgets_reject_before_tensor_loading",
                "swift_session::tests::selected_cache_capacity_covers_packed_geometry_and_staged_windows",
                "swift_session::tests::selected_cache_speculation_reserves_full_scaled_checkpoint",
            )],
            prefix.replace("-p prism-inference ", "-p prism-inference-server ")
            + "--lib native_child::tests::native_child_capacity_uses_final_cache_selection_in_either_argument_order -- --exact",
        ])
        self.assertIn("StaticRuntimeUnwindControlsPassed = 2", workflow)
        self.assertIn("ScopedContextControlsPassed = 6", workflow)
        self.assertEqual(workflow.count("--expect-rust-test "), 8)
        self.assertNotIn("2>&1 | Out-String", workflow)
        self.assertIn("if: failure() && env.PRISM_CHECK_EVIDENCE != ''", workflow)
        self.assertIn("/*.failure.log", workflow)
        self.assertNotRegex((root / "Cargo.toml").read_text(encoding="utf-8"),
                            r'(?m)^panic\s*=\s*"abort"\s*$')


if __name__ == "__main__":
    unittest.main()
