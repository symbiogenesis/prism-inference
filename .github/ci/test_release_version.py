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
        self.assertIn("^Ran 3 tests in ", workflow)
        self.assertIn("^Ran 6 tests in ", workflow)
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
            prefix + "--lib kernels::parallel_rows::tile::tests::tiled_parallel_rows_worker_panic_cannot_strand_caller -- --exact 2>&1 | Out-String)",
            prefix + "--test backend_threadpool failed_worker_group_drains_before_the_plan_is_reused -- --exact 2>&1 | Out-String)",
        ])
        self.assertIn("StaticRuntimeUnwindControlsPassed = 2", workflow)
        self.assertIn("^running 1 test", workflow)
        self.assertIn("1 passed; 0 failed; 0 ignored", workflow)
        self.assertNotRegex((root / "Cargo.toml").read_text(encoding="utf-8"),
                            r'(?m)^panic\s*=\s*"abort"\s*$')


if __name__ == "__main__":
    unittest.main()
