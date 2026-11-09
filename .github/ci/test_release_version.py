"""Model-free bundle-version and bounded-workflow contract controls."""

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

    def test_workflow_is_a_separate_manual_hosted_four_minute_bundle(self):
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
        self.assertRegex(workflow, r"(?m)^    timeout-minutes: 4$")
        self.assertEqual(re.findall(r"(?m)^  ([a-z][a-z0-9_-]*):$", workflow.split("jobs:\n", 1)[1]),
                         ["bundle"])
        self.assertNotIn("matrix:", workflow)
        self.assertIn(".github/ci/release_version.py", workflow)
        self.assertIn("scripts/New-ReleaseBundle.ps1", workflow)
        self.assertIn(".github/ci/test_release_version.py", workflow)
        self.assertIn(".github/ci/test_inference_efficiency.py", workflow)
        self.assertIn("^Ran 3 tests in ", workflow)
        self.assertIn("^Ran 6 tests in ", workflow)
        self.assertIn("fmt --manifest-path $manifest --all -- --check", workflow)
        self.assertIn("--all-targets --features prism-inference/parallel-rows,prism-inference-server/parallel-rows -- -D warnings", workflow)
        self.assertIn("--release -p prism-inference --lib --no-default-features", workflow)
        self.assertIn("FinalSourceChecksPassed = $true", workflow)
        self.assertIn("FullFoundationQualified = $false", workflow)
        self.assertNotIn("cargo +nightly-2026-11-09 test", workflow)


if __name__ == "__main__":
    unittest.main()
