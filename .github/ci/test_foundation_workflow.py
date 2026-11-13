"""Keep Foundation's parallel scheduling and fresh package evidence intact."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


class FoundationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.workflow = (ROOT / ".github/workflows/foundation.yml").read_text(encoding="utf-8")

    def job(self, name):
        match = re.search(rf"(?ms)^  {re.escape(name)}:\n(.*?)(?=^  [\w-]+:\n|\Z)",
                          self.workflow.split("\njobs:\n", 1)[1])
        self.assertIsNotNone(match, f"Missing {name} job")
        return match[1]

    def step(self, name):
        match = re.search(rf"(?ms)^      - name: {re.escape(name)}\n(.*?)(?=^      - |\Z)",
                          self.job("windows"))
        self.assertIsNotNone(match, f"Missing {name} step")
        return match[1]

    def test_all_correctness_shards_and_native_architectures_remain_required(self):
        windows = self.job("windows")
        lanes = re.search(r"(?m)^        lane: \[([^]]+)\]$", windows)[1].split(", ")
        self.assertEqual(lanes, ["correctness-0", "correctness-1", "correctness-2",
                                 "correctness-3", "checks", "variants"])
        for target in ("aarch64-pc-windows-msvc", "x86_64-pc-windows-msvc"):
            self.assertIn(f"target: {target}", windows)
        self.assertIn("fail-fast: false", windows)
        self.assertIn("--index $index --count 4", windows)
        self.assertIn("needs.changes.outputs.windows == 'true'", windows)
        self.assertIn("needs: [changes, windows, native-overlay, rust-regression, codec-native, source-package]",
                      self.job("foundation"))

    def test_release_version_build_package_and_upload_share_one_owner(self):
        owner = "matrix.lane == 'correctness-2'"
        for name in ("Read the exact bundled package version",
                     "Package the five checked Windows executables",
                     "Retain the checked release bundle and source receipt"):
            with self.subTest(step=name):
                self.assertIn(f"if: {owner} && needs.changes.outputs.release == 'true'", self.step(name))
        build = self.step("Build selected default-feature executables")
        self.assertIn(f"if: {owner}\n", build)
        self.assertIn("tests/Test-ReleaseBundle.ps1", build)
        self.assertIn("--bins --locked --offline --release", build)
        package = self.step("Package the five checked Windows executables")
        self.assertIn("scripts/New-ReleaseBundle.ps1", package)
        self.assertIn("-SourceRevision $env:GITHUB_SHA", package)
        self.assertIn("if-no-files-found: error", self.step("Retain the checked release bundle and source receipt"))

    def test_mutations_and_download_writer_keep_unique_owners(self):
        mutations = self.step("Verify immutable reference fixtures and kill the bounded mutation catalog")
        self.assertIn("if: matrix.lane == 'correctness-3' && matrix.target == 'aarch64-pc-windows-msvc' && needs.changes.outputs.mutations == 'true'", mutations)
        self.assertIn("provenance.fixtureSha256", mutations)
        self.assertIn("tests/Test-Mutations.ps1", mutations)
        writer = self.step("Save locked downloads once per architecture")
        self.assertIn("if: matrix.lane == 'correctness-0' && steps.downloads.outputs.cache-hit != 'true'", writer)

    def test_bundle_seed_matches_release_owner_without_narrowing_controls(self):
        bundle = (ROOT / ".github/workflows/foundation-bundle.yml").read_text(encoding="utf-8")
        self.assertIn("prism-runtime-v2-aarch64-pc-windows-msvc-correctness-2-", bundle)
        self.assertNotIn("prism-runtime-v2-aarch64-pc-windows-msvc-correctness-0-", bundle)
        self.assertIn("--expect-rust-test", bundle)
        self.assertNotIn("timeout-minutes:", bundle)

    def test_source_package_cache_cannot_restore_archives_or_receipts(self):
        package = self.job("source-package")
        self.assertIn("id: package-toolchain", package)
        cache = package.split("- name: Reuse only source-package compiler outputs\n", 1)[1].split("      - name:", 1)[0]
        paths = re.search(r"(?m)^          path: \|\n((?:            .*\n)+)", cache)[1]
        self.assertEqual([line.strip() for line in paths.splitlines()],
                         ["${{ env.CARGO_TARGET_DIR }}/debug", "${{ env.CARGO_TARGET_DIR }}/release"])
        for line in cache.splitlines():
            if "prism-source-package-v1-" in line:
                for identity in ("matrix.target", "steps.package-toolchain.outputs.image-identity",
                                 "steps.package-toolchain.outputs.compiler-identity"):
                    self.assertIn(identity, line)
        # A cache hit must never bypass normalized packaging or any consumer arm.
        validation = package.split("- name: Verify normalized packages and an external library consumer\n", 1)[1].split("      - name:", 1)[0]
        self.assertNotIn("if:", validation)
        self.assertIn(".github/ci/source_package.py", validation)
        script = (ROOT / ".github/ci/source_package.py").read_text(encoding="utf-8")
        self.assertIn("evidence.mkdir(parents=True, exist_ok=False)", script)
        self.assertIn('for features in ("scalar", "default", "parallel-rows"):', script)
        self.assertIn('"package", "--manifest-path"', script)
        self.assertIn('CARGO_UNSTABLE_CHECKSUM_FRESHNESS: \'true\'', self.workflow)


if __name__ == "__main__":
    unittest.main()
