"""Protect native cfg lint coverage when compiler and check inputs change."""

from pathlib import Path
import re
import shlex
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import impact  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]


class HexagonWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.workflow = (ROOT / ".github/workflows/hexagon-native.yml").read_text(encoding="utf-8")

    def job(self, name):
        jobs = self.workflow.split("\njobs:\n", 1)[1]
        match = re.search(rf"(?ms)^  {re.escape(name)}:\n(.*?)(?=^  [\w-]+:\n|\Z)", jobs)
        self.assertIsNotNone(match, f"Missing independent {name} job")
        return match[1]

    def cargo_commands(self, job):
        return [shlex.split(line.strip()) for line in job.splitlines()
                if line.strip().startswith("cargo ")]

    def assert_native_lint(self, job):
        commands = self.cargo_commands(job)
        lint = [command for command in commands if "clippy" in command]
        self.assertTrue(lint, "Native test targets must be linted")
        self.assertTrue(any(
            {"--locked", "--release", "--all-targets", "bonsai-hexagon-native"}.issubset(command)
            and command[-3:] == ["--", "-D", "warnings"] for command in lint
        ), "Keep strict release Clippy over native library and test targets")

    def test_compiler_configuration_and_action_changes_select_native_checks(self):
        push = re.search(r"(?ms)^  push:\n(.*?)(?=^  [\w-]+:|\Z)", self.workflow)
        self.assertIsNotNone(push)
        paths = re.search(r"(?m)^    paths:\n((?:      - .*\n)+)", push[1])
        self.assertIsNotNone(paths)
        patterns = [line.removeprefix("      - ").strip("'\"")
                    for line in paths[1].splitlines()]
        for path in (
            "rust-toolchain.toml", "clippy.toml", "rustfmt.toml",
            ".cargo/config.toml", ".cargo/nested/check.toml",
            ".github/actions/rust-toolchain/action.yml",
            ".github/actions/rust-toolchain/scripts/install.ps1",
            "tools/hexagon-native/src/native_q2_forwarding_tests.rs",
            "tools/hexagon-runtime/src/lib.rs", "Cargo.toml", "Cargo.lock",
            ".github/workflows/hexagon-native.yml",
        ):
            with self.subTest(path=path):
                self.assertTrue(impact.any_match(path, patterns),
                                f"{path} must select independent native cfg checks")

    def test_linux_cross_check_reaches_windows_arm64_tests_with_strict_clippy(self):
        cross = self.job("cross-native-arm64")
        self.assertRegex(cross, r"(?m)^    runs-on: ubuntu-24\.04$")
        self.assertRegex(cross, r"(?m)^      CARGO_BUILD_TARGET: aarch64-pc-windows-msvc$")
        self.assertRegex(cross, r"rustup target add .* aarch64-pc-windows-msvc")
        checks = [command for command in self.cargo_commands(cross) if "check" in command]
        self.assertTrue(any({"--all-targets", "bonsai-hexagon-native"}.issubset(command)
                            for command in checks))
        self.assert_native_lint(cross)

    def test_cross_check_keeps_independent_native_arm64_and_x64_contract_jobs(self):
        native = self.job("native-contracts")
        self.assertRegex(native, r"(?m)^    runs-on: \$\{\{ matrix\.runner \}\}$")
        self.assertRegex(native, r"(?m)^      CARGO_BUILD_TARGET: \$\{\{ matrix\.target \}\}$")
        for runner, target in (("windows-11-arm", "aarch64-pc-windows-msvc"),
                               ("windows-2025", "x86_64-pc-windows-msvc")):
            with self.subTest(target=target):
                self.assertRegex(native, rf"- runner: {runner}\n\s+target: {target}\n")
        commands = self.cargo_commands(native)
        self.assertTrue(any({"test", "--all-targets", "bonsai-hexagon-native"}.issubset(command)
                            for command in commands))
        self.assertTrue(any({"test", "--doc", "bonsai-hexagon-native"}.issubset(command)
                            for command in commands))
        self.assertTrue(any({"doc", "--no-deps", "bonsai-hexagon-native"}.issubset(command)
                            for command in commands))
        self.assert_native_lint(native)


if __name__ == "__main__":
    unittest.main()
