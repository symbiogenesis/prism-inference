"""Protect distinct native cfg coverage after consolidation into Foundation."""

from pathlib import Path
import json
import re
import shlex
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import impact  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]


class HexagonWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.workflow = (ROOT / ".github/workflows/foundation.yml").read_text(encoding="utf-8")

    def job(self, name):
        jobs = self.workflow.split("\njobs:\n", 1)[1]
        match = re.search(rf"(?ms)^  {re.escape(name)}:\n(.*?)(?=^  [\w-]+:\n|\Z)", jobs)
        self.assertIsNotNone(match, f"Missing independent {name} job")
        return match[1]

    def cargo_commands(self, job):
        commands = []
        for line in job.splitlines():
            command = line.strip()
            if "run_ci_command.py" in command and " -- cargo " in command:
                command = "cargo " + command.split(" -- cargo ", 1)[1]
            if command.startswith("cargo "):
                commands.append(shlex.split(command))
        return commands

    def assert_native_lint(self, job):
        commands = self.cargo_commands(job)
        lint = [command for command in commands if "clippy" in command]
        self.assertTrue(lint, "Native test targets must be linted")
        self.assertTrue(any(
            {"--locked", "--release", "--all-targets", "bonsai-hexagon-native"}.issubset(command)
            and command[-3:] == ["--", "-D", "warnings"] for command in lint
        ), "Keep strict release Clippy over native library and test targets")

    def test_compiler_configuration_and_action_changes_select_native_checks(self):
        for path in (
            "rust-toolchain.toml", "clippy.toml", "rustfmt.toml",
            ".cargo/config.toml", ".cargo/nested/check.toml",
            ".github/actions/rust-toolchain/action.yml",
            ".github/actions/rust-toolchain/scripts/install.ps1",
            "tools/hexagon-native/src/native_q2_forwarding_tests.rs",
            "tools/hexagon-runtime/src/lib.rs", "Cargo.toml", "Cargo.lock",
            ".github/workflows/foundation.yml",
        ):
            with self.subTest(path=path):
                plan = impact.select(ROOT, [path])
                self.assertTrue(plan["groups"]["windows"])
                # Only the isolated native crate needs the Linux-hosted cross
                # lint. Its runtime consumer does not change native FFI cfgs.
                if path != "tools/hexagon-runtime/src/lib.rs":
                    self.assertTrue(plan["groups"]["hexagon"])

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
        native = self.job("windows")
        self.assertRegex(native, r"(?m)^    runs-on: \$\{\{ matrix\.runner \}\}$")
        self.assertRegex(native, r"(?m)^      CARGO_BUILD_TARGET: \$\{\{ matrix\.target \}\}$")
        for runner, target in (("windows-11-arm", "aarch64-pc-windows-msvc"),
                               ("windows-2025", "x86_64-pc-windows-msvc")):
            with self.subTest(target=target):
                self.assertRegex(native, rf"- runner: {runner}\n\s+target: {target}\n")
        self.assertIn(".github/ci/foundation_shards.py", native)
        commands = self.cargo_commands(native)
        for required in ({"clippy", "@selection", "--all-targets", "--release", "-D", "warnings"},
                         {"doc", "@selection", "--no-deps", "--release"}):
            self.assertTrue(any(required.issubset(command) for command in commands))
        for package in ("bonsai-hexagon-native", "bonsai-hexagon-runtime"):
            plan = impact.select(ROOT, ["tools/" + package.removeprefix("bonsai-") + "/src/lib.rs"])
            self.assertIn(package, plan["packages"])
            self.assertTrue(plan["groups"]["windows"])

    def test_cache_keys_isolate_compiler_image_target_and_lane(self):
        self.assertRegex(self.workflow, r"(?m)^  CARGO_UNSTABLE_CHECKSUM_FRESHNESS: 'true'$")
        self.assertRegex(self.workflow, r"(?m)^  CARGO_BUILD_FINGERPRINT: content$")
        for name, lane, target in (
            ("cross-native-arm64", "cross", "aarch64-pc-windows-msvc"),
        ):
            with self.subTest(job=name):
                job = self.job(name)
                self.assertIn("id: rust\n", job)
                prefix = (f"prism-hexagon-v1-{lane}-${{{{ runner.os }}}}-${{{{ runner.arch }}}}-"
                          f"{target}-${{{{ steps.rust.outputs.image-identity }}}}-"
                          "${{ steps.rust.outputs.compiler-identity }}-")
                self.assertIn("key: " + prefix + "${{ hashFiles(", job)
                self.assertIn("restore-keys: |\n            " + prefix + "\n", job)
                self.assertIn("key: prism-downloads-v1-${{ runner.os }}-${{ runner.arch }}-${{ hashFiles('Cargo.lock') }}", job)
                for source in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".cargo/**",
                               "src/**", "tools/**", "tests/**", "server/**"):
                    self.assertIn("'" + source + "'", job)

    def test_cached_builds_never_skip_commands_or_restore_results(self):
        for name, directory in (("cross-native-arm64", "Prism-Hexagon-Cross-Target"),):
            with self.subTest(job=name):
                job = self.job(name)
                # A cache hit reuses compiler inputs, never a prior check verdict.
                self.assertNotIn("cache-hit", job)
                self.assertEqual(len(re.findall(r"(?m)^\s+if:", job)), 1)
                self.assertIn("needs.changes.outputs.hexagon == 'true'", job)
                paths = re.findall(r"(?m)^          path: (.+)$", job)
                self.assertEqual(paths, ["|", "${{ runner.temp }}/" + directory])
                steps = re.split(r"(?m)^      - ", job)[1:]
                cache_steps = [step for step in steps if "uses: actions/cache" in step]
                self.assertEqual(len(cache_steps), 2)
                for step in steps:
                    if step in cache_steps:
                        self.assertRegex(step, r"(?m)^        continue-on-error: true$")
                    else:
                        # Cache outages may lose reuse, but actual target setup,
                        # Cargo checks and strict lints must remain blocking.
                        self.assertNotIn("continue-on-error", step)
                commands = self.cargo_commands(job)
                self.assertEqual(len(commands), 2)
                for command in commands:
                    if "fmt" not in command:
                        self.assertTrue({"--locked", "--release", "--target-dir"}.issubset(command))

    def test_scalar_runtime_control_and_required_cross_job_replace_duplicate_workflow(self):
        job = self.job("windows")
        commands = self.cargo_commands(job)
        self.assertTrue(any({"test", "bonsai-hexagon-runtime", "--all-targets", "--locked",
                             "--offline", "--release"}.issubset(command) for command in commands))
        # Foundation now owns all Windows package tests and graph projections;
        # there is no second workflow competing for the same hosted runners.
        self.assertFalse((ROOT / ".github/workflows/hexagon-native.yml").exists())
        self.assertIn("cross-native-arm64", self.job("foundation"))
        config = json.loads((ROOT / ".github/ci/impact.json").read_text(encoding="utf-8"))
        self.assertEqual(config["workflows"]["foundation"]["cross-native-arm64"], "hexagon")


if __name__ == "__main__":
    unittest.main()
