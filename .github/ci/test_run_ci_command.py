"""Focused streaming, exit-status and exact-control admission regressions."""

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

SPEC = importlib.util.spec_from_file_location("run_ci_command", Path(__file__).with_name("run_ci_command.py"))
ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ci)

RUST_PASS = ("running 1 test\ntest module::exact_control ... ok\n\n"
             "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 99 filtered out; finished in 0.00s\n")


class CiCommandTests(unittest.TestCase):
    def test_exact_control_accepts_lf_and_crlf(self):
        for output in (RUST_PASS, RUST_PASS.replace("\n", "\r\n")):
            self.assertIsNone(ci.validate_output(output, rust_test="module::exact_control"))

    def test_exact_control_rejects_absent_wrong_duplicate_ignored_and_failed_tests(self):
        for output in ("", RUST_PASS.replace("exact_control", "other_control"), RUST_PASS + RUST_PASS,
                       RUST_PASS.replace("... ok", "... ignored"),
                       RUST_PASS.replace("0 failed", "1 failed"),
                       RUST_PASS.replace("running 1 test", "running 2 tests"),
                       RUST_PASS.replace("1 passed", "0 passed")):
            with self.subTest(output=output):
                self.assertIsNotNone(ci.validate_output(output, rust_test="module::exact_control"))

    def test_unittest_control_requires_one_exact_count_and_plain_success(self):
        passing = "Ran 3 tests in 0.001s\n\nOK\n"
        self.assertIsNone(ci.validate_output(passing, unittest_count=3))
        for output in ("", passing.replace("3 tests", "2 tests"), passing + passing,
                       passing.replace("OK", "OK (skipped=1)"), passing.replace("OK", "FAILED (failures=1)")):
            self.assertIsNotNone(ci.validate_output(output, unittest_count=3))

    def test_command_flushes_diagnostics_before_waiting_for_exit(self):
        output = io.StringIO()
        process = MagicMock()
        process.__enter__.return_value = process
        def lines():
            yield "first diagnostic\n"
            self.assertEqual(output.getvalue(), "first diagnostic\n")
            yield "second diagnostic\n"
        process.stdout = lines()
        process.wait.return_value = 17
        with patch.object(ci.subprocess, "Popen", return_value=process) as popen:
            exit_code, captured = ci.stream_command(["compiler", "+pinned", "check"], output)
        self.assertEqual(exit_code, 17)
        self.assertEqual(captured, output.getvalue())
        self.assertIs(popen.call_args.kwargs["stderr"], ci.subprocess.STDOUT)

    def test_native_failure_preserves_exit_and_streams_output_before_rejection(self):
        output, errors = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, {}, clear=True), redirect_stdout(output), redirect_stderr(errors):
            result = ci.main(["--label", "synthetic-compiler", "--", sys.executable,
                              "-c", "import sys; print('primary diagnostic', flush=True); sys.exit(17)"])
        self.assertEqual(result, 17)
        self.assertIn("primary diagnostic", output.getvalue())
        self.assertIn("status 17", errors.getvalue())

    def run_fixture(self, root, exit_code, output, expect_test=False, spawn_error=False):
        environment = {"PRISM_CHECK_EVIDENCE": str(root / "evidence"),
                       "GITHUB_STEP_SUMMARY": str(root / "summary.md"),
                       "GITHUB_SHA": "a" * 40, "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "2",
                       "GITHUB_JOB": "bundle", "RUNNER_OS": "Windows", "RUNNER_ARCH": "ARM64",
                       "ImageOS": "win11-arm", "ImageVersion": "hosted-image",
                       "PRISM_LANE": "checks", "PRISM_PACKAGES": '["prism-inference"]',
                       "PRISM_CI_TARGET": "selected-target",
                       "PRISM_CI_TOOLCHAIN": "selected-toolchain", "PRISM_CI_SCOPE": "bundle only"}
        arguments = ["--label", "named-control"]
        if expect_test:
            arguments += ["--expect-rust-test", "module::exact_control"]
        arguments += ["--", "cargo", "+actual-pinned-toolchain", "test"]
        with patch.dict(os.environ, environment, clear=True), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with patch.object(ci, "stream_command", side_effect=OSError("compiler missing") if spawn_error else None,
                              return_value=(exit_code, output)):
                result = ci.main(arguments)
        ledger = json.loads((root / "evidence/command-results.json").read_text())
        return result, ledger["commands"][0]

    def test_valid_output_never_overrides_native_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, record = self.run_fixture(root, 23, RUST_PASS, expect_test=True)
            self.assertEqual(result, 23)
            self.assertEqual(record["exit_code"], 23)
            self.assertEqual(record["source_revision"], "a" * 40)
            self.assertEqual(record["run_attempt"], "2")
            self.assertEqual(record["target"], "selected-target")
            self.assertEqual(record["toolchain"], "actual-pinned-toolchain")
            self.assertEqual((root / "evidence" / record["failure_log"]).read_text(), RUST_PASS)
            summary = (root / "summary.md").read_text()
            self.assertIn("exit code: 23", summary)
            self.assertIn("command: cargo +actual-pinned-toolchain test", summary)
            self.assertIn("runner arch: ARM64", summary)
            self.assertIn("image version: hosted-image", summary)
            self.assertIn('selected packages: ["prism-inference"]', summary)

    def test_zero_exit_with_missing_named_control_remains_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, record = self.run_fixture(Path(temporary), 0, "running 0 tests\n", expect_test=True)
            self.assertEqual(result, 1)
            self.assertEqual(record["exit_code"], 0)
            self.assertIn("exactly once", record["failure"])

    def test_success_retains_only_ledger_without_build_chatter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, record = self.run_fixture(root, 0, RUST_PASS, expect_test=True)
            self.assertEqual(result, 0)
            self.assertIsNone(record["failure"])
            self.assertEqual([p.name for p in (root / "evidence").iterdir()], ["command-results.json"])
            self.assertFalse((root / "summary.md").exists())

    def test_empty_failure_has_no_placeholder_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, record = self.run_fixture(root, 4, "   \n")
            self.assertEqual(result, 4)
            self.assertNotIn("failure_log", record)
            self.assertEqual([p.name for p in (root / "evidence").iterdir()], ["command-results.json"])

    def test_spawn_error_retains_unknown_exit_and_actionable_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, record = self.run_fixture(root, 0, "", spawn_error=True)
            self.assertEqual(result, 1)
            self.assertIsNone(record["exit_code"])
            self.assertIn("compiler missing", record["failure"])
            self.assertIn("exit code: unknown", (root / "summary.md").read_text())

    def test_python_command_uses_declared_target_and_toolchain(self):
        result = ci.context("controls", ["python", "-B", "test.py"],
                            {"PRISM_CI_TARGET": "arm64", "PRISM_CI_TOOLCHAIN": "nightly-pinned"})
        self.assertEqual(result["target"], "arm64")
        self.assertEqual(result["toolchain"], "nightly-pinned")

    def test_evidence_rejects_workspace_and_relative_paths(self):
        for directory in ("relative-evidence", str(Path(ci.__file__).resolve().parents[2])):
            with self.assertRaises(ValueError):
                ci.retain_result({}, "", directory)


if __name__ == "__main__":
    unittest.main()
