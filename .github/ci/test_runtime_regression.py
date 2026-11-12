"""Regression-tier verdicts cannot masquerade as full source qualification."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate
import runtime_regression as regression


class ReportTests(unittest.TestCase):
    def report(self):
        return {
            "SchemaVersion": 1,
            "Target": "rust",
            "Measure": "upstream-source-files",
            "UpstreamCommit": "a" * 40,
            "Cases": [{"Id": "source", "Status": "pass"}],
        }

    def test_report_must_pass_the_exact_source_and_identity(self):
        valid = self.report()
        self.assertEqual(regression.validate(valid, "source", "a" * 40), valid["Cases"][0])
        mutations = [None, [], {}]
        for field, value in (
            ("SchemaVersion", 2), ("Target", "cpp"),
            ("Measure", "benchmark-workloads"), ("UpstreamCommit", "b" * 40),
            ("Cases", []), ("Cases", valid["Cases"] * 2),
            ("Cases", [{"Id": "other", "Status": "pass"}]),
        ):
            report = copy.deepcopy(valid)
            report[field] = value
            mutations.append(report)
        for status in ("fail", "unsupported", "skip", None):
            report = copy.deepcopy(valid)
            report["Cases"][0]["Status"] = status
            mutations.append(report)
        for report in mutations:
            with self.subTest(report=report), self.assertRaises(ValueError):
                regression.validate(report, "source", "a" * 40)

    def run_controller(self, directory, status="pass", error=None):
        harness = directory / "harness.exe"
        harness.write_bytes(b"synthetic executable identity; never launched")
        evidence = directory / "evidence"
        root = Path(regression.__file__).resolve().parents[2]
        upstream = json.loads((root / "tools/prism/source.lock.json").read_text())["Commit"]

        def execute(command, **kwargs):
            if error:
                raise error
            report = self.report()
            report["UpstreamCommit"] = upstream
            report["Cases"] = [{"Id": command[4], "Status": status}]
            Path(command[6]).write_text(json.dumps(report))
            return subprocess.CompletedProcess(command, 0)

        with patch.dict(os.environ, {
            "GITHUB_ACTIONS": "true", "GITHUB_REF": "refs/heads/main",
            "GITHUB_SHA": "c" * 40, "GITHUB_STEP_SUMMARY": "",
        }), patch.object(sys, "argv", [
            "runtime_regression.py", "--harness", str(harness), "--evidence", str(evidence),
        ]), patch.object(regression, "CASES", ("source", "another")), patch.object(
            regression.subprocess, "run", side_effect=execute,
        ) as run:
            if status != "pass" or error:
                with self.assertRaises((ValueError, subprocess.TimeoutExpired)):
                    regression.main()
            else:
                regression.main()
        return json.loads((evidence / "regression.json").read_text()), run

    def test_success_retains_bounded_scope_and_binary_revision_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            report, run = self.run_controller(Path(directory))
        self.assertEqual(report["Status"], "pass")
        self.assertIs(report["FullSourceGate"], False)
        self.assertIs(report["NativeReferenceReplay"], False)
        self.assertEqual(report["ProjectRevision"], "c" * 40)
        self.assertEqual(len(report["HarnessSha256"]), 64)
        self.assertEqual([case["Id"] for case in report["Cases"]], report["ExpectedIds"])
        self.assertEqual(run.call_count, 2)
        self.assertTrue(all(0 < call.kwargs["timeout"] <= 30 for call in run.call_args_list))

    def test_zero_exit_assertion_failure_and_timeout_retain_failed_receipts(self):
        for status, error in (
            ("fail", None), ("unsupported", None),
            ("pass", subprocess.TimeoutExpired("synthetic", 30)),
        ):
            with self.subTest(status=status, error=error), tempfile.TemporaryDirectory() as directory:
                report, run = self.run_controller(Path(directory), status, error)
                self.assertEqual(report["Status"], "fail")
                self.assertIn("Error", report)
                self.assertIs(report["FullSourceGate"], False)
                self.assertEqual(run.call_count, 1)

    def test_regression_gate_rejects_failed_selected_jobs_and_full_gate_inputs(self):
        config = json.loads((Path(__file__).resolve().parent / "impact.json").read_text())
        jobs = config["workflows"]["foundation"]
        needs = {
            "changes": {"result": "success", "outputs": {"plan": json.dumps({
                "groups": {group: group == "conformance" for group in config["groups"]},
            })}},
            **{job: {"result": "success" if group == "conformance" else "skipped"}
               for job, group in jobs.items()},
        }
        self.assertFalse(gate.failures(needs, jobs))
        self.assertTrue(gate.failures(needs, config["workflows"]["runtime-conformance"]))
        for result in ("skipped", "failure", "cancelled"):
            needs["rust-regression"]["result"] = result
            self.assertTrue(gate.failures(needs, jobs))


if __name__ == "__main__":
    unittest.main()
