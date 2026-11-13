"""Focused failure-context checks; reporting must not change gate verdicts."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate


class GateReportTests(unittest.TestCase):
    def test_summary_distinguishes_dependency_failure_from_aggregate(self):
        needs = {"changes": {"result": "success", "outputs": {"plan": json.dumps({"groups": {"rust": True}})}},
                 "windows": {"result": "failure"}}
        errors = gate.failures(needs, {"windows": "rust"})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary.md"
            with patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": str(output), "GITHUB_SHA": "a" * 40,
                                         "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "2"}):
                gate.report(needs, errors, "foundation")
            text = output.read_text(encoding="utf-8")
        self.assertIn('"windows" | "failure"', text)
        self.assertIn("first failing step", text)
        self.assertIn("a" * 40, text)
        self.assertIn('GITHUB_RUN_ATTEMPT: "2"', text)
        self.assertEqual(errors, ["windows: selected=True, result=failure"])
        self.assertEqual(gate.failures(needs, {"windows": "rust"}), errors)

    def test_malformed_dependency_is_reported_without_hiding_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary.md"
            with patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": str(output)}):
                gate.report({"bad|name": None}, ["missing"], "foundation")
            text = output.read_text(encoding="utf-8")
        self.assertIn('"bad&#124;name" | "malformed"', text)
        self.assertIn("Rejected", text)

    def test_absent_summary_needs_no_output_file(self):
        with patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": ""}):
            gate.report({}, ["failure"], "foundation")


if __name__ == "__main__":
    unittest.main()
