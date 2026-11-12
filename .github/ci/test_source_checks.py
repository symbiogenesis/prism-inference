"""Focused routing and negative aggregate controls for consolidated source checks."""

import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_checks as checks  # noqa: E402


MAIN = "refs/heads/main"
MERGE = "refs/pull/7/merge"
BEFORE = "a" * 40
AFTER = "b" * 40
HEAD = "c" * 40


class SelectionTests(unittest.TestCase):
    def selected(self, *paths, event="push", git_ref=MAIN, **kwargs):
        return checks.select(paths, event, git_ref, **kwargs)["groups"]

    def test_documentation_push_runs_markdown_without_native_jobs(self):
        selected = self.selected("README.md", "docs/nested/contract.md")
        self.assertEqual(selected, {"markdown": True, "standards": False,
                                    "probe": False, "parity": False})

    def test_script_and_csharp_changes_keep_main_only_standards(self):
        for path in ("tests/Test-Current.ps1", "scripts/deep/Module.psm1",
                     "PSScriptAnalyzerSettings.psd1", "tools/validation/Rules.cs",
                     "tools/validation/project.json", ".editorconfig", "global.json"):
            with self.subTest(path=path):
                self.assertTrue(self.selected(path)["standards"])
                self.assertFalse(self.selected(path, event="pull_request", git_ref=MERGE)["standards"])

    def test_compiler_inputs_keep_stable_nightly_comparison(self):
        for path in ("tools/rust-trial/fixtures/borrowing.rs", "config/rust-nightly-trial.json",
                     "rust-toolchain.toml", "tests/Test-RustNightly.ps1",
                     "scripts/Bonsai.Common.psm1", "scripts/Bonsai.Native.psm1",
                     ".github/actions/rust-toolchain/action.yml"):
            with self.subTest(path=path):
                self.assertTrue(self.selected(path)["probe"])
                self.assertFalse(self.selected(path, event="pull_request", git_ref=MERGE)["probe"])

    def test_runtime_parity_keeps_pr_and_manual_scope_without_duplicate_push(self):
        for path in ("Cargo.toml", "Cargo.lock", "src/runtime.rs", "server/src/main.rs",
                     "tools/gguf/src/lib.rs", "tools/native-hash/src/lib.rs", "tests/swift_session.rs"):
            with self.subTest(path=path):
                self.assertFalse(self.selected(path)["parity"])
                self.assertTrue(self.selected(path, event="pull_request", git_ref=MERGE)["parity"])

    def test_mixed_changes_select_each_eligible_owner(self):
        selected = self.selected("docs/contract.md", "src/lib.rs", "tests/Test-Check.ps1",
                                 event="pull_request", git_ref=MERGE)
        self.assertEqual(selected, {"markdown": True, "standards": False,
                                    "probe": False, "parity": True})

    def test_shared_workflow_and_helper_edits_run_every_eligible_check(self):
        for path in (".github/workflows/source-checks.yml", ".github/ci/source_checks.py",
                     ".github/ci/test_source_checks.py", ".github/ci/impact.py", ".github/ci/gate.py"):
            with self.subTest(path=path):
                self.assertEqual(self.selected(path), {"markdown": True, "standards": True,
                                                      "probe": True, "parity": False})
                self.assertEqual(self.selected(path, event="pull_request", git_ref=MERGE),
                                 {"markdown": True, "standards": False, "probe": False, "parity": True})

    def test_empty_or_unrelated_changes_skip_all_jobs(self):
        self.assertFalse(any(self.selected().values()))
        self.assertFalse(any(self.selected("tools/ternary/src/lib.rs").values()))

    def test_manual_main_runs_all_and_other_refs_keep_original_branch_boundary(self):
        self.assertTrue(all(checks.plan(Path('.'), "workflow_dispatch", MAIN, AFTER)["groups"].values()))
        self.assertEqual(checks.plan(Path('.'), "workflow_dispatch", "refs/heads/feature", AFTER)["groups"],
                         {"markdown": True, "standards": False, "probe": False, "parity": True})

    def test_unknown_event_is_rejected(self):
        with self.assertRaises(ValueError):
            checks.plan(Path('.'), "pull_request_target", MAIN, AFTER)


class BaselineTests(unittest.TestCase):
    def test_push_carries_cancelled_changes_from_successful_workflow_ancestor(self):
        with patch.object(checks.impact, "successful_base", return_value=BEFORE) as ancestor, \
             patch.object(checks.impact, "git", return_value=b"docs/earlier.md\0tests/Test-Later.ps1\0") as diff:
            result = checks.plan(Path('.'), "push", MAIN, AFTER)
        ancestor.assert_called_once_with(Path('.'), "source-checks.yml", AFTER)
        diff.assert_called_once_with(Path('.'), "diff", "--no-renames", "--name-only", "-z", BEFORE, AFTER, "--")
        self.assertTrue(result["groups"]["markdown"])
        self.assertTrue(result["groups"]["standards"])
        self.assertFalse(result["full"])

    def test_missing_successful_push_baseline_runs_all_eligible_checks(self):
        for baseline in (None, "", "invalid"):
            with self.subTest(baseline=baseline), \
                 patch.object(checks.impact, "successful_base", return_value=baseline), \
                 patch.object(checks.impact, "git") as diff:
                result = checks.plan(Path('.'), "push", MAIN, AFTER)
                self.assertTrue(result["full"])
                self.assertEqual(result["groups"], {"markdown": True, "standards": True,
                                                    "probe": True, "parity": False})
                diff.assert_not_called()

    def test_git_failure_selects_all_eligible_jobs_instead_of_silent_skip(self):
        for failure in (FileNotFoundError("git missing"), subprocess.CalledProcessError(128, "git")):
            with self.subTest(failure=failure), \
                 patch.object(checks.impact, "git", side_effect=failure):
                result = checks.plan(Path('.'), "pull_request", MERGE, AFTER, base=BEFORE, head=HEAD)
                self.assertTrue(result["full"])
                self.assertTrue(result["groups"]["markdown"])
                self.assertTrue(result["groups"]["parity"])
                self.assertFalse(result["groups"]["standards"])
                self.assertFalse(result["groups"]["probe"])

    def test_pr_uses_merge_base_and_both_move_paths_without_querying_push_history(self):
        with patch.object(checks.impact, "successful_base") as ancestor, \
             patch.object(checks.impact, "git", return_value=b"src/old.rs\0docs/new.md\0") as diff:
            result = checks.plan(Path('.'), "pull_request", MERGE, AFTER, base=BEFORE, head=HEAD)
        ancestor.assert_not_called()
        diff.assert_called_once_with(Path('.'), "diff", "--no-renames", "--name-only", "-z",
                                     BEFORE + "..." + HEAD, "--")
        self.assertTrue(result["groups"]["markdown"])
        self.assertTrue(result["groups"]["parity"])
        self.assertFalse(result["full"])

    def test_pr_identity_failure_conservatively_selects_eligible_checks(self):
        for base, head in (("", HEAD), (BEFORE, ""), ("--output=unexpected", HEAD)):
            with self.subTest(base=base, head=head), patch.object(checks.impact, "git") as diff:
                result = checks.plan(Path('.'), "pull_request", MERGE, AFTER, base=base, head=head)
                self.assertTrue(result["full"])
                diff.assert_not_called()

    def test_unreadable_pr_payload_falls_back_to_all_eligible_jobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            environment = {"GITHUB_EVENT_NAME": "pull_request", "GITHUB_REF": MERGE,
                           "GITHUB_SHA": AFTER, "GITHUB_EVENT_PATH": str(Path(temporary) / "absent.json")}
            with patch.dict(checks.os.environ, environment, clear=True), \
                 patch.object(sys, "argv", ["source_checks.py"]), patch("sys.stdout", new_callable=io.StringIO) as output:
                checks.main()
            result = json.loads(output.getvalue())
            self.assertTrue(result["full"])
            self.assertEqual(result["groups"], {"markdown": True, "standards": False,
                                                "probe": False, "parity": True})


class AggregateTests(unittest.TestCase):
    def needs(self, groups):
        result = {name: {"result": "success" if groups[group] else "skipped"}
                  for name, group in checks.JOBS.items()}
        result["changes"] = {"result": "success", "outputs": {"plan": json.dumps({"groups": groups})}}
        return result

    def test_gate_accepts_only_success_for_selected_checks_including_compatibility(self):
        groups = {"markdown": True, "standards": True, "probe": True, "parity": False}
        needs = self.needs(groups)
        self.assertEqual(checks.failures(needs), [])
        for job in ("markdown", "standards", "probe", "compatibility"):
            for status in ("failure", "cancelled", "skipped"):
                with self.subTest(job=job, status=status):
                    changed = copy.deepcopy(needs)
                    changed[job]["result"] = status
                    self.assertTrue(checks.failures(changed))

    def test_gate_rejects_selected_parity_skip_and_unselected_job_failure(self):
        groups = {"markdown": False, "standards": False, "probe": False, "parity": True}
        needs = self.needs(groups)
        self.assertEqual(checks.failures(needs), [])
        needs["model-free"]["result"] = "skipped"
        self.assertTrue(checks.failures(needs))
        needs = self.needs(groups)
        needs["standards"]["result"] = "failure"
        self.assertTrue(checks.failures(needs))

    def test_failed_planner_missing_dependency_and_malformed_plan_reject(self):
        needs = self.needs(dict.fromkeys(checks.GROUPS, False))
        for status in ("failure", "cancelled", "skipped"):
            with self.subTest(status=status):
                changed = copy.deepcopy(needs)
                changed["changes"]["result"] = status
                self.assertTrue(checks.failures(changed))
        del needs["compatibility"]
        self.assertTrue(checks.failures(needs))
        for groups in ({}, dict.fromkeys(checks.GROUPS, "false"),
                       dict.fromkeys((*checks.GROUPS, "unexpected"), False)):
            with self.subTest(groups=groups):
                changed = self.needs(dict.fromkeys(checks.GROUPS, False))
                changed["changes"]["outputs"]["plan"] = json.dumps({"groups": groups})
                self.assertTrue(checks.failures(changed))


if __name__ == "__main__":
    unittest.main()
