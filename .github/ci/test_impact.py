"""Regressions for crate-aware CI routing without invoking Cargo or full suites."""

import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate  # noqa: E402
import impact  # noqa: E402


class WorkspaceCase(unittest.TestCase):
    """Build synthetic manifest graphs in external temporary directories."""

    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="ci-impact-regression-")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.config = {
            "groups": {"rust": {"packages": ["*"]}},
            "shared": ["Cargo.toml", "Cargo.lock", "rust-toolchain*", ".cargo/**"],
            "ignored": ["*.md", "docs/**"],
        }

    def write(self, path, text):
        destination = self.root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")

    def manifest(self, directory, name, dependencies=""):
        self.write(
            f"{directory}/Cargo.toml" if directory else "Cargo.toml",
            f'[package]\nname = "{name}"\nversion = "0.1.0"\n{dependencies}\n',
        )

    def workspace(self, members, extra="", root_package=False):
        package = '[package]\nname = "root"\nversion = "0.1.0"\n' if root_package else ""
        self.write(
            "Cargo.toml",
            package + "[workspace]\nmembers = " + json.dumps(members) + "\n" + extra,
        )

    def plan(self, *paths, **kwargs):
        return impact.select(self.root, paths, config=self.config, **kwargs)


class DependencySelectionTests(WorkspaceCase):
    def test_implicit_workspace_path_members_are_discovered_recursively(self):
        self.workspace(["app", "other"])
        self.manifest("app", "app", '[dependencies]\nmiddle = { path = "../middle" }')
        self.manifest("middle", "middle", '[dependencies]\nleaf = { path = "../leaf" }')
        self.manifest("leaf", "leaf")
        self.manifest("other", "other")
        plan = self.plan("leaf/src/lib.rs")
        self.assertEqual(plan["packages"], ["app", "leaf", "middle"])
        self.assertFalse(plan["full"])

    def test_leaf_change_selects_transitive_consumers_without_sibling(self):
        self.workspace(["leaf", "middle", "app", "other"])
        self.manifest("leaf", "leaf")
        self.manifest("middle", "middle", '[dependencies]\nleaf = { path = "../leaf" }')
        self.manifest("app", "app", '[dependencies]\nmiddle = { path = "../middle" }')
        self.manifest("other", "other")
        plan = self.plan("leaf/src/lib.rs")
        self.assertEqual(plan["packages"], ["app", "leaf", "middle"])
        self.assertTrue(plan["groups"]["rust"])
        self.assertFalse(plan["full"])

    def test_aliases_and_each_dependency_kind_reach_consumers(self):
        dependencies = {
            "runtime": '[dependencies]\nrenamed = { package = "leaf", path = "../leaf" }',
            "development": '[dev-dependencies]\nleaf = { path = "../leaf" }',
            "build": '[build-dependencies]\nleaf = { path = "../leaf" }',
            "target": '[target.\'cfg(windows)\'.dependencies]\nleaf = { path = "../leaf" }',
            "target-development": '[target.\'cfg(unix)\'.dev-dependencies]\nleaf = { path = "../leaf" }',
            "target-build": '[target.\'cfg(windows)\'.build-dependencies]\nleaf = { path = "../leaf" }',
        }
        self.workspace(["leaf", *dependencies])
        self.manifest("leaf", "leaf")
        for directory, text in dependencies.items():
            self.manifest(directory, directory, text)
        self.assertEqual(
            set(self.plan("leaf/src/lib.rs")["packages"]), {"leaf", *dependencies}
        )

    def test_inherited_alias_path_is_relative_to_workspace_root(self):
        self.workspace(
            ["crates/leaf", "applications/app"],
            '[workspace.dependencies]\nrenamed = { package = "leaf", path = "crates/leaf" }\n',
        )
        self.manifest("crates/leaf", "leaf")
        self.manifest("applications/app", "app", "[dependencies]\nrenamed.workspace = true")
        self.assertEqual(self.plan("crates/leaf/src/lib.rs")["packages"], ["app", "leaf"])

    def test_nested_member_owns_file_instead_of_root_package(self):
        self.workspace(["src/nested"], root_package=True)
        self.manifest("src/nested", "nested")
        self.assertEqual(self.plan("src/nested/src/lib.rs")["packages"], ["nested"])
        self.assertEqual(self.plan("src/main.rs")["packages"], ["root"])

    def test_move_between_crates_selects_old_and_new_dependents(self):
        self.workspace(["old", "new", "old-app", "new-app", "other"])
        for name in ("old", "new", "other"):
            self.manifest(name, name)
        for name in ("old", "new"):
            self.manifest(
                f"{name}-app", f"{name}-app", f'[dependencies]\n{name} = {{ path = "../{name}" }}'
            )
        # git diff --no-renames supplies both delete and add paths for a move.
        plan = self.plan("old/src/moved.rs", "new/src/moved.rs")
        self.assertEqual(plan["packages"], ["new", "new-app", "old", "old-app"])
        self.assertFalse(plan["full"])

    def test_deleted_regular_source_still_selects_owner(self):
        self.workspace(["leaf", "other"])
        self.manifest("leaf", "leaf")
        self.manifest("other", "other")
        self.assertFalse((self.root / "leaf/src/deleted.rs").exists())
        self.assertEqual(self.plan("leaf/src/deleted.rs")["packages"], ["leaf"])


class RoutingPolicyTests(WorkspaceCase):
    def setUp(self):
        super().setUp()
        self.workspace(["leaf", "other"])
        self.manifest("leaf", "leaf")
        self.manifest("other", "other")

    def test_existing_crate_manifest_selects_only_affected_crate(self):
        plan = self.plan("leaf/Cargo.toml")
        self.assertEqual(plan["packages"], ["leaf"])
        self.assertFalse(plan["full"])

    def test_unregistered_or_removed_manifest_requires_full_validation(self):
        for path in ("new/Cargo.toml", "removed/Cargo.toml"):
            with self.subTest(path=path):
                plan = self.plan(path)
                self.assertTrue(plan["full"])
                self.assertEqual(plan["packages"], ["leaf", "other"])

    def test_shared_build_inputs_require_all_groups(self):
        self.config["groups"]["scripts"] = {"paths": ["scripts/**"]}
        for path in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".cargo/config.toml"):
            with self.subTest(path=path):
                plan = self.plan(path)
                self.assertTrue(plan["full"])
                self.assertTrue(all(plan["groups"].values()))

    def test_root_documentation_does_not_start_expensive_jobs(self):
        plan = self.plan("README.md", "docs/validation.md")
        self.assertEqual(plan["packages"], [])
        self.assertFalse(any(plan["groups"].values()))
        self.assertFalse(plan["full"])

    def test_mixed_documentation_and_code_keeps_code_coverage(self):
        plan = self.plan("docs/validation.md", "leaf/src/lib.rs")
        self.assertEqual(plan["packages"], ["leaf"])
        self.assertFalse(plan["full"])

    def test_unknown_input_requires_conservative_full_validation(self):
        plan = self.plan("new-code/contract.json")
        self.assertTrue(plan["full"])
        self.assertEqual(plan["packages"], ["leaf", "other"])

    def test_lint_observation_does_not_hide_unclassified_behavior(self):
        self.config["groups"]["standards"] = {"observes": ["scripts/**/*.ps1"]}
        self.config["groups"]["scripts"] = {"paths": ["scripts/known.ps1"]}
        unknown = self.plan("scripts/new-helper.ps1")
        self.assertTrue(unknown["full"])
        known = self.plan("scripts/known.ps1")
        self.assertFalse(known["full"])
        self.assertEqual(known["packages"], [])
        self.assertTrue(known["groups"]["scripts"])
        self.assertTrue(known["groups"]["standards"])

    def test_explicit_full_validation_ignores_empty_diff(self):
        self.config["groups"]["scripts"] = {"paths": ["scripts/**"]}
        plan = self.plan(full=True)
        self.assertTrue(plan["full"])
        self.assertEqual(plan["packages"], ["leaf", "other"])
        self.assertTrue(all(plan["groups"].values()))

    def test_package_specific_shared_input_selects_its_consumers(self):
        self.manifest("other", "other", '[dependencies]\nleaf = { path = "../leaf" }')
        self.config["package_inputs"] = {"leaf": ["fixtures/leaf/**"]}
        plan = self.plan("fixtures/leaf/golden.json")
        self.assertEqual(plan["packages"], ["leaf", "other"])
        self.assertFalse(plan["full"])

    def test_group_dependencies_expand_transitively(self):
        self.config["groups"] = {
            "release": {"groups": ["scripts"]},
            "scripts": {"groups": ["tools"]},
            "tools": {"packages": ["leaf"]},
            "unrelated": {"packages": ["other"]},
        }
        plan = self.plan("leaf/src/lib.rs")
        self.assertEqual(
            plan["groups"], {"release": True, "scripts": True, "tools": True, "unrelated": False}
        )

    def test_package_path_override_does_not_claim_unrelated_root_files(self):
        self.config["package_paths"] = {"leaf": ["leaf/src/**", "shared/leaf.rs"]}
        self.assertEqual(self.plan("shared/leaf.rs")["packages"], ["leaf"])
        self.assertTrue(self.plan("leaf/unknown-input.bin")["full"])


class AggregateGateTests(unittest.TestCase):
    def needs(self, selected, result="success"):
        return {
            "changes": {
                "result": "success",
                "outputs": {"plan": json.dumps({"groups": {"rust": selected}})},
            },
            "build": {"result": result},
        }

    def test_selected_job_must_succeed(self):
        for result in ("success", "skipped", "failure", "cancelled"):
            with self.subTest(result=result):
                errors = gate.failures(self.needs(True, result), {"build": "rust"})
                self.assertEqual(bool(errors), result != "success")

    def test_only_unselected_skips_are_allowed(self):
        for result in ("success", "skipped", "failure", "cancelled"):
            with self.subTest(result=result):
                errors = gate.failures(self.needs(False, result), {"build": "rust"})
                self.assertEqual(bool(errors), result not in ("success", "skipped"))

    def test_failed_planner_and_malformed_results_cannot_pass(self):
        valid = self.needs(True)
        cases = []
        for result in ("skipped", "failure", "cancelled"):
            needs = copy.deepcopy(valid)
            needs["changes"]["result"] = result
            cases.append(needs)
        missing_job = copy.deepcopy(valid)
        del missing_job["build"]
        cases.append(missing_job)
        unknown_job = copy.deepcopy(valid)
        unknown_job["unchecked"] = {"result": "failure"}
        cases.append(unknown_job)
        missing_plan = copy.deepcopy(valid)
        del missing_plan["changes"]["outputs"]["plan"]
        cases.append(missing_plan)
        for plan in ("not-json", "[]", '{"groups":{"rust":"false"}}'):
            needs = copy.deepcopy(valid)
            needs["changes"]["outputs"]["plan"] = plan
            cases.append(needs)
        for index, needs in enumerate(cases):
            with self.subTest(case=index):
                self.assertTrue(gate.failures(needs, {"build": "rust"}))


class SuccessfulBaselineTests(unittest.TestCase):
    def environment(self):
        return patch.dict(
            impact.os.environ,
            {
                "GITHUB_REPOSITORY": "owner/project",
                "GITHUB_TOKEN": "unit-test-token",
                "GITHUB_EVENT_NAME": "push",
                "GITHUB_OUTPUT": "",
                "GITHUB_STEP_SUMMARY": "",
            },
        )

    def test_newest_successful_ancestor_skips_current_and_unrelated_heads(self):
        after, unrelated, newest, older = (character * 40 for character in "abcd")
        response = io.BytesIO(
            json.dumps(
                {"workflow_runs": [{"head_sha": sha} for sha in (after, "invalid", unrelated, newest, older)]}
            ).encode()
        )

        def ancestry(command, **kwargs):
            if command[-2] == unrelated:
                raise impact.subprocess.CalledProcessError(1, command)
            return impact.subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

        with self.environment(), patch.object(impact.urllib.request, "urlopen", return_value=response), patch.object(
            impact.subprocess, "run", side_effect=ancestry
        ) as run:
            baseline = impact.successful_base(Path(tempfile.gettempdir()), "foundation.yml", after)
        self.assertEqual(baseline, newest)
        self.assertEqual([call.args[0][-2] for call in run.call_args_list], [unrelated, newest])

    def test_missing_token_and_unavailable_api_return_no_baseline(self):
        with self.environment(), patch.dict(impact.os.environ, {"GITHUB_TOKEN": ""}), patch.object(
            impact.urllib.request, "urlopen"
        ) as request:
            self.assertIsNone(impact.successful_base(Path(tempfile.gettempdir()), "foundation.yml", "a" * 40))
            request.assert_not_called()
        with self.environment(), patch.object(
            impact.urllib.request, "urlopen", side_effect=impact.urllib.error.URLError("unavailable")
        ):
            self.assertIsNone(impact.successful_base(Path(tempfile.gettempdir()), "foundation.yml", "a" * 40))

    def invoke_main(self, arguments, baseline=None, diff=b"leaf/src/lib.rs\0", event="push"):
        plan = {"packages": [], "groups": {"rust": False}, "full": False, "reason": "test"}
        with self.environment(), patch.dict(impact.os.environ, {"GITHUB_EVENT_NAME": event}), patch.object(
            sys, "argv", ["impact.py", *arguments]
        ), patch.object(impact, "successful_base", return_value=baseline) as successful, patch.object(
            impact, "git", return_value=diff
        ) as git, patch.object(impact, "select", return_value=plan) as select, patch("sys.stdout", new_callable=io.StringIO):
            impact.main()
        return successful, git, select

    def test_unavailable_workflow_baseline_requires_full_instead_of_event_parent(self):
        successful, git, select = self.invoke_main(
            ["--workflow", "foundation.yml", "--before", "b" * 40, "--after", "a" * 40]
        )
        successful.assert_called_once()
        git.assert_not_called()
        self.assertTrue(select.call_args.kwargs["full"])

    def test_workflow_uses_validated_ancestor_instead_of_event_parent(self):
        baseline, after = "c" * 40, "a" * 40
        _, git, select = self.invoke_main(
            ["--workflow", "foundation.yml", "--before", "b" * 40, "--after", after], baseline=baseline
        )
        self.assertEqual(git.call_args.args[1:], ("diff", "--no-renames", "--name-only", "-z", baseline, after, "--"))
        self.assertEqual(select.call_args.args[1], ["leaf/src/lib.rs"])
        self.assertFalse(select.call_args.kwargs["full"])

    def test_manual_dispatch_always_selects_full_without_baseline_lookup(self):
        successful, git, select = self.invoke_main(
            ["--workflow", "foundation.yml", "--before", "b" * 40], event="workflow_dispatch"
        )
        successful.assert_not_called()
        git.assert_not_called()
        self.assertTrue(select.call_args.kwargs["full"])

    def test_without_workflow_before_is_used_and_zero_before_requires_full(self):
        successful, git, select = self.invoke_main(["--before", "b" * 40, "--after", "a" * 40])
        successful.assert_not_called()
        self.assertEqual(git.call_args.args[-3:], ("b" * 40, "a" * 40, "--"))
        self.assertFalse(select.call_args.kwargs["full"])
        _, git, select = self.invoke_main(["--before", "0" * 40])
        git.assert_not_called()
        self.assertTrue(select.call_args.kwargs["full"])


if __name__ == "__main__":
    unittest.main()
