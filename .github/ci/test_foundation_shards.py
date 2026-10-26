"""Focused regressions for exhaustive correctness sharding, without Cargo tests."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import foundation_shards as shards  # noqa: E402


def target(name, kind="test", test=True, doctest=False, required=()):
    return {"name": name, "kind": [kind], "test": test, "doctest": doctest,
            "required-features": list(required)}


def metadata():
    return {
        "version": 1,
        "workspace_members": ["runtime-id", "tool-id"],
        "packages": [
            {"name": "runtime", "id": "runtime-id", "features": {"default": ["cli"], "cli": [], "native": []},
             "targets": [target("runtime", "lib", doctest=True), target("worker", "bin"),
                         target("capture", "example", required=["cli"]), target("protocol"),
                         target("sampling"), target("tokens"), target("native", required=["native"]),
                         target("ignored-worker", "bin", test=False)]},
            {"name": "tool", "id": "tool-id", "features": {},
             "targets": [target("tool", "lib", test=False, doctest=True), target("protocol"),
                         target("model"), target("codec"), target("build", "custom-build", test=False)]},
        ],
    }


class FoundationShardTests(unittest.TestCase):
    def test_every_enabled_target_is_assigned_once_with_same_name_together(self):
        data = metadata()
        expected, skipped = shards.inventory(data, ["runtime", "tool"])
        plan = shards.partition(data, ["runtime", "tool"], 4)
        actual = [test for shard in plan["shards"] for test in shard]
        self.assertCountEqual(actual, expected)
        self.assertEqual(len(actual), len(set(actual)))
        owners = [index for index, shard in enumerate(plan["shards"]) if any(test[2] == "protocol" for test in shard)]
        self.assertEqual(len(owners), 1)
        self.assertIn(("runtime", "example", "capture"), plan["shards"][0])
        self.assertIn(("runtime", "doc", "runtime"), plan["shards"][0])
        self.assertIn(("tool", "doc", "tool"), plan["shards"][0])
        self.assertEqual(len(skipped), 1)

    def test_commands_preserve_selected_package_feature_unification(self):
        plan = shards.partition(metadata(), ["runtime", "tool"], 4)
        for index in range(4):
            commands = shards.commands(plan, ["runtime", "tool"], index)
            for command in commands:
                self.assertEqual(command[:4], ["--package", "runtime", "--package", "tool"])
            if index:
                self.assertFalse(any("--doc" in command or "--lib" in command for command in commands))
        commands = shards.commands(plan, ["runtime", "tool"], 0)
        self.assertIn(["--package", "runtime", "--package", "tool", "--doc"], commands)
        self.assertFalse(any("ignored-worker" in command for command in commands))

    def test_new_target_automatically_enters_exactly_one_shard(self):
        data = metadata()
        before = shards.partition(data, ["runtime", "tool"], 4)
        data["packages"][0]["targets"].append(target("newly_added_regression"))
        after = shards.partition(data, ["runtime", "tool"], 4)
        before_count = sum(map(len, before["shards"]))
        after_tests = [test for shard in after["shards"] for test in shard]
        self.assertEqual(len(after_tests), before_count + 1)
        self.assertEqual(after_tests.count(("runtime", "test", "newly_added_regression")), 1)

    def test_malformed_unknown_duplicate_and_empty_package_inputs_fail(self):
        for packages in ([], ["unknown"], ["runtime", "runtime"]):
            with self.subTest(packages=packages), self.assertRaises(ValueError):
                shards.partition(metadata(), packages, 4)
        for change in ("kind", "flag", "duplicate", "feature", "version"):
            data = metadata()
            if change == "kind":
                data["packages"][0]["targets"][0]["kind"] = ["future-target-kind"]
            elif change == "flag":
                data["packages"][0]["targets"][0]["test"] = "true"
            elif change == "duplicate":
                data["packages"][0]["targets"].append(copy.deepcopy(data["packages"][0]["targets"][0]))
            elif change == "feature":
                data["packages"][0]["targets"][0]["required-features"] = ["missing"]
            else:
                data["version"] = 2
            with self.subTest(change=change), self.assertRaises(ValueError):
                shards.partition(data, ["runtime"], 4)

    def test_dependency_activated_gated_targets_fail_closed_including_aliases(self):
        for style in ("direct", "alias", "feature-alias"):
            data = metadata()
            data["packages"][1]["dependencies"] = [{"name": "runtime", "rename": "engine", "features": []}]
            if style == "direct":
                data["packages"][1]["dependencies"][0]["features"] = ["native"]
            elif style == "alias":
                data["packages"][1]["features"] = {"enable": ["engine/native"]}
            else:
                data["packages"][0]["features"]["forwarded"] = ["native"]
                data["packages"][1]["dependencies"][0]["features"] = ["forwarded"]
            with self.subTest(style=style), self.assertRaisesRegex(ValueError, "Dependency-activated"):
                shards.partition(data, ["runtime", "tool"], 4)

    def test_default_feature_alias_cycles_and_invalid_shard_bounds(self):
        data = metadata()
        data["packages"][0]["features"] = {"default": ["alias"], "alias": ["cli"], "cli": ["alias"], "native": []}
        self.assertEqual(shards.default_features(data["packages"][0]), {"default", "alias", "cli"})
        for count in (0, -1, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                shards.partition(data, ["runtime"], count)
        plan = shards.partition(data, ["runtime"], 4)
        for index in (-1, 4):
            with self.subTest(index=index), self.assertRaises(ValueError):
                shards.commands(plan, ["runtime"], index)


if __name__ == "__main__":
    unittest.main()
