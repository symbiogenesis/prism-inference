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


def unit_graph(data, enabled=None):
    """Cargo root fixture: no unselected or host-only dependency units are roots."""
    enabled = enabled or {package["name"]: shards.default_features(package) for package in data["packages"]}
    units = []
    for package in data["packages"]:
        for item in package["targets"]:
            if (item["test"] and item["kind"][0] not in {"bench", "custom-build"}
                    and set(item["required-features"]) <= enabled[package["name"]]):
                units.append({"pkg_id": package["id"], "platform": "native-target", "mode": "test",
                              "target": {"name": item["name"], "kind": item["kind"]},
                              "features": sorted(enabled[package["name"]])})
    return {"version": 1, "units": units, "roots": list(range(len(units)))}


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

    def test_inactive_optional_dependency_forwarding_does_not_enable_gated_target(self):
        data = metadata()
        data["packages"][1]["dependencies"] = [{"name": "runtime", "rename": "engine", "features": []}]
        data["packages"][1]["features"] = {"optional": ["engine/native"], "weak": ["engine?/native"]}
        enabled = shards.resolved_features(data, ["runtime", "tool"], unit_graph(data), "native-target")
        plan = shards.partition(data, ["runtime", "tool"], 4, enabled)
        self.assertNotIn(("runtime", "test", "native"), [item for part in plan["shards"] for item in part])
        self.assertIn(["runtime", "test", "native", "disabled required features"], plan["disabled"])

    def test_dependency_activated_gated_target_is_inventoried_exactly_once(self):
        data = metadata()
        enabled = {"runtime": {"default", "cli", "native"}, "tool": set()}
        resolved = shards.resolved_features(data, ["runtime", "tool"], unit_graph(data, enabled), "native-target")
        plan = shards.partition(data, ["runtime", "tool"], 4, resolved)
        self.assertEqual([item for part in plan["shards"] for item in part].count(("runtime", "test", "native")), 1)

    def test_host_dependency_features_are_not_merged_with_selected_target_roots(self):
        data = metadata()
        graph = unit_graph(data)
        host = copy.deepcopy(graph["units"][0])
        host["platform"] = None
        host["features"].append("native")
        graph["units"].append(host)
        resolved = shards.resolved_features(data, ["runtime", "tool"], graph, "native-target")
        self.assertNotIn("native", resolved["runtime"])
        graph["roots"].append(len(graph["units"]) - 1)
        with self.assertRaisesRegex(ValueError, "target mismatch"):
            shards.resolved_features(data, ["runtime", "tool"], graph, "native-target")

    def test_inconsistent_missing_and_unselected_unit_roots_fail_with_context(self):
        for change, message in (("features", "feature sets disagree"), ("missing", "omits enabled test"),
                                ("package", "outside selected packages"), ("target", "absent from metadata"),
                                ("index", "invalid root index"), ("version", "unit graph v1")):
            data = metadata()
            graph = unit_graph(data)
            if change == "features":
                graph["units"][1]["features"].append("native")
            elif change == "missing":
                graph["roots"].pop(0)
            elif change == "package":
                graph["units"][0]["pkg_id"] = "external-id"
            elif change == "target":
                graph["units"][0]["target"]["name"] = "unknown-target"
            elif change == "index":
                graph["roots"].append(True)
            else:
                graph["version"] = 2
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, message):
                shards.resolved_features(data, ["runtime", "tool"], graph, "native-target")

    def test_doctests_remain_selected_without_a_separate_compilation_root(self):
        data = metadata()
        resolved = shards.resolved_features(data, ["runtime", "tool"], unit_graph(data), "native-target")
        plan = shards.partition(data, ["runtime", "tool"], 4, resolved)
        self.assertIn(("runtime", "doc", "runtime"), plan["shards"][0])
        self.assertIn(("tool", "doc", "tool"), plan["shards"][0])

    def test_gated_package_without_roots_and_partial_feature_map_fail_closed(self):
        data = metadata()
        graph = unit_graph(data)
        graph["roots"] = [index for index in graph["roots"] if graph["units"][index]["pkg_id"] != "runtime-id"]
        with self.assertRaisesRegex(ValueError, "No resolved Cargo root features for gated package runtime"):
            shards.resolved_features(data, ["runtime", "tool"], graph, "native-target")
        with self.assertRaisesRegex(ValueError, "exactly the selected packages"):
            shards.partition(data, ["runtime", "tool"], 4, {"runtime": {"cli"}})

    def test_feature_independent_empty_package_may_have_no_cargo_roots(self):
        data = metadata()
        data["packages"][1]["targets"] = [target("empty", "lib", test=False, doctest=False)]
        graph = unit_graph(data)
        resolved = shards.resolved_features(data, ["runtime", "tool"], graph, "native-target")
        self.assertEqual(resolved["tool"], set())

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
