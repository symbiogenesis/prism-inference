"""Run every default Cargo correctness target in one deterministic CI shard."""

import argparse
import json
import os
from pathlib import Path
import subprocess


LIB_KINDS = {"lib", "rlib", "dylib", "cdylib", "staticlib", "proc-macro"}


def default_features(package):
    """Expand local default-feature aliases, including cyclic feature graphs."""
    features = package["features"]
    pending = ["default"] if "default" in features else []
    enabled = set()
    while pending:
        feature = pending.pop()
        if feature in enabled:
            continue
        enabled.add(feature)
        for value in features[feature]:
            if value in features:
                pending.append(value)
    return enabled


def resolved_features(metadata, packages, graph, build_target):
    """Read Cargo's exact selected test roots, without merging host features.

    Cargo metadata's no-deps mode omits feature resolution. Inspecting every
    possible feature forwarding edge instead rejects inactive optional features.
    The pinned nightly's planning-only unit graph resolves the same package set
    and target as the test invocations, including dev dependencies and aliases.
    """
    if (graph.get("version") != 1 or not isinstance(graph.get("units"), list)
            or not isinstance(graph.get("roots"), list) or not build_target):
        raise ValueError("Expected Cargo unit graph v1 and an explicit build target")
    known = {package["name"]: package for package in metadata["packages"]
             if package["id"] in metadata["workspace_members"]}
    if set(packages) - known.keys():
        raise ValueError("Selected package is absent from workspace metadata")
    selected = {known[name]["id"]: name for name in packages}
    enabled = {}
    roots = set()
    for index in graph["roots"]:
        if type(index) is not int or not 0 <= index < len(graph["units"]):
            raise ValueError("Cargo unit graph contains an invalid root index")
        unit = graph["units"][index]
        name = selected.get(unit.get("pkg_id"))
        if name is None:
            raise ValueError(f"Cargo test root is outside selected packages: {unit.get('pkg_id')}")
        target = unit.get("target", {})
        identity = (target.get("name"), tuple(target.get("kind", [])))
        if not any(identity == (item["name"], tuple(item["kind"])) for item in known[name]["targets"]):
            raise ValueError(f"Cargo test root is absent from metadata: {name}/{identity}")
        # Proc macros compile on the host even when explicitly testing a target.
        # Every ordinary root must use the requested target, never a host helper.
        if unit.get("platform") != build_target and not (
                unit.get("platform") is None and "proc-macro" in target.get("kind", [])):
            raise ValueError(f"Cargo root target mismatch: {name}/{identity}: {unit.get('platform')} != {build_target}")
        features = unit.get("features")
        if (not isinstance(features, list) or any(not isinstance(feature, str) for feature in features)
                or len(features) != len(set(features)) or set(features) - known[name]["features"].keys()):
            raise ValueError(f"Invalid resolved Cargo root features: {name}/{identity}")
        if name in enabled and enabled[name] != set(features):
            raise ValueError(f"Cargo root feature sets disagree for selected package {name}")
        enabled[name] = set(features)
        kind = "lib" if set(target["kind"]) <= LIB_KINDS else target["kind"][0]
        mode = unit.get("mode")
        if mode in {"test", "doctest"}:
            roots.add((name, "doc" if mode == "doctest" else kind, target["name"]))
        elif mode != "build":
            raise ValueError(f"Unsupported Cargo root mode: {name}/{identity}: {mode}")
    for name in packages:
        if name not in enabled:
            if any(target.get("required-features") for target in known[name]["targets"]):
                raise ValueError(f"No resolved Cargo root features for gated package {name}")
            # With no gated targets, features cannot affect this inventory. Cargo
            # may omit an empty test=false package from its default test roots.
            enabled[name] = default_features(known[name])
    expected, _ = inventory(metadata, packages, enabled)
    # Doctests may be executed directly by Cargo without a separate compile unit.
    missing = {identity for identity in set(expected) - roots if identity[1] != "doc"}
    if missing:
        raise ValueError(f"Cargo unit graph omits enabled test targets: {sorted(missing)}")
    additional = roots - set(expected)
    if additional:
        raise ValueError(f"Cargo unit graph contains unaccounted test targets: {sorted(additional)}")
    return enabled


def inventory(metadata, packages, enabled_features=None):
    """Inventory enabled tests and doctests; reject ambiguous feature metadata."""
    if (metadata.get("version") != 1 or not isinstance(packages, list) or not packages
            or any(not isinstance(name, str) or not name for name in packages)
            or len(set(packages)) != len(packages)):
        raise ValueError("Expected Cargo metadata v1 and unique selected packages")
    members = set(metadata["workspace_members"])
    known = {package["name"]: package for package in metadata["packages"] if package["id"] in members}
    if set(packages) - known.keys():
        raise ValueError("Selected package is absent from workspace metadata")
    if enabled_features is not None and set(enabled_features) != set(packages):
        raise ValueError("Resolved Cargo features must cover exactly the selected packages")
    tests, skipped = [], []
    for name in sorted(packages):
        package = known[name]
        enabled = default_features(package) if enabled_features is None else set(enabled_features[name])
        seen = set()
        for target in package["targets"]:
            kinds = set(target["kind"])
            if kinds <= LIB_KINDS and kinds:
                kind = "lib"
            elif len(kinds) == 1 and next(iter(kinds)) in {"bin", "test", "example", "bench", "custom-build"}:
                kind = next(iter(kinds))
            else:
                raise ValueError(f"Unsupported Cargo target kinds: {target['kind']}")
            if type(target["test"]) is not bool or type(target["doctest"]) is not bool:
                raise ValueError("Cargo target coverage flags must be boolean")
            identity = (name, kind, target["name"])
            if identity in seen:
                raise ValueError("Duplicate Cargo target identity")
            seen.add(identity)
            required = set(target.get("required-features", []))
            if required - package["features"].keys():
                raise ValueError("Required Cargo feature is missing from the package")
            if required - enabled:
                skipped.append([*identity, "disabled required features"])
                continue
            if kind in {"bench", "custom-build"}:
                continue
            if target["test"]:
                tests.append(identity)
            if kind == "lib" and target["doctest"]:
                tests.append((name, "doc", target["name"]))
    return sorted(tests), skipped


def partition(metadata, packages, count, enabled_features=None):
    """Assign each Cargo selector once, balancing selector groups across shards."""
    if type(count) is not int or count < 1:
        raise ValueError("Expected a positive shard count")
    tests, skipped = inventory(metadata, packages, enabled_features)
    groups = {}
    for test in tests:
        # --lib and --doc select every eligible library in the package set.
        # Other selectors identify one kind/name, including same-named targets
        # from several packages. Keeping each selector whole prevents overlap.
        key = (test[1], "" if test[1] in {"lib", "doc"} else test[2])
        groups.setdefault(key, []).append(test)
    owners = {}
    loads = [0] * count
    for key, owner in (("lib", 0), ("doc", 1 % count)):
        if (key, "") in groups:
            owners[(key, "")] = owner
            loads[owner] += 1
    for key in sorted(groups.keys() - owners.keys()):
        owner = min(range(count), key=lambda index: (loads[index], index))
        owners[key] = owner
        loads[owner] += 1
    shards = [[] for _ in range(count)]
    for key, group in groups.items():
        shards[owners[key]].extend(group)
    for shard in shards:
        shard.sort()
    flattened = [test for shard in shards for test in shard]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(tests):
        raise ValueError("Cargo test inventory was lost or duplicated during sharding")
    additional = []
    if any(kind == "lib" for _, kind, _ in tests):
        for package in metadata["packages"]:
            if package["name"] in packages:
                for target in package["targets"]:
                    if set(target["kind"]) <= LIB_KINDS and not target["test"]:
                        additional.append((package["name"], "lib", target["name"]))
    return {"shards": shards, "disabled": skipped, "additional_unit_roots": sorted(additional)}


def commands(plan, packages, index):
    """Batch target selectors with the complete selected Cargo package set."""
    if not 0 <= index < len(plan["shards"]):
        raise ValueError("Correctness shard index is out of range")
    tests = plan["shards"][index]
    result = []
    # Every invocation retains Cargo's usual package feature unification.
    # Same-named targets share a shard and selector. --lib also selects the
    # workspace's three empty test=false library roots; this small superset
    # avoids changing features by invoking their consumers package-by-package.
    selection = [argument for package in packages for argument in ("--package", package)]
    ordinary = sorted({(kind, name) for _, kind, name in tests if kind not in {"doc", "lib"}})
    arguments = ["--lib"] if any(kind == "lib" for _, kind, _ in tests) else []
    for kind, name in ordinary:
        arguments.extend(["--" + kind, name])
    if arguments:
        # One invocation lets Cargo compile and link every selected target as
        # resources permit, rather than serializing integrations before units
        # and examples. --doc is incompatible with other target selectors.
        result.append(selection + arguments)
    if any(kind == "doc" for _, kind, _ in tests):
        result.append(selection + ["--doc"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    arguments = parser.parse_args()
    if os.environ.get("GITHUB_ACTIONS") != "true" or os.environ.get("GITHUB_REF") != "refs/heads/main":
        raise SystemExit("Complete correctness shards run only in GitHub Actions on main")
    packages = json.loads(os.environ["PRISM_PACKAGES"])
    toolchain = "+nightly-2026-11-09"
    manifest = str(arguments.manifest.resolve(strict=True))
    metadata = json.loads(subprocess.check_output([
        "cargo", toolchain, "metadata", "--manifest-path", manifest,
        "--format-version", "1", "--no-deps", "--locked", "--offline",
    ], text=True))
    workspace = Path(metadata["workspace_root"]).resolve()
    if not arguments.evidence.is_absolute() or arguments.evidence.resolve().is_relative_to(workspace):
        raise SystemExit("Correctness evidence requires an absolute directory outside the workspace")
    build_target = os.environ["CARGO_BUILD_TARGET"]
    package_selection = [argument for package in packages for argument in ("--package", package)]
    graph_command = ["cargo", toolchain, "test", "--manifest-path", manifest, *package_selection,
                     "--target", build_target, "--locked", "--offline", "--release",
                     "--unit-graph", "-Z", "unstable-options"]
    print("Resolve test features without compiling:", " ".join(graph_command), flush=True)
    graph = json.loads(subprocess.check_output(graph_command, text=True))
    enabled = resolved_features(metadata, packages, graph, build_target)
    plan = partition(metadata, packages, arguments.count, enabled)
    selected = commands(plan, packages, arguments.index)
    arguments.evidence.mkdir(parents=True, exist_ok=True)
    receipt = {"index": arguments.index, "count": arguments.count, "target": build_target,
               "feature_query": graph_command,
               "resolved_features": {name: sorted(features) for name, features in enabled.items()},
               "coverage": plan, "commands": selected}
    (arguments.evidence / "correctness-shard.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    for selection in selected:
        command = ["cargo", toolchain, "test", "--manifest-path", manifest, *selection,
                   "--target", build_target, "--locked", "--offline", "--release", "--no-fail-fast"]
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)
    # A package with no tests, or fewer integration targets than shards, has a
    # deliberately empty assignment. Its complete inventory is still retained.
    print(f"Correctness shard {arguments.index}/{arguments.count}: {len(plan['shards'][arguments.index])} targets passed")


if __name__ == "__main__":
    main()
