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


def inventory(metadata, packages):
    """Inventory enabled default tests and doctests; reject ambiguous metadata."""
    if (metadata.get("version") != 1 or not isinstance(packages, list) or not packages
            or any(not isinstance(name, str) or not name for name in packages)
            or len(set(packages)) != len(packages)):
        raise ValueError("Expected Cargo metadata v1 and unique selected packages")
    members = set(metadata["workspace_members"])
    known = {package["name"]: package for package in metadata["packages"] if package["id"] in members}
    if set(packages) - known.keys():
        raise ValueError("Selected package is absent from workspace metadata")
    # --no-deps metadata does not contain Cargo's resolved feature graph. Refuse
    # a possibly dependency-activated gated test instead of silently dropping it.
    # This is intentionally conservative, including optional/target dependencies.
    requested = {name: set() for name in known}
    for package in known.values():
        dependencies = package.get("dependencies", [])
        aliases = {dependency.get("rename") or dependency["name"]: dependency["name"] for dependency in dependencies}
        for dependency in dependencies:
            if dependency["name"] in requested:
                requested[dependency["name"]].update(dependency.get("features", []))
        for values in package["features"].values():
            for value in values:
                if "/" in value:
                    alias, feature = value.split("/", 1)
                    owner = aliases.get(alias.removesuffix("?"))
                    if owner in requested:
                        requested[owner].add(feature)
    for name, package in known.items():
        pending = list(requested[name])
        while pending:
            for value in package["features"].get(pending.pop(), []):
                if value in package["features"] and value not in requested[name]:
                    requested[name].add(value)
                    pending.append(value)
    tests, skipped = [], []
    for name in sorted(packages):
        package = known[name]
        enabled = default_features(package)
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
                if (required - enabled) & requested[name]:
                    raise ValueError("Dependency-activated required test feature needs resolved-feature sharding support")
                skipped.append([*identity, "disabled required features"])
                continue
            if kind in {"bench", "custom-build"}:
                continue
            if target["test"]:
                tests.append(identity)
            if kind == "lib" and target["doctest"]:
                tests.append((name, "doc", target["name"]))
    return sorted(tests), skipped


def partition(metadata, packages, count):
    """Assign every integration-target name once; unit and doc tests use shard 0."""
    if type(count) is not int or count < 1:
        raise ValueError("Expected a positive shard count")
    tests, skipped = inventory(metadata, packages)
    names = sorted({name for _, kind, name in tests if kind == "test"})
    owners = {name: index % count for index, name in enumerate(names)}
    shards = [[] for _ in range(count)]
    for test in tests:
        shards[owners[test[2]] if test[1] == "test" else 0].append(test)
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
    return {"shards": shards, "disabled": skipped, "additional_unit_roots": additional}


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
    integration = sorted({name for _, kind, name in tests if kind == "test"})
    if integration:
        result.append(selection + [argument for name in integration for argument in ("--test", name)])
    ordinary = sorted({(kind, name) for _, kind, name in tests if kind not in {"test", "doc", "lib"}})
    arguments = ["--lib"] if any(kind == "lib" for _, kind, _ in tests) else []
    for kind, name in ordinary:
        arguments.extend(["--" + kind, name])
    if arguments:
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
    toolchain = "+nightly-2026-10-22"
    manifest = str(arguments.manifest.resolve(strict=True))
    metadata = json.loads(subprocess.check_output([
        "cargo", toolchain, "metadata", "--manifest-path", manifest,
        "--format-version", "1", "--no-deps", "--locked", "--offline",
    ], text=True))
    workspace = Path(metadata["workspace_root"]).resolve()
    if not arguments.evidence.is_absolute() or arguments.evidence.resolve().is_relative_to(workspace):
        raise SystemExit("Correctness evidence requires an absolute directory outside the workspace")
    plan = partition(metadata, packages, arguments.count)
    selected = commands(plan, packages, arguments.index)
    arguments.evidence.mkdir(parents=True, exist_ok=True)
    receipt = {"index": arguments.index, "count": arguments.count, "coverage": plan, "commands": selected}
    (arguments.evidence / "correctness-shard.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    for selection in selected:
        command = ["cargo", toolchain, "test", "--manifest-path", manifest, *selection,
                   "--locked", "--offline", "--release", "--no-fail-fast"]
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)
    # A package with no tests, or fewer integration targets than shards, has a
    # deliberately empty assignment. Its complete inventory is still retained.
    print(f"Correctness shard {arguments.index}/{arguments.count}: {len(plan['shards'][arguments.index])} targets passed")


if __name__ == "__main__":
    main()
