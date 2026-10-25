"""Select changed workspace crates, their consumers and relevant CI groups."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tomllib
import urllib.error
import urllib.parse
import urllib.request


def matches(path, pattern):
    """Match repository-relative globs; a single star never crosses a slash."""
    parts = []
    index = 0
    while index < len(pattern):
        if pattern[index:index + 3] == "**/":
            parts.append("(?:.*/)?")
            index += 3
        elif pattern[index:index + 2] == "**":
            parts.append(".*")
            index += 2
        elif pattern[index] == "*":
            parts.append("[^/]*")
            index += 1
        elif pattern[index] == "?":
            parts.append("[^/]")
            index += 1
        else:
            parts.append(re.escape(pattern[index]))
            index += 1
    return re.fullmatch("".join(parts), path) is not None


def any_match(path, patterns):
    return any(matches(path, pattern) for pattern in patterns)


def workspace(root):
    """Read Cargo manifests without installing Rust or resolving the registry."""
    manifest = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
    paths = [root / "Cargo.toml"] if "package" in manifest else []
    settings = manifest.get("workspace", {})
    for pattern in settings.get("members", []):
        for directory in sorted(root.glob(pattern)):
            relative = directory.relative_to(root).as_posix()
            if not any_match(relative, settings.get("exclude", [])):
                paths.append(directory / "Cargo.toml")
    def local_dependencies(data, base):
        for table in [data, *data.get("target", {}).values()]:
            for kind in ("dependencies", "build-dependencies", "dev-dependencies"):
                for alias, dependency in table.get(kind, {}).items():
                    if not isinstance(dependency, dict):
                        continue
                    directory = base
                    if dependency.get("workspace"):
                        dependency = settings.get("dependencies", {})[alias]
                        directory = root
                    if isinstance(dependency, dict) and "path" in dependency:
                        yield (directory / dependency["path"]).resolve()

    packages = {}
    paths = list(dict.fromkeys(paths))
    for path in paths:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        name = data["package"]["name"]
        if name in packages:
            raise ValueError(f"Duplicate workspace package: {name}")
        packages[name] = {"path": path.parent.resolve(), "manifest": data}
        # Cargo automatically includes in-workspace path dependencies, even when
        # they were not explicitly listed in workspace.members.
        for directory in local_dependencies(data, path.parent):
            if directory.is_relative_to(root) and not any_match(directory.relative_to(root).as_posix(), settings.get("exclude", [])):
                candidate = directory / "Cargo.toml"
                if candidate not in paths:
                    paths.append(candidate)
    by_directory = {package["path"]: name for name, package in packages.items()}
    reverse = {name: set() for name in packages}
    for name, package in packages.items():
        data = package["manifest"]
        for directory in local_dependencies(data, package["path"]):
            if directory in by_directory:
                reverse[by_directory[directory]].add(name)
    return packages, reverse


def select(root, paths, full=False, config=None, reason="changed inputs"):
    root = Path(root).resolve()
    if config is None:
        config = json.loads((root / ".github/ci/impact.json").read_text(encoding="utf-8"))
    packages, reverse = workspace(root)
    groups = config["groups"]
    for group in groups.values():
        if set(group.get("packages", [])) - packages.keys() - {"*"}:
            raise ValueError("CI group references an unknown workspace package")
        if set(group.get("groups", [])) - groups.keys():
            raise ValueError("CI group references an unknown group")
    if (set(config.get("package_inputs", {})) | set(config.get("package_paths", {}))) - packages.keys():
        raise ValueError("CI input map references an unknown workspace package")
    paths = sorted(set(paths))
    affected = set()
    selected = {name: False for name in groups}
    manifests = {(package["path"] / "Cargo.toml").relative_to(root).as_posix() for package in packages.values()}
    owners = sorted(packages, key=lambda name: len(packages[name]["path"].parts), reverse=True)
    for path in paths:
        if any_match(path, config.get("shared", [])):
            full, reason = True, "shared build or CI input"
        if (path == "Cargo.toml" or path.endswith("/Cargo.toml")) and path not in manifests:
            full, reason = True, "added or removed workspace manifest"
        known = False
        for name in owners:
            directory = packages[name]["path"].relative_to(root).as_posix()
            patterns = config.get("package_paths", {}).get(name)
            if patterns is not None:
                owned = any_match(path, patterns)
            elif directory == ".":
                owned = any_match(path, ["src/**", "tests/**/*.rs", "examples/**", "benches/**", "build.rs"])
            else:
                owned = path.startswith(directory + "/")
            if owned:
                affected.add(name)
                known = True
                break
        for name, patterns in config.get("package_inputs", {}).items():
            if any_match(path, patterns):
                affected.add(name)
                known = True
        for name, group in groups.items():
            if any_match(path, group.get("paths", [])):
                selected[name] = True
                known = True
            if any_match(path, group.get("observes", [])):
                # Linting a new helper does not establish behavioral coverage.
                selected[name] = True
        if not known and not any_match(path, config.get("ignored", [])):
            full, reason = True, "unclassified input (conservative full validation)"
    pending = list(affected)
    while pending:
        for consumer in reverse[pending.pop()] - affected:
            affected.add(consumer)
            pending.append(consumer)
    for name, group in groups.items():
        watched = set(group.get("packages", []))
        relevant = affected - set(group.get("exclude_packages", []))
        selected[name] |= bool(relevant & watched) or ("*" in watched and bool(relevant))
    for _ in groups:
        for name, group in groups.items():
            selected[name] |= any(selected[dependency] for dependency in group.get("groups", []))
    if full:
        affected = set(packages)
        selected = {name: True for name in groups}
    return {"packages": sorted(affected), "groups": selected, "full": full, "paths": paths, "reason": reason}


def git(root, *arguments):
    return subprocess.run(["git", "-C", str(root), *arguments], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def successful_base(root, workflow, after):
    """Carry forward changes from failed/cancelled pushes in this workflow."""
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    token = os.environ.get("GITHUB_TOKEN", "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) or not token:
        return None
    workflow = urllib.parse.quote(workflow, safe="")
    url = f"https://api.github.com/repos/{repository}/actions/workflows/{workflow}/runs?branch=main&status=success&event=push&per_page=100"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json", "User-Agent": "workspace-ci-impact"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            runs = json.load(response)["workflow_runs"]
        for run in runs:
            candidate = run["head_sha"]
            if candidate == after or not re.fullmatch(r"[0-9a-f]{40}", candidate):
                continue
            try:
                git(root, "merge-base", "--is-ancestor", candidate, after)
                return candidate
            except subprocess.CalledProcessError:
                continue
    except (OSError, ValueError, KeyError):
        return None
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--before", default="")
    parser.add_argument("--after", default="HEAD")
    parser.add_argument("--workflow")
    parser.add_argument("--full", action="store_true")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config = json.loads(arguments.config.read_text(encoding="utf-8")) if arguments.config else None
    full = arguments.full or os.environ.get("GITHUB_EVENT_NAME") != "push"
    reason = "manual full validation" if full else "changes since successful validation"
    before = successful_base(root, arguments.workflow, arguments.after) if arguments.workflow and not full else arguments.before
    paths = []
    if not full:
        if not before or re.fullmatch("0+", before):
            full, reason = True, "no successful ancestor baseline"
        else:
            try:
                # Disable rename collapsing so both the old and new crate own a move.
                raw = git(root, "diff", "--no-renames", "--name-only", "-z", before, arguments.after, "--")
                paths = [path.decode("utf-8", errors="surrogateescape") for path in raw.split(b"\0") if path]
            except (OSError, subprocess.CalledProcessError):
                full, reason = True, "baseline diff unavailable"
    plan = select(root, paths, full=full, config=config, reason=reason)
    serialized = json.dumps(plan, separators=(",", ":"), ensure_ascii=True)
    print(serialized)
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a", encoding="utf-8") as stream:
            stream.write(f"plan={serialized}\npackages={json.dumps(plan['packages'])}\n")
            for name, selected in plan["groups"].items():
                stream.write(f"{name}={str(selected).lower()}\n")
            for name, choices in (config or json.loads((root / ".github/ci/impact.json").read_text(encoding="utf-8"))).get("matrices", {}).items():
                values = [choice for choice, group in choices.items() if plan["groups"][group]]
                stream.write(f"{name}={json.dumps(values)}\n")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write("Affected crates: " + (", ".join(plan["packages"]) or "none") + "\n\n")
            stream.write("Selected groups: " + (", ".join(name for name, value in plan["groups"].items() if value) or "none") + "\n\n")
            stream.write("Selection: " + plan["reason"] + "\n")


if __name__ == "__main__":
    main()
