"""Route routine source checks while preserving their original event boundaries."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

import gate
import impact


WORKFLOW = "source-checks.yml"
GROUPS = ("markdown", "standards", "probe", "parity")
JOBS = {"markdown": "markdown", "standards": "standards", "probe": "probe",
        "compatibility": "probe", "model-free": "parity"}
SHARED = (".github/workflows/source-checks.yml", ".github/ci/**")
INPUTS = {
    "markdown": ("**/*.md", ".rumdl.toml"),
    "standards": ("**/*.ps1", "**/*.psm1", "**/*.psd1", "**/*.cs",
                  "tools/validation/**", ".editorconfig", "global.json"),
    "probe": ("tools/rust-trial/fixtures/**", "config/rust-nightly-trial.json",
              "rust-toolchain.toml", "tests/Test-RustNightly.ps1",
              "scripts/Bonsai.Common.psm1", "scripts/Bonsai.Native.psm1",
              ".github/actions/rust-toolchain/**"),
    "parity": ("Cargo.toml", "Cargo.lock", "src/**", "server/**", "tools/gguf/**",
               "tools/native-hash/**", "tests/swift*", "rust-toolchain.toml",
               ".github/actions/rust-toolchain/**"),
}
SHA = re.compile(r"[0-9a-f]{40}\Z")


def eligible(event, git_ref):
    if event not in ("push", "pull_request", "workflow_dispatch"):
        raise ValueError("Unsupported source-check event")
    main = git_ref == "refs/heads/main" and event != "pull_request"
    return {"markdown": True, "standards": main, "probe": main,
            "parity": event in ("pull_request", "workflow_dispatch")}


def select(paths, event, git_ref, full=False, reason="changed inputs"):
    paths = sorted(set(paths))
    allowed = eligible(event, git_ref)
    if any(impact.any_match(path, SHARED) for path in paths):
        full, reason = True, "shared source-check workflow or helper input"
    groups = {name: allowed[name] and (full or any(
        impact.any_match(path, INPUTS[name]) for path in paths)) for name in GROUPS}
    return {"groups": groups, "paths": paths, "full": full, "reason": reason}


def plan(root, event, git_ref, after, base="", head=""):
    if event == "workflow_dispatch":
        return select([], event, git_ref, full=True, reason="manual eligible checks")
    eligible(event, git_ref)
    paths = []
    full = False
    try:
        if not SHA.fullmatch(after):
            raise ValueError("A complete checkout revision is required")
        if event == "push":
            before = impact.successful_base(root, WORKFLOW, after)
            if not before or not SHA.fullmatch(before):
                raise ValueError("no successful source-check ancestor baseline")
            comparison = (before, after)
            reason = "changes since successful source checks"
        else:
            if not SHA.fullmatch(base) or not SHA.fullmatch(head):
                raise ValueError("pull-request base/head identity unavailable")
            # Compare only this PR's changes since its common ancestor. A moving
            # base branch must not hide changes or add unrelated base commits.
            comparison = (base + "..." + head,)
            reason = "pull-request merge-base changes"
        raw = impact.git(root, "diff", "--no-renames", "--name-only", "-z", *comparison, "--")
        paths = [path.decode("utf-8", errors="surrogateescape") for path in raw.split(b"\0") if path]
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        full, reason = True, f"baseline unavailable; all eligible checks: {error}"
    return select(paths, event, git_ref, full=full, reason=reason)


def failures(needs):
    # The existing aggregate validator rejects failed/cancelled selected jobs,
    # selected skips, missing dependencies and malformed boolean selections.
    errors = gate.failures(needs, JOBS)
    if not errors:
        try:
            groups = json.loads(needs["changes"]["outputs"]["plan"])["groups"]
            if set(groups) != set(GROUPS):
                raise ValueError("Source-check groups differ")
        except (KeyError, ValueError, TypeError):
            errors.append("Missing or malformed source-check selection")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    if args.gate:
        errors = failures(json.loads(os.environ["CI_NEEDS"]))
        if errors:
            raise SystemExit("\n".join(errors))
        print("All selected source checks passed; unselected checks were skipped.")
        return
    root = Path(__file__).resolve().parents[2]
    event = os.environ["GITHUB_EVENT_NAME"]
    payload = {}
    if event == "pull_request":
        try:
            payload = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
            request = payload["pull_request"]
            base, head = request["base"]["sha"], request["head"]["sha"]
            if not isinstance(base, str) or not isinstance(head, str):
                raise ValueError("Invalid pull-request identities")
        except (OSError, KeyError, ValueError, TypeError):
            base = head = ""
    else:
        base = head = ""
    result = plan(root, event, os.environ.get("GITHUB_REF", ""),
                  os.environ.get("GITHUB_SHA", ""), base=base, head=head)
    serialized = json.dumps(result, separators=(",", ":"), ensure_ascii=True)
    print(serialized)
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a", encoding="utf-8") as stream:
            stream.write(f"plan={serialized}\n")
            for name, selected in result["groups"].items():
                stream.write(f"{name}={str(selected).lower()}\n")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write("Selected source checks: " +
                         (", ".join(name for name, selected in result["groups"].items() if selected) or "none") +
                         "\n\nSelection: " + result["reason"] + "\n")


if __name__ == "__main__":
    main()
