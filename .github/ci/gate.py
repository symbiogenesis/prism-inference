"""Require successful selected jobs; permit only deliberately unselected skips."""

import argparse
import json
import os
from pathlib import Path


def failures(needs, jobs):
    if needs.get("changes", {}).get("result") != "success":
        return ["CI impact selection did not succeed"]
    try:
        if set(needs) != set(jobs) | {"changes"}:
            raise ValueError("CI gate dependencies and routing map disagree")
        selected = json.loads(needs["changes"]["outputs"]["plan"])["groups"]
        errors = []
        for job, group in jobs.items():
            result = needs[job]["result"]
            expected = selected[group]
            if not isinstance(expected, bool):
                raise ValueError("Expected a boolean CI selection")
            if result != "success" and (expected or result != "skipped"):
                errors.append(f"{job}: selected={expected}, result={result}")
        return errors
    except (KeyError, ValueError, TypeError):
        return ["Missing or malformed CI selection/job result"]



def report(needs, errors, workflow):
    """Show dependency outcomes without presenting the aggregate as a root cause."""
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary:
        return
    # JSON string escaping keeps multiline or malformed values in one table row.
    def cell(value):
        return json.dumps(value, ensure_ascii=True).replace("|", "&#124;")
    with open(summary, "a", encoding="utf-8") as stream:
        stream.write(f"CI aggregate: {cell(workflow)}\n\n")
        for key in ("GITHUB_SHA", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT"):
            stream.write(f"{key}: {cell(os.environ.get(key, 'unknown'))}\n\n")
        stream.write("| Dependency | Result |\n| --- | --- |\n")
        if isinstance(needs, dict):
            for name, value in sorted(needs.items()):
                result = value.get("result", "missing") if isinstance(value, dict) else "malformed"
                stream.write(f"| {cell(name)} | {cell(result)} |\n")
        stream.write("\n" + ("Rejected" if errors else "Passed") + ": " + cell(errors) + "\n\n")
        stream.write("This aggregate reports dependency results. Open a failed dependency's first failing step for its original command and diagnostic. A selected skip, cancellation or missing result still rejects.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", required=True)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / ".github/ci/impact.json").read_text(encoding="utf-8"))
    needs = json.loads(os.environ["CI_NEEDS"])
    errors = failures(needs, config["workflows"][arguments.workflow])
    report(needs, errors, arguments.workflow)
    if errors:
        raise SystemExit("\n".join(errors))
    print("All selected CI groups passed; unselected groups were skipped.")


if __name__ == "__main__":
    main()
