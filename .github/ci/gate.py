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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", required=True)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / ".github/ci/impact.json").read_text(encoding="utf-8"))
    needs = json.loads(os.environ["CI_NEEDS"])
    errors = failures(needs, config["workflows"][arguments.workflow])
    if errors:
        raise SystemExit("\n".join(errors))
    print("All selected CI groups passed; unselected groups were skipped.")


if __name__ == "__main__":
    main()
