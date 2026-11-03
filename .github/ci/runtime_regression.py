"""Run a bounded Rust regression corpus; full reference qualification is separate."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


# Existing source assertions and fixed inputs, not shortened source schedules.
# These all passed on Windows ARM64 in full run 41535008203. Timed/exhaustive
# workloads, external model corpora and C++ reference replay stay in full CI.
CASES = (
    "tests/test-pq2-row-shapes.cpp",
    "tests/test-ptq1_0-element-map.cpp",
    "tests/test-col2im-1d.cpp",
    "tests/test-dfly-fusion.cpp",
    "tests/test-rope.cpp",
    "tests/test-moe-weighted-reduce.cpp",
    "tests/test-unicode.cpp",
    "tests/test-mtp-catchup-batch.cpp",
    "tests/test-sampling.cpp",
    "tests/test-grammar-parser.cpp",
    "tests/test-llama-grammar.cpp",
    "tests/test-grammar-integration.cpp",
    "tests/test-jinja.cpp",
    "tests/test-chat-template.cpp",
    "tests/test-reasoning-budget.cpp",
    "tests/test-gguf.cpp",
    "tests/test-alloc.cpp",
    "tests/test-batch-alloc.cpp",
    "tests/peg-parser/test-basic.cpp",
    "tests/peg-parser/test-gbnf-generation.cpp",
    "tests/peg-parser/test-json-parser.cpp",
    "tests/peg-parser/test-python-dict-parser.cpp",
    "tests/peg-parser/test-unicode.cpp",
)
REPORT_LIMIT = 1024 * 1024
WORKLOAD_SECONDS = 120


def validate(report, source_id, upstream_commit):
    """A successful process exit alone is not a passing source assertion."""
    if (
        not isinstance(report, dict)
        or report.get("SchemaVersion") != 1
        or report.get("Target") != "rust"
        or report.get("UpstreamCommit") != upstream_commit
        or report.get("Measure") != "upstream-source-files"
    ):
        raise ValueError("Regression report identity differs")
    cases = report.get("Cases")
    if (
        not isinstance(cases, list)
        or len(cases) != 1
        or not isinstance(cases[0], dict)
        or cases[0].get("Id") != source_id
        or cases[0].get("Status") != "pass"
    ):
        raise ValueError(f"Required regression did not pass: {source_id}")
    return cases[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("GITHUB_ACTIONS") != "true" or os.environ.get("GITHUB_REF") != "refs/heads/main":
        raise SystemExit("The regression corpus runs only in main hosted CI")
    revision = os.environ.get("GITHUB_SHA", "")
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise SystemExit("A complete project revision is required")
    root = Path(__file__).resolve().parents[2]
    upstream = json.loads((root / "tools/prism/source.lock.json").read_text())["Commit"]
    harness = args.harness.resolve(strict=True)
    evidence = args.evidence.resolve()
    evidence.mkdir(parents=True, exist_ok=False)
    with harness.open("rb") as binary:
        harness_hash = hashlib.file_digest(binary, "sha256").hexdigest()
    record = {
        "SchemaVersion": 1,
        "Measure": "runtime-regression",
        "FullSourceGate": False,
        "NativeReferenceReplay": False,
        "ProjectRevision": revision,
        "UpstreamCommit": upstream,
        "HarnessSha256": harness_hash,
        "RunId": os.environ.get("GITHUB_RUN_ID"),
        "ExpectedIds": list(CASES),
        "Status": "fail",
        "Cases": [],
    }
    start = time.monotonic()
    try:
        for ordinal, source_id in enumerate(CASES):
            remaining = WORKLOAD_SECONDS - (time.monotonic() - start)
            if remaining <= 0:
                raise TimeoutError("The aggregate regression workload exceeded 120 seconds")
            output = evidence / f"case-{ordinal:02}.json"
            with (evidence / f"case-{ordinal:02}.log").open("wb") as log:
                subprocess.run(
                    [str(harness), "--target", "rust", "--case", source_id, "--output", str(output)],
                    cwd=evidence,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=min(30, remaining),
                )
            if output.stat().st_size > REPORT_LIMIT:
                raise ValueError("Regression report exceeds 1 MiB")
            case = validate(json.loads(output.read_text(encoding="utf-8")), source_id, upstream)
            record["Cases"].append(case)
        if [case["Id"] for case in record["Cases"]] != list(CASES):
            raise ValueError("Regression coverage differs")
        record["Status"] = "pass"
    except Exception as error:
        record["Error"] = str(error)
        raise
    finally:
        record["ElapsedSeconds"] = time.monotonic() - start
        (evidence / "regression.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(summary, "a", encoding="utf-8") as stream:
                stream.write(
                    f"Runtime regression: {record['Status']}; {len(record['Cases'])}/{len(CASES)} selected Rust contracts.\n\n"
                    "FullSourceGate=false; no C++ reference replay. Dispatch Runtime conformance for complete release qualification.\n"
                )


if __name__ == "__main__":
    main()
