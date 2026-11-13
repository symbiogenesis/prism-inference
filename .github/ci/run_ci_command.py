"""Stream a CI command, require named controls and retain failure context only."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def validate_output(output, rust_test=None, unittest_count=None):
    """Keep exact-control admission separate from a command's native exit code."""
    output = output.replace("\r\n", "\n")
    if rust_test is not None:
        patterns = [r"^running 1 test$", r"^test " + re.escape(rust_test) + r" \.\.\. ok$",
                    r"^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; [0-9]+ filtered out; finished in .+$"]
        if any(len(re.findall(pattern, output, re.MULTILINE)) != 1 for pattern in patterns):
            return f"The exact Rust control must pass exactly once: {rust_test}"
    if unittest_count is not None:
        count = re.findall(r"^Ran ([0-9]+) tests? in .+$", output, re.MULTILINE)
        if count != [str(unittest_count)] or len(re.findall(r"^OK$", output, re.MULTILINE)) != 1:
            return f"Expected exactly {unittest_count} passing unittest controls"
    return None


def stream_command(command, output):
    """Merge stderr into stdout so diagnostics precede an exit-status rejection."""
    lines = []
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
        for line in process.stdout:
            output.write(line)
            output.flush()
            lines.append(line)
        return process.wait(), "".join(lines)


def context(label, command, environment):
    return {"label": label, "command": command,
            "source_revision": environment.get("GITHUB_SHA"),
            "run_id": environment.get("GITHUB_RUN_ID"),
            "run_attempt": environment.get("GITHUB_RUN_ATTEMPT"),
            "job": environment.get("GITHUB_JOB"),
            "runner_os": environment.get("RUNNER_OS"),
            "runner_arch": environment.get("RUNNER_ARCH"),
            "image_os": environment.get("ImageOS"),
            "image_version": environment.get("ImageVersion"),
            "lane": environment.get("PRISM_LANE"),
            "selected_packages": environment.get("PRISM_PACKAGES"),
            "target": environment.get("PRISM_CI_TARGET") or environment.get("CARGO_BUILD_TARGET") or environment.get("EXPECTED_TARGET"),
            "toolchain": next((arg[1:] for arg in command[1:] if arg.startswith("+")), environment.get("PRISM_CI_TOOLCHAIN")),
            "scope": environment.get("PRISM_CI_SCOPE", "This command only; no inference or performance qualification"),
            "started_at": datetime.now(timezone.utc).isoformat()}


def retain_result(result, output, directory):
    """Keep one concise command ledger and nonempty failed-command output."""
    directory = Path(directory)
    workspace = Path(__file__).resolve().parents[2]
    if not directory.is_absolute() or directory.resolve().is_relative_to(workspace):
        raise ValueError("CI command evidence requires an absolute directory outside the workspace")
    directory.mkdir(parents=True, exist_ok=True)
    ledger = directory / "command-results.json"
    data = json.loads(ledger.read_text(encoding="utf-8")) if ledger.exists() else {"schema_version": 1, "commands": []}
    if data.get("schema_version") != 1 or not isinstance(data.get("commands"), list):
        raise ValueError("Unsupported CI command ledger")
    if result["failure"] and output.strip():
        slug = re.sub(r"[^a-zA-Z0-9_-]", "-", result["label"])[:60]
        identity = hashlib.sha256(json.dumps(result["command"]).encode()).hexdigest()[:12]
        log = directory / f"{len(data['commands']) + 1:02d}-{slug}-{identity}.failure.log"
        log.write_text(output, encoding="utf-8")
        result["failure_log"] = log.name
    data["commands"].append(result)
    temporary = ledger.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(ledger)


def failure_summary(result, filename):
    """Summarize a failure without making a gate or inferring its root cause."""
    if not filename:
        return
    with open(filename, "a", encoding="utf-8") as summary:
        summary.write("\n### Failed CI command\n\n")
        for name in ("label", "command", "source_revision", "run_id", "run_attempt", "job", "runner_os", "runner_arch", "image_os", "image_version", "lane", "selected_packages", "target", "toolchain", "exit_code", "failure", "scope"):
            value = (subprocess.list2cmdline(result["command"]) if name == "command" else
                     str(result.get(name) if result.get(name) is not None else "unknown"))
            summary.write(f"- {name.replace('_', ' ')}: {value.replace(chr(10), ' ')}\n")
        summary.write("\nCommand diagnostics were streamed above. The failure remains required; this summary grants no acceptance.\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument("--expect-rust-test")
    controls.add_argument("--expect-unittest-count", type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args(argv)
    command = arguments.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command or arguments.expect_unittest_count is not None and arguments.expect_unittest_count < 1:
        parser.error("A command and a positive expected control count are required")
    result = context(arguments.label, command, os.environ)
    print(f"Running {arguments.label}: {subprocess.list2cmdline(command)}", flush=True)
    try:
        exit_code, output = stream_command(command, sys.stdout)
        failure = (f"Command exited with status {exit_code}" if exit_code else
                   validate_output(output, arguments.expect_rust_test, arguments.expect_unittest_count))
    except OSError as error:
        exit_code, output, failure = None, "", f"Could not start command: {error}"
        print(failure, file=sys.stderr, flush=True)
    result.update({"exit_code": exit_code, "failure": failure, "output_bytes": len(output.encode("utf-8")),
                   "finished_at": datetime.now(timezone.utc).isoformat()})
    evidence = os.environ.get("PRISM_CHECK_EVIDENCE")
    if evidence:
        retain_result(result, output, evidence)
    if failure:
        failure_summary(result, os.environ.get("GITHUB_STEP_SUMMARY"))
        print(f"{arguments.label}: {failure}", file=sys.stderr, flush=True)
        return exit_code if exit_code else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
