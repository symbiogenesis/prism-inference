"""Prepare checked CPU-only source arms and admit one fixed ABBA capture offline.

Preparation builds software only on hosted CI. This module never starts a model,
dispatches CI, changes a runtime pin, or promotes a runtime/profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib

BASELINE = "055c7430cb4771dfddb24ddaa30494af6ad182da"
CHANNEL = "nightly-2026-11-09"
TARGET = "aarch64-pc-windows-msvc"
FEATURES = ["cli", "experimental-neon", "parallel-rows"]
CHECK_PHASES = ["source_analysis", "portable_library", "state_fixture", "collector_analysis",
                "collector_controls", "collector_link"]
SCHEDULE = ["baseline", "candidate", "candidate", "baseline"]
MESSAGES = [{"role": "user", "content": "Count from 1 to 10. Answer with only the numbers."}]
PROFILE = {
    "adapter_scale_bits": "3fa66666", "row_workers": 4,
    "backend_requested": "ExperimentalNeon", "execution_device": "cpu",
    "execution_basis": "direct-session-without-projection-executor",
    "backend_statistics": None, "kv_cache": "f16", "prompt_reuse": "disabled",
    "seed": 42, "thinking_enabled": False, "maximum_prompt_tokens": 128,
}
MODEL_SHA256 = "5912bb739217cf25b4283a098c7b893d9baff36bdf2e72d3dbf4cb999d5633d5"
ADAPTER_SHA256 = "f1669534803d340a496015f5c45125f3437b4d13ec764f40e34488ce83967f42"
REGISTRY_REVISION = "99bab71270b1e6a6fb6d47eec2e34b8097cdcd4c"
REGISTRY_PACKAGES = {
    "prism-inference": ("0.1.25", "0a117db7589a821b27c224e41aaababa5ddd2ac9852344c0a8121a0849046363"),
    "bonsai-neon-dot": ("0.1.2", "f932c3309ef3645b055b91bdd9ee6986b7ef98c24bc1831701138e3ab9d3a31b"),
    "bonsai-native-hash": ("0.1.3", "315284b1f3fc7e98f2cb0f749e63690853f1c441d2b3a3eec2ad68abe45180a2"),
}
MAX_JSON_BYTES = 4 * 1024 * 1024


class AdmissionError(ValueError):
    """A missing, changed or failed proof leaves qualification pending."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AdmissionError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def strict_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict:
    require(path.is_file() and path.stat().st_size <= MAX_JSON_BYTES,
            "Missing or oversized JSON proof.")
    value = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=strict_object,
                       parse_constant=lambda value: (_ for _ in ()).throw(
                           AdmissionError(f"Nonfinite JSON constant: {value}")))
    require(isinstance(value, dict), "Expected a JSON object.")
    return value


def write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def integer(value: object, minimum: int, maximum: int, name: str) -> int:
    require(type(value) is int and minimum <= value <= maximum, f"Invalid {name}.")
    return value


def positive(value: object, name: str) -> float:
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            f"Missing positive finite {name}.")
    return float(value)


def hex_value(value: object, size: int, name: str) -> str:
    require(isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{size}}}", value) is not None,
            f"Invalid {name}.")
    return value


def member(root: Path, relative: object) -> Path:
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(),
            "Evidence member must be relative.")
    # Backslashes are separators on Windows; reject them on every replay platform.
    require("\\" not in relative and ":" not in relative, "Invalid evidence member.")
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()) and path != root.resolve(),
            "Evidence member escapes its root.")
    return path


def checked_json(root: Path, reference: dict) -> dict:
    require(isinstance(reference, dict) and set(reference) == {"path", "sha256"},
            "Expected a digest-bound JSON reference.")
    path = member(root, reference["path"])
    require(file_digest(path) == hex_value(reference["sha256"], 64, "proof digest"),
            "Evidence member digest differs.")
    return read_json(path)


def command(args: list[str], cwd: Path, env: dict | None = None) -> str:
    completed = subprocess.run(args, cwd=cwd, env=env, check=False,
                               capture_output=True, text=True, encoding="utf-8")
    if completed.returncode:
        raise AdmissionError(f"Command failed ({completed.returncode}): {args!r}\n"
                             f"{completed.stderr[-12000:]}")
    return completed.stdout.strip()


def git(root: Path, *args: str) -> str:
    return command(["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), *args], root)


def source_identity(root: Path, revision: str) -> dict:
    require(root.is_absolute() and root.is_dir(), "Source root must be absolute and present.")
    require(git(root, "rev-parse", "HEAD") == revision, "Source revision differs.")
    require(not git(root, "status", "--porcelain", "--untracked-files=normal"),
            "Source checkout is not clean.")
    return {"revision": revision, "tree": git(root, "rev-parse", "HEAD^{tree}"),
            "clean": True, "manifest_sha256": file_digest(root / "Cargo.toml"),
            "lock_sha256": file_digest(root / "Cargo.lock"),
            "toolchain_sha256": file_digest(root / "rust-toolchain.toml")}


def lock_subset(driver_lock: Path, source_lock: Path, registry: bool = False) -> None:
    """An external consumer may omit source packages, never resolve new versions."""
    source = tomllib.loads(source_lock.read_text(encoding="utf-8"))["package"]
    inventory = {(item["name"], item["version"], item.get("source")): item.get("checksum")
                 for item in source}
    for item in tomllib.loads(driver_lock.read_text(encoding="utf-8"))["package"]:
        if item["name"] == "prism-cpu-bundle-driver":
            continue
        key = item["name"], item["version"], item.get("source")
        if registry and item["name"] in REGISTRY_PACKAGES:
            version, checksum = REGISTRY_PACKAGES[item["name"]]
            require(item["version"] == version and item.get("checksum") == checksum
                    and item.get("source") == "registry+https://github.com/rust-lang/crates.io-index",
                    "Registry dependency differs from the approved package bytes.")
            if (item["name"], item["version"], None) in inventory:
                continue  # The packaged root lock describes itself as its root package.
        require(key in inventory and inventory[key] == item.get("checksum"),
                "External collector resolved a dependency outside the source lock.")


def registry_source(output: Path) -> tuple[Path, dict, dict]:
    """Let Cargo verify/extract immutable registry bytes; never unpack tar paths here."""
    with tempfile.TemporaryDirectory(prefix="prism-cpu-registry-resolve-") as temporary:
        lane = Path(temporary)
        (lane / "lib.rs").write_text("// Registry identity resolution only.\n", encoding="utf-8")
        (lane / "Cargo.toml").write_text(
            '[workspace]\n[package]\nname="prism-cpu-registry-resolver"\nversion="0.0.0"\nedition="2024"\n'
            '[lib]\npath="lib.rs"\n[dependencies]\n'
            'prism-inference = { version="=0.1.25", default-features=false, '
            'features=["cli","experimental-neon","parallel-rows"] }\n'
            'bonsai-neon-dot="=0.1.2"\nbonsai-native-hash="=0.1.3"\n', encoding="utf-8")
        cargo = ["cargo", f"+{CHANNEL}"]
        metadata = json.loads(command(cargo + ["metadata", "--manifest-path", str(lane / "Cargo.toml"),
                                              "--format-version", "1"], output))
        locked = tomllib.loads((lane / "Cargo.lock").read_text(encoding="utf-8"))["package"]
        packages = {}
        for name, (version, checksum) in REGISTRY_PACKAGES.items():
            matches = [item for item in locked if item["name"] == name and item["version"] == version]
            require(len(matches) == 1 and matches[0].get("checksum") == checksum
                    and matches[0].get("source") == "registry+https://github.com/rust-lang/crates.io-index",
                    "Cargo registry resolution differs from approved bytes.")
            selected = [item for item in metadata["packages"] if item["name"] == name
                        and item["version"] == version and item.get("source") == matches[0]["source"]]
            require(len(selected) == 1, "Missing unambiguous Cargo-verified registry package.")
            root = Path(selected[0]["manifest_path"]).parent
            verify_registry_tree(root, checksum)
            vcs = read_json(root / ".cargo_vcs_info.json")["git"]["sha1"]
            hex_value(vcs, 40, "registry package VCS identity")
            if name in ("prism-inference", "bonsai-neon-dot"):
                require(vcs == REGISTRY_REVISION, "Registry package VCS source differs.")
            packages[name] = {"name": name, "version": version, "sha256": checksum,
                              "revision": vcs, "source": matches[0]["source"],
                              "manifest_sha256": file_digest(root / "Cargo.toml"),
                              "root": root}
        runtime = packages["prism-inference"]
        root = runtime["root"]
        identity = {"kind": "registry", "revision": runtime["revision"], "clean": True,
                    "clean_scope": "Cargo-verified immutable registry archive; not a Git checkout",
                    "version": runtime["version"], "archive_sha256": runtime["sha256"],
                    "manifest_sha256": runtime["manifest_sha256"], "lock_sha256": file_digest(root / "Cargo.lock")}
        return root, identity, packages


def verify_registry_tree(root: Path, archive_sha256: str) -> None:
    # Cargo registry sources have .cargo-ok, not the vendor-only checksum map.
    # Read the verified archive; compare bytes already extracted by Cargo. Never
    # extract an archive entry or write a path supplied by an archive.
    archive = root.parent.parent.parent / "cache" / root.parent.name / f"{root.name}.crate"
    require(file_digest(archive) == hex_value(archive_sha256, 64, "registry archive digest"),
            "Cargo cache archive differs from the approved publication.")
    seen, total = set(), 0
    with tarfile.open(archive, "r|gz") as stream:
        for entry in stream:
            if entry.isdir():
                continue
            require(entry.isfile(), "Registry source archive contains a non-regular member.")
            parts = PurePosixPath(entry.name).parts
            require(len(parts) >= 2 and parts[0] == root.name and ".." not in parts
                    and "\\" not in entry.name and ":" not in entry.name,
                    "Registry archive member has an invalid source path.")
            relative = "/".join(parts[1:])
            require(relative not in seen and len(seen) < 20000, "Duplicate or excessive registry source members.")
            seen.add(relative)
            total += entry.size
            require(0 <= entry.size <= 64 * 1024 * 1024 and total <= 256 * 1024 * 1024,
                    "Registry source archive exceeds the preparation bound.")
            captured = stream.extractfile(entry)
            require(captured is not None, "Missing regular registry source bytes.")
            with captured:
                expected = hashlib.file_digest(captured, "sha256").hexdigest()
            path = member(root, relative)
            require(path.stat().st_size == entry.size and file_digest(path) == expected,
                    "Cargo-extracted registry source bytes differ from the approved archive.")
    require(seen, "Empty registry source archive.")


def hosted_preparation(runner_revision: str | None = None) -> dict:
    for name, expected in {"GITHUB_ACTIONS": "true", "GITHUB_REF": "refs/heads/main",
                           "RUNNER_ENVIRONMENT": "github-hosted", "RUNNER_ARCH": "ARM64"}.items():
        require(os.environ.get(name) == expected, "Preparation requires owning hosted ARM64 main CI.")
    require(os.environ.get("GITHUB_REPOSITORY") in
            ("symbiogenesis/prism-inference", "symbiogenesis/SwiftBonsai2Runner"),
            "Preparation requires an owning repository workflow.")
    require(sys.platform == "win32", "Preparation requires native Windows ARM64.")
    revision = hex_value(os.environ.get("GITHUB_SHA"), 40, "orchestrator revision")
    require(runner_revision is None or revision == runner_revision, "Runner source revision differs.")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    require(re.fullmatch("[1-9][0-9]*", run_id) is not None, "Missing hosted run identity.")
    return {"repository": os.environ["GITHUB_REPOSITORY"], "revision": revision,
            "run_id": run_id, "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT")}


def prepare(baseline: Path, candidate: Path, revision: str, output: Path,
            runner_revision: str | None = None, registry_version: str | None = None) -> dict:
    orchestrator = hosted_preparation(runner_revision)
    hex_value(revision, 40, "candidate revision")
    require(revision != BASELINE, "Candidate must differ from baseline.")
    require(baseline.is_absolute() and candidate.is_absolute(), "Use absolute source paths.")
    tooling = candidate.resolve()
    require(baseline.resolve() != tooling, "Use independent source checkouts.")
    require(output.is_absolute() and not output.exists(), "Use a new external artifact directory.")
    require(all(not output.resolve().is_relative_to(root) for root in (baseline.resolve(), tooling)),
            "Artifact directory must be outside both source trees.")
    tooling_identity = source_identity(tooling, git(tooling, "rev-parse", "HEAD"))
    require(file_digest(Path(__file__)) == file_digest(tooling / ".github/ci/cpu_bundle.py"),
            "Preparation code differs from its declared tooling source.")
    identities = {"baseline": source_identity(baseline.resolve(), BASELINE)}
    driver_files = ["examples/cpu_bundle_check.rs"] + sorted(
        str(path.relative_to(tooling)).replace("\\", "/")
        for path in (tooling / "examples/cpu_bundle_check").rglob("*.rs"))
    driver_hashes = {name: file_digest(tooling / name) for name in driver_files}
    require(len(driver_hashes) >= 2, "Missing collector source or focused controls.")
    output.mkdir(parents=True)
    packages = {}
    if registry_version is not None:
        require(registry_version == "0.1.25" and revision == REGISTRY_REVISION,
                "Only the reviewed exact registry candidate is admitted.")
        candidate_source, identities["candidate"], packages = registry_source(output)
    else:
        candidate_source = tooling
        identities["candidate"] = source_identity(tooling, revision)
    roots = {"baseline": baseline.resolve(), "candidate": candidate_source}
    messages = json.dumps(MESSAGES, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    (output / "messages.json").write_bytes(messages)
    compiler = command(["rustc", f"+{CHANNEL}", "-vV"], output)
    require(f"host: {TARGET}" in compiler, "Compiler is not native Windows ARM64.")
    arms = {}
    for role, root in roots.items():
        manifest = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
        if (root / "rust-toolchain.toml").is_file():
            require(tomllib.loads((root / "rust-toolchain.toml").read_text(encoding="utf-8"))
                    ["toolchain"]["channel"] == CHANNEL, "Unexpected pinned compiler.")
        legacy = role == "baseline"
        registry = bool(packages) and not legacy
        hash_root = packages["bonsai-native-hash"]["root"] if registry else root / "tools/native-hash"
        hash_manifest = tomllib.loads((hash_root / "Cargo.toml").read_text(encoding="utf-8"))
        flags = "--check-cfg=cfg(cpu_bundle_legacy_f16)" + (" --cfg cpu_bundle_legacy_f16" if legacy else "")
        env = dict(os.environ)
        env.pop("CARGO_ENCODED_RUSTFLAGS", None)
        env["RUSTFLAGS"] = flags
        checks = []
        with tempfile.TemporaryDirectory(prefix=f"prism-cpu-bundle-{role}-") as temporary:
            lane = Path(temporary)
            env["CARGO_TARGET_DIR"] = str(lane / "build")
            driver_manifest = lane / "Cargo.toml"
            runtime_dependency = (f'version = "=0.1.25", ' if registry else
                                  f'path = {json.dumps(root.as_posix())}, version = "={manifest["package"]["version"]}", ')
            hash_dependency = (f'version = "=0.1.3"' if registry else
                               f'path = {json.dumps(hash_root.as_posix())}, version = "={hash_manifest["package"]["version"]}"')
            serde_version = (manifest["dependencies"]["serde_json"]["version"] if registry else
                             manifest["workspace"]["dependencies"]["serde_json"])
            serde_version = "=" + serde_version.lstrip("=")
            driver_manifest.write_text(
                '[workspace]\nresolver = "3"\n[package]\nname = "prism-cpu-bundle-driver"\n'
                'version = "0.0.0"\nedition = "2024"\n[[bin]]\nname = "cpu-bundle-check"\n'
                f'path = {json.dumps((tooling / driver_files[0]).as_posix())}\n'
                '[dependencies]\n'
                f'bonsai-prism-runtime = {{ package = "prism-inference", {runtime_dependency} default-features = false, '
                f'features = {json.dumps(FEATURES)} }}\n'
                f'bonsai-native-hash = {{ {hash_dependency} }}\n'
                f'serde_json = {json.dumps(serde_version)}\n',
                encoding="utf-8")
            cargo = ["cargo", f"+{CHANNEL}"]
            command(cargo + ["fetch", "--locked", "--manifest-path", str(root / "Cargo.toml")], lane, env)
            command(cargo + ["generate-lockfile", "--offline", "--manifest-path", str(driver_manifest)], lane, env)
            lock_subset(lane / "Cargo.lock", root / "Cargo.lock", registry)
            commands = [
                cargo + ["clippy", "--manifest-path", str(root / "Cargo.toml"), "--locked", "--offline",
                         "--release", "-p", "prism-inference", "-p", "bonsai-neon-dot", "-p", "bonsai-native-hash",
                         "--all-targets", "--features", "prism-inference/parallel-rows", "--", "-D", "warnings"],
                cargo + ["check", "--manifest-path", str(root / "Cargo.toml"), "--locked", "--offline",
                         "--release", "-p", "prism-inference", "--lib", "--no-default-features"],
                cargo + ["test", "--manifest-path", str(root / "Cargo.toml"), "--locked", "--offline", "--release",
                         "-p", "prism-inference", "--features", "parallel-rows", "--test", "graph",
                         "graph_parallel_rows_preserve_logits_checkpoints_and_recovery", "--", "--exact"],
                cargo + ["clippy", "--manifest-path", str(driver_manifest), "--locked", "--offline",
                         "--release", "--all-targets", "--", "-D", "warnings"],
                cargo + ["test", "--manifest-path", str(driver_manifest), "--locked", "--offline", "--release",
                         "--bin", "cpu-bundle-check"],
                cargo + ["build", "--manifest-path", str(driver_manifest), "--locked", "--offline", "--release",
                         "--bin", "cpu-bundle-check"],
            ]
            for phase, args in zip(CHECK_PHASES, commands, strict=True):
                command(args, lane, env)
                checks.append({"phase": phase, "command": args, "exit_code": 0})
            executable = output / f"{role}.exe"
            shutil.copyfile(lane / "build/release/cpu-bundle-check.exe", executable)
            arms[role] = {"source": identities[role], "legacy_f16_build": legacy,
                          "executable": {"path": executable.name, "bytes": executable.stat().st_size,
                                         "sha256": file_digest(executable)},
                          "driver_manifest_sha256": file_digest(driver_manifest),
                          "driver_lock_sha256": file_digest(lane / "Cargo.lock"), "rustflags": flags,
                          "checks": checks}
        if not registry:
            require(source_identity(root, identities[role]["revision"]) == identities[role],
                    "Source checkout changed during preparation.")
        else:
            require(file_digest(root / "Cargo.toml") == identities[role]["manifest_sha256"]
                    and file_digest(root / "Cargo.lock") == identities[role]["lock_sha256"],
                    "Registry source build inputs changed during preparation.")
            for package in packages.values():
                verify_registry_tree(package["root"], package["sha256"])
    require(source_identity(tooling, tooling_identity["revision"]) == tooling_identity,
            "Collector/tooling checkout changed during preparation.")
    require({name: file_digest(tooling / name) for name in driver_files} == driver_hashes,
            "Collector sources changed during preparation.")
    shutil.copyfile(Path(__file__), output / "evaluator.py")
    receipt = {"schema_version": 1, "kind": "prism-cpu-bundle-preparation", "compiler": compiler,
               "orchestrator": orchestrator, "evaluator_sha256": file_digest(output / "evaluator.py"),
               "tooling_source": tooling_identity,
               "registry_packages": {name: {key: value for key, value in package.items() if key != "root"}
                                     for name, package in packages.items()},
               "retained_files": ["preparation.json", "messages.json", "evaluator.py"],
               "target": TARGET, "features": FEATURES, "collector_sources": driver_hashes,
               "messages": {"path": "messages.json", "sha256": digest(messages)}, "output_tokens": 32,
               "model_sha256": MODEL_SHA256, "model_bytes": 7_206_168_928,
               "adapter_sha256": ADAPTER_SHA256, "adapter_bytes": 9_682_464,
               "per_arm_timeout_seconds": 90,
               "schedule": SCHEDULE, "profile": PROFILE, "arms": arms,
               "minimum_relative_request_improvement": 0.10, "maximum_owned_arm_regression": 0.10,
               "model_executed": False, "promotion_qualified": False}
    write_new(output / "preparation.json", receipt)
    return receipt


def power(value: dict) -> tuple[bool, bool]:
    require(isinstance(value, dict) and value.get("status") == "observed"
            and value.get("source") == "GetSystemPowerStatus", "Unknown power source.")
    require(type(value.get("ac_online")) is bool and type(value.get("battery_saver")) is bool,
            "Missing known AC/saver observations.")
    charge = value.get("battery_percent")
    require(charge is None or (type(charge) in (int, float) and math.isfinite(charge) and 0 <= charge <= 100),
            "Invalid battery charge observation.")
    return value["ac_online"], value["battery_saver"]


def native_conditions(report: dict, expected_host: dict) -> dict:
    require(isinstance(report, dict) and report.get("schema_version") == 1
            and report.get("kind") == "runner-native-cpu-observations"
            and report.get("status") == "observed" and report.get("architecture") == "aarch64"
            and report.get("scope") == "current-process"
            and report.get("source") == "GetSystemCpuSetInformation", "Unknown native CPU conditions.")
    host = report.get("host_identity")
    require(host == expected_host and host.get("status") == "observed"
            and host.get("source") == "HKLM/HARDWARE/DESCRIPTION/System/CentralProcessor/0;RtlGetVersion;IsProcessorFeaturePresent",
            "Unknown or changed native host identity.")
    for field in ("processor_name", "processor_identifier", "vendor_identifier"):
        require(isinstance(host.get(field), str) and host[field].strip()
                and host[field].strip().lower() != "unknown", "Incomplete native host identity.")
    require("Snapdragon" in host["processor_name"] and "X2" in host["processor_name"],
            "This fixed comparison requires the selected X2 family.")
    version = host.get("windows_version")
    features = host.get("processor_features")
    require(isinstance(version, list) and len(version) == 3
            and all(type(value) is int and 0 <= value < 2**32 for value in version)
            and version[0] > 0 and version[2] > 0, "Missing native kernel version.")
    require(isinstance(features, list) and len(features) <= 256
            and all(type(value) is int and 0 <= value < 256 for value in features)
            and features == sorted(set(features)), "Invalid native feature observations.")
    topology = report.get("topology")
    require(isinstance(topology, dict) and topology.get("cpu_sets"), "Missing CPU topology.")
    integer(topology.get("logical_processors"), 4, 4096, "native logical processor count")
    policy = report.get("power_policy_snapshot")
    require(isinstance(policy, dict) and policy.get("status") == "observed"
            and policy.get("source") == "PowerGetActiveScheme/PowerRegisterForEffectivePowerModeNotifications"
            and policy.get("notification_version") == 2, "Unknown effective power policy.")
    scheme = policy.get("active_scheme")
    require(isinstance(scheme, str) and re.fullmatch("[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}", scheme)
            is not None and scheme != "00000000-0000-0000-0000-000000000000", "Invalid power scheme.")
    require(policy.get("effective_mode") in {"battery-saver", "better-battery", "balanced",
                                            "high-performance", "max-performance", "game-mode", "mixed-reality"},
            "Unknown effective power mode.")
    return {"host": host, "topology": topology, "power": power(report.get("power_snapshot")),
            "scheme": scheme, "mode": policy["effective_mode"]}


def validate_arm(receipt: dict, arm: dict, preparation: dict, role: str, source_deadline: int) -> None:
    expected = dict(PROFILE)
    expected.update({"schema_version": 1, "kind": "swift-cpu-bundle-arm",
                     "source_commit_declaration": preparation["arms"][role]["source"]["revision"],
                     "executable_sha256": preparation["arms"][role]["executable"]["sha256"],
                     "input_sha256": preparation["messages"]["sha256"],
                     "legacy_f16_build": role == "baseline", "output_tokens_reserved": preparation["output_tokens"],
                     "original_deadline_unix_milliseconds": source_deadline,
                     "capture_status": "complete", "arm_succeeded": True, "session_succeeded": True,
                     "terminal_delivered": True, "stage": "complete", "terminal_accounting_matches": True,
                     "reused_tokens": 0, "model_sha256": arm["model_sha256"], "adapter_sha256": arm["adapter_sha256"],
                     "model_bytes": 7_206_168_928, "weight_representation": "PQ2_0", "effective_row_workers": 4,
                     "observed_adapter_sha256": arm["adapter_sha256"], "model_loaded": True,
                     "output_committed": True, "receipt_reserve_milliseconds": 2000,
                     "sampler": "production SwiftSampler and model-declared stops"})
    require(all(key in receipt and type(receipt[key]) is type(value) and receipt[key] == value
                for key, value in expected.items()), "Collector identity/settings/completion differ.")
    integer(receipt.get("prompt_tokens"), 1, 128, "prompt count")
    integer(receipt.get("generated_tokens"), 1, preparation["output_tokens"], "output count")
    require(receipt.get("finish") in ("stop", "length"), "Missing terminal reason.")
    text = receipt.get("output_text")
    require(isinstance(text, str) and text.strip(), "Missing complete returned text.")
    encoded = text.encode("utf-8")
    integer(receipt.get("output_bytes"), 1, 64 * 1024, "returned output bytes")
    require(receipt.get("output_bytes") == len(encoded) and receipt.get("output_sha256") == digest(encoded),
            "Returned text digest or byte count differs.")
    timings = [positive(receipt.get(key), key) for key in
               ("load_milliseconds", "preparation_milliseconds", "request_milliseconds", "elapsed_milliseconds")]
    require(sum(timings[:3]) <= timings[3], "Operation durations exceed their enclosing clock.")


def evaluate(preparation_path: Path, owner_path: Path) -> dict:
    preparation = read_json(preparation_path)
    owner = read_json(owner_path)
    require(preparation.get("schema_version") == 1 and preparation.get("kind") == "prism-cpu-bundle-preparation"
            and preparation.get("target") == TARGET and preparation.get("features") == FEATURES
            and preparation.get("schedule") == SCHEDULE and preparation.get("profile") == PROFILE
            and preparation.get("per_arm_timeout_seconds") == 90
            and preparation.get("model_sha256") == MODEL_SHA256
            and preparation.get("adapter_sha256") == ADAPTER_SHA256
            and preparation.get("model_executed") is False and preparation.get("promotion_qualified") is False,
            "Preparation identity or fixed policy differs.")
    require(preparation.get("retained_files") == ["preparation.json", "messages.json", "evaluator.py"],
            "Evidence selection differs from the declared replay inputs.")
    messages_path = member(preparation_path.parent, preparation.get("messages", {}).get("path"))
    require(file_digest(messages_path) == preparation.get("messages", {}).get("sha256")
            and json.loads(messages_path.read_text(encoding="utf-8")) == MESSAGES,
            "Prepared public workload changed.")
    require(file_digest(preparation_path.parent / "evaluator.py") == preparation.get("evaluator_sha256"),
            "Prepared evaluator bytes changed.")
    collector_sources = preparation.get("collector_sources")
    require(isinstance(collector_sources, dict) and len(collector_sources) >= 2,
            "Missing collector source identities.")
    for name, value in collector_sources.items():
        require(isinstance(name, str) and name.startswith("examples/cpu_bundle_check") and name.endswith(".rs"),
                "Unreviewed collector source member.")
        hex_value(value, 64, "collector source digest")
    require(set(preparation.get("arms", {})) == {"baseline", "candidate"}, "Missing prepared arms.")
    require(preparation["arms"]["baseline"]["source"]["revision"] == BASELINE,
            "Unreviewed baseline source.")
    for role in ("baseline", "candidate"):
        prepared = preparation["arms"][role]
        hex_value(prepared["source"]["revision"], 40, "source revision")
        hex_value(prepared["executable"]["sha256"], 64, "executable digest")
        require(prepared["source"].get("clean") is True and prepared.get("legacy_f16_build") is (role == "baseline")
                and isinstance(prepared.get("checks"), list)
                and [check.get("phase") for check in prepared["checks"]] == CHECK_PHASES
                and all(type(check.get("exit_code")) is int and check["exit_code"] == 0
                        and isinstance(check.get("command"), list) and check["command"]
                        for check in prepared["checks"]),
                "Missing successful source-bound preparation checks.")
    require(preparation["arms"]["candidate"]["source"]["revision"] != BASELINE,
            "Candidate is the baseline source.")
    if preparation["arms"]["candidate"]["source"].get("kind") == "registry":
        candidate = preparation["arms"]["candidate"]["source"]
        require(candidate.get("revision") == REGISTRY_REVISION and candidate.get("version") == "0.1.25"
                and candidate.get("archive_sha256") == REGISTRY_PACKAGES["prism-inference"][1],
                "Registry candidate differs from the reviewed publication.")
        registry = preparation.get("registry_packages")
        require(isinstance(registry, dict) and set(registry) == set(REGISTRY_PACKAGES),
                "Missing exact registry dependency identities.")
        for name, (version, checksum) in REGISTRY_PACKAGES.items():
            package = registry[name]
            require(package.get("name") == name and package.get("version") == version
                    and package.get("sha256") == checksum, "Changed registry dependency bytes.")
            hex_value(package.get("revision"), 40, "registry package VCS identity")
            if name in ("prism-inference", "bonsai-neon-dot"):
                require(package["revision"] == REGISTRY_REVISION, "Changed registry package source identity.")
    integer(preparation.get("output_tokens"), 1, 32, "prepared output cap")
    hex_value(preparation.get("messages", {}).get("sha256"), 64, "messages digest")
    require(owner.get("schema_version") == 1 and owner.get("kind") == "prism-cpu-bundle-owner"
            and owner.get("preparation_sha256") == file_digest(preparation_path)
            and owner.get("workload_predeclared") is True and owner.get("completed") is True,
            "Owner proof is missing, changed or incomplete.")
    budget = integer(owner.get("budget_seconds"), 1, 900, "aggregate allowance")
    frequency = integer(owner.get("clock_frequency"), 1, 10**12, "clock frequency")
    started = integer(owner.get("started_tick"), 0, 2**63 - 1, "original start")
    deadline = integer(owner.get("deadline_tick"), 1, 2**63 - 1, "original deadline")
    retained = integer(owner.get("retention_completed_tick"), 1, 2**63 - 1, "retention completion")
    require(deadline == started + budget * frequency and started < retained <= deadline,
            "Work or retention escaped the original aggregate allowance.")
    source_deadline = integer(owner.get("original_deadline_unix_milliseconds"), 1, 2**63 - 1,
                              "collector original deadline")
    require(owner.get("model_sha256") == MODEL_SHA256 and owner.get("adapter_sha256") == ADAPTER_SHA256,
            "Owner artifact identities differ from the approved preparation.")
    require(owner.get("host_identity") and isinstance(owner["host_identity"], dict), "Missing native host identity.")
    conditions = [power(owner.get("power_before")), power(owner.get("power_after"))]
    arms = owner.get("arms")
    require(isinstance(arms, list) and len(arms) == 4, "The complete fixed ABBA workload is required.")
    receipts, wall_ms, owned_ms, native_snapshots, observations, previous_end = [], [], [], [], [], started
    seen_paths = set()
    for index, (arm, role) in enumerate(zip(arms, SCHEDULE, strict=True)):
        require(arm.get("id") == f"arm-{index + 1}" and arm.get("role") == role,
                "Arm order differs from fixed ABBA schedule.")
        begin = integer(arm.get("started_tick"), 0, 2**63 - 1, "arm start")
        end = integer(arm.get("cleanup_completed_tick"), 1, 2**63 - 1, "arm cleanup")
        require(previous_end <= begin < end <= retained, "Arms overlap or cleanup is incomplete.")
        previous_end = end
        execution_deadline = integer(arm.get("execution_deadline_tick"), 1, 2**63 - 1, "arm deadline")
        arm_unix_deadline = integer(arm.get("collector_deadline_unix_milliseconds"), 1, 2**63 - 1,
                                    "collector arm deadline")
        derived_unix_deadline = source_deadline - ((deadline - execution_deadline) * 1000 // frequency)
        require(begin < execution_deadline <= deadline
                and execution_deadline - begin <= 90 * frequency
                and abs(derived_unix_deadline - arm_unix_deadline) <= 2,
                "Arm deadline is not derived from the original aggregate clock.")
        require(arm.get("host_identity") == owner["host_identity"], "Native host identity changed.")
        conditions += [power(arm.get("power_before")), power(arm.get("power_after"))]
        require(arm.get("owned_job_empty") is True and arm.get("cleanup_succeeded") is True,
                "Owned model cleanup was not confirmed.")
        for name in ("receipt", "started_receipt", "supervision"):
            reference = arm.get(name)
            require(isinstance(reference, dict) and reference.get("path") not in seen_paths,
                    "Missing or reused arm evidence path.")
            seen_paths.add(reference["path"])
        receipt = checked_json(owner_path.parent, arm["receipt"])
        started_receipt = checked_json(owner_path.parent, arm["started_receipt"])
        supervision = checked_json(owner_path.parent, arm["supervision"])
        require(started_receipt.get("source_commit_declaration") == receipt.get("source_commit_declaration")
                and started_receipt.get("original_deadline_unix_milliseconds") == arm_unix_deadline,
                "Started/final source deadline identity differs.")
        require(supervision.get("Status") == "completed" and supervision.get("ExitCode") == 0
                and type(supervision.get("ExitCode")) is int
                and supervision.get("ModelProcessStarted") is True and not supervision.get("CancellationFailure"),
                "Native supervision did not complete successfully.")
        cpu = supervision.get("CpuConditions", {})
        native_snapshots += [native_conditions(cpu.get("Before"), owner["host_identity"]),
                             native_conditions(cpu.get("After"), owner["host_identity"])]
        wall = positive(supervision.get("WallMilliseconds"), "supervised wall time")
        minimum = integer(supervision.get("MinimumAvailablePhysicalBytes"), 4_294_967_296, 2**63 - 1,
                          "physical reserve")
        peak = integer(supervision.get("PeakJobMemoryBytes"), 1, 12_348_030_976, "Job memory peak")
        require(minimum > 0 and peak > 0, "Missing resource observation.")
        validate_arm(receipt, owner, preparation, role, arm_unix_deadline)
        elapsed = (end - begin) * 1000.0 / frequency
        require(receipt["elapsed_milliseconds"] <= wall + 1.0 and wall <= elapsed + 1.0,
                "Collector, supervisor and cleanup durations are inconsistent.")
        first_delta = receipt.get("first_text_delta_milliseconds")
        if first_delta is not None:
            require(positive(first_delta, "first nonempty text callback") <= receipt["request_milliseconds"],
                    "First text callback lies outside the request.")
        phases = receipt.get("generation_phases")
        if phases is not None:
            require(isinstance(phases, list) and phases, "Invalid phase telemetry.")
            names = set()
            for phase in phases:
                require(isinstance(phase, dict) and isinstance(phase.get("phase"), str)
                        and phase["phase"] and phase["phase"] not in names, "Invalid or repeated phase name.")
                names.add(phase["phase"])
                milliseconds = phase.get("milliseconds")
                require(type(milliseconds) in (int, float) and math.isfinite(milliseconds) and milliseconds >= 0,
                        "Invalid phase time.")
                for field in ("attempted_calls", "completed_calls", "attempted_tokens", "completed_tokens_lower_bound"):
                    integer(phase.get(field), 0, 2**63 - 1, field)
                require(phase["completed_calls"] <= phase["attempted_calls"]
                        and phase["completed_tokens_lower_bound"] <= phase["attempted_tokens"]
                        and type(phase.get("partial_progress_unknown")) is bool,
                        "Invalid phase accounting.")
        observations.append({"id": arm["id"], "role": role,
                             "source_revision": receipt["source_commit_declaration"],
                             "prompt_tokens": receipt["prompt_tokens"], "generated_tokens": receipt["generated_tokens"],
                             "finish": receipt["finish"], "load_milliseconds": receipt["load_milliseconds"],
                             "preparation_milliseconds": receipt["preparation_milliseconds"],
                             "request_milliseconds": receipt["request_milliseconds"],
                             "process_wall_milliseconds": wall, "owned_arm_milliseconds": elapsed,
                             "source_request_tokens_per_second": receipt["generated_tokens"] * 1000.0 / receipt["request_milliseconds"],
                             "source_request_tps_scope": "committed output tokens divided by SwiftSession completion time; "
                                                         "includes prefill/sampling, excludes loading/preparation; not post-first-token decode TPS",
                             "first_text_delta_milliseconds": first_delta,
                             "first_text_delta_scope": "first nonempty text callback after completion entry; includes prefill/sampling; "
                                                       "not a model token timestamp or broker HTTP TTFT",
                             "generation_phases": phases, "os_lifetime_peak_job_commit_bytes": peak,
                             "minimum_available_physical_bytes": minimum,
                             "process_memory": supervision.get("ProcessMemory"),
                             "memory_scope": "Job commit, process counters and available RAM overlap; never sum them or claim total RAM; "
                                             "sampled counters may miss short peaks"})
        receipts.append(receipt)
        wall_ms.append(wall)
        owned_ms.append(elapsed)
    require(all(condition == conditions[0] for condition in conditions), "Unknown or mismatched power conditions.")
    require(all(condition == native_snapshots[0] for condition in native_snapshots)
            and all(condition["power"] == conditions[0] for condition in native_snapshots),
            "Native CPU or effective power conditions changed across arms.")
    comparison_fields = ("output_text", "output_sha256", "output_bytes", "prompt_tokens", "generated_tokens", "finish")
    require(all(all(receipt[field] == receipts[0][field] for field in comparison_fields) for receipt in receipts),
            "Output, token counts or terminal reason differ across arms.")
    require(preparation.get("minimum_relative_request_improvement") == 0.10
            and preparation.get("maximum_owned_arm_regression") == 0.10, "Benefit thresholds differ.")
    request_ms = [receipt["preparation_milliseconds"] + receipt["request_milliseconds"] for receipt in receipts]
    ratios = [request_ms[1] / request_ms[0], request_ms[2] / request_ms[3]]
    wall_ratios = [wall_ms[1] / wall_ms[0], wall_ms[2] / wall_ms[3]]
    owned_ratios = [owned_ms[1] / owned_ms[0], owned_ms[2] / owned_ms[3]]
    benefit = all(ratio <= 0.90 for ratio in wall_ratios) and all(ratio <= 1.10 for ratio in owned_ratios)
    return {"schema_version": 1, "kind": "prism-cpu-bundle-comparison", "status": "admitted",
            "short_request_benefit_observed": benefit,
            "primary_metric": "supervised_load_inclusive_process_wall_milliseconds",
            "candidate_over_baseline_process_wall_ratios": wall_ratios,
            "candidate_over_baseline_request_ratios": ratios,
            "candidate_over_baseline_owned_arm_ratios": owned_ratios,
            "observations": observations,
            "ratio_range": [min(wall_ratios), max(wall_ratios)],
            "process_wall_ratio_range": [min(wall_ratios), max(wall_ratios)],
            "request_ratio_range": [min(ratios), max(ratios)], "pairs": 2,
            "complete_answer_observed": all(receipt["finish"] == "stop" for receipt in receipts),
            "budget_seconds": budget, "aggregate_elapsed_seconds": (retained - started) / frequency,
            "power_mode": "ac" if conditions[0][0] else "battery",
            "continuous_power_stability_proven": False, "statistical_significance_qualified": False,
            "benefit_gate_qualified": False, "promotion_qualified": False,
            "DefaultMayChange": False,
            "individual_optimization_verdicts": False, "state_quality_sustained_acceptance_qualified": False,
            "scope": "Two matched reversed short-request pairs for the complete source bundle only; "
                     "source/state/resource and uncertainty acceptance remain independently required."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    build = sub.add_parser("prepare")
    build.add_argument("--baseline-root", type=Path, required=True)
    roots = build.add_mutually_exclusive_group(required=True)
    roots.add_argument("--candidate-root", type=Path)
    roots.add_argument("--tooling-root", type=Path)
    build.add_argument("--candidate-revision")
    build.add_argument("--candidate-registry-version", choices=["0.1.25"])
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--runner-revision")
    compare = sub.add_parser("evaluate")
    compare.add_argument("--preparation", type=Path, required=True)
    compare.add_argument("--owner", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.operation == "prepare":
            registry = args.candidate_registry_version
            require((registry is not None and args.tooling_root is not None and args.candidate_revision is None)
                    or (registry is None and args.candidate_root is not None and args.candidate_revision is not None),
                    "Select either exact registry candidate plus tooling root, or a source candidate root/revision.")
            result = prepare(args.baseline_root, args.tooling_root or args.candidate_root,
                             REGISTRY_REVISION if registry else args.candidate_revision,
                             args.output, args.runner_revision, registry)
        else:
            result = evaluate(args.preparation, args.owner)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (AdmissionError, OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"status": "rejected", "reason": str(error), "promotion_qualified": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
