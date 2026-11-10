"""Read retained registry evidence without network, compilation or model work."""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import tomllib


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    raw = Path(sys.argv[1]).resolve()
    receipt = json.loads((raw / "receipt.json").read_text())
    clock = json.loads((raw / "final-clock.json").read_text())
    registry = json.loads((raw / "registry-after.json").read_text())["version"]
    inventory = json.loads((raw / "archive.json").read_text())["files"]
    checks = {}
    checks["rawInventoryHashesAndSizes"] = all(
        (raw / name).stat().st_size == item["bytes"]
        and digest(raw / name) == item["sha256"]
        for name, item in inventory.items()
    )
    package = raw / "prism-inference-0.1.24.crate"
    checks["registryChecksumMatchesRetainedPackage"] = (
        digest(package) == registry["checksum"] == receipt["packageAudit"]["sha256"]
        and package.stat().st_size == registry["crate_size"] == receipt["packageAudit"]["bytes"]
        and registry["num"] == "0.1.24" and not registry["yanked"]
    )
    checks["previousVersionAbsent"] = json.loads((raw / "registry-before.json").read_text()) is None
    with tarfile.open(package, "r:gz") as archive:
        prefix = "prism-inference-0.1.24/"
        manifest = tomllib.loads(archive.extractfile(prefix + "Cargo.toml").read().decode())
        vcs = json.load(archive.extractfile(prefix + ".cargo_vcs_info.json"))
        checks["normalizedVcsIdentity"] = (
            manifest["package"]["name"] == "prism-inference"
            and manifest["package"]["version"] == "0.1.24"
            and vcs["git"]["sha1"] == receipt["sourceRevision"]
            and not vcs["git"].get("dirty", False)
            and len(archive.getmembers()) == receipt["packageAudit"]["members"] == 555
        )
        pins = {}
        forbidden = []
        expected = {item["name"]: "=" + item["version"] for item in receipt["helpers"]}
        for group in (manifest, *manifest.get("target", {}).values()):
            for section in ("dependencies", "build-dependencies", "dev-dependencies"):
                for name, dependency in group.get(section, {}).items():
                    forbidden.extend(key for key in ("path", "git", "workspace") if key in dependency)
                    if name in expected:
                        pins[name] = dependency["version"]
        checks["normalizedExactRegistryHelperPins"] = not forbidden and pins == expected == receipt["packageAudit"]["helperPins"]
    source_archive = raw.parent / "source.tar"
    checks["sourceArchiveIdentity"] = digest(source_archive) == receipt["sourceArchiveSha256"]
    with tarfile.open(source_archive) as archive:
        sources = {member.name: archive.extractfile(member).read() for member in archive if member.isfile()}
    helper_checks = []
    folders = {"bonsai-ternary": "ternary", "bonsai-prism": "prism", "bonsai-native-hash": "native-hash", "bonsai-neon-dot": "neon-dot", "bonsai-gguf": "gguf"}
    for helper in receipt["helpers"]:
        name = helper["name"]
        version = helper["version"]
        metadata = json.loads((raw / (name + ".json")).read_text())["version"]
        path = raw / f"{name}-{version}.crate"
        compared = []
        equal = True
        with tarfile.open(path, "r:gz") as archive:
            for member in archive:
                relative = member.name.removeprefix(f"{name}-{version}/")
                if not member.isfile() or relative in ("Cargo.toml", "Cargo.lock", ".cargo_vcs_info.json"):
                    continue
                local = "Cargo.toml" if relative == "Cargo.toml.orig" else relative
                retained = archive.extractfile(member).read()
                committed = sources.get(f"tools/{folders[name]}/{local}")
                equal = equal and committed is not None and retained.replace(b"\r\n", b"\n") == committed.replace(b"\r\n", b"\n")
                compared.append(relative)
        helper_checks.append({"name": name, "version": version, "sha256": digest(path), "membersCompared": len(compared), "passed": equal and compared == helper["sourceFilesCompared"] and digest(path) == metadata["checksum"] == helper["checksum"] and not metadata["yanked"]})
    checks["helperArchiveChecksumsAndSourceMembers"] = len(helper_checks) == 5 and all(item["passed"] for item in helper_checks)
    checks["cleanMainSourceIntakeAndCompletion"] = (
        (raw / "01-source-head.log").read_text().strip() == receipt["sourceRevision"]
        and (raw / "19-source-head.log").read_text().strip() == receipt["sourceRevision"]
        and (raw / "02-source-branch.log").read_text().strip() == "main"
        and (raw / "20-source-branch.log").read_text().strip() == "main"
        and not (raw / "03-source-status.log").read_text().strip()
        and not (raw / "21-source-status.log").read_text().strip()
        and not (raw / "04-unchanged-helper-trees.log").read_text().strip()
        and all(step["exitCode"] == 0 for step in receipt["steps"])
    )
    checks["defaultVerificationAndUploadLogs"] = (
        "Verifying prism-inference v0.1.24" in (raw / "18-normalized-package.log").read_text()
        and "Verifying prism-inference v0.1.24" in (raw / "22-publish.log").read_text()
        and "Published prism-inference v0.1.24 at registry `crates-io`" in (raw / "22-publish.log").read_text()
    )
    original = datetime.fromisoformat(receipt["originalStartedUtc"])
    final = datetime.fromisoformat(clock["powerAfterRetention"]["utc"])
    elapsed = (final - original).total_seconds()
    checks["aggregateClockIncludesPreEntryFailure"] = (
        receipt["originalStartedUtc"] == "2026-11-10T05:49:47.307184+00:00"
        and abs(elapsed - clock["elapsedSecondsThroughRetention"]) < 0.01
        and clock["withinDeclaredAllowance"] and elapsed < 300
        and receipt["clockRestarted"] is False
        and receipt["powerAtOriginalStart"].startswith("unknown")
    )
    samples = [receipt["powerBefore"], receipt["powerAfter"], clock["powerAfterRetention"], *receipt["powerSamples"]]
    for step in receipt["steps"]:
        samples.extend((step["powerBefore"], step["powerAfter"]))
    checks["recordedContinuationPowerSamplesAgree"] = all(item["acConnected"] is True and item["batterySaver"] is False and item["chargePercent"] == 99 for item in samples)
    result = {
        "schemaVersion": 1,
        "kind": "independent-retained-prism-0124-registry-review",
        "sourceRevision": receipt["sourceRevision"],
        "verdict": "passed-scoped-registry-evidence" if all(checks.values()) else "failed",
        "checks": checks,
        "helpers": helper_checks,
        "packageSha256": digest(package),
        "archiveInventoryMembersVerified": len(inventory),
        "powerSamplesVerified": len(samples),
        "aggregateSecondsThroughRetention": clock["elapsedSecondsThroughRetention"],
        "powerAtOriginalStart": "unknown; parser entry failed before power-provider execution",
        "continuationPower": "All retained continuation samples report AC connected, battery saver off, charge 99%; these do not establish uninterrupted stability or the initial power state.",
        "preEntryFailure": "Receipt retains a 0.4715499-second all-zero copied-driver parser failure; continuation and correction remain charged to the original timestamp.",
        "reviewScope": "Offline static hashes, retained API JSON, normalized package metadata, source-member comparison and logs only; no network, workload rerun, compilation, models or CI.",
        "limitations": ["Registry API JSON is retained evidence; review performs no fresh network query.", "Default Cargo verification does not qualify full Foundation, external consumers, private server, Hexagon, model/state/adapter acceptance or performance.", "Original-start power remains unknown; no matched performance comparison is qualified.", "Initial parser-failure provenance is stated by the retained receipt; the review does not independently reproduce the failure."],
    }
    output = Path(__file__).with_name("independent-review.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
