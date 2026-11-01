"""Hosted-only validation of Cargo archives and an independent Rust consumer.

Cargo's multi-package packaging stages unpublished members in a temporary
registry. The external consumer then uses only extracted normalized archives
for the new packages; unchanged dependencies come from crates.io. Nothing is
published, and no model is loaded.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tomllib


PACKAGES = ("bonsai-native-hash", "bonsai-gguf", "prism-inference")

CONSUMER = """use bonsai_prism_runtime::{
    kernels::{MAX_ROW_WORKERS, RowParallelism},
    sampling::{Greedy, Sampler},
    token_input::read_tokens,
};

fn main() -> bonsai_prism_runtime::Result<()> {
    let mut input = &b"[0, 7, 7, 4294967295]"[..];
    assert_eq!(read_tokens(&mut input, 4)?, [0, 7, 7, u32::MAX]);
    assert!(read_tokens(&mut &b"[1, 2]"[..], 1).is_err());
    assert_eq!(Greedy.sample(&[0.0, 1.0, 1.0])?, 1);
    assert_eq!(RowParallelism::new(1)?.workers(), 1);
    assert!(RowParallelism::new(0).is_err());
    assert!(RowParallelism::new(MAX_ROW_WORKERS + 1).is_err());
    #[cfg(feature = "parallel-rows")]
    assert_eq!(RowParallelism::new(4)?.workers(), 4);
    #[cfg(not(feature = "parallel-rows"))]
    assert!(RowParallelism::new(4).is_err());
    Ok(())
}
"""


def main():
    root = Path(__file__).resolve().parents[2]
    evidence = Path(os.environ["PRISM_PACKAGE_EVIDENCE"]).resolve()
    target = Path(os.environ["CARGO_TARGET_DIR"]).resolve()
    if evidence.is_relative_to(root) or target.is_relative_to(root):
        raise RuntimeError("Source-package builds and consumers must be external.")
    evidence.mkdir(parents=True, exist_ok=False)
    commands = []

    def run(*arguments):
        commands.append(list(arguments))
        (evidence / "commands.json").write_text(
            json.dumps(commands, indent=2) + "\n", encoding="utf-8"
        )
        subprocess.run(arguments, cwd=evidence, check=True)

    # Cargo >= 1.90 verifies interdependent unpublished members together.
    selection = [item for name in PACKAGES for item in ("-p", name)]
    run(
        "cargo", "+nightly-2026-10-22", "package", "--manifest-path",
        str(root / "Cargo.toml"), "--locked", *selection,
    )
    versions = {
        "bonsai-native-hash": tomllib.loads((root / "Cargo.toml").read_text())[
            "workspace"
        ]["package"]["version"],
        "bonsai-gguf": tomllib.loads((root / "tools/gguf/Cargo.toml").read_text())[
            "package"
        ]["version"],
        "prism-inference": tomllib.loads((root / "Cargo.toml").read_text())[
            "package"
        ]["version"],
    }
    extracted = evidence / "archives"
    extracted.mkdir()
    inventory = []
    for name, version in versions.items():
        archive = target / "package" / f"{name}-{version}.crate"
        with tarfile.open(archive, "r:gz") as contents:
            contents.extractall(extracted, filter="data")
        package = extracted / f"{name}-{version}"
        manifest = tomllib.loads((package / "Cargo.toml").read_text())
        if "workspace" in manifest or manifest["package"]["version"] != version:
            raise RuntimeError(f"Package {name} did not normalize its manifest.")
        for key in ("dependencies", "build-dependencies", "dev-dependencies"):
            for dependency in manifest.get(key, {}).values():
                if "path" in dependency:
                    raise RuntimeError(f"Package {name} retained a checkout path.")
        with archive.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        inventory.append({
            "name": name,
            "version": version,
            "bytes": archive.stat().st_size,
            "sha256": digest,
        })

    consumer = evidence / "consumer"
    (consumer / "src").mkdir(parents=True)
    runtime = f"../archives/prism-inference-{versions['prism-inference']}"
    gguf = f"../archives/bonsai-gguf-{versions['bonsai-gguf']}"
    native_hash = f"../archives/bonsai-native-hash-{versions['bonsai-native-hash']}"
    (consumer / "Cargo.toml").write_text(
        f'''[package]
name = "prism-package-consumer"
version = "0.0.0"
edition = "2024"
publish = false

[features]
default = ["bonsai-prism-runtime/default"]
parallel-rows = ["bonsai-prism-runtime/parallel-rows"]

[dependencies]
bonsai-prism-runtime = {{ package = "prism-inference", path = "{runtime}", default-features = false }}

[patch.crates-io]
bonsai-gguf = {{ path = "{gguf}" }}
bonsai-native-hash = {{ path = "{native_hash}" }}
''',
        encoding="utf-8",
    )
    (consumer / "src/main.rs").write_text(CONSUMER, encoding="utf-8")
    manifest = str(consumer / "Cargo.toml")
    run("cargo", "+nightly-2026-10-22", "generate-lockfile", "--manifest-path", manifest)
    for features in ("scalar", "default", "parallel-rows"):
        options = {
            "scalar": ("--no-default-features",),
            "default": (),
            "parallel-rows": ("--features", "parallel-rows"),
        }[features]
        run(
            "cargo", "+nightly-2026-10-22", "run", "--manifest-path", manifest,
            "--locked", "--release", *options,
        )
    (evidence / "receipt.json").write_text(
        json.dumps({
            "schema_version": 1,
            "source_revision": os.environ["GITHUB_SHA"],
            "run_id": os.environ["GITHUB_RUN_ID"],
            "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
            "packages": inventory,
            "consumer_variants": ["scalar", "default", "parallel-rows"],
            "models_loaded": False,
            "registry_publication": False,
        }, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
