"""Read the exact shared runtime/server version for source-bound bundles."""

from pathlib import Path
import re
import tomllib


def release_version(root):
    """Reject package or server dependency drift before naming a bundle."""
    runtime = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
    server = tomllib.loads((root / "server/Cargo.toml").read_text(encoding="utf-8"))
    version = runtime["package"]["version"]
    dependency = server["dependencies"]["bonsai-prism-runtime"]
    if (runtime["package"]["name"] != "prism-inference"
            or server["package"]["name"] != "prism-inference-server"
            or not isinstance(version, str)
            or re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version) is None
            or server["package"]["version"] != version
            or dependency.get("package") != "prism-inference"
            or dependency.get("path") != ".."
            or dependency.get("version") != "=" + version):
        raise ValueError("Runtime, server and exact server dependency versions must match")
    return version


if __name__ == "__main__":
    print(release_version(Path(__file__).resolve().parents[2]))
