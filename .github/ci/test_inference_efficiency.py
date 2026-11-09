"""Negative controls for exact model-free inference efficiency admission."""

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inference_efficiency as efficiency  # noqa: E402


class EfficiencyReceiptTests(unittest.TestCase):
    def test_exact_control_rejects_zero_ignored_failed_or_duplicate_matches(self):
        output = ("\nrunning 1 test\ntest selected ... ok\n\n"
                  "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 7 filtered out; finished in 0.01s\n")
        self.assertTrue(efficiency.successful_exact_control(output, "selected"))
        self.assertTrue(efficiency.successful_exact_control(output.replace("\n", "\r\n"), "selected"))
        for invalid in [output.replace("selected", "different"), output.replace("1 passed", "0 passed"),
                        output.replace("0 ignored", "1 ignored"), output + output,
                        output.replace("... ok", "... FAILED")]:
            self.assertFalse(efficiency.successful_exact_control(invalid, "selected"))

    def test_artifacts_require_all_unique_selected_test_executables(self):
        records = []
        for key in efficiency.CONTROLS:
            libraries = {value: name for name, value in efficiency.LIBRARY_TARGETS.items()}
            name = libraries.get(key, key)
            records.append({"reason": "compiler-artifact", "profile": {"test": True},
                "target": {"name": name, "kind": ["lib"] if key in libraries else ["test"]},
                "executable": f"/tmp/{key}"})
        encoded = "\n".join(json.dumps(record) for record in records)
        self.assertEqual(set(efficiency.test_artifacts(encoded)), set(efficiency.CONTROLS))
        for invalid in [encoded + "\n" + json.dumps(records[0]),
                        "\n".join(json.dumps(record) for record in records[1:])]:
            with self.assertRaises(ValueError):
                efficiency.test_artifacts(invalid)

    def test_fixed_inventory_is_unique_and_all_controls_exist_in_source(self):
        root = Path(__file__).resolve().parents[2]
        names = [name for controls in efficiency.controls_for_target("aarch64-pc-windows-msvc").values()
                 for name in controls]
        self.assertEqual(len(names), len(set(names)))
        sources = "\n".join(path.read_text(encoding="utf-8")
                            for directory in [root / "src", root / "tests", root / "server/src",
                                              root / "tools/hexagon-native/src", root / "tools/hexagon-runtime/src"]
                            for path in directory.rglob("*.rs"))
        for name in names:
            self.assertIn("fn " + name.split("::")[-1] + "(", sources)

    def test_neon_controls_require_the_exact_supported_target_without_mutating_portable_gate(self):
        before = {key: list(names) for key, names in efficiency.CONTROLS.items()}
        selected = efficiency.controls_for_target("aarch64-pc-windows-msvc")
        self.assertEqual(set(selected), set(before))
        self.assertEqual(selected["lib"], before["lib"] + efficiency.WINDOWS_ARM64_NEON_CONTROLS)
        self.assertEqual(selected["hexagon_native"],
                         before["hexagon_native"] + efficiency.WINDOWS_ARM64_NATIVE_CONTROLS)
        for target in ["x86_64-pc-windows-msvc", "aarch64-unknown-linux-gnu", "unknown"]:
            self.assertEqual(efficiency.controls_for_target(target), before)
        self.assertEqual(efficiency.CONTROLS, before)
        selected["lib"].clear()
        self.assertEqual(efficiency.CONTROLS, before)


if __name__ == "__main__":
    unittest.main()
