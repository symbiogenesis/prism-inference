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
            name = {"lib": "bonsai_prism_runtime", "server": "prism_inference_server"}.get(key, key)
            records.append({"reason": "compiler-artifact", "profile": {"test": True},
                "target": {"name": name, "kind": ["lib"] if key in {"lib", "server"} else ["test"]},
                "executable": f"/tmp/{key}"})
        encoded = "\n".join(json.dumps(record) for record in records)
        self.assertEqual(set(efficiency.test_artifacts(encoded)), set(efficiency.CONTROLS))
        for invalid in [encoded + "\n" + json.dumps(records[0]),
                        "\n".join(json.dumps(record) for record in records[1:])]:
            with self.assertRaises(ValueError):
                efficiency.test_artifacts(invalid)

    def test_fixed_inventory_is_unique_and_all_controls_exist_in_source(self):
        root = Path(__file__).resolve().parents[2]
        names = [name for controls in efficiency.CONTROLS.values() for name in controls]
        self.assertEqual(len(names), len(set(names)))
        sources = "\n".join(path.read_text(encoding="utf-8")
                            for directory in [root / "src", root / "tests", root / "server/src"]
                            for path in directory.rglob("*.rs"))
        for name in names:
            self.assertIn("fn " + name.split("::")[-1] + "(", sources)


if __name__ == "__main__":
    unittest.main()
