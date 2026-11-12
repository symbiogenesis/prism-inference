"""Focused synthetic receipt admission controls; no model, compiler or CI access."""

import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import tarfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("cpu_bundle", Path(__file__).with_name("cpu_bundle.py"))
cpu = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cpu)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


class CaptureFixture:
    def __init__(self, root):
        self.root = root
        self.packet = root / "packet"
        self.packet.mkdir()
        self.preparation_path = self.packet / "preparation.json"
        self.owner_path = root / "owner.json"
        self.power = {"status": "observed", "source": "GetSystemPowerStatus", "ac_online": True,
                      "battery_saver": False, "battery_percent": 90}
        self.host = {"status": "observed",
                     "source": "HKLM/HARDWARE/DESCRIPTION/System/CentralProcessor/0;RtlGetVersion;IsProcessorFeaturePresent",
                     "processor_name": "Snapdragon(R) X2 Elite Extreme - X2E94100 - Qualcomm Oryon(TM) CPU",
                     "processor_identifier": "ARMv8 (64-bit)", "vendor_identifier": "Qualcomm Technologies Inc",
                     "windows_version": [10, 0, 26300], "processor_features": [19, 43, 66]}
        self.native = {"schema_version": 1, "kind": "runner-native-cpu-observations", "status": "observed",
                       "architecture": "aarch64", "scope": "current-process", "source": "GetSystemCpuSetInformation",
                       "host_identity": self.host,
                       "topology": {"logical_processors": 18, "cpu_sets": [{"id": 1}]},
                       "power_snapshot": self.power,
                       "power_policy_snapshot": {"status": "observed", "notification_version": 2,
                          "source": "PowerGetActiveScheme/PowerRegisterForEffectivePowerModeNotifications",
                          "active_scheme": "381b4222-f694-41f0-9685-ff5bb260df2e", "effective_mode": "balanced"}}
        (self.packet / "messages.json").write_text(json.dumps(cpu.MESSAGES), encoding="utf-8")
        (self.packet / "evaluator.py").write_text("# synthetic retained evaluator fixture\n", encoding="utf-8")
        self.preparation = {"schema_version": 1, "kind": "prism-cpu-bundle-preparation",
                            "target": cpu.TARGET, "features": cpu.FEATURES, "schedule": cpu.SCHEDULE,
                            "profile": cpu.PROFILE, "model_executed": False, "promotion_qualified": False,
                            "retained_files": ["preparation.json", "messages.json", "evaluator.py"],
                            "evaluator_sha256": cpu.file_digest(self.packet / "evaluator.py"),
                            "collector_sources": {"examples/cpu_bundle_check.rs": "1" * 64,
                                                  "examples/cpu_bundle_check/tests.rs": "2" * 64},
                            "messages": {"path": "messages.json", "sha256": cpu.file_digest(self.packet / "messages.json")},
                            "model_sha256": cpu.MODEL_SHA256, "adapter_sha256": cpu.ADAPTER_SHA256,
                            "per_arm_timeout_seconds": 60, "output_tokens": 32,
                            "minimum_relative_request_improvement": 0.10, "maximum_owned_arm_regression": 0.10,
                            "arms": {}}
        for role in ("baseline", "candidate"):
            self.preparation["arms"][role] = {
                "source": {"revision": cpu.BASELINE if role == "baseline" else "a" * 40, "clean": True},
                "legacy_f16_build": role == "baseline", "executable": {"sha256": "b" * 64 if role == "baseline" else "c" * 64},
                "checks": [{"phase": phase, "exit_code": 0, "command": ["synthetic-fixture"]}
                           for phase in cpu.CHECK_PHASES]}
            for check in self.preparation["arms"][role]["checks"]:
                if check["phase"] in cpu.TEST_INVENTORIES:
                    names = sorted(cpu.TEST_INVENTORIES[check["phase"]])
                    check["test_summary"] = {"passed": len(names), "failed": 0, "ignored": 0,
                                             "measured": 0, "names": names}
                if check["phase"] == "source_analysis":
                    check["helper_checks"] = [{"package": package, "exit_code": 0,
                                                "command": ["synthetic-helper-fixture"],
                                                "manifest_sha256": "1" * 64, "lock_sha256": "2" * 64}
                                               for package in ("bonsai-neon-dot", "bonsai-native-hash")]
        self.owner = {"schema_version": 1, "kind": "prism-cpu-bundle-owner", "workload_predeclared": True,
                      "completed": True, "budget_seconds": 600, "clock_frequency": 1000,
                      "started_tick": 100000, "deadline_tick": 700000, "retention_completed_tick": 300000,
                      "original_deadline_unix_milliseconds": 2000000000000,
                      "model_sha256": cpu.MODEL_SHA256, "adapter_sha256": cpu.ADAPTER_SHA256,
                      "host_identity": self.host, "power_before": self.power, "power_after": self.power,
                      "arms": []}
        self.receipts, self.supervisions, self.started_receipts = [], [], []
        for index, role in enumerate(cpu.SCHEDULE):
            begin = 110000 + index * 20000
            candidate = role == "candidate"
            finish = begin + (1000 if candidate else 1500)
            execution_deadline = begin + 60000
            unix_deadline = self.owner["original_deadline_unix_milliseconds"] - (700000 - execution_deadline)
            output = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10"
            receipt = dict(cpu.PROFILE)
            receipt.update({"schema_version": 1, "kind": "swift-cpu-bundle-arm",
                            "source_commit_declaration": self.preparation["arms"][role]["source"]["revision"],
                            "executable_sha256": self.preparation["arms"][role]["executable"]["sha256"],
                            "input_sha256": self.preparation["messages"]["sha256"],
                            "legacy_f16_build": role == "baseline", "output_tokens_reserved": 32,
                            "original_deadline_unix_milliseconds": unix_deadline,
                            "capture_status": "complete", "arm_succeeded": True, "session_succeeded": True,
                            "terminal_delivered": True, "stage": "complete", "terminal_accounting_matches": True,
                            "reused_tokens": 0, "model_sha256": cpu.MODEL_SHA256, "adapter_sha256": cpu.ADAPTER_SHA256,
                            "model_bytes": 7206168928, "weight_representation": "PQ2_0", "effective_row_workers": 4,
                            "observed_adapter_sha256": cpu.ADAPTER_SHA256, "model_loaded": True,
                            "output_committed": True, "receipt_reserve_milliseconds": 2000,
                            "sampler": "production SwiftSampler and model-declared stops", "prompt_tokens": 20,
                            "generated_tokens": 29, "finish": "stop", "output_text": output,
                            "output_bytes": len(output.encode()), "output_sha256": cpu.digest(output.encode()),
                            "load_milliseconds": 500 if candidate else 800, "preparation_milliseconds": 10,
                            "request_milliseconds": 300 if candidate else 500,
                            "elapsed_milliseconds": 850 if candidate else 1350})
            self.receipts.append(receipt)
            self.started_receipts.append({"source_commit_declaration": receipt["source_commit_declaration"],
                                          "original_deadline_unix_milliseconds": unix_deadline})
            self.supervisions.append({"Status": "completed", "ExitCode": 0, "ModelProcessStarted": True,
                                      "WallMilliseconds": 900 if candidate else 1400,
                                      "MinimumAvailablePhysicalBytes": 8 * 1024**3, "PeakJobMemoryBytes": 2 * 1024**3,
                                      "CpuConditions": {"Before": self.native, "After": self.native}})
            self.owner["arms"].append({"id": f"arm-{index + 1}", "role": role, "started_tick": begin,
                                      "cleanup_completed_tick": finish, "execution_deadline_tick": execution_deadline,
                                      "collector_deadline_unix_milliseconds": unix_deadline,
                                      "host_identity": self.host, "power_before": self.power, "power_after": self.power,
                                      "owned_job_empty": True, "cleanup_succeeded": True})
        self.save()

    def save(self):
        write(self.preparation_path, self.preparation)
        self.owner["preparation_sha256"] = cpu.file_digest(self.preparation_path)
        for index, arm in enumerate(self.owner["arms"]):
            for key, filename, values in (("receipt", "cpu-bundle.json", self.receipts),
                                          ("started_receipt", "cpu-bundle-started.json", self.started_receipts),
                                          ("supervision", "supervision.json", self.supervisions)):
                relative = f"arms/{index + 1}/{filename}"
                path = self.root / relative
                write(path, values[index])
                arm[key] = {"path": relative, "sha256": cpu.file_digest(path)}
        write(self.owner_path, self.owner)

    def evaluate(self):
        self.save()
        return cpu.evaluate(self.preparation_path, self.owner_path)


class CpuBundleAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.capture = CaptureFixture(Path(self.temporary.name))

    def assert_rejected(self):
        with self.assertRaises((cpu.AdmissionError, KeyError, TypeError, OSError)):
            self.capture.evaluate()

    def test_admitted_reversed_pairs_report_observations_without_qualifying_promotion(self):
        result = self.capture.evaluate()
        self.assertTrue(result["short_request_benefit_observed"])
        self.assertEqual(result["pairs"], 2)
        self.assertFalse(result["promotion_qualified"])
        self.assertFalse(result["benefit_gate_qualified"])
        self.assertFalse(result["DefaultMayChange"])
        self.assertFalse(result["individual_optimization_verdicts"])

    def test_losing_request_bundle_remains_admitted_without_a_benefit(self):
        for index in (1, 2):
            self.capture.supervisions[index]["WallMilliseconds"] = 1500
            self.capture.owner["arms"][index]["cleanup_completed_tick"] = self.capture.owner["arms"][index]["started_tick"] + 1600
        result = self.capture.evaluate()
        self.assertFalse(result["short_request_benefit_observed"])
        self.assertFalse(result["promotion_qualified"])

    def test_optional_callback_and_phase_telemetry_keeps_distinct_metric_scopes(self):
        for index, receipt in enumerate(self.capture.receipts):
            receipt["first_text_delta_milliseconds"] = 100
            receipt["generation_phases"] = [{"phase": "decode", "milliseconds": 100,
                                              "attempted_calls": 1, "completed_calls": 1,
                                              "attempted_tokens": 1, "completed_tokens_lower_bound": 1,
                                              "partial_progress_unknown": False}]
            if index in (1, 2):
                receipt["generation_phases"].append({"phase": "new-candidate-phase", "milliseconds": 0,
                                                      "attempted_calls": 0, "completed_calls": 0,
                                                      "attempted_tokens": 0, "completed_tokens_lower_bound": 0,
                                                      "partial_progress_unknown": False})
        result = self.capture.evaluate()
        self.assertEqual(result["observations"][1]["first_text_delta_milliseconds"], 100)
        self.assertEqual(result["observations"][1]["source_request_tokens_per_second"], 29 * 1000 / 300)
        self.assertIn("not post-first-token decode TPS", result["observations"][1]["source_request_tps_scope"])
        self.assertFalse(result["promotion_qualified"])

    def test_optional_callback_outside_request_rejects(self):
        self.capture.receipts[1]["first_text_delta_milliseconds"] = 301
        self.assert_rejected()

    def test_any_missing_failed_or_misnamed_preparation_check_rejects(self):
        for mutation in (lambda checks: checks.pop(), lambda checks: checks[0].update(exit_code=1),
                         lambda checks: checks[0].update(phase="unknown")):
            with self.subTest(mutation=mutation):
                saved = copy.deepcopy(self.capture.preparation)
                mutation(self.capture.preparation["arms"]["candidate"]["checks"])
                self.assert_rejected()
                self.capture.preparation = saved

    def test_libtest_stdout_requires_exact_positive_named_controls(self):
        name = cpu.STATE_CONTROL
        good = f"running 1 test\ntest {name} ... ok\n\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 99 filtered out; finished in 0.01s"
        self.assertEqual(cpu.test_summary(good, [name])["passed"], 1)
        for output in ("", good.replace("1 passed", "0 passed"), good.replace("0 ignored", "1 ignored"),
                       good + "\n" + good, good.replace(f"test {name} ... ok", "test different_control ... ok")):
            with self.subTest(output=output), self.assertRaises(cpu.AdmissionError):
                cpu.test_summary(output, [name])

    def test_preparation_cannot_replace_positive_controls_with_zero_test_receipts(self):
        self.capture.preparation["arms"]["candidate"]["checks"][2]["test_summary"]["passed"] = 0
        self.assert_rejected()

    def test_changed_artifact_backend_precision_or_worker_identity_rejects(self):
        for field, value in (("executable_sha256", "d" * 64), ("execution_device", "npu"),
                             ("kv_cache", "rsq4"), ("effective_row_workers", 6),
                             ("observed_adapter_sha256", "e" * 64), ("output_committed", False)):
            with self.subTest(field=field):
                saved = copy.deepcopy(self.capture.receipts[1])
                self.capture.receipts[1][field] = value
                self.assert_rejected()
                self.capture.receipts[1] = saved

    def test_empty_output_zero_tokens_or_failed_terminal_rejects(self):
        for field, value in (("output_text", ""), ("generated_tokens", 0), ("terminal_delivered", False),
                             ("terminal_accounting_matches", False), ("capture_status", "incomplete")):
            with self.subTest(field=field):
                saved = copy.deepcopy(self.capture.receipts[1])
                self.capture.receipts[1][field] = value
                self.assert_rejected()
                self.capture.receipts[1] = saved

    def test_output_or_accounting_difference_rejects_even_if_each_arm_is_valid(self):
        self.capture.receipts[1]["generated_tokens"] = 28
        self.assert_rejected()

    def test_length_completion_is_never_a_complete_answer_claim(self):
        for receipt in self.capture.receipts:
            receipt["finish"] = "length"
        self.assertFalse(self.capture.evaluate()["complete_answer_observed"])

    def test_unknown_or_mismatched_per_arm_power_rejects(self):
        self.capture.owner["arms"][1]["power_after"] = dict(self.capture.power, ac_online=False)
        self.assert_rejected()

    def test_power_scheme_excursion_rejects_despite_equal_normalized_power(self):
        self.capture.supervisions[1]["CpuConditions"] = copy.deepcopy(self.capture.supervisions[1]["CpuConditions"])
        self.capture.supervisions[1]["CpuConditions"]["After"]["power_policy_snapshot"]["effective_mode"] = "high-performance"
        self.assert_rejected()

    def test_changed_kernel_or_unknown_native_host_rejects(self):
        self.capture.supervisions[1]["CpuConditions"] = copy.deepcopy(self.capture.supervisions[1]["CpuConditions"])
        self.capture.supervisions[1]["CpuConditions"]["After"]["host_identity"]["windows_version"] = [10, 0, 26301]
        self.assert_rejected()

    def test_retention_over_deadline_clock_restart_and_arm_deadline_extension_reject(self):
        for change in (lambda owner: owner.update(retention_completed_tick=700001),
                       lambda owner: owner.update(deadline_tick=800000),
                       lambda owner: owner["arms"][1].update(execution_deadline_tick=500000)):
            with self.subTest(change=change):
                saved = copy.deepcopy(self.capture.owner)
                change(self.capture.owner)
                self.assert_rejected()
                self.capture.owner = saved

    def test_observed_wall_time_cannot_exceed_its_declared_execution_window(self):
        self.capture.supervisions[0]["WallMilliseconds"] = 60002
        self.capture.owner["arms"][0]["cleanup_completed_tick"] = self.capture.owner["arms"][0]["started_tick"] + 61000
        self.assert_rejected()

    def test_incomplete_or_overlapping_cleanup_and_reordered_arms_reject(self):
        for field, value in (("cleanup_succeeded", False), ("owned_job_empty", False),
                             ("started_tick", 110001), ("role", "baseline")):
            with self.subTest(field=field):
                saved = copy.deepcopy(self.capture.owner["arms"][1])
                self.capture.owner["arms"][1][field] = value
                self.assert_rejected()
                self.capture.owner["arms"][1] = saved

    def test_resource_unknowns_failed_exit_and_nonfinite_timing_reject(self):
        for field, value in (("MinimumAvailablePhysicalBytes", None), ("PeakJobMemoryBytes", 13 * 1024**3),
                             ("ExitCode", False), ("Status", "failed")):
            with self.subTest(field=field):
                saved = copy.deepcopy(self.capture.supervisions[1])
                self.capture.supervisions[1][field] = value
                self.assert_rejected()
                self.capture.supervisions[1] = saved
        self.capture.receipts[1]["request_milliseconds"] = -1
        self.assert_rejected()

    def test_raw_member_tampering_and_path_traversal_reject(self):
        self.capture.save()
        path = self.capture.root / self.capture.owner["arms"][0]["receipt"]["path"]
        path.write_text("{}", encoding="utf-8")
        with self.assertRaises(cpu.AdmissionError):
            cpu.evaluate(self.capture.preparation_path, self.capture.owner_path)
        with self.assertRaises(cpu.AdmissionError):
            cpu.member(self.capture.root, "../outside.json")
        with self.assertRaises(cpu.AdmissionError):
            cpu.member(self.capture.root, "C:\\outside.json")

    def test_duplicate_json_and_nonfinite_constants_reject(self):
        path = self.capture.root / "hostile.json"
        for text in ('{"schema_version":1,"schema_version":1}', '{"value":NaN}'):
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(cpu.AdmissionError):
                cpu.read_json(path)

    def test_local_preparation_is_rejected_before_any_command_runs(self):
        with patch.dict(cpu.os.environ, {}, clear=True), patch.object(cpu, "command") as run:
            with self.assertRaises(cpu.AdmissionError):
                cpu.hosted_preparation()
            run.assert_not_called()

    def test_external_consumer_cannot_silently_resolve_new_dependency_versions(self):
        source = self.capture.root / "source.lock"
        consumer = self.capture.root / "consumer.lock"
        source.write_text('[[package]]\nname="dependency"\nversion="1.0.0"\n', encoding="utf-8")
        consumer.write_text('[[package]]\nname="dependency"\nversion="2.0.0"\n', encoding="utf-8")
        with self.assertRaises(cpu.AdmissionError):
            cpu.lock_subset(consumer, source)

    def test_registry_candidate_requires_all_exact_archive_and_vcs_identities(self):
        candidate = self.capture.preparation["arms"]["candidate"]
        candidate["source"].update(kind="registry", revision=cpu.REGISTRY_REVISION, version="0.1.25",
                                   archive_sha256=cpu.REGISTRY_PACKAGES["prism-inference"][1])
        packages = {name: {"name": name, "version": version, "sha256": checksum,
                           "revision": cpu.REGISTRY_REVISION}
                    for name, (version, checksum) in cpu.REGISTRY_PACKAGES.items()}
        self.capture.preparation["registry_packages"] = packages
        for index in (1, 2):
            self.capture.receipts[index]["source_commit_declaration"] = cpu.REGISTRY_REVISION
            self.capture.started_receipts[index]["source_commit_declaration"] = cpu.REGISTRY_REVISION
        self.assertEqual(self.capture.evaluate()["status"], "admitted")
        packages["bonsai-neon-dot"]["sha256"] = "f" * 64
        self.assert_rejected()

    def registry_tree_fixture(self, member_name):
        root = self.capture.root / "registry/src/fixture-index/package-1.0.0"
        root.mkdir(parents=True)
        (root / "Cargo.toml").write_bytes(b"fixed source bytes")
        archive = self.capture.root / "registry/cache/fixture-index/package-1.0.0.crate"
        archive.parent.mkdir(parents=True)
        with tarfile.open(archive, "w:gz") as stream:
            entry = tarfile.TarInfo(member_name)
            entry.size = len(b"fixed source bytes")
            stream.addfile(entry, io.BytesIO(b"fixed source bytes"))
        return root, cpu.file_digest(archive)

    def test_registry_source_bytes_are_compared_to_the_verified_cache_archive(self):
        root, checksum = self.registry_tree_fixture("package-1.0.0/Cargo.toml")
        cpu.verify_registry_tree(root, checksum)
        (root / "Cargo.toml").write_bytes(b"changed source")
        with self.assertRaises(cpu.AdmissionError):
            cpu.verify_registry_tree(root, checksum)

    def test_registry_source_rejects_added_discoverable_files(self):
        root, checksum = self.registry_tree_fixture("package-1.0.0/Cargo.toml")
        (root / "extra.rs").write_text("// unexpected test source\n", encoding="utf-8")
        with self.assertRaises(cpu.AdmissionError):
            cpu.verify_registry_tree(root, checksum)

    def test_registry_archive_traversal_is_rejected_without_extracting_anything(self):
        root, checksum = self.registry_tree_fixture("package-1.0.0/../escape.rs")
        with self.assertRaises(cpu.AdmissionError):
            cpu.verify_registry_tree(root, checksum)
        self.assertFalse((root.parent / "escape.rs").exists())


if __name__ == "__main__":
    unittest.main()
