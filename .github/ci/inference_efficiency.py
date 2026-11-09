"""Run the fixed model-free CPU efficiency controls on an owning hosted runner."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


TOOLCHAIN = "+nightly-2026-11-09"
CONTROLS = {
    "lib": [
        "kernels::tests::scalar_column_projection_matches_decoded_rows_without_row_storage",
        "kernels::tests::scalar_column_projection_rejects_nonfinite_and_recovers_output",
        "kernels::tests::bf16_projection_matches_decoded_rows_without_row_storage",
        "kernels::tests::bf16_projection_rejects_bad_payload_input_and_geometry_then_recovers",
        "kernels::pinned_float_tests::aligned_pinned_float_projections_match_native_without_changing_generic_or_bf16",
        "sampling::swift::tests::bounded_top_k_matches_full_candidate_filter_bit_for_bit",
        "sampling::swift::tests::fixed_chain_reuses_filter_and_preserves_original_stage_bits_and_rng",
        "sampling::swift::tests::candidate_storage_reuses_top_k_backing_across_filters_rejection_and_reset",
        "sampling::swift::tests::selection_rejects_nonfinite_discarded_logits_without_losing_rng_or_storage",
        "kernels::pinned_attention::padding_tests::pruned_dot_padding_matches_full_physical_rows_bitwise",
        "kernels::pinned_attention::padding_tests::pruned_value_padding_matches_full_rows_at_every_lane_boundary",
        "kernels::pinned_attention::padding_tests::pruned_softmax_padding_matches_full_physical_groups_bitwise",
        "kernels::pinned_attention::tests::pinned_value_tiles_match_native_capture_and_visit_each_cached_row_once",
        "kernels::pinned_attention::tests::pinned_value_tiles_reject_malformed_extents_and_nonfinite_results",
        "kernels::pinned_attention::tile_tests::pinned_value_vector_tiles_preserve_scalar_bits_at_channel_and_padding_boundaries",
        "kernels::pinned_attention::tile_tests::pinned_value_vector_tiles_preserve_validation_and_finite_failure_publication",
        "kernels::pinned_gdn::tests::pinned_gdn_dispatch_matches_scalar_bits_and_tail",
        "kernels::pinned_gdn::tests::pinned_gdn_retains_fused_vector_and_nonfused_tail_rounding",
        "kernels::pinned_gdn::tests::pinned_gdn_preserves_exceptional_bits_and_first_failure_mutation",
        "kernels::pinned_gdn::tests::pinned_gdn_rejects_mismatched_rows_and_nonfinite_results",
        "graph::tests::pinned_gdn_arithmetic_matches_independent_native_capture",
        "graph::tests::attention_value_tiles_preserve_strided_oracle_bits_across_cache_pages",
        "graph::tests::full_attention_reads_original_interleaved_gates_across_worker_groups",
        "graph::tests::decayed_heads_preserve_bits_and_independent_state_across_scratch_reuse",
        "kernels::prefill::tests::tiled_packed_projection_reuses_blocks_and_preserves_independent_rows",
        "kernels::prefill::tests::tiled_projection_clears_all_outputs_on_late_rejection",
        "kernels::prefill::tests::tiled_projection_clears_completed_rows_on_cancel_and_bad_scale",
        "kernels::prefill::tests::tiled_parallel_projection_matches_serial_and_recovers",
        "kernels::parallel_rows::tile::tests::tiled_parallel_rows_borrow_outputs_and_keep_callback_on_caller",
        "kernels::parallel_rows::tile::tests::tiled_parallel_rows_join_cancellation_and_select_first_row_failure",
        "kernels::parallel_rows::tile::tests::tiled_parallel_rows_worker_panic_cannot_strand_caller",
        "kernels::prefill::dot::tests::tiled_integer_reductions_preserve_extreme_values_and_native_lanes",
        "graph::prefill::gdn::tests::chunkwise_gdn_triangular_solve_matches_sequential_signed_and_zero_decay",
        "graph::prefill::gdn::tests::chunkwise_gdn_rejects_invalid_shapes_and_gates",
        "speculative::tests::suffix_drafter_exhaustively_matches_length_first_overlap_and_tie_order",
        "speculative::tests::suffix_drafter_maximum_windows_match_reference_and_reject_before_writing",
        "runtime::deadline_tests::speculative_suffix_buffered_rounds_restore_state_after_cancel_and_delivery_failure",
        "runtime::deadline_tests::speculative_suffix_generation_and_block_verifier_preserve_exact_greedy_state",
        "runtime::deadline_tests::speculative_rejected_suffix_correction_matches_independent_greedy_requests",
        "runtime::deadline_tests::speculative_rejected_suffix_keeps_both_correction_cancellation_boundaries",
    ],
    "graph": [
        "rank_one_zero_scale_preserves_full_and_recurrent_logits_and_state_bits",
        "rank_one_checkpoint_and_failed_generation_preserve_complete_state",
        "empty_reset_preserves_hybrid_state_after_late_failure_and_checkpoint_restore",
        "empty_checkpoint_omits_recurrent_payload_after_construction_and_reset",
        "empty_checkpoint_restores_hybrid_pages_and_preserves_nonempty_snapshot",
        "empty_checkpoint_rejects_foreign_and_stale_adapter_provenance",
        "empty_checkpoint_request_rollback_replays_fresh_after_delivery_failure",
        "profile_reuse_matches_fresh_state_and_sampled_failure_replay",
        "profile_reuse_preserves_only_exact_scale_bits_and_explicit_reset_boundary",
        "profile_reuse_preserves_mixed_state_and_stop_draws_across_kv_page_seals",
        "prefill::tiled_prefill_exact_hybrid_state_and_logits_cross_pages",
        "prefill::tiled_prefill_packed_ffn_and_adapter_preserve_original_basis_bits",
        "prefill::tiled_prefill_parallel_workers_preserve_state_restore_and_policy_changes",
        "prefill::tiled_prefill_parallel_grouped_kv_and_head_major_recurrent_match_single_token",
        "full_attention_binary16_overflow_does_not_commit_cache",
        "prefill::tiled_prefill_failure_restores_request_and_explicit_reset_clears_first_tile",
        "prefill::chunkwise_gdn_prefill_continuation_restore_and_identity_are_separate",
        "prefill::tiled_prefill_rejects_unadmitted_geometry_and_preserves_state",
        "prefill::tiled_prefill_runtime_failure_restores_all_layer_state_before_replay",
        "prefill::tiled_prefill_adapter_scale_reset_preserves_execution_identity_and_provenance",
    ],
    "swift_sampling": [
        "swift_rng_rolls_back_after_cancelled_or_abandoned_stream_then_retries",
        "swift_rejects_invalid_full_vocabulary_without_consuming_draws_and_resets_seed",
    ],
    "server": [
        "native_backend::tests::private_latest_prefix_policy_preserves_counting_and_transactional_reset_boundaries",
    ],
    "hexagon_runtime": [
        "native_q2::tests::native_q2_batches_two_and_three_resident_sibling_projections_in_single_queue_writes",
        "native_q2::tests::native_q2_lazy_preparation_allocates_only_copied_scratch_after_admission",
        "native_q2::tests::native_q2_lazy_entrypoints_match_copied_bytes_and_clear_failed_publication",
        "native_q2::tests::native_q2_lazy_direct_scale_failure_commits_no_handle_or_output",
        "native_q2::tests::native_q2_nonresident_single_and_tiled_fallback_preserve_packing_and_bounds",
        "native_q2::tests::native_q2_direct_prewarm_matches_copied_tiles_without_scratch",
        "native_q2::tests::native_q2_direct_prewarm_rejects_partially_packed_invalid_scale",
    ],
    "hexagon_native": [
        "native_q2::tests::native_q2_resident_hvx_plans_match_independent_solver_and_wire",
        "native_q2::tests::native_q2_cached_resident_frame_matches_transient_and_refuses_invalid_plan",
    ],
}

LIBRARY_TARGETS = {
    "bonsai_prism_runtime": "lib",
    "prism_inference_server": "server",
    "bonsai_hexagon_runtime": "hexagon_runtime",
    "hexagon_native_rpc": "hexagon_native",
}

# The kernel module is compiled only by the default-feature Windows ARM64
# build. Keep its exact controls out of other targets instead of accepting
# their zero-match results as numerical coverage.
WINDOWS_ARM64_NEON_CONTROLS = [
    "kernels::neon::tests::simd_neon_integer_dot_exhaustive_packed_bytes",
    "kernels::neon::tests::simd_neon_integer_dot_each_lane_and_extremes",
    "kernels::neon::tests::simd_neon_integer_dot_unaligned_random_blocks",
    "kernels::neon::tests::simd_neon_packed_consumption_exhaustive_bytes_and_extreme_lanes",
    "kernels::neon::tests::simd_neon_pq2_pair_accumulation_extremal_mixed_lanes",
    "kernels::pinned_gdn::tests::pinned_gdn_neon_executes_exact_vector_and_tail_contract",
    "kernels::pinned_attention::neon::tests::neon_value_tile_baseline_is_available_and_accumulator_remains_two_kib",
    "kernels::pinned_gdn::neon::tests::pinned_gdn_staged_rows_match_independent_scalar_bits",
    "kernels::pinned_gdn::neon::tests::pinned_gdn_staged_group_failure_keeps_destinations_untouched",
    "kernels::pinned_gdn::neon::tests::pinned_gdn_staged_single_row_retains_late_failure_bits",
    "kernels::pinned_gdn::neon::tests::pinned_gdn_decode_capacities_match_independent_scalar_bits",
    "kernels::pinned_gdn::neon::tests::pinned_gdn_decode_capacities_preserve_failure_and_refusal_bits",
    "kernels::pinned_gdn::neon::tests::pinned_gdn_grouped_decode_capacities_match_independent_scalar_bits",
    "kernels::pinned_gdn::neon::tests::pinned_gdn_grouped_decode_capacities_keep_failure_atomic",
]

WINDOWS_ARM64_NATIVE_CONTROLS = [
    "windows::compute_native::projection_native::tests::native_q2_in_place_preparation_preserves_extents_and_atomic_publication",
    "windows::compute_native::projection_native::tests::resident_native_q2_multi_arena_populates_secondary_buffer_and_batches_across_arenas",
]


def controls_for_target(target):
    """Select fixed exact controls without changing the shared portable gate."""
    controls = {key: list(names) for key, names in CONTROLS.items()}
    if target == "aarch64-pc-windows-msvc":
        controls["lib"].extend(WINDOWS_ARM64_NEON_CONTROLS)
        controls["hexagon_native"].extend(WINDOWS_ARM64_NATIVE_CONTROLS)
    return controls


def successful_exact_control(output, name):
    """A zero-match, ignored, failed or duplicated test can never close a gate."""
    lines = output.replace("\r\n", "\n").splitlines()
    return (
        lines.count("running 1 test") == 1
        and lines.count(f"test {name} ... ok") == 1
        and sum(bool(re.fullmatch(
            r"test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; \d+ filtered out; finished in .+",
            line,
        )) for line in lines) == 1
    )


def test_artifacts(output):
    """Select only explicitly built test executables from Cargo JSON."""
    selected = {}
    for line in output.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("reason") != "compiler-artifact" or not record.get("executable"):
            continue
        target = record["target"]
        if not record.get("profile", {}).get("test"):
            continue
        key = (LIBRARY_TARGETS.get(target["name"], target["name"])
               if target["kind"] == ["lib"] else target["name"])
        if key not in CONTROLS:
            continue
        if key in selected:
            raise ValueError(f"Duplicate test executable for {key}")
        selected[key] = record["executable"]
    if set(selected) != set(CONTROLS):
        raise ValueError("Cargo did not retain every fixed efficiency test executable")
    return selected


def source_hashes(root):
    """Bind all runtime source and selected fixtures rather than a display version."""
    files = {root / "Cargo.toml", root / "Cargo.lock", root / "rust-toolchain.toml"}
    for directory in ["src", "server/src", "tools/ternary/src", "tools/gguf/src", "tools/native-hash/src", "tools/prism/src",
                      "tools/hexagon-native/src", "tools/hexagon-runtime/src", "tools/neon-dot/src"]:
        files.update((root / directory).rglob("*.rs"))
    for directory in ["server", "tools/ternary", "tools/gguf", "tools/native-hash", "tools/prism",
                      "tools/hexagon-native", "tools/hexagon-runtime", "tools/neon-dot"]:
        files.add(root / directory / "Cargo.toml")
        build_script = root / directory / "build.rs"
        if build_script.is_file():
            files.add(build_script)
    files.update((root / "tests/graph").rglob("*.rs"))
    files.update(path for path in (root / "tests/fixtures").rglob("*") if path.is_file())
    files.update(root / path for path in [
        "tests/graph.rs", "tests/swift_sampling.rs", ".github/ci/inference_efficiency.py",
        ".github/ci/test_inference_efficiency.py", ".github/workflows/inference-efficiency.yml",
    ])
    return {str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    arguments = parser.parse_args()
    revision = os.environ.get("GITHUB_SHA", "")
    if (os.environ.get("GITHUB_ACTIONS") != "true"
            or os.environ.get("GITHUB_REF") != "refs/heads/main"
            or not re.fullmatch("[0-9a-f]{40}", revision)
            or os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted"):
        raise SystemExit("Efficiency qualification requires an owning hosted main runner")
    root = arguments.manifest.resolve(strict=True).parent
    evidence = arguments.evidence
    if not evidence.is_absolute() or evidence.resolve().is_relative_to(root):
        raise SystemExit("Efficiency receipts require external absolute evidence storage")
    evidence.mkdir(parents=True, exist_ok=False)
    initial_hashes = source_hashes(root)
    receipt = {
        "schemaVersion": 1,
        "kind": "hosted-model-free-inference-efficiency",
        "sourceCommit": revision,
        "toolchain": TOOLCHAIN.removeprefix("+"),
        "target": os.environ.get("EFFICIENCY_TARGET", "unknown"),
        "sourceSha256": initial_hashes,
        "controls": [],
        "status": "failed",
        "modelsExecuted": 0,
        "benchmarksExecuted": 0,
        "performanceQualified": False,
        "deviceAcceptanceQualified": False,
    }
    try:
        compiler = subprocess.run(["rustc", TOOLCHAIN, "-vV"], capture_output=True, text=True, check=False)
        receipt["compiler"] = {"exitCode": compiler.returncode, "identity": compiler.stdout}
        if compiler.returncode or f"host: {receipt['target']}" not in compiler.stdout.splitlines():
            raise RuntimeError("Efficiency compiler host differs from the selected target")
        command = ["cargo", TOOLCHAIN, "test", "--manifest-path", str(root / "Cargo.toml"),
                   "--locked", "--release", "--package", "prism-inference", "--package", "prism-inference-server",
                   "--package", "bonsai-hexagon-native", "--package", "bonsai-hexagon-runtime",
                   "--features", "prism-inference/parallel-rows,prism-inference-server/parallel-rows",
                   "--lib", "--test", "graph", "--test", "swift_sampling", "--no-run", "--message-format=json"]
        compiled = subprocess.run(command, capture_output=True, text=True, check=False)
        (evidence / "compile.log").write_text(compiled.stdout + compiled.stderr, encoding="utf-8")
        receipt["compile"] = {"arguments": command, "exitCode": compiled.returncode}
        if compiled.returncode:
            raise RuntimeError("Efficiency control compilation failed")
        executables = test_artifacts(compiled.stdout)
        for target, names in controls_for_target(receipt["target"]).items():
            executable = Path(executables[target]).resolve(strict=True)
            executable_hash = hashlib.sha256(executable.read_bytes()).hexdigest()
            for name in names:
                command = [str(executable), "--exact", name, "--nocapture"]
                result = subprocess.run(command, capture_output=True, text=True, check=False)
                output = result.stdout + result.stderr
                log = f"{target}-{name.replace('::', '-')}.log"
                (evidence / log).write_text(output, encoding="utf-8")
                passed = result.returncode == 0 and successful_exact_control(output, name)
                receipt["controls"].append({"name": name, "target": target,
                    "executableSha256": executable_hash, "exitCode": result.returncode,
                    "status": "passed" if passed else "failed", "log": log})
                if not passed:
                    raise RuntimeError(f"Exact efficiency control failed or was absent: {name}")
        if source_hashes(root) != initial_hashes:
            raise RuntimeError("Efficiency sources changed during qualification")
        receipt["status"] = "passed"
    finally:
        (evidence / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
