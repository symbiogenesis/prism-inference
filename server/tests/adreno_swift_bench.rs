//! Bounded owner-device end-to-end Swift-Bonsai-2 Vulkan vs CPU benchmark arms.

#![forbid(unsafe_code)]

#[cfg(windows)]
mod live {
    use bonsai_adreno_vulkan::VulkanPq2Executor;
    use bonsai_prism_runtime::{
        graph::KvCachePolicy,
        kernels::{MatvecBackend, RowParallelism},
        model::swift::Representation,
        runtime::{Cancellation, GenerationPhase, PromptReuse},
        speculative::SuffixDraftConfig,
        swift_session::{
            BranchCheckpoints, CpuPrefill, CpuSpeculation, SwiftSession, TextEvent,
            selected_state_budget,
        },
    };
    use serde_json::json;
    use std::{
        io::Write as _,
        path::PathBuf,
        time::{Duration, Instant},
    };

    type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

    const CONTEXT_TOKENS: usize = 2048;
    const OUTPUT_TOKENS: usize = 16;
    const ROW_WORKERS: usize = 12;
    const PREFILL_TILE_TOKENS: usize = 8;
    const PREFILL_SCRATCH_BYTES: usize = 256 * 1024 * 1024;
    const SPEC_DRAFT_TOKENS: usize = 4;
    const SPEC_VERIFICATION_SCRATCH_BYTES: usize = 32 * 1024 * 1024;
    const VULKAN_WEIGHT_BUDGET_BYTES: usize = 8 * 1024 * 1024 * 1024;

    const PROMPT_TEXT: &str = "Explain how native ARM64 SIMD integer dot-product instructions, coalesced GPU weight layouts, and tiled layer-major prefill scheduling combine to accelerate local quantized LLM inference on Snapdragon X2 Elite processors. Provide three concise technical bullet points covering memory bandwidth, register pressure, and dispatch overhead.";

    const LAYER_PROJECTION_SUFFIXES: [&str; 10] = [
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_qkv.weight",
        "attn_gate.weight",
        "attn_output.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
        "ssm_out.weight",
    ];

    fn resolve_model_path() -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(explicit) = std::env::var_os("PRISM_SWIFT_MODEL_PATH") {
            let path = PathBuf::from(explicit);
            if path.is_file() {
                return Ok(path);
            }
            return Err(format!("missing PRISM_SWIFT_MODEL_PATH file: {}", path.display()).into());
        }
        let local_app_data = std::env::var_os("LOCALAPPDATA")
            .ok_or("LOCALAPPDATA is unset and PRISM_SWIFT_MODEL_PATH was not provided")?;
        let path = PathBuf::from(local_app_data)
            .join("SwiftBonsai2")
            .join("models")
            .join("a3bdac086bbb6b04d87d0108d18943b5f63868f7")
            .join("Swift-Bonsai-2-PQ2_0.gguf");
        if !path.is_file() {
            return Err(format!("missing Swift model file: {}", path.display()).into());
        }
        Ok(path)
    }

    fn load_session(
        cpu_prefill: Option<CpuPrefill>,
        cpu_speculation: Option<CpuSpeculation>,
    ) -> Result<SwiftSession, Box<dyn std::error::Error + Send + Sync>> {
        let model_path = resolve_model_path()?;
        let parallelism = RowParallelism::new(ROW_WORKERS)?;
        let session = SwiftSession::load_profile_with_execution_and_reuse(
            &model_path,
            Representation::Pq2,
            None,
            MatvecBackend::Auto,
            42,
            parallelism,
            KvCachePolicy::F16,
            CONTEXT_TOKENS,
            cpu_prefill,
            cpu_speculation,
            PromptReuse::Disabled,
            BranchCheckpoints::default(),
        )?;
        Ok(session)
    }

    fn prewarm_swift_vulkan(
        session: &SwiftSession,
        executor: &mut VulkanPq2Executor,
    ) -> Result<(usize, f64), Box<dyn std::error::Error + Send + Sync>> {
        let mut tensors = Vec::new();
        for layer in 0..64 {
            for suffix in LAYER_PROJECTION_SUFFIXES {
                if let Some(tensor) = session.projection_tensor(&format!("blk.{layer}.{suffix}")) {
                    tensors.push(tensor);
                }
            }
        }
        if let Some(output) = session.projection_tensor("output.weight") {
            tensors.push(output);
        }
        let started = Instant::now();
        let warmed = executor.prewarm_tensors(&tensors)?;
        Ok((warmed, started.elapsed().as_secs_f64() * 1000.0))
    }

    fn tok_per_sec(tokens: usize, elapsed: Duration) -> f64 {
        let secs = elapsed.as_secs_f64();
        if secs <= 0.0 || tokens == 0 {
            0.0
        } else {
            let count = u32::try_from(tokens).unwrap_or(u32::MAX);
            f64::from(count) / secs
        }
    }

    fn run_standard_arm(
        label: &str,
        use_vulkan: bool,
        tiled: bool,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let load_started = Instant::now();
        let prefill = if tiled {
            Some(CpuPrefill::new(PREFILL_TILE_TOKENS, PREFILL_SCRATCH_BYTES)?)
        } else {
            None
        };
        let mut session = load_session(prefill, None)?;
        let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;

        let (warmed_tensors, cached_bytes, prewarm_ms) = if use_vulkan {
            let mut executor = VulkanPq2Executor::open_default(VULKAN_WEIGHT_BUDGET_BYTES)?;
            let (warmed, ms) = prewarm_swift_vulkan(&session, &mut executor)?;
            let bytes = executor.cached_weight_bytes();
            session.set_projection_executor(Some(Box::new(executor)));
            (warmed, bytes, ms)
        } else {
            (0, 0, 0.0)
        };

        let messages = json!([{ "role": "user", "content": PROMPT_TEXT }]);
        let prompt = session.prepare_with_thinking(&messages, OUTPUT_TOKENS, false)?;

        let cancellation = Cancellation::default();
        let mut timings = bonsai_prism_runtime::runtime::GenerationTimings::default();
        let mut text = String::new();
        let gen_started = Instant::now();
        let completion = session.complete_with_timings(
            &prompt,
            &cancellation,
            None,
            &mut timings,
            |event| {
                if let TextEvent::Delta(chunk) = event {
                    text.push_str(chunk);
                }
                Ok(())
            },
        )?;
        let total_ms = gen_started.elapsed().as_secs_f64() * 1000.0;

        let ingestion = timings.phase(GenerationPhase::PrefillIngestion);
        let first_logits = timings.phase(GenerationPhase::PrefillEvaluation);
        let decode = timings.phase(GenerationPhase::DecodeEvaluation);
        let prefill_dur = ingestion.elapsed + first_logits.elapsed;
        let prefill_toks =
            ingestion.completed_tokens_lower_bound + first_logits.completed_tokens_lower_bound;
        let decode_toks = decode.completed_tokens_lower_bound;

        let mut stdout = std::io::stdout().lock();
        writeln!(
            stdout,
            "{label} load_ms={load_ms:.1} warmed_tensors={warmed_tensors} cached_weight_bytes={cached_bytes} prewarm_ms={prewarm_ms:.1} prompt_tokens={} generated_tokens={} prefill_ms={:.2} prefill_tok_s={:.2} decode_evals={decode_toks} decode_ms={:.2} decode_tok_s={:.2} total_ms={total_ms:.2} text={text:?}",
            completion.prompt_tokens,
            completion.generated_tokens,
            prefill_dur.as_secs_f64() * 1000.0,
            tok_per_sec(prefill_toks, prefill_dur),
            decode.elapsed.as_secs_f64() * 1000.0,
            tok_per_sec(decode_toks, decode.elapsed),
        )?;
        Ok(())
    }

    #[test]
    #[ignore = "Predeclared bounded owner-device Swift-Bonsai-2 timing comparison."]
    fn bench_swift_arm_a_cpu_tiled() -> TestResult {
        run_standard_arm("SWIFT_ARM_A_CPU_TILED", false, true)
    }

    #[test]
    #[ignore = "Predeclared bounded owner-device Swift-Bonsai-2 timing comparison."]
    fn bench_swift_arm_b_vulkan_seq() -> TestResult {
        run_standard_arm("SWIFT_ARM_B_VULKAN_SEQ", true, false)
    }

    #[test]
    #[ignore = "Predeclared bounded owner-device Swift-Bonsai-2 timing comparison."]
    fn bench_swift_arm_c_vulkan_tiled() -> TestResult {
        run_standard_arm("SWIFT_ARM_C_VULKAN_TILED", true, true)
    }

    #[test]
    #[ignore = "Predeclared bounded owner-device Swift-Bonsai-2 timing comparison."]
    fn bench_swift_arm_d_vulkan_tiled_spec() -> TestResult {
        let load_started = Instant::now();
        let prefill = CpuPrefill::new(PREFILL_TILE_TOKENS, PREFILL_SCRATCH_BYTES)?;
        let checkpoint_bytes = selected_state_budget(CONTEXT_TOKENS, KvCachePolicy::F16)?;
        let spec = CpuSpeculation::new(
            SPEC_DRAFT_TOKENS,
            SPEC_VERIFICATION_SCRATCH_BYTES,
            checkpoint_bytes,
        )?;
        let mut session = load_session(Some(prefill), Some(spec))?;
        let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;

        let mut executor = VulkanPq2Executor::open_default(VULKAN_WEIGHT_BUDGET_BYTES)?;
        let (warmed_tensors, prewarm_ms) = prewarm_swift_vulkan(&session, &mut executor)?;
        let cached_bytes = executor.cached_weight_bytes();
        session.set_projection_executor(Some(Box::new(executor)));

        let messages = json!([{ "role": "user", "content": PROMPT_TEXT }]);
        let prompt = session.prepare_with_thinking(&messages, OUTPUT_TOKENS, false)?;

        let cancellation = Cancellation::default();
        let suffix_config = SuffixDraftConfig::new(2, 8, SPEC_DRAFT_TOKENS)?;
        let mut text = String::new();
        let gen_started = Instant::now();
        let (completion, stats) = session.complete_speculative(
            &prompt,
            suffix_config,
            &cancellation,
            None,
            |event| {
                if let TextEvent::Delta(chunk) = event {
                    text.push_str(chunk);
                }
                Ok(())
            },
        )?;
        let total_dur = gen_started.elapsed();
        let total_ms = total_dur.as_secs_f64() * 1000.0;

        let mut stdout = std::io::stdout().lock();
        writeln!(
            stdout,
            "SWIFT_ARM_D_VULKAN_TILED_SPEC load_ms={load_ms:.1} warmed_tensors={warmed_tensors} cached_weight_bytes={cached_bytes} prewarm_ms={prewarm_ms:.1} prompt_tokens={} generated_tokens={} draft_rounds={} proposed={} accepted={} target_evals={} total_ms={total_ms:.2} e2e_tok_s={:.2} text={text:?}",
            completion.prompt_tokens,
            completion.generated_tokens,
            stats.draft_rounds,
            stats.draft_tokens_proposed,
            stats.draft_tokens_accepted,
            stats.target_evaluations,
            tok_per_sec(completion.generated_tokens, total_dur),
        )?;
        Ok(())
    }
}
