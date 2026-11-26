//! Native whole-graph Adreno Vulkan session for `--placement vulkan`.
//!
//! Keeps [`bonsai_adreno_vulkan::VulkanBatchedGraphExecutor`] and its owning
//! [`bonsai_prism_runtime::runtime::Runtime`] strictly pinned to the worker
//! model thread inside thread-local storage, avoiding duplicate host CPU
//! `Qwen35` graph/state allocation and providing deterministic teardown.
#![forbid(unsafe_code)]

use super::{
    NativeBackendError, Placement, WorkerOptions,
    profile::{ModelProfile, ProjectionStartup},
    worker::Engine,
};
use crate::{AdmittedChat, PendingChat, Rejection};
use bonsai_prism_runtime::{
    graph::KvCachePolicy,
    model::swift::Representation,
    runtime::{Cancellation, Completion, PromptReuse},
    swift_session::{BranchCheckpoints, MultiRankAdapterSpec, TextEvent},
};
use std::{path::Path, sync::Arc, time::Instant};

#[cfg(windows)]
mod platform {
    use super::{
        AdmittedChat, Arc, BranchCheckpoints, Cancellation, Completion, Engine, Instant,
        KvCachePolicy, ModelProfile, MultiRankAdapterSpec, NativeBackendError, Path, PendingChat,
        Placement, ProjectionStartup, PromptReuse, Rejection, Representation, TextEvent,
        WorkerOptions,
    };
    use bonsai_adreno_vulkan::{
        TdrChunkPolicy, VulkanBatchedAdapters, VulkanBatchedCheckpoint, VulkanBatchedConfig,
        VulkanBatchedGraphExecutor, VulkanKvPrecision, VulkanMultiRankAdapterSpec,
    };
    use bonsai_prism_runtime::{
        Decoder, Result as RuntimeResult,
        model::{
            Model,
            lora::{MultiRankAdapter, MultiRankPolicy},
            swift,
        },
        runtime::{GenerationTimings, Limits, Prefill, Runtime},
        sampling::SwiftSampler,
        swift_chat::SwiftChatSession,
        swift_session::{
            CHECKPOINT_MEMORY_BYTES, MAX_OUTPUT_BYTES, MODEL_MEMORY_BYTES, selected_state_budget,
            validate_execution_capacity_with_reuse,
        },
        text_output::Utf8Output,
        vocabulary::Vocabulary,
    };
    use std::{cell::RefCell, path::PathBuf, rc::Rc};

    const MAX_CHECKPOINT_SLOTS: usize =
        bonsai_prism_runtime::swift_session::MAX_REUSABLE_CHECKPOINTS;

    struct SharedVulkanBatchedExecutor {
        executor: VulkanBatchedGraphExecutor,
        pending_multi_rank_scale: Option<f32>,
    }

    pub(super) struct VulkanBatchedDecoder {
        shared: Rc<RefCell<SharedVulkanBatchedExecutor>>,
        identity: String,
        vocab_size: usize,
    }

    thread_local! {
        static WORKER_VULKAN_RUNTIME: RefCell<Option<Runtime<VulkanBatchedDecoder>>> =
            const { RefCell::new(None) };
    }

    impl Decoder for VulkanBatchedDecoder {
        type Checkpoint = VulkanBatchedCheckpoint;

        fn identity(&self) -> &str {
            &self.identity
        }

        fn vocab_size(&self) -> usize {
            self.vocab_size
        }

        fn checkpoint_bytes(&self) -> usize {
            self.shared
                .try_borrow()
                .map_or(0, |inner| inner.executor.checkpoint_bytes())
        }

        fn admit_work(&self, ingest_tokens: usize, evaluate_tokens: usize) -> RuntimeResult<()> {
            let inner = self
                .shared
                .try_borrow()
                .map_err(|_| runtime_invalid("Vulkan batched executor borrow conflict"))?;
            Decoder::admit_work(&inner.executor, ingest_tokens, evaluate_tokens)
        }

        fn reset(&mut self) -> RuntimeResult<()> {
            let mut inner = self
                .shared
                .try_borrow_mut()
                .map_err(|_| runtime_invalid("Vulkan batched executor borrow conflict"))?;
            if let Some(multi_rank_scale) = inner.pending_multi_rank_scale.take() {
                inner
                    .executor
                    .reset_with_adapter_scales(Some(multi_rank_scale))
                    .map_err(bonsai_adreno_vulkan::Error::into_runtime_error)?;
                self.identity.clear();
                self.identity.push_str(inner.executor.identity_sha256());
                Ok(())
            } else {
                Decoder::reset(&mut inner.executor)
            }
        }

        fn evaluate(&mut self, token: u32) -> RuntimeResult<Vec<f32>> {
            let mut inner = self
                .shared
                .try_borrow_mut()
                .map_err(|_| runtime_invalid("Vulkan batched executor borrow conflict"))?;
            Decoder::evaluate(&mut inner.executor, token)
        }

        fn evaluate_into(&mut self, token: u32, output: &mut Vec<f32>) -> RuntimeResult<()> {
            let mut inner = self
                .shared
                .try_borrow_mut()
                .map_err(|_| runtime_invalid("Vulkan batched executor borrow conflict"))?;
            Decoder::evaluate_into(&mut inner.executor, token, output)
        }

        fn evaluate_finite_into<'a>(
            &mut self,
            token: u32,
            output: &'a mut Vec<f32>,
        ) -> RuntimeResult<bonsai_prism_runtime::sampling::FiniteLogits<'a>> {
            let mut inner = self
                .shared
                .try_borrow_mut()
                .map_err(|_| runtime_invalid("Vulkan batched executor borrow conflict"))?;
            Decoder::evaluate_finite_into(&mut inner.executor, token, output)
        }

        fn ingest(&mut self, token: u32) -> RuntimeResult<()> {
            let mut inner = self
                .shared
                .try_borrow_mut()
                .map_err(|_| runtime_invalid("Vulkan batched executor borrow conflict"))?;
            Decoder::ingest(&mut inner.executor, token)
        }

        fn ingest_block(
            &mut self,
            tokens: &[u32],
            check: &dyn Fn() -> RuntimeResult<()>,
        ) -> RuntimeResult<()> {
            let mut inner = self
                .shared
                .try_borrow_mut()
                .map_err(|_| runtime_invalid("Vulkan batched executor borrow conflict"))?;
            Decoder::ingest_block(&mut inner.executor, tokens, check)
        }

        fn ingest_block_and_evaluate_last_into(
            &mut self,
            tokens: &[u32],
            output: &mut Vec<f32>,
            check: &dyn Fn() -> RuntimeResult<()>,
        ) -> RuntimeResult<()> {
            Self::ingest_block_and_evaluate_last_into(self, tokens, output, check)
        }

        fn batched_evaluation_capacity(&self) -> usize {
            self.shared.try_borrow().map_or(0, |inner| {
                Decoder::batched_evaluation_capacity(&inner.executor)
            })
        }

        fn evaluate_batch_into(
            &mut self,
            tokens: &[u32],
            output: &mut Vec<Vec<f32>>,
            check: &dyn Fn() -> RuntimeResult<()>,
        ) -> RuntimeResult<()> {
            let mut inner = self
                .shared
                .try_borrow_mut()
                .map_err(|_| runtime_invalid("Vulkan batched executor borrow conflict"))?;
            Decoder::evaluate_batch_into(&mut inner.executor, tokens, output, check)
        }

        fn checkpoint(&self) -> RuntimeResult<Self::Checkpoint> {
            let inner = self
                .shared
                .try_borrow()
                .map_err(|_| runtime_invalid("Vulkan batched executor borrow conflict"))?;
            Decoder::checkpoint(&inner.executor)
        }

        fn checkpoint_into(&self, target: &mut Self::Checkpoint) -> RuntimeResult<()> {
            let inner = self
                .shared
                .try_borrow()
                .map_err(|_| runtime_invalid("Vulkan batched executor borrow conflict"))?;
            Decoder::checkpoint_into(&inner.executor, target)
        }

        fn restore(&mut self, checkpoint: &Self::Checkpoint) -> RuntimeResult<()> {
            let mut inner = self
                .shared
                .try_borrow_mut()
                .map_err(|_| runtime_invalid("Vulkan batched executor borrow conflict"))?;
            Decoder::restore(&mut inner.executor, checkpoint)
        }
    }

    impl VulkanBatchedDecoder {
        pub(super) fn ingest_block_and_evaluate_last_into(
            &self,
            tokens: &[u32],
            output: &mut Vec<f32>,
            check: &dyn Fn() -> RuntimeResult<()>,
        ) -> RuntimeResult<()> {
            let mut inner = self
                .shared
                .try_borrow_mut()
                .map_err(|_| runtime_invalid("Vulkan batched executor borrow conflict"))?;
            let mut pending_check_err: Option<bonsai_prism_runtime::Error> = None;
            let mut wrapped_check = || -> Result<(), bonsai_adreno_vulkan::Error> {
                if let Err(err) = check() {
                    pending_check_err = Some(err);
                    return Err(bonsai_adreno_vulkan::Error::Cancelled);
                }
                Ok(())
            };
            inner
                .executor
                .ingest_block_and_evaluate_last_with_check_into(tokens, output, &mut wrapped_check)
                .map_err(|err| {
                    pending_check_err
                        .take()
                        .unwrap_or_else(|| err.into_runtime_error())
                })
        }
    }

    struct PendingVulkanUpload {
        model: Model,
        multi_rank_model: Option<Model>,
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub(in super::super) enum AdapterScaleAction {
        Reuse,
        Reset { update_scale: Option<f32> },
    }

    pub(in super::super) struct VulkanPreparedChat {
        owner: Arc<()>,
        admitted: AdmittedChat,
        multi_rank_scale: Option<f32>,
        tokens: Vec<u32>,
        output_tokens: usize,
        enable_thinking: bool,
    }

    impl VulkanPreparedChat {
        #[must_use]
        pub(in super::super) const fn admitted(&self) -> &AdmittedChat {
            &self.admitted
        }

        #[must_use]
        pub(in super::super) const fn prompt_tokens(&self) -> usize {
            self.tokens.len()
        }

        #[must_use]
        pub(in super::super) const fn thinking_enabled(&self) -> bool {
            self.enable_thinking
        }
    }

    /// Verified whole-graph Adreno Vulkan session without duplicate host `Qwen35` graph allocation.
    #[expect(
        clippy::redundant_pub_crate,
        reason = "Crate-internal re-export to native_child satisfies unreachable_pub."
    )]
    pub(crate) struct VulkanSwiftSession {
        owner: Arc<()>,
        model_path: PathBuf,
        representation: Representation,
        daisyrose_lora: Option<(PathBuf, u64, String)>,
        secondary_adapter_identity: Option<Arc<str>>,
        secondary_adapter_max_rank: Option<u16>,
        chat: SwiftChatSession,
        vocabulary: Vocabulary,
        sampler: SwiftSampler,
        seed: u32,
        row_workers: usize,
        context_tokens: usize,
        prompt_reuse: PromptReuse,
        branch_checkpoints: BranchCheckpoints,
        runtime_checkpoints: usize,
        checkpoint_budget: usize,
        tdr_policy: TdrChunkPolicy,
        delivery_fault: bool,
        pending_upload: Option<PendingVulkanUpload>,
    }

    impl Drop for VulkanSwiftSession {
        fn drop(&mut self) {
            WORKER_VULKAN_RUNTIME.with_borrow_mut(|slot| {
                let _ = slot.take();
            });
        }
    }

    impl VulkanSwiftSession {
        #[expect(
            clippy::too_many_arguments,
            reason = "Matches SwiftSession::load_profile_with_adapters configuration boundary cleanly."
        )]
        pub(crate) fn load_profile_with_adapters(
            path: &Path,
            representation: Representation,
            multi_rank_adapter: Option<MultiRankAdapterSpec<'_>>,
            seed: u32,
            row_workers: usize,
            context_tokens: usize,
            prompt_reuse: PromptReuse,
            branch_checkpoints: BranchCheckpoints,
        ) -> RuntimeResult<Self> {
            if !(1..=bonsai_prism_runtime::kernels::MAX_ROW_WORKERS).contains(&row_workers) {
                return Err(runtime_invalid("invalid row worker count"));
            }
            validate_execution_capacity_with_reuse(
                context_tokens,
                u64::MAX,
                None,
                None,
                KvCachePolicy::F16,
                prompt_reuse,
                branch_checkpoints,
            )?;
            let state_budget = selected_state_budget(context_tokens, KvCachePolicy::F16)?;
            let branch_budget =
                branch_checkpoint_charge(context_tokens, prompt_reuse, branch_checkpoints)?;
            let checkpoint_budget = CHECKPOINT_MEMORY_BYTES
                .max(state_budget)
                .checked_add(branch_budget)
                .ok_or_else(|| runtime_invalid("Swift checkpoint reservation overflow"))?;
            let sampler = SwiftSampler::new(seed)?;
            let (model, _receipt) =
                swift::load_representation(path, MODEL_MEMORY_BYTES, representation)?;
            let chat = SwiftChatSession::from_model_representation(&model, representation)?
                .with_context_tokens(context_tokens)?;
            let vocabulary = Vocabulary::from_metadata(&model.metadata, swift::VOCAB_SIZE)?;

            let (
                multi_rank_model,
                daisyrose_lora,
                secondary_adapter_identity,
                secondary_adapter_max_rank,
            ) = match multi_rank_adapter {
                Some(spec) => {
                    let (loaded, meta, id, max_rank) =
                        load_verified_multi_rank_adapter(&model, spec)?;
                    (Some(loaded), Some(meta), Some(id), Some(max_rank))
                }
                None => (None, None, None, None),
            };

            let runtime_checkpoints = if prompt_reuse == PromptReuse::Disabled {
                0
            } else {
                branch_checkpoints.checkpoints()
            };

            Ok(Self {
                owner: Arc::new(()),
                model_path: path.to_path_buf(),
                representation,
                daisyrose_lora,
                secondary_adapter_identity,
                secondary_adapter_max_rank,
                chat,
                vocabulary,
                sampler,
                seed,
                row_workers,
                context_tokens,
                prompt_reuse,
                branch_checkpoints,
                runtime_checkpoints,
                checkpoint_budget,
                tdr_policy: TdrChunkPolicy::default(),
                delivery_fault: false,
                pending_upload: Some(PendingVulkanUpload {
                    model,
                    multi_rank_model,
                }),
            })
        }

        #[must_use]
        pub(crate) const fn seed(&self) -> u32 {
            self.seed
        }

        #[must_use]
        pub(super) fn is_faulted(&self) -> bool {
            self.delivery_fault
                || WORKER_VULKAN_RUNTIME.with_borrow(|slot| {
                    slot.as_ref().is_some_and(|runtime| {
                        runtime.is_faulted()
                            || runtime
                                .decoder()
                                .shared
                                .try_borrow()
                                .is_ok_and(|inner| inner.executor.is_faulted())
                    })
                })
        }

        pub(super) fn prepare_chat(
            &self,
            pending: PendingChat,
        ) -> Result<VulkanPreparedChat, Rejection> {
            let has_multi_rank = self.secondary_adapter_identity.is_some();
            let mut prepared = None;
            let admitted = pending.admit_with_context(self.context_tokens, |body| {
                if self.is_faulted() {
                    return Err(Rejection::TokenCountUnavailable);
                }
                let value: serde_json::Value =
                    serde_json::from_slice(body).map_err(|_| Rejection::ChatSchema)?;
                let request = crate::native_swift::parse_canonical_request(&value, has_multi_rank)?;
                validate_multi_rank_scale(has_multi_rank, request.multi_rank_scale)
                    .map_err(|_| Rejection::ChatSchema)?;
                if !request.sampler.is_canonical_default_sampling()
                    || request.sampler.reasoning_budget.is_some()
                    || request.sampler.top_logprobs.is_some()
                {
                    return Err(Rejection::ChatSchema);
                }
                let prompt = self
                    .chat
                    .prepare_with_effort(
                        request.messages,
                        request.output_tokens,
                        request.enable_thinking,
                        request.reasoning_effort,
                    )
                    .map_err(|error| {
                        crate::native_swift::classify_preparation_error(error.as_ref())
                    })?;
                let count = prompt.tokenization.token_ids.len();
                prepared = Some((
                    prompt.tokenization.token_ids,
                    request.output_tokens,
                    request.enable_thinking,
                    request
                        .multi_rank_scale
                        .map(MultiRankAdapter::normalize_scale),
                ));
                Ok(count)
            })?;
            let (tokens, output_tokens, enable_thinking, multi_rank_scale) =
                prepared.ok_or(Rejection::TokenCountUnavailable)?;
            Ok(VulkanPreparedChat {
                owner: Arc::clone(&self.owner),
                admitted,
                multi_rank_scale,
                tokens,
                output_tokens,
                enable_thinking,
            })
        }

        fn complete_with_timings(
            &mut self,
            prepared: &VulkanPreparedChat,
            cancellation: &Cancellation,
            deadline: Option<Instant>,
            timings: &mut GenerationTimings,
            mut emit: impl FnMut(TextEvent<'_>) -> RuntimeResult<()>,
        ) -> RuntimeResult<Completion> {
            *timings = GenerationTimings::default();
            if self.is_faulted() {
                return Err(runtime_invalid("Vulkan Swift session requires reset"));
            }
            if !Arc::ptr_eq(&self.owner, &prepared.owner) {
                return Err(runtime_invalid(
                    "prepared Vulkan prompt belongs to another session",
                ));
            }
            validate_multi_rank_scale(
                self.secondary_adapter_identity.is_some(),
                prepared.multi_rank_scale,
            )?;
            check_request(cancellation, deadline)?;

            let options = self.chat.generation(prepared.output_tokens)?;
            let prompt_reuse = self.prompt_reuse;
            let retain_prompt_end = prompt_reuse == PromptReuse::PromptEndAndLatest;

            let mut display = Utf8Output::new();
            let mut output_bytes = 0_usize;
            let completion = WORKER_VULKAN_RUNTIME.with_borrow_mut(|slot| {
                let runtime = slot.as_mut().ok_or_else(|| {
                    runtime_invalid("Vulkan batched runtime is not initialized on worker thread")
                })?;
                if runtime.is_faulted() {
                    return Err(runtime_invalid("Vulkan batched runtime requires reset"));
                }
                let action = {
                    let inner = runtime
                        .decoder()
                        .shared
                        .try_borrow()
                        .map_err(|_| runtime_invalid("Vulkan executor borrow conflict"))?;
                    if inner.executor.is_faulted() {
                        return Err(runtime_invalid("Vulkan batched executor faulted"));
                    }
                    decide_adapter_scale_action(
                        inner.executor.multi_rank_scale(),
                        prepared.multi_rank_scale,
                        prompt_reuse,
                    )?
                };
                if retain_prompt_end {
                    runtime.set_retain_prompt_end(true);
                }
                match action {
                    AdapterScaleAction::Reuse => {}
                    AdapterScaleAction::Reset { update_scale } => {
                        if let Some(mr) = update_scale {
                            let mut inner =
                                runtime.decoder().shared.try_borrow_mut().map_err(|_| {
                                    runtime_invalid("Vulkan executor borrow conflict")
                                })?;
                            inner.pending_multi_rank_scale = Some(mr);
                        }
                        runtime.reset()?;
                    }
                }
                let vocabulary = &self.vocabulary;
                let consume = |token| {
                    emit_decoded_token(
                        vocabulary,
                        &mut display,
                        &mut output_bytes,
                        token,
                        &mut emit,
                    )
                };
                runtime.generate_sampled_until_with_timings(
                    &prepared.tokens,
                    &options,
                    Prefill { block_tokens: 32 },
                    cancellation,
                    deadline,
                    &mut self.sampler,
                    timings,
                    consume,
                )
            })?;

            self.delivery_fault = true;
            let suffix = display.finish();
            charge_output(&mut output_bytes, suffix.len())?;
            if !suffix.is_empty() {
                emit(TextEvent::Delta(suffix))?;
            }
            check_request(cancellation, deadline)?;
            emit(TextEvent::Complete(completion))?;
            check_request(cancellation, deadline)?;
            self.delivery_fault = false;
            Ok(completion)
        }
    }

    impl Engine for VulkanSwiftSession {
        type Prepared = VulkanPreparedChat;

        fn profile(&self) -> Result<ModelProfile, NativeBackendError> {
            Ok(ModelProfile {
                placement: Placement::Vulkan,
                projection_startup: ProjectionStartup::default(),
                representation: self.representation,
                daisyrose_adapter: self.secondary_adapter_identity.is_some(),
                daisyrose_adapter_sha256: self.secondary_adapter_identity.clone(),
                daisyrose_adapter_max_rank: self.secondary_adapter_max_rank,
                vision: false,
                image_max_tokens: None,
                prompt_reuse: self.prompt_reuse,
                prompt_reuse_checkpoints: self.branch_checkpoints.checkpoints(),
                row_workers: self.row_workers,
                context_tokens: self.context_tokens,
                kv_cache: KvCachePolicy::F16,
                cpu_prefill: None,
                cpu_speculation: None,
                startup_graph_identity: None,
            })
        }

        fn prepare(&self, pending: PendingChat) -> Result<Self::Prepared, NativeBackendError> {
            self.prepare_chat(pending)
                .map_err(NativeBackendError::Admission)
        }

        fn properties(prepared: &Self::Prepared) -> (usize, bool, bool, &[u8]) {
            (
                prepared.prompt_tokens(),
                prepared.admitted().streams_response(),
                prepared.thinking_enabled(),
                prepared.admitted().backend_body(),
            )
        }

        fn complete(
            &mut self,
            prepared: &Self::Prepared,
            cancel: &Cancellation,
            deadline: Option<Instant>,
            mut emit: impl FnMut(TextEvent<'_>) -> RuntimeResult<()>,
        ) -> RuntimeResult<Completion> {
            let started = Instant::now();
            let mut first_text = None;
            let mut observe = |event: TextEvent<'_>| {
                if matches!(event, TextEvent::Delta(text) if !text.is_empty())
                    && first_text.is_none()
                {
                    first_text = Some(started.elapsed());
                }
                emit(event)
            };
            let mut timings = GenerationTimings::default();
            let result =
                self.complete_with_timings(prepared, cancel, deadline, &mut timings, &mut observe);
            let _ = super::super::diagnostics::write_generation(
                &mut std::io::stderr().lock(),
                result.is_ok(),
                started.elapsed(),
                prepared.prompt_tokens(),
                first_text,
                Some(&timings),
            );
            result
        }

        fn reset(&mut self) -> RuntimeResult<()> {
            self.delivery_fault = true;
            WORKER_VULKAN_RUNTIME.with_borrow_mut(|slot| {
                let runtime = slot.as_mut().ok_or_else(|| {
                    runtime_invalid("Vulkan batched runtime is not initialized on worker thread")
                })?;
                if let Ok(mut inner) = runtime.decoder().shared.try_borrow_mut() {
                    inner.pending_multi_rank_scale = None;
                }
                runtime.reset()
            })?;
            self.sampler = SwiftSampler::new(self.seed)?;
            self.delivery_fault = false;
            Ok(())
        }

        fn initialize_worker_thread(&mut self) -> Result<ProjectionStartup, NativeBackendError> {
            self.initialize_worker_thread_with_options(Placement::Vulkan, &WorkerOptions::default())
        }

        fn initialize_worker_thread_with_options(
            &mut self,
            placement: Placement,
            _options: &WorkerOptions,
        ) -> Result<ProjectionStartup, NativeBackendError> {
            if placement != Placement::Vulkan {
                return Err(NativeBackendError::Configuration);
            }
            let _ = (&self.model_path, &self.daisyrose_lora);
            WORKER_VULKAN_RUNTIME.with_borrow_mut(|slot| {
                let _ = slot.take();
            });
            let PendingVulkanUpload {
                model,
                multi_rank_model,
            } = self
                .pending_upload
                .take()
                .ok_or(NativeBackendError::Configuration)?;

            let vk_config = VulkanBatchedConfig {
                tdr_policy: self.tdr_policy,
                kv_precision: VulkanKvPrecision::Fp16,
                context_limit: Some(self.context_tokens),
                ..Default::default()
            };
            let adapters = VulkanBatchedAdapters {
                multi_rank: multi_rank_model
                    .as_ref()
                    .map(|adapter| VulkanMultiRankAdapterSpec {
                        adapter,
                        policy: MultiRankPolicy::NpuFriendly,
                        scale: 0.0,
                    }),
            };
            let executor =
                VulkanBatchedGraphExecutor::open_with_adapters(&model, vk_config, adapters)
                    .map_err(|_| NativeBackendError::Configuration)?;
            let layer_count = executor.layer_count();
            let (resident_tensors, resident_weight_bytes) =
                compute_active_resident_weights(&model, layer_count)
                    .ok_or(NativeBackendError::Engine)?;

            // Free host weight storage immediately after GPU upload.
            drop(model);
            drop(multi_rank_model);

            let identity: Arc<str> = Arc::from(format!(
                "adreno-vulkan-batched-v1:{}",
                executor.identity_sha256()
            ));
            let startup = ProjectionStartup::AdrenoVulkan {
                identity,
                resident_tensors,
                resident_weight_bytes,
            };
            if !startup.valid(Placement::Vulkan) {
                return Err(NativeBackendError::Configuration);
            }

            let vocab_size = executor.vocab_size();
            let exec_identity = executor.identity_sha256().to_owned();
            let decoder = VulkanBatchedDecoder {
                shared: Rc::new(RefCell::new(SharedVulkanBatchedExecutor {
                    executor,
                    pending_multi_rank_scale: None,
                })),
                identity: exec_identity,
                vocab_size,
            };
            let mut runtime = Runtime::new(
                decoder,
                Limits {
                    context_tokens: self.context_tokens,
                    checkpoints: self.runtime_checkpoints,
                    checkpoint_bytes: self.checkpoint_budget,
                },
            )
            .map_err(|_| NativeBackendError::Configuration)?;
            runtime.set_retain_prompt_end(self.prompt_reuse == PromptReuse::PromptEndAndLatest);
            WORKER_VULKAN_RUNTIME.with_borrow_mut(|slot| {
                *slot = Some(runtime);
            });
            Ok(startup)
        }
    }

    pub(in super::super) fn decide_adapter_scale_action(
        current_mr: Option<f32>,
        multi_rank_scale: Option<f32>,
        reuse: PromptReuse,
    ) -> RuntimeResult<AdapterScaleAction> {
        let normalized_mr = match (current_mr, multi_rank_scale) {
            (None, None) => None,
            (Some(_), Some(scale)) => {
                MultiRankAdapter::validate_scale(scale)?;
                Some(MultiRankAdapter::normalize_scale(scale))
            }
            _ => {
                return Err(runtime_invalid(
                    "multi-rank request scale does not match the loaded adapter",
                ));
            }
        };
        let scales_match = current_mr.map(f32::to_bits) == normalized_mr.map(f32::to_bits);
        if matches!(reuse, PromptReuse::Latest | PromptReuse::PromptEndAndLatest) && scales_match {
            return Ok(AdapterScaleAction::Reuse);
        }
        Ok(AdapterScaleAction::Reset {
            update_scale: normalized_mr,
        })
    }

    pub(in super::super) fn compute_active_resident_weights(
        model: &Model,
        layer_count: usize,
    ) -> Option<(u64, u64)> {
        let full_interval = if let Some(value) =
            model.metadata.get("qwen35.full_attention_interval")
            && let Some(raw) = value.as_u64()
            && let Ok(parsed) = usize::try_from(raw)
        {
            parsed
        } else {
            4_usize
        };
        if full_interval == 0 {
            return None;
        }
        let recurrent_flags: Vec<bool> = if let Some(value) =
            model.metadata.get("qwen35.attention.recurrent_layers")
            && let Some(arr) = value.as_array()
        {
            let mut flags = Vec::with_capacity(layer_count);
            for item in arr.iter().take(layer_count) {
                flags.push(item.as_bool()?);
            }
            if flags.len() != layer_count {
                return None;
            }
            flags
        } else {
            (0..layer_count)
                .map(|idx| !(idx + 1).is_multiple_of(full_interval))
                .collect()
        };

        let mut count = 0_u64;
        let mut bytes = 0_u64;
        let mut record = |name: &str| -> Option<()> {
            let tensor = model.tensors.get(name)?;
            let len = u64::try_from(tensor.data.len()).ok()?;
            count = count.checked_add(1)?;
            bytes = bytes.checked_add(len)?;
            Some(())
        };

        for (layer_idx, &is_recurrent) in recurrent_flags.iter().enumerate() {
            let common = [
                "attn_norm.weight",
                "post_attention_norm.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            ];
            for suffix in common {
                record(&format!("blk.{layer_idx}.{suffix}"))?;
            }
            if is_recurrent {
                for suffix in [
                    "attn_qkv.weight",
                    "attn_gate.weight",
                    "ssm_beta.weight",
                    "ssm_alpha.weight",
                    "ssm_conv1d.weight",
                    "ssm_dt.bias",
                    "ssm_a",
                    "ssm_norm.weight",
                    "ssm_out.weight",
                ] {
                    record(&format!("blk.{layer_idx}.{suffix}"))?;
                }
            } else {
                for suffix in [
                    "attn_q.weight",
                    "attn_k.weight",
                    "attn_v.weight",
                    "attn_output.weight",
                    "attn_q_norm.weight",
                    "attn_k_norm.weight",
                ] {
                    record(&format!("blk.{layer_idx}.{suffix}"))?;
                }
            }
        }
        record("output_norm.weight")?;
        if model.tensors.contains_key("output.weight") {
            record("output.weight")?;
        } else {
            record("token_embd.weight")?;
        }
        Some((count, bytes))
    }

    fn branch_checkpoint_charge(
        context_tokens: usize,
        prompt_reuse: PromptReuse,
        branch_checkpoints: BranchCheckpoints,
    ) -> RuntimeResult<usize> {
        if !(1..=MAX_CHECKPOINT_SLOTS).contains(&branch_checkpoints.checkpoints()) {
            return Err(runtime_invalid(
                "branch checkpoint count must be from 1 through 8",
            ));
        }
        if prompt_reuse == PromptReuse::Disabled {
            if branch_checkpoints.checkpoints() > 1
                || branch_checkpoints.checkpoint_bytes().is_some()
            {
                return Err(runtime_invalid(
                    "branch checkpoints require prompt reuse to be enabled",
                ));
            }
            return Ok(0);
        }
        let state_budget = selected_state_budget(context_tokens, KvCachePolicy::F16)?;
        let minimum = branch_checkpoints
            .checkpoints()
            .checked_mul(state_budget)
            .ok_or_else(|| runtime_invalid("Swift branch checkpoint reservation overflow"))?;
        if let Some(explicit_bytes) = branch_checkpoints.checkpoint_bytes() {
            if explicit_bytes < minimum {
                return Err(runtime_invalid(
                    "branch checkpoint reserve is below selected state budget",
                ));
            }
            return Ok(explicit_bytes);
        }
        if branch_checkpoints.checkpoints() > 1 || prompt_reuse == PromptReuse::PromptEndAndLatest {
            return Ok(minimum);
        }
        Ok(0)
    }

    type VerifiedMultiRank = (Model, (PathBuf, u64, String), Arc<str>, u16);

    fn load_verified_multi_rank_adapter(
        model: &Model,
        spec: MultiRankAdapterSpec<'_>,
    ) -> RuntimeResult<VerifiedMultiRank> {
        if !spec.path.is_absolute()
            || spec.bytes == 0
            || MultiRankAdapter::DEFAULT_MEMORY_LIMIT == 0
        {
            return Err(runtime_invalid(
                "multi-rank adapter requires an absolute path, positive byte length, and memory limit",
            ));
        }
        let loaded = Model::load(
            spec.path,
            spec.bytes,
            spec.sha256,
            MultiRankAdapter::DEFAULT_MEMORY_LIMIT,
        )?;
        let validated = MultiRankAdapter::from_model(model, &loaded, MultiRankPolicy::NpuFriendly)?;
        let id: Arc<str> = Arc::from(validated.identity());
        let max_rank = u16::try_from(validated.max_rank())?;
        let meta = (spec.path.to_path_buf(), spec.bytes, spec.sha256.to_owned());
        Ok((loaded, meta, id, max_rank))
    }

    fn runtime_invalid(message: &str) -> bonsai_prism_runtime::Error {
        std::io::Error::new(std::io::ErrorKind::InvalidData, message).into()
    }

    fn validate_multi_rank_scale(loaded: bool, scale: Option<f32>) -> RuntimeResult<()> {
        match (loaded, scale) {
            (false, None) => Ok(()),
            (true, Some(scale)) => MultiRankAdapter::validate_scale(scale),
            _ => Err(runtime_invalid(
                "Swift secondary adapter scale does not match the loaded profile",
            )),
        }
    }

    fn charge_output(total: &mut usize, bytes: usize) -> RuntimeResult<()> {
        *total = total
            .checked_add(bytes)
            .filter(|&n| n <= MAX_OUTPUT_BYTES)
            .ok_or_else(|| runtime_invalid("Swift output exceeds bounded byte allowance"))?;
        Ok(())
    }

    fn emit_decoded_token(
        vocabulary: &Vocabulary,
        display: &mut Utf8Output,
        output_bytes: &mut usize,
        token: u32,
        emit: &mut impl FnMut(TextEvent<'_>) -> RuntimeResult<()>,
    ) -> RuntimeResult<()> {
        if vocabulary.is_control(token)? {
            let pending = display.flush_incomplete();
            if !pending.is_empty() {
                charge_output(output_bytes, pending.len())?;
                emit(TextEvent::Delta(pending.as_ref()))?;
            }
        }
        bonsai_prism_runtime::text_output::push_with(display, vocabulary.piece(token)?, |piece| {
            let text = piece.as_ref();
            charge_output(output_bytes, text.len())?;
            if !text.is_empty() {
                emit(TextEvent::Delta(text))?;
            }
            Ok(())
        })
    }

    fn check_request(cancellation: &Cancellation, deadline: Option<Instant>) -> RuntimeResult<()> {
        if cancellation.is_cancelled() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "Swift request cancelled",
            )
            .into());
        }
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            return Err(
                std::io::Error::new(std::io::ErrorKind::TimedOut, "Swift request expired").into(),
            );
        }
        Ok(())
    }
}

#[cfg(not(windows))]
mod platform {
    use super::{
        AdmittedChat, BranchCheckpoints, Cancellation, Completion, Engine, Instant, ModelProfile,
        MultiRankAdapterSpec, NativeBackendError, Path, PendingChat, Placement, ProjectionStartup,
        PromptReuse, Representation, TextEvent, WorkerOptions,
    };

    pub(in super::super) struct VulkanPreparedChat {
        admitted: AdmittedChat,
    }

    #[expect(
        clippy::redundant_pub_crate,
        reason = "Crate-internal re-export to native_child satisfies unreachable_pub."
    )]
    pub(crate) struct VulkanSwiftSession;

    impl VulkanSwiftSession {
        #[expect(
            clippy::too_many_arguments,
            clippy::missing_const_for_fn,
            reason = "Non-Windows stub matches the Windows VulkanSwiftSession loader signature."
        )]
        pub(crate) fn load_profile_with_adapters(
            _path: &Path,
            _representation: Representation,
            _multi_rank_adapter: Option<MultiRankAdapterSpec<'_>>,
            _seed: u32,
            _row_workers: usize,
            _context_tokens: usize,
            _prompt_reuse: PromptReuse,
            _branch_checkpoints: BranchCheckpoints,
        ) -> bonsai_prism_runtime::Result<Self> {
            Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "Adreno Vulkan batched session requires Windows",
            )
            .into())
        }

        #[must_use]
        pub(crate) const fn seed(&self) -> u32 {
            0
        }
    }

    impl Engine for VulkanSwiftSession {
        type Prepared = VulkanPreparedChat;

        fn profile(&self) -> Result<ModelProfile, NativeBackendError> {
            Err(NativeBackendError::Configuration)
        }

        fn prepare(&self, _pending: PendingChat) -> Result<Self::Prepared, NativeBackendError> {
            Err(NativeBackendError::Engine)
        }

        fn properties(prepared: &Self::Prepared) -> (usize, bool, bool, &[u8]) {
            (
                0,
                prepared.admitted.streams_response(),
                false,
                prepared.admitted.backend_body(),
            )
        }

        fn complete(
            &mut self,
            _prepared: &Self::Prepared,
            _cancel: &Cancellation,
            _deadline: Option<Instant>,
            _emit: impl FnMut(TextEvent<'_>) -> bonsai_prism_runtime::Result<()>,
        ) -> bonsai_prism_runtime::Result<Completion> {
            Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "Adreno Vulkan batched session requires Windows",
            )
            .into())
        }

        fn reset(&mut self) -> bonsai_prism_runtime::Result<()> {
            Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "Adreno Vulkan batched session requires Windows",
            )
            .into())
        }

        fn initialize_worker_thread_with_options(
            &mut self,
            _placement: Placement,
            _options: &WorkerOptions,
        ) -> Result<ProjectionStartup, NativeBackendError> {
            Err(NativeBackendError::Configuration)
        }
    }
}

#[cfg(any(windows, test))]
#[expect(
    clippy::redundant_pub_crate,
    reason = "Crate-internal re-export to native_child satisfies unreachable_pub."
)]
pub(crate) use platform::VulkanSwiftSession;
#[cfg(all(windows, test))]
pub(super) use platform::{
    AdapterScaleAction, compute_active_resident_weights, decide_adapter_scale_action,
};
