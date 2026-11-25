//! A single joined model owner and acknowledged one-event output lane.

use super::{
    NativeBackendError, PendingChat, Placement, WorkerOptions,
    profile::{ModelProfile, ProjectionStartup},
};
use bonsai_prism_runtime::{
    graph::KvCachePolicy,
    runtime::{Cancellation, Completion, GenerationTimings},
    swift_session::{RichTextEvent, SwiftSession, TextEvent, TokenLogprobInfo},
};
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
use tokio::sync::{Notify, mpsc as channel};

pub(super) trait Engine: Send + 'static {
    type Prepared: Send;
    fn profile(&self) -> Result<ModelProfile, NativeBackendError> {
        Ok(ModelProfile::default())
    }
    fn prepare(&self, pending: PendingChat) -> Result<Self::Prepared, NativeBackendError>;
    fn prepare_until(
        &self,
        pending: PendingChat,
        cancel: &Cancellation,
        deadline: Option<Instant>,
    ) -> Result<Self::Prepared, NativeBackendError> {
        check_deadline_or_cancel(cancel, deadline)?;
        let prepared = self.prepare(pending)?;
        check_deadline_or_cancel(cancel, deadline)?;
        Ok(prepared)
    }
    fn properties(prepared: &Self::Prepared) -> (usize, bool, bool, &[u8]);
    fn complete(
        &mut self,
        prepared: &Self::Prepared,
        cancel: &Cancellation,
        deadline: Option<Instant>,
        emit: impl FnMut(TextEvent<'_>) -> bonsai_prism_runtime::Result<()>,
    ) -> bonsai_prism_runtime::Result<Completion>;
    fn complete_rich(
        &mut self,
        prepared: &Self::Prepared,
        cancel: &Cancellation,
        deadline: Option<Instant>,
        mut emit: impl FnMut(RichTextEvent<'_>) -> bonsai_prism_runtime::Result<()>,
    ) -> bonsai_prism_runtime::Result<Completion> {
        self.complete(prepared, cancel, deadline, |event| match event {
            TextEvent::Delta(text) => emit(RichTextEvent::Delta {
                text,
                logprobs: None,
            }),
            TextEvent::Complete(value) => emit(RichTextEvent::Complete(value)),
        })
    }
    fn reset(&mut self) -> bonsai_prism_runtime::Result<()>;
    fn initialize_worker_thread(&mut self) -> Result<ProjectionStartup, NativeBackendError> {
        Ok(ProjectionStartup::Cpu)
    }
    fn initialize_worker_thread_for_placement(
        &mut self,
        placement: Placement,
    ) -> Result<ProjectionStartup, NativeBackendError> {
        match placement {
            Placement::Cpu => Ok(ProjectionStartup::Cpu),
            Placement::Auto | Placement::Vulkan | Placement::Composite => {
                self.initialize_worker_thread()
            }
        }
    }
    fn initialize_worker_thread_with_options(
        &mut self,
        placement: Placement,
        _options: &WorkerOptions,
    ) -> Result<ProjectionStartup, NativeBackendError> {
        self.initialize_worker_thread_for_placement(placement)
    }
}

fn initialize_with_options(
    engine: &mut impl Engine,
    placement: Placement,
    options: &WorkerOptions,
) -> Result<ProjectionStartup, NativeBackendError> {
    engine.initialize_worker_thread_with_options(placement, options)
}

impl Engine for SwiftSession {
    type Prepared = crate::native_swift::NativeChat;
    fn profile(&self) -> Result<ModelProfile, NativeBackendError> {
        if self.adapter_identity().is_some_and(|identity| {
            identity != bonsai_prism_runtime::model::lora::SWIFT_ADAPTER_SHA256
        }) {
            return Err(NativeBackendError::Configuration);
        }
        Ok(ModelProfile {
            placement: Placement::Auto,
            projection_startup: ProjectionStartup::default(),
            representation: self.representation(),
            adapter: self.adapter_identity().is_some(),
            daisyrose_adapter: self.secondary_adapter_identity().is_some(),
            daisyrose_adapter_sha256: self.secondary_adapter_identity().map(Arc::from),
            daisyrose_adapter_max_rank: self
                .secondary_adapter_max_rank()
                .and_then(|rank| u16::try_from(rank).ok()),
            vision: self.has_vision(),
            image_max_tokens: self.has_vision().then(|| self.max_image_tokens()),
            prompt_reuse: self.prompt_reuse(),
            prompt_reuse_checkpoints: self.prompt_reuse_checkpoints(),
            row_workers: self.row_workers(),
            context_tokens: self.context_tokens(),
            kv_cache: self.kv_cache_policy(),
            cpu_prefill: self.cpu_prefill(),
            cpu_speculation: self.cpu_speculation(),
            startup_graph_identity: (self.kv_cache_policy() != KvCachePolicy::F16
                || self.cpu_prefill().is_some())
            .then(|| Arc::from(self.decoder_identity())),
        })
    }
    fn prepare(&self, pending: PendingChat) -> Result<Self::Prepared, NativeBackendError> {
        crate::native_swift::prepare(self, pending).map_err(NativeBackendError::Admission)
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
        mut emit: impl FnMut(TextEvent<'_>) -> bonsai_prism_runtime::Result<()>,
    ) -> bonsai_prism_runtime::Result<Completion> {
        Engine::complete_rich(self, prepared, cancel, deadline, |event| match event {
            RichTextEvent::Delta { text, .. } if !text.is_empty() => emit(TextEvent::Delta(text)),
            RichTextEvent::Delta { .. } => Ok(()),
            RichTextEvent::Complete(value) => emit(TextEvent::Complete(value)),
        })
    }
    fn complete_rich(
        &mut self,
        prepared: &Self::Prepared,
        cancel: &Cancellation,
        deadline: Option<Instant>,
        mut emit: impl FnMut(RichTextEvent<'_>) -> bonsai_prism_runtime::Result<()>,
    ) -> bonsai_prism_runtime::Result<Completion> {
        let started = Instant::now();
        let mut first_text = None;
        let mut observe = |event: RichTextEvent<'_>| {
            if matches!(event, RichTextEvent::Delta { text, .. } if !text.is_empty())
                && first_text.is_none()
            {
                first_text = Some(started.elapsed());
            }
            emit(event)
        };
        let mut timings = GenerationTimings::default();
        let measured = self.cpu_speculation().is_none();
        let result = if measured {
            crate::native_swift::complete_rich_with_timings(
                self,
                prepared,
                cancel,
                deadline,
                &mut timings,
                &mut observe,
            )
        } else {
            crate::native_swift::complete_rich(self, prepared, cancel, deadline, &mut observe)
        };
        let _ = super::diagnostics::write_generation(
            &mut std::io::stderr().lock(),
            result.is_ok(),
            started.elapsed(),
            prepared.prompt_tokens(),
            first_text,
            measured.then_some(&timings),
        );
        result
    }
    fn reset(&mut self) -> bonsai_prism_runtime::Result<()> {
        Self::reset(self)
    }
    fn initialize_worker_thread(&mut self) -> Result<ProjectionStartup, NativeBackendError> {
        initialize_auto_executor(self, None)
    }
    fn initialize_worker_thread_for_placement(
        &mut self,
        placement: Placement,
    ) -> Result<ProjectionStartup, NativeBackendError> {
        self.initialize_worker_thread_with_options(placement, &WorkerOptions::default())
    }
    fn initialize_worker_thread_with_options(
        &mut self,
        placement: Placement,
        options: &WorkerOptions,
    ) -> Result<ProjectionStartup, NativeBackendError> {
        let arena_cache_dir = options.hexagon_arena_cache.as_deref();
        match placement {
            Placement::Cpu => Ok(ProjectionStartup::Cpu),
            Placement::Auto => initialize_auto_executor(self, arena_cache_dir),
            Placement::Vulkan => install_vulkan_executor(self),
            Placement::Composite => install_composite_executor(self, arena_cache_dir),
        }
    }
}

fn initialize_auto_executor(
    session: &mut SwiftSession,
    arena_cache_dir: Option<&Path>,
) -> Result<ProjectionStartup, NativeBackendError> {
    #[cfg(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"))]
    if let Some(startup) = install_default_hexagon_executor(session, arena_cache_dir) {
        return Ok(startup);
    }
    #[cfg(not(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc")))]
    let _ = arena_cache_dir;
    let Some(identity) = session.projection_executor_identity() else {
        return Ok(ProjectionStartup::Cpu);
    };
    if identity.is_empty()
        || identity.len() > 256
        || !identity.bytes().all(|byte| byte.is_ascii_graphic())
    {
        return Err(NativeBackendError::Configuration);
    }
    Ok(ProjectionStartup::External(Arc::from(identity)))
}

#[cfg(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"))]
static HEXAGON_PACKAGE_HINT: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

#[cfg(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"))]
type WorkerCompositeExecutor = bonsai_hexagon_runtime::OwnedCompositeProjectionExecutor<
    bonsai_hexagon_runtime::NativeQ2Executor<hexagon_native_rpc::NativeQ2Session>,
    bonsai_adreno_vulkan::VulkanPq2Executor,
>;

#[cfg(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"))]
thread_local! {
    static WORKER_HEXAGON_EXECUTOR: std::cell::RefCell<
        Option<bonsai_hexagon_runtime::NativeQ2Executor<hexagon_native_rpc::NativeQ2Session>>,
    > = const { std::cell::RefCell::new(None) };
    static WORKER_COMPOSITE_EXECUTOR: std::cell::RefCell<Option<WorkerCompositeExecutor>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"))]
struct WorkerThreadHexagonExecutor;

#[cfg(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"))]
struct WorkerThreadCompositeExecutor {
    identity: Arc<str>,
}

#[cfg(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"))]
macro_rules! impl_tls_pq2_executor {
    ($ty:ty, $tls:ident, $missing:expr, |$this:ident| $identity:expr) => {
        impl bonsai_prism_runtime::kernels::Pq2ProjectionExecutor for $ty {
            fn identity(&self) -> &str {
                let $this = self;
                $identity
            }

            fn operation_capacity(&self) -> Option<u64> {
                $tls.with_borrow(|slot| {
                    slot.as_ref().and_then(
                        bonsai_prism_runtime::kernels::Pq2ProjectionExecutor::operation_capacity,
                    )
                })
            }

            fn operation_count(&self, tensor: &bonsai_prism_runtime::model::Tensor) -> Option<u64> {
                $tls.with_borrow(|slot| {
                    slot.as_ref().and_then(|executor| {
                        bonsai_prism_runtime::kernels::Pq2ProjectionExecutor::operation_count(
                            executor, tensor,
                        )
                    })
                })
            }

            fn supports(&self, tensor: &bonsai_prism_runtime::model::Tensor) -> bool {
                $tls.with_borrow(|slot| {
                    slot.as_ref().is_some_and(|executor| {
                        bonsai_prism_runtime::kernels::Pq2ProjectionExecutor::supports(
                            executor, tensor,
                        )
                    })
                })
            }

            fn supports_q8(&self, tensor: &bonsai_prism_runtime::model::Tensor) -> bool {
                $tls.with_borrow(|slot| {
                    slot.as_ref().is_some_and(|executor| {
                        bonsai_prism_runtime::kernels::Pq2ProjectionExecutor::supports_q8(
                            executor, tensor,
                        )
                    })
                })
            }

            fn supports_f32(&self, tensor: &bonsai_prism_runtime::model::Tensor) -> bool {
                $tls.with_borrow(|slot| {
                    slot.as_ref().is_some_and(|executor| {
                        bonsai_prism_runtime::kernels::Pq2ProjectionExecutor::supports_f32(
                            executor, tensor,
                        )
                    })
                })
            }

            fn supports_f32_group(&self, tensors: &[&bonsai_prism_runtime::model::Tensor]) -> bool {
                $tls.with_borrow(|slot| {
                    slot.as_ref().is_some_and(|executor| {
                        bonsai_prism_runtime::kernels::Pq2ProjectionExecutor::supports_f32_group(
                            executor, tensors,
                        )
                    })
                })
            }

            fn supports_tile_tokens(&self) -> bool {
                $tls.with_borrow(|slot| {
                    slot.as_ref().is_some_and(
                        bonsai_prism_runtime::kernels::Pq2ProjectionExecutor::supports_tile_tokens,
                    )
                })
            }

            fn release_host_weight_binding(
                &mut self,
                previous_tensor: &bonsai_prism_runtime::model::Tensor,
                released_data: &bonsai_prism_runtime::model::TensorBytes,
            ) -> bool {
                $tls.with_borrow_mut(|slot| {
                    slot.as_mut().is_some_and(|executor| {
                        bonsai_prism_runtime::kernels::Pq2ProjectionExecutor::release_host_weight_binding(
                            executor,
                            previous_tensor,
                            released_data,
                        )
                    })
                })
            }

            fn has_released_host_weight(
                &self,
                tensor: &bonsai_prism_runtime::model::Tensor,
            ) -> bool {
                $tls.with_borrow(|slot| {
                    slot.as_ref().is_some_and(|executor| {
                        bonsai_prism_runtime::kernels::Pq2ProjectionExecutor::has_released_host_weight(
                            executor, tensor,
                        )
                    })
                })
            }

            fn project(
                &mut self,
                tensor: &bonsai_prism_runtime::model::Tensor,
                activation: &[bonsai_prism_runtime::kernels::Q8KBlock],
                fused: bool,
                output: &mut Vec<f32>,
            ) -> bonsai_prism_runtime::Result<()> {
                $tls.with_borrow_mut(|slot| {
                    let Some(executor) = slot.as_mut() else {
                        output.clear();
                        return Err(std::io::Error::other($missing).into());
                    };
                    bonsai_prism_runtime::kernels::Pq2ProjectionExecutor::project(
                        executor, tensor, activation, fused, output,
                    )
                })
            }

            fn project_q8(
                &mut self,
                tensor: &bonsai_prism_runtime::model::Tensor,
                activation: &[bonsai_prism_runtime::kernels::Q8Block],
                pinned: bool,
                output: &mut Vec<f32>,
            ) -> bonsai_prism_runtime::Result<()> {
                $tls.with_borrow_mut(|slot| {
                    let Some(executor) = slot.as_mut() else {
                        output.clear();
                        return Err(std::io::Error::other($missing).into());
                    };
                    bonsai_prism_runtime::kernels::Pq2ProjectionExecutor::project_q8(
                        executor, tensor, activation, pinned, output,
                    )
                })
            }

            fn project_f32(
                &mut self,
                tensor: &bonsai_prism_runtime::model::Tensor,
                activation: &[f32],
                pinned: bool,
                output: &mut Vec<f32>,
            ) -> bonsai_prism_runtime::Result<()> {
                $tls.with_borrow_mut(|slot| {
                    let Some(executor) = slot.as_mut() else {
                        output.clear();
                        return Err(std::io::Error::other($missing).into());
                    };
                    bonsai_prism_runtime::kernels::Pq2ProjectionExecutor::project_f32(
                        executor, tensor, activation, pinned, output,
                    )
                })
            }

            fn project_f32_group(
                &mut self,
                tensors: &[&bonsai_prism_runtime::model::Tensor],
                activation: &[f32],
                pinned: bool,
                outputs: &mut [&mut Vec<f32>],
            ) -> bonsai_prism_runtime::Result<()> {
                $tls.with_borrow_mut(|slot| {
                    let Some(executor) = slot.as_mut() else {
                        for output in outputs.iter_mut() {
                            output.clear();
                        }
                        return Err(std::io::Error::other($missing).into());
                    };
                    bonsai_prism_runtime::kernels::Pq2ProjectionExecutor::project_f32_group(
                        executor, tensors, activation, pinned, outputs,
                    )
                })
            }

            fn project_f32_tokens(
                &mut self,
                tensor: &bonsai_prism_runtime::model::Tensor,
                activations: &[Vec<f32>],
                pinned: bool,
                outputs: &mut [Vec<f32>],
            ) -> bonsai_prism_runtime::Result<()> {
                $tls.with_borrow_mut(|slot| {
                    let Some(executor) = slot.as_mut() else {
                        for output in outputs.iter_mut() {
                            output.clear();
                        }
                        return Err(std::io::Error::other($missing).into());
                    };
                    bonsai_prism_runtime::kernels::Pq2ProjectionExecutor::project_f32_tokens(
                        executor,
                        tensor,
                        activations,
                        pinned,
                        outputs,
                    )
                })
            }

            fn project_f32_pair_tokens(
                &mut self,
                tensors: [&bonsai_prism_runtime::model::Tensor; 2],
                activations: &[Vec<f32>],
                pinned: bool,
                outputs_a: &mut [Vec<f32>],
                outputs_b: &mut [Vec<f32>],
            ) -> bonsai_prism_runtime::Result<()> {
                $tls.with_borrow_mut(|slot| {
                    let Some(executor) = slot.as_mut() else {
                        for output in outputs_a.iter_mut().chain(outputs_b.iter_mut()) {
                            output.clear();
                        }
                        return Err(std::io::Error::other($missing).into());
                    };
                    bonsai_prism_runtime::kernels::Pq2ProjectionExecutor::project_f32_pair_tokens(
                        executor,
                        tensors,
                        activations,
                        pinned,
                        outputs_a,
                        outputs_b,
                    )
                })
            }

            fn project_f32_triple_tokens(
                &mut self,
                tensors: [&bonsai_prism_runtime::model::Tensor; 3],
                activations: &[Vec<f32>],
                pinned: bool,
                outputs_a: &mut [Vec<f32>],
                outputs_b: &mut [Vec<f32>],
                outputs_c: &mut [Vec<f32>],
            ) -> bonsai_prism_runtime::Result<()> {
                $tls.with_borrow_mut(|slot| {
                    let Some(executor) = slot.as_mut() else {
                        for output in outputs_a
                            .iter_mut()
                            .chain(outputs_b.iter_mut())
                            .chain(outputs_c.iter_mut())
                        {
                            output.clear();
                        }
                        return Err(std::io::Error::other($missing).into());
                    };
                    bonsai_prism_runtime::kernels::Pq2ProjectionExecutor::project_f32_triple_tokens(
                        executor,
                        tensors,
                        activations,
                        pinned,
                        outputs_a,
                        outputs_b,
                        outputs_c,
                    )
                })
            }

            fn supports_f32_ffn_swiglu(
                &self,
                gate: &bonsai_prism_runtime::model::Tensor,
                up: &bonsai_prism_runtime::model::Tensor,
            ) -> bool {
                $tls.with_borrow(|slot| {
                    slot.as_ref().is_some_and(|executor| {
                        bonsai_prism_runtime::kernels::Pq2ProjectionExecutor::supports_f32_ffn_swiglu(
                            executor, gate, up,
                        )
                    })
                })
            }

            fn project_f32_ffn_swiglu(
                &mut self,
                gate: &bonsai_prism_runtime::model::Tensor,
                up: &bonsai_prism_runtime::model::Tensor,
                activation: &[f32],
                pinned: bool,
                output: &mut Vec<f32>,
            ) -> bonsai_prism_runtime::Result<()> {
                $tls.with_borrow_mut(|slot| {
                    let Some(executor) = slot.as_mut() else {
                        output.clear();
                        return Err(std::io::Error::other($missing).into());
                    };
                    bonsai_prism_runtime::kernels::Pq2ProjectionExecutor::project_f32_ffn_swiglu(
                        executor, gate, up, activation, pinned, output,
                    )
                })
            }

            fn project_f32_ffn_swiglu_tokens(
                &mut self,
                gate: &bonsai_prism_runtime::model::Tensor,
                up: &bonsai_prism_runtime::model::Tensor,
                activations: &[Vec<f32>],
                pinned: bool,
                outputs: &mut [Vec<f32>],
            ) -> bonsai_prism_runtime::Result<()> {
                $tls.with_borrow_mut(|slot| {
                    let Some(executor) = slot.as_mut() else {
                        for output in outputs.iter_mut() {
                            output.clear();
                        }
                        return Err(std::io::Error::other($missing).into());
                    };
                    bonsai_prism_runtime::kernels::Pq2ProjectionExecutor::project_f32_ffn_swiglu_tokens(
                        executor,
                        gate,
                        up,
                        activations,
                        pinned,
                        outputs,
                    )
                })
            }

            fn supports_f32_ffn_chain(
                &self,
                tensors: [&bonsai_prism_runtime::model::Tensor; 3],
            ) -> bool {
                $tls.with_borrow(|slot| {
                    slot.as_ref().is_some_and(|executor| {
                        bonsai_prism_runtime::kernels::Pq2ProjectionExecutor::supports_f32_ffn_chain(
                            executor, tensors,
                        )
                    })
                })
            }

            fn project_f32_ffn_chain(
                &mut self,
                tensors: [&bonsai_prism_runtime::model::Tensor; 3],
                activation: &[f32],
                residual: Option<&[f32]>,
                hadamard_block_size: Option<u32>,
                pinned: bool,
                output: &mut Vec<f32>,
            ) -> bonsai_prism_runtime::Result<()> {
                $tls.with_borrow_mut(|slot| {
                    let Some(executor) = slot.as_mut() else {
                        output.clear();
                        return Err(std::io::Error::other($missing).into());
                    };
                    bonsai_prism_runtime::kernels::Pq2ProjectionExecutor::project_f32_ffn_chain(
                        executor,
                        tensors,
                        activation,
                        residual,
                        hadamard_block_size,
                        pinned,
                        output,
                    )
                })
            }

            fn project_f32_ffn_chain_tokens(
                &mut self,
                tensors: [&bonsai_prism_runtime::model::Tensor; 3],
                activations: &[Vec<f32>],
                residuals: Option<&[Vec<f32>]>,
                hadamard_block_size: Option<u32>,
                pinned: bool,
                outputs: &mut [Vec<f32>],
            ) -> bonsai_prism_runtime::Result<()> {
                $tls.with_borrow_mut(|slot| {
                    let Some(executor) = slot.as_mut() else {
                        for output in outputs.iter_mut() {
                            output.clear();
                        }
                        return Err(std::io::Error::other($missing).into());
                    };
                    bonsai_prism_runtime::kernels::Pq2ProjectionExecutor::project_f32_ffn_chain_tokens(
                        executor,
                        tensors,
                        activations,
                        residuals,
                        hadamard_block_size,
                        pinned,
                        outputs,
                    )
                })
            }

            fn supports_f32_ffn_chain_lora(
                &self,
                tensors: [&bonsai_prism_runtime::model::Tensor; 3],
            ) -> bool {
                $tls.with_borrow(|slot| {
                    slot.as_ref().is_some_and(|executor| {
                        bonsai_prism_runtime::kernels::Pq2ProjectionExecutor::supports_f32_ffn_chain_lora(
                            executor, tensors,
                        )
                    })
                })
            }

            fn project_f32_ffn_chain_lora(
                &mut self,
                tensors: [&bonsai_prism_runtime::model::Tensor; 3],
                activation: &[f32],
                residual: Option<&[f32]>,
                hadamard_block_size: Option<u32>,
                lora_a: &[f32],
                pinned: bool,
                output: &mut Vec<f32>,
            ) -> bonsai_prism_runtime::Result<f32> {
                $tls.with_borrow_mut(|slot| {
                    let Some(executor) = slot.as_mut() else {
                        output.clear();
                        return Err(std::io::Error::other($missing).into());
                    };
                    bonsai_prism_runtime::kernels::Pq2ProjectionExecutor::project_f32_ffn_chain_lora(
                        executor,
                        tensors,
                        activation,
                        residual,
                        hadamard_block_size,
                        lora_a,
                        pinned,
                        output,
                    )
                })
            }
        }

        impl Drop for $ty {
            fn drop(&mut self) {
                $tls.with_borrow_mut(|slot| {
                    let _ = slot.take();
                });
            }
        }
    };
}

#[cfg(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"))]
impl_tls_pq2_executor!(
    WorkerThreadHexagonExecutor,
    WORKER_HEXAGON_EXECUTOR,
    "missing worker Hexagon executor",
    |_this| "hexagon-hvx-native-q2-resident-v1-experimental"
);

#[cfg(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"))]
impl_tls_pq2_executor!(
    WorkerThreadCompositeExecutor,
    WORKER_COMPOSITE_EXECUTOR,
    "missing worker composite executor",
    |this| this.identity.as_ref()
);

#[cfg(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"))]
fn has_signed_htp_module(dir: &Path) -> bool {
    dir.join("libggml-htp-v81.so").is_file() || dir.join("libggml-htp-v73.so").is_file()
}

#[cfg(windows)]
#[expect(
    clippy::redundant_pub_crate,
    reason = "Re-exported by native_backend for native_child startup."
)]
#[cfg_attr(
    not(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc")),
    allow(
        clippy::missing_const_for_fn,
        reason = "Non-ARM64 Windows stub matches the filesystem-probing ARM64 signature."
    )
)]
pub(crate) fn register_hexagon_model_path(model: &Path) {
    #[cfg(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"))]
    {
        for ancestor in model.ancestors() {
            for relative in [
                ["hexagon", "runtime"].as_slice(),
                ["hexagon", "signed-q2-package", "runtime"].as_slice(),
            ] {
                let mut candidate = ancestor.to_path_buf();
                for segment in relative {
                    candidate.push(segment);
                }
                if has_signed_htp_module(&candidate) {
                    let _ = HEXAGON_PACKAGE_HINT.set(candidate);
                    return;
                }
            }
        }
    }
    #[cfg(not(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc")))]
    let _ = model;
}

#[cfg(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"))]
fn discover_hexagon_package_directory() -> Option<PathBuf> {
    if let Some(hint) = HEXAGON_PACKAGE_HINT.get()
        && has_signed_htp_module(hint)
    {
        return Some(hint.clone());
    }
    if let Some(local_app_data) = std::env::var_os("LOCALAPPDATA") {
        let candidate = PathBuf::from(local_app_data)
            .join("SwiftBonsai2")
            .join("hexagon")
            .join("runtime");
        if has_signed_htp_module(&candidate) {
            return Some(candidate);
        }
    }
    let staged = std::env::temp_dir()
        .join("Bonsai-Hexagon-SessionNext-20261030")
        .join("signed-q2-package")
        .join("runtime");
    has_signed_htp_module(&staged).then_some(staged)
}

#[cfg(windows)]
const VULKAN_WEIGHT_BUDGET_BYTES: usize = 8 * 1024 * 1024 * 1024;

#[cfg(windows)]
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

#[cfg(windows)]
fn collect_layer_projection_tensors(
    session: &SwiftSession,
    layers: std::ops::Range<usize>,
    include_output: bool,
) -> Vec<&bonsai_prism_runtime::model::Tensor> {
    let mut tensors = Vec::new();
    for layer in layers {
        for suffix in LAYER_PROJECTION_SUFFIXES {
            if let Some(tensor) = session.projection_tensor(&format!("blk.{layer}.{suffix}")) {
                tensors.push(tensor);
            }
        }
    }
    if include_output && let Some(output) = session.projection_tensor("output.weight") {
        tensors.push(output);
    }
    tensors
}

#[cfg(any(
    test,
    all(target_arch = "aarch64", target_os = "windows", target_env = "msvc")
))]
pub(super) const fn supports_hexagon_native_q2_representation(
    representation: bonsai_prism_runtime::model::swift::Representation,
) -> bool {
    match representation {
        bonsai_prism_runtime::model::swift::Representation::Pq2
        // `NativeQ2Executor`, `NativeQ2ArenaCache`, `plan_native_q2_layers_for_budget`,
        // and `supports_f32` runtime dispatch (`src/graph.rs`, `src/kernels/projection.rs`,
        // `src/graph/prefill.rs`, `src/graph/projection.rs`) all admit both `PQ2` (142) and
        // `PTQ1_0` (143), including Ternary Bonsai 2.5 27B.
        | bonsai_prism_runtime::model::swift::Representation::CurrentRepackedPtq1
        | bonsai_prism_runtime::model::swift::Representation::Bonsai25Pq2
        | bonsai_prism_runtime::model::swift::Representation::Bonsai25Ptq1 => true,
        bonsai_prism_runtime::model::swift::Representation::NariQ2_5Bpw
        | bonsai_prism_runtime::model::swift::Representation::NariQ3_0Bpw
        | bonsai_prism_runtime::model::swift::Representation::NariQ2_0Bpw => false,
    }
}

#[cfg(any(
    test,
    all(target_arch = "aarch64", target_os = "windows", target_env = "msvc")
))]
const fn decode_hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(any(
    test,
    all(target_arch = "aarch64", target_os = "windows", target_env = "msvc")
))]
pub(super) fn decode_model_identity_sha256(identity_hex: &str) -> Option<[u8; 32]> {
    if identity_hex.len() != 64 {
        return None;
    }
    let bytes = identity_hex.as_bytes();
    let (chunks, remainder) = bytes.as_chunks::<2>();
    if !remainder.is_empty() {
        return None;
    }
    let mut digest = [0_u8; 32];
    for (slot, chunk) in digest.iter_mut().zip(chunks) {
        let high = decode_hex_nibble(chunk[0])?;
        let low = decode_hex_nibble(chunk[1])?;
        *slot = (high << 4) | low;
    }
    Some(digest)
}

#[cfg(any(
    test,
    all(target_arch = "aarch64", target_os = "windows", target_env = "msvc")
))]
pub(super) fn hexagon_arena_cache_file_path(
    cache_dir: &Path,
    architecture: u32,
    representation: bonsai_prism_runtime::model::swift::Representation,
    model_identity_hex: &str,
    tensor_count: usize,
) -> PathBuf {
    let repr_slug = match representation {
        bonsai_prism_runtime::model::swift::Representation::Pq2 => "pq2",
        bonsai_prism_runtime::model::swift::Representation::CurrentRepackedPtq1 => "ptq1",
        bonsai_prism_runtime::model::swift::Representation::Bonsai25Pq2 => "bonsai25-pq2",
        bonsai_prism_runtime::model::swift::Representation::Bonsai25Ptq1 => "bonsai25-ptq1",
        bonsai_prism_runtime::model::swift::Representation::NariQ2_5Bpw => "nariq-2.5bpw",
        bonsai_prism_runtime::model::swift::Representation::NariQ3_0Bpw => "nariq-3.0bpw",
        bonsai_prism_runtime::model::swift::Representation::NariQ2_0Bpw => "nariq-2.0bpw",
    };
    let safe_identity: String = model_identity_hex
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .collect();
    cache_dir.join(format!(
        "hexagon-v{architecture}-{repr_slug}-{safe_identity}-n{tensor_count}.sb2hq2c1"
    ))
}

#[cfg(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"))]
pub(super) const fn default_or_reported_vtcm_bytes(
    arch: u32,
    reported_vtcm_bytes: Option<u32>,
) -> u32 {
    match reported_vtcm_bytes {
        Some(bytes) if bytes >= 65_536 => bytes,
        _ if arch == 81 => 8 * 1024 * 1024,
        _ => 4 * 1024 * 1024,
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"))]
fn query_driver_vtcm_bytes(driver: &mut hexagon_native_rpc::Driver, arch: u32) -> u32 {
    let page = driver
        .capability(hexagon_native_rpc::Capability::VtcmPage)
        .ok();
    let count = driver
        .capability(hexagon_native_rpc::Capability::VtcmCount)
        .ok();
    default_or_reported_vtcm_bytes(arch, page.zip(count).and_then(|(p, c)| p.checked_mul(c)))
}

#[cfg(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"))]
pub(super) fn mapped_arena_budget_for_capacity(capacity_bytes: usize) -> Option<(usize, usize)> {
    if capacity_bytes == 0 {
        return None;
    }
    let mut remaining = capacity_bytes;
    let mut arena_caps = Vec::with_capacity(hexagon_native_rpc::NATIVE_Q2_MAX_WEIGHT_ARENAS);
    for idx in 0..hexagon_native_rpc::NATIVE_Q2_MAX_WEIGHT_ARENAS {
        if remaining == 0 {
            break;
        }
        let step = if idx.is_multiple_of(2) {
            hexagon_native_rpc::NATIVE_Q2_ARENA_WEIGHT_STEP_BYTES
        } else {
            hexagon_native_rpc::NATIVE_Q2_ARENA_ODD_WEIGHT_STEP_BYTES
        };
        let arena_cap = remaining.min(step);
        arena_caps.push(arena_cap);
        remaining -= arena_cap;
    }
    let full_arenas = arena_caps
        .iter()
        .filter(|&&cap| cap >= hexagon_native_rpc::NATIVE_Q2_ARENA_ODD_WEIGHT_STEP_BYTES)
        .count();
    if full_arenas > 0 {
        let per_arena = if full_arenas == 1 {
            arena_caps[0].min(hexagon_native_rpc::NATIVE_Q2_MAX_ARENA_WEIGHT_BYTES)
        } else {
            hexagon_native_rpc::NATIVE_Q2_ARENA_ODD_WEIGHT_STEP_BYTES
        };
        Some((per_arena, full_arenas))
    } else {
        Some((
            arena_caps[0].min(hexagon_native_rpc::NATIVE_Q2_MAX_ARENA_WEIGHT_BYTES),
            1,
        ))
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"))]
pub(super) fn plan_hexagon_vtcm_placement(
    model: &bonsai_prism_runtime::model::Model,
    arch: u32,
    vtcm_bytes: u32,
    capacity_limit_bytes: Option<usize>,
) -> Option<bonsai_hexagon_runtime::VtcmPlacementPlan> {
    if capacity_limit_bytes
        .is_none_or(|cap| cap >= bonsai_hexagon_runtime::DEFAULT_NATIVE_Q2_RESIDENT_WEIGHT_BYTES)
        && let Ok(plan) = bonsai_hexagon_runtime::plan_native_q2_layers_for_budget(
            model,
            arch,
            hexagon_native_rpc::NATIVE_Q2_MAX_ARENA_WEIGHT_BYTES,
            hexagon_native_rpc::NATIVE_Q2_MAX_WEIGHT_ARENAS,
            vtcm_bytes,
        )
        && capacity_limit_bytes.is_none_or(|cap| plan.resident_bytes <= cap)
    {
        return Some(plan);
    }
    let limit = capacity_limit_bytes?;
    let (max_arena_bytes, max_arenas) = mapped_arena_budget_for_capacity(limit)?;
    let plan = bonsai_hexagon_runtime::plan_native_q2_layers_for_budget(
        model,
        arch,
        max_arena_bytes,
        max_arenas,
        vtcm_bytes,
    )
    .ok()?;
    (plan.resident_bytes <= limit).then_some(plan)
}

#[cfg(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"))]
fn collect_swift_placement_model(
    session: &SwiftSession,
) -> Option<bonsai_prism_runtime::model::Model> {
    let mut tensors = std::collections::BTreeMap::new();
    let mut layer = 0_usize;
    loop {
        let mut found_in_layer = false;
        for suffix in LAYER_PROJECTION_SUFFIXES {
            let name = format!("blk.{layer}.{suffix}");
            if let Some(tensor) = session.projection_tensor(&name) {
                tensors.insert(name, tensor.clone());
                found_in_layer = true;
            }
        }
        if !found_in_layer {
            break;
        }
        layer += 1;
    }
    if tensors.is_empty() {
        return None;
    }
    Some(bonsai_prism_runtime::model::Model {
        metadata: std::collections::BTreeMap::new(),
        tensors,
        identity: session.model_identity().to_owned(),
    })
}

#[cfg(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"))]
fn collect_planned_tensors(
    model: &bonsai_prism_runtime::model::Model,
    plan: &bonsai_hexagon_runtime::VtcmPlacementPlan,
) -> Option<Vec<bonsai_prism_runtime::model::Tensor>> {
    let mut tensors = Vec::new();
    tensors.try_reserve_exact(plan.npu_projections.len()).ok()?;
    for name in &plan.npu_projections {
        tensors.push(model.tensors.get(name)?.clone());
    }
    Some(tensors)
}

#[cfg(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"))]
fn build_native_q2_limits(
    arch: u32,
    resident_weight_bytes: usize,
) -> Option<hexagon_native_rpc::NativeQ2ProjectionLimits> {
    let limits = hexagon_native_rpc::NativeQ2ProjectionLimits::new(
        hexagon_native_rpc::NATIVE_Q2_MAX_COLUMNS,
        hexagon_native_rpc::NATIVE_Q2_MAX_ROWS,
        24_000,
        Duration::from_mins(2),
    )
    .and_then(|limits| limits.with_resident_weights(resident_weight_bytes))
    .map(hexagon_native_rpc::NativeQ2ProjectionLimits::with_persistent_session)
    .ok()?;
    Some(if arch == 73 {
        limits.with_partial_secondary_arenas()
    } else {
        limits
    })
}

#[cfg(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"))]
fn probe_hexagon_hardware() -> Option<(u32, u32)> {
    let mut driver = hexagon_native_rpc::Driver::open_system().ok()?;
    let arch = driver
        .capability(hexagon_native_rpc::Capability::Architecture)
        .ok()
        .and_then(hexagon_native_rpc::decode_architecture)
        .unwrap_or(81);
    let vtcm_bytes = query_driver_vtcm_bytes(&mut driver, arch);
    drop(driver);
    hexagon_native_rpc::is_supported_htp_architecture(arch).then_some((arch, vtcm_bytes))
}

#[cfg(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"))]
fn open_native_q2_executor_once(
    package_dir: &Path,
    limits: hexagon_native_rpc::NativeQ2ProjectionLimits,
    session: &SwiftSession,
) -> Option<(
    bonsai_hexagon_runtime::NativeQ2Executor<hexagon_native_rpc::NativeQ2Session>,
    usize,
)> {
    let q2_session = hexagon_native_rpc::Driver::open_system()
        .ok()?
        .open_native_q2_session(package_dir, limits)
        .ok()?;
    let capacity = hexagon_native_rpc::NativeQ2Submitter::residency_capacity_bytes(&q2_session);
    let executor =
        bonsai_hexagon_runtime::NativeQ2Executor::new(q2_session, limits, 64 * 1024 * 1024)
            .with_row_parallelism(session.clone_row_parallelism())
            .and_then(|executor| {
                executor.with_resident_weights(
                    bonsai_hexagon_runtime::DEFAULT_NATIVE_Q2_RESIDENT_METADATA_BYTES,
                )
            })
            .ok()?;
    Some((executor, capacity))
}

#[cfg(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"))]
fn open_hexagon_native_q2_executor(
    session: &SwiftSession,
    arena_cache_dir: Option<&Path>,
) -> Option<(
    bonsai_hexagon_runtime::NativeQ2Executor<hexagon_native_rpc::NativeQ2Session>,
    Vec<bonsai_prism_runtime::model::Tensor>,
    bonsai_hexagon_runtime::NativeQ2ResidencyStatistics,
    bonsai_hexagon_runtime::VtcmPlacementPlan,
)> {
    if !supports_hexagon_native_q2_representation(session.representation()) {
        return None;
    }
    let package_dir = discover_hexagon_package_directory()?;
    let model = collect_swift_placement_model(session)?;
    let (arch, vtcm_bytes) = probe_hexagon_hardware()?;
    let initial_plan = plan_hexagon_vtcm_placement(&model, arch, vtcm_bytes, None)?;
    let requested_bytes = if initial_plan.resident_bytes
        == bonsai_hexagon_runtime::DEFAULT_NATIVE_Q2_RESIDENT_WEIGHT_BYTES
    {
        bonsai_hexagon_runtime::DEFAULT_NATIVE_Q2_RESIDENT_WEIGHT_BYTES
    } else {
        initial_plan
            .resident_bytes
            .saturating_add(64 * 1024 * 1024)
            .min(bonsai_hexagon_runtime::DEFAULT_NATIVE_Q2_RESIDENT_WEIGHT_BYTES)
    };
    let limits = build_native_q2_limits(arch, requested_bytes)?;
    let (mut executor, mut actual_capacity) =
        open_native_q2_executor_once(&package_dir, limits, session)?;
    let mut plan = if actual_capacity >= initial_plan.resident_bytes {
        initial_plan
    } else {
        plan_hexagon_vtcm_placement(&model, arch, vtcm_bytes, Some(actual_capacity))?
    };
    let mut tensors = collect_planned_tensors(&model, &plan)?;
    let model_sha256 = decode_model_identity_sha256(session.model_identity());
    if let Some(cache_dir) = arena_cache_dir
        && let Some(sha256) = model_sha256
    {
        let cache_path = hexagon_arena_cache_file_path(
            cache_dir,
            arch,
            session.representation(),
            session.model_identity(),
            tensors.len(),
        );
        if cache_path.is_file() {
            if bonsai_hexagon_runtime::NativeQ2ArenaCache::load_and_prewarm_from_verified_cache(
                &mut executor,
                &cache_path,
                sha256,
                &tensors,
            )
            .is_ok()
            {
                let residency = executor.residency_statistics();
                return Some((executor, tensors, residency, plan));
            }
            // Any load failure faults the executor; drop it and reopen cleanly.
            drop(executor);
            let _ = std::fs::remove_file(&cache_path);
            (executor, actual_capacity) =
                open_native_q2_executor_once(&package_dir, limits, session)?;
            if actual_capacity < plan.resident_bytes {
                plan =
                    plan_hexagon_vtcm_placement(&model, arch, vtcm_bytes, Some(actual_capacity))?;
                tensors = collect_planned_tensors(&model, &plan)?;
            }
        }
    }
    if executor.htp_architecture() == 73 {
        executor.select_prefix_and_prewarm_tensors(&tensors).ok()?;
    } else {
        executor.select_and_prewarm_tensors(&tensors).ok()?;
    }
    let residency = executor.residency_statistics();
    if let Some(cache_dir) = arena_cache_dir
        && let Some(sha256) = model_sha256
        && usize::try_from(residency.tensors_prepared).ok() == Some(tensors.len())
        && std::fs::create_dir_all(cache_dir).is_ok()
    {
        let cache_path = hexagon_arena_cache_file_path(
            cache_dir,
            arch,
            session.representation(),
            session.model_identity(),
            tensors.len(),
        );
        let row_parallelism = session.clone_row_parallelism();
        let _ = bonsai_hexagon_runtime::NativeQ2ArenaCache::save_verified_cache(
            &cache_path,
            arch,
            sha256,
            &tensors,
            Some(&row_parallelism),
        );
    }
    Some((executor, tensors, residency, plan))
}

/// Retain host layer weights whenever a CPU prefill or CPU speculation phase
/// may execute fallback projection tiles after installing an accelerator executor.
#[cfg_attr(
    all(
        not(test),
        not(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"))
    ),
    expect(
        dead_code,
        reason = "Host-weight retention check is called on Windows ARM64 worker setup and unit tests."
    )
)]
pub(super) const fn should_release_executor_host_weights(
    has_cpu_prefill: bool,
    has_cpu_speculation: bool,
) -> bool {
    !has_cpu_prefill && !has_cpu_speculation
}

#[cfg(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"))]
fn release_unused_executor_host_weights(session: &mut SwiftSession) {
    if should_release_executor_host_weights(
        session.cpu_prefill().is_some(),
        session.cpu_speculation().is_some(),
    ) {
        let _ = session.release_executor_resident_host_weights();
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"))]
fn install_default_hexagon_executor(
    session: &mut SwiftSession,
    arena_cache_dir: Option<&Path>,
) -> Option<ProjectionStartup> {
    let (executor, _tensors, residency, plan) =
        open_hexagon_native_q2_executor(session, arena_cache_dir)?;
    let startup = if residency.tensors_prepared
        == u64::try_from(bonsai_hexagon_runtime::DEFAULT_NATIVE_Q2_PROJECTIONS.len()).ok()?
        && residency.weight_payload_bytes_prepared
            == u64::try_from(bonsai_hexagon_runtime::DEFAULT_NATIVE_Q2_RESIDENT_WEIGHT_BYTES)
                .ok()?
    {
        ProjectionStartup::NativeQ2 {
            resident_tensors: residency.tensors_prepared,
            resident_weight_bytes: residency.weight_payload_bytes_prepared,
            host_metadata_bytes: residency.metadata_payload_bytes,
        }
    } else {
        ProjectionStartup::NativeQ2Partial {
            architecture: plan.architecture,
            resident_tensors: residency.tensors_prepared,
            resident_weight_bytes: residency.weight_payload_bytes_prepared,
            host_metadata_bytes: residency.metadata_payload_bytes,
        }
    };
    if !startup.valid(Placement::Auto) {
        return None;
    }
    WORKER_HEXAGON_EXECUTOR.with_borrow_mut(|slot| {
        *slot = Some(executor);
    });
    session.set_projection_executor(Some(Box::new(WorkerThreadHexagonExecutor)));
    release_unused_executor_host_weights(session);
    Some(startup)
}

#[cfg_attr(
    not(windows),
    allow(
        clippy::missing_const_for_fn,
        clippy::needless_pass_by_ref_mut,
        reason = "Non-Windows stub matches the Windows Vulkan executor signature."
    )
)]
fn install_vulkan_executor(
    session: &mut SwiftSession,
) -> Result<ProjectionStartup, NativeBackendError> {
    #[cfg(windows)]
    {
        let mut executor =
            bonsai_adreno_vulkan::VulkanPq2Executor::open_default(VULKAN_WEIGHT_BUDGET_BYTES)
                .map_err(|_| NativeBackendError::Configuration)?;
        let tensors = collect_layer_projection_tensors(session, 0..64, true);
        let warmed = executor
            .prewarm_tensors(&tensors)
            .map_err(|_| NativeBackendError::Configuration)?;
        let identity: Arc<str> =
            Arc::from(bonsai_prism_runtime::kernels::Pq2ProjectionExecutor::identity(&executor));
        let resident_tensors = u64::try_from(warmed).map_err(|_| NativeBackendError::Engine)?;
        let resident_weight_bytes = u64::try_from(executor.cached_weight_bytes())
            .map_err(|_| NativeBackendError::Engine)?;
        let startup = ProjectionStartup::AdrenoVulkan {
            identity,
            resident_tensors,
            resident_weight_bytes,
        };
        if !startup.valid(Placement::Vulkan) {
            return Err(NativeBackendError::Configuration);
        }
        session.set_projection_executor(Some(Box::new(executor)));
        Ok(startup)
    }
    #[cfg(not(windows))]
    {
        let _ = session;
        Err(NativeBackendError::Configuration)
    }
}

#[cfg_attr(
    not(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc")),
    allow(
        clippy::missing_const_for_fn,
        clippy::needless_pass_by_ref_mut,
        reason = "Non-ARM64 Windows stub matches the ARM64 composite executor signature."
    )
)]
fn install_composite_executor(
    session: &mut SwiftSession,
    arena_cache_dir: Option<&Path>,
) -> Result<ProjectionStartup, NativeBackendError> {
    #[cfg(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"))]
    {
        let (npu, npu_tensors, npu_residency, vtcm_plan) =
            open_hexagon_native_q2_executor(session, arena_cache_dir)
                .ok_or(NativeBackendError::Configuration)?;
        let total_layers = vtcm_plan.total_layers;
        let mut npu_layers = 0_usize;
        for layer in 0..vtcm_plan.admitted_layers {
            let prefix = format!("blk.{layer}.");
            let mut found_any = false;
            let mut all_supported = true;
            for tensor in &npu_tensors {
                if tensor.name.starts_with(&prefix) {
                    found_any = true;
                    if !bonsai_prism_runtime::kernels::Pq2ProjectionExecutor::supports_f32(
                        &npu, tensor,
                    ) {
                        all_supported = false;
                        break;
                    }
                }
            }
            if found_any && all_supported {
                npu_layers = layer + 1;
            } else {
                break;
            }
        }
        if npu_layers == 0 || npu_layers >= total_layers {
            return Err(NativeBackendError::Configuration);
        }
        let plan = bonsai_hexagon_runtime::plan_three_processor_placement(
            total_layers,
            npu_layers,
            total_layers - npu_layers,
            4,
            bonsai_hexagon_runtime::AcceleratorTarget::AdrenoGpuVulkan,
        )
        .map_err(|_| NativeBackendError::Configuration)?;
        let mut gpu =
            bonsai_adreno_vulkan::VulkanPq2Executor::open_default(VULKAN_WEIGHT_BUDGET_BYTES)
                .map_err(|_| NativeBackendError::Configuration)?;
        let gpu_tensors =
            collect_layer_projection_tensors(session, npu_layers..total_layers, false);
        let gpu_warmed = gpu
            .prewarm_tensors(&gpu_tensors)
            .map_err(|_| NativeBackendError::Configuration)?;
        let gpu_bytes =
            u64::try_from(gpu.cached_weight_bytes()).map_err(|_| NativeBackendError::Engine)?;
        let composite =
            bonsai_hexagon_runtime::OwnedCompositeProjectionExecutor::new(&plan, npu, gpu)
                .map_err(|_| NativeBackendError::Configuration)?;
        let identity: Arc<str> =
            Arc::from(bonsai_prism_runtime::kernels::Pq2ProjectionExecutor::identity(&composite));
        let startup = ProjectionStartup::Composite {
            identity: Arc::clone(&identity),
            npu_resident_tensors: npu_residency.tensors_prepared,
            gpu_resident_tensors: u64::try_from(gpu_warmed)
                .map_err(|_| NativeBackendError::Engine)?,
            resident_weight_bytes: npu_residency
                .weight_payload_bytes_prepared
                .saturating_add(gpu_bytes),
            host_metadata_bytes: npu_residency.metadata_payload_bytes,
        };
        if !startup.valid(Placement::Composite) {
            return Err(NativeBackendError::Configuration);
        }
        WORKER_COMPOSITE_EXECUTOR.with_borrow_mut(|slot| {
            *slot = Some(composite);
        });
        session.set_projection_executor(Some(Box::new(WorkerThreadCompositeExecutor { identity })));
        release_unused_executor_host_weights(session);
        Ok(startup)
    }
    #[cfg(not(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc")))]
    {
        let _ = (session, arena_cache_dir);
        Err(NativeBackendError::Configuration)
    }
}

pub(super) enum Input {
    Count(PendingChat),
    Generate(PendingChat),
    Reset,
}

pub(super) enum Output {
    Count(usize),
    Started {
        prompt: usize,
        streaming: bool,
        thinking: bool,
    },
    Delta {
        text: String,
        confidence: Option<TokenLogprobInfo>,
    },
    Complete(Completion),
    Reset,
    Done(Result<(), NativeBackendError>),
}

pub(super) struct Event {
    pub output: Output,
    pub ack: Option<mpsc::SyncSender<()>>,
}

struct Command {
    input: Input,
    context: CommandContext,
}

struct CommandContext {
    cancel: Cancellation,
    deadline: Option<Instant>,
    events: channel::Sender<Event>,
}

pub(super) struct Worker {
    pub(super) seed: u32,
    pub(super) profile: ModelProfile,
    commands: Option<mpsc::SyncSender<Command>>,
    thread: Option<JoinHandle<()>>,
    busy: Arc<AtomicBool>,
    idle: Arc<Notify>,
    cancellation: Option<Cancellation>,
}

struct Settlement<'a> {
    busy: &'a AtomicBool,
    idle: &'a Notify,
}

impl Drop for Settlement<'_> {
    fn drop(&mut self) {
        self.busy.store(false, Ordering::Release);
        self.idle.notify_waiters();
    }
}

fn valid_graph_profile(profile: &ModelProfile) -> bool {
    if profile.cpu_speculation.is_some() && profile.cpu_prefill.is_none() {
        return false;
    }
    if !(1..=bonsai_prism_runtime::swift_session::MAX_REUSABLE_CHECKPOINTS)
        .contains(&profile.prompt_reuse_checkpoints)
        || (profile.prompt_reuse == bonsai_prism_runtime::runtime::PromptReuse::Disabled
            && profile.prompt_reuse_checkpoints > 1)
    {
        return false;
    }
    if profile.placement == Placement::Vulkan {
        return profile.kv_cache == KvCachePolicy::F16
            && profile.cpu_prefill.is_none()
            && profile.cpu_speculation.is_none()
            && !profile.vision
            && profile
                .startup_graph_identity
                .as_deref()
                .is_none_or(|identity| {
                    !identity.is_empty()
                        && identity.len() <= 4096
                        && identity.bytes().all(|byte| byte.is_ascii_graphic())
                });
    }
    let Some(identity) = profile.startup_graph_identity.as_deref() else {
        return profile.kv_cache == KvCachePolicy::F16 && profile.cpu_prefill.is_none();
    };
    if identity.is_empty() || identity.len() > 4096 {
        return false;
    }
    // Match one complete typed cache identity component, never an arbitrary
    // substring: rsq4 and rsq4k3v must not validate one another or FP16.
    let mut cache_components = identity.split(':').filter(|part| part.starts_with("kv-"));
    let expected_cache = if profile.kv_cache == KvCachePolicy::F16 {
        "kv-f16".to_owned()
    } else {
        format!("kv-{}-d256-rht3-v1", profile.kv_cache.name())
    };
    let cache_matches = cache_components.next() == Some(expected_cache.as_str())
        && cache_components.next().is_none();
    if !cache_matches {
        return false;
    }
    profile.cpu_prefill.map_or_else(
        || {
            profile.kv_cache != KvCachePolicy::F16
                && !identity.contains(":cpu-tiled-")
                && !identity.contains(":gdn-wy-f32-v1")
        },
        |prefill| {
            if prefill.chunkwise_gdn() && profile.cpu_speculation.is_some() {
                return false;
            }
            let mut suffix = format!(":cpu-tiled-ffn-v1:{}", prefill.tokens());
            if prefill.chunkwise_gdn() {
                suffix.push_str(":gdn-wy-f32-v1");
            }
            if profile.cpu_speculation.is_some() {
                suffix.push_str(":cpu-tiled-verify-v1");
            }
            identity.strip_suffix(&suffix).is_some_and(|base| {
                !base.contains(":cpu-tiled-") && !base.contains(":gdn-wy-f32-v1")
            })
        },
    )
}

impl Worker {
    #[cfg(test)]
    pub(super) fn new(engine: impl Engine, seed: u32) -> Result<Self, NativeBackendError> {
        Self::with_placement(engine, seed, Placement::Auto)
    }

    #[cfg(test)]
    pub(super) fn with_placement(
        engine: impl Engine,
        seed: u32,
        placement: Placement,
    ) -> Result<Self, NativeBackendError> {
        Self::with_options(engine, seed, placement, &WorkerOptions::default())
    }

    pub(super) fn with_options(
        engine: impl Engine,
        seed: u32,
        placement: Placement,
        options: &WorkerOptions,
    ) -> Result<Self, NativeBackendError> {
        let mut profile = engine.profile()?;
        profile.placement = placement;
        let Ok(branch_checkpoints) = bonsai_prism_runtime::swift_session::BranchCheckpoints::new(
            profile.prompt_reuse_checkpoints,
        ) else {
            return Err(NativeBackendError::Configuration);
        };
        if !(1..=bonsai_prism_runtime::kernels::MAX_ROW_WORKERS).contains(&profile.row_workers)
            || bonsai_prism_runtime::swift_session::validate_execution_capacity_with_reuse(
                profile.context_tokens,
                u64::MAX,
                profile.cpu_prefill,
                profile.cpu_speculation,
                profile.kv_cache,
                profile.prompt_reuse,
                branch_checkpoints,
            )
            .is_err()
            || !valid_graph_profile(&profile)
        {
            return Err(NativeBackendError::Configuration);
        }
        let (commands, receive) = mpsc::sync_channel::<Command>(1);
        let (initialized, startup) = mpsc::sync_channel(1);
        let busy = Arc::new(AtomicBool::new(false));
        let idle = Arc::new(Notify::new());
        let active = busy.clone();
        let notified = idle.clone();
        let worker_options = options.clone();
        let thread = thread::Builder::new()
            .name("swift-native-model".into())
            .spawn(move || {
                let mut state = State {
                    engine,
                    counted: None,
                    faulted: false,
                };
                // Publish no readiness until the model owner finishes native
                // setup and verifies its complete immutable placement facts.
                let observation =
                    initialize_with_options(&mut state.engine, placement, &worker_options)
                        .and_then(|observation| {
                            if observation.valid(placement) {
                                Ok(observation)
                            } else {
                                Err(NativeBackendError::Configuration)
                            }
                        });
                let succeeded = observation.is_ok();
                if initialized.send(observation).is_err() || !succeeded {
                    return;
                }
                while let Ok(command) = receive.recv() {
                    let _settlement = Settlement {
                        busy: &active,
                        idle: &notified,
                    };
                    let Command { input, context } = command;
                    let result = state.execute(input, &context);
                    let _ = context.events.try_send(Event {
                        output: Output::Done(result),
                        ack: None,
                    });
                }
            })
            .map_err(|_| NativeBackendError::Engine)?;
        profile.projection_startup = match startup.recv() {
            Ok(Ok(observation)) => observation,
            result => {
                // Closing the command lane precedes joining, including an
                // initializer panic or an explicit failed startup observation.
                drop(commands);
                let _ = thread.join();
                return Err(result
                    .ok()
                    .and_then(Result::err)
                    .unwrap_or(NativeBackendError::Engine));
            }
        };
        Ok(Self {
            seed,
            profile,
            commands: Some(commands),
            thread: Some(thread),
            busy,
            idle,
            cancellation: None,
        })
    }

    pub(super) fn start(
        &mut self,
        input: Input,
        cancellation: Cancellation,
        deadline: impl Into<Option<Instant>>,
    ) -> Result<channel::Receiver<Event>, NativeBackendError> {
        let deadline = deadline.into();
        if self.busy.swap(true, Ordering::AcqRel) {
            return Err(NativeBackendError::Engine);
        }
        let (events, receiver) = channel::channel(1);
        self.cancellation = Some(cancellation.clone());
        let command = Command {
            input,
            context: CommandContext {
                cancel: cancellation,
                deadline,
                events,
            },
        };
        if self
            .commands
            .as_ref()
            .ok_or(NativeBackendError::Engine)?
            .try_send(command)
            .is_err()
        {
            self.busy.store(false, Ordering::Release);
            return Err(NativeBackendError::Engine);
        }
        Ok(receiver)
    }

    pub(super) async fn settled(&mut self) {
        loop {
            let notified = self.idle.notified();
            if !self.busy.load(Ordering::Acquire) {
                break;
            }
            notified.await;
        }
        self.cancellation = None;
    }

    pub(super) async fn settled_within(&mut self, timeout: Duration) -> bool {
        if tokio::time::timeout(timeout, self.settled()).await.is_ok() {
            true
        } else {
            if let Some(cancel) = &self.cancellation {
                cancel.cancel();
            }
            false
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        if let Some(cancel) = &self.cancellation {
            cancel.cancel();
        }
        self.commands.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

struct State<E: Engine> {
    engine: E,
    counted: Option<E::Prepared>,
    faulted: bool,
}

impl<E: Engine> State<E> {
    fn execute(
        &mut self,
        input: Input,
        context: &CommandContext,
    ) -> Result<(), NativeBackendError> {
        check(context)?;
        match input {
            Input::Reset => {
                self.faulted = true;
                self.counted = None;
                self.engine
                    .reset()
                    .map_err(|_| NativeBackendError::Engine)?;
                acknowledge(context, Output::Reset)?;
                self.faulted = false;
                Ok(())
            }
            Input::Count(_) if self.faulted => Err(NativeBackendError::Engine),
            Input::Count(pending) => self.count(context, pending),
            Input::Generate(pending) => self.generate(context, &pending),
        }
    }

    fn count(
        &mut self,
        context: &CommandContext,
        pending: PendingChat,
    ) -> Result<(), NativeBackendError> {
        self.counted = None;
        let prepared = self
            .engine
            .prepare_until(pending, &context.cancel, context.deadline)?;
        check(context)?;
        acknowledge(context, Output::Count(E::properties(&prepared).0))?;
        self.counted = Some(prepared);
        Ok(())
    }

    fn generate(
        &mut self,
        context: &CommandContext,
        pending: &PendingChat,
    ) -> Result<(), NativeBackendError> {
        if self.faulted {
            return Err(NativeBackendError::Engine);
        }
        let prepared = self
            .counted
            .take()
            .ok_or(NativeBackendError::CountMismatch)?;
        let (prompt, streaming, thinking, body) = E::properties(&prepared);
        if body != pending.body {
            return Err(NativeBackendError::CountMismatch);
        }
        // Every failure after this point closes admission until an explicit
        // successful reset, including missing write/terminal acknowledgements.
        self.faulted = true;
        acknowledge(
            context,
            Output::Started {
                prompt,
                streaming,
                thinking,
            },
        )?;
        self.engine
            .complete_rich(&prepared, &context.cancel, context.deadline, |event| {
                let output = match event {
                    RichTextEvent::Delta { text, logprobs } => Output::Delta {
                        text: text.to_owned(),
                        confidence: logprobs,
                    },
                    RichTextEvent::Complete(value) => Output::Complete(value),
                };
                acknowledge(context, output).map_err(|_| {
                    std::io::Error::other("native output acknowledgement failed").into()
                })
            })
            .map_err(|_| NativeBackendError::Engine)?;
        check(context)?;
        self.faulted = false;
        Ok(())
    }
}

fn check_deadline_or_cancel(
    cancel: &Cancellation,
    deadline: Option<Instant>,
) -> Result<(), NativeBackendError> {
    if cancel.is_cancelled() {
        return Err(NativeBackendError::ClientAbandoned);
    }
    if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
        return Err(NativeBackendError::Deadline);
    }
    Ok(())
}

fn check(context: &CommandContext) -> Result<(), NativeBackendError> {
    check_deadline_or_cancel(&context.cancel, context.deadline)
}

fn acknowledge(context: &CommandContext, output: Output) -> Result<(), NativeBackendError> {
    check(context)?;
    let (send, receive) = mpsc::sync_channel(1);
    context
        .events
        .blocking_send(Event {
            output,
            ack: Some(send),
        })
        .map_err(|_| NativeBackendError::ClientAbandoned)?;
    loop {
        check(context)?;
        let remaining = context
            .deadline
            .map_or(Ok(Duration::from_millis(50)), |deadline| {
                deadline
                    .checked_duration_since(Instant::now())
                    .ok_or(NativeBackendError::Deadline)
            })?;
        match receive.recv_timeout(remaining.min(Duration::from_millis(50))) {
            Ok(()) => break,
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err(NativeBackendError::ClientAbandoned);
            }
        }
    }
    check(context)
}

#[cfg(test)]
mod tests {
    use super::{Engine, Input, Worker};
    use crate::{AuthorizedHead, PendingChat, native_backend::NativeBackendError};
    use bonsai_prism_runtime::{
        runtime::{Cancellation, Completion, Finish},
        swift_session::TextEvent,
    };
    use std::{
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        thread,
        time::{Duration, Instant},
    };

    type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

    struct StubEngine {
        release: Arc<AtomicBool>,
        cancel_during_prepare: Option<Cancellation>,
    }

    impl Engine for StubEngine {
        type Prepared = PendingChat;

        fn prepare(&self, pending: PendingChat) -> Result<Self::Prepared, NativeBackendError> {
            if let Some(cancel) = &self.cancel_during_prepare {
                cancel.cancel();
            }
            Ok(pending)
        }

        fn properties(prepared: &Self::Prepared) -> (usize, bool, bool, &[u8]) {
            (1, false, false, &prepared.body)
        }

        fn complete(
            &mut self,
            _prepared: &Self::Prepared,
            _cancel: &Cancellation,
            _deadline: Option<Instant>,
            _emit: impl FnMut(TextEvent<'_>) -> bonsai_prism_runtime::Result<()>,
        ) -> bonsai_prism_runtime::Result<Completion> {
            Ok(Completion {
                prompt_tokens: 0,
                reused_tokens: 0,
                generated_tokens: 0,
                finish: Finish::Length,
            })
        }

        fn reset(&mut self) -> bonsai_prism_runtime::Result<()> {
            while !self.release.load(Ordering::Acquire) {
                thread::sleep(Duration::from_millis(2));
            }
            Ok(())
        }
    }

    fn sample_pending() -> Result<PendingChat, String> {
        let body = br#"{"messages":[{"role":"user","content":"hi"}]}"#;
        AuthorizedHead {
            length: body.len(),
            vision: false,
        }
        .prepare(body)
        .map_err(|error| format!("{error:?}"))
    }

    #[test]
    fn prepare_until_enforces_cancellation_and_deadline() -> Result<(), String> {
        let engine = StubEngine {
            release: Arc::new(AtomicBool::new(true)),
            cancel_during_prepare: None,
        };
        let cancel = Cancellation::default();
        assert!(
            engine
                .prepare_until(sample_pending()?, &cancel, None)
                .is_ok()
        );

        let expired = Some(
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .unwrap_or_else(Instant::now),
        );
        assert_eq!(
            engine
                .prepare_until(sample_pending()?, &cancel, expired)
                .err(),
            Some(NativeBackendError::Deadline)
        );

        cancel.cancel();
        assert_eq!(
            engine.prepare_until(sample_pending()?, &cancel, None).err(),
            Some(NativeBackendError::ClientAbandoned)
        );

        let mid_cancel = Cancellation::default();
        let cancelling_engine = StubEngine {
            release: Arc::new(AtomicBool::new(true)),
            cancel_during_prepare: Some(mid_cancel.clone()),
        };
        assert_eq!(
            cancelling_engine
                .prepare_until(sample_pending()?, &mid_cancel, None)
                .err(),
            Some(NativeBackendError::ClientAbandoned)
        );
        Ok(())
    }

    #[test]
    fn worker_settled_within_times_out_when_busy_and_completes_after_release() -> TestResult {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()?;
        runtime.block_on(async {
            let release = Arc::new(AtomicBool::new(false));
            let mut worker = Worker::new(
                StubEngine {
                    release: release.clone(),
                    cancel_during_prepare: None,
                },
                42,
            )
            .map_err(|error| format!("{error:?}"))?;

            assert!(worker.settled_within(Duration::from_millis(20)).await);

            let cancel = Cancellation::default();
            let _events = worker
                .start(Input::Reset, cancel.clone(), None)
                .map_err(|error| format!("{error:?}"))?;
            assert!(!worker.settled_within(Duration::from_millis(15)).await);
            assert!(cancel.is_cancelled());

            release.store(true, Ordering::Release);
            assert!(worker.settled_within(Duration::from_secs(2)).await);
            Ok(())
        })
    }
}
