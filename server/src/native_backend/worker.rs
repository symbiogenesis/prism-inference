//! A single joined model owner and acknowledged one-event output lane.

use super::{NativeBackendError, PendingChat, profile::ModelProfile};
use bonsai_prism_runtime::{
    graph::KvCachePolicy,
    runtime::{Cancellation, Completion},
    swift_session::{SwiftSession, TextEvent},
};
use std::{
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
    fn properties(prepared: &Self::Prepared) -> (usize, bool, bool, &[u8]);
    fn complete(
        &mut self,
        prepared: &Self::Prepared,
        cancel: &Cancellation,
        deadline: Instant,
        emit: impl FnMut(TextEvent<'_>) -> bonsai_prism_runtime::Result<()>,
    ) -> bonsai_prism_runtime::Result<Completion>;
    fn reset(&mut self) -> bonsai_prism_runtime::Result<()>;
    fn initialize_worker_thread(&mut self) {}
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
            representation: self.representation(),
            adapter: self.adapter_identity().is_some(),
            prompt_reuse: self.prompt_reuse(),
            row_workers: self.row_workers(),
            context_tokens: self.context_tokens(),
            kv_cache: self.kv_cache_policy(),
            cpu_prefill: self.cpu_prefill(),
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
        deadline: Instant,
        emit: impl FnMut(TextEvent<'_>) -> bonsai_prism_runtime::Result<()>,
    ) -> bonsai_prism_runtime::Result<Completion> {
        crate::native_swift::complete(self, prepared, cancel, deadline, emit)
    }
    fn reset(&mut self) -> bonsai_prism_runtime::Result<()> {
        Self::reset(self)
    }
    fn initialize_worker_thread(&mut self) {
        #[cfg(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"))]
        install_default_hexagon_executor(self);
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"))]
static HEXAGON_PACKAGE_HINT: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();

#[cfg(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"))]
thread_local! {
    static WORKER_HEXAGON_EXECUTOR: std::cell::RefCell<
        Option<bonsai_hexagon_runtime::NativeQ2Executor<hexagon_native_rpc::NativeQ2Session>>,
    > = const { std::cell::RefCell::new(None) };
}

#[cfg(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"))]
struct WorkerThreadHexagonExecutor;

#[cfg(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"))]
impl bonsai_prism_runtime::kernels::Pq2ProjectionExecutor for WorkerThreadHexagonExecutor {
    fn identity(&self) -> &'static str {
        "hexagon-hvx-native-q2-resident-v1-experimental"
    }

    fn operation_capacity(&self) -> Option<u64> {
        WORKER_HEXAGON_EXECUTOR.with_borrow(|slot| {
            slot.as_ref()
                .and_then(bonsai_prism_runtime::kernels::Pq2ProjectionExecutor::operation_capacity)
        })
    }

    fn operation_count(&self, tensor: &bonsai_prism_runtime::model::Tensor) -> Option<u64> {
        WORKER_HEXAGON_EXECUTOR.with_borrow(|slot| {
            slot.as_ref()
                .and_then(|executor| executor.operation_count(tensor))
        })
    }

    fn supports(&self, tensor: &bonsai_prism_runtime::model::Tensor) -> bool {
        WORKER_HEXAGON_EXECUTOR.with_borrow(|slot| {
            slot.as_ref()
                .is_some_and(|executor| executor.supports(tensor))
        })
    }

    fn project(
        &mut self,
        tensor: &bonsai_prism_runtime::model::Tensor,
        activation: &[bonsai_prism_runtime::kernels::Q8KBlock],
        fused: bool,
        output: &mut Vec<f32>,
    ) -> bonsai_prism_runtime::Result<()> {
        WORKER_HEXAGON_EXECUTOR.with_borrow_mut(|slot| {
            let Some(executor) = slot.as_mut() else {
                output.clear();
                return Err(std::io::Error::other("missing worker Hexagon executor").into());
            };
            executor.project(tensor, activation, fused, output)
        })
    }

    fn supports_f32(&self, tensor: &bonsai_prism_runtime::model::Tensor) -> bool {
        WORKER_HEXAGON_EXECUTOR.with_borrow(|slot| {
            slot.as_ref()
                .is_some_and(|executor| executor.supports_f32(tensor))
        })
    }

    fn project_f32(
        &mut self,
        tensor: &bonsai_prism_runtime::model::Tensor,
        activation: &[f32],
        pinned: bool,
        output: &mut Vec<f32>,
    ) -> bonsai_prism_runtime::Result<()> {
        WORKER_HEXAGON_EXECUTOR.with_borrow_mut(|slot| {
            let Some(executor) = slot.as_mut() else {
                output.clear();
                return Err(std::io::Error::other("missing worker Hexagon executor").into());
            };
            executor.project_f32(tensor, activation, pinned, output)
        })
    }

    fn supports_f32_group(&self, tensors: &[&bonsai_prism_runtime::model::Tensor]) -> bool {
        WORKER_HEXAGON_EXECUTOR.with_borrow(|slot| {
            slot.as_ref()
                .is_some_and(|executor| executor.supports_f32_group(tensors))
        })
    }

    fn project_f32_group(
        &mut self,
        tensors: &[&bonsai_prism_runtime::model::Tensor],
        activation: &[f32],
        pinned: bool,
        outputs: &mut [&mut Vec<f32>],
    ) -> bonsai_prism_runtime::Result<()> {
        WORKER_HEXAGON_EXECUTOR.with_borrow_mut(|slot| {
            let Some(executor) = slot.as_mut() else {
                for output in outputs.iter_mut() {
                    output.clear();
                }
                return Err(std::io::Error::other("missing worker Hexagon executor").into());
            };
            executor.project_f32_group(tensors, activation, pinned, outputs)
        })
    }

    fn project_f32_tokens(
        &mut self,
        tensor: &bonsai_prism_runtime::model::Tensor,
        activations: &[Vec<f32>],
        pinned: bool,
        outputs: &mut [Vec<f32>],
    ) -> bonsai_prism_runtime::Result<()> {
        WORKER_HEXAGON_EXECUTOR.with_borrow_mut(|slot| {
            let Some(executor) = slot.as_mut() else {
                for output in outputs.iter_mut() {
                    output.clear();
                }
                return Err(std::io::Error::other("missing worker Hexagon executor").into());
            };
            executor.project_f32_tokens(tensor, activations, pinned, outputs)
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
        WORKER_HEXAGON_EXECUTOR.with_borrow_mut(|slot| {
            let Some(executor) = slot.as_mut() else {
                for output in outputs_a.iter_mut() {
                    output.clear();
                }
                for output in outputs_b.iter_mut() {
                    output.clear();
                }
                return Err(std::io::Error::other("missing worker Hexagon executor").into());
            };
            executor.project_f32_pair_tokens(tensors, activations, pinned, outputs_a, outputs_b)
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
        WORKER_HEXAGON_EXECUTOR.with_borrow_mut(|slot| {
            let Some(executor) = slot.as_mut() else {
                for output in outputs_a.iter_mut() {
                    output.clear();
                }
                for output in outputs_b.iter_mut() {
                    output.clear();
                }
                for output in outputs_c.iter_mut() {
                    output.clear();
                }
                return Err(std::io::Error::other("missing worker Hexagon executor").into());
            };
            executor.project_f32_triple_tokens(
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
        WORKER_HEXAGON_EXECUTOR.with_borrow(|slot| {
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
        WORKER_HEXAGON_EXECUTOR.with_borrow_mut(|slot| {
            let Some(executor) = slot.as_mut() else {
                output.clear();
                return Err(std::io::Error::other("missing worker Hexagon executor").into());
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
        WORKER_HEXAGON_EXECUTOR.with_borrow_mut(|slot| {
            let Some(executor) = slot.as_mut() else {
                for output in outputs.iter_mut() {
                    output.clear();
                }
                return Err(std::io::Error::other("missing worker Hexagon executor").into());
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
}

#[cfg(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"))]
impl Drop for WorkerThreadHexagonExecutor {
    fn drop(&mut self) {
        WORKER_HEXAGON_EXECUTOR.with_borrow_mut(|slot| {
            let _ = slot.take();
        });
    }
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
pub(crate) fn register_hexagon_model_path(model: &std::path::Path) {
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
                if candidate.join("libggml-htp-v81.so").is_file() {
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
fn discover_hexagon_package_directory() -> Option<std::path::PathBuf> {
    if let Some(hint) = HEXAGON_PACKAGE_HINT.get()
        && hint.join("libggml-htp-v81.so").is_file()
    {
        return Some(hint.clone());
    }
    if let Some(local_app_data) = std::env::var_os("LOCALAPPDATA") {
        let candidate = std::path::PathBuf::from(local_app_data)
            .join("SwiftBonsai2")
            .join("hexagon")
            .join("runtime");
        if candidate.join("libggml-htp-v81.so").is_file() {
            return Some(candidate);
        }
    }
    let staged = std::env::temp_dir()
        .join("Bonsai-Hexagon-SessionNext-20261030")
        .join("signed-q2-package")
        .join("runtime");
    staged
        .join("libggml-htp-v81.so")
        .is_file()
        .then_some(staged)
}

#[cfg(all(target_arch = "aarch64", target_os = "windows", target_env = "msvc"))]
fn install_default_hexagon_executor(session: &mut SwiftSession) {
    if session.representation() != bonsai_prism_runtime::model::swift::Representation::Pq2
        || session.cpu_prefill().is_some()
    {
        return;
    }
    let Some(package_dir) = discover_hexagon_package_directory() else {
        return;
    };
    let Ok(limits) = hexagon_native_rpc::NativeQ2ProjectionLimits::new(
        hexagon_native_rpc::NATIVE_Q2_MAX_COLUMNS,
        hexagon_native_rpc::NATIVE_Q2_MAX_ROWS,
        24_000,
        Duration::from_mins(2),
    )
    .and_then(|limits| {
        limits
            .with_resident_weights(bonsai_hexagon_runtime::DEFAULT_NATIVE_Q2_RESIDENT_WEIGHT_BYTES)
    })
    .map(hexagon_native_rpc::NativeQ2ProjectionLimits::with_persistent_session) else {
        return;
    };
    let Ok(driver) = hexagon_native_rpc::Driver::open_system() else {
        return;
    };
    let Ok(q2_session) = driver.open_native_q2_session(&package_dir, limits) else {
        return;
    };
    let Ok(mut executor) =
        bonsai_hexagon_runtime::NativeQ2Executor::new(q2_session, limits, 64 * 1024 * 1024)
            .with_row_parallelism(session.clone_row_parallelism())
            .and_then(|executor| {
                executor.with_resident_weights(
                    bonsai_hexagon_runtime::DEFAULT_NATIVE_Q2_RESIDENT_METADATA_BYTES,
                )
            })
    else {
        return;
    };
    let mut tensors =
        Vec::with_capacity(bonsai_hexagon_runtime::DEFAULT_NATIVE_Q2_PROJECTIONS.len());
    for &name in &bonsai_hexagon_runtime::DEFAULT_NATIVE_Q2_PROJECTIONS {
        let Some(tensor) = session.projection_tensor(name) else {
            return;
        };
        tensors.push(tensor.clone());
    }
    if executor.select_and_prewarm_tensors(&tensors).is_ok() {
        WORKER_HEXAGON_EXECUTOR.with_borrow_mut(|slot| {
            *slot = Some(executor);
        });
        session.set_projection_executor(Some(Box::new(WorkerThreadHexagonExecutor)));
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
    Delta(String),
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
    deadline: Instant,
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
    let Some(identity) = profile.startup_graph_identity.as_deref() else {
        return profile.kv_cache == KvCachePolicy::F16 && profile.cpu_prefill.is_none();
    };
    if identity.is_empty() || identity.len() > 4096 {
        return false;
    }
    if let Some(prefill) = profile.cpu_prefill {
        profile.kv_cache == KvCachePolicy::F16
            && identity.ends_with(&format!(":cpu-tiled-ffn-v1:{}", prefill.tokens()))
    } else {
        profile.kv_cache != KvCachePolicy::F16
    }
}

impl Worker {
    pub(super) fn new(engine: impl Engine, seed: u32) -> Result<Self, NativeBackendError> {
        let profile = engine.profile()?;
        if !(1..=bonsai_prism_runtime::kernels::MAX_ROW_WORKERS).contains(&profile.row_workers)
            || !valid_graph_profile(&profile)
        {
            return Err(NativeBackendError::Configuration);
        }
        let (commands, receive) = mpsc::sync_channel::<Command>(1);
        let busy = Arc::new(AtomicBool::new(false));
        let idle = Arc::new(Notify::new());
        let active = busy.clone();
        let notified = idle.clone();
        let thread = thread::Builder::new()
            .name("swift-native-model".into())
            .spawn(move || {
                let mut state = State {
                    engine,
                    counted: None,
                    faulted: false,
                };
                state.engine.initialize_worker_thread();
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
        deadline: Instant,
    ) -> Result<channel::Receiver<Event>, NativeBackendError> {
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
        let prepared = self.engine.prepare(pending)?;
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
            .complete(&prepared, &context.cancel, context.deadline, |event| {
                let output = match event {
                    TextEvent::Delta(text) => Output::Delta(text.to_owned()),
                    TextEvent::Complete(value) => Output::Complete(value),
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

fn check(context: &CommandContext) -> Result<(), NativeBackendError> {
    if context.cancel.is_cancelled() {
        return Err(NativeBackendError::ClientAbandoned);
    }
    if Instant::now() >= context.deadline {
        return Err(NativeBackendError::Deadline);
    }
    Ok(())
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
            .checked_duration_since(Instant::now())
            .ok_or(NativeBackendError::Deadline)?;
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
