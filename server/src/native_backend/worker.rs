//! A single joined model owner and acknowledged one-event output lane.

use super::{NativeBackendError, PendingChat};
use bonsai_prism_runtime::{
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
    fn prepare(&self, pending: PendingChat) -> Result<Self::Prepared, NativeBackendError>;
    fn properties(prepared: &Self::Prepared) -> (usize, bool, &[u8]);
    fn complete(
        &mut self,
        prepared: &Self::Prepared,
        cancel: &Cancellation,
        deadline: Instant,
        emit: impl FnMut(TextEvent<'_>) -> bonsai_prism_runtime::Result<()>,
    ) -> bonsai_prism_runtime::Result<Completion>;
    fn reset(&mut self) -> bonsai_prism_runtime::Result<()>;
}

impl Engine for SwiftSession {
    type Prepared = crate::native_swift::NativeChat;
    fn prepare(&self, pending: PendingChat) -> Result<Self::Prepared, NativeBackendError> {
        crate::native_swift::prepare(self, pending).map_err(NativeBackendError::Admission)
    }
    fn properties(prepared: &Self::Prepared) -> (usize, bool, &[u8]) {
        (
            prepared.prompt_tokens(),
            prepared.admitted().streams_response(),
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
}

pub(super) enum Input {
    Count(PendingChat),
    Generate(PendingChat),
    Reset,
}

pub(super) enum Output {
    Count(usize),
    Started { prompt: usize, streaming: bool },
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
    cancel: Cancellation,
    deadline: Instant,
    events: channel::Sender<Event>,
}

pub(super) struct Worker {
    pub(super) seed: u32,
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

impl Worker {
    pub(super) fn new(engine: impl Engine, seed: u32) -> Result<Self, NativeBackendError> {
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
                while let Ok(command) = receive.recv() {
                    let _settlement = Settlement {
                        busy: &active,
                        idle: &notified,
                    };
                    let result = state.execute(&command);
                    let _ = command.events.try_send(Event {
                        output: Output::Done(result),
                        ack: None,
                    });
                }
            })
            .map_err(|_| NativeBackendError::Engine)?;
        Ok(Self {
            seed,
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
            cancel: cancellation,
            deadline,
            events,
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
    fn execute(&mut self, command: &Command) -> Result<(), NativeBackendError> {
        check(command)?;
        match &command.input {
            Input::Reset => {
                self.faulted = true;
                self.counted = None;
                self.engine
                    .reset()
                    .map_err(|_| NativeBackendError::Engine)?;
                acknowledge(command, Output::Reset)?;
                self.faulted = false;
                Ok(())
            }
            Input::Count(_) if self.faulted => Err(NativeBackendError::Engine),
            Input::Count(_) => self.count(command),
            Input::Generate(pending) => self.generate(command, pending),
        }
    }

    fn count(&mut self, command: &Command) -> Result<(), NativeBackendError> {
        self.counted = None;
        let Input::Count(pending) = &command.input else {
            return Err(NativeBackendError::Engine);
        };
        let prepared = self.engine.prepare(PendingChat {
            body: pending.body.clone(),
            max_completion_tokens: pending.max_completion_tokens,
            stream: pending.stream,
        })?;
        check(command)?;
        acknowledge(command, Output::Count(E::properties(&prepared).0))?;
        self.counted = Some(prepared);
        Ok(())
    }

    fn generate(
        &mut self,
        command: &Command,
        pending: &PendingChat,
    ) -> Result<(), NativeBackendError> {
        if self.faulted {
            return Err(NativeBackendError::Engine);
        }
        let prepared = self
            .counted
            .take()
            .ok_or(NativeBackendError::CountMismatch)?;
        let (prompt, streaming, body) = E::properties(&prepared);
        if body != pending.body {
            return Err(NativeBackendError::CountMismatch);
        }
        // Every failure after this point closes admission until an explicit
        // successful reset, including missing write/terminal acknowledgements.
        self.faulted = true;
        acknowledge(command, Output::Started { prompt, streaming })?;
        self.engine
            .complete(&prepared, &command.cancel, command.deadline, |event| {
                let output = match event {
                    TextEvent::Delta(text) => Output::Delta(text.to_owned()),
                    TextEvent::Complete(value) => Output::Complete(value),
                };
                acknowledge(command, output).map_err(|_| {
                    std::io::Error::other("native output acknowledgement failed").into()
                })
            })
            .map_err(|_| NativeBackendError::Engine)?;
        check(command)?;
        self.faulted = false;
        Ok(())
    }
}

fn check(command: &Command) -> Result<(), NativeBackendError> {
    if command.cancel.is_cancelled() {
        return Err(NativeBackendError::ClientAbandoned);
    }
    if Instant::now() >= command.deadline {
        return Err(NativeBackendError::Deadline);
    }
    Ok(())
}

fn acknowledge(command: &Command, output: Output) -> Result<(), NativeBackendError> {
    check(command)?;
    let (send, receive) = mpsc::sync_channel(1);
    command
        .events
        .blocking_send(Event {
            output,
            ack: Some(send),
        })
        .map_err(|_| NativeBackendError::ClientAbandoned)?;
    loop {
        check(command)?;
        let remaining = command
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
    check(command)
}
