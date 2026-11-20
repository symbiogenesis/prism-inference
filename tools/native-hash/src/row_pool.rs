//! Persistent spin-then-park worker pool with zero per-dispatch heap allocation.

use std::any::Any;
use std::cell::UnsafeCell;
use std::io;
use std::marker::PhantomData;
use std::ops::Deref;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const WORKER_STACK_BYTES: usize = 512 * 1024;
const SPIN_ITERATIONS: u32 = 32_768;
const CHECK_WAIT_INTERVAL: Duration = Duration::from_millis(1);

#[repr(align(64))]
struct CachePadded<T>(T);

impl<T> CachePadded<T> {
    const fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T> Deref for CachePadded<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

type TaskInvoke = unsafe fn(*const (), usize);

#[derive(Clone, Copy)]
struct Job {
    invoke: TaskInvoke,
    ctx: *const (),
    task_count: usize,
}

// SAFETY: `Job` carries a `Sync` closure pointer whose target lives for the
// synchronous duration of `dispatch_with_wait` while callers hold `dispatch_lock`.
unsafe impl Send for Job {}

// SAFETY: `Job` exposes shared immutable dispatch metadata until workers finish.
unsafe impl Sync for Job {}

struct JobSlot(UnsafeCell<Option<Job>>);

// SAFETY: Access to `JobSlot` is strictly serialized: the caller writes under
// `dispatch_lock` + `state_lock` before advancing `generation` (`Release`), and
// workers only read after observing the new `generation` (`Acquire`) and exit
// before decrementing `pending_workers` (`Release`) prior to `ScopeGuard` exit.
unsafe impl Send for JobSlot {}

// SAFETY: Workers only perform shared reads of `Job` while `pending_workers > 0`.
unsafe impl Sync for JobSlot {}

struct Shared {
    dispatch_lock: Mutex<()>,
    state_lock: Mutex<()>,
    wake_workers: Condvar,
    done_lock: Mutex<()>,
    done_cv: Condvar,
    generation: CachePadded<AtomicU64>,
    next_task: CachePadded<AtomicUsize>,
    pending_workers: CachePadded<AtomicUsize>,
    sleeping_workers: CachePadded<AtomicUsize>,
    caller_sleeping: CachePadded<AtomicBool>,
    shutdown: CachePadded<AtomicBool>,
    panicked: CachePadded<AtomicBool>,
    worker_panic: Mutex<Option<Box<dyn Any + Send + 'static>>>,
    job: CachePadded<JobSlot>,
}

struct WorkerDoneGuard<'a> {
    shared: &'a Shared,
}

impl Drop for WorkerDoneGuard<'_> {
    fn drop(&mut self) {
        if self.shared.pending_workers.fetch_sub(1, Ordering::SeqCst) == 1
            && self.shared.caller_sleeping.load(Ordering::SeqCst)
        {
            let _guard = self
                .shared
                .done_lock
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            self.shared.done_cv.notify_all();
        }
    }
}

struct ScopeGuard<'a, 'env, F> {
    shared: &'a Shared,
    _closure: PhantomData<&'env F>,
}

impl<F> Drop for ScopeGuard<'_, '_, F> {
    fn drop(&mut self) {
        wait_until_workers_idle(self.shared);
        // SAFETY: All workers for the current generation have exited `Job::invoke`
        // (`pending_workers == 0`), and no new generation can start while `dispatch_lock`
        // remains held by the caller.
        unsafe { *self.shared.job.0.0.get() = None };
    }
}

fn wait_until_workers_idle(shared: &Shared) {
    let mut spins = 0_u32;
    while shared.pending_workers.load(Ordering::Acquire) != 0 {
        if spins < SPIN_ITERATIONS {
            std::hint::spin_loop();
            spins += 1;
            continue;
        }
        let mut guard = shared
            .done_lock
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        shared.caller_sleeping.store(true, Ordering::SeqCst);
        while shared.pending_workers.load(Ordering::SeqCst) != 0 {
            guard = shared
                .done_cv
                .wait(guard)
                .unwrap_or_else(PoisonError::into_inner);
        }
        shared.caller_sleeping.store(false, Ordering::Release);
        drop(guard);
    }
}

unsafe fn invoke_task<F: Fn(usize) + Sync>(ctx: *const (), task_index: usize) {
    // SAFETY: Caller's `ScopeGuard` keeps the referenced closure alive and pinned
    // on the calling stack until every worker thread finishes this generation.
    let task = unsafe { &*ctx.cast::<F>() };
    task(task_index);
}

struct DisjointSlice<T> {
    base: NonNull<T>,
    len: usize,
}

impl<T> DisjointSlice<T> {
    const fn new(slice: &mut [T]) -> Self {
        let len = slice.len();
        // SAFETY: `slice.as_mut_ptr()` from a live slice reference is always non-null.
        let base = unsafe { NonNull::new_unchecked(slice.as_mut_ptr()) };
        Self { base, len }
    }

    fn with_slot_mut(&self, index: usize, visit: impl FnOnce(&mut T)) {
        if index >= self.len {
            return;
        }
        // SAFETY: `index < self.len`, so `base + index` stays within the caller's slice.
        let elem = unsafe { self.base.as_ptr().add(index) };
        // SAFETY: Each `index` is claimed by at most one worker from `next_task`,
        // `T: Send`, and `ScopeGuard` ensures `slice` outlives every worker access.
        let slot = unsafe { &mut *elem };
        visit(slot);
    }
}

// SAFETY: Disjoint indices yield non-overlapping `&mut T` elements and `T: Send`.
unsafe impl<T: Send> Sync for DisjointSlice<T> {}

/// Persistent pool of `prism-row-{idx}` workers using spin-then-park scheduling.
pub struct NativeRowPool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

impl NativeRowPool {
    /// Spawns `worker_count` persistent worker threads with 512 KiB stacks.
    ///
    /// # Errors
    /// Rejects a zero worker count or propagates any OS thread creation failure
    /// after shutting down any workers already created.
    pub fn new(worker_count: usize) -> io::Result<Self> {
        if worker_count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Row worker pool requires at least one thread.",
            ));
        }
        let shared = Arc::new(Shared {
            dispatch_lock: Mutex::new(()),
            state_lock: Mutex::new(()),
            wake_workers: Condvar::new(),
            done_lock: Mutex::new(()),
            done_cv: Condvar::new(),
            generation: CachePadded::new(AtomicU64::new(0)),
            next_task: CachePadded::new(AtomicUsize::new(0)),
            pending_workers: CachePadded::new(AtomicUsize::new(0)),
            sleeping_workers: CachePadded::new(AtomicUsize::new(0)),
            caller_sleeping: CachePadded::new(AtomicBool::new(false)),
            shutdown: CachePadded::new(AtomicBool::new(false)),
            panicked: CachePadded::new(AtomicBool::new(false)),
            worker_panic: Mutex::new(None),
            job: CachePadded::new(JobSlot(UnsafeCell::new(None))),
        });

        let mut workers = Vec::new();
        workers
            .try_reserve_exact(worker_count)
            .map_err(|_| io::Error::other("Failed to reserve row worker handles."))?;

        for index in 0..worker_count {
            let worker_shared = Arc::clone(&shared);
            let spawn_result = thread::Builder::new()
                .name(format!("prism-row-{index}"))
                .stack_size(WORKER_STACK_BYTES)
                .spawn(move || worker_loop(&worker_shared));
            match spawn_result {
                Ok(handle) => workers.push(handle),
                Err(error) => {
                    shutdown_workers(&shared, &mut workers);
                    return Err(error);
                }
            }
        }

        Ok(Self { shared, workers })
    }

    /// Returns the fixed number of persistent worker threads in this pool.
    #[must_use]
    pub const fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Executes `task(0..task_count)` across the worker pool and joins all workers.
    pub fn dispatch<F: Fn(usize) + Sync>(&self, task_count: usize, task: F) {
        self.dispatch_with_wait(task_count, task, || {});
    }

    /// Executes `task(0..task_count)` across the worker pool, invoking `on_wait`
    /// while workers remain active.
    pub fn dispatch_with_wait<F: Fn(usize) + Sync, W: FnMut()>(
        &self,
        task_count: usize,
        task: F,
        mut on_wait: W,
    ) {
        if task_count == 0 {
            return;
        }

        let dispatch_guard = self
            .shared
            .dispatch_lock
            .lock()
            .unwrap_or_else(PoisonError::into_inner);

        let worker_count = self.workers.len();
        {
            if self.shared.panicked.swap(false, Ordering::AcqRel) {
                let mut panic_slot = self
                    .shared
                    .worker_panic
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner);
                *panic_slot = None;
            }
            let job = Job {
                invoke: invoke_task::<F>,
                ctx: ptr::from_ref(&task).cast::<()>(),
                task_count,
            };
            // SAFETY: `dispatch_lock` guarantees the previous generation exited
            // completely (`pending_workers == 0`), so no worker reads `job` until
            // `generation` advances below with `SeqCst` release semantics.
            unsafe { *self.shared.job.0.0.get() = Some(job) };
            self.shared.next_task.store(0, Ordering::Release);
            self.shared
                .pending_workers
                .store(worker_count, Ordering::Release);
            self.shared.generation.fetch_add(1, Ordering::SeqCst);
            if self.shared.sleeping_workers.load(Ordering::SeqCst) != 0 {
                let _state_guard = self
                    .shared
                    .state_lock
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner);
                self.shared.wake_workers.notify_all();
            }
        }

        {
            let _scope_guard = ScopeGuard::<F> {
                shared: &self.shared,
                _closure: PhantomData,
            };

            let mut spins = 0_u32;
            while self.shared.pending_workers.load(Ordering::Acquire) != 0 {
                if spins < SPIN_ITERATIONS {
                    if spins.trailing_zeros() >= 8 {
                        on_wait();
                    }
                    std::hint::spin_loop();
                    spins += 1;
                    continue;
                }
                on_wait();
                if self.shared.pending_workers.load(Ordering::Acquire) == 0 {
                    break;
                }
                let guard = self
                    .shared
                    .done_lock
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner);
                self.shared.caller_sleeping.store(true, Ordering::SeqCst);
                let guard = if self.shared.pending_workers.load(Ordering::SeqCst) != 0 {
                    self.shared
                        .done_cv
                        .wait_timeout(guard, CHECK_WAIT_INTERVAL)
                        .unwrap_or_else(PoisonError::into_inner)
                        .0
                } else {
                    guard
                };
                self.shared.caller_sleeping.store(false, Ordering::Release);
                drop(guard);
            }
        }

        if self.shared.panicked.swap(false, Ordering::AcqRel) {
            let mut panic_slot = self
                .shared
                .worker_panic
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            let payload = panic_slot.take();
            drop(panic_slot);
            if let Some(payload) = payload {
                drop(dispatch_guard);
                resume_unwind(payload);
            }
        }
    }

    /// Executes `task(index, slot)` over every element of `slots` with zero heap
    /// allocation, invoking `on_wait` while waiting for workers.
    pub fn dispatch_mut_with_wait<T: Send, F: Fn(usize, &mut T) + Sync, W: FnMut()>(
        &self,
        slots: &mut [T],
        task: F,
        on_wait: W,
    ) {
        let count = slots.len();
        if count == 0 {
            return;
        }
        let slice = DisjointSlice::new(slots);
        self.dispatch_with_wait(
            count,
            |index| {
                slice.with_slot_mut(index, |slot| task(index, slot));
            },
            on_wait,
        );
    }

    /// Executes `task(index, slot)` over every element of `slots` and joins all workers.
    pub fn dispatch_mut<T: Send, F: Fn(usize, &mut T) + Sync>(&self, slots: &mut [T], task: F) {
        self.dispatch_mut_with_wait(slots, task, || {});
    }
}

impl Drop for NativeRowPool {
    fn drop(&mut self) {
        shutdown_workers(&self.shared, &mut self.workers);
    }
}

fn shutdown_workers(shared: &Shared, workers: &mut Vec<JoinHandle<()>>) {
    shared.shutdown.store(true, Ordering::SeqCst);
    if shared.sleeping_workers.load(Ordering::SeqCst) != 0 {
        let _guard = shared
            .state_lock
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        shared.wake_workers.notify_all();
    }
    for handle in workers.drain(..) {
        let _ = handle.join();
    }
}

fn worker_loop(shared: &Shared) {
    let mut seen_generation = 0_u64;
    loop {
        let mut spins = 0_u32;
        let next_generation = loop {
            if shared.shutdown.load(Ordering::Acquire) {
                return;
            }
            let current = shared.generation.load(Ordering::Acquire);
            if current != seen_generation {
                break current;
            }
            if spins < SPIN_ITERATIONS {
                std::hint::spin_loop();
                spins += 1;
                continue;
            }
            let mut guard = shared
                .state_lock
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            shared.sleeping_workers.fetch_add(1, Ordering::SeqCst);
            let current = loop {
                if shared.shutdown.load(Ordering::SeqCst) {
                    shared.sleeping_workers.fetch_sub(1, Ordering::Release);
                    drop(guard);
                    return;
                }
                let current = shared.generation.load(Ordering::SeqCst);
                if current != seen_generation {
                    break current;
                }
                guard = shared
                    .wake_workers
                    .wait(guard)
                    .unwrap_or_else(PoisonError::into_inner);
            };
            shared.sleeping_workers.fetch_sub(1, Ordering::Release);
            drop(guard);
            break current;
        };
        seen_generation = next_generation;

        let _done = WorkerDoneGuard { shared };
        // SAFETY: Caller published `job` under `dispatch_lock` before advancing
        // `generation` with `SeqCst` release semantics, and `ScopeGuard` waits for
        // `_done` to decrement `pending_workers` before modifying `job`.
        let maybe_job = unsafe { *shared.job.0.0.get() };
        if let Some(job) = maybe_job {
            loop {
                let index = shared.next_task.fetch_add(1, Ordering::Relaxed);
                if index >= job.task_count {
                    break;
                }
                let result = catch_unwind(AssertUnwindSafe(|| {
                    // SAFETY: `index < job.task_count`, `job.ctx` outlives this call,
                    // and `job.invoke` matches `job.ctx`'s monomorphized closure type.
                    unsafe { (job.invoke)(job.ctx, index) };
                }));
                if let Err(payload) = result {
                    let mut panic_slot = shared
                        .worker_panic
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner);
                    if panic_slot.is_none() {
                        *panic_slot = Some(payload);
                    }
                    drop(panic_slot);
                    shared.panicked.store(true, Ordering::Release);
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_workers_and_preserves_names_and_disjoint_updates() -> io::Result<()> {
        assert!(NativeRowPool::new(0).is_err());

        let pool = NativeRowPool::new(4)?;
        assert_eq!(pool.worker_count(), 4);

        let mut items = vec![0_u64; 32];
        let mut names = vec![String::new(); 32];
        let mut slots: Vec<_> = items.iter_mut().zip(names.iter_mut()).collect();
        pool.dispatch_mut(&mut slots, |index, (value, name)| {
            **value = u64::try_from(index * 7 + 3).unwrap_or(0);
            **name = thread::current().name().unwrap_or("").to_owned();
        });
        for (index, (value, name)) in items.iter().zip(names.iter()).enumerate() {
            assert_eq!(*value, u64::try_from(index * 7 + 3).unwrap_or(0));
            assert!(
                name.starts_with("prism-row-"),
                "unexpected worker thread name: {name}"
            );
        }

        let wait_ticks = AtomicUsize::new(0);
        pool.dispatch_with_wait(
            8,
            |_| {
                thread::sleep(Duration::from_millis(2));
            },
            || {
                wait_ticks.fetch_add(1, Ordering::Relaxed);
            },
        );
        assert!(wait_ticks.load(Ordering::Relaxed) >= 1);

        let panicked = catch_unwind(AssertUnwindSafe(|| {
            pool.dispatch(2, |index| {
                assert!(index != 1, "injected row pool panic");
            });
        }));
        assert!(panicked.is_err());

        let mut recovered = [0_usize; 4];
        pool.dispatch_mut(&mut recovered, |index, slot| {
            *slot = index + 1;
        });
        assert_eq!(recovered, [1, 2, 3, 4]);
        Ok(())
    }

    #[test]
    fn rapid_back_to_back_and_idle_sleep_wake_transitions() -> io::Result<()> {
        let pool = NativeRowPool::new(4)?;
        let mut slots = [0_usize; 16];

        // Exercise lockless rapid back-to-back spin-phase dispatches.
        for round in 1..=2_000_usize {
            pool.dispatch_mut(&mut slots, |index, slot| {
                *slot = round.wrapping_add(index);
            });
            for (index, value) in slots.iter().enumerate() {
                assert_eq!(*value, round.wrapping_add(index));
            }
        }

        // Wait until all workers finish their spin window and enter idle sleep.
        let mut waited_ms = 0_u32;
        while pool.shared.sleeping_workers.load(Ordering::Acquire) < 4 && waited_ms < 500 {
            thread::sleep(Duration::from_millis(1));
            waited_ms += 1;
        }
        assert_eq!(pool.shared.sleeping_workers.load(Ordering::Acquire), 4);

        // Wake all four sleeping workers simultaneously across a 4-way barrier.
        let barrier = std::sync::Barrier::new(4);
        let awake_seen = AtomicUsize::new(0);
        pool.dispatch(4, |_| {
            barrier.wait();
            if pool.shared.sleeping_workers.load(Ordering::Acquire) == 0 {
                awake_seen.fetch_add(1, Ordering::Relaxed);
            }
        });
        assert_eq!(awake_seen.load(Ordering::Relaxed), 4);

        for round in 1..=32_usize {
            pool.dispatch_mut(&mut slots, |index, slot| {
                *slot = round.wrapping_mul(17).wrapping_add(index);
            });
            for (index, value) in slots.iter().enumerate() {
                assert_eq!(*value, round.wrapping_mul(17).wrapping_add(index));
            }
        }
        Ok(())
    }
}
