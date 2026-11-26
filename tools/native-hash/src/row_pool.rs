//! Persistent spin-then-park worker pool with zero per-dispatch heap allocation.

use std::any::Any;
use std::cell::UnsafeCell;
use std::io;
use std::marker::PhantomData;
use std::ops::Deref;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering, fence};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const WORKER_STACK_BYTES: usize = 512 * 1024;
const SPIN_ITERATIONS: u32 = 32_768;
const CHECK_WAIT_INTERVAL: Duration = Duration::from_millis(1);
const SLOT_BITS: u32 = 16;
const SLOT_MASK: u64 = (1_u64 << SLOT_BITS) - 1;
const GEN_MASK: u64 = (1_u64 << (64 - SLOT_BITS)) - 1;

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
// `dispatch_lock` before advancing `dispatch_state` (`SeqCst`), and workers only
// read after observing the new generation (`Acquire`/`SeqCst`) and exit before
// decrementing `pending_workers` (`Release`) prior to `ScopeGuard` exit.
unsafe impl Send for JobSlot {}

// SAFETY: Workers only perform shared reads of `Job` while `pending_workers > 0`.
unsafe impl Sync for JobSlot {}

struct WorkerSleep {
    lock: Mutex<()>,
    cv: Condvar,
    sleeping: AtomicBool,
}

struct Shared {
    dispatch_lock: Mutex<()>,
    done_lock: Mutex<()>,
    done_cv: Condvar,
    dispatch_state: CachePadded<AtomicU64>,
    next_task: CachePadded<AtomicUsize>,
    pending_workers: CachePadded<AtomicUsize>,
    sleeping_workers: CachePadded<AtomicUsize>,
    caller_sleeping: CachePadded<AtomicBool>,
    shutdown: CachePadded<AtomicBool>,
    panicked: CachePadded<AtomicBool>,
    worker_panic: Mutex<Option<Box<dyn Any + Send + 'static>>>,
    worker_sleeps: Box<[CachePadded<WorkerSleep>]>,
    job: CachePadded<JobSlot>,
}

struct WorkerDoneGuard<'a> {
    shared: &'a Shared,
}

impl Drop for WorkerDoneGuard<'_> {
    fn drop(&mut self) {
        if self.shared.pending_workers.fetch_sub(1, Ordering::Release) == 1 {
            fence(Ordering::SeqCst);
            if self.shared.caller_sleeping.load(Ordering::Relaxed) {
                let _guard = self
                    .shared
                    .done_lock
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner);
                self.shared.done_cv.notify_all();
            }
        }
    }
}

struct ScopeGuard<'a, 'env, F> {
    shared: &'a Shared,
    _closure: PhantomData<&'env F>,
}

impl<F> Drop for ScopeGuard<'_, '_, F> {
    fn drop(&mut self) {
        wait_for_workers(self.shared, || {}, false);
        // SAFETY: All background workers for the current generation have exited
        // `Job::invoke` (`pending_workers == 0`), and no new generation can start
        // while `dispatch_lock` remains held by the caller.
        unsafe { *self.shared.job.0.0.get() = None };
    }
}

fn wait_for_workers<W: FnMut()>(shared: &Shared, mut on_wait: W, use_timeout: bool) {
    let mut spins = 0_u32;
    while shared.pending_workers.load(Ordering::Acquire) != 0 {
        if spins < SPIN_ITERATIONS {
            if use_timeout && spins.trailing_zeros() >= 8 {
                on_wait();
            }
            std::hint::spin_loop();
            spins += 1;
            continue;
        }
        if use_timeout {
            on_wait();
            if shared.pending_workers.load(Ordering::Acquire) == 0 {
                break;
            }
        }
        let mut guard = shared
            .done_lock
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        shared.caller_sleeping.store(true, Ordering::SeqCst);
        if use_timeout {
            if shared.pending_workers.load(Ordering::SeqCst) != 0 {
                guard = shared
                    .done_cv
                    .wait_timeout(guard, CHECK_WAIT_INTERVAL)
                    .unwrap_or_else(PoisonError::into_inner)
                    .0;
            }
        } else {
            while shared.pending_workers.load(Ordering::SeqCst) != 0 {
                guard = shared
                    .done_cv
                    .wait(guard)
                    .unwrap_or_else(PoisonError::into_inner);
            }
        }
        shared.caller_sleeping.store(false, Ordering::Release);
        drop(guard);
    }
}

fn take_worker_panic(shared: &Shared) -> Option<Box<dyn Any + Send + 'static>> {
    if shared.panicked.swap(false, Ordering::AcqRel) {
        let mut panic_slot = shared
            .worker_panic
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        return panic_slot.take();
    }
    None
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

/// Persistent caller-participating pool of `prism-row-{idx}` workers using spin-then-park scheduling.
pub struct NativeRowPool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

impl NativeRowPool {
    /// Spawns `worker_count - 1` persistent background worker threads with 512 KiB stacks.
    ///
    /// The calling thread participates directly in task execution, so a
    /// `worker_count` of `1` spawns zero background threads.
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
        if worker_count - 1 > usize::from(u16::MAX) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Row worker pool exceeds supported background worker limit.",
            ));
        }
        let bg_workers = worker_count - 1;
        let mut worker_sleeps = Vec::new();
        if bg_workers > 0 {
            worker_sleeps
                .try_reserve_exact(bg_workers)
                .map_err(|_| io::Error::other("Failed to reserve row worker wake states."))?;
            for _ in 0..bg_workers {
                worker_sleeps.push(CachePadded::new(WorkerSleep {
                    lock: Mutex::new(()),
                    cv: Condvar::new(),
                    sleeping: AtomicBool::new(false),
                }));
            }
        }
        let shared = Arc::new(Shared {
            dispatch_lock: Mutex::new(()),
            done_lock: Mutex::new(()),
            done_cv: Condvar::new(),
            dispatch_state: CachePadded::new(AtomicU64::new(0)),
            next_task: CachePadded::new(AtomicUsize::new(0)),
            pending_workers: CachePadded::new(AtomicUsize::new(0)),
            sleeping_workers: CachePadded::new(AtomicUsize::new(0)),
            caller_sleeping: CachePadded::new(AtomicBool::new(false)),
            shutdown: CachePadded::new(AtomicBool::new(false)),
            panicked: CachePadded::new(AtomicBool::new(false)),
            worker_panic: Mutex::new(None),
            worker_sleeps: worker_sleeps.into_boxed_slice(),
            job: CachePadded::new(JobSlot(UnsafeCell::new(None))),
        });

        let mut workers = Vec::new();
        if bg_workers > 0 {
            workers
                .try_reserve_exact(bg_workers)
                .map_err(|_| io::Error::other("Failed to reserve row worker handles."))?;

            for index in 0..bg_workers {
                let worker_shared = Arc::clone(&shared);
                let spawn_result = thread::Builder::new()
                    .name(format!("prism-row-{index}"))
                    .stack_size(WORKER_STACK_BYTES)
                    .spawn(move || worker_loop(&worker_shared, index));
                match spawn_result {
                    Ok(handle) => workers.push(handle),
                    Err(error) => {
                        shutdown_workers(&shared, &mut workers);
                        return Err(error);
                    }
                }
            }
        }

        Ok(Self { shared, workers })
    }

    /// Returns the total worker concurrency of this pool (calling thread plus background threads).
    #[must_use]
    pub const fn worker_count(&self) -> usize {
        self.workers.len() + 1
    }

    /// Executes `task(0..task_count)` across the worker pool and joins all workers.
    pub fn dispatch<F: Fn(usize) + Sync>(&self, task_count: usize, task: F) {
        self.dispatch_with_wait(task_count, task, || {});
    }

    /// Executes `task(0..task_count)` across the calling thread and background
    /// workers, invoking `on_wait` between caller tasks and while waiting for
    /// background workers.
    pub fn dispatch_with_wait<F: Fn(usize) + Sync, W: FnMut()>(
        &self,
        task_count: usize,
        task: F,
        mut on_wait: W,
    ) {
        if task_count == 0 {
            return;
        }

        let bg_workers = self.workers.len();
        let needed_bg = bg_workers.min(task_count.saturating_sub(1));
        if needed_bg == 0 {
            on_wait();
            for index in 0..task_count {
                task(index);
                on_wait();
            }
            return;
        }

        let dispatch_guard = self
            .shared
            .dispatch_lock
            .lock()
            .unwrap_or_else(PoisonError::into_inner);

        {
            let _ = take_worker_panic(&self.shared);
            let job = Job {
                invoke: invoke_task::<F>,
                ctx: ptr::from_ref(&task).cast::<()>(),
                task_count,
            };
            // SAFETY: `dispatch_lock` guarantees the previous generation exited
            // completely (`pending_workers == 0`), so no worker reads `job` until
            // `dispatch_state` advances below with `SeqCst` release semantics.
            unsafe { *self.shared.job.0.0.get() = Some(job) };
            self.shared.next_task.store(0, Ordering::Release);
            self.shared
                .pending_workers
                .store(needed_bg, Ordering::Release);
            let prev_state = self.shared.dispatch_state.load(Ordering::Relaxed);
            let mut next_gen = ((prev_state >> SLOT_BITS).wrapping_add(1)) & GEN_MASK;
            if next_gen == 0 {
                next_gen = 1;
            }
            let active = u64::try_from(needed_bg).unwrap_or(SLOT_MASK);
            self.shared
                .dispatch_state
                .store((next_gen << SLOT_BITS) | active, Ordering::SeqCst);
            if self.shared.sleeping_workers.load(Ordering::SeqCst) != 0 {
                wake_sleeping_workers(&self.shared.worker_sleeps[..needed_bg]);
            }
        }

        let mut caller_panic = None;
        {
            let _scope_guard = ScopeGuard::<F> {
                shared: &self.shared,
                _closure: PhantomData,
            };

            on_wait();
            loop {
                let index = self.shared.next_task.fetch_add(1, Ordering::Relaxed);
                if index >= task_count {
                    break;
                }
                let result = catch_unwind(AssertUnwindSafe(|| {
                    task(index);
                }));
                if let Err(payload) = result {
                    caller_panic = Some(payload);
                    break;
                }
                on_wait();
            }

            if caller_panic.is_none() {
                wait_for_workers(&self.shared, on_wait, true);
            }
        }

        if let Some(payload) = caller_panic {
            let _ = take_worker_panic(&self.shared);
            drop(dispatch_guard);
            resume_unwind(payload);
        }

        if let Some(payload) = take_worker_panic(&self.shared) {
            drop(dispatch_guard);
            resume_unwind(payload);
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

fn wake_sleeping_workers(sleeps: &[CachePadded<WorkerSleep>]) {
    for ws in sleeps {
        if ws.sleeping.load(Ordering::SeqCst) {
            let _guard = ws.lock.lock().unwrap_or_else(PoisonError::into_inner);
            ws.cv.notify_one();
        }
    }
}

fn shutdown_workers(shared: &Shared, workers: &mut Vec<JoinHandle<()>>) {
    if workers.is_empty() {
        return;
    }
    shared.shutdown.store(true, Ordering::SeqCst);
    if shared.sleeping_workers.load(Ordering::SeqCst) != 0 {
        wake_sleeping_workers(&shared.worker_sleeps);
    }
    for handle in workers.drain(..) {
        let _ = handle.join();
    }
}

fn check_generation(
    dispatch_state: &AtomicU64,
    worker_id: usize,
    seen_generation: &mut u64,
    ordering: Ordering,
) -> bool {
    let current = dispatch_state.load(ordering);
    let generation = current >> SLOT_BITS;
    if generation == *seen_generation {
        return false;
    }
    *seen_generation = generation;
    let active = usize::try_from(current & SLOT_MASK).unwrap_or(usize::MAX);
    worker_id < active
}

fn worker_loop(shared: &Shared, worker_id: usize) {
    let mut seen_generation = 0_u64;
    let ws = &shared.worker_sleeps[worker_id];
    loop {
        let mut spins = 0_u32;
        loop {
            if shared.shutdown.load(Ordering::Acquire) {
                return;
            }
            if check_generation(
                &shared.dispatch_state,
                worker_id,
                &mut seen_generation,
                Ordering::Acquire,
            ) {
                break;
            }
            if spins < SPIN_ITERATIONS {
                std::hint::spin_loop();
                spins += 1;
                continue;
            }
            let mut guard = ws.lock.lock().unwrap_or_else(PoisonError::into_inner);
            ws.sleeping.store(true, Ordering::SeqCst);
            shared.sleeping_workers.fetch_add(1, Ordering::SeqCst);
            loop {
                if shared.shutdown.load(Ordering::SeqCst) {
                    ws.sleeping.store(false, Ordering::Release);
                    shared.sleeping_workers.fetch_sub(1, Ordering::Release);
                    drop(guard);
                    return;
                }
                if check_generation(
                    &shared.dispatch_state,
                    worker_id,
                    &mut seen_generation,
                    Ordering::SeqCst,
                ) {
                    break;
                }
                guard = ws.cv.wait(guard).unwrap_or_else(PoisonError::into_inner);
            }
            ws.sleeping.store(false, Ordering::Release);
            shared.sleeping_workers.fetch_sub(1, Ordering::Release);
            drop(guard);
            break;
        }

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

        let serial = NativeRowPool::new(1)?;
        assert_eq!(serial.worker_count(), 1);
        assert!(serial.workers.is_empty());

        let pool = NativeRowPool::new(4)?;
        assert_eq!(pool.worker_count(), 4);
        assert_eq!(pool.workers.len(), 3);

        let caller_id = thread::current().id();
        let mut items = vec![0_u64; 32];
        let mut ids = vec![caller_id; 32];
        let mut names = vec![String::new(); 32];
        let mut slots: Vec<_> = items
            .iter_mut()
            .zip(ids.iter_mut().zip(names.iter_mut()))
            .collect();
        pool.dispatch_mut(&mut slots, |index, (value, (id, name))| {
            **value = u64::try_from(index * 7 + 3).unwrap_or(0);
            let cur = thread::current();
            **id = cur.id();
            **name = cur.name().unwrap_or("").to_owned();
        });
        for (index, ((value, id), name)) in
            items.iter().zip(ids.iter()).zip(names.iter()).enumerate()
        {
            assert_eq!(*value, u64::try_from(index * 7 + 3).unwrap_or(0));
            assert!(
                *id == caller_id || name.starts_with("prism-row-"),
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

        // Wait until all 3 background workers finish their spin window and enter idle sleep.
        let mut waited_ms = 0_u32;
        while pool.shared.sleeping_workers.load(Ordering::Acquire) < 3 && waited_ms < 500 {
            thread::sleep(Duration::from_millis(1));
            waited_ms += 1;
        }
        assert_eq!(pool.shared.sleeping_workers.load(Ordering::Acquire), 3);

        // Wake all 3 sleeping background workers plus caller across a 4-way barrier.
        // Hold a second barrier after reading `sleeping_workers` so no worker can
        // finish `task` and re-enter the spin->sleep path before all 4 threads check.
        let wake_barrier = std::sync::Barrier::new(4);
        let check_barrier = std::sync::Barrier::new(4);
        let awake_seen = AtomicUsize::new(0);
        pool.dispatch(4, |_| {
            wake_barrier.wait();
            if pool.shared.sleeping_workers.load(Ordering::Acquire) == 0 {
                awake_seen.fetch_add(1, Ordering::Relaxed);
            }
            check_barrier.wait();
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

    #[test]
    fn stress_workers_and_task_counts_cover_caller_and_background_execution() -> io::Result<()> {
        let caller_id = thread::current().id();
        for workers in [1_usize, 2, 4, 12, 18] {
            let pool = NativeRowPool::new(workers)?;
            assert_eq!(pool.worker_count(), workers);
            assert_eq!(pool.workers.len(), workers - 1);

            for task_count in [0_usize, 1, 2, 7, 18, 256] {
                let counts: Vec<AtomicUsize> =
                    (0..task_count).map(|_| AtomicUsize::new(0)).collect();
                let wait_ticks = AtomicUsize::new(0);
                let bad_caller = AtomicBool::new(false);
                pool.dispatch_with_wait(
                    task_count,
                    |index| {
                        counts[index].fetch_add(1, Ordering::Relaxed);
                        let cur = thread::current();
                        let allowed = cur.id() == caller_id
                            || (workers > 1
                                && cur.name().is_some_and(|n| n.starts_with("prism-row-")));
                        assert!(allowed);
                    },
                    || {
                        if thread::current().id() != caller_id {
                            bad_caller.store(true, Ordering::Release);
                        }
                        wait_ticks.fetch_add(1, Ordering::Relaxed);
                    },
                );
                assert!(!bad_caller.load(Ordering::Acquire));
                for count in &counts {
                    assert_eq!(count.load(Ordering::Relaxed), 1);
                }
                if task_count == 0 {
                    assert_eq!(wait_ticks.load(Ordering::Relaxed), 0);
                } else {
                    assert!(wait_ticks.load(Ordering::Relaxed) >= 1);
                    if workers == 1 {
                        assert_eq!(wait_ticks.load(Ordering::Relaxed), task_count + 1);
                    }
                }

                let mut slots = vec![0_usize; task_count];
                pool.dispatch_mut(&mut slots, |index, slot| {
                    *slot = index.wrapping_mul(31).wrapping_add(7);
                });
                for (index, &val) in slots.iter().enumerate() {
                    assert_eq!(val, index.wrapping_mul(31).wrapping_add(7));
                }
            }
        }
        Ok(())
    }

    #[test]
    #[expect(
        clippy::panic,
        reason = "Injects caller, worker, and on_wait panics to verify recovery and cleanup."
    )]
    fn caller_worker_and_wait_panics_propagate_and_recover() -> io::Result<()> {
        let caller_id = thread::current().id();
        for workers in [1_usize, 2, 4, 18] {
            let pool = NativeRowPool::new(workers)?;

            // Caller thread panic propagation and recovery.
            let caller_panicked = catch_unwind(AssertUnwindSafe(|| {
                pool.dispatch(workers * 4, |_| {
                    assert!(thread::current().id() != caller_id, "injected caller panic");
                    thread::sleep(Duration::from_millis(1));
                });
            }));
            assert!(caller_panicked.is_err());

            if workers > 1 {
                // Background worker panic propagation and recovery.
                let worker_panicked = AtomicBool::new(false);
                let res = catch_unwind(AssertUnwindSafe(|| {
                    pool.dispatch(workers * 4, |_| {
                        if thread::current().id() == caller_id {
                            thread::sleep(Duration::from_millis(2));
                        } else {
                            assert!(
                                worker_panicked.swap(true, Ordering::AcqRel),
                                "injected worker panic"
                            );
                        }
                    });
                }));
                assert!(res.is_err());
            }

            if workers == 4 {
                // Simultaneous caller and worker panic propagation and recovery.
                let worker_panicked = AtomicBool::new(false);
                let res = catch_unwind(AssertUnwindSafe(|| {
                    pool.dispatch(16, |_| {
                        assert!(
                            thread::current().id() != caller_id,
                            "injected simultaneous caller panic"
                        );
                        assert!(
                            worker_panicked.swap(true, Ordering::AcqRel),
                            "injected simultaneous worker panic"
                        );
                    });
                }));
                assert!(res.is_err());
            }

            // `on_wait` panic propagation and recovery.
            let wait_panicked = catch_unwind(AssertUnwindSafe(|| {
                pool.dispatch_with_wait(
                    workers * 2,
                    |_| {
                        thread::sleep(Duration::from_millis(1));
                    },
                    || {
                        panic!("injected wait callback panic");
                    },
                );
            }));
            assert!(wait_panicked.is_err());

            let mut recovered = vec![0_usize; workers * 3];
            pool.dispatch_mut(&mut recovered, |index, slot| {
                *slot = index + 11;
            });
            for (index, &val) in recovered.iter().enumerate() {
                assert_eq!(val, index + 11);
            }
        }
        Ok(())
    }

    #[test]
    fn repeated_sleep_wake_transitions_across_worker_counts() -> io::Result<()> {
        for workers in [1_usize, 2, 4, 12, 18] {
            let pool = NativeRowPool::new(workers)?;
            let bg_workers = workers - 1;
            for round in 0..4_usize {
                let mut waited_ms = 0_u32;
                while pool.shared.sleeping_workers.load(Ordering::Acquire) < bg_workers
                    && waited_ms < 500
                {
                    thread::sleep(Duration::from_millis(1));
                    waited_ms += 1;
                }
                assert_eq!(
                    pool.shared.sleeping_workers.load(Ordering::Acquire),
                    bg_workers
                );

                let barrier = std::sync::Barrier::new(workers);
                let seen = AtomicUsize::new(0);
                pool.dispatch(workers, |_| {
                    barrier.wait();
                    seen.fetch_add(1, Ordering::Relaxed);
                });
                assert_eq!(seen.load(Ordering::Relaxed), workers, "round={round}");

                let mut slots = vec![0_usize; 32];
                pool.dispatch_mut(&mut slots, |idx, slot| {
                    *slot = round.wrapping_mul(100).wrapping_add(idx);
                });
                for (idx, &val) in slots.iter().enumerate() {
                    assert_eq!(val, round.wrapping_mul(100).wrapping_add(idx));
                }
            }
        }
        Ok(())
    }

    #[test]
    fn small_dispatch_keeps_excess_prefill_workers_parked() -> io::Result<()> {
        let pool = NativeRowPool::new(16)?;
        let bg_workers = 15;
        let mut waited_ms = 0_u32;
        while pool.shared.sleeping_workers.load(Ordering::Acquire) < bg_workers && waited_ms < 500 {
            thread::sleep(Duration::from_millis(1));
            waited_ms += 1;
        }
        assert_eq!(pool.shared.sleeping_workers.load(Ordering::Acquire), 15);

        // Single-task dispatch executes directly on caller without waking any parked worker.
        let mut single = [0_usize; 1];
        pool.dispatch_mut(&mut single, |idx, slot| {
            *slot = idx + 99;
        });
        assert_eq!(single, [99]);
        assert_eq!(pool.shared.sleeping_workers.load(Ordering::Acquire), 15);

        // An 8-task dispatch wakes only 7 background workers (+ caller), leaving >= 8 parked.
        let barrier = std::sync::Barrier::new(8);
        let min_sleeping = AtomicUsize::new(usize::MAX);
        let mut decode_slots = [0_usize; 8];
        pool.dispatch_mut(&mut decode_slots, |idx, slot| {
            barrier.wait();
            let sleeping = pool.shared.sleeping_workers.load(Ordering::Acquire);
            min_sleeping.fetch_min(sleeping, Ordering::Relaxed);
            barrier.wait();
            *slot = idx + 1;
        });
        assert_eq!(decode_slots, [1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(min_sleeping.load(Ordering::Relaxed) >= 8);

        // Subsequent 16-task prefill dispatch still engages all 16 threads cleanly.
        let mut prefill_slots = [0_usize; 16];
        let prefill_barrier = std::sync::Barrier::new(16);
        pool.dispatch_mut(&mut prefill_slots, |idx, slot| {
            prefill_barrier.wait();
            *slot = idx + 10;
        });
        for (idx, &val) in prefill_slots.iter().enumerate() {
            assert_eq!(val, idx + 10);
        }
        Ok(())
    }
}
