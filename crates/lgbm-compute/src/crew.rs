//! A spin-waiting persistent worker crew for sub-100µs parallel regions.
//!
//! ## Why not rayon
//!
//! The tree-grow hot loop dispatches a parallel region every ~100-200µs (per-leaf
//! histogram fold, per-leaf split scan, sibling subtract). A rayon fork/join costs
//! ~30µs on an M1 even with warm workers (measured 2026-08-06: 50×0.4µs tasks =
//! 19µs serial, 32µs under `par_iter`), because idle rayon workers sleep on a
//! condvar and must be woken through the kernel (`__psynch_cvwait` dominated the
//! 255-leaf profile). OpenMP — what the C++ reference uses — keeps its workers
//! SPINNING between regions, so its barrier is ~2µs and LightGBM scales at high
//! leaf counts where the port did not (1.19× vs 2.23× at `num_leaves=255`).
//!
//! This module is the OpenMP-shaped answer: N persistent threads spin-wait for
//! work, claim task indices off a shared atomic counter, and go back to
//! spinning. After [`spin_us`] microseconds without work they park (so a long
//! serial phase or an idle process costs nothing); the dispatcher unparks them
//! on the next region.
//!
//! ## Bit-exactness contract
//!
//! The crew only RUNS closures; it never decides work decomposition. Callers keep
//! the invariant that tasks own disjoint outputs and each task's internal
//! sequential order is unchanged from the serial path — the same contract the
//! existing rayon arms document. Every task index in `0..n_tasks` is claimed
//! EXACTLY ONCE, on any thread, so results are identical regardless of crew size
//! or scheduling; a crew of 1 (or a disabled / contended crew) degrades to the
//! caller's thread running every task in ascending index order — the serial path.
//!
//! ## Concurrency protocol (why the `unsafe` below is sound)
//!
//! The claim counter and the region epoch live in ONE `AtomicU64`
//! (`claim = epoch << COUNTER_BITS | task_index`), so a claim is epoch-tagged by
//! construction — there is no window in which a stale `fetch_add` can claim an
//! index of a region whose counter it did not observe (the classic ABA of a
//! plain reset counter).
//!
//! - One dispatcher at a time: `dispatch_lock` is held for the whole region; a
//!   re-entrant or cross-thread second dispatch runs serial (`try_lock` failure)
//!   instead of deadlocking.
//! - Publication order (dispatcher): job cell (plain write) → `n_packed`
//!   `(epoch, n_tasks)` (Release) → `tasks_done = 0` (Relaxed) → `claim_ctr =
//!   (epoch, 0)` (Release) → unpark. A claimant's `fetch_add` (Acquire) on
//!   `claim_ctr` synchronizes with that final Release store (release-sequence
//!   through intervening RMWs), so an in-range claim sees the matching job and
//!   `n_packed` writes.
//! - Range check: a claim `(ce, t)` reads `n_packed = (ne, n)`. `ne < ce` is
//!   impossible (the dispatcher wrote `n_packed` before the counter it read
//!   from). `ne > ce` proves the claimed region ALREADY drained — every
//!   in-range claim of epoch `ce` must execute before the dispatcher can
//!   publish a later epoch (the drain waits on `tasks_done == n`), so a claim
//!   observing a newer `n_packed` cannot be in-range; it returns without
//!   touching the job. `ne == ce` compares `t < n` normally.
//! - Job-cell reads happen ONLY after an in-range claim. An in-range claim
//!   blocks the region's drain until it executes (its `tasks_done` increment is
//!   required), and the dispatcher only rewrites the job cell after the drain —
//!   so the read can never race a republication.
//! - A panicking task sets `poisoned` and still counts itself done (scope
//!   guard), so the dispatcher never hangs; it re-panics on the dispatching
//!   thread after the drain.
//!
//! Epoch wrap: 40 epoch bits ≈ 10¹² regions; counter creep from out-of-range
//! claims is bounded by ~one per worker per region, far below the 2²⁴ counter
//! space. Both are unreachable in any real train.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::thread::{self, Thread};

/// Low bits of the packed word hold the task index; high bits the epoch.
const COUNTER_BITS: u32 = 24;
const COUNTER_MASK: u64 = (1 << COUNTER_BITS) - 1;
/// Maximum tasks per region (fits the counter field with headroom for creep).
pub const MAX_TASKS: usize = (COUNTER_MASK as usize) / 2;

/// Type-erased job: `call(data, task_idx)` runs one task. `data` points at the
/// dispatcher's stack-borrowed closure; validity is guaranteed by the
/// completion protocol above.
#[derive(Clone, Copy)]
struct RawJob {
    data: *const (),
    call: unsafe fn(*const (), usize),
}

/// The `Sync` wrapper for the job cell. Safety: see the module-level protocol.
struct JobCell(UnsafeCell<RawJob>);
// SAFETY: writes only by the lock-holding dispatcher between drains; reads only
// after an in-range claim (module-level protocol).
unsafe impl Sync for JobCell {}

struct Shared {
    /// Packed `(epoch << COUNTER_BITS) | next_task` claim counter.
    claim_ctr: AtomicU64,
    /// Packed `(epoch << COUNTER_BITS) | n_tasks` for the same epoch.
    n_packed: AtomicU64,
    /// Completed-task count for the current region.
    tasks_done: AtomicUsize,
    /// Set when a task panicked; the dispatcher re-panics after the drain.
    poisoned: AtomicBool,
    /// Crew shutdown.
    shutdown: AtomicBool,
    /// The current job.
    job: JobCell,
    /// Per-worker parked flags + thread handles for unparking.
    parked: Vec<AtomicBool>,
    threads: Mutex<Vec<Thread>>,
}

pub struct Crew {
    shared: &'static Shared,
    dispatch_lock: Mutex<u64>, // current epoch, guarded by the dispatch lock
    n_workers: usize,
}

/// Spin budget before a worker parks, in microseconds. Regions arrive every
/// ~100-200µs in the grow loop, so 50µs of spinning bridges the common gap while
/// a worker still parks quickly during long serial phases (binning, prediction).
fn spin_us() -> u64 {
    static V: OnceLock<u64> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("LGBM_CREW_SPIN_US")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(50)
    })
}

/// Number of PERFORMANCE cores on Apple Silicon (`hw.perflevel0.physicalcpu`),
/// `None` elsewhere or on error. Spinning workers scheduled onto E-cores
/// measurably regress the crew (255-leaf bench: 8 threads 4267ms vs 4 threads
/// 3903ms on an M1 4P+4E) — the same shape as the C++ reference, whose best
/// OpenMP thread count on this box is also the P-core count.
fn performance_cores() -> Option<usize> {
    #[cfg(target_os = "macos")]
    {
        let mut val: u32 = 0;
        let mut len = std::mem::size_of::<u32>();
        let name = c"hw.perflevel0.physicalcpu";
        // SAFETY: standard sysctlbyname out-param call; `len` matches `val`.
        let rc = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                std::ptr::addr_of_mut!(val).cast(),
                &raw mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc == 0 && val >= 1 {
            return Some(val as usize);
        }
        None
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

/// Crew thread count. Default: the rayon pool size (honors `RAYON_NUM_THREADS`,
/// so the 1-thread determinism gate exercises the serial arm), capped at the
/// P-core count on Apple Silicon (see [`performance_cores`]). `LGBM_CREW_THREADS`
/// overrides; `LGBM_CREW=0` disables the crew entirely (every region runs serial
/// on the caller).
fn crew_threads() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        if std::env::var("LGBM_CREW").is_ok_and(|s| s == "0") {
            return 1;
        }
        if let Some(n) = std::env::var("LGBM_CREW_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
        {
            return n;
        }
        let rayon_n = rayon::current_num_threads().max(1);
        performance_cores().map_or(rayon_n, |p| p.min(rayon_n))
    })
}

impl Crew {
    fn new(n_workers: usize) -> Crew {
        let shared: &'static Shared = Box::leak(Box::new(Shared {
            claim_ctr: AtomicU64::new(0),
            n_packed: AtomicU64::new(0),
            tasks_done: AtomicUsize::new(0),
            poisoned: AtomicBool::new(false),
            shutdown: AtomicBool::new(false),
            job: JobCell(UnsafeCell::new(RawJob {
                data: std::ptr::null(),
                call: |_, _| unreachable!("job executed before first dispatch"),
            })),
            parked: (0..n_workers.saturating_sub(1)).map(|_| AtomicBool::new(false)).collect(),
            threads: Mutex::new(Vec::new()),
        }));
        // n_workers includes the dispatching thread; spawn n_workers-1 helpers.
        let mut handles = Vec::new();
        for wid in 0..n_workers.saturating_sub(1) {
            let b = thread::Builder::new().name(format!("lgbm-crew-{wid}"));
            let h = b
                .spawn(move || worker_loop(shared, wid))
                .expect("crew worker spawn");
            handles.push(h.thread().clone());
        }
        *shared.threads.lock().expect("crew threads lock") = handles;
        Crew {
            shared,
            dispatch_lock: Mutex::new(0),
            n_workers,
        }
    }

    /// The process-wide crew, created on first use.
    pub fn global() -> &'static Crew {
        static CREW: OnceLock<Crew> = OnceLock::new();
        CREW.get_or_init(|| Crew::new(crew_threads()))
    }

    /// Whether dispatching through the crew can parallelize at all.
    #[inline]
    #[must_use]
    pub fn is_parallel(&self) -> bool {
        self.n_workers > 1
    }

    /// Total threads a region can run on (workers + the dispatcher).
    #[inline]
    #[must_use]
    pub fn n_threads(&self) -> usize {
        self.n_workers
    }

    /// Run `task(i)` for every `i in 0..n_tasks`, potentially in parallel across
    /// the crew (the dispatching thread participates). Falls back to the serial
    /// ascending loop when the crew is size 1, another dispatch is in flight, or
    /// `n_tasks` exceeds [`MAX_TASKS`]. Returns when every task has completed.
    ///
    /// The caller guarantees tasks are independent (disjoint outputs); the crew
    /// guarantees each index runs exactly once.
    pub fn run(&self, n_tasks: usize, task: &(dyn Fn(usize) + Sync)) {
        if n_tasks == 0 {
            return;
        }
        if !self.is_parallel() || n_tasks == 1 || n_tasks > MAX_TASKS {
            for i in 0..n_tasks {
                task(i);
            }
            return;
        }
        let Ok(mut epoch_guard) = self.dispatch_lock.try_lock() else {
            // Concurrent or re-entrant dispatch: run serial — identical results.
            for i in 0..n_tasks {
                task(i);
            }
            return;
        };
        let s = self.shared;
        *epoch_guard += 1;
        let epoch = *epoch_guard;

        unsafe fn call_impl(data: *const (), idx: usize) {
            // SAFETY: `data` is the dispatcher's `&&(dyn Fn..)`, valid for the
            // whole region (the drain holds the dispatcher until completion).
            let f = unsafe { &*(data as *const &(dyn Fn(usize) + Sync)) };
            f(idx);
        }
        let data_ref: &&(dyn Fn(usize) + Sync) = &task;
        let job = RawJob {
            data: std::ptr::from_ref(data_ref).cast(),
            call: call_impl,
        };

        // Publish (order is load-bearing; see module docs).
        // SAFETY: single writer (dispatch_lock held); previous region drained.
        unsafe { *s.job.0.get() = job };
        s.n_packed
            .store((epoch << COUNTER_BITS) | n_tasks as u64, Ordering::Release);
        s.tasks_done.store(0, Ordering::Relaxed);
        s.claim_ctr.store(epoch << COUNTER_BITS, Ordering::Release);
        // Wake parked workers (permit-based; racing with a re-check is fine).
        for (wid, flag) in s.parked.iter().enumerate() {
            if flag.load(Ordering::Relaxed) {
                let threads = s.threads.lock().expect("crew threads lock");
                if let Some(t) = threads.get(wid) {
                    t.unpark();
                }
            }
        }

        // The dispatcher participates.
        run_tasks(s);

        // Drain: every in-range claim contributes exactly one increment.
        let mut spins = 0u32;
        while s.tasks_done.load(Ordering::Acquire) < n_tasks {
            std::hint::spin_loop();
            spins += 1;
            if spins & ((1 << 14) - 1) == 0 {
                thread::yield_now();
            }
        }
        if s.poisoned.swap(false, Ordering::Relaxed) {
            panic!("a crew task panicked; crew region poisoned");
        }
    }
}

/// Claim-and-run loop shared by workers and the dispatcher. Exits on the first
/// out-of-range (or stale-region) claim. See the module docs for why the job
/// read after an in-range claim can never race a republication.
fn run_tasks(s: &Shared) {
    loop {
        let v = s.claim_ctr.fetch_add(1, Ordering::Acquire);
        let (ce, t) = (v >> COUNTER_BITS, v & COUNTER_MASK);
        let np = s.n_packed.load(Ordering::Acquire);
        let (ne, n) = (np >> COUNTER_BITS, np & COUNTER_MASK);
        // `ne < ce` impossible (publication order); `ne > ce` ⇒ region drained.
        if ne != ce || t >= n {
            return;
        }
        struct DoneGuard<'a>(&'a Shared);
        impl Drop for DoneGuard<'_> {
            fn drop(&mut self) {
                self.0.tasks_done.fetch_add(1, Ordering::Release);
            }
        }
        let _done = DoneGuard(s);
        // SAFETY: in-range claim ⇒ this is the live region's job (module docs).
        let job = unsafe { *s.job.0.get() };
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            (job.call)(job.data, t as usize)
        }));
        if r.is_err() {
            s.poisoned.store(true, Ordering::Relaxed);
        }
    }
}

fn worker_loop(s: &'static Shared, wid: usize) {
    let mut seen_epoch = 0u64;
    // ~40 spin_loop hints ≈ 1µs is a coarse machine-dependent estimate; the
    // budget only trades wake latency vs idle burn, never correctness.
    let spin_budget = spin_us().saturating_mul(40);
    loop {
        // Wait for a new epoch: spin, then park.
        let mut spins: u64 = 0;
        loop {
            let e = s.claim_ctr.load(Ordering::Acquire) >> COUNTER_BITS;
            if e != seen_epoch {
                seen_epoch = e;
                break;
            }
            if s.shutdown.load(Ordering::Relaxed) {
                return;
            }
            spins += 1;
            if spins > spin_budget {
                s.parked[wid].store(true, Ordering::SeqCst);
                // Re-check to close the publish/park race, then park.
                if s.claim_ctr.load(Ordering::Acquire) >> COUNTER_BITS == seen_epoch
                    && !s.shutdown.load(Ordering::Relaxed)
                {
                    thread::park();
                }
                s.parked[wid].store(false, Ordering::SeqCst);
                spins = 0;
            } else {
                std::hint::spin_loop();
            }
        }
        run_tasks(s);
    }
}

/// Parallel for-each over `&mut` items: crew analog of
/// `items.par_iter_mut().for_each(|x| f(i, x))`. Each index is claimed exactly
/// once, so handing task `i` a `&mut items[i]` is sound (disjoint by
/// construction).
pub fn for_each_mut<T: Send, F: Fn(usize, &mut T) + Sync>(items: &mut [T], f: F) {
    struct SendPtr<T>(*mut T);
    // SAFETY: unique-index claims make the concurrent `&mut`s disjoint.
    unsafe impl<T> Sync for SendPtr<T> {}
    let ptr = SendPtr(items.as_mut_ptr());
    let n = items.len();
    // Capture the WRAPPER (not the raw-pointer field) so the closure stays
    // `Sync` — edition-2021 disjoint capture would otherwise grab `ptr.0`.
    let ptr_ref = &ptr;
    Crew::global().run(n, &|i| {
        // SAFETY: i < n (claim protocol); each i claimed exactly once.
        let item = unsafe { &mut *ptr_ref.0.add(i) };
        f(i, item);
    });
}

/// [`for_each_mut`] when the crew can parallelize, else a rayon
/// `par_iter_mut` — for call sites whose PRE-crew behavior was rayon-parallel,
/// so `LGBM_CREW=0` restores the old execution shape instead of degrading a
/// large region to serial. Results are identical in every arm (disjoint
/// indexed writes; per-item order unchanged).
pub fn for_each_mut_or_rayon<T: Send, F: Fn(usize, &mut T) + Sync>(items: &mut [T], f: F) {
    if Crew::global().is_parallel() {
        for_each_mut(items, f);
    } else {
        use rayon::prelude::*;
        items.par_iter_mut().enumerate().for_each(|(i, item)| f(i, item));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    #[test]
    fn every_index_runs_exactly_once() {
        for n in [0usize, 1, 2, 7, 50, 128, 1000] {
            let hits: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(0)).collect();
            Crew::global().run(n, &|i| {
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
            for (i, h) in hits.iter().enumerate() {
                assert_eq!(h.load(Ordering::Relaxed), 1, "index {i} of {n}");
            }
        }
    }

    #[test]
    fn repeated_regions_reuse_workers() {
        let total = AtomicU32::new(0);
        for _ in 0..2000 {
            Crew::global().run(16, &|_| {
                total.fetch_add(1, Ordering::Relaxed);
            });
        }
        assert_eq!(total.load(Ordering::Relaxed), 2000 * 16);
    }

    #[test]
    fn for_each_mut_writes_disjoint_items() {
        let mut v = vec![0u64; 333];
        for_each_mut(&mut v, |i, x| *x = i as u64 * 3);
        for (i, x) in v.iter().enumerate() {
            assert_eq!(*x, i as u64 * 3);
        }
    }

    #[test]
    fn nested_dispatch_falls_back_serial_without_deadlock() {
        let outer_hits = AtomicU32::new(0);
        let inner_hits = AtomicU32::new(0);
        Crew::global().run(8, &|_| {
            outer_hits.fetch_add(1, Ordering::Relaxed);
            // Inner dispatch from a crew task: the lock is held (or contended),
            // so this must degrade to serial and complete.
            Crew::global().run(4, &|_| {
                inner_hits.fetch_add(1, Ordering::Relaxed);
            });
        });
        assert_eq!(outer_hits.load(Ordering::Relaxed), 8);
        assert_eq!(inner_hits.load(Ordering::Relaxed), 8 * 4);
    }

    #[test]
    fn task_panic_poisons_region_and_repanics_on_dispatcher() {
        let r = std::panic::catch_unwind(|| {
            Crew::global().run(8, &|i| {
                if i == 3 {
                    panic!("boom");
                }
            });
        });
        assert!(r.is_err(), "dispatcher must observe the poisoned region");
        // The crew must remain usable after a poisoned region.
        let ok = AtomicU32::new(0);
        Crew::global().run(8, &|_| {
            ok.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(ok.load(Ordering::Relaxed), 8);
    }

    #[test]
    fn concurrent_dispatch_from_many_threads_is_safe() {
        // Only one thread wins the crew; the rest run serial. All must complete
        // with exact counts.
        let total = AtomicU32::new(0);
        thread::scope(|sc| {
            for _ in 0..4 {
                sc.spawn(|| {
                    for _ in 0..200 {
                        Crew::global().run(10, &|_| {
                            total.fetch_add(1, Ordering::Relaxed);
                        });
                    }
                });
            }
        });
        assert_eq!(total.load(Ordering::Relaxed), 4 * 200 * 10);
    }
}
