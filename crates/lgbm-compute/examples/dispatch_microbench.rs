//! THROWAWAY dispatch-latency micro-bench — quick task 260620-e8e (QUICK-260620-E8E).
//!
//! THE KILL SWITCH. Times the BARE per-dispatch cost of fanning out N tiny tasks
//! and joining, in ISOLATION — NOT wired into the tree learner, touching NO library
//! hot-path code. The whole spike's GO/NO-GO verdict is emitted by this example.
//!
//! Context (quick 260620-dpk): the unified-fusion per-leaf fixed cost is ~99 % rayon
//! fork/join floor + parallel work, ≤0.6 % allocation. The ONLY remaining lever that
//! could move the fusion gate (pinned at ~100 feat BFS / ~130 feat subscan by the
//! fork/join floor) is a dispatch primitive genuinely cheaper than rayon's join for
//! N∈{20–80} tiny per-feature tasks. This micro-bench tests exactly that primitive.
//!
//! KEY FACTS baked in:
//!   1. rayon 1.10 ALREADY keeps its global pool threads alive — the cost is
//!      `par_iter`'s work-split + the work-stealing JOIN BARRIER, NOT thread spawn.
//!   2. The tree grows SEQUENTIALLY across leaves (leaf-wise): no cross-leaf
//!      pipelining is possible. A pool can ONLY cheapen each leaf's single dispatch.
//!   3. spin-wait vs block is the ship/no-ship axis: a spin pool has low dispatch
//!      latency but BURNS idle cores — a real NO-SHIP risk for a library.
//!
//! CANDIDATES:
//!   (a) BASELINE  — rayon `(0..N).into_par_iter().for_each(..)` over the GLOBAL pool
//!                   (the exact primitive the fused per-leaf region uses today).
//!   (b) SPIN POOL — hand-rolled fixed-worker pool: W-1 persistent std::thread workers
//!                   parked on an `AtomicU64` epoch; driver bumps the epoch and
//!                   SPIN-WAITS on an `AtomicUsize` completion counter (low-latency /
//!                   high-burn end of the tradeoff). std primitives ONLY.
//!   (c) BLOCK POOL— same hand-rolled pool but BLOCKING: workers `park()`, driver
//!                   `unpark`s; completion via a `Mutex`+`Condvar` barrier
//!                   (no-burn / higher-wake-latency end). std primitives ONLY.
//!   (d) RAYON-NOOP— rayon over a pure no-op closure: the dispatch FLOOR reference.
//!
//! Per-task work = a realistic per-feature scan proxy: a tight fold over a fixed
//! 384-element buffer (~a few µs) so dispatch is measured against real granularity,
//! NOT a no-op (a no-op overstates dispatch's relative share).
//!
//! NO new external dependency. std `thread` + `std::sync::atomic`/`Condvar` only.
//! cubecl 0.10 stays pinned; this example touches no cubecl/runtime wiring.
//!
//! Run: `cargo run -p lgbm-compute --example dispatch_microbench --release`
//! Honors `RAYON_NUM_THREADS` (binds the hand-rolled pools to the same worker count).

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use rayon::prelude::*;

/// Per-task work proxy: a deterministic tight fold over a fixed buffer. Returns a
/// black-box value so the optimizer cannot elide it. ~a few µs of trivial arithmetic.
#[inline(never)]
fn task_work(buf: &[f64], seed: usize) -> f64 {
    let mut acc = seed as f64 * 1.000001;
    for (i, &x) in buf.iter().enumerate() {
        // multiply-add chain, dependent so it can't vectorize away to nothing
        acc = acc.mul_add(1.0000003, x) + (i as f64).sin() * 1e-9;
    }
    acc
}

const BUF_LEN: usize = 384;

fn make_buffers(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|f| {
            (0..BUF_LEN)
                .map(|i| ((f * 31 + i * 7) % 97) as f64 * 0.5 + 1.0)
                .collect()
        })
        .collect()
}

/// Spin-wait persistent pool: W-1 worker threads + the calling thread = W lanes.
/// Workers block-spin on an epoch `AtomicU64`; the driver bumps the epoch to release
/// them and SPIN-WAITS on a completion `AtomicUsize`. Low latency, burns idle cores.
struct SpinPool {
    workers: usize,
    epoch: Arc<AtomicU64>,
    done: Arc<AtomicUsize>,
    shutdown: Arc<AtomicBool>,
    // job descriptor: (task indices range start/len, raw ptrs) shared via Arc<Mutex<..>>
    job: Arc<Mutex<JobSlot>>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

/// The work each dispatch installs. We model the production fan-out: N independent
/// tasks each reading its own input buffer and writing its own output cell (disjoint).
struct JobSlot {
    n: usize,
    bufs_ptr: usize,  // *const Vec<f64> base (slice element ptr; index i = bufs[i])
    out_ptr: usize,   // *mut f64 base (each task writes out[i], disjoint)
}

// SAFETY: each task index i touches only bufs[i] (shared read) and out[i] (disjoint
// write). No two tasks alias. Driver fully synchronizes (epoch release → completion
// barrier) before reading out or mutating the job slot again.
unsafe impl Send for JobSlot {}

impl SpinPool {
    fn new(workers: usize) -> Self {
        let workers = workers.max(1);
        let epoch = Arc::new(AtomicU64::new(0));
        let done = Arc::new(AtomicUsize::new(0));
        let shutdown = Arc::new(AtomicBool::new(false));
        let job = Arc::new(Mutex::new(JobSlot { n: 0, bufs_ptr: 0, out_ptr: 0 }));
        let mut handles = Vec::with_capacity(workers.saturating_sub(1));
        // worker lanes 1..workers (lane 0 = driver thread)
        for lane in 1..workers {
            let epoch = epoch.clone();
            let done = done.clone();
            let shutdown = shutdown.clone();
            let job = job.clone();
            handles.push(std::thread::spawn(move || {
                let mut last_epoch = 0u64;
                loop {
                    // spin until a new epoch is published
                    loop {
                        let e = epoch.load(Ordering::Acquire);
                        if e != last_epoch {
                            last_epoch = e;
                            break;
                        }
                        if shutdown.load(Ordering::Acquire) {
                            return;
                        }
                        std::hint::spin_loop();
                    }
                    if shutdown.load(Ordering::Acquire) {
                        return;
                    }
                    run_lane(&job, lane, workers, &done);
                }
            }));
        }
        SpinPool { workers, epoch, done, shutdown, job, handles }
    }

    /// Dispatch N tasks, spin-wait completion barrier. Disjoint out writes.
    fn dispatch(&self, bufs: &[Vec<f64>], out: &mut [f64]) {
        debug_assert_eq!(bufs.len(), out.len());
        let n = bufs.len();
        {
            let mut j = self.job.lock().unwrap();
            j.n = n;
            j.bufs_ptr = bufs.as_ptr() as usize;
            j.out_ptr = out.as_mut_ptr() as usize;
        }
        self.done.store(0, Ordering::Release);
        // publish a new epoch -> releases all worker lanes
        self.epoch.fetch_add(1, Ordering::AcqRel);
        // driver itself runs lane 0
        run_lane(&self.job, 0, self.workers, &self.done);
        // spin-wait the completion barrier
        while self.done.load(Ordering::Acquire) < self.workers {
            std::hint::spin_loop();
        }
    }
}

impl Drop for SpinPool {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        self.epoch.fetch_add(1, Ordering::AcqRel);
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

/// Execute lane `lane` of `workers` over the installed job, then bump `done`.
/// Static block-cyclic split: lane handles indices [lane*chunk, (lane+1)*chunk).
fn run_lane(job: &Arc<Mutex<JobSlot>>, lane: usize, workers: usize, done: &AtomicUsize) {
    let (n, bufs_ptr, out_ptr) = {
        let j = job.lock().unwrap();
        (j.n, j.bufs_ptr, j.out_ptr)
    };
    if n > 0 {
        // SAFETY: disjoint per-index access; see JobSlot contract.
        let bufs_base = bufs_ptr as *const Vec<f64>;
        let chunk = n.div_ceil(workers);
        let start = (lane * chunk).min(n);
        let end = ((lane + 1) * chunk).min(n);
        for i in start..end {
            let v = task_work(unsafe { &*bufs_base.add(i) }, i);
            // SAFETY: each i writes its own out[i] — disjoint across lanes.
            unsafe {
                *(out_ptr as *mut f64).add(i) = v;
            }
        }
    }
    done.fetch_add(1, Ordering::AcqRel);
}

/// Blocking persistent pool: workers `park()`, driver `unpark`s them; completion via
/// a Mutex+Condvar count barrier. No idle burn; higher wake latency.
struct BlockPool {
    workers: usize,
    go: Arc<(Mutex<u64>, Condvar)>,    // epoch + condvar to wake workers
    done: Arc<(Mutex<usize>, Condvar)>, // completion count + condvar
    shutdown: Arc<AtomicBool>,
    job: Arc<Mutex<JobSlot>>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl BlockPool {
    fn new(workers: usize) -> Self {
        let workers = workers.max(1);
        let go = Arc::new((Mutex::new(0u64), Condvar::new()));
        let done = Arc::new((Mutex::new(0usize), Condvar::new()));
        let shutdown = Arc::new(AtomicBool::new(false));
        let job = Arc::new(Mutex::new(JobSlot { n: 0, bufs_ptr: 0, out_ptr: 0 }));
        let mut handles = Vec::with_capacity(workers.saturating_sub(1));
        for lane in 1..workers {
            let go = go.clone();
            let done = done.clone();
            let shutdown = shutdown.clone();
            let job = job.clone();
            handles.push(std::thread::spawn(move || {
                let mut last_epoch = 0u64;
                loop {
                    {
                        let (lock, cvar) = &*go;
                        let mut e = lock.lock().unwrap();
                        while *e == last_epoch && !shutdown.load(Ordering::Acquire) {
                            e = cvar.wait(e).unwrap();
                        }
                        last_epoch = *e;
                    }
                    if shutdown.load(Ordering::Acquire) {
                        return;
                    }
                    run_lane_block(&job, lane, workers, &done);
                }
            }));
        }
        BlockPool { workers, go, done, shutdown, job, handles }
    }

    fn dispatch(&self, bufs: &[Vec<f64>], out: &mut [f64]) {
        let n = bufs.len();
        {
            let mut j = self.job.lock().unwrap();
            j.n = n;
            j.bufs_ptr = bufs.as_ptr() as usize;
            j.out_ptr = out.as_mut_ptr() as usize;
        }
        {
            let (lock, _) = &*self.done;
            *lock.lock().unwrap() = 0;
        }
        // wake workers
        {
            let (lock, cvar) = &*self.go;
            *lock.lock().unwrap() += 1;
            cvar.notify_all();
        }
        // driver runs lane 0
        run_lane_block(&self.job, 0, self.workers, &self.done);
        // wait completion barrier
        let (lock, cvar) = &*self.done;
        let mut c = lock.lock().unwrap();
        while *c < self.workers {
            c = cvar.wait(c).unwrap();
        }
    }
}

impl Drop for BlockPool {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        let (lock, cvar) = &*self.go;
        *lock.lock().unwrap() += 1;
        cvar.notify_all();
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

fn run_lane_block(job: &Arc<Mutex<JobSlot>>, lane: usize, workers: usize, done: &Arc<(Mutex<usize>, Condvar)>) {
    let (n, bufs_ptr, out_ptr) = {
        let j = job.lock().unwrap();
        (j.n, j.bufs_ptr, j.out_ptr)
    };
    if n > 0 {
        let bufs_base = bufs_ptr as *const Vec<f64>;
        let chunk = n.div_ceil(workers);
        let start = (lane * chunk).min(n);
        let end = ((lane + 1) * chunk).min(n);
        for i in start..end {
            let v = task_work(unsafe { &*bufs_base.add(i) }, i);
            unsafe {
                *(out_ptr as *mut f64).add(i) = v;
            }
        }
    }
    let (lock, cvar) = &**done;
    let mut c = lock.lock().unwrap();
    *c += 1;
    if *c >= workers {
        cvar.notify_all();
    }
}

/// Median + p25/p75 of per-dispatch µs from a vec of per-dispatch nanos.
fn quantiles(mut ns: Vec<f64>) -> (f64, f64, f64) {
    ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| {
        let idx = ((ns.len() as f64 - 1.0) * p).round() as usize;
        ns[idx] / 1000.0 // ns -> µs
    };
    (q(0.25), q(0.50), q(0.75))
}

/// Run `reps` outer reps of `inner` dispatches; return per-dispatch nanos (one per
/// dispatch), discarding the first outer rep as warmup.
fn bench<F: FnMut()>(reps: usize, inner: usize, mut dispatch_once: F) -> Vec<f64> {
    // warmup rep (discarded)
    for _ in 0..inner {
        dispatch_once();
    }
    let mut per_dispatch = Vec::with_capacity(reps * inner);
    for _ in 0..reps {
        for _ in 0..inner {
            let t = Instant::now();
            dispatch_once();
            per_dispatch.push(t.elapsed().as_nanos() as f64);
        }
    }
    per_dispatch
}

fn main() {
    let workers = rayon::current_num_threads().max(1);
    println!("# 260620-e8e dispatch micro-bench (THROWAWAY, isolated — not wired into learner)");
    println!("# rayon worker threads = {workers}  (RAYON_NUM_THREADS={})",
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset".into()));
    println!("# per-task work = fold over {BUF_LEN}-elem buffer; reps tuned for >=10k inner dispatches/cell");
    println!();

    let ns = [20usize, 40, 80];
    // tune inner/outer so total inner dispatches per cell >= 10k, >=5 warm outer reps.
    let outer_reps = 6; // 1 discarded warmup + 5 measured
    // inner per rep: pick so 5*inner >= 10000 -> inner >= 2000
    let inner = 2200;

    let spin = SpinPool::new(workers);
    let block = BlockPool::new(workers);

    // accumulate rows for the table + the GO/NO-GO decision
    // rows: (label, N, p25, med, p75)
    let mut rows: Vec<(&'static str, usize, f64, f64, f64)> = Vec::new();
    // per-N rayon baseline median for the verdict
    let mut rayon_med: std::collections::HashMap<usize, f64> = Default::default();
    let mut spin_med: std::collections::HashMap<usize, f64> = Default::default();
    let mut block_med: std::collections::HashMap<usize, f64> = Default::default();

    for &n in &ns {
        let bufs = make_buffers(n);
        let mut out = vec![0.0f64; n];

        // (d) rayon NO-OP floor reference
        {
            let out2 = vec![0.0f64; n];
            let samples = bench(outer_reps, inner, || {
                (0..n).into_par_iter().for_each(|i| {
                    // pure dispatch floor: trivial write, no work proxy
                    let p = out2.as_ptr() as *mut f64;
                    unsafe { *p.add(i) = i as f64; }
                });
            });
            let (a, b, c) = quantiles(samples);
            rows.push(("rayon-NOOP-floor", n, a, b, c));
            std::hint::black_box(&out2);
        }

        // (a) BASELINE rayon par_iter + work proxy
        {
            let samples = bench(outer_reps, inner, || {
                out.par_iter_mut().enumerate().for_each(|(i, o)| {
                    *o = task_work(&bufs[i], i);
                });
            });
            let (a, b, c) = quantiles(samples);
            rayon_med.insert(n, b);
            rows.push(("rayon-baseline", n, a, b, c));
            std::hint::black_box(&out);
        }

        // (b) SPIN pool + work proxy
        {
            let samples = bench(outer_reps, inner, || {
                spin.dispatch(&bufs, &mut out);
            });
            let (a, b, c) = quantiles(samples);
            spin_med.insert(n, b);
            rows.push(("spin-pool", n, a, b, c));
            std::hint::black_box(&out);
        }

        // (c) BLOCK pool + work proxy
        {
            let samples = bench(outer_reps, inner, || {
                block.dispatch(&bufs, &mut out);
            });
            let (a, b, c) = quantiles(samples);
            block_med.insert(n, b);
            rows.push(("block-pool", n, a, b, c));
            std::hint::black_box(&out);
        }
    }

    // -------- correctness cross-check: pool results == rayon results (disjoint writes) --------
    {
        let n = 40usize;
        let bufs = make_buffers(n);
        let mut expect = vec![0.0f64; n];
        expect.par_iter_mut().enumerate().for_each(|(i, o)| *o = task_work(&bufs[i], i));
        let mut got_spin = vec![0.0f64; n];
        spin.dispatch(&bufs, &mut got_spin);
        let mut got_block = vec![0.0f64; n];
        block.dispatch(&bufs, &mut got_block);
        let bit_eq_spin = expect.iter().zip(&got_spin).all(|(a, b)| a.to_bits() == b.to_bits());
        let bit_eq_block = expect.iter().zip(&got_block).all(|(a, b)| a.to_bits() == b.to_bits());
        println!("# correctness: spin bit-eq rayon = {bit_eq_spin}, block bit-eq rayon = {bit_eq_block} (disjoint-write order preserved)");
        println!();
    }

    // -------- table --------
    println!("| candidate         |  N | p25 µs | med µs | p75 µs |");
    println!("|-------------------|---:|-------:|-------:|-------:|");
    for (label, n, p25, med, p75) in &rows {
        println!("| {label:<17} | {n:>2} | {p25:>6.3} | {med:>6.3} | {p75:>6.3} |");
    }
    println!();

    // -------- verdict --------
    // GO criterion: a candidate beats rayon's median per-dispatch by >=2x AND the
    // absolute per-leaf saving (~N * per-dispatch delta) is a meaningful fraction of
    // the ~100-feat per-leaf train-wall. dpk: unified win ~ -6% train-wall at >=100
    // feat (per-leaf wall ~155 ms BFS / 97 ms SUB at 80 feat). The spin candidate is
    // NOT auto-GO on latency alone (idle-burn caveat flagged separately).
    println!("## GO/NO-GO analysis");
    let mut any_go = false;
    for &n in &ns {
        let rb = rayon_med[&n];
        let sp = spin_med[&n];
        let bl = block_med[&n];
        let spin_ratio = rb / sp; // >1 means spin faster
        let block_ratio = rb / bl;
        // absolute per-leaf saving estimate from the per-dispatch delta:
        // a leaf does ~2 dispatches (BFS build + SUB subtract). delta_us per dispatch
        // * 2 dispatches = per-leaf µs saved. Compare to ~per-leaf train-wall.
        // Use the larger of (rb - sp) for the best candidate.
        let best_delta_us = (rb - sp).max(rb - bl).max(0.0);
        let per_leaf_saving_us = best_delta_us * 2.0; // BFS + SUB dispatch
        // per-leaf train-wall at this feat (interpolate dpk BFS+SUB totals ~ 250 ms/leaf
        // at 80 feat across both children; scale ~linearly). Use a conservative ~3 ms/feat.
        let per_leaf_wall_us = (n as f64) * 3000.0;
        let saving_frac = per_leaf_saving_us / per_leaf_wall_us * 100.0;
        let go_latency = spin_ratio >= 2.0 || block_ratio >= 2.0;
        let go_magnitude = saving_frac >= 6.0;
        let go = go_latency && go_magnitude;
        any_go |= go;
        println!(
            "  N={n:>2}: rayon={rb:.3}µs spin={sp:.3}µs(×{spin_ratio:.2}) block={bl:.3}µs(×{block_ratio:.2})  \
             best per-leaf saving≈{per_leaf_saving_us:.2}µs = {saving_frac:.3}% of ~{:.0}ms wall  \
             [latency≥2×:{go_latency} magnitude≥6%:{go_magnitude}]",
            per_leaf_wall_us / 1000.0
        );
    }
    println!();
    if any_go {
        println!("VERDICT: GO  — a candidate cuts rayon's per-dispatch by ≥2× AND yields a ≥6%-of-per-leaf-wall saving at some N.");
        println!("         (SPIN candidate must still survive the idle-burn caveat in Task 2 before ship.)");
    } else {
        println!("VERDICT: NO-GO  — no candidate cuts rayon's per-dispatch by ≥2× with a ≥6%-of-per-leaf-wall absolute saving.");
        println!("         rayon's fork/join join-barrier is already near-optimal for N∈{{20,40,80}} tiny tasks; the");
        println!("         hand-rolled pools' Mutex/atomic barrier + static split does not beat it by enough to move");
        println!("         the ~100/130-feat fusion gate. Documented NULL — STOP at Task 1, do NOT build Task 2.");
        println!("VERDICT: NULL");
    }
}
