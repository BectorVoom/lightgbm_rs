//! quick 260620-dpk — env-gated per-bucket wall-clock accumulators for the unified
//! host-fusion per-leaf fixed cost. Inert unless `LGBM_FUSION_PROF=1`; in that case the
//! four per-leaf buckets of [`build_fix_scan_impl`](crate::CpuBackend) and
//! [`subtract_scan_impl`](crate::CpuBackend) accumulate into process-global atomics that
//! the Task-1 bench harness prints via [`dump`].
//!
//! Measurement-only: every counter sits behind the [`enabled`] gate, so when the env var
//! is off the instrumented sites add NOTHING — no value is read, written, gathered, or
//! reordered. Parity is therefore unaffected (proven FORCED-ON in the Task-4 gate, which
//! never sets `LGBM_FUSION_PROF`). Lives in `lgbm-compute` (not the treelearner
//! `phase_prof`) because the fused impls are in this crate and `lgbm-compute` must NOT
//! depend on `lgbm-treelearner` (that would be a dependency CYCLE).

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// --- build_fix_scan_impl (smaller / directly-built child) buckets ---
/// (1) SERIAL GATHER: the `ord_g`/`ord_h` per-leaf gather loop.
pub static BFS_GATHER_NS: AtomicU64 = AtomicU64::new(0);
/// (2) ALLOCATIONS: the per-leaf `Vec::with_capacity`/`vec![..]` setup (lever-A-reducible).
pub static BFS_ALLOC_NS: AtomicU64 = AtomicU64::new(0);
/// (3+4) FORK/JOIN + PARALLEL WORK: the `par_iter` region (fork/join floor + fold/fix/scan).
pub static BFS_PAR_NS: AtomicU64 = AtomicU64::new(0);

// --- subtract_scan_impl (larger / subtract-derived child) buckets ---
/// (2) ALLOCATIONS: the per-leaf `ranges`/`out` setup (lever-A-reducible).
pub static SUB_ALLOC_NS: AtomicU64 = AtomicU64::new(0);
/// (3+4) FORK/JOIN + PARALLEL WORK: the `par_iter` subtract+scan region.
pub static SUB_PAR_NS: AtomicU64 = AtomicU64::new(0);

/// Whether `LGBM_FUSION_PROF=1` is set (cached once).
#[inline]
pub fn enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("LGBM_FUSION_PROF").map(|v| v == "1").unwrap_or(false))
}

/// Time `f` into `counter` (nanoseconds). Zero-overhead passthrough when the gate is off
/// — the closure runs identically either way, so values/order are untouched.
#[inline]
pub fn time<T>(counter: &AtomicU64, f: impl FnOnce() -> T) -> T {
    if !enabled() {
        return f();
    }
    let t = Instant::now();
    let r = f();
    counter.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    r
}

/// Print the accumulated per-bucket fusion breakdown to stderr and reset the counters.
/// No-op when the gate is off.
pub fn dump(label: &str) {
    if !enabled() {
        return;
    }
    let g = BFS_GATHER_NS.swap(0, Ordering::Relaxed);
    let ba = BFS_ALLOC_NS.swap(0, Ordering::Relaxed);
    let bp = BFS_PAR_NS.swap(0, Ordering::Relaxed);
    let sa = SUB_ALLOC_NS.swap(0, Ordering::Relaxed);
    let sp = SUB_PAR_NS.swap(0, Ordering::Relaxed);
    let bfs_tot = (g + ba + bp) as f64 / 1e6;
    let sub_tot = (sa + sp) as f64 / 1e6;
    eprintln!(
        "[fusion_prof:{label}] BFS: gather={:.3}ms alloc={:.3}ms par={:.3}ms total={:.3}ms",
        g as f64 / 1e6,
        ba as f64 / 1e6,
        bp as f64 / 1e6,
        bfs_tot
    );
    if bfs_tot > 0.0 {
        eprintln!(
            "[fusion_prof:{label}] BFS %: gather={:.1} alloc={:.1} par={:.1}",
            g as f64 / 1e4 / bfs_tot,
            ba as f64 / 1e4 / bfs_tot,
            bp as f64 / 1e4 / bfs_tot
        );
    }
    eprintln!(
        "[fusion_prof:{label}] SUB: alloc={:.3}ms par={:.3}ms total={:.3}ms",
        sa as f64 / 1e6,
        sp as f64 / 1e6,
        sub_tot
    );
    if sub_tot > 0.0 {
        eprintln!(
            "[fusion_prof:{label}] SUB %: alloc={:.1} par={:.1}",
            sa as f64 / 1e4 / sub_tot,
            sp as f64 / 1e4 / sub_tot
        );
    }
}
