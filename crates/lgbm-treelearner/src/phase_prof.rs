//! Spike 002 — env-gated per-phase wall-clock accumulator for the serial
//! tree-learner hot loop. Inert unless `LGBM_PHASE_PROF=1`; in that case the
//! three growth-loop phases accumulate into process-global atomics that the
//! bench harness prints via [`dump`]. Measurement-only; never changes train
//! semantics, and the `time` wrapper is a direct passthrough when disabled.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

pub static BEFORE_NS: AtomicU64 = AtomicU64::new(0);
pub static HISTSPLIT_NS: AtomicU64 = AtomicU64::new(0);
pub static PARTITION_NS: AtomicU64 = AtomicU64::new(0);
// Sub-phases of HISTSPLIT: BUILD = histogram construction (the per-feature gather),
// SCAN = the per-feature find_best_split scan. BUILD+SCAN ≈ HISTSPLIT (minus
// subtract/compact glue), used to localize the hist-vs-split gap.
pub static BUILD_NS: AtomicU64 = AtomicU64::new(0);
pub static SCAN_NS: AtomicU64 = AtomicU64::new(0);

/// RAII timer: accumulates its lifetime into `c`. Inert when the env gate is off.
pub struct Guard {
    c: &'static AtomicU64,
    t: Instant,
    on: bool,
}
impl Drop for Guard {
    fn drop(&mut self) {
        if self.on {
            self.c
                .fetch_add(self.t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }
}
/// Start an RAII phase timer accumulating into `c` until the guard drops.
pub fn guard(c: &'static AtomicU64) -> Guard {
    Guard { c, t: Instant::now(), on: enabled() }
}

fn enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("LGBM_PHASE_PROF").map(|v| v == "1").unwrap_or(false))
}

/// Time `f` into `counter` (nanoseconds). Zero-overhead passthrough when the
/// env gate is off.
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

/// Print the accumulated phase breakdown to stderr and reset the counters.
pub fn dump(label: &str) {
    if !enabled() {
        return;
    }
    let b = BEFORE_NS.swap(0, Ordering::Relaxed);
    let h = HISTSPLIT_NS.swap(0, Ordering::Relaxed);
    let p = PARTITION_NS.swap(0, Ordering::Relaxed);
    let build = BUILD_NS.swap(0, Ordering::Relaxed);
    let scan = SCAN_NS.swap(0, Ordering::Relaxed);
    let tot = (b + h + p) as f64 / 1e6;
    eprintln!(
        "[phase_prof:{label}] before={:.3}ms hist+split={:.3}ms (build={:.3} scan={:.3}) partition={:.3}ms total={:.3}ms",
        b as f64 / 1e6,
        h as f64 / 1e6,
        build as f64 / 1e6,
        scan as f64 / 1e6,
        p as f64 / 1e6,
        tot
    );
    if tot > 0.0 {
        eprintln!(
            "[phase_prof:{label}] %: before={:.1} hist+split={:.1} (build={:.1} scan={:.1}) partition={:.1}",
            b as f64 / 1e4 / tot,
            h as f64 / 1e4 / tot,
            build as f64 / 1e4 / tot,
            scan as f64 / 1e4 / tot,
            p as f64 / 1e4 / tot
        );
    }
}
