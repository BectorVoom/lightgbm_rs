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
// Spike-014b WHOLE-TRAIN BUDGET counters — the growth-loop phases above cover only
// ~31–45% of GPU train wall-clock (spike-014a); these attribute the uninstrumented
// majority. Wrapped at the boosting/binning seam, NOT nested inside the growth loop:
//   BINNING  = once-per-train `build_feature_columns` (fixed setup; bench-repeated).
//   GRAD     = per-iter objective `get_gradients` (grad/hess compute).
//   LEARNER  = per-iter `learner.train_*` call — SUPERSET of BEFORE+HISTSPLIT+PARTITION;
//              (LEARNER − those) = in-learner orchestration + per-tree GPU grad/hess
//              upload + resident-pool/partition setup OUTSIDE the phase guards.
//   SCORE    = per-iter `score_updater.update` (UpdateScore scatter).
pub static BINNING_NS: AtomicU64 = AtomicU64::new(0);
pub static GRAD_NS: AtomicU64 = AtomicU64::new(0);
pub static LEARNER_NS: AtomicU64 = AtomicU64::new(0);
pub static SCORE_NS: AtomicU64 = AtomicU64::new(0);
// Spike-014b drill-down: the per-`train_inner` (= per-TREE) resident-bin device upload
// (`wants_resident_bins` block, learner.rs) — two `[num_features × num_data]` u32 host
// re-allocations + a host→device `create_from_slice`, redundantly repeated every tree
// even though the binned columns are IMMUTABLE for the whole train. A subset of
// `in_learner_other`; GPU-only (CpuBackend `wants_resident_bins()==false`).
pub static UPLOAD_NS: AtomicU64 = AtomicU64::new(0);

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
    // Spike-014b WHOLE-TRAIN BUDGET. `learner` is the per-iter tree-train call and is a
    // SUPERSET of before+hist+split+partition; `in_learner_other = learner − that` is the
    // previously-uninstrumented in-learner cost (per-tree GPU grad/hess upload, resident-
    // pool / partition setup, host orchestration outside the growth-phase guards).
    let binning = BINNING_NS.swap(0, Ordering::Relaxed);
    let grad = GRAD_NS.swap(0, Ordering::Relaxed);
    let learner = LEARNER_NS.swap(0, Ordering::Relaxed);
    let score = SCORE_NS.swap(0, Ordering::Relaxed);
    if binning + grad + learner + score > 0 {
        let in_learner_other = learner.saturating_sub(b + h + p);
        let upload = UPLOAD_NS.swap(0, Ordering::Relaxed);
        eprintln!(
            "[phase_prof:{label}] BUDGET: binning={:.3}ms grad={:.3}ms learner={:.3}ms (phases={:.3} in_learner_other={:.3} [of which resident_bin_upload={:.3}]) score={:.3}ms",
            binning as f64 / 1e6,
            grad as f64 / 1e6,
            learner as f64 / 1e6,
            (b + h + p) as f64 / 1e6,
            in_learner_other as f64 / 1e6,
            upload as f64 / 1e6,
            score as f64 / 1e6,
        );
    }
}
