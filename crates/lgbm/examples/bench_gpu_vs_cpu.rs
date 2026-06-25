//! GPU(rocm)-vs-CPU whole-tree-learner train benchmark for `lightgbm_rs`
//! (quick task 260619-j9t — the speed-comparison deliverable for the CubeCL CUDA
//! mirror port).
//!
//! The GPU vs CPU backend is selected at **COMPILE TIME** by `--features rocm`
//! (`booster.rs` cfg-gates `RocmBackend` vs the `CpuBackend` f64 anchor). So this
//! example reports which backend it was built with and the operator runs it TWICE
//! to get the side-by-side numbers:
//!
//!   CPU (multi-threaded f64 anchor):
//!     cargo run --release --example bench_gpu_vs_cpu
//!   GPU (gfx1100 via cubecl-hip):
//!     cargo run --release --features rocm --example bench_gpu_vs_cpu
//!
//! WARM-VS-COLD RULE (spike-findings SKILL.md, load-bearing): the cold isolated
//! ceiling overstates the warm win 3–7× (allocator/JIT amortization). So BOTH paths
//! run `WARMUP` train iterations (DISCARDED) before the timed median loop, and we
//! report the **median of `TRAIN_REPS`** warm runs.
//!
//! The CPU column is the MULTI-THREADED anchor (feature-parallel histogram build is
//! rayon-parallel at leaf_rows >= 16384, spike-005) — at large leaves the CPU path
//! uses all cores, so the GPU only competes well above small leaves (the GPU build
//! is atomic/latency-bound, spike-006). The >=200k-row size below is where the GPU
//! path stops being purely launch-bound.

use std::time::{Duration, Instant};

use lgbm::{train, DenseCorpus, TrainingBuilder};

// ===========================================================================
// Phase 12 (spike-024 / SC-3 + SC-4) — co-pack ON/OFF A/B
// ===========================================================================
//
// Gated behind `LGBM_BENCH_COPACK_AB=1` (the default bench output is unchanged).
// MUST be run under `LGBM_PHASE_PROF=1` so the `scan_resident` sync counter is
// live, and `--features rocm` so the resident co-pack path actually fires (the
// CpuBackend f64 anchor has no resident pool, so this section is GPU-only).
//
//   LGBM_BENCH_COPACK_AB=1 LGBM_PHASE_PROF=1 \
//     cargo run --release --features rocm --example bench_gpu_vs_cpu
//
// HONEST FRAMING (load-bearing — do NOT misread):
//   * SC-3 (structural, REAL): co-packing the two per-sibling resident scans into
//     ONE 2-slot launch + ONE readback halves the per-tree `scan_resident` SYNC
//     count (~59 -> ~30, spike-023 COUNTS). This is counter-exact and
//     shape-independent — it is the deliverable.
//   * SC-4 (e2e, SIGN-ONLY): the spike-024 ISOLATED scan A/B was ~2.0× — that is
//     the launch+readback COMPONENT only and is NOT the e2e number. Per spike-023's
//     scan-sync fraction of total train, the e2e ceiling is ~10–15% at small/medium
//     and ~1.5% at wide. We report median train OFF vs ON and a NOT-SLOWER /
//     trends-faster verdict; we DO NOT assert a pass/fail on the e2e ratio.
//   * THIS BOX is a spoofed 8-CU APU (gfx1152, HSA-overridden) — absolute perf is
//     APU-confounded. Judge SIGN only; run >=2 processes for sign-stability
//     (CONVENTIONS.md device-time discipline: warm median, >=2 restarts).
//   * Wide (1M×500) is expected ~unaffected (~1.5% sync fraction); routing unchanged.

#[cfg(feature = "rocm")]
const COPACK_OFF: &str = "0";
#[cfg(feature = "rocm")]
const COPACK_ON: &str = "1";

/// A benchmark size: `rows` × `features`, each feature identity-binned into
/// `bins` distinct integer values (`0..bins-1`, all present).
struct Size {
    name: &'static str,
    rows: usize,
    features: usize,
    bins: usize,
}

/// Build a deterministic identity-binned corpus (no RNG → fully reproducible across
/// builds, so the CPU and GPU runs train on byte-identical data). Mirrors
/// `bench_train.rs::make_corpus`: the first `bins` rows force full bin coverage (the
/// `DenseCorpus` identity-binning precondition); the rest are a fixed integer hash.
fn make_corpus(s: &Size) -> DenseCorpus {
    let mut features: Vec<Vec<f64>> = Vec::with_capacity(s.rows);
    let mut labels: Vec<f32> = Vec::with_capacity(s.rows);
    for row in 0..s.rows {
        let mut frow: Vec<f64> = Vec::with_capacity(s.features);
        let mut acc: f64 = 0.0;
        for f in 0..s.features {
            let v = if row < s.bins {
                (row + f) % s.bins
            } else {
                let h = row
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(f.wrapping_mul(40_503).wrapping_add(0x9E37_79B9));
                h % s.bins
            };
            frow.push(v as f64);
            acc += (v as f64) * (1.0 + (f % 5) as f64);
        }
        features.push(frow);
        labels.push((acc * 0.01) as f32);
    }
    DenseCorpus { features, labels }
}

/// Median of a slice of durations (sorts a copy).
fn median(mut ds: Vec<Duration>) -> Duration {
    ds.sort();
    ds[ds.len() / 2]
}

/// Run `warmup` discarded trains then `reps` timed trains on `corpus`/`cfg`,
/// returning `(median_train_time, scan_resident_sync_count_per_train)`.
///
/// The `scan_resident` count is captured by swapping the `phase_prof`
/// `SCAN_RESIDENT_CNT` atomic to 0 right before the FIRST timed rep and reading
/// it after the LAST, then dividing by `reps` to get the per-train sync count
/// (the per-tree count is `per_train / iters`). Inert (reads 0) unless
/// `LGBM_PHASE_PROF=1`. GPU-only: the resident co-pack/scan path never fires on
/// the CpuBackend anchor, so this helper is compiled only under `--features rocm`.
#[cfg(feature = "rocm")]
fn timed_run(
    cfg: &lgbm::Config,
    corpus: &DenseCorpus,
    warmup: usize,
    reps: usize,
) -> (Duration, u64) {
    use std::sync::atomic::Ordering;
    // Warm-up (discarded) — amortize allocator/JIT/launch caches.
    for _ in 0..warmup {
        let _ = train(cfg, corpus).expect("warm-up train ok");
    }
    // Reset the sync counter AFTER warmup so we count only the timed reps.
    lgbm_treelearner::phase_prof::SCAN_RESIDENT_CNT.swap(0, Ordering::Relaxed);

    let mut times = Vec::with_capacity(reps);
    let mut sink = 0.0f64;
    for _ in 0..reps {
        let t0 = Instant::now();
        let booster = train(cfg, corpus).expect("train ok");
        times.push(t0.elapsed());
        sink += booster.predict(&corpus.features[0..1])[0][0] as f64;
    }
    std::hint::black_box(sink);
    let total_syncs = lgbm_treelearner::phase_prof::SCAN_RESIDENT_CNT.swap(0, Ordering::Relaxed);
    let per_train = if reps > 0 { total_syncs / reps as u64 } else { 0 };
    (median(times), per_train)
}

/// Phase 12 (SC-3 + SC-4) — co-pack ON/OFF A/B. Returns `true` when it ran (so
/// `main` skips the default single-config bench). GPU-only.
#[cfg(feature = "rocm")]
fn run_copack_ab(sizes: &[Size], cfg_for: &dyn Fn(&Size) -> lgbm::Config, iters: i32, warmup: usize, reps: usize) {
    println!("\n# === Phase 12 co-pack ON/OFF A/B (LGBM_SIBLING_COPACK 0 vs 1) ===");
    println!(
        "# HONEST FRAMING: the isolated scan A/B was ~2.0× (spike-024), which is the\n\
         # launch+readback COMPONENT only — NOT the e2e number. Per spike-023's scan-sync\n\
         # fraction, the e2e ceiling is ~10–15% at small/medium and ~1.5% at wide.\n\
         # SC-3 = the scan_resident SYNC count ~halves (~59 -> ~30/tree) — structural, real.\n\
         # SC-4 = median train is NOT-SLOWER and trends faster — SIGN-ONLY on this spoofed\n\
         #        8-CU APU (judge SIGN; run >=2 PROCESSES for sign-stability; no pass/fail gate).\n\
         # Wide is expected ~unaffected (~1.5%). CPU/GPU routing is unchanged."
    );
    if !std::env::var("LGBM_PHASE_PROF").map(|v| v == "1").unwrap_or(false) {
        println!(
            "# WARNING: LGBM_PHASE_PROF is not 1 — scan_resident sync counts will read 0.\n\
             #          Re-run with: LGBM_BENCH_COPACK_AB=1 LGBM_PHASE_PROF=1 cargo run \\\n\
             #            --release --features rocm --example bench_gpu_vs_cpu"
        );
    }
    println!(
        "{:<8} {:>8} {:>5} {:>5}  {:>12} {:>12} {:>9}  {:>12} {:>12} {:>7}  verdict",
        "size",
        "rows",
        "feat",
        "bins",
        "syncs_off",
        "syncs_on",
        "sync/tree",
        "train_off",
        "train_on",
        "off/on",
    );

    for s in sizes {
        let corpus = make_corpus(s);
        let cfg = cfg_for(s);

        // OFF: byte-unchanged two-separate-scans path. The override reads the env
        // per query (not memoized), so an in-process toggle is sufficient.
        unsafe { std::env::set_var("LGBM_SIBLING_COPACK", COPACK_OFF) };
        let (off_t, off_syncs) = timed_run(&cfg, &corpus, warmup, reps);

        // ON: co-pack engages whenever the structural correctness gate holds.
        unsafe { std::env::set_var("LGBM_SIBLING_COPACK", COPACK_ON) };
        let (on_t, on_syncs) = timed_run(&cfg, &corpus, warmup, reps);

        let off_s = off_t.as_secs_f64();
        let on_s = on_t.as_secs_f64();
        let ratio = if on_s > 0.0 { off_s / on_s } else { f64::NAN };
        // Sync-count per tree (per-train / iters) — the ~59 -> ~30 SC-3 signal.
        let on_per_tree = if iters > 0 { on_syncs as f64 / iters as f64 } else { 0.0 };
        // Verdict: NOT-SLOWER when ON is within ~3% noise of OFF or faster;
        // trends-faster when ON is faster beyond that noise band. SIGN-only.
        let verdict = if ratio.is_nan() {
            "n/a"
        } else if ratio >= 1.03 {
            "trends-faster"
        } else if ratio >= 0.97 {
            "NOT-SLOWER"
        } else {
            "SLOWER(noise? rerun)"
        };

        println!(
            "{:<8} {:>8} {:>5} {:>5}  {:>12} {:>12} {:>9.1}  {:>10.2?} {:>10.2?} {:>7.3}  {}",
            s.name.trim(),
            s.rows,
            s.features,
            s.bins,
            off_syncs,
            on_syncs,
            on_per_tree,
            off_t,
            on_t,
            ratio,
            verdict,
        );
    }
    // Reset the override so we don't leak A/B state into any later run.
    unsafe { std::env::remove_var("LGBM_SIBLING_COPACK") };
    println!(
        "# DIAGNOSIS: SC-3 (sync drop) is structural + counter-exact (syncs_on ~= syncs_off/2,\n\
         #   ~30/tree vs ~59/tree). SC-4 (off/on >= 1) is SIGN-ONLY + Amdahl-capped (~10–15%\n\
         #   small/medium, ~1.5% wide) — the isolated 2× is NOT the e2e number. Run >=2\n\
         #   processes and judge the SIGN; the absolute magnitude is APU-confounded."
    );
}

/// CPU-only stub: the resident co-pack path never fires on the f64 anchor, so the
/// A/B is GPU-only. Mirrors the harness's `--features rocm` gating.
#[cfg(not(feature = "rocm"))]
fn run_copack_ab(_sizes: &[Size], _cfg_for: &dyn Fn(&Size) -> lgbm::Config, _iters: i32, _warmup: usize, _reps: usize) {
    println!(
        "\n# === Phase 12 co-pack ON/OFF A/B requested (LGBM_BENCH_COPACK_AB=1) ===\n\
         # SKIPPED: this CPU-only build has no resident histogram pool, so the co-pack\n\
         # scan path never fires (CpuBackend f64 anchor). Rebuild with --features rocm:\n\
         #   LGBM_BENCH_COPACK_AB=1 LGBM_PHASE_PROF=1 cargo run --release \\\n\
         #     --features rocm --example bench_gpu_vs_cpu"
    );
}

fn main() {
    // The backend is selected at COMPILE TIME by `--features rocm`.
    #[cfg(feature = "rocm")]
    let backend = "rocm(gfx1100)";
    #[cfg(not(feature = "rocm"))]
    let backend = "cpu-f64-anchor";

    // Standard regression GBDT, same config for both backends.
    const LEAVES: i32 = 31;
    let sweep = std::env::var("LGBM_BENCH_SWEEP").ok();
    // WARM-VS-COLD RULE: discard WARMUP iters before timing; report median of TRAIN_REPS.
    // `wide` mode (spike-014: 1M×500 attribution) trains very heavy corpora, so it uses a
    // lighter warmup/rep budget to keep the wall time bounded while still warm-amortizing —
    // the per-phase RATIO (the spike's deliverable) is stable under fewer reps.
    let (warmup, train_reps) = match sweep.as_deref() {
        Some("wide") => (1usize, 3usize),
        _ => (2usize, 5usize),
    };
    // Sweep mode trains larger corpora; fewer iters keep the multi-restart A/B wall time
    // bounded while still warm-amortizing. Default keeps 50.
    let iters: i32 = match sweep.as_deref() {
        Some("medocc") => 12,
        Some("wide") => std::env::var("LGBM_BENCH_ITERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8),
        _ => 50,
    };

    // Sizes: small/medium are launch-bound on the GPU; the >=200k size is where the
    // GPU path stops being purely launch-bound (and where the CPU anchor goes
    // multi-threaded). Fewer iters at 200k to keep the run time reasonable.
    //
    // SWEEP MODE (quick-260620-sqf): when `LGBM_BENCH_SWEEP=medocc` is set, replace
    // the default sizes with the medium-width occupancy A/B grid — feature counts
    // {30,60,100} × rows {50k,120k,256k,512k} bracketing the 256k row-partition gate.
    // The A/B (P=1 vs row-partitioned) is driven externally via `LGBM_ROWPART_MIN`
    // (the existing override at histogram.rs:731) so NO source change to the gate is
    // needed for the measurement. Methodology (warm/median) is unchanged.
    let default_sizes = vec![
        Size { name: "small ", rows: 2_000, features: 12, bins: 32 },
        Size { name: "medium", rows: 20_000, features: 30, bins: 64 },
        Size { name: "large ", rows: 200_000, features: 40, bins: 128 },
    ];
    // WIDE MODE (spike-014a): the never-before-benched 1M×500 attribution shape, plus a
    // lighter sweep up to it so we read the wide-shape TREND, not one fragile point. A
    // single custom point can be forced with `LGBM_BENCH_ROWS` / `LGBM_BENCH_FEAT`
    // (e.g. ROWS=1000000 FEAT=500). bins fixed at 128 (matches the `large` default).
    let wide_sizes: Vec<Size> = {
        let feat = std::env::var("LGBM_BENCH_FEAT").ok().and_then(|v| v.parse().ok());
        let rows = std::env::var("LGBM_BENCH_ROWS").ok().and_then(|v| v.parse().ok());
        match (rows, feat) {
            (Some(r), Some(f)) => {
                let name: &'static str = Box::leak(format!("r{}kf{f}", r / 1000).into_boxed_str());
                vec![Size { name, rows: r, features: f, bins: 128 }]
            }
            _ => {
                let f = feat.unwrap_or(500);
                let mut v = Vec::new();
                for &rows in &[250_000usize, 500_000, 1_000_000] {
                    let name: &'static str =
                        Box::leak(format!("r{}kf{f}", rows / 1000).into_boxed_str());
                    v.push(Size { name, rows, features: f, bins: 128 });
                }
                v
            }
        }
    };
    let medocc_sizes: Vec<Size> = {
        let mut v = Vec::new();
        for &feat in &[30usize, 60, 100] {
            for &rows in &[50_000usize, 120_000, 256_000, 512_000] {
                // Leak a small static-ish name; bench example, fine to leak.
                let name: &'static str =
                    Box::leak(format!("f{feat}r{}k", rows / 1000).into_boxed_str());
                v.push(Size { name, rows, features: feat, bins: 128 });
            }
        }
        v
    };
    let sizes: Vec<Size> = match sweep.as_deref() {
        Some("medocc") => medocc_sizes,
        Some("wide") => wide_sizes,
        _ => default_sizes,
    };

    // The training config is identical for every size (and for the co-pack A/B's
    // OFF/ON arms — the ONLY variable is `LGBM_SIBLING_COPACK`).
    let cfg_for = |_s: &Size| -> lgbm::Config {
        TrainingBuilder::new()
            .objective("regression")
            .num_iterations(iters)
            .num_leaves(LEAVES)
            .learning_rate(0.1)
            .min_data_in_leaf(20)
            .seed(42)
            .deterministic(true)
            .build()
            .expect("config builds")
    };

    // Phase 12 (SC-3 + SC-4): co-pack ON/OFF A/B. Gated behind LGBM_BENCH_COPACK_AB=1
    // so the default bench output is unchanged. Covers the launch-bound small/medium
    // regimes (where the win lands) plus a wide regime (showing ~0 effect) — whichever
    // `sizes` the harness was configured with (default = small/medium/large;
    // LGBM_BENCH_SWEEP=wide = 250k/500k/1M × 500).
    if std::env::var("LGBM_BENCH_COPACK_AB").map(|v| v == "1").unwrap_or(false) {
        run_copack_ab(&sizes, &cfg_for, iters, warmup, train_reps);
        return;
    }

    println!(
        "# lightgbm_rs GPU-vs-CPU bench  (backend: {backend}, iters: {iters}, leaves: {LEAVES}, warmup: {warmup}, reps: {train_reps}, rowpart_min: {})",
        std::env::var("LGBM_ROWPART_MIN").unwrap_or_else(|_| "default(256000)".into())
    );
    println!(
        "# RUN BOTH FEATURE CONFIGS FOR THE SIDE-BY-SIDE COMPARISON:\n\
         #   CPU: cargo run --release --example bench_gpu_vs_cpu\n\
         #   GPU: cargo run --release --features rocm --example bench_gpu_vs_cpu\n\
         # (CPU column is the MULTI-THREADED f64 anchor; GPU build is atomic/latency-bound.)"
    );
    println!(
        "{:<7} {:>8} {:>5} {:>5}  {:>14} {:>14}",
        "size", "rows", "feat", "bins", "train_median", "train_rows/s"
    );

    for s in &sizes {
        let corpus = make_corpus(s);
        let cfg = cfg_for(s);

        // ---- WARM-UP (discarded): amortize allocator/JIT so the timed loop is warm.
        for _ in 0..warmup {
            let _ = train(&cfg, &corpus).expect("warm-up train ok");
        }
        // Reset any phase_prof counters the warmup accumulated so the per-phase split
        // below reflects only the timed reps. Inert unless LGBM_PHASE_PROF=1.
        lgbm_treelearner::phase_prof::dump("warmup-discard");
        lgbm_compute::fusion_prof::dump_scan("warmup-discard");

        // ---- timed train (median of train_reps warm runs) ----
        let mut train_times = Vec::with_capacity(train_reps);
        let mut sink = 0.0f64; // defeat dead-code elimination
        for _ in 0..train_reps {
            let t0 = Instant::now();
            let booster = train(&cfg, &corpus).expect("train ok");
            train_times.push(t0.elapsed());
            // touch the model so the train isn't optimized away.
            sink += booster.predict(&corpus.features[0..1])[0][0] as f64;
        }
        std::hint::black_box(sink);
        // Per-phase wall-clock split accumulated over the timed reps (build/scan/partition).
        // The spike-014a deliverable: which phase dominates at the wide shape. Resets here.
        lgbm_treelearner::phase_prof::dump(s.name.trim());
        // spike-015: per-leaf GPU scan round-trip decomposition (inert unless LGBM_SCAN_PROF=1).
        lgbm_compute::fusion_prof::dump_scan(s.name.trim());
        let train_med = median(train_times);
        let rows_per_s = s.rows as f64 / train_med.as_secs_f64();

        println!(
            "{:<7} {:>8} {:>5} {:>5}  {:>12.2?} {:>14.0}",
            s.name, s.rows, s.features, s.bins, train_med, rows_per_s
        );
    }
}
