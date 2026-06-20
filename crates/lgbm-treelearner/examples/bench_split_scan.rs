//! Quick-task 260620-8v4 — SCAN-fraction measurement harness (go/no-go gate).
//!
//! Quantifies what fraction of per-leaf HISTSPLIT time is the **SCAN**
//! (`find_best_split` / split-finding) versus **BUILD** (per-feature histogram
//! construction), using the EXISTING `lgbm_treelearner::phase_prof` counters —
//! no new timing mechanism is invented. The full production training path is
//! driven through the facade `lgbm::train`, so `BUILD_NS` / `SCAN_NS` accumulate
//! over the exact per-leaf hot path production uses
//! (`build_leaf_histogram_into` → `BUILD_NS`, `scan_leaf_histogram` → `SCAN_NS`).
//!
//! The candidate optimisation under audit (260620-8v4) is a 2-lane parallel
//! REVERSE/FORWARD split scan. It can at best HALVE the scan, so the per-leaf
//! ceiling on its win is `0.5 * SCAN/(BUILD+SCAN)`. The go/no-go threshold is
//! therefore SCAN ≥ 20% of (BUILD+SCAN) on the wide config (a <20% scan caps the
//! achievable per-leaf win at <10%, below this project's bar for a parity-risky
//! additive kernel variant).
//!
//! Run (cubecl-cpu f64 anchor — the authoritative gate; no GPU needed):
//!   LGBM_PHASE_PROF=1 cargo run --release -p lgbm-treelearner --example bench_split_scan
//!
//! Run on the gfx1100 ROCm GPU (informational; non-blocking):
//!   LGBM_PHASE_PROF=1 cargo run --release --features rocm -p lgbm-treelearner --example bench_split_scan
//!
//! The harness is deterministic (no RNG; identity-binned synthetic corpora) and
//! re-runnable. It discards a cold warmup iteration before the measured warm
//! window (cold-ceiling overstates warm, per project memory
//! `vec-vec-optimisation-spikes-010-013`). Each config prints a `SUMMARY` line
//! the FINDINGS.md quotes verbatim.

use std::sync::atomic::Ordering;
use std::time::Instant;

use lgbm::{train, DenseCorpus, TrainingBuilder};
use lgbm_treelearner::phase_prof::{BUILD_NS, SCAN_NS};

/// A benchmark size: `rows` × `features`, each feature identity-binned into
/// `bins` distinct integer values.
struct Size {
    name: &'static str,
    rows: usize,
    features: usize,
    bins: usize,
}

/// Deterministic identity-binned corpus — the SAME recipe as `bench_train.rs` /
/// `bench_crossover.rs` (no RNG → fully reproducible). Each feature `f` takes
/// integer values in `0..bins`; the first `bins` rows are forced to
/// `0,1,..,bins-1` to guarantee full bin coverage; the rest are a fixed integer
/// hash. The label is a deterministic linear combination so trees have real
/// structure (and the scan actually finds splits).
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

/// quick 260620-njg Task-2: isolated A/B of the f64-pregather fold lever.
///
/// Builds a synthetic leaf (rows x features, narrow u32 bins) and folds the whole
/// feature set repeatedly, two ways:
///   A = f32 ord + `f64::from(ord_g[k])` widen INSIDE the per-feature loop (current).
///   B = pre-widened `Vec<f64>` ord (widened ONCE per leaf), read directly in the loop.
/// Both produce byte-identical f64 histograms (asserted). Reports 3-run medians of the
/// fold wall for each, isolating the conversion-elimination vs the cache-density penalty
/// of the 2x-bytes f64 ord arrays. Throwaway — no production change.
fn njg_pregather_ab() {
    use std::hint::black_box;

    let rows = 20_000usize;
    let bins = 255usize;
    // Sweep the gate-relevant widths; per-leaf the SAME ord arrays are folded by every
    // feature, so the lever's effect scales with feature count.
    for &features in &[60usize, 90, 120] {
        // Synthetic narrow bin columns (u32, < bins) — one Vec<u32> per feature.
        let cols: Vec<Vec<u32>> = (0..features)
            .map(|f| {
                (0..rows)
                    .map(|row| {
                        let h = row
                            .wrapping_mul(2_654_435_761)
                            .wrapping_add(f.wrapping_mul(40_503).wrapping_add(0x9E37_79B9));
                        (h % bins) as u32
                    })
                    .collect()
            })
            .collect();
        // Once-gathered ordered grad/hess (f32, spike-003) + the pregathered f64 variant.
        let ord_g: Vec<f32> = (0..rows).map(|k| ((k % 97) as f32) * 0.013 - 0.6).collect();
        let ord_h: Vec<f32> = (0..rows).map(|k| ((k % 31) as f32) * 0.007 + 1.0).collect();
        let ord_g64: Vec<f64> = ord_g.iter().map(|&x| f64::from(x)).collect();
        let ord_h64: Vec<f64> = ord_h.iter().map(|&x| f64::from(x)).collect();

        let cells = 2 * bins;
        const REPS: usize = 3;
        const INNER: usize = 20; // fold the whole feature set INNER times per measured rep.

        // ---- variant A: f32 ord + per-feature widen-in-loop.
        let fold_a = || {
            let mut acc = 0.0f64;
            for col in &cols {
                let mut h = vec![0.0f64; cells];
                for (k, &bin) in col.iter().enumerate() {
                    let b = bin as usize;
                    h[b * 2] += f64::from(ord_g[k]);
                    h[b * 2 + 1] += f64::from(ord_h[k]);
                }
                acc += h[0] + h[cells - 1];
                black_box(&h);
            }
            acc
        };
        // ---- variant B: pre-widened f64 ord, read directly.
        let fold_b = || {
            let mut acc = 0.0f64;
            for col in &cols {
                let mut h = vec![0.0f64; cells];
                for (k, &bin) in col.iter().enumerate() {
                    let b = bin as usize;
                    h[b * 2] += ord_g64[k];
                    h[b * 2 + 1] += ord_h64[k];
                }
                acc += h[0] + h[cells - 1];
                black_box(&h);
            }
            acc
        };

        // Bit-exact check: A and B must produce identical f64 (lossless widen).
        assert_eq!(fold_a().to_bits(), fold_b().to_bits(), "njg A/B fold not bit-exact");

        // Cold warmup (discarded), then 3 measured reps each; report medians.
        black_box(fold_a());
        black_box(fold_b());
        let mut ta = [0.0f64; REPS];
        let mut tb = [0.0f64; REPS];
        for r in 0..REPS {
            let t0 = Instant::now();
            for _ in 0..INNER {
                black_box(fold_a());
            }
            ta[r] = t0.elapsed().as_secs_f64() * 1e3;
            let t1 = Instant::now();
            for _ in 0..INNER {
                black_box(fold_b());
            }
            tb[r] = t1.elapsed().as_secs_f64() * 1e3;
        }
        let med = |mut v: [f64; REPS]| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[REPS / 2]
        };
        let (ma, mb) = (med(ta), med(tb));
        let delta = (ma - mb) / ma * 100.0;
        println!(
            "njg_ab feat={features} A(f32+widen)_med={ma:.3}ms B(f64-pregather)_med={mb:.3}ms \
             B_vs_A={delta:+.2}%  A_runs={ta:?} B_runs={tb:?}"
        );
    }
    println!(
        "njg_ab NOTE: cold microbench overstates warm 3-7x; ship ONLY on a sign-stable \
         bench_train/bench_real train-wall win, never on this isolated fold."
    );
}

fn main() {
    #[cfg(feature = "rocm")]
    let backend = "ROCm-GPU(gfx1100,f32-mirror)";
    #[cfg(not(feature = "rocm"))]
    let backend = "cubecl-cpu(f64-anchor)";

    if std::env::var("LGBM_PHASE_PROF").ok().as_deref() != Some("1") {
        eprintln!(
            "WARNING: LGBM_PHASE_PROF is not set to 1 — BUILD_NS/SCAN_NS will be \
             zero. Re-run with `LGBM_PHASE_PROF=1`."
        );
    }

    // Fixed GBDT config — a standard regression learner. Width is the variable
    // of interest (per-feature single-lane scan cost scales with num_bin; the
    // cube-per-feature parallelism scales with num_features), so we sweep narrow
    // vs wide at a fixed row count.
    const ITERS: i32 = 30;
    const LEAVES: i32 = 31;
    // Warm window: number of MEASURED train repeats after the discarded cold one.
    const WARM_REPS: usize = 3;

    // narrow ~10 features, wide ~120 features (>100, per the plan). Same rows &
    // bins so only width changes between the two rows.
    let sizes = [
        Size { name: "narrow", rows: 20_000, features: 10, bins: 255 },
        Size { name: "wide", rows: 20_000, features: 120, bins: 255 },
    ];

    println!(
        "# bench_split_scan  backend={backend}  iters={ITERS}  leaves={LEAVES}  \
         warm_reps={WARM_REPS}  (cold iteration discarded)"
    );
    println!(
        "{:<7} {:>7} {:>5} {:>5}  {:>11} {:>11}  {:>9}  {:>9}",
        "config", "rows", "feat", "bins", "build_ms", "scan_ms", "scan%", "verdict"
    );

    // Collect the wide-config scan fraction for the explicit go/no-go line.
    let mut wide_scan_pct: Option<f64> = None;

    for s in &sizes {
        let corpus = make_corpus(s);
        let cfg = TrainingBuilder::new()
            .objective("regression")
            .num_iterations(ITERS)
            .num_leaves(LEAVES)
            .learning_rate(0.1)
            .min_data_in_leaf(20)
            .seed(42)
            .deterministic(true)
            .build()
            .expect("config builds");

        // ---- cold warmup (discarded): touches allocator/pool/code paths so the
        // measured window reflects warm steady-state, not first-touch ceiling.
        let cold = train(&cfg, &corpus).expect("train ok (cold)");
        std::hint::black_box(&cold);

        // Reset the per-phase counters AFTER the cold run so the measured window
        // starts from zero (the counters are process-global accumulators).
        BUILD_NS.store(0, Ordering::Relaxed);
        SCAN_NS.store(0, Ordering::Relaxed);

        // ---- warm window: WARM_REPS measured train repeats. BUILD_NS/SCAN_NS
        // accumulate across all of them; the RATIO is rep-count-invariant.
        let t0 = Instant::now();
        for _ in 0..WARM_REPS {
            let b = train(&cfg, &corpus).expect("train ok (warm)");
            std::hint::black_box(&b);
        }
        let wall = t0.elapsed();

        let build = BUILD_NS.load(Ordering::Relaxed);
        let scan = SCAN_NS.load(Ordering::Relaxed);
        let denom = build + scan;
        let scan_pct = if denom > 0 { scan as f64 * 100.0 / denom as f64 } else { 0.0 };

        // Per-config go/no-go evaluation (the wide row drives the verdict; narrow
        // is reported for context — the scan fraction is width-dependent).
        let verdict = if scan_pct >= 20.0 { "GO?" } else { "null" };

        println!(
            "{:<7} {:>7} {:>5} {:>5}  {:>11.3} {:>11.3}  {:>8.2}%  {:>9}",
            s.name,
            s.rows,
            s.features,
            s.bins,
            build as f64 / 1e6,
            scan as f64 / 1e6,
            scan_pct,
            verdict
        );
        // A machine-greppable line the FINDINGS.md quotes verbatim.
        println!(
            "SCAN_NS={scan} BUILD_NS={build} SCAN%={scan_pct:.2} \
             config={} feat={} backend={backend} warm_wall={:.3}ms",
            s.name,
            s.features,
            wall.as_secs_f64() * 1e3
        );

        if s.name == "wide" {
            wide_scan_pct = Some(scan_pct);
        }
    }

    // ---- quick 260620-dpk Task-1: per-leaf FIXED-COST decomposition sweep.
    // Drives the FORCED-ON unified path (set LGBM_UNIFIED_BFS_THRESHOLD=0
    // LGBM_UNIFIED_SUBSCAN_THRESHOLD=0 LGBM_FUSION_PROF=1) across 20/40/60/80 features
    // and dumps the fusion_prof buckets (gather / alloc / par) per feature count so the
    // FINDINGS table can quantify ALLOC (lever-A-reducible) vs FORK/JOIN-floor fractions.
    if std::env::var("LGBM_FUSION_PROF").ok().as_deref() == Some("1") {
        println!("# --- 260620-dpk per-leaf fixed-cost decomposition (FORCED-ON unified) ---");
        // quick 260620-njg: representative widths 60/90/120 feat (~20k rows) so the
        // BUILD/FIX+COMPACT/SCAN WORK split is measured at the gate-relevant width.
        for feat in [60usize, 90, 120] {
            let s = Size { name: "dpk", rows: 20_000, features: feat, bins: 255 };
            let corpus = make_corpus(&s);
            let cfg = TrainingBuilder::new()
                .objective("regression")
                .num_iterations(ITERS)
                .num_leaves(LEAVES)
                .learning_rate(0.1)
                .min_data_in_leaf(20)
                .seed(42)
                .deterministic(true)
                .build()
                .expect("config builds");

            // Cold warmup (discarded), then reset the fusion buckets so the dump reflects
            // ONLY the measured warm window.
            let cold = train(&cfg, &corpus).expect("train ok (cold dpk)");
            std::hint::black_box(&cold);
            lgbm_compute::fusion_prof::BFS_GATHER_NS.store(0, Ordering::Relaxed);
            lgbm_compute::fusion_prof::BFS_ALLOC_NS.store(0, Ordering::Relaxed);
            lgbm_compute::fusion_prof::BFS_PAR_NS.store(0, Ordering::Relaxed);
            // quick 260620-njg: reset the BUILD/FIX+COMPACT/SCAN sub-buckets too.
            lgbm_compute::fusion_prof::BFS_BUILD_NS.store(0, Ordering::Relaxed);
            lgbm_compute::fusion_prof::BFS_FIXCOMPACT_NS.store(0, Ordering::Relaxed);
            lgbm_compute::fusion_prof::BFS_SCAN_NS.store(0, Ordering::Relaxed);
            lgbm_compute::fusion_prof::SUB_ALLOC_NS.store(0, Ordering::Relaxed);
            lgbm_compute::fusion_prof::SUB_PAR_NS.store(0, Ordering::Relaxed);

            let t0 = Instant::now();
            for _ in 0..WARM_REPS {
                let b = train(&cfg, &corpus).expect("train ok (warm dpk)");
                std::hint::black_box(&b);
            }
            let wall = t0.elapsed();
            println!("dpk_feat={feat} warm_wall={:.3}ms", wall.as_secs_f64() * 1e3);
            lgbm_compute::fusion_prof::dump(&format!("dpk_feat{feat}"));
        }
    } else {
        eprintln!(
            "NOTE: set LGBM_FUSION_PROF=1 (with LGBM_UNIFIED_BFS_THRESHOLD=0 \
             LGBM_UNIFIED_SUBSCAN_THRESHOLD=0) to run the 260620-dpk fixed-cost sweep."
        );
    }

    // ---- quick 260620-njg Task-2: throwaway isolated A/B of the f64-pregather lever.
    // Task 1 showed BUILD (the fold) is ~66-70% of WORK, so the fold body is the only
    // place a cheaper-work lever can land. The lever: drop the per-feature
    // `f64::from(ord_g[k])` widening (recomputed num_features times per row) by
    // pre-widening the once-gathered ord arrays to Vec<f64> at gather time. This A/B
    // isolates EXACTLY that — two byte-identical fold loops over the SAME synthetic bin
    // columns, differing only in (A) read f32 + widen-in-loop vs (B) read pre-widened
    // f64. Bit-exact (f32->f64 widen is lossless ⇒ identical f64 addends in identical
    // order); asserted. Gated on LGBM_NJG_AB=1 (throwaway, no hot-path plumbing).
    if std::env::var("LGBM_NJG_AB").ok().as_deref() == Some("1") {
        njg_pregather_ab();
    }

    // Explicit, numeric go/no-go line (threshold = 20% on the WIDE config).
    if let Some(pct) = wide_scan_pct {
        let go = pct >= 20.0;
        println!(
            "SUMMARY: wide SCAN%={pct:.2}  threshold=20.00  verdict={}  \
             (2-lane lever ceiling on per-leaf win ≈ {:.2}%)",
            if go { "GO" } else { "NO-GO(NULL)" },
            pct / 2.0
        );
    }
}
