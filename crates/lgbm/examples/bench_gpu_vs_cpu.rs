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

/// RAII guard for the `LGBM_SIBLING_COPACK` A/B override (WR-04).
///
/// `std::env::set_var`/`remove_var` are `unsafe` because they are not
/// thread-safe; this guard keeps the env-toggle path but makes the reset
/// panic-safe: on `Drop` the var is always removed, so an ON-arm panic before
/// the explicit reset can never leak the `1` state forward into later sizes.
///
/// Note on thread-safety: `set_var` is still only sound here because the
/// override is read by `sibling_copack_override()` on the main growth thread
/// BEFORE any cubecl/rayon parallel region spawns workers that read the env.
/// The guard does not change that invariant; it only fixes the panic-leak.
#[cfg(feature = "rocm")]
struct CopackEnvGuard;

#[cfg(feature = "rocm")]
impl CopackEnvGuard {
    /// Set `LGBM_SIBLING_COPACK` to `value` and return a guard that removes it
    /// on drop.
    fn set(value: &str) -> Self {
        // SAFETY: see the type-level note — only read on the main growth thread
        // before parallel regions spawn.
        unsafe { std::env::set_var("LGBM_SIBLING_COPACK", value) };
        CopackEnvGuard
    }
}

#[cfg(feature = "rocm")]
impl Drop for CopackEnvGuard {
    fn drop(&mut self) {
        // SAFETY: same single-main-thread invariant as `set`.
        unsafe { std::env::remove_var("LGBM_SIBLING_COPACK") };
    }
}

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
///
/// Returns the UPPER-middle element for an even-length input (the element at
/// `len / 2`, not the mean of the two centers) — adequate for `reps >= 3` warm
/// medians where the deliverable is the per-phase ratio, not a precise central
/// tendency. Returns `Duration::ZERO` on empty input so an operator-tuned
/// `reps == 0` cannot panic (IN-04).
fn median(mut ds: Vec<Duration>) -> Duration {
    if ds.is_empty() {
        return Duration::ZERO;
    }
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
        // per query (not memoized), so an in-process toggle is sufficient. The
        // RAII guard (WR-04) guarantees the var is cleared even if `timed_run`
        // panics, so ON-state can never leak forward into a later size.
        let (off_t, off_syncs) = {
            let _g = CopackEnvGuard::set(COPACK_OFF);
            timed_run(&cfg, &corpus, warmup, reps)
        };

        // ON: co-pack engages whenever the structural correctness gate holds.
        let (on_t, on_syncs) = {
            let _g = CopackEnvGuard::set(COPACK_ON);
            timed_run(&cfg, &corpus, warmup, reps)
        };

        let off_s = off_t.as_secs_f64();
        let on_s = on_t.as_secs_f64();
        let ratio = if on_s > 0.0 { off_s / on_s } else { f64::NAN };
        // Sync-count per tree (per-train / iters) — the ~59 -> ~30 SC-3 signal.
        let on_per_tree = if iters > 0 { on_syncs as f64 / iters as f64 } else { 0.0 };
        // Verdict (IN-03): SIGN-ONLY on a single-process median. The file header
        // demands >=2 PROCESSES for sign-stability, and on the spoofed 8-CU APU a
        // single-process median can swing beyond 3%, so the band is deliberately
        // wide and the verdict is labelled "(single-proc, sign-only)" — it reads
        // the TREND, not a confidence-bearing pass/fail. For a real verdict,
        // compute the ratio across >=2 processes.
        let verdict = if ratio.is_nan() {
            "n/a"
        } else if ratio >= 1.05 {
            "trends-faster (single-proc, sign-only)"
        } else if ratio >= 0.95 {
            "NOT-SLOWER (single-proc, sign-only)"
        } else {
            "SLOWER? (single-proc noise — rerun >=2 procs)"
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
    // Each arm's `CopackEnvGuard` already clears the override on drop (WR-04),
    // so no state leaks past the loop; this final remove is a belt-and-braces
    // reset in case the env was set before this function ran.
    unsafe { std::env::remove_var("LGBM_SIBLING_COPACK") };
    println!(
        "# DIAGNOSIS: SC-3 (sync drop) is structural + counter-exact (syncs_on ~= syncs_off/2,\n\
         #   ~30/tree vs ~59/tree). SC-4 (off/on >= 1) is SIGN-ONLY + Amdahl-capped (~10–15%\n\
         #   small/medium, ~1.5% wide) — the isolated 2× is NOT the e2e number. Run >=2\n\
         #   processes and judge the SIGN; the absolute magnitude is APU-confounded."
    );
}

// ===========================================================================
// Phase 13 (13-04) — autotune-ON vs heuristic (LGBM_AUTOTUNE=0) e2e A/B
// ===========================================================================
//
// Gated behind `LGBM_BENCH_AUTOTUNE_AB=1` (default bench output unchanged). GPU-only:
// autotune drives the rocm launch-config (build row-partition `P` + scan `CubeDim`);
// the CpuBackend f64 anchor has no GPU launch to tune.
//
//   LGBM_BENCH_AUTOTUNE_AB=1 \
//     cargo run --release --features rocm --example bench_gpu_vs_cpu
//
// HONEST BOUND (load-bearing — do NOT misread as an absolute-speed record):
//   * SIGN-ONLY: this is a spoofed 8-CU APU (gfx1152, HSA-overridden); absolute
//     wall-clock is APU-confounded. We report median device-time for BOTH arms +
//     the ratio t(heur)/t(auto) (>= 1 expected: autotune is NOT-SLOWER) and judge the
//     SIGN only. Run >=2 PROCESS restarts for sign-stability. NO pass/fail gate.
//   * The ~10% autotune win (spike-040) is on the GPU BUILD, which the 16-core CPU
//     beats end-to-end on THIS box. The durable deliverable is the METHOD
//     (measure-don't-model) + portability to real discrete GPUs (gfx110x / NVIDIA
//     self-calibrate with zero re-tuning), NOT a local e2e train-time win here.
//   * The heuristic `row_partition_count(50, n)` under-partitions to P=1 at the
//     production 50-feature width (spike-040); autotune recovers a P != 1. We read
//     the autotune-selected build P from the persisted cache to SHOW that recovery.

/// Set an env var for the lifetime of the guard, removing it on drop (panic-safe). The
/// autotune off-switch / pins are read FRESH per launch on the main growth thread before
/// any cubecl/rayon parallel region, so an in-process toggle is sound (mirrors
/// `CopackEnvGuard`).
#[cfg(feature = "rocm")]
struct ScopedEnv(&'static str);
#[cfg(feature = "rocm")]
impl ScopedEnv {
    fn set(key: &'static str, val: &str) -> Self {
        // SAFETY: single main-thread toggle before parallel regions spawn.
        unsafe { std::env::set_var(key, val) };
        ScopedEnv(key)
    }
}
#[cfg(feature = "rocm")]
impl Drop for ScopedEnv {
    fn drop(&mut self) {
        // SAFETY: same single-main-thread invariant as `set`.
        unsafe { std::env::remove_var(self.0) };
    }
}

/// Read the autotune-selected build row-partition `P`(s) from the persisted cache log
/// (`target/autotune/0.10.0/rocm_0/*build*.json.log`). Returns `(key, P)` per cached
/// entry, reading the WINNING tunable's `"name":"build_P{p}"` at `fastest_index` directly
/// (more robust than index→PSET mapping). (Mirrors spike-040's `read_winner`.) Cache line
/// shape: `{"key":{"key":{"bucket":B,"feats":F,"bins":N},..},"value":{"fastest_index":I,
/// "results":[{"outcome":{"Ok":{"name":"build_P16","index":3,..}}},..]}}`.
#[cfg(feature = "rocm")]
fn read_autotune_build_picks() -> Vec<(String, u32)> {
    let dir = "target/autotune/0.10.0/rocm_0";
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return out;
    };
    // Pull the bin/feat/bucket key object: from `{"bucket":` to the first closing `}`.
    let parse_key = |line: &str| -> String {
        line.find("{\"bucket\":")
            .and_then(|i| line[i..].find('}').map(|j| line[i..i + j + 1].to_string()))
            .unwrap_or_else(|| "?".into())
    };
    // Find the winning `build_P{p}` name whose `"index":N` equals fastest_index.
    let parse_winner_p = |line: &str| -> Option<u32> {
        let fi: usize = line
            .split("\"fastest_index\":")
            .nth(1)?
            .trim_start()
            .split([',', '}'])
            .next()?
            .parse()
            .ok()?;
        // Scan each `"name":"build_P{p}",...,"index":{n}` result; match n == fi.
        for seg in line.split("\"name\":\"build_P").skip(1) {
            let p: u32 = seg.split('"').next()?.parse().ok()?;
            if let Some(idx_str) = seg.split("\"index\":").nth(1) {
                let idx: usize = idx_str.trim_start().split([',', '}']).next()?.parse().ok()?;
                if idx == fi {
                    return Some(p);
                }
            }
        }
        None
    };
    for e in rd.flatten() {
        // The BUILD tuner's on-disk cache file carries the local_tuner!("build") name;
        // the scan tuners are "scan" / "scan_siblings" — filter to the build family.
        let fname = e.file_name().to_string_lossy().to_string();
        if !fname.contains("build") {
            continue;
        }
        let Ok(txt) = std::fs::read_to_string(e.path()) else {
            continue;
        };
        for line in txt.lines() {
            if let Some(p) = parse_winner_p(line) {
                out.push((parse_key(line), p));
            }
        }
    }
    out
}

/// Phase 13 (13-04): autotune-ON vs heuristic (`LGBM_AUTOTUNE=0`) e2e A/B at the
/// production ~50-feature width. GPU-only; SIGN-ONLY reporting (see the section header
/// for the honest bound). Returns having printed both medians, the ratio, and the
/// autotune-recovered build `P`.
#[cfg(feature = "rocm")]
fn run_autotune_ab(warmup: usize, reps: usize) {
    println!("\n# === Phase 13 autotune-ON vs heuristic A/B (LGBM_AUTOTUNE unset vs =0) ===");
    println!(
        "# HONEST BOUND: SIGN-ONLY on this spoofed 8-CU APU (absolute wall-clock confounded).\n\
         # Expect t(heur)/t(auto) >= 1 (autotune NOT-SLOWER); run >=2 PROCESS restarts for\n\
         # sign-stability; NO pass/fail gate. The ~10% win is on the GPU BUILD (which the\n\
         # 16-core CPU beats e2e here) — the durable deliverable is the METHOD + portability\n\
         # to discrete gfx110x/NVIDIA, not a local e2e number. The heuristic under-partitions\n\
         # to P=1 at 50 feats (spike-040); autotune recovers a P != 1 (read from the cache)."
    );

    // Production-width train: 50 features, a >=256k-row leaf regime so the build
    // row-partition `P` actually matters (below ROWPART_MIN_LEAF the heuristic is P=1
    // regardless). Deterministic identity-binned corpus (reproducible across arms).
    let s = Size { name: "prod50", rows: 300_000, features: 50, bins: 128 };
    let corpus = make_corpus(&s);
    const LEAVES: i32 = 31;
    const ITERS: i32 = 12; // bounded wall-time; the per-arm median ratio is the signal.
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

    // ---- AUTO arm: autotune default-on. Clear the persisted cache first so the pick is
    //      a FRESH cold tune we can read back (warm hits are ~µs; cold ~300–500ms/key).
    let _ = std::fs::remove_dir_all("target/autotune");
    // Belt-and-braces: ensure no stale LGBM_AUTOTUNE=0 leaks in from the environment.
    unsafe { std::env::remove_var("LGBM_AUTOTUNE") };
    let (auto_t, _auto_syncs) = timed_run(&cfg, &corpus, warmup, reps);
    let picks = read_autotune_build_picks();

    // ---- HEUR arm: LGBM_AUTOTUNE=0 → the `row_partition_count` heuristic (P=1 here).
    let (heur_t, _heur_syncs) = {
        let _g = ScopedEnv::set("LGBM_AUTOTUNE", "0");
        timed_run(&cfg, &corpus, warmup, reps)
    };

    let auto_s = auto_t.as_secs_f64();
    let heur_s = heur_t.as_secs_f64();
    let ratio = if auto_s > 0.0 { heur_s / auto_s } else { f64::NAN };
    let verdict = if ratio.is_nan() {
        "n/a"
    } else if ratio >= 1.03 {
        "autotune FASTER (single-proc, sign-only)"
    } else if ratio >= 0.97 {
        "NOT-SLOWER (single-proc, sign-only)"
    } else {
        "SLOWER? (single-proc noise — rerun >=2 procs)"
    };

    println!(
        "{:<8} {:>8} {:>5} {:>5}  {:>12} {:>12} {:>9}  verdict",
        "size", "rows", "feat", "bins", "train_heur", "train_auto", "heur/auto",
    );
    println!(
        "{:<8} {:>8} {:>5} {:>5}  {:>10.2?} {:>10.2?} {:>9.3}  {}",
        s.name.trim(),
        s.rows,
        s.features,
        s.bins,
        heur_t,
        auto_t,
        ratio,
        verdict,
    );

    // Show the autotune-recovered build P (the spike-040 P=1 under-partition recovery).
    if picks.is_empty() {
        println!(
            "# autotune build pick: <none persisted> — the leaf regime may have stayed below\n\
             #   ROWPART_MIN_LEAF (P=1 path) or the cache dir was unwritable. Increase rows/iters."
        );
    } else {
        let recovered = picks.iter().any(|(_, p)| *p > 1);
        for (key, p) in &picks {
            println!("# autotune build pick: key={key} -> P{p}");
        }
        println!(
            "# RECOVERY: autotune selected {} (heuristic = P1 at 50 feats, spike-040). {}",
            if recovered { "a P != 1" } else { "P1" },
            if recovered {
                "It recovered the under-partition."
            } else {
                "No P>1 recovered this run (rerun; the P4-P16 curve is flat/noisy on the APU)."
            }
        );
    }
    println!(
        "# DIAGNOSIS: judge the SIGN of heur/auto (>= 1 expected) across >=2 PROCESS restarts.\n\
         #   Absolute magnitude is APU-confounded; the durable deliverable is the autotune\n\
         #   SELECTION method + portability, not a local e2e speed record."
    );
}

/// CPU-only stub: autotune drives the rocm GPU launch-config only, so the A/B is GPU-only.
#[cfg(not(feature = "rocm"))]
fn run_autotune_ab(_warmup: usize, _reps: usize) {
    println!(
        "\n# === Phase 13 autotune-ON vs heuristic A/B requested (LGBM_BENCH_AUTOTUNE_AB=1) ===\n\
         # SKIPPED: this CPU-only build has no GPU launch-config to autotune (CpuBackend f64\n\
         # anchor). Rebuild with --features rocm:\n\
         #   LGBM_BENCH_AUTOTUNE_AB=1 cargo run --release --features rocm --example bench_gpu_vs_cpu"
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

    // Phase 13 (13-04): autotune-ON vs heuristic e2e A/B. Gated behind
    // LGBM_BENCH_AUTOTUNE_AB=1 so the default bench output is unchanged. Uses its own
    // production-width corpus (50 feat × 300k rows) so the build row-partition `P`
    // actually matters; reports SIGN-ONLY + the autotune-recovered P (GPU-only).
    if std::env::var("LGBM_BENCH_AUTOTUNE_AB").map(|v| v == "1").unwrap_or(false) {
        run_autotune_ab(warmup, train_reps);
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
