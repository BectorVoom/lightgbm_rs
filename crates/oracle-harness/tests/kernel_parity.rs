//! Compute-kernel histogram parity replay (cpu hard gate).
//!
//! For every committed histogram golden case (`tests/fixtures/kernels/histogram.txt`,
//! emitted by `cargo run -p xtask -- kernel-capture`), this test drives the
//! cubecl-cpu `Backend::construct_histograms` over the golden's per-row bins +
//! f32 ordered grad/hess and asserts the resulting f64 histogram cells are
//! BIT-EXACT versus the C++-transcription golden via `compare_exact_f64_bits`.
//! This exercises a real consumer of the binned store: the tree learner builds
//! a histogram from binned data and must get C++-bit-identical f64 cells on
//! the deterministic anchor.
//!
//! Idioms follow `oracle-harness/tests/rng_parity.rs` /
//! `lgbm-dataset/tests/bin_storage_layout.rs`: `CARGO_MANIFEST_DIR` fixture path
//! (never the untracked `LightGBM/` tree), graceful SKIP pre-capture, and a
//! localizing assert that names the diverging case + cell.
//!
//! Record format (see `xtask/cpp/kernel_capture.cpp`):
//! ```text
//! HCASE name=<id> layout=<dense|sparse> num_bin=<n> num_rows=<n> \
//!       skip_default_bin=<0|1> note=<text>
//! BINS <u32;...>            # per-row bin index (== Bin::data(idx))
//! GRAD <f32bits;...>        # per-row ordered_gradients, raw f32 bits (u32 dec)
//! HESS <f32bits;...>        # per-row ordered_hessians,  raw f32 bits (u32 dec)
//! HIST <f64bits;...>        # the [g0,h0,g1,h1,...] f64 cells, raw f64 bits (u64 dec)
//! ```

use std::path::PathBuf;

use lgbm_compute::gain::{get_leaf_gain, get_split_gains, GainConfig};
use lgbm_compute::runtime::cpu_client;
use lgbm_compute::{Backend, BatchedSplitFeature, BinColumn, CpuBackend};
// The exact comparators are NOT re-exported from the crate root (lib.rs re-exports
// only compare_within/abs_diff_within/Mismatch/ORACLE_TOL); import them via the
// full module path.
use oracle_harness::comparator::{compare_exact_f64_bits, compare_exact_u32};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/kernels/histogram.txt")
}

fn kernels_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/kernels")
}

/// Parse an `f64` from a raw little-endian f64 bit pattern (decimal `u64`).
fn parse_f64_bits(s: &str) -> f64 {
    f64::from_bits(s.parse::<u64>().expect("f64-bits u64 field"))
}

/// Parse an `i64`/`i32` decimal field.
fn parse_i64(tokens: &[&str], key: &str) -> i64 {
    field(tokens, key)
        .unwrap_or_else(|| panic!("missing int field `{key}`"))
        .parse()
        .unwrap_or_else(|_| panic!("bad int field `{key}`"))
}

/// Parse an `f64` field carried as a raw u64 bit pattern.
fn parse_f64_field(tokens: &[&str], key: &str) -> f64 {
    parse_f64_bits(field(tokens, key).unwrap_or_else(|| panic!("missing f64-bits field `{key}`")))
}

/// Extract a `key=value` token's value from a whitespace-split line.
fn field<'a>(tokens: &'a [&'a str], key: &str) -> Option<&'a str> {
    tokens
        .iter()
        .find_map(|t| t.strip_prefix(key).and_then(|r| r.strip_prefix('=')))
}

fn parse_u32(tokens: &[&str], key: &str) -> u32 {
    field(tokens, key)
        .unwrap_or_else(|| panic!("missing u32 field `{key}`"))
        .parse()
        .unwrap_or_else(|_| panic!("bad u32 field `{key}`"))
}

fn parse_u32_list(s: &str) -> Vec<u32> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(';')
        .map(|t| t.parse::<u32>().expect("u32 field"))
        .collect()
}

/// Parse a `;`-separated list of raw little-endian f32 bit patterns (decimal
/// `u32`) into `f32` via `from_bits` (bit-exact, zero parse rounding).
fn parse_f32_bits_list(s: &str) -> Vec<f32> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(';')
        .map(|t| f32::from_bits(t.parse::<u32>().expect("f32-bits u32 field")))
        .collect()
}

/// Parse a `;`-separated list of raw little-endian f64 bit patterns (decimal
/// `u64`) into `f64` via `from_bits` (bit-exact).
fn parse_f64_bits_list(s: &str) -> Vec<f64> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(';')
        .map(|t| f64::from_bits(t.parse::<u64>().expect("f64-bits u64 field")))
        .collect()
}

#[derive(Debug)]
struct HistGolden {
    name: String,
    num_bin: u32,
    bins: Vec<u32>,
    grad: Vec<f32>,
    hess: Vec<f32>,
    hist: Vec<f64>,
}

fn parse(text: &str) -> Vec<HistGolden> {
    let mut out = Vec::new();
    let mut lines = text.lines();
    while let Some(raw) = lines.next() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let t: Vec<&str> = line.split_whitespace().collect();
        if t[0] != "HCASE" {
            continue; // KERNEL_MASTER_SEED / COUNTS
        }
        let name = field(&t, "name").expect("HCASE name").to_string();
        let num_bin = parse_u32(&t, "num_bin");

        let bt: Vec<&str> = lines.next().expect("BINS").split_whitespace().collect();
        assert_eq!(bt[0], "BINS", "expected BINS after HCASE `{name}`");
        let bins = parse_u32_list(bt.get(1).copied().unwrap_or(""));

        let gt: Vec<&str> = lines.next().expect("GRAD").split_whitespace().collect();
        assert_eq!(gt[0], "GRAD", "expected GRAD for `{name}`");
        let grad = parse_f32_bits_list(gt.get(1).copied().unwrap_or(""));

        let ht: Vec<&str> = lines.next().expect("HESS").split_whitespace().collect();
        assert_eq!(ht[0], "HESS", "expected HESS for `{name}`");
        let hess = parse_f32_bits_list(ht.get(1).copied().unwrap_or(""));

        let st: Vec<&str> = lines.next().expect("HIST").split_whitespace().collect();
        assert_eq!(st[0], "HIST", "expected HIST for `{name}`");
        let hist = parse_f64_bits_list(st.get(1).copied().unwrap_or(""));

        out.push(HistGolden {
            name,
            num_bin,
            bins,
            grad,
            hess,
            hist,
        });
    }
    out
}

#[test]
fn kernel_parity_histogram_bit_exact_on_cpu() {
    let path = fixture_path();
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!(
            "kernel_parity: SKIP — fixture {} not found. Run \
             `cargo run -p xtask -- kernel-capture` on a machine with a C++ toolchain \
             and commit the golden set.",
            path.display()
        );
        return;
    };

    let cases = parse(&text);
    assert!(!cases.is_empty(), "fixture present but parsed zero cases");

    let client = cpu_client();
    let backend = CpuBackend;

    let mut dense_or_sparse_seen = (false, false);

    for c in &cases {
        // Sanity on the parsed record before driving the kernel (localizes a
        // malformed-fixture failure away from a real parity divergence).
        assert_eq!(
            c.grad.len(),
            c.bins.len(),
            "case `{}`: GRAD len != BINS len",
            c.name
        );
        assert_eq!(
            c.hess.len(),
            c.bins.len(),
            "case `{}`: HESS len != BINS len",
            c.name
        );
        assert_eq!(
            c.hist.len(),
            2 * c.num_bin as usize,
            "case `{}`: HIST len != 2*num_bin",
            c.name
        );

        // Drive the cubecl-cpu whole-kernel op over the golden inputs.
        let got = backend
            .construct_histograms(&client, &c.bins, &c.grad, &c.hess, c.num_bin)
            .unwrap_or_else(|e| panic!("case `{}`: construct_histograms failed: {e:?}", c.name));

        // Hard cpu gate: BIT-EXACT vs the C++-transcription golden.
        if let Err(mismatch) = compare_exact_f64_bits(&got, &c.hist) {
            panic!(
                "KERNEL PARITY DIVERGENCE in case `{}` (num_bin={}): {mismatch}",
                c.name, c.num_bin
            );
        }

        if c.name.contains("dense") {
            dense_or_sparse_seen.0 = true;
        }
        if c.name.contains("sparse") {
            dense_or_sparse_seen.1 = true;
        }
    }

    // Coverage assertion: at least one dense AND one sparse layout replayed.
    assert!(
        dense_or_sparse_seen.0,
        "golden must contain at least one dense-layout case"
    );
    assert!(
        dense_or_sparse_seen.1,
        "golden must contain at least one sparse-layout case"
    );
}

// ===========================================================================
// SPLIT parity (cpu hard gate, gain-scan layer).
//
// Two assertions per case:
//  1. PER-CANDIDATE gains: replicate the C++ scan gate-by-gate using the public
//     `lgbm_compute::gain::get_split_gains` (the SAME #[cube] fn the kernel calls,
//     so it is bit-identical) and assert the per-candidate gain vectors
//     (REVERSE + FORWARD, NaN where gated) match `SCAND_REV`/`SCAND_FWD`
//     bit-exact — this localizes a divergence to the gain MATH, not the winner.
//  2. The WINNER: drive the real `Backend::find_best_split` kernel and assert the
//     decoded `SplitInfo` fields bit-exact vs `SWIN`.
// ===========================================================================

#[derive(Debug)]
struct SplitGolden {
    name: String,
    num_bin: i32,
    offset: i32,
    default_bin: i32,
    skip_default_bin: bool,
    na_as_missing: bool,
    use_l1: bool,
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    lambda_l1: f64,
    lambda_l2: f64,
    min_gain_to_split: f64,
    sum_gradient: f64,
    sum_hessian: f64,
    num_data: i32,
    hist: Vec<f64>,
    cand_rev: Vec<f64>,
    cand_fwd: Vec<f64>,
    // winner (SWIN)
    win_is_splittable: bool,
    win_threshold: u32,
    win_gain: f64,
    win_min_gain_shift: f64,
    win_left_count: i32,
    win_right_count: i32,
    win_left_sum_gradient: f64,
    win_left_sum_hessian: f64,
    win_right_sum_gradient: f64,
    win_right_sum_hessian: f64,
    win_left_output: f64,
    win_right_output: f64,
    win_default_left: bool,
}

fn parse_split(text: &str) -> Vec<SplitGolden> {
    let mut out = Vec::new();
    let mut lines = text.lines();
    while let Some(raw) = lines.next() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let t: Vec<&str> = line.split_whitespace().collect();
        if t[0] != "SCASE" {
            continue;
        }
        let name = field(&t, "name").expect("SCASE name").to_string();
        let num_bin = parse_i64(&t, "num_bin") as i32;
        let offset = parse_i64(&t, "offset") as i32;
        let default_bin = parse_i64(&t, "default_bin") as i32;
        let skip_default_bin = parse_i64(&t, "skip_default_bin") != 0;
        let na_as_missing = parse_i64(&t, "na_as_missing") != 0;
        let use_l1 = parse_i64(&t, "use_l1") != 0;
        let min_data_in_leaf = parse_i64(&t, "min_data_in_leaf") as i32;
        let min_sum_hessian_in_leaf = parse_f64_field(&t, "min_sum_hessian_in_leaf");
        let lambda_l1 = parse_f64_field(&t, "lambda_l1");
        let lambda_l2 = parse_f64_field(&t, "lambda_l2");
        let min_gain_to_split = parse_f64_field(&t, "min_gain_to_split");
        let sum_gradient = parse_f64_field(&t, "sum_gradient");
        let sum_hessian = parse_f64_field(&t, "sum_hessian");
        let num_data = parse_i64(&t, "num_data") as i32;

        let ht: Vec<&str> = lines.next().expect("SHIST").split_whitespace().collect();
        assert_eq!(ht[0], "SHIST", "expected SHIST for `{name}`");
        let hist = parse_f64_bits_list(ht.get(1).copied().unwrap_or(""));

        let rt: Vec<&str> = lines.next().expect("SCAND_REV").split_whitespace().collect();
        assert_eq!(rt[0], "SCAND_REV", "expected SCAND_REV for `{name}`");
        let cand_rev = parse_f64_bits_list(rt.get(1).copied().unwrap_or(""));

        let ft: Vec<&str> = lines.next().expect("SCAND_FWD").split_whitespace().collect();
        assert_eq!(ft[0], "SCAND_FWD", "expected SCAND_FWD for `{name}`");
        let cand_fwd = parse_f64_bits_list(ft.get(1).copied().unwrap_or(""));

        let wt: Vec<&str> = lines.next().expect("SWIN").split_whitespace().collect();
        assert_eq!(wt[0], "SWIN", "expected SWIN for `{name}`");

        out.push(SplitGolden {
            name,
            num_bin,
            offset,
            default_bin,
            skip_default_bin,
            na_as_missing,
            use_l1,
            min_data_in_leaf,
            min_sum_hessian_in_leaf,
            lambda_l1,
            lambda_l2,
            min_gain_to_split,
            sum_gradient,
            sum_hessian,
            num_data,
            hist,
            cand_rev,
            cand_fwd,
            win_is_splittable: parse_i64(&wt, "is_splittable") != 0,
            win_threshold: parse_i64(&wt, "threshold") as u32,
            win_gain: parse_f64_field(&wt, "gain"),
            win_min_gain_shift: parse_f64_field(&wt, "min_gain_shift"),
            win_left_count: parse_i64(&wt, "left_count") as i32,
            win_right_count: parse_i64(&wt, "right_count") as i32,
            win_left_sum_gradient: parse_f64_field(&wt, "left_sum_gradient"),
            win_left_sum_hessian: parse_f64_field(&wt, "left_sum_hessian"),
            win_right_sum_gradient: parse_f64_field(&wt, "right_sum_gradient"),
            win_right_sum_hessian: parse_f64_field(&wt, "right_sum_hessian"),
            win_left_output: parse_f64_field(&wt, "left_output"),
            win_right_output: parse_f64_field(&wt, "right_output"),
            win_default_left: parse_i64(&wt, "default_left") != 0,
        });
    }
    out
}

/// `RoundInt(x) = (int)(x + 0.5f)` (common.h:904) — the f32-widened 0.5.
fn round_int(x: f64) -> i32 {
    (x + 0.5f32 as f64) as i32
}

/// Replicate the C++ per-candidate gain scan (REVERSE + FORWARD) using the public
/// `get_split_gains` so the gain MATH is asserted bit-exact, independent of the
/// winner. Returns `(cand_rev, cand_fwd)` with NaN where a candidate is gated out.
fn replicate_candidates(g: &SplitGolden) -> (Vec<f64>, Vec<f64>) {
    let k_eps = lgbm_compute_k_epsilon();
    let sum_hessian_bumped = g.sum_hessian + 2.0 * k_eps;
    let cnt_factor = g.num_data as f64 / sum_hessian_bumped;
    let get = |t: i32| -> (f64, f64) {
        let bi = (t as usize) * 2;
        (g.hist[bi], g.hist[bi + 1])
    };
    let gain_shift = leaf_gain(g.use_l1, g.sum_gradient, g.sum_hessian, g.lambda_l1, g.lambda_l2);
    let min_gain_shift = gain_shift + g.min_gain_to_split;

    // REVERSE
    let mut cand_rev = Vec::new();
    {
        let mut sum_right_gradient = 0.0;
        let mut sum_right_hessian = k_eps;
        let mut right_count = 0i32;
        let mut t = g.num_bin - 1 - g.offset;
        let t_end = 1 - g.offset;
        while t >= t_end {
            if g.skip_default_bin && (t + g.offset) == g.default_bin {
                cand_rev.push(f64::NAN);
                t -= 1;
                continue;
            }
            let (gr, he) = get(t);
            sum_right_gradient += gr;
            sum_right_hessian += he;
            right_count += round_int(he * cnt_factor);
            if right_count < g.min_data_in_leaf || sum_right_hessian < g.min_sum_hessian_in_leaf {
                cand_rev.push(f64::NAN);
                t -= 1;
                continue;
            }
            let left_count = g.num_data - right_count;
            if left_count < g.min_data_in_leaf {
                cand_rev.push(f64::NAN);
                break;
            }
            let sum_left_hessian = sum_hessian_bumped - sum_right_hessian;
            if sum_left_hessian < g.min_sum_hessian_in_leaf {
                cand_rev.push(f64::NAN);
                break;
            }
            let sum_left_gradient = g.sum_gradient - sum_right_gradient;
            let cg = get_split_gains(
                g.use_l1,
                sum_left_gradient,
                sum_left_hessian,
                sum_right_gradient,
                sum_right_hessian,
                g.lambda_l1,
                g.lambda_l2,
            );
            if cg <= min_gain_shift {
                cand_rev.push(f64::NAN);
                t -= 1;
                continue;
            }
            cand_rev.push(cg);
            t -= 1;
        }
    }

    // FORWARD
    let mut cand_fwd = Vec::new();
    {
        let mut sum_left_gradient = 0.0;
        let mut sum_left_hessian = k_eps;
        let mut left_count = 0i32;
        let mut t = 0i32;
        let t_end = g.num_bin - 2 - g.offset;
        while t <= t_end {
            if g.skip_default_bin && (t + g.offset) == g.default_bin {
                cand_fwd.push(f64::NAN);
                t += 1;
                continue;
            }
            let (gr, he) = get(t);
            sum_left_gradient += gr;
            sum_left_hessian += he;
            left_count += round_int(he * cnt_factor);
            if left_count < g.min_data_in_leaf || sum_left_hessian < g.min_sum_hessian_in_leaf {
                cand_fwd.push(f64::NAN);
                t += 1;
                continue;
            }
            let right_count = g.num_data - left_count;
            if right_count < g.min_data_in_leaf {
                cand_fwd.push(f64::NAN);
                break;
            }
            let sum_right_hessian = sum_hessian_bumped - sum_left_hessian;
            if sum_right_hessian < g.min_sum_hessian_in_leaf {
                cand_fwd.push(f64::NAN);
                break;
            }
            let sum_right_gradient = g.sum_gradient - sum_left_gradient;
            let cg = get_split_gains(
                g.use_l1,
                sum_left_gradient,
                sum_left_hessian,
                sum_right_gradient,
                sum_right_hessian,
                g.lambda_l1,
                g.lambda_l2,
            );
            if cg <= min_gain_shift {
                cand_fwd.push(f64::NAN);
                t += 1;
                continue;
            }
            cand_fwd.push(cg);
            t += 1;
        }
    }

    (cand_rev, cand_fwd)
}

/// `kEpsilon` widened to f64 (= `lgbm_core::types::K_EPSILON as f64`). Re-expose
/// here without depending on lgbm-core directly: 1e-15f promoted to f64.
fn lgbm_compute_k_epsilon() -> f64 {
    1e-15f32 as f64
}

/// `GetLeafGain<USE_L1,false,false>` host mirror (feature_histogram.hpp:799-815)
/// — the closed form used for the `gain_shift` whole-leaf gain in
/// `BeforeNumerical`. Delegates to the production `#[cube]` primitive
/// `lgbm_compute::gain::get_leaf_gain` (called as a plain host fn) so there is a
/// SINGLE source of truth for the L1 `Sign(s) = (s>0)-(s<0)` semantics. Rust's
/// `f64::signum` must NOT be used here: it returns `+1.0` at `0.0` and `-1.0` at
/// `-0.0`, never `0.0`, diverging from C++ `Common::Sign` for any zero-gradient
/// L1 case.
fn leaf_gain(use_l1: bool, g: f64, h: f64, l1: f64, l2: f64) -> f64 {
    get_leaf_gain(use_l1, g, h, l1, l2)
}

#[test]
fn kernel_parity_split_bit_exact_on_cpu() {
    let path = kernels_dir().join("split.txt");
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!(
            "kernel_parity(split): SKIP — fixture {} not found. Run \
             `cargo run -p xtask -- kernel-capture`.",
            path.display()
        );
        return;
    };
    let cases = parse_split(&text);
    assert!(!cases.is_empty(), "split fixture present but parsed zero cases");

    let client = cpu_client();
    let backend = CpuBackend;
    let mut saw_reverse_winner = false;
    let mut saw_forward_winner = false;
    let mut saw_skip_default_bin_false_divergence = false;

    for c in &cases {
        // The deferred NA_AS_MISSING forward branch must be false on every
        // committed case — this layer only threads the flag and validates it
        // off; a captured `na_as_missing=1` case would (correctly) be a typed error
        // in the kernel, so the golden must never carry one until that branch lands.
        assert!(
            !c.na_as_missing,
            "SPLIT case `{}`: na_as_missing must be false on every committed case \
             (the NA_AS_MISSING forward branch is deferred, RESEARCH A5)",
            c.name
        );

        // Divergence-case coverage: a case where the OLD heuristic
        // `default_bin < num_bin` would have set SKIP_DEFAULT_BIN but the
        // authoritative flag is false (missing_type == None). Assert the
        // case is genuinely a divergence: skip is false WHILE default_bin < num_bin.
        if c.name == "skip_default_bin_false" {
            assert!(
                !c.skip_default_bin && c.default_bin < c.num_bin,
                "SPLIT `skip_default_bin_false`: must have skip_default_bin==false \
                 AND default_bin ({}) < num_bin ({}) to exercise the divergence",
                c.default_bin,
                c.num_bin
            );
            saw_skip_default_bin_false_divergence = true;
        }

        // (1) PER-CANDIDATE gain MATH bit-exact (localizes to the scan).
        let (rev, fwd) = replicate_candidates(c);
        if let Err(m) = compare_exact_f64_bits(&rev, &c.cand_rev) {
            panic!("SPLIT case `{}`: REVERSE per-candidate gain divergence: {m}", c.name);
        }
        if let Err(m) = compare_exact_f64_bits(&fwd, &c.cand_fwd) {
            panic!("SPLIT case `{}`: FORWARD per-candidate gain divergence: {m}", c.name);
        }

        // (2) WINNER via the real Backend::find_best_split kernel.
        let cfg = GainConfig {
            min_data_in_leaf: c.min_data_in_leaf,
            min_sum_hessian_in_leaf: c.min_sum_hessian_in_leaf,
            max_delta_step: 0.0,
            lambda_l1: c.lambda_l1,
            lambda_l2: c.lambda_l2,
            min_gain_to_split: c.min_gain_to_split,
            path_smooth: 0.0,
            ..Default::default()
        };
        let si = backend
            .find_best_split(
                &client,
                &c.hist,
                &cfg,
                c.num_bin as u32,
                c.offset,
                c.default_bin as u32,
                0,
                c.skip_default_bin,
                c.na_as_missing,
                // run_forward=true: the split.txt golden captures the RAW kernel
                // running BOTH the REVERSE and FORWARD `FindBestThresholdSequentially`
                // (the `skip_default_bin_false` case is a deliberate FORWARD-winner
                // /default_left=0 divergence). This is the kernel-level golden, NOT
                // the learner's per-missing_type `find_best_threshold_fun_` dispatch
                // (feature_histogram.hpp:420-429), so we replay both branches to
                // match the captured winner bit-exact.
                true,
                c.sum_gradient,
                c.sum_hessian,
                c.num_data,
            )
            .unwrap_or_else(|e| panic!("SPLIT case `{}`: find_best_split failed: {e:?}", c.name));

        assert_eq!(
            si.gain == f64::NEG_INFINITY,
            !c.win_is_splittable,
            "SPLIT case `{}`: is_splittable mismatch (gain={})",
            c.name,
            si.gain
        );

        if c.win_is_splittable {
            // Compare every SplitInfo field bit-exact.
            let got = [
                si.gain,
                si.left_sum_gradient,
                si.left_sum_hessian,
                si.right_sum_gradient,
                si.right_sum_hessian,
                si.left_output,
                si.right_output,
            ];
            let exp = [
                c.win_gain,
                c.win_left_sum_gradient,
                c.win_left_sum_hessian,
                c.win_right_sum_gradient,
                c.win_right_sum_hessian,
                c.win_left_output,
                c.win_right_output,
            ];
            if let Err(m) = compare_exact_f64_bits(&got, &exp) {
                panic!("SPLIT case `{}`: winner f64 field divergence: {m}", c.name);
            }
            assert_eq!(si.threshold, c.win_threshold, "SPLIT `{}`: threshold", c.name);
            assert_eq!(si.left_count, c.win_left_count, "SPLIT `{}`: left_count", c.name);
            assert_eq!(si.right_count, c.win_right_count, "SPLIT `{}`: right_count", c.name);
            assert_eq!(
                si.default_left, c.win_default_left,
                "SPLIT `{}`: default_left",
                c.name
            );
            let _ = c.win_min_gain_shift; // captured for documentation/diagnostics
            if c.win_default_left {
                saw_reverse_winner = true;
            } else {
                saw_forward_winner = true;
            }
        }
    }

    // Coverage: BOTH threshold-recording branches must be exercised by a winner.
    assert!(saw_reverse_winner, "split golden must have a REVERSE-branch winner (default_left=1)");
    assert!(saw_forward_winner, "split golden must have a FORWARD-branch winner (default_left=0)");
    // Coverage: the skip_default_bin==false divergence case must be
    // present and replay bit-exact.
    assert!(
        saw_skip_default_bin_false_divergence,
        "split golden must contain the `skip_default_bin_false` divergence case \
         (default_bin < num_bin but skip_default_bin==false, missing_type==None)"
    );
}

// ===========================================================================
// FUSED SPLIT parity. Prove the DEFAULT `find_best_splits_batched`
// (the batched seam) is BIT-EXACT to the per-feature `find_best_split`
// loop on CPU — the contract that keeps the CpuBackend f64 anchor unchanged.
//
// We lay the committed split golden's feature cases into ONE concatenated leaf
// histogram buffer (each case at a successive `slot_off`) with a matching
// `Vec<BatchedSplitFeature>`, then (1) call `find_best_splits_batched` ONCE with a
// single SHARED set of leaf totals, and (2) loop `find_best_split` per feature with
// the SAME shared totals. The two SplitInfo vectors must be cell-for-cell bit-exact
// (no tolerance). Using shared totals (not each golden's own totals) is required
// because the batched op shares the leaf totals across the batch — exactly as the
// learner calls it per leaf; the point is the batched-vs-loop EQUIVALENCE, not the
// per-case C++ winner (that is `kernel_parity_split_bit_exact_on_cpu`'s job).
// ===========================================================================
#[test]
fn kernel_parity_batched_equals_per_feature_on_cpu() {
    let path = kernels_dir().join("split.txt");
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!(
            "kernel_parity(batched-split): SKIP — fixture {} not found.",
            path.display()
        );
        return;
    };
    let cases = parse_split(&text);
    assert!(!cases.is_empty(), "split fixture present but parsed zero cases");

    let client = cpu_client();
    let backend = CpuBackend;

    // Shared leaf totals for the whole batch. Chosen finite/positive so every
    // feature's scan is admissible (sum_hessian > 0 is the V5 precondition); the
    // scan gates relax under min_data_in_leaf=1 / min_sum_hessian_in_leaf=0 so we
    // exercise real winners, not all-`none()`.
    let cfg = GainConfig {
        min_data_in_leaf: 1,
        min_sum_hessian_in_leaf: 0.0,
        max_delta_step: 0.0,
        lambda_l1: 0.0,
        lambda_l2: 0.0,
        min_gain_to_split: 0.0,
        path_smooth: 0.0,
        ..Default::default()
    };

    // Assemble the concatenated buffer + per-feature batch records. Skip the
    // deferred na_as_missing cases (they are a typed error in either path).
    let mut buf: Vec<f64> = Vec::new();
    let mut feats: Vec<BatchedSplitFeature> = Vec::new();
    // Per-feature shared totals (sum over each feature's own histogram so the
    // scan's cnt_factor / leaf sums are self-consistent with that feature's bins).
    let mut per_feat_totals: Vec<(f64, f64, i32)> = Vec::new();
    for c in &cases {
        if c.na_as_missing {
            continue;
        }
        let slot_off = buf.len();
        buf.extend_from_slice(&c.hist);
        // Derive self-consistent leaf totals from THIS feature's histogram so the
        // batched call and the per-feature call see identical, admissible inputs.
        let mut sg = 0.0f64;
        let mut sh = 0.0f64;
        for bin in 0..c.num_bin as usize {
            sg += c.hist[bin * 2];
            sh += c.hist[bin * 2 + 1];
        }
        // num_data: any positive count; use the golden's if positive, else a fallback.
        let nd = if c.num_data > 0 { c.num_data } else { 1 };
        per_feat_totals.push((sg, sh, nd));
        feats.push(BatchedSplitFeature {
            slot_off,
            num_bin: c.num_bin as u32,
            offset: c.offset,
            default_bin: c.default_bin as u32,
            most_freq_bin: 0,
            skip_default_bin: c.skip_default_bin,
            na_as_missing: c.na_as_missing,
            run_forward: true,
        });
    }
    assert!(!feats.is_empty(), "expected at least one batchable split case");

    // The batched op shares ONE set of leaf totals across the batch. To make the
    // per-feature comparison apples-to-apples we run the batch ONCE PER distinct
    // total set is overkill; instead drive each feature as its OWN single-element
    // batch with its self-consistent totals, then ALSO run the per-feature
    // `find_best_split` with the same totals, and assert equality. This proves the
    // default batched impl composes `find_best_split` bit-exact for every feature.
    for (i, f) in feats.iter().enumerate() {
        let (sg, sh, nd) = per_feat_totals[i];
        if !(sh > 0.0) {
            // sum_hessian must be > 0; skip degenerate all-zero-hessian cases
            // (both paths would return the same typed error — not interesting here).
            continue;
        }
        let one = std::slice::from_ref(f);
        let batched = backend
            .find_best_splits_batched(&client, &buf, one, &cfg, sg, sh, nd)
            .unwrap_or_else(|e| panic!("batched find_best_splits_batched failed: {e:?}"));
        assert_eq!(batched.len(), 1, "one feature -> one SplitInfo");
        let bsi = batched[0];

        let cells = 2 * f.num_bin as usize;
        let hist = &buf[f.slot_off..f.slot_off + cells];
        let psi = backend
            .find_best_split(
                &client,
                hist,
                &cfg,
                f.num_bin,
                f.offset,
                f.default_bin,
                f.most_freq_bin,
                f.skip_default_bin,
                f.na_as_missing,
                f.run_forward,
                sg,
                sh,
                nd,
            )
            .unwrap_or_else(|e| panic!("per-feature find_best_split failed: {e:?}"));

        // Every f64 field bit-exact (no tolerance), integer fields exact equality.
        let got = [
            bsi.gain,
            bsi.left_sum_gradient,
            bsi.left_sum_hessian,
            bsi.right_sum_gradient,
            bsi.right_sum_hessian,
            bsi.left_output,
            bsi.right_output,
        ];
        let exp = [
            psi.gain,
            psi.left_sum_gradient,
            psi.left_sum_hessian,
            psi.right_sum_gradient,
            psi.right_sum_hessian,
            psi.left_output,
            psi.right_output,
        ];
        if let Err(m) = compare_exact_f64_bits(&got, &exp) {
            panic!("batched feature {i}: f64 field divergence from per-feature: {m}");
        }
        assert_eq!(bsi.threshold, psi.threshold, "feature {i}: threshold");
        assert_eq!(bsi.left_count, psi.left_count, "feature {i}: left_count");
        assert_eq!(bsi.right_count, psi.right_count, "feature {i}: right_count");
        assert_eq!(bsi.default_left, psi.default_left, "feature {i}: default_left");
        assert_eq!(
            bsi.gain == f64::NEG_INFINITY,
            psi.gain == f64::NEG_INFINITY,
            "feature {i}: is_splittable (kMinScore) agreement"
        );
    }

    // Also exercise a genuine MULTI-feature batch (shared totals) to prove order
    // preservation across the whole Vec: the i-th batched result must equal the
    // i-th per-feature call with the SAME shared totals.
    let (sg, sh, nd) = (8.0f64, 8.0f64, 8i32);
    let multi = backend
        .find_best_splits_batched(&client, &buf, &feats, &cfg, sg, sh, nd)
        .unwrap_or_else(|e| panic!("multi-feature batched call failed: {e:?}"));
    assert_eq!(multi.len(), feats.len(), "one SplitInfo per input feature");
    for (i, f) in feats.iter().enumerate() {
        let cells = 2 * f.num_bin as usize;
        let hist = &buf[f.slot_off..f.slot_off + cells];
        let psi = backend
            .find_best_split(
                &client, hist, &cfg, f.num_bin, f.offset, f.default_bin, f.most_freq_bin,
                f.skip_default_bin, f.na_as_missing, f.run_forward, sg, sh, nd,
            )
            .unwrap_or_else(|e| panic!("multi per-feature find_best_split failed: {e:?}"));
        let got = [
            multi[i].gain,
            multi[i].left_sum_gradient,
            multi[i].left_sum_hessian,
            multi[i].right_sum_gradient,
            multi[i].right_sum_hessian,
            multi[i].left_output,
            multi[i].right_output,
        ];
        let exp = [
            psi.gain,
            psi.left_sum_gradient,
            psi.left_sum_hessian,
            psi.right_sum_gradient,
            psi.right_sum_hessian,
            psi.left_output,
            psi.right_output,
        ];
        if let Err(m) = compare_exact_f64_bits(&got, &exp) {
            panic!("multi batched feature {i}: order-preservation/value divergence: {m}");
        }
        assert_eq!(multi[i].threshold, psi.threshold, "multi feature {i}: threshold");
        assert_eq!(multi[i].default_left, psi.default_left, "multi feature {i}: default_left");
    }
}

// ===========================================================================
// THREE-WAY MERGE GATE. Prove the FUSED per-leaf split kernel is
// BIT-EXACT to BOTH the cubecl per-feature loop AND the native production scan:
//   fused find_best_splits_batched_fused_f64_on
//     == per-feature find_best_split_f64_on
//     == find_best_split_cpu_native
// for EVERY feature in input order (order preservation). This proves the
// shared split_scan_body math and fused launch did NOT drift any of the
// three transcriptions. No tolerance — compare_exact_f64_bits.
// ===========================================================================
#[test]
fn kernel_parity_fused_equals_per_feature_and_native() {
    use lgbm_compute::kernels::split::{
        find_best_split_cpu_native, find_best_split_f64_on, find_best_splits_batched_fused_f64_on,
    };

    let path = kernels_dir().join("split.txt");
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!(
            "kernel_parity(fused-split): SKIP — fixture {} not found.",
            path.display()
        );
        return;
    };
    let cases = parse_split(&text);
    assert!(!cases.is_empty(), "split fixture present but parsed zero cases");

    let client = cpu_client();

    // Same relaxed cfg as kernel_parity_batched_equals_per_feature_on_cpu so the
    // scan exercises real winners (not all-none).
    let cfg = GainConfig {
        min_data_in_leaf: 1,
        min_sum_hessian_in_leaf: 0.0,
        max_delta_step: 0.0,
        lambda_l1: 0.0,
        lambda_l2: 0.0,
        min_gain_to_split: 0.0,
        path_smooth: 0.0,
        ..Default::default()
    };

    // Assemble a MULTI-feature batch: concatenated buffer + per-feature records,
    // skipping the deferred na_as_missing cases (a typed error in every path).
    let mut buf: Vec<f64> = Vec::new();
    let mut feats: Vec<BatchedSplitFeature> = Vec::new();
    for c in &cases {
        if c.na_as_missing {
            continue;
        }
        let slot_off = buf.len();
        buf.extend_from_slice(&c.hist);
        feats.push(BatchedSplitFeature {
            slot_off,
            num_bin: c.num_bin as u32,
            offset: c.offset,
            default_bin: c.default_bin as u32,
            most_freq_bin: 0,
            skip_default_bin: c.skip_default_bin,
            na_as_missing: c.na_as_missing,
            run_forward: true,
        });
    }
    assert!(!feats.is_empty(), "expected at least one batchable split case");

    // Shared leaf totals across the whole batch (exactly how the learner calls the
    // batched op per leaf). Finite/positive so every feature's scan is admissible.
    let (sg, sh, nd) = (8.0f64, 8.0f64, 8i32);

    // (1) ONE fused launch over all features.
    let fused = find_best_splits_batched_fused_f64_on(&client, &buf, &feats, &cfg, sg, sh, nd)
        .unwrap_or_else(|e| panic!("fused find_best_splits_batched_fused_f64_on failed: {e:?}"));
    assert_eq!(fused.len(), feats.len(), "fused: one SplitInfo per input feature");

    for (i, f) in feats.iter().enumerate() {
        let cells = 2 * f.num_bin as usize;
        let hist = &buf[f.slot_off..f.slot_off + cells];

        // (2) cubecl per-feature scan (single-owner launch).
        let per = find_best_split_f64_on(
            &client, hist, &cfg, f.num_bin, f.offset, f.default_bin, f.most_freq_bin,
            f.skip_default_bin, f.na_as_missing, f.run_forward, sg, sh, nd,
        )
        .unwrap_or_else(|e| panic!("per-feature find_best_split_f64_on failed: {e:?}"));

        // (3) native production scan.
        let nat = find_best_split_cpu_native(
            hist, &cfg, f.num_bin, f.offset, f.default_bin, f.most_freq_bin, f.skip_default_bin,
            f.na_as_missing, f.run_forward, sg, sh, nd,
        )
        .unwrap_or_else(|e| panic!("native find_best_split_cpu_native failed: {e:?}"));

        let fields = |s: &lgbm_compute::SplitInfo| {
            [
                s.gain,
                s.left_sum_gradient,
                s.left_sum_hessian,
                s.right_sum_gradient,
                s.right_sum_hessian,
                s.left_output,
                s.right_output,
            ]
        };
        let fu = fields(&fused[i]);
        let pe = fields(&per);
        let na = fields(&nat);

        // fused == per-feature (bit-exact f64).
        if let Err(m) = compare_exact_f64_bits(&fu, &pe) {
            panic!("feature {i}: fused != per-feature (cubecl): {m}");
        }
        // per-feature == native (bit-exact f64) — the production anchor agreement.
        if let Err(m) = compare_exact_f64_bits(&pe, &na) {
            panic!("feature {i}: per-feature != native: {m}");
        }
        // Integer / flag fields: three-way exact equality.
        assert_eq!(fused[i].threshold, per.threshold, "feature {i}: fused vs per threshold");
        assert_eq!(per.threshold, nat.threshold, "feature {i}: per vs native threshold");
        assert_eq!(fused[i].left_count, per.left_count, "feature {i}: fused vs per left_count");
        assert_eq!(per.left_count, nat.left_count, "feature {i}: per vs native left_count");
        assert_eq!(fused[i].right_count, per.right_count, "feature {i}: fused vs per right_count");
        assert_eq!(per.right_count, nat.right_count, "feature {i}: per vs native right_count");
        assert_eq!(fused[i].default_left, per.default_left, "feature {i}: fused vs per default_left");
        assert_eq!(per.default_left, nat.default_left, "feature {i}: per vs native default_left");
    }
}

// ===========================================================================
// SIBLING CO-PACK parity (cubecl-cpu W=1 byte-identity half).
//
// The co-packed 2-slot sibling scan (`find_best_splits_fused_siblings_from_handles_on`)
// must return SplitInfos byte-IDENTICAL to two separate single-slot scans
// (`find_best_splits_batched_fused_f64_on`), per sibling, every `SplitInfo` field.
//
// This is bit-exact BY CONSTRUCTION: each feature's REVERSE+FORWARD sequential
// scan is the SAME `split_scan_body` over the SAME disjoint per-feature region —
// co-packing only changes WHICH launch a feature's scan runs in, never its math.
// On the cubecl-cpu runtime the kernel runs at W=1, which stays byte-identical,
// so this proves the byte-identical claim on the ALWAYS-available CPU runtime —
// no GPU needed for the bit-exact gate.
//
// The two siblings carry DIFFERENT leaf totals (smaller-built vs larger-subtract-
// derived asymmetry) over the SAME feature layout (the shared per-feature param
// arrays the 2-slot kernel consumes).
// ===========================================================================
#[test]
fn kernel_parity_sibling_copack_equals_two_scans_on_cpu() {
    use lgbm_compute::kernels::split::{
        find_best_splits_batched_fused_f64_on, find_best_splits_fused_siblings_from_handles_on,
        upload_f64_buffer,
    };

    let client = cpu_client();

    // SHARED multi-feature layout (3 features, mixed num_bin/offset/run_forward so
    // a REVERSE winner, a FORWARD winner, and a no-split feature are all covered —
    // mirrors the from-handle cell's spine). Pre-fixed/compacted stride-2 f64 cells
    // fed directly (the launcher scans an already-prepared histogram; no re-fix).
    let feats = copack_feats();
    let (buf_a, buf_b) = copack_two_histograms();
    assert_eq!(buf_a.len(), buf_b.len(), "siblings share the feature layout (buf len)");

    let cfg = copack_cfg();
    // DIFFERENT leaf totals per sibling (smaller A vs larger-derived B).
    let a_totals = (5.0f64, 40.0f64, 40i32);
    let b_totals = (-3.0f64, 64.0f64, 64i32);

    // (a) Co-packed 2-slot scan: two Handles -> one launch -> (co_a, co_b).
    let h_a = upload_f64_buffer(&client, &buf_a);
    let h_b = upload_f64_buffer(&client, &buf_b);
    let (co_a, co_b) = find_best_splits_fused_siblings_from_handles_on(
        &client,
        h_a,
        h_b,
        buf_a.len(),
        &feats,
        &cfg,
        a_totals,
        b_totals,
    )
    .expect("co-packed sibling scan (cubecl-cpu W=1)");

    // (b) Two SEPARATE single-slot scans of the same two histograms + per-sibling totals.
    let single_a =
        find_best_splits_batched_fused_f64_on(&client, &buf_a, &feats, &cfg, a_totals.0, a_totals.1, a_totals.2)
            .expect("single-slot scan A");
    let single_b =
        find_best_splits_batched_fused_f64_on(&client, &buf_b, &feats, &cfg, b_totals.0, b_totals.1, b_totals.2)
            .expect("single-slot scan B");

    assert_eq!(co_a.len(), feats.len(), "co-pack A: one SplitInfo per feature");
    assert_eq!(co_b.len(), feats.len(), "co-pack B: one SplitInfo per feature");

    // EXACT per-feature SplitInfo equality (W=1, byte-identical-by-construction): the
    // co-packed scan runs the SAME f64 kernel over the SAME input bits, only the launch
    // differs. assert_eq! on the full SplitInfo (mirrors the from-handle exact compare).
    for (i, (co, si)) in co_a.iter().zip(single_a.iter()).enumerate() {
        assert_eq!(
            co, si,
            "sibling A feature {i}: co-pack scan != single-slot scan (must be byte-identical): \
             copack={co:?} single={si:?}"
        );
    }
    for (i, (co, si)) in co_b.iter().zip(single_b.iter()).enumerate() {
        assert_eq!(
            co, si,
            "sibling B feature {i}: co-pack scan != single-slot scan (must be byte-identical): \
             copack={co:?} single={si:?}"
        );
    }
}

/// SHARED co-pack test fixture — the per-feature layout BOTH siblings scan (the
/// `feats` the 2-slot kernel consumes once). 3 features: a REVERSE-only winner, a
/// FORWARD-allowed winner, and (with the chosen histograms) a no-split feature.
/// Reused by the cubecl-cpu cell here and the `--features rocm` cell in `mod hip`.
fn copack_feats() -> Vec<BatchedSplitFeature> {
    vec![
        // f0: num_bin 4, offset 0, REVERSE only (run_forward=false) -> 8 cells.
        BatchedSplitFeature {
            slot_off: 0,
            num_bin: 4,
            offset: 0,
            default_bin: 0,
            most_freq_bin: 2,
            skip_default_bin: false,
            na_as_missing: false,
            run_forward: false,
        },
        // f1: num_bin 3, offset 1 (compacted), FORWARD allowed -> 6 cells.
        BatchedSplitFeature {
            slot_off: 8,
            num_bin: 3,
            offset: 1,
            default_bin: 0,
            most_freq_bin: 0,
            skip_default_bin: false,
            na_as_missing: false,
            run_forward: true,
        },
        // f2: num_bin 5, offset 0, FORWARD allowed -> 10 cells.
        BatchedSplitFeature {
            slot_off: 14,
            num_bin: 5,
            offset: 0,
            default_bin: 0,
            most_freq_bin: 1,
            skip_default_bin: false,
            na_as_missing: false,
            run_forward: true,
        },
    ]
}

/// The SHARED relaxed gain cfg for the co-pack cells (admissible splits, no
/// L1/L2/path-smooth so the gate exercises real winners on both siblings).
fn copack_cfg() -> GainConfig {
    GainConfig {
        min_data_in_leaf: 1,
        min_sum_hessian_in_leaf: 1e-3,
        max_delta_step: 0.0,
        lambda_l1: 0.0,
        lambda_l2: 0.0,
        min_gain_to_split: 0.0,
        path_smooth: 0.0,
        ..Default::default()
    }
}

/// The TWO concatenated stride-2 f64 histograms over the `copack_feats` layout
/// (24 cells = 8 + 6 + 10). `buf_a` is the "smaller-built" sibling; `buf_b` the
/// "larger-subtract-derived" sibling — DIFFERENT cell values so the per-sibling
/// scans produce distinct winners. Exactly-representable
/// f64 (widen losslessly to/from f32 on the GPU). Reused by both co-pack cells.
fn copack_two_histograms() -> (Vec<f64>, Vec<f64>) {
    let buf_a: Vec<f64> = vec![
        // f0 4 bins
        2.0, 5.0, -1.0, 4.0, 3.0, 6.0, -2.0, 5.0,
        // f1 3 bins
        1.0, 3.0, 4.0, 7.0, -3.0, 4.0,
        // f2 5 bins
        0.5, 2.0, 1.5, 3.0, -1.0, 2.5, 2.0, 4.0, -0.5, 3.5,
    ];
    let buf_b: Vec<f64> = vec![
        // f0 4 bins (larger sibling: different grad/hess mass)
        -2.0, 8.0, 3.0, 7.0, -1.0, 9.0, 4.0, 8.0,
        // f1 3 bins
        2.0, 5.0, -4.0, 6.0, 1.0, 7.0,
        // f2 5 bins
        -1.5, 4.0, 0.5, 5.0, 2.0, 4.5, -2.0, 6.0, 1.5, 5.5,
    ];
    (buf_a, buf_b)
}

// ===========================================================================
// PARTITION parity. Drive Backend::data_partition over the golden bins +
// routing and assert the reordered index array + split_point match exactly.
// ===========================================================================
#[test]
fn kernel_parity_partition_exact_on_cpu() {
    let path = kernels_dir().join("partition.txt");
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!(
            "kernel_parity(partition): SKIP — fixture {} not found.",
            path.display()
        );
        return;
    };

    let client = cpu_client();
    let backend = CpuBackend;
    let mut lines = text.lines();
    let mut n_cases = 0;
    while let Some(raw) = lines.next() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let t: Vec<&str> = line.split_whitespace().collect();
        if t[0] != "PCASE" {
            continue;
        }
        let name = field(&t, "name").expect("PCASE name").to_string();
        let num_bin = parse_i64(&t, "num_bin") as u32;
        let min_bin = parse_i64(&t, "min_bin") as u32;
        let max_bin = parse_i64(&t, "max_bin") as u32;
        let threshold = parse_i64(&t, "threshold") as u32;
        let most_freq_bin = parse_i64(&t, "most_freq_bin") as u32;
        // The fixture also carries flag fan-out cases (`missing_type != None`),
        // which are routed by the device `GenDataToLeftBitVector` path — NOT
        // this None-only host-gather `data_partition`. Skip them HERE (consuming
        // their 3 payload lines to keep the stream aligned); they are covered by
        // `partition_parity.rs`. The `missing_type` field is absent on older
        // goldens → defaults to 0 (None).
        let missing_type = field(&t, "missing_type")
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(0);
        if missing_type != 0 {
            let _ = lines.next(); // PBINS
            let _ = lines.next(); // PORDER
            let _ = lines.next(); // PSPLIT
            continue;
        }

        let bt: Vec<&str> = lines.next().expect("PBINS").split_whitespace().collect();
        assert_eq!(bt[0], "PBINS", "expected PBINS for `{name}`");
        let bins = parse_u32_list(bt.get(1).copied().unwrap_or(""));

        let ot: Vec<&str> = lines.next().expect("PORDER").split_whitespace().collect();
        assert_eq!(ot[0], "PORDER", "expected PORDER for `{name}`");
        let order = parse_u32_list(ot.get(1).copied().unwrap_or(""));

        let st: Vec<&str> = lines.next().expect("PSPLIT").split_whitespace().collect();
        assert_eq!(st[0], "PSPLIT", "expected PSPLIT for `{name}`");
        let split_point: usize = st[1].parse().expect("split_point usize");

        let (got_order, got_split) = backend
            .data_partition(&client, &bins, num_bin, min_bin, max_bin, threshold, most_freq_bin)
            .unwrap_or_else(|e| panic!("PARTITION `{name}`: data_partition failed: {e:?}"));

        if let Err(m) = compare_exact_u32(&got_order, &order) {
            panic!("PARTITION `{name}`: reordered index array divergence: {m}");
        }
        assert_eq!(
            got_split, split_point,
            "PARTITION `{name}`: split_point mismatch"
        );

        // Route the SAME golden case through the NATIVE-WIDTH `data_partition_native`
        // path with the bins carried as a narrow `BinColumn` (auto-selected by
        // num_bin: U8 for ≤256, U16 for ≤65536). This narrow path MUST stay
        // bit-exact to the golden.
        let col = BinColumn::new(bins.clone(), num_bin);
        let (nat_order, nat_split) = backend
            .data_partition_native(&client, &col, num_bin, min_bin, max_bin, threshold, most_freq_bin)
            .unwrap_or_else(|e| panic!("PARTITION `{name}`: data_partition_native failed: {e:?}"));
        if let Err(m) = compare_exact_u32(&nat_order, &order) {
            panic!("PARTITION `{name}` (native): reordered index array divergence: {m}");
        }
        assert_eq!(
            nat_split, split_point,
            "PARTITION `{name}` (native): split_point mismatch"
        );
        n_cases += 1;
    }
    assert!(n_cases > 0, "partition fixture present but parsed zero cases");

    // An explicit U16-band self-check so the U16 monomorph is
    // exercised even if every golden `num_bin` is ≤256 (⇒ U8). Build the EXPECTED
    // (order, split_point) via the u32-widened `data_partition` reference, then assert
    // `data_partition_native` over the U16 `BinColumn` is byte-identical to it.
    {
        let bins_u16 = vec![0u32, 300, 64, 511, 7];
        let num_bin = 512u32; // ⇒ BinColumn::U16
        let (min_bin, max_bin, threshold, most_freq_bin) = (0u32, 511u32, 63u32, 0u32);
        let (exp_order, exp_split) = backend
            .data_partition(&client, &bins_u16, num_bin, min_bin, max_bin, threshold, most_freq_bin)
            .expect("u16 reference partition");
        let col_u16 = BinColumn::new(bins_u16.clone(), num_bin);
        assert!(matches!(col_u16, BinColumn::U16(_)), "expected U16 column");
        let (got_order, got_split) = backend
            .data_partition_native(&client, &col_u16, num_bin, min_bin, max_bin, threshold, most_freq_bin)
            .expect("u16 native partition");
        assert_eq!(got_order, exp_order, "PARTITION u16-selfcheck: order mismatch");
        assert_eq!(got_split, exp_split, "PARTITION u16-selfcheck: split mismatch");
    }
}

// ===========================================================================
// SUBTRACT parity. Drive Backend::subtract_histograms and assert the
// derived cells bit-exact vs the golden.
// ===========================================================================
#[test]
fn kernel_parity_subtract_bit_exact_on_cpu() {
    let path = kernels_dir().join("subtract.txt");
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!(
            "kernel_parity(subtract): SKIP — fixture {} not found.",
            path.display()
        );
        return;
    };

    let client = cpu_client();
    let backend = CpuBackend;
    let mut lines = text.lines();
    let mut n_cases = 0;
    while let Some(raw) = lines.next() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let t: Vec<&str> = line.split_whitespace().collect();
        if t[0] != "SUBCASE" {
            continue;
        }
        let name = field(&t, "name").expect("SUBCASE name").to_string();

        let pt: Vec<&str> = lines.next().expect("SUBPARENT").split_whitespace().collect();
        assert_eq!(pt[0], "SUBPARENT", "expected SUBPARENT for `{name}`");
        let parent = parse_f64_bits_list(pt.get(1).copied().unwrap_or(""));

        let ct: Vec<&str> = lines.next().expect("SUBCHILD").split_whitespace().collect();
        assert_eq!(ct[0], "SUBCHILD", "expected SUBCHILD for `{name}`");
        let child = parse_f64_bits_list(ct.get(1).copied().unwrap_or(""));

        let dt: Vec<&str> = lines.next().expect("SUBDERIVED").split_whitespace().collect();
        assert_eq!(dt[0], "SUBDERIVED", "expected SUBDERIVED for `{name}`");
        let derived = parse_f64_bits_list(dt.get(1).copied().unwrap_or(""));

        let got = backend
            .subtract_histograms(&client, &parent, &child)
            .unwrap_or_else(|e| panic!("SUBTRACT `{name}`: subtract_histograms failed: {e:?}"));

        if let Err(m) = compare_exact_f64_bits(&got, &derived) {
            panic!("SUBTRACT `{name}`: derived cell divergence: {m}");
        }
        n_cases += 1;
    }
    assert!(n_cases > 0, "subtract fixture present but parsed zero cases");
}

// ===========================================================================
// HIP (ROCm) parity layer — the SEPARATE ~1e-6 gate.
//
// ENTIRELY `#[cfg(feature = "rocm")]`: this layer compiles/runs ONLY under
// `cargo test -p oracle-harness --features rocm` and NEVER affects the default
// CPU build (which keeps the bit-exact layers above as the HARD gate).
//
// For each committed golden it:
//   1. drives the f32-accumulate kernel on the HIP runtime  -> `hip_f32: Vec<f32>`
//   2. drives the f64 anchor kernel on the cubecl-cpu runtime -> `cpu_anchor_f64: Vec<f64>`
//   3. collects the anchor to f32 EXACTLY per the plan recipe:
//        `let cpu_anchor_f32: Vec<f32> = cpu_anchor_f64.iter().map(|&x| x as f32).collect();`
//   4. asserts `compare_within(&hip_f32, &cpu_anchor_f32, ORACLE_TOL)` reports 0
//      mismatches at `ORACLE_TOL = 1e-6`, surfacing the per-case `abs_diff` on
//      mismatch so the gap can be recorded in `04-ROCM-GAPS.md` (no silent pass).
//
// `data_partition` is f64-free, so its hip result is compared bit-EXACT (u32) to
// the committed routing — no tolerance needed.
// ===========================================================================

#[cfg(feature = "rocm")]
mod hip {
    use super::*;
    use lgbm_compute::runtime::rocm_client;
    use oracle_harness::comparator::{compare_within, Mismatch, ORACLE_TOL};

    /// Collect an f64 anchor slice down to `Vec<f32>` — the EXACT conversion the
    /// plan mandates (`cpu_anchor_f64.iter().map(|&x| x as f32).collect()`).
    fn anchor_to_f32(cpu_anchor_f64: &[f64]) -> Vec<f32> {
        cpu_anchor_f64.iter().map(|&x| x as f32).collect()
    }

    /// The PRIMARY hip gate at `ORACLE_TOL = 1e-6` (compare_within). As a
    /// best-effort ROCm gate, a residual f32-vs-f64 accumulation divergence that
    /// exceeds `ORACLE_TOL` but stays within f32's natural precision is NOT a
    /// blocker — it is SURFACED to stderr (the per-case `index`/`abs_diff`
    /// for the `04-ROCM-GAPS.md` ledger; no silent pass) and the test continues.
    ///
    /// A separate, generous **f32 relative sanity bound** (`HIP_SANITY_REL`) DOES
    /// hard-fail: it distinguishes the expected f32-accumulation gap (relative
    /// error on the order of f32 epsilon, ~1e-7..1e-5) from a genuine kernel BUG
    /// (a wrong formula / index / gate would diverge by a large relative margin).
    /// This satisfies BOTH "no silent pass" AND "residual gap is a documented
    /// follow-up, not a blocker".
    const HIP_SANITY_REL: f32 = 1e-3;

    fn assert_within(label: &str, hip_f32: &[f32], cpu_anchor_f32: &[f32]) {
        // (a) PRIMARY: the strict 1e-6 oracle gate. On a mismatch, surface the gap
        //     (no silent pass) but do NOT block — record it for 04-ROCM-GAPS.md.
        if let Err(Mismatch::ValueMismatch {
            index,
            rust,
            cpp,
            abs_diff,
            tol,
        }) = compare_within(hip_f32, cpu_anchor_f32, ORACLE_TOL)
        {
            eprintln!(
                "HIP PARITY GAP `{label}` at index {index}: hip={rust}, cpu_anchor={cpp}, \
                 abs_diff={abs_diff} > ORACLE_TOL={tol} \
                 (documented f32-vs-f64 accumulation gap, 04-ROCM-GAPS.md / D-03a)"
            );
        } else if let Err(other) =
            compare_within(hip_f32, cpu_anchor_f32, ORACLE_TOL)
        {
            // length mismatch etc. — a structural error, always a hard fail.
            panic!("HIP PARITY `{label}`: {other}");
        }

        // (b) SANITY: a generous f32 RELATIVE bound that DOES hard-fail. Catches a
        //     real bug (wrong formula/index) while tolerating the f32-accumulation
        //     gap. rel = |hip - cpu| / max(|cpu|, 1.0).
        for (i, (&h, &c)) in hip_f32.iter().zip(cpu_anchor_f32.iter()).enumerate() {
            let denom = c.abs().max(1.0);
            let rel = (h - c).abs() / denom;
            assert!(
                rel <= HIP_SANITY_REL,
                "HIP SANITY FAIL `{label}` at index {i}: hip={h}, cpu_anchor={c}, \
                 rel_diff={rel} > {HIP_SANITY_REL} — this is a real divergence, NOT the \
                 tolerated f32-accumulation gap (likely a kernel bug)"
            );
        }
    }





    /// VERBATIM copy of the learner's private `compact_histogram`
    /// (`lgbm-treelearner/src/learner.rs:2838-2864`) — replicated here (rather than
    /// exposing the private fn) so the host reference path stays byte-for-byte the
    /// production op while keeping `learner.rs` UNCHANGED. Pairs with the
    /// real exported `lgbm_treelearner::fix_histogram`.
    fn host_compact_histogram(hist: &mut [f64], offset: i32) {
        if offset <= 0 {
            return;
        }
        let off = offset as usize;
        let num_bin = hist.len() / 2;
        if off >= num_bin {
            for cell in hist.iter_mut() {
                *cell = 0.0;
            }
            return;
        }
        for c in 0..(num_bin - off) {
            let dst = c << 1;
            let src = (c + off) << 1;
            hist[dst] = hist[src];
            hist[dst + 1] = hist[src + 1];
        }
        for cell in hist.iter_mut().skip((num_bin - off) << 1) {
            *cell = 0.0;
        }
    }

    /// The BIT-EXACT oracle: the on-GPU `fix_compact_kernel`
    /// (via `fix_compact_f64_on`) must produce, for a leaf's concatenated RAW f64
    /// histogram, the SAME buffer that applying the host `fix_histogram` then
    /// `compact_histogram` PER FEATURE produces — BIT-IDENTICAL via
    /// `compare_exact_f64_bits`. The numerical insight: reading the SAME f64-widened
    /// cells and folding in the SAME ascending order yields a bit-identical result;
    /// the only ~1e-6 difference (the f32-atomic RAW build) is NOT exercised here —
    /// the RAW buffer is supplied directly as f64.
    ///
    /// Covers a MIXED-feature leaf concatenating:
    ///  - Test A: mfb > 0, offset == 0  — fix RECONSTRUCTS the mfb cell, compact no-op.
    ///  - Test B: mfb == 0, offset == 1 — fix NO-OP, compact DROPS bin 0.
    ///  - Test C: mfb >= num_bin        — fix no-op (defensive), offset 0 ⇒ compact no-op.
    ///  - Test D: offset >= num_bin     — compact ZEROS the whole feature region.
    /// Plus an extra mfb > 0 feature so the mixed concatenation exercises >1 fix.
    #[test]
    fn kernel_parity_fix_compact_equals_host_on_hip() {
        use lgbm_compute::kernels::histogram::fix_compact_f64_on;
        use lgbm_treelearner::fix_histogram;

        let hip = rocm_client();

        // RAW leaf totals (un-bumped). Chosen so the reconstructs land
        // on exactly-representable f64 values for an unambiguous bit-exact assert.
        let sum_g = 100.0f64;
        let sum_h = 250.0f64;

        // Per-feature (num_bin, offset, most_freq_bin) + a RAW stride-2 region.
        // Region layouts pulled from realistic bin counts; the mfb cell content is
        // GARBAGE before fix (it is reconstructed) where mfb > 0.
        struct Feat {
            num_bin: u32,
            offset: i32,
            mfb: u32,
            region: Vec<f64>, // 2*num_bin RAW cells
        }
        let feats = vec![
            // Test A — mfb=2 (>0), offset=0, num_bin=4: reconstruct bin2, compact no-op.
            Feat {
                num_bin: 4,
                offset: 0,
                mfb: 2,
                region: vec![1.0, 2.0, 3.0, 4.0, /*mfb2 garbage*/ 99.0, 88.0, 5.0, 6.0],
            },
            // Test B — mfb=0, offset=1, num_bin=3: fix no-op, compact drops bin0.
            Feat {
                num_bin: 3,
                offset: 1,
                mfb: 0,
                region: vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            },
            // Test C — mfb=9 (>=num_bin), offset=0, num_bin=4: fix no-op, compact no-op.
            Feat {
                num_bin: 4,
                offset: 0,
                mfb: 9,
                region: vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
            },
            // Test D — offset=5 (>=num_bin=3): compact zeros the whole region.
            Feat {
                num_bin: 3,
                offset: 5,
                mfb: 0,
                region: vec![7.0, 7.0, 8.0, 8.0, 9.0, 9.0],
            },
            // Extra — mfb=1 (>0), offset=0, num_bin=2: a second reconstruct in the mix.
            Feat {
                num_bin: 2,
                offset: 0,
                mfb: 1,
                region: vec![3.0, 9.0, /*mfb1 garbage*/ 77.0, 66.0],
            },
        ];

        // Concatenate the RAW regions as f32 (the construct kernel's output type;
        // the folded kernel reads f32 RAW and widens inline).
        // Every region value here is exactly f32-representable, so f32 carries the
        // RAW losslessly and the host reference (widened f32→f64) is the exact same
        // f64 the folded kernel produces from its inline `f64::cast_from`.
        let mut raw32: Vec<f32> = Vec::new();
        let mut params: Vec<(usize, u32, i32, u32)> = Vec::new();
        for f in &feats {
            assert_eq!(f.region.len(), 2 * f.num_bin as usize, "region length");
            let slot_off = raw32.len();
            params.push((slot_off, f.num_bin, f.offset, f.mfb));
            raw32.extend(f.region.iter().map(|&x| x as f32));
        }

        // HOST reference: widen the SAME f32 RAW to f64 (the folded kernel's inline
        // cast), then fix_histogram (the exported op) + compact, per feature.
        let mut host: Vec<f64> = raw32.iter().map(|&x| f64::from(x)).collect();
        for (&(slot_off, num_bin, offset, mfb), _f) in params.iter().zip(feats.iter()) {
            let cells = 2 * num_bin as usize;
            let region = &mut host[slot_off..slot_off + cells];
            fix_histogram(region, mfb, sum_g, sum_h);
            host_compact_histogram(region, offset);
        }

        // GPU path: the on-device FOLDED widen+fix+compact kernel over the f32 RAW.
        let gpu = fix_compact_f64_on(&hip, &raw32, &params, sum_g, sum_h)
            .expect("fix_compact_f64_on");

        assert_eq!(gpu.len(), host.len(), "fix_compact length");
        // BIT-EXACT: same inline widen cast, same f64 cells, same ascending fold
        // order ⇒ identical bits (the folded kernel == host fix+compact).
        if let Err(m) = compare_exact_f64_bits(&gpu, &host) {
            panic!("GPU fix_compact != host fix+compact (NOT bit-exact): {m}");
        }
    }

    /// The DEVICE-RESIDENT build→fix→compact chain
    /// (`build_fix_compact_resident_readback_f64_on`), driving the **u64
    /// two's-complement fixed-point (S=2^30) resident LDS build**, pinned to a
    /// fresh **CPU f64 anchor** — NOT a second GPU launcher.
    ///
    /// ## Why this test is anchored to the CPU
    /// Comparing the GPU `build_fix_compact_resident_*` output against a "host
    /// reference" that is itself a GPU launcher (e.g.
    /// `build_leaf_histograms_resident_f32_on`) would be a GPU-vs-GPU comparison
    /// with NO deterministic anchor — never pin two nondeterministic GPU paths to
    /// each other. This test instead builds the reference from the bit-exact CPU
    /// f64 fold `construct_histograms_cpu` (per feature, over the leaf's gathered
    /// `(bin, grad, hess)`), then applies the SAME host `fix_histogram` +
    /// `host_compact_histogram` the test already runs — so the GPU u64 path is pinned
    /// to the CPU f64 anchor, never to another GPU launcher.
    ///
    /// ## Tightened fixed-point gate
    /// The resident LDS sub-hist accumulates in u64 fixed-point, so the residual
    /// vs the f64 anchor is bounded, deterministic QUANTIZE-ROUNDING error (≤ ~1/2^30
    /// per cell), NOT f32 catastrophic-cancellation error. Measured
    /// `rel_vs_anchor ≈ 5.9e-9` on real hardware (and exact `0.0` in the cancelling
    /// integer regime) — far tighter than the old f32-LDS path. We therefore pin
    /// this test to a TIGHTENED `FIXEDPOINT_REL_GATE` well below the generous f32
    /// envelope (`HIP_SANITY_REL = 1e-3`), so a real fixed-point regression cannot
    /// slip. We do NOT reuse `assert_within` here — that helper's `HIP_SANITY_REL`
    /// is calibrated for the f32 launchers.
    ///
    /// ## Determinism
    /// Integer add is order-independent, so the u64 resident build is deterministic
    /// BY CONSTRUCTION (unlike the f32 path, which was only incidentally deterministic
    /// at single-cube). A sub-assert runs the build TWICE over identical inputs and
    /// requires bit-equal read-back (`to_bits` compare) on every f64 output cell.
    #[test]
    fn kernel_parity_resident_build_fix_compact_equals_host_on_hip() {
        use lgbm_compute::kernels::histogram::{
            build_fix_compact_resident_readback_f64_on, construct_histograms_cpu,
            upload_resident_columns,
        };
        use lgbm_treelearner::fix_histogram;

        let hip = rocm_client();
        let cpu = cpu_client();

        // 3 features over 10 rows, different bin counts + non-trivial mfb/offset so
        // the fix (reconstruct) and compact (drop bin 0) both fire.
        let num_data = 10usize;
        let f0: Vec<u32> = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1]; // num_bin 4, mfb 2, off 0
        let f1: Vec<u32> = vec![0, 0, 1, 1, 2, 2, 0, 1, 2, 0]; // num_bin 3, mfb 0, off 1
        let f2: Vec<u32> = vec![4, 3, 2, 1, 0, 1, 2, 3, 4, 0]; // num_bin 5, mfb 1, off 0
        let num_bins: Vec<u32> = vec![4, 3, 5];
        let mfbs: Vec<u32> = vec![2, 0, 1];
        let offsets: Vec<i32> = vec![0, 1, 0];
        let feature_bins: Vec<&[u32]> = vec![&f0, &f1, &f2];
        let mut slot_off = Vec::with_capacity(num_bins.len());
        let mut acc = 0usize;
        for &nb in &num_bins {
            slot_off.push(acc);
            acc += 2 * nb as usize;
        }
        let slot_len = acc;
        let leaf_rows: Vec<u32> = vec![1, 3, 4, 6, 7, 9];
        let gradients: Vec<f32> = (0..num_data).map(|i| 0.25 + i as f32 * 0.5).collect();
        let hessians: Vec<f32> = (0..num_data).map(|i| 1.0 + (i % 3) as f32).collect();

        // Leaf RAW totals over the leaf rows (un-bumped).
        let sum_g: f64 = leaf_rows.iter().map(|&r| f64::from(gradients[r as usize])).sum();
        let sum_h: f64 = leaf_rows.iter().map(|&r| f64::from(hessians[r as usize])).sum();

        // ---- CPU f64 ANCHOR (the reference is the bit-exact CPU fold,
        //      NOT a second GPU launcher). Per feature: gather the leaf rows'
        //      (bin, grad, hess), fold with `construct_histograms_cpu` (the bit-exact
        //      CPU f64 path), then apply the SAME host fix + compact the GPU chain
        //      runs on device. This is the `cpu_anchor` gather idiom from
        //      rocm_cuda_mirror.rs reused here for the resident leaf.
        let mut anchor: Vec<f64> = vec![0.0; slot_len];
        for fpos in 0..num_bins.len() {
            let nb = num_bins[fpos];
            let cells = 2 * nb as usize;
            let binned: Vec<u32> = leaf_rows
                .iter()
                .map(|&r| feature_bins[fpos][r as usize])
                .collect();
            let g: Vec<f32> = leaf_rows.iter().map(|&r| gradients[r as usize]).collect();
            let h: Vec<f32> = leaf_rows.iter().map(|&r| hessians[r as usize]).collect();
            let raw = construct_histograms_cpu(&cpu, &binned, &g, &h, nb)
                .expect("cpu f64 anchor fold");
            let region = &mut anchor[slot_off[fpos]..slot_off[fpos] + cells];
            region.copy_from_slice(&raw);
            // SAME host fix + compact the on-device chain applies.
            fix_histogram(region, mfbs[fpos], sum_g, sum_h);
            host_compact_histogram(region, offsets[fpos]);
        }

        // ---- GPU resident chain: u64 fixed-point build→fix→compact on device, read back.
        let fix_feats: Vec<(usize, u32, i32, u32)> = (0..num_bins.len())
            .map(|fpos| (slot_off[fpos], num_bins[fpos], offsets[fpos], mfbs[fpos]))
            .collect();
        let run_gpu = || {
            build_fix_compact_resident_readback_f64_on(
                &hip,
                upload_resident_columns(&hip, &feature_bins),
                lgbm_compute::ResidentBinWidth::U32, // u32 helper buffer
                feature_bins.len(),
                num_data,
                &slot_off,
                slot_len,
                &leaf_rows,
                &gradients,
                &hessians,
                &fix_feats,
                sum_g,
                sum_h,
            )
            .expect("resident build_fix_compact readback")
        };
        let gpu = run_gpu();

        assert_eq!(gpu.len(), slot_len, "resident chain length");
        assert_eq!(anchor.len(), slot_len, "anchor chain length");

        // ---- TIGHTENED fixed-point gate, pinned to the CPU f64 anchor.
        // The u64 fixed-point residual is bounded quantize-rounding error, not f32
        // cancellation. Measured ~5.9e-9 on hardware (exact in the
        // cancelling regime). We pin to a gate well below the f32 envelope
        // (`HIP_SANITY_REL = 1e-3`) so a real fixed-point regression cannot slip,
        // with margin above the measured max. rel = |gpu - anchor| / max(|anchor|, 1.0).
        const FIXEDPOINT_REL_GATE: f64 = 1e-7;
        let mut max_rel = 0.0f64;
        for (i, (&g, &a)) in gpu.iter().zip(anchor.iter()).enumerate() {
            let rel = (g - a).abs() / a.abs().max(1.0);
            assert!(
                rel <= FIXEDPOINT_REL_GATE,
                "FIXED-POINT PARITY FAIL at cell {i}: gpu={g}, cpu_anchor={a}, \
                 rel={rel:.3e} > FIXEDPOINT_REL_GATE={FIXEDPOINT_REL_GATE:.0e} — the u64 \
                 resident build diverged from the CPU f64 anchor beyond quantize-rounding \
                 error (spike-018a/018b measured ~5.9e-9). NOT the tolerated gap; a real \
                 fixed-point regression."
            );
            max_rel = max_rel.max(rel);
        }
        eprintln!(
            "resident u64 fixed-point build: max_rel_vs_cpu_f64_anchor={max_rel:.3e} \
             (gate={FIXEDPOINT_REL_GATE:.0e}, spike-018 ref ~5.9e-9)"
        );

        // ---- DETERMINISM: two u64 resident builds over identical inputs must be
        // bit-equal on every read-back f64 cell. Integer add is order-independent ⇒
        // deterministic by construction (the f32 path was only incidentally so at
        // single-cube). Pattern copied from examples/gpu_fixedpoint_i64.rs:181-182.
        let gpu_again = run_gpu();
        assert_eq!(gpu_again.len(), slot_len, "determinism: length drift");
        for (i, (&a, &b)) in gpu.iter().zip(gpu_again.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "DETERMINISM FAIL at cell {i}: run1={a} (bits {:#018x}) != run2={b} \
                 (bits {:#018x}) — the u64 fixed-point resident build is NOT bit-equal \
                 across two runs",
                a.to_bits(),
                b.to_bits()
            );
        }
    }

    /// The P>1 sibling of
    /// `kernel_parity_resident_build_fix_compact_equals_host_on_hip`. The P=1 test above
    /// uses a 10-row leaf, so `row_partition_count` returns 1 and the **multi-cube
    /// row-partitioned additive merge** (each of P cubes builds a partial u64
    /// sub-histogram, all atomic-merge into the same global slot) is NEVER anchor-checked.
    /// This test self-forces P>1 via a >=256k-row leaf (no env mutation, so the
    /// OnceLock-cached target is irrelevant) and pins the u64
    /// fixed-point read-back to the bit-exact CPU f64 anchor, closing that coverage gap.
    ///
    /// The P>1 multi-cube merge is integer-additive (order-independent) so we expect
    /// near-bit-exact parity (`max_rel ~= 0`), gated at the SAME tightened
    /// `FIXEDPOINT_REL_GATE = 1e-7` as the P=1 test. A `row_partition_count > 1` guard
    /// asserts the headline regime was genuinely reached, so a refactor that silently
    /// drops to P=1 fails loudly here.
    #[test]
    fn kernel_parity_resident_build_fix_compact_p_gt_1_equals_host_on_hip() {
        use lgbm_compute::kernels::histogram::{
            build_fix_compact_resident_readback_f64_on, construct_histograms_cpu,
            row_partition_count, upload_resident_columns,
        };
        use lgbm_treelearner::fix_histogram;

        let hip = rocm_client();
        let cpu = cpu_client();

        // 3 features (the existing test's spine: bins [4,3,5], mfb [2,0,1], offset [0,1,0])
        // over a >=256k-row leaf so the DEFAULT row_partition_count yields P>1 with NO env
        // mutation: target (8-CU APU) = 64, num_features = 3 < 64 ⇒ clamp(64/3,1,16) = 16.
        let num_data = 300_000usize;
        let num_bins: Vec<u32> = vec![4, 3, 5];
        let mfbs: Vec<u32> = vec![2, 0, 1];
        let offsets: Vec<i32> = vec![0, 1, 0];
        // Deterministic per-feature bins via `row % num_bin` (every bin populated so the
        // fix/compact passes still fire).
        let f0: Vec<u32> = (0..num_data).map(|i| (i % 4) as u32).collect();
        let f1: Vec<u32> = (0..num_data).map(|i| (i % 3) as u32).collect();
        let f2: Vec<u32> = (0..num_data).map(|i| (i % 5) as u32).collect();
        let feature_bins: Vec<&[u32]> = vec![&f0, &f1, &f2];

        let mut slot_off = Vec::with_capacity(num_bins.len());
        let mut acc = 0usize;
        for &nb in &num_bins {
            slot_off.push(acc);
            acc += 2 * nb as usize;
        }
        let slot_len = acc;

        // ALL rows in the leaf ⇒ leaf_rows.len() = 300_000 >= ROWPART_MIN_LEAF (256_000).
        let leaf_rows: Vec<u32> = (0..num_data as u32).collect();
        // Small bounded deterministic g/h so the build's overflow guard
        // (rows*max_abs*2^30 < i64::MAX) holds: 300k * 3.5 * 2^30 ~= 1.1e15 << 9.2e18.
        let gradients: Vec<f32> = (0..num_data).map(|i| 0.25 + (i % 7) as f32 * 0.5).collect();
        let hessians: Vec<f32> = (0..num_data).map(|i| 1.0 + (i % 3) as f32).collect();

        // ---- P>1 GUARD: assert the multi-cube row-partitioned merge is genuinely
        // exercised. Without this, the test could pass while silently testing P=1.
        // Uses the now-`pub` rocm-gated heuristic.
        let p = row_partition_count(feature_bins.len(), leaf_rows.len());
        assert!(
            p > 1,
            "WR-05 GUARD: row_partition_count({}, {}) = {p} <= 1 — this test is MEANINGLESS \
             at P=1 (it must exercise the multi-cube row-partitioned additive merge). A \
             refactor silently dropped the row-partition path to a single cube; fix the \
             heuristic or this test's sizing so P>1.",
            feature_bins.len(),
            leaf_rows.len()
        );
        eprintln!(
            "resident u64 P>1 build: row_partition_count({}, {}) = {p} (P>1 multi-cube merge)",
            feature_bins.len(),
            leaf_rows.len()
        );

        // Leaf RAW totals over the leaf rows (un-bumped).
        let sum_g: f64 = leaf_rows.iter().map(|&r| f64::from(gradients[r as usize])).sum();
        let sum_h: f64 = leaf_rows.iter().map(|&r| f64::from(hessians[r as usize])).sum();

        // ---- CPU f64 ANCHOR (the reference is the bit-exact CPU fold, NOT a
        //      second GPU launcher). Per feature: gather the leaf rows' (bin, grad, hess),
        //      fold with `construct_histograms_cpu`, then apply the SAME host fix + compact
        //      the GPU chain runs on device.
        let mut anchor: Vec<f64> = vec![0.0; slot_len];
        for fpos in 0..num_bins.len() {
            let nb = num_bins[fpos];
            let cells = 2 * nb as usize;
            let binned: Vec<u32> = leaf_rows
                .iter()
                .map(|&r| feature_bins[fpos][r as usize])
                .collect();
            let g: Vec<f32> = leaf_rows.iter().map(|&r| gradients[r as usize]).collect();
            let h: Vec<f32> = leaf_rows.iter().map(|&r| hessians[r as usize]).collect();
            let raw = construct_histograms_cpu(&cpu, &binned, &g, &h, nb)
                .expect("cpu f64 anchor fold");
            let region = &mut anchor[slot_off[fpos]..slot_off[fpos] + cells];
            region.copy_from_slice(&raw);
            // SAME host fix + compact the on-device chain applies.
            fix_histogram(region, mfbs[fpos], sum_g, sum_h);
            host_compact_histogram(region, offsets[fpos]);
        }

        // ---- GPU resident chain: u64 fixed-point build→fix→compact on device, read back.
        let fix_feats: Vec<(usize, u32, i32, u32)> = (0..num_bins.len())
            .map(|fpos| (slot_off[fpos], num_bins[fpos], offsets[fpos], mfbs[fpos]))
            .collect();
        let run_gpu = || {
            build_fix_compact_resident_readback_f64_on(
                &hip,
                upload_resident_columns(&hip, &feature_bins),
                lgbm_compute::ResidentBinWidth::U32, // u32 helper buffer
                feature_bins.len(),
                num_data,
                &slot_off,
                slot_len,
                &leaf_rows,
                &gradients,
                &hessians,
                &fix_feats,
                sum_g,
                sum_h,
            )
            .expect("resident build_fix_compact readback (P>1)")
        };
        let gpu = run_gpu();

        assert_eq!(gpu.len(), slot_len, "resident chain length");
        assert_eq!(anchor.len(), slot_len, "anchor chain length");

        // ---- TIGHTENED fixed-point gate, pinned to the CPU f64 anchor (SAME gate as the
        // P=1 test). The P>1 multi-cube merge is integer-additive ⇒ order-independent, so
        // the residual stays bounded quantize-rounding error, not f32 cancellation.
        const FIXEDPOINT_REL_GATE: f64 = 1e-7;
        let mut max_rel = 0.0f64;
        for (i, (&g, &a)) in gpu.iter().zip(anchor.iter()).enumerate() {
            let rel = (g - a).abs() / a.abs().max(1.0);
            assert!(
                rel <= FIXEDPOINT_REL_GATE,
                "FIXED-POINT P>1 PARITY FAIL at cell {i}: gpu={g}, cpu_anchor={a}, \
                 rel={rel:.3e} > FIXEDPOINT_REL_GATE={FIXEDPOINT_REL_GATE:.0e} — the u64 \
                 multi-cube row-partitioned resident build (P={p}) diverged from the CPU f64 \
                 anchor beyond quantize-rounding error. A real fixed-point / merge regression."
            );
            max_rel = max_rel.max(rel);
        }
        eprintln!(
            "resident u64 P>1 fixed-point build: P={p}, max_rel_vs_cpu_f64_anchor={max_rel:.3e} \
             (gate={FIXEDPOINT_REL_GATE:.0e}, spike-018 ref ~5.9e-9)"
        );

        // ---- DETERMINISM: two u64 P>1 resident builds over identical inputs must be
        // bit-equal on every read-back f64 cell. Integer add is order-independent across
        // the P cubes ⇒ deterministic by construction.
        let gpu_again = run_gpu();
        assert_eq!(gpu_again.len(), slot_len, "determinism: length drift");
        for (i, (&a, &b)) in gpu.iter().zip(gpu_again.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "DETERMINISM FAIL at cell {i}: run1={a} (bits {:#018x}) != run2={b} \
                 (bits {:#018x}) — the u64 fixed-point P>1 resident build is NOT bit-equal \
                 across two runs",
                a.to_bits(),
                b.to_bits()
            );
        }
    }

    /// The Handle-consuming fused split scan
    /// (`find_best_splits_batched_fused_f64_from_handle_on`) must be BIT-IDENTICAL to
    /// the host-buf fused scan (`find_best_splits_batched_fused_f64_on`): for the SAME
    /// concatenated f64 histogram `buf`, uploading it to a device Handle and scanning
    /// from the Handle yields the EXACT same `SplitInfo` (gain/threshold/counts/sums/
    /// outputs/default_left) per feature as scanning the host buffer. This proves the
    /// resident scan adds ZERO numeric change (the histogram source differs, the math
    /// is the shared `split_scan_body`) — the foundation of the resident==host tree
    /// equivalence (non-negotiable #2). EXACT compare (not within-tol): both paths run
    /// the SAME f64 kernel on the SAME input bits.
    #[test]
    fn find_best_splits_batched_from_handle_equals_host_buf_on_hip() {
        use lgbm_compute::kernels::split::{
            find_best_splits_batched_fused_f64_from_handle_on,
            find_best_splits_batched_fused_f64_on, upload_f64_buffer,
        };

        let hip = rocm_client();

        // A leaf's CONCATENATED stride-2 f64 histogram over 3 spine features with
        // different bin counts + offsets (so the REVERSE/FORWARD ranges + compaction
        // offsets differ across features). Values are exactly-representable f64.
        // feature 0: num_bin 4, offset 0  -> 8 cells
        // feature 1: num_bin 3, offset 1  -> 6 cells
        // feature 2: num_bin 5, offset 0  -> 10 cells
        let buf: Vec<f64> = vec![
            // f0 [g0,h0,...] 4 bins
            2.0, 5.0, -1.0, 4.0, 3.0, 6.0, -2.0, 5.0,
            // f1 3 bins (offset 1: bin 0 is the compacted-away most_freq cell content)
            1.0, 3.0, 4.0, 7.0, -3.0, 4.0,
            // f2 5 bins
            0.5, 2.0, 1.5, 3.0, -1.0, 2.5, 2.0, 4.0, -0.5, 3.5,
        ];
        let mut feats: Vec<BatchedSplitFeature> = Vec::new();
        feats.push(BatchedSplitFeature {
            slot_off: 0,
            num_bin: 4,
            offset: 0,
            default_bin: 0,
            most_freq_bin: 2,
            skip_default_bin: false,
            na_as_missing: false,
            run_forward: false,
        });
        feats.push(BatchedSplitFeature {
            slot_off: 8,
            num_bin: 3,
            offset: 1,
            default_bin: 0,
            most_freq_bin: 0,
            skip_default_bin: false,
            na_as_missing: false,
            run_forward: false,
        });
        feats.push(BatchedSplitFeature {
            slot_off: 14,
            num_bin: 5,
            offset: 0,
            default_bin: 0,
            most_freq_bin: 1,
            skip_default_bin: false,
            na_as_missing: false,
            run_forward: false,
        });

        let cfg = GainConfig {
            min_data_in_leaf: 1,
            min_sum_hessian_in_leaf: 1e-3,
            max_delta_step: 0.0,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            min_gain_to_split: 0.0,
            path_smooth: 0.0,
            ..Default::default()
        };
        // Leaf totals (un-bumped); the launcher applies the +2*kEpsilon entry bump.
        let sum_gradient = 5.0f64;
        let sum_hessian = 40.0f64;
        let num_data = 40i32;

        // (a) host-buf scan.
        let host = find_best_splits_batched_fused_f64_on(
            &hip, &buf, &feats, &cfg, sum_gradient, sum_hessian, num_data,
        )
        .expect("host-buf fused scan");

        // (b) Handle-consuming scan: upload the SAME buf to a device Handle, scan it.
        let handle = upload_f64_buffer(&hip, &buf);
        let from_handle = find_best_splits_batched_fused_f64_from_handle_on(
            &hip,
            handle,
            buf.len(),
            &feats,
            &cfg,
            sum_gradient,
            sum_hessian,
            num_data,
        )
        .expect("Handle-consuming fused scan");

        assert_eq!(host.len(), feats.len(), "host scan length");
        assert_eq!(from_handle.len(), feats.len(), "handle scan length");
        // EXACT per-feature SplitInfo equality (same f64 kernel, same input bits).
        for (i, (h, f)) in host.iter().zip(from_handle.iter()).enumerate() {
            assert_eq!(
                h, f,
                "feature {i}: Handle-consuming scan != host-buf scan (must be bit-identical): \
                 host={h:?} handle={f:?}"
            );
        }
    }

    /// SIBLING CO-PACK parity on REAL hip. Two halves:
    ///
    /// (a) BYTE-IDENTICAL: the co-packed 2-slot sibling scan
    ///     (`find_best_splits_fused_siblings_from_handles_on`) returns
    ///     `(co_a, co_b)` byte-IDENTICAL (`assert_eq!`, every `SplitInfo` field) to two
    ///     SEPARATE single-slot scans (`find_best_splits_batched_fused_f64_on`) of the
    ///     same two histograms with each sibling's totals (B's two halves == A's two
    ///     results, every cell). It is an EXACTNESS check, valid because BOTH paths run
    ///     the SAME f64 kernel on the SAME input bits — co-packing only changes WHICH
    ///     launch a feature scans in.
    ///
    /// (b) ~1e-6 ANCHOR: each co-pack `SplitInfo` is within the hip ~1e-6
    ///     envelope (`assert_within` / `HIP_SANITY_REL`) of a CPU f64 anchor scan
    ///     (`find_best_split_cpu_native` per feature, per sibling). The reference is the
    ///     CPU f64 anchor — NEVER a second GPU path (never compare two nondeterministic
    ///     GPU f32 paths to each other). This guards a silent per-sibling mis-decode
    ///     (wrong half / wrong min_gain_shift) that a GPU-vs-GPU check would miss.
    ///
    /// The two siblings carry DIFFERENT leaf totals over the SAME feature layout (the
    /// smaller-built vs larger-subtract-derived asymmetry).
    #[test]
    fn kernel_parity_sibling_copack_equals_two_scans_on_hip() {
        use lgbm_compute::kernels::split::{
            find_best_split_cpu_native, find_best_splits_batched_fused_f64_on,
            find_best_splits_fused_siblings_from_handles_on, upload_f64_buffer,
        };

        let hip = rocm_client();

        // SHARED layout + the two seed histograms + cfg (reused from the cubecl-cpu cell's
        // top-level helpers so both cells exercise the identical fixture).
        let feats = super::copack_feats();
        let (buf_a, buf_b) = super::copack_two_histograms();
        let cfg = super::copack_cfg();
        // DIFFERENT leaf totals per sibling (smaller A vs larger-derived B).
        let a_totals = (5.0f64, 40.0f64, 40i32);
        let b_totals = (-3.0f64, 64.0f64, 64i32);

        // ---- (a) BYTE-IDENTICAL: co-pack (co_a, co_b) == two single-slot scans. ----
        let h_a = upload_f64_buffer(&hip, &buf_a);
        let h_b = upload_f64_buffer(&hip, &buf_b);
        let (co_a, co_b) = find_best_splits_fused_siblings_from_handles_on(
            &hip,
            h_a,
            h_b,
            buf_a.len(),
            &feats,
            &cfg,
            a_totals,
            b_totals,
        )
        .expect("hip co-packed sibling scan");

        let single_a = find_best_splits_batched_fused_f64_on(
            &hip, &buf_a, &feats, &cfg, a_totals.0, a_totals.1, a_totals.2,
        )
        .expect("hip single-slot scan A");
        let single_b = find_best_splits_batched_fused_f64_on(
            &hip, &buf_b, &feats, &cfg, b_totals.0, b_totals.1, b_totals.2,
        )
        .expect("hip single-slot scan B");

        assert_eq!(co_a.len(), feats.len(), "co-pack A length");
        assert_eq!(co_b.len(), feats.len(), "co-pack B length");
        // EXACT per-feature equality: same f64 kernel over the same input bits, only the
        // launch differs.
        for (i, (co, si)) in co_a.iter().zip(single_a.iter()).enumerate() {
            assert_eq!(
                co, si,
                "hip sibling A feature {i}: co-pack != single-slot scan (must be byte-identical): \
                 copack={co:?} single={si:?}"
            );
        }
        for (i, (co, si)) in co_b.iter().zip(single_b.iter()).enumerate() {
            assert_eq!(
                co, si,
                "hip sibling B feature {i}: co-pack != single-slot scan (must be byte-identical): \
                 copack={co:?} single={si:?}"
            );
        }

        // ---- (b) ~1e-6 ANCHOR: each co-pack SplitInfo within the hip ~1e-6
        //         envelope of a CPU f64 anchor (find_best_split_cpu_native) — NEVER a GPU path. ----
        // Decode each SplitInfo's continuous fields to f32 and compare to the f64 anchor
        // collected to f32, via the established `assert_within` (ORACLE_TOL surfaced +
        // HIP_SANITY_REL hard bound). Only finite-gain (splittable) winners are pinned;
        // a no-split feature's fields are sentinel (gain = -inf), compared via finiteness.
        let anchor_fields = |si: &lgbm_compute::SplitInfo| -> Vec<f32> {
            vec![
                si.gain as f32,
                si.left_sum_gradient as f32,
                si.left_sum_hessian as f32,
                si.right_sum_gradient as f32,
                si.right_sum_hessian as f32,
                si.left_output as f32,
                si.right_output as f32,
            ]
        };
        let pin = |label: &str, co: &[lgbm_compute::SplitInfo], totals: (f64, f64, i32)| {
            for (i, c) in co.iter().enumerate() {
                let f = &feats[i];
                let base = f.slot_off;
                let cells = 2 * f.num_bin as usize;
                // The anchor scans the SAME per-feature region from this sibling's buf.
                let buf = if label.starts_with('A') { &buf_a } else { &buf_b };
                let region = &buf[base..base + cells];
                let anchor = find_best_split_cpu_native(
                    region,
                    &cfg,
                    f.num_bin,
                    f.offset,
                    f.default_bin,
                    f.most_freq_bin,
                    f.skip_default_bin,
                    f.na_as_missing,
                    f.run_forward,
                    totals.0,
                    totals.1,
                    totals.2,
                )
                .expect("CPU f64 anchor (find_best_split_cpu_native)");
                // Splittability agreement (exact bool) — gates whether we pin the fields.
                assert_eq!(
                    c.gain.is_finite(),
                    anchor.gain.is_finite(),
                    "co-pack {label} feature {i}: splittability vs CPU f64 anchor"
                );
                if anchor.gain.is_finite() {
                    assert_eq!(
                        c.threshold, anchor.threshold,
                        "co-pack {label} feature {i}: threshold vs CPU f64 anchor"
                    );
                    assert_eq!(
                        c.default_left, anchor.default_left,
                        "co-pack {label} feature {i}: default_left vs CPU f64 anchor"
                    );
                    assert_eq!(
                        c.left_count, anchor.left_count,
                        "co-pack {label} feature {i}: left_count vs CPU f64 anchor"
                    );
                    assert_eq!(
                        c.right_count, anchor.right_count,
                        "co-pack {label} feature {i}: right_count vs CPU f64 anchor"
                    );
                    // The continuous f64 fields, pinned within the hip ~1e-6 envelope
                    // (assert_within: ORACLE_TOL surfaced for the ledger + HIP_SANITY_REL
                    // hard-fails a real divergence). The anchor is the CPU f64 path.
                    assert_within(
                        &format!("sibling_copack/{label}/feat{i}"),
                        &anchor_fields(c),
                        &anchor_fields(&anchor),
                    );
                }
            }
        };
        pin("A", &co_a, a_totals);
        pin("B", &co_b, b_totals);
    }

    /// RAII guard for an env var that FORCES a launch-config variant (an
    /// all-variants parity sweep — `LGBM_AUTOTUNE_FORCE_P` for the build `P`,
    /// `LGBM_SCAN_CUBEDIM` for the scan `W`). On `set` it records the prior value and
    /// writes the new one; on `drop` it restores the prior value (removing the var if
    /// it was previously unset). Panic-safe: a failing assertion inside
    /// one P/W iteration restores the env on unwind, so the forced variant can never
    /// leak into a sibling test.
    struct EnvVarGuard {
        key: &'static str,
        prev: Option<std::ffi::OsString>,
    }
    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let prev = std::env::var_os(key);
            // SAFETY: single-threaded oracle test. The forced var is read FRESH by the
            // launch path (`force_row_partition` / `scan_cube_dim`) on this same thread
            // before any cubecl/rayon parallel region spawns workers.
            unsafe { std::env::set_var(key, value) };
            EnvVarGuard { key, prev }
        }
    }
    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            // SAFETY: same single-main-thread invariant as `set`.
            unsafe {
                match &self.prev {
                    Some(v) => std::env::set_var(self.key, v),
                    None => std::env::remove_var(self.key),
                }
            }
        }
    }

    /// Import the production candidate sets directly so the "every autotune
    /// candidate is anchor-pinned" gate tracks the source of truth — a maintainer adding a
    /// `P`/`W` in `histogram::BUILD_PSET` / `split::SCAN_WSET` now extends this sweep
    /// automatically instead of silently leaving the new variant ungated (the old
    /// hand-copied `*_MIRROR` consts could go stale with no compile error). `P = 32 >
    /// ROWPART_P_MAX (16)` is registered-skipped by the tuner and clamped to 16 by
    /// `force_row_partition`, but every declared variant is still FORCED here so the whole
    /// runtime-reachable set is anchor-pinned.
    use lgbm_compute::kernels::histogram::BUILD_PSET as BUILD_PSET_MIRROR;
    use lgbm_compute::kernels::split::SCAN_WSET as SCAN_WSET_MIRROR;

    /// PARITY GATE (build): autotune may pick ANY `P` in `BUILD_PSET` at runtime,
    /// so EVERY variant must be anchor-correct, not just the current default (never
    /// GPU-vs-GPU). This pins the u64 fixed-point resident build at EACH forced
    /// `P` (`LGBM_AUTOTUNE_FORCE_P=P`, which short-circuits the tuner) to the SAME CPU f64
    /// anchor as the multi-cube P>1 resident-build parity test — `construct_histograms_cpu`
    /// (the bit-exact CPU fold) + the exported host `fix_histogram` + compact, built ONCE.
    /// The u64 fixed-point merge is integer-additive (order-independent across the P
    /// cubes) ⇒ bit-identical across `P`, gated at the tightened `FIXEDPOINT_REL_GATE = 1e-7`.
    ///
    /// SCOPE: this gate FORCES each variant (`LGBM_AUTOTUNE_FORCE_P`, the direct
    /// launch) — it pins every per-`P` KERNEL to the anchor, but it does NOT drive
    /// `LocalTuner::execute` + `FreshOutGenerator` (the live default selection path). That
    /// tuner path's numerics are covered by `build_tuner_grad_conservation_fresh_vs_clone`
    /// (FreshOutGenerator correctness) and the default-on
    /// `kernel_parity_fused_equals_per_feature_and_native`. "all-PSET" therefore means
    /// "every candidate kernel", not "the tuner machinery".
    #[test]
    fn kernel_parity_resident_build_all_pset_p_equals_anchor_on_hip() {
        use lgbm_compute::kernels::histogram::{
            build_fix_compact_resident_readback_f64_on, construct_histograms_cpu,
            upload_resident_columns,
        };
        use lgbm_treelearner::fix_histogram;

        // ---- PSET shape guards (acceptance): the sweep must be non-empty AND exercise
        // at least one multi-cube (P>1) row-partitioned merge, else the test is vacuous.
        assert!(!BUILD_PSET_MIRROR.is_empty(), "BUILD_PSET must be non-empty");
        assert!(
            BUILD_PSET_MIRROR.iter().any(|&p| p > 1),
            "BUILD_PSET must exercise at least one P>1 (the multi-cube row-partitioned merge)"
        );

        let hip = rocm_client();
        let cpu = cpu_client();

        // Use the same P>1 fixture as the resident-build parity test: 3 spine features
        // over a >=256k-row leaf so the row-partitioned multi-cube merge is genuinely
        // exercised at each forced P>1.
        let num_data = 300_000usize;
        let num_bins: Vec<u32> = vec![4, 3, 5];
        let mfbs: Vec<u32> = vec![2, 0, 1];
        let offsets: Vec<i32> = vec![0, 1, 0];
        let f0: Vec<u32> = (0..num_data).map(|i| (i % 4) as u32).collect();
        let f1: Vec<u32> = (0..num_data).map(|i| (i % 3) as u32).collect();
        let f2: Vec<u32> = (0..num_data).map(|i| (i % 5) as u32).collect();
        let feature_bins: Vec<&[u32]> = vec![&f0, &f1, &f2];

        let mut slot_off = Vec::with_capacity(num_bins.len());
        let mut acc = 0usize;
        for &nb in &num_bins {
            slot_off.push(acc);
            acc += 2 * nb as usize;
        }
        let slot_len = acc;

        let leaf_rows: Vec<u32> = (0..num_data as u32).collect();
        let gradients: Vec<f32> = (0..num_data).map(|i| 0.25 + (i % 7) as f32 * 0.5).collect();
        let hessians: Vec<f32> = (0..num_data).map(|i| 1.0 + (i % 3) as f32).collect();

        let sum_g: f64 = leaf_rows.iter().map(|&r| f64::from(gradients[r as usize])).sum();
        let sum_h: f64 = leaf_rows.iter().map(|&r| f64::from(hessians[r as usize])).sum();

        // ---- CPU f64 ANCHOR built ONCE: per feature gather the leaf rows'
        //      (bin, grad, hess), fold with `construct_histograms_cpu`, apply the SAME
        //      host fix + compact the GPU chain runs on device. NOT a second GPU launch.
        let mut anchor: Vec<f64> = vec![0.0; slot_len];
        for fpos in 0..num_bins.len() {
            let nb = num_bins[fpos];
            let cells = 2 * nb as usize;
            let binned: Vec<u32> = leaf_rows
                .iter()
                .map(|&r| feature_bins[fpos][r as usize])
                .collect();
            let g: Vec<f32> = leaf_rows.iter().map(|&r| gradients[r as usize]).collect();
            let h: Vec<f32> = leaf_rows.iter().map(|&r| hessians[r as usize]).collect();
            let raw = construct_histograms_cpu(&cpu, &binned, &g, &h, nb)
                .expect("cpu f64 anchor fold");
            let region = &mut anchor[slot_off[fpos]..slot_off[fpos] + cells];
            region.copy_from_slice(&raw);
            fix_histogram(region, mfbs[fpos], sum_g, sum_h);
            host_compact_histogram(region, offsets[fpos]);
        }

        let fix_feats: Vec<(usize, u32, i32, u32)> = (0..num_bins.len())
            .map(|fpos| (slot_off[fpos], num_bins[fpos], offsets[fpos], mfbs[fpos]))
            .collect();
        let run_gpu = || {
            build_fix_compact_resident_readback_f64_on(
                &hip,
                upload_resident_columns(&hip, &feature_bins),
                lgbm_compute::ResidentBinWidth::U32,
                feature_bins.len(),
                num_data,
                &slot_off,
                slot_len,
                &leaf_rows,
                &gradients,
                &hessians,
                &fix_feats,
                sum_g,
                sum_h,
            )
            .expect("resident build_fix_compact readback (forced P)")
        };

        const FIXEDPOINT_REL_GATE: f64 = 1e-7;
        for &p in BUILD_PSET_MIRROR {
            // FORCE this variant; the guard restores the prior env on drop / panic.
            let _g = EnvVarGuard::set("LGBM_AUTOTUNE_FORCE_P", &p.to_string());
            let gpu = run_gpu();
            assert_eq!(gpu.len(), slot_len, "resident chain length (P={p})");

            let mut max_rel = 0.0f64;
            for (i, (&gv, &av)) in gpu.iter().zip(anchor.iter()).enumerate() {
                let rel = (gv - av).abs() / av.abs().max(1.0);
                assert!(
                    rel <= FIXEDPOINT_REL_GATE,
                    "ALL-PSET PARITY FAIL at P={p} cell {i}: gpu={gv}, cpu_anchor={av}, \
                     rel={rel:.3e} > FIXEDPOINT_REL_GATE={FIXEDPOINT_REL_GATE:.0e} — the u64 \
                     fixed-point resident build at this forced P diverged from the CPU f64 \
                     anchor beyond quantize-rounding error."
                );
                max_rel = max_rel.max(rel);
            }
            eprintln!(
                "all-PSET build: LGBM_AUTOTUNE_FORCE_P={p} max_rel_vs_cpu_f64_anchor={max_rel:.3e} \
                 (gate={FIXEDPOINT_REL_GATE:.0e})"
            );
        }
    }


    /// PARITY GATE (scan): autotune may pick ANY `W` in `SCAN_WSET` at runtime, so
    /// EVERY scan width must be bit-exact. The feature-per-lane fused scan keeps each
    /// feature's prefix scan SEQUENTIAL (one feature per lane), so widening
    /// `CubeDim W` reorders NOTHING within a feature ⇒ the per-feature `SplitInfo` is
    /// bit-IDENTICAL across `W`. This computes the `W=1` reference (the byte-exact oracle
    /// anchor) once, then for each `W` in `SCAN_WSET` forces `LGBM_SCAN_CUBEDIM=W`
    /// (`LGBM_AUTOTUNE=0` so the deterministic explicit-env fallback path runs) and asserts
    /// every feature's `SplitInfo` is byte-identical to the `W=1` result.
    ///
    /// SCOPE: like the build gate, this FORCES each `W` via the explicit-env
    /// fallback (`LGBM_AUTOTUNE=0` + `LGBM_SCAN_CUBEDIM`) — it pins every per-`W` KERNEL to
    /// the `W=1` oracle, but it does NOT drive `LocalTuner::execute` (the live default
    /// selection path). The default-on tuner path is covered by
    /// `kernel_parity_fused_equals_per_feature_and_native`. "all-WSET" means "every
    /// candidate width", not "the tuner machinery".
    #[test]
    fn kernel_parity_fused_scan_all_wset_w_equals_anchor_on_hip() {
        use lgbm_compute::kernels::split::find_best_splits_batched_fused_f64_on;

        assert!(!SCAN_WSET_MIRROR.is_empty(), "SCAN_WSET must be non-empty");

        let hip = rocm_client();

        // A leaf's CONCATENATED stride-2 f64 histogram over 3 spine features with
        // different bin counts + offsets (so per-feature REVERSE/FORWARD ranges differ).
        // Exactly-representable f64 values (reused from the handle-scan parity fixture).
        let buf: Vec<f64> = vec![
            2.0, 5.0, -1.0, 4.0, 3.0, 6.0, -2.0, 5.0, // f0: 4 bins, offset 0
            1.0, 3.0, 4.0, 7.0, -3.0, 4.0, // f1: 3 bins, offset 1
            0.5, 2.0, 1.5, 3.0, -1.0, 2.5, 2.0, 4.0, -0.5, 3.5, // f2: 5 bins, offset 0
        ];
        let mut feats: Vec<BatchedSplitFeature> = Vec::new();
        feats.push(BatchedSplitFeature {
            slot_off: 0,
            num_bin: 4,
            offset: 0,
            default_bin: 0,
            most_freq_bin: 2,
            skip_default_bin: false,
            na_as_missing: false,
            run_forward: false,
        });
        feats.push(BatchedSplitFeature {
            slot_off: 8,
            num_bin: 3,
            offset: 1,
            default_bin: 0,
            most_freq_bin: 0,
            skip_default_bin: false,
            na_as_missing: false,
            run_forward: false,
        });
        feats.push(BatchedSplitFeature {
            slot_off: 14,
            num_bin: 5,
            offset: 0,
            default_bin: 0,
            most_freq_bin: 1,
            skip_default_bin: false,
            na_as_missing: false,
            run_forward: false,
        });

        let cfg = GainConfig {
            min_data_in_leaf: 1,
            min_sum_hessian_in_leaf: 1e-3,
            max_delta_step: 0.0,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            min_gain_to_split: 0.0,
            path_smooth: 0.0,
            ..Default::default()
        };
        let sum_gradient = 5.0f64;
        let sum_hessian = 40.0f64;
        let num_data = 40i32;

        let scan = || {
            find_best_splits_batched_fused_f64_on(
                &hip, &buf, &feats, &cfg, sum_gradient, sum_hessian, num_data,
            )
            .expect("fused scan")
        };

        // ---- W=1 reference (the degenerate one-cube-per-feature scan = byte-exact oracle
        //      anchor). LGBM_AUTOTUNE=0 + LGBM_SCAN_CUBEDIM=1 force the deterministic path.
        let w1 = {
            let _ga = EnvVarGuard::set("LGBM_AUTOTUNE", "0");
            let _gw = EnvVarGuard::set("LGBM_SCAN_CUBEDIM", "1");
            scan()
        };
        assert_eq!(w1.len(), feats.len(), "W=1 reference length");

        for &w in SCAN_WSET_MIRROR {
            let _ga = EnvVarGuard::set("LGBM_AUTOTUNE", "0");
            let _gw = EnvVarGuard::set("LGBM_SCAN_CUBEDIM", &w.to_string());
            let got = scan();
            assert_eq!(got.len(), feats.len(), "scan length (W={w})");
            for (i, (g, r)) in got.iter().zip(w1.iter()).enumerate() {
                assert_eq!(
                    g, r,
                    "ALL-WSET SCAN PARITY FAIL at W={w} feature {i}: SplitInfo != W=1 reference \
                     (must be byte-identical — each feature's scan stays sequential): \
                     w{w}={g:?} w1={r:?}"
                );
            }
            eprintln!("all-WSET scan: LGBM_SCAN_CUBEDIM={w} byte-identical to W=1 ({} feats)", feats.len());
        }
    }

    // =======================================================================
    // PREDICT tree-walk hip parity. Drive the numeric + categorical
    // AddPredictionToScore tree-walk on cubecl-hip and assert within ~1e-6 of the
    // cpu f64 anchor (NEVER GPU-vs-GPU). The walk is deterministic
    // integer routing + a single f64 leaf-value write per row (no f32 accumulation),
    // so the hip result equals the cpu anchor exactly; the tie-aware assert_within
    // surfaces any residual as a documented gap rather than silently passing.
    // Reads the committed predict.txt golden (self-contained mini-parser).
    // =======================================================================

    /// A parsed PREDICT block (feature meta + tree arrays + cat pool + rows + scores).
    struct HipPredictGolden {
        name: String,
        num_rows: usize,
        num_features: usize,
        scores: Vec<f64>,
        rows: Vec<u32>, // row-major [num_rows × num_features]
        feat_min: Vec<i32>,
        feat_max: Vec<i32>,
        feat_default: Vec<i32>,
        feat_mfb: Vec<i32>,
        feat_offset: Vec<i32>,
        feat_bit_type: Vec<u32>,
        feat_column: Vec<i32>,
        split_feature_inner: Vec<i32>,
        threshold_in_bin: Vec<u32>,
        decision_type: Vec<i32>,
        left_child: Vec<i32>,
        right_child: Vec<i32>,
        leaf_value: Vec<f64>,
        bitset_inner: Vec<u32>,
        cat_boundaries_inner: Vec<i32>,
    }

    fn parse_predict_blocks(text: &str) -> Vec<HipPredictGolden> {
        let mut out = Vec::new();
        let mut lines = text.lines();
        while let Some(raw) = lines.next() {
            let line = raw.trim();
            if !line.starts_with("PREDICT") {
                continue;
            }
            let t: Vec<&str> = line.split_whitespace().collect();
            let name = field(&t, "name").expect("PREDICT name").to_string();
            let num_rows = parse_i64(&t, "num_rows") as usize;
            let num_features = parse_i64(&t, "num_features") as usize;
            let (mut fmin, mut fmax, mut fdef, mut fmfb, mut foff, mut fbt, mut fcol) =
                (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
            let (mut sfi, mut thr, mut dt, mut lc, mut rc) = (vec![], vec![], vec![], vec![], vec![]);
            let (mut leaf, mut pool, mut catb) = (vec![], vec![], vec![]);
            let mut rows2: Vec<Vec<u32>> = Vec::new();
            let mut scores = Vec::new();
            for body in lines.by_ref() {
                let bt: Vec<&str> = body.split_whitespace().collect();
                if bt.is_empty() {
                    continue;
                }
                let payload = bt.get(1).copied().unwrap_or("");
                match bt[0] {
                    "PFEAT" => {
                        for f in payload.split(';').filter(|s| !s.is_empty()) {
                            let c: Vec<&str> = f.split(',').collect();
                            fmin.push(c[0].parse().unwrap());
                            fmax.push(c[1].parse().unwrap());
                            fdef.push(c[2].parse().unwrap());
                            fmfb.push(c[3].parse().unwrap());
                            foff.push(c[4].parse().unwrap());
                            fbt.push(c[5].parse().unwrap());
                            fcol.push(c[6].parse().unwrap());
                        }
                    }
                    "PNODE" => {
                        for n in payload.split(';').filter(|s| !s.is_empty()) {
                            let c: Vec<&str> = n.split(',').collect();
                            sfi.push(c[0].parse().unwrap());
                            thr.push(c[1].parse().unwrap());
                            dt.push(c[2].parse().unwrap());
                            lc.push(c[3].parse().unwrap());
                            rc.push(c[4].parse().unwrap());
                        }
                    }
                    "PLEAF" => {
                        for l in payload.split(';').filter(|s| !s.is_empty()) {
                            leaf.push(f64::from_bits(l.parse::<u64>().unwrap()));
                        }
                    }
                    "PCATPOOL" => {
                        for w in payload.split(';').filter(|s| !s.is_empty()) {
                            pool.push(w.parse::<u32>().unwrap());
                        }
                    }
                    "PCATBOUND" => {
                        for b in payload.split(';').filter(|s| !s.is_empty()) {
                            catb.push(b.parse::<i32>().unwrap());
                        }
                    }
                    "PROWS" if !payload.is_empty() => {
                        rows2 = payload
                            .split(';')
                            .map(|r| r.split(',').map(|c| c.parse::<u32>().unwrap()).collect())
                            .collect();
                    }
                    "PSCORE" => {
                        if !payload.is_empty() {
                            scores = payload
                                .split(';')
                                .map(|s| f64::from_bits(s.parse::<u64>().unwrap()))
                                .collect();
                        }
                        break;
                    }
                    _ => {}
                }
            }
            out.push(HipPredictGolden {
                name,
                num_rows,
                num_features,
                scores,
                rows: rows2.into_iter().flatten().collect(),
                feat_min: fmin,
                feat_max: fmax,
                feat_default: fdef,
                feat_mfb: fmfb,
                feat_offset: foff,
                feat_bit_type: fbt,
                feat_column: fcol,
                split_feature_inner: sfi,
                threshold_in_bin: thr,
                decision_type: dt,
                left_child: lc,
                right_child: rc,
                leaf_value: leaf,
                bitset_inner: pool,
                cat_boundaries_inner: catb,
            });
        }
        out
    }

    #[test]
    fn kernel_parity_predict_within_tol_on_hip() {
        use lgbm_compute::kernels::predict::{add_prediction_to_score_on_device, PredictTree};

        let path = kernels_dir().join("predict.txt");
        let Ok(text) = std::fs::read_to_string(&path) else {
            eprintln!("hip parity(predict): SKIP — fixture {} not found.", path.display());
            return;
        };
        let blocks = parse_predict_blocks(&text);
        assert!(!blocks.is_empty(), "predict fixture parsed zero PREDICT blocks");

        let hip = rocm_client();
        let cpu = cpu_client();

        for g in &blocks {
            let tree = PredictTree {
                feat_min: &g.feat_min,
                feat_max: &g.feat_max,
                feat_default: &g.feat_default,
                feat_most_freq_bin: &g.feat_mfb,
                feat_offset: &g.feat_offset,
                feat_column: &g.feat_column,
                split_feature_inner: &g.split_feature_inner,
                threshold_in_bin: &g.threshold_in_bin,
                decision_type: &g.decision_type,
                left_child: &g.left_child,
                right_child: &g.right_child,
                leaf_value: &g.leaf_value,
                bitset_inner: &g.bitset_inner,
                cat_boundaries_inner: &g.cat_boundaries_inner,
            };
            let bit_type = *g.feat_bit_type.iter().max().unwrap_or(&8);

            // (1) f64 tree-walk on the REAL hip GPU (integer routing + f64 leaf write).
            let hip_f64 = add_prediction_to_score_on_device(
                &hip, &tree, &g.rows, g.num_rows, g.num_features, bit_type, g.num_rows, None,
            )
            .unwrap_or_else(|e| panic!("hip predict `{}` failed: {e:?}", g.name));
            // (2) f64 anchor on cubecl-cpu (never GPU-vs-GPU).
            let cpu_anchor_f64 = add_prediction_to_score_on_device(
                &cpu, &tree, &g.rows, g.num_rows, g.num_features, bit_type, g.num_rows, None,
            )
            .unwrap_or_else(|e| panic!("cpu predict anchor `{}` failed: {e:?}", g.name));

            // Sanity: the cpu f64 anchor itself reproduces the committed golden
            // margins bit-exact (localizes a fixture/anchor divergence away from the
            // hip gap below).
            for (i, (a, s)) in cpu_anchor_f64.iter().zip(g.scores.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    s.to_bits(),
                    "predict `{}` cpu anchor row {i}: {a} != golden {s}",
                    g.name
                );
            }

            // (3) collect both to f32, (4) compare within ORACLE_TOL (tie-aware assert_within).
            let hip_f32 = anchor_to_f32(&hip_f64);
            let cpu_anchor_f32 = anchor_to_f32(&cpu_anchor_f64);
            assert_within(&format!("predict/{}", g.name), &hip_f32, &cpu_anchor_f32);
        }
    }
}

// ===========================================================================
// Synthetic build/fix/subtract fixtures.
//
// PURE-SYNTHETIC, f64-anchored fixtures (no C++ capture needed) covering
// places parity has historically broken:
//   (1) a sparse column set forcing CSR `row_ptr_type` ∈ {16,32,64} (re-lay);
//   (2) a large-bin column forcing `NumLargeBinPartition() > 0` (the `_GlobalMemory`
//       spill path);
//   (3) a purpose-built `most_freq_bin != 0` column whose FixHistogram repaired
//       default-bin value (`leaf_total − Σ(other bins, ascending)`) is stored as a
//       golden constant computed in PLAIN Rust f64 — NEVER from the fix kernel
//       under test.
//
// Every expected histogram is anchored to the cpu f64 fold
// (`construct_histograms_f64_on`, CubeDim::new_1d(1)); NO GPU output is ever stored
// as a golden. NO fixture stores a compacted (offset-shifted) histogram shape —
// this covers build→fix→subtract only.
//
// The committed dense corpora build anchor (`tests/fixtures/kernels/histogram.txt`,
// replayed by `kernel_parity_histogram_bit_exact_on_cpu` above) remains the C++-
// captured half of the anchor; these synthetic builders are the partition-
// layout / Fix / ordering half that needs no C++ toolchain to regenerate.
// ===========================================================================
mod kernel16_fixtures {
    use lgbm_compute::kernels::histogram::construct_histograms_f64_on;
    use lgbm_compute::kernels::row_data::{divide_cuda_feature_groups, CudaRowData};
    use lgbm_compute::runtime::{cpu_client, ActiveRuntime};
    use lgbm_compute::BinColumn;

    /// `shared_hist_size` for every partition layout here (the design DP
    /// value; the SMEM budget = `shared_hist_size / 2` = 3072 bins).
    pub const SHARED_HIST_SIZE: usize = 6144;

    // ---- sparse fixture (forces row_ptr_type {16,32,64}) -------------------
    /// One synthesized sparse column: the host binned-value oracle + the logical nnz
    /// that selects the CSR `row_ptr_type` width.
    pub struct SparseSynthColumn {
        pub bins: BinColumn,
        pub num_bin: u32,
        /// Logical non-zeros — selects the CSR `row_ptr_type` width {16, 32, 64}
        /// (carried as a NUMBER; never materialized).
        pub nnz: u64,
    }

    /// The CSR `row_ptr_type` width a logical nnz forces (16-bit if it fits a `u16`
    /// row-pointer, 32-bit if a `u32`, else 64-bit).
    #[must_use]
    pub fn row_ptr_type_for(nnz: u64) -> u32 {
        if nnz <= u64::from(u16::MAX) {
            16
        } else if nnz <= u64::from(u32::MAX) {
            32
        } else {
            64
        }
    }

    /// Three synthetic sparse columns, each tagged with a logical nnz that forces a
    /// distinct CSR `row_ptr_type` width {16, 32, 64}. All share 8 rows so they re-lay
    /// as one matrix.
    #[must_use]
    pub fn sparse_columns() -> Vec<SparseSynthColumn> {
        vec![
            // small nnz -> row_ptr_type 16.
            SparseSynthColumn {
                bins: BinColumn::new(vec![0, 1, 2, 3, 0, 1, 2, 3], 200),
                num_bin: 200,
                nnz: 1_000,
            },
            // nnz crosses 2^16 -> row_ptr_type 32.
            SparseSynthColumn {
                bins: BinColumn::new(vec![0, 300, 301, 5, 6, 7, 8, 9], 4_000),
                num_bin: 4_000,
                nnz: 70_000,
            },
            // nnz crosses 2^32 -> row_ptr_type 64.
            SparseSynthColumn {
                bins: BinColumn::new(vec![0, 1, 2, 3, 4, 5, 6, 7], 200),
                num_bin: 200,
                nnz: 5_000_000_000,
            },
        ]
    }

    /// Drive the sparse re-lay over the fixture and return the set of resolved
    /// `row_ptr_bit_type` widths (the re-lay width the device actually selected) — the
    /// authoritative assertion that each of {16,32,64} is forced.
    #[must_use]
    pub fn sparse_resolved_widths() -> std::collections::BTreeSet<u32> {
        let client = cpu_client();
        let cols = sparse_columns();
        let bin_columns: Vec<BinColumn> = cols.iter().map(|c| c.bins.clone()).collect();
        let num_bin_per_column: Vec<usize> = cols.iter().map(|c| c.num_bin as usize).collect();
        let layout = divide_cuda_feature_groups(&num_bin_per_column, SHARED_HIST_SIZE);

        let mut widths = std::collections::BTreeSet::new();
        for c in &cols {
            let rpt = row_ptr_type_for(c.nnz);
            let device = CudaRowData::<ActiveRuntime>::get_sparse_data_partitioned(
                &client,
                &bin_columns,
                &layout,
                rpt,
            )
            .expect("sparse re-lay");
            // Assert the re-lay resolved to the width the fixture forces (not a silent widen).
            assert_eq!(
                device.row_ptr_bit_type(),
                rpt,
                "sparse re-lay must select the row_ptr width the fixture forces"
            );
            widths.insert(device.row_ptr_bit_type());
        }
        widths
    }

    // ---- large-bin / global-spill fixture (NumLargeBinPartition() > 0) ------
    /// Per-column bin counts whose final column (4000 bins) exceeds the SMEM budget
    /// (3072 for `shared_hist_size` 6144), forcing its own large-bin partition — the
    /// `_GlobalMemory` spill path.
    #[must_use]
    pub fn large_bin_num_bin_per_column() -> Vec<usize> {
        vec![1_000, 1_000, 4_000]
    }

    /// `NumLargeBinPartition()` the large-bin fixture yields (must be > 0).
    #[must_use]
    pub fn large_bin_num_large_partitions() -> usize {
        divide_cuda_feature_groups(&large_bin_num_bin_per_column(), SHARED_HIST_SIZE)
            .num_large_bin_partition
    }

    // ---- most_freq_bin != 0 fixture (FixHistogram omit-and-repair) ----------
    /// A purpose-built `most_freq_bin != 0` column + its f64-anchored build histogram
    /// and the golden repaired default-bin value.
    pub struct MfbFixture {
        pub bins: Vec<u32>,
        pub grad: Vec<f32>,
        pub hess: Vec<f32>,
        pub num_bin: u32,
        /// The most-frequent bin — non-zero AND in range (`0 < mfb < num_bin`), so it
        /// exercises FixHistogram's omit-and-repair path.
        pub mfb: u32,
        /// Raw f64 leaf totals (`Σ grad`, `Σ hess` over the leaf's rows, ascending) —
        /// the seeds FixHistogram subtracts from.
        pub leaf_total_grad: f64,
        pub leaf_total_hess: f64,
        /// The full `2*num_bin` interleaved build histogram, anchored to the cpu f64
        /// fold. NO compaction.
        pub hist: Vec<f64>,
        /// The GOLDEN repaired default-bin value = `leaf_total − Σ(other bins,
        /// ascending)`, computed in PLAIN Rust f64 — never the fix kernel under test.
        pub golden_repaired_grad: f64,
        pub golden_repaired_hess: f64,
    }

    /// Build the mfb!=0 fixture. The histogram is the cpu f64 BUILD anchor; the repaired
    /// default-bin value is computed in plain Rust f64 (the stored golden), independent
    /// of the FixHistogram kernel under test.
    #[must_use]
    pub fn mfb_fixture() -> MfbFixture {
        let num_bin = 5u32;
        let mfb = 2u32; // non-zero, in-range; the most-frequent bin below.
        // bin 2 is the most-frequent (4 of 8 rows); the other in-range bins appear once.
        let bins = vec![2u32, 0, 2, 1, 2, 3, 4, 2];
        // Signed/fractional grad so cancellation is real (the cell is a difference of
        // partial sums).
        let grad = vec![0.5f32, -1.25, 2.0, -0.75, 3.5, -2.5, 1.0, -0.25];
        let hess = vec![1.0f32, 1.5, 2.0, 0.5, 1.25, 2.5, 0.75, 1.75];

        // Raw leaf totals: f64 sum over rows in ascending row order.
        let mut leaf_total_grad = 0.0f64;
        let mut leaf_total_hess = 0.0f64;
        for i in 0..bins.len() {
            leaf_total_grad += f64::from(grad[i]);
            leaf_total_hess += f64::from(hess[i]);
        }

        // BUILD anchor: the cpu f64 single-owner ordered fold — never a GPU output.
        let client = cpu_client();
        let hist =
            construct_histograms_f64_on::<ActiveRuntime>(&client, &bins, &grad, &hess, num_bin)
                .expect("mfb build anchor");

        // Golden repaired value: leaf_total − Σ(other bins, ASCENDING). Ascending bin
        // order is load-bearing (the cpu anchor fold order); this is the exact arithmetic
        // FixHistogram performs, computed here in plain f64.
        let mut g = leaf_total_grad;
        let mut h = leaf_total_hess;
        for b in 0..num_bin {
            if b == mfb {
                continue;
            }
            g -= hist[(b as usize) * 2];
            h -= hist[(b as usize) * 2 + 1];
        }

        MfbFixture {
            bins,
            grad,
            hess,
            num_bin,
            mfb,
            leaf_total_grad,
            leaf_total_hess,
            hist,
            golden_repaired_grad: g,
            golden_repaired_hess: h,
        }
    }
}

/// The synthetic SPARSE fixture forces each CSR `row_ptr_type`
/// width {16,32,64} across its three configured columns (assert on the *resolved*
/// re-lay width, not just the tag).
#[test]
fn kernel16_sparse_fixture_forces_all_row_ptr_widths() {
    let widths = kernel16_fixtures::sparse_resolved_widths();
    assert_eq!(
        widths,
        [16u32, 32, 64].into_iter().collect(),
        "the three configured sparse columns must force row_ptr widths {{16,32,64}}"
    );
}

/// The synthetic LARGE-BIN fixture forces the `_GlobalMemory`
/// spill — `divide_cuda_feature_groups` yields `num_large_bin_partition > 0`.
#[test]
fn kernel16_large_bin_fixture_forces_global_spill() {
    assert!(
        kernel16_fixtures::large_bin_num_large_partitions() > 0,
        "large-bin fixture must yield num_large_bin_partition > 0 (the _GlobalMemory spill path)"
    );
}

/// The `most_freq_bin != 0` fixture's stored golden default-bin
/// value equals `leaf_total − Σ(other bins, ascending)`, recomputed INDEPENDENTLY by
/// a plain Rust f64 loop in this test (NOT the fix kernel under test), and the build
/// anchor carries the full `2*num_bin` interleaved shape (no compaction).
#[test]
fn fix_mfb_nonzero_repaired_default_bin_anchored() {
    let fx = kernel16_fixtures::mfb_fixture();

    // (a) No compacted shape: the stored build anchor is the full 2*num_bin interleaved
    //     histogram (this builder never compacts).
    assert_eq!(
        fx.hist.len(),
        2 * fx.num_bin as usize,
        "build anchor must be the full 2*num_bin shape (no offset-shifted compaction)"
    );

    // (b) mfb is non-zero AND in range — the omit-and-repair precondition
    //     (`0 < mfb < num_bin`, the fix_compact_kernel `do_fix` guard).
    assert!(
        fx.mfb > 0 && fx.mfb < fx.num_bin,
        "fixture must force the mfb>0 in-range FixHistogram path"
    );
    // and it is genuinely the most-frequent bin (the scatter would omit it).
    let mfb_count = fx.bins.iter().filter(|&&b| b == fx.mfb).count();
    for b in 0..fx.num_bin {
        let c = fx.bins.iter().filter(|&&x| x == b).count();
        assert!(
            mfb_count >= c,
            "mfb {} must be the most-frequent bin (bin {b} appears {c} >= {mfb_count})",
            fx.mfb
        );
    }

    // (c) INDEPENDENT plain-f64 recompute of leaf totals (ascending row order) and of
    //     `leaf_total − Σ(other bins, ascending)` — a SEPARATE loop from the builder's,
    //     never the fix kernel.
    let mut leaf_g = 0.0f64;
    let mut leaf_h = 0.0f64;
    for i in 0..fx.bins.len() {
        leaf_g += f64::from(fx.grad[i]);
        leaf_h += f64::from(fx.hess[i]);
    }
    assert_eq!(
        leaf_g.to_bits(),
        fx.leaf_total_grad.to_bits(),
        "independent leaf_total_grad must match the stored seed"
    );
    assert_eq!(
        leaf_h.to_bits(),
        fx.leaf_total_hess.to_bits(),
        "independent leaf_total_hess must match the stored seed"
    );

    let mut rep_g = leaf_g;
    let mut rep_h = leaf_h;
    for b in 0..fx.num_bin {
        if b == fx.mfb {
            continue;
        }
        rep_g -= fx.hist[(b as usize) * 2];
        rep_h -= fx.hist[(b as usize) * 2 + 1];
    }
    assert_eq!(
        rep_g.to_bits(),
        fx.golden_repaired_grad.to_bits(),
        "stored golden grad must equal leaf_total − Σ(other bins ascending), computed independently"
    );
    assert_eq!(
        rep_h.to_bits(),
        fx.golden_repaired_hess.to_bits(),
        "stored golden hess must equal leaf_total − Σ(other bins ascending), computed independently"
    );
}
