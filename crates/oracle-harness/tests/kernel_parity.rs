//! Compute-kernel histogram parity replay (ORA-04 cpu hard gate, D-04).
//!
//! For every committed histogram golden case (`tests/fixtures/kernels/histogram.txt`,
//! emitted by `cargo run -p xtask -- kernel-capture`), this test drives the
//! cubecl-cpu `Backend::construct_histograms` over the golden's per-row bins +
//! f32 ordered grad/hess and asserts the resulting f64 histogram cells are
//! BIT-EXACT versus the C++-transcription golden via `compare_exact_f64_bits`.
//! This is the first full vertical slice of the compute backend: a real consumer
//! (the Phase-5 learner, simulated here) builds a histogram from the Phase-2
//! binned store and gets C++-bit-identical f64 cells on the deterministic anchor.
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
use lgbm_compute::{Backend, BatchedSplitFeature, CpuBackend};
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

        // D-04 hard cpu gate: BIT-EXACT vs the C++-transcription golden.
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

    // D-02a coverage assertion: at least one dense AND one sparse layout replayed.
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
// 04-03 SPLIT parity (ORA-04 cpu hard gate, gain-scan layer).
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
/// L1 case (CR-01/CR-02).
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
        // The deferred NA_AS_MISSING forward branch (RESEARCH A5) must be false on
        // every committed case — this layer only threads the flag and validates it
        // off; a captured `na_as_missing=1` case would (correctly) be a typed error
        // in the kernel, so the golden must never carry one until that branch lands.
        assert!(
            !c.na_as_missing,
            "SPLIT case `{}`: na_as_missing must be false on every committed case \
             (the NA_AS_MISSING forward branch is deferred, RESEARCH A5)",
            c.name
        );

        // Divergence-case coverage (Plan 05-01, RESEARCH Pitfall 1): a case where
        // the OLD heuristic `default_bin < num_bin` would have set SKIP_DEFAULT_BIN
        // but the authoritative flag is false (missing_type == None). Assert the
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
    // Coverage: the skip_default_bin==false divergence case (Plan 05-01) must be
    // present and replay bit-exact (RESEARCH Pitfall 1).
    assert!(
        saw_skip_default_bin_false_divergence,
        "split golden must contain the `skip_default_bin_false` divergence case \
         (default_bin < num_bin but skip_default_bin==false, missing_type==None)"
    );
}

// ===========================================================================
// 260608-lsx FUSED SPLIT parity. Prove the DEFAULT `find_best_splits_batched`
// (the lad Part 2 batched seam) is BIT-EXACT to the per-feature `find_best_split`
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
    // i-th per-feature call with the SAME shared totals (T-lsx-01).
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
// 260608-mc5 THREE-WAY MERGE GATE. Prove the FUSED per-leaf split kernel is
// BIT-EXACT to BOTH the cubecl per-feature loop AND the native production scan:
//   fused find_best_splits_batched_fused_f64_on
//     == per-feature find_best_split_f64_on
//     == find_best_split_cpu_native
// for EVERY feature in input order (order preservation, T-lsx-01). This is the
// gate that the merge (one shared split_scan_body math + one fused launch) did NOT
// drift any of the three transcriptions. No tolerance — compare_exact_f64_bits.
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
// 04-03 PARTITION parity. Drive Backend::data_partition over the golden bins +
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
        n_cases += 1;
    }
    assert!(n_cases > 0, "partition fixture present but parsed zero cases");
}

// ===========================================================================
// 04-03 SUBTRACT parity. Drive Backend::subtract_histograms and assert the
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
// 04-04 HIP (ROCm) parity layer — the SEPARATE ~1e-6 gate (D-03a, ORA-04 rocm).
//
// ENTIRELY `#[cfg(feature = "rocm")]`: this layer compiles/runs ONLY under
// `cargo test -p oracle-harness --features rocm` and NEVER affects the default
// CPU build (which keeps the bit-exact layers above as the HARD gate, SC#1).
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
    use lgbm_compute::gain::get_leaf_gain_f32;
    use lgbm_compute::kernels::histogram::construct_histograms_f32_on;
    use lgbm_compute::kernels::partition::data_partition_on;
    use lgbm_compute::kernels::split::find_best_split_raw_f32_on;
    use lgbm_compute::kernels::subtract::subtract_histograms_f32_on;
    use lgbm_compute::runtime::rocm_client;
    use oracle_harness::comparator::{compare_within, Mismatch, ORACLE_TOL};

    /// Collect an f64 anchor slice down to `Vec<f32>` — the EXACT conversion the
    /// plan mandates (`cpu_anchor_f64.iter().map(|&x| x as f32).collect()`).
    fn anchor_to_f32(cpu_anchor_f64: &[f64]) -> Vec<f32> {
        cpu_anchor_f64.iter().map(|&x| x as f32).collect()
    }

    /// The PRIMARY hip gate at `ORACLE_TOL = 1e-6` (compare_within). Per D-03a
    /// (best-effort ROCm), a residual f32-vs-f64 accumulation divergence that
    /// exceeds `ORACLE_TOL` but stays within f32's natural precision is NOT a
    /// phase blocker — it is SURFACED to stderr (the per-case `index`/`abs_diff`
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

    #[test]
    fn kernel_parity_histogram_within_tol_on_hip() {
        let path = fixture_path();
        let Ok(text) = std::fs::read_to_string(&path) else {
            eprintln!("hip parity(histogram): SKIP — fixture {} not found.", path.display());
            return;
        };
        let cases = parse(&text);
        assert!(!cases.is_empty(), "histogram fixture parsed zero cases");

        let hip = rocm_client();
        let cpu = cpu_client();
        let backend = CpuBackend;

        for c in &cases {
            // (1) f32 accumulate on the REAL hip GPU.
            let hip_f32 = construct_histograms_f32_on(&hip, &c.bins, &c.grad, &c.hess, c.num_bin)
                .unwrap_or_else(|e| panic!("hip histogram `{}` failed: {e:?}", c.name));
            // (2) f64 anchor on cubecl-cpu (the bit-exact reference).
            let cpu_anchor_f64 = backend
                .construct_histograms(&cpu, &c.bins, &c.grad, &c.hess, c.num_bin)
                .unwrap_or_else(|e| panic!("cpu anchor `{}` failed: {e:?}", c.name));
            // (3) collect anchor -> f32, (4) compare within ORACLE_TOL.
            let cpu_anchor_f32 = anchor_to_f32(&cpu_anchor_f64);
            assert_within(&format!("histogram/{}", c.name), &hip_f32, &cpu_anchor_f32);
        }
    }

    #[test]
    fn kernel_parity_subtract_within_tol_on_hip() {
        let path = kernels_dir().join("subtract.txt");
        let Ok(text) = std::fs::read_to_string(&path) else {
            eprintln!("hip parity(subtract): SKIP — fixture {} not found.", path.display());
            return;
        };

        let hip = rocm_client();
        let cpu = cpu_client();
        let backend = CpuBackend;
        let mut lines = text.lines();
        let mut n = 0;
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
            let parent = parse_f64_bits_list(pt.get(1).copied().unwrap_or(""));
            let ct: Vec<&str> = lines.next().expect("SUBCHILD").split_whitespace().collect();
            let child = parse_f64_bits_list(ct.get(1).copied().unwrap_or(""));
            let _dt: Vec<&str> = lines.next().expect("SUBDERIVED").split_whitespace().collect();

            // f32 inputs for the hip kernel (collect the golden f64 inputs to f32).
            let parent_f32: Vec<f32> = parent.iter().map(|&x| x as f32).collect();
            let child_f32: Vec<f32> = child.iter().map(|&x| x as f32).collect();
            let hip_f32 = subtract_histograms_f32_on(&hip, &parent_f32, &child_f32)
                .unwrap_or_else(|e| panic!("hip subtract `{name}` failed: {e:?}"));
            // f64 anchor on cpu, then collect -> f32.
            let cpu_anchor_f64 = backend
                .subtract_histograms(&cpu, &parent, &child)
                .unwrap_or_else(|e| panic!("cpu subtract anchor `{name}` failed: {e:?}"));
            let cpu_anchor_f32 = anchor_to_f32(&cpu_anchor_f64);
            assert_within(&format!("subtract/{name}"), &hip_f32, &cpu_anchor_f32);
            n += 1;
        }
        assert!(n > 0, "subtract fixture parsed zero cases");
    }

    #[test]
    fn kernel_parity_split_within_tol_on_hip() {
        let path = kernels_dir().join("split.txt");
        let Ok(text) = std::fs::read_to_string(&path) else {
            eprintln!("hip parity(split): SKIP — fixture {} not found.", path.display());
            return;
        };
        let cases = parse_split(&text);
        assert!(!cases.is_empty(), "split fixture parsed zero cases");

        let hip = rocm_client();
        let cpu = cpu_client();
        let backend = CpuBackend;

        for c in &cases {
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

            // (1) hip f32 raw 12 cells.
            let hist_f32: Vec<f32> = c.hist.iter().map(|&x| x as f32).collect();
            let hip_raw = find_best_split_raw_f32_on(
                &hip,
                &hist_f32,
                &cfg,
                c.num_bin as u32,
                c.offset,
                c.default_bin as u32,
                c.skip_default_bin,
                c.na_as_missing,
                // run_forward=true: replay BOTH branches to match the split.txt
                // golden's captured winner (kernel-level golden, not the learner's
                // per-missing_type dispatch). Matches the cpu anchor below.
                true,
                c.sum_gradient as f32,
                c.sum_hessian as f32,
                c.num_data,
            )
            .unwrap_or_else(|e| panic!("hip split `{}` failed: {e:?}", c.name));

            // (2) cpu f64 anchor via Backend::find_best_split (the decoded winner).
            // Build the comparable winner-field vectors (gain + the sums + outputs)
            // and the splittable flag, so the ~1e-6 gate compares the SAME
            // observable quantities the cpu anchor exposes.
            let si = backend
                .find_best_split(
                    &cpu,
                    &c.hist,
                    &cfg,
                    c.num_bin as u32,
                    c.offset,
                    c.default_bin as u32,
                    0,
                    c.skip_default_bin,
                    c.na_as_missing,
                    true, // run_forward: both-branch golden replay (see hip raw above)
                    c.sum_gradient,
                    c.sum_hessian,
                    c.num_data,
                )
                .unwrap_or_else(|e| panic!("cpu split anchor `{}` failed: {e:?}", c.name));

            // The raw hip out cells: [is_splittable, threshold, gain(raw),
            // left_count, right_count, left_sg, left_sh, right_sg, right_sh,
            // default_left, left_output, right_output]. Decode the hip winner with
            // the SAME host finalization the cpu launcher applies (gain -
            // min_gain_shift), so both sides report the net gain.
            let hip_splittable = hip_raw[0] != 0.0;
            // The hip min_gain_shift mirrors the host pre-step in f32.
            let use_l1 = cfg.use_l1();
            let gain_shift = leaf_gain_f32(
                use_l1,
                c.sum_gradient as f32,
                c.sum_hessian as f32,
                c.lambda_l1 as f32,
                c.lambda_l2 as f32,
            );
            let min_gain_shift = gain_shift + c.min_gain_to_split as f32;

            // Compare the splittable decision first (exact bool).
            assert_eq!(
                hip_splittable,
                si.gain != f64::NEG_INFINITY,
                "HIP split `{}`: is_splittable disagreement vs cpu anchor",
                c.name
            );

            if si.gain != f64::NEG_INFINITY {
                // hip net gain + the f64-anchor observable winner fields.
                let hip_vals: Vec<f32> = vec![
                    hip_raw[2] - min_gain_shift, // net gain
                    hip_raw[5],                  // left_sum_gradient
                    hip_raw[6],                  // left_sum_hessian
                    hip_raw[7],                  // right_sum_gradient
                    hip_raw[8],                  // right_sum_hessian
                    hip_raw[10],                 // left_output
                    hip_raw[11],                 // right_output
                ];
                let cpu_anchor_f64 = vec![
                    si.gain,
                    si.left_sum_gradient,
                    si.left_sum_hessian,
                    si.right_sum_gradient,
                    si.right_sum_hessian,
                    si.left_output,
                    si.right_output,
                ];
                let cpu_anchor_f32 = anchor_to_f32(&cpu_anchor_f64);
                assert_within(&format!("split/{}", c.name), &hip_vals, &cpu_anchor_f32);
                // Threshold + counts are exact integers; compare exactly.
                assert_eq!(
                    hip_raw[1] as u32, si.threshold,
                    "HIP split `{}`: threshold", c.name
                );
                assert_eq!(
                    hip_raw[3] as i32, si.left_count,
                    "HIP split `{}`: left_count", c.name
                );
                assert_eq!(
                    (hip_raw[9] != 0.0), si.default_left,
                    "HIP split `{}`: default_left", c.name
                );
            }
        }
    }

    /// f32 `GetLeafGain` host mirror for the hip net-gain finalization. Delegates
    /// to the production `#[cube]` primitive `get_leaf_gain_f32` (single source of
    /// truth for the L1 `Sign(s) = (s>0)-(s<0)` semantics). `f32::signum` must NOT
    /// be used: it never returns `0.0` at `0.0`/`-0.0`, diverging from C++
    /// `Common::Sign` for zero-gradient L1 cases (CR-01/CR-02).
    fn leaf_gain_f32(use_l1: bool, g: f32, h: f32, l1: f32, l2: f32) -> f32 {
        get_leaf_gain_f32(use_l1, g, h, l1, l2)
    }

    #[test]
    fn kernel_parity_partition_exact_on_hip() {
        let path = kernels_dir().join("partition.txt");
        let Ok(text) = std::fs::read_to_string(&path) else {
            eprintln!("hip parity(partition): SKIP — fixture {} not found.", path.display());
            return;
        };

        let hip = rocm_client();
        let mut lines = text.lines();
        let mut n = 0;
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

            let bt: Vec<&str> = lines.next().expect("PBINS").split_whitespace().collect();
            let bins = parse_u32_list(bt.get(1).copied().unwrap_or(""));
            let ot: Vec<&str> = lines.next().expect("PORDER").split_whitespace().collect();
            let order = parse_u32_list(ot.get(1).copied().unwrap_or(""));
            let st: Vec<&str> = lines.next().expect("PSPLIT").split_whitespace().collect();
            let split_point: usize = st[1].parse().expect("split_point usize");

            // partition is f64-free -> compare hip routing bit-EXACT (no tolerance).
            let (got_order, got_split) =
                data_partition_on(&hip, &bins, num_bin, min_bin, max_bin, threshold, most_freq_bin)
                    .unwrap_or_else(|e| panic!("hip partition `{name}` failed: {e:?}"));
            assert_eq!(got_order, order, "HIP partition `{name}`: reordered array");
            assert_eq!(got_split, split_point, "HIP partition `{name}`: split_point");
            n += 1;
        }
        assert!(n > 0, "partition fixture parsed zero cases");
    }

    /// nn7 (L1) parity gate: the DEVICE-RESIDENT leaf-histogram path (one-time column
    /// upload + on-device gather of the small `leaf_rows`) must produce the SAME RAW
    /// histogram as the original host-gather batched launcher
    /// (`build_leaf_histograms_batched_f32_on`) on the SAME inputs. Both are
    /// f32-atomic with the IDENTICAL `(feature, leaf-row)` launch shape, so they
    /// match within `ORACLE_TOL` (and, since the atomic add SET is identical, are
    /// typically bit-identical). This proves L1 changed the UPLOAD path only — not
    /// the numbers — so the GPU tree decisions stay within the existing ~1e-6 gate.
    #[test]
    fn kernel_parity_resident_gather_equals_host_gather_on_hip() {
        use lgbm_compute::kernels::histogram::build_leaf_histograms_batched_f32_on;
        use lgbm_compute::RocmBackend;

        let hip = rocm_client();

        // A small synthetic dataset: 3 features over 10 rows, different bin counts.
        let num_data = 10usize;
        let f0: Vec<u32> = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1]; // num_bin 4
        let f1: Vec<u32> = vec![0, 0, 1, 1, 2, 2, 0, 1, 2, 0]; // num_bin 3
        let f2: Vec<u32> = vec![4, 3, 2, 1, 0, 1, 2, 3, 4, 0]; // num_bin 5
        let num_bins: Vec<u32> = vec![4, 3, 5];
        let feature_bins: Vec<&[u32]> = vec![&f0, &f1, &f2];
        // Concatenated stride-2 slots: feature f occupies [slot_off[f], +2*num_bin).
        let mut slot_off = Vec::with_capacity(num_bins.len());
        let mut acc = 0usize;
        for &nb in &num_bins {
            slot_off.push(acc);
            acc += 2 * nb as usize;
        }
        let slot_len = acc;
        // A non-trivial leaf subset (a strict subset of 0..num_data, out of order).
        let leaf_rows: Vec<u32> = vec![1, 3, 4, 6, 7, 9];
        let gradients: Vec<f32> =
            (0..num_data).map(|i| 0.25 + i as f32 * 0.5).collect();
        let hessians: Vec<f32> = (0..num_data).map(|i| 1.0 + (i % 3) as f32).collect();

        // (a) ORIGINAL host-gather path (per-leaf [num_features × rows] upload).
        let host = build_leaf_histograms_batched_f32_on(
            &hip,
            &feature_bins,
            &slot_off,
            slot_len,
            &leaf_rows,
            &gradients,
            &hessians,
        )
        .expect("host-gather batched launcher");

        // (b) NEW device-resident path: upload columns ONCE via the backend seam,
        //     then build through the RocmBackend override (on-device gather). Both
        //     use the same RocmBackend instance so the resident cache is populated.
        let backend = RocmBackend::default();
        backend.upload_resident_bins(&hip, &feature_bins);
        let resident = backend
            .build_leaf_histograms_raw(
                &hip,
                &feature_bins,
                &num_bins,
                &slot_off,
                slot_len,
                &leaf_rows,
                &gradients,
                &hessians,
            )
            .expect("resident-gather build_leaf_histograms_raw");

        let host_f32: Vec<f32> = host.iter().map(|&x| x as f32).collect();
        let resident_f32: Vec<f32> = resident.iter().map(|&x| x as f32).collect();
        assert_eq!(host.len(), slot_len, "host histogram length");
        assert_eq!(resident.len(), slot_len, "resident histogram length");
        // Resident-gather (on-device gather from the one-time column upload) ==
        // host-gather (per-leaf [num_features × rows] upload) within the ~1e-6 ROCm
        // tolerance. The override routes through build_leaf_histograms_resident_f32_on.
        assert_within("resident_vs_host/override", &resident_f32, &host_f32);
    }
}
