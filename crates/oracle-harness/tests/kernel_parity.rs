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

use lgbm_compute::gain::{get_split_gains, GainConfig};
use lgbm_compute::runtime::cpu_client;
use lgbm_compute::{Backend, CpuBackend};
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
/// `BeforeNumerical`. Bit-identical to `lgbm_compute::gain::get_leaf_gain`.
fn leaf_gain(use_l1: bool, g: f64, h: f64, l1: f64, l2: f64) -> f64 {
    if use_l1 {
        let s = g.signum() * (g.abs() - l1).max(0.0);
        (s * s) / (h + l2)
    } else {
        (g * g) / (h + l2)
    }
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

    for c in &cases {
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
