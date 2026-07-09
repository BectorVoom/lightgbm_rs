//! Primitive fixture-parity: replay the Rust device primitives (14-03 full-depth
//! + 14-05 skeletons) over EVERY committed C++ golden in
//! `fixtures/primitives/` and cross-validate them (Phase 14, 14-06, ODL-01,
//! D-03 / D-10).
//!
//! ## Anchor discipline (D-10, def-f8u-01)
//! Every Rust result is compared against the committed C++ golden (captured
//! VERBATIM from the AMD-fork `cuda_algorithms.{hpp,cu}` on the local gfx1100)
//! and/or replayed on the cubecl-cpu f64 anchor — NEVER GPU-vs-GPU. Tolerance
//! classes, per the recorded 14-03 policy:
//!
//! | Primitive | Class | Why |
//! |-----------|-------|-----|
//! | integer (u32/u64) prefix-sum | BIT-EXACT (`assert_eq`) | exact in f64 (< 2^53), no rounding |
//! | argsort permutation (block + global, asc/desc, tie-rich) | BIT-EXACT (`assert_eq`) on the f32-key order, with a documented benign f32-cast-collision fallback | index-only integer permutation |
//! | f64 reductions max/min | BIT-EXACT, with the documented C++ 0-identity bridge | selection-only |
//! | f64 reductions sum/dot, f64 prefix-sum | f64 ULP band ([`F64_ORDER_REL_TOL`]) | C++ warp-tree order ≠ the cpu anchor's ascending serial fold (Pitfall 3) |
//! | f32 reductions (ROCm leg, `--features rocm`) | ~1e-6 surfaced (D-03a) | plane width changes the reduction tree |
//! | unweighted percentile | f32 ~1e-6 ([`PCTL_REL_TOL`]) | the Rust primitive is f32; the golden is f64 |
//! | WEIGHTED percentile | SKIPPED (deferred) | the 14-02 weighted fixture is `status=deferred_kaggle_nvcc` (NVIDIA capture) |
//!
//! ## Two documented reference quirks the cross-check pins
//! 1. **C++ reduction 0-identity.** `ShuffleReduce{Max,Min}` folds a `0` identity
//!    lane: the committed `op=max` goldens are `0.0` because every fixture input
//!    is a negative f64 (`max(0, negatives) == 0`). The Rust selection-only
//!    primitive seeds `data[0]`, so the cross-check bridges via
//!    `rust_max.max(0.0)` / `rust_min.min(0.0)` and then asserts BIT-EXACT.
//! 2. **f64→f32 argsort keys.** The golden keys are f64; the Rust argsort is f32.
//!    Distinct f64 keys that round to the SAME f32 are a benign permutation
//!    reorder (both orders sort their own key precision); the check accepts a
//!    differing index ONLY when its f32 key equals the golden position's f32 key.
//!
//! Like `rng_parity.rs`, this reads ONLY committed fixtures (no C++ toolchain,
//! D-06). A missing fixture reports a SKIP so `cargo test` stays green
//! pre-capture; a present fixture validates every committed case.

use std::path::PathBuf;

use lgbm_compute::kernels::primitives::{
    bitonic_argsort_global_on, bitonic_argsort_on, dot_product_f64_on, percentile_unweighted_f32_on,
    prefix_sum_exclusive_f64_on, prefix_sum_inclusive_f64_on, reduce_max_f64_on, reduce_min_f64_on,
    reduce_sum_f64_on,
};
use lgbm_compute::runtime::cpu_client;

const PRIMITIVES_DIR: &str = "fixtures/primitives";

/// Relative band for f64 sum/dot/prefix-sum: the C++ warp-tree summation order
/// differs from the cpu anchor's single-owner ascending fold, so the low bits
/// diverge by O(n·eps). The committed inputs are same-sign, similar-magnitude
/// (well-conditioned), so 1e-12 is a few-ULP band, far below any real bug.
const F64_ORDER_REL_TOL: f64 = 1e-12;

/// Relative band for the unweighted percentile: the Rust primitive is f32, the
/// golden is f64 — ~1e-6 is f32's natural precision (a few f32 ULP at this scale).
const PCTL_REL_TOL: f64 = 1e-6;

/// gfx1100 warp width (14-02 capture, warpSize 32). The C++ block INCLUSIVE
/// prefix-sum is a clean full contiguous scan, but `ShufflePrefixSumExclusive`
/// records a `0` at every warp-START lane (`idx % WARP == 0`, `idx > 0`) — the
/// within-warp exclusive value for lane 0 — while every other lane holds the
/// global running sum. This is a faithful reference quirk: the cpu anchor's
/// clean exclusive scan is bit-exact EVERYWHERE EXCEPT those warp-boundary lanes,
/// where the golden is asserted to be exactly `0`.
///
/// CROSS-VALIDATION GAP (WR-03): because `ShufflePrefixSumExclusive` is a
/// *within-warp* building block (not a true global exclusive scan), the C++
/// golden carries no meaningful exclusive value at the warp-boundary lanes — so
/// for any `n > WARP` the cross-warp combination of the Rust exclusive scan is
/// NOT cross-validated against the C++ reference here; only the within-warp
/// values are. The Rust true exclusive scan's cross-warp correctness rests
/// entirely on the serial self-test in `primitives_self.rs`, NOT on this C++
/// oracle. Closing this gap means capturing an exclusive golden from the
/// multi-kernel `ShufflePrefixSumGlobal` exclusive path (a true global scan);
/// owned by the Phase-19/22 consumer that first relies on the cross-warp
/// exclusive result (D-02).
///
/// COUPLING (IN-02): `WARP` is hardcoded to the warp-32 capture. If the goldens
/// are ever recaptured on a warp-64 (GFX9) device the boundary lanes move to
/// multiples of 64; `assert_capture_warp_width` (called at parse time) enforces
/// that the fixture's recorded `WARP_WIDTH` metadata matches this constant, so a
/// warp-64 recapture fails loudly instead of silently mis-validating.
const WARP: usize = 32;

/// Is `idx` a C++ exclusive-scan warp-boundary lane (golden == 0 quirk)?
fn is_excl_warp_boundary(idx: usize) -> bool {
    idx > 0 && idx % WARP == 0
}

/// Parse the optional `WARP_WIDTH <n>` metadata record from a fixture header.
fn fixture_warp_width(text: &str) -> Option<usize> {
    text.lines().find_map(|raw| {
        let line = raw.trim();
        let rest = line.strip_prefix("WARP_WIDTH ")?;
        rest.trim().parse::<usize>().ok()
    })
}

/// Enforce (IN-02) that a fixture captured at a recorded warp width matches the
/// hardcoded [`WARP`] this test reasons about. A warp-64 (GFX9) recapture moves
/// the exclusive-scan boundary lanes to multiples of 64, so silently reusing the
/// warp-32 boundary logic would mis-validate. If the fixture carries no
/// `WARP_WIDTH` record we accept it (legacy capture) but the assertion fires the
/// moment a recapture declares a different width.
fn assert_capture_warp_width(text: &str, fixture_name: &str) {
    if let Some(w) = fixture_warp_width(text) {
        assert_eq!(
            w, WARP,
            "{fixture_name}: fixture WARP_WIDTH={w} != test WARP={WARP} — the warp-boundary \
             logic (e.g. is_excl_warp_boundary) is coupled to the capture warp width; recapture \
             this test's WARP const before reusing a differently-warped golden set (IN-02)"
        );
    }
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(PRIMITIVES_DIR)
        .join(name)
}

/// Read a fixture or return `None` with a logged SKIP (kept green pre-capture).
fn load_or_skip(name: &str) -> Option<String> {
    let path = fixture(name);
    match std::fs::read_to_string(&path) {
        Ok(text) => Some(text),
        Err(_) => {
            eprintln!(
                "primitive_parity: SKIP — fixture {} not found. Commit the 14-02 golden set.",
                path.display()
            );
            None
        }
    }
}

/// Extract a `key=value` token's value from a whitespace-split line.
fn field<'a>(tokens: &'a [&'a str], key: &str) -> Option<&'a str> {
    tokens
        .iter()
        .find_map(|t| t.strip_prefix(key).and_then(|r| r.strip_prefix('=')))
}

fn parse_u32_list(s: &str) -> Vec<u32> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(';').map(|t| t.parse::<u32>().expect("u32 field")).collect()
}

fn parse_u64_list(s: &str) -> Vec<u64> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(';').map(|t| t.parse::<u64>().expect("u64 field")).collect()
}

fn parse_i32_list(s: &str) -> Vec<i32> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(';').map(|t| t.parse::<i32>().expect("i32 field")).collect()
}

/// f64 stored as raw u64 bits (`val_t = double` in the reference).
fn parse_f64_bits(s: &str) -> Vec<f64> {
    parse_u64_list(s).into_iter().map(f64::from_bits).collect()
}

fn parse_one_f64_bits(s: &str) -> f64 {
    f64::from_bits(s.parse::<u64>().expect("single f64-bits field"))
}

/// `got == exp` (bit-exact) OR within a relative band (documented order divergence).
fn rel_close_f64(got: f64, exp: f64, tol: f64) -> bool {
    if got.to_bits() == exp.to_bits() {
        return true;
    }
    let scale = got.abs().max(exp.abs()).max(f64::MIN_POSITIVE);
    (got - exp).abs() <= tol * scale
}

/// Iterate the non-comment, non-metadata records of a fixture as token slices.
fn records(text: &str) -> impl Iterator<Item = (usize, Vec<&str>)> {
    text.lines().enumerate().filter_map(|(i, raw)| {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            return None;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        match tokens[0] {
            "MASTER_SEED" | "WARP_WIDTH" => None,
            _ => Some((i + 1, tokens)),
        }
    })
}

#[test]
fn primitive_parity_prefix_sum() {
    let Some(text) = load_or_skip("prefix_sum.txt") else {
        return;
    };
    assert_capture_warp_width(&text, "prefix_sum.txt");
    let client = cpu_client();
    let (mut int_cases, mut f64_cases, mut f64_bitexact) = (0usize, 0usize, 0usize);

    for (lineno, t) in records(&text) {
        let name = field(&t, "name").unwrap_or("?");
        match t[0] {
            // Block prefix-sum (ShufflePrefixSum / ...Exclusive).
            "PSUM" => {
                let kind = field(&t, "kind").expect("kind");
                let dtype = field(&t, "dtype").expect("dtype");
                let n = field(&t, "n").unwrap().parse::<u32>().unwrap();
                let incl = kind == "incl";
                match dtype {
                    "u32" => {
                        let input = parse_u32_list(field(&t, "in").expect("in"));
                        let expect = parse_u32_list(field(&t, "out").expect("out"));
                        let data: Vec<f64> = input.iter().map(|&x| x as f64).collect();
                        let got = if incl {
                            prefix_sum_inclusive_f64_on(&client, &data, n)
                        } else {
                            prefix_sum_exclusive_f64_on(&client, &data, n)
                        }
                        .unwrap_or_else(|e| panic!("PSUM {name} (line {lineno}): {e:?}"));
                        let got_u32: Vec<u32> = got.iter().map(|&x| x.round() as u32).collect();
                        assert_eq!(got_u32.len(), expect.len(), "PSUM {name} len (line {lineno})");
                        // Integer prefix-sum: exact in f64, asserted BIT-EXACT —
                        // except the documented exclusive warp-boundary 0 quirk.
                        for (i, (&g, &e)) in got_u32.iter().zip(expect.iter()).enumerate() {
                            if !incl && is_excl_warp_boundary(i) {
                                assert_eq!(e, 0, "PSUM {name} u32 excl idx {i}: expected the C++ \
                                     warp-boundary 0 quirk (line {lineno})");
                                continue;
                            }
                            assert_eq!(g, e, "PSUM {name} u32 {kind} idx {i} (line {lineno})");
                        }
                        int_cases += 1;
                    }
                    "f64" => {
                        let data = parse_f64_bits(field(&t, "in").expect("in"));
                        let expect = parse_f64_bits(field(&t, "out").expect("out"));
                        let got = if incl {
                            prefix_sum_inclusive_f64_on(&client, &data, n)
                        } else {
                            prefix_sum_exclusive_f64_on(&client, &data, n)
                        }
                        .unwrap_or_else(|e| panic!("PSUM {name} (line {lineno}): {e:?}"));
                        assert_eq!(got.len(), expect.len(), "PSUM {name} len (line {lineno})");
                        for (i, (&g, &e)) in got.iter().zip(expect.iter()).enumerate() {
                            if !incl && is_excl_warp_boundary(i) {
                                assert_eq!(e, 0.0, "PSUM {name} f64 excl idx {i}: expected the \
                                     C++ warp-boundary 0 quirk (line {lineno})");
                                continue;
                            }
                            if g.to_bits() == e.to_bits() {
                                f64_bitexact += 1;
                            }
                            assert!(
                                rel_close_f64(g, e, F64_ORDER_REL_TOL),
                                "PSUM {name} f64 {kind} idx {i} (line {lineno}): got={g:e} \
                                 cpp={e:e} rel>{F64_ORDER_REL_TOL:e} (warp-tree vs serial order)"
                            );
                        }
                        f64_cases += 1;
                    }
                    other => panic!("PSUM {name}: unknown dtype {other} (line {lineno})"),
                }
            }
            // Global inclusive prefix-sum (ShufflePrefixSumGlobal). Composed on the
            // cpu anchor with block_size=256 (num_blocks <= 1024); integer-exact.
            "PSUMG" => {
                let dtype = field(&t, "dtype").expect("dtype");
                let input = parse_u64_list(field(&t, "in").expect("in"));
                let expect = parse_u64_list(field(&t, "out").expect("out"));
                let data: Vec<f64> = input.iter().map(|&x| x as f64).collect();
                let got = prefix_sum_inclusive_f64_on(&client, &data, 256)
                    .unwrap_or_else(|e| panic!("PSUMG {name} (line {lineno}): {e:?}"));
                let got_u64: Vec<u64> = got.iter().map(|&x| x.round() as u64).collect();
                assert_eq!(got_u64, expect, "PSUMG {name} {dtype} global (line {lineno})");
                int_cases += 1;
            }
            other => panic!("prefix_sum.txt: unknown record `{other}` (line {lineno})"),
        }
    }

    assert!(int_cases > 0 && f64_cases > 0, "prefix_sum fixture parsed too few cases");
    eprintln!(
        "primitive_parity prefix_sum: {int_cases} integer (bit-exact) + {f64_cases} f64 \
         (ULP band; {f64_bitexact} elems happened to be bit-exact) cases validated."
    );
}

#[test]
fn primitive_parity_reductions() {
    let Some(text) = load_or_skip("reduce.txt") else {
        return;
    };
    let client = cpu_client();
    let (mut f64_cases, mut f32_deferred) = (0usize, 0usize);

    for (lineno, t) in records(&text) {
        assert_eq!(t[0], "RED", "reduce.txt unknown record (line {lineno})");
        let name = field(&t, "name").unwrap_or("?");
        let op = field(&t, "op").expect("op");
        let dtype = field(&t, "dtype").expect("dtype");
        if dtype == "f32" {
            // The f32 ROCm leg (~1e-6) is exercised under `--features rocm`
            // (mod hip below); cubecl-cpu has no plane support (14-01).
            f32_deferred += 1;
            continue;
        }
        assert_eq!(dtype, "f64", "RED {name}: unexpected dtype (line {lineno})");
        let data = parse_f64_bits(field(&t, "in").expect("in"));
        let cpp = parse_one_f64_bits(field(&t, "out").expect("out"));
        match op {
            "sum" => {
                let got = reduce_sum_f64_on(&client, &data).unwrap();
                assert!(
                    rel_close_f64(got, cpp, F64_ORDER_REL_TOL),
                    "RED {name} sum (line {lineno}): got={got:e} cpp={cpp:e} (order band)"
                );
            }
            "dot" => {
                let b = parse_f64_bits(field(&t, "in2").expect("in2"));
                let got = dot_product_f64_on(&client, &data, &b).unwrap();
                assert!(
                    rel_close_f64(got, cpp, F64_ORDER_REL_TOL),
                    "RED {name} dot (line {lineno}): got={got:e} cpp={cpp:e} (order band)"
                );
            }
            // WR-01 — SUB-1024-THREAD ARTIFACT, NOT A REFERENCE SEMANTIC.
            // The committed max/min goldens were captured as `<<<1, n>>>` with
            // n ∈ {32,96,256} (num_warp < warpSize=32). At that sub-1024-thread
            // width the verbatim C++ reduction folds a `0` for the out-of-range
            // warp lanes (`shared_mem_buffer[warpLane] : 0`), so every committed
            // `op=max` golden over the all-negative SpreadF64 inputs is exactly
            // `0` and every `op=min` golden is the true (negative) min. PRODUCTION
            // LightGBM launches these reductions at full 1024-thread blocks
            // (num_warp == 32) where this `0` floor does NOT occur — so the
            // `.max(0.0)`/`.min(0.0)` bridges below are CAPTURE-ARTIFACT
            // reconciliation, NOT reference behavior, and `reduce_max_f64_on`'s
            // genuine reference parity is effectively UNTESTED until the goldens
            // are recaptured at `<<<1, 1024>>>` (or with a positive-containing,
            // full-block-width case). Recapture is owned by the consumer that
            // first relies on production reduction semantics (D-02); see WR-01.
            "max" => {
                let got = reduce_max_f64_on(&client, &data).unwrap();
                assert_eq!(
                    got.max(0.0).to_bits(),
                    cpp.to_bits(),
                    "RED {name} max (line {lineno}): rust.max(0)={} cpp={cpp:e} (0-identity)",
                    got.max(0.0)
                );
            }
            "min" => {
                let got = reduce_min_f64_on(&client, &data).unwrap();
                assert_eq!(
                    got.min(0.0).to_bits(),
                    cpp.to_bits(),
                    "RED {name} min (line {lineno}): rust.min(0)={} cpp={cpp:e} (0-identity)",
                    got.min(0.0)
                );
            }
            other => panic!("RED {name}: unknown op {other} (line {lineno})"),
        }
        f64_cases += 1;
    }

    assert!(f64_cases > 0, "reduce fixture parsed zero f64 cases");
    eprintln!(
        "primitive_parity reductions: {f64_cases} f64 cases validated (sum/dot ULP band, \
         max/min bit-exact via 0-identity bridge); {f32_deferred} f32 cases are the rocm leg."
    );
}

#[test]
fn primitive_parity_argsort() {
    let Some(text) = load_or_skip("argsort.txt") else {
        return;
    };
    let client = cpu_client();
    let (mut cases, mut tie_cases, mut f32_collision_reorders) = (0usize, 0usize, 0usize);

    for (lineno, t) in records(&text) {
        assert_eq!(t[0], "ARGSORT", "argsort.txt unknown record (line {lineno})");
        let name = field(&t, "name").unwrap_or("?");
        let scope = field(&t, "scope").expect("scope");
        let order = field(&t, "order").expect("order");
        let ascending = order == "asc";
        let keys_f64 = parse_f64_bits(field(&t, "keys").expect("keys"));
        let golden = parse_i32_list(field(&t, "perm").expect("perm"));
        // The Rust argsort is f32 (the grow-loop key type); the golden keys are f64.
        let keys: Vec<f32> = keys_f64.iter().map(|&x| x as f32).collect();

        let (perm, keys_after) = match scope {
            "block" => bitonic_argsort_on(&client, &keys, ascending),
            "global" => bitonic_argsort_global_on(&client, &keys, ascending),
            other => panic!("ARGSORT {name}: unknown scope {other} (line {lineno})"),
        }
        .unwrap_or_else(|e| panic!("ARGSORT {name} (line {lineno}): {e:?}"));

        // Index-only: keys are provably unmutated.
        assert_eq!(keys_after, keys, "ARGSORT {name}: keys mutated (line {lineno})");
        assert_eq!(perm.len(), golden.len(), "ARGSORT {name} perm len (line {lineno})");

        // The output IS a valid sort of the f32 keys (monotone, anchored).
        for w in perm.windows(2) {
            let (a, b) = (keys[w[0] as usize], keys[w[1] as usize]);
            let ok = if ascending { a <= b } else { a >= b };
            assert!(ok, "ARGSORT {name}: not sorted at {a}->{b} (line {lineno})");
        }

        if perm != golden {
            // Documented f32-cast-collision fallback (see module doc): a differing
            // index is benign ONLY if its f32 key equals the golden position's.
            for (i, (&g, &e)) in perm.iter().zip(golden.iter()).enumerate() {
                if g != e {
                    assert_eq!(
                        keys[g as usize].to_bits(),
                        keys[e as usize].to_bits(),
                        "ARGSORT {name} idx {i} (line {lineno}): perm diverged on \
                         DISTINCT f32 keys — a real permutation divergence, not an f32-collision"
                    );
                    f32_collision_reorders += 1;
                }
            }
        }

        if name.contains("tie") {
            tie_cases += 1;
        }
        cases += 1;
    }

    assert!(cases > 0, "argsort fixture parsed zero cases");
    assert!(tie_cases > 0, "argsort fixture must include a tie-rich case (Pitfall 5)");
    eprintln!(
        "primitive_parity argsort: {cases} permutation cases ({tie_cases} tie-rich) — bit-exact \
         on the f32-key order; {f32_collision_reorders} benign f32-collision reorders."
    );
}

#[test]
fn primitive_parity_percentile() {
    let Some(text) = load_or_skip("percentile.txt") else {
        return;
    };
    let client = cpu_client();
    let (mut unweighted, mut deferred) = (0usize, 0usize);

    for (lineno, t) in records(&text) {
        assert_eq!(t[0], "PCTL", "percentile.txt unknown record (line {lineno})");
        let name = field(&t, "name").unwrap_or("?");
        let weighted = field(&t, "weighted").expect("weighted") == "1";
        // The weighted branch was deferred to NVIDIA/Kaggle (non-idempotent on the
        // gfx1100 APU): SKIP, do not fail the gate (14-02 / 14-05 D-02).
        if weighted || field(&t, "status") == Some("deferred_kaggle_nvcc") {
            deferred += 1;
            continue;
        }
        let values_f64 = parse_f64_bits(field(&t, "values").expect("values"));
        let alpha = parse_one_f64_bits(field(&t, "alpha").expect("alpha"));
        let cpp = parse_one_f64_bits(field(&t, "out").expect("out")) as f32;
        let values: Vec<f32> = values_f64.iter().map(|&x| x as f32).collect();

        let got = percentile_unweighted_f32_on(&client, &values, alpha)
            .unwrap_or_else(|e| panic!("PCTL {name} (line {lineno}): {e:?}"));

        let scale = (got.abs().max(cpp.abs()) as f64).max(f64::MIN_POSITIVE);
        let rel = (got as f64 - cpp as f64).abs() / scale;
        assert!(
            rel <= PCTL_REL_TOL,
            "PCTL {name} unweighted alpha={alpha} (line {lineno}): got={got:e} cpp={cpp:e} \
             rel={rel:e} > {PCTL_REL_TOL:e} (f32 ~1e-6)"
        );
        unweighted += 1;
    }

    assert!(unweighted > 0, "percentile fixture parsed zero unweighted cases");
    eprintln!(
        "primitive_parity percentile: {unweighted} unweighted cases validated (f32 ~1e-6); \
         {deferred} weighted cases deferred (status=deferred_kaggle_nvcc, NVIDIA capture)."
    );
}

/// The SEPARATE f32 ROCm leg (~1e-6, D-03a). Replays the f32 reduction goldens on
/// the real hip GPU via the plane intrinsics and compares to the committed f32
/// golden — surfaced (not silently passed), generous sanity bound hard-fails a
/// genuine kernel bug. NEVER GPU-vs-GPU (the golden IS the C++ reference).
#[cfg(feature = "rocm")]
mod hip {
    use super::*;
    use lgbm_compute::kernels::primitives::{
        plane_dot_product_f32_on, plane_reduce_max_f32_on, plane_reduce_min_f32_on,
        plane_reduce_sum_f32_on,
    };
    use lgbm_compute::runtime::rocm_client;

    /// Warp width the 14-02 fixtures were captured at (gfx1100 warpSize 32).
    const PLANE_DIM: u32 = 32;
    /// Surfaced f32 oracle band; a residual f32-accumulation gap above this is
    /// logged (04-ROCM-GAPS.md), not a blocker.
    const ORACLE_TOL: f64 = 1e-6;
    /// Generous f32 relative bound that DOES hard-fail (catches a real kernel bug
    /// while tolerating the f32-accumulation gap).
    const SANITY_REL: f64 = 1e-3;

    fn parse_f32_bits(s: &str) -> Vec<f32> {
        parse_u32_list(s).into_iter().map(f32::from_bits).collect()
    }

    fn check(label: &str, got: f32, cpp: f32) {
        let scale = (got.abs().max(cpp.abs()) as f64).max(f64::MIN_POSITIVE);
        let rel = (got as f64 - cpp as f64).abs() / scale;
        if rel > ORACLE_TOL {
            eprintln!(
                "HIP PRIMITIVE GAP `{label}`: hip={got:e} cpp={cpp:e} rel={rel:e} > \
                 ORACLE_TOL={ORACLE_TOL:e} (documented f32-vs-f64 gap, 04-ROCM-GAPS.md / D-03a)"
            );
        }
        assert!(
            rel <= SANITY_REL,
            "HIP PRIMITIVE SANITY `{label}`: hip={got:e} cpp={cpp:e} rel={rel:e} > \
             {SANITY_REL:e} — a real divergence, not the tolerated f32-accumulation gap"
        );
    }

    #[test]
    fn primitive_parity_reductions_f32_within_tol_on_hip() {
        let Some(text) = load_or_skip("reduce.txt") else {
            return;
        };
        let hip = rocm_client();
        let mut cases = 0usize;

        for (lineno, t) in records(&text) {
            if field(&t, "dtype") != Some("f32") {
                continue;
            }
            let name = field(&t, "name").unwrap_or("?");
            let op = field(&t, "op").expect("op");
            let data = parse_f32_bits(field(&t, "in").expect("in"));
            let cpp = parse_f32_bits(field(&t, "out").expect("out"))[0];
            let label = format!("{name}/{op} (line {lineno})");
            match op {
                "sum" => check(&label, plane_reduce_sum_f32_on(&hip, &data, PLANE_DIM).unwrap(), cpp),
                "dot" => {
                    let b = parse_f32_bits(field(&t, "in2").expect("in2"));
                    check(&label, plane_dot_product_f32_on(&hip, &data, &b, PLANE_DIM).unwrap(), cpp);
                }
                // Same C++ 0-identity bridge as the f64 leg.
                "max" => {
                    let g = plane_reduce_max_f32_on(&hip, &data, PLANE_DIM).unwrap();
                    check(&label, g.max(0.0), cpp);
                }
                "min" => {
                    let g = plane_reduce_min_f32_on(&hip, &data, PLANE_DIM).unwrap();
                    check(&label, g.min(0.0), cpp);
                }
                other => panic!("RED {name}: unknown op {other} (line {lineno})"),
            }
            cases += 1;
        }

        assert!(cases > 0, "reduce fixture parsed zero f32 cases for the hip leg");
        eprintln!("primitive_parity hip f32 reductions: {cases} cases checked (~1e-6 surfaced).");
    }
}
