//! On-device best-split-finder golden-anchor parity replay (Phase 17, ODL-11/12).
//!
//! Mirrors `oracle-harness/tests/kernel_parity.rs`: parse the committed
//! `tests/fixtures/kernels/best_split.txt` golden (C++-core expected records as
//! f64 bit-hex), drive the cubecl-cpu stage-1 fold
//! ([`lgbm_compute::kernels::best_split::find_best_splits_stage1_on`]) over each
//! case, and assert the resulting `CUDASplitInfo` record is BIT-EXACT versus the
//! golden via `compare_exact_f64_bits`. Idioms follow `kernel_parity.rs`:
//! `CARGO_MANIFEST_DIR` fixture path (never the untracked `LightGBM/` tree),
//! graceful SKIP when the fixture is absent, and a localizing assert that names the
//! diverging case + field.
//!
//! ## Wave-0 status (17-01)
//! The stage-1 numerical core lands in **17-03** (Wave 2). Against the Wave-0 stub
//! ([`find_best_splits_stage1_on`] returns an `is_valid=false` sentinel) the primary
//! test RED-fails via ASSERTION on every valid golden — that is the intended
//! Nyquist scaffold. It is therefore marked `#[ignore = "Wave-0 scaffold; un-ignore
//! when 17-03 lands"]` so the merge-gate `cargo test --workspace` stays GREEN
//! (D-09 / ODL-19); 17-03 turns it green and removes the `#[ignore]`.
//!
//! ## Record format (see `tests/fixtures/kernels/best_split.txt`)
//! ```text
//! COUNTS best_split=<n>
//! SCASE name=<id> num_bin=.. offset=.. default_bin=.. skip_default_bin=0|1 \
//!       na_as_missing=0|1 use_l1=0|1 use_smoothing=0|1 use_rand=0|1 is_larger=0|1 \
//!       min_data_in_leaf=.. min_sum_hessian_in_leaf=<f64> lambda_l1=<f64> \
//!       lambda_l2=<f64> min_gain_to_split=<f64> path_smooth=<f64> parent_output=<f64> \
//!       sum_gradient=<f64> sum_hessian=<f64> num_data=.. parent_gain=<f64> rng_seed=..
//! SHIST <interleaved [g0;h0;g1;h1;...] f64-bits>
//! SWIN  is_valid=0|1 threshold=.. default_left=0|1 left_count=.. right_count=.. \
//!       left_sum_gradient=<f64> left_sum_hessian=<f64> right_sum_gradient=<f64> \
//!       right_sum_hessian=<f64> value=<f64> gain=<f64>
//! SEXPORT <8 ints>   # OPTIONAL — stage-3 cross-leaf export cases only
//! ```
//! (All floats are raw little-endian f64 bit patterns as decimal `u64`, the
//! `split.txt` convention — zero parse rounding.)

use std::path::PathBuf;

use lgbm_compute::kernels::best_split::{
    find_best_from_all_splits_on, find_best_splits_stage1_f32_on, find_best_splits_stage1_on,
    sync_best_split_for_leaf_on, SplitFindTask, Stage1Scalars,
};
use lgbm_compute::kernels::split_info::SplitScalars;
use lgbm_compute::runtime::cpu_client;
use lgbm_compute::CpuBackend;
use oracle_harness::comparator::compare_exact_f64_bits;

fn kernels_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/kernels")
}

/// Parse an `f64` from a raw little-endian f64 bit pattern (decimal `u64`).
fn parse_f64_bits(s: &str) -> f64 {
    f64::from_bits(s.parse::<u64>().expect("f64-bits u64 field"))
}

/// Extract a `key=value` token's value from a whitespace-split line.
fn field<'a>(tokens: &'a [&'a str], key: &str) -> Option<&'a str> {
    tokens
        .iter()
        .find_map(|t| t.strip_prefix(key).and_then(|r| r.strip_prefix('=')))
}

fn parse_i64(tokens: &[&str], key: &str) -> i64 {
    field(tokens, key)
        .unwrap_or_else(|| panic!("missing int field `{key}`"))
        .parse()
        .unwrap_or_else(|_| panic!("bad int field `{key}`"))
}

fn parse_u32(tokens: &[&str], key: &str) -> u32 {
    field(tokens, key)
        .unwrap_or_else(|| panic!("missing u32 field `{key}`"))
        .parse()
        .unwrap_or_else(|_| panic!("bad u32 field `{key}`"))
}

fn parse_f64_field(tokens: &[&str], key: &str) -> f64 {
    parse_f64_bits(field(tokens, key).unwrap_or_else(|| panic!("missing f64-bits field `{key}`")))
}

/// Parse a `;`-separated list of raw little-endian f64 bit patterns (decimal `u64`).
fn parse_f64_bits_list(s: &str) -> Vec<f64> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(';')
        .map(|t| f64::from_bits(t.parse::<u64>().expect("f64-bits u64 field")))
        .collect()
}

/// The parsed golden for one SCASE/SHIST/SWIN record.
#[derive(Debug)]
struct BestSplitGolden {
    name: String,
    // task scalars
    num_bin: u32,
    offset: u32,
    default_bin: u32,
    skip_default_bin: bool,
    na_as_missing: bool,
    use_l1: bool,
    use_smoothing: bool,
    use_rand: bool,
    is_larger: bool,
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    lambda_l1: f64,
    lambda_l2: f64,
    min_gain_to_split: f64,
    path_smooth: f64,
    parent_output: f64,
    sum_gradient: f64,
    sum_hessian: f64,
    num_data: i32,
    parent_gain: f64,
    rng_seed: i32,
    hist: Vec<f64>,
    // expected CUDASplitInfo record (SWIN)
    win_is_valid: bool,
    win_threshold: u32,
    win_default_left: bool,
    win_left_count: i32,
    win_right_count: i32,
    win_left_sum_gradient: f64,
    win_left_sum_hessian: f64,
    win_right_sum_gradient: f64,
    win_right_sum_hessian: f64,
    win_value: f64,
    win_gain: f64,
}

fn parse_best_split(text: &str) -> Vec<BestSplitGolden> {
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
        let num_bin = parse_u32(&t, "num_bin");
        let offset = parse_u32(&t, "offset");
        let default_bin = parse_u32(&t, "default_bin");
        let skip_default_bin = parse_i64(&t, "skip_default_bin") != 0;
        let na_as_missing = parse_i64(&t, "na_as_missing") != 0;
        let use_l1 = parse_i64(&t, "use_l1") != 0;
        let use_smoothing = parse_i64(&t, "use_smoothing") != 0;
        let use_rand = parse_i64(&t, "use_rand") != 0;
        let is_larger = parse_i64(&t, "is_larger") != 0;
        let min_data_in_leaf = parse_i64(&t, "min_data_in_leaf") as i32;
        let min_sum_hessian_in_leaf = parse_f64_field(&t, "min_sum_hessian_in_leaf");
        let lambda_l1 = parse_f64_field(&t, "lambda_l1");
        let lambda_l2 = parse_f64_field(&t, "lambda_l2");
        let min_gain_to_split = parse_f64_field(&t, "min_gain_to_split");
        let path_smooth = parse_f64_field(&t, "path_smooth");
        let parent_output = parse_f64_field(&t, "parent_output");
        let sum_gradient = parse_f64_field(&t, "sum_gradient");
        let sum_hessian = parse_f64_field(&t, "sum_hessian");
        let num_data = parse_i64(&t, "num_data") as i32;
        let parent_gain = parse_f64_field(&t, "parent_gain");
        let rng_seed = parse_i64(&t, "rng_seed") as i32;

        let ht: Vec<&str> = lines.next().expect("SHIST line").split_whitespace().collect();
        assert_eq!(ht[0], "SHIST", "expected SHIST for `{name}`");
        let hist = parse_f64_bits_list(ht.get(1).copied().unwrap_or(""));

        let wt: Vec<&str> = lines.next().expect("SWIN line").split_whitespace().collect();
        assert_eq!(wt[0], "SWIN", "expected SWIN for `{name}`");

        out.push(BestSplitGolden {
            name,
            num_bin,
            offset,
            default_bin,
            skip_default_bin,
            na_as_missing,
            use_l1,
            use_smoothing,
            use_rand,
            is_larger,
            min_data_in_leaf,
            min_sum_hessian_in_leaf,
            lambda_l1,
            lambda_l2,
            min_gain_to_split,
            path_smooth,
            parent_output,
            sum_gradient,
            sum_hessian,
            num_data,
            parent_gain,
            rng_seed,
            hist,
            win_is_valid: parse_i64(&wt, "is_valid") != 0,
            win_threshold: parse_u32(&wt, "threshold"),
            win_default_left: parse_i64(&wt, "default_left") != 0,
            win_left_count: parse_i64(&wt, "left_count") as i32,
            win_right_count: parse_i64(&wt, "right_count") as i32,
            win_left_sum_gradient: parse_f64_field(&wt, "left_sum_gradient"),
            win_left_sum_hessian: parse_f64_field(&wt, "left_sum_hessian"),
            win_right_sum_gradient: parse_f64_field(&wt, "right_sum_gradient"),
            win_right_sum_hessian: parse_f64_field(&wt, "right_sum_hessian"),
            win_value: parse_f64_field(&wt, "value"),
            win_gain: parse_f64_field(&wt, "gain"),
        });
    }
    out
}

/// Build the [`SplitFindTask`] the stage-1 fold consumes from a parsed golden.
/// `hist_offset` is 0 because SHIST already carries the feature's slice. `reverse`
/// / `assume_out_default_left` are derived from the golden's `default_left` (the
/// stage-1 kernel writes `default_left = assume_out_default_left`, 17-RESEARCH
/// Pitfall 3).
fn task_of(g: &BestSplitGolden) -> SplitFindTask {
    SplitFindTask {
        inner_feature_index: 0,
        reverse: g.win_default_left,
        skip_default_bin: g.skip_default_bin,
        na_as_missing: g.na_as_missing,
        assume_out_default_left: g.win_default_left,
        is_categorical: false,
        is_one_hot: false,
        hist_offset: 0,
        mfb_offset: g.offset as u8,
        num_bin: g.num_bin,
        default_bin: g.default_bin,
        rand_threshold: -1,
    }
}

fn scalars_of(g: &BestSplitGolden) -> Stage1Scalars {
    Stage1Scalars {
        use_l1: g.use_l1,
        use_smoothing: g.use_smoothing,
        use_rand: g.use_rand,
        is_larger: g.is_larger,
        min_data_in_leaf: g.min_data_in_leaf,
        min_sum_hessian_in_leaf: g.min_sum_hessian_in_leaf,
        lambda_l1: g.lambda_l1,
        lambda_l2: g.lambda_l2,
        min_gain_to_split: g.min_gain_to_split,
        path_smooth: g.path_smooth,
        parent_output: g.parent_output,
        sum_gradient: g.sum_gradient,
        sum_hessian: g.sum_hessian,
        num_data: g.num_data,
        parent_gain: g.parent_gain,
        rng_seed: g.rng_seed,
    }
}

/// Stage-1 per-task record bit-exact vs the CUDA-core f64 fold (ODL-11).
///
/// The 17-03 cpu f64 fold ([`find_best_splits_stage1_on`]) reproduces every
/// `CUDASplitInfo` field bit-exactly against the C++-transcribed `best_split.txt`
/// goldens (all six D-07 categories + the round-ties-even count case). Un-ignored in
/// 17-03 (Wave 2) now that the numerical core lands.
#[test]
fn best_split_parity_stage1_bit_exact_on_cpu() {
    let path = kernels_dir().join("best_split.txt");
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!(
            "best_split_parity(stage1): SKIP — fixture {} not found.",
            path.display()
        );
        return;
    };
    let cases = parse_best_split(&text);
    assert!(!cases.is_empty(), "best_split fixture present but parsed zero cases");

    let client = cpu_client();
    let backend = CpuBackend;
    let _ = backend; // CpuBackend selects the cubecl-cpu runtime; the stage fn is generic.

    // D-07 coverage: every flag/category the fixture matrix must exercise.
    let mut saw_default_template = false;
    let mut saw_use_l1 = false;
    let mut saw_smoothing = false;
    let mut saw_rand = false;
    let mut saw_empty = false;
    let mut saw_globalmem = false;
    let mut saw_default_left_tie = false;

    for g in &cases {
        if g.name.starts_with("default_") && !g.name.contains("tie") {
            saw_default_template = true;
        }
        if g.use_l1 {
            saw_use_l1 = true;
        }
        if g.use_smoothing {
            saw_smoothing = true;
        }
        if g.use_rand {
            saw_rand = true;
        }
        if !g.win_is_valid {
            saw_empty = true;
        }
        if g.num_bin > 256 {
            saw_globalmem = true;
        }
        if g.name.contains("tie") {
            saw_default_left_tie = true;
        }

        let task = task_of(g);
        let scalars = scalars_of(g);
        let si = find_best_splits_stage1_on(&client, &g.hist, &task, &scalars)
            .unwrap_or_else(|e| panic!("BEST_SPLIT `{}`: stage1 launch failed: {e:?}", g.name));

        // Structure: is_valid must match. (The empty case golden is is_valid=false,
        // so it matches the Wave-0 sentinel; the valid cases RED-fail here until
        // 17-03 fills the core — the intended scaffold.)
        assert_eq!(
            si.is_valid, g.win_is_valid,
            "BEST_SPLIT `{}`: is_valid mismatch (got {}, want {})",
            g.name, si.is_valid, g.win_is_valid
        );

        if g.win_is_valid {
            // Every f64 field bit-exact.
            let got = [
                si.left_sum_gradients,
                si.left_sum_hessians,
                si.right_sum_gradients,
                si.right_sum_hessians,
                si.left_value,
                si.gain,
            ];
            let exp = [
                g.win_left_sum_gradient,
                g.win_left_sum_hessian,
                g.win_right_sum_gradient,
                g.win_right_sum_hessian,
                g.win_value,
                g.win_gain,
            ];
            if let Err(m) = compare_exact_f64_bits(&got, &exp) {
                panic!("BEST_SPLIT `{}`: winner f64 field divergence: {m}", g.name);
            }
            assert_eq!(si.threshold, g.win_threshold, "BEST_SPLIT `{}`: threshold", g.name);
            assert_eq!(si.default_left, g.win_default_left, "BEST_SPLIT `{}`: default_left", g.name);
            assert_eq!(si.left_count, g.win_left_count, "BEST_SPLIT `{}`: left_count", g.name);
            assert_eq!(si.right_count, g.win_right_count, "BEST_SPLIT `{}`: right_count", g.name);
        }
    }

    // Every D-07 category must be present in the committed fixture.
    assert!(saw_default_template, "fixture must cover the default-template (fwd+rev, smaller+larger)");
    assert!(saw_use_l1, "fixture must cover USE_L1 (lambda_l1 > 0)");
    assert!(saw_smoothing, "fixture must cover USE_SMOOTHING (path_smooth > 0)");
    assert!(saw_rand, "fixture must cover USE_RAND (extra-trees)");
    assert!(saw_empty, "fixture must cover the empty/no-valid-split (is_valid=0) case");
    assert!(saw_globalmem, "fixture must cover the global-memory spill (num_bin > 256) case");
    assert!(saw_default_left_tie, "fixture must cover a tie-aware default_left case");
}

/// STAGE-2 cross-feature reduce (ODL-12, §8.2): drive stage-1 over several distinct
/// features (distinct `inner_feature_index`) to build the per-task record slab, then
/// assert [`sync_best_split_for_leaf_on`] picks the max-gain feature per leaf with the
/// strict-`>` lowest-index tie-break — bit-exact on the winning fields — and marks a
/// no-valid-split leaf `is_valid=false`, with NO device→host readback in stage-2 (SC#2).
#[test]
fn best_split_parity_stage2_cross_feature_reduce() {
    let path = kernels_dir().join("best_split.txt");
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!("best_split_parity(stage2): SKIP — fixture {} not found.", path.display());
        return;
    };
    let cases = parse_best_split(&text);
    let client = cpu_client();

    // Pick two distinct continuous features and run stage-1 to get real records.
    let g_default = cases
        .iter()
        .find(|g| g.name == "default_fwd_smaller")
        .expect("default_fwd_smaller golden");
    let g_l1 = cases.iter().find(|g| g.name == "use_l1").expect("use_l1 golden");

    // Two smaller-leaf tasks (feat 0 = default, feat 1 = l1) + two larger-leaf tasks.
    let run = |g: &BestSplitGolden, feat: i32| {
        let mut task = task_of(g);
        task.inner_feature_index = feat;
        find_best_splits_stage1_on(&client, &g.hist, &task, &scalars_of(g))
            .unwrap_or_else(|e| panic!("stage1 for `{}`: {e:?}", g.name))
    };
    let s0 = run(g_default, 0); // higher gain
    let s1 = run(g_l1, 1); // lower gain (L1 shrinks the gain)
    assert!(s0.gain > s1.gain, "default gain must exceed the L1-shrunk gain for this fixture");

    // Full 2·num_tasks slab: smaller half [s0, s1], larger half [s1, s0].
    let per_task = vec![s0, s1, s1, s0];

    let smaller = sync_best_split_for_leaf_on(&client, &per_task, 2, true).unwrap();
    assert!(smaller.is_valid, "stage2 smaller: a valid winner");
    assert_eq!(smaller.inner_feature_index, 0, "stage2 smaller picks the max-gain feature (0)");
    if let Err(m) = compare_exact_f64_bits(&[smaller.gain], &[s0.gain]) {
        panic!("stage2 smaller winner gain divergence: {m}");
    }
    assert_eq!(smaller.threshold, s0.threshold, "stage2 winner threshold copied verbatim");

    let larger = sync_best_split_for_leaf_on(&client, &per_task, 2, false).unwrap();
    assert!(larger.is_valid, "stage2 larger: a valid winner");
    assert_eq!(larger.inner_feature_index, 0, "stage2 larger reads [num_tasks,2·num_tasks) and picks feat 0");

    // A leaf whose tasks are all invalid yields the no-valid-split sentinel.
    let mut invalid0 = s0;
    let mut invalid1 = s1;
    invalid0.is_valid = false;
    invalid1.is_valid = false;
    let empty_slab = vec![invalid0, invalid1];
    let none = sync_best_split_for_leaf_on(&client, &empty_slab, 2, true).unwrap();
    assert!(!none.is_valid, "stage2: all-invalid tasks → is_valid=false");
    assert_eq!(none.gain, f64::NEG_INFINITY, "stage2: no-valid-split gain = kMinScore");
}

/// STAGE-3 cross-leaf argmax + 8-int export (ODL-12, §8.3): drive the full stage-1→2→3
/// pipeline on the cpu f64 fold, then assert the 8-int buffer field layout (idx 0-7), the
/// best-leaf argmax, the self-invalidation (chosen leaf + freshly-created slot), the
/// single-readback contract (SC#2), and the `best_leaf_index == -1` no-split export.
#[test]
fn best_split_parity_stage3_export() {
    let path = kernels_dir().join("best_split.txt");
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!("best_split_parity(stage3_export): SKIP — fixture {} not found.", path.display());
        return;
    };
    let cases = parse_best_split(&text);
    let client = cpu_client();

    let g_default = cases
        .iter()
        .find(|g| g.name == "default_fwd_smaller")
        .expect("default_fwd_smaller golden");
    let g_l1 = cases.iter().find(|g| g.name == "use_l1").expect("use_l1 golden");

    let run = |g: &BestSplitGolden, feat: i32| {
        let mut task = task_of(g);
        task.inner_feature_index = feat;
        find_best_splits_stage1_on(&client, &g.hist, &task, &scalars_of(g))
            .unwrap_or_else(|e| panic!("stage1 for `{}`: {e:?}", g.name))
    };

    // Two leaves' best splits from stage-2 (smaller = leaf 0 via feat 5, larger = leaf 1
    // via feat 8). Leaf 0's default template has the higher gain than leaf 1's L1-shrunk.
    let s0 = run(g_default, 5);
    let s1 = run(g_l1, 8);
    let leaf0 = sync_best_split_for_leaf_on(&client, &[s0, s0], 1, true).unwrap();
    let leaf1 = sync_best_split_for_leaf_on(&client, &[s1, s1], 1, true).unwrap();
    assert!(leaf0.gain > leaf1.gain, "leaf 0 (default) must out-gain leaf 1 (L1)");

    // per_leaf: [leaf0, leaf1, freshly-created slot]. cur_num_leaves = 2.
    let mut per_leaf = vec![leaf0, leaf1, SplitScalars::default()];
    let out = find_best_from_all_splits_on(&client, &mut per_leaf, 0, 1, 2).unwrap();

    // Field layout idx 0-7.
    assert_eq!(out[0] as i32, leaf0.inner_feature_index, "[0] smaller.inner_feature_index");
    assert_eq!(out[1] as u32, leaf0.threshold, "[1] smaller.threshold");
    assert_eq!(out[2], i64::from(leaf0.default_left), "[2] smaller.default_left");
    assert_eq!(out[3] as i32, leaf1.inner_feature_index, "[3] larger.inner_feature_index");
    assert_eq!(out[4] as u32, leaf1.threshold, "[4] larger.threshold");
    assert_eq!(out[5], i64::from(leaf1.default_left), "[5] larger.default_left");
    assert_eq!(out[6], 0, "[6] best_leaf_index = the higher-gain leaf 0");
    assert_eq!(out[7], 0, "[7] num_cat_threshold = 0 (continuous)");

    // Self-invalidation: leaf 0 (chosen) + slot 2 (freshly-created) now invalid.
    assert!(!per_leaf[0].is_valid, "chosen leaf 0 self-invalidated");
    assert!(!per_leaf[2].is_valid, "freshly-created slot 2 self-invalidated");
    assert!(per_leaf[1].is_valid, "leaf 1 untouched — re-pickable next iteration");

    // Two-iteration: leaf 1 is now the winner (leaf 0 was invalidated).
    let out2 = find_best_from_all_splits_on(&client, &mut per_leaf, 0, 1, 2).unwrap();
    assert_eq!(out2[6], 1, "iteration 2 picks leaf 1 (leaf 0 invalidated)");

    // No-valid-split export: all leaves invalid → best_leaf_index = -1.
    let mut none = vec![leaf0, leaf1];
    none[0].is_valid = false;
    none[1].is_valid = false;
    let o = find_best_from_all_splits_on(&client, &mut none, 0, 1, 2).unwrap();
    assert_eq!(o[6], -1, "empty-leaf export → best_leaf_index = -1");
}

/// The f32 stage-1 mirror drives through cubecl-cpu producing a STRUCTURALLY identical
/// record (is_valid / threshold / default_left / left+right count bit-exact) with the
/// continuous per-side sums / value / gain within the ~1e-5 f32 envelope of the f64
/// fold (17-03 Task 2, def-f8u-01 — anchored to the cpu f64 fold, NEVER GPU-vs-GPU).
/// Exercised on the default-template + USE_L1 goldens (small exact-ish values).
#[test]
fn best_split_parity_stage1_f32_within_tol_on_cpu() {
    let path = kernels_dir().join("best_split.txt");
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!("best_split_parity(f32): SKIP — fixture {} not found.", path.display());
        return;
    };
    let cases = parse_best_split(&text);
    let client = cpu_client();
    let tol = 1e-5_f64;
    let mut checked = 0;
    for g in &cases {
        // Default-template + USE_L1 only (the plan's f32 anchor scope; small values).
        if !(g.name.starts_with("default_") && !g.name.contains("tie") || g.name == "use_l1") {
            continue;
        }
        let task = task_of(g);
        let scalars = scalars_of(g);
        let hist_f32: Vec<f32> = g.hist.iter().map(|&x| x as f32).collect();
        let si = find_best_splits_stage1_f32_on(&client, &hist_f32, &task, &scalars)
            .unwrap_or_else(|e| panic!("BEST_SPLIT f32 `{}`: launch failed: {e:?}", g.name));
        // Structure must be bit-exact.
        assert_eq!(si.is_valid, g.win_is_valid, "f32 `{}`: is_valid", g.name);
        if !g.win_is_valid {
            continue;
        }
        assert_eq!(si.threshold, g.win_threshold, "f32 `{}`: threshold", g.name);
        assert_eq!(si.default_left, g.win_default_left, "f32 `{}`: default_left", g.name);
        assert_eq!(si.left_count, g.win_left_count, "f32 `{}`: left_count", g.name);
        assert_eq!(si.right_count, g.win_right_count, "f32 `{}`: right_count", g.name);
        // Continuous fields within ~1e-5 of the f64 anchor golden.
        let pairs = [
            (si.left_sum_gradients, g.win_left_sum_gradient, "left_sum_gradient"),
            (si.left_sum_hessians, g.win_left_sum_hessian, "left_sum_hessian"),
            (si.right_sum_gradients, g.win_right_sum_gradient, "right_sum_gradient"),
            (si.right_sum_hessians, g.win_right_sum_hessian, "right_sum_hessian"),
            (si.left_value, g.win_value, "value"),
            (si.gain, g.win_gain, "gain"),
        ];
        for (got, exp, field) in pairs {
            assert!(
                (got - exp).abs() <= tol,
                "f32 `{}`: {field} {got} vs {exp} exceeds {tol}",
                g.name
            );
        }
        checked += 1;
    }
    assert!(checked >= 2, "f32 mirror must exercise default-template + USE_L1 (got {checked})");
}
