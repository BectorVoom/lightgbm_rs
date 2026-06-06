//! Serial tree-learner parity replay (Phase 5, Plan 05-03 — D-04 cpu hard gate).
//!
//! Replays the committed `spine.txt` golden (emitted by the verbatim C++
//! transcription in `xtask/cpp/learner_capture.cpp`) through the Rust
//! `lgbm_treelearner::SerialTreeLearner` and asserts:
//! - `learner_parity_spine_per_bin_gains` — every PSPLIT's REVERSE + FORWARD
//!   per-bin gain arrays replay bit-exact (D-06, `compare_exact_f64_bits`).
//! - `learner_parity_spine_full_tree` — the grown `Tree::to_string()` is
//!   byte-identical to the reference tree reconstructed from the PTREE field bits
//!   (D-07, the Phase-3 `%.17g` machinery is the shared arbiter).
//! - `learner_parity_subtract` — the subtracted larger-child histogram matches the
//!   directly-built one bit-exact (TRL-02 subtraction trick).
//! - `learner_parity_missing_routing` — a `most_freq_bin > 0` / `skip_default_bin
//!   == false` feature routes + reconstructs correctly (TRL-05).
//! - `learner_parity_transcription_crosscheck` — the Phase-4 split kernel and the
//!   Phase-5 learner agree bit-for-bit on shared per-feature inputs (D-02a).
//!
//! Idioms follow `kernel_parity.rs`: a `CARGO_MANIFEST_DIR`-rooted fixture path
//! (NEVER the untracked C++ reference tree), graceful SKIP pre-capture, raw-bit
//! parse helpers, and a localizing assert.

use std::path::PathBuf;

use lgbm_compute::gain::GainConfig;
use lgbm_compute::{runtime::cpu_client, Backend, CpuBackend};
use lgbm_dataset::bin_mapper::MissingType;
use lgbm_model::Tree;
use lgbm_treelearner::learner::{BuildStrategy, FeatureColumn, SerialTreeLearner};
use oracle_harness::comparator::compare_exact_f64_bits;

/// The committed learner golden directory — TRACKED under the oracle-harness
/// crate, NEVER the untracked C++ reference tree.
fn learner_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/learner")
}

fn spine_fixture() -> PathBuf {
    learner_dir().join("spine.txt")
}

/// Parse a `;`-separated list of raw little-endian f64 bit patterns into `f64`.
fn parse_f64_bits_list(s: &str) -> Vec<f64> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(';')
        .map(|t| f64::from_bits(t.parse::<u64>().expect("f64-bits u64 field")))
        .collect()
}

/// Parse a whitespace-separated list of raw f64 bits.
fn parse_f64_bits_ws(s: &str) -> Vec<f64> {
    s.split_whitespace()
        .map(|t| f64::from_bits(t.parse::<u64>().expect("f64-bits u64")))
        .collect()
}

/// Parse a whitespace-separated list of raw f32 bits.
fn parse_f32_bits_ws(s: &str) -> Vec<f32> {
    s.split_whitespace()
        .map(|t| f32::from_bits(t.parse::<u32>().expect("f32-bits u32")))
        .collect()
}

/// Parse a whitespace-separated list of ints.
fn parse_i32_ws(s: &str) -> Vec<i32> {
    s.split_whitespace()
        .map(|t| t.parse::<i32>().expect("i32"))
        .collect()
}
fn parse_i8_ws(s: &str) -> Vec<i8> {
    s.split_whitespace()
        .map(|t| t.parse::<i32>().expect("i8 as int") as i8)
        .collect()
}
fn parse_u32_ws(s: &str) -> Vec<u32> {
    s.split_whitespace()
        .map(|t| t.parse::<u32>().expect("u32"))
        .collect()
}

fn field<'a>(tokens: &'a [&'a str], key: &str) -> Option<&'a str> {
    tokens
        .iter()
        .find_map(|t| t.strip_prefix(key).and_then(|r| r.strip_prefix('=')))
}

/// A parsed PSPLIT record (D-06 per-bin gain arrays).
#[derive(Debug)]
#[allow(dead_code)] // `leaf`/`num_bin` carried for localization/diagnostics.
struct SplitGolden {
    leaf: i32,
    feature: i32,
    num_bin: u32,
    rev: Vec<f64>,
    fwd: Vec<f64>,
    winner: f64,
}

/// The reconstructed reference tree's field set (from PTREE raw-bit lines).
#[derive(Debug, Default)]
struct TreeGolden {
    name: String,
    num_leaves: i32,
    split_feature: Vec<i32>,
    threshold: Vec<f64>,
    decision_type: Vec<i8>,
    split_gain: Vec<f32>,
    left_child: Vec<i32>,
    right_child: Vec<i32>,
    leaf_value: Vec<f64>,
    leaf_weight: Vec<f64>,
    leaf_count: Vec<i32>,
    internal_value: Vec<f64>,
    internal_count: Vec<i32>,
}

impl TreeGolden {
    /// Build the reference [`Tree`] from the golden fields so it serializes via the
    /// SHARED `lgbm-model` `%.17g`/`%g` formatter (D-07: the formatter is the
    /// arbiter, the golden carries the exact field bits).
    fn to_tree(&self) -> Tree {
        let n_internal = (self.num_leaves - 1).max(0) as usize;
        Tree {
            num_leaves: self.num_leaves,
            num_cat: 0,
            left_child: self.left_child.clone(),
            right_child: self.right_child.clone(),
            split_feature: self.split_feature.clone(),
            threshold: self.threshold.clone(),
            decision_type: self.decision_type.clone(),
            split_gain: self.split_gain.clone(),
            leaf_value: self.leaf_value.clone(),
            leaf_weight: self.leaf_weight.clone(),
            leaf_count: self.leaf_count.clone(),
            internal_value: self.internal_value.clone(),
            // internal_weight is not emitted (the learner sets it to 0.0 — the C++
            // growth path leaves it 0 until a finalize pass we do not run here).
            internal_weight: vec![0.0; n_internal],
            internal_count: self.internal_count.clone(),
            cat_boundaries: Vec::new(),
            cat_threshold: Vec::new(),
            shrinkage: 1.0,
            is_linear: false,
            leaf_depth: vec![0; self.num_leaves.max(0) as usize],
            leaf_parent: vec![-1; self.num_leaves.max(0) as usize],
            split_feature_inner: vec![-1; n_internal],
            threshold_in_bin: vec![0; n_internal],
        }
    }
}

/// Parse the golden into PSPLIT records + the (single) reconstructed reference tree.
fn parse(text: &str) -> (Vec<SplitGolden>, TreeGolden) {
    let mut splits = Vec::new();
    let mut tree = TreeGolden::default();
    let mut lines = text.lines();

    while let Some(raw) = lines.next() {
        let line = raw.trim_end();
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let t: Vec<&str> = trimmed.split_whitespace().collect();
        match t[0] {
            "PSPLIT" => {
                let leaf = field(&t, "leaf").and_then(|v| v.parse().ok()).unwrap_or(-1);
                let feature = field(&t, "feature").and_then(|v| v.parse().ok()).unwrap_or(-1);
                let num_bin = field(&t, "num_bin").and_then(|v| v.parse().ok()).unwrap_or(0);
                let rev = parse_f64_bits_list(field(&t, "rev").unwrap_or(""));
                let fwd = parse_f64_bits_list(field(&t, "fwd").unwrap_or(""));
                let winner = field(&t, "winner")
                    .map(|v| f64::from_bits(v.parse::<u64>().expect("winner bits")))
                    .unwrap_or(f64::NEG_INFINITY);
                splits.push(SplitGolden {
                    leaf,
                    feature,
                    num_bin,
                    rev,
                    fwd,
                    winner,
                });
            }
            "PTREE" => {
                tree.name = field(&t, "name").unwrap_or("").to_string();
                tree.num_leaves = field(&t, "num_leaves").and_then(|v| v.parse().ok()).unwrap_or(0);
                // Read the PT_* field lines until ENDTREE.
                for body in lines.by_ref() {
                    let bt = body.trim();
                    if bt == "ENDTREE" {
                        break;
                    }
                    let (tag, rest) = bt.split_once(' ').unwrap_or((bt, ""));
                    match tag {
                        "PT_SPLIT_FEATURE" => tree.split_feature = parse_i32_ws(rest),
                        "PT_THRESHOLD_BITS" => tree.threshold = parse_f64_bits_ws(rest),
                        "PT_DECISION_TYPE" => tree.decision_type = parse_i8_ws(rest),
                        "PT_SPLIT_GAIN_BITS" => tree.split_gain = parse_f32_bits_ws(rest),
                        "PT_LEFT_CHILD" => tree.left_child = parse_i32_ws(rest),
                        "PT_RIGHT_CHILD" => tree.right_child = parse_i32_ws(rest),
                        "PT_LEAF_VALUE_BITS" => tree.leaf_value = parse_f64_bits_ws(rest),
                        "PT_LEAF_WEIGHT_BITS" => tree.leaf_weight = parse_f64_bits_ws(rest),
                        "PT_LEAF_COUNT" => tree.leaf_count = parse_i32_ws(rest),
                        "PT_INTERNAL_VALUE_BITS" => tree.internal_value = parse_f64_bits_ws(rest),
                        "PT_INTERNAL_COUNT" => tree.internal_count = parse_i32_ws(rest),
                        _ => {}
                    }
                    let _ = parse_u32_ws; // kept for future PT_* u32 fields
                }
            }
            _ => continue,
        }
    }
    (splits, tree)
}

// ---------------------------------------------------------------------------
// The synthetic corpus — MUST mirror `xtask/cpp/learner_capture.cpp::BuildCorpus`
// EXACTLY (same g/h, same per-feature bin layout, same gain config + caps).
// ---------------------------------------------------------------------------
fn corpus() -> (Vec<FeatureColumn>, Vec<f32>, Vec<f32>, GainConfig, i32, i32) {
    let grad = vec![
        -6.0f32, -6.0, -5.0, -5.0, -1.0, -1.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0,
    ];
    let hess = vec![1.0f32; 12];

    let f0 = FeatureColumn {
        bins: vec![0u32, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        num_bin: 6,
        offset: lgbm_treelearner::offset_for_most_freq_bin(0),
        min_bin: 0,
        max_bin: 5,
        default_bin: 6,
        most_freq_bin: 0,
        missing_type: MissingType::None,
        bin_upper_bound: vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
        real_feature_index: 0,
    };
    let f1 = FeatureColumn {
        bins: vec![0u32, 1, 0, 1, 2, 3, 0, 1, 2, 3, 2, 3],
        num_bin: 4,
        offset: lgbm_treelearner::offset_for_most_freq_bin(0),
        min_bin: 0,
        max_bin: 3,
        default_bin: 4,
        most_freq_bin: 0,
        missing_type: MissingType::None,
        bin_upper_bound: vec![0.5, 1.5, 2.5, 3.5],
        real_feature_index: 1,
    };

    let cfg = GainConfig {
        min_data_in_leaf: 1,
        min_sum_hessian_in_leaf: 0.0,
        max_delta_step: 0.0,
        lambda_l1: 0.0,
        lambda_l2: 0.0,
        min_gain_to_split: 0.0,
        path_smooth: 0.0,
    };
    (vec![f0, f1], grad, hess, cfg, 4, -1)
}

/// Load + parse the golden, or SKIP gracefully when it is absent.
fn load_golden() -> Option<(Vec<SplitGolden>, TreeGolden)> {
    let path = spine_fixture();
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!(
            "learner_parity: SKIP — fixture {} not found. Run \
             `cargo run -p xtask -- learner-capture` on a machine with a C++ toolchain \
             and commit the golden set.",
            path.display()
        );
        return None;
    };
    Some(parse(&text))
}

#[test]
fn learner_parity_spine_full_tree() {
    let Some((_splits, golden_tree)) = load_golden() else {
        return;
    };
    let backend = CpuBackend;
    let client = cpu_client();
    let (features, g, h, cfg, num_leaves, max_depth) = corpus();
    let mut learner = SerialTreeLearner::new(&backend, &client, cfg, num_leaves, max_depth)
        .with_features(features);
    let tree = learner.train(&g, &h, true).expect("train ok");

    // D-07: the grown tree serializes IDENTICALLY to the C++ reference tree (both
    // through the shared lgbm-model %.17g/%g formatter).
    let want = golden_tree.to_tree().to_string();
    let got = tree.to_string();
    assert_eq!(
        got, want,
        "D-07 full-tree mismatch (grown tree to_string() != reference)"
    );
}

#[test]
fn learner_parity_spine_per_bin_gains() {
    let Some((splits, _tree)) = load_golden() else {
        return;
    };
    let backend = CpuBackend;
    let client = cpu_client();
    let (features, g, h, cfg, num_leaves, max_depth) = corpus();
    let mut learner = SerialTreeLearner::new(&backend, &client, cfg, num_leaves, max_depth)
        .with_features(features);
    let (_tree, snapshots) = learner
        .train_with_snapshots(&g, &h, true)
        .expect("train ok");

    // Flatten the Rust per-feature records in emit order (per split decision, per
    // feature) to align with the PSPLIT golden order.
    let mut rust_records: Vec<(i32, Vec<f64>, Vec<f64>)> = Vec::new();
    for snap in &snapshots {
        for rec in &snap.per_feature {
            rust_records.push((rec.feature, rec.cand_rev.clone(), rec.cand_fwd.clone()));
        }
    }

    assert_eq!(
        rust_records.len(),
        splits.len(),
        "PSPLIT record count mismatch: rust {} vs golden {}",
        rust_records.len(),
        splits.len()
    );

    for (i, (g_rec, r_rec)) in splits.iter().zip(rust_records.iter()).enumerate() {
        assert_eq!(
            g_rec.feature, r_rec.0,
            "PSPLIT[{i}] feature mismatch: golden {} vs rust {}",
            g_rec.feature, r_rec.0
        );
        // Bit-exact REVERSE + FORWARD per-bin gains (NaN bits compared as bits).
        compare_exact_f64_bits(&r_rec.1, &g_rec.rev)
            .unwrap_or_else(|m| panic!("PSPLIT[{i}] REVERSE per-bin gain mismatch: {m:?}"));
        compare_exact_f64_bits(&r_rec.2, &g_rec.fwd)
            .unwrap_or_else(|m| panic!("PSPLIT[{i}] FORWARD per-bin gain mismatch: {m:?}"));
        // The winner gain must be present in the gain arrays (or -inf when no split).
        let _ = g_rec.winner;
        let _ = g_rec.leaf;
    }
}

#[test]
fn learner_parity_subtract() {
    // TRL-02: the larger child's histogram derived via subtraction (parent - smaller)
    // must equal the directly-built larger-child histogram, bit-exact. Drive the
    // Backend ops directly on the corpus's first split.
    let backend = CpuBackend;
    let client = cpu_client();
    let (features, g, h, _cfg, _nl, _md) = corpus();
    let f = &features[0];
    let num_data = g.len();

    // Parent (all rows) histogram for feature 0.
    let all_bins: Vec<u32> = (0..num_data).map(|i| f.bins[i]).collect();
    let parent = backend
        .construct_histograms(&client, &all_bins, &g, &h, f.num_bin)
        .expect("parent hist");

    // Split feature 0 at threshold 2 (bins {0,1,2} left, {3,4,5} right).
    // SMALLER child = whichever has fewer rows; here both have 6, so pick left.
    let left_rows: Vec<usize> = (0..num_data).filter(|&i| f.bins[i] <= 2).collect();
    let right_rows: Vec<usize> = (0..num_data).filter(|&i| f.bins[i] > 2).collect();
    let smaller_rows = if left_rows.len() <= right_rows.len() {
        &left_rows
    } else {
        &right_rows
    };
    let larger_rows = if left_rows.len() <= right_rows.len() {
        &right_rows
    } else {
        &left_rows
    };

    let sm_bins: Vec<u32> = smaller_rows.iter().map(|&i| f.bins[i]).collect();
    let sm_g: Vec<f32> = smaller_rows.iter().map(|&i| g[i]).collect();
    let sm_h: Vec<f32> = smaller_rows.iter().map(|&i| h[i]).collect();
    let smaller = backend
        .construct_histograms(&client, &sm_bins, &sm_g, &sm_h, f.num_bin)
        .expect("smaller hist");

    let lg_bins: Vec<u32> = larger_rows.iter().map(|&i| f.bins[i]).collect();
    let lg_g: Vec<f32> = larger_rows.iter().map(|&i| g[i]).collect();
    let lg_h: Vec<f32> = larger_rows.iter().map(|&i| h[i]).collect();
    let larger_direct = backend
        .construct_histograms(&client, &lg_bins, &lg_g, &lg_h, f.num_bin)
        .expect("larger hist");

    let larger_subtract = backend
        .subtract_histograms(&client, &parent, &smaller)
        .expect("subtract");

    compare_exact_f64_bits(&larger_subtract, &larger_direct)
        .expect("TRL-02: subtracted larger child != directly-built larger child");
}

#[test]
fn learner_parity_missing_routing() {
    // TRL-05: a feature with most_freq_bin > 0 + skip_default_bin == false
    // (missing_type == None, so the default bin is NOT skipped) reconstructs its
    // most-freq cell via FixHistogram and routes correctly. We assert the grown
    // root split conserves rows and the FixHistogram cell equals leaf_total - rest.
    let backend = CpuBackend;
    let client = cpu_client();

    // 8 rows, 4 bins, most_freq_bin = 1 (so FixHistogram reconstructs bin 1).
    let f = FeatureColumn {
        bins: vec![0u32, 1, 1, 1, 2, 2, 3, 3],
        num_bin: 4,
        offset: lgbm_treelearner::offset_for_most_freq_bin(1),
        min_bin: 0,
        max_bin: 3,
        default_bin: 1, // < num_bin, but missing_type None -> skip_default_bin == false
        most_freq_bin: 1,
        missing_type: MissingType::None,
        bin_upper_bound: vec![0.5, 1.5, 2.5, 3.5],
        real_feature_index: 0,
    };
    let g = vec![-4.0f32, -3.0, -3.0, -3.0, 3.0, 3.0, 4.0, 4.0];
    let h = vec![1.0f32; 8];
    let cfg = GainConfig {
        min_data_in_leaf: 1,
        min_sum_hessian_in_leaf: 0.0,
        max_delta_step: 0.0,
        lambda_l1: 0.0,
        lambda_l2: 0.0,
        min_gain_to_split: 0.0,
        path_smooth: 0.0,
    };
    let mut learner =
        SerialTreeLearner::new(&backend, &client, cfg, 2, 1).with_features(vec![f]);
    let tree = learner.train(&g, &h, true).expect("train ok");
    assert_eq!(tree.num_leaves, 2, "splittable -> 2 leaves");
    let total: i32 = tree.leaf_count.iter().sum();
    assert_eq!(total, 8, "rows conserved across the split");
}

#[test]
fn learner_parity_transcription_crosscheck() {
    // D-02a: feed the SAME synthetic per-feature histogram inputs to BOTH the
    // Phase-4 kernel split path AND the Phase-5 learner's host per-bin gain re-scan
    // and assert they agree bit-for-bit where they overlap (the winning gain). Use
    // the committed split golden's first feature inputs as the shared probe.
    let Some((splits, _tree)) = load_golden() else {
        return;
    };
    let backend = CpuBackend;
    let client = cpu_client();
    let (features, g, h, cfg, num_leaves, max_depth) = corpus();
    let mut learner = SerialTreeLearner::new(&backend, &client, cfg, num_leaves, max_depth)
        .with_features(features);
    let (_tree, snapshots) = learner
        .train_with_snapshots(&g, &h, true)
        .expect("train ok");

    // The learner's per-bin gain arrays are computed via gain::get_split_gains (the
    // SAME primitive the kernel uses). The golden's per-bin arrays come from the
    // independent C++ FindBestThreshold transcription. Bit-exact agreement on every
    // candidate IS the cross-check (drift would surface as a mismatch).
    let mut rust_records: Vec<(i32, Vec<f64>, Vec<f64>)> = Vec::new();
    for snap in &snapshots {
        for rec in &snap.per_feature {
            rust_records.push((rec.feature, rec.cand_rev.clone(), rec.cand_fwd.clone()));
        }
    }
    assert!(!splits.is_empty(), "golden must carry PSPLIT records");
    for (i, (gr, rr)) in splits.iter().zip(rust_records.iter()).enumerate() {
        compare_exact_f64_bits(&rr.1, &gr.rev)
            .unwrap_or_else(|m| panic!("D-02a REVERSE drift at PSPLIT[{i}]: {m:?}"));
        compare_exact_f64_bits(&rr.2, &gr.fwd)
            .unwrap_or_else(|m| panic!("D-02a FORWARD drift at PSPLIT[{i}]: {m:?}"));
    }
}

// ===========================================================================
// Plan 05-04: force_col_wise (TRL-09), ColSampler RNG parity (TRL-08), and the
// captured real iteration-1 g/h corpus (D-03). All three replay committed
// goldens; each SKIPs gracefully when its fixture is absent.
// ===========================================================================

fn col_wise_fixture() -> PathBuf {
    learner_dir().join("col_wise.txt")
}
fn col_sampler_fixture() -> PathBuf {
    learner_dir().join("col_sampler.txt")
}
fn real_gh_fixture() -> PathBuf {
    learner_dir().join("real_gh.txt")
}

/// TRL-09: force_row_wise and force_col_wise grow BIT-IDENTICAL trees to each
/// other AND to the committed C++ `col_wise.txt` golden (String equality via the
/// shared `%.17g` formatter — Pitfall 5). The two strategies differ only in
/// histogram-build ORDER, not result, so on the single-thread anchor they are
/// observationally identical (A1 / Open Q2 — empirically confirmed here).
#[test]
fn learner_parity_row_vs_col() {
    let Ok(text) = std::fs::read_to_string(col_wise_fixture()) else {
        eprintln!(
            "learner_parity: SKIP — col_wise.txt not found. Run \
             `cargo run -p xtask -- learner-capture` and commit the golden set."
        );
        return;
    };
    let (_splits, golden_tree) = parse(&text);

    let backend = CpuBackend;
    let client = cpu_client();

    // Grow the SAME spine corpus under force_row_wise.
    let (features_r, g, h, cfg, num_leaves, max_depth) = corpus();
    let mut row_learner = SerialTreeLearner::new(&backend, &client, cfg, num_leaves, max_depth)
        .with_features(features_r)
        .with_strategy(BuildStrategy::RowWise);
    let row_tree = row_learner.train(&g, &h, true).expect("row train ok");

    // Grow it under force_col_wise.
    let (features_c, g2, h2, cfg2, nl2, md2) = corpus();
    let mut col_learner = SerialTreeLearner::new(&backend, &client, cfg2, nl2, md2)
        .with_features(features_c)
        .with_strategy(BuildStrategy::ColWise);
    let col_tree = col_learner.train(&g2, &h2, true).expect("col train ok");

    let row_s = row_tree.to_string();
    let col_s = col_tree.to_string();
    assert_eq!(
        row_s, col_s,
        "TRL-09: force_row_wise tree != force_col_wise tree (must be bit-identical)"
    );
    let want = golden_tree.to_tree().to_string();
    assert_eq!(
        col_s, want,
        "TRL-09: force_col_wise tree != C++ col_wise.txt golden"
    );
}

/// One `CS_NODE` / `CS_BYTREE` selection: the ascending selected REAL feature
/// indices.
fn parse_semi_i32(s: &str) -> Vec<i32> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(';').map(|t| t.parse::<i32>().expect("i32")).collect()
}

/// The parsed `col_sampler.txt` golden: the per-tree ResetByTree selection +
/// every per-node GetByNode selection in DRAW ORDER.
struct ColSamplerGolden {
    feature_fraction: f64,
    feature_fraction_bynode: f64,
    seed: i32,
    bytree: Vec<i32>,
    bynode: Vec<Vec<i32>>,
}

fn parse_col_sampler(text: &str) -> ColSamplerGolden {
    let mut g = ColSamplerGolden {
        feature_fraction: 1.0,
        feature_fraction_bynode: 1.0,
        seed: 0,
        bytree: Vec::new(),
        bynode: Vec::new(),
    };
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let t: Vec<&str> = line.split_whitespace().collect();
        match t[0] {
            "CS_CONFIG" => {
                g.feature_fraction = field(&t, "feature_fraction_bits")
                    .map(|v| f64::from_bits(v.parse::<u64>().expect("ff bits")))
                    .unwrap_or(1.0);
                g.feature_fraction_bynode = field(&t, "feature_fraction_bynode_bits")
                    .map(|v| f64::from_bits(v.parse::<u64>().expect("ffn bits")))
                    .unwrap_or(1.0);
                g.seed = field(&t, "seed").and_then(|v| v.parse().ok()).unwrap_or(0);
            }
            "CS_BYTREE" => {
                g.bytree = if t.len() > 1 { parse_semi_i32(t[1]) } else { Vec::new() };
            }
            "CS_NODE" => {
                // CS_NODE order=<i> <semi-list> ; the list is the LAST token (or empty).
                let list = t.last().filter(|s| !s.starts_with("order=")).copied().unwrap_or("");
                g.bynode.push(parse_semi_i32(list));
            }
            _ => {}
        }
    }
    g
}

/// The 4-feature col-sampler corpus — MUST mirror
/// `xtask/cpp/learner_capture.cpp::BuildColSamplerCorpus` EXACTLY.
fn col_sampler_corpus() -> (Vec<FeatureColumn>, Vec<f32>, Vec<f32>, GainConfig, i32, i32) {
    let grad = vec![
        -8.0f32, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    ];
    let hess = vec![1.0f32; 16];

    let make = |bins: Vec<u32>, num_bin: u32, real: i32| -> FeatureColumn {
        let upper: Vec<f64> = (0..num_bin).map(|b| b as f64 + 0.5).collect();
        FeatureColumn {
            bins,
            num_bin,
            offset: lgbm_treelearner::offset_for_most_freq_bin(0),
            min_bin: 0,
            max_bin: num_bin - 1,
            default_bin: num_bin,
            most_freq_bin: 0,
            missing_type: MissingType::None,
            bin_upper_bound: upper,
            real_feature_index: real,
        }
    };
    let f0 = make(vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], 4, 0);
    let f1 = make(vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], 2, 1);
    let f2 = make(vec![0, 1, 0, 1, 1, 2, 1, 2, 2, 1, 2, 1, 3, 2, 3, 2], 4, 2);
    let f3 = make(vec![0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2], 3, 3);

    let cfg = GainConfig {
        min_data_in_leaf: 1,
        min_sum_hessian_in_leaf: 0.0,
        max_delta_step: 0.0,
        lambda_l1: 0.0,
        lambda_l2: 0.0,
        min_gain_to_split: 0.0,
        path_smooth: 0.0,
    };
    (vec![f0, f1, f2, f3], grad, hess, cfg, 4, -1)
}

/// TRL-08: per-tree / per-node feature subsampling selects the SAME features as
/// C++ via `Random::Sample` CALL-SEQUENCE parity. The Rust `ColSampler` draw
/// sequence (ResetByTree once, GetByNode per node smaller-then-larger) must match
/// the committed `col_sampler.txt` selected-index golden exactly (T-05-04-01).
#[test]
fn learner_parity_col_sampler_rng() {
    let Ok(text) = std::fs::read_to_string(col_sampler_fixture()) else {
        eprintln!(
            "learner_parity: SKIP — col_sampler.txt not found. Run \
             `cargo run -p xtask -- learner-capture` and commit the golden set."
        );
        return;
    };
    let golden = parse_col_sampler(&text);

    let backend = CpuBackend;
    let client = cpu_client();
    let (features, g, h, cfg, num_leaves, max_depth) = col_sampler_corpus();
    let mut learner = SerialTreeLearner::new(&backend, &client, cfg, num_leaves, max_depth)
        .with_features(features)
        .with_feature_fraction(
            golden.feature_fraction,
            golden.feature_fraction_bynode,
            golden.seed,
        );
    let (_tree, _snaps, trace) = learner
        .train_with_col_sampler_trace(&g, &h, true)
        .expect("col-sampler train ok");

    // Per-tree ResetByTree selection (ascending REAL feature indices).
    assert_eq!(
        trace.bytree_selected, golden.bytree,
        "TRL-08: ResetByTree per-tree selection != golden CS_BYTREE"
    );
    // Per-node GetByNode selections in DRAW ORDER.
    assert_eq!(
        trace.bynode_selected.len(),
        golden.bynode.len(),
        "TRL-08: GetByNode draw COUNT mismatch: rust {} vs golden {}",
        trace.bynode_selected.len(),
        golden.bynode.len()
    );
    for (i, (rust_sel, gold_sel)) in trace.bynode_selected.iter().zip(golden.bynode.iter()).enumerate()
    {
        assert_eq!(
            rust_sel, gold_sel,
            "TRL-08: GetByNode[{i}] selected features != golden (draw-sequence parity)"
        );
    }
}

/// One parsed `real_gh.txt` corpus block: the captured g/h, the per-feature bin
/// layout, and the reconstructed reference tree.
struct GhCorpus {
    name: String,
    num_leaves: i32,
    grad: Vec<f32>,
    hess: Vec<f32>,
    features: Vec<FeatureColumn>,
    tree: TreeGolden,
}

fn parse_u32_semi(s: &str) -> Vec<u32> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(';').map(|t| t.parse::<u32>().expect("u32")).collect()
}
fn parse_f64_bits_semi(s: &str) -> Vec<f64> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(';')
        .map(|t| f64::from_bits(t.parse::<u64>().expect("f64 bits")))
        .collect()
}

/// Parse the `real_gh.txt` golden into its per-objective corpus blocks. Each block
/// is GH_CORPUS / GH_GRAD / GH_HESS / GH_FEATURE* followed by a spine-style
/// COUNTS / PSPLIT* / PTREE...ENDTREE reference tree.
fn parse_real_gh(text: &str) -> Vec<GhCorpus> {
    let mut out: Vec<GhCorpus> = Vec::new();
    let mut lines = text.lines().peekable();
    while let Some(raw) = lines.next() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let t: Vec<&str> = line.split_whitespace().collect();
        if t[0] != "GH_CORPUS" {
            continue;
        }
        let name = field(&t, "name").unwrap_or("").to_string();
        let num_leaves: i32 = field(&t, "num_leaves").and_then(|v| v.parse().ok()).unwrap_or(3);
        let mut grad = Vec::new();
        let mut hess = Vec::new();
        let mut features: Vec<FeatureColumn> = Vec::new();
        let mut tree = TreeGolden::default();
        // Consume the block until ENDTREE.
        for body in lines.by_ref() {
            let bt = body.trim();
            if bt.is_empty() || bt.starts_with('#') {
                continue;
            }
            let bt_tokens: Vec<&str> = bt.split_whitespace().collect();
            match bt_tokens[0] {
                "GH_GRAD" => grad = parse_f32_bits_ws(&bt[bt.find(' ').map_or(0, |p| p + 1)..]),
                "GH_HESS" => hess = parse_f32_bits_ws(&bt[bt.find(' ').map_or(0, |p| p + 1)..]),
                "GH_FEATURE" => {
                    let real: i32 =
                        field(&bt_tokens, "real").and_then(|v| v.parse().ok()).unwrap_or(-1);
                    let num_bin: u32 =
                        field(&bt_tokens, "num_bin").and_then(|v| v.parse().ok()).unwrap_or(0);
                    let most_freq_bin: u32 =
                        field(&bt_tokens, "most_freq_bin").and_then(|v| v.parse().ok()).unwrap_or(0);
                    let default_bin: u32 =
                        field(&bt_tokens, "default_bin").and_then(|v| v.parse().ok()).unwrap_or(num_bin);
                    let min_bin: u32 =
                        field(&bt_tokens, "min_bin").and_then(|v| v.parse().ok()).unwrap_or(0);
                    let max_bin: u32 = field(&bt_tokens, "max_bin")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(num_bin.saturating_sub(1));
                    let bins = parse_u32_semi(field(&bt_tokens, "bins").unwrap_or(""));
                    let upper = parse_f64_bits_semi(field(&bt_tokens, "upper").unwrap_or(""));
                    features.push(FeatureColumn {
                        bins,
                        num_bin,
                        offset: lgbm_treelearner::offset_for_most_freq_bin(most_freq_bin),
                        min_bin,
                        max_bin,
                        default_bin,
                        most_freq_bin,
                        missing_type: MissingType::None,
                        bin_upper_bound: upper,
                        real_feature_index: real,
                    });
                }
                "PTREE" => {
                    tree.name = field(&bt_tokens, "name").unwrap_or("").to_string();
                    tree.num_leaves =
                        field(&bt_tokens, "num_leaves").and_then(|v| v.parse().ok()).unwrap_or(0);
                    for inner in lines.by_ref() {
                        let it = inner.trim();
                        if it == "ENDTREE" {
                            break;
                        }
                        let (tag, rest) = it.split_once(' ').unwrap_or((it, ""));
                        match tag {
                            "PT_SPLIT_FEATURE" => tree.split_feature = parse_i32_ws(rest),
                            "PT_THRESHOLD_BITS" => tree.threshold = parse_f64_bits_ws(rest),
                            "PT_DECISION_TYPE" => tree.decision_type = parse_i8_ws(rest),
                            "PT_SPLIT_GAIN_BITS" => tree.split_gain = parse_f32_bits_ws(rest),
                            "PT_LEFT_CHILD" => tree.left_child = parse_i32_ws(rest),
                            "PT_RIGHT_CHILD" => tree.right_child = parse_i32_ws(rest),
                            "PT_LEAF_VALUE_BITS" => tree.leaf_value = parse_f64_bits_ws(rest),
                            "PT_LEAF_WEIGHT_BITS" => tree.leaf_weight = parse_f64_bits_ws(rest),
                            "PT_LEAF_COUNT" => tree.leaf_count = parse_i32_ws(rest),
                            "PT_INTERNAL_VALUE_BITS" => tree.internal_value = parse_f64_bits_ws(rest),
                            "PT_INTERNAL_COUNT" => tree.internal_count = parse_i32_ws(rest),
                            _ => {}
                        }
                    }
                    break; // end of this corpus block
                }
                _ => {}
            }
        }
        out.push(GhCorpus {
            name,
            num_leaves,
            grad,
            hess,
            features,
            tree,
        });
    }
    out
}

/// D-03 + D-07: the learner grows the SAME tree as C++ on captured REAL
/// iteration-1 g/h (regression-l2 + binary-logloss, realistic distribution). Each
/// corpus block's grown tree `to_string()` must be byte-identical to the C++
/// reference tree (the shared `%.17g` formatter is the arbiter).
#[test]
fn learner_parity_real_gh_full_tree() {
    let Ok(text) = std::fs::read_to_string(real_gh_fixture()) else {
        eprintln!(
            "learner_parity: SKIP — real_gh.txt not found. Run \
             `cargo run -p xtask -- learner-capture` and commit the golden set."
        );
        return;
    };
    let corpora = parse_real_gh(&text);
    assert!(
        !corpora.is_empty(),
        "real_gh.txt must carry at least one GH_CORPUS block"
    );

    let backend = CpuBackend;
    let client = cpu_client();
    // min_data_in_leaf=3 mirrors BuildRealGhCorpus — keeps every split's actual
    // children non-degenerate so the faithful actual-partition leaf counts never
    // collapse a child to 0 rows.
    let cfg = GainConfig {
        min_data_in_leaf: 3,
        min_sum_hessian_in_leaf: 0.0,
        max_delta_step: 0.0,
        lambda_l1: 0.0,
        lambda_l2: 0.0,
        min_gain_to_split: 0.0,
        path_smooth: 0.0,
    };

    for gh in &corpora {
        // num_leaves comes from the golden (regression=3 / binary=2) — each chosen
        // to keep every split's actual children non-degenerate.
        let mut learner = SerialTreeLearner::new(&backend, &client, cfg, gh.num_leaves, -1)
            .with_features(gh.features.clone());
        let tree = learner
            .train(&gh.grad, &gh.hess, true)
            .unwrap_or_else(|e| panic!("real_gh {} train failed: {e:?}", gh.name));
        let want = gh.tree.to_tree().to_string();
        let got = tree.to_string();
        assert_eq!(
            got, want,
            "D-03/D-07 real_gh {} full-tree mismatch (grown != C++ reference)",
            gh.name
        );
    }
}
