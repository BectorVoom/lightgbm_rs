//! ORACLE parity for the `feature_fraction` (per-tree column subsampling) family.
//!
//! # The gap this closes
//!
//! The pre-existing `learner/col_sampler.txt` golden pinned exactly ONE cell —
//! `feature_fraction = 1.0`, `feature_fraction_bynode = 0.5` — over a SINGLE tree.
//! At `feature_fraction = 1.0` the per-tree draw is a no-op, so the entire
//! `feature_fraction` code path was unverified against C++. Three defects lived in
//! that gap (all fixed 2026-08-06, each of which this test fails against):
//!
//! 1. **Never wired.** `SerialTreeLearner::with_feature_fraction` was called only
//!    from `learner_parity`, so `lgbm::train*` trained an UNSAMPLED model however
//!    `feature_fraction` was set.
//! 2. **PRNG re-seeded per tree.** `ColSampler` was reconstructed on every tree, so
//!    each tree drew the SAME subset instead of advancing the seeded stream.
//! 3. **Off by one tree.** C++ draws in `SetTrainingData` (Init) AND in
//!    `BeforeTrain` (per tree), so C++ tree `N` uses draw `N+2`; consuming the Init
//!    draw for tree 0 shifted every tree's subset by one.
//!
//! 4. **`feature_fraction_bynode` gated the SCAN, not the argmax.** C++ scans every
//!    bytree-selected feature (so `is_splittable_` reflects real data) and applies
//!    the per-node mask only at `if (new_split > *best_split && is_feature_used)`.
//!    Skipping the scan left `is_splittable_ = false`, which propagates to both
//!    children via `parent_splittable` and permanently drops features C++ still
//!    considers deeper down — a compounding, depth-dependent error. Interaction
//!    constraints, which C++ folds into the same `GetByNode` mask, were mis-gated
//!    identically and are fixed by the same change.
//!
//! Defects 2-4 are invisible to a single-tree golden, which is why this fixture
//! spans 8 trees and nine (fraction, bynode) cells and asserts the PER-TREE selected
//! feature sets, not just the final predictions.
//!
//! Regenerate: `.venv/bin/python crates/oracle-harness/tests/fixtures/feature_fraction/capture.py`

use std::collections::HashMap;
use std::path::PathBuf;

use lgbm::{train_raw, Config, RawCorpus};
use oracle_harness::comparator::{compare_within, ORACLE_TOL};

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/feature_fraction/feature_fraction_golden.json")
}

struct Golden {
    features: Vec<f64>,
    labels: Vec<f32>,
    num_rows: usize,
    num_cols: usize,
    num_iterations: i32,
    cells: Vec<Cell>,
}

struct Cell {
    feature_fraction: f64,
    feature_fraction_bynode: f64,
    pred: Vec<f64>,
    per_tree_split_features: Vec<Vec<i32>>,
}

/// Decode `"name": ["<16-hex>", ...]` as f64s from their IEEE-754 bits.
fn hex_array(src: &str, name: &str) -> Option<Vec<f64>> {
    let at = src.find(name)?;
    let open = src[at..].find('[')? + at;
    let close = src[open..].find(']')? + open;
    Some(
        src[open + 1..close]
            .split(',')
            .filter_map(|t| u64::from_str_radix(t.trim().trim_matches('"'), 16).ok())
            .map(f64::from_bits)
            .collect(),
    )
}

fn int_field(src: &str, name: &str) -> Option<i64> {
    let at = src.find(name)? + name.len();
    let rest = src[at..].trim_start().trim_start_matches(':').trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit() && c != '-')?;
    rest[..end].parse().ok()
}

fn float_field(src: &str, name: &str) -> Option<f64> {
    let at = src.find(name)? + name.len();
    let rest = src[at..].trim_start().trim_start_matches(':').trim_start();
    let end = rest
        .find(|c: char| !c.is_ascii_digit() && c != '-' && c != '.' && c != 'e')
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Parse `"per_tree_split_features": [[..], [..]]` into per-tree index lists.
fn nested_int_lists(src: &str, name: &str) -> Vec<Vec<i32>> {
    let Some(at) = src.find(name) else { return Vec::new() };
    let Some(open) = src[at..].find('[').map(|i| i + at) else { return Vec::new() };
    let mut depth = 0i32;
    let mut end = open;
    for (i, b) in src[open..].bytes().enumerate() {
        match b {
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    end = open + i;
                    break;
                }
            }
            _ => {}
        }
    }
    let inner = &src[open + 1..end];
    let mut out = Vec::new();
    let mut cur: Option<String> = None;
    for ch in inner.chars() {
        match ch {
            '[' => cur = Some(String::new()),
            ']' => {
                if let Some(buf) = cur.take() {
                    out.push(
                        buf.split(',')
                            .filter_map(|t| t.trim().parse::<i32>().ok())
                            .collect(),
                    );
                }
            }
            c => {
                if let Some(buf) = cur.as_mut() {
                    buf.push(c);
                }
            }
        }
    }
    out
}

fn load_golden() -> Option<Golden> {
    let raw = std::fs::read_to_string(golden_path()).ok()?;
    let features = hex_array(&raw, "\"features_bits\"")?;
    let labels = hex_array(&raw, "\"labels_bits\"")?
        .into_iter()
        .map(|v| v as f32)
        .collect();
    let num_rows = int_field(&raw, "\"num_rows\"")? as usize;
    let num_cols = int_field(&raw, "\"num_cols\"")? as usize;
    let num_iterations = int_field(&raw, "\"num_iterations\"")? as i32;

    // Cells are keyed "<ff>:<bynode>"; split the `cells` object on its per-cell
    // objects, which each carry their own fraction fields (so key parsing is moot).
    let cells_at = raw.find("\"cells\"")?;
    let body = &raw[cells_at..];
    let mut cells = Vec::new();
    let mut idx = 0usize;
    while let Some(rel) = body[idx..].find("\"feature_fraction\"") {
        let start = idx + rel;
        // The enclosing cell object runs from the previous '{' to its match.
        let obj_start = body[..start].rfind('{')?;
        let mut depth = 0i32;
        let mut obj_end = obj_start;
        for (i, b) in body[obj_start..].bytes().enumerate() {
            match b {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        obj_end = obj_start + i;
                        break;
                    }
                }
                _ => {}
            }
        }
        let cell = &body[obj_start..=obj_end];
        cells.push(Cell {
            feature_fraction: float_field(cell, "\"feature_fraction\"")?,
            feature_fraction_bynode: float_field(cell, "\"feature_fraction_bynode\"")?,
            pred: hex_array(cell, "\"pred_bits\"").unwrap_or_default(),
            per_tree_split_features: nested_int_lists(cell, "\"per_tree_split_features\""),
        });
        idx = obj_end + 1;
    }
    Some(Golden { features, labels, num_rows, num_cols, num_iterations, cells })
}

fn config_for(g: &Golden, cell: &Cell) -> Config {
    let params: HashMap<String, String> = [
        ("objective", "binary".to_string()),
        ("num_iterations", g.num_iterations.to_string()),
        ("learning_rate", "0.1".to_string()),
        ("num_leaves", "8".to_string()),
        ("min_data_in_leaf", "5".to_string()),
        ("max_bin", "63".to_string()),
        ("seed", "1".to_string()),
        ("deterministic", "true".to_string()),
        ("force_row_wise", "true".to_string()),
        ("num_threads", "1".to_string()),
        ("feature_fraction", cell.feature_fraction.to_string()),
        ("feature_fraction_bynode", cell.feature_fraction_bynode.to_string()),
        ("feature_fraction_seed", "2".to_string()),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    Config::from_params(&params).expect("config builds")
}

fn corpus(g: &Golden, config: &Config) -> RawCorpus {
    let columns: Vec<Vec<f64>> = (0..g.num_cols)
        .map(|c| (0..g.num_rows).map(|r| g.features[r * g.num_cols + c]).collect())
        .collect();
    let mut raw = RawCorpus::from_columns(columns, g.labels.clone());
    raw.config = config.clone();
    raw
}

/// The per-tree sorted split-feature sets of a model text — the direct witness that
/// each tree drew its OWN column subset.
fn per_tree_split_features(model: &str) -> Vec<Vec<i32>> {
    model
        .split("Tree=")
        .skip(1)
        .map(|chunk| {
            let mut feats: Vec<i32> = chunk
                .lines()
                .filter_map(|l| l.strip_prefix("split_feature="))
                .flat_map(|v| v.split_whitespace())
                .filter_map(|t| t.parse::<i32>().ok())
                .collect();
            feats.sort_unstable();
            feats.dedup();
            feats
        })
        .collect()
}

#[test]
fn feature_fraction_predictions_match_the_cpp_oracle() {
    let Some(g) = load_golden() else {
        eprintln!("SKIP: feature_fraction golden absent — run fixtures/feature_fraction/capture.py");
        return;
    };
    assert!(!g.cells.is_empty(), "golden present but parsed to zero cells");

    for cell in &g.cells {
        let config = config_for(&g, cell);
        let c = corpus(&g, &config);
        let booster = train_raw(&config, &c).expect("train ok");
        let rows = c.to_rows();
        let rust: Vec<f32> = rows
            .iter()
            .map(|r| booster.predict_row_raw(r, -1)[0] as f32)
            .collect();
        let cpp: Vec<f32> = cell.pred.iter().map(|&v| v as f32).collect();
        compare_within(&rust, &cpp, ORACLE_TOL).unwrap_or_else(|m| {
            panic!(
                "feature_fraction={} bynode={}: raw predictions diverge from C++: {m:?}",
                cell.feature_fraction, cell.feature_fraction_bynode
            )
        });
    }
}

#[test]
fn each_tree_selects_the_same_column_subset_as_cpp() {
    // The sharpest of the three regressions: a per-tree PRNG re-seed, or an
    // off-by-one in the draw sequence, leaves predictions "plausible" but makes the
    // per-tree feature sets wrong. Compare them tree by tree.
    let Some(g) = load_golden() else {
        eprintln!("SKIP: feature_fraction golden absent");
        return;
    };
    for cell in &g.cells {
        if cell.per_tree_split_features.is_empty() {
            continue;
        }
        let config = config_for(&g, cell);
        let c = corpus(&g, &config);
        let booster = train_raw(&config, &c).expect("train ok");
        let rust = per_tree_split_features(&booster.model_to_string());
        assert_eq!(
            rust.len(),
            cell.per_tree_split_features.len(),
            "feature_fraction={}: tree count differs",
            cell.feature_fraction
        );
        for (i, (r, cpp)) in rust.iter().zip(&cell.per_tree_split_features).enumerate() {
            assert_eq!(
                r, cpp,
                "feature_fraction={} tree {i}: split-feature set differs from C++\n  rust: {r:?}\n  cpp:  {cpp:?}",
                cell.feature_fraction
            );
        }
    }
}

#[test]
fn a_fraction_below_one_actually_subsamples_and_varies_per_tree() {
    // Guards defect 1 (never wired) and defect 2 (re-seeded per tree) independently
    // of the C++ goldens, so the intent survives a fixture regeneration.
    let Some(g) = load_golden() else {
        eprintln!("SKIP: feature_fraction golden absent");
        return;
    };
    let Some(cell) = g.cells.iter().find(|c| c.feature_fraction <= 0.25) else {
        return;
    };
    let config = config_for(&g, cell);
    let c = corpus(&g, &config);
    let booster = train_raw(&config, &c).expect("train ok");
    let per_tree = per_tree_split_features(&booster.model_to_string());

    // (1) It subsamples: no tree may split on more columns than the fraction allows.
    let cap = (g.num_cols as f64 * cell.feature_fraction).round().max(1.0) as usize;
    for (i, t) in per_tree.iter().enumerate() {
        assert!(
            t.len() <= cap,
            "tree {i} split on {} columns, above the feature_fraction cap {cap}",
            t.len()
        );
    }
    // (2) It re-draws per tree: a re-seeded PRNG would make every tree identical.
    let distinct = per_tree.iter().collect::<std::collections::HashSet<_>>().len();
    assert!(
        distinct > 1,
        "every tree selected the SAME column subset ({distinct} distinct across {} trees) \
         — the per-tree draw is not advancing the PRNG",
        per_tree.len()
    );
}
