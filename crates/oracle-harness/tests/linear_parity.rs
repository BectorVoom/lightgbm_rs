//! Linear-tree (`linear_tree`) parity replay against real `lib_lightgbm` 4.6.
//!
//! Three layers are asserted over the committed goldens in `tests/fixtures/linear/`
//! (each `<case>/` produced by `fixtures/linear/gen_linear_goldens.py` from the pip
//! `lightgbm==4.6.0` wheel — the model-oracle binary, ORA/D-05):
//!
//! - **Model side** — [`lgbm_model::model_text::load`] parses the C++ linear
//!   `model.txt`, [`predict::predict_mat`] reproduces the captured C++ `pred.csv`
//!   within `ORACLE_TOL`, AND each tree's `Tree::to_string()` re-serializes the C++
//!   tree block BYTE-for-byte (write-side format parity, D-07).
//! - **Fit core** — for each non-constant tree, the C++ structure is stripped of its
//!   linear model and re-fit with [`lgbm_treelearner::linear::fit_linear_leaves`]
//!   over the reconstructed L2 gradient (`score_before_i - y`); the coefficients +
//!   `leaf_features` ordering match the C++ golden within `ORACLE_TOL`.
//! - **End-to-end training** — [`lgbm::booster::train_raw`] with `linear_tree=true`
//!   trains from the raw corpus (binning → serial learner → GBDT loop → per-leaf
//!   linear fit) and its predictions match the C++ `pred.csv` within `ORACLE_TOL`.
//!
//! Idioms mirror `learner_parity.rs` / `predict_parity.rs`: a
//! `CARGO_MANIFEST_DIR`-rooted fixture path (NEVER the untracked C++ tree) and a
//! graceful SKIP (return) if the fixtures are absent, so a checkout without them
//! still builds + passes.

use std::path::{Path, PathBuf};

use lgbm::booster::{train_raw, RawCorpus};
use lgbm_core::config::Config;
use lgbm_dataset::LeafPartitionLayout;
use lgbm_model::{model_text, predict, Tree};
use lgbm_treelearner::linear::fit_linear_leaves;
use lgbm_treelearner::DataPartition;
use oracle_harness::comparator::ORACLE_TOL;

fn linear_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/linear")
}

/// The committed golden cases, or `None` (SKIP) when the fixtures are absent.
fn cases() -> Option<Vec<PathBuf>> {
    let base = linear_dir();
    let rd = std::fs::read_dir(&base).ok()?;
    let mut cases: Vec<PathBuf> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("model.txt").exists())
        .collect();
    if cases.is_empty() {
        return None;
    }
    cases.sort();
    Some(cases)
}

fn read_mat(path: &Path) -> (Vec<Vec<f64>>, usize, usize) {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    let rows: Vec<Vec<f64>> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split_whitespace().map(|t| t.parse().unwrap()).collect())
        .collect();
    let cols = rows.first().map_or(0, Vec::len);
    let n = rows.len();
    (rows, n, cols)
}

fn read_vec(path: &Path) -> Vec<f64> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {path:?}: {e}"))
        .split_whitespace()
        .map(|t| t.parse().unwrap())
        .collect()
}

fn json_num(dir: &Path, key: &str) -> Option<f64> {
    let json = std::fs::read_to_string(dir.join("params.json")).ok()?;
    let idx = json.find(&format!("\"{key}\""))?;
    let after = &json[idx + key.len() + 2..];
    let colon = after.find(':')?;
    let num: String = after[colon + 1..]
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == 'e')
        .collect();
    num.parse().ok()
}

/// Split a full model.txt into per-tree block bodies (exactly what
/// `Tree::to_string` emits: field lines each `\n` + one trailing blank line).
fn extract_tree_blocks(model_text: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut cur: Option<String> = None;
    for line in model_text.lines() {
        if line.strip_prefix("Tree=").is_some() {
            cur = Some(String::new());
        } else if line == "end of trees" {
            break;
        } else if let Some(b) = cur.as_mut() {
            if line.is_empty() {
                b.push('\n');
                blocks.push(cur.take().unwrap());
            } else {
                b.push_str(line);
                b.push('\n');
            }
        }
    }
    blocks
}

/// Build a `DataPartition` from routing each row through `tree`'s real thresholds.
/// For the golden corpora this matches C++'s bin partition closely enough to
/// validate the fit math in isolation.
fn partition_from_tree(tree: &Tree, raw: &[f64], n_rows: usize, n_feat: usize) -> DataPartition {
    let nl = tree.num_leaves.max(1) as usize;
    let mut per_leaf: Vec<Vec<u32>> = vec![Vec::new(); nl];
    for r in 0..n_rows {
        let row = &raw[r * n_feat..(r + 1) * n_feat];
        let leaf = if tree.num_leaves > 1 { tree.get_leaf(row) as usize } else { 0 };
        per_leaf[leaf].push(r as u32);
    }
    let mut indices = Vec::with_capacity(n_rows);
    let mut leaf_begin = vec![0i32; nl];
    let mut leaf_count = vec![0i32; nl];
    for (l, rows) in per_leaf.iter().enumerate() {
        leaf_begin[l] = indices.len() as i32;
        leaf_count[l] = rows.len() as i32;
        indices.extend_from_slice(rows);
    }
    DataPartition::from_payload(LeafPartitionLayout {
        num_data: n_rows as i32,
        indices,
        leaf_begin,
        leaf_count,
    })
}

fn flat_rowmajor(rows: &[Vec<f64>]) -> Vec<f64> {
    rows.iter().flatten().copied().collect()
}

#[test]
fn linear_model_predict_and_serialize_match_cpp() {
    let Some(cases) = cases() else {
        eprintln!("linear_parity: SKIP — tests/fixtures/linear absent.");
        return;
    };
    for dir in &cases {
        let name = dir.file_name().unwrap().to_string_lossy();
        let text = std::fs::read_to_string(dir.join("model.txt")).unwrap();
        let model = model_text::load(&text).expect("parse linear model");

        // Write-side byte parity: each tree round-trips to the C++ block verbatim.
        for (i, cpp_block) in extract_tree_blocks(&text).iter().enumerate() {
            assert_eq!(
                &model.trees[i].to_string(),
                cpp_block,
                "[{name}] tree {i}: Rust serialization != C++ model.txt block"
            );
        }

        // Predict parity on the held-out matrix.
        let (rows, nr, nc) = read_mat(&dir.join("X_test.csv"));
        let x: Vec<f32> = rows.iter().flatten().map(|&v| v as f32).collect();
        let expected = read_vec(&dir.join("pred.csv"));
        let out = predict::predict_mat(&model, &x, nr as i32, nc as i32).expect("predict");
        let worst = out
            .iter()
            .zip(&expected)
            .map(|(o, e)| (*o as f64 - e).abs())
            .fold(0.0f64, f64::max);
        assert!(
            worst <= f64::from(ORACLE_TOL),
            "[{name}] model-side predict max|Δ| {worst:.3e} > {ORACLE_TOL:.1e}"
        );
    }
}

#[test]
fn linear_fit_matches_cpp() {
    let Some(cases) = cases() else {
        eprintln!("linear_parity: SKIP — tests/fixtures/linear absent.");
        return;
    };
    for dir in &cases {
        let name = dir.file_name().unwrap().to_string_lossy();
        // The isolated fit check re-derives each leaf's rows from the FULL corpus;
        // a bagging case fit each tree on a per-iter sampled subset we can't
        // reconstruct here, so skip it (the e2e test covers linear+bagging).
        if json_num(dir, "bagging_freq").unwrap_or(0.0) > 0.0
            && json_num(dir, "bagging_fraction").unwrap_or(1.0) < 1.0
        {
            continue;
        }
        let model = model_text::load(&std::fs::read_to_string(dir.join("model.txt")).unwrap()).unwrap();
        let (xr, n_rows, n_feat) = read_mat(&dir.join("X_train.csv"));
        let raw = flat_rowmajor(&xr);
        let y = read_vec(&dir.join("y_train.csv"));
        let lr = json_num(dir, "learning_rate").unwrap_or(0.1);
        let lam = json_num(dir, "linear_lambda").unwrap_or(0.0);

        for (ti, tree) in model.trees.iter().enumerate() {
            let golden = match &tree.linear {
                Some(l) if l.leaf_coeff.iter().any(|c| !c.is_empty()) => l.clone(),
                _ => continue, // constant tree (e.g. first) — nothing to fit
            };
            let score_before = read_vec(&dir.join(format!("scores/score_before_{ti}.csv")));
            let grad: Vec<f32> = (0..n_rows).map(|r| (score_before[r] - y[r]) as f32).collect();
            let hess = vec![1.0f32; n_rows];

            let mut t = tree.clone();
            t.is_linear = false;
            t.linear = None;
            let partition = partition_from_tree(&t, &raw, n_rows, n_feat);
            let raw_cols =
                lgbm_treelearner::linear::RawFeatureColumns::from_fn(n_rows, n_feat, |r, c| {
                    xr[r][c]
                });
            fit_linear_leaves(&mut t, &raw_cols, &grad, &hess, lam, &partition);
            t.shrinkage(lr);

            let fitted = t.linear.as_ref().unwrap();
            for l in 0..golden.leaf_const.len() {
                assert_eq!(
                    fitted.leaf_features[l], golden.leaf_features[l],
                    "[{name}] tree {ti} leaf {l}: leaf_features order differs"
                );
                let dc = (fitted.leaf_const[l] - golden.leaf_const[l]).abs();
                assert!(dc <= f64::from(ORACLE_TOL), "[{name}] tree {ti} leaf {l}: const Δ {dc:.3e}");
                for j in 0..golden.leaf_coeff[l].len() {
                    let d = (fitted.leaf_coeff[l][j] - golden.leaf_coeff[l][j]).abs();
                    assert!(d <= f64::from(ORACLE_TOL), "[{name}] tree {ti} leaf {l} coeff {j}: Δ {d:.3e}");
                }
            }
        }
    }
}

#[test]
fn linear_train_end_to_end_matches_cpp() {
    let Some(cases) = cases() else {
        eprintln!("linear_parity: SKIP — tests/fixtures/linear absent.");
        return;
    };
    for dir in &cases {
        let name = dir.file_name().unwrap().to_string_lossy();
        let (x_train, _, _) = read_mat(&dir.join("X_train.csv"));
        let y_train: Vec<f32> = read_vec(&dir.join("y_train.csv")).iter().map(|&v| v as f32).collect();
        let (x_test, _, _) = read_mat(&dir.join("X_test.csv"));
        let expected = read_vec(&dir.join("pred.csv"));

        let mut config = Config::default();
        config.objective = "regression".to_string();
        config.num_leaves = json_num(dir, "num_leaves").unwrap() as i32;
        config.learning_rate = json_num(dir, "learning_rate").unwrap();
        config.num_iterations = json_num(dir, "num_boost_round").unwrap() as i32;
        config.min_data_in_leaf = json_num(dir, "min_data_in_leaf").unwrap() as i32;
        config.linear_tree = true;
        config.linear_lambda = json_num(dir, "linear_lambda").unwrap_or(0.0);
        config.boost_from_average = true;
        // Row bagging (linear-tree subset path), when the golden used it.
        config.bagging_fraction = json_num(dir, "bagging_fraction").unwrap_or(1.0);
        config.bagging_freq = json_num(dir, "bagging_freq").unwrap_or(0.0) as i32;
        config.bagging_seed = json_num(dir, "bagging_seed").unwrap_or(0.0) as i32;

        let corpus = RawCorpus::new(x_train, y_train);
        let booster = train_raw(&config, &corpus).expect("train linear");
        let preds = booster.predict(&x_test);
        let worst = preds
            .iter()
            .zip(&expected)
            .map(|(r, e)| (r[0] as f64 - e).abs())
            .fold(0.0f64, f64::max);
        assert!(
            worst <= f64::from(ORACLE_TOL),
            "[{name}] end-to-end train predict max|Δ| {worst:.3e} > {ORACLE_TOL:.1e}"
        );
    }
}
