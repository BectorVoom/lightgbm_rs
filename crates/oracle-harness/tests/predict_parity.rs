//! Predict-mode parity replay (Phase 7, Wave 9 — plan 07-10, PRD-04 + PRD-05).
//!
//! Two NEW prediction modes are validated against real `lib_lightgbm` 4.6:
//!
//! - **PRD-04 TreeSHAP `predict_contrib`** — `contrib_*` cells load the captured
//!   model + input matrix, run [`lgbm_model::predict::predict_contrib_mat`], and
//!   assert (1) the per-class block (`[per-feature; expected-value base]`) matches
//!   the real-binary `pred_contrib=True` golden within `ORACLE_TOL`, AND (2) the
//!   load-bearing INVARIANT `Σ block == raw margin` holds (sum of each class block
//!   equals [`predict_raw_mat`]). Covers numeric, categorical (TreeSHAP over
//!   `find_in_bitset` decision nodes), and multiclass (per-class stride) trees.
//!
//! - **PRD-05 prediction early stopping** — `early_stop_*` cells replay the
//!   captured freq×margin axis through [`GbdtModel::predict_raw_early_stop`] and
//!   assert the frozen raw score matches the real-binary `pred_early_stop` golden
//!   within `ORACLE_TOL`.
//!
//! Idioms mirror `boosting_parity.rs` / `learner_parity.rs`: a
//! `CARGO_MANIFEST_DIR`-rooted fixture path (NEVER the untracked `LightGBM/`
//! tree), the `compare_within(.., ORACLE_TOL)` precision contract, and a graceful
//! SKIP (return) when a golden file is absent so a fresh checkout without the
//! capture run still builds + passes. Populate the goldens with
//! `LGBM_CAPTURE_PYTHON=… cargo run -p xtask -- predict-mode-oracle-capture`.

use std::path::PathBuf;

use lgbm_model::GbdtModel;
use oracle_harness::comparator::{compare_within, ORACLE_TOL};

/// The committed predict-mode golden directory — TRACKED under the oracle-harness
/// crate, NEVER the untracked C++/LightGBM reference tree.
fn predict_modes_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/predict_modes")
}

/// SKIP gracefully (returning `None`) when a golden file is absent — matching the
/// boosting_parity / learner_parity idiom.
fn read_golden(corpus: &str, file: &str) -> Option<String> {
    let path = predict_modes_dir().join(corpus).join(file);
    match std::fs::read_to_string(&path) {
        Ok(s) => Some(s),
        Err(_) => {
            eprintln!(
                "predict_parity: SKIP — golden {} not found. Run \
                 `LGBM_CAPTURE_PYTHON=… cargo run -p xtask -- predict-mode-oracle-capture`.",
                path.display()
            );
            None
        }
    }
}

/// Parse a `# rows=<n> <key>=<m>` header + body of space-separated f64 bit
/// patterns into `(rows, cols, flat f64 row-major)`.
fn parse_matrix(text: &str) -> (usize, usize, Vec<f64>) {
    let mut lines = text.lines();
    let header = lines.next().expect("matrix header");
    let (rows, cols) = parse_dims(header);
    let mut flat = Vec::with_capacity(rows * cols);
    for line in lines {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        for tok in line.split_whitespace() {
            flat.push(f64::from_bits(tok.parse::<u64>().expect("u64 bits")));
        }
    }
    assert_eq!(flat.len(), rows * cols, "matrix body length != rows*cols");
    (rows, cols, flat)
}

/// Parse `rows=<n>` and the second `=<m>` (cols/width) out of a header line.
fn parse_dims(header: &str) -> (usize, usize) {
    let mut rows = 0usize;
    let mut second = 0usize;
    for tok in header.split_whitespace() {
        if let Some(v) = tok.strip_prefix("rows=") {
            rows = v.parse().expect("rows");
        } else if let Some(v) = tok.strip_prefix("cols=") {
            second = v.parse().expect("cols");
        } else if let Some(v) = tok.strip_prefix("width=") {
            second = v.parse().expect("width");
        }
    }
    (rows, second)
}

/// Load the corpus model + input matrix; returns `None` to SKIP when absent.
fn load_corpus(corpus: &str) -> Option<(GbdtModel, usize, usize, Vec<f32>)> {
    let model_text = read_golden(corpus, "model.txt")?;
    let x_text = read_golden(corpus, "X.txt")?;
    let model = lgbm_model::model_text::load(&model_text).expect("parse golden model");
    let (rows, cols, x_f64) = parse_matrix(&x_text);
    let x_f32: Vec<f32> = x_f64.iter().map(|&v| v as f32).collect();
    Some((model, rows, cols, x_f32))
}

/// Run a contrib parity cell for `corpus`: compare `predict_contrib_mat` to the
/// real-binary golden within ORACLE_TOL AND assert the sum+base==raw invariant.
fn run_contrib_cell(corpus: &str) {
    let Some((model, rows, cols, x)) = load_corpus(corpus) else {
        return;
    };
    let Some(golden_text) = read_golden(corpus, "contrib.txt") else {
        return;
    };
    let (g_rows, g_width, golden) = parse_matrix(&golden_text);
    assert_eq!(g_rows, rows, "{corpus}: contrib golden row count mismatch");

    let rust = lgbm_model::predict::predict_contrib_mat(&model, &x, rows as i32, cols as i32)
        .expect("predict_contrib_mat");
    let per_row = rust.len() / rows;
    assert_eq!(
        per_row, g_width,
        "{corpus}: contrib width rust {per_row} != golden {g_width}"
    );

    // (1) Within ORACLE_TOL vs the real-binary golden (f32 contract).
    let rust_f32: Vec<f32> = rust.iter().map(|&v| v as f32).collect();
    let golden_f32: Vec<f32> = golden.iter().map(|&v| v as f32).collect();
    compare_within(&rust_f32, &golden_f32, ORACLE_TOL)
        .unwrap_or_else(|m| panic!("{corpus}: contrib not within ORACLE_TOL vs golden: {m:?}"));

    // (2) The load-bearing INVARIANT: per class block sum == raw margin.
    let raw = lgbm_model::predict::predict_raw_mat(&model, &x, rows as i32, cols as i32)
        .expect("predict_raw_mat");
    let ntpi = model.num_tree_per_iteration.max(0) as usize;
    let nf = (model.max_feature_idx + 1).max(0) as usize;
    let block = nf + 1;
    for r in 0..rows {
        for k in 0..ntpi {
            let off = r * per_row + k * block;
            let sum: f64 = rust[off..off + block].iter().sum();
            let raw_k = raw[r * ntpi + k] as f64;
            assert!(
                (sum - raw_k).abs() <= 1e-5,
                "{corpus}: row {r} class {k} SHAP invariant: sum {sum} != raw {raw_k}"
            );
        }
    }
}

#[test]
fn contrib_numeric() {
    run_contrib_cell("numeric");
}

#[test]
fn contrib_categorical() {
    run_contrib_cell("categorical");
}

#[test]
fn contrib_multiclass() {
    run_contrib_cell("multiclass");
}

/// Run the early-stop freq×margin axis for `corpus`: replay each captured cell
/// through `predict_raw_early_stop` and compare the frozen raw score within
/// ORACLE_TOL.
fn run_early_stop_cell(corpus: &str) {
    let Some((model, rows, cols, x)) = load_corpus(corpus) else {
        return;
    };
    let Some(golden_text) = read_golden(corpus, "early_stop.txt") else {
        return;
    };

    let ntpi = model.num_tree_per_iteration.max(0) as usize;

    // Parse the CELL blocks: `CELL freq=<f> margin=<g> rows=<n> width=<w>` then rows.
    let mut lines = golden_text.lines().peekable();
    let mut any_cell = false;
    while let Some(header) = lines.next() {
        let header = header.trim();
        if !header.starts_with("CELL") {
            continue;
        }
        any_cell = true;
        let mut freq = 0i32;
        let mut margin = 0.0f64;
        let (g_rows, _g_width) = parse_dims(header);
        for tok in header.split_whitespace() {
            if let Some(v) = tok.strip_prefix("freq=") {
                freq = v.parse().expect("freq");
            } else if let Some(v) = tok.strip_prefix("margin=") {
                margin = v.parse().expect("margin");
            }
        }
        assert_eq!(g_rows, rows, "{corpus}: early_stop golden row count mismatch");

        // Read `rows` body lines into the golden score matrix.
        let mut golden = Vec::with_capacity(rows * ntpi);
        for _ in 0..rows {
            let body = lines.next().expect("early_stop body line").trim();
            for tok in body.split_whitespace() {
                golden.push(f64::from_bits(tok.parse::<u64>().expect("u64 bits")));
            }
        }

        // Replay through the model-aware early-stop driver (applies the C++
        // `!NeedAccuratePrediction()` gate: regression returns the full score,
        // binary/multiclass freeze at the margin).
        let (rust_f32, _iters) = lgbm_model::predict::predict_raw_early_stop_mat(
            &model, &x, rows as i32, cols as i32, freq, margin,
        )
        .expect("predict_raw_early_stop_mat");
        let golden_f32: Vec<f32> = golden.iter().map(|&v| v as f32).collect();
        compare_within(&rust_f32, &golden_f32, ORACLE_TOL).unwrap_or_else(|m| {
            panic!(
                "{corpus}: early_stop freq={freq} margin={margin} not within ORACLE_TOL: {m:?}"
            )
        });
    }
    assert!(any_cell, "{corpus}: early_stop.txt had no CELL blocks");
}

#[test]
fn early_stop_numeric() {
    run_early_stop_cell("numeric");
}

#[test]
fn early_stop_multiclass() {
    run_early_stop_cell("multiclass");
}

// ===========================================================================
// Phase-18 (18-01, ODL-15) on-device tree-walk predict scaffold. The new cells
// read the committed `tests/fixtures/kernels/predict.txt` golden (VERBATIM
// AddPredictionToScoreKernel, cuda_tree.cu:317-396): numeric missing/default +
// threshold and categorical `FindInBitset` membership, `score += leaf_value[~node]`
// in f64. The device tree-walk kernel lands in **18-04** (Wave 2); until then each
// cell parses + structurally validates and is `#[ignore]`d so the merge gate stays
// GREEN with `LGBM_CUDA_ON_DEVICE` unset (D-13 / ODL-19). 18-04 replaces the
// `// UN-IGNORE (18-04):` block with the real device call + `compare_within`.
//
// Record format (see `tests/fixtures/kernels/predict.txt`):
//   PREDICT name=.. kind=numeric|categorical num_rows=N num_features=F \
//           num_nodes=M num_leaves=L
//   PFEAT  <per-feature `min,max,default,mfb,offset,bit_type,column`; ';'-sep>
//   PNODE  <per-node `split_feature_inner,threshold_in_bin,decision_type,left,right`>
//   PLEAF  <leaf values, f64 bits ';'-sep>
//   PCATPOOL <bitset_inner words>  ; PCATBOUND <cat_boundaries_inner>
//   PROWS  <N rows × F raw column bins; rows ';'-sep, cols ','-sep>
//   PSCORE <N reference raw margins, f64 bits ';'-sep>
// ===========================================================================

/// The committed kernels golden dir — TRACKED under oracle-harness, never the
/// untracked C++/LightGBM reference tree.
fn kernels_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/kernels")
}

/// One parsed PREDICT block from `predict.txt` — the full synthetic model
/// (feature meta + tree arrays + categorical bitset pool) the 18-04 device
/// tree-walk replays.
#[allow(dead_code)]
struct PredictGolden {
    name: String,
    kind: String,
    num_rows: usize,
    num_features: usize,
    scores: Vec<f64>,
    rows: Vec<Vec<u32>>,
    // Per-feature meta (parallel vecs, indexed by split_feature_inner).
    feat_min: Vec<i32>,
    feat_max: Vec<i32>,
    feat_default: Vec<i32>,
    feat_mfb: Vec<i32>,
    feat_offset: Vec<i32>,
    feat_bit_type: Vec<u32>,
    feat_column: Vec<i32>,
    // Per-node tree arrays.
    split_feature_inner: Vec<i32>,
    threshold_in_bin: Vec<u32>,
    decision_type: Vec<i32>,
    left_child: Vec<i32>,
    right_child: Vec<i32>,
    leaf_value: Vec<f64>,
    // Categorical bitset pool + boundaries.
    bitset_inner: Vec<u32>,
    cat_boundaries_inner: Vec<i32>,
}

/// Parse `predict.txt` into its PREDICT blocks, SKIP (empty vec via `None`) absent.
fn read_predict_golden() -> Option<Vec<PredictGolden>> {
    let path = kernels_dir().join("predict.txt");
    let text = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(_) => {
            eprintln!("predict_parity(kernels): SKIP — golden {} not found.", path.display());
            return None;
        }
    };
    let mut out = Vec::new();
    let mut lines = text.lines();
    while let Some(raw) = lines.next() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let t: Vec<&str> = line.split_whitespace().collect();
        if t[0] != "PREDICT" {
            continue;
        }
        let get = |key: &str| -> String {
            t.iter()
                .find_map(|tok| tok.strip_prefix(key).and_then(|r| r.strip_prefix('=')))
                .unwrap_or_else(|| panic!("PREDICT missing `{key}`"))
                .to_string()
        };
        let name = get("name");
        let kind = get("kind");
        let num_rows: usize = get("num_rows").parse().expect("num_rows");
        let num_features: usize = get("num_features").parse().expect("num_features");

        // The block body: PFEAT PNODE PLEAF PCATPOOL PCATBOUND PROWS PSCORE.
        let mut rows: Vec<Vec<u32>> = Vec::new();
        let mut scores: Vec<f64> = Vec::new();
        let mut feat_min = Vec::new();
        let mut feat_max = Vec::new();
        let mut feat_default = Vec::new();
        let mut feat_mfb = Vec::new();
        let mut feat_offset = Vec::new();
        let mut feat_bit_type = Vec::new();
        let mut feat_column = Vec::new();
        let mut split_feature_inner = Vec::new();
        let mut threshold_in_bin = Vec::new();
        let mut decision_type = Vec::new();
        let mut left_child = Vec::new();
        let mut right_child = Vec::new();
        let mut leaf_value = Vec::new();
        let mut bitset_inner = Vec::new();
        let mut cat_boundaries_inner = Vec::new();
        for body in lines.by_ref() {
            let bt: Vec<&str> = body.trim().split_whitespace().collect();
            if bt.is_empty() {
                continue;
            }
            let payload = bt.get(1).copied().unwrap_or("");
            match bt[0] {
                "PFEAT" => {
                    // per feature `min,max,default,mfb,offset,bit_type,column` ';'-sep.
                    for f in payload.split(';').filter(|s| !s.is_empty()) {
                        let c: Vec<&str> = f.split(',').collect();
                        feat_min.push(c[0].parse().expect("feat min"));
                        feat_max.push(c[1].parse().expect("feat max"));
                        feat_default.push(c[2].parse().expect("feat default"));
                        feat_mfb.push(c[3].parse().expect("feat mfb"));
                        feat_offset.push(c[4].parse().expect("feat offset"));
                        feat_bit_type.push(c[5].parse().expect("feat bit_type"));
                        feat_column.push(c[6].parse().expect("feat column"));
                    }
                }
                "PNODE" => {
                    // per node `split_feature_inner,threshold_in_bin,decision_type,left,right`.
                    for n in payload.split(';').filter(|s| !s.is_empty()) {
                        let c: Vec<&str> = n.split(',').collect();
                        split_feature_inner.push(c[0].parse().expect("node sfi"));
                        threshold_in_bin.push(c[1].parse().expect("node thr"));
                        decision_type.push(c[2].parse().expect("node dt"));
                        left_child.push(c[3].parse().expect("node left"));
                        right_child.push(c[4].parse().expect("node right"));
                    }
                }
                "PLEAF" => {
                    for l in payload.split(';').filter(|s| !s.is_empty()) {
                        leaf_value.push(f64::from_bits(l.parse::<u64>().expect("leaf f64 bits")));
                    }
                }
                "PCATPOOL" => {
                    for w in payload.split(';').filter(|s| !s.is_empty()) {
                        bitset_inner.push(w.parse::<u32>().expect("cat pool word"));
                    }
                }
                "PCATBOUND" => {
                    for b in payload.split(';').filter(|s| !s.is_empty()) {
                        cat_boundaries_inner.push(b.parse::<i32>().expect("cat boundary"));
                    }
                }
                "PROWS" => {
                    if !payload.is_empty() {
                        rows = payload
                            .split(';')
                            .map(|r| r.split(',').map(|c| c.parse::<u32>().expect("row bin")).collect())
                            .collect();
                    }
                }
                "PSCORE" => {
                    if !payload.is_empty() {
                        scores = payload
                            .split(';')
                            .map(|s| f64::from_bits(s.parse::<u64>().expect("score f64 bits")))
                            .collect();
                    }
                    break; // PSCORE is the last line of a block
                }
                _ => {}
            }
        }
        out.push(PredictGolden {
            name,
            kind,
            num_rows,
            num_features,
            scores,
            rows,
            feat_min,
            feat_max,
            feat_default,
            feat_mfb,
            feat_offset,
            feat_bit_type,
            feat_column,
            split_feature_inner,
            threshold_in_bin,
            decision_type,
            left_child,
            right_child,
            leaf_value,
            bitset_inner,
            cat_boundaries_inner,
        });
    }
    Some(out)
}

/// Structural validation shared by both cells: row/score counts match the header.
fn assert_predict_shape(g: &PredictGolden) {
    assert_eq!(g.scores.len(), g.num_rows, "PREDICT `{}`: PSCORE count != num_rows", g.name);
    assert_eq!(g.rows.len(), g.num_rows, "PREDICT `{}`: PROWS count != num_rows", g.name);
    for r in &g.rows {
        assert_eq!(r.len(), g.num_features, "PREDICT `{}`: row width != num_features", g.name);
    }
    for s in &g.scores {
        assert!(s.is_finite(), "PREDICT `{}`: reference margin must be finite", g.name);
    }
}

/// Build the `PredictTree` view over a parsed golden block.
fn tree_of(g: &PredictGolden) -> lgbm_compute::kernels::predict::PredictTree<'_> {
    lgbm_compute::kernels::predict::PredictTree {
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
    }
}

/// Flatten the golden's `PROWS` into a row-major `[num_rows × num_features]` store.
fn flat_rows(g: &PredictGolden) -> Vec<u32> {
    g.rows.iter().flat_map(|r| r.iter().copied()).collect()
}

/// Drive the device tree-walk on the cpu f64 anchor for `g` at column width
/// `bit_type` and assert the raw-margin scores match the golden within `ORACLE_TOL`
/// (the f32 contract) AND bit-exact vs the cpu f64 anchor (lib_lightgbm-transcribed
/// golden). The device kernel emits the RAW pre-inverse-link margin (Phase-19
/// boundary), exactly what `predict.txt` PSCORE captures.
fn drive_and_assert(g: &PredictGolden, bit_type: u32) {
    use lgbm_compute::kernels::predict::add_prediction_to_score_on_device;
    use lgbm_compute::runtime::cpu_client;

    let client = cpu_client();
    let tree = tree_of(g);
    let rows = flat_rows(g);
    let got = add_prediction_to_score_on_device(
        &client,
        &tree,
        &rows,
        g.num_rows,
        g.num_features,
        bit_type,
        g.num_rows,
        None,
    )
    .unwrap_or_else(|e| panic!("PREDICT `{}` device walk (bit_type={bit_type}) failed: {e:?}", g.name));

    // (1) bit-exact vs the golden PSCORE on the cpu f64 anchor (D-04/D-12).
    for (i, (gt, ex)) in got.iter().zip(g.scores.iter()).enumerate() {
        assert_eq!(
            gt.to_bits(),
            ex.to_bits(),
            "PREDICT `{}` (bit_type={bit_type}) row {i}: device margin {gt} != golden {ex}",
            g.name
        );
    }
    // (2) within ORACLE_TOL under the f32 contract (the ~1e-6 numerical gate).
    let got_f32: Vec<f32> = got.iter().map(|&v| v as f32).collect();
    let exp_f32: Vec<f32> = g.scores.iter().map(|&v| v as f32).collect();
    compare_within(&got_f32, &exp_f32, ORACLE_TOL)
        .unwrap_or_else(|m| panic!("PREDICT `{}` (bit_type={bit_type}) not within ORACLE_TOL: {m:?}", g.name));
}

/// `on_device` cell (ODL-15, numeric tree-walk 8/16/32 vs the f64 anchor).
mod on_device {
    use super::*;

    #[test]
    fn predict_parity_on_device_numeric() {
        let Some(goldens) = read_predict_golden() else { return };
        let numeric: Vec<&PredictGolden> = goldens.iter().filter(|g| g.kind == "numeric").collect();
        assert!(!numeric.is_empty(), "predict.txt has no numeric PREDICT block");
        for g in numeric {
            assert_predict_shape(g);
            // A bin is a width-invariant index: drive the SAME numeric model at 8-,
            // 16-, and 32-bit column width (D-05 dispatch coverage) — each must match
            // the golden raw margins bit-exact + within ORACLE_TOL.
            for bit_type in [8u32, 16, 32] {
                drive_and_assert(g, bit_type);
            }
        }
    }
}

/// `cat` cell (ODL-15, categorical membership predict: cat_onehot / cat_manyvsmany).
mod cat {
    use super::*;

    #[test]
    #[ignore = "categorical cell + hip gate land in 18-04 Task 2"]
    fn predict_parity_cat_membership() {
        let Some(goldens) = read_predict_golden() else { return };
        let cats: Vec<&PredictGolden> = goldens.iter().filter(|g| g.kind == "categorical").collect();
        assert!(!cats.is_empty(), "predict.txt has no categorical PREDICT block");
        // Both the one-hot and many-vs-many models must be present (D-03/D-05).
        assert!(cats.iter().any(|g| g.name == "cat_onehot"), "missing cat_onehot block");
        assert!(cats.iter().any(|g| g.name == "cat_manyvsmany"), "missing cat_manyvsmany block");
        for g in cats {
            assert_predict_shape(g);
            // Drive at the model's native column width (cat_onehot 8-bit, cat_manyvsmany
            // 16-bit); the categorical branch reuses the shared find_in_bitset helper.
            let bit_type = *g.feat_bit_type.iter().max().unwrap_or(&8);
            drive_and_assert(g, bit_type);
        }
    }
}
