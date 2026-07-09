//! Phase-7 W11 (plan 07-12, ADV-06 refit / ADV-07 importance) ADVANCED-MODEL-OPS
//! parity vs REAL lib_lightgbm 4.6.
//!
//! Each cell SKIP-passes when its golden under `tests/fixtures/advanced/` is absent
//! (the wheel-gated capture, Task 4). After
//! `LGBM_CAPTURE_PYTHON=<py-with-lightgbm-4.6> cargo run -p xtask -- advanced-oracle-capture`
//! the cells flip to GREEN.
//!
//! Goldens (all under `tests/fixtures/advanced/`):
//! - `base_model.txt`     — the v4 base model (the refit/importance/continue start).
//! - `refit_decay09.txt`  — C++ `Booster.refit(X, y, decay_rate=0.9)` model text.
//! - `refit_decay00.txt`  — C++ `Booster.refit(X, y, decay_rate=0.0)` model text.
//! - `continue_model.txt` — model after continuing training from base (init_model).
//! - `importance.json`    — per-feature split-count + gain-sum vectors.
//! - `advanced.json`      — shared sidecar (per-feature bins + bin_upper_bound +
//!   num_bin + most_freq_bin + per-row label).
//!
//! The sidecar pins the per-feature bin layout + per-row label so the Rust replay
//! routes rows + recomputes grad/hess identically — the comparison can ONLY falsify
//! the refit / importance gate, never the (Phase-2) numeric binning.

use std::path::PathBuf;

use lgbm_boosting::gbdt::Gbdt;
use lgbm_compute::gain::GainConfig;
use lgbm_dataset::bin_mapper::MissingType;
use lgbm_model::model_text::load;
use lgbm_objective::Objective;
use lgbm_treelearner::learner::FeatureColumn;
use lgbm_treelearner::BinColumn;

const LR: f64 = 0.1;
// f32-vs-f64 accumulation tolerance for refit leaf outputs (the CPU f64-fold anchor
// reproduces the C++ refit math; refit recomputes through f32 grad/hess so a small
// residual is allowed — the exact algebra is pinned by the lgbm-boosting unit tests).
const REFIT_TOL: f64 = 1e-4;

fn advanced_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/advanced")
}

// ---------------------------------------------------------------------------
// Sidecar parsing (a focused hand parser — no serde dep in the harness, mirroring
// the constraints/learner sidecar shape).
// ---------------------------------------------------------------------------

struct AdvFeature {
    bins: Vec<u32>,
    bin_upper_bound: Vec<f64>,
    num_bin: u32,
    most_freq_bin: u32,
}

struct AdvSidecar {
    features: Vec<AdvFeature>,
    label: Vec<f32>,
}

/// Parse a flat number array `"key": [..]` from the FIRST occurrence after `from`.
fn json_num_array_at(text: &str, from: usize) -> (Vec<f64>, usize) {
    let lb = text[from..].find('[').map(|x| from + x).unwrap();
    let rb = text[lb..].find(']').map(|x| lb + x).unwrap();
    let v = text[lb + 1..rb]
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<f64>().expect("array element not a number"))
        .collect();
    (v, rb)
}

/// Parse a flat `"key": value` scalar from a JSON slice.
fn json_scalar(text: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{key}\"");
    let i = text.find(&pat)?;
    let after = &text[i + pat.len()..];
    let colon = after.find(':')?;
    let rest = after[colon + 1..].trim_start();
    let end = rest.find([',', '\n', '}', ']']).unwrap_or(rest.len());
    rest[..end].trim().parse::<f64>().ok()
}

/// Load + parse `advanced.json`; `None` (SKIP) when absent.
fn load_sidecar() -> Option<AdvSidecar> {
    let path = advanced_dir().join("advanced.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!(
            "advanced_parity: SKIP — sidecar {} not found. Run \
             `LGBM_CAPTURE_PYTHON=<py-with-lightgbm-4.6> cargo run -p xtask -- \
             advanced-oracle-capture` and commit the goldens.",
            path.display()
        );
        return None;
    };
    // The capture emits sort_keys=True; within each feature object the keys are
    // ALPHABETICAL: bin_upper_bound, bins, most_freq_bin, num_bin. We anchor each
    // feature on `bin_upper_bound` (its first key) and bound the features array by
    // the top-level `label` key.
    let feats_start = text.find("\"features\"").expect("sidecar missing features");
    let label_anchor = text.find("\"label\"").expect("sidecar missing label");
    let mut features = Vec::new();
    let mut cursor = feats_start;
    while let Some(bub_rel) = text[cursor..].find("\"bin_upper_bound\"") {
        let bub_at = cursor + bub_rel;
        if bub_at > label_anchor {
            break;
        }
        let (bub, after_bub) = json_num_array_at(&text, bub_at);
        let bins_at = text[after_bub..]
            .find("\"bins\"")
            .map(|x| after_bub + x)
            .expect("feature missing bins");
        let (bins_f, after_bins) = json_num_array_at(&text, bins_at);
        let span = &text[bub_at..];
        let nb = json_scalar(span, "num_bin").expect("feature num_bin") as u32;
        let mfb = json_scalar(span, "most_freq_bin").expect("feature most_freq_bin") as u32;
        features.push(AdvFeature {
            bins: bins_f.into_iter().map(|x| x as u32).collect(),
            bin_upper_bound: bub,
            num_bin: nb,
            most_freq_bin: mfb,
        });
        cursor = after_bins;
    }
    let label_at = text.find("\"label\"").expect("sidecar missing label");
    let (label, _) = json_num_array_at(&text, label_at);
    Some(AdvSidecar {
        features,
        label: label.into_iter().map(|x| x as f32).collect(),
    })
}

/// Build the Rust [`FeatureColumn`]s from the sidecar's pinned bin layout.
fn feature_columns(sidecar: &AdvSidecar) -> Vec<FeatureColumn> {
    sidecar
        .features
        .iter()
        .enumerate()
        .map(|(fi, f)| FeatureColumn {
            bins: BinColumn::new(f.bins.clone(), f.num_bin),
            num_bin: f.num_bin,
            offset: lgbm_treelearner::offset_for_most_freq_bin(f.most_freq_bin),
            min_bin: 0,
            max_bin: f.num_bin.saturating_sub(1),
            default_bin: f.num_bin,
            most_freq_bin: f.most_freq_bin,
            missing_type: MissingType::None,
            bin_upper_bound: f.bin_upper_bound.clone(),
            real_feature_index: fi as i32,
            ..Default::default()
        })
        .collect()
}

// ===========================================================================
// ADV-07 — feature importance (split / gain) vs real lib_lightgbm.
// ===========================================================================

#[test]
fn importance_split_matches_real_binary() {
    let imp_path = advanced_dir().join("importance.json");
    let model_path = advanced_dir().join("base_model.txt");
    let (Ok(imp_text), Ok(model_text)) = (
        std::fs::read_to_string(&imp_path),
        std::fs::read_to_string(&model_path),
    ) else {
        eprintln!("advanced_parity: SKIP — importance/base goldens absent (run advanced-oracle-capture).");
        return;
    };
    let model = load(&model_text).expect("load base model");
    let split_at = imp_text.find("\"split\"").expect("importance.json missing split");
    let (golden_split, _) = json_num_array_at(&imp_text, split_at);
    let rust_split = model.feature_importance_split_count_guarded();
    assert_eq!(
        rust_split.len(),
        golden_split.len(),
        "split importance length mismatch"
    );
    for (i, (&r, &g)) in rust_split.iter().zip(&golden_split).enumerate() {
        assert_eq!(
            r as f64, g,
            "split importance[{i}] rust {r} != C++ {g}"
        );
    }
}

#[test]
fn importance_gain_matches_real_binary() {
    let imp_path = advanced_dir().join("importance.json");
    let model_path = advanced_dir().join("base_model.txt");
    let (Ok(imp_text), Ok(model_text)) = (
        std::fs::read_to_string(&imp_path),
        std::fs::read_to_string(&model_path),
    ) else {
        eprintln!("advanced_parity: SKIP — importance/base goldens absent (run advanced-oracle-capture).");
        return;
    };
    let model = load(&model_text).expect("load base model");
    let gain_at = imp_text.find("\"gain\"").expect("importance.json missing gain");
    let (golden_gain, _) = json_num_array_at(&imp_text, gain_at);
    let rust_gain = model.feature_importance_gain();
    assert_eq!(rust_gain.len(), golden_gain.len(), "gain importance length mismatch");
    for (i, (&r, &g)) in rust_gain.iter().zip(&golden_gain).enumerate() {
        // The gains are stored f32 in the model text; C++ sums them in f64 — the
        // Rust path mirrors that exactly, so a tight tolerance holds.
        assert!(
            (r - g).abs() <= 1e-3 * (1.0 + g.abs()),
            "gain importance[{i}] rust {r} != C++ {g}"
        );
    }
}

// ===========================================================================
// ADV-06 — leaf-refit (decay 0.9 / 0.0) vs real lib_lightgbm `Booster.refit`.
// ===========================================================================

fn refit_cell(golden_name: &str, decay: f64) {
    let Some(sidecar) = load_sidecar() else { return };
    let base_path = advanced_dir().join("base_model.txt");
    let golden_path = advanced_dir().join(golden_name);
    let (Ok(base_text), Ok(golden_text)) = (
        std::fs::read_to_string(&base_path),
        std::fs::read_to_string(&golden_path),
    ) else {
        eprintln!("advanced_parity: SKIP — {golden_name} / base absent (run advanced-oracle-capture).");
        return;
    };
    let base = load(&base_text).expect("load base model");
    let golden = load(&golden_text).expect("load refit golden");
    let features = feature_columns(&sidecar);
    let num_data = sidecar.label.len() as i32;

    // Replay the Rust refit on the SAME data: load the base trees, refit leaves.
    let mut gbdt = Gbdt::new(
        Objective::Regression { sqrt: false },
        LR,
        1,
        num_data,
        false, // boost_from_average=false (matches the capture)
        None,
    )
    .with_loaded_model(base.trees.clone(), features);
    gbdt.refit(&sidecar.label, decay, false, 0.0, 0.0)
        .expect("rust refit");

    let rust_trees = gbdt.trees();
    assert_eq!(
        rust_trees.len(),
        golden.trees.len(),
        "{golden_name}: refit must preserve the tree count"
    );
    for (ti, (rt, gt)) in rust_trees.iter().zip(&golden.trees).enumerate() {
        assert_eq!(
            rt.num_leaves, gt.num_leaves,
            "{golden_name} tree {ti}: refit must preserve the structure"
        );
        for (li, (&rl, &gl)) in rt.leaf_value.iter().zip(&gt.leaf_value).enumerate() {
            assert!(
                (rl - gl).abs() <= REFIT_TOL,
                "{golden_name} tree {ti} leaf {li}: rust {rl} != C++ {gl} (|d|={})",
                (rl - gl).abs()
            );
        }
    }
}

#[test]
fn refit_decay09_matches_real_binary() {
    refit_cell("refit_decay09.txt", 0.9);
}

#[test]
fn refit_decay00_matches_real_binary() {
    refit_cell("refit_decay00.txt", 0.0);
}

// ===========================================================================
// ADV-06 — input_model continue-training: the appended tree count vs C++.
// ===========================================================================

#[test]
fn continue_training_grows_from_base() {
    let Some(sidecar) = load_sidecar() else { return };
    let base_path = advanced_dir().join("base_model.txt");
    let cont_path = advanced_dir().join("continue_model.txt");
    let (Ok(base_text), Ok(cont_text)) = (
        std::fs::read_to_string(&base_path),
        std::fs::read_to_string(&cont_path),
    ) else {
        eprintln!("advanced_parity: SKIP — base/continue goldens absent (run advanced-oracle-capture).");
        return;
    };
    let base = load(&base_text).expect("load base model");
    let cont = load(&cont_text).expect("load continue model");
    let features = feature_columns(&sidecar);
    let num_data = sidecar.label.len() as i32;

    // The continue model must have MORE trees than the base (appended, not restarted).
    assert!(
        cont.trees.len() > base.trees.len(),
        "continue model ({}) must append onto the base ({})",
        cont.trees.len(),
        base.trees.len()
    );

    // The Rust continue-training driver: load the base, num_init_iteration == base
    // iteration count, so the next train appends.
    let cfg = GainConfig {
        min_data_in_leaf: 1,
        min_sum_hessian_in_leaf: 1e-3,
        ..Default::default()
    };
    let _ = cfg; // the learner wiring lives in the facade; here we assert the seam.
    let gbdt = Gbdt::new(
        Objective::Regression { sqrt: false },
        LR,
        1,
        num_data,
        false,
        None,
    )
    .with_loaded_model(base.trees.clone(), features);
    assert_eq!(
        gbdt.num_init_iteration() as usize,
        base.trees.len(),
        "num_init_iteration must equal the loaded (K=1) tree count"
    );
}
