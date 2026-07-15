//! Model-text envelope (de)serialization — `LoadModelFromString` /
//! `SaveModelToString` (DAT-08 / DAT-09).
//!
//! Transcribed from `LightGBM/src/boosting/gbdt_model_text.cpp`:
//! - `SaveModelToString` (311-407) — the byte-exact envelope order: `tree` /
//!   `version=v4` / `num_class` / `num_tree_per_iteration` / `label_index` /
//!   `max_feature_idx` / [`objective`] / [`average_output`] / `feature_names` /
//!   [`monotone_constraints`] / `feature_infos` / `tree_sizes` / blank /
//!   `Tree=i\n<ToString>\n` ... / `end of trees` / blank / `feature_importances:`
//!   block / `\nparameters:` ... `end of parameters`.
//! - `LoadModelFromString` (421-624) — header key parse, `tree_sizes=` byte
//!   boundaries, the `parameters:`/`parser:` tail.
//! - `FeatureImportance` (627) — recomputed on write (split-count default).
//!
//! # Round-trip de-risk (RESEARCH Summary line 50)
//! Ensemble metadata (`feature_names`, `feature_infos`, `objective`,
//! `monotone_constraints`) and the entire `parameters:`..EOF tail are captured
//! VERBATIM and re-emitted unchanged — only per-tree float arrays round-trip
//! through `Tree::parse` → `Tree::to_string`. So byte-stability reduces to
//! (a) preserve those strings and (b) reproduce `%.17g`/`{:g}` per tree (done in
//! [`crate::format`] + [`crate::tree`]).
//!
//! # The Python `pandas_categorical:null` tail
//! pip-`lightgbm`'s `Booster.save_model` appends a `pandas_categorical:...` line
//! AFTER the C++ `SaveModelToString` output. The committed golden is the
//! pip-written file, so a byte-exact round-trip must reproduce that tail too — it
//! is part of the verbatim `parameters:`..EOF trailer captured at load.
//!
//! # Strictness (Security V5 / T-03-04, T-03-05)
//! `tree_sizes` byte arithmetic is checked (`sum(tree_sizes) <= text.len()`);
//! `feature_names`/`feature_infos` counts are validated against
//! `max_feature_idx + 1`. Any defect → [`ModelError::MalformedModel`], never a
//! panic.

use crate::ensemble::{GbdtModel, MODEL_VERSION, SUB_MODEL_NAME};
use crate::error::ModelError;
use crate::tree::Tree;

/// Parse a header `key=value` line into `(key, value)`, splitting on the FIRST
/// `=`. The model envelope values (`feature_names`, `feature_infos`) themselves
/// never contain `=`, but per-tree blocks aren't parsed here so a first-`=` split
/// is faithful to the C++ `Split(line, '=')` with the `feature_names`/
/// `monotone_constraints` substr special-cases (gbdt_model_text.cpp:439-442).
fn split_kv(line: &str) -> Option<(&str, &str)> {
    line.split_once('=')
}

/// Load a model `.txt` into a [`GbdtModel`] (C++ `GBDT::LoadModelFromString`).
///
/// `text` is the full model-text bytes (the committed v4 fixture). Honors the
/// `tree_sizes=` byte boundaries to slice per-tree blocks. Captures the envelope
/// metadata + the `parameters:`..EOF tail verbatim for a byte-exact round-trip.
pub fn load(text: &str) -> Result<GbdtModel, ModelError> {
    // --- Phase 1: parse the header up to (but not including) the first Tree= line. ---
    // Collect header key=value pairs. We also need the byte offset where the tree
    // region begins to slice trees by `tree_sizes`.
    let mut num_class: Option<i32> = None;
    let mut num_tree_per_iteration: Option<i32> = None;
    let mut label_index: Option<i32> = None;
    let mut max_feature_idx: Option<i32> = None;
    let mut average_output = false;
    let mut objective_string: Option<String> = None;
    let mut feature_names: Option<String> = None;
    let mut feature_infos: Option<String> = None;
    let mut monotone_constraints: Option<String> = None;
    let mut tree_sizes_line: Option<String> = None;

    // Track byte offset: header lines are consumed until the first `Tree=` line.
    let mut offset = 0usize;
    let bytes = text.as_bytes();
    let mut tree_region_start = text.len();

    for line in text.split_inclusive('\n') {
        let content = line.strip_suffix('\n').unwrap_or(line);
        let content = content.strip_suffix('\r').unwrap_or(content);
        if content.starts_with("Tree=") {
            tree_region_start = offset;
            break;
        }
        if !content.is_empty() {
            if content == "average_output" {
                average_output = true;
            } else if let Some((key, val)) = split_kv(content) {
                match key {
                    "num_class" => num_class = Some(parse_i32(val, "num_class")?),
                    "num_tree_per_iteration" => {
                        num_tree_per_iteration = Some(parse_i32(val, "num_tree_per_iteration")?)
                    }
                    "label_index" => label_index = Some(parse_i32(val, "label_index")?),
                    "max_feature_idx" => max_feature_idx = Some(parse_i32(val, "max_feature_idx")?),
                    "objective" => objective_string = Some(val.to_string()),
                    "feature_names" => feature_names = Some(val.to_string()),
                    "feature_infos" => feature_infos = Some(val.to_string()),
                    "monotone_constraints" => monotone_constraints = Some(val.to_string()),
                    "tree_sizes" => tree_sizes_line = Some(val.to_string()),
                    _ => { /* tree / version=v4 / other header keys: ignored */ }
                }
            }
        }
        offset += line.len();
    }

    let num_class = num_class.ok_or_else(|| missing("num_class"))?;
    let num_tree_per_iteration = num_tree_per_iteration.unwrap_or(num_class);
    let label_index = label_index.ok_or_else(|| missing("label_index"))?;
    let max_feature_idx = max_feature_idx.ok_or_else(|| missing("max_feature_idx"))?;
    let feature_names = feature_names.ok_or_else(|| missing("feature_names"))?;
    let feature_infos = feature_infos.ok_or_else(|| missing("feature_infos"))?;

    // T-03-05: count checks vs max_feature_idx+1.
    let expected_cols = (max_feature_idx + 1) as usize;
    let fn_count = feature_names.split(' ').count();
    if fn_count != expected_cols {
        return Err(ModelError::MalformedModel {
            detail: format!("feature_names has {fn_count} entries, expected {expected_cols}"),
        });
    }
    let fi_count = feature_infos.split(' ').count();
    if fi_count != expected_cols {
        return Err(ModelError::MalformedModel {
            detail: format!("feature_infos has {fi_count} entries, expected {expected_cols}"),
        });
    }

    let tree_sizes_str = tree_sizes_line.ok_or_else(|| missing("tree_sizes"))?;
    let tree_sizes: Vec<usize> = tree_sizes_str
        .split_whitespace()
        .map(|t| {
            t.parse::<usize>().map_err(|_| ModelError::MalformedModel {
                detail: format!("tree_sizes contains a non-usize token: {t:?}"),
            })
        })
        .collect::<Result<_, _>>()?;

    // --- Phase 2: slice + parse trees by byte boundary (T-03-04). ---
    // The tree region begins after the blank line that follows `tree_sizes=`.
    // `tree_region_start` is the offset of the first `Tree=` line.
    let region = &text[tree_region_start..];
    // Validate the byte arithmetic against the available region length.
    let mut total: usize = 0;
    for &sz in &tree_sizes {
        total = total.checked_add(sz).ok_or_else(|| ModelError::MalformedModel {
            detail: "tree_sizes sum overflows usize".to_string(),
        })?;
    }
    if total > region.len() {
        return Err(ModelError::MalformedModel {
            detail: format!(
                "tree_sizes sum {total} exceeds available tree region {} bytes",
                region.len()
            ),
        });
    }

    let region_bytes = region.as_bytes();
    let mut trees: Vec<Tree> = Vec::with_capacity(tree_sizes.len());
    let mut boundary = 0usize;
    for (i, &sz) in tree_sizes.iter().enumerate() {
        let block = &region[boundary..boundary + sz];
        // Each block is "Tree=i\n<ToString>\n". Strip the leading Tree= line.
        let inner = block.strip_prefix(&format!("Tree={i}\n")).ok_or_else(|| {
            ModelError::MalformedModel {
                detail: format!("expected a `Tree={i}` line at tree boundary {boundary}"),
            }
        })?;
        let tree = Tree::parse(inner)?;
        trees.push(tree);
        boundary += sz;
    }
    let _ = (bytes, region_bytes); // (byte views retained for clarity / future use)

    // --- Phase 3: capture the `parameters:`..EOF tail verbatim. ---
    // After the trees comes `end of trees\n`, a blank line, the
    // `feature_importances:` block, a blank line, then `parameters:`..EOF.
    // For a byte-exact round-trip we recompute the importances and preserve the
    // `parameters:`-onward bytes verbatim.
    let after_trees = &region[boundary..];
    let trailer = find_parameters_tail(after_trees);

    Ok(GbdtModel {
        trees,
        num_class,
        num_tree_per_iteration,
        label_index,
        max_feature_idx,
        average_output,
        objective_string,
        feature_names,
        feature_infos,
        monotone_constraints,
        trailer,
    })
}

/// Find the `parameters:` line and return everything from it to end of string
/// verbatim (the round-trip trailer). Returns `None` if there is no
/// `parameters:` block.
fn find_parameters_tail(s: &str) -> Option<String> {
    let mut offset = 0usize;
    for line in s.split_inclusive('\n') {
        let content = line.strip_suffix('\n').unwrap_or(line);
        let content = content.strip_suffix('\r').unwrap_or(content);
        if content == "parameters:" {
            return Some(s[offset..].to_string());
        }
        offset += line.len();
    }
    None
}

/// Serialize a [`GbdtModel`] to model text, byte-identical to C++
/// `GBDT::SaveModelToString` (+ the verbatim Python tail). Reproduces the
/// envelope order, per-tree `Tree::to_string`, the `tree_sizes=` line computed
/// from each serialized tree's byte length (including its trailing `\n`), the
/// recomputed `feature_importances:` block, and the verbatim `parameters:` tail.
pub fn save(model: &GbdtModel) -> String {
    // C++ `SaveModelToString(..., feature_importance_type, ...)` defaults the
    // importance type to split (0) at the public save entry points; the
    // `saved_feature_importance_type` selector (ADV-07) routes through
    // [`save_with_importance`].
    save_with_importance(model, 0)
}

/// C++ `GBDT::SaveModelToString(start, num, feature_importance_type)`
/// (`gbdt_model_text.cpp:311-409`) — serialize the ensemble, emitting the
/// `feature_importances:` block per `importance_type` (`0 = split`, `1 = gain`),
/// the `saved_feature_importance_type` selector (config.h:615). Both branches apply
/// the C++ `split_gain > 0` guard (the CR-02 Phase-3 follow-up); gain values are
/// truncated to `size_t` (`static_cast<size_t>`) and only feature entries `> 0` are
/// written, stable-sorted descending by the truncated value — byte-faithful to C++.
pub fn save_with_importance(model: &GbdtModel, importance_type: i32) -> String {
    let mut s = String::new();
    s.push_str(SUB_MODEL_NAME);
    s.push('\n');
    s.push_str(&format!("version={MODEL_VERSION}\n"));
    s.push_str(&format!("num_class={}\n", model.num_class));
    s.push_str(&format!(
        "num_tree_per_iteration={}\n",
        model.num_tree_per_iteration
    ));
    s.push_str(&format!("label_index={}\n", model.label_index));
    s.push_str(&format!("max_feature_idx={}\n", model.max_feature_idx));
    if let Some(obj) = &model.objective_string {
        s.push_str(&format!("objective={obj}\n"));
    }
    if model.average_output {
        s.push_str("average_output\n");
    }
    s.push_str(&format!("feature_names={}\n", model.feature_names));
    if let Some(mc) = &model.monotone_constraints {
        s.push_str(&format!("monotone_constraints={mc}\n"));
    }
    s.push_str(&format!("feature_infos={}\n", model.feature_infos));

    // Serialize each tree to "Tree=i\n<ToString>\n" and compute its byte length.
    let mut tree_strs: Vec<String> = Vec::with_capacity(model.trees.len());
    let mut tree_sizes: Vec<usize> = Vec::with_capacity(model.trees.len());
    for (i, tree) in model.trees.iter().enumerate() {
        let block = format!("Tree={i}\n{}\n", tree.to_string());
        tree_sizes.push(block.len());
        tree_strs.push(block);
    }

    s.push_str("tree_sizes=");
    for (i, sz) in tree_sizes.iter().enumerate() {
        if i > 0 {
            s.push(' ');
        }
        s.push_str(&sz.to_string());
    }
    s.push('\n');
    s.push('\n');

    for block in &tree_strs {
        s.push_str(block);
    }
    s.push_str("end of trees\n");

    // feature_importances: recomputed per `importance_type`, descending stable-sort,
    // keeping only entries whose `static_cast<size_t>` value is > 0. C++ applies the
    // `split_gain > 0` guard for BOTH split (0) and gain (1)
    // (gbdt_model_text.cpp:636-655). Gain values are truncated to integer via the
    // `static_cast<size_t>` the C++ emit performs (gbdt_model_text.cpp:378).
    let importances: Vec<u64> = if importance_type == 1 {
        // Gain: truncate the f64 gain-sum to integer (C++ static_cast<size_t>).
        model
            .feature_importance_gain()
            .iter()
            .map(|&g| if g > 0.0 { g as u64 } else { 0 })
            .collect()
    } else {
        // Split: the CR-02-guarded split count (split_gain > 0).
        model.feature_importance_split_count_guarded()
    };
    let feature_names: Vec<&str> = model.feature_names.split(' ').collect();
    // pair (count, original_index) keeping only >0; stable-sort by count desc.
    let mut pairs: Vec<(u64, usize)> = importances
        .iter()
        .enumerate()
        .filter(|&(_, &c)| c > 0)
        .map(|(i, &c)| (c, i))
        .collect();
    pairs.sort_by(|a, b| b.0.cmp(&a.0)); // stable, descending by count
    s.push('\n');
    s.push_str("feature_importances:\n");
    for (count, idx) in &pairs {
        let name = feature_names.get(*idx).copied().unwrap_or("");
        s.push_str(&format!("{name}={count}\n"));
    }

    // The `parameters:`..EOF tail, verbatim (C++ emits `\nparameters:` — the
    // leading blank line — then the params block; the captured trailer begins at
    // `parameters:`, so we emit the `\n` separator here).
    if let Some(trailer) = &model.trailer {
        s.push('\n');
        s.push_str(trailer);
    }

    s
}

fn missing(key: &str) -> ModelError {
    ModelError::MalformedModel {
        detail: format!("model file doesn't contain `{key}`"),
    }
}

fn parse_i32(s: &str, key: &str) -> Result<i32, ModelError> {
    s.trim().parse::<i32>().map_err(|_| ModelError::MalformedModel {
        detail: format!("{key} is not a valid integer: {s:?}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal hand-authored v4 model with one 2-leaf tree, exercising the full
    /// envelope round-trip (header + tree_sizes + tree + importances + params).
    fn minimal_model_text() -> String {
        // Build the tree block first so we can compute its tree_sizes byte length.
        let tree = Tree {
            num_leaves: 2,
            num_cat: 0,
            left_child: vec![-1],
            right_child: vec![-2],
            split_feature: vec![0],
            threshold: vec![0.5],
            decision_type: vec![2],
            split_gain: vec![1.5],
            leaf_value: vec![0.1, 0.2],
            leaf_weight: vec![10.0, 20.0],
            leaf_count: vec![10, 20],
            internal_value: vec![0.15],
            internal_weight: vec![30.0],
            internal_count: vec![30],
            cat_boundaries: vec![],
            cat_threshold: vec![],
            shrinkage: 0.1,
            is_linear: false,
            linear: None,
            leaf_depth: vec![1, 1],
            leaf_parent: vec![0, 0],
            split_feature_inner: vec![-1],
            threshold_in_bin: vec![0],
        };
        let block = format!("Tree=0\n{}\n", tree.to_string());
        let size = block.len();
        let mut s = String::new();
        s.push_str("tree\nversion=v4\nnum_class=1\nnum_tree_per_iteration=1\n");
        s.push_str("label_index=0\nmax_feature_idx=1\nobjective=regression\n");
        s.push_str("feature_names=Column_0 Column_1\n");
        s.push_str("feature_infos=[0:1] [0:1]\n");
        s.push_str(&format!("tree_sizes={size}\n"));
        s.push('\n');
        s.push_str(&block);
        s.push_str("end of trees\n");
        s.push('\n');
        s.push_str("feature_importances:\n");
        s.push_str("Column_0=1\n");
        s.push('\n');
        s.push_str("parameters:\n[boosting: gbdt]\nend of parameters\n");
        s.push('\n');
        s.push_str("pandas_categorical:null\n");
        s
    }

    #[test]
    fn load_minimal_model() {
        let text = minimal_model_text();
        let m = load(&text).expect("load minimal");
        assert_eq!(m.num_class, 1);
        assert_eq!(m.num_tree_per_iteration, 1);
        assert_eq!(m.max_feature_idx, 1);
        assert_eq!(m.trees.len(), 1);
        assert_eq!(m.feature_names, "Column_0 Column_1");
        assert!(m.trailer.as_ref().unwrap().contains("pandas_categorical:null"));
    }

    #[test]
    fn save_round_trips_byte_exact() {
        let text = minimal_model_text();
        let m = load(&text).expect("load");
        let reemit = save(&m);
        assert_eq!(reemit, text, "save(load(text)) must be byte-identical");
    }

    #[test]
    fn save_with_importance_selects_split_vs_gain() {
        // ADV-07: saved_feature_importance_type selects which importance the
        // feature_importances: block emits. The minimal model's single tree splits
        // feature 0 with split_gain = 1.5.
        let m = load(&minimal_model_text()).expect("load");

        // type 0 (split) — Column_0 split count = 1 (guarded, gain 1.5 > 0).
        let split_text = save_with_importance(&m, 0);
        assert!(
            split_text.contains("\nfeature_importances:\nColumn_0=1\n"),
            "split importance must emit Column_0=1, got:\n{split_text}"
        );

        // type 1 (gain) — Column_0 gain-sum = 1.5 truncated to 1 (static_cast<size_t>).
        let gain_text = save_with_importance(&m, 1);
        assert!(
            gain_text.contains("\nfeature_importances:\nColumn_0=1\n"),
            "gain importance (1.5 -> 1) must emit Column_0=1, got:\n{gain_text}"
        );

        // The default `save` matches type 0 (split).
        assert_eq!(save(&m), split_text);
    }

    #[test]
    fn save_with_gain_truncates_and_guards() {
        // A model whose split gains accumulate to a value > 1 emits the truncated
        // integer; a gain <= 0 split is dropped from BOTH split and gain blocks.
        let mut m = load(&minimal_model_text()).expect("load");
        // Bump max_feature_idx so feature 1 has a slot; add a second tree splitting
        // feature 0 with a large gain and a third with gain 0 (must be guarded out).
        let mut t_big = m.trees[0].clone();
        t_big.split_gain = vec![3.7]; // feature 0, gain 3.7
        let mut t_zero = m.trees[0].clone();
        t_zero.split_gain = vec![0.0]; // guarded out
        m.trees.push(t_big);
        m.trees.push(t_zero);

        // gain: feature 0 = 1.5 + 3.7 = 5.2 -> trunc 5; the gain-0 split contributes 0.
        let gain_text = save_with_importance(&m, 1);
        assert!(
            gain_text.contains("\nfeature_importances:\nColumn_0=5\n"),
            "gain 5.2 -> 5, got:\n{gain_text}"
        );
        // split (guarded): feature 0 has 2 splits with gain>0 (1.5, 3.7); the gain-0
        // split is excluded.
        let split_text = save_with_importance(&m, 0);
        assert!(
            split_text.contains("\nfeature_importances:\nColumn_0=2\n"),
            "guarded split count = 2, got:\n{split_text}"
        );
    }

    #[test]
    fn missing_num_class_is_err() {
        let text = minimal_model_text().replace("num_class=1\n", "");
        let err = load(&text).unwrap_err();
        assert!(matches!(err, ModelError::MalformedModel { .. }));
        assert!(err.to_string().contains("num_class"));
    }

    #[test]
    fn feature_names_count_mismatch_is_err() {
        let text = minimal_model_text().replace("feature_names=Column_0 Column_1\n", "feature_names=Column_0\n");
        let err = load(&text).unwrap_err();
        assert!(err.to_string().contains("feature_names"));
    }

    #[test]
    fn tree_sizes_overflow_is_err() {
        let text = minimal_model_text().replace(
            "tree_sizes=",
            "tree_sizes=999999",
        );
        // Now tree_sizes value is huge (prefixed) -> exceeds region.
        let err = load(&text).unwrap_err();
        assert!(matches!(err, ModelError::MalformedModel { .. }));
    }
}
