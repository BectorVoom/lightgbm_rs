//! JSON model serialization (G2) — hand-emitted, byte-exact against C++
//! `Tree::ToJSON` / `GBDT::DumpModel` (`src/io/tree.cpp`, `gbdt_model_text.cpp`).
//!
//! # Float formatting
//! Every float field in the JSON dump is emitted via C printf `%.17g`
//! ([`crate::format::format_g17`]) — verified empirically against the raw
//! `LGBM_BoosterDumpModel` string from `lightgbm==4.6.0` (`split_gain`,
//! `threshold`, `internal_value`, `internal_weight`, `leaf_value`,
//! `leaf_weight`, `shrinkage`, and `feature_infos` min/max all use 17 sig
//! digits). This DIFFERS from the model-text writer, where `split_gain` /
//! `internal_value` / `internal_weight` / `shrinkage` use 6-digit `%g`. No
//! `serde` dependency (DEC-1): the document is hand-emitted with `format_g17`
//! as the only float path.
//!
//! # Byte layout
//! A node object is `{\n` then one `"key":value,\n` per field (last field has
//! NO trailing comma) then `\n}` — exactly as C++ `Tree::NodeToJSON` streams it.

use crate::ensemble::GbdtModel;
use crate::format::format_g17;
use crate::tree::Tree;

/// C++ `#define kDefaultLeftMask (2)` (`tree.h:21`) — bit1 of `decision_type`.
const DEFAULT_LEFT_MASK: i8 = 2;
/// C++ `#define kCategoricalMask (1)` (`tree.h:20`) — bit0 of `decision_type`.
const CATEGORICAL_MASK: i8 = 1;

/// Serialize one tree's node graph to the C++ `Tree::ToJSON` `"tree_structure"`
/// object (recursive split/leaf nodes), byte-exact vs `lib_lightgbm` 4.6.
///
/// Returns the node object for node 0 (the root), matching what
/// `GBDT::DumpModel` places under each tree's `"tree_structure"` key.
#[must_use]
pub fn tree_to_json(tree: &Tree) -> String {
    node_to_json(tree, 0)
}

/// C++ `Tree::NodeToJSON` (`src/io/tree.cpp`). `index >= 0` is an internal node;
/// `index < 0` encodes leaf `!index` (bitwise-NOT), matching the child arrays.
fn node_to_json(tree: &Tree, index: i32) -> String {
    let mut s = String::new();
    if index >= 0 {
        let node = index as usize;
        s.push_str("{\n");
        s.push_str(&format!("\"split_index\":{index},\n"));
        s.push_str(&format!("\"split_feature\":{},\n", tree.split_feature[node]));
        s.push_str(&format!(
            "\"split_gain\":{},\n",
            format_g17(f64::from(tree.split_gain[node]))
        ));
        if (tree.decision_type[node] & CATEGORICAL_MASK) > 0 {
            let cats = categorical_categories(tree, node);
            let joined: String = cats
                .iter()
                .map(i32::to_string)
                .collect::<Vec<_>>()
                .join("||");
            s.push_str(&format!("\"threshold\":\"{joined}\",\n"));
            s.push_str("\"decision_type\":\"==\",\n");
        } else {
            s.push_str(&format!("\"threshold\":{},\n", format_g17(tree.threshold[node])));
            s.push_str("\"decision_type\":\"<=\",\n");
        }
        let default_left = (tree.decision_type[node] & DEFAULT_LEFT_MASK) > 0;
        s.push_str(&format!("\"default_left\":{default_left},\n"));
        let missing_type = match (tree.decision_type[node] >> 2) & 3 {
            0 => "None",
            1 => "Zero",
            _ => "NaN",
        };
        s.push_str(&format!("\"missing_type\":\"{missing_type}\",\n"));
        s.push_str(&format!(
            "\"internal_value\":{},\n",
            format_g17(tree.internal_value[node])
        ));
        s.push_str(&format!(
            "\"internal_weight\":{},\n",
            format_g17(tree.internal_weight[node])
        ));
        s.push_str(&format!("\"internal_count\":{},\n", tree.internal_count[node]));
        s.push_str(&format!(
            "\"left_child\":{},\n",
            node_to_json(tree, tree.left_child[node])
        ));
        s.push_str(&format!(
            "\"right_child\":{}\n",
            node_to_json(tree, tree.right_child[node])
        ));
        s.push('}');
    } else {
        let leaf = (!index) as usize;
        s.push_str("{\n");
        s.push_str(&format!("\"leaf_index\":{leaf},\n"));
        s.push_str(&format!("\"leaf_value\":{},\n", format_g17(tree.leaf_value[leaf])));
        s.push_str(&format!(
            "\"leaf_weight\":{},\n",
            format_g17(tree.leaf_weight[leaf])
        ));
        s.push_str(&format!("\"leaf_count\":{}\n", tree.leaf_count[leaf]));
        s.push('}');
    }
    s
}

/// The category values of a categorical split, ascending — the set bits of the
/// node's bitset block `cat_threshold[lo..hi]` where
/// `lo = cat_boundaries[cat_idx]`, `hi = cat_boundaries[cat_idx+1]`, and
/// `cat_idx = threshold[node] as i32`. Category value = `(block - lo)*32 + bit`.
/// Mirrors the C++ `Tree::NodeToJSON` categorical branch.
fn categorical_categories(tree: &Tree, node: usize) -> Vec<i32> {
    let cat_idx = tree.threshold[node] as i32 as usize;
    let lo = tree.cat_boundaries[cat_idx] as usize;
    let hi = tree.cat_boundaries[cat_idx + 1] as usize;
    let mut cats = Vec::new();
    for (offset, block_i) in (lo..hi).enumerate() {
        let block = tree.cat_threshold[block_i];
        for bit in 0..32i32 {
            if (block >> bit) & 1 == 1 {
                cats.push(offset as i32 * 32 + bit);
            }
        }
    }
    cats
}

/// Serialize the full ensemble to the C++ `GBDT::DumpModel` document, byte-exact
/// vs the raw `LGBM_BoosterDumpModel` string of `lib_lightgbm` 4.6.
///
/// Top-level field order (verified against the raw C-API string): `name`
/// (constant `"tree"`, the GBDT sub-model name), `version` (constant `"v4"`),
/// `num_class`, `num_tree_per_iteration`, `label_index`, `max_feature_idx`,
/// `objective`, `average_output` (ALWAYS emitted as `true`/`false`),
/// `feature_names`, `monotone_constraints` (ALWAYS emitted, `[]` when none),
/// `feature_infos` (an object; `none` features are OMITTED), `tree_info`.
#[must_use]
pub fn dump_model(model: &GbdtModel) -> String {
    let mut s = String::new();
    s.push_str("{\"name\":\"tree\",\n");
    s.push_str("\"version\":\"v4\",\n");
    s.push_str(&format!("\"num_class\":{},\n", model.num_class));
    s.push_str(&format!(
        "\"num_tree_per_iteration\":{},\n",
        model.num_tree_per_iteration
    ));
    s.push_str(&format!("\"label_index\":{},\n", model.label_index));
    s.push_str(&format!("\"max_feature_idx\":{},\n", model.max_feature_idx));
    s.push_str(&format!(
        "\"objective\":\"{}\",\n",
        model.objective_string.as_deref().unwrap_or("")
    ));
    s.push_str(&format!("\"average_output\":{},\n", model.average_output));

    // feature_names: verbatim space-joined -> JSON string array.
    let names: Vec<&str> = split_ws(&model.feature_names);
    let names_json = names
        .iter()
        .map(|n| format!("\"{n}\""))
        .collect::<Vec<_>>()
        .join(",");
    s.push_str(&format!("\"feature_names\":[{names_json}],\n"));

    // monotone_constraints: verbatim space-joined ints -> comma array ([] when none).
    let mono = match &model.monotone_constraints {
        Some(m) if !m.is_empty() => split_ws(m).join(","),
        _ => String::new(),
    };
    s.push_str(&format!("\"monotone_constraints\":[{mono}],\n"));

    // feature_infos: object keyed by feature name; `none` features are OMITTED.
    let infos: Vec<&str> = split_ws(&model.feature_infos);
    let fi_parts: Vec<String> = names
        .iter()
        .zip(infos.iter())
        .filter_map(|(name, info)| feature_info_json(info).map(|body| format!("\"{name}\":{body}")))
        .collect();
    s.push_str(&format!("\"feature_infos\":{{{}}},\n", fi_parts.join(",")));

    // tree_info: array of {tree_index, num_leaves, num_cat, shrinkage, tree_structure}.
    s.push_str("\"tree_info\":[");
    for (i, tree) in model.trees.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!(
            "{{\"tree_index\":{i},\"num_leaves\":{},\n",
            tree.num_leaves
        ));
        s.push_str(&format!("\"num_cat\":{},\n", tree.num_cat));
        s.push_str(&format!("\"shrinkage\":{},\n", format_g17(tree.shrinkage)));
        s.push_str(&format!("\"tree_structure\":{}\n}}", tree_to_json(tree)));
    }
    s.push(']');

    // feature_importances: split-count (importance_type=0, gain-guarded) per
    // feature, features with 0 count OMITTED, sorted by count DESCENDING (stable
    // — ties keep feature-index order, matching C++ `std::stable_sort`). Emitted
    // as `"name":count` after `tree_info`, then the root close + trailing newline.
    let counts = model.feature_importance_split_count_guarded();
    let mut pairs: Vec<(usize, u64)> = counts
        .iter()
        .enumerate()
        .filter_map(|(i, &c)| (c > 0).then_some((i, c)))
        .collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1)); // Rust sort_by is stable → C++ stable_sort parity
    let fi = pairs
        .iter()
        .map(|(i, c)| format!("\"{}\":{c}", names[*i]))
        .collect::<Vec<_>>()
        .join(",");
    s.push_str(&format!(",\n\n\"feature_importances\":{{{fi}}}\n}}\n"));
    s
}

/// Split a verbatim space-joined model-text value into tokens; empty string → no
/// tokens (NOT one empty token).
fn split_ws(value: &str) -> Vec<&str> {
    if value.is_empty() {
        Vec::new()
    } else {
        value.split(' ').collect()
    }
}

/// One `feature_infos` token → its JSON object body, or `None` when the feature
/// is unused (`"none"` → OMITTED from the JSON object, matching C++). Numeric
/// `[min:max]` → `{"min_value":..,"max_value":..,"values":[]}`; categorical
/// `a:b:c` → `{"min_value":min,"max_value":max,"values":[a,b,c]}`. Floats via
/// `format_g17` (the C-API JSON precision).
fn feature_info_json(token: &str) -> Option<String> {
    if token == "none" {
        return None;
    }
    if let Some(inner) = token.strip_prefix('[').and_then(|t| t.strip_suffix(']')) {
        let (lo, hi) = inner.split_once(':')?;
        let lo_f: f64 = lo.parse().ok()?;
        let hi_f: f64 = hi.parse().ok()?;
        Some(format!(
            "{{\"min_value\":{},\"max_value\":{},\"values\":[]}}",
            format_g17(lo_f),
            format_g17(hi_f)
        ))
    } else {
        let vals: Vec<i64> = token.split(':').filter_map(|v| v.parse().ok()).collect();
        let min = vals.iter().copied().min().unwrap_or(0);
        let max = vals.iter().copied().max().unwrap_or(0);
        let joined = vals
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        Some(format!(
            "{{\"min_value\":{},\"max_value\":{},\"values\":[{joined}]}}",
            format_g17(min as f64),
            format_g17(max as f64)
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::Tree;

    /// A minimal 2-leaf numeric-split tree (mirrors `tree.rs`'s `tiny_tree`):
    /// root split on feature 0 at threshold 1.5, default-left, missing None.
    fn tiny_tree() -> Tree {
        Tree {
            num_leaves: 2,
            num_cat: 0,
            left_child: vec![-1],
            right_child: vec![-2],
            split_feature: vec![0],
            threshold: vec![1.5],
            decision_type: vec![2], // default-left, missing None
            split_gain: vec![0.0],
            leaf_value: vec![10.0, 20.0],
            leaf_weight: vec![1.0, 1.0],
            leaf_count: vec![1, 1],
            internal_value: vec![0.0],
            internal_weight: vec![0.0],
            internal_count: vec![2],
            cat_boundaries: vec![],
            cat_threshold: vec![],
            shrinkage: 1.0,
            is_linear: false,
            linear: None,
            leaf_depth: vec![0, 0],
            leaf_parent: vec![-1, -1],
            split_feature_inner: vec![-1],
            threshold_in_bin: vec![0],
        }
    }

    /// Byte-exact structural test: the emitted `"tree_structure"` object matches
    /// the C++ `Tree::NodeToJSON` layout verified against `lightgbm==4.6.0`
    /// (`{\n` open, one `"key":value,\n` per field, last field no comma, `\n}`,
    /// numeric splits use `"<=" `, floats via `%.17g`).
    #[test]
    fn tree_to_json_numeric_root() {
        let expected = "\
{
\"split_index\":0,
\"split_feature\":0,
\"split_gain\":0,
\"threshold\":1.5,
\"decision_type\":\"<=\",
\"default_left\":true,
\"missing_type\":\"None\",
\"internal_value\":0,
\"internal_weight\":0,
\"internal_count\":2,
\"left_child\":{
\"leaf_index\":0,
\"leaf_value\":10,
\"leaf_weight\":1,
\"leaf_count\":1
},
\"right_child\":{
\"leaf_index\":1,
\"leaf_value\":20,
\"leaf_weight\":1,
\"leaf_count\":1
}
}";
        assert_eq!(tree_to_json(&tiny_tree()), expected);
    }

    fn two_tree_model() -> GbdtModel {
        GbdtModel {
            trees: vec![tiny_tree(), tiny_tree()],
            num_class: 1,
            num_tree_per_iteration: 1,
            label_index: 0,
            max_feature_idx: 0,
            average_output: false,
            objective_string: Some("regression".to_string()),
            feature_names: "Column_0".to_string(),
            feature_infos: "[0.5:1.5]".to_string(),
            monotone_constraints: None,
            trailer: None,
        }
    }

    /// Structural coverage of the ensemble document (SPEC-G2-2): required
    /// top-level keys present in order, `tree_info` has one `tree_structure`
    /// per tree, and the always-emitted `average_output`/`monotone_constraints`
    /// appear even when unset (verified against the raw C-API string).
    #[test]
    fn dump_model_two_tree_binary() {
        let js = dump_model(&two_tree_model());
        assert!(js.starts_with("{\"name\":\"tree\",\n\"version\":\"v4\",\n"));
        for key in [
            "\"num_class\":1",
            "\"num_tree_per_iteration\":1",
            "\"label_index\":0",
            "\"max_feature_idx\":0",
            "\"objective\":\"regression\"",
            "\"average_output\":false",       // ALWAYS emitted
            "\"feature_names\":[\"Column_0\"]",
            "\"monotone_constraints\":[]",     // ALWAYS emitted, [] when none
            "\"feature_infos\":{\"Column_0\":{\"min_value\":0.5,\"max_value\":1.5,\"values\":[]}}",
        ] {
            assert!(js.contains(key), "missing {key} in {js}");
        }
        assert_eq!(js.matches("\"tree_structure\":").count(), 2);
        assert_eq!(js.matches("\"tree_index\":").count(), 2);
        // feature_importances trails tree_info; tiny_tree splits have gain 0 so
        // the gain-guarded split count is empty. Document ends `}\n}\n`.
        assert!(js.contains("],\n\n\"feature_importances\":{"));
        assert!(js.ends_with("}\n}\n"));
    }

    /// A categorical `feature_infos` token becomes `values:[..]`; a `none`
    /// feature is OMITTED from the object entirely (verified against C++).
    #[test]
    fn dump_model_feature_infos_categorical_and_none() {
        let mut m = two_tree_model();
        m.feature_names = "cat num unused".to_string();
        m.feature_infos = "-1:0:2:4 [0.1:0.9] none".to_string();
        let js = dump_model(&m);
        assert!(js.contains(
            "\"cat\":{\"min_value\":-1,\"max_value\":4,\"values\":[-1,0,2,4]}"
        ));
        assert!(js.contains("\"num\":{\"min_value\":0.10000000000000001,\"max_value\":0.90000000000000002,\"values\":[]}"));
        // The `none` feature's NAME still appears in feature_names (C++ lists
        // every feature there), but it is OMITTED from the feature_infos object.
        assert!(js.contains("\"feature_names\":[\"cat\",\"num\",\"unused\"]"));
        assert!(!js.contains("\"unused\":{"), "none feature must be omitted from feature_infos: {js}");
    }

    /// `%.17g` float fidelity: a threshold that is NOT shortest-round-trip must
    /// emit its 17-significant-digit form, never `0.1`.
    #[test]
    fn tree_to_json_uses_g17_floats() {
        let mut t = tiny_tree();
        t.threshold = vec![0.1];
        let js = tree_to_json(&t);
        assert!(
            js.contains("\"threshold\":0.10000000000000001,"),
            "threshold must use %.17g, got: {js}"
        );
    }
}
