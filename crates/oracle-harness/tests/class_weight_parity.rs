//! ORACLE parity for the class-weight parameters `is_unbalance` and
//! `scale_pos_weight`.
//!
//! # The gap this closes
//!
//! Both parameters were parsed by `lgbm-core` and range-validated, then DROPPED:
//! `lgbm_objective::Binary::get_gradients` hard-coded `label_weight = 1.0` for both
//! classes, so `is_unbalance=true` and `scale_pos_weight=5` trained EXACTLY the
//! balanced model. Nothing in the suite noticed, because every committed corpus used
//! the balanced default — the same class of defect as the `feature_fraction` gap.
//!
//! C++ `BinaryLogloss::Init` (`binary_objective.hpp:85-100`) derives
//! `label_weights_[2]` (indexed by `is_pos`) and `GetGradients` (`:105-136`)
//! multiplies BOTH the gradient and the hessian by `label_weights_[is_pos]`.
//! `MulticlassOVA` (`multiclass_objective.hpp:190-193`) builds one `BinaryLogloss`
//! per class from the same Config, so both parameters apply per one-vs-all split
//! with that class's own counts — which is why the `multiclassova` cells are here
//! and not just the `binary` ones.
//!
//! The golden's labels are deliberately IMBALANCED (~20% positive); at a 50/50 split
//! `is_unbalance` derives `[1.0, 1.0]` and the cell would pass against the unwired
//! code.
//!
//! Regenerate: `.venv/bin/python crates/oracle-harness/tests/fixtures/class_weight/capture.py`

use std::collections::HashMap;
use std::path::PathBuf;

use lgbm::{train_raw, Config, RawCorpus};
use oracle_harness::comparator::{compare_within, ORACLE_TOL};

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/class_weight/class_weight_golden.json")
}

struct Golden {
    features: Vec<f64>,
    labels: HashMap<String, Vec<f32>>,
    num_rows: usize,
    num_cols: usize,
    num_iterations: i32,
    cells: Vec<Cell>,
}

struct Cell {
    objective: String,
    num_class: i32,
    is_unbalance: bool,
    scale_pos_weight: f64,
    labels: String,
    pred: Vec<f64>,
}

impl Cell {
    fn label(&self) -> String {
        format!(
            "{} is_unbalance={} scale_pos_weight={}",
            self.objective, self.is_unbalance, self.scale_pos_weight
        )
    }
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

fn bool_field(src: &str, name: &str) -> Option<bool> {
    let at = src.find(name)? + name.len();
    let rest = src[at..].trim_start().trim_start_matches(':').trim_start();
    if rest.starts_with("true") {
        Some(true)
    } else if rest.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

fn string_field(src: &str, name: &str) -> Option<String> {
    let at = src.find(name)? + name.len();
    let rest = src[at..].trim_start().trim_start_matches(':').trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn load_golden() -> Option<Golden> {
    let raw = std::fs::read_to_string(golden_path()).ok()?;
    let features = hex_array(&raw, "\"features_bits\"")?;
    let mut labels = HashMap::new();
    let labels_at = raw.find("\"labels_bits\"")?;
    for name in ["binary", "multi"] {
        let needle = format!("\"{name}\"");
        let rel = raw[labels_at..].find(&needle)?;
        let v = hex_array(&raw[labels_at + rel..], &needle)?;
        labels.insert(name.to_string(), v.into_iter().map(|x| x as f32).collect());
    }
    let num_rows = int_field(&raw, "\"num_rows\"")? as usize;
    let num_cols = int_field(&raw, "\"num_cols\"")? as usize;
    let num_iterations = int_field(&raw, "\"num_iterations\"")? as i32;

    // Each cell object is located by its `"is_unbalance"` field; the enclosing
    // object runs from the preceding `{` to its matching brace.
    let cells_at = raw.find("\"cells\"")?;
    let body = &raw[cells_at..];
    let mut cells = Vec::new();
    let mut idx = 0usize;
    while let Some(rel) = body[idx..].find("\"is_unbalance\"") {
        let start = idx + rel;
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
            objective: string_field(cell, "\"objective\"")?,
            num_class: int_field(cell, "\"num_class\"")? as i32,
            is_unbalance: bool_field(cell, "\"is_unbalance\"")?,
            scale_pos_weight: float_field(cell, "\"scale_pos_weight\"")?,
            labels: string_field(cell, "\"labels\"")?,
            pred: hex_array(cell, "\"pred_bits\"").unwrap_or_default(),
        });
        idx = obj_end + 1;
    }
    Some(Golden { features, labels, num_rows, num_cols, num_iterations, cells })
}

fn config_for(g: &Golden, cell: &Cell) -> Config {
    let mut params: HashMap<String, String> = [
        ("objective", cell.objective.clone()),
        ("num_class", cell.num_class.to_string()),
        ("num_iterations", g.num_iterations.to_string()),
        ("learning_rate", "0.1".to_string()),
        ("num_leaves", "8".to_string()),
        ("min_data_in_leaf", "5".to_string()),
        ("max_bin", "63".to_string()),
        ("seed", "1".to_string()),
        ("deterministic", "true".to_string()),
        ("force_row_wise", "true".to_string()),
        ("num_threads", "1".to_string()),
        ("scale_pos_weight", cell.scale_pos_weight.to_string()),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    if cell.is_unbalance {
        params.insert("is_unbalance".to_string(), "true".to_string());
    }
    Config::from_params(&params).expect("config builds")
}

fn corpus(g: &Golden, cell: &Cell, config: &Config) -> RawCorpus {
    let columns: Vec<Vec<f64>> = (0..g.num_cols)
        .map(|c| (0..g.num_rows).map(|r| g.features[r * g.num_cols + c]).collect())
        .collect();
    let mut raw = RawCorpus::from_columns(columns, g.labels[&cell.labels].clone());
    raw.config = config.clone();
    raw
}

#[test]
fn class_weight_predictions_match_the_cpp_oracle() {
    let Some(g) = load_golden() else {
        eprintln!("SKIP: class_weight golden absent — run fixtures/class_weight/capture.py");
        return;
    };
    assert!(!g.cells.is_empty(), "golden present but parsed to zero cells");

    for cell in &g.cells {
        let config = config_for(&g, cell);
        let c = corpus(&g, cell, &config);
        let booster = train_raw(&config, &c).unwrap_or_else(|e| panic!("{}: train failed: {e}", cell.label()));
        // The C++ capture flattens `predict(raw_score=True)` — shape (N,) for binary
        // and (N, num_class) ROW-major for multiclassova.
        let rows = c.to_rows();
        let mut rust: Vec<f32> = Vec::with_capacity(cell.pred.len());
        for row in &rows {
            rust.extend(booster.predict_row_raw(row, -1).iter().map(|&v| v as f32));
        }
        let cpp: Vec<f32> = cell.pred.iter().map(|&v| v as f32).collect();
        assert_eq!(rust.len(), cpp.len(), "{}: prediction count", cell.label());
        compare_within(&rust, &cpp, ORACLE_TOL)
            .unwrap_or_else(|m| panic!("{}: raw predictions diverge from C++: {m:?}", cell.label()));
    }
}

/// The cells must actually EXERCISE the weighting — otherwise the test above would
/// pass against the unwired `label_weight = 1.0` code it exists to catch. Each
/// non-control cell's C++ predictions must differ from its control's.
#[test]
fn the_weighted_cells_differ_from_their_unweighted_control() {
    let Some(g) = load_golden() else {
        eprintln!("SKIP: class_weight golden absent");
        return;
    };
    for objective in ["binary", "multiclassova"] {
        let control = g
            .cells
            .iter()
            .find(|c| c.objective == objective && !c.is_unbalance && c.scale_pos_weight == 1.0)
            .unwrap_or_else(|| panic!("no control cell for {objective}"));
        let weighted: Vec<&Cell> = g
            .cells
            .iter()
            .filter(|c| {
                c.objective == objective && (c.is_unbalance || c.scale_pos_weight != 1.0)
            })
            .collect();
        assert!(!weighted.is_empty(), "no weighted cells for {objective}");
        for cell in weighted {
            let differs = cell
                .pred
                .iter()
                .zip(&control.pred)
                .any(|(a, b)| (a - b).abs() > 1e-6);
            assert!(
                differs,
                "{}: C++ predictions are identical to the unweighted control — \
                 this cell has no teeth",
                cell.label()
            );
        }
    }
}

/// C++ `Log::Fatal`s on "Cannot set is_unbalance and scale_pos_weight at the same
/// time" (`binary_objective.hpp:31-33`). The port must reject it too — as a typed
/// error, never a silently-applied double weighting.
#[test]
fn is_unbalance_combined_with_scale_pos_weight_is_a_typed_error() {
    let Some(g) = load_golden() else {
        eprintln!("SKIP: class_weight golden absent");
        return;
    };
    let cell = Cell {
        objective: "binary".to_string(),
        num_class: 1,
        is_unbalance: true,
        scale_pos_weight: 3.0,
        labels: "binary".to_string(),
        pred: Vec::new(),
    };
    let config = config_for(&g, &cell);
    let c = corpus(&g, &cell, &config);
    let err = train_raw(&config, &c)
        .err()
        .expect("is_unbalance + scale_pos_weight must be rejected");
    assert!(
        err.to_string().contains("is_unbalance"),
        "expected the C++ conflict message, got: {err}"
    );
}
