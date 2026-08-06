//! ORACLE parity for `path_smooth` and `max_delta_step` — the C++ `USE_SMOOTHING`
//! and `USE_MAX_OUTPUT` template axes of `FeatureHistogram`.
//!
//! # The gap this closes
//!
//! Both parameters were parsed and validated, then REJECTED at the split kernels
//! with a typed `ComputeError` ("only the default 0.0 path is transcribed"). Honest,
//! but they simply did not work. They are now implemented across every split path —
//! the shared `#[cube] split_scan_body`, the native host scan, the categorical
//! finder, the monotone finder, the extra-trees randomized finder and the
//! forced-split threshold gatherer.
//!
//! # What each axis does (feature_histogram.hpp:715-813)
//!
//! ```text
//! ret = -ThresholdL1(g, l1) / (h + l2)                      // base
//! if USE_MAX_OUTPUT && max_delta_step > 0 && |ret| > max_delta_step:
//!     ret = Sign(ret) * max_delta_step                      // clamp
//! if USE_SMOOTHING:
//!     ret = ret * n/ps / (n/ps + 1) + parent_output / (n/ps + 1)   // blend
//! ```
//!
//! and EITHER axis also switches the gain FORM: `GetLeafGain` stops returning the
//! closed form `sg²/(h+λ)` and returns `GetLeafGainGivenOutput` evaluated at the
//! computed output instead. The two are equal in exact arithmetic and differ in
//! floating point, so the form switch is observable even when the clamp never fires.
//!
//! # Why the cells look like this
//!
//! `path_smooth` CHAINS through depth — a leaf's `parent_output` is the (already
//! smoothed) output its parent split assigned it — so an error at depth 1 compounds
//! downward and a stump would not see it. The cells use `num_leaves=16` over 8
//! trees and assert the PER-TREE LEAF VALUES, which is where the blend lands
//! directly, not just the summed predictions.
//!
//! Regenerate: `.venv/bin/python crates/oracle-harness/tests/fixtures/path_smooth/capture.py`

use std::collections::HashMap;
use std::path::PathBuf;

use lgbm::{train_raw, Config, RawCorpus};
use oracle_harness::comparator::{compare_within, ORACLE_TOL};

/// The ONE cell that does not reach bit-parity, with the reason.
///
/// `max_delta_step = 0.05` on this corpus is mathematically DEGENERATE: it binds so
/// tightly that at many candidate thresholds BOTH children clamp to the same output
/// `-max_delta_step`, and then
///
/// ```text
/// GetLeafGain(left) + GetLeafGain(right) == GetLeafGain(whole leaf) == gain_shift
/// ```
///
/// holds EXACTLY in real arithmetic. The split's net gain is therefore exactly zero,
/// and `is_splittable_` turns on the strict `current_gain > min_gain_shift`
/// comparison of two differently-associated floating-point sums —
/// `(2·gₗ·o + hₗ·o²) + (2·gᵣ·o + hᵣ·o²)` versus `2·G·o + H·o²`. This port lands ONE
/// ULP above (`net = 1.78e-15` on a gain of `8.85`); the C++ build lands exactly at
/// zero. That flips one feature's splittability at a depth-1 leaf, and C++
/// propagates non-splittability to both children (`serial_tree_learner.cpp:390-395`),
/// so the feature disappears from the whole subtree and the models diverge visibly.
///
/// This is a boundary artifact, not a formula error — DEMONSTRATED by perturbing the
/// parameter by 2 parts per million:
///
/// | `max_delta_step` | C++ tree-0 splits | C++ 4th gain |
/// |---|---|---|
/// | 0.0499999  | 3 1 1 5 … | 0.533999 |
/// | 0.05       | 3 1 1 5 … | 0.534    |
/// | 0.0500001  | 3 1 1 **3** … | **1.334** |
///
/// At `0.0500001` C++ produces EXACTLY this port's answer (feature 3, gain 1.334),
/// so both engines agree on the arithmetic and disagree only about which side of the
/// tie they land on. The residual ULP is consistent with C++ being compiled with
/// `-ffp-contract=on` (clang's default), which may fuse `a*b + c*d` into an FMA that
/// Rust — which never auto-contracts — evaluates as separate rounded operations.
///
/// Kept as a cell rather than deleted so the divergence stays measured: the sweep
/// below asserts it is the ONLY one.
const KNOWN_ULP_BOUNDARY: &[(f64, f64, f64, i32)] = &[(0.0, 0.05, 0.0, 16)];

fn is_known_boundary(cell: &Cell) -> bool {
    KNOWN_ULP_BOUNDARY.iter().any(|&(ps, mds, l1, nl)| {
        cell.path_smooth == ps
            && cell.max_delta_step == mds
            && cell.lambda_l1 == l1
            && cell.num_leaves == nl
    })
}

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/path_smooth/path_smooth_golden.json")
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
    path_smooth: f64,
    max_delta_step: f64,
    lambda_l1: f64,
    num_leaves: i32,
    pred: Vec<f64>,
    /// Per-tree `leaf_value=` lists, in tree order.
    leaf_values: Vec<Vec<f64>>,
}

impl Cell {
    fn label(&self) -> String {
        format!(
            "path_smooth={} max_delta_step={} lambda_l1={} num_leaves={}",
            self.path_smooth, self.max_delta_step, self.lambda_l1, self.num_leaves
        )
    }
    fn is_control(&self) -> bool {
        self.path_smooth == 0.0 && self.max_delta_step == 0.0 && self.lambda_l1 == 0.0
            && self.num_leaves == 16
    }
}

// --- minimal JSON reading (same fixed-schema approach as the sibling fixtures) ---

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
        .find(|c: char| !c.is_ascii_digit() && c != '-' && c != '.' && c != 'e' && c != '+')
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Parse `"leaf_values": [[..], [..]]` into per-tree f64 lists.
fn nested_float_lists(src: &str, name: &str) -> Vec<Vec<f64>> {
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
    let mut out = Vec::new();
    let mut cur: Option<String> = None;
    for ch in src[open + 1..end].chars() {
        match ch {
            '[' => cur = Some(String::new()),
            ']' => {
                if let Some(buf) = cur.take() {
                    out.push(
                        buf.split(',')
                            .filter_map(|t| t.trim().parse::<f64>().ok())
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

    let cells_at = raw.find("\"cells\"")?;
    let body = &raw[cells_at..];
    let mut cells = Vec::new();
    let mut idx = 0usize;
    while let Some(rel) = body[idx..].find("\"path_smooth\"") {
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
            path_smooth: float_field(cell, "\"path_smooth\"")?,
            max_delta_step: float_field(cell, "\"max_delta_step\"")?,
            lambda_l1: float_field(cell, "\"lambda_l1\"")?,
            num_leaves: int_field(cell, "\"num_leaves\"")? as i32,
            pred: hex_array(cell, "\"pred_bits\"").unwrap_or_default(),
            leaf_values: nested_float_lists(cell, "\"leaf_values\""),
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
        ("num_leaves", cell.num_leaves.to_string()),
        ("min_data_in_leaf", "5".to_string()),
        ("max_bin", "63".to_string()),
        ("seed", "1".to_string()),
        ("deterministic", "true".to_string()),
        ("force_row_wise", "true".to_string()),
        ("num_threads", "1".to_string()),
        ("path_smooth", cell.path_smooth.to_string()),
        ("max_delta_step", cell.max_delta_step.to_string()),
        ("lambda_l1", cell.lambda_l1.to_string()),
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

/// The per-tree `leaf_value=` lists of a model text, in tree order.
fn leaf_values(model: &str) -> Vec<Vec<f64>> {
    model
        .split("Tree=")
        .skip(1)
        .map(|chunk| {
            chunk
                .lines()
                .filter_map(|l| l.strip_prefix("leaf_value="))
                .flat_map(|v| v.split_whitespace())
                .filter_map(|t| t.parse::<f64>().ok())
                .collect()
        })
        .collect()
}

#[test]
fn path_smooth_and_max_delta_step_predictions_match_the_cpp_oracle() {
    let Some(g) = load_golden() else {
        eprintln!("SKIP: path_smooth golden absent — run fixtures/path_smooth/capture.py");
        return;
    };
    assert!(!g.cells.is_empty(), "golden present but parsed to zero cells");

    // Report EVERY diverging cell, not just the first — with 13 cells across two
    // independent axes, "which cells fail" is the whole diagnostic.
    let mut failures: Vec<String> = Vec::new();
    let mut boundary_diverged = 0usize;
    for cell in &g.cells {
        let config = config_for(&g, cell);
        let c = corpus(&g, &config);
        let booster = train_raw(&config, &c)
            .unwrap_or_else(|e| panic!("{}: train failed: {e}", cell.label()));
        let rows = c.to_rows();
        let rust: Vec<f32> = rows
            .iter()
            .map(|r| booster.predict_row_raw(r, -1)[0] as f32)
            .collect();
        let cpp: Vec<f32> = cell.pred.iter().map(|&v| v as f32).collect();
        match (compare_within(&rust, &cpp, ORACLE_TOL), is_known_boundary(cell)) {
            (Ok(()), false) => {}
            (Ok(()), true) => panic!(
                "{} is listed in KNOWN_ULP_BOUNDARY but now matches — remove it \
                 and assert real parity",
                cell.label()
            ),
            (Err(_), true) => boundary_diverged += 1,
            (Err(m), false) => failures.push(format!("{}: {m:?}", cell.label())),
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} cells diverge from the C++ oracle:\n  {}",
        failures.len(),
        g.cells.len(),
        failures.join("\n  ")
    );
    assert_eq!(
        boundary_diverged,
        KNOWN_ULP_BOUNDARY.len(),
        "the documented ULP-boundary cell(s) must still be the ONLY divergence"
    );
}

/// The LEAF VALUES are where the clamp and the blend land directly — predictions
/// only see their sum, so a compensating error could hide there.
#[test]
fn per_tree_leaf_values_match_the_cpp_oracle() {
    let Some(g) = load_golden() else {
        eprintln!("SKIP: path_smooth golden absent");
        return;
    };
    for cell in &g.cells {
        if is_known_boundary(cell) {
            continue; // see KNOWN_ULP_BOUNDARY
        }
        let config = config_for(&g, cell);
        let c = corpus(&g, &config);
        let booster = train_raw(&config, &c)
            .unwrap_or_else(|e| panic!("{}: train failed: {e}", cell.label()));
        let rust = leaf_values(&booster.model_to_string());
        assert_eq!(
            rust.len(),
            cell.leaf_values.len(),
            "{}: tree count",
            cell.label()
        );
        for (t, (r, c_)) in rust.iter().zip(&cell.leaf_values).enumerate() {
            assert_eq!(r.len(), c_.len(), "{}: tree {t} leaf count", cell.label());
            let rf: Vec<f32> = r.iter().map(|&v| v as f32).collect();
            let cf: Vec<f32> = c_.iter().map(|&v| v as f32).collect();
            compare_within(&rf, &cf, ORACLE_TOL).unwrap_or_else(|m| {
                panic!("{}: tree {t} leaf values diverge: {m:?}", cell.label())
            });
        }
    }
}

/// Teeth: every cell that turns an axis ON must produce a C++ model DIFFERENT from
/// the all-defaults control. Without this the parity test above could pass against
/// an implementation that ignored both parameters.
///
/// The one deliberate exception is `max_delta_step = 1e3`, which is larger than
/// every leaf output on this corpus so the clamp never fires. That cell exercises
/// only the gain-FORM switch (closed form → given-output form), which happens to be
/// bit-identical here; it is retained as a REGRESSION guard — the form switch must
/// not perturb a model where the clamp is inactive — and is explicitly not claimed
/// to have teeth.
#[test]
fn every_active_cell_differs_from_the_all_defaults_control() {
    let Some(g) = load_golden() else {
        eprintln!("SKIP: path_smooth golden absent");
        return;
    };
    let control = g
        .cells
        .iter()
        .find(|c| c.is_control())
        .expect("golden carries an all-defaults control cell");
    let mut active = 0usize;
    for cell in &g.cells {
        if cell.is_control() {
            continue;
        }
        // The inactive-clamp cell: assert it matches the control instead.
        if cell.max_delta_step >= 1e3 && cell.path_smooth == 0.0 && cell.lambda_l1 == 0.0 {
            let same = cell
                .pred
                .iter()
                .zip(&control.pred)
                .all(|(a, b)| (a - b).abs() <= f64::from(ORACLE_TOL));
            assert!(
                same,
                "max_delta_step={} never binds on this corpus, so C++ must reproduce \
                 the control model",
                cell.max_delta_step
            );
            continue;
        }
        let differs = cell
            .pred
            .iter()
            .zip(&control.pred)
            .any(|(a, b)| (a - b).abs() > f64::from(ORACLE_TOL));
        assert!(
            differs,
            "{}: C++ predictions are identical to the all-defaults control — \
             this cell has no teeth",
            cell.label()
        );
        active += 1;
    }
    assert!(active >= 10, "expected a broad active-cell sweep, got {active}");
}

/// Neither parameter may be REJECTED any more. This pins the removal of the
/// "only the default 0.0 path is transcribed" `ComputeError` so it cannot come back
/// silently.
#[test]
fn neither_parameter_is_rejected_by_the_split_kernels() {
    let Some(g) = load_golden() else {
        eprintln!("SKIP: path_smooth golden absent");
        return;
    };
    for (ps, mds) in [(1.0, 0.0), (0.0, 0.5), (1.0, 0.5)] {
        let cell = Cell {
            path_smooth: ps,
            max_delta_step: mds,
            lambda_l1: 0.0,
            num_leaves: 16,
            pred: Vec::new(),
            leaf_values: Vec::new(),
        };
        let config = config_for(&g, &cell);
        let c = corpus(&g, &config);
        train_raw(&config, &c).unwrap_or_else(|e| {
            panic!("path_smooth={ps} max_delta_step={mds} must train, got: {e}")
        });
    }
}
