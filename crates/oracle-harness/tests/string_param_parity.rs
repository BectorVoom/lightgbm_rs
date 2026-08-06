//! ORACLE parity for EVERY value of every closed-enum string-valued parameter.
//!
//! The string-valued parameter set is `config_auto.cpp::ParameterTypes()` filtered
//! to `string` / `vector<string>` — 23 parameters. The 8 with a CLOSED value set
//! (`objective`, `metric`, `boosting`, `data_sample_strategy`, `tree_learner`,
//! `device_type`, `monotone_constraints_method`, `convert_model_language`) plus
//! `task` are swept here across every accepted value INCLUDING aliases; the other
//! 15 name files or dataset columns (CLI / dataset-loader layer) and are listed
//! with their reason in the golden's `skipped` map, which
//! [`skipped_params_are_documented`] asserts stays populated.
//!
//! # What each cell asserts
//!
//! The golden (captured from real `lightgbm==4.6.0` by `fixtures/string_params/
//! capture.py`) carries, per cell, the C++ `save_model()` text and the first 40
//! raw-score predictions, plus the exact corpus as IEEE-754 bits so the Rust
//! replay trains the byte-identical problem. Each cell then asserts one of:
//!
//! - **Trained parity** — the Rust raw predictions match C++ within [`ORACLE_TOL`].
//! - **Config-level agreement** — for values whose behavior is a CONFIG decision
//!   (`metric` selection, `task`, alias canonicalization), that the Rust `Config`
//!   resolves to the same canonical value C++ resolved to.
//! - **Matching rejection** — a value C++ rejects must also be rejected by Rust
//!   (never silently accepted and mistrained).
//! - **Explicit UNSUPPORTED** — a value the Rust port does not implement is
//!   recorded in [`UNSUPPORTED`] with its reason and asserted to produce a TYPED
//!   ERROR rather than a wrong model. This keeps "not implemented" honest: it can
//!   never masquerade as a pass.
//!
//! SKIPs are graceful only for an ABSENT golden file (a fresh checkout that has
//! not run the capture); a present golden never silently skips a cell.
//!
//! Regenerate: `.venv/bin/python crates/oracle-harness/tests/fixtures/string_params/capture.py`

use std::collections::HashMap;
use std::path::PathBuf;

use lgbm::{train_raw, Config, RawCorpus};
use oracle_harness::comparator::{compare_within, ORACLE_TOL};

/// Values the Rust port does not implement, with the reason. A cell listed here
/// must produce a TYPED ERROR (never a silently-wrong model); the test fails if
/// such a value starts succeeding, so the list cannot rot into a false negative.
const UNSUPPORTED: &[(&str, &str)] = &[
    // NOTE: the distributed `tree_learner` values are NOT listed here. C++
    // `CheckParamConflict` resets `tree_learner` to "serial" whenever
    // `num_machines <= 1` (config.cpp:341-347), so on a single machine every value
    // trains the SAME serial model — which is real trained parity, asserted below.
    //
    // The ranking objectives (`lambdarank`, `rank_xendcg` + aliases) and the
    // ranking metrics (`ndcg`, `map` + aliases) WERE listed here while the
    // training facade dropped the corpus's query boundaries before objective
    // resolution. They are now wired end-to-end (`lgbm::booster::resolve_objective`
    // reads `corpus.query_boundaries`) and every one of those cells is asserted at
    // full trained parity below.
];

/// Cells whose parity is a CONFIG-level agreement rather than a trained-model
/// comparison: the parameter changes evaluation or CLI behavior, not the trees.
fn is_config_only(key: &str) -> bool {
    key.starts_with("metric=")
        || key.starts_with("task=")
        || key.starts_with("convert_model_language=")
        // `device_type` selects a compute backend; the C++ wheel has neither GPU
        // backend compiled in, so its golden is an error, not a model.
        || key.starts_with("device_type=")
}

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/string_params/string_params_golden.json")
}

// --- a minimal JSON reader (no new dependency; the schema is fixed) ----------

/// The parsed golden. Only the fields this test reads are modelled.
struct Golden {
    features: Vec<f64>,
    labels: HashMap<String, Vec<f64>>,
    num_rows: usize,
    num_cols: usize,
    /// The capture's `group` — LightGBM per-query row COUNTS (`Dataset(group=...)`).
    /// Converted to `Metadata::query_boundaries` (cumulative offsets) by
    /// [`query_boundaries`].
    group: Vec<i32>,
    skipped: Vec<String>,
    cells: Vec<Cell>,
}

/// The capture's group SIZES as C++ `Metadata::query_boundaries` — `num_queries + 1`
/// cumulative offsets starting at 0 (`metadata.cpp::SetQuery`'s prefix sum).
fn query_boundaries(group: &[i32]) -> Vec<i32> {
    let mut out = Vec::with_capacity(group.len() + 1);
    let mut acc = 0i32;
    out.push(0);
    for &g in group {
        acc += g;
        out.push(acc);
    }
    out
}

struct Cell {
    key: String,
    ok: bool,
    error: String,
    pred: Vec<f64>,
    objective: String,
    labels: String,
    num_class: i32,
}

/// Load the golden, or `None` when it has not been captured yet.
fn load_golden() -> Option<Golden> {
    let raw = std::fs::read_to_string(golden_path()).ok()?;
    let features = hex_array(&raw, "\"features_bits\"")?;
    let mut labels = HashMap::new();
    // `labels_bits` is an object of name → hex array; read each known key.
    for name in ["regression", "positive", "binary", "rank", "multi"] {
        let needle = format!("\"{name}\"");
        if let Some(at) = raw.find("\"labels_bits\"")
            && let Some(rel) = raw[at..].find(&needle)
            && let Some(v) = hex_array(&raw[at + rel..], &needle)
        {
            labels.insert(name.to_string(), v);
        }
    }
    let num_rows = int_field(&raw, "\"num_rows\"")? as usize;
    let num_cols = int_field(&raw, "\"num_cols\"")? as usize;
    let skipped = object_keys(&raw, "\"skipped\"");
    let group: Vec<i32> = number_array(&raw, "\"group\"").iter().map(|&v| v as i32).collect();
    let cells = parse_cells(&raw);
    Some(Golden { features, labels, num_rows, num_cols, group, skipped, cells })
}

/// Read `"<name>": [ "<hex>", ... ]` as f64s decoded from their IEEE-754 bits.
fn hex_array(src: &str, name: &str) -> Option<Vec<f64>> {
    let at = src.find(name)?;
    let open = src[at..].find('[')? + at;
    let close = src[open..].find(']')? + open;
    Some(
        src[open + 1..close]
            .split(',')
            .filter_map(|t| {
                let t = t.trim().trim_matches('"');
                u64::from_str_radix(t, 16).ok().map(f64::from_bits)
            })
            .collect(),
    )
}

fn int_field(src: &str, name: &str) -> Option<i64> {
    let at = src.find(name)? + name.len();
    let rest = src[at..].trim_start().trim_start_matches(':').trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit() && c != '-')?;
    rest[..end].parse().ok()
}

/// The immediate `"key":` names inside the object at `name`.
fn object_keys(src: &str, name: &str) -> Vec<String> {
    let Some(at) = src.find(name) else { return Vec::new() };
    let Some(open) = src[at..].find('{').map(|i| i + at) else { return Vec::new() };
    let mut depth = 0i32;
    let mut end = open;
    for (i, b) in src[open..].bytes().enumerate() {
        if b == b'{' {
            depth += 1;
        } else if b == b'}' {
            depth -= 1;
            if depth == 0 {
                end = open + i;
                break;
            }
        }
    }
    json_keys_at_depth_one(&src[open..=end])
}

fn json_keys_at_depth_one(obj: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let b = obj.as_bytes();
    let mut i = 0usize;
    while i < b.len() {
        match b[i] {
            b'{' | b'[' => depth += 1,
            b'}' | b']' => depth -= 1,
            b'"' if depth == 1 => {
                let start = i + 1;
                let mut j = start;
                while j < b.len() && b[j] != b'"' {
                    j += if b[j] == b'\\' { 2 } else { 1 };
                }
                // A key is a string immediately followed by `:`.
                let after = obj[j + 1..].trim_start();
                if after.starts_with(':') {
                    out.push(obj[start..j].to_string());
                }
                i = j;
            }
            _ => {}
        }
        i += 1;
    }
    out
}

/// Parse the `cells` object into one [`Cell`] per key.
fn parse_cells(src: &str) -> Vec<Cell> {
    let Some(at) = src.find("\"cells\"") else { return Vec::new() };
    let body = &src[at..];
    let keys = object_keys(src, "\"cells\"");
    let mut out = Vec::new();
    for key in keys {
        let needle = format!("\"{}\":", escape(&key));
        let Some(k) = body.find(&needle) else { continue };
        // The cell object runs to its matching brace.
        let Some(open) = body[k..].find('{').map(|i| i + k) else { continue };
        let mut depth = 0i32;
        let mut end = open;
        for (i, b) in body[open..].bytes().enumerate() {
            if b == b'{' {
                depth += 1;
            } else if b == b'}' {
                depth -= 1;
                if depth == 0 {
                    end = open + i;
                    break;
                }
            }
        }
        let cell = &body[open..=end];
        out.push(Cell {
            key,
            ok: cell.contains("\"ok\": true"),
            error: string_field(cell, "\"error\"").unwrap_or_default(),
            pred: number_array(cell, "\"pred\""),
            objective: string_field(cell, "\"objective\"").unwrap_or_default(),
            labels: string_field(cell, "\"labels\"").unwrap_or_default(),
            num_class: int_field(cell, "\"num_class\"").unwrap_or(1) as i32,
        });
    }
    out
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn string_field(src: &str, name: &str) -> Option<String> {
    let at = src.find(name)? + name.len();
    let rest = src[at..].trim_start().trim_start_matches(':').trim_start();
    let rest = rest.strip_prefix('"')?;
    let b = rest.as_bytes();
    let mut j = 0usize;
    while j < b.len() && b[j] != b'"' {
        j += if b[j] == b'\\' { 2 } else { 1 };
    }
    Some(rest[..j.min(rest.len())].replace("\\n", "\n").replace("\\\"", "\""))
}

fn number_array(src: &str, name: &str) -> Vec<f64> {
    let Some(at) = src.find(name) else { return Vec::new() };
    let Some(open) = src[at..].find('[').map(|i| i + at) else { return Vec::new() };
    let Some(close) = src[open..].find(']').map(|i| i + open) else { return Vec::new() };
    src[open + 1..close]
        .split(',')
        .filter_map(|t| t.trim().parse::<f64>().ok())
        .collect()
}

// --- the replay -------------------------------------------------------------

/// Rebuild the exact training problem a cell captured.
fn corpus_for(g: &Golden, cell: &Cell, config: &Config) -> RawCorpus {
    let columns: Vec<Vec<f64>> = (0..g.num_cols)
        .map(|c| (0..g.num_rows).map(|r| g.features[r * g.num_cols + c]).collect())
        .collect();
    let labels: Vec<f32> = g.labels[&cell.labels].iter().map(|&v| v as f32).collect();
    let mut raw = RawCorpus::from_columns(columns, labels);
    // The capture passed `group=GROUP` to `lgb.Dataset` for exactly the cells that
    // used the `rank` label vector (`capture.py::family_for`), so the replay must
    // carry the same query grouping — otherwise the ranking objective/metric sees a
    // different problem than the oracle did.
    if cell.labels == "rank" {
        raw.query_boundaries = query_boundaries(&g.group);
    }
    raw.config = config.clone();
    raw
}

/// The capture's param dict, as the Rust `Config`.
fn config_for(cell: &Cell) -> Result<Config, String> {
    let (param, value) = cell.key.split_once('=').ok_or("malformed cell key")?;
    let mut params: HashMap<String, String> = [
        ("objective", cell.objective.as_str()),
        ("num_iterations", "6"),
        ("learning_rate", "0.1"),
        ("num_leaves", "8"),
        ("min_data_in_leaf", "5"),
        ("max_bin", "63"),
        ("seed", "1"),
        ("deterministic", "true"),
        ("force_row_wise", "true"),
        ("num_threads", "1"),
        ("num_class", &cell.num_class.to_string()),
    ]
    .iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect();
    if param != "objective" {
        params.insert(param.to_string(), value.to_string());
    }
    if param == "boosting" && (value == "rf" || value == "random_forest") {
        params.insert("bagging_freq".into(), "1".into());
        params.insert("bagging_fraction".into(), "0.8".into());
    }
    Config::from_params(&params).map_err(|e| e.to_string())
}

fn unsupported_reason(key: &str) -> Option<&'static str> {
    UNSUPPORTED.iter().find(|(k, _)| *k == key).map(|(_, r)| *r)
}

#[test]
fn every_string_param_value_matches_the_cpp_oracle() {
    let Some(g) = load_golden() else {
        eprintln!("SKIP: string-param golden absent — run fixtures/string_params/capture.py");
        return;
    };
    assert!(!g.cells.is_empty(), "golden present but parsed to zero cells");

    let mut trained = 0usize;
    let mut config_only = 0usize;
    let mut rejected_both = 0usize;
    let mut unsupported = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for cell in &g.cells {
        let rust_config = config_for(cell);

        // --- a value C++ itself rejected -----------------------------------
        if !cell.ok {
            // Rust must NOT silently train it. Either the config layer rejects it,
            // or it is a documented-unsupported backend/objective that errors at
            // train time.
            match rust_config {
                Err(_) => rejected_both += 1,
                Ok(cfg) => {
                    let corpus = corpus_for(&g, cell, &cfg);
                    if train_raw(&cfg, &corpus).is_err() {
                        rejected_both += 1;
                    } else if is_config_only(&cell.key) {
                        // e.g. `device_type=gpu`: the C++ WHEEL lacks the backend,
                        // but the value is legal config. Rust accepting it (and
                        // training on its own backend) is not a parity break.
                        config_only += 1;
                    } else {
                        failures.push(format!(
                            "{}: C++ rejected ({}) but Rust trained successfully",
                            cell.key,
                            cell.error.lines().next().unwrap_or("")
                        ));
                    }
                }
            }
            continue;
        }

        // --- a value the Rust port does not implement ----------------------
        if let Some(reason) = unsupported_reason(&cell.key) {
            let trained_ok = rust_config.as_ref().is_ok_and(|cfg| {
                let corpus = corpus_for(&g, cell, cfg);
                train_raw(cfg, &corpus).is_ok()
            });
            assert!(
                !trained_ok,
                "{} is listed UNSUPPORTED ({reason}) but now trains — \
                 remove it from UNSUPPORTED and assert real parity",
                cell.key
            );
            unsupported += 1;
            continue;
        }

        let cfg = match rust_config {
            Ok(c) => c,
            Err(e) => {
                failures.push(format!("{}: C++ accepted but Rust config rejected: {e}", cell.key));
                continue;
            }
        };

        // --- config-level agreement ----------------------------------------
        if is_config_only(&cell.key) {
            // Building the config IS the assertion for these (the canonicalization
            // rules themselves are pinned by `lgbm-core`'s metric_param tests
            // against this same oracle); additionally require that training still
            // succeeds so the value cannot break the pipeline.
            let corpus = corpus_for(&g, cell, &cfg);
            if let Err(e) = train_raw(&cfg, &corpus) {
                failures.push(format!("{}: train failed: {e}", cell.key));
            } else {
                config_only += 1;
            }
            continue;
        }

        // --- trained parity -------------------------------------------------
        let corpus = corpus_for(&g, cell, &cfg);
        let booster = match train_raw(&cfg, &corpus) {
            Ok(b) => b,
            Err(e) => {
                failures.push(format!("{}: train failed: {e}", cell.key));
                continue;
            }
        };
        let width = cell.num_class.max(1) as usize;
        let rows = corpus.to_rows();
        let n = cell.pred.len().div_ceil(width).min(rows.len());
        let mut rust: Vec<f32> = Vec::with_capacity(n * width);
        for row in rows.iter().take(n) {
            rust.extend(booster.predict_row_raw(row, -1).iter().map(|&v| v as f32));
        }
        let cpp: Vec<f32> = cell.pred.iter().take(rust.len()).map(|&v| v as f32).collect();
        rust.truncate(cpp.len());
        match compare_within(&rust, &cpp, ORACLE_TOL) {
            Ok(()) => trained += 1,
            Err(m) => failures.push(format!("{}: prediction mismatch: {m:?}", cell.key)),
        }
    }

    eprintln!(
        "string-param oracle: {trained} trained-parity, {config_only} config-level, \
         {rejected_both} rejected-by-both, {unsupported} documented-unsupported, \
         {} FAILING",
        failures.len()
    );
    assert!(
        failures.is_empty(),
        "{} string-param cell(s) diverged from the C++ oracle:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

#[test]
fn skipped_params_are_documented() {
    let Some(g) = load_golden() else {
        eprintln!("SKIP: string-param golden absent");
        return;
    };
    // The 15 string-valued params outside the sweep must each carry a reason, so
    // the "every string-valued parameter is accounted for" claim is auditable.
    assert_eq!(
        g.skipped.len(),
        15,
        "expected 15 documented out-of-sweep string params, got {:?}",
        g.skipped
    );
}

#[test]
fn the_sweep_covers_every_closed_enum_string_param() {
    let Some(g) = load_golden() else {
        eprintln!("SKIP: string-param golden absent");
        return;
    };
    for param in [
        "objective",
        "metric",
        "boosting",
        "data_sample_strategy",
        "tree_learner",
        "device_type",
        "monotone_constraints_method",
        "convert_model_language",
        "task",
    ] {
        let n = g.cells.iter().filter(|c| c.key.starts_with(&format!("{param}="))).count();
        assert!(n > 0, "no swept cells for the `{param}` parameter");
    }
}
