//! End-to-end example-dataset parity (DAT-07, SC#5): for each copied LightGBM
//! example dataset, load the COMMITTED fixture, ingest via `from_mat`, and assert
//! every feature's `bin_upper_bound_` (bit-exact f64) AND per-row `value_to_bin`
//! vector (exact u32) match the C++ golden. This is the first end-to-end binning
//! proof on real data (layers 1+2).
//!
//! Fixture provenance (project memory: lightgbm-ref-tree-untracked): the example
//! input files are COPIED into `tests/fixtures/examples/` (git-tracked); this
//! test NEVER references a path under the untracked repo-root `LightGBM/` tree.
//!
//! Comparison is bit-exact, never the ~1e-6 oracle tolerance: a 1-ULP boundary
//! drift is a real divergence (`compare_exact_f64_bits` / `compare_exact_u32`).
//!
//! Idioms follow `oracle-harness/tests/rng_parity.rs`: `CARGO_MANIFEST_DIR`
//! fixture paths, graceful SKIP pre-capture, and localizing assert messages
//! naming dataset + feature + row.
//!
//! Golden format (`tests/fixtures/example_dataset_binning.txt`):
//! ```text
//! DATASET name=<file> num_rows=<n> num_features=<f> max_bin=<m> \
//!         min_data_in_bin=<d> seed=<s>
//! FEATURE f=<i> num_bin=<n> missing_type=<0|1|2> default_bin=<u32> \
//!         most_freq_bin=<u32> is_trivial=<0|1> upper=<f64bits;...>
//! ASSIGN f=<i> <u32;...>     # per-row value_to_bin over the (capped) column
//! (FEATURE/ASSIGN repeat per feature; DATASET repeats per dataset)
//! ```

use std::path::PathBuf;

use lgbm_core::config::Config;
use lgbm_dataset::bin_mapper::BinMapper;
use oracle_harness::comparator::{compare_exact_f64_bits, compare_exact_u32};

/// Must match `kExampleMaxRows` in `xtask/cpp/bin_capture.cpp` — the row cap the
/// golden was captured with.
const EXAMPLE_MAX_ROWS: usize = 500;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn golden_path() -> PathBuf {
    fixtures_dir().join("example_dataset_binning.txt")
}

fn field<'a>(tokens: &'a [&'a str], key: &str) -> Option<&'a str> {
    tokens
        .iter()
        .find_map(|t| t.strip_prefix(key).and_then(|r| r.strip_prefix('=')))
}

fn parse_i32(tokens: &[&str], key: &str) -> i32 {
    field(tokens, key)
        .unwrap_or_else(|| panic!("missing field `{key}`"))
        .parse()
        .unwrap_or_else(|_| panic!("bad i32 field `{key}`"))
}

fn parse_f64_bits_list(s: &str) -> Vec<f64> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(';')
        .map(|t| f64::from_bits(t.parse::<u64>().expect("f64-bits u64")))
        .collect()
}

fn parse_u32_list(s: &str) -> Vec<u32> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(';')
        .map(|t| t.parse::<u32>().expect("u32"))
        .collect()
}

/// One feature's golden: layer-1 bounds + layer-2 per-row value_to_bin.
struct FeatureGolden {
    upper: Vec<f64>,
    assign: Vec<u32>,
}

/// One dataset's golden block.
struct DatasetGolden {
    name: String,
    num_rows: i32,
    num_features: i32,
    seed: i32,
    features: Vec<FeatureGolden>,
}

fn parse_golden(text: &str) -> Vec<DatasetGolden> {
    let mut datasets = Vec::new();
    let mut cur: Option<DatasetGolden> = None;
    let mut pending_upper: Option<Vec<f64>> = None;

    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        match tokens[0] {
            "MASTER_SEED" | "COUNTS" => {}
            "DATASET" => {
                if let Some(d) = cur.take() {
                    datasets.push(d);
                }
                cur = Some(DatasetGolden {
                    name: field(&tokens, "name").expect("DATASET name").to_string(),
                    num_rows: parse_i32(&tokens, "num_rows"),
                    num_features: parse_i32(&tokens, "num_features"),
                    seed: parse_i32(&tokens, "seed"),
                    features: Vec::new(),
                });
            }
            "FEATURE" => {
                let upper = parse_f64_bits_list(field(&tokens, "upper").unwrap_or(""));
                pending_upper = Some(upper);
            }
            "ASSIGN" => {
                // ASSIGN f=<i> <u32;...> — the list is the LAST whitespace token.
                let list = tokens.last().copied().unwrap_or("");
                // guard against the degenerate "ASSIGN f=0" with empty list:
                let assign = if list.starts_with("f=") {
                    Vec::new()
                } else {
                    parse_u32_list(list)
                };
                let upper = pending_upper.take().expect("FEATURE before ASSIGN");
                cur.as_mut()
                    .expect("ASSIGN before DATASET")
                    .features
                    .push(FeatureGolden { upper, assign });
            }
            other => panic!("unexpected record `{other}`"),
        }
    }
    if let Some(d) = cur.take() {
        datasets.push(d);
    }
    datasets
}

/// Load a copied example TSV (column 0 = label, columns 1.. = features), capped
/// to `EXAMPLE_MAX_ROWS` rows. Returns `(num_rows, num_features, columns)` where
/// `columns[f][row]` is the f64-widened value (matching the C++ harness).
fn load_example(path: &PathBuf) -> (i32, i32, Vec<Vec<f64>>) {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read example fixture {}: {e}", path.display()));
    let mut columns: Vec<Vec<f64>> = Vec::new();
    let mut num_features = 0usize;
    let mut num_rows = 0usize;
    for line in text.lines() {
        if num_rows >= EXAMPLE_MAX_ROWS {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let vals: Vec<f64> = line
            .split_whitespace()
            .map(|t| t.parse::<f64>().expect("numeric example value"))
            .collect();
        if vals.is_empty() {
            continue;
        }
        let nfeat = vals.len() - 1; // drop the label (column 0).
        if num_features == 0 {
            num_features = nfeat;
            columns.resize(nfeat, Vec::new());
        }
        for f in 0..nfeat {
            columns[f].push(vals[f + 1]);
        }
        num_rows += 1;
    }
    (num_rows as i32, num_features as i32, columns)
}

#[test]
fn example_dataset_binning_matches_cpp() {
    let gpath = golden_path();
    let Ok(text) = std::fs::read_to_string(&gpath) else {
        eprintln!(
            "example_dataset_parity: SKIP — golden {} not found. Run \
             `cargo run -p xtask -- bin-capture` on a machine with a C++ toolchain \
             and commit the golden set.",
            gpath.display()
        );
        return;
    };

    let datasets = parse_golden(&text);
    assert!(!datasets.is_empty(), "example golden has no DATASET records");

    for d in &datasets {
        let in_path = fixtures_dir().join("examples").join(&d.name);
        let (num_rows, num_features, columns) = load_example(&in_path);
        assert_eq!(
            num_rows, d.num_rows,
            "{}: row count {num_rows} != golden {}",
            d.name, d.num_rows
        );
        assert_eq!(
            num_features, d.num_features,
            "{}: feature count {num_features} != golden {}",
            d.name, d.num_features
        );

        // Build each feature's mapper with the SAME config + seed the C++ harness
        // used (per-dataset seed recorded in the golden). This reproduces the
        // C-API in-memory sample -> FindBin path the ingest layer runs.
        for f in 0..num_features as usize {
            let mapper = BinMapper::find_bin_from_column(
                &columns[f],
                255,   // kExampleMaxBin
                3,     // kExampleMinDataInBin
                20,    // kExampleMinDataInLeaf
                false, // pre_filter
                true,  // use_missing
                false, // zero_as_missing
                100000, // bin_construct_sample_cnt (sample all rows)
                d.seed,
                &[],
            );

            let fg = &d.features[f];

            // Layer 1: bin_upper_bound_ bit-exact.
            compare_exact_f64_bits(&mapper.bin_upper_bound_, &fg.upper).unwrap_or_else(|m| {
                panic!("{} feature {f}: bin_upper_bound_ mismatch: {m:?}", d.name)
            });

            // Layer 2: per-row value_to_bin exact.
            let rust_assign: Vec<u32> =
                columns[f].iter().map(|&v| mapper.value_to_bin(v)).collect();
            compare_exact_u32(&rust_assign, &fg.assign).unwrap_or_else(|m| {
                panic!("{} feature {f}: per-row value_to_bin mismatch: {m:?}", d.name)
            });
        }
    }

    // Also exercise the full ingest path end-to-end on the first dataset, proving
    // from_mat produces a finished dataset over the real data (not just FindBin).
    let d0 = &datasets[0];
    let in_path = fixtures_dir().join("examples").join(&d0.name);
    let (num_rows, num_features, columns) = load_example(&in_path);
    let mut flat: Vec<f32> = Vec::with_capacity((num_rows * num_features) as usize);
    for r in 0..num_rows as usize {
        for c in 0..num_features as usize {
            flat.push(columns[c][r] as f32);
        }
    }
    let mut cfg = Config::default();
    cfg.max_bin = 255;
    cfg.min_data_in_bin = 3;
    cfg.bin_construct_sample_cnt = 100000;
    cfg.feature_pre_filter = false;
    cfg.data_random_seed = d0.seed;
    let md = lgbm_dataset::Metadata::new(
        vec![0.0f32; num_rows as usize],
        Vec::new(),
        Vec::new(),
        Vec::new(),
    )
    .unwrap();
    let (ds, _md) = lgbm_dataset::from_mat(&flat, num_rows, num_features, &cfg, md).unwrap();
    assert_eq!(ds.num_data(), num_rows);
    assert_eq!(ds.num_features(), num_features);
}
