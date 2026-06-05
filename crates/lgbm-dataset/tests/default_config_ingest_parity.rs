//! Default-configuration in-memory ingest parity (GAP-1/GAP-2 closure, SC#1 /
//! SC#5 / ORA-03 bin stage; DAT-01, DAT-07).
//!
//! Every PRIOR ingest parity test (and the C++ example golden) forces
//! `feature_pre_filter=false`, so the DEFAULT path — `feature_pre_filter=true`
//! with `bin_construct_sample_cnt < num_rows` (sampling active) — shipped with
//! zero coverage. On that default path C++
//! `DatasetLoader::ConstructFromSampleData` (dataset_loader.cpp:623-624) feeds
//! FindBin a SCALED pre-filter threshold
//!   `filter_cnt = (min_data_in_leaf * total_sample_size) / num_dist_data`
//! (integer truncation), NOT the raw `min_data_in_leaf`. For the dense
//! in-memory path (c_api.cpp:1360-1374) `total_sample_size = sample_cnt` and
//! `num_dist_data = total_nrow`.
//!
//! This golden was captured through that exact scaled-filter_cnt path with
//! `num_rows=200`, `bin_construct_sample_cnt=50`, `min_data_in_leaf=20`, so
//! `filter_cnt = (20*50)/200 = 5 != 20`. Feature **f1** is engineered so its
//! only split leaves 10 sampled rows on the minority side: PASSES need_filter at
//! filter_cnt=5 (non-trivial) but FAILS at the raw 20 (trivial). So `is_trivial_`
//! flips between the correct scaled threshold and the buggy raw one.
//!
//! PRIMARY assertion: `is_trivial_` (MEDIUM-1). A trivial feature does NOT lower
//! `num_bin_` (both C++ and Rust only set the flag), so `num_bin_` alone would
//! PASS even before the fix — `is_trivial_` is the discriminating signal. The
//! per-row check reads the STORED bins out of `feature_group().bin_data()` (the
//! bins the dataset actually stored), NOT a recomputed `value_to_bin` — the
//! kernel is unaffected by the bug, so recomputing would silently defeat the gap
//! closure (the example_dataset_parity.rs:238 recompute pattern is deliberately
//! NOT mirrored here).
//!
//! Comparison is bit-exact, never the ~1e-6 oracle tolerance.
//!
//! Golden format (`tests/fixtures/default_config_ingest.txt`):
//! ```text
//! DATASET num_rows=<n> num_features=<f> max_bin=<m> min_data_in_bin=<d> \
//!         min_data_in_leaf=<l> bin_construct_sample_cnt=<s> seed=<seed>
//! MATRIX rows=<n> cols=<f> <f64bits;...>   # row-major raw cells (single source of truth)
//! FEATURE f=<i> num_bin=<n> missing_type=<0|1|2> default_bin=<u32> \
//!         most_freq_bin=<u32> is_trivial=<0|1> filter_cnt=<c> upper=<f64bits;...>
//! ASSIGN f=<i> <u32;...>                    # STORED per-row bin (FeatureGroup store)
//! ```

use std::path::PathBuf;

use lgbm_core::config::Config;
use oracle_harness::comparator::{compare_exact_f64_bits, compare_exact_u32};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn golden_path() -> PathBuf {
    fixtures_dir().join("default_config_ingest.txt")
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

/// One feature's golden: layer-1 metadata + the STORED per-row bins.
struct FeatureGolden {
    num_bin: i32,
    default_bin: u32,
    most_freq_bin: u32,
    is_trivial: bool,
    upper: Vec<f64>,
    /// STORED per-row bins (what the FeatureGroup holds), NOT recomputed.
    stored: Vec<u32>,
}

struct Golden {
    num_rows: i32,
    num_features: i32,
    max_bin: i32,
    min_data_in_bin: i32,
    min_data_in_leaf: i32,
    bin_construct_sample_cnt: i32,
    seed: i32,
    /// Row-major raw f64 cells (`matrix[row*num_features + col]`).
    matrix: Vec<f64>,
    features: Vec<FeatureGolden>,
}

fn parse_golden(text: &str) -> Golden {
    let mut num_rows = 0i32;
    let mut num_features = 0i32;
    let mut max_bin = 0i32;
    let mut min_data_in_bin = 0i32;
    let mut min_data_in_leaf = 0i32;
    let mut bin_construct_sample_cnt = 0i32;
    let mut seed = 0i32;
    let mut matrix: Vec<f64> = Vec::new();
    let mut features: Vec<FeatureGolden> = Vec::new();
    let mut pending: Option<FeatureGolden> = None;

    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        match tokens[0] {
            "MASTER_SEED" => {}
            "DATASET" => {
                num_rows = parse_i32(&tokens, "num_rows");
                num_features = parse_i32(&tokens, "num_features");
                max_bin = parse_i32(&tokens, "max_bin");
                min_data_in_bin = parse_i32(&tokens, "min_data_in_bin");
                min_data_in_leaf = parse_i32(&tokens, "min_data_in_leaf");
                bin_construct_sample_cnt = parse_i32(&tokens, "bin_construct_sample_cnt");
                seed = parse_i32(&tokens, "seed");
            }
            "MATRIX" => {
                // The cell list is the LAST whitespace token.
                let list = tokens.last().copied().unwrap_or("");
                matrix = parse_f64_bits_list(list);
            }
            "FEATURE" => {
                if let Some(fg) = pending.take() {
                    features.push(fg);
                }
                pending = Some(FeatureGolden {
                    num_bin: parse_i32(&tokens, "num_bin"),
                    default_bin: parse_i32(&tokens, "default_bin") as u32,
                    most_freq_bin: parse_i32(&tokens, "most_freq_bin") as u32,
                    is_trivial: parse_i32(&tokens, "is_trivial") != 0,
                    upper: parse_f64_bits_list(field(&tokens, "upper").unwrap_or("")),
                    stored: Vec::new(),
                });
            }
            "ASSIGN" => {
                let list = tokens.last().copied().unwrap_or("");
                let stored = if list.starts_with("f=") {
                    Vec::new()
                } else {
                    parse_u32_list(list)
                };
                pending.as_mut().expect("FEATURE before ASSIGN").stored = stored;
            }
            other => panic!("unexpected record `{other}`"),
        }
    }
    if let Some(fg) = pending.take() {
        features.push(fg);
    }

    Golden {
        num_rows,
        num_features,
        max_bin,
        min_data_in_bin,
        min_data_in_leaf,
        bin_construct_sample_cnt,
        seed,
        matrix,
        features,
    }
}

#[test]
fn default_config_ingest_matches_cpp() {
    let gpath = golden_path();
    let Ok(text) = std::fs::read_to_string(&gpath) else {
        eprintln!(
            "default_config_ingest_parity: SKIP — golden {} not found. Run \
             `cargo run -p xtask -- bin-capture` on a machine with a C++ toolchain \
             and commit the golden.",
            gpath.display()
        );
        return;
    };

    let g = parse_golden(&text);
    assert!(g.num_rows > 0 && g.num_features > 0, "empty golden");
    assert_eq!(
        g.bin_construct_sample_cnt < g.num_rows,
        true,
        "golden must have sample_cnt ({}) < num_rows ({}) to exercise scaling",
        g.bin_construct_sample_cnt,
        g.num_rows
    );
    assert_eq!(
        g.matrix.len() as i32,
        g.num_rows * g.num_features,
        "MATRIX cell count {} != num_rows*num_features {}",
        g.matrix.len(),
        g.num_rows * g.num_features
    );
    assert_eq!(g.features.len() as i32, g.num_features, "feature count");

    // Reconstruct the synthetic matrix by READING the emitted MATRIX block (NOT
    // regenerating from seed) so it is byte-identical to the C++ harness's.
    let flat: Vec<f32> = g.matrix.iter().map(|&v| v as f32).collect();

    // Build a Config on the DEFAULT path: feature_pre_filter = TRUE, with the
    // header's min_data_in_leaf / sample_cnt / max_bin / min_data_in_bin / seed.
    let mut cfg = Config::default();
    cfg.max_bin = g.max_bin;
    cfg.min_data_in_bin = g.min_data_in_bin;
    cfg.min_data_in_leaf = g.min_data_in_leaf;
    cfg.bin_construct_sample_cnt = g.bin_construct_sample_cnt;
    cfg.feature_pre_filter = true;
    cfg.use_missing = true;
    cfg.zero_as_missing = false;
    cfg.data_random_seed = g.seed;

    let md = lgbm_dataset::Metadata::new(
        vec![0.0f32; g.num_rows as usize],
        Vec::new(),
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let (ds, _md) = lgbm_dataset::from_mat(&flat, g.num_rows, g.num_features, &cfg, md)
        .expect("from_mat on default config");
    assert_eq!(ds.num_data(), g.num_rows);
    assert_eq!(ds.num_features(), g.num_features);

    // At least one feature MUST be trivial in the golden (the engineered flip
    // feature) — otherwise the test would not discriminate the bug.
    assert!(
        g.features.iter().any(|f| f.is_trivial) && g.features.iter().any(|f| !f.is_trivial),
        "golden must contain BOTH a trivial and a non-trivial feature to exercise the flip"
    );

    for f in 0..g.num_features as usize {
        let fg = &g.features[f];
        // One-feature-per-group default: feature f lives at group f, sub-feature 0.
        let group = ds.feature_group(f);
        let mapper = group.bin_mapper(0);

        // PRIMARY discriminating assertion: is_trivial_ must match C++.
        assert_eq!(
            mapper.is_trivial_, fg.is_trivial,
            "feature {f}: is_trivial_ {} != golden {} (scaled-filter_cnt divergence)",
            mapper.is_trivial_, fg.is_trivial
        );

        // num_bin_ and the bin_upper_bound_ (f64-bit exact). num_bin_ does NOT
        // change on a trivial feature, so this passes regardless of the flip —
        // it guards the kernel, not the flip.
        assert_eq!(
            mapper.num_bin_, fg.num_bin,
            "feature {f}: num_bin_ {} != golden {}",
            mapper.num_bin_, fg.num_bin
        );
        compare_exact_f64_bits(&mapper.bin_upper_bound_, &fg.upper)
            .unwrap_or_else(|m| panic!("feature {f}: bin_upper_bound_ mismatch: {m:?}"));

        // default_bin_ / most_freq_bin_ are only populated on the non-trivial
        // branch (a trivial feature leaves them at 0). Assert them where the
        // feature is non-trivial.
        if !fg.is_trivial {
            assert_eq!(
                mapper.default_bin_, fg.default_bin,
                "feature {f}: default_bin_ {} != golden {}",
                mapper.default_bin_, fg.default_bin
            );
            assert_eq!(
                mapper.most_freq_bin_, fg.most_freq_bin,
                "feature {f}: most_freq_bin_ {} != golden {}",
                mapper.most_freq_bin_, fg.most_freq_bin
            );
        }

        // STORED per-row bins read out of the FeatureGroup store (NOT recomputed
        // via value_to_bin). A flip in is_trivial_ -> most_freq_bin_ shift -> the
        // stored bins diverge too, so this independently catches the bug.
        let store = group
            .bin_data()
            .unwrap_or_else(|| panic!("feature {f}: no single-value bin store"));
        let stored: Vec<u32> = (0..g.num_rows).map(|r| store.data(r)).collect();
        compare_exact_u32(&stored, &fg.stored)
            .unwrap_or_else(|m| panic!("feature {f}: STORED per-row bin mismatch: {m:?}"));
    }
}
