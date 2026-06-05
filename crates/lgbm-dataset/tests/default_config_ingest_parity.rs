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
///
/// Representation contract (agreed with the bin_capture.cpp emitter, Task 2):
/// a TRIVIAL feature (`is_trivial=1`) carries NO `group=`/`subfeature=` fields
/// and NO `ASSIGN` line (it is dropped from the store, dataset.cpp:337-343); a
/// NON-trivial feature carries `group=`/`subfeature=` AND exactly one `ASSIGN`.
struct FeatureGolden {
    num_bin: i32,
    default_bin: u32,
    most_freq_bin: u32,
    is_trivial: bool,
    upper: Vec<f64>,
    /// C++-Construct group id (`feature2group_`); `None` for a trivial feature.
    group: Option<i32>,
    /// C++-Construct sub-feature index (`feature2subfeature_`); `None` if trivial.
    subfeature: Option<i32>,
    /// STORED per-row bins (what the FeatureGroup holds), NOT recomputed. Empty
    /// for a trivial feature (no ASSIGN line).
    stored: Vec<u32>,
    /// Whether an ASSIGN line was seen for this feature (representation check).
    has_assign: bool,
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
                let is_trivial = parse_i32(&tokens, "is_trivial") != 0;
                // group=/subfeature= present iff non-trivial (Task 2 contract).
                let group = field(&tokens, "group").map(|s| s.parse::<i32>().expect("group i32"));
                let subfeature = field(&tokens, "subfeature")
                    .map(|s| s.parse::<i32>().expect("subfeature i32"));
                pending = Some(FeatureGolden {
                    num_bin: parse_i32(&tokens, "num_bin"),
                    default_bin: parse_i32(&tokens, "default_bin") as u32,
                    most_freq_bin: parse_i32(&tokens, "most_freq_bin") as u32,
                    is_trivial,
                    upper: parse_f64_bits_list(field(&tokens, "upper").unwrap_or("")),
                    group,
                    subfeature,
                    stored: Vec::new(),
                    has_assign: false,
                });
            }
            "ASSIGN" => {
                let list = tokens.last().copied().unwrap_or("");
                let stored = if list.starts_with("f=") {
                    Vec::new()
                } else {
                    parse_u32_list(list)
                };
                let fg = pending.as_mut().expect("FEATURE before ASSIGN");
                fg.stored = stored;
                fg.has_assign = true;
            }
            other => panic!("unexpected record `{other}`"),
        }
    }
    if let Some(fg) = pending.take() {
        features.push(fg);
    }

    // Representation-agreement check (hard parse-time panic, NOT a skip): a
    // trivial feature MUST have no group/subfeature and no ASSIGN; a non-trivial
    // feature MUST have group=/subfeature= AND exactly one ASSIGN. A mismatch
    // means the emitter (Task 2) and the test disagree — that is a hard error.
    for (f, fg) in features.iter().enumerate() {
        if fg.is_trivial {
            assert!(
                fg.group.is_none() && fg.subfeature.is_none() && !fg.has_assign,
                "feature {f}: trivial feature must carry NO group/subfeature and NO ASSIGN \
                 (group={:?} subfeature={:?} has_assign={})",
                fg.group, fg.subfeature, fg.has_assign
            );
        } else {
            assert!(
                fg.group.is_some() && fg.subfeature.is_some() && fg.has_assign,
                "feature {f}: non-trivial feature must carry group=/subfeature= AND one ASSIGN \
                 (group={:?} subfeature={:?} has_assign={})",
                fg.group, fg.subfeature, fg.has_assign
            );
        }
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
    // WR-01: the golden is COMMITTED, so a missing/unreadable file is a HARD
    // failure (panic), never a silent SKIP. A silent skip would let the parity
    // gate vanish without anyone noticing on a determinism-root phase.
    let text = std::fs::read_to_string(&gpath)
        .unwrap_or_else(|e| panic!("committed golden {} unreadable: {e}", gpath.display()));

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
    // CR-01: trivial features are DROPPED, so ds.num_features() is the count of
    // NON-trivial (used) golden features, NOT the total column count. Whenever a
    // feature is trivial this is strictly less than g.num_features — exactly the
    // C++ Construct behaviour (used_features only). An unfixed construct that kept
    // every feature would give ds.num_features() == g.num_features and FAIL here.
    let used_count = g.features.iter().filter(|f| !f.is_trivial).count() as i32;
    assert_eq!(
        ds.num_features(),
        used_count,
        "ds.num_features {} != non-trivial golden feature count {} \
         (CR-01: trivial features must be dropped)",
        ds.num_features(),
        used_count
    );

    // At least one feature MUST be trivial in the golden (the engineered flip
    // feature) — otherwise the test would not discriminate the bug.
    assert!(
        g.features.iter().any(|f| f.is_trivial) && g.features.iter().any(|f| !f.is_trivial),
        "golden must contain BOTH a trivial and a non-trivial feature to exercise the flip"
    );

    // Whole-grouping cross-check: the dataset's group count equals the number of
    // distinct golden group ids (one feature per group on this dense fixture).
    let distinct_groups: std::collections::BTreeSet<i32> =
        g.features.iter().filter_map(|f| f.group).collect();
    assert_eq!(
        ds.num_groups() as usize,
        distinct_groups.len(),
        "num_groups {} != distinct golden group ids {}",
        ds.num_groups(),
        distinct_groups.len()
    );

    for f in 0..g.num_features as usize {
        let fg = &g.features[f];

        if fg.is_trivial {
            // TRIVIAL feature: DROPPED on the default ingest path. It is excluded
            // from used_feature_map_ (used_feature_map_[f] = -1), so it has no
            // group and no stored bins. This is the PRIMARY CR-01 assertion: an
            // unfixed construct that stored the trivial feature would give
            // feature_to_group(f) != -1 and FAIL here.
            assert_eq!(
                ds.feature_to_group(f),
                -1,
                "feature {f}: trivial feature must be EXCLUDED \
                 (feature_to_group == -1), got {} (CR-01: trivial feature wrongly stored)",
                ds.feature_to_group(f)
            );
            // Do NOT call ds.feature_group(f) for a trivial feature — it has none.
            continue;
        }

        // NON-trivial feature: assert the GROUPING matches the C++-Construct
        // golden FIRST (closes the EFB parity hole — an incorrectly-built
        // EfbSamples that re-groups features FAILS here, not silently).
        let want_group = fg.group.expect("non-trivial golden has group");
        let want_sub = fg.subfeature.expect("non-trivial golden has subfeature");
        assert_eq!(
            ds.feature_to_group(f),
            want_group,
            "feature {f}: feature_to_group {} != C++-Construct golden group {} \
             (EFB grouping divergence)",
            ds.feature_to_group(f),
            want_group
        );
        assert_eq!(
            ds.feature_to_subfeature(f),
            want_sub,
            "feature {f}: feature_to_subfeature {} != C++-Construct golden subfeature {}",
            ds.feature_to_subfeature(f),
            want_sub
        );

        // Locate the group/sub via the verified maps and read the mapper.
        let group = ds.feature_group(ds.feature_to_group(f) as usize);
        let sub = ds.feature_to_subfeature(f) as usize;
        let mapper = group.bin_mapper(sub);

        // is_trivial_ must be false here (matches golden).
        assert_eq!(
            mapper.is_trivial_, fg.is_trivial,
            "feature {f}: is_trivial_ {} != golden {} (scaled-filter_cnt divergence)",
            mapper.is_trivial_, fg.is_trivial
        );

        // num_bin_ + bin_upper_bound_ (f64-bit exact), default_bin_, most_freq_bin_.
        assert_eq!(
            mapper.num_bin_, fg.num_bin,
            "feature {f}: num_bin_ {} != golden {}",
            mapper.num_bin_, fg.num_bin
        );
        compare_exact_f64_bits(&mapper.bin_upper_bound_, &fg.upper)
            .unwrap_or_else(|m| panic!("feature {f}: bin_upper_bound_ mismatch: {m:?}"));
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

        // STORED per-row bins read out of the FeatureGroup store (NOT recomputed
        // via value_to_bin) — bit-exact against the C++-Construct golden ASSIGN.
        let store = group
            .bin_data()
            .unwrap_or_else(|| panic!("feature {f}: no single-value bin store"));
        let stored: Vec<u32> = (0..g.num_rows).map(|r| store.data(r)).collect();
        compare_exact_u32(&stored, &fg.stored)
            .unwrap_or_else(|m| panic!("feature {f}: STORED per-row bin mismatch: {m:?}"));
    }
}
