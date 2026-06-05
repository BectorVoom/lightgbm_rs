//! Metadata golden replay (DAT-06): for every committed metadata case, build the
//! Rust `Metadata`, run `finish_load`, and assert the stored
//! label/weights/init_score round-trip AND the derived `query_weights` are
//! BIT-IDENTICAL to the C++ `Metadata::CalculateQueryWeights` golden.
//!
//! This proves the query-weight derivation (mean weight per group, computed in
//! `label_t`==f32 arithmetic — accumulate in f32, divide by the int group size)
//! is reproduced exactly. Query weights are compared bit-exact via f32-bit
//! equality, never the ~1e-6 oracle tolerance: a 1-ULP drift is a real
//! divergence (SC#2).
//!
//! Idioms follow `oracle-harness/tests/rng_parity.rs`: `CARGO_MANIFEST_DIR`
//! fixture path (never the untracked `LightGBM/` tree), graceful SKIP
//! pre-capture, and localizing assert messages (case name + index).
//!
//! Fixture written by `cargo run -p xtask -- bin-capture` into
//! `tests/fixtures/metadata.txt`. Record format:
//! ```text
//! MCASE name=<id> num_rows=<n>
//! LABEL <f32bits;...>            # per-row label, raw f32 bits (u32 dec)
//! WEIGHTS <f32bits;...>          # per-row weight (empty = none)
//! INIT_SCORE <f64bits;...>       # per-row init score (empty = none)
//! QUERY_BOUNDARIES <i32;...>     # prefix-sum group boundaries (empty = none)
//! QUERY_WEIGHTS <f32bits;...>    # derived mean weight per group
//! ```

use std::path::PathBuf;

use lgbm_dataset::metadata::Metadata;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/metadata.txt")
}

fn field<'a>(tokens: &'a [&'a str], key: &str) -> Option<&'a str> {
    tokens
        .iter()
        .find_map(|t| t.strip_prefix(key).and_then(|r| r.strip_prefix('=')))
}

fn parse_f32_bits_list(s: &str) -> Vec<f32> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(';')
        .map(|t| f32::from_bits(t.parse::<u32>().expect("f32-bits u32 field")))
        .collect()
}

fn parse_f64_bits_list(s: &str) -> Vec<f64> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(';')
        .map(|t| f64::from_bits(t.parse::<u64>().expect("f64-bits u64 field")))
        .collect()
}

fn parse_i32_list(s: &str) -> Vec<i32> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(';')
        .map(|t| t.parse::<i32>().expect("i32 field"))
        .collect()
}

/// The payload of one metadata golden case.
struct MetaCase {
    name: String,
    label: Vec<f32>,
    weights: Vec<f32>,
    init_score: Vec<f64>,
    query_boundaries: Vec<i32>,
    query_weights: Vec<f32>,
}

/// The value portion of a `KEY <list>` line (everything after the first token).
fn rest_after_key<'a>(line: &'a str, key: &str) -> &'a str {
    line.strip_prefix(key)
        .map(|r| r.trim_start())
        .unwrap_or("")
}

fn parse_cases(text: &str) -> Vec<MetaCase> {
    let mut cases = Vec::new();
    let mut lines = text.lines().peekable();
    while let Some(raw) = lines.next() {
        let line = raw.trim();
        if !line.starts_with("MCASE") {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let name = field(&tokens, "name").expect("MCASE name").to_string();

        let label_line = lines.next().expect("LABEL line");
        let weights_line = lines.next().expect("WEIGHTS line");
        let init_line = lines.next().expect("INIT_SCORE line");
        let qb_line = lines.next().expect("QUERY_BOUNDARIES line");
        let qw_line = lines.next().expect("QUERY_WEIGHTS line");

        cases.push(MetaCase {
            name,
            label: parse_f32_bits_list(rest_after_key(label_line.trim(), "LABEL")),
            weights: parse_f32_bits_list(rest_after_key(weights_line.trim(), "WEIGHTS")),
            init_score: parse_f64_bits_list(rest_after_key(init_line.trim(), "INIT_SCORE")),
            query_boundaries: parse_i32_list(rest_after_key(qb_line.trim(), "QUERY_BOUNDARIES")),
            query_weights: parse_f32_bits_list(rest_after_key(qw_line.trim(), "QUERY_WEIGHTS")),
        });
    }
    cases
}

/// Bit-exact f32 slice compare (never the ~1e-6 oracle tolerance).
fn assert_f32_bits_eq(case: &str, what: &str, rust: &[f32], cpp: &[f32]) {
    assert_eq!(
        rust.len(),
        cpp.len(),
        "{case}: {what} length mismatch ({} vs {})",
        rust.len(),
        cpp.len()
    );
    for (i, (&r, &c)) in rust.iter().zip(cpp.iter()).enumerate() {
        assert_eq!(
            r.to_bits(),
            c.to_bits(),
            "{case}: {what}[{i}] mismatch: rust={r} (0x{:08X}) vs cpp={c} (0x{:08X})",
            r.to_bits(),
            c.to_bits()
        );
    }
}

#[test]
fn metadata_query_weights_match_cpp_golden() {
    let path = fixture_path();
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!(
            "metadata: SKIP — fixture {} not found. Run `cargo run -p xtask -- bin-capture` \
             on a machine with a C++ toolchain and commit the golden set.",
            path.display()
        );
        return;
    };

    let cases = parse_cases(&text);
    assert!(!cases.is_empty(), "metadata golden has no MCASE records");

    let mut saw_groups = false;
    let mut saw_no_groups = false;

    for case in &cases {
        // Build + validate; the golden is always well-formed so this must succeed.
        let mut md = Metadata::new(
            case.label.clone(),
            case.weights.clone(),
            case.init_score.clone(),
            case.query_boundaries.clone(),
        )
        .unwrap_or_else(|e| panic!("{}: Metadata::new failed: {e}", case.name));

        // Stored vectors round-trip (label/weights bit-exact f32, init_score f64).
        assert_f32_bits_eq(&case.name, "label", &md.label, &case.label);
        assert_f32_bits_eq(&case.name, "weights", &md.weights, &case.weights);
        assert_eq!(
            md.init_score.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            case.init_score.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "{}: init_score bit-exact round-trip",
            case.name
        );
        assert_eq!(
            md.query_boundaries, case.query_boundaries,
            "{}: query_boundaries round-trip",
            case.name
        );

        // Derive query weights and compare bit-exact to the C++ golden.
        md.finish_load();
        assert_f32_bits_eq(&case.name, "query_weights", &md.query_weights, &case.query_weights);

        if case.query_boundaries.is_empty() {
            saw_no_groups = true;
        } else {
            saw_groups = true;
        }
    }

    // Coverage guards: the corpus must exercise both the grouped (query weights
    // derived) and ungrouped (empty query weights) paths so neither can silently
    // regress.
    assert!(saw_groups, "metadata corpus must include a grouped (ranking) case");
    assert!(
        saw_no_groups,
        "metadata corpus must include an ungrouped case (empty query_weights)"
    );
}

// ---------------------------------------------------------------------------
// Malformed-input boundary tests (Security V5): never panic; typed errors.
// ---------------------------------------------------------------------------

#[test]
fn wrong_length_weights_returns_shape_mismatch() {
    use lgbm_dataset::DatasetError;
    let err = Metadata::new(vec![1.0, 2.0, 3.0], vec![1.0, 1.0], Vec::new(), Vec::new())
        .expect_err("weights length 2 != num_rows 3 must error");
    assert!(matches!(err, DatasetError::ShapeMismatch { .. }));
}

#[test]
fn non_monotone_query_boundaries_returns_query_boundary_error() {
    use lgbm_dataset::DatasetError;
    let err = Metadata::new(
        vec![0.0, 0.0, 0.0, 0.0],
        Vec::new(),
        Vec::new(),
        vec![0, 3, 2, 4],
    )
    .expect_err("non-monotone query boundaries must error");
    assert!(matches!(err, DatasetError::QueryBoundary { .. }));
}

#[test]
fn query_boundaries_not_summing_to_num_rows_returns_query_boundary_error() {
    use lgbm_dataset::DatasetError;
    let err = Metadata::new(vec![0.0, 0.0, 0.0, 0.0], Vec::new(), Vec::new(), vec![0, 2, 3])
        .expect_err("query boundaries not ending at num_rows must error");
    assert!(matches!(err, DatasetError::QueryBoundary { .. }));
}
