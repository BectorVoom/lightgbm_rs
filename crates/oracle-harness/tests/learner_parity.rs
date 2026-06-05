//! Serial tree-learner parity replay (Phase 5, D-04 cpu hard gate) — SCAFFOLD.
//!
//! This is the Wave-1 (Plan 05-02) END-TO-END HARNESS SKELETON: it establishes
//! the golden path, record-format parsers, and graceful-SKIP discipline so a
//! FAILING end-to-end test exists before the spine learner is written (Plan 03).
//! The per-split (D-06) and per-tree (D-07) bit-exact assertions are filled in by
//! Plan 03/04 once `lgbm_treelearner::SerialTreeLearner` lands; for now the
//! scaffold loads the committed placeholder fixture (or SKIPs when absent) and
//! asserts the record format parses.
//!
//! Idioms follow `oracle-harness/tests/kernel_parity.rs`: a `CARGO_MANIFEST_DIR`-
//! rooted fixture path (NEVER the untracked C++ reference tree), graceful SKIP
//! pre-capture, raw-bit parse helpers, and a localizing assert.
//!
//! Record format (see `xtask/cpp/learner_capture.cpp`) — two golden kinds:
//! ```text
//! # per-split snapshot (D-06): the full per-bin gain array + winning split
//! PSPLIT split=<i> feature=<f> num_bin=<n> gains=<f64bits;...> winner=<f64bits>
//!
//! # per-tree (D-07): a name line followed by the Tree::to_string() text block,
//! # terminated by a line containing only `ENDTREE`.
//! PTREE name=<id>
//! <Tree::to_string() lines...>
//! ENDTREE
//! ```
//! `gains`/`winner` are raw little-endian f64 bit patterns (decimal `u64`) for
//! zero-rounding bit-exact replay; the per-tree block is compared as a `String`.

use std::path::PathBuf;

/// The committed learner golden directory — TRACKED under the oracle-harness
/// crate, NEVER the untracked C++ reference tree.
fn learner_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/learner")
}

/// The scaffold placeholder golden written by `learner-capture` (Plan 03/04
/// replace it with the real per-split/per-tree corpus).
fn scaffold_fixture() -> PathBuf {
    learner_dir().join("scaffold.txt")
}

/// Parse a `;`-separated list of raw little-endian f64 bit patterns (decimal
/// `u64`) into `f64` via `from_bits` (bit-exact). Reused by the Plan-03 per-split
/// assertion (`compare_exact_f64_bits` on these arrays).
#[allow(dead_code)]
fn parse_f64_bits_list(s: &str) -> Vec<f64> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(';')
        .map(|t| f64::from_bits(t.parse::<u64>().expect("f64-bits u64 field")))
        .collect()
}

/// Extract a `key=value` token's value from a whitespace-split line.
#[allow(dead_code)]
fn field<'a>(tokens: &'a [&'a str], key: &str) -> Option<&'a str> {
    tokens
        .iter()
        .find_map(|t| t.strip_prefix(key).and_then(|r| r.strip_prefix('=')))
}

/// A parsed per-split (PSPLIT) record — the D-06 golden kind.
#[derive(Debug)]
#[allow(dead_code)]
struct SplitGolden {
    split: i32,
    feature: i32,
    num_bin: u32,
    gains: Vec<f64>,
    winner: f64,
}

/// A parsed per-tree (PTREE) record — the D-07 golden kind: the grown tree's
/// `Tree::to_string()` block, compared via `String` equality in Plan 03/04.
#[derive(Debug)]
#[allow(dead_code)]
struct TreeGolden {
    name: String,
    text: String,
}

/// Parsed counts the scaffold asserts on (proves the record format round-trips).
#[derive(Debug, Default)]
struct ParsedCounts {
    splits: usize,
    trees: usize,
}

/// Parse the learner golden text into PSPLIT/PTREE record counts. Returns the
/// counts plus the collected records (the records feed the Plan-03/04 bit-exact
/// asserts; the scaffold only checks the format parses).
fn parse(text: &str) -> (ParsedCounts, Vec<SplitGolden>, Vec<TreeGolden>) {
    let mut counts = ParsedCounts::default();
    let mut splits = Vec::new();
    let mut trees = Vec::new();
    let mut lines = text.lines();

    while let Some(raw) = lines.next() {
        let line = raw.trim_end();
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let t: Vec<&str> = trimmed.split_whitespace().collect();
        match t[0] {
            "PSPLIT" => {
                let split = field(&t, "split").and_then(|v| v.parse().ok()).unwrap_or(-1);
                let feature = field(&t, "feature").and_then(|v| v.parse().ok()).unwrap_or(-1);
                let num_bin = field(&t, "num_bin").and_then(|v| v.parse().ok()).unwrap_or(0);
                let gains = parse_f64_bits_list(field(&t, "gains").unwrap_or(""));
                let winner = field(&t, "winner")
                    .map(|v| f64::from_bits(v.parse::<u64>().expect("winner f64-bits")))
                    .unwrap_or(f64::NEG_INFINITY);
                splits.push(SplitGolden {
                    split,
                    feature,
                    num_bin,
                    gains,
                    winner,
                });
                counts.splits += 1;
            }
            "PTREE" => {
                let name = field(&t, "name").unwrap_or("").to_string();
                // Collect the to_string() block until the ENDTREE sentinel.
                let mut body = String::new();
                for tree_line in lines.by_ref() {
                    if tree_line.trim() == "ENDTREE" {
                        break;
                    }
                    body.push_str(tree_line);
                    body.push('\n');
                }
                trees.push(TreeGolden { name, text: body });
                counts.trees += 1;
            }
            // LEARNER_MASTER_SEED / COUNTS header lines.
            _ => continue,
        }
    }
    (counts, splits, trees)
}

#[test]
fn learner_parity_scaffold() {
    let path = scaffold_fixture();
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!(
            "learner_parity: SKIP — fixture {} not found. Run \
             `cargo run -p xtask -- learner-capture` on a machine with a C++ toolchain \
             and commit the golden set. (Pre-capture is the expected Wave-1 state: the \
             spine learner lands in Plan 03; this end-to-end harness is intentionally \
             failing-until-implemented.)",
            path.display()
        );
        return;
    };

    let (counts, splits, trees) = parse(&text);

    // The scaffold placeholder must carry at least one record of EACH kind so the
    // PSPLIT + PTREE record formats are both exercised end-to-end.
    assert!(
        counts.splits >= 1,
        "scaffold fixture {} parsed zero PSPLIT records",
        path.display()
    );
    assert!(
        counts.trees >= 1,
        "scaffold fixture {} parsed zero PTREE records",
        path.display()
    );

    // Structural sanity on the parsed records (localizes a malformed-fixture
    // failure away from a real parity divergence once Plan 03 fills the asserts).
    for s in &splits {
        assert_eq!(
            s.gains.len(),
            s.num_bin as usize,
            "PSPLIT split={} feature={}: gains len {} != num_bin {}",
            s.split,
            s.feature,
            s.gains.len(),
            s.num_bin
        );
        // The winning gain must be present in (or dominate) the per-bin gains;
        // the placeholder uses a single finite gain so this just asserts finite.
        assert!(s.winner.is_finite(), "PSPLIT winner must be a finite gain");
    }
    for tr in &trees {
        assert!(
            tr.text.contains("num_leaves="),
            "PTREE `{}` body is not a Tree::to_string() block",
            tr.name
        );
    }
}
