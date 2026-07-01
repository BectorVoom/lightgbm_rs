//! Shared objective-parity harness (Phase 19, 19-00) — reused in shape from
//! `boosting_parity.rs` and `rank_parity.rs`.
//!
//! This is a **subdir helper module**, NOT its own test binary: Cargo only treats
//! top-level `tests/*.rs` as integration-test crates, so `tests/objective_common/
//! mod.rs` is compiled only when a family test file pulls it in via
//! `mod objective_common;`. The four Wave-2 family plans (19-01..04) each include it
//! and drive their own `#[test]` fns; this file intentionally defines **no `#[test]`**.
//!
//! Comparators are NOT re-exported here — family tests pull them directly from
//! `oracle_harness::comparator::{compare_exact_u32, compare_within, ORACLE_TOL}`.
//!
//! ## Capture-gated skip-pass contract
//! [`read_boosting_golden`] / [`read_rank_golden`] return `None` (with an eprintln
//! SKIP note) when the fixture is absent, so a fresh checkout that has not run the
//! `xtask *-oracle-capture` steps still **builds and skip-passes** (the
//! `learner_parity` / `boosting_parity` idiom).

#![allow(dead_code)]

use std::path::PathBuf;

/// Parse a whitespace-separated line of u32 bit patterns into `Vec<f32>` via
/// `f32::from_bits` (the `*_gh` grad/hess golden encoding; reused in shape from
/// `boosting_parity.rs`).
pub fn parse_f32_bits_line(line: &str) -> Vec<f32> {
    line.split_whitespace()
        .map(|t| f32::from_bits(t.parse::<u32>().expect("u32 bits")))
        .collect()
}

/// Parse a `GRAD <u32 bits…>\nHESS <u32 bits…>` g/h golden into `(grad, hess)`.
///
/// Comment (`#`) and blank lines are ignored; only the `GRAD `/`HESS ` prefixed
/// lines are read — identical to the `boosting/*_gh_iter*.txt` and
/// `rank/lambdarank_gh_iter*.txt` fixture format.
pub fn parse_gh(text: &str) -> (Vec<f32>, Vec<f32>) {
    let mut grad = Vec::new();
    let mut hess = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("GRAD ") {
            grad = parse_f32_bits_line(rest);
        } else if let Some(rest) = line.strip_prefix("HESS ") {
            hess = parse_f32_bits_line(rest);
        }
    }
    (grad, hess)
}

/// The committed boosting golden directory — TRACKED under the oracle-harness crate
/// (`tests/fixtures/boosting`), NEVER the untracked C++/LightGBM reference tree.
pub fn boosting_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/boosting")
}

/// The committed ranking golden directory (`tests/fixtures/rank`) — TRACKED under
/// the oracle-harness crate.
pub fn rank_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/rank")
}

/// Read a boosting golden by file name, returning `None` (with a SKIP note) when the
/// fixture is absent so a fresh checkout skip-passes (capture-gated contract).
pub fn read_boosting_golden(name: &str) -> Option<String> {
    read_optional(boosting_dir().join(name), "boosting-oracle-capture")
}

/// Read a ranking golden by file name, returning `None` (with a SKIP note) when the
/// fixture is absent so a fresh checkout skip-passes (capture-gated contract).
pub fn read_rank_golden(name: &str) -> Option<String> {
    read_optional(rank_dir().join(name), "rank-oracle-capture")
}

/// Shared capture-gated read: `Some(contents)` when present, else `None` + a SKIP
/// note pointing at the xtask capture that would populate it.
fn read_optional(path: PathBuf, capture_cmd: &str) -> Option<String> {
    match std::fs::read_to_string(&path) {
        Ok(s) => Some(s),
        Err(_) => {
            eprintln!(
                "objective_common: SKIP — golden {} not found. Run \
                 `LGBM_CAPTURE_PYTHON=… cargo run -p xtask -- {capture_cmd}`.",
                path.display()
            );
            None
        }
    }
}
