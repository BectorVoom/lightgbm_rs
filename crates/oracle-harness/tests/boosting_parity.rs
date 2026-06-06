//! GBDT spine end-to-end boosting parity replay (Phase 6, Wave-0 scaffold).
//!
//! This is the **Nyquist scaffold** (01-config `nyquist_validation = true`): the
//! failing/`#[ignore]`d end-to-end test that names the spine golden it WILL replay
//! once the boosting loop lands. It exists now so the test surface is sampled
//! before the implementation — the slice in 06-02 fills in the body and removes
//! the `#[ignore]`.
//!
//! Mirrors the `learner_parity.rs` idioms: a `CARGO_MANIFEST_DIR`-rooted fixture
//! path (NEVER the untracked `LightGBM/` tree), the comparator precision contract
//! (`compare_exact_f64_bits` for per-iter scores / model-text leaf values,
//! `compare_within(.., ORACLE_TOL)` for g/h and metrics), and a localizing assert.
//!
//! ## Validation layers (RESEARCH §Validation Architecture, L1–L5)
//! - L1 `gradients`        — per-row g/h from the objective (~1e-6).
//! - L2 `score_accumulation` — per-iter raw scores (bit-exact f64).
//! - L3 `early_stopping`   — eval-history / best-iteration (D-12).
//! - L4 `bagging_rng`      — bagged row indices via RNG-replay (D-13 Option A,
//!   exact u32) — DEFERRED to 06-05; named here as the seam.
//! - L5 `spine_end_to_end` — `save_model()` text + `predict()` (D-13/L5).
//! - `custom_objective`    — the D-04 closure objective path.
//!
//! Allowed D-07 collapse: the spine == bagging-off / early-stopping-off /
//! boost-from-average-on cell (RESEARCH §Cross-Product Collapse Analysis).

use std::path::PathBuf;

/// The committed boosting golden directory — TRACKED under the oracle-harness
/// crate, NEVER the untracked C++/LightGBM reference tree. Populated by
/// `cargo run -p xtask -- boosting-oracle-capture` (wave 2+).
#[allow(dead_code)]
fn boosting_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/boosting")
}

/// The spine model golden this end-to-end replay will assert against once the
/// boosting loop lands (06-02).
#[allow(dead_code)]
fn spine_model_fixture() -> PathBuf {
    boosting_dir().join("regression_spine_model.txt")
}

#[test]
#[ignore = "MISSING — implemented in wave 2 (06-02): GBDT spine loop + save_model parity"]
fn spine_end_to_end() {
    // L5: train the regression L2 spine for N iterations and assert the grown
    // ensemble's model text + predict() match `regression_spine_model.txt`
    // bit-exact (compare_exact_f64_bits on leaf values).
    panic!("MISSING — implemented in wave 2 (06-02)");
}

#[test]
#[ignore = "MISSING — implemented in wave 2 (06-02): per-iter raw-score accumulation parity"]
fn score_accumulation() {
    // L2: per-iter `predict(raw_score=True, num_iteration=k)` bit-exact f64.
    panic!("MISSING — implemented in wave 2 (06-02)");
}

#[test]
#[ignore = "MISSING — implemented in wave 2 (06-02): per-row gradient/hessian parity"]
fn gradients() {
    // L1: per-row g/h from the L2 objective vs the captured reference (~1e-6).
    panic!("MISSING — implemented in wave 2 (06-02)");
}

#[test]
#[ignore = "MISSING — implemented in wave 4 (06-05): early-stopping / eval-history parity"]
fn early_stopping() {
    // L3: eval-history + best_iteration vs `record_evaluation` (D-12).
    panic!("MISSING — implemented in wave 4 (06-05)");
}

#[test]
#[ignore = "MISSING — implemented in wave 3 (06-03+): custom (D-04 closure) objective parity"]
fn custom_objective() {
    // The D-04 custom-objective closure path produces the same g/h + tree.
    panic!("MISSING — implemented in wave 3 (06-03)");
}

#[test]
#[ignore = "MISSING — implemented in wave 4 (06-05): bagging RNG-replay (D-13 Option A) bagged-index parity"]
fn bagging_rng() {
    // L4: bagged row indices derived in-Rust from the replayed RNG sequence,
    // asserted exact (compare_exact_u32) against the captured bag.
    panic!("MISSING — implemented in wave 4 (06-05)");
}
