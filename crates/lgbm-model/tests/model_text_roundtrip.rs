//! D-06 layer 1 (DAT-08 + DAT-09): model-text round-trip is BYTE-IDENTICAL.
//!
//! Load the committed C++ regression `model.txt`, re-`save` it, and assert the
//! Rust-written bytes equal the committed bytes exactly (`compare_exact_bytes`).
//! This proves the load→write spine: the parser captures everything the writer
//! needs (verbatim metadata + per-tree arrays) and the `%.17g`/`{:g}` formatters
//! + `tree_sizes` + recomputed `feature_importances` reproduce the C++ output to
//! the byte.
//!
//! Fixture provenance (project memory `lightgbm-ref-tree-untracked`): resolved
//! via `CARGO_MANIFEST_DIR`; graceful SKIP pre-capture (so the crate tests stay
//! green before fixtures land). Never references a path under `LightGBM/`.

mod golden;

use oracle_harness::comparator::compare_exact_bytes;

use golden::corpus_dir;

fn round_trip_corpus(corpus: &str) {
    let path = corpus_dir(corpus).join("model.txt");
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!(
            "model_text_roundtrip[{corpus}]: SKIP — golden {} not found. \
             Run `cargo run -p xtask -- model-capture` to emit it.",
            path.display()
        );
        return;
    };

    let model = lgbm_model::model_text::load(&text)
        .unwrap_or_else(|e| panic!("[{corpus}] load {}: {e}", path.display()));
    let reemit = lgbm_model::model_text::save(&model);

    match compare_exact_bytes(reemit.as_bytes(), text.as_bytes()) {
        Ok(()) => {}
        Err(m) => panic!(
            "[{corpus}] model-text round-trip diverged from {}: {m}",
            path.display()
        ),
    }
}

#[test]
fn regression_model_text_round_trips_byte_exact() {
    round_trip_corpus("regression");
}
