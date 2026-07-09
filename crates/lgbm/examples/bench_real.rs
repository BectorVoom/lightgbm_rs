//! Real-workload train benchmark + deterministic model/prediction dump for the
//! pure-Rust `lightgbm_rs` engine — quick task 260620-d3v.
//!
//! This is a THRESHOLD-AGNOSTIC harness: it never reads or sets the
//! `LGBM_UNIFIED_BFS_THRESHOLD` / `LGBM_UNIFIED_SUBSCAN_THRESHOLD` env vars. The
//! split-fusion campaign A/B (a48 smaller-child fusion + b97 larger-child fusion +
//! c5v core-derived gate thresholds) is driven ENTIRELY by those two env vars at
//! invocation time — the library's `lgbm_compute::unified_*_threshold()` gates read
//! them and the env value takes ultimate precedence over the core-derived default.
//! The same binary therefore measures both arms with zero harness skew:
//!   - Campaign OFF (= exact pre-campaign baseline):
//!       LGBM_UNIFIED_BFS_THRESHOLD=18446744073709551615 \
//!       LGBM_UNIFIED_SUBSCAN_THRESHOLD=18446744073709551615   (usize::MAX → never fires)
//!   - Campaign ON: both vars UNSET (core-derived ~100/130 at 16 cores).
//!
//! Loading is via `RawCorpus::from_columns` (column-major, no transpose) + a
//! realistic training `Config`, driven through `lgbm::train_raw` UNCHANGED — real
//! continuous features go through the bit-exact `BinMapper`, NOT the identity-bin
//! `DenseCorpus` path. The loader/Config shape mirrors
//! `crates/oracle-harness/tests/raw_bin_train_parity.rs`.
//!
//! Usage:
//!   cargo run --release -p lgbm --example bench_real -- <path> <format> <objective> <mode>
//!
//!   <path>      dataset file (TSV/whitespace-separated rows)
//!   <format>    "tsv-label0" : whitespace columns, col 0 = label, cols 1.. = features
//!                              (binary.train / regression.train layout AND the
//!                              high-dim export, which is written label-first)
//!   <objective> "regression" | "binary" (drives Config.objective)
//!   <mode>      "bench"      : warm (discard run 0), >=3 timed reps, print MEDIAN train-wall ms.
//!                              With LGBM_PHASE_PROF=1 also dumps BUILD_NS/SCAN_NS via phase_prof.
//!               "dump-model" : train once, print booster.model_to_string() (bit-identical diff)
//!               "dump-pred"  : train once, predict every row, print one f32 per row at full
//!                              precision as raw hex bits (bit-identical diff)
//!
//! NO library hot-path edits, NO new external Rust dep (std + existing workspace crates only).

use std::time::Instant;

use lgbm::{train_raw, Config, RawCorpus, TrainingBuilder};

/// Parse a whitespace/TSV file in "tsv-label0" layout: each non-empty line is
/// `label feat0 feat1 ... featN`. Returns `(columns, labels)` where `columns[j]`
/// is feature `j`'s per-row values (column-major, ready for `from_columns`).
fn load_tsv_label0(path: &str) -> (Vec<Vec<f64>>, Vec<f32>) {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read dataset {path}: {e}"));
    let mut labels: Vec<f32> = Vec::new();
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut num_feat: Option<usize> = None;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut it = line.split_whitespace();
        let label: f32 = it
            .next()
            .expect("row has a label column")
            .parse()
            .expect("label parses as f32");
        let feats: Vec<f64> = it.map(|t| t.parse::<f64>().expect("feature parses as f64")).collect();
        match num_feat {
            None => num_feat = Some(feats.len()),
            Some(n) => assert_eq!(n, feats.len(), "ragged row: expected {n} features, got {}", feats.len()),
        }
        labels.push(label);
        rows.push(feats);
    }
    let nf = num_feat.expect("dataset has at least one row");
    let nr = rows.len();
    // Transpose row-major → column-major once, up front (off the train hot path).
    let mut columns: Vec<Vec<f64>> = vec![Vec::with_capacity(nr); nf];
    for row in &rows {
        for (j, &v) in row.iter().enumerate() {
            columns[j].push(v);
        }
    }
    (columns, labels)
}

/// A realistic training config: max_bin=255, num_leaves=31, num_iterations=100,
/// learning_rate=0.1, with the supplied objective. Deterministic so the bench is
/// reproducible AND the campaign-ON / campaign-OFF model texts are comparable.
fn bench_config(objective: &str) -> Config {
    let mut cfg = TrainingBuilder::new()
        .objective(objective)
        .num_iterations(100)
        .learning_rate(0.1)
        .num_leaves(31)
        .seed(1)
        .deterministic(true)
        .build()
        .expect("config builds");
    cfg.max_bin = 255;
    cfg.data_random_seed = 1;
    cfg
}

/// Build a `RawCorpus` from column-major data + labels, carrying the training config
/// so binning honors max_bin/min_data_in_bin (the train path passes the same config).
fn build_corpus(columns: Vec<Vec<f64>>, labels: Vec<f32>, cfg: &Config) -> RawCorpus {
    let mut raw = RawCorpus::from_columns(columns, labels);
    raw.config = cfg.clone();
    raw
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!(
            "usage: bench_real <path> <format=tsv-label0> <objective=regression|binary> <mode=bench|dump-model|dump-pred>"
        );
        std::process::exit(2);
    }
    let path = &args[1];
    let format = &args[2];
    let objective = &args[3];
    let mode = &args[4];

    assert_eq!(
        format, "tsv-label0",
        "only the tsv-label0 format is implemented (col 0 = label, cols 1.. = features)"
    );

    let (columns, labels) = load_tsv_label0(path);
    let num_rows = labels.len();
    let num_feat = columns.len();
    let cfg = bench_config(objective);
    let corpus = build_corpus(columns, labels, &cfg);

    let label = format!("{}|{}feat", path.rsplit('/').next().unwrap_or(path), num_feat);

    match mode.as_str() {
        "bench" => {
            eprintln!(
                "[bench_real] {label}: rows={num_rows} feat={num_feat} obj={objective} \
                 max_bin={} num_leaves={} num_iter={}",
                cfg.max_bin, cfg.num_leaves, cfg.num_iterations
            );
            // Warm: train once and DISCARD (page-in, allocator warmup, branch predictors).
            let _ = train_raw(&cfg, &corpus).expect("warmup train ok");
            // Reset any phase_prof counters the warmup accumulated, so BUILD/SCAN below
            // reflect only the timed reps.
            lgbm_treelearner::phase_prof::dump("warmup-discard");

            let reps = 3usize;
            let mut walls_ms: Vec<f64> = Vec::with_capacity(reps);
            for r in 0..reps {
                let t = Instant::now();
                let booster = train_raw(&cfg, &corpus).expect("timed train ok");
                let ms = t.elapsed().as_secs_f64() * 1e3;
                // Defeat dead-code elimination of the trained model.
                std::hint::black_box(&booster);
                walls_ms.push(ms);
                eprintln!("[bench_real] {label} rep {} = {ms:.3} ms", r + 1);
            }
            let mut sorted = walls_ms.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = sorted[sorted.len() / 2];
            // phase_prof accumulated over ALL timed reps; report it then reset.
            lgbm_treelearner::phase_prof::dump(&label);
            println!("{label}\tmedian_train_ms={median:.3}\treps={reps}\twalls={walls_ms:?}");
        }
        "dump-model" => {
            let booster = train_raw(&cfg, &corpus).expect("train ok");
            // Deterministic LightGBM-compatible v4 model text (byte-stable formatter).
            print!("{}", booster.model_to_string());
        }
        "dump-pred" => {
            let booster = train_raw(&cfg, &corpus).expect("train ok");
            let rows = corpus.to_rows();
            // One transformed f32 prediction per row, emitted as RAW IEEE-754 bits so
            // the bit-identical diff is exact (no formatting rounding can mask a flip).
            for pred in booster.predict(&rows) {
                let bits: Vec<String> = pred.iter().map(|v| format!("{:08x}", v.to_bits())).collect();
                println!("{}", bits.join(" "));
            }
        }
        other => {
            eprintln!("unknown mode '{other}' (expected bench|dump-model|dump-pred)");
            std::process::exit(2);
        }
    }
}
