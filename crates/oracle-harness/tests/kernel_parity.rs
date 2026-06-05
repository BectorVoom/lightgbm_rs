//! Compute-kernel histogram parity replay (ORA-04 cpu hard gate, D-04).
//!
//! For every committed histogram golden case (`tests/fixtures/kernels/histogram.txt`,
//! emitted by `cargo run -p xtask -- kernel-capture`), this test drives the
//! cubecl-cpu `Backend::construct_histograms` over the golden's per-row bins +
//! f32 ordered grad/hess and asserts the resulting f64 histogram cells are
//! BIT-EXACT versus the C++-transcription golden via `compare_exact_f64_bits`.
//! This is the first full vertical slice of the compute backend: a real consumer
//! (the Phase-5 learner, simulated here) builds a histogram from the Phase-2
//! binned store and gets C++-bit-identical f64 cells on the deterministic anchor.
//!
//! Idioms follow `oracle-harness/tests/rng_parity.rs` /
//! `lgbm-dataset/tests/bin_storage_layout.rs`: `CARGO_MANIFEST_DIR` fixture path
//! (never the untracked `LightGBM/` tree), graceful SKIP pre-capture, and a
//! localizing assert that names the diverging case + cell.
//!
//! Record format (see `xtask/cpp/kernel_capture.cpp`):
//! ```text
//! HCASE name=<id> layout=<dense|sparse> num_bin=<n> num_rows=<n> \
//!       skip_default_bin=<0|1> note=<text>
//! BINS <u32;...>            # per-row bin index (== Bin::data(idx))
//! GRAD <f32bits;...>        # per-row ordered_gradients, raw f32 bits (u32 dec)
//! HESS <f32bits;...>        # per-row ordered_hessians,  raw f32 bits (u32 dec)
//! HIST <f64bits;...>        # the [g0,h0,g1,h1,...] f64 cells, raw f64 bits (u64 dec)
//! ```

use std::path::PathBuf;

use lgbm_compute::runtime::cpu_client;
use lgbm_compute::{Backend, CpuBackend};
// The exact comparator is NOT re-exported from the crate root (lib.rs re-exports
// only compare_within/abs_diff_within/Mismatch/ORACLE_TOL); import it via the
// full module path.
use oracle_harness::comparator::compare_exact_f64_bits;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/kernels/histogram.txt")
}

/// Extract a `key=value` token's value from a whitespace-split line.
fn field<'a>(tokens: &'a [&'a str], key: &str) -> Option<&'a str> {
    tokens
        .iter()
        .find_map(|t| t.strip_prefix(key).and_then(|r| r.strip_prefix('=')))
}

fn parse_u32(tokens: &[&str], key: &str) -> u32 {
    field(tokens, key)
        .unwrap_or_else(|| panic!("missing u32 field `{key}`"))
        .parse()
        .unwrap_or_else(|_| panic!("bad u32 field `{key}`"))
}

fn parse_u32_list(s: &str) -> Vec<u32> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(';')
        .map(|t| t.parse::<u32>().expect("u32 field"))
        .collect()
}

/// Parse a `;`-separated list of raw little-endian f32 bit patterns (decimal
/// `u32`) into `f32` via `from_bits` (bit-exact, zero parse rounding).
fn parse_f32_bits_list(s: &str) -> Vec<f32> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(';')
        .map(|t| f32::from_bits(t.parse::<u32>().expect("f32-bits u32 field")))
        .collect()
}

/// Parse a `;`-separated list of raw little-endian f64 bit patterns (decimal
/// `u64`) into `f64` via `from_bits` (bit-exact).
fn parse_f64_bits_list(s: &str) -> Vec<f64> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(';')
        .map(|t| f64::from_bits(t.parse::<u64>().expect("f64-bits u64 field")))
        .collect()
}

#[derive(Debug)]
struct HistGolden {
    name: String,
    num_bin: u32,
    bins: Vec<u32>,
    grad: Vec<f32>,
    hess: Vec<f32>,
    hist: Vec<f64>,
}

fn parse(text: &str) -> Vec<HistGolden> {
    let mut out = Vec::new();
    let mut lines = text.lines();
    while let Some(raw) = lines.next() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let t: Vec<&str> = line.split_whitespace().collect();
        if t[0] != "HCASE" {
            continue; // KERNEL_MASTER_SEED / COUNTS
        }
        let name = field(&t, "name").expect("HCASE name").to_string();
        let num_bin = parse_u32(&t, "num_bin");

        let bt: Vec<&str> = lines.next().expect("BINS").split_whitespace().collect();
        assert_eq!(bt[0], "BINS", "expected BINS after HCASE `{name}`");
        let bins = parse_u32_list(bt.get(1).copied().unwrap_or(""));

        let gt: Vec<&str> = lines.next().expect("GRAD").split_whitespace().collect();
        assert_eq!(gt[0], "GRAD", "expected GRAD for `{name}`");
        let grad = parse_f32_bits_list(gt.get(1).copied().unwrap_or(""));

        let ht: Vec<&str> = lines.next().expect("HESS").split_whitespace().collect();
        assert_eq!(ht[0], "HESS", "expected HESS for `{name}`");
        let hess = parse_f32_bits_list(ht.get(1).copied().unwrap_or(""));

        let st: Vec<&str> = lines.next().expect("HIST").split_whitespace().collect();
        assert_eq!(st[0], "HIST", "expected HIST for `{name}`");
        let hist = parse_f64_bits_list(st.get(1).copied().unwrap_or(""));

        out.push(HistGolden {
            name,
            num_bin,
            bins,
            grad,
            hess,
            hist,
        });
    }
    out
}

#[test]
fn kernel_parity_histogram_bit_exact_on_cpu() {
    let path = fixture_path();
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!(
            "kernel_parity: SKIP — fixture {} not found. Run \
             `cargo run -p xtask -- kernel-capture` on a machine with a C++ toolchain \
             and commit the golden set.",
            path.display()
        );
        return;
    };

    let cases = parse(&text);
    assert!(!cases.is_empty(), "fixture present but parsed zero cases");

    let client = cpu_client();
    let backend = CpuBackend;

    let mut dense_or_sparse_seen = (false, false);

    for c in &cases {
        // Sanity on the parsed record before driving the kernel (localizes a
        // malformed-fixture failure away from a real parity divergence).
        assert_eq!(
            c.grad.len(),
            c.bins.len(),
            "case `{}`: GRAD len != BINS len",
            c.name
        );
        assert_eq!(
            c.hess.len(),
            c.bins.len(),
            "case `{}`: HESS len != BINS len",
            c.name
        );
        assert_eq!(
            c.hist.len(),
            2 * c.num_bin as usize,
            "case `{}`: HIST len != 2*num_bin",
            c.name
        );

        // Drive the cubecl-cpu whole-kernel op over the golden inputs.
        let got = backend
            .construct_histograms(&client, &c.bins, &c.grad, &c.hess, c.num_bin)
            .unwrap_or_else(|e| panic!("case `{}`: construct_histograms failed: {e:?}", c.name));

        // D-04 hard cpu gate: BIT-EXACT vs the C++-transcription golden.
        if let Err(mismatch) = compare_exact_f64_bits(&got, &c.hist) {
            panic!(
                "KERNEL PARITY DIVERGENCE in case `{}` (num_bin={}): {mismatch}",
                c.name, c.num_bin
            );
        }

        if c.name.contains("dense") {
            dense_or_sparse_seen.0 = true;
        }
        if c.name.contains("sparse") {
            dense_or_sparse_seen.1 = true;
        }
    }

    // D-02a coverage assertion: at least one dense AND one sparse layout replayed.
    assert!(
        dense_or_sparse_seen.0,
        "golden must contain at least one dense-layout case"
    );
    assert!(
        dense_or_sparse_seen.1,
        "golden must contain at least one sparse-layout case"
    );
}
