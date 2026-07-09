//! Storage-layout golden replay (DAT-02): for every committed storage case,
//! rebuild the Rust `BinMapper` + single-value `FeatureGroup`, push the same
//! rows, `finish_load`, and assert the raw byte layout is BYTE-IDENTICAL to the
//! C++ golden — `DenseBin` `data_` (incl. the 4-bit packed variant), `SparseBin`
//! `deltas_`/`vals_`, and `FeatureGroup` `bin_offsets_`/`num_total_bin_`.
//!
//! This is the exact memory the Phase 4 histogram kernels read against, so it is
//! compared bit-exact via the oracle-harness exact comparators (NOT the `~1e-6`
//! oracle tolerance). Idioms follow `oracle-harness/tests/rng_parity.rs`:
//! `CARGO_MANIFEST_DIR` fixture path (never the untracked `LightGBM/` tree),
//! graceful SKIP pre-capture, and localizing assert messages.
//!
//! Fixture written by `cargo run -p xtask -- bin-capture` into
//! `tests/fixtures/bin_storage_layout.txt`. Record format:
//! ```text
//! SCASE name=<id> max_bin=<m> min_data_in_bin=<d> use_missing=<0|1> \
//!       zero_as_missing=<0|1> num_rows=<n>
//! VALUES <f64bits;...>          # full per-row column, raw f64 bits (u64 dec)
//! MAPPER num_bin=<n> missing_type=<0|1|2> most_freq_bin=<u32> is_trivial=<0|1>
//! FG num_total_bin=<u64> bin_offsets=<u32;...> is_sparse=<0|1>
//! STORE_DENSE bytes=<byte;...>
//!   OR
//! STORE_SPARSE deltas=<byte;...> vals=<byte;...>
//! ```

use std::path::PathBuf;

use lgbm_dataset::FeatureGroup;
use lgbm_dataset::bin_mapper::BinMapper;
use oracle_harness::comparator::{compare_exact_bytes, compare_exact_u32};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bin_storage_layout.txt")
}

fn field<'a>(tokens: &'a [&'a str], key: &str) -> Option<&'a str> {
    tokens
        .iter()
        .find_map(|t| t.strip_prefix(key).and_then(|r| r.strip_prefix('=')))
}

fn parse_i32(tokens: &[&str], key: &str) -> i32 {
    field(tokens, key)
        .unwrap_or_else(|| panic!("missing i32 field `{key}`"))
        .parse()
        .unwrap_or_else(|_| panic!("bad i32 field `{key}`"))
}

fn parse_u64(tokens: &[&str], key: &str) -> u64 {
    field(tokens, key)
        .unwrap_or_else(|| panic!("missing u64 field `{key}`"))
        .parse()
        .unwrap_or_else(|_| panic!("bad u64 field `{key}`"))
}

fn parse_f64_bits_list(s: &str) -> Vec<f64> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(';')
        .map(|t| f64::from_bits(t.parse::<u64>().expect("f64-bits u64 field")))
        .collect()
}

fn parse_u32_list(s: &str) -> Vec<u32> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(';')
        .map(|t| t.parse::<u32>().expect("u32 field"))
        .collect()
}

fn parse_byte_list(s: &str) -> Vec<u8> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(';')
        .map(|t| t.parse::<u8>().expect("byte field"))
        .collect()
}

#[derive(Debug)]
struct StorageGolden {
    name: String,
    column: Vec<f64>,
    max_bin: i32,
    min_data_in_bin: i32,
    use_missing: bool,
    zero_as_missing: bool,
    num_rows: i32,
    g_num_bin: i32,
    g_most_freq_bin: u32,
    g_num_total_bin: u64,
    g_bin_offsets: Vec<u32>,
    g_is_sparse: bool,
    g_dense_bytes: Option<Vec<u8>>,
    g_sparse_deltas: Option<Vec<u8>>,
    g_sparse_vals: Option<Vec<u8>>,
}

fn parse(text: &str) -> Vec<StorageGolden> {
    let mut out = Vec::new();
    let mut lines = text.lines();
    while let Some(raw) = lines.next() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let t: Vec<&str> = line.split_whitespace().collect();
        if t[0] != "SCASE" {
            continue; // MASTER_SEED / COUNTS
        }
        let name = field(&t, "name").expect("SCASE name").to_string();
        let max_bin = parse_i32(&t, "max_bin");
        let min_data_in_bin = parse_i32(&t, "min_data_in_bin");
        let use_missing = parse_i32(&t, "use_missing") != 0;
        let zero_as_missing = parse_i32(&t, "zero_as_missing") != 0;
        let num_rows = parse_i32(&t, "num_rows");

        let vt: Vec<&str> = lines.next().expect("VALUES").split_whitespace().collect();
        assert_eq!(vt[0], "VALUES", "expected VALUES after SCASE `{name}`");
        let column = parse_f64_bits_list(vt.get(1).copied().unwrap_or(""));

        let mt: Vec<&str> = lines.next().expect("MAPPER").split_whitespace().collect();
        assert_eq!(mt[0], "MAPPER", "expected MAPPER for `{name}`");
        let g_num_bin = parse_i32(&mt, "num_bin");
        let g_most_freq_bin = parse_i32(&mt, "most_freq_bin") as u32;

        let ft: Vec<&str> = lines.next().expect("FG").split_whitespace().collect();
        assert_eq!(ft[0], "FG", "expected FG for `{name}`");
        let g_num_total_bin = parse_u64(&ft, "num_total_bin");
        let g_bin_offsets = parse_u32_list(field(&ft, "bin_offsets").expect("bin_offsets"));
        let g_is_sparse = parse_i32(&ft, "is_sparse") != 0;

        let st: Vec<&str> = lines.next().expect("STORE").split_whitespace().collect();
        let (g_dense_bytes, g_sparse_deltas, g_sparse_vals) = match st[0] {
            "STORE_DENSE" => (
                Some(parse_byte_list(field(&st, "bytes").expect("dense bytes"))),
                None,
                None,
            ),
            "STORE_SPARSE" => (
                None,
                Some(parse_byte_list(field(&st, "deltas").expect("sparse deltas"))),
                Some(parse_byte_list(field(&st, "vals").expect("sparse vals"))),
            ),
            other => panic!("unknown STORE_* `{other}` for `{name}`"),
        };

        out.push(StorageGolden {
            name,
            column,
            max_bin,
            min_data_in_bin,
            use_missing,
            zero_as_missing,
            num_rows,
            g_num_bin,
            g_most_freq_bin,
            g_num_total_bin,
            g_bin_offsets,
            g_is_sparse,
            g_dense_bytes,
            g_sparse_deltas,
            g_sparse_vals,
        });
    }
    out
}

#[test]
fn bin_storage_layout_replays_every_committed_case() {
    let path = fixture_path();
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!(
            "bin_storage_layout: SKIP — fixture {} not found. Run \
             `cargo run -p xtask -- bin-capture` on a machine with a C++ toolchain \
             and commit the golden set.",
            path.display()
        );
        return;
    };

    let cases = parse(&text);
    assert!(!cases.is_empty(), "fixture present but parsed zero cases");

    let mut dense_4bit_seen = false;
    let mut sparse_seen = false;

    for c in &cases {
        // Rebuild the mapper from the FULL column (sample_cnt == num_rows; the
        // C++ harness builds the storage mapper without sub-sampling so the
        // layout is self-contained). NaN-strip happens inside find_bin_numeric.
        let mapper = BinMapper::find_bin_numeric(
            c.column.clone(),
            c.max_bin,
            c.min_data_in_bin,
            /*min_split_data=*/ 1,
            /*pre_filter=*/ false,
            c.use_missing,
            c.zero_as_missing,
            c.num_rows as usize,
            &[],
        );

        // Cross-check the mapper meta matches the golden before comparing storage
        // (localizes a divergence to binning vs storage).
        assert_eq!(
            mapper.num_bin_, c.g_num_bin,
            "case `{}`: num_bin mismatch (binning drift, not storage)",
            c.name
        );
        assert_eq!(
            mapper.most_freq_bin_, c.g_most_freq_bin,
            "case `{}`: most_freq_bin mismatch",
            c.name
        );

        // Build the single-value FeatureGroup, push every row, finish_load.
        let mut fg = FeatureGroup::new_single(mapper, c.num_rows);
        for (row, &v) in c.column.iter().enumerate() {
            fg.push_data(0, row as i32, v);
        }
        fg.finish_load();

        // FeatureGroup offset layout parity.
        assert_eq!(
            fg.num_total_bin_, c.g_num_total_bin,
            "case `{}`: num_total_bin_ mismatch",
            c.name
        );
        compare_exact_u32(&fg.bin_offsets_, &c.g_bin_offsets).unwrap_or_else(|m| {
            panic!("case `{}`: bin_offsets_ {m}", c.name);
        });
        assert_eq!(
            fg.is_sparse(),
            c.g_is_sparse,
            "case `{}`: is_sparse mismatch",
            c.name
        );

        // Storage byte parity.
        let bin = fg.bin_data().expect("single-value group has bin_data");
        if c.g_is_sparse {
            sparse_seen = true;
            let deltas = bin
                .sparse_deltas()
                .unwrap_or_else(|| panic!("case `{}`: expected sparse deltas", c.name));
            compare_exact_bytes(deltas, c.g_sparse_deltas.as_ref().unwrap()).unwrap_or_else(|m| {
                panic!("case `{}`: SparseBin deltas_ {m}", c.name);
            });
            let vals = bin
                .sparse_vals_bytes()
                .unwrap_or_else(|| panic!("case `{}`: expected sparse vals", c.name));
            compare_exact_bytes(&vals, c.g_sparse_vals.as_ref().unwrap()).unwrap_or_else(|m| {
                panic!("case `{}`: SparseBin vals_ {m}", c.name);
            });
        } else {
            if c.g_num_total_bin <= 16 {
                dense_4bit_seen = true;
            }
            let rust_bytes = bin.raw_bytes();
            compare_exact_bytes(&rust_bytes, c.g_dense_bytes.as_ref().unwrap()).unwrap_or_else(
                |m| {
                    panic!(
                        "case `{}` (num_total_bin={}): DenseBin raw data_ {m}",
                        c.name, c.g_num_total_bin
                    );
                },
            );
        }
    }

    // Guarantee the corpus actually exercised the 4-bit and sparse paths.
    assert!(
        dense_4bit_seen,
        "corpus did not cover a 4-bit DenseBin (num_total_bin <= 16) case"
    );
    assert!(sparse_seen, "corpus did not cover a SparseBin case");
}
