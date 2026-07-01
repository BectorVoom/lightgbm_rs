//! On-device data-partition golden-anchor parity replay (Phase 18, ODL-13).
//!
//! Mirrors `oracle-harness/tests/best_split_parity.rs`: parse the committed
//! `tests/fixtures/kernels/partition.txt` golden (Phase-4 numeric PCASE + the
//! Phase-18 D-02 flag fan-out, the D-03 categorical `PCAT` membership blocks, and
//! the D-08 16-int `PPACKET` child-stats packet), then drive the cubecl-cpu
//! device `mark → prefix-sum → scatter` fold and assert the post-partition row
//! order / membership route / packet fields are BIT-EXACT vs the golden. Idioms
//! follow `best_split_parity.rs`: `CARGO_MANIFEST_DIR` fixture path (never the
//! untracked `LightGBM/` tree), raw-integer / raw-f64-bits parsing (zero
//! rounding), graceful SKIP when the fixture is absent, and a localizing assert.
//!
//! ## Wave-0 status (18-01)
//! The §9-faithful device partition core lands in **18-02** (Wave 1). Until then
//! there is no device entry point to drive, so every cell here parses + structurally
//! validates its golden and is marked
//! `#[ignore = "Wave-0 scaffold; un-ignore when 18-02 lands"]` so the merge-gate
//! `cargo test --workspace` stays GREEN with `LGBM_CUDA_ON_DEVICE` unset
//! (D-13 / ODL-19). 18-02 replaces each `// UN-IGNORE (18-02):` block with the
//! real device call + `compare_exact_*` and removes the `#[ignore]`.
//!
//! ## Record format (see `tests/fixtures/kernels/partition.txt`)
//! ```text
//! COUNTS partition=<np> cat=<nc> packet=<npk>
//! PCASE name=.. num_bin=.. min_bin=.. max_bin=.. threshold=.. most_freq_bin=.. \
//!       missing_type=0|1|2 default_bin=.. default_left=0|1 note=..
//! PBINS  <u32;..>      # per-row bin
//! PORDER <u32;..>      # post-partition reordered index array (left then right)
//! PSPLIT <n>           # left/right split point (to_left_total)
//! PCAT name=.. num_bin=.. min_bin=.. max_bin=.. most_freq_bin=.. num_threshold=.. note=..
//! PCATBITSET <u32;..>  # membership threshold bitset words
//! PCATBINS <u32;..> ; PCATORDER <u32;..> ; PCATSPLIT <n>
//! PPACKET name=.. left_leaf=.. left_num_data=.. left_data_start=.. right_leaf=.. \
//!       right_num_data=.. right_data_start=.. smaller=.. larger=.. \
//!       left_sum_hessians=<f64bits> right_sum_hessians=<f64bits> \
//!       left_sum_gradients=<f64bits> right_sum_gradients=<f64bits> note=..
//! ```

use std::path::PathBuf;

fn kernels_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/kernels")
}

/// Parse an `f64` from a raw little-endian f64 bit pattern (decimal `u64`).
#[allow(dead_code)]
fn parse_f64_bits(s: &str) -> f64 {
    f64::from_bits(s.parse::<u64>().expect("f64-bits u64 field"))
}

/// Extract a `key=value` token's value from a whitespace-split line.
fn field<'a>(tokens: &'a [&'a str], key: &str) -> Option<&'a str> {
    tokens
        .iter()
        .find_map(|t| t.strip_prefix(key).and_then(|r| r.strip_prefix('=')))
}

fn parse_i64(tokens: &[&str], key: &str) -> i64 {
    field(tokens, key)
        .unwrap_or_else(|| panic!("missing int field `{key}`"))
        .parse()
        .unwrap_or_else(|_| panic!("bad int field `{key}`"))
}

/// Parse a `;`-separated list of decimal `u32`.
fn parse_u32_list(s: &str) -> Vec<u32> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(';')
        .map(|t| t.parse::<u32>().expect("u32 field"))
        .collect()
}

/// Read the partition golden, SKIP (returning `None`) when absent.
fn read_partition() -> Option<String> {
    let path = kernels_dir().join("partition.txt");
    match std::fs::read_to_string(&path) {
        Ok(s) => Some(s),
        Err(_) => {
            eprintln!("partition_parity: SKIP — fixture {} not found.", path.display());
            None
        }
    }
}

/// Assert `order` is a permutation of `0..n` split at `split_point` — the
/// structural invariant of a stable partition (D-04). Real device-vs-golden
/// bit-exact comparison is wired in 18-02.
fn assert_permutation(name: &str, order: &[u32], n: usize, split_point: usize) {
    assert_eq!(order.len(), n, "PARTITION `{name}`: order length != row count");
    assert!(split_point <= n, "PARTITION `{name}`: split_point out of range");
    let mut seen = vec![false; n];
    for &i in order {
        let i = i as usize;
        assert!(i < n && !seen[i], "PARTITION `{name}`: order is not a permutation");
        seen[i] = true;
    }
}

/// `order` cell (ODL-13, numeric row-order): parse every PCASE and assert the
/// golden PORDER is a valid stable partition. 18-02 un-ignores and drives the
/// device `mark → prefix-sum → scatter` fold, comparing bit-exact.
#[test]
#[ignore = "Wave-0 scaffold; un-ignore when 18-02 lands"]
fn partition_parity_order() {
    let Some(text) = read_partition() else { return };
    let mut lines = text.lines();
    let mut n_cases = 0;
    while let Some(raw) = lines.next() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let t: Vec<&str> = line.split_whitespace().collect();
        if t[0] != "PCASE" {
            continue;
        }
        let name = field(&t, "name").expect("PCASE name").to_string();
        let _min_bin = parse_i64(&t, "min_bin") as u32;
        let _max_bin = parse_i64(&t, "max_bin") as u32;
        let _threshold = parse_i64(&t, "threshold") as u32;
        let _most_freq_bin = parse_i64(&t, "most_freq_bin") as u32;
        let _missing_type = field(&t, "missing_type").and_then(|s| s.parse::<i64>().ok()).unwrap_or(0);
        let _default_bin = field(&t, "default_bin").and_then(|s| s.parse::<i64>().ok()).unwrap_or(0);
        let _default_left = field(&t, "default_left").and_then(|s| s.parse::<i64>().ok()).unwrap_or(0) != 0;

        let bt: Vec<&str> = lines.next().expect("PBINS").split_whitespace().collect();
        assert_eq!(bt[0], "PBINS", "expected PBINS for `{name}`");
        let bins = parse_u32_list(bt.get(1).copied().unwrap_or(""));
        let ot: Vec<&str> = lines.next().expect("PORDER").split_whitespace().collect();
        assert_eq!(ot[0], "PORDER", "expected PORDER for `{name}`");
        let order = parse_u32_list(ot.get(1).copied().unwrap_or(""));
        let st: Vec<&str> = lines.next().expect("PSPLIT").split_whitespace().collect();
        assert_eq!(st[0], "PSPLIT", "expected PSPLIT for `{name}`");
        let split_point: usize = st[1].parse().expect("split_point usize");

        assert_permutation(&name, &order, bins.len(), split_point);

        // UN-IGNORE (18-02): drive the device fold and compare bit-exact, e.g.
        //   let (got_order, got_split) = lgbm_compute::kernels::data_partition::
        //       partition_on_device(&client, &bins, num_bin, min_bin, max_bin,
        //           default_bin, most_freq_bin, missing_type, default_left, threshold)?;
        //   oracle_harness::comparator::compare_exact_u32(&got_order, &order)?;
        //   assert_eq!(got_split, split_point);
        n_cases += 1;
    }
    assert!(n_cases > 0, "partition fixture present but parsed zero PCASE");
}

/// `cat` cell (ODL-13, categorical membership routing, D-03).
mod cat {
    use super::*;

    /// Parse every PCAT block and assert the membership golden PCATORDER is a valid
    /// stable partition. 18-02 un-ignores and drives the device
    /// `GenDataToLeftBitVectorKernel_Categorical` (`FindInBitsetCUDA`) fold.
    #[test]
    #[ignore = "Wave-0 scaffold; un-ignore when 18-02 lands"]
    fn partition_parity_cat_membership() {
        let Some(text) = read_partition() else { return };
        let mut lines = text.lines();
        let mut n_cases = 0;
        while let Some(raw) = lines.next() {
            let line = raw.trim();
            let t: Vec<&str> = line.split_whitespace().collect();
            if t.is_empty() || t[0] != "PCAT" {
                continue;
            }
            let name = field(&t, "name").expect("PCAT name").to_string();
            let _min_bin = parse_i64(&t, "min_bin") as u32;
            let _max_bin = parse_i64(&t, "max_bin") as u32;
            let _most_freq_bin = parse_i64(&t, "most_freq_bin") as u32;

            let bs: Vec<&str> = lines.next().expect("PCATBITSET").split_whitespace().collect();
            assert_eq!(bs[0], "PCATBITSET", "expected PCATBITSET for `{name}`");
            let _bitset = parse_u32_list(bs.get(1).copied().unwrap_or(""));
            let bt: Vec<&str> = lines.next().expect("PCATBINS").split_whitespace().collect();
            assert_eq!(bt[0], "PCATBINS", "expected PCATBINS for `{name}`");
            let bins = parse_u32_list(bt.get(1).copied().unwrap_or(""));
            let ot: Vec<&str> = lines.next().expect("PCATORDER").split_whitespace().collect();
            assert_eq!(ot[0], "PCATORDER", "expected PCATORDER for `{name}`");
            let order = parse_u32_list(ot.get(1).copied().unwrap_or(""));
            let st: Vec<&str> = lines.next().expect("PCATSPLIT").split_whitespace().collect();
            assert_eq!(st[0], "PCATSPLIT", "expected PCATSPLIT for `{name}`");
            let split_point: usize = st[1].parse().expect("cat split_point usize");

            assert_permutation(&name, &order, bins.len(), split_point);

            // UN-IGNORE (18-02): drive the categorical device fold and compare, e.g.
            //   let (got_order, got_split) = lgbm_compute::kernels::data_partition::
            //       partition_categorical_on_device(&client, &bins, min_bin, max_bin,
            //           most_freq_bin, &bitset)?;
            //   oracle_harness::comparator::compare_exact_u32(&got_order, &order)?;
            n_cases += 1;
        }
        assert!(n_cases > 0, "partition fixture present but parsed zero PCAT");
    }
}

/// `packet` cell (ODL-13, 16-int SplitTreeStructure child-stats packet, D-08).
mod packet {
    use super::*;

    /// Parse every PPACKET block and assert the packet field self-consistency
    /// (smaller/larger chosen by num_data). 18-02 un-ignores and compares the
    /// device-reconstructed 16-int buffer field-for-field (ints exact, the 4 f64
    /// sums via `compare_exact_f64_bits`).
    #[test]
    #[ignore = "Wave-0 scaffold; un-ignore when 18-02 lands"]
    fn partition_parity_packet_fields() {
        let Some(text) = read_partition() else { return };
        let mut n_cases = 0;
        for raw in text.lines() {
            let t: Vec<&str> = raw.trim().split_whitespace().collect();
            if t.is_empty() || t[0] != "PPACKET" {
                continue;
            }
            let name = field(&t, "name").expect("PPACKET name").to_string();
            let left_leaf = parse_i64(&t, "left_leaf");
            let right_leaf = parse_i64(&t, "right_leaf");
            let left_num = parse_i64(&t, "left_num_data");
            let right_num = parse_i64(&t, "right_num_data");
            let smaller = parse_i64(&t, "smaller");
            let larger = parse_i64(&t, "larger");
            // The 4 f64 packet sums (parsed to prove the bit-hex convention).
            let _lh = parse_f64_bits(field(&t, "left_sum_hessians").expect("lh"));
            let _rh = parse_f64_bits(field(&t, "right_sum_hessians").expect("rh"));
            let _lg = parse_f64_bits(field(&t, "left_sum_gradients").expect("lg"));
            let _rg = parse_f64_bits(field(&t, "right_sum_gradients").expect("rg"));

            // buffer[6]/[7] = smaller/larger by num_data[left] < num_data[right]
            // (SplitTreeStructureKernel:823) — self-consistency of the golden.
            let (exp_smaller, exp_larger) = if left_num < right_num {
                (left_leaf, right_leaf)
            } else {
                (right_leaf, left_leaf)
            };
            assert_eq!(smaller, exp_smaller, "PPACKET `{name}`: smaller-child index");
            assert_eq!(larger, exp_larger, "PPACKET `{name}`: larger-child index");

            // UN-IGNORE (18-02): reconstruct the 16-int packet on-device and compare
            // field-for-field, e.g.
            //   let pk = lgbm_compute::kernels::data_partition::split_tree_structure_packet(..);
            //   assert_eq!(pk.ints, [left_leaf, left_num, left_start, right_leaf, ...]);
            //   oracle_harness::comparator::compare_exact_f64_bits(&pk.sums, &[lh, rh, lg, rg])?;
            n_cases += 1;
        }
        assert!(n_cases > 0, "partition fixture present but parsed zero PPACKET");
    }
}
