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
//! ## 18-02 status (Wave 1, ODL-13)
//! The §9-faithful device partition core landed in **18-02**. Each cell drives the
//! cpu f64 stable-partition ANCHOR (`partition_leaf_stable` /
//! `partition_categorical_stable`, D-04 CONFIRMED) AND the full device fold
//! (`partition_on_device` / `partition_categorical_on_device`) where the cpu
//! device client is available, comparing bit-exact to the golden — never
//! GPU-vs-GPU (D-12).
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

use lgbm_compute::kernels::data_partition::{
    partition_categorical_on_device, partition_categorical_stable, partition_leaf_stable,
    partition_on_device, split_tree_structure_packet,
};
use lgbm_compute::kernels::partition::{partition_child_ranges_device, DeviceLeafSplits};
use lgbm_compute::kernels::split_info::SplitScalars;
use lgbm_compute::runtime::cpu_client;
use lgbm_compute::BinColumn;
use oracle_harness::comparator::{compare_exact_f64_bits, compare_exact_u32};

fn kernels_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/kernels")
}

/// Parse an `f64` from a raw little-endian f64 bit pattern (decimal `u64`).
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
/// structural invariant of a stable partition (D-04).
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

/// `order` cell (ODL-13, numeric row-order): parse every PCASE, drive the cpu f64
/// stable-partition ANCHOR AND the device `mark → prefix-sum → scatter` fold, and
/// assert both equal the golden PORDER bit-exact across the full flag fan-out.
#[test]
fn partition_parity_order() {
    let Some(text) = read_partition() else { return };
    let client = cpu_client();
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
        let num_bin = parse_i64(&t, "num_bin") as u32;
        let min_bin = parse_i64(&t, "min_bin") as u32;
        let max_bin = parse_i64(&t, "max_bin") as u32;
        let threshold = parse_i64(&t, "threshold") as u32;
        let most_freq_bin = parse_i64(&t, "most_freq_bin") as u32;
        let missing_type = field(&t, "missing_type").and_then(|s| s.parse::<u8>().ok()).unwrap_or(0);
        let default_bin = field(&t, "default_bin").and_then(|s| s.parse::<u32>().ok()).unwrap_or(0);
        let default_left =
            field(&t, "default_left").and_then(|s| s.parse::<i64>().ok()).unwrap_or(0) != 0;

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

        let col = BinColumn::new(bins.clone(), num_bin);
        let data_indices: Vec<u32> = (0..bins.len() as u32).collect();

        // cpu f64 stable-partition ANCHOR (D-04 CONFIRMED).
        let (got_order, got_split) = partition_leaf_stable(
            &col,
            &data_indices,
            num_bin,
            min_bin,
            max_bin,
            default_bin,
            most_freq_bin,
            missing_type,
            default_left,
            threshold,
        )
        .unwrap_or_else(|e| panic!("PCASE `{name}` anchor errored: {e:?}"));
        compare_exact_u32(&got_order, &order)
            .unwrap_or_else(|m| panic!("PCASE `{name}` anchor PORDER mismatch: {m:?}"));
        assert_eq!(got_split, split_point, "PCASE `{name}` anchor split_point");

        // Full device mark → prefix-sum → scatter fold (still cpu-fold-anchored).
        let (dev_order, dev_split) = partition_on_device(
            &client,
            &col,
            &data_indices,
            num_bin,
            min_bin,
            max_bin,
            default_bin,
            most_freq_bin,
            missing_type,
            default_left,
            threshold,
        )
        .unwrap_or_else(|e| panic!("PCASE `{name}` device errored: {e:?}"));
        compare_exact_u32(&dev_order, &order)
            .unwrap_or_else(|m| panic!("PCASE `{name}` device PORDER mismatch: {m:?}"));
        assert_eq!(dev_split, split_point, "PCASE `{name}` device split_point");

        n_cases += 1;
    }
    assert!(n_cases > 0, "partition fixture present but parsed zero PCASE");
}

/// 28-02 (R2, ODF-02/ODF-05/ODF-06): the device child-range path
/// (`partition_child_ranges_device`) writes each PCASE's child left/right
/// start/end/count into a resident [`DeviceLeafSplits`] slot ON DEVICE; a golden reads
/// them back ONCE and asserts they equal the host-computed split point + ranges across
/// the full missing-type × default-direction PCASE fan-out. The returned permutation is
/// bit-exact to the `partition_leaf_stable` anchor (existing ordering gate re-asserted),
/// and the split point never crosses back as a host scalar (R2 retired).
#[test]
fn partition_child_ranges_device_matches_split_point() {
    let Some(text) = read_partition() else { return };
    let client = cpu_client();
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
        let num_bin = parse_i64(&t, "num_bin") as u32;
        let min_bin = parse_i64(&t, "min_bin") as u32;
        let max_bin = parse_i64(&t, "max_bin") as u32;
        let threshold = parse_i64(&t, "threshold") as u32;
        let most_freq_bin = parse_i64(&t, "most_freq_bin") as u32;
        let missing_type = field(&t, "missing_type").and_then(|s| s.parse::<u8>().ok()).unwrap_or(0);
        let default_bin = field(&t, "default_bin").and_then(|s| s.parse::<u32>().ok()).unwrap_or(0);
        let default_left =
            field(&t, "default_left").and_then(|s| s.parse::<i64>().ok()).unwrap_or(0) != 0;

        let bt: Vec<&str> = lines.next().expect("PBINS").split_whitespace().collect();
        assert_eq!(bt[0], "PBINS", "expected PBINS for `{name}`");
        let bins = parse_u32_list(bt.get(1).copied().unwrap_or(""));
        let ot: Vec<&str> = lines.next().expect("PORDER").split_whitespace().collect();
        assert_eq!(ot[0], "PORDER", "expected PORDER for `{name}`");
        let order = parse_u32_list(ot.get(1).copied().unwrap_or(""));
        let st: Vec<&str> = lines.next().expect("PSPLIT").split_whitespace().collect();
        assert_eq!(st[0], "PSPLIT", "expected PSPLIT for `{name}`");
        let split_point: usize = st[1].parse().expect("split_point usize");

        let n = bins.len();
        if n == 0 {
            continue;
        }
        let col = BinColumn::new(bins.clone(), num_bin);
        let data_indices: Vec<u32> = (0..n as u32).collect();

        // Host anchor split point (D-04) — the value the device ranges must encode.
        let (_anchor_order, anchor_split) = partition_leaf_stable(
            &col, &data_indices, num_bin, min_bin, max_bin, default_bin, most_freq_bin,
            missing_type, default_left, threshold,
        )
        .unwrap_or_else(|e| panic!("PCASE `{name}` anchor errored: {e:?}"));
        assert_eq!(anchor_split, split_point, "PCASE `{name}`: fixture split_point vs anchor");

        // Exercise a NON-zero p_begin + NON-zero leaf_id so the resident-span offset math
        // and the leaf-id indexing (`[6*leaf_id, 6*leaf_id+6)`) are both covered.
        let p_begin = 5i32;
        let leaf_id = 2usize;
        let leaf_splits = DeviceLeafSplits::new(&client, 4).expect("DeviceLeafSplits");
        let reordered = partition_child_ranges_device(
            &client, &col, &data_indices, num_bin, min_bin, max_bin, default_bin, most_freq_bin,
            missing_type, default_left, threshold, &leaf_splits, leaf_id, p_begin, n as i32,
        )
        .unwrap_or_else(|e| panic!("PCASE `{name}` device errored: {e:?}"));

        // The permutation is bit-exact to the golden ordering (stable partition unchanged).
        compare_exact_u32(&reordered, &order)
            .unwrap_or_else(|m| panic!("PCASE `{name}` device PORDER mismatch: {m:?}"));

        // The child ranges (read back ONCE, test-only) equal the host split point + ranges.
        let cr = leaf_splits.read_leaf(&client, leaf_id);
        let sp = split_point as i32;
        assert_eq!(cr.left_start, p_begin, "PCASE `{name}`: left_start");
        assert_eq!(cr.left_end, p_begin + sp, "PCASE `{name}`: left_end");
        assert_eq!(cr.left_count, sp, "PCASE `{name}`: left_count == split_point");
        assert_eq!(cr.right_start, p_begin + sp, "PCASE `{name}`: right_start == left_end");
        assert_eq!(cr.right_end, p_begin + n as i32, "PCASE `{name}`: right_end");
        assert_eq!(cr.right_count, n as i32 - sp, "PCASE `{name}`: right_count");

        // A leaf slot that was never written stays zeroed (no cross-slot bleed).
        let untouched = leaf_splits.read_leaf(&client, 0);
        assert_eq!(
            [untouched.left_start, untouched.left_end, untouched.left_count],
            [0, 0, 0],
            "PCASE `{name}`: untouched leaf slot 0 must stay zeroed"
        );

        n_cases += 1;
    }
    assert!(n_cases > 0, "partition fixture present but parsed zero PCASE");
}

/// `cat` cell (ODL-13, categorical membership routing, D-03).
mod cat {
    use super::*;

    /// Parse every PCAT block, drive the categorical anchor + device fold
    /// (`FindInBitset` membership), and compare bit-exact to the golden PCATORDER.
    #[test]
    fn partition_parity_cat_membership() {
        let Some(text) = read_partition() else { return };
        let client = cpu_client();
        let mut lines = text.lines();
        let mut n_cases = 0;
        while let Some(raw) = lines.next() {
            let line = raw.trim();
            let t: Vec<&str> = line.split_whitespace().collect();
            if t.is_empty() || t[0] != "PCAT" {
                continue;
            }
            let name = field(&t, "name").expect("PCAT name").to_string();
            let num_bin = parse_i64(&t, "num_bin") as u32;
            let min_bin = parse_i64(&t, "min_bin") as u32;
            let max_bin = parse_i64(&t, "max_bin") as u32;
            let most_freq_bin = parse_i64(&t, "most_freq_bin") as u32;

            let bs: Vec<&str> = lines.next().expect("PCATBITSET").split_whitespace().collect();
            assert_eq!(bs[0], "PCATBITSET", "expected PCATBITSET for `{name}`");
            let bitset = parse_u32_list(bs.get(1).copied().unwrap_or(""));
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

            let col = BinColumn::new(bins.clone(), num_bin);
            let data_indices: Vec<u32> = (0..bins.len() as u32).collect();

            let (got_order, got_split) =
                partition_categorical_stable(&col, &data_indices, num_bin, min_bin, max_bin, most_freq_bin, &bitset)
                    .unwrap_or_else(|e| panic!("PCAT `{name}` anchor errored: {e:?}"));
            compare_exact_u32(&got_order, &order)
                .unwrap_or_else(|m| panic!("PCAT `{name}` anchor PCATORDER mismatch: {m:?}"));
            assert_eq!(got_split, split_point, "PCAT `{name}` anchor split_point");

            let (dev_order, dev_split) = partition_categorical_on_device(
                &client, &col, &data_indices, num_bin, min_bin, max_bin, most_freq_bin, &bitset,
            )
            .unwrap_or_else(|e| panic!("PCAT `{name}` device errored: {e:?}"));
            compare_exact_u32(&dev_order, &order)
                .unwrap_or_else(|m| panic!("PCAT `{name}` device PCATORDER mismatch: {m:?}"));
            assert_eq!(dev_split, split_point, "PCAT `{name}` device split_point");

            n_cases += 1;
        }
        assert!(n_cases > 0, "partition fixture present but parsed zero PCAT");
    }
}

/// `packet` cell (ODL-13, 16-int SplitTreeStructure child-stats packet, D-08).
mod packet {
    use super::*;

    /// Parse every PPACKET block, reconstruct the 16-int packet
    /// (`split_tree_structure_packet`), and compare field-for-field (8 ints exact +
    /// the 4 f64 sums via `compare_exact_f64_bits`).
    #[test]
    fn partition_parity_packet_fields() {
        let Some(text) = read_partition() else { return };
        let mut n_cases = 0;
        for raw in text.lines() {
            let t: Vec<&str> = raw.split_whitespace().collect();
            if t.is_empty() || t[0] != "PPACKET" {
                continue;
            }
            let name = field(&t, "name").expect("PPACKET name").to_string();
            let left_leaf = parse_i64(&t, "left_leaf") as i32;
            let right_leaf = parse_i64(&t, "right_leaf") as i32;
            let left_num = parse_i64(&t, "left_num_data") as i32;
            let right_num = parse_i64(&t, "right_num_data") as i32;
            let left_start = parse_i64(&t, "left_data_start") as i32;
            let right_start = parse_i64(&t, "right_data_start") as i32;
            let smaller = parse_i64(&t, "smaller") as i32;
            let larger = parse_i64(&t, "larger") as i32;
            let lh = parse_f64_bits(field(&t, "left_sum_hessians").expect("lh"));
            let rh = parse_f64_bits(field(&t, "right_sum_hessians").expect("rh"));
            let lg = parse_f64_bits(field(&t, "left_sum_gradients").expect("lg"));
            let rg = parse_f64_bits(field(&t, "right_sum_gradients").expect("rg"));

            // Reconstruct the packet from the CUDASplitInfo per-side sums (D-08).
            let scalars = SplitScalars {
                left_sum_hessians: lh,
                right_sum_hessians: rh,
                left_sum_gradients: lg,
                right_sum_gradients: rg,
                ..SplitScalars::default()
            };
            let pk = split_tree_structure_packet(
                left_leaf, left_num, left_start, right_leaf, right_num, right_start, &scalars,
            );

            assert_eq!(
                pk.ints,
                [left_leaf, left_num, left_start, right_leaf, right_num, right_start, smaller, larger],
                "PPACKET `{name}`: 16-int packet ints mismatch"
            );
            compare_exact_f64_bits(&pk.sums, &[lh, rh, lg, rg])
                .unwrap_or_else(|m| panic!("PPACKET `{name}`: packet f64 sums mismatch: {m:?}"));
            n_cases += 1;
        }
        assert!(n_cases > 0, "partition fixture present but parsed zero PPACKET");
    }
}
