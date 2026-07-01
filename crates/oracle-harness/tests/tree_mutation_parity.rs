//! On-device tree-mutation golden-anchor parity replay (Phase 18, ODL-14).
//!
//! Mirrors `oracle-harness/tests/best_split_parity.rs`: parse the committed
//! `tests/fixtures/kernels/split.txt` golden (the Phase-4 `SCASE`/`SWIN`
//! `CUDASplitInfo` records — reused per the RESEARCH test map) and drive the
//! device flat `CUDATree` `SplitKernel`, asserting the 14 mutated node fields are
//! BIT-EXACT vs the golden `SWIN` record. A second cell asserts the hard §1/§10
//! ordering invariant (Pitfall 3): `CUDATree.Split` runs BEFORE the partition and
//! returns the `right_leaf_index` the partition launch consumes. Idioms follow
//! `best_split_parity.rs`: `CARGO_MANIFEST_DIR` fixture path (never the untracked
//! `LightGBM/` tree), raw-f64-bits parse, graceful SKIP, localizing assert.
//!
//! ## Wave-0 status (18-01)
//! The device `SplitKernel` / `ShrinkageKernel` / `AddBiasKernel` land in **18-03**
//! (Wave 1). Until then there is no device tree to mutate, so each cell parses +
//! structurally validates its golden and is marked
//! `#[ignore = "Wave-0 scaffold; un-ignore when 18-03 lands"]` so the merge-gate
//! `cargo test --workspace` stays GREEN with `LGBM_CUDA_ON_DEVICE` unset
//! (D-13 / ODL-19). 18-03 replaces each `// UN-IGNORE (18-03):` block with the
//! real device call + `compare_exact_f64_bits` and removes the `#[ignore]`.

use std::path::PathBuf;

fn kernels_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/kernels")
}

fn parse_f64_bits(s: &str) -> f64 {
    f64::from_bits(s.parse::<u64>().expect("f64-bits u64 field"))
}

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

fn parse_f64_field(tokens: &[&str], key: &str) -> f64 {
    parse_f64_bits(field(tokens, key).unwrap_or_else(|| panic!("missing f64-bits field `{key}`")))
}

fn read_split() -> Option<String> {
    let path = kernels_dir().join("split.txt");
    match std::fs::read_to_string(&path) {
        Ok(s) => Some(s),
        Err(_) => {
            eprintln!("tree_mutation_parity: SKIP — fixture {} not found.", path.display());
            None
        }
    }
}

/// SplitKernel field-write cell (ODL-14): parse every `SWIN` winner record and
/// validate the fields the device `SplitKernel` writes into the flat `CUDATree`
/// (threshold, default_left, per-side sums/outputs, NaN→0). 18-03 un-ignores and
/// drives the device tree mutation, comparing bit-exact.
#[test]
#[ignore = "Wave-0 scaffold; un-ignore when 18-03 lands"]
fn tree_mutation_parity_split_field_writes() {
    let Some(text) = read_split() else { return };
    let mut lines = text.lines();
    let mut n_cases = 0;
    while let Some(raw) = lines.next() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let t: Vec<&str> = line.split_whitespace().collect();
        if t[0] != "SCASE" {
            continue;
        }
        let name = field(&t, "name").expect("SCASE name").to_string();
        // SHIST / SCAND_REV / SCAND_FWD precede the SWIN winner record.
        let mut win: Option<Vec<String>> = None;
        for next in lines.by_ref() {
            let nt: Vec<&str> = next.trim().split_whitespace().collect();
            if nt.is_empty() {
                continue;
            }
            if nt[0] == "SWIN" {
                win = Some(nt.iter().map(|s| s.to_string()).collect());
                break;
            }
            if nt[0] == "SCASE" {
                panic!("SCASE `{name}`: reached next SCASE before SWIN");
            }
        }
        let win = win.unwrap_or_else(|| panic!("SCASE `{name}`: no SWIN record"));
        let wt: Vec<&str> = win.iter().map(String::as_str).collect();

        let is_splittable = parse_i64(&wt, "is_splittable") != 0;
        if is_splittable {
            // The 14 CUDASplitInfo fields SplitKernel writes into the device tree.
            let _threshold = parse_i64(&wt, "threshold");
            let _default_left = parse_i64(&wt, "default_left") != 0;
            let _left_output = parse_f64_field(&wt, "left_output");
            let _right_output = parse_f64_field(&wt, "right_output");
            let _left_sum_gradient = parse_f64_field(&wt, "left_sum_gradient");
            let _left_sum_hessian = parse_f64_field(&wt, "left_sum_hessian");
            let _right_sum_gradient = parse_f64_field(&wt, "right_sum_gradient");
            let _right_sum_hessian = parse_f64_field(&wt, "right_sum_hessian");
            let gain = parse_f64_field(&wt, "gain");
            assert!(gain.is_finite(), "SWIN `{name}`: splittable winner gain must be finite");

            // UN-IGNORE (18-03): mutate the device flat CUDATree and compare the node
            // fields bit-exact, e.g.
            //   let node = lgbm_compute::kernels::tree::split_on_device(&client,
            //       &mut cuda_tree, leaf, &SplitScalars { threshold, default_left, .. })?;
            //   oracle_harness::comparator::compare_exact_f64_bits(
            //       &[cuda_tree.leaf_value(node.left), cuda_tree.leaf_value(node.right)],
            //       &[left_output, right_output])?;
        }
        n_cases += 1;
    }
    assert!(n_cases > 0, "split fixture present but parsed zero SCASE");
}

/// Split-before-partition ordering cell (ODL-14, §1/§10 hard invariant, Pitfall 3):
/// the device `SplitKernel` returns the `right_leaf_index` the partition launch
/// consumes; a partition run before the tree mutation would consume a stale leaf
/// id. 18-03 un-ignores and asserts the returned index feeds the partition.
#[test]
#[ignore = "Wave-0 scaffold; un-ignore when 18-03 lands"]
fn tree_mutation_parity_split_before_partition_ordering() {
    let Some(_text) = read_split() else { return };

    // UN-IGNORE (18-03): drive the ordered per-split device sequence and assert the
    // returned right_leaf_index is the one fed to the partition launch, e.g.
    //   let right_leaf_index = lgbm_compute::kernels::tree::split_on_device(..)?.right;
    //   let (_order, _split) = lgbm_compute::kernels::data_partition::partition_on_device(
    //       &client, .., left_leaf_index, right_leaf_index, ..)?;
    //   assert_eq!(right_leaf_index, expected_new_leaf_id,
    //       "SplitKernel must return the leaf id the partition consumes");
    //
    // Wave-0: the invariant is asserted structurally — the device split entry point
    // (`tree::split_on_device`) returns `right_leaf_index` and the partition entry
    // point (`data_partition::partition_on_device`) takes it as a parameter, so the
    // ordering is enforced by the type signature once both land in 18-02/18-03.
}
