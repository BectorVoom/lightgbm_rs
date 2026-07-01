//! On-device tree-mutation golden-anchor parity replay (Phase 18, ODL-14).
//!
//! Mirrors `oracle-harness/tests/best_split_parity.rs`: parse the committed
//! `tests/fixtures/kernels/split.txt` golden (the Phase-4 `SCASE`/`SWIN`
//! `CUDASplitInfo` records — reused per the RESEARCH test map) and drive the
//! device flat `CUDATree` `SplitKernel`
//! ([`lgbm_compute::kernels::tree::DeviceCudaTree::split_on_device`]), asserting the
//! mutated node/leaf fields are BIT-EXACT vs the golden `SWIN` record AND vs a
//! plain-Rust host mirror (the cpu f64 anchor, D-12 — never GPU-vs-GPU). A second
//! cell asserts the hard §1/§10 ordering invariant (Pitfall 3): `CUDATree.Split`
//! runs BEFORE the partition and returns the `right_leaf_index` the partition launch
//! consumes. A third cell drives the categorical `SplitCategoricalKernel` fan-out
//! (kCategoricalMask / num_cat / cat_boundaries append) vs the host mirror.
//! Idioms follow `best_split_parity.rs`: `CARGO_MANIFEST_DIR` fixture path (never the
//! untracked `LightGBM/` tree), raw-f64-bits parse, graceful SKIP, localizing assert.

use std::path::PathBuf;

use lgbm_compute::kernels::split_info::SplitScalars;
use lgbm_compute::kernels::tree::{get_decision_type, DeviceCudaTree, K_CATEGORICAL_MASK, K_DEFAULT_LEFT_MASK};
use lgbm_compute::runtime::cpu_client;
use oracle_harness::comparator::compare_exact_f64_bits;

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

/// The `SWIN` winner fields the `SplitKernel` writes into the flat `CUDATree`.
struct Winner {
    threshold: u32,
    default_left: bool,
    left_output: f64,
    right_output: f64,
    left_sum_hessian: f64,
    right_sum_hessian: f64,
    left_count: i32,
    right_count: i32,
    gain: f64,
}

/// Build a [`SplitScalars`] from a parsed `SWIN` winner (the `CUDASplitInfo` fields
/// the device `SplitKernel` consumes). `inner_feature_index` is a synthetic test id.
fn scalars_from(w: &Winner, inner_feature_index: i32) -> SplitScalars {
    SplitScalars {
        is_valid: true,
        leaf_index: 0,
        gain: w.gain,
        inner_feature_index,
        threshold: w.threshold,
        default_left: w.default_left,
        left_sum_gradients: 0.0,
        left_sum_hessians: w.left_sum_hessian,
        left_sum_gh_quant: 0,
        left_count: w.left_count,
        left_gain: 0.0,
        left_value: w.left_output,
        right_sum_gradients: 0.0,
        right_sum_hessians: w.right_sum_hessian,
        right_sum_gh_quant: 0,
        right_count: w.right_count,
        right_gain: 0.0,
        right_value: w.right_output,
        num_cat_threshold: 0,
    }
}

/// Iterate every `SCASE`/`SWIN` pair in `split.txt`, yielding the winner record for
/// each splittable case.
fn for_each_winner(text: &str, mut visit: impl FnMut(&str, &Winner)) -> usize {
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
        let mut win: Option<Vec<String>> = None;
        for next in lines.by_ref() {
            let nt: Vec<&str> = next.split_whitespace().collect();
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

        if parse_i64(&wt, "is_splittable") != 0 {
            let w = Winner {
                threshold: parse_i64(&wt, "threshold") as u32,
                default_left: parse_i64(&wt, "default_left") != 0,
                left_output: parse_f64_field(&wt, "left_output"),
                right_output: parse_f64_field(&wt, "right_output"),
                left_sum_hessian: parse_f64_field(&wt, "left_sum_hessian"),
                right_sum_hessian: parse_f64_field(&wt, "right_sum_hessian"),
                left_count: parse_i64(&wt, "left_count") as i32,
                right_count: parse_i64(&wt, "right_count") as i32,
                gain: parse_f64_field(&wt, "gain"),
            };
            assert!(w.gain.is_finite(), "SWIN `{name}`: splittable winner gain must be finite");
            visit(&name, &w);
            n_cases += 1;
        }
    }
    n_cases
}

/// `isnan(x) ? 0.0 : x` — the reference NaN→0 leaf-output coercion (`cuda_tree.cu:100`).
fn nan_to_zero(x: f64) -> f64 {
    if x.is_nan() {
        0.0
    } else {
        x
    }
}

/// SplitKernel field-write cell (ODL-14): drive the device `DeviceCudaTree.Split`
/// per `SWIN` winner and assert every mutated tree field is BIT-EXACT vs the golden
/// `SWIN` record (leaf outputs NaN→0) AND vs a plain-Rust host mirror.
#[test]
fn tree_mutation_parity_split_field_writes() {
    let Some(text) = read_split() else { return };
    let client = cpu_client();

    let n_cases = for_each_winner(&text, |name, w| {
        let missing_type = 0i32; // na_as_missing=0 goldens → MissingType::None.
        let real_feature = 7i32;
        let real_threshold = 1.5f64;
        let inner_feature = 3i32;
        let root_count = w.left_count + w.right_count;
        let scalars = scalars_from(w, inner_feature);

        let mut tree = DeviceCudaTree::new(&client, 2, root_count).expect("device tree");
        // Single allocation site (D-15): NUM_TREE_BUFFERS buffers, allocated once.
        let result = tree
            .split_on_device(&client, 0, real_feature, real_threshold, missing_type, &scalars)
            .expect("split_on_device");
        assert_eq!(result.new_node_index, 0, "case `{name}`: new node index");
        assert_eq!(result.left_leaf_index, 0, "case `{name}`: left leaf id");
        assert_eq!(result.right_leaf_index, 1, "case `{name}`: right leaf id");

        let host = tree.to_host_tree(&client);

        // Leaf outputs bit-exact vs the golden (NaN→0).
        compare_exact_f64_bits(
            &[host.leaf_value[0], host.leaf_value[1]],
            &[nan_to_zero(w.left_output), nan_to_zero(w.right_output)],
        )
        .unwrap_or_else(|m| panic!("case `{name}`: leaf_value vs golden: {m:?}"));

        // Per-side hessian weights bit-exact vs the golden.
        compare_exact_f64_bits(
            &[host.leaf_weight[0], host.leaf_weight[1], host.internal_weight[0]],
            &[w.left_sum_hessian, w.right_sum_hessian, w.left_sum_hessian + w.right_sum_hessian],
        )
        .unwrap_or_else(|m| panic!("case `{name}`: weights vs golden: {m:?}"));

        // Integer / structural fields.
        assert_eq!(host.threshold_in_bin[0], w.threshold, "case `{name}`: threshold_in_bin");
        assert_eq!(host.leaf_count[0], w.left_count, "case `{name}`: left leaf_count");
        assert_eq!(host.leaf_count[1], w.right_count, "case `{name}`: right leaf_count");
        assert_eq!(
            host.internal_count[0],
            w.left_count + w.right_count,
            "case `{name}`: internal_count"
        );
        assert_eq!(host.left_child[0], -1, "case `{name}`: left_child (~0)");
        assert_eq!(host.right_child[0], -2, "case `{name}`: right_child (~1)");
        assert_eq!(host.split_feature[0], real_feature, "case `{name}`: split_feature");
        assert_eq!(host.split_feature_inner[0], inner_feature, "case `{name}`: split_feature_inner");
        assert_eq!(host.split_gain[0], w.gain as f32, "case `{name}`: split_gain (f32)");
        assert_eq!(host.leaf_parent[0], 0, "case `{name}`: left leaf_parent");
        assert_eq!(host.leaf_parent[1], 0, "case `{name}`: right leaf_parent");
        assert_eq!(host.leaf_depth[0], 1, "case `{name}`: left leaf_depth");
        assert_eq!(host.leaf_depth[1], 1, "case `{name}`: right leaf_depth");

        // decision_type: default_left bit matches the golden, categorical bit clear.
        assert_eq!(
            get_decision_type(host.decision_type[0], K_DEFAULT_LEFT_MASK as i8),
            w.default_left,
            "case `{name}`: default_left bit"
        );
        assert!(
            !get_decision_type(host.decision_type[0], K_CATEGORICAL_MASK as i8),
            "case `{name}`: categorical bit must be clear for a numeric split"
        );

        // cpu f64 anchor (D-12): the device reconstruction equals a plain-Rust mirror
        // of the exact SplitKernel field writes, structure bit-exact.
        let anchor = host_split_mirror(root_count, real_feature, inner_feature, real_threshold, missing_type, &scalars);
        assert_host_tree_eq(name, &host, &anchor);
    });

    assert!(n_cases > 0, "split fixture present but parsed zero splittable SCASE");
}

/// Split-before-partition ordering cell (ODL-14, §1/§10 hard invariant, Pitfall 3):
/// the device `SplitKernel` returns the `right_leaf_index` the partition launch
/// consumes. A partition run before the tree mutation would consume a stale leaf id.
#[test]
fn tree_mutation_parity_split_before_partition_ordering() {
    let Some(text) = read_split() else { return };
    let client = cpu_client();

    let n_cases = for_each_winner(&text, |name, w| {
        let root_count = w.left_count + w.right_count;
        let scalars = scalars_from(w, 0);
        let mut tree = DeviceCudaTree::new(&client, 2, root_count).expect("device tree");

        // 1) Split runs FIRST and yields the new leaf id.
        let result = tree
            .split_on_device(&client, 0, 0, 0.0, 0, &scalars)
            .expect("split_on_device");
        let right_leaf_index = result.right_leaf_index;
        assert_eq!(right_leaf_index, 1, "case `{name}`: root split yields right leaf id 1");

        // 2) The partition step THEN consumes exactly that id. Mark two rows (one
        //    left, one right) and assert the right-routed row takes right_leaf_index.
        let data_indices = vec![0u32, 1u32];
        let to_left = vec![1u32, 0u32]; // row 0 → left leaf, row 1 → right leaf.
        let leaf_map = lgbm_compute::kernels::data_partition::update_data_index_to_leaf_on(
            &client,
            &data_indices,
            &to_left,
            2,
            result.left_leaf_index,
            right_leaf_index,
        )
        .expect("update_data_index_to_leaf");
        assert_eq!(leaf_map[0], result.left_leaf_index, "case `{name}`: left row leaf id");
        assert_eq!(
            leaf_map[1], right_leaf_index,
            "case `{name}`: right row consumes the SplitKernel-produced right_leaf_index"
        );
    });

    assert!(n_cases > 0, "split fixture present but parsed zero splittable SCASE");
}

/// Categorical-split field-write cell (ODL-14): drive `SplitCategoricalKernel` and
/// assert the kCategoricalMask / num_cat / threshold / cat_boundaries-append fields
/// vs a plain-Rust host mirror (cpu f64 anchor, D-12). No captured categorical tree
/// golden exists (split.txt is numeric), so the anchor is the host fan-out mirror.
#[test]
fn tree_mutation_parity_split_categorical_field_writes() {
    let client = cpu_client();

    // Synthetic categorical winner + a real/inner bitset (2 blocks / 1 block).
    let scalars = SplitScalars {
        is_valid: true,
        leaf_index: 0,
        gain: 12.5,
        inner_feature_index: 4,
        threshold: 0,
        default_left: false,
        left_sum_gradients: 0.0,
        left_sum_hessians: 3.0,
        left_sum_gh_quant: 0,
        left_count: 6,
        left_gain: 0.0,
        left_value: -0.5,
        right_sum_gradients: 0.0,
        right_sum_hessians: 4.0,
        right_sum_gh_quant: 0,
        right_count: 4,
        right_gain: 0.0,
        right_value: f64::NAN, // exercises the NaN→0 coercion on the right leaf.
        num_cat_threshold: 2,
    };
    let bitset = vec![0b101u32, 0b1u32]; // len 2 → cat_boundaries[1] = 2.
    let bitset_inner = vec![0b11u32]; // len 1 → cat_boundaries_inner[1] = 1.

    let mut tree = DeviceCudaTree::new(&client, 2, 10).expect("device tree");
    let result = tree
        .split_categorical_on_device(&client, 0, 9, 0, &scalars, &bitset, &bitset_inner)
        .expect("split_categorical_on_device");
    assert_eq!(result.right_leaf_index, 1);

    let host = tree.to_host_tree(&client);

    // Categorical bit set; default_left NOT encoded.
    assert!(
        get_decision_type(host.decision_type[0], K_CATEGORICAL_MASK as i8),
        "categorical bit must be set"
    );
    assert_eq!(host.num_cat, 1, "num_cat incremented");
    // threshold_in_bin / threshold both == num_cat (0 for the first categorical node).
    assert_eq!(host.threshold_in_bin[0], 0, "threshold_in_bin == num_cat");
    assert_eq!(host.threshold[0].to_bits(), 0.0f64.to_bits(), "threshold == num_cat as f64");
    // cat_boundaries appended by the bitset lengths (seeded 0 at num_cat==0).
    assert_eq!(host.cat_boundaries, vec![0, 2], "cat_boundaries append == bitset len");
    // Right leaf output NaN→0.
    compare_exact_f64_bits(&[host.leaf_value[1]], &[0.0]).expect("right leaf NaN→0");
    // Left leaf output written through.
    compare_exact_f64_bits(&[host.leaf_value[0]], &[-0.5]).expect("left leaf output");
    assert_eq!(host.split_feature[0], 9, "split_feature");
    assert_eq!(host.left_child[0], -1, "left_child (~0)");
    assert_eq!(host.right_child[0], -2, "right_child (~1)");
}

// =========================================================================
// Plain-Rust host mirror of the SplitKernel field writes — the cpu f64 anchor
// (D-12; never GPU-vs-GPU). Reproduces `cuda_tree.cu:48-138` on a single-leaf
// tree, producing the SAME `lgbm_model::Tree` the device reconstruction yields.
// =========================================================================

fn host_split_mirror(
    root_count: i32,
    real_feature: i32,
    inner_feature: i32,
    real_threshold: f64,
    missing_type: i32,
    s: &SplitScalars,
) -> lgbm_model::Tree {
    // Fresh single-leaf root (leaf 0, value 0, count root_count, parent -1).
    let mut t = lgbm_model::Tree::as_constant(0.0, root_count);
    // decision_type: clear categorical, set default_left, set missing.
    let mut dt: i8 = 0;
    if s.default_left {
        dt |= K_DEFAULT_LEFT_MASK as i8;
    }
    dt = (dt & 3) | ((missing_type as i8 & 3) << 2);

    // Append the new internal node (node 0).
    t.split_feature_inner.push(inner_feature);
    t.split_feature.push(real_feature);
    t.split_gain.push(s.gain as f32);
    t.internal_weight.push(s.left_sum_hessians + s.right_sum_hessians);
    t.internal_value.push(t.leaf_value[0]);
    t.internal_count.push(s.left_count + s.right_count);
    t.left_child.push(-1); // ~0
    t.right_child.push(-2); // ~1
    t.decision_type.push(dt);
    t.threshold_in_bin.push(s.threshold);
    t.threshold.push(real_threshold);

    // Left child == old leaf 0.
    t.leaf_value[0] = nan_to_zero(s.left_value);
    t.leaf_weight[0] = s.left_sum_hessians;
    t.leaf_count[0] = s.left_count;
    t.leaf_depth[0] += 1;
    t.leaf_parent[0] = 0;

    // Right child == new leaf 1.
    t.leaf_value.push(nan_to_zero(s.right_value));
    t.leaf_weight.push(s.right_sum_hessians);
    t.leaf_count.push(s.right_count);
    t.leaf_depth.push(1);
    t.leaf_parent.push(0);

    t.num_leaves = 2;
    t
}

fn assert_host_tree_eq(name: &str, got: &lgbm_model::Tree, want: &lgbm_model::Tree) {
    assert_eq!(got.num_leaves, want.num_leaves, "case `{name}`: num_leaves");
    assert_eq!(got.left_child, want.left_child, "case `{name}`: left_child");
    assert_eq!(got.right_child, want.right_child, "case `{name}`: right_child");
    assert_eq!(got.split_feature, want.split_feature, "case `{name}`: split_feature");
    assert_eq!(got.split_feature_inner, want.split_feature_inner, "case `{name}`: split_feature_inner");
    assert_eq!(got.decision_type, want.decision_type, "case `{name}`: decision_type");
    assert_eq!(got.threshold_in_bin, want.threshold_in_bin, "case `{name}`: threshold_in_bin");
    assert_eq!(got.leaf_count, want.leaf_count, "case `{name}`: leaf_count");
    assert_eq!(got.internal_count, want.internal_count, "case `{name}`: internal_count");
    assert_eq!(got.leaf_parent, want.leaf_parent, "case `{name}`: leaf_parent");
    assert_eq!(got.leaf_depth, want.leaf_depth, "case `{name}`: leaf_depth");
    // f64/f32 fields compared by raw bits (bit-exact anchor).
    let bits64 = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let bits32 = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits64(&got.leaf_value), bits64(&want.leaf_value), "case `{name}`: leaf_value bits");
    assert_eq!(bits64(&got.leaf_weight), bits64(&want.leaf_weight), "case `{name}`: leaf_weight bits");
    assert_eq!(bits64(&got.threshold), bits64(&want.threshold), "case `{name}`: threshold bits");
    assert_eq!(bits64(&got.internal_value), bits64(&want.internal_value), "case `{name}`: internal_value bits");
    assert_eq!(bits64(&got.internal_weight), bits64(&want.internal_weight), "case `{name}`: internal_weight bits");
    assert_eq!(bits32(&got.split_gain), bits32(&want.split_gain), "case `{name}`: split_gain bits");
}
