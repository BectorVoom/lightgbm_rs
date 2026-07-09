//! Phase-29 29-04 (OCX-04) — the on-device NON-FINITE TRIPWIRE + ZERO-BIN CANARY gates.
//!
//! # Why this file exists (spike-072 Results 5, method takeaway (d))
//!
//! Two coverage holes let the spike-072 f32-root-fold corruption ship on the opt-in
//! on-device path (`LGBM_CUDA_ON_DEVICE=1`):
//!
//! 1. **No loud failure on NaN.** GBDT's C++-faithful no-split POP-LOOP turns ONE NaN
//!    boosting iteration into a SILENTLY-truncated model (the observed 13-tree collapse):
//!    a NaN root grad/hess sum makes the root unscannable, the loop emits a 1-leaf tree,
//!    GBDT stops, and the DEF-07-13-01 pop-loop repeats the stop — nothing failed loudly.
//!    An explicit NaN guard would have surfaced the defect five phases earlier.
//!
//! 2. **No gate watched the forced-empty zero/most-frequent bin.** On a no-exact-zeros
//!    corpus, `find_bin_with_zero_as_one_bin` carves a dedicated bin for value 0.0 (which
//!    NO row ever hits ⇒ 0 rows globally), and the `sparse_rate < 0.7` override forces
//!    `most_freq_bin = default_bin` onto that empty bin (spike-072 items 9–11). Because the
//!    on-device FixHistogram reconstructs the mfb cell, ANY seed-vs-build disagreement lands
//!    THERE FIRST — it is a PERFECT seed-vs-build-disagreement detector. 29-03 (OCX-03)
//!    root-caused + fixed the residue that polluted it; this canary ARMS that cell as a
//!    committed gate so any FUTURE disagreement fails a named test.
//!
//! # What this file lands (Tasks 1 + 2)
//!
//! - **Task 1 — the non-finite tripwire** (driver-side, `grow_driver.rs`): a NaN root seed
//!   or a non-finite emitted leaf value now returns a typed `ComputeError::NonFinite`, which
//!   the learner seam propagates (`?`) as a training failure instead of a truncated
//!   `Ok(tree)`. Tested here on the cpu ANCHOR lane (the driver's f64 arm). The tripwire
//!   lives ONLY on the on-device grow return path — GBDT's pop-loop (in `lgbm-boosting`) is
//!   NOT touched (29-CONTEXT locked rule; its bit-faithful semantics are the merge gate).
//!
//! - **Task 2 — the zero-bin canary**: on a forced-empty-mfb corpus (precondition ASSERTED:
//!   the mfb bin has 0 global rows — the canary self-validates its trap is armed), the mfb
//!   cell stays EXACTLY 0 across every resident-histogram slot class the on-device grow
//!   produces — root build, subtract-derived larger child, and both co-pack sibling slots —
//!   plus down a subtract SPINE (num_leaves ≥ 8 spirit). A failure here means seed-vs-build
//!   disagreement re-entered the resident chain (see `29-REDUCTION-AUDIT.md` /
//!   `29-SUBTRACT-RESIDUE.md`). NON-UNIFORM (binary-logloss-like) hessians make the canary
//!   sensitive to the exact f32-bias class that hid at uniform 0.25 (spike-072 item 13).
//!
//! # Two lanes (mirrors `on_device_integer_anchor.rs` / `on_device_subtract_residue.rs`)
//!   - **cpu ANCHOR lane (always runnable):** the driver's f64 fold arm + the generic-over-R
//!     `construct_histograms_f64_on` / `subtract_histograms_f64_on` on the cubecl-cpu client.
//!     Canary tolerance is EXACTLY 0.0 (the f64 fold never touches a globally-empty bin, and
//!     `x − x = +0.0` for identical zero cells — the ordered-f64/integer-exact anchor chain).
//!   - **rocm RESIDENT lane (`#[cfg(feature = "rocm")]`, hardware-only):** the SAME chain on
//!     the hip client, PLUS `fix_compact_f64_on` (the resident u64 FixHistogram 29-03 fixed),
//!     at the 29-03 disposition bound (exactly 0 for a forced-empty cell at all scales).

use lgbm_compute::kernels::grow_driver::grow_tree_on_device_driver;
use lgbm_compute::kernels::histogram::construct_histograms_f64_on;
use lgbm_compute::kernels::subtract::subtract_histograms_f64_on;
use lgbm_compute::runtime::cpu_client;
use lgbm_compute::{BinColumn, ComputeError, CpuBackend, GrowFeature};
use lgbm_dataset::bin_mapper::{BinType, MissingType};

/// `offset_for_most_freq_bin(0)` == 1 (drop bin 0 under the compacted convention). The helper
/// lives in `lgbm-treelearner` (ABOVE this crate); hardcode it (as `on_device_sync_count.rs`).
const OFFSET_MFB_0: i32 = 1;

// ===========================================================================
// Task 1 — the NON-FINITE TRIPWIRE (cpu anchor lane)
// ===========================================================================

/// A dominant-feature numeric corpus that grows a real multi-leaf tree on the anchor arm:
/// feature 0's bin rises monotonically with the row index so it dominates every split.
/// Gradients are non-uniform (monotone) and hessians are binary-logloss-like (non-uniform)
/// so the corpus resembles a real second-iteration grow, not a degenerate uniform one.
fn dominant_corpus(num_features: usize, num_bin: u32, num_data: usize) -> (Vec<GrowFeature>, Vec<f32>, Vec<f32>) {
    let features: Vec<GrowFeature> = (0..num_features)
        .map(|fi| {
            let bins: Vec<u32> = (0..num_data)
                .map(|r| (((r + fi * 37) * num_bin as usize / num_data) as u32) % num_bin)
                .collect();
            GrowFeature {
                bins: BinColumn::new(bins, num_bin),
                num_bin,
                offset: OFFSET_MFB_0,
                min_bin: 0,
                max_bin: num_bin - 1,
                default_bin: num_bin,
                most_freq_bin: 0,
                missing_type: MissingType::None,
                bin_upper_bound: (0..num_bin).map(|b| b as f64 + 0.5).collect(),
                real_feature_index: fi as i32,
                bin_type: BinType::Numerical,
                bin_to_category: Vec::new(),
                cat_smooth: 10.0,
                cat_l2: 10.0,
                max_cat_threshold: 32,
                max_cat_to_onehot: 4,
                min_data_per_group: 100,
            }
        })
        .collect();
    // Non-uniform gradients + binary-logloss-like non-uniform hessians h = p(1-p).
    let half = num_data as f32 / 2.0;
    let gradients: Vec<f32> = (0..num_data).map(|r| (r as f32 - half) * 0.1).collect();
    let hessians: Vec<f32> = (0..num_data)
        .map(|r| {
            let p = 0.05 + 0.9 * (r as f32 / num_data as f32);
            p * (1.0 - p)
        })
        .collect();
    (features, gradients, hessians)
}

/// Task 1 — NaN-injection: growing on the on-device arm with a NaN gradient FAILS LOUDLY
/// with a typed `ComputeError::NonFinite` naming the non-finite quantity — no `Ok(tree)`
/// escapes. This is the spike-072 iter-13 signature (a NaN grad/hess seed) that GBDT's
/// pop-loop would otherwise turn into a silently-truncated model.
#[test]
fn nan_gradient_grow_fails_loudly() {
    let client = cpu_client();
    let (features, mut gradients, hessians) = dominant_corpus(3, 64, 512);
    // Inject a single NaN gradient — the root f64 fold sums to NaN (the iter-13 fingerprint).
    gradients[123] = f32::NAN;

    let result =
        grow_tree_on_device_driver(&CpuBackend, &client, &gradients, &hessians, &features, 8, -1);

    match result {
        Err(ComputeError::NonFinite { detail }) => {
            assert!(
                detail.contains("root") || detail.contains("leaf"),
                "the tripwire error must name the non-finite quantity (root seed / leaf): {detail}"
            );
        }
        Err(other) => panic!("expected a loud NonFinite tripwire error, got a different error: {other:?}"),
        Ok((tree, _)) => panic!(
            "a NaN gradient produced a silent Ok(tree) with {} leaves — the pop-loop \
             would have truncated the model (OCX-04 tripwire failed to fire)",
            tree.num_leaves
        ),
    }
}

/// Task 1 — healthy path: a normal grow is UNAFFECTED by the read-only tripwire. It returns
/// `Ok`, grows a real multi-leaf tree, and every emitted leaf value is finite (the tripwire
/// never mutates the tree, so the output is byte-identical to the pre-tripwire driver).
#[test]
fn healthy_grow_returns_finite_tree() {
    let client = cpu_client();
    let (features, gradients, hessians) = dominant_corpus(3, 64, 512);

    let (tree, _layout) =
        grow_tree_on_device_driver(&CpuBackend, &client, &gradients, &hessians, &features, 8, -1)
            .expect("a healthy grow must return Ok — the tripwire is read-only");

    assert!(tree.num_leaves > 1, "the healthy corpus must grow a real multi-leaf tree, got {}", tree.num_leaves);
    for (leaf, &v) in tree.leaf_value.iter().enumerate() {
        assert!(v.is_finite(), "healthy leaf {leaf} value must be finite, got {v}");
        assert!(v.abs() < 1e3, "healthy leaf {leaf} value {v} is implausibly large (spike-072 blow-up signature)");
    }
}

// ===========================================================================
// Task 2 — the ZERO-BIN CANARY (both lanes)
//
// The mfb/default cell of a forced-empty bin must stay ~0 across every resident-histogram
// slot class. This exercises the SHARED f64 build + subtract kernels (generic over the
// runtime) on the selected client; the rocm lane additionally drives the resident u64
// `fix_compact_f64_on` FixHistogram (the exact site 29-03 fixed).
// ===========================================================================

/// The forced-empty most-frequent bin (spike-072 item 10): a dedicated bin index that NO row
/// ever maps to (`find_bin_with_zero_as_one_bin` + the `sparse_rate < 0.7` mfb override).
const CANARY_MFB: usize = 3;
const CANARY_NUM_BIN: u32 = 8;

/// Build a single feature column whose bins NEVER equal [`CANARY_MFB`] (the corpus has no
/// exact zeros ⇒ the mfb bin has 0 rows globally), plus non-uniform gradients + binary-
/// logloss-like non-uniform hessians. Returns `(bins, grad, hess)`.
fn canary_column(num_data: usize) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    // Bins drawn from {0,1,2,4,5,6,7} — the mfb bin (3) is never emitted.
    let live_bins: [u32; 7] = [0, 1, 2, 4, 5, 6, 7];
    let bins: Vec<u32> = (0..num_data).map(|r| live_bins[r % live_bins.len()]).collect();
    let half = num_data as f32 / 2.0;
    let grad: Vec<f32> = (0..num_data).map(|r| (r as f32 - half) * 0.1).collect();
    let hess: Vec<f32> = (0..num_data)
        .map(|r| {
            let p = 0.05 + 0.9 * (r as f32 / num_data as f32);
            p * (1.0 - p)
        })
        .collect();
    (bins, grad, hess)
}

/// Read the (g, h) of the mfb cell out of a stride-2 `[g0,h0,g1,h1,…]` histogram.
fn mfb_cell(hist: &[f64]) -> (f64, f64) {
    (hist[2 * CANARY_MFB], hist[2 * CANARY_MFB + 1])
}

/// The shared canary body — runs the forced-empty-mfb build+subtract chain on `$client` and
/// asserts the mfb cell stays within `$tol` of 0 across EVERY inspected slot class. `$tol` is
/// EXACTLY 0.0 on both lanes (the f64 fold never touches the empty bin; `x − x = +0.0`).
///
/// A MACRO (not a generic fn) so the runtime `R` is inferred from the concrete client type —
/// keeps oracle-harness cubecl-FREE (CMP-01: no direct `cubecl::Runtime` bound named here),
/// the same containment discipline `on_device_subtract_residue.rs` follows.
macro_rules! run_zero_bin_canary {
    ($client:expr, $tol:expr) => {{
    let client = $client;
    let tol: f64 = $tol;
    let num_data = 4096usize;
    let (bins, grad, hess) = canary_column(num_data);

    // ---- PRECONDITION (armed-trap self-validation): the mfb bin has 0 global rows. If a
    // future binning change routes ANY row here, the canary must fail LOUDLY on its
    // precondition rather than silently watch a live bin. ----
    let mfb_rows = bins.iter().filter(|&&b| b as usize == CANARY_MFB).count();
    assert_eq!(mfb_rows, 0, "canary precondition: the mfb bin must have 0 global rows (got {mfb_rows}) — trap not armed");

    // ---- ROOT build (slot class 1: root post-build). ----
    let root = construct_histograms_f64_on(client, &bins, &grad, &hess, CANARY_NUM_BIN)
        .expect("root histogram build");
    let (rg, rh) = mfb_cell(&root);
    assert!(rg.abs() <= tol && rh.abs() <= tol, "ROOT mfb cell must be ~0, got (g={rg}, h={rh})");

    // ---- SMALLER child build over the left half of the rows (a co-pack sibling slot). ----
    let left: Vec<usize> = (0..num_data).filter(|r| bins[*r] < 4).collect();
    let lb: Vec<u32> = left.iter().map(|&r| bins[r]).collect();
    let lg: Vec<f32> = left.iter().map(|&r| grad[r]).collect();
    let lh: Vec<f32> = left.iter().map(|&r| hess[r]).collect();
    let smaller = construct_histograms_f64_on(client, &lb, &lg, &lh, CANARY_NUM_BIN)
        .expect("smaller child histogram build");
    let (sg, sh) = mfb_cell(&smaller);
    assert!(sg.abs() <= tol && sh.abs() <= tol, "co-pack SMALLER mfb cell must be ~0, got (g={sg}, h={sh})");

    // ---- SUBTRACT-DERIVED larger child (slot class 2: larger = root − smaller). The other
    // co-pack sibling slot. This is the class the spike-072 ~0.36 phantom rode. ----
    let larger = subtract_histograms_f64_on(client, &root, &smaller).expect("subtract-derived larger child");
    let (dg, dh) = mfb_cell(&larger);
    assert!(dg.abs() <= tol && dh.abs() <= tol, "SUBTRACT-DERIVED larger-child mfb cell must be ~0, got (g={dg}, h={dh})");

    // ---- SUBTRACT SPINE (num_leaves ≥ 8 spirit): derive down a chain of larger children,
    // asserting the mfb cell never accumulates residue along the spine (spike-072 leaf-16 was
    // a subtract-derived larger child on the right-spine slot-0 chain). ----
    let mut parent = larger;
    for level in 0..6 {
        // Split the parent's live bins again; build the smaller side and subtract.
        let child = construct_histograms_f64_on(client, &lb, &lg, &lh, CANARY_NUM_BIN)
            .expect("spine child histogram build");
        let derived = subtract_histograms_f64_on(client, &parent, &child).expect("spine subtract");
        let (pg, ph) = mfb_cell(&derived);
        assert!(
            pg.abs() <= tol && ph.abs() <= tol,
            "SPINE level {level} subtract-derived mfb cell must be ~0, got (g={pg}, h={ph})"
        );
        parent = derived;
    }

    // ---- ARMED-TRAP self-validation: prove the canary CAN detect pollution. Inject a
    // δ = 0.37 phantom into the ROOT mfb cell (the spike-072-class residue) and subtract a
    // clean zero-mfb child — the derived cell must carry δ through, so the |·| ≤ tol
    // assertion above WOULD fire on a real regression. ----
    let mut polluted = root.clone();
    let delta = 0.37f64;
    polluted[2 * CANARY_MFB] += delta; // phantom g
    polluted[2 * CANARY_MFB + 1] += delta; // phantom h
    let poll_derived = subtract_histograms_f64_on(client, &polluted, &smaller).expect("polluted subtract");
    let (eg, eh) = mfb_cell(&poll_derived);
    assert!(
        eg.abs() > tol && eh.abs() > tol,
        "ARMED-TRAP: an injected δ={delta} phantom must survive to the mfb cell so the canary \
         detects pollution (got g={eg}, h={eh}); if this trips the canary is blind"
    );
    }};
}

/// Task 2 (cpu anchor lane) — the zero-bin canary at EXACTLY-0.0 tolerance on the cubecl-cpu
/// client (the ordered-f64 / integer-exact anchor chain). See the module header for the
/// seed-vs-build-disagreement semantics (spike-072 items 9–11; 29-SUBTRACT-RESIDUE.md).
#[test]
fn zero_bin_canary_cpu_anchor_exact_zero() {
    let client = cpu_client();
    run_zero_bin_canary!(&client, 0.0);
}

/// Task 2 (rocm resident lane) — the SAME canary on the hip client, PLUS the resident u64
/// `fix_compact_f64_on` FixHistogram (the exact site 29-03 fixed). Tolerance = the 29-03
/// disposition bound: EXACTLY 0 for a forced-empty cell at all scales (the built cell is
/// preserved, not reconstructed from the row-order seed).
#[cfg(feature = "rocm")]
#[test]
fn zero_bin_canary_rocm_resident() {
    use lgbm_compute::kernels::histogram::fix_compact_f64_on;
    use lgbm_compute::runtime::rocm_client;

    let client = rocm_client();
    // 29-03 disposition bound: exactly 0 for a forced-empty cell at all scales.
    run_zero_bin_canary!(&client, 0.0f64);

    // ---- The RESIDENT u64 FixHistogram slot (build+fix): a raw histogram with the mfb cell
    // built as exactly 0 and a leaf seed that DISAGREES with the built total by δ must still
    // leave the mfb cell EXACTLY 0 (29-03: preserve the built cell, do NOT reconstruct from
    // the seed). This is the resident-path guard the cpu lane cannot exercise. ----
    let num_bin = CANARY_NUM_BIN;
    let mfb = CANARY_MFB as u32;
    let raw: Vec<f32> = {
        let mut v = vec![0.0f32; 2 * num_bin as usize];
        let masses: [(usize, f32, f32); 7] = [
            (0, -5.0, 7.0), (1, 3.0, 4.0), (2, -2.0, 9.0),
            (4, 6.0, 5.0), (5, -1.0, 8.0), (6, 4.0, 3.0), (7, -7.0, 6.0),
        ];
        for (b, g, h) in masses {
            v[2 * b] = g;
            v[2 * b + 1] = h;
        }
        v // bin 3 (mfb) stays exactly (0, 0)
    };
    let built_g: f64 = raw.iter().step_by(2).map(|&x| f64::from(x)).sum();
    let built_h: f64 = raw.iter().skip(1).step_by(2).map(|&x| f64::from(x)).sum();
    let feats: [(usize, u32, i32, u32); 1] = [(0, num_bin, 0, mfb)];

    // Self-consistent seed → mfb exactly 0.
    let fixed = fix_compact_f64_on(&client, &raw, &feats, built_g, built_h).expect("fix (self-consistent seed)");
    let (fg, fh) = mfb_cell(&fixed);
    assert_eq!((fg, fh), (0.0, 0.0), "resident FixHistogram: self-consistent seed must leave mfb exactly 0");

    // Offset seed (δ=0.37) → STILL exactly 0 (the 29-03 fix: the built cell is preserved).
    let fixed_off = fix_compact_f64_on(&client, &raw, &feats, built_g + 0.37, built_h + 0.37)
        .expect("fix (offset seed)");
    let (og, oh) = mfb_cell(&fixed_off);
    assert_eq!(
        (og, oh),
        (0.0, 0.0),
        "resident FixHistogram: an offset external seed must NOT re-inject residue into the \
         forced-empty mfb cell (the 29-03 seed-vs-build-disagreement fix)"
    );
}
