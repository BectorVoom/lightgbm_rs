//! gfx1100 parity for the CubeCL mirror of LightGBM's CUDA histogram kernels,
//! PLUS the Phase-16 cpu f64-anchor scaffold (build-before-subtract ordering
//! invariant + the `[2b]/[2b+1]` interleave assert + §13-partition-aware anchor
//! case entry points).
//!
//! ## Two halves (D-05, def-f8u-01)
//! - The **cpu f64-anchor scaffold** (top of this file) runs EVERYWHERE — it is the
//!   hard merge gate. It pins every numeric expectation to the cubecl-cpu f64 fold
//!   (`construct_histograms_f64_on`, CubeDim::new_1d(1)); it NEVER compares two GPU
//!   outputs to each other.
//! - The **GPU mirror tests** (`mod hip`) are gated on `--features rocm` and compare
//!   the hip kernel to that same cpu f64 anchor within the ~1e-6 f32-atomic envelope
//!   (ABS 5e-6 / REL 1e-5), never GPU-vs-GPU (memory DEF-f8u-01).
//!
//! 16-03/16-04 extend the partition-aware case entry points below by adding the GPU
//! kernel call + an `assert_close(anchor, gpu)` — the cpu anchor is already in place.

use lgbm_compute::kernels::histogram::construct_histograms_f64_on;
use lgbm_compute::kernels::row_data::divide_cuda_feature_groups;
use lgbm_compute::kernels::subtract::subtract_histograms_cpu;
use lgbm_compute::runtime::{cpu_client, ActiveRuntime};
use lgbm_compute::BinColumn;

// ===========================================================================
// cpu f64-anchor envelope (the ~1e-6 ROCm gate, shared by `mod hip`).
// ===========================================================================

/// Non-panicking core of [`assert_close`]: returns `Err` describing the first cell
/// that diverges beyond the f32-atomic accumulation envelope (ABS 5e-6 / REL 1e-5),
/// else `Ok`. Separated so the envelope itself is unit-testable without `catch_unwind`.
fn close_within(anchor: &[f64], gpu: &[f64]) -> Result<(), String> {
    if anchor.len() != gpu.len() {
        return Err(format!("length mismatch: anchor {} vs gpu {}", anchor.len(), gpu.len()));
    }
    // f32-atomic accumulation envelope (04-ROCM-GAPS): ABS 5e-6 covers the cancellation
    // residual on grad cells; REL 1e-5 covers larger-magnitude cells.
    const ABS: f64 = 5e-6;
    const REL: f64 = 1e-5;
    for (i, (a, b)) in anchor.iter().zip(gpu).enumerate() {
        let diff = (a - b).abs();
        let tol = ABS + REL * a.abs();
        if diff > tol {
            return Err(format!(
                "cell {i} diverged beyond the f32-atomic envelope — anchor {a} vs gpu {b} \
                 (|diff| {diff} > tol {tol})"
            ));
        }
    }
    Ok(())
}

/// Assert two RAW histograms agree within the f32-atomic accumulation envelope —
/// the ~1e-6 ROCm gate the mirror was designed for (NOT bit-exact, by design).
///
/// 04-ROCM-GAPS NOTE (documented residual, NOT a kernel bug): the mirror accumulates
/// grad/hess in **f32 atomics in nondeterministic order**, exactly as the C++ CUDA
/// reference does (`score_t = float`, the SP_SHARED_HIST path —
/// cuda_histogram_constructor.cu:521). The GRAD histogram cells are sums of values
/// that partially cancel (positive and negative gradients), so a small true sum is
/// the difference of larger f32 partial sums — amplifying the f32 rounding residual.
/// The empty-leaf case (no accumulation) matches the anchor EXACTLY, confirming the
/// kernel is structurally faithful; the residual here is purely the f32-vs-f64
/// accumulation gap. Measured max |diff| on this corpus is ~2.4e-6 (full-corpus leaf,
/// ~2000 rows); the theoretical f32-atomic bound is ~1e-3. We gate at ABS 5e-6 — well
/// above the observed residual, far below the worst-case envelope, and the same
/// precision class as the C++ CUDA SP path. The CPU f64 anchor (the bit-exact merge
/// gate) is untouched.
fn assert_close(anchor: &[f64], gpu: &[f64], what: &str) {
    close_within(anchor, gpu).unwrap_or_else(|e| panic!("{what}: {e}"));
}

// ===========================================================================
// Phase 16 (Plan 16-01, Task 2): the `[2b]/[2b+1]` interleave assert helper.
// ===========================================================================

/// Non-panicking core of [`interleave_layout`]: assert grad lives at cell `bin*2`
/// and hess at cell `bin*2+1` for every bin of a per-feature histogram slice
/// (`hist.len() == 2 * num_bin`, the interleaved `[g0,h0,g1,h1,…]` layout, §7.4).
/// `expected_grad[b]` / `expected_hess[b]` are the per-bin truth (`bit`-exact).
fn check_interleave_layout(
    hist: &[f64],
    expected_grad: &[f64],
    expected_hess: &[f64],
) -> Result<(), String> {
    if expected_grad.len() != expected_hess.len() {
        return Err("expected_grad / expected_hess length mismatch".to_string());
    }
    if hist.len() != expected_grad.len() * 2 {
        return Err(format!(
            "hist len {} != 2 * num_bin {}",
            hist.len(),
            expected_grad.len() * 2
        ));
    }
    for b in 0..expected_grad.len() {
        let g = hist[b * 2];
        let h = hist[b * 2 + 1];
        if g.to_bits() != expected_grad[b].to_bits() {
            return Err(format!(
                "bin {b}: grad must live at cell {} — got {g}, expected {}",
                b * 2,
                expected_grad[b]
            ));
        }
        if h.to_bits() != expected_hess[b].to_bits() {
            return Err(format!(
                "bin {b}: hess must live at cell {} — got {h}, expected {}",
                b * 2 + 1,
                expected_hess[b]
            ));
        }
    }
    Ok(())
}

/// Reusable `[2b]/[2b+1]` layout checker the build/fix tests call (panics on
/// violation). grad at `bin*2`, hess at `bin*2+1` for every bin.
#[allow(dead_code)] // exercised by the build/fix tests landing in 16-03/16-04.
fn interleave_layout(hist: &[f64], expected_grad: &[f64], expected_hess: &[f64]) {
    check_interleave_layout(hist, expected_grad, expected_hess)
        .unwrap_or_else(|e| panic!("interleave layout violation: {e}"));
}

// ===========================================================================
// Phase 16 (Plan 16-01, Task 2): build-fully-synced-before-subtract ordering
// invariant (the 8aed100-class guard, D-05 / Pitfall 1).
// ===========================================================================

/// Models the build-fully-synced-before-subtract ordering invariant: a child's
/// `SubtractHistogram` may read the parent leaf histogram ONLY AFTER the parent build
/// has fully completed and published it. Reading the parent before the build is synced
/// is the 8aed100 bug (Phase-12 co-pack scan deferral ran subtract before the fused
/// smaller histogram was built). The "sync with no grid barrier" is modeled here as an
/// explicit host-level publish: `build_parent` populates the parent handle; only then
/// can `subtract_smaller` read it.
struct SubtractOrderingGuard {
    parent: Option<Vec<f64>>,
}

impl SubtractOrderingGuard {
    fn new() -> Self {
        Self { parent: None }
    }

    /// The build sync point — the parent leaf histogram completes and is published.
    fn build_parent(&mut self, parent: Vec<f64>) {
        self.parent = Some(parent);
    }

    /// Subtract the smaller child from the parent (the cpu f64 anchor subtract). Returns
    /// `Err` if invoked BEFORE the parent build is synced (the 8aed100 mis-order guard);
    /// otherwise the bit-exact `parent[i] - smaller[i]` fold.
    fn subtract_smaller(
        &self,
        client: &cubecl::prelude::ComputeClient<ActiveRuntime>,
        smaller: &[f64],
    ) -> Result<Vec<f64>, String> {
        match &self.parent {
            None => Err(
                "subtract invoked before parent build synced (8aed100 ordering violation)"
                    .to_string(),
            ),
            Some(parent) => {
                subtract_histograms_cpu(client, parent, smaller).map_err(|e| format!("{e:?}"))
            }
        }
    }
}

// ===========================================================================
// Phase 16 (Plan 16-01, Task 2): §13-partition-aware cpu f64 anchor case entry
// points (sparse, large-bin spill). 16-03/16-04 add the GPU kernel + assert_close.
// ===========================================================================

/// §13-partition-aware cpu f64 anchor: for each feature column, gather the leaf's rows'
/// (bin, grad, hess) and run the bit-exact cpu f64 fold (`construct_histograms_f64_on`,
/// CubeDim::new_1d(1)), returning the per-feature histograms concatenated
/// `[feat0 (2*num_bin) | feat1 | …]`. The SAME anchor `cpu_anchor` (mod hip) uses,
/// extended to a multi-column §13 layout so 16-03/16-04 only add the kernel call +
/// `assert_close`. Pinned to the cpu f64 fold — NEVER a GPU golden (def-f8u-01).
fn cpu_anchor_columns(
    columns: &[BinColumn],
    num_bins: &[u32],
    data_indices: &[u32],
    grad: &[f32],
    hess: &[f32],
) -> Vec<f64> {
    assert_eq!(columns.len(), num_bins.len(), "one num_bin per column");
    let client = cpu_client();
    let mut out = Vec::new();
    for (col, &nb) in columns.iter().zip(num_bins) {
        let binned: Vec<u32> = data_indices.iter().map(|&di| col.bin(di as usize)).collect();
        let g: Vec<f32> = data_indices.iter().map(|&di| grad[di as usize]).collect();
        let h: Vec<f32> = data_indices.iter().map(|&di| hess[di as usize]).collect();
        let hist = construct_histograms_f64_on::<ActiveRuntime>(&client, &binned, &g, &h, nb)
            .expect("partition-aware cpu f64 anchor fold");
        out.extend_from_slice(&hist);
    }
    out
}

// ===========================================================================
// cpu-anchor scaffold tests (run EVERYWHERE — the hard merge gate).
// ===========================================================================

/// The `[2b]/[2b+1]` interleave checker accepts the correct layout (grad at `2b`,
/// hess at `2b+1`) and REJECTS a swapped slice (hess at `2b`, grad at `2b+1`).
#[test]
fn interleave_layout_accepts_correct_and_rejects_swapped() {
    let grad = vec![1.0f64, 2.0, 3.0];
    let hess = vec![10.0f64, 20.0, 30.0];

    // Correct interleave: grad at 2b, hess at 2b+1.
    let correct = vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0];
    assert!(check_interleave_layout(&correct, &grad, &hess).is_ok(), "correct layout must pass");
    interleave_layout(&correct, &grad, &hess); // panicking wrapper must not panic.

    // Swapped interleave: hess at 2b, grad at 2b+1 -> must be rejected.
    let swapped = vec![10.0, 1.0, 20.0, 2.0, 30.0, 3.0];
    assert!(
        check_interleave_layout(&swapped, &grad, &hess).is_err(),
        "swapped [2b]/[2b+1] layout must fail"
    );
}

/// The build-fully-synced-before-subtract ordering invariant: the correctly-ordered
/// run (build parent, THEN subtract) reproduces `parent - smaller` bit-exactly on the
/// cpu f64 anchor; the mis-ordered control (subtract before the parent build) is
/// detected (the 8aed100 guard fires).
#[test]
fn subtract_ordering_parent_synced_before_subtract() {
    let client = cpu_client();
    // A parent leaf histogram and a smaller child (stride-2 f64 cells, [g0,h0,g1,h1,…]).
    let parent = vec![10.0f64, 5.0, 8.0, 4.0, 6.0, 3.0];
    let smaller = vec![3.0f64, 2.0, 1.0, 1.0, 2.0, 1.0];

    // INDEPENDENT reference: plain serial parent[i] - smaller[i].
    let reference: Vec<f64> = parent.iter().zip(&smaller).map(|(p, s)| p - s).collect();

    // (1) Correct ordering: build parent (sync), THEN subtract -> bit-exact to reference.
    let mut guard = SubtractOrderingGuard::new();
    guard.build_parent(parent.clone());
    let derived = guard
        .subtract_smaller(&client, &smaller)
        .expect("subtract after parent build must succeed");
    assert_eq!(derived.len(), reference.len(), "derived length");
    for i in 0..reference.len() {
        assert_eq!(
            derived[i].to_bits(),
            reference[i].to_bits(),
            "cell {i}: ordered subtract must reproduce parent - smaller bit-exactly"
        );
    }

    // (2) NEGATIVE control: subtract BEFORE the parent build is synced -> guard fires.
    let premature = SubtractOrderingGuard::new();
    assert!(
        premature.subtract_smaller(&client, &smaller).is_err(),
        "subtract before the parent build is synced must be rejected (8aed100 guard)"
    );
}

/// §13 SPARSE partition-aware case entry point: the cpu f64 anchor is built over a
/// multi-column sparse layout (the same fixture shape forcing row_ptr {16,32,64}),
/// and the per-feature `[2b]/[2b+1]` interleave holds. 16-03/16-04 add the GPU sparse
/// build-kernel call + `assert_close(anchor, gpu)` here. Pinned to the cpu f64 anchor.
#[test]
fn partition_aware_sparse_cpu_anchor_scaffold() {
    // A 2-column layout (num_bin per column) — the §13 partition input shape.
    let columns = vec![
        BinColumn::new(vec![0, 1, 2, 3, 0, 1, 2, 3], 200), // u8
        BinColumn::new(vec![0, 300, 301, 5, 6, 7, 8, 9], 4_000), // u16
    ];
    let num_bins = vec![200u32, 4_000];
    let layout = divide_cuda_feature_groups(&[200usize, 4_000], 6144);
    assert!(layout.num_feature_partitions >= 1, "sparse layout must yield >= 1 partition");

    // A non-trivial leaf subset (the indirect gather is genuinely exercised).
    let data_indices: Vec<u32> = vec![1, 2, 4, 6, 7];
    let grad = vec![0.5f32, -1.0, 2.0, -0.5, 1.5, -2.0, 0.25, -0.75];
    let hess = vec![1.0f32, 1.5, 2.0, 0.5, 1.25, 2.5, 0.75, 1.75];

    let anchor = cpu_anchor_columns(&columns, &num_bins, &data_indices, &grad, &hess);
    assert_eq!(
        anchor.len(),
        2 * (num_bins[0] as usize + num_bins[1] as usize),
        "concatenated per-feature 2*num_bin shape (no compaction)"
    );

    // Per-feature interleave layout holds: recompute each feature's per-bin grad/hess in
    // plain f64 and assert the anchor laid them at [2b]/[2b+1].
    let mut base = 0usize;
    for (col, &nb) in columns.iter().zip(&num_bins) {
        let mut eg = vec![0.0f64; nb as usize];
        let mut eh = vec![0.0f64; nb as usize];
        for &di in &data_indices {
            let b = col.bin(di as usize) as usize;
            eg[b] += f64::from(grad[di as usize]);
            eh[b] += f64::from(hess[di as usize]);
        }
        let feat = &anchor[base..base + 2 * nb as usize];
        interleave_layout(feat, &eg, &eh);
        base += 2 * nb as usize;
    }
    // TODO(16-03/16-04): build the GPU sparse histogram over `data_indices` and
    // `assert_close(&anchor, &gpu, "sparse_partition")` (never GPU-vs-GPU).
}

/// §13 LARGE-BIN / `_GlobalMemory` spill partition-aware case entry point: a column
/// whose bin count exceeds the SMEM budget forces `num_large_bin_partition > 0`; the
/// cpu f64 anchor is built over that spill layout and the interleave holds. 16-03 adds
/// the `_GlobalMemory` spill-variant kernel call + `assert_close`.
#[test]
fn partition_aware_largebin_spill_cpu_anchor_scaffold() {
    // num_bin 4096 > budget 3072 -> the spill column owns its own large-bin partition.
    let layout = divide_cuda_feature_groups(&[200usize, 4_096], 6144);
    assert!(
        layout.num_large_bin_partition > 0,
        "large-bin layout must yield the _GlobalMemory spill partition"
    );

    let columns = vec![
        BinColumn::new(vec![0, 1, 2, 3, 0, 1, 2, 3], 200), // small, partition 0
        BinColumn::new(vec![0, 1, 2, 3, 4, 5, 6, 4_000], 4_096), // large-bin spill
    ];
    let num_bins = vec![200u32, 4_096];
    let data_indices: Vec<u32> = vec![0, 2, 3, 5, 7];
    let grad = vec![0.5f32, -1.0, 2.0, -0.5, 1.5, -2.0, 0.25, -0.75];
    let hess = vec![1.0f32, 1.5, 2.0, 0.5, 1.25, 2.5, 0.75, 1.75];

    let anchor = cpu_anchor_columns(&columns, &num_bins, &data_indices, &grad, &hess);
    assert_eq!(
        anchor.len(),
        2 * (num_bins[0] as usize + num_bins[1] as usize),
        "concatenated per-feature 2*num_bin shape (no compaction)"
    );

    // Interleave holds on the spill feature too.
    let mut base = 0usize;
    for (col, &nb) in columns.iter().zip(&num_bins) {
        let mut eg = vec![0.0f64; nb as usize];
        let mut eh = vec![0.0f64; nb as usize];
        for &di in &data_indices {
            let b = col.bin(di as usize) as usize;
            eg[b] += f64::from(grad[di as usize]);
            eh[b] += f64::from(hess[di as usize]);
        }
        interleave_layout(&anchor[base..base + 2 * nb as usize], &eg, &eh);
        base += 2 * nb as usize;
    }
    // TODO(16-03): build the GPU `_GlobalMemory` spill histogram and
    // `assert_close(&anchor, &gpu, "largebin_spill")` (never GPU-vs-GPU).
}

/// Self-test of the `assert_close` envelope (the shared ~1e-6 ROCm gate): a within-tol
/// pair passes, an identical pair passes, and a beyond-tol pair is rejected.
#[test]
fn assert_close_envelope_self_test() {
    let anchor = vec![0.0f64, 1.0, -2.5, 1000.0];
    // identical -> Ok.
    assert!(close_within(&anchor, &anchor).is_ok(), "identical must pass");
    // within ABS 5e-6 on the small cells, within REL 1e-5 on the large cell.
    let near = vec![3e-6f64, 1.0 + 4e-6, -2.5 + 2e-6, 1000.0 + 5e-3];
    assert!(close_within(&anchor, &near).is_ok(), "within-envelope must pass");
    // beyond the envelope on cell 0 (1e-4 >> ABS 5e-6).
    let far = vec![1e-4f64, 1.0, -2.5, 1000.0];
    assert!(close_within(&anchor, &far).is_err(), "beyond-envelope must be rejected");
    // assert_close panics on the far case (the panicking wrapper).
    let panicked = std::panic::catch_unwind(|| assert_close(&anchor, &far, "self_test")).is_err();
    assert!(panicked, "assert_close must panic beyond the envelope");
}

// ===========================================================================
// GPU mirror tests (rocm only) — pinned GPU-vs-cpu-f64-anchor, NEVER GPU-vs-GPU.
//
// The CubeCL mirror of LightGBM's CUDA `CUDAConstructHistogramDenseKernel`
// (`lgbm_compute::kernels::histogram::construct_hist_cuda_mirror_kernel`): it
// gathers leaf rows INDIRECTLY in-kernel (via `data_indices`), reads grad/hess in
// FULL-CORPUS order (`grad[data_index]`), and indexes a RESIDENT feature-major bin
// buffer. Because it accumulates in f32 atomics in nondeterministic order, it is
// pinned to the CPU **f64 anchor** within ~1e-6 (ABS 5e-6 / REL 1e-5), NEVER
// GPU-vs-GPU (memory DEF-f8u-01).
// ===========================================================================
#[cfg(feature = "rocm")]
mod hip {
    use super::{assert_close, cpu_client};
    use cubecl::prelude::CubeElement;
    use lgbm_compute::kernels::histogram::{
        construct_histograms_cpu, construct_histograms_cuda_mirror_on,
        construct_histograms_cuda_mirror_resident_on,
    };
    use lgbm_compute::runtime::rocm_client;

    /// A small dense corpus: 3 features, ~2000 rows, 16 bins. Returns the resident
    /// feature-major bin buffer (`data[f * num_data + row]`), the full-corpus grad/hess.
    struct Corpus {
        num_data: usize,
        num_features: usize,
        num_bin: u32,
        /// feature-major: resident[f * num_data + row]
        resident: Vec<u32>,
        grad: Vec<f32>,
        hess: Vec<f32>,
    }

    fn make_corpus() -> Corpus {
        let num_data = 2000usize;
        let num_features = 3usize;
        let num_bin = 16u32;
        let mut resident = vec![0u32; num_features * num_data];
        let mut grad = vec![0.0f32; num_data];
        let mut hess = vec![0.0f32; num_data];
        for row in 0..num_data {
            // Deterministic, well-spread bins per feature (no RNG).
            for f in 0..num_features {
                let h = row
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(f.wrapping_mul(40_503).wrapping_add(0x9E37_79B9));
                resident[f * num_data + row] = (h % num_bin as usize) as u32;
            }
            // grad/hess vary per row so a wrong indirect gather would diverge.
            grad[row] = (row as f32) * 0.001 - 1.0;
            hess[row] = 1.0 + ((row % 7) as f32) * 0.1;
        }
        Corpus { num_data, num_features, num_bin, resident, grad, hess }
    }

    /// CPU f64 anchor: per-feature, gather the leaf's rows' (bin, grad, hess) and run
    /// the bit-exact `construct_histograms_cpu` fold, returning the concatenated RAW
    /// f64 histogram laid out `[feature0 (2*num_bin cells) | feature1 | feature2]`.
    fn cpu_anchor(corpus: &Corpus, data_indices: &[u32]) -> Vec<f64> {
        let cc = cpu_client();
        let mut out = Vec::new();
        for f in 0..corpus.num_features {
            let col = f * corpus.num_data;
            let binned: Vec<u32> = data_indices
                .iter()
                .map(|&di| corpus.resident[col + di as usize])
                .collect();
            let g: Vec<f32> = data_indices.iter().map(|&di| corpus.grad[di as usize]).collect();
            let h: Vec<f32> = data_indices.iter().map(|&di| corpus.hess[di as usize]).collect();
            let hist = construct_histograms_cpu(&cc, &binned, &g, &h, corpus.num_bin).unwrap();
            out.extend_from_slice(&hist);
        }
        out
    }

    #[test]
    fn cuda_mirror_dense_matches_cpu_anchor_within_tol() {
        let corpus = make_corpus();
        let gc = rocm_client();

        // A non-trivial leaf subset (an explicit `data_indices_in_leaf`): every 3rd row
        // from an offset, so the indirect gather is genuinely exercised (not identity).
        let data_indices: Vec<u32> = (7..corpus.num_data).step_by(3).map(|r| r as u32).collect();
        assert!(data_indices.len() > 100, "leaf must be non-trivial");

        let slot_len = corpus.num_features * 2 * corpus.num_bin as usize;
        let slot_off: Vec<usize> = (0..corpus.num_features)
            .map(|f| f * 2 * corpus.num_bin as usize)
            .collect();

        let gpu = construct_histograms_cuda_mirror_on(
            &gc,
            &corpus.resident,
            corpus.num_data,
            corpus.num_features,
            &data_indices,
            &corpus.grad,
            &corpus.hess,
            &slot_off,
            slot_len,
            corpus.num_bin,
        )
        .unwrap();

        let anchor = cpu_anchor(&corpus, &data_indices);
        assert_close(&anchor, &gpu, "cuda_mirror_dense");
    }

    #[test]
    fn cuda_mirror_empty_leaf_returns_zero_histogram() {
        let corpus = make_corpus();
        let gc = rocm_client();
        let data_indices: Vec<u32> = Vec::new();
        let slot_len = corpus.num_features * 2 * corpus.num_bin as usize;
        let slot_off: Vec<usize> = (0..corpus.num_features)
            .map(|f| f * 2 * corpus.num_bin as usize)
            .collect();

        let gpu = construct_histograms_cuda_mirror_on(
            &gc,
            &corpus.resident,
            corpus.num_data,
            corpus.num_features,
            &data_indices,
            &corpus.grad,
            &corpus.hess,
            &slot_off,
            slot_len,
            corpus.num_bin,
        )
        .unwrap();

        assert_eq!(gpu.len(), slot_len, "empty leaf: wrong length");
        assert!(gpu.iter().all(|&c| c == 0.0), "empty leaf must be all-zero");
    }

    #[test]
    fn cuda_mirror_full_corpus_leaf_matches_anchor() {
        // All rows in the leaf — exercises the multi-feature slot layout at scale and
        // confirms the column-partition / LDS-length arithmetic produces correct offsets.
        let corpus = make_corpus();
        let gc = rocm_client();
        let data_indices: Vec<u32> = (0..corpus.num_data as u32).collect();
        let slot_len = corpus.num_features * 2 * corpus.num_bin as usize;
        let slot_off: Vec<usize> = (0..corpus.num_features)
            .map(|f| f * 2 * corpus.num_bin as usize)
            .collect();

        let gpu = construct_histograms_cuda_mirror_on(
            &gc,
            &corpus.resident,
            corpus.num_data,
            corpus.num_features,
            &data_indices,
            &corpus.grad,
            &corpus.hess,
            &slot_off,
            slot_len,
            corpus.num_bin,
        )
        .unwrap();

        let anchor = cpu_anchor(&corpus, &data_indices);
        assert_close(&anchor, &gpu, "cuda_mirror_full");
    }

    #[test]
    fn cuda_mirror_resident_matches_cpu_anchor_within_tol() {
        // The upload-once / CUDA-faithful path (MWR-02): the feature-major bin buffer is
        // uploaded to the device ONCE, then the resident launcher is called with that
        // Handle (re-uploading only the per-leaf data_indices + grad/hess). It must produce
        // the SAME histogram as the per-call path and the CPU f64 anchor — i.e. switching
        // to launch_unchecked AND to a resident Handle did NOT change numerics. Pinned
        // GPU-vs-CPU-f64-anchor, NEVER GPU-vs-GPU (memory DEF-f8u-01).
        let corpus = make_corpus();
        let gc = rocm_client();

        // Same non-trivial leaf subset as the per-call dense test.
        let data_indices: Vec<u32> = (7..corpus.num_data).step_by(3).map(|r| r as u32).collect();
        assert!(data_indices.len() > 100, "leaf must be non-trivial");

        let slot_len = corpus.num_features * 2 * corpus.num_bin as usize;
        let slot_off: Vec<usize> = (0..corpus.num_features)
            .map(|f| f * 2 * corpus.num_bin as usize)
            .collect();

        // Upload the feature-major resident bin buffer ONCE (the caller's contract), then
        // pass its Handle into the resident launcher.
        let resident = gc.create_from_slice(u32::as_bytes(&corpus.resident));

        let gpu = construct_histograms_cuda_mirror_resident_on(
            &gc,
            resident,
            corpus.num_data,
            corpus.num_features,
            &data_indices,
            &corpus.grad,
            &corpus.hess,
            &slot_off,
            slot_len,
            corpus.num_bin,
        )
        .unwrap();

        let anchor = cpu_anchor(&corpus, &data_indices);
        assert_close(&anchor, &gpu, "cuda_mirror_resident");
    }
}
