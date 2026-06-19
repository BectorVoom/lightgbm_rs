//! gfx1100 parity for the CubeCL mirror of LightGBM's CUDA
//! `CUDAConstructHistogramDenseKernel` (the signature inner kernel of
//! `cuda_single_gpu_tree_learner`), ported in
//! `lgbm_compute::kernels::histogram::construct_hist_cuda_mirror_kernel`.
//!
//! Feature-gated on `rocm` — runs only under
//! `cargo test -p lgbm-compute --features rocm`, never in a CPU-only build.
//!
//! The mirror gathers leaf rows INDIRECTLY in-kernel (via `data_indices`), reads
//! grad/hess in FULL-CORPUS order (`grad[data_index]`), and indexes a RESIDENT
//! feature-major bin buffer (`data[column_start * num_data + data_index]`) — the
//! signature CUDA indirection. Because it accumulates in f32 atomics in
//! nondeterministic order, it is pinned to the CPU **f64 anchor** within ~1e-6
//! (ABS 1e-6 / REL 1e-5), NEVER GPU-vs-GPU (memory DEF-f8u-01).
#![cfg(feature = "rocm")]

use lgbm_compute::kernels::histogram::{construct_histograms_cpu, construct_histograms_cuda_mirror_on};
use lgbm_compute::runtime::{cpu_client, rocm_client};

/// Assert two RAW histograms agree within ABS 1e-6 / REL 1e-5 — the ~1e-6 ROCm
/// gate the f32-atomic mirror was designed for (NOT bit-exact, by design).
fn assert_close(anchor: &[f64], gpu: &[f64], what: &str) {
    assert_eq!(anchor.len(), gpu.len(), "{what}: length mismatch");
    const ABS: f64 = 1e-6;
    const REL: f64 = 1e-5;
    for (i, (a, b)) in anchor.iter().zip(gpu).enumerate() {
        let diff = (a - b).abs();
        let tol = ABS + REL * a.abs();
        assert!(
            diff <= tol,
            "{what}: cell {i} diverged beyond ~1e-6 — anchor {a} vs gpu {b} (|diff| {diff} > tol {tol})"
        );
    }
}

/// A small dense corpus: 3 features, ~2000 rows, 16 bins. Returns the resident
/// feature-major bin buffer (`data[f * num_data + row]`), the full-corpus grad/hess,
/// and per-feature bin columns (row-major-per-feature, for the CPU anchor gather).
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

/// CPU f64 anchor: per-feature, gather the leaf's rows' (bin, grad, hess) and run the
/// bit-exact `construct_histograms_cpu` fold, returning the concatenated RAW f64
/// histogram laid out `[feature0 (2*num_bin cells) | feature1 | feature2]`.
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
