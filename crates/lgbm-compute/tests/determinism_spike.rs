//! D-04a bit-determinism spike (RUN-FIRST gate).
//!
//! This is the load-bearing empirical test for the entire Phase-4 architecture.
//! D-04 bets that the cubecl-cpu single-owner ordered f64 fold is *bit-exact*
//! and can serve as the deterministic anchor. RESEARCH Pitfall 1 shows cubecl-cpu
//! spawns one OS worker thread per cube unit — it is NOT a single-threaded
//! sequential executor — so this bet only holds if exactly one unit owns the
//! fold in a fixed order (`CubeDim::new_1d(1)`).
//!
//! Three behaviors are asserted (all on the cubecl-cpu reference runtime):
//!  1. Repeat-launch invariance: N>=20 launches on the SAME inputs yield
//!     byte-identical f64 `out` (compare `to_bits()` of every cell).
//!  2. Sequential-fold anchor: the kernel `out` is bit-identical to a
//!     hand-computed sequential f64 fold in C++ `dense_bin.hpp` order.
//!  3. Decision: if either invariant fails, the test FAILS LOUDLY with the
//!     D-04a fallback instruction — never silently passes.

use lgbm_compute::kernels::histogram::construct_histograms_cpu;
use lgbm_compute::runtime::cpu_client;
use lgbm_compute::ComputeError;

const N_LAUNCHES: usize = 25; // >= 20 per the D-04a mandate

/// D-04a fallback banner emitted on any spike failure (no silent pass).
const D04A_FALLBACK: &str = "\
D-04a FALLBACK REQUIRED: cubecl-cpu is NOT bit-stable as the deterministic anchor. \
Before 04-02/04-03 build the kernel suite, relax the cubecl-cpu anchor to ~1e-6 against \
the C++ golden and re-evaluate whether a separate scalar reference is needed (CONTEXT.md D-04a).";

/// Synthetic input set: a spread of grad/hess signs and magnitudes across a
/// handful of bins (D-02a — stress the f32→f64 reduction).
fn synthetic_inputs() -> (Vec<u32>, Vec<f32>, Vec<f32>, u32) {
    // 6 bins, 12 rows; bins repeat so multiple rows fold into the same cell
    // (exercising accumulation order). Magnitudes span ~1e-3 .. ~1e3 and mix
    // signs so non-associative f64 summation order is observable.
    let binned: Vec<u32> = vec![0, 3, 1, 5, 2, 0, 4, 1, 3, 5, 2, 0];
    let grad: Vec<f32> = vec![
        1.5, -2.25, 1e3, -1e-3, 0.125, 7.5, -0.0625, 3.333_333, -1e2, 2.5e-2, -4.75, 100.0,
    ];
    let hess: Vec<f32> = vec![
        0.5, 1.25, 2.0e2, 1e-3, 0.0625, 0.75, 1.5, 0.333_333, 1e1, 5.0e-2, 0.25, 9.0,
    ];
    let num_bin = 6u32;
    (binned, grad, hess, num_bin)
}

/// Hand-computed sequential f64 fold in the exact C++ `dense_bin.hpp:120-135`
/// order: iterate i in 0..n, `ti = bin<<1`, `out[ti] += grad[i] as f64`,
/// `out[ti+1] += hess[i] as f64`. f32 read, f64 accumulate (`hist_t = double`).
fn sequential_fold(binned: &[u32], grad: &[f32], hess: &[f32], num_bin: u32) -> Vec<f64> {
    let mut out = vec![0.0f64; 2 * num_bin as usize];
    for i in 0..binned.len() {
        let ti = binned[i] as usize * 2;
        out[ti] += grad[i] as f64;
        out[ti + 1] += hess[i] as f64;
    }
    out
}

/// Bit-level comparison of two f64 buffers (raw `to_bits()`), returning the
/// first divergent index for a loud, diagnostic failure.
fn first_bit_divergence(a: &[f64], b: &[f64]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(usize::MAX);
    }
    a.iter()
        .zip(b.iter())
        .position(|(x, y)| x.to_bits() != y.to_bits())
}

#[test]
fn determinism_spike_cpu_bit_exact() {
    let client = cpu_client();
    let (binned, grad, hess, num_bin) = synthetic_inputs();

    // --- Behavior 1: repeat-launch invariance (N>=20 byte-identical) ---
    let first = construct_histograms_cpu(&client, &binned, &grad, &hess, num_bin)
        .expect("first launch must succeed");
    for launch in 1..N_LAUNCHES {
        let again = construct_histograms_cpu(&client, &binned, &grad, &hess, num_bin)
            .expect("repeat launch must succeed");
        if let Some(idx) = first_bit_divergence(&first, &again) {
            panic!(
                "REPEAT-LAUNCH NONDETERMINISM at launch {launch}, cell {idx}: \
                 {:#018x} (launch 0) vs {:#018x} (launch {launch}).\n{D04A_FALLBACK}",
                first.get(idx).map_or(0, |v| v.to_bits()),
                again.get(idx).map_or(0, |v| v.to_bits()),
            );
        }
    }

    // --- Behavior 2: bit-exact vs the hand-computed sequential f64 fold ---
    let expected = sequential_fold(&binned, &grad, &hess, num_bin);
    if let Some(idx) = first_bit_divergence(&first, &expected) {
        panic!(
            "SEQUENTIAL-FOLD MISMATCH at cell {idx}: kernel {:#018x} vs C++-order fold {:#018x}.\n{D04A_FALLBACK}",
            first.get(idx).map_or(0, |v| v.to_bits()),
            expected.get(idx).map_or(0, |v| v.to_bits()),
        );
    }

    // Sanity: the histogram actually accumulated (not all zero) so the assertions
    // above were meaningful.
    assert!(
        first.iter().any(|&c| c != 0.0),
        "histogram must be non-trivially populated"
    );
}

#[test]
fn boundary_validation_returns_typed_errors() {
    let client = cpu_client();

    // Out-of-range bin index -> typed error, never a panic (V5, T-04-01).
    let err = construct_histograms_cpu(&client, &[0, 9], &[1.0, 1.0], &[1.0, 1.0], 6)
        .expect_err("bin 9 >= num_bin 6 must be rejected");
    assert!(
        matches!(err, ComputeError::BinIndexOutOfRange { bin: 9, num_bin: 6, .. }),
        "expected BinIndexOutOfRange, got {err:?}"
    );

    // grad/hess/binned length mismatch -> typed error.
    let err = construct_histograms_cpu(&client, &[0, 1], &[1.0], &[1.0, 1.0], 6)
        .expect_err("length mismatch must be rejected");
    assert!(
        matches!(err, ComputeError::LengthMismatch { .. }),
        "expected LengthMismatch, got {err:?}"
    );
}
