//! `subtract_histograms` cube kernel — the histogram-subtraction MATH (A3).
//!
//! VERBATIM transcription of `FeatureHistogram::Subtract`
//! (`LightGBM/src/treelearner/feature_histogram.hpp:99-145`, commit 195c26fc,
//! VERSION 4.6.0.99), the default (`USE_DIST_GRAD = false`) f64 path:
//!
//! ```cpp
//! for (int i = 0; i < (meta_->num_bin - meta_->offset) * 2; ++i) {
//!   data_[i] -= other.data_[i];
//! }
//! ```
//!
//! i.e. element-wise `derived[i] = parent[i] - child[i]` over the stride-2
//! `[g0,h0,g1,h1,…]` f64 cells (both the gradient AND hessian cells). This is the
//! histogram-subtraction trick's MATH; WHICH child is subtracted from the parent
//! (the smaller sibling) is Phase-5 learner orchestration — RESEARCH open
//! question A3 resolves the subtract OP itself as in-scope at the kernel layer
//! (it is histogram-layer math).
//!
//! ## Determinism / launch shape
//! The op is element-wise (each output cell is independent), so it is launched
//! across a real workgroup as a 1D grid-stride loop (`CubeDim::new_1d(256)` ×
//! `CubeCount::Static(64,1,1)`): each thread owns disjoint indices
//! `{ABSOLUTE_POS, ABSOLUTE_POS+stride, …}`, `stride = CUBE_COUNT_X * CUBE_DIM_X`.
//! This is BIT-EXACT to the prior single-thread serial loop — no atomics, no
//! reduction, no ordering, no contention — so the byte-identical result holds on
//! cubecl-cpu AND cubecl-hip (CMP-04). There is no reduction non-determinism here.

use cubecl::prelude::*;

use crate::error::ComputeError;
use crate::runtime::ActiveRuntime;

/// The element-wise `parent - child` fold (FeatureHistogram::Subtract, the
/// default f64 path — the cpu anchor). 1D grid-stride loop: each thread owns the
/// disjoint indices `{ABSOLUTE_POS, ABSOLUTE_POS+stride, …}` with
/// `stride = CUBE_COUNT_X * CUBE_DIM_X`. BIT-EXACT-by-construction vs the serial
/// loop — every `out[i]` is an independent f64 subtract with no atomics, no
/// reduction, and no cross-thread/cross-element ordering, so disjoint-thread
/// execution yields the byte-identical result on any backend (CMP-04 cpu/hip
/// parity holds). The `while i < parent.len()` bound guards every write, so the
/// grid may over-cover any `len` (the launch over-provisions lanes).
#[cube(launch)]
pub fn subtract_hist_kernel(parent: &Array<f64>, child: &Array<f64>, out: &mut Array<f64>) {
    let stride = CUBE_COUNT_X as usize * CUBE_DIM_X as usize;
    let mut i = ABSOLUTE_POS as usize;
    let n = parent.len() as usize;
    while i < n {
        out[i] = parent[i] - child[i];
        i += stride;
    }
}


/// SIMD-vectorized twin of [`subtract_hist_kernel`] / `subtract_hist_kernel_f32`
/// over `Array<Vector<F, N>>` (spike-041, VALIDATED on cubecl-hip + cubecl-cpu).
///
/// IDENTICAL 1D grid-stride structure to the scalar kernels — the only difference is
/// that each lane subtracts a whole `Vector<F, N>` (one SIMD load/sub/store per `N`
/// cells) instead of a single cell, so the loop runs over `n_vec = n / N` vector
/// units laid over the SAME byte buffer. `Vector<F, N>` implements element-wise `Sub`
/// (`vector/ops.rs`), so `out[i] = parent[i] - child[i]` is **BIT-EXACT-by-construction**
/// to the scalar kernel: every component is an independent `F` subtract, no float op
/// is reordered, and there are no atomics, no reduction, and no cross-lane ordering
/// (spike-041 confirmed `bit_exact=true` on every width × size × backend cell; see also
/// CONVENTIONS "SIMD vectorization with `Vector<P,N>`" 313–351). The `N: Size` width is
/// supplied as a RUNTIME `usize` positional launch arg right after `CubeDim`, and array
/// lengths are passed in vector units (`n / N`). The `while i < n_vec` bound guards every
/// write, so the grid may over-cover.
///
/// Callers gate this kernel behind an exact-divisibility check (`n % N == 0 && N > 1`)
/// and fall back to the scalar kernels otherwise — there is NO tail handling here.
/// Mixed-cardinality tail vectorization (a masked/scalar remainder for non-divisible
/// lengths) is a possible follow-on, intentionally omitted to keep the wire minimal and
/// the bit-exact merge gate trivially safe.
#[cube(launch)]
pub fn subtract_hist_kernel_vec<F: Float, N: Size>(
    parent: &Array<Vector<F, N>>,
    child: &Array<Vector<F, N>>,
    out: &mut Array<Vector<F, N>>,
    n_vec: usize,
) {
    let stride = CUBE_COUNT_X as usize * CUBE_DIM_X as usize;
    let mut i = ABSOLUTE_POS as usize;
    while i < n_vec {
        out[i] = parent[i] - child[i];
        i += stride;
    }
}

/// Pick the SIMD vectorization width for a flat length `n` of `elem_size`-byte cells.
///
/// Returns the backend's widest `io_optimized_vector_sizes(elem_size)` width that is
/// `> 1` AND exactly divides `n` (so the vectorized kernel needs no tail logic), else
/// `1` — the sentinel meaning "use the scalar kernel". The iterator yields widths
/// widest-first (hip f32 → `[4,2,1]`; cpu f64 → `[8,4,2,1]`); the production all-256-bin
/// histogram shape (`2 * num_bin` divisible by the max width) takes the wide path, while
/// mixed-cardinality / odd lengths fall back to scalar (spike-041 launch recipe).
fn pick_vec_width<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    elem_size: usize,
    n: usize,
) -> usize {
    match client.io_optimized_vector_sizes(elem_size).next() {
        Some(w) if w > 1 && n % w == 0 => w,
        _ => 1,
    }
}

/// Host-side `subtract_histograms` on the cpu reference runtime.
///
/// Computes `derived[i] = parent[i] - child[i]` over the `2 * num_bin` f64 cells
/// (FeatureHistogram::Subtract). Validates equal lengths (V5, threat T-04-01)
/// before the unsafe launch.
///
/// # Errors
/// [`ComputeError::LengthMismatch`] if `parent.len() != child.len()`.
pub fn subtract_histograms_cpu(
    client: &cubecl::prelude::ComputeClient<ActiveRuntime>,
    parent: &[f64],
    child: &[f64],
) -> Result<Vec<f64>, ComputeError> {
    subtract_histograms_f64_on(client, parent, child)
}

/// The f64 `subtract_histograms` cube path, **generic over the runtime** `R` so it
/// runs on the cubecl-cpu anchor (via [`subtract_histograms_cpu`]) AND on
/// cubecl-hip (the GPU `RocmBackend`) — the same f64 element-wise kernel, bit-exact
/// across both.
///
/// # Errors
/// [`ComputeError::LengthMismatch`] if `parent.len() != child.len()`.
pub fn subtract_histograms_f64_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    parent: &[f64],
    child: &[f64],
) -> Result<Vec<f64>, ComputeError> {
    // --- V5 boundary validation (T-04-01): equal `2*num_bin` lengths ---
    if parent.len() != child.len() {
        return Err(ComputeError::LengthMismatch {
            expected: parent.len(),
            actual: child.len(),
        });
    }

    let n = parent.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let h_parent = client.create_from_slice(f64::as_bytes(parent));
    let h_child = client.create_from_slice(f64::as_bytes(child));
    let zeros = vec![0.0f64; n];
    let h_out = client.create_from_slice(f64::as_bytes(&zeros));

    // Width-gated SIMD dispatch (spike-041): vectorize when, and only when, the chosen
    // `io_optimized` width divides `n` exactly (`vs > 1`); otherwise the proven scalar
    // kernel. Both read/write the SAME byte buffers — `n` f64 cells whether viewed as
    // scalar (`n` units) or as `n / vs` `Vector<f64, vs>` units — and `Vector::sub` is
    // element-wise, so the result is bit-exact either way.
    let vs = pick_vec_width(client, std::mem::size_of::<f64>(), n);

    // SAFETY: `h_parent`/`h_child`/`h_out` were each allocated for exactly `n`
    // f64 elements (validated equal above) and outlive the launch; the kernel
    // reads/writes only indices `0..n` (the grid over-covers but the bound check
    // guards every write). When vectorized, the same `n` f64 cells are addressed as
    // `n / vs` `Vector<f64, vs>` units over the identical byte buffer (`n % vs == 0`).
    // All cubecl unsafe is confined here (CMP-01).
    unsafe {
        if vs > 1 {
            subtract_hist_kernel_vec::launch::<f64, R>(
                client,
                CubeCount::Static(64, 1, 1),
                CubeDim::new_1d(256),
                vs,
                ArrayArg::from_raw_parts(h_parent, n / vs),
                ArrayArg::from_raw_parts(h_child, n / vs),
                ArrayArg::from_raw_parts(h_out.clone(), n / vs),
                n / vs,
            );
        } else {
            subtract_hist_kernel::launch(
                client,
                CubeCount::Static(64, 1, 1),
                CubeDim::new_1d(256),
                ArrayArg::from_raw_parts(h_parent, n),
                ArrayArg::from_raw_parts(h_child, n),
                ArrayArg::from_raw_parts(h_out.clone(), n),
            );
        }
    }

    let bytes = client.read_one_unchecked(h_out);
    Ok(f64::from_bytes(&bytes).to_vec())
}

/// **Native** host f64 element-wise `parent - child` — the production cpu-anchor
/// path (R2). Bit-IDENTICAL to [`subtract_histograms_cpu`] (the single-unit
/// `subtract_hist_kernel`): the same `derived[i] = parent[i] - child[i]` over the
/// `2*num_bin` cells and the same V5 length check, without the cubecl launch.
/// The cubecl path is retained for the kernel-parity / ROCm-mirror tests.
///
/// # Errors
/// [`ComputeError::LengthMismatch`] if `parent.len() != child.len()`.
pub fn subtract_histograms_cpu_native(
    parent: &[f64],
    child: &[f64],
) -> Result<Vec<f64>, ComputeError> {
    if parent.len() != child.len() {
        return Err(ComputeError::LengthMismatch {
            expected: parent.len(),
            actual: child.len(),
        });
    }
    Ok(parent.iter().zip(child).map(|(p, c)| p - c).collect())
}

/// Handle-in/Handle-out resident subtract (260608-p90 Task 2A) — the device-resident
/// sibling of [`subtract_histograms_f64_on`]. Derives `out[i] = parent[i] - child[i]`
/// over `len` stride-2 f64 cells entirely on device: it CONSUMES the `parent` and
/// `child` device Handles, allocates a fresh `out` Handle of `len` f64 cells, launches
/// the VERBATIM element-wise [`subtract_hist_kernel`] (the same EXACT-math kernel the
/// host path uses — no new math), and RETURNS the `out` Handle. NO read-back — the
/// derived larger child's histogram never leaves the device. The caller guarantees
/// both input Handles describe `len` f64 cells (the pool's `slot_len`).
///
/// # Errors
/// [`ComputeError::Runtime`] if `len == 0` (degenerate — no cells to subtract).
#[cfg(feature = "gpu")]
pub fn subtract_histograms_f64_from_handles_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    parent: cubecl::server::Handle,
    child: cubecl::server::Handle,
    len: usize,
) -> Result<cubecl::server::Handle, ComputeError> {
    if len == 0 {
        return Err(ComputeError::Runtime {
            detail: "subtract_histograms_from_handles: len must be > 0".to_string(),
        });
    }
    let zeros = vec![0.0f64; len];
    let h_out = client.create_from_slice(f64::as_bytes(&zeros));

    // Width-gated SIMD dispatch (spike-041) on the resident hot path: vectorize when the
    // chosen `io_optimized` width divides `len` exactly, else the proven scalar kernel.
    // No read-back — the derived child's histogram stays on device either way.
    let vs = pick_vec_width(client, std::mem::size_of::<f64>(), len);

    // SAFETY: `parent`/`child`/`h_out` each describe exactly `len` f64 cells (the
    // caller guarantees the inputs; `h_out` is allocated for `len` here) and outlive
    // the launch; the kernel reads/writes only indices `0..len` (the grid over-covers
    // but the bound check guards every write). When vectorized, the same `len` f64 cells
    // are addressed as `len / vs` `Vector<f64, vs>` units over the identical byte buffer
    // (`len % vs == 0`). All cubecl unsafe is confined here (CMP-01).
    unsafe {
        if vs > 1 {
            subtract_hist_kernel_vec::launch::<f64, R>(
                client,
                CubeCount::Static(64, 1, 1),
                CubeDim::new_1d(256),
                vs,
                ArrayArg::from_raw_parts(parent, len / vs),
                ArrayArg::from_raw_parts(child, len / vs),
                ArrayArg::from_raw_parts(h_out.clone(), len / vs),
                len / vs,
            );
        } else {
            subtract_hist_kernel::launch(
                client,
                CubeCount::Static(64, 1, 1),
                CubeDim::new_1d(256),
                ArrayArg::from_raw_parts(parent, len),
                ArrayArg::from_raw_parts(child, len),
                ArrayArg::from_raw_parts(h_out.clone(), len),
            );
        }
    }

    Ok(h_out)
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu_client;

    #[test]
    fn subtract_elementwise() {
        let client = cpu_client();
        let parent = vec![10.0f64, 5.0, 8.0, 4.0];
        let child = vec![3.0f64, 2.0, 1.0, 1.0];
        let got = subtract_histograms_cpu(&client, &parent, &child).unwrap();
        assert_eq!(got, vec![7.0, 3.0, 7.0, 3.0]);
    }

    #[test]
    fn subtract_length_mismatch() {
        let client = cpu_client();
        let err = subtract_histograms_cpu(&client, &[1.0, 2.0], &[1.0]).unwrap_err();
        assert!(matches!(err, ComputeError::LengthMismatch { .. }));
    }

    #[test]
    fn subtract_empty_ok() {
        let client = cpu_client();
        let got = subtract_histograms_cpu(&client, &[], &[]).unwrap();
        assert!(got.is_empty());
    }

    /// The grid-stride parallel f64 kernel must be `to_bits()`-identical, cell by
    /// cell, to a plain serial Rust `parent[i] - child[i]` reference — on a
    /// representative length (25600 = 50 feat × 256 bins × 2) AND an odd length
    /// (12345) that exercises the stride remainder (not a multiple of the grid).
    #[test]
    fn subtract_parallel_equals_serial_f64() {
        let client = cpu_client();
        for &len in &[25600usize, 12345usize] {
            // Deterministic, varied (incl. negatives + fractions) inputs.
            let parent: Vec<f64> = (0..len)
                .map(|i| (i as f64) * 0.5 - 1234.5 + (i as f64).sin())
                .collect();
            let child: Vec<f64> = (0..len)
                .map(|i| (i as f64) * 0.25 + 7.0 - (i as f64).cos())
                .collect();

            let serial: Vec<f64> = parent.iter().zip(&child).map(|(p, c)| p - c).collect();
            let parallel = subtract_histograms_f64_on(&client, &parent, &child).unwrap();

            assert_eq!(parallel.len(), len, "len {len}: output length");
            for i in 0..len {
                assert_eq!(
                    parallel[i].to_bits(),
                    serial[i].to_bits(),
                    "len {len}, cell {i}: parallel {} vs serial {} not byte-identical",
                    parallel[i],
                    serial[i]
                );
            }
        }
    }

    /// OCX-03 (spike-072 item 15b), disposition (a): a subtract-derived cell whose
    /// parent and child carry IDENTICAL mass must be EXACTLY `+0.0` — the property that
    /// makes a true-zero-mass subtract-derived histogram cell exactly 0. This is the
    /// leak-free contract of the subtract op itself: IEEE f64 `x − x = +0.0` for every
    /// finite `x` (incl. the forced-empty mfb cell, `0.0 − 0.0`), so if the upstream
    /// FixHistogram leaves parent and child true-zero cells identical, the subtract never
    /// manufactures phantom mass. Asserted `to_bits()`-exact (`+0.0`, NOT `-0.0`) across
    /// EVERY shipped entry point: the scalar cube kernel, the SIMD-vectorized cube kernel
    /// (256000 divisible by the cpu f64 widths), and the native host fold.
    #[test]
    fn subtract_identical_cells_are_exact_positive_zero() {
        let client = cpu_client();
        let pos_zero_bits = 0.0f64.to_bits();
        // Representative widths: 12345 → scalar path; 256_000 → vectorized path.
        for &len in &[12345usize, 256_000usize] {
            // Craft parent == child with varied mass (negatives, fractions) AND explicit
            // forced-empty cells (exact 0.0) at a stride — the mfb/default cell analog.
            let hist: Vec<f64> = (0..len)
                .map(|i| {
                    if i % 37 == 0 {
                        0.0 // forced-empty (globally-zero) cell — must subtract to +0.0
                    } else {
                        (i as f64) * 0.5 - 987.25 + (i as f64).sin()
                    }
                })
                .collect();
            let parent = hist.clone();
            let child = hist; // identical mass in every cell (true-zero derived everywhere)

            let cube = subtract_histograms_f64_on(&client, &parent, &child).unwrap();
            let native = subtract_histograms_cpu_native(&parent, &child).unwrap();
            assert_eq!(cube.len(), len);
            assert_eq!(native.len(), len);
            for i in 0..len {
                assert_eq!(
                    cube[i].to_bits(),
                    pos_zero_bits,
                    "len {len} cell {i}: cube subtract of identical cells must be +0.0, got {}",
                    cube[i]
                );
                assert_eq!(
                    native[i].to_bits(),
                    pos_zero_bits,
                    "len {len} cell {i}: native subtract of identical cells must be +0.0, got {}",
                    native[i]
                );
            }
        }
    }

    /// The VECTORIZED branch itself must be bit-exact. 256000 = 500 feat × 256 bin × 2
    /// is divisible by every cpu f64 `io_optimized` width (8/4/2), so `pick_vec_width`
    /// selects the SIMD `subtract_hist_kernel_vec` path here — and its `to_bits()` output
    /// must match a plain serial `parent[i] - child[i]` on every cell (spike-041's
    /// bit-exact-by-construction claim, asserted on the cpu client). The existing
    /// 12345-length cases above already cover the non-divisible scalar fallback.
    #[test]
    fn subtract_vec_equals_serial_f64() {
        let client = cpu_client();
        let len = 256_000usize;
        let parent: Vec<f64> = (0..len)
            .map(|i| (i as f64) * 0.5 - 1234.5 + (i as f64).sin())
            .collect();
        let child: Vec<f64> = (0..len)
            .map(|i| (i as f64) * 0.25 + 7.0 - (i as f64).cos())
            .collect();

        let serial: Vec<f64> = parent.iter().zip(&child).map(|(p, c)| p - c).collect();
        let vectorized = subtract_histograms_f64_on(&client, &parent, &child).unwrap();

        assert_eq!(vectorized.len(), len, "len {len}: output length");
        for i in 0..len {
            assert_eq!(
                vectorized[i].to_bits(),
                serial[i].to_bits(),
                "len {len}, cell {i}: vectorized {} vs serial {} not byte-identical",
                vectorized[i],
                serial[i]
            );
        }
    }

}
