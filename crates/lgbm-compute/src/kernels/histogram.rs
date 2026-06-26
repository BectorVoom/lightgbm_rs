//! Minimal `construct_histograms` cube kernel — the D-04a determinism anchor.
//!
//! Transcribes the C++ accumulation body verbatim from
//! `LightGBM/src/io/dense_bin.hpp:99-141` (`ConstructHistogramInner`,
//! `USE_HESSIAN` path):
//!
//! ```cpp
//! hist_t* grad = out;        // hist_t = double
//! hist_t* hess = out + 1;
//! const auto ti = static_cast<uint32_t>(data(idx)) << 1;   // bin<<1
//! grad[ti] += ordered_gradients[i];   // f32 read, f64 accumulate
//! hess[ti] += ordered_hessians[i];
//! ```
//!
//! i.e. the histogram is laid out stride-2 interleaved `[g0,h0,g1,h1,…]` with
//! the grad cell at `bin*2` and the hess cell at `bin*2 + 1`. Gradients and
//! hessians are read as f32 (`score_t = float`) but summed into f64 cells
//! (`hist_t = double`, RESEARCH Pitfall 3).
//!
//! **Determinism mandate (RESEARCH Pitfall 1, D-04/D-04a):** cubecl-cpu spawns
//! one OS worker thread per cube unit — it is NOT a single-threaded sequential
//! executor. To make the f64 fold bit-stable (matching the C++ `num_threads=1`
//! ordered fold), the kernel is launched with `CubeDim::new_1d(1)` so exactly
//! ONE unit owns the entire fold, in row order. Any multi-unit accumulation
//! into shared cells (atomics) would be order-nondeterministic — and atomics
//! aren't supported on cubecl-cpu anyway.

use cubecl::prelude::*;

use crate::error::ComputeError;
use crate::runtime::ActiveRuntime;

/// The single shared `#[cube]` single-owner ordered histogram fold — the SINGLE
/// SOURCE OF TRUTH for the deterministic fold math (260608-n9j THE MERGE; the
/// structural analog of 260608-mc5's [`crate::kernels::split::split_scan_body`]).
///
/// Generic over the accumulation cell type `N: Numeric`: both the f64 cpu-anchor
/// launch kernel ([`construct_hist_kernel`], `N = f64`) AND the f32 hip-mirror
/// launch kernel ([`construct_hist_kernel_f32`], `N = f32`) call this helper, so
/// the single-owner ordered fold, the `UNIT_POS == 0` ownership, the ascending
/// row order, and the `bin<<1` stride-2 cell layout exist exactly ONCE. The ONLY
/// difference between the two launch entry points is the `N` it is instantiated
/// with — eliminating the prior hand-duplicated fold loop (a drift hazard).
///
/// Only `UNIT_POS == 0` executes the fold, in ascending row order, so the
/// summation order is fixed and matches the C++ `num_threads=1` reference
/// (RESEARCH Pitfall 1, D-04/D-04a).
///
/// Gradients/hessians are always READ as f32 (`score_t = float`); the cast
/// `N::cast_from(grad[i])` is the accumulation widening:
/// - For `N = f64` it is the f32→f64 widening — byte-identical to the prior
///   `f64::cast_from(grad[i])` (the bit-exact cpu anchor, Pitfall 3).
/// - For `N = f32` it is the identity cast — observably identical to the prior
///   `out[ti] += grad[i]` (the ~1e-6-tolerated hip mirror, Pitfall 2/3).
#[cube]
fn hist_fold_body<N: Numeric>(
    binned: &Array<u32>,
    grad: &Array<f32>,
    hess: &Array<f32>,
    out: &mut Array<N>,
) {
    // Single-owner ordered fold — the deterministic anchor (Pitfall 1).
    if UNIT_POS == 0 {
        for i in 0..binned.len() {
            // ti = bin<<1; grad cell at ti, hess cell at ti+1 (dense_bin.hpp:120).
            // `binned[i]` is u32; widen to usize for indexing the `out` array.
            let ti = binned[i] as usize * 2;
            out[ti] += N::cast_from(grad[i]); // f32 read, N-cell accumulate
            out[ti + 1] += N::cast_from(hess[i]);
        }
    }
}

/// The single-owner ordered f64 fold (RESEARCH Pattern 1 + `dense_bin.hpp`) — a
/// THIN `#[cube(launch)]` wrapper that delegates to the shared generic
/// [`hist_fold_body`] with `N = f64`. After the 260608-n9j merge this kernel
/// holds NO fold logic of its own; the math lives once in `hist_fold_body`,
/// shared with the f32 hip mirror.
///
/// This is the **cpu anchor** path: gradients/hessians are read as f32 but
/// summed into f64 cells (`hist_t = double`) — bit-exact vs C++ (Pitfall 3).
/// `f64::cast_from(grad[i])` is exactly what `N::cast_from` lowers to for
/// `N = f64`, so this is byte-identical to the pre-merge kernel.
#[cube(launch)]
pub fn construct_hist_kernel(
    binned: &Array<u32>,
    grad: &Array<f32>,
    hess: &Array<f32>,
    out: &mut Array<f64>,
) {
    hist_fold_body::<f64>(binned, grad, hess, out);
}

/// The f32-cell mirror of [`construct_hist_kernel`] for the no-f64 hip device
/// (RESEARCH Pitfall 2/3, CMP-04) — a THIN `#[cube(launch)]` wrapper that
/// delegates to the shared generic [`hist_fold_body`] with `N = f32`. IDENTICAL
/// fold structure and row order to the f64 kernel (they share the helper) — the
/// ONLY difference is the accumulation cell type (`f32` instead of `f64`). hip
/// (gfx1100) cannot allocate f64, so the histogram accumulates in f32, accepting
/// the ~1e-6-tolerated divergence from the cpu f64 anchor (the divergence the
/// oracle contract was designed to absorb, NOT a bug). The capability gate
/// (`has_f64 == false`) routes the hip launch here; cpu keeps the f64 kernel.
/// For `N = f32`, `N::cast_from(grad[i])` is the identity cast — observably
/// identical to the pre-merge `out[ti] += grad[i]`.
#[cube(launch)]
pub fn construct_hist_kernel_f32(
    binned: &Array<u32>,
    grad: &Array<f32>,
    hess: &Array<f32>,
    out: &mut Array<f32>,
) {
    hist_fold_body::<f32>(binned, grad, hess, out);
}

/// Host-side `construct_histograms` on the cpu reference runtime.
///
/// Validates every kernel input at the `Backend` boundary (Security V5, threat
/// T-04-01) BEFORE the `unsafe` launch, then runs the single-owner ordered fold
/// and returns the f64 histogram `[g0,h0,g1,h1,…]` of length `2 * num_bin`.
///
/// # Errors
/// - [`ComputeError::LengthMismatch`] if `grad`/`hess`/`binned` lengths differ.
/// - [`ComputeError::BinIndexOutOfRange`] if any `binned[i] >= num_bin`.
pub fn construct_histograms_cpu(
    client: &cubecl::prelude::ComputeClient<ActiveRuntime>,
    binned: &[u32],
    grad: &[f32],
    hess: &[f32],
    num_bin: u32,
) -> Result<Vec<f64>, ComputeError> {
    construct_histograms_f64_on(client, binned, grad, hess, num_bin)
}

/// The f64 `construct_histograms` cube path, **generic over the runtime** `R` so it
/// runs on the cubecl-cpu anchor (via [`construct_histograms_cpu`]) AND on
/// cubecl-hip (the GPU `RocmBackend`). The gfx1100 executes this f64 kernel
/// bit-exactly to the CPU anchor (verified: `max_abs_diff=0`), even though
/// `probe_capabilities().has_f64` is reported `false` — the flag is conservative,
/// the f64 op is real. Same single-owner ordered fold (`CubeDim::new_1d(1)`) and V5
/// validation as before.
///
/// # Errors
/// Same as [`construct_histograms_cpu`] (length / bin-range validation, V5).
pub fn construct_histograms_f64_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    binned: &[u32],
    grad: &[f32],
    hess: &[f32],
    num_bin: u32,
) -> Result<Vec<f64>, ComputeError> {
    // --- V5 boundary validation (T-04-01): never panic / UB on caller input ---
    // Shared with the f32 hip path (`construct_histograms_f32_on`); `out` is sized
    // 2 * num_bin cells, the `bin<<1` index math is overflow-guarded, and every
    // bin is range-checked.
    let out_len = validate_histogram_inputs(binned, grad, hess, num_bin)?;

    let n = binned.len();
    let h_bin = client.create_from_slice(u32::as_bytes(binned));
    let h_grad = client.create_from_slice(f32::as_bytes(grad));
    let h_hess = client.create_from_slice(f32::as_bytes(hess));
    // The kernel ACCUMULATES into `out` (`out[ti] += ...`), so `out` must start
    // zeroed. `client.empty` returns UNINITIALIZED device memory from the pool —
    // it may recycle a prior launch's buffer, so a fresh launch would fold on top
    // of stale values. Allocate from an explicit zero slice to match the C++
    // histogram being zeroed before accumulation.
    let zeros = vec![0.0f64; out_len];
    let h_out = client.create_from_slice(f64::as_bytes(&zeros));

    // SAFETY: `ArrayArg::from_raw_parts(handle, len)` requires that each handle
    // was allocated for exactly `len` elements of the declared element type and
    // outlives the launch. We just allocated `h_bin`/`h_grad`/`h_hess` from
    // slices of length `n` and `h_out` for `out_len` f64 cells, and the input
    // validation above guarantees every `binned[i] < num_bin` so the kernel's
    // `out[bin*2 + 1]` write stays within the `out_len` allocation (T-04-01/02).
    // All cubecl `unsafe` is confined to this crate (CMP-01).
    unsafe {
        construct_hist_kernel::launch(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1), // single unit owns the entire ordered fold (Pitfall 1)
            ArrayArg::from_raw_parts(h_bin, n),
            ArrayArg::from_raw_parts(h_grad, n),
            ArrayArg::from_raw_parts(h_hess, n),
            ArrayArg::from_raw_parts(h_out.clone(), out_len),
        );
    }

    let bytes = client.read_one_unchecked(h_out);
    Ok(f64::from_bytes(&bytes).to_vec())
}

/// **Native** host f64 fold — the production cpu-anchor path (R2).
///
/// Bit-IDENTICAL to [`construct_histograms_cpu`] (the single-unit
/// `construct_hist_kernel`): the exact same ascending-row-order accumulation of
/// `f32`-read gradients/hessians into `f64` cells, with the same `bin << 1` index
/// math and the same V5 boundary validation. The cubecl-cpu kernel launches that
/// loop as a `CubeDim::new_1d(1)` single owner — a fixed ~20–50µs dispatch cost
/// per call wrapping a trivial sequential loop. This native version drops that
/// overhead (5–210× faster per call; `probe_hist` measured bit_exact=true at
/// R=300/2000/20000) while producing byte-identical output, because the
/// arithmetic and order are the same.
///
/// `construct_histograms_cpu` is retained for the kernel-parity / ROCm-mirror
/// tests; the f32 hip path ([`construct_histograms_f32_on`]) is untouched.
///
/// # Errors
/// Same as [`construct_histograms_cpu`] (length / bin-range validation, V5).
pub fn construct_histograms_cpu_native(
    binned: &[u32],
    grad: &[f32],
    hess: &[f32],
    num_bin: u32,
) -> Result<Vec<f64>, ComputeError> {
    let out_len = validate_histogram_inputs(binned, grad, hess, num_bin)?;
    let mut out = vec![0.0f64; out_len];
    // Ascending row order, f32 read → f64 accumulate, grad at bin<<1 / hess at +1 —
    // the verbatim `construct_hist_kernel` body (dense_bin.hpp:99-141). The
    // validation above guarantees every `binned[i] < num_bin`, so `ti + 1` stays in
    // bounds; the loop uses checked indexing regardless (no `unsafe`).
    //
    // This loop is intentionally INLINE here (not delegated to
    // `accumulate_histogram_into`): the two bodies are kept byte-identical by the
    // `accumulate_into_is_bit_identical_to_native` unit test (f64::to_bits cell-by-
    // cell on multiple shapes) — a STRONGER drift guard than textual sharing, and one
    // that does not perturb this hot allocate-then-fold path's large-row codegen
    // (measured: routing through a shared `&mut [f64]` helper / a second validation
    // pass regressed the 200k-row build ~5%).
    for (i, &bin) in binned.iter().enumerate() {
        let ti = bin as usize * 2;
        out[ti] += f64::from(grad[i]);
        out[ti + 1] += f64::from(hess[i]);
    }
    Ok(out)
}

/// **Fold-in-place** native f64 accumulator — the per-feature build hot path (R3).
///
/// Folds the SAME ascending rows as [`construct_histograms_cpu_native`] directly
/// into a caller-owned, **caller-pre-zeroed** `out` sub-slice — NO intermediate
/// per-feature `Vec`, NO copy. This eliminates the per-feature alloc + memset +
/// memcpy that spike 002 localized as ~63% of low-row train time (build = 232µs/
/// iter vs C++ 44.5µs/iter).
///
/// The arithmetic is byte-for-byte identical to `construct_histograms_cpu_native`:
/// the same rows in the same ascending order, the same `bin << 1` cell layout, the
/// same `f32`-read → `f64`-accumulate. The only difference is that this writes into
/// a pre-zeroed sub-slice instead of allocating + zeroing its own buffer. The two
/// loop bodies are kept byte-identical — and the fold ORDER frozen (the project's
/// bit-exact CPU f64 merge gate depends on it) — by the
/// `accumulate_into_is_bit_identical_to_native` unit test (f64::to_bits cell-by-cell
/// on multiple shapes), NOT by textual sharing (a shared helper measurably regressed
/// the large-row native path; see `construct_histograms_cpu_native`).
///
/// `out` MUST be pre-zeroed by the caller over `out[0..2*num_bin]` (the caller owns
/// zeroing so it can zero a larger multi-feature buffer once). This function does
/// NOT zero `out`.
///
/// # Errors
/// - Same length / bin-range validation as [`construct_histograms_cpu_native`]
///   (V5, runs BEFORE any write).
/// - [`ComputeError::LengthMismatch`] if `out.len() < 2 * num_bin` (no panic, V5).
pub fn accumulate_histogram_into(
    binned: &[u32],
    grad: &[f32],
    hess: &[f32],
    num_bin: u32,
    out: &mut [f64],
) -> Result<(), ComputeError> {
    let out_len = validate_histogram_inputs(binned, grad, hess, num_bin)?;
    // V5: the caller's sub-slice must hold the full 2*num_bin histogram. Surface a
    // typed error (no panic) BEFORE writing any cell.
    if out.len() < out_len {
        return Err(ComputeError::LengthMismatch {
            expected: out_len,
            actual: out.len(),
        });
    }
    // Ascending row order, f32 read → f64 accumulate, grad at bin<<1 / hess at +1 —
    // the verbatim `construct_hist_kernel` body (dense_bin.hpp:99-141). The
    // validation above guarantees every `binned[i] < num_bin`, so `ti + 1` stays in
    // bounds; the loop uses checked indexing regardless (no `unsafe`). Folding into a
    // pre-zeroed sub-slice yields bytes identical to the allocating path.
    for (i, &bin) in binned.iter().enumerate() {
        let ti = bin as usize * 2;
        out[ti] += f64::from(grad[i]);
        out[ti + 1] += f64::from(hess[i]);
    }
    Ok(())
}

/// Validate the `construct_histograms` inputs (shared by the f64 cpu path and
/// the f32 hip path). Returns the histogram length `2 * num_bin` on success.
///
/// # Errors
/// - [`ComputeError::LengthMismatch`] if `grad`/`hess`/`binned` lengths differ.
/// - [`ComputeError::BinIndexOutOfRange`] if any `binned[i] >= num_bin`.
fn validate_histogram_inputs(
    binned: &[u32],
    grad: &[f32],
    hess: &[f32],
    num_bin: u32,
) -> Result<usize, ComputeError> {
    if grad.len() != binned.len() {
        return Err(ComputeError::LengthMismatch {
            expected: binned.len(),
            actual: grad.len(),
        });
    }
    if hess.len() != binned.len() {
        return Err(ComputeError::LengthMismatch {
            expected: binned.len(),
            actual: hess.len(),
        });
    }
    let out_len = 2usize
        .checked_mul(num_bin as usize)
        .ok_or_else(|| ComputeError::Runtime {
            detail: format!("num_bin {num_bin} overflows the histogram allocation size"),
        })?;
    for (row, &bin) in binned.iter().enumerate() {
        if bin >= num_bin {
            return Err(ComputeError::BinIndexOutOfRange { row, bin, num_bin });
        }
    }
    Ok(out_len)
}

/// Host-side `construct_histograms` in **f32 cells** on ANY runtime (the no-f64
/// hip path; CMP-03/CMP-04). Same V5 boundary validation and same single-owner
/// ordered fold as [`construct_histograms_cpu`], but accumulates into f32 cells
/// (hip cannot allocate f64). Returns the `2 * num_bin` f32 histogram. The hip
/// parity gate compares this against the cpu f64 anchor collected to `Vec<f32>`
/// within `ORACLE_TOL = 1e-6` (RESEARCH Pitfall 3, D-03a).
///
/// Generic over `R: Runtime` so it runs on the cubecl-cpu client (to produce the
/// f32 cpu reference in tests) AND on the cubecl-hip client (the real GPU path).
///
/// # Errors
/// Same as [`construct_histograms_cpu`] (length / bin-range validation, V5).
pub fn construct_histograms_f32_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    binned: &[u32],
    grad: &[f32],
    hess: &[f32],
    num_bin: u32,
) -> Result<Vec<f32>, ComputeError> {
    let out_len = validate_histogram_inputs(binned, grad, hess, num_bin)?;

    let n = binned.len();
    let h_bin = client.create_from_slice(u32::as_bytes(binned));
    let h_grad = client.create_from_slice(f32::as_bytes(grad));
    let h_hess = client.create_from_slice(f32::as_bytes(hess));
    let zeros = vec![0.0f32; out_len];
    let h_out = client.create_from_slice(f32::as_bytes(&zeros));

    // SAFETY: identical handle/length correspondence to `construct_histograms_cpu`
    // — `h_bin`/`h_grad`/`h_hess` sized `n`, `h_out` sized `out_len` f32 cells,
    // and the validation above keeps every `out[bin*2 + 1]` write in range. All
    // cubecl `unsafe` is confined to this crate (CMP-01).
    unsafe {
        construct_hist_kernel_f32::launch(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(h_bin, n),
            ArrayArg::from_raw_parts(h_grad, n),
            ArrayArg::from_raw_parts(h_hess, n),
            ArrayArg::from_raw_parts(h_out.clone(), out_len),
        );
    }

    let bytes = client.read_one_unchecked(h_out);
    Ok(f32::from_bytes(&bytes).to_vec())
}

/// PARALLEL f32-atomic histogram construction — the GPU-fast path (ROCm, ~1e-6).
///
/// Unlike the single-owner `construct_hist_kernel` (`CubeDim::new_1d(1)`, one lane
/// doing a sequential fold), this launches ONE unit PER ROW: each unit atomically
/// adds its row's gradient/hessian into the shared global histogram via f32
/// `fetch_add` (gfx1100 `has_f32_atomic == true`). This uses all the GPU's lanes.
///
/// Because the atomic adds commit in nondeterministic order and accumulate in f32,
/// the result diverges from the cpu f64 ordered anchor at the ~1e-6 level (the
/// ROCm gate the contract was designed for, D-03a) — NOT bit-exact. Feature-gated
/// to `rocm` so the CPU-only build never emits atomic codegen.
/// WARP-AGGREGATED f32-atomic histogram (quick-260619-p93) — a comptime-gated
/// `_plane` variant doing per-row f32-atomic scatter, NOT a replacement.
///
/// The baseline kernel issues `2*n` GLOBAL atomic `fetch_add`s (one grad + one
/// hess per row), so rows of a wave that share a destination `bin` serialize on
/// global-memory atomic contention. This variant optionally collapses the within-
/// plane same-bin adds into ONE global atomic per distinct bin per plane, cutting
/// up to `PLANE_DIM` adds to one — the classic CUDA "warp-aggregated atomics"
/// pattern (RESEARCH §correctness-crux).
///
/// `#[comptime] use_plane: bool` is a FEATURE-SPECIALIZATION flag (cubecl manual
/// "Feature Specialization with Comptime Flags"): each `use_plane` value compiles
/// a DISTINCT kernel binary with NO device branch on the flag. The CPU-only build
/// never compiles the `rocm` cfg, so it never emits ANY plane codegen.
///
/// - `use_plane == false`: the EXACT baseline body — `out[bin*2].fetch_add(grad)`,
///   `out[bin*2+1].fetch_add(hess)`. A byte-faithful twin of the shipped kernel
///   (the A/B baseline arm; see `examples/plane_aggregate_ab.rs`).
/// - `use_plane == true`: CORRECT same-bin warp aggregation. A naive whole-plane
///   `plane_sum` is WRONG here (each lane's `bin` is data-dependent and divergent;
///   summing all lanes and writing one bin corrupts the histogram — RESEARCH
///   Pitfall 1). cubecl 0.10 has NO `plane_match_any`, so lanes are grouped by
///   equal bin via a LEADER-ITERATION loop:
///     1. each lane holds its `bin` (active lanes only; the `idx < len` predicate
///        gates participation, so tail/out-of-range lanes never claim a group);
///     2. while any active lane is unclaimed: `plane_ballot(unclaimed && active)`
///        → pick the FIRST set lane (the ballot word selected + indexed by the
///        runtime `PLANE_DIM`, `trailing_zeros` for its lane id — NEVER a hardcoded
///        32, RESEARCH Pitfall 6), `plane_shuffle` that lane's bin to all lanes as
///        `leader_bin`;
///     3. `mine = unclaimed && active && (my_bin == leader_bin)`; the group's grad
///        / hess are reduced with a MASKED `plane_sum` (each non-member lane
///        contributes `0.0`);
///     4. the group's elected lane (its leader = the first unclaimed lane, itself a
///        member) issues a SINGLE `out[leader_bin*2].fetch_add(group_grad)` +
///        `out[leader_bin*2+1].fetch_add(group_hess)`;
///     5. `claimed |= mine`; repeat until no active lane is unclaimed.
///
/// **Parity:** the plane group reduction is an f32 TREE reduction — a DIFFERENT
/// accumulation order than the baseline's sequential atomic chain — but BOTH are
/// non-deterministic f32 accumulations held to the CPU f64 anchor at ABS 5e-6 /
/// REL 1e-5 (the gate D-03a / 04-ROCM-GAPS was designed for exactly this f32
/// reordering). A tree reduction is typically MORE accurate than a long sequential
/// chain, so the plane arm is expected to stay well inside the existing envelope.
/// This kernel does NOT touch the CPU f64 anchor ([`construct_hist_kernel`] /
/// [`construct_hist_kernel_f32`]) and widens no tolerance. `#[cfg(feature="rocm")]`.
#[cfg(feature = "rocm")]
#[cube(launch_unchecked)]
pub fn construct_hist_kernel_atomic_f32_plane(
    binned: &Array<u32>,
    grad: &Array<f32>,
    hess: &Array<f32>,
    out: &mut Array<Atomic<f32>>,
    #[comptime] use_plane: bool,
) {
    let idx = ABSOLUTE_POS;
    // Bounds check: the launch rounds the unit count up to a multiple of the cube
    // dim, so the tail units (idx >= len) must stay idle (manual §4 Safe Indexing).
    // `active` drives plane participation in the use_plane arm (tail lanes never
    // join a group); the non-plane arm just guards the scatter.
    let active = idx < binned.len();

    if use_plane {
        // --- CORRECT same-bin warp aggregation (RESEARCH §correctness-crux) ---
        // Active lanes load their bin/grad/hess; inactive (tail) lanes hold neutral
        // values and never become claimed (active==false keeps `mine` false).
        let mut my_bin = 0u32;
        let mut my_grad = 0.0f32;
        let mut my_hess = 0.0f32;
        if active {
            my_bin = binned[idx];
            my_grad = grad[idx];
            my_hess = hess[idx];
        }

        let lane = UNIT_POS_PLANE;
        // Inactive (tail) lanes start "claimed" so they never participate in a group;
        // active lanes start unclaimed.
        let mut claimed = !active;

        // Leader iteration: each pass elects one still-unclaimed bin and retires its
        // whole group. Bounded by PLANE_DIM passes (≥1 lane retired per pass). Loop
        // the runtime plane width so every lane runs the same trip count.
        let mut pass = 0u32;
        while pass < PLANE_DIM {
            // Is ANY active lane still unclaimed? (plane-wide predicate)
            let any_unclaimed = plane_any(!claimed);
            if any_unclaimed {
                // Ballot the still-unclaimed lanes; find the FIRST set lane. The
                // ballot is ALWAYS 4×u32 (128 bits) regardless of the runtime plane
                // width (Pitfall 6) — scan all 4 words; words beyond the active plane
                // are always 0, so they never produce a false leader (this is what
                // makes the scan PLANE_DIM-correct without hardcoding 32: a wave32
                // plane only ever sets bits in word 0, a wave64 plane in words 0..1).
                let ballot = plane_ballot(!claimed);
                let mut leader_word = 0usize;
                let mut found_word = false;
                let mut w = 0usize;
                while w < 4usize {
                    let bits = ballot[w];
                    if !found_word && bits != 0u32 {
                        leader_word = w;
                        found_word = true;
                    }
                    w += 1usize;
                }
                let leader_bits = ballot[leader_word];
                let leader_lane = leader_word as u32 * 32u32 + leader_bits.trailing_zeros();

                // Broadcast the leader lane's bin to all lanes. Dynamic source ⇒
                // plane_shuffle (plane_broadcast needs a const index).
                let leader_bin = plane_shuffle(my_bin, leader_lane);

                // Membership: a still-unclaimed lane whose bin equals the leader's
                // (inactive lanes are already claimed, so excluded).
                let mine = !claimed && (my_bin == leader_bin);

                // Masked group reduction (non-members contribute 0.0). A plane TREE
                // reduction — the f32 reorder the parity envelope was designed for.
                let mut cg = 0.0f32;
                let mut ch = 0.0f32;
                if mine {
                    cg = my_grad;
                    ch = my_hess;
                }
                let group_grad = plane_sum(cg);
                let group_hess = plane_sum(ch);

                // The leader lane (the first unclaimed lane, itself a group member by
                // construction) issues ONE global atomic per cell for the whole group.
                if lane == leader_lane {
                    let ti = leader_bin as usize * 2;
                    out[ti].fetch_add(group_grad);
                    out[ti + 1].fetch_add(group_hess);
                }

                // Retire the group.
                if mine {
                    claimed = true;
                }
            }
            pass += 1u32;
        }
    } else if active {
        // --- byte-faithful baseline body (the A/B baseline arm) ---
        let ti = binned[idx] as usize * 2; // grad cell at bin<<1, hess at +1
        out[ti].fetch_add(grad[idx]);
        out[ti + 1].fetch_add(hess[idx]);
    }
}

/// Host launcher for the warp-aggregated f32-atomic histogram (quick-260619-p93).
///
/// Uses the same V5 boundary checks, zeroed-f32 alloc, `ceil(n/256)` cube count of
/// 256 units, and f32→f64 widen on read-back as the LDS launcher, and launches
/// [`construct_hist_kernel_atomic_f32_plane`] with the `use_plane` comptime flag.
/// `use_plane == false` reproduces the shipped baseline arm BYTE-FAITHFULLY (so the
/// A/B bench can use ONE launcher for both arms, isolating exactly the warp-
/// aggregation codegen); `use_plane == true` selects the same-bin aggregation arm.
///
/// **`use_plane == true` requires plane collectives.** The caller MUST gate it on
/// the existing [`crate::runtime::probe_capabilities`]`(client).has_plane` (cpu =
/// false, gfx1100 = true) — this launcher does NOT re-probe; pass `use_plane=false`
/// (the byte-faithful baseline arm) when `has_plane` is false. Generic over
/// `R: Runtime` (runs on cubecl-hip). Kept a rocm-gated PRIMITIVE — wired into the
/// training path ONLY on a robust positive bench (see 260619-p93-FINDINGS.md).
///
/// # Errors
/// Same as [`construct_histograms_cpu`] (length / bin-range validation, V5).
#[cfg(feature = "rocm")]
pub fn construct_histograms_parallel_f32_plane_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    binned: &[u32],
    grad: &[f32],
    hess: &[f32],
    num_bin: u32,
    use_plane: bool,
) -> Result<Vec<f64>, ComputeError> {
    let out_len = validate_histogram_inputs(binned, grad, hess, num_bin)?;
    let n = binned.len();
    if n == 0 {
        return Ok(vec![0.0f64; out_len]);
    }
    let h_bin = client.create_from_slice(u32::as_bytes(binned));
    let h_grad = client.create_from_slice(f32::as_bytes(grad));
    let h_hess = client.create_from_slice(f32::as_bytes(hess));
    let zeros = vec![0.0f32; out_len];
    let h_out = client.create_from_slice(f32::as_bytes(&zeros));

    // One unit per row; cube dim 256 (8 × the gfx1100 wave32), cube count covers n.
    let cube_dim = 256u32;
    let cube_count = (n as u32).div_ceil(cube_dim);

    // SAFETY: identical handle/length correspondence to the LDS launcher —
    // `h_bin`/`h_grad`/`h_hess` sized `n`,
    // `h_out` sized `out_len` f32 cells, each outliving the launch. The kernel
    // bounds-checks `idx < n` (the `active` predicate), and input validation
    // guarantees every `binned[i] < num_bin` so `out[bin*2 + 1]` (baseline arm) and
    // `out[leader_bin*2 + 1]` (plane arm; `leader_bin` is some active lane's bin via
    // `plane_shuffle`) stay in the `out_len` allocation. All cubecl unsafe is
    // confined here (CMP-01).
    //
    // LAUNCH_UNCHECKED (NRW-01): `::launch_unchecked` drops the in-kernel per-access
    // bounds-check codegen; every device access is host-proven in range BEFORE
    // upload (the V5 checks discharge exactly the launch_unchecked obligations).
    // The launch does NOT change numerics — the `use_plane` flag selects the
    // accumulation ORDER (sequential atomics vs the f32 plane tree reduction); both
    // are the same nondeterministic ~1e-6 path held to the CPU f64 anchor.
    unsafe {
        construct_hist_kernel_atomic_f32_plane::launch_unchecked(
            client,
            CubeCount::Static(cube_count, 1, 1),
            CubeDim::new_1d(cube_dim),
            ArrayArg::from_raw_parts(h_bin, n),
            ArrayArg::from_raw_parts(h_grad, n),
            ArrayArg::from_raw_parts(h_hess, n),
            ArrayArg::from_raw_parts(h_out.clone(), out_len),
            use_plane,
        );
    }

    let bytes = client.read_one_unchecked(h_out);
    Ok(f32::from_bytes(&bytes).iter().map(|&x| f64::from(x)).collect())
}

/// LDS sub-histogram cap: `2 * 256` f32 cells = 2 KiB of shared memory per cube
/// (grad+hess interleaved for up to 256 bins). cubecl `SharedMemory::new` needs a
/// COMPTIME size, but num_bin varies at runtime (32/64/128/256); rather than
/// specialize one kernel binary per bin count (LightGBM's `histogram{16,64,256}.cl`
/// family approach), we allocate the fixed 256-bin max once (2 KiB ≪ the gfx1100
/// 64 KiB LDS budget) and drive the active length with the runtime `lds_len`.
///
/// Widened from `rocm` to the `gpu` umbrella (quick-260626-igc): the two LDS
/// construct items (`construct_hist_kernel_lds_f32` + `construct_histograms_lds_f32_on`)
/// that cuda/wgpu reuse reference this cap, so it must compile under any GPU backend.
/// It is a plain `usize` const — no `cubecl_hip_sys`/`rocm_client` involvement.
#[cfg(feature = "gpu")]
const HIST_LDS_MAX: usize = 512;

/// Fixed-point quantize scale S = 2^30 (phase-11, spike-018a). The resident LDS BUILD
/// accumulates `round(value * S)` as a two's-complement i64 stored BITS-as-u64 via
/// integer LDS atomics (`Atomic<u64>::fetch_add`), then `fix_compact_kernel` dequantizes
/// `(bits as i64) / S` back to f64 in its widen pass. S = 2^30 keeps ≥ ~9 fractional
/// bits while leaving the i64 magnitude safe to ~1e9 rows × |g| ≤ 8 (spike-018b). The
/// build-side constant is f32 (the quantize multiplies the f32 `ord_g`/`ord_h`); the
/// dequant-side `SCALE_F64` (in `fix_compact_kernel`) is the same value in f64.
#[cfg(feature = "rocm")]
const SCALE_F32: f32 = 1_073_741_824.0; // 2^30

/// Row-partition (`grid_dim_y` analog) tuning — spike-007 (`.planning/spikes/007-*`).
/// The LDS build launches one cube per feature; on a large leaf that is only
/// `num_features` workgroups, starving the GPU (gfx1100 = 96 CUs). Splitting a feature's
/// rows across `P` cubes raises occupancy. Spike measured a stable **1.3–1.4×** at
/// `P=16` (~8 workgroups/CU); **`P=32` over-partitioned and regressed**, so `P` is a tuned
/// target, never a maximize.
///
/// `ROWPART_MIN_LEAF`: below this leaf-row count, `P=1` (byte-identical to the pre-row-part
/// kernel). Well above the `RESIDENT_MIN_NUM_DATA=12_000` resident gate and the ≤8k-row
/// parity-test shapes, so every existing parity test runs the unchanged `P=1` path — the
/// large-leaf f32 divergence the spike found (4e-7→~2e-5 rel) only appears above this gate.
#[cfg(feature = "rocm")]
const ROWPART_MIN_LEAF: usize = 256_000;
/// Target cubes per Compute Unit — preserves spike-007's "~8 workgroups/CU" intent.
/// `target_cubes = num_cu * CUBES_PER_CU` (queried at runtime, cached once).
#[cfg(feature = "rocm")]
const CUBES_PER_CU: u32 = 8;
/// Documented safe small default for an APU-class device when EVERY CU-count query
/// fails — explicitly NOT 768 (which was the phantom-96-CU value: `8 wkgrps × 96 CU`).
/// 64 = `8 wkgrps × 8 CU`, matching the real 8-CU Radeon 860M APU on this box.
#[cfg(feature = "rocm")]
const ROWPART_TARGET_CUBES_FALLBACK: u32 = 64;
/// Spike-007 sweet spot; clamp so we never over-partition into the P=32 regression.
#[cfg(feature = "rocm")]
const ROWPART_P_MAX: u32 = 16;

/// Pure resolution of the row-partition target-cubes value, factored out so it is
/// unit-testable without an env var, a OnceLock, or a GPU (mirrors the "pure CPU
/// logic, no device handle" property of [`row_partition_count`]).
///
/// Resolution order (a)→(b)→(c):
///   (a) explicit env override (`LGBM_ROWPART_TARGET_CUBES`, >0) — used VERBATIM as the
///       literal target (NOT multiplied by `CUBES_PER_CU`); this is the A/B benching knob.
///   (b) queried device CU count → `num_cu * CUBES_PER_CU`.
///   (c) `ROWPART_TARGET_CUBES_FALLBACK` (never a silent 768).
#[cfg(feature = "rocm")]
fn resolve_target_cubes(env_override: Option<u32>, queried_cu: Option<u32>) -> u32 {
    match (env_override, queried_cu) {
        (Some(t), _) if t > 0 => t,
        (_, Some(n)) if n > 0 => n.saturating_mul(CUBES_PER_CU),
        _ => ROWPART_TARGET_CUBES_FALLBACK,
    }
}

/// Query the device's actual Compute Unit count, or `None` if unavailable.
///
/// 1. First try cubecl's reported value
///    (`rocm_client().properties().hardware.num_streaming_multiprocessors`):
///    forward-compatible — returns `None` on cubecl-hip 0.10 today, but populated on
///    cuda and possibly future hip. Used FIRST when `Some(n>0)`.
/// 2. else FFI fallback: read `hipGetDevicePropertiesR0600().multiProcessorCount` for
///    device ordinal 0 (matching `rocm_client`'s `AmdDevice::new(0)`).
#[cfg(feature = "rocm")]
fn query_num_cu() -> Option<u32> {
    // (1) cubecl's forward-compatible value (None on cubecl-hip 0.10, populated on cuda).
    if let Some(n) = crate::runtime::rocm_client()
        .properties()
        .hardware
        .num_streaming_multiprocessors
    {
        if n > 0 {
            return Some(n);
        }
    }

    // (2) FFI fallback via cubecl-hip-sys, mirroring cubecl-hip/src/runtime.rs:65-94.
    // SAFETY: `props` is zero-initialized then fully written by
    // `hipGetDevicePropertiesR0600` on a HIP_SUCCESS return; `multiProcessorCount` is
    // only read after the status is checked == HIP_SUCCESS, so we never read an
    // uninitialized struct. Device ordinal 0 matches `rocm_client`'s `AmdDevice::new(0)`.
    unsafe {
        let mut props: cubecl_hip_sys::hipDeviceProp_tR0600 = std::mem::zeroed();
        let status = cubecl_hip_sys::hipGetDevicePropertiesR0600(&mut props, 0);
        if status == cubecl_hip_sys::hipError_t_hipSuccess && props.multiProcessorCount > 0 {
            return Some(props.multiProcessorCount as u32);
        }
    }
    None
}

/// The row-partition target-cubes value, queried at most ONCE per process and cached
/// (`row_partition_count` runs per leaf, so the FFI CU-count query must not repeat —
/// T-jcr-02). Resolution: env override → queried CU count × `CUBES_PER_CU` → FALLBACK
/// (see [`resolve_target_cubes`] / [`query_num_cu`]).
///
/// Hardware note: this device is an 8-CU APU (Radeon 860M / gfx1152 spoofed as
/// gfx1100), so the target ≈ 64 (8 × `CUBES_PER_CU`), NOT 768 (which assumed a phantom
/// 96-CU gfx1100). Changing `P` alters the f32 partial-sum grouping (spike-007: `P≥2`
/// widens GPU-vs-`P=1` divergence to ~2e-5, WITHIN the ~1e-6-best-effort ROCm gate — the
/// GPU path was never bit-exact; the cpu f64 anchor is untouched).
#[cfg(feature = "rocm")]
fn rowpart_target_cubes() -> u32 {
    static TARGET: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *TARGET.get_or_init(|| {
        let env_override = std::env::var("LGBM_ROWPART_TARGET_CUBES")
            .ok()
            .and_then(|s| s.parse::<u32>().ok());
        resolve_target_cubes(env_override, query_num_cu())
    })
}

/// Row partitions `P` for the LDS build: `clamp(target_cubes / num_features, 1, P_MAX)` on
/// large leaves, else `1`. `LGBM_ROWPART_MIN` overrides the leaf threshold (benching). The
/// `target_cubes` value is the runtime, CU-count-derived [`rowpart_target_cubes`] (cached).
/// Pure CPU logic otherwise — no per-call device handle — so it is unit-testable with a
/// forced target.
#[cfg(feature = "rocm")]
pub fn row_partition_count(num_features: usize, leaf_rows: usize) -> u32 {
    let min_leaf = std::env::var("LGBM_ROWPART_MIN")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(ROWPART_MIN_LEAF);
    if num_features == 0 || leaf_rows < min_leaf {
        return 1;
    }
    let target = rowpart_target_cubes();
    let nf = num_features as u32;
    if nf >= target {
        return 1;
    }
    (target / nf).clamp(1, ROWPART_P_MAX)
}

/// PARALLEL f32 histogram with LDS-PRIVATIZED sub-histograms — the contention-
/// reducing GPU path (260609-f8u, eo5 Finding #2).
///
/// A naive per-row global-atomic scatter issues every row's grad/hess add
/// straight into the GLOBAL histogram, so rows sharing a bin (the common case)
/// serialize on global-memory atomic contention. This kernel instead gives each
/// CUBE its own sub-histogram in shared memory (LDS): all units in the cube
/// atomic-add into the LDS copy (intra-workgroup contention only — far cheaper than
/// global), then the cube merges its sub-histogram into the global output with ONE
/// global atomic per cell. Global atomic traffic drops from `2*n` to
/// `CUBE_COUNT * 2*num_bin`. This mirrors LightGBM's OpenCL `histogram*.cl` design.
///
/// Accumulation is f32 in nondeterministic order (same as the naive atomic path) ⇒
/// the SAME ~1e-6 ROCm gate vs the cpu f64 anchor (NOT bit-exact). Feature-gated to
/// `rocm`. The cpu f64 fold anchor ([`construct_hist_kernel`]) is untouched.
#[cfg(feature = "gpu")]
#[cube(launch_unchecked)]
pub fn construct_hist_kernel_lds_f32(
    binned: &Array<u32>,
    grad: &Array<f32>,
    hess: &Array<f32>,
    out: &mut Array<Atomic<f32>>,
    lds_len: u32, // active sub-hist length = 2*num_bin (<= HIST_LDS_MAX), runtime
) {
    // Per-cube private sub-histogram in shared memory (comptime-sized to the max).
    let sub = SharedMemory::<Atomic<f32>>::new(HIST_LDS_MAX);
    // Positions are usize for indexing (the per-row global-atomic scatter's
    // `binned[idx] as usize`); the builtins are u32 so cast once.
    let cd = CUBE_DIM as usize;
    let lds = lds_len as usize;
    let n = binned.len();

    // 1. Zero the ACTIVE LDS cells (strided across the cube's units).
    let mut c = UNIT_POS as usize;
    while c < lds {
        sub[c].store(0.0f32);
        c += cd;
    }
    sync_cube();

    // 2. Scatter this cube's strided rows into the LDS sub-histogram (LDS atomics).
    //    grad cell at bin<<1, hess at +1 (dense_bin.hpp:45 stride-2 layout).
    let stride = CUBE_COUNT_X as usize * cd;
    let mut i = CUBE_POS_X as usize * cd + UNIT_POS as usize;
    while i < n {
        let ti = binned[i] as usize * 2;
        sub[ti].fetch_add(grad[i]);
        sub[ti + 1].fetch_add(hess[i]);
        i += stride;
    }
    sync_cube();

    // 3. Merge the cube's sub-histogram into the global output (one global atomic
    //    per active cell — `CUBE_COUNT * lds_len` total, vs `2*n` for the naive path).
    let mut m = UNIT_POS as usize;
    while m < lds {
        out[m].fetch_add(sub[m].load());
        m += cd;
    }
}

/// Host launcher for the LDS-privatized parallel f32 histogram (GPU contention path).
///
/// Allocates a zeroed f32 histogram, launches `min(ceil(n/256), HIST_LDS_CUBES)`
/// cubes of 256 units (each owning a 2 KiB LDS sub-hist), and widens the f32 result
/// to f64. Capped at 256 bins (the LDS sub-hist size); `num_bin > 256` is rejected
/// so the caller can fall back to the naive atomic path. Generic over `R: Runtime`.
///
/// # Errors
/// Same as [`construct_histograms_cpu`] (length / bin-range validation, V5), plus a
/// [`ComputeError::Runtime`] when `num_bin > 256`.
#[cfg(feature = "gpu")]
pub fn construct_histograms_lds_f32_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    binned: &[u32],
    grad: &[f32],
    hess: &[f32],
    num_bin: u32,
) -> Result<Vec<f64>, ComputeError> {
    let out_len = validate_histogram_inputs(binned, grad, hess, num_bin)?;
    if num_bin > 256 {
        return Err(ComputeError::Runtime {
            detail: format!(
                "construct_histograms_lds: num_bin {num_bin} > 256 exceeds the LDS sub-hist cap \
                 (num_bin > 256 is unsupported by the LDS sub-hist path)"
            ),
        });
    }
    let n = binned.len();
    if n == 0 {
        return Ok(vec![0.0f64; out_len]);
    }
    let h_bin = client.create_from_slice(u32::as_bytes(binned));
    let h_grad = client.create_from_slice(f32::as_bytes(grad));
    let h_hess = client.create_from_slice(f32::as_bytes(hess));
    // MUST be zero-init: the merge step (step 3) GLOBAL-accumulates into `out`.
    let zeros = vec![0.0f32; out_len];
    let h_out = client.create_from_slice(f32::as_bytes(&zeros));

    // 256 units/cube (8 × wave32). Cube count = enough to cover the rows but capped
    // so the merge cost (cube_count * out_len global atomics) stays small relative
    // to the saved 2*n global atomics. ~96 ≈ gfx1100 CU count.
    let cube_dim = 256u32;
    let max_cubes = 96u32;
    let cube_count = (n as u32).div_ceil(cube_dim).clamp(1, max_cubes);

    // SAFETY: `h_bin`/`h_grad`/`h_hess` sized `n`, `h_out` sized `out_len` f32 cells,
    // each outliving the launch. The kernel strides `i` over `[0, n)` (bounds-checked
    // `i < n`), and input validation guarantees `binned[i] < num_bin <= 256` so every
    // `sub[bin*2 + 1]` / `out[bin*2 + 1]` index stays in `[0, out_len) ⊆ [0, 512)`.
    // All cubecl unsafe is confined here (CMP-01).
    //
    // LAUNCH_UNCHECKED (NRW-01): `::launch_unchecked` drops the in-kernel per-access
    // bounds-check codegen in the zero / scatter / merge loops. Every device access is
    // host-proven in range BEFORE upload:
    //   - `binned[i]` / `grad[i]` / `hess[i]` — the scatter strides `i` over `[0, n)`
    //     (the kernel's own `while i < n` guard); all three buffers sized `n`;
    //   - the LDS `sub[ti+1]` (`ti = binned[i]*2`) — `binned[i] < num_bin <= 256` so
    //     `ti+1 < 2*256 = HIST_LDS_MAX` (the comptime LDS size), and the zero/merge loops
    //     stride `c`/`m` over `[0, lds = out_len)`;
    //   - `out[m]` for `m < lds = out_len` — `h_out` sized `out_len`.
    // The host-side V5 checks discharge exactly the launch_unchecked obligations; the
    // launch does NOT change numerics — only bounds-check codegen is removed; the
    // f32-atomic LDS scatter / merge order is identical (~1e-6 path).
    unsafe {
        construct_hist_kernel_lds_f32::launch_unchecked(
            client,
            CubeCount::Static(cube_count, 1, 1),
            CubeDim::new_1d(cube_dim),
            ArrayArg::from_raw_parts(h_bin, n),
            ArrayArg::from_raw_parts(h_grad, n),
            ArrayArg::from_raw_parts(h_hess, n),
            ArrayArg::from_raw_parts(h_out.clone(), out_len),
            out_len as u32,
        );
    }

    let bytes = client.read_one_unchecked(h_out);
    Ok(f32::from_bytes(&bytes).iter().map(|&x| f64::from(x)).collect())
}

/// BATCHED per-leaf histogram: builds ALL features' RAW histograms for one leaf in
/// ONE launch (260608-lad part 3). One unit per `(feature, leaf-row)` pair, each
/// doing an f32 atomic add into that feature's region of the concatenated output.
/// This collapses the per-feature launch count to ONE launch per leaf — the launch
/// overhead (not the arithmetic) was the GPU bottleneck.
///
/// Layout (all host-gathered into flat buffers, so no division-by-scalar is needed
/// in-kernel — dims come from array lengths):
/// - `gathered_bins[f * R + k]` = feature `f`'s bin for the leaf's k-th row
///   (`R == ord_g.len()` == leaf-row count, `num_features == slot_off.len()`).
/// - `ord_g[k]` / `ord_h[k]` = the leaf's k-th row's gradient / hessian (gathered
///   once, shared across features).
/// - `slot_off[f]` = feature `f`'s start cell in the concatenated `out` buffer.
///
/// f32 atomics + nondeterministic order ⇒ the ~1e-6 ROCm gate (cpu anchor stays
/// bit-exact). `#[cfg(feature="rocm")]`.
#[cfg(feature = "rocm")]
#[cube(launch_unchecked)]
pub fn construct_leaf_hist_batched_kernel(
    gathered_bins: &Array<u32>,
    ord_g: &Array<f32>,
    ord_h: &Array<f32>,
    slot_off: &Array<u32>,
    out: &mut Array<Atomic<f32>>,
) {
    let idx = ABSOLUTE_POS;
    if idx < gathered_bins.len() {
        let r = ord_g.len(); // leaf-row count R
        let f = idx / r; // feature index
        let k = idx % r; // leaf-row position
        let cell = slot_off[f] as usize + gathered_bins[idx] as usize * 2;
        out[cell].fetch_add(ord_g[k]);
        out[cell + 1].fetch_add(ord_h[k]);
    }
}

/// Host launcher for the batched per-leaf histogram (the GPU-fast path, part 3).
///
/// Gathers each feature's leaf-row bins into one flat `[num_features × R]` buffer
/// and the leaf's grad/hess once, then dispatches a SINGLE kernel over all
/// `num_features × R` `(feature, row)` units. Returns the concatenated RAW f64
/// histogram (`slot_len` cells) — FixHistogram + compaction stay in the caller.
///
/// # Errors
/// [`ComputeError::Runtime`] on a degenerate layout (mismatched lengths).
#[cfg(feature = "rocm")]
pub fn build_leaf_histograms_batched_f32_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    feature_bins: &[&[u32]],
    slot_off: &[usize],
    slot_len: usize,
    leaf_rows: &[u32],
    gradients: &[f32],
    hessians: &[f32],
) -> Result<Vec<f64>, ComputeError> {
    let num_features = feature_bins.len();
    let rows = leaf_rows.len();
    if rows == 0 || num_features == 0 {
        return Ok(vec![0.0f64; slot_len]);
    }
    // Host gather: per-feature leaf-row bins (flat, feature-major) + per-row grad/hess.
    let mut gathered_bins: Vec<u32> = Vec::with_capacity(num_features * rows);
    for &bins in feature_bins {
        for &row in leaf_rows {
            gathered_bins.push(bins[row as usize]);
        }
    }
    let ord_g: Vec<f32> = leaf_rows.iter().map(|&r| gradients[r as usize]).collect();
    let ord_h: Vec<f32> = leaf_rows.iter().map(|&r| hessians[r as usize]).collect();
    let slot_off_u32: Vec<u32> = slot_off.iter().map(|&o| o as u32).collect();

    let h_bins = client.create_from_slice(u32::as_bytes(&gathered_bins));
    let h_g = client.create_from_slice(f32::as_bytes(&ord_g));
    let h_h = client.create_from_slice(f32::as_bytes(&ord_h));
    let zeros = vec![0.0f32; slot_len];
    let h_out = client.create_from_slice(f32::as_bytes(&zeros));

    let (slot_s, max_w) = slot_off_sentinel(slot_off, slot_len);
    if max_w <= HIST_LDS_MAX as u32 {
        // LDS-privatized per-feature path: `P` cubes per feature (row-partitioned, spike-007),
        // 256 units each. P=1 on small/medium leaves ⇒ byte-identical to the prior launch.
        let p = row_partition_count(num_features, rows);
        let h_slot = client.create_from_slice(u32::as_bytes(&slot_s));
        // SAFETY: handles sized to their slices; cube (f,p) reads gathered_bins[f*R..]
        // and writes only its slot region; bin < num_bin <= 256 keeps LDS/out indices
        // in range. cubecl unsafe confined here (CMP-01).
        //
        // LAUNCH_UNCHECKED (NRW-01): `::launch_unchecked` drops the in-kernel per-access
        // bounds-check codegen in the zero / scatter / merge loops. Every device access is
        // host-proven in range BEFORE upload:
        //   - `gathered_bins[fbase + k]` (`fbase = f*r`) — the host gather built
        //     `gathered_bins` as `[num_features * rows]` feature-major and the scatter
        //     strides `k` over `[0, r = rows)`, so `f*r + k < num_features*rows` for
        //     `f = CUBE_POS_X < num_features` (`h_bins` sized `num_features * rows`);
        //   - `ord_g[k]` / `ord_h[k]` for `k < r` (`h_g`/`h_h` sized `rows`);
        //   - `slot_off[f]` / `slot_off[f+1]` — `h_slot` has `num_features + 1` entries
        //     (sentinel), `f < num_features`;
        //   - the LDS `sub[ti+1]` and `out[base + m]` for `m < feat_len = slot_off[f+1] -
        //     slot_off[f]` — the LDS branch gate `max_w <= HIST_LDS_MAX` guarantees every
        //     feature width fits the comptime LDS size and stays inside its slot within
        //     `slot_len` (`h_out` sized `slot_len`).
        // The host-side V5 checks discharge exactly the launch_unchecked obligations; the
        // launch does NOT change numerics — only bounds-check codegen is removed; the
        // f32-atomic scatter / merge order is identical (~1e-6 path).
        unsafe {
            construct_leaf_hist_batched_lds_kernel::launch_unchecked(
                client,
                CubeCount::Static(num_features as u32, p, 1),
                CubeDim::new_1d(256),
                ArrayArg::from_raw_parts(h_bins, num_features * rows),
                ArrayArg::from_raw_parts(h_g, rows),
                ArrayArg::from_raw_parts(h_h, rows),
                ArrayArg::from_raw_parts(h_slot, num_features + 1),
                ArrayArg::from_raw_parts(h_out.clone(), slot_len),
            );
        }
    } else {
        // Naive fallback (a feature exceeds the 256-bin LDS cap).
        let h_slot = client.create_from_slice(u32::as_bytes(&slot_off_u32));
        let total = (num_features * rows) as u32;
        let cube_dim = 256u32;
        let cube_count = total.div_ceil(cube_dim);
        // SAFETY: every handle is sized to its slice and outlives the launch; the kernel
        // bounds-checks `idx < gathered_bins.len()`, and `gathered_bins[idx] < num_bin`
        // for the feature keeps `slot_off[f] + bin*2 + 1` inside that feature's slot
        // region within `slot_len`. All cubecl unsafe is confined here (CMP-01).
        //
        // LAUNCH_UNCHECKED (NRW-01): `::launch_unchecked` drops the in-kernel per-access
        // bounds-check codegen in the `num_features*R` scatter. Every device access is
        // host-proven in range BEFORE upload:
        //   - `gathered_bins[idx]` — `idx = ABSOLUTE_POS` is guarded by the kernel's own
        //     `idx < gathered_bins.len()` (`h_bins` sized `num_features * rows`);
        //   - `ord_g[k]` / `ord_h[k]` (`k = idx % r`, `r = ord_g.len() = rows`) — `k < r`
        //     (`h_g`/`h_h` sized `rows`);
        //   - `slot_off[f]` (`f = idx / r < num_features`) — `h_slot` sized `num_features`
        //     in this naive fallback;
        //   - `out[cell]` / `out[cell+1]` (`cell = slot_off[f] + bin*2`) — the host bins
        //     are bounded `bin < num_bin` for the feature, so the write stays inside that
        //     feature's slot within `slot_len` (`h_out` sized `slot_len`).
        // The host-side V5 checks discharge exactly the launch_unchecked obligations; the
        // launch does NOT change numerics — only bounds-check codegen is removed; the
        // f32-atomic scatter order is identical (~1e-6 path).
        unsafe {
            construct_leaf_hist_batched_kernel::launch_unchecked(
                client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new_1d(cube_dim),
                ArrayArg::from_raw_parts(h_bins, num_features * rows),
                ArrayArg::from_raw_parts(h_g, rows),
                ArrayArg::from_raw_parts(h_h, rows),
                ArrayArg::from_raw_parts(h_slot, num_features),
                ArrayArg::from_raw_parts(h_out.clone(), slot_len),
            );
        }
    }

    let bytes = client.read_one_unchecked(h_out);
    Ok(f32::from_bytes(&bytes).iter().map(|&x| f64::from(x)).collect())
}

/// DEVICE-RESIDENT batched per-leaf histogram kernel (260608-nn7 L1). Identical
/// `(feature f, leaf-row k)` unit mapping to [`construct_leaf_hist_batched_kernel`]
/// BUT gathers the bin from the device-resident feature-column buffer on device
/// (`resident_bins[f * num_data + leaf_rows[k]]`) instead of from a per-leaf
/// host-gathered `gathered_bins[idx]` re-uploaded every leaf. This is the L1 win:
/// the `[num_features × rows]` host bin upload per leaf is replaced by a one-time
/// column upload (the resident buffer) + a small per-leaf `leaf_rows` index upload.
///
/// `num_data` is the resident column stride (the full train row count) passed as a
/// scalar launch arg. Same f32-atomic accumulation ⇒ the ~1e-6 ROCm gate (cpu
/// anchor stays bit-exact). `#[cfg(feature="rocm")]`.
#[cfg(feature = "rocm")]
#[cube(launch_unchecked)]
pub fn construct_leaf_hist_resident_kernel<B: Int>(
    resident_bins: &Array<B>, // quick-260621-qix: native bin width (u8/u16/u32)
    leaf_rows: &Array<u32>,
    ord_g: &Array<f32>,
    ord_h: &Array<f32>,
    slot_off: &Array<u32>,
    num_data: usize,
    total: usize,
    out: &mut Array<Atomic<f32>>,
) {
    let idx = ABSOLUTE_POS;
    if idx < total {
        let r = ord_g.len(); // leaf-row count R
        let f = idx / r; // feature index
        let k = idx % r; // leaf-row position
        // Gather on device from the resident column: feature f, the leaf's k-th row.
        let row = leaf_rows[k] as usize;
        // quick-260621-qix: native-width read widened to a u32 INDEX (value-faithful).
        let bin = u32::cast_from(resident_bins[f * num_data + row]) as usize;
        let cell = slot_off[f] as usize + bin * 2;
        out[cell].fetch_add(ord_g[k]);
        out[cell + 1].fetch_add(ord_h[k]);
    }
}

/// Host launcher for the DEVICE-RESIDENT batched per-leaf histogram (260608-nn7 L1).
///
/// Takes the cached resident feature-column `Handle` (uploaded ONCE per train,
/// length `num_features * num_data`, feature-major) + the per-leaf `leaf_rows` index
/// array (uploaded fresh each leaf, small) + the leaf's grad/hess (gathered host-side
/// per leaf — small). Dispatches ONE kernel over all `num_features × R`
/// `(feature, row)` units; the kernel gathers each bin from the resident column on
/// device. Returns the concatenated RAW f64 histogram (`slot_len` cells) —
/// FixHistogram + compaction stay in the caller.
///
/// # Errors
/// [`ComputeError::Runtime`] on a degenerate layout (mismatched lengths).
#[cfg(feature = "rocm")]
#[allow(clippy::too_many_arguments)]
pub fn build_leaf_histograms_resident_f32_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    resident_bins: cubecl::server::Handle,
    // quick-260621-qix: native element width of `resident_bins`.
    width: crate::ResidentBinWidth,
    num_features: usize,
    num_data: usize,
    slot_off: &[usize],
    slot_len: usize,
    leaf_rows: &[u32],
    gradients: &[f32],
    hessians: &[f32],
) -> Result<Vec<f64>, ComputeError> {
    let rows = leaf_rows.len();
    if rows == 0 || num_features == 0 {
        return Ok(vec![0.0f64; slot_len]);
    }
    // Per-leaf uploads ONLY: the small leaf_rows index array + the leaf's grad/hess
    // (gathered once, shared across features). The big bins matrix is ALREADY on the
    // device (resident_bins). LDS-privatized per-feature build when every feature ≤
    // 256 bins (naive fallback otherwise) — see `resident_raw_build_into`.
    let zeros = vec![0.0f32; slot_len];
    let h_out = client.create_from_slice(f32::as_bytes(&zeros));
    resident_raw_build_into(
        client,
        resident_bins,
        width,
        num_features,
        num_data,
        slot_off,
        slot_len,
        leaf_rows,
        gradients,
        hessians,
        h_out.clone(),
        false, // f32 readback oracle: keep f32 atomics + f32 readback (phase-11)
    );

    let bytes = client.read_one_unchecked(h_out);
    Ok(f32::from_bytes(&bytes).iter().map(|&x| f64::from(x)).collect())
}

// ===========================================================================
// LDS-PRIVATIZED batched/resident RAW build (260609-fw1, eo5 Finding #2 — the
// hot-path follow-up to the single-feature `construct_hist_kernel_lds_f32`).
//
// The naive batched/resident kernels above run ONE unit per `(feature, leaf-row)`
// pair, each atomic-adding straight into the GLOBAL concatenated output — so rows
// of a feature sharing a bin serialize on global-memory atomic contention. These
// LDS kernels instead put ONE CUBE PER FEATURE: each cube owns a private
// sub-histogram in shared memory (≤ HIST_LDS_MAX = 2 KiB, one feature ≤ 256 bins),
// its units stride the leaf's rows doing cheap LDS atomics, then merge into that
// feature's global slot once per cell. Global atomic traffic per feature drops from
// `2*R` to `2*num_bin[f]`. Mirrors LightGBM's OpenCL histogram*.cl one-workgroup-
// per-feature design. `slot_off` carries a SENTINEL final entry (= slot_len) so the
// cube can read its feature's width `slot_off[f+1] - slot_off[f]`. f32 atomics ⇒ the
// same ~1e-6 ROCm gate; capped at 256 bins/feature (caller falls back to the naive
// kernel when any feature exceeds it).
// ===========================================================================

/// LDS resident RAW build: one cube per feature, gathers bins from the resident
/// column on device (`resident_bins[f*num_data + leaf_rows[k]]`). `slot_off` has
/// `num_features + 1` entries (sentinel = slot_len).
#[cfg(feature = "rocm")]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn construct_leaf_hist_resident_lds_kernel<B: Int>(
    resident_bins: &Array<B>, // quick-260621-qix: native bin width (u8/u16/u32)
    leaf_rows: &Array<u32>,
    ord_g: &Array<f32>,
    ord_h: &Array<f32>,
    slot_off: &Array<u32>, // length num_features + 1 (sentinel = slot_len)
    num_data: usize,
    out: &mut Array<Atomic<f32>>,
) {
    let f = CUBE_POS_X as usize; // ONE cube per feature
    let base = slot_off[f] as usize;
    let feat_len = slot_off[f + 1] as usize - base; // = 2*num_bin[f]
    let r = ord_g.len();
    let cd = CUBE_DIM as usize;

    let sub = SharedMemory::<Atomic<f32>>::new(HIST_LDS_MAX);
    // 1. zero this feature's active LDS cells.
    let mut c = UNIT_POS as usize;
    while c < feat_len {
        sub[c].store(0.0f32);
        c += cd;
    }
    sync_cube();
    // 2. scatter THIS partition's strided rows into LDS (resident on-device gather).
    //    Row-partitioned (260615 phase-09 / spike-007): `CubeCount = (num_features, P)`, so
    //    cube `(f, p)` owns row-slice `p*cd, +P*cd, …`. P comes from `CUBE_COUNT_Y`; P=1
    //    (the small/medium gate) reduces this to the prior `k=UNIT_POS, stride cd` loop
    //    byte-for-byte. All P cubes of feature f atomic-merge into the same global slot
    //    (step 3), so the split is additive and order-free.
    let col = f * num_data;
    let stride = CUBE_COUNT_Y as usize * cd;
    let mut k = CUBE_POS_Y as usize * cd + UNIT_POS as usize;
    while k < r {
        // quick-260621-qix: bin is read at native width B, widened to a u32 INDEX —
        // byte-faithful to the prior u32 read (value identical, only storage narrower).
        let bin = u32::cast_from(resident_bins[col + leaf_rows[k] as usize]) as usize;
        let ti = bin * 2;
        sub[ti].fetch_add(ord_g[k]);
        sub[ti + 1].fetch_add(ord_h[k]);
        k += stride;
    }
    sync_cube();
    // 3. merge LDS → this feature's global slot.
    let mut m = UNIT_POS as usize;
    while m < feat_len {
        out[base + m].fetch_add(sub[m].load());
        m += cd;
    }
}

/// u64 TWO'S-COMPLEMENT FIXED-POINT resident LDS build (phase-11, spike-018/019). A
/// byte-for-byte twin of [`construct_leaf_hist_resident_lds_kernel`] above — IDENTICAL
/// resident-column gather, `slot_off` sentinel `feat_len`, row-partition over
/// `CUBE_POS_Y`/`CUBE_COUNT_Y`, per-feature LDS sub-hist, and LDS→global merge — with
/// ONLY the accumulated CELL type + quantize/store/merge idiom swapped from f32 atomics
/// to u64 integer atomics.
///
/// Each grad/hess value is quantized `round(value * S)` (S = `SCALE_F32` = 2^30) to an
/// i64, whose BITS are stored as u64; a wrapping `Atomic<u64>::fetch_add` is exactly a
/// two's-complement signed i64 add, so the bins sum correctly with NO bias offset (each
/// bin sums a variable row count — a bias would be wrong, spike-018b). The dequant
/// `(bits as i64) / S` happens later in `fix_compact_kernel`'s widen pass.
///
/// HARD CONSTRAINT (spike-018b, CONTEXT line 24): the cell type MUST be `Atomic<u64>`
/// with `.store(0u64)` / `.fetch_add(qbits)`. NEVER `Atomic<i64>` — cubecl-hip 0.10
/// lowers `Atomic<i64>::store` to `atomicExch(long long*)`, which HIP lacks (compiles,
/// fails at runtime). LDS stays `HIST_LDS_MAX` cells (same element COUNT, 2× bytes =
/// 4 KiB/cube, well within the 64 KiB budget). f64 atomics not needed — the wide i64
/// accumulator IS the precision win (~3600× better than f32 atomics, spike-018a).
#[cfg(feature = "rocm")]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn construct_leaf_hist_resident_lds_kernel_u64<B: Int>(
    resident_bins: &Array<B>, // quick-260621-qix: native bin width (u8/u16/u32)
    leaf_rows: &Array<u32>,
    ord_g: &Array<f32>,
    ord_h: &Array<f32>,
    slot_off: &Array<u32>, // length num_features + 1 (sentinel = slot_len)
    num_data: usize,
    out: &mut Array<Atomic<u64>>,
) {
    let f = CUBE_POS_X as usize; // ONE cube per feature
    let base = slot_off[f] as usize;
    let feat_len = slot_off[f + 1] as usize - base; // = 2*num_bin[f]
    let r = ord_g.len();
    let cd = CUBE_DIM as usize;

    let sub = SharedMemory::<Atomic<u64>>::new(HIST_LDS_MAX);
    // 1. zero this feature's active LDS cells (u64 zero — the additive identity bits).
    let mut c = UNIT_POS as usize;
    while c < feat_len {
        sub[c].store(0u64);
        c += cd;
    }
    sync_cube();
    // 2. scatter THIS partition's strided rows into LDS (resident on-device gather),
    //    quantizing each value to fixed-point i64-bits before the wrapping u64 atomic.
    //    Row-partition byte-identical to the f32 twin (CubeCount = (num_features, P)).
    let col = f * num_data;
    let stride = CUBE_COUNT_Y as usize * cd;
    let mut k = CUBE_POS_Y as usize * cd + UNIT_POS as usize;
    while k < r {
        // bin INDEX read unchanged (native width B → u32 index); only the CELL type changed.
        let bin = u32::cast_from(resident_bins[col + leaf_rows[k] as usize]) as usize;
        let ti = bin * 2;
        // quantize `round(v * 2^30)` → i64 → store its bits as u64 (build_u64_rp idiom).
        let qg = u64::cast_from(i64::cast_from(f32::round(ord_g[k] * SCALE_F32)));
        let qh = u64::cast_from(i64::cast_from(f32::round(ord_h[k] * SCALE_F32)));
        sub[ti].fetch_add(qg);
        sub[ti + 1].fetch_add(qh);
        k += stride;
    }
    sync_cube();
    // 3. merge LDS → this feature's global slot (wrapping u64 add == i64 two's-complement).
    let mut m = UNIT_POS as usize;
    while m < feat_len {
        out[base + m].fetch_add(sub[m].load());
        m += cd;
    }
}

/// LDS batched RAW build: one cube per feature, reads host-gathered bins
/// (`gathered_bins[f*R + k]`). `slot_off` has `num_features + 1` entries (sentinel).
#[cfg(feature = "rocm")]
#[cube(launch_unchecked)]
pub fn construct_leaf_hist_batched_lds_kernel(
    gathered_bins: &Array<u32>, // [num_features * R], feature-major (f*R + k)
    ord_g: &Array<f32>,
    ord_h: &Array<f32>,
    slot_off: &Array<u32>, // length num_features + 1 (sentinel = slot_len)
    out: &mut Array<Atomic<f32>>,
) {
    let f = CUBE_POS_X as usize;
    let base = slot_off[f] as usize;
    let feat_len = slot_off[f + 1] as usize - base;
    let r = ord_g.len();
    let cd = CUBE_DIM as usize;

    let sub = SharedMemory::<Atomic<f32>>::new(HIST_LDS_MAX);
    let mut c = UNIT_POS as usize;
    while c < feat_len {
        sub[c].store(0.0f32);
        c += cd;
    }
    sync_cube();
    // Row-partitioned scatter (260615 phase-09): cube `(f, p)` strides row-slice
    // `p*cd, +P*cd, …`; P=`CUBE_COUNT_Y`. P=1 reduces to the prior `k=UNIT_POS, stride cd`.
    let fbase = f * r;
    let stride = CUBE_COUNT_Y as usize * cd;
    let mut k = CUBE_POS_Y as usize * cd + UNIT_POS as usize;
    while k < r {
        let ti = gathered_bins[fbase + k] as usize * 2;
        sub[ti].fetch_add(ord_g[k]);
        sub[ti + 1].fetch_add(ord_h[k]);
        k += stride;
    }
    sync_cube();
    let mut m = UNIT_POS as usize;
    while m < feat_len {
        out[base + m].fetch_add(sub[m].load());
        m += cd;
    }
}

// ===========================================================================
// FAITHFUL CubeCL MIRROR of CUDAConstructHistogramDenseKernel (quick-260619-j9t)
//
// Structural port of LightGBM's signature CUDA inner kernel
// (`LightGBM-release-4.6.0.99/src/treelearner/cuda/cuda_histogram_constructor.cu`
// lines ~18-70, `CUDAConstructHistogramDenseKernel`) — the histogram-construction
// kernel of `cuda_single_gpu_tree_learner`. It ships as a TESTED PRIMITIVE
// (rocm_cuda_mirror.rs), NOT wired into the production build/resident path (that
// live wiring is the deferred follow-up DEF-f8u-01).
//
// The signature CUDA structure this reproduces, distinct from the existing
// batched/resident LDS kernels (which pre-gather `ord_g[k]` / `leaf_rows[k]`):
//   (1) INDIRECT in-kernel gather — `data_index = data_indices[k]` (mirror of CUDA
//       `data_index = data_indices_ref_this_block[inner_data_index]`), then index a
//       RESIDENT feature-major bin buffer at `column_start * num_data + data_index`
//       (mirror of `data_ptr[data_index * ncols + tx]` via the resident
//       `f * num_data + row` layout RocmBackend::upload_resident_bins uses); grad/hess
//       are gathered in FULL-CORPUS order as `grad[data_index]` / `hess[data_index]`
//       (CUDA `cuda_gradients[data_index]`), NOT pre-gathered — the indirection the
//       existing kernels lack.
//   (2) 2D (column, row) tile — `CUBE_POS_X` selects the feature/column (CUDA
//       `threadIdx.x`/`blockIdx.x` partition column), `UNIT_POS` strides over the
//       leaf rows (CUDA `threadIdx.y`); row-partitioned over `CUBE_POS_Y` (CUDA
//       `blockIdx.y`) so a large leaf is split across `P` cubes (CUDA `gridDim.y`).
//   (3) per-CUBE LDS sub-histogram (CUDA `__shared__ shared_hist`,
//       `SharedMemory::<Atomic<f32>>::new(HIST_LDS_MAX)`): zero it strided, sync_cube,
//       atomic-add each row's (grad,hess) into the LDS cell at `bin*2` (CUDA
//       `atomicAdd_block`), sync_cube, then flush each active LDS cell to the global
//       `out` slot with ONE atomic per cell (CUDA `atomicAdd_system`).
//
// f32 atomics ⇒ nondeterministic accumulation ⇒ the ~1e-6 ROCm gate vs the CPU f64
// anchor (NOT bit-exact, by design — pinned in rocm_cuda_mirror.rs, never GPU-vs-GPU).
// Capped at 256 bins/feature (the HIST_LDS_MAX sub-hist budget). `#[cfg(feature="rocm")]`.
// ===========================================================================

/// One cube per `(feature f = CUBE_POS_X, row-partition p = CUBE_POS_Y)`. Mirrors
/// `CUDAConstructHistogramDenseKernel`: indirect in-kernel `data_indices` gather, a
/// resident feature-major bin buffer (`data[f*num_data + data_index]`), full-corpus
/// grad/hess gather (`grad[data_index]`), a per-cube LDS sub-histogram with atomic
/// accumulate, then a single global atomic flush per cell. `slot_off` has
/// `num_features + 1` entries (sentinel = slot_len) so cube `f` reads its feature's
/// width `slot_off[f+1] - slot_off[f] = 2*num_bin[f]`.
#[cfg(feature = "rocm")]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn construct_hist_cuda_mirror_kernel(
    data: &Array<u32>,         // resident feature-major bins: data[f*num_data + row]
    data_indices: &Array<u32>, // the leaf's row indices (data_indices_in_leaf)
    grad: &Array<f32>,         // FULL-corpus gradients (gathered in-kernel)
    hess: &Array<f32>,         // FULL-corpus hessians
    slot_off: &Array<u32>,     // length num_features + 1 (sentinel = slot_len)
    num_data: usize,           // resident column stride (full train row count)
    out: &mut Array<Atomic<f32>>,
) {
    let f = CUBE_POS_X as usize; // column/feature (CUDA threadIdx.x partition column)
    let base = slot_off[f] as usize;
    let feat_len = slot_off[f + 1] as usize - base; // = 2*num_bin[f]
    let r = data_indices.len(); // num_data_in_smaller_leaf
    let cd = CUBE_DIM as usize;
    let col = f * num_data; // resident column start (CUDA partition_column_start * num_data)

    // (3a) Per-cube LDS sub-histogram (CUDA __shared__ shared_hist), comptime-max-sized.
    let sub = SharedMemory::<Atomic<f32>>::new(HIST_LDS_MAX);
    // Zero this feature's active LDS cells, strided across the cube's units.
    let mut c = UNIT_POS as usize;
    while c < feat_len {
        sub[c].store(0.0f32);
        c += cd;
    }
    sync_cube();

    // (1)+(2) Indirect-gather scatter into LDS. Row-partitioned: cube (f, p) owns the
    // row-slice `p*cd, +P*cd, …` (CUDA blockIdx.y block_start striding); P=CUBE_COUNT_Y.
    // P=1 reduces to `k = UNIT_POS, stride cd`. All P cubes of feature f atomic-merge
    // into the same global slot (step 3), so the split is additive and order-free.
    let stride = CUBE_COUNT_Y as usize * cd;
    let mut k = CUBE_POS_Y as usize * cd + UNIT_POS as usize;
    while k < r {
        // CUDA: data_index = data_indices_ref_this_block[inner_data_index]
        let data_index = data_indices[k] as usize;
        // CUDA: cuda_gradients[data_index] / cuda_hessians[data_index] (full-corpus order)
        let g = grad[data_index];
        let h = hess[data_index];
        // CUDA: data_ptr[data_index * ncols + tx] via the resident f*num_data+row layout
        let bin = data[col + data_index] as usize;
        let ti = bin * 2; // grad cell at bin<<1, hess at +1 (dense_bin stride-2)
        sub[ti].fetch_add(g); // CUDA atomicAdd_block(pos_ptr, grad)
        sub[ti + 1].fetch_add(h); // CUDA atomicAdd_block(pos_ptr + 1, hess)
        k += stride;
    }
    sync_cube();

    // (3b) Flush LDS → this feature's global slot, ONE atomic per cell (CUDA
    // atomicAdd_system to feature_histogram_ptr). All P partitions accumulate here.
    let mut m = UNIT_POS as usize;
    while m < feat_len {
        out[base + m].fetch_add(sub[m].load());
        m += cd;
    }
}

/// Host launcher for the CUDA-mirror histogram kernel ([`construct_hist_cuda_mirror_kernel`]).
///
/// Mirrors `CUDAHistogramConstructor::LaunchConstructHistogramKernel`: one cube per
/// feature-column partition (`gridDim.x = num_features`), row-partitioned over
/// `CUBE_POS_Y` (CUDA `gridDim.y`) on large leaves (spike-007 occupancy). Validates
/// every input at the `Backend` boundary (Security V5 / T-j9t-01: bin-range + length
/// per feature) BEFORE the upload, early-returns zeros on an empty leaf, and widens
/// the f32 result to f64 (the learner's pool is f64; the ~1e-6 gate absorbs the gap).
///
/// `data` is the RESIDENT feature-major bin buffer (`data[f*num_data + row]`, length
/// `num_features * num_data`), `data_indices` the leaf's row indices, `grad`/`hess`
/// the FULL-corpus gradients/hessians (gathered in-kernel, length `num_data`),
/// `slot_off` the per-feature start cells (length `num_features`, sentinel appended
/// internally), `slot_len` the concatenated output length. Capped at 256 bins/feature
/// (the LDS sub-hist budget).
///
/// f32 atomics ⇒ ~1e-6 ROCm gate (documented; pinned vs the CPU f64 anchor in
/// rocm_cuda_mirror.rs, never GPU-vs-GPU — DEF-f8u-01).
///
/// # Errors
/// - [`ComputeError::LengthMismatch`] if a per-feature length is inconsistent.
/// - [`ComputeError::BinIndexOutOfRange`] if any leaf row's bin `>= num_bin`.
/// - [`ComputeError::Runtime`] on a degenerate layout, an out-of-range data index,
///   or `num_bin > 256` (LDS cap).
#[cfg(feature = "rocm")]
#[allow(clippy::too_many_arguments)]
pub fn construct_histograms_cuda_mirror_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    data: &[u32],
    num_data: usize,
    num_features: usize,
    data_indices: &[u32],
    grad: &[f32],
    hess: &[f32],
    slot_off: &[usize],
    slot_len: usize,
    num_bin: u32,
) -> Result<Vec<f64>, ComputeError> {
    // ---- V5 boundary validation (T-j9t-01) — BEFORE any device upload. ----
    if grad.len() != num_data || hess.len() != num_data {
        return Err(ComputeError::LengthMismatch {
            expected: num_data,
            actual: grad.len().min(hess.len()),
        });
    }
    if data.len() != num_features * num_data {
        return Err(ComputeError::Runtime {
            detail: format!(
                "cuda_mirror: resident data len {} != num_features {num_features} * num_data {num_data}",
                data.len()
            ),
        });
    }
    if slot_off.len() != num_features {
        return Err(ComputeError::Runtime {
            detail: format!(
                "cuda_mirror: slot_off len {} != num_features {num_features}",
                slot_off.len()
            ),
        });
    }
    if num_bin > 256 {
        return Err(ComputeError::Runtime {
            detail: format!("cuda_mirror: num_bin {num_bin} > 256 exceeds the LDS sub-hist cap"),
        });
    }
    // Every leaf data index must be in [0, num_data); every gathered bin in [0, num_bin).
    for &di in data_indices {
        let row = di as usize;
        if row >= num_data {
            return Err(ComputeError::Runtime {
                detail: format!("cuda_mirror: data_index {row} >= num_data {num_data}"),
            });
        }
        for f in 0..num_features {
            let bin = data[f * num_data + row];
            if bin >= num_bin {
                return Err(ComputeError::BinIndexOutOfRange { row, bin, num_bin });
            }
        }
    }

    // Early return on an empty leaf — no launch (mirror of the CUDA no-op path).
    if data_indices.is_empty() || num_features == 0 {
        return Ok(vec![0.0f64; slot_len]);
    }

    let rows = data_indices.len();
    let h_data = client.create_from_slice(u32::as_bytes(data));
    let h_idx = client.create_from_slice(u32::as_bytes(data_indices));
    let h_grad = client.create_from_slice(f32::as_bytes(grad));
    let h_hess = client.create_from_slice(f32::as_bytes(hess));
    let zeros = vec![0.0f32; slot_len];
    let h_out = client.create_from_slice(f32::as_bytes(&zeros));
    let (slot_s, _max_w) = slot_off_sentinel(slot_off, slot_len);
    let h_slot = client.create_from_slice(u32::as_bytes(&slot_s));

    // gridDim.x = num_features (one cube per feature-column partition); gridDim.y = P
    // row-partitions (spike-007 occupancy on a large leaf). 256 units/cube (8 × wave32).
    let p = row_partition_count(num_features, rows);

    // SAFETY: `h_data` is sized `num_features * num_data`, `h_idx`/(implicit grad/hess
    // index range) validated above so `data[f*num_data + data_indices[k]]` and
    // `grad[data_indices[k]]` stay in range; `h_grad`/`h_hess` sized `num_data`;
    // `h_slot` has `num_features + 1` entries; `h_out` sized `slot_len`. Each handle
    // outlives the launch, and the per-feature bin-range check keeps every
    // `slot_off[f] + bin*2 + 1` inside that feature's slot within `slot_len`. All
    // cubecl unsafe is confined to this launcher (CMP-01).
    //
    // LAUNCH_UNCHECKED (MWR-01): we call `::launch_unchecked`, which drops the
    // in-kernel per-access bounds-check codegen the manual emits for `::launch`.
    // This is sound because the full V5 boundary validation ABOVE already proves
    // every device access is in range BEFORE upload:
    //   - `data[col + data_index]`  — every `data_index = data_indices[k] < num_data`
    //     is checked, and `data.len() == num_features * num_data`, so `f*num_data +
    //     data_index < num_features*num_data` for `f in [0, num_features)`;
    //   - `grad[data_index]` / `hess[data_index]` — `grad.len()==hess.len()==num_data`
    //     and `data_index < num_data`;
    //   - `out[base + m]` for `m < feat_len` — every gathered `bin < num_bin <= 256`
    //     so `slot_off[f] + bin*2 + 1` stays inside that feature's slot within
    //     `slot_len`, and the LDS sub-hist write `sub[bin*2 + 1]` stays within
    //     `HIST_LDS_MAX` (num_bin<=256).
    // i.e. the host-side checks discharge exactly the obligations the launch_unchecked
    // contract requires, and the launch does NOT change numerics — only bounds-check
    // codegen is removed; the scatter order / f32-atomic accumulation is identical.
    unsafe {
        construct_hist_cuda_mirror_kernel::launch_unchecked(
            client,
            CubeCount::Static(num_features as u32, p, 1),
            CubeDim::new_1d(256),
            ArrayArg::from_raw_parts(h_data, num_features * num_data),
            ArrayArg::from_raw_parts(h_idx, rows),
            ArrayArg::from_raw_parts(h_grad, num_data),
            ArrayArg::from_raw_parts(h_hess, num_data),
            ArrayArg::from_raw_parts(h_slot, num_features + 1),
            num_data,
            ArrayArg::from_raw_parts(h_out.clone(), slot_len),
        );
    }

    let bytes = client.read_one_unchecked(h_out);
    Ok(f32::from_bytes(&bytes).iter().map(|&x| f64::from(x)).collect())
}

/// **Upload-once / CUDA-faithful** resident-`Handle` variant of
/// [`construct_histograms_cuda_mirror_on`] (MWR-02).
///
/// The per-call launcher above `create_from_slice`s the FULL feature-major bin buffer
/// (`num_features * num_data` u32 = the dominant transfer) on EVERY call. The real
/// CUDA `cuda_single_gpu_tree_learner` keeps the binned data RESIDENT on device,
/// uploaded ONCE per train; only the per-leaf `data_indices` (+ the per-iteration
/// grad/hess) change between calls. This variant mirrors that model: the caller is
/// responsible for having uploaded the feature-major bin buffer ONCE (length
/// `num_features * num_data`) and passes its device `Handle` — this function does NOT
/// re-upload it. Per call it uploads ONLY `data_indices`, `grad`, `hess`, the sentinel
/// `slot_off`, and the zeroed `out`. This is the same upload-once pattern as
/// [`build_leaf_histograms_resident_f32_on`].
///
/// Launches the SAME [`construct_hist_cuda_mirror_kernel`] with the SAME `CubeCount` /
/// `CubeDim` / argument order as the per-call launcher — only the `data` source (a
/// pre-uploaded `Handle` instead of a fresh `create_from_slice`) differs — so the
/// numerics are identical (the f32-atomic scatter order / contention model is
/// unchanged). f32 atomics ⇒ the ~1e-6 ROCm gate; the result is widened to f64 on
/// read-back.
///
/// This is a TESTED PRIMITIVE (rocm_cuda_mirror.rs), NOT wired into the production
/// histogram / build path (that live wiring is the deferred follow-up DEF-f8u-01).
///
/// Because the bins are NOT on the host here, this variant CANNOT run the per-feature
/// bin-range scan the per-call variant does (that scan reads `data[...]` from the host
/// slice). It validates everything reachable host-side instead: `grad.len()==num_data`,
/// `hess.len()==num_data`, `slot_off.len()==num_features`, `num_bin <= 256`, every
/// `data_indices[k] < num_data`, and `num_features != 0`. The bin-range invariant
/// (every resident bin `< num_bin`) is the CALLER's upload-time responsibility — the
/// resident buffer must have been validated when it was built/uploaded (the same
/// contract as [`build_leaf_histograms_resident_f32_on`], whose resident `Handle` was
/// validated at `upload_resident_bins` time).
///
/// # Errors
/// - [`ComputeError::LengthMismatch`] if `grad`/`hess` length `!= num_data`.
/// - [`ComputeError::Runtime`] on a degenerate layout (`slot_off.len() != num_features`,
///   `num_features == 0` with a non-empty leaf), an out-of-range `data_index`, or
///   `num_bin > 256` (LDS cap).
#[cfg(feature = "rocm")]
#[allow(clippy::too_many_arguments)]
pub fn construct_histograms_cuda_mirror_resident_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    resident_bins: cubecl::server::Handle,
    num_data: usize,
    num_features: usize,
    data_indices: &[u32],
    grad: &[f32],
    hess: &[f32],
    slot_off: &[usize],
    slot_len: usize,
    num_bin: u32,
) -> Result<Vec<f64>, ComputeError> {
    // ---- V5 boundary validation (T-mwr-01) — everything reachable host-side. ----
    // The bin-range scan the per-call variant runs is NOT possible here (bins are
    // resident on device, not on the host); that invariant is the caller's upload-time
    // responsibility (mirrors `build_leaf_histograms_resident_f32_on`).
    if grad.len() != num_data || hess.len() != num_data {
        return Err(ComputeError::LengthMismatch {
            expected: num_data,
            actual: grad.len().min(hess.len()),
        });
    }
    if slot_off.len() != num_features {
        return Err(ComputeError::Runtime {
            detail: format!(
                "cuda_mirror_resident: slot_off len {} != num_features {num_features}",
                slot_off.len()
            ),
        });
    }
    if num_bin > 256 {
        return Err(ComputeError::Runtime {
            detail: format!(
                "cuda_mirror_resident: num_bin {num_bin} > 256 exceeds the LDS sub-hist cap"
            ),
        });
    }
    for &di in data_indices {
        let row = di as usize;
        if row >= num_data {
            return Err(ComputeError::Runtime {
                detail: format!("cuda_mirror_resident: data_index {row} >= num_data {num_data}"),
            });
        }
    }

    // Early return on an empty leaf — no launch (mirror of the CUDA no-op path).
    if data_indices.is_empty() || num_features == 0 {
        return Ok(vec![0.0f64; slot_len]);
    }

    let rows = data_indices.len();
    // Per-call uploads ONLY: the small per-leaf data_indices + the per-iter grad/hess +
    // the sentinel slot_off + the zeroed out. The big feature-major bin buffer is
    // ALREADY on the device (`resident_bins`) — NOT re-uploaded here (the MWR-02 win).
    let h_idx = client.create_from_slice(u32::as_bytes(data_indices));
    let h_grad = client.create_from_slice(f32::as_bytes(grad));
    let h_hess = client.create_from_slice(f32::as_bytes(hess));
    let zeros = vec![0.0f32; slot_len];
    let h_out = client.create_from_slice(f32::as_bytes(&zeros));
    let (slot_s, _max_w) = slot_off_sentinel(slot_off, slot_len);
    let h_slot = client.create_from_slice(u32::as_bytes(&slot_s));

    let p = row_partition_count(num_features, rows);

    // SAFETY: same handle/length correspondence and the SAME launch config as
    // `construct_histograms_cuda_mirror_on` — `resident_bins` is the pre-uploaded
    // feature-major buffer sized `num_features * num_data` (the caller's contract),
    // `h_idx` sized `rows`, `h_grad`/`h_hess` sized `num_data`, `h_slot` has
    // `num_features + 1` entries, `h_out` sized `slot_len`; each handle outlives the
    // launch. We use `::launch_unchecked` (drops the in-kernel bounds-check codegen,
    // MWR-01): host-side V5 proves `data_indices[k] < num_data` (so `grad[data_index]`,
    // `hess[data_index]`, and `data[f*num_data + data_index]` are in range), and the
    // per-feature BIN-RANGE invariant (`bin < num_bin <= 256`, keeping `out[base+m]`
    // for `m < feat_len` inside the feature's slot and `sub[bin*2+1]` inside
    // `HIST_LDS_MAX`) is the CALLER's upload-time responsibility for the resident
    // buffer — exactly as in `build_leaf_histograms_resident_f32_on`. All cubecl
    // unsafe is confined to this launcher (CMP-01).
    unsafe {
        construct_hist_cuda_mirror_kernel::launch_unchecked(
            client,
            CubeCount::Static(num_features as u32, p, 1),
            CubeDim::new_1d(256),
            ArrayArg::from_raw_parts(resident_bins, num_features * num_data),
            ArrayArg::from_raw_parts(h_idx, rows),
            ArrayArg::from_raw_parts(h_grad, num_data),
            ArrayArg::from_raw_parts(h_hess, num_data),
            ArrayArg::from_raw_parts(h_slot, num_features + 1),
            num_data,
            ArrayArg::from_raw_parts(h_out.clone(), slot_len),
        );
    }

    let bytes = client.read_one_unchecked(h_out);
    Ok(f32::from_bytes(&bytes).iter().map(|&x| f64::from(x)).collect())
}

/// Build the sentinel `slot_off` (`num_features + 1` entries, final = `slot_len`)
/// and the max per-feature slot width. LDS is eligible iff the widest feature fits
/// `HIST_LDS_MAX` (≤ 256 bins).
#[cfg(feature = "rocm")]
fn slot_off_sentinel(slot_off: &[usize], slot_len: usize) -> (Vec<u32>, u32) {
    let mut s: Vec<u32> = Vec::with_capacity(slot_off.len() + 1);
    for &o in slot_off {
        s.push(o as u32);
    }
    s.push(slot_len as u32);
    let max_w = s.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0);
    (s, max_w)
}

/// Shared RESIDENT RAW-build launch into a caller-provided zeroed `h_out`
/// (slot_len cells): LDS per-feature path when every feature ≤ 256 bins, else the
/// naive `construct_leaf_hist_resident_kernel`. Used by both
/// [`build_leaf_histograms_resident_f32_on`] and the resident chain
/// [`build_fix_compact_resident_f64_on`] so the LDS/naive decision lives in ONE place.
///
/// `fixed_point` selects the LDS BUILD cell type (phase-11):
///   - `false` → f32 atomics ([`construct_leaf_hist_resident_lds_kernel`]); `h_out` is an
///     f32 buffer the caller reads back / widens as f32 (the readback oracle path).
///   - `true`  → u64 two's-complement fixed-point atomics
///     ([`construct_leaf_hist_resident_lds_kernel_u64`]); `h_out` is a u64 buffer the
///     caller dequantizes `(bits as i64)/2^30 → f64` (the live `fix_compact_kernel` path).
/// The NAIVE >256-bin fallback ALWAYS stays f32 (CONTEXT Claude's-discretion); a caller
/// that requests `fixed_point` MUST guarantee every feature ≤ 256 bins (the resident
/// chain does — `max_bin ≤ 255` keeps `max_w ≤ HIST_LDS_MAX`), else the f32 naive write
/// would be mis-dequantized. The `fixed_point` path asserts the LDS branch was taken.
#[cfg(feature = "rocm")]
#[allow(clippy::too_many_arguments)]
fn resident_raw_build_into<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    resident_bins: cubecl::server::Handle,
    // quick-260621-qix: element width of `resident_bins` (dispatches the kernel `<B: Int>`).
    width: crate::ResidentBinWidth,
    num_features: usize,
    num_data: usize,
    slot_off: &[usize],
    slot_len: usize,
    leaf_rows: &[u32],
    gradients: &[f32],
    hessians: &[f32],
    h_out: cubecl::server::Handle,
    // phase-11: true → u64 fixed-point LDS build; false → f32 atomics. Naive fallback is
    // always f32, so callers with `fixed_point=true` must keep every feature ≤ 256 bins.
    fixed_point: bool,
) {
    let rows = leaf_rows.len();
    if rows == 0 || num_features == 0 {
        return;
    }
    let ord_g: Vec<f32> = leaf_rows.iter().map(|&r| gradients[r as usize]).collect();
    let ord_h: Vec<f32> = leaf_rows.iter().map(|&r| hessians[r as usize]).collect();
    let h_rows = client.create_from_slice(u32::as_bytes(leaf_rows));
    let h_g = client.create_from_slice(f32::as_bytes(&ord_g));
    let h_h = client.create_from_slice(f32::as_bytes(&ord_h));
    let (slot_s, max_w) = slot_off_sentinel(slot_off, slot_len);

    if max_w <= HIST_LDS_MAX as u32 {
        // LDS per-feature path: `P` cubes per feature (row-partitioned, spike-007), 256 units
        // each. P=1 on small/medium leaves ⇒ byte-identical to the prior one-cube-per-feature
        // launch. The grid's y-dim carries P; each cube of feature f atomic-merges into the
        // same global slot (additive).
        let p = row_partition_count(num_features, rows);
        let h_slot = client.create_from_slice(u32::as_bytes(&slot_s));
        // SAFETY: resident_bins sized num_features*num_data; h_rows/h_g/h_h sized rows;
        // h_slot sized num_features+1; h_out sized slot_len. Cube (f,p) reads only its
        // feature's column + slot region; bin < num_bin <= 256 keeps LDS/out indices
        // in range. All cubecl unsafe is confined here (CMP-01).
        //
        // LAUNCH_UNCHECKED (NRW-01, copied from the mirror template at the cuda-mirror
        // launcher): we call `::launch_unchecked`, dropping the in-kernel per-access
        // bounds-check codegen `::launch` emits in the scatter hot loops. This is sound
        // because every device access is host-proven in range BEFORE upload:
        //   - `resident_bins[col + leaf_rows[k]]` (`col = f*num_data`) — `leaf_rows` ⊂
        //     `[0, num_data)` (the caller's resident contract / upload-time validation)
        //     and `resident_bins.len() == num_features*num_data`, so for `f in
        //     [0, num_features)` the index `f*num_data + leaf_rows[k] < num_features*num_data`;
        //   - `slot_off[f]` / `slot_off[f+1]` — `h_slot` has `num_features + 1` entries
        //     (sentinel = slot_len) and `f = CUBE_POS_X < num_features`;
        //   - `ord_g[k]` / `ord_h[k]` — `h_g`/`h_h` sized `rows`, `k < r = ord_g.len()`;
        //   - the LDS `sub[bin*2 + 1]` stays within `HIST_LDS_MAX` (every feature
        //     `num_bin <= 256`, the `max_w <= HIST_LDS_MAX` branch gate);
        //   - `out[base + m]` for `m < feat_len = slot_off[f+1] - slot_off[f]` stays inside
        //     that feature's slot within `slot_len` (`h_out` sized slot_len).
        // i.e. the host-side V5 checks discharge exactly the obligations the
        // launch_unchecked contract requires, and the launch does NOT change numerics —
        // only bounds-check codegen is removed; scatter order / f32-atomic accumulation
        // is identical.
        // quick-260621-qix: dispatch the matching `<B: Int>` monomorphization on the
        // resident buffer's native width. ArrayArg element COUNT is width-independent
        // (num_features*num_data); the launch generic pins the element TYPE. Only one
        // match arm executes ⇒ the by-value handle moves are exclusive (sound).
        // phase-11: the LDS per-feature path dispatches EITHER the u64 FIXED-POINT build
        // kernel (`fixed_point`, the live `fix_compact_kernel` chain) OR the f32-atomic
        // original (the readback oracle). Both kernels share an IDENTICAL signature
        // (resident gather, row-partition, slot args) — only the `out` cell type differs
        // (Atomic<u64> vs Atomic<f32>), matching the `h_out` buffer the caller allocated.
        // The naive >256-bin fallback arm below ALWAYS stays f32; a `fixed_point` caller
        // must keep every feature ≤ 256 bins (asserted) so the u64 LDS branch is taken.
        macro_rules! launch_lds_u64 {
            ($w:ty) => {
                unsafe {
                    construct_leaf_hist_resident_lds_kernel_u64::launch_unchecked::<$w, R>(
                        client,
                        CubeCount::Static(num_features as u32, p, 1),
                        CubeDim::new_1d(256),
                        ArrayArg::from_raw_parts(resident_bins.clone(), num_features * num_data),
                        ArrayArg::from_raw_parts(h_rows.clone(), rows),
                        ArrayArg::from_raw_parts(h_g.clone(), rows),
                        ArrayArg::from_raw_parts(h_h.clone(), rows),
                        ArrayArg::from_raw_parts(h_slot.clone(), num_features + 1),
                        num_data,
                        ArrayArg::from_raw_parts(h_out.clone(), slot_len),
                    );
                }
            };
        }
        macro_rules! launch_lds_f32 {
            ($w:ty) => {
                unsafe {
                    construct_leaf_hist_resident_lds_kernel::launch_unchecked::<$w, R>(
                        client,
                        CubeCount::Static(num_features as u32, p, 1),
                        CubeDim::new_1d(256),
                        ArrayArg::from_raw_parts(resident_bins.clone(), num_features * num_data),
                        ArrayArg::from_raw_parts(h_rows.clone(), rows),
                        ArrayArg::from_raw_parts(h_g.clone(), rows),
                        ArrayArg::from_raw_parts(h_h.clone(), rows),
                        ArrayArg::from_raw_parts(h_slot.clone(), num_features + 1),
                        num_data,
                        ArrayArg::from_raw_parts(h_out.clone(), slot_len),
                    );
                }
            };
        }
        if fixed_point {
            match width {
                crate::ResidentBinWidth::U8 => launch_lds_u64!(u8),
                crate::ResidentBinWidth::U16 => launch_lds_u64!(u16),
                crate::ResidentBinWidth::U32 => launch_lds_u64!(u32),
            }
        } else {
            match width {
                crate::ResidentBinWidth::U8 => launch_lds_f32!(u8),
                crate::ResidentBinWidth::U16 => launch_lds_f32!(u16),
                crate::ResidentBinWidth::U32 => launch_lds_f32!(u32),
            }
        }
    } else {
        // Naive fallback (a feature exceeds the 256-bin LDS cap). ALWAYS f32 (phase-11):
        // the u64 fixed-point kernel is LDS-only. A `fixed_point` caller dequantizes
        // `h_out` as u64 downstream, so an f32 naive write here would be mis-decoded —
        // guard the (can't-happen for max_bin ≤ 255) case loudly rather than corrupt.
        assert!(
            !fixed_point,
            "resident_raw_build_into: fixed_point u64 build requires every feature ≤ 256 \
             bins (max_w ≤ HIST_LDS_MAX); a feature exceeded the LDS cap so the f32 naive \
             fallback fired, which the u64 dequant would mis-decode. Raise max_bin handling \
             or keep the resident chain ≤ 256 bins (phase-11 spike-018)."
        );
        let slot_off_u32: Vec<u32> = slot_off.iter().map(|&o| o as u32).collect();
        let h_slot = client.create_from_slice(u32::as_bytes(&slot_off_u32));
        let total = num_features * rows;
        let cube_dim = 256u32;
        let cube_count = (total as u32).div_ceil(cube_dim);
        // SAFETY: identical to the prior in-place naive launch (idx<total bound,
        // resident read in range, slot write in range). cubecl unsafe confined (CMP-01).
        //
        // LAUNCH_UNCHECKED (NRW-01): `::launch_unchecked` drops the in-kernel per-access
        // bounds-check codegen in the `total`-wide scatter. Every device access is
        // host-proven in range BEFORE upload:
        //   - all units guarded by the kernel's own `idx < total` (`total = num_features *
        //     rows`, the cube count rounds up so tail units stay idle);
        //   - `resident_bins[f*num_data + row]` (`f = idx/r < num_features`, `row =
        //     leaf_rows[k]`) — `leaf_rows` ⊂ `[0, num_data)` (caller resident contract)
        //     and `resident_bins.len() == num_features*num_data`, so the index is in range;
        //   - `leaf_rows[k]` / `ord_g[k]` / `ord_h[k]` (`k = idx % r`, `r = ord_g.len() =
        //     rows`) — `k < r` (`h_rows`/`h_g`/`h_h` sized `rows`);
        //   - `slot_off[f]` for `f < num_features` (`h_slot` sized `num_features`);
        //   - `out[cell]` / `out[cell+1]` (`cell = slot_off[f] + bin*2`) — the resident
        //     bin-range invariant (`bin < num_bin`, upload-time) keeps the write inside
        //     that feature's slot within `slot_len` (`h_out` sized `slot_len`).
        // The host-side V5 checks discharge exactly the launch_unchecked obligations; the
        // launch does NOT change numerics — only bounds-check codegen is removed; the
        // f32-atomic scatter order is identical (~1e-6 path).
        // quick-260621-qix: native-width dispatch (see the LDS branch note above).
        macro_rules! launch_naive {
            ($w:ty) => {
                unsafe {
                    construct_leaf_hist_resident_kernel::launch_unchecked::<$w, R>(
                        client,
                        CubeCount::Static(cube_count, 1, 1),
                        CubeDim::new_1d(cube_dim),
                        ArrayArg::from_raw_parts(resident_bins, num_features * num_data),
                        ArrayArg::from_raw_parts(h_rows, rows),
                        ArrayArg::from_raw_parts(h_g, rows),
                        ArrayArg::from_raw_parts(h_h, rows),
                        ArrayArg::from_raw_parts(h_slot, num_features),
                        num_data,
                        total,
                        ArrayArg::from_raw_parts(h_out, slot_len),
                    );
                }
            };
        }
        match width {
            crate::ResidentBinWidth::U8 => launch_naive!(u8),
            crate::ResidentBinWidth::U16 => launch_naive!(u16),
            crate::ResidentBinWidth::U32 => launch_naive!(u32),
        }
    }
}

/// ON-GPU WIDEN + `FixHistogram` + compaction kernel (260608-oib L3, FOLDED by
/// 260608-s2b Lever A). ONE cube per feature
/// (`CubeCount::Static(num_features,1,1)`, `CubeDim::new_1d(1)`), mirroring
/// [`construct_leaf_hist_resident_kernel`] / the fused split kernel's
/// one-cube-per-feature precedent. Cube `f` (`CUBE_POS_X`) owns ONLY its
/// `[slot_off[f], slot_off[f] + 2*num_bin[f])` region.
///
/// 260608-s2b LEVER A — the standalone `widen_f32_to_f64_kernel` launch is FOLDED
/// IN as this kernel's FIRST pass: cube `f` widens its own region from the f32 RAW
/// histogram `h_raw` into the f64 output `hist` via `f64::cast_from(...)` — the
/// IDENTICAL cast the standalone widen performed — then runs the EXACT same in-place
/// `FixHistogram` + `compact` over `hist`. Because the widen cast and the f64 fix/
/// compact fold order are byte-for-byte unchanged, the f64 output is BIT-IDENTICAL
/// to the prior 3-launch "construct → widen → fix" chain; only the separate widen
/// LAUNCH is eliminated (3 launches → 2). The widen is now the fix kernel's own
/// first pass over its region.
///
/// `hist` (f64 OUT) must be zero-initialised by the caller (the tail/dropped cells
/// the compact step does not overwrite stay 0, matching the prior zeroed f64 alloc).
/// `h_raw` (f32 IN) holds the construct kernel's f32-atomic RAW cells.
///
/// A VERBATIM port of the host `fix_histogram` (`fix_histogram.rs:50-80`,
/// `Dataset::FixHistogram`) followed by `compact_histogram` (`learner.rs:2838-2864`),
/// preceded by the inline f32→f64 widen.
///
/// The leaf RAW (un-bumped) `sum_gradient` / `sum_hessian` are leaf-level scalars
/// shared across every feature (Pitfall 2: the RAW totals, NOT the `+2·kEpsilon`
/// bumped value). All math is f64 — gfx1100 runs f64 despite `has_f64 == false`
/// (the fused f64 split kernel precedent). Reading the SAME f64-widened cells and
/// folding in the SAME ascending order as the host yields a BIT-IDENTICAL buffer.
///
/// `#[cfg(feature="rocm")]` — the CPU anchor keeps the host fix+compact unchanged.
#[cfg(feature = "rocm")]
#[cube(launch_unchecked)]
pub fn fix_compact_kernel(
    // phase-11: u64 FIXED-POINT RAW histogram (u64 build-kernel output) — INPUT, read-only.
    // Each cell holds a two's-complement i64 (stored as u64 bits) = `round(value*2^30)`.
    h_raw: &Array<u64>,
    // f64 fixed+compacted histogram — OUTPUT (caller zeroes it before launch).
    hist: &mut Array<f64>,
    slot_off: &Array<u32>,
    num_bin: &Array<i32>,
    offset: &Array<i32>,
    most_freq_bin: &Array<i32>,
    // LEAF-LEVEL scalars (shared across the batch) — the RAW (un-bumped) f64 totals.
    // These are HOST-side exact f64 sums, NEVER quantized — used as-is below.
    sum_gradient: f64,
    sum_hessian: f64,
) {
    // phase-11 dequant scale S = 2^30 (matches the build-side SCALE_F32; f64 here).
    const SCALE_F64: f64 = 1_073_741_824.0; // 2^30
    let f = CUBE_POS_X;
    let fi = f as usize;
    let base = slot_off[fi] as usize;
    let nb = num_bin[fi];
    let mfb = most_freq_bin[fi];
    let off = offset[fi];

    // ---- FOLDED DEQUANT (phase-11; was the f32→f64 widen of 260608-s2b Lever A) ----
    // Dequantize THIS feature's whole region u64-bits → i64 → f64/2^30 into the output
    // BEFORE the fix/compact reads it (reproduces the host dequant from
    // gpu_fixedpoint_i64.rs:191 cell-by-cell). The result is an f64 buffer with the SAME
    // shape the prior f32 widen produced, so the FixHistogram fold + compact below are
    // BYTE-UNCHANGED — they operate on the already-f64 `hist`. Ascending bin order.
    for w in 0..nb {
        let wbi = base + (w as usize) * 2;
        hist[wbi] = f64::cast_from(i64::cast_from(h_raw[wbi])) / SCALE_F64;
        hist[wbi + 1] = f64::cast_from(i64::cast_from(h_raw[wbi + 1])) / SCALE_F64;
    }

    // ---- FixHistogram (fix_histogram.rs:50-80, Dataset::FixHistogram) ----
    // C++ `if (most_freq_bin > 0)`: skip when mfb == 0 (bin 0 is never directly
    // folded) OR mfb >= num_bin (defensive bound — leave untouched, no OOB write).
    // Encoded as a single guard; the body runs only for a valid in-range mfb > 0.
    let do_fix = mfb > 0 && mfb < nb;
    if do_fix {
        let mfbu = mfb as usize;
        // Seed the most-freq cell with the RAW leaf totals (dataset.cpp:1490-1491).
        // Literal-init loop-carried mutables (cubecl lowering discipline); the seed
        // is added in below so init-from-arg + reassign is avoided.
        let mut g = 0.0f64;
        let mut h = 0.0f64;
        g += sum_gradient;
        h += sum_hessian;
        // Subtract every OTHER bin's cell in ASCENDING bin order (load-bearing f64
        // fold order — never reorder / parallelize). `i != mfb` via branchless
        // select so the guarded cell is excluded without a nested-if mutation.
        let count = nb; // num_bin
        for i in 0..count {
            let bi = base + (i as usize) * 2;
            let gi = hist[bi];
            let hi = hist[bi + 1];
            let take = i != mfb;
            g -= select(take, gi, 0.0);
            h -= select(take, hi, 0.0);
        }
        let mi = base + mfbu * 2;
        hist[mi] = g;
        hist[mi + 1] = h;
    }

    // ---- compact (learner.rs:2838-2864) ----
    // offset <= 0 → no-op. offset >= num_bin → zero the whole feature region.
    // else: shift pair (c+offset) down to c for c in 0..(num_bin-offset) ASCENDING
    // (src >= dst, so an in-place forward shift is safe), then zero the tail.
    if off > 0 {
        if off >= nb {
            // Degenerate: nothing to keep — zero the whole feature region.
            for c in 0..nb {
                let dst = base + (c as usize) * 2;
                hist[dst] = 0.0;
                hist[dst + 1] = 0.0;
            }
        } else {
            let keep = nb - off; // num_bin - offset
            for c in 0..keep {
                let dst = base + (c as usize) * 2;
                let src = base + ((c + off) as usize) * 2;
                hist[dst] = hist[src];
                hist[dst + 1] = hist[src + 1];
            }
            // Zero the unused tail (the dropped-bin slots) so a stray read is inert.
            for c in keep..nb {
                let dst = base + (c as usize) * 2;
                hist[dst] = 0.0;
                hist[dst + 1] = 0.0;
            }
        }
    }
}

/// Host launcher for the on-GPU FOLDED widen+fix+compact kernel (260608-oib L3,
/// Task 1 form; signature updated by 260608-s2b Lever A).
///
/// Takes ONE leaf's concatenated stride-2 f32 RAW histogram `raw` (the construct
/// kernel's f32-atomic output — the SAME cells the host would widen to f64), the
/// per-feature `{slot_off, num_bin, offset, most_freq_bin}` arrays, and the leaf's
/// RAW (un-bumped) `sum_gradient` / `sum_hessian`; uploads, allocates a zeroed f64
/// output, launches one cube per feature (each widens its region inline then
/// fixes+compacts), reads back the fixed+compacted f64 buffer. This Task-1 form
/// keeps the readback so the kernel numerics are proven in isolation.
///
/// Lever A note: the kernel now folds the f32→f64 widen IN (`f64::cast_from`),
/// so this launcher feeds the f32 RAW directly — bit-identical to the prior
/// "supply pre-widened f64" form for any f32-representable RAW cell.
///
/// V5 boundary validation BEFORE launch (mirrors the fused split launcher):
/// `num_bin == 0` → typed error; `2*num_bin` overflow → typed error;
/// `slot_off + 2*num_bin > raw.len()` → [`ComputeError::LengthMismatch`]; empty
/// feats → `Ok(raw widened to f64)` with NO launch.
///
/// `feats` is `&[(slot_off, num_bin, offset, most_freq_bin)]` per feature, in the
/// same order as the concatenated regions in `raw`.
///
/// # Errors
/// As above (length / overflow validation, V5).
#[cfg(feature = "rocm")]
pub fn fix_compact_f64_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    raw: &[f32],
    feats: &[(usize, u32, i32, u32)],
    sum_gradient: f64,
    sum_hessian: f64,
) -> Result<Vec<f64>, ComputeError> {
    // Empty batch: no launch — return the f32 RAW widened to f64 (the degenerate
    // "widen-only" result, matching the prior f64 buffer pass-through).
    if feats.is_empty() {
        return Ok(raw.iter().map(|&x| f64::from(x)).collect());
    }

    let n = feats.len();
    let mut slot_off_a: Vec<u32> = Vec::with_capacity(n);
    let mut num_bin_a: Vec<i32> = Vec::with_capacity(n);
    let mut offset_a: Vec<i32> = Vec::with_capacity(n);
    let mut mfb_a: Vec<i32> = Vec::with_capacity(n);
    for &(slot_off, num_bin, offset, most_freq_bin) in feats {
        if num_bin == 0 {
            return Err(ComputeError::Runtime {
                detail: "fix_compact: num_bin must be > 0".to_string(),
            });
        }
        let cells = 2usize
            .checked_mul(num_bin as usize)
            .ok_or_else(|| ComputeError::Runtime {
                detail: format!("num_bin {num_bin} overflows the histogram length"),
            })?;
        let end = slot_off
            .checked_add(cells)
            .ok_or_else(|| ComputeError::Runtime {
                detail: "fix_compact: slot_off + region overflows".to_string(),
            })?;
        if end > raw.len() {
            return Err(ComputeError::LengthMismatch {
                expected: end,
                actual: raw.len(),
            });
        }
        slot_off_a.push(slot_off as u32);
        num_bin_a.push(num_bin as i32);
        offset_a.push(offset);
        mfb_a.push(most_freq_bin as i32);
    }

    // phase-11: `fix_compact_kernel` now consumes a u64 FIXED-POINT RAW buffer and
    // dequantizes `(bits as i64)/2^30 → f64` in its widen pass. This launcher receives
    // an f32 RAW histogram (the test/oracle path builds it host-side), so quantize each
    // cell `round(v*2^30) → i64 → bits-as-u64` here to match the live u64 build kernel —
    // the dequant in-kernel inverts it (round-trip exact for integer-valued cells, ≤
    // 1/2^30 abs error otherwise, well within the ~1e-6 ROCm gate). The fix/compact fold
    // below is byte-unchanged; only the RAW cell encoding changed.
    const SCALE_FC: f32 = 1_073_741_824.0; // 2^30 (matches SCALE_F32 / SCALE_F64)
    let raw_q: Vec<u64> =
        raw.iter().map(|&v| (v * SCALE_FC).round() as i64 as u64).collect();
    let h_raw = client.create_from_slice(u64::as_bytes(&raw_q));
    let zeros64 = vec![0.0f64; raw.len()];
    let h_hist = client.create_from_slice(f64::as_bytes(&zeros64));
    let h_slot = client.create_from_slice(u32::as_bytes(&slot_off_a));
    let h_numbin = client.create_from_slice(i32::as_bytes(&num_bin_a));
    let h_offset = client.create_from_slice(i32::as_bytes(&offset_a));
    let h_mfb = client.create_from_slice(i32::as_bytes(&mfb_a));

    // SAFETY: every handle is sized to its slice and outlives the launch. Cube `f`
    // (`CUBE_POS_X < n`) reads `h_raw` and reads/writes `h_hist` only within
    // `[slot_off[f], slot_off[f]+2*num_bin[f])` — each validated `<= raw.len()`
    // above — and `mfb < num_bin` keeps the reconstruct write in range; the
    // per-feature index arrays all have exactly `n` elements. All cubecl unsafe is
    // confined here (CMP-01).
    //
    // LAUNCH_UNCHECKED (NRW-01): we call `::launch_unchecked`, dropping the in-kernel
    // per-access bounds-check codegen in the per-feature fix/compact loops. This is a
    // ZERO-numeric-risk switch: the kernel is f64 and DETERMINISTIC (one cube per
    // feature, `CubeDim::new_1d(1)`, ascending fold) so the result stays bit-exact. Every
    // device access is host-proven in range BEFORE upload:
    //   - `h_raw[wbi]` / `h_raw[wbi+1]` and `h_hist[...]` for the inline widen + fix +
    //     compact — cube `f < n` touches only `[slot_off[f], slot_off[f] + 2*num_bin[f])`,
    //     and the loop above validated `slot_off[f] + 2*num_bin[f] <= raw.len()` for every
    //     feature; both `h_raw` and `h_hist` are sized `raw.len()`;
    //   - `mfb < num_bin` (the `do_fix` guard) keeps the reconstruct cell in the region;
    //   - the per-feature `slot_off`/`num_bin`/`offset`/`most_freq_bin` arrays all have
    //     exactly `n` elements and `f < n`.
    // i.e. the host-side V5 checks discharge exactly the obligations the launch_unchecked
    // contract requires, and the launch does NOT change numerics — only bounds-check
    // codegen is removed; the f64 fold order is identical (bit-exact).
    unsafe {
        fix_compact_kernel::launch_unchecked(
            client,
            CubeCount::Static(n as u32, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(h_raw, raw.len()),
            ArrayArg::from_raw_parts(h_hist.clone(), raw.len()),
            ArrayArg::from_raw_parts(h_slot, n),
            ArrayArg::from_raw_parts(h_numbin, n),
            ArrayArg::from_raw_parts(h_offset, n),
            ArrayArg::from_raw_parts(h_mfb, n),
            sum_gradient,
            sum_hessian,
        );
    }

    let bytes = client.read_one_unchecked(h_hist);
    Ok(f64::from_bytes(&bytes).to_vec())
}

/// Upload the binned feature columns to the device feature-major (feature `f`'s
/// row `r` at `f * num_data + r`) and return the resident `Handle` — the same
/// concatenated layout [`RocmBackend::upload_resident_bins`](crate::Backend::upload_resident_bins)
/// caches internally, exposed for the resident-chain oracle so it can feed a raw
/// Handle to [`build_fix_compact_resident_f64_on`] without naming `cubecl` types.
/// All columns must share `num_data` (caller guarantees). `#[cfg(feature="rocm")]`.
#[cfg(feature = "rocm")]
pub fn upload_resident_columns<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    feature_bins: &[&[u32]],
) -> cubecl::server::Handle {
    let num_features = feature_bins.len();
    let num_data = if num_features == 0 { 0 } else { feature_bins[0].len() };
    let mut concat: Vec<u32> = Vec::with_capacity(num_features * num_data);
    for &col in feature_bins {
        concat.extend_from_slice(col);
    }
    client.create_from_slice(u32::as_bytes(&concat))
}

// (260608-s2b Lever A) The standalone `widen_f32_to_f64_kernel` was REMOVED — its
// f32→f64 cast is now folded into `fix_compact_kernel`'s first pass (the inline
// `f64::cast_from` widen), eliminating a per-leaf GPU launch. See the FOLDED WIDEN
// block in `fix_compact_kernel`.

/// DEVICE-RESIDENT build→fix→compact chain (260608-oib L3, Task 2 step 1; FOLDED to
/// 2 launches by 260608-s2b Lever A).
///
/// Runs the resident build kernel ([`construct_leaf_hist_resident_kernel`]) into an
/// f32-atomic device buffer, then launches the on-GPU FOLDED widen+fix+compact
/// ([`fix_compact_kernel`]) which widens that f32 buffer to f64 INLINE (its first
/// pass, `f64::cast_from` — matching the host readback widening) and fixes+compacts
/// in one launch, and RETURNS the fixed+compacted f64 device `Handle` (NOT a Vec)
/// plus the buffer length. NO readback — the histogram VALUES never leave the device.
/// The standalone widen launch is GONE (3 launches → 2: construct + folded fix).
///
/// This is the resident analog of `build_leaf_histograms_resident_f32_on` +
/// host fix+compact: the whole per-leaf build→fix→compact chain runs on device. The
/// `fix_feats` carry the per-feature `(slot_off, num_bin, offset, most_freq_bin)`;
/// `sum_gradient`/`sum_hessian` are the leaf RAW (un-bumped) totals (Pitfall 2).
///
/// # Errors
/// [`ComputeError::Runtime`] on a degenerate layout; propagates the same V5
/// validation as [`fix_compact_f64_on`] / the resident build launcher.
#[cfg(feature = "rocm")]
#[allow(clippy::too_many_arguments)]
pub fn build_fix_compact_resident_f64_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    resident_bins: cubecl::server::Handle,
    // quick-260621-qix: native element width of `resident_bins`.
    width: crate::ResidentBinWidth,
    num_features: usize,
    num_data: usize,
    slot_off: &[usize],
    slot_len: usize,
    leaf_rows: &[u32],
    gradients: &[f32],
    hessians: &[f32],
    fix_feats: &[(usize, u32, i32, u32)],
    sum_gradient: f64,
    sum_hessian: f64,
) -> Result<(cubecl::server::Handle, usize), ComputeError> {
    // ---- 0. PHASE-11 OVERFLOW GUARD (SPEC item 4, spike-018 README:63,113-116) ----
    // The u64 fixed-point build sums `round(v * 2^30)` (i64) across a bin's rows. The
    // worst-case single-bin magnitude is `rows * max|v| * 2^30`; it MUST fit in i64 or
    // the two's-complement add wraps to a WRONG value. Bound: i64@2^30 is safe to
    // ~1e9 rows × |g| ≤ 8 (spike-018b). We bound `max|v|` by the actual leaf grad/hess
    // (a one-pass scan of the rows we are about to accumulate) — NOT a clamp; on
    // violation we return a typed error rather than silently overflow. The grad/hess are
    // small in practice (regression residuals / Newton steps), so this never trips on
    // sane data; it documents + enforces the contract for pathological extreme leaves.
    {
        let rows = leaf_rows.len() as f64;
        let mut max_abs = 0.0f64;
        for &r in leaf_rows {
            let i = r as usize;
            let g = gradients[i].abs() as f64;
            let h = hessians[i].abs() as f64;
            if g > max_abs {
                max_abs = g;
            }
            if h > max_abs {
                max_abs = h;
            }
        }
        // 2^30 scale; compare against i64::MAX in f64 (exact enough — both sides are
        // upper bounds and the margin to i64::MAX ≈ 9.2e18 is enormous for sane leaves).
        let worst = rows * max_abs * 1_073_741_824.0_f64; // rows * max|v| * 2^30
        if worst >= i64::MAX as f64 {
            return Err(ComputeError::Runtime {
                detail: "fixed-point histogram accumulation may overflow i64 at S=2^30 \
                         (rows x |value| x 2^30 exceeds i64::MAX)"
                    .to_string(),
            });
        }
    }

    // ---- 1. RESIDENT RAW build into a u64 fixed-point device buffer ----
    // LDS-privatized per-feature build when every feature ≤ 256 bins (naive fallback
    // otherwise) — the SAME `resident_raw_build_into` the readback launcher uses, so
    // the resident-pool chain and the host path share one accumulation structure.
    // phase-11: u64 fixed-point RAW merge target (was f32). The u64 LDS build accumulates
    // `round(v*2^30)` as two's-complement i64-bits; `fix_compact_kernel` dequantizes
    // `(bits as i64)/2^30 → f64` in its widen pass. The grad/hess INPUTS (`h_g`/`h_h` in
    // `resident_raw_build_into`) STAY f32 — the kernel quantizes them in-kernel; ONLY this
    // merge target widened to u64. Same `slot_len` element COUNT, 2× bytes.
    let zeros_u64 = vec![0u64; slot_len];
    let h_raw = client.create_from_slice(u64::as_bytes(&zeros_u64));
    resident_raw_build_into(
        client,
        resident_bins,
        width,
        num_features,
        num_data,
        slot_off,
        slot_len,
        leaf_rows,
        gradients,
        hessians,
        h_raw.clone(),
        true, // u64 fixed-point build → dequantized in fix_compact_kernel (phase-11)
    );

    // ---- 2. (260608-s2b Lever A) Allocate the zeroed f64 OUTPUT. The standalone
    //         widen launch is GONE — the folded fix kernel below widens each feature
    //         region from `h_raw` (f32) into `h_f64` (f64) inline as its first pass,
    //         then fixes+compacts. `fix_feats` covers EVERY feature region contiguously
    //         (the learner enumerates all features; regions tile [0, slot_len)), so the
    //         per-feature inline widen covers the whole buffer exactly as the prior
    //         full-buffer widen did. The per-leaf spine launch count drops 3 → 2
    //         (construct + folded fix). When `fix_feats` is empty (no features / no
    //         rows) there is nothing to widen and `h_f64` stays zeroed — matching the
    //         prior degenerate path (construct skipped ⇒ widen of zeros ⇒ zeros). ----
    let zeros64 = vec![0.0f64; slot_len];
    let h_f64 = client.create_from_slice(f64::as_bytes(&zeros64));

    // ---- 3. ON-GPU FOLDED widen+fix+compact over the f64 buffer (Lever A kernel) ----
    if !fix_feats.is_empty() {
        let n = fix_feats.len();
        let mut slot_off_a: Vec<u32> = Vec::with_capacity(n);
        let mut num_bin_a: Vec<i32> = Vec::with_capacity(n);
        let mut offset_a: Vec<i32> = Vec::with_capacity(n);
        let mut mfb_a: Vec<i32> = Vec::with_capacity(n);
        for &(so, nb, off, mfb) in fix_feats {
            if nb == 0 {
                return Err(ComputeError::Runtime {
                    detail: "build_fix_compact_resident: num_bin must be > 0".to_string(),
                });
            }
            let cells = 2usize.checked_mul(nb as usize).ok_or_else(|| ComputeError::Runtime {
                detail: format!("num_bin {nb} overflows the histogram length"),
            })?;
            let end = so.checked_add(cells).ok_or_else(|| ComputeError::Runtime {
                detail: "build_fix_compact_resident: slot_off + region overflows".to_string(),
            })?;
            if end > slot_len {
                return Err(ComputeError::LengthMismatch {
                    expected: end,
                    actual: slot_len,
                });
            }
            slot_off_a.push(so as u32);
            num_bin_a.push(nb as i32);
            offset_a.push(off);
            mfb_a.push(mfb as i32);
        }
        let h_slot = client.create_from_slice(u32::as_bytes(&slot_off_a));
        let h_numbin = client.create_from_slice(i32::as_bytes(&num_bin_a));
        let h_offset = client.create_from_slice(i32::as_bytes(&offset_a));
        let h_mfb = client.create_from_slice(i32::as_bytes(&mfb_a));
        // SAFETY: `h_raw` (f32 IN) and `h_f64` (f64 OUT) are both sized `slot_len`;
        // cube `f < n` reads `h_raw` and reads/writes `h_f64` only within its validated
        // `[slot_off[f], slot_off[f]+2*num_bin[f]) <= slot_len` region (inline widen +
        // fix + compact) and `mfb < num_bin` keeps the reconstruct in range. cubecl
        // unsafe confined here.
        //
        // LAUNCH_UNCHECKED (NRW-01): `::launch_unchecked` drops the in-kernel per-access
        // bounds-check codegen in the fix/compact loops. ZERO numeric risk — same f64
        // deterministic kernel as `fix_compact_f64_on` (one cube per feature, ascending
        // fold, bit-exact). Host-proven accesses BEFORE upload:
        //   - `h_raw[...]` / `h_f64[...]` — cube `f < n` touches only
        //     `[slot_off[f], slot_off[f] + 2*num_bin[f])`, validated `<= slot_len` in the
        //     loop above; both buffers sized `slot_len`;
        //   - `mfb < num_bin` (the `do_fix` guard) keeps the reconstruct cell in range;
        //   - the per-feature `slot_off`/`num_bin`/`offset`/`most_freq_bin` arrays all have
        //     exactly `n` elements and `f < n`.
        // The host-side V5 checks discharge exactly the launch_unchecked obligations; the
        // launch does NOT change numerics — only bounds-check codegen is removed; the f64
        // fold order is identical (bit-exact).
        unsafe {
            fix_compact_kernel::launch_unchecked(
                client,
                CubeCount::Static(n as u32, 1, 1),
                CubeDim::new_1d(1),
                ArrayArg::from_raw_parts(h_raw, slot_len),
                ArrayArg::from_raw_parts(h_f64.clone(), slot_len),
                ArrayArg::from_raw_parts(h_slot, n),
                ArrayArg::from_raw_parts(h_numbin, n),
                ArrayArg::from_raw_parts(h_offset, n),
                ArrayArg::from_raw_parts(h_mfb, n),
                sum_gradient,
                sum_hessian,
            );
        }
    }

    Ok((h_f64, slot_len))
}

/// Readback variant of [`build_fix_compact_resident_f64_on`] (260608-oib L3, Task 2
/// step 1 validation): runs the SAME device-resident build→fix→compact chain but
/// reads the f64 buffer back to a `Vec<f64>`. Used by the oracle to prove the
/// resident chain equals the host build (`build_leaf_histograms_resident_f32_on`)
/// + host `fix_histogram` + host `compact_histogram` (within the ~1e-6 f32-atomic
/// RAW-build tolerance; the fix+compact step itself is bit-exact, Task 1). Not on
/// the live path — the live wiring is deferred (see SUMMARY).
///
/// # Errors
/// Same as [`build_fix_compact_resident_f64_on`].
#[cfg(feature = "rocm")]
#[allow(clippy::too_many_arguments)]
pub fn build_fix_compact_resident_readback_f64_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    resident_bins: cubecl::server::Handle,
    // quick-260621-qix: native element width of `resident_bins`.
    width: crate::ResidentBinWidth,
    num_features: usize,
    num_data: usize,
    slot_off: &[usize],
    slot_len: usize,
    leaf_rows: &[u32],
    gradients: &[f32],
    hessians: &[f32],
    fix_feats: &[(usize, u32, i32, u32)],
    sum_gradient: f64,
    sum_hessian: f64,
) -> Result<Vec<f64>, ComputeError> {
    let (handle, len) = build_fix_compact_resident_f64_on(
        client,
        resident_bins,
        width,
        num_features,
        num_data,
        slot_off,
        slot_len,
        leaf_rows,
        gradients,
        hessians,
        fix_feats,
        sum_gradient,
        sum_hessian,
    )?;
    debug_assert_eq!(len, slot_len);
    let bytes = client.read_one_unchecked(handle);
    Ok(f64::from_bytes(&bytes).to_vec())
}

// ===========================================================================
// 260608-t3t: FUSED per-feature build + fix + compact + best-split scan kernel.
//
// ONE cube per feature (`CubeCount::Static(num_features,1,1)`, `CubeDim::new_1d(1)`)
// — single-owner ⇒ BIT-EXACT (the cpu-anchor f64 fold order), NO atomics, NO
// cross-cube barrier. Cube `f` (`CUBE_POS_X`) owns ONLY its region
// `[slot_off[f], slot_off[f] + 2*num_bin[f])`. The kernel collapses today's
// directly-built-leaf chain — construct_leaf_hist_resident_kernel(1) +
// fix_compact_kernel(1) + find_best_splits_fused_kernel(1) = 3 launches — into ONE.
//
// Stage 1 BUILD: SEQUENTIAL f64 gather→fold in ASCENDING leaf-row order (the CPU
//   anchor order ⇒ bit-exact, NOT the ~1e-6 f32-atomic path). Mirrors
//   `construct_leaf_hist_resident_kernel`'s bin layout / resident indexing
//   (histogram.rs:511-533) but sequential into THIS cube's f64 region.
// Stage 2 FIX: inlines `fix_compact_kernel`'s fix logic VERBATIM
//   (histogram.rs:674-703) — RAW (un-bumped) sum_gradient/sum_hessian seed
//   (Pitfall 2), ascending subtract via branchless `select`.
// Stage 3 COMPACT: inlines `fix_compact_kernel`'s compact logic VERBATIM
//   (histogram.rs:705-732) — offset shift + tail zero.
// Stage 4 SCAN: calls the SHARED `split_scan_body` (split.rs:144) over the
//   fixed+compacted region with the 2*kEpsilon-BUMPED sum_hessian + host
//   min_gain_shift (the SAME scan operands `find_best_splits_fused_kernel` uses),
//   writing the RAW 12-cell SplitInfo to `out[f*12..]`.
//
// Output BOTH the resident fixed+compacted f64 histogram (for the subtraction
// trick) AND the per-feature SplitInfos, in ONE launch.
//
// `#[cfg(feature="rocm")]` — the CPU anchor keeps the host build/fix/compact/scan
// unchanged (the fused gate is OFF on cpu).
// ===========================================================================

/// Fused per-feature BUILD + FIX + COMPACT + SCAN kernel (260608-t3t). See the
/// module-level block above. Cube `f` owns its region of `hist` (f64 OUT, the
/// resident fixed+compacted histogram, caller-zeroed) and writes its 12-cell
/// window `out[f*12 .. f*12+12]` (the RAW SplitInfo cells, host-decoded).
///
/// The LEAF-LEVEL scalars are shared across every feature: the RAW (un-bumped)
/// `sum_gradient_raw` / `sum_hessian_raw` feed the FIX (Pitfall 2); the
/// 2*kEpsilon-BUMPED `sum_hessian_bumped` + the host `min_gain_shift` feed the SCAN
/// (the distinct operands, matching `find_best_splits_fused_kernel`).
#[cfg(feature = "rocm")]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn build_fix_scan_fused_kernel<B: Int>(
    // Device-resident binned columns (feature-major, `f*num_data + row`) — INPUT.
    // quick-260621-qix: native bin width (u8/u16/u32).
    resident_bins: &Array<B>,
    // The leaf's row indices (subset of 0..num_data) — INPUT.
    leaf_rows: &Array<u32>,
    // The leaf's grad/hess gathered host-side in leaf_rows order — INPUT (f32).
    ord_g: &Array<f32>,
    ord_h: &Array<f32>,
    // f64 fixed+compacted histogram — OUTPUT (caller zeroes it before launch).
    hist: &mut Array<f64>,
    // RAW 12-cell-per-feature SplitInfo — OUTPUT.
    out: &mut Array<f64>,
    // Per-feature params (length == num_features).
    slot_off: &Array<u32>,
    num_bin: &Array<i32>,
    offset: &Array<i32>,
    most_freq_bin: &Array<i32>,
    default_bin: &Array<i32>,
    skip_default_bin: &Array<u32>,
    rev_count: &Array<i32>,
    fwd_count: &Array<i32>,
    // 0|1 per feature — whether THIS feature is scanned (a gated-out feature still has
    // its histogram BUILT+fixed+compacted, for the subtraction trick, but is NOT
    // scanned: its 12-cell out window is left zeroed ⇒ host decodes is_splittable == 0).
    scan_active: &Array<u32>,
    // Stride of a resident column = full train row count.
    num_data_stride: usize,
    // LEAF-LEVEL scalars (shared across the batch).
    sum_gradient_raw: f64,
    sum_hessian_raw: f64,
    use_l1: u32,
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    lambda_l1: f64,
    lambda_l2: f64,
    min_gain_shift: f64,
    sum_hessian_bumped: f64,
    num_data: i32,
) {
    let f = CUBE_POS_X;
    let fi = f as usize;
    let base = slot_off[fi] as usize;
    let nb = num_bin[fi];
    let mfb = most_freq_bin[fi];
    let off = offset[fi];

    // ---- Stage 1: SEQUENTIAL f64 BUILD (ascending leaf-row order = cpu anchor) ----
    // Zero this cube's region first (2 cells per bin), then gather each leaf row's
    // bin from the resident column and ASCENDING-fold f32 grad/hess into the f64
    // cells. `f64::cast_from(score_t f32)` reproduces the C++ float->double widen
    // (Pitfall 3); the ascending fold order matches the host sequential build EXACTLY
    // (the bit-exact contract — non-negotiable #2). NO atomics (single-owner cube).
    for w in 0..nb {
        let wbi = base + (w as usize) * 2;
        hist[wbi] = 0.0;
        hist[wbi + 1] = 0.0;
    }
    let rows = ord_g.len();
    for k in 0..rows {
        let row = leaf_rows[k] as usize;
        // quick-260621-qix: native-width read widened to a u32 INDEX (value-faithful).
        let bin = u32::cast_from(resident_bins[fi * num_data_stride + row]) as usize;
        let cell = base + bin * 2;
        hist[cell] += f64::cast_from(ord_g[k]);
        hist[cell + 1] += f64::cast_from(ord_h[k]);
    }

    // ---- Stage 2: FIX (fix_compact_kernel:674-703, VERBATIM) ----
    // Seed the most-freq cell with the RAW (un-bumped) leaf totals (Pitfall 2),
    // subtract every OTHER bin ASCENDING via branchless select. Runs only for a
    // valid in-range mfb > 0.
    let do_fix = mfb > 0 && mfb < nb;
    if do_fix {
        let mfbu = mfb as usize;
        let mut g = 0.0f64;
        let mut h = 0.0f64;
        g += sum_gradient_raw;
        h += sum_hessian_raw;
        let count = nb;
        for i in 0..count {
            let bi = base + (i as usize) * 2;
            let gi = hist[bi];
            let hi = hist[bi + 1];
            let take = i != mfb;
            g -= select(take, gi, 0.0);
            h -= select(take, hi, 0.0);
        }
        let mi = base + mfbu * 2;
        hist[mi] = g;
        hist[mi + 1] = h;
    }

    // ---- Stage 3: COMPACT (fix_compact_kernel:705-732, VERBATIM) ----
    if off > 0 {
        if off >= nb {
            for c in 0..nb {
                let dst = base + (c as usize) * 2;
                hist[dst] = 0.0;
                hist[dst + 1] = 0.0;
            }
        } else {
            let keep = nb - off;
            for c in 0..keep {
                let dst = base + (c as usize) * 2;
                let src = base + ((c + off) as usize) * 2;
                hist[dst] = hist[src];
                hist[dst + 1] = hist[src + 1];
            }
            for c in keep..nb {
                let dst = base + (c as usize) * 2;
                hist[dst] = 0.0;
                hist[dst + 1] = 0.0;
            }
        }
    }

    // ---- Stage 4: SCAN (shared split_scan_body, split.rs:144) ----
    // Build/fix/compact above ran for EVERY feature (the complete histogram the
    // subtraction trick needs). The SCAN runs only for SCAN-ACTIVE features (the spine
    // subset that passed the learner's col-sampler / parent-splittability / interaction
    // gates); a gated-out feature leaves its 12-cell out window ZEROED ⇒ the host
    // decodes is_splittable == 0 (a no-split sentinel) and never selects it. Over the
    // fixed+compacted region; hist_base = slot_off[f], out_base = f*12. The SAME leaf
    // scalars as find_best_splits_fused_kernel (split.rs:976-996): the 2*kEpsilon-BUMPED
    // sum_hessian + host min_gain_shift for the scan entry.
    if scan_active[fi] != 0 {
        crate::kernels::split::split_scan_body(
            hist,
            slot_off[fi],
            out,
            f * 12u32,
            nb,
            off,
            default_bin[fi],
            skip_default_bin[fi],
            use_l1,
            min_data_in_leaf,
            min_sum_hessian_in_leaf,
            lambda_l1,
            lambda_l2,
            min_gain_shift,
            sum_gradient_raw,
            sum_hessian_bumped,
            num_data,
            rev_count[fi],
            fwd_count[fi],
        );
    }
}

/// Host launcher for the FUSED build+fix+compact+scan kernel (260608-t3t).
///
/// Drives [`build_fix_scan_fused_kernel`] in ONE launch. `feats` is the FULL
/// per-feature list (every feature in fpos order) — build + fix + compact run for
/// EVERY feature so the returned resident histogram is COMPLETE (the subtraction
/// trick derives the larger child from it). `scan_active[fpos]` selects which
/// features are SCANNED (the spine subset that passed the learner's
/// col-sampler / parent-splittability / interaction gates); gated-out features are
/// BUILT but NOT scanned.
///
/// Returns BOTH the resident fixed+compacted f64 histogram `Handle` (kept on device)
/// AND one [`SplitInfo`] per SCAN-ACTIVE feature, in scan-active order (matching the
/// learner's `batched_feats`). Mirrors the V5 validation + marshalling of
/// [`build_fix_compact_resident_f64_on`] AND the host pre-step + decode/accept-gate
/// of the fused split scan (split.rs:1212-1311).
///
/// The leaf RAW (un-bumped) `sum_gradient_raw` / `sum_hessian_raw` feed the FIX
/// (Pitfall 2); the launcher computes the 2*kEpsilon-BUMPED sum_hessian +
/// min_gain_shift for the scan exactly as `find_best_splits_fused_inner` does.
///
/// # Errors
/// [`ComputeError::Runtime`] / [`ComputeError::LengthMismatch`] on degenerate
/// layout (mirrors the fused split launcher's per-feature V5 checks + the leaf-level
/// `sum_hessian > 0` / `max_delta_step`/`path_smooth` default-path checks).
#[cfg(feature = "rocm")]
#[allow(clippy::too_many_arguments)]
pub fn build_fix_scan_resident_f64_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    resident_bins: cubecl::server::Handle,
    // quick-260621-qix: native element width of `resident_bins`.
    width: crate::ResidentBinWidth,
    num_features: usize,
    num_data_stride: usize,
    // Per-feature slot offsets ride on `feats` (`f.slot_off`); this slice is accepted
    // for signature symmetry with `build_fix_compact_resident_f64_on` and asserted.
    slot_off: &[usize],
    slot_len: usize,
    leaf_rows: &[u32],
    gradients: &[f32],
    hessians: &[f32],
    // FULL per-feature list (fpos order) — build+fix+compact covers ALL of them.
    feats: &[crate::kernels::split::BatchedSplitFeature],
    // `scan_active[fpos]` — whether feature fpos is SCANNED (length == feats.len()).
    scan_active: &[bool],
    cfg: &crate::gain::GainConfig,
    sum_gradient_raw: f64,
    sum_hessian_raw: f64,
    num_data: i32,
) -> Result<(cubecl::server::Handle, usize, Vec<crate::gain::SplitInfo>), ComputeError> {
    use crate::gain::SplitInfo;

    // Empty batch / empty leaf: no launch — a zeroed resident hist + empty splits.
    let rows = leaf_rows.len();
    if feats.is_empty() || rows == 0 || num_features == 0 {
        let zeros64 = vec![0.0f64; slot_len];
        let h_f64 = client.create_from_slice(f64::as_bytes(&zeros64));
        return Ok((h_f64, slot_len, Vec::new()));
    }
    if scan_active.len() != feats.len() {
        return Err(ComputeError::LengthMismatch {
            expected: feats.len(),
            actual: scan_active.len(),
        });
    }

    // Leaf-level default-path + sum_hessian checks (identical to the fused split
    // launcher; the scan divides cnt_factor by the bumped sum_hessian).
    if cfg.max_delta_step != 0.0 || cfg.path_smooth != 0.0 {
        return Err(ComputeError::Runtime {
            detail: "build_fix_scan_resident: max_delta_step / path_smooth are Phase-7+ scope \
                     (only the default 0.0 path is transcribed)"
                .to_string(),
        });
    }
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(sum_hessian_raw > 0.0) {
        return Err(ComputeError::Runtime {
            detail: "build_fix_scan_resident: sum_hessian must be > 0 (cnt_factor divides by it)"
                .to_string(),
        });
    }

    // `feats` is the FULL per-feature list (fpos order), so it agrees positionally
    // with the pool's `slot_off` layout (consistency guard).
    debug_assert!(
        slot_off.len() >= feats.len()
            && feats.iter().enumerate().all(|(i, f)| slot_off[i] == f.slot_off),
        "feats[*].slot_off must agree positionally with the pool slot_off layout"
    );

    // Per-feature V5 validation + device-array assembly (BEFORE launch).
    let n = feats.len();
    let mut slot_off_a: Vec<u32> = Vec::with_capacity(n);
    let mut num_bin_a: Vec<i32> = Vec::with_capacity(n);
    let mut offset_a: Vec<i32> = Vec::with_capacity(n);
    let mut mfb_a: Vec<i32> = Vec::with_capacity(n);
    let mut default_bin_a: Vec<i32> = Vec::with_capacity(n);
    let mut skip_default_bin_a: Vec<u32> = Vec::with_capacity(n);
    let mut rev_count_a: Vec<i32> = Vec::with_capacity(n);
    let mut fwd_count_a: Vec<i32> = Vec::with_capacity(n);
    let mut scan_active_a: Vec<u32> = Vec::with_capacity(n);
    for (fpos, f) in feats.iter().enumerate() {
        if f.na_as_missing && scan_active[fpos] {
            return Err(ComputeError::Runtime {
                detail: "build_fix_scan_resident: na_as_missing not yet implemented".to_string(),
            });
        }
        if f.num_bin == 0 {
            return Err(ComputeError::Runtime {
                detail: "build_fix_scan_resident: num_bin must be > 0".to_string(),
            });
        }
        let cells = 2usize
            .checked_mul(f.num_bin as usize)
            .ok_or_else(|| ComputeError::Runtime {
                detail: format!("num_bin {} overflows the histogram length", f.num_bin),
            })?;
        let end = f
            .slot_off
            .checked_add(cells)
            .ok_or_else(|| ComputeError::Runtime {
                detail: "build_fix_scan_resident: slot_off + region overflows".to_string(),
            })?;
        if end > slot_len {
            return Err(ComputeError::LengthMismatch {
                expected: end,
                actual: slot_len,
            });
        }
        let num_bin_i = f.num_bin as i32;
        let rev_count = (num_bin_i - 1).max(0);
        let fwd_count = if f.run_forward {
            (num_bin_i - 1 - f.offset).max(0)
        } else {
            0
        };
        slot_off_a.push(f.slot_off as u32);
        num_bin_a.push(num_bin_i);
        offset_a.push(f.offset);
        mfb_a.push(f.most_freq_bin as i32);
        default_bin_a.push(f.default_bin as i32);
        skip_default_bin_a.push(if f.skip_default_bin { 1u32 } else { 0u32 });
        rev_count_a.push(rev_count);
        fwd_count_a.push(fwd_count);
        scan_active_a.push(if scan_active[fpos] { 1u32 } else { 0u32 });
    }

    // LEAF-LEVEL scalars computed ONCE (the 2*kEpsilon entry bump + min_gain_shift),
    // exactly as find_best_splits_fused_inner (split.rs:1213-1223).
    let two_eps = 2.0 * f64::from(lgbm_core::types::K_EPSILON);
    let sum_hessian_bumped = sum_hessian_raw + two_eps;
    let use_l1 = cfg.use_l1();
    let gain_shift = crate::gain::get_leaf_gain(
        use_l1,
        sum_gradient_raw,
        sum_hessian_bumped,
        cfg.lambda_l1,
        cfg.lambda_l2,
    );
    let min_gain_shift = gain_shift + cfg.min_gain_to_split;

    // Per-leaf uploads (leaf_rows + the leaf's gathered grad/hess) + the zeroed
    // outputs. The big resident bins matrix is ALREADY on device.
    let ord_g: Vec<f32> = leaf_rows.iter().map(|&r| gradients[r as usize]).collect();
    let ord_h: Vec<f32> = leaf_rows.iter().map(|&r| hessians[r as usize]).collect();
    let h_rows = client.create_from_slice(u32::as_bytes(leaf_rows));
    let h_g = client.create_from_slice(f32::as_bytes(&ord_g));
    let h_h = client.create_from_slice(f32::as_bytes(&ord_h));
    let zeros64 = vec![0.0f64; slot_len];
    let h_hist = client.create_from_slice(f64::as_bytes(&zeros64));
    let out_len = n * 12;
    let out_zeros = vec![0.0f64; out_len];
    let h_out = client.create_from_slice(f64::as_bytes(&out_zeros));
    let h_slot = client.create_from_slice(u32::as_bytes(&slot_off_a));
    let h_numbin = client.create_from_slice(i32::as_bytes(&num_bin_a));
    let h_offset = client.create_from_slice(i32::as_bytes(&offset_a));
    let h_mfb = client.create_from_slice(i32::as_bytes(&mfb_a));
    let h_defbin = client.create_from_slice(i32::as_bytes(&default_bin_a));
    let h_skip = client.create_from_slice(u32::as_bytes(&skip_default_bin_a));
    let h_rev = client.create_from_slice(i32::as_bytes(&rev_count_a));
    let h_fwd = client.create_from_slice(i32::as_bytes(&fwd_count_a));
    let h_scan = client.create_from_slice(u32::as_bytes(&scan_active_a));

    // SAFETY: cube `f < n` reads the resident column at `f*num_data_stride +
    // leaf_rows[k]` (leaf_rows ⊂ 0..num_data_stride keeps it in range; the resident
    // buffer is sized num_features*num_data_stride), and reads/writes `h_hist` only
    // within its validated `[slot_off[f], slot_off[f]+2*num_bin[f]) <= slot_len`
    // region, writing `h_out[f*12 .. f*12+12]` within the n*12 allocation. `bin <
    // num_bin` (resident invariant) keeps the build cell in range; `mfb < num_bin`
    // keeps the reconstruct in range. Every per-feature index array has exactly `n`
    // elements; every handle outlives the launch. All cubecl unsafe confined here
    // (CMP-01).
    //
    // LAUNCH_UNCHECKED (NRW-01): `::launch_unchecked` drops the in-kernel per-access
    // bounds-check codegen in the build + fix + scan loops. ZERO numeric risk — the kernel
    // is f64 and DETERMINISTIC (one cube per feature, `CubeDim::new_1d(1)`, SEQUENTIAL
    // ascending leaf-row fold, NO atomics) so it stays the bit-exact cpu-anchor order.
    // Every device access is host-proven in range BEFORE upload:
    //   - `resident_bins[f*num_data_stride + leaf_rows[k]]` — `leaf_rows` ⊂
    //     `[0, num_data_stride)` (caller resident contract) and the resident buffer is
    //     sized `num_features*num_data_stride`, so `f < n <= num_features` keeps it in range;
    //   - `leaf_rows[k]` for `k < rows` (`h_rows` sized `rows`); `ord_g[k]`/`ord_h[k]`
    //     (`h_g`/`h_h` sized `rows`);
    //   - `hist[...]` (the f64 build/fix/compact) — cube `f < n` touches only
    //     `[slot_off[f], slot_off[f] + 2*num_bin[f])`, validated `<= slot_len` in the
    //     per-feature loop above (`h_hist` sized `slot_len`); `bin < num_bin` keeps the
    //     build cell and `mfb < num_bin` the reconstruct cell in that region;
    //   - `out[f*12 .. f*12+12]` within the `n*12` allocation (`h_out` sized `n*12`);
    //   - every per-feature param array (`slot_off`/`num_bin`/`offset`/`most_freq_bin`/
    //     `default_bin`/`skip_default_bin`/`rev_count`/`fwd_count`/`scan_active`) has
    //     exactly `n` elements and `f < n`.
    // The host-side V5 checks discharge exactly the launch_unchecked obligations; the
    // launch does NOT change numerics — only bounds-check codegen is removed; the f64
    // sequential fold / scan order is identical (bit-exact).
    //
    // PERF/BENEFIT (measured — quick 260619-ol8, dual-kernel single-binary interleaved A/B
    // on gfx1100, `examples/launch_unchecked_ab.rs`): this is the ONE swept kernel where
    // launch_unchecked pays off measurably — ~9–16% faster launch-bound, ~40–46% (≈1.8×)
    // faster compute-bound, sign-stable with non-overlapping p25/p75 spread. Because the
    // checked/unchecked arms are bit-identical f64, the delta is PURE bounds-check codegen.
    // It surfaces HERE (and not in the f32-atomic / resident-LDS kernels, where ol8 measured
    // NULL) precisely because this kernel runs long SINGLE-UNIT SEQUENTIAL loops
    // (`CubeDim::new_1d(1)`, one cube per feature) — a per-access bounds branch compounds
    // over every leaf-row × bin iteration with nothing to hide behind, whereas the atomic
    // kernels are atomic-contention / memory-latency bound and mask it. So launch_unchecked
    // is strongly justified for this kernel on perf grounds, not merely safe.
    // quick-260621-qix: dispatch the fused kernel's `<B: Int>` monomorphization on the
    // resident buffer's native width (only the `resident_bins` ArrayArg type changes;
    // every other arg is width-independent). Exactly one match arm runs ⇒ the by-value
    // handle moves are exclusive.
    macro_rules! launch_fused {
        ($w:ty) => {
            unsafe {
                build_fix_scan_fused_kernel::launch_unchecked::<$w, R>(
                    client,
                    CubeCount::Static(n as u32, 1, 1),
                    CubeDim::new_1d(1),
                    ArrayArg::from_raw_parts(resident_bins, num_features * num_data_stride),
                    ArrayArg::from_raw_parts(h_rows, rows),
                    ArrayArg::from_raw_parts(h_g, rows),
                    ArrayArg::from_raw_parts(h_h, rows),
                    ArrayArg::from_raw_parts(h_hist.clone(), slot_len),
                    ArrayArg::from_raw_parts(h_out.clone(), out_len),
                    ArrayArg::from_raw_parts(h_slot, n),
                    ArrayArg::from_raw_parts(h_numbin, n),
                    ArrayArg::from_raw_parts(h_offset, n),
                    ArrayArg::from_raw_parts(h_mfb, n),
                    ArrayArg::from_raw_parts(h_defbin, n),
                    ArrayArg::from_raw_parts(h_skip, n),
                    ArrayArg::from_raw_parts(h_rev, n),
                    ArrayArg::from_raw_parts(h_fwd, n),
                    ArrayArg::from_raw_parts(h_scan, n),
                    num_data_stride,
                    sum_gradient_raw,
                    sum_hessian_raw,
                    if use_l1 { 1u32 } else { 0u32 },
                    cfg.min_data_in_leaf,
                    cfg.min_sum_hessian_in_leaf,
                    cfg.lambda_l1,
                    cfg.lambda_l2,
                    min_gain_shift,
                    sum_hessian_bumped,
                    num_data,
                );
            }
        };
    }
    match width {
        crate::ResidentBinWidth::U8 => launch_fused!(u8),
        crate::ResidentBinWidth::U16 => launch_fused!(u16),
        crate::ResidentBinWidth::U32 => launch_fused!(u32),
    }

    // Read back ONLY the SplitInfo cells; the histogram Handle stays resident.
    let bytes = client.read_one_unchecked(h_out);
    let cells = f64::from_bytes(&bytes);

    // Decode with the SAME accept-gate as find_best_splits_fused_inner
    // (split.rs:1277-1311). Only SCAN-ACTIVE features have a meaningful out window
    // (gated-out features were built but not scanned ⇒ their window is zeroed ⇒
    // is_splittable == 0). Push the active features' SplitInfos in scan-active order
    // (matching the learner's `batched_feats`).
    let penalty = 1.0f64;
    let active_count = scan_active.iter().filter(|&&a| a).count();
    let mut splits = Vec::with_capacity(active_count);
    for f in 0..n {
        if !scan_active[f] {
            continue;
        }
        let dbase = f * 12;
        let is_splittable = cells[dbase] != 0.0;
        let raw_threshold = cells[dbase + 1] as u32;
        let raw_gain = cells[dbase + 2];
        let left_count = cells[dbase + 3] as i32;
        let right_count = cells[dbase + 4] as i32;
        let left_sum_gradient = cells[dbase + 5];
        let left_sum_hessian = cells[dbase + 6];
        let right_sum_gradient = cells[dbase + 7];
        let right_sum_hessian = cells[dbase + 8];
        let default_left = cells[dbase + 9] != 0.0;
        let left_output = cells[dbase + 10];
        let right_output = cells[dbase + 11];

        if is_splittable && raw_gain > f64::NEG_INFINITY {
            splits.push(SplitInfo {
                threshold: raw_threshold,
                gain: (raw_gain - min_gain_shift) * penalty,
                left_count,
                right_count,
                left_sum_gradient,
                left_sum_hessian,
                right_sum_gradient,
                right_sum_hessian,
                left_output,
                right_output,
                default_left,
            });
        } else {
            splits.push(SplitInfo::none());
        }
    }

    Ok((h_hist, slot_len, splits))
}

#[cfg(test)]
mod tests {
    use super::{accumulate_histogram_into, construct_histograms_cpu_native};
    use crate::error::ComputeError;

    /// Assert two f64 histograms are bit-identical, cell-by-cell (`to_bits()`).
    fn assert_bits_eq(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len(), "histogram lengths differ");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "cell {i} differs: {x} ({:#018x}) vs {y} ({:#018x})",
                x.to_bits(),
                y.to_bits(),
            );
        }
    }

    /// The fold-in-place accumulator produces bytes identical to the allocating
    /// path on multiple shapes — the bit-exact merge-gate invariant in unit form.
    #[test]
    fn accumulate_into_is_bit_identical_to_native() {
        // Shape A: small, a few bins, repeated bins, mixed-sign gradients.
        let binned_a: Vec<u32> = vec![0, 2, 1, 2, 0, 3, 1, 2, 3, 0];
        let grad_a: Vec<f32> = vec![0.5, -1.25, 3.0, 0.125, -2.5, 7.75, -0.0625, 1.5, -4.0, 0.25];
        let hess_a: Vec<f32> = vec![1.0, 0.5, 2.0, 0.25, 1.5, 0.75, 3.0, 0.125, 2.5, 1.0];
        let num_bin_a = 4u32;

        let want_a =
            construct_histograms_cpu_native(&binned_a, &grad_a, &hess_a, num_bin_a).unwrap();
        let mut got_a = vec![0.0f64; 2 * num_bin_a as usize];
        accumulate_histogram_into(&binned_a, &grad_a, &hess_a, num_bin_a, &mut got_a).unwrap();
        assert_bits_eq(&want_a, &got_a);

        // Shape B: larger, more bins, values chosen so fold ORDER matters for f64
        // accumulation (catastrophic-cancellation-sensitive magnitudes).
        let n = 257usize;
        let num_bin_b = 16u32;
        let mut binned_b = Vec::with_capacity(n);
        let mut grad_b = Vec::with_capacity(n);
        let mut hess_b = Vec::with_capacity(n);
        for i in 0..n {
            binned_b.push((i as u32) % num_bin_b);
            // Alternating large/small magnitudes to expose any reordering.
            let g = if i % 2 == 0 {
                1e7_f32 + i as f32
            } else {
                -(1e-3_f32) * i as f32
            };
            grad_b.push(g);
            hess_b.push((i as f32).mul_add(0.001, 0.5));
        }

        let want_b =
            construct_histograms_cpu_native(&binned_b, &grad_b, &hess_b, num_bin_b).unwrap();
        let mut got_b = vec![0.0f64; 2 * num_bin_b as usize];
        accumulate_histogram_into(&binned_b, &grad_b, &hess_b, num_bin_b, &mut got_b).unwrap();
        assert_bits_eq(&want_b, &got_b);
    }

    /// Folding into a pre-zeroed sub-slice of a LARGER buffer writes only its own
    /// region and matches the standalone native build bit-for-bit (the multi-feature
    /// `out` layout `build_leaf_histograms_raw` uses).
    #[test]
    fn accumulate_into_subslice_matches_native() {
        let binned: Vec<u32> = vec![0, 1, 1, 0, 2, 2, 1];
        let grad: Vec<f32> = vec![1.0, -2.0, 0.5, 4.0, -1.0, 0.25, 3.0];
        let hess: Vec<f32> = vec![0.5, 1.0, 2.0, 0.25, 1.5, 0.75, 0.125];
        let num_bin = 3u32;
        let cells = 2 * num_bin as usize;

        let want = construct_histograms_cpu_native(&binned, &grad, &hess, num_bin).unwrap();

        // A larger pre-zeroed buffer with the histogram placed at an offset.
        let off = cells; // place feature in the second slot
        let mut buf = vec![0.0f64; 3 * cells];
        accumulate_histogram_into(&binned, &grad, &hess, num_bin, &mut buf[off..off + cells])
            .unwrap();

        assert_bits_eq(&want, &buf[off..off + cells]);
        // Untouched regions stay exactly zero.
        assert!(buf[..off].iter().all(|&v| v.to_bits() == 0.0f64.to_bits()));
        assert!(buf[off + cells..].iter().all(|&v| v.to_bits() == 0.0f64.to_bits()));
    }

    /// An undersized `out` sub-slice is a typed `LengthMismatch`, NOT a panic (V5).
    #[test]
    fn accumulate_into_rejects_short_out() {
        let binned: Vec<u32> = vec![0, 1, 2];
        let grad: Vec<f32> = vec![1.0, 2.0, 3.0];
        let hess: Vec<f32> = vec![1.0, 1.0, 1.0];
        let num_bin = 4u32; // needs 8 cells
        let mut out = vec![0.0f64; 4]; // too short

        let err =
            accumulate_histogram_into(&binned, &grad, &hess, num_bin, &mut out).unwrap_err();
        assert!(
            matches!(err, ComputeError::LengthMismatch { expected, actual }
                if expected == 8 && actual == 4),
            "expected LengthMismatch{{8,4}}, got {err:?}"
        );
    }

    /// `resolve_target_cubes` pure resolution order (a)→(b)→(c): env override used
    /// VERBATIM (>0) → queried CU × `CUBES_PER_CU` → `FALLBACK`. No env/OnceLock/GPU —
    /// exercises the resolution logic directly with explicit args (PREFERRED route (i)).
    #[cfg(feature = "rocm")]
    #[test]
    fn resolve_target_cubes_order() {
        use super::{resolve_target_cubes, CUBES_PER_CU, ROWPART_TARGET_CUBES_FALLBACK};
        // (a) env override (>0) wins verbatim, NOT multiplied by CUBES_PER_CU.
        assert_eq!(resolve_target_cubes(Some(768), Some(8)), 768);
        assert_eq!(resolve_target_cubes(Some(64), None), 64);
        // env override of 0 is ignored → falls through.
        assert_eq!(resolve_target_cubes(Some(0), Some(8)), 8 * CUBES_PER_CU);
        // (b) queried CU count → num_cu * CUBES_PER_CU (8 CUs here → 64).
        assert_eq!(resolve_target_cubes(None, Some(8)), 8 * CUBES_PER_CU);
        assert_eq!(resolve_target_cubes(None, Some(8)), 64);
        assert_eq!(resolve_target_cubes(None, Some(96)), 768); // a real gfx1100 would still get 768
        // (c) neither → documented FALLBACK (never a silent 768).
        assert_eq!(resolve_target_cubes(None, None), ROWPART_TARGET_CUBES_FALLBACK);
        assert_eq!(resolve_target_cubes(None, Some(0)), ROWPART_TARGET_CUBES_FALLBACK);
        assert_eq!(ROWPART_TARGET_CUBES_FALLBACK, 64);
    }

    /// The row-partition heuristic (spike-007): 1 on small/few-feature shapes (so the
    /// build stays byte-identical to the pre-row-part kernel), a tuned P in [2, P_MAX]
    /// on large-leaf × few-feature shapes, never exceeding P_MAX (the P=32 regression
    /// guard). Pure CPU logic — no GPU. Expressed against the runtime/cached `target`
    /// (`rowpart_target_cubes()`) so it is robust regardless of host hardware. Assumes
    /// `LGBM_ROWPART_MIN` is unset (default).
    #[cfg(feature = "rocm")]
    #[test]
    fn row_partition_count_heuristic() {
        use super::{row_partition_count, rowpart_target_cubes, ROWPART_MIN_LEAF, ROWPART_P_MAX};
        // Bind the runtime/cached target once; all asserts are expressed in terms of it
        // so the test is deterministic regardless of the queried CU count.
        let target = rowpart_target_cubes();
        assert!(target >= 2, "target_cubes={target} too small to exercise the tuned path");
        // Small leaf → P=1 (covers the ≤8k-row parity-test shapes + the 12k resident gate).
        assert_eq!(row_partition_count(50, 8_000), 1);
        assert_eq!(row_partition_count(50, ROWPART_MIN_LEAF - 1), 1);
        // Degenerate / already-saturated → 1.
        assert_eq!(row_partition_count(0, 10_000_000), 1);
        assert_eq!(row_partition_count(target as usize, ROWPART_MIN_LEAF), 1);
        assert_eq!(row_partition_count(target as usize + 50, ROWPART_MIN_LEAF), 1);
        // Large leaf + few features → the clamped tuned value matches the formula.
        let nf = 50u32;
        let expected = if nf >= target { 1 } else { (target / nf).clamp(1, ROWPART_P_MAX) };
        assert_eq!(row_partition_count(nf as usize, ROWPART_MIN_LEAF), expected);
        // Very few features → clamp to P_MAX (target/1, target/2 both ≥ P_MAX once
        // target ≥ 2*P_MAX, which holds for both the 8-CU APU (64) and a gfx1100 (768)).
        if target >= 2 * ROWPART_P_MAX {
            assert_eq!(row_partition_count(1, ROWPART_MIN_LEAF), ROWPART_P_MAX);
            assert_eq!(row_partition_count(2, ROWPART_MIN_LEAF), ROWPART_P_MAX);
        }
    }

    /// Confirm the queried Compute Unit count on THIS device. The box is a Radeon 860M
    /// APU (8 CUs, gfx1152 spoofed as gfx1100), so `query_num_cu()` should report 8 and
    /// `rowpart_target_cubes()` ≈ 64 (8 × CUBES_PER_CU), NOT the phantom-96-CU 768.
    /// Soft (eprintln + >0 check) so it never blocks the gate if the FFI query is
    /// environment-flaky; the hard CU=8 expectation is recorded in the SUMMARY.
    #[cfg(feature = "rocm")]
    #[test]
    fn queried_cu_count_is_8() {
        use super::{query_num_cu, rowpart_target_cubes};
        let cu = query_num_cu();
        let target = rowpart_target_cubes();
        eprintln!("query_num_cu() = {cu:?}; rowpart_target_cubes() = {target}");
        match cu {
            Some(n) => {
                assert!(n > 0, "queried CU count must be positive, got {n}");
                if n != 8 {
                    eprintln!("NOTE: expected 8 CUs on this Radeon 860M APU, queried {n}");
                }
            }
            None => eprintln!("NOTE: query_num_cu() returned None (FFI unavailable); using fallback"),
        }
        // With no LGBM_ROWPART_TARGET_CUBES override, target must NOT be the old 768
        // phantom-96-CU value unless the device genuinely has 96 CUs.
        assert!(target > 0, "target_cubes must be positive");
        if std::env::var("LGBM_ROWPART_TARGET_CUBES").is_err() {
            assert_ne!(
                target, 768,
                "target_cubes is still 768 with no override — phantom-96-CU value not removed"
            );
        }
    }
}
