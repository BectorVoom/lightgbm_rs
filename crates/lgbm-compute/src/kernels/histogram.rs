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
#[cfg(feature = "gpu")]
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
#[cfg(feature = "gpu")]
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
#[cfg(feature = "gpu")]
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
#[cfg(feature = "gpu")]
const ROWPART_MIN_LEAF: usize = 256_000;
/// Target cubes per Compute Unit — preserves spike-007's "~8 workgroups/CU" intent.
/// `target_cubes = num_cu * CUBES_PER_CU` (queried at runtime, cached once).
#[cfg(feature = "gpu")]
const CUBES_PER_CU: u32 = 8;
/// Documented safe small default for an APU-class device when EVERY CU-count query
/// fails — explicitly NOT 768 (which was the phantom-96-CU value: `8 wkgrps × 96 CU`).
/// 64 = `8 wkgrps × 8 CU`, matching the real 8-CU Radeon 860M APU on this box.
#[cfg(feature = "gpu")]
const ROWPART_TARGET_CUBES_FALLBACK: u32 = 64;
/// Spike-007 sweet spot; clamp so we never over-partition into the P=32 regression.
#[cfg(feature = "gpu")]
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
#[cfg(feature = "gpu")]
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
///
/// HIP-specific (the FFI fallback uses `cubecl_hip_sys`, a rocm-only dep), so this stays
/// `#[cfg(feature = "rocm")]`. Non-rocm GPU backends (cuda/wgpu) use the
/// [`ROWPART_TARGET_CUBES_FALLBACK`] heuristic via the `None`-returning twin below
/// (quick-260627-qxl) — the resident pool is the parity win; CU-derived row-partition
/// tuning is a perf refinement a CUDA build can add later by reading the cubecl-reported
/// `num_streaming_multiprocessors` (populated on cuda).
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

/// Non-rocm GPU twin (quick-260627-qxl): cuda/wgpu have no `cubecl_hip_sys`, so the
/// CU-count query returns `None` and [`rowpart_target_cubes`] falls back to
/// [`ROWPART_TARGET_CUBES_FALLBACK`]. Correct (the resident pool is the parity win);
/// a CUDA build can later read `num_streaming_multiprocessors` for a tuned target.
#[cfg(all(feature = "gpu", not(feature = "rocm")))]
fn query_num_cu() -> Option<u32> {
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
#[cfg(feature = "gpu")]
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
#[cfg(feature = "gpu")]
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
#[cfg(feature = "gpu")]
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
#[cfg(feature = "gpu")]
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
#[cfg(feature = "gpu")]
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
#[cfg(feature = "gpu")]
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
#[cfg(feature = "gpu")]
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
#[cfg(feature = "gpu")]
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

// ===========================================================================
// NET-NEW (Plan 16-03 / ODL-09): the §13 feature-partition TWO-TIER build kernel.
//
// Lifts the shipped Phase-11 u64 fixed-point LDS accumulation body
// (`construct_leaf_hist_resident_lds_kernel_u64`, above — left BYTE-UNCHANGED, D-03)
// onto the design-doc §7.1 / §13 partition geometry (`CalcConstructHistogramKernelDim`):
//   * `CUBE_POS_X` = feature PARTITION (was one cube per FEATURE);
//   * `UNIT_POS_X` = one COLUMN within the partition's `[off[p], off[p+1])` range;
//   * `UNIT_POS_Y × CUBE_POS_Y` = the leaf-row STRIPE (disjoint y-blocks).
//
// Two-tier atomics (§7.2): fast LDS block-local `Atomic<u64>` accumulation during the
// sweep, then a cross-block GLOBAL atomic merge into the leaf histogram. cubecl 0.10 has
// NO grid barrier, so the cross-block reduction MUST be a global atomic (many y-blocks
// cover disjoint row stripes of the same partition). Dense-vs-sparse is ONE `#[cube]`
// generic with a `#[comptime] is_sparse` branch (the §7.2 Dense/Sparse axis). The cpu
// f64 anchor stays the single-owner `construct_histograms_f64_on` fold (D-06); this
// kernel is the hip path only (`Atomic<u64>` is unsupported / nondeterministic on
// cubecl-cpu, Pitfall 7).
// ===========================================================================

/// Two-tier §13-geometry u64 fixed-point BUILD (NET-NEW, ODL-09 / D-03/D-06/D-08).
///
/// One `#[cube]` generic over the bin width `B: Int` with a `#[comptime] is_sparse`
/// branch (the §7.2 Dense/Sparse axis). Accumulates `round(value · 2^30)` as a
/// two's-complement i64 stored BITS-as-u64 via LDS `Atomic<u64>::fetch_add` (the wrapping
/// add IS a signed i64 add — no bias offset, spike-018b), then merges each cube's LDS
/// sub-histogram into the global leaf histogram with one cross-block global atomic per
/// cell. NO f64 in the per-row scatter (D-08) — the dequant to `hist_t` is a SEPARATE
/// pass ([`dequant_leaf_hist`], Plan 16-03 Task 3 / 16-04 Fix).
///
/// HARD CONSTRAINT (spike-018b, def-f8u-01): the cell type MUST be `Atomic<u64>` with
/// `.store(0u64)` / `.fetch_add(qbits)` — NEVER `Atomic<i64>` (cubecl-hip 0.10 link-fails).
///
/// Geometry (§7.1):
/// * `CUBE_POS_X` = partition `p`; columns `[off[p], off[p+1])`, `ncol_p` wide.
/// * `UNIT_POS_X` = local column `tx` (one column per x-thread; `tx < ncol_p` guard).
/// * `CUBE_POS_Y · CUBE_DIM_Y + UNIT_POS_Y` = the leaf-row stripe start; stride
///   `CUBE_COUNT_Y · CUBE_DIM_Y` (disjoint y-blocks → the merge is atomic, not a sync).
///
/// Bin fetch (comptime):
/// * dense — partition row-major store `data[lo·num_data + idx·ncol_p + tx]` holds the
///   RAW per-column bin; the LDS cell is `(column_hist_offsets[col] + raw) · 2` (the
///   partition-local offset is applied at accumulation time, §7.2).
/// * sparse — per-partition CSR: `row_start = row_ptr[p·(num_data+1) + idx]`, and the
///   stored value `data[row_start + tx]` is ALREADY partition-local (the §13 re-lay
///   subtracted `partition_hist_start`, Pitfall 4), so the LDS cell is `stored · 2`;
///   the per-row `tx < row_end − row_start` guard handles the nnz remainder.
///
/// LDS holds this partition's `span·2` cells (`span = partition bin count`), capped at
/// `HIST_LDS_MAX` (256 bins) — a partition whose `span·2` exceeds the cap routes to the
/// `_GlobalMemory` sibling instead (parity-neutral capacity choice, §17). `grad`/`hess`
/// are FULL-CORPUS arrays read at the gathered row `idx = leaf_rows[k]` (§7.2
/// `cuda_gradients[idx]`), matching the `cpu_anchor_columns` f64 fold.
#[cfg(feature = "gpu")]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn construct_leaf_hist_partition_u64<B: Int>(
    data: &Array<B>,                     // partition row-major bin store (dense raw / sparse local)
    row_ptr: &Array<u32>,                // sparse CSR row pointers [num_partitions·(num_data+1)]
    leaf_rows: &Array<u32>,              // data_indices_in_leaf
    grad: &Array<f32>,                   // FULL-CORPUS gradients [num_data]
    hess: &Array<f32>,                   // FULL-CORPUS hessians  [num_data]
    partition_col_offsets: &Array<u32>,  // off[]: partition p owns cols [off[p], off[p+1])
    column_hist_offsets: &Array<u32>,    // per-column partition-local bin offset [num_columns]
    partition_hist_offsets: &Array<u32>, // global bin offset per partition [num_partitions+1]
    num_data: usize,
    out: &mut Array<Atomic<u64>>,        // global leaf histogram, stride-2 [2·num_total_bin]
    #[comptime] is_sparse: bool,
) {
    let p = CUBE_POS_X as usize; // ONE cube-x per feature partition
    let lo = partition_col_offsets[p] as usize;
    let hi = partition_col_offsets[p + 1] as usize;
    let ncol_p = hi - lo;
    let phs = partition_hist_offsets[p] as usize; // partition global bin start
    let phe = partition_hist_offsets[p + 1] as usize;
    let lds_len = (phe - phs) * 2; // grad+hess cells for this partition

    let cd = CUBE_DIM as usize; // flat threads/cube (= CUBE_DIM_X · CUBE_DIM_Y)
    let sub = SharedMemory::<Atomic<u64>>::new(HIST_LDS_MAX);
    // 1. zero this partition's active LDS cells cooperatively (u64 zero = additive identity).
    let mut c = UNIT_POS as usize;
    while c < lds_len {
        sub[c].store(0u64);
        c += cd;
    }
    sync_cube();

    // 2. scatter — one column per x-thread; rows striped over (CUBE_POS_Y, UNIT_POS_Y).
    let tx = UNIT_POS_X as usize;
    if tx < ncol_p {
        let col = lo + tx;
        let col_off = column_hist_offsets[col] as usize; // partition-local bin offset (dense)
        let rp_base = p * (num_data + 1); // this partition's CSR block (sparse)
        let dense_part_base = lo * num_data; // this partition's row-major base (dense)
        let r = leaf_rows.len();
        let stride = CUBE_COUNT_Y as usize * CUBE_DIM_Y as usize;
        let mut k = CUBE_POS_Y as usize * CUBE_DIM_Y as usize + UNIT_POS_Y as usize;
        while k < r {
            let idx = leaf_rows[k] as usize;
            // quantize `round(v·2^30)` → i64 → bits-as-u64 (u64-only hot loop, D-08).
            let qg = u64::cast_from(i64::cast_from(f32::round(grad[idx] * SCALE_F32)));
            let qh = u64::cast_from(i64::cast_from(f32::round(hess[idx] * SCALE_F32)));
            if is_sparse {
                // CSR: stored bin is ALREADY partition-local (re-lay subtracted phs).
                let row_start = row_ptr[rp_base + idx] as usize;
                let row_end = row_ptr[rp_base + idx + 1] as usize;
                if tx < row_end - row_start {
                    let cell = u32::cast_from(data[row_start + tx]) as usize * 2;
                    sub[cell].fetch_add(qg);
                    sub[cell + 1].fetch_add(qh);
                }
            } else {
                // dense row-major partition store holds the RAW bin; add the local offset.
                let raw = u32::cast_from(data[dense_part_base + idx * ncol_p + tx]) as usize;
                let cell = (col_off + raw) * 2;
                sub[cell].fetch_add(qg);
                sub[cell + 1].fetch_add(qh);
            }
            k += stride;
        }
    }
    sync_cube();

    // 3. merge LDS → this partition's global slot (cross-block atomic; phs·2 base).
    let base = phs * 2;
    let mut m = UNIT_POS as usize;
    while m < lds_len {
        out[base + m].fetch_add(sub[m].load());
        m += cd;
    }
}

/// `_GlobalMemory` spill twin of [`construct_leaf_hist_partition_u64`] (Plan 16-03
/// Task 2 / D-04/D-09 — the §7.2 Dense/Sparse × `_GlobalMemory` axis).
///
/// IDENTICAL §13 geometry, gather, quantize, and cross-block merge as the shared twin —
/// the ONLY difference is the per-y-block partial histogram lives in a PRE-ALLOCATED
/// global `Array<Atomic<u64>>` (`spill`) instead of `SharedMemory`, so a partition whose
/// bin span exceeds the LDS cap (`HIST_LDS_MAX`) still builds for real (the C++
/// `CUDAConstructHistogram*Kernel_GlobalMemory` path, `NumLargeBinPartition() > 0`).
///
/// Each y-block owns the slice `spill[(CUBE_POS_Y · num_total_bin + global_bin) · 2]`
/// (§7.2: `cuda_hist_buffer_` at `(blockIdx.y · num_total_bin + phs) · 2`), so disjoint
/// y-blocks never collide WITHIN the spill; the final per-cell merge into `out` is the
/// SAME cross-block global atomic as the shared path (different y-blocks → same `out`
/// cell → atomic). Shared-vs-global is parity-neutral (§17 — a capacity choice with no
/// float-parity impact as long as the in-strategy reduction order is fixed; here both
/// fold the partition's bins in ascending order). u64 fixed-point ONLY (D-08).
///
/// The caller (`construct_leaf_hist_on_device`) pre-allocates `spill` ONCE sized
/// `grid_dim_y · num_total_bin · 2` (`checked_mul`-guarded, D-09) and ZEROES the active
/// slices via this kernel's Phase-1 cooperative store — never reallocated per tree.
#[cfg(feature = "gpu")]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn construct_leaf_hist_partition_global_u64<B: Int>(
    data: &Array<B>,
    row_ptr: &Array<u32>,
    leaf_rows: &Array<u32>,
    grad: &Array<f32>,
    hess: &Array<f32>,
    partition_col_offsets: &Array<u32>,
    column_hist_offsets: &Array<u32>,
    partition_hist_offsets: &Array<u32>,
    num_data: usize,
    num_total_bin: usize,                 // total bins across all partitions (spill stride)
    spill: &mut Array<Atomic<u64>>,       // per-y-block partials [grid_dim_y·num_total_bin·2]
    out: &mut Array<Atomic<u64>>,         // global leaf histogram, stride-2 [2·num_total_bin]
    #[comptime] is_sparse: bool,
) {
    let p = CUBE_POS_X as usize;
    let lo = partition_col_offsets[p] as usize;
    let hi = partition_col_offsets[p + 1] as usize;
    let ncol_p = hi - lo;
    let phs = partition_hist_offsets[p] as usize;
    let phe = partition_hist_offsets[p + 1] as usize;
    let span = phe - phs;

    // This y-block's spill base (§7.2 `(blockIdx.y · num_total_bin + phs) · 2`).
    let yblock_base = (CUBE_POS_Y as usize * num_total_bin + phs) * 2;
    let cd = CUBE_DIM as usize;
    // 1. zero THIS y-block's active spill cells cooperatively.
    let mut c = UNIT_POS as usize;
    while c < span * 2 {
        spill[yblock_base + c].store(0u64);
        c += cd;
    }
    sync_cube();

    // 2. scatter — same gather/quantize as the shared twin, into the global spill slice.
    let tx = UNIT_POS_X as usize;
    if tx < ncol_p {
        let col = lo + tx;
        let col_off = column_hist_offsets[col] as usize;
        let rp_base = p * (num_data + 1);
        let dense_part_base = lo * num_data;
        let r = leaf_rows.len();
        let stride = CUBE_COUNT_Y as usize * CUBE_DIM_Y as usize;
        let mut k = CUBE_POS_Y as usize * CUBE_DIM_Y as usize + UNIT_POS_Y as usize;
        while k < r {
            let idx = leaf_rows[k] as usize;
            let qg = u64::cast_from(i64::cast_from(f32::round(grad[idx] * SCALE_F32)));
            let qh = u64::cast_from(i64::cast_from(f32::round(hess[idx] * SCALE_F32)));
            if is_sparse {
                let row_start = row_ptr[rp_base + idx] as usize;
                let row_end = row_ptr[rp_base + idx + 1] as usize;
                if tx < row_end - row_start {
                    let local = u32::cast_from(data[row_start + tx]) as usize;
                    let cell = yblock_base + local * 2;
                    spill[cell].fetch_add(qg);
                    spill[cell + 1].fetch_add(qh);
                }
            } else {
                let raw = u32::cast_from(data[dense_part_base + idx * ncol_p + tx]) as usize;
                let cell = yblock_base + (col_off + raw) * 2;
                spill[cell].fetch_add(qg);
                spill[cell + 1].fetch_add(qh);
            }
            k += stride;
        }
    }
    sync_cube();

    // 3. merge THIS y-block's spill slice → the global leaf slot (cross-block atomic).
    let out_base = phs * 2;
    let mut m = UNIT_POS as usize;
    while m < span * 2 {
        out[out_base + m].fetch_add(spill[yblock_base + m].load());
        m += cd;
    }
}

/// De-quant the raw u64 fixed-point histogram to `hist_t` (f64) exactly ONCE
/// (Plan 16-03 Task 3 / D-01/D-08): `(bits as i64) / 2^30` — the SAME 2^30 scale as the
/// build side ([`SCALE_F32`]), mirroring `fix_compact_kernel`'s folded dequant first pass
/// (`(bits as i64)/SCALE_F64`). Kept a SEPARATE pass (RESEARCH Pattern 3 / Open Q1) so
/// BUILD stays a clean u64-only accumulator and the cpu-anchor split is unentangled; the
/// 16-04 Fix step then operates on the durable `hist_t`. Round-trip exact for
/// integer-valued cells, ≤ 1/2^30 abs error otherwise — well inside the ~1e-6 ROCm gate.
#[cfg(feature = "gpu")]
#[must_use]
pub fn dequant_leaf_hist(raw: &[u64]) -> Vec<f64> {
    const SCALE_F64: f64 = 1_073_741_824.0; // 2^30 (matches SCALE_F32 / fix_compact SCALE_F64)
    raw.iter().map(|&bits| (bits as i64) as f64 / SCALE_F64).collect()
}

/// f32 mirror of [`dequant_leaf_hist`] for the no-f64 hip device (CMP-04): the durable
/// `hist_t` on ROCm/CUDA is f32. Same 2^30 scale; the ~1e-6 ROCm gate absorbs the f32
/// rounding (never the cpu f64 anchor).
#[cfg(feature = "gpu")]
#[must_use]
pub fn dequant_leaf_hist_f32(raw: &[u64]) -> Vec<f32> {
    const SCALE_F32D: f32 = 1_073_741_824.0; // 2^30
    raw.iter().map(|&bits| (bits as i64) as f32 / SCALE_F32D).collect()
}

/// Spill-buffer cell count for the `_GlobalMemory` path (`grid_dim_y · num_total_bin · 2`),
/// `checked_mul`-guarded (D-09 / T-16-03-03). A separate `#[must_use]` helper so the
/// overflow guard is unit-testable without a device.
///
/// # Errors
/// [`ComputeError::Runtime`] if the product overflows `usize` (or is zero).
#[cfg(feature = "gpu")]
pub fn spill_cells(grid_dim_y: usize, num_total_bin: usize) -> Result<usize, ComputeError> {
    let cells = grid_dim_y
        .checked_mul(num_total_bin)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| ComputeError::Runtime {
            detail: format!(
                "spill buffer size grid_dim_y {grid_dim_y} · num_total_bin {num_total_bin} · 2 \
                 overflows usize"
            ),
        })?;
    if cells == 0 {
        return Err(ComputeError::Runtime {
            detail: "spill buffer size is zero (empty layout)".to_string(),
        });
    }
    Ok(cells)
}

/// Host launcher for the §13 two-tier BUILD (Plan 16-03 Task 3 / ODL-09). Derives the
/// §7.1 geometry from [`FeaturePartitionLayout`], runs the V5 bounds checks BEFORE any
/// `launch_unchecked` (T-16-03-01), zeroes the `out` accumulator from an explicit zero
/// slice (Caller-must-zero, never a pooled-uninitialized buffer), selects dense/sparse
/// (`row_ptr.is_some()`) and shared/`_GlobalMemory` (large-bin or `force_global`), and
/// returns the RAW u64 fixed-point histogram `[2·num_total_bin]`. The caller de-quants
/// once via [`dequant_leaf_hist`] (BUILD stays u64-only, D-08 / Pattern 3).
///
/// `data` is the partition row-major store (dense: RAW per-column bin; sparse: the CSR
/// values, partition-local). `grad`/`hess` are FULL-CORPUS, read at the gathered row.
/// `force_global` forces the `_GlobalMemory` path on a fitting layout — the parity-neutral
/// toggle (§17) the shared-vs-global equivalence test drives.
///
/// # Errors
/// [`ComputeError`] for: empty layout (→ `Ok(vec![])`, NO launch), `num_total_bin == 0`,
/// `2·num_total_bin` overflow, a `leaf_rows`/`grad`/`hess`/`data`/`row_ptr` length or
/// range violation, or a spill-size overflow.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub fn construct_leaf_hist_on_device<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    layout: &crate::kernels::row_data::FeaturePartitionLayout,
    data: &[u32],
    row_ptr: Option<&[u32]>,
    leaf_rows: &[u32],
    grad: &[f32],
    hess: &[f32],
    num_data: usize,
    force_global: bool,
) -> Result<Vec<u64>, ComputeError> {
    let num_partitions = layout.num_feature_partitions;
    // Empty layout: no columns / no partitions → no launch (the degenerate Ok path).
    if num_partitions == 0 || layout.feature_partition_column_index_offsets.len() < 2 {
        return Ok(Vec::new());
    }
    let num_columns = *layout
        .feature_partition_column_index_offsets
        .last()
        .expect("offsets has >= 2 entries");
    let num_total_bin = *layout
        .partition_hist_offsets
        .last()
        .expect("partition_hist_offsets has >= 2 entries");

    // --- V5 boundary validation (T-16-03-01) BEFORE any launch ---
    if num_total_bin == 0 {
        return Err(ComputeError::Runtime {
            detail: "construct_leaf_hist_on_device: num_total_bin must be > 0".to_string(),
        });
    }
    let out_len = num_total_bin
        .checked_mul(2)
        .ok_or_else(|| ComputeError::Runtime {
            detail: format!("2 · num_total_bin {num_total_bin} overflows usize"),
        })?;
    if grad.len() < num_data || hess.len() < num_data {
        return Err(ComputeError::LengthMismatch {
            expected: num_data,
            actual: grad.len().min(hess.len()),
        });
    }
    for (row, &di) in leaf_rows.iter().enumerate() {
        if di as usize >= num_data {
            return Err(ComputeError::BinIndexOutOfRange {
                row,
                bin: di,
                num_bin: num_data as u32,
            });
        }
    }
    let is_sparse = row_ptr.is_some();
    if is_sparse {
        let rp = row_ptr.expect("is_sparse implies row_ptr present");
        let expect_rp = num_partitions
            .checked_mul(num_data + 1)
            .ok_or_else(|| ComputeError::Runtime {
                detail: "num_partitions · (num_data+1) overflows usize".to_string(),
            })?;
        if rp.len() != expect_rp {
            return Err(ComputeError::LengthMismatch { expected: expect_rp, actual: rp.len() });
        }
    } else {
        // dense store must cover every partition's row-major region (num_columns · num_data).
        let expect = num_columns
            .checked_mul(num_data)
            .ok_or_else(|| ComputeError::Runtime {
                detail: "num_columns · num_data overflows usize".to_string(),
            })?;
        if data.len() < expect {
            return Err(ComputeError::LengthMismatch { expected: expect, actual: data.len() });
        }
    }

    // --- §7.1 geometry ---
    let bx = layout.max_num_column_per_partition.max(1) as u32; // block_dim_x = cols/partition
    let by = (256u32 / bx).max(1); // row workers (NUM_THREADS_PER_BLOCK analog)
    let rows = leaf_rows.len();
    // grid_dim_y: a modest row over-decomposition; correctness is gy-independent (atomic
    // merge), so keep it small here (the §7.1 floor of 160 is an occupancy knob, no parity).
    let gy = (rows.div_ceil((by as usize).max(1)))
        .clamp(1, 16)
        .max(1) as u32;

    // Per-partition max bin span decides the shared-vs-global route (LDS cap = HIST_LDS_MAX).
    let mut max_span = 0usize;
    for p in 0..num_partitions {
        let span =
            layout.partition_hist_offsets[p + 1] - layout.partition_hist_offsets[p];
        if span > max_span {
            max_span = span;
        }
    }
    let use_global =
        force_global || layout.num_large_bin_partition > 0 || max_span * 2 > HIST_LDS_MAX;

    // --- upload (caller-zeroed `out`; explicit zero slice, never pooled-uninitialized) ---
    let part_col_off: Vec<u32> = layout
        .feature_partition_column_index_offsets
        .iter()
        .map(|&v| v as u32)
        .collect();
    let col_hist_off: Vec<u32> =
        layout.column_hist_offsets.iter().map(|&v| v as u32).collect();
    let part_hist_off: Vec<u32> =
        layout.partition_hist_offsets.iter().map(|&v| v as u32).collect();
    let dummy_rp = [0u32]; // dense: kernel never reads row_ptr (is_sparse=false)
    let rp_slice = row_ptr.unwrap_or(&dummy_rp);

    let h_data = client.create_from_slice(u32::as_bytes(data));
    let h_rp = client.create_from_slice(u32::as_bytes(rp_slice));
    let h_rows = client.create_from_slice(u32::as_bytes(leaf_rows));
    let h_grad = client.create_from_slice(f32::as_bytes(grad));
    let h_hess = client.create_from_slice(f32::as_bytes(hess));
    let h_pco = client.create_from_slice(u32::as_bytes(&part_col_off));
    let h_cho = client.create_from_slice(u32::as_bytes(&col_hist_off));
    let h_pho = client.create_from_slice(u32::as_bytes(&part_hist_off));
    let zeros = vec![0u64; out_len];
    let h_out = client.create_from_slice(u64::as_bytes(&zeros));

    let cube_count = CubeCount::Static(num_partitions as u32, gy, 1);
    let cube_dim = CubeDim::new_2d(bx, by);

    if use_global {
        // Pre-allocate the spill buffer ONCE (D-09): a single `client.empty`, never in a
        // per-tree loop. The kernel zeroes each active y-block slice in its Phase 1, so an
        // uninitialized pool buffer is sound (every read is preceded by a store).
        let spill_len = spill_cells(gy as usize, num_total_bin)?;
        let h_spill = client.empty(spill_len * core::mem::size_of::<u64>());
        // SAFETY: every handle is sized to its slice/alloc and outlives the launch; the V5
        // checks above prove `leaf_rows ⊂ [0,num_data)`, `data`/`row_ptr` cover every
        // partition region, and `out`/`spill` are sized `out_len`/`spill_len`. All cubecl
        // unsafe is confined here (CMP-01).
        unsafe {
            construct_leaf_hist_partition_global_u64::launch_unchecked::<u32, R>(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(h_data, data.len()),
                ArrayArg::from_raw_parts(h_rp, rp_slice.len()),
                ArrayArg::from_raw_parts(h_rows, rows),
                ArrayArg::from_raw_parts(h_grad, grad.len()),
                ArrayArg::from_raw_parts(h_hess, hess.len()),
                ArrayArg::from_raw_parts(h_pco, part_col_off.len()),
                ArrayArg::from_raw_parts(h_cho, col_hist_off.len()),
                ArrayArg::from_raw_parts(h_pho, part_hist_off.len()),
                num_data,
                num_total_bin,
                ArrayArg::from_raw_parts(h_spill, spill_len),
                ArrayArg::from_raw_parts(h_out.clone(), out_len),
                is_sparse,
            );
        }
    } else {
        // SAFETY: as above; the shared path uses LDS (no spill), `max_span·2 <= HIST_LDS_MAX`
        // guaranteed by the `use_global` route.
        unsafe {
            construct_leaf_hist_partition_u64::launch_unchecked::<u32, R>(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(h_data, data.len()),
                ArrayArg::from_raw_parts(h_rp, rp_slice.len()),
                ArrayArg::from_raw_parts(h_rows, rows),
                ArrayArg::from_raw_parts(h_grad, grad.len()),
                ArrayArg::from_raw_parts(h_hess, hess.len()),
                ArrayArg::from_raw_parts(h_pco, part_col_off.len()),
                ArrayArg::from_raw_parts(h_cho, col_hist_off.len()),
                ArrayArg::from_raw_parts(h_pho, part_hist_off.len()),
                num_data,
                ArrayArg::from_raw_parts(h_out.clone(), out_len),
                is_sparse,
            );
        }
    }

    let bytes = client.read_one_unchecked(h_out);
    Ok(u64::from_bytes(&bytes).to_vec())
}

/// LDS batched RAW build: one cube per feature, reads host-gathered bins
/// (`gathered_bins[f*R + k]`). `slot_off` has `num_features + 1` entries (sentinel).
#[cfg(feature = "gpu")]
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
#[cfg(feature = "gpu")]
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
#[cfg(feature = "gpu")]
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
#[cfg(feature = "gpu")]
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
#[cfg(feature = "gpu")]
fn slot_off_sentinel(slot_off: &[usize], slot_len: usize) -> (Vec<u32>, u32) {
    let mut s: Vec<u32> = Vec::with_capacity(slot_off.len() + 1);
    for &o in slot_off {
        s.push(o as u32);
    }
    s.push(slot_len as u32);
    let max_w = s.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0);
    (s, max_w)
}

// ===========================================================================
// phase-13 (13-02) — CubeCL AUTOTUNE for the histogram-BUILD row-partition `P`.
//
// CONTEXT (LOCKED): autotune is the DEFAULT rocm selector for the resident-build
// `P`; `row_partition_count` becomes only the documented cold-start / cache-miss /
// `LGBM_AUTOTUNE=0` fallback bound (NOT the steady-state selector). spike-040 found
// the heuristic under-partitions to P=1 at the 50-feature production width (~10%
// slow on the 8-CU APU); autotune re-derives the measured-fastest P per occupancy
// regime and self-calibrates on any future GPU.
//
// Three corrections this wiring honors (gpu-kernel-autotuning.md / spikes 037-039):
//   - FRESH-OUTPUT InputGenerator (spike-038): the build kernel ACCUMULATES via
//     `fetch_add`, so `CloneInputGenerator` (which shares the real `out` handle)
//     corrupts it 27× during the cold benchmark reps. `FreshOutGenerator` hands each
//     benchmark a throwaway zeroed `out` ⇒ the real `out` is touched exactly ONCE
//     (the final clean winning run) ⇒ grad-conservation holds.
//   - log2(rows) occupancy-regime AutotuneKey (spike-039): keying on EXACT `rows`
//     is a per-leaf tuning STORM (every node a cold ~40ms tune). `LaunchKey.bucket =
//     size_band(rows)` so every leaf in the same power-of-two decade shares a key.
//   - serde-backed persistent cache (spike-037): the winner round-trips across
//     processes via `target/autotune/<ver>/<device>/*.json.log`.
//
// The TunableSet is rebuilt FRESH each call (`Arc::new(build_pset_tunable_set(..))`,
// NOT `LocalTuner::init`) because the per-call dimensions (`rows`/`num_features`/
// `slot_len`) bake into the launch closures; `LocalTuner::init` memoizes by closure
// TYPE-id and would freeze the FIRST call's dimensions forever. The persistent
// winner still survives across calls — it lives in `BUILD_TUNER`'s key→fastest_index
// state keyed by `LaunchKey`, and the BUILD_PSET registration order is fixed so
// `fastest_index` maps to the same `P` regardless of which call rebuilt the set.
// ===========================================================================

#[cfg(feature = "gpu")]
use crate::kernels::autotune::{self, LaunchKey};
#[cfg(feature = "gpu")]
use cubecl::tune::{local_tuner, InputGenerator, LocalTuner, Tunable, TunableSet, TuneInputs};

/// The row-partition candidate set the BUILD tuner sweeps. Each entry `> ROWPART_P_MAX`
/// is skipped (so this stays correct if the clamp ever rises). spike-040: the P4..P16
/// curve is FLAT — the only job is to AVOID P1 at the production width — so the set is
/// deliberately coarse (do not over-fine it into a slow cold tune). `1` stays in so the
/// tuner can still pick it for small leaves where partitioning only adds merge overhead.
///
/// `pub` so the 13-04 parity gate (`oracle-harness/tests/kernel_parity.rs`) imports the
/// SAME source of truth it sweeps (WR-02): a hand-copied mirror would silently stop
/// covering a newly-added `P`. Stays `#[cfg(feature = "rocm")]` so the default build is
/// byte-unchanged.
#[cfg(feature = "gpu")]
pub const BUILD_PSET: &[u32] = &[1, 4, 8, 16, 32];

/// The BUILD cache namespace — `local_tuner!("build")` ⇒ `LocalTuner<LaunchKey, String>`.
/// Holds the persistent key→fastest_index map (and mirrors it to disk via `std_io`).
#[cfg(feature = "gpu")]
static BUILD_TUNER: LocalTuner<LaunchKey, String> = local_tuner!("build");

/// The `LGBM_AUTOTUNE_FORCE_P` debug/parity seam (consumed by 13-04's all-variants
/// anchor gate). Reads the env FRESH on every call (mirrors `scan_cube_dim`, NOT
/// `OnceLock`) so a parity test can pin a single `P` per-launch within one process.
/// `Some(k)` clamps `k` to `[1, ROWPART_P_MAX]`; non-numeric / `0` / unset ⇒ `None`
/// (fall through to autotune, then the heuristic) — NEVER a no-launch (T-13-02-01).
#[cfg(feature = "gpu")]
fn force_row_partition() -> Option<u32> {
    std::env::var("LGBM_AUTOTUNE_FORCE_P")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&k| k > 0)
        .map(|k| k.clamp(1, ROWPART_P_MAX))
}

/// Launch the EXISTING resident LDS build once at a fixed `P`, reading the ordered
/// handle slice `[resident_bins, leaf_rows, ord_g, ord_h, slot_off, out]` (out at
/// index 5). This is the single launcher the BUILD tuner's PSET variants call (one per
/// `P`); it mirrors the `launch_lds_u64!` / `launch_lds_f32!` macros in
/// [`resident_raw_build_into`] EXACTLY — same kernel, same `CubeCount(num_features, P, 1)`,
/// same `CubeDim::new_1d(256)`, same arg order/sizes — only the handles arrive via a slice
/// instead of named locals. Keep the two in sync (they intentionally launch byte-identical
/// kernels; the macros serve the FORCE_P / `LGBM_AUTOTUNE=0` direct path, this serves the
/// tuner closures which only have a `Vec<Handle>`).
///
/// `fixed_point` selects the u64 fixed-point twin (`out` is `Atomic<u64>`) vs the f32
/// kernel; `width` dispatches the `<B: Int>` monomorphization on the resident buffer's
/// native element width (quick-260621-qix). SAFETY: identical to the in-place macro
/// launches — every device access is host-proven in range before upload (`leaf_rows` ⊂
/// `[0, num_data)`, `resident_bins.len() == num_features*num_data`, `slot_off` sentinel,
/// `out` sized `slot_len`), so `launch_unchecked` (dropping bounds-check codegen) is sound
/// and numerically identical (NRW-01 / CMP-01). All cubecl unsafe is confined here.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn launch_build_at<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    width: crate::ResidentBinWidth,
    fixed_point: bool,
    num_features: usize,
    num_data: usize,
    rows: usize,
    slot_len: usize,
    inputs: &[cubecl::server::Handle],
    p: u32,
) {
    macro_rules! at_p_u64 {
        ($w:ty) => {
            unsafe {
                construct_leaf_hist_resident_lds_kernel_u64::launch_unchecked::<$w, R>(
                    client,
                    CubeCount::Static(num_features as u32, p, 1),
                    CubeDim::new_1d(256),
                    ArrayArg::from_raw_parts(inputs[0].clone(), num_features * num_data),
                    ArrayArg::from_raw_parts(inputs[1].clone(), rows),
                    ArrayArg::from_raw_parts(inputs[2].clone(), rows),
                    ArrayArg::from_raw_parts(inputs[3].clone(), rows),
                    ArrayArg::from_raw_parts(inputs[4].clone(), num_features + 1),
                    num_data,
                    ArrayArg::from_raw_parts(inputs[5].clone(), slot_len),
                );
            }
        };
    }
    macro_rules! at_p_f32 {
        ($w:ty) => {
            unsafe {
                construct_leaf_hist_resident_lds_kernel::launch_unchecked::<$w, R>(
                    client,
                    CubeCount::Static(num_features as u32, p, 1),
                    CubeDim::new_1d(256),
                    ArrayArg::from_raw_parts(inputs[0].clone(), num_features * num_data),
                    ArrayArg::from_raw_parts(inputs[1].clone(), rows),
                    ArrayArg::from_raw_parts(inputs[2].clone(), rows),
                    ArrayArg::from_raw_parts(inputs[3].clone(), rows),
                    ArrayArg::from_raw_parts(inputs[4].clone(), num_features + 1),
                    num_data,
                    ArrayArg::from_raw_parts(inputs[5].clone(), slot_len),
                );
            }
        };
    }
    if fixed_point {
        match width {
            crate::ResidentBinWidth::U8 => at_p_u64!(u8),
            crate::ResidentBinWidth::U16 => at_p_u64!(u16),
            crate::ResidentBinWidth::U32 => at_p_u64!(u32),
        }
    } else {
        match width {
            crate::ResidentBinWidth::U8 => at_p_f32!(u8),
            crate::ResidentBinWidth::U16 => at_p_f32!(u16),
            crate::ResidentBinWidth::U32 => at_p_f32!(u32),
        }
    }
}

/// THE spike-038 FIX — an [`InputGenerator`] that hands each autotune BENCHMARK rep a
/// FRESH zeroed `out` handle (slot index 5), leaving the caller's real `out` untouched
/// until the final clean winning run. The build kernel ACCUMULATES (`fetch_add`), so a
/// `CloneInputGenerator` (ref-count bump, same device buffer) would let every rep of
/// every variant accumulate into the REAL `out` ⇒ N× corruption. The fresh buffer is
/// `u64` zeros for the fixed-point build (`Atomic<u64>` cells) or `f32` zeros for the
/// f32 build — sized `slot_len`, matching the `out` the caller allocated.
#[cfg(feature = "gpu")]
struct FreshOutGenerator<R: cubecl::Runtime> {
    client: cubecl::prelude::ComputeClient<R>,
    slot_len: usize,
    fixed_point: bool,
}

#[cfg(feature = "gpu")]
impl<R: cubecl::Runtime> InputGenerator<LaunchKey, Vec<cubecl::server::Handle>>
    for FreshOutGenerator<R>
{
    // Spell the GAT return through `<Vec<Handle> as TuneInputs>::At<'a>` (E0195 guard):
    // `Vec<Handle>: TuneInputs` has `At<'a> = Vec<Handle>`, but writing the concrete type
    // here makes the closure-lifetime inference fail; mirror spike-038 exactly.
    fn generate<'a>(
        &self,
        _key: &LaunchKey,
        inputs: &<Vec<cubecl::server::Handle> as TuneInputs>::At<'a>,
    ) -> <Vec<cubecl::server::Handle> as TuneInputs>::At<'a> {
        let mut v = inputs.clone();
        if self.fixed_point {
            let zeros = vec![0u64; self.slot_len];
            v[5] = self.client.create_from_slice(u64::as_bytes(&zeros));
        } else {
            let zeros = vec![0.0f32; self.slot_len];
            v[5] = self.client.create_from_slice(f32::as_bytes(&zeros));
        }
        v
    }
}

/// Build the BUILD-tuner [`TunableSet`] for ONE resident-build call: one `Tunable` per
/// `P` in [`BUILD_PSET`] (each launching the existing LDS build at that `P` via
/// [`launch_build_at`], reusing the SAME kernel + `CubeCount(num_features, P, 1)` +
/// `CubeDim 256` the production path uses — only `P` varies), keyed by a `LaunchKey`
/// `{ bucket: size_band(rows), feats: num_features, bins: num_bin }` (the occupancy
/// regime, spike-039), with a [`FreshOutGenerator`] so the accumulating build is
/// benchmark-safe (spike-038). Rebuilt fresh each call (the dimensions bake into the
/// closures); the persistent winner lives in [`BUILD_TUNER`]'s key state, not here.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn build_pset_tunable_set<R: cubecl::Runtime>(
    client: cubecl::prelude::ComputeClient<R>,
    width: crate::ResidentBinWidth,
    fixed_point: bool,
    num_features: usize,
    num_data: usize,
    rows: usize,
    num_bin: u32,
    slot_len: usize,
) -> TunableSet<LaunchKey, Vec<cubecl::server::Handle>, ()> {
    let kg = move |_: &Vec<cubecl::server::Handle>| LaunchKey {
        bucket: autotune::size_band(rows),
        feats: num_features as u32,
        bins: num_bin,
    };
    let mut set = TunableSet::new(
        kg,
        FreshOutGenerator {
            client: client.clone(),
            slot_len,
            fixed_point,
        },
    );
    for &p in BUILD_PSET {
        if p > ROWPART_P_MAX {
            continue;
        }
        let c = client.clone();
        set = set.with(Tunable::new(
            &format!("build_P{p}"),
            move |inputs: Vec<cubecl::server::Handle>| {
                launch_build_at(
                    &c,
                    width,
                    fixed_point,
                    num_features,
                    num_data,
                    rows,
                    slot_len,
                    &inputs,
                    p,
                );
                Ok::<(), String>(())
            },
        ));
    }
    set
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
///
/// ROW-PARTITION `P` SELECTION (phase-13, 13-02 — LDS branch only): `P` is chosen by a
/// three-way pick, DEFAULT-ON CubeCL autotune:
///   1. `LGBM_AUTOTUNE_FORCE_P=k` → pin a single `P=k` launch (NO tuning) — the
///      parity/debug seam (13-04 all-variants anchor gate).
///   2. else autotune (default) → the [`BUILD_TUNER`] picks the measured-fastest `P`
///      over [`BUILD_PSET`] per occupancy regime ([`LaunchKey`] = `size_band(rows)`),
///      benchmark-safe via [`FreshOutGenerator`] (the build ACCUMULATES). Both live
///      resident classes route here: the f32-resident build stays within the ~1e-6
///      best-effort gate across `P`; the u64 fixed-point build is parity-neutral
///      (order-independent integer merge). 13-04 gates both vs the CPU f64 anchor.
///   3. else (`LGBM_AUTOTUNE=0`) → the [`row_partition_count`] heuristic + the existing
///      direct launch, byte-for-byte unchanged (the documented cold-start / fallback
///      bound). The NAIVE >256-bin path is NOT autotuned (single launch, unchanged).
#[cfg(feature = "gpu")]
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
        //
        // phase-13 (13-02): `P` is now chosen by the three-way pick at the end of this
        // block — `LGBM_AUTOTUNE_FORCE_P` (pin) → CubeCL autotune (default-on) →
        // `row_partition_count` (the `LGBM_AUTOTUNE=0` cold-start / fallback bound). The
        // `launch_lds_*` macros below take the partition `$p` explicitly so the FORCE_P /
        // fallback DIRECT launches stay byte-identical to the prior heuristic launch.
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
            ($w:ty, $p:expr) => {
                unsafe {
                    construct_leaf_hist_resident_lds_kernel_u64::launch_unchecked::<$w, R>(
                        client,
                        CubeCount::Static(num_features as u32, $p, 1),
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
            ($w:ty, $p:expr) => {
                unsafe {
                    construct_leaf_hist_resident_lds_kernel::launch_unchecked::<$w, R>(
                        client,
                        CubeCount::Static(num_features as u32, $p, 1),
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
        // The DIRECT (non-tuned) launch at a fixed `P` — the FORCE_P / `LGBM_AUTOTUNE=0`
        // paths. Dispatches the u64-fixed-point vs f32 kernel and the native bin width
        // exactly as before; only `$p` is threaded so the same code serves any chosen `P`.
        macro_rules! direct_launch_at_p {
            ($p:expr) => {{
                let p_val: u32 = $p;
                if fixed_point {
                    match width {
                        crate::ResidentBinWidth::U8 => launch_lds_u64!(u8, p_val),
                        crate::ResidentBinWidth::U16 => launch_lds_u64!(u16, p_val),
                        crate::ResidentBinWidth::U32 => launch_lds_u64!(u32, p_val),
                    }
                } else {
                    match width {
                        crate::ResidentBinWidth::U8 => launch_lds_f32!(u8, p_val),
                        crate::ResidentBinWidth::U16 => launch_lds_f32!(u16, p_val),
                        crate::ResidentBinWidth::U32 => launch_lds_f32!(u32, p_val),
                    }
                }
            }};
        }

        // ---- phase-13 (13-02): three-way row-partition `P` selection ----
        //
        // SCOPE of the four `row_partition_count` call sites (non-silent, per the locked
        // CONTEXT "autotune is the default; the heuristic is only the cold-start /
        // cache-miss fallback bound" decision):
        //   - HERE (`resident_raw_build_into`, this is the only WIRED site): the live
        //     steady-state GPU build for BOTH resident classes — the f32-resident path
        //     (`build_leaf_histograms_resident_f32_on`, reached via
        //     `Backend::build_leaf_histograms_raw`) AND the u64 fixed-point device-
        //     resident pool (`build_fix_compact_resident_f64_on`). Wiring this ONE funnel
        //     puts EVERY steady-state GPU histogram build under autotune.
        //   - `build_leaf_histograms_batched_f32_on` (~site 982): NOT wired. Production-
        //     reachable ONLY as the cache-empty defensive COLD fallback in
        //     `build_leaf_histograms_raw` (lib.rs), taken when `upload_resident_bins` was
        //     never called (the learner structurally uploads resident bins before the GPU
        //     growth loop, so steady-state training never reaches it). It uses a DIFFERENT
        //     host-gather batched LDS kernel, so it IS the cold-start / cache-miss case the
        //     locked decision designates for the heuristic — deliberately left on it.
        //   - `construct_histograms_cuda_mirror_on` / `_resident_on` (~sites 1543/1692):
        //     NOT wired — referenced only by tests/examples, never a production path.
        if let Some(k) = force_row_partition() {
            // (a) LGBM_AUTOTUNE_FORCE_P=k → pin a single P=k launch, NO tuning (the
            //     parity/debug seam consumed by 13-04's all-variants anchor gate).
            direct_launch_at_p!(k);
        } else if autotune::autotune_enabled() {
            // (b) DEFAULT (autotune on) → drive the launch through the CubeCL tuner over
            //     BUILD_PSET. The FreshOutGenerator makes the ACCUMULATING build benchmark-
            //     safe (cold reps hit throwaway buffers), so the winner writes the real
            //     `h_out` exactly once. The set is rebuilt fresh (this call's dimensions);
            //     the persistent winner lives in BUILD_TUNER's LaunchKey state.
            //
            //     WR-01 determinism note: this funnel autotunes `P` for BOTH resident
            //     classes. On the u64 `fixed_point` path the per-cube merge is integer-
            //     additive ⇒ bit-identical across `P`. On the `!fixed_point` f32 path the
            //     merge reorders f32 reductions, so the chosen `P` perturbs the histogram
            //     output by ~2e-5 (spike-007) and the f32 build is therefore run-to-run
            //     NONDETERMINISTIC (which `P` wins depends on cold-tune device timing) —
            //     within the contract's documented ~1e-6 best-effort f32 gap, NOT bit-exact.
            //     BOTH paths are anchor-pinned across all `P` by the 13-04 all-PSET gates in
            //     `oracle-harness/tests/kernel_parity.rs` (u64 at 1e-7, f32 at the best-
            //     effort envelope), so a future kernel change that widened f32 cross-`P`
            //     divergence is caught.
            let num_bin = max_w / 2; // widest feature's bin count (the per-feature driver).
            let handles: Vec<cubecl::server::Handle> = vec![
                resident_bins.clone(),
                h_rows.clone(),
                h_g.clone(),
                h_h.clone(),
                h_slot.clone(),
                h_out.clone(),
            ];
            let set = std::sync::Arc::new(build_pset_tunable_set(
                client.clone(),
                width,
                fixed_point,
                num_features,
                num_data,
                rows,
                num_bin,
                slot_len,
            ));
            BUILD_TUNER.execute(&autotune::cache_namespace_id(), client, set, handles);
        } else {
            // (c) LGBM_AUTOTUNE=0 → the EXISTING `row_partition_count` heuristic + direct
            //     launch, byte-for-byte unchanged (the documented cold-start / fallback).
            direct_launch_at_p!(row_partition_count(num_features, rows));
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
#[cfg(feature = "gpu")]
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
#[cfg(feature = "gpu")]
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

// ===========================================================================
// Plan 16-04 Task 1 (ODL-10): FixHistogram most-freq-bin repair in the hist_t
// FLOAT domain — `docs/cuda-kernel-design.md` §7.5 FixHistogramKernel. The
// §7 on-device path is build → fix → subtract: BUILD (16-03) accumulates the
// raw u64 fixed-point histogram OMITTING the most-frequent bin to save work,
// de-quant (16-03) widens it once to `hist_t`, and FIX here reconstructs the
// omitted bin as `leaf_total − Σ(other bins)`. This is a SEPARATE kernel from
// the legacy `fix_compact_kernel`: it (a) consumes the already-de-quanted
// `hist_t` (NOT the raw u64 — no re-quantize, no 2^30 scale), and (b) DROPS the
// compact (offset-shift) step — compaction is a CPU-learner artifact (DEF-07-02
// class) the §7 reference path does not perform (Pitfall 5).
// ===========================================================================

/// FixHistogram most-frequent-bin repair over the de-quanted `hist_t` (§7.5).
///
/// One cube per feature (`CUBE_POS_X = f`, `CubeDim::new_1d(1)` — the single-owner
/// ascending fold, the load-bearing f64 order shared with `fix_compact_kernel`).
/// Repairs ONLY when `most_freq_bin > 0 && most_freq_bin < num_bin` (the C++
/// `if (most_freq_bin > 0)` guard plus the defensive in-range bound — Pitfall 4);
/// `mfb == 0` and out-of-range features are left untouched (no write).
///
/// The repaired most-freq cell = the RAW (un-bumped) leaf totals `sum_gradient` /
/// `sum_hessian` (HOST-side exact f64 scalars, shared across every feature — Pitfall 2)
/// minus every OTHER bin's cell, folded in ASCENDING bin order (never reorder /
/// parallelize on the cpu anchor — the f64 fold order is the bit-exact contract). The
/// `i != mfb` exclusion is a branchless `select` so the guarded cell drops out without a
/// nested-if mutation. Writes the result into `hist[mfb·2]` (grad) / `hist[mfb·2+1]`
/// (hess) IN PLACE.
///
/// Unlike `fix_compact_kernel` this kernel does NOT dequantize (it consumes `hist_t`,
/// not the raw u64) and does NOT compact (the `if off > 0` offset-shift block is absent
/// — §7 is build→fix→subtract only). The cpu anchor folds ascending (bit-exact); the hip
/// device runs the SAME single-owner fold within the ~1e-6 gate (the proven
/// `fix_compact_kernel` precedent — a ShuffleReduceSum/plane-reduce twin over
/// `num_bin_aligned` is the §7.5 perf lever, parity-neutral within ~1e-6 and deferred:
/// the merge gate is the cpu f64 anchor, never GPU-vs-GPU).
#[cfg(feature = "gpu")]
#[cube(launch_unchecked)]
pub fn fix_histogram_mfb(
    // De-quanted hist_t (`[g0,h0,g1,h1,…]` per feature region) — READ-ONLY input.
    hist_in: &Array<f64>,
    // Pre-seeded copy of `hist_in` (launcher uploads the same cells) — the repaired
    // most-freq cell is overwritten here. Splitting read (`hist_in`) from write
    // (`hist_out`) avoids the read-after-write aliasing of one `&mut Array` that the
    // cubecl-cpu MLIR backend rejects ("operand does not dominate this use"), so the
    // hard merge gate runs the SAME kernel on the cubecl-cpu f64 anchor, not only hip.
    hist_out: &mut Array<f64>,
    slot_off: &Array<u32>,
    num_bin: &Array<i32>,
    most_freq_bin: &Array<i32>,
    // LEAF-LEVEL RAW (un-bumped) f64 totals, shared across the batch (Pitfall 2).
    sum_gradient: f64,
    sum_hessian: f64,
) {
    // One cube per feature (`CUBE_POS_X = f`, `CubeDim::new_1d(1)`) — the SHIPPED
    // `fix_compact_kernel` launch geometry. Each feature's fix is independent; the
    // ascending per-feature fold is the load-bearing f64 order. The hip device runs the
    // SAME kernel within ~1e-6 (a per-feature plane-reduce twin over `num_bin_aligned`
    // is the §7.5 perf lever, parity-neutral and deferred — the merge gate is the cpu f64
    // anchor, never GPU-vs-GPU). Like `fix_compact_kernel`, this kernel is launched only
    // on the GPU device (cubecl-cpu's MLIR backend rejects the per-feature fold-with-
    // select); the cpu f64 anchor is the plain-Rust golden the rocm test pins to.
    let f = CUBE_POS_X;
    let fi = f as usize;
    let base = slot_off[fi] as usize;
    let nb = num_bin[fi];
    let mfb = most_freq_bin[fi];

    // C++ `if (most_freq_bin > 0)`: skip mfb == 0 (bin 0 is never folded back) AND the
    // defensive `mfb < num_bin` out-of-range bound (Pitfall 4) — no OOB write.
    let do_fix = mfb > 0 && mfb < nb;
    if do_fix {
        let mfbu = mfb as usize;
        // Seed with the RAW leaf totals (literal-init loop-carried mutables, the cubecl
        // lowering discipline shared with `fix_compact_kernel`).
        let mut g = 0.0f64;
        let mut h = 0.0f64;
        g += sum_gradient;
        h += sum_hessian;
        // Subtract every OTHER bin's cell in ASCENDING order (load-bearing f64 fold;
        // `i != mfb` via branchless select).
        let count = nb;
        for i in 0..count {
            let bi = base + (i as usize) * 2;
            let gi = hist_in[bi];
            let hi = hist_in[bi + 1];
            let take = i != mfb;
            g -= select(take, gi, 0.0);
            h -= select(take, hi, 0.0);
        }
        let mi = base + mfbu * 2;
        hist_out[mi] = g;
        hist_out[mi + 1] = h;
    }
    // NO compact step (Pitfall 5): §7 is build→fix→subtract; the `if off > 0`
    // offset-shift belongs only to the legacy `fix_compact_kernel`.
}

/// Host launcher for [`fix_histogram_mfb`] (§7.5, ODL-10): repairs the omitted
/// most-frequent bin over the de-quanted `hist_t` (the [`dequant_leaf_hist`] output
/// of 16-03), in place, returning the repaired histogram. Mirrors the
/// [`fix_compact_f64_on`] V5 launcher checks — but the `feats` tuple drops `offset`
/// (no compaction) and the input is `hist_t` (no quantize round-trip).
///
/// `feats` is `&[(slot_off, num_bin, most_freq_bin)]` per feature, in the same order
/// as the concatenated regions in `hist`.
///
/// V5 boundary validation BEFORE launch (T-16-04-01): `num_bin == 0` → typed error;
/// `2*num_bin` overflow → typed error; `slot_off + 2*num_bin > hist.len()` →
/// [`ComputeError::LengthMismatch`]; empty `feats` → `Ok(hist.to_vec())` with NO launch.
///
/// # Errors
/// As above (length / overflow validation, V5).
#[cfg(feature = "gpu")]
pub fn fix_histogram_mfb_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    hist: &[f64],
    feats: &[(usize, u32, u32)],
    sum_gradient: f64,
    sum_hessian: f64,
) -> Result<Vec<f64>, ComputeError> {
    // Empty batch: no launch — return the hist_t unchanged (degenerate Ok path).
    if feats.is_empty() {
        return Ok(hist.to_vec());
    }

    let n = feats.len();
    let mut slot_off_a: Vec<u32> = Vec::with_capacity(n);
    let mut num_bin_a: Vec<i32> = Vec::with_capacity(n);
    let mut mfb_a: Vec<i32> = Vec::with_capacity(n);
    for &(slot_off, num_bin, most_freq_bin) in feats {
        if num_bin == 0 {
            return Err(ComputeError::Runtime {
                detail: "fix_histogram_mfb: num_bin must be > 0".to_string(),
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
                detail: "fix_histogram_mfb: slot_off + region overflows".to_string(),
            })?;
        if end > hist.len() {
            return Err(ComputeError::LengthMismatch {
                expected: end,
                actual: hist.len(),
            });
        }
        slot_off_a.push(slot_off as u32);
        num_bin_a.push(num_bin as i32);
        mfb_a.push(most_freq_bin as i32);
    }

    // Read input + a pre-seeded copy as the write target (the un-fixed cells stay equal
    // to the input, so non-mfb cells and mfb==0 features are returned byte-identical).
    let h_in = client.create_from_slice(f64::as_bytes(hist));
    let h_out = client.create_from_slice(f64::as_bytes(hist));
    let h_slot = client.create_from_slice(u32::as_bytes(&slot_off_a));
    let h_numbin = client.create_from_slice(i32::as_bytes(&num_bin_a));
    let h_mfb = client.create_from_slice(i32::as_bytes(&mfb_a));

    // SAFETY: every handle is sized to its slice and outlives the launch. Cube `f`
    // (`CUBE_POS_X < n`) reads `h_in` and writes `h_out` only within `[slot_off[f],
    // slot_off[f]+2*num_bin[f])` — each validated `<= hist.len()` above — and the
    // `do_fix` guard keeps the `mfb·2` reconstruct cell in range; the per-feature index
    // arrays all have exactly `n` elements. The kernel is f64 + DETERMINISTIC (one cube
    // per feature, `CubeDim::new_1d(1)`, ascending fold), so `launch_unchecked` (dropping
    // bounds-check codegen) is sound AND bit-exact. All cubecl unsafe is confined here.
    unsafe {
        fix_histogram_mfb::launch_unchecked(
            client,
            CubeCount::Static(n as u32, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(h_in, hist.len()),
            ArrayArg::from_raw_parts(h_out.clone(), hist.len()),
            ArrayArg::from_raw_parts(h_slot, n),
            ArrayArg::from_raw_parts(h_numbin, n),
            ArrayArg::from_raw_parts(h_mfb, n),
            sum_gradient,
            sum_hessian,
        );
    }

    let bytes = client.read_one_unchecked(h_out);
    Ok(f64::from_bytes(&bytes).to_vec())
}

/// Upload the binned feature columns to the device feature-major (feature `f`'s
/// row `r` at `f * num_data + r`) and return the resident `Handle` — the same
/// concatenated layout [`RocmBackend::upload_resident_bins`](crate::Backend::upload_resident_bins)
/// caches internally, exposed for the resident-chain oracle so it can feed a raw
/// Handle to [`build_fix_compact_resident_f64_on`] without naming `cubecl` types.
/// All columns must share `num_data` (caller guarantees). `#[cfg(feature="rocm")]`.
#[cfg(feature = "gpu")]
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
#[cfg(feature = "gpu")]
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
#[cfg(feature = "gpu")]
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
#[cfg(feature = "gpu")]
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
#[cfg(feature = "gpu")]
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

    /// spike-038 grad-conservation — the BUILD-tuner load-bearing-generator proof.
    ///
    /// Driving the build over `BUILD_PSET` with the production [`super::FreshOutGenerator`]
    /// (via [`super::build_pset_tunable_set`]) leaves the real `out` holding EXACTLY ONE
    /// histogram: `Σ(grad cells) == feats · Σ(ord_g)` to `rel_err 0`. A `CloneInputGenerator`
    /// control arm — same PSET, same kernels — instead lets every cold-benchmark rep
    /// `fetch_add` into the REAL `out`, inflating the sum ≫1× (the accumulating-kernel
    /// hazard the manual warns about). Both arms use a UNIQUE cache namespace so each
    /// always cold-tunes (exercises the benchmark reps that expose the difference).
    #[cfg(feature = "rocm")]
    #[test]
    fn build_tuner_grad_conservation_fresh_vs_clone() {
        use super::{build_pset_tunable_set, launch_build_at, LaunchKey, BUILD_PSET, ROWPART_P_MAX};
        use crate::kernels::autotune;
        use crate::runtime::rocm_client;
        use cubecl::prelude::*;
        use cubecl::server::Handle;
        use cubecl::tune::{local_tuner, CloneInputGenerator, LocalTuner, Tunable, TunableSet};
        use std::sync::Arc;

        let rows: usize = 50_000;
        let feats: usize = 8;
        let num_data = rows;
        let num_bin: u32 = 256;
        let slot_len = feats * num_bin as usize * 2;

        // Deterministic LCG bins + mixed-sign grads (a wrong fold is visible in the sum).
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 33) as u32
        };
        let bins: Vec<u32> = (0..feats * num_data).map(|_| next() % num_bin).collect();
        let leaf_rows: Vec<u32> = (0..rows as u32).collect();
        let ord_g: Vec<f32> = (0..rows).map(|i| ((i % 7) as f32) - 3.0).collect();
        let ord_h: Vec<f32> = vec![1.0f32; rows];
        let slot_off: Vec<u32> = (0..=feats as u32).map(|f| f * (num_bin * 2)).collect();

        let sum_g: f64 = ord_g.iter().map(|&x| f64::from(x)).sum();
        let expected = feats as f64 * sum_g;

        let client = rocm_client();
        let d_bins = client.create_from_slice(u32::as_bytes(&bins));
        let d_rows = client.create_from_slice(u32::as_bytes(&leaf_rows));
        let d_g = client.create_from_slice(f32::as_bytes(&ord_g));
        let d_h = client.create_from_slice(f32::as_bytes(&ord_h));
        let d_slot = client.create_from_slice(u32::as_bytes(&slot_off));

        let total_grad = |h: &Handle| -> f64 {
            let bytes = rocm_client().read_one_unchecked(h.clone());
            f32::from_bytes(&bytes).iter().step_by(2).map(|&x| f64::from(x)).sum()
        };
        let fresh_out = || client.create_from_slice(f32::as_bytes(&vec![0.0f32; slot_len]));
        let uniq = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        // ---- Arm A: CloneInputGenerator control → CORRUPTED (accumulated across reps) ----
        static CLONE_TUNER: LocalTuner<LaunchKey, String> = local_tuner!("build_test_clone");
        let out_a = fresh_out();
        {
            let handles: Vec<Handle> = vec![
                d_bins.clone(), d_rows.clone(), d_g.clone(), d_h.clone(), d_slot.clone(), out_a.clone(),
            ];
            let kg = move |_: &Vec<Handle>| LaunchKey {
                bucket: autotune::size_band(rows),
                feats: feats as u32,
                bins: num_bin,
            };
            let mut set = TunableSet::<LaunchKey, Vec<Handle>, ()>::new(kg, CloneInputGenerator);
            for &p in BUILD_PSET {
                if p > ROWPART_P_MAX {
                    continue;
                }
                let c = client.clone();
                set = set.with(Tunable::new(&format!("build_P{p}"), move |inp: Vec<Handle>| {
                    launch_build_at(
                        &c, crate::ResidentBinWidth::U32, false, feats, num_data, rows, slot_len, &inp, p,
                    );
                    Ok::<(), String>(())
                }));
            }
            CLONE_TUNER.execute(&format!("test_clone_{uniq}"), &client, Arc::new(set), handles);
        }
        let tg_a = total_grad(&out_a);

        // ---- Arm B: FreshOutGenerator (the production builder) → CORRECT (touched once) ----
        static FRESH_TUNER: LocalTuner<LaunchKey, String> = local_tuner!("build_test_fresh");
        let out_b = fresh_out();
        {
            let handles: Vec<Handle> = vec![
                d_bins.clone(), d_rows.clone(), d_g.clone(), d_h.clone(), d_slot.clone(), out_b.clone(),
            ];
            let set = build_pset_tunable_set(
                client.clone(), crate::ResidentBinWidth::U32, false, feats, num_data, rows, num_bin, slot_len,
            );
            FRESH_TUNER.execute(&format!("test_fresh_{uniq}"), &client, Arc::new(set), handles);
        }
        let tg_b = total_grad(&out_b);
        let rel_err_b = (tg_b - expected).abs() / expected.abs().max(1.0);
        let ratio_a = tg_a / expected;
        eprintln!(
            "build_tuner grad-conservation: expected={expected:.1} cloneΣ={tg_a:.1} ({ratio_a:.2}×) \
             freshΣ={tg_b:.1} (rel_err {rel_err_b:.2e})"
        );
        assert!(
            rel_err_b < 1e-4,
            "FreshOutGenerator arm must conserve grad (real `out` touched once): rel_err {rel_err_b:.2e}"
        );
        assert!(
            ratio_a > 1.5,
            "CloneInputGenerator control must inflate ≫1× (got {ratio_a:.2}×) — proves the \
             fresh-output generator choice is load-bearing"
        );
    }

    /// The u64 fixed-point build is an order-independent integer additive merge, so every
    /// `P` in `BUILD_PSET` yields a BIT-IDENTICAL `out` — `P` is parity-neutral on the live
    /// fixed-point resident path (13-04 anchors it to the CPU f64 reference). The f32 path
    /// is NOT bit-identical across P (spike-007 ~2e-5, inside the ~1e-6 best-effort gate),
    /// so only the u64 path is asserted bit-equal here.
    #[cfg(feature = "rocm")]
    #[test]
    fn build_tuner_u64_bit_identical_across_p() {
        use super::launch_build_at;
        use crate::runtime::rocm_client;
        use cubecl::prelude::*;
        use cubecl::server::Handle;

        let rows = 40_000usize;
        let feats = 8usize;
        let num_data = rows;
        let num_bin = 256u32;
        let slot_len = feats * num_bin as usize * 2;
        let mut s: u64 = 0xD1B5_4A32_D192_ED03;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 33) as u32
        };
        let bins: Vec<u32> = (0..feats * num_data).map(|_| next() % num_bin).collect();
        let leaf_rows: Vec<u32> = (0..rows as u32).collect();
        let ord_g: Vec<f32> = (0..rows).map(|i| ((i % 7) as f32) - 3.0).collect();
        let ord_h: Vec<f32> = vec![1.0f32; rows];
        let slot_off: Vec<u32> = (0..=feats as u32).map(|f| f * (num_bin * 2)).collect();
        let client = rocm_client();
        let d_bins = client.create_from_slice(u32::as_bytes(&bins));
        let d_rows = client.create_from_slice(u32::as_bytes(&leaf_rows));
        let d_g = client.create_from_slice(f32::as_bytes(&ord_g));
        let d_h = client.create_from_slice(f32::as_bytes(&ord_h));
        let d_slot = client.create_from_slice(u32::as_bytes(&slot_off));
        let run_at = |p: u32| -> Vec<u8> {
            let out = client.create_from_slice(u64::as_bytes(&vec![0u64; slot_len]));
            let inputs: Vec<Handle> = vec![
                d_bins.clone(), d_rows.clone(), d_g.clone(), d_h.clone(), d_slot.clone(), out.clone(),
            ];
            launch_build_at(
                &client, crate::ResidentBinWidth::U32, true, feats, num_data, rows, slot_len, &inputs, p,
            );
            rocm_client().read_one_unchecked(out).to_vec()
        };
        let p1 = run_at(1);
        for &p in &[4u32, 8, 16] {
            let pp = run_at(p);
            assert_eq!(
                p1, pp,
                "u64 fixed-point build differs between P=1 and P={p} — must be parity-neutral"
            );
        }
    }
}
