//! Plane scan/reduction intrinsic per-backend lowering smoke test
//! (Phase-14 Wave-0 de-risk gate — RESEARCH Open Q1 / Pitfall 1).
//!
//! The shipped histogram path only ever exercised `plane_sum` / `plane_any` /
//! `plane_ballot` / `plane_shuffle` on hip. The cubecl-0.10 scan/reduction
//! intrinsics `plane_inclusive_sum` / `plane_exclusive_sum` / `plane_max` /
//! `plane_min` are UNPROVEN in this repo, and every Phase-14/15 primitive
//! (prefix-sum, reductions) is built on them. This test front-loads the single
//! MEDIUM unknown of the phase: do those four intrinsics actually LOWER and
//! produce correct results on the cubecl-cpu f64 anchor (default) and on
//! cubecl-hip (`--features rocm`)?
//!
//! ## Anchor discipline (D-10, def-f8u-01)
//! Each device result is compared against a plain **serial Rust reference fold**
//! (inclusive/exclusive prefix sum, running max/min) — NEVER GPU-vs-GPU. Inputs
//! are small exact integers (representable in f32), so the cpu anchor asserts
//! **bit-exact**; the hip f32 path is held to the existing ~1e-6 envelope.
//!
//! ## Plane width is NOT hard-coded (RESEARCH Pattern 1 note)
//! The launch cube-dim and the serial reference length both come from the probed
//! `Capabilities::plane_size` (1 on cubecl-cpu, 32 on gfx1100). The kernels index
//! by the `UNIT_POS_PLANE` runtime builtin. On the cpu anchor a "plane" is a
//! single lane (`plane_size == 1`), so the cross-lane behaviour is degenerate but
//! the **lowering** of every intrinsic is still proven; the cross-lane numerics
//! are exercised on hip.
//!
//! ## Silent no-op detector
//! A backend that fails to lower an intrinsic may panic at launch (loud) OR
//! silently return the input (a no-op). The discriminator that catches the silent
//! case even on a single-lane plane is **`plane_exclusive_sum` of the first lane,
//! which must be `0.0`** (an empty prefix), never the input value.
//!
//! ## Fallback
//! If any intrinsic fails to lower, the affected op must switch to a
//! `plane_shuffle_up`-based manual scan (RESEARCH Pattern 1 alternative) and this
//! test pins the manual-scan variant. The outcome (intrinsic vs fallback, per
//! intrinsic, per backend) is recorded in 14-01-SUMMARY.md for 14-03 to consume.

// The cubecl prelude (`#[cube]`, `Array`, `ArrayArg`, `CubeCount`/`CubeDim`,
// `plane_*`, `UNIT_POS_PLANE`, `*::as_bytes/from_bytes`) is only needed by the
// plane probe kernels + their launcher, which are all `rocm`-gated (cubecl-cpu
// has no plane support — see the cpu test below). Gating the import keeps the
// default cpu build warning-clean.
#[cfg(feature = "rocm")]
use cubecl::prelude::*;
use lgbm_compute::runtime::probe_capabilities;

// --- Probe kernels: one per intrinsic, isolated so a single failed lowering is
// attributable to exactly one op (a combined kernel would mask which one panicked).
// Each lane reads its own input element and writes its plane-op result.

#[cfg(feature = "rocm")]
#[cube(launch)]
fn k_inclusive_sum(input: &Array<f32>, out: &mut Array<f32>) {
    let lane = UNIT_POS_PLANE as usize;
    out[lane] = plane_inclusive_sum(input[lane]);
}

#[cfg(feature = "rocm")]
#[cube(launch)]
fn k_exclusive_sum(input: &Array<f32>, out: &mut Array<f32>) {
    let lane = UNIT_POS_PLANE as usize;
    out[lane] = plane_exclusive_sum(input[lane]);
}

#[cfg(feature = "rocm")]
#[cube(launch)]
fn k_max(input: &Array<f32>, out: &mut Array<f32>) {
    let lane = UNIT_POS_PLANE as usize;
    out[lane] = plane_max(input[lane]);
}

#[cfg(feature = "rocm")]
#[cube(launch)]
fn k_min(input: &Array<f32>, out: &mut Array<f32>) {
    let lane = UNIT_POS_PLANE as usize;
    out[lane] = plane_min(input[lane]);
}

/// Serial reference folds over the active lane window (length `plane`).
fn serial_refs(input: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = input.len();
    let mut incl = vec![0.0f32; n];
    let mut excl = vec![0.0f32; n];
    let mut acc = 0.0f32;
    for i in 0..n {
        excl[i] = acc; // empty prefix at i==0 -> 0.0 (the no-op discriminator)
        acc += input[i];
        incl[i] = acc;
    }
    let mx = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mn = input.iter().copied().fold(f32::INFINITY, f32::min);
    let pmax = vec![mx; n];
    let pmin = vec![mn; n];
    (incl, excl, pmax, pmin)
}

/// Compare device vs serial reference, bit-exact when `tol == 0.0`, else within
/// `tol` absolute. Returns Ok(()) or a descriptive failure (so the caller can
/// record per-intrinsic which lowered).
#[cfg(feature = "rocm")]
fn assert_close(label: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if tol == 0.0 {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{label}: lane {i} not bit-exact — got {g}, want {w} \
                 (intrinsic failed to lower / silent no-op? switch to plane_shuffle_up)"
            );
        } else {
            assert!(
                (g - w).abs() <= tol,
                "{label}: lane {i} outside {tol:e} — got {g}, want {w}"
            );
        }
    }
}

/// The shared body: probe all four intrinsics on `client` at the probed plane
/// width and assert each against the serial reference. `tol == 0.0` -> bit-exact
/// (cpu anchor, exact-integer inputs); `tol > 0.0` -> ~1e-6 (hip f32).
#[cfg(feature = "rocm")]
fn probe_all<R: Runtime>(client: &ComputeClient<R>, plane: u32, tol: f32) {
    // Exact small-integer inputs (1.0, 2.0, ... plane): every inclusive sum
    // (max plane*(plane+1)/2 = 2080 at wave64) is exactly representable in f32.
    let input: Vec<f32> = (0..plane).map(|i| (i + 1) as f32).collect();
    let (incl, excl, pmax, pmin) = serial_refs(&input);
    let n = input.len();

    // Allocate input once, reused (cloned) across the four launches; one fresh
    // zeroed output per intrinsic.
    let h_in = client.create_from_slice(f32::as_bytes(&input));
    let zeros = vec![0.0f32; n];

    // SAFETY (all four launches): `h_in`/`h_out` are each allocated for exactly `n`
    // (== plane) f32 elements and outlive their launch; every kernel indexes only
    // `UNIT_POS_PLANE` in `0..plane`, and each launch dispatches exactly `plane`
    // units in a single cube (`CubeCount::Static(1,1,1)`). cubecl unsafe is confined
    // to these launchers (CMP-01).

    let h_out = client.create_from_slice(f32::as_bytes(&zeros));
    unsafe {
        k_inclusive_sum::launch(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(plane),
            ArrayArg::from_raw_parts(h_in.clone(), n),
            ArrayArg::from_raw_parts(h_out.clone(), n),
        );
    }
    let got_incl = f32::from_bytes(&client.read_one_unchecked(h_out)).to_vec();
    assert_close("plane_inclusive_sum", &got_incl, &incl, tol);

    let h_out = client.create_from_slice(f32::as_bytes(&zeros));
    unsafe {
        k_exclusive_sum::launch(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(plane),
            ArrayArg::from_raw_parts(h_in.clone(), n),
            ArrayArg::from_raw_parts(h_out.clone(), n),
        );
    }
    let got_excl = f32::from_bytes(&client.read_one_unchecked(h_out)).to_vec();
    assert_close("plane_exclusive_sum", &got_excl, &excl, tol);

    let h_out = client.create_from_slice(f32::as_bytes(&zeros));
    unsafe {
        k_max::launch(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(plane),
            ArrayArg::from_raw_parts(h_in.clone(), n),
            ArrayArg::from_raw_parts(h_out.clone(), n),
        );
    }
    let got_max = f32::from_bytes(&client.read_one_unchecked(h_out)).to_vec();
    assert_close("plane_max", &got_max, &pmax, tol);

    let h_out = client.create_from_slice(f32::as_bytes(&zeros));
    unsafe {
        k_min::launch(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(plane),
            ArrayArg::from_raw_parts(h_in.clone(), n),
            ArrayArg::from_raw_parts(h_out.clone(), n),
        );
    }
    let got_min = f32::from_bytes(&client.read_one_unchecked(h_out)).to_vec();
    assert_close("plane_min", &got_min, &pmin, tol);
}

/// DE-RISK FINDING (Wave-0, RESEARCH Open Q1 / Pitfall 1 / Assumption A1):
/// **the cubecl-cpu backend does NOT lower the plane scan/reduction intrinsics
/// — nor even the `UNIT_POS_PLANE` builtin — at all.** Launching any of
/// `k_inclusive_sum` / `k_exclusive_sum` / `k_max` / `k_min` on `cpu_client()`
/// aborts the worker thread with either
///   `Unsupported builtin was used: UnitPosPlane`   (args_manager.rs)
/// or
///   `plane_inclusive_sum(..) is not supported on CPU.` (operation/mod.rs)
/// This is consistent with `probe_capabilities(cpu).has_plane == false` /
/// `plane_size == 1` (RESEARCH Pitfall 2 capability matrix): the cubecl-cpu
/// runtime has no plane (warp) concept, so plane collectives cannot run on it.
///
/// **Consequence for 14-03 (the input to consume):** the f64 cpu ANCHOR is the
/// *serial sequential fold* (`ReducePath::Sequential`, the existing
/// single-owner ordered reference) — it is NOT, and cannot be, a plane kernel.
/// The `plane_shuffle_up` "fallback" from RESEARCH Pattern 1 is itself a plane
/// op and is therefore ALSO unavailable on cpu — it is only a within-backend
/// fallback for a plane-capable device that fails to lower a *specific* scan op.
/// The plane intrinsics are a strictly GPU-side (`has_plane`) concern; their
/// lowering is proven by `plane_intrinsics_lower_on_hip` below.
///
/// This test asserts that architectural fact (so a future cubecl bump that
/// silently changed cpu plane support would flip this gate and force a re-eval),
/// and validates the serial reference folds that ARE the cpu anchor.
#[cfg(feature = "cpu")]
#[test]
fn cpu_anchor_has_no_plane_support_serial_fold_is_the_reference() {
    use lgbm_compute::runtime::cpu_client;
    let client = cpu_client();
    let caps = probe_capabilities(&client);

    // The capability fact that makes plane intrinsics a GPU-only concern.
    assert!(
        !caps.has_plane,
        "cubecl-cpu unexpectedly reports has_plane=true — plane lowering on cpu \
         may now be possible; re-evaluate the cpu-anchor==serial-fold decision"
    );
    assert_eq!(
        caps.plane_size, 1,
        "cubecl-cpu plane_size must be 1 (no warp concept); got {}",
        caps.plane_size
    );

    // The serial reference folds ARE the cpu anchor (what downstream f64 anchoring
    // compares against). Sanity-check them on a known window so the reference used
    // by the hip gate (and by 14-03) is itself proven correct.
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let (incl, excl, pmax, pmin) = serial_refs(&input);
    assert_eq!(incl, vec![1.0, 3.0, 6.0, 10.0], "inclusive prefix-sum reference");
    assert_eq!(excl, vec![0.0, 1.0, 3.0, 6.0], "exclusive prefix-sum reference");
    assert_eq!(pmax, vec![4.0, 4.0, 4.0, 4.0], "running-max reference");
    assert_eq!(pmin, vec![1.0, 1.0, 1.0, 1.0], "running-min reference");
}

/// ROCm parity half: the same four intrinsics over a FULL hardware plane (32 lanes
/// on gfx1100), exercising the genuine cross-lane scan/reduction. Guarded by
/// `has_plane`; held to the existing ~1e-6 envelope (the inputs are exact so this
/// is comfortably bit-exact in practice, but the gate matches the documented f32
/// tolerance, not a widened one). Pinned to the serial reference, never GPU-vs-GPU.
#[cfg(feature = "rocm")]
#[test]
fn plane_intrinsics_lower_on_hip() {
    use lgbm_compute::runtime::rocm_client;
    let client = rocm_client();
    let caps = probe_capabilities(&client);
    assert!(caps.has_plane, "gfx-class GPU must report has_plane");
    let plane = caps.plane_size.max(1);
    probe_all(&client, plane, 1e-6);
}
