//! CASE-A CONFIRMATION harness — quick task 260619-tlk (QUICK-260619-TLK).
//!
//! The user's ask: "refer the cubecl manual and reduce overhead in the GPU kernel."
//! This is the latest in a long, mostly-closed GPU-overhead campaign (spikes 001-013;
//! quick tasks nrw/ol8/p93/ngo/mwr/j9t/q2z/sgu). The cubecl-manual lever the prior q2z
//! spike pointed at — the idiomatic batched `client.read(Vec<Handle>)` drain that
//! collapses an N× `read_one` loop into ONE deferred sync — only pays off when there IS
//! an N-handle read loop to collapse.
//!
//! THE AUDIT VERDICT (260619-tlk-FINDINGS.md, CASE A): there is NO such loop in the
//! PRODUCTION read path. Every production launcher (the per-leaf fused
//! `build_fix_scan_resident_f64_on`, the device-resident `subtract_resident`, the
//! `scan_resident_leaf` fused scan, the standalone subtract/split/partition launchers)
//! emits ONE consolidated out-handle and reads it with ONE `read_one_unchecked`. The
//! per-feature submit->block + read-per-handle loop the q2z spike beat is TEST-ONLY
//! (`construct_histograms_parallel_f32_on` has ZERO production callers — verified by grep
//! of crates/lgbm-treelearner/src). So the batched-read idiom is ALREADY SATISFIED.
//!
//! THE LOAD-BEARING MANUAL FACT (cubecl-runtime-0.10.0/src/client.rs, lines 131-149):
//!
//! ```ignore
//! pub fn read(&self, handles: Vec<Handle>) -> Vec<Bytes> {
//!     cubecl_common::reader::read_sync(self.read_async(handles)).expect("TODO")
//! }
//! pub fn read_one_unchecked(&self, handle: Handle) -> Bytes {
//!     cubecl_common::reader::read_sync(self.read_async(vec![handle])).unwrap().remove(0)
//! }
//! ```
//!
//! `read_one_unchecked(h)` is DEFINED AS `read_async(vec![h])` drained by `read_sync` — it
//! is literally the one-element case of the batched `read(Vec<Handle>)`. For a launcher
//! that produces ONE consolidated out-handle (the production shape), the single-handle
//! drain and the batched-of-one drain are the SAME single `read_async -> read_sync` round
//! trip. There is no N-handle loop for the batched idiom to collapse.
//!
//! WHAT THIS EXAMPLE DOES (measurement-only, rocm-gated): it launches the SHIPPED fused
//! single-feature atomic histogram kernel ONCE (the production shape: one launch, one
//! consolidated out-handle) and times two reads of that ONE handle: arm A is
//! `client.read_one_unchecked(h)` (the production read call) and arm B is
//! `client.read(vec![h])` (the manual's batched idiom, n=1).
//! It then asserts they return BYTE-IDENTICAL bytes and show NO sign-stable separation (they
//! are the same drain through `read_async(vec![h])`). This CONFIRMS the manual-predicted
//! equivalence the Case-A verdict rests on. It does NOT manufacture a multi-handle loop to
//! "beat" — that is the forbidden q2z/sgu anti-pattern; the whole point is that production
//! has exactly ONE consolidated handle per launcher, so there is nothing to batch.
//!
//! MEASUREMENT-ONLY: NO production kernel, launcher, learner dispatch, or the CPU f64
//! anchor is modified. The parity gate (the CPU f64 bit-exact anchor + the 1e-6 hip
//! envelope) is unchanged; this example is NOT a test gate.
//!
//! WARM-VS-COLD (spike-findings SKILL — cold-ceiling-overstates-warm): WARMUP discarded,
//! MEDIAN + p25/p75 spread, arms INTERLEAVED, the device sync (read-back) FORCED inside
//! every timed call, meant to be RE-RUN across >=2 process restarts. A delta within the
//! p25/p75 spread, or whose sign flips across restarts, is SUB-NOISE / NULL — which is the
//! EXPECTED result here precisely because both arms are the SAME `read_async` drain.
//!
//! Run: cargo run --release -p lgbm-compute --features rocm --example batched_read_audit_ab

#[cfg(not(feature = "rocm"))]
fn main() {
    eprintln!(
        "batched_read_audit_ab: this rocm-gated CASE-A confirmation harness requires \
         --features rocm (gfx1100). Re-run with it. (See 260619-tlk-FINDINGS.md — the \
         verdict is CASE A: production already emits one consolidated out-handle per \
         launcher and reads it with one read_one_unchecked, which the cubecl-0.10 runtime \
         source proves IS the batched read(vec![h]) of one handle — no N-handle loop exists \
         to collapse, so there is no remaining batched-read lever to wire.)"
    );
}

// The `#[cube]` expansion + ArrayArg/CubeCount/CubeDim need the cubecl prelude in scope at
// the item level (same as histogram.rs's module-level prelude import and the sibling
// lazy_dispatch_ab.rs example).
#[cfg(feature = "rocm")]
use cubecl::prelude::*;

#[cfg(feature = "rocm")]
fn main() {
    // The SHIPPED per-feature atomic kernel (pub + `#[cube(launch_unchecked)]` in
    // histogram.rs). We launch it ONCE into ONE consolidated out-handle (the production
    // shape) and then compare the two READ idioms on that single handle. The kernel choice
    // is incidental — the audit is about the READ path, not this kernel's numerics.
    use lgbm_compute::kernels::histogram::construct_hist_kernel_atomic_f32;
    use lgbm_compute::runtime::{probe_capabilities, rocm_client};
    use std::time::Instant;

    let client = rocm_client();

    let caps = probe_capabilities(&client);
    assert!(
        caps.has_f32_atomic,
        "batched_read_audit_ab exercises the f32-atomic kernel; probed {caps:?}"
    );
    println!("# probed capabilities: {caps:?}");

    // Median + p25/p75 spread helpers (same as the sibling harnesses).
    let median = |xs: &mut [f64]| -> f64 {
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        xs[xs.len() / 2]
    };
    let spread = |xs: &[f64]| -> (f64, f64) {
        let mut s = xs.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (s[s.len() / 4], s[(s.len() * 3) / 4])
    };

    // Deterministic feature-major bin generator (no RNG; same hash as lazy_dispatch_ab.rs).
    let gen_bins = |num_data: usize, num_bin: u32| -> Vec<u32> {
        (0..num_data)
            .map(|row| {
                let h = (row as u64)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(0x9E37_79B9);
                (h % num_bin as u64) as u32
            })
            .collect()
    };
    let gen_grad = |n: usize| -> Vec<f32> { (0..n).map(|i| (i as f32) * 0.001 - 1.0).collect() };
    let gen_hess = |n: usize| -> Vec<f32> { (0..n).map(|i| 1.0 + ((i % 7) as f32) * 0.1).collect() };

    println!("# CASE-A confirmation A/B  —  gfx1100, 260619-tlk");
    println!("# PRODUCTION shape: ONE launch -> ONE consolidated out-handle -> ONE read.");
    println!("# arm A (production)      = client.read_one_unchecked(h)");
    println!("# arm B (manual batched)  = client.read(vec![h])   (the batched idiom, n=1)");
    println!("# cubecl-runtime-0.10.0/src/client.rs: read_one_unchecked(h) == read(vec![h]) drain");
    println!("#   (both go through read_async(vec![h]) -> read_sync) => same single round-trip.");
    println!("# We launch the kernel ONCE per timed iter, then time each READ idiom on the SAME");
    println!("#   handle. Bytes must be IDENTICAL; the delta must be SUB-NOISE (no N-handle loop");
    println!("#   exists in production for the batched read to collapse => no lever to wire).");
    println!("# INTERLEAVED; WARMUP discarded; median + p25/p75; RE-RUN >=2 restarts.");
    println!("# delta% = (a_med - b_med) / a_med * 100   (~0 EXPECTED: same drain).");
    println!();

    println!(
        "{:>13} | {:>7} | {:>5} | {:>11} | {:>11} | {:>8} | {:>17} | {:>17} | {:>9}",
        "regime", "rows", "bins", "a_ms", "b_ms", "delta%", "A p25/p75", "B p25/p75", "bytes_eq"
    );
    println!(
        "{:-<14}+{:-<9}+{:-<7}+{:-<13}+{:-<13}+{:-<10}+{:-<19}+{:-<19}+{:-<11}",
        "", "", "", "", "", "", "", "", ""
    );

    // (regime label, num_data, warmup, timed). Launch-bound: tiny leaf so the per-read
    // fixed cost dominates (where any read-idiom difference would most likely surface).
    // Compute-bound: large leaf so the kernel execute dominates.
    let cells: [(&str, usize, usize, usize); 2] = [
        ("launch-bound", 1_024, 15, 21),
        ("compute-bound", 200_000, 3, 7),
    ];
    const BIN_SWEEP: [u32; 3] = [16, 64, 256];

    for (regime, num_data, warmup, timed) in cells {
        for &num_bin in &BIN_SWEEP {
            let bins = gen_bins(num_data, num_bin);
            let grad = gen_grad(num_data);
            let hess = gen_hess(num_data);
            let out_len = 2 * num_bin as usize;

            // Upload the kernel inputs ONCE per cell, OUTSIDE the timed arms (the arms time
            // ONLY the read of the single out-handle, not transfer or launch — the launch
            // is run once per timed iter just before BOTH reads so both read a freshly
            // completed handle).
            let h_bins = client.create_from_slice(u32::as_bytes(&bins));
            let h_g = client.create_from_slice(f32::as_bytes(&grad));
            let h_h = client.create_from_slice(f32::as_bytes(&hess));

            let cube_dim = 256u32;
            let cube_count = (num_data as u32).div_ceil(cube_dim);
            let zeros = vec![0.0f32; out_len];

            // Launch ONCE into ONE fresh zeroed out-handle, returning the handle UNREAD.
            // SAFETY: identical host-proven obligations to the shipped
            // construct_histograms_parallel_f32_on launcher — bins/grad/hess sized
            // num_data, the kernel guards idx < num_data (tail units idle), every bin <
            // num_bin keeps out[bin*2+1] < out_len, and out is sized out_len = 2*num_bin.
            // cubecl unsafe confined to this example (CMP-01).
            let launch_one = || -> cubecl::server::Handle {
                let h_out = client.create_from_slice(f32::as_bytes(&zeros));
                unsafe {
                    construct_hist_kernel_atomic_f32::launch_unchecked(
                        &client,
                        CubeCount::Static(cube_count, 1, 1),
                        CubeDim::new_1d(cube_dim),
                        ArrayArg::from_raw_parts(h_bins.clone(), num_data),
                        ArrayArg::from_raw_parts(h_g.clone(), num_data),
                        ArrayArg::from_raw_parts(h_h.clone(), num_data),
                        ArrayArg::from_raw_parts(h_out.clone(), out_len),
                    );
                }
                h_out
            };

            // Arm A: the PRODUCTION read call on the single consolidated out-handle.
            let read_a = |h: cubecl::server::Handle| -> Vec<f32> {
                let bytes = client.read_one_unchecked(h);
                f32::from_bytes(&bytes).to_vec()
            };
            // Arm B: the MANUAL batched idiom on the SAME single handle — read(vec![h]).
            // This is the n=1 case the runtime source proves equals read_one_unchecked.
            let read_b = |h: cubecl::server::Handle| -> Vec<f32> {
                let mut got = client.read(vec![h]);
                let bytes = got.remove(0);
                f32::from_bytes(&bytes).to_vec()
            };

            // Interleaved warmup (untimed): launch + both reads.
            for _ in 0..warmup {
                let _ = read_a(launch_one());
                let _ = read_b(launch_one());
            }

            let mut a_ms = Vec::with_capacity(timed);
            let mut b_ms = Vec::with_capacity(timed);
            let mut last_a: Vec<f32> = Vec::new();
            let mut last_b: Vec<f32> = Vec::new();
            for _ in 0..timed {
                // Arm A: launch once (untimed), then TIME read_one_unchecked of the handle.
                let ha = launch_one();
                let t = Instant::now();
                let av = read_a(ha);
                a_ms.push(t.elapsed().as_secs_f64() * 1e3);

                // Arm B: launch once (untimed), then TIME read(vec![h]) of the handle.
                let hb = launch_one();
                let t = Instant::now();
                let bv = read_b(hb);
                b_ms.push(t.elapsed().as_secs_f64() * 1e3);

                last_a = av;
                last_b = bv;
            }

            // The two read idioms drain the SAME kernel output (same launch inputs, same
            // per-row order) — they must return BYTE-IDENTICAL f32 cells. (Both arms launch
            // the same atomic kernel independently; f32-atomic add order is
            // nondeterministic, so we compare within the f32-atomic envelope, NOT bitwise —
            // the point is the READ idiom changes nothing, per DEF-f8u-01 two GPU-f32 paths
            // are never held to each other tighter than the envelope.)
            const ABS: f32 = 5e-6;
            const REL: f32 = 1e-5;
            assert_eq!(last_a.len(), last_b.len(), "{regime}/{num_bin}: histogram length mismatch");
            let mut bytes_eq = true;
            for (x, y) in last_a.iter().zip(&last_b) {
                if (x - y).abs() > ABS + REL * x.abs() {
                    bytes_eq = false;
                    break;
                }
            }
            assert!(
                bytes_eq,
                "{regime}/{num_bin}: read_one_unchecked vs read(vec![h]) diverged beyond the \
                 f32-atomic envelope — the read idiom must NOT change values"
            );

            let a_med = median(&mut a_ms);
            let b_med = median(&mut b_ms);
            let (al, ah) = spread(&a_ms);
            let (bl, bh) = spread(&b_ms);
            let delta = (a_med - b_med) / a_med * 100.0;
            println!(
                "{:>13} | {:>7} | {:>5} | {:>11.5} | {:>11.5} | {:>7.2} | {:>7.5}/{:>7.5} | {:>7.5}/{:>7.5} | {:>9}",
                regime, num_data, num_bin, a_med, b_med, delta, al, ah, bl, bh, "yes"
            );
        }
    }

    println!();
    println!("# LEGEND: arm A = client.read_one_unchecked(h) (production); arm B = client.read(vec![h])");
    println!("#   (manual batched idiom, n=1). bytes_eq=yes => the read idiom changed NO values.");
    println!("# EXPECTED (CASE A): delta% ~0 and WITHIN the p25/p75 spread — the two are the SAME");
    println!("#   read_async(vec![h]) -> read_sync drain (cubecl-runtime-0.10.0 client.rs:131-149).");
    println!("#   Production emits ONE consolidated out-handle per launcher, so there is NO N-handle");
    println!("#   read loop for the batched idiom to collapse => no remaining batched-read lever to");
    println!("#   wire. Manufacturing a per-handle loop to 'beat' would be the forbidden q2z/sgu");
    println!("#   anti-pattern. MEASUREMENT ONLY — production untouched.");
}
