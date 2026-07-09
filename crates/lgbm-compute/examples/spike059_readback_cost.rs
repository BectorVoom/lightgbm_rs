//! Spike 059 (Kaggle deliverable) — readback-cost floor on REAL CUDA.
//!
//! The spoofed 8-CU APU cannot time PCIe blocking-readback latency — the exact
//! suspected long-pole of the on-device A/B slowdown ([[rocm-gfx1100-available]]).
//! Only real discrete NVIDIA (Kaggle T4/P100) can. This micro-bench isolates the
//! ONE cost the four failed A/B phases kept circling: a per-leaf BLOCKING
//! device→host readback between dependent kernel launches.
//!
//! Two patterns, identical device work, differ ONLY in sync discipline:
//!
//!   SYNC  — N iterations, each: launch a tiny kernel THEN `read_one_unchecked`
//!           (blocking readback). Mirrors `grow_driver.rs` scan/partition/split
//!           per-leaf `bump_sync()` (Phase 26: 6044-9044 readbacks/grow).
//!   CHAIN — the SAME N launches chained on device (handle handoff), ONE terminal
//!           read. Mirrors the reference §1 "handful of scalars per iteration".
//!
//! The SYNC/CHAIN wall-clock ratio × the ~9000 per-grow readback count bounds the
//! floor. On a launch-latency-bound device the ratio is large; if it's ~1 the
//! readback floor is NOT the long-pole and the on-device thesis must move again.
//!
//! Run on Kaggle (real CUDA): `cargo run --release -p lgbm-compute \
//!   --features cuda --example spike059_readback_cost`
//! Locally it falls back to the cpu runtime (ratio is NOT meaningful there — the
//! point is the Kaggle number).

use cubecl::prelude::*;
use std::time::Instant;

#[cube(launch)]
fn tiny_kernel(input: &Array<f32>, output: &mut Array<f32>, n: u32) {
    let i = ABSOLUTE_POS;
    if i < n as usize {
        output[i] = input[i] + 1.0;
    }
}

fn bench<R: cubecl::Runtime>(client: &ComputeClient<R>, backend: &str) {
    const N: usize = 256; // small = launch/sync-latency bound, like a per-leaf scan
    const ITERS: usize = 4000; // ~ the per-grow readback count (Phase 26: 6-9k)
    let bd = 256u32;
    let cubes = (N as u32).div_ceil(bd);
    let init = vec![0.0f32; N];

    // Warm up (JIT + first-launch alloc).
    let h0 = client.create_from_slice(f32::as_bytes(&init));
    let hw = client.empty(N * 4);
    unsafe {
        tiny_kernel::launch::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(bd),
            ArrayArg::from_raw_parts(h0.clone(), N),
            ArrayArg::from_raw_parts(hw.clone(), N),
            N as u32,
        );
    }
    let _ = client.read_one_unchecked(hw);

    // ---- SYNC: launch + blocking readback each iteration ----
    let t = Instant::now();
    let mut h = client.create_from_slice(f32::as_bytes(&init));
    for _ in 0..ITERS {
        let h_out = client.empty(N * 4);
        unsafe {
            tiny_kernel::launch::<R>(
                client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(bd),
                ArrayArg::from_raw_parts(h.clone(), N),
                ArrayArg::from_raw_parts(h_out.clone(), N),
                N as u32,
            );
        }
        let _bytes = client.read_one_unchecked(h_out.clone()); // BLOCKING readback per iter
        h = h_out;
    }
    let sync_ms = t.elapsed().as_secs_f64() * 1e3;

    // ---- CHAIN: same launches, ONE terminal read ----
    let t = Instant::now();
    let mut h = client.create_from_slice(f32::as_bytes(&init));
    for _ in 0..ITERS {
        let h_out = client.empty(N * 4);
        unsafe {
            tiny_kernel::launch::<R>(
                client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(bd),
                ArrayArg::from_raw_parts(h.clone(), N),
                ArrayArg::from_raw_parts(h_out.clone(), N),
                N as u32,
            );
        }
        h = h_out; // device handoff, NO read
    }
    let _bytes = client.read_one_unchecked(h); // ONE terminal read
    let chain_ms = t.elapsed().as_secs_f64() * 1e3;

    let per_readback_us = (sync_ms - chain_ms) / ITERS as f64 * 1e3;
    println!("=== spike059 readback-cost on [{backend}] ===");
    println!("  ITERS={ITERS}, N={N} floats/launch");
    println!("  SYNC  (launch+readback each): {sync_ms:9.2} ms");
    println!("  CHAIN (one terminal read):    {chain_ms:9.2} ms");
    println!("  ratio SYNC/CHAIN:             {:9.2}x", sync_ms / chain_ms);
    println!("  isolated cost per blocking readback: {per_readback_us:7.2} us");
    println!(
        "  => projected floor for a 9000-readback grow: {:.1} ms of pure sync",
        per_readback_us * 9000.0 / 1e3
    );
}

fn main() {
    #[cfg(feature = "cuda")]
    {
        let client = cubecl::cuda::CudaRuntime::client(&cubecl::cuda::CudaDevice::new(0));
        bench(&client, "cuda(real NVIDIA)");
        return;
    }
    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    {
        let client = cubecl::hip::HipRuntime::client(&cubecl::hip::AmdDevice::new(0));
        bench(&client, "hip(APU — NOT meaningful, needs real GPU)");
        return;
    }
    #[cfg(not(any(feature = "cuda", feature = "rocm")))]
    {
        let client = cubecl::cpu::CpuRuntime::client(&cubecl::cpu::CpuDevice);
        bench(&client, "cpu(fallback — ratio NOT meaningful, run on Kaggle cuda)");
    }
}
