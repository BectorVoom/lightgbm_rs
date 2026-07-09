//! Spike 075 — per-tree cost of the Phase-29 unconditional serial f64 root fold.
//!
//! Phase 29 (OCX-01) deleted the biased serial-f32 root sum and made
//! `root_grad_hess_sum_device` an UNCONDITIONAL serial ascending f64 fold on a
//! `CubeCount(1)`/`CubeDim(1)` single lane (bit-exactness vs the host fold requires the
//! exact ascending order). Spike-052's rule says NO f64 hot loops on consumer NVIDIA
//! (1/32 f64 throughput); spike-055's lesson says a parity-anchor-shaped serial kernel
//! leaking into production is exactly how Phase 23 failed. This spike bounds the cost:
//! a single-lane 500k-iteration dependent chain could be ~4ms/tree (register-held
//! accumulator) or ~150ms/tree (global-memory RMW round trip per iteration) — the low
//! end is a rounding error at a ~10s/100-tree train, the high end would DOMINATE the
//! whole on-device A/B gap. Measure, don't model.
//!
//! What it measures (median [p25..p75] over REPS, first WARMUP discarded, per size):
//!   * DEVICE — `root_grad_hess_sum_device` on pre-uploaded handles: one launch + the
//!     blocking 16-byte readback. This IS the per-tree production unit (called once per
//!     grow on the on-device arm; the resident buffers are already device-side).
//!   * HOST   — the same ascending f64 fold over the host slices (the CpuBackend /
//!     host-cuda-arm equivalent, `root_grad_hess_fold`'s exact order, inlined here
//!     because that fn is pub(crate)).
//!   * floor  — the n=1_000 DEVICE point ≈ launch + blocking-sync overhead; the
//!     size-to-size slope (ns/row) isolates the kernel's serial-loop cost from it.
//!   * parity — the device (f64, f64) must be BIT-EXACT (to_bits) vs the host fold at
//!     every size (same adds, same order — any mismatch is a defect, not noise).
//!
//! Run (local hip lane — sign/order valid, absolute numbers APU-confounded):
//!   cargo run --release -p lgbm-compute --features rocm --example spike075_rootfold_cost
//! Run (real CUDA, e.g. Kaggle):
//!   cargo run --release -p lgbm-compute --features cuda --example spike075_rootfold_cost
//! Optional env: SPIKE075_SIZES="1000,500000" SPIKE075_REPS=11

use std::time::Instant;

use cubecl::CubeElement;
use cubecl::prelude::*;
use lgbm_compute::kernels::best_split::root_grad_hess_sum_device;

/// VARIANT under test: the IDENTICAL serial ascending fold, but accumulating in
/// loop-carried LOCAL variables (registers) with a SINGLE `out` store at the end,
/// instead of the production kernel's per-iteration `out[0]`/`out[1]` global-array
/// RMW. Same f64 adds in the same ascending order ⇒ still BIT-EXACT vs the host
/// fold; only the accumulator's residence changes. If the production slope
/// (~85ns/row on hip) is the global-memory dependent chain, this collapses it.
/// (Conventions gotcha honored: loop-carried mutables init from a plain literal.)
#[cube(launch_unchecked)]
fn root_grad_hess_kernel_f64_reg(
    grad: &Array<f32>,
    hess: &Array<f32>,
    out: &mut Array<f64>,
    n: u32,
) {
    let mut g = 0.0f64;
    let mut h = 0.0f64;
    let nn = n as usize;
    for i in 0..nn {
        g += f64::cast_from(grad[i]);
        h += f64::cast_from(hess[i]);
    }
    out[0] = g;
    out[1] = h;
}

/// Launch wrapper for the register variant — mirrors `root_grad_hess_sum_device`
/// byte-for-byte (Static(1,1,1) single-owner geometry + blocking 16-byte readback).
fn root_fold_reg_variant<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    grad: &cubecl::server::Handle,
    hess: &cubecl::server::Handle,
    num_data: usize,
) -> (f64, f64) {
    let h_out = client.empty(2 * core::mem::size_of::<f64>());
    unsafe {
        root_grad_hess_kernel_f64_reg::launch_unchecked(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(grad.clone(), num_data),
            ArrayArg::from_raw_parts(hess.clone(), num_data),
            ArrayArg::from_raw_parts(h_out.clone(), 2),
            num_data as u32,
        );
    }
    let bytes = client.read_one_unchecked(h_out);
    let cells = f64::from_bytes(&bytes);
    (cells[0], cells[1])
}

const DEFAULT_SIZES: &[usize] = &[1_000, 10_000, 100_000, 500_000, 1_000_000, 2_000_000];
const DEFAULT_REPS: usize = 11;
const WARMUP: usize = 2;

/// Deterministic inline LCG (spike conventions: no `rand` dep). Yields f32 in [lo, hi).
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
        // Numerical Recipes LCG constants; take the top 24 bits for the mantissa.
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (self.0 >> 40) as f32 / (1u32 << 24) as f32;
        lo + u * (hi - lo)
    }
}

/// The host serial ascending f64 fold — the exact `root_grad_hess_fold` order
/// (`pub(crate)`, so replicated verbatim: widen each f32, add ascending).
fn host_fold(gradients: &[f32], hessians: &[f32]) -> (f64, f64) {
    let mut g = 0.0f64;
    let mut h = 0.0f64;
    for i in 0..gradients.len() {
        g += f64::from(gradients[i]);
        h += f64::from(hessians[i]);
    }
    (g, h)
}

fn median_ns(mut v: Vec<u128>) -> (u128, u128, u128) {
    v.sort_unstable();
    let n = v.len();
    (v[n / 4], v[n / 2], v[(3 * n) / 4])
}

fn bench<R: cubecl::Runtime>(client: &cubecl::prelude::ComputeClient<R>, backend: &str) {
    let sizes: Vec<usize> = std::env::var("SPIKE075_SIZES")
        .ok()
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| DEFAULT_SIZES.to_vec());
    let reps: usize = std::env::var("SPIKE075_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_REPS);

    println!("backend={backend} reps={reps} (+{WARMUP} warmup) sizes={sizes:?}");
    println!(
        "{:>10} | {:>12} {:>12} {:>12} | {:>12} {:>8} | {:>12} | {:>9} | parity",
        "rows", "dev p25(us)", "dev med(us)", "dev p75(us)", "reg med(us)", "dev/reg",
        "host med(us)", "ns/row"
    );

    let mut prev: Option<(usize, u128)> = None;
    for &n in &sizes {
        // Deterministic binary-objective-shaped data: grad in [-1,1), hess in (0.09, 0.25].
        let mut lcg = Lcg(0x5eed_0075 ^ n as u64);
        let grad: Vec<f32> = (0..n).map(|_| lcg.next_f32(-1.0, 1.0)).collect();
        let hess: Vec<f32> = (0..n).map(|_| lcg.next_f32(0.09, 0.25)).collect();

        // Upload ONCE (production reads RESIDENT buffers; upload is not part of the unit).
        let hg = client.create_from_slice(f32::as_bytes(&grad));
        let hh = client.create_from_slice(f32::as_bytes(&hess));

        // Device: launch + blocking 16-byte readback per rep — the per-tree unit.
        // INTERLEAVE production vs the register-accumulator variant per rep (GPU A/B
        // discipline: cancels thermal/scheduler drift within the round).
        let mut dev_ns: Vec<u128> = Vec::with_capacity(reps);
        let mut reg_ns: Vec<u128> = Vec::with_capacity(reps);
        let mut dev_result = (0.0f64, 0.0f64);
        let mut reg_result = (0.0f64, 0.0f64);
        for r in 0..(reps + WARMUP) {
            let t0 = Instant::now();
            let out = root_grad_hess_sum_device(client, &hg, &hh, n).expect("device fold");
            let dt = t0.elapsed().as_nanos();
            dev_result = out;
            let t1 = Instant::now();
            let outr = root_fold_reg_variant(client, &hg, &hh, n);
            let dtr = t1.elapsed().as_nanos();
            reg_result = outr;
            if r >= WARMUP {
                dev_ns.push(dt);
                reg_ns.push(dtr);
            }
        }
        let (p25, med, p75) = median_ns(dev_ns);
        let (_, reg_med, _) = median_ns(reg_ns);

        // Host fold (same reps discipline; it is deterministic so timing only).
        let mut host_ns: Vec<u128> = Vec::with_capacity(reps);
        let mut host_result = (0.0f64, 0.0f64);
        for r in 0..(reps + WARMUP) {
            let t0 = Instant::now();
            let out = host_fold(std::hint::black_box(&grad), std::hint::black_box(&hess));
            let dt = t0.elapsed().as_nanos();
            host_result = std::hint::black_box(out);
            if r >= WARMUP {
                host_ns.push(dt);
            }
        }
        let (_, host_med, _) = median_ns(host_ns);

        // Parity: BIT-EXACT (same f64 adds in the same ascending order) — any mismatch
        // is a defect, not float noise. The register variant is held to the SAME bar.
        let bit_eq = dev_result.0.to_bits() == host_result.0.to_bits()
            && dev_result.1.to_bits() == host_result.1.to_bits();
        let reg_bit_eq = reg_result.0.to_bits() == host_result.0.to_bits()
            && reg_result.1.to_bits() == host_result.1.to_bits();

        // Slope vs the previous size isolates the serial-loop cost from the launch+sync floor.
        let slope = prev
            .map(|(pn, pmed)| (med.saturating_sub(pmed)) as f64 / (n - pn).max(1) as f64)
            .unwrap_or(f64::NAN);
        prev = Some((n, med));

        println!(
            "{:>10} | {:>12.1} {:>12.1} {:>12.1} | {:>12.1} {:>7.2}x | {:>12.1} | {:>9.3} | {} / reg:{}",
            n,
            p25 as f64 / 1e3,
            med as f64 / 1e3,
            p75 as f64 / 1e3,
            reg_med as f64 / 1e3,
            med as f64 / reg_med.max(1) as f64,
            host_med as f64 / 1e3,
            slope,
            if bit_eq { "BIT-EXACT" } else { "MISMATCH <-- DEFECT" },
            if reg_bit_eq { "BIT-EXACT" } else { "MISMATCH <-- DEFECT" }
        );
    }

    println!();
    println!("interpretation: per-tree cost = the 500000-row DEVICE median (launch+fold+sync);");
    println!("x100 trees = the whole-train contribution. Compare against the ~2-3s A/B gap");
    println!("(069-071 ledger) and the host column (what the host arms pay for the same sum).");
}

fn main() {
    #[cfg(feature = "cuda")]
    {
        let client = lgbm_compute::runtime::cuda_client();
        bench::<lgbm_compute::runtime::CudaRuntime>(&client, "cuda");
        return;
    }
    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    {
        let client = lgbm_compute::runtime::rocm_client();
        bench::<lgbm_compute::runtime::RocmRuntime>(&client, "hip");
        return;
    }
    #[cfg(not(any(feature = "cuda", feature = "rocm")))]
    {
        // cpu runtime fallback: sign/order NOT meaningful for the GPU question (and
        // cubecl-cpu per-launch overhead is pathological, spike-059) — build with a GPU
        // feature for a real answer.
        let client = lgbm_compute::runtime::cpu_client();
        bench::<cubecl::cpu::CpuRuntime>(&client, "cpu (NOT meaningful for the GPU question)");
    }
}
