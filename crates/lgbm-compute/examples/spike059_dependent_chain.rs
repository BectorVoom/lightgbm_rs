//! Spike 059 — cubecl-dependent-chain-no-sync.
//!
//! Feasibility question (make-or-break for the on-device CUDA learner, 4 A/B fails):
//! can cubecl 0.10 grow a *data-dependent* kernel chain (build→scan→argmax→partition
//! →next-build) with ZERO per-leaf blocking device→host readback?
//!
//! The per-leaf `bump_sync()` readbacks in `grow_driver.rs` (scan SplitInfo, tree
//! right_leaf_index, partition split-point) are the pinned root cause of the on-device
//! slowdown (Phase 26: 6044/9044 blocking readbacks/grow vs host-cuda's 0). This spike
//! isolates the cubecl *programming model* to see whether those readbacks are REMOVABLE
//! or STRUCTURALLY FORCED by the runtime.
//!
//! Three routes are tested, all against the always-available `cpu` runtime (the
//! CUDA/HIP claims are cross-checked against the runtime SOURCE, quoted in the README):
//!
//!   A. STATIC dependent chain — kernel B reads kernel A's device output, chained N
//!      times, ONE read at the end. Proves dependent chaining needs no per-step sync.
//!   B. DYNAMIC dispatch — a kernel writes a device `[x,y,z]` count buffer; the next
//!      launch uses `CubeCount::Dynamic(count)`. Proves the count is honored *and*
//!      (per source) that the launcher block_on-reads the buffer = a HIDDEN host sync.
//!   C. FIXED over-launch + in-kernel device guard — static max geometry, the real
//!      row count read from a DEVICE buffer inside the kernel. Same data-dependent
//!      effect as B with NO extra host read = the only viable escape route.
//!
//! Run: `cargo run -p lgbm-compute --example spike059_dependent_chain`

use cubecl::prelude::*;

type R = cubecl::cpu::CpuRuntime;

fn client() -> ComputeClient<R> {
    cubecl::cpu::CpuRuntime::client(&cubecl::cpu::CpuDevice)
}

// ---------------------------------------------------------------------------
// Route A — a trivial dependent kernel: out[i] = in[i] + add (device scalar in a
// 1-elem array so it too stays resident). Chaining it reads each prior output.
// ---------------------------------------------------------------------------
#[cube(launch)]
fn add_kernel(input: &Array<f32>, add: &Array<f32>, output: &mut Array<f32>, n: u32) {
    let i = ABSOLUTE_POS;
    if i < n as usize {
        output[i] = input[i] + add[0];
    }
}

// ---------------------------------------------------------------------------
// Route B/C helper — write a device [x,y,z] cube-count buffer derived from a
// device-resident element count `count[0]`, block dim `bd`. This is the "compute
// the next launch geometry ON DEVICE" step a data-dependent grow loop needs.
// ---------------------------------------------------------------------------
#[cube(launch)]
fn make_cube_count(count: &Array<u32>, bd: u32, xyz: &mut Array<u32>) {
    if ABSOLUTE_POS == 0 {
        // ceil(count / bd) cubes in x, 1×1 in y,z.
        xyz[0] = (count[0] + bd - 1) / bd;
        xyz[1] = 1;
        xyz[2] = 1;
    }
}

// ---------------------------------------------------------------------------
// Route C kernel — fixed over-launch: dispatched with a host MAX cube count, but
// the ACTUAL work bound `count[0]` is read from a DEVICE buffer inside the kernel.
// Idle lanes above the real count guard out. No host readback of the count.
// ---------------------------------------------------------------------------
#[cube(launch)]
fn guarded_fill(count: &Array<u32>, output: &mut Array<f32>) {
    let i = ABSOLUTE_POS;
    if i < count[0] as usize {
        output[i] = 1.0;
    }
}

fn main() {
    let c = client();
    println!("=== Spike 059 — cubecl dependent-chain / readback-floor feasibility ===");
    println!("runtime: cubecl-cpu (CUDA/HIP behavior cross-checked vs runtime source)\n");

    // ---- Route A: static dependent chain, N launches, ONE read ----------------
    const N: usize = 8;
    const CHAIN: usize = 6;
    let bd = 8u32;

    let mut host_reads = 0usize;
    let init = vec![0.0f32; N];
    let mut h_buf = c.create_from_slice(f32::as_bytes(&init));
    let h_add = c.create_from_slice(f32::as_bytes(&[1.0f32; 1]));

    for step in 0..CHAIN {
        eprintln!("  [A] launch {step}/{CHAIN} ...");
        // ping into a fresh output handle, then the output becomes the next input.
        let h_out = c.empty(N * core::mem::size_of::<f32>());
        unsafe {
            add_kernel::launch::<R>(
                &c,
                CubeCount::Static((N as u32).div_ceil(bd), 1, 1),
                CubeDim::new_1d(bd),
                ArrayArg::from_raw_parts(h_buf.clone(), N),
                ArrayArg::from_raw_parts(h_add.clone(), 1),
                ArrayArg::from_raw_parts(h_out.clone(), N),
                N as u32,
            );
        }
        h_buf = h_out; // <-- device handle handoff, NO read between iterations
    }
    let bytes = c.read_one_unchecked(h_buf);
    host_reads += 1; // the ONE terminal read
    let got = f32::from_bytes(&bytes);
    let ok_a = got.iter().all(|&v| (v - CHAIN as f32).abs() < 1e-4);
    println!(
        "[A] static dependent chain: {CHAIN} launches, out={:.1} (expect {CHAIN}.0) -> {}",
        got[0],
        if ok_a { "PASS" } else { "FAIL" }
    );
    println!("    host reads for the whole {CHAIN}-deep chain = {host_reads}  (per-step sync NOT required)\n");

    // Shared device-resident "leaf row count" for routes C and B — a value the host
    // must NOT read to launch the next data-dependent kernel.
    let real_count = 5u32;
    let h_count = c.create_from_slice(u32::as_bytes(&[real_count]));
    let cap = 64usize;

    // ---- Route C: fixed over-launch + in-kernel device guard -------------------
    // Same data-dependent effect (fill `count[0]` lanes) with STATIC host geometry
    // (a fixed MAX) and the bound read from a DEVICE buffer. No host readback.
    let max_cubes = (cap as u32).div_ceil(bd); // host-fixed MAX, count-independent
    let h_out_c = c.empty(cap * core::mem::size_of::<f32>());
    unsafe {
        guarded_fill::launch::<R>(
            &c,
            CubeCount::Static(max_cubes, 1, 1), // <-- static, no device read to launch
            CubeDim::new_1d(bd),
            ArrayArg::from_raw_parts(h_count.clone(), 1),
            ArrayArg::from_raw_parts(h_out_c.clone(), cap),
        );
    }
    let bytes_c = c.read_one_unchecked(h_out_c);
    let out_c = f32::from_bytes(&bytes_c);
    let filled_c = out_c.iter().filter(|&&v| v == 1.0).count();
    let ok_c = filled_c == real_count as usize;
    println!(
        "[C] fixed over-launch + device guard: {filled_c} lanes filled (expect {real_count}) -> {}",
        if ok_c { "PASS" } else { "FAIL" }
    );
    println!("    static geometry + in-kernel `if i < count[0]` reads the bound ON DEVICE.");
    println!("    NO host readback of the count to launch. This is the only route that removes the sync.\n");

    // ---- Route B: DYNAMIC dispatch (gated — HANGS on cubecl-cpu) ----------------
    // A prior kernel computes the [x,y,z] count on device; the next launch consumes
    // it via CubeCount::Dynamic. On cubecl-cpu this HANGS (Dynamic dispatch spins in
    // the stream flush+read) — itself corroboration that Dynamic is not a clean path.
    // On CUDA/HIP the launcher does `future::block_on(read_async(buffer))` — a HIDDEN
    // blocking device->host readback. Either way Dynamic RELOCATES the per-leaf sync,
    // it does not remove it. Gated behind SPIKE059_DYNAMIC=1 so the default run finishes cleanly.
    if std::env::var("SPIKE059_DYNAMIC").is_ok() {
        let h_xyz = c.empty(3 * core::mem::size_of::<u32>());
        unsafe {
            make_cube_count::launch::<R>(
                &c,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                ArrayArg::from_raw_parts(h_count.clone(), 1),
                bd,
                ArrayArg::from_raw_parts(h_xyz.clone(), 3),
            );
        }
        let h_out_b = c.empty(cap * core::mem::size_of::<f32>());
        unsafe {
            guarded_fill::launch::<R>(
                &c,
                CubeCount::Dynamic(h_xyz.clone().binding()),
                CubeDim::new_1d(bd),
                ArrayArg::from_raw_parts(h_count.clone(), 1),
                ArrayArg::from_raw_parts(h_out_b.clone(), cap),
            );
        }
        let bytes_b = c.read_one_unchecked(h_out_b);
        let filled = f32::from_bytes(&bytes_b).iter().filter(|&&v| v == 1.0).count();
        println!("[B] Dynamic dispatch honored device count: {filled} lanes filled (source: CUDA/HIP block_on-reads the count).\n");
    } else {
        println!("[B] Dynamic dispatch: SKIPPED (hangs on cubecl-cpu; set SPIKE059_DYNAMIC=1 to force).");
        println!("    Source-proven on ALL backends: CubeCount::Dynamic reads the count buffer host-side");
        println!("    (cuda/hip: future::block_on(read_async); cpu: stream flush + resource.read()).");
        println!("    => Dynamic does NOT remove the per-leaf sync on CUDA/HIP; it RELOCATES it.\n");
    }

    println!("=== VERDICT ===");
    println!("Dependent chaining (A) + device-guarded static over-launch (C) can express a");
    println!("grow iteration with NO per-step host readback. Dynamic dispatch (B) is a TRAP on");
    println!("CUDA/HIP (hidden block_on read). The readback floor is REMOVABLE only by keeping");
    println!("the frontier/geometry device-resident with STATIC over-launched launches.");
}
