//! Spike 029 — GPU narrow-upload fuse for the per-split partition routing (rocm-gated).
//!
//! On the ROCm partition path, `DataPartition::split` host-gathers a leaf's bins into a
//! `Vec<u32>` (WIDENED to u32 even on a U8 column), then `RocmBackend::data_partition`
//! uploads that u32 buffer to the device, routes per-row, and reads back. This spike A/Bs
//! that against a NARROW fuse: host-gather to native u8, upload u8 (4× fewer bytes), route
//! with a u8-reading kernel.
//!
//! TWO components, measured separately:
//!   (A) HOST gather — u32-widen vs u8-native (real CPU cost, NOT APU-confounded; this is
//!       the spike-027 mechanism applied to the rocm path's host prep).
//!   (B) UPLOAD + kernel + readback — the device transfer. On THIS box the "GPU" is a
//!       spoofed 8-CU APU with SHARED DDR5, so host→device "upload" is a same-memory copy:
//!       the transfer-volume win is expected APU-MASKED here, its real value being on a
//!       discrete gfx110x with a PCIe bus. SIGN-judged, not magnitude.
//!
//! Correctness: the u8 and u32 routing kernels must produce a byte-identical `route[]`.
//!
//! Run: `cargo run -p lgbm-compute --example spike029_gpu_narrow_upload_ab --release --features rocm`

#[cfg(not(feature = "rocm"))]
fn main() {
    eprintln!("spike-029 is rocm-only — re-run with `--features rocm`.");
}

#[cfg(feature = "rocm")]
use cubecl::prelude::*;

/// Route kernel reading U32 bins (the current path's upload type).
#[cfg(feature = "rocm")]
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn route_u32_kernel(
    bins: &Array<u32>,
    route: &mut Array<u32>,
    min_bin: i32,
    max_bin: i32,
    threshold: i32,
    most_freq_bin: i32,
) {
    let i = ABSOLUTE_POS;
    if i < bins.len() {
        let mut th = threshold + min_bin;
        if most_freq_bin == 0 {
            th -= 1;
        }
        let default_to_right = most_freq_bin > threshold;
        let bin = bins[i] as i32;
        let is_default = bin < min_bin || bin > max_bin;
        let gt = bin > th;
        let go_right = select(is_default, default_to_right, gt);
        route[i] = select(go_right, 1u32, 0u32);
    }
}

/// Route kernel reading NATIVE U8 bins (the narrow fuse). Value-identical routing; only the
/// input storage width differs (u8 → 4× smaller upload).
#[cfg(feature = "rocm")]
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn route_u8_kernel(
    bins: &Array<u8>,
    route: &mut Array<u32>,
    min_bin: i32,
    max_bin: i32,
    threshold: i32,
    most_freq_bin: i32,
) {
    let i = ABSOLUTE_POS;
    if i < bins.len() {
        let mut th = threshold + min_bin;
        if most_freq_bin == 0 {
            th -= 1;
        }
        let default_to_right = most_freq_bin > threshold;
        let bin = u32::cast_from(bins[i]) as i32;
        let is_default = bin < min_bin || bin > max_bin;
        let gt = bin > th;
        let go_right = select(is_default, default_to_right, gt);
        route[i] = select(go_right, 1u32, 0u32);
    }
}

#[cfg(feature = "rocm")]
fn lcg(seed: u64) -> impl FnMut() -> u32 {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 33) as u32
    }
}

#[cfg(feature = "rocm")]
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[cfg(feature = "rocm")]
fn main() {
    use std::time::Instant;
    use lgbm_compute::runtime::rocm_client;

    let run = std::env::var("LGBM_SPIKE_RUN").unwrap_or_else(|_| "1".into());
    let client = rocm_client();

    let num_bin = 64u32;
    let (min_bin, max_bin, threshold, most_freq_bin) = (0i32, 63i32, 31i32, 0i32);
    let reps = 30;

    println!("# spike-029 GPU narrow-upload fuse (rocm, gfx1152 8-CU APU / shared DDR5)  run={run}");
    println!("# host: u32-widen gather vs u8-native gather.  dev: upload+route+readback, u32 vs u8.");
    println!("# ratio>1 ⇒ narrow FASTER. APU shared-mem ⇒ judge SIGN; transfer win is discrete-GPU value.");
    println!(
        "{:>10} {:>13} {:>13} {:>9} | {:>12} {:>12} {:>9} {:>6}",
        "rows", "host_u32(ms)", "host_u8(ms)", "h_ratio", "dev_u32(ms)", "dev_u8(ms)", "d_ratio", "parity"
    );

    // Full-dataset U8 bin column; scattered leaf = shuffled row ids (random gather).
    for &rows in &[100_000usize, 500_000, 1_000_000, 4_000_000] {
        let mut next = lcg(0xC0FFEE ^ rows as u64);
        let col: Vec<u8> = (0..rows).map(|_| (next() % num_bin) as u8).collect();
        let mut leaf: Vec<u32> = (0..rows as u32).collect();
        let mut nx = lcg(0xBEEF ^ rows as u64);
        for i in (1..rows).rev() {
            let j = (nx() as usize) % (i + 1);
            leaf.swap(i, j);
        }
        let n = rows;

        // ---- host gather variants (CPU, real hardware) ----
        let gather_u32 = |leaf: &[u32], col: &[u8]| -> Vec<u32> {
            leaf.iter().map(|&r| col[r as usize] as u32).collect()
        };
        let gather_u8 = |leaf: &[u32], col: &[u8]| -> Vec<u8> {
            leaf.iter().map(|&r| col[r as usize]).collect()
        };
        let bins_u32 = gather_u32(&leaf, &col);
        let bins_u8 = gather_u8(&leaf, &col);

        // ---- device round-trip closures ----
        let dev_u32 = |bins: &[u32]| -> Vec<u32> {
            let h = client.create_from_slice(u32::as_bytes(bins));
            let z = vec![0u32; n];
            let hr = client.create_from_slice(u32::as_bytes(&z));
            let cd = 256u32;
            unsafe {
                route_u32_kernel::launch(
                    &client,
                    CubeCount::Static((n as u32).div_ceil(cd), 1, 1),
                    CubeDim::new_1d(cd),
                    ArrayArg::from_raw_parts(h, n),
                    ArrayArg::from_raw_parts(hr.clone(), n),
                    min_bin, max_bin, threshold, most_freq_bin,
                );
            }
            u32::from_bytes(&client.read_one_unchecked(hr)).to_vec()
        };
        let dev_u8 = |bins: &[u8]| -> Vec<u32> {
            let h = client.create_from_slice(u8::as_bytes(bins));
            let z = vec![0u32; n];
            let hr = client.create_from_slice(u32::as_bytes(&z));
            let cd = 256u32;
            unsafe {
                route_u8_kernel::launch(
                    &client,
                    CubeCount::Static((n as u32).div_ceil(cd), 1, 1),
                    CubeDim::new_1d(cd),
                    ArrayArg::from_raw_parts(h, n),
                    ArrayArg::from_raw_parts(hr.clone(), n),
                    min_bin, max_bin, threshold, most_freq_bin,
                );
            }
            u32::from_bytes(&client.read_one_unchecked(hr)).to_vec()
        };

        // correctness: routes byte-identical
        let r32 = dev_u32(&bins_u32);
        let r8 = dev_u8(&bins_u8);
        let parity = r32 == r8;

        // warmup
        let _ = dev_u32(&bins_u32);
        let _ = dev_u8(&bins_u8);

        let mut hu32 = Vec::new();
        let mut hu8 = Vec::new();
        let mut du32 = Vec::new();
        let mut du8 = Vec::new();
        for _ in 0..reps {
            let t = Instant::now();
            let b = gather_u32(&leaf, &col);
            hu32.push(t.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box(&b);

            let t = Instant::now();
            let b = gather_u8(&leaf, &col);
            hu8.push(t.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box(&b);

            let t = Instant::now();
            let r = dev_u32(&bins_u32);
            du32.push(t.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box(&r);

            let t = Instant::now();
            let r = dev_u8(&bins_u8);
            du8.push(t.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box(&r);
        }
        let (mh32, mh8) = (median(hu32), median(hu8));
        let (md32, md8) = (median(du32), median(du8));
        println!(
            "{:>10} {:>13.3} {:>13.3} {:>8.2}x | {:>12.3} {:>12.3} {:>8.2}x {:>6}",
            rows, mh32, mh8, mh32 / mh8, md32, md8, md32 / md8, if parity { "OK" } else { "FAIL" }
        );
        assert!(parity, "route parity FAIL at rows={rows}");
    }
}
