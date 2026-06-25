//! Spike 026 — cubecl-cpu scan+scatter `DataPartition::split`, A/B vs serial-native.
//!
//! QUESTION: the production CpuBackend partition (`data_partition_cpu_native`) does
//! a SERIAL, single-threaded stable two-pass gather. The per-row ROUTING is already
//! a parallel `#[cube]` kernel (b141a82) but the expensive part — the stable
//! compaction (the gather that preserves order) — stays serial. Can a **cubecl-cpu
//! scan+scatter kernel** do that compaction faster, BIT-EXACTLY (byte-identical
//! `reordered[]`), using the 16 REAL CPU cores + SIMD?
//!
//! WHY THIS, NOT RAYON: a rayon version was already built + proven bit-exact +
//! benched (quick-260622-ia0) and REVERTED as an end-to-end NULL — but only because
//! it contends with the already-rayon-parallel histogram BUILD on the CPU train
//! path. On the GPU train path the build runs ON THE DEVICE, so the host pool is
//! idle: the contention that nulled rayon is absent. This spike asks the cubecl-cpu
//! variant (the user's unified-kernel preference; SIMD; "same value" = bit-exact).
//!
//! ALGORITHM (bit-exact parallel stable partition, mirrors ia0 + C++
//! `ParallelPartitionRunner` `schedule(static, 512)`):
//!   1. COUNT kernel  — one cube per CHUNK; each chunk counts its LEFT rows.   (parallel)
//!   2. host prefix-sum of per-chunk left-counts → per-chunk left/right write bases. (tiny serial)
//!   3. SCATTER kernel — one cube per chunk; walk rows ASCENDING, scatter each
//!      into [all-left | all-right] at its chunk's disjoint base.              (parallel)
//! Within a chunk the walk is sequential ⇒ stable order preserved; chunks are
//! contiguous ⇒ global order == the serial two-pass gather, byte-for-byte.
//!
//! The CPU here is REAL hardware (only the GPU is the spoofed APU), so these
//! wall-clock ratios are legitimate, not SIGN-only.
//!
//! Run: `cargo run -p lgbm-compute --example spike026_partition_scan_scatter_ab --release`
//! Env: LGBM_SPIKE_CHUNK (default 512), LGBM_SPIKE_RUN (label for >=2-process repeats).

use std::time::Instant;

use cubecl::prelude::*;
use lgbm_compute::kernels::partition::data_partition_cpu_native;
use lgbm_compute::runtime::cpu_client;

type Client = ComputeClient<cubecl::cpu::CpuRuntime>;

/// COUNT kernel — one cube per chunk (`ABSOLUTE_POS` = chunk index, `CubeDim(1)`).
/// Writes `chunk_left[c]` = number of rows in chunk `c` routed LEFT. The routing is
/// byte-identical to `data_partition_cpu_native` (dense_bin.hpp:322-365).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn partition_count_kernel(
    bins: &Array<u32>,
    chunk_left: &mut Array<u32>,
    n: u32,
    chunk: u32,
    min_bin: i32,
    max_bin: i32,
    threshold: i32,
    most_freq_bin: i32,
) {
    let c = ABSOLUTE_POS;
    if c < chunk_left.len() {
        let chunk_u = usize::cast_from(chunk);
        let n_u = usize::cast_from(n);
        let start = c * chunk_u;
        let mut end = start + chunk_u;
        if end > n_u {
            end = n_u;
        }
        // Routing constants (identical to the serial native path).
        let mut th = threshold + min_bin;
        if most_freq_bin == 0 {
            th -= 1;
        }
        let default_to_right = most_freq_bin > threshold;

        let mut cnt = 0u32;
        let mut i = start;
        while i < end {
            let bin = bins[i] as i32;
            let is_default = bin < min_bin || bin > max_bin;
            let gt = bin > th;
            let go_right = select(is_default, default_to_right, gt);
            // count LEFT (go_right == false)
            cnt += select(go_right, 0u32, 1u32);
            i += 1;
        }
        chunk_left[c] = cnt;
    }
}

/// SCATTER kernel — one cube per chunk. Walks the chunk's rows ASCENDING and
/// scatters each global row index into the stable `[all-left | all-right]` layout
/// at its chunk's disjoint `left_base[c]` / `right_base[c]`. Disjoint write ranges
/// (the host prefix-sum guarantees no overlap) ⇒ no atomics, deterministic order.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn partition_scatter_kernel(
    bins: &Array<u32>,
    left_base: &Array<u32>,
    right_base: &Array<u32>,
    reordered: &mut Array<u32>,
    n: u32,
    chunk: u32,
    min_bin: i32,
    max_bin: i32,
    threshold: i32,
    most_freq_bin: i32,
) {
    let c = ABSOLUTE_POS;
    if c < left_base.len() {
        let chunk_u = usize::cast_from(chunk);
        let n_u = usize::cast_from(n);
        let start = c * chunk_u;
        let mut end = start + chunk_u;
        if end > n_u {
            end = n_u;
        }
        let mut th = threshold + min_bin;
        if most_freq_bin == 0 {
            th -= 1;
        }
        let default_to_right = most_freq_bin > threshold;

        // write cursors (positions are usize for indexing; bases are u32 in the array)
        let mut l = usize::cast_from(left_base[c]);
        let mut r = usize::cast_from(right_base[c]);
        let mut i = start;
        while i < end {
            let bin = bins[i] as i32;
            let is_default = bin < min_bin || bin > max_bin;
            let gt = bin > th;
            let go_right = select(is_default, default_to_right, gt);
            // branchless: write to the active cursor, bump only that cursor.
            let pos = select(go_right, r, l);
            reordered[pos] = u32::cast_from(i);
            r += select(go_right, 1usize, 0usize);
            l += select(go_right, 0usize, 1usize);
            i += 1;
        }
    }
}

/// The cubecl-cpu scan+scatter `data_partition` op. Returns `(reordered, split_point)`
/// byte-identical to `data_partition_cpu_native`.
fn cubecl_partition(
    client: &Client,
    bins: &[u32],
    min_bin: u32,
    max_bin: u32,
    threshold: u32,
    most_freq_bin: u32,
    chunk: u32,
) -> (Vec<u32>, usize) {
    let prof = std::env::var("LGBM_SPIKE_PROF").is_ok();
    let n = bins.len();
    if n == 0 {
        return (Vec::new(), 0);
    }
    let n_u = n as u32;
    let n_chunks = n_u.div_ceil(chunk);
    // Launch geometry: ABSOLUTE_POS = chunk index regardless of split. cubecl-cpu
    // parallelism may live on the UNIT (CubeDim) axis, so make it tunable.
    let cd: u32 = std::env::var("LGBM_SPIKE_CUBEDIM").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    let grid = n_chunks.div_ceil(cd);

    let t = Instant::now();
    let h_bins = client.create_from_slice(u32::as_bytes(bins));
    let zeros_chunks = vec![0u32; n_chunks as usize];
    let h_chunk_left = client.create_from_slice(u32::as_bytes(&zeros_chunks));
    let t_marshal = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();

    // --- Phase 1: per-chunk LEFT count (one cube per chunk) ---
    // SAFETY: h_bins[n], h_chunk_left[n_chunks] outlive the launch; the kernel
    // bounds-guards c < n_chunks and i < end <= n.
    unsafe {
        partition_count_kernel::launch(
            client,
            CubeCount::Static(grid, 1, 1),
            CubeDim::new_1d(cd),
            ArrayArg::from_raw_parts(h_bins.clone(), n),
            ArrayArg::from_raw_parts(h_chunk_left.clone(), n_chunks as usize),
            n_u,
            chunk,
            min_bin as i32,
            max_bin as i32,
            threshold as i32,
            most_freq_bin as i32,
        );
    }
    let cl_bytes = client.read_one_unchecked(h_chunk_left.clone());
    let chunk_left = u32::from_bytes(&cl_bytes);
    let t_count = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();

    // --- Phase 2: host exclusive prefix-sum → disjoint write bases (tiny serial) ---
    let nc = n_chunks as usize;
    let mut left_base = vec![0u32; nc];
    let mut acc = 0u32;
    for c in 0..nc {
        left_base[c] = acc;
        acc += chunk_left[c];
    }
    let total_left = acc;
    // right rows of chunk c start at total_left + (rows-before-c − left-before-c).
    // rows-before-c = c*chunk (all prior chunks are full size `chunk`).
    let mut right_base = vec![0u32; nc];
    for c in 0..nc {
        let chunk_start = (c as u32) * chunk;
        right_base[c] = total_left + (chunk_start - left_base[c]);
    }

    let t_scan = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();

    let h_left_base = client.create_from_slice(u32::as_bytes(&left_base));
    let h_right_base = client.create_from_slice(u32::as_bytes(&right_base));
    let zeros_n = vec![0u32; n];
    let h_reordered = client.create_from_slice(u32::as_bytes(&zeros_n));

    // --- Phase 3: scatter (one cube per chunk) ---
    // SAFETY: every output slot 0..n is written exactly once (disjoint bases); all
    // handles outlive the launch and are bounds-guarded in the kernel.
    unsafe {
        partition_scatter_kernel::launch(
            client,
            CubeCount::Static(grid, 1, 1),
            CubeDim::new_1d(cd),
            ArrayArg::from_raw_parts(h_bins.clone(), n),
            ArrayArg::from_raw_parts(h_left_base.clone(), nc),
            ArrayArg::from_raw_parts(h_right_base.clone(), nc),
            ArrayArg::from_raw_parts(h_reordered.clone(), n),
            n_u,
            chunk,
            min_bin as i32,
            max_bin as i32,
            threshold as i32,
            most_freq_bin as i32,
        );
    }
    client.read_one_unchecked(h_reordered.clone()); // force scatter completion
    let t_scatter = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let re_bytes = client.read_one_unchecked(h_reordered.clone());
    let reordered = u32::from_bytes(&re_bytes).to_vec();
    let t_read = t.elapsed().as_secs_f64() * 1e3;
    if prof {
        eprintln!(
            "  [prof n={n} chunk={chunk} nchunks={n_chunks}] marshal={t_marshal:.3} count+read={t_count:.3} hostscan={t_scan:.3} scatter={t_scatter:.3} read_out={t_read:.3} ms"
        );
    }
    (reordered, total_left as usize)
}

/// Deterministic bins in [0, num_bin) via a simple LCG. `skew` in [0,1]: fraction of
/// rows forced into the low half (exercises an unbalanced left/right split).
fn gen_bins(n: usize, num_bin: u32, skew: f64, seed: u64) -> Vec<u32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 33) as u32
    };
    (0..n)
        .map(|_| {
            let r = next();
            if (r as f64 / u32::MAX as f64) < skew {
                r % (num_bin / 2).max(1) // low half
            } else {
                r % num_bin
            }
        })
        .collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let chunk: u32 = std::env::var("LGBM_SPIKE_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
    let run = std::env::var("LGBM_SPIKE_RUN").unwrap_or_else(|_| "1".into());
    let client = cpu_client();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0);

    println!("# spike-026 cubecl-cpu scan+scatter partition A/B  (run={run}, chunk={chunk}, cores={threads})");
    println!("# ratio = serial_native / cubecl  (>1 ⇒ cubecl FASTER). CPU is real hardware.");
    println!("{:>10} {:>6} {:>5} {:>12} {:>12} {:>7} {:>6}", "rows", "nbin", "skew", "serial(ms)", "cubecl(ms)", "ratio", "parity");

    let num_bin = 64u32;
    let min_bin = 0u32;
    let max_bin = num_bin - 1;
    let threshold = 31u32; // ~balanced split at skew=0
    let most_freq_bin = 0u32; // exercises the `th -= 1` branch
    let reps = 25;

    for &rows in &[1_000usize, 16_384, 100_000, 500_000, 1_000_000, 4_000_000] {
        for &skew in &[0.0f64, 0.9] {
            let bins = gen_bins(rows, num_bin, skew, 0xC0FFEE ^ rows as u64);

            // correctness: cubecl output must be byte-identical to serial native.
            let (s_re, s_sp) =
                data_partition_cpu_native(&bins, num_bin, min_bin, max_bin, threshold, most_freq_bin).unwrap();
            let (c_re, c_sp) = cubecl_partition(&client, &bins, min_bin, max_bin, threshold, most_freq_bin, chunk);
            let parity = s_sp == c_sp && s_re == c_re;

            // warmup
            let _ = data_partition_cpu_native(&bins, num_bin, min_bin, max_bin, threshold, most_freq_bin).unwrap();
            let _ = cubecl_partition(&client, &bins, min_bin, max_bin, threshold, most_freq_bin, chunk);

            let mut st = Vec::with_capacity(reps);
            let mut ct = Vec::with_capacity(reps);
            for _ in 0..reps {
                let t = Instant::now();
                let r = data_partition_cpu_native(&bins, num_bin, min_bin, max_bin, threshold, most_freq_bin).unwrap();
                st.push(t.elapsed().as_secs_f64() * 1e3);
                std::hint::black_box(r);

                let t = Instant::now();
                let r = cubecl_partition(&client, &bins, min_bin, max_bin, threshold, most_freq_bin, chunk);
                ct.push(t.elapsed().as_secs_f64() * 1e3);
                std::hint::black_box(r);
            }
            let sm = median(st);
            let cm = median(ct);
            println!(
                "{:>10} {:>6} {:>5.1} {:>12.3} {:>12.3} {:>6.2}x {:>6}",
                rows, num_bin, skew, sm, cm, sm / cm, if parity { "OK" } else { "FAIL" }
            );
            assert!(parity, "PARITY FAIL at rows={rows} skew={skew}: cubecl != serial native");
        }
    }
}
