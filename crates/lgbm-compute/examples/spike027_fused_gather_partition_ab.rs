//! Spike 027 — fuse the per-leaf bin-GATHER into the partition routing, cutting
//! memory traffic. A/B vs the current materialize-then-partition path.
//!
//! WHY: spike-026 proved the host `DataPartition::split` is MEMORY-BANDWIDTH-bound at
//! scale (shared DDR5) — parallelism (rayon or cubecl-cpu) can't reclaim it. The one
//! lever 026 pointed at: cut the TRAFFIC. The current path
//! (`data_partition.rs:141-173`) materializes THREE per-leaf `Vec`s every split:
//!   1. `leaf_rows`          = indices[begin..].to_vec()                 (count × u32)
//!   2. `leaf_feature_bins`  = leaf_rows.map(|r| feature_bins.bin(r))    (count × u32 —
//!                             WIDENED to u32 even when the column is U8!)
//!   3. `reordered` (local)  from the op, then a writeback maps local→row via leaf_rows.
//! That is ~4 count-sized u32 buffers written+reread per split. This spike FUSES the
//! gather into routing: read the `indices` slice in place, route directly off
//! `feature_bins.bin(row)`, and scatter the ROW ids straight into one output buffer —
//! no `leaf_rows`, no u32-widened `leaf_feature_bins`, no local→row remap.
//!
//! Bit-exact by construction (identical stable [left|right] order). CPU is REAL
//! hardware (only the GPU is the spoofed APU) ⇒ legitimate wall-clock.
//!
//! Run: `cargo run -p lgbm-compute --example spike027_fused_gather_partition_ab --release`
//! Env: LGBM_SPIKE_RUN (label for >=2-process repeats).

use std::time::Instant;

use lgbm_compute::kernels::partition::data_partition_cpu_native;
use lgbm_compute::BinColumn;

/// Shared routing decision (dense_bin.hpp:322-365), identical to the native path.
#[inline(always)]
fn make_router(min_bin: u32, max_bin: u32, threshold: u32, most_freq_bin: u32) -> impl Fn(u32) -> bool {
    let min_b = min_bin as i32;
    let max_b = max_bin as i32;
    let thr = threshold as i32;
    let mut th = thr + min_b;
    if most_freq_bin == 0 {
        th -= 1;
    }
    let default_to_right = most_freq_bin as i32 > thr;
    move |b: u32| -> bool {
        let bin = b as i32;
        if bin < min_b || bin > max_b {
            default_to_right
        } else {
            bin > th
        }
    }
}

/// V0 — faithful replica of the CURRENT `DataPartition::split` numeric branch:
/// materialize leaf_rows + (u32-widened) leaf_feature_bins, call the native op, remap.
fn v0_baseline(
    indices: &mut [u32],
    begin: usize,
    count: usize,
    feature_bins: &BinColumn,
    num_bin: u32,
    min_bin: u32,
    max_bin: u32,
    threshold: u32,
    most_freq_bin: u32,
) {
    let leaf_rows: Vec<u32> = indices[begin..begin + count].to_vec();
    let leaf_feature_bins: Vec<u32> = leaf_rows.iter().map(|&r| feature_bins.bin(r as usize)).collect();
    let (reordered, _sp) =
        data_partition_cpu_native(&leaf_feature_bins, num_bin, min_bin, max_bin, threshold, most_freq_bin).unwrap();
    let mut tmp = vec![0u32; count];
    for (slot, &local) in reordered.iter().enumerate() {
        tmp[slot] = leaf_rows[local as usize];
    }
    indices[begin..begin + count].copy_from_slice(&tmp);
}

/// V1 — FUSED, one gather + a u8 route scratch (1/4 the width of the u32 leaf bins),
/// scatter ROW ids directly. No leaf_rows, no u32-widened leaf_feature_bins, no remap.
fn v1_fused_u8route(
    indices: &mut [u32],
    begin: usize,
    count: usize,
    feature_bins: &BinColumn,
    min_bin: u32,
    max_bin: u32,
    threshold: u32,
    most_freq_bin: u32,
) {
    let go_right = make_router(min_bin, max_bin, threshold, most_freq_bin);
    let mut route = vec![0u8; count];
    let mut left_count = 0usize;
    // pass 1: gather + route + count (the ONE random gather)
    for i in 0..count {
        let row = indices[begin + i];
        let gr = go_right(feature_bins.bin(row as usize));
        route[i] = gr as u8;
        left_count += (!gr) as usize;
    }
    // pass 2: scatter rows directly into a single output buffer (route read sequentially)
    let mut out = vec![0u32; count];
    let mut l = 0usize;
    let mut r = left_count;
    for i in 0..count {
        let row = indices[begin + i];
        if route[i] == 0 {
            out[l] = row;
            l += 1;
        } else {
            out[r] = row;
            r += 1;
        }
    }
    indices[begin..begin + count].copy_from_slice(&out);
}

/// V2 — FUSED, NO route scratch: recompute the route in pass 2 (TWO random gathers).
/// Tests whether eliding the scratch beats paying a second gather (depends on bin width).
fn v2_fused_2gather(
    indices: &mut [u32],
    begin: usize,
    count: usize,
    feature_bins: &BinColumn,
    min_bin: u32,
    max_bin: u32,
    threshold: u32,
    most_freq_bin: u32,
) {
    let go_right = make_router(min_bin, max_bin, threshold, most_freq_bin);
    let mut left_count = 0usize;
    for i in 0..count {
        let row = indices[begin + i];
        if !go_right(feature_bins.bin(row as usize)) {
            left_count += 1;
        }
    }
    let mut out = vec![0u32; count];
    let mut l = 0usize;
    let mut r = left_count;
    for i in 0..count {
        let row = indices[begin + i];
        if !go_right(feature_bins.bin(row as usize)) {
            out[l] = row;
            l += 1;
        } else {
            out[r] = row;
            r += 1;
        }
    }
    indices[begin..begin + count].copy_from_slice(&out);
}

/// Deterministic LCG.
fn lcg(seed: u64) -> impl FnMut() -> u32 {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 33) as u32
    }
}

/// A full-dataset feature bin column of `n` rows, values in [0,num_bin), with `skew`
/// fraction forced into the low half.
fn gen_column(n: usize, num_bin: u32, skew: f64, width: u8, seed: u64) -> BinColumn {
    let mut next = lcg(seed);
    let vals: Vec<u32> = (0..n)
        .map(|_| {
            let r = next();
            if (r as f64 / u32::MAX as f64) < skew {
                r % (num_bin / 2).max(1)
            } else {
                r % num_bin
            }
        })
        .collect();
    match width {
        8 => BinColumn::U8(vals.iter().map(|&v| v as u8).collect()),
        _ => BinColumn::U32(vals),
    }
}

/// A scattered leaf occupying indices[0..count]: a shuffled set of row ids into the
/// column ⇒ the bin gather is RANDOM (models a deep leaf whose rows are not contiguous).
fn gen_indices(count: usize, seed: u64) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..count as u32).collect();
    let mut next = lcg(seed);
    for i in (1..count).rev() {
        let j = (next() as usize) % (i + 1);
        idx.swap(i, j);
    }
    idx
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let run = std::env::var("LGBM_SPIKE_RUN").unwrap_or_else(|_| "1".into());
    println!("# spike-027 fuse leaf-bin gather into routing  (run={run})");
    println!("# ratio = V0_baseline / Vk  (>1 ⇒ fused FASTER). CPU is real hardware. width = bin-column storage.");
    println!(
        "{:>10} {:>6} {:>5} {:>6} {:>10} {:>10} {:>10} {:>8} {:>8} {:>6}",
        "rows", "nbin", "skew", "width", "v0(ms)", "v1_u8(ms)", "v2_2gat(ms)", "v1/v0", "v2/v0", "parity"
    );

    let num_bin = 64u32;
    let min_bin = 0u32;
    let max_bin = num_bin - 1;
    let threshold = 31u32;
    let most_freq_bin = 0u32;
    let reps = 21;

    for &width in &[8u8, 32] {
        for &rows in &[16_384usize, 100_000, 500_000, 1_000_000, 4_000_000] {
            for &skew in &[0.0f64, 0.9] {
                let col = gen_column(rows, num_bin, skew, width, 0xC0FFEE ^ rows as u64);
                let base_idx = gen_indices(rows, 0xBEEF ^ rows as u64);
                let begin = 0;
                let count = rows;

                // correctness: all three produce identical indices[begin..begin+count].
                let mut a = base_idx.clone();
                v0_baseline(&mut a, begin, count, &col, num_bin, min_bin, max_bin, threshold, most_freq_bin);
                let mut b = base_idx.clone();
                v1_fused_u8route(&mut b, begin, count, &col, min_bin, max_bin, threshold, most_freq_bin);
                let mut c = base_idx.clone();
                v2_fused_2gather(&mut c, begin, count, &col, min_bin, max_bin, threshold, most_freq_bin);
                let parity = a == b && a == c;

                let mut t0 = Vec::with_capacity(reps);
                let mut t1 = Vec::with_capacity(reps);
                let mut t2 = Vec::with_capacity(reps);
                for _ in 0..reps {
                    let mut w = base_idx.clone();
                    let t = Instant::now();
                    v0_baseline(&mut w, begin, count, &col, num_bin, min_bin, max_bin, threshold, most_freq_bin);
                    t0.push(t.elapsed().as_secs_f64() * 1e3);
                    std::hint::black_box(&w);

                    let mut w = base_idx.clone();
                    let t = Instant::now();
                    v1_fused_u8route(&mut w, begin, count, &col, min_bin, max_bin, threshold, most_freq_bin);
                    t1.push(t.elapsed().as_secs_f64() * 1e3);
                    std::hint::black_box(&w);

                    let mut w = base_idx.clone();
                    let t = Instant::now();
                    v2_fused_2gather(&mut w, begin, count, &col, min_bin, max_bin, threshold, most_freq_bin);
                    t2.push(t.elapsed().as_secs_f64() * 1e3);
                    std::hint::black_box(&w);
                }
                let m0 = median(t0);
                let m1 = median(t1);
                let m2 = median(t2);
                println!(
                    "{:>10} {:>6} {:>5.1} {:>6} {:>10.3} {:>10.3} {:>10.3} {:>7.2}x {:>7.2}x {:>6}",
                    rows, num_bin, skew, width, m0, m1, m2, m0 / m1, m0 / m2, if parity { "OK" } else { "FAIL" }
                );
                assert!(parity, "PARITY FAIL rows={rows} skew={skew} width={width}");
            }
        }
    }
}
