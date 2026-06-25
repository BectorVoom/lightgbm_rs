//! Spike 032 — eliminate the REDUNDANT validation random-gather in the shipped
//! spike-027 host partition (`DataPartition::split_fused_host`, data_partition.rs:206).
//!
//! WHY: spike-027 validated a ONE-random-gather fused partition. But the PRODUCTION
//! wiring (`split_fused_host`) does TWO random gathers over the leaf:
//!   1. validation pass (data_partition.rs:236-246): `for i { feature_bins.bin(row) }`
//!      range-checks every leaf row's bin and surfaces the lowest-index offender.
//!   2. pass-1 route+count (data_partition.rs:270-275): `feature_bins.bin(row)` AGAIN.
//! On a MEMORY-BANDWIDTH/LATENCY-bound op (026/027's core finding) that second random
//! gather is ~the most expensive thing in the function — exactly the traffic the whole
//! 026->027 arc fought to cut. At scale the column exceeds cache so the 2nd gather
//! re-misses; at small sizes it's cached (validation ~free) ⇒ expect a regime-split win.
//!
//! VARIANTS (user: "try both, A/B them"):
//!   V0 baseline  = the SHIPPED `split_fused_host`: validation gather + route gather
//!                  + scatter + copyback. TWO random gathers.
//!   V1 fold      = fold the `b >= num_bin` range-check INTO pass-1's route gather
//!                  (one gather + a branch). Early-return on the first offender in
//!                  ascending leaf order = SAME lowest-index error; `indices` is not
//!                  mutated until the final copy_from_slice ⇒ BIT-EXACT on success AND
//!                  on the error path.
//!   V2 relocate  = NO per-split range-check at all (relocated once-per-train, the
//!                  spike-003b/r4o precedent; C++ trusts binning). == the original
//!                  spike-027 `v1_fused_u8route`. ONE gather, no branch.
//!
//! Both V1 and V2 are bit-exact to V0 on valid bins (all real training bins are valid).
//! The V0/V1 and V0/V2 ratios isolate the redundant-validation-gather cost; V1 vs V2
//! isolates the cost of keeping the defensive branch on the hot path.
//!
//! CPU is REAL hardware (only the GPU is the spoofed APU) ⇒ legitimate wall-clock.
//!
//! Run: `cargo run -p lgbm-compute --example spike032_partition_validation_fold_ab --release`
//! Env: LGBM_SPIKE_RUN (label for >=2-process repeats).

use std::time::Instant;

use lgbm_compute::BinColumn;

/// Shared routing decision (dense_bin.hpp:322-365), identical to the production path.
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

/// V0 — faithful replica of the SHIPPED `DataPartition::split_fused_host`
/// (data_partition.rs:206-298): a SEPARATE validation gather, THEN the route gather,
/// THEN the scatter + copyback. Returns `Err` semantics elided (bins always valid in
/// real training); the validation LOOP cost is what we measure. TWO random gathers.
#[inline(never)]
fn v0_shipped(
    indices: &mut [u32],
    begin: usize,
    count: usize,
    feature_bins: &BinColumn,
    num_bin: u32,
    min_bin: u32,
    max_bin: u32,
    threshold: u32,
    most_freq_bin: u32,
) -> bool {
    // validation pass (data_partition.rs:236-246) — random gather #1.
    let mut ok = true;
    for i in 0..count {
        let row = indices[begin + i] as usize;
        let b = feature_bins.bin(row);
        if b >= num_bin {
            ok = false;
            break;
        }
    }
    std::hint::black_box(ok);

    let go_right = make_router(min_bin, max_bin, threshold, most_freq_bin);
    // pass 1: route + count — random gather #2.
    let mut route = vec![0u8; count];
    let mut left_count = 0usize;
    for i in 0..count {
        let row = indices[begin + i];
        let gr = go_right(feature_bins.bin(row as usize));
        route[i] = gr as u8;
        left_count += (!gr) as usize;
    }
    // pass 2: scatter rows into one output buffer (route read sequentially), copyback.
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
    ok
}

/// V1 — FOLD the range-check into pass-1's route gather. ONE random gather + a branch.
/// On the first out-of-range bin in ascending leaf order, return early WITHOUT having
/// mutated `indices` (only `route`/`left_count` written) ⇒ same lowest-index error,
/// bit-exact.
#[inline(never)]
fn v1_fold(
    indices: &mut [u32],
    begin: usize,
    count: usize,
    feature_bins: &BinColumn,
    num_bin: u32,
    min_bin: u32,
    max_bin: u32,
    threshold: u32,
    most_freq_bin: u32,
) -> bool {
    let go_right = make_router(min_bin, max_bin, threshold, most_freq_bin);
    let mut route = vec![0u8; count];
    let mut left_count = 0usize;
    // pass 1: gather + RANGE-CHECK + route + count — the ONE random gather.
    for i in 0..count {
        let row = indices[begin + i];
        let b = feature_bins.bin(row as usize);
        if b >= num_bin {
            return false; // indices not yet mutated ⇒ same as V0's pre-mutation error
        }
        let gr = go_right(b);
        route[i] = gr as u8;
        left_count += (!gr) as usize;
    }
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
    true
}

/// V2 — RELOCATE validation once-per-train (no per-split check). == the original
/// spike-027 `v1_fused_u8route`. ONE random gather, no branch.
#[inline(never)]
fn v2_relocate(
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
    for i in 0..count {
        let row = indices[begin + i];
        let gr = go_right(feature_bins.bin(row as usize));
        route[i] = gr as u8;
        left_count += (!gr) as usize;
    }
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
    println!("# spike-032 eliminate the redundant validation gather in split_fused_host  (run={run})");
    println!("# ratio = V0_shipped / Vk  (>1 ⇒ one-gather FASTER). CPU is real hardware. width = bin-column storage.");
    println!(
        "{:>10} {:>6} {:>5} {:>6} {:>10} {:>10} {:>10} {:>8} {:>8} {:>6}",
        "rows", "nbin", "skew", "width", "v0(ms)", "v1fold(ms)", "v2reloc(ms)", "v0/v1", "v0/v2", "parity"
    );

    let num_bin = 64u32;
    let min_bin = 0u32;
    let max_bin = num_bin - 1;
    let threshold = 31u32;
    let most_freq_bin = 0u32;
    let reps = 25;

    for &width in &[8u8, 32] {
        for &rows in &[16_384usize, 100_000, 500_000, 1_000_000, 4_000_000] {
            for &skew in &[0.0f64, 0.9] {
                let col = gen_column(rows, num_bin, skew, width, 0xC0FFEE ^ rows as u64);
                let base_idx = gen_indices(rows, 0xBEEF ^ rows as u64);
                let begin = 0;
                let count = rows;

                // correctness: all three produce identical indices[begin..begin+count].
                let mut a = base_idx.clone();
                let ok0 = v0_shipped(&mut a, begin, count, &col, num_bin, min_bin, max_bin, threshold, most_freq_bin);
                let mut b = base_idx.clone();
                let ok1 = v1_fold(&mut b, begin, count, &col, num_bin, min_bin, max_bin, threshold, most_freq_bin);
                let mut c = base_idx.clone();
                v2_relocate(&mut c, begin, count, &col, min_bin, max_bin, threshold, most_freq_bin);
                let parity = ok0 && ok1 && a == b && a == c;

                // warmup (discard) — touch the column once so the FIRST timed rep isn't
                // cold-page inflated; the win is steady-state cache-miss traffic.
                {
                    let mut w = base_idx.clone();
                    v0_shipped(&mut w, begin, count, &col, num_bin, min_bin, max_bin, threshold, most_freq_bin);
                    std::hint::black_box(&w);
                }

                let mut t0 = Vec::with_capacity(reps);
                let mut t1 = Vec::with_capacity(reps);
                let mut t2 = Vec::with_capacity(reps);
                for _ in 0..reps {
                    // interleave V0/V1/V2 per rep to cancel thermal/scheduler drift.
                    let mut w = base_idx.clone();
                    let t = Instant::now();
                    v0_shipped(&mut w, begin, count, &col, num_bin, min_bin, max_bin, threshold, most_freq_bin);
                    t0.push(t.elapsed().as_secs_f64() * 1e3);
                    std::hint::black_box(&w);

                    let mut w = base_idx.clone();
                    let t = Instant::now();
                    v1_fold(&mut w, begin, count, &col, num_bin, min_bin, max_bin, threshold, most_freq_bin);
                    t1.push(t.elapsed().as_secs_f64() * 1e3);
                    std::hint::black_box(&w);

                    let mut w = base_idx.clone();
                    let t = Instant::now();
                    v2_relocate(&mut w, begin, count, &col, min_bin, max_bin, threshold, most_freq_bin);
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
