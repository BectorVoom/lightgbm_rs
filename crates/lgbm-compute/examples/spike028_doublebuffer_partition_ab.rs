//! Spike 028 — double-buffer the partition `indices` to drop the V1 copy-back.
//!
//! Follow-on to spike-027 (the fused u8-route host partition, now WIRED). V1 ends with
//! `self.indices[begin..begin+count].copy_from_slice(&out)` — it scatters into a scratch
//! `out` Vec then copies back, because an in-place scatter would clobber unread rows. This
//! spike asks: keep a PERSISTENT second `indices` buffer and scatter directly into
//! `alt[begin..]` (ping-pong), skipping the copy-back + the `out` alloc. Does it measurably
//! beat V1, and is the win worth the cross-leaf bookkeeping?
//!
//! HONEST FRAMING: C++ LightGBM's DataPartition::Split ALSO copies back (via
//! `temp_indices_` inside ParallelPartitionRunner), so the copy-back is not obviously
//! removable. This spike measures (a) the copy-back's share of V1 and (b) the op-level
//! ceiling of removing it — the cross-leaf consistency cost (which buffer is canonical for
//! each leaf region as the tree grows leaf-wise) is assessed separately, not benched here.
//!
//! CPU is REAL hardware ⇒ legitimate wall-clock.
//!
//! Run: `cargo run -p lgbm-compute --example spike028_doublebuffer_partition_ab --release`

use std::time::Instant;

use lgbm_compute::BinColumn;

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

/// V1 (WIRED) — fused u8-route, scatter into a reused `out` scratch, then copy_from_slice
/// back into `cur[begin..]`. `scratch` is the reused `out` buffer (sized >= count).
/// Returns (left_count, copyback_ms) — copyback timed in isolation.
fn v1_copyback(
    cur: &mut [u32],
    scratch: &mut [u32],
    begin: usize,
    count: usize,
    feature_bins: &BinColumn,
    min_bin: u32,
    max_bin: u32,
    threshold: u32,
    most_freq_bin: u32,
) -> (usize, f64) {
    let go_right = make_router(min_bin, max_bin, threshold, most_freq_bin);
    let mut route = vec![0u8; count];
    let mut left_count = 0usize;
    for i in 0..count {
        let gr = go_right(feature_bins.bin(cur[begin + i] as usize));
        route[i] = gr as u8;
        left_count += (!gr) as usize;
    }
    let out = &mut scratch[..count];
    let mut l = 0usize;
    let mut r = left_count;
    for i in 0..count {
        let row = cur[begin + i];
        if route[i] == 0 {
            out[l] = row;
            l += 1;
        } else {
            out[r] = row;
            r += 1;
        }
    }
    let t = Instant::now();
    cur[begin..begin + count].copy_from_slice(out);
    let copyback_ms = t.elapsed().as_secs_f64() * 1e3;
    (left_count, copyback_ms)
}

/// V1-DBUF — same fused u8-route, but scatter ROW ids DIRECTLY into a persistent
/// `alt[begin..]` (the ping-pong target). No `out` scratch, no copy-back. The caller would
/// then treat `alt` as canonical for this leaf's region (cross-leaf bookkeeping out of scope).
fn v1_doublebuffer(
    cur: &[u32],
    alt: &mut [u32],
    begin: usize,
    count: usize,
    feature_bins: &BinColumn,
    min_bin: u32,
    max_bin: u32,
    threshold: u32,
    most_freq_bin: u32,
) -> usize {
    let go_right = make_router(min_bin, max_bin, threshold, most_freq_bin);
    let mut route = vec![0u8; count];
    let mut left_count = 0usize;
    for i in 0..count {
        let gr = go_right(feature_bins.bin(cur[begin + i] as usize));
        route[i] = gr as u8;
        left_count += (!gr) as usize;
    }
    let mut l = begin;
    let mut r = begin + left_count;
    for i in 0..count {
        let row = cur[begin + i];
        if route[i] == 0 {
            alt[l] = row;
            l += 1;
        } else {
            alt[r] = row;
            r += 1;
        }
    }
    left_count
}

fn lcg(seed: u64) -> impl FnMut() -> u32 {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 33) as u32
    }
}

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
    println!("# spike-028 double-buffer partition (drop the V1 copy-back)  (run={run})");
    println!("# ratio = V1_copyback / V1_dbuf  (>1 ⇒ double-buffer FASTER). copyback% = copy-back share of V1.");
    println!(
        "{:>10} {:>5} {:>6} {:>11} {:>11} {:>9} {:>9} {:>6}",
        "rows", "skew", "width", "v1_cb(ms)", "v1_dbuf(ms)", "ratio", "copyback%", "parity"
    );

    let num_bin = 64u32;
    let (min_bin, max_bin, threshold, most_freq_bin) = (0u32, num_bin - 1, 31u32, 0u32);
    let reps = 21;

    for &width in &[8u8, 32] {
        for &rows in &[100_000usize, 1_000_000, 4_000_000] {
            for &skew in &[0.0f64, 0.9] {
                let col = gen_column(rows, num_bin, skew, width, 0xC0FFEE ^ rows as u64);
                let base = gen_indices(rows, 0xBEEF ^ rows as u64);
                let (begin, count) = (0usize, rows);

                // correctness: copyback's cur[begin..] must equal dbuf's alt[begin..].
                let mut cur = base.clone();
                let mut scr = vec![0u32; rows];
                let (lc_a, _) =
                    v1_copyback(&mut cur, &mut scr, begin, count, &col, min_bin, max_bin, threshold, most_freq_bin);
                let curc = base.clone();
                let mut alt = vec![0u32; rows];
                let lc_b =
                    v1_doublebuffer(&curc, &mut alt, begin, count, &col, min_bin, max_bin, threshold, most_freq_bin);
                let parity = lc_a == lc_b && cur[begin..begin + count] == alt[begin..begin + count];

                let mut tcb = Vec::with_capacity(reps);
                let mut tdb = Vec::with_capacity(reps);
                let mut tcopy = Vec::with_capacity(reps);
                for _ in 0..reps {
                    let mut cur = base.clone();
                    let mut scr = vec![0u32; rows];
                    let t = Instant::now();
                    let (_, cb) =
                        v1_copyback(&mut cur, &mut scr, begin, count, &col, min_bin, max_bin, threshold, most_freq_bin);
                    tcb.push(t.elapsed().as_secs_f64() * 1e3);
                    tcopy.push(cb);
                    std::hint::black_box(&cur);

                    let curc = base.clone();
                    let mut alt = vec![0u32; rows];
                    let t = Instant::now();
                    v1_doublebuffer(&curc, &mut alt, begin, count, &col, min_bin, max_bin, threshold, most_freq_bin);
                    tdb.push(t.elapsed().as_secs_f64() * 1e3);
                    std::hint::black_box(&alt);
                }
                let mcb = median(tcb);
                let mdb = median(tdb);
                let mcopy = median(tcopy);
                println!(
                    "{:>10} {:>5.1} {:>6} {:>11.3} {:>11.3} {:>8.2}x {:>8.1}% {:>6}",
                    rows, skew, width, mcb, mdb, mcb / mdb, 100.0 * mcopy / mcb, if parity { "OK" } else { "FAIL" }
                );
                assert!(parity, "PARITY FAIL rows={rows} skew={skew} width={width}");
            }
        }
    }
}
