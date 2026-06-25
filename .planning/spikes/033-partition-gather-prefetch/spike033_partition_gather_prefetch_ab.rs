//! Spike 033 — software-PREFETCH the residual random bin-gather in the host
//! partition (`split_fused_host` pass-1, post-spike-032 one-gather path).
//!
//! WHY: after spike-032 the partition does ONE random gather per leaf row:
//! `feature_bins.bin(self.indices[begin+i])` — a random index into the bin column.
//! At scale the column exceeds LLC ⇒ miss-latency-bound (spike-030's mechanism;
//! spike-032's win regime). Classic latency-hiding = software prefetch: while at
//! row i, issue `prefetch(col[indices[begin+i+D]])` so row i+D's line is in flight
//! before use. `indices[begin+i+D]` is a SEQUENTIAL read (cheap/HW-prefetched); the
//! random part is the dereference into the column.
//!
//! HONEST RISK — may be NULL: the gather loads col[idx[i]] are INDEPENDENT across i,
//! so a wide OoO core already extracts MLP up to its load-buffer depth (~10-12) with
//! no help. SW prefetch wins only if lookahead beyond the ROB is needed or HW MLP is
//! under-saturated. A negative result is a valuable outcome (saves a portability-
//! burdened wiring dud; cf 006/008/011/028).
//!
//! CLEAN ATTRIBUTION (CONVENTIONS: vary one thing) — two stacked effects, separate:
//!   V0 prod   = faithful production: `.bin()` enum-match per row, bounds-checked.
//!   V1 hoist  = match the BinColumn variant ONCE, gather off a typed &[T] slice.
//!   V2 pf     = V1 + `_mm_prefetch` at distance D (pure prefetch over V1).
//! Measured on PASS-1 ALONE with PREALLOCATED buffers (the sensitive isolation —
//! prefetch acts only on the gather; allocator noise would swamp it). Also report the
//! honest WHOLE-OP dilution (pass1 + shared scatter). Parity byte-identical by
//! construction (prefetch/hoist cannot change values) — asserted every cell.
//!
//! CPU is REAL hardware (only the GPU is the spoofed APU) ⇒ legitimate wall-clock.
//!
//! Run: `cargo run -p lgbm-compute --example spike033_partition_gather_prefetch_ab --release`
//! Env: LGBM_SPIKE_RUN (label for >=2-process repeats).

use std::time::Instant;

use lgbm_compute::BinColumn;

// ---- prefetch intrinsics (x86_64 stable; no-op fallback elsewhere) -------------
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn pf_t0<T>(base: *const T, idx: usize) {
    unsafe { core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(base.add(idx) as *const i8) }
}
#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
unsafe fn pf_t0<T>(_base: *const T, _idx: usize) {}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn pf_nta<T>(base: *const T, idx: usize) {
    unsafe { core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_NTA }>(base.add(idx) as *const i8) }
}
#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
unsafe fn pf_nta<T>(_base: *const T, _idx: usize) {}

// ---- the routing decision (dense_bin.hpp:322-365), inlined identically ----------
#[inline(always)]
fn route_decision(b: u32, min_b: i32, max_b: i32, th: i32, dflt: bool) -> bool {
    let bin = b as i32;
    if bin < min_b || bin > max_b {
        dflt
    } else {
        bin > th
    }
}

/// Precompute the router constants exactly as `split_fused_host` / `make_router`.
fn router_consts(min_bin: u32, max_bin: u32, threshold: u32, most_freq_bin: u32) -> (i32, i32, i32, bool) {
    let min_b = min_bin as i32;
    let max_b = max_bin as i32;
    let thr = threshold as i32;
    let mut th = thr + min_b;
    if most_freq_bin == 0 {
        th -= 1;
    }
    let dflt = most_freq_bin as i32 > thr;
    (min_b, max_b, th, dflt)
}

// ---- V0: faithful production pass-1 (enum-match .bin per row) -------------------
fn pass1_prod(
    indices: &[u32],
    begin: usize,
    count: usize,
    col: &BinColumn,
    route: &mut [u8],
    min_b: i32,
    max_b: i32,
    th: i32,
    dflt: bool,
) -> usize {
    let mut left = 0usize;
    for i in 0..count {
        let row = indices[begin + i] as usize;
        let gr = route_decision(col.bin(row), min_b, max_b, th, dflt);
        route[i] = gr as u8;
        left += (!gr) as usize;
    }
    left
}

// ---- V1: hoisted typed-slice gather (no prefetch) ------------------------------
fn pass1_hoist<T: Copy>(
    indices: &[u32],
    begin: usize,
    count: usize,
    col: &[T],
    route: &mut [u8],
    min_b: i32,
    max_b: i32,
    th: i32,
    dflt: bool,
) -> usize
where
    u32: From<T>,
{
    let mut left = 0usize;
    for i in 0..count {
        let row = indices[begin + i] as usize;
        let gr = route_decision(u32::from(col[row]), min_b, max_b, th, dflt);
        route[i] = gr as u8;
        left += (!gr) as usize;
    }
    left
}

// ---- V2: hoisted typed-slice gather + prefetch at distance `dist` ---------------
fn pass1_hoist_pf<T: Copy>(
    indices: &[u32],
    begin: usize,
    count: usize,
    col: &[T],
    route: &mut [u8],
    min_b: i32,
    max_b: i32,
    th: i32,
    dflt: bool,
    dist: usize,
    nta: bool,
) -> usize
where
    u32: From<T>,
{
    let base = col.as_ptr();
    let mut left = 0usize;
    for i in 0..count {
        let j = i + dist;
        if j < count {
            // indices[begin+j] is a SEQUENTIAL read; prefetch the RANDOM column line.
            let pr = indices[begin + j] as usize;
            unsafe {
                if nta {
                    pf_nta(base, pr);
                } else {
                    pf_t0(base, pr);
                }
            }
        }
        let row = indices[begin + i] as usize;
        let gr = route_decision(u32::from(col[row]), min_b, max_b, th, dflt);
        route[i] = gr as u8;
        left += (!gr) as usize;
    }
    left
}

// ---- variant dispatchers over the BinColumn enum -------------------------------
fn run_hoist(
    indices: &[u32],
    begin: usize,
    count: usize,
    col: &BinColumn,
    route: &mut [u8],
    c: (i32, i32, i32, bool),
) -> usize {
    match col {
        BinColumn::U8(v) => pass1_hoist(indices, begin, count, v, route, c.0, c.1, c.2, c.3),
        BinColumn::U16(v) => pass1_hoist(indices, begin, count, v, route, c.0, c.1, c.2, c.3),
        BinColumn::U32(v) => pass1_hoist(indices, begin, count, v, route, c.0, c.1, c.2, c.3),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_pf(
    indices: &[u32],
    begin: usize,
    count: usize,
    col: &BinColumn,
    route: &mut [u8],
    c: (i32, i32, i32, bool),
    dist: usize,
    nta: bool,
) -> usize {
    match col {
        BinColumn::U8(v) => pass1_hoist_pf(indices, begin, count, v, route, c.0, c.1, c.2, c.3, dist, nta),
        BinColumn::U16(v) => pass1_hoist_pf(indices, begin, count, v, route, c.0, c.1, c.2, c.3, dist, nta),
        BinColumn::U32(v) => pass1_hoist_pf(indices, begin, count, v, route, c.0, c.1, c.2, c.3, dist, nta),
    }
}

// ---- shared pass-2 scatter (identical across all variants) ----------------------
fn scatter(indices: &[u32], begin: usize, count: usize, route: &[u8], left: usize, out: &mut [u32]) {
    let mut l = 0usize;
    let mut r = left;
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
}

// ---- harness (LCG / column / scattered indices — identical to spike-032) --------
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
    println!("# spike-033 software-prefetch the residual random bin-gather in split_fused_host pass-1  (run={run})");
    #[cfg(not(target_arch = "x86_64"))]
    println!("# WARNING: not x86_64 — prefetch is a NO-OP, V2==V1.");
    println!("# pass-1-only (prealloc buffers): isolates the gather. ratio = p0_prod / pX (>1 ⇒ faster).");
    println!("# op = pass1 + shared scatter (honest dilution). bestD = argmin median over D in {{16,32,64,128}}, T0.");
    println!(
        "{:>9} {:>5} {:>5} {:>8} {:>8} {:>8} {:>5} {:>8} {:>8} {:>8} {:>8} {:>6}",
        "rows", "skew", "width", "p0prod", "p1hoist", "pfbest", "bestD", "scatter", "p0/pfb", "op0/opf", "p0/p1", "parity"
    );
    let dists = [16usize, 32, 64, 128];

    let num_bin = 64u32;
    let min_bin = 0u32;
    let max_bin = num_bin - 1;
    let threshold = 31u32;
    let most_freq_bin = 0u32;
    let c = router_consts(min_bin, max_bin, threshold, most_freq_bin);
    let reps = 25;
    let begin = 0usize;

    // collect (rows,skew,width,bestD,bestRatio) for an NTA spot-check at the end.
    for &width in &[8u8, 32] {
        for &rows in &[100_000usize, 500_000, 1_000_000, 4_000_000] {
            for &skew in &[0.0f64, 0.9] {
                let col = gen_column(rows, num_bin, skew, width, 0xC0FFEE ^ rows as u64);
                let idx = gen_indices(rows, 0xBEEF ^ rows as u64);
                let count = rows;

                // preallocated, reused across reps (isolate gather from allocator).
                let mut route = vec![0u8; count];
                let mut out = vec![0u32; count];

                // ---- parity: prod == hoist == pf(32) == pf-nta(32), and scatter eq.
                let l0 = pass1_prod(&idx, begin, count, &col, &mut route, c.0, c.1, c.2, c.3);
                let route_ref = route.clone();
                let mut out_ref = vec![0u32; count];
                scatter(&idx, begin, count, &route_ref, l0, &mut out_ref);

                let lh = run_hoist(&idx, begin, count, &col, &mut route, c);
                let ph = route == route_ref && lh == l0;

                let lpf = run_pf(&idx, begin, count, &col, &mut route, c, 32, false);
                scatter(&idx, begin, count, &route, lpf, &mut out);
                let ppf = route == route_ref && lpf == l0 && out == out_ref;

                let lnta = run_pf(&idx, begin, count, &col, &mut route, c, 32, true);
                let pnta = route == route_ref && lnta == l0;

                let parity = ph && ppf && pnta;
                drop(route_ref);

                // ---- warmup (discard)
                let _ = pass1_prod(&idx, begin, count, &col, &mut route, c.0, c.1, c.2, c.3);
                std::hint::black_box(&route);

                // ---- timing: interleave per rep to cancel drift.
                let mut t0 = Vec::with_capacity(reps);
                let mut t1 = Vec::with_capacity(reps);
                let mut td: Vec<Vec<f64>> = dists.iter().map(|_| Vec::with_capacity(reps)).collect();
                let mut ts = Vec::with_capacity(reps);
                for _ in 0..reps {
                    let s = Instant::now();
                    let l = pass1_prod(&idx, begin, count, &col, &mut route, c.0, c.1, c.2, c.3);
                    t0.push(s.elapsed().as_secs_f64() * 1e3);
                    std::hint::black_box((&route, l));

                    let s = Instant::now();
                    let l = run_hoist(&idx, begin, count, &col, &mut route, c);
                    t1.push(s.elapsed().as_secs_f64() * 1e3);
                    std::hint::black_box((&route, l));

                    for (k, &d) in dists.iter().enumerate() {
                        let s = Instant::now();
                        let l = run_pf(&idx, begin, count, &col, &mut route, c, d, false);
                        td[k].push(s.elapsed().as_secs_f64() * 1e3);
                        std::hint::black_box((&route, l));
                    }

                    let l = pass1_prod(&idx, begin, count, &col, &mut route, c.0, c.1, c.2, c.3);
                    let s = Instant::now();
                    scatter(&idx, begin, count, &route, l, &mut out);
                    ts.push(s.elapsed().as_secs_f64() * 1e3);
                    std::hint::black_box(&out);
                }
                let m0 = median(t0);
                let m1 = median(t1);
                let ms = median(ts);
                let md: Vec<f64> = td.into_iter().map(median).collect();
                let (best_k, &mbest) = md
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap();
                let best_d = dists[best_k];
                let op_ratio = (m0 + ms) / (mbest + ms);
                println!(
                    "{:>9} {:>5.1} {:>5} {:>8.3} {:>8.3} {:>8.3} {:>5} {:>8.3} {:>7.2}x {:>7.2}x {:>7.2}x {:>6}",
                    rows, skew, width, m0, m1, mbest, best_d, ms, m0 / mbest, op_ratio, m0 / m1, if parity { "OK" } else { "FAIL" }
                );
                assert!(parity, "PARITY FAIL rows={rows} skew={skew} width={width}");
            }
        }
    }

    // ---- T0 vs NTA spot-check at the largest shapes (pollution question) --------
    println!("\n# T0 vs NTA @ D=32, pass-1-only (does non-temporal beat keep-in-L1?)");
    println!("{:>9} {:>5} {:>8} {:>8} {:>8}", "rows", "width", "pfT0", "pfNTA", "T0/NTA");
    for &width in &[8u8, 32] {
        let rows = 4_000_000usize;
        let skew = 0.9f64;
        let col = gen_column(rows, num_bin, skew, width, 0xC0FFEE ^ rows as u64);
        let idx = gen_indices(rows, 0xBEEF ^ rows as u64);
        let count = rows;
        let mut route = vec![0u8; count];
        let _ = run_pf(&idx, begin, count, &col, &mut route, c, 32, false);
        std::hint::black_box(&route);
        let mut tt = Vec::new();
        let mut tn = Vec::new();
        for _ in 0..reps {
            let s = Instant::now();
            let l = run_pf(&idx, begin, count, &col, &mut route, c, 32, false);
            tt.push(s.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box((&route, l));
            let s = Instant::now();
            let l = run_pf(&idx, begin, count, &col, &mut route, c, 32, true);
            tn.push(s.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box((&route, l));
        }
        let mt = median(tt);
        let mn = median(tn);
        println!("{:>9} {:>5} {:>8.3} {:>8.3} {:>7.2}x", rows, width, mt, mn, mt / mn);
    }
}
