//! Spike 016 — host-side parity probe: does parallelizing the within-feature
//! best-split scan (which REORDERS the f64 prefix-sum) flip the argmax
//! (best_threshold / default_left) → tree divergence?
//!
//! The GPU scan (`split_scan_body`, split.rs) walks bins SEQUENTIALLY accumulating
//! `sum_right_grad/hess` and picks the argmax gain. A parallel within-feature scan
//! computes the same prefix sums via a TREE reduction → different f64 rounding →
//! slightly different gains → POSSIBLE argmax flip at near-ties. Bit-exactness is
//! impossible; the open question (this probe) is the FLIP RATE.
//!
//! Pure host f64, no GPU — calls the REAL `gain::get_split_gains`. Runs both
//! accumulation orders over many representative histograms and reports how often
//! the chosen split differs. Cheap probe before any GPU kernel (spike-008 method).
//!
//! Run: cargo run --release --example spike016_scan_reorder_probe

use lgbm_compute::gain::get_split_gains;

/// Sum a slice in SEQUENTIAL order (left-to-right) — matches the current scan.
fn sum_seq(v: &[f64]) -> f64 {
    let mut s = 0.0;
    for &x in v {
        s += x;
    }
    s
}

/// Sum a slice via a PAIRWISE-TREE reduction — representative of what a parallel
/// prefix-scan produces (associative reordering of the f64 additions).
fn sum_tree(v: &[f64]) -> f64 {
    if v.len() <= 1 {
        return v.first().copied().unwrap_or(0.0);
    }
    let mid = v.len() / 2;
    sum_tree(&v[..mid]) + sum_tree(&v[mid..])
}

#[derive(Clone, Copy, PartialEq, Debug)]
struct Best {
    threshold: i32,
    default_left: bool,
    gain: f64,
    splittable: bool,
}

/// Faithful-enough reverse+forward best-split scan over one feature histogram.
/// `grad[b]`, `hess[b]` are per-bin sums. `order` selects how the running
/// cumulative `sum_right`/`sum_left` is accumulated (seq vs tree) — the ONLY thing
/// that differs between the sequential and parallel kernels. Gates + gain math +
/// strict-`>` tie-break mirror split_scan_body's common (offset=0, no-skip) path.
fn scan(grad: &[f64], hess: &[f64], cnt: &[i32], cfg: &Cfg, tree_order: bool) -> Best {
    let num_bin = grad.len();
    let sum_g: f64 = sum_seq(grad); // leaf totals: same both ways (not the scan axis)
    let sum_h: f64 = sum_seq(hess);
    let num_data: i32 = cnt.iter().sum();
    let k_eps = 1e-15f64;
    let mut best = Best { threshold: 0, default_left: true, gain: 0.0, splittable: false };

    // ---- REVERSE: split point t means right = bins (t..num_bin); default_left ----
    for t in (1..num_bin).rev() {
        // sum over bins [t, num_bin)
        let sr_g = if tree_order { sum_tree(&grad[t..]) } else { sum_seq(&grad[t..]) };
        let sr_h = (if tree_order { sum_tree(&hess[t..]) } else { sum_seq(&hess[t..]) }) + k_eps;
        let right_count: i32 = cnt[t..].iter().sum();
        let left_count = num_data - right_count;
        let sl_g = sum_g - sr_g;
        let sl_h = sum_h - sr_h;
        if right_count < cfg.min_data || sr_h < cfg.min_hess {
            continue;
        }
        if left_count < cfg.min_data || sl_h < cfg.min_hess {
            continue; // (probe: treat as non-candidate; monotone break ≈ this for our synthetic data)
        }
        let g = get_split_gains(cfg.use_l1, sl_g, sl_h, sr_g, sr_h, cfg.l1, cfg.l2);
        let cand = if g > cfg.min_gain_shift { g } else { 0.0 };
        if cand > 0.0 {
            best.splittable = true;
        }
        if cand > best.gain {
            // strict `>` — keep first (we iterate t descending; FORWARD below re-checks low t)
            best = Best { threshold: (t as i32) - 1, default_left: true, gain: cand, splittable: true };
        }
    }
    // ---- FORWARD: split point t means left = bins [0, t]; default_right ----
    for t in 0..num_bin.saturating_sub(1) {
        let sl_g = if tree_order { sum_tree(&grad[..=t]) } else { sum_seq(&grad[..=t]) };
        let sl_h = (if tree_order { sum_tree(&hess[..=t]) } else { sum_seq(&hess[..=t]) }) + k_eps;
        let left_count: i32 = cnt[..=t].iter().sum();
        let right_count = num_data - left_count;
        let sr_g = sum_g - sl_g;
        let sr_h = sum_h - sl_h;
        if left_count < cfg.min_data || sl_h < cfg.min_hess {
            continue;
        }
        if right_count < cfg.min_data || sr_h < cfg.min_hess {
            continue;
        }
        let g = get_split_gains(cfg.use_l1, sl_g, sl_h, sr_g, sr_h, cfg.l1, cfg.l2);
        let cand = if g > cfg.min_gain_shift { g } else { 0.0 };
        if cand > 0.0 {
            best.splittable = true;
        }
        if cand > best.gain {
            best = Best { threshold: t as i32, default_left: false, gain: cand, splittable: true };
        }
    }
    best
}

struct Cfg {
    use_l1: bool,
    l1: f64,
    l2: f64,
    min_data: i32,
    min_hess: f64,
    min_gain_shift: f64,
}

/// Deterministic LCG → reproducible "random" histograms (no rng dep).
struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

fn main() {
    // Sweep: bin counts × hessian regimes × gradient scales. For each, generate
    // many feature histograms and compare the two accumulation orders.
    let cfg = Cfg { use_l1: false, l1: 0.0, l2: 0.0, min_data: 20, min_hess: 1e-3, min_gain_shift: 0.0 };
    let bin_counts = [16usize, 64, 128, 256];
    let per = 50_000usize; // histograms per (bins) — large sample for a rare-event flip rate

    println!("# spike016 — parallel-scan REORDER parity probe (sequential f64 vs pairwise-tree f64)");
    println!("# {} histograms per bin-count; reports best-threshold / default_left FLIP rate", per);
    println!("{:>6} {:>10} {:>12} {:>14} {:>14} {:>16}", "bins", "tested", "thr_flips", "dleft_flips", "splittab_flip", "max_gain_reldiff");

    let mut grand_tested = 0u64;
    let mut grand_thr = 0u64;
    let mut grand_dl = 0u64;
    let mut grand_max_rel = 0.0f64;
    let mut seed = 0x9E3779B97F4A7C15u64;

    for &nb in &bin_counts {
        let mut rng = Lcg(seed);
        seed = seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
        let (mut tested, mut thr_flips, mut dl_flips, mut sp_flips) = (0u64, 0u64, 0u64, 0u64);
        let mut max_rel = 0.0f64;
        for _ in 0..per {
            // Representative histogram: hessian ~ U(0.2,1.0) per bin (binary/regression-ish),
            // gradient ~ centered noise scaled so the leaf grad-sum is near 0 (real GBDT).
            // Mix in some "spiky" bins and a fraction of near-equal bins to stress near-ties.
            let scale = 10f64.powf(rng.next_f64() * 6.0 - 3.0); // 1e-3 .. 1e3 magnitude spread
            let mut grad = vec![0.0f64; nb];
            let mut hess = vec![0.0f64; nb];
            let mut cnt = vec![0i32; nb];
            for b in 0..nb {
                let h = 0.2 + 0.8 * rng.next_f64();
                let n = 1 + (rng.next_f64() * 200.0) as i32;
                hess[b] = h * n as f64;
                cnt[b] = n;
                grad[b] = (rng.next_f64() * 2.0 - 1.0) * scale * n as f64;
            }
            // make a fraction of histograms have clustered/near-equal structure (tie stress).
            // SPIKE016_TIESTRESS=0.0 → realistic baseline; 0.3 → near-tie worst case.
            let tie_frac: f64 = std::env::var("SPIKE016_TIESTRESS").ok().and_then(|s| s.parse().ok()).unwrap_or(0.3);
            if rng.next_f64() < tie_frac {
                for b in 1..nb {
                    grad[b] = grad[0] * (1.0 + (rng.next_f64() - 0.5) * 1e-9);
                    hess[b] = hess[0] * (1.0 + (rng.next_f64() - 0.5) * 1e-9);
                    cnt[b] = cnt[0];
                }
            }
            let a = scan(&grad, &hess, &cnt, &cfg, false);
            let b = scan(&grad, &hess, &cnt, &cfg, true);
            tested += 1;
            if a.threshold != b.threshold {
                thr_flips += 1;
            }
            if a.default_left != b.default_left {
                dl_flips += 1;
            }
            if a.splittable != b.splittable {
                sp_flips += 1;
            }
            if a.gain > 0.0 && b.gain > 0.0 {
                let rel = (a.gain - b.gain).abs() / a.gain.abs().max(1e-300);
                if rel > max_rel {
                    max_rel = rel;
                }
            }
        }
        println!("{:>6} {:>10} {:>12} {:>14} {:>14} {:>16.2e}", nb, tested, thr_flips, dl_flips, sp_flips, max_rel);
        grand_tested += tested;
        grand_thr += thr_flips;
        grand_dl += dl_flips;
        if max_rel > grand_max_rel {
            grand_max_rel = max_rel;
        }
    }
    println!("# TOTAL tested={} thr_flips={} ({:.4}%) dleft_flips={} ({:.4}%) max_gain_reldiff={:.2e}",
        grand_tested, grand_thr, 100.0 * grand_thr as f64 / grand_tested as f64,
        grand_dl, 100.0 * grand_dl as f64 / grand_tested as f64, grand_max_rel);
    println!("# VERDICT GUIDE: thr_flip rate ~0 + gain_reldiff << 1e-6 → parallel scan is parity-safe (within hip gate).");
    println!("#               non-trivial thr/dleft flips → reordering changes splits → tree divergence risk → do NOT wire.");
}
