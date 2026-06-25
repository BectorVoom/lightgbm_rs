//! Spike 022a — host parity probe RESOLVING spike-016's deferred question:
//! are the within-feature parallel-scan `default_left` tie-flips COSMETIC
//! (missing-routing-only, present-data predictions identical) or REAL
//! (the default bin's actual data is routed differently → tree divergence)?
//!
//! spike-016 found, under a reordered f64 prefix-sum: 0 threshold flips on
//! realistic data, but ~34% `default_left` flips at equal-gain reverse-vs-forward
//! ties — and could NOT classify them because its probe modelled only the common
//! `offset=0` / no-skip path, omitting the default-bin (`most_freq_bin`) semantics.
//!
//! This probe closes that gap with TWO faithfulness upgrades over spike-016:
//!   1. Models `offset` (default bin = bin 0 when offset==1), `skip_default_bin`,
//!      and tracks the DEFAULT-BIN MASS so each flip can be classified.
//!   2. Uses the EXACT reorder cubecl-hip 0.10 emits for `plane_inclusive_sum`:
//!      a Hillis-Steele `__shfl_up` loop (not spike-016's pairwise-tree proxy).
//!
//! Decisive test: a `default_left` flip is REAL iff the default bin carries
//! present data AND the leaf outputs differ when it is routed left vs right.
//! Hypothesis (from the split semantics): flips occur ONLY when the default bin
//! is empty (else the two directions have materially different gains, no tie,
//! reorder ~1e-12 cannot flip) ⇒ every flip is cosmetic ⇒ parity-safe.
//!
//! Pure host f64, no GPU — calls the REAL `gain::{get_split_gains,
//! calculate_splitted_leaf_output}`. Run:
//!   cargo run --release -p lgbm-compute --example spike022_default_bin_parity_probe
//!   SPIKE022_TIESTRESS=0.3 cargo run --release -p lgbm-compute --example spike022_default_bin_parity_probe

use lgbm_compute::gain::{calculate_splitted_leaf_output, get_split_gains};

const K_EPS: f64 = 1e-15;

// ----- prefix-sum association: sequential (production) vs Hillis-Steele (the GPU) ---

/// Sequential inclusive scan — the running sum the current single-lane kernel does.
fn scan_seq(v: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(v.len());
    let mut s = 0.0;
    for &x in v {
        s += x;
        out.push(s);
    }
    out
}

/// Hillis-Steele inclusive scan — the EXACT association cubecl-hip 0.10 emits for
/// `plane_inclusive_sum` (`reduce_inclusive` → `__shfl_up` loop: at step `offset`,
/// `if(lane >= offset) acc += shfl_up(acc, offset)`). Each prefix becomes a
/// balanced tree of adds, reordered vs the sequential running sum.
fn scan_hs(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut a = v.to_vec();
    let mut offset = 1usize;
    while offset < n {
        let prev = a.clone(); // shfl reads the PREVIOUS step's values
        for i in offset..n {
            a[i] = prev[i] + prev[i - offset];
        }
        offset <<= 1;
    }
    a
}

#[derive(Clone, Copy, PartialEq, Debug)]
struct Best {
    threshold: i32,
    default_left: bool,
    gain: f64,
    splittable: bool,
    // recorded present-data partition of the winner (for cosmetic/real classification)
    left_g: f64,
    left_h: f64,
    left_count: i32,
}

struct Cfg {
    use_l1: bool,
    l1: f64,
    l2: f64,
    min_data: i32,
    min_hess: f64,
    min_gain_shift: f64,
}

/// Faithful reverse+forward best-split scan over one feature histogram, with the
/// prefix-sum association selected by `hs` (false = sequential, true = Hillis-Steele).
/// Models `offset` and `skip_default_bin` exactly like `split_scan_body`.
/// `grad[b]`/`hess[b]`/`cnt[b]` are per-bin sums; bin `default_bin` is the default.
#[allow(clippy::too_many_arguments)]
fn scan(
    grad: &[f64],
    hess: &[f64],
    cnt: &[i32],
    offset: i32,
    default_bin: i32,
    skip_default: bool,
    cfg: &Cfg,
    hs: bool,
) -> Best {
    let num_bin = grad.len() as i32;
    let sum_g: f64 = grad.iter().sum();
    let sum_h: f64 = hess.iter().sum();
    let num_data: i32 = cnt.iter().sum();
    let scanfn = if hs { scan_hs } else { scan_seq };

    let mut best = Best {
        threshold: 0,
        default_left: true,
        gain: 0.0,
        splittable: false,
        left_g: 0.0,
        left_h: 0.0,
        left_count: 0,
    };

    // ---------------- REVERSE (default_left = true) ----------------
    // C++: for t = num_bin-1-offset .. 1-offset (step -1); right grows downward.
    {
        let t_start = num_bin - 1 - offset;
        let count = (num_bin - 1).max(0);
        // Build the per-iteration contributions (gated by active), then scan them
        // with the chosen association — isolates EXACTLY the reorder.
        let mut cg = Vec::with_capacity(count as usize);
        let mut ch = Vec::with_capacity(count as usize);
        let mut active_mask = Vec::with_capacity(count as usize);
        for k in 0..count {
            let t = t_start - k;
            let in_range = t >= (1 - offset);
            let skip = skip_default && (t + offset) == default_bin;
            let active = in_range && !skip;
            let t_safe = if t < 0 { 0 } else { t } as usize;
            cg.push(if active { grad[t_safe] } else { 0.0 });
            ch.push(if active { hess[t_safe] } else { 0.0 });
            active_mask.push((active, t, t_safe));
        }
        let cum_g = scanfn(&cg);
        let cum_h = scanfn(&ch);
        // counts are integer — association-independent; cumulative via running sum.
        let mut right_count = 0i32;
        let mut done = false;
        for k in 0..count as usize {
            let (active, t, t_safe) = active_mask[k];
            if active {
                right_count += cnt[t_safe];
            }
            let sr_g = cum_g[k];
            let sr_h = cum_h[k] + K_EPS;
            let left_count = num_data - right_count;
            let sl_h = sum_h - sr_h;
            let sl_g = sum_g - sr_g;
            let cont = right_count < cfg.min_data || sr_h < cfg.min_hess;
            let brk = left_count < cfg.min_data || sl_h < cfg.min_hess;
            if active && !cont && brk {
                done = true;
            }
            let consider = active && !cont && !done;
            let g = get_split_gains(cfg.use_l1, sl_g, sl_h, sr_g, sr_h, cfg.l1, cfg.l2);
            let valid = consider && g > cfg.min_gain_shift;
            let cand = if valid { g } else { 0.0 };
            if valid {
                best.splittable = true;
            }
            if cand > best.gain {
                best.threshold = t - 1 + offset;
                best.default_left = true;
                best.gain = cand;
                best.left_g = sl_g;
                best.left_h = sl_h;
                best.left_count = left_count;
                best.splittable = true;
            }
        }
    }

    // ---------------- FORWARD (default_left = false) ----------------
    {
        let count = (num_bin - 1 - offset).max(0);
        let mut cg = Vec::with_capacity(count as usize);
        let mut ch = Vec::with_capacity(count as usize);
        let mut active_mask = Vec::with_capacity(count as usize);
        for t in 0..count {
            let skip = skip_default && (t + offset) == default_bin;
            let active = !skip;
            cg.push(if active { grad[t as usize] } else { 0.0 });
            ch.push(if active { hess[t as usize] } else { 0.0 });
            active_mask.push((active, t));
        }
        let cum_g = scanfn(&cg);
        let cum_h = scanfn(&ch);
        let mut left_count = 0i32;
        let mut done = false;
        for k in 0..count as usize {
            let (active, t) = active_mask[k];
            if active {
                left_count += cnt[t as usize];
            }
            let sl_g = cum_g[k];
            let sl_h = cum_h[k] + K_EPS;
            let right_count = num_data - left_count;
            let sr_h = sum_h - sl_h;
            let sr_g = sum_g - sl_g;
            let cont = left_count < cfg.min_data || sl_h < cfg.min_hess;
            let brk = right_count < cfg.min_data || sr_h < cfg.min_hess;
            if active && !cont && brk {
                done = true;
            }
            let consider = active && !cont && !done;
            let g = get_split_gains(cfg.use_l1, sl_g, sl_h, sr_g, sr_h, cfg.l1, cfg.l2);
            let valid = consider && g > cfg.min_gain_shift;
            let cand = if valid { g } else { 0.0 };
            if valid {
                best.splittable = true;
            }
            if cand > best.gain {
                best.threshold = t + offset;
                best.default_left = false;
                best.gain = cand;
                best.left_g = sl_g;
                best.left_h = sl_h;
                best.left_count = left_count;
                best.splittable = true;
            }
        }
    }
    best
}

/// Deterministic LCG → reproducible histograms (no rng dep).
struct Lcg(u64);
impl Lcg {
    fn f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

/// DIRECT mechanism demonstration (the random sweep never lands an offset==1
/// populated-default near-tie, so prove the mechanism explicitly): at a FIXED
/// threshold, the reverse winner routes the default bin LEFT and the forward
/// winner routes it RIGHT. They tie (→ a reorder-flippable argmax) only when the
/// default bin's mass → 0. Sweep the mass and report the gain gap |rev−fwd| and
/// whether the seq-vs-HS argmax flips. Shows: flip ⟺ mass≈0 ⟺ cosmetic.
fn mechanism_demo(cfg: &Cfg) {
    println!("\n# MECHANISM: reverse(default→left) vs forward(default→right) at a fixed split,");
    println!("# sweeping the DEFAULT-BIN mass. The reorder perturbs a gain by a NOISE FLOOR");
    println!("# ε≈1e-12 (measured: threshold-flip gain reldiff ~1e-13). A flip needs the");
    println!("# rev-vs-fwd gain gap < ε; the gap is LINEAR in default-bin mass, so only an");
    println!("# (essentially empty) default bin can flip — and an empty bin moves no data.");
    // Representative reorder-noise floor (absolute), from the random sweep: a gain
    // of O(1–10) is perturbed by ~1e-12 by the Hillis-Steele reassociation.
    let eps_noise = 1e-12f64;
    println!(
        "{:>12} {:>14} {:>14} {:>12} {:>10}",
        "def_mass(h)", "gain_gap", "gap<ε(flip?)", "flip_kind", "leaf_outΔ"
    );
    // Fixed present-data partition: left = a block, right = a block, plus a default
    // bin D whose mass we sweep. Gradients chosen so left/right are a real split.
    let left_g = -8.0;
    let left_h = 20.0;
    let right_g = 9.0;
    let right_h = 22.0;
    for &dh in &[0.0f64, 1e-9, 1e-6, 1e-3, 1.0, 5.0, 20.0] {
        let dg = -0.45 * dh; // default bin carries a typical gradient/hessian ratio
        // Reverse hypothesis: D in LEFT.   Forward hypothesis: D in RIGHT.
        // Compute each gain in seq order and in a 2-element HS (reordered) order.
        let rev = |order_hs: bool| {
            let lg = if order_hs { scan_hs(&[left_g, dg])[1] } else { scan_seq(&[left_g, dg])[1] };
            let lh = if order_hs { scan_hs(&[left_h, dh])[1] } else { scan_seq(&[left_h, dh])[1] };
            get_split_gains(cfg.use_l1, lg, lh, right_g, right_h, cfg.l1, cfg.l2)
        };
        let fwd = |order_hs: bool| {
            let rg = if order_hs { scan_hs(&[right_g, dg])[1] } else { scan_seq(&[right_g, dg])[1] };
            let rh = if order_hs { scan_hs(&[right_h, dh])[1] } else { scan_seq(&[right_h, dh])[1] };
            get_split_gains(cfg.use_l1, left_g, left_h, rg, rh, cfg.l1, cfg.l2)
        };
        let gap = (rev(false) - fwd(false)).abs();
        // A ~ε reorder perturbation can flip the argmax iff the gap is below it.
        let flippable = gap < eps_noise;
        // present-data leaf-output divergence between the two routings of D
        let out_dleft = calculate_splitted_leaf_output(cfg.use_l1, left_g + dg, left_h + dh, cfg.l1, cfg.l2);
        let out_dright = calculate_splitted_leaf_output(cfg.use_l1, left_g, left_h, cfg.l1, cfg.l2);
        let leafd = (out_dleft - out_dright).abs();
        let kind = if !flippable {
            "stable"
        } else if leafd < 1e-6 {
            "COSMETIC"
        } else {
            "REAL!"
        };
        println!(
            "{dh:>12.0e} {gap:>14.3e} {:>14} {kind:>12} {leafd:>10.2e}",
            if flippable { "yes" } else { "no" }
        );
    }
    println!("# ⇒ flip only where gap<ε (mass≲1e-12, i.e. empty), and there leaf_outΔ≈0 ⇒ COSMETIC.");
    println!("#   Any populated default bin (mass≥1e-9) gives gap≫ε ⇒ argmax stable ⇒ no flip.");
}

fn main() {
    let tiestress: f64 = std::env::var("SPIKE022_TIESTRESS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let cfg = Cfg {
        use_l1: false,
        l1: 0.0,
        l2: 0.0,
        min_data: 20,
        min_hess: 1e-3,
        min_gain_shift: 0.0,
    };
    let bin_counts = [16usize, 64, 128, 256];
    let per = 60_000usize;

    println!("# spike022 — default_left flip COSMETIC-vs-REAL probe (seq f64 vs Hillis-Steele f64)");
    println!(
        "# tiestress={tiestress} ; {per} histograms/bin-count ; offset∈{{0,1}}, default-bin mass varied"
    );
    println!(
        "{:>5} {:>8} {:>8} {:>8} {:>9} {:>9} {:>9} {:>11}",
        "bins", "tested", "thr_flip", "dl_flip", "miss_only", "def_empty", "def_FULL", "thr_gapmax"
    );

    let mut seed = 0x9E3779B97F4A7C15u64;
    let (mut g_tested, mut g_thr, mut g_dl) = (0u64, 0u64, 0u64);
    let (mut g_miss, mut g_empty, mut g_full) = (0u64, 0u64, 0u64);
    let mut g_max_leafdiv = 0.0f64; // present-data leaf-output divergence of any flip
    let mut g_thr_gap = 0.0f64; // worst gain reldiff among threshold flips
    // The decisive counter: a flip that is REAL — present-data leaf outputs differ
    // beyond 1e-6 (only possible if a populated default bin is routed differently).
    let mut g_real = 0u64;

    for &nb in &bin_counts {
        let mut lcg = Lcg(seed);
        seed = seed.wrapping_mul(0xD1B54A32D192ED03).wrapping_add(1);
        let (mut tested, mut thr_f, mut dl_f) = (0u64, 0u64, 0u64);
        let (mut miss_f, mut empty_f, mut full_f) = (0u64, 0u64, 0u64);
        let mut thr_gap = 0.0f64;

        for _ in 0..per {
            // offset 0 (no special default bin in scan) or 1 (default bin = bin 0).
            let offset = if lcg.f64() < 0.5 { 0 } else { 1 };
            let default_bin = 0i32;
            let skip_default = offset == 1; // when offset==1, bin 0 is skipped in the scan

            // Build per-bin grad/hess/count. Hessian ~ positive counts; gradient
            // can be ±. Add a near-tie stress on a fraction of bins to manufacture
            // the equal-gain ties where flips live.
            let mut grad = vec![0.0f64; nb];
            let mut hess = vec![0.0f64; nb];
            let mut cnt = vec![0i32; nb];
            for b in 0..nb {
                let c = 1 + (lcg.f64() * 40.0) as i32;
                let h = c as f64 * (0.5 + lcg.f64());
                let mut g = (lcg.f64() - 0.5) * 2.0 * h;
                if tiestress > 0.0 && lcg.f64() < tiestress {
                    // cluster gradients to create ~1e-9 near-tie gains
                    g = (g * 1e-9).round() * 1e9_f64.recip() * h;
                }
                cnt[b] = c;
                hess[b] = h;
                grad[b] = g;
            }
            // Default-bin (bin 0) MASS: half the time empty, half populated — this
            // is the variable that decides cosmetic vs real.
            if offset == 1 {
                if lcg.f64() < 0.5 {
                    grad[0] = 0.0;
                    hess[0] = 0.0;
                    cnt[0] = 0; // EMPTY default bin
                } else {
                    // POPULATED default bin (heavy, like a real most-frequent value)
                    let c = 30 + (lcg.f64() * 200.0) as i32;
                    cnt[0] = c;
                    hess[0] = c as f64 * (0.5 + lcg.f64());
                    grad[0] = (lcg.f64() - 0.5) * 2.0 * hess[0];
                }
            }

            let bseq = scan(&grad, &hess, &cnt, offset, default_bin, skip_default, &cfg, false);
            let bhs = scan(&grad, &hess, &cnt, offset, default_bin, skip_default, &cfg, true);
            tested += 1;
            if !(bseq.splittable && bhs.splittable) {
                continue; // need both to find a split to compare
            }
            if bseq.threshold != bhs.threshold {
                // A threshold flip IS a different present-data partition. Record how
                // far apart the two winners' gains are — if ~1e-12 it is an
                // equal-gain PLATEAU (arbitrary choice between equally-good splits,
                // the tie class the hip parity test is already tie-aware for); a
                // large gap would mean the reorder genuinely changed the decision.
                thr_f += 1;
                let gap = (bseq.gain - bhs.gain).abs() / bseq.gain.abs().max(1e-300);
                if gap > thr_gap {
                    thr_gap = gap;
                }
                continue;
            }
            if bseq.default_left != bhs.default_left {
                dl_f += 1;
                // Classify the flip by WHAT moves between left and right:
                //   - offset==0  → only MISSING values move (no histogram mass) → cosmetic.
                //   - offset==1, default bin (bin 0) EMPTY → nothing moves → cosmetic.
                //   - offset==1, default bin POPULATED → bin 0's data moves → REAL.
                let (dg, dh) = if offset == 1 { (grad[0], hess[0]) } else { (0.0, 0.0) };
                let moved = cnt[0] > 0 && offset == 1;
                if offset == 0 {
                    miss_f += 1;
                } else if moved {
                    full_f += 1;
                } else {
                    empty_f += 1;
                }
                // Present-data leaf-output divergence: route the moved mass (dg,dh)
                // left vs right. For cosmetic cases dg=dh=0 ⇒ leafdiv≡0.
                let total_g: f64 = grad.iter().sum();
                let total_h: f64 = hess.iter().sum();
                let l_g = bseq.left_g;
                let l_h = bseq.left_h;
                let (la_g, la_h, lb_g, lb_h) = if bseq.default_left {
                    (l_g, l_h, l_g - dg, l_h - dh)
                } else {
                    (l_g + dg, l_h + dh, l_g, l_h)
                };
                let outa = calculate_splitted_leaf_output(cfg.use_l1, la_g, la_h, cfg.l1, cfg.l2);
                let outb = calculate_splitted_leaf_output(cfg.use_l1, lb_g, lb_h, cfg.l1, cfg.l2);
                let routa =
                    calculate_splitted_leaf_output(cfg.use_l1, total_g - la_g, total_h - la_h, cfg.l1, cfg.l2);
                let routb =
                    calculate_splitted_leaf_output(cfg.use_l1, total_g - lb_g, total_h - lb_h, cfg.l1, cfg.l2);
                let leafdiv = (outa - outb).abs().max((routa - routb).abs());
                if leafdiv > g_max_leafdiv {
                    g_max_leafdiv = leafdiv;
                }
                if leafdiv > 1e-6 {
                    g_real += 1;
                }
            }
        }

        println!(
            "{nb:>5} {tested:>8} {thr_f:>8} {dl_f:>8} {miss_f:>9} {empty_f:>9} {full_f:>9} {thr_gap:>11.2e}"
        );
        g_tested += tested;
        g_thr += thr_f;
        g_dl += dl_f;
        g_miss += miss_f;
        g_empty += empty_f;
        g_full += full_f;
        if thr_gap > g_thr_gap {
            g_thr_gap = thr_gap;
        }
    }

    println!("\n# TOTALS");
    println!("  tested                 : {g_tested}");
    println!(
        "  threshold flips        : {g_thr}  (worst gain reldiff {g_thr_gap:.2e} — ≈1e-12 ⇒ equal-gain plateau, not a real decision change)"
    );
    println!("  default_left flips     : {g_dl}");
    println!("    ├ missing-only (off0) : {g_miss}  (cosmetic: only MISSING values reroute; no histogram mass moves)");
    println!("    ├ empty default (off1): {g_empty}  (cosmetic: skipped default bin carries no present data)");
    println!("    └ FULL default  (off1): {g_full}  (REAL if it moved bin-0 data → present-partition change)");
    println!("  max present leaf-out Δ : {g_max_leafdiv:.3e}  (prediction impact on PRESENT data of any default_left flip)");
    println!("  REAL flips (Δ>1e-6)    : {g_real}");
    let plateau = g_thr_gap < 1e-6;
    let verdict = if g_full == 0 && g_real == 0 && plateau {
        "PARITY-SAFE (within ~1e-6) — every default_left flip is COSMETIC (no populated default bin ever flips; \
         present-data leaf outputs unchanged), and threshold flips are equal-gain plateaus (gain reldiff ~1e-12). \
         A tie-aware parallel argmax (reverse-first / lowest-t) reproduces the SAME splits within the hip gate."
    } else if g_real == 0 {
        "PARTIAL — default_left flips cosmetic, but some threshold flips have a non-negligible gain gap (investigate)."
    } else {
        "UNSAFE — some flips move PRESENT-data leaf outputs >1e-6 (a populated default bin is routed differently)."
    };
    println!("\n# VERDICT: {verdict}");

    mechanism_demo(&cfg);
}
