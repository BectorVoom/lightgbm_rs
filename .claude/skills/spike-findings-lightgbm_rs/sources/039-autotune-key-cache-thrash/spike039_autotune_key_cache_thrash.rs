//! Spike 039 — What `AutotuneKey` granularity avoids a per-leaf tuning STORM?
//!
//! In tree learning the histogram build runs once per node, and every node has a
//! DIFFERENT row count. If the AutotuneKey includes the exact `rows`, then (almost) every
//! node is a brand-new key ⇒ a full cold benchmark (~300–500ms) at EVERY node ⇒ a tuning
//! storm that would dwarf training. This spike measures three keying strategies over a
//! realistic per-node row-count sequence from a simulated tree:
//!   - EXACT  : key on exact rows            → one cold tune per distinct size (thrash)
//!   - BUCKET : key on floor(log2(rows))     → one cold tune per size DECADE (few)
//!   - FIXED  : key on feats only (rows out) → exactly one cold tune, ever
//!
//! The tension: too FINE thrashes; too COARSE may pick the wrong variant for some regime
//! (P=16 wins big leaves, P=1 small). BUCKET is the hypothesised sweet spot — few tunes,
//! one per occupancy regime. We report cold-tune count + wall time per strategy, then dump
//! which variant each persisted key chose.
//!
//! Uses the spike-038 fresh-output InputGenerator so the (irrelevant-here) output isn't
//! corrupted. Clears the persistent autotune cache at start so first-touch = true cold.
//!
//! Run: cargo run --release --features rocm --example spike039_autotune_key_cache_thrash

#[cfg(not(feature = "rocm"))]
fn main() {
    eprintln!("spike039 requires --features rocm");
}

#[cfg(feature = "rocm")]
use cubecl::prelude::*;

#[cfg(feature = "rocm")]
const HIST_LDS_MAX: usize = 512;

#[cfg(feature = "rocm")]
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn build_rp(
    resident_bins: &Array<u32>,
    leaf_rows: &Array<u32>,
    ord_g: &Array<f32>,
    ord_h: &Array<f32>,
    slot_off: &Array<u32>,
    num_data: usize,
    feat_len: u32,
    out: &mut Array<Atomic<f32>>,
) {
    let f = CUBE_POS_X as usize;
    let p = CUBE_POS_Y as usize;
    let np = CUBE_COUNT_Y as usize;
    let cd = CUBE_DIM as usize;
    let fl = feat_len as usize;
    let r = ord_g.len();
    let base = slot_off[f] as usize;
    let col = f * num_data;
    let sub = SharedMemory::<Atomic<f32>>::new(HIST_LDS_MAX);
    let mut c = UNIT_POS as usize;
    while c < fl {
        sub[c].store(0.0f32);
        c += cd;
    }
    sync_cube();
    let stride = np * cd;
    let mut k = p * cd + UNIT_POS as usize;
    while k < r {
        let row = leaf_rows[k] as usize;
        let bin = resident_bins[col + row] as usize;
        let ti = bin * 2;
        sub[ti].fetch_add(ord_g[k]);
        sub[ti + 1].fetch_add(ord_h[k]);
        k += stride;
    }
    sync_cube();
    let mut m = UNIT_POS as usize;
    while m < fl {
        out[base + m].fetch_add(sub[m].load());
        m += cd;
    }
}

#[cfg(feature = "rocm")]
mod tune_impl {
    use super::*;
    use cubecl::client::ComputeClient;
    use cubecl::server::Handle;
    use cubecl::tune::{local_tuner, InputGenerator, LocalTuner, Tunable, TunableSet, TuneInputs};
    use lgbm_compute::runtime::RocmRuntime;
    use std::fmt::Display;

    pub const NUM_BIN: usize = 256;
    pub const CUBE_DIM: u32 = 256;

    /// `a` carries the strategy-encoded discriminant (exact rows / log2 bucket / 0).
    #[derive(Clone, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    pub struct HistKey {
        pub a: usize,
        pub feats: usize,
    }
    impl cubecl::tune::AutotuneKey for HistKey {}
    impl Display for HistKey {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "HistKey(a{},f{})", self.a, self.feats)
        }
    }

    pub struct FreshOutGenerator {
        pub client: ComputeClient<RocmRuntime>,
        pub slot_len: usize,
    }
    impl InputGenerator<HistKey, Vec<Handle>> for FreshOutGenerator {
        fn generate<'a>(
            &self,
            _key: &HistKey,
            inputs: &<Vec<Handle> as TuneInputs>::At<'a>,
        ) -> <Vec<Handle> as TuneInputs>::At<'a> {
            let mut v = inputs.clone();
            v[5] = self.client.create_from_slice(f32::as_bytes(&vec![0.0f32; self.slot_len]));
            v
        }
    }

    pub static TUNER_EXACT: LocalTuner<HistKey, String> = local_tuner!("hist039_exact");
    pub static TUNER_BUCKET: LocalTuner<HistKey, String> = local_tuner!("hist039_bucket");
    pub static TUNER_FIXED: LocalTuner<HistKey, String> = local_tuner!("hist039_fixed");

    /// `key_a`: maps the (captured) leaf row count to the strategy's key discriminant.
    pub fn make_set(
        client: ComputeClient<RocmRuntime>,
        rows: usize,
        feats: usize,
        num_data: usize,
        feat_len: u32,
        slot_len: usize,
        key_a: usize,
    ) -> TunableSet<HistKey, Vec<Handle>, ()> {
        let kg = move |_: &Vec<Handle>| HistKey { a: key_a, feats };
        let launch_at = move |client: &ComputeClient<RocmRuntime>, inputs: &[Handle], p: u32| unsafe {
            build_rp::launch(
                client,
                CubeCount::Static(feats as u32, p, 1),
                CubeDim::new_1d(CUBE_DIM),
                ArrayArg::from_raw_parts(inputs[0].clone(), feats * num_data),
                ArrayArg::from_raw_parts(inputs[1].clone(), rows),
                ArrayArg::from_raw_parts(inputs[2].clone(), rows),
                ArrayArg::from_raw_parts(inputs[3].clone(), rows),
                ArrayArg::from_raw_parts(inputs[4].clone(), feats),
                num_data,
                feat_len,
                ArrayArg::from_raw_parts(inputs[5].clone(), slot_len),
            );
        };
        let c1 = client.clone();
        let c16 = client.clone();
        TunableSet::new(
            kg,
            FreshOutGenerator { client: client.clone(), slot_len },
        )
        .with(Tunable::new("build_rp_P1", move |inputs: Vec<Handle>| {
            launch_at(&c1, &inputs, 1);
            Ok::<(), String>(())
        }))
        .with(Tunable::new("build_rp_P16", move |inputs: Vec<Handle>| {
            launch_at(&c16, &inputs, 16);
            Ok::<(), String>(())
        }))
    }
}

#[cfg(feature = "rocm")]
fn main() {
    use cubecl::server::Handle;
    use cubecl::tune::LocalTuner;
    use lgbm_compute::runtime::rocm_client;
    use std::time::Instant;
    use tune_impl::*;

    // Clear the persistent autotune cache so first-touch of each key is a TRUE cold tune.
    let _ = std::fs::remove_dir_all("target/autotune");

    let feats: usize = 50;
    let root: usize = 200_000;
    let num_data = root; // resident column stride sized for the largest leaf
    let feat_len: u32 = (NUM_BIN * 2) as u32;
    let slot_len = feats * NUM_BIN * 2;

    // ---- Simulate a tree's per-NODE row counts (each node builds a histogram). ----
    // Repeatedly split the largest open node into two unequal children until all open
    // nodes are below the 12k resident gate; collect every SPLIT node's row count.
    let mut lcg: u64 = 0xDEADBEEF12345678;
    let mut frac = || {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        0.35 + 0.30 * ((lcg >> 40) as f64 / (1u64 << 24) as f64) // child fraction in [0.35,0.65]
    };
    let mut open = vec![root];
    let mut node_sizes: Vec<usize> = Vec::new();
    while let Some(pos) = open
        .iter()
        .enumerate()
        .filter(|&(_, &n)| n >= 12_000)
        .max_by_key(|&(_, &n)| n)
        .map(|(i, _)| i)
    {
        let n = open.remove(pos);
        node_sizes.push(n); // this node is split → it built a histogram
        let l = ((n as f64) * frac()) as usize;
        let r = n - l;
        open.push(l);
        open.push(r);
        if node_sizes.len() >= 40 {
            break;
        }
    }
    let distinct_exact = {
        let mut v = node_sizes.clone();
        v.sort_unstable();
        v.dedup();
        v.len()
    };
    println!("# spike039 cache-thrash  feats={feats}  {} histogram-build nodes (≥12k rows)", node_sizes.len());
    println!("# distinct exact sizes = {distinct_exact}  (range {}..{})", node_sizes.iter().min().unwrap(), node_sizes.iter().max().unwrap());

    // Device inputs sized for the root; per-node we just vary the `rows` length used.
    let client = rocm_client();
    let mut s: u64 = 0x9E3779B97F4A7C15;
    let mut next = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (s >> 33) as u32 };
    let bins: Vec<u32> = (0..feats * num_data).map(|_| next() % NUM_BIN as u32).collect();
    let leaf_rows: Vec<u32> = (0..root as u32).collect();
    let ord_g: Vec<f32> = (0..root).map(|i| ((i % 7) as f32) - 3.0).collect();
    let ord_h: Vec<f32> = vec![1.0f32; root];
    let slot_off: Vec<u32> = (0..feats as u32).map(|f| f * (NUM_BIN * 2) as u32).collect();
    let d_bins = client.create_from_slice(u32::as_bytes(&bins));
    let d_rows = client.create_from_slice(u32::as_bytes(&leaf_rows));
    let d_g = client.create_from_slice(f32::as_bytes(&ord_g));
    let d_h = client.create_from_slice(f32::as_bytes(&ord_h));
    let d_slot = client.create_from_slice(u32::as_bytes(&slot_off));
    let d_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; slot_len]));
    let handles: Vec<Handle> = vec![
        d_bins.clone(), d_rows.clone(), d_g.clone(), d_h.clone(), d_slot.clone(), d_out.clone(),
    ];

    // Run one keying strategy over the whole node sequence; return (cold_tunes, total_ms).
    let run = |label: &str, key_of: &dyn Fn(usize) -> usize,
               exec: &dyn Fn(usize, usize) -> std::time::Duration| {
        let mut cold_wall = std::time::Duration::ZERO;
        let mut warm_wall = std::time::Duration::ZERO;
        let mut total = std::time::Duration::ZERO;
        let mut seen = std::collections::HashSet::new();
        for &n in &node_sizes {
            let a = key_of(n);
            let first = seen.insert(a); // true ⇒ first time this key is tuned (cold)
            let dt = exec(n, a);
            total += dt;
            if first { cold_wall += dt } else { warm_wall += dt }
        }
        println!(
            "[{label:7}] distinct keys = {:2}/{}   cold-tune wall = {:>11?}   warm-hit wall = {:>9?}   TOTAL = {:?}",
            seen.len(), node_sizes.len(), cold_wall, warm_wall, total
        );
    };

    // NOTE: we deliberately build a FRESH TunableSet per call (not via `tuner.init`, which
    // memoizes the FIRST set per closure-type and would pin the first node's `rows`). The
    // tuner's CACHE lives in the LocalTuner keyed by AutotuneKey + a checksum of the set's
    // structure (stable across calls), so a fresh set with a seen key still hits the cache.
    let mk_exec = |tuner: &'static LocalTuner<HistKey, String>| {
        let handles = handles.clone();
        move |n: usize, a: usize| -> std::time::Duration {
            let set = std::sync::Arc::new(make_set(
                rocm_client(), n, feats, num_data, feat_len, slot_len, a,
            ));
            let t = Instant::now();
            tuner.execute(&"rocm:0".to_string(), &rocm_client(), set, handles.clone());
            t.elapsed()
        }
    };

    println!();
    run("EXACT", &|n| n, &mk_exec(&TUNER_EXACT));
    run("BUCKET", &|n| (usize::BITS - 1 - (n.max(1)).leading_zeros()) as usize, &mk_exec(&TUNER_BUCKET));
    run("FIXED", &|_| 0, &mk_exec(&TUNER_FIXED));

    // Dump which variant each persisted key chose (selection quality vs granularity).
    println!("\n# persisted winners (fastest variant per key):");
    if let Ok(rd) = std::fs::read_dir("target/autotune/0.10.0/rocm_0") {
        for e in rd.flatten() {
            let p = e.path();
            let txt = std::fs::read_to_string(&p).unwrap_or_default();
            let n_keys = txt.lines().count();
            let p16 = txt.matches("build_rp_P16\",\"index\":1").count();
            // crude: count which fastest_index appears
            let fast1 = txt.matches("\"fastest_index\":1").count();
            let fast0 = txt.matches("\"fastest_index\":0").count();
            println!(
                "  {:50}  keys={n_keys:2}  chose P16={fast1} P1={fast0}  (p16-bench seen {p16})",
                p.file_name().unwrap().to_string_lossy()
            );
        }
    }

    println!("\nVERDICT SIGNAL: EXACT cold≈distinct-sizes (storm); BUCKET cold≈#decades; FIXED cold=1.");
}
