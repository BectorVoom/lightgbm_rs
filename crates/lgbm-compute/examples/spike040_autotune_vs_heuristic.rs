//! Spike 040 — Autotune's MEASURED row-partition pick vs the shipped core-count heuristic
//! (`row_partition_count`), on this 8-CU APU. Does autotune match, beat, or lose to the
//! hand-tuned default?
//!
//! Setup discovery (the interesting part): the production heuristic resolves
//! `target_cubes = num_cu × CUBES_PER_CU = 8 × 8 = 64`, then `P = clamp(target/num_features,
//! 1, 16)`, and additionally gates on `leaf_rows ≥ ROWPART_MIN_LEAF = 256_000`. At the
//! production width (50 features) that yields **P = 1 for essentially every leaf** — the
//! 8-CU correction effectively DISABLES row-partitioning here. But spikes 037–039's
//! autotuner repeatedly measured **P = 16 faster**. This spike settles it: a rigorous
//! P-sweep (device-time median, multi-restart) vs the heuristic's pick vs autotune's pick.
//!
//! Bench discipline (CONVENTIONS, spikes 017–019): accumulate N launches into ONE reused
//! `out` + a single `read_one_unchecked` (compute-throughput), interleaved median over reps,
//! judge SIGN only (the spoofed APU confounds absolute Mr/s). Run across ≥2 process restarts.
//!
//! Run (twice, for restart stability):
//!   cargo run --release --features rocm --example spike040_autotune_vs_heuristic

#[cfg(not(feature = "rocm"))]
fn main() {
    eprintln!("spike040 requires --features rocm");
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
    pub const PSET: [u32; 5] = [1, 4, 8, 16, 32];

    #[derive(Clone, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    pub struct HistKey {
        pub bucket: u32,
        pub feats: usize,
    }
    impl cubecl::tune::AutotuneKey for HistKey {}
    impl Display for HistKey {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "HistKey(b{},f{})", self.bucket, self.feats)
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

    pub static TUNER: LocalTuner<HistKey, String> = local_tuner!("hist040");

    #[allow(clippy::too_many_arguments)]
    pub fn launch_p(
        client: &ComputeClient<RocmRuntime>,
        inputs: &[Handle],
        rows: usize,
        feats: usize,
        num_data: usize,
        feat_len: u32,
        slot_len: usize,
        p: u32,
    ) {
        unsafe {
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
        }
    }

    /// Autotune over the full PSET; returns the chosen P (read back from the cache log).
    pub fn autotune_pick(
        client: ComputeClient<RocmRuntime>,
        handles: Vec<Handle>,
        rows: usize,
        feats: usize,
        num_data: usize,
        feat_len: u32,
        slot_len: usize,
        bucket: u32,
    ) -> u32 {
        let kg = move |_: &Vec<Handle>| HistKey { bucket, feats };
        let mut set = TunableSet::new(kg, FreshOutGenerator { client: client.clone(), slot_len });
        for &p in PSET.iter() {
            let c = client.clone();
            set = set.with(Tunable::new(&format!("P{p}"), move |inputs: Vec<Handle>| {
                launch_p(&c, &inputs, rows, feats, num_data, feat_len, slot_len, p);
                Ok::<(), String>(())
            }));
        }
        TUNER.execute(&"rocm:0".to_string(), &client, std::sync::Arc::new(set), handles);
        // Read the winning tunable name (e.g. "P16") from the persisted cache.
        read_winner(bucket, feats).unwrap_or(0)
    }

    fn read_winner(bucket: u32, feats: usize) -> Option<u32> {
        let dir = "target/autotune/0.10.0/rocm_0";
        let want = format!("\"key\":{{\"bucket\":{bucket},\"feats\":{feats}}}");
        for e in std::fs::read_dir(dir).ok()?.flatten() {
            let txt = std::fs::read_to_string(e.path()).ok()?;
            for line in txt.lines() {
                if line.contains(&want) {
                    // find fastest_index then the name at that index
                    if let Some(fi) = line.split("\"fastest_index\":").nth(1) {
                        let idx: usize = fi.trim_start().split([',', '}']).next()?.parse().ok()?;
                        return Some(PSET.get(idx).copied().unwrap_or(0));
                    }
                }
            }
        }
        None
    }
}

#[cfg(feature = "rocm")]
fn main() {
    use cubecl::server::Handle;
    use lgbm_compute::kernels::histogram::row_partition_count;
    use lgbm_compute::runtime::rocm_client;
    use std::time::Instant;
    use tune_impl::*;

    let _ = std::fs::remove_dir_all("target/autotune");

    let feats: usize = 50;
    let sizes: [usize; 4] = [50_000, 100_000, 200_000, 500_000];
    let max_rows = *sizes.iter().max().unwrap();
    let num_data = max_rows;
    let feat_len: u32 = (NUM_BIN * 2) as u32;
    let slot_len = feats * NUM_BIN * 2;

    let client = rocm_client();
    let mut s: u64 = 0x9E3779B97F4A7C15;
    let mut next = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (s >> 33) as u32 };
    let bins: Vec<u32> = (0..feats * num_data).map(|_| next() % NUM_BIN as u32).collect();
    let leaf_rows: Vec<u32> = (0..max_rows as u32).collect();
    let ord_g: Vec<f32> = (0..max_rows).map(|i| ((i % 7) as f32) - 3.0).collect();
    let ord_h: Vec<f32> = vec![1.0f32; max_rows];
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

    const ACC: usize = 20; // launches accumulated per timed sample
    const REPS: usize = 11;
    // Median device-time (ms) for a given (rows, P): ACC launches → one sync, REPS reps.
    let bench = |rows: usize, p: u32| -> f64 {
        // warmup
        for _ in 0..2 {
            for _ in 0..ACC {
                launch_p(&client, &handles, rows, feats, num_data, feat_len, slot_len, p);
            }
            let _ = client.read_one_unchecked(d_out.clone());
        }
        let mut ms: Vec<f64> = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let t = Instant::now();
            for _ in 0..ACC {
                launch_p(&client, &handles, rows, feats, num_data, feat_len, slot_len, p);
            }
            let _ = client.read_one_unchecked(d_out.clone());
            ms.push(t.elapsed().as_secs_f64() * 1e3 / ACC as f64);
        }
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ms[REPS / 2]
    };

    println!("# spike040 autotune-vs-heuristic  feats={feats}  (8-CU APU; ACC={ACC} REPS={REPS})");
    println!("# heuristic row_partition_count(50, n): target_cubes=64, MIN_LEAF=256k → P=1 at this width");
    println!("# {:>8} | {:>34} | best | heur | auto | t(heur)/t(best) | t(heur)/t(auto)", "rows", "P→median ms  [1, 4, 8, 16, 32]");

    for &rows in &sizes {
        let medians: Vec<f64> = PSET.iter().map(|&p| bench(rows, p)).collect();
        let (best_i, &best_ms) = medians
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let best_p = PSET[best_i];
        let heur_p = row_partition_count(feats, rows);
        let heur_ms = medians[PSET.iter().position(|&p| p == heur_p).unwrap_or(0)];
        let bucket = (usize::BITS - 1 - rows.max(1).leading_zeros()) as u32;
        let auto_p = autotune_pick(client.clone(), handles.clone(), rows, feats, num_data, feat_len, slot_len, bucket);
        let auto_ms = medians[PSET.iter().position(|&p| p == auto_p).unwrap_or(0)];
        let curve: Vec<String> = medians.iter().map(|m| format!("{m:.3}")).collect();
        println!(
            "  {rows:>7} | {:>34} |  P{best_p:<2} |  P{heur_p:<2} |  P{auto_p:<2} | {:>13.3}× | {:>13.3}×",
            curve.join(" "),
            heur_ms / best_ms,
            heur_ms / auto_ms,
        );
    }

    println!("\nVERDICT SIGNAL: if t(heur)/t(best) ≫ 1 AND auto≈best, autotune BEATS the 8-CU heuristic (which under-partitions to P=1).");
}
