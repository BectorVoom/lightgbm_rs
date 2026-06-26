//! Spike 037 — Does CubeCL 0.10's autotune (`cubecl::tune`) compile AND run on the
//! cubecl-hip (ROCm) backend, wrapping a real production-shaped histogram-build kernel?
//!
//! KILL QUESTION. The autotune manual
//! (`cubecl_manual/manual/cubecl/12_autotuning.md`) only ever instantiates the tuner on
//! `CpuRuntime`. Our GPU path is `cubecl::hip::HipRuntime`. This spike wraps TWO launch
//! configs of a faithful row-partitioned LDS-atomic build kernel (P=1, byte-identical to
//! the production one-cube-per-feature kernel, vs P=16, the spike-007 occupancy win) as
//! `Tunable`s and drives a `LocalTuner` on `rocm_client()`.
//!
//! It also records THREE API divergences from the manual that only surface against the
//! real 0.10 source (the manual is idealized / internally inconsistent):
//!   1. `TunableSet::new`'s FIRST closure is the KEY GENERATOR — it must return the
//!      `AutotuneKey` (`HistKey`), NOT a `String` (the manual returns `"axpy-tune"`).
//!   2. `LocalTuner::execute(id, …)`'s first arg is the persistent-cache *namespace ID*
//!      (`Display`), NOT the `AutotuneKey`. The key is generated INTERNALLY from inputs.
//!   3. The `AutotuneKey` trait alias requires `serde::{Serialize, DeserializeOwned}`
//!      whenever the persistent disk cache is active (`std_io` cfg = always, on linux).
//!
//! Run: cargo run --release --features rocm --example spike037_autotune_hip_feasibility

#[cfg(not(feature = "rocm"))]
fn main() {
    eprintln!("spike037 requires --features rocm (cubecl-hip / local AMD device)");
}

#[cfg(feature = "rocm")]
use cubecl::prelude::*;

/// LDS sub-hist cap: 512 f32 cells = 256 bins (grad+hess interleaved) = 2 KiB.
#[cfg(feature = "rocm")]
const HIST_LDS_MAX: usize = 512;

/// Faithful row-partitioned LDS resident build (copied from `gpu_row_partition.rs`,
/// spike-007). `CubeCount = (num_features, P)`; cube `(f, p)` owns a private LDS
/// sub-hist, strides its partition's rows into it, then atomic-merges to feature f's
/// global slot. P=1 reduces EXACTLY to `construct_leaf_hist_resident_lds_kernel`.
///
/// NOTE: `out: &mut Array<Atomic<f32>>` ACCUMULATES — re-launching into the same buffer
/// adds again. This is the in-place-mutation hazard the manual warns about; spike-038
/// explores what it does to autotune benchmarking.
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
    use cubecl::server::Handle;
    use cubecl::tune::{local_tuner, CloneInputGenerator, LocalTuner, Tunable, TunableSet};
    use lgbm_compute::runtime::RocmRuntime;
    use std::fmt::Display;

    pub const NUM_BIN: usize = 256;
    pub const CUBE_DIM: u32 = 256;

    /// (1) The AutotuneKey. Persistent cache is active (`std_io`) ⇒ serde is REQUIRED.
    #[derive(Clone, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    pub struct HistKey {
        pub rows: usize,
        pub feats: usize,
        pub bins: usize,
    }
    impl cubecl::tune::AutotuneKey for HistKey {}
    impl Display for HistKey {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "HistKey(r{},f{},b{})", self.rows, self.feats, self.bins)
        }
    }

    /// One global static tuner per logical operation (cache namespace `…-hist037`).
    pub static TUNER: LocalTuner<HistKey, String> = local_tuner!("hist037");

    /// Build the TunableSet wrapping P=1 and P=16 launches of the SAME kernel.
    /// Inputs are `Vec<Handle>` ordered: [bins, rows, g, h, slot, out].
    pub fn make_set(
        client: cubecl::client::ComputeClient<RocmRuntime>,
        rows: usize,
        feats: usize,
        num_data: usize,
        feat_len: u32,
        slot_len: usize,
    ) -> TunableSet<HistKey, Vec<Handle>, ()> {
        // (1) KEY GENERATOR — returns the AutotuneKey, derived from the (captured) dims.
        //     The manual erroneously returns a String here.
        let key_gen = move |_inputs: &Vec<Handle>| HistKey {
            rows,
            feats,
            bins: NUM_BIN,
        };

        let launch_at = move |client: &cubecl::client::ComputeClient<RocmRuntime>,
                              inputs: &[Handle],
                              p: u32| {
            // CloneInputGenerator hands us the SAME device handles every rep (clone =
            // ref-count bump, NOT a buffer copy) — see spike-038 for why that matters.
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
        };

        let c1 = client.clone();
        let c16 = client.clone();
        TunableSet::new(key_gen, CloneInputGenerator)
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
    use lgbm_compute::runtime::rocm_client;
    use tune_impl::*;

    // ----- Problem shape: a large single-feature-group leaf (occupancy-relevant). -----
    let rows: usize = 200_000;
    let feats: usize = 50;
    let num_data = rows; // resident column stride
    let feat_len: u32 = (NUM_BIN * 2) as u32; // 512 active LDS cells
    let slot_len = feats * NUM_BIN * 2;

    // Deterministic inputs via a tiny LCG (no rand dep).
    let mut s: u64 = 0x9E3779B97F4A7C15;
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 33) as u32
    };
    let bins: Vec<u32> = (0..feats * num_data).map(|_| next() % NUM_BIN as u32).collect();
    let leaf_rows: Vec<u32> = (0..rows as u32).collect();
    let ord_g: Vec<f32> = (0..rows).map(|i| ((i % 7) as f32) - 3.0).collect();
    let ord_h: Vec<f32> = (0..rows).map(|_| 1.0f32).collect();
    let slot_off: Vec<u32> = (0..feats as u32).map(|f| f * (NUM_BIN * 2) as u32).collect();

    let client = rocm_client();
    let d_bins = client.create_from_slice(u32::as_bytes(&bins));
    let d_rows = client.create_from_slice(u32::as_bytes(&leaf_rows));
    let d_g = client.create_from_slice(f32::as_bytes(&ord_g));
    let d_h = client.create_from_slice(f32::as_bytes(&ord_h));
    let d_slot = client.create_from_slice(u32::as_bytes(&slot_off));
    let d_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; slot_len]));

    let handles: Vec<Handle> = vec![
        d_bins.clone(),
        d_rows.clone(),
        d_g.clone(),
        d_h.clone(),
        d_slot.clone(),
        d_out.clone(),
    ];

    println!("# spike037 autotune-on-hip feasibility  rows={rows} feats={feats} bins={NUM_BIN}");
    println!("# wrapping build_rp at P=1 (==production) and P=16 (spike-007 occupancy)");

    let set = TUNER.init(move || {
        make_set(client.clone(), rows, feats, num_data, feat_len, slot_len)
    });

    // (2) execute's first arg is the cache-NAMESPACE ID (a Display), NOT the AutotuneKey.
    //     The AutotuneKey is generated internally from `handles` by `key_gen`.
    let device_id = "rocm:0".to_string();

    println!("\n[cold] TUNER.execute → benchmarks both variants, caches the winner …");
    let t0 = std::time::Instant::now();
    TUNER.execute(&device_id, &rocm_client(), set.clone(), handles.clone());
    println!("[cold] returned in {:?} (includes JIT compile + benchmarking)", t0.elapsed());

    // Warm in-process: second execute with the same key must HIT the cache (no re-bench).
    let t1 = std::time::Instant::now();
    TUNER.execute(&device_id, &rocm_client(), set.clone(), handles.clone());
    let warm = t1.elapsed();
    println!("[warm] second execute (same key) returned in {warm:?} (cache hit ⇒ ≪ cold)");

    // Read the output buffer. NOTE: because autotune ran each variant multiple times into
    // the SAME `d_out` (CloneInputGenerator shares the handle), this value is ACCUMULATED
    // garbage, not a single clean histogram — the spike-038 hazard, flagged here.
    let out_bytes = rocm_client().read_one_unchecked(d_out.clone());
    let got = f32::from_bytes(&out_bytes);
    let maxv = got.iter().cloned().fold(0.0f32, f32::max);
    let nonzero = got.iter().filter(|&&x| x != 0.0).count();
    println!(
        "\n[out] {} non-zero cells, max grad-cell = {:.1} (ACCUMULATED across benchmark reps — see spike-038)",
        nonzero, maxv
    );
    println!("\nVERDICT SIGNAL: if you see two benchmarked variants + a cache hit above, autotune RUNS on hip.");
}
