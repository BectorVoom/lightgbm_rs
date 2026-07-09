//! Spike 079 — A/B the resident-bin upload HOIST (the spike-078 ledger's #1 lever).
//!
//! Spike-078 priced the on-device driver's per-grow re-upload of the IMMUTABLE binned
//! columns at 67–78% of the WHOLE remaining on-device-vs-host-cuda gap on real CUDA.
//! The fix mirrors quick-260621-p9v: the learner uploads ONCE per train (guarded by its
//! existing `resident_bins_uploaded` flag, re-armed by `with_features`) and PINS the
//! cache; the driver skips its per-grow upload only for a pinned, geometry-matching cache.
//!
//! This harness drives the FULL learner path (SerialTreeLearner::train, the production
//! seam — the pin only engages through the learner) and A/Bs via the escape hatch:
//!
//!   baseline: LGBM_ONDEVICE_BIN_HOIST=0  (per-grow re-upload, the 073/076/078 behavior)
//!   hoist   : (default)                  (once-per-train upload + pinned skip)
//!
//! Gates:
//!   (1) BIT-EXACT: the tree-stream fingerprint (Debug-serialized trees, all trains,
//!       INCLUDING a mid-run feature-set swap) must be IDENTICAL across arms.
//!   (2) The ONDEV_GROW `upload` bucket collapses to ~1/N_TREES of baseline.
//!   (3) The feature-set swap re-arms the guard (pin invalidation is exercised, not
//!       just asserted).
//!
//! Run (local hip):
//!   LGBM_CUDA_ON_DEVICE=1 LGBM_PHASE_PROF=1 [LGBM_ONDEVICE_BIN_HOIST=0] \
//!     cargo run --release -p lgbm-treelearner --features lgbm-compute/rocm \
//!     --example spike079_bin_hoist_ab

use lgbm_compute::kernels::grow_driver::on_device_grow_phase_take;
use lgbm_compute::{BinColumn, GainConfig};
use lgbm_dataset::bin_mapper::{BinType, MissingType};
use lgbm_treelearner::{FeatureColumn, SerialTreeLearner};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Instant;

const NUM_BIN: u32 = 255;
const NUM_LEAVES: i32 = 31;
const N_TREES: usize = 20;
const N_TREES_SWAP: usize = 3; // extra trees after the feature-set swap

struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

fn corpus(seed: u64, num_data: usize, num_features: usize) -> Vec<FeatureColumn> {
    let mut rng = Lcg(seed);
    (0..num_features)
        .map(|fi| {
            let bins: Vec<u32> = (0..num_data).map(|_| rng.next_u32() % NUM_BIN).collect();
            FeatureColumn {
                bins: BinColumn::new(bins, NUM_BIN),
                num_bin: NUM_BIN,
                offset: 1, // offset_for_most_freq_bin(0)
                min_bin: 0,
                max_bin: NUM_BIN - 1,
                default_bin: NUM_BIN,
                most_freq_bin: 0,
                missing_type: MissingType::None,
                bin_upper_bound: (0..NUM_BIN).map(|b| b as f64 + 0.5).collect(),
                real_feature_index: fi as i32,
                bin_type: BinType::Numerical,
                bin_to_category: Vec::new(),
            }
        })
        .collect()
}

fn grads(seed: u64, num_data: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = Lcg(seed);
    ((0..num_data).map(|_| rng.next_f32()).collect(), vec![1.0f32; num_data])
}

fn main() {
    assert_eq!(
        std::env::var("LGBM_CUDA_ON_DEVICE").as_deref(),
        Ok("1"),
        "set LGBM_CUDA_ON_DEVICE=1 — this A/B drives the on-device learner path"
    );
    if std::env::var("LGBM_PHASE_PROF").as_deref() != Ok("1") {
        eprintln!("WARN: set LGBM_PHASE_PROF=1 — the upload ledger is inert without it");
    }
    let hoist = std::env::var("LGBM_ONDEVICE_BIN_HOIST").map(|v| v != "0").unwrap_or(true);
    println!("arm: {}", if hoist { "HOIST (once-per-train pinned)" } else { "BASELINE (per-grow re-upload)" });

    #[cfg(feature = "rocm")]
    run_gpu(hoist);
    #[cfg(not(feature = "rocm"))]
    eprintln!("build with --features lgbm-compute/rocm — the CPU anchor never takes the resident arm");
}

#[cfg(feature = "rocm")]
fn run_gpu(hoist: bool) {
    let client = lgbm_compute::runtime::rocm_client();
    let backend = lgbm_compute::RocmBackend::default();

    for &(num_data, num_features) in &[(500_000usize, 50usize), (100_000, 500)] {
        let feats_a = corpus(0x5eed_0079, num_data, num_features);
        let feats_b = corpus(0xb5eed_0079, num_data, num_features); // the swap set
        let mut hasher = DefaultHasher::new();
        let mut upload_ms_total = 0.0f64;
        let mut wall_ms_total = 0.0f64;
        let mut per_train_upload = Vec::new();

        let mut learner =
            SerialTreeLearner::new(&backend, &client, GainConfig::default(), NUM_LEAVES, -1)
                .with_features(feats_a);
        let _ = on_device_grow_phase_take();
        for iter in 0..N_TREES {
            let (g, h) = grads(0x9e3779b9 ^ iter as u64, num_data);
            let t = Instant::now();
            let tree = learner.train(&g, &h, iter == 0).expect("on-device train");
            wall_ms_total += t.elapsed().as_secs_f64() * 1e3;
            let led = on_device_grow_phase_take();
            assert!(led.wall > 0, "grow ledger empty — on-device path not taken (iter {iter})");
            let up = led.upload as f64 / 1e6;
            upload_ms_total += up;
            per_train_upload.push(up);
            format!("{tree:?}").hash(&mut hasher);
        }
        // ---- feature-set SWAP (pin invalidation): with_features re-arms the guard. ----
        learner = learner.with_features(feats_b);
        let mut swap_first_upload = 0.0f64;
        for iter in 0..N_TREES_SWAP {
            let (g, h) = grads(0x51ca_5a1b ^ iter as u64, num_data);
            let tree = learner.train(&g, &h, iter == 0).expect("post-swap train");
            let led = on_device_grow_phase_take();
            if iter == 0 {
                swap_first_upload = led.upload as f64 / 1e6;
            }
            format!("{tree:?}").hash(&mut hasher);
        }

        let fingerprint = hasher.finish();
        println!(
            "SPIKE079_JSON {{\"shape\": \"{num_data}x{num_features}\", \"hoist\": {hoist}, \
             \"trees\": {N_TREES}, \"wall_ms_total\": {wall_ms_total:.1}, \
             \"upload_ms_total\": {upload_ms_total:.1}, \
             \"upload_ms_first\": {:.1}, \"upload_ms_rest_max\": {:.1}, \
             \"swap_first_upload_ms\": {swap_first_upload:.1}, \
             \"tree_stream_fingerprint\": \"{fingerprint:016x}\"}}",
            per_train_upload.first().copied().unwrap_or(0.0),
            per_train_upload[1..].iter().copied().fold(0.0f64, f64::max),
        );
    }
    println!("=== done ===");
}
