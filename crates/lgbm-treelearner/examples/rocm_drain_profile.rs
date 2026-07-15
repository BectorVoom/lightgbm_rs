//! Local-ROCm drain-mode phase profiler for the resident on-device grow driver.
//!
//! Measures WHERE on-device train time goes on the real gfx1152, at a
//! representative shape, with whatever the CURRENT shipped defaults are (round-11
//! pargain-on-rocm included). Single process (no arm A/B) — it just trains
//! `N_TREES` warm reps through the resident path and dumps the per-tree-averaged
//! drain ledger so the dominant device-compute bucket is unambiguous.
//!
//! Run (local hip):
//!   ROCM env (see memory) + \
//!   LGBM_CUDA_ON_DEVICE=1 LGBM_PHASE_PROF=1 LGBM_GROW_DRAIN=1 \
//!     cargo run --release -p lgbm-treelearner --features rocm --example rocm_drain_profile
//!
//! Knobs (env): LGBM_PROF_NDATA (100000), LGBM_PROF_NFEAT (50),
//!   LGBM_PROF_NTREES (100), LGBM_PROF_WARMUP (1).

// This harness's real body lives behind `#[cfg(feature = "rocm")]` blocks in `main`;
// in the default (non-rocm) build those blocks vanish, so the helpers/consts/imports
// they use look "unused". Silence those false positives without dropping the GPU arm —
// same idiom as `grow_driver.rs`'s `cfg_attr(not(feature = "gpu"), allow(dead_code))`.
#![cfg_attr(not(feature = "rocm"), allow(dead_code, unused_imports))]

#[cfg(feature = "rocm")]
use lgbm_compute::kernels::grow_driver::{on_device_grow_phase_take, GrowPhaseNs};
#[cfg(feature = "rocm")]
use lgbm_compute::{BinColumn, GainConfig};
#[cfg(feature = "rocm")]
use lgbm_dataset::bin_mapper::{BinType, MissingType};
#[cfg(feature = "rocm")]
use lgbm_treelearner::{FeatureColumn, SerialTreeLearner};

const NUM_LEAVES: i32 = 31;

/// Deeper-profile knob: number of bins (env `LGBM_PROF_NBIN`, default 255). Sweeping this
/// isolates the O(num_bin) scan cost (serial prefix accumulate + staging) from fixed
/// per-cube overhead — if scan time scales ~linearly with num_bin, the serial accumulate
/// dominates and a parallel-prefix redesign pays off.
#[cfg(feature = "rocm")]
fn num_bin() -> u32 {
    std::env::var("LGBM_PROF_NBIN").ok().and_then(|v| v.parse().ok()).unwrap_or(255)
}

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

#[cfg(feature = "rocm")]
fn corpus(seed: u64, num_data: usize, num_features: usize) -> Vec<FeatureColumn> {
    let nb = num_bin();
    let mut rng = Lcg(seed);
    (0..num_features)
        .map(|fi| {
            let bins: Vec<u32> = (0..num_data).map(|_| rng.next_u32() % nb).collect();
            FeatureColumn {
                bins: BinColumn::new(bins, nb),
                num_bin: nb,
                offset: 1,
                min_bin: 0,
                max_bin: nb - 1,
                default_bin: nb,
                most_freq_bin: 0,
                missing_type: MissingType::None,
                bin_upper_bound: (0..nb).map(|b| b as f64 + 0.5).collect(),
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

#[cfg(feature = "rocm")]
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

#[cfg(feature = "rocm")]
fn add_ledger(acc: &mut GrowPhaseNs, l: &GrowPhaseNs) {
    acc.wall += l.wall;
    acc.setup += l.setup;
    acc.upload += l.upload;
    acc.rootfold += l.rootfold;
    acc.build += l.build;
    acc.subtract += l.subtract;
    acc.scan += l.scan;
    acc.pick += l.pick;
    acc.partition += l.partition;
    acc.treesplit += l.treesplit;
    acc.reduce += l.reduce;
    acc.tail += l.tail;
}

#[cfg(feature = "rocm")]
fn main() {
    assert_eq!(
        std::env::var("LGBM_CUDA_ON_DEVICE").as_deref(),
        Ok("1"),
        "set LGBM_CUDA_ON_DEVICE=1 to drive the on-device path"
    );
    let ndata = env_usize("LGBM_PROF_NDATA", 100_000);
    let nfeat = env_usize("LGBM_PROF_NFEAT", 50);
    let ntrees = env_usize("LGBM_PROF_NTREES", 100);
    let warmup = env_usize("LGBM_PROF_WARMUP", 1);

    let client = lgbm_compute::runtime::rocm_client();
    let backend = lgbm_compute::RocmBackend::default();
    let feats = corpus(0x5eed_0031, ndata, nfeat);

    let mut learner =
        SerialTreeLearner::new(&backend, &client, GainConfig::default(), NUM_LEAVES, -1)
            .with_features(feats);

    let total_reps = warmup + 1;
    let mut steady = GrowPhaseNs::default();
    let mut steady_trees = 0u64;
    for rep in 0..total_reps {
        let _ = on_device_grow_phase_take();
        for iter in 0..ntrees {
            let (g, h) = grads(0x9e37_79b9 ^ ((rep * ntrees + iter) as u64), ndata);
            let _tree = learner.train(&g, &h, iter == 0).expect("on-device resident train");
            let led = on_device_grow_phase_take();
            // Only the LAST (warm) rep counts; drop tree 0 (once-per-train setup).
            if rep == total_reps - 1 && iter > 0 {
                add_ledger(&mut steady, &led);
                steady_trees += 1;
            }
        }
    }

    let ms = |ns: u64| ns as f64 / 1e6;
    let per = |ns: u64| ns as f64 / 1e6 / steady_trees as f64;
    let components = steady.setup
        + steady.upload
        + steady.rootfold
        + steady.build
        + steady.subtract
        + steady.scan
        + steady.pick
        + steady.partition
        + steady.treesplit
        + steady.reduce
        + steady.tail;
    let host_other = steady.wall.saturating_sub(components);

    println!("=== ROCm drain profile: {ndata}x{nfeat}, {ntrees} trees, {steady_trees} steady ===");
    println!("total steady wall: {:.1}ms  (per-tree {:.3}ms)", ms(steady.wall), per(steady.wall));
    let mut rows: Vec<(&str, u64)> = vec![
        ("scan", steady.scan),
        ("build", steady.build),
        ("partition", steady.partition),
        ("pick", steady.pick),
        ("treesplit", steady.treesplit),
        ("subtract", steady.subtract),
        ("reduce", steady.reduce),
        ("rootfold", steady.rootfold),
        ("upload", steady.upload),
        ("setup", steady.setup),
        ("tail", steady.tail),
        ("host_other", host_other),
    ];
    rows.sort_by(|a, b| b.1.cmp(&a.1));
    for (name, ns) in rows {
        let pct = 100.0 * ns as f64 / steady.wall as f64;
        println!("  {name:11} {:8.1}ms  ({pct:5.1}%)  per-tree {:.3}ms", ms(ns), per(ns));
    }
}

#[cfg(not(feature = "rocm"))]
fn main() {
    eprintln!("build with --features rocm to run the on-device drain profiler");
}
