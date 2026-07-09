//! Spike 077 — the on-device growth loop's PHASE LEDGER, measured locally.
//!
//! Post-spike-076 the on-device-vs-host-cuda residual (~4.9s @500k×50 on real CUDA) is
//! ~95% UNATTRIBUTED inside `grow_tree_on_device_resident` — a `phase_prof` black box
//! (`in_learner_other=100%` by design). This example drives the resident lane directly
//! (the spike-071 harness shape) and reads the new SPIKE-077 per-phase ledger after every
//! grow: setup / upload / rootfold / build / subtract / scan / pick / partition /
//! treesplit / reduce / tail, plus `host_other = wall − Σ` (the honesty check).
//!
//! Two modes:
//!   free-run (default)      — TRUE wall decomposition; async phases hold submission time
//!                             only (their device time drains inside scan/pick — the
//!                             spike-015 aliasing contract).
//!   LGBM_GROW_DRAIN=1       — de-aliased device attribution (queue blocked empty inside
//!                             each phase's own timer); ranks phases, distorts the wall.
//!
//! Run (local hip):
//!   LGBM_PHASE_PROF=1 cargo run --release -p lgbm-compute --features rocm \
//!     --example spike077_growth_ledger
//!   LGBM_PHASE_PROF=1 LGBM_GROW_DRAIN=1 cargo run --release -p lgbm-compute \
//!     --features rocm --example spike077_growth_ledger

use lgbm_compute::kernels::grow_driver::{
    grow_tree_on_device_driver, on_device_grow_phase_take, on_device_launch_count_take,
    on_device_sync_count_take, GrowPhaseNs,
};
use lgbm_compute::{BinColumn, GrowFeature};
use lgbm_dataset::bin_mapper::{BinType, MissingType};
use std::time::Instant;

/// `offset_for_most_freq_bin(0)` == 1 (compacted convention; helper lives above this crate).
const OFFSET_MFB_0: i32 = 1;
const NUM_BIN: u32 = 255; // U8 production-default width
const NUM_LEAVES: i32 = 31; // the A/B default
const WARMUP: usize = 2;
const REPS: usize = 10;

/// Deterministic LCG (no rand dep, spike convention).
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

fn corpus(num_data: usize, num_features: usize) -> (Vec<GrowFeature>, Vec<f32>, Vec<f32>) {
    let mut rng = Lcg(0x5eed_0077);
    let features: Vec<GrowFeature> = (0..num_features)
        .map(|fi| {
            let bins: Vec<u32> = (0..num_data).map(|_| rng.next_u32() % NUM_BIN).collect();
            GrowFeature {
                bins: BinColumn::new(bins, NUM_BIN),
                num_bin: NUM_BIN,
                offset: OFFSET_MFB_0,
                min_bin: 0,
                max_bin: NUM_BIN - 1,
                default_bin: NUM_BIN,
                most_freq_bin: 0,
                missing_type: MissingType::None,
                bin_upper_bound: (0..NUM_BIN).map(|b| b as f64 + 0.5).collect(),
                real_feature_index: fi as i32,
                bin_type: BinType::Numerical,
                bin_to_category: Vec::new(),
                cat_smooth: 10.0,
                cat_l2: 10.0,
                max_cat_threshold: 32,
                max_cat_to_onehot: 4,
                min_data_per_group: 100,
            }
        })
        .collect();
    let gradients: Vec<f32> = (0..num_data).map(|_| rng.next_f32()).collect();
    let hessians: Vec<f32> = vec![1.0f32; num_data];
    (features, gradients, hessians)
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

const PHASES: [&str; 11] = [
    "setup", "upload", "rootfold", "build", "subtract", "scan", "pick", "partition",
    "treesplit", "reduce", "tail",
];

fn components(g: &GrowPhaseNs) -> [u64; 11] {
    [
        g.setup, g.upload, g.rootfold, g.build, g.subtract, g.scan, g.pick, g.partition,
        g.treesplit, g.reduce, g.tail,
    ]
}

fn main() {
    if std::env::var("LGBM_PHASE_PROF").as_deref() != Ok("1") {
        eprintln!("WARN: set LGBM_PHASE_PROF=1 — the ledger is inert without it");
    }
    let drain = std::env::var("LGBM_GROW_DRAIN").as_deref() == Ok("1");
    println!("mode: {}", if drain { "DRAIN (de-aliased device attribution)" } else { "free-run (true wall decomposition)" });

    #[cfg(feature = "rocm")]
    {
        let client = lgbm_compute::runtime::rocm_client();
        let backend = lgbm_compute::GpuBackend::<cubecl::hip::HipRuntime>::default();
        run(&backend, &client, "hip(GpuBackend resident lane)");
    }
    #[cfg(not(feature = "rocm"))]
    {
        let client = lgbm_compute::runtime::cpu_client();
        let backend = lgbm_compute::CpuBackend;
        run(&backend, &client, "cpu(ANCHOR lane — resident ledger NOT exercised)");
    }
}

fn run<B: lgbm_compute::Backend<Runtime = R>, R: cubecl::Runtime>(
    backend: &B,
    client: &cubecl::prelude::ComputeClient<R>,
    label: &str,
) {
    println!("=== spike077 on-device growth-loop phase ledger on [{label}] ===");
    for &(num_data, num_features) in &[(500_000usize, 50usize), (100_000, 500)] {
        let (features, gradients, hessians) = corpus(num_data, num_features);
        // per-phase medians: [phase][rep]
        let mut phase_ms: Vec<Vec<f64>> = vec![Vec::new(); PHASES.len()];
        let mut walls = Vec::new();
        let mut others = Vec::new();
        let mut covs = Vec::new();
        for rep in 0..(WARMUP + REPS) {
            let _ = on_device_grow_phase_take();
            let _ = on_device_sync_count_take();
            let _ = on_device_launch_count_take();
            let t = Instant::now();
            let (tree, _layout) = grow_tree_on_device_driver(
                backend, client, &gradients, &hessians, &features, NUM_LEAVES, -1,
            )
            .expect("resident grow");
            let outer_ms = t.elapsed().as_secs_f64() * 1e3;
            let g = on_device_grow_phase_take();
            let syncs = on_device_sync_count_take();
            let launches = on_device_launch_count_take();
            assert!(tree.num_leaves > 1, "corpus must split");
            let comps = components(&g);
            let sum: u64 = comps.iter().sum();
            let wall_ms = g.wall as f64 / 1e6;
            let other_ms = g.wall.saturating_sub(sum) as f64 / 1e6;
            let cov = if g.wall > 0 { sum as f64 * 100.0 / g.wall as f64 } else { 0.0 };
            if rep >= WARMUP {
                for (i, &c) in comps.iter().enumerate() {
                    phase_ms[i].push(c as f64 / 1e6);
                }
                walls.push(wall_ms);
                others.push(other_ms);
                covs.push(cov);
            }
            let detail: Vec<String> = PHASES
                .iter()
                .zip(comps.iter())
                .map(|(n, &c)| format!("{n}={:.1}", c as f64 / 1e6))
                .collect();
            println!(
                "  [{num_data}x{num_features} rep={rep}{}] outer={outer_ms:.1}ms wall={wall_ms:.1}ms {} host_other={other_ms:.1} cov={cov:.1}% syncs={syncs} launches={launches} leaves={}",
                if rep < WARMUP { " warmup" } else { "" },
                detail.join(" "),
                tree.num_leaves,
            );
        }
        let w = median(&mut walls);
        let o = median(&mut others);
        let c = median(&mut covs);
        let med: Vec<f64> = phase_ms.iter_mut().map(|v| median(v)).collect();
        let json_phases: Vec<String> = PHASES
            .iter()
            .zip(med.iter())
            .map(|(n, m)| format!("\"{n}_ms\": {m:.2}"))
            .collect();
        println!(
            "SPIKE077_JSON {{\"shape\": \"{num_data}x{num_features}\", \"drain\": {}, \"wall_ms\": {w:.2}, {}, \"host_other_ms\": {o:.2}, \"coverage_pct\": {c:.1}}}",
            std::env::var("LGBM_GROW_DRAIN").as_deref() == Ok("1"),
            json_phases.join(", "),
        );
        // Ranked ledger (medians), largest first — the deliverable.
        let mut ranked: Vec<(&str, f64)> = PHASES.iter().copied().zip(med.iter().copied()).collect();
        ranked.push(("host_other", o));
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("  RANKED ({num_data}x{num_features}, median of {REPS}):");
        for (n, m) in &ranked {
            println!("    {n:<11} {m:>9.2}ms  {:.1}%", m * 100.0 / w);
        }
    }
    println!("=== done ===");
}
