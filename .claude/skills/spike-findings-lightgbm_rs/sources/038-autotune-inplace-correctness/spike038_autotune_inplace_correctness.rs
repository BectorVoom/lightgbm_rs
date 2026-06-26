//! Spike 038 — Does autotune's repeated benchmarking CORRUPT an accumulating kernel's
//! output, and is there a clean fix? (the 2nd kill question)
//!
//! The cubecl autotune manual (§3 NOTE) warns: "Because autotuning benchmarks the kernels
//! multiple times, an in-place mutation like this one will accumulate values repeatedly
//! during the cold run." Our histogram BUILD kernel accumulates grad/hess into a resident
//! `out` buffer (atomic `fetch_add`); our partition kernel mutates `indices` in place.
//! Both are exactly the hazard.
//!
//! MECHANISM (read from cubecl-runtime 0.10 `tune/tuner.rs:183` + `tune/local.rs:170`):
//!   - During tuning, every benchmark rep of every variant runs on
//!     `tunables.generate_inputs(key, inputs)` — i.e. whatever the **InputGenerator**
//!     returns.
//!   - After tuning, `execute` runs the WINNING variant ONCE on the **original** inputs.
//!   `CloneInputGenerator` returns the SAME device handles (clone = ref-count bump, NOT a
//!   buffer copy) ⇒ all reps + both variants `fetch_add` into the REAL `out` ⇒ garbage.
//!   A FRESH-OUTPUT InputGenerator hands the benchmark a throwaway `out` ⇒ the real `out`
//!   is touched exactly once (the final clean run) ⇒ correct.
//!
//! CORRECTNESS METRIC (order-independent, so f32-atomic reorder noise is irrelevant):
//!   grad conservation — each feature accumulates every row's grad exactly once, so
//!   `sum(all grad cells) == num_features * sum(ord_g)` for a correct histogram. The
//!   corrupted arm inflates this by the number of benchmark launches.
//!
//! Run: cargo run --release --features rocm --example spike038_autotune_inplace_correctness

#[cfg(not(feature = "rocm"))]
fn main() {
    eprintln!("spike038 requires --features rocm (cubecl-hip / local AMD device)");
}

#[cfg(feature = "rocm")]
use cubecl::prelude::*;

#[cfg(feature = "rocm")]
const HIST_LDS_MAX: usize = 512;

/// Same faithful row-partitioned LDS build as spike-007/037. ACCUMULATES into `out`.
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
    use cubecl::tune::{
        local_tuner, CloneInputGenerator, InputGenerator, LocalTuner, Tunable, TunableSet, TuneInputs,
    };
    use lgbm_compute::runtime::RocmRuntime;
    use std::fmt::Display;

    pub const NUM_BIN: usize = 256;
    pub const CUBE_DIM: u32 = 256;

    #[derive(Clone, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    pub struct HistKey {
        pub rows: usize,
        pub feats: usize,
    }
    impl cubecl::tune::AutotuneKey for HistKey {}
    impl Display for HistKey {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "HistKey(r{},f{})", self.rows, self.feats)
        }
    }

    /// THE FIX — an InputGenerator that hands each benchmark a FRESH zeroed `out` handle,
    /// leaving the caller's real `out` (inputs[5]) untouched until the final clean run.
    /// (`generate` returns a new `Vec<Handle>` with index 5 replaced.) This is the
    /// accumulating-kernel-safe analog of `CloneInputGenerator`.
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
            let zeros = vec![0.0f32; self.slot_len];
            v[5] = self.client.create_from_slice(f32::as_bytes(&zeros));
            v
        }
    }

    /// Static tuners — one per arm so their caches don't collide.
    pub static TUNER_CLONE: LocalTuner<HistKey, String> = local_tuner!("hist038_clone");
    pub static TUNER_FRESH: LocalTuner<HistKey, String> = local_tuner!("hist038_fresh");

    fn add_variants(
        set: TunableSet<HistKey, Vec<Handle>, ()>,
        client: ComputeClient<RocmRuntime>,
        rows: usize,
        feats: usize,
        num_data: usize,
        feat_len: u32,
        slot_len: usize,
    ) -> TunableSet<HistKey, Vec<Handle>, ()> {
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
        set.with(Tunable::new("build_rp_P1", move |inputs: Vec<Handle>| {
            launch_at(&c1, &inputs, 1);
            Ok::<(), String>(())
        }))
        .with(Tunable::new("build_rp_P16", move |inputs: Vec<Handle>| {
            launch_at(&c16, &inputs, 16);
            Ok::<(), String>(())
        }))
    }

    pub fn set_clone(
        client: ComputeClient<RocmRuntime>,
        rows: usize,
        feats: usize,
        num_data: usize,
        feat_len: u32,
        slot_len: usize,
    ) -> TunableSet<HistKey, Vec<Handle>, ()> {
        let kg = move |_: &Vec<Handle>| HistKey { rows, feats };
        let set = TunableSet::new(kg, CloneInputGenerator);
        add_variants(set, client, rows, feats, num_data, feat_len, slot_len)
    }

    pub fn set_fresh(
        client: ComputeClient<RocmRuntime>,
        rows: usize,
        feats: usize,
        num_data: usize,
        feat_len: u32,
        slot_len: usize,
    ) -> TunableSet<HistKey, Vec<Handle>, ()> {
        let kg = move |_: &Vec<Handle>| HistKey { rows, feats };
        let set = TunableSet::new(
            kg,
            FreshOutGenerator {
                client: client.clone(),
                slot_len,
            },
        );
        add_variants(set, client, rows, feats, num_data, feat_len, slot_len)
    }
}

#[cfg(feature = "rocm")]
fn main() {
    use cubecl::server::Handle;
    use lgbm_compute::runtime::rocm_client;
    use tune_impl::*;

    let rows: usize = 200_000;
    let feats: usize = 50;
    let num_data = rows;
    let feat_len: u32 = (NUM_BIN * 2) as u32;
    let slot_len = feats * NUM_BIN * 2;

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

    // Order-independent correctness invariant: per feature, every grad summed once.
    let sum_g: f64 = ord_g.iter().map(|&x| x as f64).sum();
    let expected_total_grad = feats as f64 * sum_g;

    let client = rocm_client();
    let d_bins = client.create_from_slice(u32::as_bytes(&bins));
    let d_rows = client.create_from_slice(u32::as_bytes(&leaf_rows));
    let d_g = client.create_from_slice(f32::as_bytes(&ord_g));
    let d_h = client.create_from_slice(f32::as_bytes(&ord_h));
    let d_slot = client.create_from_slice(u32::as_bytes(&slot_off));

    // Sum of the grad cells (even indices) of an `out` buffer.
    let total_grad = |h: &Handle| -> f64 {
        let bytes = rocm_client().read_one_unchecked(h.clone());
        let v = f32::from_bytes(&bytes);
        v.iter().step_by(2).map(|&x| x as f64).sum()
    };

    println!("# spike038 autotune in-place correctness  rows={rows} feats={feats}");
    println!("# expected total grad (1 clean histogram) = {expected_total_grad:.1}");

    // ---- Arm A: CloneInputGenerator (the manual's pattern) → CORRUPTED ----
    let fresh_out = || client.create_from_slice(f32::as_bytes(&vec![0.0f32; slot_len]));
    let out_a = fresh_out();
    {
        let handles: Vec<Handle> = vec![
            d_bins.clone(), d_rows.clone(), d_g.clone(), d_h.clone(), d_slot.clone(), out_a.clone(),
        ];
        let set = TUNER_CLONE.init(move || {
            set_clone(rocm_client(), rows, feats, num_data, feat_len, slot_len)
        });
        TUNER_CLONE.execute(&"rocm:0".to_string(), &rocm_client(), set, handles);
    }
    let tg_a = total_grad(&out_a);
    let ratio_a = tg_a / expected_total_grad;
    println!(
        "\n[A] CloneInputGenerator : total grad = {tg_a:.1}  ({ratio_a:.1}× expected) → {}",
        if (ratio_a - 1.0).abs() < 0.01 { "OK" } else { "CORRUPTED (accumulated across benchmark launches)" }
    );

    // ---- Arm B: FreshOutGenerator (the fix) → CORRECT ----
    let out_b = fresh_out();
    {
        let handles: Vec<Handle> = vec![
            d_bins.clone(), d_rows.clone(), d_g.clone(), d_h.clone(), d_slot.clone(), out_b.clone(),
        ];
        let set = TUNER_FRESH.init(move || {
            set_fresh(rocm_client(), rows, feats, num_data, feat_len, slot_len)
        });
        TUNER_FRESH.execute(&"rocm:0".to_string(), &rocm_client(), set, handles);
    }
    let tg_b = total_grad(&out_b);
    let ratio_b = tg_b / expected_total_grad;
    let rel_err = (tg_b - expected_total_grad).abs() / expected_total_grad.abs().max(1.0);
    println!(
        "[B] FreshOutGenerator   : total grad = {tg_b:.1}  ({ratio_b:.4}× expected, rel_err {rel_err:.2e}) → {}",
        if rel_err < 1e-4 { "CORRECT (real `out` touched once)" } else { "STILL WRONG" }
    );

    println!("\nVERDICT SIGNAL: Arm A inflated ≫1×, Arm B == 1× ⇒ accumulating kernels need a fresh-output InputGenerator.");
}
