//! Byte-identity gate for the PLANE-PARALLEL cross-feature reduce
//! (`LGBM_REDUCE_PAR`, P1a of the nsys round — `docs/ondevice-cuda-perf-plan.md` §5).
//!
//! The plane twins (`reduce_scan_output_into_leaf_par_kernel` /
//! `..._into_two_leaves_par_kernel`) must reproduce the serial single-thread fold
//! BIT-FOR-BIT: within one raw window the (raw_gain desc, real_feat asc) key set is
//! strictly totally ordered (distinct feature keys; NaN gains excluded by the
//! accept-gate), so the argmax is unique and reduction order cannot matter. These
//! tests pin that claim ON A REAL DEVICE (`cuda`/`rocm` — plane collectives do not
//! exist on cubecl-cpu; the cpu anchor keeps the serial kernel, pinned by the
//! gate test below) across tie, invalid, all-invalid, permuted-feature, and
//! non-plane-multiple-width corpora.

#![cfg(feature = "gpu")]

use std::sync::Mutex;

/// Serializes access to the process-global `LGBM_REDUCE_PAR` override across the
/// tests in this binary (they run on parallel threads by default).
static OVERRIDE_LOCK: Mutex<()> = Mutex::new(());

/// One raw scan record: the 12-cell layout the fused scan kernels emit
/// (`[0]`=is_splittable flag, `[1]`=raw_threshold, `[2]`=RAW gain, `[3..5]`=counts,
/// `[5..9]`=left/right grad/hess sums, `[9]`=default_left, `[10..12]`=child outputs).
#[derive(Clone, Copy)]
struct Rec {
    valid: bool,
    thr: f64,
    gain: f64,
    lsg: f64,
    lsh: f64,
    rsg: f64,
    rsh: f64,
    dleft: f64,
    lval: f64,
    rval: f64,
}

impl Rec {
    fn cells(&self) -> [f64; 12] {
        [
            if self.valid { 1.0 } else { 0.0 },
            self.thr,
            self.gain,
            7.0, // counts — not carried by SplitSoa, arbitrary
            9.0,
            self.lsg,
            self.lsh,
            self.rsg,
            self.rsh,
            self.dleft,
            self.lval,
            self.rval,
        ]
    }
}

/// Deterministic LCG (no rand dep).
struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn next_f64(&mut self) -> f64 {
        f64::from(self.next_u32() % 4000) / 128.0 - 15.0
    }
}

fn random_recs(seed: u64, n: usize, invalid_every: usize, tie_gain: Option<f64>) -> Vec<Rec> {
    let mut rng = Lcg(seed);
    (0..n)
        .map(|i| Rec {
            valid: invalid_every == 0 || i % invalid_every != 2,
            thr: f64::from(rng.next_u32() % 255),
            gain: match tie_gain {
                // Half the records share ONE exact gain — the tie corpus.
                Some(g) if i % 2 == 0 => g,
                _ => rng.next_f64(),
            },
            lsg: rng.next_f64(),
            lsh: rng.next_f64().abs() + 0.1,
            rsg: rng.next_f64(),
            rsh: rng.next_f64().abs() + 0.1,
            dleft: f64::from(rng.next_u32() % 2),
            lval: rng.next_f64(),
            rval: rng.next_f64(),
        })
        .collect()
}

fn flat(recs: &[Rec]) -> Vec<f64> {
    recs.iter().flat_map(|r| r.cells()).collect()
}

/// Shuffled-but-distinct real feature indices (the tie-break key is the REAL
/// feature index, NOT the record position — permute to prove it).
fn permuted_feats(n: usize, seed: u64) -> Vec<i32> {
    let mut v: Vec<i32> = (0..n as i32).map(|i| i * 3 + 1).collect();
    let mut rng = Lcg(seed);
    for i in (1..n).rev() {
        v.swap(i, (rng.next_u32() as usize) % (i + 1));
    }
    v
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
mod real_gpu_gated {
    use super::{flat, permuted_feats, random_recs, Rec, OVERRIDE_LOCK};
    use lgbm_compute::kernels::best_split::SplitSoa;
    use lgbm_compute::kernels::split::{
        launch_reduce_into_leaf, launch_reduce_into_two_leaves, set_reduce_par_override,
        upload_f64_buffer,
    };

    #[cfg(feature = "cuda")]
    type GpuRt = lgbm_compute::runtime::CudaRuntime;
    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    type GpuRt = lgbm_compute::runtime::RocmRuntime;

    #[cfg(feature = "cuda")]
    fn gpu_client() -> cubecl::prelude::ComputeClient<GpuRt> {
        lgbm_compute::runtime::cuda_client()
    }
    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    fn gpu_client() -> cubecl::prelude::ComputeClient<GpuRt> {
        lgbm_compute::runtime::rocm_client()
    }

    /// Run the SINGLE-leaf reduce over `recs` with the given arm and read back the
    /// winner slot's full record.
    fn run_single(
        client: &cubecl::prelude::ComputeClient<GpuRt>,
        recs: &[Rec],
        real_feats: &[i32],
        min_gain_shift: f64,
        par: bool,
    ) -> lgbm_compute::kernels::split_info::SplitScalars {
        let raw = flat(recs);
        let h_raw = upload_f64_buffer(client, &raw);
        let out = SplitSoa::zeroed(client, 4);
        set_reduce_par_override(Some(par));
        launch_reduce_into_leaf(
            client,
            h_raw,
            raw.len(),
            real_feats,
            recs.len(),
            &out,
            2,
            0,
            min_gain_shift,
            None,
        );
        set_reduce_par_override(None);
        out.read_record(client, 2)
    }

    fn assert_bitwise_eq(
        a: &lgbm_compute::kernels::split_info::SplitScalars,
        b: &lgbm_compute::kernels::split_info::SplitScalars,
        label: &str,
    ) {
        assert_eq!(a.is_valid, b.is_valid, "{label}: is_valid");
        assert_eq!(a.gain.to_bits(), b.gain.to_bits(), "{label}: gain {} vs {}", a.gain, b.gain);
        assert_eq!(a.inner_feature_index, b.inner_feature_index, "{label}: feat");
        assert_eq!(a.threshold, b.threshold, "{label}: threshold");
        assert_eq!(a.default_left, b.default_left, "{label}: default_left");
        assert_eq!(a.num_cat_threshold, b.num_cat_threshold, "{label}: ncat");
        assert_eq!(
            a.left_sum_gradients.to_bits(),
            b.left_sum_gradients.to_bits(),
            "{label}: lsg"
        );
        assert_eq!(
            a.left_sum_hessians.to_bits(),
            b.left_sum_hessians.to_bits(),
            "{label}: lsh"
        );
        assert_eq!(
            a.right_sum_gradients.to_bits(),
            b.right_sum_gradients.to_bits(),
            "{label}: rsg"
        );
        assert_eq!(
            a.right_sum_hessians.to_bits(),
            b.right_sum_hessians.to_bits(),
            "{label}: rsh"
        );
        assert_eq!(a.left_value.to_bits(), b.left_value.to_bits(), "{label}: lval");
        assert_eq!(a.right_value.to_bits(), b.right_value.to_bits(), "{label}: rval");
    }

    #[test]
    fn plane_reduce_byte_identical_to_serial_on_device() {
        let _g = OVERRIDE_LOCK.lock().unwrap();
        let client = gpu_client();
        // (n_feats, invalid_every, tie_gain, mgs) fan-out: plane-multiple and
        // non-multiple widths, exact ties (winner = lowest REAL feat under a
        // permuted key order), sparse invalids, and sub-plane widths.
        let cases: &[(usize, usize, Option<f64>, f64)] = &[
            (50, 0, None, 0.25),
            (50, 3, None, 0.0),
            (50, 0, Some(4.5), 1.5),
            (61, 4, Some(-2.25), 0.5),
            (128, 5, Some(7.0), 0.125),
            (31, 0, Some(0.0), 0.0),
            (1, 0, None, 0.75),
            (2, 2, None, 0.0), // record 0 valid? i%2!=2 always true → both valid
            (3, 1, None, 0.0), // invalid_every=1 ⇒ i%1==0 ≠ 2 ⇒ all valid — keep a mixed case below
        ];
        for (ci, &(n, inv, tie, mgs)) in cases.iter().enumerate() {
            let recs = random_recs(0xC0FFEE + ci as u64, n, inv, tie);
            let feats = permuted_feats(n, 0xBEEF + ci as u64);
            let serial = run_single(&client, &recs, &feats, mgs, false);
            let par = run_single(&client, &recs, &feats, mgs, true);
            assert_bitwise_eq(&serial, &par, &format!("case {ci} (n={n})"));
        }
        // ALL-INVALID window: both arms must write the identical no-split sentinel.
        let mut recs = random_recs(7, 40, 0, None);
        for r in &mut recs {
            r.valid = false;
        }
        let feats = permuted_feats(40, 11);
        let serial = run_single(&client, &recs, &feats, 0.5, false);
        let par = run_single(&client, &recs, &feats, 0.5, true);
        assert_bitwise_eq(&serial, &par, "all-invalid");
        assert!(!serial.is_valid);
        // NEG_INF gain with valid flag set: the accept-gate must reject it on both arms.
        let mut recs = random_recs(13, 20, 0, None);
        for (i, r) in recs.iter_mut().enumerate() {
            if i != 5 {
                r.gain = f64::NEG_INFINITY;
            }
        }
        let feats = permuted_feats(20, 17);
        let serial = run_single(&client, &recs, &feats, 0.0, false);
        let par = run_single(&client, &recs, &feats, 0.0, true);
        assert_bitwise_eq(&serial, &par, "neg-inf-gains");
        assert!(serial.is_valid);
    }

    #[test]
    fn plane_two_leaves_reduce_byte_identical_to_serial_on_device() {
        let _g = OVERRIDE_LOCK.lock().unwrap();
        let client = gpu_client();
        let n = 50usize;
        // Sibling A: exact-tie corpus; sibling B: all-invalid — the asymmetric pair
        // exercises both the winner path and the seed path in ONE launch.
        let recs_a = random_recs(0xA11CE, n, 4, Some(3.25));
        let mut recs_b = random_recs(0xB0B, n, 0, None);
        for r in &mut recs_b {
            r.valid = false;
        }
        let feats = permuted_feats(n, 0xD00D);
        let mut raw = flat(&recs_a);
        raw.extend(flat(&recs_b));
        for par in [false, true] {
            let h_raw = upload_f64_buffer(&client, &raw);
            let out = SplitSoa::zeroed(&client, 6);
            set_reduce_par_override(Some(par));
            launch_reduce_into_two_leaves(
                &client,
                h_raw,
                raw.len(),
                &feats,
                n,
                &out,
                1,
                4,
                0.25,
                1.75,
                None,
            );
            set_reduce_par_override(None);
            let a = out.read_record(&client, 1);
            let b = out.read_record(&client, 4);
            if par {
                let (sa, sb) = SERIAL_RESULT.lock().unwrap().take().expect("serial arm first");
                assert_bitwise_eq(&sa, &a, "two-leaves slot A");
                assert_bitwise_eq(&sb, &b, "two-leaves slot B");
                assert!(!b.is_valid, "sibling B is all-invalid");
            } else {
                *SERIAL_RESULT.lock().unwrap() = Some((a, b));
            }
        }
    }

    use std::sync::Mutex;
    #[allow(clippy::type_complexity)]
    static SERIAL_RESULT: Mutex<
        Option<(
            lgbm_compute::kernels::split_info::SplitScalars,
            lgbm_compute::kernels::split_info::SplitScalars,
        )>,
    > = Mutex::new(None);
}

/// The cpu anchor NEVER takes the plane path: [`reduce_par_enabled`] hard-gates on
/// the runtime name (cubecl-cpu has no plane collectives), so even a forced-ON
/// override keeps the serial kernel — the merge gate stays byte-unchanged.
#[cfg(feature = "cpu")]
#[test]
fn reduce_par_gate_pinned_off_on_cpu_runtime() {
    let _g = OVERRIDE_LOCK.lock().unwrap();
    let client = lgbm_compute::runtime::cpu_client();
    lgbm_compute::kernels::split::set_reduce_par_override(Some(true));
    assert!(!lgbm_compute::kernels::split::reduce_par_enabled(&client));
    lgbm_compute::kernels::split::set_reduce_par_override(None);
}
