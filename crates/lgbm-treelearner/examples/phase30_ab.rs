//! On-device partition + `GrowFeature` levers A/B harness.
//!
//! This harness does TWO things in one process:
//!
//!  (A) BIT-EXACT GATE: proves the `GrowFeature` memoize
//!      (`learner.rs`, the ~25–50MB per-tree `f.bins.clone()` hoisted to a once-per-train
//!      `grow_features_cache`) is bit-exact — the tree-stream fingerprint over a multi-tree
//!      train INCLUDING a `with_features` swap tail is IDENTICAL memo-ON vs memo-OFF.
//!
//!  (B) SAME-SESSION ONDEV_GROW LEDGER A/B: a 4-arm A/B at 500k×50
//!      driving the FULL learner path, toggling BOTH off-switches in ONE process, draining
//!      the per-train `ONDEV_GROW` ledger, and printing the two target deltas:
//!        - OHP-01 (fused partition): the ledger `partition` bucket — fused ON ≤ fused OFF.
//!        - OHP-02 (GrowFeature memo): the learner GLUE = (learner wall − grow wall), which
//!          holds the per-tree `GrowFeature` build — memo ON ≤ memo OFF.
//!      Same-session is mandatory: box noise across separate processes invents fake deltas.
//!      Local hip is a spoofed 8-CU APU ⇒ these numbers are DIRECTIONAL only (both levers
//!      are host-side data-movement/alloc elisions, so the direction is expected to transfer
//!      to real CUDA — magnitudes come from the Kaggle A/B, NOT from here).
//!
//!  The 4 arms (both switches, one process):
//!    arm 1  fused ON , memo ON   (both levers ON — the default)
//!    arm 2  fused OFF, memo ON   (isolates OHP-01: partition bucket must be ≥ arm 1)
//!    arm 3  fused ON , memo OFF  (isolates OHP-02: learner glue must be ≥ arm 1)
//!    arm 4  fused OFF, memo OFF  (both OFF — the pre-optimization baseline)
//!
//!  The fused-partition switch is read-once (`OnceLock`) in production, so this harness
//!  flips it via the same-session override `grow_driver::set_fused_partition_override`
//!  (a timing-neutral `AtomicU8`, not a per-split `getenv`). The memo switch is re-read
//!  every `train_inner`, so it is toggled via `LGBM_ONDEVICE_GROWFEAT_MEMO`.
//!
//! Gates (fail-closed — `assert_eq!` + non-zero exit):
//!   (1) BIT-EXACT: the memo-ON vs memo-OFF fingerprints match (block A).
//!   (2) PARITY ACROSS ALL 4 ARMS: every arm's tree-stream fingerprint is IDENTICAL
//!       (block B) — both levers are byte-neutral, so any drift is a correctness bug.
//!   The perf DIRECTION (partition ↓, glue ↓) is printed + soft-flagged, NOT hard-gated:
//!   the DoD verdict is the Kaggle real-CUDA A/B, and the local APU wall is directional.
//!
//! Run (local hip) — build with the crate's OWN `--features rocm` (the
//! dep-scoped `<dep>/rocm` spelling leaves the example's own `cfg(feature="rocm")`
//! false and silently compiles the GPU arm OUT — do NOT use it here):
//!   LGBM_CUDA_ON_DEVICE=1 LGBM_PHASE_PROF=1 \
//!     cargo run --release -p lgbm-treelearner --features rocm --example phase30_ab
//!   # LGBM_GROW_DRAIN=1 (optional) blocks each async phase's queue empty inside its own
//!   # timer so device time RANKS into its bucket; free-run (default) PRICES the wall.
//! (This harness sets LGBM_ONDEVICE_GROWFEAT_MEMO + the fused override itself per arm —
//!  do NOT set LGBM_ONDEVICE_FUSED_PARTITION / LGBM_ONDEVICE_GROWFEAT_MEMO yourself.)

use lgbm_compute::kernels::grow_driver::{on_device_grow_phase_take, set_fused_partition_override};
use lgbm_compute::{BinColumn, GainConfig};
use lgbm_dataset::bin_mapper::{BinType, MissingType};
use lgbm_treelearner::{FeatureColumn, SerialTreeLearner};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Instant;

const NUM_BIN: u32 = 255;
const NUM_LEAVES: i32 = 31;
const N_TREES: usize = 10;
const N_TREES_SWAP: usize = 3; // extra trees after the feature-set swap (pin re-arm)
// OHP-04 ledger A/B: the tall-narrow target shape + a modest tree count (each arm is a
// full train; the memo-OFF arms rebuild the ~100MB GrowFeature clone every tree).
const LEDGER_NUM_DATA: usize = 500_000;
const LEDGER_NUM_FEATURES: usize = 50;
const LEDGER_TREES: usize = 8;

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

    #[cfg(feature = "rocm")]
    run_gpu();
    #[cfg(not(feature = "rocm"))]
    {
        eprintln!(
            "FATAL: build with the crate's OWN --features rocm (NOT the dep-scoped \
             <dep>/rocm spelling — it silently compiles the GPU arm out); the CPU anchor \
             never takes the on-device grow arm, so this A/B would be vacuous."
        );
        std::process::exit(2);
    }
}

/// Set the memo off-switch for the arm about to run. `ondevice_growfeat_memo_enabled()`
/// re-reads this env var on every `train_inner`, so toggling it between arms flips the
/// whole train. Single-threaded example main; no concurrent reader of THIS var.
#[cfg(feature = "rocm")]
fn set_memo(memo: bool) {
    // SAFETY: single-threaded example driver; the value is only read synchronously on
    // this same thread inside `train_inner`. No other thread reads this specific var.
    unsafe {
        std::env::set_var("LGBM_ONDEVICE_GROWFEAT_MEMO", if memo { "1" } else { "0" });
    }
}

/// Set BOTH off-switches for the arm about to run: the memo via env (re-read per train),
/// the fused-partition variant via the same-session override (the env gate is read-once
/// `OnceLock`, so it cannot be re-read mid-process — the override is the only way to flip
/// it same-session, and it is timing-neutral: an `AtomicU8` store, not a per-split getenv).
#[cfg(feature = "rocm")]
fn set_levers(fused: bool, memo: bool) {
    set_memo(memo);
    set_fused_partition_override(Some(fused));
}

/// Run one full multi-tree train (N_TREES over feats_a, then a `with_features` swap to
/// feats_b + N_TREES_SWAP more) with the memo ON or OFF. Returns
/// `(full_fingerprint, swap_tail_fingerprint)` over the Debug-serialized tree stream.
///
/// Constructs a FRESH backend+client per arm (the `ComputeClient` type is private in
/// `lgbm_compute`, so it cannot be threaded through a helper signature — and a fresh
/// backend also guarantees no resident state bleeds across arms).
#[cfg(feature = "rocm")]
fn train_stream_fingerprint(num_data: usize, num_features: usize, memo: bool) -> (u64, u64) {
    set_memo(memo);
    let client = lgbm_compute::runtime::rocm_client();
    let backend = lgbm_compute::RocmBackend::default();
    let feats_a = corpus(0x5eed_0030, num_data, num_features);
    let feats_b = corpus(0xb_5eed_0030, num_data, num_features); // the swap set

    let mut full = DefaultHasher::new();
    let mut swap_tail = DefaultHasher::new();

    let mut learner =
        SerialTreeLearner::new(&backend, &client, GainConfig::default(), NUM_LEAVES, -1)
            .with_features(feats_a);
    let _ = on_device_grow_phase_take(); // drain any setup residue

    for iter in 0..N_TREES {
        let (g, h) = grads(0x9e37_79b9 ^ iter as u64, num_data);
        let tree = learner.train(&g, &h, iter == 0).expect("on-device train");
        // SEAM: drain the per-train ONDEV_GROW ledger. Asserting the arm
        // actually took the on-device path guards against a compiled-out GPU arm
        // (empty ledger ⇒ a vacuous "identical" pass).
        let led = on_device_grow_phase_take();
        assert!(led.wall > 0, "grow ledger empty — on-device path NOT taken (iter {iter}); \
             did you build with the crate's OWN --features rocm?");
        format!("{tree:?}").hash(&mut full);
    }

    // ---- feature-set SWAP: with_features re-arms the memo (D5). The post-swap trees
    // must match the OFF (always-fresh) baseline — no stale subset cache. ----
    learner = learner.with_features(feats_b);
    for iter in 0..N_TREES_SWAP {
        let (g, h) = grads(0x51ca_5a1b ^ iter as u64, num_data);
        let tree = learner.train(&g, &h, iter == 0).expect("post-swap train");
        let _ = on_device_grow_phase_take();
        let s = format!("{tree:?}");
        s.hash(&mut full);
        s.hash(&mut swap_tail);
    }

    (full.finish(), swap_tail.finish())
}

/// One arm of the OHP-04 same-session ledger A/B. Runs `LEDGER_TREES` on-device trains
/// through the FULL learner path with the given lever states, sums the per-train
/// `ONDEV_GROW` ledger (excluding tree 0, which pays once-per-train setup — bin upload +
/// the first `GrowFeature` build under BOTH memo states), and returns the tree-stream
/// fingerprint + the two target metrics: the `partition` bucket (OHP-01) and the learner
/// GLUE = (learner wall − grow wall) that holds the per-tree `GrowFeature` build (OHP-02).
/// Also EMITS a canonical `[phase_prof:...] ONDEV_GROW:` line to stderr (identical format
/// to `phase_prof::dump`) so the Kaggle parser regex (`parse_growth_ledger.ONDEV_RE`) can
/// be validated against a REAL locally-emitted line.
#[cfg(feature = "rocm")]
struct ArmResult {
    fingerprint: u64,
    partition_ms: f64,
    glue_ms: f64,
    grow_wall_ms: f64,
    learner_ms: f64,
}

/// Replicate `phase_prof::dump`'s ONDEV_GROW line VERBATIM (same prefix + field order +
/// `{:.3}`/`{:.1}` precision) from a drained ledger, so `parse_growth_ledger.ONDEV_RE`
/// matches a line this harness actually emits (the regex-validation gate).
#[cfg(feature = "rocm")]
fn emit_ondev_grow_line(label: &str, led: &lgbm_compute::kernels::grow_driver::GrowPhaseNs) {
    let sum = led.setup
        + led.upload
        + led.rootfold
        + led.build
        + led.subtract
        + led.scan
        + led.pick
        + led.partition
        + led.treesplit
        + led.reduce
        + led.tail;
    let host_other = led.wall.saturating_sub(sum);
    let ms = |n: u64| n as f64 / 1e6;
    let drain = if std::env::var("LGBM_GROW_DRAIN").map(|v| v == "1").unwrap_or(false) { 1 } else { 0 };
    eprintln!(
        "[phase_prof:{label}] ONDEV_GROW: wall={:.3}ms = setup={:.3} upload={:.3} rootfold={:.3} build={:.3} subtract={:.3} scan={:.3} pick={:.3} partition={:.3} treesplit={:.3} reduce={:.3} tail={:.3} host_other={:.3} | coverage={:.1}% drain={}",
        ms(led.wall), ms(led.setup), ms(led.upload), ms(led.rootfold), ms(led.build),
        ms(led.subtract), ms(led.scan), ms(led.pick), ms(led.partition), ms(led.treesplit),
        ms(led.reduce), ms(led.tail), ms(host_other),
        if led.wall > 0 { sum as f64 * 100.0 / led.wall as f64 } else { 0.0 },
        drain,
    );
}

#[cfg(feature = "rocm")]
fn ledger_arm(fused: bool, memo: bool, arm: &str) -> ArmResult {
    set_levers(fused, memo);
    let client = lgbm_compute::runtime::rocm_client();
    let backend = lgbm_compute::RocmBackend::default();
    let feats = corpus(0x5eed_0030, LEDGER_NUM_DATA, LEDGER_NUM_FEATURES);
    let mut learner =
        SerialTreeLearner::new(&backend, &client, GainConfig::default(), NUM_LEAVES, -1)
            .with_features(feats);
    let _ = on_device_grow_phase_take(); // drain any with_features/setup residue

    let mut fp = DefaultHasher::new();
    let mut learner_ns: u64 = 0;
    for iter in 0..LEDGER_TREES {
        let (g, h) = grads(0x9e37_79b9 ^ iter as u64, LEDGER_NUM_DATA);
        let t = Instant::now();
        let tree = learner.train(&g, &h, iter == 0).expect("on-device train");
        let learner_wall = t.elapsed().as_nanos() as u64;
        format!("{tree:?}").hash(&mut fp);
        if iter == 0 {
            // Discard tree-0's ledger (once-per-train setup) so the summed buckets reflect
            // the steady per-tree cost the levers actually move.
            let led0 = on_device_grow_phase_take();
            assert!(led0.wall > 0, "grow ledger empty at tree 0 — on-device path NOT taken \
                 (arm {arm}); did you build with the crate's OWN --features rocm?");
        } else {
            learner_ns += learner_wall;
        }
    }
    // Sum of the per-tree ONDEV_GROW ledger over trees 1..LEDGER_TREES (tree 0 drained above).
    let led = on_device_grow_phase_take();
    assert!(led.wall > 0, "grow ledger empty over the steady trees — on-device path NOT taken (arm {arm})");
    emit_ondev_grow_line(&format!("phase30_{arm}"), &led);
    // glue = the learner-train wall OUTSIDE the on-device grow driver (the per-tree
    // GrowFeature build lives here). saturating: learner_wall ⊇ grow.wall by construction.
    let glue_ns = learner_ns.saturating_sub(led.wall);
    let ms = |n: u64| n as f64 / 1e6;
    ArmResult {
        fingerprint: fp.finish(),
        partition_ms: ms(led.partition),
        glue_ms: ms(glue_ns),
        grow_wall_ms: ms(led.wall),
        learner_ms: ms(learner_ns),
    }
}

/// OHP-04: the 4-arm same-session `ONDEV_GROW` ledger A/B at 500k×50. Prints the two
/// target deltas (partition bucket for fused ON/OFF; learner glue for memo ON/OFF), a
/// compact block for pasting into the results summary, and hard-asserts the tree-stream
/// fingerprint is IDENTICAL across ALL 4 arms (the byte-neutral parity gate). The perf
/// DIRECTION is soft-flagged (local APU wall is DIRECTIONAL — the DoD verdict is Kaggle).
#[cfg(feature = "rocm")]
fn ledger_ab() {
    eprintln!(
        "\n=== OHP-04 same-session ONDEV_GROW ledger A/B @ {LEDGER_NUM_DATA}x{LEDGER_NUM_FEATURES}, \
         {LEDGER_TREES} trees/arm (tree 0 excluded from sums) ==="
    );
    // arm order: both ON, fused OFF (isolate OHP-01), memo OFF (isolate OHP-02), both OFF.
    let a_on_on = ledger_arm(true, true, "fusedON_memoON");
    let a_off_on = ledger_arm(false, true, "fusedOFF_memoON");
    let a_on_off = ledger_arm(true, false, "fusedON_memoOFF");
    let a_off_off = ledger_arm(false, false, "fusedOFF_memoOFF");
    // Restore the fused override to the env default so nothing bleeds past this A/B.
    set_fused_partition_override(None);

    // Parity gate: all four arms must produce a BYTE-IDENTICAL tree stream (both levers
    // are byte-neutral; any drift is a correctness bug — hard fail).
    let fps = [
        ("fusedON_memoON", a_on_on.fingerprint),
        ("fusedOFF_memoON", a_off_on.fingerprint),
        ("fusedON_memoOFF", a_on_off.fingerprint),
        ("fusedOFF_memoOFF", a_off_off.fingerprint),
    ];
    let parity_ok = fps.iter().all(|(_, f)| *f == a_on_on.fingerprint);

    // OHP-01: the ledger partition bucket, fused ON vs OFF (both memo ON to isolate).
    let partition_dir_ok = a_on_on.partition_ms <= a_off_on.partition_ms;
    // OHP-02: the learner glue, memo ON vs OFF (both fused ON to isolate).
    let glue_dir_ok = a_on_on.glue_ms <= a_on_off.glue_ms;

    for (name, r) in [
        ("fusedON_memoON ", &a_on_on),
        ("fusedOFF_memoON", &a_off_on),
        ("fusedON_memoOFF", &a_on_off),
        ("fusedOFF_memoOFF", &a_off_off),
    ] {
        println!(
            "PHASE30_LEDGER {{\"arm\": \"{name}\", \"shape\": \"{LEDGER_NUM_DATA}x{LEDGER_NUM_FEATURES}\", \
             \"trees\": {LEDGER_TREES}, \"partition_ms\": {:.3}, \"glue_ms\": {:.3}, \
             \"grow_wall_ms\": {:.3}, \"learner_ms\": {:.3}, \"fingerprint\": \"{:016x}\"}}",
            r.partition_ms, r.glue_ms, r.grow_wall_ms, r.learner_ms, r.fingerprint
        );
    }
    println!(
        "PHASE30_LEDGER_DELTA {{\"partition_fusedON_ms\": {:.3}, \"partition_fusedOFF_ms\": {:.3}, \
         \"partition_shrink\": {partition_dir_ok}, \"glue_memoON_ms\": {:.3}, \"glue_memoOFF_ms\": {:.3}, \
         \"glue_shrink\": {glue_dir_ok}, \"parity_all4_identical\": {parity_ok}, \"note\": \
         \"local hip = spoofed 8-CU APU; DIRECTIONAL only, magnitudes from Kaggle (spike-080 transfer rule)\"}}",
        a_on_on.partition_ms, a_off_on.partition_ms, a_on_on.glue_ms, a_on_off.glue_ms,
    );

    // Fail-closed on PARITY (correctness). Soft-flag the perf DIRECTION (APU is directional).
    assert!(
        parity_ok,
        "OHP-03 VIOLATION: the 4 ledger arms did NOT all produce an identical tree stream \
         — a lever is not byte-neutral: {fps:?}"
    );
    if !partition_dir_ok {
        eprintln!(
            "WARN (directional, not a gate): fused-ON partition bucket ({:.3}ms) > fused-OFF \
             ({:.3}ms) — APU noise? re-run; the Kaggle A/B is the verdict.",
            a_on_on.partition_ms, a_off_on.partition_ms
        );
    }
    if !glue_dir_ok {
        eprintln!(
            "WARN (directional, not a gate): memo-ON glue ({:.3}ms) > memo-OFF ({:.3}ms) \
             — APU noise? re-run; the Kaggle A/B is the verdict.",
            a_on_on.glue_ms, a_on_off.glue_ms
        );
    }
    println!(
        "=== OHP-04: parity_all4={parity_ok} partition_shrink(OHP-01)={partition_dir_ok} \
         glue_shrink(OHP-02)={glue_dir_ok} (DIRECTIONAL — Kaggle confirms magnitude) ==="
    );
}

#[cfg(feature = "rocm")]
fn run_gpu() {
    let mut all_ok = true;
    for &(num_data, num_features) in &[(50_000usize, 50usize), (20_000, 200)] {
        // memo ON (default) vs memo OFF (fresh-build ground truth), same process.
        let (on_full, on_swap) = train_stream_fingerprint(num_data, num_features, true);
        let (off_full, off_swap) = train_stream_fingerprint(num_data, num_features, false);

        let full_ok = on_full == off_full;
        let swap_ok = on_swap == off_swap;
        all_ok &= full_ok && swap_ok;

        println!(
            "PHASE30_AB {{\"shape\": \"{num_data}x{num_features}\", \"trees\": {N_TREES}, \
             \"swap_trees\": {N_TREES_SWAP}, \
             \"memo_on_full\": \"{on_full:016x}\", \"memo_off_full\": \"{off_full:016x}\", \
             \"full_identical\": {full_ok}, \
             \"memo_on_swap_tail\": \"{on_swap:016x}\", \"memo_off_swap_tail\": \"{off_swap:016x}\", \
             \"swap_tail_identical\": {swap_ok}}}"
        );

        // Fail-closed: the whole point is bit-exactness (OHP-03). A printed warning is
        // NOT acceptable — assert_eq! aborts (non-zero exit) on any fingerprint drift.
        assert_eq!(
            on_full, off_full,
            "OHP-03 VIOLATION: memo-ON vs memo-OFF FULL tree-stream fingerprints DIFFER \
             at {num_data}x{num_features} — the GrowFeature memoize is NOT bit-exact"
        );
        assert_eq!(
            on_swap, off_swap,
            "OHP-03 VIOLATION: post-swap tree fingerprints DIFFER at {num_data}x{num_features} \
             — the with_features swap did NOT re-arm the memo (stale subset cache)"
        );
    }

    assert!(all_ok, "phase30_ab: at least one fingerprint pair diverged");
    println!("=== PASS: GrowFeature memoize is bit-exact (memo ON == OFF, incl. swap tail) ===");

    // Block B (OHP-04): the same-session 4-arm ONDEV_GROW ledger A/B at 500k×50, toggling
    // BOTH off-switches in this one process. Prints the partition-bucket (OHP-01) + learner-
    // glue (OHP-02) deltas + a canonical ONDEV_GROW line per arm for the Kaggle regex check.
    ledger_ab();
}
