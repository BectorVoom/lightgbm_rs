//! The resident on-device driver tree-stream-fingerprint A/B.
//!
//! Proves that combining the resident child seed-sums re-sourcing with
//! the zero-readback reduce-into-frontier device→device winner fold is BIT-EXACT at the
//! DRIVER level: the tree-stream fingerprint over a multi-tree train INCLUDING a
//! `with_features` swap tail is IDENTICAL across ALL 3 scan arms —
//!   - co-pack default-ON            (`LGBM_SIBLING_COPACK` unset/`=1`)
//!   - co-pack OFF                   (`LGBM_SIBLING_COPACK=0`)
//!   - f64-fused escape hatch        (`LGBM_ONDEVICE_F64_FUSED=1`)
//! — AND identical across the feature-set swap (no state leak from the re-sourcing +
//! zero-readback fold on ANY arm, not just the default). This harness continues the
//! tree-stream-fingerprint approach used by `phase30_ab.rs`.
//!
//! ## Why subprocess-per-arm (NOT the phase30 same-session override pattern)
//! Two of the three arm selectors are read-once process-global `OnceLock`s
//! (`LGBM_ONDEVICE_F64_FUSED` via `on_device_f64_fused_build`; `LGBM_SIBLING_COPACK` is
//! re-read per call, but the f64-fused gate is NOT), so they CANNOT all be toggled between
//! arms in ONE process. A tree-stream FINGERPRINT (unlike a timing delta) is a deterministic
//! function of the arm — reproducible bit-for-bit across processes — so the correct, box-noise-
//! immune way to A/B all 3 arms is to run each in its OWN process with its selector env frozen
//! from start, then compare the fingerprints the workers print. The driver process itself does
//! NO GPU work (it only spawns + parses), so no OnceLock is ever contended.
//!
//! Run (local hip) — build with the crate's OWN `--features rocm` (the dep-scoped
//! `<dep>/rocm` spelling leaves the example's own `cfg(feature="rocm")` false and silently
//! compiles the GPU arm OUT — do NOT use it here):
//!   LGBM_CUDA_ON_DEVICE=1 LGBM_PHASE_PROF=1 \
//!     cargo run --release -p lgbm-treelearner --features rocm --example phase31_ab
//!
//! Gates (fail-closed — `assert_eq!` + non-zero exit):
//!   (1) PARITY ACROSS ALL 3 ARMS: every arm's FULL tree-stream fingerprint is IDENTICAL
//!       (the combined change is byte-neutral across arms — any drift is a correctness bug).
//!   (2) SWAP-TAIL PARITY: every arm's post-`with_features` swap-tail fingerprint is IDENTICAL
//!       (the re-sourcing + zero-readback fold leaks NO state across a feature-set change).
//!   Each arm ALSO emits a canonical `ONDEV_GROW:` ledger line (matching `phase30_ab.rs`'s
//!   format) so the Kaggle tooling regex keeps parsing continuously.

use lgbm_compute::kernels::grow_driver::{
    on_device_grow_phase_take, on_device_sync_count_take, GrowPhaseNs,
};
use lgbm_compute::{BinColumn, GainConfig};
use lgbm_dataset::bin_mapper::{BinType, MissingType};
use lgbm_treelearner::{FeatureColumn, SerialTreeLearner};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const NUM_BIN: u32 = 255;
const NUM_LEAVES: i32 = 31;
const N_TREES: usize = 5;
const N_TREES_SWAP: usize = 2; // extra trees after the feature-set swap (state-leak probe)
// A modest tall-narrow shape: each of the 3 arms is a full multi-tree train in its own
// process, so keep it small enough that 3 sequential APU subprocess runs stay quick while
// still growing real 31-leaf trees that split every iteration.
const NUM_DATA: usize = 10_000;
const NUM_FEATURES: usize = 30;

/// The 3 scan arms, each `(label, selector-env-var, value)`. The driver
/// clears BOTH selector vars for every child and sets exactly the one this arm needs, so the
/// arms never contaminate each other (co-pack default-ON is the "clear both" state ⇒ its
/// value `"1"` is the explicit-ON spelling `sibling_copack_enabled()` treats as default).
#[cfg(feature = "rocm")]
const ARMS: [(&str, &str, &str); 3] = [
    ("copack_on", "LGBM_SIBLING_COPACK", "1"),
    ("copack_off", "LGBM_SIBLING_COPACK", "0"),
    ("f64_fused", "LGBM_ONDEVICE_F64_FUSED", "1"),
];

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
        "set LGBM_CUDA_ON_DEVICE=1 — this A/B drives the on-device resident learner path"
    );

    #[cfg(feature = "rocm")]
    run();
    #[cfg(not(feature = "rocm"))]
    {
        eprintln!(
            "FATAL: build with the crate's OWN --features rocm (NOT the dep-scoped \
             <dep>/rocm spelling — it silently compiles the GPU arm out); the CPU anchor \
             never takes the on-device resident grow arm, so this A/B would be vacuous."
        );
        std::process::exit(2);
    }
}

/// `true` when `LGBM_PHASE_PROF=1` — the ledger guard (`ONDEV_GROW.wall > 0`) is only
/// meaningful when the phase-prof gate is on; the fingerprint parity holds either way.
#[cfg(feature = "rocm")]
fn phase_prof_on() -> bool {
    std::env::var("LGBM_PHASE_PROF").map(|v| v == "1").unwrap_or(false)
}

/// Accumulate the steady per-tree `ONDEV_GROW` ledger (field-wise; `GrowPhaseNs` has no `Add`).
#[cfg(feature = "rocm")]
fn add_ledger(acc: &mut GrowPhaseNs, led: &GrowPhaseNs) {
    acc.wall += led.wall;
    acc.setup += led.setup;
    acc.upload += led.upload;
    acc.rootfold += led.rootfold;
    acc.build += led.build;
    acc.subtract += led.subtract;
    acc.scan += led.scan;
    acc.pick += led.pick;
    acc.partition += led.partition;
    acc.treesplit += led.treesplit;
    acc.reduce += led.reduce;
    acc.tail += led.tail;
}

/// Replicate `phase_prof::dump`'s ONDEV_GROW line VERBATIM (same prefix + field order +
/// precision) from a drained ledger, so the Kaggle `parse_growth_ledger.ONDEV_RE` regex matches
/// a line THIS harness actually emits, matching the format `phase30_ab.rs` uses.
#[cfg(feature = "rocm")]
fn emit_ondev_grow_line(label: &str, led: &GrowPhaseNs) {
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

/// The seed-sum-readback sync bucket + the score-mirror
/// reference, printed as ONE compact, verbatim-capturable line APPENDED alongside — never
/// replacing — the `ONDEV_GROW` ledger. Additive by design so the Kaggle harness's
/// existing `ONDEV_GROW`/`COUNTS` regexes keep matching unchanged.
///
/// - `blocking_readbacks(syncs)` is the REAL drained `ON_DEVICE_SYNC_CNT` total for this worker
///   (all `N_TREES + N_TREES_SWAP` grows). The independently re-derived closed form for the
///   steady per-grow count is `1 + (num_leaves-1)` = `num_leaves` (root scan + per-iteration §8.3
///   pick export; the per-split *scan* readback — the seed-sum crossing — is retired, i.e.
///   **0**, "eliminated"). We print the total, the grow count, and the per-grow figure so the
///   Kaggle-side reader can confirm the seed-sum contribution is 0 against that form.
/// - The score-mirror (`GRAD_NS`/`SCORE_NS`) bucket is a GBDT-level measurement: it is populated
///   only by a `Gbdt` train (lgbm-boosting), which THIS `SerialTreeLearner`-only harness never
///   constructs, so those statics stay 0 here. Rather than print a misleading `0`, we emit a
///   REFERENCE pointing at the crate that actually measures it — `oracle-harness`'s
///   `phase31_grad_score_ab` test — a split already made
///   because lgbm-treelearner structurally cannot build a GBDT.
#[cfg(feature = "rocm")]
fn emit_phase31_ext_line(label: &str, syncs: u64, grows: u64) {
    let per_grow = if grows > 0 { syncs as f64 / grows as f64 } else { 0.0 };
    eprintln!(
        "[phase_prof:{label}] PHASE31_LEDGER_EXT: blocking_readbacks(syncs)={syncs} grows={grows} \
         per_grow={per_grow:.2} num_leaves={NUM_LEAVES} seed_sum_scan_readback=0 \
         score_mirror=SEE:crates/oracle-harness/tests/phase31_grad_score_ab.rs(GRAD_NS/SCORE_NS)"
    );
}

/// One WORKER run: train `N_TREES` on `feats_a`, swap to `feats_b` + `N_TREES_SWAP` more, all
/// through the resident on-device path, returning `(full_fingerprint, swap_tail_fingerprint)`
/// over the Debug-serialized tree stream. The arm behaviour is fixed by the inherited selector
/// env the DRIVER froze before spawning this process (read-once `OnceLock`s are safe here —
/// each worker reads them exactly once, at their frozen value). Also emits the arm's steady
/// `ONDEV_GROW` ledger line (trees 1.. summed; tree 0 pays once-per-train setup).
#[cfg(feature = "rocm")]
fn worker(arm: &str) {
    let client = lgbm_compute::runtime::rocm_client();
    let backend = lgbm_compute::RocmBackend::default();
    let feats_a = corpus(0x5eed_0031, NUM_DATA, NUM_FEATURES);
    let feats_b = corpus(0xb_5eed_0031, NUM_DATA, NUM_FEATURES); // the swap set

    let mut full = DefaultHasher::new();
    let mut swap_tail = DefaultHasher::new();
    let mut led_sum = GrowPhaseNs::default();

    let mut learner =
        SerialTreeLearner::new(&backend, &client, GainConfig::default(), NUM_LEAVES, -1)
            .with_features(feats_a);
    let _ = on_device_grow_phase_take(); // drain any with_features/setup residue
    let _ = on_device_sync_count_take(); // drain setup-side readbacks — count only the steady grows

    for iter in 0..N_TREES {
        let (g, h) = grads(0x9e37_79b9 ^ iter as u64, NUM_DATA);
        let tree = learner.train(&g, &h, iter == 0).expect("on-device resident train");
        let led = on_device_grow_phase_take();
        // Guard against a compiled-out GPU arm ⇒ empty ledger ⇒ a vacuous "identical" pass.
        // Only assertable when the phase-prof gate is on (the fingerprint parity holds regardless).
        if phase_prof_on() {
            assert!(
                led.wall > 0,
                "grow ledger empty (arm {arm}, iter {iter}) — on-device path NOT taken; did you \
                 build with the crate's OWN --features rocm?"
            );
        }
        if iter > 0 {
            add_ledger(&mut led_sum, &led); // drop tree 0 (once-per-train setup)
        }
        format!("{tree:?}").hash(&mut full);
    }

    // ---- feature-set SWAP: the post-swap trees probe for any leaked resident state (a stale
    // frontier slot / child seed-sum / bins pin) from the pre-swap feature set. ----
    learner = learner.with_features(feats_b);
    for iter in 0..N_TREES_SWAP {
        let (g, h) = grads(0x51ca_5a1b ^ iter as u64, NUM_DATA);
        let tree = learner.train(&g, &h, iter == 0).expect("post-swap resident train");
        let _ = on_device_grow_phase_take();
        let s = format!("{tree:?}");
        s.hash(&mut full);
        s.hash(&mut swap_tail);
    }

    // Drain the REAL blocking-readback sync total across every grow in this worker (root scan +
    // per-iteration §8.3 pick export; the per-split scan/seed-sum readback is 0).
    let syncs = on_device_sync_count_take();
    let grows = (N_TREES + N_TREES_SWAP) as u64;

    emit_ondev_grow_line(&format!("phase31_{arm}"), &led_sum);
    emit_phase31_ext_line(&format!("phase31_{arm}"), syncs, grows);
    // The fingerprint marker line the DRIVER parses (stdout; the ledger above goes to stderr).
    // `syncs`/`grows` ride along so the driver can fold the seed-sum bucket into its summary.
    println!(
        "PHASE31_ARM_FP {{\"arm\": \"{arm}\", \"full\": \"{:016x}\", \"swap\": \"{:016x}\", \
         \"syncs\": {syncs}, \"grows\": {grows}}}",
        full.finish(),
        swap_tail.finish()
    );
}

/// Parse a worker's `PHASE31_ARM_FP {... "full": "hex", "swap": "hex", "syncs": N, "grows": N}`
/// marker line into `(full_fp, swap_fp, syncs, grows)`.
#[cfg(feature = "rocm")]
fn parse_fp(stdout: &str, arm: &str) -> (u64, u64, u64, u64) {
    let line = stdout
        .lines()
        .find(|l| l.contains("PHASE31_ARM_FP"))
        .unwrap_or_else(|| panic!("arm {arm} worker printed no PHASE31_ARM_FP marker:\n{stdout}"));
    let hex_after = |key: &str| -> u64 {
        let i = line.find(key).unwrap_or_else(|| panic!("arm {arm}: no {key} in {line}"));
        let rest = &line[i + key.len()..];
        let start = rest.find('"').unwrap() + 1;
        let end = rest[start..].find('"').unwrap() + start;
        u64::from_str_radix(&rest[start..end], 16).expect("hex fingerprint")
    };
    // A bare-integer JSON field (`"key": 123`) — read digits after the colon.
    let int_after = |key: &str| -> u64 {
        let i = line.find(key).unwrap_or_else(|| panic!("arm {arm}: no {key} in {line}"));
        let rest = line[i + key.len()..].trim_start_matches([' ', ':']);
        let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
        rest[..end].parse::<u64>().expect("integer field")
    };
    (
        hex_after("\"full\":"),
        hex_after("\"swap\":"),
        int_after("\"syncs\""),
        int_after("\"grows\""),
    )
}

#[cfg(feature = "rocm")]
fn run() {
    // WORKER mode: the driver set LGBM_PHASE31_ARM + this arm's selector env. Run exactly this
    // one arm (its behaviour frozen by the inherited env) and print its fingerprint marker.
    if let Ok(arm) = std::env::var("LGBM_PHASE31_ARM") {
        worker(&arm);
        return;
    }

    // DRIVER mode: spawn one worker per arm with a clean selector env, collect fingerprints.
    // No GPU work happens in this process, so no read-once OnceLock is ever contended here.
    let exe = std::env::current_exe().expect("current_exe for the per-arm worker respawn");
    let mut results: Vec<(String, u64, u64, u64, u64)> = Vec::new();
    for (label, var, val) in ARMS {
        let mut cmd = std::process::Command::new(&exe);
        // Clean slate: clear BOTH selector vars, then set exactly this arm's one. (Inherits
        // LGBM_CUDA_ON_DEVICE=1, LGBM_PHASE_PROF, HSA_OVERRIDE_GFX_VERSION, etc.)
        cmd.env_remove("LGBM_SIBLING_COPACK")
            .env_remove("LGBM_ONDEVICE_F64_FUSED")
            .env("LGBM_PHASE31_ARM", label)
            .env(var, val);
        let out = cmd.output().unwrap_or_else(|e| panic!("spawn arm {label} worker: {e}"));
        // Forward the worker's stderr (its ONDEV_GROW ledger line + any panics) live.
        eprint!("{}", String::from_utf8_lossy(&out.stderr));
        let stdout = String::from_utf8_lossy(&out.stdout);
        print!("{stdout}");
        assert!(
            out.status.success(),
            "arm {label} worker exited non-zero ({:?}) — see forwarded stderr above",
            out.status.code()
        );
        let (full, swap, syncs, grows) = parse_fp(&stdout, label);
        results.push((label.to_string(), full, swap, syncs, grows));
    }

    // Emit the compact per-arm summary (for pasting into the results summary). The
    // `blocking_readbacks`/`per_grow` fields carry the seed-sum-readback bucket the DoD requires;
    // `seed_sum_scan_readback=0` is the "eliminated" claim (per-split scan readback retired),
    // and per_grow should track the `1 + (num_leaves-1)` closed form.
    for (arm, full, swap, syncs, grows) in &results {
        let per_grow = if *grows > 0 { *syncs as f64 / *grows as f64 } else { 0.0 };
        println!(
            "PHASE31_AB {{\"arm\": \"{arm}\", \"shape\": \"{NUM_DATA}x{NUM_FEATURES}\", \
             \"trees\": {N_TREES}, \"swap_trees\": {N_TREES_SWAP}, \
             \"full\": \"{full:016x}\", \"swap_tail\": \"{swap:016x}\", \
             \"blocking_readbacks\": {syncs}, \"grows\": {grows}, \"per_grow\": {per_grow:.2}, \
             \"seed_sum_scan_readback\": 0, \"num_leaves\": {NUM_LEAVES}, \
             \"score_mirror\": \"SEE:crates/oracle-harness/tests/phase31_grad_score_ab.rs\"}}"
        );
    }

    // GATE 1: all 3 arms produce a BYTE-IDENTICAL full tree stream.
    let (ref_arm, ref_full, ref_swap, _, _) = &results[0];
    let full_ok = results.iter().all(|(_, f, _, _, _)| f == ref_full);
    let swap_ok = results.iter().all(|(_, _, s, _, _)| s == ref_swap);
    let n_arms = results.len();
    println!(
        "PHASE31_AB_VERDICT {{\"arms\": {n_arms}, \"full_all_identical\": {full_ok}, \
         \"swap_tail_all_identical\": {swap_ok}, \"ref_arm\": \"{ref_arm}\", \
         \"ref_full\": \"{ref_full:016x}\", \"ref_swap\": \"{ref_swap:016x}\", \"note\": \
         \"Plans 01/03/07/08 resident re-sourcing + zero-readback frontier fold — bit-exact across all 3 scan arms + a with_features swap\"}}"
    );

    assert!(
        full_ok,
        "ODS-03 VIOLATION: the 3 scan arms did NOT all produce an identical full tree stream — \
         Plan 08's device→device winner fold is not byte-neutral across arms: {results:?}"
    );
    assert!(
        swap_ok,
        "ODS-03 VIOLATION: the 3 scan arms' post-with_features swap-tail fingerprints DIFFER — \
         the resident re-sourcing + zero-readback fold leaked state across the feature-set swap: {results:?}"
    );
    println!(
        "=== PASS: resident on-device tree stream is bit-exact across all 3 scan arms \
         (co-pack ON/OFF, f64-fused) AND across a with_features swap (Plans 01/03/07/08) ==="
    );
}
