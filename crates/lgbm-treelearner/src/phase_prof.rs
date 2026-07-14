//! Env-gated per-phase wall-clock accumulator for the serial
//! tree-learner hot loop. Inert unless `LGBM_PHASE_PROF=1`; in that case the
//! three growth-loop phases accumulate into process-global atomics that the
//! bench harness prints via [`dump`]. Measurement-only; never changes train
//! semantics, and the `time` wrapper is a direct passthrough when disabled.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

pub static BEFORE_NS: AtomicU64 = AtomicU64::new(0);
pub static HISTSPLIT_NS: AtomicU64 = AtomicU64::new(0);
pub static PARTITION_NS: AtomicU64 = AtomicU64::new(0);
// Sub-phases of HISTSPLIT: BUILD = histogram construction (the per-feature gather),
// SCAN = the per-feature find_best_split scan. BUILD+SCAN ≈ HISTSPLIT (minus
// subtract/compact glue), used to localize the hist-vs-split gap.
pub static BUILD_NS: AtomicU64 = AtomicU64::new(0);
pub static SCAN_NS: AtomicU64 = AtomicU64::new(0);
// WHOLE-TRAIN BUDGET counters — the growth-loop phases above cover only part of
// GPU train wall-clock; these attribute the uninstrumented majority. Wrapped at the
// boosting/binning seam, NOT nested inside the growth loop:
//   BINNING  = once-per-train `build_feature_columns` (fixed setup; bench-repeated).
//   GRAD     = per-iter objective `get_gradients` (grad/hess compute).
//   LEARNER  = per-iter `learner.train_*` call — SUPERSET of BEFORE+HISTSPLIT+PARTITION;
//              (LEARNER − those) = in-learner orchestration + per-tree GPU grad/hess
//              upload + resident-pool/partition setup OUTSIDE the phase guards.
//   SCORE    = per-iter `score_updater.update` (UpdateScore scatter).
pub static BINNING_NS: AtomicU64 = AtomicU64::new(0);
pub static GRAD_NS: AtomicU64 = AtomicU64::new(0);
pub static LEARNER_NS: AtomicU64 = AtomicU64::new(0);
pub static SCORE_NS: AtomicU64 = AtomicU64::new(0);
// Boosting-loop attribution. TRAIN_ONE_ITER wraps the whole
// `gbdt.train_one_iter` call (⊇ GRAD + LEARNER + SCORE + snapshot/boost/alloc);
// `loop_other = train − binning − Σtrain_one_iter` isolates the booster-loop tail
// (metric eval / valid / accumulation). SNAPSHOT = the per-iter `scores().to_vec()`
// clones; METRIC = the booster-loop `m.eval` over all rows.
pub static TRAIN_ONE_ITER_NS: AtomicU64 = AtomicU64::new(0);
pub static SNAPSHOT_NS: AtomicU64 = AtomicU64::new(0);
pub static METRIC_NS: AtomicU64 = AtomicU64::new(0);
// Per-train model-metadata setup (`feature_infos_from_rows` min/max
// over the raw matrix) — outside binning AND the boosting loop.
pub static SETUP_NS: AtomicU64 = AtomicU64::new(0);
// The per-`train_inner` (= per-TREE) resident-bin device upload
// (`wants_resident_bins` block, learner.rs) — two `[num_features × num_data]` u32 host
// re-allocations + a host→device `create_from_slice`, redundantly repeated every tree
// even though the binned columns are IMMUTABLE for the whole train. A subset of
// `in_learner_other`; GPU-only (CpuBackend `wants_resident_bins()==false`).
pub static UPLOAD_NS: AtomicU64 = AtomicU64::new(0);

// Break `in_learner_other` (= learner − before − hist+split − partition) into its
// per-tree setup components. These are all OUTSIDE the growth-loop phase guards:
//   ROOT_FOLD   = the per-tree root LeafSplits::init f64 fold over ALL rows
//                 (grad/hess sum); single-threaded CPU, backend-independent.
//   PARTITION_NEW = per-tree DataPartition::new (alloc + fill 0..num_data); CPU.
//   RESIDENT_RESET = per-tree backend.reset_resident_pool (GPU device-handle mirror
//                 reset + device-memory drop); no-op on CpuBackend, GPU-only cost.
// (in_learner_other − these) = residual per-tree allocations / orchestration.
pub static ROOT_FOLD_NS: AtomicU64 = AtomicU64::new(0);
pub static PARTITION_NEW_NS: AtomicU64 = AtomicU64::new(0);
pub static RESIDENT_RESET_NS: AtomicU64 = AtomicU64::new(0);
// SCRATCH = the per-tree scratch/constraint setup block (best_split_per_leaf +
// feature_splittable Vec<Vec> + best_cat_threshold + monotone/cegb/branch models).
// Suspected dominant `residual`: CegbModel::new allocates at num_data scale every
// tree even when CEGB is inactive (default).
pub static SCRATCH_NS: AtomicU64 = AtomicU64::new(0);

// GPU per-leaf LAUNCH/ROUND-TRIP COUNT counters. These count how many device
// launches and how many blocking host round-trips (`read_one_unchecked` syncs) a
// tree costs, at the per-leaf Backend entry points, so the round-trip floor is
// EMPIRICAL (not inferred). Inert unless LGBM_PHASE_PROF=1.
//   BUILD_RESIDENT = standalone device histogram builds (root + smaller children).
//   SUBTRACT_RESIDENT = on-device parent−smaller derivations (larger children).
//   SCAN_RESIDENT = fused per-leaf scan launches = blocking readback SYNCS (the
//                   round-trip count sibling co-packing targets halving).
//   FUSED = build_fix_scan_resident launches (OFF by default).
pub static BUILD_RESIDENT_CNT: AtomicU64 = AtomicU64::new(0);
pub static SUBTRACT_RESIDENT_CNT: AtomicU64 = AtomicU64::new(0);
pub static SCAN_RESIDENT_CNT: AtomicU64 = AtomicU64::new(0);
pub static FUSED_CNT: AtomicU64 = AtomicU64::new(0);

/// Increment a count counter by 1. Inert (no-op) when the env gate is off, so it is
/// parity-neutral and zero-overhead in the default build/tests.
#[inline]
pub fn bump(counter: &AtomicU64) {
    if enabled() {
        counter.fetch_add(1, Ordering::Relaxed);
    }
}

/// RAII timer: accumulates its lifetime into `c`. Inert when the env gate is off.
pub struct Guard {
    c: &'static AtomicU64,
    t: Instant,
    on: bool,
}
impl Drop for Guard {
    fn drop(&mut self) {
        if self.on {
            self.c
                .fetch_add(self.t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }
}
/// Start an RAII phase timer accumulating into `c` until the guard drops.
pub fn guard(c: &'static AtomicU64) -> Guard {
    Guard { c, t: Instant::now(), on: enabled() }
}

// IN-01: this is the CANONICAL `LGBM_PHASE_PROF` gate. It has a deliberate verbatim
// TWIN, `lgbm_compute::kernels::grow_driver::launch_prof_enabled()`, which cannot
// share this helper without a crate cycle (`phase_prof` lives ABOVE `lgbm-compute` in
// the DAG). Keep the two in lockstep: any change to the env interpretation here (e.g.
// accepting `"true"`) must be mirrored in the compute-side twin, and vice-versa.
fn enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("LGBM_PHASE_PROF").map(|v| v == "1").unwrap_or(false))
}

/// Time `f` into `counter` (nanoseconds). Zero-overhead passthrough when the
/// env gate is off.
#[inline]
pub fn time<T>(counter: &AtomicU64, f: impl FnOnce() -> T) -> T {
    if !enabled() {
        return f();
    }
    let t = Instant::now();
    let r = f();
    counter.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    r
}

/// Print the accumulated phase breakdown to stderr and reset the counters.
pub fn dump(label: &str) {
    if !enabled() {
        return;
    }
    let b = BEFORE_NS.swap(0, Ordering::Relaxed);
    let h = HISTSPLIT_NS.swap(0, Ordering::Relaxed);
    let p = PARTITION_NS.swap(0, Ordering::Relaxed);
    let build = BUILD_NS.swap(0, Ordering::Relaxed);
    let scan = SCAN_NS.swap(0, Ordering::Relaxed);
    let tot = (b + h + p) as f64 / 1e6;
    eprintln!(
        "[phase_prof:{label}] before={:.3}ms hist+split={:.3}ms (build={:.3} scan={:.3}) partition={:.3}ms total={:.3}ms",
        b as f64 / 1e6,
        h as f64 / 1e6,
        build as f64 / 1e6,
        scan as f64 / 1e6,
        p as f64 / 1e6,
        tot
    );
    if tot > 0.0 {
        eprintln!(
            "[phase_prof:{label}] %: before={:.1} hist+split={:.1} (build={:.1} scan={:.1}) partition={:.1}",
            b as f64 / 1e4 / tot,
            h as f64 / 1e4 / tot,
            build as f64 / 1e4 / tot,
            scan as f64 / 1e4 / tot,
            p as f64 / 1e4 / tot
        );
    }
    // WHOLE-TRAIN BUDGET. `learner` is the per-iter tree-train call and is a
    // SUPERSET of before+hist+split+partition; `in_learner_other = learner − that` is the
    // previously-uninstrumented in-learner cost (per-tree GPU grad/hess upload, resident-
    // pool / partition setup, host orchestration outside the growth-phase guards).
    let binning = BINNING_NS.swap(0, Ordering::Relaxed);
    let grad = GRAD_NS.swap(0, Ordering::Relaxed);
    let learner = LEARNER_NS.swap(0, Ordering::Relaxed);
    let score = SCORE_NS.swap(0, Ordering::Relaxed);
    let toi = TRAIN_ONE_ITER_NS.swap(0, Ordering::Relaxed);
    let snapshot = SNAPSHOT_NS.swap(0, Ordering::Relaxed);
    let metric = METRIC_NS.swap(0, Ordering::Relaxed);
    if binning + grad + learner + score + toi > 0 {
        // in-train_one_iter overhead NOT in grad/learner/score/snapshot
        // (boost_from_average, bagging, grad/hess Vec alloc, snapshot bookkeeping).
        let in_iter_other = toi.saturating_sub(grad + learner + score + snapshot);
        let setup = SETUP_NS.swap(0, Ordering::Relaxed);
        eprintln!(
            "[phase_prof:{label}] LOOP: train_one_iter={:.3}ms (grad={:.3} learner={:.3} score={:.3} snapshot={:.3} in_iter_other={:.3}) metric={:.3}ms feature_infos_setup={:.3}ms",
            toi as f64 / 1e6,
            grad as f64 / 1e6,
            learner as f64 / 1e6,
            score as f64 / 1e6,
            snapshot as f64 / 1e6,
            in_iter_other as f64 / 1e6,
            metric as f64 / 1e6,
            setup as f64 / 1e6,
        );
    }
    // Per-train LAUNCH/ROUND-TRIP COUNTS. `scan_resident` is the blocking
    // host round-trip (sync) count; build+subtract+fused+scan ≈ total device launches.
    //
    // UNIT CONTRACT (WR-01/IN-02): EVERY term summed into `device_launches=` counts
    // PER-LEAF build / subtract / scan operations (one batched launch over all
    // features per leaf), NOT per-feature. The on-device `on_device=` term
    // (`on_device_launch_count_take`) is bumped at the same per-leaf granularity in
    // `grow_driver` so the sum is apples-to-apples across the host and on-device paths.
    // NOTE (IN-02): this is a build+subtract+scan SUBTOTAL, not a true total device
    // count — per-split tree-mutation / partition / add_bias device dispatches are
    // intentionally excluded (both host and on-device omit them, so the A/B is not
    // skewed); read `device_launches=` as "histogram build+subtract+scan launches".
    let bld_cnt = BUILD_RESIDENT_CNT.swap(0, Ordering::Relaxed);
    let sub_cnt = SUBTRACT_RESIDENT_CNT.swap(0, Ordering::Relaxed);
    let scn_cnt = SCAN_RESIDENT_CNT.swap(0, Ordering::Relaxed);
    let fus_cnt = FUSED_CNT.swap(0, Ordering::Relaxed);
    // Fold the on-device driver's own launch count (bumped in lgbm-compute
    // once per leaf-level build/subtract/scan — the host `*_RESIDENT_CNT` counters
    // stay 0 on the on-device path) into the SAME `device_launches=` total. The
    // consumer captures the total via the SHORT regex
    // `device_launches=(?P<launches>\d+)`, so `on_device=` MUST live INSIDE the
    // parenthesized breakdown (never before the total) to keep that capture stable.
    // Without this fold an on-device train emits `device_launches=0` and the
    // whole line is suppressed by the `> 0` guard.
    let on_dev = lgbm_compute::kernels::grow_driver::on_device_launch_count_take();
    // The ROOT + directly-built-child parallel-u64 build sub-counter. NONZERO proves
    // those sites run the parallel u64 fixed-point kernel (distinct from the
    // subtract-path smaller child, which is already u64). Taken unconditionally so the
    // process-global counter resets each dump; folded INSIDE the parenthetical
    // breakdown (never before the leading total) so the harness capture of the launch
    // total stays byte-stable and the launch-total field key is untouched.
    let on_dev_rootbuild_u64 =
        lgbm_compute::kernels::grow_driver::on_device_rootbuild_u64_count_take();
    // The NEGATIVE guard — the f64 single-owner fused build counter. It bumps ONLY via
    // the `LGBM_ONDEVICE_F64_FUSED=1` escape hatch, so on the DEFAULT (swapped)
    // on-device path it stays 0. Emitting it INSIDE the parenthetical breakdown (after
    // on_device_rootbuild_u64, never before the leading total) lets an A/B harness prove
    // the slow f64-fused kernel did not silently return, from the log rather than by
    // code inspection alone. The `device_launches=` key and its `(?P<launches>\d+)`
    // capture are untouched.
    let on_dev_f64_fused =
        lgbm_compute::kernels::grow_driver::on_device_f64_fused_count_take();
    // The BLOCKING-READBACK sync total — DISTINCT from `device_launches` (dispatches).
    // Counts only real device→host syncs (scan / on-device partition / tree-split
    // readbacks; a co-packed sibling scan is ONE). Taken unconditionally so the
    // process-global counter resets each dump; folded into the COUNTS line below.
    let on_dev_syncs =
        lgbm_compute::kernels::grow_driver::on_device_sync_count_take();
    // The RESIDENT-PERM partition tripwire (LGBM_PARTITION_RESIDENT=1): one bump per
    // split partitioned on the device-resident perm arm. NONZERO proves the arm ran
    // (bench protocol: counts confirmation before trusting a wall delta); 0 on the
    // default host-partition path. Taken unconditionally so the counter resets each
    // dump; folded INSIDE the parenthetical breakdown (never before the leading
    // total) so the `device_launches=(?P<launches>\d+)` capture stays stable.
    let on_dev_partition_resident =
        lgbm_compute::kernels::grow_driver::on_device_partition_resident_count_take();
    // The PARGAIN scan tripwire (LGBM_SCAN_PARGAIN=1): one bump per staged-scan
    // launch that dispatched the parallel-candidate kernel. NONZERO proves the
    // pargain arm ran; 0 on the default serial-branch staged path.
    let scan_pargain = lgbm_compute::kernels::split::scan_pargain_count_take();
    // The PARPREFIX scan tripwire (LGBM_SCAN_PARPREFIX=1): one bump per staged-scan
    // launch that dispatched the parallel-PREFIX kernel (replaces pargain's phase 1).
    // NONZERO proves the parprefix arm ran.
    let scan_parprefix = lgbm_compute::kernels::split::scan_parprefix_count_take();
    // The FUSED-SUBTRACT tripwire (LGBM_SUBTRACT_FUSE=1): one bump per split that folded
    // the subtraction trick into the co-pack scan (dropping the separate subtract launch).
    // NONZERO proves the fused arm ran; 0 on the default separate-subtract path.
    let subtract_fused =
        lgbm_compute::kernels::grow_driver::on_device_subtract_fused_count_take();
    // The DESC-HOIST tripwire (LGBM_DESC_HOIST=1): one bump per scan/build launch that
    // consumed the per-grow CACHED descriptor handles instead of re-uploading them.
    // NONZERO proves the hoist arm ran; 0 on the default per-launch-upload path.
    let desc_hoist = lgbm_compute::kernels::split::scan_desc_count_take();
    // The SMEM partition BC-fusion tripwire (LGBM_PARTITION_FUSE_BC_SMEM=1, real device):
    // one bump per split that took the 2-launch SharedMemory partition. NONZERO proves it
    // ran; 0 on the default 3-launch / cpu path.
    let partition_bc_smem =
        lgbm_compute::kernels::grow_driver::on_device_partition_bc_smem_count_take();
    if bld_cnt + sub_cnt + scn_cnt + fus_cnt + on_dev > 0 {
        let launches = bld_cnt + sub_cnt + scn_cnt + fus_cnt + on_dev;
        // `device_launches=` is a build+subtract+scan subtotal at PER-LEAF granularity
        // (WR-01/IN-02): both the host `*_resident=` terms and `on_device=` use the same
        // per-leaf unit, and tree-mutation/partition dispatches are excluded on both paths.
        // The `device_launches=` KEY is kept verbatim (no rename) so the harness
        // regex `device_launches=(?P<launches>\d+)` still matches; the unit is annotated
        // by the trailing `launch_unit=...` token instead of by renaming the field.
        eprintln!(
            "[phase_prof:{label}] COUNTS: device_launches={launches} (build_resident={bld_cnt} subtract_resident={sub_cnt} scan_resident={scn_cnt} fused={fus_cnt} on_device={on_dev} on_device_rootbuild_u64={on_dev_rootbuild_u64} on_device_f64_fused={on_dev_f64_fused} partition_resident={on_dev_partition_resident} scan_pargain={scan_pargain} scan_parprefix={scan_parprefix} subtract_fused={subtract_fused} desc_hoist={desc_hoist} partition_bc_smem={partition_bc_smem}) | scan_roundtrips(syncs)={scn_cnt} | blocking_readbacks(syncs)={on_dev_syncs} | launch_unit=build+subtract+scan,per-leaf"
        );
    }
    // The on-device growth-loop PHASE LEDGER — attribution inside the
    // `grow_tree_on_device_resident` black box (`in_learner_other=100%` by design on that
    // path). `host_other = wall − Σ(buckets)` is the honesty check: everything the buckets
    // missed (loop bookkeeping, SplitInfo handling, allocator). Taken unconditionally so the
    // process-global counters reset each dump; printed only when the resident grow ran.
    // NOTE the free-run aliasing contract (grow_driver ledger header): async phases hold
    // SUBMISSION time; their device time drains inside the blocking buckets (scan/pick/
    // device-partition). `LGBM_GROW_DRAIN=1` de-aliases at the cost of the schedule.
    let grow = lgbm_compute::kernels::grow_driver::on_device_grow_phase_take();
    if grow.wall > 0 {
        let sum = grow.setup
            + grow.upload
            + grow.rootfold
            + grow.build
            + grow.subtract
            + grow.scan
            + grow.pick
            + grow.partition
            + grow.treesplit
            + grow.reduce
            + grow.tail;
        let host_other = grow.wall.saturating_sub(sum);
        let ms = |n: u64| n as f64 / 1e6;
        eprintln!(
            "[phase_prof:{label}] ONDEV_GROW: wall={:.3}ms = setup={:.3} upload={:.3} rootfold={:.3} build={:.3} subtract={:.3} scan={:.3} pick={:.3} partition={:.3} treesplit={:.3} reduce={:.3} tail={:.3} host_other={:.3} | coverage={:.1}% drain={}",
            ms(grow.wall),
            ms(grow.setup),
            ms(grow.upload),
            ms(grow.rootfold),
            ms(grow.build),
            ms(grow.subtract),
            ms(grow.scan),
            ms(grow.pick),
            ms(grow.partition),
            ms(grow.treesplit),
            ms(grow.reduce),
            ms(grow.tail),
            ms(host_other),
            if grow.wall > 0 { sum as f64 * 100.0 / grow.wall as f64 } else { 0.0 },
            if std::env::var("LGBM_GROW_DRAIN").map(|v| v == "1").unwrap_or(false) { 1 } else { 0 },
        );
    }

    // in_learner_other sub-breakdown.
    let root_fold = ROOT_FOLD_NS.swap(0, Ordering::Relaxed);
    let partition_new = PARTITION_NEW_NS.swap(0, Ordering::Relaxed);
    let resident_reset = RESIDENT_RESET_NS.swap(0, Ordering::Relaxed);
    if binning + grad + learner + score > 0 {
        let in_learner_other = learner.saturating_sub(b + h + p);
        let upload = UPLOAD_NS.swap(0, Ordering::Relaxed);
        let scratch = SCRATCH_NS.swap(0, Ordering::Relaxed);
        let residual = in_learner_other
            .saturating_sub(root_fold + partition_new + resident_reset + upload + scratch);
        eprintln!(
            "[phase_prof:{label}] IN_LEARNER_OTHER={:.3}ms = root_fold={:.3} + partition_new={:.3} + scratch_setup={:.3} + resident_reset={:.3} + resident_bin_upload={:.3} + residual={:.3}",
            in_learner_other as f64 / 1e6,
            root_fold as f64 / 1e6,
            partition_new as f64 / 1e6,
            scratch as f64 / 1e6,
            resident_reset as f64 / 1e6,
            upload as f64 / 1e6,
            residual as f64 / 1e6,
        );
        eprintln!(
            "[phase_prof:{label}] BUDGET: binning={:.3}ms grad={:.3}ms learner={:.3}ms (phases={:.3} in_learner_other={:.3} [of which resident_bin_upload={:.3}]) score={:.3}ms",
            binning as f64 / 1e6,
            grad as f64 / 1e6,
            learner as f64 / 1e6,
            (b + h + p) as f64 / 1e6,
            in_learner_other as f64 / 1e6,
            upload as f64 / 1e6,
            score as f64 / 1e6,
        );
    }
}
