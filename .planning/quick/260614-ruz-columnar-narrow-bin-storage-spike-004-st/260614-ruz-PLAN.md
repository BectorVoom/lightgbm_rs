---
phase: quick-260614-ruz
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/lgbm-treelearner/src/learner.rs
  - crates/lgbm-treelearner/src/data_partition.rs
  - crates/lgbm-compute/src/lib.rs
  - crates/lgbm/src/booster.rs
  - crates/lgbm-boosting/src/gbdt.rs
  - crates/oracle-harness/tests/learner_parity.rs
  - crates/oracle-harness/tests/advanced_parity.rs
autonomous: true
requirements: [SPIKE-004]

must_haves:
  truths:
    - "Each FeatureColumn stores its bin column in the narrowest unsigned type for its num_bin (u8 <=256, u16 <=65536, else u32) — no parallel u32 copy in production."
    - "The hot CPU build_leaf_histograms_raw fold reads the narrow type directly per feature (monomorphic match on width, widen to usize at read) — the f64 fold order/values are byte-identical to HEAD, so the tree is bit-exact."
    - "Every COLD reader of a feature's bins (partition split, categorical split, bagging subset gather, once-per-train bin-range validation, DART/RF/predict scatter, GPU/Rocm upload) reads via a widening accessor and is unchanged in behavior."
    - "The GPU/Rocm path is byte-unchanged: the learner widens the narrow column to u32 once at upload (cold), and RocmBackend::build_leaf_histograms_raw still receives &[&BinColumn] but widens to u32 internally (its resident u32 upload path is byte-unchanged)."
    - "Large-row train improves substantially (build win) AND small-row train does not regress, measured by interleaved A/B bench_crossover."
  artifacts:
    - path: "crates/lgbm-compute/src/lib.rs"
      provides: "BinColumn enum { U8(Vec<u8>), U16(Vec<u16>), U32(Vec<u32>) } DEFINED HERE (lowest crate, owns Backend trait + hot fold) with new(Vec<u32>, num_bin)->narrowest, bin(row)->u32 (widening), len(), gather(&[u32])->BinColumn, iter_u32(), to_u32_vec(); CPU default build_leaf_histograms_raw takes &[&BinColumn] and folds per-feature via a width match"
      contains: "enum BinColumn"
    - path: "crates/lgbm-treelearner/src/learner.rs"
      provides: "re-exports BinColumn from lgbm-compute; FeatureColumn.bins: BinColumn; every reader migrated to the widening accessor / gather / to_u32_vec"
      contains: "BinColumn"
  key_links:
    - from: "crates/lgbm/src/booster.rs:484"
      to: "BinColumn::new"
      via: "narrow column built ONCE at FeatureColumn construction from value_to_bin + num_bin"
      pattern: "BinColumn::new"
    - from: "crates/lgbm-treelearner/src/learner.rs build_leaf_histogram_into"
      to: "Backend::build_leaf_histograms_raw"
      via: "&[&BinColumn] for the CPU fold"
      pattern: "build_leaf_histograms_raw"
    - from: "crates/lgbm-treelearner/src/data_partition.rs:127"
      to: "BinColumn::bin"
      via: "widening accessor in the leaf bin gather"
      pattern: "\\.bin\\("
---

<objective>
Store each `FeatureColumn`'s bins in the narrowest unsigned type for its `num_bin`
(u8 <=256, u16 <=65536, else u32), so the HOT histogram fold gathers from a cache-dense
column. Spike 004 (validated micro-bench) proved the isolated gather+fold drops -58% (u8)
/ -41% (u16) at 200k x 32 from L2 cache density; build is ~90% of large-row train (2.74s
after spike-003 + r4o), so this is a large-row lever. Small rows already fit cache, so no
small regression is expected.

Purpose: close more of the R3 perf gap at large rows, bit-exactly, faithful to C++
`DenseBin<uint8_t>/<uint16_t>/<uint32_t>` (which picks the narrowest bin type per feature
for exactly this reason). The bin VALUE is unchanged — stored narrower, widened at fold
time — so the f64 fold order + values are identical => bit-exact. This is NOT a fold-order
change.

Output: a `BinColumn { U8/U16/U32 }` enum DEFINED in `lgbm-compute` (the lowest crate, which
owns the `Backend` trait + the hot fold) and RE-EXPORTED from `lgbm-treelearner`, replacing
`FeatureColumn.bins: Vec<u32>`, with a widening `bin(row)->u32` accessor for cold readers and
a per-width monomorphic fold in the CPU `build_leaf_histograms_raw`. The GPU/Rocm override
stays byte-unchanged (widens to u32 internally / keeps its resident u32 upload). Full 3-way
enum (u8 + u16 + u32) is in scope — u8 (default max_bin=255) carries the win and is the
MUST-HAVE; u16/u32 are cheap to add in the same enum.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/spikes/004-columnar-u8-bins/README.md
@CLAUDE.md

# The validated micro-bench (the proof, the exact inner loop to monomorphize)
@crates/lgbm/examples/bin_width_microbench.rs

# FeatureColumn struct + every reader of .bins
@crates/lgbm-treelearner/src/learner.rs
@crates/lgbm-treelearner/src/data_partition.rs

# The CPU hot fold (default impl) AND the Rocm override — BinColumn is DEFINED in this crate
# (lowest crate; lgbm-treelearner depends on lgbm-compute, NOT vice versa — Cargo.toml:13)
@crates/lgbm-compute/src/lib.rs

# The two production construction sites
@crates/lgbm/src/booster.rs

# Production DART/RF/predict scatter readers of f.bins[row] (hard compile errors under BinColumn)
@crates/lgbm-boosting/src/gbdt.rs
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: BinColumn enum (in lgbm-compute) + widening accessor + migrate ALL readers/constructors</name>
  <files>crates/lgbm-compute/src/lib.rs, crates/lgbm-treelearner/src/learner.rs, crates/lgbm-treelearner/src/data_partition.rs, crates/lgbm/src/booster.rs, crates/lgbm-boosting/src/gbdt.rs, crates/oracle-harness/tests/learner_parity.rs, crates/oracle-harness/tests/advanced_parity.rs</files>
  <behavior>
    - BinColumn::new(vec![0,1,255], num_bin=256) selects U8; len 3; bin(2)==255.
    - BinColumn::new(vec![0,300], num_bin=300) selects U16; bin(1)==300.
    - BinColumn::new(vec![0,70000], num_bin=70000) selects U32; bin(1)==70000.
    - Width is selected by num_bin (NOT by observed max): new(vec![0,1], num_bin=256)==U8 even though max==1.
    - bin(row) widens to u32 for every variant; gather(&rows) re-narrows to the SAME width as self.
    - to_u32_vec() round-trips: BinColumn::new(v.clone(), nb).to_u32_vec()==v for all three widths.
  </behavior>
  <action>
    DEFINE `pub enum BinColumn { U8(Vec<u8>), U16(Vec<u16>), U32(Vec<u32>) }` in **lgbm-compute** (crates/lgbm-compute/src/lib.rs — the lowest crate, which owns the `Backend` trait + the hot fold). lgbm-treelearner ALREADY depends on lgbm-compute (Cargo.toml:13), so this is the dependency-correct home; do NOT put BinColumn in lgbm-treelearner and import it into lgbm-compute (that is a dependency CYCLE). `#[derive(Clone, Debug, PartialEq)]` on the enum (Clone is needed by the `bins: x.bins.clone()` test sites; PartialEq by the booster parity assert). Methods:
    `new(bins: Vec<u32>, num_bin: u32) -> Self` selecting U8 if num_bin<=256, U16 if num_bin<=65536, else U32 (cast each element; the once-per-train gate guarantees bin<num_bin so the cast never truncates — assert in debug via `debug_assert!(b < num_bin)`); `bin(&self, row: usize) -> u32` (widening match per variant); `len(&self) -> usize`; `is_empty`; `gather(&self, rows: &[u32]) -> BinColumn` that re-gathers preserving the SAME width (match self, map each row's value into the same Vec<T>); `to_u32_vec(&self) -> Vec<u32>` (widen the whole column — used only by the cold GPU upload + parity test asserts); `iter_u32(&self) -> impl Iterator<Item=u32> + '_` for the validation/most_freq scans.
    In lgbm-treelearner/src/learner.rs RE-EXPORT it (`pub use lgbm_compute::BinColumn;`) and replace `pub bins: Vec<u32>` with `pub bins: BinColumn` on FeatureColumn; in `impl Default` set `bins: BinColumn::U32(Vec::new())`.
    BLANKET RULE (apply across ALL files listed in <files>): EVERY `f.bins[<idx>]` index read (in production OR test) migrates to `f.bins.bin(<idx> as usize)` (returns u32; apply the same `as f64`/`as usize` cast the prior `f.bins[..] as T` did). EVERY `&f.bins` passed where a `&[u32]` slice was expected migrates per the accessor/seam rules below (`&BinColumn` + `.bin(..)`, or `.to_u32_vec()` for a one-shot owned widen). EVERY `bins: <vec>` struct-literal field becomes `bins: BinColumn::new(<vec>, <num_bin>)`. After this task there must be ZERO residual `.bins[<idx>]` indexing and ZERO `Vec<u32>`-typed `.bins`.
    Migrate EVERY reader (enumerate — wider blast radius is the whole point of this task):
    (1) booster.rs:484/493 (build_feature_columns_from_raw) — `bins: BinColumn::new(bins, num_bin)`; booster.rs:223 (build_feature_columns identity path) — same, `BinColumn::new(bins, num_bin)`.
    (2) learner.rs:512 + 556 (bagging subset gathers in train_on_subset / train_on_subset_returning_partition) — replace `let bins: Vec<u32> = in_bag.iter()...collect()` with `let bins = f.bins.gather(<in_bag widened to u32>)` so the subset column keeps the narrow width (in_bag is `&[i32]`; cast to u32). Construct the subset FeatureColumn with this BinColumn.
    (3) learner.rs:700-726 (once-per-train V5 bin-range + length gate) — `f.bins.len()` stays; the per-element loop reads `for b in f.bins.iter_u32() { if b >= f.num_bin {..BinIndexOutOfRange..} }`. KEEP this gate verbatim in semantics: width selection guarantees the bin VALUE FITS the type, but the `bin < num_bin` VALUE check must still run (a u8 column can still hold a value >= a num_bin that is < 256). Do NOT weaken it.
    (4) learner.rs:784 (GPU upload) — `let upload_bins: Vec<Vec<u32>> = features.iter().map(|f| f.bins.to_u32_vec()).collect();` then pass `&[&[u32]]` views to `upload_resident_bins` (widen ONCE here, cold; the Rocm path is byte-unchanged).
    (5) learner.rs:2808 (data_partition.split) + 2984 (split_categorical) — these take `feature_bins: &[u32]`. The cleanest: pass `&f.bins` (a `&BinColumn`) and migrate data_partition.rs:107-130/184 to accept `&BinColumn` and gather via `.bin(row as usize)` in the leaf_feature_bins loop (line 127). Keep the partition LOGIC identical — only the per-row bin READ switches to the accessor.
    (6) PRODUCTION scatter readers in gbdt.rs — gbdt.rs:761, 1177, 1308 each read `v[f.real_feature_index as usize] = f.bins[row as usize] as f64;` in the DART/RF/predict-path feature-row scatter over `Vec<FeatureColumn>`. These are NOT test ctors — they are hard compile errors under `bins: BinColumn`. Migrate each to `v[f.real_feature_index as usize] = f.bins.bin(row as usize) as f64;` (semantics identical; `.bin()` returns the same widened u32 the old index produced).
    (7) TEST index readers in learner_parity.rs — lines 337 (`f.bins[i]`), 344/345 (`f.bins[i] <= 2` / `> 2` split-row filters), 357/364 (`f.bins[i]` in smaller/larger row gathers), 808 (`f.bins[row] as usize`) all index-read `.bins` and break compile under BinColumn; migrate each to `f.bins.bin(i)` / `f.bins.bin(row) as usize`. Where a whole `Vec<u32>` is collected (337/357/364), either `.map(|&i| f.bins.bin(i))` or `f.bins.to_u32_vec()` then index. Keep the comparison/filter LOGIC identical.
    (8) All TEST construction sites that write `bins: vec![...]` (learner.rs:3435/3469/3482/3840 + the FeatureColumn{} literals at 3353/3434/3468/3481/3839; learner_parity.rs:233/246/389/554/713/1024/1444/1786/1986; advanced_parity.rs:138; gbdt.rs:1692/1705/1991/1996/2149/2154) — wrap each `bins: <vec>` as `bins: BinColumn::new(<vec>, <num_bin literal already in that struct>)`. learner_parity.rs:1445 (`bins: s.bins.clone()`) and 1787 (`bins: f.bins.clone()`) already produce a `BinColumn` once the source field is migrated — the derived `Clone` makes these compile unchanged. booster.rs:1925 parity assert (`assert_eq!(idc.bins, brc.bins)`) — the derived `PartialEq` makes this compile unchanged (or compare via `.to_u32_vec()` on both sides).
    Add unit tests for the behaviors above (width selection by num_bin, widening accessor, gather-preserves-width, to_u32_vec round-trip) in lgbm-compute's test module.
    Do NOT add a parallel u32 copy anywhere in production (no memory doubling). Do NOT touch the Rocm/cubecl kernels. Never git-add LightGBM/.
  </action>
  <verify>
    <automated>cargo test -p lgbm-compute && cargo test -p lgbm-treelearner && cargo build -p lgbm -p lgbm-boosting -p oracle-harness --tests</automated>
  </verify>
  <done>BinColumn defined in lgbm-compute (3 width variants + accessors, Clone/Debug/PartialEq) and re-exported from lgbm-treelearner; every reader migrated (incl. gbdt.rs:761/1177/1308 production scatter + learner_parity.rs:337/344/345/357/364/808 test index reads); no residual `.bins[..]` indexing or `Vec<u32>` `.bins`; new BinColumn unit tests pass; lgbm-compute + lgbm-treelearner test suites green; lgbm/lgbm-boosting/oracle-harness build green. No production parallel u32 copy. Rocm path untouched.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Monomorphic narrow fold in CPU build_leaf_histograms_raw (single &[&BinColumn] seam) + bit-exact HARD GATE</name>
  <files>crates/lgbm-compute/src/lib.rs, crates/lgbm-treelearner/src/learner.rs</files>
  <behavior>
    - The CPU default build_leaf_histograms_raw, given narrow BinColumns, produces the SAME f64 out buffer (byte-for-byte) as the prior u32 path on the same inputs.
    - The per-feature inner fold dispatches on width (U8/U16/U32) to a monomorphic tight loop that reads the narrow element and widens to usize at the bin index — no per-iteration branch on width inside the row loop.
    - The GPU/Rocm override receives the SAME &[&BinColumn] signature but widens to u32 internally / uses its resident u32 upload, producing a byte-identical kernel input to HEAD.
  </behavior>
  <action>
    SINGLE CONCRETE SEAM (no per-impl divergence — `build_leaf_histograms_raw` is ONE shared trait method: CPU default at lib.rs ~228, Rocm override at lib.rs ~900, so both impls MUST share the same param type). Change the trait method's `feature_bins` param from `&[&[u32]]` to `&[&BinColumn]` (BinColumn now lives in lgbm-compute per Task 1, so this is in-crate — no cross-crate import, no cycle). Do NOT add a "import BinColumn from lgbm-treelearner" variant; that direction is a dependency cycle and is dropped.
    CPU default body: keep the spike-003 once-per-leaf ord_g/ord_h gather + the reused hot scratch + the branchless fold EXACTLY (no per-element bin check, only debug_assert), but split the inner `for (k,&row) in leaf_rows` loop per width: `match bins { BinColumn::U8(v)=>{ for ... let bin = v[row] as usize; ...} BinColumn::U16(v)=>{...} BinColumn::U32(v)=>{...} }` so each arm is a monomorphic tight loop. The grad at `bin*2`, hess at `bin*2+1`, f32-read->f64-accumulate, ascending leaf_rows order — IDENTICAL to HEAD.
    Rocm/GPU override (lib.rs ~900): now ALSO takes `&[&BinColumn]` (forced by the shared trait sig). Internally widen to u32 (`bc.to_u32_vec()` per feature) OR — preferred, since the GPU already uploads bins resident — ignore the param and use its existing resident u32 upload path; either way the kernel input bytes are IDENTICAL to HEAD. The GPU kernel itself is byte-unchanged.
    Caller (learner.rs ~1687, build_leaf_histogram_into): build `let feature_bins: Vec<&BinColumn> = features.iter().map(|f| &f.bins).collect()` and pass `&feature_bins`. The GPU resident path at learner.rs:1775 continues to use the resident handle / `to_u32_vec` upload (Task 1 item 4) — unchanged. Keep num_bins descriptor cache (build_num_bins) unchanged.
    Then run the HARD GATE (fold order frozen): `cargo test -p lgbm-compute`, `cargo test -p lgbm-treelearner`, `cargo test -p oracle-harness --test learner_parity` (expect 29/0 BIT-EXACT), and full `cargo test -p oracle-harness` (0-failed; DEF-07-02 #[ignore] cells unchanged). If ANY cell that was bit-exact at HEAD diverges, STOP — the fold was perturbed; do not weaken any tolerance.
  </action>
  <verify>
    <automated>cargo test -p lgbm-compute && cargo test -p lgbm-treelearner && cargo test -p oracle-harness --test learner_parity && cargo test -p oracle-harness 2>&1 | tail -40</automated>
  </verify>
  <done>Single `&[&BinColumn]` seam on the shared trait method; CPU fold reads the narrow column per-width monomorphically; Rocm override compiles on the same signature and its kernel input bytes are unchanged (widen-internally or resident-u32); lgbm-compute green; lgbm-treelearner green; learner_parity 29 passed / 0 failed BIT-EXACT; full oracle-harness 0-failed with DEF-07-02 #[ignore] cells unchanged. GPU/Rocm kernel behavior byte-unchanged. clippy clean on edited code.</done>
</task>

<task type="auto">
  <name>Task 3: Interleaved A/B measurement gate (small no-regress, large must-improve)</name>
  <files>crates/lgbm/examples/bench_crossover.rs</files>
  <action>
    Measure the REAL train delta (the standing lesson: A/B both scales before declaring a win). Baseline = current HEAD (small ~26.9ms, large ~2.74s). Run interleaved A/B, 2 rounds each, with `LGBM_PHASE_PROF=1`:
    - SMALL (must NOT regress): `BENCH_SIZES="small:2000:12:32" BENCH_ITERS=100 BENCH_REPS=9 LGBM_PHASE_PROF=1 cargo run --release --example bench_crossover`
    - LARGE (expect a substantial build/train win, < the isolated -58%): `BENCH_SIZES="large:200000:32:64" BENCH_ITERS=50 BENCH_REPS=5 LGBM_PHASE_PROF=1 cargo run --release --example bench_crossover`
    Interleave: run the post-change binary, then `git stash`/checkout HEAD for the baseline binary, alternating to cancel thermal drift (2 rounds). Capture the per-phase BUILD ns from phase_prof for both. REJECT if small regresses (treat noise band ~±2-3%) or large does not improve on the build/train phase. Note u16/u32 coverage in the result (u8 carries the win at default max_bin; large uses 64 bins => u8). Record numbers in the SUMMARY. Do NOT modify production code in this task; bench_crossover edits only if a missing knob is needed (it already supports BENCH_SIZES/ITERS/REPS + phase_prof).
  </action>
  <verify>
    <automated>BENCH_SIZES="large:200000:32:64" BENCH_ITERS=10 BENCH_REPS=2 cargo run --release --example bench_crossover 2>&1 | tail -20</automated>
  </verify>
  <done>Interleaved A/B (2 rounds) captured for small + large with phase_prof BUILD breakdown; small train within the noise band (no regression); large train + build phase improved vs HEAD; numbers recorded in SUMMARY. If small regresses or large fails to improve, the result is REJECTED and the change is reverted/re-scoped (note in SUMMARY).</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| feature-column construction -> learner | bin values from BinMapper::value_to_bin cross into the narrow column; the once-per-train gate is the validation point |
| narrow column -> hot fold | bin index used as an array offset into the 2*num_bin histogram scratch (memory-safety relevant) |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-ruz-01 | Tampering | BinColumn::new width cast (u32->u8/u16) | mitigate | Width is chosen by num_bin (u8<=256/u16<=65536); the once-per-train gate (learner.rs:719) STILL runs the `bin < num_bin` VALUE check via iter_u32 BEFORE any narrowing matters — the accessor does NOT change validation semantics. debug_assert!(b<num_bin) in new() is defense-in-depth. |
| T-ruz-02 | Information disclosure | bin(row) widening accessor as a histogram offset | mitigate | bin(row) widens to u32 identically to the prior Vec<u32> read; the upstream gate guarantees bin<num_bin so `bin*2(+1)` stays inside the 2*num_bin scratch. No per-element check is removed (the relocated T-04-01 gate at learner.rs:707-726 is preserved verbatim — only its READ switches to iter_u32). |
| T-ruz-03 | Tampering | GPU/Rocm path receiving narrowed data | accept/mitigate | The Rocm override shares the `&[&BinColumn]` trait sig but widens to u32 internally (or uses its existing resident u32 upload at learner.rs:784); the GPU kernel input is byte-identical to HEAD. Kernel itself out of scope/untouched. |
| T-ruz-SC | Tampering | npm/pip/cargo installs | mitigate | No new dependencies introduced (BinColumn is in-crate, lgbm-compute). No package-manager install tasks => legitimacy gate N/A. |
</threat_model>

<verification>
- BinColumn unit tests (in lgbm-compute): width selection by num_bin (u8/u16/u32), widening bin() per variant, gather preserves width, to_u32_vec round-trip.
- Full blast-radius migrated: NO residual `.bins[<idx>]` index reads (incl. gbdt.rs:761/1177/1308 production scatter + learner_parity.rs:337/344/345/357/364/808) and NO `Vec<u32>`-typed `.bins` anywhere.
- HARD bit-exact gate (fold order frozen): `cargo test -p lgbm-compute`, `-p lgbm-treelearner`, `-p oracle-harness --test learner_parity` (29/0 BIT-EXACT), full `-p oracle-harness` (0-failed; DEF-07-02 #[ignore] cells unchanged).
- Once-per-train V5 bin-range gate (learner.rs:707-726) preserved verbatim (semantics unchanged; only its read switches to iter_u32).
- Single `&[&BinColumn]` trait seam (no dependency cycle, no per-impl param divergence); GPU/Rocm override compiles on it + kernel input byte-unchanged (widen at upload / resident u32).
- Interleaved A/B (2 rounds) bench_crossover with LGBM_PHASE_PROF=1: small no-regress + large build/train improvement vs HEAD.
- clippy clean on edited code; LightGBM/ never git-added.
</verification>

<success_criteria>
- BinColumn enum DEFINED in lgbm-compute (the lowest crate, owns Backend + hot fold) and re-exported from lgbm-treelearner; FeatureColumn.bins is a BinColumn storing the narrowest width per num_bin; NO production parallel u32 copy.
- The CPU hot fold reads the narrow type per-width monomorphically via a single `&[&BinColumn]` trait seam (no cycle, no per-impl param split); the f64 fold order/values are byte-identical to HEAD (learner_parity 29/0 BIT-EXACT, full oracle 0-failed).
- Every cold reader (partition split, categorical split, bagging subset gather, once-per-train validation, DART/RF/predict scatter in gbdt.rs, GPU upload) migrated to the widening accessor / gather / to_u32_vec; zero residual `.bins[..]` indexing.
- GPU/Rocm kernel path byte-unchanged.
- Interleaved A/B: small train does not regress AND large train/build improves vs HEAD (numbers in SUMMARY). If not, REJECT + revert/re-scope (noted in SUMMARY).
</success_criteria>

<output>
Create `.planning/quick/260614-ruz-columnar-narrow-bin-storage-spike-004-st/260614-ruz-SUMMARY.md` when done, including the interleaved A/B numbers (small + large, build phase breakdown) and the bit-exact gate results.
</output>
