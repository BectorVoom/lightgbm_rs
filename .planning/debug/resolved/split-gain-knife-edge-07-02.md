---
slug: split-gain-knife-edge-07-02
status: resolved
trigger: "fix ignore (errors) — un-defer the DEF-07-02/03 ignored boosting_parity cells (fair/gamma/quantile/tweedie) by finding and fixing the non-constant-hessian learner-level f64 split-gain knife-edge"
created: 2026-06-07
updated: 2026-06-07
goal: find_and_fix
scope: "Family A only (DEF-07-02 + 07-03 extension). Family B (DEF-07-11 monotone/forced/extra-trees) is explicitly OUT OF SCOPE for this session."
---

# Debug Session: DEF-07-02/03 split-gain knife-edge

## Symptoms (prefilled — user ran `cargo test --workspace`)

- **Expected:** the fair/gamma/quantile/tweedie parity cells assert real-`lib_lightgbm`-4.6 parity and pass (un-`#[ignore]`d).
- **Actual:** 13 cells in `crates/oracle-harness/tests/boosting_parity.rs` are `#[ignore]`d with honest reasons pointing at `.planning/phases/07-parity-completing-variants/deferred-items.md` (DEF-07-02 / 07-03). `cargo test --workspace` = 0 failed, but these are deferred, not fixed.
- **Errors:** no panic/failure — the divergence is numerical: tree leaf values / tree counts diverge from real C++ once a borderline split flips on a single-/few-ULP f64 boundary.
- **Timeline:** deferred during phase 07-02 (fair, quantile) and 07-03 (gamma, tweedie). Blocked on the same prerequisite the 07-01 D-05 fix used.
- **Reproduction:** `LGBM_CAPTURE_PYTHON=/tmp/lgbm-capture-venv/bin/python cargo test --workspace` shows them as `ignored`.

## Affected (ignored) cells — Family A

DEF-07-02:
- `fair_spine_end_to_end`, `fair_score_accumulation`, `fair_gradients`, `fair_loop_matrix`, `fair_c_axis` (fair = only family-A objective with NON-constant hessian `c²/(|x|+c)²`)
- `quantile_loop_matrix`, `quantile_alpha_axis` (bagged: 12-vs-10-tree structural divergence at tree 4; non-bagged iterated: tree-11 flip)

DEF-07-03 extension (same root cause):
- `gamma_spine_end_to_end`, `gamma_score_accumulation`, `gamma_gradients`, `gamma_loop_matrix` (hessian `label·exp(-score)`, non-uniform at iter 0 → diverges tree 0)
- `tweedie_loop_matrix`, `tweedie_variance_power_axis` (bfa-OFF + ρ-axis diverge tree 0; tweedie SPINE stays GREEN)

GREEN guard cells that must NOT regress: `*_spine`/`*_gradients` for poisson, cross_entropy, cross_entropy_lambda, huber, mape, tweedie-default-ρ, quantile-spine; plus `kernel_parity` (4/4 bit-exact), `learner_parity` (spine_real_binary / mfb_pos_real_binary), `subset_determinism_diagnostic`.

## Current Focus

hypothesis: "The non-constant-hessian families hit the SAME class of f64 split-gain operand bug that 07-01/D-05 fixed for the bagged-subset case — a `current_gain` vs `min_gain_shift` comparison landing on a 1-few ULP f64 boundary in `find_best_split`. g/h INTO each tree are already bit-exact (verified: tweedie_gradients passes iter-1+iter-4, gamma_gradients passes iter-1), so this is NOT an objective bug — it's the tree learner's f64 histogram/split selection over the faithful non-constant-hessian g/h. Suspect: another min_gain_shift / cnt_factor / get_leaf_gain operand that diverges only when the hessian is non-uniform (07-01 fixed the 2*kEpsilon-bumped sum_hessian; this may be a sibling operand that the constant-hessian families happened to round identically on)."
test: "Build instrumented lib_lightgbm 4.6 FP trace; drive the EXACT diverging cell (start with gamma_spine_end_to_end — diverges at tree 0, simplest reproduction); dump per-node current_gain / min_gain_shift / cnt_factor / sum_gradient / sum_hessian as .to_bits() and compare to the Rust find_best_split per_bin_gains diagnostic for the same node."
expecting: "A specific f64 operand (or rounding/order) in find_best_split that differs by 1-few ULPs between Rust and C++ ONLY for non-constant hessians, flipping which split is accepted."
next_action: "ROOT CAUSE FOUND (see Resolution). Fix is architectural (Rule 4) — plan a
dedicated learner-side most_freq_bin==0/offset histogram-representation fix (build_leaf_histogram_into
/ fix_histogram / compact_histogram) under the strict no-regression constraint, then un-ignore the
DEF-07-02/03 Family-A cells. Reproduction assets retained: /tmp/LightGBM (source build, reverted),
/tmp/gamma_fp/{train.csv,train.conf,fptrace.log,model_cpp.txt}."
reasoning_checkpoint: "Investigation complete. Eliminated min_gain_shift-ULP, subtraction-trick,
objective-math, and the scan-kernel hypotheses by FP trace + LGBM_NO_SUBTRACT control. Localized to
the offset/most_freq_bin histogram representation for non-constant hessians via a bit-matched-header
per-bin g/h diff against a source-built lib_lightgbm 4.6 that reproduces the golden bit-exact."

## Reusable playbook — the 07-01 / D-05 method (proven, un-deferred DEF-06-01)

Source: `.planning/phases/07-parity-completing-variants/07-01-SUMMARY.md` + `07-D05-DECISION.md`.

1. **Build the reference:** lib_lightgbm 4.6 (`VERSION.txt = 4.6.0.99`) CPU-only single-thread, CMake flags `-DUSE_GPU=OFF -DUSE_CUDA=OFF -DUSE_OPENMP=OFF -DBUILD_CLI=ON`, into `/tmp`, against the repo's populated `external_libs` (eigen/fmt/fast_double_parser). NEVER `git add LightGBM/` (memory: lightgbm-ref-tree-untracked). Revert C++ instrumentation after capture.
2. **Instrument:** `FindBestThresholdSequentially` / `FindBestThreshold` env-gated on an env var (07-01 used `LGBM_FP_TRACE`), dumping per-node HEADER (`sum_gradient`/`sum_hessian`/`cnt_factor`/`min_gain_shift`), per-bin `SUBSET_HIST`, and per-candidate `current_gain` vs `min_gain_shift` + accept flag, all as `.to_bits()` hex.
3. **Validate the build reproduces the wheel:** confirm the source CLI model is bit-identical to the wheel-captured trace before trusting it.
4. **Read the genuine operand:** find the 1-few-ULP divergence. 07-01's was: C++ bumps `sum_hessian` by `2*kEpsilon` at the FindBestThreshold call site (`feature_histogram.hpp:174`) so `BeforeNumerical` divides by the BUMPED value; Rust used the RAW value → min_gain_shift ~7 ULPs too high → rejected splits whose current_gain beat the C++ shift by 1 ULP.
5. **Faithful fix in:** `crates/lgbm-compute/src/kernels/split.rs` (`find_best_split_cpu` f64 AND `find_best_split_raw_f32_on` f32), `crates/lgbm-treelearner/src/learner.rs` (`per_bin_gains` diagnostic — keep bit-identical to the live kernel), and check `xtask/cpp/kernel_capture.cpp` (`EmitSCase`) for the same transcription bug → regenerate `split.txt` golden (byte-idempotent).
6. **Un-defer:** remove `#[ignore]` from the now-passing cells, assert real-binary parity, clear the entries in `deferred-items.md`.

## Key source locations

- Split-gain kernel: `crates/lgbm-compute/src/kernels/split.rs` (`find_best_split_cpu`, `find_best_split_raw_f32_on`, `get_leaf_gain`)
- Learner diagnostic: `crates/lgbm-treelearner/src/learner.rs` (`per_bin_gains`)
- Tests: `crates/oracle-harness/tests/boosting_parity.rs` (ignored fair/gamma/quantile/tweedie cells)
- Golden capture transcription: `xtask/cpp/kernel_capture.cpp` (`EmitSCase`)
- Objective hessians (faithful, do NOT change): fair `c²/(|x|+c)²`, gamma `label·exp(-score)`, tweedie `-label·(1-ρ)·exp((1-ρ)·score)+(2-ρ)·exp((2-ρ)·score)`
- Reference C++: `LightGBM/src/treelearner/feature_histogram.hpp` (line ~174 = the 2*kEpsilon bump call site), `LightGBM/src/io/...`

## Environment facts (memory-confirmed)

- Real working ROCm GPU (gfx1100) + cubecl-hip available; CPU f64-fold path is the hard merge gate.
- `external_libs` CAN be fetched and real CPU `lib_lightgbm` 4.6 builds here for FP-trace parity debugging (memory: lightgbm-ref-tree-untracked). A pip wheel `lib_lightgbm.so` exists at `/tmp/lgbm-capture-venv/.../lib_lightgbm.so` but the wheel CANNOT emit the internal FP trace — an instrumented source build is required (per deferred-items.md).
- NEVER `git add LightGBM/`; worktrees break for phases needing it.

## Evidence

- timestamp: 2026-06-07 (session split-gain-knife-edge-07-02)
  finding: "gamma_spine tree-0 is NOT a 1-ULP knife-edge — it is a GROSS gain
  divergence on the histogram-SUBTRACTION (larger) child. Concrete dump (rust vs golden):
  ROOT (node0): BOTH split f0@1.5 gain=5.069 → left=4-row (f0∈{0,1}), right=8-row (f0∈{2,3,4,5}). MATCH.
  The 8-row right node's best-split gain: RUST=1.0416667 vs GOLDEN=0.375479 (~2.8x too high).
  Because the right-node gain is wrong, leaf-selection order + final topology flip:
  RUST grows leaf_count=[1,8,2,1] (keeps splitting the 4-row left side to 1-row leaves, makes
  the 8-row side a LEAF), GOLDEN grows leaf_count=[2,4,2,4] (splits the 8-row side f0@3.5 gain 0.375).
  RUST tree0 split_gain=[5.0689654, 1.8, 1.0416667]; GOLDEN=[5.06897, 1.8, 0.375479].
  Note BOTH the root (5.069) and the 4-row DIRECT-built child (1.8) match — only the
  SUBTRACTION-derived 8-row child's scan is wrong. tweedie/fair share this (non-constant hessian)."
  source: temp_gamma_tree0_dump (boosting_parity.rs), golden gamma_spine_model.txt

- timestamp: 2026-06-07 (FP-trace, instrumented lib_lightgbm 4.6)
  finding: "REUSED the 07-01 source-built /tmp/LightGBM build (separate clone, NOT the repo
  reference tree — instrumenting it never touches repo LightGBM/). Instrumented
  FindBestThresholdSequentially (env LGBM_FP_TRACE) with HEADER/BIN/RESULT .to_bits() dumps.
  CLI config (force_row_wise, min_data_in_bin=1, bin_construct_sample_cnt=1e6, gamma, 5 iter,
  lr 0.1, num_leaves 4, min_data_in_leaf 1, bfa on, seed 1610903552, num_threads 1)
  REPRODUCES the golden gamma tree-0 BIT-EXACT (split_gain=5.06897 1.8 0.375479,
  leaf_count=2 4 2 4) — playbook step-3 validation PASSED.
  C++ ground-truth for the 8-row larger node (sum_g=-2.5454545915126801,
  sum_h=10.545454621315002, num_data=8, cnt_factor=0.75862068419791007,
  min_gain_shift=0.61442008051095487), feature-0 (num_bin=6, the SUBTRACTED child):
  the C++ subtracted histogram bin 5 = h=1.0325971998082053e-321 (bits 0x...00d1) — a
  DENORMAL garbage residual from parent-minus-smaller, effectively 0. C++ best_gain=0.98989901,
  net_gain=0.3754789337383051 (= golden). Rust computed 1.0416667 for this node.
  → The divergence is in the 8-row SUBTRACTED histogram / its scan, NOT min_gain_shift.
  Need the Rust subtracted-histogram bins for the SAME node to localize the differing bin."
  source: /tmp/gamma_fp/fptrace.log lines 32-39

- timestamp: 2026-06-07 (FP-trace, decisive divergence found)
  finding: "DECISIVE: the 8-row SUBTRACTED node is CORRECT in Rust (find_best_split returns
  net_gain 0.37547893373830510, BIT-IDENTICAL to C++). The real divergence is a 2-ROW node
  (a grandchild — child of the 4-row f0∈{0,1} node). SAME header in C++ and Rust
  (sum_g=1.0000000298023224, sum_h=1.0000000298023244, num_data=2, min_gain_shift=1.0000000298),
  but DIFFERENT histograms → different gain:
    C++  feat num_bin=3 bins: t0 g=0 h=0; t1 g=0.5454 h=0.4545; t2 g=1.0 h=1.0  → net_gain 0.033333
    RUST feat num_bin=3 bins: t0 g=0.7272 h=0.2727; t1 g=0 h=0; t2 g=0 h=0       → net_gain 1.041667
  Rust's histogram mass is all in BIN 0; C++'s is in bins 1,2. Because Rust's bogus 1.0417 gain
  beats the 8-row node's correct 0.375479, Rust picks the wrong 3rd split → topology flip
  (leaf_count [1,8,2,1] vs golden [2,4,2,4]).
  The 2-row node is the SUBTRACTION-derived child (mass-in-bin-0 == a subtract artifact when the
  parent's mass and the smaller sibling's mass cancel into the most_freq/offset bin). This is the
  histogram-SUBTRACTION-trick interacting with non-constant hessians + the offset/most_freq_bin
  compaction — the 4-row node's TWO 2-row children: one built direct (correct), one via subtract
  (mass collapses to bin 0). NEXT: localize in build_leaf_histogram_into / subtract_histograms /
  compact_histogram / fix_histogram for the grandchild subtract path."
  source: /tmp/gamma_fp/fptrace.log line 40-44 (C++) vs RST trace (num_data=2, sum_g=1.0)

- timestamp: 2026-06-07 (root cause confirmed — compaction/offset histogram representation)
  finding: "ROOT CAUSE: histogram COMPACTION (offset for most_freq_bin==0) drops real bin 0's
  mass and Rust's compacted layout places the surviving mass in the WRONG bin vs C++ for the
  SAME node. Same node confirmed bit-exact by header match: rows[0,1], feature 1, num_bin=3,
  sum_g=1.0000000298023224, sum_h=1.0000000298023244, num_data=2.
    C++ physical f1 hist (GET_GRAD(data_,t)): bin0=(0,0)  bin1=(0.5454,0.4545)  bin2=(1.0,1.0)
       C++ scan reads bins t=1..0 → {bin0:(0,0), bin1:(0.5454,0.4545)}; bin2 is the
       FixHistogram/leaf-total cell NEVER read by the offset=1 scan.
    RUST compacted f1 hist: bin0=(0.7272,0.2727)  bin1=(0,0)  bin2=(0,0)
       Rust scan reads compacted bins → {bin0:(0.7272,0.2727), bin1:(0,0)}.
  C++ has the split mass at the HIGHER scan bin (0.5454); Rust collapsed it to compacted bin0
  with a DIFFERENT value (0.7272). Net: C++ best split here net_gain=0.033333, Rust=1.041667.
  This is the histogram-SUBTRACTION + COMPACTION (compact_histogram offset shift) path for the
  grandchild leaf rows[0,1]: derived f1=[0.7272,0.2727,0,0,0,0] (from RSTB SUBTRACT trace,
  parent f1=[0.7272,0.2727,0.5454,0.4545,0,0] minus smaller f1=[0,0,0.5454,0.4545,0,0]).
  The DENORMAL garbage in the 8-row C++ subtracted bin5 (0x..d1) earlier was a benign sign the
  C++ subtract is over the FULL (non-compacted) buffer including the leaf-total cell, whereas
  Rust subtracts over a COMPACTED buffer where the leaf-total/mfb cell is positioned differently.
  => The compaction model (`compact_histogram` dropping real bin 0 + FixHistogram no-op for
  mfb==0) is INCONSISTENT with C++'s offset model for the SUBTRACTION child: C++ never physically
  compacts; it keeps the full buffer and the mfb/leaf-total cell, scanning a bounded range.
  Rust's physical compaction loses the bin alignment under subtraction for mfb==0 features."
  source: header bit-match rows[0,1] f1; RSTB SUBTRACT + RST find_best_split traces; C++ line 40-44

- timestamp: 2026-06-07 (CORRECTED + CONFIRMED root cause — NOT subtraction)
  finding: "CORRECTION: the partition trace `rows=[0,1]` are PARTITION-LOCAL indices, not
  original row ids. Per-row gamma g/h at iter0 (init=ln(11)=2.39789527): row0(f1=0)g=0.8181 h=0.1818;
  row1(f1=1)g=0.7272 h=0.2727; row2(f1=2)g=0.5454 h=0.4545; row3(f1=0)g=0.4545 h=0.5454.
  The divergent sum_g=1.0 node = ORIGINAL rows {2,3} (f0=1): g=0.5454+0.4545=1.0, h=1.0. ✓
  CONFIRMED the bug is NOT the subtraction trick: LGBM_NO_SUBTRACT=1 (force direct build of the
  larger child) reproduces the IDENTICAL divergence (gamma_spine tree0 leaf0 still 2.3603 vs golden
  2.0578). The DIRECT build of node{2,3} f1 ALSO yields compacted=[0.7272,0.2727,0,0,0,0] →
  net_gain 1.0417, vs C++ net_gain 0.0333.
  ROOT CAUSE (architectural, Rule 4): for most_freq_bin==0 features (offset=1), Rust's histogram
  representation (construct real-bin → fix_histogram NO-OP for mfb==0 → compact_histogram physically
  shifts real bin c+1→c and DROPS real bin 0) does NOT reproduce C++'s offset MODEL for the per-bin
  g/h placement on these leaves. C++ never physically compacts: it keeps the full buffer with the
  most_freq/leaf-total cell and bounds the scan range via offset, so the bin0 (most_freq) mass is
  carried as the implicit `sum_total − Σ(scanned)` default. Node{2,3}: row3 is f1=0 (==most_freq_bin);
  C++ treats row3's mass as the implicit default and scans row2 at the higher bin → correct split
  {row3}|{row2} net_gain 0.0333. Rust's compaction places node{2,3}'s mass into compacted bin0 with a
  value that yields the wrong split partition → 1.0417.
  WHY ONLY non-constant hessians: with constant hessian the per-bin g²/(h+λ) gains + count gates land
  identically regardless of the bin0-default placement (binary/poisson-iter0/spine pass bit-exact);
  with label-dependent hessians (gamma/fair/tweedie/quantile-iterated) the mis-placed bin0 mass
  changes which partition wins. This is the SAME family as DEF-07-02/03 (all share the offset=1
  most_freq_bin==0 histogram path on small non-constant-hessian leaves).
  SCOPE: must fix the most_freq_bin==0 / offset histogram representation in the learner
  (build_leaf_histogram_into → fix_histogram / compact_histogram) so the bin0-default mass is carried
  faithfully to C++ for non-constant hessians, WITHOUT regressing the bit-exact constant-hessian
  spine/binary/poisson/kernel_parity/learner_parity cells."
  source: /tmp/gamma_fp/check.py (per-row g/h); LGBM_NO_SUBTRACT=1 reproduction; RST/RSTB traces; C++ line 1-4,14-18,40-44

## Eliminated

- "1-few ULP min_gain_shift knife-edge (the 07-01/D-05 sibling-operand hypothesis)" —
  ELIMINATED. The 8-row larger node's find_best_split is BIT-IDENTICAL to C++
  (raw_gain 0x3fefad40b29161f9, net_gain 0.37547893373830510). min_gain_shift is already
  bit-exact (bumped sum_hessian fix from 07-01 holds). The divergence is a GROSS gain
  (1.0417 vs 0.0333), not a ULP flip.
- "histogram subtraction trick" — ELIMINATED. LGBM_NO_SUBTRACT=1 (force direct build of
  the larger child) reproduces the IDENTICAL divergence. The direct build of the offending
  2-row node also yields the wrong histogram/gain.
- "objective gradient/hessian math" — ELIMINATED (already known; re-confirmed: per-row g/h
  match C++ bit-exact; gamma_gradients iter-1 passes).
- "find_best_split scan kernel" — ELIMINATED. Given the same histogram + header it returns
  bit-identical results to C++ (verified on the 8-row node).

## Specialist Review

Intended specialist: numerical/floating-point + systems (Rust). The fix is an architectural
change (deviation Rule 4) to the most_freq_bin==0 / offset histogram representation, which
must precisely mirror C++ `BinMapper::GetMostFreqBin` / `default_bin` / `meta_->offset` and
`Dataset::FixHistogram` cell placement. No specialist tool was available in this isolated
session; the review should confirm C++'s f1 mfb/offset semantics (the C++ FP trace shows the
leaf-total/most_freq cell at the TOP physical bin, not bin 0, for these features) before the
fix lands.

## Resolution

root_cause: "Histogram representation for most_freq_bin==0 features (offset=1) diverges from
C++ for NON-CONSTANT-HESSIAN leaves. Rust physically compacts (construct real-bin →
fix_histogram NO-OP for mfb==0 → compact_histogram shifts real bin c+1→c and DROPS real bin 0),
while C++ never physically compacts — it keeps the full buffer incl. the most_freq/leaf-total
cell and carries bin-0 (most_freq) mass as the implicit `sum_total − Σ(scanned)` default,
bounding the scan via offset. On a small non-constant-hessian leaf (e.g. gamma tree-0 node
{rows2,3}, row3 f1=0==most_freq_bin) Rust mis-places the per-bin g/h so the split partition
flips: net_gain 1.0417 (Rust) vs 0.0333 (C++). The bogus high gain beats the correct 8-row
node's 0.375479, flipping which leaf grows next ⇒ tree-0 topology leaf_count [1,8,2,1] vs
golden [2,4,2,4]. Constant-hessian families are unaffected (the bin-0-default placement cancels
in g²/(h+λ) + the integer count gates), which is why binary/poisson/spine stay bit-exact and
only gamma/fair/tweedie-bfa-off/quantile-iterated (DEF-07-02/03 Family A) fail. Proven by a
source-built lib_lightgbm 4.6 FP execution trace (07-01/D-05 method) that reproduces the golden
gamma tree-0 bit-exact and exposes the per-bin g/h divergence on the identical (bit-matched
header) node."
fix: "NOT YET APPLIED — architectural (Rule 4). Recommended: correct the most_freq_bin==0 /
offset histogram model in crates/lgbm-treelearner (build_leaf_histogram_into / fix_histogram /
compact_histogram) so the most_freq_bin (bin-0 default) mass is carried faithfully to C++ for
non-constant hessians, then un-#[ignore] the DEF-07-02/03 Family-A cells and clear
.planning/phases/07-parity-completing-variants/deferred-items.md. Hard constraint: NO regression
of the bit-exact constant-hessian cells (kernel_parity 4/4, learner_parity spine_real_binary /
mfb_pos_real_binary, subset_determinism_diagnostic, all *_spine/*_gradients greens) and no
weakened tolerance. Family B (DEF-07-11) is OUT OF SCOPE."
verification: "Investigation verification: instrumented /tmp/LightGBM (separate clone, repo
LightGBM/ never touched) reproduces golden gamma tree-0 BIT-EXACT under the captured config;
the 8-row node + min_gain_shift + subtraction were each independently eliminated; the divergent
node localized by bit-matched header + per-row g/h. All diagnostic instrumentation reverted
(Rust files via git checkout — clean; C++ via git checkout in /tmp clone — clean; temp test
removed). cargo workspace state byte-identical to session start."
files_changed: "NONE (investigation-only; all instrumentation reverted). The fix will touch
crates/lgbm-treelearner/src/{learner.rs (build_leaf_histogram_into/compact_histogram),
fix_histogram.rs} + crates/oracle-harness/tests/boosting_parity.rs (un-ignore Family A) +
deferred-items.md."

## Execution update (07-13, 2026-06-08) — CORRECTED root cause + PAUSED (gate fail + multi-agent collision)

**The 07-13 executor empirically CORRECTED the root cause above.** The histogram-compaction/offset
theory was directionally right (learner-level split selection over faithful g/h) but WRONG on the
exact operand — the histograms are actually correct. The real defect:

- **`split_inner` seeded the smaller/larger leaf-splits from the SplitInfo `round_int(hess·cnt_factor)`
  counts (`best.left_count < best.right_count`), while the histogram-pool slot dance
  (`BeforeFindBestSplit`) + the tree node key off the DATA-PARTITION leaf counts.** C++ uses partition
  counts for BOTH (it overwrites `best_split_info.left/right_count` with partition counts at
  `serial_tree_learner.cpp:790-791`, `update_cnt=true`, before the `:851` tie-break; the histogram
  dance uses `GetGlobalDataCountInLeaf` = `data_partition_->leaf_count`). For fractional (non-constant)
  hessians the two sources disagree on the tie/±1 case (gamma node{0,1}|{2,3}: SplitInfo (1,3) vs
  partition (2,2)), attaching the wrong child's sums to the histogram and flipping the gain
  (1.0417 vs 0.0333). Constant-hessian families round the two sources identically → only Family A trips.
- **Fix applied (one line):** compare `part_left < part_right` (partition counts) in `split_inner`.
  Committed: `194abb3` (failing diagnostic) + `15263df` (fix). Confirmed gamma tree-0 1.0417→0.0333,
  leaf_count [2,4,2,4]; lgbm-treelearner 64/64, lgbm-compute, kernel_parity 4/4, learner_parity 25/25 GREEN.

**BLOCKED — Task 4 no-regression gate FAILS (fix is necessary but NOT sufficient):**
1. `goss_parity_matrix` REGRESSED (bit-exact real-lib_lightgbm golden, `goss_t200_o50_es0_bfa0` tree 11,
   abs 0.015). The OLD buggy SplitInfo-count tie-break was COINCIDENTALLY COMPENSATING a SECOND latent
   GOSS-specific defect: under GOSS gradient amplification Rust's SplitInfo `round_int(hess·cnt_factor)`
   counts are grossly INVERTED vs partition counts (e.g. partition (1,4) vs SplitInfo (4,1)). The correct
   partition-count tie-break un-masks it. Needs its own GOSS FP-trace investigation (the SplitInfo
   count/sum computation under amplification).
2. 3 Family-A cells still RED: `fair_loop_matrix` (tree 5, abs 2.085 — large), `quantile_loop_matrix`
   (tree 11, 0.0025), `quantile_alpha_axis` (tree 11, 0.0094).
3. Task 3 (un-ignore the 13 cells) NOT committed — parked in `git stash@{0}` ("pre-phase8: park
   DEF-07-02 fair un-ignore") by a CONCURRENT Phase-8 session.

**Multi-agent collision (why PAUSED, user-directed):** a concurrent Phase-8 session committed `c13d380`
+ `7a2fa3a` on top of `15263df`, stashed 07-13 Task 3 into `stash@{0}`, and left `crates/lgbm/src/booster.rs`
NON-COMPILING (`ModelError::Parse`/`MetricError` variant not found) — the full `cargo test --workspace`
no-regression gate cannot run until Phase-8 stabilizes its tree. 07-13's own commits + the
lgbm-treelearner/lgbm-compute crates build clean. NOTHING of Phase-8's (booster.rs/error.rs/STATE.md/stash)
was touched.

**RESUME CHECKLIST (after Phase-8 is coordinated + tree compiles):**
1. Confirm `cargo build --workspace` is green (Phase-8 booster.rs fixed).
2. Recover Task 3: `git stash show -p stash@{0}` → re-apply the 13 un-ignores (verify it's the full set,
   not partial) OR redo them from the plan's named list.
3. Root-cause + fix the GOSS SplitInfo count/sum inversion under amplification (second defect) so
   `goss_parity_matrix` stays bit-exact GREEN with the partition-count seeding.
4. Resolve the 3 remaining red Family-A cells (fair_loop_matrix tree-5 is the largest; quantile tree-11).
5. Re-run the full Task 4 no-regression gate; only on GREEN do Task 5 (clear DEF-07-02/03) + SUMMARY.
Retained assets: `/tmp/LightGBM` (reverted source build), `/tmp/gamma_fp/`. Commits on master: `194abb3`,
`15263df`."
