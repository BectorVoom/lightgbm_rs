# 260620-d3v — FINDINGS: full split-fusion campaign A/B on REAL workloads

**Campaign** = three shipped changes, all gated by two env-overridable thresholds:

- **a48** — smaller-child build+fix+scan fusion (unified BFS path).
- **b97** — larger-child subtract+scan fusion (no separate build).
- **c5v** — core-derived gate thresholds (`unified_bfs_threshold` ~100,
  `unified_subscan_threshold` ~130 at 16 cores).

## A/B toggle mechanism — single binary, env-only (no recompile, no worktree)

Every campaign change is gated on `features.len() >= threshold()`, where
`lgbm_compute::unified_bfs_threshold()` / `unified_subscan_threshold()` read
`LGBM_UNIFIED_BFS_THRESHOLD` / `LGBM_UNIFIED_SUBSCAN_THRESHOLD` and **the env var takes
ULTIMATE precedence** over the core-derived default
(`crates/lgbm-compute/src/lib.rs:430,473`). Each fusion's below-threshold fallback is the
byte-unchanged pre-campaign two-step code (proven bit-exact in a48/b97). Therefore the SAME
release binary measures both arms with zero harness skew — only two env values change:

- **A = campaign OFF (exact pre-campaign baseline):**
  `LGBM_UNIFIED_BFS_THRESHOLD=18446744073709551615 LGBM_UNIFIED_SUBSCAN_THRESHOLD=18446744073709551615`
  (`usize::MAX` → `features.len() >= MAX` always false → both fusions never fire → original
  two-step build→scan + subtract→scan).
- **B = campaign ON:** both vars UNSET → core-derived defaults (~100 / ~130 at 16 cores).

This is cleaner and MORE accurate than checking out the pre-a48 commit: same binary, same
harness, the only delta between A and B is the two env values.

## Environment

- Box: 16 logical cores (`nproc` = 16) → core-derived defaults reproduce 100 / 130 exactly.
- Build: `cargo build --release -p lgbm --example bench_real`.
- Harness: `crates/lgbm/examples/bench_real.rs`, "bench" mode — warm (discard run 0),
  3 timed reps, MEDIAN train-wall; `LGBM_PHASE_PROF=1` dumps BUILD_NS / SCAN_NS.
- Config (all workloads): `max_bin=255, num_leaves=31, num_iterations=100, learning_rate=0.1,
  deterministic=true`, via `RawCorpus::from_columns` → `train_raw` (bit-exact BinMapper path —
  real continuous floats, NOT the identity-bin DenseCorpus path).

## High-dim dataset provenance

- **Dataset used: MNIST-784** (`sklearn.datasets.fetch_openml('mnist_784', version=1)`),
  fetched successfully on this box (PRIMARY path — network reachable / cached).
- **Rows × features actually used: 15000 × 784** (subsampled from 70000 via a fixed-seed
  `np.random.default_rng(1).permutation`).
- **Label:** digit 0..9, fit with the **regression** objective (this is a SPEED bench, not an
  accuracy bench — the only requirement is real, non-uniform, BinMapper-binned features with
  ≥100 columns so the fusion gate fires).
- Fallback (NOT triggered here): `fetch_olivetti_faces()` (4096×400, bundled, no network).
- Exported to `target/bench_data/highdim.tsv` — **gitignored, NOT committed.**

## A/B results table (warm, 3-rep MEDIAN train-wall)

| Workload                | feat | A median ms | B median ms | delta %  | sign-stable? | BUILD_NS A→B (ms, 3 reps) | SCAN_NS A→B (ms, 3 reps) | bit-identical model? | bit-identical pred? |
| ----------------------- | ---- | ----------- | ----------- | -------- | ------------ | ------------------------- | ------------------------ | -------------------- | ------------------- |
| binary.train            | 28   | 224.682     | 229.681     | +2.2%    | NO (overlap) | 160.18 → 164.19           | 270.19 → 273.33          | YES                  | YES                 |
| regression.train        | 28   | 193.602     | 192.514     | −0.6%    | NO (overlap) | 160.40 → 158.58           | 263.11 → 263.20          | YES                  | YES                 |
| highdim MNIST-784       | 784  | 8379.573    | 2846.795    | **−66.0%** | **YES**    | **17888.9 → 0.0**         | 4496.6 → 5580.4          | YES                  | YES                 |

Notes:
- BUILD_NS / SCAN_NS are the phase_prof accumulators summed over all 3 timed reps
  (`[phase_prof:<label>]` lines); the per-rep median is BUILD/3, SCAN/3.
- "delta %" is `(B − A) / A` on the medians: negative = campaign ON is faster.

### Verbatim A/B median lines (from the run)

```
binary.train|28feat      A median_train_ms=224.682   B median_train_ms=229.681
regression.train|28feat  A median_train_ms=193.602   B median_train_ms=192.514
highdim.tsv|784feat      A median_train_ms=8379.573  B median_train_ms=2846.795
  A reps = [8314.056, 8379.573, 8448.645]   B reps = [2750.233, 2846.795, 3262.537]
```

Sign-stability of the high-dim win: every B rep (2750/2847/3263 ms) is below every A rep
(8314/8380/8449 ms) — no distribution overlap.

## Bit-identical correctness check (the end-to-end speed-only proof)

For each workload, `dump-model` and `dump-pred` were run under A (env MAX) and B (defaults)
and diffed. **All six diffs were EMPTY:**

```
binary.train(28f)     MODEL: IDENTICAL   PRED: IDENTICAL
regression.train(28f) MODEL: IDENTICAL   PRED: IDENTICAL
highdim-mnist(784f)   MODEL: IDENTICAL   PRED: IDENTICAL
```

Campaign-ON and campaign-OFF emit a **bit-identical model text AND prediction vector** on
every real dataset. The campaign is a **pure speed change** — output unchanged. (Both fusion
paths and their below-gate fallbacks are bit-exact to the C++ f64 anchor, so this is the
expected and required result; a non-empty diff would have been a campaign correctness failure
→ STOP. None occurred.)

## Honest framing: high-dim WIN, typical NEUTRAL-by-design

- **Typical real datasets (28 feat — binary.train, regression.train):** train-wall delta is
  within noise (+2.2% / −0.6%, both with run-overlapping distributions → NOT sign-stable).
  This is **neutral by design**: 28 < ~100/130, so the gate keeps fusion OFF and both arms run
  the byte-identical two-step path. The gate correctly protects typical workloads from the
  narrow a48/9cp build/scan-contention regression.

- **High-dim real dataset (784 feat — MNIST-784):** a large, **sign-stable −66% train-wall
  WIN** with campaign ON. The phase_prof breakdown localizes the mechanism precisely: BUILD_NS
  collapses from ~17.9 s (A: separate whole-buffer histogram build) to **0.0 ms** (B: build is
  fused into the per-feature unified BFS/subtract+scan region — a48 + b97 both fire above the
  gate at 784 feat). SCAN_NS rises modestly (4.5 s → 5.6 s) as the fused region absorbs the
  build work, but the net is a ~3× faster train.

- **Stronger than the synthetic A/B.** On synthetic identity-binned corpora the campaign
  showed single-digit-% wins at 120–200 feat. On REAL BinMapper-binned MNIST-784 the win is
  far larger (−66%) — at 784 features the eliminated build dominates the loop, so the
  whole-buffer build the fusion removes was a much bigger fraction of train time than the
  synthetic 120–200-feat probes suggested. The high-dim win materializes on real data and is
  in fact understated by the synthetic campaign benches. This is a valuable, truthful finding,
  not a manufactured one.

## Conclusion

The full split-fusion campaign (a48 + b97 + c5v) is a **pure speed change** (bit-identical
model + prediction on every real dataset) that is **neutral-by-design on typical (<100 feat)
workloads** and delivers a **sign-stable ~66% train-wall reduction on a high-dimensional
(784 feat) real dataset (MNIST-784, 15000×784)**, driven entirely by the env-gated
build→fused-scan collapse the phase_prof BUILD_NS=0 confirms.
