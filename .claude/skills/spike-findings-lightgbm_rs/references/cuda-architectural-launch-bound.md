# Real-discrete-CUDA optimisation — the architectural launch-bound arc (spikes 051–054)

This is the **optimisation** chapter that follows the **attribution** chapter
(`cuda-discrete-gpu-bottleneck.md`, spikes 046/048/049/050). After the metric-eval fix
(−26%) and parallel binning (6.5×), 049 left the **GPU histogram phases (53%)** as the
dominant remaining real-NVIDIA cost. Spikes 051–054 attacked it and **closed the
cheap-win search**: every cheap lever (occupancy, launch-fusion, sync-reduction) is
**refuted on real NVIDIA**. The ~5–6× narrow-shape gap is **architectural** — a
host-driven, per-leaf growth loop issuing ~8,570 small serial kernel launches vs
official's `CUDASingleGPUTreeLearner` (whole tree on-device).

## Requirements (honored)

- The CPU f64 anchor stays **bit-exact to C++**; these are all measurement spikes using
  pre-existing env toggles (no anchor change, no parity risk). Gate any future kernel work
  with `cargo test -p oracle-harness -p lgbm-treelearner -p lgbm`.
- "GPU is faster" is only claimed where the data supports it — and here, **it never beats
  official CUDA** in the measured range.
- Measure on **real hardware** before planning — the spoofed APU mis-predicted *every*
  lever in this arc (occupancy mattered on the APU, doesn't on CUDA; fusion was "flat" on
  the APU, "catastrophic" on CUDA).

## How to run it (the reusable zero-code probe harness)

The cheapest real-CUDA spike pushes **no code** — it sweeps existing env toggles on the
current master and reads `phase_prof`. Driver template: `sources/051-*/spike051_kaggle.py`
(inner bench inlined as a string ⇒ no git push; one wheel build, N env-arm subprocesses).

1. **Dedicated kernel per spike**: `boomvector/lgb-rs-cuda-spike0NN`,
   `kernel_type=script`, `enable_gpu/internet=true`. Push with `kaggle kernels push -p <dir>`;
   poll `kaggle kernels status …` to `COMPLETE`; `kaggle kernels output … -p <out>`.
2. **One backend / one arm per subprocess** under `LGBM_PHASE_PROF=1` (the `phase_prof`
   atomics are process-global — a fresh process per arm keeps stderr attributable).
3. **The env toggles that probe the GPU build/scan/partition** (all on current master):
   - `LGBM_AUTOTUNE_FORCE_P=k` — pin the build row-partition P (unclamped; bypasses `ROWPART_P_MAX=16`).
   - `LGBM_AUTOTUNE=0` — force the `row_partition_count` heuristic (P=1 at 50 feat).
   - `LGBM_FUSED_FORCE=1` — force the `build_fix_scan` directly-built-child fusion (default-off).
   - `LGBM_SIBLING_COPACK=0` — disable the default-ON sibling scan co-pack.
4. **PARSE RULE (cost me a misread in 051):** each `fit` emits TWO `[phase_prof:train]`
   dumps — a **warmup** (`device_launches≈445`, absorbs cold CUDA-context + kernel-JIT,
   several seconds) then the **timed** 100-tree dump (`device_launches=8570+`). Select the
   **max-launches** record. And read the **absolute-ms** line (`before=… hist+split=4897ms`),
   NOT the `%: … hist+split=73.8` percentage line — a naive `hist\+split=([\d.]+)` regex
   matches both; key off the line *starting with* `before=`.
5. **Absolute walls are NOT cross-session comparable** (Kaggle assigns T4/P100/T4×2). Trust
   **in-session A/B deltas**.
6. **Official CUDA reference** (for the ratio, spike-054): `pip install --no-binary lightgbm
   lightgbm -C cmake.define.USE_CUDA=ON` (source build, several min; probe-import first).

## What's true on real NVIDIA (the findings — don't re-spike these)

| Lever | Verdict | Evidence (500k×50, 100 trees) |
|---|---|---|
| **Build occupancy / row-partition P** | **REFUTED (053)** | `FORCE_P {1..128}` flat-to-worse; **P=1 optimal** (P=1 9862ms ≤ P=16 10546ms). No PSET-ceiling headroom. APU's ~10% P-sensitivity (040) does NOT transfer. |
| **`build_fix_scan` launch fusion** | **REFUTED hard (052)** | `LGBM_FUSED_FORCE=1` = **5.4× WORSE** (58.3s vs 10.7s; ~571ms/tree vs 95ms). The fused kernel is **f64** → tanks on consumer-NVIDIA (1/32 f64 rate). Keep default-off. |
| **Sync reduction** | **no headroom (052)** | Readback syncs cost **~0.14ms each** (`copack=0` doubles syncs 2890→5680 for only +3.6%). Sibling co-pack (default-on) IS load-bearing — keep it. |
| **`LGBM_AUTOTUNE=0`** | **~4% WIN (051)** | The plain P=1 heuristic beats default-autotune on the narrow CUDA shape (autotune's APU value is negative here). The one cheap cuda win. |
| **Shape / feature width** | **mitigates, doesn't fix (054)** | lgb_rs/official ratio HALVES with width (3.90×@50f → 1.93×@500f) — launches CONSTANT (8900), ms/launch rises 0.93→2.38 ⇒ launch-bound fraction shrinks. But lgb_rs CUDA **never beats official** (asymptotes ~2×). |

**The mechanism (triangulated 3 ways):** the wall is **8,570 small SERIAL kernel launches**
gated by the best-first **build→subtract→scan** per-node dependency chain. `build=0` (async
issue), occupancy-insensitive (051), sync-cheap (052), and width-amortizing (054) all point
to the same thing — it is **launch-latency-bound, not compute-throughput- nor sync-bound**.

## How to build it (the one real lever)

**The on-device multi-leaf tree learner** — mirror official's `CUDASingleGPUTreeLearner`:
keep the whole growth frontier on-device and drive the build→split→partition loop with far
fewer, bigger kernels instead of ~86 host-orchestrated launches/tree. This is the gap at
**every** feature width (most acute when narrow). It is **milestone-sized, high-uncertainty**
— scope it as a phase, not a spike. Everything in 001–054 that tuned individual kernels was
necessary but is **not sufficient**: the launch-orchestration architecture is the ceiling.

## What to avoid

- **Don't tune build occupancy / lift `BUILD_PSET` for CUDA** — refuted (051/053).
- **Don't force `build_fix_scan` fusion on CUDA** — 5.4× regression (052).
- **Don't chase sync reduction** — syncs are ~0.14ms; co-pack already banks the win (052).
- **Don't write f64 hot loops in new CUDA kernels** — consumer NVIDIA f64 is 1/32 f32;
  prefer the **u64 fixed-point** build path (spike-018) that makes the separate path fast.
- **Don't trust APU lever-signs for discrete CUDA** — occupancy and fusion both flipped.

## Constraints

- Kaggle CLI auth = ACCESS_TOKEN at `/home/user/.kaggle` (no kaggle.json), user `boomvector`.
- A code change must be pushed to GitHub `master` before the kernel clones it — but the whole
  051–054 arc needed **zero pushes** (existing toggles). Output download pulls the cloned tree
  (slow); `rm -rf` the `lightgbm_rs/` subdir after, keep `*.log`.

## Origin

Synthesized from spikes 051 (occupancy refuted), 052 (fusion refuted + sync-cheap), 053
(refuted by 051, no code), 054 (width crossover). Source READMEs + the `spike0NN_kaggle.py`
zero-code probe drivers + `051/phase_prof_dumps.txt` in `sources/051-*/ … 054-*/`. Companion:
`cuda-discrete-gpu-bottleneck.md` (the 046–050 attribution chapter this follows).
