---
spike: 054
name: cuda-shape-crossover
type: standard
validates: "Given 051/052's finding that the narrow-shape lgb_rs-CUDA gap is architectural (8570 small serial launches, launch-bound), when feature count is swept {50,200,500} at 500k rows on real NVIDIA, then the lgb_rs/official ratio drops toward 1 with width (route wide to CUDA) OR stays flat (architectural gap dominates everywhere)"
verdict: VALIDATED
related: [051, 052, 048, 001]
tags: [gpu, cuda, kaggle, crossover, routing, wide-shape, launch-bound]
---

# Spike 054: CUDA shape crossover — where does lgb_rs CUDA approach official?

## What This Validates

051 + 052 closed the cheap-win search: the narrow 50-feature lgb_rs-CUDA gap (~5–6× vs
official) is **architectural** — 8570 small serial kernel launches gated by the best-first
build→subtract→scan dependency chain. Launch **count** scales with leaves (≈constant across
feature width); per-launch **work** scales with feature count. So at **wide** shapes each
build kernel is well-fed and the fixed per-launch overhead amortizes ⇒ the launch-bound
*fraction* should shrink and lgb_rs CUDA should get **relatively** closer to official. This
spike measures that crossover to produce **routing guidance** + confirm the 051/052 diagnosis.

## Research

Reuses the proven 046/048 harness. Sweeps `n_features ∈ {50, 200, 500}` at 500k rows,
100 trees, `num_leaves=31`, `n_informative = feats/2` (so width = more real splits, not noise).
Per shape: lgb_rs CUDA wall + `phase_prof` (phases / launches / ms-per-launch) vs **official
LightGBM CUDA** (built from source with `USE_CUDA=ON` for the reference ratio). In-session
ratios (absolute Kaggle walls drift across sessions). Kernel `boomvector/lgb-rs-cuda-spike054`.

| Read-out | Conclusion |
|---|---|
| ratio(rs/official) DROPS toward 1 as feats rise | narrow gap is the architectural long-pole; **route wide → CUDA** |
| ms/launch RISES with feats, launches ≈ constant | launch-bound *fraction* shrinks at width — confirms the 051/052 launch-bound diagnosis |
| ratio FLAT/rising | architectural gap dominates at every width; GPU not competitive vs official regardless of shape |

## How to Run

```bash
kaggle kernels push  -p kaggle_push_054
kaggle kernels status boomvector/lgb-rs-cuda-spike054
kaggle kernels output boomvector/lgb-rs-cuda-spike054 -p kaggle_out_054
```

## Investigation Trail

- 051/052 established the architectural launch-bound diagnosis; 054 bounds the practical
  routing crossover across feature width.
- Official LightGBM CUDA built from source on Kaggle (the reference ratio) — succeeded this run.

## Results

**VERDICT: VALIDATED.** Real CUDA, 500k rows, 100 trees, num_leaves=31:

| feats | rs_wall | official_wall | ratio | t1iter | phases | launches | ms/launch |
|---|---|---|---|---|---|---|---|
| 50  | 13.103 | 3.363  | **3.90×** | 11786 | 8262  | 8900 | 0.928 |
| 200 | 27.580 | 11.293 | **2.44×** | 22597 | 12100 | 8900 | 1.360 |
| 500 | 58.128 | 30.061 | **1.93×** | 46045 | 21180 | 8900 | 2.380 |

### Finding 1 — the gap HALVES with width (3.90× → 1.93×) ⇒ confirms launch-bound
As features widen the lgb_rs/official ratio drops monotonically (3.90 → 2.44 → 1.93). The
narrow-shape penalty is the **fixed per-launch overhead** of the 8900 host-driven launches;
at 500 feat each launch does ~2.6× more work so that overhead amortizes. This is the direct
confirmation of the 051/052 **launch-bound** diagnosis on a third axis.

### Finding 2 — launches CONSTANT (8900), ms/launch RISES (0.93→2.38) ⇒ the mechanism
`device_launches` is **identical (8900) at every feature width** — it scales with the tree
shape (leaves/nodes), NOT feature count. Per-launch device work (`phases`/launches) rises
0.93→2.38ms. So the launch-bound *fraction* of the wall shrinks with width — the mechanism
behind Finding 1, and a clean independent corroboration of 051/052.

### Finding 3 — lgb_rs CUDA NEVER beats official (asymptotes ~2×) ⇒ architectural is universal
Even at 500 features lgb_rs CUDA is **1.93× official** and the trend is flattening toward a
~1.5–2× floor — the residual is the architectural on-device-learner advantage (official keeps
the whole growth loop on-device; lgb_rs round-trips per leaf). **There is no shape in this
range where lgb_rs CUDA is competitive with official CUDA.** Width *mitigates* the gap but
does not close it.

### Signal for the build (routing + the universal lever)
- **The on-device multi-leaf tree learner is the universal lever** — it's the gap at every
  width, just most acute when narrow. Width only amortizes the launch overhead on top of it.
- **Practical routing today:** lgb_rs CUDA is least-bad at wide shapes (~1.9× vs ~3.9× narrow);
  if the GPU path must be used, prefer it for wide data. (lgb_rs-CUDA-vs-lgb_rs-CPU routing is a
  separate, environment-dependent question — 048: CUDA beats lgb_rs CPU on Kaggle's few-vCPU box,
  CPU wins on the 16-core dev box.)
- Evidence: `kaggle-run.log`. Official CUDA built from source this session (USE_CUDA=ON).
