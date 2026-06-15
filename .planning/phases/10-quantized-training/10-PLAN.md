---
phase: 10-quantized-training
type: feature (standalone, multi-wave)
status: in-progress
started: 2026-06-15
branch: feat/10-quantized-training
source_artifacts:
  - .planning/spikes/008-16bit-discretized-hist/README.md   # WHY this is opt-in/approximate
  - LightGBM-release-4.6.0.99/src/treelearner/gradient_discretizer.{hpp,cpp}  # CPU parity target
  - LightGBM-release-4.6.0.99/src/treelearner/cuda/cuda_histogram_constructor.cu  # packed-int kernel
goal: >
  Add LightGBM's opt-in `use_quantized_grad` APPROXIMATE training mode (gradients/hessians
  quantized to int, integer histogram accumulation, de-quant for split-finding). NOT held to
  the exact ~1e-6 contract — held to parity with C++ `use_quantized_grad=true` under its own
  approximate contract. The exact (default) path is UNTOUCHED.
---

# Phase 10: Quantized (`use_quantized_grad`) training mode

## Why this exists / contract change

Spike-008 proved the 16-bit discretized path is irreducibly APPROXIMATE (even full int16 drifts
~3e-4). LightGBM exposes it as **opt-in `use_quantized_grad` (default false)**. So this phase adds
a SEPARATE mode with its OWN parity contract:

- **Default (exact) path: byte-for-byte UNCHANGED.** All existing gates stay green. Quantized code
  is reached only when `use_quantized_grad=true`.
- **Quantized contract:** parity to **C++ `use_quantized_grad=true, stochastic_rounding=false`**
  (deterministic rounding → fully reproducible → bit-exact-tractable). Stochastic rounding +
  `quant_train_renew_leaf` are deferred (RNG-match is a separate problem).
- **The win:** integer histogram = packed (grad,hess) → ONE int atomic/row instead of two f32
  (halves atomic count — the one lever that targets our atomic-bound bottleneck; measured in Wave 5).

## C++ reference (faithful targets)

- Scale (`gradient_discretizer.cpp:107-114`): `grad_scale = max|grad| / (num_grad_quant_bins/2)`,
  `hess_scale = is_constant_hess ? max|hess| : max|hess| / num_grad_quant_bins`; inverses are f64.
- Quantize deterministic (`:145-158`): `int8 = (g>=0 ? g*inv_scale + 0.5 : g*inv_scale - 0.5)`,
  hess `= h*inv_hess_scale + 0.5` (or constant `1`). Storage: pairs `[hess_i, grad_i]` (even=hess).
- Dynamic width (`SetNumBitsInHistogramBin`): int16 hist if `num_data_in_leaf * num_grad_quant_bins`
  fits int16, else int32. Avoids int16 overflow on large leaves.
- De-quant: bin sums (int) × scale → f64 grad/hess sums for split gain.

## Wave status (2026-06-15)

- **W1 ✅** GradientDiscretizer core (53cc4a1) — 5 tests
- **W2 ✅** integer histogram + de-quant (9af04f7) — 3 tests
- **W3a ✅** config knobs + e2e split pipeline (0328f6a) — recovers exact split @ <5% gain
- **W4 ✅ (oracle set up)** C++ quantized goldens + harness (112dcc2); comparison `#[ignore]` pending W3b
- **W3b ⬜ NEXT** production GBDT-loop/backend wiring — activates the W4 parity gate
- **W5 ⬜** GPU packed-int kernel + speed · **W6 ⬜** stochastic rounding + renew_leaf
- Discovery: `num_grad_quant_bins ≤ 254` (i8 storage; 256 overflows → constant model)

## Waves

### Wave 1 — GradientDiscretizer core (CPU, deterministic) ✅ DONE
`crates/lgbm-treelearner/src/gradient_discretizer.rs`. Pure numeric core:
- `grad_scale`/`hess_scale` from max-abs (constant-hessian branch).
- Deterministic quantize f32 grad/hess → i8 pairs `[hess, grad]` (sign-aware ±0.5, the verbatim
  C++ cast-truncation toward zero).
- De-quant helpers (int sum × scale → f64).
- Unit tests: hand-computed quantization on small vectors; round-trip; constant-hessian; the exact
  C++ truncation semantics (`static_cast<int8_t>` truncates toward zero, NOT round).
- **No GPU, no integration yet.** Stochastic rounding stubbed (deterministic only).
- Acceptance: `cargo test -p lgbm-treelearner gradient_discretizer` green; exact path untouched.

### Wave 2 — Integer histogram accumulation (CPU)
Accumulate i8 grad/hess → i32 (or i16) per-bin sums; dynamic bit-width selection; de-quant to the
f64 `[g0,h0,...]` the split-finder consumes. Validate vs an f64 reference of the SAME quantized
inputs (self-consistent), and confirm the de-quant matches C++'s `int_sum * scale`.

### Wave 3 — Split-finding + GBDT integration + config
Config surface (`use_quantized_grad`, `num_grad_quant_bins`, `stochastic_rounding=false`,
`quant_train_renew_leaf=false`). Route GBDT to call the discretizer per-iter and the quantized
histogram path when enabled. Split gain uses de-quantized sums. Exact path branch-guarded.

### Wave 4 — C++ `use_quantized_grad` parity oracle
Build lib_lightgbm CPU with `use_quantized_grad=true, stochastic_rounding=false`; generate goldens
on a committed corpus; Rust quantized train matches (bit-exact target for deterministic rounding;
document any residual). This is the merge gate for the mode.

### Wave 5 — GPU packed-int discretized kernel + speed
Packed-int32 LDS histogram (grad16<<16 | hess16), one int atomic/row, row-partitioned (reuse
phase-09 P). Held to ~1e-6 vs the CPU quantized anchor. **Measure the speed win** vs f32 two-atomic.

### Wave 6 — (deferred) stochastic rounding + renew_leaf
RNG-matched stochastic rounding (`gradient_random_values_` + start offset) and
`RenewIntGradTreeOutput`. Separate, harder parity; out of the initial mode.

## Guardrails (every wave)
- The exact default path stays byte-identical — quantized code behind `use_quantized_grad`.
- Quantized parity is to C++ quantized goldens, NOT the f64 exact anchor (spike-008).
- No `unsafe`; integer overflow handled by the dynamic bit-width (Wave 2).

## Artifacts this phase produces
`gradient_discretizer.rs` (GradientDiscretizer: scales, quantize, de-quant), the int-histogram path,
config knobs, the C++ quantized oracle + goldens, the packed-int GPU kernel.
