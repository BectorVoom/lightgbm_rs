# Requirements: LightGBM-rs — Milestone v1.1 (GPU Training-Speed: CUDA On-Device Tree Learner)

**Defined:** 2026-06-28
**Core Value:** For identical inputs/config, reproduce C++ LightGBM within ~1e-6 on every backend (f32 end-to-end); the cubecl-cpu f64 fold is the bit-exact merge gate. v1.1 adds an on-device CUDA tree learner that closes the architectural GPU training-speed gap WITHOUT weakening that contract.

## Milestone v1.1 Requirements

Scoped by spikes 051–054 (real-NVIDIA Kaggle): the cheap GPU-histogram levers (occupancy/fusion/sync) are refuted; the on-device learner — mirroring official `CUDASingleGPUTreeLearner` — is the one architectural lever. Each requirement is anchor-gated and additive (CPU/ROCm/existing-host-CUDA paths stay byte-unchanged).

### Foundation (scaffold + oracle — Slice 0)

- [ ] **ODL-01**: An additive `Backend::grow_tree_on_device` method + default-false `on_device_growth_supported()` discriminator routes GPU tree growth on-device, leaving CPU, ROCm, and the existing host-CUDA path byte-unchanged. Off by default behind `LGBM_CUDA_ON_DEVICE`.
- [ ] **ODL-02**: An anchor-pinned oracle asserts the on-device tree's STRUCTURE is bit-exact to the cpu f64 anchor (tie-aware `default_left`), with leaf values within a ~1e-5 f32 envelope — never comparing two nondeterministic GPU paths to each other.

### On-Device Growth Loop (the core port)

- [ ] **ODL-03**: The on-device learner grows a continuous-feature tree end-to-end on real CUDA (resident frontier; reuses the shipped u64-fixed-point build / feature-per-lane scan / sibling co-pack kernels + host `Tree::split` replay), anchor-pinned to the host learner's tree (Slice 1, the thinnest proving slice).
- [ ] **ODL-04**: Best-split selection across the full leaf frontier runs on-device (cross-leaf reduce), eliminating the per-leaf scan readbacks (Slice 2; tie-aware assert lands here).
- [ ] **ODL-05**: Data partition + leaf-index update runs on-device (the Split kernel), with a single small scalar readback per split (Slice 3).
- [ ] **ODL-06**: The on-device histogram pool implements the subtraction trick via `hist_t**` pointer rotation (larger child inherits the parent buffer; smaller child gets a fresh slot), no bulk histogram copy.

### Kernel Constraint (the f64 trap guard)

- [ ] **ODL-07**: New CUDA kernels keep f32 + the u64 fixed-point build with NO f64 per-row hot loops (the measured 5.4× consumer-NVIDIA f64 regression, spike-052), staying within the ~1e-6 anchor envelope. f64 is permitted only in scalar/storage gain math where the reference uses it.

### Feature Coverage (pulled into v1.1)

- [ ] **ODL-08**: On-device growth handles categorical splits faithfully (anchor-pinned), via a CubeCL-compatible pre-allocated threshold representation (NOT the reference's per-`SplitInfo` device alloc, which has no clean CubeCL analog).
- [ ] **ODL-09**: On-device growth supports bagging / GOSS row subsampling, anchor-pinned to the host bagging RNG draw sequence.
- [ ] **ODL-10**: Score/prediction update runs on-device (replacing the host `add_prediction_to_score` scatter), within the ~1e-6 anchor envelope.

### Performance & Rollout (the DoD)

- [ ] **ODL-11**: A real-CUDA Kaggle A/B harness measures the on-device path's `device_launches` (target: well below the 8,570/100-trees baseline) and the lgb_rs/official wall-clock ratio at 500k×50 and a wide shape.
- [ ] **ODL-12**: The on-device learner becomes the DEFAULT CUDA tree-learner path — contingent on anchor-pinned parity (~1e-6) AND not-slower than the current host-CUDA path on the Kaggle A/B — with the host path retained as an off-switch fallback (`LGBM_CUDA_ON_DEVICE=0`).

## v2 Requirements

Deferred to a future milestone. Tracked but not in this roadmap.

### On-Device Quantized Training

- **ODL-13**: On-device integration of the opt-in quantized-grad training mode (Phase 10 `use_quantized_grad`) with the on-device learner.

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Device-side megakernel / in-kernel grow loop | cubecl 0.10 has no global grid barrier; the reference `CUDASingleGPUTreeLearner` is itself host-driven (kernel boundaries are the barrier). The win is fewer/larger launches over resident state, not a megakernel. |
| Multi-GPU on-device learning | Single-GPU parity first; distributed is a separate milestone. |
| Changing CPU or ROCm routing | v1.1 is additive and CUDA-targeted; the CPU f64 anchor and ROCm host-partition paths stay byte-unchanged. |
| Per-`SplitInfo` device `cudaMalloc` for categorical thresholds | No clean CubeCL analog; ODL-08 uses pre-allocation instead. |
| On-device quantized training | Deferred to v2 (ODL-13). |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| ODL-01 | TBD | Pending |
| ODL-02 | TBD | Pending |
| ODL-03 | TBD | Pending |
| ODL-04 | TBD | Pending |
| ODL-05 | TBD | Pending |
| ODL-06 | TBD | Pending |
| ODL-07 | TBD | Pending |
| ODL-08 | TBD | Pending |
| ODL-09 | TBD | Pending |
| ODL-10 | TBD | Pending |
| ODL-11 | TBD | Pending |
| ODL-12 | TBD | Pending |

**Coverage:**
- v1.1 requirements: 12 total
- Mapped to phases: 0 (roadmapper fills)
- Unmapped: 12 ⚠️ (until roadmap)

---
*Requirements defined: 2026-06-28 (milestone v1.1)*
*Last updated: 2026-06-28 after initial definition*
