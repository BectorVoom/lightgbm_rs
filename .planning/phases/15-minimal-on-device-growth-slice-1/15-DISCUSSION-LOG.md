# Phase 15: Minimal On-Device Growth (Slice 1) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-29
**Phase:** 15-minimal-on-device-growth-slice-1
**Areas discussed:** Build shape, Kernel reuse, Slice boundary, Eligibility gate, Exit gate

> A read-only investigation of the official LightGBM CUDA single-GPU tree learner
> (`LightGBM/src/treelearner/cuda/*`) preceded the decisions and reshaped the questions:
> the mainline is fully on-device (selection + partition + tree on device, 2 tiny scalar
> readbacks/node, ~13–15 launches/node), so the Slice-1 "device boundary" is *determined*
> by the roadmap slicing (host argmax/partition/tree) rather than open, and the histogram
> rotation is largely existing machinery (the resident pool already `Move`s the parent
> buffer to the larger child). The genuinely-open questions were re-posed accordingly.

---

## Build shape

| Option | Description | Selected |
|--------|-------------|----------|
| Thin orchestrator over shipped kernels | Drive the existing build_resident_leaf → subtract_resident → scan_resident_siblings behind the seam; host argmax + Tree::split + host partition. Max reuse, lowest risk. | |
| Dedicated on-device growth path | Purpose-built device-resident growth driver/loop, not constrained by the per-leaf resident API; more freedom to collapse the per-node launch sequence. | ✓ |

**User's choice:** Dedicated on-device growth path
**Notes:** Chosen over the thin orchestrator despite higher risk — the shipped per-leaf
resident API is shaped for the host-driven per-leaf chain; a dedicated driver can collapse
the per-node launch sequence (the spike-051/052/054 architectural long-pole).

---

## Kernel reuse

| Option | Description | Selected |
|--------|-------------|----------|
| New driver, reuse compute kernels | New orchestration/loop, but call the SHIPPED u64-build / feature-per-lane scan / subtract kernels as primitives. Honors SC#1 literally. | |
| New driver AND new compute kernels | Freedom to write net-new compute kernels (e.g. fused build+subtract+scan) if they collapse launches better. Larger ODL-07 no-f64 audit surface. | ✓ |

**User's choice:** New driver AND new compute kernels
**Notes:** Deliberately relaxes SC#1's literal "reusing the shipped kernels" wording.
Load-bearing constraint retained: any new build kernel keeps u64 fixed-point with NO f64
hot loops (ODL-07; spike-052 = f64 mega-kernel 5.4× worse on consumer NVIDIA). The
new-kernel freedom widens the ODL-07 no-f64 audit surface.

---

## Slice boundary

| Option | Description | Selected |
|--------|-------------|----------|
| Keep host boundary (ODL-03/06/07 only) | Cross-leaf argmax + Tree::split + partition stay host; one best-split packet readback per touched leaf. Phase scope unchanged; 16/17 remain separate. | ✓ |
| Pull on-device argmax forward (absorb ODL-04) | Move cross-leaf argmax on-device now; also requires activating the tie-aware assert. Bigger first slice. | |
| Pull argmax + partition forward (absorb ODL-04/05) | Near-full mainline mirror in one slice. Highest risk; collapses three slices into one. | |

**User's choice:** Keep host boundary (ODL-03/06/07 only)
**Notes:** Preserves the thin-slice de-risking. The reference investigation confirmed one
~120 B `CUDASplitInfo`-equivalent packet per touched leaf fully determines host Tree::split
replay + host partition (mainline rebuilds the whole host tree from an 8-int + 16-int
packet/node).

---

## Eligibility gate

| Option | Description | Selected |
|--------|-------------|----------|
| Silent fall-through to host | Outside the supported envelope → quiet Ok(None) → byte-unchanged host path. Mirrors resident_eligible fail-safe. | |
| Fall-through + debug log | Silent fall-through plus an LGBM_*-gated debug line naming the reason. | |
| Hard-assert when forced | With LGBM_CUDA_ON_DEVICE=1 AND unsupported shape → error loudly (NotSupported), do not silently fall through. | ✓ |

**User's choice:** Hard-assert when forced
**Notes:** Catch "thought it was on-device but wasn't" during dev — the toggle is a
developer/bench affordance this slice. Diverges intentionally from the resident_eligible
silent-fallthrough idiom. Merge gate stays green because the default (toggle unset) keeps
on_device_eligible=false on every backend; the assert only fires under explicit opt-in.

---

## Exit gate

| Option | Description | Selected |
|--------|-------------|----------|
| Correctness + local launch-count | Anchor-pinned correctness + byte-unchanged paths + a local launches/tree instrument (count faithful on the APU). Kaggle is a non-blocking confirmation. | ✓ |
| Correctness only this slice | Defer ALL launch measurement (local + Kaggle) to Phase 19. Ships no evidence the architecture helps. | |
| Kaggle device_launches blocking | Not done until a real-NVIDIA Kaggle run confirms the drop. Couples the merge gate to a manual paid harness for a genuinely-open win. | |

**User's choice:** Correctness + local launch-count
**Notes:** The win magnitude is genuinely open (Slice 1's host boundary adds traffic
mainline avoids; best-first still serializes), so a paid external manual harness must not
gate the merge. Launch COUNT is faithful on the spoofed 8-CU APU even though timing is
confounded; the Kaggle device_launches number runs as confirmation, not a gate.

---

## Claude's Discretion

- The `hist_t**` rotation mechanism under cubecl-0.10 (Handle in-place aliasing vs ping-pong
  double-buffering) + batched `client.read(vec![h])` readback semantics on cubecl-cuda — a
  planning verification spike (ROADMAP research flag); the shipped resident pool already
  rotates, so the planner verifies the dedicated path meets ODL-06's "no bulk copy".
- The additive parameters to extend `grow_tree_on_device` with (binned store / per-feature
  metadata / resident handles).
- Naming of the dedicated driver, the launch-count instrument env var, and the new kernels.

## Deferred Ideas

- On-device cross-leaf argmax + tie-aware assert activation → Phase 16 (ODL-04).
- On-device data partition / leaf-index update → Phase 17 (ODL-05).
- Categorical / bagging / GOSS / on-device score update → Phase 18.
- `num_leaves > 8` / production-depth on-device growth → Phases 16+.
- Kaggle `device_launches` A/B + default-on rollout → Phase 19 (ODL-11/12).
