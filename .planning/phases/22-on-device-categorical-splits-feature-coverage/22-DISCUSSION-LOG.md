# Phase 22: On-Device Categorical Splits (Feature Coverage) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-02
**Phase:** 22-on-device-categorical-splits-feature-coverage
**Areas discussed:** Parity anchor, Eval path scope, max_cat overflow, Quantized combo

---

## Parity anchor

| Option | Description | Selected |
|--------|-------------|----------|
| Both: real goldens + structure gate | Pin bitset/decision-type/num_cat bit-exact to real 4.6 cat_onehot + cat_manyvsmany goldens AND run the on-device tree through the cubecl-cpu f64 structure gate. | ✓ |
| Structure gate only | Reuse Phase-21 structure gate against cubecl-cpu f64 host path only; real goldens sit unused. | |
| Real goldens only | Pin only to the real 4.6 split goldens; skip the end-to-end device structure gate. | |

**User's choice:** Both: real goldens + structure gate
**Notes:** Categorical is the first on-device subsystem with a REAL reference anchor (numerical spine goldens are host re-transcriptions). Goldens prove reference fidelity; structure gate proves the full device grow loop routes categorical rows correctly. → CONTEXT D-01.

---

## Eval path scope

| Option | Description | Selected |
|--------|-------------|----------|
| Both paths this phase | Port one-hot AND many-vs-many (bitonic-sorted); both real goldens exist and ODL-22/SC#2 name both. Reuses Phase-14 bitonic argsort primitive. | ✓ |
| One-hot first, defer many-vs-many | Ship one-hot only; many-vs-many as a follow-up needing a roadmap re-cut. | |

**User's choice:** Both paths this phase → CONTEXT D-02.

---

## max_cat overflow

| Option | Description | Selected |
|--------|-------------|----------|
| Size slab from config at init | Allocate slab width from config.max_cat_threshold once at driver init (default 32); allocate-once preserved, no truncation, faithful to any config. | ✓ |
| Clamp to 32 + host-fallback over | Keep const-32 slab; route max_cat_threshold>32 to host. | |
| Hard clamp to 32 | Silently cap at 32 — diverges from reference, breaks parity. Not recommended. | |

**User's choice:** Size slab from config at init → CONTEXT D-03. MAX_CAT_PER_SPLIT becomes the default, not a hard cap.

---

## Quantized combo

| Option | Description | Selected |
|--------|-------------|----------|
| Host-fallback, logged | on_device_growth_supported() returns false for categorical + use_quantized_grad; mirrors the reference's asm("trap;") non-support + Phase-19 fallback precedent. | ✓ |
| Non-quantized categorical on-device, quantized part host | Bespoke path the reference lacks; breaks parity. Not recommended. | |
| Defer / out of scope | Leave the combo unhandled; risky (silent wrong results). | |

**User's choice:** Host-fallback, logged → CONTEXT D-06.

---

## Claude's Discretion
- Kernel geometry / thread-block mapping (§6.3 / §8.1 launch idioms; cubecl-0.10 gotcha checklist).
- Bitset-construction atomic mechanics (CubeCL-safe equivalent of `atomicAdd_system(out+val/32, 1<<(val%32))`).
- Parity fixture parameters (row/feature/category counts, num_leaves, max_cat_threshold/max_cat_to_onehot).
- Single-block vs global-memory bitonic sort for the many-vs-many path.

## Deferred Ideas
- Categorical + use_quantized_grad on-device path (host-fallback this phase; reference traps).
- Low-VRAM global-memory categorical eval (`_GlobalMemory` variant) — optional.
- Categorical perf-validation (Kaggle A/B, device_launches) → Phase 23 DoD.
