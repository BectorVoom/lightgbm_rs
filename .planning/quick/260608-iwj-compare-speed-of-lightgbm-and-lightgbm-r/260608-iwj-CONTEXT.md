# Quick Task 260608-iwj: Compare speed of lightgbm vs lightgbm_rs and optimise code — Context

**Gathered:** 2026-06-08
**Status:** Ready for planning

<domain>
## Task Boundary

Measure the training/prediction speed of the pure-Rust `lightgbm_rs` engine, then
apply optimization techniques from `/home/user/Documents/workspace/optimisor/manual`
**without changing numerics** — the bit-exact CPU f64-fold parity gate and the
~1e-6 ROCm gate are non-negotiable (CLAUDE.md core value).

</domain>

<decisions>
## Implementation Decisions

### Scope
- **Bench + parity-safe wins.** Build a benchmark harness, capture a before/after
  speed report, then apply ONLY optimizations that cannot change numeric output.
  The oracle-harness parity suite is the gate after every code change.

### Optimization levers (user selected all four)
- **Release profile** — add `[profile.release]` (lto=fat, codegen-units=1,
  opt-level=3). Zero numeric risk; typically the single biggest win.
- **Global allocator** — `#[global_allocator]` = mimalloc, wired into the bench
  harness and the `lgbm-python` cdylib (the shipped artifact). Parity-neutral.
- **smallvec hot paths** — replace small short-lived `Vec`s in the serial tree
  learner per-node/per-split loops. Parity-neutral (same contents/order).
- **Zero-copy ingest** — bytemuck is ALREADY used at the compute boundary
  (`as_bytes`/`from_bytes`/`create_from_slice`), so this lever becomes
  buffer-reuse / redundant-allocation elimination in the kernels + learner gather.

### Baseline measurement
- **Rust before/after** is the core instrument (per-lever deltas across commits).
- Supplementary: one cheap `lightgbm==4.6` pip-wheel timing point (the pinned
  reference already in the project `.venv`) on the same synthetic data, to honor
  the literal "compare lightgbm vs lightgbm_rs" ask. Best-effort, secondary.

### Half-precision (EXPLICITLY OUT OF SCOPE)
- The manual's `HALF_PRECISION_CUBECL.md` (f16/bf16) is OUT: ~1e-3 precision
  shatters both the bit-exact CPU gate and the ~1e-6 ROCm gate. Not applied.

### Claude's Discretion
- Exact synthetic dataset sizes, smallvec inline capacities, and which specific
  buffer-reuse sites to touch (guided by the parity gate — revert any change that
  breaks a bit-exact test).

</decisions>

<specifics>
## Specific Ideas

- `DenseCorpus` requires identity-binned features (distinct values == `0..K-1`),
  so the synthetic generator forces full bin coverage per feature.
- Per-lever measurement runs: M0 baseline → M1 +profile → M2 +mimalloc →
  M3 +smallvec/buffer-reuse. mimalloc gated behind an `lgbm` cargo feature so M0/M1
  use the system allocator and M2/M3 use mimalloc.

</specifics>

<canonical_refs>
## Canonical References

- `/home/user/Documents/workspace/optimisor/manual/*.md` — optimization technique manuals
- `CLAUDE.md` — core value: bit-exact CPU / ~1e-6 ROCm parity (the hard gate)
- Known pre-existing failure (NOT a regression): `goss_parity_matrix` (DEF-08-OOS-01)

</canonical_refs>
