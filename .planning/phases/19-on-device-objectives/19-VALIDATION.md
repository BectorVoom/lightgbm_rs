---
phase: 19
slug: on-device-objectives
status: approved
nyquist_compliant: true
wave_0_complete: false
created: 2026-07-01
---

# Phase 19 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Lifted from `19-RESEARCH.md` → `## Validation Architecture`. Anchor discipline (D-05):
> every numeric output is pinned to the **cubecl-cpu f64 fold** (never GPU-vs-GPU), with a
> real compiled-`lib_lightgbm` golden cross-check for one representative per family.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust `cargo test` (workspace); new parity tests in `crates/oracle-harness/tests/` |
| **Config file** | none — standard `cargo test`; ROCm cross-check gated by `--features rocm` |
| **Quick run command** | `cargo test -p oracle-harness --test objective_parity_<family>` (per-family test binary: `_regression`/`_binary`/`_multiclass`/`_rank`; shared harness in `objective_parity.rs`) |
| **Full suite command** | `cargo test --workspace` (cpu hard merge gate, `LGBM_CUDA_ON_DEVICE` unset) |
| **Estimated runtime** | quick ~30 s per family · full suite ~several min |

> **Test-file layout (reconciled to the plans):** the plans split parity into per-family
> test binaries (`objective_parity_<family>.rs`) sharing a common harness (`objective_parity.rs`,
> created in plan 19-00) so Wave-2 family plans own **disjoint** files. The original single-file
> `objective_parity <family>` filter form below is superseded by the per-binary form.

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p oracle-harness --test objective_parity_<family>` (the family under edit).
- **After every plan wave:** Run `cargo test -p oracle-harness` (all objective + existing parity).
- **Before `/gsd-verify-work`:** `cargo test --workspace` must be green with `LGBM_CUDA_ON_DEVICE` unset (ODL-19 / D-06); optional `cargo test -p oracle-harness --features rocm` for the ~1e-6 hip cross-check.
- **Max feedback latency:** ~30 seconds (per-family quick run).

---

## Anchor & Tolerance Policy (D-05)

| Output class | Anchor | Assertion |
|--------------|--------|-----------|
| Elementwise grad/hess (L2, L1, Quantile, Huber, Fair, Poisson, softmax, binary) | cpu f64 fold + real `lib_lightgbm` golden | **bit-exact** `compare_exact_u32` on f32 bits |
| BoostFromScore mean (L2/Huber/Fair) | cpu f64 fold | bit-exact if serial fold; `compare_within` if atomic reduce |
| BoostFromScore **binary logit** (`atomicAdd` sums) | cpu f64 fold | `compare_within(ORACLE_TOL=1e-6)` — atomic-order residual |
| BoostFromScore median (L1/Quantile via percentile) | cpu f64 fold | bit-exact (deterministic sort) |
| **Lambdarank** λ / hess (`atomicAdd_block`) | cpu f64 fold + `lambdarank_gh` golden | `compare_within(ORACLE_TOL)` / tie-aware |
| RankXENDCG grad/hess (softmax + RNG) | cpu f64 fold | `compare_within(ORACLE_TOL)` on transcendental cells; RNG stream **bit-exact** |
| ConvertOutput (sigmoid/exp/sign·x²/softmax) | host `ObjectiveKind::convert_output` | bit-exact where no transcendental; `compare_within` on exp/log |
| RenewTreeOutput (per-leaf median/quantile) | cpu f64 fold + `regression_l1` renewed-leaf golden | bit-exact (deterministic percentile) |
| Per-item ranking RNG stream | `draw_next_float_on` reference | **bit-exact** `compare_exact_u32` |

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 19-00-01 | 00 | 1 | ODL-05..08 | — | N/A | scaffold | `cargo test -p oracle-harness --test objective_parity` (harness compiles) | ❌ W0 | ⬜ pending |
| 19-00-03 | 00 | 1 | ODL-08 | — | N/A | fixture | `test -f crates/oracle-harness/tests/fixtures/rank/lambdarank_gh_iter1.txt` | ❌ W0 | ⬜ pending |
| 19-01-xx | 01 | 2 | ODL-05 | — | N/A | parity | `cargo test -p oracle-harness --test objective_parity_regression` | ❌ W0; goldens ✅ | ⬜ pending |
| 19-01-xx | 01 | 2 | ODL-05 | — | N/A | parity | `... objective_parity_regression boost_from_score` | ❌ W0 | ⬜ pending |
| 19-01-xx | 01 | 2 | ODL-05 | — | N/A | parity | `... objective_parity_regression renew_leaf` | ❌ W0; `regression_l1` renewed ✅ | ⬜ pending |
| 19-01-xx | 01 | 2 | ODL-05 | — | N/A | parity | `... objective_parity_regression convert` | ❌ W0 | ⬜ pending |
| 19-02-xx | 02 | 2 | ODL-06 | — | N/A | parity | `... objective_parity_binary` | ❌ W0; `binary_gh_*` ✅ | ⬜ pending |
| 19-02-xx | 02 | 2 | ODL-06 | — | N/A | parity | `... objective_parity_binary boost` | ❌ W0 | ⬜ pending |
| 19-03-xx | 03 | 2 | ODL-07 | — | N/A | parity | `... objective_parity_multiclass` | ❌ W0; `multiclass_gh_*` ✅ | ⬜ pending |
| 19-03-xx | 03 | 2 | ODL-07 | — | N/A | parity | `... objective_parity_multiclass ova` | ❌ W0; `multiclassova_gh_*` ✅ | ⬜ pending |
| 19-04-xx | 04 | 2 | ODL-08 | — | N/A | parity | `... objective_parity_rank lambdarank` | ❌ W0; golden from 19-00 T3 | ⬜ pending |
| 19-04-xx | 04 | 2 | ODL-08 | — | N/A | parity | `... objective_parity_rank xendcg` | ❌ W0; `rank_xendcg_objseed5` ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky. Task IDs are indicative — the planner assigns final IDs; the family→command mapping is the load-bearing contract. Foundation (harness + lambdarank golden) is plan 19-00 (wave 1); families are 19-01..04 (wave 2).*

---

## Property-Based / Held-Out Backstop

- **Determinism property:** run each device grad/hess kernel twice on the cpu backend → bit-identical (guards against accidental nondeterministic reduction in a supposedly-deterministic kernel).
- **Weight-branch equivalence:** `use_weight=true` with all-1.0 weights == `use_weight=false` (bit-exact) — catches comptime-branch divergence.
- **Class-major invariant:** multiclass grad summed over classes per row ≈ 0 for softmax (Σ(p−δ) = 0 up to f32) — a cheap held-out sanity net independent of the golden.
- **RNG replay:** `rank_xendcg_objseed5` seed-replay (existing rank_parity precedent) — the per-item draw stream must bit-match across the host anchor and device draw.

---

## Wave 0 Requirements

- [ ] `crates/oracle-harness/tests/objective_parity.rs` — shared harness (`parse_gh_golden` / `assert_gradients` / `compare_exact_u32` / `compare_within(ORACLE_TOL=1e-6)` reused from `boosting_parity.rs`), created in plan 19-00; consumed by the per-family binaries `objective_parity_{regression,binary,multiclass,rank}.rs` (plans 19-01..04, disjoint ownership).
- [ ] `crates/oracle-harness/tests/fixtures/rank/lambdarank_gh_iter{1,N}.txt` — **capture** the one missing golden (extend `xtask/py/rank_oracle_capture.py`, score-derivation route; fall back to custom-`fobj` interception if within-query λ math doesn't derive cleanly — Open Question A1).
- [ ] Confirm the uv `.venv` has `lightgbm==4.6` before the capture task (Open Question A4).
- [ ] `crates/lgbm-compute/src/kernels/objective_*.rs` + `mod.rs` exports (greenfield — no objective module exists today).

*Goldens for L2 / binary / multiclass / multiclassova / regression_l1 / poisson / huber / fair / quantile already exist in `tests/fixtures/boosting/` — no capture needed; only lambdarank is missing.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| ROCm ~1e-6 cross-check on the spoofed APU | ODL-05..08 | GPU hardware backend, APU-confounded (perf numbers only; parity still valid) | `cargo test -p oracle-harness --features rocm` on the local ROCm device; assert within-tol vs cpu f64 anchor, never GPU-vs-GPU |

*All numeric parity behaviors have automated cpu-anchor verification; the ROCm layer is a separate best-effort ~1e-6 cross-check.*

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (objective_parity.rs harness + lambdarank golden — plan 19-00)
- [x] No watch-mode flags
- [x] Feedback latency < 30s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-07-01 (plan-checker Dimension 8 PASS; reconciled to per-family test-file layout)
