---
phase: 23
slug: perf-validation-default-on-rollout-dod
status: verified
threats_open: 0
asvs_level: 1
created: 2026-07-03
---

# Phase 23 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

Phase 23 is an infrastructure/DoD phase (tri-state on-device env resolver, on-device launch-count profiling counter, credential-free Kaggle A/B harness, verdict-gated rollout decision). The real-CUDA A/B run returned **FAIL**, so the behavior-changing default-on flip was intentionally WITHHELD — `on_device_default()` remains the literal `false` stub and no crate code changed in plan 04. On-device stays opt-in via `LGBM_CUDA_ON_DEVICE=1`.

Register origin: `register_authored_at_plan_time: true` — all four PLAN files carried parseable `<threat_model>` blocks. The auditor verified mitigations exist (did not scan for new threats).

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| process env → routing code path | `LGBM_CUDA_ON_DEVICE` (operator input) selects the tree-learner path | tri-state string `"1"`/`"0"`/unset |
| env → profiling code path | `LGBM_PHASE_PROF` gates whether launch counters bump/emit | on/off flag; no untrusted data into logic |
| Kaggle CI ↔ public repo | harness clones + builds `BectorVoom/lightgbm_rs` from source on Kaggle | source code (pinned repo) |
| harness ↔ Kaggle credentials | Kaggle CLI auth (`boomvector`) is out-of-band; must not enter the committed script | credentials (must NOT cross) |
| compiled cargo feature → routing default | on-device default decided at compile time by the cubecl binding | build-time `cfg!(feature="cuda")` |
| evidence artifact → rollout decision | `results.md` verdict gates whether the behavior-changing flip is made | aggregate timings/launch/parity numbers |

---

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-23-01 | Tampering | `cuda_on_device_override()` env parse | mitigate | Exact-string closed-enum `match` (`"1"`=>Some(true), `"0"`=>Some(false), `_`=>None); no eval/exec/path/format interpretation (ASVS V5). Evidence: `crates/lgbm-compute/src/lib.rs:1345-1351` | closed |
| T-23-01-SC | Tampering (safety invariant) | `on_device_default()` default resolution | mitigate | Returns literal `false` (pre-verdict); cpu/rocm/cuda byte-unchanged. SC-4 anchor `!cuda_on_device_enabled()` at `lib.rs:3639`; body at `lib.rs:1378-1380` | closed |
| T-23-02 | Tampering (safety invariant) | on-device launch counter | mitigate | `bump_launch()` no-op unless `launch_prof_enabled()` (exact `v=="1"` OnceLock); only mutates the `AtomicU64`, never tree structure/values. Evidence: `crates/lgbm-compute/src/kernels/grow_driver.rs:78-91` | closed |
| T-23-02-I | Information disclosure | COUNTS stderr line | accept | Emits only aggregate launch/timing integers; no dataset content or secrets. Gated behind `LGBM_PHASE_PROF=1`. Evidence: `crates/lgbm-treelearner/src/phase_prof.rs:226` | closed |
| T-23-03 | Information disclosure | committed `ab_harness.py` | mitigate | No Kaggle token/credential literal (grep-clean); auth stays out-of-band, `_setup_kaggle_build` does not authenticate. Evidence: `ab_harness.py:190` | closed |
| T-23-03-SC | Tampering / supply chain | Kaggle clone + build | mitigate | Pinned `git clone BectorVoom/lightgbm_rs`; `maturin build --release -F cuda` + official lightgbm `--no-binary` USE_CUDA=ON — all from source, no third-party binary wheel. Evidence: `ab_harness.py:202,206,218` | closed |
| T-23-03-I | Information disclosure | `results.{md,json}` artifact | mitigate | Aggregate only (verdict, shapes, tolerances, wall-hours; launch/parity `null` this run); no dataset content or tokens (grep-clean) | closed |
| T-23-04-SC | Tampering (safety invariant) | `on_device_default()` flip | mitigate | Flip NOT performed (A/B FAIL) → `on_device_default()` stays `false`; no plan-04 crate change → cpu/rocm byte-unchanged; SC-4 trivially upheld. Evidence: `lib.rs:1378-1380` + `results.md` "Decision: NO FLIP" | closed |
| T-23-04-V | Repudiation / process | the rollout decision | mitigate | Blocking human-verify checkpoint required a PASS verdict; FAIL left the flip uncommitted (D-09); evidence committed. Evidence: `results.md` FAIL + `results.json` `flip_performed:false`; `evidence/kaggle-ab-run.log` | closed |
| T-23-04-D10 | Denial of service (correctness) | unsupported-config fallback | accept | `on_device_eligible_gate` routes categorical+quantized to host silently via `Ok(None)` (D-10, unchanged); correctness-preserving demotion. Evidence: `crates/lgbm-treelearner/src/learner.rs:468-473` | closed |

*Status: open · closed*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-23-01 | T-23-02-I | Launch/timing COUNTS line is gated behind `LGBM_PHASE_PROF=1` and emits only aggregate integer launch/sync counts to stderr — no dataset rows, labels, feature values, or credentials. Diagnostic-only, off by default. | gsd-security-auditor | 2026-07-03 |
| AR-23-02 | T-23-04-D10 | Unsupported (categorical + quantized) configs fall back to the host path silently via `Ok(None)` rather than erroring — a correctness-preserving demotion (D-10), unchanged by this phase, logged once via `cat_quant_fallback_logged`. | gsd-security-auditor | 2026-07-03 |

*Accepted risks do not resurface in future audit runs.*

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-07-03 | 10 | 10 | 0 | gsd-security-auditor |

Corroborating structural checks:
- Duplicate env parse removed — `grep 'env::var("LGBM_CUDA_ON_DEVICE")'` in `crates/lgbm-treelearner/src/learner.rs` returns none; the learner reads the tri-state only through `backend.on_device_growth_supported()` → `cuda_on_device_enabled()` (single parse point, `lib.rs:1360-1363`).
- Counter fold path is crate-cycle-safe — `on_device_launch_count_take()` (`grow_driver.rs:96-98`) is consumed by `phase_prof::dump`; the harness regex `device_launches=(?P<launches>\d+)` is preserved (`phase_prof.rs:223-226`).
- No unregistered flags — every threat maps to a PLAN `<threat_model>` entry; no SUMMARY `## Threat Flags` and no unmapped attack surface.

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-07-03
