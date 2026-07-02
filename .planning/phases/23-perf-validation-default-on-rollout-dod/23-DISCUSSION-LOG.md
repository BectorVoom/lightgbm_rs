# Phase 23: Perf-Validation + Default-On Rollout (DoD) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-02
**Phase:** 23-perf-validation-default-on-rollout-dod
**Areas discussed:** Default-on flip semantics, 'Not-slower' pass/fail rule, Kaggle A/B harness scope, Rollout guardrails & fallback

---

## Default-on flip semantics

### Env semantics + device gate structure
| Option | Description | Selected |
|--------|-------------|----------|
| Tri-state + device AND-gate | Env tri-state (unset⇒default, "0"⇒off, "1"⇒on); CUDA-only default at routing seam as `device_type==cuda AND enabled`. One knob, honors off-switch. | ✓ |
| Split into two functions | Keep override fn pure, add separate `on_device_default_for(device_type)`. | |
| Bake default-on unconditionally | Hardcode on-device as CUDA path, no runtime device check. | |

**User's choice:** Tri-state + device AND-gate

### CUDA vs ROCm/HIP discrimination
| Option | Description | Selected |
|--------|-------------|----------|
| Runtime binding (cargo feature) | cubecl-cuda⇒default-on, cubecl-hip⇒default-off; no string sniffing. ROCm can't default-on. | ✓ |
| Config device_type string | Check `Config.device_type == "cuda"` at routing. | |
| Both (feature gate + assert) | Runtime-binding decides; assert device_type string agrees. | |

**User's choice:** Runtime binding (cargo feature)

---

## 'Not-slower' pass/fail rule

### Pass bar
| Option | Description | Selected |
|--------|-------------|----------|
| Within noise (≤5% slower OK) | Median wall-clock ≤5% above host-CUDA counts as not-slower. | ✓ |
| Strictly faster (median win) | Must be strictly faster to flip. | |
| Launch-collapse OR speed | Pass if EITHER device_launches drops OR wall-clock not-slower. | |

**User's choice:** Within noise (≤5% slower OK)

### Sign-stability & shape disagreement
| Option | Description | Selected |
|--------|-------------|----------|
| 3 runs median, both shapes must pass | 3 in-session repeats, median; flip only if BOTH shapes pass. | ✓ |
| 3 runs median, per-shape verdict | Flip only for passing shapes (needs shape-aware routing). | |
| 5 runs, both shapes must pass | 5 repeats for tighter sign-stability; both shapes gate. | |

**User's choice:** 3 runs median, both shapes must pass

---

## Kaggle A/B harness scope

### Wide shape
| Option | Description | Selected |
|--------|-------------|----------|
| 100k × 500 | 10× feature count, fits T4; stresses per-feature histogram fan-out. | ✓ |
| 50k × 1000 | Very wide, small rows; maximizes feature-parallel pressure. | |
| 1M × 500 | Wide + tall; heaviest, T4 OOM/quota risk. | |

**User's choice:** 100k × 500

### Harness packaging & artifact
| Option | Description | Selected |
|--------|-------------|----------|
| Committed script + results MD/JSON | Reusable in-repo harness emitting structured results file as DoD evidence. | ✓ |
| Kaggle notebook + pasted results | Notebook run, numbers pasted into a results MD. | |
| Script only, results in PR body | Commit harness; numbers in VERIFICATION/commit message. | |

**User's choice:** Committed script + results MD/JSON

---

## Rollout guardrails & fallback

### If A/B fails the bar
| Option | Description | Selected |
|--------|-------------|----------|
| Ship harness + evidence, keep default OFF | Harness always lands; default-ON flip is a separate verdict-gated commit; phase still DoD-complete. | ✓ |
| Block phase until it passes | Treat failing A/B as phase failure; iterate perf until not-slower. | |
| Flip anyway on launch-collapse | Flip if device_launches collapsed even if wall-clock lagged. | |

**User's choice:** Ship harness + evidence, keep default OFF

### Unsupported-feature fallback UX
| Option | Description | Selected |
|--------|-------------|----------|
| Silent fallback (Ok(None) as-is) | Keep quiet Ok(None)⇒host-path behavior; D-06 gate routes correctly. | ✓ |
| One-time Info log | Emit a single [Info] line noting the on-device skip + reason. | |
| Debug-level only | Log fallback reason at Debug level. | |

**User's choice:** Silent fallback (Ok(None) as-is)

### Parity evidence for the flip
| Option | Description | Selected |
|--------|-------------|----------|
| Merge gate + prior gates + Kaggle sanity | Rely on cpu-f64 merge gate, per-phase gates, and final-metric sanity vs official. | |
| Add a real-CUDA end-to-end parity assertion | Harness asserts on-device predictions match host-CUDA/official within ~1e-6 on real datasets, committed hard check. | ✓ |
| Merge gate only | Treat cpu-f64 merge gate as sufficient; Kaggle run perf-only. | |

**User's choice:** Add a real-CUDA end-to-end parity assertion

---

## Claude's Discretion

- `device_launches` capture mechanism on Kaggle (parsing the existing `phase_prof.rs:197` COUNTS line under `LGBM_PHASE_PROF`).
- Results-file schema beyond required metrics; Kaggle GPU-quota budgeting across the 3-run × 2-shape × 2-path matrix.

## Deferred Ideas

- Multi-stream overlap — stretch spike only if launch-collapse underdelivers on wall-clock.
- Shape-aware routing (per-shape default-on) — considered under sign-stability rule, rejected for this phase.
- Four GPU-profiling todos (score 0.6) reviewed but not folded — superseded by the completed spike campaign (001–054) and prior phases.
