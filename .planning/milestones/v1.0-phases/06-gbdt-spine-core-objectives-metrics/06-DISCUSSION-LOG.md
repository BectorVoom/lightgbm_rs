# Phase 6: GBDT Spine + Core Objectives/Metrics - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-07
**Phase:** 6-gbdt-spine-core-objectives-metrics
**Areas discussed:** Rust-native API shape, Oracle corpus matrix, Validation granularity, Spine-first sequencing

---

## Rust-native API shape (API-01, OBJ-02)

### Top-level training entry point
| Option | Description | Selected |
|--------|-------------|----------|
| Free `train()` fn (mirror lgb.train) | Closely mirrors Python `lgb.train()` / C++ `GBDT`; most parity-faithful | |
| Builder pattern | `Booster::builder()…build()` — idiomatic Rust | ✓ |
| Both | Free fn + thin builder layer | |

### Params supply
| Option | Description | Selected |
|--------|-------------|----------|
| Reuse lgbm-core::Config | Pass the existing Config bag directly | |
| New training-params builder | A separate typed training-params struct/builder | ✓ |

### Custom objective grad/hess
| Option | Description | Selected |
|--------|-------------|----------|
| Closure mirroring Python fobj | `Fn(preds, &Dataset) -> (grad, hess)` | ✓ |
| Trait object (impl ObjectiveFn) | User implements a `CustomObjective` trait | |

### Eval history + early-stopping output
| Option | Description | Selected |
|--------|-------------|----------|
| Booster fields (mirror Python) | `best_iteration` + eval history on the Booster | ✓ |
| Returned EvalResults struct | `train()` returns `(Booster, EvalResults)` | |
| You decide | Leave to planner | |

### Follow-up: params builder ↔ Config relationship
| Option | Description | Selected |
|--------|-------------|----------|
| Builder produces a Config internally | Builder is a front-end; Config stays single source of truth | ✓ |
| Builder replaces Config at the API boundary | New struct is the real surface; Config internal/legacy | |

### Follow-up: param surface scope
| Option | Description | Selected |
|--------|-------------|----------|
| Curated subset + raw escape hatch | Common params + `.set_raw`/`.from_config` | |
| Full surface | A builder method per in-scope param | ✓ |
| You decide | Leave to planner | |

**User's choice:** Idiomatic-Rust builder on the outside; the new training-params builder resolves to `lgbm-core::Config` internally (Config = single source of truth); full param surface + escape hatch; Python-mirroring `fobj` closure + Booster eval fields.
**Notes:** Deliberate split — idiomatic ergonomics on the public surface, Python-compatible semantics where Phase-8 bindings need a 1:1 map, faithful C++ mirror below the API boundary. Config never forked.

---

## Oracle corpus matrix

### Objective coverage
| Option | Description | Selected |
|--------|-------------|----------|
| All 5 core + one custom | End-to-end goldens for all core objectives + a custom run vs Python fobj | ✓ |
| All 5 core, custom unit-tested only | Core goldens; custom narrower pass-through test | |
| You decide | Leave to researcher | |

### Config axes
| Option | Description | Selected |
|--------|-------------|----------|
| Spine golden + targeted single-axis goldens | One canonical per objective + isolated bagging/early-stop/bfa goldens | |
| Full cross-product | Every objective × bagging × early-stop × bfa combination | ✓ |
| Spine only this phase | Per-objective goldens; bagging/early-stop via unit tests | |

### Datasets
| Option | Description | Selected |
|--------|-------------|----------|
| Small per-objective synthetic corpora | New deterministic datasets with objective-appropriate labels | ✓ |
| Reuse Phase-2/5 corpora where possible | Reuse existing fixtures + minimal new | |
| You decide | Leave to researcher | |

### Boosting depth
| Option | Description | Selected |
|--------|-------------|----------|
| Modest multi-iter (~10–20 iters) | Exercises accumulation, shrinkage, bfa, early-stop trigger | ✓ |
| Single/few iterations | Minimal | |
| You decide | Leave to researcher | |

**User's choice:** All 5 core objectives + one custom; full cross-product of bagging × early-stop × bfa per objective (~40 cells); small per-objective synthetic corpora; ~10–20 iterations.
**Notes:** Deliberately exhaustive matrix, consistent with the keystone-fidelity ethos. Researcher may collapse only provably byte-identical cells, with documented equivalence.

---

## Validation granularity (max-diagnostic)

### Grad/hess depth
| Option | Description | Selected |
|--------|-------------|----------|
| Per-row snapshot, iter-1 + a later iter | Localizes objective-math divergence per row, early + later | ✓ |
| Per-row snapshot, iter-1 only | Catches formula, not score-dependent drift | |
| End-to-end only | Rely on final parity | |

### Per-iteration score snapshot
| Option | Description | Selected |
|--------|-------------|----------|
| Yes — per-iteration score snapshot | Localizes loop/shrinkage/bfa divergence to the iteration | ✓ |
| No — final scores only | Less wiring | |

### Per-round metric values
| Option | Description | Selected |
|--------|-------------|----------|
| Yes — per-round metric values | Validates each metric + early-stopping input | ✓ |
| No — final metric only | Lighter | |

### Bagging RNG parity
| Option | Description | Selected |
|--------|-------------|----------|
| Commit selected-index sequence per iter | Dedicated golden asserting RNG draw sequence + call order | ✓ |
| Implicit via end-to-end parity | Trust matching models | |
| You decide | Leave to researcher | |

**User's choice:** Per-row grad/hess (iter-1 + a later iter); per-iteration accumulated score snapshot; per-round metric values; dedicated committed bagged-index sequence asserting RNG parity.
**Notes:** Maximal diagnostic resolution across every axis — no validation blind spots in the first end-to-end run.

---

## Spine-first sequencing

### Minimal spine pick
| Option | Description | Selected |
|--------|-------------|----------|
| regression(L2) + l2/rmse | Simplest single-output objective + natural metrics | ✓ |
| binary + binary_logloss | Closer to a real classifier, more objective math first | |
| You decide | Leave to planner | |

### boost_from_average in spine?
| Option | Description | Selected |
|--------|-------------|----------|
| Yes — it's the faithful default | C++ regression default; load-bearing BoostFromScore path | ✓ |
| No — add it as a parity addition | Prove rawest loop first | |

### Multiclass per-class trees timing
| Option | Description | Selected |
|--------|-------------|----------|
| After single-output spine is proven | Add per-class trees as a later structural addition | ✓ |
| Part of the initial spine | Build per-class loop from the start | |

### Addition order
| Option | Description | Selected |
|--------|-------------|----------|
| Objectives → multiclass → bagging → early-stop | Widen breadth, then per-class, then RNG, then stopping | ✓ |
| Loop features → objectives | Bagging + early-stop on regression first | |
| You decide | Leave to planner | |

**User's choice:** Minimal spine = regression(L2) + l2/rmse, including boost_from_average; multiclass per-class trees after the single-output spine; addition order objectives → multiclass → bagging → early-stop.
**Notes:** Vertical-slice spine-first, one axis at a time; spine matches the C++ default config (bfa on).

---

## Claude's Discretion

- Crate placement for the boosting layer (new `lgbm-boosting` + umbrella `lgbm` facade vs folding in) and boosting↔learner wiring.
- Objective/metric crate-vs-module placement and internal trait shape (bounded by C++ factory semantics).
- Exact ownership/borrow shape of the custom-objective closure (bounded by the Python fobj contract).
- Golden serialization/layering format for grad/hess, scores, metrics, bagged-index, model/predict fixtures.
- AUC tie-handling / sort determinism, `metric_freq`/`first_metric_only` cadence, per-class score memory layout (bounded by "match C++").
- Captured-g/h capture path config + which iteration is "a later iteration" for the iter-2 grad/hess snapshot.

## Deferred Ideas

- GOSS / DART / Random Forest variants — Phase 7 (BST-04/05/06).
- Categorical / EFB splits (TRL-06) — Phase 7.
- Remaining objectives (huber/fair/poisson/quantile/mape/gamma/tweedie, cross-entropy, ranking) — Phase 7.
- Extended + ranking metrics, per-query eval — Phase 7.
- SHAP/predict_contrib, prediction early stopping, monotone/interaction constraints, forced splits/bins, extra-trees, CEGB, refit, feature importance — Phase 7.
- Python/PyO3 bindings — Phase 8.
- Parallel (rayon) CPU / multi-GPU boosting path — post-MVP optimization.
- ROCm cross-check of the full train→predict loop — research/planning call (CPU bit-exact is the hard gate).
