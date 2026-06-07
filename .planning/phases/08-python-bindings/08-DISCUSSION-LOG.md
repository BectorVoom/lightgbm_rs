# Phase 8: Python Bindings - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-07
**Phase:** 8-Python Bindings
**Areas discussed:** Data input fidelity, Parameter interface, API surface + persistence, Packaging & naming

---

## Data input fidelity

User clarified the input intent mid-area: **"numpy + polars"** — both first-class raw-data inputs.

### Q1 — Bin model
| Option | Description | Selected |
|--------|-------------|----------|
| Yes, bin internally | numpy/polars hold raw values → BinMapper → FeatureColumns → train; pre-binned no longer the Python contract | ✓ |
| Yes + keep pre-binned escape | Raw default, also expose pre-binned Dataset for harness/advanced | |
| Let me explain | Free-text | |

### Q2 — Polars ingest path
| Option | Description | Selected |
|--------|-------------|----------|
| Zero-copy via Arrow | Consume polars Arrow columns directly in Rust (pyo3-polars/arrow FFI) | ✓ |
| Convert to numpy first | df.to_numpy() → numpy path | |
| You decide (research it) | Defer mechanism | |

### Q3 — Dtype routing
| Option | Description | Selected |
|--------|-------------|----------|
| Yes, auto-route by dtype | Categorical/Enum/string → categorical features; numeric → numeric | ✓ |
| Numeric only for v1 | Defer categorical-from-DataFrame | |
| Explicit categorical_feature arg | User names categorical columns | |

### Q4 — Wiring home
| Option | Description | Selected |
|--------|-------------|----------|
| In the Rust facade | raw→bin→train added to `lgbm` crate; Python thin wrapper | ✓ |
| In the Python layer only | Facade stays identity-binned; Python assembles FeatureColumns | |
| You decide (planner) | Capture requirement, planner chooses | |

### Q5 — Sparse input
| Option | Description | Selected |
|--------|-------------|----------|
| Yes, scipy CSR/CSC in v1 | Route through existing bit-exact CSR/CSC ingest | ✓ |
| Dense v1, sparse later | Defer scipy sparse | |
| You decide | Planner weighs cost | |

**User's choice:** Bin internally; polars zero-copy via Arrow; dtype auto-routing; wiring in the Rust facade; scipy CSR/CSC in v1.
**Notes:** The "wiring gap" (facade `train()` is identity-binned today; `BinMapper`/`Dataset::construct` exist but aren't wired in) becomes in-scope Phase-8 work — framed as integration of already-validated binning, not new binning.

---

## Parameter interface

(Grounded on the discovery that `Config::from_params` already ports C++ `Config::Set` with the full alias table + unknown-warn semantics.)

### Q1 — Primary config surface
| Option | Description | Selected |
|--------|-------------|----------|
| params dict (mirror official) | dict → HashMap → Config::from_params; builder stays Rust-only | ✓ |
| params dict + kwargs sugar | dict canonical + kwargs fold in | |
| Typed/keyword only | No dict — diverges from official idiom | |

### Q2 — Unsupported params
| Option | Description | Selected |
|--------|-------------|----------|
| Error on unsupported | Explicit recognized-but-unimplemented set raises; typos still warn | ✓ |
| Warn only (full C++ fidelity) | Mirror C++ exactly, everything just warns | |
| You decide | Planner picks | |

### Q3 — Value coercion
| Option | Description | Selected |
|--------|-------------|----------|
| Full coercion layer | bool/int/float + list params joined per C++ convention | ✓ |
| Scalars now, lists case-by-case | Scalars robust, lists incremental | |
| You decide | Planner scopes | |

**User's choice:** params dict (mirror official); error on recognized-but-unimplemented params; full coercion layer.
**Notes:** Silent divergence (e.g. `device_type=gpu` no-op) is the risk that justifies erroring on known-but-unported params.

---

## API surface + persistence

### Q1 — In-scope pieces beyond locked core (multi-select)
| Option | Description | Selected |
|--------|-------------|----------|
| Training callbacks (list) | early_stopping/log_evaluation/record_evaluation/reset_parameter | ✓ |
| lgb.cv cross-validation | k-fold helper, pure-Python over train() | ✓ |
| Feature importance + plotting | feature_importance() + plot_importance/tree/metric | ✓ |
| Dask / distributed wrapper | lightgbm.dask — initially selected, then resolved (see follow-up) | ✓ → deferred |

### Q1b — Dask vs distributed boundary (follow-up after flagging the conflict)
| Option | Description | Selected |
|--------|-------------|----------|
| Defer Dask (recommended) | Drop from Phase 8; track blocked on distributed engine | ✓ |
| Single-node Dask shim | API-shaped but not true distributed; document gap | |
| Pull distributed into scope | Reopen v1 boundary — multi-phase effort | |

### Q2 — Persistence
| Option | Description | Selected |
|--------|-------------|----------|
| Full text I/O + pickle | C++-compatible save_model/model_to_string/load + pickle | ✓ |
| Text I/O only (no pickle) | C++-compatible text only | |
| You decide | Planner scopes | |

**User's choice:** Callbacks + cv + importance/plotting IN; Dask deferred; persistence = C++-compatible text I/O + pickle.
**Notes:** Claude flagged that a faithful `lightgbm.dask` is unbuildable without the v1-deferred distributed/allreduce engine; user agreed to defer.

---

## Packaging & naming

### Q1 — Import name
| Option | Description | Selected |
|--------|-------------|----------|
| Distinct (e.g. lightgbm_rs) | No collision; coexists with real lightgbm for A/B parity | ✓ |
| Drop-in 'lightgbm' | True drop-in but can't coexist with official | |
| Let me specify | Free text | |

### Q2 — Layout
| Option | Description | Selected |
|--------|-------------|----------|
| New workspace crate | crates/lgbm-python (PyO3 cdylib) + maturin + thin python/ pkg | ✓ |
| Separate bindings/ dir | bindings/python outside crates/ | |
| You decide | Planner chooses | |

### Q3 — Wheels
| Option | Description | Selected |
|--------|-------------|----------|
| abi3, broad range | Single stable-ABI wheel/platform, CPython 3.8/3.9+ | ✓ |
| Per-version wheels | Separate wheel per minor version | |
| You decide | CI scopes matrix | |

**User's choice:** Distinct name `lightgbm_rs` (PyPI `lightgbm-rs`); new workspace crate `crates/lgbm-python`; single abi3 broad-range wheel.
**Notes:** Distinct name chosen specifically to enable side-by-side install with the real `lightgbm` for parity testing.

---

## Claude's Discretion

- Exact `categorical_feature` override API shape.
- Precise CPython version floor + wheel/CI matrix details (within the abi3 broad-range decision).
- Error/exception taxonomy mapping (`LgbmError` → Python exceptions), custom-callback grad/hess marshalling, sklearn wrapper semantic depth — bounded implementation details left for research/planning; user chose to stop the discussion here.

## Deferred Ideas

- **Dask / distributed wrapper** — blocked on the v1-deferred distributed/network (allreduce) engine.
- **File/Arrow-file/binary-cache ingestion (ING-01/02/03)** — already v2-deferred at the Rust level; the Python file-load path inherits it.
