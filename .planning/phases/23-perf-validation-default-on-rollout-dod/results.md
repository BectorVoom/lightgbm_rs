# Phase 23 On-Device A/B Results — verdict: **FAIL**

> **Audit-before-wire outcome (D-09):** the on-device path did NOT clear the D-04
> not-slower bar on real discrete NVIDIA CUDA. The ODL-21 default-on flip (plan 23-04
> Tasks 1 & 2) is therefore **intentionally NOT performed** — the CUDA default stays
> **OFF**, opt-in only via `LGBM_CUDA_ON_DEVICE=1`. The phase is closed DoD-complete
> with these documented numbers + a follow-up note.

## Verdict

**FAIL** — emitted literally as `>>> A/B VERDICT: FAIL <<<` in the run log.

## Run provenance

| Field | Value |
|-------|-------|
| Kernel | `yensen2/lgb-rs-phase23-ab` |
| Kernel URL | https://www.kaggle.com/code/yensen2/lgb-rs-phase23-ab |
| Hardware | Kaggle real **discrete NVIDIA CUDA** GPU (not the local spoofed 8-CU APU) |
| Repo | `BectorVoom/lightgbm_rs` (pinned clone, T-23-03-SC) |
| lgb_rs build | `maturin build --release -F cuda` (from source) |
| official lightgbm | rebuilt from source `--no-binary` with `USE_CUDA=ON` (documented fallback: the pip binary lacked CUDA) |
| Kernel terminal status | ERROR — **BY DESIGN**: the harness prints the verdict then `sys.exit(1)` on FAIL, and Kaggle maps any non-zero exit to ERROR. This is NOT a build/crash failure. |

### Build + matrix timeline (log seconds)

| Milestone | ~t (s) | Note |
|-----------|--------|------|
| `maturin build --release -F cuda` starts | ~40 | |
| lgb_rs `-F cuda` wheel built + installed | ~688 (~10m15s) | |
| pip lightgbm lacked CUDA → source rebuild fallback triggered | ~708 | `Please recompile with CMake option -DUSE_CUDA=1` |
| official lightgbm rebuilt from source (USE_CUDA=ON) | ~1577 | |
| 3-backend × 2-shape × 3-run matrix runs | ~1577 → ~26874 | **~7.0 hours** of matrix wall-clock |
| `>>> A/B VERDICT: FAIL <<<` emitted | 26873.9 | non-zero exit → Kaggle ERROR |

## Interpretation

The full A/B matrix consumed **~7.0 hours** (~1577s → ~26874s) — roughly **~23 min per
arm** for a 100-tree GBDT. That is a strong signal of a **severe on-device slowdown vs
host-CUDA**: the on-device per-leaf launch-bound path blew past the D-04 `<= 1.05x`
not-slower bar (the launch-bound architectural risk that the audit-before-wire gate
existed to catch). The verdict `FAIL` is definitive.

## CAVEAT — exact numbers NOT captured this run

**Kaggle did not commit `results.{md,json}` from this run.** The run left a very large
`/kaggle/working` (the cloned repo + Rust `target/` + many 500k-row `.npy` prediction
files) and **exceeded the Kaggle output-size cap, so the tiny results files were dropped**.
Only the run log and `_ab_worker.py` survived.

Consequences:

- The **exact per-shape wall-clock ratios** (on-device / host-CUDA) are **NOT captured**.
- The **exact real-CUDA parity number** (`max_abs_on_host`, D-11) is **NOT captured**.
- The **exact `device_launches/tree`** (SC-2) values are **NOT captured**.

The **verdict (FAIL) is definitive**; the numeric breakdown behind it is **not** recorded
from this run. This capture gap is fixed for any future re-run — see the harness fix in
this same plan (23-04): `_emit_results` now also echoes the full `results.json` to stdout
bracketed by `<<<AB_RESULTS_JSON` / `AB_RESULTS_JSON>>>` sentinels, so the numbers survive
in the Kaggle **log** even when Kaggle drops the output files.

## Surviving evidence

- `evidence/kaggle-ab-run.log` — the full ~103KB JSON-stream run log, including the
  `>>> A/B VERDICT: FAIL <<<` line and the build/matrix timeline above. **This is the
  authoritative surviving evidence.**
- `evidence/kernel-metadata.json` — the Kaggle kernel spec used for the run.

## Decision

**NO FLIP.** `on_device_default()` remains `false` (its pre-verdict value from plan 23-01);
the CUDA on-device learner stays **opt-in** via `LGBM_CUDA_ON_DEVICE=1`. On-device-as-default
is **deferred** pending on-device CUDA perf work (the per-leaf launch-bound slowdown must be
closed before a re-audit). Per D-09, the phase is **DoD-complete**: the audit-before-wire gate
was exercised and the behavior-changing flip was correctly withheld on a failing proof.
