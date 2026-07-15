# Evidence Ledger — Parity Gap Closure

Verified-this-session facts backing SPEC.md / PLAN.md. Labels per skill.

## Symbols & files (verified LOCAL)
- `format_g17`/`format_g6` — `crates/lgbm-model/src/format.rs:43,52`. `%g` formatters (DEC-1 substrate).
- `model_text::save` / `save_with_importance` — `crates/lgbm-model/src/model_text.rs:216,231`; field-order template for G2 (`:216-260`).
- `Booster::model_to_string` — `crates/lgbm/src/booster.rs:728` (facade template for G1/G2 methods, `model_from_string:752`).
- `Tree` node arrays — `crates/lgbm-model/src/tree.rs:94-155`; `decision_type` bit meaning `:24-26,45-50` (bit0 categorical, bit1 default-left, bits2-3 missing_type); `Tree::predict` `:269`; `Tree::to_string` `:790`.
- `GainConfig` — `crates/lgbm-compute/src/gain.rs:377-430`; already carries `max_delta_step`, `path_smooth`, `lambda_l*`, `min_*` from `Config::from_config`.
- `find_best_split_cpu` — `crates/lgbm-compute/src/kernels/split.rs:433-460`; doc states `penalty` hard-coded `1.0` "not yet implemented" and `max_delta_step`/`path_smooth` non-default → `ComputeError::Runtime`.
- NA gate — `crates/lgbm-treelearner/src/learner.rs:1113-1121`: `f.na_as_missing()` → `TreeLearnerError::Compute(ComputeError::Runtime{ detail:"…NA_AS_MISSING forward branch not implemented" })`.
- Config fields — `max_delta_step` `config/mod.rs:101`; `path_smooth` `:181` (IN_SCOPE); `convert_model`/`convert_model_language` `:260-263` (parsed, inert); alias `convert_model_file`→`convert_model` `alias.rs:160`; parsed in `set.rs:333`.
- Oracle idiom — `crates/oracle-harness/tests/predict_parity.rs:1-40`: `CARGO_MANIFEST_DIR` fixture root, `compare_within(.., ORACLE_TOL)`, SKIP-graceful `read_golden`.
- Fixture dirs — `crates/oracle-harness/tests/fixtures/{predict_modes,quantized,linear,...}`; test files listed research §3/§7.

## Dependencies (verified LOCAL — research §6)
- No new crate needed for G1/G2/G4/G5. `serde_json` absent from workspace (grep) → G2 hand-emits (DEC-1).
- Toolchain Rust 1.95.0 / edition 2024 / resolver 3; `cubecl` 0.10.0.

## Blockers (verified LOCAL — research §1,§8)
- `LightGBM/` C++ tree ABSENT this sandbox → G4/G5 algorithm details `[UNVERIFIED against C++ source]`; `config_drift.rs` cannot run. **P-1** checkout required.
- `lgbm-python` link failure (`mold: library not found: python3.14`) → Python-level G1/G2 exposure unvalidated; kept out of v1 scope.
- No `lightgbm==4.6` install here → goldens not generated this session; **P-2**.
- Commit `42249ca` on-device regression → run treelearner/on-device tests with `LGBM_CUDA_ON_DEVICE=0` (**P-3**); memory `resident-score-host-update-gotcha.md`.

## User decisions (this session)
- Scope = G2 + G1 + G4 + G5.
- DEC-1 G2 emitter = hand-emit `%g` (no serde_json).
- DEC-2 G1 API = `Booster::model_to_cpp() -> String` (no file side effect).

## Unverified / to confirm during implementation
- Exact C++ JSON key set/order (`Tree::NodeToJSON`, `GBDT::DumpModel`).
- Exact C++ if-else skeleton (`SaveModelToIfElse`/`Tree::NodeToIfElse`).
- NA missing-bin index + default-branch rule (serial_tree_learner.cpp / feature_histogram.hpp).
- `path_smooth`/`max_delta_step`/`penalty` formulas (feature_histogram.hpp GetLeafGain / CalculateSplittedLeafOutput).
- G5 per-feature penalty source (CEGB vs `meta_->penalty`); parent-output availability at `find_best_split_cpu` call site.
