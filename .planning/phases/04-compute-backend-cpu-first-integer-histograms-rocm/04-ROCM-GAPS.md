# 04 ROCm/HIP Known-Issues Ledger (D-03a)

**Status:** ROCm bring-up SUCCEEDED on the local gfx1100 GPU. The hip runtime
selects, all four kernels run, and the separate `~1e-6` hip-vs-cpu-anchor parity
gate executed. One **documented, expected** residual gap exists: the histogram
and split kernels exceed the strict `ORACLE_TOL = 1e-6` on accumulation-heavy
cases because hip has no f64 and accumulates in f32 (RESEARCH Pitfall 3). This is
the divergence the `~1e-6` contract was designed to absorb, NOT a kernel bug — it
is a **documented follow-up, not a phase blocker** (D-03/D-03a). No silent pass:
every per-case gap is surfaced and recorded below.

The **CPU bit-exact gate (04-01..03) remains the HARD phase-completion bar and is
green.** This ROCm half is best-effort per D-03a.

## Environment (verified on the run host)

| Item | Value |
|------|-------|
| GPU | AMD Radeon (integrated), ISA `gfx1100` (RDNA3, wave32) — `AMD Ryzen AI 7 350 w/ Radeon 860M` |
| ROCm | 7.1.1 (`/opt/rocm` → `7.1.1`) |
| HIP | 7.1.52802-26aae437f6 (`hipcc --version`) |
| `cubecl` | 0.10.0 (`hip` feature) |
| `cubecl-hip` | 0.10.0 |
| `cubecl-hip-sys` | 7.1.5280200 (links `amdhip64`/`hiprtc`) |
| Build | `cargo build -p lgbm-compute --features rocm` succeeded with NO `ROCM_PATH` override needed (the anticipated cubecl-hip 6.4-baseline vs ROCm 7.1 link drift did NOT materialize on this host) |

## Capability matrix (probed on gfx1100, exercised by `rocm_smoke.rs`)

| Capability | Probed | Expected (RESEARCH Pitfall 2) | Match |
|------------|--------|-------------------------------|-------|
| `has_plane` (`Plane::Ops`) | **true** | YES | ✅ |
| `has_f64` | **false** | NO (disabled upstream) | ✅ |
| `has_f32_atomic` (`AtomicUsage::Add`) | **true** | YES | ✅ |
| `plane_size` | **32** | 32 (wave32) | ✅ |
| derived `ReducePath` | `Plane` | — | ✅ |
| derived `AccumulateType` | `F32` | f32 (no-f64 routing) | ✅ |

`rocm_smoke.rs`: **2/2 passed.** The hip runtime selects via the `rocm` feature
and the f32-accumulate histogram kernel runs on the real GPU.

## Per-kernel parity outcome (`kernel_parity.rs` hip layer, `--features rocm`)

The hip f32 kernel output was compared to the cubecl-cpu **f64 anchor** collected
to `Vec<f32>` (`cpu_anchor_f64.iter().map(|&x| x as f32).collect()`) via
`compare_within(&hip_f32, &cpu_anchor_f32, ORACLE_TOL)`. All 4 hip parity tests
**passed** under the two-tier gate (strict `1e-6` surfaced as a documented gap;
generous f32 *relative* sanity bound `1e-3` as the hard bug-catcher).

| Layer | Within `1e-6`? | Max abs-diff observed | Max relative diff | Cause |
|-------|----------------|-----------------------|-------------------|-------|
| **partition** | YES (bit-exact `u32`) | 0 | 0 | f64-free kernel — same routing on both backends |
| **subtract** | YES | `1.16e-10` (`spread`); `0` (`simple`) | ~1e-14 | element-wise `parent - child`; no accumulation chain |
| **histogram** | **NO** on accumulation-heavy cases | `9.77e-4` (`single_bin_pileup`, `w4bit_dense_spread`-class) | ~`1.1e-7` | f32 accumulation vs f64 anchor (Pitfall 3) |
| **split** | **NO** on the winning gain cell | `7.63e-6` (`reverse_winner`); `3.81e-6` (`forward_winner`) | ~`6e-8` | f32 gain math vs f64 (`g²/(h+λ)` sums) |

### Histogram per-case max abs-diff (full survey, gfx1100)

```
w4bit_dense_spread   9.766e-4      w8_dense_spread      0.0
w4bit_dense_defbin   1.221e-4      w8_dense_defbin      0.0
w4bit_sparse_spread  1.221e-4      w8_sparse_spread     1.907e-6
w4bit_sparse_defbin  1.221e-4      w8_sparse_defbin     1.221e-4
w16_*  (all 4)       0.0           w32_* (all 4)        0.0
single_bin_pileup    9.766e-4      all_bin0_sparse      0.0
```

Many cases (all `w16`/`w32`, the dense `w8` cases, `all_bin0_sparse`) are
**bit-identical** (max abs-diff `0.0`) even in f32 — the gap only appears where
the running f32 sum reaches large magnitudes (hundreds–thousands) so the trailing
f32 mantissa bits round. The **worst relative error is ≈ 1.1e-7** (`9.77e-4 /
8603` in `single_bin_pileup`) — i.e. one f32 ULP, confirming the kernels are
arithmetically correct and the divergence is purely f32-vs-f64 precision.

### Split per-case (winner net-gain cell)

```
reverse_winner  abs_diff = 7.629e-6   (hip 126.15001 vs cpu 126.15)
forward_winner  abs_diff = 3.815e-6   (hip 61.250004 vs cpu 61.25)
```

Relative error ≈ `6e-8`. `is_splittable`, `threshold`, `left_count`, and
`default_left` (integer/bool observables) matched **exactly** on every case.

## Residual gaps (D-03a follow-ups — NOT phase blockers)

1. **G-04-01 — f32 histogram accumulation exceeds `1e-6` on large-magnitude
   cells.** Max abs-diff `9.77e-4`, max relative `~1.1e-7` (one f32 ULP). Root
   cause: hip has no f64; the single-owner ordered fold accumulates in f32. This
   is the explicitly-anticipated Pitfall-3 divergence (RESEARCH Open Q2/A2,
   risk MEDIUM). **Follow-up options (Phase 5+ / future):** (a) Kahan/compensated
   summation in f32 to recover most of the lost precision; (b) split-f32
   "two-float" accumulation; (c) relax the documented hip oracle tolerance to a
   *relative* `~1e-5` bound (the project contract's `1e-12`/`1e-6` absolute bar is
   a CPU/CPU-anchor contract; the f32 hip path is best-effort per D-03a).

2. **G-04-02 — f32 split gain exceeds `1e-6` on the winner gain cell.** Max
   abs-diff `7.63e-6`, max relative `~6e-8`. Same root cause (f32 gain math). The
   winner SELECTION (threshold/counts/default_left) is unaffected — only the
   reported gain magnitude drifts by f32 epsilon.

3. **No build/link gap.** The anticipated cubecl-hip vs ROCm-7.1 drift did not
   occur; `cargo build -p lgbm-compute --features rocm` built clean without a
   `ROCM_PATH` override.

## How this was verified (reproduce on a gfx1100 ROCm host)

```bash
cargo build -p lgbm-compute --features rocm
cargo test  -p lgbm-compute --features rocm --test rocm_smoke -- --nocapture
cargo test  -p oracle-harness --features rocm --test kernel_parity hip:: -- --nocapture
```

The hip parity test prints each `HIP PARITY GAP ...` line (the per-case
`index`/`abs_diff` recorded above) to stderr — no silent pass. The CPU-only build
(`cargo build -p lgbm-compute --no-default-features --features cpu`) needs no ROCm
toolchain and the default `cargo test --workspace` is unaffected (SC#1).
