# Phase 4: Compute Backend (CPU-first f32 histograms → ROCm) - Pattern Map

**Mapped:** 2026-06-05
**Files analyzed:** 11 new/modified
**Analogs found:** 10 / 11 (1 partial — the `#[cube]` kernels have no in-repo Rust analog; cubecl source + C++ reference are the templates)

This is a faithful-1:1-C++-mirror Rust port. Every new file should copy the idioms the existing `lgbm-*` crates already established (committed-golden + idempotent-regen + header-only-transcription discipline, `thiserror` boundary errors, `CARGO_MANIFEST_DIR` fixture replay, bit-exact-vs-`~1e-6` comparator split). The kernel *bodies* are transcribed from the C++ reference, but their *surrounding scaffolding* (crate layout, error type, capture subcommand, parity test) copies the patterns below verbatim.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/lgbm-compute/src/lib.rs` (modify) | trait/seam | request-response | self (existing `Backend` skeleton) + `crates/lgbm-dataset/src/lib.rs` (re-export layout) | exact (the seam being filled) |
| `crates/lgbm-compute/src/error.rs` (new) | error | — | `crates/lgbm-core/src/error.rs` | exact |
| `crates/lgbm-compute/src/runtime.rs` (new) | config/runtime-select | request-response | `crates/lgbm-dataset/src/bin/mod.rs` (factory dispatch) + cubecl source (no Rust analog for caps query) | role-match |
| `crates/lgbm-compute/src/gain.rs` (new) | utility (math) | transform | `xtask/cpp/bin_capture.cpp` (verbatim-transcription discipline) + `crates/lgbm-core/src/types.rs` (constants) | role-match |
| `crates/lgbm-compute/src/kernels/histogram.rs` (new) | kernel | transform/batch | C++ `dense_bin.hpp:99-141` + cubecl `#[cube]` shape (no in-repo Rust analog) | partial (transcription) |
| `crates/lgbm-compute/src/kernels/split.rs` (new) | kernel | transform | C++ `feature_histogram.hpp:711-1000` + `gain.rs` | partial (transcription) |
| `crates/lgbm-compute/src/kernels/partition.rs` (new) | kernel | transform/batch | C++ `data_partition.hpp:101` | partial (transcription) |
| `crates/lgbm-compute/Cargo.toml` (modify) | config | — | `crates/lgbm-dataset/Cargo.toml` (features + dev-deps) | exact |
| `xtask/cpp/kernel_capture.cpp` (new) | test-harness (C++) | file-I/O | `xtask/cpp/bin_capture.cpp` | exact |
| `xtask/src/main.rs` (modify: `kernel-capture` subcommand) | test-harness driver | file-I/O | self (`bin_capture()` / `model_capture()` fns) | exact |
| `crates/oracle-harness/tests/kernel_parity.rs` (new) | test | file-I/O | `crates/lgbm-dataset/tests/bin_storage_layout.rs` + `crates/oracle-harness/tests/rng_parity.rs` | exact |
| `xtask/cpp/CMakeLists.txt` (modify: add `kernel_capture` target) | config | — | self (`bin_capture` target block) | exact |
| `tests/fixtures/kernels/` → **actually** `crates/lgbm-compute/tests/fixtures/kernels/` | fixtures | — | `crates/lgbm-dataset/tests/fixtures/` | exact |

> **Fixture-location correction (load-bearing):** there is NO top-level `tests/fixtures/` in this repo. Committed goldens live **inside the consuming crate** under `crates/<crate>/tests/fixtures/` (dataset) or `crates/oracle-harness/fixtures/` (rng). Per the `CARGO_MANIFEST_DIR` replay idiom (below), kernel goldens must live under the crate that runs `kernel_parity.rs` — i.e. `crates/oracle-harness/tests/fixtures/kernels/` (replayed via `env!("CARGO_MANIFEST_DIR")`), NOT a repo-root `tests/fixtures/`. The planner should resolve this against the RESEARCH "Recommended Project Structure" which wrote `tests/fixtures/kernels/`.

## Pattern Assignments

### `crates/lgbm-compute/src/error.rs` (new — error, thiserror boundary)

**Analog:** `crates/lgbm-core/src/error.rs` (exact). Copy the module-doc → `#[derive(Debug, Error)]` enum → `#[error("...")]` per-variant → `#[from]` transparent wrapper structure verbatim. Security V5 (RESEARCH §Security): validate kernel inputs at the `Backend` boundary into typed `ComputeError` variants — never panic.

**Module doc + derive pattern** (`error.rs:1-19`):
```rust
//! Structured domain error types at the `lgbm-core` boundary (FND-04).
//! Uses `thiserror` derive (CLAUDE.md mandate) — never hand-roll
//! `impl std::error::Error`.
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ConfigError {
    #[error("parameter `{param}` has invalid value `{value}` for its expected type")]
    InvalidType { param: String, value: String },
    ...
}
```

**For `ComputeError`** mirror this: variants such as `BinIndexOutOfRange { row, bin, num_bin }` (Security V5 — out-of-bounds bin → typed error not UB), `LengthMismatch { expected, actual }` (array-length consistency), `CapabilityUnavailable { feature }` (Plane/f64/atomic gate miss), and a `Runtime { detail }` wrapper for cubecl launch failures. Keep it `thiserror`-derived; do NOT depend on `anyhow` inside the library crate (anyhow is for app/high-level layers per CLAUDE.md — `xtask` uses it, library crates use `thiserror`).

---

### `crates/lgbm-compute/src/lib.rs` (modify — trait/seam)

**Analog:** the existing skeleton (lines 1-29, shown below) + `crates/lgbm-dataset/src/lib.rs:13-31` for the `pub mod` + `pub use` re-export layout.

**Existing skeleton to fill** (`lib.rs:22-28`):
```rust
pub trait Backend {
    /// The concrete CubeCL runtime this backend dispatches kernels to.
    type Runtime;
}
```

**Fill pattern (D-01 whole-kernel ops):** bind `type Runtime: cubecl::Runtime;` and add the three coarse methods named exactly per RESEARCH (`construct_histograms`, `find_best_split`, `data_partition`). Per RESEARCH Pitfall 6 / Assumption A3, confirm with the planner whether a fourth `subtract_histograms` op is in scope (recommended: yes — it is histogram-layer math the Phase-5 learner orchestrates). All method signatures take `lgbm-dataset` bin types + f32 grad/hess slices and return f64 histogram cells / a `SplitInfo` struct / partition indices — see Shared Pattern "Kernel I/O shape" below.

**Module layout to add** (mirror `lgbm-dataset/src/lib.rs:13-31`):
```rust
pub mod error;
pub mod runtime;
pub mod gain;
pub mod kernels;   // contains histogram, split, partition submodules

pub use error::ComputeError;
pub use runtime::{Backend, ...};  // Backend trait lives where the seam is cleanest
```

**CMP-01 containment (carried-forward, non-negotiable):** this crate is the ONLY place `cubecl` type names appear. Keep the existing module doc's promise ("downstream crates should depend only on the [`Backend`] abstraction, never on `cubecl`"). The CMP-01 guard test (RESEARCH Test Map) greps that no crate above `lgbm-compute` names a cubecl runtime.

---

### `crates/lgbm-compute/Cargo.toml` (modify — config)

**Analog:** `crates/lgbm-dataset/Cargo.toml` (exact — feature block + `[dependencies]` path-deps + `[dev-dependencies] oracle-harness`).

**Current state** (`crates/lgbm-compute/Cargo.toml:7-8`):
```toml
[dependencies]
cubecl.workspace = true
```

**Add** (RESEARCH §Installation; feature names are Claude's discretion, `cubecl/cpu` + `cubecl/hip` are the verified upstream feature names):
```toml
[dependencies]
cubecl = { workspace = true, features = ["cpu"] }
lgbm-core = { path = "../lgbm-core" }
lgbm-dataset = { path = "../lgbm-dataset" }
thiserror = { workspace = true }

[features]
default = ["cpu"]
cpu = []
rocm = ["cubecl/hip"]   # opt-in; a CPU-only build omits this (SC#1)

[dev-dependencies]
oracle-harness = { path = "../oracle-harness" }   # for in-crate determinism_spike / capability tests
```
Workspace already pins `cubecl = "0.10.0"`, `thiserror = "2.0.18"`, `anyhow = "1.0.102"` (root `Cargo.toml:17-19`) — use `.workspace = true`, never re-pin versions. Note `[workspace.dependencies]` does NOT list `lgbm-core`/`lgbm-dataset`, so those are `{ path = "..." }` exactly as `lgbm-dataset/Cargo.toml` does for `lgbm-core`.

---

### `crates/lgbm-compute/src/runtime.rs` (new — runtime selection + capability gate)

**Analog:** `crates/lgbm-dataset/src/bin/mod.rs:115-...` (factory width-selection dispatch, the closest Rust pattern for "branch on a runtime property") + RESEARCH Pattern 2 (cubecl caps query — no in-repo analog).

**Factory-dispatch idiom to mirror** (from `bin/mod.rs` `create_dense_bin` doc, lines 115-120):
```rust
/// num_bin <= 16  -> DenseBin<u8, true>  (IS_4BIT)
/// num_bin <= 256 -> DenseBin<u8, false>
```
Apply the same "branch on a runtime fact, return a `Box<dyn>` / enum" shape to runtime selection (`cpu` vs `rocm`) and to the capability gate (`ReducePath::Plane` vs `ReducePath::Sequential`).

**Capability gate (CMP-04, RESEARCH Pattern 2)** — verified cubecl 0.10.0 API:
```rust
use cubecl::ir::features::{Plane, AtomicUsage};
let has_plane = client.features().plane.contains(Plane::Ops);
let has_f64   = client.features().supports_type(/* f64 storage */);
let has_f32_atomic = client.properties().atomic_type_usage(atomic_ty).contains(AtomicUsage::Add);
let reduce = if has_plane { ReducePath::Plane } else { ReducePath::Sequential };
```
Matrix (RESEARCH Pitfall 2): cubecl-cpu → `plane=false, f64=true, atomic=false` (Sequential fold, the anchor); cubecl-hip gfx1100 → `plane=true, f64=false, atomic=true`. Gate EVERY divergent feature; the sequential fold IS the CPU path, not a fallback option.

---

### `crates/lgbm-compute/src/gain.rs` (new — gain math, host helpers mirrored into kernel)

**Analog:** the verbatim-transcription discipline of `xtask/cpp/bin_capture.cpp` (transcribe C++ exactly, cite source lines) + `crates/lgbm-core/src/types.rs` for the constants.

**Constants already ported — REUSE, do not redefine** (`lgbm-core/src/types.rs:28-35`):
```rust
pub const K_EPSILON: f32 = 1e-15;       // C++ kEpsilon = 1e-15f (float literal — preserve f32→f64 widening)
pub const K_ZERO_THRESHOLD: f64 = 1e-35;
```
RESEARCH Pitfall 4: `2*kEpsilon` hessian bump at `feature_histogram.hpp:172`; scan seeds `sum_*_hessian = kEpsilon` at `:862`/`:935`. Transcribe these placements exactly.

**Gain math to transcribe verbatim** (C++ `feature_histogram.hpp:711-734`, cited in RESEARCH Code Examples):
```cpp
static double ThresholdL1(double s, double l1) {
    const double reg_s = std::max(0.0, std::fabs(s) - l1);
    return Common::Sign(s) * reg_s;
}
// GetLeafGain: USE_L1 -> (ThresholdL1(g,l1)^2)/(h+l2)  else (g*g)/(h+l2)
// GetSplitGains = GetLeafGain(left) + GetLeafGain(right)
// CalculateSplittedLeafOutput: USE_L1 -> -ThresholdL1(g,l1)/(h+l2)  else -g/(h+l2)
```

**Gain-config surface (D-01a)** — all fields already exist on `lgbm-core::Config` (verified `config/mod.rs`): `min_data_in_leaf` (i32), `min_sum_hessian_in_leaf` (f64), `max_delta_step` (f64), `lambda_l1` (f64), `lambda_l2` (f64), `min_gain_to_split` (f64), `path_smooth` (f64). Pass these into `find_best_split` as a small `GainConfig` struct extracted from `Config` — do NOT take a `&Config` into the kernel (keep the kernel surface minimal/comptime-friendly).

---

### `crates/lgbm-compute/src/kernels/{histogram,split,partition}.rs` (new — `#[cube]` kernels)

**Analog:** PARTIAL — no in-repo Rust `#[cube]` kernel exists. Template = cubecl 0.10.0 source (`#[cube(launch)]` shape, RESEARCH Pattern 1, verified against vendored source) for the *scaffolding*, and the C++ reference for the *body* (transcribed verbatim like `bin_capture.cpp`).

**`#[cube]` launch shape** (RESEARCH Pattern 1, verified `cubecl-core-0.10.0/src/runtime_tests/atomic.rs`):
```rust
#[cube(launch)]
fn construct_hist_kernel(binned: &Array<u32>, grad: &Array<f32>, hess: &Array<f32>, out: &mut Array<f32 /*f64 in practice*/>) {
    if UNIT_POS == 0 {            // single-owner ordered fold — the deterministic anchor (Pitfall 1)
        for i in 0..binned.len() {
            let ti = binned[i] * 2;
            out[ti] += grad[i];
            out[ti + 1] += hess[i];
        }
    }
}
// launch: kernel::launch::<R>(&client, CubeCount::new_single(), CubeDim::new_1d(1), args...)
```

**Histogram body to transcribe** (C++ `dense_bin.hpp:99-141`, RESEARCH Code Examples):
```cpp
const auto ti = static_cast<uint32_t>(data(idx)) << 1;   // bin<<1, stride-2 [grad,hess]
grad[ti] += ordered_gradients[i];   // f32 read, f64 accumulate (hist_t=double, Pitfall 3)
hess[ti] += ordered_hessians[i];
```

**Determinism mandate (Pitfall 1, D-04/D-04a):** cubecl-cpu spawns one OS thread per cube unit — it is NOT sequential. Use `CubeDim::new_1d(1)` (single-owner fold) so the f64 fold order matches C++ `num_threads=1` exactly. The Wave-0 spike (`tests/determinism_spike.rs`) must prove byte-identical f64 output across N≥20 launches BEFORE the full suite is built; failure → D-04a fallback (relax cubecl-cpu anchor to ~1e-6).

**Split scan body** (C++ `feature_histogram.hpp:862-934` REVERSE branch, RESEARCH Code Examples) — preserve the `SKIP_DEFAULT_BIN` continue, the `offset` arithmetic, threshold `t-1+offset` (REVERSE) / `t+offset` (forward), and the exact `min_data_in_leaf`/`min_sum_hessian_in_leaf`/`min_gain_shift` gate order (Pitfall 5). Do NOT restructure the loop.

**Anti-patterns (RESEARCH):** no atomics for the anchor (nondeterministic + unsupported on cpu); no f32 histogram accumulation (must be f64 on cpu to match `hist_t=double`); no "improving" the subtraction trick / scan order; never name a cubecl runtime above this crate.

---

### `xtask/cpp/kernel_capture.cpp` (new — C++ header-only transcription harness)

**Analog:** `xtask/cpp/bin_capture.cpp` (EXACT — same file, same discipline). Copy its top-of-file rationale comment, the `#include <LightGBM/utils/random.h>` header-only include, the `namespace {`-wrapped verbatim transcriptions with cited C++ source line ranges, the `F64Bits`/`F32Bits` raw-bit serializers (`bin_capture.cpp:1061-1071`), the `CaseGen`/`CaseSpec`/`EmitCase` deterministic-corpus generator (`bin_capture.cpp:1073-1095`), and the line-delimited `#`-comment fixture format.

**Why header-only (copy this rationale verbatim, retargeted):** `bin_capture.cpp:9-37` explains that `bin.cpp`/`feature_histogram.cpp` pull in `common.h` → `fast_double_parser.h` + `fmt/format.h` from `external_libs/`, which are EMPTY dirs here (LightGBM tree untracked, submodules unvendored). So `kernel_capture.cpp` VERBATIM-transcribes `ConstructHistogram` (`dense_bin.hpp`/`sparse_bin.hpp`), `FindBestThreshold*`/`GetSplitGains`/`GetLeafGain`/`ThresholdL1` (`feature_histogram.hpp:711-1000`), and the data-partition routing (`data_partition.hpp:101`) — depending only on `std` + the genuine header-only `LightGBM::Random` for synthetic inputs. Reuse the `IBin`/`DenseBin`/`SparseBin`/`BinMapper` transcriptions ALREADY PRESENT in `bin_capture.cpp` (lines 588-714, 110-147) as the bin-storage side of the histogram input (D-02a: reuse Phase-2 forms).

**Serialization** (`bin_capture.cpp:1061-1071`):
```cpp
uint64_t F64Bits(double d) { uint64_t b; std::memcpy(&b, &d, sizeof(b)); return b; }
uint32_t F32Bits(float f)  { uint32_t b; std::memcpy(&b, &f, sizeof(b)); return b; }
```
Emit f64 histogram cells as raw u64 bits, f32 grad/hess/leaf-output as raw u32 bits — for bit-exact replay. **Layered goldens (Specifics):** separate histogram-accumulation / best-split / data-partition records so a failure localizes to accumulate vs gain-scan vs partition.

---

### `xtask/cpp/CMakeLists.txt` (modify — add `kernel_capture` target)

**Analog:** the `bin_capture` target block in the same file (EXACT). Copy verbatim:
```cmake
add_executable(kernel_capture kernel_capture.cpp)
target_include_directories(kernel_capture PRIVATE "${LIGHTGBM_DIR}/include")
```
Same `cmake_minimum_required(3.28)` / `CMAKE_CXX_STANDARD 11` / `-DLIGHTGBM_DIR` guard already at the top of the file. No `add_subdirectory`, no `external_libs` include dirs (header-only).

---

### `xtask/src/main.rs` (modify — add `kernel-capture` subcommand)

**Analog:** the `bin_capture()` fn (`xtask/src/main.rs:201-333`, EXACT). Copy its structure step-for-step:
1. add `Some("kernel-capture") => kernel_capture(),` to the `match` (`main.rs:67-73`) and update the usage strings;
2. add a `KERNEL_MASTER_SEED` const next to `BIN_MASTER_SEED` (`main.rs:43`) — the single source of randomness (D-14 idempotency);
3. `workspace_root()` → `verify_toolchain()` → assert `LightGBM/include/.../random.h` exists → cmake configure + `cmake --build --target kernel_capture` (reuse `target/xtask-cpp-build`) → `locate_exe(&build_dir, "kernel_capture")` → run with the fixture output path(s) + seed args → assert outputs written → `write_manifest()` → print the idempotency-check reminder.

**Output fixture dir** (mirror `bin_capture`'s `crates/lgbm-dataset/tests/fixtures`, but for the crate that replays — `oracle-harness`):
```rust
let fixtures_dir = root.join("crates/oracle-harness/tests/fixtures/kernels");
std::fs::create_dir_all(&fixtures_dir)?;
```
**Manifest:** extend the `write_manifest` content (`main.rs:549-779`) with a "## Kernel Golden Set (Phase 4)" section recording `KERNEL_MASTER_SEED`, the synthetic-input path coverage (D-02a), and the kernel-capture command — keeping it a pure function of recorded constants (idempotent).

---

### `crates/oracle-harness/tests/kernel_parity.rs` (new — golden replay test)

**Analog:** `crates/lgbm-dataset/tests/bin_storage_layout.rs` (EXACT for the parse/replay/compare structure) + `crates/oracle-harness/tests/rng_parity.rs` (EXACT for the graceful-SKIP-when-fixture-absent idiom).

**Fixture-path + SKIP idiom** (`bin_storage_layout.rs:32-34`, `rng_parity.rs:52-61`):
```rust
fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/kernels/...")
}
// in the test:
let Ok(text) = std::fs::read_to_string(&path) else {
    eprintln!("kernel_parity: SKIP — fixture not found. Run `cargo run -p xtask -- kernel-capture` ...");
    return;   // keeps `cargo test` green pre-capture
};
```

**Line-parse helpers** (`bin_storage_layout.rs:36-81`): copy `field()`, `parse_i32`/`parse_u64`, `parse_f64_bits_list` (f64 via `from_bits(u64)`), `parse_u32_list`, `parse_byte_list` verbatim.

**Comparator import (LOAD-BEARING — `lib.rs` re-export gap):** `oracle-harness/src/lib.rs:10` re-exports ONLY `compare_within`/`abs_diff_within`/`Mismatch`/`ORACLE_TOL` — NOT the exact comparators. Import the exact ones via the full module path exactly as `bin_storage_layout.rs:30` does:
```rust
use oracle_harness::comparator::{compare_exact_f64_bits, compare_within, ORACLE_TOL};
```
- **cubecl-cpu anchor (D-04, hard CPU gate):** compare f64 histogram cells / f64 fold results BIT-EXACT via `compare_exact_f64_bits` (`comparator.rs:150`).
- **cubecl-hip (ROCm, separate gate, D-03a):** compare via `compare_within(rust, cpp, ORACLE_TOL)` at `ORACLE_TOL = 1e-6` (`comparator.rs:15`). Surface any ROCm gap explicitly (no silent pass).

**Test deps:** `oracle-harness/Cargo.toml` currently has only `lgbm-core` as a dev-dep. Add `lgbm-compute = { path = "../lgbm-compute" }` and `lgbm-dataset = { path = "../lgbm-dataset" }` to `[dev-dependencies]` so the test can drive the `Backend` and build inputs. (`oracle-harness` library itself stays cubecl-free — the dev-dep is test-only, CMP-01 intact.)

---

## Shared Patterns

### Committed-golden + idempotent-regen + header-only-transcription (carried Phase 1/2/3)
**Source:** `xtask/src/main.rs` (`bin_capture`/`model_capture` + `write_manifest`), `xtask/cpp/bin_capture.cpp` (transcription rationale), `crates/oracle-harness/fixtures/REFERENCE_MANIFEST.md`.
**Apply to:** `kernel_capture.cpp`, the `kernel-capture` subcommand, the kernel goldens.
Single recorded master seed → byte-identical regen (empty `git diff`). C++ toolchain needed ONLY at `kernel-capture` time; `cargo test` reads committed fixtures. Goldens committed under a TRACKED crate dir, NEVER referenced from the untracked `LightGBM/` tree at test time (memory: `lightgbm-ref-tree-untracked`).

### Bit-exact-vs-`~1e-6` comparator split
**Source:** `crates/oracle-harness/src/comparator.rs` — `compare_exact_f64_bits` (`:150`, raw `to_bits()`), `compare_exact_u32` (`:125`), `compare_exact_bytes` (`:172`), `compare_within` (`:92`, `ORACLE_TOL=1e-6` at `:15`), `Mismatch` first-divergence enum (`:20-56`).
**Apply to:** `kernel_parity.rs` (cpu anchor = exact f64-bits; hip = `~1e-6`), input validation in `runtime.rs`.

### `thiserror` boundary errors (library) vs `anyhow` (xtask)
**Source:** `crates/lgbm-core/src/error.rs` (thiserror); `xtask/src/main.rs:20` (`anyhow::{Context, Result, bail}`).
**Apply to:** `lgbm-compute/src/error.rs` uses `thiserror` (ComputeError); the `kernel-capture` subcommand uses `anyhow` like its sibling subcommands. Never mix.

### `CARGO_MANIFEST_DIR` fixture replay + graceful pre-capture SKIP
**Source:** `rng_parity.rs:22-24,52-61`, `bin_storage_layout.rs:32-34`.
**Apply to:** `kernel_parity.rs` and any in-crate `lgbm-compute/tests/*.rs` (determinism_spike, capability) that read fixtures.

### Kernel I/O shape (D-01 whole-kernel, faithful-mirror)
**Source:** C++ `dense_bin.hpp:99-141` (stride-2 `[grad,hess]`, `bin<<1`, f64 cells), `feature_histogram.hpp:711-1000` (gain + scan), `data_partition.hpp:101` (row reorder + `leaf_begin_`/`leaf_count_`); inputs from `lgbm-dataset` `Bin::data(idx)->u32` / `num_data()` (`bin/mod.rs:91-102`) and `FeatureGroup.bin_offsets_`/`num_total_bin_` (`feature_group.rs:56-59`).
**Apply to:** all three kernel signatures on the `Backend` trait. Inputs = f32 ordered grad/hess + Phase-2 binned store (do NOT re-bin); histogram output cells = f64. `data_partition` mirrors the C++ stable row→{left,right} reorder; Phase-5 owns `leaf_begin_`/`leaf_count_` bookkeeping (RESEARCH Open Q3).

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `crates/lgbm-compute/src/kernels/*.rs` (`#[cube]` bodies) | kernel | transform | No prior Rust `#[cube]` kernel exists in-repo. Scaffolding copies cubecl 0.10.0 source (RESEARCH Pattern 1, verified vendored); bodies transcribe the C++ reference (`dense_bin.hpp`, `feature_histogram.hpp`, `data_partition.hpp`) under the `bin_capture.cpp` verbatim-transcription discipline. The cubecl-cpu determinism bet (D-04) has no precedent — the mandatory Wave-0 `determinism_spike` settles it empirically. |

## Metadata

**Analog search scope:** `crates/lgbm-compute`, `crates/lgbm-core`, `crates/lgbm-dataset`, `crates/oracle-harness`, `xtask` (`src/main.rs`, `cpp/`), root `Cargo.toml`. C++ reference line numbers sourced from CONTEXT/RESEARCH canonical refs (not re-read here — the untracked `LightGBM/` tree is authoritative).
**Files scanned:** 14 (4 read in full, 6 grepped, plus the existing skeleton + Cargo manifests).
**Pattern extraction date:** 2026-06-05
