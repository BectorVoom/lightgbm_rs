# ROCm / HIP environment and the CubeCL kernel cache

Two things this project needs that are **configuration, not code**, so they never
show up in a code review — you only find them by running something.

---

## 1. Environment

```bash
source ~/rocm/env.sh          # the local userspace ROCm 7.1.5 install
cargo test --release -p lgbm-compute --features rocm
```

`env.sh` sets the three variables that matter, and they do different jobs at
different times:

| variable | when | failure mode if wrong |
|---|---|---|
| `ROCM_PATH` / `HIP_PATH` | **build** — `cubecl-hip-sys`'s `build.rs` shells out to `hipconfig` to pick the bindings version | silent: `build.rs` degrades to a `cargo::warning` so clippy works without a GPU |
| `PATH` (`$ROCM_PATH/bin`) | **build** — so `hipconfig` is findable at all | same silent degradation |
| `LD_LIBRARY_PATH` (`$ROCM_PATH/lib`) | **run** — `rustc-link-search` is link-time only and does not become an rpath | loud: `error while loading shared libraries: libhiprtc.so.7` |

Consequence worth internalising: **`cargo check --features rocm` passing tells you
nothing about your ROCm install.** `build.rs` deliberately falls back to the newest
bindings with only a warning when `hipconfig` is missing, so a clean check is not
evidence the toolchain is wired up. Run a test.

The local box also needs `HSA_OVERRIDE_GFX_VERSION=11.5.1` (gfx1152 Krackan Point
reports as gfx1151), which `env.sh` already exports.

---

## 2. The on-disk kernel cache (`cubecl.toml`)

CubeCL has **two** caches with **different defaults**:

| cache | field | default |
|---|---|---|
| autotune results | `AutotuneConfig::cache: CacheConfig` | `Target` — **on** |
| compiled kernels | `CompilationConfig::cache: Option<CacheConfig>` | `None` — **off** |

The asymmetry is the trap: `target/autotune` appearing on disk makes it look like
caching is already handled while every kernel is still re-JITted in every process.

`cubecl.toml` at the repo root turns the compiled-kernel cache on. See that file for
why the root is `"global"` rather than the usual `"target"` (short version: this is a
workspace, and `CacheConfig::Target` resolves against the *nearest* `Cargo.toml`, so
cargo's per-package cwd would fragment it into one cache per crate).

Measured on the local gfx1151, the 19 `--features rocm` test binaries back to back:

| | wall |
|---|---|
| cache off | 41.1 s |
| cache on, cold | 25.9 s |
| cache on, warm | **12.9 s** (3.2×) |

Which backends honour it, in cubecl 0.10:

| backend | on-disk cache | directory |
|---|---|---|
| `cubecl-hip` | yes | `<root>/hip/<ver>/hip-kernel/` (device binaries) |
| `cubecl-cuda` | yes | `<root>/cuda/<ver>/ptx/` |
| `cubecl-wgpu` | only with the `spirv` feature | `<root>/vulkan/<ver>/spirv/` |
| `cubecl-cpu` | **never** — in-memory `HashMap` only | — |

So the f64 CPU anchor (the merge gate) is entirely unaffected by this setting; it has
no compilation to amortise.

### Verifying it

```bash
ls ~/.config/hip/                                     # the directory should exist
RUST_LOG=cubecl_hip=trace cargo test --release --features rocm 2>&1 \
  | grep -c "Using compilation cache"
```

If the directory never appears, the config file is not being found. Lookup walks **up**
from the *process's* working directory, so a binary launched from outside the repo gets
defaults, silently — a typo or a missing file is never an error.

---

## 3. `HIPRTC_COMPILE_OPTIONS_APPEND=-ffp-contract=off`

Not currently set, and **not currently needed** — but worth understanding, because it
is the one setting here that changes *results* rather than whether things run.

hipRTC defaults to `-ffp-contract=fast`, letting the device compiler fuse `a * b + c`
into a single multiply-add. Rust never auto-contracts, so wherever this port
transcribes a C++ expression literally, the device may round once where the CPU f64
anchor rounds twice.

That is *not* a live problem for this port, for a specific reason: the one place the
C++ reference's own contraction is load-bearing
(`get_leaf_gain_given_output`'s `-(2·g·o + (h+λ)·o²)` under `max_delta_step` /
`path_smooth`) is written with an **explicit** `fused_mul_add`, chosen by measurement
against the reference — see `crates/lgbm-compute/src/gain.rs`. Explicit `fma` is a real
single-rounding instruction regardless of the contraction flag, so that call is
unaffected either way.

Verified on the local gfx1151: the full `-p lgbm-compute --features rocm` suite passes
identically with and without the flag (cache cleared in between).

**If you ever do set it, clear the cache first.** `KernelId::stable_hash()` describes
the *kernel* — its source, generics, and `#[comptime]` arguments. It knows nothing
about the environment hipRTC was invoked in, so a cache populated without the flag is
served back unchanged to a run that sets it, with no diagnostic:

```bash
rm -rf ~/.config/hip
HIPRTC_COMPILE_OPTIONS_APPEND=-ffp-contract=off cargo test --release --features rocm
```

Treat a device compiler flag exactly like a toolchain upgrade: changing `RUSTFLAGS`
triggers a rebuild, changing `HIPRTC_COMPILE_OPTIONS_APPEND` does not.
