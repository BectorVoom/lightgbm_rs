# Standalone CMake target for the dev-only RNG capture harness.
#
# This file does NOT live inside, and never modifies, the read-only LightGBM
# submodule tree (D-07 recommendation). It compiles a tiny `rng_capture`
# executable that includes `LightGBM/include` and exercises the reference
# `Random` LCG.
#
# `LightGBM::Random` (include/LightGBM/utils/random.h) is a self-contained,
# header-only class depending only on the C++ standard library, so the capture
# program compiles directly against the pinned header — NO lib_lightgbm build or
# link is required. (Deviation from the original plan, which assumed linking
# lib_lightgbm: the in-repo submodule's `external_libs/` are not checked out
# here, so a full lib build is unavailable; and since the RNG draws come entirely
# from the header source, a header-only capture is numerically identical and
# remains faithful to the exact pinned reference. See REFERENCE_MANIFEST.md.)
#
# Invoked by `cargo run -p xtask -- regen`; not part of `cargo build`.

cmake_minimum_required(VERSION 3.28)
project(lgbm_rs_rng_capture LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Path to the in-repo, read-only LightGBM submodule. Overridable via
# -DLIGHTGBM_DIR=... ; defaults to ../../../LightGBM relative to this file
# (xtask/cpp -> repo root -> LightGBM).
if(NOT DEFINED LIGHTGBM_DIR)
  set(LIGHTGBM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../LightGBM")
endif()
get_filename_component(LIGHTGBM_DIR "${LIGHTGBM_DIR}" ABSOLUTE)

if(NOT EXISTS "${LIGHTGBM_DIR}/include/LightGBM/utils/random.h")
  message(FATAL_ERROR
    "LightGBM submodule not found at ${LIGHTGBM_DIR} "
    "(expected include/LightGBM/utils/random.h). Pass -DLIGHTGBM_DIR=<path>.")
endif()

# The reference `Random` is header-only (see header comment above), so we compile
# `rng_capture` directly against the pinned `LightGBM/include` headers with the
# default f32 width — `SCORE_T_USE_DOUBLE` / `LABEL_T_USE_DOUBLE` are NOT defined
# (D-01). No `add_subdirectory(LIGHTGBM_DIR)` / lib link: the RNG draws depend
# only on the header source, and the submodule's external_libs are not vendored
# here. The deterministic CPU-only build flags for future (training) goldens are
# recorded in REFERENCE_MANIFEST.md, not forced here.
add_executable(rng_capture rng_capture.cpp)
target_include_directories(rng_capture PRIVATE "${LIGHTGBM_DIR}/include")

# bin_capture: numeric binning goldens (Phase 2, layers 1+2).
#
# The authoritative BinMapper::FindBin/ValueToBin live in src/io/bin.cpp, which
# transitively #includes common.h -> fast_double_parser.h + fmt/format.h from
# external_libs/. Those submodules are present here only as EMPTY directories
# (the LightGBM tree is git-untracked and its external_libs are not vendored),
# so bin.cpp itself cannot be compiled. bin_capture.cpp therefore VERBATIM-
# transcribes the numeric FindBin/GreedyFindBin/.../ValueToBin/NeedFilter from
# the pinned bin.cpp/bin.h using the real std::nextafter (== GetDoubleUpperBound)
# — emitting byte-identical numeric goldens — and includes the header-only
# LightGBM/include only for the genuine reference Random (sampling). This mirrors
# the rng_capture header-only discipline (see REFERENCE_MANIFEST.md). It does NOT
# add_subdirectory the submodule and references NO external_libs header dirs
# (they are empty / unbuildable in this environment).
add_executable(bin_capture bin_capture.cpp)
target_include_directories(bin_capture PRIVATE "${LIGHTGBM_DIR}/include")

# kernel_capture: compute-kernel histogram goldens (Phase 4, D-02 / D-02a).
#
# The authoritative ConstructHistogram lives in src/io/dense_bin.hpp /
# sparse_bin.hpp, which (via <LightGBM/bin.h> -> common.h) transitively #include
# fast_double_parser.h + fmt/format.h from external_libs/. Those submodules are
# present here only as EMPTY directories (the LightGBM tree is git-untracked and
# its external_libs are not vendored), so neither header can be compiled. So
# kernel_capture.cpp VERBATIM-transcribes the ConstructHistogram accumulation
# bodies from the pinned dense_bin.hpp:130-141 / sparse_bin.hpp:138-152 (f32 read
# / hist_t=double accumulate) — emitting byte-identical histogram goldens — and
# includes the header-only LightGBM/include only for the genuine reference Random
# (synthetic inputs). This mirrors the rng_capture / bin_capture header-only
# discipline (see REFERENCE_MANIFEST.md). It does NOT add_subdirectory the
# submodule and references NO external_libs header dirs (empty / unbuildable here).
add_executable(kernel_capture kernel_capture.cpp)
target_include_directories(kernel_capture PRIVATE "${LIGHTGBM_DIR}/include")

# learner_capture: serial tree-learner goldens (Phase 5, D-06 per-split / D-07
# per-tree). SCAFFOLD in Plan 05-02; full transcription in Plan 03/04.
#
# The authoritative SerialTreeLearner lives in
# src/treelearner/serial_tree_learner.cpp, which (via <LightGBM/dataset.h> ->
# common.h) transitively #includes fast_double_parser.h + fmt/format.h from
# external_libs/. Those submodules are present here only as EMPTY directories
# (the LightGBM tree is git-untracked and its external_libs are not vendored), so
# serial_tree_learner.cpp cannot be compiled and no lib_lightgbm can be built or
# linked. learner_capture.cpp therefore VERBATIM-transcribes the learner growth
# loop (Plan 03/04) — reusing kernel_capture's already-transcribed gain/split math
# — and includes the header-only LightGBM/include only for the genuine reference
# Random. Same discipline as rng_capture / bin_capture / kernel_capture: no
# external_libs, no lib_lightgbm link, no C++ toolchain at `cargo test` time (the
# golden is committed). See REFERENCE_MANIFEST.md.
add_executable(learner_capture learner_capture.cpp)
target_include_directories(learner_capture PRIVATE "${LIGHTGBM_DIR}/include")

# primitive_capture: device-primitive HIP goldens (Phase 14, 14-02, ODL-01 / D-03).
#
# Unlike every sibling target above, this is a HIP (.cu) harness built with `hipcc`:
# primitive_capture.cu VERBATIM-transcribes the AMD-fork device primitives + the
# multi-kernel global wrappers (cuda_algorithms.hpp / cuda_algorithms.cu) and wraps
# each __device__ block helper in a one-line __global__ shim so it is launchable.
# Same self-contained discipline as the other harnesses — NO external_libs, NO
# lib_lightgbm link (the LightGBM external_libs are empty here); it only needs
# `-I LightGBM/include` for the header-only LightGBM::Random (synthetic inputs).
#
# It is built via a custom command invoking hipcc DIRECTLY rather than
# `enable_language(HIP)`, so the project never makes HIP a hard requirement of this
# standalone build — the rng/bin/kernel/learner C++ targets must stay buildable on a
# box with no HIP toolchain. The target is skipped cleanly when hipcc is absent
# (capture then needs a ROCm/CUDA box — RESEARCH A3). The offload arch defaults to
# the local spoofed gfx1100 APU and is overridable via -DLGBM_HIP_ARCH=<arch>.
find_program(LGBM_HIPCC NAMES hipcc)
set(LGBM_HIP_ARCH "gfx1100" CACHE STRING
  "HIP offload arch for primitive_capture (default: local spoofed gfx1100 APU)")
if(LGBM_HIPCC)
  # Target name intentionally differs from the OUTPUT file basename to avoid the
  # CMake "circular <target> <- <file>" self-dependency. The COMMAND always runs
  # when the target is built (hipcc is fast), so the exe is always fresh w.r.t. the
  # .cu source. `cargo run -p xtask -- primitive-capture` builds `--target
  # primitive_capture` and then locates the `primitive_capture` executable.
  add_custom_target(primitive_capture
    COMMAND "${LGBM_HIPCC}" -O2 "--offload-arch=${LGBM_HIP_ARCH}"
            -I "${LIGHTGBM_DIR}/include"
            -o "${CMAKE_CURRENT_BINARY_DIR}/primitive_capture"
            "${CMAKE_CURRENT_SOURCE_DIR}/primitive_capture.cu"
    BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/primitive_capture"
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/primitive_capture.cu"
    COMMENT "Building primitive_capture (hipcc, ${LGBM_HIP_ARCH})"
    VERBATIM)
else()
  message(STATUS
    "hipcc not found — primitive_capture (14-02 device-primitive goldens) target "
    "skipped. Capture needs a ROCm/CUDA box; committed fixtures need no toolchain.")
endif()
