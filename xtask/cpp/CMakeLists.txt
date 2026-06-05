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
