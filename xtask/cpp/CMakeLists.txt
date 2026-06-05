# Standalone CMake target for the dev-only RNG capture harness.
#
# This file does NOT live inside, and never modifies, the read-only LightGBM
# submodule tree (D-07 recommendation). It builds the pinned `lib_lightgbm`
# from the in-repo submodule and links a tiny `rng_capture` executable that
# includes `LightGBM/include` and exercises the reference `Random` LCG.
#
# `LightGBM::Random` is header-only, so the capture program compiles purely
# against the headers; we still build and link `_lightgbm` to (a) prove the
# pinned commit builds with the recorded deterministic flags and (b) keep the
# manifest's "links lib_lightgbm" contract (ORA-02 / D-05).
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

# Build lib_lightgbm from the pinned submodule with deterministic-friendly flags.
# We force the CPU-only, single-precision (default float) configuration recorded
# in REFERENCE_MANIFEST.md: no GPU/CUDA/MPI, no SWIG, no CLI, no C++ tests.
set(USE_GPU OFF CACHE BOOL "" FORCE)
set(USE_CUDA OFF CACHE BOOL "" FORCE)
set(USE_MPI OFF CACHE BOOL "" FORCE)
set(USE_SWIG OFF CACHE BOOL "" FORCE)
set(BUILD_CLI OFF CACHE BOOL "" FORCE)
set(BUILD_CPP_TEST OFF CACHE BOOL "" FORCE)
set(BUILD_STATIC_LIB ON CACHE BOOL "" FORCE)
set(INSTALL_HEADERS OFF CACHE BOOL "" FORCE)
# NOTE: do NOT define SCORE_T_USE_DOUBLE / LABEL_T_USE_DOUBLE — keep the f32
# default width (D-01).

add_subdirectory("${LIGHTGBM_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/lightgbm_build")

add_executable(rng_capture rng_capture.cpp)
target_include_directories(rng_capture PRIVATE "${LIGHTGBM_DIR}/include")
target_link_libraries(rng_capture PRIVATE _lightgbm)
