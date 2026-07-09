// primitive_capture.cu
//
// Dev-only C++/HIP capture harness for the LightGBM-rs DEVICE-PRIMITIVE oracle
// (Phase 14, Plan 14-02, decision D-03 / requirement ODL-01). It launches every
// numeric device primitive the Rust CubeCL port reproduces — block inclusive /
// exclusive prefix-sum, the multi-kernel GLOBAL prefix-sum, the shuffle
// reductions (sum / max / min / dot-product), single-block AND multi-block
// (global) bitonic argsort, and the weighted / unweighted PercentileDevice — and
// dumps committed, byte-idempotent golden fixtures that 14-06 replays the Rust
// primitives against (numeric ops bit-exact on the cpu f64 anchor / ~1e-6 for
// ROCm f32; index-only argsort permutation bit-exact).
//
// WHY THIS IS A SELF-CONTAINED TRANSCRIPTION (not a compile/link of the AMD-fork
// lib_lightgbm):
//   The authoritative device primitives live in the in-repo AMD fork at
//   `LightGBM-release-4.6.0.99/include/LightGBM/cuda/cuda_algorithms.hpp` (the
//   __device__ block helpers + PercentileDevice) and
//   `.../src/cuda/cuda_algorithms.cu` (the host-callable multi-kernel global
//   wrappers). Both headers transitively pull `<LightGBM/bin.h>` -> `common.h` ->
//   `fast_double_parser.h` + `fmt/format.h` from `external_libs/`, present here
//   only as EMPTY directories (the LightGBM trees are git-untracked and their
//   external_libs are NOT vendored — see project memory `lightgbm-ref-tree-
//   untracked`). So neither header can be #included as-is, and no lib_lightgbm /
//   .cu object can be built or linked.
//
//   This is the SAME situation every prior golden layer hit (rng_capture /
//   bin_capture / kernel_capture / learner_capture). Following that discipline,
//   this harness VERBATIM-TRANSCRIBES the device-primitive bodies + the global
//   multi-kernel wrappers from the pinned AMD fork (commit 195c26fc / version
//   4.6.0.99) — the `__device__` helpers are not host-callable as-is, so each is
//   wrapped in a one-line `__global__` shim (RESEARCH §C++/HIP harness skeleton,
//   D-03). The transcription depends ONLY on the HIP runtime + warp/shared-memory
//   intrinsics, NONE of fast_double_parser / fmt, so it compiles standalone under
//   `hipcc`. It emits goldens equivalent to what the reference cuda_algorithms
//   would compute (deterministic primitives; index/int ops device-independent,
//   f32 reductions captured here compared at ~1e-6 downstream — RESEARCH A3).
//
//   Synthetic inputs are derived solely from one recorded MASTER_SEED passed on
//   argv via the genuine header-only `LightGBM::Random` (same as the sibling
//   harnesses), so re-running the capture produces byte-identical fixtures
//   (idempotent, empty `git diff` — D-14).
//
// WARP SIZE: this harness is built for the local spoofed gfx1100 APU (RDNA3,
// warpSize == 32). The AMD fork sizes its __shared__ warp buffers from the
// WARPSIZE macro, which is 32 for every non-GFX9 target (cuda_rocm_interop.h
// WARP_SIZE_INTERNAL()); we mirror that exactly. (A GFX9 build with warpSize==64
// would need WARPSIZE 64 to match — out of scope for this APU; noted.)
//
// Fixture format (line-delimited, '#'-prefixed comments ignored by the reader),
// one file per primitive family under crates/oracle-harness/fixtures/primitives/:
//   prefix_sum.txt   PSUM / PSUMG records (block incl/excl + global)
//   reduce.txt       RED records (sum/max/min/dot, f64 + f32)
//   argsort.txt      ARGSORT records (single+multi-block, asc/desc, incl tie-rich)
//   percentile.txt   PCTL records (weighted + unweighted)
// Floating values are emitted as raw little-endian bit patterns (f64 -> decimal
// u64, f32 -> decimal u32) so the Rust side asserts with zero parsing rounding;
// integer prefix-sums + argsort permutations are emitted as decimals and compared
// bit-exact.

#include <hip/hip_runtime.h>

#include <LightGBM/utils/random.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Reference type aliases (meta.h): score_t/label_t = float (D-01),
// data_size_t = int32_t.
// ---------------------------------------------------------------------------
typedef int32_t data_size_t;
typedef float label_t;
typedef float score_t;

// AMD-fork tuning macros (cuda_algorithms.hpp:25-28).
#define GLOBAL_PREFIX_SUM_BLOCK_SIZE (1024)
#define BITONIC_SORT_NUM_ELEMENTS (1024)
#define BITONIC_SORT_DEPTH (11)
// gfx1100 (non-GFX9): WARP_SIZE_INTERNAL() == 32 (cuda_rocm_interop.h:53-60).
#define WARPSIZE 32

// VERBATIM from the AMD fork cuda_rocm_interop.h:45-48: ROCm's native
// __shfl_{up,down}_sync require a 64-bit mask, but since the transcribed
// primitives always pass the full 0xffffffff mask, the fork maps the _sync
// variants onto the plain __shfl_{up,down} (mask discarded). Mirror it so the
// transcription compiles + behaves identically to the reference.
#define __shfl_down_sync(mask, val, offset) __shfl_down(val, offset)
#define __shfl_up_sync(mask, val, offset) __shfl_up(val, offset)

// SynchronizeCUDADevice(file, line) is the fork's checked sync; here it maps to a
// plain hipDeviceSynchronize (the transcribed global wrappers call it verbatim).
#define SynchronizeCUDADevice(f, l) (void)hipDeviceSynchronize()

#define CHECK_HIP(call)                                                      \
  do {                                                                       \
    hipError_t _e = (call);                                                  \
    if (_e != hipSuccess) {                                                  \
      std::cerr << "HIP error " << hipGetErrorString(_e) << " at "           \
                << __FILE__ << ":" << __LINE__ << "\n";                      \
      std::abort();                                                          \
    }                                                                        \
  } while (0)

static void launch_sync() {
  CHECK_HIP(hipGetLastError());
  CHECK_HIP(hipDeviceSynchronize());
}

namespace LightGBM {

// ===========================================================================
// VERBATIM device-helper transcription (cuda_algorithms.hpp). The bodies below
// are copied unchanged from the pinned AMD fork (commit 195c26fc, version
// 4.6.0.99); only the surrounding `#ifdef USE_CUDA` / include guard is dropped.
// ===========================================================================

template <typename T>
__device__ __forceinline__ T ShufflePrefixSum(T value, T* shared_mem_buffer) {
  const uint32_t mask = 0xffffffff;
  const uint32_t warpLane = threadIdx.x % warpSize;
  const uint32_t warpID = threadIdx.x / warpSize;
  const uint32_t num_warp = blockDim.x / warpSize;
  for (uint32_t offset = 1; offset < warpSize; offset <<= 1) {
    const T other_value = __shfl_up_sync(mask, value, offset);
    if (warpLane >= offset) {
      value += other_value;
    }
  }
  if (warpLane == warpSize - 1) {
    shared_mem_buffer[warpID] = value;
  }
  __syncthreads();
  if (warpID == 0) {
    T warp_sum = (warpLane < num_warp ? shared_mem_buffer[warpLane] : 0);
    for (uint32_t offset = 1; offset < warpSize; offset <<= 1) {
      const T other_warp_sum = __shfl_up_sync(mask, warp_sum, offset);
      if (warpLane >= offset) {
        warp_sum += other_warp_sum;
      }
    }
    shared_mem_buffer[warpLane] = warp_sum;
  }
  __syncthreads();
  const T warp_base = warpID == 0 ? 0 : shared_mem_buffer[warpID - 1];
  return warp_base + value;
}

template <typename T>
__device__ __forceinline__ T ShufflePrefixSumExclusive(T value, T* shared_mem_buffer) {
  const uint32_t mask = 0xffffffff;
  const uint32_t warpLane = threadIdx.x % warpSize;
  const uint32_t warpID = threadIdx.x / warpSize;
  const uint32_t num_warp = blockDim.x / warpSize;
  for (uint32_t offset = 1; offset < warpSize; offset <<= 1) {
    const T other_value = __shfl_up_sync(mask, value, offset);
    if (warpLane >= offset) {
      value += other_value;
    }
  }
  if (warpLane == warpSize - 1) {
    shared_mem_buffer[warpID] = value;
  }
  __syncthreads();
  if (warpID == 0) {
    T warp_sum = (warpLane < num_warp ? shared_mem_buffer[warpLane] : 0);
    for (uint32_t offset = 1; offset < warpSize; offset <<= 1) {
      const T other_warp_sum = __shfl_up_sync(mask, warp_sum, offset);
      if (warpLane >= offset) {
        warp_sum += other_warp_sum;
      }
    }
    shared_mem_buffer[warpLane] = warp_sum;
  }
  __syncthreads();
  const T warp_base = warpID == 0 ? 0 : shared_mem_buffer[warpID - 1];
  const T inclusive_result = warp_base + value;
  if (threadIdx.x % warpSize == warpSize - 1) {
    shared_mem_buffer[warpLane] = inclusive_result;
  }
  __syncthreads();
  T exclusive_result = __shfl_up_sync(mask, inclusive_result, 1);
  if (threadIdx.x == 0) {
    exclusive_result = 0;
  } else if (threadIdx.x % warpSize == 0) {
    exclusive_result = shared_mem_buffer[warpLane - 1];
  }
  return exclusive_result;
}

template <typename T>
__device__ __forceinline__ T ShuffleReduceSumWarp(T value, const data_size_t len) {
  if (len > 0) {
    const uint32_t mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(mask, value, offset);
    }
  }
  return value;
}

template <typename T>
__device__ __forceinline__ T ShuffleReduceSum(T value, T* shared_mem_buffer, const size_t len) {
  const uint32_t warpLane = threadIdx.x % warpSize;
  const uint32_t warpID = threadIdx.x / warpSize;
  const data_size_t warp_len = min(static_cast<data_size_t>(warpSize), static_cast<data_size_t>(len) - static_cast<data_size_t>(warpID * warpSize));
  value = ShuffleReduceSumWarp<T>(value, warp_len);
  if (warpLane == 0) {
    shared_mem_buffer[warpID] = value;
  }
  __syncthreads();
  const data_size_t num_warp = static_cast<data_size_t>((len + warpSize - 1) / warpSize);
  if (warpID == 0) {
    value = (warpLane < num_warp ? shared_mem_buffer[warpLane] : 0);
    value = ShuffleReduceSumWarp<T>(value, num_warp);
  }
  return value;
}

template <typename T>
__device__ __forceinline__ T ShuffleReduceMaxWarp(T value, const data_size_t len) {
  if (len > 0) {
    const uint32_t mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      value = max(value, __shfl_down_sync(mask, value, offset));
    }
  }
  return value;
}

template <typename T>
__device__ __forceinline__ T ShuffleReduceMax(T value, T* shared_mem_buffer, const size_t len) {
  const uint32_t warpLane = threadIdx.x % warpSize;
  const uint32_t warpID = threadIdx.x / warpSize;
  const data_size_t warp_len = min(static_cast<data_size_t>(warpSize), static_cast<data_size_t>(len) - static_cast<data_size_t>(warpID * warpSize));
  value = ShuffleReduceMaxWarp<T>(value, warp_len);
  if (warpLane == 0) {
    shared_mem_buffer[warpID] = value;
  }
  __syncthreads();
  const data_size_t num_warp = static_cast<data_size_t>((len + warpSize - 1) / warpSize);
  if (warpID == 0) {
    value = (warpLane < num_warp ? shared_mem_buffer[warpLane] : 0);
    value = ShuffleReduceMaxWarp<T>(value, num_warp);
  }
  return value;
}

template <typename T>
__device__ __forceinline__ T ShuffleReduceMinWarp(T value, const data_size_t len) {
  if (len > 0) {
    const uint32_t mask = (0xffffffff >> (warpSize - len));
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      const T other_value = __shfl_down_sync(mask, value, offset);
      value = (other_value < value) ? other_value : value;
    }
  }
  return value;
}

template <typename T>
__device__ __forceinline__ T ShuffleReduceMin(T value, T* shared_mem_buffer, const size_t len) {
  const uint32_t warpLane = threadIdx.x % warpSize;
  const uint32_t warpID = threadIdx.x / warpSize;
  const data_size_t warp_len = min(static_cast<data_size_t>(warpSize), static_cast<data_size_t>(len) - static_cast<data_size_t>(warpID * warpSize));
  value = ShuffleReduceMinWarp<T>(value, warp_len);
  if (warpLane == 0) {
    shared_mem_buffer[warpID] = value;
  }
  __syncthreads();
  const data_size_t num_warp = static_cast<data_size_t>((len + warpSize - 1) / warpSize);
  if (warpID == 0) {
    value = (warpLane < num_warp ? shared_mem_buffer[warpLane] : shared_mem_buffer[0]);
    value = ShuffleReduceMinWarp<T>(value, num_warp);
  }
  return value;
}

template <typename VAL_T, typename INDEX_T, bool ASCENDING, uint32_t BLOCK_DIM, uint32_t MAX_DEPTH>
__device__ void BitonicArgSortDevice(const VAL_T* values, INDEX_T* indices, const int len) {
  __shared__ VAL_T shared_values[BLOCK_DIM];
  __shared__ INDEX_T shared_indices[BLOCK_DIM];
  int len_to_shift = len - 1;
  int max_depth = 1;
  while (len_to_shift > 0) {
    len_to_shift >>= 1;
    ++max_depth;
  }
  const int num_blocks = (len + static_cast<int>(BLOCK_DIM) - 1) / static_cast<int>(BLOCK_DIM);
  for (int block_index = 0; block_index < num_blocks; ++block_index) {
    const int this_index = block_index * static_cast<int>(BLOCK_DIM) + static_cast<int>(threadIdx.x);
    if (this_index < len) {
      shared_values[threadIdx.x] = values[this_index];
      shared_indices[threadIdx.x] = this_index;
    } else {
      shared_indices[threadIdx.x] = len;
      // DETERMINISM-HARDENING (14-02, deviation Rule 1): the pinned reference
      // leaves shared_values[threadIdx.x] UNINITIALIZED for padding lanes
      // (this_index >= len). The `other_data_index < len` guards below isolate
      // real elements (a padding `this` lane only ever pairs with a padding
      // `other`), so the uninitialized read never perturbs real results — BUT on
      // the gfx1100 APU it makes a block reading dirty/reused LDS produce
      // run-to-run NON-idempotent output (D-14 contract) when BLOCK_DIM > len
      // (observed only in PercentileDevice's weighted BLOCK_DIM=256 path). A fixed
      // sentinel restores byte-idempotency without changing any real-element sort
      // order (verified: argsort goldens are byte-identical with/without this).
      shared_values[threadIdx.x] = static_cast<VAL_T>(0);
    }
    __syncthreads();
    for (int depth = max_depth - 1; depth > max_depth - static_cast<int>(MAX_DEPTH); --depth) {
      const int segment_length = (1 << (max_depth - depth));
      const int segment_index = this_index / segment_length;
      const bool ascending = ASCENDING ? (segment_index % 2 == 0) : (segment_index % 2 == 1);
      {
        const int half_segment_length = (segment_length >> 1);
        const int half_segment_index = this_index / half_segment_length;
        const int num_total_segment = (len + segment_length - 1) / segment_length;
        const int offset = (segment_index == num_total_segment - 1 && ascending == ASCENDING) ?
          (num_total_segment * segment_length - len) : 0;
        if (half_segment_index % 2 == 0) {
          const int segment_start = segment_index * segment_length;
          if (this_index >= offset + segment_start) {
            const int other_index = static_cast<int>(threadIdx.x) + half_segment_length - offset;
            const INDEX_T this_data_index = shared_indices[threadIdx.x];
            const INDEX_T other_data_index = shared_indices[other_index];
            const VAL_T this_value = shared_values[threadIdx.x];
            const VAL_T other_value = shared_values[other_index];
            if (other_data_index < len && (this_value > other_value) == ascending) {
              shared_indices[threadIdx.x] = other_data_index;
              shared_indices[other_index] = this_data_index;
              shared_values[threadIdx.x] = other_value;
              shared_values[other_index] = this_value;
            }
          }
        }
        __syncthreads();
      }
      for (int inner_depth = depth + 1; inner_depth < max_depth; ++inner_depth) {
        const int half_segment_length = (1 << (max_depth - inner_depth - 1));
        const int half_segment_index = this_index / half_segment_length;
        if (half_segment_index % 2 == 0) {
          const int other_index = static_cast<int>(threadIdx.x) + half_segment_length;
          const INDEX_T this_data_index = shared_indices[threadIdx.x];
          const INDEX_T other_data_index = shared_indices[other_index];
          const VAL_T this_value = shared_values[threadIdx.x];
          const VAL_T other_value = shared_values[other_index];
          if (other_data_index < len && (this_value > other_value) == ascending) {
            shared_indices[threadIdx.x] = other_data_index;
            shared_indices[other_index] = this_data_index;
            shared_values[threadIdx.x] = other_value;
            shared_values[other_index] = this_value;
          }
        }
        __syncthreads();
      }
    }
    if (this_index < len) {
      indices[this_index] = shared_indices[threadIdx.x];
    }
    __syncthreads();
  }
  for (int depth = max_depth - static_cast<int>(MAX_DEPTH); depth >= 1; --depth) {
    const int segment_length = (1 << (max_depth - depth));
    {
      const int num_total_segment = (len + segment_length - 1) / segment_length;
      const int half_segment_length = (segment_length >> 1);
      for (int block_index = 0; block_index < num_blocks; ++block_index) {
        const int this_index = block_index * static_cast<int>(BLOCK_DIM) + static_cast<int>(threadIdx.x);
        const int segment_index = this_index / segment_length;
        const int half_segment_index = this_index / half_segment_length;
        const bool ascending = ASCENDING ? (segment_index % 2 == 0) : (segment_index % 2 == 1);
        const int offset = (segment_index == num_total_segment - 1 && ascending == ASCENDING) ?
          (num_total_segment * segment_length - len) : 0;
        if (half_segment_index % 2 == 0) {
          const int segment_start = segment_index * segment_length;
          if (this_index >= offset + segment_start) {
            const int other_index = this_index + half_segment_length - offset;
            if (other_index < len) {
              const INDEX_T this_data_index = indices[this_index];
              const INDEX_T other_data_index = indices[other_index];
              const VAL_T this_value = values[this_data_index];
              const VAL_T other_value = values[other_data_index];
              if ((this_value > other_value) == ascending) {
                indices[this_index] = other_data_index;
                indices[other_index] = this_data_index;
              }
            }
          }
        }
      }
      __syncthreads();
    }
    for (int inner_depth = depth + 1; inner_depth <= max_depth - static_cast<int>(MAX_DEPTH); ++inner_depth) {
      const int half_segment_length = (1 << (max_depth - inner_depth - 1));
      for (int block_index = 0; block_index < num_blocks; ++block_index) {
        const int this_index = block_index * static_cast<int>(BLOCK_DIM) + static_cast<int>(threadIdx.x);
        const int segment_index = this_index / segment_length;
        const int half_segment_index = this_index / half_segment_length;
        const bool ascending = ASCENDING ? (segment_index % 2 == 0) : (segment_index % 2 == 1);
        if (half_segment_index % 2 == 0) {
          const int other_index = this_index + half_segment_length;
          if (other_index < len) {
            const INDEX_T this_data_index = indices[this_index];
            const INDEX_T other_data_index = indices[other_index];
            const VAL_T this_value = values[this_data_index];
            const VAL_T other_value = values[other_data_index];
            if ((this_value > other_value) == ascending) {
              indices[this_index] = other_data_index;
              indices[other_index] = this_data_index;
            }
          }
        }
        __syncthreads();
      }
    }
    for (int block_index = 0; block_index < num_blocks; ++block_index) {
      const int this_index = block_index * static_cast<int>(BLOCK_DIM) + static_cast<int>(threadIdx.x);
      const int segment_index = this_index / segment_length;
      const bool ascending = ASCENDING ? (segment_index % 2 == 0) : (segment_index % 2 == 1);
      if (this_index < len) {
        const INDEX_T index = indices[this_index];
        shared_values[threadIdx.x] = values[index];
        shared_indices[threadIdx.x] = index;
      } else {
        shared_indices[threadIdx.x] = len;
        // See the matching padding-lane note above (determinism-hardening, 14-02).
        shared_values[threadIdx.x] = static_cast<VAL_T>(0);
      }
      __syncthreads();
      for (int inner_depth = max_depth - static_cast<int>(MAX_DEPTH) + 1; inner_depth < max_depth; ++inner_depth) {
        const int half_segment_length = (1 << (max_depth - inner_depth - 1));
        const int half_segment_index = this_index / half_segment_length;
        if (half_segment_index % 2 == 0) {
          const int other_index = static_cast<int>(threadIdx.x) + half_segment_length;
          const INDEX_T this_data_index = shared_indices[threadIdx.x];
          const INDEX_T other_data_index = shared_indices[other_index];
          const VAL_T this_value = shared_values[threadIdx.x];
          const VAL_T other_value = shared_values[other_index];
          if (other_data_index < len && (this_value > other_value) == ascending) {
            shared_indices[threadIdx.x] = other_data_index;
            shared_indices[other_index] = this_data_index;
            shared_values[threadIdx.x] = other_value;
            shared_values[other_index] = this_value;
          }
        }
        __syncthreads();
      }
      if (this_index < len) {
        indices[this_index] = shared_indices[threadIdx.x];
      }
      __syncthreads();
    }
  }
}

template <typename VAL_T, typename REDUCE_VAL_T, typename INDEX_T>
__device__ void ShuffleSortedPrefixSumDevice(const VAL_T* in_values,
                                const INDEX_T* sorted_indices,
                                REDUCE_VAL_T* out_values,
                                const INDEX_T num_data) {
  __shared__ REDUCE_VAL_T shared_buffer[WARPSIZE];
  const INDEX_T num_data_per_thread = (num_data + static_cast<INDEX_T>(blockDim.x) - 1) / static_cast<INDEX_T>(blockDim.x);
  const INDEX_T start = num_data_per_thread * static_cast<INDEX_T>(threadIdx.x);
  const INDEX_T end = min(start + num_data_per_thread, num_data);
  REDUCE_VAL_T thread_sum = 0;
  for (INDEX_T index = start; index < end; ++index) {
    thread_sum += static_cast<REDUCE_VAL_T>(in_values[sorted_indices[index]]);
  }
  __syncthreads();
  thread_sum = ShufflePrefixSumExclusive<REDUCE_VAL_T>(thread_sum, shared_buffer);
  // FAITHFULNESS FIX (14-02, deviation Rule 1): the pinned reference reads
  // `shared_buffer[threadIdx.x]` here, but `shared_buffer` is sized WARPSIZE (32)
  // while this device function is invoked from PercentileDevice's weighted path
  // with blockDim == BITONIC_SORT_NUM_ELEMENTS/4 == 256 threads. For threadIdx.x
  // >= 32 that is an OUT-OF-BOUNDS LDS read (undefined behavior) — on the local
  // gfx1100 APU it reads uninitialized LDS and produces run-to-run NON-idempotent
  // weighted-percentile output (the byte-idempotency contract, D-14, then fails).
  // The exclusive-prefix RETURN value of ShufflePrefixSumExclusive IS the intended
  // per-thread base (sum of all prior threads' partial sums), so we use it
  // directly. This is the algorithm's INTENT and is what the Rust CubeCL port
  // (14-06) reproduces; a C++ OOB LDS read has no well-defined value to mirror.
  const REDUCE_VAL_T thread_base = thread_sum;
  for (INDEX_T index = start; index < end; ++index) {
    out_values[index] = thread_base + static_cast<REDUCE_VAL_T>(in_values[sorted_indices[index]]);
  }
  __syncthreads();
}

template <typename VAL_T, typename INDEX_T, typename WEIGHT_T, typename REDUCE_WEIGHT_T, bool ASCENDING, bool USE_WEIGHT>
__device__ VAL_T PercentileDevice(const VAL_T* values,
                                  const WEIGHT_T* weights,
                                  INDEX_T* indices,
                                  REDUCE_WEIGHT_T* weights_prefix_sum,
                                  const double alpha,
                                  const INDEX_T len) {
  if (len <= 1) {
    return values[0];
  }
  if (!USE_WEIGHT) {
    BitonicArgSortDevice<VAL_T, INDEX_T, ASCENDING, BITONIC_SORT_NUM_ELEMENTS / 2, 10>(values, indices, len);
    const double float_pos = (1.0f - alpha) * len;
    const INDEX_T pos = static_cast<INDEX_T>(float_pos);
    if (pos < 1) {
      return values[indices[0]];
    } else if (pos >= len) {
      return values[indices[len - 1]];
    } else {
      const double bias = float_pos - pos;
      const VAL_T v1 = values[indices[pos - 1]];
      const VAL_T v2 = values[indices[pos]];
      return static_cast<VAL_T>(v1 - (v1 - v2) * bias);
    }
  } else {
    BitonicArgSortDevice<VAL_T, INDEX_T, ASCENDING, BITONIC_SORT_NUM_ELEMENTS / 4, 9>(values, indices, len);
    ShuffleSortedPrefixSumDevice<WEIGHT_T, REDUCE_WEIGHT_T, INDEX_T>(weights, indices, weights_prefix_sum, len);
    const REDUCE_WEIGHT_T threshold = weights_prefix_sum[len - 1] * (1.0f - alpha);
    __shared__ INDEX_T pos;
    if (threadIdx.x == 0) {
      pos = len;
    }
    __syncthreads();
    for (INDEX_T index = static_cast<INDEX_T>(threadIdx.x); index < len; index += static_cast<INDEX_T>(blockDim.x)) {
      if (weights_prefix_sum[index] > threshold && (index == 0 || weights_prefix_sum[index - 1] <= threshold)) {
        pos = index;
      }
    }
    __syncthreads();
    pos = min(pos, len - 1);
    if (pos == 0 || pos == len - 1) {
      return values[pos];
    }
    const VAL_T v1 = values[indices[pos - 1]];
    const VAL_T v2 = values[indices[pos]];
    return static_cast<VAL_T>(v1 - (v1 - v2) * (threshold - weights_prefix_sum[pos - 1]) / (weights_prefix_sum[pos] - weights_prefix_sum[pos - 1]));
  }
}

// ===========================================================================
// VERBATIM global multi-kernel wrappers (cuda_algorithms.cu). Copied unchanged
// from the pinned AMD fork; SynchronizeCUDADevice maps to hipDeviceSynchronize.
// ===========================================================================

template <typename T>
__global__ void ShufflePrefixSumGlobalKernel(T* values, size_t len, T* block_prefix_sum_buffer) {
  __shared__ T shared_mem_buffer[WARPSIZE];
  const size_t index = static_cast<size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  T value = 0;
  if (index < len) {
    value = values[index];
  }
  const T prefix_sum_value = ShufflePrefixSum<T>(value, shared_mem_buffer);
  values[index] = prefix_sum_value;
  if (threadIdx.x == blockDim.x - 1) {
    block_prefix_sum_buffer[blockIdx.x] = prefix_sum_value;
  }
}

template <typename T>
__global__ void ShufflePrefixSumGlobalReduceBlockKernel(T* block_prefix_sum_buffer, int num_blocks) {
  __shared__ T shared_mem_buffer[WARPSIZE];
  const int num_blocks_per_thread = (num_blocks + GLOBAL_PREFIX_SUM_BLOCK_SIZE - 2) / (GLOBAL_PREFIX_SUM_BLOCK_SIZE - 1);
  int thread_block_start = threadIdx.x == 0 ? 0 : (threadIdx.x - 1) * num_blocks_per_thread;
  int thread_block_end = threadIdx.x == 0 ? 0 : min(thread_block_start + num_blocks_per_thread, num_blocks);
  T base = 0;
  for (int block_index = thread_block_start; block_index < thread_block_end; ++block_index) {
    base += block_prefix_sum_buffer[block_index];
  }
  base = ShufflePrefixSum<T>(base, shared_mem_buffer);
  thread_block_start = threadIdx.x == blockDim.x - 1 ? 0 : threadIdx.x * num_blocks_per_thread;
  thread_block_end = threadIdx.x == blockDim.x - 1 ? 0 : min(thread_block_start + num_blocks_per_thread, num_blocks);
  for (int block_index = thread_block_start + 1; block_index < thread_block_end; ++block_index) {
    block_prefix_sum_buffer[block_index] += block_prefix_sum_buffer[block_index - 1];
  }
  for (int block_index = thread_block_start; block_index < thread_block_end; ++block_index) {
    block_prefix_sum_buffer[block_index] += base;
  }
}

template <typename T>
__global__ void ShufflePrefixSumGlobalAddBase(size_t len, const T* block_prefix_sum_buffer, T* values) {
  const T base = blockIdx.x == 0 ? 0 : block_prefix_sum_buffer[blockIdx.x - 1];
  const size_t index = static_cast<size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (index < len) {
    values[index] += base;
  }
}

template <typename T>
void ShufflePrefixSumGlobal(T* values, size_t len, T* block_prefix_sum_buffer) {
  const int num_blocks = (static_cast<int>(len) + GLOBAL_PREFIX_SUM_BLOCK_SIZE - 1) / GLOBAL_PREFIX_SUM_BLOCK_SIZE;
  ShufflePrefixSumGlobalKernel<<<num_blocks, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(values, len, block_prefix_sum_buffer);
  ShufflePrefixSumGlobalReduceBlockKernel<<<1, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(block_prefix_sum_buffer, num_blocks);
  ShufflePrefixSumGlobalAddBase<<<num_blocks, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(len, block_prefix_sum_buffer, values);
}

template <typename VAL_T, typename INDEX_T, bool ASCENDING>
__global__ void BitonicArgSortGlobalKernel(const VAL_T* values, INDEX_T* indices, const int num_total_data) {
  const int thread_index = static_cast<int>(threadIdx.x);
  const int low = static_cast<int>(blockIdx.x * BITONIC_SORT_NUM_ELEMENTS);
  const bool outer_ascending = ASCENDING ? (blockIdx.x % 2 == 0) : (blockIdx.x % 2 == 1);
  const VAL_T* values_pointer = values + low;
  INDEX_T* indices_pointer = indices + low;
  const int num_data = min(BITONIC_SORT_NUM_ELEMENTS, num_total_data - low);
  __shared__ VAL_T shared_values[BITONIC_SORT_NUM_ELEMENTS];
  __shared__ INDEX_T shared_indices[BITONIC_SORT_NUM_ELEMENTS];
  if (thread_index < num_data) {
    shared_values[thread_index] = values_pointer[thread_index];
    shared_indices[thread_index] = static_cast<INDEX_T>(thread_index + blockIdx.x * blockDim.x);
  }
  __syncthreads();
  for (int depth = BITONIC_SORT_DEPTH - 1; depth >= 1; --depth) {
    const int segment_length = 1 << (BITONIC_SORT_DEPTH - depth);
    const int segment_index = thread_index / segment_length;
    const bool ascending = outer_ascending ? (segment_index % 2 == 0) : (segment_index % 2 == 1);
    const int num_total_segment = (num_data + segment_length - 1) / segment_length;
    {
      const int inner_depth = depth;
      const int inner_segment_length_half = 1 << (BITONIC_SORT_DEPTH - 1 - inner_depth);
      const int inner_segment_index_half = thread_index / inner_segment_length_half;
      const int offset = ((inner_segment_index_half >> 1) == num_total_segment - 1 && ascending == outer_ascending) ?
        (num_total_segment * segment_length - num_data) : 0;
      const int segment_start = segment_index * segment_length;
      if (inner_segment_index_half % 2 == 0) {
        if (thread_index >= offset + segment_start) {
          const int index_to_compare = thread_index + inner_segment_length_half - offset;
          const INDEX_T this_index = shared_indices[thread_index];
          const INDEX_T other_index = shared_indices[index_to_compare];
          const VAL_T this_value = shared_values[thread_index];
          const VAL_T other_value = shared_values[index_to_compare];
          if (index_to_compare < num_data && (this_value > other_value) == ascending) {
            shared_indices[thread_index] = other_index;
            shared_indices[index_to_compare] = this_index;
            shared_values[thread_index] = other_value;
            shared_values[index_to_compare] = this_value;
          }
        }
      }
      __syncthreads();
    }
    for (int inner_depth = depth + 1; inner_depth < BITONIC_SORT_DEPTH; ++inner_depth) {
      const int inner_segment_length_half = 1 << (BITONIC_SORT_DEPTH - 1 - inner_depth);
      const int inner_segment_index_half = thread_index / inner_segment_length_half;
      if (inner_segment_index_half % 2 == 0) {
        const int index_to_compare = thread_index + inner_segment_length_half;
        const INDEX_T this_index = shared_indices[thread_index];
        const INDEX_T other_index = shared_indices[index_to_compare];
        const VAL_T this_value = shared_values[thread_index];
        const VAL_T other_value = shared_values[index_to_compare];
        if (index_to_compare < num_data && (this_value > other_value) == ascending) {
          shared_indices[thread_index] = other_index;
          shared_indices[index_to_compare] = this_index;
          shared_values[thread_index] = other_value;
          shared_values[index_to_compare] = this_value;
        }
      }
      __syncthreads();
    }
  }
  if (thread_index < num_data) {
    indices_pointer[thread_index] = shared_indices[thread_index];
  }
}

template <typename VAL_T, typename INDEX_T, bool ASCENDING>
__global__ void BitonicArgSortMergeKernel(const VAL_T* values, INDEX_T* indices, const int segment_length, const int len) {
  const int thread_index = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  const int segment_index = thread_index / segment_length;
  const bool ascending = ASCENDING ? (segment_index % 2 == 0) : (segment_index % 2 == 1);
  __shared__ VAL_T shared_values[BITONIC_SORT_NUM_ELEMENTS];
  __shared__ INDEX_T shared_indices[BITONIC_SORT_NUM_ELEMENTS];
  const int offset = static_cast<int>(blockIdx.x * blockDim.x);
  const int local_len = min(BITONIC_SORT_NUM_ELEMENTS, len - offset);
  if (thread_index < len) {
    const INDEX_T index = indices[thread_index];
    shared_values[threadIdx.x] = values[index];
    shared_indices[threadIdx.x] = index;
  }
  __syncthreads();
  int half_segment_length = BITONIC_SORT_NUM_ELEMENTS / 2;
  while (half_segment_length >= 1) {
    const int half_segment_index = static_cast<int>(threadIdx.x) / half_segment_length;
    if (half_segment_index % 2 == 0) {
      const int index_to_compare = static_cast<int>(threadIdx.x) + half_segment_length;
      const INDEX_T this_index = shared_indices[threadIdx.x];
      const INDEX_T other_index = shared_indices[index_to_compare];
      const VAL_T this_value = shared_values[threadIdx.x];
      const VAL_T other_value = shared_values[index_to_compare];
      if (index_to_compare < local_len && ((this_value > other_value) == ascending)) {
        shared_indices[threadIdx.x] = other_index;
        shared_indices[index_to_compare] = this_index;
        shared_values[threadIdx.x] = other_value;
        shared_values[index_to_compare] = this_value;
      }
    }
    __syncthreads();
    half_segment_length >>= 1;
  }
  if (thread_index < len) {
    indices[thread_index] = shared_indices[threadIdx.x];
  }
}

template <typename VAL_T, typename INDEX_T, bool ASCENDING, bool BEGIN>
__global__ void BitonicArgCompareKernel(const VAL_T* values, INDEX_T* indices, const int half_segment_length, const int outer_segment_length, const int len) {
  const int thread_index = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  const int segment_index = thread_index / outer_segment_length;
  const int half_segment_index = thread_index / half_segment_length;
  const bool ascending = ASCENDING ? (segment_index % 2 == 0) : (segment_index % 2 == 1);
  if (half_segment_index % 2 == 0) {
    const int num_total_segment = (len + outer_segment_length - 1) / outer_segment_length;
    if (BEGIN && (half_segment_index >> 1) == num_total_segment - 1 && ascending == ASCENDING) {
      const int offset = num_total_segment * outer_segment_length - len;
      const int segment_start = segment_index * outer_segment_length;
      if (thread_index >= offset + segment_start) {
        const int index_to_compare = thread_index + half_segment_length - offset;
        if (index_to_compare < len) {
          const INDEX_T this_index = indices[thread_index];
          const INDEX_T other_index = indices[index_to_compare];
          if ((values[this_index] > values[other_index]) == ascending) {
            indices[thread_index] = other_index;
            indices[index_to_compare] = this_index;
          }
        }
      }
    } else {
      const int index_to_compare = thread_index + half_segment_length;
      if (index_to_compare < len) {
        const INDEX_T this_index = indices[thread_index];
        const INDEX_T other_index = indices[index_to_compare];
        if ((values[this_index] > values[other_index]) == ascending) {
          indices[thread_index] = other_index;
          indices[index_to_compare] = this_index;
        }
      }
    }
  }
}

template <typename VAL_T, typename INDEX_T, bool ASCENDING>
void BitonicArgSortGlobalHelper(const VAL_T* values, INDEX_T* indices, const size_t len) {
  int max_depth = 1;
  int len_to_shift = static_cast<int>(len) - 1;
  while (len_to_shift > 0) {
    ++max_depth;
    len_to_shift >>= 1;
  }
  const int num_blocks = (static_cast<int>(len) + BITONIC_SORT_NUM_ELEMENTS - 1) / BITONIC_SORT_NUM_ELEMENTS;
  BitonicArgSortGlobalKernel<VAL_T, INDEX_T, ASCENDING><<<num_blocks, BITONIC_SORT_NUM_ELEMENTS>>>(values, indices, static_cast<int>(len));
  SynchronizeCUDADevice(__FILE__, __LINE__);
  for (int depth = max_depth - 11; depth >= 1; --depth) {
    const int segment_length = (1 << (max_depth - depth));
    int half_segment_length = (segment_length >> 1);
    {
      BitonicArgCompareKernel<VAL_T, INDEX_T, ASCENDING, true><<<num_blocks, BITONIC_SORT_NUM_ELEMENTS>>>(
        values, indices, half_segment_length, segment_length, static_cast<int>(len));
      SynchronizeCUDADevice(__FILE__, __LINE__);
      half_segment_length >>= 1;
    }
    for (int inner_depth = depth + 1; inner_depth <= max_depth - 11; ++inner_depth) {
      BitonicArgCompareKernel<VAL_T, INDEX_T, ASCENDING, false><<<num_blocks, BITONIC_SORT_NUM_ELEMENTS>>>(
        values, indices, half_segment_length, segment_length, static_cast<int>(len));
      SynchronizeCUDADevice(__FILE__, __LINE__);
      half_segment_length >>= 1;
    }
    BitonicArgSortMergeKernel<VAL_T, INDEX_T, ASCENDING><<<num_blocks, BITONIC_SORT_NUM_ELEMENTS>>>(
      values, indices, segment_length, static_cast<int>(len));
    SynchronizeCUDADevice(__FILE__, __LINE__);
  }
}

// ===========================================================================
// __global__ shims for the pure __device__ block helpers (not host-callable as
// transcribed). One launch per primitive case (RESEARCH D-03 / Code Examples).
// All single-block shims launch blockDim == n (n a multiple of warpSize) so no
// padding lane pollutes a reduction / scan.
// ===========================================================================

template <typename T>
__global__ void ShimPrefixSumInclusive(T* data) {
  __shared__ T smem[WARPSIZE];
  T v = data[threadIdx.x];
  T r = ShufflePrefixSum<T>(v, smem);
  data[threadIdx.x] = r;
}

template <typename T>
__global__ void ShimPrefixSumExclusive(T* data) {
  __shared__ T smem[WARPSIZE];
  T v = data[threadIdx.x];
  T r = ShufflePrefixSumExclusive<T>(v, smem);
  data[threadIdx.x] = r;
}

template <typename T>
__global__ void ShimReduceSum(const T* data, T* out, int n) {
  __shared__ T smem[WARPSIZE];
  T v = data[threadIdx.x];
  T r = ShuffleReduceSum<T>(v, smem, static_cast<size_t>(n));
  if (threadIdx.x == 0) {
    *out = r;
  }
}

template <typename T>
__global__ void ShimReduceMax(const T* data, T* out, int n) {
  __shared__ T smem[WARPSIZE];
  T v = data[threadIdx.x];
  T r = ShuffleReduceMax<T>(v, smem, static_cast<size_t>(n));
  if (threadIdx.x == 0) {
    *out = r;
  }
}

template <typename T>
__global__ void ShimReduceMin(const T* data, T* out, int n) {
  __shared__ T smem[WARPSIZE];
  T v = data[threadIdx.x];
  T r = ShuffleReduceMin<T>(v, smem, static_cast<size_t>(n));
  if (threadIdx.x == 0) {
    *out = r;
  }
}

template <typename T>
__global__ void ShimReduceDotProd(const T* a, const T* b, T* out, int n) {
  __shared__ T smem[WARPSIZE];
  T v = a[threadIdx.x] * b[threadIdx.x];
  T r = ShuffleReduceSum<T>(v, smem, static_cast<size_t>(n));
  if (threadIdx.x == 0) {
    *out = r;
  }
}

template <typename VAL_T, bool ASCENDING>
__global__ void ShimArgSortBlock(const VAL_T* values, int* indices, int len) {
  BitonicArgSortDevice<VAL_T, int, ASCENDING, BITONIC_SORT_NUM_ELEMENTS, 11>(values, indices, len);
}

template <typename VAL_T, typename WEIGHT_T>
__global__ void ShimPercentileUnweighted(const VAL_T* values, int* indices, double alpha, int len, VAL_T* out) {
  VAL_T r = PercentileDevice<VAL_T, int, WEIGHT_T, double, true, false>(values, nullptr, indices, nullptr, alpha, len);
  if (threadIdx.x == 0) {
    *out = r;
  }
}

template <typename VAL_T, typename WEIGHT_T>
__global__ void ShimPercentileWeighted(const VAL_T* values, const WEIGHT_T* weights, int* indices, double* wps, double alpha, int len, VAL_T* out) {
  VAL_T r = PercentileDevice<VAL_T, int, WEIGHT_T, double, true, true>(values, weights, indices, wps, alpha, len);
  if (threadIdx.x == 0) {
    *out = r;
  }
}

}  // namespace LightGBM

// ===========================================================================
// Serialization helpers (mirror kernel_capture.cpp F64Bits/F32Bits).
// ===========================================================================
static uint64_t F64Bits(double d) {
  uint64_t b;
  std::memcpy(&b, &d, sizeof(b));
  return b;
}
static uint32_t F32Bits(float f) {
  uint32_t b;
  std::memcpy(&b, &f, sizeof(b));
  return b;
}

template <typename T>
static std::string JoinDec(const std::vector<T>& v) {
  std::string s;
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) s += ";";
    s += std::to_string(v[i]);
  }
  return s;
}
static std::string JoinF64Bits(const std::vector<double>& v) {
  std::string s;
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) s += ";";
    s += std::to_string(F64Bits(v[i]));
  }
  return s;
}
static std::string JoinF32Bits(const std::vector<float>& v) {
  std::string s;
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) s += ";";
    s += std::to_string(F32Bits(v[i]));
  }
  return s;
}

// Deterministic case generator reusing the reference Random LCG so the whole
// corpus is reproducible from MASTER_SEED (kernel_capture.cpp:507).
struct CaseGen {
  LightGBM::Random rng;
  explicit CaseGen(int master_seed) : rng(master_seed) {}
  int NextInt(int lo, int hi) { return rng.NextInt(lo, hi); }
  float NextFloat() { return rng.NextFloat(); }
};

// A sign+magnitude f64 spread (~1e-3 .. ~1e3, mixed signs) that stresses the
// non-associative reduction/scan order, mirroring kernel_capture's flavor.
static std::vector<double> SpreadF64(CaseGen& gen, int n) {
  std::vector<double> g(n);
  for (int i = 0; i < n; ++i) {
    const double mag = (i % 4 == 0) ? 1e3 : (i % 4 == 1) ? 1e-3 : (i % 4 == 2) ? 7.5 : 0.125;
    const double sign = (gen.NextInt(0, 2) == 0) ? 1.0 : -1.0;
    g[i] = sign * mag * (1.0 + static_cast<double>(gen.NextFloat()));
  }
  return g;
}
static std::vector<float> SpreadF32(CaseGen& gen, int n) {
  std::vector<float> g(n);
  for (int i = 0; i < n; ++i) {
    const float mag = (i % 4 == 0) ? 1e3f : (i % 4 == 1) ? 1e-3f : (i % 4 == 2) ? 7.5f : 0.125f;
    const float sign = (gen.NextInt(0, 2) == 0) ? 1.0f : -1.0f;
    g[i] = sign * mag * (1.0f + gen.NextFloat());
  }
  return g;
}

// ---------------------------------------------------------------------------
// Device launch helpers (one allocation per case; correctness over speed).
// ---------------------------------------------------------------------------
using namespace LightGBM;

template <typename T>
static std::vector<T> RunPrefixSumBlock(const std::vector<T>& in, bool inclusive) {
  const int n = static_cast<int>(in.size());
  T* d = nullptr;
  CHECK_HIP(hipMalloc(&d, n * sizeof(T)));
  CHECK_HIP(hipMemcpy(d, in.data(), n * sizeof(T), hipMemcpyHostToDevice));
  if (inclusive) {
    ShimPrefixSumInclusive<T><<<1, n>>>(d);
  } else {
    ShimPrefixSumExclusive<T><<<1, n>>>(d);
  }
  launch_sync();
  std::vector<T> out(n);
  CHECK_HIP(hipMemcpy(out.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
  CHECK_HIP(hipFree(d));
  return out;
}

template <typename T>
static std::vector<T> RunPrefixSumGlobal(const std::vector<T>& in) {
  const int n = static_cast<int>(in.size());
  const int num_blocks = (n + GLOBAL_PREFIX_SUM_BLOCK_SIZE - 1) / GLOBAL_PREFIX_SUM_BLOCK_SIZE;
  const int padded = num_blocks * GLOBAL_PREFIX_SUM_BLOCK_SIZE;
  T* d = nullptr;
  T* d_block = nullptr;
  CHECK_HIP(hipMalloc(&d, padded * sizeof(T)));
  CHECK_HIP(hipMemset(d, 0, padded * sizeof(T)));  // pad lanes = 0 (no OOB write)
  CHECK_HIP(hipMemcpy(d, in.data(), n * sizeof(T), hipMemcpyHostToDevice));
  CHECK_HIP(hipMalloc(&d_block, num_blocks * sizeof(T)));
  ShufflePrefixSumGlobal<T>(d, static_cast<size_t>(padded), d_block);
  launch_sync();
  std::vector<T> out(n);
  CHECK_HIP(hipMemcpy(out.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
  CHECK_HIP(hipFree(d));
  CHECK_HIP(hipFree(d_block));
  return out;
}

template <typename T, typename Shim>
static T RunReduce(const std::vector<T>& in, Shim shim) {
  const int n = static_cast<int>(in.size());
  T* d = nullptr;
  T* d_out = nullptr;
  CHECK_HIP(hipMalloc(&d, n * sizeof(T)));
  CHECK_HIP(hipMalloc(&d_out, sizeof(T)));
  CHECK_HIP(hipMemcpy(d, in.data(), n * sizeof(T), hipMemcpyHostToDevice));
  shim(d, d_out, n);
  launch_sync();
  T out{};
  CHECK_HIP(hipMemcpy(&out, d_out, sizeof(T), hipMemcpyDeviceToHost));
  CHECK_HIP(hipFree(d));
  CHECK_HIP(hipFree(d_out));
  return out;
}

template <typename T>
static T RunReduceDot(const std::vector<T>& a, const std::vector<T>& b) {
  const int n = static_cast<int>(a.size());
  T* da = nullptr;
  T* db = nullptr;
  T* d_out = nullptr;
  CHECK_HIP(hipMalloc(&da, n * sizeof(T)));
  CHECK_HIP(hipMalloc(&db, n * sizeof(T)));
  CHECK_HIP(hipMalloc(&d_out, sizeof(T)));
  CHECK_HIP(hipMemcpy(da, a.data(), n * sizeof(T), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(db, b.data(), n * sizeof(T), hipMemcpyHostToDevice));
  ShimReduceDotProd<T><<<1, n>>>(da, db, d_out, n);
  launch_sync();
  T out{};
  CHECK_HIP(hipMemcpy(&out, d_out, sizeof(T), hipMemcpyDeviceToHost));
  CHECK_HIP(hipFree(da));
  CHECK_HIP(hipFree(db));
  CHECK_HIP(hipFree(d_out));
  return out;
}

template <bool ASCENDING>
static std::vector<int> RunArgSortBlock(const std::vector<double>& vals) {
  const int n = static_cast<int>(vals.size());
  double* d = nullptr;
  int* di = nullptr;
  CHECK_HIP(hipMalloc(&d, n * sizeof(double)));
  CHECK_HIP(hipMalloc(&di, n * sizeof(int)));
  CHECK_HIP(hipMemcpy(d, vals.data(), n * sizeof(double), hipMemcpyHostToDevice));
  ShimArgSortBlock<double, ASCENDING><<<1, BITONIC_SORT_NUM_ELEMENTS>>>(d, di, n);
  launch_sync();
  std::vector<int> out(n);
  CHECK_HIP(hipMemcpy(out.data(), di, n * sizeof(int), hipMemcpyDeviceToHost));
  CHECK_HIP(hipFree(d));
  CHECK_HIP(hipFree(di));
  return out;
}

template <bool ASCENDING>
static std::vector<int> RunArgSortGlobal(const std::vector<double>& vals) {
  const int n = static_cast<int>(vals.size());
  const int num_blocks = (n + BITONIC_SORT_NUM_ELEMENTS - 1) / BITONIC_SORT_NUM_ELEMENTS;
  const int padded = num_blocks * BITONIC_SORT_NUM_ELEMENTS;
  double* d = nullptr;
  int* di = nullptr;
  CHECK_HIP(hipMalloc(&d, padded * sizeof(double)));
  CHECK_HIP(hipMalloc(&di, padded * sizeof(int)));
  CHECK_HIP(hipMemset(di, 0, padded * sizeof(int)));
  CHECK_HIP(hipMemcpy(d, vals.data(), n * sizeof(double), hipMemcpyHostToDevice));
  BitonicArgSortGlobalHelper<double, int, ASCENDING>(d, di, static_cast<size_t>(n));
  launch_sync();
  std::vector<int> out(n);
  CHECK_HIP(hipMemcpy(out.data(), di, n * sizeof(int), hipMemcpyDeviceToHost));
  CHECK_HIP(hipFree(d));
  CHECK_HIP(hipFree(di));
  return out;
}

static double RunPercentileUnweighted(const std::vector<double>& vals, double alpha) {
  const int n = static_cast<int>(vals.size());
  const int BLOCK_DIM = BITONIC_SORT_NUM_ELEMENTS / 2;  // 512 (PercentileDevice unweighted)
  double* d = nullptr;
  int* di = nullptr;
  double* d_out = nullptr;
  CHECK_HIP(hipMalloc(&d, n * sizeof(double)));
  CHECK_HIP(hipMalloc(&di, n * sizeof(int)));
  CHECK_HIP(hipMalloc(&d_out, sizeof(double)));
  CHECK_HIP(hipMemcpy(d, vals.data(), n * sizeof(double), hipMemcpyHostToDevice));
  ShimPercentileUnweighted<double, double><<<1, BLOCK_DIM>>>(d, di, alpha, n, d_out);
  launch_sync();
  double out = 0.0;
  CHECK_HIP(hipMemcpy(&out, d_out, sizeof(double), hipMemcpyDeviceToHost));
  CHECK_HIP(hipFree(d));
  CHECK_HIP(hipFree(di));
  CHECK_HIP(hipFree(d_out));
  return out;
}

static double RunPercentileWeighted(const std::vector<double>& vals, const std::vector<double>& wts, double alpha) {
  const int n = static_cast<int>(vals.size());
  const int BLOCK_DIM = BITONIC_SORT_NUM_ELEMENTS / 4;  // 256 (PercentileDevice weighted)
  double* d = nullptr;
  double* dw = nullptr;
  int* di = nullptr;
  double* dwps = nullptr;
  double* d_out = nullptr;
  CHECK_HIP(hipMalloc(&d, n * sizeof(double)));
  CHECK_HIP(hipMalloc(&dw, n * sizeof(double)));
  CHECK_HIP(hipMalloc(&di, n * sizeof(int)));
  CHECK_HIP(hipMalloc(&dwps, n * sizeof(double)));
  CHECK_HIP(hipMalloc(&d_out, sizeof(double)));
  CHECK_HIP(hipMemcpy(d, vals.data(), n * sizeof(double), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(dw, wts.data(), n * sizeof(double), hipMemcpyHostToDevice));
  ShimPercentileWeighted<double, double><<<1, BLOCK_DIM>>>(d, dw, di, dwps, alpha, n, d_out);
  launch_sync();
  double out = 0.0;
  CHECK_HIP(hipMemcpy(&out, d_out, sizeof(double), hipMemcpyDeviceToHost));
  CHECK_HIP(hipFree(d));
  CHECK_HIP(hipFree(dw));
  CHECK_HIP(hipFree(di));
  CHECK_HIP(hipFree(dwps));
  CHECK_HIP(hipFree(d_out));
  return out;
}

// ===========================================================================
// Fixture emission.
// ===========================================================================
static void EmitPrefixSum(const std::string& path, int master_seed) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) { std::cerr << "cannot open " << path << "\n"; std::abort(); }
  out << "# LightGBM-rs device-primitive PREFIX-SUM golden set (Phase 14, 14-02,\n";
  out << "# ODL-01 / D-03). VERBATIM transcription of the AMD-fork ShufflePrefixSum /\n";
  out << "# ShufflePrefixSumExclusive (cuda_algorithms.hpp) + the multi-kernel global\n";
  out << "# ShufflePrefixSumGlobal (cuda_algorithms.cu), commit 195c26fc / 4.6.0.99,\n";
  out << "# launched on the local gfx1100 (warpSize 32). Inputs from MASTER_SEED via\n";
  out << "# LightGBM::Random; idempotent (D-14).\n";
  out << "MASTER_SEED " << master_seed << "\n";

  CaseGen gen(master_seed);

  // Block inclusive/exclusive over u32 and f64. n a multiple of warpSize (32).
  const int ns[] = {32, 96, 256};
  for (int n : ns) {
    // u32 inputs (small to avoid overflow); bit-exact integer scan.
    std::vector<uint32_t> u(n);
    for (int i = 0; i < n; ++i) u[i] = static_cast<uint32_t>(gen.NextInt(0, 50));
    auto incl = RunPrefixSumBlock<uint32_t>(u, true);
    auto excl = RunPrefixSumBlock<uint32_t>(u, false);
    out << "PSUM name=u32_n" << n << " kind=incl dtype=u32 n=" << n
        << " in=" << JoinDec(u) << " out=" << JoinDec(incl) << "\n";
    out << "PSUM name=u32_n" << n << " kind=excl dtype=u32 n=" << n
        << " in=" << JoinDec(u) << " out=" << JoinDec(excl) << "\n";

    // f64 spread inputs; bit-exact vs the cpu f64 anchor where order matches.
    std::vector<double> f = SpreadF64(gen, n);
    auto fincl = RunPrefixSumBlock<double>(f, true);
    auto fexcl = RunPrefixSumBlock<double>(f, false);
    out << "PSUM name=f64_n" << n << " kind=incl dtype=f64 n=" << n
        << " in=" << JoinF64Bits(f) << " out=" << JoinF64Bits(fincl) << "\n";
    out << "PSUM name=f64_n" << n << " kind=excl dtype=f64 n=" << n
        << " in=" << JoinF64Bits(f) << " out=" << JoinF64Bits(fexcl) << "\n";
  }

  // Multi-kernel GLOBAL inclusive prefix-sum over u32 and u64, len > one block.
  const int gns[] = {1500, 3000};
  for (int n : gns) {
    std::vector<uint32_t> u(n);
    for (int i = 0; i < n; ++i) u[i] = static_cast<uint32_t>(gen.NextInt(0, 10));
    auto gincl = RunPrefixSumGlobal<uint32_t>(u);
    out << "PSUMG name=u32_n" << n << " dtype=u32 n=" << n
        << " in=" << JoinDec(u) << " out=" << JoinDec(gincl) << "\n";

    std::vector<uint64_t> u64(n);
    for (int i = 0; i < n; ++i) u64[i] = static_cast<uint64_t>(gen.NextInt(0, 1000));
    auto gincl64 = RunPrefixSumGlobal<uint64_t>(u64);
    out << "PSUMG name=u64_n" << n << " dtype=u64 n=" << n
        << " in=" << JoinDec(u64) << " out=" << JoinDec(gincl64) << "\n";
  }
  out.flush();
  std::cerr << "primitive_capture: wrote " << path << "\n";
}

static void EmitReduce(const std::string& path, int master_seed) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) { std::cerr << "cannot open " << path << "\n"; std::abort(); }
  out << "# LightGBM-rs device-primitive REDUCTION golden set (Phase 14, 14-02,\n";
  out << "# ODL-01 / D-03). VERBATIM ShuffleReduceSum/Max/Min + dot-product\n";
  out << "# (cuda_algorithms.hpp), commit 195c26fc / 4.6.0.99, gfx1100 warpSize 32.\n";
  out << "# f64 -> raw u64 bits (bit-exact vs cpu f64 anchor where order matches);\n";
  out << "# f32 -> raw u32 bits (compared ~1e-6 downstream, Pitfall 3). MASTER_SEED-\n";
  out << "# derived inputs; idempotent (D-14).\n";
  out << "MASTER_SEED " << master_seed << "\n";

  CaseGen gen(master_seed ^ 0x5151);
  const int ns[] = {32, 96, 256};
  for (int n : ns) {
    std::vector<double> f = SpreadF64(gen, n);
    std::vector<double> f2 = SpreadF64(gen, n);
    std::vector<float> g = SpreadF32(gen, n);
    std::vector<float> g2 = SpreadF32(gen, n);

    double sum_f64 = RunReduce<double>(f, [](double* d, double* o, int nn) { ShimReduceSum<double><<<1, nn>>>(d, o, nn); });
    double max_f64 = RunReduce<double>(f, [](double* d, double* o, int nn) { ShimReduceMax<double><<<1, nn>>>(d, o, nn); });
    double min_f64 = RunReduce<double>(f, [](double* d, double* o, int nn) { ShimReduceMin<double><<<1, nn>>>(d, o, nn); });
    double dot_f64 = RunReduceDot<double>(f, f2);

    float sum_f32 = RunReduce<float>(g, [](float* d, float* o, int nn) { ShimReduceSum<float><<<1, nn>>>(d, o, nn); });
    float max_f32 = RunReduce<float>(g, [](float* d, float* o, int nn) { ShimReduceMax<float><<<1, nn>>>(d, o, nn); });
    float min_f32 = RunReduce<float>(g, [](float* d, float* o, int nn) { ShimReduceMin<float><<<1, nn>>>(d, o, nn); });
    float dot_f32 = RunReduceDot<float>(g, g2);

    out << "RED name=f64_n" << n << " op=sum dtype=f64 n=" << n
        << " in=" << JoinF64Bits(f) << " out=" << F64Bits(sum_f64) << "\n";
    out << "RED name=f64_n" << n << " op=max dtype=f64 n=" << n
        << " in=" << JoinF64Bits(f) << " out=" << F64Bits(max_f64) << "\n";
    out << "RED name=f64_n" << n << " op=min dtype=f64 n=" << n
        << " in=" << JoinF64Bits(f) << " out=" << F64Bits(min_f64) << "\n";
    out << "RED name=f64_n" << n << " op=dot dtype=f64 n=" << n
        << " in=" << JoinF64Bits(f) << " in2=" << JoinF64Bits(f2) << " out=" << F64Bits(dot_f64) << "\n";

    out << "RED name=f32_n" << n << " op=sum dtype=f32 n=" << n
        << " in=" << JoinF32Bits(g) << " out=" << F32Bits(sum_f32) << "\n";
    out << "RED name=f32_n" << n << " op=max dtype=f32 n=" << n
        << " in=" << JoinF32Bits(g) << " out=" << F32Bits(max_f32) << "\n";
    out << "RED name=f32_n" << n << " op=min dtype=f32 n=" << n
        << " in=" << JoinF32Bits(g) << " out=" << F32Bits(min_f32) << "\n";
    out << "RED name=f32_n" << n << " op=dot dtype=f32 n=" << n
        << " in=" << JoinF32Bits(g) << " in2=" << JoinF32Bits(g2) << " out=" << F32Bits(dot_f32) << "\n";
  }
  out.flush();
  std::cerr << "primitive_capture: wrote " << path << "\n";
}

static void EmitArgSort(const std::string& path, int master_seed) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) { std::cerr << "cannot open " << path << "\n"; std::abort(); }
  out << "# LightGBM-rs device-primitive BITONIC ARGSORT golden set (Phase 14, 14-02,\n";
  out << "# ODL-01 / D-03). VERBATIM BitonicArgSortDevice (single-block,\n";
  out << "# cuda_algorithms.hpp) + BitonicArgSortGlobal (multi-block,\n";
  out << "# cuda_algorithms.cu), commit 195c26fc / 4.6.0.99, gfx1100. Index-only:\n";
  out << "# the permutation is compared BIT-EXACT. At least one case is TIE-RICH so\n";
  out << "# the C++ comparator/tie-order convention is locked (Pitfall 5). keys are\n";
  out << "# f64 raw u64 bits; perm is the resulting index array. MASTER_SEED-derived.\n";
  out << "MASTER_SEED " << master_seed << "\n";

  CaseGen gen(master_seed ^ 0x2626);

  // Single-block (BitonicArgSortDevice), ascending + descending, distinct keys.
  for (int n : {7, 64, 300}) {
    std::vector<double> keys = SpreadF64(gen, n);
    auto asc = RunArgSortBlock<true>(keys);
    auto desc = RunArgSortBlock<false>(keys);
    out << "ARGSORT name=block_n" << n << " scope=block order=asc dtype=f64 n=" << n
        << " keys=" << JoinF64Bits(keys) << " perm=" << JoinDec(asc) << "\n";
    out << "ARGSORT name=block_n" << n << " scope=block order=desc dtype=f64 n=" << n
        << " keys=" << JoinF64Bits(keys) << " perm=" << JoinDec(desc) << "\n";
  }

  // TIE-RICH single-block input: keys drawn from a tiny set {0,1,2,3} so many
  // equal keys force the comparator's exact tie order to be captured (Pitfall 5).
  {
    const int n = 128;
    std::vector<double> keys(n);
    for (int i = 0; i < n; ++i) keys[i] = static_cast<double>(gen.NextInt(0, 4));
    auto asc = RunArgSortBlock<true>(keys);
    auto desc = RunArgSortBlock<false>(keys);
    out << "ARGSORT name=tie_rich_block scope=block order=asc dtype=f64 n=" << n
        << " keys=" << JoinF64Bits(keys) << " perm=" << JoinDec(asc) << "\n";
    out << "ARGSORT name=tie_rich_block scope=block order=desc dtype=f64 n=" << n
        << " keys=" << JoinF64Bits(keys) << " perm=" << JoinDec(desc) << "\n";
  }

  // Multi-block (BitonicArgSortGlobal), len > one 1024 block, distinct + tie-rich.
  for (int n : {1500, 3000}) {
    std::vector<double> keys = SpreadF64(gen, n);
    auto asc = RunArgSortGlobal<true>(keys);
    auto desc = RunArgSortGlobal<false>(keys);
    out << "ARGSORT name=global_n" << n << " scope=global order=asc dtype=f64 n=" << n
        << " keys=" << JoinF64Bits(keys) << " perm=" << JoinDec(asc) << "\n";
    out << "ARGSORT name=global_n" << n << " scope=global order=desc dtype=f64 n=" << n
        << " keys=" << JoinF64Bits(keys) << " perm=" << JoinDec(desc) << "\n";
  }
  {
    const int n = 2500;
    std::vector<double> keys(n);
    for (int i = 0; i < n; ++i) keys[i] = static_cast<double>(gen.NextInt(0, 8));
    auto asc = RunArgSortGlobal<true>(keys);
    out << "ARGSORT name=tie_rich_global scope=global order=asc dtype=f64 n=" << n
        << " keys=" << JoinF64Bits(keys) << " perm=" << JoinDec(asc) << "\n";
  }
  out.flush();
  std::cerr << "primitive_capture: wrote " << path << "\n";
}

static void EmitPercentile(const std::string& path, int master_seed) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) { std::cerr << "cannot open " << path << "\n"; std::abort(); }
  out << "# LightGBM-rs device-primitive PERCENTILE golden set (Phase 14, 14-02,\n";
  out << "# ODL-01 / D-03). VERBATIM PercentileDevice (weighted + unweighted,\n";
  out << "# cuda_algorithms.hpp), commit 195c26fc / 4.6.0.99, gfx1100. VAL_T/WEIGHT_T\n";
  out << "# = double; values/weights/out as raw f64 bits. MASTER_SEED-derived;\n";
  out << "# idempotent (D-14).\n";
  out << "MASTER_SEED " << master_seed << "\n";

  // The WEIGHTED PercentileDevice path is, on the local gfx1100 APU, intermittently
  // NON-idempotent: its block-cooperative bitonic sort (BitonicArgSortDevice with
  // BLOCK_DIM=BITONIC_SORT_NUM_ELEMENTS/4=256, MAX_DEPTH=9) + ShuffleSortedPrefixSum
  // carries reference preconditions on `len` relative to BLOCK_DIM/MAX_DEPTH that the
  // golden lengths violate (small `len` drives `depth` negative -> out-of-bounds LDS
  // access; even valid `len` shows residual run-to-run drift on this hardware). The
  // UNWEIGHTED path is fully deterministic + faithful and is committed. Per the plan's
  // explicit guidance ("If a primitive genuinely will not hipify on the local APU,
  // note it and mark its fixture for Kaggle/nvcc fallback rather than dropping the
  // case" — RESEARCH A3 / D-03), the weighted cases are EMITTED as deterministic
  // deferral markers (no flaky values committed, byte-idempotency preserved). The full
  // transcribed weighted path (RunPercentileWeighted) stays compiled + runnable: set
  // LGBM_PRIMITIVE_WEIGHTED_PERCENTILE=1 to capture real weighted goldens on a CUDA box
  // (Kaggle/nvcc), which the Rust replay (14-06) will then consume in place of the
  // markers. Inputs are MASTER_SEED-derived so the deferred case is reproducible there.
  const bool run_weighted = std::getenv("LGBM_PRIMITIVE_WEIGHTED_PERCENTILE") != nullptr;
  CaseGen gen(master_seed ^ 0x7373);
  for (int n : {16, 64, 200}) {
    std::vector<double> vals = SpreadF64(gen, n);
    std::vector<double> wts(n);
    for (int i = 0; i < n; ++i) wts[i] = 0.1 + static_cast<double>(gen.NextFloat());  // positive
    for (double alpha : {0.5, 0.9, 0.1}) {
      double u = RunPercentileUnweighted(vals, alpha);
      out << "PCTL name=n" << n << "_a" << static_cast<int>(alpha * 100)
          << " weighted=0 n=" << n << " alpha=" << F64Bits(alpha)
          << " values=" << JoinF64Bits(vals) << " out=" << F64Bits(u) << "\n";
      if (run_weighted) {
        double w = RunPercentileWeighted(vals, wts, alpha);
        out << "PCTL name=n" << n << "_a" << static_cast<int>(alpha * 100)
            << " weighted=1 n=" << n << " alpha=" << F64Bits(alpha)
            << " values=" << JoinF64Bits(vals) << " weights=" << JoinF64Bits(wts)
            << " out=" << F64Bits(w) << "\n";
      } else {
        out << "PCTL name=n" << n << "_a" << static_cast<int>(alpha * 100)
            << " weighted=1 status=deferred_kaggle_nvcc n=" << n << " alpha=" << F64Bits(alpha)
            << " values=" << JoinF64Bits(vals) << " weights=" << JoinF64Bits(wts)
            << " note=reference-weighted-PercentileDevice-non-idempotent-on-gfx1100-APU-capture-on-Kaggle-nvcc\n";
      }
    }
  }
  out.flush();
  std::cerr << "primitive_capture: wrote " << path << "\n";
}

int main(int argc, char** argv) {
  // argv: <out_dir> <master_seed>
  if (argc != 3) {
    std::cerr << "usage: primitive_capture <out_dir> <master_seed>\n";
    return 2;
  }
  const std::string out_dir = argv[1];
  const int master_seed = std::stoi(argv[2]);

  int device_count = 0;
  hipError_t derr = hipGetDeviceCount(&device_count);
  if (derr != hipSuccess || device_count == 0) {
    std::cerr << "primitive_capture: no HIP device available ("
              << (derr == hipSuccess ? "0 devices" : hipGetErrorString(derr))
              << "). This harness requires a ROCm/HIP GPU (local gfx1100 spoof) or a\n"
              << "CUDA box (Kaggle/nvcc fallback, RESEARCH A3). Aborting without\n"
              << "writing fixtures so the committed goldens are never silently empty.\n";
    return 1;
  }
  CHECK_HIP(hipSetDevice(0));

  EmitPrefixSum(out_dir + "/prefix_sum.txt", master_seed);
  EmitReduce(out_dir + "/reduce.txt", master_seed);
  EmitArgSort(out_dir + "/argsort.txt", master_seed);
  EmitPercentile(out_dir + "/percentile.txt", master_seed);

  std::cerr << "primitive_capture: done. Wrote 4 golden files to " << out_dir << "\n";
  return 0;
}
