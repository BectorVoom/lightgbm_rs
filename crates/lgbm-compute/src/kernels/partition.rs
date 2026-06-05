//! `data_partition` cube kernel — the stable row->{left,right} routing.
//!
//! Mirrors `DataPartition::Split` (`data_partition.hpp:101-120`), whose per-row
//! left/right decision is `DenseBin::SplitInner`
//! (`LightGBM/src/io/dense_bin.hpp:314-394`, commit 195c26fc). This plan
//! transcribes the `MissingType::None` instantiation
//! `SplitInner<MISS_IS_ZERO=false, MISS_IS_NA=false, MFB_IS_ZERO=false,
//! MFB_IS_NA=false, USE_MIN_BIN=true>` — the default numeric routing (no missing
//! handling). Missing/NA routing is Phase-7+ scope.
//!
//! RESEARCH Open Q3 is RESOLVED to ONE shape: the op returns a STABLE reordered
//! index array — left rows in original relative order followed by right rows in
//! original relative order — plus a `split_point` (= the left-row count; left
//! indices occupy `[0, split_point)`, right `[split_point, len)`). The Phase-5
//! learner owns `leaf_begin_`/`leaf_count_` bookkeeping, so this op returns only
//! the partition, not the leaf-tree state.
//!
//! ## Design
//! The kernel computes a per-row routing flag (`route[i] == 1` ⇒ right/`gt`,
//! `0` ⇒ left/`lte`) faithfully from the C++ `SplitInner` body; the host then
//! does the trivial STABLE two-pass gather (all left rows first in original
//! order, then all right rows). Splitting the work this way keeps the cube kernel
//! a flat per-row map (cubecl-cpu-friendly) while the load-bearing routing
//! decision still lives in the kernel.

use cubecl::prelude::*;

use crate::error::ComputeError;
use crate::runtime::ActiveRuntime;

/// Per-row routing map (the `SplitInner` decision, `MissingType::None` path).
///
/// For each row, writes `route[i] = 1` if the row goes RIGHT (`gt_indices`),
/// `0` if LEFT (`lte_indices`), exactly mirroring `dense_bin.hpp:346-365`:
///
/// ```cpp
/// auto th = threshold + min_bin;  (--th if most_freq_bin == 0)
/// // default direction: default_indices = (most_freq_bin <= threshold) ? lte : gt
/// if (bin < minb || bin > maxb) -> default      // USE_MIN_BIN, no-missing
/// else if (bin > th)            -> gt
/// else                          -> lte
/// ```
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn data_partition_kernel(
    bins: &Array<u32>,
    route: &mut Array<u32>,
    min_bin: i32,
    max_bin: i32,
    threshold: i32,
    most_freq_bin: i32,
) {
    if UNIT_POS == 0 {
        // th = threshold + min_bin; if most_freq_bin == 0 then --th
        // (dense_bin.hpp:322-327). default_to_right = !(most_freq_bin <= threshold)
        // i.e. the default (out-of-[min,max]) rows go gt unless most_freq_bin <=
        // threshold (then they go lte) (:336-339).
        let mut th = threshold + min_bin;
        if most_freq_bin == 0 {
            th -= 1;
        }
        // default (out-of-[min,max]) rows go gt unless `most_freq_bin <=
        // threshold` (then lte) — dense_bin.hpp:336-339. Equivalent to
        // `most_freq_bin > threshold`.
        let default_to_right = most_freq_bin > threshold; // 1=gt, 0=lte
        for i in 0..bins.len() {
            let bin = bins[i] as i32;
            // USE_MIN_BIN, no-missing: out-of-[minb,maxb] -> default direction.
            let is_default = bin < min_bin || bin > max_bin;
            let gt = bin > th; // in-range: bin > th -> gt, else lte
            // route = default ? default_to_right : (bin > th)
            let go_right = select(is_default, default_to_right, gt);
            route[i] = select(go_right, 1u32, 0u32);
        }
    }
}

/// Host-side `data_partition` on the cpu reference runtime.
///
/// Returns `(reordered, split_point)`: a STABLE reordered index array (left rows
/// in original relative order, then right rows in original relative order) and
/// the left-row count. Validates `threshold` is in `[0, num_bin)` and bin
/// indices before the unsafe launch (V5, threat T-04-01).
///
/// # Errors
/// - [`ComputeError::Runtime`] if `num_bin == 0` or `threshold >= num_bin`.
#[allow(clippy::too_many_arguments)]
pub fn data_partition_cpu(
    client: &cubecl::prelude::ComputeClient<ActiveRuntime>,
    bins: &[u32],
    num_bin: u32,
    min_bin: u32,
    max_bin: u32,
    threshold: u32,
    most_freq_bin: u32,
) -> Result<(Vec<u32>, usize), ComputeError> {
    data_partition_on(client, bins, num_bin, min_bin, max_bin, threshold, most_freq_bin)
}

/// Host-side `data_partition` on ANY runtime (generic over `R: Runtime`).
///
/// `data_partition` is **f64-free** (the routing kernel reads/writes only `u32`),
/// so the SAME kernel runs bit-identically on the cubecl-cpu anchor AND the
/// cubecl-hip GPU — no f32/f64 split is needed (CMP-03/CMP-04). The cpu entry
/// [`data_partition_cpu`] delegates here; the hip path calls this directly.
///
/// # Errors
/// - [`ComputeError::Runtime`] if `num_bin == 0` or `threshold >= num_bin`.
/// - [`ComputeError::BinIndexOutOfRange`] for any `bins[i] >= num_bin`.
#[allow(clippy::too_many_arguments)]
pub fn data_partition_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    bins: &[u32],
    num_bin: u32,
    min_bin: u32,
    max_bin: u32,
    threshold: u32,
    most_freq_bin: u32,
) -> Result<(Vec<u32>, usize), ComputeError> {
    // --- V5 boundary validation (T-04-01) ---
    if num_bin == 0 {
        return Err(ComputeError::Runtime {
            detail: "data_partition: num_bin must be > 0".to_string(),
        });
    }
    if threshold >= num_bin {
        return Err(ComputeError::Runtime {
            detail: format!("data_partition: threshold {threshold} >= num_bin {num_bin}"),
        });
    }
    for (row, &b) in bins.iter().enumerate() {
        if b >= num_bin {
            return Err(ComputeError::BinIndexOutOfRange {
                row,
                bin: b,
                num_bin,
            });
        }
    }

    let n = bins.len();
    if n == 0 {
        return Ok((Vec::new(), 0));
    }

    let h_bins = client.create_from_slice(u32::as_bytes(bins));
    let zeros = vec![0u32; n];
    let h_route = client.create_from_slice(u32::as_bytes(&zeros));

    // SAFETY: `h_bins`/`h_route` each allocated for exactly `n` u32 elements and
    // outlive the launch; the kernel reads/writes only indices `0..n`. All cubecl
    // unsafe is confined here (CMP-01).
    unsafe {
        data_partition_kernel::launch(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(h_bins, n),
            ArrayArg::from_raw_parts(h_route.clone(), n),
            min_bin as i32,
            max_bin as i32,
            threshold as i32,
            most_freq_bin as i32,
        );
    }

    let bytes = client.read_one_unchecked(h_route);
    let route = u32::from_bytes(&bytes);

    // Stable two-pass gather: left rows (route==0) in original order, then right
    // rows (route==1) in original order. split_point = left-row count.
    let mut reordered: Vec<u32> = Vec::with_capacity(n);
    for (i, &r) in route.iter().enumerate().take(n) {
        if r == 0 {
            reordered.push(i as u32);
        }
    }
    let split_point = reordered.len();
    for (i, &r) in route.iter().enumerate().take(n) {
        if r != 0 {
            reordered.push(i as u32);
        }
    }

    Ok((reordered, split_point))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu_client;

    #[test]
    fn partition_basic_threshold() {
        let client = cpu_client();
        // num_bin = 8, min_bin=0, max_bin=7, threshold=3, most_freq_bin=8 (>thr,
        // so out-of-range default would go gt — but all bins are in range here).
        // bin <= 3 -> left; bin > 3 -> right. Stable order preserved per side.
        let bins = vec![1u32, 5, 3, 7, 0, 4, 2, 6];
        let (reordered, split_point) =
            data_partition_cpu(&client, &bins, 8, 0, 7, 3, 8).unwrap();
        // left rows (bin<=3) in original order: idx 0(b1),2(b3),4(b0),6(b2)
        // right rows (bin>3): idx 1(b5),3(b7),5(b4),7(b6)
        assert_eq!(split_point, 4);
        assert_eq!(reordered, vec![0, 2, 4, 6, 1, 3, 5, 7]);
    }

    #[test]
    fn partition_rejects_threshold_out_of_range() {
        let client = cpu_client();
        let err = data_partition_cpu(&client, &[0, 1, 2], 3, 0, 2, 3, 3).unwrap_err();
        assert!(matches!(err, ComputeError::Runtime { .. }));
    }

    #[test]
    fn partition_rejects_bad_bin() {
        let client = cpu_client();
        let err = data_partition_cpu(&client, &[0, 9, 1], 3, 0, 2, 1, 3).unwrap_err();
        assert!(matches!(err, ComputeError::BinIndexOutOfRange { .. }));
    }
}
