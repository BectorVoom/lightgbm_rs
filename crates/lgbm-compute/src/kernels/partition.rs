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
use crate::BinColumn;

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
pub fn data_partition_kernel<B: Int>(
    // quick-260625-j1l (spike-029): the bin column is now NATIVE-WIDTH (u8/u16/u32),
    // read via `u32::cast_from` to a u32 INDEX — value-identical to the prior `u32`
    // monomorph (`u32::cast_from(x: u32)` is the identity cast), so the `<u32>` launch
    // is byte-for-byte the previous kernel. The narrow widths upload 4× fewer bytes
    // and read 4× less device memory. Exactly the qix histogram `<B: Int>` precedent
    // (histogram.rs:1069, `u32::cast_from(resident_bins[...])`).
    bins: &Array<B>,
    route: &mut Array<u32>,
    min_bin: i32,
    max_bin: i32,
    threshold: i32,
    most_freq_bin: i32,
) {
    // ONE unit PER ROW (`ABSOLUTE_POS`). The `SplitInner` decision is per-row
    // INDEPENDENT (`route[i] = f(bins[i])`, no cross-row carry) and integer-only,
    // so there is no order to preserve — unlike the histogram f64 fold, which must
    // stay single-owner sequential for bit-exactness. Each unit writes its OWN
    // `route[i]` (disjoint, no atomics). Previously this scanned all rows on a
    // single lane (`UNIT_POS == 0`); the parallel form is bit-identical (the host
    // gather below is unchanged) and lets the GPU use all its lanes (260609-eu9).
    let i = ABSOLUTE_POS;
    // Tail units (i >= len) stay idle: the launch rounds the unit count up to a
    // multiple of the cube dim (manual §4 Safe Indexing).
    if i < bins.len() {
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
        // quick-260625-j1l: widen the native-width bin to a u32 INDEX, then to i32 for
        // the signed compares — value-identical to the prior `bins[i] as i32` on the
        // `<u32>` monomorph (`u32::cast_from(x: u32)` is the identity).
        let bin = u32::cast_from(bins[i]) as i32;
        // USE_MIN_BIN, no-missing: out-of-[minb,maxb] -> default direction.
        let is_default = bin < min_bin || bin > max_bin;
        let gt = bin > th; // in-range: bin > th -> gt, else lte
        // route = default ? default_to_right : (bin > th)
        let go_right = select(is_default, default_to_right, gt);
        route[i] = select(go_right, 1u32, 0u32);
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

/// **Native** host `data_partition` — the production cpu-anchor path (R2).
///
/// Bit-IDENTICAL to [`data_partition_cpu`] (the one-unit-per-row
/// `data_partition_kernel` + host gather): the SAME integer `SplitInner` routing
/// decision and the SAME stable two-pass gather (left rows in original order, then
/// right), without the cubecl launch. The op is u32-only so there is no float order
/// to preserve. The cubecl path is retained for the kernel-parity / ROCm-mirror
/// tests.
///
/// # Errors
/// Same as [`data_partition_cpu`] (V5: `num_bin > 0`, `threshold < num_bin`,
/// every `bins[i] < num_bin`).
#[allow(clippy::too_many_arguments)]
pub fn data_partition_cpu_native(
    bins: &[u32],
    num_bin: u32,
    min_bin: u32,
    max_bin: u32,
    threshold: u32,
    most_freq_bin: u32,
) -> Result<(Vec<u32>, usize), ComputeError> {
    // --- V5 boundary validation (identical to data_partition_on) ---
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

    // Routing decision (dense_bin.hpp:322-365), integer-only — identical to the
    // kernel: th = threshold + min_bin (−1 if most_freq_bin == 0); out-of-[min,max]
    // rows take the default direction (`most_freq_bin > threshold` ⇒ gt).
    let min_b = min_bin as i32;
    let max_b = max_bin as i32;
    let thr = threshold as i32;
    let mut th = thr + min_b;
    if most_freq_bin == 0 {
        th -= 1;
    }
    let default_to_right = most_freq_bin as i32 > thr;
    let go_right = |b: u32| -> bool {
        let bin = b as i32;
        if bin < min_b || bin > max_b {
            default_to_right
        } else {
            bin > th
        }
    };

    // Stable two-pass gather: left rows (route==0) then right rows (route==1), each
    // in original order. split_point = left-row count.
    let mut reordered: Vec<u32> = Vec::with_capacity(n);
    for (i, &b) in bins.iter().enumerate() {
        if !go_right(b) {
            reordered.push(i as u32);
        }
    }
    let split_point = reordered.len();
    for (i, &b) in bins.iter().enumerate() {
        if go_right(b) {
            reordered.push(i as u32);
        }
    }
    Ok((reordered, split_point))
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

    // One unit per row; cube dim 256 (8 × the gfx1100 wave32), cube count covers n
    // (mirrors the parallel histogram launcher). Each unit writes its own
    // `route[idx]` (disjoint, no atomics); tail units `idx >= n` are bounds-guarded
    // idle in the kernel.
    let cube_dim = 256u32;
    let cube_count = (n as u32).div_ceil(cube_dim);

    // SAFETY: `h_bins`/`h_route` each allocated for exactly `n` u32 elements and
    // outlive the launch; the kernel bounds-checks `idx < n` and writes only
    // indices `0..n`. All cubecl unsafe is confined here (CMP-01).
    unsafe {
        // quick-260625-j1l: explicit `<u32, R>` monomorph of the now-generic kernel.
        // `u32::cast_from(x: u32)` is the identity cast, so this is byte-for-byte the
        // prior non-generic launch (existing partition unit tests stay green).
        data_partition_kernel::launch::<u32, R>(
            client,
            CubeCount::Static(cube_count, 1, 1),
            CubeDim::new_1d(cube_dim),
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

    Ok(gather_route(&route, n))
}

/// Stable two-pass gather of a per-row `route[]` into a `(reordered, split_point)`
/// partition: left rows (`route==0`) in original order, then right rows
/// (`route==1`) in original order; `split_point` = the left-row count. Shared by
/// [`data_partition_on`] and [`data_partition_native_on`] so the gather tail is
/// byte-identical across the widened and native-width upload paths.
fn gather_route(route: &[u32], n: usize) -> (Vec<u32>, usize) {
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
    (reordered, split_point)
}

/// **Native-width** host `data_partition` on ANY runtime — the spike-029 narrow
/// upload. Identical routing + stable gather to [`data_partition_on`], but uploads
/// the leaf's bins at their NATIVE [`BinColumn`] width (u8/u16/u32) instead of a
/// u32-widened buffer: a U8 column uploads `count × 1` bytes (4× fewer) and launches
/// the `::<u8>` kernel monomorph; U16 → `::<u16>`; U32 → `::<u32>`.
///
/// Returns a `(reordered, split_point)` BYTE-IDENTICAL to `data_partition_on` fed the
/// same column widened to u32 — the u8/u16/u32 kernels read the same bin value via
/// `u32::cast_from`, so the routing (and thus the gather) is value-identical. Bit-EXACT
/// by construction (partition is f64-free).
///
/// # Errors
/// Same V5 as [`data_partition_on`]: [`ComputeError::Runtime`] if `num_bin == 0` or
/// `threshold >= num_bin`; [`ComputeError::BinIndexOutOfRange`] for any
/// `bins.bin(i) >= num_bin`.
#[allow(clippy::too_many_arguments)]
pub fn data_partition_native_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    bins: &BinColumn,
    num_bin: u32,
    min_bin: u32,
    max_bin: u32,
    threshold: u32,
    most_freq_bin: u32,
) -> Result<(Vec<u32>, usize), ComputeError> {
    use cubecl::prelude::CubeElement;

    // --- V5 boundary validation (T-04-01 / T-j1l-01), reading each bin via the
    // `BinColumn` widening accessor BEFORE the unsafe create_from_slice + launch. ---
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
    let n = bins.len();
    for i in 0..n {
        let bin = bins.bin(i);
        if bin >= num_bin {
            return Err(ComputeError::BinIndexOutOfRange {
                row: i,
                bin,
                num_bin,
            });
        }
    }

    if n == 0 {
        return Ok((Vec::new(), 0));
    }

    let zeros = vec![0u32; n];
    let h_route = client.create_from_slice(u32::as_bytes(&zeros));

    let cube_dim = 256u32;
    let cube_count = (n as u32).div_ceil(cube_dim);

    // SAFETY: `h_bins`/`h_route` each allocated for exactly `n` elements and outlive
    // the launch; the kernel bounds-checks `i < n` and writes only indices `0..n`. The
    // narrow upload is `n` elements of the native width (value-faithful — the bin is an
    // index, byte-identical across widths). All cubecl unsafe is confined here (CMP-01).
    macro_rules! launch_native {
        ($w:ty, $slice:expr) => {{
            let h_bins = client.create_from_slice(<$w>::as_bytes($slice));
            unsafe {
                data_partition_kernel::launch::<$w, R>(
                    client,
                    CubeCount::Static(cube_count, 1, 1),
                    CubeDim::new_1d(cube_dim),
                    ArrayArg::from_raw_parts(h_bins, n),
                    ArrayArg::from_raw_parts(h_route.clone(), n),
                    min_bin as i32,
                    max_bin as i32,
                    threshold as i32,
                    most_freq_bin as i32,
                );
            }
        }};
    }
    match bins {
        BinColumn::U8(v) => launch_native!(u8, v),
        BinColumn::U16(v) => launch_native!(u16, v),
        BinColumn::U32(v) => launch_native!(u32, v),
    }

    let bytes = client.read_one_unchecked(h_route);
    let route = u32::from_bytes(&bytes);

    Ok(gather_route(&route, n))
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

    // quick-260625-j1l (spike-029): the native-width path must route BYTE-IDENTICALLY
    // to the u32-widened `data_partition_on` — value-identical routing across the
    // u8/u16/u32 monomorphs (`u32::cast_from`).

    #[test]
    fn partition_native_u8_matches_widened() {
        let client = cpu_client();
        let bins = vec![1u32, 5, 3, 7, 0, 4, 2, 6]; // num_bin=8 -> BinColumn::U8
        let expected = data_partition_on(&client, &bins, 8, 0, 7, 3, 8).unwrap();
        let col = BinColumn::new(bins.clone(), 8);
        assert!(matches!(col, BinColumn::U8(_)));
        let got = data_partition_native_on(&client, &col, 8, 0, 7, 3, 8).unwrap();
        assert_eq!(got, expected);
        assert_eq!(got.1, 4);
        assert_eq!(got.0, vec![0, 2, 4, 6, 1, 3, 5, 7]);
    }

    #[test]
    fn partition_native_u16_matches_widened() {
        let client = cpu_client();
        // num_bin=512 -> BinColumn::U16. Representative split params.
        let bins = vec![0u32, 300, 64, 511, 7];
        let expected = data_partition_on(&client, &bins, 512, 0, 511, 63, 0).unwrap();
        let col = BinColumn::new(bins.clone(), 512);
        assert!(matches!(col, BinColumn::U16(_)));
        let got = data_partition_native_on(&client, &col, 512, 0, 511, 63, 0).unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn partition_native_u32_matches_widened() {
        let client = cpu_client();
        // num_bin > 65536 -> BinColumn::U32.
        let bins = vec![0u32, 70_000, 100, 65_540, 9];
        let expected = data_partition_on(&client, &bins, 70_001, 0, 70_000, 50, 0).unwrap();
        let col = BinColumn::new(bins.clone(), 70_001);
        assert!(matches!(col, BinColumn::U32(_)));
        let got = data_partition_native_on(&client, &col, 70_001, 0, 70_000, 50, 0).unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn partition_native_rejects_threshold_out_of_range() {
        let client = cpu_client();
        let col = BinColumn::new(vec![0, 1, 2], 3);
        let err = data_partition_native_on(&client, &col, 3, 0, 2, 3, 3).unwrap_err();
        assert!(matches!(err, ComputeError::Runtime { .. }));
    }

    #[test]
    fn partition_native_rejects_bad_bin() {
        let client = cpu_client();
        // bin 9 >= num_bin 3; build a U32 column so the out-of-range value survives.
        let col = BinColumn::U32(vec![0, 9, 1]);
        let err = data_partition_native_on(&client, &col, 3, 0, 2, 1, 3).unwrap_err();
        assert!(matches!(err, ComputeError::BinIndexOutOfRange { .. }));
    }
}
