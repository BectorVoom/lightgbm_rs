//! LIGHTGBM_RS FORK (perf): per-launch transient-upload arena + launch-path profiler.
//!
//! # Why
//!
//! On sm_60 (P100) cubecl lacks `grid_constants`, so EVERY kernel launch uploads its
//! info/metadata+scalars buffer through the full storage stack: pinned-pool reserve →
//! host copy → GPU-pool reserve → `cuMemcpyHtoDAsync` → handle drop → drop-queue
//! staging → periodic BLOCKING fence flush. At ~18.5k launches per 100-tree train the
//! per-launch transport tax dominates the wall clock (see
//! `docs/ondevice-cuda-perf-plan.md` §11).
//!
//! # Arena (`CUBECL_CUDA_INFO_ARENA=1`, default OFF)
//!
//! One persistent device ring (`cuMemAlloc`) + one persistent pinned host ring
//! (`cuMemHostAlloc`). Per launch: bump-allocate a 256-B-aligned slot, memcpy the info
//! bytes into the pinned slot, one `cuMemcpyHtoDAsync` device-slot ← pinned-slot, and
//! hand the kernel a raw [`GpuResource`] pointing into the ring. No pool reserve, no
//! `Bytes` staging, no drop-queue pressure, no handle churn.
//!
//! ## Safety argument
//!
//! - Device slot reuse: all traffic is on ONE stream; at ring wrap we
//!   `cuStreamSynchronize` before reusing offset 0, so any kernel still reading old
//!   slot contents has completed. Between wraps every slot is written exactly once.
//! - Pinned slot lifetime: the async H2D copy of a slot completes before the wrap
//!   synchronize; between wraps the slot is never rewritten.
//! - Multi-stream: the arena is bound to the first stream that uses it. A launch on a
//!   DIFFERENT stream falls back to the default `create_with_data` path (returns
//!   `None`) — correctness never depends on single-stream usage, only the fast path.
//! - The `binding` (`void**` cell for `cuLaunchKernel`) comes from a fixed-size slot
//!   ring inside the arena, stable for the duration of the launch call (the params
//!   array is consumed synchronously by `cuLaunchKernel`).
//!
//! # Profiler (`CUBECL_CUDA_LAUNCH_PROF=1`, default OFF)
//!
//! Cumulative counters over the launch path segments (entry→CP1 command/stream
//! resolve, CP1→CP2 count+info upload, CP2→CP3 resource resolution, CP3→CP4 kernel +
//! drop-queue flush), plus drop-queue flush count/wait time and blocking fence-wait
//! time. Dumped to stderr every [`DUMP_EVERY`] launches as a single
//! `cubecl-launch-prof:` line the bench harness can grep.

use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::compute::storage::gpu::GpuResource;

// ------------------------------- env gates -------------------------------

/// True when the transient-upload arena is enabled (`CUBECL_CUDA_INFO_ARENA=1`).
pub fn arena_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("CUBECL_CUDA_INFO_ARENA").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    })
}

/// True when the launch-path profiler is enabled (`CUBECL_CUDA_LAUNCH_PROF=1`).
pub fn prof_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("CUBECL_CUDA_LAUNCH_PROF").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    })
}

// ------------------------------- profiler -------------------------------

pub static LAUNCH_COUNT: AtomicU64 = AtomicU64::new(0);
pub static SEG_COMMAND_NS: AtomicU64 = AtomicU64::new(0);
pub static SEG_INFO_NS: AtomicU64 = AtomicU64::new(0);
pub static SEG_RESOURCE_NS: AtomicU64 = AtomicU64::new(0);
pub static SEG_KERNEL_NS: AtomicU64 = AtomicU64::new(0);
pub static DROP_FLUSH_COUNT: AtomicU64 = AtomicU64::new(0);
pub static DROP_FLUSH_NS: AtomicU64 = AtomicU64::new(0);
pub static FENCE_WAIT_COUNT: AtomicU64 = AtomicU64::new(0);
pub static FENCE_WAIT_NS: AtomicU64 = AtomicU64::new(0);
pub static CREATE_DATA_COUNT: AtomicU64 = AtomicU64::new(0);
pub static CREATE_DATA_BYTES: AtomicU64 = AtomicU64::new(0);
pub static ARENA_HIT_COUNT: AtomicU64 = AtomicU64::new(0);
pub static ARENA_WRAP_COUNT: AtomicU64 = AtomicU64::new(0);

const DUMP_EVERY: u64 = 20_000;

/// Record one launch's segment timings (ns) and emit the periodic dump line.
pub fn record_launch(command_ns: u64, info_ns: u64, resource_ns: u64, kernel_ns: u64) {
    SEG_COMMAND_NS.fetch_add(command_ns, Ordering::Relaxed);
    SEG_INFO_NS.fetch_add(info_ns, Ordering::Relaxed);
    SEG_RESOURCE_NS.fetch_add(resource_ns, Ordering::Relaxed);
    SEG_KERNEL_NS.fetch_add(kernel_ns, Ordering::Relaxed);
    let n = LAUNCH_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    if n % DUMP_EVERY == 0 {
        dump(n);
    }
}

fn dump(n: u64) {
    let ms = |a: &AtomicU64| a.load(Ordering::Relaxed) as f64 / 1e6;
    eprintln!(
        "cubecl-launch-prof: launches={} command_ms={:.1} info_ms={:.1} resource_ms={:.1} kernel_ms={:.1} \
         drop_flush(n={} wait_ms={:.1}) fence_wait(n={} ms={:.1}) create_data(n={} bytes={}) \
         arena(hits={} wraps={})",
        n,
        ms(&SEG_COMMAND_NS),
        ms(&SEG_INFO_NS),
        ms(&SEG_RESOURCE_NS),
        ms(&SEG_KERNEL_NS),
        DROP_FLUSH_COUNT.load(Ordering::Relaxed),
        ms(&DROP_FLUSH_NS),
        FENCE_WAIT_COUNT.load(Ordering::Relaxed),
        ms(&FENCE_WAIT_NS),
        CREATE_DATA_COUNT.load(Ordering::Relaxed),
        CREATE_DATA_BYTES.load(Ordering::Relaxed),
        ARENA_HIT_COUNT.load(Ordering::Relaxed),
        ARENA_WRAP_COUNT.load(Ordering::Relaxed),
    );
}

// ------------------------------- arena -------------------------------

/// Ring capacity. A 100-tree train moves ~6 MB of info bytes, so 32 MB wraps
/// (and pays its one `cuStreamSynchronize`) roughly every 5 trains.
const ARENA_BYTES: usize = 32 * 1024 * 1024;
/// Slot alignment: covers every info element type and keeps accesses coalesced.
const ARENA_ALIGN: usize = 256;
/// Stable `void**` cells for `cuLaunchKernel`; one per in-flight launch is enough,
/// sized generously.
const PTR_SLOTS: usize = 4096;

struct InfoArena {
    dev_base: u64,
    host_base: *mut u8,
    cursor: usize,
    /// The stream the ring is bound to (all copies + consuming kernels).
    stream: usize,
    /// Stable cells whose addresses are handed out as `GpuResource::binding`.
    ptr_slots: Box<[u64; PTR_SLOTS]>,
    ptr_cursor: usize,
}

// SAFETY: the raw pointers reference process-lifetime CUDA allocations; access is
// serialized by the enclosing `Mutex`.
unsafe impl Send for InfoArena {}

static ARENA: Mutex<Option<InfoArena>> = Mutex::new(None);

/// Bump-allocate + upload `data` into the arena on `stream`; returns a raw
/// [`GpuResource`] for the kernel binding, or `None` when the arena must fall back
/// (different stream, oversized payload, allocation failure).
///
/// # Safety
///
/// Caller must hold the CUDA context current on this thread and `stream` must be a
/// valid stream on that context (both guaranteed inside a `Command`).
pub unsafe fn arena_upload(stream: cudarc::driver::sys::CUstream, data: &[u8]) -> Option<GpuResource> {
    if data.is_empty() || data.len() > ARENA_BYTES / 4 {
        return None;
    }
    let mut guard = ARENA.lock().ok()?;
    let arena = match guard.as_mut() {
        Some(a) => {
            if a.stream != stream as usize {
                return None;
            }
            a
        }
        None => {
            let dev_base = unsafe { cudarc::driver::result::malloc_sync(ARENA_BYTES).ok()? };
            let host_base =
                unsafe { cudarc::driver::result::malloc_host(ARENA_BYTES, 0).ok()? } as *mut u8;
            *guard = Some(InfoArena {
                dev_base,
                host_base,
                cursor: 0,
                stream: stream as usize,
                ptr_slots: Box::new([0u64; PTR_SLOTS]),
                ptr_cursor: 0,
            });
            guard.as_mut().expect("just inserted")
        }
    };

    let len = data.len();
    let aligned = len.div_ceil(ARENA_ALIGN) * ARENA_ALIGN;
    if arena.cursor + aligned > ARENA_BYTES {
        // Ring wrap: everything previously enqueued on this stream (copies + kernels
        // reading earlier slots) must complete before offset 0 is rewritten.
        unsafe { cudarc::driver::result::stream::synchronize(stream).ok()? };
        ARENA_WRAP_COUNT.fetch_add(1, Ordering::Relaxed);
        arena.cursor = 0;
    }
    let off = arena.cursor;
    arena.cursor += aligned;

    // SAFETY: `off + len <= ARENA_BYTES`; the pinned slot is not concurrently accessed
    // (mutex-held) and not rewritten until after the next wrap synchronize.
    unsafe {
        core::ptr::copy_nonoverlapping(data.as_ptr(), arena.host_base.add(off), len);
        cudarc::driver::result::memcpy_htod_async(
            arena.dev_base + off as u64,
            core::slice::from_raw_parts(arena.host_base.add(off), len),
            stream,
        )
        .ok()?;
    }

    let dev_ptr = arena.dev_base + off as u64;
    let slot = arena.ptr_cursor;
    arena.ptr_slots[slot] = dev_ptr;
    arena.ptr_cursor = (arena.ptr_cursor + 1) % PTR_SLOTS;
    let binding = &arena.ptr_slots[slot] as *const u64 as *mut std::ffi::c_void;

    ARENA_HIT_COUNT.fetch_add(1, Ordering::Relaxed);
    Some(GpuResource::new(dev_ptr, binding, len as u64))
}
