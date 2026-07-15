//! CUDA-graph capture/replay for the lightgbm_rs perf fork (server-thread driven).
//!
//! cubecl-cuda 0.10 runs its `ComputeServer` — including every `cuLaunchKernel` — on a
//! dedicated internal device thread (`DeviceHandle::submit`). CUDA stream capture is
//! THREAD-SCOPED, so capture MUST be started/stopped on that same server thread or it
//! cannot see cubecl's launches (Phase-0 proved external capture records an empty graph).
//!
//! Mechanism: the server calls [`server_poll`] at the top of every `command()` (server
//! thread). The client thread ARMS a begin/end command via atomics; `server_poll` acts on
//! it in-order with the launches:
//!   - client [`server_arm_begin`] → next `command()` runs `begin_capture` → that op and
//!     all following ops are recorded;
//!   - client issues the launch batch (recorded on the server thread);
//!   - client [`server_arm_end`] then `client.sync()` → that `command()` runs
//!     `end_capture` + instantiate and stores the exec handle.
//! Replay ([`replay`]) is an ordinary stream op (not thread-scoped) and runs fine from the
//! client thread once the context is current.
//!
//! Single-device / single-stream only. Handles are opaque `usize` so callers need not
//! depend on `cudarc`.

use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use cudarc::driver::sys;
use std::sync::Mutex;

/// The stream cubecl launches on (recorded each `command()`).
static CURRENT_STREAM: AtomicUsize = AtomicUsize::new(0);
/// cubecl's CUDA context (recorded each `command()`) — needed for graph ops off-thread.
static CURRENT_CTX: AtomicUsize = AtomicUsize::new(0);
/// Hash of the thread that last ran `command()` (diagnostic).
static LAUNCH_THREAD: AtomicU64 = AtomicU64::new(0);

/// Armed capture command for the server thread: 0 = none, 1 = begin, 2 = end.
static CAPTURE_CMD: AtomicU8 = AtomicU8::new(0);
/// True between a successful begin_capture and its end. Used to suppress cubecl's
/// periodic drop-queue flush (which does a BLOCKING fence sync that would invalidate the
/// in-progress capture).
static CAPTURING: AtomicU8 = AtomicU8::new(0);
/// The instantiated `CUgraphExec` handle id the server stored on end (valid iff READY).
static CAPTURED_HANDLE: AtomicUsize = AtomicUsize::new(0);
/// Set true once the server finished an end+instantiate and stored a handle.
static HANDLE_READY: AtomicU8 = AtomicU8::new(0);

/// Instantiated `CUgraphExec` handles (as usize), indexed by handle id.
static EXECS: Mutex<Vec<usize>> = Mutex::new(Vec::new());
/// Last error produced on the server thread (for the client to surface).
static LAST_ERR: Mutex<Option<String>> = Mutex::new(None);
/// Diagnostic recorded at the END branch (thread ids + status right before end_capture).
static END_DIAG: Mutex<Option<String>> = Mutex::new(None);
/// Diagnostic recorded at the BEGIN branch (thread + begin_capture result).
static BEGIN_DIAG: Mutex<Option<String>> = Mutex::new(None);
/// Thread hash captured at a successful begin_capture.
static BEGIN_THREAD: AtomicU64 = AtomicU64::new(0);

const CMD_NONE: u8 = 0;
const CMD_BEGIN: u8 = 1;
const CMD_END: u8 = 2;

// ------------------------- diagnostics / accessors -------------------------

/// A stable-ish u64 hash of the current thread id (diagnostic only).
pub fn thread_hash() -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    std::thread::current().id().hash(&mut h);
    h.finish()
}

/// The thread hash that last issued a cubecl command/launch (the server thread).
#[must_use]
pub fn launch_thread() -> u64 {
    LAUNCH_THREAD.load(Ordering::SeqCst)
}

/// The recorded (stream, ctx) pointers as usize (diagnostic).
#[must_use]
pub fn debug_ptrs() -> (usize, usize) {
    (CURRENT_STREAM.load(Ordering::SeqCst), CURRENT_CTX.load(Ordering::SeqCst))
}

/// The stream cubecl launches on (null until the first command runs).
#[must_use]
pub fn captured_stream() -> sys::CUstream {
    CURRENT_STREAM.load(Ordering::SeqCst) as sys::CUstream
}

fn set_err(e: String) {
    *LAST_ERR.lock().unwrap() = Some(e);
}

/// Take (and clear) the last server-thread error, if any.
pub fn take_err() -> Option<String> {
    LAST_ERR.lock().unwrap().take()
}

/// Take (and clear) the END-branch diagnostic string, if any.
pub fn take_end_diag() -> Option<String> {
    END_DIAG.lock().unwrap().take()
}

/// Take (and clear) the BEGIN-branch diagnostic string, if any.
pub fn take_begin_diag() -> Option<String> {
    BEGIN_DIAG.lock().unwrap().take()
}

// ------------------------- client-thread arming -------------------------

/// Arm "begin capture": the next server `command()` starts stream capture.
pub fn server_arm_begin() {
    HANDLE_READY.store(0, Ordering::SeqCst);
    CAPTURE_CMD.store(CMD_BEGIN, Ordering::SeqCst);
}

/// Arm "end capture": the next server `command()` (e.g. from `client.sync()`) ends
/// capture, instantiates the graph, and stores the exec handle.
pub fn server_arm_end() {
    CAPTURE_CMD.store(CMD_END, Ordering::SeqCst);
}

/// True while a capture is in progress on the server thread. cubecl's launch path checks
/// this to skip its blocking drop-queue flush (which would invalidate the capture).
#[must_use]
pub fn is_capturing_now() -> bool {
    CAPTURING.load(Ordering::SeqCst) != 0
}

/// The first checkpoint id at which the capture was observed non-ACTIVE (diagnostic).
static FIRST_BAD_CP: AtomicU8 = AtomicU8::new(0);

/// Diagnostic checkpoint: while capturing, query the stream's capture status and, if it is
/// no longer ACTIVE (i.e. something just invalidated the capture), record `n` as the first
/// bad checkpoint. Placed at successive points inside the launch path to localize the
/// invalidating call.
///
/// # Safety
/// The stream must be valid.
pub unsafe fn cp(n: u8) {
    if !is_capturing_now() {
        return;
    }
    let active = matches!(
        unsafe { cudarc::driver::result::stream::is_capturing(captured_stream()) },
        Ok(sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE)
    );
    if !active {
        let _ = FIRST_BAD_CP.compare_exchange(0, n, Ordering::SeqCst, Ordering::SeqCst);
    }
}

/// The first checkpoint id that observed a non-ACTIVE capture (0 = none seen).
#[must_use]
pub fn first_bad_cp() -> u8 {
    FIRST_BAD_CP.load(Ordering::SeqCst)
}

/// The instantiated graph handle, once the server finished the end+instantiate.
pub fn take_exec() -> Option<usize> {
    if HANDLE_READY.swap(0, Ordering::SeqCst) != 0 {
        Some(CAPTURED_HANDLE.load(Ordering::SeqCst))
    } else {
        None
    }
}

// ------------------------- server-thread poll -------------------------

/// Called at the top of `CudaServer::command()` (SERVER THREAD). Records the stream/ctx
/// and, if a capture command is armed, performs begin or end capture on this thread so it
/// is in-scope for CUDA stream capture.
///
/// # Safety
/// `stream`/`ctx` are valid for the server's lifetime; the context is current on this thread.
pub unsafe fn server_poll(stream: sys::CUstream, ctx: sys::CUcontext) {
    CURRENT_STREAM.store(stream as usize, Ordering::SeqCst);
    CURRENT_CTX.store(ctx as usize, Ordering::SeqCst);
    LAUNCH_THREAD.store(thread_hash(), Ordering::SeqCst);

    match CAPTURE_CMD.swap(CMD_NONE, Ordering::SeqCst) {
        CMD_BEGIN => {
            let r = unsafe {
                cudarc::driver::result::stream::begin_capture(
                    stream,
                    sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )
            };
            *BEGIN_DIAG.lock().unwrap() = Some(format!(
                "BEGIN observed on thread={:x} stream=0x{:x} result={:?}",
                thread_hash(),
                stream as usize,
                r
            ));
            match r.map_err(|e| format!("begin_capture: {e:?}")) {
                Ok(()) => {
                    CAPTURING.store(1, Ordering::SeqCst);
                    BEGIN_THREAD.store(thread_hash(), Ordering::SeqCst);
                }
                Err(e) => set_err(e),
            }
        }
        CMD_END => {
            // Diagnostic: is the END on the SAME thread as begin, and is the stream still
            // ACTIVE right before end_capture? (THREAD_LOCAL capture requires same-thread end.)
            let begin_t = BEGIN_THREAD.load(Ordering::SeqCst);
            let end_t = thread_hash();
            let st = unsafe {
                match cudarc::driver::result::stream::is_capturing(stream) {
                    Ok(sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE) => "active",
                    Ok(sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE) => "none",
                    Ok(sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_INVALIDATED) => "invalidated",
                    Err(e) => Box::leak(format!("err({e:?})").into_boxed_str()),
                }
            };
            *END_DIAG.lock().unwrap() = Some(format!(
                "begin_thread={begin_t:x} end_thread={end_t:x} same={} status_before_end={st}",
                begin_t == end_t
            ));
            CAPTURING.store(0, Ordering::SeqCst);
            match unsafe { end_on(stream) } {
                Ok(handle) => {
                    CAPTURED_HANDLE.store(handle, Ordering::SeqCst);
                    HANDLE_READY.store(1, Ordering::SeqCst);
                }
                Err(e) => set_err(e),
            }
        }
        _ => {}
    }
}

/// End capture on `stream` and instantiate; returns an opaque handle id. Server thread.
unsafe fn end_on(stream: sys::CUstream) -> Result<usize, String> {
    let graph = unsafe { cudarc::driver::result::stream::end_capture(stream) }
        .map_err(|e| format!("end_capture: {e:?}"))?;
    let mut exec = MaybeUninit::uninit();
    // Raw sys call so we can pass flags = 0 (the safe wrapper's enum has no 0 variant).
    unsafe { sys::cuGraphInstantiateWithFlags(exec.as_mut_ptr(), graph, 0u64) }
        .result()
        .map_err(|e| format!("instantiate: {e:?}"))?;
    let exec = unsafe { exec.assume_init() };
    let mut execs = EXECS.lock().unwrap();
    execs.push(exec as usize);
    Ok(execs.len() - 1)
}

// ------------------------- client-thread replay -------------------------

/// Make cubecl's CUDA context current on the calling thread (replay/sync run off the
/// server thread). # Safety: the recorded context must be valid.
unsafe fn ensure_ctx_current() -> Result<(), String> {
    let ctx = CURRENT_CTX.load(Ordering::SeqCst) as sys::CUcontext;
    if ctx.is_null() {
        return Err("capture: no context noted yet".into());
    }
    unsafe { cudarc::driver::result::ctx::set_current(ctx) }
        .map_err(|e| format!("ensure_ctx_current: {e:?}"))
}

/// Replay a captured graph via `cuGraphLaunch` (a single host call). Not thread-scoped,
/// so it runs from the client thread once the context is current.
///
/// # Safety
/// `handle` must come from [`take_exec`]; the stream/context must be valid.
pub unsafe fn replay(handle: usize) -> Result<(), String> {
    unsafe { ensure_ctx_current()? };
    let exec = {
        let execs = EXECS.lock().unwrap();
        *execs.get(handle).ok_or_else(|| format!("replay: bad handle {handle}"))? as sys::CUgraphExec
    };
    unsafe { cudarc::driver::result::graph::launch(exec, captured_stream()) }
        .map_err(|e| format!("replay: {e:?}"))
}

/// Block until cubecl's stream drains (raw replays bypass cubecl's sync bookkeeping).
///
/// # Safety
/// The stream/context must be valid.
pub unsafe fn device_sync() -> Result<(), String> {
    unsafe { ensure_ctx_current()? };
    unsafe { cudarc::driver::result::stream::synchronize(captured_stream()) }
        .map_err(|e| format!("device_sync: {e:?}"))
}
