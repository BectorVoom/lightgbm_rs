mod base;

pub use base::*;

use crate::device::{DeviceId, DeviceService, ServerUtilitiesHandle};

#[cfg(feature = "std")]
#[allow(dead_code)]
mod channel;

#[allow(dead_code)]
mod mutex;

#[cfg(feature = "std")]
#[allow(dead_code)]
mod reentrant;

#[cfg(all(feature = "std", not(multi_threading)))]
type Inner<S> = reentrant::ReentrantMutexDeviceHandle<S>;
#[cfg(all(not(feature = "std"), not(multi_threading)))]
type Inner<S> = mutex::MutexDeviceHandle<S>;

/// LIGHTGBM_RS FORK (perf): runtime-selectable device handle.
///
/// Upstream hardwires `Inner<S> = channel::ChannelDeviceHandle<S>` on `multi_threading`
/// targets: every `submit`/`submit_blocking` crosses a channel to a dedicated server
/// thread. For a launch+sync-heavy single-GPU workload on a small shared-vCPU host, the
/// per-call thread hop (send + wakeup + reschedule) can dominate dispatch cost, and every
/// blocking readback pays a two-way ping-pong.
///
/// `CUBECL_DEVICE_INLINE=1` selects `reentrant::ReentrantMutexDeviceHandle<S>` instead:
/// tasks run inline on the calling thread under a reentrant mutex — zero thread hops, at
/// the price of the caller paying enqueue cost synchronously. Default (unset or any other
/// value) is the upstream channel handle: byte-for-byte upstream behavior.
///
/// The choice is process-global and latched on first use, so every handle in the process
/// agrees. `is_blocking()` (upstream `const fn`) becomes a plain `fn`; its only callers
/// gate multi-device collectives, which the inline mode does not support (they panic,
/// exactly as upstream's non-channel configurations would).
/// Programmatic default for the inline device handle, consulted only when the
/// `CUBECL_DEVICE_INLINE` env var is UNSET. Lets an embedding application (e.g. the
/// lightgbm_rs CUDA backend, where the inline handle measured a byte-identical win)
/// flip the default without touching the process environment — the env var remains
/// the user-facing override in BOTH directions (`0` forces channel, `1` forces
/// inline). Must be called before the first `DeviceHandle` use; the choice latches
/// process-globally on first use.
#[cfg(multi_threading)]
static INLINE_DEFAULT_ON: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// See [`INLINE_DEFAULT_ON`]. No-op after the first `DeviceHandle` use has latched.
#[cfg(multi_threading)]
pub fn set_device_inline_default(on: bool) {
    INLINE_DEFAULT_ON.store(on, core::sync::atomic::Ordering::SeqCst);
}

#[cfg(multi_threading)]
fn inline_enabled() -> bool {
    static INLINE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *INLINE.get_or_init(|| match std::env::var("CUBECL_DEVICE_INLINE") {
        Ok(v) => v == "1" || v.eq_ignore_ascii_case("true"),
        Err(_) => INLINE_DEFAULT_ON.load(core::sync::atomic::Ordering::SeqCst),
    })
}

#[cfg(multi_threading)]
enum InnerDyn<S: DeviceService> {
    Channel(channel::ChannelDeviceHandle<S>),
    Inline(reentrant::ReentrantMutexDeviceHandle<S>),
}

#[cfg(multi_threading)]
impl<S: DeviceService> Clone for InnerDyn<S> {
    fn clone(&self) -> Self {
        match self {
            InnerDyn::Channel(h) => InnerDyn::Channel(h.clone()),
            InnerDyn::Inline(h) => InnerDyn::Inline(h.clone()),
        }
    }
}

/// TODO: Docs
#[cfg(multi_threading)]
pub struct DeviceHandle<S: DeviceService> {
    handle: InnerDyn<S>,
}

#[cfg(multi_threading)]
impl<S: DeviceService> Clone for DeviceHandle<S> {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
        }
    }
}

#[allow(missing_docs)]
#[cfg(multi_threading)]
impl<S: DeviceService> DeviceHandle<S> {
    pub fn is_blocking() -> bool {
        inline_enabled()
    }

    pub fn insert(device_id: super::DeviceId, service: S) -> Result<Self, ServiceCreationError> {
        let handle = if inline_enabled() {
            InnerDyn::Inline(
                <reentrant::ReentrantMutexDeviceHandle<S> as DeviceHandleSpec<S>>::insert(
                    device_id, service,
                )?,
            )
        } else {
            InnerDyn::Channel(
                <channel::ChannelDeviceHandle<S> as DeviceHandleSpec<S>>::insert(
                    device_id, service,
                )?,
            )
        };
        Ok(Self { handle })
    }

    pub fn new(device_id: super::DeviceId) -> Self {
        let handle = if inline_enabled() {
            InnerDyn::Inline(
                <reentrant::ReentrantMutexDeviceHandle<S> as DeviceHandleSpec<S>>::new(device_id),
            )
        } else {
            InnerDyn::Channel(
                <channel::ChannelDeviceHandle<S> as DeviceHandleSpec<S>>::new(device_id),
            )
        };
        Self { handle }
    }

    pub fn device_id(&self) -> DeviceId {
        match &self.handle {
            InnerDyn::Channel(h) => h.device_id(),
            InnerDyn::Inline(h) => h.device_id(),
        }
    }

    pub fn utilities(&self) -> ServerUtilitiesHandle {
        match &self.handle {
            InnerDyn::Channel(h) => h.utilities(),
            InnerDyn::Inline(h) => h.utilities(),
        }
    }

    pub fn submit_blocking<'a, R: Send, T: FnOnce(&mut S) -> R + Send + 'a>(
        &self,
        task: T,
    ) -> Result<R, CallError> {
        match &self.handle {
            InnerDyn::Channel(h) => h.submit_blocking(task),
            InnerDyn::Inline(h) => h.submit_blocking(task),
        }
    }

    pub fn submit<T: FnOnce(&mut S) + Send + 'static>(&self, task: T) {
        match &self.handle {
            InnerDyn::Channel(h) => h.submit(task),
            InnerDyn::Inline(h) => h.submit(task),
        }
    }

    pub fn flush_queue(&self) {
        match &self.handle {
            InnerDyn::Channel(h) => h.flush_queue(),
            InnerDyn::Inline(h) => h.flush_queue(),
        }
    }

    pub fn exclusive<R: Send, T: FnOnce() -> R + Send>(&self, task: T) -> Result<R, CallError> {
        match &self.handle {
            InnerDyn::Channel(h) => h.exclusive(task),
            InnerDyn::Inline(h) => h.exclusive(task),
        }
    }
}

/// TODO: Docs
#[cfg(not(multi_threading))]
pub struct DeviceHandle<S: DeviceService> {
    handle: Inner<S>,
}

#[cfg(not(multi_threading))]
impl<S: DeviceService> Clone for DeviceHandle<S> {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
        }
    }
}

#[allow(missing_docs)]
#[cfg(not(multi_threading))]
impl<S: DeviceService> DeviceHandle<S> {
    pub const fn is_blocking() -> bool {
        Inner::<S>::BLOCKING
    }

    pub fn insert(device_id: super::DeviceId, service: S) -> Result<Self, ServiceCreationError> {
        Ok(Self {
            handle: <Inner<S> as DeviceHandleSpec<S>>::insert(device_id, service)?,
        })
    }

    pub fn new(device_id: super::DeviceId) -> Self {
        Self {
            handle: <Inner<S> as DeviceHandleSpec<S>>::new(device_id),
        }
    }

    pub fn device_id(&self) -> DeviceId {
        self.handle.device_id()
    }

    pub fn utilities(&self) -> ServerUtilitiesHandle {
        self.handle.utilities()
    }

    pub fn submit_blocking<'a, R: Send, T: FnOnce(&mut S) -> R + Send + 'a>(
        &self,
        task: T,
    ) -> Result<R, CallError> {
        self.handle.submit_blocking(task)
    }

    pub fn submit<T: FnOnce(&mut S) + Send + 'static>(&self, task: T) {
        self.handle.submit(task)
    }

    pub fn flush_queue(&self) {
        self.handle.flush_queue();
    }

    pub fn exclusive<R: Send, T: FnOnce() -> R + Send>(&self, task: T) -> Result<R, CallError> {
        self.handle.exclusive(task)
    }
}

#[cfg(test)]
mod tests_channel {
    type DeviceHandle<S> = channel::ChannelDeviceHandle<S>;

    include!("./tests.rs");
    include!("./tests_recursive.rs");
}

#[cfg(test)]
mod tests_mutex {
    type DeviceHandle<S> = mutex::MutexDeviceHandle<S>;

    include!("./tests.rs");
}

#[cfg(test)]
mod tests_reentrant {
    type DeviceHandle<S> = reentrant::ReentrantMutexDeviceHandle<S>;

    include!("./tests.rs");
    include!("./tests_recursive.rs");
}
