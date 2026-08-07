mod base;

pub use base::*;

pub(crate) mod handle;

/// LIGHTGBM_RS FORK: re-export the inline-handle default setter (see
/// `handle::set_device_inline_default`).
#[cfg(all(feature = "std", not(target_family = "wasm")))]
pub use handle::set_device_inline_default;
