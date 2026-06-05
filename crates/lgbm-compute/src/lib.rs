//! `lgbm-compute` — the single CubeCL isolation seam (CMP-01).
//!
//! This crate exists to confine all `cubecl` type names and API churn to one
//! place so the alpha-stage CubeCL surface can evolve without leaking into
//! `lgbm-core` or any other crate. **No compute kernels live here yet** — per
//! D-09 this phase ships only a trait skeleton. The actual histogram /
//! score-update kernels (and CPU ↔ ROCm runtime selection) arrive in Phase 4.
//!
//! Keep this crate dependency-light and kernel-free until then; downstream
//! crates should depend only on the [`Backend`] abstraction, never on `cubecl`
//! directly.

/// The compute backend seam (CMP-01).
///
/// A deliberately minimal skeleton: it names a `Runtime` associated type that
/// Phase 4 will bind to a concrete CubeCL runtime (CPU or ROCm/HIP). No methods
/// are defined yet because no kernels exist this phase. Implementors and kernel
/// methods are added when the compute layer is filled in.
///
/// This trait is the ONLY place where CubeCL runtime types should appear; that
/// is the whole point of the seam.
pub trait Backend {
    /// The concrete CubeCL runtime this backend dispatches kernels to.
    ///
    /// Left unconstrained this phase (skeleton only); Phase 4 binds it to a
    /// `cubecl::Runtime` implementor and adds the associated kernel methods.
    type Runtime;
}
