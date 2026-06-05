//! CMP-04 startup capability gate test.
//!
//! Constructs a cubecl-cpu client, probes its capabilities, and asserts the
//! verified asymmetric cpu matrix (RESEARCH Pitfall 2): `has_plane=false`,
//! `has_f64=true`, `has_f32_atomic=false`, `plane_size=1`, and that the derived
//! reduce path is [`ReducePath::Sequential`] — i.e. the single-owner ordered
//! fold IS the cpu path, not a fallback.

use lgbm_compute::runtime::{probe_capabilities, cpu_client, ReducePath};

#[test]
fn capability_cpu_matrix() {
    let client = cpu_client();
    let caps = probe_capabilities(&client);

    assert!(
        !caps.has_plane,
        "cubecl-cpu must NOT advertise Plane::Ops (plane_size 1, not inserted)"
    );
    assert!(
        caps.has_f64,
        "cubecl-cpu must support f64 (the hist_t=double anchor accumulator)"
    );
    assert!(
        !caps.has_f32_atomic,
        "cubecl-cpu must NOT advertise f32 atomic-add (commented out upstream)"
    );
    assert_eq!(
        caps.plane_size, 1,
        "cubecl-cpu plane size must be 1 (no warp collectives)"
    );
    assert_eq!(
        caps.reduce_path(),
        ReducePath::Sequential,
        "absent Plane::Ops must select the Sequential single-owner fold"
    );
}
