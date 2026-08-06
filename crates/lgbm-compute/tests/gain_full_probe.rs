//! Host-side probe of the two-axis gain primitives.
use lgbm_compute::gain::*;

#[test]
fn clamp_and_form_switch_behave() {
    let (g, h, l1, l2) = (4.0f64, 2.0f64, 0.0f64, 0.0f64);
    // base output = -g/(h+l2) = -2.0
    let base = calculate_splitted_leaf_output(false, g, h, l1, l2);
    assert_eq!(base, -2.0);

    // max_delta_step = 0 -> no clamp, no smoothing -> identical to base
    assert_eq!(
        calculate_splitted_leaf_output_full(false, g, h, l1, l2, 0.0, false, 0.0, 10, 0.0),
        base
    );
    // max_delta_step = 0.5 -> clamped to -0.5
    assert_eq!(
        calculate_splitted_leaf_output_full(false, g, h, l1, l2, 0.5, false, 0.0, 10, 0.0),
        -0.5
    );
    // max_delta_step = 100 -> guard does not fire
    assert_eq!(
        calculate_splitted_leaf_output_full(false, g, h, l1, l2, 100.0, false, 0.0, 10, 0.0),
        base
    );

    // gain: closed form when both axes off
    let closed = get_leaf_gain(false, g, h, l1, l2);
    assert_eq!(
        get_leaf_gain_full(false, g, h, l1, l2, 0.0, false, 0.0, 10, 0.0),
        closed
    );
    // with max_delta_step > 0 the FORM switches to given-output, evaluated at the
    // clamped output.
    let at_clamp = get_leaf_gain_given_output(false, g, h, l1, l2, -0.5);
    assert_eq!(
        get_leaf_gain_full(false, g, h, l1, l2, 0.5, false, 0.0, 10, 0.0),
        at_clamp
    );
    // ...and when the clamp does not bind, the form is still given-output at base.
    let at_base = get_leaf_gain_given_output(false, g, h, l1, l2, base);
    assert_eq!(
        get_leaf_gain_full(false, g, h, l1, l2, 100.0, false, 0.0, 10, 0.0),
        at_base
    );

    // smoothing blend
    let ps = 2.0f64;
    let n = 10i32;
    let parent = 0.5f64;
    let nps = f64::from(n) / ps;
    let expect = base * nps / (nps + 1.0) + parent / (nps + 1.0);
    assert_eq!(
        calculate_splitted_leaf_output_full(false, g, h, l1, l2, 0.0, true, ps, n, parent),
        expect
    );
    // clamp THEN blend
    let expect2 = -0.5 * nps / (nps + 1.0) + parent / (nps + 1.0);
    assert_eq!(
        calculate_splitted_leaf_output_full(false, g, h, l1, l2, 0.5, true, ps, n, parent),
        expect2
    );
}

/// Pin the FUSED multiply-add in `get_leaf_gain_given_output` against the reference's
/// own operands.
///
/// These bits are the real ones from the leaf where `max_delta_step = 0.05` clamps both
/// children to the same output, so the split gain equals the no-split gain EXACTLY in
/// real arithmetic and the three candidate formulations differ by a single bit. A
/// regression to the un-fused `-(2·g·o + (h+λ)·o²)` — or to fusing the OTHER multiply —
/// makes `candidate != shift` here, and downstream turns a whole feature splittable
/// that the reference excludes from an entire subtree.
///
/// Kept as a direct unit test rather than relying on the 13-cell oracle replay so the
/// failure names the cause instead of surfacing as a mismatched tree.
#[test]
fn the_given_output_gain_is_fused_exactly_as_the_reference_is() {
    let f = f64::from_bits;
    let (g, h) = (f(0xc056c1479e000000), f(0x40591d6d5a200000));
    let (gl, hl) = (f(0xc05651479f000000), f(0x40578d9650100000));
    let (gr, hr) = (f(0xbffbffffc0000000), f(0x4018fd70a1000001));
    let (l1, l2, mds) = (0.0f64, 0.0f64, 0.05f64);

    let out = |g: f64, h: f64| {
        calculate_splitted_leaf_output_full(false, g, h, l1, l2, mds, false, 0.0, 0, 0.0)
    };
    let gain = |g: f64, h: f64| {
        get_leaf_gain_full(false, g, h, l1, l2, mds, false, 0.0, 0, 0.0)
    };

    // Both children really do clamp, to the SAME output — that is what makes the
    // identity exact and the comparison a single-bit decision.
    assert_eq!(out(gl, hl).abs(), mds, "left child must be clamped");
    assert_eq!(out(gr, hr).abs(), mds, "right child must be clamped");
    assert_eq!(out(gl, hl), out(gr, hr), "both children clamp the SAME way");

    let candidate = gain(gl, hl) + gain(gr, hr);
    let shift = gain(g, h);
    assert_eq!(
        candidate.to_bits(),
        shift.to_bits(),
        "split gain must equal the no-split gain BIT-FOR-BIT here \
         (candidate {candidate:e} vs shift {shift:e}, diff {:e}); the reference build \
         contracts `2·g·o + (h+l2)·o²` into one fma, so `get_leaf_gain_given_output` \
         must fuse the FIRST multiply — see its doc comment",
        candidate - shift
    );
}
