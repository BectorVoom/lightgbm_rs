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
