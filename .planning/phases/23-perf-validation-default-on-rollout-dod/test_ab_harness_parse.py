#!/usr/bin/env python3
"""Local dry-run test of the A/B harness parse + median + verdict + parity-envelope logic.

Runs WITHOUT Kaggle and WITHOUT a GPU (the harness itself only runs on Kaggle real CUDA;
the local GPU is a spoofed 8-CU APU). Imports the REAL importable functions from
`ab_harness.py` (never a reimplementation) and asserts them against the committed
`fixtures/sample_phase_prof.stderr` plus synthetic medians/parity values.

Run: `python test_ab_harness_parse.py`  (plain assertions, no pytest required).
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from ab_harness import (  # noqa: E402 — path insert must precede import
    PARITY_ATOL,
    PARITY_RTOL,
    compute_verdict,
    launches_per_tree,
    median,
    parity_ok,
    parse_launches,
    shape_key,
)

FIXTURE = os.path.join(_HERE, "fixtures", "sample_phase_prof.stderr")


def _fixture_lines():
    with open(FIXTURE) as f:
        return [ln for ln in f if "COUNTS" in ln]


def test_parse_host_and_on_device():
    """(a)+(b): on-device line parses the collapsed total; host line parses ~8570 —
    both via the SHORT total-only regex, with on_device= captured from inside the parens."""
    host_line, on_line = None, None
    for ln in _fixture_lines():
        parsed = parse_launches(ln)
        assert parsed is not None, f"COUNTS line failed to parse: {ln!r}"
        if parsed["launches"] == 8570:
            host_line = parsed
        else:
            on_line = parsed
    assert host_line is not None and host_line["launches"] == 8570, "host ~8570 parse"
    assert host_line.get("on_device") == 0, "host arm on_device=0 sub-field"
    assert on_line is not None, "on-device COUNTS line present"
    assert 0 < on_line["launches"] < 8570, "on-device launches collapsed below baseline"
    assert on_line.get("on_device") == on_line["launches"], "on-device sub-field == total"


def test_launches_per_tree():
    """(c): device_launches/tree = launches / 100."""
    assert launches_per_tree(8570) == 85.7
    assert launches_per_tree(2900) == 29.0
    assert launches_per_tree(2900) < 85.7, "on-device launches/tree below SC-2 baseline"


def test_median_middle_of_three():
    """(d): the median helper returns the middle of 3 synthetic wall-clocks."""
    assert median([3.0, 1.0, 2.0]) == 2.0
    assert median([10.5, 10.1, 10.9]) == 10.5


def test_verdict_pass_branch():
    """(e): PASS when BOTH shapes' on-device medians are <= 1.05x host AND parity_ok."""
    medians = {
        shape_key(500000, 50): {"on_device": 10.0, "host_cuda": 10.2},   # faster
        shape_key(100000, 500): {"on_device": 5.10, "host_cuda": 5.00},  # +2% (within 5%)
    }
    verdict, not_slower = compute_verdict(medians, parity_ok_flag=True)
    assert not_slower is True
    assert verdict == "PASS", f"expected PASS, got {verdict}"


def test_verdict_fail_on_slowdown():
    """(e): FAIL when either shape exceeds the 1.05x bar (even with parity_ok True)."""
    medians = {
        shape_key(500000, 50): {"on_device": 10.0, "host_cuda": 10.2},
        shape_key(100000, 500): {"on_device": 6.0, "host_cuda": 5.0},  # +20% > 5%
    }
    verdict, not_slower = compute_verdict(medians, parity_ok_flag=True)
    assert not_slower is False
    assert verdict == "FAIL", f"expected FAIL on slowdown, got {verdict}"


def test_verdict_fail_on_parity():
    """(e): FAIL when parity_ok is False even if both shapes are not-slower."""
    medians = {
        shape_key(500000, 50): {"on_device": 10.0, "host_cuda": 10.2},
        shape_key(100000, 500): {"on_device": 5.0, "host_cuda": 5.0},
    }
    verdict, not_slower = compute_verdict(medians, parity_ok_flag=False)
    assert not_slower is True
    assert verdict == "FAIL", f"expected FAIL on parity, got {verdict}"


def test_parity_envelope_near_tie_true():
    """(f): parity_ok True for a near-tie just inside atol + rtol*|p_host| (W2)."""
    p_host_absmax = 4.0
    gap = PARITY_ATOL + PARITY_RTOL * p_host_absmax  # exactly on the envelope boundary
    assert parity_ok(gap, p_host_absmax) is True, "boundary near-tie must pass"
    assert parity_ok(gap * 0.5, p_host_absmax) is True, "well-inside near-tie passes"


def test_parity_envelope_divergence_false():
    """(f): parity_ok False for a real divergence beyond the envelope (W2)."""
    p_host_absmax = 4.0
    gap = (PARITY_ATOL + PARITY_RTOL * p_host_absmax) * 100.0
    assert parity_ok(gap, p_host_absmax) is False, "real divergence must fail"


def main():
    tests = [
        test_parse_host_and_on_device,
        test_launches_per_tree,
        test_median_middle_of_three,
        test_verdict_pass_branch,
        test_verdict_fail_on_slowdown,
        test_verdict_fail_on_parity,
        test_parity_envelope_near_tie_true,
        test_parity_envelope_divergence_false,
    ]
    failures = []
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} tests passed.")
    if failures:
        sys.exit(1)
    print("ALL PASS — parse + median + verdict + parity envelope proven locally.")


if __name__ == "__main__":
    main()
