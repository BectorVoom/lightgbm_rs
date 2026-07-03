#!/usr/bin/env python3
"""Kaggle A/B harness (ODL-20) — on-device vs host-CUDA real-discrete-CUDA measurement.

Extends the `continue_benchmark.py` family (clone BectorVoom/lightgbm_rs -> maturin
`-F cuda` -> install -> official lightgbm from source) and runs the D-05 matrix:

    3 runs x 2 shapes {(500000, 50), (100000, 500)} x 2 paths {host-cuda, on-device}

Per-arm environment (P-1 — env MUST be set BEFORE the worker process starts, because
`LGBM_CUDA_ON_DEVICE` is resolved once per process via an OnceLock):

    host-cuda arm : LGBM_CUDA_ON_DEVICE=0   (the 23-01 tri-state force-OFF off-switch)
    on-device arm : LGBM_CUDA_ON_DEVICE=1   (force-ON)
    BOTH arms     : LGBM_PHASE_PROF=1        (surface the phase_prof COUNTS line)

Verdict (D-04 <=5% "within noise", D-05 both-shapes) folds in the D-11 real-CUDA
end-to-end parity gate (`parity_ok`), and the whole matrix/parity flow is wrapped so
`results.{md,json}` land UNCONDITIONALLY on a `finally`/before-exit path (D-09) — the
committed evidence artifact is the phase's core value regardless of the verdict.

SECURITY (T-23-03): this committed script embeds NO Kaggle credential. Auth stays in the
Kaggle CLI's out-of-band config (`boomvector`). Supply chain (T-23-03-SC): the repo is
pinned (BectorVoom/lightgbm_rs) and both lgb_rs (maturin from source) and official
lightgbm (`--no-binary`) are built from source — no third-party binary wheel.

The parse / median / verdict / parity-envelope logic below is factored into importable
pure functions so `test_ab_harness_parse.py` can prove them LOCALLY (no Kaggle, no GPU).
Only these light stdlib imports live at module scope; the heavy numpy/lightgbm imports are
deferred into the Kaggle-only run path so the module imports cleanly on any machine.
"""

import json
import os
import re
import statistics
import subprocess
import sys
import time

# --------------------------------------------------------------------------------------
# Constants (D-06 shapes, D-08 baseline convention, D-04 tolerance, D-11 parity envelope)
# --------------------------------------------------------------------------------------
N_ESTIMATORS = 100                       # D-08 baseline: 100-trees denominator
SHAPES = [(500000, 50), (100000, 500)]   # D-06: tall-narrow AND wide
N_RUNS = 3                               # D-05: 3 runs per (shape, path), take the median
SLOWDOWN_TOL = 1.05                      # D-04: on-device <= 1.05 * host-cuda ("within 5% noise")
HOST_BASELINE_LAUNCHES_PER_TREE = 8570 / 100  # SC-2: 85.7 launches/tree host baseline

# D-11 real-CUDA parity envelope (W2, def-f8u-01):
#   The PRIMARY parity proof is the cpu-f64 structural anchor — the on-device tree is
#   bit-exact on the cubecl-cpu lane (green in the merge gate + per-phase gates 14-22).
#   The Kaggle real-CUDA `max_abs_on_host` is CORROBORATING evidence only, held to a
#   documented small envelope so a legitimate f32-vs-f32 near-tie cannot false-negative
#   and wrongly veto the ODL-21 default-on flip. NEVER compare two nondeterministic GPU
#   f32 paths at a bare 1e-6 without this envelope + the cpu-f64 anchor behind it.
PARITY_ATOL = 1e-6
PARITY_RTOL = 1e-6

# SHORT total-only COUNTS regex (W3 producer/consumer contract): 23-02 keeps
# `device_launches=<total>` as the FIRST COUNTS field and folds `on_device=` INSIDE the
# parenthesized breakdown, so this deliberately does NOT use the full paren-terminated
# form — it captures only the total, which stays stable across the 23-02 change.
COUNTS_RE = re.compile(r"\[phase_prof:\w+\] COUNTS: device_launches=(?P<launches>\d+)")
# Optional corroborating sub-field (non-anchored) — the on-device-only launch count.
ON_DEVICE_RE = re.compile(r"on_device=(?P<on_device>\d+)")


# --------------------------------------------------------------------------------------
# Pure, importable logic (tested locally by test_ab_harness_parse.py — no Kaggle needed)
# --------------------------------------------------------------------------------------
def parse_launches(stderr):
    """Parse the phase_prof COUNTS line, returning the total device_launches (+ optional
    on_device sub-field), or None if no COUNTS line is present.

    Uses the SHORT total-only regex (W3): matches `device_launches=<total>` regardless of
    the parenthesized breakdown that follows.
    """
    m = COUNTS_RE.search(stderr)
    if not m:
        return None
    out = {"launches": int(m.group("launches"))}
    sub = ON_DEVICE_RE.search(stderr)
    if sub:
        out["on_device"] = int(sub.group("on_device"))
    return out


def launches_per_tree(launches, n_estimators=N_ESTIMATORS):
    """device_launches / tree — the SC-2 quantity compared against the 85.7 host baseline."""
    return launches / float(n_estimators)


def median(values):
    """Median of the run wall-clocks (middle of 3)."""
    return statistics.median(values)


def parity_ok(max_abs_on_host, p_host_absmax, atol=PARITY_ATOL, rtol=PARITY_RTOL):
    """D-11 near-tie envelope (W2, def-f8u-01).

    Returns True iff the on-device-vs-host-CUDA max abs prediction gap sits inside the
    documented envelope `atol + rtol * max(|p_host|)`. The cpu-f64 structural anchor is
    the primary parity proof; this real-CUDA check is corroboration under the envelope so
    a legitimate f32-vs-f32 near-tie cannot produce a false-negative veto of the flip.
    """
    return max_abs_on_host <= (atol + rtol * p_host_absmax)


def compute_verdict(medians, parity_ok_flag, tol=SLOWDOWN_TOL):
    """D-04 (<=5%) / D-05 (both shapes) verdict, folding in the D-11 `parity_ok` gate.

    `medians` maps a shape key -> {"on_device": m, "host_cuda": m}. PASS iff for BOTH
    shapes median(on_device) <= tol * median(host_cuda) AND `parity_ok_flag` is True.
    Returns (verdict, not_slower) where verdict is "PASS"/"FAIL".
    """
    not_slower = all(
        v["on_device"] <= tol * v["host_cuda"] for v in medians.values()
    )
    verdict = "PASS" if (not_slower and parity_ok_flag) else "FAIL"
    return verdict, not_slower


def shape_key(n_samples, n_features):
    return f"{n_samples}x{n_features}"


# --------------------------------------------------------------------------------------
# Kaggle-only worker: one training arm per subprocess (fresh env per P-1 OnceLock rule)
# --------------------------------------------------------------------------------------
# Runs a single (shape, backend) training arm, saves raw_score predictions to a .npy for
# the parity comparison, and prints a WALLCLOCK= line the parent parses from stdout. The
# phase_prof COUNTS line is emitted on stderr (the parent captures it). backend is 'rs'
# (lightgbm_rs, gated by the inherited LGBM_CUDA_ON_DEVICE env) or 'official' (context).
_WORKER_SRC = r'''
import sys, time
import numpy as np
from sklearn.datasets import make_classification

n_samples = int(sys.argv[1]); n_features = int(sys.argv[2])
backend = sys.argv[3]; out_pred = sys.argv[4]; seed = int(sys.argv[5])

X, y = make_classification(n_samples=n_samples, n_features=n_features, random_state=seed)
params = dict(objective="binary", metric="binary_logloss", device_type="cuda",
              num_leaves=31, learning_rate=0.1, n_estimators=100, verbose=-1)

if backend == "official":
    import lightgbm as lgb
    model = lgb.LGBMClassifier(**params)
else:
    import lightgbm_rs as lgb_rs
    model = lgb_rs.LGBMClassifier(**params)

start = time.time()
model.fit(X, y)
wall = time.time() - start
p = np.asarray(model.predict(X, raw_score=True), dtype=np.float64)
np.save(out_pred, p)
print(f"WALLCLOCK={wall}")
'''


def _run_arm(n_samples, n_features, backend, on_device, out_pred, seed, worker_path):
    """Launch one training arm as a subprocess with the per-arm env, capturing stdout
    (WALLCLOCK) and stderr (phase_prof COUNTS)."""
    env = dict(os.environ)
    env["LGBM_PHASE_PROF"] = "1"            # both arms surface the COUNTS line
    if backend == "rs":
        # host-cuda arm => "0" (23-01 tri-state force-OFF); on-device arm => "1".
        env["LGBM_CUDA_ON_DEVICE"] = "1" if on_device else "0"
    proc = subprocess.run(
        [sys.executable, worker_path, str(n_samples), str(n_features),
         backend, out_pred, str(seed)],
        env=env, capture_output=True, text=True,
    )
    wall = None
    for line in proc.stdout.splitlines():
        if line.startswith("WALLCLOCK="):
            wall = float(line.split("=", 1)[1])
    parsed = parse_launches(proc.stderr)
    return {
        "wall": wall,
        "launches": parsed["launches"] if parsed else None,
        "on_device_launches": parsed.get("on_device") if parsed else None,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _setup_kaggle_build():
    """Clone the PINNED repo (T-23-03-SC) + build lgb_rs (maturin -F cuda) + install +
    build official lightgbm from source (`--no-binary`, USE_CUDA=ON) for the context ratio.
    No credential literal here — Kaggle auth is out-of-band (T-23-03)."""
    def run(cmd):
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True, check=True)

    run("pip install -U numpy scipy scikit-learn maturin")
    run("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
    os.environ["PATH"] += ":/root/.cargo/bin"
    if not os.path.exists("lightgbm_rs"):
        run("git clone https://github.com/BectorVoom/lightgbm_rs.git")
    else:
        run("cd lightgbm_rs && git checkout -- . && git pull")
    run("rm -rf lightgbm_rs/target/wheels/")
    run("cd lightgbm_rs/crates/lgbm-python && maturin build --release -F cuda -j 2")
    run("pip install $(ls lightgbm_rs/target/wheels/*.whl | head -n 1) --force-reinstall")
    # Official lightgbm from source with CUDA for the context ratio (T-23-03-SC: --no-binary).
    try:
        subprocess.run(
            "python3 -c 'import lightgbm as lgb; import numpy as np; "
            "lgb.LGBMClassifier(device_type=\"cuda\", num_leaves=2, n_estimators=1)"
            ".fit(np.zeros((10,2)), np.zeros(10))'",
            shell=True, check=True,
        )
    except subprocess.CalledProcessError:
        run("pip uninstall -y lightgbm")
        run("pip install --no-binary lightgbm lightgbm -C cmake.define.USE_CUDA=ON")


def _emit_results(results, out_dir):
    """Write results.json + results.md UNCONDITIONALLY (D-09 — the committed evidence
    artifact ALWAYS lands, even on a parity/run failure).

    ALSO echo the full results.json to stdout bracketed by unique sentinels so the
    numbers survive in the Kaggle LOG even if Kaggle drops the output files under its
    output-size cap (the phase-23 first-run capture bug: the large /kaggle/working —
    cloned repo + Rust target/ + 500k-row .npy preds — blew the output cap and the tiny
    results.{md,json} were dropped, so the exact ratios/parity were lost while the log
    survived). The stdout echo is the durable capture; the files stay best-effort."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Durable log-borne capture (survives the Kaggle output-size cap that drops files).
    print("<<<AB_RESULTS_JSON")
    json.dump(results, sys.stdout)
    print("\nAB_RESULTS_JSON>>>")

    lines = []
    lines.append(f"# Phase 23 On-Device A/B Results — verdict: **{results['verdict']}**")
    lines.append("")
    lines.append(f"- n_estimators (denominator): {N_ESTIMATORS}")
    lines.append(f"- runs per (shape, path): {N_RUNS} (median reported)")
    lines.append(f"- not-slower bar (D-04): on-device <= {SLOWDOWN_TOL:g} x host-cuda")
    lines.append(f"- SC-2 host baseline: {HOST_BASELINE_LAUNCHES_PER_TREE:g} launches/tree")
    lines.append("")
    lines.append("## Wall-clock medians + not-slower ratio (D-05)")
    lines.append("")
    lines.append("| shape | on-device med (s) | host-cuda med (s) | ratio (on/host) | official med (s) | official/host |")
    lines.append("|-------|-------------------|-------------------|-----------------|------------------|---------------|")
    for sk, row in results["medians"].items():
        on = row["on_device"]; host = row["host_cuda"]; off = row.get("official")
        ratio = on / host if host else float("nan")
        off_ratio = (off / host) if (off and host) else float("nan")
        off_str = f"{off:.3f}" if off is not None else "n/a"
        lines.append(f"| {sk} | {on:.3f} | {host:.3f} | {ratio:.3f} | {off_str} | {off_ratio:.3f} |")
    lines.append("")
    lines.append("## device_launches/tree vs 8,570/100-trees baseline (SC-2)")
    lines.append("")
    lines.append("| shape | on-device launches/tree | host launches/tree | baseline |")
    lines.append("|-------|-------------------------|--------------------|----------|")
    for sk, lr in results["launches"].items():
        on_lt = lr.get("on_device_per_tree")
        host_lt = lr.get("host_per_tree")
        on_s = f"{on_lt:.2f}" if on_lt is not None else "n/a"
        host_s = f"{host_lt:.2f}" if host_lt is not None else "n/a"
        lines.append(f"| {sk} | {on_s} | {host_s} | {HOST_BASELINE_LAUNCHES_PER_TREE:g} |")
    lines.append("")
    lines.append("## Real-CUDA parity (D-11)")
    lines.append("")
    p = results["parity"]
    lines.append(f"- max_abs_on_host (|p_on - p_host|): {p['max_abs_on_host']}")
    lines.append(f"- envelope: atol={PARITY_ATOL:g}, rtol={PARITY_RTOL:g}")
    lines.append(f"- parity_ok: {p['parity_ok']}")
    lines.append(f"- context only (|p_on - p_official|, NOT a same-tol gate — f32 reference): "
                 f"{p.get('max_abs_on_official')}")
    lines.append("")
    lines.append("> **Primary parity proof is the cpu-f64 structural anchor** — the on-device "
                 "tree is bit-exact on the cubecl-cpu lane (merge gate + per-phase gates 14-22). "
                 "This real-CUDA max_abs is CORROBORATING evidence under the documented near-tie "
                 "envelope (W2, def-f8u-01); it is not the sole veto of the ODL-21 flip.")
    lines.append("")
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def run_matrix(out_dir=".", seed=42):
    """Run the full D-05 3x2x2 matrix, compute medians/ratios/launches/parity/verdict, and
    emit results.{md,json} on a finally/before-exit path (D-09). Returns the verdict string.
    """
    import numpy as np  # deferred heavy import (Kaggle-only path)

    worker_path = os.path.join(out_dir, "_ab_worker.py")
    with open(worker_path, "w") as f:
        f.write(_WORKER_SRC)

    results = {
        "verdict": "FAIL",  # pessimistic default so a crash-before-verdict lands as FAIL
        "shapes": [list(s) for s in SHAPES],
        "n_estimators": N_ESTIMATORS,
        "n_runs": N_RUNS,
        "slowdown_tol": SLOWDOWN_TOL,
        "parity_envelope": {"atol": PARITY_ATOL, "rtol": PARITY_RTOL},
        "raw_runs": {},
        "medians": {},
        "launches": {},
        "parity": {"max_abs_on_host": None, "parity_ok": False, "max_abs_on_official": None},
        "errors": [],
    }

    try:
        parity_max_abs = 0.0
        parity_host_absmax = 0.0
        parity_max_abs_official = 0.0
        for (n, f) in SHAPES:
            sk = shape_key(n, f)
            results["raw_runs"][sk] = {"on_device": [], "host_cuda": [], "official": []}
            on_walls, host_walls, off_walls = [], [], []
            on_launch_total, host_launch_total = None, None
            for run_i in range(N_RUNS):
                on_pred = os.path.join(out_dir, f"pred_on_{sk}_{run_i}.npy")
                host_pred = os.path.join(out_dir, f"pred_host_{sk}_{run_i}.npy")
                off_pred = os.path.join(out_dir, f"pred_off_{sk}_{run_i}.npy")

                on = _run_arm(n, f, "rs", True, on_pred, seed, worker_path)
                host = _run_arm(n, f, "rs", False, host_pred, seed, worker_path)
                off = _run_arm(n, f, "official", False, off_pred, seed, worker_path)

                results["raw_runs"][sk]["on_device"].append(on)
                results["raw_runs"][sk]["host_cuda"].append(host)
                results["raw_runs"][sk]["official"].append(off)
                if on["wall"] is not None:
                    on_walls.append(on["wall"])
                if host["wall"] is not None:
                    host_walls.append(host["wall"])
                if off["wall"] is not None:
                    off_walls.append(off["wall"])
                if on["launches"] is not None:
                    on_launch_total = on["launches"]
                if host["launches"] is not None:
                    host_launch_total = host["launches"]

                # Parity: compare on-device vs host-cuda (and context vs official) on the
                # SAME (X, y) — worst case across runs/shapes folds into the single gate.
                if on["returncode"] == 0 and host["returncode"] == 0 \
                        and os.path.exists(on_pred) and os.path.exists(host_pred):
                    p_on = np.load(on_pred)
                    p_host = np.load(host_pred)
                    parity_max_abs = max(parity_max_abs, float(np.max(np.abs(p_on - p_host))))
                    parity_host_absmax = max(parity_host_absmax, float(np.max(np.abs(p_host))))
                    if os.path.exists(off_pred):
                        p_off = np.load(off_pred)
                        parity_max_abs_official = max(
                            parity_max_abs_official, float(np.max(np.abs(p_on - p_off)))
                        )

            results["medians"][sk] = {
                "on_device": median(on_walls) if on_walls else float("inf"),
                "host_cuda": median(host_walls) if host_walls else float("inf"),
                "official": median(off_walls) if off_walls else None,
            }
            results["launches"][sk] = {
                "on_device_total": on_launch_total,
                "host_total": host_launch_total,
                "on_device_per_tree": launches_per_tree(on_launch_total) if on_launch_total is not None else None,
                "host_per_tree": launches_per_tree(host_launch_total) if host_launch_total is not None else None,
            }

        # Fold the D-11 parity result into a boolean BEFORE any verdict comparison (W1 —
        # NEVER a bare inline assert that would abort before results are written).
        pk = parity_ok(parity_max_abs, parity_host_absmax)
        results["parity"] = {
            "max_abs_on_host": parity_max_abs,
            "host_absmax": parity_host_absmax,
            "parity_ok": bool(pk),
            "max_abs_on_official": parity_max_abs_official,
        }

        verdict, not_slower = compute_verdict(results["medians"], pk)
        results["verdict"] = verdict
        results["not_slower"] = not_slower
    except Exception as e:  # noqa: BLE001 — capture so results still land (D-09)
        results["errors"].append(repr(e))
    finally:
        # D-09: the committed evidence artifact ALWAYS lands, verdict or crash either way.
        _emit_results(results, out_dir)

    return results["verdict"]


def main():
    out_dir = os.environ.get("AB_OUT_DIR", ".")
    if os.environ.get("AB_SKIP_BUILD") != "1":
        _setup_kaggle_build()
        # The maturin build clones into ./lightgbm_rs; run the matrix from repo-adjacent cwd.
    verdict = run_matrix(out_dir=out_dir)
    print(f"\n>>> A/B VERDICT: {verdict} <<<")
    # AFTER emitting results, exit non-zero on FAIL (either shape >5% OR parity_ok false).
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
