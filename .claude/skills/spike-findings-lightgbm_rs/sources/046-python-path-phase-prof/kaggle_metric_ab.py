#!/usr/bin/env python3
"""Spike-048 metric-eval A/B — isolate the per-iteration training-metric cost.

Diagnosis from kaggle-run-v5: lgb_rs CUDA = 17.04s vs official 3.26s. The
phase_prof BUDGET attributed metric=4.49s (26% of wall) to host-side training-
metric eval over 500k rows every iteration — official LightGBM does NOT compute a
training metric without an eval_set (is_provide_training_metric default false),
but lgb_rs forces it via `provide_train = ... || valid.is_none()` (booster.rs).

This A/B PROVES the win with ZERO code change: re-run lgb_rs CUDA with metric_freq
high so the metric evaluates only on the last iter (~1 eval vs 100). Builds ONLY
the CUDA wheel (fast).

Kaggle: kernel_type=script, enable_gpu=true, enable_internet=true.
"""
import os
import subprocess
import sys

REPO_URL = os.environ.get("REPO_URL", "https://github.com/BectorVoom/lightgbm_rs.git")
REPO_BRANCH = os.environ.get("REPO_BRANCH", "")

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def run(cmd, check=True):
    print(f"\n$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check)


def run_bench(which, device_type, metric_freq):
    env = dict(os.environ)
    env["LGBM_PHASE_PROF"] = "1"
    print(f"\n===== BENCH {which}/{device_type} metric_freq={metric_freq} =====", flush=True)
    res = subprocess.run(
        f"python3 {SPIKE_DIR}/bench_runner.py {which} {device_type} {metric_freq}",
        shell=True, capture_output=True, text=True, env=env,
    )
    print("--- stdout ---\n" + res.stdout, flush=True)
    print("--- stderr (phase_prof) ---\n" + res.stderr, flush=True)
    return res.stdout, res.stderr


def main():
    global SPIKE_DIR
    run("pip install -q -U numpy scipy scikit-learn maturin")
    run("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
    os.environ["PATH"] += ":/root/.cargo/bin"

    if not os.path.exists("lightgbm_rs"):
        run(f"git clone {REPO_URL}")
    else:
        run("cd lightgbm_rs && git checkout -- . && git fetch --all")
    if REPO_BRANCH:
        run(f"cd lightgbm_rs && git checkout {REPO_BRANCH} && git pull")
    else:
        run("cd lightgbm_rs && git pull")
    SPIKE_DIR = "lightgbm_rs/.planning/spikes/046-python-path-phase-prof"

    print("\n########## BUILD lightgbm_rs CUDA wheel (-F cuda) ##########")
    run("rm -rf lightgbm_rs/target/wheels/")
    run("cd lightgbm_rs/crates/lgbm-python && maturin build --release -F cuda -j 2")
    run("pip install $(ls lightgbm_rs/target/wheels/*.whl | head -n 1) --force-reinstall")

    outs = {}
    outs["baseline"], _ = run_bench("rs", "cuda", 1)     # metric every iter (current)
    outs["mfreq200"], _ = run_bench("rs", "cuda", 200)   # metric only on last iter

    def t(key):
        for line in outs.get(key, "").splitlines():
            if line.startswith("RESULT"):
                return float(line.split("train_time_s=")[-1])
        return float("nan")

    base, fast = t("baseline"), t("mfreq200")
    print("\n\n==================== METRIC A/B SUMMARY ====================")
    print(f"  lgb_rs CUDA  metric_freq=1   (per-iter eval): {base:.3f} s")
    print(f"  lgb_rs CUDA  metric_freq=200 (eval last only): {fast:.3f} s")
    print(f"  >>> metric-eval cost removed: {base - fast:.3f} s "
          f"({100*(base-fast)/base:.1f}% of wall) <<<")
    print(f"  official LightGBM CUDA reference: 3.26 s")
    print(f"  gap after removal: {fast/3.26:.2f}x (was {base/3.26:.2f}x)")


if __name__ == "__main__":
    main()
