#!/usr/bin/env python3
"""Spike-048 / quick-260628-f57 — confirm the metric-eval fix end-to-end on Kaggle.

The fix (commit dd3e3be, on master) made `provide_train` C++-faithful so the
default Python path (no eval_set) no longer evaluates the training metric every
iteration. This rebuilds the CUDA wheel from master and confirms on REAL NVIDIA
hardware, via the spike-046 phase_prof hook, that:
  1. the lgb_rs CUDA DEFAULT run (metric_freq=1) now shows metric phase ≈ 0
     (was ~4.5s / 26% of wall before the fix), and
  2. the default wall has dropped accordingly vs an in-session official-CUDA ref.

Also re-runs the metric_freq=200 arm: with the fix, default ≈ mfreq200 (the
workaround is no longer needed — the A/B delta should collapse to ~0).

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

    # Fail loud if the fix isn't in the clone.
    grep = subprocess.run(
        'grep -n "quick-260628-f57" lightgbm_rs/crates/lgbm/src/booster.rs',
        shell=True, capture_output=True, text=True)
    print("fix present in clone:" , grep.stdout.strip() or "!!! NOT FOUND — push master first")

    print("\n########## BUILD lightgbm_rs CUDA wheel (-F cuda, from master w/ fix) ##########")
    run("rm -rf lightgbm_rs/target/wheels/")
    run("cd lightgbm_rs/crates/lgbm-python && maturin build --release -F cuda -j 2")
    run("pip install $(ls lightgbm_rs/target/wheels/*.whl | head -n 1) --force-reinstall")

    outs, errs = {}, {}
    outs["rs_default"], errs["rs_default"] = run_bench("rs", "cuda", 1)    # DEFAULT path (fix active)
    outs["rs_mfreq200"], errs["rs_mfreq200"] = run_bench("rs", "cuda", 200)  # workaround arm
    outs["off"], _ = run_bench("off", "cuda", 1)                            # in-session reference

    # Ensure official lightgbm has CUDA (build from source if the import path lacks it).
    def t(key):
        for line in outs.get(key, "").splitlines():
            if line.startswith("RESULT"):
                return float(line.split("train_time_s=")[-1])
        return float("nan")

    def metric_ms(key):
        # last BUDGET line's metric=... (the timed run)
        val = None
        for line in errs.get(key, "").splitlines():
            if "LOOP:" in line and "metric=" in line:
                seg = line.split("metric=")[-1]
                val = seg.split("ms")[0]
        return val

    base, fast, off = t("rs_default"), t("rs_mfreq200"), t("off")
    print("\n\n==================== FIX CONFIRMATION ====================")
    print(f"  lgb_rs CUDA  DEFAULT (metric_freq=1, FIX active): {base:.3f} s   metric_phase={metric_ms('rs_default')} ms")
    print(f"  lgb_rs CUDA  metric_freq=200 (workaround)       : {fast:.3f} s   metric_phase={metric_ms('rs_mfreq200')} ms")
    print(f"  official LightGBM CUDA (in-session reference)   : {off:.3f} s")
    print(f"  >>> default-vs-workaround delta: {base - fast:+.3f} s "
          f"(EXPECT ~0 — the fix removed the per-iter metric by default) <<<")
    print(f"  gap (default lgb_rs / official): {base/off:.2f}x")


if __name__ == "__main__":
    main()
