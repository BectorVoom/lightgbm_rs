#!/usr/bin/env python3
"""Spike-046 / 048 — instrumented Kaggle CUDA bottleneck benchmark.

Goal: ATTRIBUTE lightgbm_rs's ~6x CUDA slowdown (500k x 50, 100 trees) into
per-phase costs on REAL discrete NVIDIA hardware, and prove whether routing this
narrow shape to lightgbm_rs's own CPU anchor is faster.

What this captures that the prior run did NOT:
  1. LGBM_PHASE_PROF=1 breakdown for the SHIPPED Python path (spike-046 added the
     dump("train") hook to booster.rs::train_inner_columns_full — without it the
     Python wheel emits ZERO attribution).
  2. lightgbm_rs CPU vs CUDA at the same shape (the "route narrow to CPU" fix).
  3. Official LightGBM CPU + CUDA as references.
  4. COUNTS line = device launches + scan round-trip syncs/tree (the per-leaf
     sync-floor hypothesis: cheap on the dev APU's shared DDR5, expensive on
     discrete PCIe).

PREREQUISITE: the spike-046 dump("train") patch must be pushed to the GitHub repo
this clones (set REPO_BRANCH if it lives on a branch other than the default).

Kaggle setup: kernel_type=script, enable_gpu=true, enable_internet=true.
"""
import os
import subprocess
import sys

REPO_URL = os.environ.get("REPO_URL", "https://github.com/BectorVoom/lightgbm_rs.git")
REPO_BRANCH = os.environ.get("REPO_BRANCH", "")  # e.g. "master"; empty = default branch

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def run(cmd, check=True):
    print(f"\n$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check)


def run_bench(which, device_type, env_extra=None):
    """Run one backend in its own process; return (stdout, stderr)."""
    env = dict(os.environ)
    env["LGBM_PHASE_PROF"] = "1"
    if env_extra:
        env.update(env_extra)
    print(f"\n===== BENCH {which}/{device_type} =====", flush=True)
    res = subprocess.run(
        f"python3 {SPIKE_DIR}/bench_runner.py {which} {device_type}",
        shell=True, capture_output=True, text=True, env=env,
    )
    print("--- stdout ---\n" + res.stdout, flush=True)
    print("--- stderr (phase_prof + warnings) ---\n" + res.stderr, flush=True)
    return res.stdout, res.stderr


# --- 0. locate the spike dir (bench_runner.py lives beside the cloned tree) ---
def main():
    global SPIKE_DIR
    print("Setting up toolchain...")
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
    # Confirm the spike-046 patch is present in the clone (fail loud if not pushed).
    grep = subprocess.run(
        'grep -n "Spike-046" lightgbm_rs/crates/lgbm/src/booster.rs',
        shell=True, capture_output=True, text=True,
    )
    if "Spike-046" not in grep.stdout:
        print("!! WARNING: spike-046 dump() patch NOT found in clone — "
              "phase_prof will print nothing. Push the patch (or set REPO_BRANCH).")
    else:
        print(f"spike-046 patch present: {grep.stdout.strip()}")

    results = {}
    stderrs = {}

    # --- 1. lightgbm_rs CPU wheel (default features — fast build, cubecl-cpu only) ---
    print("\n########## BUILD lightgbm_rs CPU wheel (default features) ##########")
    run("rm -rf lightgbm_rs/target/wheels/")
    run("cd lightgbm_rs/crates/lgbm-python && maturin build --release -j 2")
    run("pip install $(ls lightgbm_rs/target/wheels/*.whl | head -n 1) --force-reinstall")
    o, e = run_bench("rs", "cpu")
    results["rs_cpu"], stderrs["rs_cpu"] = o, e

    # --- 2. lightgbm_rs CUDA wheel (-F cuda) ---
    print("\n########## BUILD lightgbm_rs CUDA wheel (-F cuda) ##########")
    run("rm -rf lightgbm_rs/target/wheels/")
    run("cd lightgbm_rs/crates/lgbm-python && maturin build --release -F cuda -j 2")
    run("pip install $(ls lightgbm_rs/target/wheels/*.whl | head -n 1) --force-reinstall")
    o, e = run_bench("rs", "cuda")
    results["rs_cuda"], stderrs["rs_cuda"] = o, e

    # --- 3. official LightGBM (ensure CUDA build), CPU + CUDA references ---
    print("\n########## official LightGBM references ##########")
    try:
        run("python3 -c 'import lightgbm as lgb, numpy as np; "
            "lgb.LGBMClassifier(device_type=\"cuda\", num_leaves=2, n_estimators=1)"
            ".fit(np.zeros((10,2)), np.zeros(10))'")
    except subprocess.CalledProcessError:
        print("Installing official lightgbm with USE_CUDA=ON...")
        run("pip uninstall -y lightgbm", check=False)
        run("pip install --no-binary lightgbm lightgbm -C cmake.define.USE_CUDA=ON")
    o, _ = run_bench("off", "cpu")
    results["off_cpu"] = o
    o, _ = run_bench("off", "cuda")
    results["off_cuda"] = o

    # --- 4. summary ---
    print("\n\n==================== SUMMARY ====================")
    def t(key):
        for line in results.get(key, "").splitlines():
            if line.startswith("RESULT"):
                return line.split("train_time_s=")[-1]
        return "?"
    print(f"  lightgbm_rs   CPU : {t('rs_cpu')} s")
    print(f"  lightgbm_rs   CUDA: {t('rs_cuda')} s")
    print(f"  official LGBM CPU : {t('off_cpu')} s")
    print(f"  official LGBM CUDA: {t('off_cuda')} s")
    print("\n  --- lightgbm_rs CUDA phase_prof (the attribution) ---")
    for line in stderrs.get("rs_cuda", "").splitlines():
        if "phase_prof" in line:
            print("   " + line)
    print("\n  --- lightgbm_rs CPU phase_prof (same shape, for contrast) ---")
    for line in stderrs.get("rs_cpu", "").splitlines():
        if "phase_prof" in line:
            print("   " + line)


if __name__ == "__main__":
    main()
