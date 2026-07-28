# lightgbm-rs

Python bindings for **LightGBM-rs** — a pure-Rust reimplementation of Microsoft's
LightGBM gradient-boosting library, numerically faithful to the C++ reference
(~1e-6 absolute on the CPU anchor) and built on [PyO3](https://pyo3.rs).

The package mirrors the official `lightgbm` low-level surface for the in-scope
APIs, so you can switch `import lightgbm` → `import lightgbm_rs`:

```python
import numpy as np
import lightgbm_rs as lgb

X = np.random.standard_normal((1000, 10))
y = X @ np.random.standard_normal(10)

ds = lgb.Dataset(X, y)
model = lgb.train({"objective": "regression", "num_leaves": 31}, ds, num_boost_round=100)
pred = model.predict(X)  # owned numpy array
```

Input is marshalled into owned Rust buffers and the GIL is released
(`Python::detach`) around the CPU-bound train/predict, so background Python
threads make progress during training.

## Building from source

```bash
pip install maturin
maturin develop --release   # inside this directory, into your active venv
```

See the workspace root for the full project, license, and the numerical-parity
contract.

## Known issue: `ImportError: cannot allocate memory in static TLS block`

`_core.abi3.so` embeds LLVM (used by the cubecl-cpu JIT backend), which
declares many `thread_local` globals. On some glibc builds this exceeds the
small "static TLS surplus" glibc reserves for shared objects loaded late via
`dlopen()` (which is how Python imports extension modules), and the import
fails with this error. It depends on your glibc version and what else is
already loaded in the process — it is **not** tied to any specific Python
version (reproduces on 3.13 and 3.14 alike on a recent glibc; the package's
0.0.5 release predates the embedded-LLVM backend and is unaffected).

Confirmed workaround (glibc >= 2.35, i.e. most current distros): raise the
static TLS surplus before starting Python:

```bash
GLIBC_TUNABLES=glibc.rtld.optional_static_tls=4096 python -c "import lightgbm_rs"
```

or export `GLIBC_TUNABLES` in your shell profile / the environment your
process runs in. `2048` was sufficient in testing; `4096` leaves headroom.
The import error message includes this same guidance.
