"""lightgbm_rs — pure-Rust LightGBM with a numpy-native Python surface (D-11).

This thin pure-Python package re-exports the compiled extension
``lightgbm_rs._core`` (the PyO3 cdylib). It mirrors the official ``lightgbm``
package's low-level surface (``Dataset``, ``Booster``, ``train``) so existing
LightGBM code can switch ``import lightgbm`` -> ``import lightgbm_rs`` for the
in-scope APIs.
"""

from ._core import Booster, Dataset, LightGBMError
from ._core import train as _core_train
from .callback import (
    CallbackEnv,
    EarlyStopException,
    early_stopping,
    log_evaluation,
    record_evaluation,
    reset_parameter,
)
from .engine import CVBooster, cv
from .engine import train as train


def dataset_from_csr(csr, label):
    """Build a :class:`Dataset` from a scipy CSR matrix (D-05).

    Coerces the three scipy arrays to the widths the bit-exact ingest expects
    (``indptr`` -> int64, ``indices`` -> int32, ``data`` -> float32; RESEARCH A5)
    and delegates to :meth:`Dataset.from_csr`, which densifies into the SAME f64
    rows the dense path bins. ``csr`` is any scipy ``csr_matrix``/``csr_array``.
    """
    import numpy as np

    return Dataset.from_csr(
        np.ascontiguousarray(csr.indptr, dtype=np.int64),
        np.ascontiguousarray(csr.indices, dtype=np.int32),
        np.ascontiguousarray(csr.data, dtype=np.float32),
        (int(csr.shape[0]), int(csr.shape[1])),
        np.asarray(label, dtype=np.float64),
    )


def dataset_from_csc(csc, label):
    """Build a :class:`Dataset` from a scipy CSC matrix (D-05).

    Column-oriented mirror of :func:`dataset_from_csr`; same dtype coercion and
    dense-equivalence guarantee.
    """
    import numpy as np

    return Dataset.from_csc(
        np.ascontiguousarray(csc.indptr, dtype=np.int64),
        np.ascontiguousarray(csc.indices, dtype=np.int32),
        np.ascontiguousarray(csc.data, dtype=np.float32),
        (int(csc.shape[0]), int(csc.shape[1])),
        np.asarray(label, dtype=np.float64),
    )


def dataset_from_polars(df, label, categorical_feature="auto"):
    """Build a :class:`Dataset` from a polars ``DataFrame`` (D-03/D-04).

    Columns are consumed zero-copy Arrow-side in Rust (no numpy round-trip, which
    would erase the ``Categorical``/``Enum`` dtype). Routing follows the official
    package's ``categorical_feature`` semantics:

    - ``"auto"`` (default): dtype routing — ``Categorical``/``Enum``/``String``
      columns become categorical features, numeric columns numeric.
    - a list of column **names** and/or integer **indices**: that list fully
      decides which columns are categorical (precedence over dtype).

    ``label`` is any array-like coerced to float64.
    """
    import numpy as np

    if categorical_feature == "auto":
        names = None
    else:
        cols = list(df.columns)
        names = []
        for c in categorical_feature:
            if isinstance(c, int):
                names.append(cols[c])
            else:
                names.append(str(c))

    return Dataset.from_polars(df, np.asarray(label, dtype=np.float64), names)


__all__ = [
    "Booster",
    "Dataset",
    "LightGBMError",
    "train",
    "cv",
    "CVBooster",
    "CallbackEnv",
    "EarlyStopException",
    "early_stopping",
    "log_evaluation",
    "record_evaluation",
    "reset_parameter",
    "dataset_from_csr",
    "dataset_from_csc",
    "dataset_from_polars",
]
