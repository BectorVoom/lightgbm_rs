"""lightgbm_rs — pure-Rust LightGBM with a numpy-native Python surface (D-11).

This thin pure-Python package re-exports the compiled extension
``lightgbm_rs._core`` (the PyO3 cdylib). It mirrors the official ``lightgbm``
package's low-level surface (``Dataset``, ``Booster``, ``train``) so existing
LightGBM code can switch ``import lightgbm`` -> ``import lightgbm_rs`` for the
in-scope APIs.
"""

from ._core import Booster, Dataset, LightGBMError, train

__all__ = ["Booster", "Dataset", "LightGBMError", "train"]
