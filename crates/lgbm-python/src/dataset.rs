//! `#[pyclass] Dataset` — an owned RAW training corpus built from a numpy dense
//! f64 matrix + labels (D-01: the binding bins raw values internally via the
//! bit-exact `BinMapper` inside the facade `train_raw`). This wrapper holds the
//! owned [`lgbm::RawCorpus`] and adds NO algorithm.

use lgbm::RawCorpus;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::marshal::{numpy_dense_to_rows, numpy_labels_to_f32};

/// A LightGBM-rs training dataset over raw (arbitrary-valued) dense numpy input.
///
/// Mirrors the official `lightgbm.Dataset(data, label)` surface for the dense
/// numpy slice. The raw rows + f32 labels are owned; `train` bins them internally
/// via the facade `BinMapper` (D-01/D-02).
#[pyclass(module = "lightgbm_rs._core")]
pub struct Dataset {
    /// The owned raw corpus (raw rows + f32 labels + default binning config).
    pub(crate) corpus: RawCorpus,
}

#[pymethods]
impl Dataset {
    /// Build a `Dataset` from a dense numpy f64 matrix `data` (num_rows x
    /// num_features) and a 1-D `label` array. All marshalling copies into owned
    /// Rust buffers WHILE the GIL is held (the input is read here, never lent).
    ///
    /// Validates `label.len() == num_rows` at the boundary BEFORE any further work
    /// (Security V5, T-08-02-02); a mismatch raises `ValueError`.
    #[new]
    fn new(
        data: PyReadonlyArray2<'_, f64>,
        label: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        let rows = numpy_dense_to_rows(&data)?;
        let labels = numpy_labels_to_f32(&label)?;
        if labels.len() != rows.len() {
            return Err(PyValueError::new_err(format!(
                "label length {} != number of data rows {}",
                labels.len(),
                rows.len()
            )));
        }
        Ok(Dataset {
            corpus: RawCorpus::new(rows, labels),
        })
    }

    /// Number of rows in the dataset (mirrors `Dataset.num_data()`).
    fn num_data(&self) -> usize {
        self.corpus.features.len()
    }

    /// Number of features (columns) in the dataset (mirrors
    /// `Dataset.num_feature()`).
    fn num_feature(&self) -> usize {
        self.corpus.features.first().map_or(0, Vec::len)
    }
}
