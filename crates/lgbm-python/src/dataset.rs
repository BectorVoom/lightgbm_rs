//! `#[pyclass] Dataset` — an owned RAW training corpus built from a numpy dense
//! f64 matrix + labels (D-01: the binding bins raw values internally via the
//! bit-exact `BinMapper` inside the facade `train_raw`). This wrapper holds the
//! owned [`lgbm::RawCorpus`] and adds NO algorithm.

use lgbm::RawCorpus;
use numpy::{PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::marshal::{
    numpy_dense_f32_to_rows, numpy_dense_to_rows, numpy_labels_to_f32, polars_df_to_corpus,
    scipy_csc_to_rows, scipy_csr_to_rows,
};

/// Marshal a dense numpy matrix of EITHER `f32` or `f64` dtype into owned f64
/// rows (PYB-02 SC#2). Inspects the runtime dtype and downcasts to the matching
/// typed readonly array, routing both widths through the SINGLE widen site in
/// `marshal.rs` so f32 and f64 of the same data yield identical rows
/// (self-consistency, T-08-03-03). Any other dtype is a `ValueError`.
fn dense_any_to_rows(data: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
    if let Ok(arr) = data.cast::<numpy::PyArray2<f64>>() {
        let ro: PyReadonlyArray2<'_, f64> = arr.readonly();
        return numpy_dense_to_rows(&ro);
    }
    if let Ok(arr) = data.cast::<numpy::PyArray2<f32>>() {
        let ro: PyReadonlyArray2<'_, f32> = arr.readonly();
        return numpy_dense_f32_to_rows(&ro);
    }
    // Surface the offending dtype/shape rather than panicking (Security V5).
    let detail = data
        .cast::<numpy::PyUntypedArray>()
        .map(|a| format!("dtype {:?}, ndim {}", a.dtype(), a.ndim()))
        .unwrap_or_else(|_| "not a 2-D numpy array".to_string());
    Err(PyValueError::new_err(format!(
        "data must be a 2-D numpy array of dtype float32 or float64 (got {detail})"
    )))
}

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
    fn new(data: &Bound<'_, PyAny>, label: PyReadonlyArray1<'_, f64>) -> PyResult<Self> {
        let rows = dense_any_to_rows(data)?;
        let labels = numpy_labels_to_f32(&label)?;
        Self::from_rows(rows, labels)
    }

    /// Build a `Dataset` from a scipy **CSR** sparse matrix, mirroring the official
    /// `lightgbm.Dataset(csr_matrix, label)` path (D-05). The three scipy arrays
    /// `indptr` (int64), `indices` (int32), `data` (float32) plus the `shape`
    /// `(num_rows, num_cols)` are densified into the SAME owned f64 rows the dense
    /// path uses, so a CSR matrix and its dense equivalent train to the same model.
    ///
    /// Malformed `indptr`/`indices` are rejected as `ValueError` BEFORE any
    /// indexing (Security V5 / T-08-03-01). Coerce the scipy arrays on the Python
    /// side (`indptr.astype('int64')`, `indices.astype('int32')`,
    /// `data.astype('float32')`) — the thin `lightgbm_rs` wrapper does this.
    #[staticmethod]
    fn from_csr(
        indptr: PyReadonlyArray1<'_, i64>,
        indices: PyReadonlyArray1<'_, i32>,
        data: PyReadonlyArray1<'_, f32>,
        shape: (usize, usize),
        label: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        let (num_rows, num_cols) = shape;
        let rows = scipy_csr_to_rows(&indptr, &indices, &data, num_rows, num_cols)?;
        let labels = numpy_labels_to_f32(&label)?;
        Self::from_rows(rows, labels)
    }

    /// Build a `Dataset` from a scipy **CSC** sparse matrix (column-oriented mirror
    /// of [`Dataset::from_csr`], D-05). `indptr` has `num_cols + 1` entries;
    /// `indices` are ROW indices. Same Security-V5 validation and dense
    /// equivalence guarantee.
    #[staticmethod]
    fn from_csc(
        indptr: PyReadonlyArray1<'_, i64>,
        indices: PyReadonlyArray1<'_, i32>,
        data: PyReadonlyArray1<'_, f32>,
        shape: (usize, usize),
        label: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        let (num_rows, num_cols) = shape;
        let rows = scipy_csc_to_rows(&indptr, &indices, &data, num_rows, num_cols)?;
        let labels = numpy_labels_to_f32(&label)?;
        Self::from_rows(rows, labels)
    }

    /// Build a `Dataset` from a **polars DataFrame** (D-03/D-04). Columns are
    /// consumed Arrow-side in Rust (no numpy round-trip, which would erase the
    /// Categorical/Enum dtype). With `categorical_feature = None` (the default
    /// `'auto'`), `Categorical`/`Enum`/`String` columns route to LightGBM
    /// categorical features and numeric columns to numeric; a `Some(names)` list
    /// FULLY decides routing (official-package precedence). The frame is densified
    /// into the SAME owned f64 rows the numpy/sparse paths use (categorical columns
    /// carry their integer category codes), so a DataFrame and its numeric-array
    /// equivalent train to the same model.
    ///
    /// `categorical_feature` must contain column NAMES; the thin `lightgbm_rs`
    /// wrapper resolves integer indices to names before calling.
    #[staticmethod]
    #[pyo3(signature = (df, label, categorical_feature = None))]
    fn from_polars(
        df: PyDataFrame,
        label: PyReadonlyArray1<'_, f64>,
        categorical_feature: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let (rows, _names, cat_indices) = polars_df_to_corpus(df, categorical_feature)?;
        let labels = numpy_labels_to_f32(&label)?;
        Self::from_rows_with_categorical(rows, labels, cat_indices)
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

impl Dataset {
    /// Shared constructor tail for every ingest path (dense f32/f64, CSR, CSC):
    /// validate `label.len() == num_rows` at the boundary (Security V5,
    /// T-08-03-01) BEFORE building the owned [`RawCorpus`]. A mismatch raises
    /// `ValueError`; never panics.
    fn from_rows(rows: Vec<Vec<f64>>, labels: Vec<f32>) -> PyResult<Self> {
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

    /// Constructor tail for the polars path: same boundary validation as
    /// [`Dataset::from_rows`], then set `categorical_features` so
    /// `build_feature_columns_from_raw` routes those columns through
    /// `find_bin_categorical` (D-04).
    fn from_rows_with_categorical(
        rows: Vec<Vec<f64>>,
        labels: Vec<f32>,
        categorical_indices: Vec<usize>,
    ) -> PyResult<Self> {
        if labels.len() != rows.len() {
            return Err(PyValueError::new_err(format!(
                "label length {} != number of data rows {}",
                labels.len(),
                rows.len()
            )));
        }
        let mut corpus = RawCorpus::new(rows, labels);
        corpus.categorical_features = categorical_indices;
        Ok(Dataset { corpus })
    }
}
