//! The public [`Booster`] — the trained ensemble + eval history, plus the
//! `train` / `predict` entry points.
//!
//! `train` drives the full spine: builder→Config → identity-binned feature
//! columns → [`lgbm_boosting::Gbdt`] loop (BoostFromAverage → GetGradients →
//! per-class tree → Shrinkage → UpdateScore → AddBias) → [`GbdtModel`]. `predict`
//! delegates to [`GbdtModel::predict_raw`] then the predict-side
//! [`lgbm_model::ObjectiveKind::convert`] transform (identity for regression).

use lgbm_boosting::objective::BoostObjective;
use lgbm_boosting::{Gbdt, IterSnapshot};
use lgbm_compute::gain::GainConfig;
// Backend dispatch (feature-switched). Default: the native-f64 CpuBackend (the
// bit-exact anchor). With `--features rocm`: RocmBackend (SAME f64 kernels on the
// local gfx1100 GPU). With `--features cuda`/`wgpu`: CudaBackend/WgpuBackend, which
// dispatch the SAME runtime-generic GPU kernels. The cascade priority is
// rocm > cuda > wgpu > cpu, so any combination of enabled features selects exactly
// one backend (the `not(...)` guards make the arms mutually exclusive).
#[cfg(feature = "rocm")]
use lgbm_compute::runtime::rocm_client;
#[cfg(feature = "rocm")]
use lgbm_compute::RocmBackend;
#[cfg(all(feature = "cuda", not(feature = "rocm")))]
use lgbm_compute::runtime::cuda_client;
#[cfg(all(feature = "cuda", not(feature = "rocm")))]
use lgbm_compute::CudaBackend;
#[cfg(all(feature = "wgpu", not(feature = "rocm"), not(feature = "cuda")))]
use lgbm_compute::runtime::wgpu_client;
#[cfg(all(feature = "wgpu", not(feature = "rocm"), not(feature = "cuda")))]
use lgbm_compute::WgpuBackend;
#[cfg(not(any(feature = "rocm", feature = "cuda", feature = "wgpu")))]
use lgbm_compute::runtime::cpu_client;
#[cfg(not(any(feature = "rocm", feature = "cuda", feature = "wgpu")))]
use lgbm_compute::CpuBackend;
use lgbm_core::Config;
use lgbm_dataset::bin_mapper::{BinMapper, MissingType};
use lgbm_metric::{BinaryMetric, Metric, MultiLogloss};
use lgbm_model::{GbdtModel, ObjectiveKind};
use lgbm_objective::{
    Binary, CustomObjective, MulticlassOva, MulticlassSoftmax, Objective, Xentropy,
};
use lgbm_treelearner::learner::FeatureColumn;
use lgbm_treelearner::BinColumn;
use lgbm_treelearner::{offset_for_most_freq_bin, SerialTreeLearner};

use crate::error::LgbmError;

/// One configured eval metric — either a regression metric (raw-score) or a
/// binary metric (prob-space / AUC). Unifies the per-round eval-history loop
/// across objectives.
/// A user-supplied custom-metric (feval) closure: `(raw_scores, labels) ->
/// (name, value, is_higher_better)`, mirroring the C++ `_EvalFunctionWrapper`
/// contract the Python `feval` marshals into. The closure is called once
/// per eval cadence with the SAME `(scores, labels)` the built-in metrics see,
/// so it feeds the SAME eval-history loop.
pub type CustomMetricClosure = Box<dyn Fn(&[f64], &[f32]) -> (String, f64, bool) + Send>;

enum EvalMetric {
    /// A regression metric (`l1`/`l2`/`rmse`) over the raw score.
    Reg(Metric),
    /// A binary metric (`binary_logloss`/`binary_error`/`auc`).
    Bin(BinaryMetric),
    /// The `multi_logloss` metric over the class-major score buffer.
    Multi(MultiLogloss),
    /// A user custom-metric (feval) closure. Its `name()` and value come
    /// from the closure; it feeds the SAME eval-history loop as the built-ins.
    Custom(CustomMetricClosure),
}

impl EvalMetric {
    fn name(&self) -> String {
        match self {
            EvalMetric::Reg(m) => m.name().to_string(),
            EvalMetric::Bin(m) => m.name().to_string(),
            EvalMetric::Multi(m) => m.name().to_string(),
            // The custom-metric name is dynamic; resolve it by calling the closure
            // with empty inputs is NOT valid, so we record the name lazily during
            // eval. For the history-key setup we expose a stable placeholder that
            // the first eval overwrites. To keep the name stable we instead cache
            // it on first construction is not possible (closure is opaque), so the
            // caller (train path) uses the closure-returned name from eval.
            EvalMetric::Custom(_) => "custom".to_string(),
        }
    }

    fn eval(&self, scores: &[f64], labels: &[f32]) -> Result<f64, LgbmError> {
        match self {
            EvalMetric::Reg(m) => m.eval(scores, labels).map_err(LgbmError::Metric),
            EvalMetric::Bin(m) => m.eval(scores, labels).map_err(LgbmError::Metric),
            EvalMetric::Multi(m) => m.eval(scores, labels).map_err(LgbmError::Metric),
            EvalMetric::Custom(f) => {
                let (_name, value, _higher) = f(scores, labels);
                // A NaN / non-finite custom value is surfaced as a typed error,
                // never silently recorded.
                if !value.is_finite() {
                    return Err(LgbmError::CustomMetric {
                        detail: "custom metric (feval) returned a non-finite value".into(),
                    });
                }
                Ok(value)
            }
        }
    }

    /// The closure-supplied metric name (for `Custom`) or the built-in name. For
    /// `Custom` this calls the closure with the actual eval inputs so the recorded
    /// history key matches the user's name.
    fn resolved_name(&self, scores: &[f64], labels: &[f32]) -> String {
        match self {
            EvalMetric::Custom(f) => f(scores, labels).0,
            other => other.name(),
        }
    }

    /// C++ `factor_to_bigger_better` — `+1` for AUC, `-1` for the losses. Drives
    /// the early-stopping comparison direction.
    fn factor_to_bigger_better(&self) -> f64 {
        match self {
            EvalMetric::Reg(m) => m.factor_to_bigger_better(),
            EvalMetric::Bin(m) => m.factor_to_bigger_better(),
            // multi_logloss is a loss: lower is better.
            EvalMetric::Multi(_) => -1.0,
            // The custom metric's direction is dynamic; default to "lower is
            // better" for the ES factor (the value-recording path is unaffected).
            EvalMetric::Custom(_) => -1.0,
        }
    }
}

/// A dense, identity-binned training corpus: raw integer-valued features (one
/// column per feature, bin index == raw value) + f32 labels.
///
/// This is the identity-binned spine training input — it mirrors the capture's
/// identity binning (distinct consecutive integers `0..K-1` per feature) so the
/// Rust-grown trees are bit-comparable to the real-binary goldens. Arbitrary-value
/// binning is handled by the `RawCorpus`/`BinMapper` path below.
#[derive(Debug, Clone)]
pub struct DenseCorpus {
    /// Row-major raw feature values, `num_data` rows × `num_features` columns.
    pub features: Vec<Vec<f64>>,
    /// Per-row labels (length `num_data`).
    pub labels: Vec<f32>,
}

/// LightGBM's real per-bin upper bound for an identity-binned integer feature:
/// the `(b + 0.5)` midpoint nudged up by exactly one ULP — LightGBM's
/// `GetDoubleUpperBound`, matching the `2.5000000000000004` /
/// `1.5000000000000002` values the capture emits (verified against
/// `learner_parity::real_upper_bounds`).
fn real_upper_bounds(num_bin: u32) -> Vec<f64> {
    (0..num_bin)
        .map(|b| {
            let mid = b as f64 + 0.5;
            f64::from_bits(mid.to_bits() + 1) // mid + 1 ULP (mid > 0)
        })
        .collect()
}

/// Build identity-binned [`FeatureColumn`]s from a dense corpus. Each feature's
/// distinct raw values must be the consecutive integers `0..K-1` (the
/// identity-binning precondition); `num_bin = K`, `bin == raw value`,
/// `most_freq_bin = the modal bin`, and `bin_upper_bound = real_upper_bounds(K)`.
///
/// # Errors
/// [`LgbmError::InvalidCorpus`] when a column is not identity-binnable (a raw
/// value is negative or non-integer, or the distinct values are not `0..K-1`).
fn build_feature_columns(corpus: &DenseCorpus) -> Result<Vec<FeatureColumn>, LgbmError> {
    let num_data = corpus.features.len();
    if num_data == 0 {
        return Err(LgbmError::InvalidCorpus {
            detail: "empty corpus (no rows)".into(),
        });
    }
    let num_features = corpus.features[0].len();
    if corpus.labels.len() != num_data {
        return Err(LgbmError::InvalidCorpus {
            detail: format!(
                "labels length {} != num_data {num_data}",
                corpus.labels.len()
            ),
        });
    }
    // ONE cache-friendly pass over rows (each `row` is contiguous), SCATTERING
    // `row[j] as u32` into per-feature bin vectors — instead of `num_features`
    // STRIDED column passes. The resulting bins are byte-identical regardless of
    // pass order (same values, same per-feature row order) ⇒ FeatureColumns
    // byte-identical ⇒ bit-exact. Validation runs in row-major discovery order —
    // same `LgbmError::InvalidCorpus` variant + same checks (only `is_err()` is
    // asserted, booster.rs test, not the specific (feature,row)).
    let mut cols_bins: Vec<Vec<u32>> =
        (0..num_features).map(|_| Vec::with_capacity(num_data)).collect();
    for (i, row) in corpus.features.iter().enumerate() {
        if row.len() != num_features {
            return Err(LgbmError::InvalidCorpus {
                detail: format!("row {i} has {} cols, expected {num_features}", row.len()),
            });
        }
        for (j, col) in cols_bins.iter_mut().enumerate() {
            let v = row[j];
            if v < 0.0 || v.fract() != 0.0 {
                return Err(LgbmError::InvalidCorpus {
                    detail: format!(
                        "feature {j} row {i} value {v} is not a non-negative integer \
                         (identity binning requires consecutive integers 0..K-1)"
                    ),
                });
            }
            col.push(v as u32);
        }
    }

    let mut columns = Vec::with_capacity(num_features);
    for (j, bins) in cols_bins.into_iter().enumerate() {
        // num_bin = max bin + 1; verify distinct values are exactly 0..num_bin-1.
        let max_bin = *bins.iter().max().unwrap();
        let num_bin = max_bin + 1;
        let mut seen = vec![false; num_bin as usize];
        for &b in &bins {
            seen[b as usize] = true;
        }
        if !seen.iter().all(|&s| s) {
            return Err(LgbmError::InvalidCorpus {
                detail: format!(
                    "feature {j} distinct bins are not the consecutive integers \
                     0..{num_bin}-1 (identity binning precondition)"
                ),
            });
        }
        // most_freq_bin = the modal bin (ties broken to the lowest, matching the
        // capture's expectation).
        let mut counts = vec![0i32; num_bin as usize];
        for &b in &bins {
            counts[b as usize] += 1;
        }
        let most_freq_bin = counts
            .iter()
            .enumerate()
            .max_by_key(|&(idx, &c)| (c, std::cmp::Reverse(idx)))
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0);
        columns.push(FeatureColumn {
            bins: BinColumn::new(bins, num_bin),
            num_bin,
            offset: offset_for_most_freq_bin(most_freq_bin),
            min_bin: 0,
            max_bin,
            default_bin: num_bin,
            most_freq_bin,
            missing_type: MissingType::None,
            bin_upper_bound: real_upper_bounds(num_bin),
            real_feature_index: j as i32,
            // This identity-binned facade builds NUMERIC features only; the
            // categorical path is driven via the parity harness / builder.
            // bin_type defaults to Numerical so the spine is unchanged.
            ..FeatureColumn::default()
        });
    }
    Ok(columns)
}

/// A dense training corpus of RAW (arbitrary real-valued) features — the
/// raw→bin→train input. Unlike [`DenseCorpus`] (which requires identity-binnable
/// integer columns), `RawCorpus` carries arbitrary continuous (or categorical)
/// values that are binned internally by the bit-exact [`BinMapper`]
/// (`find_bin_from_column` / `find_bin_categorical`) before training.
///
/// This is the slice the Python wrapper depends on: a Python user passes a numpy
/// matrix of real values, which marshals into a `RawCorpus`; the facade bins it
/// via the same `BinMapper` the parity harness proves bit-exact, so the
/// raw→bin→train path reproduces the identity-bin / C++ golden output.
#[derive(Debug, Clone)]
pub struct RawCorpus {
    /// COLUMN-MAJOR flat RAW feature values (O1/O2): `value(row, col)` lives at
    /// `col_major[col * num_rows + row]`. Was a row-major `Vec<Vec<f64>>` before
    /// O1/O2 — flattened + transposed so the Python/Arrow ingest writes columns
    /// directly ([`from_columns`](RawCorpus::from_columns)) and binning reads each
    /// feature as a contiguous slice ([`column`](RawCorpus::column)), eliminating
    /// the old col→row→col double transpose. Private: access via the methods so the
    /// `col_major.len() == num_rows * num_cols` rectangularity invariant holds.
    col_major: Vec<f64>,
    /// Number of rows (`num_data`).
    num_rows: usize,
    /// Number of features (columns).
    num_cols: usize,
    /// Per-row labels (length `num_data`).
    pub labels: Vec<f32>,
    /// Real-feature indices (columns) to bin as CATEGORICAL via
    /// [`BinMapper::find_bin_categorical`]. Empty ⇒ all numeric.
    pub categorical_features: Vec<usize>,
    /// The binning + training [`Config`] (`max_bin`, `min_data_in_bin`,
    /// `bin_construct_sample_cnt`, `data_random_seed`, `use_missing`,
    /// `zero_as_missing`, `min_data_in_leaf`). Defaults to [`Config::default`].
    pub config: Config,
}

impl RawCorpus {
    /// Construct from ROW-MAJOR `rows` (`num_data` rows × `num_features` cols) — the
    /// numpy-dense / scipy-CSR / scipy-CSC ingest paths, which produce row-major
    /// data. Transposes into the column-major store ONCE here (the single transpose
    /// those row-major inputs require; it replaces the old per-column gather inside
    /// [`build_feature_columns_from_raw`]). All-numeric, default [`Config`]; set
    /// `categorical_features` / `config` on the returned value.
    ///
    /// A RAGGED `rows` (rows of differing length) leaves `col_major` empty so the
    /// `col_major.len() == num_rows * num_cols` invariant fails and
    /// [`build_feature_columns_from_raw`] surfaces a typed
    /// [`LgbmError::InvalidCorpus`] (never panics).
    pub fn new(rows: Vec<Vec<f64>>, labels: Vec<f32>) -> Self {
        let num_rows = rows.len();
        let num_cols = rows.first().map_or(0, Vec::len);
        let rectangular = rows.iter().all(|r| r.len() == num_cols);
        let col_major = if rectangular {
            let mut cm = vec![0.0f64; num_rows * num_cols];
            for (i, row) in rows.iter().enumerate() {
                for (j, &v) in row.iter().enumerate() {
                    cm[j * num_rows + i] = v;
                }
            }
            cm
        } else {
            // Sentinel: empty buffer with non-zero dims ⇒ invariant fails ⇒ ragged
            // surfaces as a typed error in build_feature_columns_from_raw.
            Vec::new()
        };
        Self {
            col_major,
            num_rows,
            num_cols,
            labels,
            categorical_features: Vec::new(),
            config: Config::default(),
        }
    }

    /// Construct directly from COLUMN-MAJOR `columns` (`columns[j]` = feature `j`'s
    /// per-row values) — the polars/Arrow ingest path, where the source is already
    /// columnar. Stores the columns VERBATIM with NO transpose (O1). All columns
    /// must have the same length (= `num_rows`); a mismatch leaves the invariant
    /// failing so [`build_feature_columns_from_raw`] returns a typed error.
    /// All-numeric, default [`Config`]; set `categorical_features` / `config` after.
    pub fn from_columns(columns: Vec<Vec<f64>>, labels: Vec<f32>) -> Self {
        let num_cols = columns.len();
        let num_rows = columns.first().map_or(0, Vec::len);
        let rectangular = columns.iter().all(|c| c.len() == num_rows);
        let mut col_major = Vec::with_capacity(num_rows * num_cols);
        if rectangular {
            for col in &columns {
                col_major.extend_from_slice(col);
            }
        }
        Self {
            col_major,
            num_rows,
            num_cols,
            labels,
            categorical_features: Vec::new(),
            config: Config::default(),
        }
    }

    /// Number of rows (`num_data`).
    pub fn num_data(&self) -> usize {
        self.num_rows
    }

    /// Number of features (columns).
    pub fn num_features(&self) -> usize {
        self.num_cols
    }

    /// `true` when the flat store is rectangular (`col_major.len() == rows*cols`).
    /// `false` flags a ragged/mismatched construction (see [`new`](RawCorpus::new)).
    fn is_rectangular(&self) -> bool {
        self.col_major.len() == self.num_rows * self.num_cols
    }

    /// Contiguous slice of feature column `j` (O2: binning reads this directly with
    /// no per-row gather). Caller must ensure rectangularity (checked upstream in
    /// [`build_feature_columns_from_raw`]).
    pub fn column(&self, j: usize) -> &[f64] {
        let start = j * self.num_rows;
        &self.col_major[start..start + self.num_rows]
    }

    /// Value at (`row`, `col`).
    pub fn value(&self, row: usize, col: usize) -> f64 {
        self.col_major[col * self.num_rows + row]
    }

    /// Materialise the row-major `Vec<Vec<f64>>` view (the col→row transpose) for
    /// the few consumers that still want rows (e.g. batch predict in tests/benches).
    /// NOT on the ingest hot path — kept off it deliberately.
    pub fn to_rows(&self) -> Vec<Vec<f64>> {
        (0..self.num_rows)
            .map(|i| (0..self.num_cols).map(|j| self.value(i, j)).collect())
            .collect()
    }
}

/// Build [`FeatureColumn`]s from a RAW corpus by binning each column with the
/// bit-exact [`BinMapper`]. This is the raw→bin→train bridge: it mirrors
/// the EXACT `FeatureColumn { .. }` construction shape of [`build_feature_columns`]
/// but drives every bin-layout field from a per-column `BinMapper` instead of the
/// identity-binning assumption. The authoritative
/// [`offset_for_most_freq_bin`](lgbm_treelearner::offset_for_most_freq_bin) helper
/// is REUSED unchanged (the single offset source).
///
/// The per-column mapper is built via [`BinMapper::find_bin_from_column`] (numeric)
/// or [`BinMapper::find_bin_categorical`] (categorical), then each row's raw value
/// is mapped with [`BinMapper::value_to_bin`].
///
/// # Errors
/// [`LgbmError::InvalidCorpus`] for an empty corpus, a labels/num_data mismatch,
/// a ragged row, or a non-finite categorical index — validated BEFORE any binning.
pub fn build_feature_columns_from_raw(
    corpus: &RawCorpus,
) -> Result<Vec<FeatureColumn>, LgbmError> {
    // Back-compat shim: bin with the corpus's OWN config. The train paths instead call
    // `_with_config` with the TRAINING config so a `config` passed to `train_raw`/`train`
    // is authoritative for binning (max_bin/min_data_in_bin), not silently shadowed by the
    // `RawCorpus::new` default.
    build_feature_columns_from_raw_with_config(corpus, &corpus.config)
}

/// As [`build_feature_columns_from_raw`] but binning uses the EXPLICIT `bin_config`
/// (max_bin, min_data_in_bin, use_missing, …) rather than `corpus.config`. Categorical
/// feature indices still come from the corpus. The train facade passes its training config
/// here so binning honors the config the caller actually supplied.
pub fn build_feature_columns_from_raw_with_config(
    corpus: &RawCorpus,
    bin_config: &Config,
) -> Result<Vec<FeatureColumn>, LgbmError> {
    // ---- shape validation BEFORE any binning / indexing ----
    let num_data = corpus.num_data();
    if num_data == 0 {
        return Err(LgbmError::InvalidCorpus {
            detail: "empty corpus (no rows)".into(),
        });
    }
    let num_features = corpus.num_features();
    if num_features == 0 {
        return Err(LgbmError::InvalidCorpus {
            detail: "corpus row 0 has no features".into(),
        });
    }
    if corpus.labels.len() != num_data {
        return Err(LgbmError::InvalidCorpus {
            detail: format!(
                "labels length {} != num_data {num_data}",
                corpus.labels.len()
            ),
        });
    }
    // Rectangularity: a ragged/mismatched construction leaves the flat store sized
    // wrong (col_major.len() != num_data*num_features). Mirrors the old per-row
    // length CHECK; surfaces as a typed error, never a panic in column()/value().
    if !corpus.is_rectangular() {
        return Err(LgbmError::InvalidCorpus {
            detail: format!(
                "ragged corpus: feature store has {} values, expected num_data*num_features = {}",
                corpus.col_major.len(),
                num_data * num_features
            ),
        });
    }
    for &c in &corpus.categorical_features {
        if c >= num_features {
            return Err(LgbmError::InvalidCorpus {
                detail: format!(
                    "categorical feature index {c} out of range (num_features {num_features})"
                ),
            });
        }
    }

    let cfg = bin_config;
    // The pre-filter threshold is OFF by default (no min_data_in_leaf filtering of
    // bins) for the facade — matching the in-memory sample path. The BinMapper
    // builders take care of scaling internally.
    let pre_filter = false;
    // Each feature's BinMapper construction + bin assignment is INDEPENDENT, so bin
    // all features in parallel (matches C++ LightGBM's OpenMP-over-features binning).
    // Order-preserving: `map(j).collect()` keeps `columns[j]` == feature j, and each
    // BinMapper is per-feature deterministic (fixed `data_random_seed`) ⇒ BIT-EXACT vs
    // the serial path. Env var LGBM_PAR_BIN=0 forces the serial path.
    let bin_feature = |j: usize| -> FeatureColumn {
        // Read the raw column as a CONTIGUOUS slice (O2 — no per-row gather; the
        // column-major store already lays feature j out contiguously).
        let column: &[f64] = corpus.column(j);

        // Build the per-column BinMapper (Route A).
        let mapper: BinMapper = if corpus.categorical_features.contains(&j) {
            BinMapper::find_bin_categorical(
                column.to_vec(),
                cfg.max_bin,
                cfg.min_data_in_bin,
                cfg.min_data_in_leaf,
                pre_filter,
                cfg.use_missing,
                cfg.zero_as_missing,
                column.len(),
            )
        } else {
            BinMapper::find_bin_from_column(
                column,
                cfg.max_bin,
                cfg.min_data_in_bin,
                cfg.min_data_in_leaf,
                pre_filter,
                cfg.use_missing,
                cfg.zero_as_missing,
                cfg.bin_construct_sample_cnt,
                cfg.data_random_seed,
                &[],
            )
        };

        // Per-row raw→bin via the mapper (bin_mapper.rs:148).
        let bins: Vec<u32> = column.iter().map(|&v| mapper.value_to_bin(v)).collect();

        let num_bin = mapper.num_bin_ as u32;
        let most_freq_bin = mapper.most_freq_bin_;
        let bin_type = mapper.bin_type_;
        // bin_2_categorical_ maps bin → original category value (BinToValue);
        // carried so a categorical split can rebuild its cat_threshold bitset.
        let bin_to_category = mapper.bin_2_categorical_.clone();

        FeatureColumn {
            bins: BinColumn::new(bins, num_bin),
            num_bin,
            offset: offset_for_most_freq_bin(most_freq_bin),
            // Single-feature group: the feature owns bins [0, num_bin-1].
            min_bin: 0,
            max_bin: num_bin.saturating_sub(1),
            default_bin: mapper.default_bin_,
            most_freq_bin,
            missing_type: mapper.missing_type_,
            bin_upper_bound: mapper.bin_upper_bound_.clone(),
            real_feature_index: j as i32,
            bin_type,
            bin_to_category,
        }
    };

    let par = std::env::var("LGBM_PAR_BIN").map(|v| v != "0").unwrap_or(true);
    let columns: Vec<FeatureColumn> = if par {
        use rayon::prelude::*;
        (0..num_features).into_par_iter().map(bin_feature).collect()
    } else {
        (0..num_features).map(bin_feature).collect()
    };
    Ok(columns)
}

/// Train a [`Booster`] from a RAW corpus (the raw→bin→train entry point).
/// Bins each feature with the bit-exact [`BinMapper`] via
/// [`build_feature_columns_from_raw`], then drives the SAME `train_inner_full`
/// consumer used by [`train`]. The objective/metric resolution mirrors `train`.
///
/// # Errors
/// [`LgbmError`] for an invalid corpus, an unsupported objective/metric, or a
/// loop/learner failure — never a panic.
pub fn train_raw(config: &Config, corpus: &RawCorpus) -> Result<Booster, LgbmError> {
    let first = config.objective.split_whitespace().next().unwrap_or("");
    // Resolve the objective over a thin DenseCorpus view (labels only are used by
    // resolve_objective's label guards; the features are NOT read there, so the
    // view carries an empty feature matrix — no col→row transpose).
    let label_view = DenseCorpus {
        features: Vec::new(),
        labels: corpus.labels.clone(),
    };
    let (boost_obj, transformed_labels) = resolve_objective(config, &label_view)?;
    let metrics = eval_metrics_for(first, config);
    // Wrap the RAW→bin step in BINNING_NS so phase_prof attributes it correctly
    // (previously only the DenseCorpus path's build_feature_columns was wrapped).
    let features = lgbm_treelearner::phase_prof::time(
        &lgbm_treelearner::phase_prof::BINNING_NS,
        || build_feature_columns_from_raw_with_config(corpus, config),
    )?;
    let raw_features = raw_matrix_from_columns(corpus, config);
    train_inner_columns(
        config,
        corpus.num_data() as i32,
        feature_infos_from_columns(corpus),
        &corpus.labels,
        features,
        boost_obj,
        transformed_labels,
        metrics,
        raw_features,
    )
}

/// The trained ensemble + eval history. Mirrors the Python `Booster`'s
/// `best_iteration_` / `best_score_` / `record_evaluation` surface.
#[derive(Debug, Clone)]
pub struct Booster {
    /// The serializable ensemble container.
    model: GbdtModel,
    /// The parsed predict-side objective transform.
    objective_kind: ObjectiveKind,
    /// C++ `best_iteration_` (1-based round).
    pub best_iteration: i32,
    /// Per-metric eval history (metric name → per-round value), mirroring Python
    /// `record_evaluation` / `evals_result_`.
    pub eval_history: Vec<(String, Vec<f64>)>,
    /// The per-iteration L2 raw-score snapshots (the internal `score_` after each
    /// iter) — exposed for golden-replay verification.
    pub iter_scores: Vec<Vec<f64>>,
    /// The per-iteration L1 grad/hess snapshots — exposed for the L1 golden replay.
    pub iter_grad_hess: Vec<(Vec<f32>, Vec<f32>)>,
}

impl Booster {
    /// The trained ensemble (for model-text serialization / inspection).
    pub fn model(&self) -> &GbdtModel {
        &self.model
    }

    /// Transformed prediction for one raw feature row (width `max_feature_idx +
    /// 1`): `GbdtModel::predict_raw` then the objective's `ConvertOutput`
    /// (identity for regression). Returns a `num_tree_per_iteration`-wide f32
    /// vector.
    pub fn predict_row(&self, features: &[f64]) -> Vec<f32> {
        let mut raw = self.model.predict_raw(features, 0, -1);
        // RF (average_output) divides the per-tree SUM by num_iteration BEFORE the
        // ConvertOutput (C++ gbdt_prediction.cpp:57-59: the `average_output_` block
        // runs in `Predict` after `PredictRaw`). The stored leaf values are the RAW
        // per-tree outputs; averaging happens at predict time.
        if self.model.average_output {
            let (_start, num) = self.model.init_predict(0, -1);
            if num > 0 {
                for v in &mut raw {
                    *v /= num as f64;
                }
            }
        }
        let mut transformed = vec![0.0f64; raw.len()];
        // For multiclass the convert reads the whole class vector; for the spine
        // (single output) it is the identity. ObjectiveKind::convert handles both.
        self.objective_kind.convert(&raw, &mut transformed);
        transformed.into_iter().map(|v| v as f32).collect()
    }

    /// Raw (untransformed) accumulated score for one feature row, over the first
    /// `num_iteration` trees (`<= 0` = all). This is the public mirror of
    /// `predict(raw_score=True, num_iteration=k)`.
    pub fn predict_row_raw(&self, features: &[f64], num_iteration: i32) -> Vec<f64> {
        self.model.predict_raw(features, 0, num_iteration)
    }

    /// Batch transformed prediction: one transformed score vector per row, equal
    /// to calling [`predict_row`](Self::predict_row) per row. Thin delegation —
    /// no new algorithm.
    pub fn predict(&self, rows: &[Vec<f64>]) -> Vec<Vec<f32>> {
        rows.iter().map(|r| self.predict_row(r)).collect()
    }

    /// Batch RAW (untransformed) prediction over the first `num_iteration` trees
    /// (`<= 0` = all), one raw score vector per row. Delegates to
    /// [`GbdtModel::predict_raw`].
    pub fn predict_raw_batch(
        &self,
        rows: &[Vec<f64>],
        start_iteration: i32,
        num_iteration: i32,
    ) -> Vec<Vec<f64>> {
        rows.iter()
            .map(|r| self.model.predict_raw(r, start_iteration, num_iteration))
            .collect()
    }

    /// Per-feature split-count importance. Delegates to the C++-faithful
    /// [`GbdtModel::feature_importance_split_count_guarded`], which counts a split
    /// toward its feature ONLY when `split_gain > 0` (`gbdt_model_text.cpp:636-642`,
    /// `importance_type=0`) — matching the official `Booster.feature_importance`
    /// 'split' semantics.
    pub fn feature_importance_split(&self) -> Vec<u64> {
        self.model.feature_importance_split_count_guarded()
    }

    /// Per-feature gain-sum importance. Delegates to
    /// [`GbdtModel::feature_importance_gain`].
    pub fn feature_importance_gain(&self) -> Vec<f64> {
        self.model.feature_importance_gain()
    }

    /// Refit one tree's leaf values to new data/gradients. Delegates to
    /// [`GbdtModel::refit_one_tree`].
    #[allow(clippy::too_many_arguments)]
    pub fn refit(
        &mut self,
        tree_index: usize,
        rows: &[Vec<f64>],
        gradients: &[f32],
        hessians: &[f32],
        decay: f64,
        use_l1: bool,
        l1: f64,
        l2: f64,
    ) {
        self.model.refit_one_tree(
            tree_index, rows, gradients, hessians, decay, use_l1, l1, l2,
        );
    }

    /// Whole-ensemble leaf-refit on new `(rows, labels)` for the regression (L2)
    /// default objective — the high-level mirror of `Booster.refit(data, label)`.
    /// Delegates to [`GbdtModel::refit_ensemble_l2`], which
    /// reproduces the C++ `RefitTree` iterative loop (grad/hess on the score
    /// accumulated from the refit trees, per-tree leaf blend by `decay`). In place.
    pub fn refit_data(
        &mut self,
        rows: &[Vec<f64>],
        labels: &[f32],
        decay: f64,
        use_l1: bool,
        l1: f64,
        l2: f64,
    ) {
        self.model
            .refit_ensemble_l2(rows, labels, decay, use_l1, l1, l2);
    }

    /// Serialize the model to LightGBM-compatible v4 model text (the
    /// byte-stable `%.17g`/`%g` formatter). Delegates to
    /// [`lgbm_model::model_text::save`].
    pub fn model_to_string(&self) -> String {
        lgbm_model::model_text::save(&self.model)
    }

    /// Write the model text to `path`. An I/O failure is mapped to a typed
    /// [`LgbmError::Io`] so the caller never panics.
    ///
    /// # Errors
    /// [`LgbmError::Io`] wrapping the I/O failure detail.
    pub fn save_model(&self, path: &std::path::Path) -> Result<(), LgbmError> {
        let text = self.model_to_string();
        std::fs::write(path, text).map_err(|e| LgbmError::Io {
            detail: format!("failed to write model to {}: {e}", path.display()),
        })
    }

    /// Reconstruct a [`Booster`] from LightGBM model text (via
    /// [`lgbm_model::model_text::load`]). Carries the loaded model's objective so
    /// `predict` applies the same transform. The eval history / iter snapshots are
    /// empty (a loaded model carries no training trace).
    ///
    /// # Errors
    /// [`LgbmError::Model`] when the text fails to parse (untrusted-text boundary)
    /// — never a panic.
    pub fn model_from_string(text: &str) -> Result<Booster, LgbmError> {
        let model = lgbm_model::model_text::load(text).map_err(LgbmError::Model)?;
        let objective_string = model
            .objective_string
            .clone()
            .unwrap_or_else(|| "regression".to_string());
        let objective_kind = ObjectiveKind::parse(&objective_string)
            .unwrap_or(ObjectiveKind::Regression { sqrt: false });
        let best_iteration = model.num_iteration();
        Ok(Booster {
            model,
            objective_kind,
            best_iteration,
            eval_history: Vec::new(),
            iter_scores: Vec::new(),
            iter_grad_hess: Vec::new(),
        })
    }
}

/// Train a regression spine [`Booster`] from a [`Config`] + a dense identity-binned
/// corpus (the public entry point).
///
/// Drives the full vertical spine: objective/metric resolution → identity-binned
/// feature columns → the [`Gbdt`] loop → [`GbdtModel`]. The eval history is
/// populated when the config names an l2/rmse metric.
///
/// # Errors
/// [`LgbmError`] for an unsupported objective/metric, an invalid corpus, or a
/// loop/learner failure — never a panic.
pub fn train(config: &Config, corpus: &DenseCorpus) -> Result<Booster, LgbmError> {
    // Resolve the training-side objective from config.objective. The custom-closure
    // path is `train_custom`.
    let first = config.objective.split_whitespace().next().unwrap_or("");
    let (boost_obj, transformed_labels) = resolve_objective(config, corpus)?;
    let metrics = eval_metrics_for(first, config);
    train_inner(config, corpus, boost_obj, transformed_labels, metrics)
}

/// Resolve the training-side [`BoostObjective`] + the (possibly transformed)
/// labels from `config.objective` over `corpus`. Mirrors the C++ string-keyed
/// `CreateObjectiveFunction` factory + the per-objective `Init` label guards
/// (binary/multiclass class checks; poisson/gamma/tweedie `>= 0`; xentropy
/// `[0, 1]`) surfaced as typed errors — never a panic.
fn resolve_objective(
    config: &Config,
    corpus: &DenseCorpus,
) -> Result<(BoostObjective<'static>, Vec<f32>), LgbmError> {
    let first = config.objective.split_whitespace().next().unwrap_or("");
    Ok(match first {
        "binary" => {
            let b = Binary::new(config.sigmoid).map_err(LgbmError::Objective)?;
            (BoostObjective::Binary(b), corpus.labels.clone())
        }
        "multiclass" | "softmax" => {
            // The integer class labels feed Init (label range check) + the per-row
            // strided softmax gather; they pass through unchanged (no transform).
            let m = MulticlassSoftmax::new(config.num_class, &corpus.labels)
                .map_err(LgbmError::Objective)?;
            (BoostObjective::Multiclass(m), corpus.labels.clone())
        }
        "multiclassova" | "multiclass_ova" | "ova" | "ovr" => {
            let o = MulticlassOva::new(config.num_class, config.sigmoid, &corpus.labels)
                .map_err(LgbmError::Objective)?;
            (BoostObjective::MulticlassOva(o), corpus.labels.clone())
        }
        // cross_entropy / cross_entropy_lambda: single-output xentropy with
        // an Init `[0, 1]` label guard.
        "cross_entropy" | "xentropy" | "cross_entropy_lambda" | "xentlambda" => {
            let x = Xentropy::parse(&config.objective).map_err(LgbmError::Objective)?;
            x.check_labels(&corpus.labels).map_err(LgbmError::Objective)?;
            (BoostObjective::Xentropy(x), corpus.labels.clone())
        }
        _ => {
            // regression / regression_l1 / huber / fair / quantile / mape / poisson /
            // gamma / tweedie (+ aliases) route through the enum factory. The exp/log
            // objectives carry an Init `>= 0` (+ non-zero-sum) label guard.
            let o = Objective::from_config(config).map_err(LgbmError::Objective)?;
            o.check_labels(&corpus.labels).map_err(LgbmError::Objective)?;
            let lbl = o.transform_labels(&corpus.labels);
            (BoostObjective::Builtin(o), lbl)
        }
    })
}

/// Train a [`Booster`] with an optional validation set, enabling early stopping
/// and valid-metric eval history. When `config.early_stopping_round > 0` a
/// validation set is REQUIRED (else a typed error); when bagging is configured
/// (`bagging_freq > 0 && bagging_fraction < 1`) the per-iter loop draws the bag
/// over the RNG and scores in-bag + OOB rows.
///
/// # Errors
/// [`LgbmError`] for an unsupported objective/metric, an invalid corpus,
/// early stopping without a valid set, `bagging_by_query = true` (not supported
/// by this training facade), or a loop/learner failure — never a panic.
pub fn train_with_valid(
    config: &Config,
    corpus: &DenseCorpus,
    valid: &DenseCorpus,
) -> Result<Booster, LgbmError> {
    let first = config.objective.split_whitespace().next().unwrap_or("");
    let (boost_obj, transformed_labels) = resolve_objective(config, corpus)?;
    let metrics = eval_metrics_for(first, config);
    train_inner_full(config, corpus, Some(valid), boost_obj, transformed_labels, metrics)
}

/// Train with a user-supplied `custom` objective closure. The
/// closure maps the current raw scores (f32) to `(grad, hess)`; `boost_from_average`
/// is forced OFF for custom (mirroring the C++ `obj == null` path).
///
/// # Errors
/// [`LgbmError`] for an invalid corpus or a loop/learner failure; a wrong-length
/// closure return surfaces as `LgbmError::Objective`, never a panic.
pub fn train_custom<'a, F>(
    config: &Config,
    corpus: &DenseCorpus,
    closure: F,
) -> Result<Booster, LgbmError>
where
    F: Fn(&[f64]) -> (Vec<f32>, Vec<f32>) + 'a,
{
    train_custom_with_metric(config, corpus, closure, None)
}

/// Train with a user-supplied `custom` objective closure AND an OPTIONAL
/// custom-metric (feval) closure. The feval
/// closure maps `(raw_scores, labels) -> (name, value, is_higher_better)`
/// (mirroring C++ `_EvalFunctionWrapper`); when supplied it REPLACES the built-in
/// `l2` eval metric and feeds the SAME eval-history loop, recording the
/// closure-supplied name. When `feval` is `None` this behaves identically to
/// [`train_custom`] (the built-in `l2` over the raw score).
///
/// This is the upstream hook the Python `feval` marshalling consumes: the
/// Python layer wraps the user's `feval` into a [`CustomMetricClosure`] and calls
/// this entry, so a user metric is wired into eval history without disturbing the
/// built-in path.
///
/// # Errors
/// [`LgbmError`] for an invalid corpus or a loop/learner failure; a wrong-length
/// objective-closure return surfaces as `LgbmError::Objective`, a non-finite
/// feval value as `LgbmError::CustomMetric`, never a panic.
pub fn train_custom_with_metric<'a, F>(
    config: &Config,
    corpus: &DenseCorpus,
    closure: F,
    feval: Option<CustomMetricClosure>,
) -> Result<Booster, LgbmError>
where
    F: Fn(&[f64]) -> (Vec<f32>, Vec<f32>) + 'a,
{
    let custom = CustomObjective::new(closure);
    // When a feval is supplied, the custom-metric closure REPLACES the built-in
    // metric list ([l2]) and feeds the SAME eval-history loop; otherwise the
    // custom run's eval metric mirrors the capture (l2 over the raw score).
    let metrics = match feval {
        Some(f) => vec![EvalMetric::Custom(f)],
        None => vec![EvalMetric::Reg(Metric::L2)],
    };
    train_inner(
        config,
        corpus,
        BoostObjective::Custom(custom),
        corpus.labels.clone(),
        metrics,
    )
}

/// Train with a user-supplied `custom` objective closure AND an OPTIONAL custom
/// metric over a RAW (arbitrary-valued) corpus — the raw→bin→train bridge for
/// the custom path. This is [`train_custom_with_metric`]'s sibling for
/// raw input: it bins each feature with the bit-exact [`BinMapper`] via
/// [`build_feature_columns_from_raw`] (exactly like [`train_raw`]) and then drives
/// the SAME custom-objective + custom-metric eval-history loop the identity-binned
/// `train_custom_with_metric` uses. `boost_from_average` is forced OFF for custom
/// (mirroring C++ `obj == null`).
///
/// The Python binding consumes this entry: a user passes a numpy matrix (→
/// `RawCorpus`) plus a Python `fobj` (→ `closure`) and optional `feval` (→
/// `feval`); this bins the matrix and runs the custom path, so a Python custom
/// objective trains on real-valued features without a separate identity-binning
/// step.
///
/// # Errors
/// [`LgbmError`] for an invalid corpus or a loop/learner failure; a wrong-length
/// objective-closure return surfaces as `LgbmError::Objective`, a non-finite feval
/// value as `LgbmError::CustomMetric`, never a panic.
pub fn train_custom_raw_with_metric<'a, F>(
    config: &Config,
    corpus: &RawCorpus,
    closure: F,
    feval: Option<CustomMetricClosure>,
) -> Result<Booster, LgbmError>
where
    F: Fn(&[f64]) -> (Vec<f32>, Vec<f32>) + 'a,
{
    let custom = CustomObjective::new(closure);
    let metrics = match feval {
        Some(f) => vec![EvalMetric::Custom(f)],
        None => vec![EvalMetric::Reg(Metric::L2)],
    };
    let features = build_feature_columns_from_raw_with_config(corpus, config)?;
    let raw_features = raw_matrix_from_columns(corpus, config);
    train_inner_columns(
        config,
        corpus.num_data() as i32,
        feature_infos_from_columns(corpus),
        &corpus.labels,
        features,
        BoostObjective::Custom(custom),
        corpus.labels.clone(),
        metrics,
        raw_features,
    )
}

/// The shared training driver: identity-binned columns → the [`Gbdt`] loop →
/// per-round eval history → [`Booster`]. Generic over the [`BoostObjective`].
/// Delegates to [`train_inner_full`] with no validation set / no early stopping,
/// preserved byte-for-byte.
fn train_inner(
    config: &Config,
    corpus: &DenseCorpus,
    boost_obj: BoostObjective<'_>,
    labels: Vec<f32>,
    metrics: Vec<EvalMetric>,
) -> Result<Booster, LgbmError> {
    train_inner_full(config, corpus, None, boost_obj, labels, metrics)
}

/// The full training driver: per-iteration loop with bagging, metric-eval cadence
/// (`metric_freq`, multi-metric, `is_provide_training_metric`), early stopping over
/// an optional validation set, and the trailing-tree pop. When `valid` is `None`,
/// `early_stopping_round == 0`, `bagging_freq == 0` / `bagging_fraction == 1`, this
/// reproduces the simple no-bagging/no-early-stopping loop exactly.
fn train_inner_full(
    config: &Config,
    corpus: &DenseCorpus,
    valid: Option<&DenseCorpus>,
    boost_obj: BoostObjective<'_>,
    labels: Vec<f32>,
    metrics: Vec<EvalMetric>,
) -> Result<Booster, LgbmError> {
    // The identity path builds its feature columns from the integer-binnable
    // corpus, then delegates to the shared column-based driver. The raw→bin→train
    // path (train_raw) supplies pre-binned columns to `train_inner_columns` directly.
    // Time the once-per-train binning into the whole-train budget (the fixed-setup
    // bucket; amortizes in bin-once-train-many usage).
    let features = lgbm_treelearner::phase_prof::time(
        &lgbm_treelearner::phase_prof::BINNING_NS,
        || build_feature_columns(corpus),
    )?;
    let num_features = features.len();
    let feature_infos = lgbm_treelearner::phase_prof::time(
        &lgbm_treelearner::phase_prof::SETUP_NS,
        || feature_infos_from_rows(&corpus.features, num_features),
    );
    // Linear tree: the DenseCorpus is identity-binned (raw value == bin index), so
    // its row-major `features` ARE the raw feature matrix the linear fit needs.
    let raw_features = if config.linear_tree {
        Some(corpus.features.iter().flatten().copied().collect::<Vec<f64>>())
    } else {
        None
    };
    train_inner_columns_full(
        config,
        corpus.features.len() as i32,
        feature_infos,
        &corpus.labels,
        valid,
        features,
        boost_obj,
        labels,
        metrics,
        raw_features,
    )
}

/// The column-based training driver used by the raw→bin→train path (no validation
/// set; early stopping / bagging still honored from config when `valid` is `None`).
/// Pre-binned [`FeatureColumn`]s are supplied directly (already built via the
/// bit-exact `BinMapper`), so this NEVER re-bins.
#[allow(clippy::too_many_arguments)]
fn train_inner_columns(
    config: &Config,
    num_data: i32,
    feature_infos: String,
    corpus_labels: &[f32],
    features: Vec<FeatureColumn>,
    boost_obj: BoostObjective<'_>,
    labels: Vec<f32>,
    metrics: Vec<EvalMetric>,
    raw_features: Option<Vec<f64>>,
) -> Result<Booster, LgbmError> {
    train_inner_columns_full(
        config,
        num_data,
        feature_infos,
        corpus_labels,
        None,
        features,
        boost_obj,
        labels,
        metrics,
        raw_features,
    )
}

/// Row-major (`num_data * num_features`) raw feature matrix for the linear-tree
/// leaf fit, or `None` when `linear_tree` is off. Indexed by ORIGINAL feature
/// index (the same space as `Tree::split_feature`).
fn raw_matrix_from_columns(corpus: &RawCorpus, config: &Config) -> Option<Vec<f64>> {
    if !config.linear_tree {
        return None;
    }
    let (n, m) = (corpus.num_data(), corpus.num_features());
    let mut v = vec![0.0f64; n * m];
    for r in 0..n {
        for c in 0..m {
            v[r * m + c] = corpus.value(r, c);
        }
    }
    Some(v)
}

/// The full column-based training driver: takes `num_data` and the precomputed
/// per-feature `feature_infos` string (the caller derives these from its own
/// representation — row-major for the [`DenseCorpus`] spine, column-major for the
/// [`RawCorpus`] path), the corpus labels (for metric eval), and the PRE-BUILT
/// feature columns (identity-binned OR `BinMapper`-binned). All binning happens
/// BEFORE this is called.
#[allow(clippy::too_many_arguments)]
fn train_inner_columns_full(
    config: &Config,
    num_data: i32,
    feature_infos: String,
    corpus_labels: &[f32],
    valid: Option<&DenseCorpus>,
    features: Vec<FeatureColumn>,
    boost_obj: BoostObjective<'_>,
    labels: Vec<f32>,
    metrics: Vec<EvalMetric>,
    raw_features: Option<Vec<f64>>,
) -> Result<Booster, LgbmError> {
    use lgbm_boosting::{BaggingConfig, BaggingSampleStrategy, EarlyStopping, EvalSnapshot, MetricSpec};

    let objective_string = canonical_objective_string(config);
    let objective_kind = ObjectiveKind::parse(&objective_string)
        .unwrap_or(ObjectiveKind::Regression { sqrt: false });

    // ---- feature columns (pre-built by the caller) ----
    // num_data + feature_infos are supplied by the caller (derived from its own
    // row-major / column-major store) — this driver no longer holds raw feature rows.
    let num_class = config.num_class.max(1);
    let num_features = features.len();
    let max_feature_idx = features
        .iter()
        .map(|f| f.real_feature_index)
        .max()
        .unwrap_or(-1);

    // Per-feature constraint/penalty vectors must match num_features
    // when non-empty — mirrors the C++ length CHECKs in GBDT::Init (gbdt.cpp:58)
    // and CostEfficientGradientBoosting::Init (cost_effective_gradient_boosting.hpp
    // :47-60). Validate BEFORE any tree grows; a wrong-length vector is otherwise
    // silently accepted (the .get()-guarded accessors treat missing entries as
    // unconstrained / zero-penalty), applying constraints to the wrong features.
    for (param, len) in [
        ("monotone_constraints", config.monotone_constraints.len()),
        (
            "cegb_penalty_feature_coupled",
            config.cegb_penalty_feature_coupled.len(),
        ),
        (
            "cegb_penalty_feature_lazy",
            config.cegb_penalty_feature_lazy.len(),
        ),
    ] {
        if len != 0 && len != num_features {
            return Err(LgbmError::InvalidConstraintLength {
                param,
                actual: len,
                num_features,
            });
        }
    }

    // ---- the learner ----
    // Backend dispatch is feature-switched (see the gated imports above): the
    // default build trains on the native-f64 CpuBackend; `--features rocm` trains on
    // the gfx1100 GPU via RocmBackend (same f64 kernels, bit-exact); `--features
    // cuda`/`wgpu` train via CudaBackend/WgpuBackend (the SAME runtime-generic GPU
    // kernels). Priority rocm > cuda > wgpu > cpu — the `not(...)` cfg guards make the
    // arms mutually exclusive so exactly one backend is selected for any feature
    // combination. The learner + GBDT loop below are generic over `B: Backend`, so
    // only this construction site differs.
    #[cfg(not(any(feature = "rocm", feature = "cuda", feature = "wgpu")))]
    let backend = CpuBackend;
    #[cfg(not(any(feature = "rocm", feature = "cuda", feature = "wgpu")))]
    let client = cpu_client();
    // RocmBackend carries interior-mutable device-resident state, so it
    // is constructed via Default (no longer a unit struct). One instance per train()
    // call (outside the GBDT iter loop) ⇒ the resident-bin cache persists across all
    // trees in the train.
    #[cfg(feature = "rocm")]
    let backend = RocmBackend::default();
    #[cfg(feature = "rocm")]
    let client = rocm_client();
    // CudaBackend/WgpuBackend are `GpuBackend<R>` aliases carrying
    // the SAME on-device resident histogram pool RocmBackend uses — `::default()`
    // enables residency, reaching ROCm-parity speed.
    #[cfg(all(feature = "cuda", not(feature = "rocm")))]
    let backend = CudaBackend::default();
    #[cfg(all(feature = "cuda", not(feature = "rocm")))]
    let client = cuda_client();
    #[cfg(all(feature = "wgpu", not(feature = "rocm"), not(feature = "cuda")))]
    let backend = WgpuBackend::default();
    #[cfg(all(feature = "wgpu", not(feature = "rocm"), not(feature = "cuda")))]
    let client = wgpu_client();
    let gain = GainConfig::from_config(config);
    let mut learner = SerialTreeLearner::new(
        &backend,
        &client,
        gain,
        config.num_leaves,
        config.max_depth,
    )
    .with_features(features.clone());

    // ---- the GBDT loop (optionally with bagging / GOSS) ----
    // GOSS (data_sample_strategy=goss) and bagging are mutually exclusive — GOSS
    // forbids bagging (goss.hpp:87-89). The `boosting=goss` alias-expansion sets
    // data_sample_strategy=goss + boosting=gbdt (set.rs:472-476).
    let goss_on = config.data_sample_strategy == "goss";
    let bagging_on =
        !goss_on && config.bagging_freq > 0 && config.bagging_fraction < 1.0;
    let mut gbdt = Gbdt::with_objective(
        boost_obj,
        config.learning_rate,
        num_class,
        num_data,
        config.boost_from_average,
        None,
    );
    // Opt-in `use_quantized_grad` APPROXIMATE mode: quantizes grad/hess each iter
    // before the learner. No-op when false (the default) — the exact path is untouched.
    gbdt = gbdt
        .with_quantized_grad(
            config.use_quantized_grad,
            config.num_grad_quant_bins,
            config.stochastic_rounding,
        )
        .with_quant_renew_leaf(config.quant_train_renew_leaf, config.lambda_l1, config.lambda_l2);
    // Linear tree (`config.linear_tree`): fit per-leaf linear models over the RAW
    // features after each non-first tree grows (C++ `LinearTreeLearner`). Needs the
    // raw continuous feature matrix (the binned columns are insufficient); the
    // caller supplies it as row-major `num_data * num_features`. No-op when the raw
    // matrix is absent or `linear_tree` is off.
    if config.linear_tree {
        if let Some(raw) = raw_features {
            gbdt = gbdt.with_linear_tree(true, config.linear_lambda, raw);
        }
    }
    // DART (an enum field on Gbdt): `boosting=dart`
    // selects the DART drop+normalize variant. DART subclasses GBDT in C++ and can
    // coexist with bagging (the sample strategy is independent); the spine validates
    // plain DART. DroppingTrees runs BEFORE GetGradients each iter, Normalize after the
    // new tree (dart.hpp). It is NOT compatible with GOSS (data_sample_strategy=goss
    // forbids bagging-style subsampling under DART's drop semantics; the spine never
    // combines them).
    let dart_on = config.boosting == "dart";
    if dart_on {
        let dart_cfg = lgbm_boosting::DartConfig {
            drop_rate: config.drop_rate,
            max_drop: config.max_drop,
            skip_drop: config.skip_drop,
            xgboost_dart_mode: config.xgboost_dart_mode,
            uniform_drop: config.uniform_drop,
            drop_seed: config.drop_seed,
        };
        gbdt = gbdt.with_dart(dart_cfg, features.clone());
    }
    // Random Forest (an enum field on Gbdt):
    // `boosting=rf` selects the averaged-tree variant with mandatory randomization
    // (rf.hpp). RF re-derives grad/hess once from a constant init-score buffer,
    // averages trees (no learning-rate accumulation), and renews leaves to the mean
    // residual. The two RF CHECKs (objective != null; bagging OR feature_fraction
    // active) surface as a typed BoostingError::RfConfig at the top of the RF train
    // path. RF reuses the proven BaggingSampleStrategy (the same bit-exact bagging
    // RNG the training path uses); the `with_bagging` call below also fires when
    // bagging is active so the strategy is attached.
    let rf_on = config.boosting == "rf";
    if rf_on {
        let feature_subsampling_active =
            config.feature_fraction < 1.0 && config.feature_fraction > 0.0;
        let rf_cfg = lgbm_boosting::RfConfig {
            bagging_active: bagging_on,
            feature_subsampling_active,
        };
        gbdt = gbdt.with_rf(rf_cfg, features.clone());
    }
    if goss_on {
        // GOSSStrategy::ResetSampleConfig CHECKs (top+other<=1, both>0) surface as a
        // typed Result. The per-block RNG seed base is bagging_seed (goss.hpp:97).
        let goss = lgbm_boosting::GossSampleStrategy::reset_sample_config(
            config.top_rate,
            config.other_rate,
            config.learning_rate,
            num_data,
            num_class,
            config.bagging_seed,
        )
        .map_err(LgbmError::Boosting)?;
        gbdt = gbdt.with_goss(goss, features.clone());
    } else if bagging_on {
        // bagging_by_query: the query-grouped draw + expansion is implemented
        // in BaggingSampleStrategy::bagging_by_query and proven by the rank_parity
        // RNG-replay golden, but it requires query/group boundaries to bag by. The
        // `DenseCorpus` facade carries no query metadata yet (ranking end-to-end
        // training over the facade is a later surface), so a `bagging_by_query=true`
        // request here is rejected with an honest typed error (mirroring the C++
        // `Log::Fatal("Ranking tasks require query information")`) rather than silently
        // falling through to ROW bagging.
        if config.bagging_by_query {
            return Err(LgbmError::Boosting(lgbm_boosting::BoostingError::Objective(
                lgbm_objective::ObjectiveError::Unsupported {
                    name: "bagging_by_query requires query/group boundaries, which the \
                           DenseCorpus training facade does not carry yet (the query-grouped \
                           draw itself is implemented + RNG-replay tested in rank_parity)"
                        .to_string(),
                },
            )));
        }
        let bag_cfg = BaggingConfig::new(
            config.bagging_fraction,
            config.pos_bagging_fraction,
            config.neg_bagging_fraction,
            config.bagging_freq,
            config.bagging_seed,
            config.bagging_by_query,
        )
        .map_err(LgbmError::Boosting)?;
        let strat = BaggingSampleStrategy::reset_sample_config(bag_cfg, num_data, &labels);
        gbdt = gbdt.with_bagging(strat, features.clone());
    }

    // ---- early stopping setup ----
    let es_enabled = config.early_stopping_round > 0;
    if es_enabled && valid.is_none() {
        return Err(LgbmError::Boosting(
            lgbm_boosting::BoostingError::EarlyStoppingWithoutValidSet,
        ));
    }
    // The validation set's metric values drive the stop decision; we maintain an
    // incremental f64 valid score accumulator (class-major) updated each iter.
    let (valid_feat_rows, valid_labels): (Vec<Vec<f64>>, Vec<f32>) = match valid {
        Some(v) => (v.features.clone(), v.labels.clone()),
        None => (Vec::new(), Vec::new()),
    };
    let valid_nd = valid_feat_rows.len() as i32;
    // class-major valid score buffer (num_class blocks of valid_nd).
    let mut valid_score = vec![0.0f64; (valid_nd.max(0) as usize) * num_class as usize];
    let mut prev_tree_count = 0usize;
    // Inject the valid set's boost_from_average init (C++ BoostFromAverage adds the
    // init to EVERY valid score updater). The training init is folded into tree 0's
    // leaves via AddBias, so re-predicting trees over the valid rows already includes
    // it — we therefore re-derive valid scores by predicting the grown trees per iter
    // (which carry the AddBias-folded init), needing no separate init injection.

    let metric_specs: Vec<MetricSpec> = metrics
        .iter()
        .map(|m| MetricSpec {
            name: m.name(),
            factor_to_bigger_better: m.factor_to_bigger_better(),
        })
        .collect();
    let mut early = EarlyStopping::new(
        config.early_stopping_round,
        config.early_stopping_min_delta,
        config.first_metric_only,
        num_class,
        if valid.is_some() { 1 } else { 0 },
        metric_specs,
    );

    // eval history: training metrics (when is_provide_training_metric) + valid metrics.
    // The training metric is computed ONLY when `is_provide_training_metric` is set,
    // matching gbdt.cpp (default false ⇒ empty training history, even with no valid
    // set). Callers that want training-metric history without a valid set must
    // opt in via `is_provide_training_metric=true` (the C++ behavior).
    let provide_train = config.is_provide_training_metric;
    let metric_freq = config.metric_freq.max(1);
    let mut train_eval_history: Vec<(String, Vec<f64>)> = metrics
        .iter()
        .map(|m| (format!("training {}", m.name()), Vec::new()))
        .collect();
    let mut valid_eval_history: Vec<(String, Vec<f64>)> = metrics
        .iter()
        .map(|m| (format!("valid_0 {}", m.name()), Vec::new()))
        .collect();
    // The legacy training-only eval history (keyed by bare metric name).
    let mut legacy_eval_history: Vec<(String, Vec<f64>)> = metrics
        .iter()
        .map(|m| (m.name(), Vec::new()))
        .collect();

    let mut iter_scores: Vec<Vec<f64>> = Vec::new();
    let mut iter_grad_hess: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();

    let total_iters = config.num_iterations.max(0);
    let mut ran_iters = 0i32;
    for it in 0..total_iters {
        // Time the whole train_one_iter (grad+learner+score+snapshot) so the
        // booster-loop tail (metric/valid/accumulation) is isolated as loop_other.
        let snap: IterSnapshot = lgbm_treelearner::phase_prof::time(
            &lgbm_treelearner::phase_prof::TRAIN_ONE_ITER_NS,
            || gbdt.train_one_iter(&mut learner, &labels, num_features),
        )
        .map_err(LgbmError::Boosting)?;

        // A NON-FIRST no-split bagged round is POPPED by
        // `train_one_iter` (C++ gbdt.cpp:440-447): no tree emitted, `self.iter` not
        // advanced, score unchanged. Mirror the wheel/`lgb.train` driver: do NOT count
        // it as an emitted iteration. Skip the per-iter score/grad-hess accumulation
        // (a duplicate `snap.score` push would mis-align `iter_scores` with the emitted
        // trees and corrupt the L2 per-iter golden), skip metric eval (no score
        // change), and leave `ran_iters` / the eval cadence to the NEXT real round.
        // The bag re-draws on the next round (`bagging_freq=1` re-bags every call) so
        // the loop still grows the target tree count over `total_iters` boost rounds.
        if !snap.emitted {
            continue;
        }

        ran_iters = it + 1;
        // Move the per-iter snapshots into the golden-replay history instead of
        // cloning them: `snap` is consumed by these moves (`snap.gradients` /
        // `snap.hessians` have no later use, and the training-metric eval below now
        // reads the just-pushed `iter_scores` element). `snap.score` is otherwise
        // double-allocated — `train_one_iter` already `to_vec()`s it (gbdt.rs:911)
        // and the previous `.clone()` here allocated it a second time. Pure alloc
        // reduction; the retained data is byte-identical (parity-neutral).
        iter_scores.push(snap.score);
        iter_grad_hess.push((snap.gradients, snap.hessians));
        let cur_score = iter_scores.last().expect("pushed above");

        // Incremental valid-score update: predict the trees grown THIS iter over the
        // valid rows (class-major), adding to the running valid_score.
        if valid_nd > 0 {
            let trees = gbdt.trees();
            for (t_idx, tree) in trees.iter().enumerate().skip(prev_tree_count) {
                let cur_tree_id = (t_idx % num_class as usize) as i32;
                let off = (cur_tree_id as usize) * valid_nd as usize;
                for (r, row) in valid_feat_rows.iter().enumerate() {
                    valid_score[off + r] += tree.predict(row);
                }
            }
            prev_tree_count = trees.len();
        }

        // Metric eval cadence (metric_freq gate). Always eval on the LAST
        // iter and on every freq multiple, matching the C++ OutputMetric cadence.
        //
        // `metric_freq` gates ONLY the recorded eval-HISTORY (the
        // pushes into train/valid/legacy_eval_history and any future logging). The
        // valid-score eval that FEEDS the early-stop DECISION + the `early.update`
        // call run EVERY iteration when ES is on, INDEPENDENT of metric_freq —
        // mirroring gbdt.cpp:574 where the valid-metric+ES block is
        // `if (need_output || early_stopping_round_ > 0)` and `need_output`
        // (= `iter % metric_freq == 0`) only guards the `Log::Info`.
        let do_eval = (it + 1) % metric_freq == 0 || it + 1 == total_iters;

        // Training metrics: history is metric_freq-gated.
        if do_eval && provide_train {
            for (mi, m) in metrics.iter().enumerate() {
                // Time the per-iter training-metric eval over all rows.
                let v = lgbm_treelearner::phase_prof::time(
                    &lgbm_treelearner::phase_prof::METRIC_NS,
                    || m.eval(cur_score, corpus_labels),
                )?;
                // A custom-metric (feval) key is resolved lazily from the closure
                // (the placeholder "custom" set at history-setup is overwritten on
                // the first eval with the user-supplied name) so the recorded
                // history key matches the user's metric name.
                if matches!(m, EvalMetric::Custom(_)) {
                    let name = m.resolved_name(cur_score, corpus_labels);
                    legacy_eval_history[mi].0 = name.clone();
                    train_eval_history[mi].0 = format!("training {name}");
                }
                train_eval_history[mi].1.push(v);
                legacy_eval_history[mi].1.push(v);
            }
        }

        // Valid metrics: eval whenever we either record history (`do_eval`) OR need
        // the ES decision this iter (`es_enabled`). Push to valid_eval_history ONLY
        // on `do_eval` (so metric_freq still thins the RECORDED history, and the
        // metric_freq_thins_eval_history test stays green); feed `row` to
        // `early.update` whenever ES is on (EVERY iteration).
        if valid_nd > 0 && (do_eval || es_enabled) {
            let mut row = Vec::with_capacity(metrics.len());
            for (mi, m) in metrics.iter().enumerate() {
                let v = m.eval(&valid_score, &valid_labels)?;
                if do_eval {
                    valid_eval_history[mi].1.push(v);
                }
                row.push(v);
            }
            if es_enabled {
                let stop = early.update(it, &EvalSnapshot { values: vec![row] });
                if stop {
                    break;
                }
            }
        }
    }

    // best_iteration + trailing-tree pop.
    let best_iteration = if es_enabled {
        let pop = early.trailing_trees_to_pop(ran_iters);
        if pop > 0 {
            gbdt.pop_trailing_trees(pop as usize);
        }
        early.best_iteration()
    } else {
        gbdt.num_iteration()
    };

    // Assemble the public eval_history: legacy bare-name training metrics (so
    // existing replay keys still resolve) PLUS the valid_0 metrics when present.
    let mut eval_history: Vec<(String, Vec<f64>)> = legacy_eval_history;
    if valid_nd > 0 {
        eval_history.extend(valid_eval_history);
    }
    let _ = train_eval_history; // training metrics are captured in legacy keys above

    let model = gbdt.into_model(
        objective_string,
        max_feature_idx,
        feature_names(num_features),
        feature_infos,
    );

    // Emit the per-phase BUDGET/LOOP/COUNTS attribution for the shipped train path
    // (the one the Python wheel drives). Inert unless LGBM_PHASE_PROF=1. Parity-neutral
    // (prints to stderr + resets the accumulators; never touches train semantics).
    lgbm_treelearner::phase_prof::dump("train");

    Ok(Booster {
        model,
        objective_kind,
        best_iteration,
        eval_history,
        iter_scores,
        iter_grad_hess,
    })
}

/// Build the canonical LightGBM `objective=` model line from the config. The
/// multiclass objectives append the `num_class:`/`sigmoid:` tokens exactly as the
/// C++ `MulticlassSoftmax::ToString` / `MulticlassOVA::ToString` do (so the model
/// text round-trips and `ObjectiveKind::parse` recovers num_class); the
/// single-output objectives use the bare `config.objective` name.
fn canonical_objective_string(config: &Config) -> String {
    let first = config.objective.split_whitespace().next().unwrap_or("");
    match first {
        "multiclass" | "softmax" => format!("multiclass num_class:{}", config.num_class),
        "multiclassova" | "multiclass_ova" | "ova" | "ovr" => format!(
            "multiclassova num_class:{} sigmoid:{}",
            config.num_class,
            // C++ ToString emits the sigmoid as the default-formatted double; the
            // spine uses 1 → "1" matches the golden `sigmoid:1`.
            format_sigmoid(config.sigmoid),
        ),
        _ => config.objective.clone(),
    }
}

/// Format `sigmoid` for the objective line: an integral value prints without a
/// decimal point (`1` not `1.0`), matching the C++ ostream default the golden uses.
fn format_sigmoid(sigmoid: f64) -> String {
    if sigmoid.fract() == 0.0 {
        format!("{}", sigmoid as i64)
    } else {
        format!("{sigmoid}")
    }
}

/// The per-objective default eval metrics (matching the capture's `metric=` list):
/// `[l2, rmse]` for regression, `[l1, l2, rmse]` for regression_l1,
/// `[binary_logloss, binary_error, auc]` for binary, `[multi_logloss]` for
/// multiclass/multiclassova.
fn eval_metrics_for(objective_first_token: &str, config: &Config) -> Vec<EvalMetric> {
    let sigmoid = config.sigmoid;
    match objective_first_token {
        "binary" => vec![
            EvalMetric::Bin(BinaryMetric::BinaryLogloss { sigmoid }),
            EvalMetric::Bin(BinaryMetric::BinaryError { sigmoid }),
            EvalMetric::Bin(BinaryMetric::Auc),
        ],
        "multiclass" | "softmax" => vec![EvalMetric::Multi(MultiLogloss::new(
            ObjectiveKind::Multiclass { num_class: config.num_class },
            config.num_class,
        ))],
        "multiclassova" | "multiclass_ova" | "ova" | "ovr" => {
            vec![EvalMetric::Multi(MultiLogloss::new(
                ObjectiveKind::MulticlassOva { num_class: config.num_class, sigmoid },
                config.num_class,
            ))]
        }
        "regression_l1" | "l1" | "mean_absolute_error" | "mae" => {
            vec![EvalMetric::Reg(Metric::L1), EvalMetric::Reg(Metric::L2), EvalMetric::Reg(Metric::Rmse)]
        }
        _ => vec![EvalMetric::Reg(Metric::L2), EvalMetric::Reg(Metric::Rmse)],
    }
}

/// `feature_names=` for the model text: `Column_0 Column_1 ...` (the LightGBM
/// default when no names are supplied — matching the capture).
fn feature_names(num_features: usize) -> String {
    (0..num_features)
        .map(|j| format!("Column_{j}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// `feature_infos=` for the model text: `[min:max]` per feature. For the identity
/// path the raw values are the integer bins `0..num_bin-1`, so integral min/max
/// print without a decimal point (matching the capture's `[0:5] [0:2]`); the
/// raw→bin→train path's continuous values print with their real value.
fn feature_infos_from_rows(feature_rows: &[Vec<f64>], num_features: usize) -> String {
    // ONE cache-friendly pass over the row-major matrix (each `row` is contiguous)
    // accumulating per-feature min/max — instead of `num_features` strided COLUMN
    // passes. Same `f64::min`/`f64::max` calls, only loop order changed; min/max
    // are commutative+associative ⇒ BYTE-IDENTICAL per-feature bounds (the
    // `feature_infos` model-text line is parity-checked).
    let mut min = vec![f64::INFINITY; num_features];
    let mut max = vec![f64::NEG_INFINITY; num_features];
    for row in feature_rows {
        for j in 0..num_features {
            min[j] = min[j].min(row[j]);
            max[j] = max[j].max(row[j]);
        }
    }
    (0..num_features)
        .map(|j| format!("[{}:{}]", fmt_bound(min[j]), fmt_bound(max[j])))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Column-major twin of [`feature_infos_from_rows`] for the [`RawCorpus`] path.
/// Reads each feature as a contiguous slice; produces BYTE-IDENTICAL output to the
/// row-major version for the same data (same per-feature min/max, same formatting) —
/// the model-text `feature_infos` line is parity-checked, so this must not differ.
fn feature_infos_from_columns(corpus: &RawCorpus) -> String {
    (0..corpus.num_features())
        .map(|j| {
            let mut min = f64::INFINITY;
            let mut max = f64::NEG_INFINITY;
            for &v in corpus.column(j) {
                min = min.min(v);
                max = max.max(v);
            }
            format!("[{}:{}]", fmt_bound(min), fmt_bound(max))
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Format a feature-info bound: integral values print as an integer (`5` not
/// `5.0`, preserving the identity-path capture bytes); non-integral values print
/// their real value.
fn fmt_bound(v: f64) -> String {
    if v.is_finite() && v.fract() == 0.0 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::TrainingBuilder;

    fn spine_corpus() -> DenseCorpus {
        let f0 = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5];
        let f1 = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];
        let features: Vec<Vec<f64>> = (0..12)
            .map(|i| vec![f0[i] as f64, f1[i] as f64])
            .collect();
        let labels = vec![
            2.0f32, 3.0, 5.0, 6.0, 9.0, 10.0, 12.0, 13.0, 16.0, 17.0, 19.0, 20.0,
        ];
        DenseCorpus { features, labels }
    }

    #[test]
    fn public_api_train_predict_round_trip() {
        let cfg = TrainingBuilder::new()
            .objective("regression")
            .num_iterations(10)
            .learning_rate(0.1)
            .num_leaves(4)
            .min_data_in_leaf(1)
            .boost_from_average(true)
            .metric("l2,rmse")
            // Training-metric history is opt-in (C++-faithful), so request it
            // explicitly (this test asserts training l2/rmse history).
            .is_provide_training_metric(true)
            .seed(1)
            .deterministic(true)
            .build()
            .unwrap();
        let corpus = spine_corpus();
        let booster = train(&cfg, &corpus).expect("train ok");
        // 10 iterations stored.
        assert_eq!(booster.model().num_iteration(), 10);
        assert_eq!(booster.best_iteration, 10);
        // Eval history has l2 + rmse, one value per round.
        assert_eq!(booster.eval_history.len(), 2);
        assert_eq!(booster.eval_history[0].0, "l2");
        assert_eq!(booster.eval_history[0].1.len(), 10);
        // predict round-trips: low-label rows predict lower than high-label rows.
        let p_low = booster.predict_row(&[0.0, 0.0]);
        let p_high = booster.predict_row(&[5.0, 2.0]);
        assert!(p_low[0] < p_high[0], "{} < {}", p_low[0], p_high[0]);
    }

    #[test]
    fn dart_train_predict_uses_normalized_tree_weights() {
        // boosting=dart trains end-to-end, and predict() applies the normalized
        // DART tree weights — which DART bakes into the STORED leaf values via its
        // Shrinkage sequence (DroppingTrees + Normalize). So predict() must equal the
        // plain sum of the stored trees' per-row outputs (the integration proof that the
        // predict-side accumulation reflects the normalized weights, not the raw lr).
        let cfg = TrainingBuilder::new()
            .objective("regression")
            .boosting("dart")
            .drop_rate(0.3)
            .skip_drop(0.0) // force drops to actually happen so normalize runs
            .max_drop(50)
            .drop_seed(4)
            .num_iterations(8)
            .learning_rate(0.5)
            .num_leaves(4)
            .min_data_in_leaf(1)
            .boost_from_average(true)
            .seed(1)
            .deterministic(true)
            .build()
            .unwrap();
        assert_eq!(cfg.boosting, "dart");
        let corpus = spine_corpus();
        let booster = train(&cfg, &corpus).expect("dart train ok");
        // A model was grown (one tree per iter for the single-output spine).
        assert_eq!(booster.model().num_iteration(), 8);

        // predict() == manual sum over the stored (normalized) trees for every row.
        // predict_row casts the f64 accumulation to f32 (the public score_t contract),
        // so compare against the SAME f32 cast of the manual f64 sum (the residual is
        // pure f32 rounding, ~1e-6 — the predict-side weights ARE the normalized ones).
        for row in &corpus.features {
            let manual = booster.model().trees.iter().map(|t| t.predict(row)).sum::<f64>() as f32;
            let got = booster.predict_row(row)[0];
            assert!(
                (got - manual).abs() < 1e-6,
                "DART predict {got} must equal sum of normalized stored trees {manual}"
            );
        }
        // Monotone sanity: low-label rows predict lower than high-label rows.
        let p_low = booster.predict_row(&[0.0, 0.0]);
        let p_high = booster.predict_row(&[5.0, 2.0]);
        assert!(p_low[0] < p_high[0], "{} < {}", p_low[0], p_high[0]);
    }

    #[test]
    fn rf_train_predict_averages_tree_outputs() {
        // boosting=rf trains end-to-end (averaged trees, mandatory bagging,
        // no shrinkage) and predict() AVERAGES the stored RAW tree outputs (the
        // average_output path divides the per-tree sum by num_iteration). The
        // integration proof: predict() == (sum of stored trees' per-row outputs) /
        // num_iteration for every row.
        let cfg = TrainingBuilder::new()
            .objective("regression")
            .boosting("rf")
            .bagging_fraction(0.7)
            .bagging_freq(1)
            .bagging_seed(3)
            .num_iterations(8)
            .num_leaves(4)
            .min_data_in_leaf(1)
            .boost_from_average(true)
            .seed(1)
            .deterministic(true)
            .build()
            .unwrap();
        assert_eq!(cfg.boosting, "rf");
        let corpus = spine_corpus();
        let booster = train(&cfg, &corpus).expect("rf train ok");
        assert_eq!(booster.model().num_iteration(), 8);
        // RF emits average_output.
        assert!(
            booster.model().average_output,
            "RF model must set average_output"
        );

        // predict() == AVERAGE of the stored raw trees for every row.
        let n_iter = booster.model().num_iteration();
        for row in &corpus.features {
            let raw_sum = booster.model().trees.iter().map(|t| t.predict(row)).sum::<f64>();
            let manual_avg = (raw_sum / n_iter as f64) as f32;
            let got = booster.predict_row(row)[0];
            assert!(
                (got - manual_avg).abs() < 1e-6,
                "RF predict {got} must equal the average of the stored trees {manual_avg}"
            );
        }
        // Monotone sanity: low-label rows predict lower than high-label rows.
        let p_low = booster.predict_row(&[0.0, 0.0]);
        let p_high = booster.predict_row(&[5.0, 2.0]);
        assert!(p_low[0] < p_high[0], "{} < {}", p_low[0], p_high[0]);
    }

    #[test]
    fn rf_without_randomization_is_typed_error() {
        // CHECK: boosting=rf with neither bagging nor feature_fraction<1 must
        // surface a typed error (not a panic, not a silent collapse to one tree).
        let cfg = TrainingBuilder::new()
            .objective("regression")
            .boosting("rf")
            .num_iterations(5)
            .num_leaves(4)
            .min_data_in_leaf(1)
            .boost_from_average(true)
            .seed(1)
            .deterministic(true)
            .build()
            .unwrap();
        let corpus = spine_corpus();
        let err = train(&cfg, &corpus).expect_err("RF with no randomization must error");
        assert!(
            matches!(
                err,
                LgbmError::Boosting(lgbm_boosting::BoostingError::RfConfig { .. })
            ),
            "expected RfConfig, got {err:?}"
        );
    }

    #[test]
    fn wrong_length_constraint_vectors_are_typed_errors() {
        // A per-feature constraint/penalty vector whose length != the
        // 2-feature corpus must surface a typed InvalidConstraintLength BEFORE any
        // tree grows — mirroring the C++ GBDT::Init / CEGB::Init length CHECKs —
        // rather than silently applying constraints to the wrong features.
        let corpus = spine_corpus(); // 2 features
        let expect_err = |cfg: Config, param: &str| {
            let err = train(&cfg, &corpus)
                .expect_err("wrong-length constraint vector must error");
            match err {
                LgbmError::InvalidConstraintLength {
                    param: p,
                    actual,
                    num_features,
                } => {
                    assert_eq!(p, param, "param name");
                    assert_eq!(actual, 3, "actual len");
                    assert_eq!(num_features, 2, "num_features");
                }
                other => panic!("expected InvalidConstraintLength for {param}, got {other:?}"),
            }
        };

        // monotone_constraints: 3 entries (all valid values) vs 2 features.
        expect_err(
            TrainingBuilder::new()
                .objective("regression")
                .num_iterations(1)
                .num_leaves(4)
                .min_data_in_leaf(1)
                .monotone_constraints(&[1, -1, 0])
                .build()
                .unwrap(),
            "monotone_constraints",
        );

        // cegb_penalty_feature_coupled: 3 entries vs 2 features.
        let mut coupled = Config::default();
        coupled.objective = "regression".into();
        coupled.num_iterations = 1;
        coupled.num_leaves = 4;
        coupled.min_data_in_leaf = 1;
        coupled.cegb_penalty_feature_coupled = vec![0.1, 0.2, 0.3];
        expect_err(
            TrainingBuilder::new().from_config(coupled).build().unwrap(),
            "cegb_penalty_feature_coupled",
        );

        // cegb_penalty_feature_lazy: 3 entries vs 2 features.
        let mut lazy = Config::default();
        lazy.objective = "regression".into();
        lazy.num_iterations = 1;
        lazy.num_leaves = 4;
        lazy.min_data_in_leaf = 1;
        lazy.cegb_penalty_feature_lazy = vec![0.1, 0.2, 0.3];
        expect_err(
            TrainingBuilder::new().from_config(lazy).build().unwrap(),
            "cegb_penalty_feature_lazy",
        );

        // A correctly-sized (len == 2) monotone vector still trains.
        let ok = TrainingBuilder::new()
            .objective("regression")
            .num_iterations(1)
            .num_leaves(4)
            .min_data_in_leaf(1)
            .monotone_constraints(&[1, 0])
            .build()
            .unwrap();
        assert!(train(&ok, &corpus).is_ok(), "len==num_features must train");
    }

    #[test]
    fn predict_raw_equals_internal_score_open_q2() {
        // predict(raw_score=True, num_iteration=k) == the internal
        // score_ after k iters, for every training row.
        let cfg = TrainingBuilder::new()
            .objective("regression")
            .num_iterations(10)
            .learning_rate(0.1)
            .num_leaves(4)
            .min_data_in_leaf(1)
            .boost_from_average(true)
            .seed(1)
            .deterministic(true)
            .build()
            .unwrap();
        let corpus = spine_corpus();
        let booster = train(&cfg, &corpus).unwrap();
        for k in 1..=10 {
            let internal = &booster.iter_scores[(k - 1) as usize];
            for (i, row) in corpus.features.iter().enumerate() {
                let raw = booster.predict_row_raw(row, k)[0];
                // The internal score_ folds the init via BoostFromAverage→AddScore;
                // predict_row_raw folds it via AddBias into tree 0. They must agree
                // bit-for-bit on this cell (the gate for the L2 bit-exactness contract).
                assert_eq!(
                    raw.to_bits(),
                    internal[i].to_bits(),
                    "row {i} iter {k}: predict_raw {raw} != internal score_ {} (bit-exact)",
                    internal[i]
                );
            }
        }
    }

    #[test]
    fn early_stopping_fires_and_pops_trailing_trees() {
        // Train with a valid set that plateaus quickly so early stopping FIRES before
        // num_iterations; best_iteration < num_iterations and the trailing trees are
        // popped (model tree count == best_iteration).
        let train_corpus = spine_corpus();
        // A valid set whose labels are CONSTANT so the metric plateaus fast (the
        // model can't keep improving valid l2 after the first couple of trees).
        let valid_corpus = DenseCorpus {
            features: spine_corpus().features,
            labels: vec![10.0f32; 12], // constant => valid metric plateaus
        };
        let cfg = TrainingBuilder::new()
            .objective("regression")
            .num_iterations(30)
            .learning_rate(0.3)
            .num_leaves(4)
            .min_data_in_leaf(1)
            .boost_from_average(true)
            .early_stopping_round(2)
            .seed(1)
            .deterministic(true)
            .build()
            .unwrap();
        let booster = train_with_valid(&cfg, &train_corpus, &valid_corpus).unwrap();
        assert!(
            booster.best_iteration < 30,
            "early stopping must fire before num_iterations (best_iteration={})",
            booster.best_iteration
        );
        assert!(booster.best_iteration >= 1);
        // The model's tree count == best_iteration (trailing trees popped).
        assert_eq!(
            booster.model().trees.len() as i32,
            booster.best_iteration,
            "trailing trees must be popped to best_iteration"
        );
        // valid_0 metrics are in the eval history.
        assert!(
            booster.eval_history.iter().any(|(n, _)| n.starts_with("valid_0")),
            "valid metrics must be recorded: {:?}",
            booster.eval_history.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn early_stopping_without_valid_set_is_typed_error() {
        let cfg = TrainingBuilder::new()
            .objective("regression")
            .num_iterations(10)
            .early_stopping_round(3)
            .num_leaves(4)
            .min_data_in_leaf(1)
            .seed(1)
            .deterministic(true)
            .build()
            .unwrap();
        let err = train(&cfg, &spine_corpus()).unwrap_err();
        assert!(matches!(err, LgbmError::Boosting(_)), "got {err:?}");
    }

    #[test]
    fn bagging_by_query_rejected_via_facade() {
        // The query-grouped draw is implemented + RNG-replay tested
        // (rank_parity), but the DenseCorpus facade carries no query/group boundaries
        // to bag by, so bagging_by_query=true here surfaces as a typed error (honest
        // "query information required"), NOT a silent fall-through to row bagging.
        let mut base = Config::default();
        base.objective = "regression".into();
        base.num_iterations = 5;
        base.num_leaves = 4;
        base.min_data_in_leaf = 1;
        base.bagging_freq = 1;
        base.bagging_fraction = 0.7;
        base.bagging_by_query = true;
        let cfg = TrainingBuilder::new().from_config(base).build().unwrap();
        let err = train(&cfg, &spine_corpus()).unwrap_err();
        assert!(matches!(err, LgbmError::Boosting(_)), "got {err:?}");
    }

    #[test]
    fn metric_freq_thins_eval_history() {
        // metric_freq=3 over 9 iters => training metrics recorded on iters 3,6,9
        // (the cadence gate), i.e. 3 values, not 9.
        let cfg = TrainingBuilder::new()
            .objective("regression")
            .num_iterations(9)
            .learning_rate(0.1)
            .num_leaves(4)
            .min_data_in_leaf(1)
            .boost_from_average(true)
            .metric_freq(3)
            // Opt in to training-metric history (C++-faithful).
            .is_provide_training_metric(true)
            .seed(1)
            .deterministic(true)
            .build()
            .unwrap();
        let booster = train(&cfg, &spine_corpus()).unwrap();
        // legacy l2 history thinned to the freq cadence.
        let (_n, vals) = booster
            .eval_history
            .iter()
            .find(|(n, _)| n == "l2")
            .unwrap();
        assert_eq!(vals.len(), 3, "metric_freq=3 over 9 iters => 3 recorded rounds");
    }

    #[test]
    fn build_feature_columns_rejects_non_identity() {
        // A non-consecutive column (missing bin 1) must be rejected.
        let corpus = DenseCorpus {
            features: vec![vec![0.0], vec![2.0]],
            labels: vec![1.0, 2.0],
        };
        assert!(build_feature_columns(&corpus).is_err());
    }

    // ---- raw→bin→train bridge ----

    /// A deterministic binning config: single-thread, fixed seed, large sample
    /// count so every row is sampled (so `find_bin_from_column` sees the full
    /// column) — the REFERENCE_MANIFEST shape for parity-stable binning.
    fn det_config(objective: &str) -> Config {
        let mut cfg = TrainingBuilder::new()
            .objective(objective)
            .num_iterations(10)
            .learning_rate(0.1)
            .num_leaves(4)
            .min_data_in_leaf(1)
            .boost_from_average(true)
            .metric("l2,rmse")
            .seed(1)
            .deterministic(true)
            .build()
            .unwrap();
        // min_data_in_bin=1 so each distinct integer value gets its OWN bin (the
        // identity-binning precondition the trivial-case equivalence relies on —
        // the default min_data_in_bin=3 would MERGE adjacent integer bins).
        cfg.min_data_in_bin = 1;
        cfg
    }

    #[test]
    fn raw_bin_equals_identity_bin_on_trivial_case() {
        // When the raw values ARE the consecutive integers 0..K-1, the BinMapper
        // collapses to identity binning, so build_feature_columns_from_raw must
        // produce the SAME bin indices + num_bin as build_feature_columns.
        let corpus = spine_corpus();
        let identity = build_feature_columns(&corpus).expect("identity ok");
        let raw = {
            let mut r = RawCorpus::new(corpus.features.clone(), corpus.labels.clone());
            r.config = det_config("regression");
            r
        };
        let bridged = build_feature_columns_from_raw(&raw).expect("raw ok");
        assert_eq!(identity.len(), bridged.len());
        for (i, (idc, brc)) in identity.iter().zip(bridged.iter()).enumerate() {
            assert_eq!(idc.bins, brc.bins, "feature {i} bins must match identity");
            assert_eq!(idc.num_bin, brc.num_bin, "feature {i} num_bin");
            assert_eq!(
                idc.most_freq_bin, brc.most_freq_bin,
                "feature {i} most_freq_bin"
            );
            assert_eq!(idc.offset, brc.offset, "feature {i} offset");
        }
    }

    #[test]
    fn raw_bin_train_equals_identity_bin_train_on_trivial_case() {
        // Training the SAME integer-valued corpus via the raw path and the identity
        // path must produce bit-identical model text (the raw bridge collapses to
        // the identity path when values are already 0..K-1).
        let cfg = det_config("regression");
        let corpus = spine_corpus();
        let id_booster = train(&cfg, &corpus).expect("identity train");
        let raw = {
            let mut r = RawCorpus::new(corpus.features.clone(), corpus.labels.clone());
            r.config = cfg.clone();
            r
        };
        let raw_booster = train_raw(&cfg, &raw).expect("raw train");
        // Leaf values bit-exact across all trees.
        assert_eq!(
            id_booster.model().trees.len(),
            raw_booster.model().trees.len()
        );
        for (ti, (idt, rwt)) in id_booster
            .model()
            .trees
            .iter()
            .zip(raw_booster.model().trees.iter())
            .enumerate()
        {
            for (li, (iv, rv)) in idt
                .leaf_value
                .iter()
                .zip(rwt.leaf_value.iter())
                .enumerate()
            {
                assert_eq!(
                    iv.to_bits(),
                    rv.to_bits(),
                    "tree {ti} leaf {li}: identity {iv} != raw {rv} (bit-exact)"
                );
            }
        }
    }

    #[test]
    fn raw_bin_train_real_values_trains_and_orders() {
        // A corpus of REAL continuous values trains via the raw path (binned by the
        // BinMapper) and predicts monotonically (low-label rows < high-label rows).
        let cfg = det_config("regression");
        let f0 = [0.1f64, 0.15, 1.2, 1.25, 2.7, 2.75, 3.9, 3.95, 5.1, 5.15, 6.6, 6.65];
        let f1 = [0.0f64, 1.1, 2.2, 0.05, 1.15, 2.25, 0.1, 1.2, 2.3, 0.0, 1.1, 2.2];
        let features: Vec<Vec<f64>> = (0..12).map(|i| vec![f0[i], f1[i]]).collect();
        let labels = vec![
            2.0f32, 3.0, 5.0, 6.0, 9.0, 10.0, 12.0, 13.0, 16.0, 17.0, 19.0, 20.0,
        ];
        let raw = {
            let mut r = RawCorpus::new(features.clone(), labels);
            r.config = cfg.clone();
            r
        };
        let booster = train_raw(&cfg, &raw).expect("raw real train");
        assert_eq!(booster.model().num_iteration(), 10);
        let p_low = booster.predict_row(&[0.1, 0.0]);
        let p_high = booster.predict_row(&[6.6, 2.2]);
        assert!(p_low[0] < p_high[0], "{} < {}", p_low[0], p_high[0]);
    }

    #[test]
    fn raw_bin_train_validates_shape() {
        let cfg = det_config("regression");
        // empty corpus
        let empty = RawCorpus::new(Vec::new(), Vec::new());
        assert!(matches!(
            build_feature_columns_from_raw(&empty),
            Err(LgbmError::InvalidCorpus { .. })
        ));
        // labels/num_data mismatch
        let mismatch = RawCorpus::new(vec![vec![0.0], vec![1.0]], vec![1.0]);
        assert!(matches!(
            build_feature_columns_from_raw(&mismatch),
            Err(LgbmError::InvalidCorpus { .. })
        ));
        // ragged row
        let ragged = RawCorpus::new(vec![vec![0.0, 1.0], vec![1.0]], vec![1.0, 2.0]);
        assert!(matches!(
            build_feature_columns_from_raw(&ragged),
            Err(LgbmError::InvalidCorpus { .. })
        ));
        // out-of-range categorical index
        let bad_cat = {
            let mut r = RawCorpus::new(vec![vec![0.0], vec![1.0]], vec![1.0, 2.0]);
            r.categorical_features = vec![5];
            r.config = cfg;
            r
        };
        assert!(matches!(
            build_feature_columns_from_raw(&bad_cat),
            Err(LgbmError::InvalidCorpus { .. })
        ));
    }

    // ---- Booster facade methods + custom-metric feval hook ----

    fn trained_spine() -> Booster {
        let cfg = TrainingBuilder::new()
            .objective("regression")
            .num_iterations(10)
            .learning_rate(0.1)
            .num_leaves(4)
            .min_data_in_leaf(1)
            .boost_from_average(true)
            .metric("l2,rmse")
            .seed(1)
            .deterministic(true)
            .build()
            .unwrap();
        train(&cfg, &spine_corpus()).expect("train ok")
    }

    #[test]
    fn booster_facade_batch_predict_equals_per_row() {
        let booster = trained_spine();
        let corpus = spine_corpus();
        let batch = booster.predict(&corpus.features);
        assert_eq!(batch.len(), corpus.features.len());
        for (i, row) in corpus.features.iter().enumerate() {
            let per_row = booster.predict_row(row);
            assert_eq!(batch[i].len(), per_row.len());
            for (a, b) in batch[i].iter().zip(per_row.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "row {i} batch != per-row");
            }
        }
    }

    #[test]
    fn booster_facade_importance_delegates() {
        let booster = trained_spine();
        // The facade 'split' importance is the C++-faithful GUARDED count
        // (split_gain > 0), matching the official Booster.feature_importance.
        assert_eq!(
            booster.feature_importance_split(),
            booster.model().feature_importance_split_count_guarded()
        );
        assert_eq!(
            booster.feature_importance_gain(),
            booster.model().feature_importance_gain()
        );
    }

    #[test]
    fn booster_facade_model_text_round_trips() {
        let booster = trained_spine();
        let text = booster.model_to_string();
        let reloaded = Booster::model_from_string(&text).expect("reload ok");
        let corpus = spine_corpus();
        for row in &corpus.features {
            let a = booster.predict_row(row);
            let b = reloaded.predict_row(row);
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(b.iter()) {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "round-trip predict must be bit-exact"
                );
            }
        }
    }

    #[test]
    fn booster_facade_refit_changes_leaves() {
        let mut booster = trained_spine();
        let corpus = spine_corpus();
        let before: Vec<f64> = booster.model().trees[0].leaf_value.clone();
        // Refit tree 0 with synthetic gradients (decay=0.0 fully replaces leaves).
        let grads = vec![5.0f32; corpus.features.len()];
        let hess = vec![1.0f32; corpus.features.len()];
        booster.refit(0, &corpus.features, &grads, &hess, 0.0, false, 0.0, 0.0);
        let after: Vec<f64> = booster.model().trees[0].leaf_value.clone();
        assert_ne!(before, after, "refit must change leaf values");
    }

    #[test]
    fn model_from_string_rejects_garbage() {
        let err = Booster::model_from_string("not a valid model").unwrap_err();
        assert!(matches!(err, LgbmError::Model(_)), "got {err:?}");
    }

    #[test]
    fn custom_metric_feval_matches_builtin_l2() {
        // The custom-metric (feval) hook records a user-supplied metric in eval
        // history under the closure name, and its per-iteration values bit-match
        // the built-in Metric::L2 over the same scores/labels — proving the hook
        // feeds the SAME eval-history loop, not a parallel one.
        let cfg = TrainingBuilder::new()
            .objective("regression")
            .num_iterations(8)
            .learning_rate(0.1)
            .num_leaves(4)
            .min_data_in_leaf(1)
            .boost_from_average(false) // custom forces bfa OFF
            // Opt in to training-metric history (C++-faithful).
            .is_provide_training_metric(true)
            .seed(1)
            .deterministic(true)
            .build()
            .unwrap();
        let corpus = spine_corpus();
        // Custom objective: L2 gradient/hessian (grad = score - label, hess = 1).
        let labels = corpus.labels.clone();
        let obj = move |scores: &[f64]| -> (Vec<f32>, Vec<f32>) {
            let g: Vec<f32> = scores
                .iter()
                .zip(labels.iter())
                .map(|(s, l)| (*s as f32) - *l)
                .collect();
            let h: Vec<f32> = vec![1.0f32; scores.len()];
            (g, h)
        };

        // Baseline: custom objective, built-in l2 metric.
        let baseline = train_custom_with_metric(&cfg, &corpus, obj.clone(), None)
            .expect("baseline custom train");
        let (_, baseline_l2) = baseline
            .eval_history
            .iter()
            .find(|(n, _)| n == "l2")
            .expect("built-in l2 recorded");

        // Custom metric: an in-closure L2 (mean (pred-label)^2), is_higher_better=false.
        let feval: CustomMetricClosure = Box::new(|scores: &[f64], labels: &[f32]| {
            let n = scores.len().max(1) as f64;
            let sse: f64 = scores
                .iter()
                .zip(labels.iter())
                .map(|(s, l)| {
                    let d = s - *l as f64;
                    d * d
                })
                .sum();
            ("my_l2".to_string(), sse / n, false)
        });
        let custom = train_custom_with_metric(&cfg, &corpus, obj, Some(feval))
            .expect("custom-metric train");
        let (_, custom_vals) = custom
            .eval_history
            .iter()
            .find(|(n, _)| n == "my_l2")
            .expect("custom metric recorded under closure name");

        assert_eq!(
            baseline_l2.len(),
            custom_vals.len(),
            "same number of recorded rounds"
        );
        for (i, (b, c)) in baseline_l2.iter().zip(custom_vals.iter()).enumerate() {
            assert_eq!(
                b.to_bits(),
                c.to_bits(),
                "iter {i}: custom feval {c} must bit-match built-in l2 {b}"
            );
        }
    }

    #[test]
    fn custom_metric_nan_is_typed_error() {
        let cfg = TrainingBuilder::new()
            .objective("regression")
            .num_iterations(3)
            .num_leaves(4)
            .min_data_in_leaf(1)
            .boost_from_average(false)
            // Opt in to training-metric history (C++-faithful).
            .is_provide_training_metric(true)
            .seed(1)
            .deterministic(true)
            .build()
            .unwrap();
        let corpus = spine_corpus();
        let obj = |scores: &[f64]| -> (Vec<f32>, Vec<f32>) {
            (vec![0.0f32; scores.len()], vec![1.0f32; scores.len()])
        };
        let feval: CustomMetricClosure =
            Box::new(|_s: &[f64], _l: &[f32]| ("bad".to_string(), f64::NAN, false));
        let err = train_custom_with_metric(&cfg, &corpus, obj, Some(feval))
            .expect_err("NaN feval must error");
        assert!(matches!(err, LgbmError::CustomMetric { .. }), "got {err:?}");
    }
}
