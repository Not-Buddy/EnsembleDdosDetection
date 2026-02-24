//! Configuration types deserialized from JSON model artifacts.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

/// Scaler config from `scaler.json`.
/// Contains StandardScaler parameters and feature engineering metadata.
#[derive(Debug, Deserialize)]
pub struct ScalerConfig {
    pub mean: Vec<f64>,
    pub scale: Vec<f64>,
    pub feature_names: Vec<String>,
    pub n_features: usize,
    pub log_transformed_columns: Vec<String>,
    pub dropped_mi_columns: Vec<String>,
}

/// Per-model normalization parameters from `normalization_params.json`.
#[derive(Debug, Deserialize)]
pub struct NormalizationParams {
    pub isolation_forest: IFNormParams,
    pub autoencoder: AENormParams,
    pub one_class_svm: SVMNormParams,
}

#[derive(Debug, Deserialize)]
pub struct IFNormParams {
    pub p1: f64,
    pub p99: f64,
}

#[derive(Debug, Deserialize)]
pub struct AENormParams {
    pub score_min: f64,
    pub score_max: f64,
}

#[derive(Debug, Deserialize)]
pub struct SVMNormParams {
    pub score_min: f64,
    pub score_max: f64,
}

/// Ensemble config from `ensemble_config.json`.
/// Contains LR coefficients and decision threshold.
#[derive(Debug, Deserialize)]
pub struct EnsembleConfig {
    #[serde(rename = "type")]
    pub combiner_type: String,
    pub n_models: usize,
    pub coefficients: Vec<f64>,
    pub intercept: f64,
    pub threshold: f64,
    pub model_names: Vec<String>,
    pub optimized: bool,
}

/// All model configs loaded from disk.
pub struct ModelConfigs {
    pub scaler: ScalerConfig,
    pub normalization: NormalizationParams,
    pub ensemble: EnsembleConfig,
}

impl ModelConfigs {
    /// Load all JSON configs from the models directory.
    pub fn load(models_dir: &Path) -> Result<Self> {
        let scaler: ScalerConfig = load_json(models_dir.join("scaler.json"))?;
        let normalization: NormalizationParams =
            load_json(models_dir.join("normalization_params.json"))?;
        let ensemble: EnsembleConfig = load_json(models_dir.join("ensemble_config.json"))?;

        tracing::info!(
            "Loaded configs: {} features, {} models",
            scaler.n_features,
            ensemble.n_models
        );

        Ok(Self {
            scaler,
            normalization,
            ensemble,
        })
    }
}

fn load_json<T: serde::de::DeserializeOwned>(path: impl AsRef<Path>) -> Result<T> {
    let path = path.as_ref();
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    serde_json::from_str(&content).with_context(|| format!("Failed to parse {}", path.display()))
}
