//! Preprocessing: log-transform and StandardScaler.
//!
//! Replicates the Python preprocessing pipeline for Rust inference.

use crate::config::ScalerConfig;
use std::collections::HashSet;

/// Preprocessor that applies log-transform + StandardScaler.
pub struct Preprocessor {
    mean: Vec<f64>,
    scale: Vec<f64>,
    n_features: usize,
    log_transform_indices: Vec<usize>,
}

impl Preprocessor {
    /// Create from scaler config.
    pub fn from_config(config: &ScalerConfig) -> Self {
        // Map log-transformed column names to feature indices
        let log_cols: HashSet<&str> = config
            .log_transformed_columns
            .iter()
            .map(|s| s.as_str())
            .collect();

        let log_transform_indices: Vec<usize> = config
            .feature_names
            .iter()
            .enumerate()
            .filter_map(|(i, name)| {
                if log_cols.contains(name.as_str()) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();

        tracing::info!(
            "Preprocessor: {} features, {} log-transformed",
            config.n_features,
            log_transform_indices.len()
        );

        Self {
            mean: config.mean.clone(),
            scale: config.scale.clone(),
            n_features: config.n_features,
            log_transform_indices,
        }
    }

    /// Transform a raw feature vector into a scaled one.
    ///
    /// 1. Apply signed log1p to skewed features
    /// 2. StandardScaler: (x - mean) / scale
    pub fn transform(&self, features: &mut [f64]) {
        assert_eq!(
            features.len(),
            self.n_features,
            "Expected {} features, got {}",
            self.n_features,
            features.len()
        );

        // 1. Log-transform
        for &idx in &self.log_transform_indices {
            let x = features[idx];
            features[idx] = x.signum() * (1.0 + x.abs()).ln();
        }

        // 2. StandardScaler
        for (i, feat) in features.iter_mut().enumerate().take(self.n_features) {
            let scale = self.scale[i];
            if scale.abs() > 1e-10 {
                *feat = (*feat - self.mean[i]) / scale;
            } else {
                *feat = 0.0;
            }
        }
    }
}
