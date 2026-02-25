use crate::config::{EnsembleConfig, ModelConfigs, NormalizationParams, ScalerConfig};
use std::path::Path;

#[test]
fn parse_scaler_json() {
    let json = r#"{
        "mean": [1.0, 2.0, 3.0],
        "scale": [0.5, 1.5, 2.5],
        "feature_names": ["feat_a", "feat_b", "feat_c"],
        "n_features": 3,
        "log_transformed_columns": ["feat_a"],
        "dropped_mi_columns": ["dropped_col"]
    }"#;

    let config: ScalerConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.n_features, 3);
    assert_eq!(config.mean, vec![1.0, 2.0, 3.0]);
    assert_eq!(config.scale, vec![0.5, 1.5, 2.5]);
    assert_eq!(config.feature_names.len(), 3);
    assert_eq!(config.log_transformed_columns, vec!["feat_a"]);
}

#[test]
fn parse_normalization_params_json() {
    let json = r#"{
        "isolation_forest": { "p1": -0.5, "p99": 0.1 },
        "autoencoder": { "score_min": 0.001, "score_max": 5.0 },
        "one_class_svm": { "score_min": -2.0, "score_max": 1.0 }
    }"#;

    let params: NormalizationParams = serde_json::from_str(json).unwrap();
    assert!((params.isolation_forest.p1 - (-0.5)).abs() < 1e-10);
    assert!((params.isolation_forest.p99 - 0.1).abs() < 1e-10);
    assert!((params.autoencoder.score_min - 0.001).abs() < 1e-10);
    assert!((params.autoencoder.score_max - 5.0).abs() < 1e-10);
    assert!((params.one_class_svm.score_min - (-2.0)).abs() < 1e-10);
    assert!((params.one_class_svm.score_max - 1.0).abs() < 1e-10);
}

#[test]
fn parse_ensemble_config_json() {
    let json = r#"{
        "type": "logistic_regression",
        "n_models": 3,
        "coefficients": [-1.45, 14.21, 1.49],
        "intercept": -3.37,
        "threshold": 0.325,
        "model_names": ["isolation_forest", "autoencoder", "one_class_svm"],
        "optimized": true,
        "ensemble_result": {}
    }"#;

    let config: EnsembleConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.coefficients.len(), 3);
    assert!((config.coefficients[0] - (-1.45)).abs() < 1e-10);
    assert!((config.intercept - (-3.37)).abs() < 1e-10);
    assert!((config.threshold - 0.325).abs() < 1e-10);
}

#[test]
fn load_real_configs_from_disk() {
    let models_dir = Path::new("models/exported/onnx");
    if !models_dir.exists() {
        eprintln!("Skipping: models dir not found (run training first)");
        return;
    }

    let configs = ModelConfigs::load(models_dir).expect("Failed to load model configs");
    assert_eq!(configs.scaler.n_features, 54);
    assert_eq!(configs.scaler.mean.len(), 54);
    assert_eq!(configs.scaler.scale.len(), 54);
    assert_eq!(configs.ensemble.coefficients.len(), 3);
    assert!(configs.ensemble.threshold > 0.0 && configs.ensemble.threshold < 1.0);
}

#[test]
fn config_rejects_malformed_json() {
    let bad_json = r#"{ "mean": [1.0], "scale": "not_an_array" }"#;
    let result: Result<ScalerConfig, _> = serde_json::from_str(bad_json);
    assert!(result.is_err());
}
