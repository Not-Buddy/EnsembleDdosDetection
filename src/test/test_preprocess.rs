use crate::config::ScalerConfig;
use crate::preprocess::Preprocessor;

fn make_test_config(n: usize, log_cols: Vec<&str>) -> ScalerConfig {
    ScalerConfig {
        mean: vec![0.0; n],
        scale: vec![1.0; n],
        feature_names: (0..n).map(|i| format!("feat_{}", i)).collect(),
        n_features: n,
        log_transformed_columns: log_cols.into_iter().map(String::from).collect(),
        dropped_mi_columns: vec![],
    }
}

#[test]
fn preprocessor_identity_transform() {
    // mean=0, scale=1, no log cols → features unchanged
    let config = make_test_config(3, vec![]);
    let pp = Preprocessor::from_config(&config);

    let mut features = vec![1.0, 2.0, 3.0];
    pp.transform(&mut features);

    assert!((features[0] - 1.0).abs() < 1e-10);
    assert!((features[1] - 2.0).abs() < 1e-10);
    assert!((features[2] - 3.0).abs() < 1e-10);
}

#[test]
fn preprocessor_standard_scaling() {
    let mut config = make_test_config(3, vec![]);
    config.mean = vec![10.0, 20.0, 30.0];
    config.scale = vec![2.0, 5.0, 10.0];

    let pp = Preprocessor::from_config(&config);

    let mut features = vec![12.0, 25.0, 50.0];
    pp.transform(&mut features);

    // (12 - 10) / 2 = 1.0
    assert!((features[0] - 1.0).abs() < 1e-10);
    // (25 - 20) / 5 = 1.0
    assert!((features[1] - 1.0).abs() < 1e-10);
    // (50 - 30) / 10 = 2.0
    assert!((features[2] - 2.0).abs() < 1e-10);
}

#[test]
fn preprocessor_zero_scale_becomes_zero() {
    let mut config = make_test_config(2, vec![]);
    config.mean = vec![5.0, 10.0];
    config.scale = vec![0.0, 1.0]; // Zero scale on first feature

    let pp = Preprocessor::from_config(&config);

    let mut features = vec![42.0, 11.0];
    pp.transform(&mut features);

    // Zero-scale feature → 0.0
    assert!((features[0] - 0.0).abs() < 1e-10);
    // Normal scaling: (11 - 10) / 1 = 1.0
    assert!((features[1] - 1.0).abs() < 1e-10);
}

#[test]
fn preprocessor_log_transform_positive() {
    // Log transform on feat_0: signed_log1p(x) = sign(x) * ln(1 + |x|)
    let config = make_test_config(3, vec!["feat_0"]);
    let pp = Preprocessor::from_config(&config);

    let mut features = vec![100.0, 5.0, 10.0];
    pp.transform(&mut features);

    // feat_0 should be log1p(100) = ln(101)
    let expected = (101.0_f64).ln();
    assert!(
        (features[0] - expected).abs() < 1e-10,
        "Expected {} but got {}",
        expected,
        features[0]
    );
    // Others unchanged (mean=0, scale=1)
    assert!((features[1] - 5.0).abs() < 1e-10);
    assert!((features[2] - 10.0).abs() < 1e-10);
}

#[test]
fn preprocessor_log_transform_negative() {
    // Signed log1p: negative values should get -ln(1 + |x|)
    let config = make_test_config(2, vec!["feat_0"]);
    let pp = Preprocessor::from_config(&config);

    let mut features = vec![-50.0, 1.0];
    pp.transform(&mut features);

    let expected = -(51.0_f64).ln();
    assert!(
        (features[0] - expected).abs() < 1e-10,
        "Expected {} but got {}",
        expected,
        features[0]
    );
}

#[test]
fn preprocessor_log_transform_zero() {
    let config = make_test_config(1, vec!["feat_0"]);
    let pp = Preprocessor::from_config(&config);

    let mut features = vec![0.0];
    pp.transform(&mut features);

    // sign(0) * ln(1 + 0) = 0 * 0 = 0
    assert!((features[0] - 0.0).abs() < 1e-10);
}

#[test]
fn preprocessor_log_then_scale() {
    // Verify order: log-transform THEN scale
    let mut config = make_test_config(1, vec!["feat_0"]);
    config.mean = vec![1.0];
    config.scale = vec![2.0];

    let pp = Preprocessor::from_config(&config);

    let mut features = vec![100.0];
    pp.transform(&mut features);

    // Step 1: log1p(100) = ln(101) ≈ 4.6151
    // Step 2: (4.6151 - 1.0) / 2.0 ≈ 1.8076
    let log_val = (101.0_f64).ln();
    let expected = (log_val - 1.0) / 2.0;
    assert!(
        (features[0] - expected).abs() < 1e-8,
        "Expected {} but got {}",
        expected,
        features[0]
    );
}

#[test]
#[should_panic(expected = "Expected 3 features, got 2")]
fn preprocessor_rejects_wrong_length() {
    let config = make_test_config(3, vec![]);
    let pp = Preprocessor::from_config(&config);

    let mut features = vec![1.0, 2.0]; // Wrong length
    pp.transform(&mut features);
}

#[test]
fn preprocessor_from_real_config() {
    // Load real scaler config and verify it creates without error
    let path = std::path::Path::new("models/exported/onnx/scaler.json");
    if !path.exists() {
        eprintln!("Skipping: scaler.json not found");
        return;
    }

    let content = std::fs::read_to_string(path).unwrap();
    let config: ScalerConfig = serde_json::from_str(&content).unwrap();
    let pp = Preprocessor::from_config(&config);

    // Create a 54-feature vector and transform it
    let mut features = vec![1.0; config.n_features];
    pp.transform(&mut features);

    // All features should be finite after transformation
    for (i, &f) in features.iter().enumerate() {
        assert!(f.is_finite(), "Feature {} is not finite after transform", i);
    }
}
