use crate::capture::PacketInfo;
use crate::config::{ModelConfigs, ScalerConfig};
use crate::features::extract_features;
use crate::flow::{FlowKey, FlowState, FlowTable};
use crate::inference::InferenceEngine;
use crate::preprocess::Preprocessor;
use std::net::{IpAddr, Ipv4Addr};
use std::path::Path;
use std::time::Instant;

fn mock_packet(
    src: [u8; 4],
    dst: [u8; 4],
    src_port: u16,
    dst_port: u16,
    payload: u32,
    flags: u8,
) -> PacketInfo {
    PacketInfo {
        src_ip: IpAddr::V4(Ipv4Addr::from(src)),
        dst_ip: IpAddr::V4(Ipv4Addr::from(dst)),
        src_port,
        dst_port,
        protocol: 6,
        payload_len: payload,
        header_len: 20,
        tcp_flags: flags,
        window_size: 65535,
        timestamp: Instant::now(),
    }
}

/// End-to-end test: flow table → feature extraction → preprocessing → ONNX inference.
///
/// Requires ONNX models on disk. Skips gracefully when models are not available.
#[test]
fn end_to_end_pipeline() {
    let models_dir = Path::new("models/exported/onnx");
    if !models_dir.exists() {
        eprintln!("Skipping end-to-end test: models not found (run training first)");
        return;
    }

    // 1. Load configs
    let configs = ModelConfigs::load(models_dir).expect("Failed to load configs");
    let preprocessor = Preprocessor::from_config(&configs.scaler);
    let n_features = configs.scaler.n_features;
    let mut engine = InferenceEngine::load(
        models_dir,
        configs.normalization,
        configs.ensemble,
        n_features,
    )
    .expect("Failed to load ONNX models");

    // 2. Simulate a small flow
    let table = FlowTable::new(0); // 0s timeout for immediate expiry

    for i in 0..5 {
        let pkt = mock_packet(
            [192, 168, 1, 100],
            [10, 0, 0, 1],
            5000,
            80,
            100 + i * 50,
            0x10,
        );
        table.process_packet(&pkt);
    }
    for _ in 0..3 {
        let pkt = mock_packet([10, 0, 0, 1], [192, 168, 1, 100], 80, 5000, 200, 0x10);
        table.process_packet(&pkt);
    }

    // 3. Wait for timeout and sweep
    std::thread::sleep(std::time::Duration::from_millis(10));
    let expired = table.sweep_expired();
    assert!(!expired.is_empty(), "Should have at least one expired flow");

    for flow in &expired {
        // 4. Extract features
        let mut features = extract_features(flow);
        assert_eq!(features.len(), 54);

        // 5. Preprocess
        preprocessor.transform(&mut features);

        // All features should be finite
        for (i, &f) in features.iter().enumerate() {
            assert!(
                f.is_finite(),
                "Feature {} is not finite after preprocessing",
                i
            );
        }

        // 6. Run inference
        let result = engine.predict(&features).expect("Inference should succeed");

        // Scores should be in [0, 1]
        assert!(
            (0.0..=1.0).contains(&result.if_score),
            "IF score out of range: {}",
            result.if_score
        );
        assert!(
            (0.0..=1.0).contains(&result.ae_score),
            "AE score out of range: {}",
            result.ae_score
        );
        assert!(
            (0.0..=1.0).contains(&result.svm_score),
            "SVM score out of range: {}",
            result.svm_score
        );

        // Combined prob should be in [0, 1]
        assert!(
            (0.0..=1.0).contains(&result.combined_prob),
            "Combined prob out of range: {}",
            result.combined_prob
        );

        eprintln!(
            "  Flow: {} pkts | IF={:.3} AE={:.3} SVM={:.3} | prob={:.3} attack={}",
            flow.total_packets(),
            result.if_score,
            result.ae_score,
            result.svm_score,
            result.combined_prob,
            result.is_attack
        );
    }
}

/// Test that the feature count from extraction matches what the scaler expects.
#[test]
fn feature_count_matches_scaler() {
    let models_dir = Path::new("models/exported/onnx");
    if !models_dir.exists() {
        eprintln!("Skipping: models dir not found");
        return;
    }

    let content = std::fs::read_to_string(models_dir.join("scaler.json")).unwrap();
    let config: ScalerConfig = serde_json::from_str(&content).unwrap();

    let pkt = mock_packet([10, 0, 0, 1], [10, 0, 0, 2], 1234, 80, 100, 0);
    let (key, _) = FlowKey::from_packet(&pkt);
    let state = FlowState::new(key, &pkt);

    let features = extract_features(&state);
    assert_eq!(
        features.len(),
        config.n_features,
        "Extracted {} features but scaler expects {}",
        features.len(),
        config.n_features
    );
}
