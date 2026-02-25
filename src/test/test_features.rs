use crate::capture::PacketInfo;
use crate::features::extract_features;
use crate::flow::{FlowKey, FlowState};
use std::net::{IpAddr, Ipv4Addr};
use std::time::Instant;

fn mock_packet(payload_len: u32, tcp_flags: u8) -> PacketInfo {
    PacketInfo {
        src_ip: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
        dst_ip: IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)),
        src_port: 5000,
        dst_port: 80,
        protocol: 6,
        payload_len,
        header_len: 20,
        tcp_flags,
        window_size: 65535,
        timestamp: Instant::now(),
    }
}

fn mock_bwd_packet(payload_len: u32) -> PacketInfo {
    PacketInfo {
        src_ip: IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)),
        dst_ip: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
        src_port: 80,
        dst_port: 5000,
        protocol: 6,
        payload_len,
        header_len: 20,
        tcp_flags: 0x10, // ACK
        window_size: 32768,
        timestamp: Instant::now(),
    }
}

#[test]
fn feature_vector_has_54_elements() {
    let pkt = mock_packet(100, 0x02);
    let (key, _) = FlowKey::from_packet(&pkt);
    let mut state = FlowState::new(key, &pkt);

    // Add another packet so flow is non-trivial
    let pkt2 = mock_packet(200, 0x10);
    state.update(&pkt2);

    let features = extract_features(&state);
    assert_eq!(
        features.len(),
        54,
        "Feature vector must have exactly 54 elements"
    );
}

#[test]
fn feature_protocol_is_first() {
    let pkt = mock_packet(100, 0);
    let (key, _) = FlowKey::from_packet(&pkt);
    let state = FlowState::new(key, &pkt);

    let features = extract_features(&state);
    // Feature 0 = Protocol (TCP = 6)
    assert!(
        (features[0] - 6.0).abs() < 1e-10,
        "features[0] should be protocol number"
    );
}

#[test]
fn feature_packet_counts_correct() {
    let fwd1 = mock_packet(100, 0x02);
    let (key, _) = FlowKey::from_packet(&fwd1);
    let mut state = FlowState::new(key, &fwd1);

    let fwd2 = mock_packet(200, 0x10);
    state.update(&fwd2);

    let bwd1 = mock_bwd_packet(150);
    state.update(&bwd1);

    let features = extract_features(&state);

    // Feature 2 = Total Fwd Packets
    assert!((features[2] - 2.0).abs() < 1e-10, "Expected 2 fwd packets");
    // Feature 3 = Total Backward Packets
    assert!((features[3] - 1.0).abs() < 1e-10, "Expected 1 bwd packet");
}

#[test]
fn feature_byte_totals_correct() {
    let fwd1 = mock_packet(100, 0);
    let (key, _) = FlowKey::from_packet(&fwd1);
    let mut state = FlowState::new(key, &fwd1);

    let fwd2 = mock_packet(200, 0);
    state.update(&fwd2);

    let bwd1 = mock_bwd_packet(150);
    state.update(&bwd1);

    let features = extract_features(&state);

    // Feature 4 = Fwd Packets Length Total
    assert!((features[4] - 300.0).abs() < 1e-10, "Fwd bytes = 100 + 200");
    // Feature 5 = Bwd Packets Length Total
    assert!((features[5] - 150.0).abs() < 1e-10, "Bwd bytes = 150");
}

#[test]
fn feature_fwd_packet_length_stats() {
    let fwd1 = mock_packet(100, 0);
    let (key, _) = FlowKey::from_packet(&fwd1);
    let mut state = FlowState::new(key, &fwd1);

    let fwd2 = mock_packet(200, 0);
    state.update(&fwd2);

    let features = extract_features(&state);

    // Feature 6 = Fwd Packet Length Max
    assert!((features[6] - 200.0).abs() < 1e-10);
    // Feature 7 = Fwd Packet Length Min
    assert!((features[7] - 100.0).abs() < 1e-10);
    // Feature 8 = Fwd Packet Length Mean
    assert!((features[8] - 150.0).abs() < 1e-10);
}

#[test]
fn feature_no_nan_or_inf() {
    // Single packet flow — many zero-division edge cases
    let pkt = mock_packet(0, 0);
    let (key, _) = FlowKey::from_packet(&pkt);
    let state = FlowState::new(key, &pkt);

    let features = extract_features(&state);
    for (i, &f) in features.iter().enumerate() {
        assert!(f.is_finite(), "Feature {} is not finite: {}", i, f);
    }
}

#[test]
fn feature_init_window_sizes() {
    let fwd1 = mock_packet(100, 0);
    let (key, _) = FlowKey::from_packet(&fwd1);
    let mut state = FlowState::new(key, &fwd1);

    let bwd1 = mock_bwd_packet(100);
    state.update(&bwd1);

    let features = extract_features(&state);

    // Feature 48 = Init Fwd Win Bytes
    assert!((features[48] - 65535.0).abs() < 1e-10);
    // Feature 49 = Init Bwd Win Bytes
    assert!((features[49] - 32768.0).abs() < 1e-10);
}

#[test]
fn feature_subflow_mirrors_totals() {
    let fwd1 = mock_packet(100, 0);
    let (key, _) = FlowKey::from_packet(&fwd1);
    let mut state = FlowState::new(key, &fwd1);
    let bwd1 = mock_bwd_packet(50);
    state.update(&bwd1);

    let features = extract_features(&state);

    // Subflow Fwd Packets (44) == Total Fwd Packets (2)
    assert!((features[44] - features[2]).abs() < 1e-10);
    // Subflow Fwd Bytes (45) == Fwd Packets Length Total (4)
    assert!((features[45] - features[4]).abs() < 1e-10);
    // Subflow Bwd Packets (46) == Total Backward Packets (3)
    assert!((features[46] - features[3]).abs() < 1e-10);
    // Subflow Bwd Bytes (47) == Bwd Packets Length Total (5)
    assert!((features[47] - features[5]).abs() < 1e-10);
}
