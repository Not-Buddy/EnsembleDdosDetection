use crate::capture::PacketInfo;
use crate::flow::{FlowKey, FlowState, FlowTable, RunningStats};
use std::net::{IpAddr, Ipv4Addr};
use std::time::{Duration, Instant};

/// Helper: create a mock PacketInfo
fn mock_packet(
    src_ip: [u8; 4],
    dst_ip: [u8; 4],
    src_port: u16,
    dst_port: u16,
    protocol: u8,
    payload_len: u32,
    tcp_flags: u8,
) -> PacketInfo {
    PacketInfo {
        src_ip: IpAddr::V4(Ipv4Addr::from(src_ip)),
        dst_ip: IpAddr::V4(Ipv4Addr::from(dst_ip)),
        src_port,
        dst_port,
        protocol,
        payload_len,
        header_len: 20,
        tcp_flags,
        window_size: 65535,
        timestamp: Instant::now(),
    }
}

// ── RunningStats ──────────────────────────────────────────────────

#[test]
fn running_stats_empty() {
    let stats = RunningStats::new();
    assert_eq!(stats.count, 0);
    assert_eq!(stats.mean(), 0.0);
    assert_eq!(stats.std_dev(), 0.0);
    assert_eq!(stats.variance(), 0.0);
    assert_eq!(stats.safe_min(), 0.0);
    assert_eq!(stats.safe_max(), 0.0);
}

#[test]
fn running_stats_single_value() {
    let mut stats = RunningStats::new();
    stats.update(42.0);

    assert_eq!(stats.count, 1);
    assert!((stats.mean() - 42.0).abs() < 1e-10);
    assert_eq!(stats.std_dev(), 0.0); // need >= 2 for std
    assert!((stats.safe_min() - 42.0).abs() < 1e-10);
    assert!((stats.safe_max() - 42.0).abs() < 1e-10);
}

#[test]
fn running_stats_known_values() {
    let mut stats = RunningStats::new();
    let values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    for v in &values {
        stats.update(*v);
    }

    assert_eq!(stats.count, 8);
    assert!((stats.mean() - 5.0).abs() < 1e-10);
    assert!((stats.safe_min() - 2.0).abs() < 1e-10);
    assert!((stats.safe_max() - 9.0).abs() < 1e-10);
    assert!((stats.sum - 40.0).abs() < 1e-10);

    // Sample variance = sum((x - mean)^2) / (n-1) = 32/7 ≈ 4.571
    let expected_var = 32.0 / 7.0;
    assert!(
        (stats.variance() - expected_var).abs() < 1e-10,
        "Expected variance {}, got {}",
        expected_var,
        stats.variance()
    );

    // Sample std dev = sqrt(32/7) ≈ 2.138
    let expected_std = expected_var.sqrt();
    assert!(
        (stats.std_dev() - expected_std).abs() < 1e-10,
        "Expected std_dev {}, got {}",
        expected_std,
        stats.std_dev()
    );
}

// ── FlowKey ───────────────────────────────────────────────────────

#[test]
fn flow_key_from_packet() {
    let pkt = mock_packet([10, 0, 0, 1], [10, 0, 0, 2], 12345, 80, 6, 100, 0);
    let (key, is_fwd) = FlowKey::from_packet(&pkt);

    assert!(is_fwd);
    assert_eq!(key.src_port, 12345);
    assert_eq!(key.dst_port, 80);
    assert_eq!(key.protocol, 6);
}

#[test]
fn flow_key_direction_detection() {
    let pkt = mock_packet([10, 0, 0, 1], [10, 0, 0, 2], 12345, 80, 6, 100, 0);
    let (key, _) = FlowKey::from_packet(&pkt);

    // Same direction packet → forward
    let fwd_pkt = mock_packet([10, 0, 0, 1], [10, 0, 0, 2], 12345, 80, 6, 50, 0);
    assert!(key.is_forward(&fwd_pkt));

    // Reverse direction → backward
    let bwd_pkt = mock_packet([10, 0, 0, 2], [10, 0, 0, 1], 80, 12345, 6, 50, 0);
    assert!(!key.is_forward(&bwd_pkt));
}

// ── FlowState ─────────────────────────────────────────────────────

#[test]
fn flow_state_initial_packet() {
    let pkt = mock_packet([192, 168, 1, 1], [10, 0, 0, 1], 5000, 80, 6, 500, 0x02); // SYN
    let (key, _) = FlowKey::from_packet(&pkt);
    let state = FlowState::new(key, &pkt);

    assert_eq!(state.fwd_packet_count, 1);
    assert_eq!(state.bwd_packet_count, 0);
    assert_eq!(state.fwd_bytes_total, 500);
    assert_eq!(state.bwd_bytes_total, 0);
    assert_eq!(state.syn_count, 1);
    assert_eq!(state.protocol, 6);
    assert_eq!(state.total_packets(), 1);
}

#[test]
fn flow_state_bidirectional() {
    let fwd1 = mock_packet([192, 168, 1, 1], [10, 0, 0, 1], 5000, 80, 6, 500, 0x02);
    let (key, _) = FlowKey::from_packet(&fwd1);
    let mut state = FlowState::new(key, &fwd1);

    // Backward packet
    let bwd1 = mock_packet([10, 0, 0, 1], [192, 168, 1, 1], 80, 5000, 6, 200, 0x12); // SYN-ACK
    state.update(&bwd1);

    // Another forward
    let fwd2 = mock_packet([192, 168, 1, 1], [10, 0, 0, 1], 5000, 80, 6, 100, 0x10); // ACK
    state.update(&fwd2);

    assert_eq!(state.fwd_packet_count, 2);
    assert_eq!(state.bwd_packet_count, 1);
    assert_eq!(state.fwd_bytes_total, 600);
    assert_eq!(state.bwd_bytes_total, 200);
    assert_eq!(state.total_packets(), 3);
    assert_eq!(state.syn_count, 2); // SYN + SYN-ACK
    assert_eq!(state.ack_count, 2); // SYN-ACK + ACK
}

#[test]
fn flow_state_duration() {
    let pkt = mock_packet([10, 0, 0, 1], [10, 0, 0, 2], 1234, 80, 6, 100, 0);
    let (key, _) = FlowKey::from_packet(&pkt);
    let state = FlowState::new(key, &pkt);

    // Duration for a single-packet flow should be ~0
    assert!(state.duration_us() < 1000.0); // < 1ms
}

#[test]
fn flow_state_window_init() {
    let pkt = mock_packet([10, 0, 0, 1], [10, 0, 0, 2], 1234, 80, 6, 100, 0);
    let (key, _) = FlowKey::from_packet(&pkt);
    let state = FlowState::new(key, &pkt);

    assert_eq!(state.init_fwd_win, Some(65535));
    assert_eq!(state.init_bwd_win, None);
}

// ── FlowTable ─────────────────────────────────────────────────────

#[test]
fn flow_table_insert_and_count() {
    let table = FlowTable::new(120);
    assert_eq!(table.active_flow_count(), 0);

    let pkt1 = mock_packet([10, 0, 0, 1], [10, 0, 0, 2], 1000, 80, 6, 100, 0);
    table.process_packet(&pkt1);
    assert_eq!(table.active_flow_count(), 1);

    // Same flow
    let pkt2 = mock_packet([10, 0, 0, 1], [10, 0, 0, 2], 1000, 80, 6, 200, 0);
    table.process_packet(&pkt2);
    assert_eq!(table.active_flow_count(), 1); // still 1 flow

    // New flow
    let pkt3 = mock_packet([10, 0, 0, 3], [10, 0, 0, 4], 2000, 443, 6, 300, 0);
    table.process_packet(&pkt3);
    assert_eq!(table.active_flow_count(), 2);
}

#[test]
fn flow_table_reverse_direction_merges() {
    let table = FlowTable::new(120);

    // Forward packet
    let fwd = mock_packet([10, 0, 0, 1], [10, 0, 0, 2], 5000, 80, 6, 100, 0);
    table.process_packet(&fwd);

    // Reverse direction should merge into same flow
    let bwd = mock_packet([10, 0, 0, 2], [10, 0, 0, 1], 80, 5000, 6, 200, 0);
    table.process_packet(&bwd);

    assert_eq!(table.active_flow_count(), 1);
}

#[test]
fn flow_table_sweep_no_expired() {
    let table = FlowTable::new(120);

    let pkt = mock_packet([10, 0, 0, 1], [10, 0, 0, 2], 1000, 80, 6, 100, 0);
    table.process_packet(&pkt);

    let expired = table.sweep_expired();
    assert!(expired.is_empty()); // Nothing expired yet
    assert_eq!(table.active_flow_count(), 1);
}

#[test]
fn flow_table_sweep_with_timeout() {
    // Use a very short timeout (0 seconds) so flows expire immediately
    let table = FlowTable::new(0);

    let pkt1 = mock_packet([10, 0, 0, 1], [10, 0, 0, 2], 1000, 80, 6, 100, 0);
    table.process_packet(&pkt1);
    // Add a second packet so it meets the >= 2 threshold
    let pkt2 = mock_packet([10, 0, 0, 1], [10, 0, 0, 2], 1000, 80, 6, 100, 0);
    table.process_packet(&pkt2);

    // Wait a tiny bit for the flow to become "expired"
    std::thread::sleep(Duration::from_millis(10));

    let expired = table.sweep_expired();
    assert_eq!(expired.len(), 1);
    assert_eq!(table.active_flow_count(), 0);
}

#[test]
fn flow_table_sweep_skips_single_packet_flows() {
    let table = FlowTable::new(0);

    let pkt = mock_packet([10, 0, 0, 1], [10, 0, 0, 2], 1000, 80, 6, 100, 0);
    table.process_packet(&pkt);

    std::thread::sleep(Duration::from_millis(10));

    let expired = table.sweep_expired();
    assert!(expired.is_empty()); // Dropped because total_packets < 2
}
