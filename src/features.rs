//! CICFlowMeter-style feature extraction from flow state.
//!
//! Converts a FlowState into a feature vector of exactly 54 features,
//! matching the order in `scaler.json["feature_names"]`.

use crate::flow::FlowState;

/// Extract the 54-feature vector from a completed flow.
///
/// Feature order MUST exactly match `scaler.json["feature_names"]`.
/// The order is hardcoded here to match the Python training pipeline.
pub fn extract_features(flow: &FlowState) -> Vec<f64> {
    let total_fwd = flow.fwd_packet_count as f64;
    let total_bwd = flow.bwd_packet_count as f64;
    let total_pkts = total_fwd + total_bwd;
    let duration_us = flow.duration_us();
    let duration_s = duration_us / 1_000_000.0;

    // Packets/second
    let flow_pkts_per_s = if duration_s > 0.0 {
        total_pkts / duration_s
    } else {
        0.0
    };
    let fwd_pkts_per_s = if duration_s > 0.0 {
        total_fwd / duration_s
    } else {
        0.0
    };
    let bwd_pkts_per_s = if duration_s > 0.0 {
        total_bwd / duration_s
    } else {
        0.0
    };

    // Bytes/second
    let flow_bytes_per_s = if duration_s > 0.0 {
        (flow.fwd_bytes_total + flow.bwd_bytes_total) as f64 / duration_s
    } else {
        0.0
    };

    // Down/Up ratio
    let down_up_ratio = if total_fwd > 0.0 {
        total_bwd / total_fwd
    } else {
        0.0
    };

    // Avg packet size
    let avg_pkt_size = if total_pkts > 0.0 {
        (flow.fwd_bytes_total + flow.bwd_bytes_total) as f64 / total_pkts
    } else {
        0.0
    };

    // Fwd seg size min
    let fwd_seg_min = if flow.fwd_seg_size_min == u32::MAX {
        0.0
    } else {
        flow.fwd_seg_size_min as f64
    };

    // Build the 54-feature vector in exact scaler.json order:
    vec![
        // 0: Protocol
        flow.protocol as f64,
        // 1: Flow Duration
        duration_us,
        // 2: Total Fwd Packets
        total_fwd,
        // 3: Total Backward Packets
        total_bwd,
        // 4: Fwd Packets Length Total
        flow.fwd_bytes_total as f64,
        // 5: Bwd Packets Length Total
        flow.bwd_bytes_total as f64,
        // 6: Fwd Packet Length Max
        flow.fwd_pkt_len.safe_max(),
        // 7: Fwd Packet Length Min
        flow.fwd_pkt_len.safe_min(),
        // 8: Fwd Packet Length Mean
        flow.fwd_pkt_len.mean(),
        // 9: Fwd Packet Length Std
        flow.fwd_pkt_len.std_dev(),
        // 10: Bwd Packet Length Max
        flow.bwd_pkt_len.safe_max(),
        // 11: Bwd Packet Length Min
        flow.bwd_pkt_len.safe_min(),
        // 12: Bwd Packet Length Mean
        flow.bwd_pkt_len.mean(),
        // 13: Bwd Packet Length Std
        flow.bwd_pkt_len.std_dev(),
        // 14: Flow Bytes/s
        flow_bytes_per_s,
        // 15: Flow Packets/s
        flow_pkts_per_s,
        // 16: Flow IAT Mean
        flow.flow_iat.mean(),
        // 17: Flow IAT Std
        flow.flow_iat.std_dev(),
        // 18: Flow IAT Max
        flow.flow_iat.safe_max(),
        // 19: Flow IAT Min
        flow.flow_iat.safe_min(),
        // 20: Fwd IAT Total
        flow.fwd_iat.sum,
        // 21: Fwd IAT Mean
        flow.fwd_iat.mean(),
        // 22: Fwd IAT Std
        flow.fwd_iat.std_dev(),
        // 23: Fwd IAT Max
        flow.fwd_iat.safe_max(),
        // 24: Fwd IAT Min
        flow.fwd_iat.safe_min(),
        // 25: Bwd IAT Total
        flow.bwd_iat.sum,
        // 26: Bwd IAT Mean
        flow.bwd_iat.mean(),
        // 27: Bwd IAT Std
        flow.bwd_iat.std_dev(),
        // 28: Bwd IAT Max
        flow.bwd_iat.safe_max(),
        // 29: Bwd IAT Min
        flow.bwd_iat.safe_min(),
        // 30: Fwd Header Length
        flow.fwd_header_len_total as f64,
        // 31: Bwd Header Length
        flow.bwd_header_len_total as f64,
        // 32: Fwd Packets/s
        fwd_pkts_per_s,
        // 33: Bwd Packets/s
        bwd_pkts_per_s,
        // 34: Packet Length Min
        flow.all_pkt_len.safe_min(),
        // 35: Packet Length Max
        flow.all_pkt_len.safe_max(),
        // 36: Packet Length Mean
        flow.all_pkt_len.mean(),
        // 37: Packet Length Std
        flow.all_pkt_len.std_dev(),
        // 38: Packet Length Variance
        flow.all_pkt_len.variance(),
        // 39: URG Flag Count
        flow.urg_count as f64,
        // 40: Down/Up Ratio
        down_up_ratio,
        // 41: Avg Packet Size
        avg_pkt_size,
        // 42: Avg Fwd Segment Size  (= Fwd Packet Length Mean)
        flow.fwd_pkt_len.mean(),
        // 43: Avg Bwd Segment Size  (= Bwd Packet Length Mean)
        flow.bwd_pkt_len.mean(),
        // 44: Subflow Fwd Packets   (= Total Fwd Packets for single subflow)
        total_fwd,
        // 45: Subflow Fwd Bytes     (= Fwd Packets Length Total)
        flow.fwd_bytes_total as f64,
        // 46: Subflow Bwd Packets   (= Total Backward Packets)
        total_bwd,
        // 47: Subflow Bwd Bytes     (= Bwd Packets Length Total)
        flow.bwd_bytes_total as f64,
        // 48: Init Fwd Win Bytes
        flow.init_fwd_win.unwrap_or(0) as f64,
        // 49: Init Bwd Win Bytes
        flow.init_bwd_win.unwrap_or(0) as f64,
        // 50: Fwd Act Data Packets
        flow.fwd_act_data_pkts as f64,
        // 51: Fwd Seg Size Min
        fwd_seg_min,
        // 52: Active Min
        flow.active_times.safe_min(),
        // 53: Idle Max
        flow.idle_times.safe_max(),
    ]
}
