//! Flow tracking: groups packets into bidirectional flows by 5-tuple.
//!
//! Accumulates per-flow statistics needed for CICFlowMeter feature extraction.

use crate::capture::{PacketInfo, TCP_ACK, TCP_PSH, TCP_RST, TCP_SYN, TCP_URG};
use dashmap::DashMap;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// 5-tuple flow key (direction-normalized: smaller IP is always "source").
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct FlowKey {
    pub src_ip: IpAddr,
    pub dst_ip: IpAddr,
    pub src_port: u16,
    pub dst_port: u16,
    pub protocol: u8,
}

impl FlowKey {
    /// Create a normalized key where the "forward" direction is from
    /// the first packet's source to destination.
    pub fn from_packet(pkt: &PacketInfo) -> (Self, bool) {
        // Forward direction = first packet direction
        let key = FlowKey {
            src_ip: pkt.src_ip,
            dst_ip: pkt.dst_ip,
            src_port: pkt.src_port,
            dst_port: pkt.dst_port,
            protocol: pkt.protocol,
        };
        (key, true) // true = forward
    }

    /// Check if a packet is in the forward or backward direction.
    pub fn is_forward(&self, pkt: &PacketInfo) -> bool {
        pkt.src_ip == self.src_ip && pkt.src_port == self.src_port
    }
}

/// Running statistics tracker (Welford's online algorithm).
#[derive(Debug, Clone)]
pub struct RunningStats {
    pub count: u64,
    pub sum: f64,
    pub min: f64,
    pub max: f64,
    mean: f64,
    m2: f64, // sum of squares of differences from the current mean
}

impl RunningStats {
    pub fn new() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            min: f64::MAX,
            max: f64::MIN,
            mean: 0.0,
            m2: 0.0,
        }
    }

    pub fn update(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
        // Welford's online algorithm for mean and variance
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    pub fn mean(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.mean }
    }

    pub fn std_dev(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            (self.m2 / (self.count - 1) as f64).sqrt()
        }
    }

    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count - 1) as f64
        }
    }

    pub fn safe_min(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.min }
    }

    pub fn safe_max(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.max }
    }
}

/// Per-flow accumulated state for CICFlowMeter feature extraction.
#[derive(Debug, Clone)]
pub struct FlowState {
    pub key: FlowKey,
    pub start_time: Instant,
    pub last_seen: Instant,

    // ── Packet counts ──
    pub fwd_packet_count: u64,
    pub bwd_packet_count: u64,

    // ── Byte totals ──
    pub fwd_bytes_total: u64,
    pub bwd_bytes_total: u64,

    // ── Packet length stats ──
    pub fwd_pkt_len: RunningStats,
    pub bwd_pkt_len: RunningStats,
    pub all_pkt_len: RunningStats,

    // ── Inter-arrival times ──
    pub flow_iat: RunningStats,
    pub fwd_iat: RunningStats,
    pub bwd_iat: RunningStats,
    pub last_fwd_time: Option<Instant>,
    pub last_bwd_time: Option<Instant>,
    pub last_pkt_time: Option<Instant>,

    // ── Header lengths ──
    pub fwd_header_len_total: u64,
    pub bwd_header_len_total: u64,

    // ── TCP flags ──
    pub syn_count: u32,
    pub ack_count: u32,
    pub urg_count: u32,
    pub rst_count: u32,
    pub psh_fwd_count: u32,

    // ── Window sizes ──
    pub init_fwd_win: Option<u16>,
    pub init_bwd_win: Option<u16>,

    // ── Data packets ──
    pub fwd_act_data_pkts: u64,
    pub fwd_seg_size_min: u32,

    // ── Active/Idle times ──
    pub active_times: RunningStats,
    pub idle_times: RunningStats,
    active_start: Option<Instant>,
    pub protocol: u8,
}

impl FlowState {
    pub fn new(key: FlowKey, pkt: &PacketInfo) -> Self {
        let mut state = Self {
            key,
            start_time: pkt.timestamp,
            last_seen: pkt.timestamp,
            fwd_packet_count: 0,
            bwd_packet_count: 0,
            fwd_bytes_total: 0,
            bwd_bytes_total: 0,
            fwd_pkt_len: RunningStats::new(),
            bwd_pkt_len: RunningStats::new(),
            all_pkt_len: RunningStats::new(),
            flow_iat: RunningStats::new(),
            fwd_iat: RunningStats::new(),
            bwd_iat: RunningStats::new(),
            last_fwd_time: None,
            last_bwd_time: None,
            last_pkt_time: None,
            fwd_header_len_total: 0,
            bwd_header_len_total: 0,
            syn_count: 0,
            ack_count: 0,
            urg_count: 0,
            rst_count: 0,
            psh_fwd_count: 0,
            init_fwd_win: None,
            init_bwd_win: None,
            fwd_act_data_pkts: 0,
            fwd_seg_size_min: u32::MAX,
            active_times: RunningStats::new(),
            idle_times: RunningStats::new(),
            active_start: Some(pkt.timestamp),
            protocol: pkt.protocol,
        };
        state.update(pkt);
        state
    }

    /// Update flow state with a new packet.
    pub fn update(&mut self, pkt: &PacketInfo) {
        let is_fwd = self.key.is_forward(pkt);
        let pkt_len = pkt.payload_len as f64;

        // ── Flow-level IAT ──
        if let Some(last) = self.last_pkt_time {
            let iat = pkt.timestamp.duration_since(last).as_micros() as f64;
            self.flow_iat.update(iat);

            // Active/idle tracking (threshold: 5 seconds)
            if iat > 5_000_000.0 {
                // Idle period
                if let Some(active_start) = self.active_start.take() {
                    let active_dur = last.duration_since(active_start).as_micros() as f64;
                    self.active_times.update(active_dur);
                }
                self.idle_times.update(iat);
                self.active_start = Some(pkt.timestamp);
            }
        }
        self.last_pkt_time = Some(pkt.timestamp);
        self.last_seen = pkt.timestamp;

        // ── Packet length stats ──
        self.all_pkt_len.update(pkt_len);

        if is_fwd {
            self.fwd_packet_count += 1;
            self.fwd_bytes_total += pkt.payload_len as u64;
            self.fwd_pkt_len.update(pkt_len);
            self.fwd_header_len_total += pkt.header_len as u64;

            if let Some(last) = self.last_fwd_time {
                let iat = pkt.timestamp.duration_since(last).as_micros() as f64;
                self.fwd_iat.update(iat);
            }
            self.last_fwd_time = Some(pkt.timestamp);

            if self.init_fwd_win.is_none() {
                self.init_fwd_win = Some(pkt.window_size);
            }
            if pkt.payload_len > 0 {
                self.fwd_act_data_pkts += 1;
            }
            if (pkt.header_len as u32) < self.fwd_seg_size_min {
                self.fwd_seg_size_min = pkt.header_len as u32;
            }
            if pkt.tcp_flags & TCP_PSH != 0 {
                self.psh_fwd_count += 1;
            }
        } else {
            self.bwd_packet_count += 1;
            self.bwd_bytes_total += pkt.payload_len as u64;
            self.bwd_pkt_len.update(pkt_len);
            self.bwd_header_len_total += pkt.header_len as u64;

            if let Some(last) = self.last_bwd_time {
                let iat = pkt.timestamp.duration_since(last).as_micros() as f64;
                self.bwd_iat.update(iat);
            }
            self.last_bwd_time = Some(pkt.timestamp);

            if self.init_bwd_win.is_none() {
                self.init_bwd_win = Some(pkt.window_size);
            }
        }

        // ── TCP flags ──
        if pkt.tcp_flags & TCP_SYN != 0 {
            self.syn_count += 1;
        }
        if pkt.tcp_flags & TCP_ACK != 0 {
            self.ack_count += 1;
        }
        if pkt.tcp_flags & TCP_URG != 0 {
            self.urg_count += 1;
        }
        if pkt.tcp_flags & TCP_RST != 0 {
            self.rst_count += 1;
        }
    }

    /// Flow duration in microseconds.
    pub fn duration_us(&self) -> f64 {
        self.last_seen.duration_since(self.start_time).as_micros() as f64
    }

    /// Check if flow should be exported (timed out or terminated).
    pub fn is_expired(&self, timeout: Duration) -> bool {
        self.last_seen.elapsed() > timeout
    }

    /// Total packets in flow.
    pub fn total_packets(&self) -> u64 {
        self.fwd_packet_count + self.bwd_packet_count
    }
}

/// Concurrent flow table.
pub struct FlowTable {
    flows: Arc<DashMap<FlowKey, FlowState>>,
    timeout: Duration,
}

impl FlowTable {
    pub fn new(timeout_secs: u64) -> Self {
        Self {
            flows: Arc::new(DashMap::new()),
            timeout: Duration::from_secs(timeout_secs),
        }
    }

    /// Process a packet: create new flow or update existing one.
    pub fn process_packet(&self, pkt: &PacketInfo) {
        let (key, _is_fwd) = FlowKey::from_packet(pkt);

        // Try exact match first
        if let Some(mut entry) = self.flows.get_mut(&key) {
            entry.update(pkt);
            return;
        }

        // Try reverse direction
        let rev_key = FlowKey {
            src_ip: key.dst_ip,
            dst_ip: key.src_ip,
            src_port: key.dst_port,
            dst_port: key.src_port,
            protocol: key.protocol,
        };

        if let Some(mut entry) = self.flows.get_mut(&rev_key) {
            entry.update(pkt);
            return;
        }

        // New flow
        let state = FlowState::new(key.clone(), pkt);
        self.flows.insert(key, state);
    }

    /// Sweep expired flows and return them for classification.
    pub fn sweep_expired(&self) -> Vec<FlowState> {
        let mut expired = Vec::new();
        let mut keys_to_remove = Vec::new();

        for entry in self.flows.iter() {
            if entry.value().is_expired(self.timeout) {
                keys_to_remove.push(entry.key().clone());
            }
        }

        for key in keys_to_remove {
            if let Some((_, state)) = self.flows.remove(&key)
                && state.total_packets() >= 2
            {
                expired.push(state);
            }
        }

        expired
    }

    /// Number of active flows.
    pub fn active_flow_count(&self) -> usize {
        self.flows.len()
    }
}
