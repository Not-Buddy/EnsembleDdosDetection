//! Packet capture via pnet.
//!
//! Captures raw packets from a network interface, parses Ethernet/IP/TCP/UDP
//! headers, and emits `PacketInfo` structs.

use anyhow::{Context, Result};
use pnet::datalink::{self, Channel, NetworkInterface};
use pnet::packet::Packet;
use pnet::packet::ethernet::{EtherTypes, EthernetPacket};
use pnet::packet::ip::IpNextHeaderProtocols;
use pnet::packet::ipv4::Ipv4Packet;
use pnet::packet::tcp::TcpPacket;
use pnet::packet::udp::UdpPacket;
use std::net::IpAddr;
use std::sync::mpsc;
use std::time::Instant;

/// Parsed packet metadata needed for flow tracking.
#[derive(Debug, Clone)]
pub struct PacketInfo {
    pub src_ip: IpAddr,
    pub dst_ip: IpAddr,
    pub src_port: u16,
    pub dst_port: u16,
    pub protocol: u8,     // 6=TCP, 17=UDP
    pub payload_len: u32, // IP payload length
    pub header_len: u16,  // Transport header length
    pub tcp_flags: u8,    // TCP flags (SYN, ACK, etc.) or 0 for UDP
    pub window_size: u16, // TCP window size or 0
    pub timestamp: Instant,
}

/// TCP flag bitmasks
pub const TCP_SYN: u8 = 0x02;
pub const TCP_RST: u8 = 0x04;
pub const TCP_PSH: u8 = 0x08;
pub const TCP_ACK: u8 = 0x10;
pub const TCP_URG: u8 = 0x20;

/// Find a network interface by name.
pub fn find_interface(name: &str) -> Result<NetworkInterface> {
    datalink::interfaces()
        .into_iter()
        .find(|iface| iface.name == name)
        .with_context(|| {
            let available: Vec<String> = datalink::interfaces()
                .iter()
                .map(|i| i.name.clone())
                .collect();
            format!("Interface '{}' not found. Available: {:?}", name, available)
        })
}

/// Start capturing packets on the given interface.
/// Sends parsed `PacketInfo` over the channel.
pub fn start_capture(interface: NetworkInterface, tx: mpsc::Sender<PacketInfo>) -> Result<()> {
    let (_, mut rx) = match datalink::channel(&interface, Default::default())? {
        Channel::Ethernet(tx_chan, rx_chan) => (tx_chan, rx_chan),
        _ => anyhow::bail!("Unsupported channel type for interface {}", interface.name),
    };

    tracing::info!("Capturing on interface: {}", interface.name);

    loop {
        match rx.next() {
            Ok(packet_data) => {
                if let Some(info) = parse_packet(packet_data)
                    && tx.send(info).is_err()
                {
                    tracing::warn!("Packet channel closed, stopping capture");
                    break;
                }
            }
            Err(e) => {
                tracing::error!("Capture error: {}", e);
            }
        }
    }

    Ok(())
}

/// Parse raw Ethernet frame into PacketInfo.
fn parse_packet(data: &[u8]) -> Option<PacketInfo> {
    let eth = EthernetPacket::new(data)?;

    match eth.get_ethertype() {
        EtherTypes::Ipv4 => parse_ipv4(eth.payload()),
        _ => None, // Skip IPv6, ARP, etc. for now
    }
}

/// Parse IPv4 packet and extract transport layer info.
fn parse_ipv4(data: &[u8]) -> Option<PacketInfo> {
    let ipv4 = Ipv4Packet::new(data)?;
    let src_ip = IpAddr::V4(ipv4.get_source());
    let dst_ip = IpAddr::V4(ipv4.get_destination());
    let protocol = ipv4.get_next_level_protocol().0;
    let payload = ipv4.payload();

    match ipv4.get_next_level_protocol() {
        IpNextHeaderProtocols::Tcp => {
            let tcp = TcpPacket::new(payload)?;
            Some(PacketInfo {
                src_ip,
                dst_ip,
                src_port: tcp.get_source(),
                dst_port: tcp.get_destination(),
                protocol,
                payload_len: payload
                    .len()
                    .saturating_sub(tcp.get_data_offset() as usize * 4)
                    as u32,
                header_len: tcp.get_data_offset() as u16 * 4,
                tcp_flags: tcp.get_flags(),
                window_size: tcp.get_window(),
                timestamp: Instant::now(),
            })
        }
        IpNextHeaderProtocols::Udp => {
            let udp = UdpPacket::new(payload)?;
            Some(PacketInfo {
                src_ip,
                dst_ip,
                src_port: udp.get_source(),
                dst_port: udp.get_destination(),
                protocol,
                payload_len: udp.get_length().saturating_sub(8) as u32, // UDP header is 8 bytes
                header_len: 8,
                tcp_flags: 0,
                window_size: 0,
                timestamp: Instant::now(),
            })
        }
        _ => None,
    }
}
