//! Ensemble DDoS Detection — Real-Time Packet Ingestion Agent
//!
//! Captures live network traffic, computes CICFlowMeter-style features,
//! runs ONNX inference through 3 anomaly detectors (IF, VAE, SVM),
//! and alerts on detected DDoS attacks.

mod capture;
mod config;
mod features;
mod flow;
mod inference;
mod preprocess;

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Duration;

use crate::config::ModelConfigs;
use crate::flow::FlowTable;
use crate::inference::InferenceEngine;
use crate::preprocess::Preprocessor;

/// Real-time DDoS detection using ensemble anomaly detection.
#[derive(Parser, Debug)]
#[command(name = "ensemble-ddos-detect", version, about)]
struct Cli {
    /// Network interface to capture on (e.g. eth0, wlan0)
    #[arg(short, long)]
    interface: String,

    /// Path to the models directory (containing ONNX + JSON configs)
    #[arg(short, long, default_value = "models/exported/onnx")]
    models_dir: PathBuf,

    /// Flow inactivity timeout in seconds
    #[arg(short, long, default_value_t = 120)]
    timeout: u64,

    /// Flow sweep interval in seconds
    #[arg(short, long, default_value_t = 10)]
    sweep_interval: u64,
}

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("ensemble_ddos_detection=info".parse()?),
        )
        .init();

    let cli = Cli::parse();

    tracing::info!("═══════════════════════════════════════════════════");
    tracing::info!(
        "  Ensemble DDoS Detection Agent v{}",
        env!("CARGO_PKG_VERSION")
    );
    tracing::info!("═══════════════════════════════════════════════════");
    tracing::info!("Interface:     {}", cli.interface);
    tracing::info!("Models dir:    {}", cli.models_dir.display());
    tracing::info!("Flow timeout:  {}s", cli.timeout);

    // ── Load configs ──────────────────────────────────────────────────
    let configs = ModelConfigs::load(&cli.models_dir)?;
    let preprocessor = Preprocessor::from_config(&configs.scaler);
    let n_features = configs.scaler.n_features;
    let mut engine = InferenceEngine::load(
        &cli.models_dir,
        configs.normalization,
        configs.ensemble,
        n_features,
    )?;

    // ── Flow table ────────────────────────────────────────────────────
    let flow_table = FlowTable::new(cli.timeout);

    // ── Packet capture channel ────────────────────────────────────────
    let (pkt_tx, pkt_rx) = mpsc::channel();

    // ── Start capture thread ──────────────────────────────────────────
    let iface = capture::find_interface(&cli.interface)?;
    let capture_handle = std::thread::Builder::new()
        .name("packet-capture".into())
        .spawn(move || {
            if let Err(e) = capture::start_capture(iface, pkt_tx) {
                tracing::error!("Capture thread error: {}", e);
            }
        })?;

    tracing::info!("Capture started. Waiting for packets...\n");

    // ── Main loop ─────────────────────────────────────────────────────
    let sweep_interval = Duration::from_secs(cli.sweep_interval);
    let mut last_sweep = std::time::Instant::now();
    let mut total_packets: u64 = 0;
    let mut total_flows_classified: u64 = 0;
    let mut total_attacks: u64 = 0;

    loop {
        // Process incoming packets (with timeout so we can sweep)
        match pkt_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(pkt) => {
                flow_table.process_packet(&pkt);
                total_packets += 1;

                if total_packets % 10_000 == 0 {
                    tracing::info!(
                        "Packets: {} | Active flows: {} | Classified: {} | Attacks: {}",
                        total_packets,
                        flow_table.active_flow_count(),
                        total_flows_classified,
                        total_attacks
                    );
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                tracing::warn!("Capture channel disconnected");
                break;
            }
        }

        // ── Periodic sweep ────────────────────────────────────────────
        if last_sweep.elapsed() >= sweep_interval {
            let expired_flows = flow_table.sweep_expired();

            for flow_state in &expired_flows {
                // Extract features
                let mut features = features::extract_features(flow_state);

                // Preprocess
                preprocessor.transform(&mut features);

                // Inference
                match engine.predict(&features) {
                    Ok(result) => {
                        total_flows_classified += 1;
                        if result.is_attack {
                            total_attacks += 1;
                            tracing::warn!(
                                "🚨 ATTACK DETECTED | {} → {} | proto={} | pkts={} | prob={:.3} | IF={:.3} AE={:.3} SVM={:.3}",
                                flow_state.key.src_ip,
                                flow_state.key.dst_ip,
                                flow_state.key.protocol,
                                flow_state.total_packets(),
                                result.combined_prob,
                                result.if_score,
                                result.ae_score,
                                result.svm_score
                            );
                        } else {
                            tracing::debug!(
                                "✓ Benign | {} → {} | prob={:.3}",
                                flow_state.key.src_ip,
                                flow_state.key.dst_ip,
                                result.combined_prob
                            );
                        }
                    }
                    Err(e) => {
                        tracing::error!("Inference error: {}", e);
                    }
                }
            }

            if !expired_flows.is_empty() {
                tracing::info!(
                    "Swept {} flows ({} attacks / {} benign)",
                    expired_flows.len(),
                    expired_flows.iter().filter(|_| true).count(),
                    0
                );
            }

            last_sweep = std::time::Instant::now();
        }
    }

    capture_handle.join().ok();
    Ok(())
}
