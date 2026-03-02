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
mod tui;

#[cfg(test)]
mod test;

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

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

/// Format an elapsed duration as MM:SS
fn format_elapsed(start: Instant) -> String {
    let secs = start.elapsed().as_secs();
    format!("{:02}:{:02}", secs / 60, secs % 60)
}

/// Format a protocol number as a name
fn proto_name(proto: u8) -> &'static str {
    match proto {
        6 => "TCP",
        17 => "UDP",
        1 => "ICMP",
        _ => "???",
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Set up file logging so stdout is free for the TUI
    std::fs::create_dir_all("logs").ok();
    let file_appender = tracing_appender::rolling::never("logs", "ddos.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    tracing_subscriber::fmt()
        .with_writer(non_blocking)
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("ensemble_ddos_detection=info".parse()?),
        )
        .with_timer(tracing_subscriber::fmt::time::uptime())
        .with_ansi(false)
        .init();

    let cli = Cli::parse();

    // ── Startup Banner ────────────────────────────────────────────────
    println!();
    println!("  ╔═══════════════════════════════════════════════════╗");
    println!(
        "  ║  Ensemble DDoS Detection Agent v{}             ║",
        env!("CARGO_PKG_VERSION")
    );
    println!("  ╚═══════════════════════════════════════════════════╝");
    println!();

    // ── Load configs ──────────────────────────────────────────────────
    tracing::info!("Loading configuration...");
    let configs = ModelConfigs::load(&cli.models_dir)?;
    let preprocessor = Preprocessor::from_config(&configs.scaler);
    let n_features = configs.scaler.n_features;
    let n_log_cols = configs.scaler.log_transformed_columns.len();

    // Print config details
    tracing::info!(
        "  Features    : {} ({} log-transformed)",
        n_features,
        n_log_cols
    );
    tracing::info!(
        "  LR coefs    : [{:.2}, {:.2}, {:.2}]",
        configs.ensemble.coefficients[0],
        configs.ensemble.coefficients[1],
        configs.ensemble.coefficients[2],
    );
    tracing::info!(
        "  Threshold   : {:.3} | Intercept: {:.3}",
        configs.ensemble.threshold,
        configs.ensemble.intercept,
    );

    // ── Load ONNX models ──────────────────────────────────────────────
    let mut engine = InferenceEngine::load(
        &cli.models_dir,
        configs.normalization,
        configs.ensemble,
        n_features,
    )?;

    // ── Flow table ────────────────────────────────────────────────────
    let flow_table = FlowTable::new(cli.timeout);

    tracing::info!("─────────────────────────────────────────────────");
    tracing::info!("  Interface   : {}", cli.interface,);
    tracing::info!(
        "  Flow timeout: {}s | Sweep: every {}s",
        cli.timeout,
        cli.sweep_interval
    );
    tracing::info!("─────────────────────────────────────────────────");

    // ── Packet capture channel ────────────────────────────────────────
    let (pkt_tx, pkt_rx) = mpsc::channel();

    // ── Start capture thread ──────────────────────────────────────────
    let iface = capture::find_interface(&cli.interface)?;
    let iface_name = iface.name.clone();
    let _capture_handle = std::thread::Builder::new()
        .name("packet-capture".into())
        .spawn(move || {
            if let Err(e) = capture::start_capture(iface, pkt_tx) {
                tracing::error!("Capture thread error: {}", e);
            }
        })?;

    tracing::info!(
        "🎯 Capture started on {}. Waiting for packets...\n",
        iface_name
    );

    // ── Shared ML Logs Queue ───────────────────────────────────────
    let ml_logs = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let ml_logs_worker = std::sync::Arc::clone(&ml_logs);

    // ── Start ML inference loop in background ──────────────────────────
    std::thread::Builder::new()
        .name("ml-inference".into())
        .spawn(move || {
            let start_time = Instant::now();
            let sweep_interval = Duration::from_secs(cli.sweep_interval);
            let mut last_sweep = Instant::now();
            let mut last_rate_check = Instant::now();
            let mut total_packets: u64 = 0;
            let mut last_rate_packets: u64 = 0;
            let mut total_flows_classified: u64 = 0;
            let mut total_attacks: u64 = 0;

            loop {
                // Process incoming packets (with timeout so we can sweep)
                match pkt_rx.recv_timeout(Duration::from_millis(100)) {
                    Ok(pkt) => {
                        flow_table.process_packet(&pkt);
                        total_packets += 1;

                        // Log stats every 1,000 packets
                        if total_packets.is_multiple_of(1_000) {
                            let elapsed = last_rate_check.elapsed().as_secs_f64();
                            let pkt_delta = total_packets - last_rate_packets;
                            let pps = if elapsed > 0.0 {
                                pkt_delta as f64 / elapsed
                            } else {
                                0.0
                            };

                            let stats_msg = format!(
                                "[{}] 📦 {} pkts ({:.0}/s) | {} active flows | {} classified | {} attacks",
                                format_elapsed(start_time),
                                total_packets,
                                pps,
                                flow_table.active_flow_count(),
                                total_flows_classified,
                                total_attacks,
                            );

                            if let Ok(mut logs) = ml_logs_worker.lock() {
                                logs.push(stats_msg.clone());
                                if logs.len() > 1000 {
                                    logs.remove(0);
                                }
                            }

                            tracing::info!("{}", stats_msg);

                            last_rate_check = Instant::now();
                            last_rate_packets = total_packets;
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
                    let mut sweep_attacks = 0u64;
                    let mut sweep_benign = 0u64;

                    for flow_state in &expired_flows {
                        // Extract features
                        let mut flow_features = features::extract_features(flow_state);

                        // Preprocess
                        preprocessor.transform(&mut flow_features);

                        // Inference
                        match engine.predict(&flow_features) {
                            Ok(result) => {
                                total_flows_classified += 1;

                                if result.is_attack {
                                    total_attacks += 1;
                                    sweep_attacks += 1;
                                    let alert_msg = format!(
                                    "[{}] 🚨 ATTACK  {}:{} → {}:{} {} | prob={:.3}",
                                    format_elapsed(start_time),
                                    flow_state.key.src_ip,
                                    flow_state.key.src_port,
                                    flow_state.key.dst_ip,
                                    flow_state.key.dst_port,
                                    proto_name(flow_state.key.protocol),
                                    result.combined_prob,
                                );
                                if let Ok(mut alerts) = ml_logs_worker.lock() {
                                    alerts.push(alert_msg);
                                    if alerts.len() > 1000 {
                                        alerts.remove(0); // keep last 1000 alerts
                                    }
                                }

                                tracing::warn!(
                                        "[{}] 🚨 ATTACK  {}:{} → {}:{} {} | {} pkts | prob={:.3} | IF={:.3} AE={:.3} SVM={:.3}",
                                        format_elapsed(start_time),
                                        flow_state.key.src_ip,
                                        flow_state.key.src_port,
                                        flow_state.key.dst_ip,
                                        flow_state.key.dst_port,
                                        proto_name(flow_state.key.protocol),
                                        flow_state.total_packets(),
                                        result.combined_prob,
                                        result.if_score,
                                        result.ae_score,
                                        result.svm_score,
                                    );
                                } else {
                                    sweep_benign += 1;
                                    tracing::info!(
                                        "[{}] ✅ BENIGN  {}:{} → {}:{} {} | {} pkts | prob={:.3} | IF={:.3} AE={:.3} SVM={:.3}",
                                        format_elapsed(start_time),
                                        flow_state.key.src_ip,
                                        flow_state.key.src_port,
                                        flow_state.key.dst_ip,
                                        flow_state.key.dst_port,
                                        proto_name(flow_state.key.protocol),
                                        flow_state.total_packets(),
                                        result.combined_prob,
                                        result.if_score,
                                        result.ae_score,
                                        result.svm_score,
                                    );
                                }
                            }
                            Err(e) => {
                                tracing::error!(
                                    "Inference error for flow {}:{} → {}:{}: {}",
                                    flow_state.key.src_ip,
                                    flow_state.key.src_port,
                                    flow_state.key.dst_ip,
                                    flow_state.key.dst_port,
                                    e,
                                );
                            }
                        }
                    }

                    // Sweep summary
                    if !expired_flows.is_empty() {
                        let sweep_msg = format!(
                            "[{}] ── Sweep: {} flows expired, {} benign, {} attacks ──",
                            format_elapsed(start_time),
                            expired_flows.len(),
                            sweep_benign,
                            sweep_attacks,
                        );

                        if let Ok(mut logs) = ml_logs_worker.lock() {
                            logs.push(sweep_msg.clone());
                            if logs.len() > 1000 {
                                logs.remove(0);
                            }
                        }

                        tracing::info!("{}", sweep_msg);
                    }

                    last_sweep = Instant::now();
                }
            }
        }).expect("Failed to start ML inference thread");

    // ── Run TUI ───────────────────────────────────────────────────────
    crossterm::terminal::enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    crossterm::execute!(
        stdout,
        crossterm::terminal::EnterAlternateScreen,
        crossterm::event::EnableMouseCapture
    )?;

    let backend = ratatui::backend::CrosstermBackend::new(stdout);
    let mut terminal = ratatui::Terminal::new(backend)?;

    let result = tui::app::run(&mut terminal, ml_logs).await;

    crossterm::terminal::disable_raw_mode()?;
    crossterm::execute!(
        terminal.backend_mut(),
        crossterm::terminal::LeaveAlternateScreen,
        crossterm::event::DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(e) = result {
        eprintln!("Error running TUI: {:?}", e);
    }

    Ok(())
}
