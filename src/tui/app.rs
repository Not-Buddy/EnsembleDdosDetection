use crate::tui::collectors::config::ConfigCollector;
use crate::tui::collectors::connections::{ConnectionCollector, ConnectionTimeline};
use crate::tui::collectors::geo::GeoCache;
use crate::tui::collectors::health::HealthProber;
use crate::tui::collectors::traffic::TrafficCollector;
use crate::tui::event::{AppEvent, EventHandler};
use crate::tui::platform::{self, InterfaceInfo};
use crate::tui::ui;
use anyhow::Result;
use crossterm::event::{KeyCode, KeyModifiers};
use ratatui::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimelineWindow {
    Min1,
    Min5,
    Min15,
    Min30,
    Hour1,
}

impl TimelineWindow {
    pub fn seconds(&self) -> u64 {
        match self {
            Self::Min1 => 60,
            Self::Min5 => 300,
            Self::Min15 => 900,
            Self::Min30 => 1800,
            Self::Hour1 => 3600,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Min1 => "1m",
            Self::Min5 => "5m",
            Self::Min15 => "15m",
            Self::Min30 => "30m",
            Self::Hour1 => "1h",
        }
    }

    fn next(self) -> Self {
        match self {
            Self::Min1 => Self::Min5,
            Self::Min5 => Self::Min15,
            Self::Min15 => Self::Min30,
            Self::Min30 => Self::Hour1,
            Self::Hour1 => Self::Min1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Dashboard,
    Connections,
    Interfaces,
    Topology,
    Timeline,
    DdosLogs,
}

pub struct App {
    pub traffic: TrafficCollector,
    pub interface_info: Vec<InterfaceInfo>,
    pub connection_collector: ConnectionCollector,
    pub config_collector: ConfigCollector,
    pub health_prober: HealthProber,
    pub selected_interface: Option<usize>,
    pub paused: bool,
    pub current_tab: Tab,
    pub connection_scroll: usize,
    pub sort_column: usize,
    pub show_help: bool,
    pub help_scroll: usize,
    pub geo_cache: GeoCache,
    pub show_geo: bool,
    pub topology_scroll: usize,
    pub connection_timeline: ConnectionTimeline,
    pub timeline_scroll: usize,
    pub timeline_window: TimelineWindow,
    info_tick: u32,
    conn_tick: u32,
    health_tick: u32,
    pub ml_logs: Arc<Mutex<Vec<String>>>,
    pub ddos_logs_scroll: usize,
    pub ddos_logs_follow: bool,
}

impl App {
    pub fn new(ml_logs: Arc<Mutex<Vec<String>>>) -> Self {
        let interface_info = platform::collect_interface_info().unwrap_or_default();
        let mut config_collector = ConfigCollector::new();
        config_collector.update();

        Self {
            traffic: TrafficCollector::new(),
            interface_info,
            connection_collector: ConnectionCollector::new(),
            config_collector,
            health_prober: HealthProber::new(),
            selected_interface: None,
            paused: false,
            current_tab: Tab::Dashboard,
            connection_scroll: 0,
            sort_column: 0,
            show_help: false,
            help_scroll: 0,
            geo_cache: GeoCache::new(),
            show_geo: true,
            topology_scroll: 0,
            connection_timeline: ConnectionTimeline::new(),
            timeline_scroll: 0,
            timeline_window: TimelineWindow::Min5,
            info_tick: 0,
            conn_tick: 0,
            health_tick: 0,
            ml_logs,
            ddos_logs_scroll: 0,
            ddos_logs_follow: true,
        }
    }

    fn tick(&mut self) {
        if self.paused {
            return;
        }
        self.traffic.update();

        // Refresh interface info every ~10 ticks (10s at 1s tick rate)
        self.info_tick += 1;
        if self.info_tick >= 10 {
            self.info_tick = 0;
            if let Ok(info) = platform::collect_interface_info() {
                self.interface_info = info;
            }
            self.config_collector.update();
        }

        // Refresh connections every ~2 ticks (2s)
        self.conn_tick += 1;
        if self.conn_tick >= 2 {
            self.conn_tick = 0;
            self.connection_collector.update();
            let conns = self.connection_collector.connections.lock().unwrap();
            self.connection_timeline.update(&conns);
        }

        // Refresh health every ~5 ticks (5s)
        self.health_tick += 1;
        if self.health_tick >= 5 {
            self.health_tick = 0;
            let gateway = self.config_collector.config.gateway.clone();
            let dns = self.config_collector.config.dns_servers.first().cloned();
            self.health_prober.probe(gateway.as_deref(), dns.as_deref());
        }
    }
}

fn parse_addr_parts(addr: &str) -> (Option<String>, Option<String>) {
    if addr == "*:*" || addr.is_empty() {
        return (None, None);
    }
    if let Some(bracket_end) = addr.rfind("]:") {
        let ip = addr[1..bracket_end].to_string();
        let port = addr[bracket_end + 2..].to_string();
        (Some(ip), Some(port))
    } else if let Some(colon) = addr.rfind(':') {
        let ip = &addr[..colon];
        let port = &addr[colon + 1..];
        let ip = if ip == "*" {
            None
        } else {
            Some(ip.to_string())
        };
        let port = if port == "*" {
            None
        } else {
            Some(port.to_string())
        };
        (ip, port)
    } else {
        (Some(addr.to_string()), None)
    }
}

pub async fn run<B: Backend>(
    terminal: &mut Terminal<B>,
    ml_logs: Arc<Mutex<Vec<String>>>,
) -> Result<()> {
    let mut app = App::new(ml_logs);
    let mut events = EventHandler::new(1000);

    // Initial data collection
    app.traffic.update();
    app.connection_collector.update();
    {
        let conns = app.connection_collector.connections.lock().unwrap();
        app.connection_timeline.update(&conns);
    }
    let gateway = app.config_collector.config.gateway.clone();
    let dns = app.config_collector.config.dns_servers.first().cloned();
    app.health_prober.probe(gateway.as_deref(), dns.as_deref());

    loop {
        terminal.draw(|f| {
            let area = f.size();
            match app.current_tab {
                Tab::Dashboard => ui::dashboard::render(f, &app, area),
                Tab::Connections => ui::connections::render(f, &app, area),
                Tab::Interfaces => ui::interfaces::render(f, &app, area),
                Tab::Topology => ui::topology::render(f, &app, area),
                Tab::Timeline => ui::timeline::render(f, &app, area),
                Tab::DdosLogs => ui::ddos_logs::render(f, &app, area),
            }
            if app.show_help {
                ui::help::render(f, &app, area);
            }
        })?;

        match events.next().await? {
            AppEvent::Key(key) => {
                // Help overlay — intercept keys first
                if app.show_help {
                    match key.code {
                        KeyCode::Char('?') | KeyCode::Esc => {
                            app.show_help = false;
                            app.help_scroll = 0;
                        }
                        KeyCode::Up => {
                            app.help_scroll = app.help_scroll.saturating_sub(1);
                        }
                        KeyCode::Down => {
                            app.help_scroll += 1;
                        }
                        KeyCode::Char('q') => {
                            return Ok(());
                        }
                        _ => {}
                    }
                    continue;
                }
                match key.code {
                    KeyCode::Char('q') => {
                        return Ok(());
                    }
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        return Ok(());
                    }
                    KeyCode::Char('?') => {
                        app.show_help = !app.show_help;
                        app.help_scroll = 0;
                    }
                    KeyCode::Char('g') => app.show_geo = !app.show_geo,
                    KeyCode::Char('p') => app.paused = !app.paused,
                    KeyCode::Char('r') => {
                        app.traffic.update();
                        if let Ok(info) = platform::collect_interface_info() {
                            app.interface_info = info;
                        }
                        app.connection_collector.update();
                        app.config_collector.update();
                        let gateway = app.config_collector.config.gateway.clone();
                        let dns = app.config_collector.config.dns_servers.first().cloned();
                        app.health_prober.probe(gateway.as_deref(), dns.as_deref());
                    }
                    KeyCode::Char('1') => app.current_tab = Tab::Dashboard,
                    KeyCode::Char('2') => app.current_tab = Tab::Connections,
                    KeyCode::Char('3') => app.current_tab = Tab::Interfaces,
                    KeyCode::Char('4') => app.current_tab = Tab::Topology,
                    KeyCode::Char('5') => app.current_tab = Tab::Timeline,
                    KeyCode::Char('6') => app.current_tab = Tab::DdosLogs,

                    KeyCode::Char('s') => {
                        if app.current_tab == Tab::Connections {
                            app.sort_column = (app.sort_column + 1) % 6;
                        }
                    }
                    KeyCode::Char('t') if app.current_tab == Tab::Timeline => {
                        app.timeline_window = app.timeline_window.next();
                    }
                    KeyCode::Enter if app.current_tab == Tab::Timeline => {
                        let window_secs = app.timeline_window.seconds();
                        let now = std::time::Instant::now();
                        let window_start = now - std::time::Duration::from_secs(window_secs);
                        let mut sorted: Vec<
                            &crate::tui::collectors::connections::TrackedConnection,
                        > = app
                            .connection_timeline
                            .tracked
                            .iter()
                            .filter(|t| t.last_seen >= window_start)
                            .collect();
                        sorted.sort_by(|a, b| {
                            b.is_active
                                .cmp(&a.is_active)
                                .then_with(|| a.first_seen.cmp(&b.first_seen))
                        });
                        if let Some(tracked) = sorted.get(app.timeline_scroll) {
                            let (remote_ip, _) = parse_addr_parts(&tracked.key.remote_addr);
                            if remote_ip.is_some() {
                                app.current_tab = Tab::Connections;
                            }
                        }
                    }

                    KeyCode::Enter if app.current_tab == Tab::Topology => {
                        let mut counts: HashMap<String, usize> = HashMap::new();
                        let conns = app.connection_collector.connections.lock().unwrap();
                        for conn in conns.iter() {
                            let (remote_ip, _) = parse_addr_parts(&conn.remote_addr);
                            if let Some(ip) = remote_ip {
                                *counts.entry(ip).or_insert(0) += 1;
                            }
                        }
                        drop(conns);
                        let mut remote_ips: Vec<(String, usize)> = counts.into_iter().collect();
                        remote_ips.sort_by(|a, b| b.1.cmp(&a.1));
                        if let Some((_ip, _)) = remote_ips.get(app.topology_scroll) {
                            app.current_tab = Tab::Connections;
                        }
                    }

                    KeyCode::Up => match app.current_tab {
                        Tab::Connections => {
                            app.connection_scroll = app.connection_scroll.saturating_sub(1);
                        }

                        Tab::Topology => {
                            app.topology_scroll = app.topology_scroll.saturating_sub(1);
                        }
                        Tab::Timeline => {
                            app.timeline_scroll = app.timeline_scroll.saturating_sub(1);
                        }

                        Tab::DdosLogs => {
                            if app.ddos_logs_follow {
                                app.ddos_logs_follow = false;
                                let term_height =
                                    crossterm::terminal::size().map(|s| s.1).unwrap_or(24);
                                let capacity = term_height.saturating_sub(10) as usize;
                                let max = {
                                    let alerts = app.ml_logs.lock().unwrap();
                                    alerts.len().saturating_sub(capacity)
                                };
                                app.ddos_logs_scroll = max;
                            }
                            app.ddos_logs_scroll = app.ddos_logs_scroll.saturating_sub(1);
                        }
                        _ => {
                            app.selected_interface = match app.selected_interface {
                                Some(0) | None => None,
                                Some(i) => Some(i - 1),
                            };
                        }
                    },
                    KeyCode::Down => match app.current_tab {
                        Tab::Connections => {
                            let max = app
                                .connection_collector
                                .connections
                                .lock()
                                .unwrap()
                                .len()
                                .saturating_sub(1);
                            if app.connection_scroll < max {
                                app.connection_scroll += 1;
                            }
                        }

                        Tab::Topology => {
                            app.topology_scroll += 1;
                        }
                        Tab::Timeline => {
                            app.timeline_scroll += 1;
                        }

                        Tab::DdosLogs => {
                            let term_height =
                                crossterm::terminal::size().map(|s| s.1).unwrap_or(24);
                            let capacity = term_height.saturating_sub(10) as usize;
                            let max = {
                                let alerts = app.ml_logs.lock().unwrap();
                                alerts.len().saturating_sub(capacity)
                            };
                            if app.ddos_logs_scroll < max {
                                app.ddos_logs_scroll += 1;
                            } else {
                                app.ddos_logs_follow = true;
                            }
                        }
                        _ => {
                            let max = app.traffic.interfaces.len().saturating_sub(1);
                            app.selected_interface = match app.selected_interface {
                                None => Some(0),
                                Some(i) if i < max => Some(i + 1),
                                other => other,
                            };
                        }
                    },

                    _ => {}
                }
            }
            AppEvent::Tick => {
                app.tick();
            }
        }
    }
}
