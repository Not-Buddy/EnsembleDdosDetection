use crate::tui::app::App;
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Clear, Paragraph},
};

pub fn render(f: &mut Frame, app: &App, area: Rect) {
    let popup_width = (area.width * 80 / 100).max(60).min(area.width);
    let popup_height = (area.height * 80 / 100).max(20).min(area.height);

    let x = area.x + (area.width.saturating_sub(popup_width)) / 2;
    let y = area.y + (area.height.saturating_sub(popup_height)) / 2;
    let popup = Rect::new(x, y, popup_width, popup_height);

    f.render_widget(Clear, popup);

    let block = Block::default()
        .title(" Help ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));
    let inner = block.inner(popup);
    f.render_widget(block, popup);

    let lines = build_help_lines();

    // Reserve 1 line for the footer hint
    let visible_height = inner.height.saturating_sub(1) as usize;
    let max_scroll = lines.len().saturating_sub(visible_height);
    let offset = app.help_scroll.min(max_scroll);

    let visible: Vec<Line> = lines
        .into_iter()
        .skip(offset)
        .take(visible_height)
        .collect();

    let content = Paragraph::new(visible);
    f.render_widget(content, Rect::new(inner.x, inner.y, inner.width, inner.height.saturating_sub(1)));

    // Footer hint
    let footer = Paragraph::new(Line::from(vec![
        Span::styled("↑↓", Style::default().fg(Color::Yellow).bold()),
        Span::raw(":Scroll  "),
        Span::styled("?", Style::default().fg(Color::Yellow).bold()),
        Span::raw("/"),
        Span::styled("Esc", Style::default().fg(Color::Yellow).bold()),
        Span::raw(":Close"),
    ]))
    .alignment(Alignment::Center);
    let footer_area = Rect::new(inner.x, inner.y + inner.height.saturating_sub(1), inner.width, 1);
    f.render_widget(footer, footer_area);
}

fn section_header(title: &str) -> Line<'static> {
    Line::from(Span::styled(
        title.to_string(),
        Style::default().fg(Color::Cyan).bold(),
    ))
}

fn key_line(key: &str, desc: &str) -> Line<'static> {
    Line::from(vec![
        Span::raw("  "),
        Span::styled(
            format!("{:<16}", key),
            Style::default().fg(Color::Yellow).bold(),
        ),
        Span::styled(desc.to_string(), Style::default().fg(Color::White)),
    ])
}

fn build_help_lines() -> Vec<Line<'static>> {
    vec![
        // GLOBAL KEYS
        section_header("GLOBAL KEYS"),
        key_line("q", "Quit"),
        key_line("Ctrl+C", "Quit"),
        key_line("1-8", "Switch tab (Dash/Conn/Iface/Pkt/Stats/Topo/Time/Insights)"),
        key_line("p", "Pause/resume data collection"),
        key_line("r", "Force refresh all data"),
        key_line("a", "Request AI analysis (from any tab)"),
        key_line("?", "Toggle this help overlay"),
        key_line("g", "Toggle GeoIP location display"),
        Line::raw(""),
        // DASHBOARD
        section_header("DASHBOARD (Tab 1)"),
        key_line("↑↓", "Select interface"),
        Line::raw(""),
        // CONNECTIONS
        section_header("CONNECTIONS (Tab 2)"),
        key_line("↑↓", "Scroll connection list"),
        key_line("s", "Cycle sort column"),
        key_line("Enter", "Jump to Packets tab with filter for selected connection"),
        key_line("W (shift)", "Whois lookup for selected connection's remote IP"),
        Line::raw(""),
        // INTERFACES
        section_header("INTERFACES (Tab 3)"),
        key_line("↑↓", "Select interface"),
        Line::raw(""),
        // PACKETS
        section_header("PACKETS (Tab 4)"),
        key_line("↑↓", "Scroll packet list"),
        key_line("Enter", "Select packet at cursor"),
        key_line("c", "Start/stop capture"),
        key_line("i", "Cycle capture interface (when stopped)"),
        key_line("b", "Set BPF capture filter (when stopped)"),
        key_line("/", "Open display filter bar"),
        key_line("Esc", "Clear display filter"),
        key_line("s", "Open stream view for selected packet"),
        key_line("w", "Export packets to .pcap file"),
        key_line("f", "Toggle auto-follow"),
        key_line("x", "Clear all captured packets"),
        key_line("m", "Toggle bookmark on selected packet"),
        key_line("n", "Jump to next bookmarked packet"),
        key_line("N (shift)", "Jump to previous bookmarked packet"),
        key_line("W (shift)", "Whois lookup for selected packet IPs"),
        Line::raw(""),
        // STREAM VIEW
        section_header("STREAM VIEW (in Packets tab)"),
        key_line("Esc", "Close stream view"),
        key_line("↑↓", "Scroll stream content"),
        key_line("→←", "Filter direction (A→B / B→A)"),
        key_line("a", "Show both directions"),
        key_line("h", "Toggle hex/text mode"),
        Line::raw(""),
        // STATS
        section_header("STATS (Tab 5)"),
        key_line("↑↓", "Scroll protocol list"),
        Line::raw(""),
        // TOPOLOGY
        section_header("TOPOLOGY (Tab 6)"),
        key_line("↑↓", "Scroll topology view"),
        key_line("Enter", "Jump to Connections tab"),
        Line::raw(""),
        // TIMELINE
        section_header("TIMELINE (Tab 7)"),
        key_line("↑↓", "Scroll connection list"),
        key_line("t", "Cycle time window (30s/1m/5m/15m/1h)"),
        key_line("Enter", "Jump to Connections tab"),
        Line::raw(""),
        // INSIGHTS
        section_header("INSIGHTS (Tab 8)"),
        key_line("a", "Trigger on-demand AI analysis"),
        key_line("↑↓", "Scroll insights"),
        Line::raw(""),
        // DISPLAY FILTER SYNTAX
        section_header("DISPLAY FILTER SYNTAX"),
        key_line("tcp, udp, dns, icmp, arp", "Filter by protocol"),
        key_line("192.168.1.1", "Match source or destination IP"),
        key_line("ip.src == X / ip.dst == X", "Match specific direction"),
        key_line("port 443", "Match source or destination port"),
        key_line("stream 7", "Match stream index"),
        key_line("contains \"text\"", "Search in info/payload"),
        key_line("and, or, not / !", "Combine filters"),
        key_line("bare word", "Shorthand for contains"),
        Line::raw(""),
        // TCP HANDSHAKE TIMING
        section_header("TCP HANDSHAKE TIMING"),
        key_line("⏱ in stream header", "Total 3-way handshake time (SYN→ACK)"),
        key_line("SYN→SA", "Client→Server network RTT (SYN to SYN-ACK)"),
        key_line("SA→ACK", "Server→Client network RTT (SYN-ACK to ACK)"),
        key_line("Shown in:", "Stream view header, status bar, packet detail"),
        Line::raw(""),
        // EXPERT INFO INDICATORS
        section_header("EXPERT INFO INDICATORS"),
        key_line("● (red)", "Error: TCP RST, DNS NXDOMAIN/SERVFAIL"),
        key_line("▲ (yellow)", "Warning: Zero window, ICMP unreachable, HTTP 4xx/5xx"),
        key_line("· (cyan)", "Note: TCP FIN, DNS response, TLS Server Hello"),
        key_line("(space)", "Chat: SYN, DNS query, ARP, normal traffic"),
    ]
}
