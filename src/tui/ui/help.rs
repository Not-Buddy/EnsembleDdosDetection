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
    f.render_widget(
        content,
        Rect::new(
            inner.x,
            inner.y,
            inner.width,
            inner.height.saturating_sub(1),
        ),
    );

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
    let footer_area = Rect::new(
        inner.x,
        inner.y + inner.height.saturating_sub(1),
        inner.width,
        1,
    );
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
        key_line("1-6", "Switch tab (Dash/Conn/Iface/Topo/Time/DDoS)"),
        key_line("p", "Pause/resume data collection"),
        key_line("r", "Force refresh all data"),
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
        key_line(
            "W (shift)",
            "Whois lookup for selected connection's remote IP",
        ),
        Line::raw(""),
        // INTERFACES
        section_header("INTERFACES (Tab 3)"),
        key_line("↑↓", "Select interface"),
        Line::raw(""),
        // TOPOLOGY
        section_header("TOPOLOGY (Tab 4)"),
        key_line("↑↓", "Scroll topology view"),
        key_line("Enter", "Jump to Connections tab"),
        Line::raw(""),
        // TIMELINE
        section_header("TIMELINE (Tab 5)"),
        key_line("↑↓", "Scroll connection list"),
        key_line("t", "Cycle time window (30s/1m/5m/15m/1h)"),
        key_line("Enter", "Jump to Connections tab"),
        Line::raw(""),
        // EXPERT INFO INDICATORS
        section_header("EXPERT INFO INDICATORS"),
        key_line("● (red)", "Error: TCP RST, DNS NXDOMAIN/SERVFAIL"),
        key_line(
            "▲ (yellow)",
            "Warning: Zero window, ICMP unreachable, HTTP 4xx/5xx",
        ),
        key_line("· (cyan)", "Note: TCP FIN, DNS response, TLS Server Hello"),
        key_line("(space)", "Chat: SYN, DNS query, ARP, normal traffic"),
    ]
}
