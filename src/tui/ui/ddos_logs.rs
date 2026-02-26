use crate::tui::app::App;
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Cell, Paragraph, Row, Table},
};

pub fn render(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // header
            Constraint::Min(10),   // Logs Table
            Constraint::Length(3), // footer
        ])
        .split(area);

    render_header(f, chunks[0]);
    render_logs(f, app, chunks[1]);
    render_footer(f, app, chunks[2]);
}

fn render_header(f: &mut Frame, area: Rect) {
    let now = chrono::Local::now().format("%H:%M:%S").to_string();
    let header = Paragraph::new(Line::from(vec![
        Span::styled(" NetWatch ", Style::default().fg(Color::Cyan).bold()),
        Span::raw("│ "),
        Span::raw("[1] Dashboard  [2] Connections  [3] Interfaces  [4] Packets  [5] Stats  [6] Topology  [7] Timeline  [8] Insights  "),
        Span::styled("[9] DDoS Logs", Style::default().fg(Color::Yellow).bold()),
        Span::raw("  │ "),
        Span::styled(now, Style::default().fg(Color::DarkGray)),
    ]))
    .block(Block::default().borders(Borders::BOTTOM).border_style(Style::default().fg(Color::DarkGray)));
    f.render_widget(header, area);
}

fn render_logs(f: &mut Frame, app: &App, area: Rect) {
    let alerts = match app.ml_logs.lock() {
        Ok(a) => a.clone(),
        Err(_) => vec![],
    };

    let capacity = area.height.saturating_sub(4) as usize; // account for borders and header

    // Determine total logs available
    let total_logs = alerts.len();
    let max_scroll = total_logs.saturating_sub(capacity);

    // If following, stay at bottom. Otherwise use the manual scroll clamp
    let start_idx = if app.ddos_logs_follow {
        max_scroll
    } else {
        app.ddos_logs_scroll.min(max_scroll)
    };

    let visible_alerts: Vec<String> = alerts.into_iter().skip(start_idx).take(capacity).collect();

    let rows: Vec<Row> = visible_alerts
        .into_iter()
        .map(|alert| {
            let style = if alert.contains("ATTACK") {
                Style::default().fg(Color::Red).bold()
            } else if alert.contains("BENIGN") {
                Style::default().fg(Color::Green)
            } else {
                Style::default().fg(Color::Cyan) // Aggregates
            };
            Row::new(vec![Cell::from(alert)]).style(style)
        })
        .collect();

    let table = Table::new(rows, [Constraint::Percentage(100)]).block(
        Block::default()
            .title(" Live ML DDoS Inference Logs ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray)),
    );

    f.render_widget(table, area);
}

fn render_footer(f: &mut Frame, app: &App, area: Rect) {
    let footer = Paragraph::new(Line::from(vec![
        Span::styled(" q", Style::default().fg(Color::Yellow).bold()),
        Span::raw(":Quit  "),
        Span::styled("↑↓", Style::default().fg(Color::Yellow).bold()),
        Span::raw(":Scroll Logs  "),
        Span::styled("f", Style::default().fg(Color::Yellow).bold()),
        Span::raw(if app.ddos_logs_follow {
            ":Unfollow  "
        } else {
            ":Follow  "
        }),
        Span::styled("1-9", Style::default().fg(Color::Yellow).bold()),
        Span::raw(":Tab"),
    ]))
    .block(
        Block::default()
            .borders(Borders::TOP)
            .border_style(Style::default().fg(Color::DarkGray)),
    );
    f.render_widget(footer, area);
}
