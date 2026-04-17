"""Reusable dashboard UI components for Plotly Dash.

Provides styled KPI cards, charts, and data tables with
a dark-themed professional design system.
"""

from __future__ import annotations

import plotly.graph_objects as go
from dash import dcc, html

# ── Color Palette ─────────────────────────────────────────────────────────
COLORS = {
    "bg_primary": "#0f1117",
    "bg_card": "#1a1d29",
    "bg_card_hover": "#242838",
    "accent_blue": "#3b82f6",
    "accent_green": "#10b981",
    "accent_red": "#ef4444",
    "accent_amber": "#f59e0b",
    "accent_purple": "#8b5cf6",
    "text_primary": "#f1f5f9",
    "text_secondary": "#94a3b8",
    "border": "#2d3348",
}

CARD_STYLE = {
    "backgroundColor": COLORS["bg_card"],
    "borderRadius": "12px",
    "padding": "24px",
    "border": f"1px solid {COLORS['border']}",
    "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.3)",
    "transition": "transform 0.2s, box-shadow 0.2s",
}


def kpi_card(title: str, value: str, subtitle: str = "", color: str = "accent_blue") -> html.Div:
    """Create a styled KPI metric card."""
    accent = COLORS.get(color, COLORS["accent_blue"])
    return html.Div(
        [
            html.P(title, style={"color": COLORS["text_secondary"], "fontSize": "13px", "margin": "0 0 8px 0", "textTransform": "uppercase", "letterSpacing": "1px"}),
            html.H2(value, style={"color": accent, "fontSize": "32px", "fontWeight": "700", "margin": "0 0 4px 0"}),
            html.P(subtitle, style={"color": COLORS["text_secondary"], "fontSize": "12px", "margin": "0"}),
        ],
        style=CARD_STYLE,
    )


def risk_gauge(value: float, title: str = "Risk Level") -> dcc.Graph:
    """Create a risk gauge chart."""
    color = COLORS["accent_green"] if value < 0.3 else COLORS["accent_amber"] if value < 0.6 else COLORS["accent_red"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={"text": title, "font": {"color": COLORS["text_primary"], "size": 14}},
        number={"suffix": "%", "font": {"color": COLORS["text_primary"]}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": COLORS["text_secondary"]},
            "bar": {"color": color},
            "bgcolor": COLORS["bg_card"],
            "steps": [
                {"range": [0, 30], "color": "rgba(16, 185, 129, 0.2)"},
                {"range": [30, 60], "color": "rgba(245, 158, 11, 0.2)"},
                {"range": [60, 100], "color": "rgba(239, 68, 68, 0.2)"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=200, margin=dict(t=40, b=0, l=20, r=20),
    )
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def sensor_trend_chart(
    x_data: list, y_data: dict[str, list], title: str = "Sensor Trends",
) -> dcc.Graph:
    """Create a multi-line sensor trend chart."""
    sensor_colors = [
        COLORS["accent_blue"], COLORS["accent_green"], COLORS["accent_amber"],
        COLORS["accent_red"], COLORS["accent_purple"], "#06b6d4",
    ]
    fig = go.Figure()
    for i, (name, values) in enumerate(y_data.items()):
        fig.add_trace(go.Scatter(
            x=x_data, y=values, name=name, mode="lines",
            line=dict(color=sensor_colors[i % len(sensor_colors)], width=2),
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(color=COLORS["text_primary"], size=16)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor=COLORS["border"], color=COLORS["text_secondary"]),
        yaxis=dict(gridcolor=COLORS["border"], color=COLORS["text_secondary"]),
        legend=dict(font=dict(color=COLORS["text_secondary"])),
        height=350, margin=dict(t=50, b=40, l=50, r=20),
    )
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def distribution_bar(labels: list[str], values: list[int], title: str = "") -> dcc.Graph:
    """Create a horizontal bar chart for distributions."""
    bar_colors = [COLORS["accent_green"], COLORS["accent_amber"], COLORS["accent_red"], COLORS["accent_purple"]]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=bar_colors[: len(labels)],
        text=values, textposition="auto",
        textfont=dict(color=COLORS["text_primary"]),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color=COLORS["text_primary"], size=16)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor=COLORS["border"], color=COLORS["text_secondary"]),
        yaxis=dict(color=COLORS["text_secondary"]),
        height=250, margin=dict(t=50, b=30, l=80, r=20),
    )
    return dcc.Graph(figure=fig, config={"displayModeBar": False})
