"""Interactive web dashboard for Predictive Maintenance.

Plotly Dash application with tabs for Machine Health, Failure Risk,
RUL, Anomaly Detection, What-If Simulation, and Business KPIs.
"""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import Input, Output, State, dcc, html

from src.config import get_config
from src.dashboards.components import (
    CARD_STYLE,
    COLORS,
    distribution_bar,
    kpi_card,
    risk_gauge,
    sensor_trend_chart,
)
from src.dashboards.kpi_engine import KPIEngine
from src.dashboards.what_if import WhatIfSimulator

# ── Generate demo data ────────────────────────────────────────────────────
_config = get_config()
_rng = np.random.default_rng(42)
_demo_probs = _rng.beta(2, 5, size=50)
_demo_ruls = _rng.uniform(10, 200, size=50)
_kpi_engine = KPIEngine(_config.business)
_demo_kpis = _kpi_engine.compute_kpis(_demo_probs, _demo_ruls)

PAGE_STYLE = {
    "backgroundColor": COLORS["bg_primary"],
    "minHeight": "100vh",
    "padding": "24px",
    "fontFamily": "'Inter', 'Segoe UI', sans-serif",
    "color": COLORS["text_primary"],
}

HEADER_STYLE = {
    "background": "linear-gradient(135deg, #1e3a5f 0%, #0f1117 100%)",
    "borderRadius": "16px",
    "padding": "32px",
    "marginBottom": "24px",
    "border": f"1px solid {COLORS['border']}",
}


def _build_header() -> html.Div:
    return html.Div([
        html.H1("⚙️ Predictive Maintenance Dashboard", style={"fontSize": "28px", "fontWeight": "700", "margin": "0 0 8px 0"}),
        html.P("Real-time machine health monitoring & failure prediction", style={"color": COLORS["text_secondary"], "margin": 0, "fontSize": "14px"}),
    ], style=HEADER_STYLE)


def _build_kpi_row() -> html.Div:
    return html.Div([
        html.Div(kpi_card("Fleet Health Score", f"{_demo_kpis.avg_health_score}%", "Avg across all machines", "accent_green"), style={"flex": "1", "minWidth": "200px"}),
        html.Div(kpi_card("Machines at Risk", str(_demo_kpis.machines_at_risk), f"of {_demo_kpis.total_machines} total", "accent_red"), style={"flex": "1", "minWidth": "200px"}),
        html.Div(kpi_card("Annual Savings", f"${_demo_kpis.estimated_annual_savings:,.0f}", "Estimated cost reduction", "accent_blue"), style={"flex": "1", "minWidth": "200px"}),
        html.Div(kpi_card("ROI", f"{_demo_kpis.roi_percentage:.0f}%", "Return on investment", "accent_purple"), style={"flex": "1", "minWidth": "200px"}),
    ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "24px"})


def _build_risk_section() -> html.Div:
    dist = _demo_kpis.risk_distribution
    return html.Div([
        html.Div([
            risk_gauge(float(np.mean(_demo_probs)), "Avg Failure Risk"),
        ], style={**CARD_STYLE, "flex": "1", "minWidth": "280px"}),
        html.Div([
            distribution_bar(
                ["Low", "Medium", "High", "Critical"],
                [dist["low"], dist["medium"], dist["high"], dist["critical"]],
                "Risk Distribution",
            ),
        ], style={**CARD_STYLE, "flex": "2", "minWidth": "400px"}),
    ], style={"display": "flex", "gap": "16px", "marginBottom": "24px", "flexWrap": "wrap"})


def _build_sensor_section() -> html.Div:
    cycles = list(range(1, 101))
    rng = np.random.default_rng(7)
    trends = {
        "Temperature": [520 + i * 0.2 + rng.normal(0, 2) for i in cycles],
        "Vibration": [0.02 + i * 0.0003 + rng.normal(0, 0.003) for i in cycles],
        "Pressure": [14.7 - i * 0.01 + rng.normal(0, 0.2) for i in cycles],
    }
    return html.Div([
        sensor_trend_chart(cycles, trends, "Sensor Trends (Engine #1)"),
    ], style={**CARD_STYLE, "marginBottom": "24px"})


def _build_whatif_section() -> html.Div:
    return html.Div([
        html.H3("🔮 What-If Simulation", style={"margin": "0 0 16px 0", "fontSize": "18px"}),
        html.Div([
            html.Div([
                html.Label("Current Cycle", style={"color": COLORS["text_secondary"], "fontSize": "13px"}),
                dcc.Input(id="whatif-current-cycle", type="number", value=150, min=1, max=500,
                          style={"width": "100%", "padding": "8px", "borderRadius": "8px", "border": f"1px solid {COLORS['border']}", "backgroundColor": COLORS["bg_primary"], "color": COLORS["text_primary"]}),
            ], style={"flex": "1"}),
            html.Div([
                html.Label("Maintenance At Cycle", style={"color": COLORS["text_secondary"], "fontSize": "13px"}),
                dcc.Input(id="whatif-maint-cycle", type="number", value=180, min=1, max=500,
                          style={"width": "100%", "padding": "8px", "borderRadius": "8px", "border": f"1px solid {COLORS['border']}", "backgroundColor": COLORS["bg_primary"], "color": COLORS["text_primary"]}),
            ], style={"flex": "1"}),
            html.Div([
                html.Label("Current Failure Prob", style={"color": COLORS["text_secondary"], "fontSize": "13px"}),
                dcc.Slider(id="whatif-prob", min=0, max=1, step=0.05, value=0.45,
                           marks={0: "0%", 0.5: "50%", 1: "100%"},
                           tooltip={"placement": "bottom"}),
            ], style={"flex": "2"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px", "flexWrap": "wrap"}),
        html.Button("Run Simulation", id="whatif-btn",
                     style={"padding": "10px 24px", "borderRadius": "8px", "border": "none", "backgroundColor": COLORS["accent_blue"], "color": "white", "cursor": "pointer", "fontWeight": "600"}),
        html.Div(id="whatif-results", style={"marginTop": "16px"}),
    ], style=CARD_STYLE)


def create_dashboard() -> dash.Dash:
    """Create the Dash application."""
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        title="Predictive Maintenance Dashboard",
    )

    app.layout = html.Div([
        _build_header(),
        _build_kpi_row(),
        _build_risk_section(),
        _build_sensor_section(),
        _build_whatif_section(),
    ], style=PAGE_STYLE)

    @app.callback(
        Output("whatif-results", "children"),
        Input("whatif-btn", "n_clicks"),
        State("whatif-current-cycle", "value"),
        State("whatif-maint-cycle", "value"),
        State("whatif-prob", "value"),
        prevent_initial_call=True,
    )
    def run_whatif(n_clicks, current_cycle, maint_cycle, prob):
        sim = WhatIfSimulator()
        scenarios = sim.simulate_maintenance_timing(
            current_cycle=current_cycle or 150,
            current_failure_prob=prob or 0.45,
            rul_estimate=50,
            maintenance_cycles=[maint_cycle or 180],
        )
        s = scenarios[0]
        color = COLORS["accent_green"] if s.is_recommended else COLORS["accent_red"]
        return html.Div([
            html.P(s.explanation, style={"fontSize": "14px", "lineHeight": "1.6"}),
            html.P(f"💰 Expected savings: ${s.expected_savings:,.0f}", style={"color": color, "fontWeight": "600", "fontSize": "16px"}),
            html.P(f"{'✅ Recommended' if s.is_recommended else '❌ Not recommended'}", style={"color": color}),
        ])

    return app


# Entry point for running the dashboard
if __name__ == "__main__":
    dashboard_app = create_dashboard()
    dashboard_app.run(host="0.0.0.0", port=8050, debug=True)
