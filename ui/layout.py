"""
ui/layout.py
============
All Dash layout components: navbar, tabs, and their child cards.

Callbacks are NOT defined here – see ui/callbacks/.
This file only builds static (or mostly-static) component trees.
"""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table

from config import (
    MODE_EXISTING, MODE_FIXED, MODE_PUMP, MODE_BAL,
    ROOM_TYPE_OPTIONS, ROOM_TYPE_HELP_MD, CHART_HEIGHT_PX,
    PUMP_LIBRARY,
)
from services.room_service import default_room_table
from domain.valve import Valve

# ---------------------------------------------------------------------------
# Navbar
# ---------------------------------------------------------------------------
navbar = dbc.Navbar(
    dbc.Container([
        html.A(
            dbc.Row([
                dbc.Col(html.Img(src="/assets/Recover_logo.png", height="50px"), width="auto"),
                dbc.Col(dbc.NavbarBrand("Smart Heating Design Tool", className="ms-2"), width="auto"),
            ], align="center", className="g-0"),
            href="/", style={"textDecoration": "none"},
        ),
        dbc.Nav([
            dbc.NavItem(dbc.NavLink(html.I(className="bi bi-house-door me-1"), " Home", href="/home")),
            dbc.NavItem(dbc.NavLink(html.I(className="bi bi-info-circle me-1"), " Help", href="/help")),
        ], navbar=True, className="ms-auto"),
    ], fluid=True),
    color="dark", dark=True, sticky="top",
)


# ---------------------------------------------------------------------------
# Tab 0 — Start / mode selection
# ---------------------------------------------------------------------------
def build_start_tab() -> dbc.Tab:
    return dbc.Tab(
        label="0️⃣ Start", tab_id="tab-0",
        children=[html.Div([
            html.H2("Welkom in de Smart Heating Design Tool", className="mb-4"),
            html.Hr(className="my-4"),
            html.H4("Stap 1 — Ontwerpmodus", className="mt-4"),
            dcc.Dropdown(
                id="design-mode",
                options=[
                    {"label": "Bestaand systeem — Bereken aanvoertemperatuur",  "value": MODE_EXISTING},
                    {"label": "Dimensionering op aanvoertemperatuur → extra vermogen", "value": MODE_FIXED},
                    {"label": "Pump-based design — Pompcurve bepaalt debiet",  "value": MODE_PUMP},
                    {"label": "Balancering modus — TRV-inregeling",            "value": MODE_BAL},
                ],
                value=MODE_EXISTING, clearable=False, className="mb-3",
            ),
            html.Div(id="design-mode-help", className="text-muted small mb-4"),
            dbc.Alert(
                "De gekozen ontwerpmodus bepaalt welke parameters je in Tab 2 ziet "
                "en welke berekeningen Tab 3 uitvoert.",
                color="secondary",
            ),
            html.H4("Stap 2 — Heat loss modus", className="mt-3"),
            dbc.RadioItems(
                id="heat-load-mode",
                options=[
                    {"label": "Warmteverliezen zijn BEKEND (manuele invoer)",      "value": "known"},
                    {"label": "Warmteverliezen zijn NIET gekend (berekenen)", "value": "unknown"},
                ],
                value=None, className="mb-4",
            ),
            dbc.Alert(
                "Gebruik deze stap om te bepalen hoe de tool de warmteverliezen per kamer moet kennen.",
                color="info",
            ),
        ], className="p-4")]
    )


# ---------------------------------------------------------------------------
# Tab 1 — Heat Loss
# ---------------------------------------------------------------------------
def _insulation_card() -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader("📐 Insulation level"),
        dbc.CardBody([
            dbc.Label("Wall"),
            dcc.Dropdown(id="wall_insulation_state", clearable=False, className="mb-2",
                options=[
                    {"label": "Not Insulated (0 cm)",   "value": "not insulated"},
                    {"label": "Insulated (5 cm)",        "value": "bit insulated"},
                    {"label": "Insulated well (10 cm)",  "value": "insulated well"},
                ], value="bit insulated"),
            dbc.Label("Roof"),
            dcc.Dropdown(id="roof_insulation_state", clearable=False,
                options=[
                    {"label": "Not insulated (0 cm)",    "value": "not insulated"},
                    {"label": "Insulated (5 cm)",         "value": "bit insulated"},
                    {"label": "Insulated well (>10 cm)", "value": "insulated well"},
                ], value="bit insulated"),
            dbc.Label("Ground"),
            dcc.Dropdown(id="ground_insulation_state", clearable=False, className="mb-2",
                options=[
                    {"label": "Not insulated (0 cm)",    "value": "not insulated"},
                    {"label": "Insulated (5 cm)",         "value": "bit insulated"},
                    {"label": "Insulated well (>10 cm)", "value": "insulated well"},
                ], value="bit insulated"),
            dbc.Label("Glazing Type"),
            dcc.Dropdown(id="glazing_type", clearable=False, className="mb-2",
                options=[
                    {"label": "Single", "value": "single"},
                    {"label": "Double", "value": "double"},
                    {"label": "Triple", "value": "triple"},
                ], value="double"),
            dbc.FormText("Selecting a preset sets a typical U-value, but you can override it manually."),
        ])
    ], className="mb-4", id="building-insulation-card")


def _envelope_card() -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader("🏠 Building Envelope"),
        dbc.CardBody([
            dbc.Label("Wall U-value (W/m²K)"),
            dbc.Input(id="uw", type="number", value=1.0, step=0.05),
            dbc.FormText("External wall U-value"), html.Br(),
            dbc.Label("Roof U-value (W/m²K)"),
            dbc.Input(id="u_roof", type="number", value=0.2, step=0.05),
            dbc.FormText("Roof U-value"), html.Br(),
            dbc.Label("Ground U-value (W/m²K)"),
            dbc.Input(id="u_ground", type="number", value=0.3, step=0.05),
            dbc.FormText("Slab-on-grade U-value"), html.Br(),
            dbc.Label("Glazing U-value (W/m²K)"),
            dbc.Input(id="u_glass", type="number", value=2.8, step=0.05),
            dbc.FormText("Window glazing U-value"),
        ])
    ], className="mb-4", id="building-envelope-card")


def _outdoor_card() -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader("🌡️ Outdoor Conditions"),
        dbc.CardBody([
            dbc.Label("Outdoor Temperature (°C)"),
            dbc.Input(id="tout", type="number", min=-12.0, max=40.0, value=-7.0, step=0.5),
            dbc.FormText("Design outdoor winter temperature."),
        ])
    ], className="mb-4", id="outdoor-conditions-card")


def _additional_settings_card() -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader("⚙️ Additional Settings"),
        dbc.CardBody([dbc.Accordion([
            dbc.AccordionItem([
                dbc.Label("Ventilation Calculation Method"),
                dcc.Dropdown(id="ventilation_calculation_method", clearable=False,
                    options=[{"label": "simple", "value": "simple"},
                             {"label": "NBN-D-50-001", "value": "NBN-D-50-001"}],
                    value="simple", className="dash-dropdown"),
                html.Br(),
                dbc.Label("Ventilation System"),
                dcc.Dropdown(id="v_system", clearable=False,
                    options=[{"label": k, "value": k} for k in ["C", "D"]],
                    value="C", className="dash-dropdown"),
                html.Br(),
                dbc.Label("Air Tightness (v50)"),
                dbc.Input(id="v50", type="number", min=0, max=12, value=6.0, step=0.5),
                dbc.FormText("Air changes per hour at 50 Pa (1/h)"),
            ], title="💨 Ventilation Settings", item_id="ventilation"),
            dbc.AccordionItem([
                dbc.Label("Neighbour Temperature (°C)"),
                dbc.Input(id="neighbour_t", type="number", value=18.0, step=0.5), html.Br(),
                dbc.Label("Neighbour Loss Coefficient (Un)"),
                dbc.Input(id="un", type="number", value=1.0, step=0.1), html.Br(),
                dbc.Label("Infiltration Rate (LIR)"),
                dbc.Input(id="lir", type="number", value=0.2, step=0.05), html.Br(),
                dbc.Label("Wall Height (m)"),
                dbc.Input(id="wall_height", type="number", value=2.7, step=0.1), html.Br(),
            ], title="🔍 Advanced Settings", item_id="advanced"),
        ], start_collapsed=True, always_open=False, flush=True)])
    ], className="mb-4", id="additional-settings-card")


def build_heat_tab() -> dbc.Tab:
    return dbc.Tab(
        label="1️⃣ Heat Loss", tab_id="tab-1",
        children=[dbc.Card([dbc.CardBody([
            dbc.Alert(id="heat-load-mode-banner", color="secondary", className="mb-3"),
            html.Hr(),
            dbc.Row([
                dbc.Col([_insulation_card()], md=3),
                dbc.Col([_envelope_card()], md=3),
                dbc.Col([
                    _outdoor_card(),
                    dbc.Card([
                        dbc.CardHeader("Number of Rooms"),
                        dbc.CardBody([
                            dbc.Label("Number of Rooms"),
                            dbc.Input(id="num_rooms", type="number", min=1, value=3, step=1,
                                      style={"maxWidth": "160px"}),
                            dbc.FormText("Add rooms first, then fill in details."),
                        ])
                    ], className="mb-4"),
                ], md=3),
                dbc.Col([_additional_settings_card()], md=3),
            ]),
            # Room config table
            dbc.Card([
                dbc.CardHeader("🧾 Room Configuration"),
                dbc.CardBody([
                    dash_table.DataTable(
                        id="room-table", editable=True, row_deletable=False,
                        columns=[
                            {"name": "Room #",          "id": "Room #",          "type": "numeric", "editable": False},
                            {"name": "Indoor Temp (°C)", "id": "Indoor Temp (°C)", "type": "numeric", "editable": True, "presentation": "input"},
                            {"name": "Floor Area (m²)", "id": "Floor Area (m²)", "type": "numeric"},
                            {"name": "Walls external",  "id": "Walls external",  "type": "numeric", "presentation": "dropdown"},
                            {"name": "Room Type",       "id": "Room Type",       "type": "text",    "presentation": "dropdown"},
                            {"name": "On Ground",       "id": "On Ground",       "type": "any",     "presentation": "dropdown"},
                            {"name": "Under Roof",      "id": "Under Roof",      "type": "any",     "presentation": "dropdown"},
                        ],
                        dropdown={
                            "Room Type":     {"options": [{"label": v, "value": v} for v in ROOM_TYPE_OPTIONS]},
                            "On Ground":     {"options": [{"label": "No", "value": False}, {"label": "Yes", "value": True}]},
                            "Under Roof":    {"options": [{"label": "No", "value": False}, {"label": "Yes", "value": True}]},
                            "Walls external":{"options": [{"label": str(v), "value": str(v)} for v in [1, 2, 3, 4]]},
                        },
                        tooltip_header={
                            "Room Type":      ROOM_TYPE_HELP_MD,
                            "On Ground":      "Space on ground slab / exposed to ground.",
                            "Under Roof":     "Space directly under roof.",
                            "Walls external": "Number of external walls (1-4).",
                            "Indoor Temp (°C)": "Set between 10 °C and 24 °C.",
                        },
                        data=default_room_table(3), page_size=25,
                        style_header={"backgroundColor": "#f8f9fa", "fontWeight": "bold", "textAlign": "center"},
                        style_cell={"padding": "8px", "textAlign": "left", "border": "1px solid #dee2e6"},
                        style_data_conditional=[
                            {"if": {"column_id": c}, "backgroundColor": "#fffef0"}
                            for c in ["Indoor Temp (°C)", "Floor Area (m²)", "Walls external", "Room Type", "On Ground", "Under Roof"]
                        ] + [{"if": {"row_index": "odd"}, "backgroundColor": "rgb(248,248,248)"}],
                        tooltip_delay=200, tooltip_duration=None,
                    ),
                    html.Div([html.Small(
                        "Tip: Double-click cells to edit. 'Walls external': 1–4. 'Indoor Temp': 10–24 °C.",
                        className="text-muted")], className="mt-2"),
                ])
            ], className="mb-4", id="room-config-card"),

            # Manual heat loss table
            dbc.Card([
                dbc.CardHeader("✍️ Manual Room Heat Loss (visible when 'known' is selected)"),
                dbc.CardBody([
                    dash_table.DataTable(
                        id="manual-loss-table", editable=True, row_deletable=False,
                        columns=[
                            {"name": "Room #",               "id": "Room #",               "type": "numeric", "editable": False},
                            {"name": "Manual Heat Loss (W)", "id": "Manual Heat Loss (W)", "type": "numeric"},
                        ],
                        data=[{"Room #": i, "Manual Heat Loss (W)": 0.0} for i in range(1, 4)],
                        page_size=10,
                        style_header={"backgroundColor": "#f8f9fa", "fontWeight": "bold", "textAlign": "center"},
                        style_cell={"padding": "8px", "border": "1px solid #dee2e6"},
                        style_data_conditional=[
                            {"if": {"row_index": "odd"}, "backgroundColor": "rgb(248,248,248)"},
                            {"if": {"column_id": "Manual Heat Loss (W)"}, "backgroundColor": "#fffef0"},
                        ],
                        tooltip_header={"Manual Heat Loss (W)": "Enter the design heat loss for each room (W)."},
                        tooltip_delay=200, tooltip_duration=None,
                    ),
                    html.Small("When visible, Tab 2 & 3 use these values instead of calculated ones.",
                               className="text-muted"),
                ])
            ], id="manual-loss-card", style={"display": "none"}, className="mb-4"),

            # Results table
            dbc.Card([
                dbc.CardHeader("📊 Room Heat Loss Results"),
                dbc.CardBody([html.Div(id="room-results-table")])
            ], className="mb-4",
               style={"backgroundColor": "#f0f4ff", "border": "1px solid #cce", "boxShadow": "0 0 6px rgba(0,0,0,0.1)"}),
        ])])]
    )


# ---------------------------------------------------------------------------
# Tab 2 — Radiators & Collectors
# ---------------------------------------------------------------------------
def _system_config_card() -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader("🛠️ System Configuration"),
        dbc.CardBody([
            dbc.Label("Number of radiators"),
            dbc.Input(id="num_radiators", type="number", min=1, value=3, step=1), html.Br(),
            dbc.Label("Number of collectors"),
            dbc.Input(id="num_collectors", type="number", min=1, value=1, step=1), html.Br(),
            dbc.Label("Delta T (°C)"),
            dbc.Input(id="delta_T", type="number", min=3, max=20, step=1, value=10), html.Br(),
            html.Div([
                html.Span("Optional Inputs", className="form-label fw-bold mt-3 me-2"),
                html.Span(html.I(className="bi bi-info-circle text-warning", id="experimental-tooltip"),
                          className="align-middle"),
                dbc.Tooltip("⚠️ Experimental: may cause unexpected behaviour. Use with caution.",
                            target="experimental-tooltip", placement="right"),
            ]),
            dbc.Label("Supply temperature (°C)"),
            dbc.Input(id="supply_temp_input", type="number", placeholder="(optional)"), html.Br(),
            dbc.Checklist(id="fix_diameter",
                options=[{"label": " Fix diameter per radiator", "value": "yes"}], value=[]),
            html.Div(id="fixed_diameter_container"),
        ])
    ], className="mb-4")


def _pump_settings_card() -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader("♻️ Pump settings"),
        dbc.CardBody([
            dcc.Dropdown(id="pump_model", clearable=False, className="mb-2",
                options=[{"label": name, "value": name} for name in PUMP_LIBRARY],
                value="Grundfos UPM3 15-70"),
            dcc.Dropdown(id="pump_speed", clearable=False,
                options=[{"label": "Speed 1", "value": "speed_1"},
                         {"label": "Speed 2", "value": "speed_2"},
                         {"label": "Speed 3", "value": "speed_3"}],
                value="speed_2"),
            html.Small("Kies een interne pompcurve en snelheid.", className="text-muted"),
        ])
    ], className="mb-4", id="pump-settings-card", style={"display": "none"})


def _valve_settings_card() -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader("🧩 Valve Settings"),
        dbc.CardBody([
            dbc.Label("Valve Type"),
            dcc.Dropdown(id="valve-type-dropdown", clearable=False, className="mb-3",
                options=[{"label": name, "value": name} for name in Valve.get_valve_names()],
                value="Custom"),
            html.Div(id="valve-specs", className="small text-muted mb-3"),
            html.Div(id="valve-custom-settings", children=[
                dbc.Row([
                    dbc.Col([dbc.Label("Positions"), dbc.Input(id="positions", type="number", min=2, value=8, step=1, className="mb-3")]),
                    dbc.Col([dbc.Label("Kv max"),    dbc.Input(id="kv_max",    type="number", min=0.1, value=0.7, step=0.1)]),
                ])
            ]),
        ])
    ], className="mb-4")


def _radiator_table_card() -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader("🌡️ Radiator Inputs"),
        dbc.CardBody([
            dash_table.DataTable(
                id="radiator-table", editable=True, row_deletable=False,
                columns=[
                    {"name": "Radiator",                "id": "Radiator nr",             "type": "numeric", "editable": False},
                    {"name": "Room",                    "id": "Room",                    "presentation": "dropdown"},
                    {"name": "Collector",               "id": "Collector",               "presentation": "dropdown"},
                    {"name": "Radiator power 75/65/20", "id": "Radiator power 75/65/20", "type": "numeric"},
                    {"name": "Length circuit",          "id": "Length circuit",          "type": "numeric"},
                    {"name": "Electric power",          "id": "Electric power",          "type": "numeric"},
                ],
                data=[], dropdown={}, page_size=10,
                style_cell={"padding": "8px", "border": "1px solid #dee2e6"},
                style_header={"backgroundColor": "#f8f9fa", "fontWeight": "bold", "textAlign": "center"},
                style_data_conditional=[
                    {"if": {"column_id": c}, "backgroundColor": "#fffef0"}
                    for c in ["Room", "Collector", "Radiator power 75/65/20", "Length circuit", "Electric power"]
                ] + [{"if": {"row_index": "odd"}, "backgroundColor": "rgb(248,248,248)"}],
                tooltip_header={
                    "Radiator": "Radiator number.",
                    "Room":     "Select the correct room.",
                    "Collector":"Select the corresponding collector.",
                    "Radiator power 75/65/20": "Nominal power at 75/65/20 (W).",
                    "Length circuit": "Circuit length collector → radiator (m).",
                    "Electric power": "Extra electric power added for heating.",
                },
                tooltip_delay=200, tooltip_duration=None,
            )
        ])
    ], className="mb-4")


def _collector_table_card() -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader("🧰 Collectors"),
        dbc.CardBody([
            dash_table.DataTable(
                id="collector-table", editable=True,
                columns=[
                    {"name": "Collector",               "id": "Collector",               "type": "text"},
                    {"name": "Collector circuit length", "id": "Collector circuit length", "type": "numeric"},
                ],
                data=[], page_size=10,
                tooltip_header={"Collector": "Collector name.", "Collector circuit length": "Distance generator → collector (m)."},
                style_cell={"padding": "8px", "border": "1px solid #dee2e6"},
                style_header={"backgroundColor": "#f8f9fa", "fontWeight": "bold", "textAlign": "center"},
                style_data_conditional=[{"if": {"column_id": "Collector circuit length"}, "backgroundColor": "#fffef0"}],
            )
        ])
    ], className="mb-4")


def build_radiator_tab() -> dbc.Tab:
    return dbc.Tab(
        label="2️⃣ Radiators & Collectors", tab_id="tab-2",
        children=[dbc.Card([dbc.CardBody([
            dbc.Row([
                dbc.Col([_system_config_card(), _pump_settings_card()], md=3),
                dbc.Col([_valve_settings_card()], md=3),
                dbc.Col([_radiator_table_card(), _collector_table_card()], md=6),
            ]),
            dbc.Card([
                dbc.CardHeader("📊 Radiator/Room Heat Loss Results"),
                dbc.CardBody([
                    dash_table.DataTable(
                        id="heat-loss-split-table",
                        columns=[
                            {"name": "Radiator",                  "id": "Radiator nr",              "type": "numeric"},
                            {"name": "Room",                      "id": "Room",                     "type": "any"},
                            {"name": "Calculated Heat Loss (W)", "id": "Calculated Heat Loss (W)", "type": "numeric"},
                        ],
                        data=[], page_size=10,
                        style_cell={"padding": "8px", "border": "1px solid #dee2e6"},
                        style_header={"backgroundColor": "#f8f9fa", "fontWeight": "bold", "textAlign": "center"},
                    )
                ])
            ], style={"backgroundColor": "#f0f4ff", "border": "1px solid #cce", "boxShadow": "0 0 6px rgba(0,0,0,0.1)"},
               className="mb-4"),
        ])])]
    )


# ---------------------------------------------------------------------------
# Tab 3 — Results
# ---------------------------------------------------------------------------
def _metric_card(icon_cls: str, icon_color: str, metric_id: str, label: str, default: str, md: int) -> dbc.Col:
    return dbc.Col([dbc.Card([dbc.CardBody([
        html.Div([
            html.I(className=f"{icon_cls} me-2", style={"fontSize": "2rem", "color": icon_color}),
            html.Div([
                html.H3(id=metric_id, children=default, className="mb-0"),
                html.P(label, className="text-muted mb-0 small"),
            ]),
        ], className="d-flex align-items-center"),
    ])], className="shadow-sm border-0 h-100")], md=md, className="mb-3")


def build_results_tab() -> dbc.Tab:
    chart = lambda cid: dcc.Graph(id=cid, style={"height": f"{CHART_HEIGHT_PX}px"}, config={"displayModeBar": True})
    def chart_card(header, cid, md=6): return dbc.Col([
        dbc.Card([dbc.CardHeader(header, className="fw-bold"), dbc.CardBody(chart(cid), className="p-2")])
    ], md=md, className="mb-3")

    return dbc.Tab(
        label="3️⃣ Results", tab_id="tab-3",
        children=[dbc.Card([dbc.CardBody([
            html.H4("Results"),
            html.Div(id="results-warnings", className="alert alert-warning", role="alert"),
            html.Hr(),
            html.H5("Performance Metrics"),
            dbc.Row([
                _metric_card("bi bi-building",        "#e74c3c", "metric-total-heat-loss",  "Total Heat Loss",     "0 W",    md=2),
                _metric_card("bi bi-fire",             "#f39c12", "metric-total-power",      "Total Radiator Power","0 W",    md=2),
                _metric_card("bi bi-droplet",          "#3498db", "metric-flow-rate",        "Total Flow Rate",     "0 kg/h", md=2),
                _metric_card("bi bi-speedometer2",     "#27ae60", "metric-delta-t",          "Weighted ΔT",         "0 °C",   md=2),
                _metric_card("bi bi-thermometer-high", "#e74c3c", "metric-highest-supply",   "Highest Supply T",    "N/A",    md=3),
            ], className="mb-4"),
            html.Hr(),
            html.H5("System Performance"),
            dbc.Row([chart_card("Power Distribution", "power-distribution-chart", md=8),
                     chart_card("Temperature Profile",  "temperature-profile-chart", md=4)], className="g-3"),
            dbc.Row([chart_card("Pressure Loss Analysis", "pressure-loss-chart"),
                     chart_card("Mass Flow Rate",          "mass-flow-chart")], className="g-3"),
            dbc.Row([chart_card("Pump vs System Curve", "pump-curve-chart", md=12)], className="g-3"),
            dbc.Row([chart_card("Valve Position Analysis", "valve-position-chart", md=12)], className="g-3"),
            html.Div(id="valve-balancing-section", style={"display": "none"}, children=[
                html.Hr(),
                html.H5("🔧 Valve Balancing"),
                dbc.Alert([
                    html.Strong("Balancing mode: "),
                    "The table shows each radiator's calculated optimal valve position, along with "
                    "the resulting mass flow, return temperature and pressure losses. ",
                    "Enter a position in the ",
                    html.Strong("Position Override"),
                    " column to fix a valve manually, then click ",
                    html.Strong("Apply"),
                    " — a full hydraulic network re-solve will recalculate mass flows, "
                    "return temperatures and pressure losses for all radiators.",
                ], color="info", className="mb-3"),
                dash_table.DataTable(
                    id="valve-balancing-table",
                    editable=True,
                    page_size=20,
                    style_table={"overflowX": "auto"},
                    style_header={"backgroundColor": "#f8f9fa", "fontWeight": "bold", "textAlign": "center"},
                    style_cell={"padding": "8px", "textAlign": "left", "border": "1px solid #dee2e6"},
                    style_data_conditional=[
                        {"if": {"row_index": "odd"}, "backgroundColor": "rgb(248,248,248)"},
                        {"if": {"column_id": "Position Override"}, "backgroundColor": "#fff8e1"},
                    ],
                    tooltip_header={
                        "Radiator nr":        "Radiator number.",
                        "Room":               "Room this radiator serves.",
                        "Flow (kg/h)":        "Actual mass flow rate after network solve.",
                        "T return (°C)":      "Return water temperature at this mass flow.",
                        "Calc. Position":     "Auto-calculated optimal valve position.",
                        "Valve ΔP (Pa)":      "Valve pressure loss at the actual mass flow.",
                        "Total ΔP (Pa)":      "Total circuit pressure loss (pipes + valve).",
                        "Position Override":  "Enter a valve position to fix it, then click Apply to trigger a full recalculation.",
                    },
                    tooltip_delay=200, tooltip_duration=None,
                ),
                dbc.Button("▶ Apply overrides", id="apply-valve-overrides-btn",
                           color="primary", className="mt-3"),
                html.Div(id="valve-override-feedback", className="text-success small mt-2"),
            ]),
            html.Hr(),
            html.H5("Summary"),
            html.Div(id="summary-metrics", className="alert alert-info"),
            html.Hr(),
            html.H5("Detailed Results"),
            html.H6("Radiators", className="mt-4 mb-3"),
            dash_table.DataTable(
                id="merged-results-table", page_size=15, style_table={"overflowX": "auto"},
                style_header={"backgroundColor": "#f8f9fa", "fontWeight": "bold", "textAlign": "center"},
                style_cell={"padding": "8px", "textAlign": "left", "border": "1px solid #dee2e6"},
                style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "rgb(248,248,248)"}],
            ),
            html.H6("Collectors", className="mt-4 mb-3"),
            dash_table.DataTable(
                id="collector-results-table", page_size=10, style_table={"overflowX": "auto"},
                style_header={"backgroundColor": "#f8f9fa", "fontWeight": "bold", "textAlign": "center"},
                style_cell={"padding": "8px", "textAlign": "left", "border": "1px solid #dee2e6"},
                style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "rgb(248,248,248)"}],
            ),
        ])])]
    )


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------
def build_stores() -> list:
    return [
        dcc.Store(id="room-results-store"),
        dcc.Store(id="radiator-data-store"),
        dcc.Store(id="collector-data-store"),
        dcc.Store(id="heat-loss-split-store"),
        dcc.Store(id="config-store"),
        dcc.Store(id="heat-load-mode-store", data="unknown"),
        dcc.Store(id="valve-override-store", data=[]),
    ]