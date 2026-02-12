import hashlib
import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, dash_table, Input, Output, State, no_update, callback_context
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

from simpleLoadModel import RoomLoadCalculator
from utils.helpers import (
    POSSIBLE_DIAMETERS, Radiator, Circuit, Collector, Valve,
    validate_data, calculate_weighted_delta_t, EXPONENT_RADIATOR,
    calculate_extra_power_needed, calc_velocity
)


# ==========================
# Constants & small helpers
# ==========================
INSULATION_U_VALUES = {
    "not insulated": {"wall": 1.3, "roof": 1.0, "ground": 1.2},
    "bit insulated": {"wall": 0.6, "roof": 0.4, "ground": 0.5},
    "insulated well": {"wall": 0.3, "roof": 0.2, "ground": 0.3},
}
GLAZING_U_VALUES = {"single": 5.0, "double": 2.8, "triple": 0.8}

CHART_HEIGHT_PX = 460
ROOM_TYPE_OPTIONS = ["Living", "Kitchen", "Bedroom", "Laundry", "Bathroom", "Toilet"]
ROOM_TYPE_HELP_MD = (
    "**Room Type** influences heat loss / ventilation defaults.\n\n"
    "- **Living**: 20–21 °C\n"
    "- **Kitchen**: 18–20 °C\n"
    "- **Bedroom**: 16–18 °C\n"
    "- **Laundry**: humid space, more ventilation\n"
    "- **Bathroom**: 22 °C design, short peaks\n"
    "- **Toilet**: ~18 °C"
)

# --------- Design modes ---------
MODE_EXISTING = "existing"   # Bestaande radiatoren → bereken benodigde supply T
MODE_FIXED    = "fixed"      # LT dimensioning → vaste supply T → extra vermogen
MODE_PUMP     = "pump"       # Pomp-gedreven ontwerp → werkpunt pomp vs systeem (debiet gestuurde ΔT)
MODE_BAL      = "balancing"  # Hydraulische inregeling / TRV balancing

# ========= Pump library =========
# Voorbeeldcurves (kg/h, kPa). Representatief voor UX/test.
PUMP_LIBRARY: Dict[str, Dict[str, List[Tuple[float, float]]]] = {
    "Grundfos UPM3 15-70": {
        "speed_1": [(0, 55), (200, 50), (400, 42), (600, 30), (800, 18), (1000, 6), (1100, 2)],
        "speed_2": [(0, 65), (250, 60), (500, 51), (750, 38), (1000, 24), (1150, 12), (1250, 5)],
        "speed_3": [(0, 75), (300, 70), (600, 60), (900, 44), (1200, 28), (1400, 16), (1500, 8)],
    },
    "Wilo Yonos PICO 25-1/6": {
        "speed_1": [(0, 50), (250, 44), (500, 36), (750, 26), (1000, 15), (1200, 7)],
        "speed_2": [(0, 60), (300, 54), (600, 45), (900, 33), (1200, 20), (1400, 12)],
        "speed_3": [(0, 70), (350, 64), (700, 54), (1050, 40), (1400, 26), (1600, 15)],
    },
    "Generic 25-60": {
        "speed_1": [(0, 48), (250, 42), (500, 34), (750, 24), (1000, 13), (1200, 6)],
        "speed_2": [(0, 58), (300, 52), (600, 44), (900, 32), (1200, 19), (1400, 11)],
        "speed_3": [(0, 68), (350, 62), (700, 52), (1050, 38), (1400, 24), (1600, 14)],
    }
}

# ==========================
# Utility functions (UI)
# ==========================
def hash_dataframe(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df.fillna("__NaN__"), index=True).values).hexdigest()

def default_room_table(num_rooms: int) -> List[Dict[str, Any]]:
    rows = []
    for i in range(1, num_rooms + 1):
        rows.append({
            "Room #": i, "Indoor Temp (°C)": 20.0, "Floor Area (m²)": 20.0, "Walls external": 2,
            "Room Type": "Living", "On Ground": False, "Under Roof": False,
        })
    return rows

def compute_room_results(
    room_rows: List[Dict[str, Any]],
    uw: float, u_roof: float, u_ground: float, u_glass: float,
    tout: float,
    heat_loss_area_estimation: str,
    ventilation_calculation_method: str,
    v_system: str,
    v50: float,
    neighbour_t: float, un: float, lir: float,
    wall_height: float,
    return_detail: bool,
    add_neighbour_losses: bool
) -> pd.DataFrame:
    room_results = []
    for row in room_rows:
        calc = RoomLoadCalculator(
            floor_area=row.get("Floor Area (m²)", 0.0),
            uw=uw, u_roof=u_roof, u_ground=u_ground,
            v_system=v_system, v50=v50,
            tin=row.get("Indoor Temp (°C)", 20.0), tout=tout,
            neighbour_t=neighbour_t, un=un, lir=lir,
            heat_loss_area_estimation=heat_loss_area_estimation,
            ventilation_calculation_method=ventilation_calculation_method,
            on_ground=row.get("On Ground", False), under_roof=row.get("Under Roof", False),
            add_neighbour_losses=add_neighbour_losses,
            neighbour_perimeter=row.get("Neighbour Perimeter (m)", 0.0),
            room_type=row.get("Room Type", "Living"), wall_height=wall_height,
            wall_outside=row.get("Walls external", 0),
            return_detail=return_detail, window=True, u_glass=u_glass
        )
        result = calc.compute()
        total = result if isinstance(result, (int, float)) else result.get("totalHeatLoss", 0.0)
        room_results.append({"Room": row["Room #"], "Total Heat Loss (W)": float(total) if total is not None else 0.0})
    return pd.DataFrame(room_results)

def init_radiator_rows(n: int, collector_options: List[str], room_options: List[Any]) -> List[Dict[str, Any]]:
    rows = []
    for i in range(1, n + 1):
        rows.append({
            "Radiator nr": i,
            "Collector": collector_options[0] if collector_options else "Collector 1",
            "Radiator power 75/65/20": 0.0,
            "Length circuit": 0.0,
            "Space Temperature": 20.0,
            "Electric power": 0.0,
            "Room": room_options[(i - 1) % max(1, len(room_options))] if room_options else 1,
        })
    return rows

def resize_radiator_rows(current: List[Dict[str, Any]], desired_num: int, collector_options: List[str], room_options: List[Any]) -> List[Dict[str, Any]]:
    rows = (current or []).copy()
    cur_n = len(rows)
    if desired_num > cur_n:
        rows.extend(init_radiator_rows(desired_num - cur_n, collector_options, room_options))
    for idx, r in enumerate(rows, start=1):
        r["Radiator nr"] = idx
    if desired_num < cur_n:
        rows = rows[:desired_num]
    for idx, r in enumerate(rows, start=1):
        r["Radiator nr"] = idx
    return rows

def init_collector_rows(n: int, start: int = 1) -> List[Dict[str, Any]]:
    return [{"Collector": f"Collector {i}", "Collector circuit length": 0.0} for i in range(start, start + n)]

def resize_collector_rows(current: List[Dict[str, Any]], desired: int) -> List[Dict[str, Any]]:
    rows = (current or []).copy()
    cur_n = len(rows)
    if desired > cur_n:
        rows.extend(init_collector_rows(desired - cur_n, start=cur_n + 1))
    elif desired < cur_n:
        rows = rows[:desired]
    return rows

def split_heat_loss_to_radiators(radiator_rows: List[Dict[str, Any]], room_results_df: pd.DataFrame) -> pd.DataFrame:
    if room_results_df is None or room_results_df.empty or not radiator_rows:
        return pd.DataFrame(columns=["Radiator nr", "Calculated Heat Loss (W)", "Room"])
    rad_df = pd.DataFrame(radiator_rows)
    heat_loss_df = pd.DataFrame({
        "Radiator nr": rad_df["Radiator nr"], "Calculated Heat Loss (W)": 0.0, "Room": rad_df["Room"]
    })
    room_heat_map = room_results_df.set_index("Room")["Total Heat Loss (W)"].to_dict()
    for room, idxs in rad_df.groupby("Room").groups.items():
        n_rad = len(idxs)
        if room in room_heat_map and n_rad > 0:
            split_load = room_heat_map[room] / n_rad
            heat_loss_df.loc[idxs, "Calculated Heat Loss (W)"] = split_load
    return heat_loss_df

def safe_to_float(x, default=None):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default

def fix_fig(fig, title=None, height=CHART_HEIGHT_PX):
    fig.update_layout(
        height=height, autosize=False, margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    bgcolor='rgba(255,255,255,0.8)', bordercolor='#ddd', borderwidth=1),
        transition_duration=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif", size=12, color="#333"),
        hoverlabel=dict(bgcolor="white", font_size=13, font_family="Arial, sans-serif")
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0.5, xanchor='center',
                                     font=dict(size=16, family="Arial, sans-serif", color="#2c3e50")))
    fig.update_yaxes(automargin=True, showline=True, linewidth=1, linecolor='#ddd', gridcolor='#eee', zeroline=False)
    fig.update_xaxes(automargin=True, showline=True, linewidth=1, linecolor='#ddd', gridcolor='#eee')
    return fig

def empty_fig(title="", height=CHART_HEIGHT_PX):
    fig = px.scatter()
    return fix_fig(fig, title=title, height=height)

def determine_system_supply_temperature(calc_rows: List[Radiator], cfg: Dict[str, Any]) -> float:
    """Bepaal supply T uit per-radiator kandidaten; fallback 55°C."""
    candidate_supply_ts = []
    for r in calc_rows:
        t_sup = getattr(r, "supply_temperature", None)
        if t_sup is None:
            for m in ("required_supply_temperature", "calculate_supply_temperature", "compute_supply_temperature"):
                if hasattr(r, m):
                    try:
                        t_sup = float(getattr(r, m)())
                        break
                    except Exception:
                        t_sup = None
        if t_sup is not None:
            try:
                t_val = float(t_sup)
                if not math.isnan(t_val):
                    candidate_supply_ts.append(t_val)
            except Exception:
                pass
    if candidate_supply_ts:
        return max(candidate_supply_ts)
    return 55.0


# ==========================
# Centrale exception wrapper
# ==========================
def capture_value_errors(fn, label, warnings, fallback):
    """Run fn() en zet ValueError tekst in warnings; geef fallback terug zodat pipeline doorgaat."""
    try:
        return fn()
    except ValueError as e:
        warnings.append(f"{label}: {e}")
        return fallback


# ==========================
# Pump curve helpers
# ==========================
def interpolate_curve(points: List[Tuple[float, float]], x_vals: np.ndarray) -> np.ndarray:
    """Lineaire interpolatie van pomp- of systeemcurve; buiten het bereik wordt NaN teruggegeven."""
    if not points:
        return np.full_like(x_vals, np.nan, dtype=float)
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    y_interp = np.interp(x_vals, xs, ys, left=np.nan, right=np.nan)
    return y_interp

def derive_system_curve_kpa(Q_total_kgph: float, branch_kpa: float) -> float:
    """Benader K in ΔP_sys = K * Q^2 met huidig werkpunt: branch_kpa bij Q_total."""
    if Q_total_kgph <= 0:
        return 0.0
    return max(branch_kpa, 0.0) / max(Q_total_kgph, 1e-6)**2

def find_operating_point(Q_grid: np.ndarray, pump_kpa: np.ndarray, K_sys: float) -> Tuple[Optional[float], Optional[float], List[str]]:
    """Vind kruising van pompcurve (ΔP_pump vs Q) en systeemcurve (K*Q^2).
    Return (Q*, ΔP*), plus warnings als er geen geldige kruising is."""
    warnings: List[str] = []
    sys_kpa = K_sys * (Q_grid ** 2)
    diff = pump_kpa - sys_kpa
    mask = ~np.isnan(pump_kpa) & ~np.isnan(sys_kpa)
    if mask.sum() < 2:
        warnings.append("Pump/system curve insufficient data to determine operating point.")
        return None, None, warnings
    Qm = Q_grid[mask]
    Dm = diff[mask]
    sign = np.sign(Dm)
    sign_change_idx = np.where(np.diff(sign) != 0)[0]
    if len(sign_change_idx) == 0:
        if np.all(Dm > 0):
            warnings.append("Pump curve stays above system curve in range — operating point may be at higher flow than provided curve.")
        else:
            warnings.append("System curve stays above pump curve in range — pump likely insufficient for required head.")
        return None, None, warnings
    i = sign_change_idx[0]
    x0, x1 = Qm[i], Qm[i+1]
    y0, y1 = Dm[i], Dm[i+1]
    q_star = x0 - y0 * (x1 - x0) / (y1 - y0) if (y1 - y0) != 0 else x0
    dp_pump_star = np.interp(q_star, Qm, pump_kpa[mask])
    return float(q_star), float(dp_pump_star), warnings


# ---------- Dash app ----------
# Use a modern Bootstrap theme + icons
external_stylesheets = [dbc.themes.ZEPHYR, dbc.icons.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app.title = "Smart Heating Design Tool (Dash)"
server = app.server

# Stores (state)
stores = [
    dcc.Store(id="room-results-store"),
    dcc.Store(id="radiator-data-store"),
    dcc.Store(id="collector-data-store"),
    dcc.Store(id="heat-loss-split-store"),
    dcc.Store(id="config-store"),
    dcc.Store(id="heat-load-mode-store", data="unknown"),  # "unknown" (calculate) | "known" (manual)
]

# -----------
# Shared layout
# -----------
navbar = dbc.Navbar(
    dbc.Container([
        html.A(
            dbc.Row([
                dbc.Col(html.Img(src="/assets/Recover_logo.png", height="50px"), width="auto"),
                dbc.Col(dbc.NavbarBrand("Smart Heating Design Tool", className="ms-2"), width="auto"),
            ], align="center", className="g-0"),
            href="/", style={"textDecoration": "none"},
        ),
        dbc.Nav(
            [
                dbc.NavItem(dbc.NavLink(html.I(className="bi bi-house-door me-1"), " Home", href="/home")),
                dbc.NavItem(dbc.NavLink(html.I(className="bi bi-info-circle me-1"), " Help", href="/help")),
            ],
            navbar=True, className="ms-auto",
        ),
    ], fluid=True),
    color="dark", dark=True, sticky="top",
)

# -----------
# Tab 0 — Start
# -----------
start_tab = dbc.Tab(
    label="0️⃣ Start",
    tab_id="tab-0",
    children=[
        html.Div(
            [
                html.H2("Welkom in de Smart Heating Design Tool", className="mb-4"),
                html.Hr(className="my-4"),

                # Stap 2
                html.H4("Stap 1 — Ontwerpmodus", className="mt-4"),
                dcc.Dropdown(
                    id="design-mode",
                    options=[
                        {"label": "Bestaand systeem — Bereken aanvoertemperatuur", "value": MODE_EXISTING},
                        {"label": "Dimensionering op aanvoertemperatuur → berekend extra vermogen", "value": MODE_FIXED},
                        {"label": "Pump-based design — Pompcurve bepaalt debiet", "value": MODE_PUMP},
                        {"label": "Balancering modus — TRV-inregeling", "value": MODE_BAL},
                    ],
                    value=MODE_EXISTING,
                    clearable=False,
                    className="mb-3",
                ),
                html.Div(id="design-mode-help", className="text-muted small mb-4"),
                dbc.Alert(
                    "De gekozen ontwerpmodus bepaalt welke parameters je in Tab 2 ziet "
                    "en welke berekeningen Tab 3 uitvoert.",
                    color="secondary",
                ),
                # Stap 1
                html.H4("Stap 2 — Heat loss modus", className="mt-3"),
                dbc.RadioItems(
                    id="heat-load-mode",
                    options=[
                        {"label": "Warmteverliezen zijn BEKEND (manuele invoer)", "value": "known"},
                        {"label": "Warmteverliezen zijn NIET gekend (berekenen)", "value": "unknown"},
                    ],
                    value=None,
                    className="mb-4"
                ),
                dbc.Alert(
                    "Gebruik deze stap om te bepalen hoe de tool de warmteverliezen per kamer moet kennen.",
                    color="info",
                ),
            ],
            className="p-4"
        )
    ]
)

# -----------
# Tab 1 — Heat Loss
# -----------
heat_tab = dbc.Tab(
    label="1️⃣ Heat Loss", tab_id="tab-1",
    children=[
        dbc.Card([
            dbc.CardBody([
                dbc.Alert(id="heat-load-mode-banner", color="secondary", className="mb-3"),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("📐 Insulation level"),
                            dbc.CardBody([
                                dbc.Label("Wall"),
                                dcc.Dropdown(
                                    id="wall_insulation_state",
                                    options=[
                                        {"label": "Not Insulated (0 cm)", "value": "not insulated"},
                                        {"label": "Insulated (5 cm)", "value": "bit insulated"},
                                        {"label": "Insulated well (10 cm)", "value": "insulated well"},
                                    ],
                                    value="bit insulated", clearable=False, className="mb-2"
                                ),
                                dbc.Label("Roof"),
                                dcc.Dropdown(
                                    id="roof_insulation_state",
                                    options=[
                                        {"label": "Not insulated (0 cm)", "value": "not insulated"},
                                        {"label": "Insulated (5 cm)", "value": "bit insulated"},
                                        {"label": "Insulated well (>10 cm)", "value": "insulated well"},
                                    ],
                                    value="bit insulated", clearable=False,
                                ),
                                dbc.Label("Ground"),
                                dcc.Dropdown(
                                    id="ground_insulation_state",
                                    options=[
                                        {"label": "Not insulated (0 cm)", "value": "not insulated"},
                                        {"label": "Insulated (5 cm)", "value": "bit insulated"},
                                        {"label": "Insulated well (>10 cm)", "value": "insulated well"},
                                    ],
                                    value="bit insulated", clearable=False, className="mb-2"
                                ),
                                dbc.Label("Glazing Type"),
                                dcc.Dropdown(
                                    id="glazing_type",
                                    options=[
                                        {"label": "Single", "value": "single"},
                                        {"label": "Double", "value": "double"},
                                        {"label": "Triple", "value": "triple"},
                                    ],
                                    value="double", clearable=False, className="mb-2"
                                ),
                                dbc.FormText("Selecting an insulation or glazing type sets a typical U-value, but you can override it manually"),
                            ])
                        ], className="mb-4", id="building-insulation-card"),
                    ], md=3),

                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("🏠 Building Envelope"),
                            dbc.CardBody([
                                dbc.Label("Wall U-value (W/m²K)"),
                                dbc.Input(id="uw", type="number", value=1.0, step=0.05), dbc.FormText("External wall U-value"), html.Br(),
                                dbc.Label("Roof U-value (W/m²K)"),
                                dbc.Input(id="u_roof", type="number", value=0.2, step=0.05), dbc.FormText("Roof U-value"), html.Br(),
                                dbc.Label("Ground U-value (W/m²K)"),
                                dbc.Input(id="u_ground", type="number", value=0.3, step=0.05), dbc.FormText("Slab-on-grade U-value"), html.Br(),
                                dbc.Label("Glazing U-value (W/m²K)"),
                                dbc.Input(id="u_glass", type="number", value=2.8, step=0.05), dbc.FormText("Window glazing U-value"),
                            ])
                        ], className="mb-4", id="building-envelope-card"),
                    ], md=3),

                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("🌡️ Outdoor Conditions"),
                            dbc.CardBody([
                                dbc.Label("Outdoor Temperature (°C)"),
                                dbc.Input(id="tout", type="number", min=-12.0, max=40.0, value=-7.0, step=0.5),
                                dbc.FormText("Design outdoor winter temperature"),
                            ])
                        ], className="mb-4", id="outdoor-conditions-card"),
                        dbc.Card([
                            dbc.CardHeader("Number of Rooms"),
                            dbc.CardBody([
                                dbc.Label("Number of Rooms"),
                                dbc.Input(id="num_rooms", type="number", min=1, value=3, step=1, style={"maxWidth": "160px"}),
                                dbc.FormText("Add rooms first then details."),
                            ])
                        ], className="mb-4"),
                    ], md=3),

                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("⚙️ Additional Settings"),
                            dbc.CardBody([
                                dbc.Accordion([
                                    dbc.AccordionItem([
                                        dbc.Label("Ventilation Calculation Method"),
                                        dcc.Dropdown(
                                            id="ventilation_calculation_method",
                                            options=[{"label": "simple", "value": "simple"}, {"label": "NBN-D-50-001", "value": "NBN-D-50-001"}],
                                            value="simple", clearable=False, className="dash-dropdown"
                                        ),
                                        html.Br(),
                                        dbc.Label("Ventilation System"),
                                        dcc.Dropdown(
                                            id="v_system", options=[{"label": k, "value": k} for k in ["C", "D"]],
                                            value="C", clearable=False, className="dash-dropdown"
                                        ),
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
                                ], start_collapsed=True, always_open=False, flush=True),
                            ])
                        ], className="mb-4", id="additional-settings-card"),
                    ], md=3),
                ]),

                dbc.Card([
                    dbc.CardHeader("🧾 Room Configuration"),
                    dbc.CardBody([
                        dash_table.DataTable(
                            id="room-table", editable=True, row_deletable=False,
                            columns=[
                                {"name": "Room #", "id": "Room #", "type": "numeric", "editable": False},
                                {"name": "Indoor Temp (°C)", "id": "Indoor Temp (°C)", "type": "numeric", "editable": True, "presentation": "input"},
                                {"name": "Floor Area (m²)", "id": "Floor Area (m²)", "type": "numeric"},
                                {"name": "Walls external", "id": "Walls external", "type": "numeric", "presentation": "dropdown"},
                                {"name": "Room Type", "id": "Room Type", "type": "text", "presentation": "dropdown"},
                                {"name": "On Ground", "id": "On Ground", "type": "any", "presentation": "dropdown"},
                                {"name": "Under Roof", "id": "Under Roof", "type": "any", "presentation": "dropdown"},
                            ],
                            dropdown={
                                "Room Type": {"options": [{"label": str(v), "value": v} for v in ROOM_TYPE_OPTIONS]},
                                "On Ground": {"options": [{"label": "No", "value": False}, {"label": "Yes", "value": True}]},
                                "Under Roof": {"options": [{"label": "No", "value": False}, {"label": "Yes", "value": True}]},
                                "Walls external": {"options": [{"label": str(v), "value": v} for v in [1, 2, 3, 4]]},
                            },
                            tooltip_header={
                                "Room Type": ROOM_TYPE_HELP_MD, "On Ground": "Space on ground slab / exposed to ground.",
                                "Under Roof": "Space directly under roof.", "Walls external": "Number of external walls (1-4).",
                                "Indoor Temp (°C)": "Set between 10°C and 24°C.",
                            },
                            data=default_room_table(3), page_size=25,
                            style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold', 'textAlign': 'center'},
                            style_cell={'padding': '8px', 'textAlign': 'left', 'border': '1px solid #dee2e6'},
                            style_data_conditional=[
                                {'if': {'column_id': c}, 'backgroundColor': '#fffef0'}
                                for c in ["Indoor Temp (°C)", "Floor Area (m²)", "Walls external", "Room Type", "On Ground", "Under Roof"]
                            ] + [{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}],
                            tooltip_delay=200, tooltip_duration=None
                        ),
                        html.Div([html.Small(
                            "Tip: Double-click cells to edit. For 'Walls external', choose 1-4. For 'Indoor Temp', use 10–24°C.",
                            className="text-muted")], className="mt-2"),
                    ])
                ], className="mb-4", id="room-config-card"),

                dbc.Card([
                    dbc.CardHeader("✍️ Manual Room Heat Loss (visible when 'known' is selected)"),
                    dbc.CardBody([
                        dash_table.DataTable(
                            id="manual-loss-table", editable=True, row_deletable=False,
                            columns=[{"name": "Room #", "id": "Room #", "type": "numeric", "editable": False},
                                     {"name": "Manual Heat Loss (W)", "id": "Manual Heat Loss (W)", "type": "numeric"}],
                            data=[{"Room #": i, "Manual Heat Loss (W)": 0.0} for i in range(1, 4)], page_size=10,
                            style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold', 'textAlign': 'center'},
                            style_cell={'padding': '8px', 'border': '1px solid #dee2e6'},
                            style_data_conditional=[
                                {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'},
                                {'if': {'column_id': 'Manual Heat Loss (W)'}, 'backgroundColor': '#fffef0'}],
                            tooltip_header={"Manual Heat Loss (W)": "Enter the design heat loss for each room (W)."},
                            tooltip_delay=200, tooltip_duration=None
                        ),
                        html.Small("When this table is visible, Tab 2 & 3 will use these values instead of the calculated ones.",
                                   className="text-muted"),
                    ])
                ], id="manual-loss-card", style={"display": "none"}, className="mb-4"),

                dbc.Card([
                    dbc.CardHeader("📊 Room Heat Loss Results"),
                    dbc.CardBody([html.Div(id="room-results-table")])
                ], className="mb-4", style={"backgroundColor": "#f0f4ff", "border": "1px solid #cce",
                                            "boxShadow": "0 0 6px rgba(0,0,0,0.1)"}),
            ])
        ])
    ]
)

# -----------
# Tab 2 — Radiators & Collectors
# -----------
tab2_system = dbc.Card([
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🛠️ System Configuration"),
                    dbc.CardBody([
                        dbc.Label("Number of radiators"), dbc.Input(id="num_radiators", type="number", min=1, value=3, step=1), html.Br(),
                        dbc.Label("Number of collectors"), dbc.Input(id="num_collectors", type="number", min=1, value=1, step=1), html.Br(),
                        dbc.Label("Delta T (°C)"), dbc.Input(id="delta_T", type="number", min=3, max=20, step=1, value=10), html.Br(),
                        html.Div([
                            html.Span("Optional Inputs", className="form-label fw-bold mt-3 me-2"),
                            html.Span(html.I(className="bi bi-info-circle text-warning", id="experimental-tooltip"), className="align-middle"),
                            dbc.Tooltip("⚠️ Experimental: These inputs may result in unexpected behavior. Use with caution.",
                                        target="experimental-tooltip", placement="right"),
                        ]),
                        dbc.Label("Supply temperature (°C)"),
                        dbc.Input(id="supply_temp_input", type="number", placeholder="(optional)"),
                        html.Br(),
                        dbc.Checklist(id="fix_diameter", options=[{"label": " Fix diameter for all radiators", "value": "yes"}], value=[]),
                        html.Div([
                            dbc.Label("Fixed diameter (mm)"),
                            dcc.Dropdown(id="fixed_diameter",
                                         options=[{"label": str(mm), "value": mm} for mm in [12, 14, 16, 18, 20, 22, 25, 28, 36]],
                                         value=16, clearable=False),
                        ], id="fixed_diameter_container"),
                    ])
                ], className="mb-4"),

                dbc.Card([
                    dbc.CardHeader("♻️ Pump settings"),
                    dbc.CardBody([
                        dcc.Dropdown(
                            id="pump_model",
                            options=[{"label": name, "value": name} for name in PUMP_LIBRARY.keys()],
                            value="Grundfos UPM3 15-70", clearable=False, className="mb-2"
                        ),
                        dcc.Dropdown(
                            id="pump_speed",
                            options=[{"label": "Speed 1", "value": "speed_1"},
                                     {"label": "Speed 2", "value": "speed_2"},
                                     {"label": "Speed 3", "value": "speed_3"}],
                            value="speed_2", clearable=False
                        ),
                        html.Small("Kies een interne pompcurve (bibliotheek) en snelheid.", className="text-muted")
                    ])
                ], className="mb-4", id="pump-settings-card", style={"display": "none"}),
            ], md=3),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🧩 Valve Settings"),
                    dbc.CardBody([
                        dbc.Label("Valve Type"),
                        dcc.Dropdown(
                            id="valve-type-dropdown",
                            options=[
                                {"label": "Custom", "value": "Custom"},
                                {"label": "Danfoss RA-N 10 (3/8)", "value": "Danfoss RA-N 10 (3/8)"},
                                {"label": "Danfoss RA-N 15 (1/2)", "value": "Danfoss RA-N 15 (1/2)"},
                                {"label": "Danfoss RA-N 20 (3/4)", "value": "Danfoss RA-N 20 (3/4)"},
                                {"label": "Oventrop DN15 (1/2)", "value": "Oventrop DN15 (1/2)"},
                                {"label": "Heimeier (1/2)", "value": "Heimeier (1/2)"},
                                {"label": "Vogel und Noot", "value": "Vogel und Noot"},
                                {"label": "Comap", "value": "Comap"},
                            ],
                            value="Custom", clearable=False, className="mb-3"
                        ),
                        html.Div(id="valve-specs", className="small text-muted mb-3"),
                        html.Div(id="valve-custom-settings", children=[
                            dbc.Row([
                                dbc.Col([dbc.Label("Positions"), dbc.Input(id="positions", type="number", min=2, value=8, step=1, className="mb-3")]),
                                dbc.Col([dbc.Label("Kv max"), dbc.Input(id="kv_max", type="number", min=0.1, value=0.7, step=0.1)])
                            ])
                        ])
                    ])
                ], className="mb-4"),
            ], md=3),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🌡️ Radiator Inputs"),
                    dbc.CardBody([
                        dash_table.DataTable(
                            id="radiator-table",
                            editable=True, row_deletable=False,
                            columns=[
                                {"name": "Radiator", "id": "Radiator nr", "type": "numeric", "editable": False},
                                {"name": "Room", "id": "Room", "presentation": "dropdown"},
                                {"name": "Collector", "id": "Collector", "presentation": "dropdown"},
                                {"name": "Radiator power 75/65/20", "id": "Radiator power 75/65/20", "type": "numeric"},
                                {"name": "Length circuit", "id": "Length circuit", "type": "numeric"},
                                {"name": "Electric power", "id": "Electric power", "type": "numeric"},
                            ],
                            data=[], dropdown={}, page_size=10,
                            style_cell={'padding': '8px', 'border': '1px solid #dee2e6'},
                            style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold', 'textAlign': 'center'},
                            style_data_conditional=[
                                {'if': {'column_id': c}, 'backgroundColor': '#fffef0'}
                                for c in ["Room", "Collector", "Radiator power 75/65/20", "Length circuit", "Electric power"]
                            ] + [{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}],
                            tooltip_header={
                                "Radiator": "Radiator number.", "Room": "Select the correct room for the radiator.",
                                "Collector": "Select the corresponding collector.", "Radiator power 75/65/20": "Nominal power at 75/65/20 (W).",
                                "Length circuit": "Circuit length (m) = distance collector to the radiator.", "Electric power": "Extra electric power added for heating."
                            },
                            tooltip_delay=200, tooltip_duration=None
                        )
                    ])
                ], className="mb-4"),

                dbc.Card([
                    dbc.CardHeader("🧰 Collectors"),
                    dbc.CardBody([
                        dash_table.DataTable(
                            id="collector-table",
                            editable=True,
                            columns=[
                                {"name": "Collector", "id": "Collector", "type": "text"},
                                {"name": "Collector circuit length", "id": "Collector circuit length", "type": "numeric"},
                            ],
                            data=[], page_size=10,
                            tooltip_header={"Collector": "Collector name.", "Collector circuit length": "Distance generator to collector (m)."},
                            style_cell={'padding': '8px', 'border': '1px solid #dee2e6'},
                            style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold', 'textAlign': 'center'},
                            style_data_conditional=[{'if': {'column_id': 'Collector circuit length'}, 'backgroundColor': '#fffef0'}]
                        )
                    ])
                ], className="mb-4"),
            ], md=6),
        ]),

        dbc.Card([
            dbc.CardHeader("📊 Radiator/Room Heat Loss Results"),
            dbc.CardBody([
                dash_table.DataTable(
                    id="heat-loss-split-table",
                    columns=[
                        {"name": "Radiator", "id": "Radiator nr", "type": "numeric"},
                        {"name": "Room", "id": "Room", "type": "any"},
                        {"name": "Calculated Heat Loss (W)", "id": "Calculated Heat Loss (W)", "type": "numeric"},
                    ],
                    data=[], page_size=10,
                    style_cell={'padding': '8px', 'border': '1px solid #dee2e6'},
                    style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold', 'textAlign': 'center'}
                )
            ])
        ], style={"backgroundColor": "#f0f4ff", "border": "1px solid #cce", "boxShadow": "0 0 6px rgba(0,0,0,0.1)"},
           className="mb-4"),
    ])
])

rad_tab = dbc.Tab(label="2️⃣ Radiators & Collectors", tab_id="tab-2", children=[tab2_system])

# -----------
# Tab 3 — Results & Charts
# -----------
results_tab = dbc.Tab(
    label="3️⃣ Results", tab_id="tab-3",
    children=[
        dbc.Card([
            dbc.CardBody([
                html.H4("Results"),
                html.Div(id="results-warnings", className="alert alert-warning", role="alert"),
                html.Hr(),
                html.H5("Performance Metrics"),
                dbc.Row([
                    dbc.Col([dbc.Card([dbc.CardBody([
                        html.Div([html.I(className="bi bi-building me-2", style={"fontSize": "2rem", "color": "#e74c3c"}),
                                  html.Div([html.H3(id="metric-total-heat-loss", children="0 W", className="mb-0"),
                                            html.P("Total Heat Loss", className="text-muted mb-0 small")])],
                                className="d-flex align-items-center")
                    ])], className="shadow-sm border-0 h-100")], md=2, className="mb-3"),

                    dbc.Col([dbc.Card([dbc.CardBody([
                        html.Div([html.I(className="bi bi-fire me-2", style={"fontSize": "2rem", "color": "#f39c12"}),
                                  html.Div([html.H3(id="metric-total-power", children="0 W", className="mb-0"),
                                            html.P("Total Radiator Power", className="text-muted mb-0 small")])],
                                className="d-flex align-items-center")
                    ])], className="shadow-sm border-0 h-100")], md=2, className="mb-3"),

                    dbc.Col([dbc.Card([dbc.CardBody([
                        html.Div([html.I(className="bi bi-droplet me-2", style={"fontSize": "2rem", "color": "#3498db"}),
                                  html.Div([html.H3(id="metric-flow-rate", children="0 kg/h", className="mb-0"),
                                            html.P("Total Flow Rate", className="text-muted mb-0 small")])],
                                className="d-flex align-items-center")
                    ])], className="shadow-sm border-0 h-100")], md=2, className="mb-3"),

                    dbc.Col([dbc.Card([dbc.CardBody([
                        html.Div([html.I(className="bi bi-speedometer2 me-2", style={"fontSize": "2rem", "color": "#27ae60"}),
                                  html.Div([html.H3(id="metric-delta-t", children="0 °C", className="mb-0"),
                                            html.P("Weighted ΔT", className="text-muted mb-0 small")])],
                                className="d-flex align-items-center")
                    ])], className="shadow-sm border-0 h-100")], md=2, className="mb-3"),

                    dbc.Col([dbc.Card([dbc.CardBody([
                        html.Div([html.I(className="bi bi-thermometer-high me-2", style={"fontSize": "2rem", "color": "#e74c3c"}),
                                  html.Div([html.H3(id="metric-highest-supply", children="N/A", className="mb-0"),
                                            html.P("Highest Supply T", className="text-muted mb-0 small")])],
                                className="d-flex align-items-center")
                    ])], className="shadow-sm border-0 h-100")], md=3, className="mb-3"),
                ], className="mb-4"),

                html.Hr(),
                html.H5("System Performance"),
                dbc.Row([
                    dbc.Col([dbc.Card([dbc.CardHeader("Power Distribution", className="fw-bold"),
                                       dbc.CardBody(dcc.Graph(id="power-distribution-chart", style={"height": f"{CHART_HEIGHT_PX}px"},
                                                              config={"displayModeBar": True}), className="p-2")])], md=8, className="mb-3"),
                    dbc.Col([dbc.Card([dbc.CardHeader("Temperature Profile", className="fw-bold"),
                                       dbc.CardBody(dcc.Graph(id="temperature-profile-chart", style={"height": f"{CHART_HEIGHT_PX}px"},
                                                              config={"displayModeBar": True}), className="p-2")])], md=4, className="mb-3"),
                ], className="g-3"),
                dbc.Row([
                    dbc.Col([dbc.Card([dbc.CardHeader("Pressure Loss Analysis", className="fw-bold"),
                                       dbc.CardBody(dcc.Graph(id="pressure-loss-chart", style={"height": f"{CHART_HEIGHT_PX}px"},
                                                              config={"displayModeBar": True}), className="p-2")])], md=6, className="mb-3"),
                    dbc.Col([dbc.Card([dbc.CardHeader("Mass Flow Rate", className="fw-bold"),
                                       dbc.CardBody(dcc.Graph(id="mass-flow-chart", style={"height": f"{CHART_HEIGHT_PX}px"},
                                                              config={"displayModeBar": True}), className="p-2")])], md=6, className="mb-3"),
                ], className="g-3"),
                dbc.Row([
                    dbc.Col([dbc.Card([dbc.CardHeader("Pump vs System Curve", className="fw-bold"),
                                       dbc.CardBody(dcc.Graph(id="pump-curve-chart", style={"height": f"{CHART_HEIGHT_PX}px"},
                                                              config={"displayModeBar": True}), className="p-2")])], md=12, className="mb-3"),
                ], className="g-3"),
                dbc.Row([
                    dbc.Col([dbc.Card([dbc.CardHeader("Valve Position Analysis", className="fw-bold"),
                                       dbc.CardBody(dcc.Graph(id="valve-position-chart", style={"height": f"{CHART_HEIGHT_PX}px"},
                                                              config={"displayModeBar": True}), className="p-2")])], md=12, className="mb-3"),
                ], className="g-3"),

                html.Hr(),
                html.H5("Summary"),
                html.Div(id="summary-metrics", className="alert alert-info"),
                html.Hr(),

                html.H5("Detailed Results"),
                html.H6("Radiators", className="mt-4 mb-3"),
                dash_table.DataTable(
                    id="merged-results-table", page_size=15, style_table={"overflowX": "auto"},
                    style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold', 'textAlign': 'center'},
                    style_cell={'padding':'8px','textAlign':'left','border':'1px solid #dee2e6'},
                    style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}]
                ),
                html.H6("Collectors", className="mt-4 mb-3"),
                dash_table.DataTable(
                    id="collector-results-table", page_size=10, style_table={"overflowX": "auto"},
                    style_header={'backgroundColor':'#f8f9fa','fontWeight':'bold','textAlign':'center'},
                    style_cell={'padding':'8px','textAlign':'left','border':'1px solid #dee2e6'},
                    style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}]
                ),
            ])
        ])
    ]
)

# -----------
# Full app layout
# -----------
app.layout = dbc.Container(
    [
        navbar,
        dcc.Location(id="url", refresh=False),
        *stores,
        dbc.Tabs(id="tabs", active_tab="tab-0", className="justify-content-center",
                 children=[start_tab, heat_tab, rad_tab, results_tab]),
    ],
    fluid=True, style={"backgroundColor": "#f4f6fa", "padding": "24px 0 24px 0"}
)


# ====================
# Helpblok per mode
# ====================
@app.callback(Output("design-mode-help", "children"), Input("design-mode", "value"))
def help_block_for_mode(mode):
    if mode == MODE_EXISTING:
        return "Existing: bereken de benodigde aanvoertemperatuur met de huidige radiatoren. Pas ΔT/diameter aan om LT te benaderen; supply input is hier uitgeschakeld."
    if mode == MODE_FIXED:
        return "LT dimensioning: kies een vaste aanvoertemperatuur; de tool berekent het extra radiatorvermogen per kamer/radiator."
    if mode == MODE_PUMP:
        return "Pump-based: de gekozen pomp (snelheid) bepaalt het haalbare debiet; via ΔT-iteratie wordt een werkpunt gezocht waar pomp- en systeemcurve samenkomen."
    if mode == MODE_BAL:
        return "Balancing: bepaal TRV-standen en debietverdeling (kv-needed & posities) op basis van de berekende massastromen en drukverschillen."
    return ""


# ====================
# Navigation callbacks
# ====================
@app.callback(Output("heat-load-mode-store", "data"),
              Output("tabs", "active_tab"),
              Input("heat-load-mode", "value"))
def choose_mode_and_go(mode):
    if mode in ("known", "unknown"):
        return mode, "tab-1"
    return no_update, no_update

@app.callback(
    Output("manual-loss-card", "style"),
    Output("heat-load-mode-banner", "children"),
    Input("heat-load-mode-store", "data"))
def toggle_manual_ui(mode):
    if mode == "known":
        return {"display": "block"}, "Mode: 🔒 Heat load is KNOWN — enter per-room heat losses below."
    return {"display": "none"}, "Mode: 🧮 Heat load is UNKNOWN — the tool will calculate room heat losses."

@app.callback(
    Output("building-envelope-card", "style"),
    Output("outdoor-conditions-card", "style"),
    Output("building-insulation-card", "style"),
    Output("additional-settings-card", "style"),
    Output("room-config-card", "style"),
    Input("heat-load-mode-store", "data"))
def toggle_known_mode_visibility(mode):
    hidden, shown = {"display": "none"}, {"display": "block"}
    if mode == "known":
        return hidden, hidden, hidden, hidden, hidden
    return shown, shown, shown, shown, shown

# Heat tab: set U-values
@app.callback(
    [Output("uw", "value"), Output("u_roof", "value"), Output("u_ground", "value"), Output("u_glass", "value")],
    [Input("wall_insulation_state", "value"), Input("roof_insulation_state", "value"),
     Input("ground_insulation_state", "value"), Input("glazing_type", "value")],
    [State("uw", "value"), State("u_roof", "value"), State("u_ground", "value"), State("u_glass", "value")]
)
def set_default_u_values(wall_state, roof_state, ground_state, glazing_type, uw, u_roof, u_ground, u_glass):
    triggered = [t["prop_id"] for t in callback_context.triggered]
    if "wall_insulation_state.value" in triggered and wall_state in INSULATION_U_VALUES:
        uw = INSULATION_U_VALUES[wall_state]["wall"]
    if "roof_insulation_state.value" in triggered and roof_state in INSULATION_U_VALUES:
        u_roof = INSULATION_U_VALUES[roof_state]["roof"]
    if "ground_insulation_state.value" in triggered and ground_state in INSULATION_U_VALUES:
        u_ground = INSULATION_U_VALUES[ground_state]["ground"]
    if "glazing_type.value" in triggered and glazing_type in GLAZING_U_VALUES:
        pass  # Typo guard; will be set properly below

    if "glazing_type.value" in triggered and glazing_type in GLAZING_U_VALUES:
        u_glass = GLAZING_U_VALUES[glazing_type]
    return uw, u_roof, u_ground, u_glass

# Build room table
@app.callback(
    Output("room-table", "columns"),
    Output("room-table", "data"),
    Output("room-table", "dropdown"),
    Input("num_rooms", "value"),
    State("room-table", "data"),
    prevent_initial_call=False
)
def build_room_table(num_rooms, existing_data):
    try:
        num_rooms = int(num_rooms) if (num_rooms and int(num_rooms) > 0) else 1
    except Exception:
        num_rooms = 1
    columns = [
        {"name": "Room #", "id": "Room #", "type": "numeric"},
        {"name": "Indoor Temp (°C)", "id": "Indoor Temp (°C)", "type": "numeric", "editable": True, "presentation": "input"},
        {"name": "Floor Area (m²)", "id": "Floor Area (m²)", "type": "numeric"},
        {"name": "Walls external", "id": "Walls external", "type": "numeric", "presentation": "dropdown"},
        {"name": "Room Type", "id": "Room Type", "type": "text", "presentation": "dropdown"},
        {"name": "On Ground", "id": "On Ground", "type": "any", "presentation": "dropdown"},
        {"name": "Under Roof", "id": "Under Roof", "type": "any", "presentation": "dropdown"},
    ]
    if not existing_data:
        data = default_room_table(num_rooms)
    else:
        current_rows = len(existing_data)
        if num_rooms > current_rows:
            data = existing_data.copy()
            for i in range(current_rows + 1, num_rooms + 1):
                data.append({"Room #": i, "Indoor Temp (°C)": 20.0, "Floor Area (m²)": 10.0,
                             "Walls external": 2, "Room Type": "room", "On Ground": False, "Under Roof": False})
        elif num_rooms < current_rows:
            data = existing_data[:num_rooms]
        else:
            data = existing_data
    dropdown = {
        "Room Type": {"options": [{"label": str(v), "value": v} for v in ROOM_TYPE_OPTIONS]},
        "On Ground": {"options": [{"label": "No", "value": False}, {"label": "Yes", "value": True}]},
        "Under Roof": {"options": [{"label": "No", "value": False}, {"label": "Yes", "value": True}]},
        "Walls external": {"options": [{"label": str(v), "value": v} for v in [1, 2, 3, 4]]},
    }
    return columns, data, dropdown

# Manual table
@app.callback(
    Output("manual-loss-table", "data"),
    Input("num_rooms", "value"),
    State("manual-loss-table", "data"),
    prevent_initial_call=False
)
def build_manual_loss_table(num_rooms, current):
    try:
        n = int(num_rooms) if (num_rooms and int(num_rooms) > 0) else 1
    except Exception:
        n = 1
    rows = (current or []).copy()
    by_room = {r.get("Room #"): r for r in rows if "Room #" in r}
    new_rows = []
    for i in range(1, n + 1):
        if i in by_room:
            r = by_room[i]
            r["Room #"] = i
            if "Manual Heat Loss (W)" not in r or r["Manual Heat Loss (W)"] is None:
                r["Manual Heat Loss (W)"] = 0.0
            new_rows.append(r)
        else:
            new_rows.append({"Room #": i, "Manual Heat Loss (W)": 0.0})
    return new_rows

# Compute rooms or manual
@app.callback(
    Output("room-results-store", "data"),
    Output("room-results-table", "children"),
    Input("room-table", "data"),
    Input("uw", "value"), Input("u_roof", "value"), Input("u_ground", "value"), Input("u_glass", "value"),
    Input("tout", "value"),
    Input("ventilation_calculation_method", "value"), Input("v_system", "value"),
    Input("v50", "value"),
    Input("neighbour_t", "value"), Input("un", "value"), Input("lir", "value"),
    Input("wall_height", "value"),
    Input("heat-load-mode-store", "data"), Input("manual-loss-table", "data"),
)
def compute_rooms_and_table(room_rows, uw, u_roof, u_ground, u_glass, tout,
                            vcalc, vsys, v50, neighbour_t, un, lir, wall_height,
                            mode, manual_rows):
    if mode == "known":
        df_manual = pd.DataFrame(manual_rows or [])
        if not df_manual.empty:
            df_manual = df_manual.copy()
            df_manual["Room"] = pd.to_numeric(df_manual.get("Room #", 0), errors="coerce").fillna(0).astype(int)
            df_manual["Total Heat Loss (W)"] = pd.to_numeric(df_manual.get("Manual Heat Loss (W)", 0.0), errors="coerce").fillna(0.0)
            df_manual = df_manual[["Room", "Total Heat Loss (W)"]]
        else:
            df_manual = pd.DataFrame(columns=["Room", "Total Heat Loss (W)"])
        records = df_manual.to_dict("records")
        table = dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in df_manual.columns], data=records, page_size=10,
            style_table={"overflowX": "auto"},
            style_cell={'padding': '8px', 'border': '1px solid #dee2e6'},
            style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold', 'textAlign': 'center'},
        )
        return records, table

    # Validate room input
    for row in (room_rows or []):
        if "Indoor Temp (°C)" in row:
            try:
                t = float(row["Indoor Temp (°C)"])
                if t < 10:
                    row["Indoor Temp (°C)"] = 10.0
                elif t > 24:
                    row["Indoor Temp (°C)"] = 24.0
            except Exception:
                row["Indoor Temp (°C)"] = 20.0
        if "Walls external" in row:
            try:
                w = int(row["Walls external"])
                if w not in [1, 2, 3, 4]:
                    row["Walls external"] = 2
            except Exception:
                row["Walls external"] = 2

    df_results = compute_room_results(
        room_rows=room_rows,
        uw=safe_to_float(uw, 1.0) or 1.0, u_roof=safe_to_float(u_roof, 0.2) or 0.2,
        u_ground=safe_to_float(u_ground, 0.3) or 0.3, u_glass=safe_to_float(u_glass, 2.8) or 2.8,
        tout=safe_to_float(tout) if tout is not None else -7.0,
        heat_loss_area_estimation="fromFloorArea",
        ventilation_calculation_method=vcalc or "simple",
        v_system=vsys or "C",
        v50=safe_to_float(v50, 6.0) or 6.0, neighbour_t=safe_to_float(neighbour_t, 18.0) or 18.0,
        un=safe_to_float(un, 1.0) or 1.0, lir=safe_to_float(lir, 0.2) or 0.2,
        wall_height=safe_to_float(wall_height, 2.7) or 2.7,
        return_detail=False, add_neighbour_losses=True
    )
    records = df_results.to_dict("records")
    table = dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in df_results.columns], data=records, page_size=10,
        style_table={"overflowX": "auto"},
        style_cell={'padding': '8px', 'border': '1px solid #dee2e6'},
        style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold', 'textAlign': 'center'},
    )
    return records, table


# ==========================
# Tab 2 — UI visibility
# ==========================
@app.callback(Output("fixed_diameter_container", "style"), Input("fix_diameter", "value"))
def toggle_fixed_diameter(checklist):
    return {"display": "block"} if "yes" in (checklist or []) else {"display": "none"}

# Disable supply input in modes where it is irrelevant
@app.callback(Output("supply_temp_input", "disabled"), Input("design-mode", "value"))
def disable_supply_input(mode):
    return mode in (MODE_EXISTING, MODE_PUMP, MODE_BAL)

# Toggle Pump settings UI
@app.callback(Output("pump-settings-card", "style"), Input("design-mode", "value"))
def toggle_pump_ui(mode):
    return {"display": "block"} if mode == MODE_PUMP else {"display": "none"}


# ==========================
# Config store
# ==========================
@app.callback(
    Output("config-store", "data"),
    Input("num_radiators", "value"), Input("num_collectors", "value"),
    Input("positions", "value"), Input("kv_max", "value"),
    Input("delta_T", "value"), Input("supply_temp_input", "value"),
    Input("fix_diameter", "value"), Input("fixed_diameter", "value"),
    Input("design-mode", "value"),
    Input("pump_model", "value"), Input("pump_speed", "value"),
)
def update_config(nr, nc, pos, kvmax, dT, tsupply, fixlist, fixed_mm, design_mode, pump_model, pump_speed):
    # Alleen in fixed gebruiken we supply; zo niet → None
    supply_for_cfg = safe_to_float(tsupply, None) if (design_mode == MODE_FIXED) else None
    # Default 55°C voor fixed als leeg
    if design_mode == MODE_FIXED and supply_for_cfg is None:
        supply_for_cfg = 55.0

    cfg = {
        "num_radiators": int(nr or 1),
        "num_collectors": int(nc or 1),
        "positions": int(pos or 8),
        "kv_max": safe_to_float(kvmax, 0.7) or 0.7,
        "delta_T": int(dT or 10),
        "supply_temp_input": supply_for_cfg,
        "fix_diameter": ("yes" in (fixlist or [])),
        "fixed_diameter": int(fixed_mm or 16) if ("yes" in (fixlist or [])) else None,
        "design_mode": design_mode or MODE_EXISTING,
        "pump_model": pump_model or "Grundfos UPM3 15-70",
        "pump_speed": pump_speed or "speed_2",
    }
    return cfg


# ================
# Table builders
# ================
@app.callback(
    Output("radiator-data-store", "data"),
    Output("collector-data-store", "data"),
    Output("radiator-table", "data"),
    Output("radiator-table", "dropdown"),
    Output("collector-table", "data"),
    Input("config-store", "data"),
    Input("room-results-store", "data"),
    State("radiator-data-store", "data"),
    State("collector-data-store", "data"),
)
def ensure_tables_and_dropdowns(cfg, room_results_records, radiator_store, collector_store):
    cfg = cfg or {}
    num_radiators = int(cfg.get("num_radiators", 3))
    num_collectors = int(cfg.get("num_collectors", 1))

    room_df = pd.DataFrame(room_results_records or [])
    room_options = sorted(room_df["Room"].unique().tolist()) if not room_df.empty else [1, 2, 3]
    collector_options = [f"Collector {i+1}" for i in range(num_collectors)]

    current_rows = radiator_store or []
    if not current_rows:
        current_rows = init_radiator_rows(num_radiators, collector_options, room_options)
    else:
        current_rows = resize_radiator_rows(current_rows, num_radiators, collector_options, room_options)

    current_collectors = collector_store or []
    if not current_collectors:
        current_collectors = init_collector_rows(num_collectors)
    else:
        current_collectors = resize_collector_rows(current_collectors, num_collectors)

    dropdown = {
        "Room": {"options": [{"label": str(o), "value": o} for o in room_options]},
        "Collector": {"options": [{"label": c, "value": c} for c in collector_options]}
    }
    return current_rows, current_collectors, current_rows, dropdown, current_collectors

@app.callback(Output("radiator-data-store", "data", allow_duplicate=True), Input("radiator-table", "data"), prevent_initial_call=True)
def write_radiator_edits_back(rows): return rows or []

@app.callback(Output("collector-data-store", "data", allow_duplicate=True), Input("collector-table", "data"), prevent_initial_call=True)
def write_collector_edits_back(rows): return rows or []

@app.callback(
    Output("heat-loss-split-store", "data"),
    Output("heat-loss-split-table", "data"),
    Input("radiator-data-store", "data"),
    Input("room-results-store", "data"),
)
def recompute_heat_loss_split(radiator_rows, room_results_records):
    rad_rows = radiator_rows or []
    room_df = pd.DataFrame(room_results_records or [])
    split_df = split_heat_loss_to_radiators(rad_rows, room_df)
    return split_df.to_dict("records"), split_df.to_dict("records")


# ==========================
# Tab 3 — Core computation
# ==========================
@app.callback(
    Output("results-warnings", "children"),
    Output("merged-results-table", "columns"),
    Output("merged-results-table", "data"),
    Output("collector-results-table", "columns"),
    Output("collector-results-table", "data"),
    Output("pressure-loss-chart", "figure"),
    Output("mass-flow-chart", "figure"),
    Output("valve-position-chart", "figure"),
    Output("summary-metrics", "children"),
    Output("power-distribution-chart", "figure"),
    Output("temperature-profile-chart", "figure"),
    Output("pump-curve-chart", "figure"),
    Output("metric-total-heat-loss", "children"),
    Output("metric-total-power", "children"),
    Output("metric-flow-rate", "children"),
    Output("metric-delta-t", "children"),
    Output("metric-highest-supply", "children"),
    Input("radiator-data-store", "data"),
    Input("collector-data-store", "data"),
    Input("heat-loss-split-store", "data"),
    Input("config-store", "data"),
    Input("room-table", "data"),
)
def compute_results(radiator_rows, collector_rows, split_rows, cfg, room_rows):
    warnings = []
    empty_block = (html.Div("⚠️ Compute heat loss and configure radiators/collectors in Tab 1 & 2 first."),
                   [], [], [], [],
                   empty_fig(""), empty_fig(""), empty_fig(""), "",
                   empty_fig(""), empty_fig(""),
                   empty_fig(""),
                   "0 W", "0 W", "0 kg/h", "0 °C", "0 °C")
    if not radiator_rows or not collector_rows or not split_rows:
        return empty_block

    # Merge & inputs
    rad_df = pd.DataFrame(radiator_rows)
    col_df = pd.DataFrame(collector_rows)
    split_df = pd.DataFrame(split_rows).rename(columns={"Calculated Heat Loss (W)": "Calculated heat loss"})
    if "Radiator nr" not in rad_df.columns or "Radiator nr" not in split_df.columns:
        warnings.append("Radiator nr column missing unexpectedly.")
        return empty_block

    rad_df = rad_df.merge(split_df[["Radiator nr", "Calculated heat loss"]], on="Radiator nr", how="left")

    # Merge room temps to get Space Temperature
    if room_rows:
        room_temp_df = pd.DataFrame(room_rows)
        if "Room #" in room_temp_df.columns and "Indoor Temp (°C)" in room_temp_df.columns:
            room_temp_df = room_temp_df[["Room #", "Indoor Temp (°C)"]].rename(columns={"Room #": "Room"})
            rad_df = rad_df.merge(room_temp_df, on="Room", how="left")
    rad_df["Space Temperature"] = rad_df["Indoor Temp (°C)"].fillna(rad_df.get("Space Temperature", 20.0)).fillna(20.0)

    for c in ["Radiator power 75/65/20", "Calculated heat loss", "Length circuit", "Space Temperature", "Electric power"]:
        if c in rad_df.columns:
            rad_df[c] = pd.to_numeric(rad_df[c], errors="coerce")

    design_mode = (cfg or {}).get("design_mode", MODE_EXISTING)
    delta_T = int(cfg.get("delta_T", 10))
    tsupply_user = cfg.get("supply_temp_input", None)  # only set in fixed
    calc_rows: List[Radiator] = []

    # Flag om hydraulica-blok te skippen wanneer MODE_PUMP al alles berekend heeft
    hydraulics_done = False
    merged_df = None  # type: Optional[pd.DataFrame]

    # -------------------------
    # Mode-specific calculations
    # -------------------------
    def simulate_system_for_deltaT(delta_T_for_run: int) -> Tuple[pd.DataFrame, pd.DataFrame, List[Radiator],
    float, float, Optional[float], Optional[float], List[str]]:
        """
        Simulate the system for a given ΔT:
        - Build calc_rows (Radiator) with q_ratio as 'existing'
        - Determine supply/return/flow rates
        - Calculate pressure losses and totals (radiators + collectors)
        - Derive K_sys from current operating point
        - Find intersection with pump (Q*, ΔP*)
        Returns:
            merged_df_local, col_df_local, calc_rows_local, Q_total, K_sys, q_star, dp_star, warnings_local
        """
        warnings_local: List[str] = []

        # Radiator calculation (no extra power in pump-mode)
        calc_rows_local: List[Radiator] = []
        rad_df_local = rad_df.copy()
        rad_df_local["Extra radiator power"] = 0.0

        for _, row in rad_df_local.iterrows():
            base = row.get("Radiator power 75/65/20", 0.0) or 0.0
            heat_loss_i = row.get("Calculated heat loss", 0.0) or 0.0
            extra_elec = row.get("Electric power", 0.0) or 0.0
            want = max(heat_loss_i - extra_elec, 0.0)
            q_ratio = (want / base) if base else 0.0

            r = Radiator(
                q_ratio=q_ratio,
                delta_t=delta_T_for_run,
                space_temperature=row.get("Space Temperature", 20.0) or 20.0,
                heat_loss=heat_loss_i
            )
            calc_rows_local.append(r)

        # Supply/return/flow rates
        supply_T = determine_system_supply_temperature(calc_rows_local, cfg)  # fallback 55°C if needed
        for r in calc_rows_local:
            r.supply_temperature = supply_T
            r.return_temperature = r.calculate_treturn(supply_T)
            r.mass_flow_rate = r.calculate_mass_flow_rate()

        rad_df_local["Supply Temperature"] = supply_T
        rad_df_local["Return Temperature"] = [r.return_temperature for r in calc_rows_local]
        rad_df_local["Mass flow rate"] = [r.mass_flow_rate for r in calc_rows_local]

        # Diameter (uniform if fix_diameter; else calculate and optionally take max() to keep uniform)
        if cfg.get("fix_diameter"):
            rad_df_local["Diameter"] = cfg.get("fixed_diameter", 16)
        else:
            rad_numbers = (
                rad_df_local["Radiator nr"].tolist()
                if "Radiator nr" in rad_df_local.columns
                else list(range(1, len(calc_rows_local) + 1))
            )
            rad_diams = []
            for idx, radiator in enumerate(calc_rows_local):
                label = f"Radiator {rad_numbers[idx] if idx < len(rad_numbers) else (idx + 1)}"
                d = capture_value_errors(
                    lambda: radiator.calculate_diameter(POSSIBLE_DIAMETERS),
                    label,
                    warnings_local,
                    fallback=max(POSSIBLE_DIAMETERS),
                )
                rad_diams.append(d)
            rad_df_local["Diameter"] = max(rad_diams) if rad_diams else 16

        # Pressure loss per radiator
        press_losses = []
        for _, row_i in rad_df_local.iterrows():
            label = f"Radiator {row_i.get('Radiator nr', '?')}"
            pl = capture_value_errors(
                lambda r_i=row_i: Circuit(
                    length_circuit=r_i.get("Length circuit", 0.0) or 0.0,
                    diameter=r_i.get("Diameter", 16) or 16,
                    mass_flow_rate=r_i.get("Mass flow rate", 0.0) or 0.0
                ).calculate_pressure_radiator_kv(),
                label, warnings_local, fallback=float("nan")
            )
            press_losses.append(pl)
        rad_df_local["Pressure loss"] = press_losses

        # Collectors
        col_df_local = col_df.copy()
        collector_options = col_df_local["Collector"].tolist()
        collectors = [Collector(name=name) for name in collector_options]
        for c in collectors:
            c.update_mass_flow_rate(rad_df_local)
        col_df_local["Mass flow rate"] = [c.mass_flow_rate for c in collectors]

        col_diams = []
        for idx, collector in enumerate(collectors):
            cname = (
                collector_options[idx]
                if idx < len(collector_options)
                else f"Collector {idx + 1}"
            )
            d = capture_value_errors(
                lambda: collector.calculate_diameter(POSSIBLE_DIAMETERS),
                cname,
                warnings_local,
                fallback=max(POSSIBLE_DIAMETERS),
            )
            col_diams.append(d)
        col_df_local["Diameter"] = col_diams

        col_press = []
        for idx, (_, row) in enumerate(col_df_local.iterrows()):
            cname = row.get("Collector", f"Collector {idx + 1}")
            pl = capture_value_errors(
                lambda: Circuit(
                    length_circuit=row.get("Collector circuit length", 0.0) or 0.0,
                    diameter=row.get("Diameter", 16) or 16,
                    mass_flow_rate=row.get("Mass flow rate", 0.0) or 0.0,
                ).calculate_pressure_collector_kv(),
                cname,
                warnings_local,
                fallback=float("nan"),
            )
            col_press.append(pl)
        col_df_local["Collector pressure loss"] = col_press

        # Total pressure per radiator
        merged_df_local = Collector(name="").calculate_total_pressure_loss(rad_df_local, col_df_local)

        # Derive system curve constant K_sys
        Q_total = float(
            pd.to_numeric(merged_df_local.get("Mass flow rate", pd.Series()), errors="coerce").fillna(0).sum())
        branch_kpa = float(
            pd.to_numeric(merged_df_local.get("Total Pressure Loss", pd.Series()), errors="coerce").fillna(
                0).max() / 1000)
        K_sys = derive_system_curve_kpa(Q_total, branch_kpa)

        # Get pump curve and find operating point
        pump_name = cfg.get("pump_model", "Grundfos UPM3 15-70")
        pump_speed = cfg.get("pump_speed", "speed_2")
        pump_points = (PUMP_LIBRARY.get(pump_name, {}) or {}).get(pump_speed, [])
        q_star, dp_star, pump_warn = (None, None, [])
        if pump_points:
            q_min, q_max = min(p[0] for p in pump_points), max(p[0] for p in pump_points)
            Q_grid = np.linspace(q_min, q_max, 150)
            pump_kpa = interpolate_curve(pump_points, Q_grid)
            q_star, dp_star, pump_warn = find_operating_point(Q_grid, pump_kpa, K_sys)
        warnings_local.extend(pump_warn)

        return merged_df_local, col_df_local, calc_rows_local, Q_total, K_sys, q_star, dp_star, warnings_local

    # --- Pump mode: find minimal ΔT where Q_tot(ΔT) ≤ Q_pump ---
    if design_mode == MODE_PUMP:
        dt_low, dt_high = 0.1, max(20, int(cfg.get("delta_T", 10)))
        best_solution = None
        pump_iter_warnings: List[str] = []

        # Check feasibility at high ΔT
        merged_hi, col_hi, calc_hi, Q_hi, K_hi, qstar_hi, dp_hi, warn_hi = simulate_system_for_deltaT(dt_high)
        pump_iter_warnings.extend(warn_hi)
        feasible_hi = (qstar_hi is not None) and (Q_hi <= qstar_hi)

        if not feasible_hi:
            warnings.append(
                "Pump mode: The selected pump/speed provides insufficient flow for the system, even at high ΔT. "
                "Consider a larger pump, higher speed, or adjust piping/radiators."
            )
            merged_df, col_df, calc_rows = merged_hi, col_hi, calc_hi
            hydraulics_done = True
        else:
            # Check feasibility at low ΔT
            merged_lo, col_lo, calc_lo, Q_lo, K_lo, qstar_lo, dp_lo, warn_lo = simulate_system_for_deltaT(dt_low)
            pump_iter_warnings.extend(warn_lo)

            if (qstar_lo is not None) and (Q_lo <= qstar_lo):
                best_solution = (merged_lo, col_lo, calc_lo, Q_lo, K_lo, qstar_lo, dp_lo, dt_low)
            else:
                # Bisection on ΔT with tolerance-based stopping
                l, r = dt_low, dt_high
                best_solution = (merged_hi, col_hi, calc_hi, Q_hi, K_hi, qstar_hi, dp_hi, dt_high)
                tolerance = 0.1  # Stop when ΔT changes by less than 0.1°C
                for _ in range(50):  # Max iterations as safety
                    m = (l + r) / 2
                    merged_m, col_m, calc_m, Q_m, K_m, qstar_m, dp_m, warn_m = simulate_system_for_deltaT(m)
                    pump_iter_warnings.extend(warn_m)
                    feasible_m = (qstar_m is not None) and (Q_m <= qstar_m)
                    if feasible_m:
                        best_solution = (merged_m, col_m, calc_m, Q_m, K_m, qstar_m, dp_m, m)
                        r = m
                    else:
                        l = m
                    if (r - l) < tolerance:
                        break

            # Set best solution and mark hydraulics as done
            merged_df, col_df, calc_rows, Q_best, K_best, qstar_best, dp_best, dt_best = best_solution
            warnings.extend(pump_iter_warnings)
            warnings.append(
                f"Pump mode: Operating point found at ΔT={dt_best:.1f} °C "
                f"with Q_tot≈{Q_best:.0f} kg/h and pump Q*≈{(qstar_best or 0):.0f} kg/h."
            )
            hydraulics_done = True


    elif design_mode in (MODE_EXISTING, MODE_BAL):
        # --- Existing/balancing: standaard pad ---
        rad_df["Extra radiator power"] = 0.0
        calc_rows = []
        for _, row in rad_df.iterrows():
            base = row.get("Radiator power 75/65/20", 0.0) or 0.0
            heat_loss = row.get("Calculated heat loss", 0.0) or 0.0
            extra_elec = row.get("Electric power", 0.0) or 0.0
            want = max(heat_loss - extra_elec, 0.0)
            q_ratio = (want / base) if base else 0.0

            radiator = Radiator(
                q_ratio=q_ratio,
                delta_t=delta_T,
                space_temperature=row.get("Space Temperature", 20.0) or 20.0,
                heat_loss=heat_loss
            )
            calc_rows.append(radiator)

        max_supply_temperature = determine_system_supply_temperature(calc_rows, cfg)
        for r in calc_rows:
            r.supply_temperature = max_supply_temperature
            r.return_temperature = r.calculate_treturn(max_supply_temperature)
            r.mass_flow_rate = r.calculate_mass_flow_rate()

        rad_df["Supply Temperature"] = max_supply_temperature
        rad_df["Return Temperature"] = [r.return_temperature for r in calc_rows]
        rad_df["Mass flow rate"]     = [r.mass_flow_rate for r in calc_rows]

        if design_mode != MODE_EXISTING and tsupply_user is not None:
            warnings.append("In deze modus wordt de door jou ingevoerde Supply temperature genegeerd.")

    elif design_mode == MODE_FIXED:
        # --- Fixed (ongewijzigd) ---
        if tsupply_user is None:
            warn = html.Div([
                html.Div("❌ Error during calculation:", className="fw-bold"),
                html.Div("In modus 'LT dimensioning (fixed)' moet je een Supply temperature invullen in Tab 2.")
            ], className="alert alert-danger")
            return (warn, [], [], [], [],
                    empty_fig(""), empty_fig(""), empty_fig(""), "",
                    empty_fig(""), empty_fig(""),
                    empty_fig(""),
                    "0 W", "0 W", "0 kg/h", "0 °C", "0 °C")

        extra_list = []
        #calc_rows = []
        for _, row in rad_df.iterrows():
            base = row.get("Radiator power 75/65/20", 0.0) or 0.0
            heat_loss = row.get("Calculated heat loss", 0.0) or 0.0
            space_T   = row.get("Space Temperature", 20.0) or 20.0
            extra_norm = calculate_extra_power_needed(
                radiator_power=base,
                heat_loss=heat_loss,
                supply_temp=float(tsupply_user),
                delta_t=delta_T,
                space_temperature=space_T
            )
            extra_list.append(extra_norm)
        rad_df["Extra radiator power"] = extra_list

        for _, row in rad_df.iterrows():
            base       = row.get("Radiator power 75/65/20", 0.0) or 0.0
            heat_loss  = row.get("Calculated heat loss", 0.0) or 0.0
            extra_elec = row.get("Electric power", 0.0) or 0.0
            extra_norm = row.get("Extra radiator power", 0.0) or 0.0
            want = max(heat_loss - extra_elec, 0.0)
            q_ratio = (want / (base+extra_norm)) if base else 0.0
            r = Radiator(
                q_ratio=q_ratio,
                delta_t=delta_T,
                space_temperature=row.get("Space Temperature", 20.0) or 20.0,
                heat_loss=heat_loss + extra_norm
            )
            r.supply_temperature = float(tsupply_user)
            r.return_temperature = r.calculate_treturn(float(tsupply_user))
            r.mass_flow_rate     = r.calculate_mass_flow_rate()
            calc_rows.append(r)
        rad_df["Supply Temperature"] = float(tsupply_user)
        rad_df["Return Temperature"] = [r.return_temperature for r in calc_rows]
        rad_df["Mass flow rate"]     = [r.mass_flow_rate for r in calc_rows]

    # ---------------------------------------
    # Diameter & drukverlies (als nog niet gedaan door MODE_PUMP)
    # ---------------------------------------
    if not hydraulics_done:
        if cfg.get("fix_diameter"):
            rad_df["Diameter"] = cfg.get("fixed_diameter", 16)
        else:
            rad_numbers = (
                rad_df["Radiator nr"].tolist()
                if "Radiator nr" in rad_df.columns
                else list(range(1, len(calc_rows) + 1))
            )
            rad_diams = []
            for idx, radiator in enumerate(calc_rows):
                label = f"Radiator {rad_numbers[idx] if idx < len(rad_numbers) else (idx + 1)}"
                d = capture_value_errors(
                    lambda: radiator.calculate_diameter(POSSIBLE_DIAMETERS),
                    label,
                    warnings,
                    fallback=max(POSSIBLE_DIAMETERS),
                )
                rad_diams.append(d)
            # Uniforme diameter zoals jouw huidige gedrag:
            rad_df["Diameter"] = max(rad_diams) if rad_diams else 16

        press_losses = []
        for _, row in rad_df.iterrows():
            label = f"Radiator {row.get('Radiator nr', '?')}"
            pl = capture_value_errors(
                lambda: Circuit(
                    length_circuit=row.get("Length circuit", 0.0) or 0.0,
                    diameter=row.get("Diameter", 16) or 16,
                    mass_flow_rate=row.get("Mass flow rate", 0.0) or 0.0,
                ).calculate_pressure_radiator_kv(),
                label,
                warnings,
                fallback=float("nan"),
            )
            press_losses.append(pl)
        rad_df["Pressure loss"] = press_losses

        # Collectors
        collector_options = col_df["Collector"].tolist()
        collectors = [Collector(name=name) for name in collector_options]
        for c in collectors:
            c.update_mass_flow_rate(rad_df)
        col_df["Mass flow rate"] = [c.mass_flow_rate for c in collectors]
        col_diams = []
        for idx, collector in enumerate(collectors):
            cname = collector_options[idx] if idx < len(collector_options) else f"Collector {idx + 1}"

            d = capture_value_errors(
                lambda: collector.calculate_diameter(POSSIBLE_DIAMETERS),
                cname,
                warnings,
                fallback=max(POSSIBLE_DIAMETERS),
            )

            col_diams.append(d)
        col_df["Diameter"] = col_diams
        col_press = []
        for idx, (_, row) in enumerate(col_df.iterrows()):
            cname = row.get("Collector", f"Collector {idx + 1}")
            pl = capture_value_errors(
                lambda: Circuit(
                    length_circuit=row.get("Collector circuit length", 0.0) or 0.0,
                    diameter=row.get("Diameter", 16) or 16,
                    mass_flow_rate=row.get("Mass flow rate", 0.0) or 0.0,
                ).calculate_pressure_collector_kv(),
                cname,
                warnings,
                fallback=float("nan"),
            )
            col_press.append(pl)
        col_df["Collector pressure loss"] = col_press
        merged_df = Collector(name="").calculate_total_pressure_loss(rad_df, col_df)
    else:
        # MODE_PUMP had already produced merged_df/col_df and calc_rows
        # Zorg dat rad_df gelijkloopt met merged_df voor downstream charts
        rad_df = merged_df.copy()

    # Highest supply metric
    try:
        max_return_idx = merged_df['Return Temperature'].idxmax()
        radiator_nr = merged_df.loc[max_return_idx, 'Radiator nr']
        supply_temp = merged_df.loc[max_return_idx, 'Supply Temperature']
        metric_highest_supply = f"{float(supply_temp):.1f} °C - Radiator {radiator_nr}"
    except Exception:
        metric_highest_supply = "N/A"

    # Valve config
    valve_type = cfg.get("valve_type", "Custom")
    if valve_type == "Custom":
        kv_max = float(cfg.get("kv_max", 0.7) or 0.7)
        n_positions = int(cfg.get("positions", 8) or 8)
        valve = Valve(kv_max=kv_max, n=n_positions, valve_name="Custom")
    else:
        valve = Valve(valve_name=valve_type)
        config = valve.get_config()
        if config:
            kv_max = config["kv_values"][-1]
            n_positions = config["positions"]
        else:
            kv_max, n_positions = 0.7, 8
            valve = Valve(kv_max=kv_max, n=n_positions, valve_name="Custom")

    # Valve losses & posities
    merged_df["Valve pressure loss N"] = merged_df["Mass flow rate"].apply(valve.calculate_pressure_valve_kv)
    merged_df = valve.calculate_kv_position_valve(merged_df) if valve_type != "Custom" else valve.calculate_kv_position_valve(
        merged_df, custom_kv_max=kv_max, n=n_positions
    )

    merged_cols = [{"name": c, "id": c} for c in merged_df.columns]
    merged_data = merged_df.to_dict("records")

    collector_cols = [{"name": c, "id": c} for c in col_df.columns]
    collector_data = col_df.to_dict("records")

    # Metrics
    weighted_delta_t = calculate_weighted_delta_t(calc_rows, merged_df)
    total_mass_flow_rate = float(pd.to_numeric(merged_df.get("Mass flow rate", pd.Series()), errors="coerce").fillna(0).sum())
    total_heat_loss = merged_df.get("Calculated heat loss", pd.Series()).fillna(0).sum()
    total_power = merged_df.get("Radiator power 75/65/20", pd.Series()).fillna(0).sum()
    sum_extra = float(pd.to_numeric(merged_df.get("Extra radiator power", pd.Series()), errors="coerce").fillna(0).sum())

    # Charts
    if "Radiator power 75/65/20" in merged_df.columns and "Calculated heat loss" in merged_df.columns:
        fig_power = go.Figure()
        fig_power.add_trace(go.Bar(x=merged_df["Radiator nr"], y=merged_df["Radiator power 75/65/20"], name='Radiator Power', marker_color='#3498db'))
        fig_power.add_trace(go.Bar(x=merged_df["Radiator nr"], y=merged_df["Calculated heat loss"], name='Required Power', marker_color='#e74c3c'))
        if "Extra radiator power" in merged_df.columns:
            fig_power.add_trace(go.Bar(x=merged_df["Radiator nr"], y=merged_df["Extra radiator power"].fillna(0), name='Extra radiator power', marker_color='#9b59b6'))
        fig_power.update_layout(barmode='group')
        fig_power = fix_fig(fig_power, "Radiator Power vs Required Heat Loss (+ Extra)")
    else:
        fig_power = empty_fig("Power distribution data not available")

    if {"Supply Temperature", "Return Temperature", "Space Temperature"}.issubset(merged_df.columns):
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(x=merged_df["Radiator nr"], y=merged_df["Supply Temperature"], mode='lines+markers', name='Supply',
                                      line=dict(color='#e74c3c', width=3), marker=dict(size=8)))
        fig_temp.add_trace(go.Scatter(x=merged_df["Radiator nr"], y=merged_df["Return Temperature"], mode='lines+markers', name='Return',
                                      line=dict(color='#3498db', width=3), marker=dict(size=8)))
        fig_temp.add_trace(go.Scatter(x=merged_df["Radiator nr"], y=merged_df["Space Temperature"], mode='lines+markers', name='Space',
                                      line=dict(color='#27ae60', width=2, dash='dash'), marker=dict(size=6)))
        fig_temp = fix_fig(fig_temp, "Temperature Profile")
    else:
        fig_temp = empty_fig("Temperature data not available")

    if "Total Pressure Loss" in merged_df.columns:
        fig_pressure = px.bar(merged_df, x="Radiator nr", y="Total Pressure Loss", color="Total Pressure Loss", color_continuous_scale="Viridis")
        fig_pressure = fix_fig(fig_pressure, "Total Pressure Loss per Radiator")
    else:
        fig_pressure = empty_fig("Total Pressure Loss (column not found)")

    if "Mass flow rate" in merged_df.columns:
        fig_mass = px.bar(merged_df, x="Radiator nr", y="Mass flow rate", color="Mass flow rate", color_continuous_scale="Blues")
        fig_mass = fix_fig(fig_mass, "Mass Flow Rate per Radiator")
    else:
        fig_mass = empty_fig("Mass Flow Rate (column not found)")

    valve_y = next((c for c in ["Valve position", "kv_position", "Valve pressure loss N"] if c in merged_df.columns), None)
    if valve_y:
        fig_valve = px.bar(merged_df, x="Radiator nr", y=valve_y, color=valve_y, color_continuous_scale="RdYlGn_r")
        fig_valve = fix_fig(fig_valve, f"Valve Analysis: {valve_y}")
    else:
        fig_valve = empty_fig("Valve (column not found)")

    # Pump curve chart (alleen zinvol in MODE_PUMP)
    fig_pump = empty_fig("Pump curve (pump mode)", height=CHART_HEIGHT_PX)
    if design_mode == MODE_PUMP:
        pump_name = cfg.get("pump_model", "Grundfos UPM3 15-70")
        pump_speed = cfg.get("pump_speed", "speed_2")
        pump_points = (PUMP_LIBRARY.get(pump_name, {}) or {}).get(pump_speed, [])
        if pump_points:
            Q_total = float(pd.to_numeric(merged_df.get("Mass flow rate", pd.Series()), errors="coerce").fillna(0).sum())
            branch_kpa = float(pd.to_numeric(merged_df.get("Total Pressure Loss", pd.Series()), errors="coerce").fillna(0).max()/1000)
            K_sys = derive_system_curve_kpa(Q_total, branch_kpa)
            q_min, q_max = min(p[0] for p in pump_points), max(p[0] for p in pump_points)
            Q_grid = np.linspace(q_min, q_max, 150)
            pump_kpa = interpolate_curve(pump_points, Q_grid)
            q_star, dp_star, pump_warn = find_operating_point(Q_grid, pump_kpa, K_sys)
            warnings.extend(pump_warn)
            fig_pump = go.Figure()
            fig_pump.add_trace(go.Scatter(x=Q_grid, y=pump_kpa, mode='lines', name=f"{pump_name} ({pump_speed})",
                                          line=dict(color='#1f77b4', width=3)))
            sys_kpa = K_sys * (Q_grid ** 2)
            fig_pump.add_trace(go.Scatter(x=Q_grid, y=sys_kpa, mode='lines', name="System curve (K·Q²)",
                                          line=dict(color='#e74c3c', width=3, dash='dash')))
            if q_star is not None and dp_star is not None:
                fig_pump.add_trace(go.Scatter(x=[q_star], y=[dp_star], mode='markers+text',
                                              name="Operating point",
                                              marker=dict(color='#2ca02c', size=10),
                                              text=[f"Q={q_star:.0f} kg/h, ΔP={dp_star:.1f} kPa"],
                                              textposition="top center"))
            fig_pump = fix_fig(fig_pump, f"Pump vs System — {pump_name} ({pump_speed})")

    metric_total_heat_loss = f"{total_heat_loss:.0f} W" if total_heat_loss > 0 else "0 W"
    metric_total_power = f"{total_power:.0f} W" if total_power > 0 else "0 W"
    metric_flow_rate = f"{total_mass_flow_rate:.2f} kg/h" if total_mass_flow_rate > 0 else "0 kg/h"
    metric_delta_t = f"{weighted_delta_t:.2f} °C" if weighted_delta_t > 0 else "0 °C"

    summary = html.Ul([
        html.Li(f"Mode: {design_mode}"),
        html.Li(f"Weighted ΔT: {weighted_delta_t:.2f} °C"),
        html.Li(f"Total Mass Flow Rate: {total_mass_flow_rate:.2f} kg/h"),
        html.Li(f"Total Heat Loss: {total_heat_loss:.0f} W"),
        html.Li(f"Total Radiator Power: {total_power:.0f} W"),
        html.Li(f"Som extra vermogen (genormaliseerd): {sum_extra:.0f} W"),
        html.Li(f"Highest supply T: {metric_highest_supply}"),
        html.Li(f"Radiators: {len(rad_df)} — Collectors: {len(col_df)}"),
    ])

    warn_div = (
        html.Div([html.Div("⚠️ Warnings", className="fw-bold mb-1")] + [html.Div(w) for w in warnings], className="alert alert-warning")
        if warnings else html.Div()
    )

    return (
        warn_div, merged_cols, merged_data, collector_cols, collector_data,
        fig_pressure, fig_mass, fig_valve, summary,
        fig_power, fig_temp, fig_pump,
        metric_total_heat_loss, metric_total_power, metric_flow_rate, metric_delta_t, metric_highest_supply
    )


# ==========================
# Valve selection callbacks
# ==========================
@app.callback(
    [Output("valve-specs", "children"), Output("valve-custom-settings", "style")],
    [Input("valve-type-dropdown", "value")]
)
def update_valve_info(selected_valve):
    if selected_valve == "Custom":
        return "", {"display": "block"}
    try:
        valve = Valve(valve_name=selected_valve)
        config = valve.get_config()
        if config:
            specs = [html.Div(f"Positions: {config['positions']}"),
                     html.Div(f"Kv Range: {config['kv_values'][0]} - {config['kv_values'][-1]} m³/h")]
            return specs, {"display": "none"}
    except Exception as e:
        print(f"Error getting valve config: {e}")
    return "Error loading valve specs", {"display": "block"}

@app.callback(
    [Output("positions", "value"),
     Output("kv_max", "value")],
    [Input("valve-type-dropdown", "value")]
)
def update_valve_defaults(selected_valve):
    if selected_valve == "Custom":
        return no_update, no_update
    try:
        valve = Valve(valve_name=selected_valve)
        config = valve.get_config()
        if config:
            return config["positions"], config["kv_values"][-1]
    except Exception as e:
        print(f"Error updating valve defaults: {e}")
    return no_update, no_update

@app.callback(
    Output("config-store", "data", allow_duplicate=True),
    [Input("valve-type-dropdown", "value"),
     Input("positions", "value"),
     Input("kv_max", "value")],
    [State("config-store", "data")],
    prevent_initial_call=True
)
def update_valve_config(selected_valve, positions, kv_max, current_cfg):
    if current_cfg is None:
        current_cfg = {}
    if selected_valve == "Custom":
        current_cfg["valve_type"] = "Custom"
        current_cfg["positions"] = positions
        current_cfg["kv_max"] = kv_max
    else:
        try:
            valve = Valve(valve_name=selected_valve)
            config = valve.get_config()
            if config:
                current_cfg["valve_type"] = selected_valve
                current_cfg["positions"] = config["positions"]
                current_cfg["kv_max"] = config["kv_values"][-1]
        except Exception as e:
            print(f"Error updating valve defaults: {e}")
    return current_cfg



