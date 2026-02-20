"""
ui/callbacks/hydraulics.py
==========================
Callbacks for Tab 2 & 3: config store, table management,
heat-loss splitting, and the main hydraulic computation.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, html, no_update

from config import MODE_EXISTING, MODE_FIXED, MODE_PUMP, MODE_BAL, PUMP_LIBRARY, CHART_HEIGHT_PX
from domain.radiator import POSSIBLE_DIAMETERS, Radiator
from domain.hydraulics import Circuit, Collector, check_pipe_velocities, HYDRAULIC_CONSTANT
from domain.valve import Valve
from domain.valve_override import recalculate_with_overrides
from services.radiator_service import (
    calculate_weighted_delta_t, calculate_extra_power_needed,
    init_radiator_rows, resize_radiator_rows,
    init_collector_rows, resize_collector_rows,
)
from services.room_service import split_heat_loss_to_radiators
from services.pump_service import interpolate_curve, derive_system_curve_k, find_operating_point


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def _safe_float(x, default=None):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def _fix_fig(fig, title=None, height=CHART_HEIGHT_PX):
    fig.update_layout(
        height=height, autosize=False,
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(255,255,255,0.8)", bordercolor="#ddd", borderwidth=1),
        transition_duration=300,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", size=12, color="#333"),
        hoverlabel=dict(bgcolor="white", font_size=13, font_family="Arial, sans-serif"),
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0.5, xanchor="center",
                                     font=dict(size=16, family="Arial, sans-serif", color="#2c3e50")))
    fig.update_yaxes(automargin=True, showline=True, linewidth=1, linecolor="#ddd",
                     gridcolor="#eee", zeroline=False)
    fig.update_xaxes(automargin=True, showline=True, linewidth=1, linecolor="#ddd", gridcolor="#eee")
    return fig


def _empty_fig(title="", height=CHART_HEIGHT_PX):
    return _fix_fig(px.scatter(), title=title, height=height)


def _capture(fn, label, warnings, fallback):
    """Run fn(); on ValueError append to warnings and return fallback."""
    try:
        return fn()
    except ValueError as e:
        warnings.append(f"{label}: {e}")
        return fallback


def _determine_supply_temperature(calc_rows: List[Radiator]) -> float:
    """Max supply temperature across all Radiator objects; fallback 55 °C."""
    candidates = []
    for r in calc_rows:
        t = getattr(r, "supply_temperature", None)
        if t is not None:
            try:
                v = float(t)
                if math.isfinite(v):
                    candidates.append(v)
            except Exception:
                pass
    return max(candidates) if candidates else 55.0


def register(app):

    # ------------------------------------------------------------------
    # Config store
    # ------------------------------------------------------------------
    @app.callback(
        Output("config-store", "data"),
        Input("num_radiators",   "value"), Input("num_collectors", "value"),
        Input("positions",       "value"), Input("kv_max",         "value"),
        Input("delta_T",         "value"), Input("supply_temp_input","value"),
        Input("fix_diameter",    "value"),
        Input("design-mode",     "value"),
        Input("pump_model",      "value"), Input("pump_speed",      "value"),
        Input("valve-type-dropdown", "value"),
    )
    def update_config(nr, nc, pos, kvmax, dT, tsupply, fixlist, design_mode, pump_model, pump_speed, valve_type):
        supply = _safe_float(tsupply, None) if design_mode == MODE_FIXED else None
        if design_mode == MODE_FIXED and supply is None:
            supply = 55.0
        return {
            "num_radiators":    int(nr or 1),
            "num_collectors":   int(nc or 1),
            "positions":        int(pos or 8),
            "kv_max":           _safe_float(kvmax, 0.7) or 0.7,
            "delta_T":          int(dT or 10),
            "supply_temp_input": supply,
            "fix_diameter":     "yes" in (fixlist or []),
            "design_mode":      design_mode or MODE_EXISTING,
            "pump_model":       pump_model or "Grundfos UPM3 15-70",
            "pump_speed":       pump_speed or "speed_2",
            "valve_type": valve_type or "Custom",
        }

    # ------------------------------------------------------------------
    # Table builders
    # ------------------------------------------------------------------
    @app.callback(
        Output("radiator-data-store", "data"),
        Output("collector-data-store", "data"),
        Output("radiator-table",  "data"),
        Output("radiator-table",  "columns"),
        Output("radiator-table",  "dropdown"),
        Output("collector-table", "data"),
        Input("config-store",       "data"),
        Input("room-results-store", "data"),
        Input("fix_diameter",       "value"),
        State("radiator-data-store",  "data"),
        State("collector-data-store", "data"),
    )
    def ensure_tables_and_dropdowns(cfg, room_results_records, fix_diameter_val, radiator_store, collector_store):
        cfg = cfg or {}
        num_radiators  = int(cfg.get("num_radiators",  3))
        num_collectors = int(cfg.get("num_collectors", 1))

        room_df = pd.DataFrame(room_results_records or [])
        room_options      = sorted(room_df["Room"].unique().tolist()) if not room_df.empty else [1, 2, 3]
        collector_options = [f"Collector {i + 1}" for i in range(num_collectors)]

        current_rows = radiator_store or []
        current_rows = (
            init_radiator_rows(num_radiators, collector_options, room_options)
            if not current_rows
            else resize_radiator_rows(current_rows, num_radiators, collector_options, room_options)
        )
        current_collectors = collector_store or []
        current_collectors = (
            init_collector_rows(num_collectors)
            if not current_collectors
            else resize_collector_rows(current_collectors, num_collectors)
        )

        fix_diam = "yes" in (fix_diameter_val or [])

        # Ensure per-radiator diameter column data
        diameter_options = [{"label": str(mm), "value": mm} for mm in [12, 14, 16, 18, 20, 22, 25, 28, 36]]
        if fix_diam:
            for row in current_rows:
                if "Fixed Diameter (mm)" not in row or row["Fixed Diameter (mm)"] is None:
                    row["Fixed Diameter (mm)"] = 16
        else:
            for row in current_rows:
                row.pop("Fixed Diameter (mm)", None)

        base_columns = [
            {"name": "Radiator",                "id": "Radiator nr",             "type": "numeric", "editable": False},
            {"name": "Room",                    "id": "Room",                    "presentation": "dropdown"},
            {"name": "Collector",               "id": "Collector",               "presentation": "dropdown"},
            {"name": "Radiator power 75/65/20", "id": "Radiator power 75/65/20", "type": "numeric"},
            {"name": "Length circuit",          "id": "Length circuit",          "type": "numeric"},
            {"name": "Electric power",          "id": "Electric power",          "type": "numeric"},
        ]
        if fix_diam:
            base_columns.insert(3, {
                "name": "Fixed Diameter (mm)",
                "id": "Fixed Diameter (mm)",
                "presentation": "dropdown",
            })

        dropdown = {
            "Room":      {"options": [{"label": str(o), "value": o} for o in room_options]},
            "Collector": {"options": [{"label": c, "value": c} for c in collector_options]},
        }
        if fix_diam:
            dropdown["Fixed Diameter (mm)"] = {"options": diameter_options}

        return current_rows, current_collectors, current_rows, base_columns, dropdown, current_collectors

    @app.callback(
        Output("radiator-data-store",  "data", allow_duplicate=True),
        Input("radiator-table", "data"), prevent_initial_call=True,
    )
    def write_radiator_edits_back(rows): return rows or []

    @app.callback(
        Output("collector-data-store", "data", allow_duplicate=True),
        Input("collector-table", "data"), prevent_initial_call=True,
    )
    def write_collector_edits_back(rows): return rows or []


    @app.callback(
        Output("valve-balancing-section", "style"),
        Input("config-store", "data"),
    )
    def toggle_valve_balancing_section(cfg):
        if (cfg or {}).get("design_mode") == MODE_BAL:
            return {"display": "block"}
        return {"display": "none"}

    @app.callback(
        Output("heat-loss-split-store", "data"),
        Output("heat-loss-split-table", "data"),
        Input("radiator-data-store",  "data"),
        Input("room-results-store",   "data"),
    )
    def recompute_heat_loss_split(radiator_rows, room_results_records):
        room_df  = pd.DataFrame(room_results_records or [])
        split_df = split_heat_loss_to_radiators(radiator_rows or [], room_df)
        return split_df.to_dict("records"), split_df.to_dict("records")

    # ------------------------------------------------------------------
    # Core computation (Tab 3)
    # ------------------------------------------------------------------
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
        Input("valve-override-store", "data"),
    )
    def compute_results(radiator_rows, collector_rows, split_rows, cfg, room_rows, valve_override_rows):
        _empty = (
            html.Div("⚠️ Complete Tab 1 & 2 first."),
            [], [], [], [],
            _empty_fig(""), _empty_fig(""), _empty_fig(""), "",
            _empty_fig(""), _empty_fig(""), _empty_fig(""),
            "0 W", "0 W", "0 kg/h", "0 °C", "0 °C",
        )
        if not radiator_rows or not collector_rows or not split_rows:
            return _empty

        warnings: List[str] = []
        rad_df   = pd.DataFrame(radiator_rows)
        col_df   = pd.DataFrame(collector_rows)
        split_df = pd.DataFrame(split_rows).rename(
            columns={"Calculated Heat Loss (W)": "Calculated heat loss"}
        )

        if "Radiator nr" not in rad_df.columns or "Radiator nr" not in split_df.columns:
            warnings.append("'Radiator nr' column missing.")
            return _empty

        rad_df = rad_df.merge(split_df[["Radiator nr", "Calculated heat loss"]],
                              on="Radiator nr", how="left")

        # Merge room temperatures
        if room_rows:
            rt = pd.DataFrame(room_rows)
            if "Room #" in rt.columns and "Indoor Temp (°C)" in rt.columns:
                rt = rt[["Room #", "Indoor Temp (°C)"]].rename(columns={"Room #": "Room"})
                rad_df = rad_df.merge(rt, on="Room", how="left")
        rad_df["Space Temperature"] = (
            rad_df.get("Indoor Temp (°C)", pd.Series(dtype=float))
            .fillna(rad_df.get("Space Temperature", pd.Series(dtype=float)))
            .fillna(20.0)
        )

        for col in ["Radiator power 75/65/20", "Calculated heat loss",
                    "Length circuit", "Space Temperature", "Electric power"]:
            if col in rad_df.columns:
                rad_df[col] = pd.to_numeric(rad_df[col], errors="coerce")

        design_mode  = (cfg or {}).get("design_mode", MODE_EXISTING)
        delta_T      = int((cfg or {}).get("delta_T", 10))
        tsupply_user = (cfg or {}).get("supply_temp_input", None)

        calc_rows: List[Radiator] = []
        hydraulics_done = False
        merged_df = None

        # ----------------------------------------------------------
        # Shared sub-routine: build radiators + hydraulics for a ΔT
        # ----------------------------------------------------------
        def _simulate(dt: float):
            _warnings: List[str] = []
            _rad = rad_df.copy()
            _rad["Extra radiator power"] = 0.0
            _calc: List[Radiator] = []

            for _, row in _rad.iterrows():
                base       = row.get("Radiator power 75/65/20", 0.0) or 0.0
                heat_loss  = row.get("Calculated heat loss", 0.0) or 0.0
                extra_elec = row.get("Electric power", 0.0) or 0.0
                want       = max(heat_loss - extra_elec, 0.0)
                q_ratio    = (want / base) if base else 0.0
                _calc.append(Radiator(
                    q_ratio=q_ratio, delta_t=dt,
                    space_temperature=row.get("Space Temperature", 20.0) or 20.0,
                    heat_loss=heat_loss,
                ))

            supply_T = _determine_supply_temperature(_calc)
            for r in _calc:
                r.supply_temperature = supply_T
                r.return_temperature  = r.calculate_treturn(supply_T)
                r.mass_flow_rate      = r.calculate_mass_flow_rate()

            _rad["Supply Temperature"] = supply_T
            _rad["Return Temperature"] = [r.return_temperature for r in _calc]
            _rad["Mass flow rate"]     = [r.mass_flow_rate     for r in _calc]

            # Diameter
            if (cfg or {}).get("fix_diameter"):
                _rad["Diameter"] = _rad["Fixed Diameter (mm)"].fillna(16).astype(int) if "Fixed Diameter (mm)" in _rad.columns else 16
            else:
                diams = [
                    _capture(lambda r=r: r.calculate_diameter(POSSIBLE_DIAMETERS),
                             f"Radiator {i+1}", _warnings, max(POSSIBLE_DIAMETERS))
                    for i, r in enumerate(_calc)
                ]
                _rad["Diameter"] = max(diams) if diams else 16

            # Pressure losses
            _rad["Pressure loss"] = [
                _capture(
                    lambda row=row: Circuit(
                        length_circuit=row.get("Length circuit", 0.0) or 0.0,
                        diameter=row.get("Diameter", 16) or 16,
                        mass_flow_rate=row.get("Mass flow rate", 0.0) or 0.0,
                    ).calculate_pressure_radiator_kv(),
                    f"Radiator {row.get('Radiator nr','?')}", _warnings, float("nan"),
                )
                for _, row in _rad.iterrows()
            ]

            # Collectors
            _col = col_df.copy()
            _collectors = [Collector(name=n) for n in _col["Collector"].tolist()]
            for c in _collectors:
                c.update_mass_flow_rate(_rad)
            _col["Mass flow rate"] = [c.mass_flow_rate for c in _collectors]
            _col["Diameter"] = [
                _capture(lambda c=c: c.calculate_diameter(POSSIBLE_DIAMETERS),
                         c.name, _warnings, max(POSSIBLE_DIAMETERS))
                for c in _collectors
            ]
            _col["Collector pressure loss"] = [
                _capture(
                    lambda row=row: Circuit(
                        length_circuit=row.get("Collector circuit length", 0.0) or 0.0,
                        diameter=row.get("Diameter", 16) or 16,
                        mass_flow_rate=row.get("Mass flow rate", 0.0) or 0.0,
                    ).calculate_pressure_collector_kv(),
                    row.get("Collector", "?"), _warnings, float("nan"),
                )
                for _, row in _col.iterrows()
            ]

            _merged = Collector(name="").calculate_total_pressure_loss(_rad, _col)

            Q_tot     = float(pd.to_numeric(_merged.get("Mass flow rate", pd.Series()), errors="coerce").fillna(0).sum())
            branch_kpa = float(pd.to_numeric(_merged.get("Total Pressure Loss", pd.Series()), errors="coerce").fillna(0).max() / 1000)
            K_sys = derive_system_curve_k(Q_tot, branch_kpa)

            pump_name   = (cfg or {}).get("pump_model", "Grundfos UPM3 15-70")
            pump_speed  = (cfg or {}).get("pump_speed", "speed_2")
            pump_points = (PUMP_LIBRARY.get(pump_name, {}) or {}).get(pump_speed, [])
            q_star = dp_star = None
            if pump_points:
                q_min, q_max = min(p[0] for p in pump_points), max(p[0] for p in pump_points)
                Q_grid   = np.linspace(q_min, q_max, 150)
                pump_kpa = interpolate_curve(pump_points, Q_grid)
                q_star, dp_star, pw = find_operating_point(Q_grid, pump_kpa, K_sys)
                _warnings.extend(pw)

            return _merged, _col, _calc, Q_tot, K_sys, q_star, dp_star, _warnings

        # ----------------------------------------------------------
        # Mode dispatch
        # ----------------------------------------------------------
        if design_mode == MODE_PUMP:
            dt_high   = max(20, delta_T)
            m_hi, c_hi, r_hi, Q_hi, K_hi, qs_hi, dp_hi, w_hi = _simulate(dt_high)
            warnings.extend(w_hi)

            if (qs_hi is None) or (Q_hi > qs_hi):
                warnings.append("Pump insufficient even at max ΔT – consider a larger pump or wider pipes.")
                merged_df, col_df, calc_rows = m_hi, c_hi, r_hi
            else:
                best = (m_hi, c_hi, r_hi, Q_hi, K_hi, qs_hi, dp_hi, dt_high)
                lo, hi = 0.1, float(dt_high)
                for _ in range(50):
                    mid = (lo + hi) / 2
                    m_m, c_m, r_m, Q_m, K_m, qs_m, dp_m, w_m = _simulate(mid)
                    warnings.extend(w_m)
                    if (qs_m is not None) and (Q_m <= qs_m):
                        best = (m_m, c_m, r_m, Q_m, K_m, qs_m, dp_m, mid)
                        hi = mid
                    else:
                        lo = mid
                    if (hi - lo) < 0.1:
                        break
                merged_df, col_df, calc_rows, Q_best, _, qs_best, _, dt_best = best
                warnings.append(
                    f"Pump mode: ΔT={dt_best:.1f} °C, Q≈{Q_best:.0f} kg/h, pump Q*≈{(qs_best or 0):.0f} kg/h."
                )
            hydraulics_done = True

        elif design_mode in (MODE_EXISTING, MODE_BAL):
            rad_df["Extra radiator power"] = 0.0
            for _, row in rad_df.iterrows():
                base       = row.get("Radiator power 75/65/20", 0.0) or 0.0
                heat_loss  = row.get("Calculated heat loss", 0.0) or 0.0
                extra_elec = row.get("Electric power", 0.0) or 0.0
                q_ratio    = max(heat_loss - extra_elec, 0.0) / base if base else 0.0
                r = Radiator(q_ratio=q_ratio, delta_t=delta_T,
                             space_temperature=row.get("Space Temperature", 20.0) or 20.0,
                             heat_loss=heat_loss)
                calc_rows.append(r)

            max_supply = _determine_supply_temperature(calc_rows)
            for r in calc_rows:
                r.supply_temperature = max_supply
                r.return_temperature  = r.calculate_treturn(max_supply)
                r.mass_flow_rate      = r.calculate_mass_flow_rate()

            rad_df["Supply Temperature"] = max_supply
            rad_df["Return Temperature"] = [r.return_temperature for r in calc_rows]
            rad_df["Mass flow rate"]     = [r.mass_flow_rate     for r in calc_rows]

        elif design_mode == MODE_FIXED:
            if tsupply_user is None:
                err = html.Div([
                    html.Div("❌ Error:", className="fw-bold"),
                    html.Div("Enter a supply temperature in Tab 2 for 'LT dimensioning' mode."),
                ], className="alert alert-danger")
                return (err, [], [], [], [],
                        _empty_fig(""), _empty_fig(""), _empty_fig(""), "",
                        _empty_fig(""), _empty_fig(""), _empty_fig(""),
                        "0 W", "0 W", "0 kg/h", "0 °C", "0 °C")

            for _, row in rad_df.iterrows():
                base      = row.get("Radiator power 75/65/20", 0.0) or 0.0
                heat_loss = row.get("Calculated heat loss", 0.0) or 0.0
                space_T   = row.get("Space Temperature", 20.0) or 20.0
                extra = calculate_extra_power_needed(base, heat_loss, float(tsupply_user), delta_T, space_T)
                rad_df.at[row.name, "Extra radiator power"] = extra
                q_ratio = max(heat_loss - (row.get("Electric power", 0) or 0), 0.0) / (base + extra) if base else 0.0
                r = Radiator(q_ratio=q_ratio, delta_t=delta_T, space_temperature=space_T,
                             heat_loss=heat_loss + extra)
                r.supply_temperature = float(tsupply_user)
                r.return_temperature  = r.calculate_treturn(float(tsupply_user))
                r.mass_flow_rate      = r.calculate_mass_flow_rate()
                calc_rows.append(r)

            rad_df["Supply Temperature"] = float(tsupply_user)
            rad_df["Return Temperature"] = [r.return_temperature for r in calc_rows]
            rad_df["Mass flow rate"]     = [r.mass_flow_rate     for r in calc_rows]

        # ----------------------------------------------------------
        # Diameter & pressure losses (non-pump modes)
        # ----------------------------------------------------------

        if not hydraulics_done:
            if (cfg or {}).get("fix_diameter"):
                rad_df["Diameter"] = rad_df["Fixed Diameter (mm)"].fillna(16).astype(int) if "Fixed Diameter (mm)" in rad_df.columns else 16
            else:
                diams = [
                    _capture(lambda r=r: r.calculate_diameter(POSSIBLE_DIAMETERS),
                             f"Radiator {i+1}", warnings, max(POSSIBLE_DIAMETERS))
                    for i, r in enumerate(calc_rows)
                ]
                rad_df["Diameter"] = max(diams) if diams else 16

            rad_df["Pressure loss"] = [
                _capture(
                    lambda row=row: Circuit(
                        length_circuit=row.get("Length circuit", 0.0) or 0.0,
                        diameter=row.get("Diameter", 16) or 16,
                        mass_flow_rate=row.get("Mass flow rate", 0.0) or 0.0,
                    ).calculate_pressure_radiator_kv(),
                    f"Radiator {row.get('Radiator nr','?')}", warnings, float("nan"),
                )
                for _, row in rad_df.iterrows()
            ]


            collector_names = col_df["Collector"].tolist()
            collectors = [Collector(name=n) for n in collector_names]
            for c in collectors:
                c.update_mass_flow_rate(rad_df)
            col_df["Mass flow rate"] = [c.mass_flow_rate for c in collectors]
            col_df["Diameter"]       = [
                _capture(lambda c=c: c.calculate_diameter(POSSIBLE_DIAMETERS),
                         c.name, warnings, max(POSSIBLE_DIAMETERS))
                for c in collectors
            ]
            col_df["Collector pressure loss"] = [
                _capture(
                    lambda row=row: Circuit(
                        length_circuit=row.get("Collector circuit length", 0.0) or 0.0,
                        diameter=row.get("Diameter", 16) or 16,
                        mass_flow_rate=row.get("Mass flow rate", 0.0) or 0.0,
                    ).calculate_pressure_collector_kv(),
                    row.get("Collector", "?"), warnings, float("nan"),
                )
                for _, row in col_df.iterrows()
            ]
            # Check velocities for radiators and collectors
            rad_df, col_df, warnings = check_pipe_velocities(rad_df, col_df, max_velocity=0.5, warnings_list=warnings)
            merged_df = Collector(name="").calculate_total_pressure_loss(rad_df, col_df)
        else:
            rad_df = merged_df.copy()

        # ----------------------------------------------------------
        # Valve — baseline (free calculation)
        # ----------------------------------------------------------
        valve_type  = (cfg or {}).get("valve_type", "Custom")
        kv_max_cfg  = float((cfg or {}).get("kv_max", 0.7) or 0.7)
        n_pos_cfg   = int((cfg or {}).get("positions", 8) or 8)
        if valve_type == "Custom":
            valve = Valve(kv_max=kv_max_cfg, n=n_pos_cfg, valve_name="Custom")
        else:
            valve = Valve(valve_name=valve_type)
            vconf = valve.get_config()
            if vconf:
                kv_max_cfg = vconf["kv_values"][-1]
                n_pos_cfg  = vconf["positions"]
            else:
                valve = Valve(kv_max=kv_max_cfg, n=n_pos_cfg, valve_name="Custom")

        merged_df["Valve pressure loss N"] = merged_df["Mass flow rate"].apply(
            valve.calculate_pressure_valve_kv
        )
        merged_df = (
            valve.calculate_kv_position_valve(merged_df)
            if valve_type != "Custom"
            else valve.calculate_kv_position_valve(merged_df, custom_kv_max=kv_max_cfg, n=n_pos_cfg)
        )
        _pos_col = next((c for c in ["Valve position", "kv_position"] if c in merged_df.columns), None)

        # ----------------------------------------------------------
        # Valve position overrides — full hydraulic network re-solve
        # ----------------------------------------------------------
        if design_mode == MODE_BAL and valve_override_rows:
            override_lookup: dict = {}
            for ov_row in valve_override_rows:
                rad_key      = ov_row.get("Radiator nr")
                override_val = ov_row.get("Position Override")
                if rad_key is not None and override_val not in (None, ""):
                    try:
                        override_lookup[int(rad_key)] = int(override_val)
                    except (ValueError, TypeError):
                        pass

            if override_lookup:
                space_temps_map: dict = {}
                for _, row in merged_df.iterrows():
                    room = str(row.get("Room", ""))
                    t    = row.get("Space Temperature") or row.get("Indoor Temp (°C)") or 20.0
                    space_temps_map[room] = float(t)

                supply_T_fixed = (
                    float(merged_df["Supply Temperature"].iloc[0])
                    if "Supply Temperature" in merged_df.columns
                    else None
                )

                merged_df, col_df, solve_logs = recalculate_with_overrides(
                    merged_df=merged_df,
                    collector_df=col_df,
                    overrides=override_lookup,
                    valve=valve,
                    delta_t=delta_T,
                    space_temps=space_temps_map,
                    supply_temp=supply_T_fixed,
                )
                warnings.extend(solve_logs)
                warnings.append(
                    f"Full hydraulic re-solve completed for "
                    f"{len(override_lookup)} fixed valve(s): "
                    f"radiator(s) {sorted(override_lookup.keys())}. "
                    "Mass flows, return temperatures and pressure losses updated."
                )

        # ----------------------------------------------------------
        # Metrics
        # ----------------------------------------------------------
        # After a valve override the calc_rows objects are stale, so derive
        # every metric directly from merged_df which always reflects the
        # latest network solve.

        # Normalise column names: the base calc writes "Return Temperature"
        # and "Supply Temperature"; valve_override.py writes lower-case "t".
        # Work with whichever variant is present.
        def _col(df, *candidates):
            """Return first matching column Series, or an empty Series."""
            for c in candidates:
                if c in df.columns:
                    return df[c]
            return pd.Series(dtype=float)

        ret_temp_s  = _col(merged_df, "Return temperature",  "Return Temperature")
        sup_temp_s  = _col(merged_df, "Supply temperature",  "Supply Temperature")
        space_temp_s = _col(merged_df, "Space Temperature")

        # Weighted ΔT: flow-weighted average of (supply − return) per radiator.
        # Falls back to calculate_weighted_delta_t when calc_rows is fresh.
        try:
            flows_s = pd.to_numeric(merged_df.get("Mass flow rate", pd.Series()), errors="coerce").fillna(0)
            dt_s    = (sup_temp_s - ret_temp_s).fillna(0)
            total_flow_for_dt = flows_s.sum()
            weighted_dt = float((flows_s * dt_s).sum() / total_flow_for_dt) if total_flow_for_dt > 0 else 0.0
        except Exception:
            weighted_dt = calculate_weighted_delta_t(calc_rows, merged_df)

        total_flow      = float(pd.to_numeric(merged_df.get("Mass flow rate", pd.Series()), errors="coerce").fillna(0).sum())
        total_heat_loss = merged_df.get("Calculated heat loss", pd.Series()).fillna(0).sum()
        sum_extra       = float(pd.to_numeric(merged_df.get("Extra radiator power", pd.Series()), errors="coerce").fillna(0).sum())

        # After an override, show actual heat output; otherwise show nominal.
        #if "Actual heat output (W)" in merged_df.columns:
         #   total_power = float(pd.to_numeric(merged_df["Actual heat output (W)"], errors="coerce").fillna(0).sum())
        #else:
        total_power = merged_df.get("Radiator power 75/65/20", pd.Series()).fillna(0).sum()

        try:
            _ret_num = pd.to_numeric(ret_temp_s, errors="coerce")
            idx_max  = _ret_num.idxmax()
            sup_val  = sup_temp_s.iloc[idx_max] if not sup_temp_s.empty else "N/A"
            rad_nr   = merged_df["Radiator nr"].iloc[idx_max]
            metric_highest_supply = f"{float(sup_val):.1f} °C – Radiator {rad_nr}"
        except Exception:
            metric_highest_supply = "N/A"

        # ----------------------------------------------------------
        # Charts
        # ----------------------------------------------------------
        def _bar(x, ys, names, colors):
            fig = go.Figure()
            for y, name, color in zip(ys, names, colors):
                fig.add_trace(go.Bar(x=x, y=y, name=name, marker_color=color))
            return fig

        rad_ids = merged_df["Radiator nr"] if "Radiator nr" in merged_df.columns else merged_df.index

        if {"Radiator power 75/65/20", "Calculated heat loss"}.issubset(merged_df.columns):
            fig_power = _bar(
                rad_ids,
                [merged_df["Radiator power 75/65/20"], merged_df["Calculated heat loss"]],
                ["Nominal power (75/65/20)", "Required heat loss"], ["#3498db", "#e74c3c"],
            )
            if "Extra radiator power" in merged_df.columns:
                fig_power.add_trace(go.Bar(x=rad_ids, y=merged_df["Extra radiator power"].fillna(0),
                                           name="Extra radiator power", marker_color="#9b59b6"))
            # After an override: overlay the actual heat output at the new flow rate
            if "Actual heat output (W)" in merged_df.columns:
                fig_power.add_trace(go.Bar(
                    x=rad_ids, y=merged_df["Actual heat output (W)"].fillna(0),
                    name="Actual heat output (after override)", marker_color="#f39c12",
                ))
            fig_power.update_layout(barmode="group")
            fig_power = _fix_fig(fig_power, "Radiator Power vs Required Heat Loss")
        else:
            fig_power = _empty_fig("Power data not available")

        # Temperature chart — accept both capitalisation variants from different code paths
        _sup_col   = next((c for c in ["Supply temperature",  "Supply Temperature"]  if c in merged_df.columns), None)
        _ret_col   = next((c for c in ["Return temperature",  "Return Temperature"]  if c in merged_df.columns), None)
        _spc_col   = next((c for c in ["Space Temperature"]                          if c in merged_df.columns), None)
        if _sup_col and _ret_col:
            fig_temp = go.Figure()
            for col, name, color, width, dash in [
                (_sup_col, "Supply", "#e74c3c", 3, None),
                (_ret_col, "Return", "#3498db", 3, None),
            ] + ([(_spc_col, "Space", "#27ae60", 2, "dash")] if _spc_col else []):
                fig_temp.add_trace(go.Scatter(
                    x=rad_ids, y=merged_df[col], mode="lines+markers", name=name,
                    line=dict(color=color, width=width, **({"dash": dash} if dash else {})),
                    marker=dict(size=8 if width == 3 else 6),
                ))
            fig_temp = _fix_fig(fig_temp, "Temperature Profile")
        else:
            fig_temp = _empty_fig("Temperature data not available")

        fig_pressure = (
            _fix_fig(px.bar(merged_df, x="Radiator nr", y="Total Pressure Loss",
                            color="Total Pressure Loss", color_continuous_scale="Viridis"),
                     "Total Pressure Loss per Radiator")
            if "Total Pressure Loss" in merged_df.columns
            else _empty_fig("Total Pressure Loss")
        )
        fig_mass = (
            _fix_fig(px.bar(merged_df, x="Radiator nr", y="Mass flow rate",
                            color="Mass flow rate", color_continuous_scale="Blues"),
                     "Mass Flow Rate per Radiator")
            if "Mass flow rate" in merged_df.columns
            else _empty_fig("Mass Flow Rate")
        )
        valve_col = next((c for c in ["Valve position", "kv_position", "Valve pressure loss N"]
                          if c in merged_df.columns), None)
        if valve_col:
            # Distinguish auto-calculated positions from user overrides visually.
            if "Position Override" in merged_df.columns:
                _has_override = pd.to_numeric(merged_df["Position Override"], errors="coerce").notna()
                _colors = ["#e67e22" if ov else "#2ecc71" for ov in _has_override]
                fig_valve = go.Figure()
                fig_valve.add_trace(go.Bar(
                    x=rad_ids, y=merged_df[valve_col],
                    marker_color=_colors,
                    text=[f"pos {v}" for v in merged_df[valve_col]],
                    textposition="outside",
                    name="Valve position",
                    customdata=merged_df["Position Override"].fillna("auto"),
                    hovertemplate=(
                        "Radiator %{x}<br>"
                        "Position: %{y}<br>"
                        "Override: %{customdata}<extra></extra>"
                    ),
                ))
                # Legend proxies
                fig_valve.add_trace(go.Bar(x=[None], y=[None], name="Auto-calculated", marker_color="#2ecc71"))
                fig_valve.add_trace(go.Bar(x=[None], y=[None], name="Override",        marker_color="#e67e22"))
                fig_valve = _fix_fig(fig_valve, "Valve Position (green = auto, orange = override)")
            else:
                fig_valve = _fix_fig(
                    px.bar(merged_df, x="Radiator nr", y=valve_col,
                           color=valve_col, color_continuous_scale="RdYlGn_r"),
                    f"Valve Analysis: {valve_col}",
                )
        else:
            fig_valve = _empty_fig("Valve data not available")

        # Pump curve chart
        fig_pump = _empty_fig("Pump curve (pump mode only)")
        if design_mode == MODE_PUMP:
            pump_name   = (cfg or {}).get("pump_model", "Grundfos UPM3 15-70")
            pump_speed  = (cfg or {}).get("pump_speed", "speed_2")
            pump_points = (PUMP_LIBRARY.get(pump_name, {}) or {}).get(pump_speed, [])
            if pump_points:
                Q_tot2     = float(pd.to_numeric(merged_df.get("Mass flow rate", pd.Series()), errors="coerce").fillna(0).sum())
                branch2    = float(pd.to_numeric(merged_df.get("Total Pressure Loss", pd.Series()), errors="coerce").fillna(0).max() / 1000)
                K2         = derive_system_curve_k(Q_tot2, branch2)
                q_min, q_max = min(p[0] for p in pump_points), max(p[0] for p in pump_points)
                Q_grid     = np.linspace(q_min, q_max, 150)
                pump_kpa   = interpolate_curve(pump_points, Q_grid)
                q_star2, dp_star2, _ = find_operating_point(Q_grid, pump_kpa, K2)
                fig_pump = go.Figure()
                fig_pump.add_trace(go.Scatter(x=Q_grid, y=pump_kpa, mode="lines",
                                              name=f"{pump_name} ({pump_speed})",
                                              line=dict(color="#1f77b4", width=3)))
                fig_pump.add_trace(go.Scatter(x=Q_grid, y=K2 * Q_grid ** 2, mode="lines",
                                              name="System curve (K·Q²)",
                                              line=dict(color="#e74c3c", width=3, dash="dash")))
                if q_star2 and dp_star2:
                    fig_pump.add_trace(go.Scatter(
                        x=[q_star2], y=[dp_star2], mode="markers+text",
                        name="Operating point", marker=dict(color="#2ca02c", size=10),
                        text=[f"Q={q_star2:.0f} kg/h, ΔP={dp_star2:.1f} kPa"],
                        textposition="top center",
                    ))
                fig_pump = _fix_fig(fig_pump, f"Pump vs System — {pump_name} ({pump_speed})")

        # ----------------------------------------------------------
        # Summary block
        # ----------------------------------------------------------
        _power_label = (
            "Total actual heat output" if "Actual heat output (W)" in merged_df.columns
            else "Total radiator power (nominal)"
        )
        summary = html.Ul([
            html.Li(f"Mode: {design_mode}"),
            html.Li(f"Weighted ΔT: {weighted_dt:.2f} °C"),
            html.Li(f"Total mass flow rate: {total_flow:.2f} kg/h"),
            html.Li(f"Total heat loss: {total_heat_loss:.0f} W"),
            html.Li(f"{_power_label}: {total_power:.0f} W"),
            html.Li(f"Sum extra power (normalised): {sum_extra:.0f} W"),
            html.Li(f"Highest supply T: {metric_highest_supply}"),
            html.Li(f"Radiators: {len(rad_df)} — Collectors: {len(col_df)}"),
        ])
        warn_div = (
            html.Div([html.Div("⚠️ Warnings", className="fw-bold mb-1")]
                     + [html.Div(w) for w in warnings],
                     className="alert alert-warning")
            if warnings else html.Div()
        )

        return (
            warn_div,
            [{"name": c, "id": c} for c in merged_df.columns], merged_df.to_dict("records"),
            [{"name": c, "id": c} for c in col_df.columns],    col_df.to_dict("records"),
            fig_pressure, fig_mass, fig_valve, summary,
            fig_power, fig_temp, fig_pump,
            f"{total_heat_loss:.0f} W", f"{total_power:.0f} W",
            f"{total_flow:.2f} kg/h",   f"{weighted_dt:.2f} °C",
            metric_highest_supply,
        )

    # ------------------------------------------------------------------
    # Valve balancing table — built from merged results, not from compute_results
    # ------------------------------------------------------------------
    @app.callback(
        Output("valve-balancing-table", "columns"),
        Output("valve-balancing-table", "data"),
        Input("merged-results-table",   "data"),
        Input("config-store",           "data"),
        State("valve-override-store",   "data"),
    )
    def build_valve_balancing_table(merged_rows, cfg, override_rows):
        if (cfg or {}).get("design_mode") != MODE_BAL or not merged_rows:
            return [], []

        pos_col = next(
            (c for c in ["Valve position", "kv_position"]
             if any(c in (r or {}) for r in merged_rows)),
            None,
        )
        override_lookup = {
            int(r["Radiator nr"]): r.get("Position Override", "")
            for r in (override_rows or [])
            if r.get("Radiator nr") is not None
        }

        def _round(v, n):
            try:
                return round(float(v), n)
            except (TypeError, ValueError):
                return None

        bal_data = []
        for row in merged_rows:
            rad_nr   = row.get("Radiator nr")
            eff_pos  = row.get(pos_col) if pos_col else None
            mfr      = row.get("Mass flow rate")
            t_ret    = row.get("Return temperature") or row.get("Return Temperature")
            dp_valve = row.get("Valve pressure loss N")
            dp_total = row.get("Total Pressure Loss")
            q_actual = row.get("Actual heat output (W)")
            bal_data.append({
                "Radiator nr":       rad_nr,
                "Room":              row.get("Room", ""),
                "Flow (kg/h)":       _round(mfr, 2),
                "T return (°C)":    _round(t_ret, 1),
                "Valve position":    _round(eff_pos, 0),
                "Heat output (W)":   _round(q_actual, 0),
                "Valve ΔP (Pa)":    _round(dp_valve, 1),
                "Total ΔP (Pa)":    _round(dp_total, 1),
                "Position Override": override_lookup.get(
                    int(rad_nr) if rad_nr is not None else -1, ""),
            })

        bal_columns = [
            {"name": "Radiator",          "id": "Radiator nr",       "type": "numeric", "editable": False},
            {"name": "Room",              "id": "Room",              "type": "any",     "editable": False},
            {"name": "Flow (kg/h)",       "id": "Flow (kg/h)",      "type": "numeric", "editable": False},
            {"name": "T return (°C)",    "id": "T return (°C)",  "type": "numeric", "editable": False},
            {"name": "Valve position",    "id": "Valve position",   "type": "numeric", "editable": False},
            {"name": "Heat output (W)",   "id": "Heat output (W)",  "type": "numeric", "editable": False},
            {"name": "Valve ΔP (Pa)",    "id": "Valve ΔP (Pa)",   "type": "numeric", "editable": False},
            {"name": "Total ΔP (Pa)",    "id": "Total ΔP (Pa)",   "type": "numeric", "editable": False},
            {"name": "Position Override", "id": "Position Override", "type": "numeric", "editable": True},
        ]
        return bal_columns, bal_data

    # ------------------------------------------------------------------
    # Persist overrides on button click → triggers full recompute via store
    # ------------------------------------------------------------------
    @app.callback(
        Output("valve-override-store",    "data"),
        Output("valve-override-feedback", "children"),
        Input("apply-valve-overrides-btn", "n_clicks"),
        State("valve-balancing-table",    "data"),
        prevent_initial_call=True,
    )
    def persist_valve_overrides(n_clicks, table_rows):
        if not table_rows:
            return [], ""
        # Store rows as-is; the solver reads "Position Override" directly.
        valid = [r for r in table_rows if r.get("Position Override") not in (None, "")]
        n = len(valid)
        msg = (
            f"✅ {n} override(s) applied — recalculating..."
            if n else
            "⚠️ No overrides entered. Fill the Position Override column and try again."
        )
        return table_rows, msg