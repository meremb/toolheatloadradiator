"""
ui/callbacks/heat_loss.py
=========================
Callbacks for Tab 1: room table building and heat-loss computation.
"""
from __future__ import annotations

import pandas as pd
from dash import Input, Output, State, dash_table, html

from config import ROOM_TYPE_OPTIONS
from services.room_service import compute_room_results, default_room_table


def _safe_float(x, default=None):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def register(app):

    @app.callback(
        Output("room-table", "columns"),
        Output("room-table", "data"),
        Output("room-table", "dropdown"),
        Input("num_rooms", "value"),
        State("room-table", "data"),
        prevent_initial_call=False,
    )
    def build_room_table(num_rooms, existing):
        try:
            num_rooms = int(num_rooms) if (num_rooms and int(num_rooms) > 0) else 1
        except Exception:
            num_rooms = 1

        columns = [
            {"name": "Room #",          "id": "Room #",          "type": "numeric"},
            {"name": "Indoor Temp (°C)", "id": "Indoor Temp (°C)", "type": "numeric", "editable": True, "presentation": "input"},
            {"name": "Floor Area (m²)", "id": "Floor Area (m²)", "type": "numeric"},
            {"name": "Walls external",  "id": "Walls external",  "type": "numeric", "presentation": "dropdown"},
            {"name": "Room Type",       "id": "Room Type",       "type": "text",    "presentation": "dropdown"},
            {"name": "On Ground",       "id": "On Ground",       "type": "any",     "presentation": "dropdown"},
            {"name": "Under Roof",      "id": "Under Roof",      "type": "any",     "presentation": "dropdown"},
        ]
        if not existing:
            data = default_room_table(num_rooms)
        else:
            cur = len(existing)
            if num_rooms > cur:
                data = existing.copy()
                for i in range(cur + 1, num_rooms + 1):
                    data.append({"Room #": i, "Indoor Temp (°C)": 20.0, "Floor Area (m²)": 10.0,
                                 "Walls external": 2, "Room Type": "room",
                                 "On Ground": False, "Under Roof": False})
            else:
                data = existing[:num_rooms]

        dropdown = {
            "Room Type":     {"options": [{"label": v, "value": v} for v in ROOM_TYPE_OPTIONS]},
            "On Ground":     {"options": [{"label": "No", "value": False}, {"label": "Yes", "value": True}]},
            "Under Roof":    {"options": [{"label": "No", "value": False}, {"label": "Yes", "value": True}]},
            "Walls external":{"options": [{"label": str(v), "value": v} for v in [1, 2, 3, 4]]},
        }
        return columns, data, dropdown

    @app.callback(
        Output("manual-loss-table", "data"),
        Input("num_rooms", "value"),
        State("manual-loss-table", "data"),
        prevent_initial_call=False,
    )
    def build_manual_loss_table(num_rooms, current):
        try:
            n = int(num_rooms) if (num_rooms and int(num_rooms) > 0) else 1
        except Exception:
            n = 1
        rows   = (current or []).copy()
        by_room = {r.get("Room #"): r for r in rows if "Room #" in r}
        new_rows = []
        for i in range(1, n + 1):
            if i in by_room:
                r = by_room[i]
                r["Room #"] = i
                r.setdefault("Manual Heat Loss (W)", 0.0)
                new_rows.append(r)
            else:
                new_rows.append({"Room #": i, "Manual Heat Loss (W)": 0.0})
        return new_rows

    @app.callback(
        Output("room-results-store", "data"),
        Output("room-results-table", "children"),
        Input("room-table", "data"),
        Input("uw",    "value"), Input("u_roof",  "value"),
        Input("u_ground", "value"), Input("u_glass", "value"),
        Input("tout",  "value"),
        Input("ventilation_calculation_method", "value"),
        Input("v_system", "value"), Input("v50", "value"),
        Input("neighbour_t", "value"), Input("un", "value"),
        Input("lir", "value"), Input("wall_height", "value"),
        Input("heat-load-mode-store", "data"),
        Input("manual-loss-table", "data"),
    )
    def compute_rooms_and_table(
        room_rows, uw, u_roof, u_ground, u_glass, tout,
        vcalc, vsys, v50, neighbour_t, un, lir, wall_height,
        mode, manual_rows,
    ):
        _table_style = dict(
            style_table={"overflowX": "auto"},
            style_cell={"padding": "8px", "border": "1px solid #dee2e6"},
            style_header={"backgroundColor": "#f8f9fa", "fontWeight": "bold", "textAlign": "center"},
        )

        if mode == "known":
            df = pd.DataFrame(manual_rows or [])
            if not df.empty:
                df["Room"] = pd.to_numeric(df.get("Room #", 0), errors="coerce").fillna(0).astype(int)
                df["Total Heat Loss (W)"] = pd.to_numeric(df.get("Manual Heat Loss (W)", 0.0), errors="coerce").fillna(0.0)
                df = df[["Room", "Total Heat Loss (W)"]]
            else:
                df = pd.DataFrame(columns=["Room", "Total Heat Loss (W)"])
            records = df.to_dict("records")
            table = dash_table.DataTable(
                columns=[{"name": c, "id": c} for c in df.columns],
                data=records, page_size=10, **_table_style,
            )
            return records, table

        # Validate and clamp room inputs
        for row in (room_rows or []):
            if "Indoor Temp (°C)" in row:
                try:
                    t = float(row["Indoor Temp (°C)"])
                    row["Indoor Temp (°C)"] = max(10.0, min(24.0, t))
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
            uw             = _safe_float(uw,          1.0) or 1.0,
            u_roof         = _safe_float(u_roof,       0.2) or 0.2,
            u_ground       = _safe_float(u_ground,     0.3) or 0.3,
            u_glass        = _safe_float(u_glass,      2.8) or 2.8,
            tout           = _safe_float(tout,        -7.0) if tout is not None else -7.0,
            heat_loss_area_estimation      = "fromFloorArea",
            ventilation_calculation_method = vcalc or "simple",
            v_system       = vsys or "C",
            v50            = _safe_float(v50,          6.0) or 6.0,
            neighbour_t    = _safe_float(neighbour_t, 18.0) or 18.0,
            un             = _safe_float(un,           1.0) or 1.0,
            lir            = _safe_float(lir,          0.2) or 0.2,
            wall_height    = _safe_float(wall_height,  2.7) or 2.7,
            return_detail  = False,
            add_neighbour_losses = True,
        )
        records = df_results.to_dict("records")
        table = dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in df_results.columns],
            data=records, page_size=10, **_table_style,
        )
        return records, table
