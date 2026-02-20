"""
ui/callbacks/navigation.py
==========================
Callbacks that control navigation, mode selection, and UI visibility.
"""
from __future__ import annotations

from dash import Input, Output, no_update, callback_context

from config import (
    MODE_EXISTING, MODE_FIXED, MODE_PUMP, MODE_BAL,
    INSULATION_U_VALUES, GLAZING_U_VALUES,
)


def register(app):
    """Register all navigation/visibility callbacks on the given Dash app."""

    @app.callback(Output("design-mode-help", "children"), Input("design-mode", "value"))
    def help_block_for_mode(mode):
        messages = {
            MODE_EXISTING: "Existing: calculate required supply temperature for current radiators.",
            MODE_FIXED:    "LT dimensioning: choose a fixed supply temperature; tool calculates extra radiator power.",
            MODE_PUMP:     "Pump-based: selected pump/speed determines achievable flow via ΔT iteration.",
            MODE_BAL:      "Balancing: determine TRV positions and flow distribution.",
        }
        return messages.get(mode, "")

    @app.callback(
        Output("heat-load-mode-store", "data"),
        Output("tabs", "active_tab"),
        Input("heat-load-mode", "value"),
    )
    def choose_mode_and_go(mode):
        if mode in ("known", "unknown"):
            return mode, "tab-1"
        return no_update, no_update

    @app.callback(
        Output("manual-loss-card", "style"),
        Output("heat-load-mode-banner", "children"),
        Input("heat-load-mode-store", "data"),
    )
    def toggle_manual_ui(mode):
        if mode == "known":
            return {"display": "block"}, "Mode: 🔒 Heat load is KNOWN — enter per-room heat losses below."
        return {"display": "none"}, "Mode: 🧮 Heat load is UNKNOWN — the tool will calculate room heat losses."

    @app.callback(
        Output("building-envelope-card",   "style"),
        Output("outdoor-conditions-card",  "style"),
        Output("building-insulation-card", "style"),
        Output("additional-settings-card", "style"),
        Output("room-config-card",         "style"),
        Input("heat-load-mode-store", "data"),
    )
    def toggle_known_mode_visibility(mode):
        hidden, shown = {"display": "none"}, {"display": "block"}
        if mode == "known":
            return hidden, hidden, hidden, hidden, hidden
        return shown, shown, shown, shown, shown

    @app.callback(
        Output("uw", "value"), Output("u_roof", "value"),
        Output("u_ground", "value"), Output("u_glass", "value"),
        Input("wall_insulation_state",   "value"),
        Input("roof_insulation_state",   "value"),
        Input("ground_insulation_state", "value"),
        Input("glazing_type",            "value"),
        Input("uw", "value"), Input("u_roof", "value"),
        Input("u_ground", "value"), Input("u_glass", "value"),
    )
    def set_default_u_values(wall_state, roof_state, ground_state, glazing,
                             uw, u_roof, u_ground, u_glass):
        triggered = [t["prop_id"] for t in callback_context.triggered]
        if "wall_insulation_state.value" in triggered and wall_state in INSULATION_U_VALUES:
            uw = INSULATION_U_VALUES[wall_state]["wall"]
        if "roof_insulation_state.value" in triggered and roof_state in INSULATION_U_VALUES:
            u_roof = INSULATION_U_VALUES[roof_state]["roof"]
        if "ground_insulation_state.value" in triggered and ground_state in INSULATION_U_VALUES:
            u_ground = INSULATION_U_VALUES[ground_state]["ground"]
        if "glazing_type.value" in triggered and glazing in GLAZING_U_VALUES:
            u_glass = GLAZING_U_VALUES[glazing]
        return uw, u_roof, u_ground, u_glass

    # Tab 2 UI toggles
    @app.callback(
        Output("fixed_diameter_container", "style"),
        Input("fix_diameter", "value"),
    )
    def toggle_fixed_diameter(checklist):
        return {"display": "block"} if "yes" in (checklist or []) else {"display": "none"}

    @app.callback(
        Output("supply_temp_input", "disabled"),
        Input("design-mode", "value"),
    )
    def disable_supply_input(mode):
        return mode in (MODE_EXISTING, MODE_PUMP, MODE_BAL)

    @app.callback(
        Output("pump-settings-card", "style"),
        Input("design-mode", "value"),
    )
    def toggle_pump_ui(mode):
        return {"display": "block"} if mode == MODE_PUMP else {"display": "none"}
