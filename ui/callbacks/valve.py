"""
ui/callbacks/valve.py
=====================
Callbacks for the valve-selection panel in Tab 2.
"""
from __future__ import annotations

from dash import Input, Output, State, html, no_update

from domain.valve import Valve


def register(app):

    @app.callback(
        Output("valve-specs",           "children"),
        Output("valve-custom-settings", "style"),
        Input("valve-type-dropdown",    "value"),
    )
    def update_valve_info(selected_valve):
        if selected_valve == "Custom":
            return "", {"display": "block"}
        try:
            config = Valve(valve_name=selected_valve).get_config()
            if config:
                specs = [
                    html.Div(f"Positions: {config['positions']}"),
                    html.Div(f"Kv range: {config['kv_values'][0]} – {config['kv_values'][-1]} m³/h"),
                ]
                return specs, {"display": "none"}
        except Exception as e:
            print(f"update_valve_info error: {e}")
        return "Error loading valve specs", {"display": "block"}

    @app.callback(
        Output("positions", "value"),
        Output("kv_max",    "value"),
        Input("valve-type-dropdown", "value"),
    )
    def update_valve_defaults(selected_valve):
        if selected_valve == "Custom":
            return no_update, no_update
        try:
            config = Valve(valve_name=selected_valve).get_config()
            if config:
                return config["positions"], config["kv_values"][-1]
        except Exception as e:
            print(f"update_valve_defaults error: {e}")
        return no_update, no_update

    @app.callback(
        Output("config-store", "data", allow_duplicate=True),
        Input("valve-type-dropdown", "value"),
        Input("positions", "value"),
        Input("kv_max",    "value"),
        State("config-store", "data"),
        prevent_initial_call=True,
    )
    def update_valve_config(selected_valve, positions, kv_max, current_cfg):
        cfg = current_cfg or {}
        if selected_valve == "Custom":
            cfg.update(valve_type="Custom", positions=positions, kv_max=kv_max)
        else:
            try:
                config = Valve(valve_name=selected_valve).get_config()
                if config:
                    cfg.update(
                        valve_type=selected_valve,
                        positions=config["positions"],
                        kv_max=config["kv_values"][-1],
                    )
            except Exception as e:
                print(f"update_valve_config error: {e}")
        return cfg
