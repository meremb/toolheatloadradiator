"""
app.py
======
Application entry point.

Responsibilities
----------------
1. Create the Dash app instance.
2. Assemble the layout from ui/layout.py.
3. Register all callbacks from ui/callbacks/.
4. Run the dev server.

Nothing else belongs here.
"""
import dash_bootstrap_components as dbc
from dash import Dash, dcc

from ui.layout import (
    navbar, build_stores,
    build_start_tab, build_heat_tab, build_radiator_tab, build_results_tab,
)
from ui.callbacks import navigation, heat_loss, hydraulics, valve

# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.ZEPHYR, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
app.title = "Smart Heating Design Tool"
server = app.server  # Expose for deployment (Gunicorn / Railway / etc.)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
app.layout = dbc.Container(
    [
        navbar,
        dcc.Location(id="url", refresh=False),
        *build_stores(),
        dbc.Tabs(
            id="tabs",
            active_tab="tab-0",
            className="justify-content-center",
            children=[
                build_start_tab(),
                build_heat_tab(),
                build_radiator_tab(),
                build_results_tab(),
            ],
        ),
    ],
    fluid=True,
    style={"backgroundColor": "#f4f6fa", "padding": "24px 0"},
)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
navigation.register(app)
heat_loss.register(app)
hydraulics.register(app)
valve.register(app)

# ---------------------------------------------------------------------------
# Dev server
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
