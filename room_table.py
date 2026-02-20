from dash import Dash, dash_table, html
import dash_bootstrap_components as dbc

app = Dash()

ROOM_TYPE_OPTIONS = ["Living", "Kitchen", "Bedroom", "Laundry", "Bathroom", "Toilet"]

app.layout = html.Div([
    dash_table.DataTable(
        id="room-table",
        columns=[
            {"name": "Room #", "id": "Room #", "type": "numeric", "editable": False},
            {"name": "Room Type", "id": "Room Type", "type": "text", "presentation": "dropdown"},
        ],
        editable=True,
        dropdown={
            "Room Type": {"options": [{"label": v, "value": v} for v in ROOM_TYPE_OPTIONS]},
        },
        data=[{"Room #": 1, "Room Type": "Living"}],
    )
])

if __name__ == "__main__":
    app.run(debug=True)

