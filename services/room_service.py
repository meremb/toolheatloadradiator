"""
services/room_service.py
========================
Orchestrates room-level heat loss calculations.

Takes raw UI inputs (dicts/DataFrames) → delegates to domain objects → returns
DataFrames ready for the UI layer. No Dash imports here.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from domain.heat_load import RoomLoadCalculator


def compute_room_results(
    room_rows: List[Dict[str, Any]],
    uw: float,
    u_roof: float,
    u_ground: float,
    u_glass: float,
    tout: float,
    heat_loss_area_estimation: str,
    ventilation_calculation_method: str,
    v_system: str,
    v50: float,
    neighbour_t: float,
    un: float,
    lir: float,
    wall_height: float,
    return_detail: bool,
    add_neighbour_losses: bool,
) -> pd.DataFrame:
    """
    Run RoomLoadCalculator for each room row and return a results DataFrame.

    Parameters mirror the UI inputs directly; each row represents one room.
    Returns a DataFrame with columns ['Room', 'Total Heat Loss (W)'].
    """
    results = []
    for row in room_rows:
        calc = RoomLoadCalculator(
            floor_area  = float(row.get("Floor Area (m²)", 0.0) or 0.0),
            uw=uw, u_roof=u_roof, u_ground=u_ground, u_glass=u_glass,
            v_system=v_system, v50=v50,
            tin         = float(row.get("Indoor Temp (°C)", 20.0) or 20.0),
            tout=tout,
            neighbour_t=neighbour_t, un=un, lir=lir,
            heat_loss_area_estimation     = heat_loss_area_estimation,
            ventilation_calculation_method = ventilation_calculation_method,
            on_ground    = bool(row.get("On Ground", False)),
            under_roof   = bool(row.get("Under Roof", False)),
            add_neighbour_losses = add_neighbour_losses,
            neighbour_perimeter  = float(row.get("Neighbour Perimeter (m)", 0.0) or 0.0),
            room_type    = row.get("Room Type", "Living"),
            wall_height  = wall_height,
            wall_outside = float(row.get("Walls external", 2) or 2),
            return_detail = return_detail,
            window = True,
        )
        result = calc.compute()
        total  = result if isinstance(result, (int, float)) else result.get("totalHeatLoss", 0.0)
        results.append({
            "Room": row["Room #"],
            "Total Heat Loss (W)": float(total or 0.0),
        })
    return pd.DataFrame(results)


def split_heat_loss_to_radiators(
    radiator_rows: List[Dict[str, Any]],
    room_results_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Distribute each room's heat loss equally across the radiators assigned to it.

    Returns a DataFrame with columns:
        ['Radiator nr', 'Calculated Heat Loss (W)', 'Room']
    """
    if room_results_df is None or room_results_df.empty or not radiator_rows:
        return pd.DataFrame(columns=["Radiator nr", "Calculated Heat Loss (W)", "Room"])

    rad_df = pd.DataFrame(radiator_rows)
    heat_loss_df = pd.DataFrame({
        "Radiator nr":              rad_df["Radiator nr"],
        "Calculated Heat Loss (W)": 0.0,
        "Room":                     rad_df["Room"],
    })
    room_heat_map = room_results_df.set_index("Room")["Total Heat Loss (W)"].to_dict()

    for room, idxs in rad_df.groupby("Room").groups.items():
        n_rad = len(idxs)
        if room in room_heat_map and n_rad > 0:
            split_load = room_heat_map[room] / n_rad
            heat_loss_df.loc[idxs, "Calculated Heat Loss (W)"] = split_load

    return heat_loss_df


def default_room_table(num_rooms: int) -> List[Dict[str, Any]]:
    """Initial room table rows with sensible defaults."""
    return [
        {
            "Room #": i,
            "Indoor Temp (°C)": 20.0,
            "Floor Area (m²)": 20.0,
            "Walls external": 2,
            "Room Type": "Living",
            "On Ground": False,
            "Under Roof": False,
        }
        for i in range(1, num_rooms + 1)
    ]
