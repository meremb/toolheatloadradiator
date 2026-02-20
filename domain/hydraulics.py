"""
domain/hydraulics.py
====================
Pipe-circuit and collector-manifold hydraulic models.

Calculates pressure losses, water volumes, and cumulative branch pressures.
No Dash, no UI, no business-logic orchestration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from domain.radiator import POSSIBLE_DIAMETERS, _select_pipe_diameter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PRESSURE_LOSS_BOILER: float = 350.0    # Pa – fixed boiler/manifold loss

HYDRAULIC_CONSTANT: float  = 97180.0  # Pa at mass_flow [kg/s] / kv [m³/h]
LOCAL_LOSS_COEFFICIENT: float = 1.3   # Accounts for fittings and bends

KV_RADIATOR:  float = 2.0    # m³/h – radiator body Kv
KV_COLLECTOR: float = 14.66  # m³/h – collector manifold Kv

# Pipe kv polynomial: kv = A·d² + B·d + C  (d in metres)
KV_PIPE_A: float =  51626.0
KV_PIPE_B: float = -417.39
KV_PIPE_C: float =   1.5541

WATER_DENSITY:        float = 1000.0  # kg/m³
MAX_VELOCITY_DEFAULT: float = 0.5     # m/s comfort limit


# ---------------------------------------------------------------------------
# Circuit
# ---------------------------------------------------------------------------
@dataclass
class Circuit:
    """
    Pressure-loss and water-volume model for one pipe circuit.

    Parameters
    ----------
    length_circuit : One-way run length  [m]
    diameter       : Internal diameter   [mm]
    mass_flow_rate : Water mass flow     [kg/h]
    """

    length_circuit: float
    diameter: float
    mass_flow_rate: float

    def calculate_pressure_loss_piping(self) -> float:
        """Friction + local losses for the pipe run [Pa]."""
        kv_pipe = self._kv_pipe()
        r_per_m  = HYDRAULIC_CONSTANT * (self.mass_flow_rate / 1000.0 / kv_pipe) ** 2
        return round(r_per_m * self.length_circuit * 2 * LOCAL_LOSS_COEFFICIENT, 1)

    def calculate_pressure_radiator_kv(self) -> float:
        """Total: piping + radiator body [Pa]."""
        radiator_loss = HYDRAULIC_CONSTANT * (self.mass_flow_rate / 1000.0 / KV_RADIATOR) ** 2
        return round(self.calculate_pressure_loss_piping() + radiator_loss, 1)

    def calculate_pressure_collector_kv(self) -> float:
        """Total: piping + collector manifold [Pa]."""
        collector_loss = HYDRAULIC_CONSTANT * (self.mass_flow_rate / 1000.0 / KV_COLLECTOR) ** 2
        return round(self.calculate_pressure_loss_piping() + collector_loss, 1)

    def calculate_water_volume(self) -> float:
        """Water volume in this circuit [litres]."""
        r_m = (self.diameter / 2.0) / 1000.0
        return round(math.pi * r_m ** 2 * self.length_circuit * 1000.0, 2)

    def _kv_pipe(self) -> float:
        d = self.diameter / 1000.0  # mm → m
        return KV_PIPE_A * d ** 2 + KV_PIPE_B * d + KV_PIPE_C


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------
@dataclass
class Collector:
    """
    Hydraulic model for a manifold serving multiple radiators.

    Parameters
    ----------
    name           : Unique collector label (e.g. 'Collector 1')
    pressure_loss  : Branch pressure loss [Pa]
    mass_flow_rate : Sum of connected radiator flows [kg/h]
    """

    name: str
    pressure_loss: float = 0.0
    mass_flow_rate: float = 0.0

    def update_mass_flow_rate(self, radiator_df: pd.DataFrame) -> None:
        """Sum mass flows of all radiators on this collector."""
        mask = radiator_df["Collector"] == self.name
        self.mass_flow_rate = float(radiator_df.loc[mask, "Mass flow rate"].sum())

    def calculate_diameter(
        self, possible_diameters: List[int] = POSSIBLE_DIAMETERS
    ) -> float:
        """Smallest adequate pipe diameter for the collector [mm]."""
        if math.isnan(self.mass_flow_rate):
            raise ValueError("Collector mass flow rate is NaN.")
        return _select_pipe_diameter(self.mass_flow_rate, possible_diameters)

    def calculate_total_pressure_loss(
        self,
        radiator_df: pd.DataFrame,
        collector_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute cumulative pressure loss per radiator, including all downstream
        collector losses (daisy-chain topology) and the fixed boiler loss.

        Returns a merged DataFrame with a 'Total Pressure Loss' column.
        """
        sorted_collectors = collector_df.sort_values("Collector")
        merged = pd.merge(
            radiator_df,
            sorted_collectors[["Collector", "Collector pressure loss"]],
            on="Collector", how="left",
        )
        loss_map: Dict[str, float] = (
            sorted_collectors.set_index("Collector")["Collector pressure loss"].to_dict()
        )
        collector_names = list(loss_map.keys())

        totals = []
        for _, row in merged.iterrows():
            idx = collector_names.index(row["Collector"])
            downstream = sum(loss_map[collector_names[i]] for i in range(idx, len(collector_names)))
            totals.append(row["Pressure loss"] + downstream + PRESSURE_LOSS_BOILER)

        merged["Total Pressure Loss"] = totals
        return merged


# ---------------------------------------------------------------------------
# Velocity utilities
# ---------------------------------------------------------------------------

def calc_velocity(mass_flow_rate_kgph: float, diameter_mm: float) -> float:
    """Water velocity in a circular pipe [m/s]."""
    m_dot = float(mass_flow_rate_kgph) / 3600.0
    d_m   = float(diameter_mm) / 1000.0
    area  = math.pi * (d_m / 2.0) ** 2
    if area == 0:
        return 0.0
    return m_dot / (WATER_DENSITY * area)


def check_pipe_velocities(
    rad_df: pd.DataFrame,
    col_df: pd.DataFrame,
    max_velocity: float = MAX_VELOCITY_DEFAULT,
    warnings_list: "list[str] | None" = None,
) -> "tuple[pd.DataFrame, pd.DataFrame, list[str]]":
    """
    Append 'Velocity (m/s)' to both DataFrames and flag high-velocity circuits.

    Returns (rad_df, col_df, warnings_list) — all modified copies.
    """
    if warnings_list is None:
        warnings_list = []

    rad_df = rad_df.copy()
    rad_vels = []
    for _, row in rad_df.iterrows():
        v = calc_velocity(row.get("Mass flow rate", 0.0) or 0.0, row.get("Diameter", 0.0) or 0.0)
        rad_vels.append(round(v, 3))
        if v > max_velocity:
            rid = row.get("Radiator nr", row.get("Radiator", "?"))
            warnings_list.append(f"High velocity radiator {rid}: {v:.2f} m/s > {max_velocity:.2f} m/s")
    rad_df["Velocity (m/s)"] = rad_vels

    col_df = col_df.copy()
    col_vels = []
    for _, row in col_df.iterrows():
        v = calc_velocity(row.get("Mass flow rate", 0.0) or 0.0, row.get("Diameter", 0.0) or 0.0)
        col_vels.append(round(v, 3))
        if v > max_velocity:
            warnings_list.append(
                f"High velocity collector {row.get('Collector', '?')}: {v:.2f} m/s > {max_velocity:.2f} m/s"
            )
    col_df["Velocity (m/s)"] = col_vels

    return rad_df, col_df, warnings_list
