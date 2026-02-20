"""
services/radiator_service.py
============================
Orchestrates batch radiator and collector calculations.

Bridges the gap between raw UI DataFrames and the domain objects
(Radiator, Circuit, Collector). Also owns validation and table-initialisation
helpers that the UI callbacks depend on.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import pandas as pd

from domain.radiator import POSSIBLE_DIAMETERS, Radiator
from domain.hydraulics import Circuit, Collector

# ---------------------------------------------------------------------------
# Constants shared with domain (referenced here so callers import one place)
# ---------------------------------------------------------------------------
EXPONENT_RADIATOR: float = 1.34
DELTA_T_REF:       float = (75.0 + 65.0) / 2.0 - 20.0   # 50 K
AVAILABLE_RADIATOR_POWERS: List[int] = [2000, 2500, 3000, 3500, 4000]


# ---------------------------------------------------------------------------
# Weighted ΔT
# ---------------------------------------------------------------------------

def calculate_weighted_delta_t(
    radiators: List[Radiator],
    radiator_df: pd.DataFrame,
) -> float:
    """Flow-weighted system ΔT [K]. Returns 0 when total flow is zero."""
    total_flow = sum(r.mass_flow_rate for r in radiators)
    if total_flow <= 0:
        return 0.0
    weighted = sum(
        row["Mass flow rate"] * (row["Supply Temperature"] - row["Return Temperature"])
        for (_, row), _ in zip(radiator_df.iterrows(), radiators)
    )
    return weighted / total_flow


# ---------------------------------------------------------------------------
# Extra power at low-temperature operation
# ---------------------------------------------------------------------------

def calculate_extra_power_needed(
    radiator_power: float,
    heat_loss: float,
    supply_temp: float,
    delta_t: float,
    space_temperature: float,
) -> float:
    """
    Extra rated power (at 75/65/20) needed when an existing radiator operates
    below its reference condition.

    Returns 0.0 for invalid inputs or when the radiator is already adequate.
    """
    try:
        radiator_power    = float(radiator_power or 0.0)
        heat_loss         = float(heat_loss or 0.0)
        supply_temp       = float(supply_temp)
        delta_t           = float(delta_t)
        space_temperature = float(space_temperature)
    except (TypeError, ValueError):
        return 0.0

    if not all(map(math.isfinite, [radiator_power, heat_loss, supply_temp, delta_t, space_temperature])):
        return 0.0
    if delta_t <= 0 or radiator_power <= 0:
        return 0.0

    t_return      = supply_temp - delta_t
    delta_t_actual = (supply_temp + t_return) / 2.0 - space_temperature
    if delta_t_actual <= 0:
        return 0.0

    phi             = max(delta_t_actual / DELTA_T_REF, 1e-6)
    available_power = max(0.0, radiator_power) * (phi ** EXPONENT_RADIATOR)
    extra_actual    = max(0.0, heat_loss - available_power)
    return float(extra_actual / (phi ** EXPONENT_RADIATOR))


# ---------------------------------------------------------------------------
# Batch radiator calculation (MODE_FIXED / MODE_PUMP)
# ---------------------------------------------------------------------------

def calculate_radiator_data_with_extra_power(
    radiator_df: pd.DataFrame,
    delta_T: float = 3.0,
    supply_temp_input: Optional[float] = None,
    fixed_diameter: Optional[float] = None,
) -> pd.DataFrame:
    """
    Full hydraulics for fixed-temperature or pump-driven modes.

    Steps
    -----
    1. Extra power per radiator (normalised to 75/65/20).
    2. Build Radiator objects.
    3. Lock all to the highest required supply temperature.
    4. Re-derive return temperatures and mass flows.
    5. Select uniform pipe diameter (worst case).
    6. Calculate pressure losses.
    """
    radiator_df = radiator_df.copy()
    radiator_df["Extra radiator power"] = 0.0

    radiators: List[Radiator] = []
    for idx, row in radiator_df.iterrows():
        supply = supply_temp_input if supply_temp_input is not None \
                 else row["Space Temperature"] + delta_T
        extra = calculate_extra_power_needed(
            radiator_power    = row["Radiator power 75/65/20"],
            heat_loss         = row["Calculated heat loss"],
            supply_temp       = supply,
            delta_t           = delta_T,
            space_temperature = row["Space Temperature"],
        )
        radiator_df.at[idx, "Extra radiator power"] = extra
        adjusted_loss = row["Calculated heat loss"] + extra
        q_ratio = adjusted_loss / row["Radiator power 75/65/20"]

        radiators.append(Radiator(
            q_ratio           = q_ratio,
            delta_t           = delta_T,
            space_temperature = row["Space Temperature"],
            heat_loss         = adjusted_loss,
            supply_temperature = supply,
        ))

    max_supply = supply_temp_input if supply_temp_input is not None \
                 else max(r.supply_temperature for r in radiators)
    for r in radiators:
        r.supply_temperature  = max_supply
        r.return_temperature  = r._calc_return_temperature(max_supply)
        r.mass_flow_rate      = r._calc_mass_flow_rate()

    radiator_df["Supply Temperature"] = max_supply
    radiator_df["Return Temperature"] = [r.return_temperature for r in radiators]
    radiator_df["Mass flow rate"]     = [r.mass_flow_rate for r in radiators]

    uniform_diameter = max(
        r.calculate_diameter(POSSIBLE_DIAMETERS, fixed_diameter) for r in radiators
    )
    radiator_df["Diameter"] = uniform_diameter

    radiator_df["Pressure loss"] = [
        Circuit(
            length_circuit = row["Length circuit"],
            diameter       = uniform_diameter,
            mass_flow_rate = row["Mass flow rate"],
        ).calculate_pressure_radiator_kv()
        for _, row in radiator_df.iterrows()
    ]
    return radiator_df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_data(df: pd.DataFrame) -> bool:
    """
    Check that all required radiator columns are non-null and positive.
    Auto-corrects rated power when it falls below heat loss.
    Returns True when all checks pass.
    """
    required = [
        "Radiator power 75/65/20",
        "Calculated heat loss",
        "Length circuit",
        "Space Temperature",
    ]
    for col in required:
        if df[col].isnull().any() or (df[col] <= 0).any():
            return False

    for idx, row in df.iterrows():
        if row["Radiator power 75/65/20"] < row["Calculated heat loss"]:
            suggested = _suggest_radiator_power(row["Calculated heat loss"])
            print(
                f"Warning: Radiator {idx} rated power ({row['Radiator power 75/65/20']:.0f} W) "
                f"< heat loss ({row['Calculated heat loss']:.0f} W). "
                f"Auto-correcting to {suggested} W."
            )
            df.at[idx, "Radiator power 75/65/20"] = suggested
    return True


# ---------------------------------------------------------------------------
# Table initialisation helpers  (used by UI callbacks)
# ---------------------------------------------------------------------------

def init_radiator_rows(
    n: int,
    collector_options: List[str],
    room_options: List[Any],
) -> List[Dict[str, Any]]:
    """Build n default radiator rows."""
    return [
        {
            "Radiator nr":              i,
            "Collector":                collector_options[0] if collector_options else "Collector 1",
            "Radiator power 75/65/20":  0.0,
            "Length circuit":           0.0,
            "Space Temperature":        20.0,
            "Electric power":           0.0,
            "Room":                     room_options[(i - 1) % max(1, len(room_options))]
                                        if room_options else 1,
        }
        for i in range(1, n + 1)
    ]


def resize_radiator_rows(
    current: List[Dict[str, Any]],
    desired_num: int,
    collector_options: List[str],
    room_options: List[Any],
) -> List[Dict[str, Any]]:
    """Grow or shrink the radiator list to desired_num, preserving existing data."""
    rows = (current or []).copy()
    if desired_num > len(rows):
        rows.extend(init_radiator_rows(desired_num - len(rows), collector_options, room_options))
    rows = rows[:desired_num]
    for idx, r in enumerate(rows, start=1):
        r["Radiator nr"] = idx
    return rows


def init_collector_rows(n: int, start: int = 1) -> List[Dict[str, Any]]:
    """Build n default collector rows."""
    return [
        {"Collector": f"Collector {i}", "Collector circuit length": 0.0}
        for i in range(start, start + n)
    ]


def resize_collector_rows(
    current: List[Dict[str, Any]], desired: int
) -> List[Dict[str, Any]]:
    """Grow or shrink the collector list to desired."""
    rows = (current or []).copy()
    if desired > len(rows):
        rows.extend(init_collector_rows(desired - len(rows), start=len(rows) + 1))
    return rows[:desired]


def load_radiator_data(num_radiators: int, collector_options: List[str]) -> pd.DataFrame:
    """Return an initialised radiator DataFrame (all zeros / defaults)."""
    return pd.DataFrame({
        "Radiator nr":              list(range(1, num_radiators + 1)),
        "Collector":                [collector_options[0]] * num_radiators,
        "Radiator power 75/65/20":  [0.0] * num_radiators,
        "Calculated heat loss":     [0.0] * num_radiators,
        "Length circuit":           [0.0] * num_radiators,
        "Space Temperature":        [20.0] * num_radiators,
        "Extra power":              [0.0] * num_radiators,
    })


def load_collector_data(num_collectors: int) -> pd.DataFrame:
    """Return an initialised collector DataFrame."""
    return pd.DataFrame({
        "Collector":              [f"Collector {i + 1}" for i in range(num_collectors)],
        "Collector circuit length": [0.0] * num_collectors,
    })


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _suggest_radiator_power(heat_loss: float) -> Optional[int]:
    for power in sorted(AVAILABLE_RADIATOR_POWERS):
        if power > heat_loss + 100:
            return power
    return None
