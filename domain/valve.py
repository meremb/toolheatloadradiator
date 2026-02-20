"""
domain/valve.py
===============
Thermostatic radiator valve (TRV) sizing model.

Supports named valves from the built-in catalogue or a custom linear model.
No Dash / service / DataFrame dependencies (except for the kv-position
calculation which needs a merged radiator DataFrame).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from domain.hydraulics import HYDRAULIC_CONSTANT


class Valve:
    """
    TRV sizing: maps required kv to a discrete valve position.

    Parameters
    ----------
    kv_max     : Full-open kv for custom valves  [m³/h]
    n          : Number of positions for custom valves
    valve_name : Named valve from _CATALOGUE, or 'Custom'
    """

    # Class-level catalogue – not a dataclass field to avoid mutable default issues
    _CATALOGUE: Dict[str, dict] = {
        "Danfoss RA-N 10 (3/8)": {
            "positions": 8,
            "kv_values": [0.04, 0.08, 0.12, 0.19, 0.25, 0.33, 0.38, 0.56],
            "description": "Danfoss RA-N 10 (3/8) – 8-position TRV",
        },
        "Danfoss RA-N 15 (1/2)": {
            "positions": 8,
            "kv_values": [0.04, 0.08, 0.12, 0.20, 0.30, 0.40, 0.51, 0.73],
            "description": "Danfoss RA-N 15 (1/2) – 8-position TRV",
        },
        "Danfoss RA-N 20 (3/4)": {
            "positions": 8,
            "kv_values": [0.10, 0.15, 0.17, 0.26, 0.35, 0.46, 0.73, 1.04],
            "description": "Danfoss RA-N 20 (3/4) – 8-position TRV",
        },
        "Oventrop DN15 (1/2)": {
            "positions": 9,
            "kv_values": [0.05, 0.09, 0.14, 0.20, 0.26, 0.32, 0.43, 0.57, 0.67],
            "description": "Oventrop DN15 (1/2) – 9-position TRV",
        },
        "Heimeier (1/2)": {
            "positions": 8,
            "kv_values": [0.049, 0.09, 0.15, 0.265, 0.33, 0.47, 0.59, 0.67],
            "description": "Heimeier (1/2) – 8-position TRV",
        },
        "Vogel und Noot": {
            "positions": 5,
            "kv_values": [0.13, 0.30, 0.43, 0.58, 0.75],
            "description": "Vogel und Noot – 5-position TRV",
        },
        "Comap": {
            "positions": 6,
            "kv_values": [0.028, 0.08, 0.125, 0.24, 0.335, 0.49],
            "description": "Comap – 6-position TRV",
        },
    }

    def __init__(
        self,
        kv_max: float = 0.7,
        n: int = 100,
        valve_name: Optional[str] = None,
    ) -> None:
        self.kv_max     = kv_max
        self.n          = n
        self.valve_name = valve_name
        # Backward-compatible alias used in app.py
        self.VALVE_CONFIGS = self._CATALOGUE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def get_valve_names(cls) -> List[str]:
        """All valve names, starting with 'Custom'."""
        return ["Custom"] + list(cls._CATALOGUE.keys())

    def get_config(self) -> Optional[dict]:
        """Catalogue entry for the selected valve, or None for Custom."""
        if not self.valve_name or self.valve_name == "Custom":
            return None
        return self._CATALOGUE.get(self.valve_name)

    def get_kv_at_position(self, position: int) -> float:
        """kv [m³/h] at a given discrete position."""
        config = self.get_config()
        if config:
            idx = min(position, len(config["kv_values"]) - 1)
            return config["kv_values"][idx]
        return (position / (self.n - 1)) * self.kv_max if self.n > 1 else 0.0

    def calculate_pressure_valve_kv(self, mass_flow_rate: float) -> float:
        """Pressure loss across fully-open valve [Pa]."""
        config = self.get_config()
        kv = config["kv_values"][-1] if config else self.kv_max
        if kv <= 0:
            return float("inf")
        return round(HYDRAULIC_CONSTANT * (mass_flow_rate / 1000.0 / kv) ** 2, 1)

    def calculate_kv_position_valve(
        self,
        merged_df: pd.DataFrame,
        custom_kv_max: Optional[float] = None,
        n: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Add 'Valve position' (and 'Valve kv' for catalogue valves) to merged_df.

        Catalogue valves: step-search for first kv that meets requirement.
        Custom valves:    polynomial or ratio-based estimation.
        """
        merged_df = self._calc_kv_needed(merged_df)
        config    = self.get_config()

        if config:
            kv_values = np.array(config["kv_values"])
            merged_df["Valve position"] = merged_df["kv_needed"].apply(
                lambda kv: _find_valve_position(kv, kv_values)
            )
            merged_df["Valve kv"] = merged_df["Valve position"].apply(
                lambda pos: kv_values[pos]
            )
        else:
            if custom_kv_max is not None and n is not None:
                self.kv_max = custom_kv_max
                self.n      = n
                positions = self._adjust_position_custom(merged_df["kv_needed"].to_numpy())
            else:
                a, b, c = 0.0114, -0.0086, 0.0446
                positions = np.ceil(_solve_valve_polynomial(a, b, c, merged_df["kv_needed"].to_numpy()))
            merged_df["Valve position"] = positions.flatten()

        return merged_df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _calc_kv_needed(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """Required kv to balance each circuit."""
        merged_df = merged_df.copy()
        merged_df["Total pressure valve circuit"] = (
            merged_df["Total Pressure Loss"] + merged_df["Valve pressure loss N"]
        )
        max_p = merged_df["Total pressure valve circuit"].max()
        merged_df["Pressure difference valve"] = max_p - merged_df["Total Pressure Loss"]
        merged_df["kv_needed"] = (merged_df["Mass flow rate"] / 1000.0) / (
            (merged_df["Pressure difference valve"] / 100_000).clip(lower=1e-9) ** 0.5
        )
        return merged_df

    def _adjust_position_custom(self, kv_needed: np.ndarray) -> np.ndarray:
        ratio_kv       = np.clip(kv_needed / self.kv_max, 0, 1)
        ratio_position = np.clip(np.sqrt(ratio_kv), 0, 1)
        return np.ceil(ratio_position * self.n)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _find_valve_position(kv_needed: float, kv_values: np.ndarray) -> int:
    """Index of the first kv_value ≥ kv_needed."""
    for i, kv in enumerate(kv_values):
        if kv >= kv_needed:
            return i
    return len(kv_values) - 1


def _solve_valve_polynomial(
    a: float, b: float, c: float, kv_needed: np.ndarray
) -> np.ndarray:
    """Solve quadratic a·x² + b·x + c = kv_needed for x (valve position ratio)."""
    disc = np.maximum(b ** 2 - 4 * a * (c - kv_needed), 0.0)
    root = (-b + np.sqrt(disc)) / (2 * a)
    return np.where(disc <= 0, 0.1, root)
