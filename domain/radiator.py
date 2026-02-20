"""
domain/radiator.py
==================
Radiator thermal and hydraulic model (EN 442).

Zero Dash / pandas / service-layer dependencies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
T_FACTOR: float = 49.83          # Characteristic temperature factor [K]
EXPONENT_RADIATOR: float = 1.34  # Radiator exponent n (–)

POSSIBLE_DIAMETERS: List[int] = [8, 10, 12, 13, 14, 16, 20, 22, 25, 28, 36, 50]  # mm


@dataclass
class Radiator:
    """
    Thermal and hydraulic model for a single radiator.

    Parameters
    ----------
    q_ratio            : Actual / rated heat output (–)
    delta_t            : System ΔT (supply – return)  [K]
    space_temperature  : Room set-point               [°C]
    heat_loss          : Room design heat loss         [W]
    supply_temperature : Override; calculated if None  [°C]
    """

    q_ratio: float
    delta_t: float
    space_temperature: float
    heat_loss: float
    supply_temperature: float = field(default=None)
    return_temperature: float = field(init=False)
    mass_flow_rate: float     = field(init=False)

    def __post_init__(self) -> None:
        if self.supply_temperature is None:
            self.supply_temperature = self._calc_supply_temperature()
        self.return_temperature = self._calc_return_temperature(self.supply_temperature)
        self.mass_flow_rate     = self._calc_mass_flow_rate()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_diameter(
        self,
        possible_diameters: List[int] = POSSIBLE_DIAMETERS,
        fixed_diameter: Optional[float] = None,
    ) -> float:
        """Smallest standard pipe diameter adequate for the mass flow [mm]."""
        if fixed_diameter is not None:
            return float(fixed_diameter)
        if math.isnan(self.mass_flow_rate):
            raise ValueError("Mass flow rate is NaN – check collector configuration.")
        if self.mass_flow_rate < 0:
            raise ValueError(
                f"Negative mass flow rate ({self.mass_flow_rate:.1f} kg/h). "
                "Increase radiator power, ΔT, or supply temperature."
            )
        return _select_pipe_diameter(self.mass_flow_rate, possible_diameters)

    # Aliases kept for backward compatibility
    def calculate_tsupply(self)           -> float: return self._calc_supply_temperature()
    def calculate_treturn(self, ts: float) -> float: return self._calc_return_temperature(ts)
    def calculate_mass_flow_rate(self)    -> float: return self._calc_mass_flow_rate()

    # ------------------------------------------------------------------
    # Private calculations
    # ------------------------------------------------------------------

    def _calc_c(self) -> float:
        """Intermediate constant relating q_ratio to the temperature lift."""
        if self.q_ratio <= 0:
            return float("inf")
        return math.exp(self.delta_t / T_FACTOR / self.q_ratio ** (1.0 / EXPONENT_RADIATOR))

    def _calc_supply_temperature(self) -> float:
        c = self._calc_c()
        if c <= 1:
            return round(self.space_temperature + max(self.delta_t, 3.0), 1)
        return round(self.space_temperature + (c / (c - 1)) * self.delta_t, 1)

    def _calc_return_temperature(self, supply: float) -> float:
        lift = supply - self.space_temperature
        if lift <= 0:
            return round(supply, 1)
        t_return = (
            (self.q_ratio ** (1.0 / EXPONENT_RADIATOR) * T_FACTOR) ** 2 / lift
            + self.space_temperature
        )
        return round(t_return, 1)

    def _calc_mass_flow_rate(self) -> float:
        d_t = max(self.supply_temperature - self.return_temperature, 0.1)
        return round(self.heat_loss / 4180.0 / d_t * 3600.0, 1)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _select_pipe_diameter(mass_flow_rate: float, diameters: List[int]) -> float:
    """Pick the smallest standard diameter [mm] adequate for mass_flow_rate."""
    min_d = 1.4641 * mass_flow_rate ** 0.4217
    candidates = [d for d in diameters if d >= min_d]
    if not candidates:
        raise ValueError(
            f"Mass flow {mass_flow_rate:.1f} kg/h exceeds all standard diameters. "
            "Consider increasing ΔT or splitting into parallel radiators."
        )
    return float(min(candidates, key=lambda d: abs(d - min_d)))
