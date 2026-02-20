"""
domain/heat_load.py
===================
Room-level heat-load calculator following EN 12831 / EPB simplified approach.

All U-values in W/(m²·K), temperatures in °C, heat losses in W.
This module has zero dependencies on Dash, pandas, or any service layer.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, Optional, Union

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
BRIDGE_CORRECTION: float = 0.05          # Linear thermal bridge addition  [W/(m²·K)]
GROUND_TEMP_DEFAULT: float = 10.0        # Assumed ground temperature       [°C]
GROUND_CORRECTION_FACTOR: float = 1.15 * 1.45  # EN 12831 ground area factors (combined)
INFILTRATION_FACTOR: float = 0.34        # ρ·c_p for air  [Wh/(m³·K)]
WALL_OFFSET: float = 0.3                 # Half-thickness for gross area estimation [m]

VENTILATION_ACH: Dict[str, float] = {
    "C": 0.5,        # Mechanical extraction – full outdoor air
    "D": 0.5 * 0.3,  # Balanced – 30 % of nominal (heat-recovery credit)
}


@dataclasses.dataclass
class RoomLoadCalculator:
    """
    Design heat load for a single room.

    Required
    --------
    floor_area  : Net floor area            [m²]
    uw          : Wall U-value              [W/(m²·K)]
    u_roof      : Roof U-value              [W/(m²·K)]
    u_ground    : Ground-floor U-value      [W/(m²·K)]
    v_system    : Ventilation type ('C'|'D')

    Key defaults (Belgian climate / typical construction)
    -----------------------------------------------------
    tin=20, tout=-7, wall_height=3.0, v50=6.0, lir=0.1
    """

    # --- mandatory ---
    floor_area: float
    uw: float
    u_roof: float
    u_ground: float
    v_system: str

    # --- envelope defaults ---
    wall_outside: float = 2.0
    v50: float = 6.0
    tin: float = 20.0
    tout: float = -7.0
    tattic: float = 10.0
    neighbour_t: float = 18.0
    un: float = 2.0
    u_glass: float = 1.0
    lir: float = 0.1
    wall_height: float = 3.0

    # --- geometry flags ---
    window: bool = False
    on_ground: bool = False
    under_roof: bool = False
    under_insulated_attic: bool = False

    # --- neighbour losses ---
    add_neighbour_losses: bool = False
    neighbour_perimeter: float = 0.0

    # --- area estimation ---
    heat_loss_area_estimation: str = "fromFloorArea"
    exposed_perimeter: float = 0.0

    # --- ventilation ---
    ventilation_calculation_method: str = "simple"
    room_type: Optional[str] = None

    # --- output ---
    return_detail: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self) -> Union[float, Dict[str, float]]:
        """Return total design heat load [W], or a detailed breakdown dict."""
        delta_t = self.tin - self.tout
        areas = self._compute_heat_loss_areas()

        transmission = self._compute_transmission_loss(areas, delta_t)
        ventilation  = self._compute_ventilation_loss(delta_t)
        infiltration = self._compute_infiltration_loss(areas, delta_t)
        neighbour    = self._compute_neighbour_loss(areas)
        attic        = self._compute_attic_loss(areas)

        air_loss = max(ventilation, infiltration)  # EN 12831: take the larger
        total = transmission + air_loss + neighbour + attic

        return self._format_result(total, transmission, ventilation, infiltration, neighbour, attic)

    # ------------------------------------------------------------------
    # Loss components
    # ------------------------------------------------------------------

    def _compute_transmission_loss(self, areas: Dict[str, float], delta_t: float) -> float:
        walls  = areas["walls"] * (self.uw + BRIDGE_CORRECTION)
        roof   = areas["roof"]  * (self.u_roof + BRIDGE_CORRECTION)
        ground = (
            areas["ground"]
            * GROUND_CORRECTION_FACTOR
            * (self.u_ground + BRIDGE_CORRECTION)
            * (self.tin - GROUND_TEMP_DEFAULT)
        )
        loss = (walls + roof) * delta_t + ground
        if self.window:
            window_fraction = areas["walls"] * 0.2
            loss += window_fraction * (self.u_glass - self.uw) * delta_t
        return loss

    def _compute_ventilation_loss(self, delta_t: float) -> float:
        flows = self._get_ventilation_flows()
        delta_t_neighbour = max(0.0, self.tin - self.neighbour_t)
        return INFILTRATION_FACTOR * (
            flows["outdoor"]   * delta_t +
            flows["neighbour"] * delta_t_neighbour
        )

    def _compute_infiltration_loss(self, areas: Dict[str, float], delta_t: float) -> float:
        envelope = areas["walls"] + areas["roof"] + areas["ground"]
        return INFILTRATION_FACTOR * self.lir * self.v50 * envelope * delta_t

    def _compute_neighbour_loss(self, areas: Dict[str, float]) -> float:
        if not self.add_neighbour_losses:
            return 0.0
        total_area = areas["neighbours"] + areas["neighbourfloor"] + areas["neighbourceiling"]
        return self.un * max(0.0, self.tin - self.neighbour_t) * total_area

    def _compute_attic_loss(self, areas: Dict[str, float]) -> float:
        return areas["attic"] * self.un * (self.tin - self.tattic)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _compute_heat_loss_areas(self) -> Dict[str, float]:
        e = WALL_OFFSET
        gross = self.floor_area + 4 * e * np.sqrt(self.floor_area) + 4 * e ** 2

        if self.heat_loss_area_estimation == "fromFloorArea":
            side          = np.sqrt(gross)
            wall_external = side * self.wall_height * self.wall_outside
            wall_neighbour = side * self.wall_height * (4.0 - self.wall_outside)
        elif self.heat_loss_area_estimation == "fromExposedPerimeter":
            wall_external  = self.exposed_perimeter  * self.wall_height
            wall_neighbour = self.neighbour_perimeter * self.wall_height
        else:
            wall_external = wall_neighbour = 0.0

        ground_area = gross if self.on_ground           else 0.0
        roof_area   = gross if self.under_roof          else 0.0
        attic_area  = gross if self.under_insulated_attic else 0.0

        neighbour_floor   = 0.0 if ground_area               else self.floor_area
        neighbour_ceiling = 0.0 if (roof_area or attic_area) else self.floor_area

        return {
            "walls":           wall_external,
            "neighbours":      wall_neighbour,
            "ground":          ground_area,
            "roof":            roof_area,
            "attic":           attic_area,
            "neighbourfloor":  neighbour_floor,
            "neighbourceiling": neighbour_ceiling,
        }

    # ------------------------------------------------------------------
    # Ventilation flows
    # ------------------------------------------------------------------

    def _get_ventilation_flows(self) -> Dict[str, float]:
        if self.ventilation_calculation_method == "simple":
            return self._simple_ventilation_flows()
        if self.ventilation_calculation_method == "NBN-D-50-001":
            return self._detailed_ventilation_flows()
        return {"outdoor": 0.0, "neighbour": 0.0}

    def _simple_ventilation_flows(self) -> Dict[str, float]:
        ach    = VENTILATION_ACH.get(self.v_system, 0.0)
        volume = self.floor_area * self.wall_height
        return {"outdoor": volume * ach, "neighbour": 0.0}

    def _detailed_ventilation_flows(self) -> Dict[str, float]:
        bounds: Dict[Optional[str], Dict[str, float]] = {
            "Living":   {"min": 75,  "max": 150},
            "Kitchen":  {"min": 50,  "max": 75},
            "Bedroom":  {"min": 25,  "max": 72},
            "Study":    {"min": 25,  "max": 72},
            "Laundry":  {"min": 50,  "max": 75},
            "Bathroom": {"min": 50,  "max": 150},
            "Toilet":   {"min": 25,  "max": 25},
            "Hallway":  {"min": 0,   "max": 75},
            None:       {"min": 0,   "max": 150},
        }
        b = bounds.get(self.room_type, bounds[None])
        nom_flow = float(np.clip(3.6 * self.floor_area, b["min"], b["max"]))

        supply_rooms = {"Living", "Bedroom", "Bureau", None}
        if self.room_type in supply_rooms:
            outdoor = nom_flow * (0.3 if self.v_system == "D" else 1.0)
            return {"outdoor": outdoor, "neighbour": 0.0}
        return {"outdoor": 0.0, "neighbour": nom_flow}

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    def _format_result(
        self,
        total: float, transmission: float,
        ventilation: float, infiltration: float,
        neighbour: float, attic: float,
    ) -> Union[float, Dict[str, float]]:
        if self.return_detail:
            return {
                "totalHeatLoss":        total,
                "transmissionHeatLoss": transmission,
                "ventilationHeatLoss":  ventilation,
                "infiltrationHeatLoss": infiltration,
                "neighbourLosses":      neighbour,
                "atticLosses":          attic,
            }
        return float(np.round(total, 0))
