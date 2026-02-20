"""
domain/valve_override.py
========================
Hydraulic network re-solver for valve-position overrides.

Physics
-------
Each radiator circuit has a combined hydraulic resistance:

    ΔP = R_circuit · ṁ²

where R_circuit = R_pipe + R_valve + R_radiator_body (all in Pa·(h/kg)²).

All circuits share the *same available differential pressure* ΔP_sys
(the pump / boiler delivers one pressure that serves every branch in parallel).

For a given set of valve positions the solve proceeds as:

1.  Each valve position → kv  [m³/h]  from the catalogue (or custom model).
2.  kv → R_valve              [Pa·(h/kg)²]  via  R = C / kv²
3.  R_pipe is recalculated from the (unchanged) circuit length & diameter.
4.  ΔP_sys = max total circuit resistance × ṁ²  (the index / most resistive
    circuit already has its valve fully open, so it sets the system pressure).
    We iterate: guess ΔP_sys → solve all ṁ → recompute collector flows →
    recompute collector pressure losses → repeat until convergence.
5.  New ṁ → new return temperature and heat output via EN 442.

No Dash / callback dependencies.  Import and call `recalculate_with_overrides`.
"""
from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from domain.hydraulics import (
    Circuit,
    Collector,
    HYDRAULIC_CONSTANT,
    KV_RADIATOR,
    LOCAL_LOSS_COEFFICIENT,
    PRESSURE_LOSS_BOILER,
    KV_PIPE_A, KV_PIPE_B, KV_PIPE_C,
)
from domain.radiator import T_FACTOR, EXPONENT_RADIATOR
from domain.valve import Valve, _find_valve_position

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_ITERATIONS: int   = 60
TOLERANCE_PA:   float = 0.5      # Pa — convergence criterion for ΔP_sys
MIN_FLOW_KGH:   float = 0.01     # kg/h — floor to avoid divide-by-zero


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def recalculate_with_overrides(
    merged_df:        pd.DataFrame,
    collector_df:     pd.DataFrame,
    overrides:        Dict[int, int],          # {radiator_nr: valve_position}
    valve:            Valve,
    delta_t:          float,                   # system ΔT  [K]
    space_temps:      Dict[str, float],        # room_name → space temperature [°C]
    supply_temp:      Optional[float] = None,  # fixed supply temperature [°C] or None
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Re-solve the hydraulic network after user valve-position overrides.

    Parameters
    ----------
    merged_df     : Full radiator results DataFrame (output of the normal calc).
                    Must contain columns:
                      'Radiator nr', 'Room', 'Collector', 'Length circuit',
                      'Diameter', 'Mass flow rate', 'Calc. Position' (or 'Valve position'),
                      'Pressure loss', 'Total Pressure Loss',
                      'Radiator power 75/65/20', 'Supply temperature'
    collector_df  : Collector DataFrame with 'Collector', 'Collector circuit length',
                    'Diameter', 'Collector pressure loss'
    overrides     : Mapping of radiator number → new valve position (1-based).
                    Radiators not in this dict keep their auto-calculated position.
    valve         : Valve instance (catalogue or custom).
    delta_t       : System design ΔT  [K].
    space_temps   : Room name → space (room) temperature [°C].
    supply_temp   : Fixed supply temperature.  If None it is taken from merged_df.

    Returns
    -------
    (updated_rad_df, updated_col_df, warnings_list)
    """
    logs: list[str] = []
    rad  = merged_df.copy().reset_index(drop=True)

    # ------------------------------------------------------------------
    # Step 1 — Determine effective valve position for every radiator
    # ------------------------------------------------------------------
    # The auto-calculated position column may be named differently
    pos_col = "Calc. Position" if "Calc. Position" in rad.columns else "Valve position"

    rad["_eff_position"] = rad.apply(
        lambda row: overrides.get(int(row["Radiator nr"]), int(row.get(pos_col, 1))),
        axis=1,
    )

    # ------------------------------------------------------------------
    # Step 2 — kv at each effective position
    # ------------------------------------------------------------------
    rad["_kv_valve"] = rad["_eff_position"].apply(lambda pos: _kv_at_pos(valve, pos))

    # ------------------------------------------------------------------
    # Step 3 — Per-circuit hydraulic resistance coefficients
    #          ΔP [Pa] = R · (ṁ [kg/h])²
    # ------------------------------------------------------------------
    rad["_R_pipe"]  = rad.apply(_pipe_resistance, axis=1)
    rad["_R_valve"] = rad["_kv_valve"].apply(_valve_resistance_from_kv)
    rad["_R_rad"]   = _radiator_body_resistance()        # scalar, same for all
    rad["_R_total"] = rad["_R_pipe"] + rad["_R_valve"] + rad["_R_rad"]

    # ------------------------------------------------------------------
    # Step 4 — Iterative network solve
    #
    # Topology: parallel radiator circuits feed into series collector legs.
    # The system pressure available to each radiator circuit is:
    #
    #   ΔP_available(i) = ΔP_sys − ΔP_collectors_upstream(i) − ΔP_boiler
    #
    # We iterate:
    #   a. Guess ΔP_sys from current flows (or from the initial data).
    #   b. For each radiator, ṁ = sqrt(ΔP_available / R_total).
    #   c. Sum flows per collector → new collector ΔP.
    #   d. New ΔP_sys = max(ΔP_circuit_total) — the index circuit
    #      (most resistive) determines the pump working point.
    #   Repeat until ΔP_sys converges.
    # ------------------------------------------------------------------

    col_df  = collector_df.copy().reset_index(drop=True)
    col_names = _sorted_collector_names(col_df)

    # Build collector series resistance lookup {name: R_collector}
    col_R = _collector_resistances(col_df)

    # Initial ΔP_sys guess: use the max "Total Pressure Loss" from original calc
    dp_sys = float(rad["Total Pressure Loss"].max()) + float(
        rad["Valve pressure loss N"].max() if "Valve pressure loss N" in rad.columns else 0.0
    )
    dp_sys = max(dp_sys, 500.0)   # safety floor

    flows = rad["Mass flow rate"].copy().to_numpy(dtype=float)

    for iteration in range(MAX_ITERATIONS):
        # Collector flows from current radiator flows
        col_flows = _sum_collector_flows(rad, flows, col_names)

        # Collector pressure losses (pipe + collector body)
        col_dp = _collector_pressure_drops(col_df, col_flows, col_names, col_R)

        # Cumulative downstream collector pressure for each radiator
        rad["_dp_upstream"] = rad["Collector"].apply(
            lambda c: _cumulative_upstream_dp(c, col_names, col_dp)
        )

        # Available pressure for each radiator circuit
        dp_available = dp_sys - rad["_dp_upstream"] - PRESSURE_LOSS_BOILER
        dp_available = np.maximum(dp_available.to_numpy(), 0.0)

        # New flows
        new_flows = np.sqrt(np.maximum(dp_available / rad["_R_total"].to_numpy(), 0.0))
        new_flows = np.maximum(new_flows, MIN_FLOW_KGH)

        # New system pressure: the most restrictive circuit sets ΔP_sys
        # ΔP_circuit_total = R_total * ṁ² + upstream_collector_losses + boiler
        circuit_total_dp = (
            rad["_R_total"].to_numpy() * new_flows ** 2
            + rad["_dp_upstream"].to_numpy()
            + PRESSURE_LOSS_BOILER
        )
        new_dp_sys = float(np.max(circuit_total_dp))

        # Convergence check
        if abs(new_dp_sys - dp_sys) < TOLERANCE_PA and np.allclose(new_flows, flows, rtol=1e-4):
            flows  = new_flows
            dp_sys = new_dp_sys
            logs.append(f"Network converged in {iteration + 1} iterations. ΔP_sys = {dp_sys:.0f} Pa")
            break

        flows  = new_flows
        dp_sys = new_dp_sys
    else:
        logs.append(
            f"Warning: network did not fully converge after {MAX_ITERATIONS} iterations. "
            f"Last ΔP_sys = {dp_sys:.0f} Pa — results are approximate."
        )

    rad["Mass flow rate"] = np.round(flows, 1)

    # ------------------------------------------------------------------
    # Step 5 — Recalculate thermal quantities with new flows
    # ------------------------------------------------------------------
    rad = _recalculate_temperatures(rad, space_temps, delta_t, supply_temp, logs)

    # ------------------------------------------------------------------
    # Step 6 — Update pressure loss columns
    # ------------------------------------------------------------------
    rad["Pressure loss"] = rad.apply(
        lambda row: _circuit_pressure_loss(row), axis=1
    )
    rad["Valve pressure loss N"] = rad.apply(
        lambda row: round(
            HYDRAULIC_CONSTANT * (row["Mass flow rate"] / 1000.0 / max(row["_kv_valve"], 1e-9)) ** 2, 1
        ),
        axis=1,
    )

    # Rebuild total pressure loss with new collector losses
    col_flows_final  = _sum_collector_flows(rad, rad["Mass flow rate"].to_numpy(), col_names)
    col_dp_final     = _collector_pressure_drops(col_df, col_flows_final, col_names, col_R)

    rad["_dp_upstream"] = rad["Collector"].apply(
        lambda c: _cumulative_upstream_dp(c, col_names, col_dp_final)
    )
    rad["Total Pressure Loss"] = (
        rad["Pressure loss"]
        + rad["_dp_upstream"]
        + PRESSURE_LOSS_BOILER
    ).round(1)

    # Expose override info
    rad["Position Override"] = rad["Radiator nr"].apply(
        lambda nr: overrides.get(int(nr), None)
    )
    rad["Valve position"] = rad["_eff_position"]
    rad["Valve kv"]       = rad["_kv_valve"].round(4)

    # Update collector summary
    col_df["Mass flow rate"]         = [col_flows_final.get(n, 0.0) for n in col_names]
    col_df["Collector pressure loss"] = [col_dp_final.get(n, 0.0)   for n in col_names]

    # Drop working columns
    drop_cols = [c for c in rad.columns if c.startswith("_")]
    rad = rad.drop(columns=drop_cols)

    return rad, col_df, logs


# ---------------------------------------------------------------------------
# Thermal recalculation  (EN 442)
# ---------------------------------------------------------------------------

def _recalculate_temperatures(
    rad:         pd.DataFrame,
    space_temps: Dict[str, float],
    delta_t:     float,
    supply_temp: Optional[float],
    logs:        list[str],
) -> pd.DataFrame:
    """
    Recalculate supply temperature, return temperature and actual heat output
    for each radiator given new mass flow rates.

    EN 442 defines the heat output as:
        Q = Q_nom · (ΔT_m / ΔT_m_nom)^n

    where ΔT_m is the log-mean temperature difference between radiator water
    and room air.  We invert this to find the new ΔT_m, then derive T_return.

    Since supply temperature may be fixed or circuit-specific, we use the
    original supply temperature per radiator and only update T_return + Q_actual.
    """
    supply_temps   = []
    return_temps   = []
    actual_outputs = []

    for _, row in rad.iterrows():
        room        = str(row.get("Room", ""))
        t_room      = float(space_temps.get(room, 20.0))
        q_nom       = float(row.get("Radiator power 75/65/20", 1000.0))
        m_dot       = float(row["Mass flow rate"])          # kg/h
        t_sup       = float(supply_temp if supply_temp is not None
                            else row.get("Supply temperature", t_room + delta_t + 5))

        # Specific heat capacity of water  [J/(kg·K)]  →  [W per (kg/h · K)]
        cp = 4180.0 / 3600.0    # ≈ 1.161 W·h/(kg·K)

        # Nominal conditions (75/65/20): ΔT_m_nom via LMTD
        t_sup_nom, t_ret_nom, t_room_nom = 75.0, 65.0, 20.0
        lmtd_nom = _lmtd(t_sup_nom, t_ret_nom, t_room_nom)

        if m_dot < MIN_FLOW_KGH:
            # No flow → no heat output
            supply_temps.append(round(t_sup, 1))
            return_temps.append(round(t_sup, 1))
            actual_outputs.append(0.0)
            logs.append(f"  Radiator {int(row['Radiator nr'])}: zero flow — heat output = 0 W")
            continue

        # Iterate to find Q_actual and T_return that are self-consistent:
        #   Q = m_dot · cp · (T_sup − T_ret)          [energy balance]
        #   Q = Q_nom · (LMTD / LMTD_nom)^n           [EN 442]
        # Start with a guess for T_ret and iterate.
        t_ret = t_sup - delta_t     # initial guess

        for _ in range(30):
            lmtd = _lmtd(t_sup, t_ret, t_room)
            if lmtd <= 0:
                t_ret = t_sup - 0.1
                break
            q_calc   = q_nom * (lmtd / lmtd_nom) ** EXPONENT_RADIATOR
            t_ret_new = t_sup - q_calc / (m_dot * cp)
            if abs(t_ret_new - t_ret) < 0.01:
                t_ret = t_ret_new
                break
            t_ret = t_ret_new

        lmtd     = _lmtd(t_sup, t_ret, t_room)
        q_actual = q_nom * (lmtd / lmtd_nom) ** EXPONENT_RADIATOR if lmtd > 0 else 0.0

        supply_temps.append(round(t_sup, 1))
        return_temps.append(round(t_ret, 1))
        actual_outputs.append(round(q_actual, 1))

    rad = rad.copy()
    rad["Supply temperature"]  = supply_temps
    rad["Return temperature"]  = return_temps
    rad["Actual heat output (W)"] = actual_outputs
    return rad


def _lmtd(t_sup: float, t_ret: float, t_room: float) -> float:
    """Log-mean temperature difference between radiator and room [K]."""
    dt1 = t_sup - t_room
    dt2 = t_ret - t_room
    if dt1 <= 0 or dt2 <= 0:
        return 0.0
    if abs(dt1 - dt2) < 1e-6:
        return dt1
    return (dt1 - dt2) / math.log(dt1 / dt2)


# ---------------------------------------------------------------------------
# Hydraulic helpers
# ---------------------------------------------------------------------------

def _kv_at_pos(valve: Valve, position: int) -> float:
    """kv [m³/h] for a given 1-based valve position."""
    config = valve.get_config()
    if config:
        idx = min(max(int(position) - 1, 0), len(config["kv_values"]) - 1)
        return float(config["kv_values"][idx])
    # Custom valve: linear kv model
    n   = max(valve.n - 1, 1)
    pos = min(max(int(position) - 1, 0), n)
    return (pos / n) * valve.kv_max


def _pipe_resistance(row) -> float:
    """
    Circuit pipe resistance coefficient R such that ΔP_pipe = R · ṁ² [Pa/(kg/h)²].

    Based on Circuit.calculate_pressure_loss_piping():
        ΔP = C / kv_pipe² · L · 2 · LOCAL_LOSS_COEFFICIENT · ṁ²
    where C = HYDRAULIC_CONSTANT / 1e6  (converting kg/h → kg/s internally)
    """
    d_mm   = float(row.get("Diameter", 12))
    L      = float(row.get("Length circuit", 10))
    d_m    = d_mm / 1000.0
    kv_pipe = KV_PIPE_A * d_m ** 2 + KV_PIPE_B * d_m + KV_PIPE_C
    # ΔP = HYDRAULIC_CONSTANT · (ṁ/1000 / kv)² · L · 2 · LCF
    # = HYDRAULIC_CONSTANT / (1000² · kv²) · L · 2 · LCF  · ṁ²
    return HYDRAULIC_CONSTANT / (1_000_000.0 * kv_pipe ** 2) * L * 2.0 * LOCAL_LOSS_COEFFICIENT


def _valve_resistance_from_kv(kv: float) -> float:
    """R_valve = HYDRAULIC_CONSTANT / (1000² · kv²)  [Pa/(kg/h)²]."""
    if kv <= 0:
        return 1e12   # effectively infinite resistance (closed valve)
    return HYDRAULIC_CONSTANT / (1_000_000.0 * kv ** 2)


# Patch the usage — reassign in the main function
def _radiator_body_resistance() -> float:
    """Fixed resistance of the radiator body (kv = KV_RADIATOR)."""
    return _valve_resistance_from_kv(KV_RADIATOR)


def _circuit_pressure_loss(row) -> float:
    """Pipe + radiator body pressure loss [Pa] at current mass flow."""
    m = float(row["Mass flow rate"])
    R = float(row["_R_pipe"]) + _radiator_body_resistance()
    return round(R * m ** 2, 1)


def _sorted_collector_names(col_df: pd.DataFrame) -> List[str]:
    return list(col_df.sort_values("Collector")["Collector"])


def _collector_resistances(col_df: pd.DataFrame) -> Dict[str, float]:
    """
    For each collector: pipe resistance for the collector's own circuit leg.
    Uses the same polynomial kv model as for radiator pipes.
    """
    result = {}
    for _, row in col_df.iterrows():
        name  = row["Collector"]
        d_mm  = float(row.get("Diameter", 16))
        L     = float(row.get("Collector circuit length", 5))
        d_m   = d_mm / 1000.0
        kv_p  = KV_PIPE_A * d_m ** 2 + KV_PIPE_B * d_m + KV_PIPE_C
        R     = HYDRAULIC_CONSTANT / (1_000_000.0 * kv_p ** 2) * L * 2.0 * LOCAL_LOSS_COEFFICIENT
        result[name] = R
    return result


def _sum_collector_flows(
    rad: pd.DataFrame,
    flows: np.ndarray,
    col_names: List[str],
) -> Dict[str, float]:
    totals = {n: 0.0 for n in col_names}
    for i, (_, row) in enumerate(rad.iterrows()):
        c = row["Collector"]
        if c in totals:
            totals[c] += float(flows[i])
    return totals


def _collector_pressure_drops(
    col_df:    pd.DataFrame,
    col_flows: Dict[str, float],
    col_names: List[str],
    col_R:     Dict[str, float],
) -> Dict[str, float]:
    """Pressure drop for each individual collector leg [Pa]."""
    result = {}
    for name in col_names:
        m = col_flows.get(name, 0.0)
        R = col_R.get(name, 0.0)
        result[name] = round(R * m ** 2, 1)
    return result


def _cumulative_upstream_dp(
    collector_name: str,
    col_names:      List[str],
    col_dp:         Dict[str, float],
) -> float:
    """
    Sum of pressure drops for all collector legs at or downstream of this
    collector in the daisy-chain topology (matching original logic in
    Collector.calculate_total_pressure_loss).
    """
    try:
        idx = col_names.index(collector_name)
    except ValueError:
        return 0.0
    return sum(col_dp.get(col_names[i], 0.0) for i in range(idx, len(col_names)))