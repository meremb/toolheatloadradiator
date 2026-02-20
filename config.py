"""
config.py
=========
Application-wide constants: insulation presets, design mode names,
pump library, and UI helpers. No business logic lives here.
"""
from __future__ import annotations
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Design modes
# ---------------------------------------------------------------------------
MODE_EXISTING = "existing"   # Existing radiators → calculate required supply T
MODE_FIXED    = "fixed"      # LT dimensioning → fixed supply T → extra power
MODE_PUMP     = "pump"       # Pump-driven → pump curve determines flow
MODE_BAL      = "balancing"  # Hydraulic balancing / TRV positioning

# ---------------------------------------------------------------------------
# Building envelope presets
# ---------------------------------------------------------------------------
INSULATION_U_VALUES: Dict[str, Dict[str, float]] = {
    "not insulated": {"wall": 1.3, "roof": 1.0, "ground": 1.2},
    "bit insulated":  {"wall": 0.6, "roof": 0.4, "ground": 0.5},
    "insulated well": {"wall": 0.3, "roof": 0.2, "ground": 0.3},
}

GLAZING_U_VALUES: Dict[str, float] = {
    "single": 5.0,
    "double": 2.8,
    "triple": 0.8,
}

# ---------------------------------------------------------------------------
# UI display
# ---------------------------------------------------------------------------
CHART_HEIGHT_PX = 460
ROOM_TYPE_OPTIONS = ["Living", "Kitchen", "Bedroom", "Laundry", "Bathroom", "Toilet"]
ROOM_TYPE_HELP_MD = (
    "**Room Type** influences heat loss / ventilation defaults.\n\n"
    "- **Living**: 20–21 °C\n"
    "- **Kitchen**: 18–20 °C\n"
    "- **Bedroom**: 16–18 °C\n"
    "- **Laundry**: humid space, more ventilation\n"
    "- **Bathroom**: 22 °C design, short peaks\n"
    "- **Toilet**: ~18 °C"
)

# ---------------------------------------------------------------------------
# Pump library  (model → speed → [(flow kg/h, head kPa), ...])
# ---------------------------------------------------------------------------
PumpCurve = List[Tuple[float, float]]
PUMP_LIBRARY: Dict[str, Dict[str, PumpCurve]] = {
    "Grundfos UPM3 15-70": {
        "speed_1": [(0, 55), (200, 50), (400, 42), (600, 30), (800, 18), (1000, 6),  (1100, 2)],
        "speed_2": [(0, 65), (250, 60), (500, 51), (750, 38), (1000, 24), (1150, 12), (1250, 5)],
        "speed_3": [(0, 75), (300, 70), (600, 60), (900, 44), (1200, 28), (1400, 16), (1500, 8)],
    },
    "Wilo Yonos PICO 25-1/6": {
        "speed_1": [(0, 50), (250, 44), (500, 36), (750, 26), (1000, 15), (1200, 7)],
        "speed_2": [(0, 60), (300, 54), (600, 45), (900, 33), (1200, 20), (1400, 12)],
        "speed_3": [(0, 70), (350, 64), (700, 54), (1050, 40), (1400, 26), (1600, 15)],
    },
    "Generic 25-60": {
        "speed_1": [(0, 48), (250, 42), (500, 34), (750, 24), (1000, 13), (1200, 6)],
        "speed_2": [(0, 58), (300, 52), (600, 44), (900, 32), (1200, 19), (1400, 11)],
        "speed_3": [(0, 68), (350, 62), (700, 52), (1050, 38), (1400, 24), (1600, 14)],
    },
}
