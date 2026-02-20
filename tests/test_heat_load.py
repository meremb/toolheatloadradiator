"""
tests/domain/test_heat_load.py
Tests for domain/heat_load.py (RoomLoadCalculator)
and domain/radiator.py (Radiator).
"""
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from domain.heat_load import RoomLoadCalculator
from domain.radiator import Radiator, POSSIBLE_DIAMETERS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def living_room():
    return RoomLoadCalculator(
        floor_area=25.0, uw=0.3, u_roof=0.2, u_ground=0.3,
        v_system="C", wall_outside=2, on_ground=True, window=True,
    )


@pytest.fixture
def radiator():
    return Radiator(q_ratio=0.5, delta_t=10.0, space_temperature=20.0, heat_loss=1000.0)


# ---------------------------------------------------------------------------
# RoomLoadCalculator
# ---------------------------------------------------------------------------
class TestRoomLoadCalculator:

    def test_returns_positive_float(self, living_room):
        assert living_room.compute() > 0

    def test_detail_mode_returns_dict(self, living_room):
        living_room.return_detail = True
        result = living_room.compute()
        assert isinstance(result, dict)
        assert {"totalHeatLoss", "transmissionHeatLoss", "ventilationHeatLoss",
                "infiltrationHeatLoss", "neighbourLosses", "atticLosses"} == result.keys()

    def test_total_equals_sum_of_components(self, living_room):
        living_room.return_detail = True
        d = living_room.compute()
        components = (
            d["transmissionHeatLoss"]
            + max(d["ventilationHeatLoss"], d["infiltrationHeatLoss"])
            + d["neighbourLosses"] + d["atticLosses"]
        )
        assert abs(d["totalHeatLoss"] - components) < 0.01

    def test_warmer_outdoor_means_lower_load(self):
        base = dict(floor_area=20, uw=0.3, u_roof=0.2, u_ground=0.3, v_system="C")
        cold = RoomLoadCalculator(tout=-7, **base).compute()
        mild = RoomLoadCalculator(tout=0,  **base).compute()
        assert mild < cold

    def test_better_insulation_means_lower_load(self):
        bad  = RoomLoadCalculator(floor_area=20, uw=1.3, u_roof=1.0, u_ground=1.2, v_system="C").compute()
        good = RoomLoadCalculator(floor_area=20, uw=0.3, u_roof=0.2, u_ground=0.3, v_system="C").compute()
        assert good < bad

    def test_system_d_lower_ventilation_than_c(self):
        base = dict(floor_area=20, uw=0.3, u_roof=0.2, u_ground=0.3, return_detail=True)
        c_vent = RoomLoadCalculator(v_system="C", **base).compute()["ventilationHeatLoss"]
        d_vent = RoomLoadCalculator(v_system="D", **base).compute()["ventilationHeatLoss"]
        assert d_vent < c_vent

    def test_unknown_vent_method_gives_zero(self):
        r = RoomLoadCalculator(floor_area=20, uw=0.3, u_roof=0.2, u_ground=0.3,
                               v_system="C", ventilation_calculation_method="unknown",
                               return_detail=True)
        assert r.compute()["ventilationHeatLoss"] == 0.0

    def test_neighbour_losses_zero_when_disabled(self, living_room):
        living_room.return_detail = True
        assert living_room.compute()["neighbourLosses"] == 0.0

    def test_neighbour_losses_positive_when_enabled(self, living_room):
        living_room.add_neighbour_losses = True
        living_room.return_detail = True
        assert living_room.compute()["neighbourLosses"] > 0.0

    @pytest.mark.parametrize("area", [5.0, 20.0, 60.0, 120.0])
    def test_various_floor_areas_positive(self, area):
        r = RoomLoadCalculator(floor_area=area, uw=0.3, u_roof=0.2, u_ground=0.3, v_system="C")
        assert r.compute() > 0


# ---------------------------------------------------------------------------
# Radiator
# ---------------------------------------------------------------------------
class TestRadiator:

    def test_supply_above_room(self, radiator):
        assert radiator.supply_temperature > radiator.space_temperature

    def test_return_below_supply(self, radiator):
        assert radiator.return_temperature < radiator.supply_temperature

    def test_mass_flow_positive(self, radiator):
        assert radiator.mass_flow_rate > 0

    def test_fixed_supply_respected(self):
        r = Radiator(q_ratio=0.5, delta_t=10, space_temperature=20, heat_loss=1000,
                     supply_temperature=55.0)
        assert r.supply_temperature == 55.0

    def test_energy_balance(self, radiator):
        dT = radiator.supply_temperature - radiator.return_temperature
        q  = radiator.mass_flow_rate / 3600 * 4180 * dT
        assert abs(q - radiator.heat_loss) < 5.0

    def test_diameter_in_standard_list(self, radiator):
        assert radiator.calculate_diameter(POSSIBLE_DIAMETERS) in POSSIBLE_DIAMETERS

    def test_fixed_diameter_respected(self, radiator):
        assert radiator.calculate_diameter(POSSIBLE_DIAMETERS, fixed_diameter=22.0) == 22.0
