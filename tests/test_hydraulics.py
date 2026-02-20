"""
tests/domain/test_hydraulics.py
Tests for domain/hydraulics.py and domain/valve.py.
"""
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import pandas as pd
from domain.hydraulics import Circuit, Collector, calc_velocity, check_pipe_velocities
from domain.radiator import POSSIBLE_DIAMETERS
from domain.valve import Valve


class TestCircuit:

    @pytest.fixture
    def circuit(self):
        return Circuit(length_circuit=15.0, diameter=16.0, mass_flow_rate=120.0)

    def test_piping_loss_positive(self, circuit):
        assert circuit.calculate_pressure_loss_piping() > 0

    def test_radiator_kv_exceeds_piping(self, circuit):
        assert circuit.calculate_pressure_radiator_kv() > circuit.calculate_pressure_loss_piping()

    def test_collector_kv_exceeds_piping(self, circuit):
        assert circuit.calculate_pressure_collector_kv() > circuit.calculate_pressure_loss_piping()

    def test_water_volume_positive(self, circuit):
        assert circuit.calculate_water_volume() > 0

    def test_longer_pipe_higher_loss(self):
        short = Circuit(5.0,  16.0, 100.0).calculate_pressure_loss_piping()
        long_ = Circuit(20.0, 16.0, 100.0).calculate_pressure_loss_piping()
        assert long_ > short

    def test_larger_diameter_lower_loss(self):
        small = Circuit(10.0, 10.0, 100.0).calculate_pressure_loss_piping()
        large = Circuit(10.0, 22.0, 100.0).calculate_pressure_loss_piping()
        assert large < small


class TestCollector:

    @pytest.fixture
    def rad_df(self):
        return pd.DataFrame({
            "Radiator nr": [1, 2],
            "Collector": ["Collector 1", "Collector 1"],
            "Mass flow rate": [50.0, 80.0],
        })

    def test_update_mass_flow_rate(self, rad_df):
        col = Collector(name="Collector 1")
        col.update_mass_flow_rate(rad_df)
        assert col.mass_flow_rate == pytest.approx(130.0)

    def test_unknown_collector_zero_flow(self, rad_df):
        col = Collector(name="Collector 99")
        col.update_mass_flow_rate(rad_df)
        assert col.mass_flow_rate == 0.0

    def test_diameter_in_standard_list(self, rad_df):
        col = Collector(name="Collector 1")
        col.update_mass_flow_rate(rad_df)
        assert col.calculate_diameter(POSSIBLE_DIAMETERS) in POSSIBLE_DIAMETERS


class TestVelocity:

    def test_positive_for_valid_inputs(self):
        assert calc_velocity(120.0, 16.0) > 0

    def test_zero_diameter(self):
        assert calc_velocity(120.0, 0.0) == 0.0

    def test_zero_flow(self):
        assert calc_velocity(0.0, 16.0) == 0.0

    def test_larger_diameter_lower_velocity(self):
        assert calc_velocity(100.0, 22.0) < calc_velocity(100.0, 10.0)

    def test_warning_on_high_velocity(self):
        rad = pd.DataFrame({"Radiator nr": [1], "Mass flow rate": [5000.0], "Diameter": [8.0]})
        col = pd.DataFrame({"Collector": ["C1"],  "Mass flow rate": [1.0],    "Diameter": [22.0]})
        _, _, warnings = check_pipe_velocities(rad, col)
        assert any("High velocity" in w for w in warnings)

    def test_velocity_column_added(self):
        rad = pd.DataFrame({"Radiator nr": [1], "Mass flow rate": [50.0], "Diameter": [16.0]})
        col = pd.DataFrame({"Collector": ["C1"],  "Mass flow rate": [50.0], "Diameter": [16.0]})
        r, c, _ = check_pipe_velocities(rad, col)
        assert "Velocity (m/s)" in r.columns
        assert "Velocity (m/s)" in c.columns


class TestValve:

    def test_get_valve_names_includes_custom(self):
        assert "Custom" in Valve.get_valve_names()
        assert len(Valve.get_valve_names()) > 1

    def test_known_valve_config(self):
        v = Valve(valve_name="Danfoss RA-N 15 (1/2)")
        cfg = v.get_config()
        assert cfg is not None and "kv_values" in cfg

    def test_custom_returns_none_config(self):
        assert Valve(valve_name="Custom").get_config() is None

    def test_pressure_loss_positive(self):
        assert Valve(valve_name="Danfoss RA-N 15 (1/2)").calculate_pressure_valve_kv(100.0) > 0

    def test_zero_kv_returns_inf(self):
        assert Valve(kv_max=0.0).calculate_pressure_valve_kv(100.0) == float("inf")
