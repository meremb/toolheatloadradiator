"""
tests/services/test_radiator_service.py
Tests for services/radiator_service.py and services/pump_service.py.
"""
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from domain.radiator import Radiator
from services.radiator_service import (
    calculate_weighted_delta_t,
    calculate_extra_power_needed,
    calculate_radiator_data_with_extra_power,
    validate_data,
    load_radiator_data,
    load_collector_data,
    init_radiator_rows,
    resize_radiator_rows,
    init_collector_rows,
    resize_collector_rows,
)
from services.pump_service import interpolate_curve, derive_system_curve_k, find_operating_point


class TestWeightedDeltaT:

    def test_single_radiator(self):
        r = Radiator(q_ratio=0.5, delta_t=10.0, space_temperature=20.0, heat_loss=1000.0)
        df = pd.DataFrame({
            "Mass flow rate":       [r.mass_flow_rate],
            "Supply Temperature":   [r.supply_temperature],
            "Return Temperature":   [r.return_temperature],
        })
        result = calculate_weighted_delta_t([r], df)
        assert abs(result - (r.supply_temperature - r.return_temperature)) < 0.01

    def test_zero_flow_returns_zero(self):
        r = Radiator(q_ratio=0.5, delta_t=10.0, space_temperature=20.0, heat_loss=1000.0)
        r.mass_flow_rate = 0.0
        df = pd.DataFrame({"Mass flow rate": [0.0], "Supply Temperature": [55.0], "Return Temperature": [45.0]})
        assert calculate_weighted_delta_t([r], df) == 0.0


class TestExtraPowerNeeded:

    def test_sufficient_power_zero_extra(self):
        assert calculate_extra_power_needed(3000, 500, 75, 10, 20) == 0.0

    def test_invalid_inputs_zero(self):
        assert calculate_extra_power_needed(None, None, 55, 10, 20) == 0.0

    def test_negative_delta_t_zero(self):
        assert calculate_extra_power_needed(2000, 800, 55, -5, 20) == 0.0

    def test_non_negative_result(self):
        result = calculate_extra_power_needed(1000, 900, 35, 5, 20)
        assert result >= 0.0


class TestBatchRadiatorData:

    @pytest.fixture
    def rad_df(self):
        return pd.DataFrame({
            "Radiator nr":              [1, 2],
            "Collector":                ["Collector 1", "Collector 1"],
            "Radiator power 75/65/20":  [2000.0, 2500.0],
            "Calculated heat loss":     [800.0,  1200.0],
            "Length circuit":           [10.0,   15.0],
            "Space Temperature":        [20.0,   22.0],
            "Mass flow rate":           [0.0, 0.0],
            "Supply Temperature":       [0.0, 0.0],
            "Return Temperature":       [0.0, 0.0],
            "Diameter":                 [0.0, 0.0],
        })

    def test_expected_columns_present(self, rad_df):
        result = calculate_radiator_data_with_extra_power(rad_df, delta_T=10.0, supply_temp_input=55.0)
        for col in ["Supply Temperature", "Return Temperature", "Mass flow rate", "Diameter", "Pressure loss"]:
            assert col in result.columns

    def test_fixed_supply_applied(self, rad_df):
        result = calculate_radiator_data_with_extra_power(rad_df, delta_T=10.0, supply_temp_input=55.0)
        assert (result["Supply Temperature"] == 55.0).all()

    def test_uniform_diameter(self, rad_df):
        result = calculate_radiator_data_with_extra_power(rad_df, delta_T=10.0, supply_temp_input=55.0)
        assert result["Diameter"].nunique() == 1


class TestValidateData:

    def test_accepts_valid(self):
        df = pd.DataFrame({
            "Radiator power 75/65/20": [2000.0, 2500.0],
            "Calculated heat loss":    [800.0,  1000.0],
            "Length circuit":          [10.0,   12.0],
            "Space Temperature":       [20.0,   22.0],
        })
        assert validate_data(df) is True

    def test_rejects_zero(self):
        df = pd.DataFrame({
            "Radiator power 75/65/20": [2000.0, 0.0],
            "Calculated heat loss":    [800.0,  600.0],
            "Length circuit":          [10.0,   10.0],
            "Space Temperature":       [20.0,   20.0],
        })
        assert validate_data(df) is False


class TestTableHelpers:

    def test_load_radiator_data(self):
        df = load_radiator_data(4, ["Collector 1"])
        assert len(df) == 4 and "Radiator nr" in df.columns

    def test_load_collector_data(self):
        df = load_collector_data(3)
        assert len(df) == 3 and "Collector" in df.columns

    def test_resize_radiator_rows_grows(self):
        rows = init_radiator_rows(2, ["C1"], [1, 2])
        grown = resize_radiator_rows(rows, 5, ["C1"], [1, 2])
        assert len(grown) == 5 and grown[0]["Radiator nr"] == 1

    def test_resize_radiator_rows_shrinks(self):
        rows = init_radiator_rows(5, ["C1"], [1, 2])
        shrunk = resize_radiator_rows(rows, 2, ["C1"], [1, 2])
        assert len(shrunk) == 2

    def test_resize_collector_rows_grows(self):
        rows  = init_collector_rows(2)
        grown = resize_collector_rows(rows, 4)
        assert len(grown) == 4

    def test_resize_collector_rows_shrinks(self):
        rows    = init_collector_rows(4)
        shrunk  = resize_collector_rows(rows, 2)
        assert len(shrunk) == 2


class TestPumpService:

    def test_interpolate_empty_returns_nan(self):
        result = interpolate_curve([], np.array([100.0, 200.0]))
        assert all(np.isnan(result))

    def test_interpolate_valid(self):
        points = [(0, 60), (500, 40), (1000, 10)]
        y = interpolate_curve(points, np.array([250.0, 750.0]))
        assert not any(np.isnan(y))
        assert y[0] > y[1]  # head decreases with flow

    def test_derive_system_curve_k_zero_flow(self):
        assert derive_system_curve_k(0.0, 50.0) == 0.0

    def test_find_operating_point_found(self):
        points  = [(0, 60), (500, 40), (1000, 10)]
        Q_grid  = np.linspace(0, 1000, 200)
        pump_kpa = interpolate_curve(points, Q_grid)
        K_sys    = derive_system_curve_k(400, 20)
        q, dp, w = find_operating_point(Q_grid, pump_kpa, K_sys)
        assert q is not None and dp is not None
        assert len(w) == 0
