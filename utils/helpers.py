import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from typing import List, Dict

POSSIBLE_DIAMETERS = [8, 10, 12, 13, 14, 16, 20, 22, 28, 36]
T_FACTOR = 49.83
EXPONENT_RADIATOR = 1.34
PRESSURE_LOSS_BOILER = 350
AVAILABLE_RADIATOR_POWERS = [2000, 2500, 3000, 3500, 4000]


@dataclass
class Radiator:
    q_ratio: float
    delta_t: float
    space_temperature: float
    heat_loss: float
    return_temperature: float = field(init=False)
    supply_temperature: float = field(default=None)
    mass_flow_rate: float = field(init=False)

    def __post_init__(self):
        self.supply_temperature = self.calculate_tsupply()
        self.return_temperature = self.calculate_treturn(self.supply_temperature)
        self.mass_flow_rate = self.calculate_mass_flow_rate()

    def calculate_c(self) -> float:
        """Calculate the constant 'c' based on Q_ratio and delta_T."""
        return math.exp(self.delta_t / T_FACTOR / self.q_ratio ** (1 / EXPONENT_RADIATOR))

    def calculate_tsupply(self) -> float:
        """Calculate the supply temperature based on space temperature, constant_c, and delta_T."""
        constant_c = self.calculate_c()
        t_supply = self.space_temperature + (constant_c / (constant_c - 1)) * self.delta_t
        return round(t_supply, 1)

    def calculate_treturn(self, max_supply_temperature) -> float:
        """Calculate the return temperature."""
        t_return = (((self.q_ratio ** (1 / EXPONENT_RADIATOR) * T_FACTOR) ** 2) /
                (max_supply_temperature - self.space_temperature) + self.space_temperature)
        return round(t_return, 1)

    def calculate_mass_flow_rate(self) -> float:
        """
        Calculate the mass flow rate based on supply and return temperatures and heat loss.
        """
        mass_flow_rate = self.heat_loss / 4180 / (self.supply_temperature - self.return_temperature) * 3600
        return round(mass_flow_rate, 1)

    def calculate_diameter(self, possible_diameters: List[int]) -> float:
        """Calculate the nearest acceptable pipe diameter based on the mass flow rate."""
        if math.isnan(self.mass_flow_rate):
            raise ValueError("The mass flow rate cannot be NaN. Check the configuration of the number of collectors.")
        diameter = 1.4641 * self.mass_flow_rate ** 0.4217
        acceptable_diameters = [d for d in possible_diameters if d >= diameter]

        if not acceptable_diameters:
            raise ValueError(
                f"Calculated diameter exceeds the maximum allowable diameter for mass flow rate: {self.mass_flow_rate}")
        return min(acceptable_diameters, key=lambda x: abs(x - diameter))





@dataclass
class Circuit:
    length_circuit: float
    diameter: float
    mass_flow_rate: float

    def calculate_pressure_loss_piping(self) -> float:
        """Calculate the pressure loss for the piping."""
        kv_piping = 51626 * (self.diameter / 1000) ** 2 - 417.39 * (self.diameter / 1000) + 1.5541
        resistance_meter = 97180 * (self.mass_flow_rate / 1000 / kv_piping) ** 2
        coefficient_local_losses = 1.3
        pressure_loss_pipe = resistance_meter * self.length_circuit * coefficient_local_losses
        return round(pressure_loss_pipe, 1)

    def calculate_pressure_radiator_kv(self) -> float:
        """Calculate the pressure loss for the radiator circuit."""
        pressure_loss_piping = self.calculate_pressure_loss_piping()
        kv_radiator = 2
        pressure_loss_radiator = 97180 * (self.mass_flow_rate / 1000 / kv_radiator) ** 2
        return round((pressure_loss_piping + pressure_loss_radiator), 1)

    def calculate_water_volume(self) -> float:
        """Calculate the water volume for the circuit."""
        water_volume = (np.pi * (self.diameter / 2) ** 2) / 1000000 * self.length_circuit * 1000
        return round(water_volume, 2)

    def calculate_pressure_collector_kv(self) -> float:
        """Using simplified functions for the kv of a component the pressure loss for the head circuit is calculated."""
        pressure_loss_piping = self.calculate_pressure_loss_piping()
        kv_collector = 14.66
        pressure_loss_collector = 97180 * (self.mass_flow_rate / 1000 / kv_collector) ** 2
        return round((pressure_loss_piping + pressure_loss_collector), 1)


@dataclass
class Collector:
    name: str
    pressure_loss: float = 0.0
    mass_flow_rate: float = 0.0

    def update_mass_flow_rate(self, radiator_df: pd.DataFrame) -> None:
        """Update the mass flow rate for the collector based on radiator data."""
        if self.name in radiator_df['Collector'].unique():
            self.mass_flow_rate = radiator_df[radiator_df['Collector'] == self.name]['Mass flow rate'].sum()

    def calculate_diameter(self, possible_diameters: List[int]) -> float:
        """Calculate the nearest acceptable pipe diameter based on the updated mass flow rate."""
        if math.isnan(self.mass_flow_rate):
            raise ValueError("The mass flow rate cannot be NaN. Check the configuration of the number of collectors.")
        diameter = 1.4641 * self.mass_flow_rate ** 0.4217
        acceptable_diameters = [d for d in possible_diameters if d >= diameter]

        if not acceptable_diameters:
            raise ValueError(
                f"Calculated diameter exceeds the maximum allowable diameter for mass flow rate: {self.mass_flow_rate}")
        return min(acceptable_diameters, key=lambda x: abs(x - diameter))

    def calculate_total_pressure_loss(self, radiator_df: pd.DataFrame, collector_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge radiator DataFrame with collector DataFrame on 'Collector' column and calculate total pressure loss.

        For radiators connected to Collector 1, add pressure losses from all subsequent collectors (2, 3, etc.).
        For radiators connected to Collector 2, add pressure losses from subsequent collectors (3, etc.).
        For radiators connected to the last collector, only add the pressure loss of its own collector.
        """
        edited_collector_df = collector_df.sort_values('Collector')
        merged_df = pd.merge(radiator_df, edited_collector_df[['Collector', 'Collector pressure loss']],
                             on='Collector', how='left')
        collector_pressure_loss_map = edited_collector_df.set_index('Collector')['Collector pressure loss'].to_dict()
        collectors = list(collector_pressure_loss_map.keys())
        total_pressure_losses = []
        for idx, row in merged_df.iterrows():
            current_collector_index = collectors.index(row['Collector'])
            additional_pressure_loss = sum(
                collector_pressure_loss_map[collectors[i]] for i in range(current_collector_index, len(collectors)))
            total_pressure_loss = row['Pressure loss'] + additional_pressure_loss + PRESSURE_LOSS_BOILER
            total_pressure_losses.append(total_pressure_loss)

        merged_df['Total Pressure Loss'] = total_pressure_losses
        return merged_df


@dataclass
class Valve:
    kv_max: float = 0.7  # Default kv_max value
    n: int = 100  # Default number of valve positions

    def calculate_pressure_valve_kv(self, mass_flow_rate: float) -> float:
        """
        Calculate pressure loss for thermostatic valve at position N.
        """
        pressure_loss_valve = 97180 * (mass_flow_rate / 1000 / self.kv_max) ** 2
        return round(pressure_loss_valve, 1)

    def calculate_kv_needed(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the kv needed for each valve position based on pressure loss and mass flow rate.
        """
        merged_df = merged_df.copy()
        merged_df['Total pressure valve circuit'] = merged_df['Total Pressure Loss'] + merged_df['Valve pressure loss N']
        maximum_pressure = max(merged_df['Total pressure valve circuit'])
        merged_df['Pressure difference valve'] = maximum_pressure - merged_df['Total Pressure Loss']
        merged_df['kv_needed'] = (merged_df['Mass flow rate'] / 1000) / (merged_df['Pressure difference valve'] / 100000) ** 0.5
        return merged_df

    def calculate_valve_position(self, a: float, b: float, c: float, kv_needed: np.ndarray) -> np.ndarray:
        """
        Calculate the valve position based on kv needed and polynomial coefficients.
        """
        discriminant = b ** 2 - 4 * a * (c - kv_needed)
        discriminant = np.where(discriminant < 0, 0, discriminant)
        root = -b + np.sqrt(discriminant) / (2 * a)
        root = np.where(discriminant <= 0, 0.1, root)
        return root

    def adjust_position_with_custom_values(self, kv_needed: np.ndarray) -> np.ndarray:
        """
        Adjust the valve position using custom kv_max and n values.
        """
        ratio_kv = kv_needed / 0.7054
        adjusted_ratio_kv = (ratio_kv * self.kv_max) / self.kv_max
        ratio_position = np.clip(np.sqrt(adjusted_ratio_kv), 0, 1)
        adjusted_position = np.ceil(ratio_position * self.n)
        return adjusted_position

    def calculate_kv_position_valve(self, merged_df: pd.DataFrame, custom_kv_max: float = None, n: int = None) -> pd.DataFrame:
        """
        Calculate the valve positions and update the DataFrame with the results.
        """
        merged_df = self.calculate_kv_needed(merged_df)
        a, b, c = 0.0114, -0.0086, 0.0446
        initial_positions = self.calculate_valve_position(a, b, c, merged_df['kv_needed'].to_numpy())

        if custom_kv_max is not None and n is not None:
            self.kv_max = custom_kv_max
            self.n = n
            adjusted_positions = self.adjust_position_with_custom_values(merged_df['kv_needed'].to_numpy())
            merged_df['Valve position'] = adjusted_positions.flatten()
        else:
            initial_positions = np.ceil(initial_positions)
            merged_df['Valve position'] = initial_positions.flatten()

        return merged_df

    def calculate_position_valve_with_ratio(self, kv_max: float, n: int, kv_needed: np.ndarray) -> np.ndarray:
        """
        Calculate the valve position based on a custom kv_max value and number of positions.
        """
        ratio_kv = kv_needed / kv_max
        a, b, c = 0.8053, 0.1269, 0.0468
        ratio_position = self.calculate_valve_position(a, b, c, ratio_kv)
        final_position = np.ceil(ratio_position * n)
        return final_position


def validate_data(df: pd.DataFrame) -> bool:
    """Validate the input data to ensure all required fields are correctly filled."""
    required_columns = ['Radiator power 75/65/20', 'Calculated heat loss', 'Length circuit', 'Space Temperature']
    for col in required_columns:
        if df[col].isnull().any() or (df[col] <= 0).any():
            return False
    for index, row in df.iterrows():
        radiator_power = row['Radiator power 75/65/20']
        heat_loss = row['Calculated heat loss']
        space_temperature = row['Space Temperature']
        if radiator_power < heat_loss:
            radiator = warn_radiator_power(radiator_power, heat_loss, space_temperature)  # Trigger warning
            df.at[index, 'Radiator power 75/65/20'] = radiator  # Overwrite with best power
    return True


def warn_radiator_power(radiator_power: float, heat_loss: float, space_temperature: float) -> float:
    """Issue a warning if radiator power is lower than heat loss and suggest the best possible radiator power."""
    best_radiator_power = suggest_best_radiator_power(heat_loss, space_temperature=space_temperature)
    print(f"Warning: Radiator power ({radiator_power}) is lower than heat loss ({heat_loss}).")
    print(f"Consider using a radiator with at least {best_radiator_power} power for optimal performance.")
    return best_radiator_power


#todo add the supply temeprature as input and calculate than need power and add that to the column add power
def suggest_best_radiator_power(heat_loss: float, space_temperature: float) -> float:
    """Suggest the best possible radiator power based on the supply temperature."""
    radiator_needed = None

    for value in sorted(AVAILABLE_RADIATOR_POWERS):
        if value > heat_loss + 100:
            radiator_needed = value
            break
    return radiator_needed


def calculate_radiator_data(edited_radiator_df: pd.DataFrame, delta_T: float,
                            supply_temp_input: float = None) -> pd.DataFrame:
    """Method to perform the calculations and return the updated radiator DataFrame."""

    # Validate data
    numeric_columns = [
        'Radiator power 75/65/20', 'Calculated heat loss', 'Length circuit', 'Space Temperature'
    ]
    edited_radiator_df[numeric_columns] = edited_radiator_df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    if not validate_data(edited_radiator_df):
        raise ValueError("Invalid input data. Please check your inputs.")

    # Initialize radiators for calculations
    radiators = []
    for _, row in edited_radiator_df.iterrows():
        radiator = Radiator(
            q_ratio=(row['Calculated heat loss'] - row['Extra power']) / row['Radiator power 75/65/20'],  # moeten het mogelijk maken om als er een toevoertempeartuur gevraagd word hiermee aan te duiden hoeveel extra genormaliseerd vermogen nodig is extra als de huidige radiator niet voldoet
            delta_t=delta_T,
            space_temperature=row['Space Temperature'],
            heat_loss=row['Calculated heat loss']
        )
        radiators.append(radiator)

    # Calculate supply temperature if not manually set
    if supply_temp_input is not None:
        max_supply_temperature = supply_temp_input
        if max_supply_temperature < max(r.supply_temperature for r in radiators):
            raise ValueError(
                "Error: The maximum supply temperature must be greater than the maximum radiator supply temperature.")
    else:
        max_supply_temperature = max(r.supply_temperature for r in radiators)

    # Perform calculations for supply temperature, return temperature, mass flow rate, etc.
    for r in radiators:
        r.supply_temperature = max_supply_temperature
        r.return_temperature = r.calculate_treturn(max_supply_temperature)
        r.mass_flow_rate = r.calculate_mass_flow_rate()

    # Add calculated values to the DataFrame
    edited_radiator_df['Supply Temperature'] = max_supply_temperature
    edited_radiator_df['Return Temperature'] = [r.return_temperature for r in radiators]
    edited_radiator_df['Mass flow rate'] = [r.mass_flow_rate for r in radiators]
    edited_radiator_df['Diameter'] = [r.calculate_diameter(POSSIBLE_DIAMETERS) for r in radiators]
    edited_radiator_df['Diameter'] = edited_radiator_df['Diameter'].max()
    edited_radiator_df['Pressure loss'] = [
        Circuit(length_circuit=row['Length circuit'], diameter=row['Diameter'], mass_flow_rate=row['Mass flow rate'])
        .calculate_pressure_radiator_kv() for _, row in edited_radiator_df.iterrows()
    ]

    return edited_radiator_df


def calculate_weighted_delta_t(radiators, radiator_df):
    """
    Bereken de gewogen delta T voor het systeem.

    Args:
        radiators (List[Radiator]): Lijst van radiatorobjecten.
        radiator_df (pd.DataFrame): DataFrame met radiatorgegevens.

    Returns:
        float: Gewogen delta T.
    """
    total_mass_flow_rate = sum(radiator.mass_flow_rate for radiator in radiators)
    weighted_delta_t = sum(
        row['Mass flow rate'] * (row['Supply Temperature'] - row['Return Temperature'])
        for (_, row), radiator in zip(radiator_df.iterrows(), radiators)
    ) / total_mass_flow_rate

    return weighted_delta_t

def load_radiator_data(num_radiators: int, collector_options: List[str]) -> pd.DataFrame:
    radiator_columns: List[str] = [
        'Radiator nr', 'Collector', 'Radiator power 75/65/20', 'Calculated heat loss',
        'Length circuit', 'Space Temperature', 'Extra power'
    ]
    radiator_initial_data: Dict[str, List] = {
        'Radiator nr': list(range(1, num_radiators + 1)),
        'Collector': [collector_options[0]] * num_radiators,
        'Radiator power 75/65/20': [0.0] * num_radiators,
        'Calculated heat loss': [0.0] * num_radiators,
        'Length circuit': [0.0] * num_radiators,
        'Space Temperature': [20.0] * num_radiators,  # Default space temperature
        'Extra power': [0.0] * num_radiators,
    }
    return pd.DataFrame(radiator_initial_data, columns=radiator_columns)


def load_collector_data(num_collectors: int) -> pd.DataFrame:
    """
    Laad de collector data.
    """
    collector_columns: List[str] = ['Collector', 'Collector circuit length']
    collector_initial_data: Dict[str, List] = {
        'Collector': [f'Collector {i + 1}' for i in range(num_collectors)],
        'Collector circuit length': [0.0] * num_collectors,
    }

    return pd.DataFrame(collector_initial_data, columns=collector_columns)



