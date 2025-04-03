import pandas as pd
import numpy as np

np.random.seed(42)
class Feature:
    def __init__(self, name, generator, **kwargs):
        self.name = name
        self.generator = generator
        self.params = kwargs
    def generate(self, n):
        return self.generator(n, **self.params)

def uniform(n, low, high): return np.random.uniform(low, high, size=n)
def randint(n, low, high): return np.random.randint(low, high, size=n)
def choice(n, choices): return np.random.choice(choices, size=n)
def poisson(n, lam): return np.random.poisson(lam, size=n)

schema = [
            Feature("metal_type", choice, choices=["Aluminum", "Zinc", "Magnesium"]),
            Feature("chemical_additive_type", choice, choices=["None", "Catalyst_A", "Catalyst_B"]),
            Feature("operator_shift", choice, choices=["Day", "Night"]),
            Feature("input_material_source", choice, choices=["Local", "Imported"]),
            Feature("equipment_type", choice, choices=["Type_A", "Type_B", "Type_C"]),

            Feature("mass_input_kg", uniform, low=50, high=500),
            Feature("reaction_temp_c", uniform, low=50, high=150),
            Feature("reaction_time_min", uniform, low=30, high=180),
            Feature("energy_kWh", uniform, low=100, high=2000),
            Feature("batch_size_kg", uniform, low=40, high=480),
            Feature("raw_material_cost_per_kg", uniform, low=1, high=10),
            Feature("material_purity_pct", uniform, low=80, high=100),
            Feature("moisture_content_pct", uniform, low=0, high=10),
            Feature("ambient_temp_c", uniform, low=10, high=40),
            Feature("reaction_pressure_bar", uniform, low=1, high=10),
            Feature("process_yield_pct", uniform, low=70, high=100),
            Feature("cooling_time_min", uniform, low=10, high=60),
            Feature("maintenance_hours_last_month", uniform, low=0, high=40),
            Feature("downtime_last_month_hr", uniform, low=0, high=20),
            Feature("regulatory_compliance_level", uniform, low=0.5, high=1.0),

            Feature("equipment_age_years", randint, low=1, high=20),
            Feature("operator_experience_years", randint, low=1, high=30),
            Feature("num_operators", randint, low=1, high=6),
            Feature("shift_duration_hr", choice, choices=[8, 12]),
            Feature("week_of_year", randint, low=1, high=53),
            Feature("quality_check_passed", choice, choices=[0, 1]),
            Feature("num_defects", poisson, lam=2),
        ]

def generate_dataset(schema, n_samples):
    data = {feature.name: feature.generate(n_samples) for feature in schema}
    return pd.DataFrame(data)

def generate_timestamps(start, n, freq="10T"):
    return pd.date_range(start=pd.to_datetime(start), periods=n, freq=freq)
