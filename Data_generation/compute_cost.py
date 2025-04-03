import numpy as np
import pandas as pd

def compute_cost(df: pd.DataFrame) -> pd.DataFrame:
    ENERGY_COST = 0.12
    IMPURITY_PENEALTY = 2.0
    COOLING_RATIO = 0.05
    REACTOR_DEPRECIATION = 5.0
    LABOR_COST = 8.0
    REWORK_PENALTY = 20.0
    NOISE_STD = 20

    ADDITIVE_COSTS = {
                        "None" : 0,
                        "Calalyst_A" : 15,
                        "Catalyst_B" : 30
    }
    df["cooling_energy_kWh"] = df["energy_kWh"] * COOLING_RATIO
    df["process_additive_cost"] = df["chemical_additive_type"].map(ADDITIVE_COSTS)
    df["impurity_penalty"] = (
        (1 - df["material_purity_pct"]/100.0) * df["mass_input_kg"] * IMPURITY_PENEALTY
    )
    df["rework_cost"] = (1 - df["quality_check_passed"]) * REWORK_PENALTY
    df["cost_eur"] = (
            df["mass_input_kg"] * df["raw_material_cost_per_kg"] +
            df["impurity_penalty"] +
            df["energy_kWh"] * ENERGY_COST +
            df["cooling_energy_kWh"] * ENERGY_COST +
            df["process_additive_cost"] +
            REACTOR_DEPRECIATION +
            LABOR_COST +
            df["rework_cost"] +
            np.random.normal(0, NOISE_STD, size=len(df))
    )
    df["cost_per_kg"] = df["cost_eur"] / df["batch_size_kg"]
    return df



