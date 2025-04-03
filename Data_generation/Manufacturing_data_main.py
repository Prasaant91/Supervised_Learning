import pandas as pd
from synthetic_data_generation_manufacturing import schema, generate_dataset, generate_timestamps
from compute_cost import compute_cost

n_samples = 100000
start_time = "2022-01-01 00:00:00"
frequency = "10T"

df = generate_dataset(schema, n_samples)
df["timestamp"] = generate_timestamps(start_time, n_samples, frequency)

df = compute_cost(df)
df.to_csv("data/Generated_synthetic_manufacturing_data.csv", index=False)


