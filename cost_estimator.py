import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "data/Generated_synthetic_manufacturing_data.csv"
df = pd.read_csv(file_path)

print("\n Data shape:", df.shape)
print("\n Columns:", df.columns.tolist())
print("\n Data types:\n", df.dtypes)
print("\n Null values:\n", df.isnull().sum())

if "cost_per_kg" in df.columns:
    df = df[df["batch_size_kg"] != 0]
    df["cost_per_kg"] = df["cost_eur"] / df["batch_size_kg"]

plt.figure(figsize=(8, 4))
sns.histplot(df["cost_per_kg"], bins=50, kde=True)
plt.title("Distribution of cost per kg")
plt.xlabel("Cost per kg")
plt.tight_layout()
plt.show()