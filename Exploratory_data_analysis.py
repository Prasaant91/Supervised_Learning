import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

data = pd.read_csv(r"C:\Users\prasa\PycharmProjects\pythonProject\Supervised_Learning\data\Generated_synthetic_manufacturing_data.csv", sep=",")
df = pd.DataFrame(data)

df_clean = df.dropna(subset=["cost_eur"])
df_clean["energy_per_kg"] = df["energy_kWh"] / df["batch_size_kg"]
X = df_clean.drop(columns=["timestamp", "cost_per_kg", "cost_eur"])
Y = df_clean["cost_eur"]

print(df_clean.columns)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(y=df_clean["cost_per_kg"], ax=axs[0])
axs[0].set_title("Cost per kg")

sns.boxplot(y=df_clean["energy_per_kg"], ax=axs[1])
axs[1].set_title("Energy per kg")
plt.tight_layout()
plt.show()

def detect_iqr_anomalies(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return (series < lower) | (series > upper)

# Apply to both metrics
df_clean["anomaly_cost_per_kg"] = detect_iqr_anomalies(df_clean["cost_per_kg"])
df_clean["anomaly_energy_per_kg"] = detect_iqr_anomalies(df_clean["energy_per_kg"])

# Model input: per-kg metrics
X = df_clean[["cost_per_kg", "energy_per_kg"]]

# Isolation forest
iso_model = IsolationForest(contamination=0.01, random_state=42)
df_clean["anomaly_isoforest"] = iso_model.fit_predict(X) == -1  # True for anomalies

# Combine all anomaly flags
anomalies = df_clean[df_clean["anomaly_cost_per_kg"] | df_clean["anomaly_energy_per_kg"] | df_clean["anomaly_isoforest"]]

# Sort by most costly/inefficient
anomalies_sorted = anomalies.sort_values(by=["cost_per_kg", "energy_per_kg"], ascending=False)

# Preview top anomalies
cols = [
    "timestamp", "cost_eur", "energy_kWh", "batch_size_kg",
    "cost_per_kg", "energy_per_kg",
    "anomaly_cost_per_kg", "anomaly_energy_per_kg", "anomaly_isoforest"
]
print(anomalies_sorted[cols].head(20))
anomalies_sorted.to_csv("flagged_anomalous_batches.csv", index=False)

df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"])

# Sort by timestamp
df_clean = df_clean.sort_values("timestamp")

# Plot cost_per_kg anomalies over time
plt.figure(figsize=(14, 5))
plt.scatter(df_clean["timestamp"], df_clean["cost_per_kg"],
            c=df_clean["anomaly_isoforest"].map({True: "red", False: "black"}),
            s=10, alpha=0.6)
plt.title("Cost per kg Over Time (Anomalies Highlighted)")
plt.xlabel("Timestamp")
plt.ylabel("Cost per kg")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()

df_clean.loc[:, "anomaly_label"] = df_clean["anomaly_isoforest"].map({True: "Anomaly", False: "Normal"})
sns.scatterplot(x=df_clean["cost_per_kg"], y=df_clean["energy_per_kg"], hue=df_clean["anomaly_label"], style=df_clean["metal_type"], size=df_clean["material_purity_pct"])
plt.show()