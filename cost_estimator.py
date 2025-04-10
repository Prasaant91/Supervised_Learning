import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import joblib

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

