import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path for direct execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.helper import detect_iqr_anomalies, detect_isolation_forest

class GenericAnomalyDetector:
    def __init__(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        self.filepath = filepath
        self.df = pd.read_csv(filepath)
        self.anomaly_columns = []

    def load_data(self, dropna_columns=None, drop_zero_columns=None, engineered_features=None):
        if dropna_columns:
            self.df = self.df.dropna(subset=dropna_columns)
        if drop_zero_columns:
            for col in drop_zero_columns:
                self.df = self.df[self.df[col] != 0]
        if engineered_features:
            for new_col, func in engineered_features.items():
                self.df[new_col] = self.df.apply(func, axis=1)

    def set_anomaly_columns(self, columns):
        self.anomaly_columns = columns

    def detect_iqr_anomalies(self):
        for col in self.anomaly_columns:
            self.df[f"anomaly_{col}"] = detect_iqr_anomalies(self.df[col])

    def detect_isolation_forest(self, output_col="anomaly_isoforest"):
        self.df[output_col] = detect_isolation_forest(
            self.df,
            feature_cols=self.anomaly_columns,
            output_col=output_col
        )[output_col]

    def plot_time_series(self, time_column, target_column):
        if time_column not in self.df.columns:
            print(f"Warning: '{time_column}' not found in DataFrame.")
            return
        self.df[time_column] = pd.to_datetime(self.df[time_column])
        self.df = self.df.sort_values(time_column)
        plt.figure(figsize=(14, 5))
        plt.scatter(
            self.df[time_column], self.df[target_column],
            c=self.df.get("anomaly_isoforest", False).map({True: "red", False: "black"}),
            s=10, alpha=0.6
        )
        plt.title(f"{target_column} Over Time (Anomalies Highlighted)")
        plt.xlabel("Timestamp")
        plt.ylabel(target_column)
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_anomaly_clusters(self, x_column, y_column, category_column=None, size_column=None):
        if "anomaly_isoforest" not in self.df.columns:
            print("Warning: No isolation forest anomalies detected.")
            return
        self.df["anomaly_label"] = self.df["anomaly_isoforest"].map({True: "Anomaly", False: "Normal"})
        sns.scatterplot(
            data=self.df,
            x=x_column,
            y=y_column,
            hue="anomaly_label",
            style=category_column,
            size=size_column,
            sizes=(20, 100),
            alpha=0.7
        )
        plt.title("Anomaly Cluster Visualization")
        plt.tight_layout()
        plt.show()

    def save_anomalies(self, output_file):
        if "anomaly_isoforest" not in self.df.columns:
            print("No anomalies to save.")
            return
        anomalies = self.df[self.df["anomaly_isoforest"] == True]
        anomalies.to_csv(output_file, index=False)
        print(f"Saved {len(anomalies)} anomalies to {output_file}")