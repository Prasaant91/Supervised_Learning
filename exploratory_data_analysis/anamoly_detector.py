import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

class GenericAnomalyDetector:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.df_clean = None
        self.anomaly_columns = []

    def load_data(self, dropna_columns=None, drop_zero_columns=None, engineered_features=None):
        self.df = pd.read_csv(self.csv_path)
        dropna_columns = dropna_columns or []
        drop_zero_columns = drop_zero_columns or []
        engineered_features = engineered_features or {}

        self.df_clean = self.df.dropna(subset=dropna_columns)
        for columns in drop_zero_columns:
            self.df_clean = self.df_clean[self.df_clean[columns]>0]

        for new_columns, function in engineered_features.items():
            self.df_clean[new_columns] = self.df_clean.apply(function, axis=1)

    def set_anomaly_columns(self, columns):
            self.anomaly_columns = columns

    def detect_iqr_anomalies(self):
        for columns in self.anomaly_columns:
            Q1 = self.df_clean[columns].quantile(0.25)
            Q3 = self.df_clean[columns].quantile(0.25)
            IQR = Q3 -Q1
            lower_bound= Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.df_clean[f"anomaly_iqr_{columns}"] = (self.df_clean[columns]<lower_bound) | (self.df_clean[columns] > upper_bound)

    def detect_isolation_forest(self, contamination=0.01):
        X= self.df_clean[self.anomaly_columns]
        model = IsolationForest(contamination=contamination, random_state=42)
        self.df_clean["anomaly_isoforest"] = model.fit_predict(X) == -1

    def plot_time_series(self, time_column, target_column):
        self.df_clean[time_column] = pd.to_datetime(self.df_clean[time_column])
        self.df_clean = self.df_clean.sort_values(time_column)
        plt.figure(figsize=(14, 5))
        plt.scatter(
            self.df_clean[time_column],
            self.df_clean[target_column],
            c=self.df_clean["anomaly_isoforest"].map({True: "red", False: "black"}),
            s=10, alpha=0.6
        )
        plt.title(f"{target_column} Over Time (Anomalies Highlighted)")
        plt.xlabel("Timestamp")
        plt.ylabel(target_column)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_anomaly_clusters(self, x_column, y_column, category_column=None, size_column=None):
        self.df_clean["anomaly_label"] = self.df_clean["anomaly_isoforest"].map({True: "Anomaly", False: "Normal"})

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=self.df_clean,
            x=x_column,
            y=y_column,
            hue="anomaly_label",
            style=category_column,
            size=size_column,
            alpha=0.7
        )
        plt.title(f"Anomaly scatter plot: {x_column} vs {y_column}")
        plt.tight_layout()
        plt.show()

    def save_anomalies(self, output_path="anomalies_detected.csv"):
        anomaly_flags = [columns for columns in self.df_clean.columns if columns.startswith("anomaly_iqr_")] + ["anomaly_isoforest"]
        anomaly_filter = self.df_clean[anomaly_flags].any(axis=1)
        anomalies = self.df_clean[anomaly_filter]
        anomalies.to_csv(output_path, index=False)