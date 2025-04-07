import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def detect_iqr_anomalies(self):
    for columns in self.anomaly_columns:
        Q1 = self.df_clean[columns].quantile(0.25)
        Q3 = self.df_clean[columns].quantile(0.25)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        self.df_clean[f"anomaly_iqr_{columns}"] = (self.df_clean[columns] < lower_bound) | (self.df_clean[columns] > upper_bound)


def detect_isolation_forest(self, contamination=0.01):
    X= self.df_clean[self.anomaly_columns]
    model = IsolationForest(contamination=contamination, random_state=42)
    self.df_clean["anomaly_isoforest"] = model.fit_predict(X) == -1



