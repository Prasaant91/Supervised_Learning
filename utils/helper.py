from sklearn.ensemble import IsolationForest

def detect_iqr_anomalies(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return (series < lower) | (series > upper)

def detect_isolation_forest(df, feature_cols, contamination=0.01, random_state=42, output_col="anomaly_isoforest"):
    model = IsolationForest(contamination=contamination, random_state=random_state)
    df[output_col] = model.fit_predict(df[feature_cols]) == -1
    return df
