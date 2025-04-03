from anamoly_detector import GenericAnomalyDetector

def main():
    detector = GenericAnomalyDetector("data/Generated_synthetic_manufacturing_data.csv")

    # Step 2: Load and clean data
    detector.load_data(
        dropna_columns=["cost_eur", "batch_size_kg", "energy_kWh"],
        drop_zero_columns=["batch_size_kg"],
        engineered_features={
            "cost_per_kg": lambda row: row["cost_eur"] / row["batch_size_kg"],
            "energy_per_kg": lambda row: row["energy_kWh"] / row["batch_size_kg"]
        }
    )

    detector.set_anomaly_columns(["cost_per_kg", "energy_per_kg"])

    detector.detect_iqr_anomalies()
    detector.detect_isolation_forest()

    detector.plot_time_series(time_column="timestamp", target_column="cost_per_kg")
    detector.plot_anomaly_clusters(x_column="cost_per_kg", y_column="energy_per_kg",
                                   category_column="metal_type", size_column="material_purity_pct")

    detector.save_anomalies("flagged_anomalies.csv")

if __name__ == "__main__":
    main()