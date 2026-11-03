import numpy as np


def calculate_metrics(actual, predicted):
    errors = np.asarray(actual) - np.asarray(predicted)
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    return {"mae": float(mae), "rmse": float(rmse)}


def metrics_table(predictions_frame, period):
    rows = []
    for name in ["OLS", "Quantile_50", "Mean_baseline", "Median_baseline"]:
        values = calculate_metrics(predictions_frame["actual"], predictions_frame[name])
        rows.append({"model": name, "period": str(period),
                     "n": len(predictions_frame), **values})
    return rows
