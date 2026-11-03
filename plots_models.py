import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plots_history import save

MODELS = ["OLS", "Quantile_50", "Mean_baseline", "Median_baseline"]


def plot_models(holdout, scores, walk_scores, regimes):
    recent = holdout.tail(120)
    plt.figure(figsize=(12, 5))
    for column in ["actual", "OLS", "Quantile_50"]:
        plt.plot(recent.index, recent[column] * 100, label=column)
    plt.title("Last 120 holdout dates: next-session return magnitude")
    plt.xlabel("Feature date t")
    plt.ylabel("Next-session absolute return (%)")
    plt.legend()
    save("05_predictions.png")

    overall = scores[scores["period"] == "2023-2025"].set_index("model").loc[MODELS]
    x = np.arange(4)
    plt.figure(figsize=(10, 5))
    plt.bar(x - 0.18, overall["mae"] * 100, 0.36, label="MAE")
    plt.bar(x + 0.18, overall["rmse"] * 100, 0.36, label="RMSE")
    plt.xticks(x, ["OLS", "Quantile", "Mean", "Median"])
    plt.ylabel("Error (percentage points)")
    plt.title("Fixed holdout 2023-2025")
    plt.legend()
    save("06_model_comparison.png")

    yearly = scores[scores["period"] != "2023-2025"]
    yearly = yearly.pivot(index="period", columns="model", values="mae")
    (yearly[MODELS] * 100).plot(kind="bar", figsize=(11, 5), rot=0)
    plt.title("Holdout MAE by year")
    plt.ylabel("MAE (percentage points)")
    save("07_yearly_performance.png")

    walk = walk_scores.pivot(index="period", columns="model", values="mae")
    (walk[MODELS] * 100).plot(marker="o", figsize=(11, 5))
    plt.title("Annual expanding-window refits (retrospective)")
    plt.xlabel("Year")
    plt.ylabel("MAE (percentage points)")
    save("08_walk_forward_mae.png")

    plt.figure(figsize=(8, 5))
    plt.bar(regimes["regime"].astype(str), regimes["mean_abs_return"] * 100)
    plt.title("Holdout next-day magnitude by volatility regime")
    plt.xlabel("Regime (train thresholds)")
    plt.ylabel("Mean next-day absolute return (%)")
    save("09_volatility_regimes.png")
