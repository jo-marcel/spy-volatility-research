import numpy as np
import pandas as pd
from config import RESULTS


def analyze_regimes(train, test):
    low = train["volatility_20d"].quantile(0.33)
    high = train["volatility_20d"].quantile(0.67)
    table = test.copy()
    table["regime"] = pd.cut(
        table["volatility_20d"],
        bins=[-np.inf, low, high, np.inf],
        labels=["Low", "Medium", "High"]
    )
    summary = table.groupby("regime", observed=True).agg(
        n=("next_day_abs_return", "size"),
        mean_abs_return=("next_day_abs_return", "mean"),
        median_abs_return=("next_day_abs_return", "median")
    ).reset_index()
    summary.to_csv(RESULTS / "volatility_regimes.csv", index=False)
    return summary, low, high
