import numpy as np


def constant_predictions(train, test):
    y = train["next_day_abs_return"]
    return {
        "Mean_baseline": np.full(len(test), y.mean()),
        "Median_baseline": np.full(len(test), y.median()),
    }
