import pandas as pd
from split_data import split_train_test
from predict import predict_all
from metrics import metrics_table
from config import RESULTS


def run_holdout(data):
    train, test = split_train_test(data)
    ols, quantile, predictions = predict_all(train, test)
    frame = pd.DataFrame(predictions, index=test.index)
    frame.insert(0, "actual", test["next_day_abs_return"])
    rows = metrics_table(frame, "2023-2025")
    for year, group in frame.groupby(frame.index.year):
        rows.extend(metrics_table(group, year))
    scores = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "holdout_predictions.csv", index_label="feature_date")
    scores.to_csv(RESULTS / "holdout_metrics.csv", index=False)
    return train, test, ols, quantile, frame, scores
