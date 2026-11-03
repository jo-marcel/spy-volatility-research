import pandas as pd
from config import FIRST_WALK_YEAR, LAST_WALK_YEAR, RESULTS
from split_data import split_for_year
from predict import predict_all
from metrics import metrics_table


def run_walk_forward(data):
    all_predictions = []
    all_scores = []
    for year in range(FIRST_WALK_YEAR, LAST_WALK_YEAR + 1):
        train, test = split_for_year(data, year)
        if len(train) < 250 or test.empty:
            continue
        _, _, predictions = predict_all(train, test)
        frame = pd.DataFrame(predictions, index=test.index)
        frame.insert(0, "actual", test["next_day_abs_return"])
        frame["year"] = year
        all_predictions.append(frame)
        for row in metrics_table(frame, year):
            row["train_n"] = len(train)
            all_scores.append(row)
    predictions = pd.concat(all_predictions)
    scores = pd.DataFrame(all_scores)
    predictions.to_csv(RESULTS / "walk_forward_predictions.csv", index_label="feature_date")
    scores.to_csv(RESULTS / "walk_forward_metrics.csv", index=False)
    return predictions, scores
