import json
from config import RESULTS


def save_summary(df, data, train, test, corr, hac, ols, quantile,
                 scores, walk_scores, low, high):
    overall = scores[scores["period"] == "2023-2025"].set_index("model")
    summary = {
        "price_rows": len(df), "research_rows": len(data),
        "train_rows": len(train), "test_rows": len(test),
        "pearson_r": corr,
        "hac": {
            "slope": float(hac.params["volatility_20d"]),
            "pvalue": float(hac.pvalues["volatility_20d"]),
            "ci95": hac.conf_int().loc["volatility_20d"].tolist(),
            "r_squared": float(hac.rsquared)
        },
        "ols_train": ols.params.to_dict(),
        "quantile_train": quantile.params.to_dict(),
        "holdout_metrics": overall[["mae", "rmse"]].to_dict(orient="index"),
        "walk_forward_years": sorted(walk_scores["period"].unique().tolist()),
        "regime_thresholds_daily": {"low": float(low), "high": float(high)}
    }
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
