# Volatility Regimes and Next-Day Market Risk

**A modular, beginner-readable quantitative research project on SPY (2015–2025).**

## Research question

Does trailing 20-session realized volatility help predict the **magnitude** of the next SPY daily log return? This is **not** a directional or trading strategy.

## Reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
python -m unittest discover -s tests -v
```

For a simple, printed, step-by-step run:

```bash
python run_steps.py
```

## How the files connect

```text
main.py
 ├── data_loader.py       -> reads SPY_2015_2025.csv
 ├── features.py          -> computes returns, rolling volatility, target
 ├── inference.py         -> full-sample Pearson and HAC OLS association
 ├── holdout.py
 │    ├── split_data.py   -> chronological train/test
 │    ├── predict.py
 │    │    ├── models.py   -> OLS and median quantile regression
 │    │    └── baselines.py -> training mean and median
 │    └── metrics.py      -> MAE and RMSE
 ├── walk_forward.py      -> yearly expanding-window refits
 ├── regimes.py           -> low/medium/high volatility on train cutoffs
 ├── plots_history.py    -> figures 01–04
 ├── plots_models.py     -> figures 05–09
 └── export_summary.py   -> results/summary.json

config.py                -> shared paths and settings
run_steps.py             -> educational walkthrough
analysis.py              -> compatibility entrypoint
```

Each file has a short docstring and straightforward functions. You can read them in the order above.

## Key results

2,766 price rows; 2,745 complete observations; Pearson r = 0.4572; HAC OLS beta = 0.6207 (95% CI approximately 0.415–0.826), full-sample R² = 0.209. Full-sample association is **not** out-of-sample performance.

Fixed chronological test: train 2015–2022 (1,993 rows, final boundary feature removed); test 2023–2025 (751 rows).

| Model | Test MAE | Test RMSE |
|---|---:|---:|
| OLS | 0.004638 | **0.006953** |
| Quantile q=0.5 | 0.004422 | 0.007036 |
| Train mean baseline | 0.004895 | 0.007109 |
| Train median baseline | **0.004406** | 0.007269 |

All errors are in fractional log-return units. Multiply by 100 for percentage points. OLS beats the training mean in RMSE by 2.20%, but not the training median in MAE. Quantile q=0.5 nearly matches, but does not beat, the median baseline in MAE.

## Outputs

- `figures/01_spy_price.png` to `09_volatility_regimes.png` (nine charts)
- `results/holdout_predictions.csv`, `holdout_metrics.csv`
- `results/walk_forward_predictions.csv`, `walk_forward_metrics.csv`
- `results/volatility_regimes.csv`, `summary.json`

## Methodology and caveats

1. `log_return[t] = log(Close[t] / Close[t-1])`.
2. `volatility_20d[t]` uses 20 returns up to and including date `t`.
3. Target `next_day_abs_return[t] = abs(log_return[t+1])`.
4. Split is chronological. We remove the final training feature date so its target is not in the test period.
5. All model coefficients, mean/median baselines and regime thresholds use only the relevant training period.
6. Walk-forward refits yearly on all previous years, removing each year's boundary feature.
7. The fixed test has been examined while developing the study; later model choices (such as quantile regression) are exploratory on that period, not an untouched confirmatory test.
8. HAC errors correct the uncertainty estimate under serial dependence assumptions; they do not make forecasts better.
9. The study does not estimate trading profitability or causal effects.

## Teaching materials

- `Guide_FR.md`: **start here**, each module explained with its inputs, outputs and mathematics.
- `research_report.pdf`: scientific report .

## Data

`SPY_2015_2025.csv` is included so that the study runs without internet. `download_data.py` shows how the source data was originally downloaded; rerunning it may retrieve revised prices and lead to slightly different results.
