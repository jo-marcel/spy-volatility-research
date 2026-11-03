# Reading Guide — Modular SPY Volatility Research Project

## The Project in One Sentence

Using historical SPY prices, we investigate whether the standard deviation of daily returns over the previous 20 trading sessions helps predict the **magnitude** of the next trading session's return.

The objective is to predict **how large a price movement might be**, not whether the price will increase or decrease.

## How to Run the Project

From the project's root directory, execute:

```bash
python3 -m pip install -r requirements.txt
python3 run_steps.py
python3 main.py
python3 -m unittest discover -s tests -v
```

`run_steps.py` provides a step-by-step explanation of the research workflow.

`main.py` runs the complete experiment and regenerates the figures and result tables.

## 0 — `config.py`: Shared Configuration

This module defines the parameters used throughout the project:

- `WINDOW = 20`: Number of trading sessions used to calculate historical volatility.
- `TRADING_DAYS = 252`: Conventional number of trading sessions per year.
- `TEST_START = "2023-01-01"`: Beginning of the test period.

It also defines the paths to the data, figures, and results directories.

Keeping these settings in one place prevents duplication and makes the project easier to maintain.

## 1 — `data_loader.py`: Loading Market Data

The following instruction loads the CSV file:

```python
pd.read_csv(
    ...,
    parse_dates=["Date"],
    index_col="Date"
)
```

`parse_dates` converts the Date column into actual date values, while `index_col` uses these dates as the DataFrame index.

`sort_index()` ensures that observations are ordered chronologically.

The module also checks for duplicate dates and invalid prices, including prices that are zero or negative.

**Input:** CSV file containing historical SPY prices.

**Output:** A pandas DataFrame containing validated and chronologically ordered price observations.

## 2 — `features.py`: Feature Engineering

This module transforms raw prices into variables suitable for statistical analysis.

### Simple returns

```python
Close.pct_change()
```

The simple return measures the relative price change between two consecutive trading sessions.

**Formula:**

R(t) = (C(t) - C(t-1)) / C(t-1)

### Logarithmic returns

```python
np.log(Close / Close.shift(1))
```

Logarithmic returns are calculated as:

r(t) = ln(C(t) / C(t-1))

They are particularly useful in financial analysis because logarithmic returns are additive across consecutive periods.

### Historical daily volatility

```python
log_returns.rolling(20).std()
```

This calculates the sample standard deviation of the most recent 20 daily logarithmic returns.

The resulting variable measures recent historical volatility using information available up to the current trading session.

### Annualized volatility

Annualized volatility is calculated by multiplying daily volatility by:

sqrt(252)

This transformation is used for historical visualization only.

### Prediction target

```python
log_returns.shift(-1).abs()
```

The target variable is:

y(t) = |r(t+1)|

It represents the absolute logarithmic return of the following trading session.

The `.shift(-1)` operation aligns tomorrow's return with today's explanatory variables.

The `.abs()` operation removes the direction of the return, retaining only its magnitude.

The first return, the initial rolling-volatility observations, and the final target are undefined. The `research_data()` function removes incomplete rows.

**Final dataset: 2,745 observations.**

**Why is annualized volatility not used as the model's predictor?**

The model uses daily volatility because its target is a daily return magnitude. Annualization is reserved for visualization, keeping the units of the model coefficients straightforward to interpret.

## 3 — `split_data.py`: Separating Past and Future Observations

The dataset is divided chronologically:

```python
train = data.loc[
    data.index < "2023-01-01"
].iloc[:-1]

test = data.loc[
    data.index >= "2023-01-01"
]
```

The training period covers 2015–2022, while the test period covers 2023–2025.

### Why remove the final training row?

The last feature observation from 2022 is associated with the return of the first trading session in 2023.

Keeping this observation in the training set would introduce information from the test period into model estimation.

Removing it prevents this particular form of target leakage.

**Final split:**

- Training set: 1,993 observations.
- Test set: 751 observations.

## 4 — `models.py`: Regression Models

This module implements two statistical approaches.

### Ordinary Least Squares (OLS)

The `make_X()` function adds an intercept to the historical volatility variable:

```python
X = [1, volatility_20d]
```

The regression model is:

|r(t+1)| = alpha + beta × volatility(t) + error(t)

OLS estimates the coefficients by minimizing the sum of squared prediction errors.

It is therefore naturally associated with squared-error evaluation metrics.

### Median Quantile Regression

Quantile regression with `q = 0.5` estimates the conditional median of the next-session absolute return.

Unlike OLS, which estimates a conditional mean, median quantile regression minimizes an objective based on absolute deviations.

Both models are fitted **using training data only**.

Once estimated, their coefficients remain fixed during the 2023–2025 holdout evaluation.

## 5 — `baselines.py`: Simple Benchmark Models

A forecasting model should be compared against simple alternatives.

This project uses two constant baselines.

### Training Mean Baseline

Every prediction is equal to the average target value observed during training.

The mean is the optimal constant predictor under squared-error loss.

### Training Median Baseline

Every prediction is equal to the median target value observed during training.

The median is the optimal constant predictor under absolute-error loss.

Both baselines are calculated exclusively from the training dataset.

These benchmarks help determine whether the regression models provide meaningful improvements over simple predictions.

## 6 — `predict.py`: Generating Predictions

This module generates predictions from the two regression models and the two baselines.

For the regression models, predictions are calculated using the test features.

For the baselines, the corresponding training mean or median is repeated across the test observations.

The following operation ensures that predicted return magnitudes cannot be negative:

```python
np.maximum(prediction, 0)
```

This is appropriate because the target represents an absolute return.

**Output:** A dictionary containing four prediction series:

- OLS regression
- Median quantile regression
- Training mean baseline
- Training median baseline

## 7 — `metrics.py`: Measuring Prediction Errors

This module evaluates forecasting accuracy using two metrics.

### Mean Absolute Error (MAE)

MAE = mean(|actual - prediction|)

MAE measures the average absolute difference between predicted and observed values.

It is relatively less sensitive to extreme errors than squared-error metrics.

### Root Mean Squared Error (RMSE)

RMSE = sqrt(mean((actual - prediction)²))

RMSE penalizes larger errors more strongly because deviations are squared before averaging.

A 5% improvement in MAE means that the average absolute prediction error decreased by 5% relative to a benchmark.

It does not mean that the model correctly predicted 5% more trading sessions.

## 8 — `inference.py`: Descriptive Statistical Inference

This module examines the statistical relationship between historical volatility and future absolute returns.

The full-sample Pearson correlation is approximately:

**r = 0.4572**

This indicates a positive association between recent historical volatility and the magnitude of the following session's return.

An OLS regression with heteroskedasticity-and-autocorrelation-consistent (HAC) standard errors, using 20 lags, produces:

- Estimated slope: approximately 0.6207.
- R-squared: approximately 0.209.

HAC standard errors account for potential heteroskedasticity and autocorrelation when estimating coefficient uncertainty.

Importantly, this analysis uses the **entire dataset**.

It is intended to describe the statistical association, not to measure out-of-sample forecasting performance.

## 9 — `holdout.py`: The 2023–2025 Evaluation

This module coordinates the main forecasting experiment.

It uses:

1. `split_data.py` to separate training and test observations.
2. `predict.py` to generate forecasts.
3. `metrics.py` to evaluate their accuracy.

It also saves the predictions and performance metrics.

### Out-of-Sample Results

| Model | MAE | RMSE |
|---|---:|---:|
| OLS | 0.004638 | **0.006953** |
| Quantile Regression (50%) | 0.004422 | 0.007036 |
| Training Mean Baseline | 0.004895 | 0.007109 |
| Training Median Baseline | **0.004406** | 0.007269 |

**Main findings:**

OLS achieves the lowest RMSE, while the constant training median achieves the lowest MAE.

Both conclusions are important.

The results demonstrate that model rankings depend on the evaluation metric. A more sophisticated model does not necessarily outperform a simple benchmark under every loss function.

## 10 — `walk_forward.py`: Expanding-Window Evaluation

This module performs a series of chronological forecasting experiments from 2019 to 2025.

For each evaluation year:

1. Train the models using observations from earlier years.
2. Remove the boundary observation to prevent target leakage.
3. Generate predictions for the current evaluation year.
4. Calculate prediction errors.
5. Expand the training period for the following year.

The models are re-estimated annually as additional historical observations become available.

This approach helps investigate how forecasting performance changes over time.

However, these experiments are retrospective diagnostics. They should not be presented as fully independent validation because some results were already examined during model development.

## 11 — `regimes.py`: Volatility Regime Analysis

This module divides historical volatility into three categories:

- Low volatility
- Medium volatility
- High volatility

The thresholds are defined using the 33rd and 67th percentiles of historical volatility calculated from the training data.

These fixed thresholds are then applied to the test dataset.

For each volatility regime, the module measures the average magnitude of the following session's return.

This provides a descriptive comparison of market behavior under different volatility conditions.

The thresholds must not be estimated from the test dataset because that would introduce information from the evaluation period.

## 12 — `plots_history.py` and `plots_models.py`: Data Visualization

The visualization code is divided into two modules.

### `plots_history.py`

Creates Figures 01–04:

1. Historical SPY adjusted prices.
2. Daily logarithmic returns.
3. Historical rolling volatility.
4. Scatter plot of historical volatility against next-session absolute returns.

These figures help explain the dataset and explore its statistical properties.

### `plots_models.py`

Creates Figures 05–09:

5. Observed versus predicted return magnitudes.
6. Comparison of MAE and RMSE across models.
7. Year-by-year holdout performance.
8. Expanding-window forecasting performance.
9. Return magnitudes across volatility regimes.

Each plotting function saves its output in the `figures/` directory.

Together, these visualizations support both exploratory analysis and evaluation of the forecasting models.

## 13 — `export_summary.py`: Exporting Research Results

This module saves the experiment's main numerical results to:

```text
results/summary.json
```

The JSON file includes information such as:

- Dataset sizes.
- Estimated model coefficients.
- Evaluation metrics.
- Volatility regime thresholds.

Storing results in a structured file improves reproducibility and reduces the risk of manually copying incorrect values into reports.

## 14 — `main.py`: The Main Orchestrator

`main.py` coordinates the complete research workflow.

It calls the appropriate modules in sequence to:

1. Load market data.
2. Construct features and targets.
3. Perform statistical analysis.
4. Train and evaluate forecasting models.
5. Run chronological validation experiments.
6. Generate visualizations.
7. Export the results.

The file does not need to contain the detailed mathematical implementation of every operation.

Instead, it connects specialized modules, making the codebase easier to read, test, maintain, and extend.

This separation of responsibilities is the central principle behind the project's modular architecture.

## 15 — `tests/test_project.py`: Automated Validation

The project includes three automated tests.

### Test 1 — No Future Information in Historical Volatility

Checks that adding future observations does not change previously calculated historical volatility values.

### Test 2 — Chronological Data Separation

Verifies that the training and test periods do not overlap.

### Test 3 — Evaluation Metrics

Checks MAE and RMSE calculations using a simple example with known results.

These tests help ensure that fundamental assumptions remain valid when the code is modified.