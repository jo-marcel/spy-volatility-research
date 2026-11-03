import numpy as np
from config import WINDOW, TRADING_DAYS


def add_features(prices):
    df = prices.copy()
    df["simple_return"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["volatility_20d"] = df["log_return"].rolling(WINDOW).std()
    df["annualized_volatility"] = df["volatility_20d"] * np.sqrt(TRADING_DAYS)
    df["next_day_abs_return"] = df["log_return"].shift(-1).abs()
    return df


def research_data(df):
    columns = ["volatility_20d", "next_day_abs_return"]
    return df[columns].dropna()
