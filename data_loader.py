import pandas as pd
from config import DATA_FILE


def load_prices():
    df = pd.read_csv(DATA_FILE, parse_dates=["Date"], index_col="Date")
    df = df.sort_index()
    if df.index.has_duplicates:
        raise ValueError("Dates dupliquées dans le CSV")
    if (df["Close"] <= 0).any():
        raise ValueError("Prix de clôture non positif")
    return df
