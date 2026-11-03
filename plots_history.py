import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import FIGURES


def save(name):
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=180)
    plt.close()


def plot_history(df, data, correlation):
    plt.figure(figsize=(11, 5))
    plt.plot(df.index, df["Close"])
    plt.title("SPY adjusted close | 2015-2025")
    plt.xlabel("Date")
    plt.ylabel("USD")
    save("01_spy_price.png")

    plt.figure(figsize=(11, 5))
    plt.plot(df.index, df["log_return"] * 100, linewidth=0.7)
    plt.title("SPY daily log returns")
    plt.xlabel("Date")
    plt.ylabel("Return (%)")
    save("02_daily_returns.png")

    plt.figure(figsize=(11, 5))
    plt.plot(df.index, df["annualized_volatility"] * 100)
    plt.title("20-session annualized realized volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility (%)")
    save("03_rolling_volatility.png")

    plt.figure(figsize=(9, 6))
    plt.scatter(data["volatility_20d"] * 100,
                data["next_day_abs_return"] * 100, alpha=0.18, s=12)
    plt.title(f"Volatility vs next-day magnitude | r = {correlation:.3f}")
    plt.xlabel("Daily historical volatility (%)")
    plt.ylabel("Next-day absolute return (%)")
    save("04_correlation.png")
