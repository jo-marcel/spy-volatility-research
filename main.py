from data_loader import load_prices
from features import add_features, research_data
from inference import analyze_association
from holdout import run_holdout
from walk_forward import run_walk_forward
from regimes import analyze_regimes
from plots_history import plot_history
from plots_models import plot_models
from export_summary import save_summary


def main():
    prices = load_prices()
    df = add_features(prices)
    data = research_data(df)

    correlation, hac = analyze_association(data)
    train, test, ols, quantile, holdout, scores = run_holdout(data)
    _, walk_scores = run_walk_forward(data)
    regimes, low, high = analyze_regimes(train, test)

    plot_history(df, data, correlation)
    plot_models(holdout, scores, walk_scores, regimes)
    save_summary(df, data, train, test, correlation, hac,
                 ols, quantile, scores, walk_scores, low, high)

    print("Prix:", len(df), "| Observations utiles:", len(data))
    print("Train:", len(train), "| Test:", len(test))
    print("Corrélation:", round(correlation, 4))
    print("\nRésultats sur 2023-2025:")
    print(scores[scores["period"] == "2023-2025"].to_string(index=False))
    print("\n9 graphics in  figures/ ; Arrays in  results/")


if __name__ == "__main__":
    main()
