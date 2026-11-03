from data_loader import load_prices
from features import add_features, research_data
from split_data import split_train_test
from predict import predict_all
from metrics import calculate_metrics

prices = load_prices()
print("1. Prix chargés :", len(prices))

df = add_features(prices)
data = research_data(df)
print("2. Rendements et volatilité calculés :", len(data))

train, test = split_train_test(data)
print("3. Train :", len(train), "| Test :", len(test))

ols, quantile, predictions = predict_all(train, test)
print("4. Coefficient OLS :", round(ols.params["volatility_20d"], 4))
print("5. Coefficient quantile :", round(quantile.params["volatility_20d"], 4))

for name, values in predictions.items():
    result = calculate_metrics(test["next_day_abs_return"], values)
    print("6.", name, result)
