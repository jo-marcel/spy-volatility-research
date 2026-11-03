import numpy as np
from models import make_X, train_models
from baselines import constant_predictions


def predict_all(train, test):
    ols, quantile = train_models(train)
    X_test = make_X(test)
    predictions = {
        "OLS": np.maximum(ols.predict(X_test).to_numpy(), 0),
        "Quantile_50": np.maximum(quantile.predict(X_test).to_numpy(), 0),
    }
    predictions.update(constant_predictions(train, test))
    return ols, quantile, predictions
