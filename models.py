import statsmodels.api as sm


def make_X(data):
    return sm.add_constant(data[["volatility_20d"]], has_constant="add")


def train_models(train):
    X = make_X(train)
    y = train["next_day_abs_return"]
    ols = sm.OLS(y, X).fit()
    quantile = sm.QuantReg(y, X).fit(q=0.5, max_iter=10000)
    return ols, quantile
