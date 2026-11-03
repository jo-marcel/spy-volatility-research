import statsmodels.api as sm
from models import make_X


def analyze_association(data):
    correlation = data["volatility_20d"].corr(data["next_day_abs_return"])
    model = sm.OLS(data["next_day_abs_return"], make_X(data)).fit(
        cov_type="HAC", cov_kwds={"maxlags": 20}
    )
    return float(correlation), model
