import pandas as pd

from favorita_forecasting.modeling.baselines import forecast_baseline


def test_seasonal_naive_uses_lag_7():
    dates = pd.date_range("2021-01-01", periods=20, freq="D")
    panel = pd.DataFrame(
        {
            "date": dates,
            "store_nbr": 1,
            "family": "A",
            "sales": range(20),
        }
    )
    train_end = dates[13]
    result = forecast_baseline(panel, train_end, [dates[14]], "seasonal_naive_7")
    assert result.loc[0, "prediction"] == 7
