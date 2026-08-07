import pandas as pd

from favorita_forecasting.modeling.backtesting import temporal_windows


def test_holdout_is_after_validation_and_has_16_days():
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    validation, holdout = temporal_windows(pd.Series(dates), horizon=16, validation_folds=2, holdout_days=16)
    assert (holdout.forecast_end - holdout.forecast_start).days + 1 == 16
    assert validation[-1].forecast_end < holdout.forecast_start
    assert validation[0].forecast_start < validation[1].forecast_start
