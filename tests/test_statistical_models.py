import numpy as np

from favorita_forecasting.modeling.statistical import croston_forecast, tsb_forecast


def test_intermittent_forecasts_are_nonnegative_and_correct_length():
    y = np.array([0, 0, 4, 0, 0, 5, 0, 0, 0, 3], dtype=float)
    for prediction in [croston_forecast(y, 6), tsb_forecast(y, 6)]:
        assert len(prediction) == 6
        assert np.all(prediction >= 0)
