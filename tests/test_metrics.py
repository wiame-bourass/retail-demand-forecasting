import numpy as np

from favorita_forecasting.evaluation.metrics import metric_bundle, normalized_bias, rmsle, wape


def test_perfect_metrics():
    y = [0, 1, 2, 10]
    assert wape(y, y) == 0
    assert normalized_bias(y, y) == 0
    assert rmsle(y, y) == 0
    metrics = metric_bundle(y, y)
    assert metrics["mae"] == 0
    assert metrics["forecast_accuracy"] == 1


def test_bias_sign():
    y = [10, 10]
    assert normalized_bias(y, [8, 8]) < 0
    assert normalized_bias(y, [12, 12]) > 0


def test_zero_denominator_returns_nan():
    assert np.isnan(wape([0, 0], [1, 2]))
