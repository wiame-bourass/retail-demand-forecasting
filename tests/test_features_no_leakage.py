import numpy as np
import pandas as pd

from favorita_forecasting.features.build import build_features


def sample_panel():
    dates = pd.date_range("2021-01-01", periods=40, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "store_nbr": 1,
            "family": "A",
            "sales": np.arange(40, dtype=float),
            "any_promotion": [False] * 40,
            "is_test": False,
            "returns_units": 0.0,
        }
    )


def test_same_row_target_is_not_in_lags_or_rollings():
    panel = sample_panel()
    features, _ = build_features(panel, {"sales_lags": [1, 7], "rolling_windows": [7]})
    row = features.iloc[20]
    assert row["sales_lag_1"] == panel.iloc[19]["sales"]
    expected = panel.iloc[13:20]["sales"].mean()
    assert np.isclose(row["sales_roll_mean_7"], expected)


def test_current_target_perturbation_does_not_change_same_row_features():
    panel = sample_panel()
    base, _ = build_features(panel, {"sales_lags": [1, 7], "rolling_windows": [7]})
    changed_panel = panel.copy()
    changed_panel.loc[20, "sales"] = 99999
    changed, _ = build_features(changed_panel, {"sales_lags": [1, 7], "rolling_windows": [7]})
    columns = ["sales_lag_1", "sales_lag_7", "sales_roll_mean_7", "sales_roll_std_7"]
    for column in columns:
        assert np.isclose(base.loc[20, column], changed.loc[20, column], equal_nan=True)
