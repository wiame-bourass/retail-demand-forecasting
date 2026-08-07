import pandas as pd

from favorita_forecasting.features.build import build_features


def test_oil_lag_is_date_based_not_cross_series():
    rows = []
    dates = pd.date_range("2021-01-01", periods=10, freq="D")
    for family in ["A", "B"]:
        for i, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "store_nbr": 1,
                    "family": family,
                    "sales": float(i + 1),
                    "returns_units": 0.0,
                    "is_test": False,
                    "any_promotion": False,
                    "dcoilwtico": 50.0 + i,
                }
            )
    panel = pd.DataFrame(rows)
    features, _ = build_features(
        panel,
        {"sales_lags": [1], "rolling_windows": [3], "include_oil": True},
    )
    date = dates[5]
    values = features.loc[features["date"].eq(date), "oil_lag_1"].unique()
    assert len(values) == 1
    assert values[0] == 54.0
