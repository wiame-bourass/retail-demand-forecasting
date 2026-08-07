import pandas as pd

from favorita_forecasting.inventory.simulator import simulate_inventory


def test_inventory_simulation_returns_costs():
    dates = pd.date_range("2021-01-01", periods=20, freq="D")
    panel = pd.DataFrame(
        {
            "date": dates,
            "store_nbr": 1,
            "family": "A",
            "sales": [10.0] * 20,
            "is_test": False,
        }
    )
    forecast_dates = dates[-5:]
    scored = pd.DataFrame(
        {
            "date": forecast_dates,
            "store_nbr": 1,
            "family": "A",
            "actual": [10.0] * 5,
            "prediction": [10.0] * 5,
        }
    )
    result = simulate_inventory(
        panel,
        {"model": scored},
        cutoff=dates[-6],
        cfg={
            "lead_time_days": 2,
            "review_period_days": 1,
            "service_level": 0.95,
            "holding_cost_per_unit_day": 0.1,
            "stockout_cost_per_unit": 2.0,
            "initial_stock_days_of_cover": 3,
        },
    )
    assert len(result) == 1
    assert result.loc[0, "total_cost"] >= 0
    assert 0 <= result.loc[0, "fill_rate"] <= 1
