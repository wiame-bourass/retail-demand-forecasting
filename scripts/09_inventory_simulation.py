from __future__ import annotations

import pandas as pd
from common import config_argument

from favorita_forecasting.inventory.simulator import simulate_inventory
from favorita_forecasting.storage import read_frame

if __name__ == "__main__":
    _, cfg = config_argument("Compare inventory outcomes for baseline and champion")
    panel = read_frame(cfg.path("processed_dir") / "curated_panel.parquet")
    champion = read_frame(cfg.path("outputs_dir") / "holdout_predictions.parquet")
    baseline = read_frame(cfg.path("outputs_dir") / "holdout_baseline_predictions.parquet")
    cutoff = pd.to_datetime(champion["date"]).min() - pd.Timedelta(days=1)
    result = simulate_inventory(
        panel,
        {"champion": champion, "seasonal_naive_7": baseline},
        cutoff,
        cfg.section("inventory"),
    )
    result.to_csv(cfg.path("outputs_dir") / "inventory_simulation_by_series.csv", index=False)
    summary = result.groupby("policy", as_index=False).agg(
        demand_units=("demand_units", "sum"),
        fulfilled_units=("fulfilled_units", "sum"),
        lost_sales_units=("lost_sales_units", "sum"),
        holding_cost=("holding_cost", "sum"),
        stockout_cost=("stockout_cost", "sum"),
        total_cost=("total_cost", "sum"),
        orders=("orders", "sum"),
    )
    summary["fill_rate"] = summary["fulfilled_units"] / summary["demand_units"].replace(0, pd.NA)
    summary.to_csv(cfg.path("outputs_dir") / "inventory_comparison.csv", index=False)
    print(summary.to_string(index=False))
