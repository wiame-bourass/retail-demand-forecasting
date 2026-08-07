from __future__ import annotations

from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

SERIES_KEYS = ["store_nbr", "family"]


def _residual_std_from_history(panel: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    history = panel.loc[(~panel["is_test"]) & panel["date"].le(cutoff)].copy()
    history = history.sort_values(SERIES_KEYS + ["date"])
    history["lag7"] = history.groupby(SERIES_KEYS)["sales"].shift(7)
    history["residual"] = history["sales"] - history["lag7"]
    return history.groupby(SERIES_KEYS, as_index=False)["residual"].std().rename(columns={"residual": "residual_std"})


def _simulate_one(
    frame: pd.DataFrame,
    residual_std: float,
    cfg: dict[str, Any],
) -> dict[str, float]:
    frame = frame.sort_values("date").reset_index(drop=True)
    lead = int(cfg["lead_time_days"])
    review = int(cfg.get("review_period_days", 1))
    z = NormalDist().inv_cdf(float(cfg["service_level"]))
    holding_cost = float(cfg["holding_cost_per_unit_day"])
    stockout_cost = float(cfg["stockout_cost_per_unit"])
    cover = int(cfg.get("initial_stock_days_of_cover", 5))
    stock = float(frame["prediction"].head(cover).sum())
    arrivals: dict[int, float] = {}
    total_holding = total_stockout = lost = fulfilled = total_demand = 0.0
    orders = 0

    for day in range(len(frame)):
        stock += arrivals.pop(day, 0.0)
        demand = max(float(frame.loc[day, "actual"]), 0.0)
        available = stock
        served = min(available, demand)
        shortage = max(demand - available, 0.0)
        stock = max(available - demand, 0.0)
        fulfilled += served
        lost += shortage
        total_demand += demand
        total_holding += stock * holding_cost
        total_stockout += shortage * stockout_cost

        end = min(len(frame), day + lead + review)
        demand_forecast = float(frame.loc[day:end - 1, "prediction"].sum()) if end > day else 0.0
        safety_stock = z * max(float(residual_std or 0.0), 0.0) * np.sqrt(max(lead + review, 1))
        inventory_position = stock + sum(quantity for arrival_day, quantity in arrivals.items() if arrival_day >= day)
        order_qty = max(demand_forecast + safety_stock - inventory_position, 0.0)
        if order_qty > 0:
            arrivals[day + lead] = arrivals.get(day + lead, 0.0) + order_qty
            orders += 1

    return {
        "demand_units": total_demand,
        "fulfilled_units": fulfilled,
        "lost_sales_units": lost,
        "fill_rate": fulfilled / total_demand if total_demand > 0 else np.nan,
        "holding_cost": total_holding,
        "stockout_cost": total_stockout,
        "total_cost": total_holding + total_stockout,
        "orders": orders,
    }


def simulate_inventory(
    panel: pd.DataFrame,
    forecast_sets: dict[str, pd.DataFrame],
    cutoff: pd.Timestamp,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    residual = _residual_std_from_history(panel, cutoff)
    rows = []
    for policy, predictions in forecast_sets.items():
        scored = predictions.copy()
        for keys, group in scored.groupby(SERIES_KEYS, sort=False):
            match = residual.loc[
                residual["store_nbr"].eq(keys[0]) & residual["family"].eq(keys[1]), "residual_std"
            ]
            std = float(match.iloc[0]) if len(match) and pd.notna(match.iloc[0]) else 0.0
            result = _simulate_one(group, std, cfg)
            rows.append({"policy": policy, "store_nbr": keys[0], "family": keys[1], **result})
    return pd.DataFrame(rows)
