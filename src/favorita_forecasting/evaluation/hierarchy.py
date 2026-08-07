from __future__ import annotations

import numpy as np
import pandas as pd

from favorita_forecasting.evaluation.metrics import metric_bundle


def hierarchy_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    levels = {
        "total": [],
        "state": ["state"],
        "city": ["city"],
        "store": ["store_nbr"],
        "family": ["family"],
        "state_family": ["state", "family"],
        "city_family": ["city", "family"],
        "store_family": ["store_nbr", "family"],
    }
    rows = []
    for level, keys in levels.items():
        grouped = predictions.groupby(["date"] + keys, as_index=False).agg(
            actual=("actual", "sum"), prediction=("prediction", "sum")
        )
        metrics = metric_bundle(grouped["actual"], grouped["prediction"])
        rows.append({"level": level, **metrics, "groups": grouped[keys].drop_duplicates().shape[0] if keys else 1})
    return pd.DataFrame(rows)


def bottom_up_coherence(predictions: pd.DataFrame, tolerance: float = 1e-8) -> dict[str, float | bool]:
    bottom = predictions.groupby("date", as_index=False)["prediction"].sum().rename(columns={"prediction": "bottom_sum"})
    total = predictions.groupby("date", as_index=False)["prediction"].sum().rename(columns={"prediction": "total_sum"})
    merged = bottom.merge(total, on="date")
    max_gap = float(np.abs(merged["bottom_sum"] - merged["total_sum"]).max())
    return {"coherent": bool(max_gap <= tolerance), "max_absolute_gap": max_gap}
