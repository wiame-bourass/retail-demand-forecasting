from __future__ import annotations

import numpy as np
import pandas as pd

SERIES_KEYS = ["store_nbr", "family"]


def forecast_baseline(
    panel: pd.DataFrame,
    train_end: pd.Timestamp,
    forecast_dates: list[pd.Timestamp],
    name: str,
) -> pd.DataFrame:
    work = panel.copy().sort_values(SERIES_KEYS + ["date"]).reset_index(drop=True)
    work["date"] = pd.to_datetime(work["date"])
    work.loc[work["date"] > train_end, "sales"] = np.nan

    history = work.loc[work["date"] <= train_end].copy()
    history["dow"] = history["date"].dt.dayofweek
    dow_means = history.groupby(SERIES_KEYS + ["dow"])["sales"].mean()
    series_mean = history.groupby(SERIES_KEYS)["sales"].mean()
    global_mean = float(history["sales"].mean())

    predictions = []
    for current_date in forecast_dates:
        current_idx = work.index[work["date"].eq(current_date)]
        for idx in current_idx:
            row = work.loc[idx]
            key = (row["store_nbr"], row["family"])
            candidate = np.nan
            if name == "seasonal_naive_7":
                past_date = current_date - pd.Timedelta(days=7)
                values = work.loc[
                    work["date"].eq(past_date)
                    & work["store_nbr"].eq(key[0])
                    & work["family"].eq(key[1]),
                    "sales",
                ]
                candidate = values.iloc[0] if len(values) else np.nan
            elif name == "four_week_mean":
                values = []
                for lag in [7, 14, 21, 28]:
                    past_date = current_date - pd.Timedelta(days=lag)
                    match = work.loc[
                        work["date"].eq(past_date)
                        & work["store_nbr"].eq(key[0])
                        & work["family"].eq(key[1]),
                        "sales",
                    ]
                    if len(match) and pd.notna(match.iloc[0]):
                        values.append(float(match.iloc[0]))
                candidate = np.mean(values) if values else np.nan
            elif name == "dow_mean":
                candidate = dow_means.get((key[0], key[1], current_date.dayofweek), np.nan)
            else:
                raise ValueError(name)
            if pd.isna(candidate):
                candidate = series_mean.get(key, global_mean)
            candidate = max(float(candidate), 0.0)
            work.loc[idx, "sales"] = candidate
            predictions.append(
                {
                    "date": current_date,
                    "store_nbr": key[0],
                    "family": key[1],
                    "prediction": candidate,
                    "model": name,
                }
            )
    return pd.DataFrame(predictions)
