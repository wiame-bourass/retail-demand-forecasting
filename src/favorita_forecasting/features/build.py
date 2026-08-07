from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

SERIES_KEYS = ["store_nbr", "family"]


@dataclass(frozen=True)
class FeatureSpec:
    feature_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]


def _days_since_true(values: pd.Series) -> pd.Series:
    out = np.full(len(values), np.nan, dtype=float)
    last = None
    for idx, flag in enumerate(values.fillna(False).astype(bool).to_numpy()):
        if flag:
            last = idx
            out[idx] = 0.0
        elif last is not None:
            out[idx] = float(idx - last)
    return pd.Series(out, index=values.index)


def _streak(values: pd.Series) -> pd.Series:
    result = []
    run = 0
    for flag in values.fillna(False).astype(bool):
        run = run + 1 if flag else 0
        result.append(run)
    return pd.Series(result, index=values.index, dtype="int32")


def _prior_expanding_mean(frame: pd.DataFrame, mask_col: str) -> pd.Series:
    result = pd.Series(index=frame.index, dtype=float)
    for _, group in frame.groupby(SERIES_KEYS, sort=False):
        sales = group["sales"]
        mask = group[mask_col].fillna(False).astype(bool)
        cumulative_sum = sales.where(mask).fillna(0).cumsum().shift(1)
        cumulative_count = mask.astype(int).cumsum().shift(1)
        result.loc[group.index] = cumulative_sum / cumulative_count.replace(0, np.nan)
    return result


def build_features(panel: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, FeatureSpec]:
    frame = panel.copy().sort_values(SERIES_KEYS + ["date"]).reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"])

    frame["day_of_week"] = frame["date"].dt.dayofweek.astype("int8")
    frame["day_of_month"] = frame["date"].dt.day.astype("int8")
    frame["month"] = frame["date"].dt.month.astype("int8")
    frame["week_of_year"] = frame["date"].dt.isocalendar().week.astype("int16")
    frame["year"] = frame["date"].dt.year.astype("int16")
    frame["is_weekend"] = frame["day_of_week"].ge(5)
    frame["is_month_start"] = frame["date"].dt.is_month_start
    frame["is_month_end"] = frame["date"].dt.is_month_end
    frame["dow_sin"] = np.sin(2 * np.pi * frame["day_of_week"] / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * frame["day_of_week"] / 7)
    frame["month_sin"] = np.sin(2 * np.pi * frame["month"] / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame["month"] / 12)

    grouped_sales = frame.groupby(SERIES_KEYS, sort=False)["sales"]
    for lag in cfg.get("sales_lags", [1, 7, 14, 21, 28]):
        frame[f"sales_lag_{lag}"] = grouped_sales.shift(int(lag))

    shifted_sales = grouped_sales.shift(1)
    for window in cfg.get("rolling_windows", [7, 14, 28]):
        window = int(window)
        grouped_shifted = shifted_sales.groupby(
            [frame[k] for k in SERIES_KEYS],
            sort=False,
        )

        frame[f"sales_roll_mean_{window}"] = grouped_shifted.transform(
            lambda s, w=window: s.rolling(
                w,
                min_periods=max(2, w // 3),
            ).mean()
        )

        frame[f"sales_roll_median_{window}"] = grouped_shifted.transform(
            lambda s, w=window: s.rolling(
                w,
                min_periods=max(2, w // 3),
            ).median()
        )

        frame[f"sales_roll_std_{window}"] = grouped_shifted.transform(
            lambda s, w=window: s.rolling(
                w,
                min_periods=max(2, w // 3),
            ).std()
        )
    if {"sales_roll_mean_7", "sales_roll_mean_28"}.issubset(frame.columns):
        frame["recent_trend_7_vs_28"] = frame["sales_roll_mean_7"] - frame["sales_roll_mean_28"]

    frame["days_since_promotion"] = frame.groupby(SERIES_KEYS, group_keys=False)["any_promotion"].apply(
        _days_since_true
    )
    frame["promotion_streak_days"] = frame.groupby(SERIES_KEYS, group_keys=False)["any_promotion"].apply(_streak)
    frame["family_promo_interaction"] = frame["family"].astype(str) + "__" + frame["any_promotion"].astype(int).astype(str)

    if cfg.get("include_promotion_history", True):
        frame["prior_promo_sales_mean"] = _prior_expanding_mean(frame, "any_promotion")
        nonpromo = frame.copy()
        nonpromo["not_promotion"] = ~nonpromo["any_promotion"].fillna(False).astype(bool)
        frame["prior_nonpromo_sales_mean"] = _prior_expanding_mean(nonpromo, "not_promotion")
        frame["prior_promo_uplift"] = (
            frame["prior_promo_sales_mean"] - frame["prior_nonpromo_sales_mean"]
        )

    if cfg.get("include_oil", True) and "dcoilwtico" in frame:
        oil_daily = frame[["date", "dcoilwtico"]].drop_duplicates("date").sort_values("date")
        oil_daily["oil_lag_1"] = oil_daily["dcoilwtico"].shift(1)
        oil_daily["oil_change_7"] = oil_daily["dcoilwtico"] - oil_daily["dcoilwtico"].shift(7)
        frame = frame.merge(oil_daily[["date", "oil_lag_1", "oil_change_7"]], on="date", how="left")
        frame = frame.sort_values(SERIES_KEYS + ["date"]).reset_index(drop=True)

    if cfg.get("include_transactions", False) and "transactions" in frame:
        tx_daily = frame[["date", "store_nbr", "transactions"]].drop_duplicates(["date", "store_nbr"])
        tx_daily = tx_daily.sort_values(["store_nbr", "date"])
        tx_group = tx_daily.groupby("store_nbr", sort=False)["transactions"]
        for lag in cfg.get("transaction_lags", [7, 14, 28]):
            tx_daily[f"transactions_lag_{lag}"] = tx_group.shift(int(lag))
        tx_daily["transactions_roll_mean_28"] = tx_group.shift(1).groupby(tx_daily["store_nbr"]).transform(
            lambda s: s.rolling(28, min_periods=7).mean()
        )
        tx_cols = [c for c in tx_daily if c.startswith("transactions_lag_") or c == "transactions_roll_mean_28"]
        frame = frame.merge(tx_daily[["date", "store_nbr"] + tx_cols], on=["date", "store_nbr"], how="left")
        frame = frame.sort_values(SERIES_KEYS + ["date"]).reset_index(drop=True)

    excluded = {
        "sales",
        "returns_units",
        "date",
        "is_test",
        "transactions",
        "holiday_descriptions",
        "series_start_date",
    }
    feature_columns = [c for c in frame.columns if c not in excluded]
    categorical = [
        c
        for c in [
            "store_nbr",
            "family",
            "city",
            "state",
            "store_type",
            "cluster",
            "holiday_types",
            "family_promo_interaction",
        ]
        if c in feature_columns
    ]
    numeric = [c for c in feature_columns if c not in categorical]
    return frame, FeatureSpec(feature_columns, categorical, numeric)


def assert_target_features_are_causal(panel: pd.DataFrame, features: pd.DataFrame) -> None:
    """Perturb each current target and verify same-row target-derived features do not change."""
    if panel.empty:
        return
    idx = panel.index[len(panel) // 2]
    mutated = panel.copy()
    original = mutated.loc[idx, "sales"]
    if pd.isna(original):
        return
    mutated.loc[idx, "sales"] = float(original) + 10_000.0
    cfg = {"sales_lags": [1, 7], "rolling_windows": [7], "include_promotion_history": True}
    base, _ = build_features(panel, cfg)
    changed, _ = build_features(mutated, cfg)
    causal_cols = [
        c
        for c in base.columns
        if c.startswith(("sales_lag_", "sales_roll_", "prior_"))
        or c == "recent_trend_7_vs_28"
    ]
    for col in causal_cols:
        a, b = base.loc[idx, col], changed.loc[idx, col]
        if (pd.isna(a) and pd.isna(b)) or np.isclose(a, b, equal_nan=True):
            continue
        raise AssertionError(f"Target leakage detected in same-row feature {col}")
