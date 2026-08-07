from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def _arrays(y_true: Iterable[float], y_pred: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    mask = np.isfinite(true) & np.isfinite(pred)
    return true[mask], pred[mask]


def wape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    true, pred = _arrays(y_true, y_pred)
    denominator = np.abs(true).sum()
    return float(np.abs(true - pred).sum() / denominator) if denominator > 0 else np.nan


def normalized_bias(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    true, pred = _arrays(y_true, y_pred)
    denominator = np.abs(true).sum()
    return float((pred - true).sum() / denominator) if denominator > 0 else np.nan


def rmsle(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    true, pred = _arrays(y_true, y_pred)
    true = np.clip(true, 0, None)
    pred = np.clip(pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(true)) ** 2))) if len(true) else np.nan


def metric_bundle(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    true, pred = _arrays(y_true, y_pred)
    if not len(true):
        return {"mae": np.nan, "wape": np.nan, "bias": np.nan, "rmsle": np.nan, "rmse": np.nan, "forecast_accuracy": np.nan, "n": 0}
    value_wape = wape(true, pred)
    return {
        "mae": float(mean_absolute_error(true, pred)),
        "wape": value_wape,
        "bias": normalized_bias(true, pred),
        "rmsle": rmsle(true, pred),
        "rmse": float(np.sqrt(np.mean((true - pred) ** 2))),
        "forecast_accuracy": float(1 - value_wape) if np.isfinite(value_wape) else np.nan,
        "n": len(true),
    }


def weighted_rmsle(
    y_true: Iterable[float], y_pred: Iterable[float], weights: Iterable[float]
) -> float:
    true = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    weight = np.asarray(list(weights), dtype=float)
    length = min(len(true), len(pred), len(weight))
    true, pred, weight = true[:length], pred[:length], weight[:length]
    mask = np.isfinite(true) & np.isfinite(pred) & np.isfinite(weight) & (weight >= 0)
    if not mask.any() or weight[mask].sum() == 0:
        return np.nan
    errors = (np.log1p(np.clip(pred[mask], 0, None)) - np.log1p(np.clip(true[mask], 0, None))) ** 2
    return float(np.sqrt(np.average(errors, weights=weight[mask])))


def segmented_metrics(
    predictions: pd.DataFrame,
    group_columns: list[str],
    actual_col: str = "actual",
    prediction_col: str = "prediction",
) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(group_columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, keys))
        row.update(metric_bundle(group[actual_col], group[prediction_col]))
        rows.append(row)
    return pd.DataFrame(rows)
