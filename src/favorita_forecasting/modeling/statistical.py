from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger
def croston_forecast(y: np.ndarray, horizon: int, alpha: float = 0.1) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    positive_idx = np.flatnonzero(y > 0)
    if not len(positive_idx):
        return np.zeros(horizon)
    first = int(positive_idx[0])
    z = y[first]
    interval = first + 1.0
    last_positive = first
    for idx in range(first + 1, len(y)):
        if y[idx] > 0:
            gap = idx - last_positive
            z = alpha * y[idx] + (1 - alpha) * z
            interval = alpha * gap + (1 - alpha) * interval
            last_positive = idx
    value = z / max(interval, 1e-9)
    return np.repeat(max(value, 0.0), horizon)


def tsb_forecast(
    y: np.ndarray,
    horizon: int,
    alpha_demand: float = 0.1,
    alpha_probability: float = 0.1,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    positive = y[y > 0]
    if not len(positive):
        return np.zeros(horizon)
    demand = float(positive[0])
    probability = float(y[0] > 0)
    for value in y[1:]:
        occurrence = float(value > 0)
        probability = alpha_probability * occurrence + (1 - alpha_probability) * probability
        if occurrence:
            demand = alpha_demand * value + (1 - alpha_demand) * demand
    return np.repeat(max(probability * demand, 0.0), horizon)


def ets_forecast(y: np.ndarray, horizon: int, seasonal_periods: int = 7) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    y = np.asarray(y, dtype=float)
    seasonal = "add" if len(y) >= seasonal_periods * 2 else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            y,
            trend="add",
            damped_trend=True,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods if seasonal else None,
            initialization_method="estimated",
        ).fit(optimized=True)
    return np.clip(np.asarray(model.forecast(horizon), dtype=float), 0, None)


def sarima_forecast(y: np.ndarray, horizon: int) -> np.ndarray:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y = np.asarray(y, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            y,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=100)
    return np.clip(np.asarray(model.forecast(horizon), dtype=float), 0, None)


def benchmark_statistical_models(
    panel: pd.DataFrame,
    forecast_start: pd.Timestamp,
    horizon: int,
    selected_series: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    forecast_end = forecast_start + pd.Timedelta(days=horizon - 1)
    for series in selected_series.itertuples(index=False):
        group = panel[
            panel["store_nbr"].eq(series.store_nbr) & panel["family"].eq(series.family)
        ].sort_values("date")
        train = group[group["date"].lt(forecast_start)]["sales"].dropna().to_numpy(float)
        actual = group[group["date"].between(forecast_start, forecast_end)]["sales"].to_numpy(float)
        if len(train) < 30 or len(actual) != horizon:
            continue
        candidates = {
            "seasonal_naive_7": np.resize(train[-7:], horizon),
            "croston": croston_forecast(train, horizon),
            "tsb": tsb_forecast(train, horizon),
        }
        try:
            candidates["ets"] = ets_forecast(train, horizon)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ETS forecast skipped for store=%s family=%s: %s",
                series.store_nbr,
                series.family,
                exc,
            )

        try:
            candidates["sarima"] = sarima_forecast(train, horizon)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "SARIMA forecast skipped for store=%s family=%s: %s",
                series.store_nbr,
                series.family,
                exc,
            )
        from favorita_forecasting.evaluation.metrics import metric_bundle

        for model, prediction in candidates.items():
            rows.append(
                {
                    "store_nbr": series.store_nbr,
                    "family": series.family,
                    "demand_class": getattr(series, "demand_class", "unknown"),
                    "model": model,
                    **metric_bundle(actual, prediction),
                }
            )
    return pd.DataFrame(rows)
