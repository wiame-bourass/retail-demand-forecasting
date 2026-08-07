from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from favorita_forecasting.evaluation.metrics import metric_bundle
from favorita_forecasting.features.build import FeatureSpec, build_features
from favorita_forecasting.mlflow_utils import log_experiment
from favorita_forecasting.modeling.baselines import forecast_baseline
from favorita_forecasting.modeling.estimators import build_estimator
from favorita_forecasting.storage import write_frame
from favorita_forecasting.utils import ensure_dir, write_json

LOGGER = logging.getLogger(__name__)
BASELINES = {"seasonal_naive_7", "four_week_mean", "dow_mean"}


@dataclass(frozen=True)
class TemporalWindow:
    fold: str
    train_end: pd.Timestamp
    forecast_start: pd.Timestamp
    forecast_end: pd.Timestamp


def temporal_windows(
    historical_dates: pd.Series,
    horizon: int,
    validation_folds: int,
    holdout_days: int,
) -> tuple[list[TemporalWindow], TemporalWindow]:
    unique_dates = pd.Series(pd.to_datetime(historical_dates).dropna().unique()).sort_values().reset_index(drop=True)
    hist_end = unique_dates.max()
    holdout_start = hist_end - pd.Timedelta(days=holdout_days - 1)
    holdout = TemporalWindow("holdout", holdout_start - pd.Timedelta(days=1), holdout_start, hist_end)
    windows = []
    cursor_end = holdout_start - pd.Timedelta(days=1)
    for index in range(validation_folds, 0, -1):
        forecast_end = cursor_end - pd.Timedelta(days=(index - 1) * horizon)
        forecast_start = forecast_end - pd.Timedelta(days=horizon - 1)
        windows.append(
            TemporalWindow(
                fold=f"validation_{validation_folds - index + 1}",
                train_end=forecast_start - pd.Timedelta(days=1),
                forecast_start=forecast_start,
                forecast_end=forecast_end,
            )
        )
    return windows, holdout


def parameter_grid(config: dict[str, Any]) -> list[dict[str, Any]]:
    if not config:
        return [{}]
    keys = list(config)
    values = [v if isinstance(v, list) else [v] for v in config.values()]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _fit_model(
    panel: pd.DataFrame,
    train_end: pd.Timestamp,
    model_name: str,
    params: dict[str, Any],
    feature_cfg: dict[str, Any],
    seed: int,
    target_transform: str,
):
    feature_frame, spec = build_features(panel, feature_cfg)
    train = feature_frame.loc[
        feature_frame["date"].le(train_end) & feature_frame["sales"].notna()
    ].copy()

    # Training-only feature screening: missingness, constants and near-duplicate numeric variables.
    threshold_missing = float(feature_cfg.get("high_missingness_threshold", 0.95))
    selected = [
        column
        for column in spec.feature_columns
        if train[column].isna().mean() <= threshold_missing and train[column].nunique(dropna=False) > 1
    ]
    selected_numeric = [column for column in spec.numeric_columns if column in selected]
    correlation_threshold = float(feature_cfg.get("high_correlation_threshold", 0.995))
    if selected_numeric:
        correlation = train[selected_numeric].corr().abs()
        dropped: set[str] = set()
        for index, left in enumerate(correlation.columns):
            if left in dropped:
                continue
            for right in correlation.columns[index + 1 :]:
                if right not in dropped and pd.notna(correlation.loc[left, right]) and correlation.loc[left, right] >= correlation_threshold:
                    dropped.add(right)
        selected = [column for column in selected if column not in dropped]
    spec = FeatureSpec(
        feature_columns=selected,
        categorical_columns=[column for column in spec.categorical_columns if column in selected],
        numeric_columns=[column for column in spec.numeric_columns if column in selected],
    )
    if not spec.feature_columns:
        raise RuntimeError("Training-only feature screening removed every feature")

    estimator = build_estimator(
        model_name,
        params,
        numeric_columns=spec.numeric_columns,
        categorical_columns=spec.categorical_columns,
        seed=seed,
        target_transform=target_transform,
    )
    estimator.fit(train[spec.feature_columns], train["sales"].astype(float))
    return estimator, spec


def recursive_model_forecast(
    panel: pd.DataFrame,
    train_end: pd.Timestamp,
    forecast_dates: list[pd.Timestamp],
    estimator,
    spec: FeatureSpec,
    feature_cfg: dict[str, Any],
    model_name: str,
) -> pd.DataFrame:
    work = panel.copy().sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    work["date"] = pd.to_datetime(work["date"])
    work.loc[work["date"].gt(train_end), "sales"] = np.nan
    rows = []
    for horizon_step, current_date in enumerate(forecast_dates, start=1):
        feature_frame, _ = build_features(work, feature_cfg)
        mask = feature_frame["date"].eq(current_date)
        current = feature_frame.loc[mask]
        if current.empty:
            continue
        predictions = np.clip(estimator.predict(current[spec.feature_columns]), 0, None)
        work.loc[mask.to_numpy(), "sales"] = predictions
        for row, prediction in zip(current.itertuples(index=False), predictions):
            rows.append(
                {
                    "date": current_date,
                    "store_nbr": row.store_nbr,
                    "family": row.family,
                    "prediction": float(prediction),
                    "model": model_name,
                    "horizon_step": horizon_step,
                    "any_promotion": bool(row.any_promotion),
                }
            )
    return pd.DataFrame(rows)


def forecast_candidate(
    panel: pd.DataFrame,
    window: TemporalWindow,
    model_name: str,
    params: dict[str, Any],
    feature_cfg: dict[str, Any],
    seed: int,
    target_transform: str,
):
    dates = list(pd.date_range(window.forecast_start, window.forecast_end, freq="D"))
    if model_name in BASELINES:
        return forecast_baseline(panel, window.train_end, dates, model_name), None, None
    estimator, spec = _fit_model(
        panel, window.train_end, model_name, params, feature_cfg, seed, target_transform
    )
    predictions = recursive_model_forecast(
        panel, window.train_end, dates, estimator, spec, feature_cfg, model_name
    )
    return predictions, estimator, spec


def attach_actuals(panel: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    actual = panel[["date", "store_nbr", "family", "sales", "any_promotion", "city", "state"]].rename(
        columns={"sales": "actual", "any_promotion": "actual_any_promotion"}
    )
    merged = predictions.merge(actual, on=["date", "store_nbr", "family"], how="left")
    if "any_promotion" not in merged:
        merged["any_promotion"] = merged["actual_any_promotion"]
    else:
        merged["any_promotion"] = merged["any_promotion"].fillna(merged["actual_any_promotion"])
    return merged


def run_validation_selection(
    panel: pd.DataFrame,
    model_cfg: dict[str, Any],
    feature_cfg: dict[str, Any],
    forecast_cfg: dict[str, Any],
    seed: int,
    outputs_dir: Path,
    artifacts_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    historical = panel.loc[~panel["is_test"]]
    windows, holdout = temporal_windows(
        historical["date"],
        int(forecast_cfg["horizon_days"]),
        int(forecast_cfg["validation_folds"]),
        int(forecast_cfg["holdout_days"]),
    )
    rows = []
    candidates = model_cfg.get("candidates", [])
    for model_name in candidates:
        grids = [{}] if model_name in BASELINES else parameter_grid(model_cfg.get(model_name, {}))
        for params in grids:
            candidate_id = f"{model_name}:{json.dumps(params, sort_keys=True)}"
            for window in windows:
                LOGGER.info("Backtest %s on %s", candidate_id, window.fold)
                try:
                    pred, _, _ = forecast_candidate(
                        panel,
                        window,
                        model_name,
                        params,
                        feature_cfg,
                        seed,
                        forecast_cfg.get("target_transform", "log1p"),
                    )
                    scored = attach_actuals(panel, pred)
                    metrics = metric_bundle(scored["actual"], scored["prediction"])
                    promo_metrics = metric_bundle(
                        scored.loc[scored["any_promotion"], "actual"],
                        scored.loc[scored["any_promotion"], "prediction"],
                    )
                    row_result = {
                        "candidate_id": candidate_id,
                        "model": model_name,
                        "params_json": json.dumps(params, sort_keys=True),
                        "fold": window.fold,
                        "train_end": window.train_end,
                        "forecast_start": window.forecast_start,
                        "forecast_end": window.forecast_end,
                        **metrics,
                        "promo_wape": promo_metrics["wape"],
                        "status": "ok",
                    }
                    rows.append(row_result)
                    log_experiment(
                        run_name=f"{model_name}-{window.fold}",
                        params={"model": model_name, **params},
                        metrics={**metrics, "promo_wape": promo_metrics["wape"]},
                        tags={"fold": window.fold, "stage": "validation"},
                    )
                except RuntimeError as exc:
                    LOGGER.warning("Skipping %s: %s", model_name, exc)
                    rows.append(
                        {
                            "candidate_id": candidate_id,
                            "model": model_name,
                            "params_json": json.dumps(params, sort_keys=True),
                            "fold": window.fold,
                            "status": f"skipped:{exc}",
                        }
                    )
                    break
    results = pd.DataFrame(rows)
    ensure_dir(outputs_dir)
    results.to_csv(outputs_dir / "validation_results.csv", index=False)
    ok = results[results["status"].eq("ok")].copy()
    if ok.empty:
        raise RuntimeError("No model completed validation")
    summary = (
        ok.groupby(["candidate_id", "model", "params_json"], as_index=False)
        .agg(
            mean_wape=("wape", "mean"),
            std_wape=("wape", "std"),
            mean_bias=("bias", "mean"),
            mean_mae=("mae", "mean"),
            mean_rmsle=("rmsle", "mean"),
            mean_promo_wape=("promo_wape", "mean"),
            folds=("fold", "nunique"),
        )
    )
    summary.to_csv(outputs_dir / "validation_summary.csv", index=False)
    guardrail = float(model_cfg.get("bias_guardrail_abs", 0.10))
    eligible = summary[summary["mean_bias"].abs().le(guardrail)]
    pool = eligible if not eligible.empty else summary
    winner = pool.sort_values(["mean_wape", "std_wape", "mean_promo_wape"], na_position="last").iloc[0]
    champion = {
        "model": winner["model"],
        "params": json.loads(winner["params_json"]),
        "candidate_id": winner["candidate_id"],
        "selection_metric": model_cfg.get("primary_metric", "wape"),
        "validation_mean_wape": winner["mean_wape"],
        "validation_mean_bias": winner["mean_bias"],
        "validation_mean_promo_wape": winner["mean_promo_wape"],
        "bias_guardrail_abs": guardrail,
        "holdout": {
            "train_end": str(holdout.train_end.date()),
            "forecast_start": str(holdout.forecast_start.date()),
            "forecast_end": str(holdout.forecast_end.date()),
        },
    }
    ensure_dir(artifacts_dir)
    write_json(champion, artifacts_dir / "champion_spec.json")
    write_json(
        [
            {
                "fold": w.fold,
                "train_end": str(w.train_end.date()),
                "forecast_start": str(w.forecast_start.date()),
                "forecast_end": str(w.forecast_end.date()),
            }
            for w in windows
        ],
        artifacts_dir / "validation_windows.json",
    )
    return results, champion


def evaluate_champion_holdout(
    panel: pd.DataFrame,
    champion: dict[str, Any],
    feature_cfg: dict[str, Any],
    forecast_cfg: dict[str, Any],
    seed: int,
    outputs_dir: Path,
    artifacts_dir: Path,
) -> tuple[pd.DataFrame, dict[str, float]]:
    holdout = champion["holdout"]
    window = TemporalWindow(
        "holdout",
        pd.Timestamp(holdout["train_end"]),
        pd.Timestamp(holdout["forecast_start"]),
        pd.Timestamp(holdout["forecast_end"]),
    )
    pred, estimator, spec = forecast_candidate(
        panel,
        window,
        champion["model"],
        champion.get("params", {}),
        feature_cfg,
        seed,
        forecast_cfg.get("target_transform", "log1p"),
    )
    scored = attach_actuals(panel, pred)
    metrics = metric_bundle(scored["actual"], scored["prediction"])
    log_experiment(
        run_name=f"{champion['model']}-holdout",
        params={"model": champion["model"], **champion.get("params", {})},
        metrics=metrics,
        tags={"stage": "holdout", "frozen_champion": True},
    )
    ensure_dir(outputs_dir)
    write_frame(scored, outputs_dir / "holdout_predictions.parquet", index=False)
    pd.DataFrame([metrics]).to_csv(outputs_dir / "holdout_metrics.csv", index=False)

    baseline = forecast_baseline(
        panel,
        window.train_end,
        list(pd.date_range(window.forecast_start, window.forecast_end, freq="D")),
        "seasonal_naive_7",
    )
    write_frame(attach_actuals(panel, baseline), outputs_dir / "holdout_baseline_predictions.parquet", index=False)

    if estimator is not None:
        joblib.dump(estimator, artifacts_dir / "holdout_champion_model.joblib")
        write_json(
            {
                "feature_columns": spec.feature_columns,
                "categorical_columns": spec.categorical_columns,
                "numeric_columns": spec.numeric_columns,
            },
            artifacts_dir / "feature_spec.json",
        )
    return scored, metrics


def train_final_and_predict_test(
    panel: pd.DataFrame,
    champion: dict[str, Any],
    feature_cfg: dict[str, Any],
    forecast_cfg: dict[str, Any],
    seed: int,
    outputs_dir: Path,
    artifacts_dir: Path,
) -> pd.DataFrame:
    historical = panel.loc[~panel["is_test"]]
    train_end = pd.to_datetime(historical["date"]).max()
    future_dates = sorted(pd.to_datetime(panel.loc[panel["is_test"], "date"].unique()))
    if champion["model"] in BASELINES:
        forecast = forecast_baseline(panel, train_end, future_dates, champion["model"])
    else:
        estimator, spec = _fit_model(
            panel,
            train_end,
            champion["model"],
            champion.get("params", {}),
            feature_cfg,
            seed,
            forecast_cfg.get("target_transform", "log1p"),
        )
        forecast = recursive_model_forecast(
            panel, train_end, future_dates, estimator, spec, feature_cfg, champion["model"]
        )
        joblib.dump(estimator, artifacts_dir / "final_champion_model.joblib")
        write_json(
            {
                "feature_columns": spec.feature_columns,
                "categorical_columns": spec.categorical_columns,
                "numeric_columns": spec.numeric_columns,
            },
            artifacts_dir / "feature_spec.json",
        )
    ensure_dir(outputs_dir)
    write_frame(forecast, outputs_dir / "test_forecasts.parquet", index=False)
    return forecast
