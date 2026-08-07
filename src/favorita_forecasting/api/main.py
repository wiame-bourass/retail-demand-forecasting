from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from favorita_forecasting.storage import frame_exists, read_frame
from fastapi import FastAPI, HTTPException, Query

ROOT = Path(__file__).resolve().parents[3]
FORECAST_PATH = Path(os.getenv("FORECAST_OUTPUT", ROOT / "outputs" / "test_forecasts.parquet"))
MODEL_CARD_PATH = Path(os.getenv("MODEL_CARD", ROOT / "artifacts" / "model_card.json"))

app = FastAPI(
    title="Retail Demand Forecast API",
    version="0.1.0",
    description="Consultation API for precomputed 16-day store-family forecasts.",
)


def _forecasts() -> pd.DataFrame:
    if not FORECAST_PATH.exists():
        raise HTTPException(status_code=503, detail=f"Forecast artifact not found: {FORECAST_PATH}")
    frame = read_frame(FORECAST_PATH)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "forecast_available": frame_exists(FORECAST_PATH),
        "model_card_available": MODEL_CARD_PATH.exists(),
    }


@app.get("/model-card")
def model_card() -> dict[str, Any]:
    if not MODEL_CARD_PATH.exists():
        raise HTTPException(status_code=404, detail="Model card not found")
    return json.loads(MODEL_CARD_PATH.read_text(encoding="utf-8"))


@app.get("/forecast")
def forecast(
    store_nbr: int | None = Query(default=None),
    family: str | None = Query(default=None),
    start_date: str | None = Query(default=None),
    end_date: str | None = Query(default=None),
) -> list[dict[str, Any]]:
    frame = _forecasts()
    if store_nbr is not None:
        frame = frame[frame["store_nbr"].eq(store_nbr)]
    if family is not None:
        frame = frame[frame["family"].eq(family)]
    if start_date:
        frame = frame[frame["date"].ge(pd.Timestamp(start_date))]
    if end_date:
        frame = frame[frame["date"].le(pd.Timestamp(end_date))]
    frame = frame.sort_values(["date", "store_nbr", "family"])
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    return frame.to_dict(orient="records")


@app.get("/forecast/aggregate")
def aggregate_forecast(level: str = Query(default="store", pattern="^(total|store|family)$")) -> list[dict[str, Any]]:
    frame = _forecasts()
    keys = {"total": ["date"], "store": ["date", "store_nbr"], "family": ["date", "family"]}[level]
    result = frame.groupby(keys, as_index=False)["prediction"].sum()
    result["date"] = pd.to_datetime(result["date"]).dt.strftime("%Y-%m-%d")
    return result.to_dict(orient="records")
