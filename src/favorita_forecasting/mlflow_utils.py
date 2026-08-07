from __future__ import annotations

import logging
import math
from typing import Any

LOGGER = logging.getLogger(__name__)


def log_experiment(
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    tags: dict[str, Any] | None = None,
    experiment_name: str = "favorita-retail-forecasting",
) -> None:
    """Log to MLflow when installed; never make local execution depend on it."""
    try:
        import mlflow
    except ImportError:
        LOGGER.debug("MLflow is not installed; skipping tracking")
        return
    try:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({key: str(value) for key, value in params.items()})
            clean_metrics = {
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float)) and not math.isnan(float(value))
            }
            mlflow.log_metrics(clean_metrics)
            if tags:
                mlflow.set_tags({key: str(value) for key, value in tags.items()})
    except Exception as exc:  # noqa: BLE001  # pragma: no cover
        LOGGER.warning("MLflow logging failed: %s", exc)
