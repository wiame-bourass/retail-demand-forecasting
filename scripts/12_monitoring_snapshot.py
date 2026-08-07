from __future__ import annotations

import pandas as pd

from common import config_argument
from favorita_forecasting.monitoring.drift import drift_snapshot
from favorita_forecasting.storage import read_frame
from favorita_forecasting.utils import read_json


if __name__ == "__main__":
    _, cfg = config_argument("Generate feature drift and data-quality snapshot")
    features = read_frame(cfg.path("processed_dir") / "features.parquet")
    spec = read_json(cfg.path("artifacts_dir") / "feature_spec_pre_model.json")
    champion = read_json(cfg.path("artifacts_dir") / "champion_spec.json")
    current_start = pd.Timestamp(champion["holdout"]["forecast_start"])
    current_end = pd.Timestamp(champion["holdout"]["forecast_end"])
    reference_end = current_start - pd.Timedelta(days=1)
    reference_start = reference_end - pd.Timedelta(days=89)
    snapshot = drift_snapshot(
        features,
        spec["feature_columns"],
        spec["categorical_columns"],
        reference_start,
        reference_end,
        current_start,
        current_end,
    )
    snapshot.to_csv(cfg.path("outputs_dir") / "monitoring_snapshot.csv", index=False)
    print(snapshot.sort_values(["alert", "psi"], ascending=[False, False]).head(25).to_string(index=False))
