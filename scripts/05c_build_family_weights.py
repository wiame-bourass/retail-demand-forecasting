from __future__ import annotations

import pandas as pd

from common import config_argument
from favorita_forecasting.data.family_weights import build_family_weights
from favorita_forecasting.modeling.backtesting import temporal_windows
from favorita_forecasting.storage import read_frame


if __name__ == "__main__":
    _, cfg = config_argument("Build train-only family weights for adapted weighted RMSLE")
    panel = read_frame(cfg.path("processed_dir") / "curated_panel.parquet")
    historical = panel[~panel["is_test"]]
    _, holdout = temporal_windows(
        historical["date"],
        int(cfg.section("forecast")["horizon_days"]),
        int(cfg.section("forecast")["validation_folds"]),
        int(cfg.section("forecast")["holdout_days"]),
    )
    weights = build_family_weights(cfg.path("raw_dir"), cfg.path("artifacts_dir"), holdout.train_end)
    print(weights.to_string(index=False))
