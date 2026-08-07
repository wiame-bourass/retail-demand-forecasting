from __future__ import annotations

import pandas as pd

from common import config_argument
from favorita_forecasting.storage import read_frame, write_frame
from favorita_forecasting.features.build import assert_target_features_are_causal, build_features
from favorita_forecasting.utils import write_json


if __name__ == "__main__":
    _, cfg = config_argument("Build causal feature dataset")
    panel = read_frame(cfg.path("processed_dir") / "curated_panel.parquet")
    features, spec = build_features(panel, cfg.section("features"))
    assert_target_features_are_causal(panel.loc[~panel["is_test"]].reset_index(drop=True), features)
    write_frame(features, cfg.path("processed_dir") / "features.parquet", index=False)
    write_json(
        {
            "feature_columns": spec.feature_columns,
            "categorical_columns": spec.categorical_columns,
            "numeric_columns": spec.numeric_columns,
        },
        cfg.path("artifacts_dir") / "feature_spec_pre_model.json",
    )
    print(f"Features: {len(spec.feature_columns)} | rows: {len(features)}")
