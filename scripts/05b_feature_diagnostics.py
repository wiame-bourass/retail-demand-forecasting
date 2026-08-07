from __future__ import annotations

import pandas as pd

from common import config_argument
from favorita_forecasting.storage import read_frame
from favorita_forecasting.features.build import FeatureSpec
from favorita_forecasting.features.diagnostics import feature_diagnostics
from favorita_forecasting.utils import read_json


if __name__ == "__main__":
    _, cfg = config_argument("Run feature diagnostics using training data only")
    features = read_frame(cfg.path("processed_dir") / "features.parquet")
    raw_spec = read_json(cfg.path("artifacts_dir") / "feature_spec_pre_model.json")
    spec = FeatureSpec(**raw_spec)
    tables = feature_diagnostics(
        features,
        spec,
        cfg.path("outputs_dir") / "feature_diagnostics",
        cfg.section("features"),
    )
    print({name: table.shape for name, table in tables.items()})
