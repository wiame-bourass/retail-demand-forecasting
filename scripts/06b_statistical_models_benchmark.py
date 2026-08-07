from __future__ import annotations

import pandas as pd
from common import config_argument

from favorita_forecasting.modeling.backtesting import temporal_windows
from favorita_forecasting.modeling.statistical import benchmark_statistical_models
from favorita_forecasting.storage import read_frame

if __name__ == "__main__":
    _, cfg = config_argument("Benchmark ETS, SARIMA, Croston and TSB on representative series")
    panel = read_frame(cfg.path("processed_dir") / "curated_panel.parquet")
    profiles = pd.read_csv(cfg.path("outputs_dir") / "statistical_analysis" / "series_profiles.csv")
    selected = (
        profiles.sort_values("mean_sales", ascending=False)
        .groupby("demand_class", as_index=False)
        .head(1)
        .head(4)
    )
    historical = panel[~panel["is_test"]]
    validation, _ = temporal_windows(
        historical["date"],
        int(cfg.section("forecast")["horizon_days"]),
        int(cfg.section("forecast")["validation_folds"]),
        int(cfg.section("forecast")["holdout_days"]),
    )
    window = validation[-1]
    results = benchmark_statistical_models(
        panel,
        window.forecast_start,
        int(cfg.section("forecast")["horizon_days"]),
        selected,
    )
    results.to_csv(cfg.path("outputs_dir") / "statistical_model_benchmark.csv", index=False)
    print(results.sort_values(["store_nbr", "family", "wape"]).to_string(index=False))
