from __future__ import annotations

from common import config_argument

from favorita_forecasting.modeling.backtesting import run_validation_selection
from favorita_forecasting.storage import read_frame

if __name__ == "__main__":
    _, cfg = config_argument("Temporal backtesting and model selection")
    panel = read_frame(cfg.path("processed_dir") / "curated_panel.parquet")
    results, champion = run_validation_selection(
        panel,
        cfg.section("models"),
        cfg.section("features"),
        cfg.section("forecast"),
        int(cfg.section("project").get("random_seed", 42)),
        cfg.path("outputs_dir"),
        cfg.path("artifacts_dir"),
    )
    print(results[["model", "fold", "wape", "bias", "status"]].to_string(index=False))
    print("Champion:", champion)
