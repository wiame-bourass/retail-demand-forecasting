from __future__ import annotations

from common import config_argument

from favorita_forecasting.modeling.backtesting import train_final_and_predict_test
from favorita_forecasting.storage import read_frame
from favorita_forecasting.utils import read_json

if __name__ == "__main__":
    _, cfg = config_argument("Train frozen champion on all history and forecast Kaggle test")
    panel = read_frame(cfg.path("processed_dir") / "curated_panel.parquet")
    champion = read_json(cfg.path("artifacts_dir") / "champion_spec.json")
    forecast = train_final_and_predict_test(
        panel,
        champion,
        cfg.section("features"),
        cfg.section("forecast"),
        int(cfg.section("project").get("random_seed", 42)),
        cfg.path("outputs_dir"),
        cfg.path("artifacts_dir"),
    )
    print(forecast.head())
