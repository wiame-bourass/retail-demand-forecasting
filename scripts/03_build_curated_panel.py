from __future__ import annotations

from common import config_argument

from favorita_forecasting.data.curate import build_curated_panel

if __name__ == "__main__":
    _, cfg = config_argument("Build curated daily store-family panel")
    panel = build_curated_panel(
        cfg.path("raw_dir"),
        cfg.path("processed_dir"),
        cfg.path("artifacts_dir"),
        cfg.section("cleaning"),
    )
    print(panel.shape)
    print(panel.head())
