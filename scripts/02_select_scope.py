from __future__ import annotations

from common import config_argument
from favorita_forecasting.data.scope import select_scope


if __name__ == "__main__":
    _, cfg = config_argument("Select representative stores and families")
    manifest = select_scope(cfg.path("raw_dir"), cfg.path("artifacts_dir"), cfg.section("scope"))
    print(manifest)
