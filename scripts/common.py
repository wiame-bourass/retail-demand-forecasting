from __future__ import annotations

import argparse

from favorita_forecasting.config import ProjectConfig, load_config
from favorita_forecasting.utils import configure_logging, ensure_dir, set_seed


def config_argument(description: str) -> tuple[argparse.Namespace, ProjectConfig]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="config/project.yaml")
    args = parser.parse_args()
    configure_logging()
    cfg = load_config(args.config)
    set_seed(int(cfg.section("project").get("random_seed", 42)))
    for key in ["interim_dir", "processed_dir", "artifacts_dir", "outputs_dir", "reports_dir"]:
        ensure_dir(cfg.path(key))
    return args, cfg
