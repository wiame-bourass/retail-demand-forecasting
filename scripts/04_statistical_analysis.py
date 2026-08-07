from __future__ import annotations

from common import config_argument

from favorita_forecasting.analysis.plots import save_core_plots
from favorita_forecasting.analysis.statistical import run_statistical_analysis
from favorita_forecasting.storage import read_frame

if __name__ == "__main__":
    _, cfg = config_argument("Run statistical analysis")
    panel = read_frame(cfg.path("processed_dir") / "curated_panel.parquet")
    tables = run_statistical_analysis(panel, cfg.path("outputs_dir") / "statistical_analysis")
    save_core_plots(panel, cfg.path("reports_dir") / "figures")
    print({name: table.shape for name, table in tables.items()})
