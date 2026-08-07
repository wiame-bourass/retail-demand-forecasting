from __future__ import annotations

import pandas as pd
from common import config_argument

from favorita_forecasting.evaluation.hierarchy import bottom_up_coherence, hierarchy_metrics
from favorita_forecasting.evaluation.metrics import segmented_metrics, weighted_rmsle
from favorita_forecasting.modeling.backtesting import evaluate_champion_holdout
from favorita_forecasting.storage import read_frame
from favorita_forecasting.utils import read_json, write_json

if __name__ == "__main__":
    _, cfg = config_argument("Evaluate frozen champion on final holdout")
    panel = read_frame(cfg.path("processed_dir") / "curated_panel.parquet")
    champion = read_json(cfg.path("artifacts_dir") / "champion_spec.json")
    scored, metrics = evaluate_champion_holdout(
        panel,
        champion,
        cfg.section("features"),
        cfg.section("forecast"),
        int(cfg.section("project").get("random_seed", 42)),
        cfg.path("outputs_dir"),
        cfg.path("artifacts_dir"),
    )
    segmented_metrics(scored, ["family"]).to_csv(cfg.path("outputs_dir") / "holdout_metrics_by_family.csv", index=False)
    segmented_metrics(scored, ["store_nbr"]).to_csv(cfg.path("outputs_dir") / "holdout_metrics_by_store.csv", index=False)
    segmented_metrics(scored, ["any_promotion"]).to_csv(cfg.path("outputs_dir") / "holdout_metrics_by_promotion.csv", index=False)
    segmented_metrics(scored, ["horizon_step"]).to_csv(cfg.path("outputs_dir") / "holdout_metrics_by_horizon.csv", index=False)
    hierarchy_metrics(scored).to_csv(cfg.path("outputs_dir") / "holdout_hierarchy_metrics.csv", index=False)
    weight_path = cfg.path("artifacts_dir") / "family_weights_train_only.csv"
    if weight_path.exists():
        weights = pd.read_csv(weight_path)
        family_daily = scored.groupby(["date", "family"], as_index=False).agg(actual=("actual", "sum"), prediction=("prediction", "sum"))
        family_daily = family_daily.merge(weights[["family", "family_weight"]], on="family", how="left")
        adapted = weighted_rmsle(family_daily["actual"], family_daily["prediction"], family_daily["family_weight"])
        pd.DataFrame([{"adapted_family_weighted_rmsle": adapted, "official_kaggle_metric": False}]).to_csv(
            cfg.path("outputs_dir") / "holdout_adapted_weighted_rmsle.csv", index=False
        )
    write_json(bottom_up_coherence(scored), cfg.path("artifacts_dir") / "bottom_up_coherence.json")
    write_json(
        {
            "project": cfg.section("project")["name"],
            "model": champion["model"],
            "params": champion.get("params", {}),
            "validation_mean_wape": champion.get("validation_mean_wape"),
            "holdout_metrics": metrics,
            "target": "observed sales used as proxy for demand",
            "horizon_days": cfg.section("forecast")["horizon_days"],
            "known_limitations": [
                "No stock or stockout data",
                "Promotion variable has no discount depth",
                "Inventory parameters are simulated",
            ],
        },
        cfg.path("artifacts_dir") / "model_card.json",
    )
    print(metrics)
