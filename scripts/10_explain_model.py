from __future__ import annotations

import joblib
import pandas as pd
from common import config_argument
from sklearn.inspection import permutation_importance

from favorita_forecasting.storage import read_frame
from favorita_forecasting.utils import read_json

if __name__ == "__main__":
    _, cfg = config_argument("Compute train-only permutation importance for the frozen model")
    champion = read_json(cfg.path("artifacts_dir") / "champion_spec.json")
    if champion["model"] in {"seasonal_naive_7", "four_week_mean", "dow_mean"}:
        print("Baseline champion: no fitted estimator to explain")
        raise SystemExit(0)
    model = joblib.load(cfg.path("artifacts_dir") / "holdout_champion_model.joblib")
    spec = read_json(cfg.path("artifacts_dir") / "feature_spec.json")
    features = read_frame(cfg.path("processed_dir") / "features.parquet")
    cutoff = pd.Timestamp(champion["holdout"]["train_end"])
    train = features[features["date"].le(cutoff) & features["sales"].notna()].copy()
    sample = train.tail(min(5000, len(train)))
    result = permutation_importance(
        model,
        sample[spec["feature_columns"]],
        sample["sales"],
        scoring="neg_mean_absolute_error",
        n_repeats=3,
        random_state=42,
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": spec["feature_columns"],
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance.to_csv(cfg.path("outputs_dir") / "permutation_importance_train_only.csv", index=False)
    print(importance.head(20).to_string(index=False))
