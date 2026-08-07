from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import OrdinalEncoder

from favorita_forecasting.features.build import FeatureSpec
from favorita_forecasting.utils import ensure_dir


def feature_diagnostics(
    features: pd.DataFrame,
    spec: FeatureSpec,
    output_dir: Path,
    cfg: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    ensure_dir(output_dir)
    train = features.loc[(~features["is_test"]) & features["sales"].notna()].copy()
    missing = pd.DataFrame(
        {
            "feature": spec.feature_columns,
            "missing_rate": [train[c].isna().mean() for c in spec.feature_columns],
            "n_unique": [train[c].nunique(dropna=False) for c in spec.feature_columns],
        }
    )
    missing["recommended_keep"] = (
        missing["missing_rate"].le(float(cfg.get("high_missingness_threshold", 0.95)))
        & missing["n_unique"].gt(1)
    )
    missing.to_csv(output_dir / "feature_missingness_and_constants.csv", index=False)

    numeric = [c for c in spec.numeric_columns if c in train and pd.api.types.is_numeric_dtype(train[c])]
    corr_pairs = []
    if numeric:
        corr = train[numeric].corr().abs()
        threshold = float(cfg.get("high_correlation_threshold", 0.995))
        for i, left in enumerate(corr.columns):
            for right in corr.columns[i + 1 :]:
                value = corr.loc[left, right]
                if pd.notna(value) and value >= threshold:
                    corr_pairs.append({"feature_a": left, "feature_b": right, "abs_correlation": value})
    correlation = pd.DataFrame(corr_pairs)
    correlation.to_csv(output_dir / "high_correlation_pairs.csv", index=False)

    sample = train.sample(min(50_000, len(train)), random_state=42) if len(train) else train
    mi_rows = []
    if len(sample) >= 20:
        candidates = [c for c in spec.feature_columns if c in sample]
        encoded = sample[candidates].copy()
        cat_cols = [c for c in spec.categorical_columns if c in candidates]
        if cat_cols:
            encoded[cat_cols] = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit_transform(
                encoded[cat_cols].astype(str).fillna("__MISSING__")
            )
        encoded = encoded.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        encoded = encoded.fillna(encoded.median(numeric_only=True)).fillna(0)
        values = mutual_info_regression(encoded, sample["sales"].astype(float), random_state=42)
        mi_rows = [{"feature": name, "mutual_information": value} for name, value in zip(candidates, values)]
    mutual_info = pd.DataFrame(mi_rows).sort_values("mutual_information", ascending=False) if mi_rows else pd.DataFrame(columns=["feature", "mutual_information"])
    mutual_info.to_csv(output_dir / "feature_mutual_information_train_only.csv", index=False)
    return {"missingness": missing, "correlation": correlation, "mutual_information": mutual_info}
