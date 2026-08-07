from __future__ import annotations

import numpy as np
import pandas as pd


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(float)
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy(float)
    if len(ref) < 20 or len(cur) < 5 or np.nanstd(ref) == 0:
        return np.nan
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return np.nan
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_snapshot(
    features: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    reference_start: pd.Timestamp,
    reference_end: pd.Timestamp,
    current_start: pd.Timestamp,
    current_end: pd.Timestamp,
) -> pd.DataFrame:
    reference = features[features["date"].between(reference_start, reference_end)]
    current = features[features["date"].between(current_start, current_end)]
    rows = []
    for column in feature_columns:
        if column not in features:
            continue
        row = {
            "feature": column,
            "reference_missing_rate": reference[column].isna().mean(),
            "current_missing_rate": current[column].isna().mean(),
        }
        if column in categorical_columns:
            ref_categories = set(reference[column].dropna().astype(str).unique())
            current_values = current[column].dropna().astype(str)
            row.update(
                {
                    "feature_type": "categorical",
                    "psi": np.nan,
                    "new_category_rate": (
                        ~current_values.isin(ref_categories)
                    ).mean()
                    if len(current_values)
                    else np.nan,
                }
            )
        else:
            row.update(
                {
                    "feature_type": "numeric",
                    "psi": population_stability_index(reference[column], current[column]),
                    "new_category_rate": np.nan,
                }
            )
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result["alert"] = (
            result["psi"].fillna(0).gt(0.25)
            | result["new_category_rate"].fillna(0).gt(0.05)
            | (result["current_missing_rate"] - result["reference_missing_rate"]).abs().gt(0.10)
        )
    return result
