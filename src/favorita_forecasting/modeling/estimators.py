from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer


def _base_model(name: str, params: dict[str, Any], seed: int):
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(random_state=seed, **params)
    if name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("LightGBM n'est pas installé. Installe requirements-models.txt") from exc
        return LGBMRegressor(random_state=seed, n_jobs=-1, verbosity=-1, **params)
    if name == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("CatBoost n'est pas installé. Installe requirements-models.txt") from exc
        return CatBoostRegressor(random_seed=seed, verbose=False, allow_writing_files=False, **params)
    raise ValueError(f"Unknown model: {name}")


def build_estimator(
    name: str,
    params: dict[str, Any],
    numeric_columns: list[str],
    categorical_columns: list[str],
    seed: int,
    target_transform: str = "log1p",
):
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True))])
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1),
            ),
        ]
    )
    transformer = ColumnTransformer(
        [
            ("numeric", numeric_pipe, numeric_columns),
            ("categorical", categorical_pipe, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    pipeline = Pipeline([("preprocess", transformer), ("model", _base_model(name, params, seed))])
    if target_transform == "log1p":
        return TransformedTargetRegressor(
            regressor=pipeline,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )
    return pipeline
