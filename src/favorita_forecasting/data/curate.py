from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

try:
    import duckdb
except ImportError:  # demo fallback
    duckdb = None

import numpy as np
import pandas as pd

from favorita_forecasting.data.raw import raw_file_paths
from favorita_forecasting.storage import write_frame
from favorita_forecasting.utils import ensure_dir, read_json

LOGGER = logging.getLogger(__name__)


def _p(path: Path) -> str:
    return str(path).replace("'", "''")


def _bool(series: pd.Series | bool) -> pd.Series:
    if isinstance(series, bool):
        return pd.Series([series])
    return series.fillna(False).astype(bool)


def _holiday_features(holidays: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    holidays = holidays.copy()
    holidays["date"] = pd.to_datetime(holidays["date"])
    if "transferred" not in holidays:
        holidays["transferred"] = False
    holidays["transferred"] = holidays["transferred"].fillna(False).astype(bool)
    holidays["effective_event"] = (~holidays["transferred"]) | holidays["type"].isin(
        ["Transfer", "Bridge", "Additional"]
    )
    rows: list[pd.DataFrame] = []
    for locale, key in [("National", None), ("Regional", "state"), ("Local", "city")]:
        subset = holidays[holidays["locale"].eq(locale)].copy()
        if subset.empty:
            continue
        if key is None:
            expanded = (
                stores[["store_nbr"]]
                .assign(_k=1)
                .merge(subset.assign(_k=1), on="_k")
                .drop(columns="_k")
            )
        else:
            expanded = stores[["store_nbr", key]].merge(
                subset, left_on=key, right_on="locale_name", how="inner"
            )
        expanded["holiday_scope"] = locale.lower()
        rows.append(expanded)
    if not rows:
        return pd.DataFrame(columns=["date", "store_nbr"])
    events = pd.concat(rows, ignore_index=True)
    events["is_holiday"] = events["effective_event"] & ~events["type"].eq("Work Day")
    events["is_work_day_adjustment"] = events["type"].eq("Work Day")
    for scope in ["national", "regional", "local"]:
        events[f"is_{scope}_event"] = events["holiday_scope"].eq(scope) & events["is_holiday"]
    return (
        events.groupby(["date", "store_nbr"], as_index=False)
        .agg(
            is_holiday=("is_holiday", "max"),
            is_work_day_adjustment=("is_work_day_adjustment", "max"),
            is_national_event=("is_national_event", "max"),
            is_regional_event=("is_regional_event", "max"),
            is_local_event=("is_local_event", "max"),
            holiday_types=("type", lambda x: "|".join(sorted(set(map(str, x))))),
            holiday_descriptions=("description", lambda x: "|".join(sorted(set(map(str, x))))),
        )
    )


def _aggregate_with_pandas(
    files: dict[str, Path],
    stores_selected: list[int],
    families_selected: list[str],
    history_start: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if files["train"].stat().st_size > 250_000_000:
        raise RuntimeError("DuckDB is required for the full Favorita train.csv. Install requirements.txt")
    items = pd.read_csv(files["items"])
    train_raw = pd.read_csv(files["train"], parse_dates=["date"])
    train_raw = train_raw[
        train_raw["date"].ge(pd.Timestamp(history_start))
        & train_raw["store_nbr"].isin(stores_selected)
    ].merge(items, on="item_nbr", how="inner")
    train_raw = train_raw[train_raw["family"].isin(families_selected)].copy()
    train_raw["unit_sales"] = pd.to_numeric(train_raw["unit_sales"], errors="coerce").fillna(0.0)
    train_raw["positive_sales"] = train_raw["unit_sales"].clip(lower=0)
    train_raw["returns_units"] = (-train_raw["unit_sales"].clip(upper=0)).abs()
    train_raw["onpromotion"] = train_raw["onpromotion"].fillna(False).astype(bool)
    train = (
        train_raw.groupby(["date", "store_nbr", "family"], as_index=False)
        .agg(
            sales=("positive_sales", "sum"),
            returns_units=("returns_units", "sum"),
            n_items_observed=("item_nbr", "nunique"),
            n_items_on_promotion=("item_nbr", lambda s: s[train_raw.loc[s.index, "onpromotion"]].nunique()),
        )
    )
    train["is_test"] = False

    test_raw = pd.read_csv(files["test"], parse_dates=["date"])
    test_raw = test_raw[test_raw["store_nbr"].isin(stores_selected)].merge(items, on="item_nbr", how="inner")
    test_raw = test_raw[test_raw["family"].isin(families_selected)].copy()
    test_raw["onpromotion"] = test_raw["onpromotion"].fillna(False).astype(bool)
    future = (
        test_raw.groupby(["date", "store_nbr", "family"], as_index=False)
        .agg(
            n_items_observed=("item_nbr", "nunique"),
            n_items_on_promotion=("item_nbr", lambda s: s[test_raw.loc[s.index, "onpromotion"]].nunique()),
        )
    )
    future["sales"] = np.nan
    future["returns_units"] = np.nan
    future["is_test"] = True

    item_agg = (
        items[items["family"].isin(families_selected)]
        .groupby("family", as_index=False)
        .agg(
            catalog_items_family=("item_nbr", "nunique"),
            n_classes_family=("class", "nunique"),
            perishable_share=("perishable", "mean"),
        )
    )
    stores = pd.read_csv(files["stores"])
    stores = stores[stores["store_nbr"].isin(stores_selected)].rename(columns={"type": "store_type"})
    return train, future, item_agg, stores


def _aggregate_with_duckdb(
    files: dict[str, Path],
    stores_selected: list[int],
    families_selected: list[str],
    history_start: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect(database=":memory:")
    stores_sql = ",".join(map(str, stores_selected))
    families_sql = ",".join("'" + x.replace("'", "''") + "'" for x in families_selected)
    train = con.execute(
        f"""
        WITH item_meta AS (
            SELECT CAST(item_nbr AS BIGINT) item_nbr, family
            FROM read_csv_auto('{_p(files['items'])}', header=true)
        )
        SELECT TRY_CAST(t.date AS DATE) date,
               CAST(t.store_nbr AS INTEGER) store_nbr,
               i.family,
               SUM(GREATEST(TRY_CAST(t.unit_sales AS DOUBLE), 0)) sales,
               SUM(ABS(LEAST(TRY_CAST(t.unit_sales AS DOUBLE), 0))) returns_units,
               COUNT(DISTINCT CAST(t.item_nbr AS BIGINT)) n_items_observed,
               COUNT(DISTINCT CASE WHEN COALESCE(TRY_CAST(t.onpromotion AS BOOLEAN), FALSE) THEN CAST(t.item_nbr AS BIGINT) END) n_items_on_promotion
        FROM read_csv_auto('{_p(files['train'])}', header=true, sample_size=100000) t
        JOIN item_meta i ON CAST(t.item_nbr AS BIGINT) = i.item_nbr
        WHERE TRY_CAST(t.date AS DATE) >= DATE '{history_start}'
          AND CAST(t.store_nbr AS INTEGER) IN ({stores_sql})
          AND i.family IN ({families_sql})
        GROUP BY 1,2,3
        """
    ).df()
    train["date"] = pd.to_datetime(train["date"])
    train["is_test"] = False

    future = con.execute(
        f"""
        WITH item_meta AS (
            SELECT CAST(item_nbr AS BIGINT) item_nbr, family
            FROM read_csv_auto('{_p(files['items'])}', header=true)
        )
        SELECT TRY_CAST(t.date AS DATE) date,
               CAST(t.store_nbr AS INTEGER) store_nbr,
               i.family,
               COUNT(DISTINCT CAST(t.item_nbr AS BIGINT)) n_items_observed,
               COUNT(DISTINCT CASE WHEN COALESCE(TRY_CAST(t.onpromotion AS BOOLEAN), FALSE) THEN CAST(t.item_nbr AS BIGINT) END) n_items_on_promotion
        FROM read_csv_auto('{_p(files['test'])}', header=true, sample_size=100000) t
        JOIN item_meta i ON CAST(t.item_nbr AS BIGINT) = i.item_nbr
        WHERE CAST(t.store_nbr AS INTEGER) IN ({stores_sql})
          AND i.family IN ({families_sql})
        GROUP BY 1,2,3
        """
    ).df()
    future["date"] = pd.to_datetime(future["date"])
    future["sales"] = np.nan
    future["returns_units"] = np.nan
    future["is_test"] = True

    item_agg = con.execute(
        f"""
        SELECT family,
               COUNT(DISTINCT CAST(item_nbr AS BIGINT)) catalog_items_family,
               COUNT(DISTINCT CAST(class AS INTEGER)) n_classes_family,
               AVG(CAST(perishable AS DOUBLE)) perishable_share
        FROM read_csv_auto('{_p(files['items'])}', header=true)
        WHERE family IN ({families_sql})
        GROUP BY 1
        """
    ).df()
    stores = con.execute(
        f"SELECT CAST(store_nbr AS INTEGER) AS store_nbr, city, state, \"type\" AS store_type, CAST(\"cluster\" AS INTEGER) AS \"cluster\" FROM read_csv_auto('{_p(files['stores'])}', header=true) WHERE CAST(store_nbr AS INTEGER) IN ({stores_sql})"
    ).df()
    return train, future, item_agg, stores


def build_curated_panel(
    raw_dir: Path,
    processed_dir: Path,
    artifacts_dir: Path,
    cleaning_cfg: dict[str, Any],
) -> pd.DataFrame:
    del cleaning_cfg  # policies are encoded explicitly below and documented
    files = raw_file_paths(raw_dir)
    manifest = read_json(artifacts_dir / "scope_manifest.json")
    stores_selected = [int(x) for x in manifest["stores"]]
    families_selected = [str(x) for x in manifest["families"]]
    history_start = manifest["history_start"]

    if duckdb is None:
        train, future, item_agg, stores = _aggregate_with_pandas(
            files, stores_selected, families_selected, history_start
        )
    else:
        train, future, item_agg, stores = _aggregate_with_duckdb(
            files, stores_selected, families_selected, history_start
        )

    all_dates_end = max(train["date"].max(), future["date"].max())
    series_start = (
        train.groupby(["store_nbr", "family"], as_index=False)["date"]
        .min()
        .rename(columns={"date": "series_start_date"})
    )
    grid_parts: list[pd.DataFrame] = []
    for row in series_start.itertuples(index=False):
        dates = pd.date_range(row.series_start_date, all_dates_end, freq="D")
        grid_parts.append(
            pd.DataFrame({"date": dates, "store_nbr": row.store_nbr, "family": row.family})
        )
    if not grid_parts:
        raise RuntimeError("No active store-family series remained after scope filtering")
    grid = pd.concat(grid_parts, ignore_index=True)

    panel = grid.merge(
        pd.concat([train, future], ignore_index=True),
        on=["date", "store_nbr", "family"],
        how="left",
    )
    test_start = future["date"].min()
    panel["is_test"] = panel["date"].ge(test_start)
    historical_missing = (~panel["is_test"]) & panel["sales"].isna()
    panel["was_missing_historical_row"] = historical_missing
    panel.loc[historical_missing, "sales"] = 0.0
    panel.loc[historical_missing, "returns_units"] = 0.0
    panel["n_items_observed"] = panel["n_items_observed"].fillna(0).astype("int32")
    panel["n_items_on_promotion"] = panel["n_items_on_promotion"].fillna(0).astype("int32")

    panel = panel.merge(item_agg, on="family", how="left").merge(
        stores, on="store_nbr", how="left"
    )
    panel["promotion_share_catalog"] = (
        panel["n_items_on_promotion"] / panel["catalog_items_family"].replace(0, np.nan)
    ).fillna(0.0)
    panel["any_promotion"] = panel["n_items_on_promotion"].gt(0)

    oil = pd.read_csv(files["oil"], parse_dates=["date"]).sort_values("date")
    oil["dcoilwtico"] = pd.to_numeric(oil["dcoilwtico"], errors="coerce").interpolate(
        limit_direction="both"
    )
    panel = panel.merge(oil[["date", "dcoilwtico"]], on="date", how="left")

    holidays = pd.read_csv(files["holidays"])
    holiday_features = _holiday_features(holidays, stores)
    panel = panel.merge(holiday_features, on=["date", "store_nbr"], how="left")
    for col in [
        "is_holiday",
        "is_work_day_adjustment",
        "is_national_event",
        "is_regional_event",
        "is_local_event",
    ]:
        panel[col] = panel[col].fillna(False).astype(bool)
    panel["holiday_types"] = panel["holiday_types"].fillna("None")
    panel["holiday_descriptions"] = panel["holiday_descriptions"].fillna("None")

    transactions = pd.read_csv(files["transactions"], parse_dates=["date"])
    transactions["transactions"] = pd.to_numeric(transactions["transactions"], errors="coerce")
    panel = panel.merge(transactions, on=["date", "store_nbr"], how="left")

    panel = panel.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    ensure_dir(processed_dir)
    write_frame(panel, processed_dir / "curated_panel.parquet", index=False)

    qa = pd.DataFrame(
        [
            {"check": "rows", "value": len(panel)},
            {
                "check": "series",
                "value": panel[["store_nbr", "family"]].drop_duplicates().shape[0],
            },
            {"check": "date_min", "value": panel["date"].min()},
            {"check": "date_max", "value": panel["date"].max()},
            {
                "check": "historical_imputed_zero_rows",
                "value": int(panel["was_missing_historical_row"].sum()),
            },
            {"check": "test_rows", "value": int(panel["is_test"].sum())},
        ]
    )
    qa.to_csv(artifacts_dir / "curated_panel_qa.csv", index=False)
    return panel
