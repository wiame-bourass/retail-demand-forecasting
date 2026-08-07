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
from favorita_forecasting.utils import ensure_dir, write_json

LOGGER = logging.getLogger(__name__)


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _diverse_store_selection(stats: pd.DataFrame, stores: pd.DataFrame, n: int) -> list[int]:
    merged = stats.merge(stores, on="store_nbr", how="left").sort_values("total_sales", ascending=False)
    selected: list[int] = []
    for _, group in merged.groupby("type", dropna=False):
        if len(selected) >= n:
            break
        selected.append(int(group.iloc[0]["store_nbr"]))
    for store in merged["store_nbr"].astype(int):
        if store not in selected:
            selected.append(store)
        if len(selected) >= n:
            break
    return selected[:n]


def _family_selection(stats: pd.DataFrame, n: int) -> list[str]:
    if len(stats) <= n:
        return stats["family"].astype(str).tolist()
    selected: list[str] = []
    quotas = {
        "total_sales": max(1, n // 3),
        "promo_rate": max(1, n // 4),
        "intermittency_proxy": max(1, n // 4),
        "perishable_share": max(1, n // 6),
    }
    directions = {
        "total_sales": False,
        "promo_rate": False,
        "intermittency_proxy": False,
        "perishable_share": False,
    }
    for column, quota in quotas.items():
        ranked = stats.sort_values(column, ascending=directions[column])["family"].astype(str)
        for family in ranked:
            if family not in selected:
                selected.append(family)
            if sum(x in selected for x in ranked.head(quota)) >= quota:
                break
    for family in stats.sort_values("total_sales", ascending=False)["family"].astype(str):
        if family not in selected:
            selected.append(family)
        if len(selected) >= n:
            break
    return selected[:n]


def select_scope(raw_dir: Path, artifacts_dir: Path, scope_cfg: dict[str, Any]) -> dict[str, Any]:
    files = raw_file_paths(raw_dir)
    if duckdb is None:
        if files["train"].stat().st_size > 250_000_000:
            raise RuntimeError("DuckDB is required for the full Favorita train.csv. Install requirements-core.txt")
        train_df = pd.read_csv(files["train"], parse_dates=["date"])
        items_df = pd.read_csv(files["items"])
        stores = pd.read_csv(files["stores"])
        stores["store_nbr"] = stores["store_nbr"].astype(int)
        max_date = train_df["date"].max()
        selection_days = int(scope_cfg.get("selection_window_days", 365))
        start_date = pd.Timestamp(max_date) - pd.Timedelta(days=selection_days - 1)
        recent = train_df[train_df["date"].ge(start_date)].copy()
        recent["positive_sales"] = pd.to_numeric(recent["unit_sales"], errors="coerce").clip(lower=0)
        store_stats = recent.groupby("store_nbr", as_index=False).agg(total_sales=("positive_sales", "sum"), observed_rows=("positive_sales", "size"))
        explicit_stores = [int(value) for value in scope_cfg.get("explicit_stores", [])]
        selected_stores = explicit_stores or _diverse_store_selection(store_stats, stores, int(scope_cfg.get("n_stores", 8)))
        joined = recent[recent["store_nbr"].isin(selected_stores)].merge(items_df, on="item_nbr", how="inner")
        joined["onpromotion"] = joined["onpromotion"].fillna(False).astype(bool)
        daily = joined.groupby(["date", "family"], as_index=False).agg(sales=("positive_sales", "sum"), any_promo=("onpromotion", "max"))
        family_base = daily.groupby("family", as_index=False).agg(total_sales=("sales", "sum"), promo_rate=("any_promo", "mean"), first_date=("date", "min"), last_date=("date", "max"), active_days=("date", "nunique"))
        family_base["intermittency_proxy"] = 1 - family_base["active_days"] / ((family_base["last_date"] - family_base["first_date"]).dt.days + 1).clip(lower=1)
        perishable = items_df.groupby("family", as_index=False)["perishable"].mean().rename(columns={"perishable": "perishable_share"})
        family_stats = family_base.merge(perishable, on="family", how="left")
        explicit_families = [str(value) for value in scope_cfg.get("explicit_families", [])]
        selected_families = explicit_families or _family_selection(family_stats, int(scope_cfg.get("n_families", 12)))
        history_years = int(scope_cfg.get("history_years", 2))
        history_start = pd.Timestamp(max_date) - pd.DateOffset(years=history_years) + pd.Timedelta(days=1)
        manifest = {
            "selection_date_max": str(pd.Timestamp(max_date).date()),
            "selection_window_start": str(start_date.date()),
            "history_start": str(history_start.date()),
            "stores": selected_stores,
            "families": selected_families,
            "selection_method": {"stores": "pandas demo fallback", "families": "pandas demo fallback"},
        }
        ensure_dir(artifacts_dir)
        write_json(manifest, artifacts_dir / "scope_manifest.json")
        store_stats.merge(stores, on="store_nbr", how="left").to_csv(artifacts_dir / "store_selection_candidates.csv", index=False)
        family_stats.to_csv(artifacts_dir / "family_selection_candidates.csv", index=False)
        return manifest

    con = duckdb.connect(database=":memory:")
    train = _sql_path(files["train"])
    items = _sql_path(files["items"])
    stores_path = _sql_path(files["stores"])

    max_date = con.execute(
        f"SELECT MAX(TRY_CAST(date AS DATE)) FROM read_csv_auto('{train}', header=true, sample_size=100000)"
    ).fetchone()[0]
    selection_days = int(scope_cfg.get("selection_window_days", 365))
    start_date = pd.Timestamp(max_date) - pd.Timedelta(days=selection_days - 1)

    store_stats = con.execute(
        f"""
        SELECT CAST(store_nbr AS INTEGER) AS store_nbr,
               SUM(GREATEST(TRY_CAST(unit_sales AS DOUBLE), 0)) AS total_sales,
               COUNT(*) AS observed_rows
        FROM read_csv_auto('{train}', header=true, sample_size=100000)
        WHERE TRY_CAST(date AS DATE) >= DATE '{start_date.date()}'
        GROUP BY 1
        """
    ).df()
    stores = con.execute(
        f"SELECT CAST(store_nbr AS INTEGER) AS store_nbr, city, state, \"type\" AS \"type\", CAST(\"cluster\" AS INTEGER) AS \"cluster\" FROM read_csv_auto('{stores_path}', header=true)"
    ).df()

    explicit_stores = [int(value) for value in scope_cfg.get("explicit_stores", [])]
    selected_stores = explicit_stores or _diverse_store_selection(
        store_stats, stores, int(scope_cfg.get("n_stores", 8))
    )
    store_list_sql = ",".join(str(x) for x in selected_stores)

    family_stats = con.execute(
        f"""
        WITH joined AS (
            SELECT TRY_CAST(t.date AS DATE) date,
                   CAST(t.store_nbr AS INTEGER) store_nbr,
                   CAST(t.item_nbr AS BIGINT) item_nbr,
                   GREATEST(TRY_CAST(t.unit_sales AS DOUBLE), 0) sales,
                   COALESCE(TRY_CAST(t.onpromotion AS BOOLEAN), FALSE) onpromotion,
                   i.family,
                   CAST(i.perishable AS INTEGER) perishable
            FROM read_csv_auto('{train}', header=true, sample_size=100000) t
            JOIN read_csv_auto('{items}', header=true) i USING(item_nbr)
            WHERE TRY_CAST(t.date AS DATE) >= DATE '{start_date.date()}'
              AND CAST(t.store_nbr AS INTEGER) IN ({store_list_sql})
        ), daily AS (
            SELECT date, family,
                   SUM(sales) sales,
                   MAX(CAST(onpromotion AS INTEGER)) any_promo
            FROM joined GROUP BY 1,2
        ), item_meta AS (
            SELECT family, AVG(CAST(perishable AS DOUBLE)) perishable_share
            FROM read_csv_auto('{items}', header=true) GROUP BY 1
        )
        SELECT d.family,
               SUM(d.sales) total_sales,
               AVG(d.any_promo) promo_rate,
               1.0 - COUNT(DISTINCT d.date) / NULLIF(DATE_DIFF('day', MIN(d.date), MAX(d.date)) + 1, 0) intermittency_proxy,
               m.perishable_share
        FROM daily d JOIN item_meta m USING(family)
        GROUP BY d.family, m.perishable_share
        """
    ).df()

    explicit_families = [str(value) for value in scope_cfg.get("explicit_families", [])]
    selected_families = explicit_families or _family_selection(
        family_stats, int(scope_cfg.get("n_families", 12))
    )

    history_years = int(scope_cfg.get("history_years", 2))
    history_start = pd.Timestamp(max_date) - pd.DateOffset(years=history_years) + pd.Timedelta(days=1)

    manifest = {
        "selection_date_max": str(max_date),
        "selection_window_start": str(start_date.date()),
        "history_start": str(history_start.date()),
        "stores": selected_stores,
        "families": selected_families,
        "selection_method": {
            "stores": "volume ranking with store-type diversity unless explicit list is configured",
            "families": "union of high-volume, promotion-heavy, intermittent and perishable families",
        },
    }
    ensure_dir(artifacts_dir)
    write_json(manifest, artifacts_dir / "scope_manifest.json")
    store_stats.merge(stores, on="store_nbr", how="left").to_csv(
        artifacts_dir / "store_selection_candidates.csv", index=False
    )
    family_stats.to_csv(artifacts_dir / "family_selection_candidates.csv", index=False)
    return manifest
