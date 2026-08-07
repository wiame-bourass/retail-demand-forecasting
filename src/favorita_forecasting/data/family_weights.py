from __future__ import annotations

from pathlib import Path

try:
    import duckdb
except ImportError:
    duckdb = None

import pandas as pd

from favorita_forecasting.data.raw import raw_file_paths
from favorita_forecasting.utils import read_json


def build_family_weights(
    raw_dir: Path,
    artifacts_dir: Path,
    training_end: pd.Timestamp,
) -> pd.DataFrame:
    files = raw_file_paths(raw_dir)
    manifest = read_json(artifacts_dir / "scope_manifest.json")
    stores = [int(x) for x in manifest["stores"]]
    families = [str(x) for x in manifest["families"]]
    history_start = pd.Timestamp(manifest["history_start"])

    if duckdb is not None:
        store_sql = ",".join(map(str, stores))
        family_sql = ",".join("'" + x.replace("'", "''") + "'" for x in families)
        train_path = str(files["train"]).replace("'", "''")
        items_path = str(files["items"]).replace("'", "''")
        frame = duckdb.connect(":memory:").execute(
            f"""
            SELECT i.family,
                   CAST(t.item_nbr AS BIGINT) item_nbr,
                   CAST(i.perishable AS INTEGER) perishable,
                   SUM(GREATEST(TRY_CAST(t.unit_sales AS DOUBLE), 0)) volume
            FROM read_csv_auto('{train_path}', header=true, sample_size=100000) t
            JOIN read_csv_auto('{items_path}', header=true) i USING(item_nbr)
            WHERE TRY_CAST(t.date AS DATE) BETWEEN DATE '{history_start.date()}' AND DATE '{training_end.date()}'
              AND CAST(t.store_nbr AS INTEGER) IN ({store_sql})
              AND i.family IN ({family_sql})
            GROUP BY 1,2,3
            """
        ).df()
    else:
        if files["train"].stat().st_size > 250_000_000:
            raise RuntimeError("DuckDB is required to calculate family weights on the full dataset")
        train = pd.read_csv(files["train"], parse_dates=["date"])
        items = pd.read_csv(files["items"])
        frame = train[
            train["date"].between(history_start, training_end)
            & train["store_nbr"].isin(stores)
        ].merge(items, on="item_nbr", how="inner")
        frame = frame[frame["family"].isin(families)].copy()
        frame["volume"] = pd.to_numeric(frame["unit_sales"], errors="coerce").clip(lower=0)
        frame = frame.groupby(["family", "item_nbr", "perishable"], as_index=False)["volume"].sum()

    frame["item_weight"] = 1.0 + 0.25 * frame["perishable"].astype(float)
    rows = []
    for family, group in frame.groupby("family"):
        total = group["volume"].sum()
        weight = (group["item_weight"] * group["volume"]).sum() / total if total > 0 else group["item_weight"].mean()
        rows.append(
            {
                "family": family,
                "family_weight": weight,
                "training_end": training_end.date(),
                "weight_definition": "sales-weighted mean of item weights calculated on training only",
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(artifacts_dir / "family_weights_train_only.csv", index=False)
    return result
