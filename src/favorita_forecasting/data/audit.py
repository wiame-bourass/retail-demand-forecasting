from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

try:
    import duckdb
except ImportError:  # demo fallback
    duckdb = None
import pandas as pd

from favorita_forecasting.data.raw import raw_file_paths
from favorita_forecasting.utils import ensure_dir, write_json

LOGGER = logging.getLogger(__name__)

EXPECTED_KEYS = {
    "train": ["date", "store_nbr", "item_nbr"],
    "test": ["id"],
    "items": ["item_nbr"],
    "stores": ["store_nbr"],
    "oil": ["date"],
    "holidays": ["date", "type", "locale", "locale_name", "description"],
    "transactions": ["date", "store_nbr"],
}


def _header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as stream:
        return next(csv.reader(stream))


def _duckdb_relation(path: Path) -> str:
    escaped = str(path).replace("'", "''")
    return f"read_csv_auto('{escaped}', header=true, sample_size=100000, ignore_errors=true)"


def _safe_scalar(con: Any, query: str) -> Any:
    try:
        return con.execute(query).fetchone()[0]
    except Exception as exc:  # pragma: no cover - defensive for malformed external files
        LOGGER.warning("Audit query failed: %s", exc)
        return None


def audit_raw_files(raw_dir: Path, output_dir: Path) -> pd.DataFrame:
    files = raw_file_paths(raw_dir, require_all=True)
    con = duckdb.connect(database=":memory:") if duckdb is not None else None
    rows: list[dict[str, Any]] = []
    detailed: dict[str, Any] = {}

    for name, path in files.items():
        columns = _header(path)
        keys = EXPECTED_KEYS[name]
        if con is not None:
            relation = _duckdb_relation(path)
            row_count = _safe_scalar(con, f"SELECT COUNT(*) FROM {relation}")
            duplicate_count = None
            if set(keys).issubset(columns):
                key_expr = ", ".join(f'"{key}"' for key in keys)
                duplicate_count = _safe_scalar(
                    con,
                    f"SELECT COALESCE(SUM(n-1), 0) FROM (SELECT COUNT(*) n FROM {relation} GROUP BY {key_expr} HAVING COUNT(*) > 1)",
                )
            date_min = date_max = None
            if "date" in columns:
                date_min = _safe_scalar(con, f"SELECT MIN(TRY_CAST(date AS DATE)) FROM {relation}")
                date_max = _safe_scalar(con, f"SELECT MAX(TRY_CAST(date AS DATE)) FROM {relation}")
        else:
            usecols = list(dict.fromkeys(keys + (["date"] if "date" in columns else [])))
            chunks = pd.read_csv(path, usecols=[c for c in usecols if c in columns], chunksize=250_000)
            row_count = 0
            date_min = date_max = None
            key_frames = []
            for chunk in chunks:
                row_count += len(chunk)
                if "date" in chunk:
                    dates = pd.to_datetime(chunk["date"], errors="coerce")
                    current_min, current_max = dates.min(), dates.max()
                    date_min = current_min if date_min is None or current_min < date_min else date_min
                    date_max = current_max if date_max is None or current_max > date_max else date_max
                if set(keys).issubset(chunk.columns) and path.stat().st_size < 250_000_000:
                    key_frames.append(chunk[keys])
            duplicate_count = None
            if key_frames:
                key_df = pd.concat(key_frames, ignore_index=True)
                duplicate_count = int(key_df.duplicated(keys).sum())

        rows.append(
            {
                "file": path.name,
                "size_mb": round(path.stat().st_size / 1024**2, 3),
                "row_count": row_count,
                "column_count": len(columns),
                "duplicate_key_rows": duplicate_count,
                "date_min": date_min,
                "date_max": date_max,
                "columns": "|".join(columns),
            }
        )
        detailed[name] = {"path": str(path), "columns": columns, "expected_keys": keys}

    audit = pd.DataFrame(rows)
    ensure_dir(output_dir)
    audit.to_csv(output_dir / "raw_file_audit.csv", index=False)
    write_json(detailed, output_dir / "raw_schema_dictionary.json")
    return audit
