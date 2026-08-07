from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


def write_frame(frame: pd.DataFrame, parquet_path: Path, index: bool = False) -> Path:
    """Write Parquet when available, otherwise a pickle fallback for constrained demo environments."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(parquet_path, index=index)
        return parquet_path
    except ImportError:
        fallback = parquet_path.with_suffix(".pkl")
        LOGGER.warning("Parquet engine unavailable; writing fallback %s", fallback)
        frame.to_pickle(fallback)
        return fallback


def read_frame(parquet_path: Path) -> pd.DataFrame:
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except ImportError:
            pass
    fallback = parquet_path.with_suffix(".pkl")
    if fallback.exists():
        return pd.read_pickle(fallback)
    raise FileNotFoundError(f"Neither {parquet_path} nor fallback {fallback} exists")


def frame_exists(parquet_path: Path) -> bool:
    return parquet_path.exists() or parquet_path.with_suffix(".pkl").exists()
