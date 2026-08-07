from __future__ import annotations

from pathlib import Path

REQUIRED_FILES = {
    "train": "train.csv",
    "test": "test.csv",
    "items": "items.csv",
    "stores": "stores.csv",
    "oil": "oil.csv",
    "holidays": "holidays_events.csv",
    "transactions": "transactions.csv",
}


def raw_file_paths(raw_dir: Path, require_all: bool = True) -> dict[str, Path]:
    paths = {name: raw_dir / filename for name, filename in REQUIRED_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if require_all and missing:
        raise FileNotFoundError(
            "Fichiers bruts manquants. Extrait les archives Kaggle dans data/raw.\n- "
            + "\n- ".join(missing)
        )
    return paths
