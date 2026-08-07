from __future__ import annotations

from common import config_argument

from favorita_forecasting.data.extraction import extract_all_archives

if __name__ == "__main__":
    _, cfg = config_argument("Extract Kaggle .7z archives")
    archives = extract_all_archives(cfg.path("raw_dir"))
    print(f"Extracted {len(archives)} archive(s)")
