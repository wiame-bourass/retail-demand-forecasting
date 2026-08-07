from __future__ import annotations

from common import config_argument
from favorita_forecasting.data.audit import audit_raw_files


if __name__ == "__main__":
    _, cfg = config_argument("Audit Favorita raw files")
    result = audit_raw_files(cfg.path("raw_dir"), cfg.path("outputs_dir") / "audit")
    print(result.to_string(index=False))
