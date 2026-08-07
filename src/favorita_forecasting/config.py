from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    raw: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.raw.get(name, {}))

    def path(self, key: str) -> Path:
        value = self.raw["paths"][key]
        path = Path(value)
        return path if path.is_absolute() else self.root / path


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    root = config_path.parent.parent
    return ProjectConfig(root=root, raw=raw)
