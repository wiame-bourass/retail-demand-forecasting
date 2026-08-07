from __future__ import annotations

import logging
from pathlib import Path

try:
    import py7zr
except ImportError:  # optional until archives are used
    py7zr = None

LOGGER = logging.getLogger(__name__)


def extract_all_archives(raw_dir: Path, overwrite: bool = False) -> list[Path]:
    extracted: list[Path] = []
    for archive in sorted(raw_dir.glob("*.7z")):
        expected = raw_dir / archive.stem
        if expected.exists() and not overwrite:
            LOGGER.info("Skip existing %s", expected.name)
            continue
        if py7zr is None:
            raise RuntimeError("py7zr is required to extract .7z files. Install requirements-core.txt")
        LOGGER.info("Extracting %s", archive.name)
        with py7zr.SevenZipFile(archive, mode="r") as handle:
            handle.extractall(path=raw_dir)
        extracted.append(archive)
    return extracted
