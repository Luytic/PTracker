from __future__ import annotations

import json
from pathlib import Path


class JsonlLogger:
    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("w", encoding="utf-8")

    def write(self, row: dict) -> None:
        self._fh.write(json.dumps(row, ensure_ascii=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
