"""Small JSON key-value cache (keys are audio file hashes)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class JsonCache:
    def __init__(self, path: Path, meta: dict[str, Any] | None = None) -> None:
        self.path = path
        self.meta = meta or {}
        self.entries: dict[str, Any] = {}
        self.dirty = False
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self.entries = data.get("entries", {})

    def get(self, key: str) -> Any:
        return self.entries.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def set(self, key: str, value: Any) -> None:
        self.entries[key] = value
        self.dirty = True

    def save(self) -> None:
        if not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = {"meta": self.meta, "entries": self.entries}
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)
        self.dirty = False
