from __future__ import annotations

import time
import hashlib
import json
from typing import Any, Optional, Dict, Tuple

class TTLCache:
    def __init__(self, ttl_seconds: int = 300) -> None:
        self.ttl = max(1, int(ttl_seconds))
        self._store: Dict[str, Tuple[float, Any]] = {}

    def _now(self) -> float:
        return time.time()

    def _expired(self, ts: float) -> bool:
        return (self._now() - ts) > self.ttl

    def get(self, key: str) -> Optional[Any]:
        item = self._store.get(key)
        if not item:
            return None
        ts, val = item
        if self._expired(ts):
            self._store.pop(key, None)
            return None
        return val

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (self._now(), value)

def stable_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
