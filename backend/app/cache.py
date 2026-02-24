from __future__ import annotations

import time
import hashlib
import json
from typing import Any, Optional, Dict, Tuple


class TTLCache:
    """
    Minimal in‑memory cache with time‑to‑live semantics.

    The backend uses this both for:
    - LLM planning responses keyed by question text, and
    - executed plan results keyed by a stable hash of the plan.
    """

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
    """
    Produce a deterministic hash for an arbitrary JSON‑serialisable object.

    This is used to key both plans and results so that semantically identical
    questions can reuse work across requests.
    """
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
