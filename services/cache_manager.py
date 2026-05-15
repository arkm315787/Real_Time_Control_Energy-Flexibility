"""Small cache manager for service-layer dataframes and metadata."""

from __future__ import annotations

import hashlib
import pickle
import time
from pathlib import Path
from typing import Callable, Generic, TypeVar


T = TypeVar("T")


class CacheManager(Generic[T]):
    """In-memory cache with optional pickle-backed persistence."""

    def __init__(self, ttl_seconds: int = 900, cache_dir: str | None = None) -> None:
        self.ttl_seconds = int(ttl_seconds)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._memory: dict[str, tuple[float, T]] = {}
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(*parts: object) -> str:
        digest = hashlib.sha256()
        for part in parts:
            digest.update(repr(part).encode("utf-8"))
            digest.update(b"|")
        return digest.hexdigest()

    def get(self, key: str) -> T | None:
        now = time.time()
        cached = self._memory.get(key)
        if cached and now - cached[0] <= self.ttl_seconds:
            return cached[1]
        if self.cache_dir:
            path = self.cache_dir / f"{key}.pkl"
            if path.exists() and now - path.stat().st_mtime <= self.ttl_seconds:
                with path.open("rb") as handle:
                    value = pickle.load(handle)
                self._memory[key] = (now, value)
                return value
        return None

    def set(self, key: str, value: T) -> T:
        now = time.time()
        self._memory[key] = (now, value)
        if self.cache_dir:
            path = self.cache_dir / f"{key}.pkl"
            with path.open("wb") as handle:
                pickle.dump(value, handle)
        return value

    def get_or_set(self, key: str, factory: Callable[[], T]) -> T:
        cached = self.get(key)
        if cached is not None:
            return cached
        return self.set(key, factory())
