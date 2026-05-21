"""A small in-memory cache with TTL support."""

import time
from threading import Lock
from typing import Any, Optional


class TTLCache:
    def __init__(self, max_size: int = 1024, default_ttl: float = 60.0):
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._store: dict[str, tuple[Any, float]] = {}
        self._lock = Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if time.monotonic() >= expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        if ttl is None:
            ttl = self._default_ttl
        expires_at = time.monotonic() + ttl
        with self._lock:
            if len(self._store) >= self._max_size and key not in self._store:
                self._evict_oldest()
            self._store[key] = (value, expires_at)

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._store.pop(key, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def _evict_oldest(self) -> None:
        if not self._store:
            return
        oldest_key = min(self._store, key=lambda k: self._store[k][1])
        del self._store[oldest_key]

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None


def memoize(ttl: float = 60.0, max_size: int = 1024):
    """Decorator that caches function results with a TTL."""
    cache = TTLCache(max_size=max_size, default_ttl=ttl)

    def decorator(func):
        def wrapper(*args, **kwargs):
            key = repr((args, tuple(sorted(kwargs.items()))))
            cached = cache.get(key)
            if cached is not None:
                return cached
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result
        wrapper.cache = cache
        return wrapper
    return decorator
