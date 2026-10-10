"""
Thread-safe in-memory TTL (Time-To-Live) cache decorator.
Eliminates redundant external HTTP requests, DuckDuckGo searches, and AMFI queries.
Provides LRU eviction, expiration checks, zero external dependencies, and execution telemetry.
"""
from collections import OrderedDict
from dataclasses import dataclass
from functools import wraps
from threading import Lock
import time
from typing import Any, Callable, Dict, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

_REGISTERED_CACHES: Dict[str, Callable[[], "CacheInfo"]] = {}


@dataclass(frozen=True)
class CacheInfo:
    name: str
    hits: int
    misses: int
    maxsize: int
    currsize: int
    ttl_seconds: int

    @property
    def total_calls(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate_pct(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return round((self.hits / self.total_calls) * 100.0, 1)


def ttl_cache(maxsize: int = 128, ttl_seconds: int = 3600) -> Callable[[F], F]:
    """
    Decorator to wrap a function with an in-memory TTL + LRU cache.

    Args:
        maxsize: Maximum number of distinct arguments to cache before evicting LRU item.
        ttl_seconds: Cache validity duration in seconds.
    """
    def decorator(func: F) -> F:
        cache: OrderedDict[Any, tuple[Any, float]] = OrderedDict()
        lock = Lock()
        hits = 0
        misses = 0
        fn_name = getattr(func, "__name__", "unknown_fn")

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal hits, misses

            # Build hashable key from args and sorted kwargs
            key = (args, tuple(sorted(kwargs.items())))
            now = time.monotonic()

            with lock:
                if key in cache:
                    val, expire_at = cache[key]
                    if now < expire_at:
                        hits += 1
                        cache.move_to_end(key)
                        return val
                    else:
                        del cache[key]
                misses += 1

            result = func(*args, **kwargs)

            with lock:
                if len(cache) >= maxsize:
                    cache.popitem(last=False)
                cache[key] = (result, now + ttl_seconds)

            return result

        def cache_info() -> CacheInfo:
            with lock:
                return CacheInfo(
                    name=fn_name,
                    hits=hits,
                    misses=misses,
                    maxsize=maxsize,
                    currsize=len(cache),
                    ttl_seconds=ttl_seconds,
                )

        def cache_clear() -> None:
            nonlocal hits, misses
            with lock:
                cache.clear()
                hits = 0
                misses = 0

        wrapper.cache_info = cache_info  # type: ignore[attr-defined]
        wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]

        _REGISTERED_CACHES[fn_name] = cache_info
        return wrapper  # type: ignore[return-value]

    return decorator


def get_all_tool_cache_metrics() -> Dict[str, CacheInfo]:
    """Returns a dictionary of current cache statistics across all decorated tools."""
    return {name: getter() for name, getter in _REGISTERED_CACHES.items()}


def get_aggregated_cache_summary() -> Dict[str, Any]:
    """Returns overall total hits, total misses, and savings summary."""
    stats = get_all_tool_cache_metrics()
    total_hits = sum(s.hits for s in stats.values())
    total_misses = sum(s.misses for s in stats.values())
    total_calls = total_hits + total_misses
    hit_rate = round((total_hits / total_calls * 100.0), 1) if total_calls > 0 else 0.0

    return {
        "total_hits": total_hits,
        "total_misses": total_misses,
        "total_calls": total_calls,
        "hit_rate_pct": hit_rate,
        "caches": stats,
    }
