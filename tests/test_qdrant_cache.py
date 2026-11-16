"""
Unit tests for QdrantSemanticCache with Pattern B Native Range Filtering TTL.
Uses a deterministic in-memory vector embedder to verify TTL range filtering.
"""
import time
import uuid
import numpy as np
import pytest
from qdrant_client.models import PointStruct

from my_invest.cache.qdrant_cache import QdrantSemanticCache
from my_invest.settings import settings


class MockEmbedder:
    """Fast, deterministic embedder producing 384-dimensional unit vectors."""
    def embed(self, texts):
        for t in texts:
            # Deterministic vector based on prompt length
            v = np.zeros(384, dtype=float)
            v[0] = 1.0
            yield v


def create_test_cache(collection_name: str, threshold: float = 0.85, ttl_days: float = 2.0) -> QdrantSemanticCache:
    cache = QdrantSemanticCache(
        url=":memory:",
        collection_name=collection_name,
        threshold=threshold,
        enabled=True,
        ttl_days=ttl_days,
    )
    cache._embedder = MockEmbedder()
    return cache


def test_settings_semantic_cache_ttl():
    assert hasattr(settings, "semantic_cache_ttl_days")
    assert settings.semantic_cache_ttl_days == 2.0
    assert settings.semantic_cache_ttl_seconds == 2.0 * 86400.0


def test_qdrant_cache_ttl_filtering_fresh_hit():
    """Freshly stored point (within 2 days) should be returned as a cache hit."""
    cache = create_test_cache("test_ttl_fresh", threshold=0.85, ttl_days=2.0)

    prompt = "I want to invest 2 lakhs in bank FD for 3 years"
    text = "Recommended allocation: 100% in SBI FD at 7.00%."
    stored = cache.store(
        prompt=prompt,
        text=text,
        tools_used=("get_indian_bank_fd_rates", "calculate_fd_maturity"),
        model_name="test-model",
    )
    assert stored is True

    match = cache.lookup(prompt)
    assert match is not None
    assert match.text == text
    assert match.similarity_score >= 0.85
    assert "get_indian_bank_fd_rates" in match.tools_used


def test_qdrant_cache_ttl_filtering_expired_miss():
    """Point older than ttl_days (e.g. 3 days old when TTL is 2 days) must be filtered out natively."""
    cache = create_test_cache("test_ttl_expired", threshold=0.85, ttl_days=2.0)

    prompt = "Invest 5 lakhs in mutual funds for 5 years"
    text = "Recommended: Parag Parikh Flexi Cap Direct Growth."

    embedder = cache._get_embedder()
    vector = list(embedder.embed([prompt.strip()]))[0].tolist()
    expired_time = time.time() - (3 * 86400.0)  # 3 days ago

    cache._client.upsert(
        collection_name=cache.collection_name,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "prompt": prompt,
                    "text": text,
                    "tools_used": ["get_top_indian_mutual_funds_by_category"],
                    "model_name": "test-model",
                    "cached_at": expired_time,
                },
            )
        ],
    )

    # With TTL = 2.0 days, this 3-day-old point must be filtered out by Qdrant's native range filter
    match = cache.lookup(prompt)
    assert match is None, "Expired point (>2 days old) should have been filtered out by Qdrant range filter"


def test_qdrant_cache_ttl_custom_window():
    """If TTL is extended to 5 days, the 3-day-old point should now match."""
    cache = create_test_cache("test_ttl_window", threshold=0.85, ttl_days=5.0)

    prompt = "Invest 10 lakhs in FD for senior citizen"
    text = "Recommended: Unity SFB at 9.00% under DICGC."

    embedder = cache._get_embedder()
    vector = list(embedder.embed([prompt.strip()]))[0].tolist()
    three_days_ago = time.time() - (3 * 86400.0)

    cache._client.upsert(
        collection_name=cache.collection_name,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "prompt": prompt,
                    "text": text,
                    "tools_used": ["get_indian_bank_fd_rates"],
                    "model_name": "test-model",
                    "cached_at": three_days_ago,
                },
            )
        ],
    )

    match = cache.lookup(prompt)
    assert match is not None
    assert match.text == text


def test_qdrant_cache_get_stats_includes_ttl():
    cache = create_test_cache("test_ttl_stats", threshold=0.92, ttl_days=2.0)
    stats = cache.get_stats()
    assert "ttl_days" in stats
    assert stats["ttl_days"] == 2.0
    assert "ttl_seconds" in stats
    assert stats["ttl_seconds"] == 172800.0
