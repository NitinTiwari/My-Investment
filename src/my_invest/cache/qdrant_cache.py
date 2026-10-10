"""
Semantic Caching Engine powered by Qdrant (Cloud or Local Embedded) and FastEmbed.
Matches incoming investment prompts against pre-computed advisory blueprints using
cosine similarity embeddings, completely eliminating redundant Groq LLM inferences.
"""
from dataclasses import dataclass
from functools import lru_cache
import logging
import os
from pathlib import Path
from threading import Lock
import time
from typing import Any, Dict, List, Optional
import uuid

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from my_invest.settings import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SemanticCacheMatch:
    text: str
    original_prompt: str
    similarity_score: float
    tools_used: tuple[str, ...]
    model_name: str


class QdrantSemanticCache:
    """
    Tier-2 Semantic Cache using Qdrant Vector Database.
    Connects to Qdrant Cloud when QDRANT_URL and QDRANT_API_KEY are configured,
    with automatic graceful fallback to local embedded on-disk storage.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
        threshold: Optional[float] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self.url = url or settings.qdrant_url
        self.api_key = api_key or settings.qdrant_api_key
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.threshold = threshold if threshold is not None else settings.semantic_cache_threshold
        self.enabled = enabled if enabled is not None else settings.semantic_cache_enabled

        self._lock = Lock()
        self._client: Optional[QdrantClient] = None
        self._embedder: Optional[TextEmbedding] = None
        self._mode = "uninitialized"
        self._hits = 0
        self._misses = 0

        if self.enabled:
            self._initialize_client()

    def _get_embedder(self) -> TextEmbedding:
        if self._embedder is None:
            self._embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        return self._embedder

    def _initialize_client(self) -> None:
        try:
            if self.url and self.api_key:
                logger.info(f"Connecting to Qdrant Cloud at {self.url}...")
                self._client = QdrantClient(
                    url=self.url,
                    api_key=self.api_key,
                    timeout=10.0,
                )
                self._mode = "Qdrant Cloud"
            else:
                # Fallback to local on-disk storage
                storage_dir = Path("./data/qdrant_storage").resolve()
                storage_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Connecting to local embedded Qdrant storage at {storage_dir}...")
                self._client = QdrantClient(path=str(storage_dir))
                self._mode = "Local Embedded (Disk)"

            self._ensure_collection()
        except Exception as e:
            logger.warning(f"Could not connect to Qdrant ({e}). Falling back to in-memory Qdrant instance.")
            self._client = QdrantClient(":memory:")
            self._mode = "In-Memory (Fallback)"
            self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self._client:
            return
        try:
            if not self._client.collection_exists(self.collection_name):
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )
                logger.info(f"Created Qdrant collection '{self.collection_name}' (vector size 384, Cosine).")
        except Exception as e:
            logger.error(f"Error ensuring Qdrant collection '{self.collection_name}': {e}")

    def lookup(self, prompt: str) -> Optional[SemanticCacheMatch]:
        """
        Queries Qdrant for a semantically similar previous investment prompt.
        Returns the cached advisory blueprint if similarity >= threshold, else None.
        """
        if not self.enabled or not self._client:
            return None

        try:
            embedder = self._get_embedder()
            query_vector = list(embedder.embed([prompt.strip()]))[0].tolist()

            results = self._client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=1,
                score_threshold=self.threshold,
            )

            if results.points:
                top_match = results.points[0]
                payload = top_match.payload or {}
                with self._lock:
                    self._hits += 1

                logger.info(
                    f"Semantic Cache Hit! Similarity: {top_match.score:.4f} for prompt: '{prompt[:60]}...'"
                )
                return SemanticCacheMatch(
                    text=payload.get("text", ""),
                    original_prompt=payload.get("prompt", ""),
                    similarity_score=float(top_match.score),
                    tools_used=tuple(payload.get("tools_used", [])),
                    model_name=payload.get("model_name", "qdrant-cached"),
                )

            with self._lock:
                self._misses += 1
            return None

        except Exception as e:
            logger.warning(f"Error querying semantic cache: {e}")
            with self._lock:
                self._misses += 1
            return None

    def store(
        self,
        prompt: str,
        text: str,
        tools_used: tuple[str, ...],
        model_name: str,
    ) -> bool:
        """
        Embeds and upserts an investment prompt and its generated advisory blueprint into Qdrant.
        """
        if not self.enabled or not self._client or not prompt or not text:
            return False

        try:
            embedder = self._get_embedder()
            vector = list(embedder.embed([prompt.strip()]))[0].tolist()

            point_id = str(uuid.uuid4())
            point = PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "prompt": prompt.strip(),
                    "text": text,
                    "tools_used": list(tools_used),
                    "model_name": model_name,
                    "cached_at": time.time(),
                },
            )

            self._client.upsert(
                collection_name=self.collection_name,
                points=[point],
            )
            logger.info(f"Stored prompt in Qdrant semantic cache. Point ID: {point_id}")
            return True
        except Exception as e:
            logger.error(f"Error storing result in Qdrant semantic cache: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Returns diagnostic metrics for the semantic cache."""
        total_queries = self._hits + self._misses
        hit_rate = round((self._hits / total_queries) * 100.0, 1) if total_queries > 0 else 0.0

        points_count = 0
        if self._client:
            try:
                info = self._client.get_collection(self.collection_name)
                points_count = getattr(info, "points_count", 0) or 0
            except Exception:
                pass

        return {
            "enabled": self.enabled,
            "mode": self._mode,
            "hits": self._hits,
            "misses": self._misses,
            "total_queries": total_queries,
            "hit_rate_pct": hit_rate,
            "threshold": self.threshold,
            "collection_name": self.collection_name,
            "total_cached_blueprints": points_count,
            "cloud_url": self.url if self.url else "Local Disk/Memory",
        }


@lru_cache
def get_semantic_cache() -> QdrantSemanticCache:
    """Singleton provider for QdrantSemanticCache."""
    return QdrantSemanticCache()
