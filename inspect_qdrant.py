"""
Utility script to inspect and query Qdrant Vector DB in My-Invest.

Usage:
    # 1. View collection stats and configuration
    python inspect_qdrant.py --stats

    # 2. List all cached entries and payloads
    python inspect_qdrant.py --list

    # 3. Test a semantic similarity search query
    python inspect_qdrant.py --query "I want to invest 50000 for 3 years"
"""
import argparse
import json
import sys
from pathlib import Path

# Ensure 'src' is in PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Reconfigure stdout for safe Windows terminal output (emojis/unicode)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from my_invest.cache import get_semantic_cache


def show_stats(cache):
    print("\n--- [QDRANT COLLECTION STATS] ---")
    stats = cache.get_stats()
    for k, v in stats.items():
        print(f"  {k:25}: {v}")
    print()


def list_records(cache, limit: int = 10):
    print(f"\n--- [SAVED POINTS IN '{cache.collection_name}'] (showing up to {limit}) ---")
    if not cache._client:
        print("  Client not initialized.")
        return

    points, _ = cache._client.scroll(
        collection_name=cache.collection_name,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    if not points:
        print("  No records found in collection.")
        return

    for idx, point in enumerate(points, 1):
        payload = point.payload or {}
        print(f"\n[Point #{idx}] ID: {point.id}")
        print(f"  Prompt     : {payload.get('prompt', 'N/A')}")
        print(f"  Model      : {payload.get('model_name', 'N/A')}")
        print(f"  Tools Used : {payload.get('tools_used', [])}")
        cached_at = payload.get("cached_at")
        if cached_at:
            import datetime
            dt = datetime.datetime.fromtimestamp(cached_at).strftime("%Y-%m-%d %H:%M:%S")
            print(f"  Cached At  : {dt}")
        preview = payload.get("text", "")[:120].replace("\n", " ")
        print(f"  Advisory   : {preview}...")


def run_query(cache, query_text: str):
    print(f"\n--- [SEMANTIC SEARCH QUERY: '{query_text}'] ---")
    match = cache.lookup(query_text)
    if match:
        print(f"  Status           : HIT (Score >= {cache.threshold})")
        print(f"  Similarity Score : {match.similarity_score * 100:.2f}%")
        print(f"  Original Prompt  : {match.original_prompt}")
        print(f"  Tools Used       : {match.tools_used}")
        print(f"  Model            : {match.model_name}")
        print(f"\n--- Cached Advisory Output ---\n{match.text[:500]}...\n")
    else:
        print(f"  Status           : MISS (No match found above threshold {cache.threshold})")
        # Run raw search to show nearest match regardless of threshold
        embedder = cache._get_embedder()
        query_vector = list(embedder.embed([query_text.strip()]))[0].tolist()
        raw_res = cache._client.query_points(
            collection_name=cache.collection_name,
            query=query_vector,
            limit=3,
        )
        if raw_res.points:
            print(f"\n  Top nearest points below threshold:")
            for p in raw_res.points:
                p_text = (p.payload or {}).get("prompt", "")
                print(f"   - Score: {p.score:.4f} | Prompt: {p_text}")


def main():
    parser = argparse.ArgumentParser(description="Inspect Qdrant vector database.")
    parser.add_argument("--stats", action="store_true", help="Display collection statistics.")
    parser.add_argument("--list", action="store_true", help="List stored vector points and payloads.")
    parser.add_argument("--limit", type=int, default=10, help="Max points to display (default: 10).")
    parser.add_argument("--url", type=str, default=None, help="Qdrant Cloud URL (e.g. https://xyz.cloud.qdrant.io:6333)")
    parser.add_argument("--api-key", type=str, default=None, help="Qdrant Cloud API Key")
    parser.add_argument("--collection", type=str, default=None, help="Collection name (default: investment_advisory_cache)")
    parser.add_argument("--query", type=str, help="Run a semantic similarity query against the collection.")

    args = parser.parse_args()

    if args.url or args.api_key or args.collection:
        from my_invest.cache.qdrant_cache import QdrantSemanticCache
        cache = QdrantSemanticCache(url=args.url, api_key=args.api_key, collection_name=args.collection)
    else:
        cache = get_semantic_cache()

    try:
        if not any([args.stats, args.list, args.query]):
            show_stats(cache)
            list_records(cache, limit=args.limit)
            return

        if args.stats:
            show_stats(cache)
        if args.list:
            list_records(cache, limit=args.limit)
        if args.query:
            run_query(cache, args.query)
    finally:
        if cache._client:
            try:
                cache._client.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
