"""
OpenRouter API Gateway Client & Telemetry Monitor for My-Invest.

Provides:
1. Real-time account usage and budget tracking via OpenRouter /auth/key API.
2. TTL caching to prevent excessive polling.
3. Cost estimation per query based on active model and token counts.
"""
from dataclasses import dataclass
from functools import lru_cache
import logging
from time import time
from typing import Any, Optional
import requests

from my_invest.settings import settings

logger = logging.getLogger(__name__)

# Cache TTL for auth key info (30 seconds)
_AUTH_KEY_CACHE: dict[str, Any] = {}
_AUTH_KEY_CACHE_TTL = 30.0


@dataclass(frozen=True)
class OpenRouterAccountInfo:
    success: bool
    label: str = "Default Key"
    usage_usd: float = 0.0
    usage_daily_usd: float = 0.0
    limit_usd: Optional[float] = None
    limit_remaining_usd: Optional[float] = None
    is_free_tier: bool = True
    free_requests_used: int = 0
    free_requests_limit: int = 50
    free_requests_remaining: int = 0
    error: Optional[str] = None


def get_openrouter_account_info(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    force_refresh: bool = False,
) -> OpenRouterAccountInfo:
    """
    Fetches real-time usage, budget limit, and free tier status from OpenRouter's /auth/key endpoint.
    Cached for 30s to keep Streamlit UI snappy.
    """
    key = api_key or settings.llm_gateway_api_key
    if not key or not key.strip():
        return OpenRouterAccountInfo(
            success=False,
            error="No OpenRouter API key configured.",
        )

    now = time()
    cache_entry = _AUTH_KEY_CACHE.get(key)
    if not force_refresh and cache_entry and (now - cache_entry["timestamp"] < _AUTH_KEY_CACHE_TTL):
        return cache_entry["data"]

    url = (base_url or settings.llm_gateway_url or "https://openrouter.ai/api/v1").rstrip("/")
    # Auth key endpoint is always at /auth/key relative to /api/v1
    auth_url = f"{url}/auth/key"

    headers = {
        "Authorization": f"Bearer {key.strip()}",
        "HTTP-Referer": "https://github.com/my-invest",
        "X-Title": "My-Invest Financial Advisor",
    }

    try:
        resp = requests.get(auth_url, headers=headers, timeout=6.0)
        if resp.status_code == 200:
            payload = resp.json().get("data", {})
            free_req = payload.get("free_model_daily_requests") or {}
            info = OpenRouterAccountInfo(
                success=True,
                label=payload.get("label") or "OpenRouter Key",
                usage_usd=float(payload.get("usage", 0.0)),
                usage_daily_usd=float(payload.get("usage_daily", 0.0)),
                limit_usd=float(payload["limit"]) if payload.get("limit") is not None else None,
                limit_remaining_usd=float(payload["limit_remaining"]) if payload.get("limit_remaining") is not None else None,
                is_free_tier=bool(payload.get("is_free_tier", True)),
                free_requests_used=int(free_req.get("used", 0)),
                free_requests_limit=int(free_req.get("limit", 50)),
                free_requests_remaining=int(free_req.get("remaining", 0)),
            )
            _AUTH_KEY_CACHE[key] = {"data": info, "timestamp": now}
            return info
        else:
            err_msg = f"HTTP {resp.status_code}: {resp.text[:120]}"
            logger.warning(f"Failed to fetch OpenRouter account info: {err_msg}")
            return OpenRouterAccountInfo(success=False, error=err_msg)
    except Exception as exc:
        logger.warning(f"Exception checking OpenRouter key: {exc}")
        return OpenRouterAccountInfo(success=False, error=str(exc))


# Approximate pricing per 1M tokens ($ USD) for popular models
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # model_id_prefix: (input_per_m, output_per_m)
    "qwen/qwen3.6-27b": (0.32, 3.20),
    "qwen/qwen3.8-27b": (0.24, 2.20),
    "qwen/": (0.30, 2.50),
    "nvidia/": (0.0, 0.0),  # Free models
    "liquid/": (0.0, 0.0),
    "google/": (0.0, 0.0),
}


def estimate_token_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimates query cost in USD based on model identifier and token metrics.
    Free tier models (:free) return $0.00.
    """
    if not model_name or ":free" in model_name:
        return 0.0

    in_price = 0.30
    out_price = 2.50

    for prefix, (ip, op) in MODEL_PRICING.items():
        if model_name.startswith(prefix) or prefix in model_name:
            in_price, out_price = ip, op
            break

    cost = (input_tokens / 1_000_000.0 * in_price) + (output_tokens / 1_000_000.0 * out_price)
    return round(cost, 6)
