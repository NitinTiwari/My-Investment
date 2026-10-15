from functools import lru_cache
import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env from project root or current dir
proj_env = Path(__file__).resolve().parent.parent.parent / '.env'
if proj_env.exists():
    load_dotenv(proj_env)
else:
    load_dotenv()


def _clean_str(val: Any) -> Any:
    if isinstance(val, str):
        val = val.strip()
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1].strip()
    return val


def _get_config(key: str, default: Any = None) -> Any:
    """Retrieve config from os.environ or Streamlit Cloud st.secrets."""
    val = os.getenv(key)
    if val is not None and val != "":
        return _clean_str(val)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            s_val = st.secrets[key]
            if s_val is not None and s_val != "":
                return _clean_str(str(s_val))
    except Exception:
        pass
    return default


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        extra='ignore',
        populate_by_name=True,
    )

    app_name: str = 'My-Invest: FD & Mutual Fund Advisor'

    # Tier 1: LLM Gateway Configuration (OpenRouter)
    llm_gateway_url: str = Field(
        default_factory=lambda: _get_config('LLM_GATEWAY_URL', _get_config('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1'))
    )
    llm_gateway_api_key: str | None = Field(
        default_factory=lambda: _get_config('LLM_GATEWAY_API_KEY', _get_config('OPENROUTER_API_KEY'))
    )
    llm_gateway_model: str = Field(
        default_factory=lambda: _get_config('LLM_GATEWAY_MODEL', _get_config('OPENROUTER_MODEL', 'qwen/qwen3.6-27b'))
    )

    # Direct Groq Configuration (Fallback / Dedicated Guardrail)
    groq_api_key: str | None = Field(default_factory=lambda: _get_config('GROQ_API_KEY'))
    model_name: str = Field(default_factory=lambda: _get_config('GROQ_MODEL', 'qwen/qwen3.8-27b'))
    guard_model_name: str = Field(
        default_factory=lambda: _get_config('GUARDRAIL_MODEL', 'meta-llama/Llama-Prompt-Guard-2-22M'),
        alias='guardModelName',
    )

    @property
    def guardModelName(self) -> str:
        return self.guard_model_name

    @property
    def gateway_enabled(self) -> bool:
        """Returns True if a valid LLM gateway API key is configured."""
        return bool(self.llm_gateway_api_key and self.llm_gateway_api_key.strip())

    @property
    def active_model_name(self) -> str:
        """Returns the active model name based on whether gateway is enabled."""
        if self.gateway_enabled:
            return self.llm_gateway_model
        return self.model_name

    @property
    def active_provider(self) -> str:
        """Returns provider display name."""
        return "OpenRouter Gateway" if self.gateway_enabled else "Groq"

    default_investment_amount: float = 200000.0
    agent_tool_call_limit: int = 8
    max_tokens: int = 1500
    temperature: float = Field(default_factory=lambda: float(_get_config('LLM_TEMPERATURE', 0.2)))
    log_level: str = 'INFO'
    qdrant_url: str | None = Field(default_factory=lambda: _get_config('QDRANT_URL'))
    qdrant_api_key: str | None = Field(default_factory=lambda: _get_config('QDRANT_API_KEY'))
    qdrant_collection_name: str = Field(default_factory=lambda: _get_config('QDRANT_COLLECTION', 'investment_advisory_cache'))
    semantic_cache_enabled: bool = Field(default_factory=lambda: str(_get_config('SEMANTIC_CACHE_ENABLED', 'true')).lower() in ('true', '1', 'yes'))
    semantic_cache_threshold: float = Field(default_factory=lambda: float(_get_config('SEMANTIC_CACHE_THRESHOLD', 0.92)))


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
