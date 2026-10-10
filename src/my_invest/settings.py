from functools import lru_cache
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env from project root or current dir
proj_env = Path(__file__).resolve().parent.parent.parent / '.env'
if proj_env.exists():
    load_dotenv(proj_env)
else:
    load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        extra='ignore',
    )

    app_name: str = 'My-Invest: FD & Mutual Fund Advisor'
    groq_api_key: str | None = Field(default_factory=lambda: os.getenv('GROQ_API_KEY'))
    model_name: str = Field(default_factory=lambda: os.getenv('GROQ_MODEL', 'qwen/qwen3.8-27b'))
    default_investment_amount: float = 200000.0
    agent_tool_call_limit: int = 8
    max_tokens: int = 900
    log_level: str = 'INFO'
    qdrant_url: str | None = Field(default_factory=lambda: os.getenv('QDRANT_URL'))
    qdrant_api_key: str | None = Field(default_factory=lambda: os.getenv('QDRANT_API_KEY'))
    qdrant_collection_name: str = Field(default_factory=lambda: os.getenv('QDRANT_COLLECTION', 'investment_advisory_cache'))
    semantic_cache_enabled: bool = Field(default_factory=lambda: os.getenv('SEMANTIC_CACHE_ENABLED', 'true').lower() in ('true', '1', 'yes'))
    semantic_cache_threshold: float = Field(default_factory=lambda: float(os.getenv('SEMANTIC_CACHE_THRESHOLD', '0.92')))


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
