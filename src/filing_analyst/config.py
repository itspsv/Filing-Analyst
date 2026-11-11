"""Settings read from environment variables and the .env file."""

from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

COMPANIES: tuple[str, ...] = (
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "AMAT", "TSLA", "NFLX",
    "CRM", "ORCL", "ADBE", "QCOM", "AVGO", "TXN", "MU", "NOW", "CSCO", "PYPL",
)  # fmt: skip
# INTC and IBM were swapped out: Intel's 10-K is not laid out by Item number, and IBM's
# Item 7 only points to a separate annual-report exhibit.


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    sec_user_agent: str
    data_dir: Path = PROJECT_ROOT / "data"

    # LLM: any OpenAI-compatible chat API. Defaults are Groq's free tier for gpt-oss-120b
    # (limits page, Nov 2026: 30 req/min, 1K req/day, 8K tokens/min, 200K tokens/day).
    llm_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("llm_api_key", "groq_api_key")
    )
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-120b"
    llm_reasoning_effort: str | None = "low"
    llm_strict_schema: bool = True  # Groq gpt-oss guarantees the JSON schema; off for others
    llm_tokens_per_minute: int = 8_000
    llm_requests_per_minute: int = 30
    llm_daily_token_limit: int = 190_000  # 200K minus a safety margin
    llm_max_concurrency: int = 2
    llm_eur_per_million_in: float = 0.0  # free tier; set for a paid backup provider
    llm_eur_per_million_out: float = 0.0

    @field_validator("sec_user_agent")
    @classmethod
    def must_look_like_name_and_email(cls, value: str) -> str:
        if " " not in value.strip() or "@" not in value:
            raise ValueError('SEC_USER_AGENT must be "Name email@example.com"')
        return value.strip()

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def sections_dir(self) -> Path:
        return self.data_dir / "sections"

    @property
    def facts_db(self) -> Path:
        return self.data_dir / "facts.duckdb"

    @property
    def chunks_dir(self) -> Path:
        return self.data_dir / "chunks"

    @property
    def risks_dir(self) -> Path:
        return self.data_dir / "risks"

    @property
    def llm_cache_dir(self) -> Path:
        return self.data_dir / "llm_cache"

    @property
    def run_log(self) -> Path:
        return self.data_dir / "run_log.jsonl"
