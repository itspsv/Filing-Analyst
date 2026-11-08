"""Settings read from environment variables and the .env file."""

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

COMPANIES: tuple[str, ...] = (
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "INTC", "TSLA", "NFLX",
    "CRM", "ORCL", "ADBE", "QCOM", "AVGO", "TXN", "MU", "IBM", "CSCO", "PYPL",
)  # fmt: skip


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    sec_user_agent: str
    data_dir: Path = PROJECT_ROOT / "data"

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
