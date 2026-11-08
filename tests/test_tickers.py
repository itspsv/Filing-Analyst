import pytest
from pydantic import ValidationError

from filing_analyst.config import COMPANIES, Settings
from filing_analyst.sec.tickers import load_ticker_table, lookup, pad_cik, parse_ticker_table

FAKE_SEC_FILE = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
}


@pytest.mark.parametrize("raw", [320193, "320193", "0000320193"])
def test_pad_cik_gives_ten_digits(raw: int | str) -> None:
    assert pad_cik(raw) == "0000320193"


@pytest.mark.parametrize("bad", [-5, 12345678901, "abc"])
def test_pad_cik_rejects_invalid(bad: int | str) -> None:
    with pytest.raises(ValueError):
        pad_cik(bad)


def test_parse_ticker_table() -> None:
    table = parse_ticker_table(FAKE_SEC_FILE)
    assert table["MSFT"].cik == "0000789019"
    assert table["AAPL"].name == "Apple Inc."


def test_lookup_unknown_ticker_raises() -> None:
    table = parse_ticker_table(FAKE_SEC_FILE)
    with pytest.raises(KeyError, match="ZZZZ"):
        lookup(["AAPL", "ZZZZ"], table)


def test_settings_reject_user_agent_without_email() -> None:
    with pytest.raises(ValidationError):
        Settings(sec_user_agent="just-a-name")


def _real_settings() -> Settings:
    try:
        return Settings()
    except ValidationError:
        pytest.skip("SEC_USER_AGENT not set in .env")


@pytest.mark.network
def test_real_sec_file_apple_and_all_companies() -> None:
    table = load_ticker_table(_real_settings())
    assert table["AAPL"].cik == "0000320193"
    assert len(lookup(COMPANIES, table)) == len(COMPANIES)
