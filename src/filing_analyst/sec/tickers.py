"""Map stock tickers to SEC company ids (CIKs), zero-padded to 10 digits."""

import json
from collections.abc import Iterable

from pydantic import BaseModel

from filing_analyst.config import COMPANIES, Settings
from filing_analyst.sec.http import fetch_cached

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
CIK_LENGTH = 10


class Company(BaseModel):
    ticker: str
    cik: str
    name: str


def pad_cik(cik: int | str) -> str:
    """Turn 320193 or "320193" into "0000320193"."""
    number = int(cik)
    padded = str(number).zfill(CIK_LENGTH)
    if number < 0 or len(padded) > CIK_LENGTH:
        raise ValueError(f"not a valid CIK: {cik!r}")
    return padded


def parse_ticker_table(raw: dict[str, dict]) -> dict[str, Company]:
    """Turn the SEC file ({"0": {"cik_str", "ticker", "title"}, ...}) into ticker -> Company."""
    return {
        row["ticker"].upper(): Company(
            ticker=row["ticker"].upper(), cik=pad_cik(row["cik_str"]), name=row["title"]
        )
        for row in raw.values()
    }


def lookup(tickers: Iterable[str], table: dict[str, Company]) -> list[Company]:
    """Return the Company for each ticker; fail loudly if any ticker is unknown."""
    wanted = [t.upper() for t in tickers]
    missing = [t for t in wanted if t not in table]
    if missing:
        raise KeyError(f"tickers not in SEC table: {missing}")
    return [table[t] for t in wanted]


def load_ticker_table(settings: Settings) -> dict[str, Company]:
    """Download the SEC ticker file once (cached on disk) and parse it."""
    path = fetch_cached(
        TICKERS_URL, settings.raw_dir / "company_tickers.json", settings.sec_user_agent
    )
    return parse_ticker_table(json.loads(path.read_text()))


def main() -> None:
    table = load_ticker_table(Settings())
    for company in lookup(COMPANIES, table):
        print(f"{company.ticker:<6} {company.cik}  {company.name}")


if __name__ == "__main__":
    main()
