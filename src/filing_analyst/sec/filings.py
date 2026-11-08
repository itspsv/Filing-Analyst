"""Find each company's latest 10-K in its SEC filing list and download it (cached)."""

import json
from datetime import date
from pathlib import Path

from pydantic import BaseModel, computed_field

from filing_analyst.config import COMPANIES, Settings
from filing_analyst.sec.http import fetch_cached
from filing_analyst.sec.tickers import Company, load_ticker_table, lookup

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
FORM = "10-K"


class Filing(BaseModel):
    ticker: str
    cik: str
    form: str
    accession: str
    filing_date: date
    report_date: date
    primary_document: str

    @computed_field
    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(
            cik=int(self.cik),
            accession=self.accession.replace("-", ""),
            document=self.primary_document,
        )

    @computed_field
    @property
    def local_name(self) -> str:
        return f"{self.ticker}_{self.accession}.htm"


def latest_10k(company: Company, submissions: dict) -> Filing:
    """Pick the newest plain 10-K (not 10-K/A, not 10-Q) from a submissions JSON."""
    recent = submissions["filings"]["recent"]
    hits = [i for i, form in enumerate(recent["form"]) if form == FORM]
    if not hits:
        raise LookupError(f"no {FORM} in recent filings for {company.ticker}")
    i = max(hits, key=lambda j: recent["filingDate"][j])
    return Filing(
        ticker=company.ticker,
        cik=company.cik,
        form=FORM,
        accession=recent["accessionNumber"][i],
        filing_date=recent["filingDate"][i],
        report_date=recent["reportDate"][i],
        primary_document=recent["primaryDocument"][i],
    )


def download_latest_10k(company: Company, settings: Settings) -> tuple[Filing, bool]:
    """Download one company's latest 10-K. Returns the filing and whether it was already cached."""
    submissions_path = fetch_cached(
        SUBMISSIONS_URL.format(cik=company.cik),
        settings.raw_dir / "submissions" / f"CIK{company.cik}.json",
        settings.sec_user_agent,
    )
    filing = latest_10k(company, json.loads(submissions_path.read_text()))
    dest = settings.raw_dir / "10k" / filing.local_name
    was_cached = dest.exists()
    fetch_cached(filing.url, dest, settings.sec_user_agent)
    return filing, was_cached


def download_all(settings: Settings) -> list[tuple[Filing, bool]]:
    companies = lookup(COMPANIES, load_ticker_table(settings))
    return [download_latest_10k(company, settings) for company in companies]


def write_manifest(filings: list[Filing], path: Path) -> None:
    path.write_text(json.dumps([f.model_dump(mode="json") for f in filings], indent=2))


def main() -> None:
    settings = Settings()
    results = download_all(settings)
    for filing, was_cached in results:
        status = "cached    " if was_cached else "downloaded"
        print(f"{status}  {filing.ticker:<6} {filing.report_date}  {filing.local_name}")
    write_manifest([f for f, _ in results], settings.raw_dir / "10k" / "manifest.json")
    downloaded = sum(not cached for _, cached in results)
    print(f"{len(results)} filings, {downloaded} downloaded, {len(results) - downloaded} cached")


if __name__ == "__main__":
    main()
