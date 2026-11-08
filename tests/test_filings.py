import pytest
from pydantic import ValidationError

from filing_analyst.config import COMPANIES, Settings
from filing_analyst.sec.filings import download_all, latest_10k
from filing_analyst.sec.tickers import Company

APPLE = Company(ticker="AAPL", cik="0000320193", name="Apple Inc.")


def fake_submissions(rows: list[tuple[str, str, str]]) -> dict:
    """Build a minimal SEC submissions JSON from (form, accession, filing_date) rows."""
    return {
        "filings": {
            "recent": {
                "form": [form for form, _, _ in rows],
                "accessionNumber": [acc for _, acc, _ in rows],
                "filingDate": [filed for _, _, filed in rows],
                "reportDate": ["2025-09-27"] * len(rows),
                "primaryDocument": [f"doc-{acc}.htm" for _, acc, _ in rows],
            }
        }
    }


def test_latest_10k_skips_other_forms_and_picks_newest() -> None:
    subs = fake_submissions(
        [
            ("10-Q", "0000320193-26-000005", "2026-05-01"),
            ("10-K/A", "0000320193-26-000002", "2026-02-01"),
            ("10-K", "0000320193-24-000123", "2024-11-01"),
            ("10-K", "0000320193-25-000079", "2025-10-31"),
        ]
    )
    filing = latest_10k(APPLE, subs)
    assert filing.accession == "0000320193-25-000079"
    assert filing.form == "10-K"


def test_latest_10k_raises_when_missing() -> None:
    subs = fake_submissions([("10-Q", "0000320193-26-000005", "2026-05-01")])
    with pytest.raises(LookupError, match="AAPL"):
        latest_10k(APPLE, subs)


def test_filing_url_and_local_name() -> None:
    subs = fake_submissions([("10-K", "0000320193-25-000079", "2025-10-31")])
    filing = latest_10k(APPLE, subs)
    assert filing.url == (
        "https://www.sec.gov/Archives/edgar/data/320193/"
        "000032019325000079/doc-0000320193-25-000079.htm"
    )
    assert filing.local_name == "AAPL_0000320193-25-000079.htm"


@pytest.mark.network
def test_all_10ks_on_disk_and_second_run_downloads_nothing() -> None:
    try:
        settings = Settings()
    except ValidationError:
        pytest.skip("SEC_USER_AGENT not set in .env")
    download_all(settings)
    second = download_all(settings)
    assert len(second) == len(COMPANIES)
    assert all(was_cached for _, was_cached in second)
    for filing, _ in second:
        path = settings.raw_dir / "10k" / filing.local_name
        assert path.stat().st_size > 100_000, f"{filing.ticker} 10-K looks too small"
