from datetime import date
from pathlib import Path

import duckdb
import pytest
from pydantic import ValidationError

from filing_analyst.config import Settings
from filing_analyst.db import write_facts
from filing_analyst.sec.tickers import Company
from filing_analyst.sec.xbrl import annual_facts, coverage_gaps

ACME = Company(ticker="ACME", cik="0000000001", name="Acme Corp")


def raw(val: float, end: str, filed: str, start: str | None, form: str = "10-K") -> dict:
    fact = {"val": val, "end": end, "filed": filed, "form": form, "accn": f"acc-{filed}"}
    if start is not None:
        fact["start"] = start
    return fact


FAKE_COMPANYFACTS = {
    "facts": {
        "us-gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        raw(100, "2024-09-28", "2024-11-01", "2023-10-01"),
                        raw(105, "2024-09-28", "2025-11-01", "2023-10-01"),  # restated later
                        raw(30, "2024-09-28", "2024-11-01", "2024-06-30"),  # a quarter
                        raw(28, "2024-06-29", "2024-08-01", "2023-07-01", form="10-Q"),
                        raw(200, "2026-01-31", "2026-03-01", "2025-02-01"),  # Salesforce-style
                    ]
                }
            },
            "SalesRevenueNet": {
                "units": {
                    "USD": [
                        raw(80, "2015-09-26", "2015-11-01", "2014-09-28"),  # only old name
                        raw(999, "2024-09-28", "2024-11-01", "2023-10-01"),  # loses to Revenues
                    ]
                }
            },
            "Assets": {"units": {"USD": [raw(500, "2024-09-28", "2024-11-01", None)]}},
        }
    }
}


def revenue_by_year() -> dict[int, tuple[float, str]]:
    facts = annual_facts(ACME, FAKE_COMPANYFACTS)
    return {f.fiscal_year: (f.value, f.concept) for f in facts if f.label == "revenue"}


def test_first_concept_wins_and_latest_filed_wins() -> None:
    assert revenue_by_year()[2024] == (105, "Revenues")


def test_falls_back_to_older_concept_name() -> None:
    assert revenue_by_year()[2015] == (80, "SalesRevenueNet")


def test_quarters_and_10q_are_dropped() -> None:
    assert sorted(revenue_by_year()) == [2015, 2024, 2026]


def test_fiscal_year_comes_from_period_end() -> None:
    assert revenue_by_year()[2026] == (200, "Revenues")


def test_balance_sheet_value_is_a_snapshot() -> None:
    facts = annual_facts(ACME, FAKE_COMPANYFACTS)
    assets = [f for f in facts if f.label == "total_assets"]
    assert len(assets) == 1
    assert assets[0].period_start is None
    assert assets[0].value == 500


def test_coverage_gaps_lists_missing_metrics_for_latest_year() -> None:
    gaps = coverage_gaps(annual_facts(ACME, FAKE_COMPANYFACTS))
    assert "ACME FY2026: no net_income" in gaps
    assert "ACME FY2026: no revenue" not in gaps


def test_duckdb_round_trip(tmp_path: Path) -> None:
    db = tmp_path / "facts.duckdb"
    assert write_facts(annual_facts(ACME, FAKE_COMPANYFACTS), db) == 4
    with duckdb.connect(str(db), read_only=True) as con:
        row = con.execute(
            "SELECT value, period_end FROM facts WHERE label = 'revenue' AND fiscal_year = 2024"
        ).fetchone()
    assert row == (105.0, date(2024, 9, 28))


@pytest.mark.data
def test_real_apple_revenue_by_fiscal_year() -> None:
    try:
        db = Settings().facts_db
    except ValidationError:
        pytest.skip("SEC_USER_AGENT not set in .env")
    if not db.exists():
        pytest.skip("facts.duckdb missing; run python -m filing_analyst.db first")
    with duckdb.connect(str(db), read_only=True) as con:
        rows = con.execute(
            "SELECT fiscal_year, value FROM facts "
            "WHERE ticker = 'AAPL' AND label = 'revenue' ORDER BY fiscal_year"
        ).fetchall()
    assert len(rows) >= 3
    assert all(value > 0 for _, value in rows)
