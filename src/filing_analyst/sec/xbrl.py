"""Yearly financial numbers from SEC XBRL company facts, one row per company, metric and period."""

import json
from datetime import date

from pydantic import BaseModel

from filing_analyst.config import COMPANIES, Settings
from filing_analyst.sec.http import fetch_cached
from filing_analyst.sec.tickers import Company, load_ticker_table, lookup

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# Our metric name -> us-gaap concepts to try, in order. Companies use different names for the
# same number and switch names over the years, so for each period the first name found wins.
METRICS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
    ),
    "gross_profit": ("GrossProfit",),
    "cost_of_revenue": ("CostOfRevenue", "CostOfGoodsAndServicesSold"),
    "operating_income": ("OperatingIncomeLoss",),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),  # Broadcom FY2025 is only under ProfitLoss
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "total_assets": ("Assets",),
    "total_liabilities": ("Liabilities",),
    "stockholders_equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
}
# Balance-sheet numbers are a snapshot on one date; all other metrics cover a whole year.
SNAPSHOT_METRICS = {"total_assets", "total_liabilities", "stockholders_equity"}
YEAR_DAYS = range(350, 381)  # 52/53-week fiscal years land in here; quarters do not


class Fact(BaseModel):
    ticker: str
    cik: str
    company_name: str
    concept: str
    label: str
    period_start: date | None
    period_end: date
    fiscal_year: int
    value: float
    unit: str
    form: str
    filed_date: date
    accession: str


def _to_fact(company: Company, label: str, concept: str, raw: dict) -> Fact | None:
    """Turn one raw XBRL value into a Fact, or None if it is not a yearly 10-K value."""
    if raw.get("form") != "10-K":
        return None
    end = date.fromisoformat(raw["end"])
    start = date.fromisoformat(raw["start"]) if "start" in raw else None
    if label in SNAPSHOT_METRICS:
        if start is not None:
            return None
    elif start is None or (end - start).days not in YEAR_DAYS:
        return None
    return Fact(
        ticker=company.ticker,
        cik=company.cik,
        company_name=company.name,
        concept=concept,
        label=label,
        period_start=start,
        period_end=end,
        fiscal_year=end.year,  # the SEC "fy" field is unreliable (Salesforce FY2026 says 2025)
        value=raw["val"],
        unit="USD",
        form=raw["form"],
        filed_date=date.fromisoformat(raw["filed"]),
        accession=raw["accn"],
    )


def annual_facts(company: Company, companyfacts: dict) -> list[Fact]:
    """One Fact per (metric, period end): earliest concept in METRICS wins, then latest filed."""
    gaap = companyfacts["facts"].get("us-gaap", {})
    candidates: list[tuple[int, Fact]] = []
    for label, concepts in METRICS.items():
        for rank, concept in enumerate(concepts):
            for raw in gaap.get(concept, {}).get("units", {}).get("USD", []):
                fact = _to_fact(company, label, concept, raw)
                if fact is not None:
                    candidates.append((rank, fact))

    chosen: dict[tuple[str, date], Fact] = {}
    for _, fact in sorted(candidates, key=lambda c: (c[0], -c[1].filed_date.toordinal())):
        chosen.setdefault((fact.label, fact.period_end), fact)
    return sorted(chosen.values(), key=lambda f: (f.label, f.period_end))


def load_company_facts(company: Company, settings: Settings) -> dict:
    path = fetch_cached(
        COMPANYFACTS_URL.format(cik=company.cik),
        settings.raw_dir / "companyfacts" / f"CIK{company.cik}.json",
        settings.sec_user_agent,
    )
    return json.loads(path.read_text())


def coverage_gaps(facts: list[Fact]) -> list[str]:
    """Metrics a company does not report for its latest fiscal year."""
    gaps = []
    for ticker in dict.fromkeys(f.ticker for f in facts):
        own = [f for f in facts if f.ticker == ticker]
        latest = max(f.fiscal_year for f in own)
        have = {f.label for f in own if f.fiscal_year == latest}
        gaps += [f"{ticker} FY{latest}: no {label}" for label in METRICS if label not in have]
    return gaps


def load_all_facts(settings: Settings) -> list[Fact]:
    facts: list[Fact] = []
    for company in lookup(COMPANIES, load_ticker_table(settings)):
        facts += annual_facts(company, load_company_facts(company, settings))
    return facts
