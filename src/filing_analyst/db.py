"""DuckDB storage for financial facts. `python -m filing_analyst.db` loads all companies."""

from pathlib import Path

import duckdb

from filing_analyst.config import Settings
from filing_analyst.sec.xbrl import Fact, coverage_gaps, load_all_facts

CREATE_FACTS = """
CREATE OR REPLACE TABLE facts (
    ticker       VARCHAR NOT NULL,
    cik          VARCHAR NOT NULL,
    company_name VARCHAR NOT NULL,
    concept      VARCHAR NOT NULL,
    label        VARCHAR NOT NULL,
    period_start DATE,
    period_end   DATE NOT NULL,
    fiscal_year  INTEGER NOT NULL,
    value        DOUBLE NOT NULL,
    unit         VARCHAR NOT NULL,
    form         VARCHAR NOT NULL,
    filed_date   DATE NOT NULL,
    accession    VARCHAR NOT NULL,
    PRIMARY KEY (ticker, label, period_end)
)
"""


def write_facts(facts: list[Fact], db_path: Path) -> int:
    """Replace the facts table with these rows. Returns the row count in the table."""
    columns = list(Fact.model_fields)
    insert = f"INSERT INTO facts ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as con:
        con.execute(CREATE_FACTS)
        con.executemany(insert, [[getattr(f, c) for c in columns] for f in facts])
        return con.execute("SELECT count(*) FROM facts").fetchone()[0]


def main() -> None:
    settings = Settings()
    facts = load_all_facts(settings)
    for ticker in dict.fromkeys(f.ticker for f in facts):
        years = [f.fiscal_year for f in facts if f.ticker == ticker]
        print(f"{ticker:<6} {len(years):>4} rows  FY{min(years)}-FY{max(years)}")
    rows = write_facts(facts, settings.facts_db)
    print(f"\n{rows} rows written to {settings.facts_db.name}")
    gaps = coverage_gaps(facts)
    print(f"\ngaps in latest fiscal year ({len(gaps)}):")
    for gap in gaps:
        print(f"  {gap}")


if __name__ == "__main__":
    main()
