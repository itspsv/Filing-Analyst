from pathlib import Path

import duckdb
import pytest

from filing_analyst.tools.query_financials import (
    QueryFailed,
    SQLRejected,
    check_sql,
    query_financials,
)


@pytest.fixture
def db(tmp_path: Path) -> Path:
    """A tiny facts table (2,000 rows) plus a `secrets` table the agent must never see."""
    path = tmp_path / "facts.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute(
            "CREATE TABLE facts AS SELECT 'NVDA' AS ticker, 'revenue' AS label, "
            "2000 + (i % 30) AS fiscal_year, DATE '2026-01-25' AS period_end, "
            "CAST(i AS DOUBLE) AS value FROM range(2000) t(i)"
        )
        con.execute("CREATE TABLE secrets AS SELECT 'api-key-123' AS key")
    return path


@pytest.mark.parametrize(
    ("sql", "reason"),
    [
        ("DROP TABLE facts", "only SELECT"),
        ("UPDATE facts SET value = 0", "only SELECT"),
        ("DELETE FROM facts", "only SELECT"),
        ("INSERT INTO facts VALUES ('X', 'revenue', 2026, DATE '2026-01-01', 1)", "only SELECT"),
        ("SELECT 1; DROP TABLE facts", "exactly one statement"),
        ("SELECT * FROM secrets", "unknown table"),
        ("SELECT * FROM facts JOIN secrets ON true", "unknown table"),
        ("SELECT * FROM read_csv('/etc/passwd')", "only plain table names"),
        ("SELECT * FROM '/etc/passwd'", "unknown table"),
        ("SELECT * FROM main.secrets", "schema-qualified"),
        ("ATTACH 'other.db'", "only SELECT"),
        ("COPY facts TO 'out.csv'", "only SELECT"),
        ("PRAGMA database_list", "only SELECT"),
        ("SELECT getenv('HOME')", "function getenv"),
        ("SELECT * FROM facts LIMIT (SELECT 5)", "LIMIT must be a plain number"),
        ("SELEC * FROM facts", "could not parse"),
    ],
)
def test_unsafe_sql_is_rejected_before_it_runs(db: Path, sql: str, reason: str) -> None:
    with pytest.raises(SQLRejected, match=reason):
        query_financials(sql, db)


def test_normal_select_with_aggregates_and_cte_works(db: Path) -> None:
    sql = (
        "WITH yearly AS (SELECT fiscal_year, SUM(value) AS total FROM facts GROUP BY fiscal_year) "
        "SELECT fiscal_year, ROUND(total, 0) AS total FROM yearly ORDER BY fiscal_year LIMIT 3"
    )
    result = query_financials(sql, db)
    assert result.columns == ["fiscal_year", "total"]
    assert [row[0] for row in result.rows] == [2000, 2001, 2002]
    assert result.truncated is False
    assert result.query_id.startswith("sql-")


def test_dates_come_back_as_text(db: Path) -> None:
    result = query_financials("SELECT period_end FROM facts LIMIT 1", db)
    assert result.rows == [["2026-01-25"]]


def test_row_limit_is_added_and_truncation_reported(db: Path) -> None:
    result = query_financials("SELECT value FROM facts", db, max_rows=50)
    assert len(result.rows) == 50 and result.truncated is True
    assert "LIMIT 51" in result.sql  # one extra row tells us more exist


def test_a_slow_query_is_stopped_by_the_timeout(db: Path) -> None:
    slow = "SELECT COUNT(*) FROM facts a, facts b, facts c"  # 8 billion row pairs
    with pytest.raises(QueryFailed, match="longer than"):
        query_financials(slow, db, timeout_seconds=0.3)


def test_a_bad_column_is_a_fixable_error_not_a_rejection(db: Path) -> None:
    with pytest.raises(QueryFailed, match="revenue_usd"):
        query_financials("SELECT revenue_usd FROM facts", db)


def test_database_is_opened_read_only_even_if_a_write_slipped_through(db: Path) -> None:
    """Second layer: bypass the parser check and DuckDB itself must still refuse writes."""
    with duckdb.connect(str(db), read_only=True, config={"enable_external_access": False}) as con:
        with pytest.raises(duckdb.Error):
            con.execute("DELETE FROM facts")
        with pytest.raises(duckdb.Error):
            con.execute("SELECT * FROM read_csv('/etc/passwd')")


def test_cte_name_is_allowed_but_its_body_is_still_checked() -> None:
    check_sql("WITH t AS (SELECT * FROM facts) SELECT * FROM t")
    with pytest.raises(SQLRejected, match="unknown table"):
        check_sql("WITH t AS (SELECT * FROM secrets) SELECT * FROM t")
