"""Guarded SQL over the financial facts: the agent may only read the `facts` table.

Defence in layers, so one missed trick is not enough:
1. sqlglot parses the SQL before it runs: exactly one SELECT, only allowlisted tables,
   no schema-qualified names, no file readers, no unknown functions, no write statements.
2. DuckDB is opened read-only with external access off (no files, no network).
3. A row limit and a timeout.
"""

import hashlib
import threading
from datetime import date
from pathlib import Path

import duckdb
import sqlglot
from pydantic import BaseModel
from sqlglot import exp

ALLOWED_TABLES = {"facts"}
MAX_ROWS = 200
TIMEOUT_SECONDS = 5.0
FORBIDDEN_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Drop, exp.Create, exp.Alter,
    exp.TruncateTable, exp.Command, exp.Pragma, exp.Copy, exp.Attach, exp.Set, exp.Use,
)  # fmt: skip

SCHEMA_DESCRIPTION = """Table `facts`: one row per company, metric and fiscal period (SEC XBRL).
  ticker        VARCHAR  e.g. 'NVDA'
  company_name  VARCHAR
  label         VARCHAR  one of: revenue, gross_profit, cost_of_revenue, operating_income,
                         net_income, operating_cash_flow, total_assets, total_liabilities,
                         stockholders_equity
  fiscal_year   INTEGER  the year the fiscal period ends (NVIDIA FY2026 ends Jan 2026)
  period_start  DATE     NULL for balance-sheet labels (assets, liabilities, equity)
  period_end    DATE
  value         DOUBLE   US dollars
  concept       VARCHAR  the XBRL tag the value came from
  form, filed_date, accession, cik, unit: filing metadata"""


class SQLRejected(ValueError):
    """The query broke a safety rule; it was never run."""


class QueryFailed(RuntimeError):
    """The query was safe but failed to run (bad column, too slow); the agent can fix it."""


class QueryResult(BaseModel):
    query_id: str  # short hash of the SQL that ran; used to cite these rows
    sql: str
    columns: list[str]
    rows: list[list[str | int | float | None]]
    truncated: bool  # more rows existed than MAX_ROWS


def check_sql(sql: str) -> exp.Expression:
    """Parse and check the SQL; return the parsed query or raise SQLRejected with the reason."""
    try:
        statements = [s for s in sqlglot.parse(sql, dialect="duckdb") if s is not None]
    except sqlglot.errors.ParseError as err:
        raise SQLRejected(f"could not parse SQL: {str(err).splitlines()[0]}") from err
    if len(statements) != 1:
        raise SQLRejected(f"exactly one statement allowed, got {len(statements)}")
    tree = statements[0]
    if not isinstance(tree, exp.Select):
        raise SQLRejected(f"only SELECT is allowed, got {type(tree).__name__.upper()}")
    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            raise SQLRejected(f"{type(node).__name__.upper()} is not allowed")
        if isinstance(node, exp.Anonymous):  # a function sqlglot does not know: refuse it
            raise SQLRejected(f"function {node.name}() is not allowed")
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):  # read_csv(...), a file path, ...
            raise SQLRejected(f"only plain table names are allowed, got {table.sql('duckdb')}")
        if table.args.get("db") or table.args.get("catalog"):
            raise SQLRejected(f"schema-qualified names are not allowed: {table.sql('duckdb')}")
        if table.name.lower() not in ALLOWED_TABLES | cte_names:
            raise SQLRejected(f"unknown table {table.name!r}; allowed: {sorted(ALLOWED_TABLES)}")
    return tree


def with_row_limit(tree: exp.Expression, max_rows: int) -> exp.Expression:
    """Fetch at most max_rows + 1 rows, so we can tell whether the result was cut."""
    limit = tree.args.get("limit")
    if limit is None:
        return tree.limit(max_rows + 1)
    value = limit.expression
    if not (isinstance(value, exp.Literal) and value.is_int):
        raise SQLRejected("LIMIT must be a plain number")
    return tree if int(value.name) <= max_rows else tree.limit(max_rows + 1)


def _plain(value: object) -> str | int | float | None:
    """JSON-friendly cell: dates as ISO strings, anything unusual as text."""
    if isinstance(value, date):
        return value.isoformat()
    if value is None or isinstance(value, str | int | float):
        return value
    return str(value)


def query_financials(
    sql: str,
    db_path: Path,
    max_rows: int = MAX_ROWS,
    timeout_seconds: float = TIMEOUT_SECONDS,
) -> QueryResult:
    safe_sql = with_row_limit(check_sql(sql), max_rows).sql(dialect="duckdb")
    con = duckdb.connect(str(db_path), read_only=True, config={"enable_external_access": False})
    timer = threading.Timer(timeout_seconds, con.interrupt)
    timer.start()
    try:
        cursor = con.execute(safe_sql)
        rows = cursor.fetchmany(max_rows + 1)
        columns = [d[0] for d in cursor.description]
    except duckdb.InterruptException as err:
        raise QueryFailed(f"query took longer than {timeout_seconds:g} s and was stopped") from err
    except duckdb.Error as err:
        raise QueryFailed(str(err).splitlines()[0]) from err
    finally:
        timer.cancel()
        con.close()
    return QueryResult(
        query_id="sql-" + hashlib.sha256(safe_sql.encode()).hexdigest()[:10],
        sql=safe_sql,
        columns=columns,
        rows=[[_plain(v) for v in row] for row in rows[:max_rows]],
        truncated=len(rows) > max_rows,
    )
