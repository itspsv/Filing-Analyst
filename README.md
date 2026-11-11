# Filing Analyst

An AI research assistant that answers questions about 20 large US tech companies using their official SEC 10-K filings, and shows the filing passage or database row behind every claim. Built as a portfolio project for data and AI engineering roles.

> **Status: v0, Week 1 of 4.** The data pipeline works end to end. Search, cited answers, the agent, evals and the live demo come in Weeks 2 to 4. Every number below comes from a real run log, not an estimate, unless it says otherwise.

## What works today

One command builds all the data:

```bash
uv run python -m filing_analyst.pipeline            # all 5 steps
uv run python -m filing_analyst.pipeline --no-llm   # data steps only, no LLM tokens
```

```
1. tickers:  20/20 companies mapped to CIKs
2. filings:  20 10-Ks (0 downloaded, 20 cached)
3. sections: 40/40 sections found
4. facts:    2,914 rows in DuckDB, 12 known gaps in latest year
5. risks:    LLM risk extraction (in progress, see below)
```

## Architecture

Two data lanes that only meet at the agent (Weeks 2 and 3).

```
SEC EDGAR
  |-- text lane:    10-K HTML -> Item 1A + Item 7 -> risks (LLM)  [built]
  |                                              -> chunks -> search index  [Week 2]
  |-- numbers lane: XBRL company facts -> typed rows -> DuckDB  [built]

Agent [Week 3]: search_filings, query_financials (guarded SQL), compute_ratio (plain Python)
```

- `sec/`: ticker to CIK, latest 10-K per company, XBRL facts. Polite to the SEC: a User-Agent, 0.2 s between requests, and a disk cache so nothing is downloaded twice.
- `parse/`: HTML to clean text, then Item 1A (risk factors) and Item 7 (management's discussion).
- `db.py`: XBRL facts into DuckDB, one row per company, metric and period.
- `llm.py`: the one client for every LLM call (rate limiter, daily token budget, retries with backoff and jitter, disk cache, Pydantic validation with re-asks, one log line per call).
- `extract.py`: Item 1A into a list of risks (category, one-line summary, severity 1 to 5).

## Results so far

| What | Result |
|---|---|
| Companies | 20 (AAPL, MSFT, GOOGL, AMZN, META, NVDA, AMD, AMAT, TSLA, NFLX, CRM, ORCL, ADBE, QCOM, AVGO, TXN, MU, NOW, CSCO, PYPL) |
| 10-K sections found | 40/40 (Item 1A and Item 7 for every company) |
| Financial facts in DuckDB | 2,914 rows, 9 metrics |
| Data pipeline runtime (`--no-llm`, cached) | 19.1 s |
| Risk extraction | 5/20 companies done (AAPL, MSFT, NVDA, TXN, AMZN); rest in progress |
| First-try valid LLM output, full run | 31/34 calls (91%)* |
| LLM latency, full run (31 calls) | median 2,030 ms, p95 3,105 ms |
| LLM tokens used so far (logged calls) | 184,501 |
| Total cost so far | 0.00 EUR (Groq free tier) |
| Tests | 58 passing |

\* The 3 failures were Groq schema rejections (HTTP 400) on Nov 10, before failed calls were written to the run log; they are counted from that run's console output. Since the fix, every failed call is logged and re-asked.

Check, by hand: Apple revenue in DuckDB for FY2022, FY2023 and FY2024 is $394,328M, $383,285M and $391,035M, matching Apple's reported figures.

## Design decisions and tradeoffs

- **Two lanes: text for words, SQL for numbers.** A number in an answer must come from an exact database row, never from model memory or a text summary. XBRL gives official, machine-readable values that can be checked.
- **DuckDB.** In-process, no server, fast SQL on a laptop. Enough for a few thousand rows; no Docker during development.
- **Mapping table with fallbacks for XBRL names.** Companies tag the same number differently and change names over time (Apple's revenue uses three different tags between 2007 and 2026; Broadcom's FY2025 net income is only under `ProfitLoss`). Each row stores the tag it came from, so every number can be traced.
- **Fiscal year from the period end date.** The SEC's own fiscal-year field is wrong for some filers (Salesforce's year ending Jan 31, 2026 is labelled 2025).
- **Swapped Intel and IBM for Applied Materials and ServiceNow.** Intel's 10-K is not organised by Item number, and IBM's Item 7 only points to a separate annual-report exhibit. I measured the failures, tested replacements with the same section finder, and swapped instead of writing two special cases.
- **Cache everything.** Downloads and LLM answers are cached on disk, so every step can be rerun safely and a stopped run resumes for free. This matters on a free tier with 200K tokens per day.
- **Strict JSON schema plus our own validation.** Groq's strict mode checks the model's JSON against the schema and rejects bad output with an error; it does not stop the model from producing it. So the client re-asks on both kinds of failure (Groq rejection and Pydantic rejection), and every failure is logged.
- **Filing text is treated as untrusted.** It only goes in the user message, inside tags, and the system prompt says never to follow instructions inside it.

## Limitations (known, measured)

- **Risk categories and severity are unstable on borderline cases.** Two runs on the same Amazon text put 10 and then 6 risks in "operations". Severity is the model's judgement, not a verified fact. Risk lists are an overview; exact questions will be answered from the filing text (Week 2).
- **Some metrics are not reported by some companies.** In each company's latest year, 12 metric gaps: no gross profit for GOOGL, AMZN, META, NFLX, ORCL, QCOM, PYPL; no total liabilities for AMZN, AMD, ORCL; no cost of revenue for ORCL, PYPL. The agent must say "not reported" rather than guess.
- **Only the latest 10-K per company** is parsed for text; the numbers lane covers many years.
- **Page footers** (for example "Apple Inc. | 2026 Form 10-K | 19") are still in the section text.

## How to run

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/itspsv/Filing-Analyst.git
cd Filing-Analyst
cp .env.example .env    # add your name and email (SEC requirement) and a Groq API key
uv sync
uv run pytest
uv run python -m filing_analyst.pipeline --no-llm
```

## Roadmap

- **Week 2:** chunking, BM25 + dense search + Reciprocal Rank Fusion (first from scratch in NumPy, then Qdrant), a 40-question retrieval set, cited answers, refusals, `POST /ask`.
- **Week 3:** agent with three tools, guarded text-to-SQL (sqlglot, SELECT only, read-only), prompt-injection test, MCP server.
- **Week 4:** 50-question eval suite with XBRL ground truth in CI, LLM judge calibrated against human labels, tracing, cost per question, p95 latency, live demo.
