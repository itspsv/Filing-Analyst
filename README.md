# Filing Analyst

An AI research assistant that answers questions about 20 large US tech companies using their official SEC 10-K filings, and shows the filing passage or database row behind every claim. Built as a portfolio project for data and AI engineering roles.

> **Status: Week 2 of 4.** The data pipeline and the search layer are built and measured. Cited answers, refusals and `POST /ask` come next; then the agent, evals and the live demo. Every number below comes from a run log or an eval report in this repo, not an estimate, unless it says otherwise.

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
5. risks:    20/20 companies done, 0 failed
```

Search over the filings, with the best setup from the experiment grid below:

```bash
uv run python -m filing_analyst.retrieval.rerank "Which companies depend on TSMC?" --ticker AVGO
```

## Architecture

Two data lanes that only meet at the agent (Week 3).

```
SEC EDGAR
  |-- text lane:    10-K HTML -> Item 1A + Item 7 -> risks (LLM)                       [built]
  |                                              -> chunks -> BM25 + dense (Qdrant)
  |                                                 -> RRF fusion -> cross-encoder     [built]
  |-- numbers lane: XBRL company facts -> typed rows -> DuckDB                         [built]

Cited answers + refusals + POST /ask                                                   [next]
Agent: search_filings, query_financials (guarded SQL), compute_ratio (plain Python)    [Week 3]
```

- `sec/`: ticker to CIK, latest 10-K per company, XBRL facts. Polite to the SEC: a User-Agent, 0.2 s between requests, and a disk cache so nothing is downloaded twice.
- `parse/`: HTML to clean text, Item 1A (risk factors) and Item 7 (management's discussion), and three chunking strategies.
- `db.py`: XBRL facts into DuckDB, one row per company, metric and period.
- `llm.py`: the one client for every LLM call (rate limiter, per-model daily token budget, retries with backoff and jitter, disk cache, Pydantic validation with re-asks, one log line per call).
- `extract.py`: Item 1A into a list of risks (category, one-line summary, severity 1 to 5).
- `retrieval/`: BM25 and cosine search from scratch, Reciprocal Rank Fusion, Qdrant (local mode) with company, year and section filters, cross-encoder re-ranking.
- `evaluation/`: the retrieval test set (LLM-drafted, human-reviewed) and the retrieval eval.

## Results

### Retrieval: 3 chunking strategies x 4 search methods

39 questions over the 20 filings, drafted by an LLM and **reviewed by hand** (11 of 50 drafts dropped; 20 literal, 19 paraphrased). Each question is labelled with a short quote from the filing; a retrieved chunk counts as correct if it is from the right company and contains the quote. Scores are **recall@5** (right passage in the top 5) / **MRR@10** (how high the first right passage ranks).

**With company filter** (how the agent will search: `search_filings(query, company)`):

| Chunking | dense | BM25 | hybrid (RRF) | hybrid + re-rank |
|---|---|---|---|---|
| fixed (300 tokens, 50 overlap) | 0.897 / 0.734 | 0.923 / 0.760 | 0.949 / 0.842 | **0.949 / 0.880** |
| recursive (paragraph, sentence, word) | 0.821 / 0.645 | 0.872 / 0.733 | 0.846 / 0.731 | **1.000** / 0.832 |
| section-aware (heading + paragraphs) | 0.821 / 0.681 | 0.974 / 0.771 | 0.949 / 0.767 | 0.949 / 0.846 |

**Without filter** (all 20 companies; harder):

| Chunking | dense | BM25 | hybrid (RRF) | hybrid + re-rank |
|---|---|---|---|---|
| fixed | 0.769 / 0.567 | 0.769 / 0.658 | 0.795 / 0.713 | **0.872 / 0.737** |
| recursive | 0.667 / 0.536 | 0.744 / 0.579 | 0.718 / 0.639 | 0.821 / 0.702 |
| section-aware | 0.692 / 0.544 | 0.795 / 0.634 | 0.795 / 0.628 | 0.795 / 0.703 |

**Search latency** (fixed chunks, company filter, 39 questions x 5 passes = 195 searches each, M1 laptop CPU):

| Method | p50 | p95 |
|---|---|---|
| BM25 | 0.2 ms | 0.2 ms |
| dense | 11.2 ms | 13.2 ms |
| hybrid (RRF) | 11.7 ms | 13.9 ms |
| hybrid + re-rank | 356.8 ms | 407.8 ms |

**Why the winner won: fixed chunks + hybrid + cross-encoder re-rank.** It has the best ranking in both settings (MRR 0.880 with the filter, 0.737 without) and the best recall without the filter (0.872). Keyword search (BM25) is strongest on literal questions (0.850 recall without filter) and meaning search (dense) on paraphrased ones (0.842), so fusing them with RRF beats either alone; the re-ranker then reads question and passage together and pushes the best passage up (MRR 0.842 to 0.880). Fixed chunks look the worst (74% start mid-sentence, against 4% for the other two strategies), but their 50-token overlap keeps facts near a cut in at least one chunk, and that measured better. Recursive + re-rank found all 39 passages with the filter, but ranked them lower (MRR 0.832). The re-ranker costs ~0.35 s per search, small next to an LLM answer.

**Caveats.** 39 questions is a small set: one question moves recall by 2.6 points, so only differences of several questions are meaningful (fixed and section-aware hybrid tie at 37/39). The questions were drafted from fixed-size chunks, which may favour that strategy. Knowing the company matters most: with the filter, hybrid misses drop from 8 to 2.

### Data pipeline and LLM extraction

| What | Result |
|---|---|
| Companies | 20 (AAPL, MSFT, GOOGL, AMZN, META, NVDA, AMD, AMAT, TSLA, NFLX, CRM, ORCL, ADBE, QCOM, AVGO, TXN, MU, NOW, CSCO, PYPL) |
| 10-K sections found | 40/40 (Item 1A and Item 7 for every company) |
| Financial facts in DuckDB | 2,914 rows, 9 metrics |
| Data pipeline runtime (`--no-llm`, cached) | 19.1 s |
| Risks extracted | 1,253 risks for 20/20 companies, 0 failed |
| **First-try valid LLM output** (production runs, Nov 10 to 15) | **121/128 calls (94.5%)*** |
| LLM tokens, extraction (production runs) | 455,501 logged + ~24,000 estimated for answers Groq rejected |
| LLM latency, extraction (125 logged calls) | median 1,854 ms, p95 4,709 ms |
| **Total cost** | **0.00 EUR** (Groq free tier, `openai/gpt-oss-120b`) |
| Tests | 119 passing |

\* 125 calls are in the run log; 3 more failed on Nov 10 (Groq schema rejections) before failed calls were written to the log, and are counted from that run's console output. After the fix (re-ask on rejection), runs went from 3 failed companies to 0, with 44/47 and 46/47 first-try valid.

Check, by hand: Apple revenue in DuckDB for FY2022, FY2023 and FY2024 is $394,328M, $383,285M and $391,035M, matching Apple's reported figures.

## Design decisions and tradeoffs

- **Two lanes: text for words, SQL for numbers.** A number in an answer must come from an exact database row, never from model memory or a text summary. XBRL gives official, machine-readable values that can be checked.
- **Hybrid search, then re-rank.** BM25 finds exact names and codes ("TSMC", "H200"); dense search finds the same meaning in other words. RRF fuses them by rank, because their scores live on different scales. A cross-encoder is too slow for all 2,107 chunks, so it only re-orders the top 20.
- **Search built from scratch first, then Qdrant.** NumPy cosine, BM25 with an inverted index, and RRF are under 200 lines of core logic and unit tested. Qdrant (local mode, no server) then returned the identical top 10 for 39/39 questions, with and without filters.
- **Labels are quotes, not chunk ids.** The same test set can score any chunking strategy, and code checks that every quote exists word for word in the filing (the drafting LLM invented 3 quotes; they were dropped automatically).
- **DuckDB.** In-process, no server, fast SQL on a laptop. Enough for a few thousand rows; no Docker during development.
- **Mapping table with fallbacks for XBRL names.** Companies tag the same number differently and change names over time (Apple's revenue uses three different tags between 2007 and 2026; Broadcom's FY2025 net income is only under `ProfitLoss`). Each row stores the tag it came from, so every number can be traced.
- **Fiscal year from the period end date.** The SEC's own fiscal-year field is wrong for some filers (Salesforce's year ending Jan 31, 2026 is labelled 2025).
- **Swapped Intel and IBM for Applied Materials and ServiceNow.** Intel's 10-K is not organised by Item number, and IBM's Item 7 only points to a separate annual-report exhibit. I measured the failures, tested replacements with the same section finder, and swapped instead of writing two special cases.
- **Cache everything.** Downloads, embeddings and LLM answers are cached on disk, so every step can be rerun safely and a stopped run resumes for free. This matters on a free tier with 200K tokens per day per model.
- **Strict JSON schema plus our own validation.** Groq's strict mode checks the model's JSON against the schema and rejects bad output with an error; it does not stop the model from producing it. So the client re-asks on both kinds of failure (Groq rejection and Pydantic rejection), and every failure is logged.
- **Filing text is treated as untrusted.** It only goes in the user message, inside tags, and the system prompt says never to follow instructions inside it.

## Limitations (known, measured)

- **Risk categories and severity are unstable on borderline cases.** Two runs on the same Amazon text put 10 and then 6 risks in "operations". Severity is the model's judgement, not a verified fact. Risk lists are an overview; exact questions are answered from the filing text.
- **Broad questions have many right passages.** The test set labels one or a few quotes per question, so a search that finds a different, also-correct passage counts as a miss (for example "AMD export risks", where several AMD passages qualify).
- **Some metrics are not reported by some companies.** In each company's latest year, 12 metric gaps: no gross profit for GOOGL, AMZN, META, NFLX, ORCL, QCOM, PYPL; no total liabilities for AMZN, AMD, ORCL; no cost of revenue for ORCL, PYPL. The agent must say "not reported" rather than guess.
- **Only the latest 10-K per company** is parsed for text; the numbers lane covers many years.
- **Re-ranker workaround.** On the development Mac, the cross-encoder returned NaN scores when its weights were read straight from the memory-mapped file; it now copies them into memory once (~90 MB), and a test guards it.

## How to run

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/itspsv/Filing-Analyst.git
cd Filing-Analyst
cp .env.example .env    # add your name and email (SEC requirement) and a Groq API key
uv sync
uv run pytest
uv run python -m filing_analyst.pipeline --no-llm
uv run python -m filing_analyst.parse.chunk --strategy fixed
uv run python -m filing_analyst.evaluation.retrieval_eval --chunking fixed
```

## Roadmap

- **Week 2 (rest):** cited answers that use full passages (several for list questions), refusals when evidence is missing, `POST /ask`.
- **Week 3:** agent with three tools, guarded text-to-SQL (sqlglot, SELECT only, read-only), prompt-injection test, MCP server.
- **Week 4:** 50-question eval suite with XBRL ground truth in CI, LLM judge calibrated against human labels, tracing, cost per question, p95 latency, live demo.
