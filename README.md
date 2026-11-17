# Filing Analyst

An AI research assistant that answers questions about 20 large US tech companies using their official SEC 10-K filings, and shows the filing passage or database row behind every claim. Built as a portfolio project for data and AI engineering roles.

> **Status: Week 3 of 4.** Data pipeline, search, cited answers, refusals and `POST /ask` are built and measured. The agent (safe SQL over the numbers, plain-Python ratios), evals in CI and the live demo come next. Every number below comes from a run log or an eval report in this repo, not an estimate, unless it says otherwise.

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

Ask a question and get a cited answer, from the command line or over HTTP:

```bash
uv run python -m filing_analyst.answer "What share of Broadcom's wafers come from TSMC?" --ticker AVGO
uv run uvicorn filing_analyst.api:app --port 8000   # then POST /ask {"question": ..., "ticker": ...}
```

## Architecture

Two data lanes that only meet at the agent (Week 3).

```
SEC EDGAR
  |-- text lane:    10-K HTML -> Item 1A + Item 7 -> risks (LLM)                       [built]
  |                                              -> chunks -> BM25 + dense (Qdrant)
  |                                                 -> RRF fusion -> cross-encoder     [built]
  |-- numbers lane: XBRL company facts -> typed rows -> DuckDB                         [built]

Cited answers (citations checked by code) + refusals + POST /ask                       [built]
Agent: search_filings, query_financials (guarded SQL), compute_ratio (plain Python)    [Week 3]
```

- `sec/`: ticker to CIK, latest 10-K per company, XBRL facts. Polite to the SEC: a User-Agent, 0.2 s between requests, and a disk cache so nothing is downloaded twice.
- `parse/`: HTML to clean text, Item 1A (risk factors) and Item 7 (management's discussion), and three chunking strategies.
- `db.py`: XBRL facts into DuckDB, one row per company, metric and period.
- `llm.py`: the one client for every LLM call (rate limiter, per-model daily token budget, retries with backoff and jitter, disk cache, Pydantic validation with re-asks, one log line per call).
- `extract.py`: Item 1A into a list of risks (category, one-line summary, severity 1 to 5).
- `retrieval/`: BM25 and cosine search from scratch, Reciprocal Rank Fusion, Qdrant (local mode) with company, year and section filters, cross-encoder re-ranking.
- `answer.py`: answers from the top 5 retrieved chunks; every citation's chunk and quote are checked in code; no valid citation means a refusal. `api.py` serves it as `POST /ask`.
- `tools/compute_ratio.py`: margins and growth in plain Python, with the formula and inputs, so the LLM never does arithmetic.
- `evaluation/`: the retrieval test set (LLM-drafted, human-reviewed), the retrieval eval, and the answer and refusal eval.

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

### Cited answers: error analysis, before and after (`gpt-oss-120b`)

The answer model gets the top 5 chunks from the winning search (with the company filter) and must give, for every claim, a chunk id and a word-for-word quote. Code drops a citation if its chunk was not retrieved or its quote is not in that chunk; an answer with no valid citation becomes a refusal ("Not enough evidence in the filings."). **Grounded** means the answer cites a chunk that contains the question's labelled quote: a strict code check, no LLM judge.

| Round | Change | Grounded | Refused | Failed | Citation validity |
|---|---|---|---|---|---|
| 0 | first run: exact quote match, quotes up to 400 characters | 11/24* | 10/24* | run crashed | 43.6% (24/55)* |
| 1 | accept quotes shortened with "..." if every piece matches | 31/39 | 0 | 3 | 89.7% (70/78) |
| 2 | quote limit 400 to 1,000 characters (answers regenerated) | 32/39 | 1 | 0 | 77.2% (71/92) |
| 3 | ignore a full stop added at the end of a quote (same answers as round 2) | **32/39 (82.1%)** | 1 | **0** | **84.8% (78/92)** |

\* Round 0 stopped after 24 questions. Round 1 re-judged those **same 24 cached answers**: grounded 11 to 20, refused 10 to 0, citation validity 43.6% to 90.9%, so that jump comes from the checker fix alone.

**How the fixes were found.** In round 0, 30 of 31 rejected citations were real quotes the model had shortened with "..." (1 was paraphrased). The 3 failures in round 1 were quotes of 436, 471 and 589 characters against a 400 limit, which also wasted ~59,000 estimated tokens on rejected answers. In round 2, 7 of 21 rejected citations only had a full stop added where the model cut a sentence short; the other 13 changed words and 1 cited the wrong chunk, and those are still rejected.

| Other answer numbers (round 2, from the run log) | Result |
|---|---|
| Unanswerable questions refused | **5/5** (4 by the model; Intel by the "company not covered" rule) |
| Labelled passage among the 5 shown to the model | 37/39 |
| LLM calls valid on the first try | 43/43 |
| Tokens per question | ~2,400 |
| LLM latency per answer (API time) | p50 934 ms, p95 2,252 ms |
| Cost | 0.00 EUR |

"Not grounded" is not the same as wrong: the label is one passage, and 4 of the 6 "not grounded" answers cite the chunk right next to the labelled one, which often continues the same text. Whether those answers are correct is not checked yet; a faithfulness judge calibrated against human labels comes in Week 4.

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
| Tests | 156 passing |

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
- **Citations are verified by code, not trusted.** A citation must point to a chunk that was retrieved, and its quote (each piece, if shortened with "...") must be in that chunk. An answer with no surviving citation becomes a refusal: on `gpt-oss-20b` this caught an attempted answer to "What is the favourite food of Apple's CEO?".
- **Refuse before searching a company we do not cover.** Otherwise another company's text about it (AMD writing about Intel) could answer the question.
- **One change per round, measured on the same answers where possible.** Cached answers let a checker change be judged on identical model output, so before/after numbers show the effect of that change alone.

## Limitations (known, measured)

- **Risk categories and severity are unstable on borderline cases.** Two runs on the same Amazon text put 10 and then 6 risks in "operations". Severity is the model's judgement, not a verified fact. Risk lists are an overview; exact questions are answered from the filing text.
- **Broad questions have many right passages.** The test set labels one or a few quotes per question, so a search that finds a different, also-correct passage counts as a miss (for example "AMD export risks", where several AMD passages qualify).
- **Some metrics are not reported by some companies.** In each company's latest year, 12 metric gaps: no gross profit for GOOGL, AMZN, META, NFLX, ORCL, QCOM, PYPL; no total liabilities for AMZN, AMD, ORCL; no cost of revenue for ORCL, PYPL. The agent must say "not reported" rather than guess.
- **Only the latest 10-K per company** is parsed for text; the numbers lane covers many years.
- **The citation check is strict on purpose.** A correct answer whose only quote has a small word change is refused (q036, Broadcom). Answers are regenerated when the request changes, so a few questions flip between rounds for reasons other than the fix (3 of 39 between rounds 1 and 2).
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
uv run python -m filing_analyst.evaluation.answer_eval                 # needs a Groq key
uv run python -m filing_analyst.evaluation.answer_eval --unanswerable
```

## Roadmap

- **Week 3:** guarded text-to-SQL (sqlglot, SELECT only, read-only), agent with three tools (search, SQL, ratios), answers citing both chunks and SQL rows, prompt-injection test, MCP server.
- **Week 4:** 50-question eval suite with XBRL ground truth in CI, LLM judge calibrated against human labels, tracing, cost per question, p95 latency, live demo.
