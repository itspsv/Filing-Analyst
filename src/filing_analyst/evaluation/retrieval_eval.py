"""Score the search methods on the reviewed retrieval set.

A retrieved chunk is relevant if it belongs to the question's company and contains one of its
quotes. No LLM is involved, so a run is free and repeatable.

    recall@5 = share of a question's quotes found in the top 5 (one quote: hit or miss)
    MRR@10   = average of 1 / rank of the first relevant chunk in the top 10 (0 if none)

    uv run python -m filing_analyst.evaluation.retrieval_eval
"""

import argparse
import json
import math
import re
import statistics
import time
from collections.abc import Callable, Sequence
from typing import get_args

from pydantic import BaseModel

from filing_analyst.config import Settings
from filing_analyst.evaluation.retrieval_set import RetrievalItem, contains_quote
from filing_analyst.parse.chunk import Chunk, Strategy
from filing_analyst.retrieval.bm25 import BM25Index
from filing_analyst.retrieval.dense import embed_query, load_embedder, load_or_build_index
from filing_analyst.retrieval.rerank import RerankedSearcher, load_reranker
from filing_analyst.retrieval.rrf import HybridSearcher

RECALL_K = 5
MRR_K = 10
ITEM_MENTION = re.compile(r"\bitem\s*(1a|7)\b", re.IGNORECASE)

Search = Callable[[str, str | None], list[Chunk]]  # (question, ticker filter) -> best first

# "no filter": search all 20 companies (hardest). "company filter": search only the asked
# company, as the agent's search_filings(query, company) tool will.
SETTINGS = {"no filter": False, "company filter": True}


class QuestionResult(BaseModel):
    qid: str
    style: str
    names_item: bool
    recall: float  # share of quotes found in the top RECALL_K
    reciprocal_rank: float  # 1 / rank of the first relevant chunk, 0 if not in top MRR_K
    first_hit_rank: int | None
    latency_ms: float


def is_relevant(chunk: Chunk, item: RetrievalItem) -> bool:
    return chunk.ticker == item.ticker and any(contains_quote(chunk.text, q) for q in item.quotes)


def score(item: RetrievalItem, results: Sequence[Chunk], latency_ms: float) -> QuestionResult:
    top = results[:RECALL_K]
    found = sum(any(contains_quote(c.text, q) for c in top if c.ticker == item.ticker)
                for q in item.quotes)  # fmt: skip
    first = next((i for i, c in enumerate(results[:MRR_K], 1) if is_relevant(c, item)), None)
    return QuestionResult(
        qid=item.qid,
        style=item.style,
        names_item=bool(ITEM_MENTION.search(item.question)),
        recall=found / len(item.quotes),
        reciprocal_rank=1 / first if first else 0.0,
        first_hit_rank=first,
        latency_ms=latency_ms,
    )


def summarize(results: Sequence[QuestionResult]) -> dict[str, float]:
    latencies = sorted(r.latency_ms for r in results)
    return {
        "questions": len(results),
        "recall@5": round(statistics.mean(r.recall for r in results), 3),
        "mrr@10": round(statistics.mean(r.reciprocal_rank for r in results), 3),
        "latency_p50_ms": round(statistics.median(latencies), 1),
        "latency_p95_ms": round(latencies[math.ceil(0.95 * len(latencies)) - 1], 1),
    }


def groups(results: Sequence[QuestionResult]) -> dict[str, list[QuestionResult]]:
    named = {
        "all": list(results),
        "literal": [r for r in results if r.style == "literal"],
        "paraphrase": [r for r in results if r.style == "paraphrase"],
        "names an Item": [r for r in results if r.names_item],
    }
    return {name: rs for name, rs in named.items() if rs}


def evaluate(
    items: Sequence[RetrievalItem], search: Search, company_filter: bool = False
) -> list[QuestionResult]:
    results = []
    for item in items:
        started = time.perf_counter()
        found = search(item.question, item.ticker if company_filter else None)
        results.append(score(item, found, (time.perf_counter() - started) * 1000))
    return results


def unreachable(items: Sequence[RetrievalItem], chunks: Sequence[Chunk]) -> list[str]:
    """Questions whose quote sits in no single chunk (cut by a chunk boundary): no method
    can score on them with this chunking, so they are reported, not hidden."""
    return [i.qid for i in items if not any(is_relevant(c, i) for c in chunks)]


def load_items(settings: Settings) -> list[RetrievalItem]:
    path = settings.evals_dir / "retrieval_set.jsonl"
    return [RetrievalItem.model_validate_json(line) for line in path.read_text().splitlines()]


def build_searchers(
    settings: Settings, chunking: Strategy = "fixed"
) -> tuple[list[Chunk], dict[str, Search]]:
    embedder = load_embedder()
    dense = load_or_build_index(settings, name=chunking, make_embedder=lambda: embedder)
    bm25 = BM25Index(dense.chunks)
    hybrid = HybridSearcher(dense, bm25, embedder)
    reranked = RerankedSearcher(hybrid, load_reranker())
    searchers: dict[str, Search] = {
        "dense": lambda q, t: [
            c for c, _ in dense.search(embed_query(q, embedder), k=MRR_K, ticker=t)
        ],
        "bm25": lambda q, t: [c for c, _ in bm25.search(q, k=MRR_K, ticker=t)],
        "hybrid": lambda q, t: [r.chunk for r in hybrid.search(q, k=MRR_K, ticker=t)],
        "hybrid+rerank": lambda q, t: [c for c, _ in reranked.search(q, k=MRR_K, ticker=t)],
    }
    return dense.chunks, searchers


def print_table(setting: str, methods: dict) -> None:
    print(f"\n== {setting} ==")
    print(f"{'method':<14} {'group':<14} {'n':>3} {'recall@5':>9} {'mrr@10':>7} {'p50 ms':>7}")
    for name, data in methods.items():
        for group, s in data["summary"].items():
            print(
                f"{name:<14} {group:<14} {s['questions']:>3} {s['recall@5']:>9.3f} "
                f"{s['mrr@10']:>7.3f} {s['latency_p50_ms']:>7.1f}"
            )
    for name, data in methods.items():
        misses = [q["qid"] for q in data["questions"] if q["recall"] < 1]
        print(f"{name} misses in top 5 ({len(misses)}): {', '.join(misses) or 'none'}")


def measure_latency(
    items: Sequence[RetrievalItem], search: Search, repeats: int, company_filter: bool = True
) -> dict[str, float]:
    """Time every question `repeats` times on warm models, for steadier p50/p95 than one pass."""
    samples: list[float] = []
    for _ in range(repeats):
        for item in items:
            started = time.perf_counter()
            search(item.question, item.ticker if company_filter else None)
            samples.append((time.perf_counter() - started) * 1000)
    samples.sort()
    return {
        "searches": len(samples),
        "p50_ms": round(statistics.median(samples), 1),
        "p95_ms": round(samples[math.ceil(0.95 * len(samples)) - 1], 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the search methods on the eval set.")
    parser.add_argument("--chunking", choices=get_args(Strategy), default="fixed")
    parser.add_argument("--latency-repeats", type=int, default=0, help="extra timed passes")
    args = parser.parse_args()
    chunking = args.chunking
    settings = Settings()
    items = load_items(settings)
    chunks, searchers = build_searchers(settings, chunking)
    for search in searchers.values():  # first calls load model weights; keep out of latency
        search("warm-up", None)

    report: dict = {"chunking": chunking, "unreachable": unreachable(items, chunks), "settings": {}}
    for setting, company_filter in SETTINGS.items():
        methods = {}
        for name, search in searchers.items():
            results = evaluate(items, search, company_filter)
            methods[name] = {
                "summary": {g: summarize(rs) for g, rs in groups(results).items()},
                "questions": [r.model_dump() for r in results],
            }
        report["settings"][setting] = methods

    print(f"{len(items)} questions, {chunking} chunks ({len(chunks):,}); ", end="")
    print(f"unreachable: {report['unreachable'] or 'none'}")
    for setting, methods in report["settings"].items():
        print_table(setting, methods)

    if args.latency_repeats:
        report["latency"] = {
            name: measure_latency(items, search, args.latency_repeats)
            for name, search in searchers.items()
        }
        print(f"\n== latency, company filter, {args.latency_repeats} passes ==")
        for name, lat in report["latency"].items():
            print(
                f"{name:<14} {lat['searches']:>4} searches  p50 {lat['p50_ms']:>7.1f} ms  ", end=""
            )
            print(f"p95 {lat['p95_ms']:>7.1f} ms")

    out = settings.evals_dir / f"retrieval_report_{chunking}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nreport: {out}")


if __name__ == "__main__":
    main()
