"""Score the search methods on the reviewed retrieval set.

A retrieved chunk is relevant if it belongs to the question's company and contains one of its
quotes. No LLM is involved, so a run is free and repeatable.

    recall@5 = share of a question's quotes found in the top 5 (one quote: hit or miss)
    MRR@10   = average of 1 / rank of the first relevant chunk in the top 10 (0 if none)

    uv run python -m filing_analyst.evaluation.retrieval_eval
"""

import json
import math
import re
import statistics
import time
from collections.abc import Callable, Sequence

from pydantic import BaseModel

from filing_analyst.config import Settings
from filing_analyst.evaluation.retrieval_set import RetrievalItem, contains_quote
from filing_analyst.parse.chunk import Chunk
from filing_analyst.retrieval.bm25 import BM25Index
from filing_analyst.retrieval.dense import embed_query, load_embedder, load_or_build_index
from filing_analyst.retrieval.rrf import HybridSearcher

RECALL_K = 5
MRR_K = 10
ITEM_MENTION = re.compile(r"\bitem\s*(1a|7)\b", re.IGNORECASE)

Search = Callable[[str], list[Chunk]]  # question -> best chunks first


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


def evaluate(items: Sequence[RetrievalItem], search: Search) -> list[QuestionResult]:
    results = []
    for item in items:
        started = time.perf_counter()
        found = search(item.question)
        results.append(score(item, found, (time.perf_counter() - started) * 1000))
    return results


def unreachable(items: Sequence[RetrievalItem], chunks: Sequence[Chunk]) -> list[str]:
    """Questions whose quote sits in no single chunk (cut by a chunk boundary): no method
    can score on them with this chunking, so they are reported, not hidden."""
    return [i.qid for i in items if not any(is_relevant(c, i) for c in chunks)]


def load_items(settings: Settings) -> list[RetrievalItem]:
    path = settings.evals_dir / "retrieval_set.jsonl"
    return [RetrievalItem.model_validate_json(line) for line in path.read_text().splitlines()]


def build_searchers(settings: Settings) -> tuple[list[Chunk], dict[str, Search]]:
    embedder = load_embedder()
    dense = load_or_build_index(settings, make_embedder=lambda: embedder)
    bm25 = BM25Index(dense.chunks)
    hybrid = HybridSearcher(dense, bm25, embedder)
    searchers: dict[str, Search] = {
        "dense": lambda q: [c for c, _ in dense.search(embed_query(q, embedder), k=MRR_K)],
        "bm25": lambda q: [c for c, _ in bm25.search(q, k=MRR_K)],
        "hybrid": lambda q: [r.chunk for r in hybrid.search(q, k=MRR_K)],
    }
    return dense.chunks, searchers


def main() -> None:
    settings = Settings()
    items = load_items(settings)
    chunks, searchers = build_searchers(settings)
    searchers["dense"]("warm-up")  # first call loads model weights; keep it out of latency

    report: dict = {"chunking": "fixed", "unreachable": unreachable(items, chunks), "methods": {}}
    per_question: dict[str, list[QuestionResult]] = {}
    for name, search in searchers.items():
        per_question[name] = evaluate(items, search)
        summary = {g: summarize(rs) for g, rs in groups(per_question[name]).items()}
        report["methods"][name] = {
            "summary": summary,
            "questions": [r.model_dump() for r in per_question[name]],
        }

    print(f"{len(items)} questions, fixed chunks; unreachable: {report['unreachable'] or 'none'}")
    print(f"\n{'method':<8} {'group':<14} {'n':>3} {'recall@5':>9} {'mrr@10':>7} {'p50 ms':>7}")
    for name, data in report["methods"].items():
        for group, s in data["summary"].items():
            print(
                f"{name:<8} {group:<14} {s['questions']:>3} {s['recall@5']:>9.3f} "
                f"{s['mrr@10']:>7.3f} {s['latency_p50_ms']:>7.1f}"
            )
    for name, results in per_question.items():
        misses = [r.qid for r in results if r.recall < 1]
        print(f"\n{name} misses in top 5 ({len(misses)}): {', '.join(misses) or 'none'}")

    out = settings.evals_dir / "retrieval_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nreport: {out}")


if __name__ == "__main__":
    main()
