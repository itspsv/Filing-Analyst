"""Measure cited answers on the reviewed retrieval set (code checks only, no LLM judge).

label retrieved = the labelled passage is among the 5 chunks shown to the model
grounded        = the answer cites a chunk that contains the labelled quote
citation valid  = share of the model's citations that passed the code check

uv run python -m filing_analyst.evaluation.answer_eval --model openai/gpt-oss-20b --limit 5
"""

import argparse
import json
import math
import statistics
import time
from collections.abc import Sequence

from pydantic import BaseModel

from filing_analyst.answer import Answer, answer_question, load_default_search
from filing_analyst.config import Settings
from filing_analyst.evaluation.retrieval_eval import load_items
from filing_analyst.evaluation.retrieval_set import RetrievalItem
from filing_analyst.llm import LLMClient
from filing_analyst.parse.chunk import load_chunks
from filing_analyst.quotes import contains_quote


class AnswerResult(BaseModel):
    qid: str
    style: str
    question: str
    answer: str
    refused: bool
    refusal_reason: str | None
    label_retrieved: bool
    grounded: bool
    citations: int
    dropped_citations: int
    tokens: int
    cost_eur: float
    seconds: float
    cited: list[str]


def judge(
    item: RetrievalItem, answer: Answer, texts: dict[str, str], seconds: float
) -> AnswerResult:
    def has_label(chunk_id: str) -> bool:
        return any(contains_quote(texts[chunk_id], q) for q in item.quotes)

    cited = [c.ref for c in answer.citations]
    return AnswerResult(
        qid=item.qid,
        style=item.style,
        question=item.question,
        answer=answer.answer,
        refused=answer.refused,
        refusal_reason=answer.refusal_reason,
        label_retrieved=any(has_label(cid) for cid in answer.retrieved),
        grounded=any(has_label(cid) for cid in cited),
        citations=len(cited),
        dropped_citations=answer.dropped_citations,
        tokens=answer.tokens,
        cost_eur=answer.cost_eur,
        seconds=round(seconds, 2),
        cited=cited,
    )


def summarize(results: Sequence[AnswerResult]) -> dict[str, float]:
    n = len(results)
    answered = [r for r in results if not r.refused]
    total_citations = sum(r.citations + r.dropped_citations for r in results)
    seconds = sorted(r.seconds for r in results)
    return {
        "questions": n,
        "refused": sum(r.refused for r in results),
        "label_retrieved": round(sum(r.label_retrieved for r in results) / n, 3),
        "grounded": round(sum(r.grounded for r in results) / n, 3),
        "grounded_when_answered": round(
            sum(r.grounded for r in answered) / len(answered), 3) if answered else 0.0,
        "citation_valid": round(
            sum(r.citations for r in results) / total_citations, 3) if total_citations else 0.0,
        "tokens_mean": round(statistics.mean(r.tokens for r in results)),
        "cost_eur_total": round(sum(r.cost_eur for r in results), 4),
        "seconds_p50": round(statistics.median(seconds), 2),
        "seconds_p95": seconds[math.ceil(0.95 * n) - 1],
    }  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure cited answers on the eval set.")
    parser.add_argument("--model", default=None, help="override LLM_MODEL")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    args = parser.parse_args()

    settings = Settings()
    if args.model:
        settings = settings.model_copy(update={"llm_model": args.model})
    items = load_items(settings)[: args.limit]
    texts = {c.chunk_id: c.text for c in load_chunks(settings.chunks_dir / "fixed.jsonl")}
    search, client = load_default_search(settings), LLMClient(settings)
    search("warm-up", None)  # model loading stays out of the timings

    results = []
    for item in items:
        started = time.perf_counter()
        answer = answer_question(item.question, item.ticker, search, client)
        results.append(judge(item, answer, texts, time.perf_counter() - started))
        r = results[-1]
        flag = "REFUSED" if r.refused else ("grounded" if r.grounded else "NOT grounded")
        print(f"{r.qid} {flag:<12} cites {r.citations} (dropped {r.dropped_citations}) "
              f"{r.tokens} tokens {r.seconds:.1f} s")  # fmt: skip

    summary = summarize(results)
    print(f"\n{settings.llm_model}: {json.dumps(summary, indent=2)}")
    name = settings.llm_model.split("/")[-1]
    out = settings.evals_dir / f"answer_report_{name}.json"
    report = {
        "model": settings.llm_model,
        "summary": summary,
        "questions": [r.model_dump() for r in results],
    }
    out.write_text(json.dumps(report, indent=2))
    print(f"report: {out}")


if __name__ == "__main__":
    main()
