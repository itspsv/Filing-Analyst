from filing_analyst.answer import Answer, Citation
from filing_analyst.evaluation.answer_eval import judge, summarize
from filing_analyst.evaluation.retrieval_set import RetrievalItem

ITEM = RetrievalItem(
    qid="q1",
    ticker="AVGO",
    section="1A",
    style="literal",
    question="Who makes Broadcom's wafers?",
    quotes=["95% of the wafers were produced by TSMC"],
    source_chunk_id="AVGO-2025-1A-0018",
)
TEXTS = {
    "AVGO-2025-1A-0018": "approximately 95% of the wafers were produced by TSMC last year",
    "AVGO-2025-1A-0017": "we outsource manufacturing",
}


def answer(cited: list[str], refused: bool = False, dropped: int = 0) -> Answer:
    return Answer(
        answer="x",
        citations=[Citation(kind="chunk", ref=c, ticker="AVGO", section="1A", quote_or_row="q")
                   for c in cited],
        refused=refused,
        steps=1,
        tokens=1000,
        cost_eur=0.0,
        dropped_citations=dropped,
        retrieved=list(TEXTS),
    )  # fmt: skip


def test_grounded_only_when_the_cited_chunk_holds_the_label() -> None:
    assert judge(ITEM, answer(["AVGO-2025-1A-0018"]), TEXTS, 1.0).grounded
    wrong = judge(ITEM, answer(["AVGO-2025-1A-0017"]), TEXTS, 1.0)
    assert not wrong.grounded and wrong.label_retrieved  # shown to the model, but not cited


def test_summary_counts_refusals_and_citation_validity() -> None:
    results = [
        judge(ITEM, answer(["AVGO-2025-1A-0018"], dropped=1), TEXTS, 2.0),
        judge(ITEM, answer([], refused=True), TEXTS, 1.0),
    ]
    s = summarize(results)
    assert s["refused"] == 1
    assert s["grounded"] == 0.5 and s["grounded_when_answered"] == 1.0
    assert s["citation_valid"] == 0.5  # 1 kept, 1 dropped
