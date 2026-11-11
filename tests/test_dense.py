from pathlib import Path

import numpy as np
import pytest

from filing_analyst.config import Settings
from filing_analyst.parse.chunk import Chunk, save_chunks
from filing_analyst.retrieval.dense import (
    DenseIndex,
    embed_query,
    load_or_build_index,
    normalize,
    top_k,
)

WORDS = ["export", "weather", "revenue"]


class KeywordEmbedder:
    """Fake model: one dimension per keyword, so similarity is easy to reason about."""

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, sentences: list[str], **kwargs: object) -> np.ndarray:
        self.calls += 1
        return np.array([[s.lower().count(w) + 0.01 for w in WORDS] for s in sentences])


def chunk(i: int, ticker: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{ticker}-2025-1A-{i:04d}",
        ticker=ticker,
        fiscal_year=2025,
        section="1A",
        text=text,
        token_count=len(text.split()),
    )


CHUNKS = [
    chunk(0, "NVDA", "export controls on chips, export licences"),
    chunk(1, "NVDA", "weather and natural disasters"),
    chunk(2, "AAPL", "export rules for phones"),
    chunk(3, "AAPL", "revenue grew"),
]


def test_normalize_gives_unit_length_rows() -> None:
    rows = normalize(np.array([[3.0, 4.0], [0.0, 2.0]]))
    assert np.allclose(np.linalg.norm(rows, axis=1), 1.0)
    assert np.allclose(rows[0], [0.6, 0.8])


def test_top_k_orders_best_first_and_caps_k() -> None:
    vectors = normalize(np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]]))
    query = normalize(np.array([[1.0, 0.1]]))[0]
    assert [i for i, _ in top_k(query, vectors, k=2)] == [0, 1]
    assert len(top_k(query, vectors, k=10)) == 3  # k bigger than the number of rows


def test_mask_skips_rows() -> None:
    vectors = normalize(np.array([[1.0, 0.0], [0.9, 0.1]]))
    query = np.array([1.0, 0.0], dtype=np.float32)
    assert [i for i, _ in top_k(query, vectors, k=2, mask=np.array([False, True]))] == [1]
    assert top_k(query, vectors, k=2, mask=np.array([False, False])) == []


def test_search_finds_export_chunks_and_filters_by_ticker() -> None:
    embedder = KeywordEmbedder()
    vectors = normalize(embedder.encode([c.text for c in CHUNKS]))
    index = DenseIndex(CHUNKS, vectors)
    query = embed_query("export", embedder)
    top_two = {c.chunk_id for c, _ in index.search(query, k=2)}
    assert top_two == {"NVDA-2025-1A-0000", "AAPL-2025-1A-0002"}  # the two export chunks
    only_aapl = index.search(query, k=5, ticker="aapl")
    assert {c.ticker for c, _ in only_aapl} == {"AAPL"}


def test_vectors_are_cached_until_chunks_change(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, sec_user_agent="Test test@example.com", data_dir=tmp_path)
    save_chunks(CHUNKS, settings)
    embedder = KeywordEmbedder()

    load_or_build_index(settings, make_embedder=lambda: embedder)
    load_or_build_index(settings, make_embedder=lambda: embedder)
    assert embedder.calls == 1  # second load used the saved vectors

    save_chunks([*CHUNKS, chunk(4, "AAPL", "new text")], settings)
    index = load_or_build_index(settings, make_embedder=lambda: embedder)
    assert embedder.calls == 2  # chunk file changed: embedded again
    assert index.vectors.shape == (5, 3)


@pytest.mark.network
def test_real_model_puts_export_rules_closer_than_weather() -> None:
    from filing_analyst.retrieval.dense import embed_texts, load_embedder

    try:
        embedder = load_embedder()
    except Exception:
        pytest.skip("could not download the bge model")
    passages = embed_texts(
        [
            "U.S. export controls restrict sales of advanced chips to China.",
            "Severe weather and earthquakes could disrupt our offices.",
        ],
        embedder,
    )
    scores = passages @ embed_query("Which companies are hurt by export restrictions?", embedder)
    assert scores[0] > scores[1]
