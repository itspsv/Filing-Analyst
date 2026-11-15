from pathlib import Path

import numpy as np
import pytest
from qdrant_client import QdrantClient

from filing_analyst.parse.chunk import Chunk
from filing_analyst.retrieval.dense import DenseIndex, normalize
from filing_analyst.retrieval.store import load_or_build_qdrant

TICKERS = ["AAPL", "NVDA", "AMD"]


def make_dense(n: int = 60, dim: int = 8, seed: int = 7) -> DenseIndex:
    rng = np.random.default_rng(seed)
    chunks = [
        Chunk(
            chunk_id=f"{TICKERS[i % 3]}-{2025 + i % 2}-{'1A' if i % 4 else '7'}-{i:04d}",
            ticker=TICKERS[i % 3],
            fiscal_year=2025 + i % 2,
            section="1A" if i % 4 else "7",
            text=f"chunk {i}",
            token_count=2,
        )
        for i in range(n)
    ]
    return DenseIndex(chunks, normalize(rng.normal(size=(n, dim))))


@pytest.fixture
def dense() -> DenseIndex:
    return make_dense()


@pytest.fixture
def qdrant_client() -> QdrantClient:
    return QdrantClient(":memory:")


def queries(dense: DenseIndex, count: int = 5) -> list[np.ndarray]:
    rng = np.random.default_rng(1)
    return list(normalize(rng.normal(size=(count, dense.vectors.shape[1]))))


def test_same_top_10_as_numpy(
    dense: DenseIndex, qdrant_client: QdrantClient, tmp_path: Path
) -> None:
    qdrant = load_or_build_qdrant(qdrant_client, tmp_path / "meta.json", dense)
    for q in queries(dense):
        numpy_hits = dense.search(q, k=10)
        qdrant_hits = qdrant.search(q, k=10)
        assert [c.chunk_id for c, _ in qdrant_hits] == [c.chunk_id for c, _ in numpy_hits]
        assert [s for _, s in qdrant_hits] == pytest.approx([s for _, s in numpy_hits], abs=1e-5)


def test_company_filter_matches_numpy(
    dense: DenseIndex, qdrant_client: QdrantClient, tmp_path: Path
) -> None:
    qdrant = load_or_build_qdrant(qdrant_client, tmp_path / "meta.json", dense)
    for q in queries(dense):
        numpy_ids = [c.chunk_id for c, _ in dense.search(q, k=5, ticker="nvda")]
        qdrant_ids = [c.chunk_id for c, _ in qdrant.search(q, k=5, ticker="nvda")]
        assert qdrant_ids == numpy_ids
        assert all(i.startswith("NVDA") for i in qdrant_ids)


def test_year_and_section_filters(
    dense: DenseIndex, qdrant_client: QdrantClient, tmp_path: Path
) -> None:
    qdrant = load_or_build_qdrant(qdrant_client, tmp_path / "meta.json", dense)
    # fixture: chunk i is AAPL when i % 3 == 0, year 2025 when i is even, Item 7 when i % 4 == 0,
    # so AAPL + 2025 + Item 7 = every 12th chunk: 0, 12, 24, 36, 48
    hits = qdrant.search(queries(dense)[0], k=50, ticker="AAPL", fiscal_year=2025, section="7")
    assert sorted(int(c.chunk_id[-4:]) for c, _ in hits) == [0, 12, 24, 36, 48]
    none = qdrant.search(queries(dense)[0], k=50, ticker="AAPL", fiscal_year=2026, section="7")
    assert none == []  # that combination does not exist


def test_collection_reused_until_vectors_change(
    dense: DenseIndex, qdrant_client: QdrantClient, tmp_path: Path
) -> None:
    meta = tmp_path / "meta.json"
    assert load_or_build_qdrant(qdrant_client, meta, dense).rebuilt is True
    assert load_or_build_qdrant(qdrant_client, meta, dense).rebuilt is False
    changed = make_dense(seed=99)  # same chunks, different vectors
    assert load_or_build_qdrant(qdrant_client, meta, changed).rebuilt is True
