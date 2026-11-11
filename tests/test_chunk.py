import re
from dataclasses import dataclass

import pytest

from filing_analyst.parse.chunk import chunk_text, clean_text, token_windows


@dataclass
class FakeEncoding:
    offsets: list[tuple[int, int]]
    word_ids: list[int | None]


class WordTokenizer:
    """Test tokenizer: one token per word, except words with a '#' are split into two tokens,
    like a real subword tokenizer splitting 'semiconductors' into 'semi' + '##conductors'."""

    def encode(self, text: str, add_special_tokens: bool = False) -> FakeEncoding:
        offsets, word_ids = [], []
        for word_id, match in enumerate(re.finditer(r"\S+", text)):
            start, end = match.span()
            cut = text.find("#", start, end)
            parts = [(start, cut), (cut, end)] if cut > start else [(start, end)]
            offsets += parts
            word_ids += [word_id] * len(parts)
        return FakeEncoding(offsets, word_ids)


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_clean_text_drops_page_furniture_keeps_content() -> None:
    raw = "\n".join(
        [
            "Item 1A. Risk Factors",
            "Table of Contents",
            "Export controls could limit our sales.",
            "19",
            "Apple Inc. | 2026 Form 10-K | 19",
            "FY 2025 FORM 10-K",
            "This Form 10-K describes risks in 2025.",  # a real sentence: must stay
        ]
    )
    assert clean_text(raw).splitlines() == [
        "Item 1A. Risk Factors",
        "Export controls could limit our sales.",
        "This Form 10-K describes risks in 2025.",
    ]


def test_windows_have_max_size_and_overlap() -> None:
    windows = token_windows(list(range(1_000)), size=300, overlap=50)
    assert windows[0] == (0, 300)
    assert windows[1] == (250, 550)  # starts 50 tokens before the previous end
    assert windows[-1][1] == 1_000  # nothing at the end is lost
    assert all(end - start <= 300 for start, end in windows)


def test_window_never_cuts_a_word_in_half() -> None:
    word_ids = [0, 1, 2, 2, 3]  # word 2 is two tokens
    assert token_windows(word_ids, size=3, overlap=0)[0] == (0, 2)  # stops before word 2


def test_chunks_lose_no_text_and_ids_are_stable() -> None:
    text = words(700)
    chunks = chunk_text(text, "ACME", 2025, "1A", WordTokenizer(), size=300, overlap=50)
    assert [c.chunk_id for c in chunks] == [
        "ACME-2025-1A-0000",
        "ACME-2025-1A-0001",
        "ACME-2025-1A-0002",
    ]
    assert chunks[0].text.startswith("w0 ") and chunks[-1].text.endswith("w699")
    covered = set()
    for c in chunks:
        covered |= set(c.text.split())
    assert covered == set(text.split())
    assert all(c.token_count <= 300 for c in chunks)
    again = chunk_text(text, "ACME", 2025, "1A", WordTokenizer(), size=300, overlap=50)
    assert again == chunks  # same input, same ids and text


def test_subword_tokens_stay_in_one_chunk() -> None:
    text = "alpha beta semi#conductors gamma"  # 'semi#conductors' = 2 tokens, 1 word
    chunks = chunk_text(text, "ACME", 2025, "7", WordTokenizer(), size=3, overlap=0)
    assert [c.text for c in chunks] == ["alpha beta", "semi#conductors gamma"]


@pytest.mark.network
def test_real_bge_tokenizer_keeps_chunks_under_model_limit() -> None:
    from filing_analyst.parse.chunk import load_tokenizer

    try:
        tokenizer = load_tokenizer()
    except Exception:
        pytest.skip("could not download the bge tokenizer")
    text = "Export controls on advanced semiconductors could reduce our revenue. " * 200
    chunks = chunk_text(text, "ACME", 2025, "1A", tokenizer)
    assert all(c.token_count <= 300 for c in chunks)
    assert len(tokenizer.encode(chunks[0].text).ids) <= 512  # with [CLS]/[SEP] added
