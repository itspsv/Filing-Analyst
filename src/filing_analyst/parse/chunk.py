"""Cut section text into small, searchable chunks.

Strategy 1 (this file): fixed-size windows of 300 tokens with 50 tokens of overlap, counted with
the real tokenizer of the embedding model (bge-small reads at most 512 tokens per input).
`python -m filing_analyst.parse.chunk` writes data/chunks/fixed.jsonl.
"""

import re
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel

from filing_analyst.config import COMPANIES, Settings
from filing_analyst.parse.sections import SECTIONS, load_manifest

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_TOKENS = 300
OVERLAP_TOKENS = 50

# Page furniture left over from the HTML: measured 482 "Table of Contents" lines, 551 bare page
# numbers and ~50 "Form 10-K" headers/footers across the 40 section files.
NOISE_LINES = [
    re.compile(r"^table of contents$", re.IGNORECASE),
    re.compile(r"^\d{1,3}$"),  # page numbers (also drops a few tiny table cells; fine for text)
    re.compile(r"\|.*form 10-k|form 10-k.*\|", re.IGNORECASE),  # "Apple Inc. | 2026 Form 10-K | 19"
    re.compile(r"^(fy\s*)?\d{4}\s+form 10-k$", re.IGNORECASE),  # "FY 2025 FORM 10-K"
]


class Chunk(BaseModel):
    chunk_id: str  # e.g. "AAPL-2026-1A-0007"; the retrieval test set points at these ids
    ticker: str
    fiscal_year: int
    section: Literal["1A", "7"]
    text: str
    token_count: int


class Encoding(Protocol):
    offsets: Sequence[tuple[int, int]]
    word_ids: Sequence[int | None]


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = ...) -> Encoding: ...


def clean_text(text: str) -> str:
    """Drop page headers, footers and page numbers; keep everything else."""
    kept = [line for line in text.splitlines() if not any(p.search(line) for p in NOISE_LINES)]
    return "\n".join(kept)


def _word_start(word_ids: Sequence[int | None], i: int) -> int:
    """Move i left until it is the first token of a word, so no word is cut in half."""
    while 0 < i < len(word_ids) and word_ids[i] is not None and word_ids[i] == word_ids[i - 1]:
        i -= 1
    return i


def token_windows(word_ids: Sequence[int | None], size: int, overlap: int) -> list[tuple[int, int]]:
    """(start, end) token ranges of at most `size` tokens, each overlapping the last by ~overlap."""
    windows: list[tuple[int, int]] = []
    start, n = 0, len(word_ids)
    while start < n:
        end = min(start + size, n)
        if end < n:
            boundary = _word_start(word_ids, end)
            end = boundary if boundary > start else end  # one giant "word": cut it anyway
        windows.append((start, end))
        if end == n:
            break
        next_start = _word_start(word_ids, end - overlap)
        start = next_start if next_start > start else end
    return windows


def chunk_text(
    text: str,
    ticker: str,
    fiscal_year: int,
    section: Literal["1A", "7"],
    tokenizer: Tokenizer,
    size: int = CHUNK_TOKENS,
    overlap: int = OVERLAP_TOKENS,
) -> list[Chunk]:
    encoding = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i, (start, end) in enumerate(token_windows(encoding.word_ids, size, overlap)):
        piece = text[encoding.offsets[start][0] : encoding.offsets[end - 1][1]]
        chunks.append(
            Chunk(
                chunk_id=f"{ticker}-{fiscal_year}-{section}-{i:04d}",
                ticker=ticker,
                fiscal_year=fiscal_year,
                section=section,
                text=piece,
                token_count=end - start,
            )
        )
    return chunks


def load_tokenizer() -> Tokenizer:
    from tokenizers import Tokenizer as HFTokenizer

    tokenizer = HFTokenizer.from_pretrained(EMBEDDING_MODEL)
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def build_chunks(settings: Settings, tokenizer: Tokenizer) -> list[Chunk]:
    fiscal_years = {f.ticker: f.report_date.year for f in load_manifest(settings)}
    chunks: list[Chunk] = []
    for ticker in COMPANIES:
        for section in SECTIONS:
            raw = (settings.sections_dir / f"{ticker}_{section}.txt").read_text()
            chunks += chunk_text(clean_text(raw), ticker, fiscal_years[ticker], section, tokenizer)
    return chunks


def save_chunks(chunks: list[Chunk], settings: Settings, name: str = "fixed") -> None:
    settings.chunks_dir.mkdir(parents=True, exist_ok=True)
    lines = (c.model_dump_json() for c in chunks)
    (settings.chunks_dir / f"{name}.jsonl").write_text("\n".join(lines) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    return [Chunk.model_validate_json(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    settings = Settings()
    chunks = build_chunks(settings, load_tokenizer())
    save_chunks(chunks, settings)
    for ticker in COMPANIES:
        counts = {s: sum(c.ticker == ticker and c.section == s for c in chunks) for s in SECTIONS}
        print(f"{ticker:<6} 1A={counts['1A']:>4}  7={counts['7']:>4}")
    sizes = [c.token_count for c in chunks]
    median = statistics.median(sizes)
    print(f"\n{len(chunks):,} chunks; tokens min {min(sizes)}, median {median}, max {max(sizes)}")
    print(f"written to {settings.chunks_dir / 'fixed.jsonl'}")


if __name__ == "__main__":
    main()
