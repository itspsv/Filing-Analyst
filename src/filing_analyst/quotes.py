"""Does a quote really appear in a text? Used for test-set labels and for answer citations."""

import re

FANCY_TO_PLAIN = {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"}


def normalize_for_match(text: str) -> str:
    """For matching only: lowercase, straight quotes and dashes, single spaces."""
    for fancy, plain in FANCY_TO_PLAIN.items():
        text = text.replace(fancy, plain)
    return re.sub(r"\s+", " ", text).strip().lower()


ELLIPSIS = re.compile(r"\s*(?:\.{3}|…)\s*")
END_PUNCTUATION = " .,;:!?"


def contains_quote(text: str, quote: str) -> bool:
    return normalize_for_match(quote) in normalize_for_match(text)


def quote_pieces(quote: str) -> list[str]:
    """'A grew ... B fell.' -> ['A grew', 'B fell']. A quote without '...' is one piece.
    Punctuation at the end of a piece is dropped: models cut a sentence short and add a full
    stop ("...technology." where the filing goes on ", including network ..."). 7 of 21 rejected
    citations on Nov 17. Every word must still match exactly and in order."""
    pieces = (piece.strip().rstrip(END_PUNCTUATION) for piece in ELLIPSIS.split(quote))
    return [p for p in pieces if p]


def text_supports_quote(text: str, quote: str, min_words: int) -> bool:
    """True if every piece of the quote appears word for word in the text, and at least one
    piece has min_words words. Models shorten quotes with '...' (30 of 31 rejected citations
    on Nov 16); each piece is still checked exactly, so the claim still traces to the text."""
    pieces = quote_pieces(quote)
    if not pieces or max(len(p.split()) for p in pieces) < min_words:
        return False
    return all(contains_quote(text, piece) for piece in pieces)
