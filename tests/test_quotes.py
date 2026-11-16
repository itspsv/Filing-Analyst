from filing_analyst.quotes import quote_pieces, text_supports_quote

APPLE = (
    "Global markets for the Company’s products and services are highly competitive and subject "
    "to rapid technological change. The Company faces aggressive price competition."
)


def test_pieces_split_on_three_dots_or_ellipsis_character() -> None:
    assert quote_pieces("A grew ... B fell…C rose") == ["A grew", "B fell", "C rose"]
    assert quote_pieces("no dots here") == ["no dots here"]


def test_shortened_quote_with_real_pieces_is_supported() -> None:
    """Real case, Nov 16 (q001, Apple): two exact pieces joined by '...'."""
    quote = "are highly competitive and subject to rapid technological change... aggressive price"
    assert text_supports_quote(APPLE, quote, min_words=3)


def test_one_invented_piece_makes_the_whole_quote_unsupported() -> None:
    quote = "are highly competitive and subject to rapid technological change... prices never fall"
    assert not text_supports_quote(APPLE, quote, min_words=3)


def test_quote_needs_at_least_one_specific_piece() -> None:
    assert not text_supports_quote(APPLE, "Global ... change", min_words=3)  # pieces too short
    assert not text_supports_quote(APPLE, "...", min_words=3)  # nothing left
