import pytest
from pydantic import ValidationError

from filing_analyst.config import Settings
from filing_analyst.parse.html_text import html_to_text
from filing_analyst.parse.sections import (
    MIN_SECTION_CHARS,
    extract_section,
    find_section,
    heading_positions,
    load_manifest,
)

FAKE_10K = """<?xml version='1.0' encoding='ASCII'?>
<html><body>
<div style="display:none"><ix:header>Item 1A hidden xbrl junk</ix:header></div>
<table>
  <tr><td>Item 1A.</td><td>Risk Factors</td><td>12</td></tr>
  <tr><td>Item 1C.</td><td>Cybersecurity</td><td>30</td></tr>
</table>
<p>Forward-looking statements: see Item 1A of this report for risks.</p>
<p>ITEM 1A. RISK FACTORS</p>
<p>Export controls could hurt our sales in China.</p>
<p>Supply&nbsp;chain   problems could delay products.</p>
<p>ITEM 1C. CYBERSECURITY</p>
<p>We have a security program.</p>
</body></html>"""


def test_html_to_text_removes_hidden_and_normalises_spaces() -> None:
    text = html_to_text(FAKE_10K)
    assert "hidden xbrl junk" not in text
    assert "Supply chain problems could delay products." in text


def test_cross_reference_mid_sentence_is_not_a_heading() -> None:
    text = html_to_text(FAKE_10K)
    starts = [text[pos : pos + 30] for pos in heading_positions(text, "1A")]
    assert not any("of this report" in s for s in starts)
    assert len(starts) == 2  # the TOC row and the real heading


def test_find_section_picks_real_section_not_toc_and_ends_at_1c() -> None:
    section = find_section(html_to_text(FAKE_10K), "1A", ("1B", "1C", "2"))
    assert section is not None
    assert section.startswith("ITEM 1A. RISK FACTORS")
    assert "Export controls" in section
    assert "CYBERSECURITY" not in section


def test_item_7_heading_does_not_match_item_7a() -> None:
    assert heading_positions("Item 7A. Market risk\n", "7") == []
    assert heading_positions("Item 7. MD&A\n", "7") == [0]


def test_too_short_section_counts_as_missing() -> None:
    text = html_to_text(FAKE_10K)
    assert extract_section(text, "1A", min_chars=10) is not None
    assert extract_section(text, "1A", min_chars=MIN_SECTION_CHARS) is None


@pytest.mark.data
def test_real_apple_10k_has_both_sections() -> None:
    try:
        settings = Settings()
        filing = next(f for f in load_manifest(settings) if f.ticker == "AAPL")
    except (ValidationError, FileNotFoundError):
        pytest.skip("no settings or no downloaded 10-Ks; run filing_analyst.sec.filings first")
    text = html_to_text((settings.raw_dir / "10k" / filing.local_name).read_bytes())
    assert extract_section(text, "1A") is not None
    assert extract_section(text, "7") is not None
