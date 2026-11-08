"""Turn 10-K HTML (inline XBRL) into clean plain text, one block of text per line."""

import re
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

BLOCK_TAGS = ["p", "div", "tr", "li", "br", "table", "h1", "h2", "h3", "h4", "h5", "h6"]
HIDDEN_STYLE = re.compile(r"display:\s*none", re.IGNORECASE)
XBRL_HEADER = re.compile(r"^ix:header$")


def html_to_text(html: bytes | str) -> str:
    with warnings.catch_warnings():
        # 10-K files start with an XML declaration; the HTML parser still reads them fine.
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for tag in soup.find_all(XBRL_HEADER):
        tag.decompose()
    for tag in soup.find_all(style=HIDDEN_STYLE):
        tag.decompose()
    for tag in soup.find_all(BLOCK_TAGS):
        tag.append("\n")
    for tag in soup.find_all("td"):
        tag.append(" ")
    return normalise_whitespace(soup.get_text(""))


def normalise_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n\s*", "\n", text)  # trim around line breaks, drop empty lines
    return text.strip()
