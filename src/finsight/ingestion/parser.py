"""Parses raw 10-K HTML into clean text, segmented by standard Item sections.

EDGAR 10-Ks don't have a consistent DOM structure across filers, so section
detection is done on the flattened text using the "Item N" heading pattern
common to nearly all 10-Ks, rather than relying on HTML tags/classes.
"""

from __future__ import annotations

import re
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from finsight.config import TEN_K_SECTIONS

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Matches headings like "Item 1A. Risk Factors" / "ITEM 7 — MD&A" at line start
_ITEM_HEADING_RE = re.compile(
    r"^\s*item\s+(\d{1,2}[a-c]?)\.?\s*[-—:]?\s*(.{0,80})$",
    re.IGNORECASE | re.MULTILINE,
)


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # collapse excess blank lines left behind by table/div stripping
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def split_into_sections(text: str) -> dict[str, str]:
    """Best-effort split of filing text into {item_key: section_text}.

    Falls back to a single "full" section if headings can't be confidently
    located (some filers use image-based or non-standard headers).
    """
    matches = list(_ITEM_HEADING_RE.finditer(text))
    if len(matches) < 3:
        # too few matches to trust as real section boundaries
        return {"full": text}

    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        item_num = m.group(1).lower()
        key = f"item{item_num}"
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        # keep the longest occurrence of a given item key (later ones in a
        # filing are often just table-of-contents references, not real body)
        if key not in sections or len(body) > len(sections[key]):
            sections[key] = body

    known = {k: v for k, v in sections.items() if k in TEN_K_SECTIONS}
    return known or {"full": text}
