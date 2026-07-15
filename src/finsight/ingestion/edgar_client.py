"""Thin client over SEC EDGAR's free public JSON APIs (no API key required).

Uses the official company_tickers.json (CIK lookup) and the per-company
submissions API to find recent filings, then downloads the primary filing
document straight from EDGAR's Archives.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

import requests

from finsight.config import (
    REQUEST_DELAY_SECONDS,
    SEC_ARCHIVES_BASE,
    SEC_SUBMISSIONS_URL,
    SEC_TICKER_MAP_URL,
    SEC_USER_AGENT,
)

_HEADERS = {"User-Agent": SEC_USER_AGENT}


@dataclass(frozen=True)
class FilingRef:
    ticker: str
    cik: int
    accession_number: str  # dashed, e.g. 0000320193-24-000123
    form: str
    filing_date: str
    primary_document: str

    @property
    def accession_no_dashes(self) -> str:
        return self.accession_number.replace("-", "")

    @property
    def document_url(self) -> str:
        return (
            f"{SEC_ARCHIVES_BASE}/{self.cik}/{self.accession_no_dashes}/"
            f"{self.primary_document}"
        )


@lru_cache(maxsize=1)
def _ticker_to_cik_map() -> dict[str, int]:
    resp = requests.get(SEC_TICKER_MAP_URL, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    raw = resp.json()
    return {row["ticker"].upper(): int(row["cik_str"]) for row in raw.values()}


def get_cik(ticker: str) -> int:
    mapping = _ticker_to_cik_map()
    try:
        return mapping[ticker.upper()]
    except KeyError as exc:
        raise ValueError(f"Unknown ticker: {ticker}") from exc


def get_recent_filings(ticker: str, form_type: str, limit: int) -> list[FilingRef]:
    """Return the `limit` most recent filings of `form_type` for `ticker`."""
    cik = get_cik(ticker)
    url = SEC_SUBMISSIONS_URL.format(cik=cik)
    time.sleep(REQUEST_DELAY_SECONDS)
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    recent = resp.json()["filings"]["recent"]

    matches: list[FilingRef] = []
    for form, accession, filing_date, primary_doc in zip(
        recent["form"],
        recent["accessionNumber"],
        recent["filingDate"],
        recent["primaryDocument"],
    ):
        if form == form_type:
            matches.append(
                FilingRef(
                    ticker=ticker.upper(),
                    cik=cik,
                    accession_number=accession,
                    form=form,
                    filing_date=filing_date,
                    primary_document=primary_doc,
                )
            )
        if len(matches) == limit:
            break
    return matches


def download_filing_html(ref: FilingRef) -> str:
    time.sleep(REQUEST_DELAY_SECONDS)
    resp = requests.get(ref.document_url, headers=_HEADERS, timeout=60)
    resp.raise_for_status()
    return resp.text
