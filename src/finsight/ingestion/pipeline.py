"""Phase 1 orchestration: EDGAR -> raw HTML on disk -> parsed sections -> chunks.jsonl.

Storage tradeoff (see README): raw + processed docs are written to local disk
under data/. That's fine for a portfolio-scale corpus (dozens of filings);
at production scale this would move to S3, with the manifest/chunk index in a
database instead of flat JSONL, so ingestion workers don't need a shared disk.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict

from finsight.config import COMPANIES, FILINGS_PER_COMPANY, FORM_TYPE, PROCESSED_DIR, RAW_DIR
from finsight.ingestion.chunker import chunk_filing
from finsight.ingestion.edgar_client import download_filing_html, get_recent_filings
from finsight.ingestion.parser import html_to_text, split_into_sections

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def ingest_all(companies: dict[str, str] | None = None) -> dict:
    """Runs the full Phase 1 pipeline and returns a summary dict."""
    companies = companies or COMPANIES
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    chunks_path = PROCESSED_DIR / "chunks.jsonl"
    total_chunks = 0

    with chunks_path.open("w", encoding="utf-8") as chunks_file:
        for ticker, sector in companies.items():
            logger.info("Fetching %s filings for %s (%s)", FORM_TYPE, ticker, sector)
            try:
                refs = get_recent_filings(ticker, FORM_TYPE, FILINGS_PER_COMPANY)
            except Exception:
                logger.exception("Failed to list filings for %s, skipping", ticker)
                continue

            for ref in refs:
                try:
                    html = download_filing_html(ref)
                except Exception:
                    logger.exception("Failed to download %s, skipping", ref.accession_number)
                    continue

                raw_path = RAW_DIR / f"{ticker}_{ref.accession_no_dashes}.html"
                raw_path.write_text(html, encoding="utf-8")

                text = html_to_text(html)
                sections = split_into_sections(text)
                chunks = chunk_filing(ref, sector, sections)
                for chunk in chunks:
                    chunks_file.write(json.dumps(asdict(chunk)) + "\n")
                total_chunks += len(chunks)

                manifest.append(
                    {
                        "ticker": ticker,
                        "sector": sector,
                        "form": ref.form,
                        "filing_date": ref.filing_date,
                        "accession_number": ref.accession_number,
                        "source_url": ref.document_url,
                        "raw_path": str(raw_path),
                        "sections_found": list(sections.keys()),
                        "chunk_count": len(chunks),
                    }
                )
                logger.info(
                    "  %s %s (%s): %d sections, %d chunks",
                    ticker,
                    ref.filing_date,
                    ref.accession_number,
                    len(sections),
                    len(chunks),
                )

    manifest_path = PROCESSED_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary = {
        "companies": len(companies),
        "filings_ingested": len(manifest),
        "total_chunks": total_chunks,
        "manifest_path": str(manifest_path),
        "chunks_path": str(chunks_path),
    }
    logger.info("Done: %s", summary)
    return summary
