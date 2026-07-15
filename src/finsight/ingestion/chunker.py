"""Recursive text chunking with per-chunk metadata tagging."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from finsight.config import CHUNK_OVERLAP, CHUNK_SIZE, TEN_K_SECTIONS
from finsight.ingestion.edgar_client import FilingRef

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    ticker: str
    sector: str
    filing_type: str
    filing_date: str
    accession_number: str
    section_key: str
    section_title: str
    source_url: str
    chunk_index: int = field(compare=False)


def chunk_filing(
    ref: FilingRef, sector: str, sections: dict[str, str]
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section_key, section_text in sections.items():
        section_title = TEN_K_SECTIONS.get(section_key, "Full Filing")
        for i, piece in enumerate(_splitter.split_text(section_text)):
            digest = hashlib.sha1(
                f"{ref.accession_number}:{section_key}:{i}".encode()
            ).hexdigest()[:12]
            chunks.append(
                Chunk(
                    chunk_id=digest,
                    text=piece,
                    ticker=ref.ticker,
                    sector=sector,
                    filing_type=ref.form,
                    filing_date=ref.filing_date,
                    accession_number=ref.accession_number,
                    section_key=section_key,
                    section_title=section_title,
                    source_url=ref.document_url,
                    chunk_index=i,
                )
            )
    return chunks
