from unittest.mock import patch

from finsight.ingestion.chunker import chunk_filing
from finsight.ingestion.edgar_client import FilingRef, get_recent_filings
from finsight.ingestion.parser import html_to_text, split_into_sections

SAMPLE_10K_TEXT = """
Item 1. Business
We design, manufacture, and market widgets. Our fiscal year ends in September.

Item 1A. Risk Factors
Our business is subject to numerous risks including supply chain disruption,
competition, and regulatory changes that could materially harm results.

Item 7. Management's Discussion and Analysis
Revenue increased 12% year over year driven by strong widget demand.
""".strip()


def make_filing_ref(**overrides) -> FilingRef:
    defaults = dict(
        ticker="TEST",
        cik=1234567890,
        accession_number="0001234567-24-000001",
        form="10-K",
        filing_date="2024-01-01",
        primary_document="test-10k.htm",
    )
    defaults.update(overrides)
    return FilingRef(**defaults)


def test_html_to_text_strips_tags_and_scripts():
    html = "<html><body><script>evil()</script><p>Hello&nbsp;World</p></body></html>"
    text = html_to_text(html)
    assert "evil()" not in text
    assert "Hello" in text and "World" in text


def test_html_to_text_decodes_numeric_entities():
    html = "<p>Company&#8217;s results</p>"
    text = html_to_text(html)
    assert "’" in text  # right single quote, not a literal entity or replacement char


def test_split_into_sections_finds_known_items():
    sections = split_into_sections(SAMPLE_10K_TEXT)
    assert "item1a" in sections
    assert "supply chain disruption" in sections["item1a"]
    assert "item7" in sections
    assert "Revenue increased 12%" in sections["item7"]


def test_split_into_sections_falls_back_when_no_headings_found():
    sections = split_into_sections("just some plain text with no item headings at all")
    assert sections == {"full": "just some plain text with no item headings at all"}


def test_chunk_filing_tags_metadata_correctly():
    ref = make_filing_ref()
    sections = split_into_sections(SAMPLE_10K_TEXT)
    chunks = chunk_filing(ref, sector="Technology", sections=sections)

    assert len(chunks) > 0
    risk_chunks = [c for c in chunks if c.section_key == "item1a"]
    assert risk_chunks, "expected at least one Risk Factors chunk"
    c = risk_chunks[0]
    assert c.ticker == "TEST"
    assert c.sector == "Technology"
    assert c.filing_type == "10-K"
    assert c.accession_number == "0001234567-24-000001"
    assert c.section_title == "Risk Factors"
    assert c.source_url.endswith("test-10k.htm")


def test_chunk_ids_are_unique_and_stable():
    ref = make_filing_ref()
    sections = split_into_sections(SAMPLE_10K_TEXT)
    chunks_a = chunk_filing(ref, sector="Technology", sections=sections)
    chunks_b = chunk_filing(ref, sector="Technology", sections=sections)

    ids_a = [c.chunk_id for c in chunks_a]
    assert len(ids_a) == len(set(ids_a)), "chunk ids must be unique within a filing"
    assert ids_a == [c.chunk_id for c in chunks_b], "chunk ids must be stable/deterministic"


@patch("finsight.ingestion.edgar_client.requests.get")
@patch("finsight.ingestion.edgar_client.get_cik", return_value=320193)
def test_get_recent_filings_filters_by_form_and_respects_limit(mock_cik, mock_get):
    mock_get.return_value.json.return_value = {
        "filings": {
            "recent": {
                "form": ["10-Q", "10-K", "10-K", "10-K"],
                "accessionNumber": ["a1", "a2", "a3", "a4"],
                "filingDate": ["2024-04-01", "2024-01-01", "2023-01-01", "2022-01-01"],
                "primaryDocument": ["d1.htm", "d2.htm", "d3.htm", "d4.htm"],
            }
        }
    }
    mock_get.return_value.raise_for_status.return_value = None

    refs = get_recent_filings("AAPL", "10-K", limit=2)

    assert len(refs) == 2
    assert all(r.form == "10-K" for r in refs)
    assert [r.accession_number for r in refs] == ["a2", "a3"]
