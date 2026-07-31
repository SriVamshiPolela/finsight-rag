from finsight.retrieval.corpus import get_section_chunks

CHUNKS = [
    {"ticker": "AAPL", "section_key": "item1a", "filing_date": "2024-11-01", "chunk_index": 1, "text": "b"},
    {"ticker": "AAPL", "section_key": "item1a", "filing_date": "2024-11-01", "chunk_index": 0, "text": "a"},
    {"ticker": "AAPL", "section_key": "item1a", "filing_date": "2023-11-03", "chunk_index": 0, "text": "old"},
    {"ticker": "AAPL", "section_key": "item1", "filing_date": "2024-11-01", "chunk_index": 0, "text": "biz"},
    {"ticker": "MSFT", "section_key": "item1a", "filing_date": "2024-11-01", "chunk_index": 0, "text": "msft"},
]


def test_filters_by_ticker_and_section():
    result = get_section_chunks("AAPL", "item1", chunks=CHUNKS)
    assert len(result) == 1
    assert result[0]["text"] == "biz"


def test_defaults_to_most_recent_filing_date():
    result = get_section_chunks("AAPL", "item1a", chunks=CHUNKS)
    assert all(c["filing_date"] == "2024-11-01" for c in result)
    assert len(result) == 2


def test_explicit_filing_date_overrides_default():
    result = get_section_chunks("AAPL", "item1a", filing_date="2023-11-03", chunks=CHUNKS)
    assert len(result) == 1
    assert result[0]["text"] == "old"


def test_sorted_by_chunk_index():
    result = get_section_chunks("AAPL", "item1a", chunks=CHUNKS)
    assert [c["text"] for c in result] == ["a", "b"]


def test_no_matches_returns_empty_list():
    assert get_section_chunks("TSLA", "item1a", chunks=CHUNKS) == []
