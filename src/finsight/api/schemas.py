from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    llm_configured: bool


class FilingSummary(BaseModel):
    ticker: str
    sector: str
    form: str
    filing_date: str
    accession_number: str
    source_url: str
    chunk_count: int


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)


class CompareRequest(BaseModel):
    tickers: list[str] = Field(min_length=2)
    topic: str = Field(min_length=1)


class Citation(BaseModel):
    ticker: str
    section_title: str
    filing_date: str
    source_url: str


class AgentAnswer(BaseModel):
    agent: str
    answer: str
    citations: list[Citation]
