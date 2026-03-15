"""
Pydantic models for API request/response validation.

Defines the data contracts between the Streamlit frontend and FastAPI backend.
"""

from pydantic import BaseModel, Field


# ─── Request Models ───────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    """Request body for the /rag/query endpoint."""

    query: str = Field(
        ...,
        min_length=1,
        description="The user's natural-language question.",
        examples=["What is Adaptive RAG?"],
    )
    model: str = Field(
        default="gpt",
        description="LLM provider to use. Options: gpt | gemini | claude",
        examples=["gpt", "gemini", "claude"],
    )


# ─── Response Models ─────────────────────────────────────────────────────────


class QueryResponse(BaseModel):
    """Response body returned by the /rag/query endpoint."""

    answer: str = Field(
        ..., description="The generated answer text."
    )
    strategy: str = Field(
        ..., description="The pipeline strategy used (e.g. retriever, general_llm, web_search)."
    )
    sources: list[str] = Field(
        default_factory=list,
        description="List of source document names or URLs used to generate the answer.",
    )


class UploadResponse(BaseModel):
    """Response body returned by the /rag/documents/upload endpoint."""

    message: str = Field(
        ..., description="Human-readable success or error message."
    )
    filename: str = Field(
        ..., description="Name of the uploaded file."
    )
    chunks_created: int = Field(
        ..., description="Number of text chunks created and indexed."
    )


class HealthResponse(BaseModel):
    """Response body returned by the /health endpoint."""

    status: str = Field(default="ok")
    service: str = Field(default="adaptive-rag-backend")
