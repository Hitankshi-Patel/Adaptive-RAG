"""
RAG Service.

Bridges the API layer to the LangGraph orchestration pipeline.
The API routes call this service; this service calls the pipeline.
"""

import logging

from pipelines.langgraph_pipeline import run_adaptive_rag
from schemas.request_models import QueryResponse
from vectorstore.vector_db import VectorStoreManager

logger = logging.getLogger(__name__)


async def process_query(
    query: str,
    model: str,
    vector_store_manager: VectorStoreManager,
) -> QueryResponse:
    """
    Process a user query through the Adaptive-RAG pipeline.

    Args:
        query:                The user's question.
        model:                LLM provider key (gpt / gemini / claude).
        vector_store_manager: Initialised VectorStoreManager instance.

    Returns:
        A validated ``QueryResponse`` Pydantic model.
    """
    logger.info("Service processing query: '%s' with model '%s'", query[:80], model)

    # Run the full LangGraph pipeline
    result = run_adaptive_rag(
        query=query,
        model=model,
        vector_store_manager=vector_store_manager,
    )

    # Map pipeline output to the API response schema
    response = QueryResponse(
        answer=result.get("response", "No answer generated."),
        strategy=result.get("classification", "unknown"),
        sources=result.get("sources", []),
    )

    logger.info("Service returning strategy=%s, answer_len=%d", response.strategy, len(response.answer))
    return response
