"""
Response Delivery Module.

Responsible for the final step in the pipeline: converting the internal
response dict into a validated Pydantic model and preparing it for
JSON serialization back to the Streamlit UI via FastAPI.

Workflow:
    LLM Response (from Response Generator)
        → Validate & structure via Pydantic
        → Wrap errors if needed
        → Return via FastAPI → JSON → Streamlit UI

Functions:
    return_api_response  — convert pipeline output to QueryResponse
    build_error_response — create a structured error response
"""

import logging

from schemas.request_models import QueryResponse

logger = logging.getLogger(__name__)


def return_api_response(response_data: dict) -> QueryResponse:
    """
    Convert an internal pipeline response dict into a validated
    ``QueryResponse`` Pydantic model ready for FastAPI serialization.

    This function acts as the final checkpoint before the response
    leaves the backend.  It validates the structure, applies defaults,
    and logs the outgoing response.

    Args:
        response_data: Dict produced by the Response Generator or
                       the RAG service.  Expected keys:
                       ``answer`` (or ``response``), ``strategy``
                       (or ``classification``), ``sources``.

    Returns:
        A validated ``QueryResponse`` Pydantic model.

    Example::

        >>> data = {
        ...     "answer": "Adaptive RAG dynamically selects...",
        ...     "strategy": "retriever",
        ...     "sources": ["paper.pdf"],
        ... }
        >>> resp = return_api_response(data)
        >>> resp.model_dump()
        {'answer': 'Adaptive RAG dynamically selects...', 'strategy': 'retriever', 'sources': ['paper.pdf']}
    """
    # Normalise key names (pipeline uses "response"/"classification",
    # API schema uses "answer"/"strategy")
    answer = response_data.get("answer") or response_data.get("response", "")
    strategy = response_data.get("strategy") or response_data.get("classification", "unknown")
    sources = response_data.get("sources", [])

    # Ensure answer is never empty
    if not answer or not answer.strip():
        answer = "No answer could be generated for this query."
        logger.warning("Empty answer detected — using fallback message.")

    # Ensure sources is a list of strings
    if not isinstance(sources, list):
        sources = [str(sources)]
    sources = [str(s) for s in sources if s]
    if not sources:
        sources = ["llm_knowledge"]

    # Build the validated Pydantic model
    response = QueryResponse(
        answer=answer.strip(),
        strategy=strategy,
        sources=sources,
    )

    logger.info(
        "Response delivery — strategy=%s, sources=%d, answer_len=%d",
        response.strategy,
        len(response.sources),
        len(response.answer),
    )

    return response


def build_error_response(
    error_message: str,
    strategy: str = "error",
) -> QueryResponse:
    """
    Create a structured error response when the pipeline fails.

    Instead of raising an HTTP exception for all errors, this allows
    the frontend to display a user-friendly error within the chat UI.

    Args:
        error_message: Human-readable error description.
        strategy:      Strategy label (defaults to ``'error'``).

    Returns:
        A ``QueryResponse`` containing the error message as the answer.

    Example::

        >>> resp = build_error_response("Tavily API key is missing.")
        >>> resp.answer
        'An error occurred: Tavily API key is missing.'
    """
    logger.error("Building error response: %s", error_message)

    return QueryResponse(
        answer=f"An error occurred: {error_message}",
        strategy=strategy,
        sources=[],
    )
