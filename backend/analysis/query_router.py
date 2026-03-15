"""
Query Router / Strategy Selection Module.

Determines which execution pipeline should handle a query based on
the classification result from the Query Classification engine.

Routing Map:
    retriever   → retriever_pipeline
    web_search  → web_search_pipeline
    general_llm → direct_llm_pipeline

This module is used by the LangGraph orchestration layer as a
conditional edge function.
"""

import logging

logger = logging.getLogger(__name__)

# ── Route Map ─────────────────────────────────────────────────────────────────
# Maps classification labels to LangGraph node names.

ROUTE_MAP: dict[str, str] = {
    "retriever": "retriever_pipeline",
    "web_search": "web_search_pipeline",
    "general_llm": "direct_llm_pipeline",
}

DEFAULT_ROUTE = "direct_llm_pipeline"


# ── Public API ────────────────────────────────────────────────────────────────


def route_query(classification: str) -> str:
    """
    Select the next pipeline node based on the query classification.

    This function is designed to be used directly as a **LangGraph
    conditional edge function**.  It reads the classification from
    the graph state and returns the name of the next node.

    Args:
        classification: One of ``'retriever'``, ``'web_search'``,
                        or ``'general_llm'``.

    Returns:
        The LangGraph node name to execute next.

    Example::

        >>> route_query("retriever")
        'retriever_pipeline'

        >>> route_query("web_search")
        'web_search_pipeline'

        >>> route_query("general_llm")
        'direct_llm_pipeline'
    """
    route = ROUTE_MAP.get(classification, DEFAULT_ROUTE)

    if classification not in ROUTE_MAP:
        logger.warning(
            "Unknown classification '%s' — defaulting to '%s'.",
            classification,
            DEFAULT_ROUTE,
        )
    else:
        logger.info(
            "Routing classification '%s' → node '%s'.",
            classification,
            route,
        )

    return route


def route_from_state(state: dict) -> str:
    """
    LangGraph-compatible wrapper that extracts the classification
    from the shared state dict and delegates to :func:`route_query`.

    This is the function you pass to ``graph.add_conditional_edges()``.

    Args:
        state: The LangGraph ``AdaptiveRAGState`` dictionary.

    Returns:
        The LangGraph node name to execute next.

    Example integration with LangGraph::

        from analysis.query_router import route_from_state

        graph.add_conditional_edges(
            "classify_query_node",
            route_from_state,
            {
                "retriever_pipeline":  "retriever_pipeline",
                "web_search_pipeline": "web_search_pipeline",
                "direct_llm_pipeline": "direct_llm_pipeline",
            },
        )
    """
    classification = state.get("classification", "general_llm")
    return route_query(classification)
