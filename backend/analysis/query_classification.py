"""
Query Classification Engine.

Receives the output of the Query Analysis module and classifies the
query into one of three pipeline categories:

    retriever   — use the vector-store retrieval pipeline
    web_search  — use the Tavily web search pipeline
    general_llm — answer directly with the LLM

The classification result drives the conditional routing in the
LangGraph orchestration layer.
"""

import json
import logging

from langchain_core.language_models import BaseChatModel

from prompts.query_analysis_prompt import QUERY_CLASSIFICATION_PROMPT
from utils.llm_helpers import parse_llm_json_response

logger = logging.getLogger(__name__)

# Allowed classification labels
VALID_CLASSIFICATIONS = {"retriever", "web_search", "general_llm"}


def classify_query(query: str, analysis_result: dict, llm: BaseChatModel) -> str:
    """
    Classify a query into a pipeline category.

    First attempts LLM-based classification using the analysis metadata
    for context, then falls back to a rule-based approach if the LLM
    call fails or returns an invalid label.

    Args:
        query:           The user's original question.
        analysis_result: Dictionary returned by ``analyze_query()``.
        llm:             An initialised LangChain chat model.

    Returns:
        One of: 'retriever', 'web_search', 'general_llm'.

    Example::

        >>> analysis = {"requires_retrieval": True, "requires_web_search": False, ...}
        >>> classify_query("Explain my uploaded doc", analysis, llm)
        'retriever'
    """
    logger.info("Classifying query: '%s'", query[:120])

    try:
        chain = QUERY_CLASSIFICATION_PROMPT | llm
        response = chain.invoke({
            "query": query,
            "analysis_json": json.dumps(analysis_result, default=str),
        })
        parsed = parse_llm_json_response(
            response.content,
            defaults={"classification": "general_llm"},
        )
        classification = parsed.get("classification", "general_llm").lower().strip()

        # Validate the label
        if classification not in VALID_CLASSIFICATIONS:
            logger.warning(
                "LLM returned invalid classification '%s'; falling back to rules.",
                classification,
            )
            classification = _rule_based_classification(analysis_result)

    except Exception as e:
        logger.error("LLM classification failed: %s — using rule-based fallback.", e)
        classification = _rule_based_classification(analysis_result)

    logger.info("Classification result: %s", classification)
    return classification


# ─── Rule-Based Fallback ──────────────────────────────────────────────────────


def _rule_based_classification(analysis_result: dict) -> str:
    """
    Deterministic fallback when the LLM is unavailable or returns
    an invalid classification.

    Rules (in priority order):
        1. If ``requires_retrieval`` is True  → ``retriever``
        2. If ``requires_web_search`` is True → ``web_search``
        3. Otherwise                          → ``general_llm``
    """
    if analysis_result.get("requires_retrieval"):
        return "retriever"
    if analysis_result.get("requires_web_search"):
        return "web_search"
    return "general_llm"
