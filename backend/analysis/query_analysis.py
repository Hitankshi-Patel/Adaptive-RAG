"""
Query Analysis Module.

Analyses a user query to determine intent, information source
requirements, and retrieval / web-search needs.  The structured
output feeds into the Query Classification engine and ultimately
the LangGraph routing layer.

Functions:
    analyze_query          — main orchestrator
    extract_intent         — LLM-based intent classification
    detect_information_need — determines required knowledge source
"""

import json
import logging

from langchain_core.language_models import BaseChatModel

from prompts.query_analysis_prompt import QUERY_ANALYSIS_PROMPT
from utils.llm_helpers import parse_llm_json_response

logger = logging.getLogger(__name__)

# Defaults used when the LLM response cannot be parsed
_DEFAULT_ANALYSIS = {
    "intent": "information_lookup",
    "requires_retrieval": False,
    "requires_web_search": False,
    "query_type": "factual_question",
}


# ─── Public API ───────────────────────────────────────────────────────────────


def analyze_query(query: str, llm: BaseChatModel) -> dict:
    """
    Analyse a user query and return structured metadata.

    This is the main entry point of the module.  It calls the LLM
    via a LangChain prompt template, parses the JSON response, and
    enriches it with derived fields used by downstream nodes.

    Args:
        query: The user's natural-language question.
        llm:   An initialised LangChain chat model.

    Returns:
        Dictionary with keys:
            query, intent, information_source,
            requires_retrieval, requires_web_search, query_type
    """
    logger.info("Analyzing query: '%s'", query[:120])

    try:
        # Run the analysis prompt
        chain = QUERY_ANALYSIS_PROMPT | llm
        response = chain.invoke({"query": query})
        parsed = parse_llm_json_response(response.content, defaults=_DEFAULT_ANALYSIS)

        # Extract individual fields (with safe defaults)
        intent = parsed.get("intent", _DEFAULT_ANALYSIS["intent"])
        requires_retrieval = parsed.get("requires_retrieval", False)
        requires_web_search = parsed.get("requires_web_search", False)
        query_type = parsed.get("query_type", _DEFAULT_ANALYSIS["query_type"])

    except Exception as e:
        logger.error("LLM analysis failed, using heuristic fallback: %s", e)
        intent = extract_intent(query, llm=None)  # keyword fallback
        info_need = detect_information_need(query, llm=None)
        requires_retrieval = info_need == "retrieval_needed"
        requires_web_search = info_need == "web_search_needed"
        query_type = "factual_question"

    # Derive the canonical information_source label
    information_source = detect_information_need(
        query, llm=None,
        override_retrieval=requires_retrieval,
        override_web=requires_web_search,
    )

    result = {
        "query": query,
        "intent": intent,
        "information_source": information_source,
        "requires_retrieval": requires_retrieval,
        "requires_web_search": requires_web_search,
        "query_type": query_type,
    }
    logger.info("Analysis result: %s", json.dumps(result, default=str))
    return result


# ─── Sub-functions ────────────────────────────────────────────────────────────


def extract_intent(query: str, llm: BaseChatModel | None = None) -> str:
    """
    Determine the user's intent behind the query.

    When an LLM is provided, intent extraction is delegated to the
    full ``analyze_query`` flow.  When called with ``llm=None``
    (e.g. as a fallback), a simple keyword heuristic is used.

    Args:
        query: The user's question.
        llm:   Optional LangChain chat model.

    Returns:
        One of: information_lookup, conversational, summarization,
        explanation, comparison, real_time_information.
    """
    if llm is not None:
        result = analyze_query(query, llm)
        return result["intent"]

    # ── Keyword-based fallback ────────────────────────────────────────
    q = query.lower()

    if any(kw in q for kw in ("summarize", "summary", "summarise", "tldr")):
        return "summarization"
    if any(kw in q for kw in ("compare", "difference between", "vs")):
        return "comparison"
    if any(kw in q for kw in ("latest", "news", "today", "current", "recent")):
        return "real_time_information"
    if any(kw in q for kw in ("explain", "what is", "how does", "define")):
        return "explanation"
    if any(kw in q for kw in ("hello", "hi", "hey", "thanks", "thank you")):
        return "conversational"

    return "information_lookup"


def detect_information_need(
    query: str,
    llm: BaseChatModel | None = None,
    *,
    override_retrieval: bool | None = None,
    override_web: bool | None = None,
) -> str:
    """
    Determine the type of knowledge source required for a query.

    Args:
        query:              The user's question.
        llm:                Optional LangChain chat model (unused in override mode).
        override_retrieval: If set, use this value instead of running the LLM.
        override_web:       If set, use this value instead of running the LLM.

    Returns:
        One of: retrieval_needed, web_search_needed, general_knowledge.
    """
    # When explicit flags are provided (used by analyze_query)
    if override_retrieval is not None or override_web is not None:
        if override_retrieval:
            return "retrieval_needed"
        if override_web:
            return "web_search_needed"
        return "general_knowledge"

    # When called with an LLM, delegate to full analysis
    if llm is not None:
        result = analyze_query(query, llm)
        return result["information_source"]

    # ── Keyword-based fallback ────────────────────────────────────────
    q = query.lower()

    if any(kw in q for kw in ("document", "uploaded", "file", "pdf", "my data")):
        return "retrieval_needed"
    if any(kw in q for kw in ("latest", "news", "today", "current", "recent", "live")):
        return "web_search_needed"

    return "general_knowledge"
