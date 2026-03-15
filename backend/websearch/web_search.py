"""
Web Search Module — Tavily Integration.

Retrieves real-time information from the web for queries that need
up-to-date knowledge (news, live data, current events).

Workflow:
    User Query
        → Tavily Search API
        → Retrieve top results
        → Extract relevant content
        → Summarize results via LLM
        → Return structured context

Functions:
    search_web       — high-level orchestrator (search + summarize)
    fetch_results    — raw Tavily API call
    summarize_results — LLM-based summarization of search results
"""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


# ─── Core Functions ──────────────────────────────────────────────────────────


def fetch_results(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """
    Perform a web search via the Tavily API and return raw results.

    Each result dict contains ``url``, ``content``, and ``title`` keys.

    Args:
        query:       The search query string.
        max_results: Maximum number of results to retrieve (1–10).

    Returns:
        List of result dicts.  Returns an empty list if Tavily is
        unavailable or the search fails.

    Example::

        >>> results = fetch_results("Latest AI research news")
        >>> results[0]
        {'url': 'https://...', 'content': '...', 'title': '...'}
    """
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults

        search_tool = TavilySearchResults(max_results=max_results)
        raw_results: Any = search_tool.invoke({"query": query})

        results: list[dict[str, str]] = []
        if isinstance(raw_results, list):
            for item in raw_results:
                if isinstance(item, dict):
                    results.append({
                        "url": item.get("url", ""),
                        "content": item.get("content", ""),
                        "title": item.get("title", ""),
                    })

        logger.info(
            "Tavily search for '%s' returned %d results.",
            query[:80],
            len(results),
        )
        return results

    except ImportError:
        logger.error(
            "tavily-python is not installed. "
            "Install it with: pip install tavily-python"
        )
        return []

    except Exception as e:
        logger.error("Tavily search failed: %s", e)
        return []


def summarize_results(
    query: str,
    results: list[dict[str, str]],
    llm: BaseChatModel,
) -> str:
    """
    Summarize web search results into a concise answer using an LLM.

    Args:
        query:   The original user query.
        results: List of result dicts from :func:`fetch_results`.
        llm:     An initialised LangChain chat model.

    Returns:
        A summarized text string.  If no results are available, returns
        a fallback message.
    """
    if not results:
        return "No web search results were found for this query."

    # Build a numbered context block from the results
    context_parts: list[str] = []
    for i, result in enumerate(results, 1):
        title = result.get("title", f"Source {i}")
        content = result.get("content", "")
        url = result.get("url", "")
        context_parts.append(
            f"[{i}] {title}\n"
            f"URL: {url}\n"
            f"Content: {content}\n"
        )
    context_block = "\n---\n".join(context_parts)

    prompt = (
        f"You are a research assistant. Summarize the following web search "
        f"results to answer the user's question.\n\n"
        f"Question: {query}\n\n"
        f"Web Search Results:\n{context_block}\n\n"
        f"Instructions:\n"
        f"- Provide a clear, comprehensive answer\n"
        f"- Cite sources using [1], [2], etc. where appropriate\n"
        f"- If the results are insufficient, say so honestly"
    )

    try:
        response = llm.invoke(prompt)
        summary = response.content
        logger.info("Summarized %d web results into %d chars.", len(results), len(summary))
        return summary

    except Exception as e:
        logger.error("LLM summarization failed: %s", e)
        # Fallback: concatenate raw content
        fallback = "\n\n".join(
            r.get("content", "") for r in results if r.get("content")
        )
        return fallback or "Web search results could not be summarized."


def search_web(
    query: str,
    llm: BaseChatModel,
    max_results: int = 5,
) -> dict:
    """
    High-level orchestrator: search the web and return summarized context.

    This is the main entry point used by the LangGraph pipeline.
    It calls :func:`fetch_results` then :func:`summarize_results` and
    returns a structured dict ready for injection into the graph state.

    Args:
        query:       The user's question.
        llm:         An initialised LangChain chat model.
        max_results: Number of Tavily results to fetch.

    Returns:
        Dict with keys:
            ``web_context``  — summarized answer text
            ``sources``      — list of source URLs
            ``raw_results``  — list of raw result dicts (for debugging)

    Example::

        >>> from pipelines.model_loader import load_model
        >>> llm = load_model("gpt")
        >>> result = search_web("Latest AI research news", llm)
        >>> result["sources"]
        ['https://...', 'https://...']
    """
    logger.info("Web search initiated for: '%s'", query[:80])

    # 1. Fetch raw results from Tavily
    results = fetch_results(query, max_results=max_results)

    # 2. Extract source URLs
    sources = [r["url"] for r in results if r.get("url")]

    # 3. Summarize with LLM
    web_context = summarize_results(query, results, llm)

    output = {
        "web_context": web_context,
        "sources": sources,
        "raw_results": results,
    }

    logger.info(
        "Web search complete — %d sources, context length=%d.",
        len(sources),
        len(web_context),
    )
    return output
