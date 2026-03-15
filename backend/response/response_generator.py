"""
Response Generator Module.

Final stage in the Adaptive-RAG pipeline.  Receives the raw output
from whichever pipeline executed (retriever, web search, or direct LLM)
and produces a polished, structured response compatible with the
FastAPI ``QueryResponse`` schema.

Responsibilities:
    - Inject retrieved documents / web results into a final prompt
    - Generate a coherent, citation-aware answer via the LLM
    - Attach source references
    - Format the output for the API layer

Functions:
    generate_response  — create the final answer from pipeline context
    format_response    — normalise & validate the output dict
"""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


# ─── Prompt Templates ────────────────────────────────────────────────────────

RESPONSE_SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are the response synthesis engine of an Adaptive-RAG system.\n\n"
            "You will receive:\n"
            "  1. The user's original question\n"
            "  2. A preliminary answer from the pipeline\n"
            "  3. Supporting context (retrieved docs and/or web results)\n\n"
            "Your task:\n"
            "  - Refine the preliminary answer into a clear, well-structured response\n"
            "  - Incorporate relevant details from the supporting context\n"
            "  - Cite sources where applicable using [source_name] notation\n"
            "  - If the context is empty or insufficient, rely on the preliminary answer\n"
            "  - Keep the response informative yet concise"
        ),
    ),
    (
        "human",
        (
            "Question: {query}\n\n"
            "Preliminary Answer:\n{preliminary_answer}\n\n"
            "Supporting Context:\n{context}\n\n"
            "Generate the final polished response."
        ),
    ),
])


# ─── Public API ──────────────────────────────────────────────────────────────


def generate_response(
    query: str,
    preliminary_answer: str,
    documents: list[dict],
    sources: list[str],
    llm: BaseChatModel | None = None,
    strategy: str = "unknown",
) -> dict:
    """
    Generate the final response by synthesising the pipeline output.

    When an LLM is provided and there is supporting context (documents
    or web results), the preliminary answer is refined through a
    synthesis prompt.  Otherwise the preliminary answer is returned
    directly after formatting.

    Args:
        query:              The user's original question.
        preliminary_answer: Raw answer from the pipeline node.
        documents:          List of document/result dicts from the pipeline.
        sources:            List of source identifiers (filenames or URLs).
        llm:                Optional LLM for answer refinement.
        strategy:           Pipeline strategy label (``retriever`` /
                            ``web_search`` / ``general_llm``).

    Returns:
        Structured dict compatible with ``QueryResponse``:
            ``answer``, ``strategy``, ``sources``.

    Example::

        >>> result = generate_response(
        ...     query="What is Adaptive RAG?",
        ...     preliminary_answer="Adaptive RAG selects strategies...",
        ...     documents=[{"content": "...", "source": "paper.pdf"}],
        ...     sources=["paper.pdf"],
        ...     llm=load_model("gpt"),
        ...     strategy="retriever",
        ... )
        >>> result["answer"]
        'Adaptive RAG dynamically selects ...'
    """
    logger.info(
        "Generating final response — strategy=%s, docs=%d, sources=%d",
        strategy,
        len(documents),
        len(sources),
    )

    # Build context block from documents
    context = _build_context_block(documents)

    # If we have an LLM and meaningful context, refine the answer
    if llm and context.strip():
        try:
            chain = RESPONSE_SYNTHESIS_PROMPT | llm
            refined = chain.invoke({
                "query": query,
                "preliminary_answer": preliminary_answer,
                "context": context,
            })
            final_answer = refined.content
            logger.info("Answer refined via LLM — length=%d chars.", len(final_answer))
        except Exception as e:
            logger.error("LLM refinement failed: %s — using preliminary answer.", e)
            final_answer = preliminary_answer
    else:
        # No context or no LLM — use preliminary answer as-is
        final_answer = preliminary_answer

    return format_response(
        answer=final_answer,
        sources=sources,
        strategy=strategy,
    )


def format_response(
    answer: str,
    sources: list[str] | None = None,
    strategy: str = "unknown",
) -> dict:
    """
    Normalise and validate the final output dict.

    Ensures the response matches the ``QueryResponse`` Pydantic schema
    expected by the FastAPI layer.

    Args:
        answer:   The generated answer text.
        sources:  List of source identifiers.
        strategy: Pipeline strategy label.

    Returns:
        Dict with keys ``answer``, ``strategy``, ``sources``.

    Example::

        >>> format_response(
        ...     answer="Machine learning is ...",
        ...     sources=["llm_knowledge"],
        ...     strategy="general_llm",
        ... )
        {'answer': 'Machine learning is ...', 'strategy': 'general_llm', 'sources': ['llm_knowledge']}
    """
    # Deduplicate and clean sources
    cleaned_sources = _deduplicate(sources or [])

    # Ensure we always have at least one source
    if not cleaned_sources:
        cleaned_sources = ["llm_knowledge"]

    result = {
        "answer": answer.strip() if answer else "No answer could be generated.",
        "strategy": strategy,
        "sources": cleaned_sources,
    }

    logger.info(
        "Formatted response — strategy=%s, sources=%s, answer_len=%d",
        result["strategy"],
        result["sources"],
        len(result["answer"]),
    )
    return result


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _build_context_block(documents: list[dict]) -> str:
    """
    Build a numbered context string from a list of document dicts.

    Each dict should have at least a ``content`` key; ``source``,
    ``title``, and ``score`` are optional.
    """
    if not documents:
        return ""

    parts: list[str] = []
    for i, doc in enumerate(documents, 1):
        content = doc.get("content", "")
        source = doc.get("source", doc.get("title", f"Document {i}"))
        score = doc.get("score")

        header = f"[{i}] {source}"
        if score is not None:
            header += f"  (relevance: {score})"

        parts.append(f"{header}\n{content}")

    return "\n\n---\n\n".join(parts)


def _deduplicate(items: list[str]) -> list[str]:
    """Return a list with duplicates removed, preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        clean = item.strip()
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result
