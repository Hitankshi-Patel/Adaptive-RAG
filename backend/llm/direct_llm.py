"""
Direct LLM Processing Module.

Handles queries that can be answered entirely from the LLM's
parametric knowledge — no document retrieval or web search needed.

Supports dynamic model selection:
    gpt    → OpenAI GPT-4o-mini
    gemini → Google Gemini 2.0 Flash
    claude → Anthropic Claude 3.5 Haiku

Workflow:
    User Query
        → Build prompt from template
        → Send to selected LLM
        → Return structured response

Functions:
    generate_direct_response  — main entry point
"""

import logging

from langchain_core.prompts import ChatPromptTemplate

from pipelines.model_loader import load_model

logger = logging.getLogger(__name__)


# ─── Prompt Templates ────────────────────────────────────────────────────────

# Standard Q&A prompt — used for most direct LLM queries
QA_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert AI assistant. Answer the user's question "
            "clearly, accurately, and comprehensively.\n\n"
            "Guidelines:\n"
            "- Be concise but thorough\n"
            "- Use examples where helpful\n"
            "- Structure long answers with headings or bullet points\n"
            "- If you are unsure, say so honestly rather than guessing"
        ),
    ),
    ("human", "{query}"),
])

# Explanation prompt — used when the intent is "explanation"
EXPLANATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert educator and AI assistant. Explain the "
            "following topic in a clear, structured way.\n\n"
            "Guidelines:\n"
            "- Start with a concise definition\n"
            "- Break down complex ideas step by step\n"
            "- Use analogies or examples to aid understanding\n"
            "- End with a brief summary"
        ),
    ),
    ("human", "{query}"),
])

# Comparison prompt — used when the intent is "comparison"
COMPARISON_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert analyst. Compare and contrast the items "
            "mentioned in the user's question.\n\n"
            "Guidelines:\n"
            "- Identify the key dimensions of comparison\n"
            "- Present similarities and differences clearly\n"
            "- Use a structured format (e.g. table or bullet points)\n"
            "- Conclude with a brief recommendation if appropriate"
        ),
    ),
    ("human", "{query}"),
])

# Conversational prompt — used for casual / greeting queries
CONVERSATIONAL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a friendly AI assistant for the Adaptive-RAG system. "
            "Respond naturally and helpfully to conversational messages."
        ),
    ),
    ("human", "{query}"),
])

# Map intent labels to prompts (used when analysis metadata is available)
INTENT_PROMPT_MAP: dict[str, ChatPromptTemplate] = {
    "explanation": EXPLANATION_PROMPT,
    "comparison": COMPARISON_PROMPT,
    "conversational": CONVERSATIONAL_PROMPT,
    # Everything else falls through to the default QA_PROMPT
}


# ─── Public API ──────────────────────────────────────────────────────────────


def generate_direct_response(
    query: str,
    model_name: str = "gpt",
    intent: str | None = None,
) -> dict:
    """
    Generate an LLM response without any external retrieval.

    Dynamically loads the selected model, picks the best prompt
    template based on the detected intent, and returns a structured
    response dict.

    Args:
        query:      The user's question.
        model_name: LLM provider key (``'gpt'`` | ``'gemini'`` | ``'claude'``).
        intent:     Optional intent label from the Query Analysis module
                    (e.g. ``'explanation'``, ``'comparison'``).  When
                    provided, a specialised prompt template is selected.

    Returns:
        Dict with keys:
            ``response``  — the generated answer text
            ``model``     — the model key that was used
            ``sources``   — always ``['llm_knowledge']``

    Example::

        >>> result = generate_direct_response(
        ...     "What is machine learning?",
        ...     model_name="gpt",
        ...     intent="explanation",
        ... )
        >>> result["response"]
        'Machine learning is a subset of AI ...'
    """
    logger.info(
        "Direct LLM — query='%s', model='%s', intent='%s'",
        query[:80],
        model_name,
        intent,
    )

    # 1. Load the selected LLM
    try:
        llm = load_model(model_name)
    except ValueError as e:
        logger.error("Model loading failed: %s", e)
        return {
            "response": f"Error: {e}",
            "model": model_name,
            "sources": ["llm_knowledge"],
        }

    # 2. Select the best prompt template
    prompt = INTENT_PROMPT_MAP.get(intent, QA_PROMPT) if intent else QA_PROMPT
    logger.info("Using prompt template for intent: %s", intent or "default_qa")

    # 3. Build the chain and invoke
    try:
        chain = prompt | llm
        response = chain.invoke({"query": query})
        answer = response.content

        logger.info(
            "Direct LLM response generated — model=%s, length=%d chars.",
            model_name,
            len(answer),
        )

        return {
            "response": answer,
            "model": model_name,
            "sources": ["llm_knowledge"],
        }

    except Exception as e:
        logger.error("Direct LLM invocation failed: %s", e, exc_info=True)
        return {
            "response": f"An error occurred while generating the response: {e}",
            "model": model_name,
            "sources": ["llm_knowledge"],
        }
