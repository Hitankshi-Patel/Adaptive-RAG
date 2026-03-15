"""
Multi-LLM Model Loader.

Dynamically instantiates the correct LangChain chat model based on
a string identifier sent by the frontend.

Supported providers:
    gpt    → OpenAI  (gpt-4o-mini)
    gemini → Google  (gemini-2.0-flash)
    claude → Anthropic (claude-3-5-haiku-latest)
"""

import logging

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# ── Supported model identifiers ──────────────────────────────────────────────
SUPPORTED_MODELS = {"gpt", "gemini", "claude"}


def load_model(model_name: str) -> BaseChatModel:
    """
    Return an initialised LangChain chat model for the given provider.

    The corresponding API key must be set as an environment variable:
        gpt    → OPENAI_API_KEY
        gemini → GOOGLE_API_KEY
        claude → ANTHROPIC_API_KEY

    Args:
        model_name: One of 'gpt', 'gemini', or 'claude'.

    Returns:
        A LangChain ``BaseChatModel`` instance.

    Raises:
        ValueError: If ``model_name`` is not recognised.
    """
    name = model_name.lower().strip()
    logger.info("Loading LLM model: %s", name)

    if name == "gpt":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)

    if name == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    if name == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-3-5-haiku-latest", temperature=0)

    raise ValueError(
        f"Unsupported model '{model_name}'. "
        f"Choose one of: {', '.join(sorted(SUPPORTED_MODELS))}"
    )
